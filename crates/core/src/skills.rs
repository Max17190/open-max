//! Skills: progressive disclosure. Only a skill's name and one-line
//! description are ever resident in the system prompt (~15 tokens each); the
//! model reads the full SKILL.md body on demand with read_file. This is the
//! structural answer to adding capability without taxing every prefill.
//!
//! Discovery: `~/.openmax/skills/<name>/SKILL.md` (global), then the
//! project's `.agents/skills/<name>/SKILL.md` — the emerging cross-harness
//! convention — with the project winning on name collision. A SKILL.md
//! carries `---`-delimited frontmatter with `name:` and `description:`;
//! only those two keys are read (bare, double-quoted, or a `>`/`|` block
//! scalar for the description), so no YAML dependency is needed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Skills beyond this count stop being "a small index" and start being a
/// prompt tax; the sorted head wins and the trailer says what was dropped.
pub const MAX_SKILLS: usize = 50;
pub const MAX_SKILL_DESC_CHARS: usize = 200;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillSpec {
    pub name: String,
    pub description: String,
    /// SKILL.md location, so the model can read_file the full body.
    pub path: PathBuf,
}

/// Discover skills for a project: global first, project overwrites on name
/// collision. Malformed skill files are skipped, never fatal.
pub fn discover(data_dir: &Path, project_root: &Path) -> Vec<SkillSpec> {
    discover_in(&skill_dirs(data_dir, project_root))
}

/// Global then project skill dirs; later dirs win on name collision.
pub(crate) fn skill_dirs(data_dir: &Path, project_root: &Path) -> [PathBuf; 2] {
    [
        data_dir.join("skills"),
        project_root.join(".agents").join("skills"),
    ]
}

pub(crate) fn discover_in(dirs: &[PathBuf]) -> Vec<SkillSpec> {
    let mut by_name: HashMap<String, SkillSpec> = HashMap::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut skill_files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path().join("SKILL.md"))
            .filter(|p| p.is_file())
            .collect();
        skill_files.sort();
        for path in skill_files {
            if let Ok(spec) = parse_skill_md(&path) {
                by_name.insert(spec.name.clone(), spec);
            }
        }
    }
    let mut skills: Vec<SkillSpec> = by_name.into_values().collect();
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills.truncate(MAX_SKILLS);
    skills
}

/// Pull `name:` and `description:` out of the frontmatter block. Values may
/// be bare or double-quoted; anything more exotic belongs in the body.
/// Errors are ignored by discovery and surfaced verbatim by `openmax --check`.
pub(crate) fn parse_skill_md(path: &Path) -> Result<SkillSpec, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("unreadable: {e}"))?;
    parse_skill_source(path, &text)
}

pub(crate) fn parse_skill_source(path: &Path, text: &str) -> Result<SkillSpec, String> {
    let body = text
        .strip_prefix("---")
        .ok_or("missing `---` frontmatter block at the top")?;
    let end = body
        .find("\n---")
        .ok_or("frontmatter never closes with `---`")?;
    let mut name = None;
    for line in body[..end].lines() {
        // Only a key at the block's top level (no leading indentation) is the
        // skill's key. An indented `name:` is someone else's data: a field of
        // a nested map like the standard `metadata:` block, or a line inside
        // a description block scalar (whose content is always indented past
        // its key). Matching those let a nested key silently steal the
        // skill's identity while --check certified the file ok.
        if line.trim_start().len() != line.len() {
            continue;
        }
        if let Some(v) = line.strip_prefix("name:") {
            // The name is an identity rendered into the skills index (one line
            // each), the collision clause of a refreeze receipt (a user-role
            // message), and `/context`. A control character in it would forge
            // a second line, clause, or row, so it is flattened to one line
            // here; length and every printable character survive, so the
            // Agent Skills portability warning still sees the real name.
            name = Some(crate::text::one_line(v.trim().trim_matches('"')));
        }
    }
    let name = name
        .filter(|n| !n.is_empty())
        .ok_or("frontmatter has no non-empty `name:`")?;
    let mut description = raw_description(text).unwrap_or_default();
    if description.chars().count() > MAX_SKILL_DESC_CHARS {
        description = description.chars().take(MAX_SKILL_DESC_CHARS).collect::<String>() + "…";
    }
    Ok(SkillSpec { name, description, path: path.to_path_buf() })
}

/// The `description:` exactly as the frontmatter wrote it, before the index
/// cap clamps it. `openmax --check` reads it from the same bytes the parse
/// used, so a report can say the written line is longer than the indexed one
/// without a second read of the file.
pub(crate) fn raw_description(text: &str) -> Option<String> {
    let body = text.strip_prefix("---")?;
    let end = body.find("\n---")?;
    // The last `description:` key wins, as it always has for SKILL.md.
    frontmatter_descriptions(&body[..end]).pop()
}

/// Every top-level `description:` value of one frontmatter block, in order,
/// each folded to a single line. Three spellings are read: a bare value, a
/// double-quoted value, and a YAML block scalar (`>` folded or `|` literal,
/// with an optional `-`/`+` chomping indicator and an optional explicit
/// indentation digit), whose value is the indented lines that follow.
/// Multi-line values fold to one line because the index line is one line;
/// the frontmatter's `>` says the author meant that too. Anything more
/// exotic (single quotes, flow mappings, anchors) reads as its literal
/// spelling. An INDENTED `description:` is someone else's data (a nested
/// map's field, like the standard `metadata:` block) and never one of these
/// values. Callers pick first or last; each surface keeps the choice it
/// always made.
pub(crate) fn frontmatter_descriptions(block: &str) -> Vec<String> {
    let lines: Vec<&str> = block.lines().collect();
    let mut descriptions = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let indent = line.len() - line.trim_start().len();
        i += 1;
        if indent > 0 {
            continue;
        }
        let Some(value) = line.trim().strip_prefix("description:") else {
            continue;
        };
        let value = value.trim();
        if let Some(explicit) = block_scalar_indent(value) {
            // The value is the run of lines at the block's indentation or
            // deeper: the explicit digit when the header gives one, else the
            // depth of the first non-blank line, which YAML infers the same
            // way. Either way that depth must exceed the key's, and the first
            // non-blank line shallower than it ends the block, so under- or
            // unevenly indented text never populates the index. Blank lines
            // inside the run are allowed.
            let mut min_indent = explicit.map(|n| indent + n);
            let mut parts: Vec<&str> = Vec::new();
            while i < lines.len() {
                let next = lines[i];
                if next.trim().is_empty() {
                    i += 1;
                    continue;
                }
                let next_indent = next.len() - next.trim_start().len();
                let floor = *min_indent.get_or_insert(next_indent);
                if next_indent < floor || next_indent <= indent {
                    break;
                }
                parts.push(next.trim());
                i += 1;
            }
            descriptions.push(crate::text::one_line(&parts.join(" ")));
        } else {
            descriptions.push(crate::text::one_line(value.trim_matches('"')));
        }
    }
    descriptions
}

/// Whether a value is a YAML block scalar header (`>` or `|`, optionally
/// followed by a chomping indicator (`-`, `+`) and/or an explicit indentation
/// digit, in either order): the spellings that reach frontmatter in
/// practice. `Some(None)` for a header without a digit, `Some(Some(n))` for
/// one with, `None` for anything else.
fn block_scalar_indent(value: &str) -> Option<Option<usize>> {
    let mut chars = value.chars();
    if !matches!(chars.next(), Some('>') | Some('|')) {
        return None;
    }
    let rest: Vec<char> = chars.collect();
    if rest.len() > 2
        || !rest.iter().all(|c| matches!(c, '-' | '+') || c.is_ascii_digit())
        || rest.iter().filter(|c| matches!(c, '-' | '+')).count() > 1
        || rest.iter().filter(|c| c.is_ascii_digit()).count() > 1
    {
        return None;
    }
    let digit = rest.iter().find_map(|c| c.to_digit(10)).map(|d| d as usize);
    // `>0` is not a header (YAML requires 1-9); read it as a plain value.
    match digit {
        Some(0) => None,
        other => Some(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-skill-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_skill(root: &Path, dir_name: &str, frontmatter: &str) {
        let dir = root.join(dir_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), format!("---\n{frontmatter}\n---\nFull body here.\n")).unwrap();
    }

    /// A skill name is rendered into the skills index (one line per skill),
    /// the collision clause of a refreeze receipt (a user-role message), and
    /// `/context`. A carriage return or escape in it would forge a second
    /// line, clause, or row, so the parse flattens the name to one line while
    /// keeping every printable character (the portability warning still reads
    /// the real name).
    #[test]
    fn a_skill_name_is_flattened_to_one_line() {
        let spec = parse_skill_source(
            Path::new(".agents/skills/x/SKILL.md"),
            "---\nname: real\rname: forged\ndescription: d\n---\nBody.\n",
        )
        .unwrap();
        assert!(!spec.name.contains('\r'), "name kept a carriage return: {:?}", spec.name);
        assert!(!spec.name.chars().any(|c| c.is_control()), "{:?}", spec.name);
        assert_eq!(spec.name, "real name: forged");
    }

    /// A `>`/`|` block-scalar description folds to one line like the bare form,
    /// carriage returns and all: the bare branch already replaced them, and a
    /// folded description carrying one reached the index unchanged.
    #[test]
    fn a_folded_description_carries_no_line_break() {
        let spec = parse_skill_source(
            Path::new(".agents/skills/x/SKILL.md"),
            "---\nname: x\ndescription: >\n  first\rsecond\n  third\n---\nBody.\n",
        )
        .unwrap();
        assert!(
            !spec.description.chars().any(|c| c.is_control()),
            "folded description kept a control char: {:?}",
            spec.description
        );
    }

    /// A nested key is someone else's data. The standard skill frontmatter
    /// commonly carries a `metadata:` block whose fields may be spelled
    /// `name:`/`description:`; an indentation-blind, last-wins match let that
    /// block silently steal the skill's identity, the frozen index carried
    /// the stolen strings, and --check certified the file ok (
    /// reproduced on the binary). Keys are read at top level only.
    #[test]
    fn a_nested_frontmatter_key_does_not_override_the_skill_name() {
        let root = temp_dir("nested");
        write_skill(
            &root,
            "pdf-tools",
            "name: pdf-tools\ndescription: Extract text and tables from PDF files.\nmetadata:\n  name: internal-scratch\n  description: not for the index",
        );
        let skills = discover_in(std::slice::from_ref(&root));
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "pdf-tools", "the top-level name is the skill's name");
        assert_eq!(
            skills[0].description, "Extract text and tables from PDF files.",
            "the top-level description is the indexed one"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A block scalar's content is always indented past its key, so a line of
    /// it that happens to start with `name:` is prose, not the skill's key.
    #[test]
    fn a_block_scalar_line_spelling_name_does_not_become_the_name() {
        let root = temp_dir("scalar-name");
        write_skill(
            &root,
            "config-audit",
            "name: config-audit\ndescription: >\n  Audits config files.\n  name: fields inside YAML are checked too.",
        );
        let skills = discover_in(std::slice::from_ref(&root));
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "config-audit");
        assert_eq!(
            skills[0].description,
            "Audits config files. name: fields inside YAML are checked too."
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn parses_minimal_frontmatter() {
        let root = temp_dir("min");
        write_skill(&root, "review", "name: code-review\ndescription: reviews a diff");
        let skills = discover_in(std::slice::from_ref(&root));
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "code-review");
        assert_eq!(skills[0].description, "reviews a diff");
        assert!(skills[0].path.ends_with("review/SKILL.md"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn project_skill_wins_and_malformed_is_skipped() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        write_skill(&global, "deploy", "name: deploy\ndescription: global variant");
        write_skill(&project, "deploy", "name: deploy\ndescription: project variant");
        // No frontmatter at all: skipped without failing discovery.
        let broken = project.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("SKILL.md"), "just prose, no frontmatter").unwrap();
        let skills = discover_in(&[global.clone(), project.clone()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "project variant");
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn skills_sorted_capped_and_descriptions_clamped() {
        let root = temp_dir("cap");
        for i in 0..(MAX_SKILLS + 5) {
            write_skill(&root, &format!("s{i:03}"), &format!("name: skill-{i:03}\ndescription: {}", "x".repeat(400)));
        }
        let skills = discover_in(std::slice::from_ref(&root));
        assert_eq!(skills.len(), MAX_SKILLS);
        let names: Vec<_> = skills.iter().map(|s| s.name.clone()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "must be sorted for deterministic prompts");
        assert!(skills[0].description.chars().count() <= MAX_SKILL_DESC_CHARS + 1);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A description written as a YAML block scalar (the multi-line spelling
    /// third-party skill packages ship) folds to one index line. Reading the
    /// header alone indexed such a skill as `>`, a line that says nothing.
    #[test]
    fn block_scalar_descriptions_fold_to_one_line() {
        let folded = "name: stack\ndescription: >\n  Manages stacked branches.\n  Use for stack creation, sync, or merge;\n\n  whenever a stack is checked out.\nmetadata:\n  author: someone\n  version: \"0.1.0\"";
        let spec = parse_skill_source(Path::new("SKILL.md"), &format!("---\n{folded}\n---\nbody\n"))
            .unwrap();
        assert_eq!(
            spec.description,
            "Manages stacked branches. Use for stack creation, sync, or merge; whenever a stack is checked out."
        );

        // Literal and chomping spellings read the same way; the next key at
        // the parent's indentation ends the value.
        for header in [">-", "|", "|+", ">2", "|-2", ">2+"] {
            let text = format!("---\nname: n\ndescription: {header}\n    first line\n    second line\ntags: x\n---\nbody\n");
            let spec = parse_skill_source(Path::new("SKILL.md"), &text).unwrap();
            assert_eq!(spec.description, "first line second line", "header {header}");
        }

        // An explicit indentation digit is enforced: content shallower than
        // it ends the block, so under-indented text never reaches the index
        // (review finding). Deeper content still reads.
        let shallow = parse_skill_source(Path::new("SKILL.md"), "---\nname: n\ndescription: >4\n  only two spaces\n---\nbody\n").unwrap();
        assert_eq!(shallow.description, "");
        let deep = parse_skill_source(Path::new("SKILL.md"), "---\nname: n\ndescription: >4\n      six spaces\n---\nbody\n").unwrap();
        assert_eq!(deep.description, "six spaces");
        let zero = parse_skill_source(Path::new("SKILL.md"), "---\nname: n\ndescription: >0\n  text\n---\nbody\n").unwrap();
        assert_eq!(zero.description, ">0", "a zero digit is not a YAML header");

        // Without a digit the first content line sets the depth, as YAML
        // infers it; a later shallower line ends the block instead of being
        // folded in (review finding), even when it is deeper than the key.
        let uneven = parse_skill_source(Path::new("SKILL.md"), "---\nname: n\ndescription: >\n    four deep\n  two deep\n---\nbody\n").unwrap();
        assert_eq!(uneven.description, "four deep");
        let deeper_later = parse_skill_source(Path::new("SKILL.md"), "---\nname: n\ndescription: >\n  two deep\n    four deep\n---\nbody\n").unwrap();
        assert_eq!(deeper_later.description, "two deep four deep", "deeper continuation lines fold in");

        // Duplicate keys: SKILL.md keeps its last-key reading, prompt
        // templates keep their first-key reading (each as before).
        let dupes = "description: first\ndescription: second";
        assert_eq!(frontmatter_descriptions(dupes), vec!["first", "second"]);
        let last = parse_skill_source(Path::new("SKILL.md"), &format!("---\nname: n\n{dupes}\n---\nbody\n")).unwrap();
        assert_eq!(last.description, "second");

        // An indicator with nothing under it is an empty description, which
        // --check names; it is never the indicator character itself.
        let empty = parse_skill_source(Path::new("SKILL.md"), "---\nname: n\ndescription: >\ntags: x\n---\nbody\n").unwrap();
        assert_eq!(empty.description, "");

        // A bare `>` inside a quoted or bare value is not a block header.
        let bare = parse_skill_source(Path::new("SKILL.md"), "---\nname: n\ndescription: use > for redirects\n---\nbody\n").unwrap();
        assert_eq!(bare.description, "use > for redirects");
    }
}
