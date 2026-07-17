//! Prompt templates: reusable markdown prompts the user invokes as `/name
//! args` in the composer. The file body becomes the user message after
//! argument substitution ($ARGUMENTS, $1..$9), so a template is pure message
//! content: it never touches the frozen system prompt or tool schemas, and it
//! is re-read at every invocation. Zero prompt tax when none exist.
//!
//! Discovery: `~/.openmax/prompts/<name>.md` (global), then the project's
//! `.agents/prompts/<name>.md`, project winning on name collision. The file
//! stem is the command name; an optional `---` frontmatter block may carry a
//! one-line `description:` for the completion popup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Same ceiling as skills: past this the popup is noise, not an index.
pub const MAX_TEMPLATES: usize = 50;
pub const MAX_TEMPLATE_DESC_CHARS: usize = 200;

#[derive(Clone, Debug)]
pub struct TemplateSpec {
    /// The slash-command name (the file stem).
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

/// Discover templates for a project: global first, project overwrites on
/// name collision. Malformed or oddly named files are skipped, never fatal.
pub fn discover(project_root: &Path) -> Vec<TemplateSpec> {
    discover_in(&[
        crate::state::default_data_dir().join("prompts"),
        project_root.join(".agents").join("prompts"),
    ])
}

pub(crate) fn discover_in(dirs: &[PathBuf]) -> Vec<TemplateSpec> {
    let mut by_name: HashMap<String, TemplateSpec> = HashMap::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "md") && p.is_file())
            .collect();
        paths.sort();
        for path in paths {
            if let Some(spec) = parse_template(&path) {
                by_name.insert(spec.name.clone(), spec);
            }
        }
    }
    let mut templates: Vec<TemplateSpec> = by_name.into_values().collect();
    templates.sort_by(|a, b| a.name.cmp(&b.name));
    templates.truncate(MAX_TEMPLATES);
    templates
}

/// Expand a composer invocation (`name args...`, no leading slash) against
/// the discovered templates. Returns the substituted user message, or None
/// when no template matches the head token.
pub fn expand_invocation(project_root: &Path, input: &str) -> Option<String> {
    let input = input.trim_start();
    let (head, args) = match input.find(char::is_whitespace) {
        Some(i) => (&input[..i], input[i..].trim()),
        None => (input, ""),
    };
    if head.is_empty() {
        return None;
    }
    let spec = discover(project_root).into_iter().find(|t| t.name == head)?;
    // Re-read at invocation time: templates are message content, not frozen
    // session state, so an edit applies to the very next use.
    let text = std::fs::read_to_string(&spec.path).ok()?;
    Some(substitute(body_of(&text), args))
}

/// Command-name discipline mirrors external tools: boring names only.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn parse_template(path: &Path) -> Option<TemplateSpec> {
    let name = path.file_stem()?.to_str()?.to_string();
    if !valid_name(&name) {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    if body_of(&text).trim().is_empty() {
        return None;
    }
    let mut description = frontmatter_description(&text).unwrap_or_default();
    if description.chars().count() > MAX_TEMPLATE_DESC_CHARS {
        description =
            description.chars().take(MAX_TEMPLATE_DESC_CHARS).collect::<String>() + "…";
    }
    Some(TemplateSpec { name, description, path: path.to_path_buf() })
}

/// The template body: everything after an optional `---` frontmatter block.
fn body_of(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("---") else { return text };
    let Some(end) = rest.find("\n---") else { return text };
    let after = &rest[end + 4..];
    after.strip_prefix('\n').unwrap_or(after)
}

fn frontmatter_description(text: &str) -> Option<String> {
    let rest = text.strip_prefix("---")?;
    let end = rest.find("\n---")?;
    for line in rest[..end].lines() {
        if let Some(v) = line.trim().strip_prefix("description:") {
            return Some(v.trim().trim_matches('"').replace(['\n', '\r'], " "));
        }
    }
    None
}

/// Substitute `$ARGUMENTS` (the raw argument string) and `$1`..`$9`
/// (whitespace-split positionals; missing ones become empty). A template with
/// no placeholders gets the arguments appended after a blank line, so plain
/// prompt files still accept input.
fn substitute(body: &str, args: &str) -> String {
    let positional: Vec<&str> = args.split_whitespace().collect();
    let mut out = String::with_capacity(body.len() + args.len());
    let mut used_placeholder = false;
    let mut chars = body.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c != '$' {
            out.push(c);
            continue;
        }
        let rest = &body[i + 1..];
        if rest.starts_with("ARGUMENTS") {
            out.push_str(args);
            used_placeholder = true;
            for _ in 0.."ARGUMENTS".len() {
                chars.next();
            }
            continue;
        }
        let mut digit = None;
        if let Some((_, d)) = chars.peek().copied() {
            if ('1'..='9').contains(&d) {
                // `$12` stays literal: only single-digit positionals exist.
                let after = rest[1..].chars().next();
                if !after.is_some_and(|a| a.is_ascii_digit()) {
                    digit = Some(d as usize - '0' as usize);
                }
            }
        }
        match digit {
            Some(n) => {
                out.push_str(positional.get(n - 1).copied().unwrap_or(""));
                used_placeholder = true;
                chars.next();
            }
            None => out.push('$'),
        }
    }
    if !used_placeholder && !args.is_empty() {
        out = format!("{}\n\n{args}", out.trim_end());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-tmpl-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_template(root: &Path, name: &str, content: &str) {
        std::fs::write(root.join(format!("{name}.md")), content).unwrap();
    }

    #[test]
    fn discovers_and_reads_frontmatter_description() {
        let root = temp_dir("disc");
        write_template(&root, "fix-issue", "---\ndescription: fix a GitHub issue\n---\nFix issue $1.\n");
        write_template(&root, "plain", "Just review the diff.\n");
        write_template(&root, "bad name!", "never valid\n");
        let templates = discover_in(std::slice::from_ref(&root));
        assert_eq!(templates.len(), 2);
        assert_eq!(templates[0].name, "fix-issue");
        assert_eq!(templates[0].description, "fix a GitHub issue");
        assert_eq!(templates[1].name, "plain");
        assert_eq!(templates[1].description, "");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn project_template_wins_over_global() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        write_template(&global, "deploy", "global body\n");
        write_template(&project, "deploy", "project body\n");
        let templates = discover_in(&[global.clone(), project.clone()]);
        assert_eq!(templates.len(), 1);
        assert!(templates[0].path.starts_with(&project));
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn substitutes_arguments_and_positionals() {
        assert_eq!(
            substitute("Fix issue $1 with priority $2.", "42 high"),
            "Fix issue 42 with priority high."
        );
        assert_eq!(substitute("Run: $ARGUMENTS", "cargo test --all"), "Run: cargo test --all");
        // Missing positionals become empty; $12 and bare $ stay literal.
        assert_eq!(substitute("a $3 b", "x"), "a  b");
        assert_eq!(substitute("cost $12 and $ttl for $1", "x"), "cost $12 and $ttl for x");
    }

    #[test]
    fn appends_args_when_no_placeholder() {
        assert_eq!(substitute("Review this diff.\n", "focus on unsafe"), "Review this diff.\n\nfocus on unsafe");
        assert_eq!(substitute("Review this diff.\n", ""), "Review this diff.\n");
    }

    #[test]
    fn expand_invocation_matches_head_and_strips_frontmatter() {
        let root = temp_dir("exp");
        let prompts = root.join(".agents").join("prompts");
        std::fs::create_dir_all(&prompts).unwrap();
        // Unique name so a real template in ~/.openmax/prompts cannot collide.
        write_template(&prompts, "omx-test-issue", "---\ndescription: d\n---\nFix issue $1 now.\n");
        let expanded = expand_invocation(&root, "omx-test-issue 42").unwrap();
        assert_eq!(expanded, "Fix issue 42 now.\n");
        assert!(expand_invocation(&root, "omx-test-nosuch 42").is_none());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn empty_body_is_skipped() {
        let root = temp_dir("empty");
        write_template(&root, "hollow", "---\ndescription: nothing\n---\n\n");
        assert!(discover_in(std::slice::from_ref(&root)).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }
}
