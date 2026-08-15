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
pub fn discover(data_dir: &Path, project_root: &Path) -> Vec<TemplateSpec> {
    discover_in(&template_dirs(data_dir, project_root))
}

/// Global then project template dirs; later dirs win on name collision.
pub(crate) fn template_dirs(data_dir: &Path, project_root: &Path) -> [PathBuf; 2] {
    [
        data_dir.join("prompts"),
        project_root.join(".agents").join("prompts"),
    ]
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
            if let Ok(spec) = parse_template(&path) {
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
pub fn expand_invocation(data_dir: &Path, project_root: &Path, input: &str) -> Option<String> {
    let input = input.trim_start();
    let (head, args) = match input.find(char::is_whitespace) {
        Some(i) => (&input[..i], input[i..].trim()),
        None => (input, ""),
    };
    if head.is_empty() {
        return None;
    }
    let spec = resolve(data_dir, project_root, head)?;
    // Re-read at invocation time: templates are message content, not frozen
    // session state, so an edit applies to the very next use.
    let text = std::fs::read_to_string(&spec.path).ok()?;
    Some(substitute(body_of(&text), args))
}

/// Expand a whole submitted line (`/name args...`). Some only when the line
/// is a slash line naming a template; the front end decides what a None
/// means (the TUI treats it as a slash command, the others as literal text).
pub fn expand_slash_line(data_dir: &Path, project_root: &Path, text: &str) -> Option<String> {
    expand_invocation(data_dir, project_root, text.strip_prefix('/')?)
}

/// Expand a leading `/name args` line into its template body, or return the
/// input unchanged when it is not a slash line or names no template. The
/// expansion is single-pass: a body that itself starts with `/` is message
/// content, not another invocation. Every front end that has no slash
/// commands of its own (`--print`, `--stdio`) submits through this.
pub fn expand_user_input(data_dir: &Path, project_root: &Path, text: &str) -> String {
    expand_slash_line(data_dir, project_root, text).unwrap_or_else(|| text.to_string())
}

/// Resolve one template by name with a direct path probe, project first.
/// Never goes through the capped discovery list: MAX_TEMPLATES bounds the
/// popup index, not which templates can be invoked.
fn resolve(data_dir: &Path, project_root: &Path, name: &str) -> Option<TemplateSpec> {
    if !valid_name(name) {
        return None;
    }
    let dirs = [
        project_root.join(".agents").join("prompts"),
        data_dir.join("prompts"),
    ];
    dirs.iter()
        .map(|dir| dir.join(format!("{name}.md")))
        .filter(|path| path.is_file())
        .find_map(|path| parse_template(&path).ok())
}

/// Command-name discipline mirrors external tools: boring names only.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Errors are ignored by discovery and surfaced verbatim by `openmax --check`.
pub(crate) fn parse_template(path: &Path) -> Result<TemplateSpec, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("unreadable: {e}"))?;
    parse_template_source(path, &text)
}

/// The same parse from bytes the caller already read, so a diagnostic
/// computed beside it (the raw description a cap will clamp) describes the
/// same generation of the file as the parse itself.
pub(crate) fn parse_template_source(path: &Path, text: &str) -> Result<TemplateSpec, String> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .ok_or("file stem is not valid UTF-8")?;
    if !valid_name(&name) {
        return Err(format!(
            "invalid template name '{name}': 1-64 chars of [a-zA-Z0-9_-] required (the stem becomes /{name})"
        ));
    }
    // A block that opens with `---` and never closes is refused by name, the
    // way SKILL.md refuses it: the fence and every key under it would
    // otherwise be expanded into the user's message as body text, which is
    // exactly what the author meant them not to be.
    if text.starts_with("---") && frontmatter_end(text).is_none() {
        return Err("frontmatter never closes with `---`".into());
    }
    if body_of(text).trim().is_empty() {
        return Err("template body is empty".into());
    }
    let mut description = frontmatter_description(text).unwrap_or_default();
    if description.chars().count() > MAX_TEMPLATE_DESC_CHARS {
        description =
            description.chars().take(MAX_TEMPLATE_DESC_CHARS).collect::<String>() + "…";
    }
    Ok(TemplateSpec { name, description, path: path.to_path_buf() })
}

/// Offset of the closing `---` within the text that follows the opening one,
/// or None when there is no frontmatter block to speak of.
fn frontmatter_end(text: &str) -> Option<usize> {
    text.strip_prefix("---")?.find("\n---")
}

/// The template body: everything after an optional `---` frontmatter block.
fn body_of(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("---") else { return text };
    let Some(end) = frontmatter_end(text) else { return text };
    let after = &rest[end + 4..];
    after.strip_prefix('\n').unwrap_or(after)
}

/// The `description:` exactly as the frontmatter wrote it, before the popup
/// cap clamps it. `openmax --check` reads it from the same bytes the parse
/// used, so a report can say the written line is longer than the shown one
/// without a second read of the file (mirrors `skills::raw_description`).
pub(crate) fn raw_description(text: &str) -> Option<String> {
    frontmatter_description(text)
}

fn frontmatter_description(text: &str) -> Option<String> {
    let rest = text.strip_prefix("---")?;
    let end = frontmatter_end(text)?;
    for line in rest[..end].lines() {
        if let Some(v) = line.trim().strip_prefix("description:") {
            return Some(v.trim().trim_matches('"').replace(['\n', '\r'], " "));
        }
    }
    None
}

/// Substitute `$ARGUMENTS` (the raw argument string) and `$1`..`$9`
/// (whitespace-split positionals; missing ones become empty). `$$` escapes a
/// literal dollar (so `$$5` survives as `$5`). A template with no
/// placeholders gets the arguments appended after a blank line, so plain
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
        if rest.starts_with('$') {
            out.push('$');
            chars.next();
            continue;
        }
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
        let data = root.join("data");
        let prompts = root.join(".agents").join("prompts");
        std::fs::create_dir_all(&prompts).unwrap();
        write_template(&prompts, "omx-test-issue", "---\ndescription: d\n---\nFix issue $1 now.\n");
        let expanded = expand_invocation(&data, &root, "omx-test-issue 42").unwrap();
        assert_eq!(expanded, "Fix issue 42 now.\n");
        assert!(expand_invocation(&data, &root, "omx-test-nosuch 42").is_none());
        let _ = std::fs::remove_dir_all(root);
    }

    /// The front-end-agnostic entry point: same expansion in the composer, in
    /// `--print`, and in a stdio `user` command.
    #[test]
    fn expand_user_input_handles_slash_lines_only_and_expands_once() {
        let root = temp_dir("user-input");
        let data = root.join("data");
        let prompts = root.join(".agents").join("prompts");
        std::fs::create_dir_all(&prompts).unwrap();
        write_template(&prompts, "omx-test-greet", "MARKER: greet $ARGUMENTS\n");
        // A body that looks like another invocation stays message content.
        write_template(&prompts, "omx-test-loop", "/omx-test-greet again\n");

        assert_eq!(
            expand_user_input(&data, &root, "/omx-test-greet world"),
            "MARKER: greet world\n"
        );
        assert_eq!(expand_user_input(&data, &root, "/omx-test-loop"), "/omx-test-greet again\n");
        // Not a template, not a slash line: unchanged either way.
        assert_eq!(expand_user_input(&data, &root, "/omx-test-nosuch x"), "/omx-test-nosuch x");
        assert_eq!(expand_user_input(&data, &root, "plain prompt"), "plain prompt");
        assert_eq!(expand_user_input(&data, &root, "path /omx-test-greet"), "path /omx-test-greet");
        assert!(expand_slash_line(&data, &root, "plain prompt").is_none());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn double_dollar_escapes_a_literal() {
        assert_eq!(substitute("The fee is $$5 for $1.", "alice"), "The fee is $5 for alice.");
        assert_eq!(substitute("literal $$ARGUMENTS", "x"), "literal $ARGUMENTS\n\nx");
    }

    /// MAX_TEMPLATES bounds the popup index only; a template sorted past the
    /// cap must still be invocable by name.
    #[test]
    fn invocation_is_independent_of_the_discovery_cap() {
        let root = temp_dir("cap-inv");
        let data = root.join("data");
        let prompts = root.join(".agents").join("prompts");
        std::fs::create_dir_all(&prompts).unwrap();
        for i in 0..MAX_TEMPLATES {
            write_template(&prompts, &format!("aaa-{i:03}"), "filler body\n");
        }
        write_template(&prompts, "zzz-omx-tail", "Tail says $1.\n");
        let discovered = discover_in(std::slice::from_ref(&prompts));
        assert_eq!(discovered.len(), MAX_TEMPLATES);
        assert!(!discovered.iter().any(|t| t.name == "zzz-omx-tail"), "sorted past the cap");
        assert_eq!(expand_invocation(&data, &root, "zzz-omx-tail hello").unwrap(), "Tail says hello.\n");
        let _ = std::fs::remove_dir_all(root);
    }

    /// An opening `---` with no closing one is the same authoring mistake
    /// SKILL.md names, and it is worse here: the fence and the `description:`
    /// line would be sent to the model as part of the user's message. Refuse
    /// the file instead, so discovery drops it and `--check` can say why.
    #[test]
    fn unclosed_frontmatter_is_rejected() {
        let root = temp_dir("unclosed");
        write_template(&root, "half-open", "---\ndescription: d\nDo the thing.\n");
        write_template(&root, "closed", "---\ndescription: d\n---\nDo the thing.\n");
        match parse_template(&root.join("half-open.md")) {
            Err(reason) => assert_eq!(reason, "frontmatter never closes with `---`"),
            Ok(spec) => panic!("frontmatter that never closes must not parse: {spec:?}"),
        }
        let discovered = discover_in(std::slice::from_ref(&root));
        assert_eq!(discovered.len(), 1, "only the closed one is a template");
        assert_eq!(discovered[0].name, "closed");
        // A body that merely contains `---` later is not frontmatter at all.
        write_template(&root, "ruled", "Section one.\n\n---\n\nSection two.\n");
        assert_eq!(parse_template(&root.join("ruled.md")).unwrap().name, "ruled");
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
