use std::path::Path;

/// AGENTS.md content is user-authored instructions; cap it so a sprawling
/// file cannot crowd a 16k window.
const MAX_AGENTS_MD_BYTES: usize = 2_000;
/// The layout map exists to save discovery round trips, not to describe the
/// tree exhaustively; ~300 tokens is the budget.
const MAX_MAP_BYTES: usize = 1_200;
const MAX_MAP_DEPTH: usize = 2;

/// System prompt deliberately tuned for small local models: short, imperative,
/// with explicit tool-use rules. Long "constitution"-style prompts measurably
/// degrade 7B–30B models, so every line here has to earn its place.
///
/// Grounding context (AGENTS.md, a shallow layout map) is appended here, once,
/// at session creation: the prompt is persisted with the session, so the token
/// prefix stays byte-stable across every turn — which is what keeps the
/// server-side prompt cache warm. Without the map, a session typically opens
/// with two or three list_dir/glob calls just to learn the layout, and each
/// of those is a full prefill+decode round trip.
pub fn system_prompt(project_root: &Path) -> String {
    let root = project_root.to_string_lossy();
    let mut prompt = format!(
        "You are Open Max, a coding agent. You work on the project at {root} using tools.\n\
        \n\
        Rules:\n\
        - Inspect before you act: use list_dir, glob, grep and read_file to ground every answer in the real code. Never invent file contents or paths.\n\
        - Before editing a file, read_file it first. Then use edit_file with an old_string copied exactly from the file.\n\
        - Prefer edit_file for changes to existing files; use write_file only for new files or full rewrites.\n\
        - Use bash to run builds, tests and git. Commands run in the project root.\n\
        - Make small, focused changes that match the existing code style.\n\
        - After making changes, verify them when possible (compile, run tests, or re-read the file).\n\
        - When the task is done, stop calling tools and reply with a short plain-text summary of what you changed and how you verified it.\n\
        - If a tool returns an error, read it carefully and correct your next call; do not repeat the same failing call.\n\
        \n\
        Keep replies brief. No filler, no apologies, no repeating file contents the user can already see."
    );
    if let Some(instructions) = agents_md(project_root) {
        prompt.push_str("\n\nProject instructions (AGENTS.md):\n");
        prompt.push_str(&instructions);
    }
    if let Some(map) = project_map(project_root) {
        prompt.push_str("\n\nProject layout (top levels; explore deeper with tools):\n");
        prompt.push_str(&map);
    }
    prompt
}

/// Project-level AGENTS.md, capped. The de facto convention for handing
/// agents project conventions; ignoring it wastes the user's own groundwork.
fn agents_md(project_root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(project_root.join("AGENTS.md")).ok()?;
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if text.len() <= MAX_AGENTS_MD_BYTES {
        return Some(text.to_string());
    }
    let mut cut = MAX_AGENTS_MD_BYTES;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    Some(format!("{}\n…[AGENTS.md truncated; read_file it for the rest]", &text[..cut]))
}

/// A shallow, gitignore-aware file map: directories first, then files, both
/// sorted, capped in depth and bytes.
fn project_map(project_root: &Path) -> Option<String> {
    let mut dirs: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let walk = ignore::WalkBuilder::new(project_root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .max_depth(Some(MAX_MAP_DEPTH))
        .build();
    for entry in walk.flatten() {
        let Ok(rel) = entry.path().strip_prefix(project_root) else { continue };
        let rel = rel.to_string_lossy();
        if rel.is_empty() {
            continue;
        }
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            dirs.push(format!("{rel}/"));
        } else {
            files.push(rel.to_string());
        }
    }
    if dirs.is_empty() && files.is_empty() {
        return None;
    }
    dirs.sort();
    files.sort();
    let mut out = String::new();
    let mut omitted = 0usize;
    for entry in dirs.iter().chain(files.iter()) {
        if out.len() + entry.len() + 1 > MAX_MAP_BYTES {
            omitted += 1;
            continue;
        }
        out.push_str(entry);
        out.push('\n');
    }
    if omitted > 0 {
        out.push_str(&format!("… {omitted} more entries\n"));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_project() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-prompt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("src/nested/deeper")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.join("src/nested/mod.rs"), "").unwrap();
        std::fs::write(dir.join("src/nested/deeper/leaf.rs"), "").unwrap();
        dir
    }

    #[test]
    fn prompt_includes_shallow_layout_but_not_deep_entries() {
        let dir = temp_project();
        let prompt = system_prompt(&dir);
        assert!(prompt.contains("Project layout"));
        assert!(prompt.contains("src/"));
        assert!(prompt.contains("Cargo.toml"));
        assert!(prompt.contains("src/main.rs"));
        // Depth 2 covers root plus one level down; deeper stays for the tools.
        assert!(!prompt.contains("deeper/leaf.rs"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn prompt_includes_capped_agents_md() {
        let dir = temp_project();
        std::fs::write(dir.join("AGENTS.md"), "Always run cargo clippy before finishing.").unwrap();
        let prompt = system_prompt(&dir);
        assert!(prompt.contains("Project instructions (AGENTS.md):"));
        assert!(prompt.contains("Always run cargo clippy"));

        std::fs::write(dir.join("AGENTS.md"), "x".repeat(10_000)).unwrap();
        let prompt = system_prompt(&dir);
        assert!(prompt.contains("AGENTS.md truncated"));
        assert!(prompt.len() < 10_000 + 2_500, "cap must hold, got {}", prompt.len());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_agents_md_adds_nothing() {
        let dir = temp_project();
        let prompt = system_prompt(&dir);
        assert!(!prompt.contains("AGENTS.md"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
