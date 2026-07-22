use std::path::Path;

use crate::registry::Registry;
use crate::skills::SkillSpec;

/// AGENTS.md content is user-authored instructions; cap it so a sprawling
/// file cannot crowd a 16k window.
const MAX_AGENTS_MD_BYTES: usize = 2_000;
/// The layout map exists to save discovery round trips, not to describe the
/// tree exhaustively; ~300 tokens is the budget.
const MAX_MAP_BYTES: usize = 1_200;
const MAX_MAP_DEPTH: usize = 2;
/// The skills index is a name+description line per skill; past this it is a
/// prompt tax, not an index.
const MAX_SKILLS_BYTES: usize = 3_000;

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
pub fn system_prompt(project_root: &Path, registry: &Registry) -> String {
    system_prompt_with_breakdown(project_root, registry).0
}

/// What one session's frozen prompt prefix is spent on, measured at the only
/// moment the components are individually known: creation. Char counts;
/// display divides by ~4 for tokens, the same heuristic the budget uses.
#[derive(Clone, Debug, Default)]
pub struct PromptBreakdown {
    /// Labeled prompt text components, in prompt order.
    pub components: Vec<(String, usize)>,
    /// (name, serialized schema chars, is_external) per tool.
    pub tools: Vec<(String, usize, bool)>,
    /// (name, index line chars) per skill.
    pub skills: Vec<(String, usize)>,
}

impl PromptBreakdown {
    /// For resumed sessions the persisted prompt is one opaque string; the
    /// per-tool/skill split still comes from the frozen registry.
    pub fn from_persisted(system_chars: usize, registry: &Registry) -> Self {
        let mut breakdown = Self {
            components: vec![("system prompt (persisted)".into(), system_chars)],
            ..Default::default()
        };
        breakdown.add_registry(registry);
        breakdown
    }

    fn add_registry(&mut self, registry: &Registry) {
        if let Some(entries) = registry.tool_schemas_json().as_array() {
            for (entry, spec) in entries.iter().zip(&registry.tools) {
                let external = !matches!(spec.kind, crate::registry::ToolKind::Builtin);
                self.tools.push((spec.name.clone(), entry.to_string().len(), external));
            }
        }
        for skill in &registry.skills {
            // The per-skill cost is its index line; the body loads on demand.
            let line = format!("- {}: {} — {}\n", skill.name, skill.description, skill.path.display());
            self.skills.push((skill.name.clone(), line.len()));
        }
    }
}

pub fn system_prompt_with_breakdown(project_root: &Path, registry: &Registry) -> (String, PromptBreakdown) {
    let root = project_root.to_string_lossy();
    let mut breakdown = PromptBreakdown::default();
    // Tool-specific guidance lives in each tool's schema description (which
    // rides in every request anyway); rules here are only the cross-cutting
    // ones. Both sides count against the frozen prompt budget in
    // `frozen_prompt_fits_token_budget`.
    let mut prompt = format!(
        "You are Open Max, a coding agent working on the project at {root}.\n\
        \n\
        Rules:\n\
        - All tool paths are project-relative.\n\
        - Never invent paths or file contents; read the real code first.\n\
        - Prefer edit_file for existing files; write_file only for new files or full rewrites.\n\
        - Make small, focused changes in the existing style; verify by compile, test, or re-read.\n\
        - On a tool error, correct the next call; never repeat a failing call.\n\
        - Never print tool-call JSON or XML as reply text; call tools only via the API.\n\
        - When done, stop calling tools; reply with a short summary of changes and verification.\n\
        \n\
        Keep replies brief: no filler, no repeating file contents."
    );
    breakdown.components.push(("base rules".into(), prompt.len()));
    {
        let before = prompt.len();
        prompt.push_str(SELF_EXTENSION);
        breakdown.components.push(("self-extension guide".into(), prompt.len() - before));
    }
    if let Some(instructions) = agents_md(project_root) {
        let before = prompt.len();
        prompt.push_str("\n\nProject instructions (AGENTS.md):\n");
        prompt.push_str(&instructions);
        breakdown.components.push(("AGENTS.md".into(), prompt.len() - before));
    }
    if let Some(map) = project_map(project_root) {
        let before = prompt.len();
        prompt.push_str("\n\nProject layout (top levels; explore deeper with tools):\n");
        prompt.push_str(&map);
        breakdown.components.push(("project layout map".into(), prompt.len() - before));
    }
    if let Some(skills) = skills_section(project_root, &registry.skills) {
        let before = prompt.len();
        prompt.push_str("\n\nSkills (before using one, read its SKILL.md. Use read_file for paths inside the project. For skill files outside the project (absolute paths), use bash: cat <path>.):\n");
        prompt.push_str(&skills);
        breakdown.components.push(("skills index".into(), prompt.len() - before));
    }
    breakdown.add_registry(registry);
    (prompt, breakdown)
}

/// The agent is responsible for its own extensibility: when the user asks for
/// a recurring capability, workflow, or policy, the right move is usually to
/// write one of these files rather than to improvise each time. Static text,
/// so the zero-cost invariant (byte-identical prompt with nothing installed)
/// still holds; roughly 360 tokens is the price of an agent that can grow and
/// compose itself without permanent orchestration features.
const SELF_EXTENSION: &str = "\n\nExtend yourself by writing files when the user asks for a reusable capability:\n\
- New tool: .openmax/tools/<name>.toml with name, description, params (JSON schema), command, args, mutating.\n\
- New skill: .agents/skills/<name>/SKILL.md with frontmatter name + description; body loads on demand.\n\
- Prompt template: .agents/prompts/<name>.md ($ARGUMENTS and $1..$9 expand); the user runs it as /<name>.\n\
- Hook: .openmax/hooks/<name>.toml with event pre_tool_use or user_prompt_submit (exit nonzero blocks), post_tool_use, session_start, compaction, or turn_end.\n\
- Permission rules: .openmax/permissions.toml with allow/deny/ask entries.\n\
- Provider: ~/.openmax/providers.json for named model endpoints.\n\
A tool or skill you write is live on the next turn (the harness re-freezes automatically; /reload forces it now). Hooks, permissions, and templates apply on their next use. Verify what you wrote with bash: openmax --check.\n\
Compose beyond the loop with CLI-backed tools + skills. Use a child openmax -p or openmax --stdio process for isolated work, tmux for durable or parallel processes, and the stdio protocol for custom frontends.\n\
\n\
Working files (there is no built-in plan mode, todo list, or memory):\n\
- PLAN.md: for multi-step work, write the plan there first and keep it current.\n\
- TODO.md: the running task list; check items off as you finish.\n\
- AGENTS.md: durable project facts worth remembering across sessions; keep it short (loads at session create and on /reload).";

/// One line per skill: name, description, and the SKILL.md path the model
/// reads on demand. Project skills show a project-relative path (read_file
/// reaches it); global skills keep their absolute path (bash reaches it).
/// None when there are no skills: an empty section would still cost tokens
/// and change the byte-stable prompt for nothing.
fn skills_section(project_root: &Path, skills: &[SkillSpec]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::new();
    let mut omitted = 0usize;
    for skill in skills {
        let shown = skill
            .path
            .strip_prefix(project_root)
            .unwrap_or(&skill.path)
            .display();
        let line = format!("- {}: {} — {}\n", skill.name, skill.description, shown);
        if out.len() + line.len() > MAX_SKILLS_BYTES {
            omitted += 1;
            continue;
        }
        out.push_str(&line);
    }
    if omitted > 0 {
        out.push_str(&format!("… {omitted} more skills\n"));
    }
    Some(out)
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

    fn builtin_prompt(dir: &Path) -> String {
        system_prompt(dir, &Registry::builtin_only())
    }

    /// The zero-cost invariant: with nothing installed, the prompt from a
    /// discovered registry is byte-identical to the builtin-only prompt and
    /// carries no skills section at all.
    #[test]
    fn zero_extensions_prompt_is_byte_identical() {
        let dir = temp_project();
        let discovered = system_prompt(&dir, &Registry::build(&dir));
        assert_eq!(discovered, builtin_prompt(&dir));
        assert!(!discovered.contains("Skills"));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The self-extension guide is part of every frozen prompt: the agent
    /// must know the file surfaces it can grow through and that /reload or
    /// /new activates frozen ones.
    #[test]
    fn prompt_carries_self_extension_guide() {
        let dir = temp_project();
        let prompt = builtin_prompt(&dir);
        assert!(prompt.contains("Extend yourself by writing files"));
        assert!(prompt.contains(".openmax/tools/<name>.toml"));
        assert!(prompt.contains(".agents/prompts/<name>.md"));
        assert!(prompt.contains("/reload"));
        assert!(prompt.contains("openmax --check"));
        assert!(prompt.contains("user_prompt_submit"));
        assert!(prompt.contains("providers.json"));
        assert!(prompt.contains("CLI-backed tools + skills"));
        assert!(prompt.contains("openmax -p or openmax --stdio"));
        assert!(prompt.contains("tmux for durable or parallel processes"));
        assert!(prompt.contains("stdio protocol for custom frontends"));
        // The design's "use instead" contract: PLAN.md over plan mode,
        // TODO.md over a todo product, AGENTS.md as durable memory.
        assert!(prompt.contains("PLAN.md"));
        assert!(prompt.contains("TODO.md"));
        assert!(prompt.contains("AGENTS.md: durable project facts"));
        assert!(prompt.contains("on /reload"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn skills_section_shows_relative_and_absolute_paths() {
        let dir = temp_project();
        let inside = dir.join(".agents/skills/review/SKILL.md");
        let outside = std::path::PathBuf::from("/somewhere/global/skills/pdf/SKILL.md");
        let registry = Registry::assemble(
            Vec::new(),
            vec![
                SkillSpec { name: "code-review".into(), description: "reviews a diff".into(), path: inside },
                SkillSpec { name: "pdf-tools".into(), description: "handles PDFs".into(), path: outside },
            ],
        );
        let prompt = system_prompt(&dir, &registry);
        assert!(prompt.contains("Skills (before using one"));
        assert!(
            prompt.contains("- code-review: reviews a diff — .agents/skills/review/SKILL.md"),
            "project skill must show a project-relative path:\n{prompt}"
        );
        assert!(
            prompt.contains("- pdf-tools: handles PDFs — /somewhere/global/skills/pdf/SKILL.md"),
            "global skill keeps its absolute path:\n{prompt}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn prompt_includes_shallow_layout_but_not_deep_entries() {
        let dir = temp_project();
        let prompt = builtin_prompt(&dir);
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
        let prompt = builtin_prompt(&dir);
        assert!(prompt.contains("Project instructions (AGENTS.md):"));
        assert!(prompt.contains("Always run cargo clippy"));

        std::fs::write(dir.join("AGENTS.md"), "x".repeat(10_000)).unwrap();
        let prompt = builtin_prompt(&dir);
        assert!(prompt.contains("AGENTS.md truncated"));
        assert!(prompt.len() < 10_000 + 2_500, "cap must hold, got {}", prompt.len());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_agents_md_adds_nothing() {
        let dir = temp_project();
        let prompt = builtin_prompt(&dir);
        // The self-extension guide names AGENTS.md as a working file; the
        // injected project-instructions section must stay absent when no
        // AGENTS.md exists on disk.
        assert!(!prompt.contains("Project instructions (AGENTS.md):"));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Budget gate for the frozen prompt prefix: base system prompt, the
    /// self-extension guide (now including the working-files contract), and
    /// the serialized builtin tool array must stay within ~1150 tokens. The
    /// cap is in chars (the core stays tokenizer-free): the pre-guide 3452
    /// chars including a 52-char project root measured 794 tokens on
    /// o200k_base and 775 on cl100k_base (2026-07-16); the guide adds ~360
    /// tokens. The interpolated root varies per machine, so it is excluded
    /// here and the cap (4900) leaves room for a typical checkout path. If
    /// this fails, re-measure with a real tokenizer before raising anything.
    /// Only builtins count: external tools are the user's own budget, and
    /// grounding sections (AGENTS.md, layout map, skills) have their own caps.
    #[test]
    fn frozen_prompt_fits_token_budget() {
        let dir = temp_project();
        let registry = crate::registry::Registry::build(&dir);
        let (_, breakdown) = system_prompt_with_breakdown(&dir, &registry);
        let base_chars = breakdown
            .components
            .iter()
            .filter(|(name, _)| name == "base rules" || name == "self-extension guide")
            .map(|(_, c)| *c)
            .sum::<usize>();
        let path_free = base_chars - dir.to_string_lossy().len();
        // Serialize the builtin entries as one array so brackets and commas
        // count, exactly as the wire payload does.
        let builtins: Vec<&serde_json::Value> = registry
            .tool_schemas_json()
            .as_array()
            .expect("schemas are an array")
            .iter()
            .zip(&registry.tools)
            .filter(|(_, spec)| matches!(spec.kind, crate::registry::ToolKind::Builtin))
            .map(|(entry, _)| entry)
            .collect();
        let tool_chars = serde_json::to_string(&builtins).expect("serialize").len();
        let total = path_free + tool_chars;
        assert!(
            total <= 4_900,
            "frozen prompt budget exceeded: base rules + guide (path-free) {path_free} + builtin tools {tool_chars} = {total} chars (cap 4900 ≈ 1150 tokens with a typical checkout path)",
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
