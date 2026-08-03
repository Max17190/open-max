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
pub const MAX_SKILLS_BYTES: usize = 3_000;

/// System prompt: short, imperative, explicit about tool use, and every line
/// has to earn its place. Long "constitution"-style prompts measurably degrade
/// model performance, and the degradation starts well below the context limit,
/// so brevity here is a quality decision before it is a token one.
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
    /// (name, index line chars) per memory the index surfaced.
    pub memory: Vec<(String, usize)>,
}

impl PromptBreakdown {
    /// For resumed sessions the persisted prompt is one opaque string; the
    /// per-tool/skill split still comes from the frozen registry.
    pub fn from_persisted(system_chars: usize, registry: &Registry, project_root: &Path) -> Self {
        let mut breakdown = Self {
            components: vec![("system prompt (persisted)".into(), system_chars)],
            ..Default::default()
        };
        breakdown.add_registry(registry, project_root);
        breakdown
    }

    fn add_registry(&mut self, registry: &Registry, project_root: &Path) {
        if let Some(entries) = registry.tool_schemas_json().as_array() {
            for (entry, spec) in entries.iter().zip(&registry.tools) {
                let external = !matches!(spec.kind, crate::registry::ToolKind::Builtin);
                self.tools.push((spec.name.clone(), entry.to_string().len(), external));
            }
        }
        // The per-skill cost is what the section actually carries: the index
        // line for a skill under the byte cap, nothing for one past it.
        self.skills = skill_index_costs(project_root, &registry.skills);
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
    // The memory index rides the frozen prefix like the skills index: a line
    // per surfaced fact, bodies loaded on demand, nothing when the project
    // has no live memories so the zero-cost invariant holds.
    if let Some((index, rows)) =
        crate::memory::index_section(project_root, crate::memory::unix_now())
    {
        let before = prompt.len();
        prompt.push_str(
            "\n\nMemory (facts saved in earlier sessions; read_file one before relying on it):\n",
        );
        prompt.push_str(&index);
        breakdown.components.push(("memory index".into(), prompt.len() - before));
        breakdown.memory = rows;
    }
    if let Some(map) = project_map(project_root) {
        let before = prompt.len();
        prompt.push_str("\n\nProject layout (top levels; explore deeper with tools):\n");
        prompt.push_str(&map);
        breakdown.components.push(("project layout map".into(), prompt.len() - before));
    }
    if let Some(skills) = skills_section(project_root, &registry.skills, registry.skills_omitted) {
        let before = prompt.len();
        prompt.push_str("\n\nSkills (before using one, read its SKILL.md. Use read_file for paths inside the project. For skill files outside the project (absolute paths), use bash: cat <path>.):\n");
        prompt.push_str(&skills);
        breakdown.components.push(("skills index".into(), prompt.len() - before));
    }
    if registry.tools_omitted > 0 {
        let before = prompt.len();
        prompt.push_str(&format!(
            "\n\n… {} more tools were discovered but not loaded ({}-tool cap): consolidate or delete files in .openmax/tools and ~/.openmax/tools; openmax --check names them.\n",
            registry.tools_omitted,
            crate::registry::MAX_EXTERNAL_TOOLS,
        ));
        breakdown.components.push(("tools trailer".into(), prompt.len() - before));
    }
    breakdown.add_registry(registry, project_root);
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
- Hook: .openmax/hooks/<name>.toml with event pre_tool_use or user_prompt_submit (exit nonzero blocks), post_tool_use, session_start, compaction, or turn_end. Unapproved hooks are inert; approval covers the .toml and the code it runs, and editing either revokes it (a revoked live gate then blocks tools).\n\
- Permission rules: .openmax/permissions.toml, one [[rules]] table per rule with effect = allow|deny|ask, tool = \"<tool name>\", optional arg_regex (unanchored). Any error in this file denies every tool, so write it exactly and check it.\n\
- Provider: use bash to edit ~/.openmax/providers.json for named model endpoints (native file tools are project-confined).\n\
A tool or skill you write goes live before your next step: the harness re-freezes after a successful mutating call and at turn start (/reload also forces it). The harness records tool/skill file changes (actor + hash); bash: openmax --ledger lists history and restorable objects. Hooks, permissions, and templates apply on their next use. Verify what you wrote with bash: openmax --check. Before writing a surface, read its full contract (fields, stdin payloads, activation) with bash: openmax --spec tools|skills|prompts|hooks|permissions|providers|memory|stdio.\n\
Compose beyond the loop with CLI-backed tools + skills. Use a child openmax -p or openmax --stdio process for isolated work, tmux for durable or parallel processes, and the stdio protocol for custom frontends.\n\
\n\
Working files (there is no built-in plan mode or todo list):\n\
- PLAN.md: for multi-step work, write the plan there first and keep it current.\n\
- TODO.md: the running task list; check items off as you finish.\n\
- AGENTS.md: standing project instructions; keep it short (loads at session create and on /reload).\n\
- Memory: one durable fact per file in .openmax/memory/<name>.md; its first line becomes an index line in future sessions. Update or delete stale facts; files never read fade from the index and are deleted after ~60 days. Contract: openmax --spec memory. Search everything past sessions kept: bash: openmax --recall \"<query>\".";

/// One line per skill: name, description, and the SKILL.md path the model
/// reads on demand. Project skills show a project-relative path (read_file
/// reaches it); global skills keep their absolute path (bash reaches it).
/// None when there are no skills: an empty section would still cost tokens
/// and change the byte-stable prompt for nothing.
///
/// `beyond_cap` is how many discovered skills the registry's `MAX_SKILLS`
/// index cap already dropped; the trailer folds those in with the byte-cap
/// omissions so no skill on disk ever vanishes without a count.
fn skills_section(project_root: &Path, skills: &[SkillSpec], beyond_cap: usize) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::new();
    let mut omitted = beyond_cap;
    for (line, included) in skill_index_lines(project_root, skills) {
        if included {
            out.push_str(&line);
        } else {
            omitted += 1;
        }
    }
    if omitted > 0 {
        out.push_str(&format!(
            "… {omitted} more skills (list them: ls .agents/skills ~/.openmax/skills)\n"
        ));
    }
    Some(out)
}

/// One index line per indexed skill, with the byte cap applied exactly as the
/// section applies it: (line, carried). The single source for both the prompt
/// text and every cost display, so what is billed is what is spent.
fn skill_index_lines(project_root: &Path, skills: &[SkillSpec]) -> Vec<(String, bool)> {
    let mut spent = 0usize;
    let mut out = Vec::with_capacity(skills.len());
    for skill in skills {
        let line = format!(
            "- {}: {} — {}\n",
            skill.name,
            skill.description,
            skill_shown_path(project_root, &skill.path)
        );
        let included = spent + line.len() <= MAX_SKILLS_BYTES;
        if included {
            spent += line.len();
        }
        out.push((line, included));
    }
    out
}

/// The path as the index shows it: project-relative for a project-tier skill,
/// absolute for a global one. When the root does not strip - a frozen
/// registry being priced under a root that has since moved or is spelled
/// differently - a project-tier path still carries its tier directory, which
/// is exactly where the freeze-time rendering started; falling back to the
/// absolute path instead would price a line the persisted prompt never held.
fn skill_shown_path(project_root: &Path, path: &Path) -> String {
    if let Ok(relative) = path.strip_prefix(project_root) {
        return relative.display().to_string();
    }
    let absolute = path.display().to_string();
    match absolute.find("/.agents/skills/") {
        Some(i) => absolute[i + 1..].to_string(),
        None => absolute,
    }
}

/// What the frozen prompt actually spends per indexed skill: the index line's
/// bytes for a skill the section carries, zero for one the byte cap dropped
/// (only the omission trailer mentions it). Pricing a dropped skill as if it
/// were carried sends an agent pruning by cost after tokens nobody is paying.
pub fn skill_index_costs(project_root: &Path, skills: &[SkillSpec]) -> Vec<(String, usize)> {
    skill_index_lines(project_root, skills)
        .into_iter()
        .zip(skills)
        .map(|((line, included), skill)| {
            (skill.name.clone(), if included { line.len() } else { 0 })
        })
        .collect()
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
        let discovered = system_prompt(&dir, &Registry::build(&dir.join("data"), &dir));
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
        // The guide is an index; the full per-surface contract is read on
        // demand, and the pointer must name every surface --spec accepts.
        assert!(prompt.contains(
            "openmax --spec tools|skills|prompts|hooks|permissions|providers|memory|stdio"
        ));
        assert!(prompt.contains("user_prompt_submit"));
        assert!(prompt.contains("providers.json"));
        assert!(prompt.contains("Provider: use bash"));
        assert!(prompt.contains("CLI-backed tools + skills"));
        assert!(prompt.contains("openmax -p or openmax --stdio"));
        assert!(prompt.contains("tmux for durable or parallel processes"));
        assert!(prompt.contains("stdio protocol for custom frontends"));
        // The design's "use instead" contract: PLAN.md over plan mode,
        // TODO.md over a todo product, AGENTS.md for standing instructions,
        // and the memory surface for facts that must survive sessions.
        assert!(prompt.contains("PLAN.md"));
        assert!(prompt.contains("TODO.md"));
        assert!(prompt.contains("AGENTS.md: standing project instructions"));
        assert!(prompt.contains(".openmax/memory/<name>.md"));
        assert!(prompt.contains("fade from the index"));
        assert!(prompt.contains("openmax --spec memory"));
        assert!(prompt.contains("openmax --recall"), "preserved history must be findable");
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

    /// Every skill on disk is accounted for: lines shown plus the trailer
    /// count must equal the number discovered, whether a skill was dropped
    /// by the index byte cap or by the registry's MAX_SKILLS cap.
    #[test]
    fn skills_trailer_counts_both_byte_cap_and_index_cap_omissions() {
        let dir = temp_project();
        let total = crate::skills::MAX_SKILLS + 4;
        for i in 0..total {
            let skill_dir = dir.join(".agents/skills").join(format!("s{i:03}"));
            std::fs::create_dir_all(&skill_dir).unwrap();
            std::fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\nname: skill-{i:03}\ndescription: {}\n---\nbody\n", "d".repeat(120)),
            )
            .unwrap();
        }
        let registry = Registry::build(&dir.join("data"), &dir);
        assert_eq!(registry.skills_omitted, 4);
        let prompt = system_prompt(&dir, &registry);
        let shown = prompt.matches("\n- skill-").count();
        let trailer_count: usize = prompt
            .split("… ")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .and_then(|n| n.parse().ok())
            .expect("trailer with a count must be present");
        assert!(prompt.contains("more skills"), "trailer must name the omission");
        assert_eq!(
            shown + trailer_count,
            total,
            "shown ({shown}) + omitted ({trailer_count}) must cover all {total} skills"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Billing equals spending: a carried skill costs exactly the line the
    /// prompt shows (project-relative path included), a byte-capped skill
    /// costs zero, and the resumed-session breakdown agrees with the fresh
    /// one. Before this held, cap-dropped skills were billed in full and
    /// paths were priced absolute: +97% per skill, +300% aggregate, aimed at
    /// an agent told to prune its toolbox by these very numbers.
    #[test]
    fn skill_costs_bill_exactly_what_the_index_carries() {
        let dir = temp_project();
        let total = 30;
        for i in 0..total {
            let skill_dir = dir.join(".agents/skills").join(format!("s{i:03}"));
            std::fs::create_dir_all(&skill_dir).unwrap();
            std::fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\nname: skill-{i:03}\ndescription: {}\n---\nbody\n", "d".repeat(120)),
            )
            .unwrap();
        }
        let registry = Registry::build(&dir.join("data"), &dir);
        assert_eq!(registry.skills_omitted, 0, "all under the count cap");
        let (prompt, breakdown) = system_prompt_with_breakdown(&dir, &registry);
        let shown = prompt.matches("\n- skill-").count();
        assert!(shown > 0 && shown < total, "the byte cap must bite for this test to mean anything");

        let costs = skill_index_costs(&dir, &registry.skills);
        assert_eq!(breakdown.skills, costs, "the breakdown is the same accounting");
        assert_eq!(
            PromptBreakdown::from_persisted(prompt.len(), &registry, &dir).skills,
            costs,
            "a resumed session bills like a fresh one"
        );

        // Carried lines are billed at the exact bytes the prompt spends on
        // them - relative path, trailing newline, nothing else.
        let spent: usize =
            prompt.lines().filter(|l| l.starts_with("- skill-")).map(|l| l.len() + 1).sum();
        let billed: usize = costs.iter().map(|(_, c)| c).sum();
        assert_eq!(billed, spent, "billed ({billed}) must equal spent ({spent})");
        assert_eq!(costs.iter().filter(|(_, c)| *c > 0).count(), shown);

        // A dropped skill is billed nothing and shown nowhere.
        let (dropped, zero) = costs.last().expect("skills exist");
        assert_eq!(*zero, 0, "the sorted tail falls past the byte cap");
        assert!(!prompt.contains(&format!("- {dropped}:")));

        // A carried skill's price is its relative-path line, not the
        // absolute-path one the old accounting invented.
        let first = &registry.skills[0];
        let relative = first.path.strip_prefix(&dir).unwrap();
        assert_eq!(
            costs[0].1,
            format!("- {}: {} — {}\n", first.name, first.description, relative.display()).len()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A frozen registry priced under a root that no longer matches its
    /// paths (a moved or re-spelled project on resume) must reproduce the
    /// persisted prompt's accounting, not re-derive different lines: a
    /// project-tier path still carries `.agents/skills/`, which is where the
    /// freeze-time rendering started.
    #[test]
    fn skill_costs_survive_a_moved_project_root() {
        let dir = temp_project();
        for i in 0..3 {
            let skill_dir = dir.join(".agents/skills").join(format!("m{i}"));
            std::fs::create_dir_all(&skill_dir).unwrap();
            std::fs::write(
                skill_dir.join("SKILL.md"),
                format!("---\nname: moved-{i}\ndescription: d\n---\nbody\n"),
            )
            .unwrap();
        }
        let registry = Registry::build(&dir.join("data"), &dir);
        let at_home = skill_index_costs(&dir, &registry.skills);
        let moved = skill_index_costs(Path::new("/somewhere/else/entirely"), &registry.skills);
        assert_eq!(at_home, moved, "pricing must not depend on where the project sits now");
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
    /// the serialized builtin tool array must stay within ~1180 tokens. The
    /// cap is in chars (the core stays tokenizer-free): the pre-guide 3452
    /// chars including a 52-char project root measured 794 tokens on
    /// o200k_base and 775 on cl100k_base (2026-07-16); the guide adds ~360
    /// tokens. The interpolated root varies per machine, so it is excluded
    /// here and the cap leaves room for a typical checkout path. If this
    /// fails, re-measure with a real tokenizer before raising anything.
    /// Only builtins count: external tools are the user's own budget, and
    /// grounding sections (AGENTS.md, layout map, skills) have their own caps.
    ///
    /// Raised from 4900 to 5020 (2026-07-30): what a hook approval covers is
    /// now stated in the guide (+119 chars, ~30 tokens). The old line said
    /// unapproved hooks are inert and stopped there, which taught the agent
    /// that rewriting a hook's script was harmless - the exact belief the
    /// approval binding exists to correct.
    ///
    /// Raised from 5020 to 5280 (2026-07-31): the working-files contract now
    /// names the memory surface (+220 chars). Re-measured per the rule above
    /// with `dump_frozen_prompt_payload_for_tokenizer`: 1196 tokens on
    /// o200k_base, 1174 on cl100k_base at 5240 path-free chars, so the cap
    /// tracks ~1200 tokens. The old guide taught "there is no memory", which
    /// held the agent to AGENTS.md hand-editing; ~20 real tokens is the price
    /// of the index being discoverable at all.
    ///
    /// Raised from 5280 to 5360 (2026-07-31): the memory line now points at
    /// `openmax --recall` (+79 chars). Re-measured: 1214 tokens on
    /// o200k_base, 1191 on cl100k_base at 5312 path-free chars. Preserved
    /// history the agent cannot find is a promise, not a capability; 18 real
    /// tokens buys the pointer that makes archives and past sessions
    /// searchable instead of merely stored.
    #[test]
    fn frozen_prompt_fits_token_budget() {
        let dir = temp_project();
        let registry = crate::registry::Registry::build(&dir.join("data"), &dir);
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
            total <= 5_360,
            "frozen prompt budget exceeded: base rules + guide (path-free) {path_free} + builtin tools {tool_chars} = {total} chars (cap 5360 ≈ 1215 tokens with a typical checkout path)",
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Measurement helper for cap raises (see the budget-gate comment): dumps
    /// the exact path-free payload the cap test measures so a real tokenizer
    /// can count it. Run with `--ignored --nocapture`; files land in the OS
    /// temp dir.
    #[test]
    #[ignore]
    fn dump_frozen_prompt_payload_for_tokenizer() {
        let dir = temp_project();
        let registry = crate::registry::Registry::build(&dir.join("data"), &dir);
        let (prompt, breakdown) = system_prompt_with_breakdown(&dir, &registry);
        let base_chars: usize = breakdown
            .components
            .iter()
            .filter(|(name, _)| name == "base rules" || name == "self-extension guide")
            .map(|(_, c)| *c)
            .sum();
        let path_free = prompt[..base_chars].replace(&dir.to_string_lossy().to_string(), "");
        let base_path = std::env::temp_dir().join("openmax-prompt-base.txt");
        let tools_path = std::env::temp_dir().join("openmax-prompt-tools.json");
        std::fs::write(&base_path, path_free).unwrap();
        std::fs::write(&tools_path, registry.tool_schemas_wire()).unwrap();
        eprintln!("wrote {} and {}", base_path.display(), tools_path.display());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The memory index is a prompt section like the skills index: one line
    /// per surfaced fact when memories exist, and nothing at all when none
    /// do, so a memoryless project's prompt stays byte-identical.
    #[test]
    fn memory_index_appears_only_when_memories_exist() {
        let dir = temp_project();
        let registry = Registry::builtin_only();
        let without = system_prompt(&dir, &registry);
        assert!(!without.contains("Memory ("), "no memories, no section:\n{without}");

        std::fs::create_dir_all(dir.join(".openmax/memory")).unwrap();
        std::fs::write(
            dir.join(".openmax/memory/deploy-port.md"),
            "# The deploy port is 7443\nSet in infra/nginx.conf.",
        )
        .unwrap();
        let (with, breakdown) = system_prompt_with_breakdown(&dir, &registry);
        assert!(
            with.contains("Memory (facts saved in earlier sessions"),
            "memory section must appear:\n{with}"
        );
        assert!(with.contains(
            "- deploy-port: The deploy port is 7443 — .openmax/memory/deploy-port.md"
        ));
        assert_eq!(breakdown.memory.len(), 1);
        assert!(breakdown.components.iter().any(|(name, _)| name == "memory index"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
