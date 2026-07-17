# Open Max (project instructions)

Copy to `AGENTS.md` to inject (capped ~2KB at session create). Concrete facts only.

## Thesis

Native Rust coding-agent harness: one focused loop, small tools, fast TUI, extensions as files. The agent shapes workflows via skills, tools, hooks, permissions, and project files. Small honest core over always-on features. Token cost is design (`/context`).

## Not in core

| Not built in | Use instead |
| --- | --- |
| MCP | CLI tools + skills |
| Nested agents | Focused tools, or second `openmax` / `openmax -p` (tmux). Skill: `parallel-explore`. |
| Plan mode | Write `PLAN.md` |
| Background bash product | tmux sessions |
| Built-in TODOs | Write `TODO.md` |

## What ships

Tools: `list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`.

- Tools: `.openmax/tools/*.toml` or `~/.openmax/tools/`
- Skills: `.agents/skills/*/SKILL.md` or `~/.openmax/skills/` (index only; read body on demand)
- Prompt templates: `.agents/prompts/<name>.md` or `~/.openmax/prompts/` (`$ARGUMENTS`, `$1`..`$9`; user runs `/<name>`)
- Hooks: `.openmax/hooks/*.toml` (`pre_tool_use` gates; `post_tool_use` / `session_start` / `compaction` observe; not in prompt)
- Permissions: `.openmax/permissions.toml` or `~/.openmax/permissions.toml` (allow/deny/ask; not in prompt)
- Providers: `~/.openmax/providers.json`; `/theme` for built-in palettes
- Built-in context compaction; tools/skills freeze at session create (`/reload` to re-freeze in place, `/new` for clean)

Not shipped: user keybindings, theme file hot reload, pluggable compactors, drop-in custom TUI.

## Repo

`crates/core/` harness. `crates/tui/` (`openmax`).

## Development

- Small focused diffs; inspect before edit; match style.
- Verify: `cargo check`, `cargo test`; release: `cargo build --release -p open-max-tui`.
- Prefer skill/tool/hook/permission file first. Always-on costs tokens.
- Never invent paths or claim missing features.
- Branches: professional kebab-case, no agent prefixes. Conventional commits; no agent co-authors. No em dashes.

When adding capability: skill/tool/hook first? Token tax? Prefer files unless it strengthens the minimal harness.
