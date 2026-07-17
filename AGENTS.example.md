# Open Max (project instructions)

Copy to `AGENTS.md` to inject (capped ~2KB at session create). Concrete facts only.

## Thesis

Native Rust coding-agent harness: one focused loop, small tools, fast TUI, extensions as files. The agent shapes workflows via skills, tools, hooks, and project files. Small honest core over always-on features. Token cost is design (`/context`).

## Not in core

| Not built in | Use instead |
| --- | --- |
| MCP | CLI tools + skills |
| Nested agents (direction) | Prefer focused search or second `openmax` / `openmax -p` (tmux). Skill: `parallel-explore`. Stock may still register read-only `task`. |
| Plan mode | Write `PLAN.md` |
| Background bash product | tmux sessions |
| Built-in TODOs | Write `TODO.md` |

## What ships

Tools: `list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`, currently `task` (prefer not to rely on nested explore).

- Tools: `.openmax/tools/*.toml` or `~/.openmax/tools/`
- Skills: `.agents/skills/*/SKILL.md` or `~/.openmax/skills/` (index only; read body on demand)
- Hooks: `.openmax/hooks/*.toml` (`pre_tool_use` / `post_tool_use`; not in prompt)
- Providers: `~/.openmax/providers.json`; `/theme` for built-in palettes
- Built-in context compaction; tools/skills freeze at session create

Not shipped: prompt templates, user slash commands/keybindings, theme file hot reload, pluggable compactors, drop-in custom TUI.

## Repo

`crates/core/` harness. `crates/tui/` (`openmax`).

## Development

- Small focused diffs; inspect before edit; match style.
- Verify: `cargo check`, `cargo test`; release: `cargo build --release -p open-max-tui`.
- Prefer skill/tool/hook first. Always-on costs tokens.
- Never invent paths or claim missing features.
- Branches: professional kebab-case, no agent prefixes. Conventional commits; no agent co-authors. No em dashes.

When adding capability: skill/tool/hook first? Token tax? Prefer files unless it strengthens the minimal harness.
