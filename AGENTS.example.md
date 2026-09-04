# Open Max (project instructions)

Copy to `AGENTS.md` to inject (capped at 2,000 bytes at session create; the rest is cut). Concrete facts only.

## Thesis

Native Rust coding-agent harness: one focused loop, small tools, fast TUI, extensions as files. The agent shapes workflows via skills, tools, hooks, permissions, and project files. Small honest core over always-on features. Token cost is design (`/context`).

## Not in core

| Not built in | Use instead |
| --- | --- |
| MCP | CLI tools + skills |
| Nested agents | Focused tools, or a child `openmax -p` / `openmax --stdio` (tmux); ask the agent to author a delegate skill when the project wants one. |
| Plan mode | Write `PLAN.md` |
| Background bash product | tmux sessions |
| Built-in TODOs | Write `TODO.md` |

## What ships

Seven tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`) plus the file surfaces the system prompt names; tools, skills, prompt templates, hooks, and permissions also load from `~/.openmax/`, project wins on a name collision. `openmax --spec <surface>` prints a contract; `openmax --check` validates every file with reasons. Hooks and permission `allow` rules wait for a human's `openmax --approve`. Headless and stdio runs need `--trust-project` once.

Not shipped: user keybindings, theme file hot reload, pluggable compactors, TUI plugin ABI (custom frontends speak `--stdio` JSONL).

## Repo

`crates/core/` harness. `crates/tui/` (`openmax`).

## Development

- Small focused diffs; inspect before edit; match style.
- Verify: `cargo check`, `cargo test`; release: `cargo build --release -p openmax`.
- Prefer skill/tool/hook/permission file first. Always-on costs tokens.
- Never invent paths or claim missing features.
- Branches: professional kebab-case, no agent prefixes. Conventional commits; no agent co-authors. No em dashes.

When adding capability: skill/tool/hook first? Token tax? Prefer files unless it strengthens the minimal harness.
