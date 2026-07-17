# Open Max

**A small Rust agent harness for coding in the terminal.**

Open Max is a single binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. Point it at the model server you choose: local, cloud, or a private proxy. No desktop shell, no heavyweight runtime, no telemetry.

You own the endpoints, the tools, the skills, and the context.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

## Features

- **Small by default.** Seven built-in tools: `list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `bash`. Short system prompt; old tool output is dropped before your task is.
- **Your model, your server.** Set one `base_url`, or name several endpoints in `providers.json` and switch with `/provider` or `--provider`. Works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud gateways (OpenRouter and similar), and private proxies.
- **Approvals by default.** `write_file`, `edit_file`, and `bash` wait for approval in `ask` mode. Use `auto` for unattended runs or `readonly` to block mutating tools.
- **File based extensions.** Drop TOML tools, `SKILL.md` skills, prompt templates, and process hooks under project or home config. No fork required. The agent knows these surfaces and writes them itself when you ask for a reusable capability; the harness re-freezes automatically on the next turn, so a tool the agent writes is a tool the agent uses.
- **Visible work.** Reads, greps, diffs, and shell commands stream as they happen in a fullscreen TUI. Headless print mode for scripts and CI.
- **Local sessions.** Conversation state lives under `~/.openmax/`. Network goes only to the model endpoint you configure (plus Hugging Face if you use managed model download).

## Install

**Requirements:** [Rust](https://rustup.rs).

```sh
git clone https://github.com/Max17190/open-max.git
cd open-max
cargo install --path crates/tui --locked
```

Or run from source:

```sh
cargo run --release -p open-max-tui
```

## Configure

Edit `~/.openmax/settings.json`:

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5-coder:7b",
  "api_key": null,
  "approval_mode": "ask"
}
```

`base_url` is the root of your model's HTTP API (the harness calls `chat/completions` on it). Set `model` to the id that server expects. Set `api_key` to a literal or `$ENV_VAR`, or export `OPENMAX_API_KEY`.

For several servers, define them in `~/.openmax/providers.json` and select with `"provider"` in settings, `--provider`, or `/provider`. Optional `compat` flags cover picky gateways (for example `max_completion_tokens` vs `max_tokens`).

On Apple Silicon, when `base_url` is the managed local port, Open Max can optionally provision and serve MLX models via `/models`.

## Use

```sh
cd ~/code/my-app
openmax
```

```sh
openmax --continue                    # resume latest session here
openmax -c
openmax --provider ollama --model qwen2.5-coder:7b
openmax -p "summarize the top level layout of this repo"
openmax -p --json "list public modules in crates/core"
```

In print mode, text goes to stdout and tool progress to stderr. With `--json`, each `AgentEvent` is one JSON line on stdout. Mutating tools still honor `approval_mode`; for unattended runs set `"approval_mode": "auto"`.

For a full interactive session over pipes, `openmax --stdio` speaks JSONL both ways: commands on stdin (`{"cmd":"user","text":...}`, `approve`, `cancel`, `quit`), `AgentEvent` envelopes on stdout, one hello line first. Approvals are forwarded to the client instead of auto-declined, and EOF drains the in-flight turn before exit. This is the contract for custom frontends, editor integrations, and one openmax driving another (see the `delegate` skill).

| Input | Action |
| --- | --- |
| **Enter** | Send (queues if the agent is busy) |
| **/** | Slash commands · **Tab** or **Enter** completes |
| **@** | Mention a project file |
| **Esc** | Close menu · cancel turn · return to composer |
| **Ctrl+C** twice | Quit |

| Slash command | Action |
| --- | --- |
| `/help` | Keybindings and commands |
| `/model <id>` | Set the active model |
| `/provider [name]` | List or switch providers |
| `/approvals auto\|ask\|readonly` | Mutating tool gates |
| `/new` · `/resume` | Fresh session · pick an earlier one |
| `/reload` | Force a re-freeze now (it also happens automatically when extension files change) |
| `/tools` · `/skills` · `/context` | Session tools, skills, token budget |
| `/<template> [args]` | Run a prompt template from `.agents/prompts/` |
| `/status` | Endpoint and network destinations |
| `/quit` | Exit |

## Extend

With nothing installed, extensions cost zero tokens. Project paths win over global ones on name collision.

**Tools.** A TOML file in `.openmax/tools/` or `~/.openmax/tools/`. The harness runs `command`, writes JSON args to stdin, and returns stdout.

```toml
# .openmax/tools/todo_scan.toml
name = "todo_scan"
description = "List TODO/FIXME comments with file and line"
command = "./scripts/todo-scan.sh"
timeout_secs = 30
mutating = false

[params]
type = "object"
[params.properties.path]
type = "string"
description = "Directory to scan"
```

**Skills.** A directory with `SKILL.md` under `.agents/skills/` or `~/.openmax/skills/`. Only `name` and `description` live in the prompt; the model reads the full file when needed.

```
.agents/skills/release/SKILL.md
---
name: release
description: How to cut a release of this project
---
Full instructions, checklists, commands...
```

**Prompt templates.** A markdown file under `.agents/prompts/` or `~/.openmax/prompts/`; the file stem becomes a slash command. `$ARGUMENTS` expands to the raw argument string, `$1`..`$9` to positionals, and `$$` escapes a literal dollar; a template without placeholders gets the arguments appended. Templates are message content: re-read on every use, zero prompt tax, never frozen.

```
.agents/prompts/fix-issue.md
---
description: Fix a GitHub issue by number
---
Fetch issue $1 with `gh issue view $1`, reproduce it, fix it, and add a test.
```

Run it as `/fix-issue 42`.

**Hooks.** Optional process gates under `.openmax/hooks/` or `~/.openmax/hooks/`. `pre_tool_use` can block a tool (nonzero exit); `post_tool_use`, `session_start` (a session's first turn), `compaction` (context was pruned; receives the digest record), and `turn_end` (receives the stop reason, fires even on cancel) observe only. Each hook gets one JSON payload on stdin. Hooks never enter the model prompt.

**Permissions.** Optional rules under `.openmax/permissions.toml` or `~/.openmax/permissions.toml` (project first). Not in the model prompt; empty discovery is free. First match wins. Order: hooks pre → permissions → `approval_mode` → execute → hooks post. If a permissions file exists but is invalid, every tool is denied (fail closed).

```toml
# .openmax/permissions.toml
[[rules]]
effect = "deny"
tool = "bash"
arg_regex = "rm\\s+-rf"

[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^cargo (test|check|build)"
```

`effect` is `allow`, `deny`, or `ask`. `arg_regex` is optional: command for `bash`, path for file tools, pattern for `glob`/`grep`. For custom tools it matches the full serialized JSON arguments. Omit `arg_regex` (or leave it empty) to match every call of that tool.

Tools and skills freeze per session for prompt-cache stability, and the harness re-freezes them automatically: at each turn start it fingerprints the extension files, and if anything changed it rebuilds the registry and prompt in place (one deliberate cache re-prefill, conversation kept, a `refrozen` event for clients). An unchanged disk costs nothing. `/reload` forces it immediately; `/new` starts clean. Hooks, permissions, and templates re-discover on every turn or invocation. Use `/tools`, `/skills`, and `/context` to inspect the frozen set and its cost.

## Privacy

Open Max does not phone home. The only network destinations are:

1. The model endpoint in `base_url` (your choice).
2. Hugging Face, only when you download or serve a model through `/models`.

Sessions, settings, tools, and skills stay under `~/.openmax/` and your project directory. `/status` lists destinations; the status bar shows `no telemetry` when idle. External tools you install may open their own network connections.

## Development

```sh
cargo check
cargo test
cargo build --release -p open-max-tui
```

Core logic is in `open-max-core` (`crates/core/src/agent.rs`). The TUI is `crates/tui/src/app.rs`.

## Status

Open Max is early software (v0.2.0). The agent loop, session persistence, extensibility, TUI, and GitHub Actions CI (test + release build + soft size gate) are in place, but there is no install script or published release channel yet. Expect rough edges. File an issue or send a PR if something breaks.

## License

MIT. See [LICENSE](LICENSE).
