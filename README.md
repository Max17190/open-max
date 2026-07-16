# Open Max

**A small Rust agent harness for coding in the terminal.**

Open Max is a single binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. Point it at the model server you choose: local, cloud, or a private proxy. No desktop shell, no heavyweight runtime, no telemetry.

You own the endpoints, the tools, the skills, and the context.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

## Features

- **Small by default.** Eight built-in tools: `list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`, and a read only `task` for context isolation. Short system prompt; old tool output is dropped before your task is.
- **Your model, your server.** Set one `base_url`, or name several endpoints in `providers.json` and switch with `/provider` or `--provider`. Works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud gateways (OpenRouter and similar), and private proxies.
- **Approvals by default.** `write_file`, `edit_file`, and `bash` wait for approval in `ask` mode. Use `auto` for unattended runs or `readonly` to block mutating tools.
- **File based extensions.** Drop TOML tools, `SKILL.md` skills, and process hooks under project or home config. No fork required.
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
| `/tools` · `/skills` · `/context` | Session tools, skills, token budget |
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

**Hooks.** Optional process gates under `.openmax/hooks/` or `~/.openmax/hooks/`. `pre_tool_use` can block a tool; `post_tool_use` observes only. Hooks never enter the model prompt.

Tools and skills are discovered once at session start and frozen for that session. Changes apply on `/new`. Use `/tools`, `/skills`, and `/context` to inspect the frozen set and its cost.

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

Open Max is early software (v0.2.0). The agent loop, session persistence, extensibility, and TUI are in place, but there is no install script, CI, or published release channel yet. Expect rough edges. File an issue or send a PR if something breaks.

## License

MIT. See [LICENSE](LICENSE).
