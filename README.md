# Open Max

**A self-extending Rust agent harness for coding in the terminal.**

Open Max is a single binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. Point it at the model server you choose: local, cloud, or a private proxy. No desktop shell, no heavyweight runtime, no telemetry.

You own the endpoints, the tools, the skills, and the context.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

## Features

- **Small by default.** Seven built-in tools: `list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `bash`. Short system prompt; old tool output is dropped before your task is, and dropped context is summarized by your own model into a compact note (heuristic digest as fallback).
- **Your model, your server.** Set one `base_url`, or name several endpoints in `providers.json` and switch model and provider together with `/model`. `/provider` and the provider CLI option remain available for direct provider changes. Works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud gateways (OpenRouter and similar), and private proxies.
- **Trust before execution.** An exact canonical project root must be trusted before any agent turn or project behavior starts. Interactive use asks once; headless and stdio runs fail closed until explicitly started with `--trust-project`.
- **Approvals by default.** `write_file`, `edit_file`, and `bash` wait for approval in `ask` mode. Use `auto` for unattended runs or `readonly` to block mutating tools. Approvals and permissions decide whether Open Max dispatches a tool call; they are not OS isolation.
- **File based extensions.** Drop TOML tools, `SKILL.md` skills, prompt templates, and process hooks under project or home config. No fork required. The agent knows these surfaces and writes them itself when you ask for a reusable capability; the harness re-freezes automatically on the next turn, so a tool the agent writes is a tool the agent uses.
- **Visible work.** Reads, greps, diffs, and shell commands stream as they happen in a fullscreen TUI. Headless print mode for scripts and CI.
- **Local sessions.** Conversation state lives under `~/.openmax/`. The harness contacts only the model endpoint you configure, plus Hugging Face if you use managed model download. Native child processes can make their own network connections with the host authority Open Max inherits.

## The intelligent harness

Open Max's thesis is that it is the world's first intelligent harness: a living system that can construct the next capability it needs from ordinary files and native processes. A new tool, skill, hook, template, provider, tmux process, or frontend is a new neuron. The harness discovers it, gives the agent the minimum necessary description, and keeps the richer behavior outside the permanent loop.

The design starts with one question: **What is the smallest capability Open Max must provide so the agent can construct richer behavior itself?** The answer is one focused native loop, seven primitive tools, a fast event-driven TUI, context management, and stable file and process contracts.

| Need | Construct it with |
| --- | --- |
| External service or specialized capability | A CLI-backed TOML tool plus an on-demand skill, without an MCP runtime |
| Reusable workflow or command | A `SKILL.md` package or prompt template slash command |
| Isolated or parallel work | A child `openmax -p` or interactive `openmax --stdio` process, usually in tmux |
| Durable background work | A named tmux session that the agent can inspect and reattach |
| Planning and task state | `PLAN.md` and `TODO.md`, visible to the user and every tool |
| Lifecycle policy and events | Process hooks and permission files |
| Compaction integration | The built-in model summary plus the `compaction` hook event |
| Model endpoints | `providers.json`, including local servers, gateways, and private proxies |
| Shortcuts or a completely different UI | Prompt templates, or a custom frontend speaking `openmax-stdio/1` |

These are deliberate boundaries, not placeholders for hidden orchestration products. Open Max does not carry an MCP host, nested-agent scheduler, plan mode, background-job product, built-in TODO database, user-keybinding engine, pluggable compactor, or TUI plugin ABI. The agent composes those richer workflows from the same host tools a developer can inspect, edit, test, and remove.

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

For several servers, define them in `~/.openmax/providers.json`. `/model` opens a searchable local catalog and selects the provider and model as one pair. Model names are optional, model ids are sent unchanged, and the configured order is preserved within each provider. Opening the picker makes no network requests.

```json
{
  "providers": {
    "ollama": {
      "base_url": "http://127.0.0.1:11434/v1",
      "models": [
        { "id": "qwen2.5-coder:7b", "name": "Qwen Coder 7B" }
      ]
    },
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "models": [
        { "id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4" },
        { "id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro" }
      ]
    }
  }
}
```

Set `"provider"` in settings, use the provider CLI option, or use `/provider` when you only want to change the endpoint. Optional `compat` flags cover picky gateways (for example `max_completion_tokens` versus `max_tokens`).

On Apple Silicon, when `base_url` is the managed local port, Open Max can optionally provision and serve MLX models via `/models`.

## Use

```sh
cd ~/code/my-app
openmax
```

On the first interactive run, inspect the project and accept the trust prompt. For headless or stdio use, make the same decision explicitly:

```sh
openmax --trust-project -p "summarize this repo"
openmax --trust-project --stdio
```

Trust is persisted for the exact canonical path in `~/.openmax/trust.json`. It authorizes the harness to run in that project; it does not sandbox project code.

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
| Mouse drag | Select transcript text |
| **y** or **Ctrl+Shift+C** | Copy selected text |
| **Esc** | Clear selection · close menu · cancel turn · return to composer |
| **Ctrl+C** twice | Quit |

| Slash command | Action |
| --- | --- |
| `/help` | Keybindings and commands |
| `/model` | Search configured providers and select a model |
| `/model <id>` | Set an exact model id on the active endpoint |
| `/copy` | Copy the latest assistant response |
| `/provider [name]` | List or switch providers |
| `/approvals auto\|ask\|readonly` | Mutating tool gates |
| `/new` · `/resume` | Fresh session · pick an earlier one |
| `/reload` | Force a re-freeze now (it also happens automatically when extension files change) |
| `/tools` · `/skills` · `/context` | Session tools, skills, token budget |
| `/<template> [args]` | Run a prompt template from `.agents/prompts/` |
| `/status` | Endpoint, cache, performance, privacy, and network details |
| `/quit` | Exit |

## Extend

With nothing installed, extensions cost zero tokens. Project paths win over global ones on name collision.

**Tools.** A TOML file in `.openmax/tools/` or `~/.openmax/tools/`. The harness runs `command`, writes JSON args to stdin, and returns stdout. These native processes inherit the host filesystem, environment, credentials, and network access of Open Max.

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

`mutating` is trusted metadata for scheduling and approval behavior. It is not a security boundary and does not restrict what the command can do.

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

**Hooks.** Optional process gates under `.openmax/hooks/` or `~/.openmax/hooks/`. `pre_tool_use` and `user_prompt_submit` can block (nonzero exit; the blocked prompt never reaches the model); `post_tool_use`, `session_start` (a session's first turn), `compaction` (context was pruned; receives the digest record), and `turn_end` (receives the stop reason, fires even on cancel) observe only. Each hook gets one JSON payload on stdin. Hooks never enter the model prompt and, like external tools and `bash`, run as native host processes with inherited filesystem, environment, credentials, and network access.

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

**Validation.** `openmax --check` parses tools, skills, templates, hooks, permissions, and `providers.json`, then prints per-file results with the reason anything would be ignored, fail closed, or fail at request time. It exits nonzero on errors. The agent is instructed to run it after writing extension files.

Tools and skills freeze per session for prompt-cache stability, and the harness re-freezes them automatically: at each turn start it fingerprints the extension files, and if anything changed it rebuilds the registry and prompt in place (one deliberate cache re-prefill, conversation kept, a `refrozen` event for clients). An unchanged disk costs nothing. `/reload` forces it immediately; `/new` starts clean. Hooks, permissions, and templates re-discover on every turn or invocation. Use `/tools`, `/skills`, and `/context` to inspect the frozen set and its cost.

## Native execution and privacy

The built-in file tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`) are confined to the project root by the harness. `bash`, external TOML tools, and hooks are native processes: they are not confined by that path check and inherit the host filesystem, environment, credentials, and network access of Open Max. Permissions, approvals, and `mutating` metadata control dispatch and user experience, not operating-system isolation.

Open Max itself does not phone home. Apart from native child processes, the harness only contacts:

1. The model endpoint in `base_url` (your choice).
2. Hugging Face, only when you download or serve a model through `/models`.

Sessions, settings, tools, and skills stay under `~/.openmax/` and your project directory. `/status` lists the destinations configured by the harness and detailed runtime information; it does not enforce or enumerate child-process network access. The persistent status line stays limited to model, context use, and approval mode so the transcript remains readable. External tools you install may open their own network connections.

## stdio protocol (`openmax-stdio/1`)

`openmax --stdio` speaks line-delimited JSON both ways, so any process that reads and writes JSONL (an editor plugin, an orchestrator, another openmax) can drive a full interactive session. This is the stable contract for custom frontends and interop adapters. Validate a stream against it with `openmax --check --stdio`, which reads JSONL on stdin, reports each line, and exits nonzero on any violation.

**Handshake.** The first stdout line is:

```json
{"type":"hello","proto":"openmax-stdio/1","protocol_version":1,"session_id":"...","version":"0.2.0","project":"/abs/path"}
```

`protocol_version` is an integer a client compares directly; `proto` carries the same major as a readable id. Any wire change bumps both.

**Commands (stdin), one JSON object per line:**

| Command | Fields | Effect |
| --- | --- | --- |
| `user` | `text` | Start a turn with the text |
| `approve` | `approval_id`, `approved` (bool) | Answer a pending approval |
| `cancel` | none | Cancel the running turn |
| `quit` | none | Drain the in-flight turn, then exit |

Unknown `cmd` values are protocol errors; extra fields on a known command are ignored; blank lines are skipped; EOF behaves like `quit`.

**Events (stdout).** Every event line carries the flattened `session_id`, then a `type` discriminator, then its fields. openmax emits keys in that order, but object key order is not significant: parse every line by field name.

| `type` | Fields |
| --- | --- |
| `token` | `text` |
| `thinking` | `text` |
| `message_done` | `text` |
| `budget` | `used_tokens`, `context_tokens` |
| `usage` | `prompt_tokens`, `completion_tokens`, `cached_tokens` (or null) |
| `tool_start` | `call_id`, `name`, `args` (object) |
| `tool_end` | `call_id`, `ok` (bool), `output` |
| `diff` | `call_id`, `path`, `diff`, `added`, `removed` |
| `approval_request` | `approval_id`, `name`, `summary`, `detail` |
| `approval_settled` | `approval_id`, `outcome` (`approved`, `declined`, `timed_out`, or `cancelled`) |
| `refrozen` | `tools`, `skills` |
| `done` | `stop_reason` |
| `error` | `message` |

Each turn ends with exactly one `done`, and `done` is the only guaranteed turn terminator: never block waiting for another event. On a normal turn a run of `token` deltas is terminated by one `message_done`, but a turn that hits a provider-stream error emits an `error` line and then `done` with no `message_done`. Bad input yields `{"type":"protocol_error","message":"..."}` and leaves the session unharmed. While a client is live, an `approval_request` is forwarded and openmax waits for an `approve`; after `quit` or EOF, pending and later approvals are declined so shutdown drains promptly. Example event line:

```json
{"session_id":"s1","type":"tool_start","call_id":"c1","name":"read_file","args":{"path":"a.rs"}}
```

## JSON-RPC bridge

`openmax-bridge` is a separate binary that exposes a running session over line-delimited JSON-RPC 2.0 on stdio, so a host that speaks JSON-RPC can drive openmax without handling its native JSONL envelope. It is a client of the `openmax-stdio/1` contract above: it spawns `openmax --stdio` (or `$OPENMAX_BIN`) and translates both directions. The harness core is unchanged.

Host methods (a request carries `id`; a notification omits it):

| Method | Params | Result and behavior |
| --- | --- | --- |
| `initialize` | none | `{ server, protocol_version, version, session_id }` |
| `prompt` | `text` | Streams `update` notifications, then resolves with `{ stop_reason }` at turn end |
| `cancel` | none | `{ ok: true }`, and cancels the running turn |
| `approve` | `approval_id`, `approved` | Answers a pending approval |
| `shutdown` | none | `{ ok: true }`, then the bridge drains and exits |

The bridge streams one `update` notification per child event, `{"jsonrpc":"2.0","method":"update","params":<event>}`, where `params` is the event exactly as the stdio protocol defines it (approval requests included; answer them with `approve`). A `prompt` request stays open until the turn's `done`, which resolves it. Closing stdin drains the in-flight turn, like the underlying protocol.

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
  '{"jsonrpc":"2.0","id":2,"method":"prompt","params":{"text":"list the crates"}}' \
  | openmax-bridge
```

## Development

```sh
cargo check
cargo test
cargo build --release -p open-max-tui
```

Set `OPENMAX_PERF=1` while running the TUI to log frame, transcript-layout, and selection-overlay timings.

Core logic is in `open-max-core` (`crates/core/src/agent.rs`). The TUI is `crates/tui/src/app.rs`.

## Status

Open Max is early software (v0.2.0). The agent loop, session persistence, extensibility, TUI, and GitHub Actions CI (test + release build + soft size gate) are in place, but there is no install script or published release channel yet. Expect rough edges. File an issue or send a PR if something breaks.

## License

MIT. See [LICENSE](LICENSE).
