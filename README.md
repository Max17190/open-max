# Open Max

**A lightweight, high-performance agent harness. Extremely configurable. Full control.**

Open Max is a single Rust binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. No desktop shell, no heavyweight runtime, no telemetry. Just a ~5 MB harness that stays out of the way.

You own the endpoints, the tools, the skills, and the context.

## Why Open Max

- **Barebones by default.** A small fixed tool set (file/shell tools plus a read-only `task` for context isolation), a short system prompt, and context budgeting that drops old tool output before it drops your task. With nothing installed beyond built-ins, the prompt stays minimal.
- **Extremely configurable.** External tools, skills, and project instructions live in local files. Shape the harness without forking it. Every extension's token cost is visible in `/context`.
- **Full control.** Settings, sessions, and secrets stay on your machine. Network traffic goes only to the model endpoint you configure (plus optional Hugging Face downloads when you ask). No silent upload, no telemetry.
- **Your model, your server.** Set one `base_url`, or name several endpoints in `providers.json` and switch with `/provider` (Ollama, LM Studio, vLLM, llama.cpp, cloud gateways, or a private proxy). Optional managed [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.
- **Visible by default.** Reads, greps, diffs, and shell commands stream as they happen. Writes and `bash` wait for approval unless you say otherwise.
- **A real session.** Fullscreen TUI: pinned header, conversation above the composer, wheel scrolling. Quit and your shell is exactly as you left it.

## Quick start

**Requirements:** [Rust](https://rustup.rs). For the optional managed MLX server on Apple Silicon, either [uv](https://docs.astral.sh/uv/) or Python 3.9+.

```sh
git clone https://github.com/Max17190/open-max.git
cd open-max
cargo install --path crates/tui --locked
```

Or run from source:

```sh
cargo run --release -p open-max-tui
```

Configure an endpoint (or use the defaults and the optional MLX path below), then inside a project:

```sh
cd ~/code/my-app
openmax
```

### Point at your backend

**Single endpoint.** Edit `~/.openmax/settings.json`:

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5-coder:7b",
  "api_key": null,
  "approval_mode": "ask"
}
```

Set `base_url` and `model` to match your model server (`/v1/chat/completions`). You can also set `api_key` to a literal or `$ENV_VAR`, or export `OPENMAX_API_KEY`.

**Several servers.** Name endpoints in `~/.openmax/providers.json`, then pick one with `"provider"` in settings, `--provider`, or `/provider`:

```json
{
  "providers": {
    "ollama": {
      "base_url": "http://127.0.0.1:11434/v1",
      "api_key": "ollama",
      "models": [
        { "id": "qwen2.5-coder:7b" },
        { "id": "llama3.1:8b", "context_tokens": 32768 }
      ]
    },
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "headers": {
        "HTTP-Referer": "https://github.com/Max17190/open-max",
        "X-Title": "Open Max"
      },
      "models": [
        { "id": "anthropic/claude-sonnet-4" },
        { "id": "openai/gpt-4.1-mini" }
      ]
    },
    "proxy": {
      "base_url": "https://llm.example.com/v1",
      "api_key": "$CORP_LLM_KEY",
      "compat": {
        "send_stream_options": false,
        "use_max_completion_tokens": true
      },
      "models": [{ "id": "default" }]
    }
  }
}
```

Then in `settings.json` set `"provider": "ollama"` (or switch later with `/provider openrouter`). Credential resolution per provider: `api_key` (literal or `$ENV`; use `$$` for a literal leading `$`) → `api_key_env` → settings `api_key` → `OPENMAX_API_KEY`. An unknown `provider` name fails the request instead of falling back to flat `base_url`.

Resume the latest session in the current directory:

```sh
openmax --continue
# or
openmax -c
```

Pick a provider and model for the run:

```sh
openmax --provider ollama --model qwen2.5-coder:7b
openmax --model your-model-id
```

### Headless (print mode)

Drive the same agent core without taking over the terminal. Useful for scripting and CI:

```sh
openmax -p "summarize the top-level layout of this repo"
openmax -p --json "list the public modules in crates/core"
```

Text tokens go to stdout; tool progress goes to stderr. With `--json`, each `AgentEvent` envelope is one JSON line on stdout. Mutating tools still honor `approval_mode`: for unattended runs set `"approval_mode": "auto"` in settings (otherwise approvals are declined so the process never hangs).

### Optional: first run with managed MLX (Apple Silicon)

When `base_url` is the managed local port (`http://127.0.0.1:8989/v1`, the default), Open Max can provision and serve MLX models:

1. Type `/models` to open the model panel.
2. Press **`u`** once to install `mlx-lm` into `~/.openmax/mlx-venv` (watch progress with `/logs`).
3. Highlight a model and press **`Enter`**: downloads weights if needed, or starts the server if already cached.
4. Describe a task. The agent reads your codebase, proposes edits, and asks before mutating files or running shell commands.

When `base_url` is not the managed MLX port, Open Max talks to your endpoint directly and skips the "serve a model first" gate.

## Commands

| Input | Action |
| --- | --- |
| **Enter** | Send message |
| **Enter** while the agent works | Queue the message; it goes out when the turn ends |
| **Shift+Enter** / **Alt+Enter** | New line in the composer |
| **/** at the start of a message | Command menu; **Tab** or **Enter** completes |
| **@** anywhere | Fuzzy-search project files and mention one |
| **Tab** | Toggle focus between composer and conversation |
| **↑↓** / **j k** (history focused) | Select a block · **Enter** fold tool · **y** copy |
| **[** / **]** (history focused) | Jump to previous or next user turn (also **Shift+↑↓** when the terminal reports it) |
| **g** / **G** (history focused) | Top of scrollback · follow latest |
| **Esc** | Close menu · cancel turn · follow latest · return to composer |
| **Mouse wheel** / **PgUp PgDn** | Scroll the conversation |
| **Ctrl+R** | Search prompt history |
| **Ctrl+O** / **o** | Expand last tool block |
| **Ctrl+T** | Toggle model thinking stream |
| **Ctrl+A / Ctrl+E / Ctrl+K / Ctrl+U / Ctrl+W** | Line editing in the composer |
| **Ctrl+C** twice | Quit (model server keeps running if you started one) |

Cancelling a turn hands any queued messages back to the composer, so nothing typed mid-turn is lost.

| Slash command | Action |
| --- | --- |
| `/help` | Show keybindings and commands |
| `/models` | Download, serve, and manage local MLX models (Apple Silicon) |
| `/model <id>` | Set the active model id |
| `/provider [name]` | List saved servers, or switch to one from `providers.json` |
| `/approvals auto\|ask\|readonly` | Control mutating tool gates |
| `/new` | Start a fresh session |
| `/resume` | Pick an earlier session in this project |
| `/tools` | List tools frozen for this session (or preview the next) |
| `/skills` | List skills frozen for this session (or preview the next) |
| `/theme dark\|light\|mono\|catppuccin` | Switch appearance (respects `NO_COLOR`) |
| `/context` | Prompt token costs per component, cache hits, budget |
| `/status` | Session, endpoint, and network destinations |
| `/logs` | Tail recent MLX server logs |
| `/quit` | Exit |

Inside `/models`: **↑/↓** or **j/k** to move, **Enter** to download or serve, **u** set up MLX, **s** stop server, **x** delete cached weights.

## Recommended local models (MLX)

Curated MLX community quantizations that work well as coding agents. RAM fit indicators appear in `/models` based on your machine. Any other Hugging Face repo id works with `/model <repo>` when using the managed server.

| Model | Approx. RAM | Notes |
| --- | --- | --- |
| Qwen3.6 35B A3B (4-bit) | ~19 GB | MoE flagship; fast agentic coding |
| Gemma 4 31B (4-bit) | ~19 GB | Flagship dense Gemma 4 instruct |
| Qwen3 Coder 30B A3B (4-bit) | ~18 GB | Agentic MoE coder; reliable tool use |
| Qwen3.6 27B (4-bit) | ~16 GB | Strong dense coder at consumer scale |
| Gemma 4 26B A4B (4-bit) | ~16 GB | MoE Gemma 4; 4B active params |
| gpt-oss 20B (MXFP4) | ~12 GB | Solid tool calling; adjustable reasoning |
| Gemma 4 12B (QAT 4-bit) | ~11 GB | Unified 12B; QAT holds quality at 4 bits |
| Gemma 4 E4B (4-bit) | ~5.5 GB | Efficient small Gemma 4 |
| Qwen2.5 Coder 7B (4-bit) | ~4.5 GB | Light starter model |

## Extending Open Max

Open Max stays barebones on purpose; you extend it per workflow. With nothing installed, extensibility costs zero tokens by default.

**External tools.** Drop a TOML file in `.openmax/tools/` (project) or `~/.openmax/tools/` (global; project wins on name collision). Any language works: the harness runs `command`, writes the call's JSON arguments to stdin, and returns stdout to the model, with the same output cap and spill-to-file behavior as `bash`.

```toml
# .openmax/tools/todo_scan.toml
name = "todo_scan"
description = "List TODO/FIXME comments with file and line"   # keep it short: it rides in every prompt
command = "./scripts/todo-scan.sh"
timeout_secs = 30
mutating = false          # true routes the tool through approvals

[params]
type = "object"
[params.properties.path]
type = "string"
description = "Directory to scan"
```

**Skills.** A directory with a `SKILL.md` under `.agents/skills/` (project) or `~/.openmax/skills/` (global). Frontmatter `name:` and `description:` are the only lines that live in the prompt (~15 tokens per skill); the model reads the full file on demand when a task matches. This is how you add large, rarely used capability without taxing every request.

```
.agents/skills/release/SKILL.md
---
name: release
description: How to cut a release of this project
---
Full instructions, checklists, commands...
```

**Hooks.** Optional process lifecycle gates under `.openmax/hooks/` (project) or `~/.openmax/hooks/` (global). Hooks never enter the model prompt: empty discovery is free. Each TOML file defines one event and a command. The harness writes a JSON payload to stdin (`event`, `session_id`, `tool`, `args`, `cwd`, and for post hooks `tool_ok`).

| Event | Behavior |
| --- | --- |
| `pre_tool_use` | Non-zero exit blocks the tool; stdout (or stderr) becomes the error returned to the model |
| `post_tool_use` | Observe only; failures are ignored |

Hooks also apply to tools run inside a `task` subagent, so lifecycle policy cannot be bypassed by delegation.

```toml
# .openmax/hooks/block_rm.toml
event = "pre_tool_use"
command = "./scripts/block-rm.sh"
tool = "bash"          # optional filter; omit to run for every tool
timeout_secs = 5
```

**Permissions.** Optional declarative rules in `.openmax/permissions.toml` (project) or `~/.openmax/permissions.toml` (global). Missing files change nothing. Project rules are evaluated before global; the first match wins. Rules refine ask/auto without forking: deny dangerous calls, auto-allow safe ones, or force an approval prompt on specific tools. Evaluation order per tool call is hooks pre → permissions → `approval_mode` → execute → hooks post. `allow` still cannot override `readonly` for mutating tools. Rules also apply inside a `task` subagent.

```toml
# .openmax/permissions.toml
[[rules]]
effect = "deny"          # allow | deny | ask
tool = "bash"
arg_regex = "rm\\s+-rf"  # optional; omit to match any args for the tool

[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^cargo (test|check|build)"

[[rules]]
effect = "ask"
tool = "write_file"
```

**Freeze semantics.** Tools and skills are discovered once, at session creation, and frozen for the session. The serialized schemas are part of the prompt prefix the server's KV cache keys on, so they must stay byte-stable. Config changes apply to the next `/new` session; `/tools`, `/skills`, and `/context` show the frozen set and its token cost. Hooks and permissions are re-discovered each turn and do not affect schemas.

**Why no MCP?** A typical MCP server dumps 10k+ tokens of tool descriptions into every request, most of a small model's whole window. External tools plus skills give the same reach with per-call processes and on-demand documentation: write a CLI, give it a README (or a skill), and let the model read it when needed.

## Privacy

Open Max does not phone home. The only network destinations are:

1. The model endpoint in `base_url` (your choice).
2. Hugging Face, only when you explicitly download or serve a model through `/models`.

Sessions, settings, tools, and skills stay under `~/.openmax/` and your project directory. Inside a session, `/status` lists those destinations; the status bar shows `no telemetry` when idle. External tools you install may still open their own network connections.

## Architecture

```
crates/
  core/                     UI-free agent harness (+ optional MLX/Hugging Face helpers)
    agent.rs                  Turn loop: stream → tools → approvals → repeat
    client.rs                 OpenAI-compatible streaming client (SSE + JSON fallback)
    providers.rs              Named providers.json registry and endpoint resolution
    tools.rs                  list_dir · read_file · write_file · edit_file · glob · grep · bash · task
    registry.rs               Session-frozen tool registry: built-ins + external TOML tools
    hooks.rs                  Optional pre/post tool process hooks
    permissions.rs            Optional permissions.toml allow/deny/ask rules
    skills.rs                 SKILL.md discovery; name+description only in the prompt
    prompt.rs                 Short system prompt tuned for coding agents
    fallback.rs               Parses tool markup when the server omits native tool_calls
    mlx.rs                    Optional mlx-lm venv provisioning and server lifecycle
    hf.rs                     Hub cache inspection, downloads, and sizing
    sessions.rs               Persisted sessions + compaction records under ~/.openmax/sessions/
  tui/                      ratatui + crossterm terminal frontend (`openmax` binary)
    app.rs                    Event loop, slash commands, approvals, model panel
    ui/                       Markdown rendering, tool cards, transcript layout
```

Design choices worth knowing:

- **The harness is the product.** Fewer, stricter tools keep small models reliable. Tool results stream as events so nothing happens off screen.
- **Safety by default.** Tools are sandboxed to the project root. `write_file`, `edit_file`, and `bash` require approval in the default `ask` mode; `readonly` blocks mutating tools entirely.
- **Context budgeting, not magic.** Old tool outputs are truncated first, then the oldest exchanges are dropped, leaving a digest of tools used, files touched, and earlier goals. The system prompt and your original request survive. Each exchange drop is also appended to `~/.openmax/sessions/<id>.compaction.jsonl` for recoverability.
- **Edits that land.** `edit_file` matches exactly first, then retries with whitespace-normalized matching (re-indented to the file), and on a miss points the model at the closest line. Consecutive read-only tool calls run concurrently; Esc kills in-flight commands, not just the stream.
- **Local serve stays decoupled.** The harness talks to your model server over HTTP. The MLX sidecar is optional convenience on Apple Silicon, not the identity of the product.

State lives in `~/.openmax/` (settings, sessions, optional MLX venv, server metadata).

## Development

```sh
cargo check
cargo test
cargo build --release -p open-max-tui
```

The release profile uses thin LTO and symbol stripping to keep the binary small. Core logic lives in `open-max-core`; start with `crates/core/src/agent.rs` and `crates/tui/src/app.rs`.

## Status

Open Max is early software (v0.2.0): the agent loop, session persistence, extensibility, and TUI are in place, but there is no install script, CI, or published release channel yet. Expect rough edges. File an issue or send a PR if something breaks.

## License

MIT. See [LICENSE](LICENSE).
