# Open Max

**A barebones, high-performance agent harness. Extremely configurable. Full control.**

Open Max is a single Rust binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. Point it at any OpenAI-compatible endpoint (cloud or local). No desktop shell, no heavyweight runtime, no telemetry. Just a ~5 MB harness that stays out of the way.

You own the endpoint, the tools, the skills, and the context.

## Why Open Max

- **Barebones by default.** A small fixed tool set (file/shell tools plus a read-only `task` for context isolation), a short system prompt, and context budgeting that drops old tool output before it drops your task. With nothing installed beyond built-ins, the prompt stays minimal.
- **Extremely configurable.** External tools, skills, and project instructions live in local files. Shape the harness without forking it. Every extension's token cost is visible in `/context`.
- **Full control.** Settings, sessions, and secrets stay on your machine. Network traffic goes only to the model endpoint you configure (plus optional Hugging Face downloads when you ask). No silent upload, no telemetry.
- **Any OpenAI-compatible backend.** Ollama, LM Studio, vLLM, llama.cpp, cloud gateways, or a managed on-device [MLX](https://github.com/ml-explore/mlx) server on Apple Silicon when you want one.
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

Edit `~/.openmax/settings.json`:

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5-coder:7b",
  "approval_mode": "ask"
}
```

Set `base_url` and `model` to match your provider. Any OpenAI-compatible `/v1/chat/completions` endpoint works.

Resume the latest session in the current directory:

```sh
openmax --continue
# or
openmax -c
```

Pick a model for the run:

```sh
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
| **Shift+Enter** / **Alt+Enter** | New line in the composer |
| **Esc** | Cancel the running turn, or jump to the latest output |
| **Mouse wheel** / **PgUp PgDn** | Scroll the conversation |
| **Ctrl+O** | Expand last tool output |
| **Ctrl+T** | Toggle model thinking stream |
| **Ctrl+A / Ctrl+E / Ctrl+K / Ctrl+U / Ctrl+W** | Line editing in the composer |
| **Ctrl+C** twice | Quit (model server keeps running if you started one) |

| Slash command | Action |
| --- | --- |
| `/help` | Show keybindings and commands |
| `/models` | Download, serve, and manage local MLX models (Apple Silicon) |
| `/model <id>` | Set the active model id |
| `/approvals auto\|ask\|readonly` | Control mutating tool gates |
| `/new` | Start a fresh session |
| `/context` | Prompt token costs per component, cache hits, budget |
| `/status` | Session, endpoint, and server state |
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

**Freeze semantics.** Tools and skills are discovered once, at session creation, and frozen for the session. The serialized schemas are part of the prompt prefix the server's KV cache keys on, so they must stay byte-stable. Config changes apply to the next `/new` session; `/context` tells you which state you are looking at.

**Why no MCP?** A typical MCP server dumps 10k+ tokens of tool descriptions into every request, most of a small model's whole window. External tools plus skills give the same reach with per-call processes and on-demand documentation: write a CLI, give it a README (or a skill), and let the model read it when needed.

## Privacy

Open Max does not phone home. The only network destinations are:

1. The model endpoint in `base_url` (your choice).
2. Hugging Face, only when you explicitly download or serve a model through `/models`.

Sessions, settings, tools, and skills stay under `~/.openmax/` and your project directory.

## Architecture

```
crates/
  core/                     UI-free agent harness (+ optional MLX/Hugging Face helpers)
    agent.rs                  Turn loop: stream → tools → approvals → repeat
    client.rs                 OpenAI-compatible streaming client (SSE + JSON fallback)
    tools.rs                  list_dir · read_file · write_file · edit_file · glob · grep · bash · task
    registry.rs               Session-frozen tool registry: built-ins + external TOML tools
    skills.rs                 SKILL.md discovery; name+description only in the prompt
    prompt.rs                 Short system prompt tuned for coding agents
    fallback.rs               Parses tool markup when the server omits native tool_calls
    mlx.rs                    Optional mlx-lm venv provisioning and server lifecycle
    hf.rs                     Hub cache inspection, downloads, and sizing
    sessions.rs               Persisted sessions under ~/.openmax/sessions/
  tui/                      ratatui + crossterm terminal frontend (`openmax` binary)
    app.rs                    Event loop, slash commands, approvals, model panel
    ui/                       Markdown rendering, tool cards, transcript layout
```

Design choices worth knowing:

- **The harness is the product.** Fewer, stricter tools keep small models reliable. Tool results stream as events so nothing happens off screen.
- **Safety by default.** Tools are sandboxed to the project root. `write_file`, `edit_file`, and `bash` require approval in the default `ask` mode; `readonly` blocks mutating tools entirely.
- **Context budgeting, not magic.** Old tool outputs are truncated first, then the oldest exchanges are dropped, leaving a digest of which tools ran and which files were touched. The system prompt and your original request survive.
- **Edits that land.** `edit_file` matches exactly first, then retries with whitespace-normalized matching (re-indented to the file), and on a miss points the model at the closest line. Consecutive read-only tool calls run concurrently; Esc kills in-flight commands, not just the stream.
- **Local serve stays decoupled.** The harness talks to any OpenAI-compatible HTTP API. The MLX sidecar is optional convenience on Apple Silicon, not the identity of the product.

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
