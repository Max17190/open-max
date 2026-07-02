# Open Max

**A minimal terminal harness for local coding models on Apple Silicon.**

Open Max is a single Rust binary that runs a focused agent loop in your project directory, streams every tool call to the terminal, and manages an on device [MLX](https://github.com/ml-explore/mlx) server when you want one. No desktop shell, no cloud account, no heavyweight runtime. Just a ~5 MB harness that stays out of the way so your Mac can spend its RAM on the model.

## Why Open Max

- **Built for MLX MacBooks.** Provisions a private `mlx-lm` environment, downloads Hugging Face weights, and serves models over a local OpenAI compatible API.
- **Terminal native.** Finished output lives in native scrollback. A small live viewport at the bottom handles streaming, approvals, and the composer.
- **Small on purpose.** Seven strict tools, a short system prompt tuned for 7B to 30B models, and context budgeting that drops old tool output before it drops your task.
- **Visible by default.** Reads, greps, diffs, and shell commands stream as they happen. Writes and `bash` wait for approval unless you say otherwise.
- **Bring any backend.** Point `~/.openmax/settings.json` at Ollama, LM Studio, vLLM, llama.cpp, or any other OpenAI compatible endpoint.

## Quick start

**Requirements:** macOS on Apple Silicon, [Rust](https://rustup.rs), and either [uv](https://docs.astral.sh/uv/) or Python 3.9+ (for the managed MLX server).

```sh
git clone https://github.com/maxloffgren/open_max.git
cd open_max
cargo install --path crates/tui --locked
```

Or run from source:

```sh
cargo run --release -p open-max-tui
```

Then, inside a project directory:

```sh
cd ~/code/my-app
openmax
```

### First run with MLX

1. Type `/models` to open the model panel.
2. Press **`u`** once to install `mlx-lm` into `~/.openmax/mlx-venv` (watch progress with `/logs`).
3. Select a model, press **`d`** to download weights if needed, then **`Enter`** to start the server.
4. Describe a task. The agent reads your codebase, proposes edits, and asks before mutating files or running shell commands.

Resume the latest session in the current directory:

```sh
openmax --continue
# or
openmax -c
```

Pick a model for the run:

```sh
openmax --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

## What it looks like

```
◆ open max v0.2.0 · mlx-community/Qwen2.5-Coder-7B-Instruct-4bit · my-app · /help

> add input validation to the signup form

· read_file  src/forms/signup.rs
· grep       validate
· edit_file  src/forms/signup.rs        +12 −3
  approve? write_file src/forms/signup.rs  [y/n/a]

Added email and password checks; ran `cargo test signup`. 4 passed.

  model · serving Qwen2.5 Coder 7B · context 38% · esc cancels
> _
```

Markdown in assistant replies is rendered inline. Tool cards show summaries; press **Ctrl+O** to expand the last tool output.

## Commands

| Input | Action |
| --- | --- |
| **Enter** | Send message |
| **Shift+Enter** / **Alt+Enter** | New line in the composer |
| **Esc** | Cancel the running turn |
| **Ctrl+O** | Expand last tool output |
| **Ctrl+T** | Toggle model thinking stream |
| **Ctrl+C** twice | Quit (model server keeps running) |

| Slash command | Action |
| --- | --- |
| `/help` | Show keybindings and commands |
| `/models` | Download, serve, and manage local models |
| `/model <repo>` | Set the active Hugging Face repo id |
| `/approvals auto\|ask\|readonly` | Control mutating tool gates |
| `/new` | Start a fresh session |
| `/status` | Session, endpoint, and server state |
| `/logs` | Tail recent MLX server logs |
| `/quit` | Exit |

Inside `/models`: **↑/↓** or **j/k** to move, **Enter** to serve, **d** download, **u** set up MLX, **s** stop server, **x** delete cached weights.

## Recommended models

Curated MLX community quantizations that work well as coding agents. RAM fit indicators appear in `/models` based on your machine.

| Model | Approx. RAM | Notes |
| --- | --- | --- |
| Qwen3.6 35B A3B (4-bit) | ~19 GB | MoE flagship; fast agentic coding |
| Qwen3.6 27B (4-bit) | ~16 GB | Strong dense coder at consumer scale |
| Qwen3 Coder 30B A3B (4-bit) | ~18 GB | Agentic MoE coder; reliable tool use |
| gpt-oss 20B (MXFP4) | ~12 GB | Solid tool calling; adjustable reasoning |
| Qwen2.5 Coder 7B (4-bit) | ~4.5 GB | Light starter model |

Any other Hugging Face repo id works with `/model <repo>`.

## Custom endpoints

By default Open Max talks to the managed server at `http://127.0.0.1:8989/v1`. To use another backend, edit `~/.openmax/settings.json`:

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5-coder:7b",
  "approval_mode": "ask"
}
```

Set `base_url` and `model` to match your provider. When `base_url` is not the managed MLX port, Open Max skips the serve a model first gate and talks to your endpoint directly.

## Architecture

```
crates/
  core/                     UI free agent harness + MLX/Hugging Face helpers
    agent.rs                  Turn loop: stream → tools → approvals → repeat
    client.rs                 OpenAI compatible streaming client (SSE + JSON fallback)
    tools.rs                  list_dir · read_file · write_file · edit_file · glob · grep · bash
    prompt.rs                 Short system prompt tuned for small local models
    fallback.rs               Parses tool markup when the server omits native tool_calls
    mlx.rs                    mlx-lm venv provisioning and server lifecycle
    hf.rs                     Hub cache inspection, downloads, and sizing
    sessions.rs               Persisted sessions under ~/.openmax/sessions/
  tui/                      ratatui + crossterm terminal frontend (`openmax` binary)
    app.rs                    Event loop, slash commands, approvals, model panel
    ui/                       Markdown rendering, tool cards, transcript layout
```

Design choices worth knowing:

- **The harness is the product.** Fewer, stricter tools measurably help small local models. Tool results stream as events so nothing happens off screen.
- **Safety by default.** Tools are sandboxed to the project root. `write_file`, `edit_file`, and `bash` require approval in the default `ask` mode; `readonly` blocks mutating tools entirely.
- **Context budgeting, not magic.** Old tool outputs are truncated first, then the oldest exchanges are dropped. The system prompt and your original request survive.
- **MLX stays decoupled.** The harness talks to `mlx_lm.server` over plain HTTP like any other backend. The sidecar manager makes running it a one step affair inside `/models`.

State lives in `~/.openmax/` (settings, sessions, MLX venv, server metadata).

## Development

```sh
cargo check
cargo test
cargo build --release -p open-max-tui
```

The release profile uses thin LTO and symbol stripping to keep the binary small. Core logic lives in `open-max-core`; start with `crates/core/src/agent.rs` and `crates/tui/src/app.rs`.

## Status

Open Max is early software (v0.2.0): the agent loop, MLX integration, session persistence, and TUI are in place, but there is no install script, CI, or published release channel yet. Expect rough edges. File an issue or send a PR if something breaks.

## License

MIT. See [LICENSE](LICENSE).
