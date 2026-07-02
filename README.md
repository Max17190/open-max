# Open Max

A sleek, minimal, **local-first coding agent** for macOS. Point it at your projects, run an open-weights model on your own machine with [MLX](https://github.com/ml-explore/mlx), and delegate coding tasks — no cloud required.

- **Codex-style UI** — clean multi-project sidebar, a single focused task thread, inline tool calls and diff review. No editor chrome, no clutter.
- **MLX native** — one click installs `mlx-lm` into a private environment and serves models like Qwen3 Coder or gpt-oss straight from HuggingFace, fully on-device on Apple Silicon.
- **Bring any backend** — anything with an OpenAI-compatible API works too: Ollama, LM Studio, vLLM, llama.cpp.
- **Minimal, Cursor-quality harness** — a purpose-built agent loop in Rust with eight sharp tools, streaming output, context budgeting, and approval gates. Prompts are tuned for small local models.
- **Lightweight** — Tauri 2 shell, hand-rolled CSS, no heavyweight frameworks. The app stays out of your way and out of your RAM.

## Quickstart

Prereqs: macOS on Apple Silicon, [Rust](https://rustup.rs), Node 20+, Python 3.9+.

```sh
npm install
npm run tauri dev
```

Then, in the app:

1. **Add a project** — `+` in the sidebar, pick a folder.
2. **Set up a model** — click the model pill → *Local MLX* → *Install mlx-lm* → pick a model → *Start server*. The first start downloads the weights (4–18 GB depending on the model).
3. **Delegate** — describe a task. The agent reads, greps, edits, and runs commands in your project, streaming everything it does. Edits and shell commands wait for your approval (configurable).

### Recommended models

| Model | RAM | Why |
| --- | --- | --- |
| Qwen3 Coder 30B (MoE, 4-bit) | ~18 GB | Strongest local coding agent; fast (3.3B active params) |
| gpt-oss 20B (MoE) | ~12 GB | Very reliable tool calling; reasoning model |
| Qwen2.5 Coder 14B (4-bit) | ~9 GB | Solid mid-size coder |
| Qwen2.5 Coder 7B (4-bit) | ~4.5 GB | Light and fast starter |

Using Ollama or LM Studio instead? Settings → *Custom endpoint* → point at `http://127.0.0.1:11434/v1` (or `:1234/v1`) and name the model.

## Architecture

```
src/                      React + TypeScript UI (Vite, zustand, hand-rolled CSS)
  components/             Sidebar · FileTree · Chat · ToolCard · Composer · FileViewer · Settings
  store.ts                App state + agent/MLX event streams
src-tauri/src/
  harness/                The agent
    agent.rs              Turn loop: stream → tool calls → approvals → results → repeat
    client.rs             Streaming client for any OpenAI-compatible endpoint (SSE + JSON fallback)
    tools.rs              read_file · write_file · edit_file · list_dir · glob · grep · bash
    prompt.rs             System prompt tuned for small local models
  mlx.rs                  MLX sidecar: venv provisioning, mlx_lm.server lifecycle, health checks
  projects.rs             Project registry + gitignore-aware file tree
  settings.rs             Persisted settings (backend, approvals, context budget)
```

Design decisions worth knowing:

- **The harness is the product.** The loop, tool schemas, and prompt are deliberately small — fewer, stricter tools measurably help 7B–30B models. Tool results stream to the UI as events (`agent_event`), so every read, grep, and diff is visible as it happens.
- **Safety by default.** Tools are sandboxed to the project root, `write`/`edit`/`bash` require approval in the default mode, and there's a read-only mode for pure exploration.
- **Context budgeting, not magic.** Old tool outputs are truncated first, then the oldest exchanges are dropped — the system prompt and your original request always survive.
- **MLX stays decoupled.** The app talks to `mlx_lm.server` over plain HTTP like any other backend; the sidecar manager just makes running it a one-click affair.

## Contributing

PRs welcome. The codebase is intentionally small enough to read in an afternoon — start with `src-tauri/src/harness/agent.rs` and `src/store.ts`.

## License

MIT
