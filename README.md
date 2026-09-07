<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openmax-wordmark-dark.svg">
    <img src="docs/assets/openmax-wordmark.svg" alt="Open Max">
  </picture>
</p>

<p align="center"><strong>A self-extending Rust agent harness for coding in the terminal.</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/language-Rust-orange.svg" alt="Rust"></a>
</p>

Open Max is a single binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. Point it at the model server you choose: local, cloud, or a private proxy. No desktop shell, no heavyweight runtime, no telemetry.

The endpoint must return structured API `tool_calls` to use tools. For local servers, configure the model's tool parser and chat template. Call markup in assistant text is displayed as text and never executed. Responses are limited to 16 MiB and 128 tool calls.


You own the endpoints, the tools, the skills, and the context.

## Features

- **Small by default.** Seven built-in tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`) and a short system prompt. Old tool output is dropped before your task is, and dropped context is summarized by your own model into a compact note (heuristic digest as fallback) whose address points at the lossless archive of everything dropped — compaction is a bounded view over a record you can always read back.
- **Your model, your server.** One `base_url`, or several named endpoints in `providers.json` switched with `/model`. Works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud gateways (OpenRouter and similar), and private proxies.
- **Trust before execution.** An exact canonical project root must be trusted before any agent turn or project behavior starts. Interactive use asks once; headless and stdio runs fail closed until explicitly started with `--trust-project`.
- **Approvals by default.** `write_file`, `edit_file`, and `bash` wait for approval in `ask` mode. Use `auto` for unattended runs or `readonly` to block mutating tools. Approvals and permissions decide whether Open Max dispatches a tool call; they are not OS isolation.
- **File based extensions.** Drop TOML tools, `SKILL.md` skills, prompt templates, and process hooks under project or home config. No fork required. The agent writes them itself and the harness re-freezes as soon as a mutating call lands, so a tool the agent writes is a tool the agent uses on its very next step.
- **File based memory.** One durable fact per file in `.openmax/memory/`, written by the agent, surfaced as an index line in future sessions, ranked by recency and frequency of observed use. Old entries fade from the index and remain searchable on disk until the user or agent deletes them. No database, no daemon, no embeddings; zero prompt cost when empty.
- **Recall over everything kept.** `openmax --recall "<query>"` searches this project's past sessions, compaction archives, digests, and memories in one bounded streaming pass — BM25 relevance fused with the same recency law, ranked excerpts, each citing the file with the full record. No index to build or maintain: the stores on disk stay the single source of truth.
- **Visible work.** Reads, greps, diffs, and shell commands stream as they happen in a fullscreen TUI. Headless print mode for scripts and CI.
- **Local sessions.** Conversation state lives under `~/.openmax/`. The harness contacts only the model endpoint you configure.

## Design

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
| Shortcuts or a completely different UI | Prompt templates, or a custom frontend speaking `openmax-stdio/5` |

These are deliberate boundaries, not placeholders for hidden orchestration products. Open Max does not carry an MCP host, nested-agent scheduler, plan mode, background-job product, built-in TODO database, user-keybinding engine, pluggable compactor, or TUI plugin ABI. The agent composes those richer workflows from the same host tools a developer can inspect, edit, test, and remove.

## Install

Homebrew, on macOS or Linux:

```sh
brew install Max17190/tap/openmax
```

Or without Homebrew:

```sh
curl -fsSL https://useopenmax.dev/install.sh | sh
```

Or with bun or npm, if you already have a JavaScript runtime:

```sh
bun install -g openmax-cli    # or: npm install -g openmax-cli
```

The command is `openmax` either way; only the package name carries the `-cli`,
because `openmax` on npm belongs to an unrelated 2015 package. This path fetches
the same native binary, so it is the one install that needs Node or Bun present.
Bun blocks install scripts by default, which is fine here: the binary is fetched
on first run instead, once, and cached.

Either way there is no Rust toolchain and no build. The script picks the right
prebuilt binary for your machine, checks it against the published SHA-256
(skipped, with a message, on a system with no `sha256sum`), and puts `openmax`
in `~/.openmax/bin` next to the settings and sessions the harness already keeps
there. It adds that directory to your `PATH` through your shell profile, so
open a new shell (or `source ~/.openmax/bin/env`) and run `openmax`.

macOS and Linux, x86_64 and arm64. Linux gets a glibc build, or a static musl
build automatically when glibc is older than 2.31, so it also runs on Alpine and
inside slim containers. Windows is not supported.

Pass `--no-modify-path` to leave your `PATH` alone, or set
`OPENMAX_INSTALL_DIR` to install somewhere else:

```sh
curl -fsSL https://useopenmax.dev/install.sh | sh -s -- --no-modify-path
```

`useopenmax.dev/install.sh` redirects to the current release's installer, so it
follows new releases without the link changing.

Every [release](https://github.com/Max17190/open-max/releases) also carries the
plain tarballs and a `sha256.sum`, so you can download and verify by hand
instead of piping a script into a shell.

**From source** instead, which needs [Rust](https://rustup.rs):

```sh
git clone https://github.com/Max17190/open-max.git
cd open-max
cargo install --path crates/tui --locked
```

Or run it in place with `cargo run --release -p openmax`.

## Configure

Edit `~/.openmax/settings.json`:

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5-coder:7b",
  "api_key": null,
  "approval_mode": "ask",
  "context_tokens": 32768,
  "max_parallel_tools": 4
}
```

`base_url` is the root of your model's HTTP API (the harness calls `chat/completions` on it). Set `model` to the id that server expects. Set `api_key` to a literal or `$ENV_VAR`, or export `OPENMAX_API_KEY`. Set `context_tokens` to the context window the server actually serves for that model; nothing is queried, and a guessed window is wrong in one direction or the other. There is no default endpoint, model, or window: until all three are configured (here or through a named provider), Open Max refuses to start a turn with an error that says exactly what to set.

`/approvals auto` saves the execution choice for the current trusted project.
Shift+Tab and the approval card's Auto for project choice save the same setting.
Auto covers extension creation and repair without further confirmation;
validation, deny rules, and execution reports remain active. See
[approval modes](docs/configuration.md#approvals).

A settings file that exists but does not parse, uses an unknown key, or sets an unrecognized `approval_mode` is a startup error (fail closed): Open Max exits with the parse reason instead of silently reverting your endpoint and approval policy to defaults.

Named endpoints in `providers.json` and `compat` flags for picky gateways are covered in [configuration](docs/configuration.md).

## Use

```sh
cd ~/code/my-app
openmax
```

On the first interactive run, inspect the project and accept the trust prompt. Headless and stdio runs make the same decision explicitly:

```sh
openmax --continue                    # resume latest session here
openmax --trust-project -p "summarize this repo"
openmax --trust-project --stdio
```

In print mode, text goes to stdout and tool progress to stderr; with `--json`, each `AgentEvent` is one JSON line. Press **/** for slash commands and **@** to mention a project file. Full flags, keybindings, and commands are in [usage](docs/usage.md).

## Extend

With nothing installed, extensions cost zero tokens. Project paths win over global ones on name collision. Each surface is a file the agent can write, `openmax --check` can validate, and the next turn picks up automatically.

| Surface | Location | What it is |
| --- | --- | --- |
| **Tools** | `.openmax/tools/*.toml` | Schema-described native processes; args on stdin, result on stdout |
| **Skills** | `.agents/skills/*/SKILL.md` | On-demand workflow packages; only name and description cost prompt tokens |
| **Prompt templates** | `.agents/prompts/*.md` | User-invoked messages; the file stem becomes a slash command |
| **Hooks** | `.openmax/hooks/` | Process gates on a fixed set of lifecycle events; two of them can block |
| **Permissions** | `.openmax/permissions.toml` | Allow, deny, or ask rules matched against tool arguments; fail closed |

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

Unknown keys are rejected everywhere, so a misspelled field surfaces in `openmax --check` rather than silently changing behavior. `openmax --spec <surface>` prints the complete authoring contract for any one of them, and the frozen prompt carries only a one-line pointer to it, so that contract costs zero tokens until the agent reads it.

File formats, hook events, permission rule syntax (including the in-session repair path for a broken permissions file), self-description, and the freezing model are documented in [extending](docs/extending.md).

## Native execution and privacy

The built-in file tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`) are confined to the project root by the harness. `bash`, external TOML tools, and hooks are native processes: they are not confined by that path check and inherit the host filesystem, environment, credentials, and network access of Open Max. Permissions, approvals, and `mutating` metadata control dispatch and user experience, not operating-system isolation.

Open Max itself does not phone home. Apart from native child processes, the harness contacts only the model endpoint you configure.

Sessions, settings, tools, and skills stay under `~/.openmax/` and your project directory. `/status` lists the destinations configured by the harness and detailed runtime information; it does not enforce or enumerate child-process network access, and external tools you install may open their own network connections.

## Documentation

- [Configuration](docs/configuration.md): settings, approvals, providers, project trust
- [Usage](docs/usage.md): CLI flags, keybindings, slash commands
- [Extending](docs/extending.md): tools, skills, templates, hooks, permissions, validation, freezing
- [stdio protocol](docs/stdio-protocol.md): the `openmax-stdio/5` contract for custom frontends

## Development

```sh
cargo check
cargo test
cargo build --release -p openmax
```

Set `OPENMAX_PERF=1` while running the TUI to log frame, transcript-layout, and selection-overlay timings. Core logic is in `open-max-core` (`crates/core/src/agent.rs`); the TUI is `crates/tui/src/app.rs`.

## Status

Open Max is early software. File an issue or send a PR if something breaks.

## License

MIT. See [LICENSE](LICENSE).
