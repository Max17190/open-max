# Design

What Open Max does, what it leaves for the agent to build, and the files it builds them from.

## Features

Open Max is a single binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. Point it at the model server you choose: local, cloud, or a private proxy. No desktop shell, no heavyweight runtime, no telemetry.

You own the endpoints, the tools, the skills, and the context.

- **Small by default.** Seven built-in tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`) and a short system prompt. Old tool output is dropped before your task is, and dropped context is summarized by your own model into a compact note (heuristic digest as fallback) whose address points at the lossless archive of everything dropped: compaction is a bounded view over a record you can always read back.
- **Your model, your server.** One `base_url`, or several named endpoints in `providers.json` switched with `/model`. Works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud gateways (OpenRouter and similar), and private proxies.
- **Trust before execution.** An exact canonical project root must be trusted before any agent turn or project behavior starts. Interactive use asks once; headless and stdio runs fail closed until a human grants trust with `--trust-project`.
- **Newly trusted projects run in auto.** Trusting a project records its approval mode, `auto` unless you pick another: tools and extensions run without per-call approval while deny rules, hooks, and execution reports stay active. Pick `ask` to have `write_file`, `edit_file`, and `bash` wait for approval, or `readonly` to block mutating tools. Approvals and permissions decide whether Open Max dispatches a tool call; they are not OS isolation.
- **File based extensions.** Drop TOML tools, `SKILL.md` skills, prompt templates, and process hooks under project or home config. No fork required. The agent writes them itself and the harness re-freezes as soon as a mutating call lands, so a tool the agent writes is a tool the agent uses on its very next step. The system prompt and the tools array stay the same, byte for byte, for the whole session, until you `/reload`: a tool, skill, or memory written after they were built rides a note stored once, where it changes, on the result of the write that changed it, so a provider's prompt cache keeps the prompt, the conversation, and the note through every write.
- **File based memory.** One durable fact per file in `.openmax/memory/`, written by the agent, surfaced as an index line in future sessions, ranked by recency and frequency of observed use. Old entries fade from the index and remain searchable on disk until the user or agent deletes them. No database, no daemon, no embeddings; zero prompt cost when empty.
- **Recall over everything kept.** `openmax --recall "<query>"` searches this project's past sessions, compaction archives, digests, and memories in one bounded streaming pass: BM25 relevance fused with the same recency law, ranked excerpts, each citing the file with the full record. No index to build or maintain: the stores on disk stay the single source of truth.
- **Visible work.** Reads, greps, diffs, and shell commands stream as they happen in a fullscreen TUI. Headless print mode for scripts and CI.
- **Local sessions.** Conversation state lives under `~/.openmax/`. The harness contacts your configured model endpoint and, when you connect ChatGPT, OpenAI's authentication service.
- **Offline replay.** Each session keeps an event log. Copied to any machine, `openmax --replay <log> --json` prints from it, byte for byte, what `-p --json` printed for every turn that finished, with no endpoint, settings, trust or data dir. The log is written when a turn ends (and mid-turn once 64 KiB of events wait), so a run killed mid-turn leaves out what that turn had not written, often all of it; `openmax -c` still continues the session from its saved transcript.

## Design

The design starts with one question: **What is the smallest capability Open Max must provide so the agent can construct richer behavior itself?** The answer is one focused native loop, seven primitive tools, a fast event-driven TUI, context management, and stable file and process contracts.

| Need | Construct it with |
| --- | --- |
| External service or specialized capability | A CLI-backed TOML tool plus an on-demand skill, without an MCP runtime |
| Reusable workflow or command | A `SKILL.md` package or prompt template slash command |
| Isolated or parallel work | `openmax --subagent "<task>" --worktree`, one child session in a new git worktree whose changes land on a branch, or a child `openmax -p` or `openmax --stdio` process, usually in tmux |
| Durable background work | A named tmux session that the agent can inspect and reattach |
| Planning and task state | `PLAN.md` and `TODO.md`, visible to the user and every tool |
| Lifecycle policy and events | Process hooks and permission files |
| Compaction integration | The built-in model summary plus the `compaction` hook event |
| Model endpoints | `providers.json`, including local servers, gateways, and private proxies |
| Shortcuts or a completely different UI | Prompt templates, or a custom frontend speaking `openmax-stdio/10` |

These are deliberate boundaries, not placeholders for hidden orchestration products. Open Max does not carry an MCP host, nested-agent scheduler, plan mode, background-job product, built-in TODO database, user-keybinding engine, pluggable compactor, or TUI plugin ABI. The agent composes those richer workflows from the same host tools a developer can inspect, edit, test, and remove.

## Extend

With nothing installed, extensions cost zero tokens. Project paths win over global ones on name collision. Each surface is a file the agent can write, `openmax --check` can validate, and the next turn picks up automatically.

| Surface | Location | What it is |
| --- | --- | --- |
| **Tools** | `.openmax/tools/*.toml` | Schema-described native processes; args on stdin, result on stdout |
| **Skills** | `.agents/skills/*/SKILL.md` | On-demand workflow packages; only name and description cost prompt tokens |
| **Prompt templates** | `.agents/prompts/*.md` | User-invoked messages; the file stem becomes a slash command |
| **Hooks** | `.openmax/hooks/` | Process gates on a fixed set of lifecycle events; three of them can block |
| **Permissions** | `.openmax/permissions.toml` | Allow, deny, or ask rules matched against tool arguments; fail closed |

```toml
# .openmax/tools/todo_scan.toml
name = "todo_scan"
description = "List TODO/FIXME comments with file and line"
command = "sh"
args = ["./scripts/todo-scan.sh"]
timeout_secs = 30
mutating = false

[params]
type = "object"
required = ["path"]
[params.properties.path]
type = "string"
description = "Directory to scan"
```

Unknown keys are rejected in tool, hook, and permission files and in `settings.json`, and `openmax --check` names any in `providers.json`, so a misspelled field there shows up in `openmax --check` rather than silently changing behavior (`SKILL.md` and template frontmatter ignore keys they do not read). `openmax --spec <surface>` prints the complete authoring contract for any one of them, and the frozen payload carries only a one-line pointer to it, in the `bash` tool's description rather than the system prompt, so that contract costs zero tokens until the agent reads it. The tools and hooks contracts leave their content approval detail for ask and readonly to `openmax --spec tools.approval` and `hooks.approval`, which they name, so a session in auto never pays for it.

File formats, hook events, permission rule syntax (including the in-session repair path for a broken permissions file), self-description, and the freezing model are documented in [extending](extending.md).
