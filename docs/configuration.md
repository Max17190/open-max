# Configuration

Open Max reads `~/.openmax/settings.json` for the active endpoint and an
optional `~/.openmax/providers.json` for a catalog of named endpoints.

## Settings

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5-coder:7b",
  "api_key": null,
  "approval_mode": "ask",
  "max_parallel_tools": 4
}
```

`base_url` is the root of your model's HTTP API (the harness calls
`chat/completions` on it). Set `model` to the id that server expects. Set
`api_key` to a literal or `$ENV_VAR`, or export `OPENMAX_API_KEY`.
`max_parallel_tools` bounds concurrent read-only tool calls, defaults to 4, and
is clamped to 1 through 32 at runtime. Mutating, approval-gated, and
non-batchable calls remain serial.

A missing settings file means defaults, and the default `base_url` and `model`
are empty: endpoint resolution fails with an actionable error until you set
both, here or through a named provider. There is no built-in localhost
fallback. A settings file that exists but does not parse, uses an unknown key,
or sets an unrecognized `approval_mode` is a startup error (fail closed): Open
Max exits with the parse reason instead of silently reverting your endpoint and
approval policy to defaults.

## Context window and compaction

Four more settings shape the context budget; all are optional, and a named
provider's per-model entries override the first two.

- `context_tokens` (default 16384): the model's context window. Nothing is
  queried from the server; set this to what your model actually has.
- `max_tokens` (default 4096): the completion reserve, clamped so it never
  eats the window (at most `context_tokens - 2048`).
- `temperature` (default 0.2).
- `max_output_bytes` (default 30000, floor 1000): per tool-result cap; bash
  keeps the tail and spills the full log to `~/.openmax/cmd-logs`.
- `max_agent_iterations` (default 50): tool-call rounds one turn may take.

Each turn budgets `context_tokens - (max_tokens + 1024)` for the transcript
plus the frozen tool schemas (estimated at ~4 chars per token). Over budget,
compaction prunes hard to 70% of the budget in one pass and then leaves
history untouched until it is crossed again, so the prompt prefix stays
byte-stable between prunes and the server-side prompt cache stays warm. A
prune truncates old tool outputs first, then drops the oldest exchanges;
everything dropped is appended verbatim to
`~/.openmax/sessions/<id>.archive.jsonl`, and the context note that replaces
the dropped span names that path, so compaction stays reversible: the model
summary (or its heuristic fallback) is the bounded view, the archive is the
lossless record. The `compaction` hook observes each prune.

## Approvals

`write_file`, `edit_file`, and `bash` wait for approval in `ask` mode. Use
`auto` for unattended runs or `readonly` to block mutating tools. Approvals and
permissions decide whether Open Max dispatches a tool call; they are not OS
isolation.

`/approvals auto|ask|readonly` writes the mode to `settings.json`. **Shift+Tab**
cycles the three for the current run only, so a keystroke never widens what
future sessions in the project may do.

## Multiple providers

For several servers, define them in `~/.openmax/providers.json`. `/model` opens
a searchable local catalog and selects the provider and model as one pair.
Model names are optional, model ids are sent unchanged, and the configured
order is preserved within each provider. Opening the picker makes no network
requests.

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

Set `"provider"` in settings, use the provider CLI option, or use `/provider`
when you only want to change the endpoint. Optional `compat` flags cover picky
gateways (for example `max_completion_tokens` versus `max_tokens`).

Open Max works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud
gateways (OpenRouter and similar), and private proxies.

## Project trust

A canonical project root must be trusted before any agent turn or project
behavior starts. Interactive use asks once; headless and stdio runs fail
closed until explicitly started with `--trust-project`. A trusted root covers
its subtree, so worktrees under it (for example `.worktrees/`) need no extra
grant; a sibling directory whose name merely extends the root never rides
along.

Trust grants are human actions. Every process the agent loop spawns carries
`OPENMAX_SESSION`, and under that marker both `--trust-project` and the
interactive trust prompt refuse: a session cannot grant itself, or a child it
starts, trust in a new directory.

```sh
openmax --trust-project -p "summarize this repo"
openmax --trust-project --stdio
```

Trust is persisted for the exact canonical path in `~/.openmax/trust.json`. It
authorizes the harness to run in that project; it does not sandbox project
code.

## Hardened profile

The honest answer to "lock the agent down" is configuration the harness
already has, not a sandbox. In auto mode, `bash` runs unprompted with the
full authority of your account; that is the deliberate default for fluid
work in a trusted project. To harden a project:

- `"approval_mode": "ask"` in settings.json puts every mutating call in
  front of you, or
- keep auto mode but add an ask rule for bash in your **global**
  `~/.openmax/permissions.toml` (outside the project root, so the agent
  cannot edit it):

```toml
[[rules]]
effect = "ask"
tool = "bash"
```

External tools are narrower than bash by construction: every tool's first
run needs your approval of its exact bytes, its environment is scrubbed to
a baseline plus the `env` names its approved manifest declares, and
unapproved tool code can only ever execute as a sandboxed probe (no
network, writes confined) via `openmax --check --run-examples`.

Residual risk to know about: an *approved* tool or a bash command can fetch
content at runtime (`curl | sh`) that no approval ever covered. Approval
binds bytes on disk at approval time, deliberately - pinning runtime
fetches is unbounded. Review what you approve, and prefer tools whose code
lives in project files where the approval reaches it.
