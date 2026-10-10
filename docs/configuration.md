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
  "context_tokens": 32768,
  "max_parallel_tools": 4
}
```

`base_url` is the root of your model's HTTP API (the harness calls
`chat/completions` on it). Set `model` to the id that server expects. Set
`api_key` to a literal or `$ENV_VAR`, or export `OPENMAX_API_KEY`. A key never
crosses plain http to another machine: with a key (or `user:password` in the
`base_url`) configured and an `http://` `base_url` that is not a loopback
address (`127.0.0.1`, `::1`, or `localhost`), every request fails with an
error instead of sending it. Use https, or a
loopback address for a server on this machine; a server that needs no key works
over http once none is configured. A redirect is followed only on the same
scheme, host, and port as `base_url`: one to another server fails the request
with an error naming that server, since following it would hand the request
and its headers to a server you did not configure.
`max_parallel_tools` bounds concurrent read-only tool calls, defaults to 4, and
is clamped to 1 through 32 at runtime. Mutating, approval-gated, and
non-batchable calls remain serial. In `auto`, read-only external tools batch
whether or not a human approved their content; outside `auto`, an unapproved
one is never batched: it takes the serial path, which asks for approval or,
in `readonly`, refuses the call.

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

- `context_tokens` (required, here or in the model's `providers.json`
  entry): the model's context window. Nothing is queried from the server,
  and there is no default: a guessed window is wrong in one direction or the
  other (too small compacts a large model's history early, too large lets a
  small model's requests fail for length), so a turn refuses with a message
  naming this field until it is set. Set it to what the server actually
  serves for the model. `openmax --check` warns while it is missing.
- `max_tokens` (default 4096): the completion reserve, clamped so it never
  eats the window (at most `context_tokens - 2048`).
- `temperature` (default unset): sent only when set, so otherwise the
  server's own default applies. Leave it unset for OpenAI reasoning models,
  which reject any value but 1. Server defaults differ: Ollama's
  OpenAI-compatible endpoint uses 1.0, not the Modelfile's value. Earlier
  versions wrote `"temperature": 0.2` on every settings save; if you never
  chose it, delete the key.
- `max_output_bytes` (default 30000, floor 1000): per tool-result cap; bash
  keeps the tail and spills the full log to `~/.openmax/cmd-logs`.
- `max_agent_iterations` (default 50): tool-call rounds one turn may take.

Each turn budgets `context_tokens - (max_tokens + 1024)` for the transcript
plus the frozen tool schemas (estimated at ~4 chars per token, scaled up when
the server reported more `prompt_tokens` than estimated for the session's last
request, as BPE tokenizers do on code). Over budget, compaction prunes hard to
70% of the budget in one pass and then leaves
history untouched until it is crossed again, so the prompt prefix stays
byte-stable between prunes and the server-side prompt cache stays warm. A
prune truncates old tool outputs first, then drops the oldest exchanges;
everything dropped is appended verbatim to
`~/.openmax/sessions/<id>.archive.jsonl`, and the context note that replaces
the dropped span names that path, so compaction stays reversible: the model
summary (or its heuristic fallback) is the bounded view, the archive is the
lossless record. The `compaction` hook observes each prune.

## Approvals

Each trusted project's mode is saved in `~/.openmax/trust.json` for its exact
canonical path, and the trust grant records it in the same write: `auto`,
unless you answer `a` (`ask`) or `r` (`readonly`) at the interactive trust
prompt. `--trust-project` records `auto`. A project that is already trusted
keeps its mode when trust is granted again. Change the mode with
`/approvals auto|ask|readonly`, **Shift+Tab**, or the approval card's **Auto
for project** choice; every selector saves the same choice. It survives new
sessions and restarts, including headless and stdio runs, and does not change
other projects. The `settings.json` `approval_mode` (default `ask`) applies
only where neither the project nor an enclosing one has a saved mode, such as
a project trusted by an earlier version; select a mode there once to save
one. Symlink aliases share the choice. A subdirectory of the project,
including a worktree placed under it (such as `.worktrees/`) where a
delegated `openmax -p` child starts, follows the nearest enclosing project's
choice unless it has a saved choice of its own. Inheritance follows the path,
not the Git repository: a worktree outside the root, such as one made with
`git worktree add ../feature`, gets neither the project's trust nor its
choice and needs its own.

- `auto` runs authorized work without confirmation, including newly created
  or repaired tools, hooks, and permission `allow` rules. It ignores
  permission `ask` requests. It neither reads nor creates content approval
  records.
- `ask` prompts for mutating calls unless a permission `allow` applies, and
  always prompts for unapproved external tool content. Hooks and permission
  `allow` rules, project or global, still require content approval for the
  directory the session starts in; unlike the mode, it is not inherited.
- `readonly` blocks mutating calls and calls that would require confirmation.

Deny rules, hook gates, parsing, output limits, and execution reports remain
active in every mode. Rules are still first-match and cannot relax during a
turn. Hook file edits apply at the next turn; explicit mode changes refresh
policy before subsequent calls. A failed save leaves the active mode unchanged.
If the mode changes during pre-tool checks, those calls are refused without
replaying hooks. The agent can request them again under the new mode.
Returning to `ask` restores content requirements for anything created in `auto`.

Only a human can select a saved mode: at the trust grant, or with the
selectors above in a human-controlled frontend. Agent-spawned processes can
use an existing choice but can neither grant trust nor change the choice
through the mode command. Trust and settings are read at process launch;
editing either file from a tool does not change that process's chosen mode.
These controls govern dispatch and are not OS isolation.

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

A provider takes its key from its own `api_key` or `api_key_env`. Without one,
it uses the settings `api_key` (or `OPENMAX_API_KEY`) only when its `base_url`
has the same scheme, host, and port as the settings `base_url`: that key was
configured for that server. Any other provider gets no key, and a 401 from it
says how to give that provider its own. A credential in a provider's `headers`
(a header whose name contains `auth`, `key`, `token`, `secret`, `password`,
`credential`, or `cookie`, such as `Authorization` or `X-API-Key`) follows the
same plain-http rule as the key.

A request has no overall deadline, since a local server can spend minutes on a
long prompt, but an endpoint that sends nothing at all (no response headers, no
reply bytes, not even an SSE keepalive comment) for 10 minutes ends the
attempt. If no response arrives, or a stream goes silent before any reply
text, the attempt is resent like a dropped connection; a stream that goes
silent after reply text is reported truncated. A provider's
`idle_timeout_secs` sets a different interval for that provider, in seconds
(0 does not disable it, and `--check` rejects it).

Open Max works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud
gateways (OpenRouter and similar), and private proxies.

## Project trust

A canonical project root must be trusted before any agent turn or project
behavior starts. Interactive use asks once; headless and stdio runs fail
closed until a human grants trust with `--trust-project`. A trusted root covers
its subtree, so worktrees under it (for example `.worktrees/`) need no extra
grant; a sibling directory whose name merely extends the root never rides
along.

Trust grants are human actions. Every process the agent loop spawns carries
`OPENMAX_SESSION`, and under that marker both `--trust-project` and the
interactive trust prompt refuse: a session cannot grant itself, or a child it
starts, trust in a new directory. Outside a session, `--trust-project` also
needs a terminal on stdin, so grant it once from a shell:

```sh
openmax --trust-project -p "summarize this repo"   # from a terminal
```

A frontend spawns `openmax --stdio` with stdin as its protocol pipe, which is
not a terminal, so it cannot grant trust: `--trust-project` there exits 3.
Trust the project once from a terminal (the command above, or the interactive
prompt); from then on the frontend starts sessions with plain
`openmax --stdio`.

Automation a human runs with no terminal, such as CI, states the human act
explicitly with `OPENMAX_HUMAN_ATTEST=1`. It is honored only outside an agent
session, and the harness strips it from every process it spawns:

```sh
OPENMAX_HUMAN_ATTEST=1 openmax --trust-project -p "summarize this repo"
```

Trust is persisted for the exact canonical path in `~/.openmax/trust.json`,
together with the project's approval mode (see [approvals](#approvals)). It
authorizes the harness to run in that project; it does not sandbox project
code.

## Hardened profile

Answer `a` at the trust prompt, or use `/approvals ask`, when you want
confirmation. In `auto`, bash and valid extensions run with your account's
host authority without a prompt. A permission `ask` rule does not override
that choice. Use `deny` to prohibit a tool or command pattern, or switch to
`readonly` to disable mutating calls.

External tools receive a scrubbed baseline environment plus the variable names
listed in their manifests. In `ask`, content approval covers the manifest and
the project-local code it names. It cannot cover code fetched or selected at
runtime. Prefer scripts named in manifest arguments when that binding matters.

Trust grants, content grants, and saved mode changes reject processes carrying
`OPENMAX_SESSION`. This prevents a normal child agent from treating its own
commands as human decisions. It is not a sandbox: an agent with unrestricted
bash can alter environment markers or files outside the project. Headless
runs decline any approval request; use a human-controlled TUI or stdio frontend
for work that needs `ask` mode decisions.
