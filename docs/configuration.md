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
address (`127.0.0.1`, `::1`, or `localhost`) or the unspecified address
(`0.0.0.0` or `::`, which reaches this machine), every request fails with an
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
are empty: endpoint resolution fails until `base_url`, `model` and
`context_tokens` are set, here or through a named provider, and one error
names every one that is missing, with the flag that writes it and an example
of both. There is no built-in localhost fallback. A settings file that exists but does not parse, uses an unknown key,
or sets an unrecognized `approval_mode` is a startup error (fail closed): Open
Max exits with the parse reason instead of silently reverting your endpoint and
approval policy to defaults.

### Settings from the command line

The first run can write settings.json for you, so the whole first run is one
command:

```sh
openmax --trust-project --base-url https://llm.example.com/v1 --model my-model --api-key '$LLM_API_KEY' --context-tokens 131072 -p "summarize this repo"   # from a terminal
```

`--base-url`, `--api-key`, `--model` and `--context-tokens` each set the key of
the same name, and one rule decides what each does:

- A key settings.json leaves unset (there is no file yet, `base_url`, `model` or
  `api_key` is empty, or `api_key` or `context_tokens` is absent) is
  written to the file before the run. The file is saved whole, as `/model`
  saves it: every value it had is kept, and a setting it left out is written
  with its default. The run prints `openmax: wrote <keys> to <path>` on
  stderr.
- A key the file sets to another value is overridden for this run only. The
  file keeps its value, and the run prints `openmax: for this run only, the
  command line overrides <keys> in <path>; the file is unchanged, so set them
  there to keep them` on stderr. A `/model` or `/provider` save during the
  session writes the file's value for each overridden key until the session
  sets that key to another value itself; from then on the session's value is
  saved, as any in-session choice is. A switch to another provider than the
  run's is the session's own choice of endpoint, so the model the session
  runs is saved with it, a flag's included: the file's `model` may be one
  that provider does not serve.
- A flag equal to the file's value changes nothing and prints nothing, so the
  first command can be run again as it is.

Keys are written only when the run uses the endpoint settings.json selects.
`--base-url` names the flat endpoint, so beside a `provider` in settings.json
it sets that provider aside for the run, and a `--provider` other than the
file's picks another endpoint, with `provider` among the keys the run
overrides; a `--base-url` other than the file's own `base_url` is another
server. On such a run every flag is for this run only, so a key meant for one
server is never written beside another's `base_url`. While the run uses that other
endpoint, a model picked with `/model` is that endpoint's, so a save keeps the
file's `model` as well. A `/provider` switch to another provider than the
run's is saved with the model the session runs, and the file then selects
the run's endpoint, so later picks are saved too.

settings.json's `api_key` is the credential for the server its `base_url`
names, by scheme, host and port. A `--base-url` naming another server is not
sent it: without `--api-key`, the run sends no settings key and prints
`openmax: this run sends no api_key: the one in <path> is for the server its
base_url names, and --base-url names another; pass --api-key to send one` on
stderr, with `api_key` among the keys the run overrides, so the file keeps
it. `OPENMAX_API_KEY` in the command's environment is still sent, as it is
to any flat endpoint. A `--base-url` on the file's server with another path
keeps the file's key, and so does one the run writes to an unset `base_url`,
which makes its server the file's.
`--provider` beside `--base-url` or `--api-key` is refused (exit 2), since a
named provider takes its `base_url` and key from providers.json. `--provider`
itself is never written by the command line.

Beside a `provider` that settings.json selects, the provider's entry in
providers.json supplies the key, so `--api-key` without `--base-url` is
refused (exit 2) before the trust grant, with nothing written: the settings
`api_key` goes only to the server `base_url` names (see [multiple providers](#multiple-providers)).
`--context-tokens` there is the window for a model the entry does not size;
when the entry gives the run's model a window, that window wins, and the run
says so on stderr.

Values are checked before anything runs, and a refused one exits 2 with
nothing written: `--base-url` must be an http or https URL, `--context-tokens`
a positive number, `--model` not the placeholder `MODEL`, an `--api-key
'$NAME'` must name a variable set in the environment the command runs in, and
no flag may be empty. Values are trimmed. Single quotes around
`'$LLM_API_KEY'` keep the shell from expanding it, so settings.json stores
the variable's name, and the key is read from the environment when a request
is sent; a literal key is stored as given, in plain text. The flags are applied once, at
launch and before the first turn: settings.json is still read only at launch,
and the run that writes a key already uses it.

## Context window and compaction

Five more settings shape the context budget and each turn; all but
`context_tokens` are optional, and a named provider's per-model entries
override the first two.

- `context_tokens` (required, here or in the model's `providers.json`
  entry): the model's context window. Ordinary turns do not query it from the server,
  and there is no default: a guessed window is wrong in one direction or the
  other (too small compacts a large model's history early, too large lets a
  small model's requests fail for length), so a turn refuses with a message
  naming this field until it is set. Set it to what the server actually
  serves for the model. `openmax --check` warns while it is missing.
- `max_tokens` (default 4096): the completion reserve, clamped so it never
  eats the window (at most `context_tokens - 2048`), and sent as the reply's
  length limit. When the server reports the cut (finish_reason `length`, or
  a usage count at the limit), a tool call cut off at that limit does not
  run; the model is told why and to send large content in parts.
- `temperature` (default unset): sent only when set, so otherwise the
  server's own default applies. Leave it unset for OpenAI reasoning models,
  which reject any value but 1. Server defaults differ: Ollama's
  OpenAI-compatible endpoint uses 1.0, not the Modelfile's value. Earlier
  versions wrote `"temperature": 0.2` on every settings save; if you never
  chose it, delete the key.
- `reasoning_effort` (default unset): an explicit requested level: `none`,
  `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. Use `/effort` for a
  keyboard picker, Shift+Tab to cycle the active protocol's levels,
  `/effort high` to select directly, or `--effort high` at launch. Session
  choices persist, including selecting the current CLI level,
  unless the endpoint is a temporary CLI override. In that case the saved
  endpoint keeps its own effort setting. `/effort default` removes the setting
  in that same scope; `--effort default` clears it for the run when the file
  already sets it. Default omits the wire field,
  unlike explicit `none`. Responses sends `reasoning.effort`; Chat Completions
  sends `reasoning_effort`. Native Anthropic Messages sends
  `output_config.effort` and offers `low`, `medium`, `high`, `xhigh`, and `max`
  plus Default. It rejects `none` and `minimal`; the picker and shortcut leave
  them out. Supported levels still depend on the model and endpoint;
  the server validates explicit selections. A rejection is reported without
  retrying at another level. Sampling and protocol settings are unchanged.
  The Anthropic compatibility endpoint ignores effort, so explicit selections
  there are refused before saving or sending. See the [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)
  and [compatibility documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk).
- `max_output_bytes` (default 30000, floor 1000): per tool-result cap; bash
  keeps up to a quarter of each stream's share of the cap from its start and
  the rest from its end, marks the cut with the bytes it dropped, and spills
  up to 16 MiB of each stream to `~/.openmax/cmd-logs`. Before any of that,
  bash keeps only the final text of a line redrawn with carriage returns
  (a progress bar): text that follows a `\r` on the same line replaces the
  whole line so far, so only its final text counts against the cap and
  reaches the log. `\r\n` stays a line ending, a run of `\r` before a `\n`
  becomes one `\r\n`, and a `\r` that ends stdout or stderr is dropped
  (`last\r` reaches the model as `last`). Once a line holds more than 8 KiB
  since its last redraw, that text and the rest of the line up to its next
  `\r` or `\n` are passed on whole, and a later redraw no longer replaces
  them. The `\r` after them is kept (one for a run of them), so the next
  frame starts after it instead of running on from their last byte. A file
  with lone-`\r` line endings printed through bash shows only its last
  record the same way; `tr '\r' '\n' < file` shows every record.
- `max_agent_iterations` (default 50): tool-call rounds one turn may take.

Each turn budgets `context_tokens - (max_tokens + 1024)` for the transcript
plus the frozen tool schemas and the trailing note every request re-sends
beside it (estimated at ~4 bytes per token, scaled up when
the server reported more `prompt_tokens` than estimated for the session's last
request, as BPE tokenizers do on code, or refused a request for length
(recognized by the error's wording) that the estimate said fit). Both are kept
in memory, so a resumed session starts on the plain estimate. Over budget,
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

Each trusted project's mode is saved in `~/.openmax/trust.json` for its exact
canonical path, and the trust grant records it in the same write: `auto`,
unless you answer `a` (`ask`) or `r` (`readonly`) at the interactive trust
prompt. `--trust-project` records `auto`. A project that is already trusted
keeps its mode when trust is granted again, and a grant or mode selection that
would leave `trust.json` as it is writes nothing. `/permissions` opens a picker;
`/permissions auto|ask|readonly` selects directly. The approval card also offers
**Auto for project**. Every selector saves the same choice. It survives new
sessions and restarts, including headless and stdio runs, and does not change
other projects. The `settings.json` `approval_mode` (default `ask`) applies
only where neither the project nor an enclosing one has a saved mode, such as
a project trusted by an earlier version; select a mode there once to save
one. Symlink aliases share the choice. A subdirectory of the project,
including a worktree placed under it (such as `.worktrees/`) where a
delegated `openmax -p` child starts, follows the nearest enclosing project's
choice unless it has a saved choice of its own. A `--subagent` child's
worktree there runs in the project's choice or a stricter one, never a looser
one (see [sub-agents](usage.md#sub-agents)). Inheritance follows the path,
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

The model reads the mode in a one-line `[execution policy: <mode>; ...]` note
placed ahead of a session's first prompt, and written again only when the mode
changes or a resumed session's history no longer holds it. The history resends
it on every later request, so it states only the mode's limits; content
approval detail is in `openmax --spec tools`, `hooks`, and `permissions`.

Only a human can select a saved mode: at the trust grant, or with the
selectors above in a human-controlled frontend. Agent-spawned processes can
use an existing choice but can neither grant trust nor change the choice
through the mode command. Settings are read at process launch, so editing
`settings.json` from a tool does not change a running session's mode.
A running session reads `trust.json` again before each tool call but takes
only a stricter mode from it, so editing that file from a tool never widens a
running session's mode either. Select `ask` in one frontend and another
session on the same project asks from its next tool call. Its status line
shows that stricter mode when refreshed. Shift+Tab changes only
reasoning effort and cannot widen permissions. Only that session's own
selector, or a new launch, loosens it again.
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

Set `"provider"` in settings, use the provider CLI option, or use `/provider <name>`
when you only want to change the endpoint. Bare `/provider` opens the searchable
model picker. Optional `compat` flags cover picky
gateways (for example `max_completion_tokens` versus `max_tokens`).
`compat.omit_temperature: true` leaves sampling temperature to that provider;
it defaults to false and leaves other providers' settings unchanged.

`wire_api` defaults to `chat_completions`, preserving existing requests.
An explicit `responses` selects the Responses protocol. The ChatGPT setup
command writes this field with `chatgpt_account`; other endpoints may use
Responses with their own API key.

Every Responses request carries a `prompt_cache_key`, a hash of the id of
the session its conversation began in. A session sends the same key on every
request, a resumed one included, and a `/fork` or `/rewind` branch keeps its
parent's key, since it repeats the parent's history and requests under
different keys share no cache entries. A session started on its own gets a
key of its own, so it never reuses what another session cached, not even the
opening every session of the project shares. On OpenAI models before GPT-5.6,
the key also influences routing: which machine serves a request depends on
load, a hash of the prompt's first tokens, and the key, which does not pin a
request to a machine or guarantee a cache hit. Every session of a project
opens with the same tokens, so without a key, sessions run side by side share
one route, traffic past about 15 requests a minute can overflow to other
machines, and a request served where its prompt is not cached processes its
uncached part again. Branches run at the same time share their conversation's
key, so together they can overflow its route the same way. From GPT-5.6,
routing needs no key, and the key only keeps each conversation's cache
entries apart. The key is request metadata the model never sees.

The Responses translation targets OpenAI's Responses API, which defines the
key, and already uses other constructs that API defines, such as the
`additional_tools` input item that carries the tools. Chat Completions
requests do not carry the key: the servers that implement that protocol vary,
and some refuse a field they do not know. Native Messages does not define it.

An explicit `anthropic` selects the native Messages protocol. Set `base_url`
to `https://api.anthropic.com/v1`; requests use `/v1/messages`, `x-api-key`
authentication, and `anthropic-version: 2023-06-01` unless a custom version
header is configured. Supply the key through `api_key` or `api_key_env`, or a
single custom `x-api-key` header, without combining key sources. For example:

```json
{
  "providers": {
    "native": {
      "base_url": "https://api.anthropic.com/v1",
      "wire_api": "anthropic",
      "api_key_env": "ANTHROPIC_API_KEY",
      "compat": { "omit_temperature": true },
      "models": []
    }
  }
}
```

Populate `models` with the IDs and context windows available to the account,
or use login to discover them. Native Messages translates the system prompt,
text, tools, and tool results while retaining the existing agent loop and trust
checks. Signed thinking blocks return only to their producing endpoint, model,
and credentials, with unchanged prior system, tools, and messages, and matching
assistant text and tool calls. Compaction, reloads, transient notes, or edits
that change this context discard stale thinking while retaining the current
text and tool calls. An interrupted,
malformed, or failed stream cannot dispatch
partial tool calls. Effort is sent as `output_config.effort`; default omits it.
Native levels are `low`, `medium`, `high`, `xhigh`, and `max`, with model-specific
support validated by the server. `none` and `minimal` are refused. Effort works
with or without thinking, so the adapter leaves the model's thinking defaults
unchanged. It also preserves configured sampling settings; omit temperature for
models that reject it. See the official [Messages API](https://platform.claude.com/docs/en/api/messages/create),
[effort](https://platform.claude.com/docs/en/build-with-claude/effort), and
[thinking](https://platform.claude.com/docs/en/build-with-claude/thinking), and
[preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking) contracts.

The trailing note (the tool files, skills, and memories a session added,
changed, or removed after its prompt and tools array were built) is appended
to the content of the request's last message, the user's prompt or the newest
tool result, after a blank line and a header that names the harness as its
source, and it is never stored. Every endpoint, a flat `base_url` or a named
provider, therefore sees the stored transcript's role order, the same as with
no note, and the system prompt stays the only system message. A note of its
own after the history is what picky servers refuse: some hosted validators
reject a system or a user message after a tool message, a chat template that
enforces user/assistant alternation rejects a user message after the prompt,
and some gateways hoist every system message into the system prompt, which
would move the cached prefix on every capability write. Only a history that
ends with an assistant reply gets the note as a user message of its own,
which those servers take there. A prefix cache keeps every stored byte before
the note.

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

## Connecting and disconnecting providers

Run `openmax --login` from a terminal, or `/login` inside a session. Choose a
provider with the arrow keys, type to filter, and press Enter. Esc cancels
a setup prompt before its changes are saved. The choices are:

- **OpenAI: ChatGPT plan:** browser sign-in using the account's plan. The
  command target is `chatgpt`; the default saved connection is `chatgpt-main`.
- **OpenAI API:** an API key with separate API billing. Target `openai`.
- **Anthropic API:** an API key with separate API billing. Target `anthropic`.
- **Other compatible API:** an HTTPS API URL and a Bearer API key. Target
  `custom`. Plain HTTP is accepted only for a server on loopback.
- Existing named connections, for updating their API key or selected model.

For example, `openmax --login anthropic` skips the provider menu. Keys are
entered in a hidden terminal prompt, outside composer history and the
transcript. When the corresponding `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
environment variable is present, the prompt offers to copy its current value.
The connection stores that value privately, rather than depending on that
environment variable in future shells. Never paste a key as a slash-command
argument.

API setup loads the endpoint's model catalog and lets you choose an exact
model. OpenAI setup omits known dedicated embedding, image, speech, realtime,
and moderation model families; other providers retain their own catalogs.
When the API does not report its context window, enter the value from
the model's documentation. An endpoint without a model-list route can be
configured with an exact model ID; setup reports that access was not verified.
The selected model is added to the saved catalog, preserving its other models.
Run login again to add another. Choose the connection with `/model` afterward,
or use the printed `openmax --provider ... --model ...` command. Login does not
silently change the default provider or project trust.

New Anthropic preset connections use the native Messages API, including
streaming, tools, usage accounting, and effort controls. Existing compatibility
connections retain their wire protocol; select `wire_api: "anthropic"` explicitly
to use native Messages. It uses API-key authentication with separate API billing.
Setup asks for the workspace ID when needed; leave it empty for a workspace-scoped key. See
[Anthropic authentication](https://platform.claude.com/docs/en/manage-claude/authentication).
New API preset connections omit temperature, so the server chooses its supported
default. Reconnecting an existing provider preserves its compatibility settings
and wire protocol, including their implicit defaults.

Use `/logout` or `openmax --logout` to choose a connection to disconnect.
ChatGPT logout revokes and clears its local tokens. For an API-key connection,
confirming logout removes that named connection and its model catalog from
`providers.json`. It does not revoke the API key at its issuer or edit shell
environment variables. A still-selected removed provider fails explicitly;
it cannot fall back to another key or endpoint. Other connections and settings
remain unchanged.

Connection setup requires a human terminal, runs without project hooks, and
grants no trust. Catalog writes preserve unrelated entries, take a shared
process lock, and atomically save owner-only files. Existing `--chatgpt`
commands and manually configured providers remain available.

## ChatGPT sign-in

Run account setup from a human terminal, independently of a project:

```sh
openmax --chatgpt login
openmax --chatgpt status
openmax --chatgpt models
openmax --chatgpt logout
```

`login` opens OpenAI's consent page in your browser and prints its URL as a
fallback. The browser must reach the callback on the machine running Open Max,
at `127.0.0.1`. Sign-in waits up to five minutes. It uses the documented public
client flow, without a client secret or an API key. Login and logout use the
same human-action gate as a trust grant. They load no project instructions,
run no hooks, and grant no project trust.

The default account label is `main`. Add `--account work` to any setup command
to use another label. `status` lists all labels. A label retains its issued
client ID and verified identity after logout, so another identity needs
another label. Successful login fetches that account's current model catalog
and adds `chatgpt-main` (or `chatgpt-work`) to `~/.openmax/providers.json`:

```json
{
  "base_url": "https://api.openai.com/v1",
  "wire_api": "responses",
  "chatgpt_account": "main",
  "models": []
}
```

This is the provider entry's shape; setup fills `models` with the service's
visible models in their reported order. Setup preserves other providers and
does not select a provider or overwrite `settings.json`. Select a returned
model with `/model` or `openmax --provider chatgpt-main --model <model-id>`.
The picker reads the local catalog; `--chatgpt models` refreshes it explicitly.
When the service omits a context window, set the model's documented
`context_tokens` in its catalog entry or pass `--context-tokens`. The existing
global `context_tokens` fallback still applies; verify it for the selected model. Refreshing
the catalog retains a manually supplied window when the service omits one.
No model name, availability, or context limit is guessed.

ChatGPT credentials live only in `~/.openmax/chatgpt.json`, an owner-only
file (mode 600), separate from provider settings and session transcripts.
Refresh uses a process lock and atomically replaces the access and refresh
tokens together. A cancelled waiting turn lets a token rotation finish,
and normal shutdown waits for it to be saved. A forced exit or process kill
can still interrupt an in-flight rotation and require another sign-in.
Logout attempts remote revocation, clears local tokens, and reports when
remote revocation was not confirmed. To revoke access directly, use
[ChatGPT Settings](https://chatgpt.com/settings/usage). Temporary network
failures retain credentials; an unusable refresh token requires sign-in again.

The ChatGPT provider requires the fixed HTTPS API root above and refuses
custom keys, key environment variables, and headers. It never inherits
`OPENMAX_API_KEY` or switches billing after an error. If sign-in succeeds
without permission to use the plan, the account remains signed in with plan
usage disabled. Run `login` for that label again to request consent, or select
another provider. Availability depends on the account, plan, workspace, and
OpenAI policy. The TUI identifies the selected ChatGPT billing route, and
limit errors link to [usage settings](https://chatgpt.com/settings/usage);
an app limit does not imply that the whole plan is exhausted.

ChatGPT inference uses streaming Responses with `store: false` and the
required local history on every request. The existing loop, seven tools,
trust checks, approvals, permissions, hooks, and sessions still apply.
Opaque reasoning is replayed only to the same model, endpoint, and registered
account. Failed, incomplete, and interrupted responses cannot dispatch their
partial tool calls. The subscription route omits unsupported sampling and
output-limit parameters: `max_tokens` remains a local context reserve and
spend estimate, not an enforced server output limit, and `temperature` is
not sent. API-key providers retain their configured behavior.

Protocol references: [registration](https://developers.openai.com/siwc/token-sharing-open-source/sign-in),
[account storage and refresh](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions),
[inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference),
and [current route limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).

## Project trust

A canonical project root must be trusted before any agent turn or project
behavior starts. Interactive use asks once; headless and stdio runs fail
closed until a human grants trust with `--trust-project`. A trusted root covers
its subtree, so worktrees under it (for example `.worktrees/`) need no extra
grant; a sibling directory whose name merely extends the root never rides
along.

Trust grants are human actions. Every process the agent loop spawns carries
`OPENMAX_SESSION`, and under that marker both `--trust-project` and the
interactive trust prompt refuse: a process that keeps the marker cannot grant
trust in a new directory. Outside a session, `--trust-project` also needs a
terminal on stdin, so grant it once from a shell:

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
session, and the harness strips it from every process it spawns. Nothing
checks who set it, so a shell the agent runs can unset `OPENMAX_SESSION` and
then either set `OPENMAX_HUMAN_ATTEST=1` or run `openmax` on a pseudo-terminal,
and the grant goes through. The marker and terminal checks stop a grant made by
accident or by the plain retry `env -u OPENMAX_SESSION openmax --trust-project`
from that shell, not one an agent sets out to make. Set
`OPENMAX_HUMAN_ATTEST=1` on the command that grants trust:

```sh
OPENMAX_HUMAN_ATTEST=1 openmax --trust-project -p "summarize this repo"
OPENMAX_HUMAN_ATTEST=1 openmax --trust-project   # the grant alone
```

With none of `-p`, `--stdio` and `--subagent`, `--trust-project` goes on to
the interactive session, which draws on stdout. At a terminal, the session
paints its first frame before the grant is recorded, and the line that reports
the grant prints on the shell once the session ends, or ahead of the reason
when the session then cannot start (settings that do not parse). The lines that say what the
settings flags wrote or overrode print with it. A signal that forces the exit
first, one before the session's loop starts (Ctrl+C at the terminal included)
or a second one, ends the run without the line. Where stdout is not a
terminal, no session starts: the run ends after the grant and says so. The
grant alone exits 0.
With `--continue` or a settings flag (`--model`, `--provider`, `--base-url`,
`--api-key`, `--context-tokens`) as well, the run asked for the session
itself, which cannot start there, so it exits 1 once the grant is recorded,
and writes no settings.

Trust is persisted for the exact canonical path in `~/.openmax/trust.json`,
together with the project's approval mode (see [approvals](#approvals)). It
authorizes the harness to run in that project; it does not sandbox project
code.

## Hardened profile

Answer `a` at the trust prompt, or use `/permissions ask`, when you want
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
