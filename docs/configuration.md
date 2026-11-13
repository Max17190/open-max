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

`context_tokens` is required. A named provider's model entry can override
`context_tokens` and `max_tokens`. The other settings are optional.

- `context_tokens` (required, here or in the model's `providers.json`
  entry): the model's context window. Ordinary turns do not query it from the server,
  and there is no default: a guessed window is wrong in one direction or the
  other (too small compacts a large model's history early, too large lets a
  small model's requests fail for length), so a turn refuses with a message
  naming this field until it is set. Set it to what the server actually
  serves for the model. `openmax --check` warns while it is missing.
- `max_tokens` (default 4096): the reply's length limit, clamped so it never
  eats the window (at most `context_tokens - 2048`). A request asks for the
  room its prompt leaves, never less than the reply reserve and never more
  than this limit (see the input budget below), so a limit up to the larger
  of 4096 and an eighth of the window, the default among them, goes out on
  every request as it is. When the server reports the cut (finish_reason
  `length`, or a usage count at the limit), the last tool call of that
  reply, the one the limit cut, never runs, even when the server closed its
  arguments so they parse, and the model is told why and asked to send it
  again in parts; the calls before it run as usual. A
  reasoning model's reasoning counts toward the limit, so at 4096 a reply
  can be cut inside its reasoning, with no text and no call. A reply cut
  before any tool call does not end the turn: it stays in the transcript
  (its reasoning too, which goes back only to the endpoint that wrote it),
  a note after it tells the model the limit and to continue briefly, and
  the turn goes on. A server can refuse a request that carries a reply
  holding reasoning and nothing else (HTTP 400 or 422): that reasoning then
  stops going back, in this session and after a resume, and the request is
  sent once more without it. After four such continuations in a row, a
  fifth cut reply ends the turn with `length` (so does a cut on the turn's
  last iteration), and the terminal UI and print mode say which
  `max_tokens` cut it and where it is set (the cut note and the refused
  call name the limit that request carried). Send a follow-up to continue.
  On a Responses connection a reply that reaches the limit is reported
  incomplete and ends the turn as truncated.
  For a reasoning model the limit is also a bound on how long it thinks
  before it acts. Setup leaves it at the default for a Chat Completions
  connection unless the catalog reports a lower output maximum, which it
  saves; for a Responses or native Messages connection it still writes the
  catalog's output maximum when one is reported, and `openmax --check`
  warns while such a connection runs at the default. To raise it, set
  `max_tokens` on the model's providers.json entry (or in settings.json),
  up to the model's longest reply: the input budget holds back at most the
  larger of 4096 and an eighth of the window for it, and a request asks for
  more only when its prompt leaves the room. A 4096 counts as chosen only on
  the model's entry: every settings save writes 4096 to settings.json when
  the limit is unset. `openmax --check --all` states the active model's
  limit and where it is set.
- `compaction_tokens` sets a soft input ceiling for automatic compaction. Open
  Max uses the explicit value in `settings.json` first, then the selected
  model's value in `providers.json`, then a default that follows the
  model's context window: 150,000 tokens, or a quarter of the window when
  that is larger (262,144 for a 1,048,576-token window). Fewer cuts on a
  large window mean fewer re-reads of removed context and fewer requests
  that re-send the whole context uncached; the price is a larger context on
  every request while no cut fires. To keep more or less
  history on a model, set the cap in that model's entry and leave the
  global setting unset.
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
  them out. xAI's grok-4.x models (`api.x.ai`, or a model id starting
  `grok-4.` behind a proxy or gateway) take `low`, `medium`, `high`, and
  `xhigh` plus Default on every wire, as xAI documents; their reasoning cannot
  be disabled and the API refuses `none` and `max` only after the request is
  spent, so Open Max refuses every other level before sending, naming the
  levels the route takes, and the picker and shortcut offer only those.
  Elsewhere supported levels depend on the model and endpoint;
  the server validates explicit selections. A rejection is reported without
  retrying at another level. Sampling and protocol settings are unchanged.
  The Anthropic compatibility endpoint ignores effort, so explicit selections
  there are refused before saving or sending. Responses and native Messages
  render that field ahead of the conversation, so changing it mid-session
  makes the whole history a prompt cache miss once. Where the model takes an
  effort change inside the conversation (GPT-6 and later over Responses, not
  on a ChatGPT plan; Claude Opus 5 and 5.5, Sonnet 5.5, Haiku 5.5, Fable 5.1
  and Mythos 5.1 over native Messages), the field keeps the first effort the
  session ran at on that route and each later choice goes just ahead of the
  prompt of the turn it was made for: a `configuration_update` input item, or
  an effort-only system message under the
  `mid-conversation-output-config-2026-07-01` beta header. Each prompt
  records its turn's effort and route (endpoint, model and credentials,
  hashed) in the session file, and only prompts the current route ran and
  answered count, so an effort another model took or the server refused is
  never sent again. A return to Default, which no such change can name,
  sends the field as before. See the [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)
  and [compatibility documentation](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk).
- `max_output_bytes` (default 16384 with no known window, floor 1000): per
  tool-result cap. Unset, it follows the model's window: three eighths of a
  byte per token of `context_tokens`, between 16384 and 49152, so 16384 up
  to a window of about 43,700 tokens and 49152 from 131,072 tokens up. A
  value set here is the cap at any window, past 49152 too. `read_file`
  returns up to this cap held between 24,000 and 49152 bytes, and ends a
  read it cuts with the offset to continue from. A result that fits the
  cap is sent whole. One over it spills up to 16 MiB of each
  stream to a log in `~/.openmax/cmd-logs` that the result names, and is
  sent as a preview a quarter of the cap's size when the log holds all the
  output. An incomplete or missing log keeps the full cap. Bash keeps up to
  a quarter of each stream's share from its start and the rest from its
  end, and marks the cut with the bytes it dropped. Every later request
  re-sends a result, and the next call after a flood narrows it, so the
  preview is sized to steer that call. Before any of that, bash keeps only
  the final text of a line redrawn with carriage returns
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
- `max_agent_iterations` (default 300): tool-call rounds one turn may take.
  A turn that reaches it stops and says so; a follow-up message (or
  `openmax --continue -p "<follow-up>"`) carries on from there. Settings are
  read at start, so a raised cap applies from the next `openmax` run.
  `--max-agent-iterations` and `--max-agent-tokens` set either cap for one
  `-p`, `--stdio` or `--subagent` run without writing it here (see
  [usage](usage.md#command-line)).

The input budget for the transcript and frozen tool schemas is
`context_tokens - (reply_reserve + 1024)`, leaving room for the reply and 1024
tokens of headroom, where

    reply_reserve = min(max_tokens, max(4096, context_tokens / 8))

with `max_tokens` after its clamp above. Each request carries the reply limit

    limit = clamp(context_tokens - 1024 - prompt, reply_reserve, max_tokens)

where `prompt` is the calibrated estimate of that request in the server's
tokens. So the limit each request carries and the room held back from the
input budget are two numbers: set to a model's longest reply, `max_tokens`
no longer holds that much input back, and a model whose longest reply is as
large as its window keeps seven eighths of it for input instead of about a
thousand tokens. A reply that reaches the limit its request carried is cut
as at any limit. A request that asked for more than the reserve and is
refused for length goes once more in the same turn at the reserve, and each
refusal for length calibrates against the limit its request carried.
Open Max estimates the input at about four bytes per token and scales the
estimate up when the server reports more prompt tokens than estimated or
refuses a request for length that the estimate said fit.
Calibration stays in memory, so a resumed session starts on the plain
estimate.

Open Max estimates the current input against the selected `compaction_tokens`
value, with a 20,000-token floor on that value. It calibrates this soft trigger
with the prompt count from the same route. A lower
reported count can allow more history within the conservative input budget.
This is a best-effort estimate because providers may count prompt tokens
differently. A normal turn request that fails or returns no usable prompt
count discards a lower calibration.

The input budget always bounds the trigger. With the defaults before
calibration, a 128,000-token window triggers at 122,880 tokens, a
500,000-token window triggers at 150,000 tokens, and a 1,048,576-token
window triggers at 262,144 tokens. If the frozen schemas or the
messages a cut must keep make the target unreachable, automatic compaction
falls back to the input budget. The `/compact` command makes the same cut
against the calibrated trigger and uses the input budget if the frozen
schemas alone fill the trigger.

A compaction is one cut. It keeps, byte for byte, the system prompt, the
opening of the conversation through your original request, the prompt of the
turn in progress, and the newest exchanges, up to a quarter of the trigger
and at most 20,000 tokens with the note in their place. It removes everything
between, and never shortens a kept message, so between cuts the history stays
byte-stable and the server can reuse its prompt cache. Before the cut, the
session's own model writes the notes it will continue from: one extra
request whose prompt is the conversation up to the cut, as earlier requests
already sent it, with the same tools, plus one ask. The notes, up to 4,000
bytes, take the removed span's place in one context note. A call in that
reply never runs, and nothing else of the exchange is kept. When no notes
come back (an error, an empty reply, a timeout, or a request
`max_agent_tokens` refuses), automatic compaction makes no cut until the
window forces one, and does not ask again in that turn before then. A cut the
window forces, or a `/compact`, is made without notes when none come back,
and its note lists the tools used, the files touched and the earlier goals
instead. Everything a cut removes is appended verbatim to
`~/.openmax/sessions/<id>.archive.jsonl`, and the context note names that
path, so compaction stays reversible: the notes are the bounded view, the
archive is the lossless record. The `compaction` hook observes each cut.

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
approval detail is in `openmax --spec tools`, `hooks`, and `permissions`. The
tools and hooks contracts state each mode's rule and leave what applies only
outside auto to `openmax --spec tools.approval` and `hooks.approval`, which
they name, so a session in auto never re-sends it.

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

Add `"compaction_tokens": 220000` to a model object to set that model's
compaction cap. Without one, the cap follows that model's `context_tokens`:
150,000 tokens, or a quarter of the window when larger. See [context window and compaction](#context-window-and-compaction).

Set `"provider"` in settings, use the provider CLI option, or use `/provider <name>`
when you only want to change the endpoint. Bare `/provider` opens the searchable
model picker. Optional `compat` flags cover picky
gateways (for example `max_completion_tokens` versus `max_tokens`).
`compat.omit_temperature: true` leaves sampling temperature to that provider;
it defaults to false and leaves other providers' settings unchanged.
`compat.prefer_reasoning_details: true` omits plaintext reasoning when the
unchanged structured reasoning blocks contain the same text. OpenRouter's Chat
Completions endpoint uses this format automatically. Enable it for a proxy
only if it accepts structured reasoning blocks on their own.

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
`additional_tools` input item that carries the tools. A Responses server
whose input has no such item refuses every request that carries one, so
there the tools go in the request's `tools` field, which every Responses
server takes. One such server is known, by its host or by its model id
(after any `vendor/` prefix, so behind a proxy or gateway too): `api.x.ai`,
whose models are `grok-*`. Chat Completions
requests do not carry the key as a field: the servers that implement that
protocol vary, and some refuse a field they do not know. A provider that keeps
its prompt cache per server and routes a conversation to one server by a
header gets the key in that header instead, known by its host or by its model
id (after any `vendor/` prefix), so a request through a proxy or gateway in
front of it carries the header too. One such provider is known: `api.x.ai`,
whose models are `grok-*`, by `x-grok-conv-id`. Without it, a request can meet
a server that never saw the conversation and prefill the whole history again
at the uncached price. A provider whose own `headers` set that header keeps
its value. Native Messages does not define the key.

A Chat Completions route that caches a model's prompt only up to a
`cache_control` marker reads nothing back from a request without one, so
every request would pay its whole prompt at the uncached price. One such route
is known, by its host and the model id: `openrouter.ai` serving an Anthropic
model (`anthropic/...`). Each request there carries two markers, as on native
Messages: one on the system prompt, which caches the tools and the system
prompt every session shares, and one on the newest stored message, which the
next request reads back at the cache price. The system prompt and a user
message carry theirs on their text, sent as one text part; a tool result or
an assistant reply carries its own beside its role, a note stored on it
included.

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
checks. Messages caches a prompt only up to a `cache_control` marker, so every
request carries two: one on the system prompt, which caches the tools and the
system prompt every session shares, and one on the newest block, which the
next request reads back at the cache price. Signed
thinking blocks return only to their producing endpoint, model, and
credentials, with unchanged prior system, tools, and messages, and matching
assistant text and tool calls. Compaction, reloads, or edits that change this
context discard stale thinking while retaining the current
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

The note (the tool files, skills, and memories a session added, changed, or
removed after its prompt and tools array were built) opens with a header that
names the harness as its source and says the latest note replaces any earlier
one. It is stored in the transcript when it changes, never moved or rewritten:
appended to the content of the newest tool result, after the receipt of the
write that changed it, or of the user's prompt at the start of a turn, after a
blank line, and only after an assistant reply as a user message of its own.
Every later request repeats it as history, so every protocol's prompt cache
reads it back like any other stored byte, and native Messages keeps replaying
the thinking a reply made after it. Every endpoint, a flat `base_url` or a
named provider, therefore sees the stored transcript's role order and the
system prompt as the only system message. A message of its own after a tool
result is what picky Chat Completions servers refuse: some hosted validators
reject a system or a user message after a tool message, a chat template that
enforces user/assistant alternation rejects a user message after the prompt,
and some gateways hoist every system message into the system prompt, which
would move the cached prefix on every capability write. A prune that drops or
shortens the stored note stores it again, and a session that changes nothing
stores none. When nothing differs from the prompt any more, the header alone
is stored.

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
silent after reply text is reported truncated, where one whose connection
drops after reply text is asked for again, at most twice in a row. A provider's
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
the model's documentation. For a Chat Completions connection setup writes
no `max_tokens`, so the model runs at the reply limit settings.json gives
(4096 by default), unless the catalog reports an output maximum under that
limit, which it saves as the model's limit. For a Responses or native Messages connection, when the
catalog reports the model's output maximum (`max_completion_tokens`,
`max_output_tokens`, `max_tokens`, or `top_provider.max_completion_tokens`),
setup writes the model entry's `max_tokens`: that maximum, capped at a
quarter of the window (the cap is never under the 4096 default, so a model
whose own maximum is lower gets that maximum). A catalog that reports no maximum gets no guess,
and `openmax --check` then warns that the model runs at the default. Setup
never replaces a `max_tokens` the entry already has. An endpoint without a
model-list route can be configured with an exact model ID; setup reports
that access was not verified.
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
global `context_tokens` fallback still applies; verify it for the selected model.
Refreshing the catalog retains a manually supplied window when the service
omits one and preserves the model's configured `compaction_tokens`.
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
