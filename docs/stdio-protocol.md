# stdio protocol (`openmax-stdio/5`)

`openmax --stdio` speaks line-delimited JSON both ways, so any process that
reads and writes JSONL (an editor plugin, an orchestrator, another openmax) can
drive a full interactive session. This is the stable contract for custom
frontends and interop adapters. Validate a stream against it with
`openmax --check --stdio`, which reads JSONL on stdin, reports each line, and
exits nonzero on any violation. `openmax --spec stdio` prints the same
contract from the binary itself.

This file is the normative reference for every field of every line.

## Handshake

The first stdout line is:

```json
{"type":"hello","proto":"openmax-stdio/5","protocol_version":5,"session_id":"...","version":"0.2.0","project":"/abs/path","continued":false}
```

`protocol_version` is an integer a client compares directly; `proto` carries
the same major as a readable id. Any wire change bumps both. `continued` is
true when `--continue` resumed a prior session; the next line is then one
`transcript` object so the client can render history without synthetic events:

```json
{"type":"transcript","session_id":"...","messages":[{"role":"user","content":"..."}],"truncated":false}
```

`messages` carries user and assistant text only (tool traffic and the system
prompt are session internals), each message capped at 4096 characters and the
whole line at 256 KiB of content; `truncated` says whether anything was cut.
The session file on disk remains the full record. No live events are replayed,
so a `token` stream always means a running turn.

## Commands (stdin)

One JSON object per line.

| Command | Fields | Effect |
| --- | --- | --- |
| `user` | `text` | Start a turn with the text |
| `approve` | `approval_id`, `approved` (bool) | Answer a pending approval |
| `approval_mode` | `mode` (`auto`, `ask`, or `readonly`) | Set and persist the approval gate; answered by an `approval_mode` line, or a `protocol_error` naming the legal values (on a save failure the mode is unchanged and the error says so) |
| `reload` | none | Re-freeze tools, skills, and prompt from current config; answered by `refrozen`, or `protocol_error` while a turn is in flight |
| `cancel` | none | Cancel the running turn |
| `quit` | none | Drain the in-flight turn, then exit |

Unknown `cmd` values are protocol errors; extra fields on a known command are
ignored; blank lines are skipped; EOF behaves like `quit`.

## Events (stdout)

Every event line carries the flattened `session_id`, then a `type`
discriminator, then its fields. openmax emits keys in that order, but object
key order is not significant: parse every line by field name. Only two lines
carry no `session_id`: `protocol_error` and the `approval_mode`
acknowledgement (`hello` and `transcript` are not events either, but carry
one).

| `type` | Fields |
| --- | --- |
| `token` | `text` |
| `thinking` | `text` |
| `message_done` | `text` |
| `budget` | `used_tokens` (estimated: transcript plus the frozen tool schemas re-sent every request), `context_tokens` |
| `usage` | `prompt_tokens`, `completion_tokens`, `cached_tokens` (or null) |
| `tool_start` | `call_id`, `name`, `args` (object) |
| `tool_end` | `call_id`, `ok` (bool), `output` |
| `harness_note` | `call_id`, `text` (a note the harness wrote into the model's transcript: a refreeze receipt, or a policy, providers, settings, or approval notice. `call_id` links it to the tool result it rode; it is empty for a note inserted before the next prompt, such as a turn-start receipt) |
| `diff` | `call_id`, `path`, `diff`, `added`, `removed` |
| `approval_request` | `approval_id`, `name`, `summary`, `detail`, `reason` (`gate`, or `unapproved_source` which unattended clients must never auto-approve), `source_path`, `source_sha`, and optional `env` (see below) |
| `approval_settled` | `approval_id`, `outcome` (`approved`, `declined`, `timed_out`, or `cancelled`) |
| `refrozen` | `tools`, `skills`, `changes` (the refreeze receipt: what changed and who) |
| `schemas_over_budget` | `schema_tokens`, `budget_tokens` (the installed tool schemas take most of what the window can spend, so compaction runs early against what little is left; once `schema_tokens` reaches `budget_tokens` it stops entirely, since pruning cannot pay a fixed per-request cost. Advisory, at most once per session; the turn still runs) |
| `compacted` | `tokens_before`, `tokens_after`, `compacted_messages` (the receipt of a forced compaction; `compacted_messages` of 0 means the transcript was already at or under the prune target and nothing changed) |
| `hook_failed` | `hook`, `event`, `detail` (a hook did not run: an observe-only hook failed, or a hook file on disk is not loaded; the turn proceeded) |
| `turn_refused` | `hook`, `reason`, `continuation`, `continuations_left` (a blocking `turn_end` hook refused the model's completion and the harness honored it; see below) |
| `done` | `stop_reason` |
| `error` | `message` |

Example event line:

```json
{"session_id":"s1","type":"tool_start","call_id":"c1","name":"read_file","args":{"path":"a.rs"}}
```

`approval_request.env` is the list of environment variable names the approved
tool will receive (its manifest's `env` allowlist): a credential grant. It is
omitted from the wire when empty, so a `gate` request and a tool that forwards
nothing carry no `env` key. When present, render it on its own line that the
card never clips; approving secrets a narrow terminal hid behind other detail
is the failure the field exists to prevent, so do not fold it into `detail`.

`turn_refused` means the turn continues after a `message_done` without a
`user` command from the client. `reason` is already in the transcript as a user
message, written to disk before this event goes out, so render it: otherwise
the live view shows the model finishing and starting again with no visible
cause, while a replay of the same session from disk shows the injected message.
`continuation` and `continuations_left` are the numbers the hook's payload
carried for this attempt: refusals honored before this one, and how many the
harness had left to honor.

## Turn guarantees

Every `user` command is answered by exactly one `done`, and `done` is the only
guaranteed terminator: never block waiting for another event. On a normal turn
a run of `token` deltas is terminated by one `message_done`, but a turn that
hits a provider-stream error emits an `error` line and then `done` with no
`message_done`. A turn that dies unexpectedly reports `error` and then `done`
with `stop_reason` `error`, so a crash is an event rather than a silent stall.
A stream the provider abandons mid-answer still emits `message_done` (with the
partial text, which is kept in the session), then an `error` line, then `done`
with `stop_reason` `truncated`: an incomplete answer is never reported as a
finished one. No tool call carried by such a stream is dispatched, even one
whose arguments parse, because a stream with no completion signal never said
which calls the model meant to make.

A `user` command that starts no turn still terminates. Empty text, or a project
that is not trusted, yields `{"type":"protocol_error","message":"..."}`
followed by `done` with `stop_reason` `refused`, so a client that blocks on
`done` is never stuck waiting on a prompt nothing will answer.

| `stop_reason` | Meaning |
| --- | --- |
| provider `finish_reason` | Passed through verbatim on a normal turn, commonly `stop` or `length`. Treat any unlisted value as a normal end |
| `truncated` | The provider stream ended with no completion signal; the reply is incomplete, any tool calls it carried were refused, and an `error` line precedes it |
| `max_iterations` | The turn hit the tool-call ceiling |
| `budget_exhausted` | The per-turn `max_agent_tokens` cap refused the next request at admission; nothing was sent, and resubmitting continues the work |
| `unverified` | A blocking `turn_end` hook refused the completion more times than the harness honors (8), or its refusal could not be persisted; the reply stands unverified |
| `blocked` | A `user_prompt_submit` hook refused the prompt |
| `cancelled` | A `cancel` command or shutdown stopped the turn |
| `error` | The turn failed; an `error` line precedes it |
| `refused` | The command started no turn; a `protocol_error` precedes it |

The one case with no `done` is a `user` sent while a turn is already in flight:
it is refused with a `protocol_error` alone, because the running turn owns the
next `done` and a second one would report that turn as finished.

The process exit code is 1 when the last turn ended with `stop_reason`
`error`, 2 when `--continue` finds no prior session in the directory, and 0
otherwise, including `max_iterations`, `budget_exhausted`, and `unverified`:
over this wire the client reads `done` and decides for itself, unlike `-p`,
which exits 4 on those three.

Bad input leaves the session unharmed. A line that is not valid UTF-8, or one
longer than 8 MiB, is reported as a `protocol_error` and skipped; the reader
resynchronizes on the next newline and keeps going.

While a client is live, an `approval_request` is forwarded and openmax waits for
an `approve`; after `quit` or EOF, pending and later approvals are declined so
shutdown drains promptly. An `approve` naming an id that was never issued or
has already settled is answered with a `protocol_error` instead of being
silently dropped: the client's picture of open gates is wrong and should say
so. A successful `approval_mode` command is acknowledged with
`{"type":"approval_mode","mode":"..."}` after the setting is persisted.

`reason` `unapproved_source` is the human content boundary: the first call of
an external (agent-writable) tool whose exact bytes, the manifest or the
project-local code it runs, no human has approved. Every external tool is
gated, whatever its `mutating` field says, because the harness spawns it as a
native host process with openmax's own authority. Such a request carries
`source_path` (project-relative where possible) and `source_sha` (first 12 hex
chars of the file's sha256); a client that cannot prompt a human must surface
`openmax --approve <source_path>` rather than decline silently. Both fields are
empty strings when `reason` is `gate`.

## What changed in `openmax-stdio/5`

`harness_note` is new. Before it, the receipts and notices the harness writes
into the model's transcript (the refreeze receipt, the permission, providers,
settings, and approval notices) were visible only to the model; a frontend saw
a bare `tool_end` and could not render, for example, that a written tool did
not load or that an approval was revoked. A `/4` client never saw these lines,
so ignoring `harness_note` is safe and loses only that surfaced text.

## What changed in `openmax-stdio/4`

`turn_refused` is new. A client written for `/3` had never seen a turn continue
after `message_done` without its own `user` command; under a blocking
`turn_end` hook that is now a normal turn shape, and this event is the only
line that says why. The `unverified` stop reason came in with that gate and is
listed above.

`approval_request.env` was added under `/4` without a bump. It is additive and
omitted when empty, so every request that grants no environment is
byte-identical to its earlier form.

## What changed in `openmax-stdio/3`

`approval_request` gained two required fields, `source_path` and `source_sha`,
described above. They are required rather than optional, so a client written
against `/2` fails to decode a `/3` request; that break is the reason for the
bump.

`budget.used_tokens` counts the frozen tool schemas that ride on every
request, not the transcript alone. The field's name and type are unchanged, but
its value is larger (a zero-extension session reports ~1270 where `/2` reported
~720) and it is exactly the total compaction enforces against
`context_tokens`. A client that calibrated its own thresholds against the `/2`
meaning must re-calibrate them; one that only renders the ratio needs no change.

`schemas_over_budget` and `compacted` are new and additive: a client that
ignores unknown types is unaffected, but while `schemas_over_budget` holds,
`used_tokens` has a floor pruning cannot go below, and once `schema_tokens`
reaches `budget_tokens` compaction stops entirely.
