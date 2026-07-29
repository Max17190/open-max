# stdio protocol (`openmax-stdio/2`)

`openmax --stdio` speaks line-delimited JSON both ways, so any process that
reads and writes JSONL (an editor plugin, an orchestrator, another openmax) can
drive a full interactive session. This is the stable contract for custom
frontends and interop adapters. Validate a stream against it with
`openmax --check --stdio`, which reads JSONL on stdin, reports each line, and
exits nonzero on any violation.

This file is the normative reference for every field of every line.

## Handshake

The first stdout line is:

```json
{"type":"hello","proto":"openmax-stdio/2","protocol_version":2,"session_id":"...","version":"0.2.0","project":"/abs/path","continued":false}
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
The session file on disk remains the full record.

## Commands (stdin)

One JSON object per line.

| Command | Fields | Effect |
| --- | --- | --- |
| `user` | `text` | Start a turn with the text |
| `approve` | `approval_id`, `approved` (bool) | Answer a pending approval |
| `approval_mode` | `mode` (`auto`, `ask`, or `readonly`) | Set and persist the approval gate; answered by an `approval_mode` line |
| `reload` | none | Re-freeze tools, skills, and prompt from current config; answered by `refrozen`, or `protocol_error` while a turn is in flight |
| `cancel` | none | Cancel the running turn |
| `quit` | none | Drain the in-flight turn, then exit |

Unknown `cmd` values are protocol errors; extra fields on a known command are
ignored; blank lines are skipped; EOF behaves like `quit`.

## Events (stdout)

Every event line carries the flattened `session_id`, then a `type`
discriminator, then its fields. openmax emits keys in that order, but object
key order is not significant: parse every line by field name.

| `type` | Fields |
| --- | --- |
| `token` | `text` |
| `thinking` | `text` |
| `message_done` | `text` |
| `budget` | `used_tokens`, `context_tokens` |
| `usage` | `prompt_tokens`, `completion_tokens`, `cached_tokens` (or null) |
| `tool_start` | `call_id`, `name`, `args` (object) |
| `tool_end` | `call_id`, `ok` (bool), `output` |
| `diff` | `call_id`, `path`, `diff`, `added`, `removed` |
| `approval_request` | `approval_id`, `name`, `summary`, `detail` |
| `approval_settled` | `approval_id`, `outcome` (`approved`, `declined`, `timed_out`, or `cancelled`) |
| `refrozen` | `tools`, `skills`, `changes` (the refreeze receipt: what changed and who) |
| `hook_failed` | `hook`, `event`, `detail` (an observe-only hook failed; the turn proceeded) |
| `done` | `stop_reason` |
| `error` | `message` |

Example event line:

```json
{"session_id":"s1","type":"tool_start","call_id":"c1","name":"read_file","args":{"path":"a.rs"}}
```

## Turn guarantees

Every `user` command is answered by exactly one `done`, and `done` is the only
guaranteed terminator: never block waiting for another event. On a normal turn
a run of `token` deltas is terminated by one `message_done`, but a turn that
hits a provider-stream error emits an `error` line and then `done` with no
`message_done`. A turn that dies unexpectedly reports `error` and then `done`
with `stop_reason` `error`, so a crash is an event rather than a silent stall.

A `user` command that starts no turn still terminates. Empty text, or a project
that is not trusted, yields `{"type":"protocol_error","message":"..."}`
followed by `done` with `stop_reason` `refused`, so a client that blocks on
`done` is never stuck waiting on a prompt nothing will answer.

| `stop_reason` | Meaning |
| --- | --- |
| provider `finish_reason` | Passed through verbatim on a normal turn, commonly `stop` or `length`. Treat any unlisted value as a normal end |
| `max_iterations` | The turn hit the tool-call ceiling |
| `blocked` | A `user_prompt_submit` hook refused the prompt |
| `cancelled` | A `cancel` command or shutdown stopped the turn |
| `error` | The turn failed; an `error` line precedes it |
| `refused` | The command started no turn; a `protocol_error` precedes it |

The one case with no `done` is a `user` sent while a turn is already in flight:
it is refused with a `protocol_error` alone, because the running turn owns the
next `done` and a second one would report that turn as finished.

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
