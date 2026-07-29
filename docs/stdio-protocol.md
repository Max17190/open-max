# stdio protocol (`openmax-stdio/1`)

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
{"type":"hello","proto":"openmax-stdio/1","protocol_version":1,"session_id":"...","version":"0.2.0","project":"/abs/path"}
```

`protocol_version` is an integer a client compares directly; `proto` carries
the same major as a readable id. Any wire change bumps both.

## Commands (stdin)

One JSON object per line.

| Command | Fields | Effect |
| --- | --- | --- |
| `user` | `text` | Start a turn with the text |
| `approve` | `approval_id`, `approved` (bool) | Answer a pending approval |
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
| `refrozen` | `tools`, `skills` |
| `done` | `stop_reason` |
| `error` | `message` |

Example event line:

```json
{"session_id":"s1","type":"tool_start","call_id":"c1","name":"read_file","args":{"path":"a.rs"}}
```

## Turn guarantees

Each turn ends with exactly one `done`, and `done` is the only guaranteed turn
terminator: never block waiting for another event. On a normal turn a run of
`token` deltas is terminated by one `message_done`, but a turn that hits a
provider-stream error emits an `error` line and then `done` with no
`message_done`. Bad input yields `{"type":"protocol_error","message":"..."}`
and leaves the session unharmed. While a client is live, an `approval_request`
is forwarded and openmax waits for an `approve`; after `quit` or EOF, pending
and later approvals are declined so shutdown drains promptly.
