# stdio protocol (`openmax-stdio/8`)

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
{"type":"hello","proto":"openmax-stdio/8","protocol_version":8,"session_id":"...","version":"0.2.0","project":"/abs/path","continued":false}
```

`protocol_version` is an integer a client compares directly; `proto` carries
the same major as a readable id. Any wire change bumps both. `continued` is
true when `--continue` resumed a prior session; the next line is then one
`transcript` object so the client can render history without synthetic events:

```json
{"type":"transcript","session_id":"...","messages":[{"role":"user","content":"..."}],"truncated":false}
```

`messages` carries user and assistant text only (tool traffic and the system
prompt are session internals), each message capped at 4096 characters, and no
message is added once the line holds 262,144 characters of content;
`truncated` says whether anything was cut.
The session file on disk remains the full record. No live events are replayed,
so a `token` stream always means a running turn.

A continued session must have a readable session index and transcript and no
other writable owner. Failure is reported on stderr with a nonzero exit before
the handshake.
The process retains ownership while waiting for commands, including between turns.

## Commands (stdin)

One JSON object per line.

| Command | Fields | Effect |
| --- | --- | --- |
| `user` | `text` | Start a turn with the text; sent while a turn runs, steer that turn instead (see below) |
| `approve` | `approval_id`, `approved` (bool) | Answer a pending approval |
| `approval_mode` | `mode` (`auto`, `ask`, or `readonly`) | Save the execution mode for this trusted project; answered by an `approval_mode` line, or a `protocol_error` naming the legal values (on a save failure the mode is unchanged and the error says so) |
| `reload` | none | Re-freeze tools, skills, and prompt from current config, rebuilding the system prompt and the tools array so the trailing note's changes move into them (the one command that starts a new prompt prefix); answered by `refrozen`, or `protocol_error` while a turn is in flight |
| `fork` | none | Branch the session at its end into a new session and drive that one from now on; answered by `forked`, or `protocol_error` while a turn is in flight or before the session holds any message |
| `rewind` | `turn` (integer, 1-based) | Branch the session at the end of turn `turn` (counted as below) and drive the branch from now on; answered by `forked`, or `protocol_error` while a turn is in flight or when the session has no such turn |
| `cancel` | none | Cancel the running turn |
| `quit` | none | Drain the in-flight turn, then exit |

Unknown `cmd` values are protocol errors; extra fields on a known command are
ignored; blank lines are skipped; EOF behaves like `quit`.

`fork` and `rewind` never change the session they leave, and never touch
files: a rewind branches the conversation, so an edit the agent made after
the cut stays on disk. The new session holds that session's first
`message_index` messages, its frozen tools, skills, and prompt, and its
compaction records, so its next request carries exactly that history. A
compaction summary in that history names the archive of the messages it
moved out, which stays the original's file: the branch shares it rather
than copying it, so deleting the original deletes it. It is answered by:

```json
{"type":"forked","session_id":"...","parent_id":"...","message_index":3}
```

and every later line carries the new `session_id`. The session left behind is
released, so `/resume` in the TUI or another process can continue it. On a
`protocol_error` the session driven is unchanged; one sent after the branch
was made names it, since `/resume` lists it and a retry would make another.

Turn `n` ends where the `n`th turn whose prompt stayed in the transcript
sent its `done`: the harness records that point once the turn's transcript
is saved. A user message the harness writes during a turn (a completion
gate's refusal, the tool-schema budget note) therefore stays in that turn.
A prompt refused before it ran (an input gate's block, an unresolvable
endpoint) leaves nothing and is no turn, so a client that counts `done`
lines skips those. A turn whose process died before it ended, or whose
closing save failed, counts as one with the next turn that ends, and as the
last turn while none has. The turns a compaction folded into its summary
end together, just after the summary. A session whose index entry records
no turn ends (saved before this version, or rewritten by one) counts one
turn per run of user messages instead, which puts a turn boundary at each
such harness message; it keeps that count for every turn up to and
including the first one that ends under this version, and counts exactly
from then on.

## Events (stdout)

Every event line carries the flattened `session_id`, then a `type`
discriminator, then its fields, and is byte for byte the line
`openmax --print --json` writes for the same event. Every other line starts
with `type`, then its fields in the order this page shows. openmax emits keys
in those orders, but object key order is not significant: parse every line by
field name. Only two lines carry no `session_id`: `protocol_error` and the
`approval_mode` acknowledgement (`hello`, `transcript`, and `forked` are not
events either, but carry one).

| `type` | Fields |
| --- | --- |
| `token` | `text` |
| `thinking` | `text` |
| `message_done` | `text` |
| `budget` | `used_tokens` (estimated: transcript plus the frozen tool schemas and the trailing note re-sent every request, scaled up when the server reported more prompt tokens than estimated for the session's last request, or refused a request for length (recognized by the error's wording) that the estimate said fit), `context_tokens` |
| `usage` | `prompt_tokens`, `completion_tokens`, `cached_tokens` (or null) |
| `tool_start` | `call_id`, `name`, `args` (object) |
| `tool_end` | `call_id`, `ok` (bool), `output` |
| `harness_note` | `call_id`, `text` (a note the harness wrote into the model's transcript: a refreeze receipt; an execution policy, policy, permission, hook, providers, settings, or approval notice; or the repeated identical call note. `--print` without `--json` prints no line for it. `call_id` links it to the tool result it rode; it is empty for a note inserted before the next prompt, such as a turn-start receipt) |
| `retry` | `attempt`, `max_attempts`, `reason` (the model request is being resent: `attempt` is the one that goes out after a backoff wait, and `max_attempts` the one the request ends on if it keeps failing the same way, `reason` having ended the previous one with a transport failure, a 429 or transient server error (500, 502, 503, 504, 529), or, before any reply text arrived, a stream that died, went silent, or carried a data line the client cannot read, or a stream or reply the provider failed with a rate limit, an overload, or a server fault. The wait is the backoff, or the response's `Retry-After` in seconds when that is longer. A `Retry-After` of more than a minute is not retried: the request fails at once and the error names the server's wait. Neither is a 429 reporting an exhausted quota (`insufficient_quota`). Emitted before the wait; a `cancel` during it ends the turn and the attempt never goes out. `thinking` lines already emitted for that attempt are void and the reply starts over; no `token` line precedes a retried stream. A request gives up before its budget after three attempts in a row that failed to connect or timed out: while that streak runs, `max_attempts` counts to its third attempt (or to the budget, if that comes first), and it returns to the budget when an attempt ends any other way) |
| `diff` | `call_id`, `path`, `diff`, `added`, `removed` |
| `approval_request` | `approval_id`, `name`, `summary`, `detail`, `reason` (`gate`, or `unapproved_source` which unattended clients must never auto-approve), `source_path`, `source_sha`, and optional `env` (see below) |
| `approval_settled` | `approval_id`, `outcome` (`approved`, `declined`, `timed_out`, or `cancelled`) |
| `refrozen` | `tools`, `skills`, `changes` (the refreeze receipt: one line per tool or skill file added, modified, or removed) |
| `schemas_over_budget` | `schema_tokens`, `budget_tokens` (the frozen tool schemas and the trailing note take most of what the window can spend, so compaction runs early against what little is left; once `schema_tokens` reaches `budget_tokens` it stops entirely, since pruning cannot pay a fixed per-request cost. Advisory, at most once per session; the turn still runs) |
| `compacted` | `tokens_before`, `tokens_after`, `compacted_messages` (the receipt of a forced compaction; `compacted_messages` of 0 means the transcript was already at or under the prune target and nothing changed) |
| `hook_failed` | `hook`, `event`, `detail` (a hook did not run: an observe-only hook failed or a hook file on disk is not loaded, and the turn proceeded; or, in auto, a `user_prompt_submit` gate could not start and did not check the prompt, or a blocking `turn_end` gate could not start and the turn ends `unverified`) |
| `turn_refused` | `hook`, `reason`, `continuation`, `continuations_left` (a blocking `turn_end` hook refused the model's completion and the harness honored it; see below) |
| `steer_accepted` | `text` (a `user` line sent while the turn ran entered the transcript as a user message here; see below) |
| `steer_blocked` | `text`, `reason` (a `user_prompt_submit` gate refused a steer, which entered nothing; see below) |
| `done` | `stop_reason` |
| `error` | `message` |

Example event line:

```json
{"session_id":"s1","type":"tool_start","call_id":"c1","name":"read_file","args":{"path":"a.rs"}}
```

Mode selection uses the same canonical project choice as the TUI and survives
restart. Agent-spawned clients cannot change it. In `auto`, the core emits no
approval requests, including for new extension content or permission `ask`
rules. Deny rules and validation still apply. A request emitted while in `ask`
remains pending until answered or cancelled, even if the mode then changes.

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

A `user` line sent while a turn runs steers it. A `/name args` line is
expanded as a prompt template first, as a turn's first prompt is, and the
expansion is the steer from then on: what the gate judges, what enters the
transcript, and the `text` of the `steer_accepted` or `steer_blocked` that
answers it. Match each answer to its steer by that `text`: the answers to
steers sent together need not come in the order sent, since a
`steer_blocked` goes out as soon as its gate refuses, and the
`steer_accepted` lines only once every steer waiting at that boundary has
been judged and the allowed ones saved.

The steer passes the same `user_prompt_submit` gate as a turn's first prompt,
then enters the transcript as a user message at the turn's next iteration
boundary: after the reply or tool results above it and before the next
request is built, so that request carries it. `steer_accepted` marks the
position and goes out after the save that records the message; when that
save fails, its own `error` line goes out first, and the steer still goes to
the model with the transcript and still gets its `steer_accepted` (the next
save that lands writes it). An accepted steer is a saved user message, like
the turn's own prompt: when the turn ends before the model answers it, it
stays in the transcript at that position and the next turn's first request
carries it, so it is not sent again. That holds even when the request
carrying it never goes out: a `cancel`, `budget_exhausted` (the
`max_agent_tokens` cap refusing that request at admission), or an `error`
from the compaction that runs ahead of it. Steers sent together are appended in
the order they were sent. A model that stops answering while a steer waits is
sent round again to read it, unless the turn is on its last iteration
(`max_agent_iterations`), which ends on that answer.

A steer the turn ends without taking gets neither answer, never reached the
model, and may be sent again as its own `user` line after the turn's `done`:
one waiting when the turn is cancelled, fails, runs out of budget, or reaches
that last iteration, and one sent after the turn took the model's final
answer (while a `turn_end` hook judges an answer it lets stand, or as the
turn saves and settles). None of those got a `protocol_error` when sent:
only the missing answer says the turn did not take it. A steer blocked by the
gate gets a `steer_blocked` instead, carrying its `text` and the gate's
`reason`, and enters nothing: an answer the model already finished stands,
and no request is sent for it. Hand the text back to the user to edit;
sending it again as a `user` line runs the gate on it a second time.

## Turn guarantees

For an admitted turn, `done` is emitted after the core releases its running
state and cancellation token. A client may submit the next turn immediately.

Every `user` command sent while no turn runs is answered by exactly one
`done` (one sent while a turn runs steers it and gets none of its own; see
below), and `done` is the only guaranteed terminator: never block waiting for
another event. On a normal turn a run of `token` deltas is terminated by
one `message_done`, but a turn that
hits a provider-stream error, including a failure the provider reports inside
the stream after some `token` lines, emits an `error` line and then `done`
with no `message_done`; that partial text is not kept. A turn that dies
unexpectedly reports `error` and then `done` with `stop_reason` `error`, so a
crash is an event rather than a silent stall. A stream that ends mid-answer
with no completion signal still emits `message_done` (with the partial text,
which is kept in the session), then an `error` line, then `done` with
`stop_reason` `truncated`: an incomplete answer is never reported as a
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
| `truncated` | The provider stream ended with no completion signal, or at a data line the client cannot read (after reply text had streamed, or on the last retry), or exceeded a client limit; the reply is incomplete, any tool calls it carried were refused, and an `error` line precedes it |
| `max_iterations` | The turn hit the tool-call ceiling |
| `budget_exhausted` | The per-turn `max_agent_tokens` cap refused the next request at admission; nothing was sent, and resubmitting continues the work |
| `unverified` | A blocking `turn_end` hook refused the completion more times than the harness honors (8), its refusal could not be persisted, or in auto it could not start; the reply stands unverified |
| `blocked` | A `user_prompt_submit` hook refused the prompt |
| `cancelled` | A `cancel` command, a signal, or shutdown stopped the turn |
| `error` | The turn failed; an `error` line precedes it |
| `refused` | The command started no turn; a `protocol_error` precedes it |

The one case with no `done` is a `user` sent while a turn is already in flight:
it steers that turn, and the running turn owns the next `done`, so a second
one would report that turn as finished. One with empty text, one that would
make the steers still waiting for the turn pass 8 MiB together (send it again
after the next `steer_accepted`, `steer_blocked`, or `done`: the turn's next
boundary takes every steer waiting there at once, and its end drops them), or
one that arrives after the turn stopped taking steers, as it settles but
before its `done` (send it again after that `done`), is refused with a
`protocol_error` alone.

The process exit code is 1 when any turn ended with `stop_reason` `error` or
the session could not be opened, 2 on a usage or configuration error or when
`--continue` finds no prior session in the directory, 3 when the project is
not trusted or trust cannot be granted from this process, and 0
otherwise, including `max_iterations`, `budget_exhausted`, and `unverified`:
over this wire the client reads `done` and decides for itself, unlike `-p`,
which exits 4 on those three. SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, or
SIGUSR2 cancels the running turn instead of draining it, declines pending
approvals, and exits 128 plus the signal number once that turn's `done` is
written and, for a new session that nothing was saved to, its index entry is
removed, or after 3 seconds if either has not finished, plus up to 100 ms
to check that the session is empty
and up to 250 ms for a tool command still being started; an entry still in
place then stays, and a warning says so on stderr. A tool command whose start
outlasts those 250 ms is not stopped, and runs on after the exit.
A signal while the session waits on the session index to open, to start a
turn, or to reload, or on the trust store to save an approval mode, ends it
the same way, and a `reload` or `approval_mode` it cuts short gets no answer;
see [usage](usage.md#command-line).

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

## Version history

Any wire change bumps `protocol_version`. What each version changed:

| Version | Change |
| --- | --- |
| 8 | A `user` line sent while a turn runs steers that turn instead of being refused; `steer_accepted` and `steer_blocked` events |
| 7 | `fork` and `rewind` commands and the `forked` line |
| 6 | `retry` event |
| 5 | `harness_note` event |
| 4 | `turn_refused` event |
| 3 | `approval_request` carries `source_path` and `source_sha` |
| 2 | `refrozen` carries `changes`, the refreeze receipt |
| 1 | The first versioned contract |
