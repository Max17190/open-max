# Extending Open Max

With nothing installed, extensions cost zero tokens. Project paths win over
global ones on name collision.

## Tools

A TOML file in `.openmax/tools/` or `~/.openmax/tools/`. The harness runs
`command`, writes JSON args to stdin, and returns stdout. These native
processes inherit the host filesystem, environment, credentials, and network
access of Open Max.

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

`mutating` is trusted metadata for scheduling and approval behavior. It is not
a security boundary and does not restrict what the command can do. Unknown keys
in a tool file are rejected, so a misspelled `mutating` surfaces in
`openmax --check` instead of silently taking the tool out of the approval gate.

`params` must declare `type = "object"`, and every `properties` entry must be
an object. The serialized schema is capped at 4096 bytes because it lives in
the frozen prompt prefix and is paid on every request; an oversized schema is
rejected, not truncated. At most 64 external tools load (the name-sorted head);
the prompt trailer reports how many were left out and `openmax --check` names
the files. Hooks are capped at 32 per event the same way.

## Skills

A directory with `SKILL.md` under `.agents/skills/` or `~/.openmax/skills/`.
Only `name` and `description` live in the prompt; the model reads the full file
when needed.

```
.agents/skills/release/SKILL.md
---
name: release
description: How to cut a release of this project
---
Full instructions, checklists, commands...
```

## Memory

One durable fact per markdown file under `.openmax/memory/`, written with the
ordinary file tools. The file stem is the memory's name (1-64 chars of
`[a-z0-9-]`), the first non-empty line is its description, and the body loads
only on demand. At session creation the live memories are ranked by ACT-R
base-level activation (each past access at age `t` hours contributes
`t^-0.5`; activation is the log of the sum, so one recall revives an old
memory) and injected as one index line each under a 1500-byte budget. The
harness appends reads and writes of memory paths to
`.openmax/memory/.access.jsonl`; a memory that would score below one access
21 days old leaves the index, and below one access 60 days old the file is
deleted at the next session creation, leaving a tombstone line (name, sha256,
description) in the log. No memories, no section, zero prompt cost. Full
contract: `openmax --spec memory`.

Everything the harness preserves is also searchable:
`openmax --recall "<query>"` scans this project's session transcripts,
compaction archives, compaction digests, session titles, and memory files,
and prints ranked excerpts, each citing the file that holds the full record.
Ranking fuses BM25 lexical relevance with the same recency law the memory
index uses; `path:<substr>` narrows to history that touched a file,
`k:<n>` and `budget:<tokens>` bound the output, and `--json` emits the
structured report. Read-only, project-scoped, no index to maintain: the
stores on disk stay the single source of truth.

```
.openmax/memory/deploy-port.md
# The staging deploy port is 7443 (set 2026-07-31)
Set in infra/nginx.conf; the health check expects /healthz on the same port.
```

## Prompt templates

A markdown file under `.agents/prompts/` or `~/.openmax/prompts/`; the file
stem becomes a slash command. `$ARGUMENTS` expands to the raw argument string,
`$1`..`$9` to positionals, and `$$` escapes a literal dollar; a template
without placeholders gets the arguments appended. Templates are message
content: re-read on every use, zero prompt tax, never frozen.

```
.agents/prompts/fix-issue.md
---
description: Fix a GitHub issue by number
---
Fetch issue $1 with `gh issue view $1`, reproduce it, fix it, and add a test.
```

Run it as `/fix-issue 42` in the composer, as `openmax -p "/fix-issue 42"`, or
as a stdio `{"cmd":"user","text":"/fix-issue 42"}` line: every front end expands
the same, so a delegated child process gets the project's templates too.

## Hooks

Optional process gates under `.openmax/hooks/` or `~/.openmax/hooks/`.
`pre_tool_use` and `user_prompt_submit` can block (nonzero exit; the blocked
prompt never reaches the model); `post_tool_use`, `session_start` (a session's
first turn), `compaction` (context was pruned; receives the digest record), and
`turn_end` (receives the stop reason, fires even on cancel) observe only. Each
hook gets one JSON payload on stdin.

A `post_tool_use` payload carries the tool result the model saw: `output` (its
first 16 KiB, cut on a character boundary), `output_bytes` (that result's
size), and `output_truncated` (whether the payload dropped part of it). That is
what an eval, audit, or telemetry hook needs to be written as a file instead of
a core feature.

Output is bounded twice and the payload reports both cuts. A tool result is
itself a bounded rendering of what a process printed: `bash` keeps the tail up
to its output cap and names a log file holding the bounded capture. So
`process_bytes` is how many bytes the command actually produced (null when the
tool ran no process, such as the file and search built-ins) and
`process_truncated` says whether the result dropped part of it. A call killed
by timeout or cancel reports the bytes it printed before it died, even though
its result carries none of them. An audit hook can therefore tell a quiet
command from a clipped one without parsing a truncation notice out of the text,
and `output_bytes` still measures the result, because that is the text the
model actually reasoned about.

It stays observation: a failing observe hook (spawn error, nonzero exit, or
timeout) never blocks the turn, but it is not silent either - the harness
emits a `hook_failed` event that the TUI shows as a note and `--json`/stdio
streams carry verbatim. Nothing a hook prints changes what the model receives. Hooks never enter the model prompt and,
like external tools and `bash`, run as native host processes with inherited
filesystem, environment, credentials, and network access.

Unknown keys in a hook file are rejected, and a hook file that a human approved
and that no longer parses blocks every tool call until it is fixed or removed
(fail closed, like permissions): a broken file might have been a gate, and
`openmax --check` prints the reason. A file no human ever approved never ran,
so a broken one is inert instead of blocking - otherwise any write, including
the write that would repair it, could brick the project. Rewriting the
offending hook file (or the code it runs) stays available either way, the same
repair carve-out `permissions.toml` has - including recreating one that was
deleted, since the file the session has to restore is exactly the one that does
not exist. The carve-out resolves the target's parent before checking
containment, so a missing file can be recreated while `../` and symlinked
parents still cannot be aimed outside the project.

## Permissions

Optional rules under `.openmax/permissions.toml` or
`~/.openmax/permissions.toml` (project first). Not in the model prompt; empty
discovery is free. First match wins. Order: hooks pre → permissions →
`approval_mode` → execute → hooks post.

```toml
# .openmax/permissions.toml
[[rules]]
effect = "deny"
tool = "bash"
arg_regex = "rm\\s+-rf"

[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^cargo (test|check|build)"
```

`effect` is `allow`, `deny`, or `ask`. `arg_regex` is optional: command for
`bash`, path for file tools, pattern for `glob`/`grep`. For custom tools it
matches the full serialized JSON arguments. Omit `arg_regex` (or leave it
empty) to match every call of that tool.

`tool` is matched by exact name, so a rule naming a tool that does not exist
never fires: a misspelled `deny` reads exactly like a `deny` that never had to
act. `openmax --check` warns about any project rule whose tool is neither a
built-in nor a tool in `.openmax/tools/`. It is a warning rather than an error
because writing the rule before the tool is a normal order to work in. Hook
`tool` filters are matched and reported the same way.

### Fail closed, with a repair path

If a permissions file exists but is invalid, every tool is denied (fail
closed), with one exception: rewriting that same file with `write_file` or
`edit_file` stays available, so a typo in a rule is repairable from inside the
session instead of ending it. The repair is exempt from the rules, not from
approvals: `approval_mode` still applies, so `ask` prompts for it and
`readonly` refuses it.

This covers the project file. A malformed global
`~/.openmax/permissions.toml` sits outside the project root, where file tools
never write, so fix that one from the shell; the deny reason names the exact
path and `openmax --check` prints the parse error.

## Proof of life

A tool file may declare one `[example]` (JSON args plus an optional
`expect_regex`). `openmax --check --run-examples` executes each declared
example through the real spawn path. For an approved tool the example is the
real command on the host, and a failed call or an output mismatch fails the
check. An unapproved tool is probed in a sandbox that denies the network and
any write outside its scratch directory, so a non-pass there is reported as a
warning and does not fail the check: under those restrictions it is
inconclusive, not proof the tool is broken (a passing probe approves nothing,
so a failing one condemns nothing). Plain `--check` stays read-only and never
executes anything; running examples is opt-in per invocation. Fuller
regression suites stay agent-authored files - the example is the smallest
honest proof that a freshly written tool actually runs.

Running an approved example is running the tool: an unsandboxed host process
in the project root with no snapshot taken. So it passes the same gates a turn
applies - the project must be trusted, a human must have approved the tool
file's exact bytes with `openmax --approve <path>`, `pre_tool_use` hooks must
allow the call, permission rules must admit it (`deny` refuses, and so does
`ask`: nothing here can prompt, so write `effect = "allow"` for a tool whose
example should run unattended), and `approval_mode` still governs
`mutating` tools (`readonly` refuses them; `ask` refuses them when the process
was started from an agent session, because there is nobody to prompt). Each
refusal prints the command that fixes it. An example that itself runs
`openmax --check --run-examples` is refused rather than recursed into.

## Content-bound approvals

An approval is a record in the hash-chained ledger below, holding the sha256
hashes a human approved and the path they approved them at. Approval binds to
content, never to a path: any edit produces a new hash and revokes itself.
Living in the chain is the point: an approval store beside the log, with none
of the log's protection, would be approvable by anything that can append a
line of JSON - including the agent's own `bash`. Forging one now means forging
the chain, which reads as tampering, and every real approval shows up in
`openmax --ledger` with its time and actor.

What gets hashed is the whole definition, not just the manifest. A tool or
hook TOML is a pointer: the file its `command` names (or a path in its `args`)
is what actually executes, and it sits at an ordinary project path the agent
writes freely. So approval covers the manifest **and** every project-local
file it hands to the host. A command that resolves outside the project root -
an absolute path, a name on `PATH` - is covered by the manifest approval
alone: that path is what the human read in the file they approved, and hashing
host binaries would revoke on every OS upgrade. The threat model is the agent
rewriting files it can write, not the system changing underneath. Paths are
resolved before that decision, so neither `../` nor a symlink moves a file to
the other side of it.

Binding reaches the files a manifest **names**, which is where its honesty
ends. A command handed a program on its own command line - `python3 -c "..."`,
`sh -c "..."` - has that program bound as text, because the text lives in the
manifest a human read; but whatever the program opens while it runs is chosen
at runtime and is not covered. `python3 -c "exec(open('payload.py').read())"`
hashes completely and still executes bytes nobody approved. Parsing
interpreter source to chase those reads would buy false confidence, so the
harness does not pretend to: put the program in a project file and name it in
`args`, and its bytes are bound like any other code. `openmax --check` warns
when inline text names a project file that exists, which is the cheap and
specific half of the problem; a program that computes its target at runtime is
beyond what any static check can promise.

A command that resolves to *no* file is a third case, and it refuses: an empty
binding means "this is a system path the human read", never "nothing resolved,
so there is nothing to check". A name that is not on `PATH`, a script that has
not been written yet, or a spelling this platform does not parse as a path all
leave the definition unapproved until the file exists.

Three hard rules ride on this, with no rule language to widen them:

- An external tool whose definition no human has approved always prompts on
  its first run - even in `auto` mode, even under a permissions `allow` rule,
  both of which the agent can write for itself. Approving that run approves the
  exact content; later runs of the same bytes are unattended. Every external
  tool is covered, whatever its `mutating` field says: that field is metadata
  the agent writes, while the call itself spawns a native host process with
  openmax's environment, credentials, and network access. A read-only tool is
  read-only by assertion, not by construction.
- Once a definition is approved, an edit to the manifest or to the code it
  runs makes the next call prompt again. Rewriting a blessed script is the same
  act as rewriting the manifest, and it is caught on every path a call can take
  - including the concurrent batch path, where read-only tools run unattended.
- Deleting an approved hook file fails closed too, and is found by reconciling
  the approved paths rather than the directory listing: a deleted file leaves
  nothing to parse and nothing to report against, and `rm gate.toml` is easier
  than rewriting it. Restore the file, or retire the approval with
  `openmax --forget <path>` when the removal was intended. `--forget` drops
  only the path memory, never a content hash: approval binds bytes, and the
  same bytes are still bytes a human read.

  `--forget` is the one command here that *removes* a protection, so it is
  guarded harder than the ones that add one: it refuses inside an agent
  session, refuses without an interactive terminal, and then asks for the
  path typed back.

  **Those guards are speed bumps, not a boundary. Do not build on them as if
  an agent cannot get past them.** They close the one-command bypass (`unset
  OPENMAX_SESSION; openmax --forget <path>`), and that is all they close. An
  attacker who allocates a pty and answers the prompt still retires the
  approval - about twenty lines of Python, measured, not assumed. What is no
  longer cheaper than that is editing the approval store: approvals live in
  the hash chain now, so appending, rewriting, or truncating one reads as
  tamper and fails closed toward `--ledger-repair` rather than granting
  anything. What a shell can still do outright is erase the whole ledger
  directory, which is indistinguishable from a project the harness has never
  seen: that surrenders every approval (fail closed, so gates stop running
  rather than start) and costs the project the memory that a gate ever
  existed. **Nothing in this design survives an agent with a shell that is
  determined to remove a gate; only an OS sandbox would.** What the guards buy
  is narrower and worth stating exactly: removal cannot happen by accident or
  in a single step, and every state it leaves behind is visible in
  `openmax --check`.
- An unapproved hook is inert, and a *revoked* gate hook fails closed. Hooks
  run with host authority on every matching call with no per-invocation gate,
  so they never load until approved. Content nobody ever approved never ran,
  so it stays inert and is reported every turn (a `hook_failed` note) as well
  as by `openmax --check`. But content at a path a human did approve is a
  modification of a live hook, and dropping a live `pre_tool_use` or
  `user_prompt_submit` gate is fail-open - a comment-only edit would be enough
  to switch it off - so those block every tool call until the approved content
  is restored or a human re-approves. A hook's bound code is re-checked
  immediately before every run, so a script rewritten between two calls of one
  turn does not run either.

Approvals happen two ways: answering the `unapproved_source` prompt an
external tool raises on its first call - the card names the manifest and its
hash, and approving blesses the manifest and the code it runs - and `openmax
--approve <path>` for everything else (a hook, a file that arrived from a
clone or an installer, an auto-mode write). Approving an in-session
`write_file` or `edit_file` of a capability file approves that write and
nothing more: the approval card shows a clipped preview of the arguments, and
a preview is not shown bytes, so a tool the agent writes still prompts on its
first call and a hook stays inert until `--approve`.
`--approve` on a manifest blesses the code it names in the same act and prints
every path and hash it blessed, because a human cannot approve bytes they were
not shown; it refuses when that code cannot be read. Like `--trust-project`,
`--approve` refuses inside agent-spawned processes: approvals are human
actions. Skills have no enforcement; they are prose, and the tools they invoke
are the gate.

## The capability ledger

Every tool and skill file a freeze reads is recorded in a per-project,
append-only, hash-chained ledger under `~/.openmax/ledger/`, together with a
content-addressed copy of the bytes. Each record carries the actor at the
strength the harness can prove: `session` (changed while an agent turn was
running), `external` (changed while none was: a human, `git pull`, an
installer), or `initial` (present when the ledger was first populated). The
ledger lives outside the project, where the confined file tools never write,
and each record chains the hash of the previous one, so tampering through the
shell is detectable.

Beside the log sits `chain-head`, the hash of its final record. The chain
alone proves internal order but not completeness - lopping off trailing
records leaves a valid prefix - so the pin is what makes truncation
detectable, and a log without one cannot be verified at all. Deleting either
file therefore reads as tampering, not as a fresh project. An append writes a
pending pin first, flushes the records, then moves the pin, so a crash
mid-append leaves a state that reads as an interrupted write (nothing was
removed) and re-pins itself on the next change, rather than an accusation.

Every re-freeze announces a receipt - the `refrozen` event lists what changed
and who changed it - so the agent's action space never mutates silently.
`openmax --ledger` prints the history with object paths, human times, the
session that observed each change, and every approval; it verifies each object
against its own hash, so a rewritten one is named instead of silently offered
for restore. Restoring an earlier version is an ordinary `cp` from the objects
directory. There is no rollback command: the core guarantees the history
exists, using it stays file work.

A ledger that cannot be verified stops appending (a chain nobody can trust
must not be extended) and revokes every approval it held, but never blocks a
turn: the failure rides the refreeze receipt. `openmax --ledger-repair` is the
way back - a human action, refused inside agent-spawned processes, that
quarantines the damaged log as evidence, keeps the objects, and starts a new
chain. Approvals in that log go with it and have to be granted again.

## Self-measurement

The dispatcher counts what only it can see: every external-tool call (with
outcome) and every skill-body read, merged once per turn into the project's
usage record beside the ledger. `openmax --spec usage` joins those counters
with each extension's frozen-prompt cost - the characters every request pays
while the file is installed - and its approval state, so the agent can see
which of its own creations are pure tax and delete them. Nothing is pruned
automatically: the core measures, the agent judges.

With enough recorded signal (50 calls), `openmax --check` warns about
extensions that were never used.

## Validation

`openmax --check` parses tools, skills, templates, hooks, permissions, and
`providers.json`, then prints per-file results with the reason anything would
be ignored, fail closed, or fail at request time. The agent is instructed to
run it after writing extension files.

Each line is `ok`, `warn`, or `err`, and only `err` exits nonzero:

- `err` is a file the loop cannot use: it does not parse, it fails closed, or
  it can never work (a tool shadowing a built-in name).
- `warn` is a file the loop reads past or cannot act on as written: a path
  nothing reads, a definition another tier overrides, a rule naming a tool
  that does not exist. Each of these is legitimate in some project, so none
  of them fails the check.

Warnings cover the ways a file goes missing without being broken. A directory
at `.openmax/tool/` or `.openmax/skills/`, a `.yaml` where a `.toml` is read,
or a skill directory with no exactly spelled `SKILL.md` are all reported with
the path that would work instead. A global file shadowed by a project file of
the same name is reported against the file that loses, naming the winner. A
broken file that something shadows is a warning, not an error, because the
loop never loads it and so never fails closed on it.

## Self-description

`openmax --spec <surface>` prints the complete authoring contract for one
surface (`tools`, `skills`, `prompts`, `hooks`, `permissions`, `providers`, or
`stdio`): file grammar, field caps and defaults, hook stdin payload shapes, and
activation timing. The frozen prompt carries only a one-line pointer to it, so
the full contract costs zero tokens until the agent reads it, and the printed
examples are parsed by the same validation code in tests so the text cannot
drift from the binary.

## Freezing and re-freezing

Tools and skills freeze per freeze window for prompt-cache stability, and the
harness re-freezes them automatically. At each turn start, and again between
iterations after a mutating tool call succeeded, it captures one immutable
generation of extension bytes, fingerprints that snapshot, and parses those
same bytes. If the generation changed, it rebuilds the registry and prompt in
place (one deliberate cache re-prefill, conversation kept, a `refrozen` event
for clients), so a tool the agent writes is callable on its very next step,
inside the same turn, with no human action. Atomic replacement and symlink
swaps cannot make the activated registry disagree with its fingerprint. An
unchanged generation does not rebuild or invalidate the prompt cache.
`/reload` forces a new capture immediately; `/new` starts clean. Hooks,
permissions, and templates re-discover on every turn or invocation. Use
`/tools`, `/skills`, and `/context` to inspect the frozen set and its cost.
