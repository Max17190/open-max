# Hold a run to your own checks

This page is for a program that hands Open Max a task: a script, a CI job or
another agent. It puts in one place what such a caller needs to hold a run to
its own checks and to read back what happened.

## What it is

A turn's end can be held to commands you choose. While one of them fails,
Open Max refuses the end and sends the failing output back to the model, which
works on from there. The result says which of your checks passed when they
were run again after the agent exited, and the whole run replays offline from
one file.

What it is not:

- Not a sandbox. In `auto` the model's shell commands, tools and hooks run
  with your user's access ([native execution](execution.md)). If you need
  isolation, run the call in a container you control.
- Not proof beyond what your checks test. A passing check says what that check
  tests and no more.
- Not out of the model's reach. The model can change any file your checks
  read, a test among them, and in `auto` it can edit the list of commands the
  gate runs. The example below answers this in two places: `run_task.py`
  keeps its own copy of your checks and runs them again after the process has
  exited, and only that run decides the result; and it returns the whole
  change as a patch, so a changed test is in front of you.

## See it in one minute

From a clone of this repository, with the `openmax` binary, Python 3.9 or
later and Git, and no model account:

```sh
python3 examples/run-task/demo.py    # --openmax <path> when openmax is not on PATH
```

A small server on 127.0.0.1 plays the model's side, and everything else is
real. The demo makes a throwaway repository whose check fails, runs
`run_task.py` on it, and prints the exchange (trimmed here):

```text
model: 'Done.'
gate:  the check exits 1, so Open Max refuses the end and sends the model:
   | Not finished: 1 of 1 checks failed. Fix the cause, then end the turn again.
   | $ test "$(sh greet.sh)" = hello || { echo "greet.sh printed: $(sh greet.sh)"; e... (exit 1)
   | greet.sh printed: hullo

model: a write_file call: {"path": "greet.sh", "content": "echo hello\n"}
openmax runs the call and sends the model its result

model: 'Fixed.'
gate:  the check exits 0, so the turn ends
```

It then prints the patch, the JSON result (`"verified": true`,
`"refusals": 1`), and the replay command run with an empty environment and
the server stopped, with whether its output matched the run's stdout byte for
byte. It works in a private temporary home and removes it at the end.

## Use it on your repository

[examples/run-task](../examples/run-task/README.md) is the caller this page
follows. It adds nothing to Open Max: it is a print run (`openmax -p --json`),
a completion gate (a `turn_end` hook with `blocking = true`) and the session's
event log, put together.

Set up once, from your clone of this repository:

```sh
mkdir -p ~/.openmax/hooks
cp examples/run-task/run-task-checks.toml ~/.openmax/hooks/
cd ~/code/my-app
openmax --trust-project -p "summarize this repo"   # from a terminal
```

A trust grant is a person's act, and the script never makes one. The grant
records `auto` for a project not trusted before, which unattended work needs:
a print run declines every approval request. A job with no terminal grants
trust as [project trust](configuration.md#project-trust) describes. The hook
is global but judges only a turn that `run_task.py` started; in any other
session it exits 0 at once. `openmax --check` in the repository lists it as an
`ok` row, `(hook on blocking turn_end)`.

Then call it, where `~/open-max` is your clone:

```sh
python3 ~/open-max/examples/run-task/run_task.py --repo ~/code/my-app \
  --check "make test" \
  --final-check "make lint" \
  "Make the failing test pass."
```

- `--check` refuses the turn's end while it fails. It runs with `sh -c` in the
  repository's root and passes when it exits 0; repeat it for more. The
  script runs each once before the work, the gate runs them at each end of the
  turn, and the script runs them once more after the process has exited.
- `--final-check` runs only after the process has exited. It decides the
  result and cannot refuse an end, so it suits a check slower than the gate
  allows.

The script prints one JSON result and writes it to `result.json`. The fields a
caller acts on: `verified` and `reason`, `stop_reason`, each check's `before`
and `after` exit codes, `refusals`, `gates_passed`, `changes` (the patch) and
`replay` (the command that prints the run again). The example's README gives
[every field](../examples/run-task/README.md#the-result) and the
[exit codes](../examples/run-task/README.md#exit-codes), in short:

| Code | Meaning |
| --- | --- |
| 0 | The run ended and every check passed when the script ran it afterwards |
| 1 | Open Max failed, for example the endpoint could not be reached |
| 2 | The call could not start: a usage error, a gate that is not installed, does not load or is not the shipped file, or a `--check` too slow for the gate |
| 3 | The repository is not trusted |
| 4 | The turn ended `unverified`, `length`, `max_iterations` or `budget_exhausted`, or a check failed afterwards |
| 128 + N | Signal N stopped the run |

## How a turn's end is judged

A completion gate is a `turn_end` hook with `blocking = true`
(`openmax --spec hooks` prints the contract). The one end it can refuse is the
model's own: a reply with no tool calls. An end the model did not ask for
(`max_iterations`, `budget_exhausted`, `cancelled`, `error`, `truncated`)
still runs the hook, but nothing is left to honor a refusal with, so it is
reported and the turn ends.

**A refusal.** A gate that exits nonzero refuses the end. Its stdout, or its
stderr when stdout is empty, reaches the model as a user message, and the turn
continues within the same iteration and token ceilings. Open Max keeps the
first 500 characters of that reason. The example's gate writes each failing
check, its exit code and the end of its output within that room. The stream
carries a `turn_refused` event (`hook`, `reason`, `continuation`,
`continuations_left`), sent once the message is saved to the transcript.

**The bound.** Open Max honors at most eight refusals in one turn. A gate's
`max_refusals` (1 to 8, default 8) lowers its own share: once that many of its
refusals have been honored, its next failing end is reported, not sent to the
model, and the turn ends `unverified`. The example's hook file sets 3. A turn
whose gate it cannot satisfy otherwise spends every refusal, each one another
round on a longer transcript; a lower bound stops it sooner.

**Time.** A completion gate may set `timeout_secs` up to 600 seconds; every
other hook gets at most 60, and 10 by default. A gate that times out refuses.
The example's hook asks for 600, and its checks share that time less 10
seconds; a `--check` with no result in that time before the work is refused
with exit 2. At an end the model did not ask for, a gate gets at most 60
seconds. A cancel (a stdio `cancel`, or a signal to a print run) stops a gate
as it stops a tool call, and the turn ends `cancelled`. After a signal Open
Max waits at most 3 seconds for the turn, then kills a `turn_end` hook still
running.

**`unverified`.** The turn ends with `stop_reason` `unverified` when:

- a gate refuses past the bound: eight in the turn, or its own `max_refusals`;
- a refusal cannot be saved to the transcript, so it is not honored;
- in `auto`, a gate cannot start (its command is not found or not executable,
  or its script does not exist): it is reported in `hook_failed` and refuses
  nothing;
- a hook file fails closed at the end (it no longer parses, an approved gate
  changed or was deleted, a gate cannot run, or the approval store cannot be
  read): the end is reported in `hook_failed` and no refusal is spent. The
  model is told at turn start, once per session, and a turn that repairs the
  file is judged by the hooks on disk at its end.

A print run exits 4 on `unverified`. A `--stdio` run exits 0, and its client
reads the `stop_reason` of `done`.

**`turn_verified`.** This event (`hooks`) goes out before `done` when the
model ended the turn and every blocking `turn_end` gate loaded for that end
ran and exited 0, named in the order they ran. A gate that refused earlier
ends of the turn and then passed is named too. There is no such event when no
gate judged the end (none is installed, or only observe-only `turn_end` hooks
are), when the turn ends `unverified`, or at an end the model did not ask for.

It asserts that each gate exited 0, and nothing about what a gate checked. The
example's hook exits 0 at once in a session `run_task.py` did not start, so
with it installed, the end of an ordinary session names it too. Nor does it
assert that your checks pass after the turn, or that the gate ran the list you
gave: in `auto` a model that empties that list ends its turn with
`turn_verified`, and the example's own run of your checks then fails. The
example reports the event as `gates_passed` and lets its own run decide
`verified`.

## Bounding a run

- `--max-agent-iterations <n>`: the tool-call rounds one turn may take
  (the settings default is 300). A turn that reaches it ends
  `max_iterations`.
- `--max-agent-tokens <n>`: the tokens one turn may spend, checked before each
  request. A request that would take the turn past it is not sent, and the
  turn ends `budget_exhausted`. The reply is not reserved, so the last request
  sent can pass the ceiling by its output. There is no such ceiling unless
  you set one.

Both bound one `-p`, `--stdio` or `--subagent` run and are never written to
`~/.openmax/settings.json`. A print run that reaches one exits 4 and its stop
line names the flag:

```sh
openmax -p --max-agent-iterations 1 "Make the failing test pass."
```

```text
openmax: stopped (max_iterations): reached the limit of 1 tool-call round for one turn. --max-agent-iterations set it for this run. Run openmax --continue -p "<follow-up>" to continue
```

`run_task.py` takes both and passes them on, and reports them in its result's
`ceilings`. It also takes `--timeout <seconds>`: at that time the run gets
SIGTERM, which cancels the turn, and SIGKILL 10 seconds later if it is still
there. The script still prints a result, with `stop_reason` `cancelled`, and
exits 143 or 137. A print run has no deadline of its own.

There is no ceiling in money. Open Max counts tokens and does not know their
price, so bound the tokens and multiply by your own price.

## Continuing a run you started

```sh
openmax --session <id> --json -p "Also handle the empty input."
```

`--session <id>` continues that session, with `-p` or `--stdio`, instead of
the latest one in the directory. The id is the `session_id` every event line
carries, and the result's `session_id`. An id that no session of this
directory has, one of another project's sessions, and one whose transcript is
missing or does not load are each refused with exit 2 and a line that says
which.

`run_task.py` starts a new session for each call and takes no `--session`. A
continued turn started this way is a plain print run, in which the example's
hook exits 0 at once. To hold it to the same checks, give it the call's
directory as the script does, here with the result's `out` in `OUT`:

```sh
RUN_TASK_DIR="$OUT" openmax --session <id> --json -p "Also handle the empty input."
```

The gate then reads its commands from `checks.json` there and appends each
end it judges to `gate.jsonl`. Nothing runs your checks again after that
process exits, so run them yourself before you act on the end.

## Reading it back

Each session keeps an event log, `~/.openmax/sessions/<id>.events.jsonl`: a
format line, then every event the session sent while an openmax held it, each
as the line `-p --json` prints. A continued session appends to the same log.
`run_task.py` copies it to `events.jsonl` in the call's directory and puts the
command that prints it in `replay`:

```sh
openmax --replay ~/.openmax/sessions/<id>.events.jsonl --json
```

`--replay` reads that one file and nothing else: no endpoint, settings, trust
or data directory, and no tool runs, so a log copied to another machine
replays there. For a `-p --json` run it prints that run's stdout byte for
byte, for every turn that finished.

What it does not give you:

- A turn that ended with its process (a crash) leaves out what had not yet
  been written to the log.
- It prints the record; it does not call the model or run a tool again. It
  shows what happened, not that the same task would go the same way twice.
- What your checks printed at each end is in the example's `gate.jsonl`, and
  each refusal's reason in its `turn_refused` event. The change to the work
  tree is in `changes.patch`.
- The log is a plain file its owner can edit: a replayable record, not
  evidence that nobody changed it.

## You want, use

| You want | Use |
| --- | --- |
| The whole path, with no model | `python3 examples/run-task/demo.py` |
| A turn's end held to a command | `run_task.py --check "<command>"`, or a `turn_end` hook with `blocking = true` |
| A slow check to decide the result | `run_task.py --final-check "<command>"` |
| A check that cannot pass to stop sooner | `max_refusals` in the gate's hook file |
| A bound on one run's work | `--max-agent-iterations <n>`, `--max-agent-tokens <n>` |
| A bound in time | `run_task.py --timeout <seconds>` |
| To know every gate let the end stand | The `turn_verified` event, or `gates_passed` |
| To know your checks pass after the run | `verified`, and each check's `after` |
| To continue the session a call started | `openmax --session <id>` |
| To see every step again, offline | `openmax --replay <file> --json` |
