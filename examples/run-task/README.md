# Run a task held to your own checks

This optional example is one entry point for a program that hands Open Max a
task: a script, a CI job or another agent. `run_task.py` runs the task in your
repository, with the model your installation is configured for, and refuses
the turn's end while one of your checks fails. It prints one JSON result: how
the turn ended, each check before and after the work, what changed, and the
command that replays every step offline.

It adds nothing to Open Max. It is a print run (`openmax -p --json`), a
`turn_end` hook with `blocking = true` (`openmax --spec hooks`) and the
session's event log (`openmax --replay`), put together.

## See it in one minute

With the `openmax` binary, Python 3.9 or later and Git, and no model account:

```sh
python3 examples/run-task/demo.py    # --openmax <path> when openmax is not on PATH
```

The model's replies are scripted by a small server on 127.0.0.1. Everything
else is real. The demo makes a throwaway repository whose check fails, runs
`run_task.py` on it, and prints, in order: the check failing before the work,
the turn's end being refused and the message the model was sent, the fix, the
JSON result, and the replay command printing the run's output again byte for
byte with the server stopped. It works in a private temporary home, where it
grants trust to the throwaway repository as a person does at setup, so it
never reads or writes your `~/.openmax`, and it removes that home at the end.

## Set up once

Use Python 3.9 or later as `python3` on `PATH`, and Git, on Linux or macOS.

```sh
mkdir -p ~/.openmax/hooks
cp examples/run-task/run-task-checks.toml ~/.openmax/hooks/
cd ~/code/my-app
openmax --trust-project -p "summarize this repo"   # from a terminal
```

A trust grant is a human action, and the script never makes one. The grant
records `auto` mode for a project not trusted before, which unattended work
needs: in `ask`, a print run declines every approval request. A job with no
terminal grants trust as
[project trust](../../docs/configuration.md#project-trust) describes.

The hook is global, so every session you start loads it. It judges only a
turn that `run_task.py` started: any other session has no `RUN_TASK_DIR` in
its environment, and the hook exits 0 at once. A project in `ask` or
`readonly` reports the hook as unapproved until you approve it from that
project with `openmax --approve ~/.openmax/hooks/run-task-checks.toml`.

`run_task.py` runs `openmax --check --json` first and exits 2 unless the hook
loads in the repository and is the file this example ships: a hook of the
same name without `blocking = true` loads too, and could refuse nothing. A
call it refuses before the run prints the reason on stderr and no result.

## Call it

```sh
python3 examples/run-task/run_task.py --repo ~/code/my-app \
  --check "cargo test -p parser" \
  --final-check "cargo test --workspace" \
  "Make parse_date accept ISO week dates."
```

- `--check` is a command that refuses the turn's end while it fails. It runs
  with `sh -c` in the repository's root and passes when it exits 0. Repeat the
  option for more. The script runs these once before the work, so the result
  shows which of them failed to begin with; the hook runs them at each end of
  the turn; and the script runs them once more after the process has exited.
- `--final-check` is a command the script runs only after the process has
  exited. It decides the result and cannot refuse the turn's end. Use it for
  a check that takes longer than the gate allows. It has no time limit.
- `--out` names a new or empty directory outside the repository for the
  call's files. Without it the script makes one and names it in the result.
- `--timeout` stops the run after that many seconds. A print run has no
  deadline of its own.
- `--max-agent-tokens` and `--max-agent-iterations` bound the turn for this
  call only: the script passes each to Open Max as its flag of the same name,
  and a turn that reaches one ends `budget_exhausted` or `max_iterations`
  (exit 4). Neither is written to your settings. There is no ceiling in
  money, since Open Max counts tokens and does not know their price.
- `--openmax` names the binary when `openmax` is not on `PATH`.

The shipped hook file lets the gate run for 600 seconds (`timeout_secs`, the
most Open Max gives a completion gate), and the `--check` commands share that
time less 10 seconds each time they run. One that has no result in that time
before the work starts is refused with exit 2, since a check that slow would
refuse every end.

When a check fails at the turn's end, Open Max refuses the end and sends the
model one message: each failing check, its exit code and the end of its
output, within the 500 characters a refusal keeps. The model works on from
there. After eight refusals Open Max ends the turn as `unverified`.

## The result

The script prints the result on stdout and writes it to `result.json`:

```json
{
  "format": "run-task/1",
  "verified": true,
  "exit_code": 0,
  "reason": null,
  "stop_reason": "stop",
  "session_id": "a92b42d0-63f8-4bb3-bc1d-284ca25822b8",
  "repo": "/home/me/code/my-app",
  "ceilings": {"max_agent_tokens": null, "max_agent_iterations": null},
  "checks": [
    {
      "command": "cargo test -p parser",
      "gates": true,
      "before": 101,
      "after": 0,
      "seconds": 3.412,
      "output_tail": "test result: ok. 41 passed; 0 failed\n"
    }
  ],
  "refusals": 1,
  "gates_passed": ["run-task-checks"],
  "gate_log": "/tmp/run-task-k2x9/gate.jsonl",
  "changes": "/tmp/run-task-k2x9/changes.patch",
  "files_changed": ["crates/parser/src/date.rs"],
  "usage": {"replies": 14, "prompt_tokens": 182340, "completion_tokens": 5120, "cached_tokens": 161280},
  "event_log": "/tmp/run-task-k2x9/events.jsonl",
  "replay": "/usr/local/bin/openmax --replay /tmp/run-task-k2x9/events.jsonl --json",
  "out": "/tmp/run-task-k2x9"
}
```

| Field | Meaning |
| --- | --- |
| `verified` | The run ended and every check passed when the script ran it afterwards |
| `exit_code` | The script's exit code, from the table below |
| `reason` | Why the result is not verified, or null |
| `stop_reason` | The `stop_reason` of the run's `done` event |
| `ceilings` | The `--max-agent-tokens` and `--max-agent-iterations` this call passed, or null |
| `checks` | Each check: whether it `gates` the turn's end, its exit code `before` the work (null for a final check) and `after` it (null when the checks were not run again), and the time and the end of the output of that last run |
| `refusals` | The ends of the turn the gate refused, counted from the `turn_refused` events |
| `gates_passed` | The gates Open Max reports passed the end the turn stopped on, from its `turn_verified` event, or empty when no gate passed it. The checks this script runs afterwards still decide `verified` |
| `gate_log` | One JSON line for each end of a turn the gate judged: the session, the `continuation`, whether a refusal could be honored (`blockable`), and each check's exit code, time and output. A session the agent starts itself inherits the gate and is logged here too |
| `changes`, `files_changed` | The work tree after the run against the work tree before it, as a patch and as paths. Untracked files are included and ignored ones are not |
| `usage` | The replies the model sent, and the sums of the run's `usage` events |
| `event_log`, `replay` | A copy of the session's event log, and the command that prints it back |

The directory in `out` also holds `stdout.jsonl` and `stderr.txt` (what the
print run wrote), `checks.json` (the gate's list) and the copies of
`run_task_checks.py` the hook ran and of the hook file it reads its time from.

## Replay

The `replay` command reads `events.jsonl` and nothing else: no endpoint,
settings, trust or tool. It prints the run's stdout byte for byte, one JSON
event on each line, in the fields [stdio protocol](../../docs/stdio-protocol.md)
gives. The log can be copied to another machine and replayed there.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | The run ended and every check passed afterwards |
| 1 | Open Max failed: the endpoint could not be reached or a reply was cut off. `reason` carries the error |
| 2 | The call could not start: a usage error, a repository that is not a git work tree's root, a gate that is not installed, does not load or is not the shipped file, or a `--check` too slow for the gate |
| 3 | The repository is not trusted |
| 4 | Not verified. The turn ended `unverified`, `length`, `max_iterations` or `budget_exhausted`, or it ended normally and a check failed afterwards |
| 128 + N | Signal N stopped the run: 143 when `--timeout` expired, or 137 when the run had to be killed 10 seconds after that |

Codes 1, 3 and 4 from a run are Open Max's own (see
[usage](../../docs/usage.md)). A SIGINT or SIGTERM sent to the script is
passed on to Open Max, which cancels the turn and stops its tools; the script
then exits 128 plus the signal's number and prints no result. The gate's
checks run in the hook's process group, so a check in flight is stopped with
the hook: Open Max gives a `turn_end` hook up to 3 seconds, then kills its
group.

The script drops the variables that point git at a repository (`GIT_DIR`,
`GIT_WORK_TREE` and the others `git rev-parse --local-env-vars` lists) from
its environment, so `--repo` decides the repository for git, for the checks
and for Open Max even when a git hook makes the call.

## What a verified result means

The verdict is the script's own run of the checks after the process has
exited, on the work tree the run left. Open Max's exit code alone is not the
verdict. In `auto` the model's shell commands run with your access, the
call's directory included, so a turn can end normally with the gate's list
emptied. The script keeps the checks it was given and runs those.

A check proves what it tests and no more. The model can change any file a
check reads, a test among them, so read `changes.patch` or check it with a
`--final-check` of your own.

The run keeps the limits of your installation: `max_agent_iterations` and
`max_agent_tokens` in `~/.openmax/settings.json` bound a turn, unless the
call's `--max-agent-iterations` or `--max-agent-tokens` sets one for that run.
Run one call at a time in a work tree.

## Test

```sh
cargo build -p openmax
python3 examples/run-task/test_run_task.py target/debug/openmax
```

The test runs the script against a scripted endpoint on loopback: a turn
whose first end is refused and whose second passes, a turn that spends all
eight refusals, a turn that empties the gate's list, a final check, a turn
stopped by a ceiling, a timeout
and a signal, a project in `ask` mode, an endpoint that is down, the calls
that are refused before any request, and the demo. The demo and the test share
the scripted endpoint in `scripted_endpoint.py`.
