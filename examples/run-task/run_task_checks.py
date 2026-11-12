#!/usr/bin/env python3
"""Run the caller's checks when a turn ends, and refuse the end while one fails.

Open Max starts this through the `turn_end` hook in run-task-checks.toml, in
the project root, with one JSON payload on stdin. `RUN_TASK_DIR` names the
directory of the call: `checks.json` there lists the commands, and each end
of the turn appends one line to `gate.jsonl` beside it. Exit 0 lets the turn
end. Exit 1 refuses the end, and stdout goes to the model as a user message.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

# A hook is stopped after 60 seconds at most, so the checks share less than
# that: one that is still running at the limit is reported by name, where the
# hook's own timeout would say only that the hook timed out.
GATE_SECONDS = 50
# Open Max keeps the first 500 characters of a refusal.
REASON_CHARS = 480
TAIL_BYTES = 4096

_running = None


def stop_running_check():
    """Kill the check in flight with every process it started."""
    if _running is not None and _running.poll() is None:
        try:
            os.killpg(_running.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_checks(commands, cwd, seconds=None):
    """Run each command with `sh -c` in `cwd`, and return one record for each.

    `exit` is the command's exit code, or None when it had no result within
    `seconds`, the time all the commands share: it was stopped, or the time
    was spent before it could start. `output_tail` is the end of what it
    printed, stdout and stderr together.
    """
    global _running
    deadline = None if seconds is None else time.monotonic() + seconds
    records = []
    for command in commands:
        started = time.monotonic()
        code, tail = None, ""
        if deadline is None or started < deadline:
            with tempfile.TemporaryFile() as output:
                # A session of its own, so stopping the check stops its children.
                _running = subprocess.Popen(["sh", "-c", command], cwd=cwd, stdin=subprocess.DEVNULL,
                                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = _running.wait(None if deadline is None else deadline - started)
                except subprocess.TimeoutExpired:
                    stop_running_check()
                    _running.wait()
                size = output.seek(0, os.SEEK_END)
                output.seek(max(size - TAIL_BYTES, 0))
                tail = output.read().decode("utf-8", "replace")
        records.append({"command": command, "exit": code, "seconds": round(time.monotonic() - started, 3),
                        "output_tail": tail})
    _running = None
    return records


def refusal(records):
    """What the model reads when the end is refused: each failing check, its
    status and the end of its output, within the characters Open Max keeps."""
    failed = [record for record in records if record["exit"] != 0]
    text = f"Not finished: {len(failed)} of {len(records)} checks failed. Fix the cause, then end the turn again.\n"
    room = (REASON_CHARS - len(text)) // len(failed)
    for record in failed:
        status = "no result within the gate's time" if record["exit"] is None else f"exit {record['exit']}"
        command = record["command"] if len(record["command"]) <= 80 else record["command"][:77] + "..."
        title = f"$ {command} ({status})\n"
        keep = max(room - len(title) - 1, 0)
        tail = record["output_tail"].strip()
        text += title + (tail[-keep:] + "\n" if keep and tail else "")
    return text[:REASON_CHARS]


def main():
    payload = json.loads(sys.stdin.readline())
    call = Path(os.environ["RUN_TASK_DIR"])
    signal.signal(signal.SIGTERM, lambda *_: (stop_running_check(), sys.exit(143)))
    records = run_checks(json.loads((call / "checks.json").read_text()), os.getcwd(), GATE_SECONDS)
    passed = all(record["exit"] == 0 for record in records)
    with open(call / "gate.jsonl", "a") as log:
        log.write(json.dumps({"session_id": payload["session_id"], "continuation": payload["continuation"],
                              "blockable": payload["blockable"], "stop_reason": payload["stop_reason"],
                              "passed": passed, "checks": records}) + "\n")
    if not passed:
        sys.stdout.write(refusal(records))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
