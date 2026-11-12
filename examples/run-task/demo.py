#!/usr/bin/env python3
"""Watch run_task.py hold a turn to a failing check, with no model account.

The model's replies are scripted: a small server on 127.0.0.1 plays the
model's side. Everything else is real: the openmax binary, the completion
gate, the check, git, the result and the replay. The demo works in a private
temporary home and repository, never reads or writes your ~/.openmax, and
removes them at the end.

    python3 examples/run-task/demo.py [--openmax PATH]
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # no __pycache__ beside the example
from scripted_endpoint import CHECK, DONE, FIX, TASK, endpoint, fixture  # noqa: E402

FIXED = {"text": "Fixed."}


def heading(text):
    print(f"\n== {text}\n", flush=True)


def indent(text):
    print(textwrap.indent(text.rstrip("\n"), "   | ", lambda line: True), flush=True)


def describe(reply):
    if "tool" in reply:
        return f"a {reply['tool']} call: {json.dumps(reply['args'])}"
    return repr(reply["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--openmax", default="openmax", help="the openmax binary (default: openmax on PATH)")
    args = parser.parse_args()
    binary = shutil.which(args.openmax)
    if binary is None:
        print(f"demo: no executable {args.openmax}; install Open Max or pass --openmax", file=sys.stderr)
        return 2
    binary = str(Path(binary).resolve())
    started = time.monotonic()

    print("The model's replies in this demo are scripted by a server on 127.0.0.1.")
    print("Everything else is real: openmax, the completion gate, the check, git, the result and the replay.")
    replies = [DONE, FIX, FIXED]
    with tempfile.TemporaryDirectory(prefix="run-task-demo-") as temp:
        root = Path(temp).resolve()
        print(f"It works in {root}, a private home and repository, and removes it at the end.")
        with endpoint(replies) as (url, requests):
            repo, env, out = fixture(root, "demo", binary, url)

            heading("1. A repository whose check fails before the work")
            print(f"greet.sh: {(repo / 'greet.sh').read_text().strip()}")
            print(f"task:     {TASK}")
            print(f"check:    {CHECK}")
            before = subprocess.run(["sh", "-c", CHECK], cwd=repo, env=env, capture_output=True, text=True)
            indent(before.stdout + before.stderr)
            print(f"exit {before.returncode}")

            heading("2. run_task.py runs the task, held to the check")
            command = [sys.executable, str(HERE / "run_task.py"), "--repo", str(repo), "--out", str(out),
                       "--openmax", binary, "--check", CHECK, TASK]
            print("$ " + shlex.join(["python3", "examples/run-task/run_task.py", "--repo", "<repo>",
                                     "--check", CHECK, TASK]))
            call = subprocess.run(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, timeout=120)
        # The scripted server has stopped: nothing below can reach a model.

        # Each reply with no tool call ends the turn, and the gate judges that
        # end: gate.jsonl has one line for each, in order.
        gate = [json.loads(line) for line in (out / "gate.jsonl").read_text().splitlines()] \
            if (out / "gate.jsonl").is_file() else []
        ends = iter(gate)
        for index, reply in enumerate(replies[:len(requests)]):
            print(f"\nmodel: {describe(reply)}")
            if "tool" in reply:
                print("openmax runs the call and sends the model its result")
                continue
            end = next(ends, None)
            if end is None:
                continue
            print(f"gate:  the check exits {end['checks'][0]['exit']}, "
                  + ("so the turn ends" if end["passed"] else "so Open Max refuses the end and sends the model:"))
            if not end["passed"] and index + 1 < len(requests):
                indent(requests[index + 1]["messages"][-1]["content"])

        heading("3. The fix")
        patch = out / "changes.patch"
        indent(patch.read_text() if patch.is_file() else "(no changes.patch)")

        heading(f"4. The result (run_task.py exited {call.returncode})")
        print(call.stdout.rstrip() or call.stderr.rstrip())
        if call.returncode != 0 or not call.stdout:
            return 1
        result = json.loads(call.stdout)

        heading("5. The replay, with an empty environment and the server stopped")
        print("$ " + result["replay"])
        replay = subprocess.run(shlex.split(result["replay"]), env={}, capture_output=True, timeout=120)
        recorded = (out / "stdout.jsonl").read_bytes()
        same = replay.returncode == 0 and replay.stdout == recorded
        lines = replay.stdout.splitlines()
        types = [json.loads(line).get("type") for line in lines if line.startswith(b"{")]
        print(f"{len(lines)} events, {len(replay.stdout)} bytes: "
              + ("identical to the run's stdout, byte for byte" if same else "DIFFERENT from the run's stdout"))
        print("event types in order: " + " ".join(type_ for i, type_ in enumerate(types) if i == 0 or types[i - 1] != type_))
    if root.exists():
        print(f"\nCould not remove {root}.")
        return 1
    print(f"\nRemoved {root}. Took {time.monotonic() - started:.1f} seconds.")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
