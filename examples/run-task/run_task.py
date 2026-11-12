#!/usr/bin/env python3
"""Run one task in a repository with Open Max, held to the caller's checks.

A `--check` command decides whether the turn may end: while one fails, the
turn's end is refused and its output goes back to the model. When the process
has exited, this script runs every check again itself, and that run is the
verdict. It prints one JSON result and exits 0 only when the run ended and
every check passed. README.md beside this file gives the result's fields and
the exit codes.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

sys.dont_write_bytecode = True  # no __pycache__ beside the example
import run_task_checks  # noqa: E402

HOOK = "run-task-checks"
HERE = Path(__file__).resolve().parent
USAGE, NOT_TRUSTED, NOT_VERIFIED = 2, 3, 4


def stop(code, message):
    print(f"run_task: {message}", file=sys.stderr)
    sys.exit(code)


def stop_on_signal(signum, _frame):
    """Outside the print run, a signal ends the script and the check in flight."""
    run_task_checks.stop_running_check()
    sys.exit(128 + signum)


def git(repo, *args, env=None):
    result = subprocess.run(["git", "-C", str(repo), *args], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, encoding="utf-8", errors="surrogateescape")
    if result.returncode != 0:
        stop(USAGE, f"git {args[0]} failed in {repo}: {result.stderr.strip()}")
    return result.stdout


def work_tree(repo, scratch, name):
    """The work tree as a git tree: tracked and untracked files, ignored ones
    left out. It stages into a copy of the index, so the repository's own
    index stays as it is."""
    index = scratch / name
    source = Path(git(repo, "rev-parse", "--absolute-git-dir").strip()) / "index"
    if source.exists():
        shutil.copyfile(source, index)
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    git(repo, "add", "-A", env=env)
    return git(repo, "write-tree", env=env).strip()


def gate_is_ready(openmax, repo):
    """Refuse to start unless `openmax --check` loads the gate in this project."""
    report = subprocess.run([openmax, "--check", "--json"], cwd=repo, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True)
    try:
        rows = json.loads(report.stdout)
    except ValueError:
        stop(USAGE, f"`{openmax} --check --json` printed no report: {report.stderr.strip()}")
    for row in rows:
        if row["surface"] == "trust":
            stop(NOT_TRUSTED, row["message"])
    shipped = HERE / f"{HOOK}.toml"
    gates = [row for row in rows if row["surface"] == "hook" and Path(row["path"]).stem == HOOK]
    if not gates:
        stop(USAGE, f"the gate is not installed: cp '{shipped}' ~/.openmax/hooks/")
    for row in gates:
        if row["status"] != "ok":
            stop(USAGE, f"the gate {row['path']} does not load here: {row['message']}")
        # A hook of this name that loads is not yet this gate: one without
        # `blocking = true` loads too, and could refuse nothing.
        if Path(row["path"]).read_bytes() != shipped.read_bytes():
            stop(USAGE, f"the gate {row['path']} is not the hook this example ships: cp '{shipped}' over it")


def run_openmax(openmax, task, repo, out, timeout):
    """One print run, its stdout and stderr kept in `out`. Returns the exit
    code, 128 plus the signal's number when a signal ended the run, and
    whether the timeout did. At the timeout the run gets SIGTERM, which
    cancels the turn, and SIGKILL if it is still there 10 seconds later."""
    child, signalled, timed_out = None, [], threading.Event()

    def forward(signum, _frame):
        signalled.append(signum)
        if child is not None:
            child.send_signal(signum)

    def expire():
        timed_out.set()
        child.terminate()

    previous = {signum: signal.signal(signum, forward) for signum in (signal.SIGINT, signal.SIGTERM)}
    timers = [threading.Timer(timeout, expire), threading.Timer(timeout + 10, lambda: child.kill())] if timeout else []
    try:
        with open(out / "stdout.jsonl", "wb") as stdout, open(out / "stderr.txt", "wb") as stderr:
            # A session of its own, so a signal reaches Open Max once, from here.
            child = subprocess.Popen([openmax, "--json", "-p", "--", task], cwd=repo,
                                     env={**os.environ, "RUN_TASK_DIR": str(out)}, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            if signalled:  # one that arrived while the run was starting
                child.send_signal(signalled[0])
            for timer in timers:
                timer.start()
            code = child.wait()
    finally:
        for timer in timers:
            timer.cancel()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if signalled:
        sys.exit(128 + signalled[0])
    code = 128 - code if code < 0 else code
    # A run that ended by itself as the timeout fired was not stopped by it.
    return code, timed_out.is_set() and code >= 128


def read_events(path):
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    return [event for event in events if isinstance(event, dict)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task", help="what to do, as you would type it into a session")
    parser.add_argument("--repo", default=".", help="the repository's root, trusted in auto mode (default: .)")
    parser.add_argument("--check", action="append", default=[], metavar="COMMAND",
                        help="a shell command that refuses the turn's end while it fails; repeat for more")
    parser.add_argument("--final-check", action="append", default=[], metavar="COMMAND",
                        help="a shell command run only after the process exits, for checks too slow for the gate")
    parser.add_argument("--out", help="a new or empty directory outside the repository for the call's files")
    parser.add_argument("--timeout", type=float, metavar="SECONDS", help="stop the run after this long")
    parser.add_argument("--openmax", default="openmax", help="the openmax binary (default: openmax on PATH)")
    args = parser.parse_args()

    if not args.check and not args.final_check:
        stop(USAGE, "give at least one --check or --final-check: with none there is nothing to verify")
    if shutil.which("git") is None:
        stop(USAGE, "no executable git")
    # A caller inside a git command, a hook for one, passes on variables that
    # select its repository over the one `--repo` names. Git lists them.
    for name in git(".", "rev-parse", "--local-env-vars").split():
        os.environ.pop(name, None)
    openmax = shutil.which(args.openmax)
    if openmax is None:
        stop(USAGE, f"no executable {args.openmax}")
    openmax = os.path.abspath(openmax)
    repo = Path(args.repo).resolve()
    if not repo.is_dir() or Path(git(repo, "rev-parse", "--show-toplevel").strip()).resolve() != repo:
        stop(USAGE, f"{repo} is not the root of a git work tree")
    out = Path(args.out).resolve() if args.out else None
    if out is not None and (out == repo or repo in out.parents):
        stop(USAGE, f"--out {out} is inside the repository, where its files would join the diff")
    if out is not None and out.exists() and (not out.is_dir() or any(out.iterdir())):
        stop(USAGE, f"--out {out} is not a new or empty directory")
    gate_is_ready(openmax, repo)
    if out is None:
        out = Path(tempfile.mkdtemp(prefix="run-task-")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_on_signal)
    (out / "checks.json").write_text(json.dumps(args.check))
    for name in ("run_task_checks.py", f"{HOOK}.toml"):  # the checks read their time from the hook file
        shutil.copyfile(HERE / name, out / name)
    # Before the start tree is taken, so files the checks leave behind are in
    # both trees and stay out of the changes.
    before = run_task_checks.run_checks(args.check, repo, run_task_checks.GATE_SECONDS)
    for record in before:
        if record["exit"] is None:
            stop(USAGE, f"`{record['command']}` had no result in the {run_task_checks.GATE_SECONDS} seconds the "
                        "gate's checks share, so it cannot decide the turn's end: pass it as --final-check")

    with tempfile.TemporaryDirectory(prefix="run-task-index-") as scratch:
        start = work_tree(repo, Path(scratch), "start")
        code, timed_out = run_openmax(openmax, args.task, repo, out, args.timeout)
        end = work_tree(repo, Path(scratch), "end")
    with open(out / "changes.patch", "wb") as patch:
        subprocess.run(["git", "-C", str(repo), "diff", "--binary", "--no-ext-diff", start, end],
                       stdin=subprocess.DEVNULL, stdout=patch, check=True)
    changed = [path for path in git(repo, "diff", "--name-only", "-z", start, end).split("\0") if path]

    events = read_events(out / "stdout.jsonl")
    session = next((event["session_id"] for event in events if "session_id" in event), None)
    stop_reason = next((event.get("stop_reason") for event in reversed(events) if event.get("type") == "done"), None)
    declined = sum(1 for event in events
                   if event.get("type") == "approval_settled" and event.get("outcome") == "declined")
    usage = [event for event in events if event.get("type") == "usage"]
    replies = sum(1 for event in events if event.get("type") == "message_done")
    log = Path(os.environ.get("HOME", repo)) / ".openmax" / "sessions" / f"{session}.events.jsonl"
    replay = None
    if session and log.is_file():
        shutil.copyfile(log, out / "events.jsonl")
        replay = shlex.join([openmax, "--replay", str(out / "events.jsonl"), "--json"])

    # The verdict: the checks as this process runs them, on the tree the run
    # left. A run stopped by a signal, or one that never started, gets none.
    ran = code in (0, 1, NOT_VERIFIED)
    after = (run_task_checks.run_checks(args.check, repo, run_task_checks.GATE_SECONDS)
             + run_task_checks.run_checks(args.final_check, repo)) if ran else []
    checks = []
    for index, command in enumerate(args.check + args.final_check):
        gates = index < len(args.check)
        record = after[index] if ran else {}
        checks.append({"command": command, "gates": gates, "before": before[index]["exit"] if gates else None,
                       "after": record.get("exit"), "seconds": record.get("seconds"),
                       "output_tail": record.get("output_tail")})
    failed = [check["command"] for check in checks if check["after"] != 0]

    reason = None
    if timed_out:
        reason = f"stopped at the --timeout of {args.timeout:g} seconds"
    elif code == NOT_VERIFIED:
        reason = f"the turn ended {stop_reason}"
    elif code != 0:
        errors = [event.get("message") for event in events if event.get("type") == "error"]
        errors += (out / "stderr.txt").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        reason = f"openmax exited {code}" + (f": {errors[0]}" if errors else "")
    elif failed:
        code = NOT_VERIFIED
        reason = f"{len(failed)} of {len(checks)} checks failed after the run: " + "; ".join(failed)
    if reason and declined:
        reason += f" (approval requests declined: {declined}, so the project is not in auto mode)"

    result = {
        "format": "run-task/1",
        "verified": code == 0,
        "exit_code": code,
        "reason": reason,
        "stop_reason": stop_reason,
        "session_id": session,
        "repo": str(repo),
        "checks": checks,
        "refusals": sum(1 for event in events if event.get("type") == "turn_refused" and event.get("hook") == HOOK),
        "gate_log": str(out / "gate.jsonl") if (out / "gate.jsonl").is_file() else None,
        "changes": str(out / "changes.patch"),
        "files_changed": changed,
        "usage": {"replies": replies,
                  "prompt_tokens": sum(event.get("prompt_tokens") or 0 for event in usage),
                  "completion_tokens": sum(event.get("completion_tokens") or 0 for event in usage),
                  "cached_tokens": sum(event.get("cached_tokens") or 0 for event in usage)},
        "event_log": str(out / "events.jsonl") if replay else None,
        "replay": replay,
        "out": str(out),
    }
    text = json.dumps(result, indent=2)
    (out / "result.json").write_text(text + "\n")
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
