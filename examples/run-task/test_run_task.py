#!/usr/bin/env python3
"""Run the example end to end with an Open Max binary and a scripted endpoint on loopback."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import run_task_checks  # noqa: E402
from scripted_endpoint import CHECK, DONE, FIX, TASK, endpoint, fixture, run  # noqa: E402

# Fails at once when the script runs it. Under the hook, where Open Max marks
# its children, it records its pid and process group and stays.
HELD = 'test -n "$OPENMAX_SESSION" || exit 1; ps -o pid=,pgid= -p $$ > "$RUN_TASK_DIR/held"; sleep 60'


def call(binary, repo, env, out, *options, code=0):
    result = run([sys.executable, HERE / "run_task.py", "--repo", repo, "--out", out, "--openmax", binary,
                  *options, TASK], repo, env, code)
    return json.loads(result.stdout) if result.stdout else result.stderr


def wait_for(condition):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out waiting")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gate_log(out):
    return [json.loads(line) for line in (out / "gate.jsonl").read_text().splitlines()]


def a_refused_end_goes_back_to_the_model_and_the_next_one_passes(root, binary):
    with endpoint([DONE, FIX, {"text": "Fixed."}]) as (url, requests):
        repo, env, out = fixture(root, "passes", binary, url)
        # As a git hook's environment would, the caller's names another repository.
        elsewhere = {**env, "GIT_DIR": str(root / "elsewhere" / ".git"), "GIT_WORK_TREE": str(root / "elsewhere")}
        result = call(binary, repo, elsewhere, out, "--check", CHECK)
    assert result["verified"] and result["exit_code"] == 0 and result["reason"] is None, result
    assert result["stop_reason"] == "stop" and result["refusals"] == 1 and len(requests) == 3, result
    check, = result["checks"]
    assert (check["command"], check["gates"], check["before"], check["after"]) == (CHECK, True, 1, 0), check
    # The refusal reached the model as the next user message, with the check's output.
    refused = requests[1]["messages"][-1]
    assert refused["role"] == "user" and refused["content"].startswith("Not finished: 1 of 1 checks failed"), refused
    assert "greet.sh printed: hullo" in refused["content"], refused
    first, second = gate_log(out)
    assert first["session_id"] == second["session_id"] == result["session_id"], first
    assert (first["continuation"], first["blockable"], first["passed"]) == (0, True, False), first
    assert (second["continuation"], second["passed"], second["checks"][0]["exit"]) == (1, True, 0), second
    # The result carries the change, and the call left nothing else in the repository.
    assert result["files_changed"] == ["greet.sh"] and "+echo hello" in Path(result["changes"]).read_text()
    assert run(["git", "status", "--porcelain"], repo, env).stdout == " M greet.sh\n"
    assert result["usage"] == {"replies": 3, "prompt_tokens": 300, "completion_tokens": 30, "cached_tokens": 0}
    # The replay command needs only the event log: no settings, endpoint or environment.
    stdout = (out / "stdout.jsonl").read_bytes()
    assert json.loads(stdout.splitlines()[0])["session_id"] == result["session_id"]
    replay = subprocess.run(shlex.split(result["replay"]), env={}, capture_output=True, timeout=120)
    assert replay.returncode == 0 and replay.stdout == stdout, replay
    assert json.loads((out / "result.json").read_text()) == result


def a_turn_whose_checks_never_pass_ends_unverified(root, binary):
    with endpoint([DONE]) as (url, requests):
        repo, env, out = fixture(root, "bound", binary, url)
        result = call(binary, repo, env, out, "--check", CHECK, code=4)
    assert not result["verified"] and result["exit_code"] == 4, result
    assert result["stop_reason"] == "unverified" and result["reason"] == "the turn ended unverified", result
    # Eight refusals are honored; the ninth end is judged and reported, and the turn ends.
    assert result["refusals"] == 8 and len(requests) == 9, result
    log = gate_log(out)
    assert [end["continuation"] for end in log] == list(range(9)), log
    assert [end["blockable"] for end in log] == [True] * 8 + [False] and not any(end["passed"] for end in log)
    assert result["checks"][0]["after"] == 1 and result["files_changed"] == [], result


def the_verdict_is_this_script_s_own_run_of_the_checks(root, binary):
    """A model that empties the gate's list ends its turn, and Open Max exits
    0. The checks the script holds still fail, so the result is not verified."""
    empty = {"tool": "bash", "args": {"command": "echo '[]' > \"$RUN_TASK_DIR/checks.json\""}}
    with endpoint([DONE, empty, DONE]) as (url, requests):
        repo, env, out = fixture(root, "defeated", binary, url)
        result = call(binary, repo, env, out, "--check", CHECK, code=4)
    assert result["stop_reason"] == "stop" and result["refusals"] == 1 and len(requests) == 3, result
    assert not result["verified"] and result["reason"].startswith("1 of 1 checks failed after the run"), result


def a_final_check_decides_the_verdict_without_gating_the_end(root, binary):
    with endpoint([DONE]) as (url, requests):
        repo, env, out = fixture(root, "final", binary, url)
        result = call(binary, repo, env, out, "--check", "true", "--final-check", "test -f notes.txt", code=4)
    assert result["stop_reason"] == "stop" and result["refusals"] == 0 and len(requests) == 1, result
    gated, final = result["checks"]
    assert (gated["gates"], gated["before"], gated["after"]) == (True, 0, 0), gated
    assert (final["gates"], final["before"], final["after"]) == (False, None, 1), final


def a_ceiling_ends_the_turn_as_open_max_s_flag_does(root, binary):
    """The ceilings reach Open Max as its own flags, for this run only."""
    with endpoint([FIX, DONE]) as (url, requests):
        repo, env, out = fixture(root, "ceiling", binary, url)
        settings = Path(env["HOME"]) / ".openmax" / "settings.json"
        before = settings.read_bytes()
        result = call(binary, repo, env, out, "--check", CHECK, "--max-agent-iterations", "1", code=4)
        assert "positive number" in call(binary, repo, env, out, "--check", CHECK, "--max-agent-tokens", "0", code=2)
    assert result["stop_reason"] == "max_iterations" and result["reason"] == "the turn ended max_iterations", result
    assert result["ceilings"] == {"max_agent_tokens": None, "max_agent_iterations": 1} and len(requests) == 1, result
    # The one round the ceiling allowed made the fix, so the checks pass afterwards.
    assert result["checks"][0]["after"] == 0 and result["files_changed"] == ["greet.sh"], result
    assert settings.read_bytes() == before


def a_timeout_or_a_signal_stops_the_run(root, binary):
    sleep = {"tool": "bash", "args": {"command": "sleep 60"}}
    with endpoint([sleep]) as (url, requests):
        repo, env, out = fixture(root, "timeout", binary, url)
        started = time.monotonic()
        result = call(binary, repo, env, out, "--check", CHECK, "--timeout", "2", code=143)
        assert time.monotonic() - started < 30
        assert result["stop_reason"] == "cancelled" and result["reason"] == "stopped at the --timeout of 2 seconds"
        assert result["checks"][0]["before"] == 1 and result["checks"][0]["after"] is None, result

    # A signal while the gate runs a check. Open Max lets a turn_end hook run
    # on for up to three seconds, then exits and kills the hook's process
    # group. The check is in that group, not one of its own, so it goes too.
    with endpoint([DONE]) as (url, requests):
        repo, env, out = fixture(root, "signal", binary, url)
        child = subprocess.Popen([sys.executable, HERE / "run_task.py", "--repo", repo, "--out", out,
                                  "--openmax", binary, "--check", HELD, TASK], cwd=repo, env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)

        def held():
            words = (out / "held").read_text().split() if (out / "held").is_file() else []
            return words if len(words) == 2 else None

        pid, group = wait_for(held)
        assert pid != group, (pid, group)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=30) == 143 and child.stdout.read() == b""
        wait_for(lambda: not alive(int(pid)))


def a_call_that_cannot_be_held_to_its_checks_does_not_start(root, binary):
    with endpoint([DONE]) as (url, requests):
        repo, env, out = fixture(root, "no-gate", binary, url, gate=False)
        assert "the gate is not installed" in call(binary, repo, env, out, "--check", CHECK, code=2)
        repo, env, out = fixture(root, "observer", binary, url)
        hook = Path(env["HOME"]) / ".openmax" / "hooks" / "run-task-checks.toml"
        hook.write_text(hook.read_text().replace("blocking = true\n", ""))
        assert "is not the hook this example ships" in call(binary, repo, env, out, "--check", CHECK, code=2)
        repo, env, out = fixture(root, "untrusted", binary, url, trust=False)
        call(binary, repo, env, out, "--check", CHECK, code=3)
        repo, env, out = fixture(root, "usage", binary, url)
        assert "nothing to verify" in call(binary, repo, env, out, code=2)
        assert "inside the repository" in call(binary, repo, env, repo / "out", "--check", CHECK, code=2)
        # The shipped gate's time is honored as written, past the 60 seconds
        # any other hook gets, and the checks share it.
        report = subprocess.run([binary, "--check", "--json"], cwd=repo, env=env, capture_output=True, text=True).stdout
        assert "clamped" not in report and run_task_checks.GATE_SECONDS > 60, report
    assert requests == []


def a_project_in_ask_mode_says_why_nothing_was_written(root, binary):
    with endpoint([FIX, DONE]) as (url, requests):
        repo, env, out = fixture(root, "ask", binary, url)
        subprocess.run([binary, "--stdio"], cwd=repo, env=env, capture_output=True, check=True,
                       input=b'{"cmd":"approval_mode","mode":"ask"}\n{"cmd":"quit"}\n')
        # Outside auto the hook is inert until a human approves it, and the call does not start.
        assert "does not load here" in call(binary, repo, env, out, "--check", CHECK, code=2)
        assert requests == []
        hook = Path(env["HOME"]) / ".openmax" / "hooks" / "run-task-checks.toml"
        run([binary, "--approve", hook], repo, {**env, "OPENMAX_HUMAN_ATTEST": "1"})
        result = call(binary, repo, env, out, "--check", CHECK, code=4)
    assert result["stop_reason"] == "unverified" and result["files_changed"] == [], result
    assert result["reason"].endswith("(approval requests declined: 1, so the project is not in auto mode)"), result


def a_run_open_max_could_not_finish_reports_its_error(root, binary):
    with endpoint([DONE]) as (url, _):
        repo, env, out = fixture(root, "down", binary, url)
    result = call(binary, repo, env, out, "--check", CHECK, code=1)
    assert result["stop_reason"] == "error" and result["reason"].startswith("openmax exited 1: "), result
    assert result["checks"][0]["after"] == 1 and result["usage"]["replies"] == 0, result


def the_gate_names_a_check_that_outlasts_it_and_fits_what_the_model_is_sent(root):
    started = time.monotonic()
    slow, unstarted = run_task_checks.run_checks(["sleep 30", "touch started"], root, 0.2)
    assert slow["exit"] is None and time.monotonic() - started < 10, slow
    assert unstarted["exit"] is None and not (root / "started").exists(), unstarted
    records = [slow] + run_task_checks.run_checks([f"seq 1 2000; exit {code}" for code in (0, 3, 4)], root)
    text = run_task_checks.refusal(records)
    assert len(text) <= run_task_checks.REASON_CHARS and text.startswith("Not finished: 3 of 4 checks failed"), text
    assert "$ sleep 30 (no result within the gate's time)" in text, text
    # Each failing check keeps the end of its output, where a test runner names the failure.
    assert "$ seq 1 2000; exit 3 (exit 3)" in text and text.count("\n2000\n") == 2, text


def the_demo_shows_a_refused_end_and_a_replay_and_leaves_nothing_behind(root, binary):
    """demo.py on a home and temporary directory of its own: it uses neither
    beyond its own directory, which it removes, and says each step it shows."""
    home, temp = root / "demo-home", root / "demo-tmp"
    home.mkdir()
    temp.mkdir()
    started = time.monotonic()
    result = run([sys.executable, HERE / "demo.py", "--openmax", binary], root,
                 {"HOME": str(home), "TMPDIR": str(temp), "PATH": os.environ["PATH"]})
    assert time.monotonic() - started < 60
    text = result.stdout
    steps = ["scripted by a server on 127.0.0.1", "greet.sh printed: hullo", "model: 'Done.'",
             "gate:  the check exits 1, so Open Max refuses the end and sends the model:",
             "| Not finished: 1 of 1 checks failed", "model: a write_file call", "+echo hello",
             '"verified": true', "identical to the run's stdout, byte for byte", "Removed "]
    assert all(step in text for step in steps), text
    assert [text.index(step) for step in steps] == sorted(text.index(step) for step in steps), text
    assert not any(home.iterdir()) and not any(temp.iterdir()), (list(home.iterdir()), list(temp.iterdir()))


if __name__ == "__main__":
    binary = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="run-task-") as temp:
        root = Path(temp).resolve()
        the_gate_names_a_check_that_outlasts_it_and_fits_what_the_model_is_sent(root)
        a_refused_end_goes_back_to_the_model_and_the_next_one_passes(root, binary)
        a_turn_whose_checks_never_pass_ends_unverified(root, binary)
        the_verdict_is_this_script_s_own_run_of_the_checks(root, binary)
        a_final_check_decides_the_verdict_without_gating_the_end(root, binary)
        a_ceiling_ends_the_turn_as_open_max_s_flag_does(root, binary)
        a_timeout_or_a_signal_stops_the_run(root, binary)
        a_call_that_cannot_be_held_to_its_checks_does_not_start(root, binary)
        a_project_in_ask_mode_says_why_nothing_was_written(root, binary)
        a_run_open_max_could_not_finish_reports_its_error(root, binary)
        the_demo_shows_a_refused_end_and_a_replay_and_leaves_nothing_behind(root, binary)
    print("The run-task example passed against Open Max")
