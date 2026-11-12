#!/usr/bin/env python3
"""Run the example end to end with an Open Max binary and a scripted endpoint on loopback."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import run_task_checks  # noqa: E402

TASK = "Make greet.sh print hello."
CHECK = 'test "$(sh greet.sh)" = hello || { echo "greet.sh printed: $(sh greet.sh)"; exit 1; }'
DONE = {"text": "Done."}
FIX = {"tool": "write_file", "args": {"path": "greet.sh", "content": "echo hello\n"}}


def run(argv, cwd, env, code=0):
    result = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == code, (argv, result.returncode, result.stdout, result.stderr)
    return result


@contextmanager
def endpoint(replies):
    """A chat completions server that answers each request with the next
    scripted reply, a text or one tool call, and repeats the last one."""
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            reply = replies[min(len(requests), len(replies)) - 1]
            if "tool" in reply:
                call = {"index": 0, "id": f"call_{len(requests)}", "type": "function",
                        "function": {"name": reply["tool"], "arguments": json.dumps(reply["args"])}}
                delta, finish = {"tool_calls": [call]}, "tool_calls"
            else:
                delta, finish = {"content": reply["text"]}, "stop"
            usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
            chunks = [{"choices": [{"delta": delta, "finish_reason": None}]},
                      {"choices": [{"delta": {}, "finish_reason": finish}], "usage": usage}]
            body = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join()


def fixture(root, name, binary, url, gate=True, trust=True):
    """A home that selects the endpoint, and a repository whose greet.sh
    prints the wrong word, trusted as a human at setup trusts it."""
    home, repo = root / name / "home", root / name / "repo"
    (home / ".openmax" / "hooks").mkdir(parents=True)
    (home / ".openmax" / "settings.json").write_text(json.dumps(
        {"base_url": url, "model": "fixture", "context_tokens": 32768, "max_tokens": 8192}))
    if gate:
        shutil.copy(HERE / "run-task-checks.toml", home / ".openmax" / "hooks")
    env = {"HOME": str(home), "PATH": os.environ["PATH"]}
    repo.mkdir()
    run(["git", "init", "-q"], repo, env)
    (repo / "greet.sh").write_text("echo hullo\n")
    run(["git", "add", "greet.sh"], repo, env)
    run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Add greet"],
        repo, env)
    if trust:
        run([binary, "--trust-project"], repo, {**env, "OPENMAX_HUMAN_ATTEST": "1"})
    return repo, env, root / name / "out"


def call(binary, repo, env, out, *options, code=0):
    result = run([sys.executable, HERE / "run_task.py", "--repo", repo, "--out", out, "--openmax", binary,
                  *options, TASK], repo, env, code)
    return json.loads(result.stdout) if result.stdout else result.stderr


def gate_log(out):
    return [json.loads(line) for line in (out / "gate.jsonl").read_text().splitlines()]


def a_refused_end_goes_back_to_the_model_and_the_next_one_passes(root, binary):
    with endpoint([DONE, FIX, {"text": "Fixed."}]) as (url, requests):
        repo, env, out = fixture(root, "passes", binary, url)
        result = call(binary, repo, env, out, "--check", CHECK)
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


def a_timeout_or_a_signal_stops_the_run(root, binary):
    sleep = {"tool": "bash", "args": {"command": "sleep 60"}}
    with endpoint([sleep]) as (url, requests):
        repo, env, out = fixture(root, "timeout", binary, url)
        started = time.monotonic()
        result = call(binary, repo, env, out, "--check", CHECK, "--timeout", "2", code=143)
        assert time.monotonic() - started < 30
        assert result["stop_reason"] == "cancelled" and result["reason"] == "stopped at the --timeout of 2 seconds"
        assert result["checks"][0]["before"] == 1 and result["checks"][0]["after"] is None, result

        repo, env, out = fixture(root, "signal", binary, url)
        waiting = len(requests)
        child = subprocess.Popen([sys.executable, HERE / "run_task.py", "--repo", repo, "--out", out,
                                  "--openmax", binary, "--check", CHECK, TASK], cwd=repo, env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        deadline = time.monotonic() + 30
        while len(requests) == waiting and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(requests) > waiting, "the run never reached the endpoint"
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=30) == 143 and child.stdout.read() == b""
        # Open Max took the signal and ended its turn before the script exited.
        assert '"stop_reason":"cancelled"' in (out / "stdout.jsonl").read_text().splitlines()[-1]


def a_call_that_cannot_be_held_to_its_checks_does_not_start(root, binary):
    with endpoint([DONE]) as (url, requests):
        repo, env, out = fixture(root, "no-gate", binary, url, gate=False)
        assert "the gate is not installed" in call(binary, repo, env, out, "--check", CHECK, code=2)
        repo, env, out = fixture(root, "untrusted", binary, url, trust=False)
        call(binary, repo, env, out, "--check", CHECK, code=3)
        repo, env, out = fixture(root, "usage", binary, url)
        assert "nothing to verify" in call(binary, repo, env, out, code=2)
        assert "inside the repository" in call(binary, repo, env, repo / "out", "--check", CHECK, code=2)
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


if __name__ == "__main__":
    binary = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="run-task-") as temp:
        root = Path(temp).resolve()
        the_gate_names_a_check_that_outlasts_it_and_fits_what_the_model_is_sent(root)
        a_refused_end_goes_back_to_the_model_and_the_next_one_passes(root, binary)
        a_turn_whose_checks_never_pass_ends_unverified(root, binary)
        the_verdict_is_this_script_s_own_run_of_the_checks(root, binary)
        a_final_check_decides_the_verdict_without_gating_the_end(root, binary)
        a_timeout_or_a_signal_stops_the_run(root, binary)
        a_call_that_cannot_be_held_to_its_checks_does_not_start(root, binary)
        a_project_in_ask_mode_says_why_nothing_was_written(root, binary)
        a_run_open_max_could_not_finish_reports_its_error(root, binary)
    print("The run-task example passed against Open Max")
