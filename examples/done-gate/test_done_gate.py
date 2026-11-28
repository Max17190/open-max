#!/usr/bin/env python3
"""Drive an Open Max binary with the done gate installed, against a scripted
model endpoint on loopback: no model account and no network."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "run-task"))
sys.dont_write_bytecode = True
from scripted_endpoint import endpoint, run  # noqa: E402

CHECK = "sh test.sh"
TEST = 'test "$(sh greet.sh)" = hello || { echo "expected hello, got $(sh greet.sh)"; exit 1; }\n'
DONE = {"text": "Done."}
FIX = {"tool": "write_file", "args": {"path": "greet.sh", "content": "echo hello\n"}}


def project(root, name, binary, url, mode):
    """A repository whose test fails, with the two hook files copied in as the
    README says and the check set to the repository's own test."""
    home, repo = root / name / "home", root / name / "repo"
    (home / ".openmax").mkdir(parents=True)
    (home / ".openmax" / "settings.json").write_text(json.dumps(
        {"base_url": url, "model": "fixture", "context_tokens": 32768, "max_tokens": 8192}))
    hooks = repo / ".openmax" / "hooks"
    hooks.mkdir(parents=True)
    shutil.copy(HERE / "done-gate.sh", hooks)
    manifest = (HERE / "done-gate.toml").read_text()
    line = 'args = [".openmax/hooks/done-gate.sh", "cargo test"]'
    assert manifest.count(line) == 1, manifest
    (hooks / "done-gate.toml").write_text(manifest.replace(line, line.replace('"cargo test"', json.dumps(CHECK))))
    (repo / "greet.sh").write_text("echo hullo\n")
    (repo / "test.sh").write_text(TEST)
    env = {"HOME": str(home), "PATH": os.environ["PATH"]}
    run([binary, "--trust-project", "--permissions", mode], repo, {**env, "OPENMAX_HUMAN_ATTEST": "1"})
    return repo, env


def turn(binary, repo, env, code):
    lines = run([binary, "-p", "--json", "Make greet.sh print hello."], repo, env, code).stdout.splitlines()
    events = [json.loads(line) for line in lines]
    done, = [event for event in events if event["type"] == "done"]
    return events, done["stop_reason"]


def of(events, kind):
    return [event for event in events if event["type"] == kind]


def a_refused_done_goes_back_to_the_model_and_a_passing_check_verifies_the_end(root, binary):
    with endpoint([DONE, FIX, {"text": "Fixed."}]) as (url, requests):
        repo, env = project(root, "auto", binary, url, "auto")
        events, stop = turn(binary, repo, env, 0)
    assert stop == "stop" and len(requests) == 3, (stop, events)
    refusal, = of(events, "turn_refused")
    assert refusal["hook"] == "done-gate", refusal
    assert refusal["reason"].startswith("Not done: `sh test.sh` exits 1."), refusal
    assert "expected hello, got hullo" in refusal["reason"], refusal
    sent = requests[1]["messages"][-1]
    assert sent["role"] == "user" and "expected hello, got hullo" in sent["content"], sent
    verified, = of(events, "turn_verified")
    assert verified["hooks"] == ["done-gate"], verified
    assert (repo / "greet.sh").read_text() == "echo hello\n"


def a_check_that_keeps_failing_ends_the_turn_unverified_after_max_refusals(root, binary):
    with endpoint([DONE]) as (url, requests):
        repo, env = project(root, "stuck", binary, url, "auto")
        events, stop = turn(binary, repo, env, 4)
    # max_refusals = 3: three refusals reach the model, the fourth Done ends the turn.
    assert stop == "unverified" and len(of(events, "turn_refused")) == 3 and len(requests) == 4, (stop, events)
    assert not of(events, "turn_verified"), events


def outside_auto_the_gate_runs_once_a_person_approves_it(root, binary):
    with endpoint([DONE]) as (url, requests):
        repo, env = project(root, "ask", binary, url, "ask")
        # Unapproved, the hook is inert: the end stands and nothing checked it.
        events, stop = turn(binary, repo, env, 0)
        assert stop == "stop" and not of(events, "turn_refused") and not of(events, "turn_verified"), events
        approved = run([binary, "--approve", ".openmax/hooks/done-gate.toml"], repo,
                       {**env, "OPENMAX_HUMAN_ATTEST": "1"}).stdout
        assert "done-gate.sh" in approved, approved
        events, stop = turn(binary, repo, env, 4)
        assert stop == "unverified" and len(of(events, "turn_refused")) == 3, (stop, events)
        # A person fixes the code, and the same approved gate lets the next Done stand.
        (repo / "greet.sh").write_text("echo hello\n")
        events, stop = turn(binary, repo, env, 0)
        assert stop == "stop" and of(events, "turn_verified")[0]["hooks"] == ["done-gate"], events
        # Swapping the check for one that always passes revokes the approval, and the gate fails closed.
        manifest = repo / ".openmax" / "hooks" / "done-gate.toml"
        manifest.write_text(manifest.read_text().replace(json.dumps(CHECK), '"true"'))
        events, stop = turn(binary, repo, env, 4)
        assert stop == "unverified" and not of(events, "turn_verified"), events
        assert "changed since it was approved" in of(events, "hook_failed")[0]["detail"], events


if __name__ == "__main__":
    binary = str(Path(sys.argv[1]).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="done-gate-") as temp:
        root = Path(temp).resolve()
        a_refused_done_goes_back_to_the_model_and_a_passing_check_verifies_the_end(root, binary)
        a_check_that_keeps_failing_ends_the_turn_unverified_after_max_refusals(root, binary)
        outside_auto_the_gate_runs_once_a_person_approves_it(root, binary)
    print("The done gate refused a failing end, verified a passing one, and ran outside auto once approved")
