#!/usr/bin/env python3
"""Load the /plan template with an Open Max binary and run it against a
scripted model endpoint on loopback: no model account and no network."""
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

TASK = "add a --verbose flag to the CLI"
PLAN = "# Plan\n\nGoal: a --verbose flag.\n"
WRITE = {"tool": "write_file", "args": {"path": "PLAN.md", "content": PLAN}}
SUMMARY = {"text": "PLAN.md has 1 step. Review it, then tell me to start."}


def project(root, name, binary, mode):
    """A repository with the template copied where the README puts it."""
    home, repo = root / name / "home", root / name / "repo"
    home.mkdir(parents=True)
    (repo / ".agents" / "prompts").mkdir(parents=True)
    shutil.copy(HERE / "plan.md", repo / ".agents" / "prompts")
    (repo / "cli.py").write_text("print('hello')\n")
    env = {"HOME": str(home), "PATH": os.environ["PATH"]}
    run([binary, "--trust-project", "--permissions", mode], repo, {**env, "OPENMAX_HUMAN_ATTEST": "1"})
    return repo, env


def plan(binary, repo, env, replies):
    with endpoint(replies) as (url, requests):
        lines = run([binary, "--base-url", url, "--model", "fixture", "--context-tokens", "32768", "-p", "--json",
                     f"/plan {TASK}"], repo, env).stdout.splitlines()
    done, = [event for event in map(json.loads, lines) if event["type"] == "done"]
    assert done["stop_reason"] == "stop", lines
    return requests


def the_template_loads_and_expands_as_a_slash_command(root, binary):
    repo, env = project(root, "loads", binary, "auto")
    listed = run([binary, "--check", "--all"], repo, env).stdout
    assert ".agents/prompts/plan.md" in listed, listed
    requests = plan(binary, repo, env, [SUMMARY])
    sent = requests[0]["messages"][-1]
    assert sent["role"] == "user", sent
    body = (HERE / "plan.md").read_text().split("---\n", 2)[2]
    assert sent["content"].strip() == body.replace("$ARGUMENTS", TASK).strip(), sent["content"]
    # The template is message content: the system prompt and tools do not carry it.
    assert "PLAN.md" not in requests[0]["messages"][0]["content"]
    assert "PLAN.md" not in json.dumps(requests[0]["tools"])


def in_auto_the_plan_is_written_to_plan_md(root, binary):
    repo, env = project(root, "auto", binary, "auto")
    plan(binary, repo, env, [WRITE, SUMMARY])
    assert (repo / "PLAN.md").read_text() == PLAN


def in_readonly_no_file_changes_and_the_plan_comes_back_as_the_reply(root, binary):
    repo, env = project(root, "readonly", binary, "readonly")
    shell = {"tool": "bash", "args": {"command": "echo '# Plan' > PLAN.md"}}
    requests = plan(binary, repo, env, [WRITE, shell, {"text": PLAN}])
    for request in requests[1:]:
        refused = request["messages"][-1]
        assert refused["role"] == "tool" and "read-only" in refused["content"], refused
    assert not (repo / "PLAN.md").exists()
    assert (repo / "cli.py").read_text() == "print('hello')\n"


if __name__ == "__main__":
    binary = str(Path(sys.argv[1]).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="plan-template-") as temp:
        root = Path(temp).resolve()
        the_template_loads_and_expands_as_a_slash_command(root, binary)
        in_auto_the_plan_is_written_to_plan_md(root, binary)
        in_readonly_no_file_changes_and_the_plan_comes_back_as_the_reply(root, binary)
    print("The /plan template loaded and expanded, wrote PLAN.md in auto, and changed no file in readonly")
