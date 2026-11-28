#!/usr/bin/env python3
"""Adopt the example MCP server with an Open Max binary: generate its skill,
call it through the bridge, and use it from a session against a scripted
model endpoint on loopback. No network and no package install."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "run-task"))
sys.dont_write_bytecode = True
from scripted_endpoint import endpoint, run  # noqa: E402

SERVER = ["python3", "scripts/notes_server.py", "notes"]
NOTE = "Deploys are frozen on Fridays.\n"


def project(root, binary):
    """A repository with the three files copied where the README puts them."""
    home, repo = root / "home", root / "repo"
    home.mkdir()
    (repo / "notes").mkdir(parents=True)
    (repo / "notes" / "deploy.md").write_text(NOTE)
    (repo / "scripts").mkdir()
    shutil.copy(HERE / "notes_server.py", repo / "scripts")
    (repo / ".openmax" / "tools").mkdir(parents=True)
    shutil.copy(HERE / "notes.toml", repo / ".openmax" / "tools")
    env = {"HOME": str(home), "PATH": os.environ["PATH"]}
    run([binary, "--trust-project"], repo, {**env, "OPENMAX_HUMAN_ATTEST": "1"})
    return repo, env


def bridge(binary, repo, env, call, code):
    result = subprocess.run([binary, "--mcp-call", "--", *SERVER], cwd=repo, env=env, input=json.dumps(call),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == code, result
    return result


def the_committed_skill_is_what_the_recipe_generates(repo, binary, env):
    # The recipe's line from `openmax --spec mcp`, run as the README gives it.
    recipe = ('tools=$("$OPENMAX" --mcp-list -- python3 scripts/notes_server.py notes) && '
              "mkdir -p .agents/skills/notes && printf '%s\\n' '---' 'name: notes' "
              "'description: Tools of the notes MCP server; call them with the notes tool' '---' \"$tools\" "
              "> .agents/skills/notes/SKILL.md")
    run(["sh", "-c", recipe], repo, {**env, "OPENMAX": binary})
    generated = (repo / ".agents/skills/notes/SKILL.md").read_text()
    assert generated == (HERE / "SKILL.md").read_text(), generated
    assert "- read_note(name: string): Read one note" in generated, generated


def the_bridge_calls_one_tool_and_reports_a_tool_error(repo, binary, env):
    assert bridge(binary, repo, env, {"tool": "read_note", "arguments": {"name": "deploy"}}, 0).stdout == NOTE
    assert bridge(binary, repo, env, {"tool": "list_notes", "arguments": {}}, 0).stdout.strip() == "deploy"
    failed = bridge(binary, repo, env, {"tool": "read_note", "arguments": {"name": "missing"}}, 1)
    assert "no note named 'missing'" in failed.stderr, failed


def a_session_pays_for_one_proxy_schema_and_reads_the_server_tools_from_the_skill(repo, binary, env):
    replies = [
        {"tool": "read_file", "args": {"path": ".agents/skills/notes/SKILL.md"}},
        {"tool": "notes", "args": {"tool": "read_note", "arguments": {"name": "deploy"}}},
        {"text": "Deploys are frozen on Fridays."},
    ]
    with endpoint(replies) as (url, requests):
        run([binary, "--base-url", url, "--model", "fixture", "--context-tokens", "32768", "-p", "--json",
             "When are deploys frozen?"], repo, env)
    assert len(requests) == 3, requests
    first = requests[0]
    proxy, = [tool for tool in first["tools"] if tool["function"]["name"] == "notes"]
    assert set(proxy["function"]["parameters"]["properties"]) == {"tool", "arguments"}, proxy
    # The server's own tools are in no request's tools array or system prompt: only in the skill's body.
    system = first["messages"][0]["content"]
    assert "Tools of the notes MCP server" in system, system
    for request in requests:
        assert "read_note" not in json.dumps(request["tools"]) and "read_note" not in request["messages"][0]["content"]
    skill_result = requests[1]["messages"][-1]
    assert skill_result["role"] == "tool" and "read_note(name: string)" in skill_result["content"], skill_result
    note_result = requests[2]["messages"][-1]
    assert note_result["role"] == "tool" and NOTE.strip() in note_result["content"], note_result


if __name__ == "__main__":
    binary = str(Path(sys.argv[1]).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="mcp-tool-") as temp:
        repo, env = project(Path(temp).resolve(), binary)
        the_committed_skill_is_what_the_recipe_generates(repo, binary, env)
        the_bridge_calls_one_tool_and_reports_a_tool_error(repo, binary, env)
        a_session_pays_for_one_proxy_schema_and_reads_the_server_tools_from_the_skill(repo, binary, env)
    print("The MCP server was adopted as one proxy tool and one skill, and a session called it")
