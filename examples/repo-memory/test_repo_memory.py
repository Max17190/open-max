#!/usr/bin/env python3
"""Exercise the example and, optionally, an Open Max binary on loopback."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

EXAMPLE = Path(__file__).with_name("repo_memory.py")
MEMORY = Path(".openmax/memory/repo-verification.md")
REPOSITORY = "https://example.invalid/project"
ENVIRONMENT = "fixture-v1"
CHECK = Path("scripts/check.sh")
RECIPE = b"printf 'verification evidence\\n'\ntouch ran-recipe\n"


def run(argv, cwd=None, env=None, code=0):
    result = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=30, start_new_session=True)
    assert result.returncode == code, (argv, result.returncode, result.stdout, result.stderr)
    return (result.stdout if code == 0 else result.stderr).strip()


def git(repo, *args):
    return run(["git", "-C", repo, *args])


def commit(repo, message):
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", message)


def memory(repo, artifact, action, *args, code=0, repository=REPOSITORY):
    return run([sys.executable, EXAMPLE, "--repo", repo, "--repository", repository,
                "--artifact", artifact, action, *args], code=code)


def clone(seed, path):
    run(["git", "clone", "-q", "--local", seed, path])
    git(path, "remote", "remove", "origin")
    return path


def mock_turn(binary, repo, home, read_recipe):
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if read_recipe and len(requests) == 1:
                call = {"index": 0, "id": "call_read", "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": str(MEMORY)})}}
                delta, finish = {"tool_calls": [call]}, "tool_calls"
            else:
                delta, finish = {"content": "Fixture turn complete."}, "stop"
            chunks = [{"choices": [{"delta": delta, "finish_reason": None}]},
                      {"choices": [{"delta": {}, "finish_reason": finish}]}]
            body = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    home.mkdir()
    env = {"HOME": str(home), "PATH": os.defpath, "OPENMAX_HUMAN_ATTEST": "1"}
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1"
            run([binary, "--trust-project", "--base-url", url, "--model", "fixture", "--context-tokens", "32768",
                 "-p", "Inspect the repository verification workflow.", "--json"], cwd=repo, env=env)
        finally:
            server.shutdown()
            thread.join()
    assert len(requests) == (2 if read_recipe else 1)
    return requests


def exercise(root, binary):
    seed = root / "seed"
    seed.mkdir()
    git(seed, "init", "-q")
    (seed / CHECK.parent).mkdir()
    (seed / CHECK).write_bytes(RECIPE)
    git(seed, "add", str(CHECK))
    commit(seed, "Add public check")
    git(seed, "remote", "add", "origin", REPOSITORY + ".git")

    artifact = root / "recipe.json"
    capture = ["--environment", ENVIRONMENT, "--cwd", "scripts", "--input", str(CHECK), "--", "sh", "check.sh"]
    (seed / CHECK).write_bytes(b"exit 1\n")
    assert "watched inputs differ" in memory(seed, artifact, "capture", *capture, code=1)
    assert not artifact.exists()
    (seed / CHECK).write_bytes(RECIPE)
    assert "verification exited" in memory(seed, artifact, "capture", *capture[:-2], "false", code=1)
    assert not artifact.exists()
    pin = memory(seed, artifact, "capture", *capture)
    assert (seed / CHECK.parent / "ran-recipe").exists()
    assert "artifact already exists" in memory(seed, artifact, "capture", *capture, code=1)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == pin
    recipe = json.loads(artifact.read_bytes())
    assert recipe["source_commit"] == git(seed, "rev-parse", "HEAD")
    assert recipe["inputs"][str(CHECK)]["sha256"] == hashlib.sha256(RECIPE).hexdigest()
    assert recipe["result"]["stdout_sha256"] == hashlib.sha256(b"verification evidence\n").hexdigest()
    assert recipe["result"]["exit_code"] == 0 and recipe["result"]["elapsed_ms"] >= 0
    assert b"verification evidence" not in artifact.read_bytes() and str(root).encode() not in artifact.read_bytes()

    cold = clone(seed, root / "cold")
    warm = clone(seed, root / "warm")
    stage = ["--environment", ENVIRONMENT, "--sha256", pin]
    assert "artifact belongs to a different repository" in memory(
        warm, artifact, "stage", *stage, repository="https://example.invalid/other", code=1)
    assert "artifact SHA-256 does not match" in memory(
        warm, artifact, "stage", *stage[:-1], "0" * 64, code=1)
    assert memory(warm, artifact, "stage", "--environment", "different", "--sha256", pin) == "stale"
    assert not (warm / MEMORY).exists()
    assert memory(warm, artifact, "stage", *stage) == "staged"
    original = (warm / MEMORY).read_bytes()
    assert memory(warm, artifact, "stage", *stage) == "staged"
    assert (warm / MEMORY).read_bytes() == original
    assert not (warm / CHECK.parent / "ran-recipe").exists()

    other = clone(seed, root / "other")
    git(other, "remote", "add", "origin", "https://example.invalid/other.git")
    assert "origin does not match" in memory(other, artifact, "stage", *stage, code=1)
    git(other, "remote", "remove", "origin")
    guard = root / "guard"
    guard.mkdir()
    (other / ".openmax").symlink_to(guard, target_is_directory=True)
    memory(other, artifact, "stage", *stage, code=1)
    assert not list(guard.iterdir())
    (other / ".openmax").unlink()
    destination = other / MEMORY
    destination.parent.mkdir(parents=True)
    protected = guard / "memory.md"
    protected.write_bytes(original)
    destination.symlink_to(protected)
    memory(other, artifact, "stage", *stage, code=1)
    memory(other, artifact, "evict", "--sha256", pin, code=1)
    assert destination.is_symlink() and protected.read_bytes() == original
    destination.unlink()
    destination.write_bytes(original + b"User edit.\n")
    assert "belongs to the user" in memory(other, artifact, "stage", *stage, code=1)
    assert "belongs to the user" in memory(other, artifact, "evict", "--sha256", pin, code=1)
    assert destination.read_bytes() == original + b"User edit.\n"

    descendant = clone(seed, root / "descendant")
    (descendant / "notes.md").write_text("Independent work.\n")
    git(descendant, "add", "notes.md")
    commit(descendant, "Add unrelated notes")
    assert memory(descendant, artifact, "stage", *stage) == "staged"
    assert memory(descendant, artifact, "evict", "--sha256", pin) == "evicted"
    (descendant / CHECK).write_bytes(b"exit 1\n")
    git(descendant, "add", str(CHECK))
    commit(descendant, "Change watched check")
    assert memory(descendant, artifact, "stage", *stage) == "stale"
    assert not (descendant / MEMORY).exists()

    if binary:
        cold_requests = mock_turn(binary, cold, root / "home-cold", False)
        warm_requests = mock_turn(binary, warm, root / "home-warm", True)
        assert str(MEMORY) not in cold_requests[0]["messages"][0]["content"]
        assert str(MEMORY) in warm_requests[0]["messages"][0]["content"]
        assert "Run the verified argv" not in warm_requests[0]["messages"][0]["content"]
        assert any("Run the verified argv" in message["content"] for message in warm_requests[1]["messages"]
                   if message["role"] == "tool")
        assert warm_requests[0]["tools"] == cold_requests[0]["tools"]
        access = (warm / MEMORY).with_name(".access.jsonl").read_text()
        assert any(json.loads(line)["kind"] == "read" for line in access.splitlines())

    (warm / CHECK).write_bytes(b"exit 1\n")
    assert "explicitly evict" in memory(warm, artifact, "stage", *stage, code=1)
    assert (warm / MEMORY).read_bytes() == original
    assert memory(warm, artifact, "evict", "--sha256", pin) == "evicted"
    assert memory(warm, artifact, "stage", *stage) == "stale"
    assert not (warm / MEMORY).exists()
    if binary:
        stale = mock_turn(binary, warm, root / "home-stale", False)
        assert str(MEMORY) not in stale[0]["messages"][0]["content"]
        assert stale[0]["tools"] == cold_requests[0]["tools"]


if __name__ == "__main__":
    binary = Path(sys.argv[1]).resolve(strict=True) if len(sys.argv) > 1 else None
    with tempfile.TemporaryDirectory(prefix="repo-memory-") as temp:
        exercise(Path(temp).resolve(), binary)
    print("Recipe capture, staging and eviction passed" + (" with fresh Open Max sessions" if binary else ""))
