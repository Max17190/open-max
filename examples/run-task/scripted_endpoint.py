"""A scripted model endpoint on loopback, and a throwaway project that uses it.

The test and the demo share these: a chat completions server that plays the
model's side from a list of replies, and a private home and repository whose
check fails until greet.sh prints hello.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

HERE = Path(__file__).resolve().parent
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
