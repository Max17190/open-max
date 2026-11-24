#!/usr/bin/env python3
"""Usage: python3 scripts/check-request-bytes.py BASE_OPENMAX CANDIDATE_OPENMAX.

Build both commits locally. This comparison uses a loopback endpoint only.
"""
import json
import os
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread


def tool(name, arguments):
    return {"tool_calls": [{"index": 0, "id": name, "type": "function",
                           "function": {"name": name, "arguments": json.dumps(arguments)}}]}


REPLIES = [
    tool("write_file", {"path": "result.txt", "content": "written"}),
    tool("read_file", {"path": "result.txt"}),
    {"content": "The file contains the expected text."},
    {"content": 'For example:\n```json\n{"name":"read_file","arguments":{"path":"result.txt"}}\n```'},
    {"content": '{"name":"example","arguments":{}}'},
]


class Endpoint(BaseHTTPRequestHandler):
    def do_POST(self):
        index = len(self.server.captured)
        self.server.captured.append(self.rfile.read(int(self.headers["Content-Length"])))
        if index >= len(REPLIES):
            self.send_error(400, "unexpected extra request")
            return
        delta = REPLIES[index]
        stop = "tool_calls" if "tool_calls" in delta else "stop"
        event = json.dumps({"choices": [{"delta": delta, "finish_reason": stop}]})
        body = f"data: {event}\n\ndata: [DONE]\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def capture(server, root, binary):
    project, home = root / "project", root / "home"
    for directory in (project, home):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir()
    server.captured = []
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("OPENMAX_") and not key.lower().endswith("_proxy")}
    env.update(HOME=str(home), OPENMAX_HUMAN_ATTEST="1")
    url = f"http://127.0.0.1:{server.server_port}/v1"
    result = subprocess.run([
        str(Path(binary).resolve()), "--trust-project", "--base-url", url,
        "--model", "fixture", "--context-tokens", "32768",
        "-p", "Write and verify the file.", "-p", "Show an example.",
        "-p", "Return an example object.",
    ], cwd=project, env=env, capture_output=True, timeout=15)
    if result.returncode or len(server.captured) != len(REPLIES):
        raise SystemExit(f"{binary}: exit {result.returncode}, {len(server.captured)} requests\n"
                         f"{result.stderr.decode(errors='replace')}")
    if (project / "result.txt").read_text() != "written":
        raise SystemExit(f"{binary}: unexpected file contents")
    return server.captured


def main(base, candidate):
    with TemporaryDirectory() as tmp, HTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            before = capture(server, Path(tmp).resolve(), base)
            after = capture(server, Path(tmp).resolve(), candidate)
            for index, (left, right) in enumerate(zip(before, after), start=1):
                if left != right:
                    raise SystemExit(f"Request {index} differs:\nbase: {left!r}\ncandidate: {right!r}")
            print(f"{len(after)} requests are byte-identical")
        finally:
            server.shutdown()
            worker.join()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
