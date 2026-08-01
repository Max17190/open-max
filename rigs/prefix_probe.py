#!/usr/bin/env python3
"""Measure prompt-prefix stability across the requests one session actually sends.

Prefix caching keys on the token sequence the server renders, which for an
OpenAI-compatible endpoint is (tools, then messages in order). So the two
things that decide a cache hit are:

  * whether `tools` is byte-identical to the previous request, and
  * how many LEADING messages are byte-identical to the previous request.

Neither needs the provider to report anything, which matters because most
OpenAI-compatible servers omit `cached_tokens` entirely. This records the real
request bodies the binary sends and reports both.

  python3 prefix_probe.py BIN [--tool-iterations N] [--json]
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = pathlib.Path(__file__).resolve().parent
PORT = 8823

REQUESTS = []
LOCK = threading.Lock()


def sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    tool_iterations = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(n)
        with LOCK:
            REQUESTS.append(json.loads(raw))
            index = len(REQUESTS)

        # Emit a few tool calls so a single turn spans several requests: that
        # is where prefix stability actually gets exercised, because every
        # tool result appends to the same growing prompt.
        if index <= Handler.tool_iterations:
            body = sse([
                {"choices": [{"delta": {"tool_calls": [{
                    "index": 0, "id": f"c{index}", "type": "function",
                    "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})},
                }]}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ])
        else:
            body = sse([
                {"choices": [{"delta": {"content": "done"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1000, "completion_tokens": 5,
                           "prompt_tokens_details": {"cached_tokens": 0}}},
            ])
        payload = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def canon(msg):
    """One message as the wire carries it, key order fixed so equality means
    equality rather than dict ordering luck."""
    return json.dumps(msg, sort_keys=True)


def leading_identical(a, b):
    n = 0
    for x, y in zip(a, b):
        if canon(x) != canon(y):
            break
        n += 1
    return n


def analyse(reqs):
    rows = []
    for i in range(1, len(reqs)):
        prev, cur = reqs[i - 1], reqs[i]
        pm, cm = prev.get("messages", []), cur.get("messages", [])
        shared = leading_identical(pm, cm)
        tools_same = json.dumps(prev.get("tools"), sort_keys=True) == json.dumps(
            cur.get("tools"), sort_keys=True)
        # Bytes of the shared message prefix, the quantity a cache actually
        # gets to reuse.
        shared_bytes = sum(len(canon(m)) for m in cm[:shared])
        total_bytes = sum(len(canon(m)) for m in cm)
        rows.append({
            "req": i + 1,
            "prev_msgs": len(pm),
            "msgs": len(cm),
            "shared_msgs": shared,
            "tools_stable": tools_same,
            "shared_bytes": shared_bytes,
            "total_bytes": total_bytes,
            "prefix_frac": (shared_bytes / total_bytes) if total_bytes else 1.0,
            "appended_only": shared == len(pm) and tools_same,
        })
    return rows


def main():
    binary = sys.argv[1]
    Handler.tool_iterations = int(sys.argv[sys.argv.index("--tool-iterations") + 1]) \
        if "--tool-iterations" in sys.argv else 4

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    root = HERE / "runs" / f"prefix-{uuid.uuid4().hex[:8]}"
    home, project = root / "home", root / "proj"
    (home / ".openmax").mkdir(parents=True)
    project.mkdir(parents=True)
    (home / ".openmax" / "settings.json").write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{PORT}/v1",
        "model": "stub", "approval_mode": "auto",
    }))
    env = dict(os.environ, HOME=str(home))
    env.pop("OPENMAX_SESSION", None)
    run = subprocess.run([binary, "--trust-project", "-p", "first turn",
                          "-p", "second turn", "-p", "third turn"],
                         cwd=project, env=env, capture_output=True, text=True, timeout=120)
    time.sleep(0.2)

    # A run that never reached the endpoint has nothing to say about prefix
    # stability, and "0 transitions, all append-only" reads exactly like a
    # clean result. Fail loudly instead: a measurement tool that reports
    # success when it measured nothing is worse than no tool.
    if run.returncode != 0:
        print(f"binary exited {run.returncode}; measured nothing\n{run.stderr[:800]}",
              file=sys.stderr)
        shutil.rmtree(root, ignore_errors=True)
        return 1
    if len(REQUESTS) < 2:
        print(f"captured {len(REQUESTS)} request(s); need at least 2 to compare a "
              f"prefix\n{run.stderr[:800]}", file=sys.stderr)
        shutil.rmtree(root, ignore_errors=True)
        return 1

    rows = analyse(REQUESTS)
    if "--json" in sys.argv:
        print(json.dumps({"requests": len(REQUESTS), "rows": rows}, indent=1))
    else:
        print(f"prefix stability over {len(REQUESTS)} real requests "
              f"({Handler.tool_iterations} tool iterations)\n")
        print(f"  {'req':>4} {'msgs':>5} {'shared':>7} {'tools':>6} {'prefix':>8}  verdict")
        for r in rows:
            verdict = "append-only" if r["appended_only"] else "PREFIX BROKEN"
            print(f"  {r['req']:>4} {r['msgs']:>5} {r['shared_msgs']:>7} "
                  f"{'ok' if r['tools_stable'] else 'CHANGED':>6} "
                  f"{r['prefix_frac']*100:>7.1f}%  {verdict}")
        broken = [r for r in rows if not r["appended_only"]]
        print(f"\n  {len(rows) - len(broken)}/{len(rows)} transitions append-only")
        mean = sum(r["prefix_frac"] for r in rows) / len(rows) if rows else 1.0
        print(f"  mean reusable prefix: {mean*100:.1f}% of prompt bytes")
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
