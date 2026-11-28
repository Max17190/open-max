#!/usr/bin/env python3
"""A tiny MCP server over stdio: newline-delimited JSON-RPC 2.0.

It serves the markdown files of one directory, given as its argument, through
two tools, so the example and its test need no network and no package. Replace
it with any stdio MCP server you use.
"""
import json
from pathlib import Path
import sys

NOTES = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
TOOLS = [
    {"name": "list_notes", "description": "List the notes by name",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "read_note", "description": "Read one note",
     "inputSchema": {"type": "object", "required": ["name"],
                     "properties": {"name": {"type": "string", "description": "A name list_notes gave"}}}},
]


def call(name, arguments):
    if name == "list_notes":
        return "\n".join(sorted(path.stem for path in NOTES.glob("*.md"))), False
    if name == "read_note":
        path = NOTES / f"{arguments.get('name', '')}.md"
        if path.parent == NOTES and path.is_file():
            return path.read_text(), False
        return f"no note named {arguments.get('name')!r}", True
    return f"unknown tool {name!r}", True


def answer(request):
    method, params = request.get("method"), request.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}}, "serverInfo": {"name": "notes", "version": "1.0.0"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        text, error = call(params.get("name"), params.get("arguments") or {})
        return {"content": [{"type": "text", "text": text}], "isError": error}
    if method == "ping":
        return {}
    raise LookupError(method)


for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue  # a notification, such as notifications/initialized
    try:
        reply = {"jsonrpc": "2.0", "id": request["id"], "result": answer(request)}
    except LookupError as missing:
        reply = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32601, "message": f"no method {missing}"}}
    print(json.dumps(reply), flush=True)
