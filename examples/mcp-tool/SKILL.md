---
name: notes
description: Tools of the notes MCP server; call them with the notes tool
---
MCP server notes 1.0.0 (protocol 2025-11-25): 2 tools. Call one through this server's proxy tool with {"tool": "<name>", "arguments": {...}}. In each line `?` marks an optional argument; `openmax --mcp-list --json -- <server command>` prints the full schemas.

- list_notes(): List the notes by name
- read_note(name: string): Read one note
