# Adopt one MCP server as one tool and one skill

Open Max has no MCP client in its agent loop. A server joins through two files
the loop already loads, and the `openmax` binary is the bridge between them
([MCP servers](../../docs/extending.md#mcp-servers), `openmax --spec mcp`):

- a proxy tool, `.openmax/tools/<server>.toml`, whose schema is only
  `{tool, arguments}`. Every request pays for this one small schema.
- a skill, `.agents/skills/<server>/SKILL.md`, whose body lists the server's
  tools. Every request pays for its one index line; the tool list is read when
  the model reads the skill, so a server with many tools costs nothing more
  until it is used.

This directory is that pattern worked through for one server:

| File | Copy to | What it is |
| --- | --- | --- |
| `notes_server.py` | `scripts/notes_server.py` | A tiny stdio MCP server (Python 3, no packages) serving the `.md` files of `notes/` through `list_notes` and `read_note`. Stands in for your server. |
| `notes.toml` | `.openmax/tools/notes.toml` | The proxy tool. |
| `SKILL.md` | `.agents/skills/notes/SKILL.md` | The skill, as the command below generates it. |

## Try it

From the root of a project, with `~/open-max` your clone of this repository:

```sh
mkdir -p scripts .openmax/tools notes
cp ~/open-max/examples/mcp-tool/notes_server.py scripts/
cp ~/open-max/examples/mcp-tool/notes.toml .openmax/tools/
printf 'Deploys are frozen on Fridays.\n' > notes/deploy.md

# Ask the server for its tools, and write them as the skill's body.
tools=$(openmax --mcp-list -- python3 scripts/notes_server.py notes) && mkdir -p .agents/skills/notes &&
printf '%s\n' '---' 'name: notes' 'description: Tools of the notes MCP server; call them with the notes tool' '---' "$tools" > .agents/skills/notes/SKILL.md

# One call through the bridge, the way the proxy tool makes it.
echo '{"tool":"read_note","arguments":{"name":"deploy"}}' | openmax --mcp-call -- python3 scripts/notes_server.py notes

openmax --check
```

`openmax --check` lists the tool and the skill. Then ask a session something
the notes answer, such as "When are deploys frozen?": the model sees the
`notes` skill's line, reads the skill for the tool list, and calls the `notes`
tool with `{"tool": "read_note", "arguments": {"name": "deploy"}}`.

In `ask` and `readonly`, an external tool needs a person's approval. Its first
call in `ask` raises an approval card, or approve it ahead of time:

```sh
openmax --approve .openmax/tools/notes.toml
```

That approves the manifest and `scripts/notes_server.py`, the project file it
runs, and an edit to either revokes it. A server started from outside the
project, such as a command on `PATH`, is covered by the manifest alone.

## Your own server

Change three things, and keep the rest of `notes.toml`:

1. The words after `"mcp"` in `args`: your server's command and arguments,
   run without a shell from the project root.
2. `name`, `description`, and the skill's directory and `name`, so the tool and
   the skill name the same server.
3. `env`: the tool's environment is scrubbed (`PATH`, `HOME`, `LANG`, `TERM`)
   and the server inherits only what the tool gets, so list each credential
   it reads, as in `env = ["NOTES_TOKEN"]`.

Keep `mutating = true` unless every tool the server offers is read-only, and
set `timeout_secs` to cover starting the server and one call; a server that is
slow to start also needs a longer `--mcp-timeout` (30 seconds by default) in
the `--mcp-call` command. Generate the skill again whenever the server's tools
change.

## The limit: no persistent session

Each `--mcp-call` starts the server, performs the initialize handshake, makes
one `tools/call`, and stops it. Nothing the server keeps in memory survives
from one call to the next, a login it does once per process happens on every
call, and every call pays the server's start. That suits a stateless server,
such as one that reads files, queries an API with a token, or runs a lookup.
A server that needs a persistent session is not supported yet.

## Test

```sh
cargo build -p openmax
python3 examples/mcp-tool/test_mcp_tool.py target/debug/openmax
```

It copies the files into a throwaway project, checks that the command above
generates exactly the committed `SKILL.md`, calls the server through
`--mcp-call` (a missing note exits 1 with the server's reason), and drives a
session against a scripted model on 127.0.0.1 that reads the skill and calls
the proxy tool. It asserts that every request's tools array holds the proxy's
`{tool, arguments}` schema and none of the server's own tools, that the system
prompt holds the skill's line but not its tool list, and that the note's text
comes back as the tool result.
