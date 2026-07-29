# Extending Open Max

With nothing installed, extensions cost zero tokens. Project paths win over
global ones on name collision.

## Tools

A TOML file in `.openmax/tools/` or `~/.openmax/tools/`. The harness runs
`command`, writes JSON args to stdin, and returns stdout. These native
processes inherit the host filesystem, environment, credentials, and network
access of Open Max.

```toml
# .openmax/tools/todo_scan.toml
name = "todo_scan"
description = "List TODO/FIXME comments with file and line"
command = "./scripts/todo-scan.sh"
timeout_secs = 30
mutating = false

[params]
type = "object"
[params.properties.path]
type = "string"
description = "Directory to scan"
```

`mutating` is trusted metadata for scheduling and approval behavior. It is not
a security boundary and does not restrict what the command can do. Unknown keys
in a tool file are rejected, so a misspelled `mutating` surfaces in
`openmax --check` instead of silently taking the tool out of the approval gate.

## Skills

A directory with `SKILL.md` under `.agents/skills/` or `~/.openmax/skills/`.
Only `name` and `description` live in the prompt; the model reads the full file
when needed.

```
.agents/skills/release/SKILL.md
---
name: release
description: How to cut a release of this project
---
Full instructions, checklists, commands...
```

## Prompt templates

A markdown file under `.agents/prompts/` or `~/.openmax/prompts/`; the file
stem becomes a slash command. `$ARGUMENTS` expands to the raw argument string,
`$1`..`$9` to positionals, and `$$` escapes a literal dollar; a template
without placeholders gets the arguments appended. Templates are message
content: re-read on every use, zero prompt tax, never frozen.

```
.agents/prompts/fix-issue.md
---
description: Fix a GitHub issue by number
---
Fetch issue $1 with `gh issue view $1`, reproduce it, fix it, and add a test.
```

Run it as `/fix-issue 42`.

## Hooks

Optional process gates under `.openmax/hooks/` or `~/.openmax/hooks/`.
`pre_tool_use` and `user_prompt_submit` can block (nonzero exit; the blocked
prompt never reaches the model); `post_tool_use`, `session_start` (a session's
first turn), `compaction` (context was pruned; receives the digest record), and
`turn_end` (receives the stop reason, fires even on cancel) observe only. Each
hook gets one JSON payload on stdin. Hooks never enter the model prompt and,
like external tools and `bash`, run as native host processes with inherited
filesystem, environment, credentials, and network access.

Unknown keys in a hook file are rejected, and a hook file that does not parse
blocks every tool call until it is fixed or removed (fail closed, like
permissions): a broken file might have been a gate, and `openmax --check`
prints the reason.

## Permissions

Optional rules under `.openmax/permissions.toml` or
`~/.openmax/permissions.toml` (project first). Not in the model prompt; empty
discovery is free. First match wins. Order: hooks pre → permissions →
`approval_mode` → execute → hooks post.

```toml
# .openmax/permissions.toml
[[rules]]
effect = "deny"
tool = "bash"
arg_regex = "rm\\s+-rf"

[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^cargo (test|check|build)"
```

`effect` is `allow`, `deny`, or `ask`. `arg_regex` is optional: command for
`bash`, path for file tools, pattern for `glob`/`grep`. For custom tools it
matches the full serialized JSON arguments. Omit `arg_regex` (or leave it
empty) to match every call of that tool.

### Fail closed, with a repair path

If a permissions file exists but is invalid, every tool is denied (fail
closed), with one exception: rewriting that same file with `write_file` or
`edit_file` stays available, so a typo in a rule is repairable from inside the
session instead of ending it. The repair is exempt from the rules, not from
approvals: `approval_mode` still applies, so `ask` prompts for it and
`readonly` refuses it.

This covers the project file. A malformed global
`~/.openmax/permissions.toml` sits outside the project root, where file tools
never write, so fix that one from the shell; the deny reason names the exact
path and `openmax --check` prints the parse error.

## Validation

`openmax --check` parses tools, skills, templates, hooks, permissions, and
`providers.json`, then prints per-file results with the reason anything would
be ignored, fail closed, or fail at request time. It exits nonzero on errors.
The agent is instructed to run it after writing extension files.

## Self-description

`openmax --spec <surface>` prints the complete authoring contract for one
surface (`tools`, `skills`, `prompts`, `hooks`, `permissions`, `providers`, or
`stdio`): file grammar, field caps and defaults, hook stdin payload shapes, and
activation timing. The frozen prompt carries only a one-line pointer to it, so
the full contract costs zero tokens until the agent reads it, and the printed
examples are parsed by the same validation code in tests so the text cannot
drift from the binary.

## Freezing and re-freezing

Tools and skills freeze per freeze window for prompt-cache stability, and the
harness re-freezes them automatically. At each turn start, and again between
iterations after a mutating tool call succeeded, it captures one immutable
generation of extension bytes, fingerprints that snapshot, and parses those
same bytes. If the generation changed, it rebuilds the registry and prompt in
place (one deliberate cache re-prefill, conversation kept, a `refrozen` event
for clients), so a tool the agent writes is callable on its very next step,
inside the same turn, with no human action. Atomic replacement and symlink
swaps cannot make the activated registry disagree with its fingerprint. An
unchanged generation does not rebuild or invalidate the prompt cache.
`/reload` forces a new capture immediately; `/new` starts clean. Hooks,
permissions, and templates re-discover on every turn or invocation. Use
`/tools`, `/skills`, and `/context` to inspect the frozen set and its cost.
