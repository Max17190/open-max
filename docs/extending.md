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
hook gets one JSON payload on stdin.

A `post_tool_use` payload carries the tool result the model saw: `output` (its
first 16 KiB, cut on a character boundary), `output_bytes` (that result's
size), and `output_truncated` (whether the payload dropped part of it). That is
what an eval, audit, or telemetry hook needs to be written as a file instead of
a core feature.

Bounding happens twice, and the fields describe the second cut. A tool result
is already a bounded rendering of what a process printed: `bash` keeps the tail
up to its output cap, prepends a `[start of output truncated...]` notice when
it dropped anything, and names a log file holding the bounded capture. A hook
that needs more than the result can read that log; `output_bytes` deliberately
measures the result rather than the process, because that is the text the model
actually reasoned about.

It stays observation: the hook's own exit status and stdout are ignored, so
nothing it does changes what the model receives. Hooks never enter the model prompt and,
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

`tool` is matched by exact name, so a rule naming a tool that does not exist
never fires: a misspelled `deny` reads exactly like a `deny` that never had to
act. `openmax --check` warns about any project rule whose tool is neither a
built-in nor a tool in `.openmax/tools/`. It is a warning rather than an error
because writing the rule before the tool is a normal order to work in. Hook
`tool` filters are matched and reported the same way.

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
be ignored, fail closed, or fail at request time. The agent is instructed to
run it after writing extension files.

Each line is `ok`, `warn`, or `err`, and only `err` exits nonzero:

- `err` is a file the loop cannot use: it does not parse, it fails closed, or
  it can never work (a tool shadowing a built-in name).
- `warn` is a file the loop reads past or cannot act on as written: a path
  nothing reads, a definition another tier overrides, a rule naming a tool
  that does not exist. Each of these is legitimate in some project, so none
  of them fails the check.

Warnings cover the ways a file goes missing without being broken. A directory
at `.openmax/tool/` or `.openmax/skills/`, a `.yaml` where a `.toml` is read,
or a skill directory with no exactly spelled `SKILL.md` are all reported with
the path that would work instead. A global file shadowed by a project file of
the same name is reported against the file that loses, naming the winner. A
broken file that something shadows is a warning, not an error, because the
loop never loads it and so never fails closed on it.

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
