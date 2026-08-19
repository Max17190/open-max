//! `openmax --spec <surface>`: the complete authoring contract for each
//! extension surface, printed by the binary that enforces it.
//!
//! The self-extension guide in the frozen prompt is an index: it names the
//! surfaces and their file paths in ~360 tokens. This module is the body the
//! agent reads on demand (via bash), the same progressive disclosure skills
//! use, so knowing the exact grammar of a hook payload or a permissions rule
//! costs zero prompt tokens until the moment it is needed. Every fenced
//! example below is written to disk and run through the real extension
//! parsers in tests, so the printed contract cannot drift from the loop.

/// Every surface `render` accepts, in the order the help text lists them.
pub const SURFACES: [&str; 11] = [
    "tools",
    "skills",
    "prompts",
    "hooks",
    "permissions",
    "providers",
    "settings",
    "memory",
    "recall",
    "stdio",
    "usage",
];

/// The authoring contract for one surface, or None for an unknown name.
pub fn render(surface: &str) -> Option<&'static str> {
    match surface {
        "tools" => Some(TOOLS),
        "skills" => Some(SKILLS),
        "prompts" => Some(PROMPTS),
        "hooks" => Some(HOOKS),
        "permissions" => Some(PERMISSIONS),
        "providers" => Some(PROVIDERS),
        "settings" => Some(SETTINGS),
        "memory" => Some(MEMORY),
        "recall" => Some(RECALL),
        "stdio" => Some(STDIO),
        _ => None,
    }
}

/// The keys each TOML manifest surface cannot omit, in the order its contract
/// above lists them. They live beside the contract so the message about an
/// absent key and the document that demands it cannot drift apart.
const REQUIRED_FIELDS: [(&str, &str); 2] =
    [("tools", "name, description, command"), ("hooks", "event, command")];

/// Render a manifest's `toml::de::Error` for one surface.
///
/// A toml error Displays as `TOML parse error at line L, column C` over a
/// caret block. For a required key that is simply absent that header is
/// false: the file parsed, serde then asked for a key nobody wrote, and the
/// span it reports covers the whole document, so the carets land on line 1
/// and invite an edit to an innocent line. Absent keys are named plainly
/// instead, with the surface's required set and where to read the rest of the
/// contract. Syntax and type errors keep the location, which is where they
/// really are.
///
/// Both manifest structs are flat, so `missing field` can only mean a
/// top-level key: there is no nested span this discards.
pub(crate) fn manifest_toml_error(err: &toml::de::Error, surface: &str) -> String {
    let Some(field) = err
        .message()
        .strip_prefix("missing field `")
        .and_then(|rest| rest.strip_suffix('`'))
    else {
        return format!("invalid TOML: {err}");
    };
    match REQUIRED_FIELDS.iter().find(|(name, _)| *name == surface) {
        Some((_, required)) => format!(
            "missing required field '{field}': required fields are {required} (openmax --spec {surface})"
        ),
        None => format!("missing required field '{field}' (openmax --spec {surface})"),
    }
}

const TOOLS: &str = r#"# External tools

(Every `openmax ...` command below means the binary running THIS session:
`$OPENMAX_BIN` is set on every process the harness spawns. A bare `openmax`
on PATH may be a different, older build that prints the same version.)

One TOML file per tool: `.openmax/tools/<name>.toml` (project) or
`~/.openmax/tools/<name>.toml` (global). Project wins on name collision.
A tool named like a built-in (list_dir, read_file, write_file, edit_file,
glob, grep, bash) is ignored.

Fields:
- `name` (required): 1-64 chars of [a-zA-Z0-9_-]. Becomes the tool's schema name.
- `description` (required): one line; newlines are collapsed; capped at 200
  chars in the schema. Rides in every request, so keep it short and put long
  usage docs in a README the description points at.
- `params` (optional): a JSON-schema object, written as TOML tables. Omitted
  means "no parameters".
- `command` (required): executable path or name, spawned in the project root.
- `args` (optional): fixed argv strings appended to `command`.
- `timeout_secs` (optional): default 60, clamped to 1..300.
- `mutating` (optional, default false): routes calls through the ordinary
  approval_mode gate (snapshots, read-only sessions, prompts). Trusted
  metadata for scheduling and UX, not a sandbox: it never widens or narrows
  the human content approval below.
- `env` (optional, default none): the environment variable NAMES this tool
  receives from the harness's environment, e.g. `env = ["GITHUB_TOKEN"]`
  (at most 16). Everything not listed is scrubbed: the tool always gets a
  baseline (PATH, HOME, LANG, TERM), and nothing else - API keys included -
  unless named here. The list is manifest bytes, so the credential grant is
  part of what the human approves.

Runtime contract: the harness spawns `command args...` in the project root,
writes the call's JSON arguments to stdin as one newline-terminated line, and
returns stdout as the result.
Nonzero exit makes the result an error carrying `exit code N` plus output.
Output is capped; overflow spills to `~/.openmax/cmd-logs`, pruned after
7 days. The process is a
native host process with the network and filesystem authority of Open Max;
its environment is the scrubbed baseline plus the manifest's `env` list.

Human approval: because of that authority, the first call of any tool file -
mutating or not - stops for a human, who approves the exact bytes. Later calls
of identical bytes run unprompted; any edit revokes and asks again. Approve
at an interactive terminal outside any session with `openmax --approve
.openmax/tools/<name>.toml` (a human act: it refuses processes carrying the
session marker and callers with no terminal, so it never happens by accident
from a session; a shell that clears the marker or attests is a command the
human's ask rule on bash puts in front of them), or by approving the write
that created it. `openmax --spec usage` lists the approval state of every
installed tool.

What "the exact bytes" covers is the whole definition: the `.toml` *and* the
project-local file its `command` (or a path in `args`) names, because that file
is the code that actually runs and the agent can rewrite it after the fact.
Editing the manifest or that script makes the next call ask again. A `command`
outside the project root (an absolute path, a name on PATH) is covered by the
manifest approval alone - that path is what the human read - while a command
resolving to no file at all is covered by nothing, so the tool asks until it
exists. `openmax --approve <tool.toml>` approves the pair up front and prints
every path and hash it blessed.

Binding reaches the files a manifest *names*. A command handed a program on
its own command line (`python3 -c "..."`, `sh -c "..."`) has that program text
bound - it is part of the manifest - but anything the program opens while it
runs is chosen at runtime and is not. Put the program in a project file and
name it in `args`, and its bytes are covered too; `openmax --check` warns when
inline text reads a project file.

Example (`.openmax/tools/todo_scan.toml`):

```toml
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

Activation: automatic when extension bytes change, checked between
iterations after a successful mutating call and at turn start; `/reload`
forces it now. Verify the file parses with `openmax --check`. Test the
script itself before first use:
`echo '{"path":"src"}' | ./scripts/todo-scan.sh`.

## Proof of life

An optional `[example]` table declares one runnable call:

```toml
[example]
expect_regex = "TODO"      # optional; output must match when set
[example.args]
path = "src"               # the JSON arguments for the example call
```

`openmax --check --run-examples` executes each declared example through the
real spawn path (stdin JSON, timeout, output caps): the call must exit 0 and
match `expect_regex` when present. `expect_regex` is matched against the same
capped rendering a tool call returns (the tail, once output exceeds the cap),
so a token printed early by a noisy command can scroll out of the window.
Plain `--check` never executes anything.

How an example runs depends on whether a human has approved the tool's exact
bytes (manifest plus the project-local code it names):

- UNAPPROVED (the common case while you are still writing it): the example
  runs as a SANDBOXED PROBE - no network, writes confined to a scratch dir
  (which is also its HOME and TMPDIR), scrubbed environment - with zero host
  authority granted, so it needs no approval, no `ask`-mode person, and you
  may run it mid-turn as often as you like. The verdict is labeled
  `[sandboxed probe: ...]`, and a passing probe leaves evidence that the
  approval card later shows the human ("probe: example passed in sandbox").
  A passing probe approves NOTHING: in-session calls still stop for the card
  until a human approves the bytes. On a host with no sandbox backend the
  probe is refused (never run unsandboxed) and the refusal says so.
- APPROVED: the example is the tool's real command with the harness's host
  authority, run in the project root with no sandbox and no snapshot, behind
  the same gates a turn applies: `pre_tool_use` hooks allow it; permission
  rules admit it (`deny` refuses, and so does `ask`, because nothing here can
  prompt - write `effect = "allow"` for a tool whose example should run
  unattended); and `approval_mode` is not `readonly` for a `mutating` tool
  (a `mutating` example under `approval_mode = "ask"` needs a human to start
  the run, so it is refused when the agent loop spawned the process).

Both need the project trusted (`openmax --trust-project`). Each refusal names
what to fix. An example must not itself run `openmax --check --run-examples`:
that is refused rather than recursed into.
"#;

const SKILLS: &str = r#"# Skills

One directory per skill with a `SKILL.md` inside: `.agents/skills/<name>/SKILL.md`
(project) or `~/.openmax/skills/<name>/SKILL.md` (global). Project wins on
name collision.

`SKILL.md` starts with a `---`-delimited frontmatter block; only two scalar
keys are read (bare or double-quoted values, one line each):
- `name:` (required): the skill's index name.
- `description:` (required in practice): one line saying what the skill does
  and when to use it; capped at 200 chars. This is the only text that enters
  the prompt (~15 tokens per skill), so it must carry the "when".

The body after the frontmatter is free-form markdown of any length: it loads
only when the skill is used. A skill directory may bundle scripts and
reference files next to SKILL.md; run them with bash.

At most 50 skills are indexed, sorted by name. The prompt shows each as
`- name: description — path/to/SKILL.md`.

Example (`.agents/skills/release/SKILL.md`):

```markdown
---
name: release
description: How to cut a release of this project
---
Full instructions, checklists, and commands, read on demand.
```

Activation: automatic when extension bytes change, checked between
iterations after a successful mutating call and at turn start; `/reload`
forces it now. Verify with `openmax --check`.
"#;

const PROMPTS: &str = r#"# Prompt templates

One markdown file per template: `.agents/prompts/<name>.md` (project) or
`~/.openmax/prompts/<name>.md` (global). Project wins. The file stem becomes
the user's slash command `/<name>` and must be 1-64 chars of [a-zA-Z0-9_-].

Structure: an optional `---` frontmatter block with a one-line `description:`
(shown in the completion popup, capped at 200 chars), then the body. An empty
body is invalid, and so is a block that opens with `---` and never closes with
`---` (the fence and its keys would otherwise expand as message text).

Substitution when the user runs `/<name> args...`:
- `$ARGUMENTS` expands to the raw argument string.
- `$1`..`$9` expand to whitespace-split positionals; missing ones become
  empty; `$12` stays literal (only single digits exist).
- `$$` escapes a literal dollar (`$$5` survives as `$5`).
- A body with no placeholders gets the arguments appended after a blank line.

Templates are message content: re-read on every invocation, never frozen,
zero prompt tax. They are user-invoked; the model does not call them (read
the file directly if its content is needed).

Every front end expands the same line: the TUI composer, `openmax -p
"/<name> args"`, and a stdio `{"cmd":"user","text":"/<name> args"}`. So a
delegated child process can invoke them too. Expansion is single-pass (a body
starting with `/` is text, not another invocation), and in the TUI a built-in
slash command with the same name wins.

Example (`.agents/prompts/fix-issue.md`):

```markdown
---
description: Fix a GitHub issue by number
---
Fetch issue $1 with `gh issue view $1`, reproduce it, fix it, and add a test.
```

Activation: next invocation (no freeze involved). Verify with `openmax --check`.
"#;

const HOOKS: &str = r#"# Hooks

One TOML file per hook: `.openmax/hooks/<name>.toml` (project) or
`~/.openmax/hooks/<name>.toml` (global). A project file shadows a global file
with the same stem. Unknown keys are rejected.

Fields:
- `event` (required): one of `pre_tool_use`, `post_tool_use`,
  `user_prompt_submit`, `session_start`, `compaction`, `turn_end`.
- `command` (required): executable, spawned in the project root.
- `args` (optional): fixed argv strings.
- `timeout_secs` (optional): default 10, clamped to 1..60.
- `tool` (optional): exact tool-name filter for `pre_tool_use`/`post_tool_use`.
- `blocking` (optional): `turn_end` only, default false. True makes the hook
  a completion gate: a nonzero exit refuses the turn's end and the reason is
  sent back as a user message. False observes only; exit status is ignored.
  Rejected on every other event, which either always gates or never does.

Gate events (`pre_tool_use`, `user_prompt_submit`): a nonzero exit blocks the
call or the prompt. The block reason is the hook's stdout (or stderr if stdout
is empty), capped at 500 chars. A blocked tool call returns to the model as a
failed tool result carrying the reason; a blocked prompt never reaches the
model. A gate that times out or fails to start blocks.

Observe events (`post_tool_use`, `session_start`, `compaction`, and `turn_end`
without `blocking`): exit status is ignored. `session_start` fires on a
session's first turn; `compaction` fires after context was pruned; `turn_end`
fires with the stop reason, even on cancel. Hooks never inject text into the
model context.

Conditional gate (`turn_end` with `blocking = true`): a nonzero exit refuses
the turn's completion. The reason is sent back as a user message and the turn
continues from there, spending the same iteration and token budgets any other
continuation would. Only one exit can be refused: the model falling silent
with no tool calls, whatever `stop_reason` it carried. `max_iterations`,
`budget_exhausted`, `cancelled`, `error`, and `truncated` still fire the hook,
but there is nothing left to honor a refusal with, so it is reported and the
turn ends. The payload says which case this is (`blockable`), how many
refusals have already been honored (`continuation`), and how many are left
(`continuations_left`). After 8 honored refusals the harness overrides the
hook and ends the turn with `stop_reason` `unverified`; `openmax -p` exits 4 on
that, and on `max_iterations` and `budget_exhausted`. A refusal whose injected
user message cannot be persisted is reported and ends the turn `unverified`
the same way: a continuation only the running process remembers would diverge
from every replay of the session. A blocking hook that
times out or fails to start refuses, like any other gate.

Each run receives one JSON payload on stdin, as one newline-terminated line:
- pre_tool_use: {"event", "session_id", "tool", "args", "cwd", "tool_ok"}
  where `tool_ok` is null, because the call has not run yet.
- post_tool_use: {"event", "session_id", "tool", "args", "cwd", "tool_ok",
  "output", "output_bytes", "output_truncated", "process_bytes",
  "process_truncated"} where `tool_ok` is a boolean. A hook sees the tool
  result the model saw, bounded twice and told about both bounds. `output` is
  the result's first 16 KiB cut on a character boundary, `output_bytes` is the
  result's size, and `output_truncated` says whether this payload dropped part
  of it. Behind that, `process_bytes` is how many bytes the command actually
  produced (null when the tool ran no process) and `process_truncated` says
  whether the result dropped part of that, so an audit hook can tell a quiet
  command from a clipped one. A call killed by timeout or cancel reports the
  bytes it printed before it died, even though the result carries none of
  them. `bash` also names its bounded output log in the result text. Reading any of this changes nothing: an observe event cannot
  alter what the model receives.
- user_prompt_submit: {"event", "session_id", "cwd", "text"}
- session_start: {"event", "session_id", "cwd"}
- compaction: {"event", "session_id", "cwd", "record"} where `record` is the
  persisted compaction digest.
- turn_end: {"event", "session_id", "cwd", "stop_reason", "blockable",
  "continuation", "continuations_left"} where `blockable` says whether a
  nonzero exit will be honored this time, `continuation` counts the refusals
  already honored in this turn (0 on the first end attempt), and
  `continuations_left` is what remains before the harness overrides the hook.
  A hook without `blocking` always sees `blockable` false.

Approval: a hook is inert until a human approves its exact content - the
`.toml` *and* the project-local file its `command` (or a path in `args`) names,
because that file is the code that actually runs and the agent can rewrite it.
`openmax --approve <hook.toml>` approves the pair and prints both; an
in-session write approval never stands in for it. A `command` outside
the project root (an absolute path, a name on PATH) is covered by the manifest
approval alone: that path is what the human read, and system binaries change
on their own schedule. The bytes are re-checked before every run, so a script
rewritten mid-turn does not run. Inline program text (`command = "sh", args =
["-c", "..."]`) is bound as text and no further: what that program opens at
runtime is not covered, so a gate written that way can be defanged by editing
the file it sources. Name a script in `args` instead - `openmax --check` warns
when inline text reads a project file.

Fail closed, four ways, all reported by `openmax --check`:
- A hook file that exists but does not parse blocks every tool until it is
  fixed or removed (a broken file might have been a gate), unless a valid
  project file shadows its stem - or unless no human ever approved that path,
  in which case it never ran and stays inert instead.
- A gate hook (`pre_tool_use`, `user_prompt_submit`, or a `turn_end` a human
  approved with `blocking = true`) whose path was approved and whose content no
  longer is blocks every tool until the approved content is restored or a human
  re-approves it. Editing a live gate cannot turn it off, and that includes a
  comment-only edit or a rewritten script. Whether a modified hook counts as a
  gate is decided by the shape a human approved, never by what the current file
  declares: rewriting an approved `pre_tool_use` gate into an observe hook, or
  dropping `blocking` from an approved `turn_end` gate, would otherwise stop it
  gating, so it reads as a demoted gate and still fails closed. A record
  written before `blocking` existed is read as the observer a human approved,
  so no upgrade promotes an old `turn_end` hook into a gate. The repair carve-out is
  scoped the same way, to the manifest plus the code the *approved* content
  named, so a rewritten hook cannot hand itself an exemption for a new path.
- A hook file a human approved that is *deleted* blocks every tool the same
  way. Deleting a gate is easier than rewriting one and leaves nothing on disk
  to notice, so what is enforced is reconciled against the approved paths, not
  against the directory listing. Restore the file, or run
  `openmax --forget <path>` if the removal was intended. `--forget` removes a
  policy rather than adding one, so it refuses inside an agent session and
  refuses without an interactive terminal, then asks for the path typed back.
  Read that as a speed bump, not a boundary: it removes the one-command
  bypass, but an attacker holding `bash` can still allocate a pty to answer
  the prompt. What it can no longer do is edit the approval store: approvals
  are records in the ledger's hash chain, so writing one outside the harness
  reads as tamper and grants nothing. Erasing the whole ledger directory still
  works, and surrenders every approval - fail closed, so gates stop running
  rather than start. Without an OS sandbox nothing here survives an agent that
  is determined and has a shell; what the harness guarantees is that the easy
  path is closed and the state is visible.
- A `command` (or `args` path) that resolves to no file at all is not covered
  by anything: there is nothing to approve, so the hook does not load. Install
  or create it, then approve.
- In every case, `write_file`/`edit_file` on the offending hook file or the
  code it runs stays available, so the repair is possible from inside the
  session (same carve-out as `permissions.toml`, and subject to
  `approval_mode`). It resolves paths before comparing them, so a file that
  does not exist yet can be recreated while `../` cannot be aimed outside.

A hook whose content was never approved is inert rather than blocking: it never
ran, so it removes no policy. It is not silent - each turn reports it, and
`openmax --check` names it. Hooks are re-discovered every turn; no reload is
needed.

Example (`.openmax/hooks/deny-rm.toml`):

```toml
event = "pre_tool_use"
command = "./scripts/deny-rm.sh"
tool = "bash"
timeout_secs = 5
```

Verify with `openmax --check`. Test the script directly:
`echo '{"event":"pre_tool_use","tool":"bash","args":{"command":"ls"}}' | ./scripts/deny-rm.sh`.
"#;

const PERMISSIONS: &str = r#"# Permission rules

Optional declarative rules: `.openmax/permissions.toml` (project), then
`~/.openmax/permissions.toml` (global). Project rules are evaluated first and
the first matching rule wins. Evaluation order per call:
hooks pre → permissions → approval_mode → execute → hooks post.

Grammar: `[[rules]]` tables only; unknown keys anywhere are rejected.
- `effect` (required): `"allow"`, `"deny"`, or `"ask"`.
- `tool` (required): exact tool name. A name no tool has never matches,
  so the rule silently does nothing; `openmax --check` warns about that.
- `arg_regex` (optional): unanchored regex; omitted or empty matches every
  call of that tool.

What the regex matches:
- `bash` → the command string.
- `read_file`, `write_file`, `edit_file`, `list_dir` → the path argument.
- `glob`, `grep` → the pattern argument.
- any other tool → the full serialized JSON arguments.

A rule (and a `pre_tool_use` hook) sees TEXT, not effects. A `deny` on
`bash` matches the command STRING, so it stops the spellings your regex
anticipates and nothing else: `rm src/x` is caught, but `python3 -c
'os.remove(...)'`, `truncate -s 0 src/x`, `find src -delete`, and `> src/x`
are all different strings that reach the same file. A `tool = "bash"` rule
also does not gate `write_file`/`edit_file` at all - those are separate tools.
So a permission rule is friction against known patterns, NOT a filesystem
guarantee: if a user asks for a hard guarantee that some path cannot be
written, say plainly that these gates cannot deliver one, and offer what they
can - `approval_mode = "ask"` so every call CLASSIFIED as mutating and not
covered by an approved `allow` rule is shown (an approved `allow` still runs it
unprompted, so drop the rules that cover the path you want to see - and that is
not only path-scoped `write_file`/`edit_file` allows: a `bash` allow matches
COMMAND TEXT, so an approved `allow` on a `bash` command that can mutate the
path, e.g. `truncate -s 0 src/x`, force-allows it with no prompt, and a
`write_file`/`edit_file` rule does not constrain bash at all - so remove or
narrow every effective allow, bash included, whose command could reach the
path), an `ask` rule so the patterns you name prompt rather than run, or moving
the files out of the agent's reach. Do not hand over a confident guard that
does not guard.

"Classified as mutating" is not "can mutate". For a builtin the class is
fixed; for an external tool it is the manifest's own `mutating` flag, which is
trusted metadata the agent writes, not an effects check the harness runs. An
approved
external tool that declares `mutating = false` is a native host process that
can still write or delete any path, and `ask` mode will NOT stop it, because
the harness took the tool at its word. So the honest recipe for "watch every
write to X" pairs `ask` mode with an audit of every approved external tool
that could reach X: read what each one actually runs, and if a self-declared
read-only tool can mutate, that is the hole to close. The durable gate is a
`deny` (or `ask`) rule on the tool's NAME, placed BEFORE any matching `allow`:
rules run before the content gate, so they stop even an approved tool, and
they keep stopping it whatever its bytes become. But evaluation is FIRST-MATCH,
so an earlier `allow` for the same tool shadows the gate and the tool runs
unprompted - and project rules are read before global ones, so a global gate
cannot override a project `allow`. Put the deny/ask above every matching
`allow` (removing or reordering any that a human approved earlier), or the gate
is inert. There is no operation today that revokes an external tool's
content approval in place - `--forget` retires a path or hook but leaves the
approved manifest and code hashes standing - and neither editing nor deleting
the file revokes that approval: both stop the tool as it is now, but the
original approved bytes still match the standing approval, so restoring them
byte-for-byte would run without a card again. Only the name rule survives such
a restore. `ask` mode alone will not surface this for you.

`allow` is the only effect that removes a gate: it skips the approval prompt
outright. The project file sits where you write, so an `allow` in it is inert
until a human approves that exact content with
`openmax --approve .openmax/permissions.toml`; until then those calls fall
through to `approval_mode` and the human is asked. Editing the file revokes
the approval, as with any other capability content. `deny` and `ask` only add
friction and always apply. The global file is outside the project root, where
your file tools cannot write, so its `allow` rules need no approval. Writing
yourself an `allow` rule therefore grants nothing; ask the user to approve the
file when they want the prompts gone.

Rules never enter the prompt and are re-read every turn; an empty or missing
file changes nothing. Fail closed: an unreadable or malformed file denies
every tool. The one exemption: `write_file`/`edit_file` targeting the broken
project file itself falls through to `approval_mode`, so a typo stays
repairable from inside the session; a broken global file is fixed from the
shell, guided by `openmax --check`.

Example (`.openmax/permissions.toml`):

```toml
[[rules]]
effect = "deny"
tool = "bash"
arg_regex = "rm\\s+-rf"

[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^cargo (test|check|build)"
```

Activation is one-directional within a turn: rules are re-read at turn start
and after every mutating call, and every snapshot the turn has observed keeps
voting, most-restrictive answer wins. So a NEW restriction (a `deny`/`ask`
you write) is in force before your next step - "install the guard, then prove
it" proves it guarded. But a RELAXATION only lands at the next turn's fresh
discovery: removing a deny, or FIXING a file you broke earlier this turn, does
not take effect until the turn ends. Concretely: if a mutating call leaves
permissions.toml malformed, the file fails closed and DENIES EVERY tool call
(including bash, so `openmax --check` cannot run) for the rest of that turn;
your repair to the file is allowed (write_file/edit_file on exactly that path
stay open; read_file does NOT - a policy that denies reading this file must not
be bypassable by corrupting it) but applies only from the next turn. The
harness says so
on the writing call. Between turns, verify with `openmax --check` (it reports
the exact fail-closed reason for a malformed file).
"#;

const PROVIDERS: &str = r#"# Providers

Named OpenAI-compatible endpoints: `~/.openmax/providers.json` (global only;
it lives outside the project root, so edit it via bash). The file is optional;
the flat `base_url`/`model` settings path keeps working without it.

Shape: `{"providers": {"<name>": { ... }}}`. Per provider:
- `base_url` (required): root of the endpoint's HTTP API; the harness calls
  `chat/completions` on it.
- `api_key` (optional): literal secret. Prefer `api_key_env`.
- `api_key_env` (optional): env var name, or list of names (first non-empty
  wins).
- `headers` (optional): extra HTTP headers as a string map.
- `models` (optional): list of `{"id", "name"?, "context_tokens"?,
  "max_tokens"?}`. Ids are sent unchanged; order is preserved in `/model`.
- `compat` (optional): wire quirks for picky servers:
  `{"use_max_completion_tokens": false, "send_stream_options": true}` are the
  defaults.

Select a provider with `"provider"` in settings.json, the `--provider` CLI
option, or `/provider`; `/model` picks provider and model as one pair. This
is endpoint configuration, not a plugin protocol.

Unknown keys anywhere in the file configure nothing: the runtime ignores
them, and `openmax --check` names each one (with the near-miss key it was
probably meant to be).

Example (`~/.openmax/providers.json`):

```json
{
  "providers": {
    "ollama": {
      "base_url": "http://127.0.0.1:11434/v1",
      "models": [{ "id": "qwen2.5-coder:7b", "name": "Qwen Coder 7B" }]
    },
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "compat": { "use_max_completion_tokens": true }
    }
  }
}
```

Activation: providers.json is re-read every turn (content-keyed cache), so
a catalog edit applies at the next turn without a restart. settings.json is
NOT hot - it is read once at launch; see `openmax --spec settings` for the
boundary and the supported model-switch recipe. Verify with `openmax
--check`, which validates providers.json and settings.json.
"#;

const SETTINGS: &str = r#"# Settings (`~/.openmax/settings.json`)

One JSON object. READ AT LAUNCH, once: the running process never re-reads
this file, so an edit made mid-session changes nothing until the next
`openmax` launch. This is deliberate - `base_url` and `api_key` are
credential routing (the same bytes that name the endpoint decide where the
key is sent), and `approval_mode` adopted hot from an agent-writable path
would be self-approval. Do not edit this file to change the current
session; tell the user instead (see the model-switch recipe below).

Parsing is strict and fail-closed: a file that exists but does not parse,
uses an unknown key, or sets an unrecognized value makes the NEXT launch
refuse to start (exit 2) until fixed. A missing file means defaults. The
TUI rewrites the whole file on `/model`, `/provider`, and `/approvals`, so
manual edits can be overwritten by the next in-app settings change.

Fields (all optional in JSON; an empty `base_url`/`model` or a missing
`context_tokens` fails at resolve, i.e. every turn refuses with the reason):
- `provider`: named entry in providers.json; supplies base_url/credentials/
  headers when set. Flat fields below are the fallback.
- `base_url`: OpenAI-compatible endpoint root (the harness calls
  `chat/completions` on it). No default, never a localhost fallback.
- `api_key`: literal, or `$ENV_VAR` indirection; `OPENMAX_API_KEY` also works.
- `model`: model id sent with every request.
- `approval_mode`: `auto` | `ask` | `readonly`.
- `context_tokens`: the model's context window in tokens. REQUIRED here or
  in the model's providers.json entry (which wins); no default, nothing is
  queried from the server, and a guessed window is wrong in one direction or
  the other. `openmax --check` warns while it is missing.
- `max_tokens`, `temperature`: request shaping; a per-model `max_tokens` in
  providers.json overrides.
- `max_output_bytes`: tool-output byte cap before tail-truncation with spill.
- `compaction_tokens`: optional early-compaction trigger; only ever earlier.
- `max_agent_tokens`: per-turn spend ceiling, admission-enforced.
- `max_agent_iterations`: tool/model iteration cap per turn (default 50).
- `max_parallel_tools`: concurrent read-only tool cap, clamped 1..=32.

```json
{
  "provider": "xai",
  "base_url": "https://api.x.ai/v1",
  "api_key": "$XAI_API_KEY",
  "model": "grok-4.5",
  "approval_mode": "ask",
  "context_tokens": 131072,
  "max_tokens": 4096,
  "temperature": 0.2
}
```

## The supported model-switch recipe

providers.json IS hot (re-read each turn, content-keyed cache), and the
/model picker re-reads it every time it opens. So to bring a new model to
this project: add the model entry to the right provider's `models` array in
`~/.openmax/providers.json` (bash; see `openmax --spec providers`), verify
with `openmax --check`, then tell the user to run `/model <id>` (or pick it
in `/model`). The user's selection writes settings.json through the
harness's own hand and takes effect on the next request - no restart.

## Activation

- Read once at launch; `/reload` re-freezes tools/skills/prompt but never
  re-reads settings.
- The harness notices settings.json drift after any mutating call and at
  turn start, and says whether the new bytes parse - a malformed file is
  named while it is still repairable, because the next launch will refuse
  to start on it.
"#;

const MEMORY: &str = r#"# Project memory

One durable fact per markdown file: `.openmax/memory/<name>.md`. The agent
writes these with the ordinary file tools; the harness only scores, surfaces,
and forgets them. There is no global memory tier and no database.

Fields (the file IS the contract):
- name: the file stem, 1-64 chars of [a-z0-9-]. Anything else is ignored.
- first non-empty line: the description, shown as the memory's index line in
  the frozen prompt (leading `#` stripped, capped at 160 chars). A file with
  no describable first line is ignored.
- body: free-form markdown, any length, read on demand with read_file.

Index: at session creation (and /reload or a re-freeze) live memories are
ranked and injected as one line each - `- name: description - path` - under a
1500-byte budget, strongest first, with a trailer counting what did not fit.
No memories, no section, zero prompt cost.

Scoring is ACT-R base-level activation, computed lazily from timestamps:
each past access at age t hours contributes t^-0.5 (ages under one hour
count as one hour; events in the same hour collapse to one, so a write's
mtime and its log line do not double-count), and activation is ln of the
sum, so recency and frequency trade off in one number and one recall
revives an old memory.
Accesses are the file's mtime plus logged events in
`.openmax/memory/.access.jsonl` (the harness appends `read` when read_file
targets a memory path and `write` for write_file/edit_file, once per kind
per turn). bash access is not tracked; prefer the file tools for recall.

Forgetting is deliberate:
- Below the activation of one access 21 days old, a memory leaves the index
  (still on disk, still greppable).
- Below the activation of one access 60 days old, the file is deleted at the
  next session creation, leaving a `gc` tombstone line (name, sha256,
  description) in the access log. Update or delete stale facts yourself
  rather than letting them fade into the index of a future session.

Supersede, do not duplicate: update the existing file when a fact changes
(date-stamp facts that can go stale). Near-duplicate files split the access
history that keeps a fact alive.

Activation: a memory write moves the extension fingerprint, so the harness
re-freezes after the writing call and the fact is indexed in your prompt from
your next step (the refreeze receipt says "Memory index indexed: <name>");
the index also rebuilds at session creation, /reload, and any other re-freeze.
Verify what you wrote with `openmax --check` (it names every ignored file and
why, and what the index will show).

## Searching what was kept

`openmax --spec recall` documents the search over this project's own history.
"#;

const RECALL: &str = r#"# Recall

`openmax --recall "<query>"` (run it with bash) searches this project's own
history - session transcripts, compaction archives, compaction digests, and
memory files - and prints ranked excerpts, each citing an absolute path.
A record inside a JSONL store is cited `path:line`, so `sed -n '<line>p'`
reads it back exactly and `head -c` bounds how much; a memory file is its
own record and is cited as a bare path, with no line to give.

Read-only and project-scoped: no session, no endpoint, no derived index.
The stores on disk are scanned directly, newest sessions first, up to a
64 MB ceiling; anything skipped is counted in the report, never silent.

Query syntax: plain terms, plus
- `path:<substr>`: keep only history that touched a matching file path
  (structured compaction-record paths or literal chunk text; the store's own
  addresses never match).
- `session:<id-prefix>`: scope to matching sessions - "more from the session
  you just cited".
- `k:<n>`: ranked results to print (default 8, max 50).
- `budget:<tokens>`: output token cap (default 2000, max 20000).
- `excerpt:<chars>`: excerpt window width (default 480, 120-1200). One page
  is the largest excerpt; ask for more and the report says it was capped.
  Read a hit's cited address for the whole record.

Ranking is BM25 lexical relevance with recency as a damped tiebreaker
(0.25 x the memory index's `age_hours^-0.5` law): relevance dominates at any
age, and age only reorders near-equals. Long texts score as overlapping
~1200-char pages so a fact inside a pasted log competes as a short document
instead of losing to length normalization. Query terms drop English
stopwords (kept if the query is nothing else), match plurals and >=5-char
prefixes ("abandon" reaches "abandoned"), and scores carry an idf-weighted
coverage factor so a page matching the query's informative terms beats a
snippet matching two common ones. Excerpts center on the rarest matched
term. System prompts never match; identical excerpts deduplicate;
bare titles never displace content hits from their own session. The report
is honest by construction: matches past k:/budget: are counted, sessions
skipped past the scan cap are counted, index entries whose files are gone
are counted unreadable, and a session index that exists but cannot be
parsed is a loud error, never an empty result. `--json` emits the
structured report. Every citation is an absolute path, whatever store it
came from, so an address keeps working wherever it is resolved. Hits from
a JSONL store also carry `line` (rendered `path:line`); memory hits carry
no line, because the file is the record.

A transcript or archive hit also names its speaker: `role` is `user`,
`assistant` or `tool`, rendered after the kind as `message/user`. Memory
files and compaction digests have no speaker and carry no role. Use it to
tell a prompt that restates the question from the answer to it before
spending a read on the address - lexical ranking cannot separate those two,
because they are about the same thing in the same words.
"#;

const STDIO: &str = r#"# stdio protocol (openmax-stdio/5)

`openmax --stdio` speaks line-delimited JSON both ways: commands on stdin,
`AgentEvent` envelopes on stdout. This is the stable contract for custom
frontends, editor integrations, and one openmax driving another.

Handshake: the first stdout line is
{"type":"hello","proto":"openmax-stdio/5","protocol_version":5,"session_id":"...","version":"...","project":"/abs/path"}.
`protocol_version` is compared as an integer; any wire change bumps it.

Commands, one JSON object per line:
- {"cmd":"user","text":"..."} starts a turn.
- {"cmd":"approve","approval_id":"...","approved":true|false} answers a
  pending approval.
- {"cmd":"approval_mode","mode":"auto"|"ask"|"readonly"} sets the gate for
  mutating tools, persisted to settings like the TUI's /approvals. Answered
  by {"type":"approval_mode","mode":"..."} once the new mode is saved, or a
  `protocol_error` naming the legal values; on a save failure the mode is
  unchanged and the error says so.
- {"cmd":"reload"} re-freezes tools, skills, and the prompt from current
  config, like /reload; answered by a `refrozen` event, and refused with a
  `protocol_error` while a turn is in flight.
- {"cmd":"cancel"} cancels the running turn.
- {"cmd":"quit"} drains the in-flight turn, then exits. EOF behaves like quit.
Unknown `cmd` values yield {"type":"protocol_error","message":"..."} and the
session continues; extra fields on a known command are ignored; blank lines
are skipped.

`openmax --stdio --continue` reattaches to the directory's latest session.
After `hello` it emits one
{"type":"transcript","session_id":"...","messages":[{"role":"user"|"assistant","content":"..."},...],"truncated":bool}
line so the client can render what came before: user and assistant text only,
bounded per message and in total (`truncated` says whether anything was cut),
with the session file remaining the full record. No synthetic live events are
replayed, so a `token` stream always means a running turn.

Events: every line carries `session_id`, a `type` discriminator, then fields.
Parse by field name, never by key order. Types: `token` (text), `thinking`
(text), `message_done` (text), `budget` (used_tokens: the transcript plus
the frozen tool schemas sent on every request, context_tokens),
`usage` (prompt_tokens, completion_tokens, cached_tokens|null), `tool_start`
(call_id, name, args), `tool_end` (call_id, ok, output), `harness_note`
(call_id, text: a note the harness wrote into the MODEL's transcript - a
refreeze receipt, or a policy/providers/settings/approval notice - surfaced
here so a frontend can render what the model sees; `call_id` links it to the
tool result it rode, or is empty for a note inserted before the next prompt
like a turn-start receipt), `diff` (call_id,
path, diff, added, removed), `approval_request` (approval_id, name, summary,
detail, reason, source_path, source_sha, and an optional `env`), `approval_settled` (approval_id,
outcome), `refrozen` (tools, skills, changes: the refreeze receipt naming
each recorded capability-file change and its actor), `schemas_over_budget`
(schema_tokens, budget_tokens: the installed tools take most of what the
window can spend, so compaction runs early and stops entirely once they
reach it; advisory, at most once per session), `compacted` (tokens_before,
tokens_after, compacted_messages: the receipt of a forced compaction;
compacted_messages of 0 means the transcript was already at or under the
prune target and nothing changed),
`hook_failed` (hook, event, detail: a hook did not run - an observe-only hook
failed, or a hook file on disk is not loaded - and the turn proceeded),
`turn_refused` (hook, reason, continuation, continuations_left: a blocking
`turn_end` hook refused the model's completion and the harness honored it;
`reason` is already in the transcript as a user message - on disk before this
event goes out - and the turn continues, so render it, or the live view shows
the model finishing and then starting again with no visible cause while a
replay of the same session from disk shows the injected message; the two
counters are the numbers the hook's payload carried for this attempt), `done`
(stop_reason), `error` (message).

`approval_request.reason` is `gate` (approval_mode or a permission rule) or
`unapproved_source`: a call of an external tool whose exact bytes - the
manifest, or the project-local code it runs - no human has approved.
`unapproved_source` is the human boundary itself and must never be
auto-approved; it carries `source_path` (project-relative where possible) and
`source_sha` (first 12 hex chars), so a client that cannot prompt can print
`openmax --approve <source_path>`. Both are empty on `gate`. `env` is the list
of environment variable NAMES the approved tool will receive (its manifest's
`env` allowlist): a credential grant. It is omitted from the wire when empty,
so a `gate` and a tool that forwards nothing carry no `env` key. When present,
render it on its OWN line the card never clips - approving secrets a narrow
terminal hid behind other detail is the failure the field exists to prevent;
do not fold it into `detail`.

Every `user` command is answered by exactly one `done`, and `done` is the
only guaranteed terminator. A command that starts no turn (empty text, an
untrusted project) still gets one, with stop_reason `refused`, after the
`protocol_error` that says why. A turn that dies unexpectedly reports
`error` and then `done` with stop_reason `error`; a provider stream that ends
mid-answer with no completion signal reports its partial `message_done`, then
`error`, then `done` with stop_reason `truncated`, and no tool call it carried
is run. The single exception is a
`user` sent while a turn is in flight: that is refused with a
`protocol_error` and no `done`, because the running turn owns the next one.

A command line over 8 MiB, or one that is not valid UTF-8, is refused with a
`protocol_error` and skipped; the session keeps reading.

While a client is live, approvals are forwarded and openmax waits for an
`approve`; after quit or EOF, pending and later approvals are declined so
shutdown drains promptly.

What changed in openmax-stdio/4: `turn_refused` is new. A client written for
/3 has never seen a turn continue after `message_done` without its own `user`
command; under a blocking `turn_end` hook that is now a normal turn shape, and
this event is the only line that says why.

What changed in openmax-stdio/5: `harness_note` is new. Before it, the
receipts and notices the harness writes into the model's transcript (the
refreeze receipt, the permission/providers/settings/approval notices) were
visible only to the model; a frontend saw a bare `tool_end` and could not
render, e.g., that a written tool did not load or that an approval was
revoked. A /4 client simply never saw these lines; it is safe to ignore
`harness_note` and lose only that surfaced text.

What changed in openmax-stdio/3: `budget.used_tokens` now counts the frozen
tool schemas sent on every request, not the transcript alone. Same field,
same type, larger value (a zero-extension session reports ~1270 where /2
reported ~720), and it is now exactly the total compaction enforces against
`context_tokens`; thresholds calibrated against the /2 meaning must be
re-calibrated. `schemas_over_budget` is new and additive, and so is
`compacted`.

Validate a stream against the contract: `openmax --check --stdio` reads JSONL
on stdin, reports each line, and exits nonzero on any violation.
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-spec-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The first fenced code block of a spec: the canonical example.
    fn example(spec: &str) -> String {
        let start = spec.find("```").expect("spec has a fenced example");
        let after_fence = &spec[start + 3..];
        let body_start = after_fence.find('\n').expect("fence has a language line") + 1;
        let body = &after_fence[body_start..];
        let end = body.find("```").expect("fence closes");
        body[..end].to_string()
    }

    #[test]
    fn every_surface_renders_and_unknown_does_not() {
        for name in SURFACES {
            // `usage` is dynamic (joined with the project's usage record) and
            // rendered by the CLI, not this static table.
            if name == "usage" {
                continue;
            }
            let text = render(name).unwrap_or_else(|| panic!("no spec for {name}"));
            assert!(!text.trim().is_empty(), "{name} spec is empty");
        }
        assert!(render("nope").is_none());
        assert!(render("").is_none());
    }

    /// The required set a missing-key message prints is the set the printed
    /// contract marks required, in the same order, or one of the two is
    /// lying to whoever is fixing the file.
    #[test]
    fn required_field_lists_match_the_printed_contracts() {
        for (surface, required) in REQUIRED_FIELDS {
            let text = render(surface).unwrap_or_else(|| panic!("no spec for {surface}"));
            let marked = text
                .lines()
                .filter(|line| line.contains("(required)"))
                .filter_map(|line| line.split('`').nth(1))
                .collect::<Vec<_>>()
                .join(", ");
            assert_eq!(marked, required, "{surface}");
        }
    }

    /// The printed contract cannot drift from the parsers: each example is
    /// written to the exact path its spec names and must pass the same
    /// validation `openmax --check` runs.
    #[test]
    fn examples_round_trip_through_check() {
        let root = temp_dir("roundtrip");
        let write = |rel: &str, body: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, body).unwrap();
        };
        // The example tools reference scripts relative to the project root;
        // create them so the command-existence check sees what a real user
        // following the spec would have.
        for script in ["scripts/todo-scan.sh", "scripts/deny-rm.sh"] {
            let path = root.join(script);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
        }
        write(".openmax/tools/todo_scan.toml", &example(TOOLS));
        write(".agents/skills/release/SKILL.md", &example(SKILLS));
        write(".agents/prompts/fix-issue.md", &example(PROMPTS));
        write(".openmax/hooks/deny-rm.toml", &example(HOOKS));
        write(".openmax/permissions.toml", &example(PERMISSIONS));

        // Hooks are inert until a human approves the exact content - the file
        // and the script it runs; the test stands in for the human, against a
        // scoped data dir.
        let data = root.join("test-data");
        let hook = root.join(".openmax/hooks/deny-rm.toml");
        let mut shas = vec![crate::ledger::sha256_hex(&std::fs::read(&hook).unwrap())];
        shas.extend(
            crate::ledger::manifest_code(&hook, &root)
                .into_iter()
                .filter_map(|c| c.sha256),
        );
        crate::ledger::approve_capability(&data, &root, &hook, &shas).unwrap();
        // The permissions example grants an `allow`, which is inert in a
        // project file until a human approves it; the human stands in here
        // too, so what --check sees is the installed example, not a half of it.
        let perms = root.join(".openmax/permissions.toml");
        let perms_sha = crate::ledger::sha256_hex(&std::fs::read(&perms).unwrap());
        crate::ledger::approve_capability(&data, &root, &perms, &[perms_sha]).unwrap();
        // The tool example too: an unapproved tool loads but --check says
        // its first call stops for approval (a Warn, not Ok), so the human
        // stands in here as well; what this test asserts is that the spec's
        // examples PARSE and LOAD, not the ledger's state.
        let tool = root.join(".openmax/tools/todo_scan.toml");
        let mut tool_shas = vec![crate::ledger::sha256_hex(&std::fs::read(&tool).unwrap())];
        tool_shas.extend(
            crate::ledger::manifest_code(&tool, &root)
                .into_iter()
                .filter_map(|c| c.sha256),
        );
        crate::ledger::approve_capability(&data, &root, &tool, &tool_shas).unwrap();

        let findings: Vec<_> = crate::doctor::check_at(&root, &data)
            .into_iter()
            .filter(|f| f.path.starts_with(&root))
            .collect();
        assert_eq!(findings.len(), 5, "one finding per example: {findings:?}");
        for finding in &findings {
            assert!(
                matches!(finding.status, crate::doctor::Status::Ok(_)),
                "spec example failed its own parser: {} → {:?}",
                finding.path.display(),
                finding.status
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// The settings example must survive the strict, fail-closed parser: a
    /// spec whose own example bricks the next launch is worse than none.
    #[test]
    fn settings_example_round_trips_through_strict_load() {
        let dir = temp_dir("settings");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("settings.json"), example(SETTINGS)).unwrap();
        let settings = crate::config::load(&dir)
            .expect("the spec's own settings example must parse strictly");
        assert_eq!(settings.provider.as_deref(), Some("xai"));
        assert_eq!(settings.model, "grok-4.5");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn providers_example_round_trips_through_check_file() {
        let dir = temp_dir("providers");
        let path = dir.join("providers.json");
        std::fs::write(&path, example(PROVIDERS)).unwrap();
        match crate::providers::check_file(&path) {
            Some(Ok((count, unknown_keys))) => {
                assert_eq!(count, 2, "example defines two providers");
                assert!(unknown_keys.is_empty(), "the spec example draws warnings: {unknown_keys:?}");
            }
            other => panic!("providers example must parse: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every file surface tells the agent how to verify what it wrote, and
    /// every spec states when the file takes effect.
    #[test]
    fn specs_name_verification_and_activation() {
        for name in ["tools", "skills", "prompts", "hooks", "permissions", "providers", "memory"] {
            let text = render(name).unwrap();
            assert!(text.contains("openmax --check"), "{name} spec must point at --check");
        }
        for name in ["tools", "skills"] {
            assert!(
                render(name).unwrap().contains("/reload"),
                "{name} freezes per session, so its spec must name /reload"
            );
        }
        assert!(render("stdio").unwrap().contains("openmax --check --stdio"));
    }

    /// `ask` mode prompts only on calls the harness CLASSIFIES as mutating,
    /// and for an external tool that class is the manifest's self-declared
    /// `mutating` flag - trusted metadata, not an effects check. A tool that
    /// declares `mutating = false` runs unprompted even though it is host code
    /// that can write anything, so the permissions spec must not promise `ask`
    /// mode shows "every mutating call" without that caveat and the audit it
    /// implies (Greptile): a reader who trusts the unqualified promise builds
    /// a guard with a hole in it.
    #[test]
    fn permissions_spec_qualifies_ask_mode_with_the_classification_caveat() {
        let text = render("permissions").unwrap();
        assert!(
            text.contains("CLASSIFIED as mutating"),
            "ask-mode guidance must say it prompts on CLASSIFIED-mutating calls"
        );
        assert!(
            text.contains("trusted metadata") && text.contains("not an effects check"),
            "the spec must say the classification is trusted metadata, not an effects check"
        );
        assert!(
            text.contains("audit") && text.contains("mutating = false"),
            "the spec must advise auditing approved external tools that declare mutating = false"
        );
        // The mitigation must be real: there is no in-place approval revoke,
        // so the spec must say so and name what actually closes the hole -
        // advising "revoke the approval" would send the reader to --forget,
        // which leaves the tool's hashes standing (Greptile). Collapse the
        // doc's line wrapping so the check does not depend on where a phrase
        // breaks across lines.
        let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            flat.contains("no operation today that revokes an external tool's content approval"),
            "the spec must state that an external-tool approval cannot be revoked in place"
        );
        // The durable mitigation is a name rule; the spec must say editing or
        // deleting the file does NOT revoke the standing approval (a
        // byte-identical restore runs cardless), or a reader would trust a
        // control that a restore defeats (Greptile).
        assert!(
            flat.contains("neither editing nor deleting the file revokes that approval"),
            "the spec must say edit/delete do not revoke the standing approval"
        );
        assert!(
            flat.contains("Only the name rule survives such a restore"),
            "the spec must name the mitigation that survives a byte-identical restore"
        );
        // Protecting a path from ask-mode bypass also means dropping bash
        // allows: a bash allow matches command text, so an approved bash allow
        // whose command mutates the path force-allows it, and path-scoped
        // write/edit rules do not constrain bash (Greptile).
        assert!(
            flat.contains("a `bash` allow matches COMMAND TEXT")
                && flat.contains("does not constrain bash"),
            "the spec must warn that a bash allow can bypass ask mode for the path"
        );
        // Evaluation is first-match, so the gate is inert behind an earlier
        // allow: the spec must state the ordering requirement, or a reader adds
        // a deny below an approved allow and it never fires (Greptile).
        assert!(
            flat.contains("evaluation is FIRST-MATCH") && flat.contains("earlier `allow`"),
            "the spec must say a name gate is shadowed by an earlier allow"
        );
        assert!(
            flat.contains("Put the deny/ask above every matching `allow`"),
            "the spec must tell the reader to place the gate before any matching allow"
        );
    }

    /// A frontend author reads `--spec stdio` and is then judged by
    /// `openmax --check --stdio`, which parses events with the real
    /// `AgentEvent`. Anything the wire carries but the printed contract omits
    /// is a recipe for failing the harness's own conformance check, so every
    /// event type and every field of it must appear here.
    #[test]
    fn stdio_spec_names_every_event_and_field_on_the_wire() {
        use crate::types::AgentEvent;
        let text = render("stdio").unwrap();
        let samples = [
            AgentEvent::Token { text: String::new() },
            AgentEvent::Thinking { text: String::new() },
            AgentEvent::MessageDone { text: String::new() },
            AgentEvent::Budget { used_tokens: 0, context_tokens: 0 },
            AgentEvent::Usage { prompt_tokens: 0, completion_tokens: 0, cached_tokens: None },
            AgentEvent::ToolStart {
                call_id: String::new(),
                name: String::new(),
                args: serde_json::Value::Null,
            },
            AgentEvent::ToolEnd { call_id: String::new(), ok: true, output: String::new() },
            AgentEvent::HarnessNote { call_id: String::new(), text: String::new() },
            AgentEvent::Diff {
                call_id: String::new(),
                path: String::new(),
                diff: String::new(),
                added: 0,
                removed: 0,
            },
            AgentEvent::ApprovalRequest {
                approval_id: String::new(),
                name: String::new(),
                summary: String::new(),
                detail: String::new(),
                reason: String::new(),
                source_path: String::new(),
                source_sha: String::new(),
                env: Vec::new(),
            },
            AgentEvent::ApprovalSettled { approval_id: String::new(), outcome: String::new() },
            AgentEvent::Refrozen { tools: 0, skills: 0, changes: Vec::new() },
            AgentEvent::Compacted { tokens_before: 0, tokens_after: 0, compacted_messages: 0 },
            AgentEvent::HookFailed {
                hook: String::new(),
                event: String::new(),
                detail: String::new(),
            },
            AgentEvent::TurnRefused {
                hook: String::new(),
                reason: String::new(),
                continuation: 0,
                continuations_left: 0,
            },
            AgentEvent::Done { stop_reason: String::new() },
            AgentEvent::Error { message: String::new() },
        ];
        for event in samples {
            let value = serde_json::to_value(&event).expect("events serialize");
            let obj = value.as_object().expect("an event is an object");
            let ty = obj["type"].as_str().expect("events are tagged");
            assert!(text.contains(&format!("`{ty}`")), "stdio spec never names `{ty}`");
            for field in obj.keys().filter(|k| *k != "type") {
                assert!(text.contains(field.as_str()), "stdio spec omits `{ty}.{field}`");
            }
        }
    }


    /// Payload documentation must track the real hook payloads per event:
    /// each event's spec entry must name exactly the keys `hooks.rs`
    /// serializes for that event, checked inside that entry's own braces so a
    /// field cannot drift to the wrong event while the test stays green.
    #[test]
    fn hook_spec_names_every_payload_field_per_event() {
        let text = render("hooks").unwrap();
        // The braced field list of one event's payload entry: from the
        // `- <marker>:` bullet to its closing brace.
        let payload_of = |marker: &str| -> String {
            let bullet = format!("- {marker}:");
            let start = text
                .find(&bullet)
                .unwrap_or_else(|| panic!("hooks spec has no payload entry for {marker}"));
            let entry = &text[start..];
            let open = entry.find('{').expect("payload entry opens a brace");
            let close = entry[open..].find('}').expect("payload entry closes its brace") + open;
            entry[open..=close].to_string()
        };
        let cases = [
            ("pre_tool_use", vec!["event", "session_id", "tool", "args", "cwd", "tool_ok"]),
            (
                "post_tool_use",
                vec![
                    "event", "session_id", "tool", "args", "cwd", "tool_ok", "output",
                    "output_bytes", "output_truncated", "process_bytes", "process_truncated",
                ],
            ),
            ("user_prompt_submit", vec!["event", "session_id", "cwd", "text"]),
            ("session_start", vec!["event", "session_id", "cwd"]),
            ("compaction", vec!["event", "session_id", "cwd", "record"]),
            (
                "turn_end",
                vec![
                    "event", "session_id", "cwd", "stop_reason", "blockable", "continuation",
                    "continuations_left",
                ],
            ),
        ];
        for (marker, fields) in cases {
            let payload = payload_of(marker);
            let named: Vec<&str> = payload
                .split('"')
                .skip(1)
                .step_by(2)
                .collect();
            assert_eq!(
                named, fields,
                "payload fields documented for `{marker}` must match hooks.rs exactly"
            );
        }
    }
}
