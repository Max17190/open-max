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
pub const SURFACES: [&str; 8] = [
    "tools",
    "skills",
    "prompts",
    "hooks",
    "permissions",
    "providers",
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
        "stdio" => Some(STDIO),
        _ => None,
    }
}

const TOOLS: &str = r#"# External tools

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

Runtime contract: the harness spawns `command args...` in the project root,
writes the call's JSON arguments to stdin, and returns stdout as the result.
Nonzero exit makes the result an error carrying `exit code N` plus output.
Output is capped; overflow spills to `~/.openmax/cmd-logs`. The process is a
native host process: it inherits the environment, credentials, and network
access of Open Max.

Human approval: because of that authority, the first call of any tool file -
mutating or not - stops for a human, who approves the exact bytes. Later calls
of identical bytes run unprompted; any edit revokes and asks again. Approve
outside a session with `openmax --approve .openmax/tools/<name>.toml`, or by
approving the write that created it. `openmax --spec usage` lists the approval
state of every installed tool.

What "the exact bytes" covers is the whole definition: the `.toml` *and* the
project-local file its `command` (or a path in `args`) names, because that file
is the code that actually runs and the agent can rewrite it after the fact.
Editing the manifest or that script makes the next call ask again. A `command`
outside the project root (an absolute path, a name on PATH) is covered by the
manifest approval alone - that path is what the human read - while a command
resolving to no file at all is covered by nothing, so the tool asks until it
exists. `openmax --approve <tool.toml>` approves the pair up front and prints
every path and hash it blessed.

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

An example is the tool's real command with the harness's full host authority,
run in the project root with no sandbox and no snapshot: it can delete files
and write outside the project. So it passes the same gates a turn applies, and
runs only when all of them admit it:

- the project is trusted (`openmax --trust-project`);
- a human approved the tool file's exact bytes (`openmax --approve <path>`);
  editing the file revokes that approval;
- `pre_tool_use` hooks allow the call, and permission rules admit it: `deny`
  refuses, and so does `ask`, because nothing here can prompt - write
  `effect = "allow"` for a tool whose example should run unattended;
- `approval_mode` is not `readonly` for a `mutating` tool. A `mutating`
  example under `approval_mode = "ask"` needs a human to start the run, so it
  is refused when the agent loop spawned the process.

Each refusal names what to fix. An example must not itself run
`openmax --check --run-examples`: that is refused rather than recursed into.
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
body is invalid.

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

Gate events (`pre_tool_use`, `user_prompt_submit`): a nonzero exit blocks the
call or the prompt. The block reason is the hook's stdout (or stderr if stdout
is empty), capped at 500 chars. A blocked tool call returns to the model as a
failed tool result carrying the reason; a blocked prompt never reaches the
model. A gate that times out or fails to start blocks.

Observe events (`post_tool_use`, `session_start`, `compaction`, `turn_end`):
exit status is ignored. `session_start` fires on a session's first turn;
`compaction` fires after context was pruned; `turn_end` fires with the stop
reason, even on cancel. Hooks never inject text into the model context.

Each run receives one JSON payload on stdin:
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
- turn_end: {"event", "session_id", "cwd", "stop_reason"}

Approval: a hook is inert until a human approves its exact content - the
`.toml` *and* the project-local file its `command` (or a path in `args`) names,
because that file is the code that actually runs and the agent can rewrite it.
`openmax --approve <hook.toml>` approves the pair and prints both; approving
the in-session write of either file approves those bytes. A `command` outside
the project root (an absolute path, a name on PATH) is covered by the manifest
approval alone: that path is what the human read, and system binaries change
on their own schedule. The bytes are re-checked before every run, so a script
rewritten mid-turn does not run.

Fail closed, four ways, all reported by `openmax --check`:
- A hook file that exists but does not parse blocks every tool until it is
  fixed or removed (a broken file might have been a gate), unless a valid
  project file shadows its stem - or unless no human ever approved that path,
  in which case it never ran and stays inert instead.
- A gate hook (`pre_tool_use`, `user_prompt_submit`) whose path was approved
  and whose content no longer is blocks every tool until the approved content
  is restored or a human re-approves it. Editing a live gate cannot turn it
  off, and that includes a comment-only edit or a rewritten script. Whether a
  modified hook counts as a gate is decided by the `event` a human approved,
  never by the `event` the current file declares: rewriting an approved
  `pre_tool_use` gate into an observe hook would otherwise stop it gating, so
  it reads as a demoted gate and still fails closed. The repair carve-out is
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
  the prompt, and can more cheaply delete the approval store itself. Without
  an OS sandbox nothing here survives an agent that is determined and has a
  shell; what the harness guarantees is that the easy path is closed and the
  state is visible.
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

Activation: next turn. Verify with `openmax --check` (it reports the exact
fail-closed reason for a malformed file).
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

Activation: resolved at the next turn (settings edits apply without a
restart). Verify with `openmax --check`.
"#;

const STDIO: &str = r#"# stdio protocol (openmax-stdio/3)

`openmax --stdio` speaks line-delimited JSON both ways: commands on stdin,
`AgentEvent` envelopes on stdout. This is the stable contract for custom
frontends, editor integrations, and one openmax driving another.

Handshake: the first stdout line is
{"type":"hello","proto":"openmax-stdio/3","protocol_version":3,"session_id":"...","version":"...","project":"/abs/path"}.
`protocol_version` is compared as an integer; any wire change bumps it.

Commands, one JSON object per line:
- {"cmd":"user","text":"..."} starts a turn.
- {"cmd":"approve","approval_id":"...","approved":true|false} answers a
  pending approval.
- {"cmd":"cancel"} cancels the running turn.
- {"cmd":"quit"} drains the in-flight turn, then exits. EOF behaves like quit.
Unknown `cmd` values yield {"type":"protocol_error","message":"..."} and the
session continues; extra fields on a known command are ignored; blank lines
are skipped.

Events: every line carries `session_id`, a `type` discriminator, then fields.
Parse by field name, never by key order. Types: `token` (text), `thinking`
(text), `message_done` (text), `budget` (used_tokens: the transcript plus
the frozen tool schemas sent on every request, context_tokens),
`usage` (prompt_tokens, completion_tokens, cached_tokens|null), `tool_start`
(call_id, name, args), `tool_end` (call_id, ok, output), `diff` (call_id,
path, diff, added, removed), `approval_request` (approval_id, name, summary,
detail, reason, source_path, source_sha), `approval_settled` (approval_id,
outcome), `refrozen` (tools, skills, changes: the refreeze receipt naming
each recorded capability-file change and its actor), `schemas_over_budget`
(schema_tokens, budget_tokens: the installed tools take most of what the
window can spend, so compaction runs early and stops entirely once they
reach it; advisory, at most once per session),
`hook_failed` (hook, event, detail: a hook did not run - an observe-only hook
failed, or a hook file on disk is not loaded - and the turn proceeded), `done`
(stop_reason), `error` (message).

`approval_request.reason` is `gate` (approval_mode or a permission rule) or
`unapproved_source`: a call of an external tool whose exact bytes - the
manifest, or the project-local code it runs - no human has approved.
`unapproved_source` is the human boundary itself and must never be
auto-approved; it carries `source_path` (project-relative where possible) and
`source_sha` (first 12 hex chars), so a client that cannot prompt can print
`openmax --approve <source_path>`. Both are empty on `gate`.

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

What changed in openmax-stdio/3: `budget.used_tokens` now counts the frozen
tool schemas sent on every request, not the transcript alone. Same field,
same type, larger value (a zero-extension session reports ~1270 where /2
reported ~720), and it is now exactly the total compaction enforces against
`context_tokens`; thresholds calibrated against the /2 meaning must be
re-calibrated. `schemas_over_budget` is new and additive.

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

    #[test]
    fn providers_example_round_trips_through_check_file() {
        let dir = temp_dir("providers");
        let path = dir.join("providers.json");
        std::fs::write(&path, example(PROVIDERS)).unwrap();
        match crate::providers::check_file(&path) {
            Some(Ok(count)) => assert_eq!(count, 2, "example defines two providers"),
            other => panic!("providers example must parse: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every file surface tells the agent how to verify what it wrote, and
    /// every spec states when the file takes effect.
    #[test]
    fn specs_name_verification_and_activation() {
        for name in ["tools", "skills", "prompts", "hooks", "permissions", "providers"] {
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
            },
            AgentEvent::ApprovalSettled { approval_id: String::new(), outcome: String::new() },
            AgentEvent::Refrozen { tools: 0, skills: 0, changes: Vec::new() },
            AgentEvent::HookFailed {
                hook: String::new(),
                event: String::new(),
                detail: String::new(),
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
            ("turn_end", vec!["event", "session_id", "cwd", "stop_reason"]),
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
