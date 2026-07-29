//! Process lifecycle hooks: optional external commands that gate or observe
//! agent lifecycle events. `pre_tool_use` and `user_prompt_submit` can block
//! (nonzero exit); `post_tool_use`, `session_start` (a session's first turn),
//! `compaction` (context was pruned), and `turn_end` (stop reason; fires even
//! on cancel) observe only. Empty discovery costs almost nothing (one
//! directory list). Hooks never change tool schemas and never inject text
//! into the model.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::execution::{
    self, CaptureSpec, ProcessError, ProcessRequest, StdinMode, Termination,
};
use crate::state::CancelToken;
use std::sync::Arc;

const DEFAULT_TIMEOUT_SECS: u64 = 10;
const MAX_TIMEOUT_SECS: u64 = 60;
/// Per-event hook cap. Every pre_tool_use hook runs on every tool call, so an
/// unbounded set turns discovery mistakes into unbounded per-call latency.
/// Stems sort deterministically; the head runs, --check names the rest.
pub const MAX_HOOKS_PER_EVENT: usize = 32;
const MAX_REASON_CHARS: usize = 500;
/// How much tool output a `post_tool_use` hook is handed. Enough for an eval
/// or telemetry hook to work with, small enough that copying it to every
/// matching hook on every tool call stays cheap.
const MAX_OUTPUT_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
    SessionStart,
    Compaction,
    TurnEnd,
}

impl HookEvent {
    /// Every event, for diagnostics that group hooks per event.
    pub(crate) const ALL: [HookEvent; 6] = [
        HookEvent::PreToolUse,
        HookEvent::PostToolUse,
        HookEvent::UserPromptSubmit,
        HookEvent::SessionStart,
        HookEvent::Compaction,
        HookEvent::TurnEnd,
    ];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            HookEvent::PreToolUse => "pre_tool_use",
            HookEvent::PostToolUse => "post_tool_use",
            HookEvent::UserPromptSubmit => "user_prompt_submit",
            HookEvent::SessionStart => "session_start",
            HookEvent::Compaction => "compaction",
            HookEvent::TurnEnd => "turn_end",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "pre_tool_use" => Some(HookEvent::PreToolUse),
            "post_tool_use" => Some(HookEvent::PostToolUse),
            "user_prompt_submit" => Some(HookEvent::UserPromptSubmit),
            "session_start" => Some(HookEvent::SessionStart),
            "compaction" => Some(HookEvent::Compaction),
            "turn_end" => Some(HookEvent::TurnEnd),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HookSpec {
    /// sha256 of the defining TOML's bytes: the identity the content-bound
    /// approval store keys on.
    pub source_sha256: String,
    pub event: HookEvent,
    pub command: String,
    pub args: Vec<String>,
    pub timeout_secs: u64,
    /// When set, the hook only runs for this tool name.
    pub tool_filter: Option<String>,
    pub source_path: PathBuf,
}

/// One observe-only hook run that failed (spawn error, nonzero exit, or
/// timeout). Observe hooks stay fail-open - the turn proceeds - but the
/// failure is returned so the frontend can say so instead of silence.
#[derive(Clone, Debug)]
pub struct HookFailure {
    /// File stem, the hook's identity.
    pub hook: String,
    pub event: &'static str,
    pub detail: String,
}

fn failure(hook: &HookSpec, detail: String) -> HookFailure {
    HookFailure {
        hook: hook
            .source_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string(),
        event: hook.event.as_str(),
        detail,
    }
}

/// Hooks discovered for the current project. Loaded once per agent turn.
#[derive(Clone, Debug, Default)]
pub struct Hooks {
    pre: Vec<HookSpec>,
    post: Vec<HookSpec>,
    user_prompt: Vec<HookSpec>,
    session_start: Vec<HookSpec>,
    compaction: Vec<HookSpec>,
    turn_end: Vec<HookSpec>,
    /// Hook files that exist but do not parse, as "path: reason". A file that
    /// fails to parse might have been a gate, so tool execution fails closed
    /// until it is fixed or removed (same policy as permissions).
    invalid: Vec<String>,
}

/// First stem wins: project dirs are listed before global, and that
/// precedence covers parse errors too. A malformed file that is shadowed by
/// an earlier valid one was never going to run, so it must not fail closed;
/// a malformed file that holds its stem blocks instead of silently falling
/// back to the definition it shadows.
fn discover_in_dirs(dirs: &[PathBuf]) -> Hooks {
    let mut by_stem: std::collections::BTreeMap<String, HookSpec> = std::collections::BTreeMap::new();
    let mut invalid_by_stem: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if stem.is_empty()
                || by_stem.contains_key(&stem)
                || invalid_by_stem.contains_key(&stem)
            {
                continue;
            }
            match parse_hook_file(&path) {
                Ok(spec) => {
                    by_stem.insert(stem, spec);
                }
                Err(reason) => {
                    invalid_by_stem.insert(stem, format!("{}: {reason}", path.display()));
                }
            }
        }
    }
    let invalid: Vec<String> = invalid_by_stem.into_values().collect();
    let mut hooks = Hooks { invalid, ..Hooks::default() };
    for spec in by_stem.into_values() {
        match spec.event {
            HookEvent::PreToolUse => hooks.pre.push(spec),
            HookEvent::PostToolUse => hooks.post.push(spec),
            HookEvent::UserPromptSubmit => hooks.user_prompt.push(spec),
            HookEvent::SessionStart => hooks.session_start.push(spec),
            HookEvent::Compaction => hooks.compaction.push(spec),
            HookEvent::TurnEnd => hooks.turn_end.push(spec),
        }
    }
    // BTreeMap iteration is stem-sorted, so each event list is deterministic
    // and the cap keeps the sorted head. --check names what is beyond it.
    for list in [
        &mut hooks.pre,
        &mut hooks.post,
        &mut hooks.user_prompt,
        &mut hooks.session_start,
        &mut hooks.compaction,
        &mut hooks.turn_end,
    ] {
        list.truncate(MAX_HOOKS_PER_EVENT);
    }
    hooks
}

impl Hooks {
    /// Discover hooks under project `.openmax/hooks/` then global
    /// `~/.openmax/hooks/`. Project entries with the same file stem win.
    ///
    /// Every file is read exactly once and the policy is parsed from those
    /// same bytes, so what a turn enforces is always a generation that
    /// existed on disk. Nothing is cached between calls: a cache key has to
    /// be computed from a read of its own, and a file edited between that
    /// read and the parse would store a policy under the fingerprint of a
    /// different one, leaving a gate that never runs while its file sits on
    /// disk. Discovery is one directory list per turn, so there is nothing
    /// worth that risk.
    pub fn discover(project_root: &Path, data_dir: &Path) -> Self {
        let mut hooks = discover_in_dirs(&hook_dirs(project_root));
        hooks.retain_approved(data_dir, project_root);
        hooks
    }

    /// Drop hooks whose exact content no human has approved. Hooks run with
    /// host authority on every matching call with no per-invocation gate, so
    /// they are the one capability file that is inert until approved - via
    /// `openmax --approve <path>`, or automatically when a human approves the
    /// in-session write that created the file. Inert is not silent:
    /// `openmax --check` names every unapproved hook.
    fn retain_approved(&mut self, data_dir: &Path, project_root: &Path) {
        let approved = crate::ledger::approved_hashes(data_dir, project_root).unwrap_or_default();
        for list in [
            &mut self.pre,
            &mut self.post,
            &mut self.user_prompt,
            &mut self.session_start,
            &mut self.compaction,
            &mut self.turn_end,
        ] {
            list.retain(|spec| approved.contains(&spec.source_sha256));
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pre.is_empty()
            && self.post.is_empty()
            && self.user_prompt.is_empty()
            && self.session_start.is_empty()
            && self.compaction.is_empty()
            && self.turn_end.is_empty()
            && self.invalid.is_empty()
    }

    /// Non-empty when any hook file failed to parse. Tool execution blocks on
    /// this: a broken file might have been a `pre_tool_use` gate, and running
    /// without it would silently drop the policy the user wrote down.
    fn fail_closed_reason(&self) -> Option<String> {
        if self.invalid.is_empty() {
            return None;
        }
        Some(format!(
            "invalid hook file(s), failing closed until fixed or removed (see openmax --check): {}",
            self.invalid.join("; ")
        ))
    }

    pub fn pre_count(&self) -> usize {
        self.pre.len()
    }

    pub fn post_count(&self) -> usize {
        self.post.len()
    }

    /// Run all matching `pre_tool_use` hooks. First block wins.
    pub async fn pre_tool_use(
        &self,
        session_id: &str,
        tool: &str,
        args: &Value,
        cwd: &Path,
        cancel: &Arc<CancelToken>,
    ) -> PreToolResult {
        if let Some(reason) = self.fail_closed_reason() {
            return PreToolResult::Block { reason };
        }
        for hook in &self.pre {
            if !hook.matches(tool) {
                continue;
            }
            let payload = tool_payload(hook, session_id, tool, args, cwd, None);
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => return PreToolResult::Block { reason },
                HookRun::Cancelled => return PreToolResult::Cancelled,
            }
        }
        PreToolResult::Allow
    }

    /// Run all matching `post_tool_use` hooks with what the call returned.
    /// Observe only - a failure never blocks and a post hook cannot change
    /// what the model sees - but failures are returned, not swallowed.
    pub async fn post_tool_use(
        &self,
        session_id: &str,
        tool: &str,
        args: &Value,
        cwd: &Path,
        outcome: &crate::tools::ToolOutcome,
        cancel: &Arc<CancelToken>,
    ) -> Vec<HookFailure> {
        let mut failures = Vec::new();
        for hook in &self.post {
            if !hook.matches(tool) {
                continue;
            }
            let payload = tool_payload(hook, session_id, tool, args, cwd, Some(outcome));
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }

    /// Run all `user_prompt_submit` hooks against the text the user typed,
    /// before it enters the transcript. First block wins (nonzero exit); the
    /// blocked turn never starts and never reaches the model. Gate only:
    /// hooks still never inject text into the context.
    pub async fn user_prompt_submit(
        &self,
        session_id: &str,
        text: &str,
        cwd: &Path,
        cancel: &Arc<CancelToken>,
    ) -> PreToolResult {
        for hook in &self.user_prompt {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "text": text,
            });
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => return PreToolResult::Block { reason },
                HookRun::Cancelled => return PreToolResult::Cancelled,
            }
        }
        PreToolResult::Allow
    }

    /// Run `session_start` hooks (a session's first turn). Observe only:
    /// nothing enters the model context, but failures are returned.
    pub async fn session_start(
        &self,
        session_id: &str,
        cwd: &Path,
        cancel: &Arc<CancelToken>,
    ) -> Vec<HookFailure> {
        let mut failures = Vec::new();
        for hook in &self.session_start {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
            });
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }

    /// Run `compaction` hooks after context was pruned, with the same digest
    /// record that was persisted. Observe only; failures are returned.
    pub async fn compaction(
        &self,
        session_id: &str,
        cwd: &Path,
        record: &Value,
        cancel: &Arc<CancelToken>,
    ) -> Vec<HookFailure> {
        let mut failures = Vec::new();
        for hook in &self.compaction {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "record": record,
            });
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }

    /// Run `turn_end` hooks with the turn's stop reason. Observe only, and
    /// deliberately run with a fresh cancel token: a cancelled turn is still
    /// a finished turn worth observing. Failures are returned.
    pub async fn turn_end(
        &self,
        session_id: &str,
        cwd: &Path,
        stop_reason: &str,
    ) -> Vec<HookFailure> {
        let cancel = Arc::new(CancelToken::default());
        let mut failures = Vec::new();
        for hook in &self.turn_end {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "stop_reason": stop_reason,
            });
            match run_hook(hook, payload, cwd, &cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }
}

/// The stdin payload for tool-scoped events, shared by pre and post. `result`
/// is the outcome a `post_tool_use` hook is reporting on, and is None before
/// the call runs.
fn tool_payload(
    hook: &HookSpec,
    session_id: &str,
    tool: &str,
    args: &Value,
    cwd: &Path,
    result: Option<&crate::tools::ToolOutcome>,
) -> Value {
    let mut payload = serde_json::json!({
        "event": hook.event.as_str(),
        "session_id": session_id,
        "tool": tool,
        "args": args,
        "cwd": cwd.display().to_string(),
        "tool_ok": result.map(|o| o.ok),
    });
    if let Some(outcome) = result {
        let head = head_bytes(&outcome.output, MAX_OUTPUT_BYTES);
        let Some(map) = payload.as_object_mut() else { return payload };
        map.insert("output".into(), Value::String(head.to_string()));
        // Output is bounded twice, and a hook is told about both cuts rather
        // than left to infer either from the length it received.
        map.insert("output_bytes".into(), Value::from(outcome.output.len()));
        map.insert(
            "output_truncated".into(),
            Value::Bool(head.len() < outcome.output.len()),
        );
        map.insert(
            "process_bytes".into(),
            outcome.process_bytes.map(Value::from).unwrap_or(Value::Null),
        );
        map.insert("process_truncated".into(), Value::Bool(outcome.process_truncated));
    }
    payload
}

/// The longest prefix of `s` that fits in `max` bytes without splitting a
/// character. A tool result can be large (a `bash` call can print megabytes)
/// and every matching hook is handed a copy, so what a hook sees is a head,
/// the same shape hook block reasons already take.
///
/// This is the second of two bounds. The tool layer has already rendered the
/// process output down to a result, keeping its tail and saying so inline when
/// it dropped anything, and that result is what the model reasoned about. It
/// is therefore what a hook is shown, and what `output_bytes` measures.
fn head_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[derive(Debug, PartialEq, Eq)]
pub enum PreToolResult {
    Allow,
    Block { reason: String },
    /// User cancelled while a gate hook was running; not a policy rejection.
    Cancelled,
}

impl HookSpec {
    fn matches(&self, tool: &str) -> bool {
        match &self.tool_filter {
            None => true,
            Some(name) => name == tool,
        }
    }
}

enum HookRun {
    Allow,
    Block(String),
    Cancelled,
}

/// Unknown keys are rejected so a misspelled `tool` filter cannot silently
/// widen a hook to every call.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HookFile {
    event: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
    /// Optional tool name filter.
    #[serde(default)]
    tool: Option<String>,
}

fn default_timeout() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

pub(crate) fn hook_dirs(project_root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![project_root.join(".openmax").join("hooks")];
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".openmax").join("hooks"));
    }
    dirs
}

/// Errors are ignored by discovery and surfaced verbatim by `openmax --check`.
pub(crate) fn parse_hook_file(path: &Path) -> Result<HookSpec, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("unreadable: {e}"))?;
    let source_sha256 = crate::ledger::sha256_hex(text.as_bytes());
    let file: HookFile = toml::from_str(&text).map_err(|e| format!("invalid TOML: {e}"))?;
    let event = HookEvent::parse(&file.event).ok_or_else(|| {
        format!(
            "unknown event '{}': expected pre_tool_use, post_tool_use, user_prompt_submit, session_start, compaction, or turn_end",
            file.event
        )
    })?;
    let command = file.command.trim().to_string();
    if command.is_empty() {
        return Err("command is empty".into());
    }
    let tool_filter = file
        .tool
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    Ok(HookSpec {
        source_sha256,
        event,
        command,
        args: file.args,
        timeout_secs: file.timeout_secs.clamp(1, MAX_TIMEOUT_SECS),
        tool_filter,
        source_path: path.to_path_buf(),
    })
}

async fn run_hook(
    hook: &HookSpec,
    payload: Value,
    cwd: &Path,
    cancel: &Arc<CancelToken>,
) -> HookRun {
    if cancel.is_cancelled() {
        return HookRun::Cancelled;
    }
    let request = ProcessRequest {
        program: hook.command.clone().into(),
        args: hook.args.iter().cloned().map(Into::into).collect(),
        cwd: cwd.to_path_buf(),
        stdin: StdinMode::Bytes(payload.to_string().into_bytes()),
        timeout: Duration::from_secs(hook.timeout_secs),
        capture: CaptureSpec {
            // Hook block reasons keep their beginning and are character-capped
            // below. Four bytes per character covers valid UTF-8.
            head_bytes: MAX_REASON_CHARS * 4,
            tail_bytes: 0,
            spill_dir: None,
            spill_bytes_per_stream: 0,
        },
    };

    match execution::run_process(request, cancel.clone()).await {
        Err(ProcessError::Spawn(e)) => {
            // Misconfigured hook: fail closed for pre, ignore for post-style.
            // Caller maps Block for pre_tool_use only.
            HookRun::Block(format!(
                "failed to start hook '{}' ({}): {e}",
                hook.command,
                hook.source_path.display()
            ))
        }
        Err(ProcessError::Wait(e)) => {
            HookRun::Block(format!("hook '{}' failed: {e}", hook.source_path.display()))
        }
        Ok(output) => match output.termination {
            Termination::Cancelled => HookRun::Cancelled,
            Termination::TimedOut => HookRun::Block(format!(
                "hook '{}' timed out after {}s",
                hook.source_path.display(),
                hook.timeout_secs
            )),
            Termination::Exited(status) => {
                if status.success() {
                    HookRun::Allow
                } else {
                    let mut reason =
                        String::from_utf8_lossy(&output.stdout.head).trim().to_string();
                    if reason.is_empty() {
                        reason = String::from_utf8_lossy(&output.stderr.head).trim().to_string();
                    }
                    if reason.is_empty() {
                        reason = format!(
                            "blocked by hook {} (exit {})",
                            hook.source_path.display(),
                            status.code().unwrap_or(-1)
                        );
                    }
                    if reason.chars().count() > MAX_REASON_CHARS {
                        reason = reason.chars().take(MAX_REASON_CHARS).collect::<String>() + "…";
                    }
                    HookRun::Block(reason)
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    fn write_script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    /// The approval boundary itself: an unapproved hook never runs, and the
    /// same content approved (as `openmax --approve` or an in-session write
    /// approval would) is live on the next discovery.
    #[test]
    fn unapproved_hooks_are_inert_until_their_content_is_approved() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audit.toml");
        std::fs::write(&path, "event = \"post_tool_use\"\ncommand = \"/bin/sh\"\n").unwrap();

        let hooks = Hooks::discover(&tmp, &data);
        assert!(hooks.post.is_empty(), "unapproved content must not load");

        let sha = crate::ledger::sha256_hex(&std::fs::read(&path).unwrap());
        crate::ledger::approve_hash(&data, &tmp, &sha).unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.post.len(), 1, "approved content loads");

        // Any edit revokes: the new bytes have a new, unapproved hash.
        std::fs::write(&path, "event = \"post_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        assert!(hooks.post.is_empty(), "an edited hook must fall back to inert");
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// Discover with every hook file under the project pre-approved, plus a
    /// throwaway data dir: these tests exercise parse and gate semantics, not
    /// the approval boundary (which has its own tests).
    fn discover_for_test(project: &Path) -> Hooks {
        let data = project.join("test-approvals-data");
        for dir in hook_dirs(project) {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for entry in rd.flatten() {
                if let Ok(bytes) = std::fs::read(entry.path()) {
                    let _ = crate::ledger::approve_hash(
                        &data,
                        project,
                        &crate::ledger::sha256_hex(&bytes),
                    );
                }
            }
        }
        Hooks::discover(project, &data)
    }

    fn write_hook_toml(dir: &Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    #[test]
    fn hooks_beyond_the_per_event_cap_never_run() {
        let dir = tempfile_dir();
        for i in 0..(MAX_HOOKS_PER_EVENT + 3) {
            write_hook_toml(
                &dir,
                &format!("hook-{i:03}.toml"),
                "event = \"post_tool_use\"\ncommand = \"/bin/true\"\n",
            );
        }
        let hooks = discover_in_dirs(&[dir.clone()]);
        assert_eq!(hooks.post.len(), MAX_HOOKS_PER_EVENT);
        // Other events are unaffected by post_tool_use volume.
        assert!(hooks.pre.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_empty_when_no_hooks_dir() {
        let tmp = tempfile_dir();
        let hooks = discover_for_test(&tmp);
        assert!(hooks.is_empty());
    }

    #[test]
    fn discover_detects_same_length_same_mtime_edit() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let body_a = "event = \"pre_tool_use\"\ncommand = \"/bin/aaaa\"\n";
        let body_b = "event = \"pre_tool_use\"\ncommand = \"/bin/bbbb\"\n";
        assert_eq!(body_a.len(), body_b.len());
        write_hook_toml(&hooks_dir, "gate.toml", body_a);
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre.len(), 1);
        assert_eq!(hooks.pre[0].command, "/bin/aaaa");

        // Same byte length, pinned mtime: a metadata fingerprint would keep
        // the obsolete policy live.
        let path = hooks_dir.join("gate.toml");
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        write_hook_toml(&hooks_dir, "gate.toml", body_b);
        let f = std::fs::File::options().write(true).open(&path).unwrap();
        f.set_modified(mtime).unwrap();
        drop(f);

        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre.len(), 1);
        assert_eq!(hooks.pre[0].command, "/bin/bbbb");
    }

    #[test]
    fn discovery_holds_no_memory_between_calls() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let gate = "event = \"pre_tool_use\"\ncommand = \"/bin/aaaa\"\n";
        let path = hooks_dir.join("gate.toml");

        write_hook_toml(&hooks_dir, "gate.toml", gate);
        assert_eq!(discover_for_test(&tmp).pre.len(), 1);

        std::fs::remove_file(&path).unwrap();
        assert!(discover_for_test(&tmp).is_empty(), "a removed gate must stop applying");

        // Restoring byte-identical content puts the policy back. Anything
        // that remembered the gap keyed by content would answer "no gate"
        // here, and a gate that does not run is a gate that is not enforced.
        write_hook_toml(&hooks_dir, "gate.toml", gate);
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre.len(), 1, "invalid: {:?}", hooks.invalid);
        assert_eq!(hooks.pre[0].command, "/bin/aaaa");
    }

    #[tokio::test]
    async fn invalid_hook_file_fails_closed() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_hook_toml(
            &hooks_dir,
            "bad.toml",
            r#"
event = "not_a_real_event"
command = "true"
"#,
        );
        let hooks = discover_for_test(&tmp);
        assert!(!hooks.is_empty());
        assert_eq!(hooks.pre_count(), 0);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        match result {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("bad.toml"), "{reason}");
                assert!(reason.contains("failing closed"), "{reason}");
            }
            PreToolResult::Allow | PreToolResult::Cancelled => panic!("expected block"),
        }
    }

    #[tokio::test]
    async fn shadowed_invalid_hook_does_not_fail_closed() {
        let tmp = tempfile_dir();
        let project = tmp.join("project");
        let global = tmp.join("global");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::create_dir_all(&global).unwrap();
        write_hook_toml(&project, "audit.toml", "event = \"post_tool_use\"\ncommand = \"true\"\n");
        write_hook_toml(&global, "audit.toml", "event = \"not_a_real_event\"\ncommand = \"true\"\n");
        // The malformed global file is shadowed by the valid project one, so
        // it was never going to run and must not block.
        let hooks = discover_in_dirs(&[project.clone(), global.clone()]);
        assert!(hooks.invalid.is_empty(), "{:?}", hooks.invalid);
        assert_eq!(hooks.post_count(), 1);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        assert_eq!(result, PreToolResult::Allow);

        // The reverse still fails closed: a malformed file in the
        // higher-precedence dir holds its stem instead of silently falling
        // back to the valid definition it shadows.
        let hooks = discover_in_dirs(&[global, project]);
        assert_eq!(hooks.invalid.len(), 1, "{:?}", hooks.invalid);
        assert_eq!(hooks.post_count(), 0);
        let result = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        assert!(matches!(result, PreToolResult::Block { .. }));
    }

    #[tokio::test]
    async fn unknown_hook_key_is_rejected_and_fails_closed() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // `tools` (plural) is the likely typo for the `tool` filter; accepting
        // it would silently widen the hook to every call.
        write_hook_toml(
            &hooks_dir,
            "typo.toml",
            r#"
event = "pre_tool_use"
command = "true"
tools = "bash"
"#,
        );
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre_count(), 0);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        assert!(matches!(result, PreToolResult::Block { .. }));
    }

    #[tokio::test]
    async fn pre_hook_can_block_with_stdout_reason() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "block.sh",
            "#!/bin/sh\necho 'blocked by test hook'\nexit 1\n",
        );
        write_hook_toml(
            &hooks_dir,
            "block.toml",
            &format!(
                r#"
event = "pre_tool_use"
command = "{}"
tool = "bash"
"#,
                script.display()
            ),
        );
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre_count(), 1);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use(
                "sess",
                "bash",
                &serde_json::json!({"command": "rm -rf /"}),
                &tmp,
                &cancel,
            )
            .await;
        match result {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("blocked by test hook"), "{reason}");
            }
            PreToolResult::Allow | PreToolResult::Cancelled => panic!("expected block"),
        }
        // Filtered tool should not run the hook path for other tools.
        let allow = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        assert_eq!(allow, PreToolResult::Allow);
    }

    #[tokio::test]
    async fn session_start_and_compaction_hooks_observe_via_stdin() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // Each hook copies its stdin payload to a marker file; observe-only
        // means a nonzero exit must not disturb the caller either way.
        let start_script = write_script(
            &tmp,
            "start.sh",
            &format!("#!/bin/sh\ncat > {}/start.json\nexit 1\n", tmp.display()),
        );
        let compact_script = write_script(
            &tmp,
            "compact.sh",
            &format!("#!/bin/sh\ncat > {}/compact.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "start.toml",
            &format!("event = \"session_start\"\ncommand = \"{}\"\n", start_script.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "compact.toml",
            &format!("event = \"compaction\"\ncommand = \"{}\"\n", compact_script.display()),
        );
        let hooks = discover_for_test(&tmp);
        assert!(!hooks.is_empty());
        let cancel = Arc::new(CancelToken::default());

        hooks.session_start("sess", &tmp, &cancel).await;
        let start: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("start.json")).unwrap()).unwrap();
        assert_eq!(start["event"], "session_start");
        assert_eq!(start["session_id"], "sess");

        let record = serde_json::json!({"message_count": 7, "digest": "d"});
        hooks.compaction("sess", &tmp, &record, &cancel).await;
        let compact: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("compact.json")).unwrap())
                .unwrap();
        assert_eq!(compact["event"], "compaction");
        assert_eq!(compact["record"]["message_count"], 7);
    }

    #[tokio::test]
    async fn post_tool_use_hook_receives_the_tool_output() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "audit.sh",
            &format!("#!/bin/sh\ncat > {}/audit.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "audit.toml",
            &format!("event = \"post_tool_use\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let args = serde_json::json!({"command": "echo hi"});

        let outcome = crate::tools::ToolOutcome::ok("hi\n".into());
        hooks.post_tool_use("sess", "bash", &args, &tmp, &outcome, &cancel).await;

        let payload: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("audit.json")).unwrap()).unwrap();
        assert_eq!(payload["event"], "post_tool_use");
        assert_eq!(payload["tool"], "bash");
        assert_eq!(payload["tool_ok"], true);
        assert_eq!(payload["output"], "hi\n");
        assert_eq!(payload["output_bytes"], 3);
        assert_eq!(payload["output_truncated"], false);
        assert!(payload["process_bytes"].is_null(), "no process ran behind this result");
        assert_eq!(payload["process_truncated"], false);
        assert_eq!(payload["args"]["command"], "echo hi");
    }

    /// A pre hook is asked whether a call may run, and there is no output yet
    /// to report. It must not look like an empty one.
    #[tokio::test]
    async fn pre_tool_use_payload_carries_no_output() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "gate.sh",
            &format!("#!/bin/sh\ncat > {}/gate.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "gate.toml",
            &format!("event = \"pre_tool_use\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());

        let result = hooks
            .pre_tool_use("sess", "bash", &serde_json::json!({}), &tmp, &cancel)
            .await;
        assert_eq!(result, PreToolResult::Allow);

        let payload: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("gate.json")).unwrap()).unwrap();
        assert!(payload.get("output").is_none(), "{payload}");
        assert!(payload.get("output_truncated").is_none());
        assert!(payload["tool_ok"].is_null());
    }

    #[test]
    fn a_long_output_is_capped_without_splitting_a_character() {
        // A multibyte character straddling the cap must not be cut in half:
        // the payload has to stay valid JSON.
        let long = "é".repeat(MAX_OUTPUT_BYTES);
        let hook = HookSpec {
            source_sha256: String::new(),
            event: HookEvent::PostToolUse,
            command: "/bin/true".into(),
            args: Vec::new(),
            timeout_secs: 1,
            tool_filter: None,
            source_path: PathBuf::from("/hooks/audit.toml"),
        };
        let payload = tool_payload(
            &hook,
            "sess",
            "bash",
            &serde_json::json!({}),
            Path::new("/project"),
            Some(&crate::tools::ToolOutcome::ok(long.clone())),
        );

        let head = payload["output"].as_str().unwrap();
        assert!(head.len() <= MAX_OUTPUT_BYTES);
        assert!(head.len() > MAX_OUTPUT_BYTES - 4, "the cap must not round down far");
        assert!(long.starts_with(head), "what a hook sees must be a real prefix");
        assert_eq!(payload["output_bytes"].as_u64().unwrap() as usize, long.len());
        assert_eq!(payload["output_truncated"], true);
        // Round-trips, which is the point of respecting the boundary.
        let encoded = serde_json::to_string(&payload).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&encoded).unwrap()["output"], head);
    }

    /// Output is bounded twice: the tool renders a process down to a result,
    /// then this payload takes a head of that result. A hook is told about
    /// both, so an audit hook never has to parse a notice out of the text to
    /// learn that a larger output existed.
    #[test]
    fn the_payload_reports_both_bounds() {
        let rendered = "[start of output truncated; bounded output log saved to /logs/x; \
                        tail or grep it with bash]\n…LINE-B\n";
        let hook = HookSpec {
            source_sha256: String::new(),
            event: HookEvent::PostToolUse,
            command: "/bin/true".into(),
            args: Vec::new(),
            timeout_secs: 1,
            tool_filter: None,
            source_path: PathBuf::from("/hooks/audit.toml"),
        };
        let payload = tool_payload(
            &hook,
            "sess",
            "bash",
            &serde_json::json!({}),
            Path::new("/project"),
            Some(&crate::tools::ToolOutcome {
                ok: true,
                output: rendered.to_string(),
                process_bytes: Some(400_001),
                process_truncated: true,
                ..Default::default()
            }),
        );

        assert_eq!(payload["output"], rendered);
        assert_eq!(
            payload["output_bytes"].as_u64().unwrap() as usize,
            rendered.len(),
            "output_bytes measures the result the model reasoned about"
        );
        assert_eq!(
            payload["output_truncated"], false,
            "this payload dropped nothing, whatever the tool layer did earlier"
        );
        assert_eq!(
            payload["process_bytes"], 400_001,
            "the size the command actually produced survives the rendering"
        );
        assert_eq!(payload["process_truncated"], true);
    }

    #[test]
    fn head_bytes_returns_everything_that_fits() {
        assert_eq!(head_bytes("short", 16), "short");
        assert_eq!(head_bytes("exact", 5), "exact");
        assert_eq!(head_bytes("abcdef", 3), "abc");
        assert_eq!(head_bytes("éé", 3), "é", "a split character is dropped whole");
        assert_eq!(head_bytes("", 0), "");
    }

    #[tokio::test]
    async fn user_prompt_submit_hook_blocks_with_reason() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // Block any prompt whose stdin payload mentions a secret marker.
        let script = write_script(
            &tmp,
            "gate.sh",
            "#!/bin/sh\nif grep -q SECRET; then echo 'input contains a secret'; exit 1; fi\nexit 0\n",
        );
        write_hook_toml(
            &hooks_dir,
            "gate.toml",
            &format!("event = \"user_prompt_submit\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let blocked = hooks
            .user_prompt_submit("sess", "here is a SECRET token", &tmp, &cancel)
            .await;
        match blocked {
            PreToolResult::Block { reason } => assert!(reason.contains("secret"), "{reason}"),
            PreToolResult::Allow | PreToolResult::Cancelled => panic!("expected block"),
        }
        let allowed = hooks.user_prompt_submit("sess", "plain request", &tmp, &cancel).await;
        assert_eq!(allowed, PreToolResult::Allow);
    }

    #[tokio::test]
    async fn user_prompt_submit_cancel_is_not_a_block() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // Hang until killed so cancel is the only way out.
        let script = write_script(&tmp, "slow.sh", "#!/bin/sh\nsleep 30\n");
        write_hook_toml(
            &hooks_dir,
            "slow.toml",
            &format!("event = \"user_prompt_submit\"\ncommand = \"{}\"\ntimeout_secs = 30\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let cancel_flag = cancel.clone();
        let task = tokio::spawn(async move {
            hooks.user_prompt_submit("sess", "prompt", &tmp, &cancel_flag).await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("join timed out")
            .expect("task panicked");
        assert_eq!(result, PreToolResult::Cancelled);
    }

    #[tokio::test]
    async fn turn_end_hook_runs_even_after_cancel() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "end.sh",
            &format!("#!/bin/sh\ncat > {}/end.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "end.toml",
            &format!("event = \"turn_end\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        // turn_end uses its own fresh token, so a cancelled turn still fires.
        hooks.turn_end("sess", &tmp, "cancelled").await;
        let end: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("end.json")).unwrap()).unwrap();
        assert_eq!(end["event"], "turn_end");
        assert_eq!(end["stop_reason"], "cancelled");
    }

    #[tokio::test]
    async fn pre_hook_allow_on_zero_exit() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(&tmp, "ok.sh", "#!/bin/sh\nexit 0\n");
        write_hook_toml(
            &hooks_dir,
            "ok.toml",
            &format!(
                r#"
event = "pre_tool_use"
command = "{}"
"#,
                script.display()
            ),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        assert_eq!(result, PreToolResult::Allow);
    }

    #[tokio::test]
    async fn failing_observe_hook_is_reported_not_silent() {
        let dir = tempfile_dir();
        write_hook_toml(
            &dir,
            "audit.toml",
            "event = \"post_tool_use\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"exit 7\"]\n",
        );
        let hooks = discover_in_dirs(&[dir.clone()]);
        let outcome = crate::tools::ToolOutcome::ok("fine".into());
        let failures = hooks
            .post_tool_use(
                "s1",
                "bash",
                &serde_json::json!({}),
                &dir,
                &outcome,
                &Arc::new(CancelToken::default()),
            )
            .await;
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].hook, "audit");
        assert_eq!(failures[0].event, "post_tool_use");
        assert!(failures[0].detail.contains("7"), "{}", failures[0].detail);
        let _ = std::fs::remove_dir_all(dir);
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-hooks-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
