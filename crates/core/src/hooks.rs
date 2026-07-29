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
const MAX_REASON_CHARS: usize = 500;

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
    pub event: HookEvent,
    pub command: String,
    pub args: Vec<String>,
    pub timeout_secs: u64,
    /// When set, the hook only runs for this tool name.
    pub tool_filter: Option<String>,
    pub source_path: PathBuf,
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
    pub fn discover(project_root: &Path) -> Self {
        discover_in_dirs(&hook_dirs(project_root))
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

    /// Run all matching `post_tool_use` hooks. Failures are ignored (observe only).
    pub async fn post_tool_use(
        &self,
        session_id: &str,
        tool: &str,
        args: &Value,
        cwd: &Path,
        tool_ok: bool,
        cancel: &Arc<CancelToken>,
    ) {
        for hook in &self.post {
            if !hook.matches(tool) {
                continue;
            }
            let payload = tool_payload(hook, session_id, tool, args, cwd, Some(tool_ok));
            let _ = run_hook(hook, payload, cwd, cancel).await;
        }
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
    /// failures are ignored and nothing enters the model context.
    pub async fn session_start(&self, session_id: &str, cwd: &Path, cancel: &Arc<CancelToken>) {
        for hook in &self.session_start {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
            });
            let _ = run_hook(hook, payload, cwd, cancel).await;
        }
    }

    /// Run `compaction` hooks after context was pruned, with the same digest
    /// record that was persisted. Observe only.
    pub async fn compaction(
        &self,
        session_id: &str,
        cwd: &Path,
        record: &Value,
        cancel: &Arc<CancelToken>,
    ) {
        for hook in &self.compaction {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "record": record,
            });
            let _ = run_hook(hook, payload, cwd, cancel).await;
        }
    }

    /// Run `turn_end` hooks with the turn's stop reason. Observe only, and
    /// deliberately run with a fresh cancel token: a cancelled turn is still
    /// a finished turn worth observing.
    pub async fn turn_end(&self, session_id: &str, cwd: &Path, stop_reason: &str) {
        let cancel = Arc::new(CancelToken::default());
        for hook in &self.turn_end {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "stop_reason": stop_reason,
            });
            let _ = run_hook(hook, payload, cwd, &cancel).await;
        }
    }
}

/// The stdin payload for tool-scoped events, shared by pre and post.
fn tool_payload(
    hook: &HookSpec,
    session_id: &str,
    tool: &str,
    args: &Value,
    cwd: &Path,
    tool_ok: Option<bool>,
) -> Value {
    serde_json::json!({
        "event": hook.event.as_str(),
        "session_id": session_id,
        "tool": tool,
        "args": args,
        "cwd": cwd.display().to_string(),
        "tool_ok": tool_ok,
    })
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

    fn write_hook_toml(dir: &Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    #[test]
    fn discover_empty_when_no_hooks_dir() {
        let tmp = tempfile_dir();
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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

        let hooks = Hooks::discover(&tmp);
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
        assert_eq!(Hooks::discover(&tmp).pre.len(), 1);

        std::fs::remove_file(&path).unwrap();
        assert!(Hooks::discover(&tmp).is_empty(), "a removed gate must stop applying");

        // Restoring byte-identical content puts the policy back. Anything
        // that remembered the gap keyed by content would answer "no gate"
        // here, and a gate that does not run is a gate that is not enforced.
        write_hook_toml(&hooks_dir, "gate.toml", gate);
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
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
        let hooks = Hooks::discover(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        assert_eq!(result, PreToolResult::Allow);
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-hooks-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
