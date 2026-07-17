//! Process lifecycle hooks: optional external commands that gate or observe
//! agent lifecycle events. `pre_tool_use` and `user_prompt_submit` can block
//! (nonzero exit); `post_tool_use`, `session_start` (a session's first turn),
//! `compaction` (context was pruned), and `turn_end` (stop reason; fires even
//! on cancel) observe only. Empty discovery costs almost nothing (one
//! directory list). Hooks never change tool schemas and never inject text
//! into the model.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

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
}

use std::sync::{Mutex, OnceLock};

struct HooksCache {
    project_root: PathBuf,
    /// Fingerprint of hook dirs/files; None means uncached.
    fingerprint: Option<u64>,
    hooks: Hooks,
}

static HOOKS_CACHE: OnceLock<Mutex<HooksCache>> = OnceLock::new();

/// Drop cached hooks (tests or after external config install).
pub fn invalidate_hooks_cache() {
    if let Some(lock) = HOOKS_CACHE.get() {
        if let Ok(mut cache) = lock.lock() {
            cache.project_root.clear();
            cache.fingerprint = None;
            cache.hooks = Hooks::default();
        }
    }
}

/// Hash the contents of every hook definition (`.toml`) file, in sorted path
/// order. Hook files gate tool execution, so the cache key must reflect what
/// the files say, not filesystem metadata: a same-length rewrite inside one
/// timestamp tick would leave an mtime+len key unchanged and keep an obsolete
/// policy live. Hook dirs hold a handful of small files; reading them per
/// discovery is cheap.
fn hooks_fingerprint(project_root: &Path) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for dir in hook_dirs(project_root) {
        dir.hash(&mut h);
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
            .collect();
        files.sort();
        for path in files {
            path.hash(&mut h);
            std::fs::read(&path).ok().hash(&mut h);
        }
    }
    h.finish()
}

fn discover_uncached(project_root: &Path) -> Hooks {
    let mut by_stem: std::collections::BTreeMap<String, HookSpec> = std::collections::BTreeMap::new();
    for dir in hook_dirs(project_root) {
        if !dir.is_dir() {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
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
            if stem.is_empty() {
                continue;
            }
            if let Ok(spec) = parse_hook_file(&path) {
                // First wins: project dirs are listed before global.
                by_stem.entry(stem).or_insert(spec);
            }
        }
    }
    let mut hooks = Hooks::default();
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
    /// Results are cached by project root + directory fingerprint.
    pub fn discover(project_root: &Path) -> Self {
        let fp = hooks_fingerprint(project_root);
        let lock = HOOKS_CACHE.get_or_init(|| {
            Mutex::new(HooksCache {
                project_root: PathBuf::new(),
                fingerprint: None,
                hooks: Hooks::default(),
            })
        });
        let mut cache = lock.lock().unwrap_or_else(|e| e.into_inner());
        if cache.project_root == project_root && cache.fingerprint == Some(fp) {
            return cache.hooks.clone();
        }
        let hooks = discover_uncached(project_root);
        cache.project_root = project_root.to_path_buf();
        cache.fingerprint = Some(fp);
        cache.hooks = hooks.clone();
        hooks
    }

    pub fn is_empty(&self) -> bool {
        self.pre.is_empty()
            && self.post.is_empty()
            && self.user_prompt.is_empty()
            && self.session_start.is_empty()
            && self.compaction.is_empty()
            && self.turn_end.is_empty()
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
        for hook in &self.pre {
            if !hook.matches(tool) {
                continue;
            }
            let payload = tool_payload(hook, session_id, tool, args, cwd, None);
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => return PreToolResult::Block { reason },
                HookRun::Cancelled => {
                    return PreToolResult::Block {
                        reason: "hook cancelled by user".into(),
                    };
                }
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
                HookRun::Cancelled => {
                    return PreToolResult::Block { reason: "hook cancelled by user".into() }
                }
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

#[derive(Deserialize)]
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
    let stdin_json = payload.to_string();

    let mut cmd = Command::new(&hook.command);
    cmd.args(&hook.args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            // Misconfigured hook: fail closed for pre, ignore for post-style.
            // Caller maps Block for pre_tool_use only.
            return HookRun::Block(format!(
                "failed to start hook '{}' ({}): {e}",
                hook.command,
                hook.source_path.display()
            ));
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(stdin_json.as_bytes()).await;
    }

    let mut child_slot = Some(child);
    tokio::select! {
        _ = cancel.cancelled() => {
            if let Some(mut c) = child_slot.take() {
                let _ = c.kill().await;
            }
            HookRun::Cancelled
        }
        _ = tokio::time::sleep(Duration::from_secs(hook.timeout_secs)) => {
            if let Some(mut c) = child_slot.take() {
                let _ = c.kill().await;
            }
            HookRun::Block(format!(
                "hook '{}' timed out after {}s",
                hook.source_path.display(),
                hook.timeout_secs
            ))
        }
        output = async {
            child_slot.take().expect("child taken twice").wait_with_output().await
        } => {
            match output {
                Err(e) => HookRun::Block(format!(
                    "hook '{}' failed: {e}",
                    hook.source_path.display()
                )),
                Ok(output) => {
                    if output.status.success() {
                        HookRun::Allow
                    } else {
                        let mut reason = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        if reason.is_empty() {
                            reason = String::from_utf8_lossy(&output.stderr).trim().to_string();
                        }
                        if reason.is_empty() {
                            reason = format!(
                                "blocked by hook {} (exit {})",
                                hook.source_path.display(),
                                output.status.code().unwrap_or(-1)
                            );
                        }
                        if reason.chars().count() > MAX_REASON_CHARS {
                            reason = reason.chars().take(MAX_REASON_CHARS).collect::<String>() + "…";
                        }
                        HookRun::Block(reason)
                    }
                }
            }
        }
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
    fn parse_ignores_unknown_event() {
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
        assert!(hooks.is_empty());
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
            PreToolResult::Allow => panic!("expected block"),
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
            PreToolResult::Allow => panic!("expected block"),
        }
        let allowed = hooks.user_prompt_submit("sess", "plain request", &tmp, &cancel).await;
        assert_eq!(allowed, PreToolResult::Allow);
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
