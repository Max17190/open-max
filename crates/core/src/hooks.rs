//! Process lifecycle hooks: optional external commands that gate or observe
//! tool calls. Empty discovery costs almost nothing (one directory list).
//! Hooks never change tool schemas and never inject text into the model.

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
}

impl HookEvent {
    fn as_str(self) -> &'static str {
        match self {
            HookEvent::PreToolUse => "pre_tool_use",
            HookEvent::PostToolUse => "post_tool_use",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "pre_tool_use" => Some(HookEvent::PreToolUse),
            "post_tool_use" => Some(HookEvent::PostToolUse),
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
}

impl Hooks {
    /// Discover hooks under project `.openmax/hooks/` then global
    /// `~/.openmax/hooks/`. Project entries with the same file stem win.
    pub fn discover(project_root: &Path) -> Self {
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
                if let Some(spec) = parse_hook_file(&path) {
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
            }
        }
        hooks
    }

    pub fn is_empty(&self) -> bool {
        self.pre.is_empty() && self.post.is_empty()
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
            match run_hook(hook, session_id, tool, args, cwd, None, cancel).await {
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
            let _ = run_hook(
                hook,
                session_id,
                tool,
                args,
                cwd,
                Some(tool_ok),
                cancel,
            )
            .await;
        }
    }
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

fn hook_dirs(project_root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![project_root.join(".openmax").join("hooks")];
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".openmax").join("hooks"));
    }
    dirs
}

fn parse_hook_file(path: &Path) -> Option<HookSpec> {
    let text = std::fs::read_to_string(path).ok()?;
    let file: HookFile = toml::from_str(&text).ok()?;
    let event = HookEvent::parse(&file.event)?;
    let command = file.command.trim().to_string();
    if command.is_empty() {
        return None;
    }
    let tool_filter = file
        .tool
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    Some(HookSpec {
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
    session_id: &str,
    tool: &str,
    args: &Value,
    cwd: &Path,
    tool_ok: Option<bool>,
    cancel: &Arc<CancelToken>,
) -> HookRun {
    if cancel.is_cancelled() {
        return HookRun::Cancelled;
    }
    let payload = serde_json::json!({
        "event": hook.event.as_str(),
        "session_id": session_id,
        "tool": tool,
        "args": args,
        "cwd": cwd.display().to_string(),
        "tool_ok": tool_ok,
    });
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
