use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::oneshot;

use crate::client::{ChatClient, StreamDelta};
use crate::config::Settings;
use crate::fallback;
use crate::prompt::{system_prompt_with_breakdown, PromptBreakdown};
use crate::registry::Registry;
use crate::sessions;
use crate::state::{CancelToken, Core, SessionData};
use crate::tools;
use crate::types::{AgentEvent, ChatMessage, ToolCall};

const MAX_ITERATIONS: usize = 50;
const APPROVAL_TIMEOUT: Duration = Duration::from_secs(600);
/// Stream tokens to the UI in ~25ms batches: keeps redraw work negligible
/// with no perceptible latency.
const FLUSH_INTERVAL: Duration = Duration::from_millis(25);
const DIGEST_PREFIX: &str = "[context note:";

/// Outcome of a mutating-tool approval prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ApprovalOutcome {
    Approved,
    Declined,
    Cancelled,
    TimedOut,
}

/// True when a native server tool call cannot be executed as-is.
fn is_native_call_broken(call: &ToolCall) -> bool {
    call.function.name.is_empty()
        || serde_json::from_str::<Value>(&call.function.arguments).is_err()
}

/// When every native call is broken, try to recover calls from content markup.
/// Broken natives are only discarded if the markup actually yields calls;
/// otherwise they are kept so each one gets its per-call error (which tells
/// the model to retry) instead of vanishing silently.
fn resolve_tool_calls(
    mut content: String,
    mut tool_calls: Vec<ToolCall>,
    known_tools: &[&str],
) -> (String, Vec<ToolCall>) {
    let all_broken = !tool_calls.is_empty() && tool_calls.iter().all(is_native_call_broken);
    if tool_calls.is_empty() || all_broken {
        if let Some((clean, calls)) = fallback::extract_tool_calls(&content, known_tools) {
            content = clean;
            tool_calls = calls;
        }
    }
    (content, tool_calls)
}

/// Detects identical tool calls repeated consecutively within one turn loop.
struct RepeatCallTracker {
    last_name: Option<String>,
    last_args: Option<String>,
    consecutive: u8,
}

impl RepeatCallTracker {
    fn new() -> Self {
        Self { last_name: None, last_args: None, consecutive: 0 }
    }

    /// Returns true when this would be the 3rd consecutive identical execution.
    fn would_block(&self, name: &str, args_key: &str) -> bool {
        self.last_name.as_deref() == Some(name)
            && self.last_args.as_deref() == Some(args_key)
            && self.consecutive >= 2
    }

    fn record_executed(&mut self, name: &str, args_key: &str) {
        if self.last_name.as_deref() == Some(name) && self.last_args.as_deref() == Some(args_key) {
            self.consecutive = self.consecutive.saturating_add(1);
        } else {
            self.last_name = Some(name.to_string());
            self.last_args = Some(args_key.to_string());
            self.consecutive = 1;
        }
    }
}

fn canonicalize_args(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<_> = map.iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            let sorted: serde_json::Map<String, Value> =
                pairs.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            serde_json::to_string(&Value::Object(sorted)).unwrap_or_default()
        }
        other => other.to_string(),
    }
}

struct CompactionDigest {
    message_count: usize,
    tools: BTreeSet<String>,
    paths: Vec<String>,
}

impl CompactionDigest {
    fn new() -> Self {
        Self { message_count: 0, tools: BTreeSet::new(), paths: Vec::new() }
    }

    fn record_message(&mut self, msg: &ChatMessage) {
        self.message_count += 1;
        if msg.role != "assistant" {
            return;
        }
        let Some(calls) = &msg.tool_calls else { return };
        for call in calls {
            if !call.function.name.is_empty() {
                self.tools.insert(call.function.name.clone());
            }
            if let Ok(v) = serde_json::from_str::<Value>(&call.function.arguments) {
                if let Some(path) = v.get("path").and_then(|p| p.as_str()) {
                    if self.paths.len() < 8 && !self.paths.iter().any(|p| p == path) {
                        self.paths.push(path.to_string());
                    }
                }
            }
        }
    }

    fn format(&self) -> String {
        let mut parts = vec![format!(
            "{DIGEST_PREFIX} {} earlier messages were compacted.",
            self.message_count
        )];
        if !self.tools.is_empty() {
            parts.push(format!("Tools used: {}.", self.tools.iter().cloned().collect::<Vec<_>>().join(", ")));
        }
        if !self.paths.is_empty() {
            parts.push(format!("Files touched: {}.", self.paths.join(", ")));
        }
        parts.push("Re-read files if you need the details.".into());
        parts.join(" ")
    }
}

fn is_digest_message(msg: &ChatMessage) -> bool {
    msg.role == "user"
        && msg.content.as_deref().is_some_and(|c| c.starts_with(DIGEST_PREFIX))
}

/// Kick off one agent turn in a session. Errors if that session is already running.
pub fn start_turn(
    core: Arc<Core>,
    session_id: String,
    project_root: PathBuf,
    user_text: String,
) -> Result<(), String> {
    {
        let mut running = core.running.lock().unwrap();
        if running.contains(&session_id) {
            return Err("the agent is already working in this session".into());
        }
        running.insert(session_id.clone());
    }
    let cancelled = Arc::new(CancelToken::default());
    core.cancel_flags
        .lock()
        .unwrap()
        .insert(session_id.clone(), cancelled.clone());

    let settings = core.settings.lock().unwrap().clone();
    sessions::set_title_if_new(&core, &session_id, &user_text);

    tokio::spawn(async move {
        run_loop(&core, &session_id, &project_root, user_text, settings, cancelled).await;
        core.running.lock().unwrap().remove(&session_id);
        core.cancel_flags.lock().unwrap().remove(&session_id);
    });
    Ok(())
}

/// Buffers streamed deltas and flushes them as batched events.
struct TokenBatcher {
    core: Arc<Core>,
    session_id: String,
    content: String,
    thinking: String,
    last_flush: Instant,
}

impl TokenBatcher {
    fn new(core: Arc<Core>, session_id: String) -> Self {
        Self { core, session_id, content: String::new(), thinking: String::new(), last_flush: Instant::now() }
    }

    fn push(&mut self, delta: StreamDelta) {
        match delta {
            StreamDelta::Content(t) => self.content.push_str(&t),
            StreamDelta::Reasoning(t) => self.thinking.push_str(&t),
        }
        if self.last_flush.elapsed() >= FLUSH_INTERVAL {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if !self.content.is_empty() {
            self.core.send_agent(&self.session_id, AgentEvent::Token { text: std::mem::take(&mut self.content) });
        }
        if !self.thinking.is_empty() {
            self.core.send_agent(&self.session_id, AgentEvent::Thinking { text: std::mem::take(&mut self.thinking) });
        }
        self.last_flush = Instant::now();
    }
}

async fn run_loop(
    core: &Arc<Core>,
    session_id: &str,
    project_root: &Path,
    user_text: String,
    settings: Settings,
    cancelled: Arc<CancelToken>,
) {
    let (mut messages, registry) = {
        let mut sessions_map = core.sessions.lock().await;
        let data = sessions_map.entry(session_id.to_string()).or_insert_with(|| {
            if let Some(messages) = sessions::load_messages(core, session_id) {
                let count = messages.len();
                // Resume with the registry frozen at creation (from the
                // manifest), never today's config: the persisted prompt and
                // the schema set must keep matching each other byte for
                // byte. No manifest means the session was builtin-only.
                let registry = match sessions::load_manifest(core, session_id) {
                    Some(manifest) => Registry::from_manifest(manifest),
                    None => Registry::builtin_only(),
                };
                let system_chars = messages
                    .first()
                    .filter(|m| m.role == "system")
                    .and_then(|m| m.content.as_deref())
                    .map(str::len)
                    .unwrap_or(0);
                let breakdown = PromptBreakdown::from_persisted(system_chars, &registry);
                SessionData {
                    messages,
                    registry: Arc::new(registry),
                    prompt_breakdown: Arc::new(breakdown),
                    persisted_count: count,
                    snapshots: Default::default(),
                }
            } else {
                let registry = Arc::new(Registry::build(project_root));
                if registry.has_extensions() {
                    sessions::save_manifest(core, session_id, &registry.to_manifest());
                }
                let (prompt, breakdown) =
                    system_prompt_with_breakdown(project_root, &registry);
                SessionData {
                    messages: vec![ChatMessage::system(prompt)],
                    registry,
                    prompt_breakdown: Arc::new(breakdown),
                    persisted_count: 0,
                    snapshots: Default::default(),
                }
            }
        });
        data.messages.push(ChatMessage::user(user_text));
        (data.messages.clone(), data.registry.clone())
    };

    let client = ChatClient::new(
        settings.base_url.clone(),
        settings.api_key.clone(),
        settings.model.clone(),
        settings.temperature,
        settings.max_tokens,
    );
    let schemas = registry.tool_schemas_json();
    let known_tools: Vec<&str> = registry.tools.iter().map(|s| s.name.as_str()).collect();
    let caps = tools::OutputCaps::from_settings(&settings);
    // Every break assigns a real reason; this survives only if the model kept
    // calling tools until the iteration cap.
    let mut stop_reason = String::from("max_iterations");
    let mut repeat_tracker = RepeatCallTracker::new();

    'turns: for _ in 0..MAX_ITERATIONS {
        let budget_changed = enforce_budget(&mut messages, settings.context_tokens.saturating_sub(settings.max_tokens + 1024));
        let used = messages.iter().map(|m| m.estimated_tokens()).sum();
        core.send_agent(session_id, AgentEvent::Budget { used_tokens: used, context_tokens: settings.context_tokens });

        let batcher = Arc::new(StdMutex::new(TokenBatcher::new(core.clone(), session_id.to_string())));
        let batcher_in = batcher.clone();
        let result = client
            .stream_chat(&messages, schemas, cancelled.clone(), move |delta| {
                batcher_in.lock().unwrap().push(delta);
            })
            .await;
        batcher.lock().unwrap().flush();

        let result = match result {
            Ok(r) => r,
            Err(message) => {
                core.send_agent(session_id, AgentEvent::Error { message });
                stop_reason = "error".into();
                break 'turns;
            }
        };

        if let Some(u) = result.usage {
            core.send_agent(session_id, AgentEvent::Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                cached_tokens: u.cached_tokens,
            });
        }

        // Prefer structured calls from the server; when there are none (or all
        // are broken), recover calls from raw markup in the content (see fallback.rs).
        let mut content = result.content.clone();
        let mut tool_calls = result.tool_calls.clone();
        // Reasoning leaked into content is display-only: persisting it would
        // re-prefill dead tokens on every later turn.
        if let Some(clean) = fallback::strip_leading_think(&content) {
            content = clean;
        }
        (content, tool_calls) = resolve_tool_calls(content, tool_calls, &known_tools);
        core.send_agent(session_id, AgentEvent::MessageDone { text: content.clone() });

        // Never persist a fully empty assistant message (e.g. a turn cancelled
        // before the first token): chat templates can reject it on replay.
        if !content.is_empty() || !tool_calls.is_empty() {
            messages.push(ChatMessage::assistant(
                if content.is_empty() { None } else { Some(content.clone()) },
                if tool_calls.is_empty() { None } else { Some(tool_calls.clone()) },
            ));
            save_messages(core, session_id, &messages, budget_changed).await;
        }

        if cancelled.is_cancelled() {
            stop_reason = "cancelled".into();
            break 'turns;
        }
        if tool_calls.is_empty() {
            stop_reason = result.finish_reason;
            break 'turns;
        }

        for call in &tool_calls {
            if cancelled.is_cancelled() {
                stop_reason = "cancelled".into();
                break 'turns;
            }
            let name = call.function.name.as_str();
            if name.is_empty() {
                let msg = "tool call has an empty function name; use a known tool name from the schema";
                core.send_agent(session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: String::new(), args: Value::Null });
                core.send_agent(session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.into() });
                messages.push(ChatMessage::tool(call.id.clone(), format!("Error: {msg}")));
                continue;
            }
            let args: Value = match serde_json::from_str(&call.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("invalid JSON in tool arguments: {e}");
                    core.send_agent(session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: name.into(), args: Value::Null });
                    core.send_agent(session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.clone() });
                    messages.push(ChatMessage::tool(call.id.clone(), format!("Error: {msg}")));
                    continue;
                }
            };

            let args_key = canonicalize_args(&args);
            if repeat_tracker.would_block(name, &args_key) {
                let msg = "You have repeated this exact call 3 times. The result will not change. Try a different approach, or explain what you are blocked on.";
                core.send_agent(session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: name.into(), args: args.clone() });
                core.send_agent(session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.into() });
                messages.push(ChatMessage::tool(call.id.clone(), msg.to_string()));
                continue;
            }

            core.send_agent(session_id, AgentEvent::ToolStart {
                call_id: call.id.clone(),
                name: name.into(),
                args: args.clone(),
            });

            if registry.is_mutating(name) {
                snapshot_file(core, session_id, project_root, &args).await;
            }

            // Read live so "[a]lways" during an approval prompt takes effect
            // for the rest of this turn, not just the next one.
            let approval_mode = core.settings.lock().unwrap().approval_mode.clone();
            let mut executed = false;
            let (outcome, turn_cancelled) = if registry.is_mutating(name) && approval_mode == "readonly" {
                (tools::ToolOutcome {
                    ok: false,
                    output: "This session is read-only; mutating tools are disabled. Explain what you would do instead.".into(),
                    diff: None,
                }, false)
            } else if registry.is_mutating(name) && approval_mode == "ask" {
                match request_approval(core, session_id, name, &args, &cancelled).await {
                    ApprovalOutcome::Approved => {
                        executed = true;
                        (registry.execute(name, &args, project_root, caps).await, false)
                    }
                    ApprovalOutcome::Declined => (tools::ToolOutcome {
                        ok: false,
                        output: "The user declined this action. Ask them how to proceed instead of retrying.".into(),
                        diff: None,
                    }, false),
                    ApprovalOutcome::TimedOut => (tools::ToolOutcome {
                        ok: false,
                        output: "Approval request timed out with no response. Stop and summarize what you were about to do.".into(),
                        diff: None,
                    }, false),
                    ApprovalOutcome::Cancelled => (tools::ToolOutcome {
                        ok: false,
                        output: "The user cancelled this turn.".into(),
                        diff: None,
                    }, true),
                }
            } else {
                executed = true;
                (registry.execute(name, &args, project_root, caps).await, false)
            };

            if turn_cancelled {
                core.send_agent(session_id, AgentEvent::ToolEnd {
                    call_id: call.id.clone(),
                    ok: false,
                    output: "The user cancelled this turn.".into(),
                });
                messages.push(ChatMessage::tool(call.id.clone(), "The user cancelled this turn."));
                stop_reason = "cancelled".into();
                break 'turns;
            }

            if let Some(diff) = &outcome.diff {
                core.send_agent(session_id, AgentEvent::Diff {
                    call_id: call.id.clone(),
                    path: diff.path.clone(),
                    diff: diff.diff.clone(),
                    added: diff.added,
                    removed: diff.removed,
                });
            }
            core.send_agent(session_id, AgentEvent::ToolEnd {
                call_id: call.id.clone(),
                ok: outcome.ok,
                output: outcome.output.clone(),
            });

            // Approval timeouts are not model errors; the "Error:" prefix
            // would push small models into pointless retry loops.
            let content = if outcome.ok || outcome.output.starts_with("Approval request timed out") {
                outcome.output
            } else {
                format!("Error: {}", outcome.output)
            };
            messages.push(ChatMessage::tool(call.id.clone(), content));
            if executed {
                repeat_tracker.record_executed(name, &args_key);
            }
        }
        save_messages(core, session_id, &messages, false).await;
    }

    save_messages(core, session_id, &messages, false).await;
    sessions::touch(core, session_id);
    core.send_agent(session_id, AgentEvent::Done { stop_reason });
}

/// Record a file's pre-edit content the first time this session touches it,
/// enabling cumulative per-file diffs.
async fn snapshot_file(core: &Arc<Core>, session_id: &str, project_root: &Path, args: &Value) {
    let Some(rel) = args["path"].as_str() else { return };
    let content = std::fs::read_to_string(project_root.join(rel)).unwrap_or_default();
    let mut sessions_map = core.sessions.lock().await;
    if let Some(data) = sessions_map.get_mut(session_id) {
        data.snapshots.entry(rel.to_string()).or_insert(content);
    }
}

async fn save_messages(core: &Arc<Core>, session_id: &str, messages: &[ChatMessage], rewrite: bool) {
    let mut sessions_map = core.sessions.lock().await;
    if let Some(data) = sessions_map.get_mut(session_id) {
        data.messages = messages.to_vec();
        sessions::save_messages(core, session_id, messages, &mut data.persisted_count, rewrite);
    }
}

async fn request_approval(
    core: &Arc<Core>,
    session_id: &str,
    name: &str,
    args: &Value,
    cancelled: &Arc<CancelToken>,
) -> ApprovalOutcome {
    let approval_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = oneshot::channel::<bool>();
    core.approvals.lock().unwrap().insert(approval_id.clone(), tx);
    core.send_agent(session_id, AgentEvent::ApprovalRequest {
        approval_id: approval_id.clone(),
        name: name.to_string(),
        summary: crate::registry::summarize_call(name, args),
    });

    let outcome = tokio::select! {
        r = rx => match r {
            Ok(true) => ApprovalOutcome::Approved,
            Ok(false) => ApprovalOutcome::Declined,
            Err(_) => ApprovalOutcome::Declined,
        },
        _ = cancelled.cancelled() => ApprovalOutcome::Cancelled,
        _ = tokio::time::sleep(APPROVAL_TIMEOUT) => ApprovalOutcome::TimedOut,
    };

    core.approvals.lock().unwrap().remove(&approval_id);
    outcome
}

/// Keep the transcript inside the model's context window: first truncate old
/// tool outputs, then drop the oldest exchanges (always preserving the system
/// prompt and the original user request). Returns true when messages changed.
///
/// Prunes with hysteresis: once the budget is crossed, compact well below it
/// (PRUNE_TARGET_PCT) in a single pass. The server-side prompt cache re-prefills
/// from the first byte that diverges, so mutating early messages every
/// iteration would force a near-full prefill per agent step; pruning hard and
/// then leaving history untouched keeps the transcript append-only (and the
/// cache warm) until the budget is crossed again.
const PRUNE_TARGET_PCT: usize = 70;

fn enforce_budget(messages: &mut Vec<ChatMessage>, budget: usize) -> bool {
    let mut total: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
    if total <= budget {
        return false;
    }
    let target = budget * PRUNE_TARGET_PCT / 100;
    let keep_tail = messages.len().saturating_sub(6);
    for msg in messages.iter_mut().take(keep_tail).skip(1) {
        if msg.role == "tool" {
            if let Some(c) = &msg.content {
                if c.len() > 600 {
                    let mut cut = 160;
                    while !c.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    let old = msg.estimated_tokens();
                    msg.content = Some(format!("{}\n…[older tool output truncated]", &c[..cut]));
                    total = total.saturating_sub(old).saturating_add(msg.estimated_tokens());
                }
            }
        }
        if total <= target {
            return true;
        }
    }
    // Drop whole exchanges starting after [system, first user]. Keep tool
    // replies consistent with the assistant message that requested them.
    let mut digest = CompactionDigest::new();
    while total > target && messages.len() > 6 {
        let removed = messages.remove(2);
        digest.record_message(&removed);
        total = total.saturating_sub(removed.estimated_tokens());
        if removed.role == "assistant" && removed.tool_calls.is_some() {
            while messages.len() > 2 && messages[2].role == "tool" {
                let tool = messages.remove(2);
                digest.record_message(&tool);
                total = total.saturating_sub(tool.estimated_tokens());
            }
        }
    }
    if digest.message_count > 0 {
        let note = ChatMessage::user(digest.format());
        if messages.len() > 2 && is_digest_message(&messages[2]) {
            messages[2] = note;
        } else {
            messages.insert(2, note);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ToolCall, ToolCallFunction};

    fn msg(role: &str, len: usize) -> ChatMessage {
        ChatMessage { role: role.into(), content: Some("x".repeat(len)), tool_calls: None, tool_call_id: None }
    }

    fn assistant_with_tools(name: &str, args: &str) -> ChatMessage {
        ChatMessage::assistant(
            None,
            Some(vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: ToolCallFunction {
                    name: name.into(),
                    arguments: args.into(),
                },
            }]),
        )
    }

    #[test]
    fn broken_native_calls_fall_back_to_markup() {
        let known = ["read_file", "bash"];
        let content = "I'll read it.\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}</tool_call>";
        let broken = vec![ToolCall {
            id: "call_0".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: String::new(),
                arguments: "{not json".into(),
            },
        }];
        let (clean, calls) = resolve_tool_calls(content.into(), broken, &known);
        assert_eq!(clean, "I'll read it.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
    }

    #[test]
    fn partial_broken_native_calls_keep_native() {
        let known = ["read_file", "bash"];
        let good = ToolCall {
            id: "call_0".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "bash".into(),
                arguments: r#"{"command":"echo hi"}"#.into(),
            },
        };
        let bad = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: String::new(),
                arguments: "nope".into(),
            },
        };
        let (clean, calls) = resolve_tool_calls("run".into(), vec![good, bad], &known);
        assert_eq!(clean, "run");
        assert_eq!(calls.len(), 2);
        assert!(is_native_call_broken(&calls[1]));
    }

    #[test]
    fn repeat_tracker_blocks_third_identical_call() {
        let mut t = RepeatCallTracker::new();
        assert!(!t.would_block("bash", r#"{"command":"ls"}"#));
        t.record_executed("bash", r#"{"command":"ls"}"#);
        assert!(!t.would_block("bash", r#"{"command":"ls"}"#));
        t.record_executed("bash", r#"{"command":"ls"}"#);
        assert!(t.would_block("bash", r#"{"command":"ls"}"#));
    }

    #[test]
    fn repeat_tracker_resets_on_different_call() {
        let mut t = RepeatCallTracker::new();
        t.record_executed("bash", r#"{"command":"ls"}"#);
        t.record_executed("bash", r#"{"command":"ls"}"#);
        assert!(!t.would_block("read_file", r#"{"path":"a.rs"}"#));
        t.record_executed("read_file", r#"{"path":"a.rs"}"#);
        assert!(!t.would_block("read_file", r#"{"path":"a.rs"}"#));
    }

    #[test]
    fn budget_preserves_system_and_first_user() {
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..20 {
            messages.push(msg("assistant", 2000));
            messages.push(msg("user", 2000));
        }
        enforce_budget(&mut messages, 2000);
        // Old exchanges are dropped down to the guaranteed floor: the system
        // prompt, the original request, a compaction digest, and the most recent tail.
        assert_eq!(messages.len(), 7);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("x".repeat(400).as_str()));
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
    }

    #[test]
    fn budget_truncates_old_tool_output_first() {
        let mut messages = vec![msg("system", 100), msg("user", 100)];
        messages.push(msg("tool", 4000));
        messages.push(msg("assistant", 100));
        // Recent tail that must stay intact.
        for _ in 0..3 {
            messages.push(msg("user", 100));
            messages.push(msg("assistant", 100));
        }
        enforce_budget(&mut messages, 700);
        assert_eq!(messages.len(), 10, "nothing should be dropped, only truncated");
        let tool_len = messages[2].content.as_deref().unwrap().len();
        assert!(tool_len < 500, "old tool output should be truncated, got {tool_len}");
    }

    /// One prune must buy headroom: after compaction the transcript sits at or
    /// below the prune target, and re-running enforce_budget mutates nothing,
    /// so the token prefix (and the server's prompt cache) stays stable while
    /// the next iterations append.
    #[test]
    fn budget_prunes_once_with_hysteresis() {
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..8 {
            messages.push(msg("assistant", 100));
            messages.push(msg("tool", 3000));
        }
        let budget = 4000;
        assert!(enforce_budget(&mut messages, budget));
        let total: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
        assert!(
            total <= budget * PRUNE_TARGET_PCT / 100,
            "prune should reach the target, got {total} of {budget}"
        );

        let snapshot: Vec<Option<String>> = messages.iter().map(|m| m.content.clone()).collect();
        assert!(!enforce_budget(&mut messages, budget), "second pass must be a no-op");
        let after: Vec<Option<String>> = messages.iter().map(|m| m.content.clone()).collect();
        assert_eq!(snapshot, after, "no message may change between prunes");
    }

    #[test]
    fn budget_digest_replaced_not_stacked() {
        let mut messages = vec![msg("system", 100), msg("user", 100)];
        for i in 0..12 {
            messages.push(assistant_with_tools("read_file", &format!(r#"{{"path":"src/{i}.rs"}}"#)));
            messages.push(msg("tool", 2500));
        }
        let budget = 3000;
        assert!(enforce_budget(&mut messages, budget));
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
        let first_digest = messages[2].content.clone();
        assert!(!enforce_budget(&mut messages, budget), "second pass must be a no-op");
        assert_eq!(messages[2].content, first_digest, "digest must not be replaced on no-op");

        for _ in 0..6 {
            messages.push(assistant_with_tools("edit_file", r#"{"path":"src/new.rs"}"#));
            messages.push(msg("tool", 2500));
        }
        assert!(enforce_budget(&mut messages, budget));
        let digest_count = messages
            .iter()
            .filter(|m| m.content.as_deref().is_some_and(|c| c.starts_with(DIGEST_PREFIX)))
            .count();
        assert_eq!(digest_count, 1, "only one digest note may exist");
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
    }
}
