use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::oneshot;

use crate::client::{ChatClient, StreamDelta};
use crate::config::Settings;
use crate::fallback;
use crate::prompt::system_prompt;
use crate::sessions;
use crate::state::{Core, SessionData};
use crate::tools;
use crate::types::{AgentEvent, ChatMessage};

const MAX_ITERATIONS: usize = 50;
const APPROVAL_TIMEOUT: Duration = Duration::from_secs(600);
/// Stream tokens to the UI in ~25ms batches: keeps redraw work negligible
/// with no perceptible latency.
const FLUSH_INTERVAL: Duration = Duration::from_millis(25);

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
    let cancelled = Arc::new(AtomicBool::new(false));
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
    cancelled: Arc<AtomicBool>,
) {
    let mut messages = {
        let mut sessions_map = core.sessions.lock().await;
        let data = sessions_map.entry(session_id.to_string()).or_insert_with(|| {
            if let Some(messages) = sessions::load_messages(core, session_id) {
                let count = messages.len();
                SessionData { messages, persisted_count: count, snapshots: Default::default() }
            } else {
                SessionData {
                    messages: vec![ChatMessage::system(system_prompt(project_root))],
                    persisted_count: 0,
                    snapshots: Default::default(),
                }
            }
        });
        data.messages.push(ChatMessage::user(user_text));
        data.messages.clone()
    };

    let client = ChatClient::new(
        settings.base_url.clone(),
        settings.api_key.clone(),
        settings.model.clone(),
        settings.temperature,
        settings.max_tokens,
    );
    let schemas = tools::tool_schemas();
    let known_tools = tools::TOOL_NAMES;
    // Every break assigns a real reason; this survives only if the model kept
    // calling tools until the iteration cap.
    let mut stop_reason = String::from("max_iterations");

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

        // Prefer structured calls from the server; when there are none,
        // recover calls from raw markup in the content (see fallback.rs).
        let mut content = result.content.clone();
        let mut tool_calls = result.tool_calls.clone();
        // Reasoning leaked into content is display-only: persisting it would
        // re-prefill dead tokens on every later turn.
        if let Some(clean) = fallback::strip_leading_think(&content) {
            content = clean;
        }
        if tool_calls.is_empty() {
            if let Some((clean, calls)) = fallback::extract_tool_calls(&content, known_tools) {
                content = clean;
                tool_calls = calls;
            }
        }
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

        if cancelled.load(Ordering::Relaxed) {
            stop_reason = "cancelled".into();
            break 'turns;
        }
        if tool_calls.is_empty() {
            stop_reason = result.finish_reason;
            break 'turns;
        }

        for call in &tool_calls {
            if cancelled.load(Ordering::Relaxed) {
                stop_reason = "cancelled".into();
                break 'turns;
            }
            let name = call.function.name.as_str();
            let args: Value = match serde_json::from_str(&call.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    // Malformed JSON from a small model: surface it so the model retries.
                    let msg = format!("invalid JSON in tool arguments: {e}");
                    core.send_agent(session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: name.into(), args: Value::Null });
                    core.send_agent(session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.clone() });
                    messages.push(ChatMessage::tool(call.id.clone(), format!("Error: {msg}")));
                    continue;
                }
            };

            core.send_agent(session_id, AgentEvent::ToolStart {
                call_id: call.id.clone(),
                name: name.into(),
                args: args.clone(),
            });

            if tools::is_mutating(name) {
                snapshot_file(core, session_id, project_root, &args).await;
            }

            // Read live so "[a]lways" during an approval prompt takes effect
            // for the rest of this turn, not just the next one.
            let approval_mode = core.settings.lock().unwrap().approval_mode.clone();
            let outcome = if tools::is_mutating(name) && approval_mode == "readonly" {
                tools::ToolOutcome {
                    ok: false,
                    output: "This session is read-only; mutating tools are disabled. Explain what you would do instead.".into(),
                    diff: None,
                }
            } else if tools::is_mutating(name) && approval_mode == "ask" {
                let approved = request_approval(core, session_id, name, &args, &cancelled).await;
                if approved {
                    tools::execute(name, &args, project_root).await
                } else {
                    tools::ToolOutcome {
                        ok: false,
                        output: "The user declined this action. Ask them how to proceed instead of retrying.".into(),
                        diff: None,
                    }
                }
            } else {
                tools::execute(name, &args, project_root).await
            };

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

            let content = if outcome.ok { outcome.output } else { format!("Error: {}", outcome.output) };
            messages.push(ChatMessage::tool(call.id.clone(), content));
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
    cancelled: &Arc<AtomicBool>,
) -> bool {
    let approval_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = oneshot::channel::<bool>();
    core.approvals.lock().unwrap().insert(approval_id.clone(), tx);
    core.send_agent(session_id, AgentEvent::ApprovalRequest {
        approval_id: approval_id.clone(),
        name: name.to_string(),
        summary: tools::summarize_call(name, args),
    });

    let cancelled = cancelled.clone();
    let wait_cancel = async move {
        loop {
            if cancelled.load(Ordering::Relaxed) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    };

    let approved = tokio::select! {
        r = rx => r.unwrap_or(false),
        _ = wait_cancel => false,
        _ = tokio::time::sleep(APPROVAL_TIMEOUT) => false,
    };

    core.approvals.lock().unwrap().remove(&approval_id);
    approved
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
    while total > target && messages.len() > 6 {
        let removed = messages.remove(2);
        total = total.saturating_sub(removed.estimated_tokens());
        if removed.role == "assistant" && removed.tool_calls.is_some() {
            while messages.len() > 2 && messages[2].role == "tool" {
                let tool = messages.remove(2);
                total = total.saturating_sub(tool.estimated_tokens());
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, len: usize) -> ChatMessage {
        ChatMessage { role: role.into(), content: Some("x".repeat(len)), tool_calls: None, tool_call_id: None }
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
        // prompt, the original request, and the most recent tail.
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("x".repeat(400).as_str()));
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
}
