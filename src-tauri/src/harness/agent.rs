use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::oneshot;

use super::client::{ChatClient, StreamDelta};
use super::prompt::system_prompt;
use super::tools;
use super::types::{AgentEvent, AgentEventEnvelope, ChatMessage};
use crate::settings::Settings;
use crate::state::{AppState, SessionData};
use crate::threads;

const MAX_ITERATIONS: usize = 50;
const APPROVAL_TIMEOUT: Duration = Duration::from_secs(600);
/// Stream tokens to the UI in ~25ms batches: cuts Tauri IPC traffic 10-40x
/// with no perceptible latency.
const FLUSH_INTERVAL: Duration = Duration::from_millis(25);

pub fn emit(app: &AppHandle, session_id: &str, event: AgentEvent) {
    let _ = app.emit(
        "agent_event",
        AgentEventEnvelope { session_id: session_id.to_string(), event },
    );
}

/// Kick off one agent turn in a thread. Errors if that thread is already running.
pub fn start_turn(
    app: AppHandle,
    session_id: String,
    project_root: PathBuf,
    user_text: String,
) -> Result<(), String> {
    let state = app.state::<AppState>();
    {
        let mut running = state.running.lock().unwrap();
        if running.contains(&session_id) {
            return Err("the agent is already working in this thread".into());
        }
        running.insert(session_id.clone());
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    state
        .cancel_flags
        .lock()
        .unwrap()
        .insert(session_id.clone(), cancelled.clone());

    let settings = state.settings.lock().unwrap().clone();
    threads::set_title_if_new(&app, &session_id, &user_text);

    tauri::async_runtime::spawn(async move {
        run_loop(&app, &session_id, &project_root, user_text, settings, cancelled).await;
        let state = app.state::<AppState>();
        state.running.lock().unwrap().remove(&session_id);
        state.cancel_flags.lock().unwrap().remove(&session_id);
    });
    Ok(())
}

/// Buffers streamed deltas and flushes them as batched events.
struct TokenBatcher {
    app: AppHandle,
    session_id: String,
    content: String,
    thinking: String,
    last_flush: Instant,
}

impl TokenBatcher {
    fn new(app: AppHandle, session_id: String) -> Self {
        Self { app, session_id, content: String::new(), thinking: String::new(), last_flush: Instant::now() }
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
            emit(&self.app, &self.session_id, AgentEvent::Token { text: std::mem::take(&mut self.content) });
        }
        if !self.thinking.is_empty() {
            emit(&self.app, &self.session_id, AgentEvent::Thinking { text: std::mem::take(&mut self.thinking) });
        }
        self.last_flush = Instant::now();
    }
}

async fn run_loop(
    app: &AppHandle,
    session_id: &str,
    project_root: &PathBuf,
    user_text: String,
    settings: Settings,
    cancelled: Arc<AtomicBool>,
) {
    let state = app.state::<AppState>();

    let mut messages = {
        let mut sessions = state.sessions.lock().await;
        let data = sessions.entry(session_id.to_string()).or_insert_with(|| SessionData {
            // Hydrate a persisted thread after app restart.
            messages: threads::load_messages(app, session_id)
                .unwrap_or_else(|| vec![ChatMessage::system(system_prompt(project_root))]),
            snapshots: Default::default(),
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
    let mut stop_reason = String::from("stop");

    'turns: for _ in 0..MAX_ITERATIONS {
        enforce_budget(&mut messages, settings.context_tokens.saturating_sub(settings.max_tokens + 1024));

        let batcher = Arc::new(StdMutex::new(TokenBatcher::new(app.clone(), session_id.to_string())));
        let batcher_in = batcher.clone();
        let result = client
            .stream_chat(&messages, &schemas, cancelled.clone(), move |delta| {
                batcher_in.lock().unwrap().push(delta);
            })
            .await;
        batcher.lock().unwrap().flush();

        let result = match result {
            Ok(r) => r,
            Err(message) => {
                emit(app, session_id, AgentEvent::Error { message });
                stop_reason = "error".into();
                break 'turns;
            }
        };

        messages.push(ChatMessage::assistant(
            if result.content.is_empty() { None } else { Some(result.content.clone()) },
            if result.tool_calls.is_empty() { None } else { Some(result.tool_calls.clone()) },
        ));
        save_messages(app, session_id, &messages).await;

        if cancelled.load(Ordering::Relaxed) {
            stop_reason = "cancelled".into();
            break 'turns;
        }
        if result.tool_calls.is_empty() {
            stop_reason = result.finish_reason;
            break 'turns;
        }

        for call in &result.tool_calls {
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
                    emit(app, session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: name.into(), args: Value::Null });
                    emit(app, session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.clone() });
                    messages.push(ChatMessage::tool(call.id.clone(), format!("Error: {msg}")));
                    continue;
                }
            };

            emit(app, session_id, AgentEvent::ToolStart {
                call_id: call.id.clone(),
                name: name.into(),
                args: args.clone(),
            });

            if tools::is_mutating(name) {
                snapshot_file(app, session_id, project_root, &args).await;
            }

            let outcome = if tools::is_mutating(name) && settings.approval_mode == "readonly" {
                tools::ToolOutcome {
                    ok: false,
                    output: "This session is read-only; mutating tools are disabled. Explain what you would do instead.".into(),
                    diff: None,
                }
            } else if tools::is_mutating(name) && settings.approval_mode == "ask" {
                let approved = request_approval(app, session_id, name, &args, &cancelled).await;
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
                emit(app, session_id, AgentEvent::Diff {
                    call_id: call.id.clone(),
                    path: diff.path.clone(),
                    diff: diff.diff.clone(),
                    added: diff.added,
                    removed: diff.removed,
                });
            }
            emit(app, session_id, AgentEvent::ToolEnd {
                call_id: call.id.clone(),
                ok: outcome.ok,
                output: outcome.output.clone(),
            });

            let content = if outcome.ok { outcome.output } else { format!("Error: {}", outcome.output) };
            messages.push(ChatMessage::tool(call.id.clone(), content));
        }
        save_messages(app, session_id, &messages).await;
    }

    save_messages(app, session_id, &messages).await;
    threads::touch(app, session_id);
    emit(app, session_id, AgentEvent::Done { stop_reason });
}

/// Record a file's pre-edit content the first time this thread touches it,
/// enabling cumulative per-file diffs.
async fn snapshot_file(app: &AppHandle, session_id: &str, project_root: &PathBuf, args: &Value) {
    let Some(rel) = args["path"].as_str() else { return };
    let content = std::fs::read_to_string(project_root.join(rel)).unwrap_or_default();
    let state = app.state::<AppState>();
    let mut sessions = state.sessions.lock().await;
    if let Some(data) = sessions.get_mut(session_id) {
        data.snapshots.entry(rel.to_string()).or_insert(content);
    }
}

async fn save_messages(app: &AppHandle, session_id: &str, messages: &[ChatMessage]) {
    let state = app.state::<AppState>();
    {
        let mut sessions = state.sessions.lock().await;
        if let Some(data) = sessions.get_mut(session_id) {
            data.messages = messages.to_vec();
        }
    }
    threads::save_messages(app, session_id, messages);
}

async fn request_approval(
    app: &AppHandle,
    session_id: &str,
    name: &str,
    args: &Value,
    cancelled: &Arc<AtomicBool>,
) -> bool {
    let approval_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = oneshot::channel::<bool>();
    {
        let state = app.state::<AppState>();
        state.approvals.lock().unwrap().insert(approval_id.clone(), tx);
    }
    emit(app, session_id, AgentEvent::ApprovalRequest {
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

    let state = app.state::<AppState>();
    state.approvals.lock().unwrap().remove(&approval_id);
    approved
}

/// Keep the transcript inside the model's context window: first truncate old
/// tool outputs, then drop the oldest exchanges (always preserving the system
/// prompt and the original user request).
fn enforce_budget(messages: &mut Vec<ChatMessage>, budget: usize) {
    let total = |msgs: &[ChatMessage]| msgs.iter().map(|m| m.estimated_tokens()).sum::<usize>();
    if total(messages) <= budget {
        return;
    }
    let keep_tail = messages.len().saturating_sub(6);
    for i in 1..keep_tail {
        if messages[i].role == "tool" {
            if let Some(c) = &messages[i].content {
                if c.len() > 600 {
                    let mut cut = 400;
                    while !c.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    messages[i].content = Some(format!("{}\n…[older tool output truncated]", &c[..cut]));
                }
            }
        }
        if total(messages) <= budget {
            return;
        }
    }
    // Drop whole exchanges starting after [system, first user]. Keep tool
    // replies consistent with the assistant message that requested them.
    while total(messages) > budget && messages.len() > 6 {
        let removed = messages.remove(2);
        if removed.role == "assistant" && removed.tool_calls.is_some() {
            while messages.len() > 2 && messages[2].role == "tool" {
                messages.remove(2);
            }
        }
    }
}
