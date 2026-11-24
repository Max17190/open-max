//! Helpers the agent-loop integration tests share: a scripted endpoint that
//! records every request body, and a driver that runs one turn to its end.
//! Each test crate uses its own subset.
#![allow(dead_code)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
mod recording;
pub use recording::recording_http_with;

/// A completions endpoint that answers each request with the next scripted
/// reply (one JSON body, no stream) and records every request body.
pub async fn recording_endpoint(responses: Vec<serde_json::Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    recording_http(responses, false).await
}

/// Each value is one terminal Responses event, framed as SSE.
pub async fn recording_responses(responses: Vec<serde_json::Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    recording_http(responses, true).await
}

async fn recording_http(responses: Vec<serde_json::Value>, sse: bool) -> (String, Arc<Mutex<Vec<String>>>) {
    let responses = responses.into_iter().map(|body| {
        let payload = body.to_string();
        if sse {
            (200, "text/event-stream", format!("data: {payload}\n\n"))
        } else {
            (200, "application/json", payload)
        }
    });
    recording_http_script(responses.collect()).await
}

pub async fn recording_http_script(responses: Vec<(u16, &'static str, String)>) -> (String, Arc<Mutex<Vec<String>>>) {
    let mut replies = responses.into_iter();
    recording_http_with(replies.len(), move |_| replies.next().unwrap()).await
}

/// Run one turn to its end, approving every card, and return its events.
pub async fn drive_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session: &str,
    project: &Path,
    text: &str,
) -> Vec<AgentEvent> {
    drive_steered_turn(core, rx, session, project, text, None).await
}

/// `drive_turn`, sending `steer` to the turn while its first card waits:
/// the turn takes it before its next request, as a message sent while a
/// tool waits for approval.
pub async fn drive_steered_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session: &str,
    project: &Path,
    text: &str,
    mut steer: Option<&str>,
) -> Vec<AgentEvent> {
    start_turn(Arc::clone(core), session.into(), project.to_path_buf(), text.into()).unwrap();
    let mut events = Vec::new();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match &envelope.event {
            AgentEvent::ApprovalRequest { approval_id, .. } => {
                if let Some(text) = steer.take() {
                    core.steer(session, text.into()).unwrap();
                }
                core.respond_approval(approval_id, true);
            }
            AgentEvent::Done { .. } => {
                events.push(envelope.event);
                break;
            }
            _ => {}
        }
        events.push(envelope.event);
    }
    // Done is emitted before the session's `running` flag clears; a second
    // start_turn racing that window is refused as "already working". Wait
    // for idle, as the headless front end does.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while core.is_running(session) {
        assert!(std::time::Instant::now() < deadline, "session {session} still runs 10s after its Done event");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    events
}

/// Let go of the session, as a frontend does when it closes: the next
/// `Core` can claim it then, which is what a resumed sitting needs. Retried
/// while the turn that just ended still holds the session state.
pub async fn close(core: &Core, session: &str) {
    while open_max_core::sessions::detach(core, session).is_err() {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// Where the trailing note starts inside the last message's content.
pub fn note_marker() -> String {
    format!("{}{}", open_max_core::prompt::NOTE_SEPARATOR, open_max_core::prompt::NOTE_HEADER)
}
