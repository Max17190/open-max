//! Helpers the agent-loop integration tests share: a scripted endpoint that
//! records every request body, and a driver that runs one turn to its end.
//! Each test crate uses its own subset.
#![allow(dead_code)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A completions endpoint that answers each request with the next scripted
/// reply (one JSON body, no stream) and records every request body.
pub async fn recording_endpoint(responses: Vec<serde_json::Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let record = bodies.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        for body in responses {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            let mut need = usize::MAX;
            loop {
                let Ok(n) = sock.read(&mut chunk).await else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if need == usize::MAX {
                    if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                        need = end
                            + 4
                            + headers
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                    }
                }
                if need != usize::MAX && buf.len() >= need {
                    break;
                }
            }
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                record.lock().unwrap().push(String::from_utf8_lossy(&buf[end + 4..]).to_string());
            }
            let payload = serde_json::to_string(&body).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            let _ = sock.write_all(response.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    });
    (format!("http://{addr}"), bodies)
}

/// Run one turn to its end, approving every card, and return its events.
pub async fn drive_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session: &str,
    project: &Path,
    text: &str,
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
                core.respond_approval(approval_id, true);
            }
            AgentEvent::Done { .. } => break,
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
