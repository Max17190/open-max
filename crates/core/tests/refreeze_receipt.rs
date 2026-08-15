//! The mid-turn refreeze receipt reaches the model, not just the UI.
//!
//! What this guards: an agent that writes a valid tool file gets a Refrozen
//! event on the wire, but that event is frontend chrome - nothing in the
//! transcript told the model its new tool became callable. The eval showed
//! the resulting failure shape twice: a correct tool authored, three
//! refreezes fired, and the model ran the script by hand instead of calling
//! the tool. The receipt must land where the model reads: in the transcript
//! the next request carries.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::AgentEvent;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A completions endpoint that answers each request with the next scripted
/// message and records every request body it saw.
async fn recording_endpoint(
    responses: Vec<serde_json::Value>,
) -> (String, Arc<Mutex<Vec<String>>>) {
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
                record
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[end + 4..]).to_string());
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

fn completion_with_tool_call(name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": format!("call-{name}"),
                    "type": "function",
                    "function": { "name": name, "arguments": args.to_string() }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn completion_with_text(text: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }]
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_request_after_a_refreeze_carries_the_receipt_and_the_new_tool_name() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    let manifest = "name = \"wordfreq\"\ndescription = \"top-n words\"\ncommand = \"/bin/echo\"\nmutating = false\n";
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/wordfreq.toml", "content": manifest }),
        ),
        completion_with_text("done"),
    ])
    .await;

    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        data.join("trust.json"),
        serde_json::json!({ "version": 1, "projects": [project.to_string_lossy()] }).to_string(),
    )
    .unwrap();

    let (core, mut rx) = Core::new(data).unwrap();
    start_turn(Arc::clone(&core), "receipt-test".into(), PathBuf::from(&project), "go".into())
        .unwrap();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        if matches!(envelope.event, AgentEvent::Done { .. }) {
            break;
        }
    }

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "the scripted turn makes exactly two requests");
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let tool_msg = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the write_file result is in the transcript");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("[extension refreeze:"),
        "the refreeze receipt must reach the model's transcript: {content}"
    );
    assert!(
        content.contains("wordfreq"),
        "the receipt names the tool that became callable: {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn write_config(data: &std::path::Path, base_url: &str, project: &std::path::Path) {
    std::fs::create_dir_all(data).unwrap();
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        data.join("trust.json"),
        serde_json::json!({ "version": 1, "projects": [project.to_string_lossy()] }).to_string(),
    )
    .unwrap();
}

async fn drive_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<open_max_core::types::AgentEventEnvelope>,
    session: &str,
    project: &std::path::Path,
    text: &str,
) {
    start_turn(Arc::clone(core), session.into(), project.to_path_buf(), text.into()).unwrap();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        if matches!(envelope.event, AgentEvent::Done { .. }) {
            break;
        }
    }
    // Done is emitted before the session's `running` flag clears; a second
    // start_turn racing that window is refused as "already working". Wait
    // for idle, as the headless front end does.
    while core.is_running(session) {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// A broken tool write refreezes too (its bytes moved the fingerprint), and
/// the receipt the model reads says the file did NOT load and why - without
/// this, "file changed" reads as "tool is live" and the model calls a tool
/// that does not exist, or worse, believes its work is done.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_broken_tool_write_gets_a_not_loaded_receipt() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    // Missing required field `command`: valid TOML, invalid manifest.
    let broken = "name = \"wordfreq\"\ndescription = \"top-n words\"\n";
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/wordfreq.toml", "content": broken }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);

    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "broken-receipt", &project, "go").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "the scripted turn makes exactly two requests");
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let tool_msg = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the write_file result is in the transcript");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("NOT loaded"),
        "the receipt must flag the failed load: {content}"
    );
    assert!(content.contains("wordfreq.toml"), "the receipt names the file: {content}");
    assert!(
        content.contains("command"),
        "the receipt carries the parse reason (missing field): {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The turn-start refreeze - a tool installed by a human, git, or another
/// session between turns - must be narrated to the MODEL, not only to the
/// UI. The next turn's first request carries a harness note naming the
/// added tool ahead of the user's prompt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_start_refreeze_note_reaches_the_model() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_text("hello"),
        completion_with_text("I see the checksum tool"),
    ])
    .await;
    write_config(&data, &base_url, &project);

    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "turnstart-receipt", &project, "first turn").await;

    // A teammate installs a tool while no turn is running.
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    std::fs::write(
        project.join(".openmax/tools/checksum.toml"),
        "name = \"checksum\"\ndescription = \"sha256\"\ncommand = \"/bin/echo\"\nmutating = false\n",
    )
    .unwrap();
    drive_turn(&core, &mut rx, "turnstart-receipt", &project, "second turn").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let messages = second["messages"].as_array().unwrap();
    let note = messages
        .iter()
        .find(|m| {
            m["role"] == "user"
                && m["content"].as_str().is_some_and(|c| c.starts_with("[extension refreeze:"))
        })
        .expect("the turn-start refreeze note is in the model's transcript");
    let content = note["content"].as_str().unwrap();
    assert!(
        content.contains("checksum"),
        "the note names the tool that appeared: {content}"
    );
    let last = messages.last().unwrap();
    assert_eq!(
        last["content"].as_str().unwrap(),
        "second turn",
        "the user's prompt stays the final message; the note precedes it"
    );
    let _ = std::fs::remove_dir_all(dir);
}
