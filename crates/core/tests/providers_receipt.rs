//! A providers.json edit gets a model-visible receipt.
//!
//! The file lives outside the project root, so the agent edits it with bash
//! and no refreeze covers it. Without a receipt, a valid edit is silent (the
//! agent cannot tell when it applies) and a malformed edit is worse than
//! silent: the runtime parses it to an EMPTY catalog, surfacing turns later
//! as "unknown provider" - an error pointed at the wrong problem.

use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::AgentEvent;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

fn tool_result_content(body: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the bash result is in the transcript")["content"]
        .as_str()
        .unwrap()
        .to_string()
}

/// Both edit shapes are narrated on the writing call's result: a valid
/// catalog edit says what loaded and when it applies; a malformed one says
/// the catalog is EMPTY - while the author can still fix it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn providers_edits_valid_and_broken_are_both_narrated() {
    // Build the environment first so the script can name the real path.
    let dir = std::env::temp_dir().join(format!("omx-prov-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let providers_path = data.join("providers.json");

    let good_script = format!(
        "printf '{{\"providers\":{{\"xai\":{{\"base_url\":\"https://api.x.ai/v1\",\"models\":[{{\"id\":\"grok-4.6\"}}]}}}}}}' > {}",
        providers_path.display()
    );
    let broken_script = format!("printf '{{\"providers\": nope' > {}", providers_path.display());

    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call("bash", serde_json::json!({ "command": good_script })),
        completion_with_tool_call("bash", serde_json::json!({ "command": broken_script })),
        completion_with_text("done"),
    ])
    .await;
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
            "context_tokens": 16384,
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        data.join("trust.json"),
        serde_json::json!({ "version": 1, "projects": [project.to_string_lossy()] }).to_string(),
    )
    .unwrap();

    let (core, mut rx) = Core::new(data.clone()).unwrap();
    start_turn(Arc::clone(&core), "prov-receipt".into(), project.clone(), "go".into()).unwrap();
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
    assert_eq!(bodies.len(), 3);
    // Request 2 carries the first bash result: the valid-edit receipt.
    let first_result = tool_result_content(&bodies[1]);
    assert!(
        first_result.contains("[providers.json changed:"),
        "a valid catalog edit is narrated: {first_result}"
    );
    assert!(first_result.contains("xai"), "{first_result}");
    assert!(
        first_result.contains("next turn"),
        "the receipt says when the edit applies: {first_result}"
    );
    // Request 3 carries the second bash result: the broken-edit receipt.
    let v: serde_json::Value = serde_json::from_str(&bodies[2]).unwrap();
    let second_result = v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "tool")
        .nth(1)
        .expect("both bash results are in the transcript")["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        second_result.contains("invalid JSON"),
        "a broken edit is named while the author can still fix it: {second_result}"
    );
    assert!(second_result.contains("EMPTY"), "{second_result}");
    let _ = std::fs::remove_dir_all(dir);
}
