//! A host that refuses a request for an image a tool just returned. Its own
//! test binary: turning image input on changes the read_file schema every
//! registry built in the process reads, which other tests would see.
mod common;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use open_max_core::{config::ApprovalMode, state::Core, types::AgentEvent};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const REFUSAL: &str = r#"{"error":{"message":"Provider returned error","code":400}}"#;

/// Answers a request that carries an image part with HTTP 400 and every
/// other one with the next scripted reply, recording each request body.
async fn refusing_images(replies: Vec<serde_json::Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let record = bodies.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        while let Ok((mut sock, _)) = listener.accept().await {
            let (mut buf, mut chunk, mut need) = (Vec::new(), [0u8; 8192], usize::MAX);
            while buf.len() < need {
                let Ok(n) = sock.read(&mut chunk).await else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if let (usize::MAX, Some(end)) = (need, buf.windows(4).position(|w| w == b"\r\n\r\n")) {
                    let headers = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                    let length = headers.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok());
                    need = end + 4 + length.unwrap_or(0);
                }
            }
            let end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
            let body = String::from_utf8_lossy(&buf[end + 4..]).to_string();
            let (status, payload) = if body.contains("image_url") {
                ("400 Bad Request", REFUSAL.to_string())
            } else {
                ("200 OK", replies.next().expect("a scripted reply").to_string())
            };
            record.lock().unwrap().push(body);
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                payload.len()
            );
            let _ = sock.write_all(response.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    });
    (format!("http://{addr}"), bodies)
}

fn reply(message: serde_json::Value, finish: &str) -> serde_json::Value {
    json!({"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":{"prompt_tokens":10,"completion_tokens":5}})
}

/// The bug this guards: a host refused a request for an image a tool result
/// had just added (a PNG whose header passed and whose body did not), and
/// the session ended. The request that carried the image is refused, the
/// image becomes a line naming the refusal, the request goes again without
/// it and the turn finishes; reopened, the session never sends it again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_image_is_sent_once_and_the_session_goes_on_without_it() {
    let call = json!({"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.png\"}"}});
    let (url, bodies) = refusing_images(vec![
        reply(json!({"role":"assistant","content":null,"tool_calls":[call]}), "tool_calls"),
        reply(json!({"role":"assistant","content":"done"}), "stop"),
        reply(json!({"role":"assistant","content":"done again"}), "stop"),
    ])
    .await;
    let dir: PathBuf = std::env::temp_dir().join(format!("openmax-refused-image-{}", uuid::Uuid::new_v4()));
    let (data, project) = (dir.join("data"), dir.join("project"));
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&project).unwrap();
    let png = [&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13][..], b"IHDR", &[0, 0, 0, 9, 0, 0, 0, 9]].concat();
    std::fs::write(project.join("a.png"), &png).unwrap();
    let settings = json!({"provider":"test","model":"seer","context_tokens":32768,"max_tokens":1024,"image_input":true});
    std::fs::write(data.join("settings.json"), settings.to_string()).unwrap();
    std::fs::write(data.join("providers.json"), json!({"providers":{"test":{"base_url":url,"api_key":"fixture-key"}}}).to_string()).unwrap();
    open_max_core::trust::grant_trust(&data, &project, ApprovalMode::Auto).unwrap();

    let (core, mut rx) = Core::new(data.clone()).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "look at a.png").await;
    let errors: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::Error { message } => Some(message.clone()), _ => None }).collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Retry { reason, .. } if reason.contains("refused images a tool just returned"))));
    let line = format!("[image 1 of this result ({} bytes) not shown: the provider refused it: Provider returned error]", png.len());
    {
        let sent = bodies.lock().unwrap();
        assert_eq!(sent.len(), 3, "the call, the refused request, the resend");
        assert!(!sent[0].contains("image_url"));
        assert!(sent[1].contains("image_url") && !sent[1].contains(&line), "the image went once: {}", sent[1]);
        assert!(!sent[2].contains("image_url") && sent[2].contains(&line), "the resend carries the line: {}", sent[2]);
    }

    common::close(&core, &session).await;
    drop(core);
    let (core, mut rx) = Core::new(data).unwrap();
    let events = common::drive_turn(&core, &mut rx, &session, &project, "again").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    let sent = bodies.lock().unwrap();
    assert_eq!(sent.len(), 4);
    assert!(!sent[3].contains("image_url") && sent[3].contains(&line), "reopened, the line and not the image: {}", sent[3]);
    let _ = std::fs::remove_dir_all(dir);
}
