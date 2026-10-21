//! Thesis gates for the agent loop: one test per target, named after it.
//!
//! Byte and count gates assert unconditionally. Each test's doc comment
//! states what it measures and how.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use serde_json::value::RawValue;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A completions endpoint that answers each request with the next scripted
/// reply (one JSON body, no stream) and records every request body.
async fn recording_endpoint(responses: Vec<serde_json::Value>) -> (String, Arc<Mutex<Vec<String>>>) {
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

fn text(reply: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": reply },
            "finish_reason": "stop"
        }]
    })
}

async fn drive_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session: &str,
    project: &Path,
    prompt: &str,
) {
    start_turn(Arc::clone(core), session.into(), project.to_path_buf(), prompt.into()).unwrap();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        if matches!(envelope.event, AgentEvent::Done { .. }) {
            break;
        }
    }
    // Done goes out before `running` clears; the next start_turn waits for idle.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while core.is_running(session) {
        assert!(std::time::Instant::now() < deadline, "session {session} still runs 10s after its Done event");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// The frozen prefix stays under 700 tokens for an empty project, checked as a
/// byte cap. The prefix is messages[0] (the base rules, the extension pointer,
/// and the project path) plus the `tools` array: what every request of a
/// session resends first, and what a provider's prompt cache keys on. Method:
/// a scripted endpoint records request 0 of a turn in an empty project trusted
/// in `auto`; the gate takes the text of messages[0] less the fixture's
/// project path, adds the `tools` array's raw wire bytes, and holds that
/// path-free payload to 2,881 bytes: 700 tokens less a 30-token allowance for
/// one project path, at 4.3 bytes per token. The path is left out because its
/// cost depends on the machine: the fixture's project sits in a uuid-named
/// dir, a 112-byte path under a macOS per-user temp dir and 60 bytes under
/// /tmp, which cost 53 to 66 and 26 to 39 tokens over 2,000 random uuids each
/// (1.5 to 2.3 bytes per token). Tokens are counted with tiktoken's o200k_base
/// on each field's own text, messages[0] and the tools array, and summed. On
/// request 0 from the release binary the path-free payload was 3,362 bytes and
/// 762 tokens before the prefix was cut and 2,599 bytes and 589 tokens after,
/// 4.41 bytes per token both times (cl100k_base counts 743 and 572), so the
/// cap takes 4.3. Paths are priced in place, in the prompt's first sentence:
/// twelve home-style project paths of 26 to 71 bytes cost 6 to 19 tokens (2.8
/// to 4.4 bytes per token), so the allowance holds an 83-byte path even at
/// 2.8. A byte cap is not a token count: text at fewer than 4.3 bytes per
/// token (short words, punctuation, identifiers) can pass it above the target,
/// which is why the budget ratchet in prompt.rs pins the same payload 63 bytes
/// over its size at this cut and asks for a tokenizer count before its cap
/// moves. The cut keeps the two instructions models break without them, and
/// the gate holds them: write_file prefers edit_file for changes, and the
/// closing summary does not repeat file contents. The execution policy note
/// that follows messages[0] (120 bytes, 22 o200k tokens, after; 258 bytes, 47
/// tokens, before) is history outside the counted prefix, though every request
/// resends it too, and is printed here beside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn frozen_prefix_is_under_700_tokens() {
    let dir = std::env::temp_dir().join(format!("omx-thesis-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let (base_url, bodies) = recording_endpoint(vec![text("ok")]).await;
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
            "context_tokens": 32768,
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
    drive_turn(&core, &mut rx, "thesis-frozen-prefix", &project, "hello").await;

    let bodies: Vec<String> = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 1, "the turn makes one request");
    let fields = serde_json::from_str::<BTreeMap<String, Box<RawValue>>>(&bodies[0]).unwrap();
    let tools = fields["tools"].get();
    let messages = serde_json::from_str::<Vec<serde_json::Value>>(fields["messages"].get()).unwrap();
    assert_eq!(messages[0]["role"], "system", "messages[0] is the system prompt");
    let system = messages[0]["content"].as_str().expect("the system prompt is text");
    let root = project.to_string_lossy();
    assert_eq!(system.matches(&*root).count(), 1, "the allowance prices one project path:\n{system}");
    let system_path_free = system.len() - root.len();
    let bytes = system_path_free + tools.len();
    const TARGET_TOKENS: usize = 700;
    const PATH_ALLOWANCE_TOKENS: usize = 30;
    const BYTES_PER_TOKEN_TENTHS: usize = 43;
    let cap = (TARGET_TOKENS - PATH_ALLOWANCE_TOKENS) * BYTES_PER_TOKEN_TENTHS / 10;
    let note = messages[1]["content"].as_str().unwrap_or_default();
    println!(
        "MEASURE frozen_prefix_path_free_bytes {bytes} = {system_path_free} messages[0] without the \
         {}-byte project path + {} tools; cap {cap}; then {} bytes of history ahead of the prompt: {note}",
        root.len(),
        tools.len(),
        note.len(),
    );
    assert!(
        bytes <= cap,
        "the path-free frozen prefix is {bytes} bytes, over {cap} ({TARGET_TOKENS} tokens less \
         {PATH_ALLOWANCE_TOKENS} for the project path, at {}.{} bytes per token): messages[0] \
         {system_path_free} bytes without the {}-byte path, plus tools {} bytes:\n{system}\n{tools}",
        BYTES_PER_TOKEN_TENTHS / 10,
        BYTES_PER_TOKEN_TENTHS % 10,
        root.len(),
        tools.len(),
    );
    // Bytes are not bought with the instructions models break without them.
    let schemas = serde_json::from_str::<Vec<serde_json::Value>>(tools).unwrap();
    let write_file = schemas
        .iter()
        .find(|tool| tool["function"]["name"] == "write_file")
        .expect("write_file is a built-in tool");
    let write_file = write_file["function"]["description"].as_str().unwrap_or_default();
    assert!(write_file.contains("prefer edit_file"), "write_file must point changes at edit_file: {write_file}");
    assert!(
        system.contains("without repeating file contents"),
        "the closing summary must not echo files back:\n{system}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
