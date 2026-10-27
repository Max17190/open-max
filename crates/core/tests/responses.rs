//! The Responses adapter drives the same trusted loop, approvals, and local sessions.
mod common;

use open_max_core::{config::ApprovalMode, state::Core, types::AgentEvent};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("openmax-responses-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("project")).unwrap();
        std::fs::create_dir_all(dir.join("data")).unwrap();
        Self(dir.canonicalize().unwrap())
    }
    fn project(&self) -> PathBuf {
        self.0.join("project")
    }
    fn data(&self) -> PathBuf {
        self.0.join("data")
    }
    fn configure(&self, url: &str) {
        std::fs::write(
            self.data().join("settings.json"),
            json!({"provider":"test", "model":"test-model","context_tokens":32768}).to_string(),
        )
        .unwrap();
        std::fs::write(
            self.data().join("providers.json"),
            json!({"providers":{"test":{"base_url":url,"wire_api":"responses"}}}).to_string(),
        )
        .unwrap();
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn event(value: Value) -> String {
    format!("event: {}\ndata: {value}\n\n", value["type"].as_str().unwrap())
}
fn done(output: Value) -> String {
    event(json!({"type":"response.completed","response":{"status":"completed","output":output,
        "usage":{"input_tokens":120,"output_tokens":15,"input_tokens_details":{"cached_tokens":90}}}}))
}
fn message(text: &str) -> Value {
    json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":text}]})
}
fn call(name: &str, args: Value) -> Value {
    json!({"type":"function_call","call_id":"call-1","name":name,"arguments":args.to_string()})
}

async fn server(replies: Vec<String>) -> (String, Arc<Mutex<Vec<Value>>>, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let records = Arc::new(Mutex::new(Vec::new()));
    let out = records.clone();
    let job = tokio::spawn(async move {
        for reply in replies {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            loop {
                let mut buf = [0; 8192];
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buf[..n]);
                if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..end]);
                    assert!(headers.starts_with("POST /responses "));
                    let len: usize = headers
                        .lines()
                        .find_map(|l| {
                            l.to_lowercase().strip_prefix("content-length:").map(|s| s.trim().parse().unwrap())
                        })
                        .unwrap();
                    if bytes.len() >= end + 4 + len {
                        out.lock().unwrap().push(serde_json::from_slice(&bytes[end + 4..end + 4 + len]).unwrap());
                        break;
                    }
                }
            }
            socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", reply.len()).as_bytes()).await.unwrap();
            // Deliberately split Unicode and SSE frames across network writes.
            for chunk in reply.as_bytes().chunks(7) {
                if socket.write_all(chunk).await.is_err() {
                    break;
                }
            }
        }
    });
    (url, records, job)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_round_and_resume_preserve_instructions_tools_reasoning_and_approvals() {
    for streamed_items in [false, true] {
        tool_round_and_resume_case(streamed_items).await;
    }
}

async fn tool_round_and_resume_case(streamed_items: bool) {
    let scratch = Scratch::new();
    let reasoning = json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-test-state"});
    let first = done(json!([reasoning, call("write_file", json!({"path":"result.txt","content":"hello"}))]));
    let second = format!(
        "{}{}",
        event(json!({"type":"response.output_text.delta","delta":"Done ✓"})),
        done(json!([message("Done ✓")]))
    );
    let replies = vec![first, second, done(json!([message("Resumed")]))];
    let replies = if streamed_items {
        replies.into_iter().map(|reply| {
            reply.split("\n\n").filter_map(|frame| {
                let data = frame.lines().find_map(|line| line.strip_prefix("data: "))?;
                let mut value: Value = serde_json::from_str(data).unwrap();
                if value["type"] != "response.completed" { return Some(event(value)); }
                let output = value["response"]["output"].take();
                let items = output.as_array().unwrap().iter().enumerate().map(|(index, item)| {
                    event(json!({"type":"response.output_item.done","output_index":index,"item":item}))
                }).collect::<String>();
                value["response"].as_object_mut().unwrap().remove("output");
                Some(items + &event(value))
            }).collect()
        }).collect()
    } else { replies };
    let (url, requests, job) = server(replies).await;
    scratch.configure(&url);
    open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Ask).unwrap();
    let (core, mut rx) = Core::new(scratch.data()).unwrap();
    let session = open_max_core::sessions::create(&core, scratch.project().display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &scratch.project(), "write result").await;
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ApprovalRequest { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Token { text } if text.contains('✓'))));
    assert_eq!(std::fs::read_to_string(scratch.project().join("result.txt")).unwrap(), "hello");
    common::close(&core, &session).await;
    drop(core);
    let (core, mut rx) = Core::new(scratch.data()).unwrap();
    let events = common::drive_turn(&core, &mut rx, &session, &scratch.project(), "continue").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    job.await.unwrap();
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    for request in requests.iter() {
        assert_eq!(request["store"], false);
        assert_eq!(request["stream"], true);
        assert_eq!(request["input"][0]["role"], "developer");
        assert_eq!(request["input"][0], requests[0]["input"][0]);
        assert_eq!(request["input"][1], requests[0]["input"][1]);
        assert!(request.get("previous_response_id").is_none());
    }
    assert_eq!(requests[0]["input"][1]["tools"].as_array().unwrap().len(), 7);
    for request in &requests[1..] {
        let items = request["input"].as_array().unwrap();
        assert!(items.contains(&reasoning));
        assert!(items.iter().any(|i| i["type"] == "function_call_output" && i["call_id"] == "call-1"));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn untrusted_project_never_reaches_responses() {
    let scratch = Scratch::new();
    let (url, requests, job) = server(vec![done(json!([message("unreachable")]))]).await;
    scratch.configure(&url);
    let (core, _) = Core::new(scratch.data()).unwrap();
    let error = open_max_core::agent::start_turn(core, "untrusted".into(), scratch.project(), "go".into()).unwrap_err();
    assert!(error.contains("not trusted"));
    assert!(requests.lock().unwrap().is_empty());
    job.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn readonly_and_deny_rules_still_gate_responses_calls() {
    for mode in [ApprovalMode::Readonly, ApprovalMode::Auto] {
        let scratch = Scratch::new();
        if mode == ApprovalMode::Auto {
            std::fs::create_dir_all(scratch.project().join(".openmax")).unwrap();
            std::fs::write(
                scratch.project().join(".openmax/permissions.toml"),
                "[[rules]]\neffect = \"deny\"\ntool = \"write_file\"\n",
            )
            .unwrap();
        }
        let (url, _, job) = server(vec![
            done(json!([call("write_file", json!({"path":"forbidden","content":"bad"}))])),
            done(json!([message("blocked")])),
        ])
        .await;
        scratch.configure(&url);
        open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), mode).unwrap();
        let (core, mut rx) = Core::new(scratch.data()).unwrap();
        let events = common::drive_turn(&core, &mut rx, "blocked", &scratch.project(), "write").await;
        assert!(!scratch.project().join("forbidden").exists());
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolEnd { ok: false, .. })), "{events:?}");
        job.await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_incomplete_and_interrupted_streams_never_run_partial_calls() {
    for ending in [
        event(
            json!({"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded","message":"limit"}}}),
        ),
        event(json!({"type":"response.incomplete","response":{"status":"incomplete"}})),
        String::new(),
    ] {
        let scratch = Scratch::new();
        let payload = format!(
            "{}{}{}",
            event(json!({"type":"response.output_text.delta","delta":"Working"})),
            event(
                json!({"type":"response.output_item.done","output_index":0,"item":call("write_file",json!({"path":"forbidden","content":"bad"}))})
            ),
            ending
        );
        let (url, requests, job) = server(vec![payload]).await;
        scratch.configure(&url);
        open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Auto).unwrap();
        let (core, mut rx) = Core::new(scratch.data()).unwrap();
        let events = common::drive_turn(&core, &mut rx, "cut", &scratch.project(), "write").await;
        assert!(!scratch.project().join("forbidden").exists());
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
        assert!(!events.iter().any(|e| matches!(e, AgentEvent::ToolStart { .. })));
        assert_eq!(requests.lock().unwrap().len(), 1);
        job.await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_switch_withholds_opaque_responses_reasoning() {
    let scratch = Scratch::new();
    let (url, _, job) = server(vec![done(
        json!([{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"private-state"},message("first")]),
    )])
    .await;
    scratch.configure(&url);
    open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Auto).unwrap();
    let (core, mut rx) = Core::new(scratch.data()).unwrap();
    common::drive_turn(&core, &mut rx, "switch", &scratch.project(), "first").await;
    let (other, requests) = common::recording_endpoint(vec![
        json!({"choices":[{"message":{"role":"assistant","content":"second"},"finish_reason":"stop"}]}),
    ])
    .await;
    {
        let mut settings = core.settings.lock().unwrap();
        settings.provider = None;
        settings.base_url = other;
    }
    common::drive_turn(&core, &mut rx, "switch", &scratch.project(), "second").await;
    assert!(!requests.lock().unwrap()[0].contains("private-state"));
    job.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn quota_and_invalid_completed_calls_fail_without_execution_or_retry() {
    let mut unfinished = call("write_file", json!({"path":"forbidden","content":"bad"}));
    unfinished["status"] = json!("incomplete");
    let mut foreign = call("write_file", json!({"path":"forbidden","content":"bad"}));
    foreign["namespace"] = json!("unknown");
    let cases = [
        (
            event(json!({"type":"response.failed","response":{"error":{
            "code":"subscription_sharing_usage_limit_exceeded","message":"App limit"}}})),
            "https://chatgpt.com/settings/usage",
        ),
        (done(json!([unfinished])), "unfinished output"),
        (done(json!([foreign])), "unknown tool namespace"),
        (done(json!([call("list_dir", json!({"path":"."})), call("list_dir", json!({"path":"."}))])), "duplicate call"),
        (format!("{}{}", event(json!({"type":"response.output_item.done","output_index":1,
            "item":call("write_file", json!({"path":"forbidden","content":"bad"}))})), done(json!([]))), "missing output"),
        (format!("{0}{0}{1}", event(json!({"type":"response.output_item.done","output_index":0,
            "item":call("write_file", json!({"path":"forbidden","content":"bad"}))})), done(json!([]))), "duplicate output"),
    ];
    for (payload, expected) in cases {
        let scratch = Scratch::new();
        let (url, requests, job) = server(vec![payload]).await;
        scratch.configure(&url);
        open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Auto).unwrap();
        let (core, mut rx) = Core::new(scratch.data()).unwrap();
        let events = common::drive_turn(&core, &mut rx, "invalid", &scratch.project(), "go").await;
        assert!(
            events.iter().any(|e| matches!(e,AgentEvent::Error {message} if message.contains(expected))),
            "{events:?}"
        );
        assert!(!events.iter().any(|e| matches!(e, AgentEvent::ToolStart { .. })));
        assert_eq!(requests.lock().unwrap().len(), 1);
        job.await.unwrap();
    }
}
