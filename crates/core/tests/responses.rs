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
    json!({"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":text}]})
}
fn call(name: &str, args: Value) -> Value {
    json!({"type":"function_call","status":"completed","call_id":"call-1","name":name,"arguments":args.to_string()})
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
    for (streamed_items, terminal_output) in [(false, true), (true, false), (true, true)] {
        tool_round_and_resume_case(streamed_items, terminal_output).await;
    }
}

async fn tool_round_and_resume_case(streamed_items: bool, terminal_output: bool) {
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
                // A full terminal output is authoritative even if only some items streamed.
                let items = output.as_array().unwrap().iter().enumerate().filter(|(index, _)| !terminal_output || *index > 0).map(|(index, item)| {
                    event(json!({"type":"response.output_item.done","output_index":index,"item":item}))
                }).collect::<String>();
                if terminal_output {
                    value["response"]["output"] = output;
                } else {
                    value["response"].as_object_mut().unwrap().remove("output");
                }
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
    let visible = events.iter().filter_map(|event| match event {
        AgentEvent::Token { text } => Some(text.as_str()),
        _ => None,
    }).collect::<String>();
    assert_eq!(visible, "Done ✓");
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Usage {
        prompt_tokens: 120, completion_tokens: 15, cached_tokens: Some(90)
    })));
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

/// Requests under different `prompt_cache_key`s share no cache entries, so a
/// conversation keeps one key on every request it sends: both of a tool
/// round, a /compact summary, the one a new process sends when it resumes
/// the session, and those of a /fork branch and of a /rewind of that branch,
/// which repeat its history. A second session of the same project, whose
/// prompt opens with the same tokens, gets a key of its own. No key is cut
/// from a session id, and each is 16 hex characters: the field adds 38 bytes
/// to a request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_conversation_sends_one_prompt_cache_key_on_every_request() {
    let scratch = Scratch::new();
    let replies = vec![
        done(json!([call("list_dir", json!({"path":"."}))])),
        done(json!([message("listed")])),
        done(json!([message("other")])),
        done(json!([message("noted")])),
        done(json!([message("summary")])),
        done(json!([message("resumed")])),
        done(json!([message("branched")])),
        done(json!([message("rewound")])),
    ];
    let (url, requests, job) = server(replies).await;
    scratch.configure(&url);
    open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Ask).unwrap();
    let (core, mut rx) = Core::new(scratch.data()).unwrap();
    let project = scratch.project().display().to_string();
    let first = open_max_core::sessions::create(&core, project.clone()).unwrap().id;
    let other = open_max_core::sessions::create(&core, project).unwrap().id;
    common::drive_turn(&core, &mut rx, &first, &scratch.project(), "list").await;
    common::drive_turn(&core, &mut rx, &other, &scratch.project(), "other").await;
    // About 21k estimated tokens (bytes / 4): past the prune target, 70% of
    // the 27,648 the 32k window leaves past the default 4,096 output and the
    // 1,024 reserve, and under that budget. So no turn compacts, and /compact
    // drops the tool round and asks for its summary.
    common::drive_turn(&core, &mut rx, &first, &scratch.project(), &"note ".repeat(17_000)).await;
    open_max_core::agent::compact_session(&core, &first, &scratch.project()).unwrap();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv()).await.unwrap().unwrap();
        match envelope.event {
            AgentEvent::Compacted { compacted_messages, .. } => break assert!(compacted_messages > 0),
            AgentEvent::Error { message } => panic!("/compact failed: {message}"),
            _ => {}
        }
    }
    assert_eq!(requests.lock().unwrap().len(), 5, "/compact sent one summary request");
    common::close(&core, &first).await;
    drop(core);
    let (core, mut rx) = Core::new(scratch.data()).unwrap();
    common::drive_turn(&core, &mut rx, &first, &scratch.project(), "resume").await;
    let branch = open_max_core::sessions::fork(&core, &first, open_max_core::sessions::ForkAt::End).unwrap().id;
    common::close(&core, &first).await;
    common::drive_turn(&core, &mut rx, &branch, &scratch.project(), "branch").await;
    let rewound = open_max_core::sessions::fork(&core, &branch, open_max_core::sessions::ForkAt::Turn(1)).unwrap().id;
    common::close(&core, &branch).await;
    common::drive_turn(&core, &mut rx, &rewound, &scratch.project(), "rewind").await;
    job.await.unwrap();
    let requests = requests.lock().unwrap();
    assert_ne!(requests[4]["input"][0], requests[0]["input"][0], "the summary opens with its own instructions");
    let keys: Vec<&str> = requests.iter()
        .map(|request| request["prompt_cache_key"].as_str().expect("every Responses request carries a prompt_cache_key"))
        .collect();
    assert_eq!(keys.len(), 8);
    for (i, key) in keys.iter().enumerate().filter(|&(i, _)| i != 2) {
        assert_eq!(*key, keys[0], "request {i} left the conversation's key");
    }
    assert_ne!(keys[2], keys[0], "two sessions share one key");
    // A UUID's runs between dashes are at most 12 characters, fewer than a
    // key's 16, so a cut of the id's hex digits shows only without its dashes.
    for id in [&first, &other, &branch, &rewound] {
        let hex = id.replace('-', "");
        assert!(keys.iter().all(|key| !id.contains(key) && !hex.contains(key)), "a key is cut from session id {id}");
    }
    assert!(keys.iter().all(|key| key.len() == 16 && key.bytes().all(|b| b.is_ascii_hexdigit())), "a key is 16 hex characters: {keys:?}");
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
    // Servers that implement Chat Completions vary and some refuse a field
    // they do not know, so the prompt cache key stays off that protocol.
    assert!(!requests.lock().unwrap()[0].contains("prompt_cache_key"));
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transient_stream_failures_retry_frozen_requests_without_running_abandoned_calls() {
    for ending in [
        event(json!({"type":"error","code":"server_error","message":"temporarily busy"})),
        event(json!({"type":"response.failed","response":{"error":{
            "code":"subscription_sharing_usage_unavailable","message":"try again"}}})),
        event(
            json!({"type":"error","code":"subscription_sharing_user_unavailable","message":"try again"}),
        ),
        String::new(),
    ] {
        let scratch = Scratch::new();
        let abandoned = event(json!({"type":"response.output_item.done","output_index":0,
            "item":call("write_file", json!({"path":"forbidden","content":"bad"}))}));
        let (url, requests, job) = server(vec![
            abandoned + &ending,
            done(json!([message("Recovered")])),
        ])
        .await;
        scratch.configure(&url);
        open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Auto)
            .unwrap();
        let (core, mut rx) = Core::new(scratch.data()).unwrap();
        let events = common::drive_turn(&core, &mut rx, "retry", &scratch.project(), "go").await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::Retry { attempt: 2, .. })),
            "{events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::Token { text } if text == "Recovered"))
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolStart { .. } | AgentEvent::Error { .. }))
        );
        assert!(!scratch.project().join("forbidden").exists());
        job.await.unwrap();
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0], requests[1]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transient_failure_after_visible_text_does_not_retry_or_run_calls() {
    let scratch = Scratch::new();
    let payload = event(json!({"type":"response.output_text.delta","delta":"Working"}))
        + &event(json!({"type":"response.output_item.done","output_index":0,
            "item":call("write_file", json!({"path":"forbidden","content":"bad"}))}))
        + &event(json!({"type":"response.failed","response":{"error":{
            "code":"subscription_sharing_usage_unavailable","message":"try again"}}}));
    let (url, requests, job) = server(vec![payload]).await;
    scratch.configure(&url);
    open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Auto)
        .unwrap();
    let (core, mut rx) = Core::new(scratch.data()).unwrap();
    let events = common::drive_turn(&core, &mut rx, "visible", &scratch.project(), "go").await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Token { text } if text == "Working"))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::Error { message } if message.contains("try again")))
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::Retry { .. } | AgentEvent::ToolStart { .. }))
    );
    assert!(!scratch.project().join("forbidden").exists());
    assert_eq!(requests.lock().unwrap().len(), 1);
    job.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn changing_responses_endpoints_withholds_opaque_reasoning_but_keeps_text() {
    let scratch = Scratch::new();
    let (first, _, first_job) = server(vec![done(json!([
        {"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"private-state"},
        message("Previous answer")
    ]))])
    .await;
    let (second, requests, second_job) = server(vec![done(json!([message("Next answer")]))]).await;
    scratch.configure(&first);
    std::fs::write(
        scratch.data().join("providers.json"),
        json!({"providers":{
            "test":{"base_url":first,"wire_api":"responses"},
            "other":{"base_url":second,"wire_api":"responses"}
        }})
        .to_string(),
    )
    .unwrap();
    open_max_core::trust::grant_trust(&scratch.data(), &scratch.project(), ApprovalMode::Auto)
        .unwrap();
    let (core, mut rx) = Core::new(scratch.data()).unwrap();
    common::drive_turn(&core, &mut rx, "switch", &scratch.project(), "first").await;
    core.settings.lock().unwrap().provider = Some("other".into());
    let events = common::drive_turn(&core, &mut rx, "switch", &scratch.project(), "second").await;
    assert!(
        !events.iter().any(|e| matches!(e, AgentEvent::Error { .. })),
        "{events:?}"
    );
    first_job.await.unwrap();
    second_job.await.unwrap();
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let input = requests[0]["input"].as_array().unwrap();
    assert!(
        input
            .iter()
            .any(|item| item["role"] == "assistant" && item["content"] == "Previous answer")
    );
    assert!(!requests[0].to_string().contains("private-state"));
    assert!(!input.iter().any(|item| item["type"] == "reasoning"));
}
