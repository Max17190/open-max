mod common;

use open_max_core::client::ChatClient;
use open_max_core::config::ApprovalMode;
use open_max_core::sessions;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, ChatMessage};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc};

const WINDOW: usize = 262_144;
const RESERVE: usize = 4096;
const THOUGHT: &str = "synthetic reasoning";
const TASK: &str = "Verify the results.";
const STEER: &str = "Report the verified receipt code.";

type WireReply = (u16, &'static str, String);

struct Scratch(PathBuf);
impl Scratch {
    fn new(url: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("openmax-wire-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("project")).unwrap();
        std::fs::create_dir_all(dir.join("data")).unwrap();
        sessions::write_atomic(
            &dir.join("data/settings.json"),
            json!({
                "provider":"fixture", "model":"fixture", "context_tokens":WINDOW,
                "max_tokens":RESERVE, "max_agent_iterations":8
            })
            .to_string(),
        )
        .unwrap();
        sessions::write_atomic(
            &dir.join("data/providers.json"),
            json!({
                "providers":{"fixture":{"base_url":url,"api_key":"fixture-key"}}
            })
            .to_string(),
        )
        .unwrap();
        open_max_core::trust::grant_trust(
            &dir.join("data"),
            &dir.join("project"),
            ApprovalMode::Auto,
        )
        .unwrap();
        Self(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Turn {
    requests: Vec<Value>,
    events: Vec<AgentEvent>,
    saved: Vec<ChatMessage>,
    usage: Vec<sessions::TokenUsage>,
}

impl Turn {
    fn limits(&self) -> Vec<usize> {
        self.requests
            .iter()
            .map(|r| r["max_tokens"].as_u64().unwrap() as usize)
            .collect()
    }

    fn stop(&self) -> &str {
        match self.events.last().unwrap() {
            AgentEvent::Done { stop_reason } => stop_reason,
            event => panic!("ended with {event:?}"),
        }
    }

    fn outcomes(&self) -> Vec<(&str, bool, &str)> {
        self.events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolEnd {
                    call_id,
                    ok,
                    output,
                    ..
                } => Some((call_id.as_str(), *ok, output.as_str())),
                _ => None,
            })
            .collect()
    }

    fn notes(&self) -> Vec<&str> {
        self.saved
            .iter()
            .filter_map(|m| m.content.as_deref())
            .filter(|s| s.starts_with("[reply cut:") || s.starts_with("[context note:"))
            .collect()
    }
}

async fn run(replies: Vec<WireReply>) -> Turn {
    let (url, requests) = common::recording_http_script(replies).await;
    let scratch = Scratch::new(&url);
    let project = scratch.0.join("project");
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    let id = sessions::create(&core, project.display().to_string())
        .unwrap()
        .id;
    let events = common::drive_turn(&core, &mut rx, &id, &project, STEER).await;
    common::close(&core, &id).await;
    drop(core);
    let (core, _) = Core::new(scratch.0.join("data")).unwrap();
    let saved = sessions::load_messages(&core, &id).unwrap().unwrap();
    let requests = requests
        .lock()
        .unwrap()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect();
    Turn {
        requests,
        events,
        saved,
        usage: sessions::load_usage(&core, &id),
    }
}

fn sse(deltas: &[Value], finish: &str, usage: Value) -> WireReply {
    let mut body = String::new();
    for delta in deltas {
        body.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":delta,"finish_reason":null}]})
        ));
    }
    body.push_str(&format!(
        "data: {}\n\n",
        json!({"choices":[{"delta":{},"finish_reason":finish}]})
    ));
    if !usage.is_null() {
        body.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[],"usage":usage})
        ));
    }
    body.push_str("data: [DONE]\n\n");
    (200, "text/event-stream", body)
}

fn answer(text: &str) -> WireReply {
    sse(&[json!({"content":text})], "stop", Value::Null)
}

#[tokio::test]
async fn measured_reasoning_keeps_a_complete_answer_complete() {
    let mut failures = Vec::new();
    for (usage, reasoning, expected, requests, billed) in [
        (
            json!({"completion_tokens_details":{"reasoning_tokens":0}}),
            false,
            Some(0),
            1,
            3000,
        ),
        (json!({}), false, None, 2, 3000),
        (json!({"reasoning_tokens":0}), false, Some(0), 1, 3000),
        (json!({"reasoning_tokens":2000}), true, Some(2000), 1, 3000),
        (json!({"reasoning_tokens":2000}), true, Some(2000), 1, 5000),
        (
            json!({"reasoning_tokens":50,"completion_tokens_details":{"reasoning_tokens":2000}}),
            true,
            Some(2000),
            1,
            5000,
        ),
    ] {
        let mut usage = usage;
        usage["prompt_tokens"] = json!(100);
        usage["completion_tokens"] = json!(3000);
        usage["total_tokens"] = json!(100 + billed);
        let mut deltas = Vec::new();
        if reasoning {
            deltas.push(json!({"reasoning_content":THOUGHT}));
        }
        deltas.push(json!({"content":"The report is complete."}));
        let reply = sse(&deltas, "stop", usage.clone());
        let turn = run(vec![reply.clone(), answer("Unneeded continuation.")]).await;
        let (url, _) = common::recording_http_script(vec![reply]).await;
        let client = ChatClient::new(url, None, "fixture".into(), None, RESERVE);
        let result = client
            .stream_chat(
                &[ChatMessage::user(TASK)],
                RESERVE,
                "[]",
                Arc::default(),
                |_| {},
            )
            .await
            .unwrap();
        let counts = result.usage.unwrap();
        assert_eq!(turn.stop(), "stop");
        assert!(turn.outcomes().is_empty());
        assert_eq!(turn.usage[0].completion_tokens, billed);
        assert_eq!(counts.output_tokens, billed);
        let actual = (turn.limits(), turn.notes().len(), counts.reasoning_tokens);
        if actual != (vec![RESERVE; requests], requests - 1, expected) {
            failures.push((usage, actual));
        }
        assert!(turn
            .saved
            .iter()
            .any(|m| m.content.as_deref() == Some("The report is complete.")));
    }
    assert!(
        failures.is_empty(),
        "usage, asks, cut notes, measured reasoning: {failures:?}"
    );
}
