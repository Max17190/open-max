mod common;

use open_max_core::config::ApprovalMode;
use open_max_core::sessions;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, ChatMessage};
use serde_json::{json, Value};
use std::path::PathBuf;

const WINDOW: usize = 262_144;
const RESERVE: usize = 4096;
const FACT: &str = "The middle result verified the receipt code cedar-86.";

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
        sessions::write_atomic(&dir.join("project/evidence.txt"), FACT).unwrap();
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
}

impl Turn {
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
}

async fn run(replies: Vec<WireReply>) -> Turn {
    let (url, requests) = common::recording_http_script(replies).await;
    let scratch = Scratch::new(&url);
    let project = scratch.0.join("project");
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    let id = sessions::create(&core, project.display().to_string())
        .unwrap()
        .id;
    let events = common::drive_turn(&core, &mut rx, &id, &project, "Check the result.").await;
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

fn tool_delta(index: usize, id: &str, name: &str, arguments: &str) -> Value {
    json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]})
}

#[tokio::test]
async fn indexed_calls_keep_their_results_when_an_earlier_argument_is_incomplete() {
    let args = r#"{"path":"evidence.txt"}"#;
    for (reply, outcomes) in [
        (
            answer("Use the read_file tool to inspect the file."),
            vec![],
        ),
        (
            sse(
                &[tool_delta(0, "first", "unknown_tool", args)],
                "tool_calls",
                Value::Null,
            ),
            vec![("first", false)],
        ),
        (
            sse(
                &[
                    tool_delta(0, "first", "read_file", args),
                    tool_delta(1, "second", "read_file", args),
                ],
                "tool_calls",
                Value::Null,
            ),
            vec![("first", true), ("second", true)],
        ),
        (
            sse(
                &[
                    tool_delta(0, "first", "read_file", r#"{"filter":{"page":1}"#),
                    tool_delta(1, "second", "read_file", args),
                ],
                "tool_calls",
                Value::Null,
            ),
            vec![("first", false), ("second", true)],
        ),
    ] {
        let turn = run(vec![reply, answer("The primary reply.")]).await;
        assert_eq!(turn.stop(), "stop");
        let actual = turn.outcomes();
        assert_eq!(
            actual
                .iter()
                .map(|(id, ok, _)| (*id, *ok))
                .collect::<Vec<_>>(),
            outcomes
        );
        if outcomes.is_empty() {
            assert_eq!(turn.requests.len(), 1);
        } else {
            let messages = turn.requests[1]["messages"].as_array().unwrap();
            let ids: Vec<_> = messages
                .iter()
                .filter_map(|m| m["tool_call_id"].as_str())
                .collect();
            assert_eq!(ids, outcomes.iter().map(|(id, _)| *id).collect::<Vec<_>>());
            if outcomes == [("first", false), ("second", true)] {
                let calls = messages
                    .iter()
                    .find_map(|m| m["tool_calls"].as_array())
                    .unwrap();
                assert_eq!(calls[0]["function"]["arguments"], "{}");
                assert_eq!(calls[1]["function"]["arguments"], args);
                assert!(
                    actual[0].2.contains("invalid JSON") && actual[0].2.contains("did not run")
                );
            }
            for (id, ok, output) in actual {
                let saved = turn
                    .saved
                    .iter()
                    .find(|m| m.tool_call_id.as_deref() == Some(id))
                    .unwrap();
                let replayed = messages.iter().find(|m| m["tool_call_id"] == id).unwrap();
                let saved = saved.content.as_deref().unwrap();
                let expected = if ok {
                    FACT
                } else if output.contains("invalid JSON") {
                    "invalid JSON"
                } else {
                    "Error"
                };
                if ok {
                    assert!(output.contains(FACT));
                }
                assert!(saved.contains(expected), "{id}: {saved}");
                assert_eq!(replayed["content"].as_str(), Some(saved), "{id}");
            }
        }
    }
}
