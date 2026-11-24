mod common;

use open_max_core::config::ApprovalMode;
use open_max_core::sessions;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, ChatMessage, ToolCall, ToolCallFunction};
use serde_json::{json, Value};
use std::path::PathBuf;

const WINDOW: usize = 262_144;
const RESERVE: usize = 4096;
const THOUGHT: &str = "synthetic reasoning";
const FACT: &str = "The middle result verified the receipt code cedar-86.";
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
    archived: Vec<ChatMessage>,
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

    fn diagnostic(&self) -> bool {
        self.events.iter().any(|e| {
            matches!(
                e,
                AgentEvent::Error { .. } | AgentEvent::CompactionSkipped { .. }
            )
        })
    }

    fn notes(&self) -> Vec<&str> {
        self.saved
            .iter()
            .filter_map(|m| m.content.as_deref())
            .filter(|s| s.starts_with("[reply cut:") || s.starts_with("[context note:"))
            .collect()
    }
}

async fn run(replies: Vec<WireReply>, seed: Vec<ChatMessage>) -> Turn {
    run_with_resume(replies, seed, false, None).await
}

async fn run_with_resume(
    replies: Vec<WireReply>,
    seed: Vec<ChatMessage>,
    resume: bool,
    cap: Option<usize>,
) -> Turn {
    let (url, requests) = common::recording_http_script(replies).await;
    let scratch = Scratch::new(&url);
    let project = scratch.0.join("project");
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    core.settings.lock().unwrap().max_agent_tokens = cap;
    let id = sessions::create(&core, project.display().to_string())
        .unwrap()
        .id;
    if !seed.is_empty() {
        assert!(sessions::save_messages(&core, &id, &seed, &mut 0, true));
    }
    let events = common::drive_turn(&core, &mut rx, &id, &project, STEER).await;
    common::close(&core, &id).await;
    drop(core);
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    if resume {
        let result = common::drive_turn(&core, &mut rx, &id, &project, "Review the result.").await;
        assert!(
            matches!(result.last(), Some(AgentEvent::Done { stop_reason }) if stop_reason == "stop")
        );
        core.settings.lock().unwrap().model = "another".into();
        let result =
            common::drive_turn(&core, &mut rx, &id, &project, "Review it on another route.").await;
        assert!(
            matches!(result.last(), Some(AgentEvent::Done { stop_reason }) if stop_reason == "stop")
        );
    }
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
        archived: std::fs::read_to_string(sessions::archive_display(&core, &id))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect(),
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

fn bad_request(message: &str) -> WireReply {
    (400, "application/json", json!({"object":"error", "message":message, "type":"BadRequestError", "param":null, "code":400}).to_string())
}

fn length_refusal(input: usize, output: usize) -> WireReply {
    let total = input + output;
    bad_request(&format!(
        "Requested token count exceeds the model's maximum context length of {WINDOW} tokens. You requested a total of {total} tokens: {input} tokens from the input messages and {output} tokens for the completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit."
    ))
}

fn thought_cut() -> WireReply {
    sse(
        &[json!({"reasoning_content":THOUGHT})],
        "length",
        json!({"prompt_tokens":100,"completion_tokens":RESERVE,"total_tokens":100+RESERVE,"completion_tokens_details":{"reasoning_tokens":RESERVE}}),
    )
}

fn call(id: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        kind: "function".into(),
        function: ToolCallFunction {
            name: "read_file".into(),
            arguments: r#"{"path":"evidence.txt"}"#.into(),
        },
        extra_content: None,
    }
}

fn exchange(id: &str, result: String) -> [ChatMessage; 2] {
    [
        ChatMessage::assistant(None, Some(vec![call(id)])),
        ChatMessage::tool(id, result),
    ]
}

#[tokio::test]
async fn a_counted_window_refusal_below_the_reserve_can_recover() {
    for (refusal, fit) in [
        (length_refusal(WINDOW - 900, RESERVE), false),
        (bad_request("Invalid tool schema."), false),
        (length_refusal(259_000, RESERVE), true),
    ] {
        let mut seed = vec![ChatMessage::system("Rules."), ChatMessage::user(TASK)];
        seed.extend(exchange(
            "saved-read",
            FACT.to_owned() + &"x".repeat(795_000),
        ));
        let reply = sse(
            &[json!({"content":"The saved result is verified."})],
            "stop",
            json!({"prompt_tokens":259_000,"completion_tokens":8,"total_tokens":259_008}),
        );
        let turn = run(vec![refusal, reply], seed).await;
        assert!(turn.requests[0].to_string().contains("saved-read"));
        assert!(turn.notes().is_empty() && turn.archived.is_empty() && turn.outcomes().is_empty());
        if !fit {
            assert_eq!(
                (turn.stop(), turn.limits(), turn.diagnostic()),
                ("error", vec![RESERVE], true)
            );
            continue;
        }
        assert_eq!(
            (
                turn.stop(),
                turn.limits(),
                turn.diagnostic(),
                turn.usage.len()
            ),
            ("stop", vec![RESERVE, 2888], false, 1)
        );
        let mut expected = turn.requests[0].clone();
        expected["max_tokens"] = json!(2888);
        assert_eq!(turn.requests[1], expected);
        assert_eq!(
            turn.saved.last().unwrap().content.as_deref(),
            Some("The saved result is verified.")
        );
    }
}

fn tool_delta(index: usize, id: &str, name: &str, arguments: &str) -> Value {
    json!({"tool_calls":[{"index":index,"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]})
}

#[tokio::test]
async fn a_window_refusal_keeps_the_same_routes_cut_thought() {
    let read = sse(
        &[
            json!({"reasoning_content":"synthetic tool reasoning"}),
            tool_delta(0, "checked", "read_file", r#"{"path":"evidence.txt"}"#),
        ],
        "tool_calls",
        Value::Null,
    );
    for (refusal, kept) in [
        (
            bad_request(
                "content items of type 'thinking' must be followed by content items of type 'text' for messages with empty 'tool_calls': 'messages[2].content[0]'",
            ),
            false,
        ),
        (length_refusal(259_000, RESERVE), true),
    ] {
        let turn = run_with_resume(
            vec![
                read.clone(),
                thought_cut(),
                refusal,
                answer("The primary reply."),
                answer("Resumed."),
                answer("Switched."),
            ],
            vec![],
            true,
            None,
        )
        .await;
        assert_eq!(turn.stop(), "stop");
        assert_eq!(turn.requests.len(), 6);
        assert_eq!(turn.limits()[..3], [RESERVE; 3]);
        assert_eq!(turn.outcomes()[0].0, "checked");
        assert!(turn.outcomes()[0].1);
        let thought = |request: &Value| {
            request["messages"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["reasoning_content"] == THOUGHT)
                .cloned()
        };
        let before = thought(&turn.requests[2]).expect("continuation includes the cut thought");
        assert_eq!(turn.limits()[3], if kept { 2888 } else { RESERVE });
        let after = thought(&turn.requests[3]);
        assert_eq!(
            after,
            thought(&turn.requests[4]),
            "same route after reopening"
        );
        assert!(
            thought(&turn.requests[5]).is_none(),
            "another route gets no thought"
        );
        let saved = turn
            .saved
            .iter()
            .find(|m| m.reasoning_content.as_deref() == Some(THOUGHT))
            .unwrap();
        assert!(
            turn.requests[3]
                .to_string()
                .contains("synthetic tool reasoning")
        );
        assert_eq!(
            (after, saved.reasoning_origin.is_some()),
            (kept.then_some(before), kept),
            "request limits: {:?}",
            turn.limits()
        );
    }
}

#[tokio::test]
async fn a_counted_retry_must_fit_the_remaining_spend_cap() {
    let mut seed = vec![ChatMessage::system("Rules."), ChatMessage::user(TASK)];
    seed.extend(exchange("saved-read", "x".repeat(795_000)));
    let turn = run_with_resume(
        vec![length_refusal(259_000, RESERVE), answer("Not requested.")],
        seed,
        false,
        Some(220_000),
    )
    .await;
    assert_eq!(
        (turn.stop(), turn.limits(), turn.usage.len()),
        ("budget_exhausted", vec![RESERVE], 0)
    );
    assert!(!turn.diagnostic());
}

#[tokio::test]
async fn a_retry_without_usage_still_calibrates_the_next_round() {
    let mut seed = vec![
        ChatMessage::system("Rules."),
        ChatMessage::user(TASK.to_owned() + &"x".repeat(795_000)),
    ];
    seed.extend(exchange("saved-read", FACT.into()));
    let read = sse(
        &[tool_delta(
            0,
            "check",
            "read_file",
            r#"{"path":"evidence.txt"}"#,
        )],
        "tool_calls",
        Value::Null,
    );
    let turn = run(
        vec![
            length_refusal(259_000, RESERVE),
            read,
            answer("Context notes."),
            answer("Verified."),
        ],
        seed,
    )
    .await;
    assert_eq!(
        turn.stop(),
        "stop",
        "{:?}: {:?}",
        turn.limits(),
        turn.events
    );
    assert_eq!(turn.limits(), [RESERVE, 2888, RESERVE, RESERVE]);
    assert_eq!(turn.requests[2]["tool_choice"], "none");
    assert_eq!(turn.outcomes()[0].0, "check");
    assert!(turn.outcomes()[0].1);
    let budgets = turn
        .events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::Budget { used_tokens, .. } => Some(*used_tokens),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(budgets[0] < 220_000 && budgets[1] >= 259_000, "{budgets:?}");
}

#[tokio::test]
async fn gateway_detail_does_not_replace_an_unrelated_error() {
    let mut thought = ChatMessage::assistant(None, None);
    thought.reasoning_content = Some(THOUGHT.into());
    thought.reasoning_origin = Some("another-route".into());
    let raw = json!({"error":{"message":"Provider returned error","metadata":{"raw":"unrelated upstream detail"}}});
    let turn = run(
        vec![(400, "application/json", raw.to_string())],
        vec![
            ChatMessage::system("Rules."),
            ChatMessage::user(TASK),
            thought,
        ],
    )
    .await;
    assert_eq!((turn.stop(), turn.limits()), ("error", vec![RESERVE]));
    assert!(!turn.requests[0].to_string().contains(THOUGHT));
    assert!(turn.events.iter().any(|e| matches!(e, AgentEvent::Error { message } if message.ends_with("Provider returned error"))));
    assert!(turn
        .saved
        .iter()
        .any(|m| m.reasoning_origin.as_deref() == Some("another-route")));
}
