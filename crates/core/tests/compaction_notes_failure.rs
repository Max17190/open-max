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
}

async fn run(replies: Vec<WireReply>, seed: Vec<ChatMessage>) -> Turn {
    run_with(json!({}), replies, seed).await
}

/// `run` with `extra` settings merged over the scratch's own.
async fn run_with(extra: Value, replies: Vec<WireReply>, seed: Vec<ChatMessage>) -> Turn {
    let (url, requests) = common::recording_http_script(replies).await;
    let scratch = Scratch::new(&url);
    let settings_path = scratch.0.join("data/settings.json");
    let mut settings: Value = serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
    settings.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    sessions::write_atomic(&settings_path, settings.to_string()).unwrap();
    let project = scratch.0.join("project");
    // Five reads of about 1,000 estimated tokens each: together past a
    // quarter of a 20,000-token trigger, the newest two well inside what a
    // cut at that trigger keeps.
    for part in 0..5 {
        let lines: String = (0..55).map(|i| format!("{part}{i:03} {}\n", "b".repeat(66))).collect();
        sessions::write_atomic(&project.join(format!("part-{part}.txt")), lines).unwrap();
    }
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    let id = sessions::create(&core, project.display().to_string())
        .unwrap()
        .id;
    if !seed.is_empty() {
        assert!(sessions::save_messages(&core, &id, &seed, &mut 0, true));
    }
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

fn thought_cut(limit: usize) -> WireReply {
    sse(
        &[json!({"reasoning_content":THOUGHT})],
        "length",
        json!({"prompt_tokens":100,"completion_tokens":limit,"total_tokens":100+limit,"completion_tokens_details":{"reasoning_tokens":limit}}),
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

fn history(tokens: usize) -> Vec<ChatMessage> {
    let mut seed = vec![ChatMessage::system("Rules."), ChatMessage::user(TASK)];
    for i in 0..16 {
        seed.push(ChatMessage::user(format!("Inspect page {i}.")));
        let value = if i == 4 { FACT } else { "Checked the page." };
        seed.extend(exchange(
            &format!("read-{i}"),
            value.to_owned() + &"x".repeat((tokens - 4500) * 4 / 16),
        ));
        seed.push(ChatMessage::assistant(Some("Page checked.".into()), None));
    }
    seed
}

#[tokio::test]
async fn notes_failure_preserves_history_when_the_window_does_not_force_a_cut() {
    let summary = sse(
        &[json!({"content":FACT})],
        "stop",
        json!({"prompt_tokens":200_000,"completion_tokens":16}),
    );
    let refusal = length_refusal(260_000, RESERVE);
    for (tokens, mut notes, expected_skip, expected_output) in [
        (215_000, vec![summary], None, vec![16]),
        // The window forces this cut: it lands on the digest, and says why it had no notes.
        (260_000, vec![refusal.clone()], Some("the notes request failed: backend returned 400"), vec![]),
        (
            215_000,
            vec![refusal],
            Some("the notes request failed: backend returned 400"),
            vec![],
        ),
        (
            215_000,
            vec![thought_cut(RESERVE), thought_cut(2 * RESERVE)],
            Some("the notes reply had no text (finish_reason length)"),
            vec![4096, 8192],
        ),
    ] {
        let notes_count = notes.len();
        notes.push(answer("The primary reply."));
        let turn = run(notes, history(tokens)).await;
        assert_eq!(turn.stop(), "stop");
        assert!(!turn
            .events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolStart { .. })));
        assert_eq!(turn.requests.len(), notes_count + 1);
        assert_eq!(turn.requests[0]["tool_choice"], "auto", "the notes request keeps the turn's tool choice");
        assert_eq!(turn.requests.last().unwrap()["tool_choice"], "auto");
        assert!(turn.requests[0].to_string().contains(FACT));
        let body = turn.requests.last().unwrap().to_string();
        assert!(body.contains(TASK) && body.contains(STEER) && body.contains("read-15"));
        let saved_fact = turn
            .saved
            .iter()
            .any(|m| m.content.as_deref().is_some_and(|s| s.contains(FACT)));
        let archived_fact = turn
            .archived
            .iter()
            .any(|m| m.content.as_deref().is_some_and(|s| s.contains(FACT)));
        let skipped: Vec<_> = turn
            .events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::CompactionSkipped { reason } => Some(reason.as_str()),
                AgentEvent::Error { message } => panic!("unexpected turn error: {message}"),
                _ => None,
            })
            .collect();
        assert_eq!(skipped.len(), usize::from(expected_skip.is_some()));
        if let Some(reason) = expected_skip {
            assert!(skipped[0].starts_with(reason), "{skipped:?}");
        }
        assert_eq!(
            turn.usage
                .iter()
                .map(|u| u.completion_tokens)
                .collect::<Vec<_>>(),
            expected_output
        );
        assert_eq!(archived_fact, expected_skip.is_none() || tokens == 260_000);
        assert_eq!(compacted(&turn).len(), usize::from(archived_fact), "each cut is reported");
        assert_eq!(saved_fact, tokens < 260_000);
        assert_eq!(body.contains(FACT), saved_fact);
        assert_eq!(
            turn.limits(),
            if notes_count == 2 {
                vec![RESERVE, 2 * RESERVE, RESERVE]
            } else {
                vec![RESERVE; 2]
            }
        );
    }
}

fn read_call(id: &str, path: &str) -> WireReply {
    sse(
        &[json!({"tool_calls":[{"index":0,"id":id,"type":"function","function":{"name":"read_file","arguments":json!({"path":path}).to_string()}}]})],
        "tool_calls",
        Value::Null,
    )
}

/// A soft cut whose notes request fails is asked again once the history has
/// grown a quarter of the trigger past the size it failed at, rather than
/// waiting for the window for the rest of the turn. The reads keep the newest
/// messages small enough for the configured trigger to hold, so the second
/// ask comes from the quarter and not from the window.
#[tokio::test]
async fn a_failed_notes_request_is_asked_again_after_a_quarter_of_the_trigger() {
    let mut replies = vec![bad_request("the notes request is refused once")];
    replies.extend((0..5).map(|part| read_call(&format!("part-{part}"), &format!("part-{part}.txt"))));
    replies.extend([answer("State: halfway."), answer("The primary reply.")]);
    let turn = run_with(json!({"compaction_tokens": 20_000}), replies, history(25_000)).await;
    assert_eq!(turn.stop(), "stop", "{:?}", turn.events);
    assert_eq!(asks(&turn), [0, 6], "asked at the trigger, then again once five reads grew it a quarter");
    assert_eq!(turn.requests.len(), 8);
    let skipped = turn.events.iter().filter(|e| matches!(e, AgentEvent::CompactionSkipped { .. })).count();
    assert_eq!(skipped, 1);
    let last = turn.requests.last().unwrap().to_string();
    assert!(last.contains("Summary: State: halfway.") && last.contains(TASK));
    assert!(turn.archived.iter().any(|m| m.content.as_deref().is_some_and(|c| c.contains(FACT))));
    assert_eq!(compacted(&turn).len(), 1, "the cut is reported");
}

/// The indexes of the requests that ask for notes.
fn asks(turn: &Turn) -> Vec<usize> {
    turn.requests
        .iter()
        .enumerate()
        .filter(|(_, r)| r["messages"].as_array().unwrap().last().unwrap()["content"].as_str().is_some_and(|c| c.starts_with("[context note: everything above")))
        .map(|(i, _)| i)
        .collect()
}

/// Each automatic cut the turn reported: the size before it, the size
/// after it and the messages it archived.
fn compacted(turn: &Turn) -> Vec<(usize, usize, usize)> {
    turn.events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::Compacted { tokens_before, tokens_after, compacted_messages } => Some((*tokens_before, *tokens_after, *compacted_messages)),
            _ => None,
        })
        .collect()
}

/// A notes reply that makes a call, with text beside it or not, runs no tool
/// and gives no notes: the same request is sent once more, its ask saying no
/// tool runs, and the notes that then arrive are the cut's. When that reply
/// makes a call too, the cut lands on the digest at once, and the event log
/// says why and what it cut, so no paid request goes unaccounted for.
#[tokio::test]
async fn a_notes_reply_that_makes_a_call_is_asked_once_more_and_reported() {
    let preamble = || sse(
        &[json!({"content":"Let me re-read the file first."}), json!({"tool_calls":[{"index":0,"id":"notes-call","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"evidence.txt\"}"}}]})],
        "tool_calls",
        Value::Null,
    );
    for (second, notes) in [(answer("State: halfway."), true), (preamble(), false)] {
        let turn = run(vec![preamble(), second, answer("The primary reply.")], history(215_000)).await;
        assert_eq!(turn.stop(), "stop", "{:?}", turn.events);
        assert!(!turn.events.iter().any(|e| matches!(e, AgentEvent::ToolStart { .. })), "the call in a notes reply never runs");
        assert_eq!(asks(&turn), [0, 1], "asked twice, then the turn's own request");
        assert_eq!(turn.requests.len(), 3);
        let (asked, again) = (&turn.requests[0]["messages"], &turn.requests[1]["messages"]);
        let last = asked.as_array().unwrap().len() - 1;
        assert_eq!(asked.as_array().unwrap()[..last], again.as_array().unwrap()[..last], "the same request up to the ask");
        assert!(again[last]["content"].as_str().unwrap().ends_with("Reply with the notes alone, in plain text.]"));
        let body = turn.requests[2].to_string();
        assert_eq!(body.contains("Summary: State: halfway."), notes);
        assert!(!body.contains("re-read") && body.contains(TASK) && body.contains(STEER));
        let reasons: Vec<&str> = turn
            .events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::CompactionSkipped { reason } => Some(reason.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasons, if notes { vec![] } else { vec!["the notes reply made a call (finish_reason tool_calls)"] });
        let cuts = compacted(&turn);
        assert!(matches!(cuts[..], [(before, after, dropped)] if before > 200_000 && after < 30_000 && dropped > 0), "{cuts:?}");
    }
}
