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
    fn new(url: &str, max_rounds: usize) -> Self {
        let dir = std::env::temp_dir().join(format!("openmax-wire-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("project")).unwrap();
        std::fs::create_dir_all(dir.join("data")).unwrap();
        sessions::write_atomic(
            &dir.join("data/settings.json"),
            json!({
                "provider":"fixture", "model":"fixture", "context_tokens":WINDOW,
                "max_tokens":RESERVE, "max_agent_iterations":max_rounds
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
    run_with_prompt(replies, 8, "Check the result.").await
}

async fn run_with_round_cap(replies: Vec<WireReply>, max_rounds: usize) -> Turn {
    run_with_prompt(replies, max_rounds, "Check the result.").await
}

async fn run_with_prompt(replies: Vec<WireReply>, max_rounds: usize, prompt: &str) -> Turn {
    let (url, requests) = common::recording_http_script(replies).await;
    let scratch = Scratch::new(&url, max_rounds);
    let project = scratch.0.join("project");
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    let id = sessions::create(&core, project.display().to_string())
        .unwrap()
        .id;
    let events = common::drive_turn(&core, &mut rx, &id, &project, prompt).await;
    assert!(!project.join("ghost.txt").exists());
    assert_eq!(
        std::fs::read_to_string(project.join("evidence.txt")).unwrap(),
        FACT
    );
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
                    assert!(output.contains(FACT));
                    output.to_owned()
                } else {
                    format!("Error: {output}")
                };
                assert_eq!(saved, expected, "{id}");
                assert_eq!(
                    replayed["content"].as_str(),
                    Some(expected.as_str()),
                    "{id}"
                );
            }
        }
    }
}

const CALL_AS_TEXT: &str =
    r#"{"name":"write_file","arguments":{"path":"ghost.txt","content":"Do not write this."}}"#;
const MALFORMED: &str = " <tool_call>\n{\"name\":\"edit_file\",\"arguments\":{\"path\":\"evidence.txt\",\"edits\":[{\"old_string\":\"middle\",\"new_string\":\"first\"}}}\n</tool_call>\n";
const PROSE_MALFORMED: &str = "First, write the line below:\n<tool_call>\n{\"name\":\"write_file\",\"arguments\":{\"path\":\"ghost.txt\",\"content\":\"one\ntwo\"}}\n</tool_call>";
const MISSING_DELIMITER: &str = r#"<tool_call>{"name":"write_file","arguments":{"path":"ghost.txt" "content":"text"}}</tool_call>"#;
const OPEN_ARRAY: &str = "<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"paths\":[\"a.txt\"\n</tool_call>";
const BROKEN_FIRST_ARGUMENT: &str = r#"<tool_call>
{"name":"read_file","arguments":{"offset":  first","path":"evidence.txt"}}
</tool_call>"#;
const TEXT_CALL_NOTE: &str = "[tool call written as text: nothing ran. Only call the tool through the API with valid JSON arguments if the user's task requires running it. Otherwise return the requested answer without running anything.]";
const BAD_JSON_NOTE: &str = "[tool call written as text: nothing ran. The arguments were not valid JSON. Only call the tool through the API with valid JSON arguments if the user's task requires running it. Otherwise return the requested answer without running anything.]";

fn real_read() -> WireReply {
    sse(
        &[tool_delta(
            0,
            "actual",
            "read_file",
            r#"{"path":"evidence.txt"}"#,
        )],
        "tool_calls",
        Value::Null,
    )
}

fn call_text_notes(turn: &Turn) -> Vec<&str> {
    turn.events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::HarnessNote { call_id, text }
                if text.starts_with("[tool call written as text:") =>
            {
                assert!(call_id.is_empty());
                Some(text.as_str())
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn written_calls_are_replayed_as_text_then_retried_through_the_api() {
    for (text, note) in [
        (BROKEN_FIRST_ARGUMENT.to_string(), BAD_JSON_NOTE),
        (MALFORMED.to_string(), BAD_JSON_NOTE),
        (PROSE_MALFORMED.to_string(), BAD_JSON_NOTE),
        (MISSING_DELIMITER.to_string(), BAD_JSON_NOTE),
        (OPEN_ARRAY.to_string(), BAD_JSON_NOTE),
        (format!("First, inspect the files:\n{OPEN_ARRAY}"), BAD_JSON_NOTE),
        (format!("<tool_call>{CALL_AS_TEXT}</tool_call>"), TEXT_CALL_NOTE),
        (CALL_AS_TEXT.to_string(), TEXT_CALL_NOTE),
        (format!("```json\n{CALL_AS_TEXT}\n```"), TEXT_CALL_NOTE),
    ] {
        let turn = run(vec![answer(&text), real_read(), answer("Verified.")]).await;
        assert_eq!(turn.stop(), "stop", "{text}");
        assert_eq!(turn.requests.len(), 3, "{text}");
        let starts = turn.events.iter().filter_map(|e| match e {
            AgentEvent::ToolStart { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        }).collect::<Vec<_>>();
        assert_eq!(starts, ["actual"], "{text}");
        let outcomes = turn.outcomes();
        assert!(matches!(outcomes.as_slice(), [("actual", true, result)] if result.contains(FACT)), "{outcomes:?}");
        assert_eq!(call_text_notes(&turn), [note]);
        let first = turn.requests[0]["messages"].as_array().unwrap();
        let second = turn.requests[1]["messages"].as_array().unwrap();
        let third = turn.requests[2]["messages"].as_array().unwrap();
        assert_eq!(&second[..first.len()], first);
        assert_eq!(&second[first.len()..], &[
            json!({"role":"assistant","content":text}),
            json!({"role":"user","content":note}),
        ]);
        assert_eq!(&third[..second.len()], second);
        let saved = &turn.saved[first.len()..];
        assert_eq!(saved[0].content.as_deref(), Some(text.as_str()));
        assert_eq!(saved[0].role, "assistant");
        assert!(saved[0].tool_calls.is_none());
        assert_eq!(saved[1].content.as_deref(), Some(note));
        assert!(saved[1].harness_note);
        assert_eq!(saved.last().unwrap().content.as_deref(), Some("Verified."));
    }
}

#[tokio::test]
async fn written_calls_get_two_recoveries_in_a_row_within_the_round_cap() {
    let diagnostic = "The reply wrote a tool call as text. Nothing from it ran.";
    for (cap, count) in [(8, 3), (2, 2)] {
        let turn = run_with_round_cap(vec![answer(CALL_AS_TEXT); count], cap).await;
        assert_eq!(turn.stop(), "stop");
        assert_eq!(turn.requests.len(), count);
        assert!(turn.outcomes().is_empty());
        assert_eq!(call_text_notes(&turn), vec![TEXT_CALL_NOTE; count - 1]);
        assert!(
            matches!(&turn.events[turn.events.len() - 2], AgentEvent::Error { message } if message == diagnostic)
        );
        let last = turn.requests.last().unwrap()["messages"]
            .as_array()
            .unwrap();
        assert_eq!(last[last.len() - 2]["content"], CALL_AS_TEXT);
        assert_eq!(last.last().unwrap()["content"], TEXT_CALL_NOTE);
        assert_eq!(
            turn.saved.last().unwrap().content.as_deref(),
            Some(CALL_AS_TEXT)
        );
        assert_eq!(
            turn.saved
                .iter()
                .filter(|m| m.content.as_deref() == Some(TEXT_CALL_NOTE) && m.harness_note)
                .count(),
            count - 1
        );
    }
}

#[tokio::test]
async fn api_calls_reset_the_written_call_streak() {
    let turn = run(vec![
        answer(CALL_AS_TEXT),
        answer(CALL_AS_TEXT),
        real_read(),
        answer(CALL_AS_TEXT),
        answer(CALL_AS_TEXT),
        answer("Verified."),
    ])
    .await;
    assert_eq!(turn.stop(), "stop");
    assert_eq!(turn.requests.len(), 6);
    assert_eq!(call_text_notes(&turn), [TEXT_CALL_NOTE; 4]);
    assert_eq!(turn.outcomes().len(), 1);
    assert_eq!(
        turn.saved.last().unwrap().content.as_deref(),
        Some("Verified.")
    );
}

#[tokio::test]
async fn answers_with_tool_names_or_examples_stay_answers() {
    for text in [
        "The bash and read_file tools are available.".to_string(),
        r#"{"name":"example","arguments":{"path":"example.txt"}}"#.to_string(),
        format!("A call looks like this:\n```json\n{CALL_AS_TEXT}\n```\nUse it to write a file."),
        format!("A call looks like this:\n```json\n{CALL_AS_TEXT}\n```"),
        "The API accepts a `name` string and an `arguments` object in each request.".to_string(),
        "To inspect a file, call `read_file` with its path. Then use its result to decide what to edit.".to_string(),
        format!("Example: `<tool_call>{CALL_AS_TEXT}</tool_call>`"),
        format!("<tool_call>\n{CALL_AS_TEXT}"),
        format!("{CALL_AS_TEXT}\nThis example is not a request to run it."),
    ] {
        let turn = run(vec![answer(&text)]).await;
        assert_eq!(turn.stop(), "stop", "{text}");
        assert_eq!(turn.requests.len(), 1, "{text}");
        assert!(call_text_notes(&turn).is_empty(), "{text}");
        assert!(turn.outcomes().is_empty());
        assert_eq!(turn.saved.len(), turn.requests[0]["messages"].as_array().unwrap().len() + 1);
        let saved = turn.saved.last().unwrap();
        assert_eq!(saved.role, "assistant");
        assert_eq!(saved.content.as_deref(), Some(text.as_str()));
    }
}

#[tokio::test]
async fn quoting_a_call_does_not_ask_for_execution() {
    let fenced = format!("```json\n{CALL_AS_TEXT}\n```");
    let tagged = format!("<tool_call>\n{CALL_AS_TEXT}\n</tool_call>");
    for text in [fenced, tagged, CALL_AS_TEXT.to_string()] {
        let turn = run_with_prompt(vec![answer(&text), answer("Here is the requested example.")], 8,
            "Quote a tool-call example without executing it.").await;
        assert_eq!(turn.requests.len(), 2);
        let messages = turn.requests[1]["messages"].as_array().unwrap();
        assert_eq!(messages[messages.len() - 2], json!({"role":"assistant","content":text}));
        assert_eq!(messages.last().unwrap(), &json!({"role":"user","content":TEXT_CALL_NOTE}));
        assert_eq!(call_text_notes(&turn), [TEXT_CALL_NOTE]);
        assert!(turn.saved.iter().any(|m| m.harness_note && m.content.as_deref() == Some(TEXT_CALL_NOTE)));
        assert!(turn.outcomes().is_empty());
        assert_eq!(turn.stop(), "stop");
    }
    let tutorial = format!("The API takes a name and arguments. For example:\n<tool_call>\n{CALL_AS_TEXT}\n</tool_call>");
    let turn = run_with_prompt(vec![answer(&tutorial)], 8, "Write a tutorial with a tagged example.").await;
    assert_eq!(turn.requests.len(), 1);
    assert!(call_text_notes(&turn).is_empty());
    assert!(turn.outcomes().is_empty());
    assert_eq!(turn.saved.last().unwrap().content.as_deref(), Some(tutorial.as_str()));
    assert_eq!(turn.stop(), "stop");
}

#[tokio::test]
async fn cut_text_and_mixed_api_replies_keep_their_existing_paths() {
    let cut = sse(&[json!({"content": MALFORMED})], "length", Value::Null);
    let turn = run(vec![cut, answer("Continued.")]).await;
    assert_eq!(turn.stop(), "stop");
    assert!(call_text_notes(&turn).is_empty());
    assert!(turn.requests[1]["messages"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .starts_with("[reply cut:"));
    assert!(turn
        .saved
        .iter()
        .any(|m| m.content.as_deref() == Some(MALFORMED)));

    let mixed = sse(
        &[
            json!({"content":CALL_AS_TEXT}),
            tool_delta(0, "actual", "read_file", r#"{"path":"evidence.txt"}"#),
        ],
        "tool_calls",
        Value::Null,
    );
    let turn = run(vec![mixed, answer("Verified.")]).await;
    assert_eq!(turn.stop(), "stop");
    assert!(call_text_notes(&turn).is_empty());
    assert_eq!(turn.outcomes().len(), 1);
    assert_eq!(turn.outcomes()[0].0, "actual");
}
