//! A reply the output limit cut while the model was still reasoning, on the
//! native Messages and Responses wires: what the continuation sends, and
//! what an endpoint the reasoning does not belong to receives instead.
#![allow(clippy::disallowed_methods)]
mod common;

use open_max_core::{config::ApprovalMode, state::Core, types::AgentEvent};
use serde_json::{json, Value};
use std::path::PathBuf;

const NOTE: &str = "[reply cut: it reached max_tokens (1024) before any tool call; continue from where it stopped: think briefly, then act.]";

struct Scratch(PathBuf);
impl Scratch {
    fn new(base_url: &str, wire_api: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("openmax-cut-reasoning-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("project")).unwrap();
        std::fs::create_dir_all(dir.join("data")).unwrap();
        let settings = json!({"provider":"test","model":"thinker","context_tokens":32768,"max_tokens":1024});
        std::fs::write(dir.join("data/settings.json"), settings.to_string()).unwrap();
        let providers = json!({"providers":{"test":{"base_url":base_url,"api_key":"fixture-key","wire_api":wire_api}}});
        std::fs::write(dir.join("data/providers.json"), providers.to_string()).unwrap();
        open_max_core::trust::grant_trust(&dir.join("data"), &dir.join("project"), ApprovalMode::Auto).unwrap();
        Self(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Two turns in one session: the first meets `replies` until it ends, the
/// second runs after a switch to another model, which owns none of the
/// first turn's reasoning. Returns the first turn's errors as one list, and
/// the first of them.
async fn two_turns(scratch: &Scratch) -> (String, Option<String>) {
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    let project = scratch.0.join("project");
    let first = common::drive_turn(&core, &mut rx, "cut", &project, "go").await;
    let errors: Vec<_> = first.iter().filter_map(|e| match e { AgentEvent::Error { message } => Some(message.clone()), _ => None }).collect();
    core.settings.lock().unwrap().model = "another".into();
    common::drive_turn(&core, &mut rx, "cut", &project, "and now?").await;
    (format!("{errors:?}"), errors.first().cloned())
}

/// Messages: the cut reply's signed thinking goes back alone as the
/// assistant turn, then the note. After a switch, the assistant turn has
/// nothing it may carry and is left out, and the note still goes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_messages_reply_cut_while_thinking_goes_back_as_its_thinking_alone() {
    let thinking = json!({"type":"thinking","thinking":"half a thought","signature":"fixture-signature"});
    let reply = |content: Value, stop: &str, used: u64| json!({"id":"msg-fixture","type":"message","role":"assistant","model":"thinker",
        "content":content,"stop_reason":stop,"usage":{"input_tokens":10,"output_tokens":used}});
    let (url, bodies) = common::recording_endpoint(vec![
        reply(json!([thinking]), "max_tokens", 1024),
        reply(json!([{"type":"text","text":"done"}]), "end_turn", 5),
        reply(json!([{"type":"text","text":"done again"}]), "end_turn", 5),
    ])
    .await;
    let scratch = Scratch::new(&url, "anthropic");
    let (errors, _) = two_turns(&scratch).await;
    assert_eq!(errors, "[]");
    let bodies: Vec<Value> = bodies.lock().unwrap().iter().map(|b| serde_json::from_str(b).unwrap()).collect();
    assert_eq!(bodies.len(), 3);
    let messages = bodies[1]["messages"].as_array().unwrap();
    let at = messages.iter().position(|m| m["role"] == "assistant").expect("the cut reply goes back");
    assert_eq!(messages[at]["content"], json!([thinking]), "{}", bodies[1]);
    assert_eq!(messages[at + 1]["role"], "user");
    assert!(messages[at + 1].to_string().contains(NOTE), "{}", bodies[1]);

    let switched = bodies[2]["messages"].as_array().unwrap();
    let text = bodies[2].to_string();
    assert!(!text.contains("half a thought") && !text.contains("fixture-signature"), "{text}");
    assert!(text.contains(NOTE), "the note still goes: {text}");
    for message in switched.iter().filter(|m| m["role"] == "assistant") {
        assert!(message["content"].as_array().is_some_and(|blocks| !blocks.is_empty()), "an empty assistant turn: {text}");
    }
    assert_eq!(switched.iter().filter(|m| m["role"] == "assistant").count(), 1, "only the answer: {text}");
}

/// Responses: a reply the limit cut while reasoning arrives either as an
/// incomplete response (`max_output_tokens`) or as a completed one that
/// carries only reasoning and spent the whole limit. Either way it is a cut:
/// its reasoning item goes back alone, then the note, and no assistant
/// message item; after a switch, neither the item nor an empty message goes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_responses_reply_cut_while_reasoning_goes_back_as_its_reasoning_item_alone() {
    let reasoning = json!({"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque-state"});
    let completed = |output: Value, used: u64| json!({"type":"response.completed","response":{"status":"completed",
        "output":output,"usage":{"input_tokens":10,"output_tokens":used}}});
    let answer = |text: &str| json!([{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":text}]}]);
    let incomplete = json!({"type":"response.incomplete","response":{"status":"incomplete",
        "incomplete_details":{"reason":"max_output_tokens"},"output":[reasoning],"usage":{"input_tokens":10,"output_tokens":1024}}});

    for cut in [incomplete, completed(json!([reasoning]), 1024)] {
        let (url, bodies) = common::recording_responses(vec![
            cut,
            completed(answer("done"), 5),
            completed(answer("done again"), 5),
        ])
        .await;
        let scratch = Scratch::new(&url, "responses");
        let (errors, _) = two_turns(&scratch).await;
        assert_eq!(errors, "[]");
        let bodies: Vec<Value> = bodies.lock().unwrap().iter().map(|b| serde_json::from_str(b).unwrap()).collect();
        assert_eq!(bodies.len(), 3);
        let input = bodies[1]["input"].as_array().unwrap();
        let at = input.iter().position(|item| item == &reasoning).expect("the reasoning goes back");
        assert_eq!((&input[at + 1]["role"], &input[at + 1]["content"]), (&json!("user"), &json!(NOTE)), "{}", bodies[1]);
        for body in &bodies[1..] {
            let input = body["input"].as_array().unwrap();
            let empty = |item: &&Value| item["role"] == "assistant" && item["content"].as_str().is_none_or(str::is_empty);
            assert!(!input.iter().any(|item| empty(&item)), "an empty assistant item: {body}");
        }
        let text = bodies[2].to_string();
        assert!(!text.contains("opaque-state") && text.contains(NOTE), "{text}");
    }
}

/// The synthetic model needs `needed` output tokens. Its answer depends on the
/// limit actually received, so a client stuck at its ceiling cannot finish.
async fn limit_turn(wire: &'static str, needed: u64, refuse_raise: bool) -> (Vec<u64>, Vec<String>, String) {
    let field = if wire == "responses" { "max_output_tokens" } else { "max_tokens" };
    let mut refused = false;
    let (url, bodies) = common::recording_http_with(10, move |body| {
        let limit = serde_json::from_str::<Value>(body).unwrap()[field].as_u64().unwrap();
        if refuse_raise && limit > 16_384 {
            refused = true;
            return (400, "application/json", json!({"error":{"message":"unsupported reply limit"}}).to_string());
        }
        let used = if refused { 8 } else { needed.min(limit) };
        let cut = !refused && needed > limit;
        let usage = json!({"input_tokens":10,"output_tokens":used,"output_tokens_details":{"reasoning_tokens":used-1}});
        match wire {
            "responses" => {
                let event = if cut {
                    json!({"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":usage}})
                } else {
                    json!({"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"finished"}]}],"usage":usage}})
                };
                (200, "text/event-stream", format!("data: {event}\n\n"))
            }
            "anthropic" => (200, "application/json", json!({"type":"message","role":"assistant","stop_reason":if cut { "max_tokens" } else { "end_turn" },
                "content":if cut { json!([{ "type":"thinking","thinking":"partial","signature":"fixture" }]) } else { json!([{ "type":"text","text":"finished" }]) },"usage":{"input_tokens":10,"output_tokens":used}}).to_string()),
            "chat_completions" => (200, "application/json", json!({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"finished"}}],
                "usage":{"prompt_tokens":10,"completion_tokens":limit-1,"completion_tokens_details":{"reasoning_tokens":limit-2}}}).to_string()),
            _ => unreachable!(),
        }
    }).await;
    let scratch = Scratch::new(&url, wire);
    std::fs::write(scratch.0.join("data/settings.json"), r#"{"provider":"test","model":"thinker","context_tokens":131072}"#).unwrap();
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    let events = common::drive_turn(&core, &mut rx, "limits", &scratch.0.join("project"), "go").await;
    let limits = bodies.lock().unwrap().iter().map(|b| serde_json::from_str::<Value>(b).unwrap()[field].as_u64().unwrap()).collect();
    let notes = events.iter().filter_map(|e| match e {
        AgentEvent::HarnessNote { text, .. } if text.starts_with("[reply cut:") => Some(text.clone()), _ => None,
    }).collect();
    let Some(AgentEvent::Done { stop_reason }) = events.last() else { panic!("{events:?}") };
    if stop_reason == "stop" {
        let answer = events.iter().rev().find_map(|e| match e { AgentEvent::MessageDone { text } => Some(text.as_str()), _ => None });
        assert_eq!(answer, Some("finished"));
    }
    (limits, notes, stop_reason.clone())
}

#[tokio::test]
async fn a_reported_completion_keeps_the_answer_after_a_long_thought() {
    for wire in ["responses", "anthropic"] {
        for used in [2869, 4096] {
            assert_eq!(limit_turn(wire, used, false).await, (vec![4096], vec![], "stop".into()), "{wire} at {used}");
        }
    }
}

#[tokio::test]
async fn reported_cuts_raise_an_unknown_reply_limit_until_the_answer_fits() {
    for wire in ["responses", "anthropic"] {
        let (limits, notes, stop) = limit_turn(wire, 20_000, false).await;
        assert_eq!((limits.as_slice(), stop.as_str()), (&[4096, 4096, 8192, 16384, 32768][..], "stop"), "{wire}");
        assert_eq!(notes.len(), 4);
        assert!(notes[3].contains("32768 tokens"));
    }
}

#[tokio::test]
async fn a_reported_cut_whose_raise_is_refused_uses_the_reserve() {
    let (limits, _, stop) = limit_turn("responses", 20_000, true).await;
    assert_eq!((limits.as_slice(), stop.as_str()), (&[4096, 4096, 8192, 16384, 32768, 4096][..], "stop"));
}

#[tokio::test]
async fn unreported_chat_cuts_keep_the_near_threshold_and_unknown_ceiling() {
    let (limits, notes, stop) = limit_turn("chat_completions", 20_000, false).await;
    assert_eq!((limits.as_slice(), stop.as_str()), (&[4096, 4096, 8192, 16384, 16384, 16384, 16384][..], "length"));
    assert_eq!(notes.len(), 6);
}
