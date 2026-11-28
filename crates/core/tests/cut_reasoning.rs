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

/// One turn against a server of another vendor's on `wire`, for a model that
/// needs `needed` output tokens in one reply. With `reports` the server says
/// when it cut a reply; without, it stops at the limit and calls the reply done.
/// Returns each request's limit, the cut notes and the turn's stop reason.
async fn other_server_turn(wire: &'static str, needed: u64, reports: bool) -> (Vec<u64>, usize, String) {
    server_turn(wire, needed, reports, false).await
}

/// [`other_server_turn`], with the provider entry's `compat.reports_cuts` set
/// to `declared`: the user's word that the server reports every cut.
async fn server_turn(wire: &'static str, needed: u64, reports: bool, declared: bool) -> (Vec<u64>, usize, String) {
    let field = if wire == "responses" { "max_output_tokens" } else { "max_tokens" };
    let (url, bodies) = common::recording_http_with(10, move |body| {
        let limit = serde_json::from_str::<Value>(body).unwrap()[field].as_u64().unwrap();
        let (used, cut) = (needed.min(limit), needed > limit && reports);
        let usage = json!({"input_tokens":10,"output_tokens":used,"output_tokens_details":{"reasoning_tokens":used-1}});
        let reply = match wire {
            "responses" if cut => json!({"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":usage}}),
            "responses" => json!({"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"finished"}]}],"usage":usage}}),
            "anthropic" => json!({"type":"message","role":"assistant","stop_reason":if cut { "max_tokens" } else { "end_turn" },"content":[{"type":"text","text":"finished"}],"usage":{"input_tokens":10,"output_tokens":used}}),
            _ => json!({"choices":[{"finish_reason":if cut { "length" } else { "stop" },"message":{"role":"assistant","content":"finished"}}],
                "usage":{"prompt_tokens":10,"completion_tokens":used,"completion_tokens_details":{"reasoning_tokens":used-1}}}),
        };
        if wire == "responses" { (200, "text/event-stream", format!("data: {reply}\n\n")) } else { (200, "application/json", reply.to_string()) }
    })
    .await;
    let scratch = Scratch::new(&url, wire);
    std::fs::write(scratch.0.join("data/settings.json"), r#"{"provider":"test","model":"thinker","context_tokens":131072}"#).unwrap();
    if declared {
        let providers = json!({"providers":{"test":{"base_url":url,"api_key":"fixture-key","wire_api":wire,"compat":{"reports_cuts":true}}}});
        std::fs::write(scratch.0.join("data/providers.json"), providers.to_string()).unwrap();
    }
    let (core, mut rx) = Core::new(scratch.0.join("data")).unwrap();
    let events = common::drive_turn(&core, &mut rx, "limits", &scratch.0.join("project"), "go").await;
    let limits = bodies.lock().unwrap().iter().map(|b| serde_json::from_str::<Value>(b).unwrap()[field].as_u64().unwrap()).collect();
    let notes = events.iter().filter(|e| matches!(e, AgentEvent::HarnessNote { text, .. } if text.starts_with("[reply cut:"))).count();
    let Some(AgentEvent::Done { stop_reason }) = events.last() else { panic!("{events:?}") };
    (limits, notes, stop_reason.clone())
}

/// A server that is not the vendor's own is not trusted to report every cut,
/// whatever wire it speaks: a reply with no call that spent 70% of the limit or
/// more on its thought is read as cut, and with no known longest reply the
/// limit grows to four times the default and no further. Every row is what
/// the same turn gets from a build before reply cuts were keyed on the server.
#[tokio::test]
async fn another_vendors_server_keeps_the_silent_cut_rules_on_every_wire() {
    let long = |limits: &[u64], notes| (limits.to_vec(), notes, "length".to_string());
    let mut got = Vec::new();
    for wire in ["chat_completions", "responses", "anthropic"] {
        got.push(other_server_turn(wire, 2869, false).await);
        for reports in [false, true] { got.push(other_server_turn(wire, u64::MAX / 4, reports).await); }
    }
    let finished = (vec![4096, 4096, 8192], 2, "stop".to_string());
    let ceiling = long(&[4096, 4096, 8192, 16384, 16384, 16384, 16384], 6);
    // A text the Messages-shaped server cut gets room once, then the rest in parts.
    assert_eq!(got, [
        finished.clone(), ceiling.clone(), ceiling.clone(),
        finished.clone(), ceiling.clone(), ceiling,
        finished, long(&[4096, 16384, 4096, 4096, 4096], 4), long(&[4096, 16384, 4096, 4096, 4096], 4),
    ]);
}

/// A provider entry that declares `compat.reports_cuts`, such as a pass-through
/// proxy in front of a vendor's own server, gets that server's rules on
/// Responses and native Messages: a completed reply is the answer however much
/// of the limit its thought spent, and reported cuts raise an unknown limit up
/// to a quarter of the window. The same server undeclared keeps the silent cut
/// rules. On Chat Completions the declaration changes nothing.
#[tokio::test]
async fn a_server_declared_to_report_cuts_gets_the_vendor_rules_on_responses_and_messages() {
    let mut got = Vec::new();
    for wire in ["responses", "anthropic", "chat_completions"] {
        for declared in [false, true] {
            for needed in [2869, 4096] { got.push(server_turn(wire, needed, true, declared).await); }
            got.push(server_turn(wire, u64::MAX / 4, true, declared).await);
        }
    }
    let row = |limits: &[u64], notes, stop: &str| (limits.to_vec(), notes, stop.to_string());
    let finished = row(&[4096, 4096, 8192], 2, "stop");
    // Undeclared, every wire: a reply that spent 70% of the limit or all of it
    // is read as cut, and the limit grows to four times the default.
    let silent = [finished.clone(), finished.clone(), row(&[4096, 4096, 8192, 16384, 16384, 16384, 16384], 6, "length")];
    // Declared: both completed replies stand, and a thought that never ends is
    // raised to a quarter of the window, as on the vendor's own server.
    let declared = |never_ends| vec![row(&[4096], 0, "stop"), row(&[4096], 0, "stop"), never_ends];
    // The Messages-shaped server cuts inside its text, which gets room once.
    let messages = [finished.clone(), row(&[4096, 16384], 1, "stop"), row(&[4096, 16384, 4096, 4096, 4096], 4, "length")];
    assert_eq!(got, [
        silent.to_vec(), declared(row(&[4096, 4096, 8192, 16384, 32768, 32768, 32768, 32768], 7, "length")),
        messages.to_vec(), declared(row(&[4096, 16384, 32768, 32768, 32768, 32768], 5, "length")),
        silent.to_vec(), silent.to_vec(),
    ].concat());
}
