//! Reply limits on a vendor's own Responses (API key) or Messages server, which
//! reports every cut and refuses a limit over the model's. The route is chosen
//! by its configured origin, so these turns resolve `https://api.openai.com/v1`
//! or `https://api.anthropic.com/v1` as usual and only the bytes go to a
//! loopback server (`to_fixture`). Routes keyed to other servers keep their
//! behaviour and are pinned through the public harness in `tests/cut_reasoning.rs`.
use super::*;
use serde_json::json;
use std::cell::RefCell;

#[path = "../tests/common/recording.rs"]
mod recording;

thread_local! { static FIXTURE: RefCell<Option<String>> = const { RefCell::new(None) }; }

/// Sends a turn's requests to the loopback server this test thread set, if any.
pub(super) fn to_fixture(client: &mut ChatClient) {
    if let Some(url) = FIXTURE.with(|f| f.borrow().clone()) { client.base_url = url; }
}

struct Turn { limits: Vec<u64>, notes: Vec<String>, stop: String, billed: u64, refused_raises: usize }

/// What the model sends besides its thought.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// Only the thought, then the answer "finished" once it fits.
    Thought,
    /// First a whole call, reported complete, that spends exactly the limit.
    Call,
    /// Every reply reported complete with its thought and no text or call.
    Silent,
    /// Native Messages only: the thoughts it is cut in fill the window past
    /// the compaction trigger right after a refused raise, and a notes
    /// request is cut at its limit before any text.
    Compacting,
}

/// One turn against a model that needs `needed` output tokens in one reply and
/// keeps needing them after a cut or a refusal (a thought that does not end
/// sooner because the harness asked again). The server refuses a limit over
/// `maximum`, bills what it writes, and reports each cut.
async fn turn(wire: &'static str, window: u64, needed: u64, maximum: u64) -> Turn {
    turn_as(Shape::Thought, wire, window, needed, maximum).await
}

async fn turn_as(shape: Shape, wire: &'static str, window: u64, needed: u64, maximum: u64) -> Turn {
    let field = if wire == "responses" { "max_output_tokens" } else { "max_tokens" };
    let served = std::sync::Arc::new(std::sync::Mutex::new(0u64));
    let (serve, mut turns) = (served.clone(), 0);
    let (url, bodies) = recording::recording_http_with(16, move |body| {
        let sent = serde_json::from_str::<Value>(body).unwrap();
        let limit = sent[field].as_u64().unwrap();
        let notes = sent["tool_choice"] == "none" || sent["tool_choice"]["type"] == "none";
        turns += usize::from(!notes);
        if limit > maximum {
            return (400, "application/json", json!({"error":{"message":"max_output_tokens is too large for this model"}}).to_string());
        }
        let call = shape == Shape::Call && turns == 1;
        let (used, cut) = if call { (limit, false) } else if notes { (limit, true) } else { (needed.min(limit), needed > limit) };
        *serve.lock().unwrap() += used;
        // Input is not reported, so the estimate stays uncalibrated and the window the thoughts fill is the estimator's.
        let usage = json!({"input_tokens":0,"output_tokens":used,"output_tokens_details":{"reasoning_tokens":used-1}});
        if wire == "responses" {
            let output = match () {
                _ if cut || shape == Shape::Silent => json!([]),
                _ if call => json!([{"type":"function_call","call_id":"call_1","name":"list_dir","arguments":"{\"path\":\".\"}"}]),
                _ => json!([{"type":"message","content":[{"type":"output_text","text":"finished"}]}]),
            };
            let event = if cut {
                json!({"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":output,"usage":usage}})
            } else {
                json!({"type":"response.completed","response":{"status":"completed","output":output,"usage":usage}})
            };
            return (200, "text/event-stream", format!("data: {event}\n\n"));
        }
        // In bytes, the estimator's four to a token: about 46,000 tokens sent with the raise the server refuses, and 3,300 more after it.
        let thought = match (shape, turns) { (Shape::Compacting, 2) => "x".repeat(184_000usize.saturating_sub(body.len())), (Shape::Compacting, 4) => "x".repeat(13_200), _ => "partial".into() };
        let thinking = json!({ "type":"thinking","thinking":thought,"signature":"fixture" });
        let (stop, content) = match () {
            _ if cut => ("max_tokens", json!([thinking])),
            _ if call => ("tool_use", json!([{"type":"tool_use","id":"call_1","name":"list_dir","input":{"path":"."}}])),
            _ if shape == Shape::Silent => ("end_turn", json!([thinking])),
            _ => ("end_turn", json!([{ "type":"text","text":"finished" }])),
        };
        (200, "application/json", json!({"type":"message","role":"assistant","stop_reason":stop,"content":content,"usage":{"input_tokens":0,"output_tokens":used}}).to_string())
    }).await;
    let base = if wire == "responses" { "https://api.openai.com/v1" } else { "https://api.anthropic.com/v1" };
    let dir = std::env::temp_dir().join(format!("openmax-limits-{}", uuid::Uuid::new_v4()));
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let (core, mut rx) = Core::new(dir.clone()).unwrap();
    std::fs::write(crate::providers::providers_path(&dir), json!({"providers":{"test":{"base_url":base,"wire_api":wire}}}).to_string()).unwrap();
    crate::trust::grant_trust(&dir, &project, ApprovalMode::Auto).unwrap();
    let settings = Settings { provider: Some("test".into()), model: "thinker".into(), context_tokens: Some(window as usize), ..Settings::default() };
    let cancel = claim_session(&core, "limits", &project, "starting a turn", true).unwrap();
    FIXTURE.with(|f| *f.borrow_mut() = Some(url));
    let stop = run_loop(&core, "limits", &project, "go".into(), settings, cancel).await;
    FIXTURE.with(|f| *f.borrow_mut() = None);
    finish_session_work(&core, "limits", vec![AgentEvent::Done { stop_reason: stop.clone() }]).await;
    let events: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok().map(|e| e.event)).collect();
    std::fs::remove_dir_all(dir).unwrap();
    let limits: Vec<u64> = bodies.lock().unwrap().iter().map(|b| serde_json::from_str::<Value>(b).unwrap()[field].as_u64().unwrap()).collect();
    // What a run can count from its event log alone: output tokens are the sum of
    // the turn's `usage` events, and a request the server refused is a `budget`
    // with no `usage` (nor `retry` or `error`) before the next `budget`.
    let billed = events.iter().filter_map(|e| match e { AgentEvent::Usage { completion_tokens, .. } => Some(*completion_tokens), _ => None }).sum();
    assert_eq!(billed, *served.lock().unwrap());
    let refused_raises = events.iter().enumerate().filter(|(i, e)| matches!(e, AgentEvent::Budget { .. })
        && events[i + 1..].iter().take_while(|e| !matches!(e, AgentEvent::Budget { .. } | AgentEvent::Done { .. }))
            .all(|e| !matches!(e, AgentEvent::Usage { .. } | AgentEvent::Retry { .. } | AgentEvent::Error { .. }))).count();
    assert_eq!(refused_raises, limits.iter().filter(|&&n| n > maximum).count());
    let notes = events.iter().filter_map(|e| match e {
        AgentEvent::HarnessNote { text, .. } if text.starts_with("[reply cut:") => Some(text.clone()), _ => None,
    }).collect();
    let answer = events.iter().rev().find_map(|e| match e { AgentEvent::MessageDone { text } => Some(text.as_str()), _ => None });
    assert_eq!(stop == "stop", answer == Some("finished"), "{events:?}");
    Turn { limits, notes, stop, billed, refused_raises }
}

/// A completed answer that spent 70% or more of the limit, or all of it, on its
/// thought is the answer, not a cut: the server says when it cut one.
#[tokio::test]
async fn a_vendor_servers_completed_answer_stands_after_a_long_thought() {
    for wire in ["responses", "anthropic"] {
        for used in [2869, 4096] {
            let t = turn(wire, 131_072, used, u64::MAX).await;
            assert_eq!((t.limits, t.notes.len(), t.stop.as_str()), (vec![4096], 0, "stop"), "{wire} at {used}");
        }
        // So is a whole call at the limit: it runs, with no room asked for it.
        let t = turn_as(Shape::Call, wire, 131_072, 100, u64::MAX).await;
        assert_eq!((t.limits, t.notes.len(), t.stop.as_str()), (vec![4096, 4096], 0, "stop"), "{wire}");
        // A completed reply with no text and no call is no answer, however long
        // its thought: asked again once. Its next two stop where it did, past
        // half the limit, a cap no server reported, so they are cut (the limit
        // doubles), and a fourth that stops below half of that ends the turn.
        let t = turn_as(Shape::Silent, wire, 131_072, 2869, u64::MAX).await;
        assert_eq!((t.limits, t.notes.len(), t.stop.as_str()), (vec![4096, 4096, 4096, 16384], 2, "empty"), "{wire}");
        // Spending all of the limit on a completed empty reply does not count as
        // a cut by token count alone; only a repeated unreported stop does.
        let t = turn_as(Shape::Silent, wire, 131_072, 4096, u64::MAX).await;
        let expected = if wire == "responses" { (vec![4096, 4096, 4096, 16384], 2, "empty") } else { (vec![4096, 4096], 0, "empty") };
        assert_eq!((t.limits, t.notes.len(), t.stop.as_str()), expected, "{wire}");
    }
}

/// With no known longest reply, reported cuts raise the limit past four times
/// the default, up to a quarter of the window: the cap setup puts on a maximum a
/// catalog reports. A thought that never ends stops there, after four replies at
/// that ceiling, as one at four times the default does elsewhere.
#[tokio::test]
async fn reported_cuts_raise_an_unknown_limit_up_to_a_quarter_of_the_window() {
    for wire in ["responses", "anthropic"] {
        let t = turn(wire, 131_072, 20_000, u64::MAX).await;
        assert_eq!((t.limits.as_slice(), t.stop.as_str()), (&[4096, 4096, 16384, 32768][..], "stop"), "{wire}");
        assert_eq!(t.notes.len(), 3);
        assert!(t.notes[2].contains("32768 tokens"), "{:?}", t.notes);
        let t = turn(wire, 131_072, u64::MAX / 4, u64::MAX).await;
        assert_eq!((t.limits.as_slice(), t.stop.as_str()), (&[4096, 4096, 16384, 32768, 32768, 32768, 32768][..], "length"), "{wire}");
        assert_eq!(t.billed, 155_648);
    }
    // The worst case grows with the window, not with the number of requests.
    let t = turn("responses", 400_000, u64::MAX / 4, u64::MAX).await;
    assert_eq!(t.limits, [4096, 4096, 16384, 32768, 65536, 100_000, 100_000, 100_000, 100_000]);
    assert_eq!((t.billed, t.stop.as_str()), (522_880, "length"));
}

/// A server that refuses a raised limit gets the request again at the last limit
/// it accepted, and the rest of the turn stays there instead of at the reserve.
#[tokio::test]
async fn a_refused_raise_keeps_the_last_accepted_limit_for_the_rest_of_the_turn() {
    for wire in ["responses", "anthropic"] {
        let t = turn(wire, 131_072, u64::MAX / 4, 24_576).await;
        assert_eq!(t.limits, [4096, 4096, 16384, 32768, 16384, 16384, 16384, 16384], "{wire}");
        assert_eq!((t.refused_raises, t.billed, t.stop.as_str()), (1, 90_112, "length"));
        // A window whose quarter is past the model's maximum: refused at 65536 of 100,000.
        let t = turn(wire, 400_000, u64::MAX / 4, 64_000).await;
        assert_eq!(t.limits, [4096, 4096, 16384, 32768, 65536, 32768, 32768, 32768, 32768], "{wire}");
        assert_eq!((t.refused_raises, t.billed, t.stop.as_str()), (1, 188_416, "length"));
        // Refused at the first raise: the last accepted limit is the first ask.
        let t = turn(wire, 131_072, u64::MAX / 4, 6_000).await;
        assert_eq!(t.limits, [4096, 4096, 16384, 4096, 4096, 4096, 4096], "{wire}");
        assert_eq!((t.refused_raises, t.billed, t.stop.as_str()), (1, 24_576, "length"));
    }
    // A notes request in that turn, cut before any text, is not asked again
    // above the limit the server refused (the turn helper counts each refusal).
    // The fifth request is the notes request.
    let t = turn_as(Shape::Compacting, "anthropic", 60_000, u64::MAX / 4, 6_000).await;
    assert_eq!((t.limits.len(), t.limits[0], t.limits[1], &t.limits[3..], t.refused_raises, t.stop.as_str()), (8, 4096, 4096, &[4096, 4096, 4096, 4096, 4096][..], 1, "length"));
    assert!((6_001..=16_384).contains(&t.limits[2]), "{:?}", t.limits);
}

/// The subscription route sends no limit and is not a server this module's rules
/// key on, even when its entry declares `compat.reports_cuts`: its ceiling stays
/// four times the default however large the window.
#[test]
fn the_subscription_route_keeps_four_times_the_default() {
    let dir = std::env::temp_dir().join(format!("openmax-limits-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    crate::chatgpt::tests::seeded(&dir);
    let route = json!({"base_url":crate::chatgpt::API,"wire_api":"responses","chatgpt_account":"main","compat":{"reports_cuts":true},"models":[{"id":"m","context_tokens":400_000}]});
    std::fs::write(crate::providers::providers_path(&dir), json!({"providers":{"sub":route}}).to_string()).unwrap();
    let settings = Settings { provider: Some("sub".into()), model: "m".into(), ..Settings::default() };
    let endpoint = crate::providers::resolve(&settings, &dir).unwrap();
    assert!(endpoint.chatgpt.is_some() && !endpoint.reports_cuts);
    assert_eq!((endpoint.max_tokens, endpoint.reply_ceiling), (4096, 16384));
    std::fs::remove_dir_all(dir).unwrap();
}
