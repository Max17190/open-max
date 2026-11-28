//! A host that refuses a request for an image a tool just returned. Its own
//! test binary: turning image input on changes the read_file schema every
//! registry built in the process reads, which other tests would see.
#![allow(clippy::disallowed_methods)]
mod common;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use open_max_core::{config::ApprovalMode, state::Core, types::AgentEvent};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The error body of a refusal with `status`, whose code is the status's.
fn refusal(status: &str) -> String {
    format!(r#"{{"error":{{"message":"Provider returned error","code":{}}}}}"#, &status[..3])
}

/// Answers a request that carries an image part with HTTP 400 and every
/// other one with the next scripted reply, recording each request body.
async fn refusing_images(replies: Vec<serde_json::Value>) -> (String, Arc<Mutex<Vec<String>>>) {
    refusing(replies, |body| body.contains("image_url").then_some("400 Bad Request")).await
}

/// Answers a request `refuse` names a status for with that status and every
/// other one with the next scripted reply, recording each request body.
async fn refusing(replies: Vec<serde_json::Value>, refuse: fn(&str) -> Option<&'static str>) -> (String, Arc<Mutex<Vec<String>>>) {
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let record = bodies.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        while let Ok((mut sock, _)) = listener.accept().await {
            let (mut buf, mut chunk, mut need) = (Vec::new(), [0u8; 8192], usize::MAX);
            while buf.len() < need {
                let Ok(n) = sock.read(&mut chunk).await else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if let (usize::MAX, Some(end)) = (need, buf.windows(4).position(|w| w == b"\r\n\r\n")) {
                    let headers = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                    let length = headers.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok());
                    need = end + 4 + length.unwrap_or(0);
                }
            }
            let end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
            let body = String::from_utf8_lossy(&buf[end + 4..]).to_string();
            let (status, payload) = if let Some(status) = refuse(&body) {
                (status, refusal(status))
            } else {
                ("200 OK", replies.next().expect("a scripted reply").to_string())
            };
            record.lock().unwrap().push(body);
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                payload.len()
            );
            let _ = sock.write_all(response.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    });
    (format!("http://{addr}"), bodies)
}

fn reply(message: serde_json::Value, finish: &str) -> serde_json::Value {
    json!({"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":{"prompt_tokens":10,"completion_tokens":5}})
}

/// The bug this guards: a host refused a request for an image a tool result
/// had just added (a PNG whose header passed and whose body did not), and
/// the session ended. The request that carried the image is refused, the
/// image becomes a line naming the refusal, the request goes again without
/// it and the turn finishes; reopened, the session never sends it again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_image_is_sent_once_and_the_session_goes_on_without_it() {
    let call = json!({"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.png\"}"}});
    let (url, bodies) = refusing_images(vec![
        reply(json!({"role":"assistant","content":null,"tool_calls":[call]}), "tool_calls"),
        reply(json!({"role":"assistant","content":"done"}), "stop"),
        reply(json!({"role":"assistant","content":"done again"}), "stop"),
    ])
    .await;
    let dir: PathBuf = std::env::temp_dir().join(format!("openmax-refused-image-{}", uuid::Uuid::new_v4()));
    let (data, project) = (dir.join("data"), dir.join("project"));
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&project).unwrap();
    let png = [&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13][..], b"IHDR", &[0, 0, 0, 9, 0, 0, 0, 9]].concat();
    std::fs::write(project.join("a.png"), &png).unwrap();
    let settings = json!({"provider":"test","model":"seer","context_tokens":32768,"max_tokens":1024,"image_input":true});
    std::fs::write(data.join("settings.json"), settings.to_string()).unwrap();
    std::fs::write(data.join("providers.json"), json!({"providers":{"test":{"base_url":url,"api_key":"fixture-key"}}}).to_string()).unwrap();
    open_max_core::trust::grant_trust(&data, &project, ApprovalMode::Auto).unwrap();

    let (core, mut rx) = Core::new(data.clone()).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "look at a.png").await;
    let errors: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::Error { message } => Some(message.clone()), _ => None }).collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Retry { reason, .. } if reason.contains("refused images a tool just returned"))));
    let line = format!("[image 1 of this result ({} bytes) not shown: the provider refused it: Provider returned error]", png.len());
    {
        let sent = bodies.lock().unwrap();
        assert_eq!(sent.len(), 3, "the call, the refused request, the resend");
        assert!(!sent[0].contains("image_url"));
        assert!(sent[1].contains("image_url") && !sent[1].contains(&line), "the image went once: {}", sent[1]);
        assert!(!sent[2].contains("image_url") && sent[2].contains(&line), "the resend carries the line: {}", sent[2]);
    }

    common::close(&core, &session).await;
    drop(core);
    let (core, mut rx) = Core::new(data).unwrap();
    let events = common::drive_turn(&core, &mut rx, &session, &project, "again").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    let sent = bodies.lock().unwrap();
    assert_eq!(sent.len(), 4);
    assert!(!sent[3].contains("image_url") && sent[3].contains(&line), "reopened, the line and not the image: {}", sent[3]);
    let _ = std::fs::remove_dir_all(dir);
}

/// Words of the ask that ends a notes request, which a cut's note does not carry.
const NOTES_ASK: &str = "Reply with the notes alone";

/// The most bytes of a request body the size-limited host below accepts.
const BODY_LIMIT: usize = 40_000;

/// A host that refuses any request body over `BODY_LIMIT` with HTTP 413.
fn over_the_limit(body: &str) -> Option<&'static str> {
    (body.len() > BODY_LIMIT).then_some("413 Payload Too Large")
}

/// A project holding a 20,000-byte PNG and four text files, the settings of a
/// session with image input against `url` whose cuts keep at most 4,000
/// tokens, and its data dir.
fn sized_project(url: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir: PathBuf = std::env::temp_dir().join(format!("openmax-oversized-{}", uuid::Uuid::new_v4()));
    let (data, project) = (dir.join("data"), dir.join("project"));
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&project).unwrap();
    let mut png = [&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13][..], b"IHDR", &[0, 0, 0, 9, 0, 0, 0, 9]].concat();
    png.resize(20_000, 7);
    std::fs::write(project.join("a.png"), &png).unwrap();
    std::fs::write(project.join("notes.txt"), "a line of notes\n".repeat(1_000)).unwrap();
    std::fs::write(project.join("short.txt"), "a short line\n".repeat(600)).unwrap();
    std::fs::write(project.join("intro.txt"), "an intro line\n".repeat(200)).unwrap();
    std::fs::write(project.join("big.txt"), "a long line of notes\n".repeat(2_200)).unwrap();
    let settings = json!({"provider":"test","model":"seer","context_tokens":200000,"max_tokens":1024,"image_input":true,"compaction_keep_tokens":4000});
    std::fs::write(data.join("settings.json"), settings.to_string()).unwrap();
    std::fs::write(data.join("providers.json"), json!({"providers":{"test":{"base_url":url,"api_key":"fixture-key"}}}).to_string()).unwrap();
    open_max_core::trust::grant_trust(&data, &project, ApprovalMode::Auto).unwrap();
    (dir, data, project)
}

/// A call of `read_file` on `path`, with `id`.
fn read(id: &str, path: &str) -> serde_json::Value {
    json!({"id":id,"type":"function","function":{"name":"read_file","arguments":json!({"path":path}).to_string()}})
}

/// The retry reasons a turn gave for a refusal for its size, in order.
fn size_steps(events: &[AgentEvent]) -> Vec<String> {
    events.iter().filter_map(|e| match e {
        AgentEvent::Retry { reason, .. } if reason.contains("refused the request for its size") => reason.split("); ").nth(1).map(str::to_string),
        _ => None,
    }).collect()
}

/// The bug this guards: a host refused a request with HTTP 413 for its size
/// in bytes, far inside the window in tokens, after a result that added no
/// image, so there was nothing new to leave out and the session ended. The
/// first step loses least: each image from before the latest reply becomes
/// a line and the request goes again, uncut, with no notes request, keeping
/// older text a cut would have removed. Reopened, the session never sends
/// the image again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn older_images_as_lines_are_enough_and_the_history_is_not_cut() {
    let (url, bodies) = refusing(vec![
        reply(json!({"role":"assistant","content":"reading the intro first","tool_calls":[read("c0", "intro.txt")]}), "tool_calls"),
        reply(json!({"role":"assistant","content":null,"tool_calls":[read("c1", "a.png")]}), "tool_calls"),
        reply(json!({"role":"assistant","content":null,"tool_calls":[read("c2", "notes.txt")]}), "tool_calls"),
        reply(json!({"role":"assistant","content":"done"}), "stop"),
        reply(json!({"role":"assistant","content":"done again"}), "stop"),
    ], over_the_limit)
    .await;
    let (dir, data, project) = sized_project(&url);
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "read intro.txt, look at a.png, then notes.txt").await;
    let errors: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::Error { message } => Some(message.clone()), _ => None }).collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(size_steps(&events), ["each image from before the latest reply is now a line"]);
    let line = "[image 1 of this result (20000 bytes) not shown: the provider refused a request this large: Provider returned error]";
    {
        let sent = bodies.lock().unwrap();
        assert_eq!(sent.len(), 5, "the call, two rounds, the refused request, the resend, and no notes request");
        assert!(sent.iter().all(|body| !body.contains(NOTES_ASK)), "no notes request");
        assert!(sent[2].contains("image_url") && sent[2].len() <= BODY_LIMIT, "the image went while it fit");
        assert!(sent[3].contains("image_url") && sent[3].len() > BODY_LIMIT, "refused for its size");
        let resend = &sent[4];
        assert!(!resend.contains("image_url") && resend.contains(line), "the image is a line");
        assert!(resend.contains("reading the intro first") && resend.contains("an intro line") && resend.contains("a line of notes"), "uncut, older text stays");
    }

    common::close(&core, &session).await;
    drop(core);
    let (core, mut rx) = Core::new(data).unwrap();
    let events = common::drive_turn(&core, &mut rx, &session, &project, "again").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    let sent = bodies.lock().unwrap();
    assert!(!sent[5].contains("image_url") && sent[5].contains(line), "reopened, the line and not the image");
    let _ = std::fs::remove_dir_all(dir);
}

/// With no older image to leave out, the next step is the cut: a history of
/// text the host refuses for its size, far inside the window, loses its
/// oldest exchanges to the cut, and the turn finishes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_text_history_refused_for_its_size_is_cut_and_goes_once_more() {
    let calls = ["first_read", "second_read", "third_read", "fourth_read"];
    let mut replies: Vec<_> = calls.iter().map(|id| reply(json!({"role":"assistant","content":null,"tool_calls":[read(id, "short.txt")]}), "tool_calls")).collect();
    replies.push(reply(json!({"role":"assistant","content":"done"}), "stop"));
    replies.push(reply(json!({"role":"assistant","content":"done again"}), "stop"));
    let (url, bodies) = refusing(replies, over_the_limit).await;
    let (dir, data, project) = sized_project(&url);
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "read short.txt four times").await;
    let errors: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::Error { message } => Some(message.clone()), _ => None }).collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(size_steps(&events), ["the history is cut"]);
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Compacted { compacted_messages, .. } if *compacted_messages > 0)), "the cut is reported as any cut is: {events:?}");
    {
        let sent = bodies.lock().unwrap();
        let refused = sent.iter().position(|body| body.len() > BODY_LIMIT).expect("a request over the limit");
        assert!(sent[refused].contains("fourth_read"), "refused after the fourth read");
        let resend = sent.last().unwrap();
        assert!(resend.len() <= BODY_LIMIT && resend.contains("fourth_read") && !resend.contains("first_read"), "the cut dropped the oldest read: {} bytes", resend.len());
        assert!(resend.contains("read short.txt four times"), "the prompt stays");
    }

    common::close(&core, &session).await;
    drop(core);
    let (core, mut rx) = Core::new(data).unwrap();
    let events = common::drive_turn(&core, &mut rx, &session, &project, "again").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    let sent = bodies.lock().unwrap();
    let reopened = sent.last().unwrap();
    assert!(reopened.contains("fourth_read") && !reopened.contains("first_read") && reopened.len() <= BODY_LIMIT, "reopened, the cut history goes");
    let _ = std::fs::remove_dir_all(dir);
}

/// The case a refusal of new images always covered: with no older image, the
/// images a tool has just returned become lines and the request goes once
/// more, uncut, with no notes request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn new_images_as_lines_are_enough_when_no_older_image_is_left() {
    let (url, bodies) = refusing(vec![
        reply(json!({"role":"assistant","content":"reading the intro first","tool_calls":[read("c0", "intro.txt")]}), "tool_calls"),
        reply(json!({"role":"assistant","content":null,"tool_calls":[read("c1", "notes.txt")]}), "tool_calls"),
        reply(json!({"role":"assistant","content":null,"tool_calls":[read("c2", "a.png")]}), "tool_calls"),
        reply(json!({"role":"assistant","content":"done"}), "stop"),
    ], over_the_limit)
    .await;
    let (dir, data, project) = sized_project(&url);
    let (core, mut rx) = Core::new(data).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "read intro.txt and notes.txt, then look at a.png").await;
    let errors: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::Error { message } => Some(message.clone()), _ => None }).collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(size_steps(&events), ["each image a tool just returned is now a line"]);
    let sent = bodies.lock().unwrap();
    assert_eq!(sent.len(), 5, "three rounds, the refused request, one resend, and no notes request");
    assert!(sent.iter().all(|body| !body.contains(NOTES_ASK)), "no notes request");
    assert!(sent[3].contains("image_url") && sent[3].len() > BODY_LIMIT, "refused for its size");
    let resend = &sent[4];
    assert!(!resend.contains("image_url") && resend.contains("not shown: the provider refused it"), "the new image is a line");
    assert!(resend.contains("reading the intro first") && resend.contains("an intro line"), "uncut, older text stays");
    let _ = std::fs::remove_dir_all(dir);
}

/// The steps are bounded: a host that refuses every request of a round for
/// its size, after each step too, gets four of them, the request and one
/// after each step (older images as lines, new ones, the cut). The cut goes
/// again, finds nothing more to drop, and the turn ends on the refusal with
/// no fifth request. The round's last result is text over the limit on its
/// own, which no step removes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_round_refused_for_its_size_at_every_step_ends_after_four_requests() {
    let (url, bodies) = refusing(vec![
        reply(json!({"role":"assistant","content":null,"tool_calls":[read("first_look", "a.png")]}), "tool_calls"),
        reply(json!({"role":"assistant","content":null,"tool_calls":[read("long_read", "big.txt"), read("second_look", "a.png")]}), "tool_calls"),
    ], over_the_limit)
    .await;
    let (dir, data, project) = sized_project(&url);
    let (core, mut rx) = Core::new(data).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "look at a.png, then read big.txt and look again").await;
    assert_eq!(size_steps(&events), ["each image from before the latest reply is now a line", "each image a tool just returned is now a line", "the history is cut", "the history is cut"]);
    let attempts: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::Retry { attempt, max_attempts, .. } => Some((*attempt, *max_attempts)), _ => None }).collect();
    assert_eq!(attempts, [(2, 6), (3, 6), (4, 6), (5, 6)], "one round's attempts rise by one under one maximum");
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Error { message } if message.starts_with("backend returned 413"))), "{events:?}");
    let sent = bodies.lock().unwrap();
    let round: Vec<_> = sent.iter().filter(|body| body.contains("long_read") && !body.contains(NOTES_ASK)).collect();
    let images: Vec<_> = round.iter().map(|body| body.matches(r#""type":"image_url""#).count()).collect();
    assert_eq!(images, [2, 1, 0, 0], "the request, then one resend after each step");
    assert!(round.iter().all(|body| body.len() > BODY_LIMIT), "each refused for its size");
    let _ = std::fs::remove_dir_all(dir);
}

/// A cut with nothing to drop sends nothing: the prompt and one result over
/// the limit by itself, which the cut keeps, are refused for their size, and
/// with no image to leave out and no exchange to cut the turn ends on that
/// refusal with no further request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cut_with_nothing_to_drop_ends_the_turn_without_a_resend() {
    let (url, bodies) = refusing(vec![
        reply(json!({"role":"assistant","content":null,"tool_calls":[read("long_read", "big.txt")]}), "tool_calls"),
    ], over_the_limit)
    .await;
    let (dir, data, project) = sized_project(&url);
    let (core, mut rx) = Core::new(data).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "read big.txt").await;
    assert_eq!(size_steps(&events), ["the history is cut"]);
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Compacted { .. })), "nothing was cut: {events:?}");
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Error { message } if message == "backend returned 413 Payload Too Large: Provider returned error")), "{events:?}");
    let sent = bodies.lock().unwrap();
    assert_eq!(sent.len(), 2, "the prompt's request, the refused one, and nothing after it");
    assert!(sent[1].len() > BODY_LIMIT && !sent.iter().any(|body| body.contains(NOTES_ASK)));
    let _ = std::fs::remove_dir_all(dir);
}

/// A refusal for size says nothing of tokens: a text history far under what
/// a cut usually keeps (20,000 tokens) still loses its older exchanges to the
/// cut that answers it, and the session carries on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_size_cut_drops_older_exchanges_under_the_usual_target() {
    let calls = ["first_read", "second_read", "third_read", "fourth_read"];
    let mut replies: Vec<_> = calls.iter().map(|id| reply(json!({"role":"assistant","content":null,"tool_calls":[read(id, "short.txt")]}), "tool_calls")).collect();
    replies.push(reply(json!({"role":"assistant","content":"done"}), "stop"));
    let (url, bodies) = refusing(replies, over_the_limit).await;
    let (dir, data, project) = sized_project(&url);
    // The usual keep, with no compaction_keep_tokens setting.
    let settings = json!({"provider":"test","model":"seer","context_tokens":200000,"max_tokens":1024,"image_input":true});
    std::fs::write(data.join("settings.json"), settings.to_string()).unwrap();
    let (core, mut rx) = Core::new(data).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "read short.txt four times").await;
    let errors: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::Error { message } => Some(message.clone()), _ => None }).collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(size_steps(&events), ["the history is cut"]);
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Compacted { compacted_messages, .. } if *compacted_messages > 0)), "{events:?}");
    let sent = bodies.lock().unwrap();
    let refused = sent.iter().position(|body| body.len() > BODY_LIMIT).expect("a request over the limit");
    assert!(sent[refused].len() / 4 < 20_000, "the refused history is under the usual keep in tokens");
    let resend = sent.last().unwrap();
    assert!(resend.len() <= BODY_LIMIT && resend.contains("fourth_read") && !resend.contains("first_read"), "the cut dropped the oldest read: {} bytes", resend.len());
    assert!(resend.contains("read short.txt four times"), "the prompt stays");
    let _ = std::fs::remove_dir_all(dir);
}

/// How many image parts each recorded request carried.
fn images_per_request(bodies: &[String]) -> Vec<usize> {
    bodies.iter().map(|body| body.matches("data:image/").count()).collect()
}

/// The messages a recorded Chat Completions request sent.
fn sent_messages(body: &str) -> Vec<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(body).unwrap()["messages"].as_array().unwrap().clone()
}

/// The bug this guards: every image a tool returned went again, whole, with
/// every later request, slowing each and growing the body until a server
/// refused it. Past `image_bytes` (here 60,000 bytes of base64, a little over
/// two of the 26,668-byte images) the oldest from before the latest reply
/// become lines down to half of it, in one batch, so the request between two
/// batches repeats the one before it and keeps its cached prefix. The image a
/// tool has just returned always goes, and a reopened session sends the lines.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn images_past_the_budget_become_lines_a_batch_at_a_time() {
    let mut replies: Vec<_> = (1..=5).map(|n| reply(json!({"role":"assistant","content":null,"tool_calls":[read(&format!("c{n}"), "a.png")]}), "tool_calls")).collect();
    replies.extend([reply(json!({"role":"assistant","content":"done"}), "stop"), reply(json!({"role":"assistant","content":"done again"}), "stop")]);
    let (url, bodies) = refusing(replies, |_| None).await;
    let (dir, data, project) = sized_project(&url);
    let mut settings: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(data.join("settings.json")).unwrap()).unwrap();
    settings["image_bytes"] = json!(60_000);
    std::fs::write(data.join("settings.json"), settings.to_string()).unwrap();
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "look at a.png five times").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    let notes: Vec<_> = events.iter().filter_map(|e| match e { AgentEvent::HarnessNote { text, .. } if text.contains("older images are now lines") => Some(text.as_str()), _ => None }).collect();
    assert_eq!(notes, ["2 older images are now lines: the images a request carries passed 60000 bytes"; 2]);
    let line = "[image 1 of this result (20000 bytes) not shown: a request carries at most 60000 bytes of images, the oldest going first; read the file again to see it]";
    {
        let sent = bodies.lock().unwrap();
        assert_eq!(images_per_request(&sent), [0, 1, 2, 1, 2, 1], "a third image drops the two before it, down to half the budget");
        assert_eq!(sent[3].matches(line).count(), 2, "each image gone is a line where it was");
        assert!(sent[5].contains(line) && sent[5].matches(line).count() == 4);
        let (fourth, fifth) = (sent_messages(&sent[3]), sent_messages(&sent[4]));
        assert_eq!(fourth[..], fifth[..fourth.len()], "between two batches a request repeats the one before it");
        let last = sent_messages(&sent[5]);
        let newest = last.iter().rposition(|m| m["role"] == "tool").unwrap();
        assert!(!last[newest]["content"].as_str().unwrap().contains("not shown"), "the image a tool has just returned goes");
    }

    common::close(&core, &session).await;
    drop(core);
    let (core, mut rx) = Core::new(data).unwrap();
    let events = common::drive_turn(&core, &mut rx, &session, &project, "again").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    let sent = bodies.lock().unwrap();
    assert_eq!(images_per_request(&sent)[6], 1, "reopened, the lines and the newest image");
    assert_eq!(sent[6].matches(line).count(), 4);
    let _ = std::fs::remove_dir_all(dir);
}

/// A request refused for its size teaches the session what the route takes:
/// after the first refusal, later requests carry at most half the image bytes
/// the refused one did, so a session that keeps reading images is not
/// refused again (here a 40,000-byte limit and two 26,668-byte images).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refusal_for_size_lowers_the_image_bytes_for_the_session() {
    let mut replies: Vec<_> = (1..=3).map(|n| reply(json!({"role":"assistant","content":null,"tool_calls":[read(&format!("c{n}"), "a.png")]}), "tool_calls")).collect();
    replies.push(reply(json!({"role":"assistant","content":"done"}), "stop"));
    let (url, bodies) = refusing(replies, over_the_limit).await;
    let (dir, data, project) = sized_project(&url);
    let (core, mut rx) = Core::new(data).unwrap();
    let session = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let events = common::drive_turn(&core, &mut rx, &session, &project, "look at a.png three times").await;
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Error { .. })), "{events:?}");
    assert_eq!(size_steps(&events), ["each image from before the latest reply is now a line"], "one refusal, no second");
    assert!(events.iter().any(|e| matches!(e, AgentEvent::HarnessNote { text, .. } if text == "1 older images are now lines: the images a request carries passed 26668 bytes")), "{events:?}");
    let sent = bodies.lock().unwrap();
    assert_eq!(images_per_request(&sent), [0, 1, 2, 1, 1]);
    assert_eq!(sent.iter().filter(|body| body.len() > BODY_LIMIT).count(), 1, "refused once");
    let _ = std::fs::remove_dir_all(dir);
}
