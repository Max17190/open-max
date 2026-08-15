//! settings.json drift gets a model-visible receipt.
//!
//! Settings are read once at launch, by design: base_url/api_key are
//! credential routing and a hot-adopted approval_mode would be
//! self-approval. The receipt makes the boundary loud instead of silent -
//! an agent edit is inert for this session (say so, and say what still
//! runs), and a malformed edit bricks the NEXT launch (exit 2), which must
//! be said while the author can still repair the file. The TUI's own saves
//! go through Core::save_settings and must never read as drift.

use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::AgentEvent;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn recording_endpoint(
    responses: Vec<serde_json::Value>,
) -> (String, Arc<Mutex<Vec<String>>>) {
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let record = bodies.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        for body in responses {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            let mut need = usize::MAX;
            loop {
                let Ok(n) = sock.read(&mut chunk).await else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if need == usize::MAX {
                    if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                        need = end
                            + 4
                            + headers
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                    }
                }
                if need != usize::MAX && buf.len() >= need {
                    break;
                }
            }
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                record
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[end + 4..]).to_string());
            }
            let payload = serde_json::to_string(&body).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            let _ = sock.write_all(response.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
    });
    (format!("http://{addr}"), bodies)
}

fn completion_with_tool_call(name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": format!("call-{name}"),
                    "type": "function",
                    "function": { "name": name, "arguments": args.to_string() }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn completion_with_text(text: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }]
    })
}

struct Env {
    dir: std::path::PathBuf,
    data: std::path::PathBuf,
    project: std::path::PathBuf,
    bodies: Arc<Mutex<Vec<String>>>,
    core: Arc<Core>,
    rx: tokio::sync::mpsc::UnboundedReceiver<open_max_core::types::AgentEventEnvelope>,
}

async fn build_env(responses: Vec<serde_json::Value>) -> Env {
    let dir = std::env::temp_dir().join(format!("omx-settings-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let (base_url, bodies) = recording_endpoint(responses).await;
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        data.join("trust.json"),
        serde_json::json!({ "version": 1, "projects": [project.to_string_lossy()] }).to_string(),
    )
    .unwrap();
    let (core, rx) = Core::new(data.clone()).unwrap();
    Env { dir, data, project, bodies, core, rx }
}

async fn drive_turn(env: &mut Env, session: &str, text: &str) {
    start_turn(Arc::clone(&env.core), session.into(), env.project.clone(), text.into()).unwrap();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), env.rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        if matches!(envelope.event, AgentEvent::Done { .. }) {
            break;
        }
    }
}

fn tool_result(body: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the bash result is in the transcript")["content"]
        .as_str()
        .unwrap()
        .to_string()
}

/// An agent bash edit to settings.json is answered on the writing call:
/// the launch-read rule, what this session still runs, and the clobber
/// warning. A follow-up bash edit that leaves the file INVALID gets the
/// brick warning while the author can still repair it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn settings_drift_valid_and_broken_are_both_narrated() {
    let mut env = build_env(vec![completion_with_text("placeholder")]).await;
    // Rebuild the script list now that the data path is known.
    let settings_path = env.data.join("settings.json");
    let valid_edit = format!(
        "python3 -c 'import json,sys; p=sys.argv[1]; s=json.load(open(p)); s[\"model\"]=\"grok-4.6\"; json.dump(s, open(p, \"w\"))' {}",
        settings_path.display()
    );
    let broken_edit = format!("printf '{{\"model\": nope' > {}", settings_path.display());
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call("bash", serde_json::json!({ "command": valid_edit })),
        completion_with_tool_call("bash", serde_json::json!({ "command": broken_edit })),
        completion_with_text("done"),
    ])
    .await;
    // Point the already-built env's settings at the new endpoint; adopt the
    // rewrite as the harness's own so only the agent's edits read as drift.
    let mut s = env.core.settings.lock().unwrap().clone();
    s.base_url = base_url;
    env.core.save_settings(&s).unwrap();
    *env.core.settings.lock().unwrap() = s;
    env.bodies = bodies;

    drive_turn(&mut env, "drift", "switch me to grok-4.6").await;

    let bodies = env.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 3);
    let first_result = tool_result(&bodies[1]);
    assert!(
        first_result.contains("[settings.json changed on disk"),
        "the valid edit is narrated on the writing call: {first_result}"
    );
    assert!(
        first_result.contains("read at launch"),
        "the receipt states the launch-read boundary: {first_result}"
    );
    assert!(
        first_result.contains("still runs scripted"),
        "the receipt names what this session still runs: {first_result}"
    );
    let v: serde_json::Value = serde_json::from_str(&bodies[2]).unwrap();
    let second_result = v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "tool")
        .nth(1)
        .expect("both bash results are in the transcript")["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        second_result.contains("INVALID"),
        "the broken edit gets the brick warning while repairable: {second_result}"
    );
    assert!(second_result.contains("exit 2"), "{second_result}");
    let _ = std::fs::remove_dir_all(&env.dir);
}

/// The harness's own saves never read as drift: a Core::save_settings write
/// produces no note on the next turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tui_authored_save_is_not_drift() {
    let mut env = build_env(vec![
        completion_with_text("first"),
        completion_with_text("second"),
    ])
    .await;
    drive_turn(&mut env, "own-save", "turn one").await;

    let mut s = env.core.settings.lock().unwrap().clone();
    s.model = "scripted-2".into();
    env.core.save_settings(&s).unwrap();
    *env.core.settings.lock().unwrap() = s;

    drive_turn(&mut env, "own-save", "turn two").await;
    let bodies = env.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    assert!(
        !bodies[1].contains("settings.json changed on disk"),
        "a save through the harness's own hand must not be narrated as drift"
    );
    let _ = std::fs::remove_dir_all(&env.dir);
}

/// Missing and unreadable are distinct drift states: a launch with no
/// settings.json followed by an edit that leaves a DIRECTORY at the path
/// (unreadable as a file, and it bricks the next launch) must be reported,
/// not treated as "still missing".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_missing_settings_file_replaced_by_a_directory_is_drift() {
    let mut env = build_env(vec![completion_with_text("placeholder")]).await;
    let settings_path = env.data.join("settings.json");
    // Recreate the endpoint script now the path is known: mkdir at the
    // settings path is exactly the shape a careless bash write can leave.
    let mkdir = format!("rm -f {p} && mkdir {p}", p = settings_path.display());
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call("bash", serde_json::json!({ "command": mkdir })),
        completion_with_text("done"),
    ])
    .await;
    // Simulate "launched with no settings.json": the process's own view is
    // Missing. Point the live settings at the endpoint in memory only and
    // remove the file the fixture wrote.
    let mut s = env.core.settings.lock().unwrap().clone();
    s.base_url = base_url;
    *env.core.settings.lock().unwrap() = s;
    std::fs::remove_file(&settings_path).unwrap();
    // Adopt the missing state as the launch view.
    let _ = env.core.settings_disk_changed();
    env.bodies = bodies;

    drive_turn(&mut env, "dir-drift", "go").await;
    let bodies = env.bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let result = tool_result(&bodies[1]);
    assert!(
        result.contains("INVALID"),
        "a directory at the settings path is drift that bricks the next launch: {result}"
    );
    let _ = std::fs::remove_dir_all(&env.dir);
}

/// The fingerprint after a harness save is of the bytes the harness WROTE,
/// never a re-read of the path. Modeled deterministically: the external
/// replacement lands BEFORE the harness adopts its own save (the interval-
/// race outcome). A re-read-based adoption would swallow the foreign bytes
/// as this process's own; fingerprinting the intended bytes keeps them
/// visible as drift.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_racing_external_replacement_is_not_adopted_by_save() {
    let env = build_env(vec![completion_with_text("only")]).await;
    let settings_path = env.data.join("settings.json");
    let mut s = env.core.settings.lock().unwrap().clone();
    s.model = "scripted-2".into();
    // The harness's write (through the direct helper, as /model does)...
    open_max_core::config::save(&env.data, &s).unwrap();
    // ...an external replacement lands in the interval...
    let mut external = s.clone();
    external.model = "external-edit".into();
    std::fs::write(&settings_path, serde_json::to_string_pretty(&external).unwrap()).unwrap();
    // ...and only now does the harness adopt what IT saved.
    env.core.adopt_saved_settings(&s);
    *env.core.settings.lock().unwrap() = s;
    assert!(
        env.core.settings_disk_changed().is_some(),
        "the foreign bytes must read as drift, not be adopted as this process's own"
    );
    let _ = std::fs::remove_dir_all(&env.dir);
}
