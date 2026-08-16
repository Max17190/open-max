//! Policy notices reach the model, not just the UI.
//!
//! Two silences this guards against: an agent writes itself a project
//! `allow` rule, the rule is inert until a human approves the file (#180),
//! and nothing ever tells the model why it keeps being prompted; and a hook
//! or permission problem present at turn start is announced as a UI event
//! the model never sees. Each distinct notice lands in the transcript once
//! per session - repetition there is token spend, and the condition holds
//! every turn once it holds at all.

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

fn write_config(data: &std::path::Path, base_url: &str, project: &std::path::Path) {
    std::fs::create_dir_all(data).unwrap();
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
}

async fn drive_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<open_max_core::types::AgentEventEnvelope>,
    session: &str,
    project: &std::path::Path,
    text: &str,
) {
    start_turn(Arc::clone(core), session.into(), project.to_path_buf(), text.into()).unwrap();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        if matches!(envelope.event, AgentEvent::Done { .. }) {
            break;
        }
    }
    // Done precedes the running flag clearing; wait for idle before the
    // caller starts another turn (a fast runner races that window).
    while core.is_running(session) {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

const ALLOW_RULE: &str = "[[rules]]\neffect = \"allow\"\ntool = \"bash\"\narg_regex = \"^git status\"\n";

/// The acute case: the agent writes itself an allow rule mid-turn. The rule
/// is inert (#180), and the very tool result that wrote it must say so and
/// name the command that lifts it - otherwise the agent keeps being
/// prompted with no stated cause.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_inert_allow_written_mid_turn_is_named_on_the_writing_call() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/permissions.toml", "content": ALLOW_RULE }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);

    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "inert-allow", &project, "allow git status").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let tool_msg = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the write_file result is in the transcript");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("[permission notice:"),
        "the inert-allow notice rides the writing call's result: {content}"
    );
    assert!(content.contains("inert"), "{content}");
    assert!(
        content.contains("openmax --approve"),
        "the notice names the command that lifts it: {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A notice already true at turn start (here: a pre-existing inert allow
/// file) is narrated to the model once - in the first turn's transcript,
/// ahead of the prompt - and not repeated on later turns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_start_policy_notice_lands_once_per_session() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax")).unwrap();
    std::fs::write(project.join(".openmax/permissions.toml"), ALLOW_RULE).unwrap();
    let project = project.canonicalize().unwrap();

    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_text("first"),
        completion_with_text("second"),
    ])
    .await;
    write_config(&data, &base_url, &project);

    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "startup-notice", &project, "turn one").await;
    drive_turn(&core, &mut rx, "startup-notice", &project, "turn two").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let count_notices = |body: &str| {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        v["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| {
                m["role"] == "user"
                    && m["content"].as_str().is_some_and(|c| c.starts_with("[policy notice:"))
            })
            .count()
    };
    assert_eq!(
        count_notices(&bodies[0]),
        1,
        "the first request carries the notice ahead of the prompt"
    );
    let first: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    let note = first["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["content"].as_str().is_some_and(|c| c.starts_with("[policy notice:")))
        .unwrap();
    assert!(note["content"].as_str().unwrap().contains("inert"));
    assert_eq!(
        count_notices(&bodies[1]),
        1,
        "turn two inherits the transcript's one note and adds no second copy"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Once per SESSION means across the session's whole life: a resumed
/// session (new process, hydrated transcript) must not re-narrate a still-
/// applicable static notice the transcript already carries.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resumed_session_does_not_repeat_a_persisted_notice() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax")).unwrap();
    std::fs::write(project.join(".openmax/permissions.toml"), ALLOW_RULE).unwrap();
    let project = project.canonicalize().unwrap();

    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_text("first process"),
        completion_with_text("second process"),
    ])
    .await;
    write_config(&data, &base_url, &project);

    // Process one: an INDEXED session (saves are silent no-ops for an
    // unindexed id, which would make this test pass vacuously), the notice
    // lands, the transcript persists.
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    let session = open_max_core::sessions::create(&core, project.to_string_lossy().into())
        .unwrap()
        .id;
    drive_turn(&core, &mut rx, &session, &project, "turn one").await;
    assert!(
        open_max_core::sessions::load_messages(&core, &session).is_some(),
        "the transcript must be on disk for this to be a real resume"
    );
    drop(rx);
    drop(core);

    // Process two: same session id, hydrated from disk.
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, &session, &project, "turn two").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let notes = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| {
            m["role"] == "user"
                && m["content"].as_str().is_some_and(|c| c.starts_with("[policy notice:"))
        })
        .count();
    assert_eq!(notes, 1, "the resumed process must not add a second copy of the notice");
    let _ = std::fs::remove_dir_all(dir);
}

/// A hook file written mid-turn gets a receipt on the writing call saying it
/// is NOT running (inert until a human approves it out of session). Hooks
/// are outside the extension fingerprint, so before this the write got no
/// receipt at all and the agent believed its gate was live for a whole turn
/// (round-4 dogfood).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hook_written_mid_turn_is_named_inert_on_the_writing_call() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let hook = "event = \"pre_tool_use\"\ncommand = \"/bin/true\"\n";
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/hooks/gate.toml", "content": hook }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "hook-write", &project, "install a gate").await;

    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let content = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(content.contains("[hook files changed"), "{content}");
    assert!(content.contains("Not running"), "the hook is named inert now, not next turn: {content}");
    assert!(content.contains("openmax --approve"), "{content}");
    let _ = std::fs::remove_dir_all(dir);
}

/// A receipt's approve command is a copyable shell line: a manifest filename
/// with a metacharacter must arrive quoted, or the paste runs the
/// metacharacter instead of naming the file (review finding).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_receipt_shell_quotes_a_metacharacter_manifest_path() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let manifest = "name = \"odd\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n";
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/a b$(x).toml", "content": manifest }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "quote", &project, "go").await;
    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let content = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        content.contains("openmax --approve '.openmax/tools/a b$(x).toml'"),
        "the path must be single-quoted in the copyable command: {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// An APPROVED gate whose bytes change mid-turn is not inert - it fails
/// closed and blocks every call from the next turn. The write receipt must
/// say that, not "approved hooks apply from the next turn" (review finding).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_revoked_gate_write_receipt_says_calls_are_blocked() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/hooks")).unwrap();
    let project = project.canonicalize().unwrap();
    let hook_path = project.join(".openmax/hooks/gate.toml");
    std::fs::write(&hook_path, "event = \"pre_tool_use\"\ncommand = \"/bin/true\"\n").unwrap();
    // The human approves the gate.
    std::fs::create_dir_all(&data).unwrap();
    let sha = open_max_core::ledger::sha256_hex(&std::fs::read(&hook_path).unwrap());
    open_max_core::ledger::approve_capability(&data, &project, &hook_path, &[sha]).unwrap();

    // The agent then edits the approved gate.
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({
                "path": ".openmax/hooks/gate.toml",
                "content": "event = \"pre_tool_use\"\ncommand = \"/bin/false\"\n"
            }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "revoke", &project, "edit the gate").await;
    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let content = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(content.contains("failing closed"), "{content}");
    assert!(content.contains("BLOCKED"), "{content}");
    assert!(
        !content.contains("approved hooks apply from the next turn"),
        "a revoked gate must not be described as merely pending: {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// An approval landed by a human at another terminal (out of session) is
/// named at the running session's next turn start - the one channel #199
/// said it reached through was none. The session's OWN in-session grants
/// are not re-narrated (the model watched those on the card).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_out_of_session_approval_is_named_at_the_next_turn() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let manifest = project.join(".openmax/tools/docsearch.toml");
    std::fs::write(&manifest, "name = \"docsearch\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n").unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_text("turn one"),
        completion_with_text("turn two"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    // Turn one: nothing approved yet.
    drive_turn(&core, &mut rx, "outside", &project, "one").await;
    // A human at another terminal approves the tool (no session_id).
    let sha = open_max_core::ledger::sha256_hex(&std::fs::read(&manifest).unwrap());
    open_max_core::ledger::approve_capability(&data, &project, &manifest, &[sha]).unwrap();
    // Turn two: the session names the approval it did not witness.
    drive_turn(&core, &mut rx, "outside", &project, "two").await;
    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let note = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| {
            m["role"] == "user"
                && m["content"].as_str().is_some_and(|c| c.starts_with("[approval activity outside this session:"))
        })
        .expect("the outside approval is named");
    let c = note["content"].as_str().unwrap();
    assert!(c.contains("approved .openmax/tools/docsearch.toml"), "{c}");
    // Turn one carried no such note (nothing had been approved).
    let first: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    assert!(
        !first["messages"].as_array().unwrap().iter().any(|m| m["content"]
            .as_str()
            .is_some_and(|c| c.starts_with("[approval activity outside"))),
        "turn one predates the approval"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A mutating call that leaves permissions.toml malformed bricks every tool
/// call for the rest of the turn (dogfood: 29 silent denies, --check itself
/// denied). The writing call now carries a receipt naming the parse reason
/// and the turn-scoped consequence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_invalidating_write_to_permissions_is_named_on_the_writing_call() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/permissions.toml", "content": "[[rule]]\neffect = \"deny\"\n" }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "brick-perms", &project, "write it").await;
    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let content = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(content.contains("permissions.toml is now malformed"), "{content}");
    assert!(content.contains("DENIED for the rest of THIS turn"), "{content}");
    assert!(content.contains("START OF THE NEXT TURN"), "{content}");
    // The recovery guidance names only the tools that are actually available:
    // read_file was removed from the repair carve-out, so advertising it would
    // send the model into another immediate denial (Greptile).
    assert!(
        content.contains("write_file/edit_file on exactly this file is still allowed"),
        "{content}"
    );
    assert!(
        !content.contains("write_file/edit_file/read_file"),
        "the receipt must not advertise the denied read_file: {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Editing the SCRIPT an approved gate runs revokes the approval and fails
/// closed, but the script is not a .toml - the hook fingerprint missed it,
/// so the edit got no receipt and the agent learned only when its next call
/// was blocked (dogfood). The bound code is now in the fingerprint, so the
/// hook-write receipt fires on the code edit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn editing_an_approved_gates_script_is_named_on_the_writing_call() {
    let dir = std::env::temp_dir().join(format!("omx-notice-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/hooks")).unwrap();
    let project = project.canonicalize().unwrap();
    let gate = project.join("gate.sh");
    std::fs::write(&gate, "#!/bin/sh\nexit 0\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&gate, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let hook = project.join(".openmax/hooks/gate.toml");
    std::fs::write(&hook, "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
    // The human approved the hook manifest and its script.
    std::fs::create_dir_all(&data).unwrap();
    let mut shas = vec![open_max_core::ledger::sha256_hex(&std::fs::read(&hook).unwrap())];
    shas.extend(
        open_max_core::ledger::manifest_code(&hook, &project)
            .into_iter()
            .filter_map(|c| c.sha256),
    );
    open_max_core::ledger::approve_capability(&data, &project, &hook, &shas).unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": "gate.sh", "content": "#!/bin/sh\n# tampered\nexit 0\n" }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "gate-script-edit", &project, "edit the gate script").await;
    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let content = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(content.contains("[hook files changed"), "{content}");
    assert!(content.contains("failing closed"), "an edit to an approved gate's code is named as revoking: {content}");
    let _ = std::fs::remove_dir_all(dir);
}
