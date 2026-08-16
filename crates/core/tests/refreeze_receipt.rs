//! The mid-turn refreeze receipt reaches the model, not just the UI.
//!
//! What this guards: an agent that writes a valid tool file gets a Refrozen
//! event on the wire, but that event is frontend chrome - nothing in the
//! transcript told the model its new tool became callable. The eval showed
//! the resulting failure shape twice: a correct tool authored, three
//! refreezes fired, and the model ran the script by hand instead of calling
//! the tool. The receipt must land where the model reads: in the transcript
//! the next request carries.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::AgentEvent;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A completions endpoint that answers each request with the next scripted
/// message and records every request body it saw.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_request_after_a_refreeze_carries_the_receipt_and_the_new_tool_name() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    let manifest = "name = \"wordfreq\"\ndescription = \"top-n words\"\ncommand = \"/bin/echo\"\nmutating = false\n";
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/wordfreq.toml", "content": manifest }),
        ),
        completion_with_text("done"),
    ])
    .await;

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

    let (core, mut rx) = Core::new(data).unwrap();
    start_turn(Arc::clone(&core), "receipt-test".into(), PathBuf::from(&project), "go".into())
        .unwrap();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        if matches!(envelope.event, AgentEvent::Done { .. }) {
            break;
        }
    }

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "the scripted turn makes exactly two requests");
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let tool_msg = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the write_file result is in the transcript");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("[extension refreeze:"),
        "the refreeze receipt must reach the model's transcript: {content}"
    );
    assert!(
        content.contains("wordfreq"),
        "the receipt names the tool that became callable: {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
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
    // Done is emitted before the session's `running` flag clears; a second
    // start_turn racing that window is refused as "already working". Wait
    // for idle, as the headless front end does.
    while core.is_running(session) {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// A broken tool write refreezes too (its bytes moved the fingerprint), and
/// the receipt the model reads says the file did NOT load and why - without
/// this, "file changed" reads as "tool is live" and the model calls a tool
/// that does not exist, or worse, believes its work is done.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_broken_tool_write_gets_a_not_loaded_receipt() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    // Missing required field `command`: valid TOML, invalid manifest.
    let broken = "name = \"wordfreq\"\ndescription = \"top-n words\"\n";
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/wordfreq.toml", "content": broken }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);

    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "broken-receipt", &project, "go").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "the scripted turn makes exactly two requests");
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let tool_msg = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the write_file result is in the transcript");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("NOT loaded"),
        "the receipt must flag the failed load: {content}"
    );
    assert!(content.contains("wordfreq.toml"), "the receipt names the file: {content}");
    assert!(
        content.contains("command"),
        "the receipt carries the parse reason (missing field): {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The turn-start refreeze - a tool installed by a human, git, or another
/// session between turns - must be narrated to the MODEL, not only to the
/// UI. The next turn's first request carries a harness note naming the
/// added tool ahead of the user's prompt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_start_refreeze_note_reaches_the_model() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_text("hello"),
        completion_with_text("I see the checksum tool"),
    ])
    .await;
    write_config(&data, &base_url, &project);

    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "turnstart-receipt", &project, "first turn").await;

    // A teammate installs a tool while no turn is running.
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    std::fs::write(
        project.join(".openmax/tools/checksum.toml"),
        "name = \"checksum\"\ndescription = \"sha256\"\ncommand = \"/bin/echo\"\nmutating = false\n",
    )
    .unwrap();
    drive_turn(&core, &mut rx, "turnstart-receipt", &project, "second turn").await;

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let messages = second["messages"].as_array().unwrap();
    let note = messages
        .iter()
        .find(|m| {
            m["role"] == "user"
                && m["content"].as_str().is_some_and(|c| c.starts_with("[extension refreeze:"))
        })
        .expect("the turn-start refreeze note is in the model's transcript");
    let content = note["content"].as_str().unwrap();
    assert!(
        content.contains("checksum"),
        "the note names the tool that appeared: {content}"
    );
    let last = messages.last().unwrap();
    assert_eq!(
        last["content"].as_str().unwrap(),
        "second turn",
        "the user's prompt stays the final message; the note precedes it"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The receipt must not oversell: an unapproved tool is registered, and its
/// first call stops for a human. Round-4 dogfooding watched the receipt say
/// "callable" and the very next step raise an approval card - the model
/// itself named the receipt as dishonest. Approved bytes keep "callable".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_receipt_distinguishes_callable_from_registered_pending_approval() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    let manifest = "name = \"wordfreq\"\ndescription = \"top-n words\"\ncommand = \"/bin/echo\"\nmutating = false\n";
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/wordfreq.toml", "content": manifest }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    drive_turn(&core, &mut rx, "unapproved-receipt", &project, "go").await;

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
        !content.contains("callable from your next step: wordfreq"),
        "an unapproved tool must not be called callable: {content}"
    );
    assert!(content.contains("registered"), "{content}");
    assert!(content.contains("human approval"), "{content}");
    assert!(
        content.contains("openmax --approve '.openmax/tools/wordfreq.toml'"),
        "the receipt names the exact approve command, shell-quoted: {content}"
    );
    assert!(content.contains("--run-examples"), "and the probe path: {content}");
    drop(bodies);
    let _ = std::fs::remove_dir_all(dir);
}

/// Approval outcomes reach the model. (1) An approved first call carries a
/// receipt naming the grant and its revocation rule - the approve path was
/// silent, so an approved call looked identical to one that never stopped,
/// and a model that saw its edited tool re-prompt then run concluded that
/// revocation does not work. (2) Editing an approved manifest is narrated at
/// the refreeze as revoking the approval. (3) Editing only the SCRIPT (no
/// fingerprint moves, no refreeze) is narrated on the writing call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn approval_grants_and_revocations_are_narrated_to_the_model() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let script = project.join("echo.sh");
    std::fs::write(&script, "#!/bin/sh\ncat\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let manifest_v1 = "name = \"echoer\"\ndescription = \"d\"\ncommand = \"./echo.sh\"\nmutating = false\n";
    let manifest_v2 = "name = \"echoer\"\ndescription = \"d2\"\ncommand = \"./echo.sh\"\nmutating = false\n";
    std::fs::write(project.join(".openmax/tools/echoer.toml"), manifest_v1).unwrap();

    let (base_url, bodies) = recording_endpoint(vec![
        // Turn 1: call the unapproved tool (card -> approved), then edit its
        // manifest (refreeze: revocation), then edit only its script.
        completion_with_tool_call("echoer", serde_json::json!({ "x": 1 })),
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/echoer.toml", "content": manifest_v2 }),
        ),
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": "echo.sh", "content": "#!/bin/sh\n# edited\ncat\n" }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    // The human answers the card when it fires.
    start_turn(Arc::clone(&core), "grant-narrated".into(), project.clone(), "go".into()).unwrap();
    let mut approved_once = false;
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match envelope.event {
            AgentEvent::ApprovalRequest { approval_id, .. } => {
                approved_once = true;
                core.respond_approval(&approval_id, true);
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert!(approved_once, "the first call must have raised the card");
    let bodies_now = bodies.lock().unwrap().clone();
    assert!(bodies_now.len() >= 3);
    let tool_msgs = |body: &str| -> Vec<String> {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        v["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["content"].as_str().unwrap_or("").to_string())
            .collect()
    };
    // Request 2 carries the approved call's result.
    let first = &tool_msgs(&bodies_now[1])[0];
    assert!(first.contains("[approved by the user: .openmax/tools/echoer.toml"), "{first}");
    assert!(first.contains("any edit to the manifest or its code asks again"), "{first}");
    // Request 3 carries the manifest edit's result + refreeze receipt.
    let second = &tool_msgs(&bodies_now[2])[1];
    assert!(second.contains("[extension refreeze:"), "{second}");
    assert!(
        second.contains("Modified tools whose current bytes no human has approved"),
        "a manifest edit of an approved tool is narrated as revoking: {second}"
    );
    assert!(second.contains("echoer (openmax --approve"), "{second}");
    let _ = std::fs::remove_dir_all(dir);
}

/// The script-only edit path: no fingerprint moves, no refreeze, and yet the
/// writing call's result must say the approval was revoked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_script_only_edit_announces_the_revocation_on_the_writing_call() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let script = project.join("echo.sh");
    std::fs::write(&script, "#!/bin/sh\ncat\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let manifest_path = project.join(".openmax/tools/echoer.toml");
    std::fs::write(&manifest_path, "name = \"echoer\"\ndescription = \"d\"\ncommand = \"./echo.sh\"\nmutating = false\n").unwrap();
    // The human approved manifest + script beforehand.
    std::fs::create_dir_all(&data).unwrap();
    let mut shas = vec![open_max_core::ledger::sha256_hex(&std::fs::read(&manifest_path).unwrap())];
    shas.extend(
        open_max_core::ledger::manifest_code(&manifest_path, &project)
            .into_iter()
            .filter_map(|c| c.sha256),
    );
    open_max_core::ledger::approve_capability(&data, &project, &manifest_path, &shas).unwrap();

    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": "echo.sh", "content": "#!/bin/sh\n# edited\ncat\n" }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "script-edit", &project, "edit the script").await;
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
        content.contains("[approval revoked:"),
        "a script edit must be narrated as revoking the tool's approval: {content}"
    );
    assert!(content.contains("echoer (openmax --approve"), "{content}");
    assert!(!content.contains("[extension refreeze:"), "no fingerprint moved: {content}");
    let _ = std::fs::remove_dir_all(dir);
}

/// If the approval cannot be recorded (the manifest changed while the card
/// was open), the receipt must NOT claim later calls run without a card -
/// the next call asks again (Greptile). It says the recording failed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unrecordable_approval_does_not_claim_cardless_future_calls() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let manifest = project.join(".openmax/tools/t.toml");
    std::fs::write(&manifest, "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n").unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        // The model calls t; the human approves; but between card and record
        // the manifest bytes change (a concurrent edit), so approve_capability
        // refuses (bytes on disk are not the vouched bytes).
        completion_with_tool_call("t", serde_json::json!({})),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    let manifest2 = manifest.clone();
    open_max_core::agent::start_turn(std::sync::Arc::clone(&core), "unrecordable".into(), project.clone(), "go".into()).unwrap();
    loop {
        let env = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv()).await.unwrap().unwrap();
        match env.event {
            AgentEvent::ApprovalRequest { approval_id, .. } => {
                // Change the manifest bytes before answering: the record will
                // refuse because disk != vouched.
                std::fs::write(&manifest2, "name = \"t\"\ndescription = \"CHANGED\"\ncommand = \"/bin/echo\"\n").unwrap();
                core.respond_approval(&approval_id, true);
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }
    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let tool = second["messages"].as_array().unwrap().iter()
        .find(|m| m["role"] == "tool").unwrap()["content"].as_str().unwrap();
    assert!(
        !tool.contains("run without a card"),
        "an unrecorded approval must not promise cardless calls: {tool}"
    );
    assert!(tool.contains("could not be recorded"), "{tool}");
    let _ = std::fs::remove_dir_all(dir);
}
