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
            "context_tokens": 16384,
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
            "context_tokens": 16384,
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
/// first call stops for a human. A receipt that says "callable" and is
/// followed immediately by an approval card is dishonest, and a model reading
/// it will say so. Approved bytes keep "callable".
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
/// the next call asks again. It says the recording failed.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_memory_write_refreezes_and_is_indexed_for_the_next_step() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({
                "path": ".openmax/memory/rotation-interval.md",
                "content": "# archive rotation interval is 17 days\n\nSeen in corpus/doc04.txt.\n"
            }),
        ),
        completion_with_text("noted"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "memory-live", &project, "remember it").await;
    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let messages = second["messages"].as_array().unwrap();
    let tool = messages.iter().find(|m| m["role"] == "tool").unwrap()["content"].as_str().unwrap();
    assert!(tool.contains("[extension refreeze:"), "the memory write refreezes: {tool}");
    assert!(
        tool.contains("Memory index indexed: rotation-interval"),
        "the receipt names the newly indexed memory: {tool}"
    );
    assert!(tool.contains("live in your prompt from your next step"), "{tool}");
    let system = messages[0]["content"].as_str().unwrap();
    assert!(
        system.contains("rotation-interval: archive rotation interval is 17 days"),
        "the next request's frozen prompt carries the index line: {system}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A memory file the prompt index REJECTS (bad stem) must not be claimed
/// live. Before this fix, memory_files listed every readable .md, so writing
/// `.openmax/memory/Invalid-Stem.md` refroze and told the model
/// "Memory index indexed: Invalid-Stem" while the next prompt had no such
/// entry. The receipt is now built from the indexed selection.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rejected_memory_name_is_never_claimed_indexed() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({
                "path": ".openmax/memory/Invalid-Stem.md",
                "content": "# a fact\n\nbody\n"
            }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "bad-mem", &project, "remember badly").await;
    let bodies = bodies.lock().unwrap();
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let messages = second["messages"].as_array().unwrap();
    for m in messages {
        if let Some(c) = m["content"].as_str() {
            assert!(
                !c.contains("Memory index indexed: Invalid-Stem"),
                "a rejected name must not be claimed indexed: {c}"
            );
        }
    }
    // And the frozen prompt carries no Invalid-Stem index line.
    let system = messages[0]["content"].as_str().unwrap();
    assert!(!system.contains("Invalid-Stem"), "the prompt omits it: {system}");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn removing_an_approved_tool_says_the_approval_outlives_it() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let manifest_path = project.join(".openmax/tools/echoer.toml");
    std::fs::write(&manifest_path, "name = \"echoer\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n").unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let sha = open_max_core::ledger::sha256_hex(&std::fs::read(&manifest_path).unwrap());
    open_max_core::ledger::approve_capability(&data, &project, &manifest_path, &[sha]).unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "bash",
            serde_json::json!({ "command": "rm .openmax/tools/echoer.toml" }),
        ),
        completion_with_text("gone"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "remove-approved", &project, "remove it").await;
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
    assert!(content.contains("Removed approved tools: echoer"), "{content}");
    assert!(content.contains("the approval outlives the file"), "{content}");
    assert!(content.contains("nothing needs forgetting"), "{content}");
    let _ = std::fs::remove_dir_all(dir);
}

/// Deleting a tool whose SCRIPT was edited (manifest unchanged, but the
/// bound code no longer matches what was approved) must NOT be classified
/// as a removed-approved tool: the runtime gate would ask again, so the
/// receipt cannot claim 'would run without a card'.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn removing_a_tool_with_edited_bound_code_is_not_called_approved() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let script = project.join("run.sh");
    std::fs::write(&script, "#!/bin/sh\ncat\n").unwrap();
    let manifest = project.join(".openmax/tools/t.toml");
    std::fs::write(&manifest, "name = \"t\"\ndescription = \"d\"\ncommand = \"./run.sh\"\n").unwrap();
    std::fs::create_dir_all(&data).unwrap();
    // Approve manifest + the ORIGINAL script.
    let mut shas = vec![open_max_core::ledger::sha256_hex(&std::fs::read(&manifest).unwrap())];
    shas.extend(open_max_core::ledger::manifest_code(&manifest, &project).into_iter().filter_map(|c| c.sha256));
    open_max_core::ledger::approve_capability(&data, &project, &manifest, &shas).unwrap();
    // Edit the script AND delete the manifest in one turn.
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call("write_file", serde_json::json!({ "path": "run.sh", "content": "#!/bin/sh\n# edited\ncat\n" })),
        completion_with_tool_call("bash", serde_json::json!({ "command": "rm .openmax/tools/t.toml" })),
        completion_with_text("gone"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "edit-then-remove", &project, "go").await;
    let bodies = bodies.lock().unwrap();
    // The removal receipt (request 3) must not name t as removed-approved.
    let third: serde_json::Value = serde_json::from_str(&bodies[2]).unwrap();
    for m in third["messages"].as_array().unwrap() {
        if let Some(c) = m["content"].as_str() {
            assert!(
                !c.contains("Removed approved tools: t"),
                "a tool with edited bound code is not approved: {c}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Deleting BOTH an approved manifest and its bound script in one turn must
/// still report that the approval SURVIVES: the ledger keeps the record and
/// the content-addressed objects, so the bytes are restorable. Recomputing
/// bound_code after the script is gone makes covers_code fail, which used to
/// drop the tool from the receipt entirely, hiding the surviving approval
///. It is now named in a distinct clause that does not claim the
/// deleted bytes would run without another approval.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn removing_an_approved_tool_and_its_script_reports_the_surviving_approval() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let script = project.join("run.sh");
    std::fs::write(&script, "#!/bin/sh\ncat\n").unwrap();
    let manifest = project.join(".openmax/tools/t.toml");
    std::fs::write(&manifest, "name = \"t\"\ndescription = \"d\"\ncommand = \"./run.sh\"\n").unwrap();
    std::fs::create_dir_all(&data).unwrap();
    // Approve manifest + the script it runs.
    let mut shas = vec![open_max_core::ledger::sha256_hex(&std::fs::read(&manifest).unwrap())];
    shas.extend(open_max_core::ledger::manifest_code(&manifest, &project).into_iter().filter_map(|c| c.sha256));
    open_max_core::ledger::approve_capability(&data, &project, &manifest, &shas).unwrap();
    // Delete BOTH the manifest and the script in one turn.
    let (base_url, bodies) = recording_endpoint(vec![
        completion_with_tool_call("bash", serde_json::json!({ "command": "rm .openmax/tools/t.toml run.sh" })),
        completion_with_text("gone"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, "remove-both", &project, "go").await;
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
        content.contains("surviving approval: t"),
        "the surviving approval must be reported, not hidden: {content}"
    );
    // ...and it must NOT claim the deleted bytes run without a card.
    assert!(
        !content.contains("Removed approved tools: t"),
        "deleted code cannot be called cardless-runnable: {content}"
    );
    // Restorability is qualified, not promised: a legacy or hash-only approval
    // keeps the sha without storing objects, so the claim is "may".
    assert!(
        content.contains("may run without a card"),
        "the receipt must not overstate object restorability: {content}"
    );
    assert!(
        !content.contains("objects remain in the ledger (restorable)"),
        "the unconditional restorable claim was removed: {content}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The receipt the model reads also reaches the WIRE as a harness_note, so a
/// custom frontend can render what the model sees. Without it the tool_end
/// output is bare and the NOT-loaded receipt is invisible off-transcript.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_harness_note_reaches_the_wire_for_a_broken_tool_write() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();
    let broken = "name = \"wf\"\ndescription = \"d\"\n"; // missing command
    let (base_url, _bodies) = recording_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/wf.toml", "content": broken }),
        ),
        completion_with_text("done"),
    ])
    .await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data).unwrap();
    open_max_core::agent::start_turn(std::sync::Arc::clone(&core), "wire-note".into(), project.clone(), "go".into()).unwrap();
    let mut note = None;
    loop {
        let env = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await.expect("30s").expect("open");
        match env.event {
            AgentEvent::HarnessNote { text, .. } => {
                if text.contains("NOT loaded") { note = Some(text); }
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }
    let note = note.expect("the NOT-loaded receipt reaches the wire as a harness_note");
    assert!(note.contains("wf.toml"), "{note}");
    let _ = std::fs::remove_dir_all(dir);
}
/// A turn-start receipt reaches the WIRE as a harness_note, not only the model
/// transcript: a protocol-v5 or interactive frontend must be able to display
/// the state change that affects the next turn. Proven through the
/// out-of-session approval path - a human approving at another terminal
/// between turns - which is one of the three turn-start notes that used to be
/// transcript-only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_start_approval_note_reaches_the_wire_as_a_harness_note() {
    let dir = std::env::temp_dir().join(format!("omx-receipt-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    let project = project.canonicalize().unwrap();
    let manifest = project.join(".openmax/tools/x.toml");
    std::fs::write(&manifest, "name = \"x\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n").unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let (base_url, _bodies) =
        recording_endpoint(vec![completion_with_text("one"), completion_with_text("two")]).await;
    write_config(&data, &base_url, &project);
    let (core, mut rx) = Core::new(data.clone()).unwrap();

    // Turn 1: a plain turn that seeds the session's seen-events watermark.
    drive_turn(&core, &mut rx, "wire-approve", &project, "hi").await;

    // A human approves the tool at ANOTHER terminal (actor External: no
    // session_id), landing a ledger event this session has not seen.
    let sha = open_max_core::ledger::sha256_hex(&std::fs::read(&manifest).unwrap());
    open_max_core::ledger::approve_capability(&data, &project, &manifest, &[sha]).unwrap();

    // Turn 2: the turn-start out-of-session note must reach the wire.
    open_max_core::agent::start_turn(
        std::sync::Arc::clone(&core),
        "wire-approve".into(),
        project.clone(),
        "again".into(),
    )
    .unwrap();
    let mut note = None;
    loop {
        let env = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("30s")
            .expect("open");
        match env.event {
            AgentEvent::HarnessNote { text, .. } => {
                if text.contains("approval activity this session did not perform") {
                    note = Some(text);
                }
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }
    let note = note.expect("the out-of-session approval note reaches the wire as a harness_note");
    assert!(note.contains("x.toml"), "{note}");
    let _ = std::fs::remove_dir_all(dir);
}
