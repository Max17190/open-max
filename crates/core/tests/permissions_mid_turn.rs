//! A permission rule written by an iteration's mutating call is in force for
//! the very next tool call of the same turn.
//!
//! The failure this guards against is not hypothetical: "install a guard,
//! then prove it works" is the natural shape of the task, and with per-turn
//! discovery the proof runs unguarded - the eval's weak-model runs deleted
//! the directory they had just written a deny rule for, four times in one
//! turn, because the rule was never going to be live until the turn ended.

use std::path::PathBuf;
use std::sync::Arc;

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::AgentEvent;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A completions endpoint that answers each request with the next scripted
/// message. Plain JSON responses (no SSE): the client parses those in one
/// shot, which keeps the script byte-obvious.
async fn scripted_endpoint(responses: Vec<serde_json::Value>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        for body in responses {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            // Drain the request: headers, then content-length bytes of body.
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            let (mut need, mut have) = (usize::MAX, 0usize);
            loop {
                let Ok(n) = sock.read(&mut chunk).await else { return };
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if need == usize::MAX {
                    if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                        let len = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        need = len;
                        have = buf.len() - (end + 4);
                    }
                }
                if need != usize::MAX && have >= need {
                    break;
                }
                if need != usize::MAX {
                    have = buf.len();
                }
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
    format!("http://{addr}")
}

fn completion_with_tool_calls(calls: &[(&str, serde_json::Value)]) -> serde_json::Value {
    let tool_calls: Vec<serde_json::Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (name, args))| {
            serde_json::json!({
                "id": format!("call-{i}-{name}"),
                "type": "function",
                "function": { "name": name, "arguments": args.to_string() }
            })
        })
        .collect();
    serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": "", "tool_calls": tool_calls },
            "finish_reason": "tool_calls"
        }]
    })
}

fn completion_with_tool_call(name: &str, args: serde_json::Value) -> serde_json::Value {
    let mut completion = completion_with_tool_calls(&[(name, args)]);
    completion["choices"][0]["message"]["tool_calls"][0]["id"] =
        serde_json::json!(format!("call-{name}"));
    completion
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
async fn a_deny_rule_written_this_turn_gates_the_same_turn() {
    let dir = std::env::temp_dir().join(format!("omx-midturn-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("reports")).unwrap();
    std::fs::write(project.join("reports/q0.md"), "# deliverable\n").unwrap();
    let project = project.canonicalize().unwrap();

    // The model's script: write the deny rule, then immediately try the very
    // command the rule is about, then stop.
    let rule = "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \"rm\\\\s+.*reports\"\n";
    let base_url = scripted_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/permissions.toml", "content": rule }),
        ),
        completion_with_tool_call("bash", serde_json::json!({ "command": "rm -rf reports" })),
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
    start_turn(Arc::clone(&core), "midturn-test".into(), PathBuf::from(&project), "go".into())
        .unwrap();

    let mut bash_outputs = Vec::new();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match envelope.event {
            AgentEvent::ToolEnd { ok, output, call_id } => {
                if call_id == "call-bash" {
                    bash_outputs.push((ok, output));
                }
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }

    assert!(
        project.join("reports/q0.md").exists(),
        "the deny rule written this turn must gate the deletion attempted this turn"
    );
    let (ok, output) = bash_outputs.first().expect("the scripted rm call ran through the gate");
    assert!(!ok, "the rm call must be refused, got: {output}");
    assert!(
        output.contains("permission rule denied"),
        "the refusal names the permission gate: {output}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_deny_live_at_turn_start_survives_its_own_removal_for_the_turn() {
    // The reload composes one-directionally: the same mechanism that puts a
    // fresh deny in force mid-turn must not lift a deny the turn started
    // under when a mutating call rewrites the file to drop it. The lift
    // arrives with the next turn.
    let dir = std::env::temp_dir().join(format!("omx-midturn-floor-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("reports")).unwrap();
    std::fs::write(project.join("reports/q0.md"), "# deliverable\n").unwrap();
    std::fs::create_dir_all(project.join(".openmax")).unwrap();
    std::fs::write(
        project.join(".openmax/permissions.toml"),
        "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \"rm\\\\s+.*reports\"\n",
    )
    .unwrap();
    let project = project.canonicalize().unwrap();

    // The model's script: gut the policy file, then run the command the
    // turn-start policy denies.
    let base_url = scripted_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/permissions.toml", "content": "" }),
        ),
        completion_with_tool_call("bash", serde_json::json!({ "command": "rm -rf reports" })),
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
    start_turn(Arc::clone(&core), "floor-test".into(), PathBuf::from(&project), "go".into())
        .unwrap();
    let mut bash_outputs = Vec::new();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match envelope.event {
            AgentEvent::ToolEnd { ok, output, call_id } => {
                if call_id == "call-bash" {
                    bash_outputs.push((ok, output));
                }
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }

    assert!(
        project.join("reports/q0.md").exists(),
        "a turn-start deny must survive its own removal until the turn ends"
    );
    let (ok, output) = bash_outputs.first().expect("the scripted rm call ran through the gate");
    assert!(!ok, "the rm call must still be refused, got: {output}");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_deny_added_mid_turn_survives_its_removal_for_the_turn() {
    // The ratchet covers every snapshot the turn observed, not only the
    // turn-start one: a deny that appeared mid-turn (a human editing the
    // file while the agent runs) must not vanish because a later mutation
    // rewrote the file without it.
    let dir = std::env::temp_dir().join(format!("omx-midturn-ratchet-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("reports")).unwrap();
    std::fs::write(project.join("reports/q0.md"), "# deliverable\n").unwrap();
    let project = project.canonicalize().unwrap();

    // The turn starts with no policy at all; the script installs a deny,
    // removes it again, then runs the command the mid-turn deny named.
    let rule = "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \"rm\\\\s+.*reports\"\n";
    let base_url = scripted_endpoint(vec![
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/permissions.toml", "content": rule }),
        ),
        completion_with_tool_call(
            "write_file",
            serde_json::json!({ "path": ".openmax/permissions.toml", "content": "" }),
        ),
        completion_with_tool_call("bash", serde_json::json!({ "command": "rm -rf reports" })),
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
    start_turn(Arc::clone(&core), "ratchet-test".into(), PathBuf::from(&project), "go".into())
        .unwrap();
    let mut bash_outputs = Vec::new();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match envelope.event {
            AgentEvent::ToolEnd { ok, output, call_id } => {
                if call_id == "call-bash" {
                    bash_outputs.push((ok, output));
                }
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }

    assert!(
        project.join("reports/q0.md").exists(),
        "a deny observed mid-turn must survive its removal until the turn ends"
    );
    let (ok, output) = bash_outputs.first().expect("the scripted rm call ran through the gate");
    assert!(!ok, "the rm call must still be refused, got: {output}");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_deny_written_earlier_in_the_same_response_gates_the_next_call() {
    // One assistant response can carry the policy write and the destructive
    // call together; the serial loop must reload after the mutating call so
    // the later call in the same response already sees the rule.
    let dir = std::env::temp_dir().join(format!("omx-midturn-batch-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("reports")).unwrap();
    std::fs::write(project.join("reports/q0.md"), "# deliverable\n").unwrap();
    let project = project.canonicalize().unwrap();

    let rule = "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \"rm\\\\s+.*reports\"\n";
    let base_url = scripted_endpoint(vec![
        completion_with_tool_calls(&[
            (
                "write_file",
                serde_json::json!({ "path": ".openmax/permissions.toml", "content": rule }),
            ),
            ("bash", serde_json::json!({ "command": "rm -rf reports" })),
        ]),
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
    start_turn(Arc::clone(&core), "batch-test".into(), PathBuf::from(&project), "go".into())
        .unwrap();
    let mut bash_outputs = Vec::new();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match envelope.event {
            AgentEvent::ToolEnd { ok, output, call_id } => {
                if call_id.contains("bash") {
                    bash_outputs.push((ok, output));
                }
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }

    assert!(
        project.join("reports/q0.md").exists(),
        "a deny written earlier in the same response must gate the next call"
    );
    let (ok, output) = bash_outputs.first().expect("the scripted rm call ran through the gate");
    assert!(!ok, "the rm call must be refused, got: {output}");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_policy_written_by_a_failed_mutation_still_gates() {
    // A bash command can persist the policy file and still exit nonzero; the
    // reload must follow every executed mutating call, not only successful
    // ones, or the persisted deny goes unobserved for the rest of the turn.
    let dir = std::env::temp_dir().join(format!("omx-midturn-failed-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("reports")).unwrap();
    std::fs::write(project.join("reports/q0.md"), "# deliverable\n").unwrap();
    let project = project.canonicalize().unwrap();

    let write_then_fail = "mkdir -p .openmax && printf '[[rules]]\\neffect = \"deny\"\\ntool = \"bash\"\\narg_regex = \"rm\\\\\\\\s+.*reports\"\\n' > .openmax/permissions.toml && exit 7";
    let base_url = scripted_endpoint(vec![
        completion_with_tool_call("bash", serde_json::json!({ "command": write_then_fail })),
        completion_with_tool_call("bash", serde_json::json!({ "command": "rm -rf reports" })),
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
    start_turn(Arc::clone(&core), "failed-write-test".into(), PathBuf::from(&project), "go".into())
        .unwrap();
    let mut rm_outputs = Vec::new();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match envelope.event {
            AgentEvent::ToolStart { call_id, args, .. } => {
                if args.get("command").and_then(|c| c.as_str()) == Some("rm -rf reports") {
                    rm_outputs.push(call_id);
                }
            }
            AgentEvent::ToolEnd { ok, output, call_id } => {
                if rm_outputs.contains(&call_id) {
                    assert!(!ok, "the rm call must be refused, got: {output}");
                }
            }
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }

    assert!(!rm_outputs.is_empty(), "the scripted rm call ran through the gate");
    assert!(
        project.join("reports/q0.md").exists(),
        "a policy persisted by a failed mutation must still gate the next call"
    );
    let _ = std::fs::remove_dir_all(dir);
}
