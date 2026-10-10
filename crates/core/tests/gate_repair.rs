//! A gate the agent can break stays repairable from inside the session.
//!
//! In auto the agent writes its own hooks, and the most natural mistakes
//! leave a gate that cannot start: a manifest written before its script, or a
//! script created by `write_file`, which does not set the executable bit. A
//! `pre_tool_use` gate in that state used to block every matching call,
//! including the writes that would fix it; a `user_prompt_submit` gate
//! refused every prompt, so no turn could start to repair it; and a blocking
//! `turn_end` gate sent its own spawn error back as a refusal eight times. A
//! gate whose script refuses everything locked the session the same way, and
//! two gates that cannot start each blocked the other's repair. Each case is
//! driven end to end here, turn by turn, in one session.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use open_max_core::agent::start_turn;
use open_max_core::state::Core;
use open_max_core::types::AgentEvent;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A scripted provider: the nth request gets the nth completion, and every
/// request body is kept.
async fn recording_endpoint(responses: Vec<serde_json::Value>) -> (String, Arc<Mutex<Vec<String>>>) {
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
                record.lock().unwrap().push(String::from_utf8_lossy(&buf[end + 4..]).to_string());
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

fn call(id: &str, name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": args.to_string() }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn text(text: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }]
    })
}

/// A trusted project in auto, the mode a newly trusted project starts in.
fn auto_project(name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("omx-gate-{name}-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join(".openmax/hooks")).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let project = project.canonicalize().unwrap();
    std::fs::write(
        data.join("trust.json"),
        serde_json::json!({ "version": 1, "projects": [project.to_string_lossy()] }).to_string(),
    )
    .unwrap();
    (dir, data, project)
}

fn write_settings(data: &Path, base_url: &str) {
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
}

/// A script as `write_file` leaves it: readable, not executable.
fn write_plain(path: &Path, body: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

fn write_exec(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    write_plain(path, body);
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[derive(Default)]
struct Turn {
    stop: String,
    /// (call id, ok, output) per finished tool call.
    calls: Vec<(String, bool, String)>,
    notes: Vec<String>,
    hook_failures: Vec<(String, String)>,
    errors: Vec<String>,
}

impl Turn {
    fn call(&self, id: &str) -> (bool, &str) {
        let (_, ok, output) = self
            .calls
            .iter()
            .find(|(call, _, _)| call == id)
            .unwrap_or_else(|| panic!("call {id} never finished: {:?}", self.calls));
        (*ok, output)
    }
}

async fn drive_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<open_max_core::types::AgentEventEnvelope>,
    session: &str,
    project: &Path,
    prompt: &str,
) -> Turn {
    start_turn(Arc::clone(core), session.into(), project.to_path_buf(), prompt.into()).unwrap();
    let mut turn = Turn::default();
    loop {
        let envelope = tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv())
            .await
            .expect("turn finishes within 30s")
            .expect("event channel stays open");
        match envelope.event {
            AgentEvent::ToolEnd { call_id, ok, output } => turn.calls.push((call_id, ok, output)),
            AgentEvent::HarnessNote { text, .. } => turn.notes.push(text),
            AgentEvent::HookFailed { event, detail, .. } => turn.hook_failures.push((event, detail)),
            AgentEvent::Error { message } => turn.errors.push(message),
            AgentEvent::Done { stop_reason } => {
                turn.stop = stop_reason;
                break;
            }
            _ => {}
        }
    }
    turn
}

/// The manifest is written first and the script it names after, the order an
/// author naturally works in, and the script comes from `write_file`, so it
/// is not executable. The gate cannot start, and it must not block its own
/// repair: the session fixes it and the gate then runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gate_that_cannot_start_never_blocks_the_writes_that_repair_it() {
    let (dir, data, project) = auto_project("unstartable");
    let script = "#!/bin/sh\ncat > ran.json\nexit 0\n";
    let (base_url, _) = recording_endpoint(vec![
        call(
            "manifest",
            "write_file",
            serde_json::json!({
                "path": ".openmax/hooks/guard.toml",
                "content": "event = \"pre_tool_use\"\ncommand = \"./scripts/guard.sh\"\n",
            }),
        ),
        text("hook written"),
        call("blocked", "bash", serde_json::json!({ "command": "echo hi" })),
        call("script", "write_file", serde_json::json!({ "path": "scripts/guard.sh", "content": script })),
        call(
            "interpreter",
            "write_file",
            serde_json::json!({
                "path": ".openmax/hooks/guard.toml",
                "content": "event = \"pre_tool_use\"\ncommand = \"sh\"\nargs = [\"./scripts/guard.sh\"]\n",
            }),
        ),
        text("repaired"),
        call("guarded", "bash", serde_json::json!({ "command": "echo hi" })),
        text("done"),
    ])
    .await;
    write_settings(&data, &base_url);
    let (core, mut rx) = Core::new(data).unwrap();

    let install = drive_turn(&core, &mut rx, "unstartable", &project, "add a guard hook").await;
    let repair = drive_turn(&core, &mut rx, "unstartable", &project, "say hi").await;
    let after = drive_turn(&core, &mut rx, "unstartable", &project, "say hi again").await;

    let (ok, output) = repair.call("script");
    assert!(ok, "the gate blocked writing its own script: {output}");
    let (ok, output) = repair.call("interpreter");
    assert!(ok, "the gate blocked rewriting its own manifest: {output}");
    let (ok, output) = repair.call("blocked");
    assert!(!ok, "a gate that cannot start still blocks the calls it matches");
    assert!(output.contains("failed to start hook"), "{output}");
    assert!(
        output.contains("command = \"sh\"") && output.contains("args = [\"./scripts/guard.sh\"]"),
        "the block names the repair: {output}"
    );
    let (ok, output) = after.call("guarded");
    assert!(ok && output.contains("hi"), "the repaired gate allows: {output}");
    let ran = std::fs::read_to_string(project.join("ran.json")).expect("the repaired gate ran");
    assert!(ran.contains("\"tool\":\"bash\""), "{ran}");

    // Named on the writing call, while the turn that wrote it can fix it.
    assert!(
        install.notes.iter().any(|n| n.contains("'guard' on pre_tool_use") && n.contains("does not exist")),
        "the manifest write names a gate that cannot start: {:?}",
        install.notes
    );
    assert!(
        repair.notes.iter().any(|n| n.contains("'guard' on pre_tool_use") && n.contains("not executable")),
        "the script write names the missing executable bit: {:?}",
        repair.notes
    );
    let last_note = repair.notes.last().expect("the manifest rewrite gets a note");
    assert!(!last_note.contains("cannot start"), "a runnable gate is not named: {last_note}");
    let _ = std::fs::remove_dir_all(dir);
}

/// Two gates that cannot start used to hold each other's repair shut: a write
/// to one gate's files skipped that gate and was blocked by the other. A gate
/// that cannot start judged nothing, so it stands between no hook file and its
/// repair, while every other call it matches stays blocked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_gates_that_cannot_start_never_block_each_others_repair() {
    let (dir, data, project) = auto_project("pair");
    for name in ["first", "second"] {
        std::fs::write(
            project.join(format!(".openmax/hooks/{name}.toml")),
            format!("event = \"pre_tool_use\"\ncommand = \"./scripts/{name}.sh\"\n"),
        )
        .unwrap();
    }
    let script = |name: &str| {
        serde_json::json!({ "path": format!("scripts/{name}.sh"), "content": "#!/bin/sh\ncat >/dev/null\nexit 0\n" })
    };
    let manifest = |name: &str| {
        serde_json::json!({
            "path": format!(".openmax/hooks/{name}.toml"),
            "content": format!("event = \"pre_tool_use\"\ncommand = \"sh\"\nargs = [\"./scripts/{name}.sh\"]\n"),
        })
    };
    let (base_url, _) = recording_endpoint(vec![
        call("other", "write_file", serde_json::json!({ "path": "notes.txt", "content": "x" })),
        call("first-script", "write_file", script("first")),
        call("first-manifest", "write_file", manifest("first")),
        call("second-script", "write_file", script("second")),
        call("second-manifest", "write_file", manifest("second")),
        call("still-blocked", "bash", serde_json::json!({ "command": "echo hi" })),
        text("repaired"),
        call("after", "bash", serde_json::json!({ "command": "echo hi" })),
        text("done"),
    ])
    .await;
    write_settings(&data, &base_url);
    let (core, mut rx) = Core::new(data).unwrap();

    let repair = drive_turn(&core, &mut rx, "pair", &project, "repair both guards").await;
    let after = drive_turn(&core, &mut rx, "pair", &project, "say hi").await;

    for id in ["first-script", "first-manifest", "second-script", "second-manifest"] {
        let (ok, output) = repair.call(id);
        assert!(ok, "{id} was blocked by a gate that cannot start: {output}");
    }
    for id in ["other", "still-blocked"] {
        let (ok, output) = repair.call(id);
        assert!(!ok && output.contains("failed to start hook"), "{id} still blocks: {output}");
    }
    let (ok, output) = after.call("after");
    assert!(ok && output.contains("hi"), "both repaired gates allow: {output}");
    let _ = std::fs::remove_dir_all(dir);
}

/// A gate whose script runs but refuses everything, a logic bug rather than a
/// missing file. It keeps enforcing on every other file, and its own manifest
/// and script stay writable, so the fix lands in the same turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gate_that_refuses_everything_still_lets_its_own_files_be_fixed() {
    let (dir, data, project) = auto_project("refuses");
    write_exec(&project.join("scripts/guard.sh"), "#!/bin/sh\necho blocked-by-bug\nexit 1\n");
    std::fs::write(
        project.join(".openmax/hooks/guard.toml"),
        "event = \"pre_tool_use\"\ncommand = \"./scripts/guard.sh\"\n",
    )
    .unwrap();
    let note = serde_json::json!({ "path": "notes.txt", "content": "x" });
    let (base_url, _) = recording_endpoint(vec![
        call("other", "write_file", note.clone()),
        call(
            "manifest",
            "edit_file",
            serde_json::json!({
                "path": ".openmax/hooks/guard.toml",
                "old_string": "command = \"./scripts/guard.sh\"",
                "new_string": "command = \"./scripts/guard.sh\"\ntimeout_secs = 5",
            }),
        ),
        call(
            "script",
            "write_file",
            serde_json::json!({ "path": "scripts/guard.sh", "content": "#!/bin/sh\ncat >/dev/null\nexit 0\n" }),
        ),
        call("fixed", "write_file", note),
        text("done"),
    ])
    .await;
    write_settings(&data, &base_url);
    let (core, mut rx) = Core::new(data).unwrap();

    let turn = drive_turn(&core, &mut rx, "refuses", &project, "fix the guard").await;

    let (ok, output) = turn.call("manifest");
    assert!(ok, "the gate blocked an edit to its own manifest: {output}");
    let (ok, output) = turn.call("script");
    assert!(ok, "the gate blocked a rewrite of its own script: {output}");
    let (ok, output) = turn.call("other");
    assert!(!ok && output.contains("blocked-by-bug"), "the gate still enforces elsewhere: {output}");
    let (ok, output) = turn.call("fixed");
    assert!(ok, "the rewritten script runs on the next call: {output}");
    let _ = std::fs::remove_dir_all(dir);
}

/// A `user_prompt_submit` gate that cannot start used to refuse every prompt,
/// so no turn could begin to repair it and only a human at a shell could. It
/// judged nothing, so the prompt goes through, the human and the model are
/// told it was not checked, and the agent repairs the hook in that turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prompt_gate_that_cannot_start_is_reported_instead_of_refusing_every_prompt() {
    let (dir, data, project) = auto_project("prompt");
    write_plain(&project.join("scripts/redact.sh"), "#!/bin/sh\ncat > seen.json\n");
    std::fs::write(
        project.join(".openmax/hooks/redact.toml"),
        "event = \"user_prompt_submit\"\ncommand = \"./scripts/redact.sh\"\n",
    )
    .unwrap();
    let (base_url, bodies) = recording_endpoint(vec![
        call(
            "interpreter",
            "edit_file",
            serde_json::json!({
                "path": ".openmax/hooks/redact.toml",
                "old_string": "command = \"./scripts/redact.sh\"",
                "new_string": "command = \"sh\"\nargs = [\"./scripts/redact.sh\"]",
            }),
        ),
        text("fixed"),
        text("checked"),
    ])
    .await;
    write_settings(&data, &base_url);
    let (core, mut rx) = Core::new(data).unwrap();

    let first = drive_turn(&core, &mut rx, "prompt", &project, "first prompt").await;
    assert_ne!(first.stop, "blocked", "{:?}", first.errors);
    assert!(
        first.hook_failures.iter().any(|(event, detail)| event == "user_prompt_submit"
            && detail.starts_with("did not check this prompt: failed to start hook")),
        "the human is told the prompt was not checked: {:?}",
        first.hook_failures
    );
    let request: serde_json::Value = serde_json::from_str(&bodies.lock().unwrap()[0]).unwrap();
    let said = request["messages"].as_array().unwrap().iter().any(|m| {
        m["role"] == "user"
            && m["content"]
                .as_str()
                .is_some_and(|c| c.contains("'redact' on user_prompt_submit: did not check this prompt"))
    });
    assert!(said, "the model is told which gate could not start: {request}");
    let (ok, output) = first.call("interpreter");
    assert!(ok, "{output}");

    let second = drive_turn(&core, &mut rx, "prompt", &project, "second prompt").await;
    assert_eq!(second.stop, "stop", "{:?}", second.errors);
    assert!(second.hook_failures.is_empty(), "{:?}", second.hook_failures);
    let seen = std::fs::read_to_string(project.join("seen.json")).expect("the repaired gate ran");
    assert!(seen.contains("second prompt"), "{seen}");
    let _ = std::fs::remove_dir_all(dir);
}

/// A blocking `turn_end` gate that cannot start used to answer each end of the
/// turn with its own spawn error, as the user, eight times before the harness
/// overrode it. Nothing in that loop could make it start, because hook edits
/// apply from the next turn. It is reported once and the turn ends
/// `unverified`, which is what the eight refusals ended in anyway.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_completion_gate_that_cannot_start_ends_the_turn_unverified_without_refusals() {
    let (dir, data, project) = auto_project("completion");
    write_plain(&project.join("scripts/gate.sh"), "#!/bin/sh\nexit 0\n");
    std::fs::write(
        project.join(".openmax/hooks/gate.toml"),
        "event = \"turn_end\"\ncommand = \"./scripts/gate.sh\"\nblocking = true\n",
    )
    .unwrap();
    let (base_url, bodies) = recording_endpoint(vec![text("4"); 12]).await;
    write_settings(&data, &base_url);
    let (core, mut rx) = Core::new(data).unwrap();

    let turn = drive_turn(&core, &mut rx, "completion", &project, "what is 2+2?").await;

    assert_eq!(bodies.lock().unwrap().len(), 1, "a gate that cannot start bought no extra requests");
    assert_eq!(turn.stop, "unverified", "an end no gate could judge is not verified");
    let failures: Vec<_> = turn.hook_failures.iter().filter(|(event, _)| event == "turn_end").collect();
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
        failures[0].1.contains("failed to start hook") && failures[0].1.contains("command = \"sh\""),
        "the report names the failure and the repair: {}",
        failures[0].1
    );
    let _ = std::fs::remove_dir_all(dir);
}
