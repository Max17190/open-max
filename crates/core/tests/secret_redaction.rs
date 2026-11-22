//! A credential in the harness's environment never reaches the model through
//! a tool result.
//!
//! The agent's shell inherits the user's environment, which a sub-agent
//! started from bash needs, so one `env` prints every key there. The values
//! are redacted where the result enters the transcript: this drives a real
//! turn whose bash call runs `env` and checks what the next request sends.
#![allow(clippy::disallowed_methods)]

mod common;

use open_max_core::state::Core;
use open_max_core::types::AgentEvent;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn env_in_bash_does_not_send_provider_keys_to_the_model() {
    // A key the settings name by indirection (its variable's name does not
    // look secret), and a well-known kind of provider key the harness does
    // not use itself. Unique per run, so a match can only be this test's.
    let configured = format!("cfg-{}", uuid::Uuid::new_v4());
    let ambient = format!("sk-{}", uuid::Uuid::new_v4());
    std::env::set_var("OMX_REDACTION_PROBE", &configured);
    std::env::set_var("OMX_REDACTION_PROBE_API_KEY", &ambient);

    let dir = std::env::temp_dir().join(format!("omx-redaction-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let project = project.canonicalize().unwrap();

    let (base_url, bodies) = common::recording_endpoint(vec![
        serde_json::json!({
            "choices": [{
                "message": { "role": "assistant", "content": "", "tool_calls": [{
                    "id": "call-env", "type": "function",
                    "function": { "name": "bash", "arguments": serde_json::json!({ "command": "env" }).to_string() }
                }] },
                "finish_reason": "tool_calls"
            }]
        }),
        serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": "done" }, "finish_reason": "stop" }] }),
    ])
    .await;
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "api_key": "$OMX_REDACTION_PROBE",
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
    let events = common::drive_turn(&core, &mut rx, "redaction", &project, "go").await;
    let output = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolEnd { call_id, output, .. } if call_id == "call-env" => Some(output.clone()),
            _ => None,
        })
        .expect("the env call ran");

    // The command saw both variables; what came back names them, not their values.
    assert!(output.contains("OMX_REDACTION_PROBE=[redacted settings api_key]"), "{output}");
    assert!(output.contains("OMX_REDACTION_PROBE_API_KEY=[redacted OMX_REDACTION_PROBE_API_KEY]"), "{output}");
    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2, "the tool result went back to the endpoint");
    assert!(bodies[1].contains("[redacted OMX_REDACTION_PROBE_API_KEY]"), "the result reached the model: {}", bodies[1]);
    for body in &bodies {
        assert!(!body.contains(&configured), "the configured key reached the model");
        assert!(!body.contains(&ambient), "the ambient provider key reached the model");
    }
    drop(core);
    let _ = std::fs::remove_dir_all(dir);
}
