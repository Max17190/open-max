//! `omit_tools` leaves named built-ins out of the tools array, and the prompt
//! then names only the tools on offer; unset, requests are unchanged. The
//! setting is process-wide (read when the core starts), so this suite stays
//! in its own test binary.

mod common;

use std::path::{Path, PathBuf};

use common::{drive_turn, recording_endpoint};
use open_max_core::state::Core;

const ALL_TOOLS_LINE: &str = "Tools: bash runs shell commands (ls, rg, git, tests; chain steps); read_file reads files; edit_file replaces exact text; write_file creates files.\n";
const NO_TYPED_EDITS_LINE: &str = "Tools: bash runs shell commands (ls, rg, git, tests; chain steps); read_file reads files.\n";

fn setup(omit: Option<serde_json::Value>, base_url: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("omx-profile-{}", uuid::Uuid::new_v4()));
    let (data, project) = (dir.join("data"), dir.join("project"));
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(project.join("a.txt"), "old\n").unwrap();
    let project = project.canonicalize().unwrap();
    let mut settings = serde_json::json!({ "base_url": base_url, "model": "scripted", "approval_mode": "auto", "context_tokens": 32768 });
    if let Some(omit) = omit {
        settings["omit_tools"] = omit;
    }
    std::fs::write(data.join("settings.json"), settings.to_string()).unwrap();
    let trust = serde_json::json!({ "version": 1, "projects": [project.to_string_lossy()] });
    std::fs::write(data.join("trust.json"), trust.to_string()).unwrap();
    (data, project)
}

fn edit_call() -> serde_json::Value {
    let args = serde_json::json!({ "path": "a.txt", "old_string": "old", "new_string": "new" });
    serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": "", "tool_calls": [{
        "id": "c1", "type": "function", "function": { "name": "edit_file", "arguments": args.to_string() }
    }] }, "finish_reason": "tool_calls" }] })
}

fn done() -> serde_json::Value {
    serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": "done" }, "finish_reason": "stop" }] })
}

/// Run one turn whose model calls edit_file once, and return the two request
/// bodies and the file's text after the turn.
async fn run(omit: Option<serde_json::Value>) -> (Vec<serde_json::Value>, String) {
    let (base_url, bodies) = recording_endpoint(vec![edit_call(), done()]).await;
    let (data, project) = setup(omit, &base_url);
    let (core, mut rx) = Core::new(data).unwrap();
    let id = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    drive_turn(&core, &mut rx, &id, &project, "change a.txt").await;
    let bodies = bodies.lock().unwrap().iter().map(|b| serde_json::from_str(b).unwrap()).collect();
    (bodies, std::fs::read_to_string(project.join(Path::new("a.txt"))).unwrap())
}

fn tool_names(body: &serde_json::Value) -> Vec<String> {
    body["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect()
}

fn prompt(body: &serde_json::Value) -> &str {
    body["messages"][0]["content"].as_str().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn omitted_builtins_leave_the_tools_array_and_the_prompt() {
    // Unset: all seven built-ins, the array byte-identical to the canonical
    // schemas, and the prompt's tool line as it has always been.
    let (sent, text) = run(None).await;
    assert_eq!(sent.len(), 2);
    assert_eq!(tool_names(&sent[0]), ["list_dir", "read_file", "write_file", "edit_file", "glob", "grep", "bash"]);
    assert_eq!(sent[0]["tools"].to_string(), open_max_core::tools::tool_schemas().to_string());
    assert!(prompt(&sent[0]).contains(ALL_TOOLS_LINE), "{}", prompt(&sent[0]));
    assert_eq!(text, "new\n", "edit_file ran");

    // Omitted: gone from the array and from the prompt, and a call to one is
    // refused as unknown without naming it as available.
    let (sent, text) = run(Some(serde_json::json!(["edit_file", "write_file"]))).await;
    assert_eq!(sent.len(), 2);
    for body in &sent {
        assert_eq!(tool_names(body), ["list_dir", "read_file", "glob", "grep", "bash"]);
        let prompt = prompt(body);
        assert!(prompt.contains(NO_TYPED_EDITS_LINE), "{prompt}");
        for name in ["edit_file", "write_file"] {
            assert!(!body["tools"].to_string().contains(name), "{name} in tools");
            assert!(!prompt.contains(name), "{name} in prompt: {prompt}");
        }
    }
    let reply = sent[1]["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().to_string();
    assert!(reply.contains("unknown tool: edit_file"), "{reply}");
    assert!(!reply.contains("write_file"), "{reply}");
    assert_eq!(text, "old\n", "edit_file did not run");

    // A name that is not a built-in fails at launch.
    let (data, _) = setup(Some(serde_json::json!(["edit_fiel"])), "http://127.0.0.1:9");
    let err = Core::new(data).err().expect("an unknown name is refused");
    assert!(err.contains("omit_tools: edit_fiel is not a built-in tool"), "{err}");
    open_max_core::tools::set_omitted_builtins(&[]).unwrap();
}
