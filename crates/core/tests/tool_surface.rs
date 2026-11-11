//! The built-ins a session's tools array sends follow the mode its prompt
//! was built in. In `auto`, where `bash` runs without asking, the array sends
//! `bash` alone of them, since one shell command lists, searches, reads and
//! edits, several steps in one call. In `ask` and `readonly`, where `bash`
//! waits for approval or is refused, and under a permission rule that denies
//! `bash` outright, it sends all seven. The choice is made with the prompt
//! and holds with it: a mode change gates calls at once and moves no byte of
//! the array, a resume keeps it, and `/reload` makes it again.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::{close, drive_turn, recording_endpoint};
use open_max_core::config::ApprovalMode;
use open_max_core::state::Core;
use open_max_core::types::AgentEvent;
use serde_json::value::RawValue;

const ALL: [&str; 7] = ["list_dir", "read_file", "write_file", "edit_file", "glob", "grep", "bash"];

fn call(id: &str, name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
        "id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}}]},
        "finish_reason": "tool_calls"}]})
}

fn text(reply: &str) -> serde_json::Value {
    serde_json::json!({"choices": [{"message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]})
}

/// A trusted project with one file to search, and a data dir whose settings
/// point at `base_url` in `mode`.
fn fixture(base_url: &str, mode: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("omx-tool-surface-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(project.join("src/lib.rs"), "pub fn needle() {}\n").unwrap();
    let project = project.canonicalize().unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let settings = serde_json::json!({"base_url": base_url, "model": "scripted", "approval_mode": mode, "context_tokens": 32768});
    std::fs::write(data.join("settings.json"), settings.to_string()).unwrap();
    let trust = serde_json::json!({"version": 1, "projects": [project.to_string_lossy()]});
    std::fs::write(data.join("trust.json"), trust.to_string()).unwrap();
    (dir, data, project)
}

/// The base rules' tools line of a request body's system prompt.
fn tools_line(body: &str) -> String {
    let body: serde_json::Value = serde_json::from_str(body).unwrap();
    let system = body["messages"][0]["content"].as_str().unwrap();
    system.lines().find(|line| line.starts_with("Tools: ")).unwrap().to_string()
}

/// The raw `tools` bytes of a request body, and the names they carry.
fn tools(body: &str) -> (String, Vec<String>) {
    let fields = serde_json::from_str::<BTreeMap<String, Box<RawValue>>>(body).unwrap();
    let raw = fields["tools"].get().to_string();
    let names = serde_json::from_str::<Vec<serde_json::Value>>(&raw).unwrap().iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap().to_string()).collect();
    (raw, names)
}

/// The output of the call `id` among a turn's events, when it succeeded.
fn ran(events: &[AgentEvent], id: &str) -> Option<String> {
    events.iter().find_map(|event| match event {
        AgentEvent::ToolEnd { call_id, ok: true, output } if call_id == id => Some(output.clone()),
        _ => None,
    })
}

fn cards(events: &[AgentEvent]) -> usize {
    events.iter().filter(|event| matches!(event, AgentEvent::ApprovalRequest { .. })).count()
}

/// Both directions of a mode change. A session built in `auto` sends `bash`
/// alone, and a `grep` call, which the array leaves out, still runs. Switched
/// to `ask`, the model is told the mode by the execution policy note as
/// before, the next requests send the same array byte for byte, `bash` waits
/// for approval, and a `read_file` call still runs without asking; a resume
/// in a new process keeps the array. `/reload` builds the prompt and its
/// array again in `ask`, with all seven built-ins, and a resume after it
/// keeps all seven though the mode is `auto` again, where `bash` runs
/// without asking.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_array_follows_the_mode_its_prompt_was_built_in_until_a_new_prompt() {
    let (base_url, bodies) = recording_endpoint(vec![
        call("call-grep", "grep", serde_json::json!({"pattern": "needle"})),
        text("found"),
        call("call-ls", "bash", serde_json::json!({"command": "ls src"})),
        call("call-read", "read_file", serde_json::json!({"path": "src/lib.rs"})),
        text("read"),
        text("resumed"),
        text("reloaded"),
        call("call-ls-auto", "bash", serde_json::json!({"command": "ls src"})),
        text("resumed again"),
    ])
    .await;
    let (dir, data, project) = fixture(&base_url, "auto");
    let id = {
        let (core, mut rx) = Core::new(data.clone()).unwrap();
        let id = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
        let events = drive_turn(&core, &mut rx, &id, &project, "find the needle").await;
        assert!(ran(&events, "call-grep").is_some_and(|out| out.contains("src/lib.rs")), "grep ran: {events:?}");
        core.set_project_approval_mode(&project, ApprovalMode::Ask).unwrap();
        let events = drive_turn(&core, &mut rx, &id, &project, "and now").await;
        assert!(ran(&events, "call-ls").is_some_and(|out| out.contains("lib.rs")), "bash ran once approved: {events:?}");
        assert!(ran(&events, "call-read").is_some_and(|out| out.contains("needle")), "read_file ran: {events:?}");
        assert_eq!(cards(&events), 1, "in ask, bash waits for approval and read_file does not: {events:?}");
        close(&core, &id).await;
        id
    };
    {
        let (core, mut rx) = Core::new(data.clone()).unwrap();
        drive_turn(&core, &mut rx, &id, &project, "resumed").await;
        open_max_core::agent::reload_session(&core, &id, &project).await.unwrap();
        drive_turn(&core, &mut rx, &id, &project, "reloaded").await;
        close(&core, &id).await;
    }
    let (core, mut rx) = Core::new(data).unwrap();
    core.set_project_approval_mode(&project, ApprovalMode::Auto).unwrap();
    let events = drive_turn(&core, &mut rx, &id, &project, "resumed again").await;
    assert!(ran(&events, "call-ls-auto").is_some(), "bash ran: {events:?}");
    assert_eq!(cards(&events), 0, "in auto, bash runs without asking: {events:?}");

    let bodies: Vec<String> = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 9, "one request per scripted reply");
    let (first, names) = tools(&bodies[0]);
    assert_eq!(names, ["bash"], "a session built in auto sends bash alone of the built-ins");
    for (n, body) in bodies.iter().enumerate().take(6).skip(1) {
        assert_eq!(tools(body).0, first, "request {n} sends the array the session's prompt was built with");
    }
    assert!(bodies[2].contains("[execution policy: ask;"), "the model is told the mode changed");
    assert_eq!(tools(&bodies[6]).1, ALL, "/reload in ask builds an array with all seven built-ins");
    for n in [7, 8] {
        assert_eq!(tools(&bodies[n]).0, tools(&bodies[6]).0, "request {n}, resumed in auto, keeps it");
    }
    // The prompt's tools line names what the array sends, and holds with it.
    let (auto, full) = (tools_line(&bodies[0]), tools_line(&bodies[6]));
    for name in ALL {
        assert_eq!(auto.contains(name), name == "bash", "{name} in the auto line {auto:?}");
    }
    assert!(auto.contains("edits"), "bash makes the edits there: {auto}");
    assert!(full.contains("read_file") && full.contains("edit_file") && full.contains("write_file"), "{full}");
    assert_eq!(tools_line(&bodies[5]), auto, "the resumed session keeps its prompt");
    assert_eq!(tools_line(&bodies[8]), full, "and so does the one resumed after /reload");
    let _ = std::fs::remove_dir_all(dir);
}

/// One request of a new session in `mode`, with `rules` as the project's
/// permission file when given and the project held at `held` when given, as
/// a sub-agent's worktree is: the built-in names its tools array sends.
async fn first_request_names(mode: &str, rules: Option<&str>, held: Option<ApprovalMode>) -> Vec<String> {
    let (base_url, bodies) = recording_endpoint(vec![text("ok")]).await;
    let (dir, data, project) = fixture(&base_url, mode);
    if let Some(rules) = rules {
        std::fs::create_dir_all(project.join(".openmax")).unwrap();
        std::fs::write(project.join(".openmax/permissions.toml"), rules).unwrap();
    }
    let (core, mut rx) = Core::new(data).unwrap();
    if let Some(held) = held {
        core.hold_approval_mode(&project, held);
    }
    let id = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    drive_turn(&core, &mut rx, &id, &project, "hello").await;
    let body = bodies.lock().unwrap()[0].clone();
    let _ = std::fs::remove_dir_all(dir);
    tools(&body).1
}

/// Where `bash` cannot run without asking, the typed tools stay: `ask` would
/// card every shell read, `readonly` refuses `bash`, and a rule that denies
/// `bash` outright leaves nothing to run them with. A project held at a
/// stricter mode, as a sub-agent's worktree is, builds its prompt in that
/// mode. A rule that denies some commands leaves the shell to cover them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ask_readonly_and_a_denied_shell_send_all_seven_built_ins() {
    assert_eq!(first_request_names("auto", None, None).await, ["bash"], "auto");
    assert_eq!(first_request_names("ask", None, None).await, ALL, "ask");
    assert_eq!(first_request_names("readonly", None, None).await, ALL, "readonly");
    assert_eq!(first_request_names("auto", None, Some(ApprovalMode::Ask)).await, ALL, "auto held at ask");
    let deny_all = "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\n";
    assert_eq!(first_request_names("auto", Some(deny_all), None).await, ALL, "auto with every command denied");
    let deny_any = "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \".\"\n";
    assert_eq!(first_request_names("auto", Some(deny_any), None).await, ALL, "auto with every non-empty command denied");
    let deny_some = "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \"rm\\\\s+-rf\"\n";
    assert_eq!(first_request_names("auto", Some(deny_some), None).await, ["bash"], "auto with some commands denied");
}
