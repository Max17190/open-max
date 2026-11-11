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

/// What a session in `mode` is told about files, from a project with a
/// memory, a skill and an `AGENTS.md` past its cap, whose model calls the
/// file tools anyway, from habit or its project's instructions: a read of a
/// directory, an edit with an empty `old_string`, an edit and a read that
/// work, and a tool no array sends. Then `AGENTS.md` changes and the session
/// resumes, and in `readonly` an edit is called again. Returns the system prompt, each call's
/// result by id, the note the change brought, the approval cards of the first
/// turn, and whether the readonly edit left the file alone.
async fn told(mode: &str) -> (String, BTreeMap<String, String>, String, usize, bool) {
    let (base_url, bodies) = recording_endpoint(vec![
        call("dir", "read_file", serde_json::json!({"path": "src"})),
        call("empty", "edit_file", serde_json::json!({"path": "src/lib.rs", "old_string": "", "new_string": "x"})),
        call("edit", "edit_file", serde_json::json!({"path": "src/lib.rs", "old_string": "needle", "new_string": "pin"})),
        call("read", "read_file", serde_json::json!({"path": "src/lib.rs"})),
        call("unknown", "view_file", serde_json::json!({"path": "src/lib.rs"})),
        text("done"),
        text("noted"),
        call("refused", "edit_file", serde_json::json!({"path": "src/lib.rs", "old_string": "pin", "new_string": "tack"})),
        text("explained"),
    ])
    .await;
    let (dir, data, project) = fixture(&base_url, mode);
    let write = |path: &str, body: &str| {
        std::fs::create_dir_all(project.join(path).parent().unwrap()).unwrap();
        std::fs::write(project.join(path), body).unwrap();
    };
    write(".openmax/memory/deploy-port.md", "# The deploy port is 7443\n");
    write(".agents/skills/release/SKILL.md", "---\nname: release\ndescription: cut a release\n---\nSteps.\n");
    write("AGENTS.md", &"Keep the build green.\n".repeat(120));
    let (core, mut rx) = Core::new(data.clone()).unwrap();
    let id = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
    let mut events = drive_turn(&core, &mut rx, &id, &project, "rename the needle").await;
    let cards = cards(&events);
    close(&core, &id).await;
    write("AGENTS.md", "Keep the build green and the diff small.\n");
    let (core, mut rx) = Core::new(data).unwrap();
    events.extend(drive_turn(&core, &mut rx, &id, &project, "and now").await);
    core.set_project_approval_mode(&project, ApprovalMode::Readonly).unwrap();
    events.extend(drive_turn(&core, &mut rx, &id, &project, "tack it").await);
    let untouched = std::fs::read_to_string(project.join("src/lib.rs")).unwrap() == "pub fn pin() {}\n";
    let results = events.iter().filter_map(|event| match event {
        AgentEvent::ToolEnd { call_id, output, .. } => Some((call_id.clone(), output.clone())),
        _ => None,
    }).collect();
    let bodies: Vec<serde_json::Value> = bodies.lock().unwrap().iter().map(|body| serde_json::from_str(body).unwrap()).collect();
    let system = bodies[0]["messages"][0]["content"].as_str().unwrap().to_string();
    let note = bodies[6]["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().to_string();
    let _ = std::fs::remove_dir_all(dir);
    (system, results, note, cards, untouched)
}

/// A session whose array sends `bash` alone is told the shell way wherever it
/// was told a file tool: the memory and skills headers, the `AGENTS.md`
/// truncation marker and change notice, a file tool's hints, and the tools an
/// unknown-tool error lists. The file tools it still calls run, gated by the
/// mode as usual: in `auto` without a card, in `readonly` refused. A session
/// that sends all seven is told what it always was, byte for byte.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_is_told_only_about_the_tools_its_array_sends() {
    let (system, results, note, cards, untouched) = told("auto").await;
    for (told, text) in [("the system prompt", &system), ("the note", &note)].into_iter().chain(results.iter().map(|(id, out)| (id.as_str(), out))) {
        for name in ["list_dir", "read_file", "write_file", "edit_file", "glob", "grep"] {
            assert!(!text.contains(name), "{told} names {name}, which the array leaves out:\n{text}");
        }
    }
    assert!(system.contains("\n\nMemory (facts saved by earlier turns or sessions; cat one before relying on it):\n"), "{system}");
    assert!(system.contains("\n\nSkills (before using one, read its SKILL.md with cat):\n"), "{system}");
    assert!(system.contains("…[AGENTS.md truncated; cat it for the rest]"), "{system}");
    assert!(note.contains("AGENTS.md changed after the system prompt was built: cat AGENTS.md for its current instructions."), "{note}");
    assert_eq!(results["dir"], "src is a directory; run ls on it to see its entries");
    assert_eq!(results["empty"], "old_string is empty; give the exact text to replace, or use a shell command to create or replace a whole file");
    assert!(results["edit"].contains("src/lib.rs") && results["read"].contains("pub fn pin()"), "{results:?}");
    assert_eq!(results["unknown"], "unknown tool: view_file; the available tools are bash");
    assert_eq!(cards, 0, "in auto the edit runs without a card");
    assert!(results["refused"].contains("read-only") && untouched, "in readonly the edit is refused: {results:?}");

    let (system, results, note, cards, _) = told("ask").await;
    assert!(system.contains("\n\nMemory (facts saved by earlier turns or sessions; read_file one before relying on it):\n"), "{system}");
    assert!(system.contains("\n\nSkills (before using one, read its SKILL.md. Use read_file for paths inside the project. For skill files outside the project (absolute paths), use bash: cat <path>.):\n"), "{system}");
    assert!(system.contains("…[AGENTS.md truncated; read_file it for the rest]"), "{system}");
    assert!(note.contains("AGENTS.md changed after the system prompt was built: read_file AGENTS.md for its current instructions."), "{note}");
    assert_eq!(results["dir"], "src is a directory; use list_dir to see its entries");
    assert_eq!(results["empty"], "old_string is empty; give the exact text to replace, or use write_file to create or replace a whole file");
    assert_eq!(results["unknown"], "unknown tool: view_file; the available tools are list_dir, read_file, write_file, edit_file, glob, grep, bash");
    assert!(cards > 0, "in ask the edits wait for approval");
}
