//! Thesis gates for the agent loop: one test per target, named after it.
//!
//! Byte and count gates assert unconditionally. Each test's doc comment
//! states what it measures and how.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::{close, drive_turn, note_marker, recording_endpoint};
use open_max_core::prompt::NOTE_HEADER;
use open_max_core::state::Core;
use serde_json::value::RawValue;

/// One message's raw JSON with the trailing note cut out of its content:
/// from the note's escaped marker to the quote that closes that string, so
/// every other byte stays as sent.
fn without_note(raw: &str, marker: &str) -> String {
    let escaped_marker = serde_json::to_string(marker).unwrap();
    let escaped_marker = &escaped_marker[1..escaped_marker.len() - 1];
    let at = raw.find(escaped_marker).expect("the message carries the note");
    let rest = &raw[at..];
    let mut escaped = false;
    let end = rest
        .char_indices()
        .find(|&(_, c)| {
            let closes = c == '"' && !escaped;
            escaped = c == '\\' && !escaped;
            closes
        })
        .map(|(i, _)| i)
        .expect("the note ends inside a JSON string");
    format!("{}{}", &raw[..at], &rest[end..])
}

/// The bytes `now` differs from `then` by: each position that holds another
/// byte, plus the difference in length.
fn changed_bytes(then: &str, now: &str) -> usize {
    then.bytes().zip(now.bytes()).filter(|(a, b)| a != b).count() + then.len().abs_diff(now.len())
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

fn text(reply: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": reply },
            "finish_reason": "stop"
        }]
    })
}

/// This repository's top two levels as its layout map showed them (hidden
/// entries aside) when the prefix came under 700 tokens: the fixture of the
/// frozen prefix gate, written out so the gate never reads the live checkout.
/// A directory ends with a slash.
const REPOSITORY_TREE: [&str; 17] = [
    "AGENTS.example.md",
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE",
    "README.md",
    "crates/core/",
    "crates/tui/",
    "docs/assets/",
    "docs/configuration.md",
    "docs/extending.md",
    "docs/stdio-protocol.md",
    "docs/usage.md",
    "scripts/check-binary-size.sh",
    "scripts/check-suite-time.sh",
    "scripts/patch-installer-checksum.sh",
    "scripts/patch-installer-first-command.sh",
    "scripts/release.sh",
];

/// The frozen prefix stays under 700 tokens in a project with this
/// repository's top two levels as `REPOSITORY_TREE` records them, checked as a
/// byte cap. The prefix is what every request of a session sends first:
/// messages[0] (the base rules, the extension pointer, the project path, and
/// the layout map) and the `tools` array, what a provider's prompt cache keys
/// on, then the execution policy note the history resends after them. Method: a
/// scripted endpoint records request 0 of a turn in a project trusted in `auto`
/// whose tree is `REPOSITORY_TREE`, without an `AGENTS.md` (in a checkout that
/// file is the user's own, and git ignores it here); the gate takes the text of
/// messages[0] less the fixture's project path, adds the `tools` array's raw
/// wire bytes and the note's text, and holds that path-free payload to 3,036
/// bytes: 700 tokens less a 10-token allowance for the checkout path, at 4.4
/// bytes per token. The path is left out because the fixture's sits in a
/// uuid-named temp dir, whose cost depends on the machine. A checkout of this
/// repository costs 9 tokens in place, in the prompt's first sentence, on
/// either CI runner (/home/runner/work/open-max/open-max,
/// /Users/runner/work/open-max/open-max), and 5 to 18 over ten home-directory
/// checkouts (/Users/alice/code/open-max costs 7, a 70-byte synced-folder path
/// 18). Tokens are counted with tiktoken's o200k_base on each field's own text,
/// messages[0], the tools array and the note, and summed. On request 0 in this
/// repository, with the tree recorded here, the path-free payload was 3,126
/// bytes and 716 tokens before the layout map named each directory once (725
/// with a 9-token checkout path), and 3,022 bytes and 678 tokens after (687),
/// so a checkout path of up to 22 tokens keeps it within 700. The map's section
/// went from 407 bytes and 106 tokens to 303 and 68; the payload measured 4.37
/// and 4.46 bytes per token, so the cap takes 4.4. The tree is a snapshot: the
/// gate does not see the repository grow, and each script or doc added at its
/// top two levels adds its name to the live map, 2 to 4 tokens (11 to 20
/// bytes), priced here only once the tree above lists it. A byte cap is not a
/// token count: text at fewer than 4.4 bytes per token (short words,
/// punctuation, identifiers) can pass it above the target, which is why the
/// budget ratchet in prompt.rs pins the harness's fixed part 63 bytes over its
/// size and asks for a tokenizer count before its cap moves. The cut keeps the
/// two instructions models break without them, and the gate holds them:
/// write_file prefers edit_file for changes, and the closing summary does not
/// repeat file contents.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn frozen_prefix_is_under_700_tokens() {
    let dir = std::env::temp_dir().join(format!("omx-thesis-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    for path in REPOSITORY_TREE {
        let at = project.join(path);
        if path.ends_with('/') {
            std::fs::create_dir_all(&at).unwrap();
        } else {
            std::fs::create_dir_all(at.parent().unwrap()).unwrap();
            std::fs::write(&at, "").unwrap();
        }
    }
    let project = project.canonicalize().unwrap();
    let (base_url, bodies) = recording_endpoint(vec![text("ok")]).await;
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
            "context_tokens": 32768,
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
    drive_turn(&core, &mut rx, "thesis-frozen-prefix", &project, "hello").await;

    let bodies: Vec<String> = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 1, "the turn makes one request");
    let fields = serde_json::from_str::<BTreeMap<String, Box<RawValue>>>(&bodies[0]).unwrap();
    let tools = fields["tools"].get();
    let messages = serde_json::from_str::<Vec<serde_json::Value>>(fields["messages"].get()).unwrap();
    assert_eq!(messages[0]["role"], "system", "messages[0] is the system prompt");
    let system = messages[0]["content"].as_str().expect("the system prompt is text");
    let note = messages[1]["content"].as_str().unwrap_or_default();
    assert!(note.starts_with("[execution policy: auto;"), "messages[1] is the auto mode's note: {note}");
    // The gate prices the fixture's whole layout map: the map's cap left no
    // entry out, so it has no trailer, and every entry of the tree is named.
    assert!(!system.contains(" more entries"), "the layout map's cap left entries out:\n{system}");
    for path in REPOSITORY_TREE {
        let name = path.trim_end_matches('/').rsplit('/').next().unwrap_or(path);
        assert!(system.contains(name), "the layout map names {path}:\n{system}");
    }
    let root = project.to_string_lossy();
    assert_eq!(system.matches(&*root).count(), 1, "the allowance prices one project path:\n{system}");
    let system_path_free = system.len() - root.len();
    let bytes = system_path_free + tools.len() + note.len();
    const TARGET_TOKENS: usize = 700;
    const PATH_ALLOWANCE_TOKENS: usize = 10;
    const BYTES_PER_TOKEN_TENTHS: usize = 44;
    let cap = (TARGET_TOKENS - PATH_ALLOWANCE_TOKENS) * BYTES_PER_TOKEN_TENTHS / 10;
    println!(
        "MEASURE frozen_prefix_path_free_bytes {bytes} = {system_path_free} messages[0] without the \
         {}-byte project path + {} tools + {} execution policy note; cap {cap}",
        root.len(),
        tools.len(),
        note.len(),
    );
    assert!(
        bytes <= cap,
        "the path-free frozen prefix is {bytes} bytes, over {cap} ({TARGET_TOKENS} tokens less \
         {PATH_ALLOWANCE_TOKENS} for the project path, at {}.{} bytes per token): messages[0] \
         {system_path_free} bytes without the {}-byte path, plus tools {} bytes, plus the note {} \
         bytes:\n{system}\n{tools}\n{note}",
        BYTES_PER_TOKEN_TENTHS / 10,
        BYTES_PER_TOKEN_TENTHS % 10,
        root.len(),
        tools.len(),
        note.len(),
    );
    // Bytes are not bought with the instructions models break without them.
    let schemas = serde_json::from_str::<Vec<serde_json::Value>>(tools).unwrap();
    let write_file = schemas
        .iter()
        .find(|tool| tool["function"]["name"] == "write_file")
        .expect("write_file is a built-in tool");
    let write_file = write_file["function"]["description"].as_str().unwrap_or_default();
    assert!(write_file.contains("prefer edit_file"), "write_file must point changes at edit_file: {write_file}");
    assert!(
        system.contains("without repeating file contents"),
        "the closing summary must not echo files back:\n{system}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Prefix identity: 0 bytes of the frozen prefix change across a session
/// that writes a tool file, a memory and a skill mid-session and is resumed.
/// The frozen prefix is messages[0] plus the `tools` array, what a
/// provider's prompt cache keys on. Method: a scripted endpoint records each
/// request body of a session whose first turn writes a tool file, runs it
/// with the bash line the trailing note gives it, writes a memory and a
/// skill, whose second turn answers, and whose third turn runs in a new
/// `Core` after the first let the session go, as a resume in a new process
/// does; the test slices the raw wire bytes of messages[0] and
/// `tools` out of every body unparsed, counts the bytes each differs from
/// the first body's by (a differing byte, or a difference in length), prints
/// the sum, and asserts it is 0. It also checks that messages[0] is the only
/// system message each body carries (some chat templates refuse a later one,
/// and some gateways hoist it into the system prompt, which moves the prefix
/// there) and that no `tools` array names the written tool, which joins it
/// only at /reload or in a new session. The first body, sent before any
/// write, carries the layout map in messages[0] and no trailing note; every
/// later one ends the content of its last message, a tool result or the
/// user's prompt, with a note that adds no message (a message of its own
/// after a tool result is what some hosted validators refuse) and holds only
/// what was written: the tool's line after the tool write, the skill's and
/// the memory's once the skill write refreezes, and the same three in the
/// resumed sitting, whose note is built from the session's files. Each body's history,
/// with the note cut from its raw bytes, starts with the one before it byte
/// for byte. The tool's line ends with the bash form that runs it as the
/// harness does, `command` and `args` shell-quoted with the JSON arguments
/// on stdin; the scripted bash call is that line with the arguments filled
/// in, and its reply is the tool's own output, which echoes the arguments
/// it read on stdin, inside the first turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_request_in_a_session_carries_the_same_prefix_bytes() {
    let dir = std::env::temp_dir().join(format!("omx-thesis-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("src")).unwrap();
    std::fs::write(project.join("src/main.rs"), "fn main() {}\n").unwrap();
    let project = project.canonicalize().unwrap();

    // The tool prints a marker, then the line it reads on stdin; its inline
    // program carries single quotes, so the bash line has to quote them.
    let manifest = "name = \"wordfreq\"\ndescription = \"top-n words\"\ncommand = \"sh\"\nargs = [\"-c\", \"printf 'wordfreq-ran '; cat\"]\nmutating = false\n\n[params]\ntype = \"object\"\nproperties = { n = { type = \"integer\" } }\nrequired = [\"n\"]\n";
    let arguments = r#"{"n":3}"#;
    let runs = r#"'sh' '-c' 'printf '\''wordfreq-ran '\''; cat'"#;
    let bash_line = format!("printf '%s\\n' '{arguments}' | {runs}");
    let skill = "---\nname: release\ndescription: cut a release\n---\nRun the release steps.\n";
    let (base_url, bodies) = recording_endpoint(vec![
        call(
            "call-write-tool",
            "write_file",
            serde_json::json!({ "path": ".openmax/tools/wordfreq.toml", "content": manifest }),
        ),
        call("call-run-tool", "bash", serde_json::json!({ "command": bash_line })),
        call(
            "call-write-memory",
            "write_file",
            serde_json::json!({ "path": ".openmax/memory/build.md", "content": "Build with cargo build --release.\n" }),
        ),
        call(
            "call-write-skill",
            "write_file",
            serde_json::json!({ "path": ".agents/skills/release/SKILL.md", "content": skill }),
        ),
        text("done"),
        text("still here"),
        text("resumed"),
    ])
    .await;

    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(
        data.join("settings.json"),
        serde_json::json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
            "context_tokens": 32768,
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        data.join("trust.json"),
        serde_json::json!({ "version": 1, "projects": [project.to_string_lossy()] }).to_string(),
    )
    .unwrap();

    // A session the store indexes, as every frontend creates one: the store
    // persists only indexed sessions, and the resume reads what it persisted.
    let id = {
        let (core, mut rx) = Core::new(data.clone()).unwrap();
        let id = open_max_core::sessions::create(&core, project.display().to_string()).unwrap().id;
        drive_turn(&core, &mut rx, &id, &project, "write and use a tool").await;
        drive_turn(&core, &mut rx, &id, &project, "second turn").await;
        close(&core, &id).await;
        id
    };
    // A new process resumes the session from its files.
    let (core, mut rx) = Core::new(data).unwrap();
    drive_turn(&core, &mut rx, &id, &project, "third turn").await;

    let bodies: Vec<String> = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 7, "five requests in the first turn, one in the second, one in the resumed third");
    let fields = |body: &str| serde_json::from_str::<BTreeMap<String, Box<RawValue>>>(body).unwrap();
    let messages = |body: &str| serde_json::from_str::<Vec<Box<RawValue>>>(fields(body)["messages"].get()).unwrap();
    let marker = note_marker();
    let (first, first_messages) = (fields(&bodies[0]), messages(&bodies[0]));
    let (first_system, first_tools) = (first_messages[0].get(), first["tools"].get());
    let changed: Vec<usize> = bodies
        .iter()
        .map(|body| changed_bytes(first_system, messages(body)[0].get()) + changed_bytes(first_tools, fields(body)["tools"].get()))
        .collect();
    let total: usize = changed.iter().sum();
    println!(
        "MEASURE frozen_prefix_bytes_changed {total} across {} requests (per request {changed:?}; messages[0] {} B + tools {} B; target 0)",
        bodies.len(),
        first_system.len(),
        first_tools.len(),
    );
    assert_eq!(total, 0, "the frozen prefix changed within the session, by request: {changed:?}");
    for (i, body) in bodies.iter().enumerate() {
        let n = i + 1;
        let sent = messages(body);
        let tools = fields(body)["tools"].get().to_string();
        assert!(!tools.contains("\"name\":\"wordfreq\""), "request {n} carried the written tool in its tools array");
        let systems: Vec<usize> = (0..sent.len())
            .filter(|&at| serde_json::from_str::<serde_json::Value>(sent[at].get()).unwrap()["role"] == "system")
            .collect();
        assert_eq!(systems, [0], "request {n} sent a system message past messages[0]");
    }
    // The index as the session began rides the prompt itself: the layout map's
    // line, which names the directory once and its entries under it.
    let map_line = "src/: main.rs";
    assert!(first_system.contains(map_line), "the prompt carries the layout map: {first_system}");

    let mut before: Vec<String> = Vec::new();
    let mut notes = Vec::new();
    for (i, body) in bodies.iter().enumerate() {
        let n = i + 1;
        let sent = messages(body);
        let parsed: Vec<serde_json::Value> = sent.iter().map(|m| serde_json::from_str(m.get()).unwrap()).collect();
        let (last, history) = parsed.split_last().unwrap();
        assert!(!history.iter().any(|m| m.to_string().contains(NOTE_HEADER)), "request {n} sent a stored note");
        let mut raw: Vec<String> = sent.iter().map(|m| m.get().to_string()).collect();
        let content = last["content"].as_str().unwrap_or_default();
        if n == 1 {
            // Nothing was written yet, so nothing changed since the prompt.
            assert!(!content.contains(NOTE_HEADER), "request 1 sent a note before any write: {last}");
        } else {
            // The note ends the last message and adds none.
            let (text, note) = content
                .split_once(&marker)
                .unwrap_or_else(|| panic!("request {n} sent the note as a message of its own, or not at all: {last}"));
            assert!(
                matches!(last["role"].as_str(), Some("user" | "tool")) && !text.is_empty(),
                "request {n} hung the note on a message that is not the newest prompt or tool result: {last}"
            );
            notes.push(note.to_string());
            let cut = without_note(raw.last().unwrap(), &marker);
            *raw.last_mut().unwrap() = cut;
        }
        // The history is only appended to: the next request keeps every
        // byte before the note.
        assert!(
            raw.len() > before.len() && raw[..before.len()] == before[..],
            "request {n} changed the history the request before it sent"
        );
        before = raw;
    }

    // What was written mid-session reaches the model after the history, so
    // the cached prefix never has to change to carry it, and only what was
    // written: the tool's line alone after the tool write, its schema and
    // the bash form that runs it.
    let index_lines = |note: &str| note.lines().filter(|l| l.starts_with("- ")).map(str::to_string).collect::<Vec<_>>();
    let schema = r#"{"properties":{"n":{"type":"integer"}},"required":["n"],"type":"object"}"#;
    let tool_line = format!("- wordfreq: top-n words {schema}; bash: printf '%s\\n' '<JSON arguments>' | {runs}");
    assert_eq!(index_lines(&notes[0]), std::slice::from_ref(&tool_line), "the tool write's note is its line: {}", notes[0]);
    let filled = tool_line.replace("<JSON arguments>", arguments);
    assert!(filled.ends_with(&format!("; bash: {bash_line}")), "the scripted call is the note's bash form: {filled}");

    // The tool written in the first request answers the call made with its
    // bash form in the second, inside the same turn, with its own output:
    // the marker, then the JSON arguments it read on stdin.
    let reply = serde_json::from_str::<serde_json::Value>(fields(&bodies[2])["messages"].get()).unwrap();
    let reply = reply
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["tool_call_id"] == "call-run-tool")
        .expect("the bash call has a reply")
        .clone();
    let output = reply["content"].as_str().unwrap();
    assert!(output.starts_with(&format!("wordfreq-ran {arguments}")), "the written tool did not run: {output}");

    let last = notes.last().unwrap();
    assert_eq!(index_lines(last).len(), 3, "the tool, the skill, and the memory: {last}");
    for needle in ["- wordfreq: top-n words", ".agents/skills/release/SKILL.md", ".openmax/memory/build.md"] {
        assert!(last.contains(needle), "the trailing note does not name {needle}: {last}");
    }
    assert!(!last.contains(map_line), "the note repeats the prompt's index: {last}");
    let _ = std::fs::remove_dir_all(PathBuf::from(&dir));
}
