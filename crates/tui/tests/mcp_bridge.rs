//! The MCP bridge end to end: `openmax --mcp-list` and `openmax --mcp-call`
//! against a fixture MCP stdio server, and a proxy tool written exactly as
//! `openmax --spec mcp` prints it, called by a model turn through the harness.
//!
//! The fixture server is this test binary run again as
//! `<binary> --mcp-fixture <mode>`. A libtest binary prints its banner on
//! stdout before any test runs, and stdout is the server's protocol channel,
//! so this target has no libtest harness (`harness = false`): `main` either
//! serves or runs the tests listed in `TESTS`.

use std::io::{BufRead, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const FIXTURE: &str = "--mcp-fixture";

const TESTS: [(&str, fn()); 8] = [
    ("a_proxy_tool_written_from_the_spec_lists_and_calls_through_openmax", proxy_tool_from_the_spec),
    ("an_is_error_result_exits_nonzero_with_its_text", is_error_result),
    ("a_malformed_reply_is_reported_and_exits_nonzero", malformed_reply),
    ("a_server_that_never_answers_times_out_and_is_stopped", silent_server),
    ("a_server_that_stops_reading_cannot_stall_a_large_call", stalled_server),
    ("every_page_of_tools_is_listed", paginated_list),
    ("a_server_request_is_answered_and_a_notification_ignored", server_traffic_mid_call),
    ("this_harness_reads_libtest_options_as_libtest_does", harness_options),
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some(FIXTURE) {
        serve(&args[1..]);
        return;
    }
    let selected = select(&args);
    if args.iter().any(|a| a == "--list") {
        for (name, _) in &selected {
            println!("{name}: test");
        }
        return;
    }
    println!("\nrunning {} tests", selected.len());
    let mut failed = Vec::new();
    for (name, test) in &selected {
        let ok = std::panic::catch_unwind(*test).is_ok();
        println!("test {name} ... {}", if ok { "ok" } else { "FAILED" });
        if !ok {
            failed.push(*name);
        }
    }
    println!(
        "\ntest result: {}. {} passed; {} failed\n",
        if failed.is_empty() { "ok" } else { "FAILED" },
        selected.len() - failed.len(),
        failed.len()
    );
    if !failed.is_empty() {
        std::process::exit(101);
    }
}

/// The tests a libtest command line selects, so `cargo test <filter>`,
/// `--exact`, `--skip`, `--list`, and `--ignored` (this target has no ignored
/// tests) mean what they mean everywhere else. The value after a libtest
/// option that takes one is that option's, never a name filter.
fn select(args: &[String]) -> Vec<(&'static str, fn())> {
    let (mut filters, mut skips) = (Vec::new(), Vec::new());
    let (mut exact, mut ignored) = (false, false);
    let mut words = args.iter().map(String::as_str);
    while let Some(word) = words.next() {
        match word {
            "--exact" => exact = true,
            "--ignored" => ignored = true,
            "--skip" => skips.extend(words.next()),
            "--test-threads" | "--color" | "--format" | "--logfile" | "--shuffle-seed" | "-Z" => {
                words.next();
            }
            _ if word.starts_with("--skip=") => skips.push(&word["--skip=".len()..]),
            _ if word.starts_with('-') => {}
            filter => filters.push(filter),
        }
    }
    let matches = |name: &str, filter: &str| if exact { name == filter } else { name.contains(filter) };
    TESTS
        .into_iter()
        .filter(|_| !ignored)
        .filter(|(name, _)| filters.is_empty() || filters.iter().any(|f| matches(name, f)))
        .filter(|(name, _)| !skips.iter().any(|f| matches(name, f)))
        .collect()
}

// ---------------------------------------------------------------- fixture

/// One fixture MCP server. Modes: `basic` (two tools, `echo` and `fail`),
/// `paged` (four tools over three tools/list pages), `malformed` (answers
/// initialize with a line that is not JSON-RPC), `numeric-content` (answers
/// every tools/call with a number for content), `unsupported` (answers with
/// a protocol version no client knows), `silent <pidfile>` (never reads or
/// writes, and ignores its closed stdin), and `stalled <pidfile>` (answers
/// initialize, then never reads again).
fn serve(args: &[String]) {
    let mode = args.first().map(String::as_str).unwrap_or("basic");
    // A line on stderr, which the bridge must pass through to its caller.
    eprintln!("fixture {mode}: serving");
    let write_pid = || {
        if let Some(pidfile) = args.get(1) {
            std::fs::write(pidfile, std::process::id().to_string()).unwrap();
        }
    };
    if mode == "silent" {
        write_pid();
        std::thread::sleep(Duration::from_secs(120));
        return;
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut out = std::io::stdout().lock();
    let mut initialized = false;
    while let Some(Ok(line)) = lines.next() {
        let msg: Value = serde_json::from_str(&line).expect("the client writes one JSON message per line");
        assert_eq!(msg["jsonrpc"], "2.0", "{line}");
        let id = msg["id"].clone();
        let result = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
        let error = |code: i64, message: String| {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
        };
        let method = msg["method"].as_str().unwrap_or_default();
        let reply = match method {
            "initialize" => {
                let requested = msg["params"]["protocolVersion"].as_str().unwrap_or_default();
                let version = if mode == "unsupported" { "2099-01-01" } else { requested };
                if mode == "malformed" {
                    writeln!(out, "this line is not JSON-RPC").unwrap();
                    out.flush().unwrap();
                    continue;
                }
                let reply = result(json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1.0.0"},
                }));
                if mode == "stalled" {
                    // Alive, but its stdin fills up and stays full. The 30
                    // seconds stay under `finish`'s bound, so a bridge that
                    // waits for this exit fails its assertions instead of
                    // hanging the suite.
                    send(&mut out, &reply);
                    write_pid();
                    std::thread::sleep(Duration::from_secs(30));
                    return;
                }
                reply
            }
            "notifications/initialized" => {
                initialized = true;
                continue;
            }
            _ if !initialized => error(-32600, format!("{method} before notifications/initialized")),
            "tools/list" => match (mode, msg["params"]["cursor"].as_str()) {
                ("paged", None) => result(json!({"tools": [tool("alpha")], "nextCursor": "page-2"})),
                ("paged", Some("page-2")) => {
                    result(json!({"tools": [tool("beta"), tool("gamma")], "nextCursor": "page-3"}))
                }
                ("paged", Some("page-3")) => result(json!({"tools": [tool("delta")]})),
                (_, None) => result(json!({"tools": [
                    {
                        "name": "echo",
                        "description": "Echo the text back\nwith the server's view of its environment",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}, "repeat": {"type": "integer"}},
                            "required": ["text"],
                        },
                    },
                    {"name": "fail", "description": "Always reports an error", "inputSchema": {"type": "object"}},
                ]})),
                (_, Some(cursor)) => error(-32602, format!("unknown cursor {cursor}")),
            },
            "tools/call" => {
                // Server traffic in the middle of a call: a notification the
                // client must ignore, and a request it must answer before the
                // call's own reply can arrive.
                send(&mut out, &json!({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "working"}}));
                send(&mut out, &json!({"jsonrpc": "2.0", "id": "fixture-ping", "method": "ping"}));
                let pong: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
                if pong != json!({"jsonrpc": "2.0", "id": "fixture-ping", "result": {}}) {
                    send(&mut out, &error(-32603, format!("the client answered ping with {pong}")));
                    continue;
                }
                let arguments = &msg["params"]["arguments"];
                let env = |name: &str| std::env::var(name).unwrap_or_else(|_| "unset".into());
                match msg["params"]["name"].as_str().unwrap_or_default() {
                    _ if mode == "numeric-content" => result(json!({"content": 123})),
                    "echo" => result(json!({"content": [
                        {"type": "text", "text": format!(
                            "echo: {}; NOTES_TOKEN={}; NOT_GRANTED={}",
                            arguments["text"].as_str().unwrap_or_default(),
                            env("NOTES_TOKEN"),
                            env("NOT_GRANTED"),
                        )},
                        {"type": "image", "mimeType": "image/png", "data": "aGk="},
                    ]})),
                    "fail" => result(json!({
                        "content": [{"type": "text", "text": "the fail tool always fails"}],
                        "isError": true,
                    })),
                    other => error(-32602, format!("unknown tool: {other}")),
                }
            }
            other => error(-32601, format!("method not found: {other}")),
        };
        send(&mut out, &reply);
    }
}

fn tool(name: &str) -> Value {
    json!({"name": name, "description": format!("The {name} tool"), "inputSchema": {"type": "object"}})
}

fn send(out: &mut impl Write, message: &Value) {
    writeln!(out, "{message}").unwrap();
    out.flush().unwrap();
}

// ---------------------------------------------------------------- helpers

fn openmax_bin() -> &'static str {
    env!("CARGO_BIN_EXE_openmax")
}

fn fixture() -> String {
    std::env::current_exe().unwrap().display().to_string()
}

/// A fresh project dir plus a fresh HOME, so trust and settings never leak
/// between tests or into the developer's real ~/.openmax.
fn fresh_dirs(tag: &str) -> (PathBuf, PathBuf) {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!("openmax-mcp-{tag}-{}-{nonce}", std::process::id()));
    let project = base.join("project");
    let home = base.join("home");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    (project, home)
}

fn openmax(project: &Path, home: &Path) -> Command {
    let mut c = Command::new(openmax_bin());
    c.current_dir(project);
    c.env("HOME", home);
    c.env_remove("OPENMAX_API_KEY");
    // A developer may run cargo test from inside a session; the harness
    // marks such children and trust would refuse.
    c.env_remove("OPENMAX_SESSION");
    // Tests are human-run automation with no terminal.
    c.env("OPENMAX_HUMAN_ATTEST", "1");
    // The fixture reports these two; only a test that sets them sees them.
    c.env_remove("NOTES_TOKEN");
    c.env_remove("NOT_GRANTED");
    c
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
    elapsed: Duration,
}

/// `openmax <flags> -- <this binary> --mcp-fixture <fixture...>`, with
/// `stdin` written to the bridge, bounded so a hang fails the test instead of
/// stalling the suite.
fn bridge(flags: &[&str], fixture_args: &[&str], stdin: &str) -> Run {
    let (project, home) = fresh_dirs("bridge");
    let mut command = openmax(&project, &home);
    command.args(flags).arg("--").arg(fixture()).arg(FIXTURE).args(fixture_args);
    let run = finish(command, stdin);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    run
}

fn finish(mut command: Command, stdin: &str) -> Run {
    let start = Instant::now();
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // A run refused before it reads stdin closes the pipe; that is its
    // exit status to report, not a test failure here.
    let _ = child.stdin.take().unwrap().write_all(stdin.as_bytes());
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let out = std::thread::spawn(move || {
        let mut text = String::new();
        stdout.read_to_string(&mut text).unwrap();
        text
    });
    let err = std::thread::spawn(move || {
        let mut text = String::new();
        stderr.read_to_string(&mut text).unwrap();
        text
    });
    let deadline = start + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("openmax did not finish: {}", err.join().unwrap());
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Run {
        code: status.code(),
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
        elapsed: start.elapsed(),
    }
}

/// The body of the first fenced block in `text` opened with "```<lang>".
fn fenced(text: &str, lang: &str) -> String {
    let open = format!("```{lang}\n");
    let start = text.find(&open).unwrap_or_else(|| panic!("no ```{lang} block in:\n{text}")) + open.len();
    let end = text[start..].find("```").expect("the block closes") + start;
    text[start..end].to_string()
}

fn shell_quote(word: &str) -> String {
    format!("'{}'", word.replace('\'', r"'\''"))
}

fn toml_string(word: &str) -> String {
    format!("\"{}\"", word.replace('\\', r"\\").replace('"', "\\\""))
}

/// One scripted SSE completion per request; every request body is appended
/// to `record` so the test can read what the model was sent afterwards.
fn spawn_recording_server(bodies: Vec<String>, record: PathBuf) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for sse in bodies {
            let Ok((mut stream, _)) = listener.accept() else { return };
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => return,
                }
            }
            let length: usize = String::from_utf8_lossy(&head)
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse().ok())?
                })
                .unwrap_or(0);
            let mut body = vec![0u8; length];
            if stream.read_exact(&mut body).is_err() {
                return;
            }
            let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&record).unwrap();
            log.write_all(&body).unwrap();
            log.write_all(b"\n").unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{sse}",
                sse.len(),
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    format!("http://{addr}/v1")
}

fn sse(chunks: &[Value]) -> String {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str(&format!("data: {chunk}\n\n"));
    }
    out.push_str("data: [DONE]\n\n");
    out
}

// ------------------------------------------------------------------ tests

/// The whole recipe, as written: the skill comes from the spec's own shell
/// lines and the proxy tool from its own TOML, with only the server command
/// swapped for the fixture. A model turn then calls a server tool through
/// that proxy, so the call crosses the harness, the tool's scrubbed
/// environment, `$OPENMAX_BIN --mcp-call`, and the server, and its result
/// comes back to the model.
fn proxy_tool_from_the_spec() {
    let (project, home) = fresh_dirs("proxy");
    let spec = openmax(&project, &home).args(["--spec", "mcp"]).output().unwrap();
    assert!(spec.status.success(), "{}", String::from_utf8_lossy(&spec.stderr));
    let spec = String::from_utf8(spec.stdout).unwrap();

    // The skill, generated by the spec's shell lines with `openmax` on PATH
    // naming the binary under test.
    let lines = fenced(&spec, "sh");
    let server = format!("{} {FIXTURE} basic", shell_quote(&fixture()));
    let script = lines.replace("notes-server --stdio", &server);
    assert_ne!(script, lines, "the recipe's shell lines start `notes-server --stdio`:\n{lines}");
    let bin = project.parent().unwrap().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::os::unix::fs::symlink(openmax_bin(), bin.join("openmax")).unwrap();
    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap_or_default());
    let generated = Command::new("sh")
        .arg("-c")
        .arg(&script)
        .current_dir(&project)
        .env("PATH", path)
        .env("HOME", &home)
        .env_remove("OPENMAX_SESSION")
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let skill = std::fs::read_to_string(project.join(".agents/skills/notes/SKILL.md")).unwrap();
    assert!(skill.starts_with("---\nname: notes\ndescription: "), "{skill}");
    assert!(
        skill.contains("\n- echo(text: string, repeat?: integer): Echo the text back with the server's view of its environment\n"),
        "{skill}"
    );
    assert!(skill.contains("\n- fail(): Always reports an error\n"), "{skill}");
    assert!(!skill.contains("fixture basic: serving"), "server stderr is not part of the list: {skill}");

    // The proxy tool, byte for byte the spec's except for the server command.
    let manifest = fenced(&spec, "toml");
    let words = [fixture(), FIXTURE.into(), "basic".into()].map(|w| toml_string(&w)).join(", ");
    let rewritten = manifest.replace("\"notes-server\", \"--stdio\"", &words);
    assert_ne!(rewritten, manifest, "the recipe's tool runs `notes-server --stdio`:\n{manifest}");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    std::fs::write(project.join(".openmax/tools/notes.toml"), rewritten).unwrap();

    let check = openmax(&project, &home).arg("--check").output().unwrap();
    let report = String::from_utf8_lossy(&check.stdout);
    assert!(check.status.success(), "{report}");
    assert!(report.lines().any(|l| l.starts_with("ok   skill") && l.contains("notes")), "{report}");
    assert!(!report.lines().any(|l| l.starts_with("err")), "{report}");

    let record = project.parent().unwrap().join("requests.jsonl");
    let call = json!({"tool": "echo", "arguments": {"text": "hello from the model"}});
    let base_url = spawn_recording_server(
        vec![
            sse(&[
                json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                    "function": {"name": "notes", "arguments": call.to_string()}}]}, "finish_reason": null}]}),
                json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            ]),
            sse(&[
                json!({"choices": [{"delta": {"content": "done"}, "finish_reason": null}]}),
                json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            ]),
        ],
        record.clone(),
    );
    std::fs::create_dir_all(home.join(".openmax")).unwrap();
    std::fs::write(
        home.join(".openmax/settings.json"),
        format!(
            r#"{{"base_url":"{base_url}","model":"stub-model","approval_mode":"auto","context_tokens":16384}}"#
        ),
    )
    .unwrap();
    let mut turn = openmax(&project, &home);
    turn.args(["--trust-project", "-p", "use the notes server"])
        .env("NOTES_TOKEN", "granted-token")
        .env("NOT_GRANTED", "leaked");
    let run = finish(turn, "");
    assert_eq!(run.code, Some(0), "stdout: {}\nstderr: {}", run.stdout, run.stderr);

    let sent = std::fs::read_to_string(&record).unwrap();
    let second: Value = serde_json::from_str(sent.lines().nth(1).expect("a second request")).unwrap();
    let result = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the tool result goes back to the model")["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        result.contains("echo: hello from the model; NOTES_TOKEN=granted-token; NOT_GRANTED=unset"),
        "the call reaches the server with exactly the granted credentials: {result}"
    );
    assert!(result.contains("[image image/png, 2 bytes]"), "{result}");
    assert!(result.contains("fixture basic: serving"), "the server's stderr passes through: {result}");
    assert!(!result.contains("exit code"), "{result}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A result the tool marks `isError` is a failed call, and the model must be
/// able to read why; an error reply is the same. Input the bridge cannot
/// read is the caller's mistake, refused before any server starts.
fn is_error_result() {
    let run = bridge(&["--mcp-call"], &["basic"], r#"{"tool":"fail","arguments":{}}"#);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("the fail tool always fails"), "{}", run.stderr);
    assert!(run.stdout.is_empty(), "{}", run.stdout);

    let run = bridge(&["--mcp-call"], &["basic"], r#"{"tool":"nope","arguments":{}}"#);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("unknown tool: nope"), "{}", run.stderr);

    let run = bridge(&["--mcp-call"], &["basic"], r#"{"tool":"echo","args":{"text":"x"}}"#);
    assert_eq!(run.code, Some(2), "{}", run.stderr);
    assert!(run.stderr.contains("'args'"), "the unknown key is named: {}", run.stderr);
    assert!(!run.stderr.contains("serving"), "no server starts for unreadable input: {}", run.stderr);

    // Input past the 64 MiB bound: the bound falls just after a whole call
    // padded with whitespace, so the part read parses. The byte after it is
    // part of the input too, and it makes the input invalid JSON, so nothing
    // may run.
    let call = r#"{"tool":"echo","arguments":{"text":"cut"}}"#;
    let input = format!("{call}{}}}", " ".repeat((64 << 20) - call.len()));
    let run = bridge(&["--mcp-call"], &["basic"], &input);
    assert_eq!(run.code, Some(2), "stdout: {}\nstderr: {}", run.stdout, run.stderr);
    assert!(run.stderr.contains("64 MiB"), "the bound is named: {}", run.stderr);
    assert!(!run.stderr.contains("serving"), "no server starts for oversized input: {}", run.stderr);

    let run = bridge(&["--mcp-call"], &["basic"], r#"{"tool":"echo","arguments":{"text":"plain"}}"#);
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert!(run.stdout.starts_with("echo: plain; "), "{}", run.stdout);
}

/// A reply that is not JSON-RPC, and a protocol version the bridge does not
/// speak, both end the run with the reason instead of a guess.
fn malformed_reply() {
    let run = bridge(&["--mcp-list"], &["malformed"], "");
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("this line is not JSON-RPC"), "the bad line is quoted: {}", run.stderr);
    assert!(run.stdout.is_empty(), "{}", run.stdout);

    let run = bridge(&["--mcp-list"], &["unsupported"], "");
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(
        run.stderr.contains("2099-01-01") && run.stderr.contains("2025-11-25"),
        "both sides' versions are named: {}",
        run.stderr
    );

    // A result's content is a list of items; anything else is not an empty
    // result to report as a completed call.
    let run = bridge(&["--mcp-call"], &["numeric-content"], r#"{"tool":"echo","arguments":{"text":"x"}}"#);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("content is a number, not an array: 123"), "{}", run.stderr);
    assert!(run.stdout.is_empty(), "{}", run.stdout);
}

/// A server that never answers costs one bounded wait, and is stopped even
/// though it ignores its closed stdin.
fn silent_server() {
    let (dir, _) = fresh_dirs("silent");
    let pidfile = dir.join("pid");
    let run = bridge(&["--mcp-timeout", "1", "--mcp-list"], &["silent", pidfile.to_str().unwrap()], "");
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("did not answer initialize within 1s"), "{}", run.stderr);
    assert!(run.elapsed < Duration::from_secs(15), "{:?}", run.elapsed);
    let pid: i32 = std::fs::read_to_string(&pidfile).unwrap().trim().parse().unwrap();
    // SAFETY: signal 0 only asks whether the process exists.
    assert_ne!(unsafe { libc::kill(pid, 0) }, 0, "the silent server outlived the run");
    let _ = std::fs::remove_dir_all(dir.parent().unwrap());
}

/// A server that stops reading its stdin cannot hold the bridge on a write:
/// a call larger than any pipe buffer still ends at the timeout, and the
/// server is stopped.
fn stalled_server() {
    let (dir, _) = fresh_dirs("stalled");
    let pidfile = dir.join("pid");
    let call = json!({"tool": "echo", "arguments": {"text": "x".repeat(1 << 20)}}).to_string();
    let run = bridge(&["--mcp-timeout", "1", "--mcp-call"], &["stalled", pidfile.to_str().unwrap()], &call);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
    assert!(run.stderr.contains("did not answer tools/call within 1s"), "{}", run.stderr);
    assert!(run.elapsed < Duration::from_secs(15), "{:?}", run.elapsed);
    let pid: i32 = std::fs::read_to_string(&pidfile).unwrap().trim().parse().unwrap();
    // SAFETY: signal 0 only asks whether the process exists.
    assert_ne!(unsafe { libc::kill(pid, 0) }, 0, "the stalled server outlived the run");
    let _ = std::fs::remove_dir_all(dir.parent().unwrap());
}

/// tools/list is paginated: every page is fetched with the cursor the last
/// one returned, and the list and JSON forms both carry every tool, once.
fn paginated_list() {
    let run = bridge(&["--mcp-list"], &["paged"], "");
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    let lines: Vec<&str> = run.stdout.lines().filter(|l| l.starts_with("- ")).collect();
    assert_eq!(
        lines,
        ["- alpha(): The alpha tool", "- beta(): The beta tool", "- gamma(): The gamma tool", "- delta(): The delta tool"],
        "{}",
        run.stdout
    );

    let run = bridge(&["--mcp-list", "--json"], &["paged"], "");
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    let tools: Value = serde_json::from_str(&run.stdout).unwrap();
    let names: Vec<&str> = tools.as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["alpha", "beta", "gamma", "delta"]);
}

/// The fixture sends a notification and a ping in the middle of a call and
/// refuses to answer the call until the ping is answered: a client that
/// treated either as the call's reply, or left the ping unanswered, fails.
fn server_traffic_mid_call() {
    let run = bridge(&["--mcp-call"], &["basic"], r#"{"tool":"echo","arguments":{"text":"x"}}"#);
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert_eq!(
        run.stdout,
        "echo: x; NOTES_TOKEN=unset; NOT_GRANTED=unset\n[image image/png, 2 bytes]\n",
        "{}",
        run.stderr
    );
    assert!(run.stderr.contains("fixture basic: serving"), "{}", run.stderr);
}

/// `cargo test -- <libtest options>` reaches this target too. An option's
/// value read as a name filter, or `--skip` read as a selection, would drop
/// this whole suite while the run still reports ok.
fn harness_options() {
    let names = |args: &[&str]| -> Vec<&str> {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        select(&args).into_iter().map(|(name, _)| name).collect()
    };
    let all: Vec<&str> = TESTS.iter().map(|(name, _)| *name).collect();
    let silent = "a_server_that_never_answers_times_out_and_is_stopped";
    let others: Vec<&str> = all.iter().copied().filter(|name| *name != silent).collect();
    assert_eq!(names(&[]), all);
    for values in [
        &["--test-threads", "1"][..],
        &["--color", "always"],
        &["--format", "pretty"],
        &["--logfile", "out.txt"],
        &["--shuffle-seed", "7"],
        &["-Z", "unstable-options"],
        &["--test-threads=1", "--nocapture", "-q"],
    ] {
        assert_eq!(names(values), all, "{values:?}");
    }
    assert_eq!(names(&["--skip", "never_answers"]), others);
    assert_eq!(names(&["--skip=never_answers"]), others);
    assert_eq!(names(&["--skip", "never_answers", "--exact"]), all, "--exact applies to --skip");
    assert_eq!(names(&["--exact", "--skip", silent]), others);
    assert_eq!(names(&["never_answers", "--test-threads", "1"]), [silent]);
    assert_eq!(names(&["never_answers", "--exact"]), Vec::<&str>::new());
    assert_eq!(names(&[silent, "--exact"]), [silent]);
    assert_eq!(names(&["--ignored"]), Vec::<&str>::new());
    assert_eq!(names(&["--include-ignored"]), all);
}
