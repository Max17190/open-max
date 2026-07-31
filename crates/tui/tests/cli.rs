//! End-to-end tests of the built binary: trust gating, --check exit codes,
//! the --stdio handshake, and a full print-mode turn against a stub
//! OpenAI-compatible server. These are the contracts scripts and frontends
//! build on, exercised exactly as a user's shell would.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn openmax_bin() -> &'static str {
    env!("CARGO_BIN_EXE_openmax")
}

/// A fresh project dir plus a fresh HOME, so trust and settings never leak
/// between tests or into the developer's real ~/.openmax.
fn fresh_dirs(tag: &str) -> (PathBuf, PathBuf) {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!("openmax-e2e-{tag}-{}-{nonce}", std::process::id()));
    let project = base.join("project");
    let home = base.join("home");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    (project, home)
}

fn write_settings(home: &Path, base_url: &str) {
    let dir = home.join(".openmax");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("settings.json"),
        format!(
            r#"{{"base_url":"{base_url}","model":"stub-model","approval_mode":"ask"}}"#
        ),
    )
    .unwrap();
}

fn cmd(project: &Path, home: &Path) -> Command {
    let mut c = Command::new(openmax_bin());
    c.current_dir(project);
    c.env("HOME", home);
    c.env_remove("OPENMAX_API_KEY");
    // A developer dogfooding openmax runs cargo test from inside a session;
    // the harness marks such children and trust would refuse (#83).
    c.env_remove("OPENMAX_SESSION");
    c
}

#[test]
fn an_untrusted_project_fails_closed_with_exit_3() {
    let (project, home) = fresh_dirs("trust");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let out = cmd(&project, &home)
        .args(["-p", "hello"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("trust"), "{stderr}");
}

#[test]
fn check_exit_codes_follow_findings() {
    let (project, home) = fresh_dirs("check");
    // Healthy config: a real tool file. --check needs no trust and no endpoint.
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(
        tools.join("ok.toml"),
        "name = \"ok\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n",
    )
    .unwrap();
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stdout));

    std::fs::write(tools.join("broken.toml"), "name = [not toml").unwrap();
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("broken.toml"), "{stdout}");
}

#[test]
fn stdio_handshake_speaks_the_contract() {
    let (project, home) = fresh_dirs("stdio");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let mut child = cmd(&project, &home)
        .args(["--trust-project", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    let mut hello = String::new();
    reader.read_line(&mut hello).unwrap();
    let hello: serde_json::Value = serde_json::from_str(&hello).unwrap();
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["proto"], "openmax-stdio/2");
    assert_eq!(hello["protocol_version"], 2);
    assert!(hello["session_id"].is_string());

    writeln!(stdin, r#"{{"cmd":"quit"}}"#).unwrap();
    drop(stdin);
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(0));
}

/// Read one HTTP request (headers, then exactly Content-Length body bytes)
/// off `stream`. False when the peer went away mid-request.
fn read_request(stream: &mut std::net::TcpStream) -> bool {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => buf.push(byte[0]),
            _ => return false,
        }
    }
    let headers = String::from_utf8_lossy(&buf).to_string();
    let content_length: usize = headers
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse().ok())?
        })
        .unwrap_or(0);
    let mut body = vec![0u8; content_length];
    content_length == 0 || stream.read_exact(&mut body).is_ok()
}

/// Minimal OpenAI-compatible streaming endpoint: one canned SSE completion
/// per request, enough for a whole print-mode turn.
fn spawn_stub_server() -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        // Serve a few requests then exit with the test.
        for _ in 0..4 {
            let Ok((mut stream, _)) = listener.accept() else { return };
            if !read_request(&mut stream) {
                return;
            }

            let sse = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"stub says hi\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3}}\n\n",
                "data: [DONE]\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{sse}",
                sse.len(),
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1"), handle)
}

/// A provider that dies mid-answer: a partial SSE body with no finish_reason
/// chunk, no `[DONE]`, and no Content-Length, then the socket closes. The
/// transfer is well-formed; only the completion signal never arrives.
fn spawn_truncating_stub_server() -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else { return };
        if !read_request(&mut stream) {
            return;
        }
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"half an ans\"},\"finish_reason\":null}]}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse}"
        );
        let _ = stream.write_all(response.as_bytes());
        // Dropping the socket ends the body part way through the answer.
    });
    (format!("http://{addr}/v1"), handle)
}

#[test]
fn a_print_turn_against_a_stub_server_reaches_stdout() {
    let (project, home) = fresh_dirs("turn");
    let (base_url, _server) = spawn_stub_server();
    write_settings(&home, &base_url);

    let out = cmd(&project, &home)
        .args(["--trust-project", "-p", "say hi"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("stub says hi"), "stdout: {stdout}\nstderr: {stderr}");
}

#[test]
fn a_json_print_turn_emits_valid_envelopes_ending_in_done() {
    let (project, home) = fresh_dirs("json");
    let (base_url, _server) = spawn_stub_server();
    write_settings(&home, &base_url);

    let out = cmd(&project, &home)
        .args(["--trust-project", "--json", "-p", "say hi"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("every stdout line is JSON"))
        .collect();
    assert!(!lines.is_empty());
    for line in &lines {
        assert!(line["session_id"].is_string(), "{line}");
        assert!(line["type"].is_string(), "{line}");
    }
    let last = lines.last().unwrap();
    assert_eq!(last["type"], "done", "the stream must end in done: {stdout}");
    assert!(
        lines.iter().any(|l| l["type"] == "message_done" && l["text"] == "stub says hi"),
        "{stdout}"
    );
}

/// A stream the provider abandons must never read as a finished answer: the
/// partial text is kept (so the session resumes), but the turn reports an
/// error, ends with stop_reason `truncated`, and print mode exits nonzero.
#[test]
fn a_truncated_stream_reports_truncation_instead_of_a_clean_stop() {
    let (project, home) = fresh_dirs("truncated");
    let (base_url, _server) = spawn_truncating_stub_server();
    write_settings(&home, &base_url);

    let out = cmd(&project, &home)
        .args(["--trust-project", "--json", "-p", "say hi"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "a cut-off answer must not exit 0: {stdout}");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("every stdout line is JSON"))
        .collect();
    // The turn terminator guarantee holds: exactly one done, and it is last.
    assert_eq!(
        lines.iter().filter(|l| l["type"] == "done").count(),
        1,
        "exactly one done per turn: {stdout}"
    );
    let last = lines.last().expect("at least one line");
    assert_eq!(last["type"], "done", "{stdout}");
    assert_eq!(last["stop_reason"], "truncated", "{stdout}");
    assert!(
        lines.iter().any(|l| l["type"] == "error"
            && l["message"].as_str().is_some_and(|m| m.contains("incomplete"))),
        "the truncation must be reported as an error: {stdout}"
    );
    assert!(
        lines.iter().any(|l| l["type"] == "message_done" && l["text"] == "half an ans"),
        "the partial answer must still be delivered: {stdout}"
    );

    // ...and it must survive on disk, or a resume would lose the partial turn.
    let sessions = home.join(".openmax").join("sessions");
    let saved: String = std::fs::read_dir(&sessions)
        .expect("sessions dir")
        .filter_map(|e| e.ok())
        .map(|e| std::fs::read_to_string(e.path()).unwrap_or_default())
        .collect();
    assert!(saved.contains("half an ans"), "partial reply must be persisted: {saved}");
}
