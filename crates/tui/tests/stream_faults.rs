//! The transcript persistence contract, held under the faults a real
//! endpoint produces: a stream cut at any byte, a reply split across reads
//! anywhere, a provider error inside a 200, a 5xx or 429 before any body, a
//! one-shot JSON reply, and an endpoint that goes quiet. Each case drives
//! the built binary in `--json` print mode and checks the event stream, the
//! transcript on disk, the exit code, and then a `--continue` resume against
//! the same endpoint, with one checker shared by every case. A single
//! scenario per fault proves that scenario; the contract is a claim about
//! every byte offset, so the seeded cases sample them and replay from the
//! seed a failure names.
//!
//! The endpoint is its own small scripted server rather than the stub in
//! cli.rs: these faults need byte-level control of a response (where a body
//! is cut, how it is framed, how it is split across writes), the time each
//! request arrived, and a schedule that decides per arrival what the
//! connection does. Any request past the script gets a 400 the client does
//! not resend, so a wrong expectation, or a regression that adds attempts,
//! fails at once instead of spending the retry budget.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use open_max_core::types::ChatMessage;
use serde_json::value::RawValue;
use serde_json::{json, Value};

const TRUNCATION_STUB: &str = "The provider stream ended before this call could run; it was not executed.";
const CANCEL_STUB: &str = "The user cancelled this turn.";
const INCOMPLETE: &str = "provider response was interrupted or exceeded client limits; the reply above is incomplete";
const STREAM_ENDED: &str = "the stream ended before the reply finished";
const SILENCE: &str = "the endpoint sent nothing for 1s";
const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n";
const JSON_TYPE: &str = "Content-Type: application/json\r\n";
const OVERLOADED: &str = r#"{"error":{"code":503,"message":"overloaded mid-reply"},"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}"#;

fn cmd(project: &Path, home: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_openmax"));
    c.current_dir(project).env("HOME", home);
    c.env_remove("OPENMAX_API_KEY");
    // A developer may run cargo test from inside a session; the harness marks
    // such children and trust would refuse them.
    c.env_remove("OPENMAX_SESSION");
    // Tests are human-run automation with no terminal, so --trust-project
    // needs the attestation.
    c.env("OPENMAX_HUMAN_ATTEST", "1");
    // A proxy between the client and these faults would answer for the
    // endpoint and turn each cut into an error of its own.
    for var in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"] {
        c.env_remove(var);
    }
    c
}

/// A fresh project dir plus a fresh HOME, so trust, settings, and sessions
/// never leak between cases or into the developer's ~/.openmax.
fn fresh_dirs(tag: &str) -> (PathBuf, PathBuf) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let n = NEXT.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!("openmax-faults-{tag}-{}-{nonce}-{n}", std::process::id()));
    let (project, home) = (base.join("project"), base.join("home"));
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(home.join(".openmax")).unwrap();
    (project, home)
}

/// Wait for a child with its output drained as it comes, so a full pipe can
/// never stall it, and kill it past a deadline: a hang becomes a failure
/// that names the case instead of a stuck test run.
fn finish_with_deadline(mut child: std::process::Child, ctx: &str) -> Output {
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            bytes
        })
    };
    let stdout = drain(Box::new(child.stdout.take().unwrap()));
    let stderr = drain(Box::new(child.stderr.take().unwrap()));
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let stderr = stderr.join().unwrap();
            panic!("{ctx}: openmax did not finish within 30s: {}", String::from_utf8_lossy(&stderr));
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    Output { status, stdout: stdout.join().unwrap(), stderr: stderr.join().unwrap() }
}

/// Read one HTTP request (headers, then exactly Content-Length body bytes)
/// off `stream`, returning the body. None when the peer went away mid-request.
fn read_request(stream: &mut TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => buf.push(byte[0]),
            _ => return None,
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
    if content_length > 0 && stream.read_exact(&mut body).is_err() {
        return None;
    }
    Some(String::from_utf8_lossy(&body).to_string())
}

/// How a cut body is framed, which decides how the client learns of the cut:
/// a clean end of a close-delimited body, a body shorter than its declared
/// length, or a chunked body that never sends its last chunk.
#[derive(Clone, Copy, Debug)]
enum Framing {
    CloseDelimited,
    ShortLength,
    ChunkedNoEnd,
}

/// What the endpoint does with one request, in script order.
enum Step {
    /// A whole SSE body with a Content-Length.
    Finished(String),
    /// The first `at` bytes of an SSE body, then a close. With `pieces`, the
    /// prefix goes out in writes of those sizes.
    Cut { body: Vec<u8>, at: usize, framing: Framing, pieces: Vec<usize> },
    /// A whole SSE body written in pieces of these sizes.
    Fragmented { body: Vec<u8>, pieces: Vec<usize> },
    /// A complete response with this status, extra headers, and body.
    Status { status: &'static str, headers: &'static str, body: String },
    /// These bytes and nothing else, then a close.
    Raw(Vec<u8>),
    /// Read the request and close without a byte.
    CloseBeforeAnyByte,
    /// Read the request and hold the connection without a byte.
    Silent,
    /// SSE headers and this prefix, then hold the connection.
    Stall(String),
    /// SSE headers, a keepalive comment every `every_ms` for `count` times,
    /// then this body.
    Keepalive { every_ms: u64, count: usize, then: String },
}

struct FaultServer {
    base_url: String,
    /// Each request's body and the moment it was fully read.
    arrivals: Arc<Mutex<Vec<(Instant, String)>>>,
    /// The moment each response began, by arrival: the client cannot have
    /// read a byte of it, or started a wait, any earlier.
    replied: Arc<Mutex<Vec<Instant>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FaultServer {
    fn start(script: Vec<Step>) -> FaultServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let arrivals = Arc::new(Mutex::new(Vec::new()));
        let replied = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, began, stopping) = (arrivals.clone(), replied.clone(), stop.clone());
        let handle = std::thread::spawn(move || {
            let mut script = script.into_iter();
            // Silent and stalled connections stay open until the server stops,
            // so only the client's own idle timeout ends them.
            let mut held = Vec::new();
            while !stopping.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(_) => return,
                };
                // Some platforms hand an accepted socket the listener's
                // nonblocking mode.
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                let Some(body) = read_request(&mut stream) else { continue };
                let n = {
                    let mut seen = seen.lock().unwrap();
                    seen.push((Instant::now(), body));
                    seen.len()
                };
                let step = script.next().unwrap_or_else(|| Step::Status {
                    status: "400 Bad Request",
                    headers: JSON_TYPE,
                    body: format!(r#"{{"error":{{"message":"unscripted request {n}"}}}}"#),
                });
                began.lock().unwrap().push(Instant::now());
                let hold = serve(&mut stream, step);
                if hold {
                    held.push(stream);
                }
            }
        });
        FaultServer { base_url, arrivals, replied, stop, handle: Some(handle) }
    }

    fn bodies(&self) -> Vec<String> {
        self.arrivals.lock().unwrap().iter().map(|(_, body)| body.clone()).collect()
    }

    /// The time from the start of response `i` to the arrival of request
    /// `i + 1`. It holds the client's whole wait and can only exceed it, so
    /// a lower bound on it does not fail on a slow machine.
    fn gap_after(&self, i: usize) -> Duration {
        let replied = self.replied.lock().unwrap()[i];
        self.arrivals.lock().unwrap()[i + 1].0.duration_since(replied)
    }
}

impl Drop for FaultServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Serve one step; true when the connection is to be held open.
fn serve(stream: &mut TcpStream, step: Step) -> bool {
    // The client may have gone by now; what it saw is what the test checks.
    fn put(stream: &mut TcpStream, bytes: &[u8]) {
        let _ = stream.write_all(bytes);
        let _ = stream.flush();
    }
    match step {
        Step::Finished(body) => put(stream, format!("{SSE_HEAD}Content-Length: {}\r\n\r\n{body}", body.len()).as_bytes()),
        Step::Cut { body, at, framing, pieces } => {
            let head = match framing {
                Framing::CloseDelimited => format!("{SSE_HEAD}\r\n"),
                Framing::ShortLength => format!("{SSE_HEAD}Content-Length: {}\r\n\r\n", body.len()),
                Framing::ChunkedNoEnd => format!("{SSE_HEAD}Transfer-Encoding: chunked\r\n\r\n{:x}\r\n", body.len()),
            };
            put(stream, head.as_bytes());
            send_in_pieces(stream, &body[..at], &pieces);
        }
        Step::Fragmented { body, pieces } => {
            put(stream, format!("{SSE_HEAD}Content-Length: {}\r\n\r\n", body.len()).as_bytes());
            send_in_pieces(stream, &body, &pieces);
        }
        Step::Status { status, headers, body } => put(
            stream,
            format!("HTTP/1.1 {status}\r\n{headers}Connection: close\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes(),
        ),
        Step::Raw(bytes) => put(stream, &bytes),
        Step::CloseBeforeAnyByte => {}
        Step::Silent => return true,
        Step::Stall(prefix) => {
            put(stream, format!("{SSE_HEAD}\r\n{prefix}").as_bytes());
            return true;
        }
        Step::Keepalive { every_ms, count, then } => {
            put(stream, format!("{SSE_HEAD}\r\n").as_bytes());
            for _ in 0..count {
                std::thread::sleep(Duration::from_millis(every_ms));
                put(stream, b": keepalive\n\n");
            }
            put(stream, then.as_bytes());
        }
    }
    false
}

/// Write `bytes` in pieces of the given sizes, each its own segment, so the
/// client's reads split where the pieces do: inside a line, between `data:`
/// and its JSON, inside a multi-byte character.
fn send_in_pieces(stream: &mut TcpStream, bytes: &[u8], pieces: &[usize]) {
    if pieces.is_empty() {
        let _ = stream.write_all(bytes);
        return;
    }
    let _ = stream.set_nodelay(true);
    let mut at = 0;
    for &size in pieces {
        let end = (at + size).min(bytes.len());
        if stream.write_all(&bytes[at..end]).is_err() {
            return;
        }
        at = end;
        if at == bytes.len() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let _ = stream.write_all(&bytes[at..]);
}

fn data(chunk: &Value) -> String {
    format!("data: {chunk}\n\n")
}

fn chunk(delta: Value, finish_reason: Value) -> Value {
    json!({"choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]})
}

fn content(text: &str) -> String {
    data(&chunk(json!({"content": text}), Value::Null))
}

fn finish(reason: &str) -> String {
    data(&chunk(json!({}), json!(reason)))
}

const DONE: &str = "data: [DONE]\n\n";

/// The bash command a call runs: one line appended to ran.log, so the log
/// holds exactly one line per execution and names the call that wrote it.
fn bash_args(marker: &str) -> String {
    json!({"command": format!("printf '%s\\n' '{marker}' >> ran.log # ✓ 日本語")}).to_string()
}

fn call(id: &str, marker: &str) -> String {
    let call = json!([{"index":0,"id":id,"type":"function","function":{"name":"bash","arguments":bash_args(marker)}}]);
    data(&chunk(json!({"tool_calls": call}), Value::Null))
}

fn sse_text(text: &str) -> String {
    format!("{}{}{DONE}", content(text), finish("stop"))
}

fn sse_call(id: &str, marker: &str) -> String {
    format!("{}{}{DONE}", call(id, marker), finish("tool_calls"))
}

/// A one-shot chat completion, the shape a server that ignores `stream`
/// sends back.
fn json_reply(text: Option<&str>, calls: &[(&str, &str)], finish_reason: &str) -> String {
    let mut message = json!({"role": "assistant", "content": text});
    if !calls.is_empty() {
        let calls: Vec<Value> = calls
            .iter()
            .map(|(id, marker)| json!({"id": id, "type": "function", "function": {"name": "bash", "arguments": bash_args(marker)}}))
            .collect();
        message["tool_calls"] = json!(calls);
    }
    json!({"id": "one-shot", "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    .to_string()
}

/// One case: a fresh project and HOME, and an endpoint serving its script.
struct Env {
    tag: String,
    ctx: String,
    project: PathBuf,
    home: PathBuf,
    server: FaultServer,
}

impl Env {
    fn new(tag: &str, ctx: &str, script: Vec<Step>) -> Env {
        Env::with_settings(tag, ctx, script, "")
    }

    /// `extra` is appended to settings.json as more fields. `auto` runs the
    /// bash calls unattended, so a call that does not run was refused for
    /// the fault and not by the approval gate.
    fn with_settings(tag: &str, ctx: &str, script: Vec<Step>, extra: &str) -> Env {
        let (project, home) = fresh_dirs(tag);
        let server = FaultServer::start(script);
        std::fs::write(
            home.join(".openmax/settings.json"),
            format!(
                r#"{{"base_url":"{}","model":"stub-model","approval_mode":"auto","context_tokens":16384{extra}}}"#,
                server.base_url
            ),
        )
        .unwrap();
        Env { tag: tag.to_string(), ctx: ctx.to_string(), project, home, server }
    }

    /// The endpoint as a provider with a one second idle timeout, so silence
    /// costs a second rather than the default ten minutes.
    fn idle(tag: &str, script: Vec<Step>) -> Env {
        let env = Env::new(tag, tag, script);
        let dir = env.home.join(".openmax");
        std::fs::write(
            dir.join("providers.json"),
            json!({"providers": {"stub": {"base_url": env.server.base_url, "idle_timeout_secs": 1}}}).to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("settings.json"),
            r#"{"provider":"stub","model":"stub-model","approval_mode":"auto","context_tokens":16384}"#,
        )
        .unwrap();
        env
    }

    fn transcript(&self, sid: &str) -> PathBuf {
        self.home.join(".openmax/sessions").join(format!("{sid}.messages.json"))
    }

    fn ran_log(&self) -> String {
        std::fs::read_to_string(self.project.join("ran.log")).unwrap_or_default()
    }

    /// Run `openmax <args> --json -p <prompt>...` and collect everything the
    /// checks read: the exit, the events, the transcript before and after,
    /// and the requests this run made.
    fn run(&self, args: &[&str], prompts: &[&str], before: &[u8], ctx: &str) -> Run {
        let first = self.server.arrivals.lock().unwrap().len();
        let mut command = cmd(&self.project, &self.home);
        command.args(args).arg("--json");
        for prompt in prompts {
            command.args(["-p", prompt]);
        }
        let out = finish_with_deadline(
            command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap(),
            ctx,
        );
        let stdout = String::from_utf8(out.stdout).unwrap_or_else(|e| panic!("{ctx}: stdout is not UTF-8: {e}"));
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let events: Vec<Value> = stdout
            .lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{ctx}: a stdout line is not JSON ({e}): {l}\nstderr: {stderr}")))
            .collect();
        let sid = events
            .first()
            .and_then(|e| e["session_id"].as_str())
            .unwrap_or_else(|| panic!("{ctx}: no events (exit {:?}): {stderr}", out.status.code()))
            .to_string();
        let after = std::fs::read(self.transcript(&sid)).unwrap_or_default();
        Run {
            ctx: ctx.to_string(),
            prompt: prompts[0].to_string(),
            code: out.status.code(),
            events,
            stdout,
            stderr,
            sid,
            before: before.to_vec(),
            after,
            requests: self.server.bodies()[first..].to_vec(),
        }
    }

    /// The case's first turn, in a new session, held to the contract.
    fn first(&self, prompt: &str) -> Run {
        let run = self.run(&["--trust-project"], &[prompt], &[], &self.ctx);
        check_contract(&run);
        run
    }

    /// Resume the session `first` left, against the same endpoint, since the
    /// reasoning origin hashes the base_url and a new port would rightly
    /// withhold reasoning. The script answers it with `resumed`.
    fn resume(&self, first: &Run, before: &[u8]) -> Run {
        let prompt = format!("resume after {}", self.tag);
        let run = self.run(&["--continue"], &[&prompt], before, &format!("{} (resume)", self.ctx));
        check_contract(&run);
        check_resume(first, &run);
        run
    }

    /// The same turn in text mode: the exit code and the stderr lines a
    /// script reads when it does not parse events.
    fn text(&self, prompt: &str) -> (Option<i32>, String) {
        let out = finish_with_deadline(
            cmd(&self.project, &self.home)
                .args(["--trust-project", "-p", prompt])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
            &self.ctx,
        );
        (out.status.code(), String::from_utf8_lossy(&out.stderr).to_string())
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.project.parent().unwrap());
    }
}

struct Run {
    ctx: String,
    prompt: String,
    code: Option<i32>,
    events: Vec<Value>,
    stdout: String,
    stderr: String,
    sid: String,
    before: Vec<u8>,
    after: Vec<u8>,
    requests: Vec<String>,
}

impl Run {
    fn log(&self) -> String {
        format!("{}\nexit {:?}\nstdout:\n{}\nstderr:\n{}", self.ctx, self.code, self.stdout, self.stderr)
    }

    fn of<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
        self.events.iter().filter(move |e| e["type"] == kind)
    }

    fn count(&self, kind: &str) -> usize {
        self.of(kind).count()
    }

    fn stop_reason(&self) -> &str {
        self.of("done").last().and_then(|e| e["stop_reason"].as_str()).unwrap_or("")
    }

    fn errors(&self) -> Vec<&str> {
        self.of("error").filter_map(|e| e["message"].as_str()).collect()
    }

    fn message_done(&self) -> Vec<&str> {
        self.of("message_done").filter_map(|e| e["text"].as_str()).collect()
    }

    fn retry_reasons(&self) -> Vec<&str> {
        self.of("retry").filter_map(|e| e["reason"].as_str()).collect()
    }

    fn messages(&self) -> Vec<ChatMessage> {
        transcript_messages(&self.after, &self.ctx)
    }
}

/// The exit code the documented table gives for these events: 4 when the
/// turn stopped short, which wins over an earlier error line; else 1 for a
/// failed turn or any error line; else 0, a finish_reason the harness does
/// not name (such as `length`) included.
fn code_from_events(events: &[Value]) -> i32 {
    let stop = events.iter().rev().find(|e| e["type"] == "done").and_then(|e| e["stop_reason"].as_str()).unwrap_or("");
    if matches!(stop, "max_iterations" | "budget_exhausted" | "unverified") {
        4
    } else if stop == "error" || events.iter().any(|e| e["type"] == "error") {
        1
    } else {
        0
    }
}

/// The bytes up to and including the last newline: the records a load
/// reads, without a torn final fragment.
fn complete_prefix(bytes: &[u8]) -> &[u8] {
    match bytes.iter().rposition(|&b| b == b'\n') {
        Some(at) => &bytes[..=at],
        None => &[],
    }
}

/// The transcript is JSONL: not empty, ending in a newline, every line a
/// message, the first the system prompt.
fn transcript_messages(bytes: &[u8], ctx: &str) -> Vec<ChatMessage> {
    assert!(!bytes.is_empty() && bytes.ends_with(b"\n"), "{ctx}: the transcript must be whole lines: {}", String::from_utf8_lossy(bytes));
    let text = std::str::from_utf8(bytes).unwrap_or_else(|e| panic!("{ctx}: the transcript is not UTF-8: {e}"));
    let messages: Vec<ChatMessage> = text
        .lines()
        .enumerate()
        .map(|(i, line)| serde_json::from_str(line).unwrap_or_else(|e| panic!("{ctx}: transcript line {i} is not a message ({e}): {line}")))
        .collect();
    assert_eq!(messages[0].role, "system", "{ctx}: line 0 is the system prompt");
    messages
}

/// Every call is answered in the run of tool messages right after it, in
/// order, and no tool message stands anywhere else. Ids are matched per run,
/// so an id that repeats across iterations is answered in each.
fn check_pairing(messages: &[ChatMessage], ctx: &str) {
    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        assert_ne!(message.role, "tool", "{ctx}: the tool message at line {i} answers no call right before it");
        i += 1;
        for call in message.tool_calls.iter().flatten() {
            let reply = messages.get(i).filter(|m| m.role == "tool");
            let reply = reply.unwrap_or_else(|| panic!("{ctx}: call {} has no reply at line {i}", call.id));
            assert_eq!(reply.tool_call_id.as_deref(), Some(call.id.as_str()), "{ctx}: the reply at line {i} answers another call");
            i += 1;
        }
    }
}

/// What a reply on disk must read for a tool_end: the output, with `Error: `
/// ahead of a failure, and then each harness note the window tied to it.
fn tool_reply(end: &Value, notes: &[&Value]) -> String {
    let output = end["output"].as_str().unwrap();
    let mut reply = if end["ok"] == true { output.to_string() } else { format!("Error: {output}") };
    for note in notes.iter().filter(|n| n["call_id"] == end["call_id"]) {
        reply.push('\n');
        reply.push_str(note["text"].as_str().unwrap());
    }
    reply
}

/// The checks every run must pass, whatever the endpoint did: one
/// terminator and an exit code that follows it, a transcript of whole
/// records that only grows, every call answered in its own run, the events
/// and the transcript in agreement window by window, and requests that only
/// extend the prompt or resend it byte for byte.
fn check_contract(run: &Run) {
    let log = run.log();
    for event in &run.events {
        assert_eq!(event["session_id"], run.sid.as_str(), "{log}");
    }
    assert_eq!(run.count("done"), 1, "exactly one done per turn: {log}");
    assert_eq!(run.events.last().unwrap()["type"], "done", "done is the last line: {log}");
    let stop = run.stop_reason();
    assert_ne!(stop, "refused", "{log}");
    assert_eq!(run.code, Some(code_from_events(&run.events)), "the exit code follows the events: {log}");
    let position = |kind: &str| run.events.iter().rposition(|e| e["type"] == kind);
    if stop == "error" || stop == "truncated" {
        assert!(position("error").is_some(), "a failed turn reports an error before done: {log}");
    }
    if stop == "truncated" {
        let (finished, error) = (position("message_done"), position("error"));
        assert!(finished.is_some() && finished < error, "a truncated turn reports its partial reply, then the error: {log}");
    }

    let messages = run.messages();
    let kept = complete_prefix(&run.before);
    assert!(run.after.starts_with(kept), "the transcript is append-only: {log}");
    check_pairing(&messages, &run.ctx);
    // A new session writes its system prompt first, and line 0 is checked
    // above, so the walk starts after it.
    let old = kept.iter().filter(|&&b| b == b'\n').count().max(1);
    check_windows(run, &messages[old..], stop, &log);
    check_requests(run, &log);
}

/// The messages one run appended, read in order against its events.
struct Appended<'a> {
    new: &'a [ChatMessage],
    at: usize,
    log: &'a str,
}

impl Appended<'_> {
    fn peek(&self) -> Option<&ChatMessage> {
        self.new.get(self.at)
    }

    fn expect(&mut self, role: &str, content: Option<&str>, what: &str) -> ChatMessage {
        let log = self.log;
        let message = self.peek().unwrap_or_else(|| panic!("{what} is missing from the transcript: {log}")).clone();
        assert_eq!(message.role, role, "{what}: {log}");
        if let Some(content) = content {
            assert_eq!(message.content.as_deref(), Some(content), "{what}: {log}");
        }
        self.at += 1;
        message
    }
}

/// Walk the messages this run appended alongside its events, one request
/// window at a time (a budget event opens each).
fn check_windows(run: &Run, new: &[ChatMessage], stop: &str, log: &str) {
    let mut disk = Appended { new, at: 0, log };
    let opens: Vec<usize> = run.events.iter().enumerate().filter(|(_, e)| e["type"] == "budget").map(|(i, _)| i).collect();
    for event in &run.events[..opens.first().copied().unwrap_or(run.events.len())] {
        match event["type"].as_str().unwrap() {
            // A note before the first request goes to the model as a user
            // message, ahead of the prompt.
            "harness_note" => {
                assert_eq!(event["call_id"], "", "{log}");
                disk.expect("user", event["text"].as_str(), "a note before the prompt");
            }
            "done" | "error" => {}
            other => panic!("unexpected {other} before the first request: {log}"),
        }
    }
    disk.expect("user", Some(&run.prompt), "the prompt");

    for (w, &open) in opens.iter().enumerate() {
        let last = w + 1 == opens.len();
        let window = &run.events[open + 1..opens.get(w + 1).copied().unwrap_or(run.events.len())];
        let of = |kind: &str| window.iter().filter(|e| e["type"] == kind).collect::<Vec<_>>();
        if let Some(retry) = window.iter().rposition(|e| e["type"] == "retry") {
            assert!(!window[..retry].iter().any(|e| e["type"] == "token"), "no reply text streams before a retry: {log}");
        }
        for (i, start) in window.iter().enumerate().filter(|(_, e)| e["type"] == "tool_start") {
            let ends = window[i + 1..].iter().filter(|e| e["type"] == "tool_end" && e["call_id"] == start["call_id"]).count();
            assert_eq!(ends, 1, "every tool_start has one tool_end in its window: {log}");
        }
        let (ends, notes) = (of("tool_end"), of("harness_note"));
        assert_eq!(of("tool_start").len(), ends.len(), "every tool_end follows its tool_start: {log}");
        let finished = of("message_done");
        assert!(finished.len() <= 1, "one message_done per request: {log}");
        let Some(text) = finished.first().map(|e| e["text"].as_str().unwrap()) else {
            // A reply that failed appends nothing, even after text streamed.
            assert!(last && ends.is_empty() && !of("error").is_empty(), "a request without message_done ends the turn in an error: {log}");
            continue;
        };
        let tokens: String = of("token").iter().map(|e| e["text"].as_str().unwrap()).collect();
        assert_eq!(tokens, text, "the token lines join into the finished message: {log}");
        let mut calls = Vec::new();
        if disk.peek().is_some_and(|m| m.role == "assistant") {
            let reply = disk.expect("assistant", None, "the reply");
            assert_eq!(reply.content.as_deref(), Some(text).filter(|t| !t.is_empty()), "the reply on disk is message_done: {log}");
            calls = reply.tool_calls.iter().flatten().map(|c| c.id.clone()).collect();
        } else {
            assert!(text.is_empty(), "a reply with text is on disk: {log}");
        }
        let ran: Vec<String> = ends.iter().map(|e| e["call_id"].as_str().unwrap().to_string()).collect();
        if ran.is_empty() && !calls.is_empty() {
            assert!(last && (stop == "truncated" || stop == "cancelled"), "calls that did not run end the turn: {log}");
        } else {
            assert_eq!(ran, calls, "the tool_end ids are the reply's calls, in order: {log}");
        }
        for end in &ends {
            let reply = disk.expect("tool", Some(&tool_reply(end, &notes)), "a tool reply");
            assert_eq!(reply.tool_call_id.as_deref(), end["call_id"].as_str(), "{log}");
        }
    }
    // Calls that never ran are answered with the stub for why the turn ended.
    let stub = match stop {
        "truncated" => TRUNCATION_STUB,
        "cancelled" => CANCEL_STUB,
        _ => "",
    };
    while disk.peek().is_some() {
        assert!(!stub.is_empty(), "line {} of this run has no event behind it: {log}", disk.at);
        disk.expect("tool", Some(stub), "a stub for a call that never ran");
    }
}

fn fields(body: &str) -> BTreeMap<String, Box<RawValue>> {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("a request body is not a JSON object ({e}): {body}"))
}

/// The endpoint saw one request per window plus one per retry. A resend is
/// the same bytes; the next window's request keeps every field but
/// `messages` byte for byte and only appends to `messages`.
fn check_requests(run: &Run, log: &str) {
    let mut attempts: Vec<usize> = Vec::new();
    for event in &run.events {
        match event["type"].as_str() {
            Some("budget") => attempts.push(1),
            Some("retry") => {
                let n = attempts.last_mut().expect("a retry inside a request window");
                *n += 1;
                assert_eq!(event["attempt"], *n, "retries count the attempt that goes out: {log}");
            }
            _ => {}
        }
    }
    assert_eq!(run.requests.len(), attempts.iter().sum::<usize>(), "one request per window and per retry: {log}");
    let mut at = 0;
    let mut previous: Option<&str> = None;
    for n in attempts {
        let first = run.requests[at].as_str();
        if let Some(previous) = previous {
            let (a, b) = (fields(previous), fields(first));
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "{log}");
            for (key, value) in a.iter().filter(|(key, _)| key.as_str() != "messages") {
                assert_eq!(value.get(), b[key].get(), "{key} changed within a turn: {log}");
            }
            let (old, new) = (a["messages"].get(), b["messages"].get());
            let stem = format!("{},", &old[..old.len() - 1]);
            assert!(new.starts_with(&stem) && new.len() > stem.len(), "a request only appends to the one before: {log}");
        }
        for resend in &run.requests[at + 1..at + n] {
            assert_eq!(resend, first, "a resend repeats the request byte for byte: {log}");
        }
        previous = Some(&run.requests[at + n - 1]);
        at += n;
    }
}

/// The wire never carries `reasoning_origin`, the stamp the transcript keeps
/// on a message with reasoning.
fn without_origin(line: &str) -> String {
    const KEY: &str = ",\"reasoning_origin\":\"";
    match line.find(KEY) {
        Some(at) => format!("{}{}", &line[..at], &line[at + KEY.len() + 17..]),
        None => line.to_string(),
    }
}

fn user_line(prompt: &str) -> String {
    format!("{{\"role\":\"user\",\"content\":{}}}", serde_json::to_string(prompt).unwrap())
}

/// A resumed session sends the records on disk exactly as stored, then the
/// new prompt with nothing between, and the same tools byte for byte; and
/// the file grows by the prompt and the reply alone.
fn check_resume(first: &Run, resumed: &Run) {
    let log = resumed.log();
    assert_eq!(resumed.sid, first.sid, "{log}");
    let kept = complete_prefix(&resumed.before);
    let lines: Vec<String> = std::str::from_utf8(kept).unwrap().lines().map(without_origin).collect();
    let expected = format!("[{},{}]", lines.join(","), user_line(&resumed.prompt));
    let body = fields(&resumed.requests[0]);
    assert_eq!(body["messages"].get(), expected, "the resume sends the transcript as stored: {log}");
    assert_eq!(body["tools"].get(), fields(&first.requests[0])["tools"].get(), "the resume sends the same tools: {log}");
    let grown = String::from_utf8_lossy(&resumed.after[kept.len()..]).to_string();
    assert_eq!(grown, format!("{}\n{}\n", user_line(&resumed.prompt), r#"{"role":"assistant","content":"resumed"}"#), "{log}");
}

/// xorshift64*, seeded per case, so a failure replays from its seed alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }

    /// Write sizes of 1 to 13 bytes covering `len`.
    fn pieces(&mut self, len: usize) -> Vec<usize> {
        let mut pieces = Vec::new();
        let mut sum = 0;
        while sum < len {
            let size = 1 + self.below(13);
            pieces.push(size);
            sum += size;
        }
        pieces
    }
}

/// One SSE line of a template.
enum Line {
    Content(String),
    CallOpen { id: String, args: String },
    CallArgs(String),
    Finish(&'static str),
    Usage,
    Done,
}

impl Line {
    fn render(&self) -> String {
        match self {
            Line::Content(text) => content(text),
            Line::CallOpen { id, args } => {
                let call = json!([{"index":0,"id":id,"type":"function","function":{"name":"bash","arguments":args}}]);
                data(&chunk(json!({"tool_calls": call}), Value::Null))
            }
            Line::CallArgs(args) => data(&chunk(json!({"tool_calls": [{"index":0,"function":{"arguments":args}}]}), Value::Null)),
            Line::Finish(reason) => finish(reason),
            Line::Usage => data(&json!({"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":7}})),
            Line::Done => DONE.to_string(),
        }
    }
}

/// A rendered reply and where each of its lines lies: a line counts once
/// its first newline has arrived, so `spans[i].1` is one past that newline.
struct Template {
    name: &'static str,
    marker: String,
    call_id: String,
    lines: Vec<Line>,
    bytes: Vec<u8>,
    spans: Vec<(usize, usize)>,
}

impl Template {
    fn new(name: &'static str, marker: String, call_id: String, lines: Vec<Line>) -> Template {
        let (mut bytes, mut spans) = (Vec::new(), Vec::new());
        for line in &lines {
            let text = line.render();
            spans.push((bytes.len(), bytes.len() + text.len() - 1));
            bytes.extend_from_slice(text.as_bytes());
        }
        Template { name, marker, call_id, lines, bytes, spans }
    }

    /// Text, then a bash call whose argument pieces carry multi-byte
    /// characters, then its finish, usage, and `[DONE]`.
    fn a(seed: u64) -> Template {
        let marker = format!("s{seed}:é🦀");
        let call_id = format!("call_s{seed}");
        let (first, second, third) = split_args(&marker);
        Template::new("A", marker, call_id.clone(), vec![
            Line::Content("Writing ".into()),
            Line::Content("✓ 日本語 🦀 é".into()),
            Line::CallOpen { id: call_id, args: first },
            Line::CallArgs(second),
            Line::CallArgs(third),
            Line::Finish("tool_calls"),
            Line::Usage,
            Line::Done,
        ])
    }

    /// A bash call and nothing else: no reply text, so a cut anywhere before
    /// the finish is resent.
    fn b(seed: u64) -> Template {
        let marker = format!("b{seed}:é🦀");
        let call_id = format!("call_b{seed}");
        let (first, second, third) = split_args(&marker);
        Template::new("B", marker, call_id.clone(), vec![
            Line::CallOpen { id: call_id, args: first },
            Line::CallArgs(second),
            Line::CallArgs(third),
            Line::Finish("tool_calls"),
            Line::Done,
        ])
    }

    fn finish_line(&self) -> usize {
        self.lines.iter().position(|l| matches!(l, Line::Finish(_))).unwrap()
    }

    fn text(&self) -> String {
        String::from_utf8(self.bytes.clone()).unwrap()
    }
}

/// The arguments in three pieces, the last two with multi-byte characters.
fn split_args(marker: &str) -> (String, String, String) {
    let args = bash_args(marker);
    let (one, two) = (args.find('é').unwrap(), args.find(".log").unwrap());
    (args[..one].to_string(), args[one..two].to_string(), args[two..].to_string())
}

#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    /// The finish line arrived: the reply stands and its call runs once.
    Finished,
    /// Reply text arrived and the finish did not: the turn ends truncated
    /// with that text, and the call is kept, never run, when its opening
    /// line arrived (its arguments as far as they parse, else `{}`).
    Truncated { text: String, call: Option<String> },
    /// No reply text arrived: the request is resent.
    Resent,
}

impl Outcome {
    fn class(&self) -> &'static str {
        match self {
            Outcome::Finished => "finished",
            Outcome::Truncated { call: Some(_), .. } => "truncated with its call",
            Outcome::Truncated { call: None, .. } => "truncated before any call",
            Outcome::Resent => "resent",
        }
    }
}

/// The documented rule, read off the bytes the client got: only complete
/// lines count; a finish_reason line or `[DONE]` finishes the reply; a cut
/// before that is resent while no reply text has streamed and is a
/// truncation once some has.
fn oracle(template: &Template, k: usize) -> Outcome {
    let (mut text, mut args, mut opened) = (String::new(), String::new(), false);
    for (line, &(_, end)) in template.lines.iter().zip(&template.spans) {
        if end > k {
            break;
        }
        match line {
            Line::Finish(_) | Line::Done => return Outcome::Finished,
            Line::Content(piece) => text.push_str(piece),
            Line::CallOpen { args: piece, .. } => {
                opened = true;
                args.push_str(piece);
            }
            Line::CallArgs(piece) => args.push_str(piece),
            Line::Usage => {}
        }
    }
    if text.is_empty() {
        return Outcome::Resent;
    }
    let recorded = if serde_json::from_str::<Value>(&args).is_ok() { args } else { "{}".to_string() };
    Outcome::Truncated { text, call: opened.then_some(recorded) }
}

/// Where in a template a seeded cut lands.
#[derive(Clone, Copy, Debug)]
enum Stratum {
    BeforeFirstText,
    LaterText,
    CallOpening,
    ArgumentString,
    InsideCharacter,
    AfterFinish,
    BeforeFinish,
    LineEdge,
}

/// Offsets strictly inside an argument piece's JSON string.
fn argument_offsets(template: &Template) -> Vec<usize> {
    let mut offsets = Vec::new();
    for (line, &(start, end)) in template.lines.iter().zip(&template.spans) {
        if let Line::CallArgs(args) = line {
            let quoted = serde_json::to_string(args).unwrap();
            let line_bytes = &template.bytes[start..end];
            let at = start + line_bytes.windows(quoted.len()).position(|w| w == quoted.as_bytes()).unwrap();
            offsets.extend(at + 1..at + quoted.len() - 1);
        }
    }
    offsets
}

fn offset_in(template: &Template, stratum: Stratum, rng: &mut Rng) -> usize {
    let spans = &template.spans;
    let finish = spans[template.finish_line()];
    let pick = |offsets: Vec<usize>, rng: &mut Rng| offsets[rng.below(offsets.len())];
    match stratum {
        Stratum::BeforeFirstText => rng.below(spans[0].1),
        Stratum::LaterText => pick((spans[0].1..=spans[2].0).collect(), rng),
        Stratum::CallOpening => pick((spans[2].0 + 1..spans[2].1).collect(), rng),
        Stratum::ArgumentString => pick(argument_offsets(template), rng),
        // A continuation byte at k means the bytes before k end mid-character.
        Stratum::InsideCharacter => pick((1..finish.0).filter(|&k| template.bytes[k] & 0xC0 == 0x80).collect(), rng),
        Stratum::AfterFinish => pick((finish.1..=template.bytes.len()).collect(), rng),
        Stratum::BeforeFinish => rng.below(finish.1),
        Stratum::LineEdge => unreachable!("line edges are enumerated, not drawn"),
    }
}

struct CutCase {
    seed: u64,
    template: Template,
    stratum: Stratum,
    k: usize,
    framing: Framing,
    pieces: Vec<usize>,
}

impl CutCase {
    fn seeded(seed: u64, template: Template, stratum: Stratum) -> CutCase {
        let mut rng = Rng::new(seed);
        let k = offset_in(&template, stratum, &mut rng);
        let framing = [Framing::CloseDelimited, Framing::ShortLength, Framing::ChunkedNoEnd][rng.below(3)];
        let pieces = if rng.below(3) == 0 { rng.pieces(k) } else { Vec::new() };
        CutCase { seed, template, stratum, k, framing, pieces }
    }

    fn label(&self) -> String {
        let tail = String::from_utf8_lossy(&self.template.bytes[self.k.saturating_sub(80)..self.k]);
        format!(
            "seed {} template {} stratum {:?} k {} of {} framing {:?} {}; last bytes delivered: {tail:?}",
            self.seed,
            self.template.name,
            self.stratum,
            self.k,
            self.template.bytes.len(),
            self.framing,
            if self.pieces.is_empty() { "in one write" } else { "in pieces" },
        )
    }
}

/// Run one cut through a first turn and a resume and hold both to the
/// contract and to the oracle's outcome. Returns the outcome's class.
fn run_cut(case: &CutCase) -> &'static str {
    let template = &case.template;
    let outcome = oracle(template, case.k);
    let cut = Step::Cut { body: template.bytes.clone(), at: case.k, framing: case.framing, pieces: case.pieces.clone() };
    let resumed = Step::Finished(sse_text("resumed"));
    let script = match outcome {
        Outcome::Finished => vec![cut, Step::Finished(sse_text("done")), resumed],
        Outcome::Truncated { .. } => vec![cut, resumed],
        Outcome::Resent => vec![cut, Step::Finished(template.text()), Step::Finished(sse_text("done")), resumed],
    };
    let tag = format!("cut-{}-{}-{}", template.name, case.seed, case.k);
    let env = Env::new(&tag, &format!("{} expecting {:?}", case.label(), outcome), script);
    let run = env.first("go");
    let log = run.log();
    match &outcome {
        Outcome::Finished | Outcome::Resent => {
            assert_eq!(run.code, Some(0), "{log}");
            assert_eq!(run.stop_reason(), "stop", "{log}");
            assert_eq!(env.ran_log(), format!("{}\n", template.marker), "the call runs exactly once: {log}");
            assert_eq!(run.count("tool_start"), 1, "{log}");
            let resent = outcome == Outcome::Resent;
            assert_eq!(run.count("retry"), usize::from(resent), "{log}");
            assert_eq!(run.requests.len(), if resent { 3 } else { 2 }, "{log}");
            if resent && matches!(case.framing, Framing::CloseDelimited) {
                assert_eq!(run.retry_reasons(), [STREAM_ENDED], "{log}");
            }
        }
        Outcome::Truncated { text, call } => {
            assert_eq!(run.code, Some(1), "{log}");
            assert_eq!(run.stop_reason(), "truncated", "{log}");
            assert_eq!(run.requests.len(), 1, "{log}");
            assert_eq!(run.count("retry") + run.count("tool_start"), 0, "{log}");
            assert_eq!(env.ran_log(), "", "a call from an unfinished reply never runs: {log}");
            assert_eq!(run.message_done(), [text.as_str()], "{log}");
            let error = match call {
                Some(_) => format!("{INCOMPLETE}, so the 1 tool call it carried did not run"),
                None => INCOMPLETE.to_string(),
            };
            assert_eq!(run.errors(), [error.as_str()], "{log}");
            let messages = run.messages();
            let reply = messages.iter().rev().find(|m| m.role == "assistant").unwrap();
            assert_eq!(reply.content.as_deref(), Some(text.as_str()), "the partial text is kept: {log}");
            let kept: Vec<(String, String)> =
                reply.tool_calls.iter().flatten().map(|c| (c.id.clone(), c.function.arguments.clone())).collect();
            let expected: Vec<(String, String)> = call.iter().map(|args| (template.call_id.clone(), args.clone())).collect();
            assert_eq!(kept, expected, "the call is kept as far as it parsed: {log}");
        }
    }
    env.resume(&run, &run.after);
    outcome.class()
}

/// Run cases on up to six threads, each in its own HOME and server, and
/// report every failing case together, so one CI run names every seed.
fn run_all<T: Sync>(cases: &[T], label: impl Fn(&T) -> String + Sync, run: impl Fn(&T) + Sync) {
    let next = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..6 {
            scope.spawn(|| {
                while let Some(case) = cases.get(next.fetch_add(1, Ordering::SeqCst)) {
                    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(case))) {
                        let message = panic
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        failures.lock().unwrap().push(format!("{}\n{message}", label(case)));
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{} of {} cases failed:\n\n{}", failures.len(), cases.len(), failures.join("\n\n"));
}

/// A reply cut at any byte leaves a transcript that pairs every call with
/// its reply, a turn that ends the way the documented rule says (finished,
/// truncated with the text that arrived, or resent with identical bytes),
/// and a session that resumes byte for byte. Thirty seeded cuts sample six
/// strata of template A, and the three offsets around every line end are
/// cut too, where an off-by-one in "only complete lines count" would show.
#[test]
fn a_stream_cut_at_any_offset_keeps_the_transcript_paired_and_resumable() {
    let strata = [
        Stratum::BeforeFirstText,
        Stratum::LaterText,
        Stratum::CallOpening,
        Stratum::ArgumentString,
        Stratum::InsideCharacter,
        Stratum::AfterFinish,
    ];
    let mut cases: Vec<CutCase> = (1..=30u64).map(|seed| CutCase::seeded(seed, Template::a(seed), strata[seed as usize % strata.len()])).collect();
    let framings = [Framing::CloseDelimited, Framing::ShortLength, Framing::ChunkedNoEnd];
    for i in 0..Template::a(0).lines.len() * 3 {
        let seed = 1000 + i as u64;
        let template = Template::a(seed);
        // One byte short of a line's newline, exactly at it, and one past.
        // The framing turns with the line, so across the lines each offset
        // meets every framing: a clean close and a cut the client reports as
        // an error end the stream in different code, and either could count
        // a line that never got its newline.
        let k = (template.spans[i / 3].1 + i % 3 - 1).min(template.bytes.len());
        let framing = framings[(i + i / 3) % 3];
        cases.push(CutCase { seed, template, stratum: Stratum::LineEdge, k, framing, pieces: Vec::new() });
    }
    let classes = Mutex::new(BTreeSet::new());
    let positions = Mutex::new(BTreeSet::new());
    run_all(&cases, CutCase::label, |case| {
        let class = run_cut(case);
        classes.lock().unwrap().insert(class);
        positions.lock().unwrap().insert(format!("{:?}", case.stratum));
    });
    // A template edit that moved every cut out of a class would leave that
    // part of the rule untested while the suite still passed.
    let classes = classes.into_inner().unwrap();
    for class in ["finished", "truncated with its call", "truncated before any call", "resent"] {
        assert!(classes.contains(class), "no case reached {class}: {classes:?}");
    }
    let positions = positions.into_inner().unwrap();
    for stratum in ["ArgumentString", "InsideCharacter"] {
        assert!(positions.contains(stratum), "no case cut at {stratum}: {positions:?}");
    }
}

/// A reply that carries only a call has no reply text, so a cut anywhere
/// before its finish, inside the argument string or a multi-byte character
/// included, is resent with identical bytes, and the call runs exactly once.
#[test]
fn a_tool_call_only_stream_cut_anywhere_is_resent_and_runs_once() {
    let cases: Vec<CutCase> = (1..=6u64)
        .map(|seed| {
            let stratum = if seed % 6 == 0 { Stratum::AfterFinish } else { Stratum::BeforeFinish };
            CutCase::seeded(seed, Template::b(seed), stratum)
        })
        .collect();
    let classes = Mutex::new(BTreeSet::new());
    run_all(&cases, CutCase::label, |case| {
        // The case runs before the lock is taken: a guard held across it
        // would run the cases one at a time, and a failing case would poison
        // the set for every case after it.
        let class = run_cut(case);
        classes.lock().unwrap().insert(class);
    });
    assert_eq!(classes.into_inner().unwrap(), BTreeSet::from(["finished", "resent"]));
}

/// The events, the transcript, and the side effects of a reply do not
/// depend on how its bytes were split across reads, inside a line, between
/// `data:` and its JSON, or inside a multi-byte character: the client reads
/// only complete lines. Token boundaries may differ; their text may not.
#[test]
fn the_way_a_reply_is_split_across_reads_changes_nothing() {
    let seeds: Vec<u64> = (101..=108).collect();
    run_all(&seeds, |seed| format!("fragmentation seed {seed}"), |&seed| {
        let template = Template::a(seed);
        let reference = Env::new(&format!("whole-{seed}"), &format!("seed {seed} in one write"), vec![
            Step::Finished(template.text()),
            Step::Finished(sse_text("done")),
        ]);
        let pieces = Rng::new(seed).pieces(template.bytes.len());
        let ctx = format!("seed {seed} in pieces {pieces:?}");
        let split = Env::new(&format!("split-{seed}"), &ctx, vec![
            Step::Fragmented { body: template.bytes.clone(), pieces },
            Step::Finished(sse_text("done")),
        ]);
        let (whole, cut) = (reference.first("go"), split.first("go"));
        assert_eq!(cut.code, Some(0), "{}", cut.log());
        assert_eq!(cut.code, whole.code, "{ctx}");
        assert_eq!(split.ran_log(), format!("{}\n", template.marker), "{}", cut.log());
        assert_eq!(split.ran_log(), reference.ran_log(), "{ctx}");
        assert_eq!(shape(&cut.events), shape(&whole.events), "{ctx}");
        // Line 0 is the system prompt, which names the project path.
        let tail = |run: &Run| String::from_utf8(run.after.clone()).unwrap().lines().skip(1).map(str::to_string).collect::<Vec<_>>();
        assert_eq!(tail(&cut), tail(&whole), "{ctx}");
    });
}

/// The events with token boundaries joined and the per-run numbers left
/// out, for comparing two runs of the same reply.
fn shape(events: &[Value]) -> Vec<String> {
    let mut shape: Vec<String> = Vec::new();
    for event in events {
        let kind = event["type"].as_str().unwrap();
        let line = match kind {
            "budget" | "usage" => kind.to_string(),
            "token" if shape.last().is_some_and(|l| l.starts_with("token ")) => {
                let joined = format!("{}{}", shape.pop().unwrap(), event["text"].as_str().unwrap());
                shape.push(joined);
                continue;
            }
            "token" => format!("token {}", event["text"].as_str().unwrap()),
            _ => {
                let mut event = event.clone();
                event.as_object_mut().unwrap().remove("session_id");
                event.to_string()
            }
        };
        shape.push(line);
    }
    shape
}

/// Once reply text has streamed, a reply the provider fails is an error and
/// is not resent: the turn fails with the provider's message, keeps none of
/// the partial text, and never runs a call that streamed before the failure.
#[test]
fn an_error_chunk_after_reply_text_fails_the_turn_and_keeps_no_partial_text() {
    let failed = format!("{}{}data: {OVERLOADED}\n\n{DONE}", content("partial "), call("e1", "never"));
    let env = Env::new("error-after-text", "error chunk after reply text", vec![Step::Finished(failed), Step::Finished(sse_text("resumed"))]);
    let run = env.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(1), "{log}");
    assert_eq!(run.stop_reason(), "error", "{log}");
    assert_eq!(run.of("token").map(|e| e["text"].as_str().unwrap()).collect::<String>(), "partial ", "{log}");
    assert_eq!(run.errors(), ["backend reported an error (503): overloaded mid-reply"], "{log}");
    assert_eq!(run.count("message_done") + run.count("tool_start") + run.count("retry"), 0, "{log}");
    assert_eq!(run.requests.len(), 1, "{log}");
    assert_eq!(env.ran_log(), "", "{log}");
    let messages = run.messages();
    assert_eq!(messages.last().map(|m| m.role.as_str()), Some("user"), "only the prompt is kept: {log}");
    env.resume(&run, &run.after);
}

/// A 500 before any body, and a 503 the provider reports in its stream
/// before any reply text, are each resent once after the backoff with the
/// same bytes, and the turn ends with the one reply that followed.
#[test]
fn a_server_error_status_is_resent_once_with_identical_bytes() {
    let five_hundred = Step::Status {
        status: "500 Internal Server Error",
        headers: JSON_TYPE,
        body: r#"{"error":{"message":"upstream exploded"}}"#.into(),
    };
    let in_stream = Step::Finished(format!("data: {OVERLOADED}\n\n{DONE}"));
    let cases = [
        ("status-500", five_hundred, "backend returned 500 Internal Server Error: upstream exploded"),
        ("chunk-503", in_stream, "backend reported an error (503): overloaded mid-reply"),
    ];
    std::thread::scope(|scope| {
        for (tag, fault, reason) in cases {
            scope.spawn(move || {
                let env = Env::new(tag, tag, vec![fault, Step::Finished(sse_text("recovered")), Step::Finished(sse_text("resumed"))]);
                let run = env.first("go");
                let log = run.log();
                assert_eq!(run.code, Some(0), "{log}");
                assert_eq!(run.stop_reason(), "stop", "{log}");
                assert_eq!(run.retry_reasons(), [reason], "{log}");
                assert_eq!(run.of("retry").next().unwrap()["max_attempts"], 8, "{log}");
                assert_eq!(run.requests.len(), 2, "{log}");
                assert!(env.server.gap_after(0) >= Duration::from_millis(950), "the resend waits out the backoff: {log}");
                assert_eq!(run.messages().iter().filter(|m| m.role == "assistant").count(), 1, "{log}");
                env.resume(&run, &run.after);
            });
        }
    });
}

/// A 500 whose own body is cut short is reported as it came, without a
/// resend: the body failed before the retry decision could read it.
#[test]
fn a_server_error_whose_body_is_cut_fails_without_a_resend() {
    let head = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: 200\r\nConnection: close\r\n\r\n";
    let env = Env::new("cut-500", "a 500 with a cut body", vec![
        Step::Raw(format!("{head}{{\"error\":{{\"mes").into_bytes()),
        Step::Finished(sse_text("resumed")),
    ]);
    let run = env.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(1), "{log}");
    assert_eq!(run.stop_reason(), "error", "{log}");
    assert_eq!(run.count("retry"), 0, "{log}");
    assert_eq!(run.requests.len(), 1, "{log}");
    let errors = run.errors();
    assert!(
        errors.len() == 1 && errors[0].starts_with("backend returned 500 Internal Server Error: response body failed: error decoding response body"),
        "{log}"
    );
    env.resume(&run, &run.after);
}

/// A 429 is resent after the backoff, or after its Retry-After when that is
/// longer. The wait is measured at the server, from the moment the 429 went
/// out to the resend's arrival, so a slow machine can only lengthen it.
#[test]
fn a_rate_limit_waits_at_least_as_long_as_retry_after_asks() {
    let cases = [("429-floor", "Content-Type: application/json\r\n", 950), ("429-asks-2", "Content-Type: application/json\r\nRetry-After: 2\r\n", 1950)];
    std::thread::scope(|scope| {
        for (tag, headers, at_least) in cases {
            scope.spawn(move || {
                let limited = Step::Status { status: "429 Too Many Requests", headers, body: r#"{"error":{"message":"slow down"}}"#.into() };
                let env = Env::new(tag, tag, vec![limited, Step::Finished(sse_text("after the wait")), Step::Finished(sse_text("resumed"))]);
                let run = env.first("go");
                let log = run.log();
                assert_eq!(run.code, Some(0), "{log}");
                assert_eq!(run.retry_reasons(), ["backend returned 429 Too Many Requests: slow down"], "{log}");
                let gap = env.server.gap_after(0);
                assert!(gap >= Duration::from_millis(at_least), "waited {gap:?}, less than {at_least} ms: {log}");
                env.resume(&run, &run.after);
            });
        }
    });
}

/// A Retry-After past the one-minute cap, and a 429 for an exhausted quota,
/// fail the turn at once: no wait the turn would take lets a resend succeed.
#[test]
fn a_retry_after_past_a_minute_fails_at_once_naming_the_wait() {
    let cases = [
        (
            "429-asks-61",
            "Content-Type: application/json\r\nRetry-After: 61\r\n",
            r#"{"error":{"message":"slow down"}}"#,
            "backend returned 429 Too Many Requests: slow down (the server asks for a retry after 61s)",
        ),
        (
            "429-quota",
            JSON_TYPE,
            r#"{"error":{"message":"out of credit","type":"insufficient_quota"}}"#,
            "backend returned 429 Too Many Requests: out of credit",
        ),
    ];
    std::thread::scope(|scope| {
        for (tag, headers, body, error) in cases {
            scope.spawn(move || {
                let limited = Step::Status { status: "429 Too Many Requests", headers, body: body.into() };
                let env = Env::new(tag, tag, vec![limited, Step::Finished(sse_text("resumed"))]);
                let run = env.first("go");
                let log = run.log();
                assert_eq!(run.code, Some(1), "{log}");
                assert_eq!(run.stop_reason(), "error", "{log}");
                assert_eq!(run.count("retry"), 0, "{log}");
                assert_eq!(run.requests.len(), 1, "{log}");
                assert!(run.errors().len() == 1 && run.errors()[0].starts_with(error), "{log}");
                env.resume(&run, &run.after);
            });
        }
    });
}

/// A server that ignores `stream` and answers in one JSON body still drives
/// the loop: its call runs once, the next one-shot reply finishes the turn,
/// and both are on disk.
#[test]
fn a_one_shot_json_reply_is_kept_and_its_call_runs_once() {
    let env = Env::new("one-shot", "one-shot JSON replies", vec![
        Step::Status { status: "200 OK", headers: JSON_TYPE, body: json_reply(None, &[("j1", "one-shot")], "tool_calls") },
        Step::Status { status: "200 OK", headers: JSON_TYPE, body: json_reply(Some("done in one"), &[], "stop") },
        Step::Finished(sse_text("resumed")),
    ]);
    let run = env.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(0), "{log}");
    assert_eq!(env.ran_log(), "one-shot\n", "{log}");
    assert_eq!(run.message_done(), ["", "done in one"], "{log}");
    env.resume(&run, &run.after);
}

/// A one-shot JSON body cut short is an error, not a reply, and is not
/// resent: no call in it runs.
#[test]
fn a_one_shot_json_reply_cut_short_runs_nothing() {
    let body = json_reply(None, &[("j1", "never")], "tool_calls");
    let head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 300\r\nConnection: close\r\n\r\n";
    let env = Env::new("one-shot-cut", "a one-shot JSON reply cut short", vec![
        Step::Raw(format!("{head}{}", &body[..20]).into_bytes()),
        Step::Finished(sse_text("resumed")),
    ]);
    let run = env.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(1), "{log}");
    assert_eq!(run.stop_reason(), "error", "{log}");
    assert_eq!(run.count("retry") + run.count("tool_start"), 0, "{log}");
    assert!(run.errors().len() == 1 && run.errors()[0].contains("response body failed: error decoding response body"), "{log}");
    assert_eq!(env.ran_log(), "", "{log}");
    env.resume(&run, &run.after);
}

/// A complete one-shot JSON reply is parsed whatever content type labels
/// it, as stream_chat documents. Labeled text/plain, text/event-stream, or
/// nothing, it used to reach the stream parser, which read the choice's
/// `delta` where a whole reply carries `message`: an answer was dropped
/// while its finish_reason ended the turn cleanly (exit 0, an empty
/// message_done, nothing on disk), and a reply carrying a call failed the
/// turn as one with no call in it.
#[test]
fn a_bare_json_reply_is_parsed_whatever_its_content_type() {
    let labels = [("plain", "Content-Type: text/plain\r\n"), ("unlabeled", ""), ("event-stream", "Content-Type: text/event-stream\r\n")];
    let answer = "a bare answer ✓ 🦀";
    std::thread::scope(|scope| {
        for (label, headers) in labels {
            let bare = move |body: String| Step::Status { status: "200 OK", headers, body: format!("{body}\n") };
            scope.spawn(move || {
                let tag = format!("bare-text-{label}");
                let env = Env::new(&tag, &format!("a one-shot JSON answer labeled {label}"), vec![
                    bare(json_reply(Some(answer), &[], "stop")),
                    Step::Finished(sse_text("resumed")),
                ]);
                let run = env.first("go");
                let log = run.log();
                assert_eq!(run.code, Some(0), "{log}");
                assert_eq!(run.message_done(), [answer], "the answer reaches the caller: {log}");
                assert_eq!(run.messages().last().and_then(|m| m.content.as_deref()), Some(answer), "the answer is on disk: {log}");
                env.resume(&run, &run.after);
            });
            scope.spawn(move || {
                let tag = format!("bare-call-{label}");
                let env = Env::new(&tag, &format!("a one-shot JSON call labeled {label}"), vec![
                    bare(json_reply(None, &[("bare", label)], "tool_calls")),
                    bare(json_reply(Some(answer), &[], "stop")),
                    Step::Finished(sse_text("resumed")),
                ]);
                let run = env.first("go");
                let log = run.log();
                assert_eq!(run.code, Some(0), "{log}");
                assert_eq!(env.ran_log(), format!("{label}\n"), "the call runs once: {log}");
                assert_eq!(run.message_done(), ["", answer], "{log}");
                env.resume(&run, &run.after);
            });
        }
    });
}

/// The same one-shot reply without a newline after it. The stream parser
/// reads only complete lines, so the body never becomes a line: the client
/// resends it until the budget is spent (about a minute) and then reports
/// the reply as truncated, where stream_chat documents a one-shot parse.
/// Reading an unterminated final line would change the rule every cut above
/// relies on, that a stream's incomplete last line never counts, so it is
/// left for a decision on that rule. Here the script answers a resend with a
/// 400, so the test fails fast instead of waiting out the budget.
#[test]
#[ignore = "a one-shot JSON reply with no final newline under a non-JSON content type is resent and reported truncated"]
fn a_bare_json_reply_without_a_final_newline_is_parsed_too() {
    let env = Env::new("bare-unterminated", "a one-shot JSON answer with no final newline", vec![Step::Status {
        status: "200 OK",
        headers: "Content-Type: text/plain\r\n",
        body: json_reply(Some("unterminated"), &[], "stop"),
    }]);
    let run = env.first("go");
    assert_eq!(run.code, Some(0), "{}", run.log());
    assert_eq!(run.message_done(), ["unterminated"], "{}", run.log());
}

/// Either completion signal alone finishes a reply: a finish_reason with no
/// `[DONE]` after it (stop, length, or tool_calls), and `[DONE]` with no
/// finish_reason (read as stop, or tool_calls when calls came). A call in a
/// finished reply runs once and the loop goes on.
#[test]
fn a_finish_reason_without_done_and_done_without_a_finish_reason_both_finish_the_reply() {
    let closed = |body: String| Step::Cut { at: body.len(), body: body.into_bytes(), framing: Framing::CloseDelimited, pieces: Vec::new() };
    let cases = [
        ("finish-stop", closed(format!("{}{}", content("all of it"), finish("stop"))), "stop", ""),
        ("finish-length", closed(format!("{}{}", content("up to the cap"), finish("length"))), "length", ""),
        ("finish-calls", closed(format!("{}{}", call("f1", "finish-calls"), finish("tool_calls"))), "stop", "finish-calls\n"),
        ("done-only", Step::Finished(format!("{}{DONE}", content("all of it"))), "stop", ""),
        ("done-calls", Step::Finished(format!("{}{DONE}", call("d1", "done-calls"))), "stop", "done-calls\n"),
    ];
    std::thread::scope(|scope| {
        for (tag, reply, stop, ran) in cases {
            scope.spawn(move || {
                let mut script = vec![reply];
                if !ran.is_empty() {
                    script.push(Step::Finished(sse_text("after the call")));
                }
                script.push(Step::Finished(sse_text("resumed")));
                let env = Env::new(tag, tag, script);
                let run = env.first("go");
                let log = run.log();
                assert_eq!(run.code, Some(0), "{log}");
                assert_eq!(run.stop_reason(), stop, "{log}");
                assert_eq!(run.count("retry"), 0, "{log}");
                assert_eq!(env.ran_log(), ran, "{log}");
                env.resume(&run, &run.after);
            });
        }
    });
}

/// A line that cannot be read after the server finished its reply leaves
/// the reply standing: no retry, no usage, and the reply on disk.
#[test]
fn a_corrupt_usage_chunk_after_the_finish_leaves_the_reply_standing() {
    let reply = format!("{}{}data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\"\n\n{DONE}", content("kept"), finish("stop"));
    let env = Env::new("corrupt-usage", "a corrupt usage chunk after the finish", vec![Step::Finished(reply), Step::Finished(sse_text("resumed"))]);
    let run = env.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(0), "{log}");
    assert_eq!(run.stop_reason(), "stop", "{log}");
    assert_eq!(run.count("retry") + run.count("usage"), 0, "{log}");
    assert_eq!(run.requests.len(), 1, "{log}");
    assert_eq!(run.messages().last().and_then(|m| m.content.as_deref()), Some("kept"), "{log}");
    env.resume(&run, &run.after);
}

/// A provider that reuses a call id in a later iteration gets each call
/// answered in its own run: both execute once and both pairs survive a
/// resume. When the second reply is cut, its call is stubbed inside its own
/// run even though the id was answered earlier, and never runs.
#[test]
fn a_call_id_repeated_across_iterations_is_answered_in_its_own_run() {
    let second = format!("{}{}", content("again "), call("dup", "2"));
    let finished = Env::new("dup", "a repeated call id", vec![
        Step::Finished(sse_call("dup", "1")),
        Step::Finished(format!("{second}{}{DONE}", finish("tool_calls"))),
        Step::Finished(sse_text("third")),
        Step::Finished(sse_text("resumed")),
    ]);
    let run = finished.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(0), "{log}");
    assert_eq!(finished.ran_log(), "1\n2\n", "{log}");
    assert_eq!(run.of("tool_end").filter(|e| e["call_id"] == "dup").count(), 2, "{log}");
    finished.resume(&run, &run.after);

    let cut = second.len();
    let truncated = Env::new("dup-cut", "a repeated call id in a cut reply", vec![
        Step::Finished(sse_call("dup", "1")),
        Step::Cut { body: second.into_bytes(), at: cut, framing: Framing::CloseDelimited, pieces: Vec::new() },
        Step::Finished(sse_text("resumed")),
    ]);
    let run = truncated.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(1), "{log}");
    assert_eq!(run.stop_reason(), "truncated", "{log}");
    assert_eq!(truncated.ran_log(), "1\n", "{log}");
    let messages = run.messages();
    assert_eq!(messages.last().and_then(|m| m.content.as_deref()), Some(TRUNCATION_STUB), "{log}");
    assert_eq!(messages.last().and_then(|m| m.tool_call_id.as_deref()), Some("dup"), "{log}");
    truncated.resume(&run, &run.after);
}

/// An endpoint that reads the request and closes without a status line is
/// a transport fault after connecting: resent once, with the same bytes.
#[test]
fn a_connection_closed_before_any_byte_is_resent_once() {
    let env = Env::new("closed", "a connection closed before any byte", vec![
        Step::CloseBeforeAnyByte,
        Step::Finished(sse_text("second try")),
        Step::Finished(sse_text("resumed")),
    ]);
    let run = env.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(0), "{log}");
    let reasons = run.retry_reasons();
    assert!(reasons.len() == 1 && reasons[0].contains("connection closed before message completed"), "{log}");
    env.resume(&run, &run.after);
}

/// Silence for the idle timeout ends an attempt. Before any reply text it is
/// resent (no status line, or only reasoning, whose first attempt is void);
/// after reply text the turn is truncated with that text; after the finish
/// the reply stands; and keepalive comments are not silence.
#[test]
fn an_endpoint_silent_for_its_idle_timeout_is_resent_or_truncated_by_where_it_stopped() {
    let reasoning = |text: &str| data(&chunk(json!({"reasoning_content": text}), Value::Null));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let env = Env::idle("idle-no-status", vec![Step::Silent, Step::Finished(sse_text("after silence")), Step::Finished(sse_text("resumed"))]);
            let run = env.first("go");
            assert_eq!(run.code, Some(0), "{}", run.log());
            assert_eq!(run.retry_reasons(), [format!("request failed: {SILENCE}").as_str()], "{}", run.log());
            env.resume(&run, &run.after);
        });
        scope.spawn(|| {
            let fresh = format!("{}{}{}{DONE}", reasoning("fresh"), content("thought twice"), finish("stop"));
            let env = Env::idle("idle-reasoning", vec![Step::Stall(reasoning("abandoned")), Step::Finished(fresh), Step::Finished(sse_text("resumed"))]);
            let run = env.first("go");
            let log = run.log();
            assert_eq!(run.code, Some(0), "{log}");
            assert_eq!(run.retry_reasons(), [SILENCE], "{log}");
            let thinking: Vec<&str> = run.of("thinking").filter_map(|e| e["text"].as_str()).collect();
            assert_eq!(thinking, ["abandoned", "fresh"], "{log}");
            let messages = run.messages();
            let reply = messages.last().unwrap();
            assert_eq!(reply.reasoning_content.as_deref(), Some("fresh"), "only the final attempt's reasoning is kept: {log}");
            env.resume(&run, &run.after);
        });
        scope.spawn(|| {
            let env = Env::idle("idle-after-text", vec![Step::Stall(content("partial")), Step::Finished(sse_text("resumed"))]);
            let run = env.first("go");
            let log = run.log();
            assert_eq!(run.code, Some(1), "{log}");
            assert_eq!(run.stop_reason(), "truncated", "{log}");
            assert_eq!(run.count("retry"), 0, "{log}");
            assert_eq!(run.message_done(), ["partial"], "{log}");
            env.resume(&run, &run.after);
        });
        scope.spawn(|| {
            let env = Env::idle("idle-after-finish", vec![
                Step::Stall(format!("{}{}", content("whole"), finish("stop"))),
                Step::Finished(sse_text("resumed")),
            ]);
            let run = env.first("go");
            let log = run.log();
            assert_eq!(run.code, Some(0), "{log}");
            assert_eq!(run.count("retry"), 0, "{log}");
            assert_eq!(run.message_done(), ["whole"], "{log}");
            env.resume(&run, &run.after);
        });
        scope.spawn(|| {
            let env = Env::idle("idle-keepalive", vec![
                Step::Keepalive { every_ms: 250, count: 6, then: sse_text("kept alive") },
                Step::Finished(sse_text("resumed")),
            ]);
            let run = env.first("go");
            let log = run.log();
            assert_eq!(run.code, Some(0), "{log}");
            assert_eq!(run.count("retry"), 0, "keepalive comments are not silence: {log}");
            env.resume(&run, &run.after);
        });
    });
}

/// A turn that exits nonzero ends a chain of prompts: the prompts after it
/// never reach the endpoint or the transcript.
#[test]
fn a_truncated_first_prompt_skips_the_prompts_after_it() {
    let half = content("half");
    let env = Env::new("chain", "a truncated first prompt", vec![Step::Cut {
        at: half.len(),
        body: half.into_bytes(),
        framing: Framing::CloseDelimited,
        pieces: Vec::new(),
    }]);
    let run = env.run(&["--trust-project"], &["first", "second"], &[], &env.ctx);
    check_contract(&run);
    let log = run.log();
    assert_eq!(run.code, Some(1), "{log}");
    assert_eq!(run.stop_reason(), "truncated", "{log}");
    assert_eq!(run.requests.len(), 1, "{log}");
    assert!(!run.messages().iter().any(|m| m.content.as_deref() == Some("second")), "{log}");
}

/// The iteration cap stops a turn with every call it ran answered, and exits
/// 4, the code that says to resubmit; the resume carries on from there.
#[test]
fn a_turn_at_its_iteration_cap_exits_four_with_every_call_answered() {
    let env = Env::with_settings(
        "cap",
        "a turn at its iteration cap",
        vec![Step::Finished(sse_call("cap1", "capped")), Step::Finished(sse_text("resumed"))],
        r#","max_agent_iterations":1"#,
    );
    let run = env.first("go");
    let log = run.log();
    assert_eq!(run.code, Some(4), "{log}");
    assert_eq!(run.stop_reason(), "max_iterations", "{log}");
    assert_eq!(env.ran_log(), "capped\n", "{log}");
    assert_eq!(run.messages().last().and_then(|m| m.tool_call_id.as_deref()), Some("cap1"), "{log}");
    env.resume(&run, &run.after);
}

/// A record cut off mid-write never became a message: the resume does not
/// send it, the next save removes it, and the complete records before it
/// stay as they were.
#[test]
fn a_torn_final_record_is_skipped_on_resume_and_cut_by_the_next_save() {
    let env = Env::new("torn", "a torn final record", vec![Step::Finished(sse_text("whole")), Step::Finished(sse_text("resumed"))]);
    let run = env.first("go");
    let mut torn = run.after.clone();
    torn.extend_from_slice(br#"{"role":"assistant","content":"torn mid-wri"#);
    std::fs::write(env.transcript(&run.sid), &torn).unwrap();
    let resumed = env.resume(&run, &torn);
    assert!(!resumed.requests[0].contains("torn mid-wri"), "{}", resumed.log());
    assert!(!String::from_utf8_lossy(&resumed.after).contains("torn mid-wri"), "{}", resumed.log());
}

/// Text mode exits with the code JSON mode does, and says why on stderr: one
/// error line per error, and `stopped` for an end that is not a clean stop,
/// unless the error line already said it.
#[test]
fn text_mode_exits_as_json_mode_does_and_names_the_failure_once() {
    let truncated = || {
        let half = content("half");
        Step::Cut { at: half.len(), body: half.into_bytes(), framing: Framing::CloseDelimited, pieces: Vec::new() }
    };
    let failed = || Step::Finished(format!("{}data: {OVERLOADED}\n\n{DONE}", content("partial ")));
    // Each mode gets its own endpoint, so each builds its own copy of the fault.
    type Fault = fn() -> Step;
    let cases: [(&str, Fault, Vec<String>); 3] = [
        ("text-clean", || Step::Finished(sse_text("fine")), vec![]),
        ("text-truncated", truncated, vec![format!("openmax: error: {INCOMPLETE}"), "openmax: stopped (truncated)".into()]),
        ("text-failed", failed, vec!["openmax: error: backend reported an error (503): overloaded mid-reply".into()]),
    ];
    std::thread::scope(|scope| {
        for (tag, step, lines) in cases {
            scope.spawn(move || {
                let json_mode = Env::new(&format!("{tag}-json"), tag, vec![step()]);
                let run = json_mode.first("go");
                let text_mode = Env::new(tag, tag, vec![step()]);
                let (code, stderr) = text_mode.text("go");
                assert_eq!(code, run.code, "{tag}: {stderr}");
                let said: Vec<&str> = stderr.lines().filter(|l| l.starts_with("openmax: error:") || l.starts_with("openmax: stopped")).collect();
                assert_eq!(said, lines, "{tag}: {stderr}");
            });
        }
    });
}
