//! Instruments for two loop metrics the thesis names: the harness time a
//! tool round costs, and how often a request's frozen prefix (the system
//! message and the tool schemas) is byte-identical to the request before it,
//! which is what a provider's prompt cache keys on.
//!
//! Both drive the real turn loop in process against a scripted
//! OpenAI-compatible SSE endpoint on a loopback socket. The endpoint keeps
//! connections alive, frames every reply with a Content-Length, stamps each
//! request with the moment its first byte arrived and the moment the last
//! byte of its reply was written, all on this test's clock, and stops
//! accepting when the test drops it. Each test prints its measurement. The
//! tool-round test asserts its 250 us target in a release build and a 20x
//! sanity bound in debug; the prefix test asserts a ratchet the tree meets
//! today, which tightens toward its target as the loop gets there. The two
//! take one lock, so neither runs while the other loads the CPU.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use open_max_core::agent::start_turn;
use open_max_core::state::{CancelToken, Core};
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use serde_json::value::RawValue;
use serde_json::{json, Value};

/// Held for the whole of each test: the tool-round gaps are CPU-bound, and
/// the prefix test's writes and refreeze would otherwise run beside them.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// One request the endpoint served.
struct Served {
    body: String,
    /// When the first byte of the request arrived.
    first_byte: Instant,
    /// When the last byte of the reply was written.
    replied: Instant,
}

/// A running scripted endpoint: its base URL and the requests it served.
struct Endpoint {
    base_url: String,
    served: Arc<Mutex<Vec<Served>>>,
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    accept: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Endpoint {
    /// Stops the accept thread and joins it. The flag is set first, then a
    /// connection of the endpoint's own wakes the blocked accept so the
    /// thread sees the flag; a blocking accept, not a polled one, keeps a
    /// poll interval out of the measured gaps. A connection already served
    /// ends when the client drops it.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if TcpStream::connect(self.addr).is_ok() {
            if let Some(accept) = self.accept.take() {
                let _ = accept.join();
            }
        }
    }
}

/// A loopback endpoint that answers request N with the Nth scripted SSE
/// body. A request past the script gets a 400, which the client does not
/// resend, so a regression that adds requests fails the count at once.
fn endpoint(script: Vec<String>) -> Endpoint {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base_url = format!("http://{addr}/v1");
    let served = Arc::new(Mutex::new(Vec::new()));
    let script = Arc::new(Mutex::new(script.into_iter()));
    let stop = Arc::new(AtomicBool::new(false));
    let (record, stopping) = (served.clone(), stop.clone());
    let accept = std::thread::spawn(move || {
        for stream in listener.incoming() {
            if stopping.load(Ordering::SeqCst) {
                return;
            }
            let Ok(stream) = stream else { return };
            let (script, record) = (script.clone(), record.clone());
            std::thread::spawn(move || serve(stream, &script, &record));
        }
    });
    Endpoint { base_url, served, addr, stop, accept: Some(accept) }
}

fn serve(
    mut stream: TcpStream,
    script: &Mutex<std::vec::IntoIter<String>>,
    record: &Mutex<Vec<Served>>,
) {
    let _ = stream.set_nodelay(true);
    let mut buf = Vec::new();
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let mut first_byte = None;
        let body = loop {
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
                let len = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if buf.len() >= end + 4 + len {
                    let body = String::from_utf8_lossy(&buf[end + 4..end + 4 + len]).to_string();
                    buf.drain(..end + 4 + len);
                    break body;
                }
            }
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    first_byte.get_or_insert_with(Instant::now);
                    buf.extend_from_slice(&chunk[..n]);
                }
            }
        };
        let first_byte = first_byte.unwrap_or_else(Instant::now);
        let response = match script.lock().unwrap().next() {
            Some(sse) => format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{sse}",
                sse.len()
            ),
            None => "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
        };
        // Locked before the reply goes out: a test that takes the record once
        // its turn is done then waits for this push instead of missing it.
        let mut log = record.lock().unwrap();
        if stream.write_all(response.as_bytes()).is_err() {
            return;
        }
        let replied = Instant::now();
        log.push(Served { body, first_byte, replied });
    }
}

fn data(delta: Value, finish_reason: Value) -> String {
    let chunk = json!({"choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]});
    format!("data: {chunk}\n\n")
}

fn sse_call(id: &str, name: &str, args: Value) -> String {
    let call = json!([{"index":0,"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}}]);
    format!(
        "{}{}data: [DONE]\n\n",
        data(json!({"role":"assistant","tool_calls":call}), Value::Null),
        data(json!({}), json!("tool_calls"))
    )
}

fn sse_text(text: &str) -> String {
    format!(
        "{}{}data: [DONE]\n\n",
        data(json!({"role":"assistant","content":text}), Value::Null),
        data(json!({}), json!("stop"))
    )
}

/// A trusted project with `files` small source files, and a data dir whose
/// settings point at `base_url` with approvals off.
fn session_dirs(tag: &str, base_url: &str, files: usize) -> (PathBuf, PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("omx-{tag}-{}", uuid::Uuid::new_v4()));
    let (data, project) = (dir.join("data"), dir.join("project"));
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&data).unwrap();
    let project = project.canonicalize().unwrap();
    for i in 0..files {
        let body: String =
            (0..40).map(|l| format!("fn item_{i}_{l}() -> u32 {{ {l} }}\n")).collect();
        std::fs::write(project.join(format!("f{i}.rs")), body).unwrap();
    }
    std::fs::write(
        data.join("settings.json"),
        json!({
            "base_url": base_url,
            "model": "scripted",
            "approval_mode": "auto",
            "context_tokens": 262_144,
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        data.join("trust.json"),
        json!({ "version": 1, "projects": [project.to_string_lossy()] }).to_string(),
    )
    .unwrap();
    (dir, data, project)
}

async fn drive_turn(
    core: &Arc<Core>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session: &str,
    project: &Path,
    text: &str,
) -> Vec<AgentEvent> {
    start_turn(Arc::clone(core), session.into(), project.to_path_buf(), text.into()).unwrap();
    let mut events = Vec::new();
    loop {
        let envelope = tokio::time::timeout(Duration::from_secs(60), rx.recv())
            .await
            .expect("turn finishes within 60s")
            .expect("event channel stays open");
        if matches!(envelope.event, AgentEvent::Done { .. }) {
            break;
        }
        events.push(envelope.event);
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while core.is_running(session) {
        assert!(Instant::now() < deadline, "session {session} still runs 10s after its Done event");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    events
}

/// The `ToolEnd` events among `events`, as (call id, ok).
fn tool_ends(events: &[AgentEvent]) -> Vec<(String, bool)> {
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolEnd { call_id, ok, .. } => Some((call_id.clone(), *ok)),
            _ => None,
        })
        .collect()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Harness time per tool round, median over 30 read_file rounds in one turn.
///
/// Method: a five-round warmup turn runs first in a separate session on the
/// same `Core`, so cold dispatch and first-touch allocation stay out of the
/// sample. Then the endpoint answers 30 requests with one read_file call
/// each (a different 40-line file per round) and the 31st with text. For
/// each round the gap runs from the moment the endpoint wrote the last byte
/// of the reply carrying the call to the moment the first byte of the next
/// request arrived: the client's parse of the stream, dispatch, the tool,
/// the transcript persisted, the next request built and sent, plus the
/// loopback wake-up on both sides and the endpoint thread's own stamping.
/// read_file's own cost is then measured apart, as the median of 31
/// `tools::execute` calls on the same 30 files in this process after a
/// warmup pass, and subtracted from every gap. Prints n, median, p25 and
/// p75. A brief spike of other load on the machine lands in a few gaps and
/// the median over 30 rounds absorbs it; sustained load lands in every gap
/// and moves the median with it.
///
/// Target 250 us. Measured on this tree on an Apple-silicon laptop with
/// other builds running. Fifteen release runs back to back: fourteen
/// medians of 160 to 207 us (p25 139 to 175, p75 165 to 233) after
/// subtracting a 35 to 46 us read_file, and one of 360 us. The target is
/// sensitive to sustained load: nine runs started the moment their build
/// finished beside other compiles measured 170 and 180 us, then 274 to
/// 925 us, with read_file itself at 68 to 143 us. Fifteen debug runs:
/// medians of 868 to 1024 us after a 46 to 63 us read_file. A release build
/// asserts the target on the median; a debug build asserts a 20x sanity
/// bound on the median.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn harness_time_per_tool_round() {
    const WARMUP: usize = 5;
    const ROUNDS: usize = 30;
    const TARGET_US: f64 = 250.0;
    let _serial = SERIAL.lock().await;
    let mut script = Vec::new();
    for rounds in [WARMUP, ROUNDS] {
        script.extend((0..rounds).map(|i| {
            sse_call(&format!("call-{i}"), "read_file", json!({ "path": format!("f{i}.rs") }))
        }));
        script.push(sse_text("done"));
    }
    let server = endpoint(script);
    let (dir, data, project) = session_dirs("rounds", &server.base_url, ROUNDS);

    let (core, mut rx) = Core::new(data.clone()).unwrap();
    for (session, rounds) in [("rounds-warmup", WARMUP), ("rounds", ROUNDS)] {
        let events = drive_turn(&core, &mut rx, session, &project, "read every file").await;
        let ends = tool_ends(&events);
        assert_eq!(ends.len(), rounds, "{session}: every scripted read_file ran");
        assert!(ends.iter().all(|(_, ok)| *ok), "{session}: every read_file succeeded: {ends:?}");
    }

    let mut served = std::mem::take(&mut *server.served.lock().unwrap());
    served.sort_by_key(|s| s.first_byte);
    assert_eq!(served.len(), WARMUP + 1 + ROUNDS + 1, "one request per round plus each turn's answer");
    let measured = &served[WARMUP + 1..];

    // read_file's own cost, the same calls outside the loop.
    let read = |i: usize| {
        let args = json!({ "path": format!("f{i}.rs") });
        let (data, project) = (data.clone(), project.clone());
        async move {
            let started = Instant::now();
            let out = open_max_core::tools::execute(
                "read_file",
                &args,
                &data,
                &project,
                open_max_core::tools::OutputCaps::default(),
                Arc::new(CancelToken::default()),
            )
            .await;
            assert!(out.ok, "read_file of f{i}.rs succeeds");
            started.elapsed()
        }
    };
    for i in 0..ROUNDS {
        read(i).await;
    }
    let mut tool_us = Vec::new();
    for i in 0..=ROUNDS {
        tool_us.push(micros(read(i % ROUNDS).await));
    }
    tool_us.sort_by(f64::total_cmp);
    let tool_median = quantile(&tool_us, 0.5);

    let mut gaps: Vec<f64> = measured
        .windows(2)
        .map(|w| micros(w[1].first_byte.duration_since(w[0].replied)) - tool_median)
        .collect();
    gaps.sort_by(f64::total_cmp);
    let (p25, median, p75) = (quantile(&gaps, 0.25), quantile(&gaps, 0.5), quantile(&gaps, 0.75));
    println!(
        "MEASURE harness_time_per_tool_round n={} median={median:.0}us p25={p25:.0}us p75={p75:.0}us \
         (read_file median {tool_median:.1}us subtracted; target {TARGET_US}us, {})",
        gaps.len(),
        if cfg!(debug_assertions) { "debug" } else { "release" },
    );
    if cfg!(debug_assertions) {
        assert!(median < TARGET_US * 20.0, "debug harness time per round {median:.0}us is past the 20x sanity bound");
    } else {
        assert!(median <= TARGET_US, "harness time per round {median:.0}us is past the {TARGET_US}us target");
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The share of requests whose frozen prefix is byte-identical to the
/// request before it.
///
/// Method: one session of two turns and 8 requests. Turn one writes a tool
/// manifest (a refreeze), writes a project memory, reads a file and answers;
/// turn two reads three files and answers. The test checks that all six
/// tool calls succeeded, that both files exist, and that the new tool's
/// index line is absent from the first request's last message and present
/// in the last request's, where the trailing note carries it, and never in
/// any request's `tools`, since a write that failed or a manifest the
/// registry refused would leave the prefix alone and raise the rate. A
/// request is a hit when the raw wire bytes of its `messages[0]` and of its
/// `tools` array, sliced out of the body unparsed, both equal the previous
/// request's; the rate is hits over the 7 requests that have a predecessor.
/// Prints the rate and which requests missed.
///
/// Target 1.0. Measured on this tree: 7/7 (1.00), a count, so it holds in
/// debug and release alike, and the ratchet asserts it. The tool write
/// moves no prefix byte: the tools array keeps the tools the session began
/// with until /reload, and the new tool reaches the model on its line in
/// the note, with the bash form that runs it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefix_cache_hit_rate() {
    const RATCHET_HITS: usize = 7;
    let _serial = SERIAL.lock().await;
    let manifest = "name = \"wordfreq\"\ndescription = \"top-n words\"\ncommand = \"/bin/echo\"\nmutating = false\n";
    let (tool_path, memory_path) = (".openmax/tools/wordfreq.toml", ".openmax/memory/build-command.md");
    let script = vec![
        sse_call("t1", "write_file", json!({ "path": tool_path, "content": manifest })),
        sse_call("m1", "write_file", json!({ "path": memory_path, "content": "Build with cargo build --release.\n" })),
        sse_call("r0", "read_file", json!({ "path": "f0.rs" })),
        sse_text("done"),
        sse_call("r1", "read_file", json!({ "path": "f1.rs" })),
        sse_call("r2", "read_file", json!({ "path": "f2.rs" })),
        sse_call("r3", "read_file", json!({ "path": "f3.rs" })),
        sse_text("done again"),
    ];
    let requests = script.len();
    let server = endpoint(script);
    let (dir, data, project) = session_dirs("prefix", &server.base_url, 4);

    let (core, mut rx) = Core::new(data).unwrap();
    let mut events = drive_turn(&core, &mut rx, "prefix", &project, "add a tool and remember the build").await;
    events.extend(drive_turn(&core, &mut rx, "prefix", &project, "now read three files").await);
    let expected: Vec<(String, bool)> =
        ["t1", "m1", "r0", "r1", "r2", "r3"].iter().map(|id| (id.to_string(), true)).collect();
    assert_eq!(tool_ends(&events), expected, "every scripted tool call ran and succeeded");
    for path in [tool_path, memory_path] {
        assert!(project.join(path).is_file(), "the session wrote {path}");
    }

    let mut served = std::mem::take(&mut *server.served.lock().unwrap());
    served.sort_by_key(|s| s.first_byte);
    assert_eq!(served.len(), requests, "the scripted session makes exactly {requests} requests");
    // The exact bytes the client wrote: a parse into `Value` and back would
    // sort keys and normalize escapes, hiding a reorder in the frozen wire.
    let (prefixes, lasts): (Vec<(String, String)>, Vec<String>) = served
        .iter()
        .map(|s| {
            let top: std::collections::HashMap<String, Box<RawValue>> =
                serde_json::from_str(&s.body).unwrap();
            let messages: Vec<&RawValue> = serde_json::from_str(top["messages"].get()).unwrap();
            let tools = top.get("tools").map(|t| t.get().to_owned()).unwrap_or_default();
            ((messages[0].get().to_owned(), tools), messages.last().unwrap().get().to_owned())
        })
        .unzip();
    assert!(
        prefixes.iter().all(|(system, _)| system.contains("\"role\":\"system\"")),
        "every request opens with a system message"
    );
    let index_line = "- wordfreq: top-n words {";
    assert!(
        !lasts[0].contains(index_line) && lasts[requests - 1].contains(index_line),
        "the written tool reaches the wire: its index line is absent from request 1 and present in request {requests}"
    );
    assert!(
        prefixes.iter().all(|(_, tools)| !tools.contains("\"name\":\"wordfreq\"")),
        "a tool file written mid-session never joins the tools array"
    );
    let misses: Vec<usize> =
        (1..prefixes.len()).filter(|&i| prefixes[i] != prefixes[i - 1]).map(|i| i + 1).collect();
    let hits = prefixes.len() - 1 - misses.len();
    let rate = hits as f64 / (prefixes.len() - 1) as f64;
    println!(
        "MEASURE prefix_cache_hit_rate {hits}/{} = {rate:.2} (requests whose prefix changed: {misses:?}; target 1.0, ratchet {RATCHET_HITS})",
        prefixes.len() - 1,
    );
    assert!(hits >= RATCHET_HITS, "prefix cache hits {hits} fell below the ratchet {RATCHET_HITS} (misses at requests {misses:?})");
    let _ = std::fs::remove_dir_all(dir);
}
