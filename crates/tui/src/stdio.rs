//! Bidirectional stdio client: the machine-facing counterpart of the TUI.
//! JSONL commands arrive on stdin, `AgentEvent` envelopes leave on stdout:
//! the full custom-frontend and self-spawn contract. Any process that can
//! speak line-delimited JSON (an editor plugin, an orchestrator, another
//! openmax) can drive a complete interactive session, approvals included.
//!
//! Protocol (`openmax-stdio/1`), one JSON object per line. The normative
//! reference (every field of every line) lives in README under "stdio
//! protocol"; `crates/core/src/types.rs` golden tests pin the event wire.
//!
//! stdin commands:
//!   {"cmd":"user","text":"..."}                      start a turn
//!   {"cmd":"approve","approval_id":"...","approved":true}
//!   {"cmd":"approval_mode","mode":"auto|ask|readonly"}
//!   {"cmd":"reload"}                                 re-freeze tools/skills/prompt
//!   {"cmd":"cancel"}                                 cancel the running turn
//!   {"cmd":"quit"}                                   finish the turn, then exit
//!
//! stdout lines:
//!   {"type":"hello","proto":"openmax-stdio/1","protocol_version":1,"session_id":"...","version":"...","project":"..."}
//!   AgentEvent envelopes exactly as `--print --json` emits them
//!   {"type":"protocol_error","message":"..."}        bad input; session unharmed
//!
//! `protocol_version` is an integer a client can compare directly; `proto`
//! carries the same major as a human-readable id. `openmax --check --stdio`
//! validates a JSONL stream of these lines against the contract.
//!
//! EOF on stdin behaves like quit: the in-flight turn drains, then the
//! process exits, so `echo '{"cmd":"user",...}' | openmax --stdio` works as
//! a one-shot. Unlike print mode, approvals are never auto-declined while
//! the client is live: the ApprovalRequest event goes to the client, which
//! answers with approve. Once quit or EOF arrives, pending and subsequent
//! approvals are declined so shutdown drains promptly instead of stalling
//! on the approval timeout.

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use open_max_core::agent;
use open_max_core::sessions;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use serde::Deserialize;
use tokio::sync::mpsc;

pub const PROTO: &str = "openmax-stdio/1";
/// Machine-comparable protocol major. A client negotiates on this integer;
/// `PROTO` embeds the same number as a human-readable id (checked in tests).
/// Bump on any wire change (event field, command shape, framing line).
pub const PROTO_VERSION: u32 = 1;

// Unknown `cmd` values are protocol errors; extra fields on a known command
// are ignored (lenient by design, so clients can annotate lines freely).
#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    User { text: String },
    Approve { approval_id: String, approved: bool },
    /// Set the approval gate for mutating tools; persisted like /approvals.
    ApprovalMode { mode: String },
    /// Re-freeze tools, skills, and prompt from current config, like /reload.
    Reload,
    Cancel,
    Quit,
}

pub struct StdioArgs {
    pub continue_session: bool,
}

pub async fn run(
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    args: StdioArgs,
) -> i32 {
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_key = project.display().to_string();

    let (session_id, continued) = if args.continue_session {
        match sessions::latest(&core, &project_key) {
            Some(meta) => (meta.id, true),
            None => {
                eprintln!("openmax: no prior session in this directory to continue");
                return 2;
            }
        }
    } else {
        match sessions::create(&core, project_key.clone()) {
            Ok(meta) => (meta.id, false),
            Err(e) => {
                eprintln!("openmax: failed to create session: {e}");
                return 1;
            }
        }
    };

    let mut stdout = std::io::stdout();
    emit(&mut stdout, &hello_value(&session_id, &project_key, continued));
    if continued {
        // One bounded history line so an attaching frontend can render what
        // came before, without replaying synthetic live events (a replayed
        // `token` stream would be indistinguishable from a running turn).
        let messages = sessions::load_messages(&core, &session_id).unwrap_or_default();
        emit(&mut stdout, &transcript_value(&session_id, &messages));
    }
    let stdin_rx = spawn_stdin_reader();
    drive(core, core_rx, session_id, project, stdin_rx, &mut stdout).await
}

/// The blocking stdin reader on its own thread; malformed input travels as
/// Err so the async loop can answer without ever blocking on the pipe.
fn spawn_stdin_reader() -> mpsc::Receiver<Result<Command, String>> {

    // The channel is bounded because every line, valid or not, costs one
    // answer written and flushed one at a time: a peer that floods faster
    // than the loop drains would otherwise queue its own backlog until
    // memory runs out. A full queue parks the reader, stdin backs up, and
    // the peer waits.
    let (stdin_tx, stdin_rx) = mpsc::channel::<Result<Command, String>>(64);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut reader = stdin.lock();
        let mut buf = Vec::new();
        loop {
            let parsed = match read_line_capped(&mut reader, &mut buf) {
                Ok(LineRead::Eof) => break,
                // A peer that never sends a newline must not be able to grow
                // this buffer without bound. The rest of the line is drained,
                // so framing survives and the next command still parses.
                Ok(LineRead::TooLong) => Err(format!(
                    "command line exceeds {MAX_LINE_BYTES} bytes and was discarded"
                )),
                Ok(LineRead::Line) => match std::str::from_utf8(&buf) {
                    // Reporting beats the silent shutdown a lossy read would
                    // hide: one bad byte is a bad line, not a dead session.
                    Err(e) => Err(format!("command line is not valid UTF-8: {e}")),
                    Ok(line) if line.trim().is_empty() => continue,
                    Ok(line) => serde_json::from_str::<Command>(line)
                        .map_err(|e| format!("bad command line: {e}")),
                },
                Err(e) => {
                    let _ = stdin_tx.blocking_send(Err(format!("stdin read failed: {e}")));
                    break;
                }
            };
            if stdin_tx.blocking_send(parsed).is_err() {
                break;
            }
        }
        // Dropping the sender closes the channel: EOF.
    });
    stdin_rx
}

/// The protocol loop, separated from real stdin/stdout so tests can drive a
/// session in-process: commands arrive on a channel, every emitted line goes
/// to `out`.
async fn drive<W: Write>(
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session_id: String,
    project: PathBuf,
    mut stdin_rx: mpsc::Receiver<Result<Command, String>>,
    out: &mut W,
) -> i32 {
    let mut running = false;
    let mut closing = false;
    let mut exit_code = 0i32;
    // Approvals awaiting a client answer; declined in bulk when the client
    // quits so the drain never sits out the approval timeout.
    let mut open_approvals: std::collections::HashSet<String> = std::collections::HashSet::new();

    loop {
        if closing && !running {
            return exit_code;
        }
        tokio::select! {
            cmd = stdin_rx.recv(), if !closing => {
                match cmd {
                    None | Some(Ok(Command::Quit)) => {
                        closing = true;
                        for id in open_approvals.drain() {
                            core.respond_approval(&id, false);
                        }
                    }
                    Some(Err(message)) => protocol_error(out, &message),
                    Some(Ok(Command::Cancel)) => core.cancel(&session_id),
                    Some(Ok(Command::Approve { approval_id, approved })) => {
                        // An id that was never issued (or already settled) is
                        // a client-state bug worth reporting, not a no-op:
                        // the client believes a gate is still open.
                        if open_approvals.remove(&approval_id) {
                            core.respond_approval(&approval_id, approved);
                        } else {
                            protocol_error(
                                out,
                                &format!("unknown or already settled approval_id '{approval_id}'"),
                            );
                        }
                    }
                    Some(Ok(Command::ApprovalMode { mode })) => {
                        match open_max_core::config::ApprovalMode::parse(&mode) {
                            Some(parsed) => {
                                {
                                    let mut s = core.settings.lock().unwrap();
                                    s.approval_mode = parsed;
                                    let _ = open_max_core::config::save(&core.data_dir, &s);
                                }
                                emit(
                                    out,
                                    &serde_json::json!({
                                        "type": "approval_mode",
                                        "mode": parsed.as_str(),
                                    }),
                                );
                            }
                            None => protocol_error(
                                out,
                                &format!("unknown approval mode '{mode}': auto|ask|readonly"),
                            ),
                        }
                    }
                    Some(Ok(Command::Reload)) => {
                        if running {
                            protocol_error(out, "a turn is in flight; wait for done");
                            continue;
                        }
                        match agent::reload_session(&core, &session_id, &project).await {
                            Ok((tools, skills)) => {
                                let env = AgentEventEnvelope {
                                    session_id: session_id.clone(),
                                    event: AgentEvent::Refrozen { tools, skills },
                                };
                                if let Ok(value) = serde_json::to_value(&env) {
                                    emit(out, &value);
                                }
                            }
                            Err(e) => protocol_error(out, &e),
                        }
                    }
                    Some(Ok(Command::User { text })) => {
                        if text.trim().is_empty() {
                            refuse(out, &session_id, "user text is empty");
                            continue;
                        }
                        if running || !crate::headless::wait_until_idle(&core, &session_id).await {
                            // The in-flight turn owns the next `done`; a second
                            // one here would tell the client that turn ended.
                            protocol_error(out, "a turn is in flight; wait for done");
                            continue;
                        }
                        match agent::start_turn(core.clone(), session_id.clone(), project.clone(), text) {
                            Ok(()) => running = true,
                            Err(e) => refuse(out, &session_id, &e),
                        }
                    }
                }
            }
            event = core_rx.recv() => {
                let Some(event) = event else {
                    eprintln!("openmax: event channel closed");
                    return 1;
                };
                let env = event;
                if env.session_id != session_id {
                    continue;
                }
                if let Ok(value) = serde_json::to_value(&env) {
                    emit(out, &value);
                }
                match &env.event {
                    AgentEvent::ApprovalRequest { approval_id, .. } => {
                        if closing {
                            // Nobody is left to answer; decline immediately.
                            core.respond_approval(approval_id, false);
                        } else {
                            open_approvals.insert(approval_id.clone());
                        }
                    }
                    AgentEvent::ApprovalSettled { approval_id, .. } => {
                        open_approvals.remove(approval_id);
                    }
                    AgentEvent::Done { stop_reason } => {
                        running = false;
                        if stop_reason == "error" {
                            exit_code = 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn emit<W: Write>(stdout: &mut W, value: &serde_json::Value) {
    if let Ok(line) = serde_json::to_string(value) {
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }
}

fn protocol_error<W: Write>(stdout: &mut W, message: &str) {
    emit(
        stdout,
        &serde_json::json!({ "type": "protocol_error", "message": message }),
    );
}

/// Refuse a `user` command that starts no turn: say why, then close it out
/// with the terminator the client is waiting on. Without the `done`, a client
/// that follows the documented rule (block until `done`) hangs forever on a
/// prompt nothing will ever answer, which an untrusted project reproduces on
/// the very first command.
fn refuse<W: Write>(stdout: &mut W, session_id: &str, message: &str) {
    protocol_error(stdout, message);
    let env = AgentEventEnvelope {
        session_id: session_id.to_string(),
        event: AgentEvent::Done { stop_reason: "refused".into() },
    };
    if let Ok(value) = serde_json::to_value(&env) {
        emit(stdout, &value);
    }
}

/// Cap on one stdin line. Prompts can carry a pasted file, so the ceiling is
/// generous; it exists only to deny a peer unbounded memory.
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

enum LineRead {
    Line,
    TooLong,
    Eof,
}

/// Read one line into `buf` without buffering more than `MAX_LINE_BYTES` of
/// content. An oversized line is drained to its newline so the next line still
/// starts at a frame boundary. Reading one byte past the cap is what separates
/// a line that ends exactly at it from one that runs past it.
fn read_line_capped<R: BufRead>(reader: &mut R, buf: &mut Vec<u8>) -> std::io::Result<LineRead> {
    const LIMIT: u64 = MAX_LINE_BYTES as u64 + 1;
    buf.clear();
    let read = (&mut *reader).take(LIMIT).read_until(b'\n', buf)?;
    if read == 0 {
        return Ok(LineRead::Eof);
    }
    // Terminated, or a final line that ends at EOF without a newline. Either
    // way it stopped within the cap, so the content fits.
    if buf.last() == Some(&b'\n') || read <= MAX_LINE_BYTES {
        return Ok(LineRead::Line);
    }
    let mut sink = Vec::new();
    loop {
        sink.clear();
        let more = (&mut *reader).take(LIMIT).read_until(b'\n', &mut sink)?;
        if more == 0 || sink.last() == Some(&b'\n') {
            return Ok(LineRead::TooLong);
        }
    }
}

/// The `hello` handshake line, single-sourced so `run` and the tests cannot
/// drift. Carries both the human-readable `proto` id and the integer
/// `protocol_version` a client compares against.
fn hello_value(session_id: &str, project: &str, continued: bool) -> serde_json::Value {
    serde_json::json!({
        "type": "hello",
        "proto": PROTO,
        "protocol_version": PROTO_VERSION,
        "session_id": session_id,
        "version": env!("CARGO_PKG_VERSION"),
        "project": project,
        "continued": continued,
    })
}

/// Per-message and total content budgets for the `transcript` line. History
/// exists to orient an attaching frontend, not to re-carry every byte; the
/// session file remains the full record.
const TRANSCRIPT_MSG_CHARS: usize = 4_096;
const TRANSCRIPT_TOTAL_CHARS: usize = 262_144;

/// One bounded history line for a continued session: user and assistant text
/// only (tool traffic and the system prompt are session internals).
fn transcript_value(
    session_id: &str,
    messages: &[open_max_core::types::ChatMessage],
) -> serde_json::Value {
    let mut out = Vec::new();
    let mut total = 0usize;
    let mut truncated = false;
    for m in messages {
        if m.role != "user" && m.role != "assistant" {
            continue;
        }
        let Some(content) = m.content.as_deref().filter(|c| !c.trim().is_empty()) else {
            continue;
        };
        if total >= TRANSCRIPT_TOTAL_CHARS {
            truncated = true;
            break;
        }
        let mut text: String = content.chars().take(TRANSCRIPT_MSG_CHARS).collect();
        if text.len() < content.len() {
            text.push('…');
            truncated = true;
        }
        total += text.chars().count();
        out.push(serde_json::json!({ "role": m.role, "content": text }));
    }
    serde_json::json!({
        "type": "transcript",
        "session_id": session_id,
        "messages": out,
        "truncated": truncated,
    })
}

/// Validate one JSONL line against the `openmax-stdio/1` contract using the
/// authoritative types (`Command` for stdin, `AgentEvent` for stdout events),
/// so there is no second schema to drift. Returns a short label on success
/// (`cmd user`, `event token`, `hello`) or a human reason on failure.
pub fn validate_line(line: &str) -> Result<String, String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "not a JSON object".to_string())?;

    // A stdin command: parse with the real deserializer, unknown cmd fails.
    if obj.contains_key("cmd") {
        let cmd: Command =
            serde_json::from_value(value.clone()).map_err(|e| format!("bad command: {e}"))?;
        let name = match cmd {
            Command::User { .. } => "user",
            Command::Approve { .. } => "approve",
            Command::ApprovalMode { .. } => "approval_mode",
            Command::Reload => "reload",
            Command::Cancel => "cancel",
            Command::Quit => "quit",
        };
        return Ok(format!("cmd {name}"));
    }

    // Otherwise a stdout line, discriminated by `type`.
    let ty = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "object has neither 'cmd' nor 'type'".to_string())?;
    match ty {
        "hello" => {
            for field in ["proto", "session_id", "version", "project"] {
                if !obj.get(field).map(serde_json::Value::is_string).unwrap_or(false) {
                    return Err(format!("hello missing string '{field}'"));
                }
            }
            // Conformance is against the contract THIS binary implements, so a
            // foreign proto or version is a real mismatch, not just a
            // well-typed line. Otherwise the validator would bless a stream it
            // cannot actually speak.
            if obj.get("proto").and_then(serde_json::Value::as_str) != Some(PROTO) {
                return Err(format!("unsupported proto; expected '{PROTO}'"));
            }
            if obj.get("protocol_version").and_then(serde_json::Value::as_u64)
                != Some(u64::from(PROTO_VERSION))
            {
                return Err(format!("unsupported protocol_version; expected {PROTO_VERSION}"));
            }
            Ok("hello".to_string())
        }
        "protocol_error" => {
            if !obj.get("message").map(serde_json::Value::is_string).unwrap_or(false) {
                return Err("protocol_error missing string 'message'".to_string());
            }
            Ok("protocol_error".to_string())
        }
        "approval_mode" => {
            if !obj.get("mode").map(serde_json::Value::is_string).unwrap_or(false) {
                return Err("approval_mode missing string 'mode'".to_string());
            }
            Ok("approval_mode".to_string())
        }
        "transcript" => {
            if !obj.get("session_id").map(serde_json::Value::is_string).unwrap_or(false) {
                return Err("transcript missing string 'session_id'".to_string());
            }
            let Some(messages) = obj.get("messages").and_then(serde_json::Value::as_array) else {
                return Err("transcript missing array 'messages'".to_string());
            };
            for m in messages {
                let ok = m.get("role").map(serde_json::Value::is_string).unwrap_or(false)
                    && m.get("content").map(serde_json::Value::is_string).unwrap_or(false);
                if !ok {
                    return Err("transcript message missing string 'role'/'content'".to_string());
                }
            }
            Ok("transcript".to_string())
        }
        // An event envelope carries a flattened session_id plus the event.
        _ => {
            if !obj.get("session_id").map(serde_json::Value::is_string).unwrap_or(false) {
                return Err(format!("event '{ty}' missing string 'session_id'"));
            }
            serde_json::from_value::<AgentEvent>(value.clone())
                .map_err(|e| format!("bad event '{ty}': {e}"))?;
            Ok(format!("event {ty}"))
        }
    }
}

/// `openmax --check --stdio`: read a JSONL protocol stream on stdin, validate
/// every line against the contract, print a per-line report (mirroring the
/// filesystem `--check`), and return exit 1 if any line is invalid. A frontend
/// or interop-adapter author pipes their command stream (or a captured openmax
/// stdout stream) through this to prove conformance.
pub fn run_conformance() -> i32 {
    let stdin = std::io::stdin();
    let mut seen = 0usize;
    let mut errors = 0usize;
    for line in stdin.lock().lines() {
        // A read failure is a validation failure, not a clean EOF: exiting
        // zero here would report an unread tail of the stream as conforming.
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                println!("err  failed to read stdin: {e}");
                return 1;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        seen += 1;
        match validate_line(&line) {
            Ok(label) => println!("ok   {label}"),
            Err(reason) => {
                errors += 1;
                println!("err  {reason}");
            }
        }
    }
    if seen == 0 {
        println!("no protocol lines on stdin");
        return 0;
    }
    if errors > 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_parse_and_reject_unknowns() {
        assert!(matches!(
            serde_json::from_str::<Command>(r#"{"cmd":"user","text":"hi"}"#).unwrap(),
            Command::User { text } if text == "hi"
        ));
        assert!(matches!(
            serde_json::from_str::<Command>(
                r#"{"cmd":"approve","approval_id":"a1","approved":false}"#
            )
            .unwrap(),
            Command::Approve { approved: false, .. }
        ));
        assert!(matches!(
            serde_json::from_str::<Command>(r#"{"cmd":"cancel"}"#).unwrap(),
            Command::Cancel
        ));
        assert!(matches!(
            serde_json::from_str::<Command>(r#"{"cmd":"quit"}"#).unwrap(),
            Command::Quit
        ));
        // Unknown commands are protocol errors; stray fields are tolerated.
        assert!(serde_json::from_str::<Command>(r#"{"cmd":"reboot"}"#).is_err());
        assert!(serde_json::from_str::<Command>(r#"{"cmd":"cancel","note":"annotated"}"#).is_ok());
    }

    #[test]
    fn hello_line_carries_protocol_version() {
        let hello = hello_value("sess-1", "/tmp/proj", false);
        assert_eq!(hello["continued"], false);
        assert_eq!(hello["type"], "hello");
        assert_eq!(hello["proto"], PROTO);
        assert_eq!(hello["protocol_version"], PROTO_VERSION);
        assert_eq!(hello["session_id"], "sess-1");
        assert_eq!(hello["project"], "/tmp/proj");
        assert!(hello["version"].is_string());
        // The validator accepts the line the handshake actually emits.
        assert_eq!(
            validate_line(&serde_json::to_string(&hello).unwrap()).unwrap(),
            "hello"
        );
    }

    /// One truth: the human-readable `proto` id and the integer version can
    /// never disagree, so a client may key on either.
    #[test]
    fn proto_string_and_version_agree() {
        assert_eq!(PROTO, format!("openmax-stdio/{PROTO_VERSION}"));
    }

    #[test]
    fn validate_line_classifies_the_contract() {
        // stdin commands.
        assert_eq!(validate_line(r#"{"cmd":"user","text":"hi"}"#).unwrap(), "cmd user");
        assert_eq!(validate_line(r#"{"cmd":"cancel"}"#).unwrap(), "cmd cancel");
        assert_eq!(
            validate_line(r#"{"cmd":"approve","approval_id":"a","approved":true}"#).unwrap(),
            "cmd approve"
        );
        // stdout events (flattened session_id + tag).
        assert_eq!(
            validate_line(r#"{"session_id":"s1","type":"token","text":"hi"}"#).unwrap(),
            "event token"
        );
        assert_eq!(
            validate_line(
                r#"{"session_id":"s1","type":"done","stop_reason":"stop"}"#
            )
            .unwrap(),
            "event done"
        );
        assert_eq!(
            validate_line(r#"{"type":"protocol_error","message":"nope"}"#).unwrap(),
            "protocol_error"
        );

        // A foreign proto or version fails: the validator only blesses the
        // contract this binary implements.
        assert!(validate_line(
            r#"{"type":"hello","proto":"other/9","protocol_version":1,"session_id":"s","version":"0","project":"/p"}"#
        )
        .is_err());
        assert!(validate_line(
            r#"{"type":"hello","proto":"openmax-stdio/1","protocol_version":99,"session_id":"s","version":"0","project":"/p"}"#
        )
        .is_err());

        // Failures: unknown cmd, missing event field, missing session_id, junk.
        assert!(validate_line(r#"{"cmd":"reboot"}"#).is_err());
        assert!(validate_line(r#"{"session_id":"s1","type":"token"}"#).is_err());
        assert!(validate_line(r#"{"type":"token","text":"hi"}"#).is_err());
        assert!(validate_line(r#"{"type":"not_a_real_event","session_id":"s1"}"#).is_err());
        assert!(validate_line(r#"{"neither":1}"#).is_err());
        assert!(validate_line("not json").is_err());
    }

    /// Mirror of the reader thread's per-line handling: each line comes back
    /// as the text a command would parse from, or the reason it was refused.
    fn read_all(input: &[u8]) -> Vec<Result<String, String>> {
        let mut reader = std::io::BufReader::new(input);
        let mut buf = Vec::new();
        let mut out = Vec::new();
        loop {
            match read_line_capped(&mut reader, &mut buf) {
                Ok(LineRead::Eof) => return out,
                Ok(LineRead::TooLong) => out.push(Err("line too long".to_string())),
                Ok(LineRead::Line) => out.push(
                    std::str::from_utf8(&buf)
                        .map(|line| line.trim().to_string())
                        .map_err(|e| e.to_string()),
                ),
                Err(e) => {
                    out.push(Err(e.to_string()));
                    return out;
                }
            }
        }
    }

    #[test]
    fn an_oversized_line_is_refused_without_losing_the_next_command() {
        let mut input = Vec::new();
        input.extend_from_slice(br#"{"cmd":"cancel"}"#);
        input.push(b'\n');
        input.extend_from_slice(vec![b'x'; MAX_LINE_BYTES + 64].as_slice());
        input.push(b'\n');
        input.extend_from_slice(br#"{"cmd":"quit"}"#);
        input.push(b'\n');

        let mut reader = std::io::BufReader::new(input.as_slice());
        let mut buf = Vec::new();
        let mut seen = Vec::new();
        loop {
            match read_line_capped(&mut reader, &mut buf).unwrap() {
                LineRead::Eof => break,
                LineRead::TooLong => seen.push("too-long".to_string()),
                LineRead::Line => seen.push(String::from_utf8_lossy(&buf).trim().to_string()),
            }
            // Content stays within the cap; the extra byte is the newline, or
            // the single byte read past it to detect an overlong line.
            assert!(buf.len() <= MAX_LINE_BYTES + 1, "buffer must stay capped");
        }
        assert_eq!(
            seen,
            vec![
                r#"{"cmd":"cancel"}"#.to_string(),
                "too-long".to_string(),
                r#"{"cmd":"quit"}"#.to_string(),
            ],
            "an oversized line must not desynchronize framing"
        );
    }

    #[test]
    fn a_line_exactly_at_the_cap_is_accepted() {
        // The cap is a limit on content, so a line of exactly that many bytes
        // is valid and only the one after it is too long.
        for (content_len, expect_line) in [
            (MAX_LINE_BYTES - 1, true),
            (MAX_LINE_BYTES, true),
            (MAX_LINE_BYTES + 1, false),
        ] {
            let mut input = vec![b'x'; content_len];
            input.push(b'\n');
            let mut reader = std::io::BufReader::new(input.as_slice());
            let mut buf = Vec::new();
            let got = read_line_capped(&mut reader, &mut buf).unwrap();
            assert_eq!(
                matches!(got, LineRead::Line),
                expect_line,
                "a {content_len} byte line was classified wrong"
            );
            if expect_line {
                assert_eq!(buf.len(), content_len + 1, "the whole line must be kept");
            }
        }
    }

    #[test]
    fn a_line_without_a_trailing_newline_still_arrives() {
        let mut reader = std::io::BufReader::new(&br#"{"cmd":"quit"}"#[..]);
        let mut buf = Vec::new();
        assert!(matches!(read_line_capped(&mut reader, &mut buf).unwrap(), LineRead::Line));
        assert_eq!(buf, br#"{"cmd":"quit"}"#);
        assert!(matches!(read_line_capped(&mut reader, &mut buf).unwrap(), LineRead::Eof));
    }

    #[test]
    fn invalid_utf8_is_one_bad_line_not_a_dead_session() {
        let input: Vec<u8> = b"{\"cmd\":\"cancel\"}\n\xff\xfe\n{\"cmd\":\"quit\"}\n".to_vec();
        let results = read_all(&input);
        assert_eq!(results.len(), 3, "reading must continue past the bad line");
        assert!(results[0].is_ok());
        assert!(results[1].is_err(), "the invalid line must be reported");
        assert!(results[2].is_ok(), "the command after it must still be read");
    }

    #[test]
    fn transcript_line_is_bounded_and_validates() {
        use open_max_core::types::ChatMessage;
        let mk = |role: &str, content: &str| ChatMessage {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        };
        let messages = vec![
            mk("system", "hidden"),
            mk("user", "hello"),
            mk("assistant", &"x".repeat(TRANSCRIPT_MSG_CHARS + 10)),
            mk("tool", "internals"),
        ];
        let t = transcript_value("s1", &messages);
        assert_eq!(t["type"], "transcript");
        let out = t["messages"].as_array().unwrap();
        assert_eq!(out.len(), 2, "system and tool messages stay internal");
        assert_eq!(out[0]["role"], "user");
        let long = out[1]["content"].as_str().unwrap();
        assert!(long.chars().count() <= TRANSCRIPT_MSG_CHARS + 1);
        assert_eq!(t["truncated"], true);
        assert_eq!(validate_line(&serde_json::to_string(&t).unwrap()).unwrap(), "transcript");
    }

    /// Drive the protocol loop in-process: commands in, emitted lines out.
    async fn drive_commands(commands: Vec<Command>) -> (Vec<serde_json::Value>, i32, Arc<Core>) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir()
            .join(format!("openmax-stdio-{}-{nonce}", std::process::id()));
        let (core, core_rx) = Core::new(dir.clone()).unwrap();
        let meta = sessions::create(&core, dir.display().to_string()).unwrap();
        let (tx, rx) = mpsc::channel(64);
        for cmd in commands {
            tx.send(Ok(cmd)).await.unwrap();
        }
        drop(tx);
        let mut out = Vec::new();
        let code = drive(core.clone(), core_rx, meta.id, dir.clone(), rx, &mut out).await;
        let lines = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let _ = std::fs::remove_dir_all(dir);
        (lines, code, core)
    }

    #[tokio::test]
    async fn empty_user_text_is_refused_with_a_terminator() {
        let (lines, code, _core) =
            drive_commands(vec![Command::User { text: "  ".into() }]).await;
        assert_eq!(lines[0]["type"], "protocol_error");
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["stop_reason"], "refused");
        assert_eq!(code, 0);
    }

    #[tokio::test]
    async fn a_user_turn_in_an_untrusted_project_is_refused_not_hung() {
        let (lines, _code, _core) =
            drive_commands(vec![Command::User { text: "hi".into() }]).await;
        // The temp project was never trusted: the refusal must still end in
        // the one terminator the client is allowed to block on.
        assert_eq!(lines[0]["type"], "protocol_error");
        assert_eq!(lines[1]["type"], "done");
        assert_eq!(lines[1]["stop_reason"], "refused");
    }

    #[tokio::test]
    async fn an_unissued_approval_id_is_a_protocol_error() {
        let (lines, _code, _core) = drive_commands(vec![Command::Approve {
            approval_id: "ghost".into(),
            approved: true,
        }])
        .await;
        assert_eq!(lines[0]["type"], "protocol_error");
        assert!(lines[0]["message"].as_str().unwrap().contains("ghost"));
    }

    #[tokio::test]
    async fn approval_mode_command_persists_and_acknowledges() {
        let (lines, _code, core) = drive_commands(vec![
            Command::ApprovalMode { mode: "auto".into() },
            Command::ApprovalMode { mode: "sometimes".into() },
        ])
        .await;
        assert_eq!(lines[0]["type"], "approval_mode");
        assert_eq!(lines[0]["mode"], "auto");
        assert_eq!(lines[1]["type"], "protocol_error");
        assert_eq!(
            core.settings.lock().unwrap().approval_mode,
            open_max_core::config::ApprovalMode::Auto
        );
    }

    #[test]
    fn a_refused_user_command_is_terminated_by_done() {
        let env = AgentEventEnvelope {
            session_id: "s1".into(),
            event: AgentEvent::Done { stop_reason: "refused".into() },
        };
        let line = serde_json::to_string(&env).unwrap();
        // The terminator a refusal writes must be a real event on the wire, so
        // a client that only knows `done` needs no new parsing to be unblocked.
        assert_eq!(validate_line(&line).unwrap(), "event done");
        assert!(line.contains(r#""session_id":"s1""#), "{line}");
    }
}
