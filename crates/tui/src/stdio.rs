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

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use open_max_core::agent;
use open_max_core::sessions;
use open_max_core::state::{Core, CoreEvent};
use open_max_core::types::AgentEvent;
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
    Cancel,
    Quit,
}

pub struct StdioArgs {
    pub continue_session: bool,
}

pub async fn run(
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<CoreEvent>,
    args: StdioArgs,
) -> i32 {
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_key = project.display().to_string();

    let session_id = if args.continue_session {
        match sessions::latest(&core, &project_key) {
            Some(meta) => meta.id,
            None => {
                eprintln!("openmax: no prior session in this directory to continue");
                return 2;
            }
        }
    } else {
        match sessions::create(&core, project_key.clone()) {
            Ok(meta) => meta.id,
            Err(e) => {
                eprintln!("openmax: failed to create session: {e}");
                return 1;
            }
        }
    };

    let mut stdout = std::io::stdout();
    emit(&mut stdout, &hello_value(&session_id, &project_key));

    // Blocking stdin reader on its own thread; parse errors travel as Err so
    // the async loop can answer without ever blocking on the pipe.
    let (stdin_tx, mut stdin_rx) = mpsc::unbounded_channel::<Result<Command, String>>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let parsed = serde_json::from_str::<Command>(&line)
                .map_err(|e| format!("bad command line: {e}"));
            if stdin_tx.send(parsed).is_err() {
                break;
            }
        }
        // Dropping the sender closes the channel: EOF.
    });

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
                    Some(Err(message)) => protocol_error(&mut stdout, &message),
                    Some(Ok(Command::Cancel)) => core.cancel(&session_id),
                    Some(Ok(Command::Approve { approval_id, approved })) => {
                        open_approvals.remove(&approval_id);
                        core.respond_approval(&approval_id, approved);
                    }
                    Some(Ok(Command::User { text })) => {
                        if text.trim().is_empty() {
                            protocol_error(&mut stdout, "user text is empty");
                            continue;
                        }
                        if running || !crate::headless::wait_until_idle(&core, &session_id).await {
                            protocol_error(&mut stdout, "a turn is in flight; wait for done");
                            continue;
                        }
                        match agent::start_turn(core.clone(), session_id.clone(), project.clone(), text) {
                            Ok(()) => running = true,
                            Err(e) => protocol_error(&mut stdout, &e),
                        }
                    }
                }
            }
            event = core_rx.recv() => {
                let Some(event) = event else {
                    eprintln!("openmax: event channel closed");
                    return 1;
                };
                let CoreEvent::Agent(env) = event else { continue };
                if env.session_id != session_id {
                    continue;
                }
                if let Ok(value) = serde_json::to_value(&env) {
                    emit(&mut stdout, &value);
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

fn emit(stdout: &mut std::io::Stdout, value: &serde_json::Value) {
    if let Ok(line) = serde_json::to_string(value) {
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
    }
}

fn protocol_error(stdout: &mut std::io::Stdout, message: &str) {
    emit(
        stdout,
        &serde_json::json!({ "type": "protocol_error", "message": message }),
    );
}

/// The `hello` handshake line, single-sourced so `run` and the tests cannot
/// drift. Carries both the human-readable `proto` id and the integer
/// `protocol_version` a client compares against.
fn hello_value(session_id: &str, project: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "hello",
        "proto": PROTO,
        "protocol_version": PROTO_VERSION,
        "session_id": session_id,
        "version": env!("CARGO_PKG_VERSION"),
        "project": project,
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
        let hello = hello_value("sess-1", "/tmp/proj");
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
}
