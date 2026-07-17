//! Bidirectional stdio client: the machine-facing counterpart of the TUI.
//! JSONL commands arrive on stdin, `AgentEvent` envelopes leave on stdout:
//! the full custom-frontend and self-spawn contract. Any process that can
//! speak line-delimited JSON (an editor plugin, an orchestrator, another
//! openmax) can drive a complete interactive session, approvals included.
//!
//! Protocol (`openmax-stdio/1`), one JSON object per line:
//!
//! stdin commands:
//!   {"cmd":"user","text":"..."}                      start a turn
//!   {"cmd":"approve","approval_id":"...","approved":true}
//!   {"cmd":"cancel"}                                 cancel the running turn
//!   {"cmd":"quit"}                                   finish the turn, then exit
//!
//! stdout lines:
//!   {"type":"hello","proto":"openmax-stdio/1","session_id":"...","version":"...","project":"..."}
//!   AgentEvent envelopes exactly as `--print --json` emits them
//!   {"type":"protocol_error","message":"..."}        bad input; session unharmed
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
    emit(
        &mut stdout,
        &serde_json::json!({
            "type": "hello",
            "proto": PROTO,
            "session_id": session_id,
            "version": env!("CARGO_PKG_VERSION"),
            "project": project_key,
        }),
    );

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
}
