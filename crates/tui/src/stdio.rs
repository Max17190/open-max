//! Bidirectional stdio client: the machine-facing counterpart of the TUI.
//! JSONL commands arrive on stdin, `AgentEvent` envelopes leave on stdout:
//! the full custom-frontend and self-spawn contract. Any process that can
//! speak line-delimited JSON (an editor plugin, an orchestrator, another
//! openmax) can drive a complete interactive session, approvals included.
//!
//! Protocol (`openmax-stdio/8`), one JSON object per line. The normative
//! reference (every field of every line) is `docs/stdio-protocol.md`, kept in
//! step with `openmax --spec stdio`; `crates/core/src/types.rs` golden tests
//! pin the event wire.
//!
//! stdin commands:
//!   {"cmd":"user","text":"..."}                      start a turn, or steer the running one
//!   {"cmd":"approve","approval_id":"...","approved":true}
//!   {"cmd":"approval_mode","mode":"auto|ask|readonly"}
//!   {"cmd":"reload"}                                 re-freeze tools/skills/prompt
//!   {"cmd":"fork"}                                   branch the session here, switch to it
//!   {"cmd":"rewind","turn":1}                        branch at the end of turn 1, switch to it
//!   {"cmd":"cancel"}                                 cancel the running turn
//!   {"cmd":"quit"}                                   finish the turn, then exit
//!
//! stdout lines:
//!   {"type":"hello","proto":"openmax-stdio/8","protocol_version":8,"session_id":"...","version":"...","project":"...","continued":false}
//!   AgentEvent envelopes exactly as `--print --json` emits them
//!   {"type":"protocol_error","message":"..."}        bad input; session unharmed
//!   {"type":"forked","session_id":"...","parent_id":"...","message_index":3}
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
use open_max_core::templates;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Notify};
use std::time::Instant;

pub const PROTO: &str = "openmax-stdio/8";
/// Machine-comparable protocol major. A client negotiates on this integer;
/// `PROTO` embeds the same number as a human-readable id (checked in tests).
/// Bump on any wire change (event field, command shape, framing line).
pub const PROTO_VERSION: u32 = 8;

// Unknown `cmd` values are protocol errors; extra fields on a known command
// are ignored (lenient by design, so clients can annotate lines freely).
#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    User { text: String },
    Approve { approval_id: String, approved: bool },
    /// Save this trusted project's execution mode, like /approvals.
    ApprovalMode { mode: String },
    /// Re-freeze tools, skills, and prompt from current config, like /reload.
    Reload,
    /// Branch the session at its end and drive the branch, like /fork.
    Fork,
    /// Branch the session at the end of turn `turn` (1-based) and drive the
    /// branch, like /rewind. The session it leaves is not changed.
    Rewind { turn: usize },
    Cancel,
    Quit,
}

pub struct StdioArgs {
    pub continue_session: bool,
    /// Notified on a quit signal (see `watch_quit_signals`).
    pub quit: Arc<Notify>,
}

pub async fn run(
    core: Arc<Core>,
    core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    args: StdioArgs,
) -> i32 {
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_key = project.display().to_string();

    // Every step of the setup can wait on the session index lock.
    let setup = {
        let (core, project_key, continue_session) = (core.clone(), project_key.clone(), args.continue_session);
        move || open_session(&core, &project_key, continue_session)
    };
    let mut stop_by = None;
    let Some(opened) = crate::until_stopped(args.quit.notified(), &mut stop_by, setup).await else {
        return crate::quit_status();
    };
    let (session_id, continued, history) = match opened {
        Ok(opened) => opened,
        Err(code) => return code,
    };
    if stop_by.is_some() {
        // A signal landed while the session opened, so no turn ever ran.
        if !continued {
            crate::discard_empty_session(&core, &session_id, &args.quit, stop_by).await;
        }
        return crate::quit_status();
    }
    let mut stdout = std::io::stdout();
    emit_own(&mut stdout, &hello(&session_id, &project_key, continued));
    if continued {
        // One bounded history line so an attaching frontend can render what
        // came before, without replaying synthetic live events (a replayed
        // `token` stream would be indistinguishable from a running turn).
        emit_own(&mut stdout, &transcript(&session_id, &history));
    }
    let stdin_rx = spawn_stdin_reader();
    drive(core, core_rx, session_id, project, stdin_rx, &args.quit, &mut stdout, !continued).await
}

/// Create or find the session, claim it, and read its history to replay:
/// Ok(id, whether it was continued, history), or Err(the exit status) once
/// the reason is on stderr.
fn open_session(
    core: &Core,
    project_key: &str,
    continue_session: bool,
) -> Result<(String, bool, Vec<open_max_core::types::ChatMessage>), i32> {
    let (session_id, continued) = if continue_session {
        match sessions::latest(core, project_key) {
            Ok(Some(meta)) => (meta.id, true),
            Ok(None) => {
                eprintln!("openmax: no prior session in this directory to continue");
                return Err(2);
            }
            Err(e) => {
                let refusal = sessions::refusal_with_repair(core, e);
                eprintln!("openmax: {}", open_max_core::text::one_line(&refusal));
                return Err(1);
            }
        }
    } else {
        match sessions::create(core, project_key.to_string()) {
            Ok(meta) => (meta.id, false),
            Err(e) => {
                let refusal = sessions::refusal_with_repair(core, e);
                eprintln!("openmax: failed to create session: {}", open_max_core::text::one_line(&refusal));
                return Err(1);
            }
        }
    };

    if let Err(e) = sessions::attach(core, &session_id) {
        eprintln!("openmax: {e}");
        return Err(1);
    }
    let history = if continued {
        match sessions::load_messages(core, &session_id) {
            Ok(messages) => messages.unwrap_or_default(),
            Err(e) => { eprintln!("openmax: {e}"); return Err(1); }
        }
    } else { Vec::new() };
    Ok((session_id, continued, history))
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
/// to `out`. `fresh` says this process created the session: a client that
/// quits before any turn persisted anything (or whose only prompt a gate
/// refused) would otherwise leave an index entry with nothing behind it.
#[allow(clippy::too_many_arguments)]
async fn drive<W: Write>(
    core: Arc<Core>,
    core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session_id: String,
    project: PathBuf,
    stdin_rx: mpsc::Receiver<Result<Command, String>>,
    quit: &Notify,
    out: &mut W,
    fresh: bool,
) -> i32 {
    let mut stop_by = None;
    let code =
        drive_loop(core.clone(), core_rx, session_id.clone(), project, stdin_rx, quit, out, &mut stop_by)
            .await;
    if fresh {
        crate::discard_empty_session(&core, &session_id, quit, stop_by).await;
    }
    code
}

/// `stop_by` holds the stop's deadline once a signal lands, here or while
/// a turn starts; the empty-session cleanup after this loop is held to it too.
#[allow(clippy::too_many_arguments)]
async fn drive_loop<W: Write>(
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session_id: String,
    project: PathBuf,
    mut stdin_rx: mpsc::Receiver<Result<Command, String>>,
    quit: &Notify,
    out: &mut W,
    stop_by: &mut Option<Instant>,
) -> i32 {
    let mut session_id = session_id;
    let mut running = false;
    let mut closing = false;
    let mut exit_code = 0i32;
    // Approvals awaiting a client answer; declined in bulk when the client
    // quits so the drain never sits out the approval timeout.
    let mut open_approvals: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Set by a signal: the status the session exits with, and when it stops
    // waiting for the cancelled turn.
    let mut stopping: Option<i32> = None;
    let mut settle: Option<tokio::sync::oneshot::Receiver<()>> = None;
    let quit = quit.notified();
    tokio::pin!(quit);

    loop {
        // A signal, here or while a turn was starting, quits by cancelling
        // the turn instead of letting it finish: the signal reached this
        // process alone, and the done arrives once the turn's tools are
        // stopped.
        if let (None, Some(by)) = (stopping, *stop_by) {
            closing = true;
            for id in open_approvals.drain() {
                core.respond_approval(&id, false);
            }
            core.cancel(&session_id);
            stopping = Some(crate::quit_status());
            settle = Some(crate::deadline(by));
        }
        if closing && !running {
            return stopping.unwrap_or(exit_code);
        }
        tokio::select! {
            _ = &mut quit, if stop_by.is_none() => *stop_by = Some(Instant::now() + crate::STOP_SETTLE),
            _ = async { if let Some(settle) = settle.as_mut() { let _ = settle.await; } },
                if settle.is_some() => return stopping.unwrap_or(exit_code),
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
                                // Persist first, mutate the live gate only on
                                // success: the acknowledged state, the live
                                // state, and the on-disk state must never
                                // disagree about how mutating tools are gated.
                                // The save waits on the trust store's lock,
                                // which another openmax process can hold for
                                // as long as its own write takes, and on this
                                // thread that wait left a signal unanswered
                                // until the lock was free. A turn can be in
                                // flight, so a signal goes to the top of this
                                // loop, which cancels and settles that turn.
                                // A save still waiting gets no answer: it
                                // finishes on its own or ends with the process.
                                let save = tokio::task::spawn_blocking({
                                    let (core, project) = (core.clone(), project.clone());
                                    move || core.set_project_approval_mode(&project, parsed)
                                });
                                let saved = tokio::select! {
                                    // A result in hand is never dropped for a
                                    // signal that came with it.
                                    biased;
                                    saved = save => {
                                        saved.unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))
                                    }
                                    _ = &mut quit, if stop_by.is_none() => {
                                        *stop_by = Some(Instant::now() + crate::STOP_SETTLE);
                                        continue;
                                    }
                                };
                                match saved {
                                    Ok(()) => {
                                        emit_own(out, &StdioLine::ApprovalMode { mode: parsed.as_str() });
                                    }
                                    Err(e) => protocol_error(
                                        out,
                                        &format!("approval mode unchanged; persisting it failed: {e}"),
                                    ),
                                }
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
                        // A reload reads every extension manifest, and a read
                        // can stall (a slow disk, a manifest that is a pipe).
                        // It also claims the session, and that waits on the
                        // session index lock, which another openmax process
                        // can hold for as long as its own write takes, with
                        // the thread it runs on blocked. So it runs as a task
                        // of its own, and this loop stays free to answer a
                        // signal. No turn runs during a reload, so a signal
                        // ends the session at once instead of waiting it out.
                        let mut reload = tokio::spawn({
                            let (core, session_id, project) = (core.clone(), session_id.clone(), project.clone());
                            async move { agent::reload_session(&core, &session_id, &project).await }
                        });
                        let reloaded = tokio::select! {
                            // A result in hand is never dropped for a signal
                            // that came with it.
                            biased;
                            reloaded = &mut reload => {
                                reloaded.unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))
                            }
                            _ = &mut quit, if stop_by.is_none() => {
                                // Stopped at its next await, so a reload the
                                // signal interrupted is not applied while the
                                // session is going away. A blocking wait it is
                                // in finishes first, or ends with the process.
                                reload.abort();
                                *stop_by = Some(Instant::now() + crate::STOP_SETTLE);
                                return crate::quit_status();
                            }
                        };
                        match reloaded {
                            Ok((tools, skills, changes, _runs)) => {
                                let env = AgentEventEnvelope {
                                    session_id: session_id.clone(),
                                    event: AgentEvent::Refrozen { tools, skills, changes, runs: Vec::new() },
                                };
                                emit_event(out, &env);
                            }
                            Err(e) => protocol_error(out, &e),
                        }
                    }
                    Some(Ok(cmd @ (Command::Fork | Command::Rewind { .. }))) => {
                        if running {
                            protocol_error(out, "a turn is in flight; wait for done");
                            continue;
                        }
                        let at = match cmd {
                            Command::Rewind { turn } => sessions::ForkAt::Turn(turn),
                            _ => sessions::ForkAt::End,
                        };
                        // Reads the transcript and writes the index, either of
                        // which can wait on another process, so a signal bounds
                        // it like a turn start.
                        let branch = {
                            let (core, parent) = (core.clone(), session_id.clone());
                            move || fork_session(&core, &parent, at)
                        };
                        match crate::until_stopped(&mut quit, stop_by, branch).await {
                            Some(Ok(child)) => {
                                emit_own(out, &StdioLine::Forked {
                                    session_id: &child.id,
                                    parent_id: &session_id,
                                    message_index: child.forked_from.as_ref().map_or(0, |f| f.message_index),
                                });
                                session_id = child.id;
                            }
                            Some(Err(e)) => protocol_error(out, &e),
                            None => return crate::quit_status(),
                        }
                    }
                    Some(Ok(Command::User { text })) => {
                        if text.trim().is_empty() {
                            // A running turn owns the next `done`, so a
                            // `refused` one here would report it finished.
                            if running {
                                protocol_error(out, "user text is empty");
                            } else {
                                refuse(out, &session_id, "user text is empty");
                            }
                            continue;
                        }
                        if running {
                            // Steer the in-flight turn: it owns the next
                            // `done`, and `steer_accepted` marks where the
                            // text entered its transcript. Expanded like a
                            // turn's prompt (below, and for the same reason
                            // bounded by a signal), so a template steers too,
                            // and the expansion is the `text` that
                            // `steer_accepted` or `steer_blocked` carries.
                            // Queued before the next command is read, as every
                            // command is handled in the order sent: a steer
                            // read before `done` goes to the turn running when
                            // it was read, never to the turn a later `user`
                            // starts. The wait is a path probe and one
                            // regular-file read for a `/name` line, and none for
                            // any other text, held like the `approval_mode` save
                            // above.
                            let steer = {
                                let (core, session_id, project) = (core.clone(), session_id.clone(), project.clone());
                                move || core.steer(&session_id, templates::expand_user_input(&core.data_dir, &project, &text))
                            };
                            match crate::until_stopped(&mut quit, stop_by, steer).await {
                                Some(Ok(())) => {}
                                Some(Err(e)) => protocol_error(out, &e),
                                None => return crate::quit_status(),
                            }
                            continue;
                        }
                        // A `/name args` line expands here exactly as it does
                        // in the composer: a frontend driving this protocol
                        // gets the project's templates for free. The expansion
                        // reads the template's file, which stalled storage can
                        // hold for any length of time, so it runs with the
                        // turn's start, where a signal bounds the wait.
                        let start = {
                            let (core, session_id, project) = (core.clone(), session_id.clone(), project.clone());
                            move || {
                                let text = templates::expand_user_input(&core.data_dir, &project, &text);
                                agent::start_turn(core, session_id, project, text)
                            }
                        };
                        match crate::until_stopped(&mut quit, stop_by, start).await {
                            Some(Ok(())) => running = true,
                            Some(Err(e)) => refuse(out, &session_id, &e),
                            None => return crate::quit_status(),
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
                emit_event(out, &env);
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

/// The lines this frontend writes that are not `AgentEvent` envelopes. Like
/// an envelope, each is serialized straight to a string, so `type` comes
/// first and the fields follow in the order declared here, which is the
/// order docs/stdio-protocol.md shows them in. A `serde_json::json!` object
/// would sort its keys instead.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StdioLine<'a> {
    Hello {
        proto: &'static str,
        protocol_version: u32,
        session_id: &'a str,
        version: &'static str,
        project: &'a str,
        continued: bool,
    },
    Transcript { session_id: &'a str, messages: Vec<TranscriptMessage<'a>>, truncated: bool },
    ProtocolError { message: &'a str },
    ApprovalMode { mode: &'a str },
    /// The session now driven is a fork of `parent_id` holding its first
    /// `message_index` messages; every later line carries the new id.
    Forked { session_id: &'a str, parent_id: &'a str, message_index: u64 },
}

/// Branch `parent` at `at`, release the parent so any frontend can
/// continue it, and claim the branch. The switch happens whole or not at
/// all: on a refusal `parent` is still the session driven, and the refusal
/// names the branch, since /resume lists it and a retry would make a second
/// one. The release goes first because it is the step that can fail while
/// nothing is wrong (the session state is busy for a moment), and undoing a
/// claim would wait on that same state.
fn fork_session(core: &Core, parent: &str, at: sessions::ForkAt) -> Result<sessions::SessionMeta, String> {
    let child = sessions::fork(core, parent, at).map_err(|e| sessions::refusal_with_repair(core, e))?;
    let made = |e: String| format!("{e}; the branch {} was made and is listed in /resume", child.id);
    sessions::detach(core, parent).map_err(made)?;
    if let Err(e) = sessions::attach(core, &child.id) {
        return Err(made(match sessions::attach(core, parent) {
            Ok(()) => e,
            Err(back) => format!("{e}; reclaiming this session failed too: {back}"),
        }));
    }
    Ok(child)
}

#[derive(Serialize)]
struct TranscriptMessage<'a> {
    role: &'a str,
    content: String,
}

/// Write one line and flush it, so a client blocked on its next read gets it
/// at once.
fn emit<W: Write>(stdout: &mut W, line: &str) {
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

/// An event, framed as the line `--print --json` writes for it.
fn emit_event<W: Write>(stdout: &mut W, env: &AgentEventEnvelope) {
    if let Ok(line) = env.to_json_line() {
        emit(stdout, &line);
    }
}

/// One of this frontend's own lines, its keys in `StdioLine`'s order.
fn emit_own<W: Write>(stdout: &mut W, line: &StdioLine) {
    if let Ok(text) = serde_json::to_string(line) {
        emit(stdout, &text);
    }
}

fn protocol_error<W: Write>(stdout: &mut W, message: &str) {
    emit_own(stdout, &StdioLine::ProtocolError { message });
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
    emit_event(stdout, &env);
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
fn hello<'a>(session_id: &'a str, project: &'a str, continued: bool) -> StdioLine<'a> {
    StdioLine::Hello {
        proto: PROTO,
        protocol_version: PROTO_VERSION,
        session_id,
        version: env!("CARGO_PKG_VERSION"),
        project,
        continued,
    }
}

/// Per-message and total content budgets for the `transcript` line. History
/// exists to orient an attaching frontend, not to re-carry every byte; the
/// session file remains the full record.
const TRANSCRIPT_MSG_CHARS: usize = 4_096;
const TRANSCRIPT_TOTAL_CHARS: usize = 262_144;

/// One bounded history line for a continued session: user and assistant text
/// only (tool traffic and the system prompt are session internals).
fn transcript<'a>(
    session_id: &'a str,
    messages: &'a [open_max_core::types::ChatMessage],
) -> StdioLine<'a> {
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
        out.push(TranscriptMessage { role: &m.role, content: text });
    }
    StdioLine::Transcript { session_id, messages: out, truncated }
}

/// Validate one JSONL line against the `openmax-stdio/8` contract using the
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
            Command::Fork => "fork",
            Command::Rewind { .. } => "rewind",
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
        "forked" => {
            for field in ["session_id", "parent_id"] {
                if !obj.get(field).map(serde_json::Value::is_string).unwrap_or(false) {
                    return Err(format!("forked missing string '{field}'"));
                }
            }
            if !obj.get("message_index").map(serde_json::Value::is_u64).unwrap_or(false) {
                return Err("forked missing integer 'message_index'".to_string());
            }
            Ok("forked".to_string())
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
            let event = serde_json::from_value::<AgentEvent>(value.clone())
                .map_err(|e| format!("bad event '{ty}': {e}"))?;
            // Well-typed is not enough for the one event a human must act on:
            // an unapproved_source prompt without its file names no command
            // anyone can run, which is the failure this field exists to fix.
            if let AgentEvent::ApprovalRequest { reason, source_path, .. } = &event {
                if reason == "unapproved_source" && source_path.is_empty() {
                    return Err(
                        "approval_request with reason 'unapproved_source' has an empty 'source_path'"
                            .to_string(),
                    );
                }
            }
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
        let line = serde_json::to_string(&hello("sess-1", "/tmp/proj", false)).unwrap();
        // Byte for byte the handshake docs/stdio-protocol.md shows: `type`
        // first, then the fields in the documented order.
        assert_eq!(
            line,
            format!(
                r#"{{"type":"hello","proto":"{PROTO}","protocol_version":{PROTO_VERSION},"session_id":"sess-1","version":"{}","project":"/tmp/proj","continued":false}}"#,
                env!("CARGO_PKG_VERSION")
            )
        );
        // The validator accepts the line the handshake actually emits.
        assert_eq!(validate_line(&line).unwrap(), "hello");
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
            validate_line(r#"{"session_id":"s1","type":"steer_accepted","text":"use v2"}"#).unwrap(),
            "event steer_accepted"
        );
        assert_eq!(
            validate_line(r#"{"session_id":"s1","type":"steer_blocked","text":"use v2","reason":"no v2 here"}"#).unwrap(),
            "event steer_blocked"
        );
        // A blocked steer without its text is the line a frontend cannot
        // match to the steer it holds, so it is no `steer_blocked`.
        assert!(validate_line(r#"{"session_id":"s1","type":"steer_blocked","reason":"no v2 here"}"#).is_err());
        assert_eq!(
            validate_line(r#"{"type":"protocol_error","message":"nope"}"#).unwrap(),
            "protocol_error"
        );
        assert_eq!(
            validate_line(r#"{"type":"forked","session_id":"s2","parent_id":"s1","message_index":3}"#).unwrap(),
            "forked"
        );
        assert!(validate_line(r#"{"type":"forked","session_id":"s2","message_index":3}"#).is_err());
        assert!(validate_line(r#"{"type":"forked","session_id":"s2","parent_id":"s1","message_index":"3"}"#).is_err());

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

        // An unapproved_source prompt is only conformant when it names the
        // file to approve; the human boundary needs a runnable command.
        assert_eq!(
            validate_line(
                r#"{"session_id":"s1","type":"approval_request","approval_id":"a","name":"danger","summary":"danger","detail":"","reason":"unapproved_source","source_path":".openmax/tools/danger.toml","source_sha":"0123456789ab"}"#
            )
            .unwrap(),
            "event approval_request"
        );
        assert!(validate_line(
            r#"{"session_id":"s1","type":"approval_request","approval_id":"a","name":"danger","summary":"danger","detail":"","reason":"unapproved_source","source_path":"","source_sha":""}"#
        )
        .is_err());
        // The ordinary gate carries no file and stays valid without one.
        assert_eq!(
            validate_line(
                r#"{"session_id":"s1","type":"approval_request","approval_id":"a","name":"bash","summary":"ls","detail":"ls","reason":"gate","source_path":"","source_sha":""}"#
            )
            .unwrap(),
            "event approval_request"
        );
        // A pre-/3 line without the source fields is not this contract.
        assert!(validate_line(
            r#"{"session_id":"s1","type":"approval_request","approval_id":"a","name":"bash","summary":"ls","detail":"ls","reason":"gate"}"#
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
            reasoning_content: None,
            reasoning: None,
            reasoning_details: None,
            reasoning_origin: None,
        };
        let messages = vec![
            mk("system", "hidden"),
            mk("user", "hello"),
            mk("assistant", &"x".repeat(TRANSCRIPT_MSG_CHARS + 10)),
            mk("tool", "internals"),
        ];
        let line = serde_json::to_string(&transcript("s1", &messages)).unwrap();
        let t: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(t["type"], "transcript");
        let out = t["messages"].as_array().unwrap();
        assert_eq!(out.len(), 2, "system and tool messages stay internal");
        assert_eq!(out[0]["role"], "user");
        let long = out[1]["content"].as_str().unwrap();
        assert!(long.chars().count() <= TRANSCRIPT_MSG_CHARS + 1);
        assert_eq!(t["truncated"], true);
        assert_eq!(validate_line(&line).unwrap(), "transcript");
        // `type` first, then the fields and each message's keys in the order
        // docs/stdio-protocol.md shows.
        assert_eq!(
            serde_json::to_string(&transcript("s1", &messages[..2])).unwrap(),
            r#"{"type":"transcript","session_id":"s1","messages":[{"role":"user","content":"hello"}],"truncated":false}"#
        );
    }

    /// Drive the protocol loop in-process: commands in, emitted lines out.
    async fn drive_commands(commands: Vec<Command>) -> (Vec<serde_json::Value>, i32, Arc<Core>) {
        let (lines, code, core, dir) = drive_in_project(|_| {}, false, commands).await;
        let _ = std::fs::remove_dir_all(dir);
        (lines, code, core)
    }

    /// Same loop, with `prepare` run against the project root (which is also
    /// the data dir) before the session starts, so a test can plant trust or
    /// extension files, and `seeded` giving the session a persisted transcript
    /// first, as one that has run a turn would have. The caller owns the
    /// directory's cleanup.
    async fn drive_in_project(
        prepare: impl FnOnce(&std::path::Path),
        seeded: bool,
        commands: Vec<Command>,
    ) -> (Vec<serde_json::Value>, i32, Arc<Core>, PathBuf) {
        let dir = crate::test_temp_dir("openmax-stdio");
        let (core, core_rx) = Core::new(dir.clone()).unwrap();
        prepare(&dir);
        let meta = sessions::create(&core, dir.display().to_string()).unwrap();
        if seeded {
            let mut persisted = 0usize;
            let seed = [open_max_core::types::ChatMessage::system("sys")];
            assert!(sessions::save_messages(&core, &meta.id, &seed, &mut persisted, false));
        }
        let (tx, rx) = mpsc::channel(64);
        for cmd in commands {
            tx.send(Ok(cmd)).await.unwrap();
        }
        drop(tx);
        let mut out = Vec::new();
        let quit = tokio::sync::Notify::new();
        let code = drive(core.clone(), core_rx, meta.id, dir.clone(), rx, &quit, &mut out, true).await;
        let lines = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        (lines, code, core, dir)
    }

    /// A `/name args` line submitted over the protocol reaches the model as
    /// the template body, not as the literal slash line: the front end a
    /// delegate or custom UI drives gets the project's templates too. The
    /// turn dies at endpoint resolution (nothing is configured), but the
    /// title is written from the submitted text before that, so it is the
    /// proof of what was submitted. Seeded, so the session is history that
    /// survives the exit and keeps the title.
    #[tokio::test]
    async fn a_template_invocation_expands_before_the_turn_starts() {
        let (lines, _code, core, dir) = drive_in_project(
            |root| {
                open_max_core::trust::trust_project(root, root).unwrap();
                let prompts = root.join(".agents").join("prompts");
                std::fs::create_dir_all(&prompts).unwrap();
                std::fs::write(
                    prompts.join("omx-stdio-greet.md"),
                    "MARKER: greet $ARGUMENTS\n",
                )
                .unwrap();
            },
            true,
            vec![Command::User { text: "/omx-stdio-greet world".into() }],
        )
        .await;

        let title = sessions::latest(&core, &dir.display().to_string()).unwrap().unwrap().title;
        assert_eq!(title, "MARKER: greet world", "the raw slash line must not be submitted");
        // The turn really started: the only error is the unconfigured
        // endpoint, model and window, and it reached the wire.
        let errors: Vec<&str> = lines
            .iter()
            .filter(|l| l["type"] == "error")
            .filter_map(|l| l["message"].as_str())
            .collect();
        assert!(
            !errors.is_empty()
                && errors.iter().all(|m| m.contains("no endpoint, model or context window configured")),
            "{errors:?}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A fork leaves the original free for any frontend: another process
    /// (here a second Core on the same store, whose session locks conflict
    /// as another process's do) can claim it while this one holds the
    /// branch. When the original cannot be released nothing switches: it
    /// stays this process's, the branch is left free, and the refusal names
    /// the branch so a retry need not make another.
    #[tokio::test]
    async fn a_fork_releases_the_original_or_switches_nothing() {
        use open_max_core::types::ChatMessage;
        let dir = crate::test_temp_dir("openmax-stdio-fork");
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let parent = sessions::create(&core, dir.display().to_string()).unwrap().id;
        sessions::attach(&core, &parent).unwrap();
        assert!(sessions::save_messages(&core, &parent, &[ChatMessage::system("sys"), ChatMessage::user("hi")], &mut 0, true));
        let (other, _other_rx) = Core::new(dir.clone()).unwrap();

        let busy = core.sessions.try_lock().unwrap();
        let err = fork_session(&core, &parent, sessions::ForkAt::End).unwrap_err();
        drop(busy);
        let listed = sessions::list(&core, &dir.display().to_string()).unwrap();
        let stray = listed.into_iter().find(|m| m.forked_from.is_some()).unwrap().id;
        assert!(err.contains(&format!("the branch {stray} was made and is listed in /resume")), "{err}");
        assert!(sessions::attach(&other, &parent).is_err(), "the original is still this process's");
        sessions::attach(&other, &stray).expect("the branch is left free");
        sessions::detach(&other, &stray).unwrap();

        let child = fork_session(&core, &parent, sessions::ForkAt::End).unwrap();
        assert_eq!(child.forked_from.unwrap().message_index, 2);
        sessions::attach(&other, &parent).expect("the original is released");
        assert!(sessions::attach(&other, &child.id).is_err(), "the branch is this process's now");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The session exists from the handshake on (its id is in `hello`), so a
    /// client that quits first has left an index entry with no file behind
    /// it: `--recall` would report it as unreadable history on every run and
    /// `--continue` would attach to it. It is discarded at exit; a session
    /// with a transcript is history and stays.
    #[tokio::test]
    async fn a_client_that_quits_before_any_turn_leaves_no_session_behind() {
        let (_lines, code, core, dir) = drive_in_project(|_| {}, false, vec![Command::Quit]).await;
        assert_eq!(code, 0);
        let key = dir.display().to_string();
        assert!(sessions::list(&core, &key).unwrap().is_empty(), "nothing to resume was left indexed");
        let _ = std::fs::remove_dir_all(dir);

        let (_lines, _code, core, dir) = drive_in_project(|_| {}, true, vec![Command::Quit]).await;
        let key = dir.display().to_string();
        assert_eq!(sessions::list(&core, &key).unwrap().len(), 1, "a transcript is history");
        let _ = std::fs::remove_dir_all(dir);
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
        let (lines, _code, core, dir) = drive_in_project(
            |root| { open_max_core::trust::trust_project(root, root).unwrap(); },
            false,
            vec![
                Command::ApprovalMode { mode: "auto".into() },
                Command::ApprovalMode { mode: "sometimes".into() },
            ],
        ).await;
        assert_eq!(lines[0]["type"], "approval_mode");
        assert_eq!(lines[0]["mode"], "auto");
        assert_eq!(lines[1]["type"], "protocol_error");
        assert_eq!(core.approval_mode(&dir), open_max_core::config::ApprovalMode::Auto);
        assert_eq!(Core::new(dir.clone()).unwrap().0.approval_mode(&dir), open_max_core::config::ApprovalMode::Auto);
        assert_eq!(core.settings.lock().unwrap().approval_mode, open_max_core::config::ApprovalMode::Ask);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every stdin command the reader accepts, by wire name. The match is
    /// exhaustive, so adding a Command variant without listing it here stops
    /// compiling, and every surface checked against this list sees it.
    fn accepted_commands() -> Vec<&'static str> {
        let wire = [
            r#"{"cmd":"user","text":"hi"}"#,
            r#"{"cmd":"approve","approval_id":"a","approved":true}"#,
            r#"{"cmd":"approval_mode","mode":"auto"}"#,
            r#"{"cmd":"reload"}"#,
            r#"{"cmd":"fork"}"#,
            r#"{"cmd":"rewind","turn":1}"#,
            r#"{"cmd":"cancel"}"#,
            r#"{"cmd":"quit"}"#,
        ];
        wire.iter()
            .map(|line| match serde_json::from_str::<Command>(line).expect(line) {
                Command::User { .. } => "user",
                Command::Approve { .. } => "approve",
                Command::ApprovalMode { .. } => "approval_mode",
                Command::Reload => "reload",
                Command::Fork => "fork",
                Command::Rewind { .. } => "rewind",
                Command::Cancel => "cancel",
                Command::Quit => "quit",
            })
            .collect()
    }

    /// Every stdin command the reader accepts and every bespoke stdout line
    /// the session emits must appear in the printed contract: a frontend is
    /// written against `openmax --spec stdio` and judged by `validate_line`,
    /// so a name the spec omits is either a capability its author cannot
    /// discover or a line their client rejects on arrival.
    #[test]
    fn the_spec_names_every_command_and_bespoke_line() {
        let text = open_max_core::spec::render("stdio").unwrap();
        for name in accepted_commands() {
            assert!(
                text.contains(&format!("\"cmd\":\"{name}\"")),
                "--spec stdio never names command '{name}'"
            );
        }
        for ty in ["hello", "protocol_error", "approval_mode", "transcript", "forked"] {
            assert!(
                text.contains(&format!("\"type\":\"{ty}\"")),
                "--spec stdio never names bespoke line '{ty}'"
            );
        }
    }

    /// `--help` is where a frontend author first meets `--stdio`, so its
    /// command list must be the protocol's: a command it omits is a
    /// capability nobody finds there. The `{"cmd":...}` list is parsed and
    /// compared as a SET, so a reflow cannot hide a missing or stale name.
    #[test]
    fn help_names_every_command() {
        let help = crate::HELP;
        let start = help.find("--stdio ").expect("--stdio is documented");
        let open = help[start..].find(r#"{"cmd":"#).expect("--stdio lists its commands")
            + start
            + r#"{"cmd":"#.len();
        let close = help[open..].find('}').expect("the list closes") + open;
        let mut listed: Vec<&str> = help[open..close]
            .split('|')
            .map(|name| name.trim().trim_matches('"'))
            .collect();
        listed.sort_unstable();
        let mut accepted = accepted_commands();
        accepted.sort_unstable();
        assert_eq!(listed, accepted, "--help's --stdio list must equal the commands stdin accepts");
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
