//! Headless (print) runner: drive the core agent loop without a TUI.
//! Proves the harness is the product and the terminal UI is one client.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use open_max_core::agent;
use open_max_core::text::one_line;
use open_max_core::sessions;
use open_max_core::state::Core;
use open_max_core::templates;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use tokio::sync::{mpsc, Notify};
use std::time::Instant;

pub struct HeadlessArgs {
    /// One or more user prompts; each runs as a sequential turn on the same session.
    pub prompts: Vec<String>,
    pub continue_session: bool,
    pub json: bool,
    /// Notified on a quit signal (see `watch_quit_signals`).
    pub quit: Arc<Notify>,
}

/// Run one or more agent turns and exit when the last finishes. Approvals in
/// `ask` mode are declined so unattended runs never hang; set `approval_mode`
/// to `auto` for unattended mutations. Multiple prompts reuse one session_id.
pub async fn run(
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    args: HeadlessArgs,
) -> i32 {
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project_key = project.display().to_string();

    // Finding or creating the session waits on the session index lock.
    let setup = {
        let (core, continue_session) = (core.clone(), args.continue_session);
        move || open_session(&core, project_key, continue_session)
    };
    let mut stop_by = None;
    let Some(opened) = crate::until_stopped(args.quit.notified(), &mut stop_by, setup).await else {
        return crate::quit_status();
    };
    let session_id = match opened {
        Ok(session_id) => session_id,
        Err(code) => return code,
    };

    // A signal that landed while the session opened leaves no turn to run.
    let code = if stop_by.is_some() {
        crate::quit_status()
    } else {
        run_prompts(&core, &mut core_rx, &session_id, &args, &mut stop_by).await
    };
    // A run whose every turn ended before anything was persisted (an endpoint
    // that never resolved, a prompt a gate refused) leaves an index entry with
    // nothing behind it.
    if !args.continue_session {
        crate::discard_empty_session(&core, &session_id, &args.quit, stop_by).await;
    }
    code
}

/// The latest session here, or a new one: Ok(its id), or Err(the exit
/// status) once the reason is on stderr.
fn open_session(core: &Core, project_key: String, continue_session: bool) -> Result<String, i32> {
    if continue_session {
        match sessions::latest(core, &project_key) {
            Ok(Some(meta)) => Ok(meta.id),
            Ok(None) => {
                eprintln!("openmax: no prior session in this directory to continue");
                Err(2)
            }
            Err(e) => {
                eprintln!("openmax: {}", one_line(&sessions::refusal_with_repair(core, e)));
                Err(1)
            }
        }
    } else {
        match sessions::create(core, project_key) {
            Ok(meta) => Ok(meta.id),
            Err(e) => {
                let refusal = sessions::refusal_with_repair(core, e);
                eprintln!("openmax: failed to create session: {}", one_line(&refusal));
                Err(1)
            }
        }
    }
}

async fn run_prompts(
    core: &Arc<Core>,
    core_rx: &mut mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session_id: &str,
    args: &HeadlessArgs,
    stop_by: &mut Option<Instant>,
) -> i32 {
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut exit_code = 0i32;
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();

    for prompt in &args.prompts {
        // A signal that landed as the previous turn ended stops the run here,
        // before the next prompt is recorded and its hooks run.
        if crate::signalled() {
            break;
        }
        // Prompt templates belong to the harness, not to the terminal UI: a
        // delegated `openmax -p "/greet world"` submits what the composer
        // would, so hooks and the transcript see the expanded text. The
        // expansion reads the template's file, which stalled storage can hold
        // for any length of time, and starting the turn claims the session,
        // which waits on the session index lock, so both run where a signal
        // bounds them.
        let start = {
            let (core, session_id, project) = (core.clone(), session_id.to_string(), project.clone());
            let prompt = prompt.clone();
            move || {
                let text = templates::expand_user_input(&core.data_dir, &project, &prompt);
                agent::start_turn(core, session_id, project, text)
            }
        };
        // A turn started under a signal is stopped by `run_turn_events`.
        match crate::until_stopped(args.quit.notified(), stop_by, start).await {
            Some(Ok(())) => {}
            Some(Err(e)) => {
                eprintln!("openmax: {e}");
                return 1;
            }
            None => return crate::quit_status(),
        }

        let mut saw_tokens = false;
        let turn_exit = run_turn_events(
            core,
            core_rx,
            session_id,
            args,
            &mut saw_tokens,
            &mut stdout,
            &mut stderr,
            stop_by,
        )
        .await;
        if turn_exit != 0 {
            exit_code = turn_exit;
            // Stop the multi-turn chain on hard failure so later prompts do not
            // run against a broken or cancelled session mid-error.
            break;
        }
    }

    exit_code
}

/// `stop_by` holds the stop's deadline once a signal lands, here or while
/// a turn starts; the empty-session cleanup after the run is held to it too.
#[allow(clippy::too_many_arguments)]
async fn run_turn_events(
    core: &Arc<Core>,
    core_rx: &mut mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session_id: &str,
    args: &HeadlessArgs,
    saw_tokens: &mut bool,
    stdout: &mut io::Stdout,
    stderr: &mut io::Stderr,
    stop_by: &mut Option<Instant>,
) -> i32 {
    let json = args.json;
    let mut exit_code = 0i32;
    // Whether this turn's error line is already on stderr.
    let mut error_printed = false;
    // Once a signal has stopped the run: the status it exits with, and when
    // it stops waiting for the turn. The stop cancels the turn and waits for
    // its done, which arrives once the tools are stopped; returning at once
    // would exit with them running.
    let mut stopping: Option<i32> = None;
    let mut settle: Option<tokio::sync::oneshot::Receiver<()>> = None;
    let quit = args.quit.notified();
    tokio::pin!(quit);

    loop {
        // A signal, here or while the turn was starting, cancels the turn.
        if let (None, Some(by)) = (stopping, *stop_by) {
            core.cancel(session_id);
            stopping = Some(crate::quit_status());
            settle = Some(crate::deadline(by));
        }
        // No clock of its own: a live turn can go quiet here for longer than
        // any fixed bound (a long tool call's arguments stream without an
        // event). The client's idle timeout already ends a dead endpoint's
        // turn with Done, and the tool timeouts bound a hung tool. The one
        // deadline is a stop's: how long its cancelled turn gets to settle.
        let env = tokio::select! {
            event = core_rx.recv() => match event {
                Some(event) => event,
                None => {
                    let _ = writeln!(stderr, "openmax: event channel closed");
                    return 1;
                }
            },
            _ = &mut quit, if stop_by.is_none() => {
                *stop_by = Some(Instant::now() + crate::STOP_SETTLE);
                continue;
            }
            _ = async { if let Some(settle) = settle.as_mut() { let _ = settle.await; } },
                if settle.is_some() => return stopping.unwrap_or(1),
        };

        if env.session_id != session_id {
            continue;
        }

        if json {
            // The line `--stdio` frames for the same event, byte for byte.
            if let Ok(line) = env.to_json_line() {
                let _ = writeln!(stdout, "{line}");
                let _ = stdout.flush();
            }
        }

        match &env.event {
            AgentEvent::Token { text } => {
                if !json {
                    *saw_tokens = true;
                    let _ = write!(stdout, "{text}");
                    let _ = stdout.flush();
                }
            }
            AgentEvent::MessageDone { text } => {
                if !json && !text.is_empty() {
                    // Some backends only deliver the final message (no stream).
                    if !*saw_tokens {
                        let _ = write!(stdout, "{text}");
                    }
                    if !text.ends_with('\n') {
                        let _ = writeln!(stdout);
                    }
                    let _ = stdout.flush();
                    *saw_tokens = false;
                }
            }
            AgentEvent::ToolStart { name, args: tool_args, .. } => {
                if !json {
                    let summary = open_max_core::registry::summarize_call(name, tool_args);
                    let _ = writeln!(stderr, "→ {}", call_line(name, &summary));
                    let _ = stderr.flush();
                }
            }
            AgentEvent::ToolEnd { ok, output, .. } => {
                if !json {
                    let status = if *ok { "ok" } else { "err" };
                    let preview = truncate_line(output, 120);
                    let _ = writeln!(stderr, "← {status}: {preview}");
                    let _ = stderr.flush();
                }
            }
            AgentEvent::ApprovalRequest {
                approval_id,
                name,
                summary,
                detail: _,
                reason,
                source_path,
                source_sha,
                // The credential grant is for an interactive card; headless
                // declines the one prompt it rides (unapproved_source) and
                // names the --approve repair, so there is no card to disclose
                // it on here.
                env: _,
            } => {
                // The core never asks in auto. An unattended frontend must
                // decline every request that does reach it.
                let hint = if reason == "unapproved_source" {
                    approve_hint(source_path, source_sha)
                } else {
                    "select /approvals auto in this project for unattended work".to_string()
                };
                let _ = writeln!(stderr, "openmax: declining {}; {hint}", call_line(name, summary));
                core.respond_approval(approval_id, false);
            }
            AgentEvent::Error { message } => {
                if !json {
                    let _ = writeln!(stderr, "openmax: error: {message}");
                    error_printed = true;
                }
                exit_code = 1;
            }
            AgentEvent::Done { stop_reason } => {
                if !json {
                    let _ = writeln!(stdout);
                    // A failed turn already printed its error line; a
                    // "stopped (error)" after it would only repeat it.
                    let reported = stop_reason == "error" && error_printed;
                    if stop_reason != "stop" && stop_reason != "tool_calls" && !reported {
                        let _ = writeln!(stderr, "openmax: stopped ({stop_reason})");
                    }
                }
                if stop_reason == "error" {
                    exit_code = 1;
                } else if matches!(
                    stop_reason.as_str(),
                    "max_iterations" | "budget_exhausted" | "unverified"
                ) {
                    // The turn ran out of room rather than finishing, so a
                    // script that reads exit 0 as "the work is done" would be
                    // wrong. Its own code: 1 is an operational failure and
                    // nothing failed, 3 is a human boundary and no human was
                    // asked. Resubmitting continues the work, except after an
                    // `unverified` from a completion gate that could not
                    // start, which repeats until the hook is repaired.
                    exit_code = 4;
                }
                return stopping.unwrap_or(exit_code);
            }
            AgentEvent::Refrozen { tools, skills, changes, .. } => {
                if !json {
                    // `changes` names the capability files that changed; a
                    // refreeze can name none (a manifest that turned
                    // unreadable), and the line must not end in a colon with
                    // nothing after it.
                    let what = match changes.is_empty() {
                        true => String::new(),
                        false => format!(": {}", changes.join(", ")),
                    };
                    let _ = writeln!(stderr, "openmax: re-frozen ({tools} tools, {skills} skills){what}");
                }
            }
            AgentEvent::HarnessNote { .. } => {
                // Addressed to the model, which reads it in its transcript;
                // `--json` carries it as a `harness_note` line. Not stderr:
                // a parent agent running `openmax -p` reads this run's stderr
                // in its tool result, so every note meant for this run's
                // model would also cost the parent's.
            }
            AgentEvent::Compacted { tokens_before, tokens_after, compacted_messages } => {
                if !json {
                    let _ = writeln!(
                        stderr,
                        "openmax: compacted ~{tokens_before} to ~{tokens_after} tokens ({compacted_messages} messages archived)"
                    );
                }
            }
            AgentEvent::Retry { attempt, max_attempts, reason } => {
                if !json {
                    let _ = writeln!(stderr, "openmax: {}; retrying ({attempt} of {max_attempts})", one_line(reason));
                }
            }
            AgentEvent::SchemasOverBudget { schema_tokens, budget_tokens } => {
                // Advisory: the turn still runs, so the exit code is untouched.
                if !json {
                    let _ = writeln!(
                        stderr,
                        "openmax: tool schemas cost ~{schema_tokens} tokens of the ~{budget_tokens} this context window can spend; history is compacted early and turns may not fit at all. Remove tools (openmax --spec usage) or raise context_tokens"
                    );
                }
            }
            AgentEvent::HookFailed { hook, event, detail } => {
                if !json {
                    let _ = writeln!(
                        stderr,
                        "openmax: hook '{hook}' failed on {event}: {}",
                        one_line(detail)
                    );
                }
            }
            AgentEvent::TurnRefused { hook, reason, continuation, continuations_left } => {
                if !json {
                    let _ = writeln!(
                        stderr,
                        "openmax: turn_end gate '{hook}' refused completion (refusal {} of {}): {}",
                        continuation + 1,
                        continuation + continuations_left,
                        one_line(reason),
                    );
                }
            }
            AgentEvent::Thinking { .. }
            | AgentEvent::Budget { .. }
            | AgentEvent::Usage { .. }
            | AgentEvent::Diff { .. }
            | AgentEvent::ApprovalSettled { .. } => {}
        }
    }
}

/// `name summary`, collapsed to just the name when the summary is the name
/// again (an external call whose arguments held no string to describe).
fn call_line(name: &str, summary: &str) -> String {
    if summary.is_empty() || summary == name {
        name.to_string()
    } else {
        format!("{name} {summary}")
    }
}

fn truncate_line(s: &str, max: usize) -> String {
    // The same rule as every other authored line in the workspace: this
    // preview is raw tool output on a terminal, so an ESC or a U+2028 must not
    // survive it either (the old inline map flattened only \n and \r). Complete
    // escape sequences go first, or flattening the ESC alone would print the
    // rest of a color code as text.
    let flattened = one_line(&crate::ui::text::strip_escapes(s));
    let trimmed = flattened.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// The line an operator copies when a headless run declines the content
/// gate. The path is the whole point of the line, and it is a file the agent
/// named, so it must survive a paste into a shell unchanged.
fn approve_hint(source_path: &str, source_sha: &str) -> String {
    format!(
        "a human must approve this tool's content first: openmax --approve {} ({source_sha})",
        open_max_core::doctor::shell_quote(std::path::Path::new(source_path))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A turn that stopped short is not a turn that finished, and a script
    /// driving `openmax -p` has no other way to tell: the text on stdout looks
    /// like an answer either way. So the incomplete reasons get their own code
    /// rather than sharing 0 with success or 1 with an operational failure.
    #[tokio::test]
    async fn an_incomplete_turn_exits_four_in_print_mode() {
        let dir = std::env::temp_dir().join(format!(
            "openmax-headless-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (core, mut rx) = open_max_core::state::Core::new(dir.clone()).unwrap();
        let mut stdout = io::stdout();
        let mut stderr = io::stderr();
        let args = HeadlessArgs {
            prompts: Vec::new(),
            continue_session: false,
            json: true,
            quit: Arc::new(Notify::new()),
        };

        // Every stop reason a Done event can carry, and what a caller gets.
        // `truncated` and `blocked` reach 0 here on purpose: both emit an
        // Error event first, which is what sets their 1 in a real run.
        let cases = [
            ("stop", 0),
            ("tool_calls", 0),
            ("cancelled", 0),
            ("error", 1),
            ("max_iterations", 4),
            ("budget_exhausted", 4),
            ("unverified", 4),
        ];
        for (stop_reason, expected) in cases {
            core.send_agent("s", AgentEvent::Done { stop_reason: stop_reason.to_string() });
            let (mut saw_tokens, mut stop_by) = (false, None);
            let code = run_turn_events(
                &core, &mut rx, "s", &args, &mut saw_tokens, &mut stdout, &mut stderr, &mut stop_by,
            )
            .await;
            assert_eq!(code, expected, "stop reason {stop_reason}");
        }

        // An Error before the Done is what a truncated or blocked turn sends.
        for stop_reason in ["truncated", "blocked"] {
            core.send_agent("s", AgentEvent::Error { message: "boom".into() });
            core.send_agent("s", AgentEvent::Done { stop_reason: stop_reason.to_string() });
            let (mut saw_tokens, mut stop_by) = (false, None);
            let code = run_turn_events(
                &core, &mut rx, "s", &args, &mut saw_tokens, &mut stdout, &mut stderr, &mut stop_by,
            )
            .await;
            assert_eq!(code, 1, "stop reason {stop_reason}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A live turn can send no event for longer than any fixed bound: a
    /// model streaming one large tool call's arguments (those deltas emit
    /// nothing), an endpoint holding the stream open with keepalives through
    /// a long prompt, a provider whose `idle_timeout_secs` is above ten
    /// minutes. The client's idle timeout already ends a dead endpoint's turn
    /// and the tool timeouts bound a hung tool, so print mode must not cancel
    /// a quiet turn on a clock of its own (it did at 600 s, exiting 1 while
    /// the model was still writing).
    #[tokio::test(start_paused = true)]
    async fn print_mode_waits_out_a_quiet_turn() {
        let dir = std::env::temp_dir().join(format!(
            "openmax-headless-quiet-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (core, mut rx) = open_max_core::state::Core::new(dir.clone()).unwrap();
        let mut stdout = io::stdout();
        let mut stderr = io::stderr();
        let (mut saw_tokens, mut stop_by) = (false, None);
        let args = HeadlessArgs {
            prompts: Vec::new(),
            continue_session: false,
            json: true,
            quit: Arc::new(Notify::new()),
        };

        let turn = run_turn_events(
            &core, &mut rx, "s", &args, &mut saw_tokens, &mut stdout, &mut stderr, &mut stop_by,
        );
        tokio::pin!(turn);
        // An hour of silence passes in no real time while the clock is paused.
        let quiet = tokio::time::timeout(std::time::Duration::from_secs(3600), &mut turn).await;
        assert!(quiet.is_err(), "print mode ended a quiet turn with exit {:?}", quiet.ok());

        core.send_agent("s", AgentEvent::Done { stop_reason: "stop".to_string() });
        assert_eq!(turn.await, 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A tool file the agent named can carry a space or a shell
    /// metacharacter; the pasted repair command must name that file, not run
    /// part of it.
    #[test]
    fn the_approve_hint_quotes_the_path_an_operator_pastes() {
        let hint = super::approve_hint(".openmax/tools/my $(x) tool.toml", "abc123def456");
        assert!(
            hint.contains("openmax --approve '.openmax/tools/my $(x) tool.toml'"),
            "{hint}"
        );
        assert!(hint.ends_with("(abc123def456)"), "{hint}");
    }

    #[test]
    fn truncate_line_flattens_and_caps_on_char_boundaries() {
        assert_eq!(truncate_line("a\nb\rc", 10), "a b c");
        // The preview is raw tool output on a terminal. The old inline rule
        // flattened only \n and \r, so an ESC could restyle the rest of the
        // line and a Unicode line separator could break it; both go through
        // the workspace's one sanitizer now. A complete color sequence goes
        // whole, or its `[31m` tail would print; an incomplete one still has
        // its ESC flattened.
        assert_eq!(truncate_line("a\u{1b}[31mb", 20), "ab");
        assert_eq!(truncate_line("\u{1b}[1m\u{1b}[31merror\u{1b}(B\u{1b}[m: x", 20), "error: x");
        assert_eq!(truncate_line("a\u{1b}[31", 20), "a [31");
        assert_eq!(truncate_line("a\u{2028}b\u{2029}c", 20), "a b c");
        assert_eq!(truncate_line("  padded  ", 10), "padded");
        let long = "é".repeat(200);
        let cut = truncate_line(&long, 20);
        assert_eq!(cut.chars().count(), 20);
        assert!(cut.ends_with('…'));
    }
}
