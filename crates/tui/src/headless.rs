//! Headless (print) runner: drive the core agent loop without a TUI.
//! Proves the harness is the product and the terminal UI is one client.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use open_max_core::agent;
use open_max_core::sessions;
use open_max_core::state::Core;
use open_max_core::templates;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use tokio::sync::mpsc;

pub struct HeadlessArgs {
    /// One or more user prompts; each runs as a sequential turn on the same session.
    pub prompts: Vec<String>,
    pub continue_session: bool,
    pub json: bool,
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

    let session_id = if args.continue_session {
        match sessions::latest(&core, &project_key) {
            Some(meta) => meta.id,
            None => {
                eprintln!("openmax: no prior session in this directory to continue");
                return 2;
            }
        }
    } else {
        match sessions::create(&core, project_key) {
            Ok(meta) => meta.id,
            Err(e) => {
                eprintln!("openmax: failed to create session: {e}");
                return 1;
            }
        }
    };

    let mut exit_code = 0i32;
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();

    for prompt in &args.prompts {
        // Wait until the previous turn's spawn has cleared `running`. Done is
        // emitted before that cleanup, so starting immediately races.
        if !wait_until_idle(&core, &session_id).await {
            eprintln!("openmax: timed out waiting for the previous turn to finish");
            return 1;
        }

        // Prompt templates belong to the harness, not to the terminal UI: a
        // delegated `openmax -p "/greet world"` submits what the composer
        // would, so hooks and the transcript see the expanded text.
        if let Err(e) = agent::start_turn(
            core.clone(),
            session_id.clone(),
            project.clone(),
            templates::expand_user_input(&core.data_dir, &project, prompt),
        ) {
            eprintln!("openmax: {e}");
            return 1;
        }

        let mut saw_tokens = false;
        let turn_exit = run_turn_events(
            &core,
            &mut core_rx,
            &session_id,
            args.json,
            &mut saw_tokens,
            &mut stdout,
            &mut stderr,
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

/// Spin until the session is not marked running (or time out).
pub(crate) async fn wait_until_idle(core: &Arc<Core>, session_id: &str) -> bool {
    if !core.is_running(session_id) {
        return true;
    }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        if !core.is_running(session_id) {
            return true;
        }
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    !core.is_running(session_id)
}

async fn run_turn_events(
    core: &Arc<Core>,
    core_rx: &mut mpsc::UnboundedReceiver<AgentEventEnvelope>,
    session_id: &str,
    json: bool,
    saw_tokens: &mut bool,
    stdout: &mut io::Stdout,
    stderr: &mut io::Stderr,
) -> i32 {
    let mut exit_code = 0i32;

    loop {
        let event = match tokio::time::timeout(Duration::from_secs(600), core_rx.recv()).await {
            Ok(Some(ev)) => ev,
            Ok(None) => {
                let _ = writeln!(stderr, "openmax: event channel closed");
                return 1;
            }
            Err(_) => {
                let _ = writeln!(stderr, "openmax: timed out waiting for the agent");
                core.cancel(session_id);
                return 1;
            }
        };

        let env = event;
        if env.session_id != session_id {
            continue;
        }

        if json {
            if let Ok(line) = serde_json::to_string(&env) {
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
            } => {
                let mode = core.approval_mode();
                // Unattended auto mode covers the ordinary mutating gate, but
                // never the human boundary itself: the first run of
                // capability content no human has approved always needs a
                // person (interactively, or via openmax --approve).
                let approve = mode == open_max_core::config::ApprovalMode::Auto
                    && reason != "unapproved_source";
                if !approve {
                    // The path is the whole point of the line: a script's
                    // operator has to be able to copy the command and run it.
                    let hint = if reason == "unapproved_source" {
                        format!(
                            "a human must approve this tool's content first: openmax --approve {source_path} ({source_sha})"
                        )
                    } else {
                        "set approval_mode to auto for unattended mutating tools".to_string()
                    };
                    let _ = writeln!(stderr, "openmax: declining {}; {hint}", call_line(name, summary));
                }
                core.respond_approval(approval_id, approve);
            }
            AgentEvent::Error { message } => {
                if !json {
                    let _ = writeln!(stderr, "openmax: error: {message}");
                }
                exit_code = 1;
            }
            AgentEvent::Done { stop_reason } => {
                if !json {
                    let _ = writeln!(stdout);
                    if stop_reason != "stop" && stop_reason != "tool_calls" {
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
                    // asked. Resubmitting continues the work.
                    exit_code = 4;
                }
                return exit_code;
            }
            AgentEvent::Refrozen { tools, skills, changes } => {
                if !json {
                    let _ = writeln!(
                        stderr,
                        "openmax: re-frozen ({tools} tools, {skills} skills): {}",
                        changes.join(", ")
                    );
                }
            }
            AgentEvent::Compacted { tokens_before, tokens_after, compacted_messages } => {
                if !json {
                    let _ = writeln!(
                        stderr,
                        "openmax: compacted ~{tokens_before} to ~{tokens_after} tokens ({compacted_messages} messages archived)"
                    );
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
                    let _ = writeln!(stderr, "openmax: hook '{hook}' failed on {event}: {detail}");
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
    let one_line: String = s.chars().map(|c| if c == '\n' || c == '\r' { ' ' } else { c }).collect();
    let trimmed = one_line.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
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
            let mut saw_tokens = false;
            let code =
                run_turn_events(&core, &mut rx, "s", true, &mut saw_tokens, &mut stdout, &mut stderr)
                    .await;
            assert_eq!(code, expected, "stop reason {stop_reason}");
        }

        // An Error before the Done is what a truncated or blocked turn sends.
        for stop_reason in ["truncated", "blocked"] {
            core.send_agent("s", AgentEvent::Error { message: "boom".into() });
            core.send_agent("s", AgentEvent::Done { stop_reason: stop_reason.to_string() });
            let mut saw_tokens = false;
            let code =
                run_turn_events(&core, &mut rx, "s", true, &mut saw_tokens, &mut stdout, &mut stderr)
                    .await;
            assert_eq!(code, 1, "stop reason {stop_reason}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn truncate_line_flattens_and_caps_on_char_boundaries() {
        assert_eq!(truncate_line("a\nb\rc", 10), "a b c");
        assert_eq!(truncate_line("  padded  ", 10), "padded");
        let long = "é".repeat(200);
        let cut = truncate_line(&long, 20);
        assert_eq!(cut.chars().count(), 20);
        assert!(cut.ends_with('…'));
    }
}
