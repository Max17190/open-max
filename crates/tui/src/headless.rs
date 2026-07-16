//! Headless (print) runner: drive the core agent loop without a TUI.
//! Proves the harness is the product and the terminal UI is one client.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use open_max_core::agent;
use open_max_core::sessions;
use open_max_core::state::{Core, CoreEvent};
use open_max_core::types::AgentEvent;
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
    mut core_rx: mpsc::UnboundedReceiver<CoreEvent>,
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
        if let Err(e) = agent::start_turn(
            core.clone(),
            session_id.clone(),
            project.clone(),
            prompt.clone(),
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

async fn run_turn_events(
    core: &Arc<Core>,
    core_rx: &mut mpsc::UnboundedReceiver<CoreEvent>,
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

        let CoreEvent::Agent(env) = event else {
            // Ignore MLX/download events in headless; the user configured the endpoint.
            continue;
        };
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
                    let _ = writeln!(stderr, "→ {name} {summary}");
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
            } => {
                let mode = core.settings.lock().unwrap().approval_mode.clone();
                let approve = mode == "auto";
                if !approve {
                    let _ = writeln!(
                        stderr,
                        "openmax: declining {name} ({summary}); set approval_mode to auto for unattended mutating tools"
                    );
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
                }
                return exit_code;
            }
            AgentEvent::SubagentProgress { kind, tool, step, .. } => {
                if !json {
                    let _ = writeln!(stderr, "  · task/{kind} step {step}: {tool}");
                    let _ = stderr.flush();
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
