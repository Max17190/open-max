use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::oneshot;

use crate::client::{ChatClient, StreamDelta};
use crate::config::Settings;
use crate::fallback;
use crate::hooks::{Hooks, PreToolResult};
use crate::permissions::{PermissionDecision, Permissions};
use crate::prompt::{system_prompt_with_breakdown, PromptBreakdown};
use crate::registry::Registry;
use crate::sessions;
use crate::state::{CancelToken, Core, SessionData};
use crate::tools;
use crate::types::{AgentEvent, ChatMessage, ToolCall};

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(600);
/// Stream tokens to the UI in ~25ms batches: keeps redraw work negligible
/// with no perceptible latency.
const FLUSH_INTERVAL: Duration = Duration::from_millis(25);
const DIGEST_PREFIX: &str = "[context note:";

/// Outcome of a mutating-tool approval prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ApprovalOutcome {
    Approved,
    Declined,
    Cancelled,
    TimedOut,
}

/// True when a native server tool call cannot be executed as-is.
fn is_native_call_broken(call: &ToolCall) -> bool {
    call.function.name.is_empty()
        || serde_json::from_str::<Value>(&call.function.arguments).is_err()
}

/// When every native call is broken, try to recover calls from content markup.
/// Broken natives are only discarded if the markup actually yields calls;
/// otherwise they are kept so each one gets its per-call error (which tells
/// the model to retry) instead of vanishing silently.
fn resolve_tool_calls(
    mut content: String,
    mut tool_calls: Vec<ToolCall>,
    known_tools: &[&str],
) -> (String, Vec<ToolCall>) {
    let all_broken = !tool_calls.is_empty() && tool_calls.iter().all(is_native_call_broken);
    if tool_calls.is_empty() || all_broken {
        if let Some((clean, calls)) = fallback::extract_tool_calls(&content, known_tools) {
            content = clean;
            tool_calls = calls;
        }
    }
    (content, tool_calls)
}

/// Detects identical tool calls repeated consecutively within one turn loop.
struct RepeatCallTracker {
    last_name: Option<String>,
    last_args: Option<String>,
    consecutive: u8,
}

impl RepeatCallTracker {
    fn new() -> Self {
        Self { last_name: None, last_args: None, consecutive: 0 }
    }

    /// Returns true when this would be the 3rd consecutive identical execution.
    fn would_block(&self, name: &str, args_key: &str) -> bool {
        self.last_name.as_deref() == Some(name)
            && self.last_args.as_deref() == Some(args_key)
            && self.consecutive >= 2
    }

    fn record_executed(&mut self, name: &str, args_key: &str) {
        if self.last_name.as_deref() == Some(name) && self.last_args.as_deref() == Some(args_key) {
            self.consecutive = self.consecutive.saturating_add(1);
        } else {
            self.last_name = Some(name.to_string());
            self.last_args = Some(args_key.to_string());
            self.consecutive = 1;
        }
    }
}

fn canonicalize_args(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<_> = map.iter().collect();
            pairs.sort_by_key(|(k, _)| *k);
            let sorted: serde_json::Map<String, Value> =
                pairs.into_iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            serde_json::to_string(&Value::Object(sorted)).unwrap_or_default()
        }
        other => other.to_string(),
    }
}

/// One consecutive run of tool calls: `[start, end)`; `concurrent` when length >= 2
/// and every call in the run is batchable.
struct ToolCallSegment {
    start: usize,
    end: usize,
    concurrent: bool,
}

/// Split tool calls into maximal consecutive runs that are eligible for concurrent
/// read-only execution. Single-call runs and non-batchable calls use the serial path.
fn partition_concurrent_runs<F>(tool_calls: &[ToolCall], is_batchable: F) -> Vec<ToolCallSegment>
where
    F: Fn(&ToolCall) -> bool,
{
    let mut segments = Vec::new();
    let mut i = 0;
    while i < tool_calls.len() {
        if !is_batchable(&tool_calls[i]) {
            segments.push(ToolCallSegment { start: i, end: i + 1, concurrent: false });
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < tool_calls.len() && is_batchable(&tool_calls[i]) {
            i += 1;
        }
        let end = i;
        segments.push(ToolCallSegment {
            start,
            end,
            concurrent: end - start >= 2,
        });
    }
    segments
}

fn batchable_call(
    call: &ToolCall,
    registry: &Registry,
    repeat_tracker: &RepeatCallTracker,
    permissions: &Permissions,
) -> bool {
    let name = call.function.name.as_str();
    if name.is_empty() {
        return false;
    }
    let Ok(args) = serde_json::from_str::<Value>(&call.function.arguments) else {
        return false;
    };
    if registry.get(name).is_none() || registry.is_mutating(name) {
        return false;
    }
    // Ask needs the serial path so the approval UI runs; Deny stays serial for
    // a single clear error path (batch still handles Deny if it ever arrives).
    match permissions.evaluate(name, &args) {
        PermissionDecision::Ask | PermissionDecision::Deny { .. } => return false,
        PermissionDecision::Allow | PermissionDecision::Default => {}
    }
    let args_key = canonicalize_args(&args);
    !repeat_tracker.would_block(name, &args_key)
}

/// Append cancel/error tool messages for any tool_call_ids on the last assistant
/// message that still lack a following tool reply. Returns true if messages grew.
///
/// Assistant messages with `tool_calls` are persisted before tools run; a cancel
/// mid-turn can leave orphan call ids that break chat-template replay on resume.
fn complete_pending_tool_replies(messages: &mut Vec<ChatMessage>, note: &str) -> bool {
    let Some(asst_idx) = messages.iter().rposition(|m| {
        m.role == "assistant" && m.tool_calls.as_ref().is_some_and(|c| !c.is_empty())
    }) else {
        return false;
    };
    let ids: Vec<String> = messages[asst_idx]
        .tool_calls
        .as_ref()
        .map(|calls| calls.iter().map(|c| c.id.clone()).collect())
        .unwrap_or_default();
    // Own the answered set so we can push stubs without fighting the borrow checker.
    let answered: BTreeSet<String> = messages[asst_idx + 1..]
        .iter()
        .filter(|m| m.role == "tool")
        .filter_map(|m| m.tool_call_id.clone())
        .collect();
    let missing: Vec<String> = ids.into_iter().filter(|id| !answered.contains(id)).collect();
    if missing.is_empty() {
        return false;
    }
    for id in missing {
        messages.push(ChatMessage::tool(id, note));
    }
    true
}

fn build_session_data(core: &Arc<Core>, session_id: &str, project_root: &Path) -> SessionData {
    if let Some(mut messages) = sessions::load_messages(core, session_id) {
        // Resume: registry frozen at creation — manifest if present, else built-ins only.
        let registry = if let Some(manifest) = sessions::load_manifest(core, session_id) {
            Arc::new(Registry::from_manifest(manifest))
        } else {
            Arc::new(Registry::builtin_only())
        };
        let count = messages.len();
        let needs_system = messages.first().map(|m| m.role.as_str()) != Some("system");
        let (prompt_breakdown, persisted_count) = if needs_system {
            let (prompt, breakdown) = system_prompt_with_breakdown(project_root, &registry);
            messages.insert(0, ChatMessage::system(prompt));
            // In-memory prefix no longer matches what's on disk; rewrite on next save.
            (Arc::new(breakdown), 0usize)
        } else {
            let system_chars = messages
                .first()
                .and_then(|m| m.content.as_deref())
                .map(str::len)
                .unwrap_or(0);
            (Arc::new(PromptBreakdown::from_persisted(system_chars, &registry)), count)
        };
        SessionData {
            messages,
            registry,
            prompt_breakdown,
            persisted_count,
            snapshots: Default::default(),
            take_seq: 0,
        }
    } else {
        // No transcript on disk: start fresh, but honor a saved manifest if the
        // messages file was lost or emptied without wiping the registry snapshot.
        let (registry, had_manifest) = if let Some(manifest) = sessions::load_manifest(core, session_id) {
            (Arc::new(Registry::from_manifest(manifest)), true)
        } else {
            (Arc::new(Registry::build(project_root)), false)
        };
        if !had_manifest {
            // Always persisted (even builtin-only) so the extension
            // fingerprint travels with the session; without it every resume
            // would look like a self-modification and re-freeze for nothing.
            sessions::save_manifest(core, session_id, &registry.to_manifest());
        }
        let (prompt, breakdown) = system_prompt_with_breakdown(project_root, &registry);
        SessionData {
            messages: vec![ChatMessage::system(prompt)],
            registry,
            prompt_breakdown: Arc::new(breakdown),
            persisted_count: 0,
            snapshots: Default::default(),
            take_seq: 0,
        }
    }
}

fn tool_message_content(outcome: &tools::ToolOutcome) -> String {
    if outcome.ok || outcome.output.starts_with("Approval request timed out") {
        outcome.output.clone()
    } else {
        format!("Error: {}", outcome.output)
    }
}

struct ReadonlyBatchCtx<'a> {
    core: &'a Arc<Core>,
    session_id: &'a str,
    registry: &'a Arc<Registry>,
    project_root: &'a Path,
    caps: tools::OutputCaps,
    cancelled: Arc<CancelToken>,
    hooks: &'a Hooks,
    permissions: &'a Permissions,
}

async fn execute_readonly_batch(
    ctx: &ReadonlyBatchCtx<'_>,
    calls: &[ToolCall],
    messages: &mut Vec<ChatMessage>,
    repeat_tracker: &mut RepeatCallTracker,
) {
    let mut parsed: Vec<(Value, String)> = Vec::with_capacity(calls.len());
    let mut blocked: Vec<Option<tools::ToolOutcome>> = Vec::with_capacity(calls.len());
    for call in calls {
        let name = call.function.name.as_str();
        // batchable_call already validated the JSON; Null (impossible) would
        // just surface as a missing-argument tool error.
        let args: Value = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
        let args_key = canonicalize_args(&args);
        parsed.push((args.clone(), args_key));
        ctx.core.send_agent(ctx.session_id, AgentEvent::ToolStart {
            call_id: call.id.clone(),
            name: name.into(),
            args: args.clone(),
        });
        // hooks pre → permissions → (readonly tools skip approval_mode)
        let block = match ctx
            .hooks
            .pre_tool_use(ctx.session_id, name, &args, ctx.project_root, &ctx.cancelled)
            .await
        {
            PreToolResult::Block { reason } => Some(tools::ToolOutcome {
                ok: false,
                output: reason,
                diff: None,
            }),
            PreToolResult::Allow => match ctx.permissions.evaluate(name, &args) {
                PermissionDecision::Deny { reason } => Some(tools::ToolOutcome {
                    ok: false,
                    output: reason,
                    diff: None,
                }),
                // Ask is excluded from batching (see batchable_call); if it
                // still lands here, block rather than silently auto-run.
                PermissionDecision::Ask => Some(tools::ToolOutcome {
                    ok: false,
                    output: "permission rule requires approval; re-run outside a concurrent batch"
                        .into(),
                    diff: None,
                }),
                // Allow/Default: readonly batch tools are non-mutating.
                PermissionDecision::Allow | PermissionDecision::Default => None,
            },
        };
        blocked.push(block);
    }

    let futures: Vec<_> = calls
        .iter()
        .zip(parsed.iter())
        .zip(blocked.iter())
        .map(|((call, (args, _)), block)| {
            let name = call.function.name.clone();
            let args = args.clone();
            let root = ctx.project_root.to_path_buf();
            let cancel = ctx.cancelled.clone();
            let registry = ctx.registry.clone();
            let caps = ctx.caps;
            let blocked_outcome = block.clone();
            async move {
                if let Some(outcome) = blocked_outcome {
                    return outcome;
                }
                registry.execute(&name, &args, &root, caps, cancel).await
            }
        })
        .collect();
    let outcomes = futures_util::future::join_all(futures).await;

    for (i, call) in calls.iter().enumerate() {
        let outcome = &outcomes[i];
        let name = call.function.name.as_str();
        let args_key = &parsed[i].1;
        let args = &parsed[i].0;
        if blocked[i].is_none() {
            ctx.hooks
                .post_tool_use(
                    ctx.session_id,
                    name,
                    args,
                    ctx.project_root,
                    outcome.ok,
                    &ctx.cancelled,
                )
                .await;
        }
        if let Some(diff) = &outcome.diff {
            ctx.core.send_agent(ctx.session_id, AgentEvent::Diff {
                call_id: call.id.clone(),
                path: diff.path.clone(),
                diff: diff.diff.clone(),
                added: diff.added,
                removed: diff.removed,
            });
        }
        ctx.core.send_agent(ctx.session_id, AgentEvent::ToolEnd {
            call_id: call.id.clone(),
            ok: outcome.ok,
            output: outcome.output.clone(),
        });
        messages.push(ChatMessage::tool(call.id.clone(), tool_message_content(outcome)));
        if blocked[i].is_none() {
            repeat_tracker.record_executed(name, args_key);
        }
    }
}

/// Raw material for the model-written compaction summary: enough of each
/// dropped message to reconstruct the thread, hard-capped so the summary
/// request itself stays small.
const MAX_DROPPED_TEXT_CHARS: usize = 6_000;
const MAX_DROPPED_MSG_CHARS: usize = 240;

struct CompactionDigest {
    message_count: usize,
    tools: BTreeSet<String>,
    paths: Vec<String>,
    /// Short snippets of compacted user goals (not the original first request).
    user_snippets: Vec<String>,
    /// Role-labeled excerpts of everything dropped, oldest first, capped.
    dropped_text: String,
}

impl CompactionDigest {
    fn new() -> Self {
        Self {
            message_count: 0,
            tools: BTreeSet::new(),
            paths: Vec::new(),
            user_snippets: Vec::new(),
            dropped_text: String::new(),
        }
    }

    fn record_message(&mut self, msg: &ChatMessage) {
        self.message_count += 1;
        if self.dropped_text.len() < MAX_DROPPED_TEXT_CHARS {
            let mut line = format!("{}: ", msg.role);
            if let Some(c) = msg.content.as_deref() {
                line.extend(c.trim().chars().take(MAX_DROPPED_MSG_CHARS));
            }
            if let Some(calls) = &msg.tool_calls {
                for call in calls {
                    line.push_str(&format!(
                        " [called {} {}]",
                        call.function.name,
                        call.function.arguments.chars().take(120).collect::<String>()
                    ));
                }
            }
            line.push('\n');
            self.dropped_text.push_str(&line);
        }
        if msg.role == "user" {
            if let Some(c) = msg.content.as_deref() {
                let trimmed = c.trim();
                if !trimmed.is_empty()
                    && !trimmed.starts_with(DIGEST_PREFIX)
                    && self.user_snippets.len() < 4
                {
                    let snippet: String = trimmed.chars().take(120).collect();
                    if !self.user_snippets.iter().any(|s| s == &snippet) {
                        self.user_snippets.push(snippet);
                    }
                }
            }
            return;
        }
        if msg.role != "assistant" {
            return;
        }
        let Some(calls) = &msg.tool_calls else { return };
        for call in calls {
            if !call.function.name.is_empty() {
                self.tools.insert(call.function.name.clone());
            }
            if let Ok(v) = serde_json::from_str::<Value>(&call.function.arguments) {
                if let Some(path) = v.get("path").and_then(|p| p.as_str()) {
                    if self.paths.len() < 8 && !self.paths.iter().any(|p| p == path) {
                        self.paths.push(path.to_string());
                    }
                }
            }
        }
    }

    fn format(&self) -> String {
        let mut parts = vec![format!(
            "{DIGEST_PREFIX} {} earlier messages were compacted.",
            self.message_count
        )];
        if !self.tools.is_empty() {
            parts.push(format!(
                "Tools used: {}.",
                self.tools.iter().cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        if !self.paths.is_empty() {
            parts.push(format!("Files touched: {}.", self.paths.join(", ")));
        }
        if !self.user_snippets.is_empty() {
            parts.push(format!(
                "Earlier goals: {}.",
                self.user_snippets.join(" | ")
            ));
        }
        parts.push("Re-read files if you need the details.".into());
        parts.join(" ")
    }

    /// The note used when the model wrote a real summary of the dropped
    /// context; exact paths stay listed because summaries paraphrase them.
    fn format_with_summary(&self, summary: &str) -> String {
        let mut parts = vec![format!(
            "{DIGEST_PREFIX} {} earlier messages were compacted. Summary: {summary}",
            self.message_count
        )];
        if !self.paths.is_empty() {
            parts.push(format!("Files touched: {}.", self.paths.join(", ")));
        }
        parts.push("Re-read files if you need the details.".into());
        parts.join(" ")
    }

    fn to_record(&self, digest: String) -> sessions::CompactionRecord {
        sessions::CompactionRecord {
            ts: sessions::unix_now(),
            message_count: self.message_count,
            tools: self.tools.iter().cloned().collect(),
            paths: self.paths.clone(),
            user_snippets: self.user_snippets.clone(),
            digest,
        }
    }
}

/// One small completion against the session's own endpoint turns the dropped
/// exchanges into a real summary; the heuristic digest note is the fallback
/// whenever this returns None (error, timeout, cancel, or empty reply). One
/// request per compaction, which is rare by construction (hysteresis prune).
const SUMMARY_TIMEOUT: Duration = Duration::from_secs(25);
const MAX_SUMMARY_CHARS: usize = 900;

async fn summarize_compaction(
    client: &ChatClient,
    digest: &CompactionDigest,
    cancelled: &Arc<CancelToken>,
) -> Option<String> {
    if digest.dropped_text.trim().is_empty() {
        return None;
    }
    let messages = vec![
        ChatMessage::system(
            "You compress dropped context from a coding-agent session. Reply with only the \
             summary, no preamble: at most 120 words covering what was being done, exact file \
             paths involved, decisions made, and anything unresolved.",
        ),
        ChatMessage::user(digest.dropped_text.clone()),
    ];
    let result = tokio::time::timeout(
        SUMMARY_TIMEOUT,
        client.stream_chat(&messages, "[]", cancelled.clone(), |_| {}),
    )
    .await
    .ok()?
    .ok()?;
    let mut summary = result.content;
    if let Some(clean) = fallback::strip_leading_think(&summary) {
        summary = clean;
    }
    let summary = summary.trim().replace(['\n', '\r'], " ");
    if summary.is_empty() {
        return None;
    }
    if summary.chars().count() > MAX_SUMMARY_CHARS {
        return Some(summary.chars().take(MAX_SUMMARY_CHARS).collect::<String>() + "…");
    }
    Some(summary)
}

fn is_digest_message(msg: &ChatMessage) -> bool {
    msg.role == "user"
        && msg.content.as_deref().is_some_and(|c| c.starts_with(DIGEST_PREFIX))
}

/// Kick off one agent turn in a session. Errors if that session is already running.
pub fn start_turn(
    core: Arc<Core>,
    session_id: String,
    project_root: PathBuf,
    user_text: String,
) -> Result<(), String> {
    {
        let mut running = core.running.lock().unwrap();
        if running.contains(&session_id) {
            return Err("the agent is already working in this session".into());
        }
        running.insert(session_id.clone());
    }
    let cancelled = Arc::new(CancelToken::default());
    core.cancel_flags
        .lock()
        .unwrap()
        .insert(session_id.clone(), cancelled.clone());

    let settings = core.settings.lock().unwrap().clone();
    // Title is set after user_prompt_submit accepts the text (see run_loop).
    // Titling here would fail-open secret/PII gates into the session index.

    tokio::spawn(async move {
        run_loop(&core, &session_id, &project_root, user_text, settings, cancelled).await;
        core.running.lock().unwrap().remove(&session_id);
        core.cancel_flags.lock().unwrap().remove(&session_id);
    });
    Ok(())
}

/// Re-freeze a live session's registry and system prompt from the current
/// on-disk config. This is the deliberate, user-triggered cache break behind
/// `/reload`: the agent authors a tool, skill, or prompt change mid-session
/// and uses it in the same conversation instead of losing context to `/new`.
/// Returns `(tool_count, skill_count)` of the newly frozen registry.
pub async fn reload_session(
    core: &Arc<Core>,
    session_id: &str,
    project_root: &Path,
) -> Result<(usize, usize), String> {
    if core.is_running(session_id) {
        return Err("a turn is in flight; run /reload after it finishes".into());
    }
    let root = project_root.to_path_buf();
    let registry = tokio::task::spawn_blocking(move || Registry::build(&root))
        .await
        .map_err(|e| format!("reload discovery failed: {e}"))?;
    let (prompt, breakdown) = system_prompt_with_breakdown(project_root, &registry);
    let counts = (registry.tools.len(), registry.skills.len());

    // Hydrate first if the session was resumed but never ran a turn, so the
    // reload applies to the real transcript rather than a fresh one.
    let hydrated = core.sessions.lock().await.contains_key(session_id);
    if !hydrated {
        let core_clone = core.clone();
        let session_id_owned = session_id.to_string();
        let project_root_owned = project_root.to_path_buf();
        let built = tokio::task::spawn_blocking(move || {
            build_session_data(&core_clone, &session_id_owned, &project_root_owned)
        })
        .await
        .map_err(|e| format!("reload hydration failed: {e}"))?;
        core.sessions.lock().await.entry(session_id.to_string()).or_insert(built);
    }

    let mut sessions_map = core.sessions.lock().await;
    let data = sessions_map
        .get_mut(session_id)
        .ok_or_else(|| "session state is unavailable; try /new".to_string())?;
    // A turn that slipped past the running check owns the transcript
    // (mem::take leaves it empty); refuse rather than clobber.
    if data.messages.is_empty() {
        return Err("a turn is in flight; run /reload after it finishes".into());
    }
    apply_freeze(core, session_id, data, registry, prompt, breakdown);
    Ok(counts)
}

/// Install a rebuilt registry + system prompt into a live session and persist
/// the new shape (manifest plus a full transcript rewrite, since the prefix
/// changed). Shared by /reload and the automatic turn-start re-freeze.
fn apply_freeze(
    core: &Arc<Core>,
    session_id: &str,
    data: &mut SessionData,
    registry: Registry,
    prompt: String,
    breakdown: PromptBreakdown,
) {
    if data.messages.first().is_some_and(|m| m.role == "system") {
        data.messages[0] = ChatMessage::system(prompt);
    } else {
        data.messages.insert(0, ChatMessage::system(prompt));
    }
    data.registry = Arc::new(registry);
    data.prompt_breakdown = Arc::new(breakdown);
    sessions::save_manifest(core, session_id, &data.registry.to_manifest());
    data.persisted_count = 0;
    sessions::save_messages(core, session_id, &data.messages, &mut data.persisted_count, true);
}

/// The self-modification loop closes here: at turn start, if the extension
/// files on disk no longer match the fingerprint the session's registry froze
/// from, rebuild registry + prompt in place. Costs one fingerprint read per
/// turn (a handful of small files) and re-prefills the prompt cache only when
/// something actually changed; a tool the agent wrote last turn is callable
/// on this one, no /new and no human /reload required.
async fn refreeze_if_extensions_changed(
    core: &Arc<Core>,
    session_id: &str,
    project_root: &Path,
) {
    let disk_fp = {
        let root = project_root.to_path_buf();
        match tokio::task::spawn_blocking(move || crate::registry::extensions_fingerprint(&root))
            .await
        {
            Ok(fp) => fp,
            Err(_) => return,
        }
    };
    let stale = {
        let sessions_map = core.sessions.lock().await;
        sessions_map
            .get(session_id)
            .is_some_and(|d| !d.messages.is_empty() && d.registry.ext_fingerprint != disk_fp)
    };
    if !stale {
        return;
    }
    let root = project_root.to_path_buf();
    let Ok(registry) = tokio::task::spawn_blocking(move || Registry::build(&root)).await else {
        return;
    };
    let (prompt, breakdown) = system_prompt_with_breakdown(project_root, &registry);
    let counts = (registry.tools.len(), registry.skills.len());
    let mut sessions_map = core.sessions.lock().await;
    if let Some(data) = sessions_map.get_mut(session_id) {
        // Re-check under the lock: this turn owns `running`, so nothing else
        // mutates the session, but stay defensive about empty (taken) state.
        if !data.messages.is_empty() && data.registry.ext_fingerprint != disk_fp {
            apply_freeze(core, session_id, data, registry, prompt, breakdown);
            core.send_agent(session_id, AgentEvent::Refrozen {
                tools: counts.0,
                skills: counts.1,
            });
        }
    }
}

/// Buffers streamed deltas and flushes them as batched events.
struct TokenBatcher {
    core: Arc<Core>,
    session_id: String,
    content: String,
    thinking: String,
    last_flush: Instant,
}

impl TokenBatcher {
    fn new(core: Arc<Core>, session_id: String) -> Self {
        Self { core, session_id, content: String::new(), thinking: String::new(), last_flush: Instant::now() }
    }

    fn push(&mut self, delta: StreamDelta) {
        match delta {
            StreamDelta::Content(t) => self.content.push_str(&t),
            StreamDelta::Reasoning(t) => self.thinking.push_str(&t),
        }
        if self.last_flush.elapsed() >= FLUSH_INTERVAL {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if !self.content.is_empty() {
            self.core.send_agent(&self.session_id, AgentEvent::Token { text: std::mem::take(&mut self.content) });
        }
        if !self.thinking.is_empty() {
            self.core.send_agent(&self.session_id, AgentEvent::Thinking { text: std::mem::take(&mut self.thinking) });
        }
        self.last_flush = Instant::now();
    }
}

async fn run_loop(
    core: &Arc<Core>,
    session_id: &str,
    project_root: &Path,
    user_text: String,
    settings: Settings,
    cancelled: Arc<CancelToken>,
) {
    // Discover hooks first: user_prompt_submit gates the input before it
    // ever enters the transcript. A blocked submit is not a started turn
    // (no title write, no session_start, no turn_end).
    let hooks = Hooks::discover(project_root);
    if let PreToolResult::Block { reason } = hooks
        .user_prompt_submit(session_id, &user_text, project_root, &cancelled)
        .await
    {
        core.send_agent(session_id, AgentEvent::Error {
            message: format!("input blocked: {reason}"),
        });
        core.send_agent(session_id, AgentEvent::Done { stop_reason: "blocked".into() });
        return;
    }

    // Accepted: title from the first real prompt, then self-modification.
    sessions::set_title_if_new(core, session_id, &user_text);

    // Self-modification: pick up extension files written since the last
    // freeze before this turn's schemas and prompt are locked in.
    refreeze_if_extensions_changed(core, session_id, project_root).await;

    // Take ownership of the in-memory transcript for this turn (no full clone).
    // MessageGuard restores it on drop so panic/abort cannot empty the session.
    let (messages, registry, take_seq, first_turn) = {
        {
            let mut sessions_map = core.sessions.lock().await;
            if let Some(data) = sessions_map.get_mut(session_id) {
                let first_turn = data.messages.len() <= 1;
                data.messages.push(ChatMessage::user(user_text));
                let (messages, seq) = take_messages(data);
                let registry = data.registry.clone();
                (messages, registry, seq, first_turn)
            } else {
                drop(sessions_map);
                let core_clone = core.clone();
                let session_id_owned = session_id.to_string();
                let project_root_owned = project_root.to_path_buf();
                let built = tokio::task::spawn_blocking(move || {
                    build_session_data(&core_clone, &session_id_owned, &project_root_owned)
                })
                .await
                .expect("session hydration task panicked");
                let mut sessions_map = core.sessions.lock().await;
                let data = sessions_map.entry(session_id.to_string()).or_insert(built);
                let first_turn = data.messages.len() <= 1;
                data.messages.push(ChatMessage::user(user_text));
                let (messages, seq) = take_messages(data);
                let registry = data.registry.clone();
                (messages, registry, seq, first_turn)
            }
        }
    };
    let mut guard = MessageGuard::new(core.clone(), session_id, messages, take_seq);

    // Discovered once per turn start; empty dirs/files are a cheap no-op.
    // Permissions never enter the prompt, so reloading next turn is fine.
    let permissions = Permissions::discover(project_root);

    // Resolve named provider (or flat base_url) once per turn so settings edits
    // apply without restarting the process. An explicit but unknown provider
    // fails closed rather than silently hitting flat base_url.
    let endpoint = match crate::providers::resolve(&settings, &core.data_dir) {
        Ok(ep) => ep,
        Err(e) => {
            core.send_agent(session_id, AgentEvent::Error { message: e.to_string() });
            // User message was already appended; restore so the next turn sees it.
            guard.commit().await;
            hooks.turn_end(session_id, project_root, "error").await;
            core.send_agent(session_id, AgentEvent::Done { stop_reason: "error".into() });
            return;
        }
    };
    let client = ChatClient::from_endpoint(&endpoint);
    // Frozen wire form once per turn: every iteration injects the same tool
    // schema bytes without re-serializing the Value array.
    let schemas_wire = registry.tool_schemas_wire();
    let known_tools: Vec<&str> = registry.tools.iter().map(|s| s.name.as_str()).collect();
    let caps = tools::OutputCaps::from_settings(&settings);
    // Fires exactly once per session, on the turn that first populates it
    // (fresh session or a resume that only had its system prompt).
    if first_turn {
        hooks.session_start(session_id, project_root, &cancelled).await;
    }
    // Every break assigns a real reason; this survives only if the model kept
    // calling tools until the iteration cap.
    let mut stop_reason = String::from("max_iterations");
    let mut repeat_tracker = RepeatCallTracker::new();
    let context_tokens = endpoint.context_tokens;
    let max_tokens = endpoint.max_tokens;
    let max_iterations = settings.max_agent_iterations.max(1);

    'turns: for _ in 0..max_iterations {
        let (budget_changed, compaction) = enforce_budget(
            guard.messages(),
            context_tokens.saturating_sub(max_tokens + 1024),
        );
        if let Some(digest) = compaction {
            // Upgrade the heuristic note to a model-written summary when the
            // endpoint cooperates; the note at index 2 was just inserted by
            // enforce_budget, so replacing it here keeps one digest message.
            let mut note = digest.format();
            if let Some(summary) = summarize_compaction(&client, &digest, &cancelled).await {
                note = digest.format_with_summary(&summary);
                let messages = guard.messages();
                if messages.len() > 2 && is_digest_message(&messages[2]) {
                    messages[2] = ChatMessage::user(note.clone());
                }
            }
            let record = digest.to_record(note);
            sessions::append_compaction(core, session_id, &record);
            if let Ok(value) = serde_json::to_value(&record) {
                hooks.compaction(session_id, project_root, &value, &cancelled).await;
            }
        }
        let used = guard.messages().iter().map(|m| m.estimated_tokens()).sum();
        core.send_agent(session_id, AgentEvent::Budget { used_tokens: used, context_tokens });

        let batcher = Arc::new(StdMutex::new(TokenBatcher::new(core.clone(), session_id.to_string())));
        let batcher_in = batcher.clone();
        let result = client
            .stream_chat(guard.messages(), schemas_wire, cancelled.clone(), move |delta| {
                batcher_in.lock().unwrap().push(delta);
            })
            .await;
        batcher.lock().unwrap().flush();

        let result = match result {
            Ok(r) => r,
            Err(message) => {
                core.send_agent(session_id, AgentEvent::Error { message });
                stop_reason = "error".into();
                break 'turns;
            }
        };

        if let Some(u) = result.usage {
            core.send_agent(session_id, AgentEvent::Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                cached_tokens: u.cached_tokens,
            });
        }

        // Prefer structured calls from the server; when there are none (or all
        // are broken), recover calls from raw markup in the content (see fallback.rs).
        let mut content = result.content.clone();
        let mut tool_calls = result.tool_calls.clone();
        // Reasoning leaked into content is display-only: persisting it would
        // re-prefill dead tokens on every later turn.
        if let Some(clean) = fallback::strip_leading_think(&content) {
            content = clean;
        }
        (content, tool_calls) = resolve_tool_calls(content, tool_calls, &known_tools);
        core.send_agent(session_id, AgentEvent::MessageDone { text: content.clone() });

        // Never persist a fully empty assistant message (e.g. a turn cancelled
        // before the first token): chat templates can reject it on replay.
        if !content.is_empty() || !tool_calls.is_empty() {
            guard.messages().push(ChatMessage::assistant(
                if content.is_empty() { None } else { Some(content.clone()) },
                if tool_calls.is_empty() { None } else { Some(tool_calls.clone()) },
            ));
            save_messages(core, session_id, guard.messages(), budget_changed).await;
        }

        if cancelled.is_cancelled() {
            stop_reason = "cancelled".into();
            break 'turns;
        }
        if tool_calls.is_empty() {
            stop_reason = result.finish_reason;
            break 'turns;
        }

        let segments = partition_concurrent_runs(&tool_calls, |call| {
            batchable_call(call, &registry, &repeat_tracker, &permissions)
        });

        'calls: for segment in segments {
            if cancelled.is_cancelled() {
                stop_reason = "cancelled".into();
                break 'turns;
            }

            if segment.concurrent {
                let batch_ctx = ReadonlyBatchCtx {
                    core,
                    session_id,
                    registry: &registry,
                    project_root,
                    caps,
                    cancelled: cancelled.clone(),
                    hooks: &hooks,
                    permissions: &permissions,
                };
                execute_readonly_batch(
                    &batch_ctx,
                    &tool_calls[segment.start..segment.end],
                    guard.messages(),
                    &mut repeat_tracker,
                )
                .await;
                continue 'calls;
            }

            for call in &tool_calls[segment.start..segment.end] {
                if cancelled.is_cancelled() {
                    stop_reason = "cancelled".into();
                    break 'turns;
                }
                let name = call.function.name.as_str();
                if name.is_empty() {
                    let msg = "tool call has an empty function name; use a known tool name from the schema";
                    core.send_agent(session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: String::new(), args: Value::Null });
                    core.send_agent(session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.into() });
                    guard.messages().push(ChatMessage::tool(call.id.clone(), format!("Error: {msg}")));
                    continue;
                }
                let args: Value = match serde_json::from_str(&call.function.arguments) {
                    Ok(v) => v,
                    Err(e) => {
                        let msg = format!("invalid JSON in tool arguments: {e}");
                        core.send_agent(session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: name.into(), args: Value::Null });
                        core.send_agent(session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.clone() });
                        guard.messages().push(ChatMessage::tool(call.id.clone(), format!("Error: {msg}")));
                        continue;
                    }
                };

                let args_key = canonicalize_args(&args);
                if repeat_tracker.would_block(name, &args_key) {
                    let msg = "You have repeated this exact call 3 times. The result will not change. Try a different approach, or explain what you are blocked on.";
                    core.send_agent(session_id, AgentEvent::ToolStart { call_id: call.id.clone(), name: name.into(), args: args.clone() });
                    core.send_agent(session_id, AgentEvent::ToolEnd { call_id: call.id.clone(), ok: false, output: msg.into() });
                    guard.messages().push(ChatMessage::tool(call.id.clone(), msg.to_string()));
                    continue;
                }

                core.send_agent(session_id, AgentEvent::ToolStart {
                    call_id: call.id.clone(),
                    name: name.into(),
                    args: args.clone(),
                });

                // Order: hooks pre → permissions → approval_mode → execute.
                // Denies never prompt the user.
                if let PreToolResult::Block { reason } = hooks
                    .pre_tool_use(session_id, name, &args, project_root, &cancelled)
                    .await
                {
                    core.send_agent(session_id, AgentEvent::ToolEnd {
                        call_id: call.id.clone(),
                        ok: false,
                        output: reason.clone(),
                    });
                    guard.messages().push(ChatMessage::tool(
                        call.id.clone(),
                        tool_message_content(&tools::ToolOutcome {
                            ok: false,
                            output: reason,
                            diff: None,
                        }),
                    ));
                    continue;
                }

                let perm = permissions.evaluate(name, &args);
                if let PermissionDecision::Deny { reason } = &perm {
                    core.send_agent(session_id, AgentEvent::ToolEnd {
                        call_id: call.id.clone(),
                        ok: false,
                        output: reason.clone(),
                    });
                    guard.messages().push(ChatMessage::tool(
                        call.id.clone(),
                        tool_message_content(&tools::ToolOutcome {
                            ok: false,
                            output: reason.clone(),
                            diff: None,
                        }),
                    ));
                    continue;
                }

                if registry.is_mutating(name) {
                    snapshot_file(core, session_id, project_root, &args).await;
                }

                // Read live so "[a]lways" during an approval prompt takes effect
                // for the rest of this turn, not just the next one.
                let approval_mode = core.settings.lock().unwrap().approval_mode.clone();
                // Allow skips the approval prompt; Ask forces it (even in auto).
                // Readonly still blocks mutating tools regardless of Allow.
                let force_allow = matches!(perm, PermissionDecision::Allow);
                let force_ask = matches!(perm, PermissionDecision::Ask);
                let mut executed = false;
                let (outcome, turn_cancelled) = if registry.is_mutating(name) && approval_mode == "readonly" {
                    (tools::ToolOutcome {
                        ok: false,
                        output: "This session is read-only; mutating tools are disabled. Explain what you would do instead.".into(),
                        diff: None,
                    }, false)
                } else if !force_allow
                    && (force_ask || (registry.is_mutating(name) && approval_mode == "ask"))
                {
                    match request_approval(core, session_id, name, &args, &cancelled).await {
                        ApprovalOutcome::Approved => {
                            executed = true;
                            (registry.execute(name, &args, project_root, caps, cancelled.clone()).await, false)
                        }
                        ApprovalOutcome::Declined => (tools::ToolOutcome {
                            ok: false,
                            output: "The user declined this action. Ask them how to proceed instead of retrying.".into(),
                            diff: None,
                        }, false),
                        ApprovalOutcome::TimedOut => (tools::ToolOutcome {
                            ok: false,
                            output: "Approval request timed out with no response. Stop and summarize what you were about to do.".into(),
                            diff: None,
                        }, false),
                        ApprovalOutcome::Cancelled => (tools::ToolOutcome {
                            ok: false,
                            output: "The user cancelled this turn.".into(),
                            diff: None,
                        }, true),
                    }
                } else {
                    executed = true;
                    (registry.execute(name, &args, project_root, caps, cancelled.clone()).await, false)
                };

                if turn_cancelled {
                    core.send_agent(session_id, AgentEvent::ToolEnd {
                        call_id: call.id.clone(),
                        ok: false,
                        output: "The user cancelled this turn.".into(),
                    });
                    guard.messages().push(ChatMessage::tool(call.id.clone(), "The user cancelled this turn."));
                    stop_reason = "cancelled".into();
                    break 'turns;
                }

                if executed {
                    hooks
                        .post_tool_use(
                            session_id,
                            name,
                            &args,
                            project_root,
                            outcome.ok,
                            &cancelled,
                        )
                        .await;
                }

                if let Some(diff) = &outcome.diff {
                    core.send_agent(session_id, AgentEvent::Diff {
                        call_id: call.id.clone(),
                        path: diff.path.clone(),
                        diff: diff.diff.clone(),
                        added: diff.added,
                        removed: diff.removed,
                    });
                }
                core.send_agent(session_id, AgentEvent::ToolEnd {
                    call_id: call.id.clone(),
                    ok: outcome.ok,
                    output: outcome.output.clone(),
                });

                // Approval timeouts are not model errors; the "Error:" prefix
                // would push small models into pointless retry loops.
                guard.messages().push(ChatMessage::tool(call.id.clone(), tool_message_content(&outcome)));
                if executed {
                    repeat_tracker.record_executed(name, &args_key);
                }
            }
        }
        save_messages(core, session_id, guard.messages(), false).await;
    }

    // Cancel mid-turn may leave the last assistant's tool_calls without tool
    // role replies (siblings after an approval cancel, or cancel before tools
    // ran). Stub them so resume templates stay well-formed.
    if stop_reason == "cancelled" || cancelled.is_cancelled() {
        let _ = complete_pending_tool_replies(guard.messages(), "The user cancelled this turn.");
    }

    save_messages(core, session_id, guard.messages(), false).await;
    // Restore in-memory transcript under the async lock (Drop is try_lock only).
    guard.commit().await;
    sessions::touch(core, session_id);
    hooks.turn_end(session_id, project_root, &stop_reason).await;
    core.send_agent(session_id, AgentEvent::Done { stop_reason });
}

/// Record a file's pre-edit content the first time this session touches it,
/// enabling cumulative per-file diffs.
async fn snapshot_file(core: &Arc<Core>, session_id: &str, project_root: &Path, args: &Value) {
    let Some(rel) = args["path"].as_str() else { return };
    let content = std::fs::read_to_string(project_root.join(rel)).unwrap_or_default();
    let mut sessions_map = core.sessions.lock().await;
    if let Some(data) = sessions_map.get_mut(session_id) {
        data.snapshots.entry(rel.to_string()).or_insert(content);
    }
}

/// Persist transcript to disk without cloning it back into SessionData.
/// The turn owns `messages` until `MessageGuard` commits on drop/finish.
async fn save_messages(core: &Arc<Core>, session_id: &str, messages: &[ChatMessage], rewrite: bool) {
    let mut sessions_map = core.sessions.lock().await;
    if let Some(data) = sessions_map.get_mut(session_id) {
        sessions::save_messages(core, session_id, messages, &mut data.persisted_count, rewrite);
    }
}

/// Process-unique ids for transcript takes. Starts at 1 so a freshly built
/// `SessionData` (`take_seq: 0`) can never match a live guard.
static TAKE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Take the transcript for a turn and stamp the session with a fresh take id.
/// The paired [`MessageGuard`] may only write back while the stamp matches.
fn take_messages(data: &mut SessionData) -> (Vec<ChatMessage>, u64) {
    let seq = TAKE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    data.take_seq = seq;
    (std::mem::take(&mut data.messages), seq)
}

/// Holds the turn-local transcript and restores it to SessionData on drop so a
/// panic or early exit cannot leave the session empty after `mem::take`.
///
/// Normal exits call [`MessageGuard::commit`] (async lock). `Drop` first
/// `try_lock`s (`blocking_lock` panics inside a Tokio async context); if the
/// lock is contended the restore is handed to a spawned task. Every write-back
/// requires the session's `take_seq` to still equal this guard's — a newer
/// turn or a recreated session re-stamps it, which turns a late restore into
/// a no-op instead of installing stale context.
struct MessageGuard {
    core: Arc<Core>,
    session_id: String,
    messages: Option<Vec<ChatMessage>>,
    take_seq: u64,
}

fn restore_if_current(
    map: &mut std::collections::HashMap<String, SessionData>,
    session_id: &str,
    take_seq: u64,
    messages: Vec<ChatMessage>,
) {
    if let Some(data) = map.get_mut(session_id) {
        if data.take_seq == take_seq && data.messages.is_empty() {
            data.messages = messages;
        }
    }
}

impl MessageGuard {
    fn new(core: Arc<Core>, session_id: &str, messages: Vec<ChatMessage>, take_seq: u64) -> Self {
        Self {
            core,
            session_id: session_id.to_string(),
            messages: Some(messages),
            take_seq,
        }
    }

    fn messages(&mut self) -> &mut Vec<ChatMessage> {
        self.messages.as_mut().expect("messages already committed")
    }

    /// Move the working transcript back into SessionData. Consumes the guard
    /// so `Drop` becomes a no-op.
    async fn commit(mut self) {
        if let Some(messages) = self.messages.take() {
            let mut map = self.core.sessions.lock().await;
            restore_if_current(&mut map, &self.session_id, self.take_seq, messages);
        }
    }
}

impl Drop for MessageGuard {
    fn drop(&mut self) {
        let Some(messages) = self.messages.take() else {
            return;
        };
        match self.core.sessions.try_lock() {
            Ok(mut map) => {
                restore_if_current(&mut map, &self.session_id, self.take_seq, messages);
            }
            Err(_) => {
                // Lock contended mid-unwind. Discarding here would leave the
                // session entry present-but-empty for the process lifetime
                // (it never rehydrates from disk once the entry exists), so
                // hand the restore to the runtime instead.
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    let core = self.core.clone();
                    let session_id = std::mem::take(&mut self.session_id);
                    let take_seq = self.take_seq;
                    handle.spawn(async move {
                        let mut map = core.sessions.lock().await;
                        restore_if_current(&mut map, &session_id, take_seq, messages);
                    });
                }
                // No runtime means process teardown; disk saves bound the loss.
            }
        }
    }
}

async fn request_approval(
    core: &Arc<Core>,
    session_id: &str,
    name: &str,
    args: &Value,
    cancelled: &Arc<CancelToken>,
) -> ApprovalOutcome {
    let approval_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = oneshot::channel::<bool>();
    core.approvals.lock().unwrap().insert(approval_id.clone(), tx);
    let summary = crate::registry::summarize_call(name, args);
    let detail = approval_detail(args);
    core.send_agent(session_id, AgentEvent::ApprovalRequest {
        approval_id: approval_id.clone(),
        name: name.to_string(),
        summary,
        detail,
    });

    let outcome = tokio::select! {
        r = rx => match r {
            Ok(true) => ApprovalOutcome::Approved,
            Ok(false) => ApprovalOutcome::Declined,
            Err(_) => ApprovalOutcome::Declined,
        },
        _ = cancelled.cancelled() => ApprovalOutcome::Cancelled,
        _ = tokio::time::sleep(APPROVAL_TIMEOUT) => ApprovalOutcome::TimedOut,
    };

    core.approvals.lock().unwrap().remove(&approval_id);
    let outcome_label = match outcome {
        ApprovalOutcome::Approved => "approved",
        ApprovalOutcome::Declined => "declined",
        ApprovalOutcome::TimedOut => "timed_out",
        ApprovalOutcome::Cancelled => "cancelled",
    };
    core.send_agent(
        session_id,
        AgentEvent::ApprovalSettled {
            approval_id,
            outcome: outcome_label.into(),
        },
    );
    outcome
}

/// Compact args preview for the approval card (paths, command head, etc.).
fn approval_detail(args: &Value) -> String {
    if let Some(obj) = args.as_object() {
        let mut parts = Vec::new();
        for key in ["path", "command", "old_string", "new_string", "content", "pattern", "glob"] {
            if let Some(v) = obj.get(key) {
                let s = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let one = s.replace(['\n', '\r'], " ");
                let clipped: String = one.chars().take(120).collect();
                if !clipped.is_empty() {
                    parts.push(format!("{key}={clipped}"));
                }
            }
        }
        if !parts.is_empty() {
            return parts.join(" · ");
        }
    }
    let raw = args.to_string();
    if raw == "null" || raw == "{}" {
        String::new()
    } else {
        raw.chars().take(160).collect()
    }
}

/// Keep the transcript inside the model's context window: first truncate old
/// tool outputs, then drop the oldest exchanges (always preserving the system
/// prompt and the original user request). Returns true when messages changed.
///
/// Prunes with hysteresis: once the budget is crossed, compact well below it
/// (PRUNE_TARGET_PCT) in a single pass. The server-side prompt cache re-prefills
/// from the first byte that diverges, so mutating early messages every
/// iteration would force a near-full prefill per agent step; pruning hard and
/// then leaving history untouched keeps the transcript append-only (and the
/// cache warm) until the budget is crossed again.
const PRUNE_TARGET_PCT: usize = 70;

/// Returns `(changed, exchange_digest)` where `exchange_digest` is set only when
/// whole exchanges were dropped (not when only tool outputs were truncated).
fn enforce_budget(
    messages: &mut Vec<ChatMessage>,
    budget: usize,
) -> (bool, Option<CompactionDigest>) {
    let mut total: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
    if total <= budget {
        return (false, None);
    }
    let target = budget * PRUNE_TARGET_PCT / 100;
    let keep_tail = messages.len().saturating_sub(6);
    let mut truncated = false;
    for msg in messages.iter_mut().take(keep_tail).skip(1) {
        if msg.role == "tool" {
            if let Some(c) = &msg.content {
                if c.len() > 600 {
                    let mut cut = 160;
                    while !c.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    let old = msg.estimated_tokens();
                    msg.content = Some(format!("{}\n…[older tool output truncated]", &c[..cut]));
                    total = total.saturating_sub(old).saturating_add(msg.estimated_tokens());
                    truncated = true;
                }
            }
        }
        if total <= target {
            return (true, None);
        }
    }
    // Drop whole exchanges starting after [system, first user]. Keep tool
    // replies consistent with the assistant message that requested them.
    let mut digest = CompactionDigest::new();
    while total > target && messages.len() > 6 {
        let removed = messages.remove(2);
        digest.record_message(&removed);
        total = total.saturating_sub(removed.estimated_tokens());
        if removed.role == "assistant" && removed.tool_calls.is_some() {
            while messages.len() > 2 && messages[2].role == "tool" {
                let tool = messages.remove(2);
                digest.record_message(&tool);
                total = total.saturating_sub(tool.estimated_tokens());
            }
        }
    }
    if digest.message_count > 0 {
        let note = ChatMessage::user(digest.format());
        if messages.len() > 2 && is_digest_message(&messages[2]) {
            messages[2] = note;
        } else {
            messages.insert(2, note);
        }
        // Digest insert can push total slightly over target; keep dropping
        // exchanges after the digest (index 3) so the next turn stays
        // append-only and does not re-mutate history for another prune.
        // Record dropped messages into the same digest so the note stays a
        // faithful summary of everything removed (not only the first pass).
        total = messages.iter().map(|m| m.estimated_tokens()).sum();
        while total > target && messages.len() > 6 {
            let removed = messages.remove(3);
            digest.record_message(&removed);
            total = total.saturating_sub(removed.estimated_tokens());
            if removed.role == "assistant" && removed.tool_calls.is_some() {
                while messages.len() > 3 && messages[3].role == "tool" {
                    let tool = messages.remove(3);
                    digest.record_message(&tool);
                    total = total.saturating_sub(tool.estimated_tokens());
                }
            }
        }
        // Always refresh the note after the drop loop so extra removals are
        // reflected even when the first-pass note was already inserted above.
        if messages.len() > 2 && is_digest_message(&messages[2]) {
            messages[2] = ChatMessage::user(digest.format());
        }
        (true, Some(digest))
    } else {
        (truncated, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ToolCall, ToolCallFunction};

    fn msg(role: &str, len: usize) -> ChatMessage {
        ChatMessage { role: role.into(), content: Some("x".repeat(len)), tool_calls: None, tool_call_id: None }
    }

    fn assistant_with_tools(name: &str, args: &str) -> ChatMessage {
        ChatMessage::assistant(
            None,
            Some(vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: ToolCallFunction {
                    name: name.into(),
                    arguments: args.into(),
                },
            }]),
        )
    }

    #[test]
    fn broken_native_calls_fall_back_to_markup() {
        let known = ["read_file", "bash"];
        let content = "I'll read it.\n<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}</tool_call>";
        let broken = vec![ToolCall {
            id: "call_0".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: String::new(),
                arguments: "{not json".into(),
            },
        }];
        let (clean, calls) = resolve_tool_calls(content.into(), broken, &known);
        assert_eq!(clean, "I'll read it.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
    }

    #[test]
    fn partial_broken_native_calls_keep_native() {
        let known = ["read_file", "bash"];
        let good = ToolCall {
            id: "call_0".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "bash".into(),
                arguments: r#"{"command":"echo hi"}"#.into(),
            },
        };
        let bad = ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: String::new(),
                arguments: "nope".into(),
            },
        };
        let (clean, calls) = resolve_tool_calls("run".into(), vec![good, bad], &known);
        assert_eq!(clean, "run");
        assert_eq!(calls.len(), 2);
        assert!(is_native_call_broken(&calls[1]));
    }

    #[test]
    fn repeat_tracker_blocks_third_identical_call() {
        let mut t = RepeatCallTracker::new();
        assert!(!t.would_block("bash", r#"{"command":"ls"}"#));
        t.record_executed("bash", r#"{"command":"ls"}"#);
        assert!(!t.would_block("bash", r#"{"command":"ls"}"#));
        t.record_executed("bash", r#"{"command":"ls"}"#);
        assert!(t.would_block("bash", r#"{"command":"ls"}"#));
    }

    #[test]
    fn repeat_tracker_resets_on_different_call() {
        let mut t = RepeatCallTracker::new();
        t.record_executed("bash", r#"{"command":"ls"}"#);
        t.record_executed("bash", r#"{"command":"ls"}"#);
        assert!(!t.would_block("read_file", r#"{"path":"a.rs"}"#));
        t.record_executed("read_file", r#"{"path":"a.rs"}"#);
        assert!(!t.would_block("read_file", r#"{"path":"a.rs"}"#));
    }

    fn tool_call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: format!("call_{name}"),
            kind: "function".into(),
            function: ToolCallFunction {
                name: name.into(),
                arguments: args.into(),
            },
        }
    }

    #[test]
    fn complete_pending_stubs_missing_tool_replies() {
        let mut messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("go"),
            ChatMessage::assistant(
                None,
                Some(vec![
                    ToolCall {
                        id: "c1".into(),
                        kind: "function".into(),
                        function: ToolCallFunction {
                            name: "read_file".into(),
                            arguments: r#"{"path":"a"}"#.into(),
                        },
                    },
                    ToolCall {
                        id: "c2".into(),
                        kind: "function".into(),
                        function: ToolCallFunction {
                            name: "read_file".into(),
                            arguments: r#"{"path":"b"}"#.into(),
                        },
                    },
                    ToolCall {
                        id: "c3".into(),
                        kind: "function".into(),
                        function: ToolCallFunction {
                            name: "grep".into(),
                            arguments: r#"{"pattern":"x"}"#.into(),
                        },
                    },
                ]),
            ),
            // Only the first call was answered before cancel.
            ChatMessage::tool("c1", "ok"),
        ];
        let note = "The user cancelled this turn.";
        assert!(complete_pending_tool_replies(&mut messages, note));
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[4].tool_call_id.as_deref(), Some("c2"));
        assert_eq!(messages[4].content.as_deref(), Some(note));
        assert_eq!(messages[5].tool_call_id.as_deref(), Some("c3"));
        assert_eq!(messages[5].content.as_deref(), Some(note));
        // Idempotent once every id has a reply.
        assert!(!complete_pending_tool_replies(&mut messages, note));
    }

    #[test]
    fn complete_pending_noop_when_all_replied_or_no_tool_calls() {
        let note = "The user cancelled this turn.";
        let mut plain = vec![
            ChatMessage::system("sys"),
            ChatMessage::assistant(Some("hi".into()), None),
        ];
        assert!(!complete_pending_tool_replies(&mut plain, note));

        let mut done = vec![
            ChatMessage::assistant(
                None,
                Some(vec![ToolCall {
                    id: "only".into(),
                    kind: "function".into(),
                    function: ToolCallFunction {
                        name: "list_dir".into(),
                        arguments: r#"{"path":"."}"#.into(),
                    },
                }]),
            ),
            ChatMessage::tool("only", "files"),
        ];
        assert!(!complete_pending_tool_replies(&mut done, note));
    }

    #[test]
    fn partition_splits_readonly_runs_and_breaks_on_mutating() {
        let registry = Registry::builtin_only();
        let tracker = RepeatCallTracker::new();
        let calls = vec![
            tool_call("read_file", r#"{"path":"a.rs"}"#),
            tool_call("read_file", r#"{"path":"b.rs"}"#),
            tool_call("write_file", r#"{"path":"c.rs","content":"x"}"#),
            tool_call("glob", r#"{"pattern":"**/*.rs"}"#),
            tool_call("grep", r#"{"pattern":"fn"}"#),
        ];
        let empty_perms = Permissions::default();
        let segments = partition_concurrent_runs(&calls, |c| {
            batchable_call(c, &registry, &tracker, &empty_perms)
        });
        assert_eq!(segments.len(), 3);
        assert!(segments[0].concurrent && segments[0].start == 0 && segments[0].end == 2);
        assert!(!segments[1].concurrent && segments[1].start == 2 && segments[1].end == 3);
        assert!(segments[2].concurrent && segments[2].start == 3 && segments[2].end == 5);
    }

    #[test]
    fn partition_four_readonly_tools_batch_concurrently() {
        let registry = Registry::builtin_only();
        let tracker = RepeatCallTracker::new();
        let empty_perms = Permissions::default();
        let calls = vec![
            tool_call("list_dir", r#"{"path":"."}"#),
            tool_call("read_file", r#"{"path":"a.rs"}"#),
            tool_call("glob", r#"{"pattern":"**/*.rs"}"#),
            tool_call("grep", r#"{"pattern":"fn"}"#),
        ];
        let segments = partition_concurrent_runs(&calls, |c| {
            batchable_call(c, &registry, &tracker, &empty_perms)
        });
        assert_eq!(segments.len(), 1);
        assert!(segments[0].concurrent);
        assert_eq!((segments[0].start, segments[0].end), (0, 4));
        assert!(!batchable_call(
            &tool_call("write_file", r#"{"path":"x","content":"y"}"#),
            &registry,
            &tracker,
            &empty_perms,
        ));
        assert!(!batchable_call(
            &tool_call("nope", r#"{}"#),
            &registry,
            &tracker,
            &empty_perms,
        ));
    }

    #[test]
    fn partition_single_readonly_call_is_serial() {
        let registry = Registry::builtin_only();
        let tracker = RepeatCallTracker::new();
        let calls = vec![tool_call("read_file", r#"{"path":"a.rs"}"#)];
        let empty_perms = Permissions::default();
        let segments = partition_concurrent_runs(&calls, |c| {
            batchable_call(c, &registry, &tracker, &empty_perms)
        });
        assert_eq!(segments.len(), 1);
        assert!(!segments[0].concurrent);
    }

    #[test]
    fn partition_breaks_on_invalid_json_and_unknown_tools() {
        let registry = Registry::builtin_only();
        let tracker = RepeatCallTracker::new();
        let calls = vec![
            tool_call("read_file", r#"{"path":"a.rs"}"#),
            ToolCall {
                id: "bad_json".into(),
                kind: "function".into(),
                function: ToolCallFunction { name: "read_file".into(), arguments: "not json".into() },
            },
            tool_call("nope", r#"{"x":1}"#),
        ];
        let empty_perms = Permissions::default();
        let segments = partition_concurrent_runs(&calls, |c| {
            batchable_call(c, &registry, &tracker, &empty_perms)
        });
        assert_eq!(segments.len(), 3);
        assert!(!segments[0].concurrent);
        assert!(!segments[1].concurrent);
        assert!(!segments[2].concurrent);
    }

    #[tokio::test]
    async fn message_guard_restores_after_contended_drop() {
        use crate::state::Core;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "guard-contended";
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();
        {
            let data = build_session_data(&core, id, &project);
            core.sessions.lock().await.insert(id.to_string(), data);
        }

        // Mirror a turn: take the transcript, then drop the guard while
        // another task holds the sessions lock (abort/unwind under contention).
        let (taken, seq) = {
            let mut map = core.sessions.lock().await;
            let data = map.get_mut(id).unwrap();
            data.messages.push(ChatMessage::user("hello"));
            take_messages(data)
        };
        let expected = taken.len();
        assert!(expected > 0);
        let guard = MessageGuard::new(core.clone(), id, taken, seq);

        let held = core.sessions.lock().await;
        drop(guard);
        drop(held);

        // The restore is handed to a spawned task; give it time to run.
        let mut restored = 0;
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let map = core.sessions.lock().await;
            restored = map.get(id).unwrap().messages.len();
            if restored == expected {
                break;
            }
        }
        assert_eq!(restored, expected, "contended drop must not lose the transcript");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn message_guard_skips_restore_after_newer_take() {
        use crate::state::Core;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "guard-stale";
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();
        {
            let data = build_session_data(&core, id, &project);
            core.sessions.lock().await.insert(id.to_string(), data);
        }

        let (taken_a, seq_a) = {
            let mut map = core.sessions.lock().await;
            take_messages(map.get_mut(id).unwrap())
        };
        assert!(!taken_a.is_empty());
        let guard_a = MessageGuard::new(core.clone(), id, taken_a, seq_a);

        // A newer turn takes the (empty) slot before guard A unwinds; guard
        // A's restore must now be a no-op, or turn B would run against stale
        // context that B's commit then silently drops.
        let (taken_b, seq_b) = {
            let mut map = core.sessions.lock().await;
            take_messages(map.get_mut(id).unwrap())
        };
        drop(guard_a);
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        {
            let map = core.sessions.lock().await;
            assert!(
                map.get(id).unwrap().messages.is_empty(),
                "stale guard must not fill a slot owned by a newer take"
            );
        }

        // Turn B commits with its own (current) take id as usual.
        let mut guard_b = MessageGuard::new(core.clone(), id, taken_b, seq_b);
        guard_b.messages().push(ChatMessage::user("from b"));
        guard_b.commit().await;
        let map = core.sessions.lock().await;
        assert_eq!(map.get(id).unwrap().messages.len(), 1);
        drop(map);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn build_session_data_honors_manifest_without_messages() {
        use crate::state::Core;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "manifest-only";
        let project = dir.join("project");
        std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
        std::fs::write(
            project.join(".openmax/tools/deploy.toml"),
            "name = \"deploy\"\ndescription = \"ships it\"\ncommand = \"/bin/true\"\nmutating = true\n",
        )
        .unwrap();
        let original = crate::registry::Registry::build(&project);
        sessions::save_manifest(&core, id, &original.to_manifest());
        std::fs::remove_dir_all(project.join(".openmax/tools")).unwrap();

        let data = build_session_data(&core, id, &project);
        assert_eq!(data.messages[0].role, "system");
        assert!(data.registry.is_mutating("deploy"));
        assert_eq!(
            data.registry.tool_schemas_json().to_string(),
            original.tool_schemas_json().to_string()
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn reload_session_refreezes_registry_prompt_and_manifest() {
        use crate::state::Core;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "reload-live";
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();
        {
            let mut data = build_session_data(&core, id, &project);
            data.messages.push(ChatMessage::user("hi"));
            data.messages.push(ChatMessage::assistant(Some("hello".into()), None));
            assert!(data.registry.get("deploy").is_none());
            core.sessions.lock().await.insert(id.to_string(), data);
        }

        // The agent writes a new tool mid-session; /reload must pick it up.
        std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
        std::fs::write(
            project.join(".openmax/tools/deploy.toml"),
            "name = \"deploy\"\ndescription = \"ships it\"\ncommand = \"/bin/true\"\nmutating = true\n",
        )
        .unwrap();

        // A running turn blocks the reload.
        core.running.lock().unwrap().insert(id.to_string());
        assert!(reload_session(&core, id, &project).await.is_err());
        core.running.lock().unwrap().remove(id);

        let (tools, skills) = reload_session(&core, id, &project).await.unwrap();
        assert_eq!(tools, tools::TOOL_NAMES.len() + 1);
        assert_eq!(skills, 0);

        let map = core.sessions.lock().await;
        let data = map.get(id).unwrap();
        assert!(data.registry.is_mutating("deploy"));
        assert_eq!(data.messages[0].role, "system");
        assert_eq!(data.messages.len(), 3, "conversation must survive the reload");
        assert_eq!(data.persisted_count, 3, "transcript must be rewritten to disk");
        drop(map);
        let manifest = sessions::load_manifest(&core, id).expect("manifest saved");
        assert!(manifest.external_tools.iter().any(|t| t.name == "deploy"));

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The agent-native loop: a tool written mid-session is frozen in at the
    /// next turn start with no human action, and an unchanged disk is a no-op
    /// (prompt cache stays warm).
    #[tokio::test]
    async fn turn_start_refreezes_only_when_extension_files_changed() {
        use crate::state::Core;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, mut rx) = Core::new(dir.clone());
        let id = "auto-refreeze";
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();
        {
            let mut data = build_session_data(&core, id, &project);
            data.messages.push(ChatMessage::user("hi"));
            core.sessions.lock().await.insert(id.to_string(), data);
        }

        // Unchanged disk: no-op, no event, same registry Arc.
        let before = core.sessions.lock().await.get(id).unwrap().registry.clone();
        refreeze_if_extensions_changed(&core, id, &project).await;
        {
            let map = core.sessions.lock().await;
            assert!(Arc::ptr_eq(&map.get(id).unwrap().registry, &before), "no-op must not rebuild");
        }

        // The agent writes a tool; the next turn start must freeze it in.
        std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
        std::fs::write(
            project.join(".openmax/tools/deploy.toml"),
            "name = \"deploy\"\ndescription = \"ships it\"\ncommand = \"/bin/true\"\nmutating = true\n",
        )
        .unwrap();
        refreeze_if_extensions_changed(&core, id, &project).await;
        {
            let map = core.sessions.lock().await;
            let data = map.get(id).unwrap();
            assert!(data.registry.is_mutating("deploy"));
            assert_eq!(data.messages.len(), 2, "conversation survives the re-freeze");
        }
        let manifest = sessions::load_manifest(&core, id).expect("manifest rewritten");
        assert!(manifest.external_tools.iter().any(|t| t.name == "deploy"));
        assert_ne!(manifest.ext_fingerprint, 0);
        let mut saw_refrozen = false;
        while let Ok(ev) = rx.try_recv() {
            if let crate::state::CoreEvent::Agent(env) = ev {
                if matches!(env.event, AgentEvent::Refrozen { tools, .. } if tools == tools::TOOL_NAMES.len() + 1) {
                    saw_refrozen = true;
                }
            }
        }
        assert!(saw_refrozen, "UI must be told the session shape changed");

        // Second check with no further writes: converged, no rebuild.
        let after = core.sessions.lock().await.get(id).unwrap().registry.clone();
        refreeze_if_extensions_changed(&core, id, &project).await;
        let map = core.sessions.lock().await;
        assert!(Arc::ptr_eq(&map.get(id).unwrap().registry, &after), "must converge");
        drop(map);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// user_prompt_submit blocks before the message enters the transcript,
    /// and the turn ends with stop_reason "blocked" (no model call).
    #[tokio::test]
    async fn user_prompt_submit_blocks_before_transcript() {
        use crate::state::{Core, CoreEvent};
        use std::io::Write as _;
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, mut rx) = Core::new(dir.clone());
        let project = dir.join("project");
        let hooks_dir = project.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = project.join("gate.sh");
        {
            let mut f = std::fs::File::create(&script).unwrap();
            f.write_all(b"#!/bin/sh\necho 'blocked by policy'; exit 1\n").unwrap();
            let mut perms = std::fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script, perms).unwrap();
        }
        std::fs::write(
            hooks_dir.join("gate.toml"),
            format!("event = \"user_prompt_submit\"\ncommand = \"{}\"\n", script.display()),
        )
        .unwrap();
        crate::hooks::invalidate_hooks_cache();

        let project_key = project.display().to_string();
        let meta = sessions::create(&core, project_key.clone()).unwrap();
        let id = meta.id.clone();
        // Pre-seed a system-only session so we can assert the blocked text
        // never lands in the transcript.
        {
            let data = build_session_data(&core, &id, &project);
            core.sessions.lock().await.insert(id.clone(), data);
        }

        start_turn(core.clone(), id.clone(), project.clone(), "should not land".into()).unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let mut stop = None;
        let mut saw_error = false;
        while tokio::time::Instant::now() < deadline {
            if let Ok(Some(CoreEvent::Agent(env))) =
                tokio::time::timeout(Duration::from_millis(200), rx.recv()).await
            {
                match env.event {
                    AgentEvent::Error { message } => {
                        assert!(message.contains("input blocked"), "{message}");
                        saw_error = true;
                    }
                    AgentEvent::Done { stop_reason } => {
                        stop = Some(stop_reason);
                        break;
                    }
                    _ => {}
                }
            }
        }
        assert!(saw_error, "must emit an Error with the block reason");
        assert_eq!(stop.as_deref(), Some("blocked"));
        let messages = core.sessions.lock().await.get(&id).unwrap().messages.clone();
        assert!(
            messages.iter().all(|m| m.content.as_deref() != Some("should not land")),
            "blocked text must not enter the transcript: {messages:?}"
        );
        // Session index title must not absorb blocked text (secret fail-open).
        let listed = sessions::list(&core, &project_key);
        let title = listed.iter().find(|m| m.id == id).expect("session in index").title.clone();
        assert_eq!(title, sessions::UNTITLED, "blocked prompt must not set the title");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every turn exit fires turn_end, including the early provider-failure
    /// return that never reaches the main loop.
    #[tokio::test]
    async fn turn_end_hook_fires_on_provider_resolution_failure() {
        use crate::state::{Core, CoreEvent};
        use std::io::Write as _;
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, mut rx) = Core::new(dir.clone());
        core.settings.lock().unwrap().provider = Some("no-such-provider".into());
        let project = dir.join("project");
        let hooks_dir = project.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = project.join("end.sh");
        {
            let mut f = std::fs::File::create(&script).unwrap();
            f.write_all(format!("#!/bin/sh\ncat > {}/end.json\n", project.display()).as_bytes())
                .unwrap();
            let mut perms = std::fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script, perms).unwrap();
        }
        std::fs::write(
            hooks_dir.join("end.toml"),
            format!("event = \"turn_end\"\ncommand = \"{}\"\n", script.display()),
        )
        .unwrap();
        crate::hooks::invalidate_hooks_cache();

        start_turn(core.clone(), "sess-early".into(), project.clone(), "hi".into()).unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let mut saw_done = false;
        while tokio::time::Instant::now() < deadline {
            if let Ok(Some(CoreEvent::Agent(env))) =
                tokio::time::timeout(Duration::from_millis(200), rx.recv()).await
            {
                if matches!(env.event, AgentEvent::Done { .. }) {
                    saw_done = true;
                    break;
                }
            }
        }
        assert!(saw_done, "early provider failure must still emit Done");
        let end: Value =
            serde_json::from_str(&std::fs::read_to_string(project.join("end.json")).unwrap())
                .unwrap();
        assert_eq!(end["event"], "turn_end");
        assert_eq!(end["stop_reason"], "error");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn build_session_data_injects_system_when_resume_lacks_one() {
        use crate::state::Core;

        let dir = std::env::temp_dir().join(format!("openmax-agent-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "legacy-no-system";
        let mut persisted = 0usize;
        sessions::save_messages(&core, id, &[ChatMessage::user("hello")], &mut persisted, false);

        let data = build_session_data(&core, id, Path::new("."));
        assert_eq!(data.messages[0].role, "system");
        assert_eq!(data.messages[1].role, "user");
        assert_eq!(data.persisted_count, 0, "must rewrite on next save after injecting system");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn budget_preserves_system_and_first_user() {
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..20 {
            messages.push(msg("assistant", 2000));
            messages.push(msg("user", 2000));
        }
        let _ = enforce_budget(&mut messages, 2000);
        // Floor is system + first user + digest + a short tail (post-digest
        // drops may trim one more exchange when the digest itself overshoots).
        assert!(messages.len() >= 6 && messages.len() <= 7, "len={}", messages.len());
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("x".repeat(400).as_str()));
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
    }

    #[test]
    fn budget_truncates_old_tool_output_first() {
        let mut messages = vec![msg("system", 100), msg("user", 100)];
        messages.push(msg("tool", 4000));
        messages.push(msg("assistant", 100));
        // Recent tail that must stay intact.
        for _ in 0..3 {
            messages.push(msg("user", 100));
            messages.push(msg("assistant", 100));
        }
        let (changed, digest) = enforce_budget(&mut messages, 700);
        assert!(changed);
        assert!(digest.is_none(), "truncate-only should not emit an exchange digest");
        assert_eq!(messages.len(), 10, "nothing should be dropped, only truncated");
        let tool_len = messages[2].content.as_deref().unwrap().len();
        assert!(tool_len < 500, "old tool output should be truncated, got {tool_len}");
    }

    /// One prune must buy headroom: after compaction the transcript sits at or
    /// below the prune target, and re-running enforce_budget mutates nothing,
    /// so the token prefix (and the server's prompt cache) stays stable while
    /// the next iterations append.
    #[test]
    fn budget_prunes_once_with_hysteresis() {
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..8 {
            messages.push(msg("assistant", 100));
            messages.push(msg("tool", 3000));
        }
        let budget = 4000;
        assert!(enforce_budget(&mut messages, budget).0);
        let total: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
        assert!(
            total <= budget * PRUNE_TARGET_PCT / 100,
            "prune should reach the target, got {total} of {budget}"
        );

        let snapshot: Vec<Option<String>> = messages.iter().map(|m| m.content.clone()).collect();
        assert!(!enforce_budget(&mut messages, budget).0, "second pass must be a no-op");
        let after: Vec<Option<String>> = messages.iter().map(|m| m.content.clone()).collect();
        assert_eq!(snapshot, after, "no message may change between prunes");
    }

    #[test]
    fn budget_digest_replaced_not_stacked() {
        let mut messages = vec![msg("system", 100), msg("user", 100)];
        for i in 0..12 {
            messages.push(assistant_with_tools("read_file", &format!(r#"{{"path":"src/{i}.rs"}}"#)));
            messages.push(msg("tool", 2500));
        }
        let budget = 3000;
        let (changed, digest) = enforce_budget(&mut messages, budget);
        assert!(changed);
        assert!(digest.is_some());
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
        let first_digest = messages[2].content.clone();
        assert!(!enforce_budget(&mut messages, budget).0, "second pass must be a no-op");
        assert_eq!(messages[2].content, first_digest, "digest must not be replaced on no-op");

        for _ in 0..6 {
            messages.push(assistant_with_tools("edit_file", r#"{"path":"src/new.rs"}"#));
            messages.push(msg("tool", 2500));
        }
        assert!(enforce_budget(&mut messages, budget).0);
        let digest_count = messages
            .iter()
            .filter(|m| m.content.as_deref().is_some_and(|c| c.starts_with(DIGEST_PREFIX)))
            .count();
        assert_eq!(digest_count, 1, "only one digest note may exist");
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
    }

    #[test]
    fn digest_captures_dropped_text_for_summarization() {
        let mut digest = CompactionDigest::new();
        digest.record_message(&ChatMessage::user("implement the auth flow"));
        digest.record_message(&assistant_with_tools("read_file", r#"{"path":"src/auth.rs"}"#));
        digest.record_message(&msg("tool", 5000));
        let text = &digest.dropped_text;
        assert!(text.contains("user: implement the auth flow"), "{text}");
        assert!(text.contains("[called read_file"), "{text}");
        // Per-message and total caps hold.
        assert!(text.len() < MAX_DROPPED_TEXT_CHARS + 1000);
        for _ in 0..100 {
            digest.record_message(&msg("assistant", 500));
        }
        assert!(digest.dropped_text.len() < MAX_DROPPED_TEXT_CHARS + 1000, "total cap must hold");

        let note = digest.format_with_summary("Was wiring auth middleware; src/auth.rs half-edited.");
        assert!(note.starts_with(DIGEST_PREFIX));
        assert!(note.contains("Summary: Was wiring auth middleware"));
        assert!(note.contains("src/auth.rs"));
    }

    #[test]
    fn budget_digest_includes_tools_paths_and_goals() {
        let mut messages = vec![msg("system", 100), msg("user", 100)];
        messages.push(ChatMessage::user("implement the auth flow carefully"));
        messages.push(assistant_with_tools("read_file", r#"{"path":"src/auth.rs"}"#));
        messages.push(msg("tool", 3000));
        for _ in 0..10 {
            messages.push(msg("assistant", 2000));
            messages.push(msg("user", 2000));
        }
        let (_, digest) = enforce_budget(&mut messages, 2500);
        let digest = digest.expect("exchange drop should produce a digest");
        let text = digest.format();
        assert!(text.contains("read_file"), "{text}");
        assert!(text.contains("src/auth.rs"), "{text}");
        assert!(text.contains("Earlier goals"), "{text}");
    }

    /// After a prune that inserts a digest, token total must sit at or below
    /// the hysteresis target so the next iteration does not re-mutate history.
    #[test]
    fn budget_post_digest_stays_at_or_below_target() {
        let mut messages = vec![msg("system", 200), msg("user", 200)];
        for i in 0..16 {
            messages.push(assistant_with_tools(
                "read_file",
                &format!(r#"{{"path":"src/module_{i}.rs"}}"#),
            ));
            messages.push(msg("tool", 1800));
        }
        let budget = 3500;
        let target = budget * PRUNE_TARGET_PCT / 100;
        let (changed, digest) = enforce_budget(&mut messages, budget);
        assert!(changed);
        assert!(digest.is_some());
        let total: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
        assert!(
            total <= target,
            "post-digest total {total} must be <= target {target} (budget {budget})"
        );
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
        assert!(!enforce_budget(&mut messages, budget).0, "second pass must be a no-op");
    }
}
