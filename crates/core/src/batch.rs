//! Concurrent read-only batches. A run of two or more consecutive calls
//! that need no card (a registered tool that does not mutate, that no
//! permission rule asks about or denies, and whose content a human approved
//! unless the mode is auto) runs at once, at most `max_parallel_tools` at a
//! time, and its replies land in the order the model sent the calls. Every
//! other call takes the serial path in `agent::run_loop`, the one place a
//! card can be raised.

use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use futures_util::StreamExt;
use serde_json::Value;

use crate::agent::{
    count_usage, declined_message, end_call, refresh_approval_policy_canonical, unapproved_capability, TurnUsage,
    POLICY_CHANGED_DURING_HOOK,
};
use crate::config::ApprovalMode;
use crate::hooks::{Hooks, PreToolResult};
use crate::notes::report_hook_failures;
use crate::permissions::{PermissionDecision, TurnPermissions};
use crate::registry::Registry;
use crate::state::{CancelToken, Core};
use crate::tools;
use crate::types::{AgentEvent, ChatMessage, ToolCall};

/// One consecutive run of tool calls: `[start, end)`; `concurrent` when length >= 2
/// and every call in the run is batchable.
pub(crate) struct ToolCallSegment {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) concurrent: bool,
}

/// Split tool calls into maximal consecutive runs that are eligible for concurrent
/// read-only execution. Single-call runs and non-batchable calls use the serial path.
pub(crate) fn partition_concurrent_runs<F>(tool_calls: &[ToolCall], is_batchable: F) -> Vec<ToolCallSegment>
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

pub(crate) fn batchable_call(
    call: &ToolCall,
    registry: &Registry,
    permissions: &TurnPermissions,
    mode: ApprovalMode,
    data_dir: &Path,
    project_root: &Path,
) -> bool {
    let name = call.function.name.as_str();
    if name.is_empty() {
        return false;
    }
    let Ok(args) = serde_json::from_str::<Value>(&call.function.arguments) else { return false };
    if registry.get(name).is_none() || registry.is_mutating(name) {
        return false;
    }
    // Ask needs the serial path so the approval UI runs; Deny stays serial for
    // a single clear error path (batch still handles Deny if it ever arrives).
    match permissions.evaluate(name, &args) {
        PermissionDecision::Ask | PermissionDecision::Deny { .. } => return false,
        PermissionDecision::Allow | PermissionDecision::Default => {}
    }
    // Same principle as Ask: outside auto, a tool whose content no human has
    // approved needs the serial path, which is where the prompt and its
    // actionable event live. This check is load-bearing rather than
    // defensive - batching selects for external non-mutating tools, which is
    // exactly the population the content gate exists to catch, so an
    // unapproved tool called twice in one message would otherwise run
    // unattended. Auto runs it unattended anyway, so there the question is
    // never asked and the ledger never read. Kept last because it is the only
    // check that touches disk, and built-ins return before the ledger is read.
    mode == ApprovalMode::Auto || unapproved_capability(registry, data_dir, project_root, name).is_none()
}

/// A batched call that auto admitted without asking about its content, still
/// waiting for a slot when the mode left auto. Not declined: requested
/// again, it takes the serial path, which raises the card.
pub(crate) const MODE_LEFT_AUTO_BEFORE_START: &str = "Execution mode changed before this call started; this call was not executed. Request it again under the current mode.";

pub(crate) struct ReadonlyBatchCtx<'a> {
    pub(crate) core: &'a Arc<Core>,
    pub(crate) session_id: &'a str,
    pub(crate) registry: &'a Arc<Registry>,
    pub(crate) project_root: &'a Path,
    /// `project_root` as the turn resolved it once, so a mode read walks no path.
    pub(crate) canonical_root: &'a Path,
    pub(crate) caps: tools::OutputCaps,
    pub(crate) cancelled: Arc<CancelToken>,
    pub(crate) hooks: &'a mut Hooks,
    pub(crate) permissions: &'a mut TurnPermissions,
    pub(crate) policy_mode: &'a mut ApprovalMode,
    pub(crate) parallelism: usize,
    /// Turn-local usage accumulator, flushed once at turn end.
    pub(crate) usage: &'a std::sync::Mutex<TurnUsage>,
}

pub(crate) fn parallel_tool_limit(configured: usize) -> usize {
    configured.clamp(1, 32)
}

pub(crate) async fn collect_bounded<F>(futures: Vec<F>, parallelism: usize) -> Vec<F::Output>
where
    F: Future,
{
    let mut completed: Vec<_> = futures_util::stream::iter(
        futures
            .into_iter()
            .enumerate()
            .map(|(index, future)| async move { (index, future.await) }),
    )
    .buffer_unordered(parallel_tool_limit(parallelism))
    .collect()
    .await;
    completed.sort_unstable_by_key(|(index, _)| *index);
    completed.into_iter().map(|(_, output)| output).collect()
}

/// Returns true when the user cancelled mid-gate so the turn should stop
/// before more tools run (matches the serial tool path).
pub(crate) async fn execute_readonly_batch(
    ctx: &mut ReadonlyBatchCtx<'_>,
    calls: &[ToolCall],
    messages: &mut Vec<ChatMessage>,
) -> bool {
    let mut parsed: Vec<Value> = Vec::with_capacity(calls.len());
    let mut blocked: Vec<Option<tools::ToolOutcome>> = Vec::with_capacity(calls.len());
    // Per call: admitted by auto without the content question being asked.
    let mut unasked: Vec<bool> = Vec::with_capacity(calls.len());
    for call in calls {
        let name = call.function.name.as_str();
        // batchable_call already validated the JSON; Null (impossible) would
        // just surface as a missing-argument tool error.
        let args: Value = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
        parsed.push(args.clone());
        ctx.core.send_agent(ctx.session_id, AgentEvent::ToolStart {
            call_id: call.id.clone(),
            name: name.into(),
            args: args.clone(),
        });
        // hooks pre → permissions → content gate. approval_mode is skipped
        // because batchable_call excludes mutating tools.
        let mode = *ctx.policy_mode;
        let pre = ctx.hooks.pre_tool_use(ctx.session_id, name, &args, ctx.project_root, &ctx.cancelled).await;
        refresh_approval_policy_canonical(ctx.core, ctx.session_id, ctx.project_root, ctx.canonical_root, ctx.policy_mode, ctx.hooks, ctx.permissions, messages).await;
        if mode != *ctx.policy_mode && !matches!(pre, PreToolResult::Cancelled) {
            // Nothing has executed. Refuse the batch without replaying hooks,
            // which may have side effects even for a non-mutating tool.
            for (index, pending) in calls.iter().enumerate() {
                if index >= parsed.len() {
                    ctx.core.send_agent(ctx.session_id, AgentEvent::ToolStart {
                        call_id: pending.id.clone(),
                        name: pending.function.name.clone(),
                        args: serde_json::from_str(&pending.function.arguments).unwrap_or(Value::Null),
                    });
                }
                end_call(ctx.core, ctx.session_id, &pending.id, &tools::ToolOutcome::err(POLICY_CHANGED_DURING_HOOK), messages);
            }
            return false;
        }
        let block = match pre {
            PreToolResult::Block { reason } => Some(tools::ToolOutcome::err(reason)),
            PreToolResult::Cancelled => {
                // Close every ToolStart already emitted in this batch so the
                // transcript stays well-formed, then stop the turn.
                for prior in calls.iter().take(parsed.len()) {
                    ctx.core.send_agent(ctx.session_id, AgentEvent::ToolEnd {
                        call_id: prior.id.clone(),
                        ok: false,
                        output: "cancelled".to_string(),
                    });
                    messages.push(ChatMessage::tool(prior.id.clone(), "cancelled".to_string()));
                }
                return true;
            }
            PreToolResult::Allow => match ctx.permissions.evaluate(name, &args) {
                PermissionDecision::Deny { reason } => Some(tools::ToolOutcome::err(reason)),
                // Ask is excluded from batching (see batchable_call); if it
                // still lands here outside auto, refuse the batch call.
                PermissionDecision::Ask if ctx.core.approval_mode_canonical(ctx.canonical_root) != ApprovalMode::Auto => Some(
                    tools::ToolOutcome::err("permission rule requires approval; re-run outside a concurrent batch"),
                ),
                // Allow/Default: readonly batch tools are non-mutating.
                PermissionDecision::Allow | PermissionDecision::Default | PermissionDecision::Ask => None,
            },
        };
        // Unapproved content is excluded from batching (see batchable_call) so
        // the serial path can prompt; if one still lands here, block rather
        // than run unapproved host code outside auto. The sites share one
        // predicate, so the batch path cannot drift away from the gate again.
        let auto = ctx.core.approval_mode_canonical(ctx.canonical_root) == ApprovalMode::Auto;
        let block = block.or_else(|| {
            if auto { return None; }
            unapproved_capability(ctx.registry, &ctx.core.data_dir, ctx.project_root, name)
                .map(|source| tools::ToolOutcome::err(declined_message(Some(&source))))
        });
        unasked.push(auto && block.is_none());
        blocked.push(block);
    }

    let core = ctx.core;
    let data_dir = &ctx.core.data_dir;
    let canonical_root = ctx.canonical_root;
    let futures: Vec<_> = calls
        .iter()
        .zip(parsed.iter())
        .zip(blocked.iter())
        .zip(unasked.iter().copied())
        .map(|(((call, args), block), unasked)| {
            let name = call.function.name.clone();
            let args = args.clone();
            let root = ctx.project_root.to_path_buf();
            let cancel = ctx.cancelled.clone();
            let registry = ctx.registry.clone();
            let caps = ctx.caps;
            let blocked_outcome = block.clone();
            async move {
                if let Some(outcome) = blocked_outcome {
                    return (outcome, false);
                }
                // Calls past the parallelism cap start only as earlier ones
                // end, long after admission. Auto admitted this one without
                // asking about its content; if the mode has left auto since,
                // here or in another frontend that saved it, unapproved
                // content must not start on that admission, or it runs under
                // ask with no card. Read the store back, then the live mode,
                // as the serial path does right before each call.
                if unasked {
                    core.refresh_saved_approval_modes();
                    if core.approval_mode_canonical(canonical_root) != ApprovalMode::Auto
                        && unapproved_capability(&registry, data_dir, &root, &name).is_some()
                    {
                        return (tools::ToolOutcome::err(MODE_LEFT_AUTO_BEFORE_START), false);
                    }
                }
                (registry.execute(&name, &args, data_dir, &root, caps, cancel).await, true)
            }
        })
        .collect();
    let outcomes = collect_bounded(futures, ctx.parallelism).await;

    for (i, call) in calls.iter().enumerate() {
        let (outcome, ran) = &outcomes[i];
        let name = call.function.name.as_str();
        let args = &parsed[i];
        if *ran {
            count_usage(ctx.usage, ctx.registry, ctx.project_root, name, args, outcome.ok);
            let failures = ctx
                .hooks
                .post_tool_use(
                    ctx.session_id,
                    name,
                    args,
                    ctx.project_root,
                    outcome,
                    &ctx.cancelled,
                )
                .await;
            report_hook_failures(ctx.core, ctx.session_id, failures);
        }
        end_call(ctx.core, ctx.session_id, &call.id, outcome, messages);
    }
    false
}
