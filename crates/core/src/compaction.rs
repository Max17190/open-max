//! Keeping the transcript inside the model's context window. Once the
//! trigger is crossed, one cut removes everything between the opening (the
//! system prompt through the user's original request, see `pinned_head`) and
//! the newest exchanges, which stay byte for byte up to `cut_target`. Before
//! the cut the session's own model writes the notes it will continue from, in
//! a request whose prompt is the transcript up to the cut plus one ask
//! (`NOTES_ASK`); the notes take the removed span's place in one context note
//! that names the archive holding every byte the cut removed. Without notes
//! the history stays whole unless the window forces the cut, which then leaves
//! a deterministic digest. A request costs its history plus
//! `request_overhead`, in estimated tokens the server's own counts calibrate
//! (`token_factor`).
//!
//! Nothing kept is ever rewritten. The server-side prompt cache re-prefills
//! from the first byte that diverges, so between cuts the transcript stays
//! append-only and the cache warm.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::client::{ChatClient, CompletionResult, TRUNCATED};
use crate::hooks::Hooks;
use crate::notes::report_hook_failures;
use crate::sessions;
use crate::state::{CancelToken, Core};
use crate::transcript::save_messages;
use crate::types::{estimate_tokens, AgentEvent, ChatMessage};

pub(crate) const DIGEST_PREFIX: &str = "[context note:";

/// Structured fields survive any number of prunes verbatim (absorb_prior), so
/// they are bounded: fresh drops may record up to the fresh cap, carry-forward
/// fills to the total cap, oldest carried entries dropped first.
const MAX_DIGEST_PATHS_FRESH: usize = 8;
const MAX_DIGEST_PATHS: usize = 12;
const MAX_DIGEST_TOOLS: usize = 16;
/// Per-entry caps on what those fields accept, enforced at record and absorb
/// time: anything longer is not a real path or tool name (the registry caps
/// names at 64 ASCII chars), and one pathological call must not blow the
/// note past `DIGEST_NOTE_ALLOWANCE_TOKENS`, which is derived from these
/// caps. Denominated in BYTES, like every cap feeding the note: the token
/// estimator is bytes/4, so a char-denominated cap would let multibyte
/// fields cost up to four times what the allowance reserves.
const MAX_DIGEST_PATH_BYTES: usize = 256;
const MAX_DIGEST_TOOL_BYTES: usize = 64;
/// Byte cap on one recorded user-goal snippet, same denomination as above.
pub(crate) const MAX_DIGEST_SNIPPET_BYTES: usize = 120;

/// Byte-capped take on a char boundary. Every note field, the notes, and the
/// arguments a tool error quotes are budgeted in bytes (the
/// estimator is bytes/4), and a plain byte slice could split a multibyte
/// char; this is the one cut they all use.
pub(crate) fn take_note_bytes(s: &str, max_bytes: usize) -> String {
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if i + c.len_utf8() > max_bytes {
            break;
        }
        end = i + c.len_utf8();
    }
    s[..end].to_string()
}

#[derive(Default)]
pub(crate) struct CompactionDigest {
    message_count: usize,
    tools: BTreeSet<String>,
    paths: Vec<String>,
    /// Short snippets of compacted user goals (not the original first request).
    user_snippets: Vec<String>,
    /// Every message as dropped, in transcript order, held for the notes
    /// request and the archive append.
    dropped: Vec<ChatMessage>,
}

impl CompactionDigest {
    fn record_message(&mut self, msg: &ChatMessage) {
        self.message_count += 1;
        self.dropped.push(msg.clone());
        if msg.role == "user" {
            if let Some(c) = msg.content.as_deref() {
                let trimmed = c.trim();
                if !trimmed.is_empty()
                    && !trimmed.starts_with(DIGEST_PREFIX)
                    && self.user_snippets.len() < 4
                {
                    let snippet = take_note_bytes(trimmed, MAX_DIGEST_SNIPPET_BYTES);
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
            if !call.function.name.is_empty()
                && call.function.name.len() <= MAX_DIGEST_TOOL_BYTES
            {
                self.tools.insert(call.function.name.clone());
            }
            if let Ok(v) = serde_json::from_str::<Value>(&call.function.arguments) {
                if let Some(path) = v.get("path").and_then(|p| p.as_str()) {
                    if path.len() <= MAX_DIGEST_PATH_BYTES
                        && self.paths.len() < MAX_DIGEST_PATHS_FRESH
                        && !self.paths.iter().any(|p| p == path)
                    {
                        self.paths.push(path.to_string());
                    }
                }
            }
        }
    }

    /// Deterministic carry-forward: a later prune drops the previous digest
    /// note, and its prose would otherwise be the only carrier of the paths
    /// and tools it condensed. Unioning the structured fields from the
    /// previous compaction record keeps addresses intact across any number of
    /// prunes by code, not through the summarizer, which paraphrases.
    fn absorb_prior(&mut self, prior: &sessions::CompactionRecord) {
        for tool in &prior.tools {
            if self.tools.len() >= MAX_DIGEST_TOOLS {
                break;
            }
            // Length-capped like fresh entries: records written before the
            // caps existed must not smuggle oversized fields into new notes.
            if tool.len() <= MAX_DIGEST_TOOL_BYTES {
                self.tools.insert(tool.clone());
            }
        }
        for path in &prior.paths {
            if self.paths.len() >= MAX_DIGEST_PATHS {
                break;
            }
            if path.len() <= MAX_DIGEST_PATH_BYTES
                && !self.paths.iter().any(|p| p == path)
            {
                self.paths.push(path.clone());
            }
        }
    }

    /// The note a cut leaves. The model's own notes take the place of the
    /// tool and goal lists; exact paths stay listed because notes paraphrase
    /// them.
    fn format(&self, summary: Option<&str>, archive: Option<&str>) -> String {
        let mut parts = vec![format!(
            "{DIGEST_PREFIX} {} earlier messages were compacted.",
            self.message_count
        )];
        if let Some(summary) = summary {
            parts.push(format!("Summary: {summary}"));
        } else if !self.tools.is_empty() {
            parts.push(format!(
                "Tools used: {}.",
                self.tools.iter().cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        if !self.paths.is_empty() {
            parts.push(format!("Files touched: {}.", self.paths.join(", ")));
        }
        if summary.is_none() && !self.user_snippets.is_empty() {
            parts.push(format!(
                "Earlier goals: {}.",
                self.user_snippets.join(" | ")
            ));
        }
        if let Some(path) = archive {
            parts.push(format!("Full dropped messages: {path} (bash: grep or tail it)."));
        }
        parts.push("The messages after this note are unchanged and continue from it; files on disk are as the work so far left them.".into());
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

const SUMMARY_TIMEOUT: Duration = Duration::from_secs(90);
const MAX_SUMMARY_BYTES: usize = 4_000;

/// The ask that ends the notes request, after the transcript up to the cut.
pub(crate) const NOTES_ASK: &str = "[context note: everything above this message except the original request is about to be removed from the conversation to keep requests small. Write the notes you will continue from. Reply with the notes alone: plain text, no tool call (a call in this reply does not run), at most 400 words, in four labeled parts. State: what is finished and verified, what is half done, and the next step. Files: every path changed or still needed, with the names and places in it that matter. Facts: commands that work in this project, each failure seen with its cause, and every instruction the user gave after the original request, in the user's words; copy identifiers, paths and error text exactly. Open: what is unresolved or unverified. If a context note appears above, carry forward what still holds.]";

/// The notes request's messages: `head`, every message the cut removes, and
/// the ask. The ask is stamped with the effort in force, as a turn's prompt
/// is, so where a turn changed the effort and its prompt stays after the cut,
/// the change rides just ahead of the ask and the effort every earlier request
/// named at its top stays there: what they sent stays a cached prefix.
pub(crate) fn notes_messages(client: &ChatClient, head: &[ChatMessage], dropped: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut messages = head.to_vec();
    messages.extend(dropped.iter().cloned());
    messages.push(ChatMessage { effort: Some(client.effort_stamp()), ..ChatMessage::user(NOTES_ASK) });
    messages
}

/// The notes the session's own model writes before a cut. The request is the
/// transcript up to the cut (`head`, then every message the cut removes, in
/// order) and `NOTES_ASK`, with the turn's tools: a prompt the session
/// already sent, so a prompt cache can serve all of it but the ask. Only the
/// reply's text is taken. A call in it never runs, nothing of this exchange
/// is stored or shown, and an empty reply, an error, a timeout, a cancel, a
/// cut stream or a request the spend ceiling refuses gives no notes (None).
///
/// The request is billed like any other: a ledger that advertises what each
/// request cost cannot omit the one the harness makes on its own behalf.
async fn summarize_compaction(
    ctx: &CompactionCtx<'_>,
    head: &[ChatMessage],
    digest: &CompactionDigest,
    spend: &mut RequestSpend,
) -> Option<String> {
    let CompactionCtx { core, session_id, client, cancelled, schemas, factor, .. } = *ctx;
    let messages = notes_messages(client, head, &digest.dropped);
    // In real tokens by the session's learned factor, as every request is admitted.
    let estimated = to_real(messages.iter().map(ChatMessage::estimated_tokens).sum::<usize>() + request_overhead(schemas), factor);
    if !spend.admits(estimated) { return None; }
    let mut resent = Vec::new();
    let outcome = tokio::time::timeout(SUMMARY_TIMEOUT, client.stream_chat(&messages, client.reply_reserve, schemas, cancelled.clone(), |delta| {
        if let crate::client::StreamDelta::Usage(usage) = delta { resent.push(usage); }
    })).await;
    // A reply the client asked for again was paid for, however this one ended.
    for usage in resent { spend.charge(core, session_id, usage); }
    let result = match outcome {
        Ok(Ok(result)) => result,
        _ => {
            // Without a completed response, charge the reply reserve it asked for.
            spend.tokens = spend.tokens.saturating_add(estimated.saturating_add(client.reply_reserve));
            return None;
        }
    };
    spend.record(core, session_id, estimated, &result);
    if result.finish_reason == TRUNCATED || result.finish_reason == "cancelled" { return None; }
    let mut summary = result.content;
    if let Some(clean) = crate::client::strip_leading_think(&summary) {
        summary = clean;
    }
    let summary = summary.trim().replace(['\n', '\r'], " ");
    if summary.is_empty() {
        return None;
    }
    if summary.len() > MAX_SUMMARY_BYTES {
        return Some(take_note_bytes(&summary, MAX_SUMMARY_BYTES) + "…");
    }
    Some(summary)
}

pub(crate) fn is_digest_message(msg: &ChatMessage) -> bool {
    msg.role == "user"
        && msg.content.as_deref().is_some_and(|c| c.starts_with(DIGEST_PREFIX))
}

/// A note an earlier release's truncation-only prune wrote, told apart from
/// a real digest by its fixed opening. Saved transcripts carry this wording,
/// so it must keep matching them.
fn is_truncation_only_note(msg: &ChatMessage) -> bool {
    is_digest_message(msg)
        && msg.content.as_deref().and_then(|c| c.strip_prefix(DIGEST_PREFIX)).is_some_and(|rest| {
            rest.trim_start_matches(|c: char| c == ' ' || c.is_ascii_digit())
                .starts_with("older tool outputs were shortened in place")
        })
}

/// One admission counter for every model request made during a turn.
pub(crate) struct RequestSpend {
    tokens: usize,
    limit: Option<usize>,
}

impl RequestSpend {
    pub(crate) fn new(limit: Option<usize>) -> Self { Self { tokens: 0, limit } }

    pub(crate) fn admits(&self, estimated: usize) -> bool {
        self.limit.is_none_or(|cap| self.tokens.saturating_add(estimated) <= cap)
    }

    pub(crate) fn record(&mut self, core: &Core, session_id: &str, estimated: usize, result: &CompletionResult) {
        match result.usage {
            Some(u) => self.charge(core, session_id, u),
            None => self.tokens = self.tokens.saturating_add(estimated.saturating_add(estimate_tokens(result.content.len()
                + result.tool_calls.iter().map(|c| c.function.arguments.len()).sum::<usize>()))),
        }
    }

    /// Charge what the provider reported for one reply, this request's or
    /// one the client asked for again: saved, shown, and spent.
    pub(crate) fn charge(&mut self, core: &Core, session_id: &str, u: crate::client::Usage) {
        sessions::append_usage(core, session_id, &sessions::TokenUsage {
            ts: sessions::unix_now(),
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.output_tokens,
            cached_tokens: u.cached_tokens,
        });
        core.send_agent(session_id, AgentEvent::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.output_tokens,
            cached_tokens: u.cached_tokens,
        });
        self.tokens = self.tokens.saturating_add(u.prompt_tokens.saturating_add(u.output_tokens) as usize);
    }
}

/// What settling a compaction digest needs from its caller. Borrowed whole,
/// like `batch::ReadonlyBatchCtx`, so the budget path and the forced `/compact`
/// share one settlement instead of drifting copies.
pub(crate) struct CompactionCtx<'a> {
    pub(crate) core: &'a Arc<Core>,
    pub(crate) session_id: &'a str,
    pub(crate) project_root: &'a Path,
    pub(crate) client: &'a ChatClient,
    pub(crate) estimate: &'a (dyn Fn(&ChatMessage) -> usize + Sync),
    pub(crate) hooks: &'a Hooks,
    pub(crate) cancelled: &'a Arc<CancelToken>,
    /// The turn's tool schemas, as every request of the turn sends them.
    pub(crate) schemas: &'a str,
    /// The session's learned real tokens per estimated token (`token_factor`),
    /// which the notes request is admitted and charged by, as every request is.
    pub(crate) factor: f64,
    /// The most a cut keeps of the newest exchanges (`requested_keep`).
    pub(crate) keep: usize,
}

/// Ask for the notes and settle the candidate's digest note: the archive
/// material and the record, which are emitted only after the candidate
/// transcript has been persisted. Without notes a cut the window does not
/// force is not worth making, so this returns None and nothing is committed.
async fn prepare_compaction_digest(
    ctx: &CompactionCtx<'_>,
    messages: &mut [ChatMessage],
    mut digest: CompactionDigest,
    forced: bool,
    spend: &mut RequestSpend,
) -> Option<(Vec<ChatMessage>, sessions::CompactionRecord)> {
    let CompactionCtx { core, session_id, .. } = *ctx;
    // The cut's note sits right after the pinned head and ends the opening
    // run, so the head of the cut candidate is its slot.
    let slot = pinned_head(messages);
    let summary = summarize_compaction(ctx, &messages[..slot], &digest, spend).await;
    if summary.is_none() && !forced {
        return None;
    }
    // Structured fields from the previous record carry forward by
    // code: the cut may have dropped the old digest note, whose
    // prose is lossy about the paths and tools it condensed.
    if let Some(prior) = sessions::last_compaction(core, session_id) {
        digest.absorb_prior(&prior);
    }
    let path = sessions::archive_display(core, session_id);
    // The note in the slot was just inserted by the cut, so replacing it
    // here keeps one digest message.
    let note = digest.format(summary.as_deref(), Some(&path));
    if messages.len() > slot && is_digest_message(&messages[slot]) {
        messages[slot] = ChatMessage::user(note.clone());
    }
    let record = digest.to_record(note);
    Some((digest.dropped, record))
}

/// A failed archive or transcript write must not discard the live history.
/// Clone only when a prune is needed; ordinary requests keep their owned vec.
pub(crate) async fn compact_messages(
    ctx: &CompactionCtx<'_>,
    messages: &mut Vec<ChatMessage>,
    trigger: usize,
    budget: usize,
    schema_tokens: usize,
    force: bool,
    spend: &mut RequestSpend,
) -> Result<usize, String> {
    let total = schema_tokens + messages.iter().map(ctx.estimate).sum::<usize>();
    if (!force && total <= trigger) || schemas_exceed_budget(trigger, schema_tokens) {
        return Ok(0);
    }
    if let Some(reason) = sessions::index_diagnostic(ctx.core) {
        return Err(format!("compaction refused: {reason}"));
    }
    // The head the prune pins, and so the index its digest note takes.
    let head = pinned_head(messages);
    // A prune that drops or shortens the stored note stores it again after
    // it, so the prune keeps room for it.
    let note = ctx.core.sessions.lock().await.get(ctx.session_id).and_then(|data| data.trailing_note.clone());
    let room = note.as_deref().map_or(0, |note| estimate_tokens(crate::prompt::NOTE_SEPARATOR.len() + note.len()));
    // Once the archive lands the digest note gains its address, the paths the
    // last record carries forward, and either a summary at its cap or the
    // carried tools (a summary replaces the tool list), each field within
    // `absorb_prior`'s caps; 100 bytes cover the labels, separators aside,
    // and the estimate's rounding. The prune keeps that much room too.
    let carried = |fields: &[String], count: usize, cap: usize| {
        fields.iter().filter(|field| field.len() <= cap).map(|field| field.len() + 2).sum::<usize>().min(count * (cap + 2))
    };
    let (paths, tools) = sessions::last_compaction(ctx.core, ctx.session_id).map_or((0, 0), |prior| {
        (carried(&prior.paths, MAX_DIGEST_PATHS, MAX_DIGEST_PATH_BYTES), carried(&prior.tools, MAX_DIGEST_TOOLS, MAX_DIGEST_TOOL_BYTES))
    });
    let address = sessions::archive_display(ctx.core, ctx.session_id).len();
    let growth = estimate_tokens(address + paths + tools.max(MAX_SUMMARY_BYTES) + 100);
    let mut candidate = messages.clone();
    let digest = if force {
        prune_transcript(&mut candidate, trigger, ctx.keep, schema_tokens + room, total + room, growth, ctx.estimate)
    } else {
        enforce_budget(&mut candidate, trigger, ctx.keep, schema_tokens + room, growth, ctx.estimate)
    };
    let Some(digest) = digest else { return Ok(0) };
    // A hydration repair still waiting for a save lands first, with its
    // boundary edits: the commit moves the boundaries as if they already
    // measured this transcript, and once it lands the edits no longer fit.
    // So the prune waits while either one fails to land.
    let pending = || async {
        ctx.core.sessions.lock().await
            .get(ctx.session_id)
            .is_some_and(|data| !data.deferred_resume_edits.is_empty())
    };
    if pending().await
        && (!save_messages(ctx.core, ctx.session_id, messages, true).await || pending().await)
    {
        return Err("compaction refused: the session transcript or its replay boundaries could not be saved".into());
    }
    let dropped = digest.message_count;
    // A cut without notes is made only when the window forces it or the
    // user asked for it; it then leaves the deterministic digest.
    let forced = force || total > budget;
    let Some((archive, record)) = prepare_compaction_digest(ctx, &mut candidate, digest, forced, spend).await else {
        return Ok(0);
    };
    crate::notes::store_note(&mut candidate, note.as_deref());
    sessions::commit_compaction(ctx.core, ctx.session_id, &candidate, &archive, Some(&record), messages.len(), head)?;
    if let Some(data) = ctx.core.sessions.lock().await.get_mut(ctx.session_id) {
        data.persisted_count = candidate.len();
    }
    *messages = candidate;
    if let Ok(value) = serde_json::to_value(&record) {
        let failures = ctx.hooks.compaction(ctx.session_id, ctx.project_root, &value, ctx.cancelled).await;
        report_hook_failures(ctx.core, ctx.session_id, failures);
    }
    Ok(dropped)
}

/// What every request carries besides the history, in estimated tokens: the
/// tool schemas, re-sent whole each time. The note is history, priced with
/// the message it is stored in. The turn loop and `/compact` both price a
/// request here, so both count the same bytes.
pub(crate) fn request_overhead(schemas_wire: &str) -> usize {
    estimate_tokens(schemas_wire.len())
}

/// The share of the budget, in percent, past which the frozen schemas leave
/// a session degraded (`schemas_outgrow_budget`).
const PRUNE_TARGET_PCT: usize = 70;

/// The most a cut keeps of the newest exchanges, its note included, in
/// estimator units: enough for the work in flight, small enough that the
/// requests after a cut are cheap again. The `compaction_keep_tokens`
/// setting replaces it (`requested_keep`).
pub(crate) const KEEP_RECENT_TOKENS: usize = 20_000;

/// How many leading messages a prune keeps: the system prompt and the
/// conversation's opening, through the user's original request. Turn-start
/// notes (the execution policy, a refreeze receipt, a policy notice) are
/// inserted ahead of the prompt they accompany, so a first turn opens
/// `[system, note, request]` and the request ends the opening run of user
/// messages rather than sitting at index 1. Pinning index 1 alone kept the
/// note forever and dropped the request first. The run ends at the first
/// model reply or at a previous prune's digest note, which sits right after
/// the head. Never below 2, the floor every transcript was already pruned
/// with, so one that opens `[system, request]` prunes exactly as before.
///
/// One saved shape needs an exception. Release 2026.10.0 inserted a
/// truncation-only prune's note at a fixed index 2, so a note-first session
/// it shortened opens `[system, note, truncation note, request]`, the
/// request still verbatim. Ending the run at that note dropped the request
/// on the next prune, so the run continues past a truncation-only note when
/// a plain user message follows it. A note written at the head instead
/// always precedes a model reply, so it still ends the run.
fn pinned_head(messages: &[ChatMessage]) -> usize {
    let plain_user = |m: &ChatMessage| m.role == "user" && !is_digest_message(m);
    let mut head = 1;
    while let Some(m) = messages.get(head) {
        let saved_ahead_of_request =
            is_truncation_only_note(m) && messages.get(head + 1).is_some_and(plain_user);
        if !plain_user(m) && !saved_ahead_of_request {
            break;
        }
        head += 1;
    }
    head.max(2)
}

/// Keep the last three messages, plus the start of a tool exchange if
/// the boundary would otherwise cut through its results.
fn protected_tail_start(messages: &[ChatMessage], head: usize) -> usize {
    let tail = messages.len().saturating_sub(3).max(head);
    if messages.get(tail).is_none_or(|msg| msg.role != "tool") {
        return tail;
    }
    (head..tail).rev().find(|&index| messages[index].role != "tool")
        .filter(|&index| messages[index].role == "assistant" && messages[index].tool_calls.is_some())
        .unwrap_or(tail)
}

/// The total (schemas plus transcript) one cut aims for: the schemas, the
/// pinned head, and the newest exchanges with the note in their place, at
/// most a quarter of `budget` and at most `keep` (`requested_keep`), and never
/// past `budget`, which a cut must reach however the schemas and the opening
/// crowd it. A quarter leaves a gap so the turns after a cut append.
pub(crate) fn cut_target(
    budget: usize,
    keep: usize,
    schema_tokens: usize,
    messages: &[ChatMessage],
    estimate: &dyn Fn(&ChatMessage) -> usize,
) -> usize {
    let head = pinned_head(messages).min(messages.len());
    (schema_tokens + messages[..head].iter().map(estimate).sum::<usize>() + (budget / 4).min(keep)).min(budget)
}

/// The prompt a cut puts back after its note for the turn in progress: the
/// newest one it removed (only prompts carry an effort stamp), if none is kept.
fn restored_prompt<'a>(kept: &[ChatMessage], dropped: &'a [ChatMessage]) -> Option<&'a ChatMessage> {
    if kept.iter().any(|m| m.effort.is_some()) { None } else { dropped.iter().rev().find(|m| m.effort.is_some()) }
}

/// True when the frozen schemas fill the window on their own. The schema cost
/// is constant per request, so here no transcript fits, not even an empty
/// one. Pruning is pure loss: it would drop history to the floor, emit a
/// digest, pay a summarization request, and still be over, every single turn.
pub(crate) fn schemas_exceed_budget(budget: usize, schema_tokens: usize) -> bool {
    schema_tokens >= budget
}

/// True when the frozen schemas reach PRUNE_TARGET_PCT of the budget. Below
/// the window they still leave room for a transcript, so compaction keeps
/// working; what is gone is the room a cut leaves for the turns after it.
/// Either way the session is degraded and says so once.
pub(crate) fn schemas_outgrow_budget(budget: usize, schema_tokens: usize) -> bool {
    schema_tokens >= budget * PRUNE_TARGET_PCT / 100
}

/// Floor for the `compaction_tokens` setting. A cut keeps at most a quarter
/// of the trigger, so the gap after it is most of the trigger; below this
/// floor the gap is too small to buy append-only turns, and the session would
/// cut again (and pay another notes request) every few iterations. In the
/// setting's own real tokens, like the conversation the gap has to hold; see
/// `requested_trigger`.
const COMPACTION_TOKENS_FLOOR: usize = 20_000;

/// The trigger a `compaction_tokens` setting asks for, in estimator units:
/// floored in the setting's real tokens, then converted, then capped by the
/// budget. The observed ratio can relax this soft trigger within the window.
pub(crate) fn requested_trigger(requested: usize, ratio: Option<f64>, budget: usize) -> usize {
    to_estimate(requested.max(COMPACTION_TOKENS_FLOOR), setting_factor(ratio)).min(budget)
}

/// Real tokens per estimated token for a compaction setting: the observed
/// ratio, which may relax a setting as well as tighten it, up to the cap.
fn setting_factor(ratio: Option<f64>) -> f64 {
    ratio.filter(|r| r.is_finite() && *r > 0.0).map_or(1.0, |r| r.min(MAX_TOKEN_FACTOR))
}

/// Floor for the `compaction_keep_tokens` setting: the digest note at its cap
/// (`DIGEST_NOTE_ALLOWANCE_TOKENS`) and about as much again of the newest work.
/// Below it a cut cannot keep the note and the exchange in flight, so its
/// trigger would fall back to the window budget on nearly every request.
const COMPACTION_KEEP_FLOOR: usize = 4_000;

/// The most a cut keeps of the newest exchanges, in estimator units: the
/// `compaction_keep_tokens` setting, floored in its real tokens and then
/// converted like the trigger, or `KEEP_RECENT_TOKENS` when it is unset. A
/// cut still keeps at most a quarter of its trigger (`cut_target`). At a high
/// learned ratio the converted value can fall below the note's allowance; the
/// cut then ends at the drop floor, and `compaction_trigger` still fires it.
pub(crate) fn requested_keep(requested: Option<usize>, ratio: Option<f64>) -> usize {
    requested.map_or(KEEP_RECENT_TOKENS, |keep| to_estimate(keep.max(COMPACTION_KEEP_FLOOR), setting_factor(ratio)))
}

/// What the digest note itself may cost after a cut, in tokens. An upper
/// bound, not a guess: the prefix and count, the notes at their 4,000-byte
/// cap (or, without notes, sixteen tool names of at most 64 bytes and four
/// user snippets of 120), twelve paths of at most 256, the archive address,
/// and the closing sentence. Every cap is in bytes because the estimator is
/// bytes/4. The gate test formats a digest at every cap and asserts it fits,
/// so this constant and the format cannot drift apart. Counted into the
/// irreducible floor below, so a trigger is only honored when its target has
/// room for the note too.
const DIGEST_NOTE_ALLOWANCE_TOKENS: usize = 1_900;

/// The token total at which compaction fires. The resolved ceiling can only
/// lower the window-derived `budget`, which keeps requests within the
/// endpoint's context window. Two unachievable shapes fall back to the budget
/// rather than leaving the loop stuck: a setting the frozen schemas outgrow (pruning
/// cannot get under it, and the futility guard would disarm compaction while
/// the transcript grows toward the window), and a setting whose target the
/// irreducible transcript outgrows; a prune would end at the floor still
/// over the target and re-fire on every iteration, paying an archive and a
/// notes request each time. That check measures the target at the larger of
/// `keep` and `KEEP_RECENT_TOKENS`: a smaller `keep` asks for a smaller cut,
/// and a cut that ends at the floor above it still leaves the gap the default
/// target leaves, so it does not re-fire. Measured against `keep` itself, a
/// small setting fell back whenever the newest tool output was large, and
/// the session ran to the window uncut.
///
/// Irreducible means exactly what a maximal prune leaves: the drop loop's
/// length floor ends at the pinned head, the digest note, the turn's prompt
/// it puts back, and the newest three messages, including any tool exchange
/// that crosses that boundary. Those are what the target must contain.
/// Counting more would fall back while a configured prune could succeed,
/// silently disabling the setting.
///
/// `ceiling` is in real tokens. The budget, schema cost, and trigger use
/// estimator units so the window can cap the trigger independently.
pub(crate) fn compaction_trigger(
    budget: usize,
    schema_tokens: usize,
    ceiling: usize,
    keep: usize,
    ratio: Option<f64>,
    messages: &[ChatMessage],
    estimate: &dyn Fn(&ChatMessage) -> usize,
) -> usize {
    let trigger = requested_trigger(ceiling, ratio, budget);
    if trigger == budget || schemas_outgrow_budget(trigger, schema_tokens) {
        return budget;
    }
    let head = pinned_head(messages).min(messages.len());
    let tail = protected_tail_start(messages, head);
    let irreducible: usize = schema_tokens
        + DIGEST_NOTE_ALLOWANCE_TOKENS
        + messages[..head]
            .iter()
            .chain(&messages[tail..])
            .chain(restored_prompt(&messages[tail..], &messages[head..tail]))
            .map(estimate)
            .sum::<usize>();
    if cut_target(trigger, keep.max(KEEP_RECENT_TOKENS), schema_tokens, messages, estimate) < irreducible {
        budget
    } else {
        trigger
    }
}

/// The context joined to a provider error when the request that failed was
/// already over budget by local accounting before it went out. A request can
/// leave here oversized on purpose: `context_tokens` is the user's setting,
/// not ground truth about the endpoint, so refusing locally would brick
/// sessions whenever it is stale or conservative for requests the provider
/// accepts. The price of sending anyway is owed here, where the bet failed:
/// a raw provider error names nothing the user can act on, and the once-per-
/// session advisory may be hundreds of turns gone. Phrased as context, not
/// diagnosis - a dead network also lands in this branch.
pub(crate) fn over_budget_error_context(
    used_tokens: usize,
    schema_tokens: usize,
    budget: usize,
    context_tokens: usize,
) -> String {
    format!(
        "this request was over budget before it was sent: ~{used_tokens} tokens ({schema_tokens} of them frozen tool schemas, re-sent every request) against a send budget of {budget} (context_tokens {context_tokens} minus response headroom). if the provider refused it for length: uninstall tools (`openmax --spec usage` ranks what each costs) and /reload, or raise context_tokens to what the endpoint really serves"
    )
}

/// Calibrate against the reported size of the same prompt. A ratio below
/// one needs a complete response with usable usage. If the response cannot
/// calibrate, keep any prior ratio that tightened the budget.
pub(crate) fn updated_prompt_ratio(
    previous: Option<f64>,
    result: &CompletionResult,
    estimated: usize,
) -> Option<f64> {
    let complete = !matches!(result.finish_reason.as_str(), TRUNCATED | "cancelled");
    result.usage
        .filter(|usage| usage.prompt_tokens > 0 && estimated > 0)
        .and_then(|usage| {
            let ratio = usage.prompt_tokens as f64 / estimated as f64;
            let cache_count_fits = usage.cached_tokens.is_none_or(|cached| cached <= usage.prompt_tokens);
            (ratio >= 1.0 || (complete && cache_count_fits)).then_some(ratio)
        })
        .or(previous.filter(|&ratio| ratio >= 1.0))
}

/// The ratio a refusal for length proves, when it proves more than `factor`
/// (the one the request was budgeted with). The provider refused a request
/// estimated at `estimated` because it did not fit what the window leaves
/// beside the reply reserve, `fits` real tokens, so the real ratio is at
/// least `fits / estimated`. Budgeting the next request with that floor makes
/// compaction fire where the plain estimate never would: a transcript the
/// estimate puts under the budget, but which the provider counts over the
/// window, is otherwise refused on every turn. A floor past
/// [`MAX_TOKEN_FACTOR`] is not an estimation error (no tokenizer is that far
/// from bytes/4), so it teaches nothing, and neither does a refusal that
/// does not read as one for length.
pub(crate) fn refusal_ratio(message: &str, estimated: usize, fits: usize, factor: f64) -> Option<f64> {
    if estimated == 0 || !refused_for_length(message) {
        return None;
    }
    let floor = fits as f64 / estimated as f64;
    (floor > factor && floor <= MAX_TOKEN_FACTOR).then_some(floor)
}

/// Whether a provider error reads as a refusal for length: the prompt, or
/// the prompt plus the reply reserve, is more than the model's window. The
/// wording differs by server, so this matches the phrases the common ones
/// use; one it misses keeps today's behavior. An error status keeps only the
/// body's message, not its code, so a message that names no window still
/// needs a phrase of its own.
pub(crate) fn refused_for_length(message: &str) -> bool {
    const PHRASES: [&str; 10] = [
        "context length",
        "context_length",
        "context size",
        "context window",
        "context limit",
        "prompt is too long",
        "maximum model length",
        "maximum prompt length",
        "input token count",
        "reduce the length of the messages",
    ];
    let message = message.to_ascii_lowercase();
    PHRASES.iter().any(|phrase| message.contains(phrase))
}

/// The session's last observed ratio, if `origin` is the route that taught
/// it. A ratio measured on another route says nothing about this one.
pub(crate) fn route_ratio(stored: Option<&(String, f64)>, origin: &str) -> Option<f64> {
    stored.filter(|(taught_by, _)| taught_by == origin).map(|&(_, ratio)| ratio)
}

/// How many real tokens one estimated token is worth on this endpoint: the
/// last observed ratio (or the floor a refusal for length proved, see
/// `refusal_ratio`), or 1.0 (the plain estimate) before there is one. It
/// only ever tightens. Below 1.0 the estimator over-counts, which is today's
/// behavior and stays as is; a server that counts only the uncached part of
/// the prompt lands there too, and must not loosen the budget past the
/// window. Above 4.0 the report is not a tokenizer's: none emits much more
/// than one token per byte, so a bytes/4 estimate is short by at most ~4x
/// plus template overhead, and the cap keeps a gateway reporting nonsense
/// from collapsing the budget.
pub(crate) fn token_factor(ratio: Option<f64>) -> f64 {
    ratio.map_or(1.0, |ratio| ratio.clamp(1.0, MAX_TOKEN_FACTOR))
}

/// The most real tokens one estimated token can be worth; see `token_factor`.
const MAX_TOKEN_FACTOR: f64 = 4.0;

/// Real tokens (the window, the setting) in the estimator units compaction
/// compares, rounded down so rounding never adds room.
pub(crate) fn to_estimate(tokens: usize, factor: f64) -> usize {
    (tokens as f64 / factor) as usize
}

/// A conservative estimate in real tokens for display and request admission.
pub(crate) fn to_real(estimate: usize, factor: f64) -> usize {
    (estimate as f64 * factor).round() as usize
}

/// `schema_tokens` is the frozen tool schema array's estimated cost: it is
/// re-sent in full on every request, so it counts against the same window as
/// the transcript. Ignoring it under-reports a zero-extension session by
/// several hundred tokens, and by more with every tool the agent writes.
///
/// Returns the digest of what the cut removed, or None when it removed
/// nothing.
pub(crate) fn enforce_budget(
    messages: &mut Vec<ChatMessage>,
    budget: usize,
    keep: usize,
    schema_tokens: usize,
    growth: usize,
    estimate: &dyn Fn(&ChatMessage) -> usize,
) -> Option<CompactionDigest> {
    let total: usize =
        schema_tokens + messages.iter().map(estimate).sum::<usize>();
    // No transcript fits under overhead this large, so do nothing rather than
    // thrash: keep the transcript (and the prompt cache) intact and let the
    // caller report the condition the user can actually act on. Short of that
    // cutting still works, so it still runs.
    if total <= budget || schemas_exceed_budget(budget, schema_tokens) {
        return None;
    }
    prune_transcript(messages, budget, keep, schema_tokens, total, growth, estimate)
}

/// The cut itself, shared by the budget path and the forced `/compact`,
/// which fires inside the gap where `enforce_budget` correctly holds still.
/// Drops whole exchanges into the digest until `cut_target` is met with
/// `growth` tokens to spare for what the digest note gains later. No message
/// it keeps is changed.
fn prune_transcript(
    messages: &mut Vec<ChatMessage>,
    budget: usize,
    keep: usize,
    schema_tokens: usize,
    mut total: usize,
    growth: usize,
    estimate: &dyn Fn(&ChatMessage) -> usize,
) -> Option<CompactionDigest> {
    let target = cut_target(budget, keep, schema_tokens, messages, estimate);
    // Fixed for the whole cut: once an exchange is dropped, later user
    // messages would join the opening run and pin themselves.
    let head = pinned_head(messages);
    let protected_start = protected_tail_start(messages, head);
    let protected = messages.len().saturating_sub(protected_start);
    let mut digest = CompactionDigest::default();
    // The slot after the head may hold an earlier cut's note.
    let has_note = messages.get(head).is_some_and(is_digest_message);
    // Drops the message at `slot` (a call with its results) into the digest
    // and returns the tokens freed. The turn's prompt frees none: it goes back
    // after the note (below), so what the cut keeps meets the target with it.
    let drop_exchange = |messages: &mut Vec<ChatMessage>, digest: &mut CompactionDigest, slot: usize| {
        let calls = messages[slot].role == "assistant" && messages[slot].tool_calls.is_some();
        let results = if calls { messages[slot + 1..].iter().take_while(|m| m.role == "tool").count() } else { 0 };
        let removed: Vec<ChatMessage> = messages.drain(slot..=slot + results).collect();
        removed.iter().map(|m| {
            digest.record_message(m);
            let restored = m.effort.is_some() && !messages[slot..].iter().any(|later| later.effort.is_some());
            if restored { 0 } else { estimate(m) }
        }).sum::<usize>()
    };
    // Drop whole exchanges starting after the pinned head. Keep tool
    // replies consistent with the assistant message that requested them.
    // Leave an existing digest alone if it is the only thing before the
    // protected tail. Otherwise this cut would only summarize itself.
    let can_drop = messages.len() > head + protected + usize::from(has_note);
    while can_drop && total > target && messages.len() > (head + 4).max(head + protected) {
        total = total.saturating_sub(drop_exchange(messages, &mut digest, head));
    }
    if digest.message_count == 0 {
        return None;
    }
    let note = ChatMessage::user(digest.format(None, None));
    if messages.len() > head && is_digest_message(&messages[head]) {
        messages[head] = note;
    } else {
        messages.insert(head, note);
    }
    // The note, and the `growth` it gains later, can push total over the
    // target; keep dropping exchanges after the digest (head + 1) so the
    // next turn stays append-only and does not re-mutate history for
    // another cut. Record dropped messages into the same digest and
    // refresh the note each time, so it stays a faithful summary of
    // everything removed and the target check counts it as it is.
    let restored = restored_prompt(&messages[head + 1..], &digest.dropped);
    total = schema_tokens + messages.iter().chain(restored).map(estimate).sum::<usize>();
    while total + growth > target && messages.len() > (head + 4).max(head + 1 + protected) {
        total = total.saturating_sub(drop_exchange(messages, &mut digest, head + 1));
        let stale = estimate(&messages[head]);
        messages[head] = ChatMessage::user(digest.format(None, None));
        total = total - stale + estimate(&messages[head]);
    }
    // The turn in progress keeps its prompt: when the cut removed it, it goes
    // back verbatim, stamp included, right after the note.
    if let Some(prompt) = restored_prompt(&messages[head + 1..], &digest.dropped) {
        messages.insert(head + 1, prompt.clone());
    }
    Some(digest)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::notes::execution_policy_note;
    use crate::config::ApprovalMode;
    use crate::types::{ToolCall, ToolCallFunction};

    pub(crate) fn msg(role: &str, len: usize) -> ChatMessage {
        ChatMessage { role: role.into(), content: Some("x".repeat(len)), tool_calls: None, tool_call_id: None, reasoning_content: None, reasoning: None, reasoning_details: None, reasoning_origin: None, effort: None, archived_tool_output: None }
    }

    /// The tool schemas the fixtures' requests carry.
    const SCHEMAS: &str = r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}]"#;

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
                extra_content: None,
            }]),
        )
    }

    /// A summary the client asks for again after a cut is charged for both
    /// replies: the usage the cut one reported and the finished one's.
    #[tokio::test]
    async fn a_summary_asked_for_again_charges_both_replies() {
        let mut messages = vec![ChatMessage::system("rules"), ChatMessage::user("Find the build result.")];
        messages.extend((0..20).map(|_| msg("assistant", 1_760)));
        let (url, served) = crate::client::tests::spawn_sse_sequence(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"Half\"}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":3}}\n\n".into(),
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"Done.\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n",
            )
            .into(),
        ]);
        let dir = std::env::temp_dir().join(format!("openmax-summary-{}", uuid::Uuid::new_v4()));
        let (core, _) = Core::new(dir.clone()).unwrap();
        let session_id = sessions::create(&core, dir.display().to_string()).unwrap().id;
        let ctx = CompactionCtx {
            core: &core,
            session_id: &session_id,
            project_root: &dir,
            client: &ChatClient::new(url, None, "fixture".into(), None, 512),
            estimate: &ChatMessage::estimated_tokens,
            hooks: &Hooks::default(),
            cancelled: &Arc::new(CancelToken::default()),
            schemas: SCHEMAS,
            factor: 1.0,
            keep: KEEP_RECENT_TOKENS,
        };
        assert!(sessions::save_messages(&core, &session_id, &messages, &mut 0, true));
        let mut spend = RequestSpend::new(None);
        assert!(compact_messages(&ctx, &mut messages, 2_000, 2_000, 0, false, &mut spend).await.unwrap() > 0);
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(*served.lock().unwrap(), 2);
        assert!(messages.iter().any(|m| m.content.as_deref().is_some_and(|c| c.contains("Summary: Done."))));
        assert_eq!(spend.tokens, 103 + 105);
    }

    #[test]
    fn budget_preserves_system_and_first_user() {
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..20 {
            messages.push(msg("assistant", 2000));
            messages.push(msg("user", 2000));
        }
        let _ = enforce_budget(&mut messages, 2000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens);
        // Floor is system + first user + digest + a short tail (post-digest
        // drops may trim one more exchange when the digest itself overshoots).
        assert!(messages.len() >= 6 && messages.len() <= 7, "len={}", messages.len());
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content.as_deref(), Some("x".repeat(400).as_str()));
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
    }

    /// Behind a note-first opening, every prune keeps `[system, note,
    /// request]` and replaces the one digest right after it. A head that ran
    /// past the digest would pin the previous digest and stack a new one per
    /// prune.
    #[test]
    fn reprunes_keep_a_note_first_opening_and_one_digest() {
        let note = execution_policy_note(ApprovalMode::Ask);
        let mut messages = vec![msg("system", 400), ChatMessage::user(note.clone()), ChatMessage::user("the task")];
        for _ in 0..2 {
            for _ in 0..20 {
                messages.push(msg("assistant", 2000));
                messages.push(msg("user", 2000));
            }
            let digest = enforce_budget(&mut messages, 2000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens);
            assert!(digest.is_some_and(|d| d.message_count > 0));
            assert_eq!(messages[1].content.as_deref(), Some(note.as_str()));
            assert_eq!(messages[2].content.as_deref(), Some("the task"));
            assert!(is_digest_message(&messages[3]), "{messages:?}");
            assert_eq!(messages.iter().filter(|m| is_digest_message(m)).count(), 1);
        }
    }

    /// Release 2026.10.0 inserted a truncation-only prune's note at a fixed
    /// index 2, so a note-first session it shortened was saved as `[system,
    /// note, truncation note, request, ...]` with the request still
    /// verbatim. Ending the head at that note dropped the request on the
    /// next prune, leaving only a clipped "Earlier goals" snippet.
    #[test]
    fn a_saved_truncation_note_ahead_of_the_request_stays_in_the_head() {
        let note = execution_policy_note(ApprovalMode::Ask);
        // The wording that release saved, which no code writes any more.
        let saved = "[context note: 1 older tool outputs were shortened in place; each ends with \"…[older tool output \
                     truncated]\". Full text: sessions/s.archive.jsonl (bash: grep or tail it). Re-issue the call if you \
                     need the rest.";
        let mut messages = vec![
            msg("system", 400),
            ChatMessage::user(note.clone()),
            ChatMessage::user(saved),
            ChatMessage::user("the task"),
        ];
        for _ in 0..2 {
            for _ in 0..20 {
                messages.push(msg("assistant", 2000));
                messages.push(msg("user", 2000));
            }
            let digest = enforce_budget(&mut messages, 2000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens);
            assert!(digest.is_some_and(|d| d.message_count > 0));
            assert_eq!(messages[1].content.as_deref(), Some(note.as_str()));
            assert_eq!(messages[2].content.as_deref(), Some(saved));
            assert_eq!(messages[3].content.as_deref(), Some("the task"), "request lost: {messages:?}");
            // One real digest, right after the request, replaced in place by
            // the second prune rather than stacked.
            let compacted = |m: &ChatMessage| {
                is_digest_message(m) && m.content.as_deref().is_some_and(|c| c.contains("earlier messages were compacted"))
            };
            assert!(compacted(&messages[4]), "{messages:?}");
            assert_eq!(messages.iter().filter(|m| compacted(m)).count(), 1);
        }
        // A real digest still ends the head when a user message follows it:
        // only the legacy truncation note is looked past.
        let real = [msg("system", 10), ChatMessage::user("task"), messages[4].clone(), ChatMessage::user("next")];
        assert_eq!(pinned_head(&real), 2);
    }

    /// The note is stored in the history, so a prune that shortens or drops
    /// the message holding it stores it again at the end, and keeps room for
    /// it: the transcript it leaves, the note restored, still meets its
    /// target.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_prune_keeps_room_for_the_note_it_stores_again() {
        use crate::state::{CancelToken, Core};

        let dir = std::env::temp_dir().join(format!("openmax-note-room-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();
        let id = sessions::create(&core, project.display().to_string()).unwrap().id;
        let settings = {
            // A refused connection: the summary falls back to the digest.
            let mut s = core.settings.lock().unwrap();
            s.base_url = "http://127.0.0.1:9".into();
            s.model = "m".into();
            s.context_tokens = Some(16384);
            s.clone()
        };
        let note = format!("{}\n\n{}", crate::prompt::NOTE_HEADER, "- wordfreq: top-n words {}\n".repeat(150));
        let mut data = crate::transcript::build_session_data(&core, &id, &project).unwrap();
        data.trailing_note = Some(Arc::from(note.as_str()));
        core.sessions.lock().await.insert(id.clone(), data);
        let mut messages = vec![msg("system", 100), msg("user", 100)];
        for i in 0..30 {
            messages.push(assistant_with_tools("read_file", &format!(r#"{{"path":"src/{i}.rs"}}"#)));
            messages.push(msg("tool", 2500));
        }
        let stored = messages[3].content.take().unwrap();
        messages[3].content = Some(format!("{stored}{}{note}", crate::prompt::NOTE_SEPARATOR));

        let endpoint = crate::providers::resolve(&settings, &core.data_dir).unwrap();
        let client = ChatClient::from_endpoint(&endpoint);
        let hooks = Hooks::discover(&project, &core.data_dir);
        let cancelled = Arc::new(CancelToken::default());
        let ctx = CompactionCtx { core: &core, session_id: &id, project_root: &project, client: &client, estimate: &ChatMessage::estimated_tokens, hooks: &hooks, cancelled: &cancelled, schemas: SCHEMAS, factor: 1.0, keep: KEEP_RECENT_TOKENS };
        let budget = 16_000;
        // At the window, so the cut is made without notes.
        compact_messages(&ctx, &mut messages, budget, budget, 0, false, &mut RequestSpend::new(None)).await.unwrap();
        let carried: Vec<&str> = messages.iter().filter_map(|m| m.content.as_deref()).filter(|c| c.contains(crate::prompt::NOTE_HEADER)).collect();
        assert_eq!(carried.len(), 1, "the note is stored again, once");
        assert!(messages.last().unwrap().content.as_deref().unwrap().ends_with(&note), "at the end");
        let total: usize = messages.iter().map(ChatMessage::estimated_tokens).sum();
        let target = cut_target(budget, KEEP_RECENT_TOKENS, 0, &messages, &ChatMessage::estimated_tokens);
        assert!(total <= target, "the transcript is {total} tokens, past the cut's target {target}");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The note a prune leaves grows once the archive lands: it gains the
    /// address, a summary at its cap, and the fields carried from the
    /// previous record. The prune keeps room for all of it, so the upgraded
    /// transcript still meets its target however long the address is.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_prune_leaves_room_for_the_note_it_upgrades() {
        const BUDGET: usize = 24_000;
        for depth in [0, 3] {
            let root = std::env::temp_dir().join(format!("openmax-note-growth-{}", uuid::Uuid::new_v4()));
            let dir = (0..depth).fold(root.clone(), |dir, _| dir.join("d".repeat(200)));
            let (core, _) = Core::new(dir.clone()).unwrap();
            let id = sessions::create(&core, dir.display().to_string()).unwrap().id;
            sessions::append_compaction(&core, &id, &sessions::CompactionRecord {
                ts: 0,
                message_count: 9,
                // More tools than carry forward, and a path past its cap.
                tools: (0..4 * MAX_DIGEST_TOOLS).map(|i| format!("{}{i:02}", "t".repeat(MAX_DIGEST_TOOL_BYTES - 2))).collect(),
                paths: std::iter::once("p".repeat(MAX_DIGEST_PATH_BYTES + 1))
                    .chain((0..MAX_DIGEST_PATHS).map(|i| format!("{}{i:02}", "p".repeat(MAX_DIGEST_PATH_BYTES - 2))))
                    .collect(),
                user_snippets: Vec::new(),
                digest: String::new(),
            });
            let reply = format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":\"stop\"}}]}}\n\n", "s".repeat(MAX_SUMMARY_BYTES * 2));
            let (url, _requests) = crate::client::tests::spawn_recording_once(format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{reply}data: [DONE]\n\n"
            ));
            let client = ChatClient::new(url, None, "fixture".into(), None, 512);
            let (hooks, cancelled) = (Hooks::default(), Arc::new(CancelToken::default()));
            let ctx = CompactionCtx { core: &core, session_id: &id, project_root: &dir, client: &client, estimate: &ChatMessage::estimated_tokens, hooks: &hooks, cancelled: &cancelled, schemas: SCHEMAS, factor: 1.0, keep: KEEP_RECENT_TOKENS };
            // Small messages, so the cut stops just under its target.
            let mut messages = vec![ChatMessage::system("rules"), ChatMessage::user("Find the build result.")];
            messages.extend((0..300).map(|_| msg("assistant", 400)));
            assert!(sessions::save_messages(&core, &id, &messages, &mut 0, true));
            assert!(compact_messages(&ctx, &mut messages, BUDGET, BUDGET * 2, 0, false, &mut RequestSpend::new(None)).await.unwrap() > 0);
            let note = messages[2].content.clone().unwrap();
            assert!(note.contains(&format!("Summary: {}…", "s".repeat(MAX_SUMMARY_BYTES))), "the summary is at its cap");
            assert!(note.contains(&sessions::archive_display(&core, &id)), "the note names the archive");
            let total: usize = messages.iter().map(ChatMessage::estimated_tokens).sum();
            let target = cut_target(BUDGET, KEEP_RECENT_TOKENS, 0, &messages, &ChatMessage::estimated_tokens);
            assert!(total <= target, "with a {}-byte address the transcript is {total} tokens, past its target {target}", sessions::archive_display(&core, &id).len());
            // Each dropped reply is 100 tokens, so a prune that keeps only the
            // room the note can take ends within about one reply of the target.
            assert!(total + 120 > target, "the prune keeps more room than the note can take: {total} of {target}");
            drop(core);
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    /// One prune must buy headroom: after compaction the transcript *plus the
    /// schema overhead* sits at or below the prune target, and re-running
    /// enforce_budget mutates nothing, so the token prefix (and the server's
    /// prompt cache) stays stable while the next iterations append.
    #[test]
    fn budget_prunes_once_with_hysteresis() {
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..16 {
            messages.push(msg("assistant", 100));
            messages.push(msg("tool", 1000));
        }
        let budget = 4000;
        let schema_tokens = 300;
        let target = cut_target(budget, KEEP_RECENT_TOKENS, schema_tokens, &messages, &ChatMessage::estimated_tokens);
        assert!(enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens).is_some());
        let total: usize =
            schema_tokens + messages.iter().map(|m| m.estimated_tokens()).sum::<usize>();
        assert!(total <= target, "the cut should reach its target {target}, got {total} of {budget}");

        let snapshot: Vec<Option<String>> = messages.iter().map(|m| m.content.clone()).collect();
        assert!(
            enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens).is_none(),
            "second pass must be a no-op"
        );
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
        assert!(enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).is_some());
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
        let first_digest = messages[2].content.clone();
        assert!(enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).is_none(), "second pass must be a no-op");
        assert_eq!(messages[2].content, first_digest, "digest must not be replaced on no-op");

        for _ in 0..6 {
            messages.push(assistant_with_tools("edit_file", r#"{"path":"src/new.rs"}"#));
            messages.push(msg("tool", 2500));
        }
        assert!(enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).is_some());
        let digest_count = messages
            .iter()
            .filter(|m| m.content.as_deref().is_some_and(|c| c.starts_with(DIGEST_PREFIX)))
            .count();
        assert_eq!(digest_count, 1, "only one digest note may exist");
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
    }

    /// A later prune drops the earlier digest note; its paths and tools must
    /// carry into the new digest by code, not through the summarizer's prose,
    /// and the caps must hold when they do.
    #[test]
    fn digest_absorbs_prior_record_within_caps() {
        let mut digest = CompactionDigest::default();
        digest.record_message(&assistant_with_tools("read_file", r#"{"path":"src/new.rs"}"#));
        let prior = sessions::CompactionRecord {
            ts: 1,
            message_count: 4,
            tools: vec!["bash".into(), "grep".into()],
            paths: (0..20).map(|i| format!("src/old_{i}.rs")).collect(),
            user_snippets: vec!["earlier ask".into()],
            digest: "[context note: 4 earlier messages were compacted.".into(),
        };
        digest.absorb_prior(&prior);
        assert!(digest.tools.contains("bash") && digest.tools.contains("grep"));
        assert_eq!(digest.paths[0], "src/new.rs", "fresh paths keep priority");
        assert!(digest.paths.iter().any(|p| p == "src/old_0.rs"));
        assert!(
            digest.paths.len() <= MAX_DIGEST_PATHS,
            "carry-forward must stay bounded, got {}",
            digest.paths.len()
        );
        let text = digest.format(None, None);
        assert!(text.contains("src/new.rs") && text.contains("src/old_0.rs"), "{text}");
    }

    /// Everything a prune removes must be retrievable: the digest carries the
    /// full dropped messages for the archive, and the note names the address.
    #[test]
    fn digest_collects_dropped_messages_and_note_names_archive() {
        let mut messages = vec![msg("system", 100), msg("user", 100)];
        for _ in 0..10 {
            messages.push(msg("assistant", 2000));
            messages.push(msg("user", 2000));
        }
        let before = messages.len();
        let digest = enforce_budget(&mut messages, 2500, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens);
        let digest = digest.expect("exchange drop should produce a digest");
        assert_eq!(
            digest.dropped.len(),
            digest.message_count,
            "every dropped message must be captured for the archive"
        );
        assert_eq!(before - messages.len() + 1, digest.dropped.len(), "digest note replaces drops");
        let note = digest.format(None, Some("/tmp/data/sessions/s1.archive.jsonl"));
        assert!(note.contains("/tmp/data/sessions/s1.archive.jsonl"), "{note}");
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
        let digest = enforce_budget(&mut messages, 2500, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens);
        let digest = digest.expect("exchange drop should produce a digest");
        let text = digest.format(None, None);
        assert!(text.contains("read_file"), "{text}");
        assert!(text.contains("src/auth.rs"), "{text}");
        assert!(text.contains("Earlier goals"), "{text}");
    }

    /// After a prune that inserts a digest, token total must sit at or below
    /// the hysteresis target so the next iteration does not re-mutate history.
    #[test]
    fn budget_post_digest_stays_at_or_below_target() {
        let mut messages = vec![msg("system", 200), msg("user", 200)];
        for i in 0..24 {
            messages.push(assistant_with_tools(
                "read_file",
                &format!(r#"{{"path":"src/module_{i}.rs"}}"#),
            ));
            messages.push(msg("tool", 600));
        }
        let budget = 3500;
        let schema_tokens = 300;
        let target = cut_target(budget, KEEP_RECENT_TOKENS, schema_tokens, &messages, &ChatMessage::estimated_tokens);
        assert!(enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens).is_some());
        let total: usize =
            schema_tokens + messages.iter().map(|m| m.estimated_tokens()).sum::<usize>();
        assert!(
            total <= target,
            "post-digest total {total} must be <= target {target} (budget {budget})"
        );
        assert!(messages[2].content.as_deref().unwrap().starts_with(DIGEST_PREFIX));
        assert!(
            enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens).is_none(),
            "second pass must be a no-op"
        );
    }

    /// The tools JSON is re-sent whole on every request, so it spends the same
    /// window the transcript does. A transcript that fits on message bytes
    /// alone can still be over once the frozen schemas are counted, and the
    /// used total the Budget event reports is messages + schemas.
    #[test]
    fn budget_counts_frozen_tool_schemas() {
        let mut messages = vec![msg("system", 200), msg("user", 200)];
        for _ in 0..12 {
            messages.push(msg("assistant", 200));
            messages.push(msg("tool", 1000));
        }
        let message_tokens: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
        let schema_tokens = 600;

        // Messages alone exactly fill the window: nothing to do.
        let mut without = messages.clone();
        assert!(
            enforce_budget(&mut without, message_tokens, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).is_none(),
            "message-only total at budget must be a no-op"
        );

        // Same messages, same window, schemas now counted: over, so compaction
        // fires. This is the case the old message-only sum missed entirely.
        let mut with = messages.clone();
        let target = cut_target(message_tokens, KEEP_RECENT_TOKENS, schema_tokens, &messages, &ChatMessage::estimated_tokens);
        assert!(
            enforce_budget(&mut with, message_tokens, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens).is_some(),
            "schema overhead must push an at-budget transcript over"
        );
        let pruned: usize = with.iter().map(|m| m.estimated_tokens()).sum();
        assert!(
            pruned + schema_tokens <= target,
            "used total (messages {pruned} + schemas {schema_tokens}) must reach target {target}"
        );
    }

    /// Schemas are a fixed per-request cost, so when they alone fill the window
    /// no transcript fits and compaction is futile. It must then do nothing at
    /// all: pruning to the floor would drop history, emit a digest, and pay a
    /// summarization request every turn while never fitting.
    #[test]
    fn budget_does_not_thrash_when_schemas_exceed_the_window() {
        // A 8k-window model with a 4k completion reserve, and enough installed
        // tools to cost more than what is left: reachable at MAX_EXTERNAL_TOOLS.
        let budget = 8192usize.saturating_sub(4096 + 1024);
        let schema_tokens = 6800;
        assert!(schemas_exceed_budget(budget, schema_tokens));

        let mut messages = vec![msg("system", 200), msg("user", 200)];
        for turn in 0..4 {
            // Each turn appends an exchange, as a running session does.
            messages.push(assistant_with_tools("read_file", r#"{"path":"src/a.rs"}"#));
            messages.push(msg("tool", 4000));
            let before = messages.clone();
            let digest = enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens);
            assert!(digest.is_none(), "turn {turn}: futile compaction must not run, nor ask for notes");
            let unchanged: Vec<Option<String>> =
                messages.iter().map(|m| m.content.clone()).collect();
            let expected: Vec<Option<String>> = before.iter().map(|m| m.content.clone()).collect();
            assert_eq!(unchanged, expected, "turn {turn}: transcript must survive intact");
        }
        assert!(
            !messages.iter().any(|m| m.content.as_deref().is_some_and(|c| c.starts_with(DIGEST_PREFIX))),
            "no digest may stack up across turns"
        );

        // Just under the target the normal machinery still runs.
        let workable = budget * PRUNE_TARGET_PCT / 100 - 1;
        assert!(!schemas_outgrow_budget(budget, workable));
        assert!(enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, workable, 0, &ChatMessage::estimated_tokens).is_some());
    }

    /// Past PRUNE_TARGET_PCT of the window, schemas crowd the transcript but
    /// do not lock it out: cutting still brings the request under budget, so
    /// it must run. Skipping here would send an oversized request the
    /// provider rejects: a turn that fails for no reason.
    #[test]
    fn budget_still_prunes_when_schemas_crowd_but_fit_the_window() {
        let budget = 10_000;
        // 80% of the window: past the 70% mark, short of the window itself.
        let schema_tokens = 8_000;
        assert!(schemas_outgrow_budget(budget, schema_tokens), "degraded, so it must advise");
        assert!(!schemas_exceed_budget(budget, schema_tokens), "but not hopeless");

        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..12 {
            messages.push(msg("assistant", 200));
            messages.push(msg("tool", 4_000));
        }
        // Recent, cheap exchanges: the tail a prune is not allowed to touch.
        for _ in 0..3 {
            messages.push(msg("assistant", 100));
            messages.push(msg("tool", 200));
        }

        let target = cut_target(budget, KEEP_RECENT_TOKENS, schema_tokens, &messages, &ChatMessage::estimated_tokens);
        let digest = enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens);
        assert!(digest.is_some(), "a cut that would work must not be skipped");
        let total: usize =
            schema_tokens + messages.iter().map(|m| m.estimated_tokens()).sum::<usize>();
        assert!(total <= budget, "the request must now fit: {total} of {budget}");
        // The schemas, the opening and a quarter of the window would pass the
        // window, so the cut aims at the window itself.
        assert_eq!(target, budget);
        assert!(messages.len() > 6, "history is pruned, not shredded to the floor");

        // And the reduced target still leaves a gap, so the next turns append
        // instead of re-compacting: this is the thrash the skip guarded against.
        assert!(
            enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens).is_none(),
            "second pass must be a no-op"
        );
        let headroom = budget - total;
        assert!(headroom > 0, "a hysteresis gap must survive, got {headroom}");
    }

    /// The setting exists for windows compaction never reaches: a transcript
    /// over the setting but far under the window budget must be cut, through
    /// the same machinery, down to the setting's own cut target.
    #[test]
    fn a_compaction_setting_pulls_the_trigger_below_the_window_budget() {
        let budget = 100_000;
        let schema_tokens = 500;
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        while schema_tokens + messages.iter().map(|m| m.estimated_tokens()).sum::<usize>()
            <= 30_000
        {
            messages.push(msg("assistant", 2_000));
            messages.push(msg("user", 2_000));
        }
        let trigger = compaction_trigger(budget, schema_tokens, 30_000, KEEP_RECENT_TOKENS, None, &messages, &ChatMessage::estimated_tokens);
        assert_eq!(trigger, 30_000);
        let target = cut_target(trigger, KEEP_RECENT_TOKENS, schema_tokens, &messages, &ChatMessage::estimated_tokens);
        let digest = enforce_budget(&mut messages, trigger, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens);
        assert!(digest.is_some(), "over the setting must cut with the window budget still far away");
        let total: usize =
            schema_tokens + messages.iter().map(|m| m.estimated_tokens()).sum::<usize>();
        assert!(total <= target, "the cut aims at the setting's own target {target}: {total}");
        assert!(
            enforce_budget(&mut messages, trigger, KEEP_RECENT_TOKENS, schema_tokens, 0, &ChatMessage::estimated_tokens).is_none(),
            "and the gap buys append-only turns, same as at the window"
        );
    }

    /// One direction only: a setting above the budget must not delay
    /// compaction past the point where the request stops fitting the
    /// endpoint.
    #[test]
    fn the_compaction_setting_never_raises_the_trigger() {
        assert_eq!(compaction_trigger(50_000, 500, 400_000, KEEP_RECENT_TOKENS, None, &[], &ChatMessage::estimated_tokens), 50_000);
    }

    /// A setting whose prune target the frozen schemas outgrow can never be
    /// reached by pruning. It must fall back to the window budget, not feed
    /// the futility guard: passing it through would disarm compaction
    /// entirely while the transcript grows toward the real window.
    #[test]
    fn an_unachievable_compaction_setting_falls_back_to_the_window_budget() {
        let budget = 150_000;
        let schema_tokens = 18_000;
        assert!(schemas_outgrow_budget(20_000, schema_tokens));
        assert_eq!(compaction_trigger(budget, schema_tokens, 20_000, KEEP_RECENT_TOKENS, None, &[], &ChatMessage::estimated_tokens), budget);
    }

    /// The other unachievable shape: a protected tail the setting's target
    /// cannot contain. Pruning would stop at the drop loop's length floor
    /// still over the target and fire again on every iteration, paying an
    /// archive and a summary request each time, so the setting must fall
    /// back to the window budget instead of arming that loop.
    #[test]
    fn a_setting_the_protected_tail_outgrows_falls_back_to_the_window_budget() {
        let budget = 200_000;
        let schema_tokens = 500;
        // Six tool-heavy messages the drop loop may never remove, together
        // well past the 5,000 tokens the floored setting's cut keeps.
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..3 {
            messages.push(msg("assistant", 200));
            messages.push(msg("tool", 30_000));
        }
        assert_eq!(
            compaction_trigger(budget, schema_tokens, 20_000, KEEP_RECENT_TOKENS, None, &messages, &ChatMessage::estimated_tokens),
            budget,
            "an unreachable target must not arm per-iteration compaction"
        );
        // The same setting over a lean transcript stays in force.
        let lean = vec![msg("system", 400), msg("user", 400)];
        assert_eq!(compaction_trigger(budget, schema_tokens, 20_000, KEEP_RECENT_TOKENS, None, &lean, &ChatMessage::estimated_tokens), 20_000);

        // And the floor counts only what a maximal cut actually leaves:
        // the pinned head, the note, and the newest three. Six mid-sized
        // messages whose newest three fit the target must not trip the
        // fallback just because all six together would not.
        let mut boundary = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..6 {
            boundary.push(msg("assistant", 3_600));
        }
        assert_eq!(
            compaction_trigger(budget, schema_tokens, 20_000, KEEP_RECENT_TOKENS, None, &boundary, &ChatMessage::estimated_tokens),
            20_000,
            "a reachable target must keep the setting in force"
        );

        // A note-first opening pins the request behind the note as well, so
        // a request the target cannot contain makes the setting unreachable
        // however much prunable history follows it: no cut ends under it.
        let note = ChatMessage::user(execution_policy_note(ApprovalMode::Ask));
        let mut note_first = vec![msg("system", 400), note, msg("user", 80_000)];
        for _ in 0..6 {
            note_first.push(msg("assistant", 400));
        }
        assert_eq!(
            compaction_trigger(budget, schema_tokens, 20_000, KEEP_RECENT_TOKENS, None, &note_first, &ChatMessage::estimated_tokens),
            budget,
            "the floor must count the whole pinned head"
        );
    }

    /// The feasibility check reserves DIGEST_NOTE_ALLOWANCE_TOKENS for the
    /// note a prune leaves, so the note must actually be bounded: garbage
    /// path arguments and hallucinated tool names, fresh or carried in from
    /// a pre-cap record, must never enter the digest fields.
    #[test]
    fn pathological_tool_calls_cannot_enter_the_digest_fields() {
        let mut digest = CompactionDigest::default();
        digest.record_message(&assistant_with_tools(
            &"t".repeat(MAX_DIGEST_TOOL_BYTES + 1),
            &format!("{{\"path\":\"{}\"}}", "p".repeat(MAX_DIGEST_PATH_BYTES * 40)),
        ));
        // Multibyte entries under the old char caps but over the byte caps:
        // the allowance is bytes/4, so these must be rejected too.
        digest.record_message(&assistant_with_tools(
            &"Ā".repeat(40),
            &format!("{{\"path\":\"{}\"}}", "€".repeat(100)),
        ));
        digest.record_message(&ChatMessage::user("€".repeat(400)));
        digest.record_message(&assistant_with_tools(
            "read_file",
            "{\"path\":\"src/lib.rs\"}",
        ));
        digest.absorb_prior(&sessions::CompactionRecord {
            ts: 0,
            message_count: 9,
            tools: vec!["x".repeat(50_000), "€".repeat(100)],
            paths: vec!["y".repeat(50_000), "Ā".repeat(200)],
            user_snippets: Vec::new(),
            digest: String::new(),
        });
        assert_eq!(digest.tools.iter().cloned().collect::<Vec<_>>(), vec!["read_file"]);
        assert_eq!(digest.paths, vec!["src/lib.rs"]);
        assert!(
            digest.user_snippets.iter().all(|s| s.len() <= MAX_DIGEST_SNIPPET_BYTES),
            "snippets are byte-capped like every other note field"
        );
    }

    /// The other half of the same contract: a digest at every cap must format
    /// to a note within the allowance, or the trigger check would arm a
    /// target enforce_budget can never reach and compact on every iteration.
    /// This is the lockstep gate between the constant and the format.
    #[test]
    fn the_digest_note_at_its_caps_stays_within_the_allowance() {
        let mut digest = CompactionDigest { message_count: 999, ..Default::default() };
        for i in 0..MAX_DIGEST_TOOLS {
            digest.tools.insert(format!("{}{i:02}", "t".repeat(MAX_DIGEST_TOOL_BYTES - 2)));
        }
        for i in 0..MAX_DIGEST_PATHS {
            digest.paths.push(format!("{}{i:02}", "p".repeat(MAX_DIGEST_PATH_BYTES - 2)));
        }
        for _ in 0..4 {
            digest.user_snippets.push("s".repeat(120));
        }
        let archive = "a".repeat(200);
        // The notes as `summarize_compaction` caps them, ellipsis included.
        let summary = take_note_bytes(&"m".repeat(MAX_SUMMARY_BYTES + 1), MAX_SUMMARY_BYTES) + "…";
        let costs = [digest.format(None, Some(&archive)), digest.format(Some(&summary), Some(&archive))]
            .map(|note| ChatMessage::user(note).estimated_tokens());
        for cost in costs {
            assert!(
                cost <= DIGEST_NOTE_ALLOWANCE_TOKENS,
                "a note at every cap costs {cost}, over the {DIGEST_NOTE_ALLOWANCE_TOKENS} allowance"
            );
        }
        // The smallest hundred that holds it, so the floor reserves no more.
        assert!(costs.iter().any(|&cost| cost + 100 > DIGEST_NOTE_ALLOWANCE_TOKENS), "{costs:?}");
    }

    /// Typos do not configure thrash: a tiny setting rides the floor, where
    /// the gap a cut leaves is still worth whole turns.
    #[test]
    fn a_tiny_compaction_setting_is_floored() {
        assert_eq!(
            compaction_trigger(100_000, 500, 1_000, KEEP_RECENT_TOKENS, None, &[], &ChatMessage::estimated_tokens),
            COMPACTION_TOKENS_FLOOR
        );
    }

    /// The floor bounds the user's setting, which counts real tokens, so it
    /// applies before the conversion. Under a learned 4x factor a 60k setting
    /// compacts at 60k real tokens (15,000 estimated), not at the floor taken
    /// in estimator units (20,000 estimated, 80k real), and a tiny setting
    /// rides the floor in real tokens too. `/compact` aims at the same mark.
    #[test]
    fn a_calibrated_setting_is_floored_in_real_tokens() {
        let ratio = Some(4.0);
        let mut transcript = vec![msg("system", 200), msg("user", 200)];
        // ~18k estimated, ~72k real: past the 60k the setting asks for.
        while transcript.iter().map(ChatMessage::estimated_tokens).sum::<usize>() < 18_000 {
            transcript.push(msg("assistant", 2_000));
            transcript.push(msg("user", 2_000));
        }
        let trigger = compaction_trigger(100_000, 500, 60_000, KEEP_RECENT_TOKENS, ratio, &transcript, &ChatMessage::estimated_tokens);
        assert_eq!(trigger, 15_000);
        assert_eq!(requested_trigger(60_000, ratio, 100_000), trigger);
        assert!(enforce_budget(&mut transcript, trigger, KEEP_RECENT_TOKENS, 500, 0, &ChatMessage::estimated_tokens).is_some(), "compaction fires at 60k real");
        assert_eq!(requested_trigger(1_000, ratio, 100_000), to_estimate(COMPACTION_TOKENS_FLOOR, token_factor(ratio)));
        assert_eq!(requested_trigger(1_000, Some(0.5), 100_000), 40_000);
        for invalid in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            assert_eq!(requested_trigger(1_000, Some(invalid), 100_000), 20_000);
        }
    }

    /// What the server reports can only tighten the budget: no observation
    /// is today's estimate, an under-count scales it, an over-count is
    /// ignored, and a nonsense report is held to what a tokenizer can emit.
    #[test]
    fn the_token_factor_only_tightens_and_is_bounded() {
        assert_eq!(token_factor(None), 1.0);
        assert_eq!(token_factor(Some(1.25)), 1.25);
        assert_eq!(token_factor(Some(0.6)), 1.0);
        assert_eq!(token_factor(Some(1_000.0)), 4.0);
        // The window converts into estimator units and usage back out of
        // them, so a 1.25x under-count compacts at 80% of the old estimate.
        assert_eq!(to_estimate(100_000, token_factor(Some(1.25))), 80_000);
        assert_eq!(to_real(80_000, token_factor(Some(1.25))), 100_000);
        assert_eq!(to_estimate(100_000, token_factor(None)), 100_000);

        let completion = |usage| CompletionResult {
            content: String::new(),
            tool_calls: Vec::new(),
            reasoning_content: None,
            reasoning: None,
            reasoning_details: None,
            finish_reason: "stop".into(),
            usage,
        };
        let reported = |prompt_tokens| {
            completion(Some(crate::client::Usage { prompt_tokens, ..Default::default() }))
        };
        assert_eq!(updated_prompt_ratio(None, &reported(1_200), 1_000), Some(1.2));
        assert_eq!(updated_prompt_ratio(Some(0.6), &reported(0), 1_000), None);
        assert_eq!(updated_prompt_ratio(Some(1.2), &completion(None), 1_000), Some(1.2));
        let mut limited = reported(600);
        limited.finish_reason = "length".into();
        assert_eq!(updated_prompt_ratio(None, &limited, 1_000), Some(0.6));
        for reason in [TRUNCATED, "cancelled"] {
            limited.finish_reason = reason.into();
            assert_eq!(updated_prompt_ratio(Some(0.6), &limited, 1_000), None);
            assert_eq!(updated_prompt_ratio(Some(1.2), &limited, 1_000), Some(1.2));
        }
        let inconsistent = |prompt_tokens| completion(Some(crate::client::Usage {
            prompt_tokens, cached_tokens: Some(prompt_tokens + 1), ..Default::default()
        }));
        assert_eq!(updated_prompt_ratio(Some(0.6), &inconsistent(600), 1_000), None);
        assert_eq!(updated_prompt_ratio(None, &inconsistent(1_200), 1_000), Some(1.2));
    }

    /// A refusal for length is a floor on the ratio: what the window leaves
    /// for the prompt over the estimate of the refused request. It teaches
    /// only what the budget did not already assume, only what a tokenizer
    /// can explain, and only when the provider's error reads as one.
    #[test]
    fn a_refusal_for_length_proves_a_floor_on_the_ratio() {
        let length = "backend returned 400 Bad Request: This model's maximum context length is 32768 tokens. However, you requested 43793 tokens";
        assert_eq!(refusal_ratio(length, 20_000, 28_672, 1.0), Some(1.4336));
        assert_eq!(refusal_ratio(length, 20_000, 28_672, 1.4), Some(1.4336));
        assert_eq!(refusal_ratio(length, 20_000, 28_672, 1.4336), None, "the budget already assumed it");
        assert_eq!(refusal_ratio(length, 30_000, 28_672, 1.0), None, "a request known to be over");
        assert_eq!(refusal_ratio(length, 7_168, 28_672, 1.0), Some(4.0), "the most a tokenizer explains");
        assert_eq!(refusal_ratio(length, 7_167, 28_672, 1.0), None, "past it the estimate is not what failed");
        assert_eq!(refusal_ratio(length, 0, 28_672, 1.0), None);
        let other = "backend returned 400 Bad Request: Unsupported value: 'temperature'";
        assert_eq!(refusal_ratio(other, 20_000, 28_672, 1.0), None, "not a refusal for length");

        for refusal in [
            "This model's maximum context length is 8192 tokens",
            "backend reported an error (context_length_exceeded): input too large",
            "the request exceeds the available context size, try increasing it",
            "Your input exceeds the context window of this model.",
            "input length and `max_tokens` exceed context limit: 190000 + 20000 > 200000",
            "prompt is too long: 213000 tokens > 200000 maximum",
            "The decoder prompt (length 40000) is longer than the maximum model length of 32768.",
            "This model's maximum prompt length is 131072 but the request contains 140000 tokens.",
            "The input token count (1100000) exceeds the maximum number of tokens allowed (1048576).",
            "backend returned 400 Bad Request: Please reduce the length of the messages or completion.",
        ] {
            assert!(refused_for_length(refusal), "{refusal}");
        }
        for other in [
            "backend returned 401 Unauthorized: invalid api key",
            "backend returned 429 Too Many Requests: Rate limit reached for requests",
            "backend returned 400 Bad Request: Unsupported value: 'temperature'",
            "request failed: connection refused",
            "backend reported an error (insufficient_quota): You exceeded your current quota",
        ] {
            assert!(!refused_for_length(other), "{other}");
        }
    }

    /// `/compact` exists to fire inside the hysteresis gap, where the budget
    /// path correctly holds still: between the prune target and the budget,
    /// enforce_budget must do nothing and the forced prune must still land
    /// on the same target.
    #[test]
    fn a_forced_prune_fires_where_the_budget_path_holds() {
        let budget = 10_000;
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        while messages.iter().map(|m| m.estimated_tokens()).sum::<usize>() <= 8_000 {
            messages.push(msg("assistant", 800));
            messages.push(msg("user", 800));
        }
        let total: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
        let target = cut_target(budget, KEEP_RECENT_TOKENS, 0, &messages, &ChatMessage::estimated_tokens);
        assert!(total > target && total <= budget, "the fixture must sit inside the gap: {total}");
        assert!(enforce_budget(&mut messages, budget, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).is_none(), "the budget path holds here");
        let digest = prune_transcript(&mut messages, budget, KEEP_RECENT_TOKENS, 0, total, 0, &ChatMessage::estimated_tokens);
        assert!(digest.is_some_and(|d| d.message_count > 0), "the forced cut fires where the budget path held");
        let after: usize = messages.iter().map(|m| m.estimated_tokens()).sum();
        assert!(after <= target, "and lands on the same target: {after}");
    }

    /// A turn's prompt, stamped with the effort it ran at, as the agent loop
    /// stamps every prompt.
    fn prompt(text: &str) -> ChatMessage {
        ChatMessage { effort: Some("default fixture".into()), ..ChatMessage::user(text) }
    }

    /// Exchange `i`: one call and its result, `len` bytes unique to `i`.
    fn exchange(i: usize, len: usize) -> [ChatMessage; 2] {
        let mut call = assistant_with_tools("read_file", r#"{"path":"src/lib.rs"}"#);
        call.tool_calls.as_mut().unwrap()[0].id = format!("call_{i}");
        [call, ChatMessage::tool(format!("call_{i}"), format!("{i:06}{}", "r".repeat(len.saturating_sub(6))))]
    }

    /// The system prompt, the original request, and `n` exchanges of `len`.
    fn session(n: usize, len: usize) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system("rules"), prompt("the task")];
        messages.extend((0..n).flat_map(|i| exchange(i, len)));
        messages
    }

    fn wire(message: &ChatMessage) -> String {
        serde_json::to_string(message).unwrap()
    }

    fn tokens(messages: &[ChatMessage]) -> usize {
        messages.iter().map(ChatMessage::estimated_tokens).sum()
    }

    /// One streamed reply holding `text`, and a call of `read_file` when
    /// `call` is set.
    fn reply(text: &str, call: bool) -> String {
        let mut delta = serde_json::json!({ "content": text });
        if call {
            delta["tool_calls"] = serde_json::json!([{ "index": 0, "id": "notes_call", "type": "function",
                "function": { "name": "read_file", "arguments": "{\"path\":\"a.rs\"}" } }]);
        }
        let chunk = serde_json::json!({ "choices": [{ "delta": delta, "finish_reason": if call { "tool_calls" } else { "stop" } }] });
        format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {chunk}\n\ndata: [DONE]\n\n")
    }

    /// The JSON body of the next request `requests` recorded.
    fn body(requests: &std::sync::mpsc::Receiver<String>) -> Value {
        let request = requests.recv_timeout(Duration::from_secs(5)).expect("a request");
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    }

    /// A session to cut in, against `url`, with one compaction hook that
    /// writes each record it observes to `hooked.json`.
    struct Fixture {
        dir: std::path::PathBuf,
        core: Arc<Core>,
        id: String,
        client: ChatClient,
        hooks: Hooks,
        cancelled: Arc<CancelToken>,
    }

    impl Fixture {
        fn new(url: &str) -> Self {
            use std::os::unix::fs::PermissionsExt as _;
            let dir = std::env::temp_dir().join(format!("openmax-cut-{}", uuid::Uuid::new_v4()));
            let (core, _) = Core::new(dir.join("data")).unwrap();
            let id = sessions::create(&core, dir.display().to_string()).unwrap().id;
            std::fs::create_dir_all(dir.join(".openmax/hooks")).unwrap();
            let script = dir.join("hook.sh");
            std::fs::write(&script, format!("#!/bin/sh\ncat > '{}'\n", dir.join("hooked.json").display())).unwrap();
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
            let hook = format!("event = \"compaction\"\ncommand = \"{}\"\n", script.display());
            std::fs::write(dir.join(".openmax/hooks/cut.toml"), hook).unwrap();
            let hooks = crate::hooks::tests::discover_for_test(&dir);
            assert!(!hooks.is_empty(), "the fixture's hook is live");
            let client = ChatClient::new(url.into(), None, "fixture".into(), None, 512);
            Self { dir, core, id, client, hooks, cancelled: Arc::new(CancelToken::default()) }
        }

        fn ctx(&self) -> CompactionCtx<'_> {
            CompactionCtx {
                core: &self.core,
                session_id: &self.id,
                project_root: &self.dir,
                client: &self.client,
                estimate: &ChatMessage::estimated_tokens,
                hooks: &self.hooks,
                cancelled: &self.cancelled,
                schemas: SCHEMAS,
                factor: 1.0,
                keep: KEEP_RECENT_TOKENS,
            }
        }

        /// Cut the saved `messages` at `trigger` in a window of `budget`.
        async fn cut(&self, messages: &mut Vec<ChatMessage>, trigger: usize, budget: usize, spend: &mut RequestSpend) -> usize {
            assert!(sessions::save_messages(&self.core, &self.id, messages, &mut 0, true));
            compact_messages(&self.ctx(), messages, trigger, budget, 0, false, spend).await.unwrap()
        }

        /// Whether a cut committed anything: an archive line, a record, or a
        /// hook run.
        fn committed(&self) -> bool {
            !sessions::load_archive(&self.core, &self.id).is_empty()
                || sessions::last_compaction(&self.core, &self.id).is_some()
                || self.dir.join("hooked.json").exists()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A cut keeps the opening and the newest exchanges byte for byte, and
    /// the first message it keeps is never a tool result cut off from its
    /// call, even where a call has more than one result.
    #[test]
    fn a_cut_keeps_the_opening_and_the_newest_exchanges_byte_for_byte() {
        let mut messages = session(0, 0);
        for i in 0..40 {
            let [mut call, result] = exchange(i, 1_500);
            if i % 2 == 1 {
                let mut second = call.tool_calls.as_ref().unwrap()[0].clone();
                second.id = format!("call_{i}_b");
                call.tool_calls.as_mut().unwrap().push(second);
                messages.extend([call, result, ChatMessage::tool(format!("call_{i}_b"), "y".repeat(1_500))]);
            } else {
                messages.extend([call, result]);
            }
        }
        let original = messages.clone();
        let digest = enforce_budget(&mut messages, 20_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("a cut");
        let head = pinned_head(&original);
        assert_eq!(messages[..head].iter().map(wire).collect::<Vec<_>>(), original[..head].iter().map(wire).collect::<Vec<_>>());
        assert!(is_digest_message(&messages[head]));
        let kept = &messages[head + 1..];
        let tail = &original[original.len() - kept.len()..];
        assert_eq!(kept.iter().map(wire).collect::<Vec<_>>(), tail.iter().map(wire).collect::<Vec<_>>(), "the newest exchanges, unchanged");
        assert_eq!(digest.message_count, original.len() - head - kept.len(), "everything between is cut");
        assert_eq!(kept[0].role, "assistant", "the first kept message is a call, not a result: {:?}", kept[0]);
    }

    /// No kept message is rewritten: a large tool result in the tail stays
    /// whole, and no result anywhere is shortened in place.
    #[test]
    fn a_cut_shortens_no_kept_message() {
        let mut messages = session(20, 4_000);
        let original = messages.clone();
        enforce_budget(&mut messages, 12_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("a cut");
        let last = messages.last().unwrap();
        assert_eq!(last.content, original.last().unwrap().content, "the newest 4,000-byte result is whole");
        assert_eq!(last.content.as_deref().map(str::len), Some(4_000));
        assert!(!messages.iter().any(|m| m.content.as_deref().is_some_and(|c| c.contains("older tool output truncated"))));
        for kept in messages.iter().filter(|m| m.role == "tool") {
            assert!(original.iter().any(|m| wire(m) == wire(kept)), "a kept result was rewritten: {kept:?}");
        }
    }

    /// A cut at a trigger of 40,000 keeps at most a quarter of it, 10,000
    /// tokens, after the opening: the note and the newest exchanges.
    #[test]
    fn a_cut_keeps_a_quarter_of_its_trigger() {
        let mut messages = session(90, 2_000);
        assert!(tokens(&messages) > 40_000);
        let head = pinned_head(&messages);
        enforce_budget(&mut messages, 40_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("a cut");
        let kept = tokens(&messages[head..]);
        assert!(kept <= 10_000, "{kept} tokens kept after the opening");
        assert!(kept > 9_000, "and close to all of it: {kept}");
    }

    /// At a trigger of 400,000 the cut keeps at most 20,000 tokens, and as
    /// many of the newest exchanges as fit: one more would not.
    #[test]
    fn a_cut_keeps_at_most_twenty_thousand_tokens_and_no_fewer_than_fit() {
        let mut messages = session(200, 8_000);
        assert!(tokens(&messages) > 400_000);
        let head = pinned_head(&messages);
        let digest = enforce_budget(&mut messages, 400_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("a cut");
        let kept = tokens(&messages[head..]);
        assert!(kept <= KEEP_RECENT_TOKENS, "{kept} tokens kept after the opening");
        let next = digest.dropped.iter().rposition(|m| m.role == "assistant").unwrap();
        let one_more = tokens(&digest.dropped[next..]);
        assert!(kept + one_more > KEEP_RECENT_TOKENS, "{kept} kept, and the next exchange of {one_more} would fit");
    }

    /// The prompt of the turn in progress stays verbatim, stamp included,
    /// right after the note when the cut reaches past it, and is still cut
    /// with everything else, so the archive and the notes have it in place.
    #[test]
    fn a_cut_keeps_the_prompt_of_the_turn_in_progress() {
        let mut messages = session(4, 2_000);
        let turn = prompt("now rename the parser types");
        messages.push(turn.clone());
        messages.extend((4..60).flat_map(|i| exchange(i, 2_000)));
        let head = pinned_head(&messages);
        let digest = enforce_budget(&mut messages, 20_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("a cut");
        assert!(is_digest_message(&messages[head]));
        assert_eq!(wire(&messages[head + 1]), wire(&turn), "the prompt follows the note verbatim");
        assert_eq!(messages[head + 1].effort, turn.effort, "with its stamp");
        assert!(digest.dropped.iter().any(|m| wire(m) == wire(&turn)), "and the cut still records it");
        assert_eq!(messages.iter().filter(|m| m.content == turn.content).count(), 1);
    }

    /// The prompt stays in place across a second cut too: the first cut put
    /// it after its note, and the second cuts it again and puts it back.
    #[test]
    fn a_second_cut_keeps_the_prompt_of_the_turn_in_progress() {
        let mut messages = session(4, 2_000);
        let turn = prompt("now rename the parser types");
        messages.push(turn.clone());
        messages.extend((4..60).flat_map(|i| exchange(i, 2_000)));
        let head = pinned_head(&messages);
        enforce_budget(&mut messages, 20_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("the first cut");
        messages.extend((60..100).flat_map(|i| exchange(i, 2_000)));
        let digest = enforce_budget(&mut messages, 20_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("the second cut");
        assert!(digest.dropped.iter().any(|m| wire(m) == wire(&turn)), "the second cut reached the prompt");
        assert!(is_digest_message(&messages[head]));
        assert_eq!(wire(&messages[head + 1]), wire(&turn));
        assert_eq!(messages.iter().filter(|m| m.content == turn.content).count(), 1);
        assert_eq!(messages.iter().filter(|m| is_digest_message(m)).count(), 1);
    }

    /// The prompt the cut puts back is counted before the cut chooses what
    /// else to keep: with a turn's prompt of 33,000 tokens in a window of
    /// 40,000, the request after a cut the window forces still fits it, and a
    /// setting the prompt alone outgrows falls back to the window.
    #[tokio::test]
    async fn a_large_prompt_put_back_after_a_cut_still_fits_the_window() {
        let fixture = Fixture::new("http://127.0.0.1:9");
        let mut messages = session(4, 2_000);
        let turn = prompt(&"p".repeat(132_000));
        messages.push(turn.clone());
        messages.extend((4..30).flat_map(|i| exchange(i, 2_000)));
        assert!(tokens(&messages) > 40_000, "the window forces the cut");
        let trigger = compaction_trigger(40_000, 0, 20_000, KEEP_RECENT_TOKENS, None, &messages, &ChatMessage::estimated_tokens);
        assert_eq!(trigger, 40_000, "no cut reaches a target of 20,000 with the prompt in it");
        assert!(fixture.cut(&mut messages, trigger, 40_000, &mut RequestSpend::new(None)).await > 0);
        assert!(tokens(&messages) <= 40_000, "the request after the cut is {} tokens", tokens(&messages));
        assert_eq!(wire(&messages[pinned_head(&messages) + 1]), wire(&turn), "the prompt follows the note");
    }

    /// Unset, a cut keeps what it always kept. Set, the setting is floored
    /// in its real tokens and converted by the learned ratio like the
    /// trigger, and a cut keeps at most a quarter of its trigger either way.
    #[test]
    fn the_keep_setting_defaults_to_the_fixed_tail_and_is_floored_and_calibrated() {
        assert_eq!(requested_keep(None, None), KEEP_RECENT_TOKENS);
        assert_eq!(requested_keep(None, Some(2.0)), KEEP_RECENT_TOKENS, "the default does not move with the ratio");
        assert_eq!(requested_keep(Some(5_000), None), 5_000);
        assert_eq!(requested_keep(Some(1), None), COMPACTION_KEEP_FLOOR);
        assert_eq!(requested_keep(Some(5_000), Some(2.0)), 2_500);
        let messages = session(4, 100);
        let head: usize = messages[..pinned_head(&messages)].iter().map(ChatMessage::estimated_tokens).sum();
        assert_eq!(cut_target(60_000, 5_000, 0, &messages, &ChatMessage::estimated_tokens), head + 5_000);
        assert_eq!(cut_target(60_000, 40_000, 0, &messages, &ChatMessage::estimated_tokens), head + 15_000);
    }

    /// A small keep whose target the protected tail outgrows still cuts, to
    /// the floor, where the default target holds that tail: falling back to
    /// the window there left a session uncut until the window forced it.
    #[test]
    fn a_small_keep_cuts_to_the_floor_where_the_newest_exchanges_outgrow_it() {
        let mut messages = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..30 {
            messages.push(msg("user", 9_000));
            messages.push(msg("assistant", 9_000));
        }
        let estimate = &ChatMessage::estimated_tokens;
        let (budget, newest) = (190_000, 3 * 2_250);
        assert!(cut_target(60_000, 5_000, 0, &messages, estimate) < newest + DIGEST_NOTE_ALLOWANCE_TOKENS, "fixture: the tail outgrows the small target");
        assert_eq!(compaction_trigger(budget, 0, 60_000, 5_000, None, &messages, estimate), 60_000);
        // At the largest learned ratio the converted keep (1,250) is below the
        // note's own allowance; where the default tail fits, the cut still
        // fires at the converted trigger.
        let ratio = Some(MAX_TOKEN_FACTOR);
        let keep = requested_keep(Some(5_000), ratio);
        assert!(keep < DIGEST_NOTE_ALLOWANCE_TOKENS, "fixture: {keep}");
        let mut lean = vec![msg("system", 400), msg("user", 400)];
        for _ in 0..60 {
            lean.push(msg("user", 1_600));
            lean.push(msg("assistant", 1_600));
        }
        let trigger = requested_trigger(60_000, ratio, budget);
        assert_eq!(compaction_trigger(budget, 0, 60_000, KEEP_RECENT_TOKENS, ratio, &lean, estimate), trigger, "fixture: the default cuts");
        assert_eq!(compaction_trigger(budget, 0, 60_000, keep, ratio, &lean, estimate), trigger);
        enforce_budget(&mut lean, trigger, keep, 0, 0, estimate).expect("a cut");
        assert!(tokens(&lean) <= 200 + 4 * 400 + DIGEST_NOTE_ALLOWANCE_TOKENS, "{}", tokens(&lean));
        let mut kept_small = messages.clone();
        enforce_budget(&mut kept_small, 60_000, 5_000, 0, 0, estimate).expect("a cut");
        let mut kept_default = messages.clone();
        enforce_budget(&mut kept_default, 60_000, KEEP_RECENT_TOKENS, 0, 0, estimate).expect("a cut");
        let (small, default) = (tokens(&kept_small), tokens(&kept_default));
        // The drop loop's floor keeps four messages past the head.
        assert!(small <= 200 + 4 * 2_250 + DIGEST_NOTE_ALLOWANCE_TOKENS && small < default, "{small} against {default}");
    }

    /// A session of one prompt keeps it in the opening, so the cut adds
    /// nothing beside its note.
    #[test]
    fn a_cut_of_a_single_prompt_session_adds_only_its_note() {
        let mut messages = session(60, 2_000);
        let before = messages.len();
        let digest = enforce_budget(&mut messages, 20_000, KEEP_RECENT_TOKENS, 0, 0, &ChatMessage::estimated_tokens).expect("a cut");
        assert_eq!(messages.len(), before - digest.message_count + 1);
        assert_eq!(messages.iter().filter(|m| m.effort.is_some()).count(), 1, "the one prompt, in the opening");
    }

    /// The notes request is the transcript up to the cut, each message as the
    /// request before the cut sent it, then the ask, with the turn's tools:
    /// every byte ahead of the ask is one a prompt cache already holds.
    #[tokio::test]
    async fn the_notes_request_is_the_transcript_up_to_the_cut_then_the_ask() {
        let (url, requests) = crate::client::tests::spawn_recording_once(reply("State: halfway.", false));
        let fixture = Fixture::new(&url);
        let mut messages = session(60, 2_000);
        let original = messages.clone();
        let (earlier, sent_before) = crate::client::tests::spawn_recording_once(reply("ok", false));
        let client = ChatClient::new(earlier, None, "fixture".into(), None, 512);
        client.stream_chat(&original, client.reply_reserve, SCHEMAS, Arc::new(CancelToken::default()), |_| {}).await.unwrap();
        assert!(fixture.cut(&mut messages, 20_000, 100_000, &mut RequestSpend::new(None)).await > 0);
        let (notes, before) = (body(&requests), body(&sent_before));
        let cut = original.len() - (messages.len() - pinned_head(&messages) - 1);
        let sent = notes["messages"].as_array().unwrap();
        assert_eq!(sent.len(), cut + 1, "the transcript up to the cut, then the ask");
        assert_eq!(sent[..cut], before["messages"].as_array().unwrap()[..cut]);
        assert_eq!((&sent[cut]["role"], &sent[cut]["content"]), (&"user".into(), &NOTES_ASK.into()));
        assert_eq!(notes["tools"], before["tools"]);
        assert_eq!(notes["tools"], serde_json::from_str::<Value>(SCHEMAS).unwrap(), "the turn's tools");
        assert!(messages[pinned_head(&messages)].content.as_deref().unwrap().contains("Summary: State: halfway."));
    }

    /// The notes are the reply's text. A call beside it never runs, and
    /// nothing of the exchange is kept: not the call, not the ask.
    #[tokio::test]
    async fn notes_beside_a_call_are_kept_and_the_call_is_not() {
        let (url, _requests) = crate::client::tests::spawn_recording_once(reply("State: wired the parser.", true));
        let fixture = Fixture::new(&url);
        let mut messages = session(60, 2_000);
        let dropped = fixture.cut(&mut messages, 20_000, 100_000, &mut RequestSpend::new(None)).await;
        assert!(dropped > 0);
        let note = messages[pinned_head(&messages)].content.clone().unwrap();
        assert!(note.contains("Summary: State: wired the parser."), "{note}");
        let archive = sessions::load_archive(&fixture.core, &fixture.id);
        assert_eq!(archive.len(), dropped, "the archive holds what the cut removed and nothing else");
        let leaked = |m: &ChatMessage| {
            m.content.as_deref() == Some(NOTES_ASK)
                || m.tool_call_id.as_deref() == Some("notes_call")
                || m.tool_calls.iter().flatten().any(|call| call.id == "notes_call")
        };
        assert!(!messages.iter().chain(&archive).any(leaked), "the notes exchange was stored");
    }

    /// Without notes a cut the window does not force is not made: the
    /// history, the archive, the record and the hook are as they were.
    async fn assert_no_cut_without_notes(url: &str, spend: &mut RequestSpend) {
        let fixture = Fixture::new(url);
        let mut messages = session(60, 2_000);
        let original: Vec<String> = messages.iter().map(wire).collect();
        assert_eq!(fixture.cut(&mut messages, 20_000, 100_000, spend).await, 0);
        assert_eq!(messages.iter().map(wire).collect::<Vec<_>>(), original, "the history is whole");
        assert!(!fixture.committed(), "nothing of a cut was committed");
    }

    #[tokio::test]
    async fn a_reply_of_only_a_call_leaves_the_history_whole() {
        let (url, requests) = crate::client::tests::spawn_recording_once(reply("", true));
        assert_no_cut_without_notes(&url, &mut RequestSpend::new(None)).await;
        assert_eq!(body(&requests)["messages"].as_array().unwrap().last().unwrap()["content"], NOTES_ASK);
    }

    #[tokio::test]
    async fn an_endpoint_error_leaves_the_history_whole() {
        assert_no_cut_without_notes("http://127.0.0.1:9", &mut RequestSpend::new(None)).await;
    }

    #[tokio::test]
    async fn notes_the_spend_ceiling_refuses_leave_the_history_whole() {
        let (url, requests) = crate::client::tests::spawn_recording_once(reply("State: halfway.", false));
        let mut spend = RequestSpend::new(Some(1_000));
        assert_no_cut_without_notes(&url, &mut spend).await;
        assert!(requests.recv_timeout(Duration::from_millis(300)).is_err(), "the refused request is never sent");
        assert_eq!(spend.tokens, 0);
    }

    /// The notes request is admitted in real tokens by the session's learned
    /// factor, as every request is: a ceiling with room for its estimate but
    /// not for twice it refuses the request when the provider counts double.
    #[tokio::test]
    async fn the_spend_ceiling_admits_notes_by_the_learned_factor() {
        let (url, requests) = crate::client::tests::spawn_recording_once(reply("State: halfway.", false));
        let fixture = Fixture::new(&url);
        let mut messages = session(60, 2_000);
        let mut spend = RequestSpend::new(None);
        assert!(fixture.cut(&mut messages, 20_000, 100_000, &mut spend).await > 0);
        let sent = body(&requests)["messages"].as_array().unwrap().len();
        assert!(sent > 2 && spend.tokens > 0, "the notes request went and was charged");
        // What that request was admitted at, less the reply it was charged for.
        let estimated = spend.tokens - estimate_tokens("State: halfway.".len());

        let (url, _requests) = crate::client::tests::spawn_recording_once(reply("State: halfway.", false));
        let fixture = Fixture::new(&url);
        let mut messages = session(60, 2_000);
        let original: Vec<String> = messages.iter().map(wire).collect();
        assert!(sessions::save_messages(&fixture.core, &fixture.id, &messages, &mut 0, true));
        let mut spend = RequestSpend::new(Some(estimated * 3 / 2));
        let ctx = CompactionCtx { factor: 2.0, ..fixture.ctx() };
        assert_eq!(compact_messages(&ctx, &mut messages, 20_000, 100_000, 0, false, &mut spend).await.unwrap(), 0);
        assert_eq!(messages.iter().map(wire).collect::<Vec<_>>(), original, "the history is whole");
        assert!(!fixture.committed(), "nothing of a cut was committed");
        assert_eq!(spend.tokens, 0, "the refused request is never sent, so nothing is charged");
    }

    /// When the window forces the cut it is made without notes, and leaves
    /// the deterministic digest: the tools, the paths and the archive.
    #[tokio::test]
    async fn a_cut_the_window_forces_leaves_the_digest_without_notes() {
        let fixture = Fixture::new("http://127.0.0.1:9");
        let mut messages = session(60, 2_000);
        assert!(fixture.cut(&mut messages, 20_000, 20_000, &mut RequestSpend::new(None)).await > 0);
        let note = messages[pinned_head(&messages)].content.clone().unwrap();
        assert!(note.contains("Tools used: read_file.") && note.contains("Files touched: src/lib.rs."), "{note}");
        assert!(!note.contains("Summary:"), "{note}");
        assert!(note.contains(&sessions::archive_display(&fixture.core, &fixture.id)), "{note}");
        assert!(note.ends_with("files on disk are as the work so far left them."), "{note}");
        assert!(fixture.committed() && fixture.dir.join("hooked.json").exists(), "the archive, the record and the hook");
    }

    /// A transcript under its trigger is left alone and costs no request.
    #[tokio::test]
    async fn a_transcript_under_its_trigger_is_untouched() {
        let (url, requests) = crate::client::tests::spawn_recording_once(reply("State: halfway.", false));
        let fixture = Fixture::new(&url);
        let mut messages = session(30, 2_000);
        let original: Vec<String> = messages.iter().map(wire).collect();
        assert!(tokens(&messages) < 20_000);
        assert_eq!(fixture.cut(&mut messages, 20_000, 100_000, &mut RequestSpend::new(None)).await, 0);
        assert_eq!(messages.iter().map(wire).collect::<Vec<_>>(), original);
        assert!(requests.recv_timeout(Duration::from_millis(300)).is_err(), "no request");
        assert!(!fixture.committed());
    }

    /// Notes past 4,000 bytes are cut there, on a char boundary, with an
    /// ellipsis to say so.
    #[tokio::test]
    async fn long_notes_are_cut_on_a_char_boundary() {
        let (url, _requests) = crate::client::tests::spawn_recording_once(reply(&"€".repeat(1_500), false));
        let fixture = Fixture::new(&url);
        let mut messages = session(60, 2_000);
        assert!(fixture.cut(&mut messages, 20_000, 100_000, &mut RequestSpend::new(None)).await > 0);
        let note = messages[pinned_head(&messages)].content.clone().unwrap();
        // 1,333 three-byte chars are 3,999 bytes; the next would pass 4,000.
        assert!(note.contains(&format!("Summary: {}… ", "€".repeat(1_333))), "{note}");
        assert!(!note.contains(&"€".repeat(1_334)));
    }
}
