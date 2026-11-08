//! A session's live transcript: built from disk the first time a process
//! needs it, with the repairs a killed process can leave owing, taken by a
//! turn under a guard that puts it back, and saved as the turn goes.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::Arc;

use crate::config::ApprovalMode;
use crate::notes::{rehydrate_approval_mode, rehydrate_policy_notices};
use crate::prompt::{session_context, trailing_note, PromptBreakdown};
use crate::registry::Registry;
use crate::sessions;
use crate::state::{Core, SessionData};
use crate::types::ChatMessage;

/// The approval events a new session counts as already seen: every one on
/// record, so its first turn narrates only what happens after it starts.
/// Auto never asks the chain anything, so a session built there is seeded
/// by its first turn in another mode instead (`narrate_outside_approvals`).
fn seen_approval_events(core: &Core, project_root: &Path) -> Option<HashSet<u64>> {
    if core.approval_mode(project_root) == ApprovalMode::Auto {
        return None;
    }
    Some(
        crate::ledger::approval_events(&core.data_dir, project_root)
            .unwrap_or_default()
            .into_iter()
            .map(|e| e.id)
            .collect(),
    )
}

pub(crate) fn build_session_data(core: &Arc<Core>, session_id: &str, project_root: &Path) -> Result<SessionData, String> {
    sessions::ensure_owned(core, session_id)?;
    Ok(if let Some(mut messages) = sessions::load_messages(core, session_id)? {
        let count = messages.len();
        // The prompt lives at index 0. A transcript without it there is from
        // a build that predates the rule (none was ever released): give the
        // model its prompt, and the file is rewritten into that shape below.
        let needs_system = messages.first().map(|m| m.role.as_str()) != Some("system");
        // Resume: the generation the session last froze, its tools array,
        // and the record of what its persisted prompt indexes, from a
        // manifest whose record describes that prompt. Without one, nothing
        // says what the prompt indexes: the session is from an older shape,
        // its manifest write failed, or a transcript write failed after the
        // manifest's landed. The prompt and the registry are then rebuilt
        // from disk once, as at creation, and saved with the manifest, so
        // every later sitting sends the persisted bytes as they are.
        let persisted = messages.first().filter(|m| m.role == "system").and_then(|m| m.content.as_deref());
        let manifest = sessions::load_manifest(core, session_id)
            .filter(|m| persisted.is_some_and(|prompt| m.prompt_index.describes(prompt)));
        let rebuild_prompt = manifest.is_none();
        let (registry, note, breakdown) = match manifest {
            Some(manifest) => {
                let mut registry = Registry::from_manifest(manifest);
                registry.scan_memory(project_root);
                let (note, rows) = trailing_note(project_root, &registry);
                let prompt = persisted.unwrap_or_default();
                let breakdown = PromptBreakdown::from_persisted(prompt, &registry, project_root).with_note(rows);
                (registry, note, breakdown)
            }
            None => {
                let mut registry = Registry::build(&core.data_dir, project_root);
                let (prompt, note, breakdown) = session_context(project_root, &mut registry);
                if needs_system {
                    messages.insert(0, ChatMessage::system(prompt));
                } else {
                    messages[0] = ChatMessage::system(prompt);
                }
                sessions::save_manifest(core, session_id, &registry.to_manifest());
                (registry, note, breakdown)
            }
        };
        let registry = Arc::new(registry);
        let prompt_breakdown = Arc::new(breakdown);
        // A process that died mid-tool left a call on record with no reply,
        // and a provider refuses every request that carries one. Answer it
        // where it stands, including in a transcript that went on growing
        // past it, then drop the reply an earlier build may have saved for
        // it after those later prompts.
        let stubs = complete_pending_tool_replies(&mut messages, INTERRUPTED_TOOL_CALL);
        let strays = drop_stray_tool_replies(&mut messages);
        // Any edit rewrites the file into the current shape at once, so
        // every later save is a plain append. Then every replay boundary
        // follows the message it marked, edit by edit in the order they were
        // made; edits the file or the index cannot take yet wait on the
        // session for a later save. A crash between the rewrite and the
        // shifts leaves a divider a message off, which changes where a
        // replay draws it and nothing the model is sent.
        let edits: Vec<sessions::ResumeEdit> = needs_system
            .then_some(0)
            .into_iter()
            .chain(stubs)
            .map(|at| sessions::ResumeEdit::Insert(at as u64))
            .chain(strays.into_iter().map(|at| sessions::ResumeEdit::Remove(at as u64)))
            .collect();
        let (persisted_count, saved) = if edits.is_empty() && !rebuild_prompt {
            (count, true)
        } else {
            let mut persisted = 0usize;
            if sessions::save_messages(core, session_id, &messages, &mut persisted, true) {
                (persisted, true)
            } else {
                // The file still holds the transcript as it was. A count of
                // zero would make every later save refuse to append to it,
                // so no turn of this sitting would reach disk; a stale count
                // makes the next save rewrite the file whole, which lands the
                // repair too, and the boundaries move with it then.
                (sessions::PERSISTED_STALE, false)
            }
        };
        let reported_policy_notices = rehydrate_policy_notices(&messages);
        let reported_approval_mode = rehydrate_approval_mode(&messages);
        // Once per session means once per session's record: the note rides a
        // tool result, so a resumed session finds it in the transcript, or in
        // the compaction archive if a prune has since dropped that exchange,
        // and does not repeat it.
        let mut data = SessionData {
            messages,
            registry,
            trailing_note: note.map(Arc::from),
            prompt_breakdown,
            persisted_count,
            deferred_resume_edits: edits,
            take_seq: 0,
            schemas_over_budget_reported: false,
            reported_policy_notices,
            reported_approval_mode,
            seen_ledger_events: seen_approval_events(core, project_root),
            prompt_ratio: None,
        };
        land_deferred_resume_edits(core, session_id, &mut data, saved);
        data
    } else {
        // No transcript on disk: start fresh, honoring a manifest saved
        // before the session wrote its first message.
        let mut registry = match sessions::load_manifest(core, session_id) {
            Some(manifest) => {
                let mut registry = Registry::from_manifest(manifest);
                registry.scan_memory(project_root);
                registry
            }
            None => Registry::build(&core.data_dir, project_root),
        };
        let (prompt, note, breakdown) = session_context(project_root, &mut registry);
        // Always persisted (even builtin-only) so the extension fingerprint
        // and the record of what the prompt indexes travel with the session;
        // without them every resume would look like a self-modification and
        // re-freeze for nothing, and rebuild the prompt.
        sessions::save_manifest(core, session_id, &registry.to_manifest());
        SessionData {
            messages: vec![ChatMessage::system(prompt)],
            registry: Arc::new(registry),
            trailing_note: note.map(Arc::from),
            prompt_breakdown: Arc::new(breakdown),
            persisted_count: 0,
            deferred_resume_edits: Vec::new(),
            take_seq: 0,
            schemas_over_budget_reported: false,
            reported_policy_notices: Default::default(),
            reported_approval_mode: None,
            seen_ledger_events: seen_approval_events(core, project_root),
            prompt_ratio: None,
        }
    })
}

/// Load a session into the in-memory map if this process has not seen it yet.
/// Resuming with `-c` or `/resume` starts with an empty map, and the freeze
/// check only inspects sessions it can find there.
pub(crate) async fn ensure_session_hydrated(core: &Arc<Core>, session_id: &str, project_root: &Path) -> Result<(), String> {
    if core.sessions.lock().await.contains_key(session_id) {
        return Ok(());
    }
    let core_clone = core.clone();
    let session_id_owned = session_id.to_string();
    let project_root_owned = project_root.to_path_buf();
    let built = tokio::task::spawn_blocking(move || {
        build_session_data(&core_clone, &session_id_owned, &project_root_owned)
    }).await.map_err(|e| format!("session hydration failed: {e}"))??;
    core.sessions.lock().await.entry(session_id.to_string()).or_insert(built);
    Ok(())
}

/// The reply for a tool call whose result was never recorded: the process
/// died, or the turn unwound, after the call was persisted and before its
/// result was. Nothing that runs later knows whether the call ran.
pub(crate) const INTERRUPTED_TOOL_CALL: &str = "This call was interrupted before its result was recorded, so whether it ran and what it changed are unknown. Check before repeating it.";

/// Answer every tool call in the transcript that has no reply with a tool
/// message carrying `note`. Returns the index of each reply inserted, in
/// ascending order, for a caller that has to move replay boundaries with it.
///
/// A provider accepts an assistant message with `tool_calls` only when a reply
/// for each id follows it before any other message, and refuses every later
/// request that carries one unanswered id. The call is persisted before its
/// tool runs and the result only at iteration end, so a cancel, a truncated
/// reply whose calls were refused, a panic, or a killed process can each
/// leave one. A reply counts only inside the run of tool messages directly
/// after its call, and a missing one goes at the end of that run rather than
/// the end of the transcript: a history that kept growing past the gap is
/// answered where the gap is.
pub(crate) fn complete_pending_tool_replies(messages: &mut Vec<ChatMessage>, note: &str) -> Vec<usize> {
    let mut inserted = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let ids: Vec<String> = match &messages[i] {
            m if m.role == "assistant" => {
                m.tool_calls.iter().flatten().map(|c| c.id.clone()).collect()
            }
            _ => Vec::new(),
        };
        i += 1;
        if ids.is_empty() {
            continue;
        }
        let mut end = i;
        while messages.get(end).is_some_and(|m| m.role == "tool") {
            end += 1;
        }
        let missing: Vec<String> = {
            let answered: BTreeSet<&str> =
                messages[i..end].iter().filter_map(|m| m.tool_call_id.as_deref()).collect();
            ids.into_iter().filter(|id| !answered.contains(id.as_str())).collect()
        };
        for id in missing {
            messages.insert(end, ChatMessage::tool(id, note));
            inserted.push(end);
            end += 1;
        }
        i = end;
    }
    inserted
}

/// Remove every tool message that is not inside the run of tool messages
/// directly after a call it answers. Returns the index of each removal in
/// the order made, each counted after the removals before it.
///
/// Earlier builds answered only the last call's missing ids, at the end of
/// the transcript, so a cancelled or cut-short turn could save that reply
/// after prompts that followed an unanswered call. A provider refuses a tool
/// message no call precedes as surely as a call with no reply. Real results
/// always land in their call's run and a prune drops a call together with
/// its run, so what this removes is such a misplaced harness reply, once its
/// call has been answered where it stands. Hydration only: it rewrites the
/// file, while a turn's own saves append.
fn drop_stray_tool_replies(messages: &mut Vec<ChatMessage>) -> Vec<usize> {
    let mut removed = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        if m.role != "tool" {
            ids = match m.role.as_str() {
                "assistant" => m.tool_calls.iter().flatten().map(|c| c.id.clone()).collect(),
                _ => Vec::new(),
            };
            i += 1;
        } else if m.tool_call_id.as_ref().is_some_and(|id| ids.contains(id)) {
            i += 1;
        } else {
            messages.remove(i);
            removed.push(i);
        }
    }
    removed
}

/// Persist transcript to disk without cloning it back into SessionData.
/// The turn owns `messages` until `MessageGuard` commits on drop/finish.
/// True when the transcript is durable on disk after this call (see
/// `sessions::save_messages`).
pub(crate) async fn save_messages(core: &Arc<Core>, session_id: &str, messages: &[ChatMessage], rewrite: bool) -> bool {
    let mut sessions_map = core.sessions.lock().await;
    if let Some(data) = sessions_map.get_mut(session_id) {
        let saved = sessions::save_messages(core, session_id, messages, &mut data.persisted_count, rewrite);
        land_deferred_resume_edits(core, session_id, data, saved);
        saved
    } else {
        false
    }
}

/// Move the replay boundaries of a hydration repair once `saved` says the
/// file holds it: the first save after a failed hydration rewrite rewrites
/// the file whole (`sessions::PERSISTED_STALE`), and every save after that
/// keeps it. The edits stay pending until the index write that applies them
/// lands, so an index that refuses the write leaves them for the next save
/// to retry, instead of leaving each later divider a message early for good.
pub(crate) fn land_deferred_resume_edits(core: &Core, session_id: &str, data: &mut SessionData, saved: bool) {
    if saved
        && !data.deferred_resume_edits.is_empty()
        && sessions::apply_resume_edits(core, session_id, &data.deferred_resume_edits).is_ok()
    {
        data.deferred_resume_edits.clear();
    }
}

/// Process-unique ids for transcript takes. Starts at 1 so a freshly built
/// `SessionData` (`take_seq: 0`) can never match a live guard.
static TAKE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Take the transcript for a turn and stamp the session with a fresh take id.
/// The paired [`MessageGuard`] may only write back while the stamp matches.
pub(crate) fn take_messages(data: &mut SessionData) -> (Vec<ChatMessage>, u64) {
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
pub(crate) struct MessageGuard {
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
    pub(crate) fn new(core: Arc<Core>, session_id: &str, messages: Vec<ChatMessage>, take_seq: u64) -> Self {
        Self {
            core,
            session_id: session_id.to_string(),
            messages: Some(messages),
            take_seq,
        }
    }

    pub(crate) fn messages(&mut self) -> &mut Vec<ChatMessage> {
        self.messages.as_mut().expect("messages already committed")
    }

    /// Move the working transcript back into SessionData. Consumes the guard
    /// so `Drop` becomes a no-op.
    pub(crate) async fn commit(mut self) {
        if let Some(messages) = self.messages.take() {
            let mut map = self.core.sessions.lock().await;
            restore_if_current(&mut map, &self.session_id, self.take_seq, messages);
        }
    }
}

impl Drop for MessageGuard {
    fn drop(&mut self) {
        let Some(mut messages) = self.messages.take() else {
            return;
        };
        // No commit means an unwind (a panic, or the turn's task dropped),
        // possibly mid-tool: the call is on record and its reply is not. The
        // session stays live in this process, so hydration never sees this
        // transcript again; what goes back here is what the next request
        // sends, and an unanswered call would make the provider refuse it.
        // The only call left unanswered is the one this turn was running, so
        // the replies land at the tail, past everything on disk, and the
        // next turn's save appends them in order. Drop cannot rewrite the
        // file, so a reply placed any earlier would leave disk and memory
        // apart.
        complete_pending_tool_replies(&mut messages, INTERRUPTED_TOOL_CALL);
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
                    let session_id = self.session_id.clone();
                    let take_seq = self.take_seq;
                    let restoration = handle.spawn(async move {
                        let mut map = core.sessions.lock().await;
                        restore_if_current(&mut map, &session_id, take_seq, messages);
                    });
                    self.core.message_restorations.lock().unwrap()
                        .insert(self.session_id.clone(), restoration);
                }
                // No runtime means process teardown; disk saves bound the loss.
            }
        }
    }
}
