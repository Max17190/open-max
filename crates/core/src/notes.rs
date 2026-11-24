//! The harness's notes: what it tells the model in the transcript, and the
//! frontend on the wire, about state the model did not produce (hook and
//! policy notices, approvals made elsewhere, the execution mode, settings
//! drift, what a mutating call changed that no refreeze covers, an
//! over-budget toolbox, a repeating call).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use crate::agent::unapproved_capability;
use crate::config::ApprovalMode;
use crate::hooks::Hooks;
use crate::permissions::TurnPermissions;
use crate::registry::Registry;
use crate::state::Core;
use crate::types::{estimate_tokens, AgentEvent, ChatMessage, ToolCall};

/// Forward observe-only hook failures to the frontend. The turn proceeds
/// (observe hooks are fail-open) but a broken observer is never silent.
pub(crate) fn report_hook_failures(
    core: &Arc<Core>,
    session_id: &str,
    failures: Vec<crate::hooks::HookFailure>,
) {
    for f in failures {
        core.send_agent(session_id, AgentEvent::HookFailed {
            hook: f.hook,
            event: f.event.to_string(),
            detail: f.detail,
        });
    }
}

/// One clause of a policy note about hook `f`: `'stem' on event<what>: detail`.
/// The stem and the detail are author-controlled and the clause sits in a
/// bracketed, `|`-joined note (a user-role message), so a newline in either
/// would forge a clause or close the note: both are flattened to one line.
pub(crate) fn hook_clause(f: &crate::hooks::HookFailure, what: &str) -> String {
    format!("'{}' on {}{what}: {}", crate::text::one_line(&f.hook), f.event, crate::text::one_line(&f.detail))
}

/// Insert a turn-start harness note just before the pending user prompt AND
/// emit it on the wire as a `HarnessNote`. The transcript half is what the
/// MODEL reads; the wire half is what a custom or interactive frontend reads,
/// so a protocol-v5 client can display the state change that affects the next
/// turn instead of it being model-only. The empty `call_id` marks
/// a note owned by no tool call, exactly as the policy-notice path does.
pub(crate) fn insert_startup_note(
    core: &Arc<Core>,
    session_id: &str,
    msgs: &mut Vec<ChatMessage>,
    text: String,
) {
    let at = msgs.len().saturating_sub(1);
    insert_note(core, session_id, msgs, at, text);
}

/// Insert a harness note as a user message at `at` and emit it on the wire,
/// as `insert_startup_note` does ahead of the pending prompt.
pub(crate) fn insert_note(
    core: &Arc<Core>,
    session_id: &str,
    msgs: &mut Vec<ChatMessage>,
    at: usize,
    text: String,
) {
    // The door, not each note. Every turn-start note is one bracketed clause
    // by construction and several are built from author-controlled bytes (an
    // approval record's capability path, a provider name read out of
    // providers.json, a hook stem and its parse reason). Flattening the
    // parameter here neutralizes all of them at once, and a note added later
    // inherits the rule instead of reintroducing the class.
    let text = crate::text::one_line(&text);
    msgs.insert(at, ChatMessage::note(text.clone()));
    core.send_agent(session_id, AgentEvent::HarnessNote { call_id: String::new(), text });
}

/// Append a harness note to the last tool message (where the MODEL reads it)
/// and emit it on the wire as a HarnessNote (where a custom FRONTEND reads
/// it): the receipts the model sees were invisible to any non-TUI client,
/// which saw only the bare tool output.
/// The `\n` prefix keeps the transcript readable; the wire note is clean.
pub(crate) fn append_and_emit_note(
    core: &Arc<Core>,
    session_id: &str,
    messages: &mut [ChatMessage],
    note: &str,
) {
    // The same door for every mid-turn note. The `\n` below is the harness's
    // own separator between the tool result and its note, so it is the note
    // that flattens, never the assembled content.
    let note = &crate::prompt::defuse_note_header(&crate::text::one_line(note)).into_owned();
    let call_id = messages
        .iter()
        .rev()
        .find(|m| m.role == "tool")
        .and_then(|m| m.tool_call_id.clone())
        .unwrap_or_default();
    if let Some(last) = messages.iter_mut().rev().find(|m| m.role == "tool") {
        let piece = format!("\n{note}");
        match &mut last.content {
            Some(content) => content.push_str(&piece),
            None => last.content = Some(note.to_string()),
        }
    }
    core.send_agent(session_id, AgentEvent::HarnessNote { call_id, text: note.to_string() });
}

/// Store `note`, what the session changed since its prompt was built, unless
/// the last one stored says the same: after `NOTE_SEPARATOR` at the end of
/// the last message when that is the user's prompt or a tool result, which
/// every endpoint takes, else as a user message of its own. Stored where it
/// changes and never moved, it is history every later request repeats, so a
/// prompt cache keeps it and the thinking a provider binds to it stays
/// valid. Each note carries every change since the prompt and replaces those
/// before it; once nothing differs, the header alone says so. Only user and
/// tool messages are searched, and tool output has the header defused.
pub(crate) fn store_note(messages: &mut Vec<ChatMessage>, note: Option<&str>) {
    let (header, note) = (crate::prompt::NOTE_HEADER, note.unwrap_or(crate::prompt::NOTE_HEADER));
    let carries = |m: &ChatMessage| m.role == "user" || m.role == "tool";
    let stored = messages.iter().rev().filter(|m| carries(m)).find_map(|m| m.content.as_deref().filter(|text| text.contains(header)));
    if stored.map_or(note == header, |text| text.ends_with(note)) {
        return;
    }
    match messages.last_mut().filter(|m| carries(m)) {
        Some(last) => last.content.get_or_insert_default().push_str(&format!("{}{note}", crate::prompt::NOTE_SEPARATOR)),
        None => messages.push(ChatMessage::note(note)),
    }
}

/// A settings.json drift note, when the file moved outside this process's
/// own writes. Settings are launch-frozen by design (base_url/api_key are
/// credential routing, approval_mode hot-adoption would be self-approval),
/// so an agent edit is inert for this process - and, unstated, that silence
/// cost real detours. Valid new bytes: state the launch-read rule and what
/// this session still runs. Invalid bytes: the brick warning (next launch
/// exits 2), delivered while the author can still repair the file.
pub(crate) fn settings_drift_note(core: &Arc<Core>) -> Option<String> {
    let changed = core.settings_disk_changed()?;
    Some(match changed {
        Ok(()) => {
            let (model, endpoint) = {
                let s = core.settings.lock().unwrap_or_else(|e| e.into_inner());
                let endpoint = s
                    .provider
                    .clone()
                    .filter(|p| !p.trim().is_empty())
                    .unwrap_or_else(|| s.base_url.clone());
                (s.model.clone(), endpoint)
            };
            format!(
                "[settings.json changed on disk. Settings are read at launch: this session \
                 still runs {model} via {endpoint}, and the new values apply from the next \
                 openmax launch. Switching this session's model is /model, which the user \
                 runs; the TUI rewrites settings.json on /model and settings changes, \
                 overwriting manual edits.]"
            )
        }
        Err(e) => format!(
            "[settings.json is now INVALID: {e}. The next launch of openmax will refuse to \
             start (exit 2) until the file parses; fix it now.]"
        ),
    })
}

/// The one canonical form of a policy notice. A notice is reported once per
/// session, and that "already said it" set does not persist on its own: on
/// resume it is rebuilt by parsing the notices back out of the transcript.
/// Identity therefore has to be taken on exactly the text that gets persisted,
/// and the text that gets persisted has to survive the round trip, which means
/// a notice may not spell either of the two things the transcript uses to
/// delimit it:
///
/// - a line break, which would forge a second clause in a user-role message
///   the model reads as the human's own words; and
/// - `NOTICE_JOINER`, which separates the notices inside one note. Rehydration
///   splits on that literal, so a notice containing it was restored as
///   fragments whose hashes matched nothing, and the session said it again
///   after every resume. Collapsing it needs no escape/unescape
///   machinery on this path: the delimiter simply cannot occur in a value.
///
/// Identity, persistence and display are all this one text, which is what
/// makes the once-per-session promise hold across a resume.
pub(crate) fn canonical_notice(notice: &str) -> String {
    crate::text::one_line(notice).replace(NOTICE_JOINER, " / ")
}

pub(crate) fn policy_notice_hash(notice: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    notice.hash(&mut h);
    h.finish()
}

pub(crate) const POLICY_NOTE_PREFIX: &str = "[policy notice: ";
const PERMISSION_NOTE_PREFIX: &str = "\n[permission notice: ";
pub(crate) const PERMISSION_NOTE_PREFIX_CLEAN: &str = "[permission notice: ";
pub(crate) const NOTICE_JOINER: &str = " | ";

/// Rebuild the reported-notice set from a persisted transcript, so a resumed
/// session (new process, hydrated messages) does not re-narrate every
/// still-applicable static notice: the transcript already carries them, and
/// once per SESSION means once across the session's whole life, not once
/// per process. Notes are split on the joiner they were assembled with, so
/// the hashes match what novel_policy_notices would compute.
pub(crate) fn rehydrate_policy_notices(messages: &[ChatMessage]) -> HashSet<u64> {
    let mut seen = HashSet::new();
    for message in messages {
        let Some(content) = message.content.as_deref() else { continue };
        let bodies: Vec<&str> = match message.role.as_str() {
            "user" => content
                .strip_prefix(POLICY_NOTE_PREFIX)
                .and_then(|s| s.strip_suffix(']'))
                .into_iter()
                .collect(),
            "tool" => content
                .split(PERMISSION_NOTE_PREFIX)
                .skip(1)
                .filter_map(|tail| tail.split(']').next())
                .collect(),
            _ => Vec::new(),
        };
        for body in bodies {
            for notice in body.split(NOTICE_JOINER) {
                seen.insert(policy_notice_hash(notice));
            }
        }
    }
    seen
}

/// Filter to policy notices this session has not yet narrated to the model,
/// and mark them reported. Identity is the full notice text - it embeds the
/// file path, which is the identity that matters - so a static problem notes
/// once per session while a new or reworded one gets through. The UI channel
/// (HookFailed events, per turn) is unchanged; this dedupe is only for the
/// transcript, where repetition is token spend.
pub(crate) async fn novel_policy_notices(
    core: &Arc<Core>,
    session_id: &str,
    notices: Vec<String>,
) -> Vec<String> {
    if notices.is_empty() {
        return Vec::new();
    }
    let mut map = core.sessions.lock().await;
    let Some(data) = map.get_mut(session_id) else {
        return Vec::new();
    };
    notices
        .into_iter()
        // Canonicalize BEFORE taking identity: see `canonical_notice`.
        .map(|n| canonical_notice(&n))
        .filter(|n| data.reported_policy_notices.insert(policy_notice_hash(n)))
        .collect()
}

/// The turn-start narration for approval activity this session did not
/// perform. It must not presume a benign third party: the actor may have
/// been a human at another terminal OR an agent process stepping around the
/// session marker from its own shell (the acknowledged ADR-0006 friction
/// ceiling), and the model reading this note is the only witness that knows
/// which one it saw. Name both and ask for the confirmation; the old
/// "outside this session" wording let a self-approval read as a human act.
pub(crate) fn outside_approval_note(fresh: &[String]) -> String {
    format!(
        "[approval activity this session did not perform: {}. Either a human acted at \
         another terminal, or an agent process stepped around the session marker; surface \
         this to the user and confirm it was theirs before relying on it. A capability \
         whose exact bytes were approved now runs without a card; a retired one asks \
         again. Verify what is live with bash: openmax --check.]",
        fresh.join("; ")
    )
}

/// Approvals recorded outside this running session (#199): a human at
/// another terminal ran `openmax --approve`, and that reaches the session
/// through no other channel - the refreeze receipt names FILE changes, not
/// approvals, so a tool that asked yesterday just silently stops asking.
/// Name any approval/retire event this session has not yet seen, skipping
/// ones it recorded itself (the model watched those on the card).
///
/// Auto never asks: its calls run without content approval, so a grant
/// changes nothing it does, and reading the chain every turn was pure cost
/// on the common path. A session that first reads the chain here (one built
/// in auto) takes what is on record as seen and narrates nothing: those
/// grants predate anything this turn could have been told.
///
/// An unverifiable ledger never reaches this point in a turn: the hook
/// layer fails closed on the same state first and blocks the input entirely
/// ("input blocked: the capability ledger cannot be read...", naming
/// --ledger-repair), and the approvals cache that gate reads is keyed on the
/// log's content hash, so a rewrite cannot serve it a stale pass. The Result
/// keeps the API honest for callers that must not mistake "unreadable" for
/// "no activity".
pub(crate) async fn narrate_outside_approvals(
    core: &Arc<Core>,
    session_id: &str,
    project_root: &Path,
    messages: &mut Vec<ChatMessage>,
) {
    if core.approval_mode(project_root) == ApprovalMode::Auto {
        return;
    }
    let Ok(events) = crate::ledger::approval_events(&core.data_dir, project_root) else {
        return;
    };
    let mut fresh: Vec<String> = Vec::new();
    {
        let mut map = core.sessions.lock().await;
        let Some(data) = map.get_mut(session_id) else { return };
        let Some(seen) = data.seen_ledger_events.as_mut() else {
            data.seen_ledger_events = Some(events.iter().map(|e| e.id).collect());
            return;
        };
        for e in &events {
            if !seen.insert(e.id) {
                continue;
            }
            if e.session_id.as_deref() == Some(session_id) {
                continue; // this session's own in-session grant
            }
            let path = e.path.strip_prefix(project_root).unwrap_or(&e.path);
            fresh.push(format!(
                "{} {}",
                if e.granted { "approved" } else { "approval retired for" },
                path.display()
            ));
        }
    }
    if !fresh.is_empty() {
        insert_startup_note(core, session_id, messages, outside_approval_note(&fresh));
    }
}

/// Tell the frontend, once per session, that the installed tool schemas
/// cost more than the window can spend. It holds on every
/// turn once it holds at all, so repeating it per turn would be noise; the
/// turn still runs, because this is the user's configuration to fix, not a
/// failure of this turn.
pub(crate) async fn report_schemas_over_budget(
    core: &Arc<Core>,
    session_id: &str,
    msgs: &mut Vec<ChatMessage>,
    schema_tokens: usize,
    budget_tokens: usize,
) {
    let first = {
        let mut map = core.sessions.lock().await;
        match map.get_mut(session_id) {
            Some(data) => !std::mem::replace(&mut data.schemas_over_budget_reported, true),
            // No live session state to dedupe against: say it rather than
            // swallow it.
            None => true,
        }
    };
    if first {
        core.send_agent(session_id, AgentEvent::SchemasOverBudget {
            schema_tokens,
            budget_tokens,
        });
        // The frontend is not the party that can act on this. The AGENT
        // installed these tools, `--spec usage` prices them one by one, and
        // retiring one is a file it can delete - but until now the advisory
        // existed only as a wire event, so the human saw it and the agent
        // never did. It goes in the transcript for the same reason every
        // other receipt does: the reader who can fix a thing has to be told.
        //
        // Written straight into the transcript rather than through
        // `insert_startup_note`, which would also emit a HarnessNote: the wire
        // already carries this as the structured SchemasOverBudget event, and
        // a note beside it renders the same advisory twice (the same reason
        // the post-approval receipt rides `outcome.output`).
        let note = crate::text::one_line(&format!(
            "[tool schemas cost ~{schema_tokens} tokens of the \
             ~{budget_tokens} this context window can spend on a request. They are resent \
             whole every request, so history compacts early and a long turn may not fit at \
             all. Price each tool with bash: openmax --spec usage, retire one you are not \
             using (delete its .toml; the tools array drops it at /reload or in a new \
             session), or raise context_tokens for this model.]"
        ));
        // APPENDED, never inserted before the tail. `insert_startup_note`
        // inserts at len-1 because at turn start the tail is the user's own
        // message; here the tail is `assistant(tool_calls)` followed by its
        // `tool` replies, and putting a user message between a tool call and
        // its reply is a transcript the provider rejects outright. This runs
        // at the top of an iteration, where the previous iteration's replies
        // are all appended, so the sequence is complete and the note lands
        // after it.
        msgs.push(ChatMessage::note(note));
    }
}

/// The mode as the model reads it. It goes out ahead of a session's first
/// prompt, and again only when the mode changes or a resumed session's
/// history no longer holds it, but the history resends it on every request
/// after, so every request pays for it like the system prompt: it names the
/// mode and its limits, and leaves content approval detail to the contracts
/// `openmax --spec` prints.
pub(crate) fn execution_policy_note(mode: ApprovalMode) -> String {
    let detail = match mode {
        ApprovalMode::Auto => "do authorized work without asking, extensions included; deny rules and validation still apply",
        ApprovalMode::Ask => "mutating calls need approval unless a permission allows them; unapproved tool content always does",
        ApprovalMode::Readonly => "mutating calls and calls needing approval are disabled",
    };
    format!("[execution policy: {}; {detail}.]", mode.as_str())
}

pub(crate) fn rehydrate_approval_mode(messages: &[ChatMessage]) -> Option<ApprovalMode> {
    let notes = [ApprovalMode::Auto, ApprovalMode::Ask, ApprovalMode::Readonly]
        .map(|mode| (mode, execution_policy_note(mode)));
    // Recover only the last visible notice. If wording changes or compaction
    // removed it, emit the current policy again. This never grants authority.
    messages.iter().rev()
        .filter(|m| m.role == "user" || m.role == "tool")
        .filter_map(|m| m.content.as_deref())
        .flat_map(|content| content.lines().rev())
        .find_map(|line| notes.iter().find(|(_, text)| line == text).map(|(mode, _)| *mode))
}

/// The second half of a refresh: tell the model about `current`, the mode
/// the first half read, once per change the session has not been told of.
pub(crate) async fn narrate_approval_policy(
    core: &Arc<Core>,
    session_id: &str,
    current: ApprovalMode,
    messages: &mut Vec<ChatMessage>,
) {
    // Keep provider tool-call/reply adjacency intact. A mode selected before
    // a batch has any reply is narrated at the next completed reply instead.
    let Some(last_role) = messages.last().map(|m| m.role.as_str()) else { return };
    let on_tool = last_role == "tool";
    if !on_tool && last_role != "user" { return; }
    let len = messages.len();
    let mut tail_saved = false;
    let changed = {
        let mut sessions = core.sessions.lock().await;
        sessions.get_mut(session_id).is_some_and(|session| {
            if session.reported_approval_mode == Some(current) { return false; }
            session.reported_approval_mode = Some(current);
            tail_saved = session.persisted_count == len;
            true
        })
    };
    if changed {
        let text = execution_policy_note(current);
        if on_tool {
            append_and_emit_note(core, session_id, messages, &text);
        } else {
            // Ahead of a pending prompt, which reaches disk with the reply. A
            // user message already saved (a steer, a turn_end refusal: each is
            // on disk before the request that carries it) is not moved: a save
            // appends from the persisted length, so a note put ahead of it
            // would write it to disk twice.
            let at = if tail_saved { len } else { len - 1 };
            insert_note(core, session_id, messages, at, text);
        }
    }
}

/// Consecutive iterations whose tool calls AND results are byte-identical.
///
/// Identical arguments alone say nothing (#225: a `wait`, a `git status`, a
/// log tail repeat verbatim while their output moves), so nothing here vetoes
/// a call; every call executes. What this tracks is the pair: the model asked
/// the same thing and the world answered the same way. That pair is how a
/// model that has finished fails to stop: its reasoning says the task is
/// complete, its reply is the same verification call again, with the same
/// output, until the iteration cap ends the turn. The shape is common enough
/// to end turns at the cap. From the third identical pair on, the tool
/// message carries [`repeated_result_note`]: advisory, counted, and it names
/// the two ways out.
#[derive(Default)]
pub(crate) struct RepeatTracker {
    last: Option<u64>,
    streak: usize,
}

/// The identical-pair count at which the note first appears.
pub(crate) const REPEAT_NOTE_AT: usize = 3;

/// The note's opening, so tests and readers can find it in a tool message.
pub(crate) const REPEAT_NOTE_MARK: &str = "[harness: identical call, identical output";

impl RepeatTracker {
    /// Record one finished iteration: its calls and the tool replies they
    /// drew (the trailing run of tool messages, one per call). Returns the
    /// length of the current run of identical iterations, this one included:
    /// 1 when it differs from the previous one.
    pub(crate) fn observe(&mut self, calls: &[ToolCall], messages: &[ChatMessage]) -> usize {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for call in calls {
            call.function.name.hash(&mut h);
            call.function.arguments.hash(&mut h);
        }
        let replies = messages.iter().rev().take(calls.len()).take_while(|m| m.role == "tool");
        for reply in replies {
            reply.content.hash(&mut h);
        }
        let key = h.finish();
        self.streak = if self.last == Some(key) { self.streak + 1 } else { 1 };
        self.last = Some(key);
        self.streak
    }
}

/// The advisory for the n-th identical call-and-result pair in a row. It
/// never claims the result cannot change (a poll can legitimately sit still
/// for longer than one call may run), so it offers the polling exit and the
/// finished exit and lets the model pick.
pub(crate) fn repeated_result_note(n: usize) -> String {
    format!("{REPEAT_NOTE_MARK}, {n} times in a row. If you are waiting on something, keep waiting. If the task is complete, stop calling tools and reply with a short plain-text summary now.]")
}

/// The opening of every request budget note (`Settings::request_budget`).
pub(crate) const BUDGET_NOTE_MARK: &str = "[harness: request budget:";

/// The percentages of the request budget whose crossing earns a note.
const BUDGET_MARKS: [usize; 4] = [50, 75, 90, 100];

/// The note that states the turn's soft request budget: before its prompt, or
/// on the round that set it from the `used` requests to the turn's first file
/// change and command (`Settings::request_budget_cycles`), with their mark.
#[inline(never)]
pub(crate) fn request_budget_start_note(budget: usize, used: usize, noted: &mut usize) -> String {
    let pace = if used == 0 { ".".into() } else { format!(", set from your own pace: {used} to your first file change and command.{}", budget_used(budget, used, noted).map_or(String::new(), |mark| format!(" {mark}"))) };
    format!("{BUDGET_NOTE_MARK} this task has a soft budget of {budget} model requests{pace} It is guidance, not a limit: plan to finish and verify the task within it.]")
}

/// The note a round of calls owes the budget after `used` requests: the
/// start of a budget set now at `paced` times them, at least one more, when
/// none is set yet, else a mark it reached. Kept out of line, so the loop's
/// frame does not grow with the formatting.
#[inline(never)]
pub(crate) fn round_budget_note(budget: &mut Option<usize>, paced: Option<f64>, used: usize, noted: &mut usize) -> Option<String> {
    if let Some(set) = paced.filter(|_| budget.is_none()).map(|cycles| ((cycles * used as f64).ceil() as usize).max(used + 1)) {
        *budget = Some(set);
        return Some(request_budget_start_note(set, used, noted));
    }
    request_budget_note((*budget)?, used, noted)
}

/// The note owed once `used` of `budget` requests reach a mark (`budget_used`).
pub(crate) fn request_budget_note(budget: usize, used: usize, noted: &mut usize) -> Option<String> {
    Some(format!("{BUDGET_NOTE_MARK} {} Prioritize finishing and verifying the task.]", budget_used(budget, used, noted)?))
}

/// The requests used and left once `used` of `budget` reach a mark past
/// `noted`, the highest one already said, which it then records. Marks
/// passed together earn one note, for the highest.
fn budget_used(budget: usize, used: usize, noted: &mut usize) -> Option<String> {
    let mark = BUDGET_MARKS.into_iter().rev().find(|&mark| used.saturating_mul(100) >= mark.saturating_mul(budget)).filter(|&mark| mark > *noted)?;
    *noted = mark;
    Some(format!("{used} of {budget} requests used ({}%), {} left.", used.saturating_mul(100) / budget.max(1), budget.saturating_sub(used)))
}

/// The budget mark owed before request `used + 1`: one the last request
/// reached with no call run (a reply cut before any call, a steer, a refused
/// end, a retried request), so no tool result carried its note. It goes on
/// the last tool result when the history ends with one, else as a note after
/// the last message, on disk before the request. After a round of calls the
/// mark was already said, so this says nothing. A paced budget's mark is
/// always a new note, since a request may already have carried that result.
pub(crate) async fn say_owed_budget_mark(core: &Arc<Core>, session_id: &str, messages: &mut Vec<ChatMessage>, budget: Option<usize>, used: usize, noted: &mut usize, paced: bool) {
    let Some(text) = budget.filter(|_| used > 0).and_then(|budget| request_budget_note(budget, used, noted)) else { return };
    // A tool result last was saved before the request that failed, so a
    // note added to it needs a rewrite to reach the disk.
    let rewrite = !paced && messages.last().is_some_and(|m| m.role == "tool");
    if rewrite { append_and_emit_note(core, session_id, messages, &text) } else { insert_note(core, session_id, messages, messages.len(), text) }
    crate::transcript::save_messages(core, session_id, messages, rewrite).await;
}

/// What a mutating call's permissions reload found, said on that call's
/// result: that it left permissions.toml malformed (`fail_closed_now`, the
/// parse reason, with the repair `permissions`, as reloaded, still allows),
/// and the allow rules that cannot fire until a human approves them (`inert`).
pub(crate) async fn narrate_permissions_reload(
    core: &Arc<Core>,
    session_id: &str,
    project_root: &Path,
    permissions: &TurnPermissions,
    fail_closed_now: Option<String>,
    inert: Vec<String>,
    messages: &mut [ChatMessage],
) {
    if let Some(reason) = fail_closed_now {
        let novel = novel_policy_notices(core, session_id, vec![format!("permissions fail-closed: {reason}")]).await;
        if !novel.is_empty() {
            // The repair carve-out returns Default for write/edit on this
            // file, but an earlier snapshot this turn may have explicitly
            // DENIED one of them, and the most restrictive answer wins - so
            // the receipt must not promise a repair tool a rule already
            // blocks. Restrictions also match the original argument, so
            // check the conventional spellings of the repair path.
            let abs = project_root.join(".openmax/permissions.toml").to_string_lossy().into_owned();
            let repair_denied = |tool: &str| {
                [abs.as_str(), ".openmax/permissions.toml", "./.openmax/permissions.toml"].iter().any(|p| {
                    matches!(
                        permissions.evaluate(tool, &serde_json::json!({ "path": p })),
                        crate::permissions::PermissionDecision::Deny { .. }
                    )
                })
            };
            let allowed: Vec<&str> = ["write_file", "edit_file"].into_iter().filter(|t| !repair_denied(t)).collect();
            let repair_clause = if allowed.is_empty() {
                "a rule earlier this turn denies write_file and edit_file on \
                 this file too, so even the repair waits for the START OF THE \
                 NEXT TURN; read_file is not allowed"
                    .to_string()
            } else {
                format!(
                    "{} on exactly this file is still allowed and takes effect \
                     at the START OF THE NEXT TURN, not this one; read_file is \
                     NOT, so rewrite it from what you intended",
                    allowed.join("/")
                )
            };
            let note = format!(
                "[permissions.toml is now malformed ({reason}). Every tool \
                 call is DENIED for the rest of THIS turn - policy snapshots \
                 only narrow within a turn, never widen - and your repair \
                 ({repair_clause}). \
                 openmax --check cannot run until then (bash is denied too).]"
            );
            append_and_emit_note(core, session_id, messages, &note);
        }
    }
    // The agent may have just written itself an allow rule that can never
    // fire (#180). Say so NOW, on this call's result, with the command that
    // lifts it - not silently at the next surprise prompt.
    let novel = novel_policy_notices(core, session_id, inert).await;
    if !novel.is_empty() {
        let note = format!("{PERMISSION_NOTE_PREFIX_CLEAN}{}]", novel.join(NOTICE_JOINER));
        append_and_emit_note(core, session_id, messages, &note);
    }
}

/// What a mutating call can change that no refreeze receipt covers, as the
/// turn last saw it, so the call that changes it hears so on its own result
/// while this turn can still act on it.
pub(crate) struct ChangeWatch {
    /// The provider catalog's content identity: an agent bash edit to
    /// ~/.openmax/providers.json is otherwise silent (the file is outside the
    /// project root, so no refreeze covers it), and a malformed edit would
    /// surface turns later as an unrelated-looking resolve error.
    providers: Option<u64>,
    hooks: u64,
    approvals: std::collections::HashMap<String, Option<String>>,
}

impl ChangeWatch {
    pub(crate) fn new(core: &Core, registry: &Registry, project_root: &Path, canonical_root: &Path) -> Self {
        Self {
            providers: crate::providers::providers_status(&core.data_dir).content_hash,
            hooks: crate::hooks::hooks_fingerprint(&core.data_dir, project_root),
            approvals: approval_state(core, registry, project_root, canonical_root),
        }
    }

    /// After a refreeze, whose receipt already narrated manifest-level
    /// approval changes: resync so the per-call diff does not repeat them.
    pub(crate) fn resync_approvals(&mut self, core: &Core, registry: &Registry, project_root: &Path, canonical_root: &Path) {
        self.approvals = approval_state(core, registry, project_root, canonical_root);
    }

    /// After an executed mutating call: whatever it changed since the watch
    /// last looked, as a note on its result.
    pub(crate) fn after_mutation(
        &mut self,
        core: &Arc<Core>,
        session_id: &str,
        project_root: &Path,
        canonical_root: &Path,
        registry: &Registry,
        messages: &mut [ChatMessage],
    ) {
        // A providers.json edit (bash: the file is outside the project root)
        // gets a receipt too: what loaded, when it applies - or that the
        // catalog is EMPTY, while the author can still fix it.
        let status = crate::providers::providers_status(&core.data_dir);
        if status.content_hash != self.providers {
            self.providers = status.content_hash;
            let note = match &status.parse_error {
                Some(err) => format!(
                    "[providers.json changed but is invalid JSON: {err}. The \
                     provider catalog is EMPTY until this parses; any turn using \
                     settings.provider will fail at its start.]"
                ),
                None => format!(
                    "[providers.json changed: {} provider(s) ({}). Endpoint \
                     resolution happens at turn start, so a provider or model \
                     switch applies from the next turn; switching this session's \
                     model is /model, which the user runs.]",
                    status.count,
                    status.names.join(", ")
                ),
            };
            append_and_emit_note(core, session_id, messages, &note);
        }
        // A script edit revokes the approval of the tool that runs it, but
        // moves no fingerprint: no refreeze, no receipt - and the model
        // learned only when the next call raised a card. Diff approval state
        // per call.
        let approval_now = approval_state(core, registry, project_root, canonical_root);
        let mut revoked: Vec<String> = Vec::new();
        for (name, state) in &approval_now {
            if let (Some(path), Some(None)) = (state, self.approvals.get(name)) {
                revoked.push(format!(
                    "{name} (openmax --approve {})",
                    crate::doctor::shell_quote(std::path::Path::new(path))
                ));
            }
        }
        self.approvals = approval_now;
        if !revoked.is_empty() {
            revoked.sort();
            let note = format!(
                "[approval revoked: the code these tools run changed since a human \
                 approved it, so the next call of each stops for a card: {}. Re-approve \
                 the exact bytes, or prove them first with openmax --check --run-examples.]",
                revoked.join(", ")
            );
            append_and_emit_note(core, session_id, messages, &note);
        }
        // A hook file written this call: hooks are outside the extension
        // fingerprint, so this write got no refreeze receipt and the
        // inertness notice would otherwise land a whole turn later, leaving
        // the agent believing its gate was live for a whole turn.
        let hooks_now = crate::hooks::hooks_fingerprint(&core.data_dir, project_root);
        if hooks_now != self.hooks {
            self.hooks = hooks_now;
            let discovered = Hooks::discover_for_mode(project_root, &core.data_dir, core.approval_mode_canonical(canonical_root));
            let inert: Vec<String> = discovered.notices().iter().map(|f| hook_clause(f, "")).collect();
            // An APPROVED gate whose bytes just changed is not inert - it
            // fails closed and blocks every tool call from the next turn
            // until restored or re-approved. Say that, not "applies next
            // turn".
            let note = if core.approval_mode_canonical(canonical_root) == ApprovalMode::Auto {
                let mut note = match discovered.fail_closed_reason() {
                    Some(reason) => format!("[hook files changed. Invalid or unavailable hooks block tools from the next turn until repaired: {reason}"),
                    None => "[hook files changed. Valid hooks apply from the next turn (in print mode, the run's next -p prompt) under auto without content approval; a run started now from bash (openmax -p ...) runs them too.".into(),
                };
                // Named on this write, while this turn can fix it, not at
                // the hook's first run next turn.
                let unstartable = discovered.cannot_start(project_root);
                if !unstartable.is_empty() {
                    if !note.ends_with('.') {
                        note.push('.');
                    }
                    note.push_str(&format!(" These cannot start as written: {}", unstartable.join(" | ")));
                }
                note + "]"
            } else if let Some(blocked) = discovered.fail_closed_reason() {
                format!(
                    "[hook files changed. A live gate is now failing closed: \
                     {blocked}. From the next turn every tool call is BLOCKED \
                     until the file is restored to its approved bytes or a human \
                     re-approves it (openmax --approve <path>, at a terminal \
                     outside any session).]"
                )
            } else if inert.is_empty() {
                "[hook files changed. Hooks are discovered at turn start; \
                 approved hooks apply from the next turn.]"
                    .to_string()
            } else {
                format!(
                    "[hook files changed. Not running: {}. Hooks activate only \
                     after a human runs openmax --approve <path> at a terminal \
                     outside any session; until then the guarantee is NOT in \
                     force - say so if asked, do not claim it.]",
                    inert.join(" | ")
                )
            };
            append_and_emit_note(core, session_id, messages, &note);
        }
        // A settings.json edit (bash: outside the project root) is inert for
        // this session and bricks the next launch when malformed - answered
        // on the writing call, while it is still repairable.
        if let Some(note) = settings_drift_note(core) {
            append_and_emit_note(core, session_id, messages, &note);
        }
    }
}

/// Which external tools currently lack an approval covering their exact
/// bytes (manifest + bound code): each one's name, with its manifest's path
/// while it does. Diffed after every mutating call so a script edit - which
/// moves no extension fingerprint and so earns no refreeze receipt - is still
/// announced the moment it revokes an approval. Empty under auto: auto runs
/// every external tool without content approval, so there is no revocation
/// to announce, and the map costs a read of the chain per external tool on
/// every mutating call. A diff across a mode switch starts from empty and so
/// announces nothing; the next call's card (or its absence) is accurate.
fn approval_state(
    core: &Core,
    registry: &Registry,
    project_root: &Path,
    canonical_root: &Path,
) -> std::collections::HashMap<String, Option<String>> {
    if core.approval_mode_canonical(canonical_root) == ApprovalMode::Auto {
        return std::collections::HashMap::new();
    }
    registry
        .tools
        .iter()
        .filter(|s| matches!(s.kind, crate::registry::ToolKind::External(_)))
        .map(|s| {
            let state = unapproved_capability(registry, &core.data_dir, project_root, &s.name).map(|u| u.path);
            (s.name.clone(), state)
        })
        .collect()
}


/// How to get content larger than the reply limit across, as the refusal of a
/// call the limit cut off (`cut_off_call_error`) and the note after a
/// call that never arrived both give it.
pub(crate) const SPLIT_ADVICE: &str = "Send large content in parts: for a file, write_file the first part, then append the rest with edit_file (old_string is the last few lines you wrote, enough to be unique; new_string is those lines followed by the next part).";

/// The reply for a call the output cap cut off partway through its
/// arguments. A bare parse error reads as a typo to fix, and the model sends
/// the same oversized call again. The advice spells out how edit_file
/// appends, because its old_string has to be unique. With `room`, the limit
/// the next request alone carries, the call is asked for whole.
pub(crate) fn cut_off_call_error(max_tokens: usize, room: Option<usize>) -> String {
    let advice = room.map_or(SPLIT_ADVICE.into(), |room| format!("Your next reply alone may be up to {room} tokens: send this call again now, whole."));
    format!("Not run: your reply reached the output limit (max_tokens = {max_tokens}) while writing this call, so its arguments were cut off. {advice}")
}

/// What a reply the output limit cut before any tool call was writing, by
/// what its usage accounts for. A provider that sends a tool call only when
/// it is complete drops a call the limit cuts, so the reply can arrive with
/// nothing in it although it spent the whole limit.
///
/// - `Lost`: with a reasoning count, the unaccounted output exceeds an eighth
///   of the limit (256 tokens at least). Without a count, readable output
///   must account for less than a quarter of the billed output, with no
///   opaque reasoning attached, and the same minimum gap applies. A route that hides most of its reasoning
///   can still look like a lost call. Readable reasoning counts at four
///   bytes per token and answer text counts twice to allow for dense text.
/// - `Unseen`: no reasoning count, and nothing at all arrived: either
///   thinking that is not shown or a call larger than the limit.
/// - `Text`: what arrived accounts for the output, and the text in it is
///   more than that same margin: the reply was cut writing its answer.
/// - `Thought`: what arrived accounts for the output and is thinking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CutKind { Thought, Text, Lost, Unseen }

pub(crate) fn cut_kind(max_tokens: usize, reply: &crate::client::CompletionResult) -> CutKind {
    let thought = [reply.reasoning_content.as_deref(), reply.reasoning.as_deref()].into_iter().flatten().max_by_key(|text| text.len()).unwrap_or_default();
    let no_unread_reasoning = || reply.reasoning_details.as_deref().is_none_or(|raw| crate::client::details_repeat_plaintext(raw, thought, false));
    let margin = (max_tokens as u64 / 8).max(256);
    let text = estimate_tokens(reply.content.len()) as u64;
    let arrived = |reasoning: u64| reasoning.max(estimate_tokens(thought.len()) as u64).saturating_add(2 * text);
    match reply.usage.map(|u| (u.output_tokens, u.reasoning_tokens)) {
        Some((output, Some(reasoning))) if output.saturating_sub(arrived(reasoning)) > margin => CutKind::Lost,
        Some((output, None)) if (!thought.is_empty() || !reply.content.is_empty()) && arrived(0).saturating_mul(4) < output && output.saturating_sub(arrived(0)) > margin && no_unread_reasoning() => CutKind::Lost,
        None | Some((_, None)) if reply.content.is_empty() && thought.is_empty() && reply.reasoning_details.is_none() => CutKind::Unseen,
        _ if text > margin => CutKind::Text,
        _ => CutKind::Thought,
    }
}

/// Recognize intent and return a recovery note, never a tool call.
#[cold]
pub(crate) fn written_call_recovery(text: &str, schemas: &serde_json::Value) -> Option<String> {
    use serde_json::Value;
    use std::sync::LazyLock;
    let text = text.trim();
    let mut names = schemas.as_array()?.iter().map(crate::registry::entry_name);
    if let Some(tagged) = text.strip_suffix("</tool_call>") {
        let (prefix, body) = tagged.split_once("<tool_call>")?;
        if !prefix.is_empty() && (!prefix.ends_with('\n') || !prefix.starts_with(char::is_alphanumeric) || !body.starts_with(['\n', '\r']) || prefix.contains("```") || prefix.contains("~~~")
            || prefix.split_whitespace().last().is_some_and(|word| word.eq_ignore_ascii_case("example:"))) {
            return None;
        }
        // The frame identifies the call. Parsing the body only selects the note.
        static CALLEE: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r#"^\{\s*("(?:[^"\\]|\\.)*")\s*:\s*("(?:[^"\\]|\\.)*")"#).unwrap());
        let body = body.trim();
        let (_, [key, name]) = CALLEE.captures(body)?.extract();
        if serde_json::from_str::<String>(key).ok()? != "name" { return None; }
        let name: String = serde_json::from_str(name).ok()?;
        if names.clone().any(|offered| offered == name) {
            return Some(written_call_note(serde_json::from_str::<Value>(body).is_ok()));
        }
        let available = names.collect::<Vec<_>>().join(", ");
        return Some(format!("[tool call written as text: nothing ran; that tool was not offered, and the available tools are {available}. Use the API if the task requires a tool.]"));
    }
    let json = if let Some(fence) = text.strip_prefix("```") {
        let (info, body) = fence.split_once('\n')?;
        if !matches!(info.trim(), "" | "json") {
            return None;
        }
        body.strip_suffix("\n```")?.trim()
    } else if text.starts_with('{') {
        text
    } else {
        return None;
    };
    let object: Value = serde_json::from_str(json).ok()?;
    let fields = object.as_object()?;
    let args = fields.get("arguments").or_else(|| fields.get("parameters"))?;
    let name = fields.get("name")?.as_str()?;
    (fields.len() == 2 && args.is_object() && names.any(|offered| offered == name)).then(|| written_call_note(true))
}

#[cold]
fn written_call_note(valid: bool) -> String {
    let issue = if valid { "" } else { " The arguments were not valid JSON." };
    format!("[tool call written as text: nothing ran.{issue} Only call the tool through the API with valid JSON arguments if the user's task requires running it. Otherwise return the requested answer without running anything.]")
}

/// The note after a reply the output limit cut before any tool call. The
/// limit bounds how long the model thinks before it acts, so a cut thought
/// is told to act. A cut inside a call or text needed room, not brevity:
/// with `room`, the limit the next request alone carries, the note says so
/// and asks for the content whole. Without it (that request was cut too, or
/// the limit leaves no more room) the note asks for the content in parts,
/// and cut text gets the note a cut thought does, byte for byte. A cut
/// thought with `room`, the raised limit after cuts in a row, is told it.
pub(crate) fn reply_cut_note(max_tokens: usize, kind: CutKind, room: Option<usize>) -> String {
    let next = |ask: &str| room.map(|room| format!("The next reply alone may be up to {room} tokens: {ask}"));
    match kind {
        CutKind::Lost => match next("send that call again now, whole.") {
            Some(next) => format!("[reply cut: it reached max_tokens ({max_tokens}) while writing something that did not arrive here, most likely one tool call larger than the limit; nothing of it ran. {next}]"),
            None => format!("[reply cut: it reached max_tokens ({max_tokens}) while writing something that did not arrive here, most likely one tool call larger than the limit; nothing of it ran, and written again whole it will be cut the same way. {SPLIT_ADVICE}]"),
        },
        CutKind::Unseen => match next("if it was a call, send it again now, whole.") {
            Some(next) => format!("[reply cut: it reached max_tokens ({max_tokens}) and nothing of the reply arrived here: either thinking that is not shown, or a tool call larger than the limit, which did not run. {next}]"),
            None => format!("[reply cut: it reached max_tokens ({max_tokens}) and nothing of the reply arrived here: either thinking that is not shown, or a tool call larger than the limit, which did not run. If it was a call, written again whole it will be cut the same way. {SPLIT_ADVICE}]"),
        },
        CutKind::Text => format!("[reply cut: it reached max_tokens ({max_tokens}) while writing text. {}]", next("continue the text from exactly where it stopped, without repeating any of it.").unwrap_or_else(|| "Continue the text from exactly where it stopped, without repeating any of it.".into())),
        CutKind::Thought => match room {
            Some(room) => format!("[reply cut: it reached max_tokens ({max_tokens}) before any tool call; the next reply may be up to {room} tokens: continue from where it stopped, then act.]"),
            None => format!("[reply cut: it reached max_tokens ({max_tokens}) before any tool call; continue from where it stopped: think briefly, then act.]"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{NOTE_HEADER, NOTE_SEPARATOR};

    #[test]
    fn recognizes_only_call_shaped_replies() {
        let registry = Registry::builtin_only();
        let call = r#"{"name":"bash","arguments":{"command":"echo hello"}}"#;
        let wrapped = format!("<tool_call>\n{call}\n</tool_call>");
        let broken = "<tool_call>\n{\"name\":\"bash\",\"arguments\":{\"command\":\"echo\nhello\"}}\n</tool_call>";
        let open_array = "<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"paths\":[\"a.txt\",2\n</tool_call>";
        for (reply, expected) in [
            (format!("  {call} \n"), Some(true)),
            (format!("```json\r\n{call}\r\n```"), Some(true)),
            (format!("```\n{call}\n```"), Some(true)),
            (r#"{"parameters":{},"name":"bash"}"#.into(), Some(true)),
            (format!("  {wrapped}\n"), Some(true)),
            (format!("First, inspect the file:\n{wrapped}"), Some(true)),
            (broken.into(), Some(false)),
            (open_array.into(), Some(false)),
            (format!("First, inspect the files:\n{open_array}"), Some(false)),
            (format!("First, inspect the files:\n{open_array}\nA final sentence."), None),
            (format!("For EXAMPLE:\n{wrapped}"), None),
            (format!("First, inspect the file:\n{broken}"), Some(false)),
            (format!("Example: {wrapped}"), None),
            (format!("Example:\n<tool_call>{call}</tool_call>"), None),
            (format!("Example:\n```xml\n{wrapped}"), None),
            (format!("Example:\n~~~xml\n{wrapped}"), None),
            (format!("Example:\n```json\n{call}\n```"), None),
            (format!("Example:\n{call}"), None),
            (format!("{wrapped} done"), None),
            (format!("{call} done"), None),
            (format!("```JSON\n{call}\n```"), None),
            (format!("```json\n{call}\n````"), None),
            (format!("<tool_call>{call}"), None),
            (format!("<call>{call}</call>"), None),
            (format!("{call}\n{wrapped}"), None),
            (format!("[{call}]"), None),
            (r#"{"name":"other","arguments":{"name":"bash","arguments":{}}}"#.into(), None),
            (r#"{"name":"bash","options":{}}"#.into(), None),
            (r#"{"name":"bash","arguments":"{}"}"#.into(), None),
            (r#"{"name":"bash","arguments":{},"example":true}"#.into(), None),
            (r#"{"name":"bash","arguments":{},}"#.into(), None),
            (r#"<tool_call>{"arguments":{},"name":"bash",}</tool_call>"#.into(), None),
            ("   ".into(), None),
        ] {
            assert_eq!(written_call_recovery(&reply, registry.tool_schemas_json()), expected.map(written_call_note), "{reply:?}");
        }
    }

    #[test]
    fn tagged_calls_do_not_require_well_formed_arguments() {
        let registry = Registry::builtin_only();
        let garbage = r#"{"name":"bash","arguments":{missing quotes <fragment> } " { </fragment>}}"#;
        let string_args = r#"{"name":"bash","arguments":"echo hello"}"#;
        let tag = |body: &str| format!("<tool_call>\n{body}\n</tool_call>");
        let valid = tag(r#"{"name":"bash","arguments":{}}"#);
        for (reply, expected) in [
            (tag(garbage), Some(false)),
            (format!("Example:\n{}", tag(garbage)), None),
            (format!("Here is a malformed example.\n{}", tag(garbage)), Some(false)),
            (format!("```json\n{garbage}\n```"), None),
            (garbage.into(), None),
            (tag(string_args), Some(true)),
            (tag(r#"{"name":"b\u0061sh","arguments":{}}"#), Some(true)),
            (tag(r#"{"na\u006de":"bash","arguments":{}}"#), Some(true)),
            (tag(r#"{"na\u006de":"bash" anything goes"#), Some(false)),
            (tag(r#"{"not_name":"bash","name":"bash","arguments":{}}"#), None),
            (tag(r#"{"name":"bash" anything goes"#), Some(false)),
            (format!("```json\n{string_args}\n```"), None),
            (format!("{valid}\n{valid}"), Some(false)),
        ] {
            assert_eq!(written_call_recovery(&reply, registry.tool_schemas_json()), expected.map(written_call_note), "{reply:?}");
        }
    }

    /// The note after a reply cut before any call follows what its usage
    /// accounts for, and is the note every cut got before whenever what
    /// arrived accounts for the output.
    #[test]
    fn a_cut_note_says_when_the_output_did_not_arrive() {
        use crate::client::{CompletionResult, Usage};
        let before = "[reply cut: it reached max_tokens (4096) before any tool call; continue from where it stopped: think briefly, then act.]";
        let usage = |output: u64, reasoning: Option<u64>| Some(Usage { output_tokens: output, reasoning_tokens: reasoning, ..Default::default() });
        let encrypted = || Some(serde_json::value::RawValue::from_string(format!(r#"[{{"type":"reasoning.encrypted","data":"{}"}}]"#, "e".repeat(40_000))).unwrap());
        let reply = |content: &str, thought: Option<&str>, usage| CompletionResult {
            content: content.into(), reasoning: thought.map(Into::into), finish_reason: "length".into(), usage, ..Default::default()
        };
        let long_thought = "t".repeat(16_000);
        let cases = [
            // A call dropped whole: the limit spent, one reasoning token, nothing arrived.
            ("a call that never arrived", reply("", None, usage(4_096, Some(1))), "lost"),
            // Text, then a call that never arrived: 1,223 output beyond 2,873 reasoning, 117 bytes shown.
            ("text, then a lost call", reply(&"x".repeat(117), Some(&"t".repeat(12_192)), usage(4_096, Some(2_873))), "lost"),
            // Reasoning that streamed and was counted.
            ("streamed reasoning", reply("", Some(&long_thought), usage(4_096, Some(4_096))), "before"),
            // Reasoning counted but not shown.
            ("hidden counted reasoning", reply("", None, usage(4_096, Some(4_090))), "before"),
            // Reasoning that streamed while the count says less.
            ("reasoning longer than its count", reply("", Some(&long_thought), usage(4_096, Some(10))), "before"),
            // Text that arrived accounts for the output.
            ("text that arrived", reply(&"x".repeat(16_000), None, usage(4_096, Some(0))), "text"),
            // A few unaccounted tokens are not a lost call.
            ("within the margin", reply(&"x".repeat(15_000), None, usage(4_096, Some(0))), "text"),
            // Dense text that did arrive: 4,096 tokens at three bytes a token.
            ("dense text that arrived", reply(&"x".repeat(12_288), None, usage(4_096, Some(0))), "text"),
            // Some text, then a call that never arrived.
            ("short text, then a lost call", reply(&"x".repeat(2_000), None, usage(4_096, Some(0))), "lost"),
            // Encrypted reasoning, its count small: its bytes do not hide the lost call.
            ("encrypted reasoning, then a lost call", CompletionResult { reasoning_details: encrypted(), ..reply("", None, usage(4_096, Some(20))) }, "lost"),
            // Encrypted reasoning counted in full: as before.
            ("encrypted counted reasoning", CompletionResult { reasoning_details: encrypted(), ..reply("", None, usage(4_096, Some(4_096))) }, "before"),
            // Encrypted reasoning arrived with no count: something arrived, as before.
            ("encrypted reasoning, no count", CompletionResult { reasoning_details: encrypted(), ..reply("", None, usage(4_096, None)) }, "before"),
            // No count, nothing arrived: thinking not shown, or a lost call.
            ("nothing arrived, no count", reply("", None, usage(4_096, None)), "unshown"),
            ("nothing arrived, no usage", reply("", None, None), "unshown"),
            // No count, something arrived: as before.
            ("text, no count", reply("so far", None, None), "before"),
            ("short reasoning, no count", reply("", Some("half a thought"), usage(4_096, None)), "lost"),
            ("below a quarter, no count", reply(&"x".repeat(1_536), Some(&"t".repeat(1_020)), usage(4_096, None)), "lost"),
            ("at a quarter, no count", reply(&"x".repeat(1_536), Some(&"t".repeat(1_024)), usage(4_096, None)), "before"),
            ("below a quarter, counted", reply("", Some(&"t".repeat(4_092)), usage(4_096, Some(3_584))), "before"),
            ("at a quarter, counted", reply("", Some(&"t".repeat(4_096)), usage(4_096, Some(3_583))), "lost"),
        ];
        for (case, reply, want) in cases {
            // With no room to offer, a cut thought and a lost call get the notes they got before, and cut text is
            // asked to go on from where it stopped.
            let kind = cut_kind(4_096, &reply);
            let note = reply_cut_note(4_096, kind, None);
            let got = if note == before { "before" } else if note == "[reply cut: it reached max_tokens (4096) while writing text. Continue the text from exactly where it stopped, without repeating any of it.]" { "text" } else if note.contains("did not arrive here") { "lost" } else if note.contains("nothing of the reply arrived") { "unshown" } else { "other" };
            assert_eq!(got, want, "{case}: {note}");
            if want != "before" && want != "text" {
                assert!(note.starts_with("[reply cut: it reached max_tokens (4096)") && note.ends_with(&format!("{SPLIT_ADVICE}]")), "{case}: {note}");
            }
            // With room, a cut thought, raised after cuts in a row, is told the next limit and to act, and anything
            // else is asked for whole, not in parts.
            let roomy = reply_cut_note(4_096, kind, Some(16_384));
            match kind {
                CutKind::Thought => assert_eq!(roomy, "[reply cut: it reached max_tokens (4096) before any tool call; the next reply may be up to 16384 tokens: continue from where it stopped, then act.]", "{case}"),
                CutKind::Text => assert_eq!(roomy, "[reply cut: it reached max_tokens (4096) while writing text. The next reply alone may be up to 16384 tokens: continue the text from exactly where it stopped, without repeating any of it.]", "{case}"),
                CutKind::Lost | CutKind::Unseen => assert!(roomy.contains("The next reply alone may be up to 16384 tokens") && roomy.ends_with("again now, whole.]") && !roomy.contains("in parts"), "{case}: {roomy}"),
            }
            assert_eq!(kind == CutKind::Text, want == "text", "{case}");
        }
    }

    /// A note at each of 50, 75, 90 and 100 percent, once each, and none
    /// before the first or after the last; marks passed in one step earn one
    /// note, for the highest.
    #[test]
    fn request_budget_notes_fall_on_their_marks_once() {
        let mut noted = 0;
        let said: Vec<(usize, String)> =
            (1..=25).filter_map(|used| request_budget_note(20, used, &mut noted).map(|note| (used, note))).collect();
        assert_eq!(said.iter().map(|(used, _)| *used).collect::<Vec<_>>(), [10, 15, 18, 20]);
        assert_eq!(said[0].1, format!("{BUDGET_NOTE_MARK} 10 of 20 requests used (50%), 10 left. Prioritize finishing and verifying the task.]"));
        assert_eq!(said[3].1, format!("{BUDGET_NOTE_MARK} 20 of 20 requests used (100%), 0 left. Prioritize finishing and verifying the task.]"));
        let mut noted = 0;
        let small: Vec<String> = (1..=5).filter_map(|used| request_budget_note(2, used, &mut noted)).collect();
        assert_eq!(small.len(), 2, "50% at 1, then 75, 90 and 100 together at 2: {small:?}");
        assert!(small[1].contains("2 of 2 requests used (100%)"), "{small:?}");
    }

    fn call() -> ToolCall {
        ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: crate::types::ToolCallFunction { name: "write_file".into(), arguments: "{}".into() },
            extra_content: None,
        }
    }

    /// A note is stored once, where it changes: at the end of the newest
    /// tool result or prompt, adding no message, and as a user message of its
    /// own only after an assistant reply or the system prompt alone. The same
    /// note again stores nothing; a changed one is stored after the earlier,
    /// which stays; once nothing differs, the header alone supersedes it. A
    /// copy of the header in an assistant reply is not a stored note.
    #[test]
    fn a_note_is_stored_once_where_it_changes() {
        let (first, second) = (format!("{NOTE_HEADER}\n\n- wordfreq: top"), format!("{NOTE_HEADER}\n\n- lint: lint"));
        let round = vec![
            ChatMessage::system("p"),
            ChatMessage::user("hi"),
            ChatMessage::assistant(None, Some(vec![call()])),
            ChatMessage::tool("c1", "wrote it"),
        ];
        let json = |messages: &[ChatMessage]| serde_json::to_string(messages).unwrap();
        let mut messages = round.clone();
        store_note(&mut messages, None);
        assert_eq!(json(&messages), json(&round), "no note, nothing stored");
        store_note(&mut messages, Some(&first));
        assert_eq!(messages.len(), round.len(), "and adds no message");
        assert_eq!(messages[3].content.as_deref(), Some(format!("wrote it{NOTE_SEPARATOR}{first}").as_str()));
        let stored = messages.clone();
        store_note(&mut messages, Some(&first));
        assert_eq!(json(&messages), json(&stored), "the same note again stores nothing");

        messages.push(ChatMessage::assistant(Some(format!("quoting {second}")), None));
        messages.push(ChatMessage::user("next"));
        store_note(&mut messages, Some(&second));
        assert_eq!(json(&messages[..4]), json(&stored), "the earlier note stays where it was stored");
        assert_eq!(messages[5].content.as_deref(), Some(format!("next{NOTE_SEPARATOR}{second}").as_str()), "it ends the prompt");
        store_note(&mut messages, None);
        assert!(messages[5].content.as_deref().unwrap().ends_with(&format!("{second}{NOTE_SEPARATOR}{NOTE_HEADER}")), "the header alone");
        let settled = messages.clone();
        store_note(&mut messages, None);
        assert_eq!(json(&messages), json(&settled), "which nothing differing keeps");

        for history in [vec![ChatMessage::system("p")], vec![ChatMessage::user("hi"), ChatMessage::assistant(Some("done".into()), None)]] {
            let mut messages = history.clone();
            store_note(&mut messages, Some(&first));
            assert_eq!(messages.len(), history.len() + 1, "a message of its own after {history:?}");
            assert_eq!(json(&messages[history.len()..]), json(&[ChatMessage::note(first.clone())]));
        }
        let mut empty = round.clone();
        empty[3].content = None;
        store_note(&mut empty, Some(&first));
        assert_eq!(empty[3].content.as_deref(), Some(format!("{NOTE_SEPARATOR}{first}").as_str()), "a result with no text carries it");
    }
}
