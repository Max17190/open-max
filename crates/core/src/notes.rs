//! The harness's notes: what it tells the model in the transcript, and the
//! frontend on the wire, about state the model did not produce (hook and
//! policy notices, approvals made elsewhere, the execution mode, settings
//! drift, what a mutating call changed that no refreeze covers, an
//! over-budget toolbox, a repeating call, a run of one-call responses).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use crate::agent::unapproved_capability;
use crate::config::ApprovalMode;
use crate::hooks::Hooks;
use crate::permissions::TurnPermissions;
use crate::registry::Registry;
use crate::state::Core;
use crate::types::{AgentEvent, ChatMessage, ToolCall};

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
fn insert_note(
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
    msgs.insert(at, ChatMessage::user(text.clone()));
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
        None => messages.push(ChatMessage::user(note)),
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
        msgs.push(ChatMessage::user(note));
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
    format!(
        "{REPEAT_NOTE_MARK}, {n} times in a row. If you are waiting on something, \
         keep waiting. If the task is complete, stop calling tools and reply with \
         a short plain-text summary now.]"
    )
}

/// Consecutive responses in a turn that each made one tool call, each a full request since every
/// request resends the conversation. The calls that complete runs of 4, 12 and 28 carry a note, at
/// most three per transcript (counted from the notes it holds, so one compaction dropped is not).
/// No call or several ends a run; a failed or timed-out call (a correction) is not counted.
#[derive(Default)]
pub(crate) struct SingleCallRuns {
    run: usize,
    notes: usize,
    edited: Option<String>,
}

/// The note's opening, so tests, readers and the count can find it.
pub(crate) const SINGLE_CALL_NOTE_MARK: &str = "[openmax: the last ";

impl SingleCallRuns {
    /// Starts a turn with the notes `messages`, the transcript, already holds.
    pub(crate) fn new(messages: &[ChatMessage]) -> Self {
        let notes = messages.iter().filter(|m| m.role == "tool").filter_map(|m| m.content.as_deref());
        Self { notes: notes.flat_map(str::lines).filter(|l| l.starts_with(SINGLE_CALL_NOTE_MARK)).count(), ..Self::default() }
    }

    /// Records one response's calls and the replies they drew; the note when it completes a run that earns one.
    pub(crate) fn observe(&mut self, calls: &[ToolCall], messages: &[ChatMessage]) -> Option<String> {
        let [call] = calls else { *self = Self { notes: self.notes, ..Self::default() }; return None };
        let args = serde_json::from_str::<serde_json::Value>(&call.function.arguments).ok();
        let path = args.as_ref().and_then(|a| a.get("path")?.as_str()).filter(|_| call.function.name == "edit_file");
        // The edit form names a file only while every step since the run began edited it, a failed one included.
        self.edited = if self.run == 0 { path.map(str::to_string) } else { self.edited.take().filter(|e| Some(e.as_str()) == path) };
        let reply = messages.iter().rev().find(|m| m.role == "tool").and_then(|m| m.content.as_deref());
        if reply.is_none_or(|r| r.starts_with("Error: ") || r.starts_with(crate::agent::APPROVAL_TIMEOUT_MESSAGE)) { return None; }
        self.run += 1;
        if self.run < *[4, 12, 28].get(self.notes)? { return None; }
        self.notes += 1;
        Some(single_call_note(self.run, self.edited.as_deref()))
    }
}

/// The note for a run of `n` one-call responses, naming `edited` when every response since it began edited that file.
pub(crate) fn single_call_note(n: usize, edited: Option<&str>) -> String {
    let ask = match edited {
        Some(path) => format!("one edit to {path}. When you know the file's remaining edits, send them together in one response"),
        None => "one tool call, and each resends the conversation. Send independent calls together, chain shell steps you \
                 already know in one command, and read whole files".into(),
    };
    format!("{SINGLE_CALL_NOTE_MARK}{n} responses each made {ask}.]")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{NOTE_HEADER, NOTE_SEPARATOR};

    fn call() -> ToolCall {
        ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: crate::types::ToolCallFunction { name: "write_file".into(), arguments: "{}".into() },
            extra_content: None,
        }
    }

    /// A call whose approval timed out ran nothing, so it counts toward a run
    /// of one-call responses no more than a failed one: the run holds across
    /// it. A reply that only mentions the note's text later in a line is not
    /// counted as a note the transcript holds.
    #[test]
    fn a_timed_out_approval_does_not_count_toward_a_one_call_run() {
        let reply = |text: &str| vec![ChatMessage::tool("c1", text)];
        let mut runs = SingleCallRuns::new(&reply(&format!("x {SINGLE_CALL_NOTE_MARK}2 responses")));
        let timed_out = reply(crate::agent::APPROVAL_TIMEOUT_MESSAGE);
        let seen: Vec<Option<String>> = [reply("ok"), reply("ok"), reply("ok"), timed_out, reply("ok")]
            .iter()
            .map(|messages| runs.observe(&[call()], messages))
            .collect();
        assert_eq!(seen, [None, None, None, None, Some(single_call_note(4, None))]);
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
            assert_eq!(json(&messages[history.len()..]), json(&[ChatMessage::user(first.clone())]));
        }
        let mut empty = round.clone();
        empty[3].content = None;
        store_note(&mut empty, Some(&first));
        assert_eq!(empty[3].content.as_deref(), Some(format!("{NOTE_SEPARATOR}{first}").as_str()), "a result with no text carries it");
    }
}
