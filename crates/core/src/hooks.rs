//! Process lifecycle hooks: optional external commands that gate or observe
//! agent lifecycle events. `pre_tool_use` and `user_prompt_submit` can block
//! (nonzero exit); `post_tool_use`, `session_start` (a session's first turn),
//! `compaction` (context was pruned), and `turn_end` (stop reason; fires even
//! on cancel) observe only. Empty discovery costs almost nothing (one
//! directory list). Hooks never change tool schemas and never inject text
//! into the model.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::execution::{
    self, CaptureSpec, ProcessError, ProcessRequest, StdinMode, Termination,
};
use crate::state::CancelToken;
use std::sync::Arc;

const DEFAULT_TIMEOUT_SECS: u64 = 10;
const MAX_TIMEOUT_SECS: u64 = 60;
/// Per-event hook cap. Every pre_tool_use hook runs on every tool call, so an
/// unbounded set turns discovery mistakes into unbounded per-call latency.
/// Stems sort deterministically; the head runs, --check names the rest.
pub const MAX_HOOKS_PER_EVENT: usize = 32;
const MAX_REASON_CHARS: usize = 500;
/// How much tool output a `post_tool_use` hook is handed. Enough for an eval
/// or telemetry hook to work with, small enough that copying it to every
/// matching hook on every tool call stays cheap.
const MAX_OUTPUT_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
    SessionStart,
    Compaction,
    TurnEnd,
}

impl HookEvent {
    /// Every event, for diagnostics that group hooks per event.
    pub(crate) const ALL: [HookEvent; 6] = [
        HookEvent::PreToolUse,
        HookEvent::PostToolUse,
        HookEvent::UserPromptSubmit,
        HookEvent::SessionStart,
        HookEvent::Compaction,
        HookEvent::TurnEnd,
    ];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            HookEvent::PreToolUse => "pre_tool_use",
            HookEvent::PostToolUse => "post_tool_use",
            HookEvent::UserPromptSubmit => "user_prompt_submit",
            HookEvent::SessionStart => "session_start",
            HookEvent::Compaction => "compaction",
            HookEvent::TurnEnd => "turn_end",
        }
    }

    /// Whether this event gates calls. Dropping a gate is fail-open, which is
    /// why the approved content's event, not the current file's, decides how a
    /// modified hook is treated.
    pub(crate) fn is_gate(self) -> bool {
        matches!(self, HookEvent::PreToolUse | HookEvent::UserPromptSubmit)
    }

    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "pre_tool_use" => Some(HookEvent::PreToolUse),
            "post_tool_use" => Some(HookEvent::PostToolUse),
            "user_prompt_submit" => Some(HookEvent::UserPromptSubmit),
            "session_start" => Some(HookEvent::SessionStart),
            "compaction" => Some(HookEvent::Compaction),
            "turn_end" => Some(HookEvent::TurnEnd),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HookSpec {
    /// sha256 of the defining TOML's bytes: the identity the content-bound
    /// approval store keys on.
    pub source_sha256: String,
    pub event: HookEvent,
    pub command: String,
    pub args: Vec<String>,
    pub timeout_secs: u64,
    /// When set, the hook only runs for this tool name.
    pub tool_filter: Option<String>,
    pub source_path: PathBuf,
    /// The project-local files this hook hands to the host (its `command`,
    /// plus any `args` naming a file inside the project), each with the
    /// sha256 a human approved. Filled by the approval filter; re-checked
    /// immediately before every run, because discovery happens once per turn
    /// and the agent can rewrite a script between two calls of one turn.
    pub(crate) bound_code: Vec<(PathBuf, String)>,
}

/// One observe-only hook run that failed (spawn error, nonzero exit, or
/// timeout). Observe hooks stay fail-open - the turn proceeds - but the
/// failure is returned so the frontend can say so instead of silence.
#[derive(Clone, Debug)]
pub struct HookFailure {
    /// File stem, the hook's identity.
    pub hook: String,
    pub event: &'static str,
    pub detail: String,
}

fn failure(hook: &HookSpec, detail: String) -> HookFailure {
    notice(hook.source_path.clone(), hook.event.as_str(), detail)
}

/// The same report, for a hook file that never got as far as running.
fn notice(path: PathBuf, event: &'static str, detail: String) -> HookFailure {
    HookFailure {
        hook: path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string(),
        event,
        detail,
    }
}

/// Hooks discovered for the current project. Loaded once per agent turn.
#[derive(Clone, Debug, Default)]
pub struct Hooks {
    pre: Vec<HookSpec>,
    post: Vec<HookSpec>,
    user_prompt: Vec<HookSpec>,
    session_start: Vec<HookSpec>,
    compaction: Vec<HookSpec>,
    turn_end: Vec<HookSpec>,
    /// Hook files that exist, were live once, and no longer parse, as (path,
    /// reason). A file that fails to parse might have been a gate, so tool
    /// execution fails closed until it is fixed or removed (same policy as
    /// permissions). A broken file no human ever approved is not here: it
    /// never ran, so it removes no policy and must not brick the project.
    invalid: Vec<(PathBuf, String)>,
    /// Gate hooks a human installed whose content no longer matches what was
    /// approved. Dropping one silently is how a comment or a rewritten script
    /// turns a human gate off, so these fail closed instead.
    revoked_gates: Vec<(PathBuf, String)>,
    /// Hook files a human approved that are no longer on disk. A deleted file
    /// parses into nothing at all, so it is found by reconciling the approved
    /// paths rather than by reading the directory. Its event is unknowable
    /// once the file is gone, which is exactly the position an unparseable
    /// file leaves us in, so it gets the same fail-closed answer.
    missing: Vec<(PathBuf, String)>,
    /// Gate hooks a human approved, still holding their approved bytes, that
    /// this discovery cannot run: another file took their stem, or they sit
    /// past the per-event cap. Neither is a change to the hook, so nothing
    /// above sees them - and both end with a human's gate not gating, which
    /// is the one outcome that must never be silent.
    not_running: Vec<(PathBuf, String)>,
    /// Every hook path this discovery actually read, one per stem. What is
    /// missing from it is what the loader never reached, which is how a
    /// shadowed file is told from a deleted one.
    considered: Vec<PathBuf>,
    /// Why the approval store could not be read, when it could not. Every
    /// bucket above keys on what a human approved, so with the ledger
    /// unreadable there is no verdict to sort by: an approved gate cannot be
    /// told from content nobody blessed. The error fires exactly when the
    /// chain's tamper detection works (a rewritten record, a partial line
    /// from an interrupted append, a deleted log with a surviving pin), so
    /// it gets the revoked-gate answer - nothing runs, tools block - never
    /// the inert one.
    ledger_error: Option<String>,
    /// Files whose rewrite is exempt from the fail-closed block: the broken or
    /// revoked hook files themselves and the code they name. Same repair path
    /// permissions already has, for the same reason - a hook the agent can
    /// break must stay fixable from inside the session.
    repair_paths: Vec<PathBuf>,
    /// Hooks that exist but are not live, reported once per turn instead of
    /// vanishing: content no human approved, or a revoked observe hook.
    notices: Vec<HookFailure>,
}

/// First stem wins: project dirs are listed before global, and that
/// precedence covers parse errors too. A malformed file that is shadowed by
/// an earlier valid one was never going to run, so it must not fail closed;
/// a malformed file that holds its stem blocks instead of silently falling
/// back to the definition it shadows.
fn discover_in_dirs(dirs: &[PathBuf]) -> Hooks {
    let mut by_stem: std::collections::BTreeMap<String, HookSpec> = std::collections::BTreeMap::new();
    let mut invalid_by_stem: std::collections::BTreeMap<String, (PathBuf, String)> =
        std::collections::BTreeMap::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if stem.is_empty()
                || by_stem.contains_key(&stem)
                || invalid_by_stem.contains_key(&stem)
            {
                continue;
            }
            match parse_hook_file(&path) {
                Ok(spec) => {
                    by_stem.insert(stem, spec);
                }
                Err(reason) => {
                    invalid_by_stem.insert(stem, (path, reason));
                }
            }
        }
    }
    let invalid: Vec<(PathBuf, String)> = invalid_by_stem.into_values().collect();
    let considered = by_stem
        .values()
        .map(|s| s.source_path.clone())
        .chain(invalid.iter().map(|(p, _)| p.clone()))
        .collect();
    let mut hooks = Hooks { invalid, considered, ..Hooks::default() };
    for spec in by_stem.into_values() {
        match spec.event {
            HookEvent::PreToolUse => hooks.pre.push(spec),
            HookEvent::PostToolUse => hooks.post.push(spec),
            HookEvent::UserPromptSubmit => hooks.user_prompt.push(spec),
            HookEvent::SessionStart => hooks.session_start.push(spec),
            HookEvent::Compaction => hooks.compaction.push(spec),
            HookEvent::TurnEnd => hooks.turn_end.push(spec),
        }
    }
    // BTreeMap iteration is stem-sorted, so each event list is deterministic.
    // The cap is applied later, over the approved set only: see `apply_cap`.
    hooks
}

impl Hooks {
    /// Discover hooks under project `.openmax/hooks/` then global
    /// `~/.openmax/hooks/`. Project entries with the same file stem win.
    ///
    /// Every file is read exactly once and the policy is parsed from those
    /// same bytes, so what a turn enforces is always a generation that
    /// existed on disk. Nothing is cached between calls: a cache key has to
    /// be computed from a read of its own, and a file edited between that
    /// read and the parse would store a policy under the fingerprint of a
    /// different one, leaving a gate that never runs while its file sits on
    /// disk. Discovery is one directory list per turn, so there is nothing
    /// worth that risk.
    pub fn discover(project_root: &Path, data_dir: &Path) -> Self {
        Self::discover_dirs(project_root, data_dir, &hook_dirs(project_root))
    }

    /// The same discovery against an explicit dir list - the list the
    /// approval reconciliation judges shadowing against.
    fn discover_dirs(project_root: &Path, data_dir: &Path, dirs: &[PathBuf]) -> Self {
        let mut hooks = discover_in_dirs(dirs);
        hooks.retain_approved(data_dir, project_root, dirs);
        hooks
    }

    /// Keep only hooks whose exact content a human approved - the TOML *and*
    /// the project-local code it runs. Hooks run with host authority on every
    /// matching call with no per-invocation gate, so they are the one
    /// capability file that is inert until approved: `openmax --approve
    /// <path>`, a human act from outside any agent turn. An in-session
    /// write approval approves the write and nothing more - the card shows
    /// a clipped preview, and a preview is not shown bytes.
    ///
    /// What happens to the rest depends on whether the path was ever live,
    /// because the two cases are opposites. Content nobody approved never
    /// ran, so dropping it removes no policy: it stays inert and is reported.
    /// Content at a path a human did approve is a *modification* of a live
    /// hook, and dropping a gate is how an edit turns a human gate off, so a
    /// revoked gate fails closed until it is restored or re-approved.
    fn retain_approved(&mut self, data_dir: &Path, project_root: &Path, dirs: &[PathBuf]) {
        let approvals = match crate::ledger::approvals(data_dir, project_root) {
            Ok(approvals) => approvals,
            Err(reason) => {
                // An unreadable ledger is a detected state, not an empty
                // store. Defaulting here would reclassify every approved
                // gate as "unapproved and inert" - the one bucket that does
                // not block - so one appended byte in log.jsonl would turn
                // every human gate off. Nothing unverifiable runs, tool
                // execution fails closed, and there is no repair carve-out:
                // the fix lives outside the project (openmax
                // --ledger-repair), not in any file the agent writes.
                for list in [
                    &mut self.pre,
                    &mut self.post,
                    &mut self.user_prompt,
                    &mut self.session_start,
                    &mut self.compaction,
                    &mut self.turn_end,
                ] {
                    list.clear();
                }
                // Loud at turn start, and pointedly not `--approve`: that
                // command reads the same broken chain and fails with the
                // same error, so prescribing it would name a dead end.
                self.notices.push(HookFailure {
                    hook: "capability ledger".into(),
                    event: "all events",
                    detail: format!(
                        "no hook approval can be verified, failing closed until the ledger is repaired (openmax --ledger-repair): {reason}"
                    ),
                });
                self.ledger_error = Some(reason);
                return;
            }
        };
        let mut revoked_gates = Vec::new();
        let mut repair_paths = Vec::new();
        let mut notices = Vec::new();
        for list in [
            &mut self.pre,
            &mut self.post,
            &mut self.user_prompt,
            &mut self.session_start,
            &mut self.compaction,
            &mut self.turn_end,
        ] {
            list.retain_mut(|spec| {
                let code = crate::ledger::bound_code(&spec.command, &spec.args, project_root);
                if approvals.contains(&spec.source_sha256) && approvals.covers_code(&code) {
                    spec.bound_code = code
                        .into_iter()
                        .filter_map(|c| Some((c.path, c.sha256?)))
                        .collect();
                    return true;
                }
                let approved = approvals.approved_hook(&spec.source_path);
                let mut reason = if approvals.contains(&spec.source_sha256) {
                    let problem = code
                        .iter()
                        .find_map(|c| c.problem(&approvals))
                        .unwrap_or_default();
                    format!("{}: the code it runs, {problem}", spec.source_path.display())
                } else {
                    format!("{}: its content changed since it was approved", spec.source_path.display())
                };
                // Classify by what a human approved, never by what the file
                // now claims to be. Reading `event` off the modified content
                // would let an approved `pre_tool_use` gate rewrite itself
                // into an observer and stop gating - deletion's trick in
                // another form, demotion instead of removal. A path that was
                // live but whose shape is not remembered is treated as a
                // gate: the conservative answer to a question we cannot ask.
                let was_gate = approved.map(|a| a.is_gate()).unwrap_or(true);
                if let Some(approved) = approved {
                    if approved.is_gate() && !spec.event.is_gate() {
                        reason = format!(
                            "{}: an approved {} gate was rewritten as a {} hook, which would stop it gating",
                            spec.source_path.display(),
                            approved.event(),
                            spec.event.as_str()
                        );
                    }
                }
                if approvals.was_live(&spec.source_path) {
                    // The exemption follows the approved content too: deriving
                    // it from the current `command` would let a rewritten hook
                    // hand itself a write exemption for a path nobody blessed.
                    repair_paths.push(spec.source_path.clone());
                    repair_paths.extend(approved.into_iter().flat_map(|a| a.code_paths()));
                    if was_gate {
                        revoked_gates.push((spec.source_path.clone(), reason));
                        return false;
                    }
                    notices.push(notice(spec.source_path.clone(), spec.event.as_str(), reason));
                } else {
                    notices.push(notice(
                        spec.source_path.clone(),
                        spec.event.as_str(),
                        format!(
                            "unapproved and inert: a human must approve this exact content with `openmax --approve {}`",
                            spec.source_path.display()
                        ),
                    ));
                }
                false
            });
        }
        // A broken file that was never approved was never running, so it can
        // remove no policy - and failing closed on one would let any write
        // brick the project, including the write that would repair it.
        self.invalid.retain(|(path, reason)| {
            if approvals.was_live(path) {
                repair_paths.push(path.clone());
                // A file that does not parse names nothing, so the code it
                // used to run comes from the approval too.
                if let Some(approved) = approvals.approved_hook(path) {
                    repair_paths.extend(approved.code_paths());
                }
                return true;
            }
            notices.push(notice(
                path.clone(),
                "invalid",
                format!("unapproved and unparseable, so it never loaded: {reason}"),
            ));
            false
        });
        // Reconcile against what a human approved, not against what discovery
        // found. A deleted hook file yields neither a spec nor an invalid
        // entry, so nothing above ever sees it - and deleting a gate is
        // strictly easier than rewriting one. Absence gets the same answer as
        // a modification: the policy is not running, so nothing runs.
        //
        // A file that is still there but that discovery never read is the
        // same outcome by a different route: another file holds its stem, or
        // its directory could not be listed. Occupation is deletion's trick
        // without the delete - the approved bytes sit untouched on disk while
        // the gate stops running - so it is caught here too, where the
        // question is what a human installed rather than what a directory
        // now contains.
        let considered: Vec<PathBuf> = std::mem::take(&mut self.considered);
        for path in approvals.live_paths() {
            if !manifest_in_dirs(path, dirs) {
                continue;
            }
            let carve_out = |repair_paths: &mut Vec<PathBuf>| {
                repair_paths.push(path.clone());
                if let Some(approved) = approvals.approved_hook(path) {
                    repair_paths.extend(approved.code_paths());
                }
            };
            if !path.exists() {
                carve_out(&mut repair_paths);
                self.missing.push((
                    path.clone(),
                    format!("{}: an approved hook file was deleted", path.display()),
                ));
                continue;
            }
            if considered.iter().any(|c| same_file(c, path)) {
                continue;
            }
            let winner = stem_of(path)
                .and_then(|stem| considered.iter().find(|c| stem_of(c) == Some(stem)));
            // A winner a human also approved is precedence they built: a
            // project hook overriding a global one is the documented way to
            // do exactly that, so it is not displacement and says nothing.
            if winner.is_some_and(|w| self.is_live(w)) {
                continue;
            }
            let reason = match winner {
                Some(winner) => format!(
                    "{}: an approved hook is shadowed by {}, which holds its file stem and is not itself approved and running",
                    path.display(),
                    winner.display()
                ),
                None => format!(
                    "{}: an approved hook file is on disk but discovery never read it; check that its directory is readable",
                    path.display()
                ),
            };
            carve_out(&mut repair_paths);
            // The occupier is what has to change for the gate to come back,
            // so the repair carve-out has to reach it. It grants nothing:
            // writing there cannot approve anything, so the only content
            // that ends the block is content a human already blessed.
            if let Some(winner) = winner {
                repair_paths.push(winner.clone());
            }
            let was_gate = approvals.approved_hook(path).map(|a| a.is_gate()).unwrap_or(true);
            if was_gate {
                self.not_running.push((path.clone(), reason));
            } else {
                notices.push(notice(path.clone(), "shadowed", reason));
            }
        }
        self.revoked_gates = revoked_gates;
        self.repair_paths = repair_paths;
        self.notices = notices;
        self.apply_cap();
    }

    /// Whether this exact path is one of the hooks that came through the
    /// approval filter and will run this turn.
    fn is_live(&self, path: &Path) -> bool {
        self.events().into_iter().flatten().any(|s| same_file(&s.source_path, path))
    }

    fn events(&self) -> [&Vec<HookSpec>; 6] {
        [
            &self.pre,
            &self.post,
            &self.user_prompt,
            &self.session_start,
            &self.compaction,
            &self.turn_end,
        ]
    }

    /// Trim each event to the per-call cap, over the approved set only.
    ///
    /// Every `pre_tool_use` hook runs on every tool call, so the cap is a
    /// latency bound and has to stay. What it must not be is a lever the
    /// agent can pull: applied to the raw directory listing it ranked files
    /// nobody approved alongside a human's gate, so 32 junk manifests with
    /// earlier-sorting stems switched that gate off - the files doing the
    /// evicting never running themselves. Ranking the approved set instead
    /// keeps the same bound and leaves the ordering entirely in the hands of
    /// the party that approves things.
    ///
    /// Over-cap gates are policy a human installed that is not running, so
    /// they fail closed; over-cap observers are reported.
    fn apply_cap(&mut self) {
        let mut not_running = Vec::new();
        let mut notices = Vec::new();
        for list in [
            &mut self.pre,
            &mut self.post,
            &mut self.user_prompt,
            &mut self.session_start,
            &mut self.compaction,
            &mut self.turn_end,
        ] {
            if list.len() <= MAX_HOOKS_PER_EVENT {
                continue;
            }
            for spec in list.split_off(MAX_HOOKS_PER_EVENT) {
                let reason = format!(
                    "{}: beyond the {MAX_HOOKS_PER_EVENT}-hook cap for {}, so it does not run; consolidate that event's hooks, or retire this one with `openmax --forget {}`",
                    spec.source_path.display(),
                    spec.event.as_str(),
                    spec.source_path.display()
                );
                if spec.event.is_gate() {
                    not_running.push((spec.source_path.clone(), reason));
                } else {
                    notices.push(notice(spec.source_path, spec.event.as_str(), reason));
                }
            }
        }
        self.not_running.extend(not_running);
        self.notices.extend(notices);
    }

    /// Hooks that exist but did not load, for the frontend to show. Inert is
    /// not silent: a hook the human expects to be running and that is not has
    /// to say so every turn, not wait for `openmax --check`.
    pub fn notices(&self) -> Vec<HookFailure> {
        self.notices.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.pre.is_empty()
            && self.post.is_empty()
            && self.user_prompt.is_empty()
            && self.session_start.is_empty()
            && self.compaction.is_empty()
            && self.turn_end.is_empty()
            && self.invalid.is_empty()
            && self.revoked_gates.is_empty()
            && self.missing.is_empty()
            && self.not_running.is_empty()
            && self.ledger_error.is_none()
    }

    /// Non-empty when a hook a human installed stopped being enforceable: its
    /// file no longer parses, or its content no longer matches what was
    /// approved. Tool execution blocks on either, because both mean a gate the
    /// user wrote down is not running, and running on without it would drop
    /// that policy silently.
    fn fail_closed_reason(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(reason) = self.ledger_fail_closed_reason() {
            parts.push(reason);
        }
        if !self.invalid.is_empty() {
            parts.push(format!(
                "invalid hook file(s), failing closed until fixed or removed (see openmax --check): {}",
                describe(&self.invalid)
            ));
        }
        if !self.revoked_gates.is_empty() {
            parts.push(format!(
                "approved gate hook(s) changed and no longer load, failing closed until the approved content is restored or a human re-approves it (openmax --approve <path>): {}",
                describe(&self.revoked_gates)
            ));
        }
        if !self.missing.is_empty() {
            parts.push(format!(
                "approved hook file(s) were deleted, failing closed until the file is restored or a human retires it (openmax --forget <path>): {}",
                describe(&self.missing)
            ));
        }
        if !self.not_running.is_empty() {
            parts.push(format!(
                "approved gate hook(s) still hold their approved content but cannot run, failing closed until they can (see openmax --check): {}",
                describe(&self.not_running)
            ));
        }
        (!parts.is_empty()).then(|| parts.join("; "))
    }

    /// The one fail-closed state that also blocks prompt submission, kept in
    /// one place so both gates give the same reason. Broken or revoked hook
    /// files keep submission open on purpose: their repair carve-out writes
    /// project files from inside a turn, so a turn must be able to start.
    /// The ledger is repaired from the shell, outside any turn, and an
    /// approved user_prompt_submit gate not running means the text reaches
    /// the model endpoint and the transcript, which no later block undoes.
    fn ledger_fail_closed_reason(&self) -> Option<String> {
        self.ledger_error.as_ref().map(|err| {
            format!(
                "the capability ledger cannot be read, so no hook approval can be verified; failing closed until a human repairs it (openmax --ledger-repair): {err}"
            )
        })
    }

    /// True when this call rewrites one of the files that is failing closed
    /// (the hook itself or the code it runs). The blocked state expresses no
    /// enforceable policy, and the agent is told to write these files, so one
    /// path stays open - exactly the carve-out `permissions.toml` has. It
    /// grants nothing: the call still goes through permissions and
    /// `approval_mode`, and no rewrite can make unapproved content approved,
    /// so the gate can be restored this way but never removed.
    fn repairs_failed_hook(&self, tool: &str, args: &Value, project_root: &Path) -> bool {
        if !matches!(tool, "write_file" | "edit_file") {
            return false;
        }
        let Some(raw) = args["path"].as_str() else {
            return false;
        };
        let (Some(candidate), Ok(root)) = (
            resolve_for_repair(&project_root.join(raw)),
            project_root.canonicalize(),
        ) else {
            return false;
        };
        candidate.starts_with(&root)
            && self
                .repair_paths
                .iter()
                .any(|p| resolve_for_repair(p).is_some_and(|p| p == candidate))
    }

    pub fn pre_count(&self) -> usize {
        self.pre.len()
    }

    pub fn post_count(&self) -> usize {
        self.post.len()
    }

    /// Run all matching `pre_tool_use` hooks. First block wins.
    pub async fn pre_tool_use(
        &self,
        session_id: &str,
        tool: &str,
        args: &Value,
        cwd: &Path,
        cancel: &Arc<CancelToken>,
    ) -> PreToolResult {
        if let Some(reason) = self.fail_closed_reason() {
            if !self.repairs_failed_hook(tool, args, cwd) {
                return PreToolResult::Block { reason };
            }
        }
        for hook in &self.pre {
            if !hook.matches(tool) {
                continue;
            }
            let payload = tool_payload(hook, session_id, tool, args, cwd, None);
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => return PreToolResult::Block { reason },
                HookRun::Cancelled => return PreToolResult::Cancelled,
            }
        }
        PreToolResult::Allow
    }

    /// Run all matching `post_tool_use` hooks with what the call returned.
    /// Observe only - a failure never blocks and a post hook cannot change
    /// what the model sees - but failures are returned, not swallowed.
    pub async fn post_tool_use(
        &self,
        session_id: &str,
        tool: &str,
        args: &Value,
        cwd: &Path,
        outcome: &crate::tools::ToolOutcome,
        cancel: &Arc<CancelToken>,
    ) -> Vec<HookFailure> {
        let mut failures = Vec::new();
        for hook in &self.post {
            if !hook.matches(tool) {
                continue;
            }
            let payload = tool_payload(hook, session_id, tool, args, cwd, Some(outcome));
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }

    /// Run all `user_prompt_submit` hooks against the text the user typed,
    /// before it enters the transcript. First block wins (nonzero exit); the
    /// blocked turn never starts and never reaches the model. Gate only:
    /// hooks still never inject text into the context.
    pub async fn user_prompt_submit(
        &self,
        session_id: &str,
        text: &str,
        cwd: &Path,
        cancel: &Arc<CancelToken>,
    ) -> PreToolResult {
        if let Some(reason) = self.ledger_fail_closed_reason() {
            return PreToolResult::Block { reason };
        }
        for hook in &self.user_prompt {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "text": text,
            });
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => return PreToolResult::Block { reason },
                HookRun::Cancelled => return PreToolResult::Cancelled,
            }
        }
        PreToolResult::Allow
    }

    /// Run `session_start` hooks (a session's first turn). Observe only:
    /// nothing enters the model context, but failures are returned.
    pub async fn session_start(
        &self,
        session_id: &str,
        cwd: &Path,
        cancel: &Arc<CancelToken>,
    ) -> Vec<HookFailure> {
        let mut failures = Vec::new();
        for hook in &self.session_start {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
            });
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }

    /// Run `compaction` hooks after context was pruned, with the same digest
    /// record that was persisted. Observe only; failures are returned.
    pub async fn compaction(
        &self,
        session_id: &str,
        cwd: &Path,
        record: &Value,
        cancel: &Arc<CancelToken>,
    ) -> Vec<HookFailure> {
        let mut failures = Vec::new();
        for hook in &self.compaction {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "record": record,
            });
            match run_hook(hook, payload, cwd, cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }

    /// Run `turn_end` hooks with the turn's stop reason. Observe only, and
    /// deliberately run with a fresh cancel token: a cancelled turn is still
    /// a finished turn worth observing. Failures are returned.
    pub async fn turn_end(
        &self,
        session_id: &str,
        cwd: &Path,
        stop_reason: &str,
    ) -> Vec<HookFailure> {
        let cancel = Arc::new(CancelToken::default());
        let mut failures = Vec::new();
        for hook in &self.turn_end {
            let payload = serde_json::json!({
                "event": hook.event.as_str(),
                "session_id": session_id,
                "cwd": cwd.display().to_string(),
                "stop_reason": stop_reason,
            });
            match run_hook(hook, payload, cwd, &cancel).await {
                HookRun::Allow => {}
                HookRun::Block(reason) => failures.push(failure(hook, reason)),
                HookRun::Cancelled => break,
            }
        }
        failures
    }
}

/// Resolve a path for the repair comparison without requiring the file to
/// exist: a deleted gate script is exactly what the session has to be able to
/// recreate, and canonicalizing a missing path fails. The parent must exist
/// and is canonicalized, so `..` and symlinked parents are resolved before the
/// caller's containment check - the carve-out still cannot be aimed outside
/// the project, and a path whose parent is missing too is refused rather than
/// compared lexically.
fn resolve_for_repair(path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?.canonicalize().ok()?;
    Some(parent.join(path.file_name()?))
}

/// Whether this path is a hook manifest: one of the two directories hooks load
/// from, holding a `.toml`. Used to reconcile approved paths that are gone,
/// where there is no file left to parse.
pub(crate) fn is_hook_manifest(path: &Path, project_root: &Path) -> bool {
    manifest_in_dirs(path, &hook_dirs(project_root))
}

/// The same question against the dirs a discovery actually read, so
/// reconciliation and discovery can never disagree about which files are in
/// scope.
fn manifest_in_dirs(path: &Path, dirs: &[PathBuf]) -> bool {
    if path.extension().and_then(|e| e.to_str()) != Some("toml") {
        return false;
    }
    let Some(parent) = path.parent() else { return false };
    dirs.iter().any(|dir| {
        parent == dir || dir.canonicalize().map(|dir| parent == dir).unwrap_or(false)
    })
}

/// One identity per file, so two spellings of one path compare equal. Approved
/// paths are stored canonical while a directory listing is not, and a file
/// that is gone cannot be canonicalized at all.
fn identity(path: &Path) -> Option<PathBuf> {
    path.canonicalize().ok().or_else(|| resolve_for_repair(path))
}

fn same_file(a: &Path, b: &Path) -> bool {
    a == b || identity(a).is_some_and(|a| identity(b) == Some(a))
}

fn stem_of(path: &Path) -> Option<&str> {
    path.file_stem().and_then(|s| s.to_str())
}

/// Hook files as "path: reason", for one fail-closed message.
fn describe(entries: &[(PathBuf, String)]) -> String {
    entries
        .iter()
        .map(|(path, reason)| {
            if reason.starts_with(&path.display().to_string()) {
                reason.clone()
            } else {
                format!("{}: {reason}", path.display())
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The stdin payload for tool-scoped events, shared by pre and post. `result`
/// is the outcome a `post_tool_use` hook is reporting on, and is None before
/// the call runs.
fn tool_payload(
    hook: &HookSpec,
    session_id: &str,
    tool: &str,
    args: &Value,
    cwd: &Path,
    result: Option<&crate::tools::ToolOutcome>,
) -> Value {
    let mut payload = serde_json::json!({
        "event": hook.event.as_str(),
        "session_id": session_id,
        "tool": tool,
        "args": args,
        "cwd": cwd.display().to_string(),
        "tool_ok": result.map(|o| o.ok),
    });
    if let Some(outcome) = result {
        let head = head_bytes(&outcome.output, MAX_OUTPUT_BYTES);
        let Some(map) = payload.as_object_mut() else { return payload };
        map.insert("output".into(), Value::String(head.to_string()));
        // Output is bounded twice, and a hook is told about both cuts rather
        // than left to infer either from the length it received.
        map.insert("output_bytes".into(), Value::from(outcome.output.len()));
        map.insert(
            "output_truncated".into(),
            Value::Bool(head.len() < outcome.output.len()),
        );
        map.insert(
            "process_bytes".into(),
            outcome.process_bytes.map(Value::from).unwrap_or(Value::Null),
        );
        map.insert("process_truncated".into(), Value::Bool(outcome.process_truncated));
    }
    payload
}

/// The longest prefix of `s` that fits in `max` bytes without splitting a
/// character. A tool result can be large (a `bash` call can print megabytes)
/// and every matching hook is handed a copy, so what a hook sees is a head,
/// the same shape hook block reasons already take.
///
/// This is the second of two bounds. The tool layer has already rendered the
/// process output down to a result, keeping its tail and saying so inline when
/// it dropped anything, and that result is what the model reasoned about. It
/// is therefore what a hook is shown, and what `output_bytes` measures.
fn head_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[derive(Debug, PartialEq, Eq)]
pub enum PreToolResult {
    Allow,
    Block { reason: String },
    /// User cancelled while a gate hook was running; not a policy rejection.
    Cancelled,
}

impl HookSpec {
    fn matches(&self, tool: &str) -> bool {
        match &self.tool_filter {
            None => true,
            Some(name) => name == tool,
        }
    }
}

enum HookRun {
    Allow,
    Block(String),
    Cancelled,
}

/// Unknown keys are rejected so a misspelled `tool` filter cannot silently
/// widen a hook to every call.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HookFile {
    event: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
    /// Optional tool name filter.
    #[serde(default)]
    tool: Option<String>,
}

fn default_timeout() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

pub(crate) fn hook_dirs(project_root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![project_root.join(".openmax").join("hooks")];
    if let Some(home) = std::env::var_os("HOME") {
        dirs.push(PathBuf::from(home).join(".openmax").join("hooks"));
    }
    dirs
}

/// Errors are ignored by discovery and surfaced verbatim by `openmax --check`.
pub(crate) fn parse_hook_file(path: &Path) -> Result<HookSpec, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("unreadable: {e}"))?;
    let source_sha256 = crate::ledger::sha256_hex(text.as_bytes());
    let file: HookFile = toml::from_str(&text).map_err(|e| format!("invalid TOML: {e}"))?;
    let event = HookEvent::parse(&file.event).ok_or_else(|| {
        format!(
            "unknown event '{}': expected pre_tool_use, post_tool_use, user_prompt_submit, session_start, compaction, or turn_end",
            file.event
        )
    })?;
    let command = file.command.trim().to_string();
    if command.is_empty() {
        return Err("command is empty".into());
    }
    let tool_filter = file
        .tool
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    Ok(HookSpec {
        source_sha256,
        event,
        command,
        args: file.args,
        timeout_secs: file.timeout_secs.clamp(1, MAX_TIMEOUT_SECS),
        tool_filter,
        source_path: path.to_path_buf(),
        // Filled by the approval filter, which is where the approved set is
        // known; a spec that never passed it runs nothing.
        bound_code: Vec::new(),
    })
}

/// Why this hook must not run right now, if it must not: one of the files it
/// hands to the host no longer holds the bytes a human approved. Discovery
/// happens once per turn, so this is re-checked before every spawn - a script
/// rewritten between two calls of the same turn would otherwise run
/// unapproved. One small read per bound file, against a process spawn.
fn code_changed_since_approval(hook: &HookSpec) -> Option<String> {
    for (path, approved) in &hook.bound_code {
        let current = std::fs::read(path).ok().map(|b| crate::ledger::sha256_hex(&b));
        if current.as_deref() != Some(approved.as_str()) {
            return Some(format!(
                "hook '{}' did not run: {} is no longer the content that was approved; restore it or re-approve with `openmax --approve {}`",
                hook.source_path.display(),
                path.display(),
                hook.source_path.display()
            ));
        }
    }
    None
}

async fn run_hook(
    hook: &HookSpec,
    payload: Value,
    cwd: &Path,
    cancel: &Arc<CancelToken>,
) -> HookRun {
    if cancel.is_cancelled() {
        return HookRun::Cancelled;
    }
    if let Some(reason) = code_changed_since_approval(hook) {
        // Block for gates, reported (never silent) for observers: the caller
        // maps this variant per event, exactly as it does a spawn failure.
        return HookRun::Block(reason);
    }
    let request = ProcessRequest {
        program: hook.command.clone().into(),
        args: hook.args.iter().cloned().map(Into::into).collect(),
        cwd: cwd.to_path_buf(),
        stdin: StdinMode::Bytes(payload.to_string().into_bytes()),
        timeout: Duration::from_secs(hook.timeout_secs),
        capture: CaptureSpec {
            // Hook block reasons keep their beginning and are character-capped
            // below. Four bytes per character covers valid UTF-8.
            head_bytes: MAX_REASON_CHARS * 4,
            tail_bytes: 0,
            spill_dir: None,
            spill_bytes_per_stream: 0,
        },
    };

    match execution::run_process(request, cancel.clone()).await {
        Err(ProcessError::Spawn(e)) => {
            // Misconfigured hook: fail closed for pre, ignore for post-style.
            // Caller maps Block for pre_tool_use only.
            HookRun::Block(format!(
                "failed to start hook '{}' ({}): {e}",
                hook.command,
                hook.source_path.display()
            ))
        }
        Err(ProcessError::Wait(e)) => {
            HookRun::Block(format!("hook '{}' failed: {e}", hook.source_path.display()))
        }
        Ok(output) => match output.termination {
            Termination::Cancelled => HookRun::Cancelled,
            Termination::TimedOut => HookRun::Block(format!(
                "hook '{}' timed out after {}s",
                hook.source_path.display(),
                hook.timeout_secs
            )),
            Termination::Exited(status) => {
                if status.success() {
                    HookRun::Allow
                } else {
                    let mut reason =
                        String::from_utf8_lossy(&output.stdout.head).trim().to_string();
                    if reason.is_empty() {
                        reason = String::from_utf8_lossy(&output.stderr.head).trim().to_string();
                    }
                    if reason.is_empty() {
                        reason = format!(
                            "blocked by hook {} (exit {})",
                            hook.source_path.display(),
                            status.code().unwrap_or(-1)
                        );
                    }
                    if reason.chars().count() > MAX_REASON_CHARS {
                        reason = reason.chars().take(MAX_REASON_CHARS).collect::<String>() + "…";
                    }
                    HookRun::Block(reason)
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    fn write_script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    /// The approval boundary itself: an unapproved hook never runs, and the
    /// same content approved (as `openmax --approve` would) is live on the
    /// next discovery.
    #[test]
    fn unapproved_hooks_are_inert_until_their_content_is_approved() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audit.toml");
        std::fs::write(&path, "event = \"post_tool_use\"\ncommand = \"/bin/sh\"\n").unwrap();

        let hooks = Hooks::discover(&tmp, &data);
        assert!(hooks.post.is_empty(), "unapproved content must not load");

        let sha = crate::ledger::sha256_hex(&std::fs::read(&path).unwrap());
        crate::ledger::approve_hash(&data, &tmp, &sha).unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.post.len(), 1, "approved content loads");

        // Any edit revokes: the new bytes have a new, unapproved hash.
        std::fs::write(&path, "event = \"post_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        assert!(hooks.post.is_empty(), "an edited hook must fall back to inert");
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// A human-approved gate whose script the agent then rewrites. Approval
    /// binds the code, not just the pointer to it, so the swapped payload
    /// never runs - and because this gate was live, its removal is not
    /// allowed to be silent either: tool calls fail closed until it is back.
    #[tokio::test]
    async fn swapping_an_approved_gates_script_fails_closed_instead_of_passing() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_script(&tmp, "gate.sh", "#!/bin/sh\necho 'blocked by the human gate'\nexit 1\n");
        let toml = hooks_dir.join("gate.toml");
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
        approve_hook_file(&tmp, &data, &toml);

        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.pre_count(), 1, "the approved pair is live");
        let cancel = Arc::new(CancelToken::default());
        let args = serde_json::json!({"command": "ls"});
        let blocked = hooks.pre_tool_use("s", "bash", &args, &tmp, &cancel).await;
        assert!(matches!(blocked, PreToolResult::Block { .. }), "{blocked:?}");

        // The agent rewrites the script with a payload that passes everything.
        // Mid-turn: this is the hook set the turn already discovered.
        write_script(&tmp, "gate.sh", "#!/bin/sh\nexit 0\n");
        match hooks.pre_tool_use("s", "bash", &args, &tmp, &cancel).await {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("gate.sh"), "{reason}");
                assert!(reason.contains("approved"), "{reason}");
            }
            other => panic!("a swapped script must not run: {other:?}"),
        }

        // Next turn: discovery drops the hook, and a dropped live gate blocks.
        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.pre_count(), 0);
        match hooks.pre_tool_use("s", "bash", &args, &tmp, &cancel).await {
            PreToolResult::Block { reason } => assert!(reason.contains("failing closed"), "{reason}"),
            other => panic!("a revoked gate must fail closed, not vanish: {other:?}"),
        }

        // Restoring the approved bytes restores the gate; nothing else does.
        write_script(&tmp, "gate.sh", "#!/bin/sh\necho 'blocked by the human gate'\nexit 1\n");
        assert_eq!(Hooks::discover(&tmp, &data).pre_count(), 1);
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// Deleting an approved gate is strictly easier than rewriting one, and
    /// it leaves nothing on disk to iterate: no spec, no invalid entry. It is
    /// caught by reconciling the approved paths, and lands in the same
    /// fail-closed state a modification does - with a message that says
    /// deleted, not changed, and a way back for either party.
    #[tokio::test]
    async fn deleting_an_approved_gate_fails_closed_and_says_so() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_script(&tmp, "gate.sh", "#!/bin/sh\nexit 1\n");
        let toml = hooks_dir.join("gate.toml");
        let body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n";
        std::fs::write(&toml, body).unwrap();
        approve_hook_file(&tmp, &data, &toml);
        assert_eq!(Hooks::discover(&tmp, &data).pre_count(), 1);

        std::fs::remove_file(&toml).unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        let cancel = Arc::new(CancelToken::default());
        match hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await
        {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("deleted"), "deleted must not read as changed: {reason}");
                assert!(reason.contains("gate.toml"), "{reason}");
                assert!(reason.contains("openmax --forget"), "the way out must be named: {reason}");
            }
            other => panic!("a deleted gate must fail closed, not disappear: {other:?}"),
        }

        // Way back one: the session recreates the file it deleted.
        let repair = serde_json::json!({"path": ".openmax/hooks/gate.toml", "content": body});
        assert_eq!(
            hooks.pre_tool_use("s", "write_file", &repair, &tmp, &cancel).await,
            PreToolResult::Allow,
            "a deleted hook must be recreatable from inside the session"
        );
        std::fs::write(&toml, body).unwrap();
        assert_eq!(Hooks::discover(&tmp, &data).pre_count(), 1, "restored bytes are approved bytes");

        // Way back two: the human meant it, and retires the approval.
        std::fs::remove_file(&toml).unwrap();
        assert!(crate::ledger::forget_capability(&data, &tmp, &toml).unwrap());
        let hooks = Hooks::discover(&tmp, &data);
        assert!(hooks.is_empty(), "a retired capability leaves no fail-closed state");
        assert_eq!(
            hooks.pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel).await,
            PreToolResult::Allow
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// A deleted bound script is the other half of the same problem: the gate
    /// fails closed, and the file that has to come back does not exist, so a
    /// repair check that canonicalizes the target can never match it.
    #[tokio::test]
    async fn a_deleted_bound_script_can_be_recreated_but_traversal_still_refuses() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::create_dir_all(tmp.join("scripts")).unwrap();
        write_script(&tmp.join("scripts"), "gate.sh", "#!/bin/sh\nexit 1\n");
        let toml = hooks_dir.join("gate.toml");
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"./scripts/gate.sh\"\n").unwrap();
        approve_hook_file(&tmp, &data, &toml);
        assert_eq!(Hooks::discover(&tmp, &data).pre_count(), 1);

        std::fs::remove_file(tmp.join("scripts/gate.sh")).unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        let cancel = Arc::new(CancelToken::default());
        assert!(matches!(
            hooks.pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel).await,
            PreToolResult::Block { .. }
        ));
        let recreate = serde_json::json!({"path": "scripts/gate.sh", "content": "#!/bin/sh\nexit 1\n"});
        assert_eq!(
            hooks.pre_tool_use("s", "write_file", &recreate, &tmp, &cancel).await,
            PreToolResult::Allow,
            "the missing script must be recreatable in session"
        );

        // The carve-out resolves paths before it compares them, so it still
        // cannot be aimed anywhere but the files that are failing closed.
        for escape in [
            "../gate.sh",                    // traversal out of the project
            "scripts/../../gate.sh",         // traversal through a real dir
            "no-such-dir/gate.sh",           // parent does not exist either
            "src/main.rs",                   // an ordinary file
        ] {
            let args = serde_json::json!({"path": escape, "content": "x"});
            assert!(
                matches!(
                    hooks.pre_tool_use("s", "write_file", &args, &tmp, &cancel).await,
                    PreToolResult::Block { .. }
                ),
                "the repair carve-out must refuse {escape}"
            );
        }
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// A symlinked parent must not smuggle the repair target out of the
    /// project: the parent is canonicalized before containment is judged.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_repair_carve_out_resolves_symlinked_parents() {
        let tmp = tempfile_dir();
        let outside = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_script(&tmp, "gate.sh", "#!/bin/sh\nexit 1\n");
        let toml = hooks_dir.join("gate.toml");
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
        approve_hook_file(&tmp, &data, &toml);
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n# edited\n").unwrap();
        std::os::unix::fs::symlink(&outside, tmp.join("escape")).unwrap();

        let hooks = Hooks::discover(&tmp, &data);
        let cancel = Arc::new(CancelToken::default());
        let args = serde_json::json!({"path": "escape/gate.toml", "content": "x"});
        assert!(
            matches!(
                hooks.pre_tool_use("s", "write_file", &args, &tmp, &cancel).await,
                PreToolResult::Block { .. }
            ),
            "a symlinked parent must not carry the carve-out outside the project"
        );
        let _ = std::fs::remove_dir_all(tmp);
        let _ = std::fs::remove_dir_all(outside);
    }

    /// Demotion instead of deletion: the agent leaves the approved gate in
    /// place and rewrites its `event` to an observe one. If reconciliation
    /// asked the file what it is, it would answer "observer", be reported
    /// rather than blocking, and the human's gate would stop gating. The
    /// approved event is what decides, so it stays a gate.
    #[tokio::test]
    async fn an_approved_gate_cannot_demote_itself_to_an_observe_hook() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_script(&tmp, "gate.sh", "#!/bin/sh\nexit 1\n");
        let toml = hooks_dir.join("gate.toml");
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
        approve_hook_file(&tmp, &data, &toml);
        assert_eq!(Hooks::discover(&tmp, &data).pre_count(), 1);

        // Same file, same script, one word changed.
        std::fs::write(&toml, "event = \"post_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.post_count(), 0, "the rewritten content is not approved");
        let cancel = Arc::new(CancelToken::default());
        match hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await
        {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("rewritten as a post_tool_use hook"), "{reason}");
                assert!(reason.contains("stop it gating"), "{reason}");
                assert!(reason.contains("gate.toml"), "{reason}");
            }
            other => panic!("a demoted gate must still fail closed: {other:?}"),
        }

        // A human who genuinely wants it as an observer re-approves, and then
        // the new shape is the approved shape.
        approve_hook_file(&tmp, &data, &toml);
        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.post_count(), 1);
        assert_eq!(
            hooks.pre_tool_use("s", "bash", &serde_json::json!({}), &tmp, &cancel).await,
            PreToolResult::Allow
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// The repair carve-out follows the approved content as well. A revoked
    /// hook rewritten to name a different script must not hand itself a write
    /// exemption for that script - while the files a human did bless stay
    /// restorable, which is the whole reason the exemption exists.
    #[tokio::test]
    async fn the_repair_exemption_follows_approved_paths_not_rewritten_ones() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::create_dir_all(tmp.join("scripts")).unwrap();
        write_script(&tmp.join("scripts"), "gate.sh", "#!/bin/sh\nexit 1\n");
        let toml = hooks_dir.join("gate.toml");
        let approved_body = "event = \"pre_tool_use\"\ncommand = \"./scripts/gate.sh\"\n";
        std::fs::write(&toml, approved_body).unwrap();
        approve_hook_file(&tmp, &data, &toml);

        // The agent rewrites the hook to run a script of its own choosing.
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"./scripts/evil.sh\"\n").unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        let cancel = Arc::new(CancelToken::default());
        let write = |path: &str| serde_json::json!({"path": path, "content": "#!/bin/sh\nexit 0\n"});
        assert!(matches!(
            hooks.pre_tool_use("s", "write_file", &write("scripts/evil.sh"), &tmp, &cancel).await,
            PreToolResult::Block { .. }
        ), "a rewritten hook must not name its own repair exemption");

        // What a human blessed stays repairable: the manifest, and the script
        // the approved content ran.
        assert_eq!(
            hooks.pre_tool_use("s", "write_file", &write(".openmax/hooks/gate.toml"), &tmp, &cancel).await,
            PreToolResult::Allow
        );
        assert_eq!(
            hooks.pre_tool_use("s", "write_file", &write("scripts/gate.sh"), &tmp, &cancel).await,
            PreToolResult::Allow
        );

        // And the full restore actually recovers the project, which is what
        // narrowing the exemption could quietly have broken.
        std::fs::write(&toml, approved_body).unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.pre_count(), 1, "restoring the approved bytes restores the gate");
        assert!(hooks.is_empty() || hooks.pre_count() == 1);
        match hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await
        {
            PreToolResult::Block { reason } => assert!(!reason.contains("failing closed"), "{reason}"),
            other => panic!("the restored gate must run and block on its own terms: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// The comment-only edit: byte-identical policy, brand-new hash. Before
    /// content binding this was inert; after it, dropping the hook silently
    /// would make a comment a way to switch a human gate off.
    #[tokio::test]
    async fn a_comment_only_edit_cannot_switch_off_a_live_gate() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_script(&tmp, "gate.sh", "#!/bin/sh\nexit 1\n");
        let toml = hooks_dir.join("gate.toml");
        let body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\ntool = \"bash\"\n";
        std::fs::write(&toml, body).unwrap();
        approve_hook_file(&tmp, &data, &toml);
        assert_eq!(Hooks::discover(&tmp, &data).pre_count(), 1);

        std::fs::write(&toml, format!("{body}# semantically inert comment\n")).unwrap();
        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.pre_count(), 0, "new bytes are unapproved bytes");
        let cancel = Arc::new(CancelToken::default());
        match hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await
        {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("gate.toml"), "{reason}");
                assert!(reason.contains("openmax --approve"), "the way out must be named: {reason}");
            }
            other => panic!("expected fail closed, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// A command outside the project is the human's reading of an absolute
    /// path: the manifest approval covers it, with no second hash to keep in
    /// sync and nothing to break on an OS upgrade.
    #[test]
    fn a_system_binary_command_needs_only_the_manifest_approval() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let path = hooks_dir.join("audit.toml");
        std::fs::write(&path, "event = \"post_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
        let sha = crate::ledger::sha256_hex(&std::fs::read(&path).unwrap());
        crate::ledger::approve_capability(&data, &tmp, &path, &[sha]).unwrap();
        assert_eq!(Hooks::discover(&tmp, &data).post_count(), 1);
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// An unapproved hook was never live, so a broken one removes no policy.
    /// Failing closed on it would let any write brick the project - including
    /// the write that repairs it - and every future session with it.
    #[tokio::test]
    async fn an_unapproved_broken_hook_is_reported_not_a_brick() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(hooks_dir.join("bad.toml"), "event = \n").unwrap();

        let hooks = Hooks::discover(&tmp, &data);
        let cancel = Arc::new(CancelToken::default());
        let repair = serde_json::json!({"path": ".openmax/hooks/bad.toml", "content": "x"});
        assert_eq!(
            hooks.pre_tool_use("s", "write_file", &repair, &tmp, &cancel).await,
            PreToolResult::Allow,
            "the repair must not be blocked by the file it repairs"
        );
        assert_eq!(
            hooks.pre_tool_use("s", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel).await,
            PreToolResult::Allow,
            "content that never ran cannot brick the project"
        );
        // Inert is not silent.
        let notices = hooks.notices();
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert_eq!(notices[0].hook, "bad");
        assert!(notices[0].detail.contains("never loaded"), "{}", notices[0].detail);
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// A hook a human did approve and that no longer parses still fails closed
    /// - it might have been a gate - but stays repairable from inside the
    /// session, the same carve-out permissions.toml has.
    #[tokio::test]
    async fn a_broken_approved_hook_fails_closed_but_stays_repairable() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let toml = hooks_dir.join("gate.toml");
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
        approve_hook_file(&tmp, &data, &toml);
        std::fs::write(&toml, "event = \n").unwrap();

        let hooks = Hooks::discover(&tmp, &data);
        let cancel = Arc::new(CancelToken::default());
        let blocked = hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        assert!(matches!(blocked, PreToolResult::Block { .. }), "{blocked:?}");
        let repair = serde_json::json!({"path": ".openmax/hooks/gate.toml", "content": "x"});
        assert_eq!(
            hooks.pre_tool_use("s", "write_file", &repair, &tmp, &cancel).await,
            PreToolResult::Allow
        );
        // The carve-out is one path, not an escape: other files stay blocked.
        let elsewhere = serde_json::json!({"path": "src/main.rs", "content": "x"});
        assert!(matches!(
            hooks.pre_tool_use("s", "write_file", &elsewhere, &tmp, &cancel).await,
            PreToolResult::Block { .. }
        ));
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// The ledger erroring is a verdict, not an empty store: it fires exactly
    /// when tamper detection works (a rewritten chain, a partial line from an
    /// interrupted append, a deleted log with a surviving pin). An approved
    /// gate must fail closed on that verdict, the same answer a revoked gate
    /// gets, never be reclassified as content nobody blessed - "unapproved
    /// and inert" is the one bucket that does not block, so collapsing the
    /// error into it lets one bash append to log.jsonl turn every human gate
    /// off.
    #[tokio::test]
    async fn an_unreadable_ledger_fails_closed_instead_of_unapproving_gates() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let toml = hooks_dir.join("gate.toml");
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
        approve_hook_file(&tmp, &data, &toml);
        assert_eq!(Hooks::discover(&tmp, &data).pre.len(), 1, "healthy chain: the gate loads");

        // One garbage byte through bash is enough to break the chain read.
        let log = crate::ledger::project_dir(&data, &tmp).join("log.jsonl");
        let mut text = std::fs::read_to_string(&log).unwrap();
        text.push_str("{\"v\":1,");
        std::fs::write(&log, text).unwrap();

        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.pre.len(), 0, "nothing unverifiable runs");
        let cancel = Arc::new(CancelToken::default());
        let blocked = hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        match blocked {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("--ledger-repair"), "the block must name the repair: {reason}");
            }
            other => panic!("tool execution must fail closed on an unreadable ledger, got {other:?}"),
        }
        // No carve-out: the repair lives outside the project, so no project
        // write is exempt from the block.
        let repair = serde_json::json!({"path": ".openmax/hooks/gate.toml", "content": "x"});
        assert!(matches!(
            hooks.pre_tool_use("s", "write_file", &repair, &tmp, &cancel).await,
            PreToolResult::Block { .. }
        ));
        // The prompt gate fails closed too: an approved user_prompt_submit
        // hook (a secret or PII screen) not running means the text would
        // reach the model endpoint and the transcript, which no later block
        // can undo.
        let submitted = hooks.user_prompt_submit("s", "the prompt", &tmp, &cancel).await;
        match submitted {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("--ledger-repair"), "{reason}");
            }
            other => panic!("prompt submission must fail closed on an unreadable ledger, got {other:?}"),
        }
        // Loud at turn start, and the notice must not prescribe --approve,
        // which fails under the same broken chain.
        let notices = hooks.notices();
        assert!(
            notices.iter().any(|n| n.detail.contains("--ledger-repair")),
            "{notices:?}"
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// Discover with every hook file under the project pre-approved (the file
    /// and the code it runs, exactly as `openmax --approve` does), plus a
    /// throwaway data dir: these tests exercise parse and gate semantics, not
    /// the approval boundary (which has its own tests).
    fn discover_for_test(project: &Path) -> Hooks {
        let data = project.join("test-approvals-data");
        approve_all_hooks(project, &data);
        Hooks::discover(project, &data)
    }

    fn approve_all_hooks(project: &Path, data: &Path) {
        for dir in hook_dirs(project) {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for entry in rd.flatten() {
                approve_hook_file(project, data, &entry.path());
            }
        }
    }

    /// What a human approving one hook file blesses: its bytes plus the bytes
    /// of the project-local code it names.
    fn approve_hook_file(project: &Path, data: &Path, path: &Path) {
        let Ok(bytes) = std::fs::read(path) else { return };
        let mut shas = vec![crate::ledger::sha256_hex(&bytes)];
        shas.extend(
            crate::ledger::manifest_code(path, project)
                .into_iter()
                .filter_map(|c| c.sha256),
        );
        let _ = crate::ledger::approve_capability(data, project, path, &shas);
    }

    fn write_hook_toml(dir: &Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).unwrap();
    }

    /// The cap ranks approved hooks only. Approved observers beyond it are
    /// reported and skipped - they gate nothing, so nothing blocks - and
    /// other events are unaffected by one event's volume.
    #[test]
    fn hooks_beyond_the_per_event_cap_never_run() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        for i in 0..(MAX_HOOKS_PER_EVENT + 3) {
            write_hook_toml(
                &hooks_dir,
                &format!("hook-{i:03}.toml"),
                "event = \"post_tool_use\"\ncommand = \"/bin/echo\"\n",
            );
        }
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.post.len(), MAX_HOOKS_PER_EVENT);
        assert!(hooks.pre.is_empty());
        // Observers beyond the cap are reported, and nothing fails closed.
        let capped: Vec<_> =
            hooks.notices().into_iter().filter(|n| n.detail.contains("cap")).collect();
        assert_eq!(capped.len(), 3, "{capped:?}");
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// The cap must rank what a human approved, not what sits in the
    /// directory: unapproved files are the one thing an agent writes without
    /// a human, and when the cap was applied before the approval filter, 32
    /// junk manifests with earlier-sorting stems pushed an approved gate out
    /// of discovery - a human's gate, switched off by files that themselves
    /// never run.
    #[tokio::test]
    async fn unapproved_files_cannot_push_an_approved_gate_past_the_cap() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_script(&tmp, "gate.sh", "#!/bin/sh\necho 'blocked by the human gate'\nexit 1\n");
        // The stem sorts after every junk file, so a pre-filter cap drops it.
        let toml = hooks_dir.join("zz-gate.toml");
        std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
        approve_hook_file(&tmp, &data, &toml);
        for i in 0..MAX_HOOKS_PER_EVENT {
            write_hook_toml(
                &hooks_dir,
                &format!("hook-{i:03}.toml"),
                "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n",
            );
        }

        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.pre_count(), 1, "only the approved gate is live");
        let cancel = Arc::new(CancelToken::default());
        let blocked = hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        match blocked {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("blocked by the human gate"), "{reason}");
            }
            other => panic!("the approved gate must still run: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// When humans genuinely approve more gates than the cap runs, the ones
    /// beyond it are policy a human installed that is not running - so they
    /// fail closed with a reason naming the cap and the way out, never
    /// silently stop gating.
    #[tokio::test]
    async fn an_approved_gate_beyond_the_cap_fails_closed_instead_of_vanishing() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        for i in 0..(MAX_HOOKS_PER_EVENT + 1) {
            let toml = hooks_dir.join(format!("gate-{i:03}.toml"));
            std::fs::write(&toml, "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
            approve_hook_file(&tmp, &data, &toml);
        }

        let hooks = Hooks::discover(&tmp, &data);
        assert_eq!(hooks.pre_count(), MAX_HOOKS_PER_EVENT, "the sorted head runs");
        let cancel = Arc::new(CancelToken::default());
        let blocked = hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        match blocked {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("cap"), "{reason}");
                assert!(
                    reason.contains(&format!("gate-{:03}.toml", MAX_HOOKS_PER_EVENT)),
                    "the dropped gate must be named: {reason}"
                );
                assert!(reason.contains("openmax --forget"), "the way out must be named: {reason}");
            }
            other => panic!("a gate beyond the cap must fail closed: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// Shadowing is deletion's trick by occupation: a project file that takes
    /// an approved global gate's stem keeps that gate out of discovery while
    /// the gate's own file sits untouched on disk, which is exactly the case
    /// the missing-file reconciliation cannot see. An unapproved occupier
    /// runs nothing in the gate's place, so the gate fails closed - and the
    /// way back in-session is writing approved content over the occupier,
    /// which the repair carve-out must allow.
    #[tokio::test]
    async fn an_unapproved_file_cannot_shadow_an_approved_gate_into_silence() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let project_hooks = tmp.join("project-hooks");
        let global_hooks = tmp.join("global-hooks");
        std::fs::create_dir_all(&project_hooks).unwrap();
        std::fs::create_dir_all(&global_hooks).unwrap();
        write_script(&tmp, "gate.sh", "#!/bin/sh\nexit 1\n");
        let approved_body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n";
        let global_gate = global_hooks.join("gate.toml");
        std::fs::write(&global_gate, approved_body).unwrap();
        approve_hook_file(&tmp, &data, &global_gate);
        let dirs = [project_hooks.clone(), global_hooks.clone()];
        assert_eq!(Hooks::discover_dirs(&tmp, &data, &dirs).pre_count(), 1, "the gate is live");

        // The agent writes a same-stem file it never got approved.
        std::fs::write(project_hooks.join("gate.toml"), "event = \"post_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
        let hooks = Hooks::discover_dirs(&tmp, &data, &dirs);
        assert_eq!(hooks.pre_count(), 0, "the shadowed gate cannot run");
        let cancel = Arc::new(CancelToken::default());
        let blocked = hooks
            .pre_tool_use("s", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        match blocked {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("shadowed"), "{reason}");
                assert!(reason.contains("gate.toml"), "{reason}");
            }
            other => panic!("a shadowed approved gate must fail closed: {other:?}"),
        }

        // The occupier is repairable from inside the session: writing the
        // approved bytes over it restores the policy under the winning stem.
        let repair = serde_json::json!({"path": "project-hooks/gate.toml", "content": approved_body});
        assert_eq!(
            hooks.pre_tool_use("s", "write_file", &repair, &tmp, &cancel).await,
            PreToolResult::Allow,
            "writing approved content over the occupier must stay possible"
        );
        std::fs::write(project_hooks.join("gate.toml"), approved_body).unwrap();
        let hooks = Hooks::discover_dirs(&tmp, &data, &dirs);
        assert_eq!(hooks.pre_count(), 1, "approved bytes under the winning stem run again");
        match hooks
            .pre_tool_use("s", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await
        {
            PreToolResult::Block { reason } => {
                assert!(!reason.contains("failing closed"), "the gate blocks on its own terms: {reason}");
            }
            other => panic!("the restored gate must run and block: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// Two approved hooks on one stem is precedence a human built - a project
    /// override of a global hook - not displacement: the override runs, and
    /// nothing blocks or nags about the file it supersedes.
    #[tokio::test]
    async fn an_approved_override_of_an_approved_hook_is_precedence_not_displacement() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let project_hooks = tmp.join("project-hooks");
        let global_hooks = tmp.join("global-hooks");
        std::fs::create_dir_all(&project_hooks).unwrap();
        std::fs::create_dir_all(&global_hooks).unwrap();
        let global_gate = global_hooks.join("gate.toml");
        std::fs::write(&global_gate, "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n").unwrap();
        approve_hook_file(&tmp, &data, &global_gate);
        let project_gate = project_hooks.join("gate.toml");
        std::fs::write(&project_gate, "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\ntool = \"bash\"\n").unwrap();
        approve_hook_file(&tmp, &data, &project_gate);

        let dirs = [project_hooks.clone(), global_hooks.clone()];
        let hooks = Hooks::discover_dirs(&tmp, &data, &dirs);
        assert_eq!(hooks.pre_count(), 1);
        assert!(hooks.notices().is_empty(), "{:?}", hooks.notices());
        let cancel = Arc::new(CancelToken::default());
        assert_eq!(
            hooks.pre_tool_use("s", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel).await,
            PreToolResult::Allow
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn discover_empty_when_no_hooks_dir() {
        let tmp = tempfile_dir();
        let hooks = discover_for_test(&tmp);
        assert!(hooks.is_empty());
    }

    #[test]
    fn discover_detects_same_length_same_mtime_edit() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let body_a = "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n";
        let body_b = "event = \"pre_tool_use\"\ncommand = \"/bin/date\"\n";
        assert_eq!(body_a.len(), body_b.len());
        write_hook_toml(&hooks_dir, "gate.toml", body_a);
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre.len(), 1);
        assert_eq!(hooks.pre[0].command, "/bin/echo");

        // Same byte length, pinned mtime: a metadata fingerprint would keep
        // the obsolete policy live.
        let path = hooks_dir.join("gate.toml");
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        write_hook_toml(&hooks_dir, "gate.toml", body_b);
        let f = std::fs::File::options().write(true).open(&path).unwrap();
        f.set_modified(mtime).unwrap();
        drop(f);

        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre.len(), 1);
        assert_eq!(hooks.pre[0].command, "/bin/date");
    }

    #[test]
    fn discovery_holds_no_memory_between_calls() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let gate = "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n";
        let path = hooks_dir.join("gate.toml");

        write_hook_toml(&hooks_dir, "gate.toml", gate);
        assert_eq!(discover_for_test(&tmp).pre.len(), 1);

        std::fs::remove_file(&path).unwrap();
        let hooks = discover_for_test(&tmp);
        assert!(hooks.pre.is_empty(), "a removed gate must stop applying");
        // Stopping is not the same as being forgotten: the removal of a gate
        // a human installed is reconciled from the approved paths, so it
        // fails closed rather than passing everything.
        assert_eq!(hooks.missing.len(), 1, "{:?}", hooks.missing);

        // Restoring byte-identical content puts the policy back. Anything
        // that remembered the gap keyed by content would answer "no gate"
        // here, and a gate that does not run is a gate that is not enforced.
        write_hook_toml(&hooks_dir, "gate.toml", gate);
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre.len(), 1, "invalid: {:?}", hooks.invalid);
        assert_eq!(hooks.pre[0].command, "/bin/echo");
    }

    #[tokio::test]
    async fn invalid_hook_file_fails_closed() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        write_hook_toml(
            &hooks_dir,
            "bad.toml",
            r#"
event = "not_a_real_event"
command = "true"
"#,
        );
        let hooks = discover_for_test(&tmp);
        assert!(!hooks.is_empty());
        assert_eq!(hooks.pre_count(), 0);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        match result {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("bad.toml"), "{reason}");
                assert!(reason.contains("failing closed"), "{reason}");
            }
            PreToolResult::Allow | PreToolResult::Cancelled => panic!("expected block"),
        }
    }

    #[tokio::test]
    async fn shadowed_invalid_hook_does_not_fail_closed() {
        let tmp = tempfile_dir();
        let project = tmp.join("project");
        let global = tmp.join("global");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::create_dir_all(&global).unwrap();
        write_hook_toml(&project, "audit.toml", "event = \"post_tool_use\"\ncommand = \"true\"\n");
        write_hook_toml(&global, "audit.toml", "event = \"not_a_real_event\"\ncommand = \"true\"\n");
        // The malformed global file is shadowed by the valid project one, so
        // it was never going to run and must not block.
        let hooks = discover_in_dirs(&[project.clone(), global.clone()]);
        assert!(hooks.invalid.is_empty(), "{:?}", hooks.invalid);
        assert_eq!(hooks.post_count(), 1);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        assert_eq!(result, PreToolResult::Allow);

        // The reverse still fails closed: a malformed file in the
        // higher-precedence dir holds its stem instead of silently falling
        // back to the valid definition it shadows.
        let hooks = discover_in_dirs(&[global, project]);
        assert_eq!(hooks.invalid.len(), 1, "{:?}", hooks.invalid);
        assert_eq!(hooks.post_count(), 0);
        let result = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        assert!(matches!(result, PreToolResult::Block { .. }));
    }

    #[tokio::test]
    async fn unknown_hook_key_is_rejected_and_fails_closed() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // `tools` (plural) is the likely typo for the `tool` filter; accepting
        // it would silently widen the hook to every call.
        write_hook_toml(
            &hooks_dir,
            "typo.toml",
            r#"
event = "pre_tool_use"
command = "true"
tools = "bash"
"#,
        );
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre_count(), 0);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        assert!(matches!(result, PreToolResult::Block { .. }));
    }

    #[tokio::test]
    async fn pre_hook_can_block_with_stdout_reason() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "block.sh",
            "#!/bin/sh\necho 'blocked by test hook'\nexit 1\n",
        );
        write_hook_toml(
            &hooks_dir,
            "block.toml",
            &format!(
                r#"
event = "pre_tool_use"
command = "{}"
tool = "bash"
"#,
                script.display()
            ),
        );
        let hooks = discover_for_test(&tmp);
        assert_eq!(hooks.pre_count(), 1);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use(
                "sess",
                "bash",
                &serde_json::json!({"command": "rm -rf /"}),
                &tmp,
                &cancel,
            )
            .await;
        match result {
            PreToolResult::Block { reason } => {
                assert!(reason.contains("blocked by test hook"), "{reason}");
            }
            PreToolResult::Allow | PreToolResult::Cancelled => panic!("expected block"),
        }
        // Filtered tool should not run the hook path for other tools.
        let allow = hooks
            .pre_tool_use("sess", "read_file", &serde_json::json!({"path": "a"}), &tmp, &cancel)
            .await;
        assert_eq!(allow, PreToolResult::Allow);
    }

    #[tokio::test]
    async fn session_start_and_compaction_hooks_observe_via_stdin() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // Each hook copies its stdin payload to a marker file; observe-only
        // means a nonzero exit must not disturb the caller either way.
        let start_script = write_script(
            &tmp,
            "start.sh",
            &format!("#!/bin/sh\ncat > {}/start.json\nexit 1\n", tmp.display()),
        );
        let compact_script = write_script(
            &tmp,
            "compact.sh",
            &format!("#!/bin/sh\ncat > {}/compact.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "start.toml",
            &format!("event = \"session_start\"\ncommand = \"{}\"\n", start_script.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "compact.toml",
            &format!("event = \"compaction\"\ncommand = \"{}\"\n", compact_script.display()),
        );
        let hooks = discover_for_test(&tmp);
        assert!(!hooks.is_empty());
        let cancel = Arc::new(CancelToken::default());

        hooks.session_start("sess", &tmp, &cancel).await;
        let start: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("start.json")).unwrap()).unwrap();
        assert_eq!(start["event"], "session_start");
        assert_eq!(start["session_id"], "sess");

        let record = serde_json::json!({"message_count": 7, "digest": "d"});
        hooks.compaction("sess", &tmp, &record, &cancel).await;
        let compact: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("compact.json")).unwrap())
                .unwrap();
        assert_eq!(compact["event"], "compaction");
        assert_eq!(compact["record"]["message_count"], 7);
    }

    #[tokio::test]
    async fn post_tool_use_hook_receives_the_tool_output() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "audit.sh",
            &format!("#!/bin/sh\ncat > {}/audit.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "audit.toml",
            &format!("event = \"post_tool_use\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let args = serde_json::json!({"command": "echo hi"});

        let outcome = crate::tools::ToolOutcome::ok("hi\n".into());
        hooks.post_tool_use("sess", "bash", &args, &tmp, &outcome, &cancel).await;

        let payload: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("audit.json")).unwrap()).unwrap();
        assert_eq!(payload["event"], "post_tool_use");
        assert_eq!(payload["tool"], "bash");
        assert_eq!(payload["tool_ok"], true);
        assert_eq!(payload["output"], "hi\n");
        assert_eq!(payload["output_bytes"], 3);
        assert_eq!(payload["output_truncated"], false);
        assert!(payload["process_bytes"].is_null(), "no process ran behind this result");
        assert_eq!(payload["process_truncated"], false);
        assert_eq!(payload["args"]["command"], "echo hi");
    }

    /// A pre hook is asked whether a call may run, and there is no output yet
    /// to report. It must not look like an empty one.
    #[tokio::test]
    async fn pre_tool_use_payload_carries_no_output() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "gate.sh",
            &format!("#!/bin/sh\ncat > {}/gate.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "gate.toml",
            &format!("event = \"pre_tool_use\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());

        let result = hooks
            .pre_tool_use("sess", "bash", &serde_json::json!({}), &tmp, &cancel)
            .await;
        assert_eq!(result, PreToolResult::Allow);

        let payload: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("gate.json")).unwrap()).unwrap();
        assert!(payload.get("output").is_none(), "{payload}");
        assert!(payload.get("output_truncated").is_none());
        assert!(payload["tool_ok"].is_null());
    }

    #[test]
    fn a_long_output_is_capped_without_splitting_a_character() {
        // A multibyte character straddling the cap must not be cut in half:
        // the payload has to stay valid JSON.
        let long = "é".repeat(MAX_OUTPUT_BYTES);
        let hook = HookSpec {
            source_sha256: String::new(),
            event: HookEvent::PostToolUse,
            command: "/bin/echo".into(),
            args: Vec::new(),
            timeout_secs: 1,
            tool_filter: None,
            source_path: PathBuf::from("/hooks/audit.toml"),
            bound_code: Vec::new(),
        };
        let payload = tool_payload(
            &hook,
            "sess",
            "bash",
            &serde_json::json!({}),
            Path::new("/project"),
            Some(&crate::tools::ToolOutcome::ok(long.clone())),
        );

        let head = payload["output"].as_str().unwrap();
        assert!(head.len() <= MAX_OUTPUT_BYTES);
        assert!(head.len() > MAX_OUTPUT_BYTES - 4, "the cap must not round down far");
        assert!(long.starts_with(head), "what a hook sees must be a real prefix");
        assert_eq!(payload["output_bytes"].as_u64().unwrap() as usize, long.len());
        assert_eq!(payload["output_truncated"], true);
        // Round-trips, which is the point of respecting the boundary.
        let encoded = serde_json::to_string(&payload).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&encoded).unwrap()["output"], head);
    }

    /// Output is bounded twice: the tool renders a process down to a result,
    /// then this payload takes a head of that result. A hook is told about
    /// both, so an audit hook never has to parse a notice out of the text to
    /// learn that a larger output existed.
    #[test]
    fn the_payload_reports_both_bounds() {
        let rendered = "[start of output truncated; bounded output log saved to /logs/x; \
                        tail or grep it with bash]\n…LINE-B\n";
        let hook = HookSpec {
            source_sha256: String::new(),
            event: HookEvent::PostToolUse,
            command: "/bin/echo".into(),
            args: Vec::new(),
            timeout_secs: 1,
            tool_filter: None,
            source_path: PathBuf::from("/hooks/audit.toml"),
            bound_code: Vec::new(),
        };
        let payload = tool_payload(
            &hook,
            "sess",
            "bash",
            &serde_json::json!({}),
            Path::new("/project"),
            Some(&crate::tools::ToolOutcome {
                ok: true,
                output: rendered.to_string(),
                process_bytes: Some(400_001),
                process_truncated: true,
                ..Default::default()
            }),
        );

        assert_eq!(payload["output"], rendered);
        assert_eq!(
            payload["output_bytes"].as_u64().unwrap() as usize,
            rendered.len(),
            "output_bytes measures the result the model reasoned about"
        );
        assert_eq!(
            payload["output_truncated"], false,
            "this payload dropped nothing, whatever the tool layer did earlier"
        );
        assert_eq!(
            payload["process_bytes"], 400_001,
            "the size the command actually produced survives the rendering"
        );
        assert_eq!(payload["process_truncated"], true);
    }

    #[test]
    fn head_bytes_returns_everything_that_fits() {
        assert_eq!(head_bytes("short", 16), "short");
        assert_eq!(head_bytes("exact", 5), "exact");
        assert_eq!(head_bytes("abcdef", 3), "abc");
        assert_eq!(head_bytes("éé", 3), "é", "a split character is dropped whole");
        assert_eq!(head_bytes("", 0), "");
    }

    #[tokio::test]
    async fn user_prompt_submit_hook_blocks_with_reason() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // Block any prompt whose stdin payload mentions a secret marker.
        let script = write_script(
            &tmp,
            "gate.sh",
            "#!/bin/sh\nif grep -q SECRET; then echo 'input contains a secret'; exit 1; fi\nexit 0\n",
        );
        write_hook_toml(
            &hooks_dir,
            "gate.toml",
            &format!("event = \"user_prompt_submit\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let blocked = hooks
            .user_prompt_submit("sess", "here is a SECRET token", &tmp, &cancel)
            .await;
        match blocked {
            PreToolResult::Block { reason } => assert!(reason.contains("secret"), "{reason}"),
            PreToolResult::Allow | PreToolResult::Cancelled => panic!("expected block"),
        }
        let allowed = hooks.user_prompt_submit("sess", "plain request", &tmp, &cancel).await;
        assert_eq!(allowed, PreToolResult::Allow);
    }

    #[tokio::test]
    async fn user_prompt_submit_cancel_is_not_a_block() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        // Hang until killed so cancel is the only way out.
        let script = write_script(&tmp, "slow.sh", "#!/bin/sh\nsleep 30\n");
        write_hook_toml(
            &hooks_dir,
            "slow.toml",
            &format!("event = \"user_prompt_submit\"\ncommand = \"{}\"\ntimeout_secs = 30\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let cancel_flag = cancel.clone();
        let task = tokio::spawn(async move {
            hooks.user_prompt_submit("sess", "prompt", &tmp, &cancel_flag).await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("join timed out")
            .expect("task panicked");
        assert_eq!(result, PreToolResult::Cancelled);
    }

    #[tokio::test]
    async fn turn_end_hook_runs_even_after_cancel() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(
            &tmp,
            "end.sh",
            &format!("#!/bin/sh\ncat > {}/end.json\n", tmp.display()),
        );
        write_hook_toml(
            &hooks_dir,
            "end.toml",
            &format!("event = \"turn_end\"\ncommand = \"{}\"\n", script.display()),
        );
        let hooks = discover_for_test(&tmp);
        // turn_end uses its own fresh token, so a cancelled turn still fires.
        hooks.turn_end("sess", &tmp, "cancelled").await;
        let end: Value =
            serde_json::from_str(&std::fs::read_to_string(tmp.join("end.json")).unwrap()).unwrap();
        assert_eq!(end["event"], "turn_end");
        assert_eq!(end["stop_reason"], "cancelled");
    }

    #[tokio::test]
    async fn pre_hook_allow_on_zero_exit() {
        let tmp = tempfile_dir();
        let hooks_dir = tmp.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        let script = write_script(&tmp, "ok.sh", "#!/bin/sh\nexit 0\n");
        write_hook_toml(
            &hooks_dir,
            "ok.toml",
            &format!(
                r#"
event = "pre_tool_use"
command = "{}"
"#,
                script.display()
            ),
        );
        let hooks = discover_for_test(&tmp);
        let cancel = Arc::new(CancelToken::default());
        let result = hooks
            .pre_tool_use("sess", "bash", &serde_json::json!({"command": "ls"}), &tmp, &cancel)
            .await;
        assert_eq!(result, PreToolResult::Allow);
    }

    #[tokio::test]
    async fn failing_observe_hook_is_reported_not_silent() {
        let dir = tempfile_dir();
        write_hook_toml(
            &dir,
            "audit.toml",
            "event = \"post_tool_use\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"exit 7\"]\n",
        );
        let hooks = discover_in_dirs(std::slice::from_ref(&dir));
        let outcome = crate::tools::ToolOutcome::ok("fine".into());
        let failures = hooks
            .post_tool_use(
                "s1",
                "bash",
                &serde_json::json!({}),
                &dir,
                &outcome,
                &Arc::new(CancelToken::default()),
            )
            .await;
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].hook, "audit");
        assert_eq!(failures[0].event, "post_tool_use");
        assert!(failures[0].detail.contains("7"), "{}", failures[0].detail);
        let _ = std::fs::remove_dir_all(dir);
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-hooks-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
