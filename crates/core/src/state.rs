//! Shared process state: `Core`, the handle every entry point takes.
//!
//! `Core` owns the data dir, resolved settings, the per-session event
//! channels, the set of running sessions, and their cancel tokens. It is the
//! only thing a frontend needs, which is what keeps `core` UI-free: the TUI,
//! `--print`, and `--stdio` are three consumers of one `AgentEvent` stream.
//!
//! `SessionData` is the in-memory half of a session. Two of its fields are
//! load-bearing rather than incidental: the frozen `registry`, whose
//! serialized schemas are part of the prompt-cache prefix and must not change
//! mid-session, and `take_seq`, which makes restoring a taken message vector
//! safe only for the turn that took it, so a newer turn or a recreated session
//! reusing an id cannot clobber it.
//!
//! `default_data_dir()` lives here, and it belongs only at entry points.
//! Discovery paths take a data dir as an argument; reading `$HOME` deep in the
//! call graph is how capabilities came to be found in one directory and
//! approved in another.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot};

use crate::config::{ApprovalMode, Settings};
use crate::registry::Registry;
use crate::types::{AgentEvent, AgentEventEnvelope, ChatMessage};

/// In-memory state of one agent session.
#[derive(Default, Clone)]
pub struct SessionData {
    pub messages: Vec<ChatMessage>,
    /// The session's current extension generation. Its serialized schemas
    /// are the tools array the session's prompt was built with, part of the
    /// prompt-cache prefix: a refreeze swaps what runs and keeps them, so it
    /// moves no byte of that prefix.
    pub registry: Arc<Registry>,
    /// The trailing note for `registry`'s generation, what it changed since
    /// the prompt was built, sent after the history on every request: rebuilt
    /// with the registry, never persisted. None when nothing changed.
    pub trailing_note: Option<Arc<str>>,
    /// Where the prompt's and the note's tokens go, for /context.
    pub prompt_breakdown: Arc<crate::prompt::PromptBreakdown>,
    /// Messages already written to disk; enables append-only persistence.
    pub persisted_count: usize,
    /// Replay-boundary edits for a hydration repair whose boundaries have not
    /// moved yet: the rewrite that should have landed it failed, or the index
    /// write after it did. Applied by the first save that has the repair on
    /// disk and gets the index write through: applied before, the boundaries
    /// would mark messages the file does not hold yet, and never applied, a
    /// replay draws each later divider early. In memory only.
    pub deferred_resume_edits: Vec<crate::sessions::ResumeEdit>,
    /// Process-unique id of the turn that last took `messages`; a restore is
    /// only valid while this still matches the taker (guards against a newer
    /// turn or a recreated session reusing the id).
    pub take_seq: u64,
    /// Whether this session already reported that its tool schemas outgrew the
    /// context window. The condition holds on every turn once it holds at all,
    /// so the advisory is emitted once and not per turn.
    pub schemas_over_budget_reported: bool,
    /// Content hashes of policy notices (inert allow rules, hooks that did
    /// not load) already narrated to the MODEL this session. The condition
    /// holds every turn once it holds at all, so the transcript gets one
    /// note per distinct notice - the UI keeps its per-turn events.
    pub reported_policy_notices: HashSet<u64>,
    /// Last execution mode narrated to the model, updated only when it changes.
    pub reported_approval_mode: Option<crate::config::ApprovalMode>,
    /// Ledger approval-event ids this session has already accounted for:
    /// seeded from the chain at build, then advanced as new events are
    /// narrated. An approval recorded outside the running session (a human
    /// at another terminal, #199) reaches it through no other channel - the
    /// refreeze receipt names file changes, not approvals - so a turn start
    /// outside auto names any it has not seen. None until the chain was
    /// first read: auto never consults it, so a session built in auto is
    /// seeded by its first turn in another mode, which narrates nothing.
    pub seen_ledger_events: Option<HashSet<u64>>,
    /// Server-reported `prompt_tokens` over the local estimate of the same
    /// request, as last observed on a turn's completion, or the floor a
    /// turn request the provider refused for length proved. The bytes/4
    /// estimator under-counts BPE tokenizers on code, by far more than the
    /// budget's headroom, so the send budget is scaled by this ratio and
    /// compaction fires before the provider refuses the request for length.
    /// Keyed by the route that taught it (`ChatClient::origin`): another
    /// model, URL, key, or header set tokenizes and templates differently,
    /// so after a switch the plain estimate applies until the new route
    /// reports. In memory only: a resumed session runs on the plain estimate
    /// until its first response or refusal for length. See
    /// `agent::token_factor`.
    pub prompt_ratio: Option<(String, f64)>,
}

/// Cooperative cancellation for one agent turn: a flag for cheap synchronous
/// checks plus a Notify so waiters wake immediately instead of polling.
/// notify_waiters only reaches tasks already registered, so `cancelled()`
/// re-checks the flag after registering — a cancel can never slip between
/// the check and the wait.
#[derive(Default)]
pub struct CancelToken {
    flag: AtomicBool,
    notify: tokio::sync::Notify,
}

impl CancelToken {
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Resolves once the token is cancelled; immediate if it already is.
    pub async fn cancelled(&self) {
        loop {
            if self.is_cancelled() {
                return;
            }
            let mut notified = std::pin::pin!(self.notify.notified());
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// Shared core state. The frontend owns an `Arc<Core>` plus the receiving half
/// of the event channel; background tasks clone the `Arc`.
pub struct Core {
    pub data_dir: PathBuf,
    pub settings: Mutex<Settings>,
    /// Project-scoped choices (see [`ProjectModes`]). Kept outside Settings
    /// so unrelated saves cannot widen the default for other projects.
    project_approval_modes: Mutex<ProjectModes>,
    /// Live sessions keyed by session id; hydrated from disk on first use.
    pub sessions: tokio::sync::Mutex<HashMap<String, SessionData>>,
    /// Sessions with an agent turn currently in flight.
    pub running: Mutex<HashSet<String>>,
    pub cancel_flags: Mutex<HashMap<String, Arc<CancelToken>>>,
    /// Messages sent to a running turn (`steer`), oldest first. An entry
    /// exists only while a turn is in flight: `start_turn` opens it under the
    /// `running` hold and settling removes it, so a steer is never queued for
    /// a turn that already settled (a compaction claims `running` but opens
    /// none). One still queued when its turn ends is dropped with the entry.
    pub(crate) steers: Mutex<HashMap<String, Vec<String>>>,
    /// Pending tool-approval prompts awaiting a user decision.
    pub approvals: Mutex<HashMap<String, oneshot::Sender<bool>>>,
    /// Serializes this process's session store writes, the index's
    /// read-modify-write cycles included, and guards what it knows about the
    /// store without reading it (`sessions::StoreMemo`).
    pub(crate) sessions_lock: Mutex<crate::sessions::StoreMemo>,
    /// Process ownership lasts as long as writable session state is attached.
    pub(crate) session_owners: Mutex<HashMap<String, crate::sessions::SessionOwner>>,
    /// Panic restorations must finish before the terminal receipt releases a turn.
    pub(crate) message_restorations: Mutex<HashMap<String, tokio::task::JoinHandle<()>>>,
    /// Content hash of settings.json as this process last knew it: the bytes
    /// read at launch, refreshed by [`Core::save_settings`]. Settings are
    /// launch-frozen, so any other change to the file is drift this process
    /// will never adopt - the drift receipt tells the model so, including
    /// the brick warning when the new bytes would not even parse. Lock
    /// discipline: never held across the `settings` lock (see
    /// `approval_mode`'s note; this mutex is leaf-only).
    settings_disk_fingerprint: Mutex<SettingsFingerprint>,
    /// The keys a command-line flag overrode for this run only, set at
    /// launch by [`Core::apply_key_flags`]: every save writes the file's
    /// value for each until the session sets it to another value, so no
    /// override reaches settings.json (see [`crate::config::RunOnly`]). Lock
    /// order: taken under `settings` when a caller holds it, and held across
    /// the `settings_disk_fingerprint` lock, which stays leaf-only.
    run_only: Mutex<Option<crate::config::RunOnly>>,
    /// The event log lines each held session has not written yet
    /// (`sessions::EventLog`). Held while an event is recorded and sent, so a
    /// log holds a session's events in the order its frontend reads them.
    /// Leaf-only: nothing else is locked while it is held.
    pub(crate) event_log: Mutex<crate::sessions::EventLog>,
    events: mpsc::UnboundedSender<AgentEventEnvelope>,
}

/// Project-scoped approval choices as this process applies them: the launch
/// snapshot, this process's own selections, and every stricter choice other
/// processes saved since.
///
/// Every frontend on a data dir saves to one trust store, so a choice saved
/// in one is read back by the others (`Core::refresh_saved_approval_modes`)
/// and applies wherever it is stricter: ask selected in one TUI reaches a
/// sibling session on the same project at its next tool call instead of
/// leaving it in auto. If that sibling's status line still shows the older
/// mode, its Shift+Tab first shows ask and saves nothing; the next press
/// steps from it. A looser choice is never read back, because the store is a
/// file a tool can write and taking a wider mode from it would be
/// self-approval; only this process's own selection loosens its modes.
#[derive(Default)]
struct ProjectModes {
    live: BTreeMap<PathBuf, ApprovalMode>,
    /// The trust store as last read, so a refresh rereads only a store that
    /// changed.
    stamp: Option<crate::sessions::Stamp>,
}

/// `live` tightened by `saved`: for each root either one names, the stricter
/// of the two modes that govern it, with `default` for a map that names none
/// at or above it. Resolving every named root this way keeps each path's mode
/// the stricter of the two everywhere, a nested root's own choice included.
/// So where the store is missing, or no longer names a live root or any root
/// above it, that root drops to `default` wherever that is stricter. A store
/// that no longer parses never reaches here and changes nothing.
fn tightened(
    live: &BTreeMap<PathBuf, ApprovalMode>,
    saved: &BTreeMap<PathBuf, ApprovalMode>,
    default: ApprovalMode,
) -> BTreeMap<PathBuf, ApprovalMode> {
    let governing = |modes, root| crate::trust::nearest_approval_mode(modes, root).unwrap_or(default);
    live.keys()
        .chain(saved.keys())
        .map(|root| (root.clone(), governing(live, root).stricter(governing(saved, root))))
        .collect()
}

/// Content identity of settings.json on disk. Missing and unreadable are
/// distinct states: a launch with no file followed by a bash action that
/// leaves a directory (or anything unreadable) at the path is drift that
/// bricks the next launch, and must not compare equal to "still missing".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SettingsFingerprint {
    Missing,
    Unreadable,
    Bytes(u64),
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

fn settings_file_fingerprint(data_dir: &std::path::Path) -> SettingsFingerprint {
    let path = crate::config::settings_path(data_dir);
    match std::fs::read(&path) {
        Ok(bytes) => SettingsFingerprint::Bytes(hash_bytes(&bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => SettingsFingerprint::Missing,
        Err(_) => SettingsFingerprint::Unreadable,
    }
}

/// The form a project's approval mode is keyed by: the canonical path, or the
/// path as given when it cannot be resolved.
pub fn canonical_root(project_root: &std::path::Path) -> PathBuf {
    std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf())
}

impl Core {
    /// The choice for this canonical project or its nearest enclosing one
    /// (see `trust::nearest_approval_mode`), or the settings default.
    /// Never call while holding the settings lock.
    pub fn approval_mode(&self, project_root: &std::path::Path) -> crate::config::ApprovalMode {
        self.approval_mode_canonical(&canonical_root(project_root))
    }

    /// `approval_mode` for a root already passed through `canonical_root`.
    /// It touches no filesystem, so a frontend can read it on every frame.
    /// Never call while holding the settings lock.
    pub fn approval_mode_canonical(&self, canonical_root: &std::path::Path) -> crate::config::ApprovalMode {
        let saved = crate::trust::nearest_approval_mode(
            &self.project_approval_modes.lock().unwrap_or_else(|e| e.into_inner()).live,
            canonical_root,
        );
        saved.unwrap_or_else(|| self.settings.lock().unwrap_or_else(|e| e.into_inner()).approval_mode)
    }

    /// Persist before changing the live gate. A failed save leaves both the
    /// acknowledged choice and the live execution policy unchanged.
    pub fn set_project_approval_mode(
        &self,
        project_root: &std::path::Path,
        mode: crate::config::ApprovalMode,
    ) -> Result<(), String> {
        let mut modes = self.project_approval_modes.lock().unwrap_or_else(|e| e.into_inner());
        let canonical = crate::trust::set_approval_mode(&self.data_dir, project_root, mode)?;
        modes.live.insert(canonical, mode);
        Ok(())
    }

    /// Hold this canonical project, and its subdirectories without a choice
    /// of their own, at `mode` or the mode that governs it now, whichever is
    /// stricter, for the life of this process. Nothing is saved, and a
    /// refresh only ever tightens it (see [`ProjectModes`]). A sub-agent's
    /// worktree is held at the mode of the project it was started from, so a
    /// looser choice saved for a path inside the worktree never applies.
    pub fn hold_approval_mode(&self, project_root: &std::path::Path, mode: crate::config::ApprovalMode) {
        let canonical = canonical_root(project_root);
        // Taken first and released: the settings lock is never held with this one.
        let default = self.settings.lock().unwrap_or_else(|e| e.into_inner()).approval_mode;
        let mut modes = self.project_approval_modes.lock().unwrap_or_else(|e| e.into_inner());
        let now = crate::trust::nearest_approval_mode(&modes.live, &canonical).unwrap_or(default);
        modes.live.insert(canonical, now.stricter(mode));
    }

    /// Read back the choices other processes saved since this one last
    /// looked, and apply each one that is stricter (see [`ProjectModes`]).
    /// Turns call it before each request and tool call, before a queued
    /// batch call starts, and once an approval card is answered; a frontend
    /// calls it where it reads the mode to act on it. The store is read only
    /// when its stamp moved. A missing store, or one that no longer names a
    /// root, drops that root to the settings default wherever that is
    /// stricter (see [`tightened`]). One that no longer parses changes
    /// nothing, as for a process that read it only at launch; the next launch
    /// fails closed on it.
    pub fn refresh_saved_approval_modes(&self) {
        // Taken first and released: the settings lock is never held with this one.
        let default = self.settings.lock().unwrap_or_else(|e| e.into_inner()).approval_mode;
        let mut modes = self.project_approval_modes.lock().unwrap_or_else(|e| e.into_inner());
        // Stamped before the read, so a write landing in between is read again.
        let seen = crate::sessions::stamp(&crate::trust::trust_path(&self.data_dir));
        if seen.is_some() && seen == modes.stamp {
            return;
        }
        if let Ok(saved) = crate::trust::approval_modes(&self.data_dir) {
            modes.live = tightened(&modes.live, &saved, default);
        }
        modes.stamp = seen;
    }

    pub fn new(
        data_dir: PathBuf,
    ) -> Result<(Arc<Self>, mpsc::UnboundedReceiver<AgentEventEnvelope>), String> {
        let (tx, rx) = mpsc::unbounded_channel();
        let _ = std::fs::create_dir_all(&data_dir);
        let settings = crate::config::load(&data_dir)?;
        let core_data_dir = data_dir.clone();
        // Stamped before the read, so a write landing in between is read again.
        let stamp = crate::sessions::stamp(&crate::trust::trust_path(&data_dir));
        let modes = crate::trust::approval_modes(&data_dir)?;
        let core = Arc::new(Self {
            data_dir,
            settings: Mutex::new(settings),
            project_approval_modes: Mutex::new(ProjectModes { live: modes, stamp }),
            sessions: Default::default(),
            running: Default::default(),
            cancel_flags: Default::default(),
            steers: Default::default(),
            approvals: Default::default(),
            sessions_lock: Default::default(),
            session_owners: Default::default(),
            message_restorations: Default::default(),
            settings_disk_fingerprint: Mutex::new(settings_file_fingerprint(&core_data_dir)),
            run_only: Mutex::new(None),
            event_log: Default::default(),
            events: tx,
        });
        Ok((core, rx))
    }

    /// A core for read-only history operations: ones that read the stores and
    /// never contact a provider, run a tool, or start a turn.
    ///
    /// Settings say how to reach an endpoint and what a turn may spend. A
    /// history search uses neither, so an unreadable settings file - a key
    /// from a newer build, a hand edit, a stray comma - must not also make the
    /// project's own history unreadable. This leaves the fail-closed rule
    /// exactly where it earns its keep: the paths that spend money and run
    /// tools.
    ///
    /// The failure is returned, never swallowed. Degrading silently to
    /// defaults would hide a real misconfiguration behind a working search;
    /// the caller reports the reason and answers the question anyway.
    pub fn read_only(
        data_dir: PathBuf,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<AgentEventEnvelope>, Option<String>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let _ = std::fs::create_dir_all(&data_dir);
        let (settings, unreadable) = match crate::config::load(&data_dir) {
            Ok(settings) => (settings, None),
            Err(reason) => (crate::config::Settings::default(), Some(reason)),
        };
        let core_data_dir = data_dir.clone();
        let core = Arc::new(Self {
            data_dir,
            settings: Mutex::new(settings),
            project_approval_modes: Mutex::new(Default::default()),
            sessions: Default::default(),
            running: Default::default(),
            cancel_flags: Default::default(),
            steers: Default::default(),
            approvals: Default::default(),
            sessions_lock: Default::default(),
            session_owners: Default::default(),
            message_restorations: Default::default(),
            settings_disk_fingerprint: Mutex::new(settings_file_fingerprint(&core_data_dir)),
            run_only: Mutex::new(None),
            event_log: Default::default(),
            events: tx,
        });
        (core, rx, unreadable)
    }

    /// Persist settings through the process's own hand: the on-disk
    /// fingerprint is refreshed with the write, so a TUI-authored save never
    /// reads as external drift. A key the command line overrode for this run
    /// is saved with the file's value until the session sets it (see
    /// `run_only`).
    pub fn save_settings(&self, settings: &Settings) -> Result<(), String> {
        let mut run_only = self.run_only.lock().unwrap_or_else(|e| e.into_inner());
        let held = run_only.as_ref().map(|r| r.save(settings));
        let settings = held.as_ref().map_or(settings, |(kept, _)| kept);
        // Hold the fingerprint lock across the write so a concurrent drift
        // check cannot observe the new bytes before this process claims
        // them, and fingerprint the exact bytes WRITTEN - not a re-read of
        // the path, which an external replacement could have swapped in the
        // interval and thereby been adopted as this process's own.
        let mut seen = self
            .settings_disk_fingerprint
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let written = crate::config::save_bytes(&self.data_dir, settings)?;
        *seen = SettingsFingerprint::Bytes(hash_bytes(&written));
        if let Some((_, next)) = held {
            *run_only = Some(next);
        }
        Ok(())
    }

    /// Apply the command line's settings flags once, at launch and before
    /// any turn (the rule is [`crate::config::KeyFlags`]'s): save the keys
    /// settings.json leaves unset, then run with every flag on top. A failed
    /// save changes nothing, here or on disk.
    pub fn apply_key_flags(&self, flags: &crate::config::KeyFlags) -> Result<crate::config::Applied, String> {
        let mut live = self.settings.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = live.clone();
        let (run, applied) = flags.apply(&mut file);
        if !applied.written.is_empty() {
            self.save_settings(&file)?;
        }
        *self.run_only.lock().unwrap_or_else(|e| e.into_inner()) =
            crate::config::RunOnly::new(file, run.clone(), &applied);
        *live = run;
        Ok(applied)
    }

    /// Whether settings.json on disk moved since this process last read or
    /// wrote it. On drift, records the new content (so each distinct change
    /// is reported once) and returns whether the new bytes would parse -
    /// the caller words the receipt. None while the disk matches.
    pub fn settings_disk_changed(&self) -> Option<Result<(), String>> {
        let current = settings_file_fingerprint(&self.data_dir);
        let mut seen = self
            .settings_disk_fingerprint
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if *seen == current {
            return None;
        }
        *seen = current;
        drop(seen);
        Some(crate::config::load(&self.data_dir).map(|_| ()))
    }

    /// Send `event` to the frontend and record it in the session's event log
    /// (`sessions::record_event`), both under one lock, so the log holds the
    /// session's events in the order the frontend reads them.
    pub fn send_agent(&self, session_id: &str, event: AgentEvent) {
        let envelope = AgentEventEnvelope { session_id: session_id.to_string(), event };
        let mut log = self.event_log.lock().unwrap_or_else(|e| e.into_inner());
        crate::sessions::record_event(&mut log, self, &envelope);
        let _ = self.events.send(envelope);
    }

    /// Send a receipt `sessions::record_receipts` already recorded.
    pub(crate) fn send_recorded(&self, receipt: AgentEventEnvelope) {
        let _ = self.events.send(receipt);
    }

    pub fn respond_approval(&self, approval_id: &str, approved: bool) {
        if let Some(tx) = self.approvals.lock().unwrap().remove(approval_id) {
            let _ = tx.send(approved);
        }
    }

    /// Ask the running turn in `session_id` to stop at the next safe point.
    pub fn cancel(&self, session_id: &str) {
        if let Some(token) = self.cancel_flags.lock().unwrap().get(session_id) {
            token.cancel();
        }
    }

    /// Send `text` to the turn running in `session_id`. The turn appends it
    /// as a user message at its next iteration boundary, before the next
    /// request is built, and reports it with `AgentEvent::SteerAccepted`; a
    /// model that stops while a steer waits is sent round again to read it,
    /// unless the turn is on its last iteration. Errors when no turn is
    /// running there. Ok means queued, not read: a turn that ends without
    /// taking the steer (cancelled, failed, out of budget or iterations, or
    /// already past the model's final answer) drops it, and only
    /// SteerAccepted says it entered the transcript (SteerBlocked says its
    /// gate refused it). An accepted steer is saved, so a turn that ends
    /// before the model answers it (out of budget at the next admission)
    /// leaves it for the next turn's first request.
    /// Also errors when the steers already waiting for the turn and `text`
    /// together pass [`MAX_WAITING_STEER_BYTES`]: a frontend that keeps
    /// sending while a long request or tool runs is refused, not buffered
    /// without bound.
    pub fn steer(&self, session_id: &str, text: String) -> Result<(), String> {
        match self.steers.lock().unwrap().get_mut(session_id) {
            Some(queue) => {
                let waiting: usize = queue.iter().map(String::len).sum();
                if waiting + text.len() > MAX_WAITING_STEER_BYTES {
                    return Err(format!(
                        "the steers waiting for this turn would pass {MAX_WAITING_STEER_BYTES} bytes; send this again after the next steer_accepted, steer_blocked, or done"
                    ));
                }
                queue.push(text);
                Ok(())
            }
            None => Err("no turn is running in this session to steer".into()),
        }
    }

    pub fn is_running(&self, session_id: &str) -> bool {
        self.running.lock().unwrap().contains(session_id)
    }
}

/// The most steer text that may wait for one turn at once. What a turn has
/// not yet taken is held in memory, so it is bounded like any other input.
pub const MAX_WAITING_STEER_BYTES: usize = 8 * 1024 * 1024;

/// `~/.openmax`, the single place Open Max keeps its state (settings, sessions,
/// logs).
pub fn default_data_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(".openmax"),
        None => PathBuf::from(".openmax"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Steers wait in memory until the turn's next boundary takes them, so
    /// together they hold at most `MAX_WAITING_STEER_BYTES`: one that would
    /// pass it is refused and leaves the queue as it was, and once the turn
    /// takes what waits there is room again.
    #[test]
    fn steers_waiting_for_one_turn_are_capped_in_bytes() {
        let dir = std::env::temp_dir().join(format!("omx-steer-cap-{}", uuid::Uuid::new_v4()));
        assert_eq!(MAX_WAITING_STEER_BYTES, 8 * 1024 * 1024, "docs/stdio-protocol.md and --spec stdio say 8 MiB");
        let (core, _) = Core::new(dir.clone()).unwrap();
        core.steers.lock().unwrap().insert("s".into(), Vec::new());
        core.steer("s", "x".repeat(MAX_WAITING_STEER_BYTES - 1)).unwrap();
        core.steer("s", "y".into()).expect("exactly at the cap is taken");
        assert!(core.steer("s", "z".into()).is_err(), "one byte past the cap is refused");
        assert_eq!(core.steers.lock().unwrap()["s"].len(), 2, "a refused steer is not queued");
        std::mem::take(core.steers.lock().unwrap().get_mut("s").unwrap());
        core.steer("s", "z".into()).expect("taken steers free their room");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn project_mode_is_scoped_and_saved_before_activation() {
        use crate::config::ApprovalMode;
        let dir = std::env::temp_dir().join(format!("omx-mode-{}", uuid::Uuid::new_v4()));
        let project = dir.join("project");
        let other = dir.join("other");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        let data = dir.join("data");
        let (core, _) = Core::new(data.clone()).unwrap();
        assert!(core.set_project_approval_mode(&project, ApprovalMode::Auto).is_err());
        crate::trust::trust_project(&data, &project).unwrap();
        core.set_project_approval_mode(&project, ApprovalMode::Auto).unwrap();
        let (restarted, _) = Core::new(data.clone()).unwrap();
        assert_eq!(restarted.approval_mode(&project), ApprovalMode::Auto);
        assert_eq!(restarted.approval_mode(&other), ApprovalMode::Ask);
        crate::trust::set_approval_mode(&data, &project, ApprovalMode::Readonly).unwrap();
        assert_eq!(core.approval_mode(&project), ApprovalMode::Auto, "a read alone never goes to the disk");
        assert_eq!(Core::new(data.clone()).unwrap().0.approval_mode(&project), ApprovalMode::Readonly);
        std::fs::remove_file(data.join("trust.json")).unwrap();
        std::fs::create_dir(data.join("trust.json")).unwrap();
        assert!(core.set_project_approval_mode(&project, ApprovalMode::Ask).is_err());
        assert_eq!(core.approval_mode(&project), ApprovalMode::Auto, "a failed save must leave the live policy unchanged");
        assert!(Core::new(data).is_err(), "unreadable authority state fails closed at launch");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A process started in a worktree of a project gates its tools with the
    /// project's choice, from the launch snapshot and after a live selection;
    /// otherwise a delegated headless child under auto declines every write.
    #[test]
    fn project_mode_covers_sessions_in_its_subdirectories() {
        use crate::config::ApprovalMode;
        let dir = std::env::temp_dir().join(format!("omx-mode-sub-{}", uuid::Uuid::new_v4()));
        let project = dir.join("project");
        let worktree = project.join(".worktrees").join("task-a");
        std::fs::create_dir_all(&worktree).unwrap();
        let data = dir.join("data");
        crate::trust::trust_project(&data, &project).unwrap();
        let (core, _) = Core::new(data.clone()).unwrap();
        core.set_project_approval_mode(&project, ApprovalMode::Auto).unwrap();
        assert_eq!(core.approval_mode(&worktree), ApprovalMode::Auto, "live selection");
        assert_eq!(Core::new(data).unwrap().0.approval_mode(&worktree), ApprovalMode::Auto, "launch snapshot");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A held worktree runs at the stricter of the hold and the choice that
    /// governs it: a looser choice saved for a path inside it never loosens
    /// the hold, a stricter one still applies, the hold saves nothing, and a
    /// refresh keeps it.
    #[test]
    fn a_held_mode_is_the_stricter_of_the_hold_and_the_saved_choice() {
        use crate::config::ApprovalMode::{Ask, Auto, Readonly};
        let dir = std::env::temp_dir().join(format!("omx-mode-hold-{}", uuid::Uuid::new_v4()));
        let (project, data) = (dir.join("project"), dir.join("data"));
        let (loose, strict) = (project.join(".worktrees").join("loose"), project.join(".worktrees").join("strict"));
        std::fs::create_dir_all(loose.join("sub")).unwrap();
        std::fs::create_dir_all(&strict).unwrap();
        crate::trust::grant_trust(&data, &project, Readonly).unwrap();
        crate::trust::set_approval_mode(&data, &loose, Auto).unwrap();
        crate::trust::set_approval_mode(&data, &strict, Readonly).unwrap();
        let (core, _) = Core::new(data.clone()).unwrap();
        let store = std::fs::read(crate::trust::trust_path(&data)).unwrap();

        core.hold_approval_mode(&loose, Ask);
        assert_eq!(core.approval_mode(&loose), Ask, "a looser saved choice does not loosen the hold");
        assert_eq!(core.approval_mode(&loose.join("sub")), Ask, "the hold covers the worktree's subdirectories");
        core.hold_approval_mode(&strict, Ask);
        assert_eq!(core.approval_mode(&strict), Readonly, "a stricter saved choice still applies");
        assert_eq!(std::fs::read(crate::trust::trust_path(&data)).unwrap(), store, "a hold saves nothing");

        crate::trust::set_approval_mode(&data, &project, Auto).unwrap();
        core.refresh_saved_approval_modes();
        assert_eq!(core.approval_mode(&loose), Ask, "a refresh never loosens the hold");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every frontend on a data dir saves to one trust store. A save rereads
    /// the store under trust.lock, so a process that loaded it at launch
    /// never writes back a stale copy over another's save. A running process
    /// reads the store back and takes another's choice wherever it is
    /// stricter, so ask selected in one TUI does not leave a sibling session
    /// on the project running auto. It never takes a looser one: the store
    /// is a file a tool can write, and only its own selection loosens it.
    #[test]
    fn a_mode_another_process_saved_applies_only_where_it_is_stricter() {
        use crate::config::ApprovalMode::{Ask, Auto, Readonly};
        let dir = std::env::temp_dir().join(format!("omx-mode-sibling-{}", uuid::Uuid::new_v4()));
        let (project, other, data) = (dir.join("project"), dir.join("other"), dir.join("data"));
        let nested = project.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        crate::trust::grant_trust(&data, &project, Auto).unwrap();
        crate::trust::grant_trust(&data, &other, Auto).unwrap();
        let (a, _) = Core::new(data.clone()).unwrap();
        let (b, _) = Core::new(data.clone()).unwrap();

        b.set_project_approval_mode(&project, Ask).unwrap();
        a.set_project_approval_mode(&other, Readonly).unwrap();
        let fresh = Core::new(data.clone()).unwrap().0;
        assert_eq!((fresh.approval_mode(&project), fresh.approval_mode(&other)), (Ask, Readonly), "interleaved saves keep both");

        a.refresh_saved_approval_modes();
        assert_eq!(a.approval_mode(&project), Ask, "the sibling's stricter choice applies here");
        assert_eq!(a.approval_mode(&other), Readonly, "next to this process's own");

        b.set_project_approval_mode(&project, Auto).unwrap();
        b.set_project_approval_mode(&other, Auto).unwrap();
        a.refresh_saved_approval_modes();
        assert_eq!((a.approval_mode(&project), a.approval_mode(&other)), (Ask, Readonly), "a looser saved choice is never taken");

        b.set_project_approval_mode(&nested, Readonly).unwrap();
        a.refresh_saved_approval_modes();
        assert_eq!((a.approval_mode(&nested), a.approval_mode(&project)), (Readonly, Ask), "a nested root's choice covers its subtree only");

        a.set_project_approval_mode(&project, Auto).unwrap();
        a.refresh_saved_approval_modes();
        assert_eq!((a.approval_mode(&project), a.approval_mode(&nested)), (Auto, Readonly), "this process's own selection loosens it");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Launch flags fill the keys settings.json leaves unset through the
    /// core's own save (never drift), and override the rest for the run.
    /// An override stays the run's: a save from the session, as `/model`
    /// makes, writes the file's `api_key` and `context_tokens`, which no
    /// in-session command sets, and the model it is given; the run keeps its
    /// own values.
    #[test]
    fn launch_flags_fill_unset_keys_and_never_save_an_override() {
        let data = std::env::temp_dir().join(format!("omx-key-flags-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(
            data.join("settings.json"),
            r#"{"base_url":"http://file/v1","api_key":"$FILE_KEY","context_tokens":1000}"#,
        )
        .unwrap();
        let (core, _) = Core::new(data.clone()).unwrap();
        let flags = crate::config::KeyFlags {
            base_url: Some("http://file/v1".into()),
            api_key: Some("$FLAG_KEY".into()),
            model: Some("m".into()),
            context_tokens: Some(2000),
            ..Default::default()
        };
        let applied = core.apply_key_flags(&flags).unwrap();
        assert_eq!(applied.written, ["model"]);
        assert_eq!(applied.overridden, ["api_key", "context_tokens"]);
        assert!(core.settings_disk_changed().is_none(), "the core's own write is not drift");
        let file = |disk: crate::config::Settings| (disk.base_url, disk.api_key, disk.model, disk.context_tokens);
        let kept = |model: &str| ("http://file/v1".to_string(), Some("$FILE_KEY".to_string()), model.to_string(), Some(1000));
        assert_eq!(file(crate::config::load(&data).unwrap()), kept("m"), "only the unset key is written");

        let mut next = core.settings.lock().unwrap().clone();
        let run = ("http://file/v1".to_string(), Some("$FLAG_KEY".to_string()), "m".to_string(), Some(2000));
        assert_eq!(file(next.clone()), run, "the run uses every flag");
        next.model = "m2".into();
        core.save_settings(&next).unwrap();
        assert_eq!(file(crate::config::load(&data).unwrap()), kept("m2"), "a session save keeps the file's key and window");
        assert!(core.settings_disk_changed().is_none());

        // With nothing overridden, a save writes what it is given.
        let (plain, _) = Core::new(data.clone()).unwrap();
        let mut next = plain.settings.lock().unwrap().clone();
        next.base_url = "http://saved/v1".into();
        plain.save_settings(&next).unwrap();
        assert_eq!(crate::config::load(&data).unwrap().base_url, "http://saved/v1");
        let _ = std::fs::remove_dir_all(data);
    }

    /// A data dir holding `json` as settings.json, launched with `flags` the
    /// way `main` applies them.
    fn launched(json: &str, flags: crate::config::KeyFlags) -> (Arc<Core>, PathBuf) {
        let data = std::env::temp_dir().join(format!("omx-run-only-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("settings.json"), json).unwrap();
        let (core, _) = Core::new(data.clone()).unwrap();
        core.apply_key_flags(&flags).unwrap();
        (core, data)
    }

    /// A save from the session as `/model` and `/provider` make one: the live
    /// settings with one change, adopted once written.
    fn session_save(core: &Core, change: impl FnOnce(&mut crate::config::Settings)) {
        let mut next = core.settings.lock().unwrap().clone();
        change(&mut next);
        core.save_settings(&next).unwrap();
        *core.settings.lock().unwrap() = next;
    }

    /// `--base-url` beside a `settings.provider` sets the provider aside for
    /// this run only. A `/model` save in that session carries the run's
    /// provider (none) and base_url, and must still leave the file selecting
    /// its provider, or the next plain launch has no endpoint. The model is
    /// picked on the run's server, so the file keeps its own.
    #[test]
    fn a_session_save_keeps_the_provider_a_flag_set_aside() {
        let flags = crate::config::KeyFlags {
            base_url: Some("http://local/v1".into()),
            context_tokens: Some(8000),
            ..Default::default()
        };
        let (core, data) = launched(r#"{"provider":"p","model":"x"}"#, flags);
        session_save(&core, |s| s.model = "m2".into());
        let disk = crate::config::load(&data).unwrap();
        assert_eq!(
            (disk.provider.as_deref(), disk.base_url.as_str(), disk.model.as_str(), disk.context_tokens),
            (Some("p"), "", "x", None),
            "the file is as launch left it"
        );
        let run = core.settings.lock().unwrap().clone();
        assert_eq!((run.provider, run.base_url.as_str(), run.model.as_str()), (None, "http://local/v1", "m2"));
        let _ = std::fs::remove_dir_all(data);
    }

    /// `--provider q` on a flat install picks another endpoint for this run
    /// only: a `/model` save carries the run's provider, and the file must
    /// stay flat, with its own model, since the one picked is q's.
    #[test]
    fn a_session_save_never_writes_a_provider_flag() {
        let flags = crate::config::KeyFlags { provider: Some("q".into()), ..Default::default() };
        let (core, data) = launched(r#"{"base_url":"http://file/v1","model":"x","context_tokens":1000}"#, flags);
        session_save(&core, |s| s.model = "m2".into());
        let disk = crate::config::load(&data).unwrap();
        assert_eq!((disk.provider.as_deref(), disk.base_url.as_str(), disk.model.as_str()), (None, "http://file/v1", "x"));
        assert_eq!(core.settings.lock().unwrap().provider.as_deref(), Some("q"), "the run keeps its provider");
        let _ = std::fs::remove_dir_all(data);
    }

    /// A model picked while the run talks to the command line's server
    /// belongs to that server, so a save keeps the file's model next to the
    /// file's endpoint. A flag that leaves the server as it is (`--api-key`)
    /// leaves the model the session's to save.
    #[test]
    fn a_model_picked_on_the_command_lines_server_stays_with_the_run() {
        let flat = r#"{"base_url":"http://file/v1","api_key":"$FILE_KEY","model":"x","context_tokens":1000}"#;
        let flags = crate::config::KeyFlags { base_url: Some("http://flag/v1".into()), ..Default::default() };
        let (core, data) = launched(flat, flags);
        session_save(&core, |s| s.model = "m2".into());
        let disk = crate::config::load(&data).unwrap();
        assert_eq!((disk.base_url.as_str(), disk.model.as_str()), ("http://file/v1", "x"));
        let _ = std::fs::remove_dir_all(data);

        let flags = crate::config::KeyFlags { api_key: Some("$FLAG_KEY".into()), ..Default::default() };
        let (core, data) = launched(flat, flags);
        session_save(&core, |s| s.model = "m2".into());
        let disk = crate::config::load(&data).unwrap();
        assert_eq!((disk.api_key.as_deref(), disk.model.as_str()), (Some("$FILE_KEY"), "m2"));
        let _ = std::fs::remove_dir_all(data);
    }

    /// A key the session sets to another value is the session's from then on:
    /// `/model` after `--model` saves each model picked, the launch value
    /// included once the session has moved off it. `/provider r` on a run
    /// whose provider the command line chose saves r and its model, and the
    /// file then selects the run's server, so later picks are saved too,
    /// while a flag the session never set (`--context-tokens`) stays the
    /// run's through every later save.
    #[test]
    fn a_key_the_session_sets_is_saved_from_then_on() {
        let flags = crate::config::KeyFlags { model: Some("m".into()), ..Default::default() };
        let (core, data) = launched(r#"{"base_url":"http://file/v1","model":"x","context_tokens":1000}"#, flags);
        let model = |data: &PathBuf| crate::config::load(data).unwrap().model;
        session_save(&core, |s| s.approval_mode = crate::config::ApprovalMode::Readonly);
        assert_eq!(model(&data), "x", "a save that leaves the model alone keeps the file's");
        session_save(&core, |s| s.model = "m2".into());
        assert_eq!(model(&data), "m2");
        session_save(&core, |s| s.model = "m".into());
        assert_eq!(model(&data), "m");
        let _ = std::fs::remove_dir_all(data);

        let flags = crate::config::KeyFlags { provider: Some("q".into()), context_tokens: Some(9000), ..Default::default() };
        let (core, data) = launched(r#"{"provider":"p","model":"x"}"#, flags);
        session_save(&core, |s| {
            s.provider = Some("r".into());
            s.model = "r1".into();
        });
        let disk = crate::config::load(&data).unwrap();
        assert_eq!((disk.provider.as_deref(), disk.model.as_str()), (Some("r"), "r1"));
        session_save(&core, |s| s.model = "r2".into());
        let disk = crate::config::load(&data).unwrap();
        assert_eq!(disk.model, "r2", "the file selects the run's server now");
        assert_eq!(disk.context_tokens, None, "the window flag stays the run's");
        let _ = std::fs::remove_dir_all(data);

        // `/provider r` keeps the run's model when r serves it too. Saved, r
        // runs that model: the file's may be one r does not serve.
        let both = crate::config::KeyFlags { provider: Some("q".into()), model: Some("common".into()), ..Default::default() };
        let model_only = crate::config::KeyFlags { model: Some("common".into()), ..Default::default() };
        for flags in [both, model_only] {
            let (core, data) = launched(r#"{"provider":"p","model":"x"}"#, flags.clone());
            session_save(&core, |s| s.provider = Some("r".into()));
            let disk = crate::config::load(&data).unwrap();
            assert_eq!((disk.provider.as_deref(), disk.model.as_str()), (Some("r"), "common"), "{flags:?}");
            let _ = std::fs::remove_dir_all(data);
        }
    }

    #[tokio::test]
    async fn cancel_token_wakes_waiters_immediately() {
        let token = Arc::new(CancelToken::default());
        let waiter_token = token.clone();
        let waiter = tokio::spawn(async move { waiter_token.cancelled().await });
        tokio::task::yield_now().await;
        token.cancel();
        tokio::time::timeout(std::time::Duration::from_millis(100), waiter)
            .await
            .expect("waiter must wake without polling delay")
            .unwrap();
        // An already-cancelled token resolves without waiting at all.
        tokio::time::timeout(std::time::Duration::from_millis(10), token.cancelled())
            .await
            .expect("immediate resolution");
        assert!(token.is_cancelled());
    }
}
