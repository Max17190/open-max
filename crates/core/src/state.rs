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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot};

use crate::config::Settings;
use crate::registry::Registry;
use crate::types::{AgentEvent, AgentEventEnvelope, ChatMessage};

/// One extension generation exactly as a freeze captured it: for each file,
/// its path, sha256, and bytes.
pub type ExtensionGeneration = Vec<(PathBuf, String, Vec<u8>)>;

/// In-memory state of one agent session.
#[derive(Default, Clone)]
pub struct SessionData {
    pub messages: Vec<ChatMessage>,
    /// The tool registry frozen at session creation; its serialized schemas
    /// are part of the prompt-cache prefix and must never change mid-session.
    pub registry: Arc<Registry>,
    /// Where the frozen prompt prefix's tokens go, for /context.
    pub prompt_breakdown: Arc<crate::prompt::PromptBreakdown>,
    /// Messages already written to disk; enables append-only persistence.
    pub persisted_count: usize,
    /// File content captured on first touch by a mutating tool, so the UI can
    /// show a cumulative diff per file at any point in the session.
    pub snapshots: HashMap<String, String>,
    /// Process-unique id of the turn that last took `messages`; a restore is
    /// only valid while this still matches the taker (guards against a newer
    /// turn or a recreated session reusing the id).
    pub take_seq: u64,
    /// Whether this session already reported that its tool schemas outgrew the
    /// context window. The condition holds on every turn once it holds at all,
    /// so the advisory is emitted once and not per turn.
    pub schemas_over_budget_reported: bool,
    /// Whether this session already reported that a reply's tool calls were
    /// recovered from its text (fallback.rs) rather than arriving as API
    /// `tool_calls`. Advisory, once per session: the condition says something
    /// about the endpoint and model pairing, not about one reply.
    pub fallback_recovery_reported: bool,
    /// Whether the ledger has reconciled with the extension files this session
    /// froze. False until the first turn start: a freeze reads disk directly,
    /// so changes made while no session was running would otherwise never be
    /// recorded - and the next mid-turn sync would sweep them up as the
    /// agent's own work.
    pub ledger_synced: bool,
    /// Deferred ledger syncs, oldest first: (extension generation, actor). A
    /// sync that cannot land is held here with the attribution it was owed,
    /// and every sync path drains this queue in order before adding its own
    /// claim - the head must never advance past an unlanded one, or its
    /// changes get recorded later under whoever syncs next, and that
    /// misattribution is permanent. Every distinct generation observed
    /// across a broken window stays queued (dropping one would erase its
    /// content from history); only an entry identical to the queue tail is
    /// skipped, which is what keeps unchanged turn starts from growing this
    /// by one per turn.
    pub pending_syncs: Vec<(ExtensionGeneration, crate::ledger::Actor)>,
    /// Content hashes of policy notices (inert allow rules, hooks that did
    /// not load) already narrated to the MODEL this session. The condition
    /// holds every turn once it holds at all, so the transcript gets one
    /// note per distinct notice - the UI keeps its per-turn events.
    pub reported_policy_notices: HashSet<u64>,
    /// Ledger approval-event ids this session has already accounted for:
    /// seeded from the chain at build, then advanced as new events are
    /// narrated. An approval recorded outside the running session (a human
    /// at another terminal, #199) reaches it through no other channel - the
    /// refreeze receipt names file changes, not approvals - so the turn
    /// start names any it has not seen.
    pub seen_ledger_events: HashSet<u64>,
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
    /// Approval mode for this run only, set by a front end's cycle key or an
    /// "allow for run" answer. Deliberately outside [`Settings`]: every save
    /// path serializes that whole struct, so a run-scoped widening kept there
    /// would ride along on the next unrelated `/model` or `/provider` write
    /// and outlive the run it was promised to. Read through
    /// [`Core::approval_mode`], never off `settings` directly.
    run_approval_mode: Mutex<Option<crate::config::ApprovalMode>>,
    /// Live sessions keyed by session id; hydrated from disk on first use.
    pub sessions: tokio::sync::Mutex<HashMap<String, SessionData>>,
    /// Sessions with an agent turn currently in flight.
    pub running: Mutex<HashSet<String>>,
    pub cancel_flags: Mutex<HashMap<String, Arc<CancelToken>>>,
    /// Pending tool-approval prompts awaiting a user decision.
    pub approvals: Mutex<HashMap<String, oneshot::Sender<bool>>>,
    /// Serializes read-modify-write cycles on the session index file.
    pub sessions_lock: Mutex<()>,
    /// Content hash of settings.json as this process last knew it: the bytes
    /// read at launch, refreshed by [`Core::save_settings`]. Settings are
    /// launch-frozen, so any other change to the file is drift this process
    /// will never adopt - the drift receipt tells the model so, including
    /// the brick warning when the new bytes would not even parse. Lock
    /// discipline: never held across the `settings` lock (see
    /// `approval_mode`'s note; this mutex is leaf-only).
    settings_disk_fingerprint: Mutex<SettingsFingerprint>,
    events: mpsc::UnboundedSender<AgentEventEnvelope>,
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

impl Core {
    /// The approval mode in force: a run-scoped override if one is set,
    /// otherwise what `settings.json` says. Every gate reads this, so where a
    /// mode came from never changes what it means.
    ///
    /// Takes the `settings` lock on the fallback path, so never call it while
    /// already holding that lock: it is a plain non-reentrant mutex.
    pub fn approval_mode(&self) -> crate::config::ApprovalMode {
        self.run_approval_mode
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .unwrap_or_else(|| {
                self.settings
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .approval_mode
            })
    }

    /// Set the mode for this run without touching what is on disk.
    pub fn set_run_approval_mode(&self, mode: crate::config::ApprovalMode) {
        *self
            .run_approval_mode
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(mode);
    }

    /// Drop the run override so the persisted mode governs again. Callers that
    /// write `settings.approval_mode` must call this, or the explicit,
    /// persisted choice would stay masked by a stale run-scoped one.
    pub fn clear_run_approval_mode(&self) {
        *self
            .run_approval_mode
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
    }

    pub fn new(
        data_dir: PathBuf,
    ) -> Result<(Arc<Self>, mpsc::UnboundedReceiver<AgentEventEnvelope>), String> {
        let (tx, rx) = mpsc::unbounded_channel();
        let _ = std::fs::create_dir_all(&data_dir);
        let settings = crate::config::load(&data_dir)?;
        let core_data_dir = data_dir.clone();
        let core = Arc::new(Self {
            data_dir,
            settings: Mutex::new(settings),
            run_approval_mode: Mutex::new(None),
            sessions: Default::default(),
            running: Default::default(),
            cancel_flags: Default::default(),
            approvals: Default::default(),
            sessions_lock: Default::default(),
            settings_disk_fingerprint: Mutex::new(settings_file_fingerprint(&core_data_dir)),
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
    /// project's own history unreadable. `--ledger` already reads its store
    /// without loading settings at all; this puts `--recall` on the same
    /// footing, and leaves the fail-closed rule exactly where it earns its
    /// keep: the paths that spend money and run tools.
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
            run_approval_mode: Mutex::new(None),
            sessions: Default::default(),
            running: Default::default(),
            cancel_flags: Default::default(),
            approvals: Default::default(),
            sessions_lock: Default::default(),
            settings_disk_fingerprint: Mutex::new(settings_file_fingerprint(&core_data_dir)),
            events: tx,
        });
        (core, rx, unreadable)
    }

    /// Persist settings through the process's own hand: the on-disk
    /// fingerprint is refreshed with the write, so a TUI-authored save never
    /// reads as external drift.
    pub fn save_settings(&self, settings: &Settings) -> Result<(), String> {
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
        Ok(())
    }

    /// Adopt these exact settings as this process's own after a save that
    /// went through `config::save` directly (an existing helper with its own
    /// tests). Fingerprints the serialization of what was saved, never a
    /// re-read of the path.
    pub fn adopt_saved_settings(&self, settings: &Settings) {
        if let Ok(json) = serde_json::to_string_pretty(settings) {
            *self
                .settings_disk_fingerprint
                .lock()
                .unwrap_or_else(|e| e.into_inner()) =
                SettingsFingerprint::Bytes(hash_bytes(json.as_bytes()));
        }
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

    pub fn send_agent(&self, session_id: &str, event: AgentEvent) {
        let _ = self.events.send(AgentEventEnvelope {
            session_id: session_id.to_string(),
            event,
        });
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

    pub fn is_running(&self, session_id: &str) -> bool {
        self.running.lock().unwrap().contains(session_id)
    }
}

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
