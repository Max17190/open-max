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
    events: mpsc::UnboundedSender<AgentEventEnvelope>,
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
        let core = Arc::new(Self {
            data_dir,
            settings: Mutex::new(settings),
            run_approval_mode: Mutex::new(None),
            sessions: Default::default(),
            running: Default::default(),
            cancel_flags: Default::default(),
            approvals: Default::default(),
            sessions_lock: Default::default(),
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
        let core = Arc::new(Self {
            data_dir,
            settings: Mutex::new(settings),
            run_approval_mode: Mutex::new(None),
            sessions: Default::default(),
            running: Default::default(),
            cancel_flags: Default::default(),
            approvals: Default::default(),
            sessions_lock: Default::default(),
            events: tx,
        });
        (core, rx, unreadable)
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
