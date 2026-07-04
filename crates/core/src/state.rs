use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot};

use crate::config::Settings;
use crate::hf::DownloadProc;
use crate::mlx::{MlxEvent, MlxProc};
use crate::registry::Registry;
use crate::types::{AgentEvent, AgentEventEnvelope, ChatMessage};

/// In-memory state of one agent session.
#[derive(Default, Clone)]
pub struct SessionData {
    pub messages: Vec<ChatMessage>,
    /// The tool registry frozen at session creation; its serialized schemas
    /// are part of the prompt-cache prefix and must never change mid-session.
    pub registry: Arc<Registry>,
    /// Messages already written to disk; enables append-only persistence.
    pub persisted_count: usize,
    /// File content captured on first touch by a mutating tool, so the UI can
    /// show a cumulative diff per file at any point in the session.
    pub snapshots: HashMap<String, String>,
}

/// Progress of a model download managed by `hf.rs`.
#[derive(Clone, Debug)]
pub enum DownloadEvent {
    Progress { repo: String, done_bytes: u64, total_bytes: u64 },
    Done { repo: String, ok: bool, message: String },
}

/// Everything the core emits toward the UI, multiplexed on one channel.
#[derive(Clone, Debug)]
pub enum CoreEvent {
    Agent(AgentEventEnvelope),
    Mlx(MlxEvent),
    Download(DownloadEvent),
}

/// Shared core state. The frontend owns an `Arc<Core>` plus the receiving half
/// of the event channel; background tasks clone the `Arc`.
pub struct Core {
    pub data_dir: PathBuf,
    pub settings: Mutex<Settings>,
    /// Live sessions keyed by session id; hydrated from disk on first use.
    pub sessions: tokio::sync::Mutex<HashMap<String, SessionData>>,
    /// Sessions with an agent turn currently in flight.
    pub running: Mutex<HashSet<String>>,
    pub cancel_flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Pending tool-approval prompts awaiting a user decision.
    pub approvals: Mutex<HashMap<String, oneshot::Sender<bool>>>,
    /// Serializes read-modify-write cycles on the session index file.
    pub sessions_lock: Mutex<()>,
    pub mlx: Mutex<MlxProc>,
    /// At most one model download runs at a time.
    pub download: Mutex<Option<DownloadProc>>,
    events: mpsc::UnboundedSender<CoreEvent>,
}

impl Core {
    pub fn new(data_dir: PathBuf) -> (Arc<Self>, mpsc::UnboundedReceiver<CoreEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let _ = std::fs::create_dir_all(&data_dir);
        let settings = crate::config::load(&data_dir);
        let core = Arc::new(Self {
            data_dir,
            settings: Mutex::new(settings),
            sessions: Default::default(),
            running: Default::default(),
            cancel_flags: Default::default(),
            approvals: Default::default(),
            sessions_lock: Default::default(),
            mlx: Default::default(),
            download: Default::default(),
            events: tx,
        });
        (core, rx)
    }

    pub fn send(&self, event: CoreEvent) {
        let _ = self.events.send(event);
    }

    pub fn send_agent(&self, session_id: &str, event: AgentEvent) {
        self.send(CoreEvent::Agent(AgentEventEnvelope {
            session_id: session_id.to_string(),
            event,
        }));
    }

    pub fn send_mlx(&self, event: MlxEvent) {
        self.send(CoreEvent::Mlx(event));
    }

    pub fn respond_approval(&self, approval_id: &str, approved: bool) {
        if let Some(tx) = self.approvals.lock().unwrap().remove(approval_id) {
            let _ = tx.send(approved);
        }
    }

    /// Ask the running turn in `session_id` to stop at the next safe point.
    pub fn cancel(&self, session_id: &str) {
        if let Some(flag) = self.cancel_flags.lock().unwrap().get(session_id) {
            flag.store(true, Ordering::Relaxed);
        }
    }

    pub fn is_running(&self, session_id: &str) -> bool {
        self.running.lock().unwrap().contains(session_id)
    }
}

/// `~/.openmax`, the single place Open Max keeps its state (settings, sessions,
/// the managed Python environment, logs).
pub fn default_data_dir() -> PathBuf {
    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(".openmax"),
        None => PathBuf::from(".openmax"),
    }
}
