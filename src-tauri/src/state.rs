use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;

use crate::harness::types::ChatMessage;
use crate::mlx::MlxProc;
use crate::settings::Settings;

/// In-memory state of one agent thread.
#[derive(Default, Clone)]
pub struct SessionData {
    pub messages: Vec<ChatMessage>,
    /// File content captured on first touch by a mutating tool, so the UI can
    /// show a cumulative diff per file at any point in the thread.
    pub snapshots: HashMap<String, String>,
}

#[derive(Default)]
pub struct AppState {
    /// Live sessions keyed by thread id; hydrated from disk on first use.
    pub sessions: tokio::sync::Mutex<HashMap<String, SessionData>>,
    /// Threads with an agent turn currently in flight.
    pub running: Mutex<HashSet<String>>,
    pub cancel_flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Pending tool-approval prompts awaiting a user decision.
    pub approvals: Mutex<HashMap<String, oneshot::Sender<bool>>>,
    /// Serializes read-modify-write cycles on the thread index file.
    pub threads_lock: Mutex<()>,
    pub settings: Mutex<Settings>,
    pub mlx: Mutex<MlxProc>,
}
