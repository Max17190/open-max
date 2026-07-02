use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::state::Core;
use crate::types::ChatMessage;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    /// Absolute path of the project the session ran in.
    pub project: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
}

pub const UNTITLED: &str = "New session";

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn index_path(core: &Core) -> PathBuf {
    sessions_dir(core).join("index.json")
}

fn sessions_dir(core: &Core) -> PathBuf {
    let dir = core.data_dir.join("sessions");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn messages_path(core: &Core, id: &str) -> PathBuf {
    sessions_dir(core).join(format!("{id}.messages.json"))
}

fn load_index(core: &Core) -> Vec<SessionMeta> {
    std::fs::read_to_string(index_path(core))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_index(core: &Core, metas: &[SessionMeta]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(metas).map_err(|e| e.to_string())?;
    std::fs::write(index_path(core), json).map_err(|e| e.to_string())
}

/// Read-modify-write the index under the state lock so concurrent agent
/// turns can't clobber each other's metadata updates.
fn with_index<R>(core: &Core, f: impl FnOnce(&mut Vec<SessionMeta>) -> R) -> Result<R, String> {
    let _guard = core.sessions_lock.lock().unwrap();
    let mut metas = load_index(core);
    let result = f(&mut metas);
    save_index(core, &metas)?;
    Ok(result)
}

/// Sessions for one project, most recently updated first.
pub fn list(core: &Core, project: &str) -> Vec<SessionMeta> {
    let mut metas: Vec<SessionMeta> = load_index(core)
        .into_iter()
        .filter(|m| m.project == project)
        .collect();
    metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    metas
}

/// Most recent session for a project, if any (used by --continue).
pub fn latest(core: &Core, project: &str) -> Option<SessionMeta> {
    list(core, project).into_iter().next()
}

pub fn create(core: &Core, project: String) -> Result<SessionMeta, String> {
    let meta = SessionMeta {
        id: uuid::Uuid::new_v4().to_string(),
        project,
        title: UNTITLED.into(),
        created_at: now(),
        updated_at: now(),
    };
    let m = meta.clone();
    with_index(core, move |metas| metas.push(m))?;
    Ok(meta)
}

pub fn delete(core: &Core, id: &str) -> Result<(), String> {
    with_index(core, |metas| metas.retain(|m| m.id != id))?;
    let _ = std::fs::remove_file(messages_path(core, id));
    Ok(())
}

/// Set the title from the first user message, once.
pub fn set_title_if_new(core: &Core, id: &str, title: &str) {
    let title = title.trim().chars().take(48).collect::<String>();
    if title.is_empty() {
        return;
    }
    let _ = with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            if m.title == UNTITLED {
                m.title = title;
            }
            m.updated_at = now();
        }
    });
}

pub fn touch(core: &Core, id: &str) {
    let _ = with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            m.updated_at = now();
        }
    });
}

pub fn load_messages(core: &Core, id: &str) -> Option<Vec<ChatMessage>> {
    std::fs::read_to_string(messages_path(core, id))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

pub fn save_messages(core: &Core, id: &str, messages: &[ChatMessage]) {
    if let Ok(json) = serde_json::to_string(messages) {
        let _ = std::fs::write(messages_path(core, id), json);
    }
}
