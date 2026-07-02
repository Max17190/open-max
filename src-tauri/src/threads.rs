use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::harness::types::ChatMessage;
use crate::settings::app_data_dir;
use crate::state::AppState;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThreadMeta {
    pub id: String,
    pub project_id: String,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
}

pub const UNTITLED: &str = "New thread";

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn index_path(app: &AppHandle) -> PathBuf {
    app_data_dir(app).join("threads.json")
}

fn threads_dir(app: &AppHandle) -> PathBuf {
    let dir = app_data_dir(app).join("threads");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn messages_path(app: &AppHandle, id: &str) -> PathBuf {
    threads_dir(app).join(format!("{id}.messages.json"))
}

fn items_path(app: &AppHandle, id: &str) -> PathBuf {
    threads_dir(app).join(format!("{id}.items.json"))
}

fn load_index(app: &AppHandle) -> Vec<ThreadMeta> {
    std::fs::read_to_string(index_path(app))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_index(app: &AppHandle, metas: &[ThreadMeta]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(metas).map_err(|e| e.to_string())?;
    std::fs::write(index_path(app), json).map_err(|e| e.to_string())
}

/// Read-modify-write the index under the state lock so concurrent agent
/// turns can't clobber each other's metadata updates.
fn with_index<R>(app: &AppHandle, f: impl FnOnce(&mut Vec<ThreadMeta>) -> R) -> Result<R, String> {
    let state = app.state::<AppState>();
    let _guard = state.threads_lock.lock().unwrap();
    let mut metas = load_index(app);
    let result = f(&mut metas);
    save_index(app, &metas)?;
    Ok(result)
}

pub fn list(app: &AppHandle, project_id: &str) -> Vec<ThreadMeta> {
    let mut metas: Vec<ThreadMeta> = load_index(app)
        .into_iter()
        .filter(|m| m.project_id == project_id)
        .collect();
    metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    metas
}

pub fn create(app: &AppHandle, project_id: String) -> Result<ThreadMeta, String> {
    let meta = ThreadMeta {
        id: uuid::Uuid::new_v4().to_string(),
        project_id,
        title: UNTITLED.into(),
        created_at: now(),
        updated_at: now(),
    };
    let m = meta.clone();
    with_index(app, move |metas| metas.push(m))?;
    Ok(meta)
}

pub fn delete(app: &AppHandle, id: &str) -> Result<(), String> {
    with_index(app, |metas| metas.retain(|m| m.id != id))?;
    let _ = std::fs::remove_file(messages_path(app, id));
    let _ = std::fs::remove_file(items_path(app, id));
    Ok(())
}

/// Set the title from the first user message, once.
pub fn set_title_if_new(app: &AppHandle, id: &str, title: &str) {
    let title = title.trim().chars().take(48).collect::<String>();
    if title.is_empty() {
        return;
    }
    let _ = with_index(app, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            if m.title == UNTITLED {
                m.title = title;
            }
            m.updated_at = now();
        }
    });
}

pub fn touch(app: &AppHandle, id: &str) {
    let _ = with_index(app, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            m.updated_at = now();
        }
    });
}

pub fn load_messages(app: &AppHandle, id: &str) -> Option<Vec<ChatMessage>> {
    std::fs::read_to_string(messages_path(app, id))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

pub fn save_messages(app: &AppHandle, id: &str, messages: &[ChatMessage]) {
    if let Ok(json) = serde_json::to_string(messages) {
        let _ = std::fs::write(messages_path(app, id), json);
    }
}

pub fn load_items(app: &AppHandle, id: &str) -> serde_json::Value {
    std::fs::read_to_string(items_path(app, id))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(serde_json::Value::Null)
}

pub fn save_items(app: &AppHandle, id: &str, items: &serde_json::Value) -> Result<(), String> {
    let json = serde_json::to_string(items).map_err(|e| e.to_string())?;
    std::fs::write(items_path(app, id), json).map_err(|e| e.to_string())
}
