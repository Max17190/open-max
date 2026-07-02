mod harness;
mod mlx;
mod projects;
mod settings;
mod state;
mod threads;

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use tauri::{AppHandle, Manager, State};

use crate::settings::Settings;
use crate::state::AppState;

// ---------- settings ----------

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings.lock().unwrap().clone()
}

#[tauri::command]
fn set_settings(app: AppHandle, state: State<'_, AppState>, new_settings: Settings) -> Result<(), String> {
    settings::save(&app, &new_settings)?;
    *state.settings.lock().unwrap() = new_settings;
    Ok(())
}

// ---------- projects ----------

#[tauri::command]
fn list_projects(app: AppHandle) -> Vec<projects::Project> {
    projects::load(&app)
}

#[tauri::command]
fn add_project(app: AppHandle, path: String) -> Result<Vec<projects::Project>, String> {
    projects::add(&app, path)
}

#[tauri::command]
fn remove_project(app: AppHandle, id: String) -> Result<Vec<projects::Project>, String> {
    projects::remove(&app, &id)
}

#[tauri::command]
fn file_tree(project_path: String, rel: String) -> Result<Vec<projects::TreeEntry>, String> {
    projects::list_tree(&PathBuf::from(project_path), &rel)
}

#[tauri::command]
fn read_project_file(project_path: String, rel: String) -> Result<String, String> {
    projects::read_text(&PathBuf::from(project_path), &rel)
}

// ---------- threads ----------

#[tauri::command]
fn list_threads(app: AppHandle, project_id: String) -> Vec<threads::ThreadMeta> {
    threads::list(&app, &project_id)
}

#[tauri::command]
fn create_thread(app: AppHandle, project_id: String) -> Result<threads::ThreadMeta, String> {
    threads::create(&app, project_id)
}

#[tauri::command]
async fn delete_thread(app: AppHandle, thread_id: String) -> Result<(), String> {
    let state = app.state::<AppState>();
    if let Some(flag) = state.cancel_flags.lock().unwrap().get(&thread_id) {
        flag.store(true, Ordering::Relaxed);
    }
    state.sessions.lock().await.remove(&thread_id);
    threads::delete(&app, &thread_id)
}

#[tauri::command]
fn load_thread_items(app: AppHandle, thread_id: String) -> serde_json::Value {
    threads::load_items(&app, &thread_id)
}

#[tauri::command]
fn save_thread_items(app: AppHandle, thread_id: String, items: serde_json::Value) -> Result<(), String> {
    threads::save_items(&app, &thread_id, &items)
}

/// Cumulative diff of a file since the thread first touched it.
#[tauri::command]
async fn thread_file_diff(
    app: AppHandle,
    thread_id: String,
    project_path: String,
    rel: String,
) -> Result<harness::tools::DiffInfo, String> {
    let state = app.state::<AppState>();
    let snapshot = {
        let sessions = state.sessions.lock().await;
        sessions
            .get(&thread_id)
            .and_then(|d| d.snapshots.get(&rel).cloned())
    }
    .ok_or("no recorded changes for this file")?;
    let current = std::fs::read_to_string(PathBuf::from(&project_path).join(&rel)).unwrap_or_default();
    Ok(harness::tools::diff_strings(&rel, &snapshot, &current))
}

// ---------- agent ----------

#[tauri::command]
fn send_message(app: AppHandle, session_id: String, project_path: String, text: String) -> Result<(), String> {
    let root = PathBuf::from(&project_path);
    if !root.is_dir() {
        return Err(format!("project directory not found: {project_path}"));
    }
    harness::agent::start_turn(app, session_id, root, text)
}

#[tauri::command]
fn stop_agent(state: State<'_, AppState>, session_id: String) {
    if let Some(flag) = state.cancel_flags.lock().unwrap().get(&session_id) {
        flag.store(true, Ordering::Relaxed);
    }
}

#[tauri::command]
fn respond_approval(state: State<'_, AppState>, approval_id: String, approved: bool) {
    if let Some(tx) = state.approvals.lock().unwrap().remove(&approval_id) {
        let _ = tx.send(approved);
    }
}

// ---------- mlx ----------

#[tauri::command]
async fn mlx_status(app: AppHandle) -> mlx::MlxStatus {
    mlx::status(&app).await
}

#[tauri::command]
fn mlx_setup(app: AppHandle) -> Result<(), String> {
    mlx::setup(app)
}

#[tauri::command]
fn mlx_start(app: AppHandle, model: String, port: u16) -> Result<(), String> {
    mlx::start(app, model, port)
}

#[tauri::command]
fn mlx_stop(app: AppHandle) {
    mlx::stop(&app);
}

#[tauri::command]
fn mlx_logs(app: AppHandle) -> Vec<String> {
    mlx::logs(&app)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .setup(|app| {
            let loaded = settings::load(&app.handle().clone());
            *app.state::<AppState>().settings.lock().unwrap() = loaded;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            set_settings,
            list_projects,
            add_project,
            remove_project,
            file_tree,
            read_project_file,
            list_threads,
            create_thread,
            delete_thread,
            load_thread_items,
            save_thread_items,
            thread_file_diff,
            send_message,
            stop_agent,
            respond_approval,
            mlx_status,
            mlx_setup,
            mlx_start,
            mlx_stop,
            mlx_logs
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Make sure the MLX sidecar dies with the app.
            if let tauri::RunEvent::Exit = event {
                mlx::stop(app_handle);
            }
        });
}
