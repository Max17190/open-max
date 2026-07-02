use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tauri::AppHandle;

use crate::settings::app_data_dir;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub path: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TreeEntry {
    pub name: String,
    /// Path relative to the project root, used for lazy child loading.
    pub rel_path: String,
    pub is_dir: bool,
}

fn projects_path(app: &AppHandle) -> PathBuf {
    app_data_dir(app).join("projects.json")
}

pub fn load(app: &AppHandle) -> Vec<Project> {
    std::fs::read_to_string(projects_path(app))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(app: &AppHandle, projects: &[Project]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(projects).map_err(|e| e.to_string())?;
    std::fs::write(projects_path(app), json).map_err(|e| e.to_string())
}

pub fn add(app: &AppHandle, path: String) -> Result<Vec<Project>, String> {
    let p = PathBuf::from(&path);
    if !p.is_dir() {
        return Err(format!("not a directory: {path}"));
    }
    let mut projects = load(app);
    if let Some(existing) = projects.iter().find(|pr| pr.path == path) {
        let _ = existing;
        return Ok(projects);
    }
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.clone());
    projects.push(Project { id: uuid::Uuid::new_v4().to_string(), name, path });
    save(app, &projects)?;
    Ok(projects)
}

pub fn remove(app: &AppHandle, id: &str) -> Result<Vec<Project>, String> {
    let mut projects = load(app);
    projects.retain(|p| p.id != id);
    save(app, &projects)?;
    Ok(projects)
}

/// List one directory level, .gitignore-aware, directories first.
pub fn list_tree(root: &Path, rel: &str) -> Result<Vec<TreeEntry>, String> {
    let dir = if rel.is_empty() || rel == "." { root.to_path_buf() } else { root.join(rel) };
    let canon = dir.canonicalize().map_err(|e| format!("cannot open directory: {e}"))?;
    let root_canon = root.canonicalize().map_err(|e| e.to_string())?;
    if !canon.starts_with(&root_canon) {
        return Err("path escapes project root".into());
    }

    let walker = ignore::WalkBuilder::new(&canon)
        .max_depth(Some(1))
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .build();

    let mut entries = Vec::new();
    for entry in walker.flatten() {
        let path = entry.path();
        if path == canon {
            continue;
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if name == ".git" {
            continue;
        }
        let rel_path = path
            .strip_prefix(&root_canon)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| name.clone());
        entries.push(TreeEntry { name, rel_path, is_dir: path.is_dir() });
    }
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    Ok(entries)
}

/// Read a file for the viewer pane (capped at 1MB).
pub fn read_text(root: &Path, rel: &str) -> Result<String, String> {
    let path = root.join(rel);
    let canon = path.canonicalize().map_err(|e| format!("cannot open file: {e}"))?;
    let root_canon = root.canonicalize().map_err(|e| e.to_string())?;
    if !canon.starts_with(&root_canon) {
        return Err("path escapes project root".into());
    }
    let meta = std::fs::metadata(&canon).map_err(|e| e.to_string())?;
    if meta.len() > 1_000_000 {
        return Err("file too large to preview".into());
    }
    std::fs::read_to_string(&canon).map_err(|_| "not a UTF-8 text file".into())
}
