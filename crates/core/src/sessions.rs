use std::io::Write;
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

fn manifest_path(core: &Core, id: &str) -> PathBuf {
    sessions_dir(core).join(format!("{id}.manifest.json"))
}

/// Persist the registry frozen at session creation. Written once; skipped
/// entirely for builtin-only sessions (absence means built-ins, which also
/// covers every session that predates the extensibility layer).
pub fn save_manifest(core: &Core, id: &str, manifest: &crate::registry::RegistryManifest) {
    if let Ok(json) = serde_json::to_string_pretty(manifest) {
        let _ = std::fs::write(manifest_path(core, id), json);
    }
}

pub fn load_manifest(core: &Core, id: &str) -> Option<crate::registry::RegistryManifest> {
    std::fs::read_to_string(manifest_path(core, id))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
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

/// Reads only a small prefix: this runs on every save and must not scale
/// with transcript size.
fn uses_legacy_array_format(path: &PathBuf) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else { return false };
    let mut head = [0u8; 64];
    let Ok(n) = file.read(&mut head) else { return false };
    head[..n].iter().find(|b| !b.is_ascii_whitespace()).is_some_and(|b| *b == b'[')
}

fn write_jsonl(path: &PathBuf, messages: &[ChatMessage]) -> Result<(), String> {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&serde_json::to_string(msg).map_err(|e| e.to_string())?);
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| e.to_string())
}

fn append_jsonl(path: &PathBuf, messages: &[ChatMessage]) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    for msg in messages {
        let line = serde_json::to_string(msg).map_err(|e| e.to_string())?;
        writeln!(file, "{line}").map_err(|e| e.to_string())?;
    }
    Ok(())
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
    let _ = std::fs::remove_file(manifest_path(core, id));
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
    let path = messages_path(core, id);
    let text = std::fs::read_to_string(&path).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some(Vec::new());
    }
    if trimmed.starts_with('[') {
        serde_json::from_str(&text).ok()
    } else {
        Some(
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect(),
        )
    }
}

/// Persist messages. Appends only new tail lines when possible; rewrites the
/// whole file after budget trimming, legacy migration, or message drops.
pub fn save_messages(core: &Core, id: &str, messages: &[ChatMessage], persisted: &mut usize, rewrite: bool) {
    let path = messages_path(core, id);
    let migrate = path.exists() && uses_legacy_array_format(&path);
    let needs_rewrite = rewrite || migrate || messages.len() < *persisted;

    let result = if needs_rewrite {
        write_jsonl(&path, messages)
    } else if messages.len() > *persisted {
        append_jsonl(&path, &messages[*persisted..])
    } else {
        Ok(())
    };

    if result.is_ok() {
        *persisted = messages.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Core;
    use crate::types::ChatMessage;

    #[test]
    fn jsonl_append_only_writes_new_tail() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "test-session";
        let mut persisted = 0usize;

        let initial = vec![ChatMessage::system("sys"), ChatMessage::user("hello")];
        save_messages(&core, id, &initial, &mut persisted, false);
        assert_eq!(persisted, 2);

        let path = messages_path(&core, id);
        let first = std::fs::read_to_string(&path).unwrap();
        assert_eq!(first.matches('\n').count(), 2);

        let mut extended = initial.clone();
        extended.push(ChatMessage::assistant(Some("hi".into()), None));
        save_messages(&core, id, &extended, &mut persisted, false);
        assert_eq!(persisted, 3);

        let second = std::fs::read_to_string(&path).unwrap();
        assert_eq!(second.matches('\n').count(), 3);
        assert!(second.ends_with('\n'));

        let loaded = load_messages(&core, id).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[2].content.as_deref(), Some("hi"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn legacy_array_loads_and_rewrites_on_save() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "legacy";
        let path = messages_path(&core, id);
        let legacy = r#"[{"role":"user","content":"old"}]"#;
        std::fs::write(&path, legacy).unwrap();

        let loaded = load_messages(&core, id).unwrap();
        assert_eq!(loaded.len(), 1);

        let mut persisted = loaded.len();
        save_messages(&core, id, &loaded, &mut persisted, false);
        assert!(!uses_legacy_array_format(&path));
        assert_eq!(load_messages(&core, id).unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The manifest must reconstruct the exact frozen registry with no config
    /// on disk at all: the fixture tool files are deleted before reload.
    #[test]
    fn manifest_round_trips_without_rediscovery() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "with-tools";

        let project = dir.join("project");
        let tools_dir = project.join(".openmax/tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        std::fs::write(
            tools_dir.join("deploy.toml"),
            "name = \"deploy\"\ndescription = \"ships it\"\ncommand = \"/bin/true\"\nmutating = true\n",
        )
        .unwrap();

        let original = crate::registry::Registry::build(&project);
        assert!(original.has_extensions());
        save_manifest(&core, id, &original.to_manifest());

        // Config disappears; the frozen session must not notice.
        std::fs::remove_dir_all(&tools_dir).unwrap();
        let reloaded = crate::registry::Registry::from_manifest(load_manifest(&core, id).unwrap());
        assert_eq!(reloaded.tool_names(), original.tool_names());
        assert!(reloaded.is_mutating("deploy"));
        assert_eq!(
            reloaded.tool_schemas_json().to_string(),
            original.tool_schemas_json().to_string(),
            "schemas must be byte-identical across resume"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_manifest_means_builtins_only() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        assert!(load_manifest(&core, "pre-feature-session").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
