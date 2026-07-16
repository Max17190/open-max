use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::state::Core;
use crate::types::{AgentEvent, ChatMessage};

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

fn compaction_path(core: &Core, id: &str) -> PathBuf {
    sessions_dir(core).join(format!("{id}.compaction.jsonl"))
}

/// One exchange-drop compaction event, append-only for recoverability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompactionRecord {
    pub ts: u64,
    pub message_count: usize,
    pub tools: Vec<String>,
    pub paths: Vec<String>,
    pub user_snippets: Vec<String>,
    pub digest: String,
}

/// Wall-clock seconds for compaction records (and session meta).
pub fn unix_now() -> u64 {
    now()
}

/// Append a compaction event. Best-effort: failures surface as an agent warning.
pub fn append_compaction(core: &Core, id: &str, record: &CompactionRecord) {
    let path = compaction_path(core, id);
    let Ok(line) = serde_json::to_string(record) else { return };
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| writeln!(f, "{line}"));
    if let Err(e) = result {
        core.send_agent(
            id,
            AgentEvent::Error {
                message: format!("warning: failed to persist compaction record: {e}"),
            },
        );
    }
}

/// Load compaction history for a session (corrupt lines skipped).
pub fn load_compaction(core: &Core, id: &str) -> Vec<CompactionRecord> {
    let Ok(text) = std::fs::read_to_string(compaction_path(core, id)) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Persist the registry frozen at session creation. Written once; skipped
/// entirely for builtin-only sessions (absence means built-ins, which also
/// covers every session that predates the extensibility layer).
pub fn save_manifest(core: &Core, id: &str, manifest: &crate::registry::RegistryManifest) {
    let Ok(json) = serde_json::to_string_pretty(manifest) else {
        return;
    };
    if let Err(e) = write_atomic(&manifest_path(core, id), json) {
        core.send_agent(
            id,
            AgentEvent::Error {
                message: format!("warning: failed to persist registry manifest: {e}"),
            },
        );
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
    write_atomic(&index_path(core), json)
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

/// Write `bytes` via a unique same-directory temp file + rename so readers
/// never see a partial target. Unique names avoid two processes clobbering
/// the same `*.tmp`.
///
/// Replacement strategy:
/// 1. Try `rename(tmp → path)` (atomic replace on Unix; works when missing
///    on every platform).
/// 2. If that fails and `path` exists (Windows), move `path` aside to a unique
///    `.bak`, rename `tmp → path`, then drop the backup. If the install rename
///    fails, restore the backup so a transient error never erases the prior
///    transcript/index/manifest.
fn write_atomic(path: &PathBuf, bytes: impl AsRef<[u8]>) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let base = path
        .file_name()
        .ok_or_else(|| "path has no file name".to_string())?
        .to_string_lossy();
    let id = uuid::Uuid::new_v4().simple();
    let tmp = parent.join(format!("{base}.{id}.tmp"));
    if let Err(e) = std::fs::write(&tmp, bytes.as_ref()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    if std::fs::rename(&tmp, path).is_ok() {
        return Ok(());
    }
    if !path.exists() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("failed to install {}", path.display()));
    }
    let backup = parent.join(format!("{base}.{id}.bak"));
    if let Err(e) = std::fs::rename(path, &backup) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&backup);
            Ok(())
        }
        Err(e) => {
            // Prior content is still in `backup`; put it back before failing.
            let _ = std::fs::rename(&backup, path);
            let _ = std::fs::remove_file(&tmp);
            Err(e.to_string())
        }
    }
}

fn write_jsonl(path: &PathBuf, messages: &[ChatMessage]) -> Result<(), String> {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&serde_json::to_string(msg).map_err(|e| e.to_string())?);
        out.push('\n');
    }
    write_atomic(path, out)
}

fn append_jsonl(path: &PathBuf, messages: &[ChatMessage]) -> Result<(), String> {
    // Serialize the whole tail first, then one write under exclusive ownership
    // of the file handle. We never truncate on failure: another process could
    // have appended past our start, and set_len would erase their lines. A
    // partial write leaves a corrupt tail that load_messages skips; the caller
    // does not advance `persisted`, and the next rewrite path heals the file.
    let mut buf = String::new();
    for msg in messages {
        buf.push_str(&serde_json::to_string(msg).map_err(|e| e.to_string())?);
        buf.push('\n');
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(buf.as_bytes()).map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())?;
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
    let _ = std::fs::remove_file(compaction_path(core, id));
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

/// Load persisted messages. Corrupt JSONL lines are skipped silently so a
/// partially damaged file still yields whatever could be parsed. Returns
/// `None` when the file is missing, empty, wholly unparseable, or the legacy
/// array payload is invalid — callers treat that as "no transcript on disk".
pub fn load_messages(core: &Core, id: &str) -> Option<Vec<ChatMessage>> {
    let path = messages_path(core, id);
    let text = std::fs::read_to_string(&path).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('[') {
        serde_json::from_str(&text).ok()
    } else {
        let parsed: Vec<ChatMessage> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        if parsed.is_empty() {
            None
        } else {
            Some(parsed)
        }
    }
}

/// Persist messages. Appends only new tail lines when possible; rewrites the
/// whole file after budget trimming, legacy migration, or message drops.
///
/// Serializes disk access with `sessions_lock` so concurrent turns in the same
/// process cannot interleave appends or rewrites of the same file.
pub fn save_messages(core: &Core, id: &str, messages: &[ChatMessage], persisted: &mut usize, rewrite: bool) {
    let path = messages_path(core, id);
    let _guard = core.sessions_lock.lock().unwrap();
    let migrate = path.exists() && uses_legacy_array_format(&path);
    let needs_rewrite = rewrite || migrate || messages.len() < *persisted;

    let result = if needs_rewrite {
        write_jsonl(&path, messages)
    } else if messages.len() > *persisted {
        append_jsonl(&path, &messages[*persisted..])
    } else {
        Ok(())
    };

    match result {
        Ok(()) => *persisted = messages.len(),
        Err(e) => {
            core.send_agent(
                id,
                AgentEvent::Error {
                    message: format!("warning: failed to persist session to disk: {e}"),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Core;
    use crate::types::ChatMessage;

    #[test]
    fn compaction_records_append_and_load() {
        let dir = std::env::temp_dir().join(format!("openmax-compact-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "c1";
        let rec = CompactionRecord {
            ts: 1,
            message_count: 3,
            tools: vec!["read_file".into()],
            paths: vec!["a.rs".into()],
            user_snippets: vec!["do the thing".into()],
            digest: "[context note: test]".into(),
        };
        append_compaction(&core, id, &rec);
        append_compaction(&core, id, &CompactionRecord {
            ts: 2,
            message_count: 2,
            tools: vec![],
            paths: vec![],
            user_snippets: vec![],
            digest: "[context note: second]".into(),
        });
        let loaded = load_compaction(&core, id);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].message_count, 3);
        assert_eq!(loaded[1].ts, 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn empty_or_corrupt_messages_file_loads_as_none() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "empty";

        std::fs::write(messages_path(&core, id), "").unwrap();
        assert!(load_messages(&core, id).is_none());

        std::fs::write(messages_path(&core, id), "not valid json\n{broken\n").unwrap();
        assert!(load_messages(&core, id).is_none());

        let _ = std::fs::remove_dir_all(dir);
    }

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
    fn save_failure_does_not_advance_persisted_count() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "fail-save";
        let mut persisted = 0usize;

        let initial = vec![ChatMessage::user("hello")];
        save_messages(&core, id, &initial, &mut persisted, false);
        assert_eq!(persisted, 1);

        let path = messages_path(&core, id);
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir_all(&path).unwrap();

        let extended = vec![ChatMessage::user("hello"), ChatMessage::assistant(Some("hi".into()), None)];
        save_messages(&core, id, &extended, &mut persisted, false);
        assert_eq!(persisted, 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_manifest_means_builtins_only() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        assert!(load_manifest(&core, "pre-feature-session").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn multi_message_append_is_all_or_nothing_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "multi-append";
        let mut persisted = 0usize;

        let seed = vec![ChatMessage::system("sys")];
        save_messages(&core, id, &seed, &mut persisted, false);
        assert_eq!(persisted, 1);

        // Append several messages in one save (single write_all of the tail).
        let batch = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("one"),
            ChatMessage::assistant(Some("two".into()), None),
            ChatMessage::user("three"),
        ];
        save_messages(&core, id, &batch, &mut persisted, false);
        assert_eq!(persisted, 4);

        let path = messages_path(&core, id);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches('\n').count(), 4);
        assert!(text.ends_with('\n'));

        let loaded = load_messages(&core, id).unwrap();
        assert_eq!(loaded.len(), 4);
        assert_eq!(loaded[1].content.as_deref(), Some("one"));
        assert_eq!(loaded[2].content.as_deref(), Some("two"));
        assert_eq!(loaded[3].content.as_deref(), Some("three"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rewrite_leaves_complete_file_without_tmp() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "atomic-rewrite";
        let mut persisted = 0usize;

        let initial = vec![
            ChatMessage::user("a"),
            ChatMessage::assistant(Some("b".into()), None),
            ChatMessage::user("c"),
        ];
        save_messages(&core, id, &initial, &mut persisted, false);
        assert_eq!(persisted, 3);

        // Force full rewrite (budget trim / drop path): shorter list than persisted.
        let trimmed = vec![ChatMessage::user("kept")];
        save_messages(&core, id, &trimmed, &mut persisted, true);
        assert_eq!(persisted, 1);

        let path = messages_path(&core, id);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches('\n').count(), 1);
        assert!(text.ends_with('\n'));
        let loaded = load_messages(&core, id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content.as_deref(), Some("kept"));

        // Atomic replace must not leave a sibling .tmp behind.
        let tmp = path.with_file_name(format!(
            "{}.tmp",
            path.file_name().unwrap().to_string_lossy()
        ));
        assert!(!tmp.exists(), "temp file left behind: {}", tmp.display());

        let sessions = sessions_dir(&core);
        let leftovers: Vec<_> = std::fs::read_dir(&sessions)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(".tmp")
            })
            .collect();
        assert!(leftovers.is_empty(), "unexpected .tmp files: {leftovers:?}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn save_manifest_writes_parseable_file_atomically() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone());
        let id = "manifest-atomic";

        let manifest = crate::registry::Registry::builtin_only().to_manifest();
        save_manifest(&core, id, &manifest);

        let path = manifest_path(&core, id);
        assert!(path.exists());
        let loaded = load_manifest(&core, id).expect("manifest should parse");
        assert_eq!(loaded.version, manifest.version);
        assert!(loaded.external_tools.is_empty());

        let tmp = path.with_file_name(format!(
            "{}.tmp",
            path.file_name().unwrap().to_string_lossy()
        ));
        assert!(!tmp.exists());

        let _ = std::fs::remove_dir_all(dir);
    }
}
