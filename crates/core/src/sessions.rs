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
    /// Message indices where a later sitting resumed this session. The TUI
    /// renders a divider at each on replay, so weeks of sittings stay
    /// distinguishable instead of collapsing into one stream.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resume_points: Vec<u64>,
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
    let _guard = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(core, id) {
        return;
    }
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

fn usage_path(core: &Core, id: &str) -> PathBuf {
    sessions_dir(core).join(format!("{id}.usage.jsonl"))
}

/// What one request actually cost, as the server reported it.
///
/// The prompt cache is the largest lever a client has over cost and latency,
/// and it is invisible from this side: the only evidence is `cached` coming
/// back smaller than it should. A harness that never records it cannot tell a
/// prefix it broke from a cache the provider evicted, and cannot notice
/// either one regressing. So this is kept for the same reason the capability
/// ledger is kept - the numbers are the product's claim.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub ts: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Server-reported cached prompt tokens. `None` means the endpoint said
    /// nothing, which is not the same as zero: most OpenAI-compatible servers
    /// simply omit the field, and reporting that as a 0% hit rate would be a
    /// measurement the harness invented.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
}

/// Append one request's accounting. Best-effort and silent on failure: usage
/// is a record of work already done, so a full disk must not fail the turn
/// that succeeded.
pub fn append_usage(core: &Core, id: &str, record: &TokenUsage) {
    let _guard = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(core, id) {
        return;
    }
    let Ok(line) = serde_json::to_string(record) else { return };
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(usage_path(core, id))
        .and_then(|mut f| writeln!(f, "{line}"));
}

/// Every recorded request for a session, oldest first (corrupt lines skipped).
pub fn load_usage(core: &Core, id: &str) -> Vec<TokenUsage> {
    let Ok(text) = std::fs::read_to_string(usage_path(core, id)) else {
        return Vec::new();
    };
    text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect()
}

/// Prompt tokens served from cache over a whole session, as
/// `(cached, prompt)`, counting only requests whose endpoint reported the
/// field. Returns `None` when none did: a session against a server that never
/// reports cache state has no hit rate, and showing 0% would be a lie about
/// the server rather than a fact about the session.
pub fn cache_hit_totals(records: &[TokenUsage]) -> Option<(u64, u64)> {
    let mut cached = 0u64;
    let mut prompt = 0u64;
    let mut reported = false;
    for record in records {
        if let Some(c) = record.cached_tokens {
            reported = true;
            cached = cached.saturating_add(c);
            prompt = prompt.saturating_add(record.prompt_tokens);
        }
    }
    reported.then_some((cached, prompt))
}

/// The most recent compaction record, parsing only the final valid line:
/// carry-forward wants one record, and re-parsing an append-only history
/// that only ever grows would make every prune slower than the last.
pub fn last_compaction(core: &Core, id: &str) -> Option<CompactionRecord> {
    let text = std::fs::read_to_string(compaction_path(core, id)).ok()?;
    text.lines()
        .rev()
        .filter(|l| !l.trim().is_empty())
        .find_map(|l| serde_json::from_str(l).ok())
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

fn archive_path(core: &Core, id: &str) -> PathBuf {
    sessions_dir(core).join(format!("{id}.archive.jsonl"))
}

/// Absolute path of a session's compaction archive: the address the digest
/// note hands the agent so dropped context stays reachable (bash: grep/tail).
pub fn archive_display(core: &Core, id: &str) -> String {
    archive_path(core, id).display().to_string()
}

/// Absolute path of a session's transcript, for recall provenance.
pub fn messages_display(core: &Core, id: &str) -> String {
    messages_path(core, id).display().to_string()
}

/// Absolute path of a session's compaction record log, for recall provenance.
pub fn compaction_display(core: &Core, id: &str) -> String {
    compaction_path(core, id).display().to_string()
}

/// Append the messages a prune dropped (or truncated in place), oldest
/// first, one JSON line each. The transcript rewrite that follows the prune
/// is destructive; this file is the lossless record behind the digest note's
/// address. Best-effort like `append_compaction`: a failure warns and the
/// prune proceeds, because fitting the window beats archiving what no longer
/// fits in it - but the caller gets `false` so the note never advertises an
/// address the archive does not honor.
pub fn append_archive(core: &Core, id: &str, messages: &[ChatMessage]) -> bool {
    if messages.is_empty() {
        return true;
    }
    let _guard = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(core, id) {
        return true;
    }
    let mut lines = String::new();
    for msg in messages {
        let Ok(line) = serde_json::to_string(msg) else { continue };
        lines.push_str(&line);
        lines.push('\n');
    }
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(archive_path(core, id))
        .and_then(|mut f| f.write_all(lines.as_bytes()));
    match result {
        Ok(()) => true,
        Err(e) => {
            core.send_agent(
                id,
                AgentEvent::Error {
                    message: format!("warning: failed to archive compacted messages: {e}"),
                },
            );
            false
        }
    }
}

/// Load a session's archived (compaction-dropped) messages, corrupt lines skipped.
pub fn load_archive(core: &Core, id: &str) -> Vec<ChatMessage> {
    let Ok(text) = std::fs::read_to_string(archive_path(core, id)) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Persist the frozen registry (including its extension fingerprint).
/// Written at session creation and rewritten on every re-freeze; absence
/// means a session that predates manifests and resolves to built-ins until
/// its first turn re-freezes it from disk.
pub fn save_manifest(core: &Core, id: &str, manifest: &crate::registry::RegistryManifest) {
    let Ok(json) = serde_json::to_string_pretty(manifest) else {
        return;
    };
    // The last of the five session files to take the rule. A refreeze can be
    // in flight when the session is deleted, and cancellation is cooperative,
    // so without this the manifest outlives everything it described.
    let _guard = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(core, id) {
        return;
    }
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
        .and_then(|s| serde_json::from_str::<crate::registry::RegistryManifest>(&s).ok())
        .filter(|m| m.version == crate::registry::MANIFEST_VERSION)
}

/// Some(reason) when a session index exists on disk but cannot be read as
/// one. Callers that enumerate history (recall) fail loudly on this instead
/// of reporting an empty past: `load_index`'s silent default is right for
/// the agent loop, and exactly wrong for a tool whose answer is trusted
/// when it says nothing was found.
pub fn index_diagnostic(core: &Core) -> Option<String> {
    let path = index_path(core);
    let text = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<Vec<SessionMeta>>(&text) {
        Ok(_) => None,
        Err(e) => Some(format!("session index {} does not parse ({e})", path.display())),
    }
}

fn load_index(core: &Core) -> Vec<SessionMeta> {
    std::fs::read_to_string(index_path(core))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Whether the session still exists, i.e. whether writing a sidecar for it is
/// still meaningful.
///
/// Every sidecar here is opened with `create`, so a write that lands after
/// `delete` recreates the file it just removed. Cancellation narrows that
/// window but cannot close it: a request already on the wire settles when it
/// settles, and its usage record arrives afterwards. Checking the index makes
/// the write a no-op instead, which is what "deleted" has to mean if it is to
/// mean anything.
///
/// One small read per append. Compaction and archive appends happen at prune
/// time, and a usage append happens once per request, next to a network round
/// trip that costs several orders of magnitude more.
/// Callers must already hold `sessions_lock`: an unlocked check is a
/// time-of-check/time-of-use bug, because `delete` can remove the entry and
/// the file between the check passing and the write landing, which recreates
/// exactly the file that was deleted.
fn still_indexed_locked(core: &Core, id: &str) -> bool {
    load_index(core).iter().any(|m| m.id == id)
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

/// True when an existing transcript file must not receive JSONL appends.
/// Historical array-shaped blobs (first non-ws byte `[`) are not loaded as
/// history, but the file may still sit on disk; append would create a mixed
/// file. Force a full JSONL rewrite instead. Not a load dual-path.
fn must_rewrite_non_jsonl(path: &PathBuf) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = [0u8; 64];
    let Ok(n) = file.read(&mut head) else {
        return false;
    };
    head[..n]
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|b| *b == b'[')
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
///    data file.
pub(crate) fn write_atomic(path: &PathBuf, bytes: impl AsRef<[u8]>) -> Result<(), String> {
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
    // Never treat a directory as a replaceable destination (would move the dir
    // aside as `.bak` and leave an orphaned tree).
    if path.is_dir() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{} is a directory", path.display()));
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
            // Prior content is still in `backup`; put it back at the canonical
            // path before failing so loaders keep working. Prefer rename; if
            // that fails (e.g. path recreated/locked), fall back to copy.
            let _ = std::fs::remove_file(&tmp);
            if std::fs::rename(&backup, path).is_ok() {
                return Err(e.to_string());
            }
            match std::fs::copy(&backup, path) {
                Ok(_) => {
                    let _ = std::fs::remove_file(&backup);
                    Err(e.to_string())
                }
                Err(ce) => Err(format!(
                    "install failed ({e}); restore rename/copy failed ({ce}); prior data at {}",
                    backup.display()
                )),
            }
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
    // Serialize the whole tail first, then one write. Callers must heal on
    // failure (rewrite the full file) so a partial write cannot be re-appended
    // and duplicate complete lines when `persisted` is left unchanged.
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
        resume_points: Vec::new(),
    };
    let m = meta.clone();
    with_index(core, move |metas| metas.push(m))?;
    Ok(meta)
}

/// One session's index entry, if it exists.
pub fn meta(core: &Core, id: &str) -> Option<SessionMeta> {
    load_index(core).into_iter().find(|m| m.id == id)
}

/// Keep resume boundaries pointing at the same messages across a transcript
/// prune that removed a net `removed` messages above the pinned prefix
/// (system plus first user). Points inside the removed region collapse onto
/// its floor; duplicates that result are dropped.
pub fn shift_resume_points_for_prune(core: &Core, id: &str, removed: u64) {
    if removed == 0 {
        return;
    }
    let _ = with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            let mut shifted: Vec<u64> = m
                .resume_points
                .iter()
                .map(|&p| if p <= 2 { p } else { p.saturating_sub(removed).max(2) })
                .collect();
            shifted.sort_unstable();
            shifted.dedup();
            m.resume_points = shifted;
        }
    });
}

/// A system message was inserted at the front of the transcript (legacy
/// sessions saved before the prompt lived at index zero); every absolute
/// boundary moves down by one.
pub fn shift_resume_points_for_system_insert(core: &Core, id: &str) {
    let _ = with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            for p in &mut m.resume_points {
                *p = p.saturating_add(1);
            }
        }
    });
}

/// Record that a new sitting resumed this session with `message_index`
/// messages already on disk. Index zero is an empty session, not a
/// boundary; repeats (resuming again before any new turn) are deduplicated.
pub fn record_resume_point(core: &Core, id: &str, message_index: u64) {
    if message_index == 0 {
        return;
    }
    let _ = with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            if !m.resume_points.contains(&message_index) {
                m.resume_points.push(message_index);
            }
        }
    });
}

pub fn delete(core: &Core, id: &str) -> Result<(), String> {
    // Stop the session's work before removing its files. A frontend may
    // delete the session it is currently running, and a turn that keeps going
    // keeps writing: every sidecar here is opened with `create`, so an
    // in-flight append recreates the file that was just deleted. Cancelling
    // first is also the behaviour a user asking to delete a session expects.
    //
    // A request already on the wire can still land after this returns and
    // append one record. That window is inherent to cooperative cancellation
    // and predates the usage sidecar - the compaction and archive logs have
    // always shared it - so it is narrowed here, not claimed closed.
    core.cancel(id);
    // The index entry and the files go under one lock. Dropping the entry
    // first and the files second would let an append pass its check against
    // the stale index and recreate what this call is removing.
    let _guard = core.sessions_lock.lock().unwrap();
    let mut metas = load_index(core);
    metas.retain(|m| m.id != id);
    save_index(core, &metas)?;
    let _ = std::fs::remove_file(messages_path(core, id));
    let _ = std::fs::remove_file(manifest_path(core, id));
    let _ = std::fs::remove_file(compaction_path(core, id));
    let _ = std::fs::remove_file(archive_path(core, id));
    let _ = std::fs::remove_file(usage_path(core, id));
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

/// Test-only backdating: recency ranking needs sessions with known ages, and
/// production code must never set `updated_at` to anything but now.
#[cfg(test)]
pub(crate) fn touch_at(core: &Core, id: &str, ts: u64) {
    let _ = with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            m.updated_at = ts;
        }
    });
}

/// Load persisted messages as JSONL only. Corrupt lines are skipped silently
/// so a partially damaged file still yields whatever could be parsed. Returns
/// `None` when the file is missing, empty, or wholly unparseable — callers
/// treat that as "no transcript on disk".
pub fn load_messages(core: &Core, id: &str) -> Option<Vec<ChatMessage>> {
    let path = messages_path(core, id);
    let text = std::fs::read_to_string(&path).ok()?;
    if text.trim().is_empty() {
        return None;
    }
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

/// Persist messages. Appends only new tail lines when possible; rewrites the
/// whole file after budget trimming or message drops.
///
/// Serializes disk access with `sessions_lock` so concurrent turns in the same
/// process cannot interleave appends or rewrites of the same file.
pub fn save_messages(core: &Core, id: &str, messages: &[ChatMessage], persisted: &mut usize, rewrite: bool) {
    let path = messages_path(core, id);
    let _guard = core.sessions_lock.lock().unwrap();
    // Same rule as the sidecars, and for the same reason: cancellation is
    // cooperative, so a turn keeps running for a while after `delete` and
    // ends with an unconditional save. Without this the transcript of a
    // deleted session comes back, which is the one file that made the
    // deletion visible in the first place.
    if !still_indexed_locked(core, id) {
        return;
    }
    // Never append onto a non-JSONL blob left on disk after a failed load.
    let needs_rewrite =
        rewrite || messages.len() < *persisted || must_rewrite_non_jsonl(&path);

    let result = if needs_rewrite {
        write_jsonl(&path, messages)
    } else if messages.len() > *persisted {
        // Append is best-effort for the common path. On any failure (including
        // partial write_all), rewrite the full transcript atomically so a
        // later append cannot duplicate complete lines that already landed.
        match append_jsonl(&path, &messages[*persisted..]) {
            Ok(()) => Ok(()),
            Err(append_err) => write_jsonl(&path, messages).map_err(|rewrite_err| {
                format!("append failed ({append_err}); rewrite also failed: {rewrite_err}")
            }),
        }
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

    #[test]
    fn prune_shifts_resume_points_and_collapses_onto_the_floor() {
        let dir = std::env::temp_dir().join(format!("openmax-resume-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        record_resume_point(&core, id, 2);
        record_resume_point(&core, id, 4);
        record_resume_point(&core, id, 10);

        // A prune removed a net 3 messages above the pinned prefix: the
        // deep boundary shifts, the shallow one collapses onto the floor,
        // and the pinned-prefix boundary is untouched.
        shift_resume_points_for_prune(&core, id, 3);
        assert_eq!(meta(&core, id).unwrap().resume_points, vec![2, 7]);

        // A legacy system-prompt insert moves every boundary down one.
        shift_resume_points_for_system_insert(&core, id);
        assert_eq!(meta(&core, id).unwrap().resume_points, vec![3, 8]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn old_index_entries_without_resume_points_still_parse() {
        let m: SessionMeta = serde_json::from_str(
            r#"{"id":"x","project":"/p","title":"t","created_at":1,"updated_at":2}"#,
        )
        .unwrap();
        assert!(m.resume_points.is_empty());
    }

    use crate::state::Core;
    use crate::types::ChatMessage;

    #[test]
    fn usage_records_append_and_aggregate_only_what_was_reported() {
        let dir = std::env::temp_dir().join(format!("openmax-usage-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        // A server that reports cache state, then one that does not.
        append_usage(&core, id, &TokenUsage {
            ts: 1,
            prompt_tokens: 1000,
            completion_tokens: 50,
            cached_tokens: Some(900),
        });
        append_usage(&core, id, &TokenUsage {
            ts: 2,
            prompt_tokens: 1200,
            completion_tokens: 60,
            cached_tokens: Some(1100),
        });
        append_usage(&core, id, &TokenUsage {
            ts: 3,
            prompt_tokens: 5000,
            completion_tokens: 10,
            cached_tokens: None,
        });
        let records = load_usage(&core, id);
        assert_eq!(records.len(), 3, "append-only, oldest first");
        assert_eq!(records[0].prompt_tokens, 1000);
        // The unreported request contributes to neither side: counting its
        // 5000 prompt tokens as a miss would report the server's silence as
        // this session's cache behaviour.
        assert_eq!(cache_hit_totals(&records), Some((2000, 2200)));
        assert_eq!(
            cache_hit_totals(&[TokenUsage { ts: 1, prompt_tokens: 9, ..Default::default() }]),
            None,
            "a session nobody reported on has no hit rate"
        );
        assert_eq!(cache_hit_totals(&[]), None);

        // Deleting a session must take its sidecars with it: a recreated id
        // would otherwise inherit a stranger's accounting.
        append_archive(&core, id, &[ChatMessage::user("dropped")]);
        append_compaction(&core, id, &CompactionRecord {
            ts: 1,
            message_count: 1,
            tools: vec![],
            paths: vec![],
            user_snippets: vec![],
            digest: "[context note: x]".into(),
        });
        create(&core, "/tmp/p".into()).ok();
        with_index(&core, |m| {
            m.push(SessionMeta {
                id: id.into(),
                project: "/tmp/p".into(),
                title: "t".into(),
                created_at: 0,
                updated_at: 0,
                resume_points: Vec::new(),
            })
        })
        .unwrap();
        delete(&core, id).unwrap();
        assert!(load_usage(&core, id).is_empty(), "usage sidecar must not outlive the session");
        assert!(load_compaction(&core, id).is_empty());
        assert!(load_archive(&core, id).is_empty());

        // Deleting a session cancels its in-flight turn: a session that keeps
        // running keeps writing, and every sidecar above is opened with
        // `create`, so the files would come back.
        let running = create(&core, "/tmp/p".into()).unwrap();
        let token = std::sync::Arc::new(crate::state::CancelToken::default());
        core.cancel_flags.lock().unwrap().insert(running.id.clone(), token.clone());
        assert!(!token.is_cancelled());
        delete(&core, &running.id).unwrap();
        assert!(token.is_cancelled(), "delete must stop the work before removing the files");

        // And the write that loses the race is a no-op rather than a
        // resurrection: cancellation cannot stop a request already on the
        // wire, so the append itself has to know the session is gone.
        append_usage(&core, &running.id, &TokenUsage {
            ts: 9,
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: Some(1),
        });
        append_compaction(&core, &running.id, &CompactionRecord {
            ts: 9,
            message_count: 1,
            tools: vec![],
            paths: vec![],
            user_snippets: vec![],
            digest: "late".into(),
        });
        append_archive(&core, &running.id, &[ChatMessage::user("late")]);
        assert!(load_usage(&core, &running.id).is_empty(), "a deleted session stays deleted");
        assert!(load_compaction(&core, &running.id).is_empty());
        assert!(load_archive(&core, &running.id).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The fix is that an append and a delete cannot interleave, so this
    /// tests exactly that: while `sessions_lock` is held, an append must
    /// block rather than write. Racing threads against a delete and hoping
    /// for the bad interleaving proves nothing - that version of this test
    /// passed against the unserialized code too.
    #[test]
    fn an_append_is_serialized_against_the_session_lock() {
        let dir = std::env::temp_dir().join(format!("openmax-race-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;

        let guard = core.sessions_lock.lock().unwrap();
        let writer = {
            let core = core.clone();
            let id = id.clone();
            std::thread::spawn(move || {
                append_usage(&core, &id, &TokenUsage {
                    ts: 1,
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    cached_tokens: Some(1),
                });
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            load_usage(&core, &id).is_empty(),
            "an append must wait for the lock delete holds, or it can write after the delete"
        );
        drop(guard);
        writer.join().unwrap();
        assert_eq!(load_usage(&core, &id).len(), 1, "and land once the lock is free");

        // The transcript is under the same rule. It is the file that made the
        // deletion visible, so a late save recreating it is the worst version
        // of this bug, not the mildest.
        delete(&core, &id).unwrap();
        let mut persisted = 0usize;
        save_messages(&core, &id, &[ChatMessage::user("late")], &mut persisted, true);
        assert!(load_messages(&core, &id).is_none(), "a deleted transcript stays deleted");
        save_manifest(&core, &id, &crate::registry::RegistryManifest {
            version: 1,
            external_tools: Vec::new(),
            skills: Vec::new(),
            ext_fingerprint: 0,
        });
        assert!(load_manifest(&core, &id).is_none(), "and so does its manifest");
        // All five session-scoped files, so this cannot regress one at a time.
        for suffix in ["messages.json", "manifest.json", "compaction.jsonl", "archive.jsonl", "usage.jsonl"] {
            let path = sessions_dir(&core).join(format!("{id}.{suffix}"));
            assert!(!path.exists(), "{suffix} came back after delete");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn compaction_records_append_and_load() {
        let dir = std::env::temp_dir().join(format!("openmax-compact-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
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

    /// Carry-forward reads one record, not the history: the final valid line
    /// wins, and trailing garbage (a torn write) falls through to the last
    /// parseable record instead of erasing the carry.
    #[test]
    fn last_compaction_parses_only_the_final_valid_line() {
        let dir = std::env::temp_dir().join(format!("openmax-lastcomp-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        assert!(last_compaction(&core, id).is_none());
        for ts in [1u64, 2] {
            append_compaction(&core, id, &CompactionRecord {
                ts,
                message_count: ts as usize,
                tools: vec![],
                paths: vec![format!("src/{ts}.rs")],
                user_snippets: vec![],
                digest: format!("[context note: {ts}]"),
            });
        }
        let path = sessions_dir(&core).join(format!("{id}.compaction.jsonl"));
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, "{{torn").unwrap();
        let last = last_compaction(&core, id).expect("a valid record exists");
        assert_eq!(last.ts, 2, "the final valid line wins over trailing garbage");
        assert_eq!(last.paths, vec!["src/2.rs".to_string()]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The archive is the lossless record behind the digest note's address:
    /// consecutive prunes append, order survives, and tool-call structure
    /// round-trips so an archived exchange can be read back whole.
    #[test]
    fn compaction_archive_appends_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("openmax-archive-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        assert!(load_archive(&core, id).is_empty(), "no archive before any prune");
        append_archive(&core, id, &[]);
        assert!(
            !std::path::Path::new(&archive_display(&core, id)).exists(),
            "an empty prune must not create the file"
        );

        let call = crate::types::ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: crate::types::ToolCallFunction {
                name: "read_file".into(),
                arguments: r#"{"path":"src/a.rs"}"#.into(),
            },
        };
        let first = vec![
            ChatMessage::user("find the bug"),
            ChatMessage::assistant(None, Some(vec![call])),
            ChatMessage::tool("c1", "fn main() {}"),
        ];
        append_archive(&core, id, &first);
        append_archive(&core, id, &[ChatMessage::user("second prune")]);

        let loaded = load_archive(&core, id);
        assert_eq!(loaded.len(), 4, "appends must accumulate in order");
        assert_eq!(loaded[0].content.as_deref(), Some("find the bug"));
        let calls = loaded[1].tool_calls.as_ref().expect("tool calls survive");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(loaded[2].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(loaded[3].content.as_deref(), Some("second prune"));
        assert!(archive_display(&core, id).ends_with(&format!("{id}.archive.jsonl")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn empty_or_corrupt_messages_file_loads_as_none() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;

        std::fs::write(messages_path(&core, id), "").unwrap();
        assert!(load_messages(&core, id).is_none());

        std::fs::write(messages_path(&core, id), "not valid json\n{broken\n").unwrap();
        assert!(load_messages(&core, id).is_none());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn jsonl_append_only_writes_new_tail() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
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
    fn array_payload_is_not_loaded() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        let path = messages_path(&core, id);
        std::fs::write(&path, r#"[{"role":"user","content":"old"}]"#).unwrap();
        assert!(load_messages(&core, id).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn save_over_array_blob_rewrites_jsonl_not_append() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        let path = messages_path(&core, id);
        std::fs::write(&path, r#"[{"role":"user","content":"old"}]"#).unwrap();
        assert!(load_messages(&core, id).is_none());

        // Fresh session after empty load: persisted_count starts at 0.
        let mut persisted = 0usize;
        let messages = vec![ChatMessage::system("sys"), ChatMessage::user("hello")];
        save_messages(&core, id, &messages, &mut persisted, false);
        assert_eq!(persisted, 2);

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.trim_start().starts_with('['), "must not leave array prefix:\n{text}");
        assert_eq!(text.matches('\n').count(), 2);
        let loaded = load_messages(&core, id).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1].content.as_deref(), Some("hello"));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The manifest must reconstruct the exact frozen registry with no config
    /// on disk at all: the fixture tool files are deleted before reload.
    #[test]
    fn manifest_round_trips_without_rediscovery() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;

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
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
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
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        assert!(load_manifest(&core, "pre-feature-session").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn multi_message_append_is_all_or_nothing_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
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
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
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

    /// A manifest from a newer format version is treated as absent: the
    /// session falls back to built-ins and re-freezes cleanly, instead of
    /// deserializing an unknown shape into this one.
    #[test]
    fn unknown_manifest_version_reads_as_no_manifest() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        let mut manifest = crate::registry::Registry::builtin_only().to_manifest();
        save_manifest(&core, id, &manifest);
        assert!(load_manifest(&core, id).is_some());

        manifest.version = crate::registry::MANIFEST_VERSION + 1;
        save_manifest(&core, id, &manifest);
        assert!(load_manifest(&core, id).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn save_manifest_writes_parseable_file_atomically() {
        let dir = std::env::temp_dir().join(format!("openmax-sess-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;

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
