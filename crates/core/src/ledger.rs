//! The capability ledger: a core-owned, append-only, hash-chained record of
//! every observed change to the capability files that enter the frozen
//! registry (external tool TOMLs and skill SKILL.mds), plus a
//! content-addressed store of the bytes themselves.
//!
//! Why the core owns this: the agent can write any file inside the project,
//! so only the process that runs turns can honestly say *when* a capability
//! file changed relative to the session lifecycle. The ledger lives outside
//! the project (like `trust.json`), where the confined file tools never
//! write. Each record embeds the sha256 of the previous record line, so
//! tampering through bash is at least detectable - the honest ceiling
//! without an OS sandbox, and the same ceiling trust already lives at.
//!
//! Rollback is deliberately a file operation, not a product: `openmax
//! --ledger` prints history with object paths, and restoring is `cp`. The
//! core guarantees the history exists; using it stays ordinary file work.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Who changed a capability file, at the strength the core can prove.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Actor {
    /// Present when this project's ledger was first populated; prior history
    /// is unknowable, so no stronger claim is made.
    Initial,
    /// Changed while an agent turn was running in the named session: the
    /// mid-turn refreeze after a successful mutating call observed it.
    Session,
    /// Changed while no turn was running (a human edit, `git pull`, a
    /// third-party install): observed at turn start or `/reload`.
    External,
}

impl Actor {
    pub fn as_str(self) -> &'static str {
        match self {
            Actor::Initial => "initial",
            Actor::Session => "session",
            Actor::External => "external",
        }
    }
}

/// One observed change. `sha256` is `None` when the file was removed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub v: u32,
    /// Unix seconds at observation time.
    pub ts: u64,
    pub path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub actor: Actor,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// sha256 of the previous record's serialized line ("" for the first).
    pub prev: String,
}

/// One entry of a sync's outcome, for the refreeze receipt.
#[derive(Clone, Debug)]
pub struct Change {
    pub path: PathBuf,
    pub actor: Actor,
    /// `added`, `modified`, or `removed`.
    pub kind: &'static str,
}

const RECORD_VERSION: u32 = 1;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Per-project ledger directory under the harness data dir, keyed by the
/// canonical root so symlinked spellings share one history.
pub fn project_dir(data_dir: &Path, project_root: &Path) -> PathBuf {
    let canonical = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());
    let key = sha256_hex(canonical.to_string_lossy().as_bytes());
    data_dir.join("ledger").join(&key[..16])
}

fn log_path(dir: &Path) -> PathBuf {
    dir.join("log.jsonl")
}

fn lock_path(dir: &Path) -> PathBuf {
    dir.join("ledger.lock")
}

/// Read the full history. A malformed line is an error, never silently
/// skipped: a ledger that cannot be read must not be read around.
pub fn history(data_dir: &Path, project_root: &Path) -> Result<Vec<Record>, String> {
    let path = log_path(&project_dir(data_dir, project_root));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(line)
            .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        out.push(record);
    }
    Ok(out)
}

/// The last known hash per path (None = removed), from the full history.
fn head(records: &[Record]) -> HashMap<PathBuf, Option<String>> {
    let mut map = HashMap::new();
    for r in records {
        map.insert(r.path.clone(), r.sha256.clone());
    }
    map
}

/// Verify the hash chain; returns the number of intact records or the index
/// where the chain breaks.
pub fn verify_chain(records_text: &str) -> Result<usize, usize> {
    let mut prev = String::new();
    for (i, line) in records_text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let Ok(record) = serde_json::from_str::<Record>(line) else {
            return Err(i);
        };
        if record.prev != prev {
            return Err(i);
        }
        prev = sha256_hex(line.as_bytes());
    }
    Ok(records_text.lines().filter(|l| !l.trim().is_empty()).count())
}

/// Populate a project's ledger the first time the harness sees it, so later
/// syncs have a baseline to attribute changes against. One existence check
/// when already seeded.
pub fn seed_if_empty(
    data_dir: &Path,
    project_root: &Path,
    files: &[(PathBuf, String, Vec<u8>)],
) -> Result<(), String> {
    if log_path(&project_dir(data_dir, project_root)).exists() {
        return Ok(());
    }
    sync(data_dir, project_root, files, Actor::Initial, None).map(|_| ())
}

/// Record the difference between the ledger head and `files` (the exact
/// generation a freeze read: path -> (sha256, bytes)). New and changed files
/// get `actor` (or `Initial` when the ledger is empty), removed paths get a
/// removal record, and changed content lands in `objects/<sha256>`. Returns
/// what changed, for the refreeze receipt. Serialized by an exclusive flock;
/// callers already hold the turn, so contention is a second harness process.
pub fn sync(
    data_dir: &Path,
    project_root: &Path,
    files: &[(PathBuf, String, Vec<u8>)],
    actor: Actor,
    session_id: Option<&str>,
) -> Result<Vec<Change>, String> {
    let dir = project_dir(data_dir, project_root);
    std::fs::create_dir_all(dir.join("objects"))
        .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path(&dir))
        .map_err(|e| format!("cannot open ledger lock: {e}"))?;
    lock.lock_exclusive().map_err(|e| format!("cannot lock ledger: {e}"))?;

    let result = sync_locked(&dir, project_root, files, actor, session_id, data_dir);
    let _ = fs2::FileExt::unlock(&lock);
    result
}

fn sync_locked(
    dir: &Path,
    project_root: &Path,
    files: &[(PathBuf, String, Vec<u8>)],
    actor: Actor,
    session_id: Option<&str>,
    data_dir: &Path,
) -> Result<Vec<Change>, String> {
    let records = history(data_dir, project_root)?;
    let effective_actor = if records.is_empty() { Actor::Initial } else { actor };
    let known = head(&records);
    let mut prev = std::fs::read_to_string(log_path(dir))
        .ok()
        .and_then(|text| {
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .next_back()
                .map(|l| sha256_hex(l.as_bytes()))
        })
        .unwrap_or_default();

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut changes = Vec::new();
    let mut lines = String::new();
    let mut seen: Vec<&PathBuf> = Vec::new();
    for (path, sha, bytes) in files {
        seen.push(path);
        let kind = match known.get(path) {
            Some(Some(existing)) if existing == sha => continue,
            Some(Some(_)) => "modified",
            Some(None) => "modified", // re-added after removal
            None => "added",
        };
        let object = dir.join("objects").join(sha);
        if !object.exists() {
            crate::sessions::write_atomic(&object, bytes)?;
        }
        let record = Record {
            v: RECORD_VERSION,
            ts,
            path: path.clone(),
            sha256: Some(sha.clone()),
            actor: effective_actor,
            session_id: session_id.map(str::to_string),
            prev: prev.clone(),
        };
        let line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
        prev = sha256_hex(line.as_bytes());
        lines.push_str(&line);
        lines.push('\n');
        changes.push(Change { path: path.clone(), actor: effective_actor, kind });
    }
    // Removals: paths the ledger knows as present that this generation lacks.
    for (path, last) in &known {
        if last.is_some() && !seen.iter().any(|p| *p == path) {
            let record = Record {
                v: RECORD_VERSION,
                ts,
                path: path.clone(),
                sha256: None,
                actor: effective_actor,
                session_id: session_id.map(str::to_string),
                prev: prev.clone(),
            };
            let line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
            prev = sha256_hex(line.as_bytes());
            lines.push_str(&line);
            lines.push('\n');
            changes.push(Change { path: path.clone(), actor: effective_actor, kind: "removed" });
        }
    }

    if !lines.is_empty() {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path(dir))
            .map_err(|e| format!("cannot append to ledger: {e}"))?;
        file.write_all(lines.as_bytes())
            .map_err(|e| format!("cannot append to ledger: {e}"))?;
    }
    Ok(changes)
}

/// A short human line per change, for the refreeze receipt. The paths are
/// project-relative where possible so the note reads like the tree.
pub fn describe(changes: &[Change], project_root: &Path) -> Vec<String> {
    changes
        .iter()
        .map(|c| {
            let path = c.path.strip_prefix(project_root).unwrap_or(&c.path);
            format!("{} {} ({})", path.display(), c.kind, c.actor.as_str())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-ledger-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(root: &Path, rel: &str, content: &str) -> (PathBuf, String, Vec<u8>) {
        (root.join(rel), sha256_hex(content.as_bytes()), content.as_bytes().to_vec())
    }

    #[test]
    fn first_sync_is_initial_then_changes_carry_the_caller_actor() {
        let data = temp("data");
        let root = temp("proj");
        let files = vec![entry(&root, ".openmax/tools/a.toml", "v1")];
        let changes = sync(&data, &root, &files, Actor::Session, Some("s1")).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].actor, Actor::Initial, "an empty ledger seeds as initial");

        // Unchanged content records nothing.
        assert!(sync(&data, &root, &files, Actor::External, None).unwrap().is_empty());

        let files = vec![entry(&root, ".openmax/tools/a.toml", "v2")];
        let changes = sync(&data, &root, &files, Actor::Session, Some("s1")).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].actor, Actor::Session);
        assert_eq!(changes[0].kind, "modified");

        // Removal is a record, not silence.
        let changes = sync(&data, &root, &[], Actor::External, None).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].kind, "removed");

        let records = history(&data, &root).unwrap();
        assert_eq!(records.len(), 3);
        assert!(records[2].sha256.is_none());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn objects_hold_the_exact_bytes_for_rollback() {
        let data = temp("obj-data");
        let root = temp("obj-proj");
        let files = vec![entry(&root, ".openmax/tools/a.toml", "name = \"a\"")];
        sync(&data, &root, &files, Actor::External, None).unwrap();
        let records = history(&data, &root).unwrap();
        let sha = records[0].sha256.clone().unwrap();
        let object = project_dir(&data, &root).join("objects").join(&sha);
        assert_eq!(std::fs::read_to_string(object).unwrap(), "name = \"a\"");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_chain_links_every_record_and_detects_tampering() {
        let data = temp("chain-data");
        let root = temp("chain-proj");
        for v in ["v1", "v2", "v3"] {
            let files = vec![entry(&root, ".openmax/tools/a.toml", v)];
            sync(&data, &root, &files, Actor::External, None).unwrap();
        }
        let log = log_path(&project_dir(&data, &root));
        let text = std::fs::read_to_string(&log).unwrap();
        assert_eq!(verify_chain(&text), Ok(3));

        // Editing an early record (rewriting its stored hash) breaks the
        // chain at the record after it.
        let first_sha = history(&data, &root).unwrap()[0].sha256.clone().unwrap();
        let tampered = text.replacen(&first_sha, &"0".repeat(first_sha.len()), 1);
        assert_ne!(tampered, text, "tampering must actually change the log");
        assert!(verify_chain(&tampered).is_err());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn malformed_ledger_is_an_error_not_an_empty_history() {
        let data = temp("bad-data");
        let root = temp("bad-proj");
        let dir = project_dir(&data, &root);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(log_path(&dir), "not json\n").unwrap();
        assert!(history(&data, &root).is_err());
        assert!(sync(&data, &root, &[], Actor::External, None).is_err());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }
}
