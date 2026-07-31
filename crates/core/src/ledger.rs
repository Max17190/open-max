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

/// The hash of the log's final record, stored beside it. The chain alone
/// proves internal order but not completeness: removing whole trailing
/// records leaves a valid prefix. This pin makes truncation detectable.
fn chain_head_path(dir: &Path) -> PathBuf {
    dir.join("chain-head")
}

/// Read the full history, verifying the hash chain as it goes. A malformed
/// line or a broken chain is an error, never silently skipped or displayed
/// as authentic: a ledger that cannot be trusted must not be read around.
pub fn history(data_dir: &Path, project_root: &Path) -> Result<Vec<Record>, String> {
    let path = log_path(&project_dir(data_dir, project_root));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let mut out = Vec::new();
    let mut prev = String::new();
    let last_line_number = text.lines().count();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(line).map_err(|e| {
            let repair_hint = if i + 1 == last_line_number && !text.ends_with('\n') {
                " (likely an interrupted write; remove the partial last line to repair)"
            } else {
                ""
            };
            format!("{} line {}: {e}{repair_hint}", path.display(), i + 1)
        })?;
        if record.prev != prev {
            return Err(format!(
                "{} line {}: hash chain broken - the ledger was modified outside the harness",
                path.display(),
                i + 1
            ));
        }
        prev = sha256_hex(line.as_bytes());
        out.push(record);
    }
    let dir = project_dir(data_dir, project_root);
    match std::fs::read_to_string(chain_head_path(&dir)) {
        Ok(stored) if stored.trim() != prev => {
            return Err(format!(
                "{}: the log's final record does not match the stored chain head - trailing records were removed or rewritten outside the harness",
                path.display()
            ));
        }
        Ok(_) => {}
        // A pre-chain-head ledger (or a fresh project) has no pin yet; the
        // next sync writes one.
        Err(_) => {}
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
        // Never trust a pre-existing object blindly: rollback follows these
        // bytes, so an object that does not hash to its own name is replaced
        // with the authentic content this generation actually read.
        let object = dir.join("objects").join(sha);
        let object_valid = std::fs::read(&object)
            .map(|existing| sha256_hex(&existing) == *sha)
            .unwrap_or(false);
        if !object_valid {
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
        // Pin the new final record so silent truncation is detectable.
        crate::sessions::write_atomic(&chain_head_path(dir), &prev)?;
    }
    Ok(changes)
}

// ---------- content-bound approvals ----------

#[derive(Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ApprovedFile {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    hashes: Vec<String>,
    /// Capability files a human has approved at least once, canonicalized.
    /// What is enforced is still the hash set; this only remembers that a
    /// path was live, so a later edit of an installed gate reads as a gate
    /// that was modified rather than a file that was never installed.
    #[serde(default)]
    paths: Vec<String>,
}

fn approved_path(dir: &Path) -> PathBuf {
    dir.join("approved.json")
}

/// What a human has approved for this project: exact content hashes, plus the
/// capability paths those approvals were granted at.
#[derive(Debug, Default)]
pub struct Approvals {
    hashes: std::collections::HashSet<String>,
    paths: std::collections::HashSet<PathBuf>,
}

impl Approvals {
    pub fn contains(&self, sha: &str) -> bool {
        self.hashes.contains(sha)
    }

    /// Whether this exact path ever held approved content. A file here was
    /// live once, so its current unapproved content is a modification of
    /// something a human installed, not an arrival nobody ever blessed.
    pub fn was_live(&self, path: &Path) -> bool {
        self.paths.contains(&canonical_or(path))
    }

    /// Every bound code file is readable and approved. An unreadable one is
    /// never covered: a command named but not yet written has no bytes to
    /// approve, and must not read as "nothing to bind".
    pub fn covers_code(&self, code: &[BoundCode]) -> bool {
        code.iter()
            .all(|c| c.sha256.as_deref().is_some_and(|sha| self.contains(sha)))
    }
}

/// Everything a human approved for this project. Approval binds to content,
/// not path: any edit changes the hash and revokes itself. Missing file means
/// nothing approved; a malformed file is an error, and callers treat that as
/// nothing approved (fail closed) while surfacing the reason.
pub fn approvals(data_dir: &Path, project_root: &Path) -> Result<Approvals, String> {
    let path = approved_path(&project_dir(data_dir, project_root));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Approvals::default()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let file: ApprovedFile = serde_json::from_str(&text)
        .map_err(|e| format!("{} is malformed ({e}); nothing is approved until it is fixed", path.display()))?;
    Ok(Approvals {
        hashes: file.hashes.into_iter().collect(),
        paths: file.paths.into_iter().map(PathBuf::from).collect(),
    })
}

/// The approved sha256 set alone, for callers with no path question to ask.
pub fn approved_hashes(
    data_dir: &Path,
    project_root: &Path,
) -> Result<std::collections::HashSet<String>, String> {
    approvals(data_dir, project_root).map(|a| a.hashes)
}

/// Record a human approval of exact content. Serialized under the ledger lock.
pub fn approve_hash(data_dir: &Path, project_root: &Path, sha: &str) -> Result<(), String> {
    approve(data_dir, project_root, &[sha.to_string()], None)
}

/// Record a human approval of a capability file: the hashes they blessed, and
/// the path they blessed them at.
pub fn approve_capability(
    data_dir: &Path,
    project_root: &Path,
    path: &Path,
    shas: &[String],
) -> Result<(), String> {
    approve(data_dir, project_root, shas, Some(path))
}

fn approve(
    data_dir: &Path,
    project_root: &Path,
    shas: &[String],
    path: Option<&Path>,
) -> Result<(), String> {
    let dir = project_dir(data_dir, project_root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path(&dir))
        .map_err(|e| format!("cannot open ledger lock: {e}"))?;
    lock.lock_exclusive().map_err(|e| format!("cannot lock ledger: {e}"))?;
    let result = (|| {
        let current = approvals(data_dir, project_root)?;
        let mut hashes = current.hashes;
        let mut paths = current.paths;
        let mut changed = false;
        for sha in shas {
            changed |= hashes.insert(sha.clone());
        }
        if let Some(path) = path {
            changed |= paths.insert(canonical_or(path));
        }
        if changed {
            let file = ApprovedFile {
                version: 1,
                hashes: hashes.into_iter().collect(),
                paths: paths.iter().map(|p| p.display().to_string()).collect(),
            };
            let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
            crate::sessions::write_atomic(&approved_path(&dir), json)?;
        }
        Ok(())
    })();
    let _ = fs2::FileExt::unlock(&lock);
    result
}

/// Whether `sha` is human-approved. Any failure reads as unapproved: the
/// enforcement this feeds must fail toward asking, never toward running.
pub fn is_approved(data_dir: &Path, project_root: &Path, sha: &str) -> bool {
    approved_hashes(data_dir, project_root)
        .map(|set| set.contains(sha))
        .unwrap_or(false)
}

// ---------- what a manifest actually executes ----------

/// One file a capability manifest hands to the host: its resolved `command`,
/// or an `args` entry naming a file. Only project-local files are listed -
/// those are the ones the agent can rewrite after a human approved the
/// manifest that names them.
#[derive(Clone, Debug)]
pub struct BoundCode {
    pub path: PathBuf,
    /// sha256 of the bytes on disk, or None when the file cannot be read.
    pub sha256: Option<String>,
}

/// Every project-local file `command args...` will execute or interpret. This
/// is what an approval has to cover: approving only the manifest binds a
/// pointer, and the pointee sits at an ordinary project path the agent writes
/// freely.
///
/// A command resolving outside the project root (a system binary, a PATH
/// entry) contributes nothing: the human read that absolute path in the
/// manifest they approved, and hashing host binaries would revoke on every OS
/// upgrade. The threat here is the agent rewriting files it can write.
pub fn bound_code(command: &str, args: &[String], project_root: &Path) -> Vec<BoundCode> {
    let mut out = Vec::new();
    if let Some(path) = resolve_command(command.trim(), project_root) {
        if inside_project(&path, project_root) {
            out.push(read_code(path));
        }
    }
    for arg in args {
        // Fixed argv can name the script an interpreter runs
        // (`command = "python3", args = ["scripts/tool.py"]`). Flags and
        // inline program text are not paths and resolve to no file.
        let arg = arg.trim();
        if arg.is_empty() || arg.starts_with('-') {
            continue;
        }
        let path = absolute_from(arg, project_root);
        if path.is_file() && inside_project(&path, project_root) {
            out.push(read_code(path));
        }
    }
    out
}

/// The project-local code the capability manifest at `path` will execute,
/// whichever surface it belongs to. Both manifest surfaces name a `command`
/// plus fixed `args`; skills name none.
pub fn manifest_code(path: &Path, project_root: &Path) -> Vec<BoundCode> {
    if let Ok(hook) = crate::hooks::parse_hook_file(path) {
        return bound_code(&hook.command, &hook.args, project_root);
    }
    if let Ok(spec) = crate::registry::parse_tool_file(path) {
        if let crate::registry::ToolKind::External(ext) = &spec.kind {
            return bound_code(&ext.command, &ext.args, project_root);
        }
    }
    Vec::new()
}

fn read_code(path: PathBuf) -> BoundCode {
    let sha256 = std::fs::read(&path).ok().map(|bytes| sha256_hex(&bytes));
    BoundCode { path, sha256 }
}

/// Where `command` will spawn from, resolved the way the spawn resolves it: a
/// path against the project root (processes run there), a bare name on PATH.
fn resolve_command(command: &str, project_root: &Path) -> Option<PathBuf> {
    if command.is_empty() {
        return None;
    }
    if command.contains('/') {
        return Some(absolute_from(command, project_root));
    }
    // Bare names are almost always system binaries, but resolving them anyway
    // keeps a PATH entry inside the project from smuggling agent-written code
    // past the check.
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| absolute_from(&dir.join(command).to_string_lossy(), project_root))
            .find(|candidate| candidate.is_file())
    })
}

fn absolute_from(path: &str, project_root: &Path) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        lexical_abs(path)
    } else {
        lexical_abs(&project_root.join(path))
    }
}

/// Resolve `.` and `..` textually. Two spellings of one file must not decide
/// the inside/outside question differently, and `..` must actually leave.
fn lexical_abs(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

fn canonical_or(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| lexical_abs(path))
}

/// Whether `path` sits inside the project, and is therefore agent-writable.
/// Judged both lexically and through symlinks: a symlink inside the project
/// cannot carry its content out of the agent's reach, and one pointing in
/// from outside still names content the agent can rewrite.
fn inside_project(path: &Path, project_root: &Path) -> bool {
    let root = canonical_or(project_root);
    if path.starts_with(&root) || path.starts_with(project_root) {
        return true;
    }
    canonical_or(path).starts_with(&root)
}

// ---------- usage accounting ----------

/// Lifetime usage counters for one extension.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UsageEntry {
    pub calls: u64,
    pub ok: u64,
    pub err: u64,
    /// Unix seconds of the most recent use.
    pub last_used: u64,
}

/// Per-project usage file: what only the dispatcher can measure. Core
/// measures; the agent judges and deletes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UsageFile {
    #[serde(default)]
    pub version: u32,
    /// Every recorded tool call, external or skill read, for base rates.
    #[serde(default)]
    pub total_calls: u64,
    #[serde(default)]
    pub tools: std::collections::BTreeMap<String, UsageEntry>,
    #[serde(default)]
    pub skills: std::collections::BTreeMap<String, UsageEntry>,
}

/// One turn's accumulated usage, merged into the file at turn end.
#[derive(Debug, Default)]
pub struct UsageDelta {
    /// (external tool name, call succeeded)
    pub tools: Vec<(String, bool)>,
    /// Skill names whose SKILL.md the model read this turn.
    pub skills: Vec<String>,
}

impl UsageDelta {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty() && self.skills.is_empty()
    }
}

fn usage_path(dir: &Path) -> PathBuf {
    dir.join("usage.json")
}

pub fn load_usage(data_dir: &Path, project_root: &Path) -> Result<UsageFile, String> {
    let path = usage_path(&project_dir(data_dir, project_root));
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(UsageFile::default()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    serde_json::from_str(&text).map_err(|e| format!("{} is malformed: {e}", path.display()))
}

/// Merge one turn's usage under the ledger lock: one atomic write per turn,
/// no per-call I/O.
pub fn record_usage(
    data_dir: &Path,
    project_root: &Path,
    delta: &UsageDelta,
) -> Result<(), String> {
    if delta.is_empty() {
        return Ok(());
    }
    let dir = project_dir(data_dir, project_root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path(&dir))
        .map_err(|e| format!("cannot open ledger lock: {e}"))?;
    lock.lock_exclusive().map_err(|e| format!("cannot lock ledger: {e}"))?;
    let result = (|| {
        // A malformed usage file starts over rather than blocking turns:
        // usage is telemetry, not policy.
        let mut file = load_usage(data_dir, project_root).unwrap_or_default();
        file.version = 1;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        for (name, ok) in &delta.tools {
            let entry = file.tools.entry(name.clone()).or_default();
            entry.calls += 1;
            if *ok {
                entry.ok += 1;
            } else {
                entry.err += 1;
            }
            entry.last_used = ts;
            file.total_calls += 1;
        }
        for name in &delta.skills {
            let entry = file.skills.entry(name.clone()).or_default();
            entry.calls += 1;
            entry.ok += 1;
            entry.last_used = ts;
            file.total_calls += 1;
        }
        let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        crate::sessions::write_atomic(&usage_path(&dir), json)
    })();
    let _ = fs2::FileExt::unlock(&lock);
    result
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
    fn removing_trailing_records_is_detected() {
        let data = temp("trunc-data");
        let root = temp("trunc-proj");
        for v in ["v1", "v2", "v3"] {
            let files = vec![entry(&root, ".openmax/tools/a.toml", v)];
            sync(&data, &root, &files, Actor::External, None).unwrap();
        }
        let log = log_path(&project_dir(&data, &root));
        let text = std::fs::read_to_string(&log).unwrap();
        let prefix: String = text.lines().take(2).map(|l| format!("{l}\n")).collect();
        std::fs::write(&log, prefix).unwrap();
        let err = history(&data, &root).unwrap_err();
        assert!(err.contains("chain head"), "{err}");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_corrupted_object_is_replaced_with_authentic_bytes() {
        let data = temp("objfix-data");
        let root = temp("objfix-proj");
        let (path, sha, bytes) = entry(&root, ".openmax/tools/a.toml", "authentic");
        let object = project_dir(&data, &root).join("objects").join(&sha);
        std::fs::create_dir_all(object.parent().unwrap()).unwrap();
        std::fs::write(&object, "forged").unwrap();
        sync(&data, &root, &[(path, sha.clone(), bytes)], Actor::External, None).unwrap();
        assert_eq!(std::fs::read_to_string(&object).unwrap(), "authentic");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn usage_merges_across_turns() {
        let data = temp("use-data");
        let root = temp("use-proj");
        let mut delta = UsageDelta::default();
        delta.tools.push(("deploy".into(), true));
        delta.tools.push(("deploy".into(), false));
        delta.skills.push("release".into());
        record_usage(&data, &root, &delta).unwrap();
        record_usage(&data, &root, &delta).unwrap();
        let usage = load_usage(&data, &root).unwrap();
        assert_eq!(usage.total_calls, 6);
        let deploy = &usage.tools["deploy"];
        assert_eq!((deploy.calls, deploy.ok, deploy.err), (4, 2, 2));
        assert_eq!(usage.skills["release"].calls, 2);
        assert!(deploy.last_used > 0);
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn approvals_bind_to_content_and_fail_closed() {
        let data = temp("appr-data");
        let root = temp("appr-proj");
        let sha = sha256_hex(b"name = \"t\"");
        assert!(!is_approved(&data, &root, &sha));
        approve_hash(&data, &root, &sha).unwrap();
        assert!(is_approved(&data, &root, &sha));
        assert!(!is_approved(&data, &root, &sha256_hex(b"edited")), "new content, new approval");

        // A malformed store approves nothing and says why.
        std::fs::write(approved_path(&project_dir(&data, &root)), "{broken").unwrap();
        assert!(!is_approved(&data, &root, &sha));
        assert!(approved_hashes(&data, &root).is_err());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The binding rule itself: what a manifest runs from inside the project
    /// is hashed, what it runs from outside is not.
    #[test]
    fn bound_code_covers_project_files_and_leaves_system_binaries_alone() {
        let root = temp("bound-proj");
        std::fs::write(root.join("run.sh"), "#!/bin/sh\ntrue\n").unwrap();
        std::fs::write(root.join("tool.py"), "print(1)\n").unwrap();

        // A system binary stays the human's reading of an absolute path.
        assert!(bound_code("/bin/echo", &[], &root).is_empty());
        assert!(bound_code("echo", &[], &root).is_empty(), "a PATH name is not project code");

        // A project script binds, by either spelling the spawn accepts (a
        // bare name is not one of them: that is a PATH lookup).
        let sha = sha256_hex(b"#!/bin/sh\ntrue\n");
        for spelling in ["./run.sh".to_string(), root.join("run.sh").display().to_string()] {
            let bound = bound_code(&spelling, &[], &root);
            assert_eq!(bound.len(), 1, "{spelling}");
            assert_eq!(bound[0].sha256.as_deref(), Some(sha.as_str()), "{spelling}");
        }

        // An interpreter's script argument is code too; flags and inline
        // program text are not paths and bind nothing.
        let bound = bound_code("/usr/bin/env", &["python3".into(), "tool.py".into()], &root);
        assert_eq!(bound.len(), 1);
        assert!(bound[0].path.ends_with("tool.py"));
        assert!(bound_code("/bin/sh", &["-c".into(), "echo hi".into()], &root).is_empty());

        // A command named but not yet written has no bytes to approve, and
        // must never read as "nothing to bind".
        let missing = bound_code("./not-written-yet.sh", &[], &root);
        assert_eq!(missing.len(), 1);
        assert!(missing[0].sha256.is_none());
        assert!(!Approvals::default().covers_code(&missing));

        // `..` actually leaves the project, so it is not project code.
        assert!(bound_code("../outside.sh", &[], &root).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A symlink cannot move a file to the other side of the inside/outside
    /// line, in either direction: what matters is whether the agent can
    /// rewrite what runs.
    #[cfg(unix)]
    #[test]
    fn symlinks_do_not_move_code_out_of_the_project() {
        let root = temp("link-proj");
        let outside = temp("link-outside");
        std::fs::write(outside.join("real.sh"), "#!/bin/sh\ntrue\n").unwrap();
        std::fs::write(root.join("plain.sh"), "#!/bin/sh\ntrue\n").unwrap();
        std::os::unix::fs::symlink(outside.join("real.sh"), root.join("alias.sh")).unwrap();
        std::os::unix::fs::symlink(root.join("plain.sh"), outside.join("into.sh")).unwrap();

        // A link inside the project is agent-writable whatever it points at.
        let bound = bound_code("./alias.sh", &[], &root);
        assert_eq!(bound.len(), 1);
        assert!(bound[0].sha256.is_some(), "the bytes that run are what is bound");

        // A link from outside that resolves back in is agent-writable too.
        let bound = bound_code(&outside.join("into.sh").display().to_string(), &[], &root);
        assert_eq!(bound.len(), 1, "a symlink pointing into the project still names project code");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    /// The approval store remembers that a path was live, so a later edit of
    /// an installed hook is distinguishable from a file nobody ever blessed.
    #[test]
    fn approving_a_capability_records_its_path_and_every_hash() {
        let data = temp("cap-data");
        let root = temp("cap-proj");
        let manifest = root.join("gate.toml");
        std::fs::write(&manifest, "event = \"pre_tool_use\"\ncommand = \"/bin/true\"\n").unwrap();
        let shas = vec![sha256_hex(b"manifest"), sha256_hex(b"script")];
        approve_capability(&data, &root, &manifest, &shas).unwrap();

        let recorded = approvals(&data, &root).unwrap();
        assert!(recorded.contains(&shas[0]) && recorded.contains(&shas[1]));
        assert!(recorded.was_live(&manifest));
        assert!(!recorded.was_live(&root.join("other.toml")));

        // A bare hash approval binds content without claiming a path was live.
        approve_hash(&data, &root, &sha256_hex(b"loose")).unwrap();
        let after = approvals(&data, &root).unwrap();
        assert!(after.contains(&sha256_hex(b"loose")));
        assert!(!after.was_live(&root.join("loose")));
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
