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

/// What a record asserts. Omitted from the wire for `Change` and defaulted on
/// read, so every line an older harness wrote keeps its exact bytes and its
/// place in the chain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A capability file appeared, changed, or was removed.
    #[default]
    Change,
    /// A human approved this exact content for unattended execution.
    Approval,
    /// One-time import of a pre-chain `approved.json` (see the approvals
    /// section): after this marker the legacy file is ignored forever.
    ApprovalsImported,
    /// A human stopped expecting an approved capability at this path
    /// (`openmax --forget` after a deliberate deletion). The hashes stay
    /// blessed - approval binds bytes - only the path memory ends.
    PathRetired,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Change => "change",
            Kind::Approval => "approval",
            Kind::ApprovalsImported => "approvals-imported",
            Kind::PathRetired => "path-retired",
        }
    }

    fn is_change(&self) -> bool {
        matches!(self, Kind::Change)
    }
}

/// One observed change. `sha256` is `None` when the file was removed.
/// Approval records reuse the shape: `sha256` is the manifest's content,
/// `also` carries the rest of the hashes blessed in the same act (the
/// project-local code that manifest runs), and `path` is where the human
/// approved it (empty when the caller only knew a hash).
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
    #[serde(default, skip_serializing_if = "Kind::is_change")]
    pub kind: Kind,
    /// Further hashes covered by the same approval act. Absent on change
    /// records, so their lines keep the bytes an older harness wrote.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub also: Vec<String>,
    /// The `event` the approved bytes declared, when the act approved a hook
    /// manifest. Read back instead of the file so an approved gate cannot
    /// rewrite itself into an observer; the chain is what makes this
    /// answer harder to forge than the file it replaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    /// The project-local files those approved bytes named as code, for the
    /// repair carve-out. Same reason: a rewritten manifest must not be able to
    /// widen its own exemption.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub code: Vec<String>,
    /// Whether those approved bytes asked the hook to gate on an event that
    /// gates only on request. Skipped when false so every line written before
    /// the field existed still serializes to the bytes the chain hashed.
    #[serde(default, skip_serializing_if = "is_false")]
    pub blocking: bool,
    /// sha256 of the previous record's serialized line ("" for the first).
    pub prev: String,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// One entry of a sync's outcome, for the refreeze receipt.
#[derive(Clone, Debug)]
pub struct Change {
    pub path: PathBuf,
    pub actor: Actor,
    /// `added`, `modified`, or `removed`.
    pub kind: &'static str,
}

/// Bumped when approvals moved into the chain, so a line written before the
/// move is legible as one (see `adopt_legacy_approvals`).
const RECORD_VERSION: u32 = 2;

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

/// A settlement claim that outlives its process (#103): the ordered
/// (generation, actor) queue every sync path maintains is about the
/// project's ledger, not about any one session, so claims persist beside
/// the ledger and are adopted by whichever session next attempts
/// settlement. The common corruption case (an unverifiable log) leaves
/// this directory writable; when it is not, callers degrade to the
/// in-memory queue exactly as before.
pub type QueuedClaim = (Vec<(PathBuf, String, Vec<u8>)>, Actor);

#[derive(serde::Serialize, serde::Deserialize)]
struct ClaimFile {
    actor: Actor,
    files: Vec<(PathBuf, String, Vec<u8>)>,
}

fn claims_dir(dir: &Path) -> PathBuf {
    dir.join("claims")
}

/// Persist one queued claim as a content-addressed file whose name sorts in
/// arrival order across sessions and processes. Arrival order comes from a
/// counter file incremented under the ledger's flock, never from the wall
/// clock: a same-millisecond race would tie-break arbitrarily and a clock
/// stepped backward would replay generations reversed, landing an older
/// head over a newer one. Best-effort by contract; must never be called
/// while the ledger lock is already held.
pub fn persist_queued_claim(
    data_dir: &Path,
    project_root: &Path,
    claim: &QueuedClaim,
) -> Result<PathBuf, String> {
    let ledger = project_dir(data_dir, project_root);
    let dir = claims_dir(&ledger);
    let body = serde_json::to_vec(&ClaimFile {
        actor: claim.1,
        files: claim.0.clone(),
    })
    .map_err(|e| e.to_string())?;
    with_lock(&ledger, || {
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let seq_path = dir.join("claims.seq");
        let seq: u64 = std::fs::read_to_string(&seq_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        std::fs::write(&seq_path, format!("{}", seq + 1)).map_err(|e| e.to_string())?;
        let name = format!("{seq:012}-{}.json", &sha256_hex(&body)[..12]);
        let path = dir.join(name);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &body).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
        Ok(path)
    })
}

/// Every persisted claim for this project, oldest first. Unreadable or
/// corrupt files are skipped in place: they stay on disk as evidence
/// rather than being deleted or blocking settlement.
pub fn load_queued_claims(data_dir: &Path, project_root: &Path) -> Vec<(PathBuf, QueuedClaim)> {
    let dir = claims_dir(&project_dir(data_dir, project_root));
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut names: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    names.sort();
    names
        .into_iter()
        .filter_map(|path| {
            let bytes = std::fs::read(&path).ok()?;
            let parsed: ClaimFile = serde_json::from_slice(&bytes).ok()?;
            Some((path.clone(), (parsed.files, parsed.actor)))
        })
        .collect()
}

/// Remove a landed claim's file. Best-effort: a claim that cannot be
/// removed re-adopts later and re-lands as a no-op delta.
pub fn remove_claim_file(path: &Path) {
    let _ = std::fs::remove_file(path);
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

/// The head an append is *about to* create, written and flushed before the
/// records are. A log that runs past its pin is only accepted when this file
/// names exactly the head those extra records produce, which is what a crash
/// between the two writes leaves behind - and which a forged tail cannot
/// claim without also writing here.
fn pending_head_path(dir: &Path) -> PathBuf {
    dir.join("chain-head.pending")
}

fn legacy_approved_path(dir: &Path) -> PathBuf {
    dir.join("approved.json")
}

/// Every unverifiable-ledger error ends with this. The old message asserted
/// tampering and stopped there, which left the ledger write-dead with no
/// stated way back (recovery was "delete the directory", documented nowhere).
const REPAIR_HINT: &str = "; the ledger will not append until a human repairs it: `openmax --ledger-repair` quarantines the damaged log (nothing is deleted) and starts a new chain";

fn tampered(message: String) -> String {
    format!("{message}{REPAIR_HINT}")
}

/// What the stored pin says about the log that was just verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pin {
    /// The stored head is the log's final record.
    Matches,
    /// No pin and no records: a project the ledger has never seen.
    Fresh,
    /// The log carries records past the pin and the pending pin names exactly
    /// the head they produce: an append that landed while its pin did not.
    Interrupted,
}

struct Verified {
    records: Vec<Record>,
    /// sha256 of the log's final line ("" when there are no records).
    head: String,
    pin: Pin,
    /// How many leading records the stored chain head covers. Records at or
    /// past this index landed without their pin (an interrupted append), and
    /// nothing there may grant or retire authority until a human repairs it:
    /// the pin is what separates history somebody vouched for from a tail
    /// anybody could have written.
    pinned: usize,
}

impl Verified {
    /// Records past the pin that grant or retire authority. A crashed sync
    /// leaves only observations behind; anything stronger in an unpinned
    /// tail has to wait for a human.
    fn unpinned_authority(&self) -> impl Iterator<Item = &Record> {
        self.records[self.pinned..].iter().filter(|r| !r.kind.is_change())
    }
}

/// Refuse to append while unpinned authority records exist: any append moves
/// the pin past them, which would convert a tail nobody vouched for into
/// approved history. `openmax --ledger-repair` is the one door out, and it
/// quarantines such a tail rather than blessing it.
fn refuse_unpinned_authority(verified: &Verified) -> Result<(), String> {
    let count = verified.unpinned_authority().count();
    if count == 0 {
        return Ok(());
    }
    Err(tampered(format!(
        "the log carries {count} approval-grade record(s) past the pinned chain head - an approval whose write was interrupted, or a tail planted outside the harness; either way they grant nothing"
    )))
}

/// The full history plus whether the last append's pin never landed.
pub struct History {
    pub records: Vec<Record>,
    /// True when a crash left records past the pin; the next sync re-pins.
    /// Nothing was removed, so this is a repairable state, not tampering.
    pub interrupted_write: bool,
    /// How many leading records the stored chain head vouches for. Records
    /// past this index are the interrupted (unpinned) tail.
    pub pinned: usize,
}

fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// Read and verify the log, the chain, and the pin together. A malformed
/// line, a broken chain, a missing pin, or a pin the log does not match is an
/// error, never silently skipped or displayed as authentic: a ledger that
/// cannot be trusted must not be read around.
fn read_verified(dir: &Path) -> Result<Verified, String> {
    let path = log_path(dir);
    let (text, log_present) = match std::fs::read_to_string(&path) {
        Ok(text) => (text, true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (String::new(), false),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let mut records = Vec::new();
    // Hash of every record line in order, so a stored pin can be located
    // inside the log rather than only compared against its end.
    let mut line_hashes = Vec::new();
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
            // Chain semantics name the record *after* the damage: this line's
            // `prev` is what fails to match, but the edit is at or before the
            // line it points at. Say both so the repair looks in one place.
            return Err(tampered(format!(
                "{} line {}: hash chain broken - this record does not follow line {}, so line {} or an earlier one was modified or removed outside the harness",
                path.display(),
                i + 1,
                i,
                i
            )));
        }
        prev = sha256_hex(line.as_bytes());
        line_hashes.push(prev.clone());
        records.push(record);
    }

    let stored = read_trimmed(&chain_head_path(dir));
    let pending = read_trimmed(&pending_head_path(dir));
    let pin = match stored.as_deref() {
        Some(stored) if stored == prev => Pin::Matches,
        Some(stored) if !records.is_empty() && pending.as_deref() == Some(prev.as_str()) => {
            // The pending pin vouches for the log's final head, but only a
            // tail that grows pinned history is an interrupted append: the
            // stored head must still name a record inside this log. A log
            // that merely ends where the pending file says, without the
            // pinned prefix in it, is a rewrite carrying a forged receipt.
            if !line_hashes.iter().any(|h| h == stored) {
                return Err(tampered(format!(
                    "{}: the stored chain head names no record in the log, yet a pending head vouches for its end - the pinned history was rewritten outside the harness",
                    path.display()
                )));
            }
            Pin::Interrupted
        }
        // A surviving pin proves history existed, so an empty or absent log is
        // deletion, not a fresh project. This is checked before the log's own
        // absence: reaching for the log first is how `rm log.jsonl` used to
        // read as "no history yet".
        Some(_) if records.is_empty() => {
            return Err(tampered(format!(
                "{}: a chain head is stored but the log is {} - the history it pins was removed outside the harness",
                dir.display(),
                if log_present { "empty" } else { "missing" }
            )))
        }
        Some(_) => {
            return Err(tampered(format!(
                "{}: the log's final record does not match the stored chain head - trailing records were removed or rewritten outside the harness",
                path.display()
            )))
        }
        // Genuine first run: nothing written, nothing pinned.
        None if records.is_empty() => Pin::Fresh,
        // Every append writes the pin, so records without one cannot be
        // checked for truncation. Deleting a file is easier than rewriting a
        // chain, so this has to read as tamper rather than as a quiet
        // downgrade to "internally consistent".
        None => {
            return Err(tampered(format!(
                "{}: the log has {} records but no chain head - the pin every append writes is missing, so trailing records cannot be ruled out as removed",
                path.display(),
                records.len()
            )))
        }
    };
    let pinned = match pin {
        Pin::Matches => records.len(),
        Pin::Fresh => 0,
        // The anchor check above guarantees the position exists.
        Pin::Interrupted => {
            let stored = stored.as_deref().unwrap_or_default();
            line_hashes.iter().position(|h| h == stored).map(|i| i + 1).unwrap_or(0)
        }
    };
    Ok(Verified { records, head: prev, pin, pinned })
}

/// Read the full history, verifying the hash chain and the truncation pin.
pub fn history(data_dir: &Path, project_root: &Path) -> Result<Vec<Record>, String> {
    read_verified(&project_dir(data_dir, project_root)).map(|v| v.records)
}

/// One human act against a capability's approval state, for surfacing an
/// approval a session did not witness (#199): the path, whether it granted
/// or retired, the actor, and the session that recorded it (None for a CLI
/// act outside any session). Identity is content-stable across reads.
#[derive(Clone, Debug)]
pub struct ApprovalEvent {
    pub path: PathBuf,
    pub granted: bool,
    pub session_id: Option<String>,
    /// A stable id for this record (path + sha + ts + kind), so a session
    /// can remember which events it has already narrated.
    pub id: u64,
}

/// Every approval/retirement the pin vouches for, oldest first. Cheap: one
/// verified read, no object hashing. Only the pinned prefix is reported:
/// an unpinned tail is bytes nobody vouched for, and enforcement treats its
/// grants as inert (`refuse_unpinned_authority`: "they grant nothing"), so
/// narrating one would tell the session a grant exists that the gates will
/// not honor. When an interrupted approval is later re-pinned by a human
/// path, its events surface then, under the same stable ids. An
/// unverifiable chain is an error the caller must narrate, not an empty
/// history: silence read as "no approval activity" in the one surface built
/// to report exactly this state.
pub fn approval_events(
    data_dir: &Path,
    project_root: &Path,
) -> Result<Vec<ApprovalEvent>, String> {
    let dir = project_dir(data_dir, project_root);
    let verified = read_verified(&dir)?;
    let generation = repair_generation(&dir);
    Ok(verified.records[..verified.pinned]
        .iter()
        .enumerate()
        .filter_map(|(idx, r)| {
            let granted = match r.kind {
                Kind::Approval => true,
                Kind::PathRetired => false,
                _ => return None,
            };
            Some(ApprovalEvent {
                path: r.path.clone(),
                granted,
                session_id: r.session_id.clone(),
                id: approval_event_id(generation, idx, r),
            })
        })
        .collect())
}

/// How many times this project's ledger has been repaired: one quarantined
/// log (`log.jsonl.unverified-<ts>`) per `--ledger-repair`. Durable (the files
/// are never deleted) and constant between repairs, so it identifies the chain
/// GENERATION without churning event ids on every read.
fn repair_generation(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("log.jsonl.unverified-"))
                .count() as u64
        })
        .unwrap_or(0)
}

/// A per-session watermark identity for one approval/retirement record. Both
/// the chain GENERATION (how many repairs preceded it) and the chain POSITION
/// are part of it, not only the (path, sha, second, kind): otherwise an
/// approve/retire/re-approve of identical bytes in one Unix second collides on
/// position (Greptile Y3), and an identical approval recorded just before and
/// just after a `--ledger-repair` - both at position zero of their chains -
/// collides across the repair (Greptile). A running session that watermarked
/// the pre-repair event would then suppress the post-repair one and keep stale
/// approval context. Both parts are stable between repairs (the chain is
/// append-only and the quarantine count only grows on repair), so a record
/// keeps its id from turn to turn while a genuinely new one gets a fresh id.
fn approval_event_id(generation: u64, chain_index: usize, r: &Record) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    generation.hash(&mut h);
    chain_index.hash(&mut h);
    r.path.hash(&mut h);
    r.sha256.hash(&mut h);
    r.ts.hash(&mut h);
    r.kind.as_str().hash(&mut h);
    h.finish()
}

/// History plus the interrupted-write flag, for callers that report state.
pub fn read(data_dir: &Path, project_root: &Path) -> Result<History, String> {
    read_verified(&project_dir(data_dir, project_root)).map(|v| History {
        records: v.records,
        interrupted_write: v.pin == Pin::Interrupted,
        pinned: v.pinned,
    })
}

/// The last known hash per path (None = removed), from the full history.
/// Only change records describe files; approvals carry a hash, not a state.
fn head(records: &[Record]) -> HashMap<PathBuf, Option<String>> {
    let mut map = HashMap::new();
    for r in records.iter().filter(|r| r.kind.is_change()) {
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
    with_lock(&dir, || sync_locked(&dir, files, actor, session_id))
}

/// Run `f` under the ledger's exclusive flock. Never call this from inside
/// another `with_lock`: flock is per open file description, so a second lock
/// in the same process would wait on itself forever.
fn with_lock<R>(dir: &Path, f: impl FnOnce() -> Result<R, String>) -> Result<R, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path(dir))
        .map_err(|e| format!("cannot open ledger lock: {e}"))?;
    lock.lock_exclusive().map_err(|e| format!("cannot lock ledger: {e}"))?;
    let result = f();
    let _ = fs2::FileExt::unlock(&lock);
    result
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Durable atomic replace. Used only where a crash that loses the write would
/// be *harmful*; the ordering below spends exactly one of these per append,
/// because on macOS each one is a full drive barrier.
fn write_durable(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path
        .file_name()
        .ok_or_else(|| format!("{}: no file name", path.display()))?
        .to_string_lossy()
        .to_string();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!("{name}.writing"));
    let mut file = std::fs::File::create(&tmp)
        .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    file.write_all(bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    file.sync_all().map_err(|e| format!("cannot flush {}: {e}", tmp.display()))?;
    drop(file);
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot replace {}: {e}", path.display())
    })?;
    // Best effort: directory fsync is the POSIX way to make the rename
    // durable, and is not available on every platform.
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Append chained lines and move the pin in an order no crash can turn into a
/// false tamper report. Three writes, and what each one costs is chosen by
/// what its loss would mean:
///
/// * the pending pin is durable *before* the records exist, so a crash can
///   never leave records whose head nothing vouched for;
/// * the records are flushed before the pin moves, so the pin can never name
///   a record the log does not have;
/// * the pin itself needs no barrier: losing it lands in exactly the state
///   the pending pin describes, which reads as an interrupted write and
///   re-pins on the next sync.
fn append_chained(dir: &Path, lines: &str, new_head: &str) -> Result<(), String> {
    write_durable(&pending_head_path(dir), new_head.as_bytes())?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path(dir))
        .map_err(|e| format!("cannot append to ledger: {e}"))?;
    file.write_all(lines.as_bytes())
        .map_err(|e| format!("cannot append to ledger: {e}"))?;
    file.sync_data().map_err(|e| format!("cannot flush ledger: {e}"))?;
    crate::sessions::write_atomic(&chain_head_path(dir), new_head)?;
    let _ = std::fs::remove_file(pending_head_path(dir));
    Ok(())
}

fn sync_locked(
    dir: &Path,
    files: &[(PathBuf, String, Vec<u8>)],
    actor: Actor,
    session_id: Option<&str>,
) -> Result<Vec<Change>, String> {
    let verified = read_verified(dir)?;
    // A sync appends and re-pins, which must never quietly bless an unpinned
    // authority tail; a change-only tail (a crashed sync) heals below.
    refuse_unpinned_authority(&verified)?;
    let known = head(&verified.records);
    // Keyed on change records, not on the log: a ledger that so far holds
    // only approvals has still never seen this project's files.
    let effective_actor = if known.is_empty() { Actor::Initial } else { actor };
    let mut prev = verified.head.clone();

    let ts = unix_now();

    let mut changes = Vec::new();
    let mut lines = String::new();
    // A turn is what gives an agent the chance to plant an `approved.json`,
    // so a turn is where a ledger with nothing to inherit says so, once.
    if let Some(mut marker) = seal_marker(dir, &verified.records, ts) {
        marker.prev = prev.clone();
        let line = serde_json::to_string(&marker).map_err(|e| e.to_string())?;
        prev = sha256_hex(line.as_bytes());
        lines.push_str(&line);
        lines.push('\n');
    }
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
            kind: Kind::Change,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
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
        if last.is_some() && !seen.contains(&path) {
            let record = Record {
                v: RECORD_VERSION,
                ts,
                path: path.clone(),
                sha256: None,
                actor: effective_actor,
                session_id: session_id.map(str::to_string),
                kind: Kind::Change,
                also: Vec::new(),
                event: None,
                code: Vec::new(),
                blocking: false,
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
        append_chained(dir, &lines, &prev)?;
    } else if verified.pin == Pin::Interrupted {
        // Heal a crash-interrupted append even when this generation changed
        // nothing: the records are intact and chained, only the pin is stale.
        crate::sessions::write_atomic(&chain_head_path(dir), &verified.head)?;
        let _ = std::fs::remove_file(pending_head_path(dir));
    }
    Ok(changes)
}

// ---------- content-bound approvals ----------
//
// Approvals are ledger records, not a file beside the ledger. `approved.json`
// was plain JSON in the same directory as the chain with none of its
// protection, so anything that could append a line of JSON - including the
// agent's own `bash`, which has full host authority - could approve its own
// hook and have the next session run it. As records they inherit the chain,
// the pin, and the audit trail: a forged approval has to forge the chain
// (detectable, and the same ceiling trust already lives at), and every real
// one shows up in `openmax --ledger` with its time, actor, and session.

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
    /// What the approved bytes at a hook path actually declared. Reconciling
    /// a modified hook has to judge it by this and never by the file on disk:
    /// the modified content is the part the agent controls, so reading its
    /// `event` would let an approved gate rewrite itself into an observer and
    /// stop gating, and reading its `command` would let it hand itself a
    /// repair exemption for a path nobody blessed.
    #[serde(default)]
    hooks: Vec<ApprovedHook>,
}

/// One hook as a human approved it: the shape reconciliation must remember,
/// because the file itself can no longer be trusted to describe it.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ApprovedHook {
    path: String,
    /// The `event` the approved bytes declared.
    event: String,
    /// The project-local files the approved bytes named as code.
    #[serde(default)]
    code: Vec<String>,
    /// Whether the approved bytes asked for a blocking hook. `default` is
    /// what makes every record written before this field existed read back
    /// as the observer a human actually approved.
    #[serde(default)]
    blocking: bool,
}

impl ApprovedHook {
    /// Whether what was approved gates calls. A gate that stops running is
    /// fail-open, which is why this question is asked of the approved content
    /// rather than of whatever now sits at the path.
    ///
    /// Both halves are load-bearing across the upgrade. A record with no
    /// `blocking` field is a hook a human approved as an observer, so it must
    /// not be promoted into a gate by a later build; a `pre_tool_use` record
    /// from the same era carries no flag either, and must not be demoted out
    /// of one.
    pub fn is_gate(&self) -> bool {
        self.blocking || matches!(self.event.as_str(), "pre_tool_use" | "user_prompt_submit")
    }

    pub fn event(&self) -> &str {
        &self.event
    }

    /// Whether the approved bytes asked to gate an event that gates only on
    /// request, for diagnostics that name the shape a human installed.
    pub fn blocking(&self) -> bool {
        self.blocking
    }

    /// The code paths the approved content named, for the repair carve-out.
    pub fn code_paths(&self) -> impl Iterator<Item = PathBuf> + '_ {
        self.code.iter().map(PathBuf::from)
    }
}

/// What a human has approved for this project: exact content hashes, plus the
/// capability paths those approvals were granted at.
#[derive(Clone, Debug, Default)]
pub struct Approvals {
    hashes: std::collections::HashSet<String>,
    paths: std::collections::HashSet<PathBuf>,
    hooks: Vec<ApprovedHook>,
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

    /// Every path a human approved a capability at. Reconciliation reads this
    /// rather than the directory listing: a file that was deleted produces no
    /// entry to iterate, and absence is exactly the case that must not pass.
    pub fn live_paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.paths.iter()
    }

    /// What the human approved at this hook path, if anything. The answer to
    /// "was this a gate" and "what code did it run" has to come from here,
    /// never from the current bytes: those are what an edit controls.
    pub fn approved_hook(&self, path: &Path) -> Option<&ApprovedHook> {
        let target = canonical_or(path);
        self.hooks.iter().find(|h| Path::new(&h.path) == target)
    }

    /// Every bound code file is readable and approved. An unreadable one is
    /// never covered: a command named but not yet written has no bytes to
    /// approve, and must not read as "nothing to bind".
    pub fn covers_code(&self, code: &[BoundCode]) -> bool {
        code.iter()
            .all(|c| c.sha256.as_deref().is_some_and(|sha| self.contains(sha)))
    }
}

/// Everything the chain says a human approved. One act can bless several
/// hashes (a manifest plus the code it runs), so a record's hashes are read
/// together; the path it was granted at is what `was_live` remembers, and the
/// shape it recorded (the event those bytes declared, and the code they named)
/// is what reconciliation judges a modified hook by, since the file on disk is
/// the part an edit controls. Records are consumed in order: a later act at
/// a path replaces the shape an earlier one recorded, so a human who
/// deliberately changes a hook's event gets the new shape, and a later
/// `PathRetired` ends the path memory and the shape together. The exhaustive
/// match is deliberate - a new record kind must decide here what it means for
/// authority.
fn approvals_from(records: &[Record]) -> Approvals {
    let mut approvals = Approvals::default();
    for r in records {
        match r.kind {
            Kind::Approval => {
                approvals.hashes.extend(r.sha256.iter().cloned());
                approvals.hashes.extend(r.also.iter().cloned());
                if r.path.as_os_str().is_empty() {
                    continue;
                }
                approvals.paths.insert(r.path.clone());
                let key = r.path.display().to_string();
                approvals.hooks.retain(|h| h.path != key);
                if let Some(event) = &r.event {
                    approvals.hooks.push(ApprovedHook {
                        path: key,
                        event: event.clone(),
                        code: r.code.clone(),
                        blocking: r.blocking,
                    });
                }
            }
            Kind::PathRetired => {
                approvals.paths.remove(&r.path);
                let key = r.path.display().to_string();
                approvals.hooks.retain(|h| h.path != key);
            }
            Kind::Change | Kind::ApprovalsImported => {}
        }
    }
    approvals
}

/// Everything a human approved for this project. Approval binds to content,
/// not path: any edit changes the hash and revokes itself. No approval records
/// means nothing approved; an unverifiable ledger is an error, and callers
/// treat that as nothing approved (fail closed) while surfacing the reason.
pub fn approvals(data_dir: &Path, project_root: &Path) -> Result<Approvals, String> {
    let dir = project_dir(data_dir, project_root);
    if legacy_approved_path(&dir).exists() {
        // Never an import: either the chain has already settled the question
        // and the file is set aside, or it waits for a human to adopt it.
        with_lock(&dir, || settle_legacy_store_locked(&dir))?;
    }
    // The pin plus the log's length name one exact chain: a different log that
    // hashed to the same pin would be a sha256 collision on the final record.
    // Every mutating call and every hook run asks this question, so verifying
    // thousands of records again each time is worth avoiding - but only
    // against a key that cannot be forged into a stale answer. A pending
    // legacy store is part of the answer and is not covered by that key, so
    // while one is on disk the cache steps aside.
    let pending = read_legacy_store(&dir);
    let key = pending
        .is_none()
        .then(|| {
            read_trimmed(&chain_head_path(&dir))
                .zip(std::fs::metadata(log_path(&dir)).ok().map(|m| m.len()))
        })
        .flatten();
    if let Some((pin, len)) = &key {
        if let Some(hit) = cached_approvals(&dir, pin, *len) {
            return Ok(hit);
        }
    }
    let verified = read_verified(&dir)?;
    // Only the pinned prefix speaks for a human: an unpinned tail is bytes
    // nobody vouched for, so what it grants or retires stays inert.
    let mut approved = approvals_from(&verified.records[..verified.pinned]);
    if let Some((_, file)) = pending {
        // An unadopted store contributes restriction and nothing else: the
        // paths it claims were live, so a gate a human installed before the
        // upgrade fails closed instead of quietly going inert. Its hashes and
        // hook shapes wait for `openmax --adopt-approvals`, because those are
        // the parts that would grant execution or relax a gate.
        if inheritable(&verified.records) {
            for path in file?.paths {
                approved.paths.insert(PathBuf::from(path));
            }
        }
        return Ok(approved);
    }
    if let Some((pin, len)) = key {
        remember_approvals(&dir, pin, len, &approved);
    }
    Ok(approved)
}

/// The approved sha256 set alone, for callers with no path question to ask.
pub fn approved_hashes(
    data_dir: &Path,
    project_root: &Path,
) -> Result<std::collections::HashSet<String>, String> {
    approvals(data_dir, project_root).map(|a| a.hashes)
}

type ApprovalCache = HashMap<PathBuf, (String, u64, Approvals)>;

fn approval_cache() -> &'static std::sync::Mutex<ApprovalCache> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<ApprovalCache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn cached_approvals(dir: &Path, pin: &str, len: u64) -> Option<Approvals> {
    let cache = approval_cache().lock().ok()?;
    match cache.get(dir) {
        Some((cached_pin, cached_len, approved)) if cached_pin == pin && *cached_len == len => {
            Some(approved.clone())
        }
        _ => None,
    }
}

fn remember_approvals(dir: &Path, pin: String, len: u64, approved: &Approvals) {
    if let Ok(mut cache) = approval_cache().lock() {
        cache.insert(dir.to_path_buf(), (pin, len, approved.clone()));
    }
}

/// A ledger whose chain has already answered the legacy question: some record
/// carries the import marker, written either by an adoption or by first
/// contact with a store-free ledger. After that, an `approved.json` is
/// something that appeared next to a modern chain - which is the forgery this
/// design exists to close - so it is set aside unread.
fn sealed(records: &[Record]) -> bool {
    records.iter().any(|r| r.kind == Kind::ApprovalsImported)
}

/// Whether an `approved.json` beside this ledger could be something this
/// project actually inherited. It takes a record written before approvals
/// joined the chain to make that plausible, and no marker settling the
/// question since. A ledger this build wrote from scratch never had a store,
/// so a file next to it is a plant and is set aside unread. The test asks for
/// *some* older record rather than only older ones, so a human who approves
/// something else first does not silently forfeit the store they were about
/// to adopt.
fn inheritable(records: &[Record]) -> bool {
    records.iter().any(|r| r.v < RECORD_VERSION) && !sealed(records)
}

/// A legacy `approved.json` no chain vouches for.
///
/// Its hashes are what grant execution, and its hook shapes are what can
/// *relax* a gate into an observer, so neither takes effect until a human
/// adopts it: an unauthenticated file must never become authority on its own,
/// which is the whole point of moving approvals into the chain. The paths it
/// claims were live are read even while it waits, because a path can only make
/// the harness stricter - content at a path a human approved is a modified
/// capability, and a gate there fails closed. So the worst a planted file can
/// do before a human looks at it is cost the project availability, which an
/// agent holding `bash` has anyway; what it cannot do is make anything run.
pub struct PendingLegacy {
    pub path: PathBuf,
    /// Digest of the exact bytes this preview describes. Adoption requires it
    /// back, so what a human vouches for is what gets chained - a file
    /// rewritten between the summary and the confirmation is a different
    /// store, and importing it would launder bytes nobody saw.
    pub sha256: String,
    pub hashes: usize,
    pub paths: Vec<PathBuf>,
    pub shapes: usize,
    /// The file exists but does not parse, so not even its paths can be read.
    pub malformed: bool,
}

/// One read of the legacy store: the digest of the exact bytes read, and what
/// they parse to. Everything downstream of a human decision keys on the
/// digest, because "the file" is not a stable thing an approval can bind to -
/// the bytes shown are.
fn read_legacy_store(dir: &Path) -> Option<(String, Result<ApprovedFile, String>)> {
    let legacy = legacy_approved_path(dir);
    let bytes = std::fs::read(&legacy).ok()?;
    let parsed = serde_json::from_slice(&bytes).map_err(|e| {
        format!(
            "{} is malformed ({e}); it can be adopted only after it is fixed, or removed",
            legacy.display()
        )
    });
    Some((sha256_hex(&bytes), parsed))
}

/// The legacy store waiting on a human, if there is one. `--check` names it
/// and `--adopt-approvals` acts on it; both need the same read.
pub fn pending_legacy(data_dir: &Path, project_root: &Path) -> Option<PendingLegacy> {
    let dir = project_dir(data_dir, project_root);
    let legacy = legacy_approved_path(&dir);
    if !legacy.exists() || !inheritable(&read_verified(&dir).ok()?.records) {
        return None;
    }
    let (sha256, parsed) = read_legacy_store(&dir)?;
    Some(match parsed {
        Ok(file) => PendingLegacy {
            path: legacy,
            sha256,
            hashes: file.hashes.len(),
            paths: file.paths.iter().map(PathBuf::from).collect(),
            shapes: file.hooks.len(),
            malformed: false,
        },
        Err(_) => PendingLegacy {
            path: legacy,
            sha256,
            hashes: 0,
            paths: Vec::new(),
            shapes: 0,
            malformed: true,
        },
    })
}

/// The marker that settles the legacy question for a pre-upgrade ledger with
/// nothing to inherit, recording that nothing was. Written by the first sync
/// after the upgrade - a write path, so reads stay reads - because otherwise
/// the window stays open for as long as the project stays quiet and an
/// `approved.json` planted later reads as an heirloom worth asking about. A
/// turn is also the only thing that gives an agent the chance to plant one.
fn seal_marker(dir: &Path, records: &[Record], ts: u64) -> Option<Record> {
    if !inheritable(records) || legacy_approved_path(dir).exists() {
        return None;
    }
    Some(Record {
        v: RECORD_VERSION,
        ts,
        path: PathBuf::new(),
        sha256: None,
        actor: Actor::Initial,
        session_id: None,
        kind: Kind::ApprovalsImported,
        also: Vec::new(),
        event: None,
        code: Vec::new(),
        blocking: false,
        prev: String::new(),
    })
}

/// Decide what a legacy file beside this ledger is. Sealed chain: it appeared
/// after the question was settled, so set it aside unread. Otherwise it stays
/// exactly where it is, pending a human - this is the one path that must never
/// quietly turn a file into authority.
fn settle_legacy_store_locked(dir: &Path) -> Result<(), String> {
    let legacy = legacy_approved_path(dir);
    if !legacy.exists() {
        return Ok(());
    }
    if !inheritable(&read_verified(dir)?.records) {
        let aside = dir.join(format!("approved.json.ignored-{}", unix_now()));
        let _ = std::fs::rename(&legacy, &aside);
        let _ = std::fs::remove_file(&legacy);
    }
    Ok(())
}

/// What `openmax --adopt-approvals` folded into the chain.
#[derive(Debug)]
pub struct Adopted {
    pub hashes: usize,
    pub paths: usize,
    pub shapes: usize,
}

/// Adopt a legacy `approved.json` into the chain, once, on a human's say-so.
/// Imported entries are `Initial`: the file carried no time, no actor, and no
/// integrity, so no stronger claim can be made about where they came from -
/// and `--ledger` names them, so a human can audit exactly what was inherited.
/// All three of its shapes are carried: the hash set, the approved paths that
/// tell a modified gate from one nobody ever installed, and the per-hook shape
/// (event plus the code it named) that says whether a modified hook used to
/// gate. A store old enough to remember no shape adopts the path alone, which
/// reconciliation already reads as a gate - the safe answer when the question
/// can no longer be asked. The marker closes the window behind it.
///
/// `vouched_sha` is the digest from the `PendingLegacy` the human was shown.
/// The confirmation prompt is an open interval any process can write across,
/// so the say-so binds to bytes, not to a path: if the file on disk no longer
/// hashes to what was previewed, nothing is adopted and the human is asked to
/// look again.
pub fn adopt_legacy_approvals(
    data_dir: &Path,
    project_root: &Path,
    vouched_sha: &str,
) -> Result<Adopted, String> {
    let dir = project_dir(data_dir, project_root);
    with_lock(&dir, || {
        let legacy = legacy_approved_path(&dir);
        let verified = read_verified(&dir)?;
        if !inheritable(&verified.records) {
            return Err(format!(
                "{} is not an inherited store: this ledger has never kept approvals anywhere but its own chain, so that file appeared beside a chain already keeping the answer",
                legacy.display()
            ));
        }
        refuse_unpinned_authority(&verified)?;
        let file = match read_legacy_store(&dir) {
            Some((sha, _)) if sha != vouched_sha => {
                return Err(format!(
                    "{} changed after it was shown: the bytes on disk are not the bytes a human vouched for, so nothing was adopted; run `openmax --adopt-approvals` again to review what is there now",
                    legacy.display()
                ));
            }
            Some((_, file)) => file?,
            None => return Err(format!("{} is gone; nothing to adopt", legacy.display())),
        };
        let ts = unix_now();
        let imported = |path: PathBuf, hashes: &[String], shape: Option<&ApprovedHook>| Record {
            v: RECORD_VERSION,
            ts,
            path,
            sha256: hashes.first().cloned(),
            actor: Actor::Initial,
            session_id: None,
            kind: Kind::Approval,
            also: hashes.iter().skip(1).cloned().collect(),
            event: shape.map(|h| h.event.clone()),
            code: shape.map(|h| h.code.clone()).unwrap_or_default(),
            blocking: shape.is_some_and(|h| h.blocking),
            prev: String::new(),
        };
        let mut records = Vec::new();
        if !file.hashes.is_empty() {
            records.push(imported(PathBuf::new(), &file.hashes, None));
        }
        // One record per approved path: a record carries a single path, and
        // dropping them would turn every installed gate into one nobody
        // blessed. The remembered shape rides the path it describes, so an
        // observe hook a human really installed does not come back as a
        // demoted gate. A shape whose path is not in the path set has nothing
        // to hang on; the released store never wrote one, and inventing a path
        // memory from it would grant authority the old file never carried.
        for path in &file.paths {
            let shape = file.hooks.iter().find(|h| h.path == *path);
            records.push(imported(PathBuf::from(path), &[], shape));
        }
        records.push(Record {
            v: RECORD_VERSION,
            ts,
            path: legacy.clone(),
            sha256: None,
            actor: Actor::Initial,
            session_id: None,
            kind: Kind::ApprovalsImported,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        });
        let (lines, head) = chain(records, &verified.head)?;
        append_chained(&dir, &lines, &head)?;
        // Only now: an unremovable legacy file must not cost the user their
        // approvals, and the marker already makes it inert.
        let _ = std::fs::remove_file(&legacy);
        Ok(Adopted {
            hashes: file.hashes.len(),
            paths: file.paths.len(),
            shapes: file.hooks.len(),
        })
    })
}

/// Link records onto `head`, returning the lines to append and the new head.
fn chain(records: Vec<Record>, head: &str) -> Result<(String, String), String> {
    let mut prev = head.to_string();
    let mut lines = String::new();
    for mut record in records {
        record.prev = prev.clone();
        let line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
        prev = sha256_hex(line.as_bytes());
        lines.push_str(&line);
        lines.push('\n');
    }
    Ok((lines, prev))
}

/// Record a human approval of exact content. Serialized under the ledger lock.
pub fn approve_hash(data_dir: &Path, project_root: &Path, sha: &str) -> Result<(), String> {
    approve(data_dir, project_root, &[sha.to_string()], None, None)
}

/// Record a human approval of a capability file: the hashes they blessed, and
/// the path they blessed them at.
pub fn approve_capability(
    data_dir: &Path,
    project_root: &Path,
    path: &Path,
    shas: &[String],
) -> Result<(), String> {
    approve(data_dir, project_root, shas, Some(path), None)
}

/// Approve a capability the way a human at THIS session's card just did, so
/// the record carries the session id (actor `Session`). The turn-start
/// reconciliation excludes a session's own grants by that id; without it, the
/// session would be told the next turn that its own card approval was
/// "activity outside this session" (Greptile). The no-session
/// `approve_capability` stays for the CLI `--approve` path, which has no
/// session to attribute the act to.
pub fn approve_capability_in_session(
    data_dir: &Path,
    project_root: &Path,
    path: &Path,
    shas: &[String],
    session_id: &str,
) -> Result<(), String> {
    approve(data_dir, project_root, shas, Some(path), Some(session_id))
}

/// One approval act, as one chained record: every hash it blessed, the path it
/// was granted at, and - for a hook - the shape those bytes declared, read
/// while they are still the approved ones. Nothing new means no record, so
/// re-approving unchanged content does not grow the log; a shape that moved is
/// something new, because that is how a human retracts a gate or installs one.
fn approve(
    data_dir: &Path,
    project_root: &Path,
    shas: &[String],
    path: Option<&Path>,
    session_id: Option<&str>,
) -> Result<(), String> {
    let dir = project_dir(data_dir, project_root);
    with_lock(&dir, || {
        settle_legacy_store_locked(&dir)?;
        let verified = read_verified(&dir)?;
        refuse_unpinned_authority(&verified)?;
        let known = approvals_from(&verified.records[..verified.pinned]);
        // The caller hashed the manifest, a human vouched for it, and only
        // now is the record written - an open interval any process with
        // `bash` can write across, the same interval adoption already
        // refuses to trust. So the record's shape (the event that decides
        // gate-or-observer, the code list the repair carve-out honors) is
        // derived from bytes proven to hash to the vouched manifest, never
        // from an unchecked read of the path: a manifest swapped inside the
        // interval would otherwise land a gate's hash on file wearing an
        // observer's shape. The code hashes in `shas[1..]` need no check -
        // they only ever bless bytes the caller read, so a swapped script
        // stays unapproved and fails closed on its own.
        let shape = match path {
            None => None,
            Some(p) => {
                let vouched = shas.first().map(String::as_str).unwrap_or_default();
                let bytes = std::fs::read(p).map_err(|e| {
                    format!(
                        "cannot read {} while recording its approval ({e}); nothing was approved",
                        p.display()
                    )
                })?;
                if sha256_hex(&bytes) != vouched {
                    return Err(format!(
                        "{} changed after it was shown: the bytes on disk are not the bytes vouched for, so nothing was approved; review the file and approve it again",
                        p.display()
                    ));
                }
                // The ledger promises `cp objects/<sha> <path>` restores what
                // an approval blessed; until now approvals stored no object
                // at all, so an approved manifest deleted before any freeze
                // saw it - and EVERY bound script, which no freeze ever
                // reads - was unrestorable while --ledger said otherwise
                // (dogfood: an 86-second hunt for an object that could not
                // exist). Store the vouched manifest bytes, and each bound
                // file whose on-disk bytes hash to a sha the human vouched.
                store_object(&dir, vouched, &bytes)?;
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    for code in manifest_code_source(p, text, project_root) {
                        let (Some(sha), Ok(code_bytes)) = (code.sha256.as_deref(), std::fs::read(&code.path)) else {
                            continue;
                        };
                        if shas.iter().any(|s| s == sha) && sha256_hex(&code_bytes) == sha {
                            store_object(&dir, sha, &code_bytes)?;
                        }
                    }
                }
                // Every code hash this act records in `also` (that is,
                // `shas[1..]`) must now have a stored object, or a restore of
                // that hash would fail. A bound file changed, deleted, or made
                // unreadable since the card hashed it leaves its vouched sha
                // unstored: reject the whole approval rather than record a hash
                // with no restorable bytes (Greptile). Code that was already
                // missing AT card time never entered `shas`, so this does not
                // fire for it - that tool is simply not covered and asks again.
                // The manifest object already written is orphaned (no record
                // references it), never a dangling approval.
                for code_sha in shas.iter().skip(1) {
                    // The object must exist AND hash to its own name: a mere
                    // is_file() check would accept a pre-existing object at
                    // objects/<sha> that holds unrelated bytes (a changed
                    // script whose sha slot was pre-populated), so a restore
                    // would produce bytes the reviewer never approved
                    // (Greptile). store_object only writes bytes that hash to
                    // the sha, so a valid object here means we stored it this
                    // act or an earlier act stored the identical bytes.
                    let intact = std::fs::read(dir.join("objects").join(code_sha))
                        .map(|b| sha256_hex(&b) == *code_sha)
                        .unwrap_or(false);
                    if !intact {
                        return Err(format!(
                            "a bound file changed, was removed, or could not be read since it was shown, so its approved bytes ({}) are not restorable; nothing was approved - review the files and approve them again",
                            &code_sha[..code_sha.len().min(12)]
                        ));
                    }
                }
                std::str::from_utf8(&bytes)
                    .ok()
                    .and_then(|text| hook_record_source(p, text, project_root))
            }
        };
        let path = path.map(canonical_or);
        let new_hash = shas.iter().any(|sha| !known.contains(sha));
        let new_path = path.as_deref().is_some_and(|p| !known.was_live(p));
        let new_shape = path
            .as_deref()
            .is_some_and(|p| shape_of(known.approved_hook(p)) != shape_of(shape.as_ref()));
        if !new_hash && !new_path && !new_shape {
            return Ok(());
        }
        let record = Record {
            v: RECORD_VERSION,
            ts: unix_now(),
            path: path.unwrap_or_default(),
            sha256: shas.first().cloned(),
            actor: if session_id.is_some() { Actor::Session } else { Actor::External },
            session_id: session_id.map(str::to_string),
            kind: Kind::Approval,
            also: shas.iter().skip(1).cloned().collect(),
            event: shape.as_ref().map(|h| h.event.clone()),
            blocking: shape.as_ref().is_some_and(|h| h.blocking),
            code: shape.map(|h| h.code).unwrap_or_default(),
            prev: String::new(),
        };
        let (lines, head) = chain(vec![record], &verified.head)?;
        append_chained(&dir, &lines, &head)
    })
}

/// The comparable part of a remembered hook shape: what it gates on, whether
/// it gates at all, and what it runs. The path is the key, so it is not part
/// of the answer. `blocking` belongs here because retracting it is exactly how
/// a human demotes a gate, and a shape that forgot it would leave the old
/// gate's memory standing forever.
fn shape_of(hook: Option<&ApprovedHook>) -> Option<(&str, bool, &[String])> {
    hook.map(|h| (h.event.as_str(), h.blocking, h.code.as_slice()))
}

/// What `openmax --ledger-repair` did.
pub struct Repair {
    /// Where the unverifiable log (or unpinned tail) was moved, when one was.
    pub quarantined: Option<PathBuf>,
    /// Lines set aside, and how many of them were approvals.
    pub records: usize,
    pub approvals: usize,
    /// True when the only problem was a crash-interrupted pin, now re-pinned.
    pub repinned: bool,
}

/// What `--ledger-repair` would do, read-only, so a front end can show the
/// stakes before asking a human to type the word that does it.
pub enum RepairPlan {
    Nothing,
    /// An interrupted append of plain observations: re-pinning loses nothing
    /// and grants nothing.
    Repin,
    /// An interrupted append whose tail grants or retires authority: repair
    /// sets the tail aside rather than blessing bytes nobody vouched for.
    /// The records are shown so the human knows what to re-approve.
    QuarantineTail { tail: Vec<Record> },
    /// An unverifiable log: quarantine sets the whole of it aside.
    Quarantine { records: usize, approvals: usize },
}

pub fn repair_plan(data_dir: &Path, project_root: &Path) -> RepairPlan {
    let dir = project_dir(data_dir, project_root);
    match read_verified(&dir) {
        Ok(v) if v.pin == Pin::Interrupted => {
            if v.unpinned_authority().next().is_some() {
                RepairPlan::QuarantineTail { tail: v.records[v.pinned..].to_vec() }
            } else {
                RepairPlan::Repin
            }
        }
        Ok(_) => RepairPlan::Nothing,
        Err(_) => {
            let text = std::fs::read_to_string(log_path(&dir)).unwrap_or_default();
            let approvals = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str::<Record>(l).ok())
                .filter(|r| !r.kind.is_change())
                .count();
            let records = text.lines().filter(|l| !l.trim().is_empty()).count();
            RepairPlan::Quarantine { records, approvals }
        }
    }
}

/// The stated way back from an unverifiable ledger. Quarantine, never delete:
/// the damaged log is evidence, and a repair that destroys it would hide the
/// tampering it exists to reveal. Approvals live in the chain, so a
/// quarantined log takes them with it - which is the fail-closed half of the
/// deal, and why the outcome says how many a human has to grant again.
///
/// An interrupted append heals two ways: a change-only tail re-pins (records
/// are observations; nothing gains authority), while a tail that grants or
/// retires authority is quarantined back to the pinned prefix. Repair never
/// blesses: the one way authority enters this ledger is a completed, pinned
/// append from a guarded human path, so a forged tail buys its author
/// nothing but this quarantine file.
pub fn repair(data_dir: &Path, project_root: &Path) -> Result<Repair, String> {
    let dir = project_dir(data_dir, project_root);
    with_lock(&dir, || match read_verified(&dir) {
        Ok(verified) if verified.pin == Pin::Interrupted => {
            if verified.unpinned_authority().next().is_some() {
                return quarantine_tail_locked(&dir, &verified);
            }
            crate::sessions::write_atomic(&chain_head_path(&dir), &verified.head)?;
            let _ = std::fs::remove_file(pending_head_path(&dir));
            Ok(Repair { quarantined: None, records: 0, approvals: 0, repinned: true })
        }
        Ok(_) => Ok(Repair { quarantined: None, records: 0, approvals: 0, repinned: false }),
        Err(_) => {
            let log = log_path(&dir);
            // Counted without verifying: the whole point is that these lines
            // cannot be trusted, but a human still deserves the size of what
            // is being set aside.
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            let parsed: Vec<Record> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str::<Record>(l).ok())
                .collect();
            let approvals = parsed.iter().filter(|r| r.kind == Kind::Approval).count();
            let records = text.lines().filter(|l| !l.trim().is_empty()).count();
            let quarantined = dir.join(format!("log.jsonl.unverified-{}", unix_now()));
            if log.exists() {
                std::fs::rename(&log, &quarantined)
                    .map_err(|e| format!("cannot quarantine {}: {e}", log.display()))?;
            }
            let head = chain_head_path(&dir);
            if head.exists() {
                let _ = std::fs::rename(&head, dir.join("chain-head.unverified"));
            }
            let _ = std::fs::remove_file(pending_head_path(&dir));
            Ok(Repair {
                quarantined: log_exists_then(&quarantined),
                records,
                approvals,
                repinned: false,
            })
        }
    })
}

fn log_exists_then(path: &Path) -> Option<PathBuf> {
    path.exists().then(|| path.to_path_buf())
}

/// Set an unpinned authority tail aside and restore the log to its pinned
/// prefix. The stored chain head already names the prefix's final record, so
/// it stays; only the pending receipt goes. Atomic on the log: the tail file
/// is written first, so a crash between the two writes loses nothing.
fn quarantine_tail_locked(dir: &Path, verified: &Verified) -> Result<Repair, String> {
    let log = log_path(dir);
    let text = std::fs::read_to_string(&log)
        .map_err(|e| format!("cannot read {}: {e}", log.display()))?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let tail = &lines[verified.pinned.min(lines.len())..];
    let approvals = verified.unpinned_authority().count();
    let quarantined = dir.join(format!("log.jsonl.unverified-{}", unix_now()));
    let mut tail_text = tail.join("\n");
    tail_text.push('\n');
    crate::sessions::write_atomic(&quarantined, tail_text)?;
    let mut prefix = lines[..verified.pinned.min(lines.len())].join("\n");
    if !prefix.is_empty() {
        prefix.push('\n');
    }
    crate::sessions::write_atomic(&log, prefix)?;
    let _ = std::fs::remove_file(pending_head_path(dir));
    Ok(Repair {
        quarantined: Some(quarantined),
        records: tail.len(),
        approvals,
        repinned: false,
    })
}

/// Whether an object still holds the exact bytes it is named for. Rollback is
/// `cp objects/<sha> <path>`, so a rewritten object is a backdoor with a
/// documented delivery route; the write path re-verifies, and this is the
/// read path doing the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectState {
    Intact,
    Missing,
    Corrupt,
}

/// Write `bytes` as `objects/<sha>` unless a valid object is already there.
/// Never trusts a pre-existing object blindly: one that does not hash to its
/// name is replaced with the authentic bytes.
fn store_object(dir: &Path, sha: &str, bytes: &[u8]) -> Result<(), String> {
    let object = dir.join("objects").join(sha);
    let valid = std::fs::read(&object)
        .map(|existing| sha256_hex(&existing) == sha)
        .unwrap_or(false);
    if valid {
        return Ok(());
    }
    std::fs::create_dir_all(dir.join("objects"))
        .map_err(|e| format!("cannot create {}: {e}", dir.join("objects").display()))?;
    crate::sessions::write_atomic(&object, bytes)
}

pub fn object_state(data_dir: &Path, project_root: &Path, sha: &str) -> ObjectState {
    let path = project_dir(data_dir, project_root).join("objects").join(sha);
    match std::fs::read(&path) {
        Ok(bytes) if sha256_hex(&bytes) == sha => ObjectState::Intact,
        Ok(_) => ObjectState::Corrupt,
        Err(_) => ObjectState::Missing,
    }
}

/// True when an approval's PRIMARY manifest bytes cannot be restored: the
/// record approved a manifest sha but its object is missing or corrupt (a
/// legacy or hash-only approval never stored one). `--ledger` surfaces this so
/// a row with intact bound objects but an unrestorable manifest does not read
/// as fully restorable (Greptile). False for non-approvals and path-only
/// approvals, which have no manifest object to restore.
pub fn approval_manifest_missing(data_dir: &Path, project_root: &Path, record: &Record) -> bool {
    record.kind == Kind::Approval
        && record
            .sha256
            .as_deref()
            .is_some_and(|sha| !matches!(object_state(data_dir, project_root, sha), ObjectState::Intact))
}

/// The `(sha, path)` pairs `--ledger`'s footer may offer as `cp` restore
/// commands for one approval's bound code: each vouched hash in `also`,
/// paired with the project file the approved bytes named for it, intact
/// objects only.
///
/// The pairing is positional because it was recorded positionally: the
/// approval hashed the manifest's bound files in `manifest_code_source`
/// order and `approve()` wrote them to `also` in that order. A hook record
/// carries the named paths in `code`; a tool record carries none (`code` is
/// filled from the hook shape only), so its paths are re-derived by parsing
/// the STORED manifest object - the bytes the hash pins, never the live
/// file, which may be exactly what needs restoring (Greptile). A bound file
/// the card could not read never entered `also`, so a hash list shorter than
/// the named-path list is ambiguous: nothing is offered then, because a
/// guessed pairing prints a command that writes approved bytes over the
/// wrong file.
pub fn approval_restore_targets(
    data_dir: &Path,
    project_root: &Path,
    record: &Record,
) -> Vec<(String, PathBuf)> {
    if record.kind != Kind::Approval || record.also.is_empty() {
        return Vec::new();
    }
    let paths: Vec<PathBuf> = if !record.code.is_empty() && record.code.len() == record.also.len() {
        record.code.iter().map(PathBuf::from).collect()
    } else {
        let named = approved_manifest_code_paths(data_dir, project_root, record);
        if named.len() != record.also.len() {
            return Vec::new();
        }
        named
    };
    record
        .also
        .iter()
        .zip(paths)
        .filter(|(sha, _)| matches!(object_state(data_dir, project_root, sha.as_str()), ObjectState::Intact))
        .map(|(sha, path)| (sha.clone(), path))
        .collect()
}

/// The bound-code paths the approved manifest bytes name. The record's sha is
/// what authenticates the bytes, not where they live, so the stored object
/// and the file at the record's path are both accepted sources when their
/// bytes hash to it - a pruned object with the manifest still installed must
/// not hide the script restore the footer exists to print (Greptile). Empty
/// when neither source hashes to the vouched sha (an edited manifest revokes,
/// so its named paths are nobody's to offer), when nothing was ever stored
/// (a hash-only approval), or when the record carries no path to parse
/// against.
fn approved_manifest_code_paths(
    data_dir: &Path,
    project_root: &Path,
    record: &Record,
) -> Vec<PathBuf> {
    if record.path.as_os_str().is_empty() {
        return Vec::new();
    }
    let Some(sha) = record.sha256.as_deref() else {
        return Vec::new();
    };
    let object = project_dir(data_dir, project_root).join("objects").join(sha);
    let Some(bytes) = [object, record.path.clone()]
        .iter()
        .filter_map(|p| std::fs::read(p).ok())
        .find(|b| sha256_hex(b) == sha)
    else {
        return Vec::new();
    };
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return Vec::new();
    };
    manifest_code_source(&record.path, text, project_root)
        .into_iter()
        .map(|c| c.path)
        .collect()
}

/// Unix seconds as `YYYY-MM-DD HH:MM:SSZ`. Raw epoch seconds are unreadable
/// in an audit trail, and a date crate is not worth pulling in for one line.
pub fn format_ts(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    // Civil-from-days (Howard Hinnant's algorithm), epoch shifted to 0000-03-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{d:02} {h:02}:{m:02}:{s:02}Z")
}

/// The approved shape of a hook file, derived from the exact bytes being
/// approved - the caller has already proven them to be the vouched ones. A
/// file that is not a hook manifest records nothing: tools and skills have no
/// event to remember, and their code is bound by hash alone.
fn hook_record_source(path: &Path, text: &str, project_root: &Path) -> Option<ApprovedHook> {
    let hook = crate::hooks::parse_hook_source(path, text).ok()?;
    Some(ApprovedHook {
        path: canonical_or(path).display().to_string(),
        event: hook.event.as_str().to_string(),
        code: bound_code(&hook.command, &hook.args, project_root)
            .into_iter()
            .map(|c| c.path.display().to_string())
            .collect(),
        // What these bytes asked for, not what they amount to: an event that
        // always gates is remembered by its event, and this flag is the half
        // of `is_gate` that only `turn_end` can ever set.
        blocking: hook.blocking,
    })
}

/// Stop expecting a capability at `path`: the human removed it on purpose.
/// Returns whether anything was remembered there. Only the path memory is
/// dropped, never a content hash - approval binds bytes, and the same bytes
/// arriving again at any path are still bytes a human read and blessed.
/// Retirement is a chained record like the approval it ends: it removes
/// enforcement, so it carries the same authentication and shows up in
/// `openmax --ledger` with its time and actor.
pub fn forget_capability(
    data_dir: &Path,
    project_root: &Path,
    path: &Path,
) -> Result<bool, String> {
    let dir = project_dir(data_dir, project_root);
    with_lock(&dir, || {
        settle_legacy_store_locked(&dir)?;
        let verified = read_verified(&dir)?;
        refuse_unpinned_authority(&verified)?;
        let known = approvals_from(&verified.records[..verified.pinned]);
        let target = canonical_or(path);
        if !known.was_live(&target) {
            return Ok(false);
        }
        let record = Record {
            v: RECORD_VERSION,
            ts: unix_now(),
            path: target,
            sha256: None,
            actor: Actor::External,
            session_id: None,
            kind: Kind::PathRetired,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        };
        let (lines, head) = chain(vec![record], &verified.head)?;
        append_chained(&dir, &lines, &head)?;
        Ok(true)
    })
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

impl BoundCode {
    /// Why this entry is not covered, for the message a human has to act on.
    /// A file that cannot be read is a different problem from a file whose
    /// bytes moved, and telling them apart is the difference between
    /// "install it" and "re-approve it".
    pub fn problem(&self, approvals: &Approvals) -> Option<String> {
        match &self.sha256 {
            None => Some(format!("{} does not exist or cannot be read", self.path.display())),
            Some(sha) if !approvals.contains(sha) => Some(format!(
                "{} is not the content that was approved",
                self.path.display()
            )),
            Some(_) => None,
        }
    }
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
    let command = command.trim();
    match resolve_command(command, project_root) {
        // Agent-writable: the bytes decide.
        Some(path) if inside_project(&path, project_root) => out.push(read_code(path)),
        // A real file outside the project: the absolute path the human read.
        Some(path) if path.exists() => {}
        // Resolved to nothing, or did not resolve at all. Either way there is
        // no code to approve, and an empty binding must never read as "this
        // command is covered": that is how an unrecognized path spelling, or
        // a name that resolves only on some other machine, would run
        // whatever eventually lands there.
        Some(path) => out.push(BoundCode { path, sha256: None }),
        None => out.push(BoundCode { path: PathBuf::from(command), sha256: None }),
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
    // An interpreter's script argument (`/bin/sh run.sh`) that resolved to a
    // project path but is now MISSING must bind to a None entry, exactly as a
    // missing command-position script does. Otherwise deleting it leaves an
    // EMPTY binding that `covers_code` reads as "nothing to cover", so the
    // deleted tool runs ungated and a removed-tool receipt calls it
    // cardless-restorable (Greptile). An existing script was already read by
    // the arg loop above, so this only fires for a genuinely absent one.
    // For an interpreter command, a MISSING script-like positional argument
    // binds to None, even behind options (`python3 -O run.py`). This is
    // deliberately more eager than `interpreter_script` (which returns None as
    // soon as any option precedes the candidate, to avoid a --check false
    // positive over an option VALUE like `node -p x.js`): here, gating a
    // missing script-shaped argument is the safe direction - an empty binding
    // would let the removed tool run ungated and read as cardless-restorable
    // (Greptile). An existing argument was already read by the arg loop above.
    let stem = Path::new(command.trim()).file_name().and_then(|s| s.to_str());
    if stem.is_some_and(|s| INTERPRETERS.contains(&s)) {
        for arg in args {
            let arg = arg.trim();
            if arg.is_empty() || arg.starts_with('-') {
                continue;
            }
            let script_like = Path::new(arg)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| SCRIPT_EXTENSIONS.contains(&e));
            if !script_like {
                continue;
            }
            let path = absolute_from(arg, project_root);
            if inside_project(&path, project_root) && !path.is_file() {
                out.push(BoundCode { path, sha256: None });
            }
            break; // the first script-like positional is the script
        }
    }
    out
}

/// A project file that inline program text reaches for at runtime.
///
/// Binding covers the manifest and the files it *names in argv*. Program text
/// passed with `-c`/`-e` is covered only as text: the manifest hash pins the
/// program, but whatever that program opens while it runs is chosen at
/// runtime and stays agent-writable. `python3 -c "exec(open('payload.py')...)"`
/// is the shape - fully hashed, and completely unbound where it matters.
///
/// Parsing interpreter source to find those reads is unbounded and would only
/// buy false confidence, so this looks for the one signal that is cheap and
/// specific: a token inside the inline program that names a file which exists
/// in the project and is not already bound. `sh -c 'echo hi'` names none and
/// stays quiet, which is what keeps the warning worth reading.
const INTERPRETERS: [&str; 12] = [
    "sh", "bash", "zsh", "dash", "ksh", "python", "python3", "node", "ruby", "perl", "deno", "bun",
];
const INLINE_FLAGS: [&str; 4] = ["-c", "-e", "--eval", "--exec"];

/// Extensions that mark an argv token as a program file rather than a module
/// name or data argument. Deliberately narrow: `--check` warns from this, and
/// a warning about a file that never was one costs more than a miss.
const SCRIPT_EXTENSIONS: [&str; 8] = ["py", "sh", "bash", "js", "mjs", "ts", "rb", "pl"];

/// The script file an interpreter-style command will run, when its argv names
/// one: the leading positional argument, judged only when it is shaped like a
/// script file, and only when no option stands before it. Any earlier option
/// may consume or redefine the operand (`node -p result.js` evaluates the
/// text `result.js`; `sh -s x.sh` reads the program from stdin and keeps
/// `x.sh` as `$1`), and option tables differ per interpreter, so a flag
/// anywhere before the candidate means None: a warning about a file that
/// never was one costs more than a miss.
pub(crate) fn interpreter_script<'a>(command: &str, args: &'a [String]) -> Option<&'a str> {
    let stem = Path::new(command.trim()).file_name()?.to_string_lossy().to_string();
    if !INTERPRETERS.iter().any(|i| stem == *i) {
        return None;
    }
    for arg in args {
        let arg = arg.trim();
        if arg.is_empty() {
            continue;
        }
        if arg.starts_with('-') {
            return None;
        }
        return Path::new(arg)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| SCRIPT_EXTENSIONS.contains(&e))
            .then_some(arg);
    }
    None
}

pub fn inline_program_read(command: &str, args: &[String], project_root: &Path) -> Option<PathBuf> {
    let stem = Path::new(command.trim()).file_name()?.to_string_lossy().to_string();
    if !INTERPRETERS.iter().any(|i| stem == *i) {
        return None;
    }
    let bound: Vec<PathBuf> = bound_code(command, args, project_root)
        .into_iter()
        .map(|c| c.path)
        .collect();
    let mut inline = false;
    for arg in args {
        let arg = arg.trim();
        if INLINE_FLAGS.contains(&arg) {
            inline = true;
            continue;
        }
        if !inline {
            continue;
        }
        // Quotes, parentheses and separators are not part of a path; splitting
        // on them is enough to surface `open('payload.py')` or `. ./lib.sh`.
        for token in arg.split(|c: char| !(c.is_alphanumeric() || "._-/".contains(c))) {
            if token.is_empty() || !token.contains('.') {
                continue;
            }
            let path = absolute_from(token, project_root);
            if path.is_file() && inside_project(&path, project_root) && !bound.contains(&path) {
                return Some(path);
            }
        }
        inline = false;
    }
    None
}

/// The project-local code the capability manifest at `path` will execute,
/// whichever surface it belongs to. Both manifest surfaces name a `command`
/// plus fixed `args`; skills name none.
pub fn manifest_code(path: &Path, project_root: &Path) -> Vec<BoundCode> {
    match std::fs::read_to_string(path) {
        Ok(text) => manifest_code_source(path, &text, project_root),
        Err(_) => Vec::new(),
    }
}

/// The same answer from bytes a caller already read. `openmax --approve`
/// hashes a manifest and lists the code that hash obliges it to bless; both
/// must describe one read, or the interval between two reads is where a swap
/// puts the hash of one file on record next to the code list of another.
pub fn manifest_code_source(path: &Path, text: &str, project_root: &Path) -> Vec<BoundCode> {
    if let Ok(hook) = crate::hooks::parse_hook_source(path, text) {
        return bound_code(&hook.command, &hook.args, project_root);
    }
    if let Ok(spec) = crate::registry::parse_tool_source(path, text) {
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
/// None means nothing resolves, which the caller treats as uncovered, not as
/// nothing to cover.
///
/// A backslash counts as path syntax even though this harness targets unix:
/// misreading `.\payload.cmd` as a bare name would leave agent-writable code
/// with an empty binding, and no legitimate unix command name contains one.
fn resolve_command(command: &str, project_root: &Path) -> Option<PathBuf> {
    if command.is_empty() {
        return None;
    }
    if command.contains('/') || command.contains('\\') {
        return Some(absolute_from(&command.replace('\\', "/"), project_root));
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

/// One spelling per file, so two references to it compare equal. A path whose
/// file is gone cannot be canonicalized at all, and that is the case
/// reconciliation depends on, so its parent is canonicalized instead: a
/// deleted capability still has to match the entry it was approved under.
fn canonical_or(path: &Path) -> PathBuf {
    if let Ok(resolved) = path.canonicalize() {
        return resolved;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        if let Ok(parent) = parent.canonicalize() {
            return parent.join(name);
        }
    }
    lexical_abs(path)
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

    /// A same-second approve/retire/re-approve of identical bytes at the same
    /// path must give the two approvals DISTINCT event ids, or the per-session
    /// watermark drops the re-approval and the next turn is told the
    /// capability was retired when it is approved to run without a card
    /// (Greptile). The chain position disambiguates records the (path, sha,
    /// second, kind) tuple cannot, and it is stable across reads.
    #[test]
    fn same_second_re_approval_gets_a_distinct_event_id() {
        let approval = |ts: u64| Record {
            v: RECORD_VERSION,
            ts,
            path: PathBuf::from(".openmax/tools/x.toml"),
            sha256: Some("a".repeat(64)),
            actor: Actor::External,
            session_id: None,
            kind: Kind::Approval,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        };
        // Index 0: the first approval. Index 2: the re-approval after a
        // retirement at index 1, same bytes, same Unix second. Same generation.
        let first = approval_event_id(0, 0, &approval(1000));
        let reapproval = approval_event_id(0, 2, &approval(1000));
        assert_ne!(
            first, reapproval,
            "a same-second re-approval must not collide with the first approval"
        );
        // ...and the SAME record keeps the SAME id across turns (stable
        // position and generation), so a real re-scan does not resurface seen.
        assert_eq!(first, approval_event_id(0, 0, &approval(1000)));

        // An identical approval at position ZERO of a chain repaired in the
        // same second (generation 1) must NOT collide with the pre-repair one
        // (generation 0), or the watermark drops the post-repair approval and
        // keeps stale approval context (Greptile).
        let post_repair = approval_event_id(1, 0, &approval(1000));
        assert_ne!(
            first, post_repair,
            "an identical approval across a repair must get a distinct id"
        );
    }


    #[test]
    fn queued_claims_round_trip_in_order_and_skip_corruption() {
        let dir = std::env::temp_dir().join(format!("openmax-claims-{}", uuid::Uuid::new_v4()));
        let data_dir = dir.join("data");
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();

        let a: QueuedClaim = (
            vec![(project.join("a.toml"), sha256_hex(b"a"), b"a".to_vec())],
            Actor::Session,
        );
        let b: QueuedClaim = (
            vec![(project.join("b.toml"), sha256_hex(b"b"), b"b".to_vec())],
            Actor::External,
        );
        let path_a = persist_queued_claim(&data_dir, &project, &a).unwrap();
        let path_b = persist_queued_claim(&data_dir, &project, &b).unwrap();
        // Arrival order comes from the locked counter, not the clock, so
        // the second file always sorts after the first.
        assert!(path_b.file_name().unwrap() > path_a.file_name().unwrap());
        // Corruption sorts first and must be skipped in place, not deleted
        // and not allowed to block the readable claims behind it.
        let claims = path_a.parent().unwrap().to_path_buf();
        std::fs::write(claims.join("000000000000-garbage.json"), "not json").unwrap();

        let loaded = load_queued_claims(&data_dir, &project);
        assert_eq!(loaded.len(), 2, "corrupt file must not block real claims");
        assert_eq!(loaded[0].1 .1, Actor::Session);
        assert_eq!(loaded[1].1 .1, Actor::External);
        assert!(claims.join("000000000000-garbage.json").exists());

        remove_claim_file(&loaded[0].0);
        let loaded = load_queued_claims(&data_dir, &project);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].1 .1, Actor::External);
        let _ = std::fs::remove_dir_all(dir);
    }

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

    /// Narration must match enforcement: an approval record past the pinned
    /// chain head is bytes nobody vouched for, and `refuse_unpinned_authority`
    /// treats its grant as inert ("they grant nothing"). `approval_events`
    /// used to iterate the FULL record list, so the turn-start reconciliation
    /// would tell a session "approved X" for a grant the gates will not
    /// honor. The state is built from real bytes: two real approvals, then
    /// the chain head rewound to after the first with the pending head
    /// vouching the second, which is exactly the interrupted-append shape.
    #[test]
    fn an_unpinned_approval_tail_is_not_narrated() {
        let data = temp("unpinned-data");
        let root = temp("unpinned-proj");
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        let m1 = root.join(".openmax/tools/a.toml");
        let m2 = root.join(".openmax/tools/b.toml");
        std::fs::write(&m1, "name = \"a\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n")
            .unwrap();
        std::fs::write(&m2, "name = \"b\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n")
            .unwrap();
        let sha1 = sha256_hex(&std::fs::read(&m1).unwrap());
        let sha2 = sha256_hex(&std::fs::read(&m2).unwrap());
        let dir = project_dir(&data, &root);
        approve_capability(&data, &root, &m1, std::slice::from_ref(&sha1)).unwrap();
        let head1 = std::fs::read_to_string(chain_head_path(&dir)).unwrap();
        approve_capability(&data, &root, &m2, std::slice::from_ref(&sha2)).unwrap();
        let head2 = std::fs::read_to_string(chain_head_path(&dir)).unwrap();

        // Rewind: the log holds both records, the pin vouches only the first,
        // the pending head names the second (an interrupted approval write).
        std::fs::write(chain_head_path(&dir), &head1).unwrap();
        std::fs::write(pending_head_path(&dir), &head2).unwrap();

        assert!(
            !approved_hashes(&data, &root).unwrap().contains(&sha2),
            "enforcement treats the unpinned grant as inert"
        );
        let events = approval_events(&data, &root).unwrap();
        assert_eq!(
            events.len(),
            1,
            "narration matches enforcement: only the vouched approval is an event: {events:?}"
        );
        // The record holds the canonicalized path (macOS /var vs /private/var).
        assert_eq!(events[0].path, m1.canonicalize().unwrap());
    }

    /// An unverifiable chain is an error the caller must narrate, not an
    /// empty history: returning no events read as "no approval activity" in
    /// the one surface built to report approval activity, while every
    /// enforcement path was refusing loudly on the same state.
    #[test]
    fn an_unverifiable_ledger_is_an_error_not_an_empty_history() {
        let data = temp("unver-data");
        let root = temp("unver-proj");
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        let m1 = root.join(".openmax/tools/a.toml");
        std::fs::write(&m1, "name = \"a\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n")
            .unwrap();
        let sha1 = sha256_hex(&std::fs::read(&m1).unwrap());
        approve_capability(&data, &root, &m1, std::slice::from_ref(&sha1)).unwrap();
        assert_eq!(approval_events(&data, &root).unwrap().len(), 1);

        // A tail with no pending pin is tampering, not a crash.
        let dir = project_dir(&data, &root);
        let mut log = std::fs::OpenOptions::new().append(true).open(log_path(&dir)).unwrap();
        log.write_all(b"{\"forged\": true}\n").unwrap();
        drop(log);
        assert!(
            approval_events(&data, &root).is_err(),
            "an unverifiable ledger is a narratable error, never an empty history"
        );
    }

    /// An approval stores what it blessed: the manifest AND every bound
    /// file, so `cp objects/<sha> <path>` restores exactly what a human
    /// approved. Before, approvals stored nothing (only freezes did, and a
    /// freeze never reads a bound script), while --ledger promised restore.
    #[test]
    fn an_approval_stores_the_manifest_and_bound_code_as_objects() {
        let data = temp("appr-obj-data");
        let root = temp("appr-obj-proj");
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        let script = root.join("run.sh");
        std::fs::write(&script, "#!/bin/sh\necho hi\n").unwrap();
        let manifest = root.join(".openmax/tools/t.toml");
        std::fs::write(&manifest, "name = \"t\"\ndescription = \"d\"\ncommand = \"./run.sh\"\n").unwrap();
        let manifest_sha = sha256_hex(&std::fs::read(&manifest).unwrap());
        let script_sha = sha256_hex(&std::fs::read(&script).unwrap());
        approve_capability(&data, &root, &manifest, &[manifest_sha.clone(), script_sha.clone()]).unwrap();
        let objects = project_dir(&data, &root).join("objects");
        assert_eq!(
            std::fs::read(objects.join(&manifest_sha)).unwrap(),
            std::fs::read(&manifest).unwrap(),
            "the approved manifest bytes are restorable"
        );
        assert_eq!(
            std::fs::read(objects.join(&script_sha)).unwrap(),
            std::fs::read(&script).unwrap(),
            "the approved bound script bytes are restorable"
        );
        assert!(matches!(object_state(&data, &root, &script_sha), ObjectState::Intact));
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A missing interpreter-script argument (`/bin/sh run.sh`, run.sh gone)
    /// binds to a None entry, so the empty binding cannot read as covered.
    /// Otherwise deleting it left an empty `bound_code`, which `covers_code`
    /// accepts, and the removed tool was wrongly called cardless-restorable
    /// (Greptile).
    #[test]
    fn a_missing_interpreter_script_arg_binds_to_none() {
        let root = temp("interp-proj");
        let bound = bound_code("/bin/sh", &["run.sh".to_string()], &root);
        assert!(
            bound.iter().any(|c| c.sha256.is_none() && c.path.ends_with("run.sh")),
            "a missing interpreter script binds to None: {bound:?}"
        );
        assert!(
            !Approvals::default().covers_code(&bound),
            "a None binding is never covered, so the tool stays gated"
        );
        // The same holds when an OPTION precedes the script (`python3 -O
        // run.py`): interpreter_script bails on the option, but bound_code
        // still binds the missing script-like positional to None (Greptile).
        let with_opt = bound_code("python3", &["-O".to_string(), "run.py".to_string()], &root);
        assert!(
            with_opt.iter().any(|c| c.sha256.is_none() && c.path.ends_with("run.py")),
            "a missing script behind an option still binds None: {with_opt:?}"
        );
        // With the script present the arg loop reads it (Some sha), no None.
        std::fs::write(root.join("run.sh"), "echo hi\n").unwrap();
        let present = bound_code("/bin/sh", &["run.sh".to_string()], &root);
        assert!(
            present.iter().all(|c| c.sha256.is_some()),
            "an existing script is read, not left None: {present:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// If a bound script changes after the card hashed it but before `approve`
    /// runs, the approval is REJECTED: recording the vouched sha in `also`
    /// while its object cannot be stored would leave an approved hash with no
    /// restorable bytes (Greptile). Nothing is recorded.
    #[test]
    fn approve_rejects_a_bound_file_changed_after_the_card() {
        let data = temp("changed-data");
        let root = temp("changed-proj");
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        let manifest = root.join(".openmax/tools/t.toml");
        std::fs::write(&manifest, "name = \"t\"\ndescription = \"d\"\ncommand = \"./run.sh\"\n").unwrap();
        let script = root.join("run.sh");
        std::fs::write(&script, "echo A\n").unwrap();
        // The bytes the card hashed.
        let manifest_sha = sha256_hex(&std::fs::read(&manifest).unwrap());
        let script_sha = sha256_hex(&std::fs::read(&script).unwrap());
        // The script changes AFTER the card, BEFORE approve.
        std::fs::write(&script, "echo B\n").unwrap();
        let err = approve_capability(&data, &root, &manifest, &[manifest_sha, script_sha])
            .expect_err("a changed bound file must be rejected");
        assert!(err.contains("changed") && err.contains("not restorable"), "{err}");
        assert!(
            history(&data, &root).unwrap().is_empty(),
            "a rejected approval appends no record"
        );
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The object check verifies CONTENT, not just existence: a changed bound
    /// script whose sha slot in `objects/` was pre-populated with unrelated
    /// bytes must still be rejected, or a restore would produce bytes the
    /// reviewer never approved (Greptile).
    #[test]
    fn approve_rejects_a_bound_file_whose_object_slot_is_corrupt() {
        let data = temp("corrupt-data");
        let root = temp("corrupt-proj");
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        let manifest = root.join(".openmax/tools/t.toml");
        std::fs::write(&manifest, "name = \"t\"\ndescription = \"d\"\ncommand = \"./run.sh\"\n").unwrap();
        let script = root.join("run.sh");
        std::fs::write(&script, "echo A\n").unwrap();
        let manifest_sha = sha256_hex(&std::fs::read(&manifest).unwrap());
        let script_sha = sha256_hex(&std::fs::read(&script).unwrap());
        // An unrelated object is planted at objects/<script_sha>.
        let objects = project_dir(&data, &root).join("objects");
        std::fs::create_dir_all(&objects).unwrap();
        std::fs::write(objects.join(&script_sha), b"unrelated corrupt bytes").unwrap();
        // The script changes, so the store loop will not overwrite the slot.
        std::fs::write(&script, "echo B\n").unwrap();
        let err = approve_capability(&data, &root, &manifest, &[manifest_sha, script_sha])
            .expect_err("a corrupt object slot must not pass as restorable");
        assert!(err.contains("not restorable"), "{err}");
        assert!(history(&data, &root).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A hash-only approval stores no manifest object, so --ledger must read
    /// its manifest as not restorable; a full (path-form) approval stores the
    /// object and reads as restorable (Greptile).
    #[test]
    fn a_hash_only_approval_reads_as_manifest_not_restorable() {
        let data = temp("mrestore-data");
        let root = temp("mestore-proj");
        let sha = sha256_hex(b"name = \"t\"\n");
        approve_hash(&data, &root, &sha).unwrap();
        let hash_only = history(&data, &root).unwrap();
        assert!(
            approval_manifest_missing(&data, &root, &hash_only[0]),
            "a hash-only approval stored no manifest object to restore from"
        );

        // A full approval stores the manifest object.
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        let manifest = root.join(".openmax/tools/t.toml");
        std::fs::write(&manifest, "name = \"full\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n").unwrap();
        let full_sha = sha256_hex(&std::fs::read(&manifest).unwrap());
        approve_capability(&data, &root, &manifest, std::slice::from_ref(&full_sha)).unwrap();
        let recs = history(&data, &root).unwrap();
        let full = recs.iter().find(|r| r.sha256.as_deref() == Some(full_sha.as_str())).unwrap();
        assert!(
            !approval_manifest_missing(&data, &root, full),
            "a full approval stored the manifest bytes, so it is restorable"
        );
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

        // The approval is a chained record, and it shows up in the history
        // an audit reads.
        let records = history(&data, &root).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kind, Kind::Approval);
        assert_eq!(records[0].sha256.as_deref(), Some(sha.as_str()));

        // Approving the same content twice records it once.
        approve_hash(&data, &root, &sha).unwrap();
        assert_eq!(history(&data, &root).unwrap().len(), 1);

        // An unverifiable ledger approves nothing and says why.
        let dir = project_dir(&data, &root);
        std::fs::write(log_path(&dir), "{broken\n").unwrap();
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

        // `..` actually leaves the project, so what it names is not project
        // code - but a spelling that resolves to no file at all is still not
        // covered, wherever it points.
        let outside = root.parent().unwrap().join("outside-real.sh");
        std::fs::write(&outside, "#!/bin/sh\ntrue\n").unwrap();
        assert!(
            bound_code(&format!("../{}", outside.file_name().unwrap().to_string_lossy()), &[], &root)
                .is_empty(),
            "a real file outside the project is the absolute path the human read"
        );
        assert!(!Approvals::default().covers_code(&bound_code("../nothing-here.sh", &[], &root)));
        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The general form of the binding hole: an empty binding must mean "this
    /// command is a system path the human read", never "nothing resolved, so
    /// nothing to check". A name that resolves only on some other machine, or
    /// in a spelling this platform does not parse as a path, is uncovered.
    #[test]
    fn a_command_that_resolves_to_nothing_is_never_covered() {
        let root = temp("unresolved-proj");
        let empty = Approvals::default();

        for command in [
            "definitely-not-a-real-binary-xyz",  // not on PATH anywhere
            "./missing.sh",                      // project path, no file
            "/opt/nowhere/missing.sh",           // absolute path, no file
            ".\\payload.cmd",                    // backslash spelling
            "subdir\\payload.sh",
        ] {
            let bound = bound_code(command, &[], &root);
            assert_eq!(bound.len(), 1, "{command} must bind something to refuse");
            assert!(bound[0].sha256.is_none(), "{command}");
            assert!(!empty.covers_code(&bound), "{command} must not read as covered");
        }

        // A backslash path that does exist is bound by its bytes, not waved
        // through: that is the case the whole check exists for.
        std::fs::create_dir_all(root.join("subdir")).unwrap();
        std::fs::write(root.join("subdir/payload.sh"), "#!/bin/sh\ntrue\n").unwrap();
        let bound = bound_code("subdir\\payload.sh", &[], &root);
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].sha256, Some(sha256_hex(b"#!/bin/sh\ntrue\n")));

        // A genuine system binary is still covered by the manifest alone.
        assert!(bound_code("/bin/echo", &[], &root).is_empty());
        assert!(bound_code("sh", &[], &root).is_empty());
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
        let body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n";
        std::fs::write(&manifest, body).unwrap();
        // gate.sh's bytes are b"script", so its sha is the second vouched hash
        // and its object is stored - the approval records only restorable bytes.
        std::fs::write(root.join("gate.sh"), "script").unwrap();
        let shas = vec![sha256_hex(body.as_bytes()), sha256_hex(b"script")];
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

    /// One approval act is one chained, auditable record: the manifest, the
    /// code it runs, and the path a human granted it at, all in the line the
    /// chain covers.
    #[test]
    fn an_approval_act_is_one_chained_record_carrying_every_hash() {
        let data = temp("act-data");
        let root = temp("act-proj");
        let manifest = root.join("gate.toml");
        let body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n";
        std::fs::write(&manifest, body).unwrap();
        std::fs::write(root.join("gate.sh"), "script").unwrap();
        let shas = vec![sha256_hex(body.as_bytes()), sha256_hex(b"script")];
        approve_capability(&data, &root, &manifest, &shas).unwrap();

        let records = history(&data, &root).unwrap();
        assert_eq!(records.len(), 1, "one act, one record");
        assert_eq!(records[0].kind, Kind::Approval);
        assert_eq!(records[0].sha256.as_deref(), Some(shas[0].as_str()));
        assert_eq!(records[0].also, vec![shas[1].clone()], "the bound code rides the same act");
        assert_eq!(records[0].path, canonical_or(&manifest));

        // Nothing new means no record, so re-approving unchanged content does
        // not grow the log.
        approve_capability(&data, &root, &manifest, &shas).unwrap();
        assert_eq!(history(&data, &root).unwrap().len(), 1);

        // A rewritten script is a new hash, so it takes a new act - and the
        // chain records that too.
        std::fs::write(root.join("gate.sh"), "rewritten script").unwrap();
        let rewritten = vec![shas[0].clone(), sha256_hex(b"rewritten script")];
        approve_capability(&data, &root, &manifest, &rewritten).unwrap();
        let records = history(&data, &root).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].prev, sha256_hex(first_line(&data, &root).as_bytes()));
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An approved external tool records its bound script's hash in `also`
    /// but no path list - `code` is filled from the hook shape only - so a
    /// footer pairing `also` with `code` offered nothing: the one intact
    /// object an operator needs after deleting the script had no cp line
    /// (Greptile). The path comes back from the stored manifest object, the
    /// approved bytes, never the live file, which here is already gone.
    #[test]
    fn an_external_tools_bound_script_gets_a_restore_target() {
        let data = temp("tool-restore-data");
        let root = temp("tool-restore-proj");
        let manifest = root.join("deploy.toml");
        let body = "name = \"deploy\"\ndescription = \"d\"\ncommand = \"./deploy.sh\"\n";
        std::fs::write(&manifest, body).unwrap();
        std::fs::write(root.join("deploy.sh"), "script").unwrap();
        let bound = manifest_code(&manifest, &root);
        assert_eq!(bound.len(), 1, "the fixture binds exactly the script");
        let script_path = bound[0].path.clone();
        let shas = vec![sha256_hex(body.as_bytes()), sha256_hex(b"script")];
        approve_capability(&data, &root, &manifest, &shas).unwrap();
        std::fs::remove_file(root.join("deploy.sh")).unwrap();

        let records = history(&data, &root).unwrap();
        assert!(records[0].code.is_empty(), "a tool record carries no path list");
        assert_eq!(
            approval_restore_targets(&data, &root, &records[0]),
            vec![(sha256_hex(b"script"), script_path)],
            "the intact script object is offered at the path the approved bytes name"
        );
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The record's sha authenticates the manifest bytes wherever they live:
    /// with the stored object pruned but the manifest file still hashing to
    /// the vouched sha, a tool's intact script object keeps its restore line
    /// (Greptile), while an EDITED manifest file authenticates nothing.
    #[test]
    fn a_tools_restore_survives_a_pruned_manifest_object_via_the_authentic_file() {
        let data = temp("tool-prune-data");
        let root = temp("tool-prune-proj");
        let manifest = root.join("deploy.toml");
        let body = "name = \"deploy\"\ndescription = \"d\"\ncommand = \"./deploy.sh\"\n";
        std::fs::write(&manifest, body).unwrap();
        std::fs::write(root.join("deploy.sh"), "script").unwrap();
        let script_path = manifest_code(&manifest, &root)[0].path.clone();
        let shas = vec![sha256_hex(body.as_bytes()), sha256_hex(b"script")];
        approve_capability(&data, &root, &manifest, &shas).unwrap();
        std::fs::remove_file(root.join("deploy.sh")).unwrap();
        std::fs::remove_file(project_dir(&data, &root).join("objects").join(&shas[0])).unwrap();

        let records = history(&data, &root).unwrap();
        assert_eq!(
            approval_restore_targets(&data, &root, &records[0]),
            vec![(sha256_hex(b"script"), script_path)],
            "the on-disk manifest still hashes to the vouched sha, so it names the path"
        );

        std::fs::write(&manifest, "name = \"deploy\"\ndescription = \"d\"\ncommand = \"./other.sh\"\n")
            .unwrap();
        assert_eq!(
            approval_restore_targets(&data, &root, &records[0]),
            Vec::<(String, PathBuf)>::new(),
            "an edited manifest authenticates nothing and offers nothing"
        );
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A bound file the card could not read never enters `also`, so its
    /// record's hash and path lists disagree in length. Positional pairing
    /// would then print a cp that writes one file's approved bytes over
    /// another file's path - the helper refuses instead.
    #[test]
    fn an_unequal_hash_and_path_list_is_refused_not_guess_paired() {
        let data = temp("gap-data");
        let root = temp("gap-proj");
        let manifest = root.join("gate.toml");
        let body = "event = \"pre_tool_use\"\ncommand = \"./wrap.sh\"\nargs = [\"helper.py\"]\n";
        std::fs::write(&manifest, body).unwrap();
        std::fs::write(root.join("wrap.sh"), "wrap").unwrap();
        std::fs::write(root.join("helper.py"), "helper").unwrap();
        // The card skips a file it cannot read; only helper.py's hash rides.
        let shas = vec![sha256_hex(body.as_bytes()), sha256_hex(b"helper")];
        approve_capability(&data, &root, &manifest, &shas).unwrap();

        let records = history(&data, &root).unwrap();
        assert_eq!(records[0].code.len(), 2, "the approved bytes name two files");
        assert_eq!(records[0].also.len(), 1, "only one hash was vouched");
        assert_eq!(
            approval_restore_targets(&data, &root, &records[0]),
            Vec::<(String, PathBuf)>::new(),
            "an ambiguous pairing offers nothing rather than the wrong path"
        );
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A hook record carries its named paths, so its bound-code restore
    /// survives a manifest object that is itself missing - the record is the
    /// pairing, no parse needed - while a bound object that is not intact is
    /// never offered.
    #[test]
    fn a_hooks_recorded_paths_pair_without_the_manifest_object() {
        let data = temp("hook-restore-data");
        let root = temp("hook-restore-proj");
        let manifest = root.join("gate.toml");
        let body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n";
        std::fs::write(&manifest, body).unwrap();
        std::fs::write(root.join("gate.sh"), "script").unwrap();
        let shas = vec![sha256_hex(body.as_bytes()), sha256_hex(b"script")];
        approve_capability(&data, &root, &manifest, &shas).unwrap();
        let records = history(&data, &root).unwrap();
        let objects = project_dir(&data, &root).join("objects");
        std::fs::remove_file(objects.join(&shas[0])).unwrap();

        assert_eq!(
            approval_restore_targets(&data, &root, &records[0]),
            vec![(sha256_hex(b"script"), PathBuf::from(records[0].code[0].clone()))],
            "the record's own path list pairs without the manifest object"
        );
        std::fs::remove_file(objects.join(&shas[1])).unwrap();
        assert_eq!(
            approval_restore_targets(&data, &root, &records[0]),
            Vec::<(String, PathBuf)>::new(),
            "a bound object that is not intact is never offered"
        );
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    fn first_line(data: &Path, root: &Path) -> String {
        let text = std::fs::read_to_string(log_path(&project_dir(data, root))).unwrap();
        text.lines().next().unwrap().to_string()
    }

    /// The record has to carry what the approved bytes *were*, not just their
    /// hash: reconciliation asks whether a modified hook used to gate, and the
    /// file on disk is the part an edit controls. Storage that forgets this
    /// reopens the demotion bypass with nothing else failing, so it is
    /// asserted here at the record, not only through the hooks that read it.
    #[test]
    fn an_approved_hooks_shape_rides_the_chain_and_ends_with_it() {
        let data = temp("shape-data");
        let root = temp("shape-proj");
        std::fs::write(root.join("gate.sh"), "#!/bin/sh\nexit 1\n").unwrap();
        let manifest = root.join("gate.toml");
        let gate_body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n";
        std::fs::write(&manifest, gate_body).unwrap();
        approve_capability(&data, &root, &manifest, &[sha256_hex(gate_body.as_bytes())]).unwrap();

        let records = history(&data, &root).unwrap();
        assert_eq!(records[0].event.as_deref(), Some("pre_tool_use"), "the shape is in the line");
        assert_eq!(records[0].code.len(), 1, "and the code those bytes named");
        let approved = approvals(&data, &root).unwrap();
        let hook = approved.approved_hook(&manifest).expect("the shape reads back");
        assert!(hook.is_gate());
        assert_eq!(hook.event(), "pre_tool_use");
        assert_eq!(hook.code_paths().count(), 1);

        // Demotion cannot rewrite the memory: the file now says observer, the
        // chain still says gate.
        let observer_body = "event = \"post_tool_use\"\ncommand = \"./gate.sh\"\n";
        std::fs::write(&manifest, observer_body).unwrap();
        assert!(approvals(&data, &root).unwrap().approved_hook(&manifest).unwrap().is_gate());

        // A human who means it re-approves, and the new shape replaces the old
        // one rather than stacking behind it.
        approve_capability(&data, &root, &manifest, &[sha256_hex(observer_body.as_bytes())]).unwrap();
        let approved = approvals(&data, &root).unwrap();
        assert_eq!(approved.approved_hook(&manifest).unwrap().event(), "post_tool_use");
        assert!(!approved.approved_hook(&manifest).unwrap().is_gate());

        // Re-approving the same shape and hashes again records nothing.
        let before = history(&data, &root).unwrap().len();
        approve_capability(&data, &root, &manifest, &[sha256_hex(observer_body.as_bytes())]).unwrap();
        assert_eq!(history(&data, &root).unwrap().len(), before);

        // Retiring the path retires the shape with it: a path nobody expects
        // has no gate to have been demoted.
        assert!(forget_capability(&data, &root, &manifest).unwrap());
        let approved = approvals(&data, &root).unwrap();
        assert!(approved.approved_hook(&manifest).is_none());
        assert!(!approved.was_live(&manifest));
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The proven exploit: one unattended session, no human, the agent writes
    /// a hook and appends its sha to the approvals store through `bash`.
    /// Approvals live in the chain now, so a file dropped beside it is not an
    /// approval - it is set aside, and the hook stays inert.
    #[test]
    fn a_forged_approvals_file_approves_nothing() {
        let data = temp("forge-data");
        let root = temp("forge-proj");
        let files = vec![entry(&root, ".openmax/tools/a.toml", "v1")];
        sync(&data, &root, &files, Actor::External, None).unwrap();

        let payload = sha256_hex(b"event = \"session_start\"\ncommand = \"/bin/sh\"\n");
        let hook = root.join(".openmax/hooks/payload.toml");
        let dir = project_dir(&data, &root);
        // Both shapes of the old store, including the path set that decides
        // whether a gate counts as installed.
        std::fs::write(
            legacy_approved_path(&dir),
            serde_json::json!({
                "version": 1,
                "hashes": [payload],
                "paths": [hook.display().to_string()],
            })
            .to_string(),
        )
        .unwrap();
        assert!(!is_approved(&data, &root, &payload), "a forged store must approve nothing");
        assert!(
            !legacy_approved_path(&dir).exists(),
            "the forgery is set aside, not left to be re-read"
        );
        let approved = approvals(&data, &root).unwrap();
        assert!(approved.hashes.is_empty());
        assert!(!approved.was_live(&hook), "a forged path claim is not an installed gate");

        // Forging the record itself has to forge the chain, which is what the
        // pin and the chain catch: appended without either, it does not read.
        let head = std::fs::read_to_string(chain_head_path(&dir)).unwrap();
        let record = Record {
            v: RECORD_VERSION,
            ts: 1,
            path: PathBuf::new(),
            sha256: Some(payload.clone()),
            actor: Actor::External,
            session_id: None,
            kind: Kind::Approval,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: head.trim().to_string(),
        };
        let line = serde_json::to_string(&record).unwrap();
        let mut log = std::fs::OpenOptions::new().append(true).open(log_path(&dir)).unwrap();
        log.write_all(format!("{line}\n").as_bytes()).unwrap();
        drop(log);
        let err = history(&data, &root).unwrap_err();
        assert!(err.contains("chain head"), "{err}");
        assert!(!is_approved(&data, &root, &payload), "an unpinned append is not an approval");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The approval act is an open interval: the caller hashes the manifest,
    /// a human vouches for it, and only then is the record written - and any
    /// process with `bash` can write the file in between. The shape the
    /// record remembers (the event, the code list the repair carve-out
    /// honors) must come from the vouched bytes, never from a fresh read of
    /// the path: a manifest swapped inside the interval would otherwise put a
    /// gate's hash on file with an observer's shape - the demotion bypass
    /// through the approval itself.
    #[test]
    fn an_approval_refuses_a_manifest_that_changed_after_the_vouch() {
        let data = temp("vouch-data");
        let root = temp("vouch-proj");
        let hooks_dir = root.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(root.join("gate.sh"), "#!/bin/sh\nexit 1\n").unwrap();
        let manifest = hooks_dir.join("gate.toml");
        let gate_body = "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n";
        std::fs::write(&manifest, gate_body).unwrap();

        // The hashes exactly as `openmax --approve` computes them: the
        // manifest's bytes, plus the code those bytes name.
        let mut shas = vec![sha256_hex(gate_body.as_bytes())];
        shas.extend(manifest_code(&manifest, &root).into_iter().filter_map(|c| c.sha256));
        assert_eq!(shas.len(), 2, "the gate script must be part of the act");

        // Swapped inside the interval: same path, observer shape.
        std::fs::write(&manifest, "event = \"post_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
        let err = approve_capability(&data, &root, &manifest, &shas).unwrap_err();
        assert!(err.contains("changed after it was shown"), "{err}");
        assert!(history(&data, &root).unwrap().is_empty(), "a refused act records nothing");

        // A deleted manifest is the same problem: no bytes on disk are the
        // bytes vouched for, so there is nothing the record can describe.
        std::fs::remove_file(&manifest).unwrap();
        assert!(approve_capability(&data, &root, &manifest, &shas).is_err());
        assert!(history(&data, &root).unwrap().is_empty());

        // Restoring the vouched bytes lets the same act land, with the shape
        // those bytes declare.
        std::fs::write(&manifest, gate_body).unwrap();
        approve_capability(&data, &root, &manifest, &shas).unwrap();
        let approved = approvals(&data, &root).unwrap();
        let hook = approved.approved_hook(&manifest).expect("the shape is remembered");
        assert!(hook.is_gate(), "the remembered shape is the vouched bytes' shape");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Every record written before `blocking` existed describes a `turn_end`
    /// hook a human approved as an observer, because that is all a `turn_end`
    /// hook could be. Reading one back as a gate would hand an installed base
    /// of observers the power to end turns, on nobody's say-so but the
    /// upgrade's.
    #[test]
    fn an_approved_turn_end_observer_never_becomes_a_gate_by_upgrade() {
        let legacy: ApprovedHook = serde_json::from_str(
            r#"{"path":"/p/.openmax/hooks/watch.toml","event":"turn_end","code":["/p/watch.sh"]}"#,
        )
        .expect("a record from before the field still parses");
        assert!(!legacy.is_gate(), "an upgrade must not promote what a human approved");
        assert!(!legacy.blocking());
        assert_eq!(legacy.event(), "turn_end");
        assert_eq!(legacy.code_paths().count(), 1, "the rest of the shape survives");
    }

    /// The other half of the same default. A `pre_tool_use` record from the
    /// same era carries no flag either, and reading the flag *instead of* the
    /// event would demote every gate a human installed before this build.
    #[test]
    fn a_pre_tool_use_record_written_before_the_blocking_field_still_gates() {
        for event in ["pre_tool_use", "user_prompt_submit"] {
            let legacy: ApprovedHook = serde_json::from_str(&format!(
                r#"{{"path":"/p/.openmax/hooks/gate.toml","event":"{event}"}}"#
            ))
            .expect("a record from before the field still parses");
            assert!(legacy.is_gate(), "{event} gates on its event, flag or no flag");
            assert!(!legacy.blocking(), "{event} never asked for the flag");
        }
    }

    /// The approval act is an open interval (see the vouched-bytes test
    /// above), and `blocking` is the part of a hook's shape that decides
    /// whether it can end a turn. So it is derived from the bytes proven to
    /// hash to what a human vouched for, and a manifest swapped inside the
    /// interval records nothing at all.
    #[test]
    fn approving_a_blocking_turn_end_records_the_blocking_shape_from_the_vouched_bytes() {
        let data = temp("blocking-vouch-data");
        let root = temp("blocking-vouch-proj");
        let hooks_dir = root.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(root.join("verify.sh"), "#!/bin/sh\nexit 1\n").unwrap();
        let manifest = hooks_dir.join("verify.toml");
        let gate_body = "event = \"turn_end\"\nblocking = true\ncommand = \"./verify.sh\"\n";
        std::fs::write(&manifest, gate_body).unwrap();
        let mut shas = vec![sha256_hex(gate_body.as_bytes())];
        shas.extend(manifest_code(&manifest, &root).into_iter().filter_map(|c| c.sha256));

        // Swapped inside the interval: the same hashes, one word short.
        std::fs::write(&manifest, "event = \"turn_end\"\ncommand = \"./verify.sh\"\n").unwrap();
        let err = approve_capability(&data, &root, &manifest, &shas).unwrap_err();
        assert!(err.contains("changed after it was shown"), "{err}");
        assert!(history(&data, &root).unwrap().is_empty(), "a refused act records nothing");

        // The vouched bytes back on disk, and the shape they declare is what
        // the chain carries.
        std::fs::write(&manifest, gate_body).unwrap();
        approve_capability(&data, &root, &manifest, &shas).unwrap();
        let records = history(&data, &root).unwrap();
        assert!(records[0].blocking, "the flag rides the record");
        assert_eq!(records[0].event.as_deref(), Some("turn_end"));
        let hook = approvals(&data, &root)
            .unwrap()
            .approved_hook(&manifest)
            .expect("the shape reads back")
            .clone();
        assert!(hook.blocking() && hook.is_gate());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// `blocking` is the whole difference between a hook that can end a turn
    /// and one that watches, so two shapes that differ only in it are two
    /// different shapes. Leaving it out of the comparison would make
    /// re-approving a gate's exact former bytes a no-op - nothing new to
    /// record - and the observer shape recorded in between would stand as the
    /// approved one forever, with the file on disk saying otherwise.
    #[test]
    fn changing_only_the_blocking_flag_is_a_new_shape() {
        let data = temp("blocking-shape-data");
        let root = temp("blocking-shape-proj");
        let hooks_dir = root.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        std::fs::write(root.join("verify.sh"), "#!/bin/sh\nexit 1\n").unwrap();
        let manifest = hooks_dir.join("verify.toml");
        let gate_body = "event = \"turn_end\"\nblocking = true\ncommand = \"./verify.sh\"\n";
        let watch_body = "event = \"turn_end\"\ncommand = \"./verify.sh\"\n";
        let approve = |body: &str| {
            std::fs::write(&manifest, body).unwrap();
            let mut shas = vec![sha256_hex(body.as_bytes())];
            shas.extend(manifest_code(&manifest, &root).into_iter().filter_map(|c| c.sha256));
            approve_capability(&data, &root, &manifest, &shas).unwrap();
        };
        let is_gate = || {
            approvals(&data, &root).unwrap().approved_hook(&manifest).unwrap().is_gate()
        };

        approve(gate_body);
        assert!(is_gate(), "the approved bytes asked to gate");

        // A human demotes it on purpose.
        approve(watch_body);
        assert!(!is_gate(), "and un-asked it on purpose");

        // Then changes their mind. The hashes are already blessed and the
        // event and code never moved, so `blocking` is the only thing that
        // makes this act new - and if it is not new, nothing is recorded and
        // the gate never comes back.
        let before = history(&data, &root).unwrap().len();
        approve(gate_body);
        assert_eq!(history(&data, &root).unwrap().len(), before + 1, "the shape moved");
        assert!(is_gate(), "re-approving the gate's own bytes restores the gate");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Every way of deleting evidence, one line each. `rm chain-head` and
    /// `rm log.jsonl` used to read as a clean history and as a fresh project.
    #[test]
    fn every_tamper_shape_is_detected_and_first_run_still_works() {
        type Tamper = fn(&Path);
        let cases: [(&str, Tamper); 5] = [
            ("truncate", |dir| truncate_log(dir, 2)),
            ("truncate + rm chain-head", |dir| {
                truncate_log(dir, 2);
                std::fs::remove_file(chain_head_path(dir)).unwrap();
            }),
            ("rm log.jsonl", |dir| std::fs::remove_file(log_path(dir)).unwrap()),
            ("empty log.jsonl", |dir| std::fs::write(log_path(dir), "").unwrap()),
            ("rm chain-head", |dir| std::fs::remove_file(chain_head_path(dir)).unwrap()),
        ];
        for (name, tamper) in cases {
            let data = temp("tamper-data");
            let root = temp("tamper-proj");
            for v in ["v1", "v2", "v3"] {
                let files = vec![entry(&root, ".openmax/tools/a.toml", v)];
                sync(&data, &root, &files, Actor::External, None).unwrap();
            }
            let dir = project_dir(&data, &root);
            tamper(&dir);
            let err = history(&data, &root).unwrap_err();
            assert!(err.contains("--ledger-repair"), "{name}: no repair path in: {err}");
            assert!(
                sync(&data, &root, &[], Actor::External, None).is_err(),
                "{name}: an unverifiable ledger must not append"
            );
            let _ = std::fs::remove_dir_all(&data);
            let _ = std::fs::remove_dir_all(&root);
        }

        // A project nobody has ever run is not tampering.
        let data = temp("fresh-data");
        let root = temp("fresh-proj");
        assert!(history(&data, &root).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    fn truncate_log(dir: &Path, drop_last: usize) {
        let text = std::fs::read_to_string(log_path(dir)).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let keep: String =
            lines[..lines.len() - drop_last].iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(log_path(dir), keep).unwrap();
    }

    /// A crash between the append and the pin is not tampering, and saying so
    /// with certainty is how a power loss used to leave the ledger accusing
    /// its user and refusing to write forever.
    #[test]
    fn an_interrupted_append_reads_as_recoverable_and_heals() {
        let data = temp("crash-data");
        let root = temp("crash-proj");
        let files = vec![entry(&root, ".openmax/tools/a.toml", "v1")];
        sync(&data, &root, &files, Actor::External, None).unwrap();
        let dir = project_dir(&data, &root);

        // Exactly what a SIGKILL after the log write leaves behind: the
        // pending pin named the head, the log has it, chain-head does not.
        let text = std::fs::read_to_string(log_path(&dir)).unwrap();
        let last = text.lines().next_back().unwrap();
        let record = Record {
            v: RECORD_VERSION,
            ts: 2,
            path: root.join(".openmax/tools/a.toml"),
            sha256: Some(sha256_hex(b"v2")),
            actor: Actor::External,
            session_id: None,
            kind: Kind::Change,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: sha256_hex(last.as_bytes()),
        };
        let line = serde_json::to_string(&record).unwrap();
        let new_head = sha256_hex(line.as_bytes());
        let mut log = std::fs::OpenOptions::new().append(true).open(log_path(&dir)).unwrap();
        log.write_all(format!("{line}\n").as_bytes()).unwrap();
        drop(log);
        std::fs::write(pending_head_path(&dir), &new_head).unwrap();

        let state = read(&data, &root).unwrap();
        assert_eq!(state.records.len(), 2, "nothing was removed, so nothing is hidden");
        assert!(state.interrupted_write, "the pin is behind the log, not the log behind the pin");

        // The next sync re-pins, even with nothing new to record.
        sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v2")], Actor::External, None)
            .unwrap();
        assert_eq!(std::fs::read_to_string(chain_head_path(&dir)).unwrap(), new_head);
        assert!(!pending_head_path(&dir).exists());
        assert!(!read(&data, &root).unwrap().interrupted_write);

        // The same tail without a pending pin is a forged append, not a
        // crash: tolerating one must not launder the other.
        let text = std::fs::read_to_string(log_path(&dir)).unwrap();
        let record = Record {
            prev: sha256_hex(text.lines().next_back().unwrap().as_bytes()),
            ts: 3,
            ..record
        };
        let line = serde_json::to_string(&record).unwrap();
        let mut log = std::fs::OpenOptions::new().append(true).open(log_path(&dir)).unwrap();
        log.write_all(format!("{line}\n").as_bytes()).unwrap();
        drop(log);
        assert!(history(&data, &root).is_err(), "an append with no pending pin is tampering");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A break must be recoverable by a stated command, not by folklore about
    /// deleting the ledger directory.
    #[test]
    fn repair_quarantines_the_damage_and_restores_writes() {
        let data = temp("repair-data");
        let root = temp("repair-proj");
        sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v1")], Actor::External, None)
            .unwrap();
        approve_hash(&data, &root, &sha256_hex(b"v1")).unwrap();
        let dir = project_dir(&data, &root);
        truncate_log(&dir, 1);

        let outcome = repair(&data, &root).unwrap();
        let quarantined = outcome.quarantined.expect("the damaged log is kept as evidence");
        assert!(quarantined.exists());
        assert_eq!(outcome.records, 1);
        assert!(!log_path(&dir).exists());
        assert!(!chain_head_path(&dir).exists());
        assert!(dir.join("objects").is_dir(), "rollback bytes survive a repair");

        // Writes work again, and the quarantined approvals really are gone.
        assert!(!is_approved(&data, &root, &sha256_hex(b"v1")));
        let changes =
            sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v1")], Actor::Session, None)
                .unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].actor, Actor::Initial, "a repaired ledger starts a new baseline");
        assert!(history(&data, &root).is_ok());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Approval's honest edge: a program handed to an interpreter on the
    /// command line is bound as text, and whatever that text opens at runtime
    /// is not. The signal has to be quiet on the common fully-bound shapes or
    /// nobody will read it.
    #[test]
    fn inline_program_reads_are_flagged_only_when_they_name_a_project_file() {
        let root = temp("inline-proj");
        std::fs::write(root.join("payload.py"), "print(1)\n").unwrap();
        std::fs::write(root.join("lib.sh"), "true\n").unwrap();

        let found = inline_program_read(
            "/usr/bin/python3",
            &["-c".into(), "exec(open('payload.py').read())".into()],
            &root,
        );
        assert_eq!(found, Some(absolute_from("payload.py", &root)));
        assert!(inline_program_read("/bin/sh", &["-c".into(), ". ./lib.sh".into()], &root).is_some());

        // Fully bound: the whole program is in the manifest a human read.
        assert!(inline_program_read("/bin/sh", &["-c".into(), "echo hi".into()], &root).is_none());
        // Names a file that does not exist here: nothing to warn about yet.
        assert!(inline_program_read(
            "/bin/sh",
            &["-c".into(), "cat /etc/hosts && echo done.txt".into()],
            &root
        )
        .is_none());
        // Already bound by argv, which is the shape the warning asks for.
        assert!(inline_program_read("/usr/bin/python3", &["payload.py".into()], &root).is_none());
        // Not an interpreter, and no inline flag.
        assert!(inline_program_read("./payload.py", &[], &root).is_none());
        assert!(inline_program_read("/bin/echo", &["payload.py".into()], &root).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A legacy store is a file with no evidence of its own, so nothing in it
    /// runs until a human says so. Until then it may only *restrict*: the
    /// paths it claims were live keep a pre-upgrade gate failing closed, while
    /// its hashes and hook shapes - the parts that would grant execution or
    /// relax a gate - stay inert.
    #[test]
    fn a_legacy_store_grants_nothing_until_a_human_adopts_it() {
        let data = temp("adopt-data");
        let root = temp("adopt-proj");
        let dir = project_dir(&data, &root);
        let sha = sha256_hex(b"old hook");

        // A ledger written before approvals joined the chain: v1 records.
        std::fs::create_dir_all(&dir).unwrap();
        let old = Record {
            v: 1,
            ts: 1,
            path: root.join(".openmax/tools/a.toml"),
            sha256: Some(sha256_hex(b"v1")),
            actor: Actor::Initial,
            session_id: None,
            kind: Kind::Change,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        };
        let line = serde_json::to_string(&old).unwrap();
        std::fs::write(log_path(&dir), format!("{line}\n")).unwrap();
        std::fs::write(chain_head_path(&dir), sha256_hex(line.as_bytes())).unwrap();
        let gate = root.join("gate.toml");
        std::fs::write(&gate, "event = \"pre_tool_use\"\ncommand = \"/bin/true\"\n").unwrap();
        let observer = root.join("watch.toml");
        std::fs::write(&observer, "event = \"post_tool_use\"\ncommand = \"/bin/true\"\n").unwrap();
        let store = serde_json::json!({
            "version": 1,
            "hashes": [sha],
            // A store old enough to remember no shape (the released one)
            // alongside one written after shapes existed.
            "paths": [
                canonical_or(&gate).display().to_string(),
                canonical_or(&observer).display().to_string(),
            ],
            "hooks": [{
                "path": canonical_or(&observer).display().to_string(),
                "event": "post_tool_use",
                "code": [],
            }],
        })
        .to_string();
        std::fs::write(legacy_approved_path(&dir), &store).unwrap();

        // Unadopted: no hash it lists is approved, and no shape it remembers
        // can turn a would-be gate into an observer.
        let waiting = approvals(&data, &root).unwrap();
        assert!(!waiting.contains(&sha), "an unauthenticated file approves nothing");
        assert!(waiting.approved_hook(&observer).is_none(), "and relaxes nothing");
        // But a path it claims was live still fails closed, which is the only
        // direction an unvouched-for file is allowed to move the answer.
        assert!(waiting.was_live(&gate));
        assert!(legacy_approved_path(&dir).exists(), "it waits rather than being consumed");
        assert!(history(&data, &root).unwrap().len() == 1, "and nothing was written for it");

        let pending = pending_legacy(&data, &root).expect("a human is told what is waiting");
        assert_eq!((pending.hashes, pending.paths.len(), pending.shapes), (1, 2, 1));
        assert!(!pending.malformed);

        // The human acts, vouching for the bytes they were shown. Now every
        // shape is inherited, as `initial`.
        let adopted = adopt_legacy_approvals(&data, &root, &pending.sha256).unwrap();
        assert_eq!((adopted.hashes, adopted.paths, adopted.shapes), (1, 2, 1));
        let inherited = approvals(&data, &root).unwrap();
        assert!(inherited.contains(&sha), "adoption keeps the approvals it inherited");
        assert!(inherited.was_live(&gate), "and the paths that say a gate was installed");
        // A remembered shape is inherited exactly: an observer a human really
        // installed must not come back as a gate that was demoted.
        let watched = inherited.approved_hook(&observer).expect("the shape is inherited");
        assert_eq!(watched.event(), "post_tool_use");
        assert!(!watched.is_gate());
        // A path whose shape was never recorded stays unremembered, which
        // reconciliation reads as a gate: the safe answer to a question the
        // old store cannot answer.
        assert!(inherited.approved_hook(&gate).is_none());
        assert!(!legacy_approved_path(&dir).exists(), "the adopted file is removed");
        let records = history(&data, &root).unwrap();
        assert_eq!(records[1].kind, Kind::Approval);
        assert_eq!(records[1].actor, Actor::Initial, "inherited provenance is unknowable");
        assert_eq!(records.last().unwrap().kind, Kind::ApprovalsImported);

        // Recreating it afterwards is not an inheritance: the chain settled
        // the question, so the file is set aside unread and cannot be adopted.
        let forged = sha256_hex(b"forged hook");
        std::fs::write(
            legacy_approved_path(&dir),
            serde_json::json!({ "version": 1, "hashes": [forged] }).to_string(),
        )
        .unwrap();
        assert!(!is_approved(&data, &root, &forged));
        assert!(is_approved(&data, &root, &sha), "the real inheritance survives");
        assert!(pending_legacy(&data, &root).is_none());
        assert!(adopt_legacy_approvals(&data, &root, &pending.sha256).is_err());
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The confirmation prompt is an open interval any process with `bash`
    /// can write across. A store rewritten inside it is a different store:
    /// the human's say-so binds to the bytes they were shown, so the
    /// substitution imports nothing and the window stays open for a fresh
    /// look at what is actually on disk.
    #[test]
    fn adoption_refuses_a_store_rewritten_after_the_preview() {
        let data = temp("swap-data");
        let root = temp("swap-proj");
        let dir = project_dir(&data, &root);
        let honest = sha256_hex(b"the tool the human remembers");
        let smuggled = sha256_hex(b"event = \"session_start\"\ncommand = \"/bin/sh\"\n");

        std::fs::create_dir_all(&dir).unwrap();
        let old = Record {
            v: 1,
            ts: 1,
            path: root.join(".openmax/tools/a.toml"),
            sha256: Some(sha256_hex(b"v1")),
            actor: Actor::Initial,
            session_id: None,
            kind: Kind::Change,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        };
        let line = serde_json::to_string(&old).unwrap();
        std::fs::write(log_path(&dir), format!("{line}\n")).unwrap();
        std::fs::write(chain_head_path(&dir), sha256_hex(line.as_bytes())).unwrap();
        std::fs::write(
            legacy_approved_path(&dir),
            serde_json::json!({ "version": 1, "hashes": [honest] }).to_string(),
        )
        .unwrap();

        // The human previews the honest store...
        let pending = pending_legacy(&data, &root).unwrap();
        assert_eq!(pending.hashes, 1);
        // ...and while they read the prompt, the file is swapped.
        std::fs::write(
            legacy_approved_path(&dir),
            serde_json::json!({ "version": 1, "hashes": [smuggled] }).to_string(),
        )
        .unwrap();

        let err = adopt_legacy_approvals(&data, &root, &pending.sha256).unwrap_err();
        assert!(err.contains("changed after it was shown"), "the refusal names the swap: {err}");
        assert!(!is_approved(&data, &root, &smuggled), "the substituted hash gained nothing");
        assert!(!is_approved(&data, &root, &honest));
        assert_eq!(history(&data, &root).unwrap().len(), 1, "nothing was chained");
        assert!(legacy_approved_path(&dir).exists(), "the file stays for a fresh preview");

        // A fresh preview of what is actually there can still be adopted: the
        // gate is the pairing of eyes and bytes, not a lockout.
        let second = pending_legacy(&data, &root).expect("the window is still open");
        assert_ne!(second.sha256, pending.sha256);
        let adopted = adopt_legacy_approvals(&data, &root, &second.sha256).unwrap();
        assert_eq!(adopted.hashes, 1);
        assert!(is_approved(&data, &root, &smuggled), "adopted only once a human saw it");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The window closes on first contact when there is nothing to inherit, so
    /// a store planted afterwards is not even offered to a human.
    #[test]
    fn a_store_planted_after_the_chain_settles_is_never_offered() {
        let data = temp("seal-data");
        let root = temp("seal-proj");
        sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v1")], Actor::External, None)
            .unwrap();
        assert!(approvals(&data, &root).unwrap().hashes.is_empty());
        let dir = project_dir(&data, &root);
        // Nothing to settle: this chain was never anything but a chain, so a
        // file beside it is a plant however early it arrives.
        assert!(!inheritable(&history(&data, &root).unwrap()));

        let payload = sha256_hex(b"event = \"session_start\"\ncommand = \"/bin/sh\"\n");
        let planted = serde_json::json!({ "version": 1, "hashes": [payload] }).to_string();
        std::fs::write(legacy_approved_path(&dir), &planted).unwrap();
        assert!(pending_legacy(&data, &root).is_none(), "nothing to ask a human about");
        assert!(!is_approved(&data, &root, &payload));
        // Even vouching for the planted bytes themselves cannot force it in:
        // the chain settled the question before the file existed.
        assert!(adopt_legacy_approvals(&data, &root, &sha256_hex(planted.as_bytes())).is_err());
        assert!(!legacy_approved_path(&dir).exists(), "set aside, not read");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn objects_are_verified_on_read_and_timestamps_are_readable() {
        let data = temp("objstate-data");
        let root = temp("objstate-proj");
        let (path, sha, bytes) = entry(&root, ".openmax/tools/a.toml", "authentic");
        sync(&data, &root, &[(path, sha.clone(), bytes)], Actor::External, None).unwrap();
        assert_eq!(object_state(&data, &root, &sha), ObjectState::Intact);

        let object = project_dir(&data, &root).join("objects").join(&sha);
        std::fs::write(&object, "### backdoor ###").unwrap();
        assert_eq!(
            object_state(&data, &root, &sha),
            ObjectState::Corrupt,
            "an object that does not hash to its name must never be restored"
        );
        std::fs::remove_file(&object).unwrap();
        assert_eq!(object_state(&data, &root, &sha), ObjectState::Missing);

        assert_eq!(format_ts(0), "1970-01-01 00:00:00Z");
        assert_eq!(format_ts(1_785_471_295), "2026-07-31 04:14:55Z");
        assert_eq!(format_ts(951_782_400), "2000-02-29 00:00:00Z", "leap day");
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

    /// Append one correctly chained record to the log and vouch for it with a
    /// pending receipt: the exact bytes `append_chained` leaves when it dies
    /// between the log write and the pin move - and the exact bytes a forger
    /// writes on purpose, because they are the cheapest spelling of a planted
    /// approval.
    fn plant_pending_tail(dir: &Path, record: Record) -> String {
        let text = std::fs::read_to_string(log_path(dir)).unwrap();
        let last = text.lines().next_back().unwrap();
        let record = Record { prev: sha256_hex(last.as_bytes()), ..record };
        let line = serde_json::to_string(&record).unwrap();
        let head = sha256_hex(line.as_bytes());
        let mut log = std::fs::OpenOptions::new().append(true).open(log_path(dir)).unwrap();
        log.write_all(format!("{line}\n").as_bytes()).unwrap();
        drop(log);
        std::fs::write(pending_head_path(dir), &head).unwrap();
        head
    }

    /// A forged approval in a pending tail must stay inert: reads honor only
    /// the pinned prefix, every append refuses to move the pin past it, and
    /// repair sets it aside instead of blessing it.
    #[test]
    fn a_pending_tail_grants_no_authority() {
        let data = temp("tail-data");
        let root = temp("tail-proj");
        sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v1")], Actor::External, None)
            .unwrap();
        approve_hash(&data, &root, &sha256_hex(b"blessed")).unwrap();
        let dir = project_dir(&data, &root);

        plant_pending_tail(&dir, Record {
            v: RECORD_VERSION,
            ts: 9,
            path: root.join(".openmax/hooks/evil.toml"),
            sha256: Some(sha256_hex(b"evil")),
            actor: Actor::External,
            session_id: None,
            kind: Kind::Approval,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        });

        // Inert on read: the real approval stands, the forged one does not.
        let approved = approvals(&data, &root).unwrap();
        assert!(approved.contains(&sha256_hex(b"blessed")));
        assert!(!approved.contains(&sha256_hex(b"evil")), "an unpinned approval grants nothing");

        // Refused on write: nothing may move the pin past the forgery.
        let next = [entry(&root, ".openmax/tools/a.toml", "v2")];
        let err = sync(&data, &root, &next, Actor::External, None).unwrap_err();
        assert!(err.contains("approval-grade"), "{err}");
        let err = approve_hash(&data, &root, &sha256_hex(b"more")).unwrap_err();
        assert!(err.contains("approval-grade"), "{err}");

        // Repair quarantines the tail instead of blessing it.
        let outcome = repair(&data, &root).unwrap();
        assert!(outcome.quarantined.is_some(), "the tail is evidence, not history");
        assert_eq!(outcome.approvals, 1);
        assert!(!outcome.repinned);
        let approved = approvals(&data, &root).unwrap();
        assert!(approved.contains(&sha256_hex(b"blessed")), "pinned history survives repair");
        assert!(!approved.contains(&sha256_hex(b"evil")));
        // And the ledger appends again.
        sync(&data, &root, &next, Actor::External, None).unwrap();
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A change-only pending tail is a crashed sync, and crash recovery must
    /// keep working exactly as before: reads see the records, the next sync
    /// re-pins, nothing needs a human.
    #[test]
    fn a_change_only_pending_tail_still_heals_itself() {
        let data = temp("healtail-data");
        let root = temp("healtail-proj");
        sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v1")], Actor::External, None)
            .unwrap();
        let dir = project_dir(&data, &root);
        let head = plant_pending_tail(&dir, Record {
            v: RECORD_VERSION,
            ts: 2,
            path: root.join(".openmax/tools/a.toml"),
            sha256: Some(sha256_hex(b"v2")),
            actor: Actor::External,
            session_id: None,
            kind: Kind::Change,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        });
        assert!(read(&data, &root).unwrap().interrupted_write);
        sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v2")], Actor::External, None)
            .unwrap();
        assert_eq!(std::fs::read_to_string(chain_head_path(&dir)).unwrap(), head);
        assert!(!read(&data, &root).unwrap().interrupted_write);
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A rewritten log carrying a pending receipt for its own new end is not
    /// an interrupted append: the stored head names no record in it. That
    /// distinction keeps the crash-recovery door from accepting a wholesale
    /// forgery.
    #[test]
    fn a_rewritten_log_with_a_pending_receipt_reads_as_tamper() {
        let data = temp("rewrite-data");
        let root = temp("rewrite-proj");
        sync(&data, &root, &[entry(&root, ".openmax/tools/a.toml", "v1")], Actor::External, None)
            .unwrap();
        let dir = project_dir(&data, &root);

        let record = Record {
            v: RECORD_VERSION,
            ts: 1,
            path: root.join(".openmax/hooks/evil.toml"),
            sha256: Some(sha256_hex(b"evil")),
            actor: Actor::External,
            session_id: None,
            kind: Kind::Approval,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        };
        let line = serde_json::to_string(&record).unwrap();
        std::fs::write(log_path(&dir), format!("{line}\n")).unwrap();
        std::fs::write(pending_head_path(&dir), sha256_hex(line.as_bytes())).unwrap();

        let err = approvals(&data, &root).unwrap_err();
        assert!(err.contains("rewritten"), "{err}");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An upgraded project with no legacy file seals the import window at
    /// first contact: an `approved.json` planted afterwards is set aside
    /// unread, not imported as an heirloom.
    #[test]
    fn first_contact_seals_the_legacy_import_window() {
        let data = temp("seal-data");
        let root = temp("seal-proj");
        let dir = project_dir(&data, &root);
        std::fs::create_dir_all(&dir).unwrap();

        // A ledger written before approvals moved into the chain: v1 records,
        // no import marker.
        let record = Record {
            v: 1,
            ts: 1,
            path: root.join(".openmax/tools/a.toml"),
            sha256: Some(sha256_hex(b"v1")),
            actor: Actor::External,
            session_id: None,
            kind: Kind::Change,
            also: Vec::new(),
            event: None,
            code: Vec::new(),
            blocking: false,
            prev: String::new(),
        };
        let line = serde_json::to_string(&record).unwrap();
        std::fs::write(log_path(&dir), format!("{line}\n")).unwrap();
        std::fs::write(chain_head_path(&dir), sha256_hex(line.as_bytes())).unwrap();

        // Reading changes nothing - a read must stay a read - but the first
        // turn after the upgrade settles it, because a turn is the only thing
        // that could have planted a store in the first place.
        let _ = approvals(&data, &root).unwrap();
        assert!(!sealed(&history(&data, &root).unwrap()), "a read must not write");
        sync(&data, &root, &[], Actor::External, None).unwrap();
        assert!(
            sealed(&history(&data, &root).unwrap()),
            "the first sync after the upgrade must settle the question"
        );

        let planted = format!(
            r#"{{"version":1,"hashes":["{}"],"paths":[]}}"#,
            sha256_hex(b"planted")
        );
        std::fs::write(legacy_approved_path(&dir), planted).unwrap();
        let approved = approvals(&data, &root).unwrap();
        assert!(
            !approved.contains(&sha256_hex(b"planted")),
            "a planted legacy file walked into the chain"
        );
        assert!(!legacy_approved_path(&dir).exists(), "the planted file must be set aside");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Retiring a path is a chained record with the same authentication as
    /// the approval it ends: the hash stays blessed, the path memory ends,
    /// and the act is auditable in history.
    #[test]
    fn forget_appends_a_retirement_record() {
        let data = temp("forget-data");
        let root = temp("forget-proj");
        std::fs::create_dir_all(root.join(".openmax/hooks")).unwrap();
        let gate = root.join(".openmax/hooks/gate.toml");
        std::fs::write(&gate, "event = \"pre_tool_use\"\ncommand = \"/usr/bin/true\"\n").unwrap();
        let sha = sha256_hex(&std::fs::read(&gate).unwrap());
        approve_capability(&data, &root, &gate, std::slice::from_ref(&sha)).unwrap();
        assert!(approvals(&data, &root).unwrap().was_live(&gate));

        std::fs::remove_file(&gate).unwrap();
        assert!(forget_capability(&data, &root, &gate).unwrap());
        let approved = approvals(&data, &root).unwrap();
        assert!(!approved.was_live(&gate), "the path memory must end");
        assert!(approved.contains(&sha), "the content approval must survive");
        assert!(!forget_capability(&data, &root, &gate).unwrap(), "nothing left to retire");
        assert!(
            history(&data, &root).unwrap().iter().any(|r| r.kind == Kind::PathRetired),
            "retirement must be auditable"
        );
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
    }
}
