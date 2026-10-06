//! The on-disk session store: transcripts, manifests, compaction digests,
//! archives, and token accounting, plus the index that lists them.
//!
//! Committed sidecars - compaction digests, the archive, and usage - are
//! append-only JSONL, so a record there keeps its line number for the life of
//! the file. That stability is what lets `recall` cite `path:line` and have
//! the address still resolve later.
//!
//! The live transcript is weaker on purpose and the difference matters. It
//! appends new tail lines when it can, but a prune that trims or drops
//! messages rewrites the whole file, so line numbers in `*.messages.json`
//! survive appends and not compaction. The index and manifest are small JSON
//! documents replaced atomically rather than appended at all. Nothing is lost
//! when the transcript is rewritten: whatever a prune removed was already
//! appended to the archive, which is append-only.
//!
//! A compaction commits its writes in the one order that keeps history whole
//! at every cut (`commit_compaction`). Other writes are best-effort and loud:
//! a failed append surfaces as an agent
//! warning rather than aborting a turn, since losing accounting must not cost
//! the user their work. The one thing that is not best-effort is ordering
//! against deletion. Every writer re-checks under `sessions_lock` that the
//! session is still indexed, so a session deleted mid-turn cannot be
//! resurrected by an append that was already in flight. The answer comes from
//! what this process knows (`StoreMemo`), not from reading the index on every
//! write.
//!
//! The index itself is shared wider than one process: it is one file per data
//! dir, and parallel openmax processes are normal usage. Its read-modify-write
//! therefore also holds an exclusive flock (`index.lock`), and a damaged index
//! is refused rather than defaulted to empty - either failure mode would end
//! with sessions silently dropped from the index. No lookup finds such a
//! session again, and the still-indexed gate silently drops the transcript
//! writes of any claim that did not see it indexed.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
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

/// The lock file lives as long as its session: removing the session unlinks
/// it, while still holding it, under the store lock. Every claim opens and
/// locks the file under that same lock. A claim that could open the file
/// before the unlink and lock it after the remover let go would own an inode
/// no other claim can reach, while the next claim created and locked a fresh
/// file under the same name: two owners of one session.
///
/// An older release takes the same flock(2) (see `lock_index`) but claims
/// without the store lock and never unlinks. The unlink comes only after the
/// session has left the index, under the index lock, so for as long as a
/// session is indexed every binary locks the one file its name holds. An
/// older binary that races a removal can still lock the unlinked file, but
/// the session it then owns is no longer indexed, and every release drops
/// writes to such a session.
pub(crate) struct SessionOwner {
    _file: FileLock,
    detach: bool,
}

/// Claim a writable attachment before reading its transcript or running hooks.
/// Repeated calls from this core share ownership, including between turns.
pub fn attach(core: &Core, id: &str) -> Result<(), String> {
    claim_session(core, id, true, true)
}

/// A finishing turn may still persist after its frontend detached. Such a
/// write retains ownership without cancelling the requested cleanup.
pub(crate) fn ensure_owned(core: &Core, id: &str) -> Result<(), String> {
    claim_session(core, id, true, false)
}

fn claim_session(core: &Core, id: &str, validate: bool, reactivate: bool) -> Result<(), String> {
    let mut owners = core.session_owners.lock().unwrap();
    if let Some(owner) = owners.get_mut(id) {
        if reactivate {
            revalidate_if_changed(core, id)?;
            owner.detach = false;
        }
        return Ok(());
    }
    // See `SessionOwner`: the open and the lock go under the store lock.
    let (mut memo, _flock) = lock_store(core)?;
    let path = lock_path(core, id);
    let file = std::fs::OpenOptions::new().create(true).write(true).truncate(false)
        .open(&path).map_err(|e| format!("cannot open session lock {}: {e}", path.display()))?;
    let file = FileLock::try_take(file).map_err(|e| match e {
        std::fs::TryLockError::WouldBlock =>
            format!("session {id} is already open in another process; close it there or start a new session"),
        std::fs::TryLockError::Error(e) => format!("cannot lock session {id}: {e}"),
    })?;
    if validate {
        let seen = stamp(&messages_path(core, id));
        load_messages_locked(core, id)?;
        remember_transcript(&mut memo, id, seen);
    }
    // The one index read the session's writes cost while this claim lasts.
    if matches!(read_index(core), IndexRead::Loaded(metas) if metas.iter().any(|m| m.id == id)) {
        memo.listed.insert(id.to_string());
    }
    owners.insert(id.to_string(), SessionOwner { _file: file, detach: false });
    Ok(())
}

/// What this process knows about the session store, so that a steady-state
/// write reads nothing back to learn it. The index lists every session of
/// every project and is never pruned: a model request or a save that read it
/// would cost more with every session ever recorded. Guarded by
/// `Core::sessions_lock`, the lock every session write and removal holds, so
/// a write never acts on an answer a removal has already changed.
#[derive(Default)]
pub(crate) struct StoreMemo {
    /// Sessions this process holds and saw indexed when it claimed them. No
    /// other process can remove one meanwhile, since every removal claims the
    /// session first, so only this process's own removal or its release of
    /// the claim can make an entry stale, and both take the entry out first.
    listed: HashSet<String>,
    /// Sessions this process removed. A write still in flight when its
    /// session was removed (cancellation is cooperative) is dropped on this
    /// record alone.
    removed: HashSet<String>,
    /// Each owned transcript as this process last validated or wrote it.
    transcripts: HashMap<String, Stamp>,
}

/// A file as a stat sees it, its length and modification time. Every write
/// moves one of them, short of an edit that keeps the length within one tick
/// of the file system's clock.
type Stamp = (u64, SystemTime);

/// None when the file is missing or cannot be stat'ed, which matches no
/// remembered stamp, so whatever is there gets read.
fn stamp(path: &Path) -> Option<Stamp> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.len(), meta.modified().ok()?))
}

/// Record the transcript this process just validated or wrote, as stamped
/// before it was read or after it was written.
fn remember_transcript(memo: &mut StoreMemo, id: &str, seen: Option<Stamp>) {
    match seen {
        Some(seen) => memo.transcripts.insert(id.to_string(), seen),
        None => memo.transcripts.remove(id),
    };
}

/// Validate an owned transcript again only when it changed since this
/// process last validated or wrote it. Every turn start claims its session
/// again, and reparsing the whole history there made each turn cost more
/// than the one before. Between turns an owned transcript changes only when
/// something outside openmax writes it, which moves its stamp, so damage
/// written then is still refused before the turn runs.
fn revalidate_if_changed(core: &Core, id: &str) -> Result<(), String> {
    let (mut memo, _flock) = lock_store(core)?;
    let seen = stamp(&messages_path(core, id));
    if seen.is_some() && memo.transcripts.get(id) == seen.as_ref() {
        return Ok(());
    }
    load_messages_locked(core, id)?;
    remember_transcript(&mut memo, id, seen);
    Ok(())
}

/// Release idle state, or defer release until an in-flight turn has settled.
/// The caller may switch frontends immediately after requesting cancellation.
pub fn detach(core: &Core, id: &str) -> Result<(), String> {
    let mut state = core.sessions.try_lock().map_err(|_| "session state is busy; try again")?;
    let running = core.running.lock().unwrap();
    let mut owners = core.session_owners.lock().unwrap();
    if running.contains(id) {
        if let Some(owner) = owners.get_mut(id) { owner.detach = true; }
    } else {
        state.remove(id);
        release(core, &mut owners, id);
    }
    Ok(())
}

/// Called under both the session map and running locks, after turn cleanup.
pub(crate) fn finish_detach(core: &Core, id: &str, state: &mut HashMap<String, crate::state::SessionData>) {
    let mut owners = core.session_owners.lock().unwrap();
    if owners.get(id).is_some_and(|owner| owner.detach) {
        state.remove(id);
        release(core, &mut owners, id);
    }
}

/// Let go of a claim. What it let this process know goes first: once the
/// flock is released another process can claim and remove the session, and
/// from then on only the index can say whether it still exists.
fn release(core: &Core, owners: &mut HashMap<String, SessionOwner>, id: &str) {
    let mut memo = core.sessions_lock.lock().unwrap();
    memo.listed.remove(id);
    memo.transcripts.remove(id);
    drop(memo);
    owners.remove(id);
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

fn lock_path(core: &Core, id: &str) -> PathBuf {
    sessions_dir(core).join(format!("{id}.lock"))
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
#[cfg(test)]
pub fn append_compaction(core: &Core, id: &str, record: &CompactionRecord) {
    let memo = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(&memo, core, id) {
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
    if let Err(message) = ensure_owned(core, id) {
        core.send_agent(id, AgentEvent::Error { message });
        return;
    }
    let memo = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(&memo, core, id) {
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
#[cfg(test)]
fn load_compaction(core: &Core, id: &str) -> Vec<CompactionRecord> {
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
/// address. A failure warns and returns false; compaction must retain the
/// original transcript until this preservation step succeeds.
#[cfg(test)]
pub fn append_archive(core: &Core, id: &str, messages: &[ChatMessage]) -> bool {
    if messages.is_empty() {
        return true;
    }
    let memo = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(&memo, core, id) {
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
    if let Err(message) = ensure_owned(core, id) {
        core.send_agent(id, AgentEvent::Error { message });
        return;
    }
    let Ok(json) = serde_json::to_string_pretty(manifest) else {
        return;
    };
    // The last of the five session files to take the rule. A refreeze can be
    // in flight when the session is deleted, and cancellation is cooperative,
    // so without this the manifest outlives everything it described.
    let memo = core.sessions_lock.lock().unwrap();
    if !still_indexed_locked(&memo, core, id) {
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
    load_index_checked(core).err()
}

/// A frontend's refusal to start, continue, list, discard, or delete a
/// session, pointing at `--check` when the refusal is the index's damage.
/// Only frontends add the pointer, never the shared reason: `--check` gives
/// the repair with the step that makes it safe (close every openmax), while
/// a bare path acted on under a running session leaves that session listed
/// nowhere: its later saves land in history no lookup finds, or are dropped
/// without a word. Any other refusal (a lock or write failure) passes
/// through as is.
pub fn refusal_with_repair(core: &Core, reason: String) -> String {
    match read_index(core) {
        IndexRead::Damaged(damage) if damage == reason => {
            format!("{reason}; run openmax --check for the repair")
        }
        _ => reason,
    }
}

/// The index under `data_dir` and the reason, when it exists but cannot be
/// read. For `--check`, which validates without a Core, so this takes no
/// lock and creates no directory; the index is only ever replaced by an
/// atomic rename, so an unlocked read sees one whole version or the other.
pub(crate) fn index_damage(data_dir: &Path) -> Option<(PathBuf, String)> {
    let path = data_dir.join("sessions").join("index.json");
    match read_index_at(&path) {
        IndexRead::Damaged(reason) => Some((path, reason)),
        IndexRead::Missing | IndexRead::Loaded(_) => None,
    }
}

/// The three states an index read can land in. A missing file is a normal
/// empty store. Unreadable and unparseable are not: they are evidence of
/// history, and writers must refuse to replace that evidence with the empty
/// default (see `with_index`).
enum IndexRead {
    Missing,
    Loaded(Vec<SessionMeta>),
    Damaged(String),
}

fn read_index(core: &Core) -> IndexRead {
    read_index_at(&index_path(core))
}

/// The reason names the file and the problem, never a repair. It also
/// reaches live sessions (the save warning of one claimed over the damage,
/// the compaction refusal, recall), and moving the index aside under one
/// leaves its session listed nowhere. A session claimed while the index was
/// healthy keeps saving history that `--continue`, `/resume`, and `--recall`
/// never find; one claimed over the damage, or claimed again later, has
/// every later save dropped without a word. The repair is `--check`'s to
/// give (see `doctor::check_at`).
fn read_index_at(path: &Path) -> IndexRead {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return IndexRead::Missing,
        Err(e) => {
            return IndexRead::Damaged(format!(
                "session index {} is unreadable ({e})",
                path.display()
            ))
        }
    };
    #[cfg(test)]
    INDEX_BYTES_READ.with(|read| read.set(read.get() + text.len()));
    match serde_json::from_str(&text) {
        Ok(metas) => IndexRead::Loaded(metas),
        Err(e) => IndexRead::Damaged(format!(
            "session index {} does not parse ({e})",
            path.display()
        )),
    }
}

fn load_index(core: &Core) -> Vec<SessionMeta> {
    load_index_checked(core).unwrap_or_default()
}

/// The index for a caller that reports what it finds to the user. A damaged
/// index is an error naming the file, never an empty list: "no previous
/// session" over a damaged index sends the user off to start fresh with
/// nothing saying where their history went.
fn load_index_checked(core: &Core) -> Result<Vec<SessionMeta>, String> {
    let _store = lock_store(core)?;
    match read_index(core) {
        IndexRead::Loaded(metas) => Ok(metas),
        IndexRead::Missing => Ok(Vec::new()),
        IndexRead::Damaged(reason) => Err(reason),
    }
}

/// Whether the session still exists, i.e. whether writing a sidecar for it is
/// still meaningful: Ok(false) once it is gone, Err when only a damaged index
/// could say.
///
/// Every sidecar here is opened with `create`, so a write that lands after
/// `delete` recreates the file it just removed. Cancellation narrows that
/// window but cannot close it: a request already on the wire settles when it
/// settles, and its usage record arrives afterwards. Answering no makes the
/// write a no-op instead, which is what "deleted" has to mean if it is to
/// mean anything.
///
/// `memo` answers for every session this process holds or removed, so a
/// steady-state write reads nothing. The index answers only for the rest,
/// such as a session claimed while the index was damaged or did not list it.
/// Callers must already hold `sessions_lock` and pass what it guards: an
/// unlocked check is a time-of-check/time-of-use bug, because `delete` can
/// remove the entry and the file between the check passing and the write
/// landing, which recreates exactly the file that was deleted.
fn indexed_locked(memo: &StoreMemo, core: &Core, id: &str) -> Result<bool, String> {
    if memo.removed.contains(id) {
        return Ok(false);
    }
    if memo.listed.contains(id) {
        return Ok(true);
    }
    match read_index(core) {
        IndexRead::Loaded(metas) => Ok(metas.iter().any(|m| m.id == id)),
        IndexRead::Missing => Ok(false),
        IndexRead::Damaged(reason) => Err(reason),
    }
}

fn still_indexed_locked(memo: &StoreMemo, core: &Core, id: &str) -> bool {
    matches!(indexed_locked(memo, core, id), Ok(true))
}

fn save_index(core: &Core, metas: &[SessionMeta]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(metas).map_err(|e| e.to_string())?;
    write_atomic(&index_path(core), json)
}

/// Serialize index read-modify-writes across processes. `sessions_lock` only
/// covers turns within one process; the index is one file per data dir, and
/// two openmax processes are normal usage. Without this, their load -> save
/// cycles interleave, the loser's `create` entry vanishes from the index, and
/// with it the session: no lookup finds it again, and the still-indexed gate
/// silently drops the writes, transcript included, of any claim that did not
/// see it indexed. Same flock discipline as the ledger and trust stores.
/// std's `File::lock` and `try_lock` are flock(2) on Linux and macOS, the
/// lock every released binary takes; where std has no file lock (Android,
/// DragonFly) they return Unsupported, so every store refuses to write rather
/// than write unlocked.
///
/// Callers must already hold `sessions_lock`: flock is per open file
/// description, so that is what keeps one process from contending with
/// itself (see the ledger's `with_lock` note). The lock releases when the
/// returned handle drops.
fn lock_index(core: &Core) -> Result<FileLock, String> {
    let dir = sessions_dir(core);
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("index.lock");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    FileLock::wait(file)
        .map_err(|e| format!("cannot lock {}: {e}", path.display()))
}

/// An exclusive lock held on an open lock file, released by an explicit
/// unlock when dropped, before the file closes. flock(2) belongs to the open
/// file description, not to one descriptor, and a child being spawned holds
/// a copy of every descriptor this process has open until it execs (bash,
/// tools, and hooks all spawn that way, see `configure_process_group`).
/// Closing only this copy would keep the lock held until that child exec'd:
/// a session released and reopened in the window is refused as open in
/// another process, and other processes waiting on the lock stall. The
/// unlock releases it whatever copies remain. The lock taken is unchanged,
/// so older binaries still exclude this one and are excluded by it.
pub(crate) struct FileLock(std::fs::File);

impl FileLock {
    /// Wait until the lock is free, then take it.
    pub(crate) fn wait(file: std::fs::File) -> std::io::Result<Self> {
        file.lock()?;
        Ok(Self(file))
    }

    /// Take the lock, or fail at once while another holder has it.
    pub(crate) fn try_take(file: std::fs::File) -> Result<Self, std::fs::TryLockError> {
        file.try_lock()?;
        Ok(Self(file))
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Should the unlock fail, closing the file still releases the lock
        // once no spawning child holds a copy.
        let _ = self.0.unlock();
    }
}

/// Take a lock file the way every released binary does: an exclusive flock(2)
/// on its own open file description, without blocking. `None` while another
/// holder has it. Tests use it to stand in for an older binary sharing the
/// data dir, because std's `File::lock` must keep excluding that binary. It
/// is released like the harness's own locks, so a child another test is
/// spawning cannot hold it past the drop.
#[cfg(all(test, unix))]
pub(crate) fn raw_flock(path: &std::path::Path) -> Option<FileLock> {
    use std::os::unix::io::AsRawFd;
    let file = std::fs::OpenOptions::new().create(true).write(true).truncate(false)
        .open(path).unwrap();
    let held = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0;
    held.then_some(FileLock(file))
}

/// Read-modify-write the index under the state lock (concurrent turns in
/// this process) and the index flock (concurrent processes). A damaged index
/// is refused, not defaulted: saving the empty fallback over it would erase
/// every session's metadata, hiding all of them from every lookup and closing
/// the still-indexed gate to every later claim of one, converting one bad
/// read into permanent, silent data loss.
fn with_index<R>(core: &Core, f: impl FnOnce(&mut Vec<SessionMeta>) -> R) -> Result<R, String> {
    let _store = lock_store(core)?;
    let mut metas = match read_index(core) {
        IndexRead::Loaded(metas) => metas,
        IndexRead::Missing => Vec::new(),
        IndexRead::Damaged(reason) => return Err(reason),
    };
    let result = f(&mut metas);
    save_index(core, &metas)?;
    Ok(result)
}

fn lock_store(core: &Core) -> Result<(std::sync::MutexGuard<'_, StoreMemo>, FileLock), String> {
    let guard = core.sessions_lock.lock().unwrap();
    let flock = lock_index(core)?;
    Ok((guard, flock))
}

/// Move replay boundaries with a prune that kept the first `head` messages
/// and wrote its digest note at index `head`. Boundaries inside the pinned
/// head are untouched, since nothing before them changed. A shrink collapses
/// the rest onto `head + 1`, after the note, never on it, or the divider
/// would render above the context note instead of below it; a note insert
/// moves them up by what it added.
fn shifted_resume_points(points: &[u64], before: usize, after: usize, head: usize) -> Vec<u64> {
    let head = head as u64;
    let mut shifted: Vec<u64> = points.iter().map(|&p| {
        if after < before && p >= head {
            p.saturating_sub((before - after) as u64).max(head + 1)
        } else if after > before && p >= head {
            p.saturating_add((after - before) as u64)
        } else { p }
    }).collect();
    shifted.sort_unstable();
    shifted.dedup();
    shifted
}

#[cfg(test)]
thread_local! {
    static FAIL_COMPACTION_INDEX_WRITE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_COMPACTION_TRANSCRIPT_WRITE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Every path this thread synced to the device, in order.
    static SYNCED: std::cell::RefCell<Vec<PathBuf>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The OS error every directory sync on this thread fails with, if any.
    static FAIL_DIR_SYNC: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
    /// Bytes this thread read from session indexes.
    static INDEX_BYTES_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Transcripts this thread read whole to load or validate them.
    static TRANSCRIPT_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Commit archive, digest record, replay boundaries, and transcript under one
/// store lock, in the order that keeps history whole at every cut. The archive
/// gains its lines before the transcript loses them, and the transcript is
/// replaced atomically, so neither a failure nor a crash can drop a message
/// that is not already on disk. Every failure leaves the live transcript and
/// its boundaries as they were: an append or the index write fails before the
/// transcript is touched, and a transcript rewrite that fails after the index
/// moved puts the boundaries back. What a failed attempt can leave behind is
/// an archive line and a digest record for a prune that then did not happen,
/// duplicates of what the transcript still holds, which the readers tolerate
/// (`--recall` deduplicates, `last_compaction` takes the newest), and a torn
/// line the next append cuts (`append_text`). Both appends are synced, with
/// the directory that names their files, before the rewrite, so they reach
/// the disk ahead of it: otherwise a power cut could keep the rewrite and
/// lose the archive lines it relies on, or the archive itself when this
/// compaction created it. `head` is how many leading messages the prune
/// kept, which is also where its digest note sits.
pub(crate) fn commit_compaction(
    core: &Core,
    id: &str,
    messages: &[ChatMessage],
    archive: &[ChatMessage],
    record: Option<&CompactionRecord>,
    before_len: usize,
    head: usize,
) -> Result<(), String> {
    ensure_owned(core, id)?;
    let (mut memo, _flock) = lock_store(core)?;
    let mut metas = match read_index(core) {
        IndexRead::Loaded(metas) => metas,
        _ => return Err("compaction requires a readable session index".into()),
    };
    let pos = metas.iter().position(|m| m.id == id).ok_or("compaction session was deleted")?;
    load_messages_locked(core, id)?;
    if !archive.is_empty() {
        append_jsonl(&archive_path(core, id), archive, true)?;
    }
    if let Some(record) = record {
        let mut line = serde_json::to_string(record).map_err(|e| e.to_string())?;
        line.push('\n');
        append_text(&compaction_path(core, id), &line, true)?;
    }
    let before = metas[pos].resume_points.clone();
    metas[pos].resume_points = shifted_resume_points(&before, before_len, messages.len(), head);
    #[cfg(test)]
    if FAIL_COMPACTION_INDEX_WRITE.with(|fail| fail.replace(false)) {
        return Err("injected compaction index failure".into());
    }
    save_index(core, &metas)?;
    #[cfg(test)]
    let transcript_written = if FAIL_COMPACTION_TRANSCRIPT_WRITE.with(|fail| fail.replace(false)) {
        Err("injected compaction transcript failure".to_string())
    } else {
        write_jsonl(&messages_path(core, id), messages)
    };
    #[cfg(not(test))]
    let transcript_written = write_jsonl(&messages_path(core, id), messages);
    if let Err(error) = transcript_written {
        // The boundaries moved for a prune that did not land: put them back,
        // and say so if even that write fails, since a boundary that stays
        // shifted renders a replay divider early until the next prune.
        metas[pos].resume_points = before;
        return Err(match save_index(core, &metas) {
            Ok(()) => error,
            Err(restore) => format!("{error}; the replay boundaries could not be restored either: {restore}"),
        });
    }
    remember_transcript(&mut memo, id, stamp(&messages_path(core, id)));
    Ok(())
}

/// Write `bytes` via a unique same-directory temp file + rename so readers
/// never see a partial target. Unique names avoid two processes clobbering
/// the same `*.tmp`. The temp file is synced (`sync`) before the rename: a
/// rename can reach the disk before the bytes it points at, and a power cut
/// between the two leaves an empty or partial file under the final name with
/// the old contents already gone. Synced, the bytes land first, so a power
/// cut leaves the old file or the new one, never a mix.
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
    let written = std::fs::File::create(&tmp).and_then(|mut file| {
        file.write_all(bytes.as_ref())?;
        sync(&file, &tmp)
    });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    // Reject directories, including symlinks to directories.
    if path.is_dir() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{} is a directory", path.display()));
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => {
            // The rename lives in the directory, so syncing the directory
            // puts it on the disk ahead of whatever is written next. Best
            // effort, and after the fact: the file is already replaced, and
            // an error here must not report a write that landed as one that
            // did not.
            let _ = sync_dir(&parent);
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e.to_string())
        }
    }
}

/// Put `file`'s writes on the device ahead of every write issued after this
/// returns. Every barrier in this module goes through here, so a test can
/// see which ones a write issued and in what order.
///
/// Ordering is all the callers need: no rename may land before the bytes it
/// names, and no transcript rewrite before the archive it relies on. On
/// Apple targets the standard library's sync is a full drive cache flush,
/// which would make every atomic write, and so every turn, wait on the drive;
/// a barrier sync gives the ordering without that wait. A file system that
/// does not take one gets a plain sync instead.
fn sync(file: &std::fs::File, path: &Path) -> std::io::Result<()> {
    sync_device(file, path).map_err(|e| cannot_sync(e, path))
}

/// `sync` with the OS error as it came, so `sync_dir` can read its code.
fn sync_device(file: &std::fs::File, path: &Path) -> std::io::Result<()> {
    #[cfg(test)]
    SYNCED.with(|synced| synced.borrow_mut().push(path.to_path_buf()));
    #[cfg(not(test))]
    let _ = path;
    #[cfg(target_vendor = "apple")]
    let result = {
        use std::os::fd::AsRawFd;
        let fd = file.as_raw_fd();
        // SAFETY: `fd` stays open while `file` is borrowed, and neither call
        // takes a pointer.
        let synced = unsafe { libc::fcntl(fd, libc::F_BARRIERFSYNC) != -1 || libc::fsync(fd) != -1 };
        if synced { Ok(()) } else { Err(std::io::Error::last_os_error()) }
    };
    #[cfg(not(target_vendor = "apple"))]
    let result = file.sync_all();
    result
}

fn cannot_sync(e: std::io::Error, path: &Path) -> std::io::Error {
    std::io::Error::new(e.kind(), format!("cannot sync {}: {e}", path.display()))
}

/// `sync` a directory, which holds the names of its files: a new name or a
/// rename reaches the disk through it, not through the file's own sync. Unix
/// only, since elsewhere a directory cannot be opened as a file to sync. A
/// file system that cannot sync a directory at all (`dir_sync_unsupported`)
/// counts as synced: there the sync never succeeds, so failing on it would
/// fail every ordered append, and so every compaction, for good.
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    if !cfg!(unix) {
        return Ok(());
    }
    let file = std::fs::File::open(dir).map_err(|e| cannot_sync(e, dir))?;
    let synced = sync_device(&file, dir);
    #[cfg(test)]
    let synced = FAIL_DIR_SYNC.with(|fail| fail.get()).map_or(synced, |code| Err(std::io::Error::from_raw_os_error(code)));
    match synced {
        Err(e) if !dir_sync_unsupported(&e) => Err(cannot_sync(e, dir)),
        _ => Ok(()),
    }
}

/// Whether a directory sync failed only because the file system does not
/// sync directories: some reject it with EINVAL or EBADF (no sync operation
/// for a directory), others with ENOTSUP. Any other error, such as EIO, is a
/// real failure to write.
fn dir_sync_unsupported(e: &std::io::Error) -> bool {
    #[cfg(unix)]
    let unsupported = [libc::EINVAL, libc::EBADF, libc::ENOTSUP, libc::EOPNOTSUPP];
    #[cfg(not(unix))]
    let unsupported: [i32; 0] = [];
    e.raw_os_error().is_some_and(|code| unsupported.contains(&code))
}

fn write_jsonl(path: &PathBuf, messages: &[ChatMessage]) -> Result<(), String> {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&serde_json::to_string(msg).map_err(|e| e.to_string())?);
        out.push('\n');
    }
    write_atomic(path, out)
}

fn append_jsonl(path: &PathBuf, messages: &[ChatMessage], ordered: bool) -> Result<(), String> {
    // Serialize the whole tail first, then one write. Callers must heal on
    // failure (rewrite the full file) so a partial write cannot be re-appended
    // and duplicate complete lines when `persisted` is left unchanged.
    let mut buf = String::new();
    for msg in messages {
        buf.push_str(&serde_json::to_string(msg).map_err(|e| e.to_string())?);
        buf.push('\n');
    }
    append_text(path, &buf, ordered)
}

/// Append whole lines to a JSONL file, starting on a fresh line after its
/// last complete record. A crash or a full disk can leave the file ending
/// without a newline in two ways. A complete record that lost only its
/// newline gets one, or the next record would be glued onto it and both
/// would be unreadable. A torn record (`is_torn`) is cut: no reader can use
/// it, and left in place it would sit mid-file after this append, where the
/// transcript loader refuses the whole session. No record moves either way,
/// and a final line that is neither (damage) is kept and gets a newline like
/// a complete record, so the loader still refuses it with its bytes intact.
/// A write that fails part way is cut back off too, since the caller will
/// append the same records again.
///
/// `ordered` syncs the bytes and then the directory (`sync`), so they and the
/// file's name, which the file's own sync does not cover when this append
/// created it, reach the disk ahead of any later write. A failed sync fails
/// the append, before anything that relies on it is written, except a
/// directory sync the file system does not support (`sync_dir`). A compaction
/// needs that for what it archives, because the transcript rewrite after it
/// deletes the same messages. A transcript append does not: a power cut
/// that loses its tail costs the newest messages, not the session, since the
/// first save wrote the file whole (`save_messages`) and what remains ends
/// in a complete record or a torn one the loader drops, and a sync per save
/// would cost every model request one.
fn append_text(path: &PathBuf, text: &str, ordered: bool) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    let start = final_line_start(&mut file, len).map_err(|e| e.to_string())?;
    let mut keep = len;
    let mut buf = String::with_capacity(text.len() + 1);
    if start < len {
        let mut line = Vec::new();
        file.seek(SeekFrom::Start(start))
            .and_then(|_| file.read_to_end(&mut line))
            .map_err(|e| e.to_string())?;
        if is_torn(&line) {
            file.set_len(start).map_err(|e| e.to_string())?;
            keep = start;
        } else {
            buf.push('\n');
        }
    }
    buf.push_str(text);
    if let Err(e) = file.write_all(buf.as_bytes()) {
        let _ = file.set_len(keep);
        return Err(e.to_string());
    }
    file.flush().map_err(|e| e.to_string())?;
    if ordered {
        let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        sync(&file, path).and_then(|()| sync_dir(dir)).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Where the final line of a file `len` bytes long starts: `len` itself when
/// the file is empty or ends in a newline. Reads backwards a block at a time,
/// so an append never reads the history it extends.
fn final_line_start(file: &mut std::fs::File, len: u64) -> std::io::Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut block = [0u8; 4096];
    let mut end = len;
    while end > 0 {
        let begin = end.saturating_sub(block.len() as u64);
        let part = &mut block[..(end - begin) as usize];
        file.seek(SeekFrom::Start(begin))?;
        file.read_exact(part)?;
        if let Some(i) = part.iter().rposition(|&b| b == b'\n') {
            return Ok(begin + i as u64 + 1);
        }
        end = begin;
    }
    Ok(0)
}

/// Whether an unterminated final line reads as a record a crash or a full
/// disk cut off mid-write: its JSON runs out before it closes. A tear keeps
/// the start of a record and nothing else, so parsing it fails only at the
/// end of input (a cut inside a multi-byte character included), never at a
/// wrong byte. A power cut can also leave the end of an append reading as
/// zeros, which no record contains (the serializer escapes control
/// characters), so trailing zeros are set aside first. Any other final line
/// is kept: one that only lost its newline is a record, and one that fails
/// before its end (a closed record that does not parse) is damage the loader
/// refuses with the bytes preserved.
fn is_torn(line: &[u8]) -> bool {
    let end = line.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    serde_json::from_slice::<serde_json::Value>(&line[..end]).is_err_and(|e| e.is_eof())
}

/// How many leading bytes of a JSONL file hold its records: all of them,
/// unless the final line is unterminated and torn.
fn records_end(bytes: &[u8]) -> usize {
    let start = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    if start < bytes.len() && is_torn(&bytes[start..]) { start } else { bytes.len() }
}

/// Sessions for one project, most recently updated first. Err names a
/// damaged index rather than listing nothing.
///
/// A session belongs to `project` when both paths name the same directory,
/// resolved on both sides (`state::canonical_root`, the form trust and
/// history key a project by). The index keeps each path as its frontend
/// spelled it and is never rewritten for this, so a directory reached
/// through a symlink, or moved and linked back, would otherwise hide its
/// sessions from `--continue`, `/resume`, and `--recall` under another
/// spelling. Each distinct stored path resolves once, not once per session.
pub fn list(core: &Core, project: &str) -> Result<Vec<SessionMeta>, String> {
    let wanted = crate::state::canonical_root(Path::new(project));
    let mut same_project: HashMap<String, bool> = HashMap::new();
    let mut metas: Vec<(usize, SessionMeta)> = load_index_checked(core)?
        .into_iter()
        .enumerate()
        .filter(|(_, m)| {
            m.project == project
                || *same_project
                    .entry(m.project.clone())
                    .or_insert_with(|| crate::state::canonical_root(Path::new(&m.project)) == wanted)
        })
        .collect();
    // updated_at is whole seconds and the index is append-ordered, so two
    // sessions touched in the same second tie on the timestamp alone; the
    // later index entry is the newer one and must sort first, or latest()
    // hands --continue an older same-second sibling.
    metas.sort_by_key(|(i, m)| std::cmp::Reverse((m.updated_at, *i)));
    Ok(metas.into_iter().map(|(_, m)| m).collect())
}

/// Most recent session for a project, if any (used by --continue).
pub fn latest(core: &Core, project: &str) -> Result<Option<SessionMeta>, String> {
    Ok(list(core, project)?.into_iter().next())
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

/// One message was inserted at index `at`, so the transcript grows by exactly
/// that message and every boundary at or after it moves up one. The mirror
/// of the boundary shift for a prune, which handles shrinkage. Hydration of
/// a transcript saved without its system prompt inserts that prompt at 0.
pub fn shift_resume_points_for_insert(core: &Core, id: &str, at: u64) -> Result<(), String> {
    ensure_owned(core, id)?;
    with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            shift_points_for_insert(&mut m.resume_points, at);
        }
    })
}

fn shift_points_for_insert(points: &mut [u64], at: u64) {
    for p in points {
        if *p >= at {
            *p = p.saturating_add(1);
        }
    }
}

/// One message was removed from index `at`, so the transcript shrinks by
/// exactly that message and every boundary after it moves down one: the
/// mirror of [`shift_resume_points_for_insert`]. A boundary on the removed
/// message stays put and now marks the message that followed it, so two
/// boundaries can meet there and become one. Hydration removes a tool reply
/// that was saved after prompts instead of after its call.
fn shift_points_for_remove(points: &mut Vec<u64>, at: u64) {
    for p in points.iter_mut() {
        if *p > at {
            *p -= 1;
        }
    }
    points.sort_unstable();
    points.dedup();
}

/// One message inserted into or removed from a transcript, at an index
/// counted after the edits before it: what a replay boundary has to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeEdit {
    Insert(u64),
    Remove(u64),
}

/// Move every replay boundary through `edits`, in the order they were made,
/// in one index write: either every edit lands or none does, so a caller
/// whose write failed keeps the edits and retries them whole, and no
/// boundary moves twice.
pub fn apply_resume_edits(core: &Core, id: &str, edits: &[ResumeEdit]) -> Result<(), String> {
    ensure_owned(core, id)?;
    with_index(core, |metas| {
        if let Some(m) = metas.iter_mut().find(|m| m.id == id) {
            for edit in edits {
                match *edit {
                    ResumeEdit::Insert(at) => shift_points_for_insert(&mut m.resume_points, at),
                    ResumeEdit::Remove(at) => shift_points_for_remove(&mut m.resume_points, at),
                }
            }
        }
    })
}

/// Record that a new sitting resumed this session with `message_index`
/// messages already on disk. Index zero is an empty session, not a
/// boundary; repeats (resuming again before any new turn) are deduplicated.
pub fn record_resume_point(core: &Core, id: &str, message_index: u64) {
    if let Err(message) = ensure_owned(core, id) {
        core.send_agent(id, AgentEvent::Error { message });
        return;
    }
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
    remove(core, id, false).map(|_| ())
}

/// Remove a session that no store backs: no transcript, archive, or
/// compaction file exists, so nothing could ever be read back from it. A
/// front end creates its session before the first turn, and a client that
/// quits first, or a first prompt a gate refuses, leaves an index entry with
/// nothing behind it; every later `--recall` then reports it as listed but
/// unreadable, and `--continue` attaches to it. The manifest alone does not
/// count: a turn that started writes one before anything else, and a turn
/// that then failed still left no history. Ok(whether it was removed); a
/// removal the index refused (damaged, unwritable) is the caller's to
/// report, because the entry it leaves behind is exactly the one this
/// exists to prevent.
pub fn discard_if_empty(core: &Core, id: &str) -> Result<bool, String> {
    // Settled without claiming: a session with history stays, whoever holds it.
    if backed(core, id) {
        return Ok(false);
    }
    remove(core, id, true)
}

fn backed(core: &Core, id: &str) -> bool {
    [messages_path(core, id), archive_path(core, id), compaction_path(core, id)]
        .iter()
        .any(|p| p.exists())
}

/// `delete`, or with `only_if_empty` the removal half of `discard_if_empty`.
/// Ok(whether the session was removed).
fn remove(core: &Core, id: &str, only_if_empty: bool) -> Result<bool, String> {
    // Explicit deletion may remove damaged data, but never a foreign writer.
    claim_session(core, id, false, false)?;
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
    //
    // A discard has not decided yet, and a session it keeps must keep its turn.
    if !only_if_empty {
        core.cancel(id);
    }
    // The index entry and the files go under one lock. Dropping the entry
    // first and the files second would let an append pass its check against
    // the stale index and recreate what this call is removing.
    let (mut memo, flock) = lock_store(core)?;
    // A turn of this process can save between the caller's look and this
    // lock. Every transcript write holds the lock, so the answer here is
    // final: a session that gained history in that window stays.
    if only_if_empty && backed(core, id) {
        return Ok(false);
    }
    let mut metas = match read_index(core) {
        IndexRead::Loaded(metas) => metas,
        IndexRead::Missing => Vec::new(),
        // Deleting one session must not cost every other session its
        // metadata; repair the index first.
        IndexRead::Damaged(reason) => return Err(reason),
    };
    metas.retain(|m| m.id != id);
    save_index(core, &metas)?;
    // Gone from the index, so gone for every write still in flight here,
    // which learns it from this record rather than from the index.
    memo.listed.remove(id);
    memo.transcripts.remove(id);
    memo.removed.insert(id.to_string());
    let _ = std::fs::remove_file(messages_path(core, id));
    let _ = std::fs::remove_file(manifest_path(core, id));
    let _ = std::fs::remove_file(compaction_path(core, id));
    let _ = std::fs::remove_file(archive_path(core, id));
    let _ = std::fs::remove_file(usage_path(core, id));
    // Still held by the claim above, and unlinked under the store lock every
    // claim opens it under (see `SessionOwner`).
    let _ = std::fs::remove_file(lock_path(core, id));
    drop((memo, flock));
    if only_if_empty {
        core.cancel(id);
    }
    let _ = detach(core, id);
    Ok(true)
}

/// Set the title from the first user message, once.
pub fn set_title_if_new(core: &Core, id: &str, title: &str) {
    if let Err(message) = ensure_owned(core, id) {
        core.send_agent(id, AgentEvent::Error { message });
        return;
    }
    let title = title.trim().chars().take(48).collect::<String>();
    if title.is_empty() {
        return;
    }
    let _ = with_index(core, |metas| {
        // Re-appended for the same reason touch() re-appends: this bump
        // rides the session's first message, exactly when it becomes the
        // most recent one.
        if let Some(i) = metas.iter().position(|m| m.id == id) {
            let mut m = metas.remove(i);
            if m.title == UNTITLED {
                m.title = title;
            }
            m.updated_at = now();
            metas.push(m);
        }
    });
}

pub fn touch(core: &Core, id: &str) {
    if let Err(message) = ensure_owned(core, id) {
        core.send_agent(id, AgentEvent::Error { message });
        return;
    }
    let _ = with_index(core, |metas| {
        // Re-append rather than update in place: updated_at is whole
        // seconds, so within one second only index position can order
        // sessions, and the position must therefore track last activity,
        // not creation.
        if let Some(i) = metas.iter().position(|m| m.id == id) {
            let mut m = metas.remove(i);
            m.updated_at = now();
            metas.push(m);
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

/// Load the authoritative transcript. Only a missing file is a fresh session.
/// A damaged or unreadable record must survive for explicit recovery. A torn
/// final record (`is_torn`) is not one: it is what a crash mid-append leaves,
/// it never became a message, and refusing it would strand any session a
/// crash interrupted mid-save. It is skipped here and cut by the next append,
/// or replaced by the next rewrite.
pub fn load_messages(core: &Core, id: &str) -> Result<Option<Vec<ChatMessage>>, String> {
    let _store = lock_store(core)?;
    load_messages_locked(core, id)
}

fn load_messages_locked(core: &Core, id: &str) -> Result<Option<Vec<ChatMessage>>, String> {
    let path = messages_path(core, id);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("transcript {} is unreadable: {e}; original file preserved", path.display())),
    };
    #[cfg(test)]
    TRANSCRIPT_READS.with(|reads| reads.set(reads.get() + 1));
    let text = std::str::from_utf8(&bytes[..records_end(&bytes)])
        .map_err(|e| format!("transcript {} is unreadable: {e}; original file preserved", path.display()))?;
    let mut parsed = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let message = serde_json::from_str(line).map_err(|e| format!(
            "transcript {} is damaged at line {}: {e}; original file preserved",
            path.display(), index + 1,
        ))?;
        parsed.push(message);
    }
    if parsed.is_empty() {
        return Err(format!("transcript {} holds no complete record; original file preserved", path.display()));
    }
    Ok(Some(parsed))
}

/// A `persisted` count past the end of any transcript, for a file known to
/// differ from memory: [`save_messages`] rewrites whenever `persisted`
/// exceeds what it is given, so the next save replaces the file whole
/// instead of appending to it or refusing to.
pub const PERSISTED_STALE: usize = usize::MAX;

/// Persist messages. Appends only new tail lines when possible; rewrites the
/// whole file after budget trimming or message drops.
///
/// Serializes disk access with `sessions_lock` so concurrent turns in the same
/// process cannot interleave appends or rewrites of the same file.
/// True when the transcript on disk cannot now diverge from `messages`:
/// either the bytes landed, or there is deliberately no recorded transcript
/// to diverge from (a deleted session's save is a silent no-op, and deletion
/// removed what a resume would replay). False when a recorded transcript
/// exists and could not be brought up to date: a damaged index, or a write
/// that failed after the rewrite fallback. Callers that only fire-and-report
/// may ignore it; a caller about to act on the recorded state (a refusal
/// continuation) must not.
pub fn save_messages(core: &Core, id: &str, messages: &[ChatMessage], persisted: &mut usize, rewrite: bool) -> bool {
    if let Err(message) = ensure_owned(core, id) {
        core.send_agent(id, AgentEvent::Error { message });
        return false;
    }
    let path = messages_path(core, id);
    let (mut memo, _flock) = match lock_store(core) {
        Ok(store) => store,
        Err(reason) => {
            core.send_agent(id, AgentEvent::Error { message: format!("failed to persist session: {reason}") });
            return false;
        }
    };
    // Same rule as the sidecars, and for the same reason: cancellation is
    // cooperative, so a turn keeps running for a while after `delete` and
    // ends with an unconditional save. Without this the transcript of a
    // deleted session comes back, which is the one file that made the
    // deletion visible in the first place.
    match indexed_locked(&memo, core, id) {
        Ok(true) => {}
        // "Deleted" stays a silent no-op.
        Ok(false) => return true,
        // A damaged index is a different state: the write is being dropped
        // for a reason the user can fix, so say so instead of losing the
        // transcript quietly.
        Err(reason) => {
            core.send_agent(
                id,
                AgentEvent::Error {
                    message: format!("warning: failed to persist session to disk: {reason}"),
                },
            );
            return false;
        }
    }
    let needs_rewrite = rewrite || messages.len() < *persisted;
    // Validate before replacing existing bytes. A failed append below may
    // rewrite our own partial append, but a damaged input is never repaired
    // implicitly. Ordinary tail appends do not reparse the growing history.
    if needs_rewrite || *persisted == 0 {
        let checked = load_messages_locked(core, id).and_then(|loaded| {
            if loaded.is_some() && *persisted == 0 && !needs_rewrite {
                Err("an existing transcript must be loaded before appending".into())
            } else { Ok(()) }
        });
        if let Err(message) = checked {
            core.send_agent(id, AgentEvent::Error { message });
            return false;
        }
    }

    // The first save creates the transcript (the check above refused an
    // existing one), and is written whole too. An append there that a crash
    // or a power cut interrupts leaves a fragment or an empty file, with no
    // complete record to resume from, so the session could never be
    // resumed; an interrupted rename leaves no transcript, a fresh session.
    let whole = needs_rewrite || (*persisted == 0 && !messages.is_empty());
    let wrote = whole || messages.len() > *persisted;
    // Whether the file this save leaves holds only what this process
    // validated or wrote, so the next turn start need not read it again
    // (`revalidate_if_changed`): a whole write does, and an append does when
    // nothing else wrote the file since this process last did.
    let known = whole || stamp(&path).is_some_and(|seen| memo.transcripts.get(id) == Some(&seen));
    let result = if whole {
        write_jsonl(&path, messages)
    } else if messages.len() > *persisted {
        // Append is best-effort for the common path. On any failure (including
        // partial write_all), rewrite the full transcript atomically so a
        // later append cannot duplicate complete lines that already landed.
        match append_jsonl(&path, &messages[*persisted..], false) {
            Ok(()) => Ok(()),
            Err(append_err) => write_jsonl(&path, messages).map_err(|rewrite_err| {
                format!("append failed ({append_err}); rewrite also failed: {rewrite_err}")
            }),
        }
    } else {
        Ok(())
    };

    match result {
        Ok(()) => {
            if wrote {
                remember_transcript(&mut memo, id, known.then(|| stamp(&path)).flatten());
            }
            *persisted = messages.len();
            true
        }
        Err(e) => {
            memo.transcripts.remove(id);
            core.send_agent(
                id,
                AgentEvent::Error {
                    message: format!("warning: failed to persist session to disk: {e}"),
                },
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every write that can fail before the transcript rewrite leaves the
    /// live transcript and its replay boundaries exactly as they were. What a
    /// failed attempt may leave behind is an archive line and a digest record
    /// for a prune that did not happen: duplicates the retry repeats and the
    /// readers tolerate, never a loss.
    #[test]
    fn an_index_write_failure_leaves_the_transcript_untouched() {
        let dir = std::env::temp_dir().join(format!("openmax-compaction-index-failure-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "project".into()).unwrap().id;
        let original = vec![ChatMessage::system("rules"), ChatMessage::user("first"), ChatMessage::user("second")];
        assert!(save_messages(&core, &id, &original, &mut 0, true));
        record_resume_point(&core, &id, 3);
        let record = CompactionRecord { ts: 1, message_count: 2, tools: vec![], paths: vec![], user_snippets: vec![], digest: "digest".into() };
        let candidate = vec![original[0].clone(), ChatMessage::user("digest")];
        FAIL_COMPACTION_INDEX_WRITE.with(|fail| fail.set(true));
        assert!(commit_compaction(&core, &id, &candidate, &original[1..], Some(&record), original.len(), 2).is_err());
        let bytes = std::fs::read_to_string(messages_path(&core, &id)).unwrap();
        assert!(bytes.contains("first") && !bytes.contains("digest"), "{bytes}");
        assert_eq!(meta(&core, &id).unwrap().resume_points, vec![3]);
        commit_compaction(&core, &id, &candidate, &original[1..], Some(&record), original.len(), 2).unwrap();
        let archived: Vec<String> = load_archive(&core, &id).into_iter().filter_map(|m| m.content).collect();
        assert!(archived.ends_with(&["first".to_string(), "second".to_string()]), "the retry archives what it drops: {archived:?}");
        assert_eq!(last_compaction(&core, &id).unwrap().digest, "digest");
        assert_eq!(serde_json::to_value(load_messages(&core, &id).unwrap().unwrap()).unwrap(), serde_json::to_value(&candidate).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The index moves before the transcript does, so a transcript rewrite
    /// that fails has to put the boundaries back, or a replay renders a
    /// divider early for a prune that never landed.
    #[test]
    fn a_transcript_write_failure_restores_the_replay_boundaries() {
        let dir = std::env::temp_dir().join(format!("openmax-compaction-transcript-failure-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "project".into()).unwrap().id;
        let mut original = vec![ChatMessage::system("rules"), ChatMessage::user("first")];
        for i in 0..6 {
            original.push(ChatMessage::user(format!("message {i}")));
        }
        assert!(save_messages(&core, &id, &original, &mut 0, true));
        record_resume_point(&core, &id, 6);
        let candidate = vec![original[0].clone(), original[1].clone(), ChatMessage::user("digest"), original[7].clone()];
        FAIL_COMPACTION_TRANSCRIPT_WRITE.with(|fail| fail.set(true));
        assert!(commit_compaction(&core, &id, &candidate, &original[2..7], None, original.len(), 2).is_err());
        assert_eq!(meta(&core, &id).unwrap().resume_points, vec![6], "boundaries restored");
        assert_eq!(serde_json::to_value(load_messages(&core, &id).unwrap().unwrap()).unwrap(), serde_json::to_value(&original).unwrap());
        commit_compaction(&core, &id, &candidate, &original[2..7], None, original.len(), 2).unwrap();
        assert_eq!(meta(&core, &id).unwrap().resume_points, vec![3], "boundaries shift with the prune that landed");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A crash or a full disk can leave the archive ending in a torn record
    /// with no newline. The next commit must start on a fresh line, or its
    /// first record is glued to the fragment and both are unreadable, and
    /// that record was just dropped from the transcript: history lost.
    #[test]
    fn a_torn_archive_tail_never_swallows_the_next_record() {
        let dir = std::env::temp_dir().join(format!("openmax-torn-archive-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "project".into()).unwrap().id;
        let original = vec![ChatMessage::system("rules"), ChatMessage::user("first"), ChatMessage::user("second")];
        assert!(save_messages(&core, &id, &original, &mut 0, true));
        append_archive(&core, &id, &[ChatMessage::user("prior")]);
        std::fs::OpenOptions::new().append(true).open(archive_path(&core, &id)).unwrap()
            .write_all(b"{\"role\":\"user\",\"content\":\"torn").unwrap();
        let candidate = vec![original[0].clone(), ChatMessage::user("digest")];
        commit_compaction(&core, &id, &candidate, &original[1..], None, original.len(), 2).unwrap();
        let archived: Vec<String> = load_archive(&core, &id).into_iter().filter_map(|m| m.content).collect();
        assert_eq!(archived, ["prior", "first", "second"], "every dropped message reads back");
        let _ = std::fs::remove_dir_all(dir);
    }

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
        with_index(&core, |metas| {
            let m = metas.iter_mut().find(|m| m.id.as_str() == id.as_str()).unwrap();
            m.resume_points = shifted_resume_points(&m.resume_points, 6, 3, 2);
        }).unwrap();
        // The prune that fires this shift also inserted its digest note at
        // the head, index 2 behind `[system, request]`; collapsed boundaries
        // land after it, never on it.
        assert_eq!(meta(&core, id).unwrap().resume_points, vec![3, 7]);

        // Behind `[system, note, request]` the head is 3 and the note lands
        // there: a boundary inside the head stays, a collapsed one lands
        // after the note, and a note insert leaves the head alone too.
        assert_eq!(shifted_resume_points(&[2, 4, 10], 9, 6, 3), vec![2, 4, 7]);
        assert_eq!(shifted_resume_points(&[2, 3, 10], 9, 10, 3), vec![2, 4, 11]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A prune that only truncated removes nothing and inserts its note at
    /// the pinned head, so the transcript grows by one and every boundary
    /// from the note onward must follow or replay dividers drift the other
    /// way.
    #[test]
    fn a_note_insert_moves_replay_boundaries_up_one() {
        let dir =
            std::env::temp_dir().join(format!("openmax-note-insert-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = &create(&core, "/tmp/p".into()).unwrap().id;
        record_resume_point(&core, id, 1);
        record_resume_point(&core, id, 3);
        record_resume_point(&core, id, 7);

        shift_resume_points_for_insert(&core, id, 2).unwrap();
        assert_eq!(meta(&core, id).unwrap().resume_points, vec![1, 4, 8]);
        // An insert at the very front moves every boundary.
        shift_resume_points_for_insert(&core, id, 0).unwrap();
        assert_eq!(meta(&core, id).unwrap().resume_points, vec![2, 5, 9]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Sessions touched within the same second: the index is append-ordered
    /// and updated_at is whole seconds, so the timestamp alone cannot order
    /// them and a stable sort would keep the OLDEST first. The later index
    /// entry is the newer session and must win, or --continue silently
    /// resumes an older same-second sibling and the /resume panel inverts.
    #[test]
    fn same_second_sessions_list_newest_first() {
        let dir = std::env::temp_dir().join(format!("openmax-tie-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let a = create(&core, "/tmp/tie".into()).unwrap().id;
        let b = create(&core, "/tmp/tie".into()).unwrap().id;
        let c = create(&core, "/tmp/tie".into()).unwrap().id;
        with_index(&core, |metas| {
            for m in metas.iter_mut() {
                m.updated_at = 1_000;
            }
        })
        .unwrap();
        let listed: Vec<String> = list(&core, "/tmp/tie").unwrap().into_iter().map(|m| m.id).collect();
        assert_eq!(listed, vec![c.clone(), b.clone(), a.clone()]);
        assert_eq!(latest(&core, "/tmp/tie").unwrap().unwrap().id, c);

        // Touch recency inside the same second: touching the OLDEST session
        // must make it the latest, so the touched entry re-appends instead
        // of keeping its creation slot. touch() stamps now(); flatten every
        // stamp back to one value so only position can decide.
        touch(&core, &a);
        with_index(&core, |metas| {
            for m in metas.iter_mut() {
                m.updated_at = 1_000;
            }
        })
        .unwrap();
        assert_eq!(latest(&core, "/tmp/tie").unwrap().unwrap().id, a);
        let listed: Vec<String> = list(&core, "/tmp/tie").unwrap().into_iter().map(|m| m.id).collect();
        assert_eq!(listed, vec![a, c, b]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A session that persisted nothing is not history: at its front end's
    /// exit it leaves the index, so recall never reports it as unreadable and
    /// `--continue` never attaches to it. One with a transcript stays, and a
    /// manifest alone (a turn that started and failed before its first save)
    /// does not keep it.
    #[test]
    fn a_session_with_no_store_behind_it_is_discarded() {
        let dir = std::env::temp_dir().join(format!("openmax-discard-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let empty = create(&core, "/tmp/p".into()).unwrap().id;
        let manifest_only = create(&core, "/tmp/p".into()).unwrap().id;
        save_manifest(&core, &manifest_only, &crate::registry::Registry::builtin_only().to_manifest());
        let kept = create(&core, "/tmp/p".into()).unwrap().id;
        let mut persisted = 0usize;
        assert!(save_messages(&core, &kept, &[ChatMessage::system("sys")], &mut persisted, false));

        assert_eq!(discard_if_empty(&core, &empty), Ok(true));
        assert_eq!(discard_if_empty(&core, &manifest_only), Ok(true));
        assert!(!manifest_path(&core, &manifest_only).exists(), "its files go with it");
        assert_eq!(discard_if_empty(&core, &kept), Ok(false), "a transcript is history");
        let ids: Vec<String> = list(&core, "/tmp/p").unwrap().into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![kept.clone()]);

        // An index that cannot be rewritten leaves the entry, and says so
        // rather than reporting a removal that did not happen.
        let ghost = create(&core, "/tmp/p".into()).unwrap().id;
        let index_path = core.data_dir.join("sessions").join("index.json");
        let good_index = std::fs::read_to_string(&index_path).unwrap();
        std::fs::write(&index_path, "[{\"id\": \"trunc").unwrap();
        let err = discard_if_empty(&core, &ghost).unwrap_err();
        assert!(err.contains("does not parse"), "{err}");
        std::fs::write(&index_path, good_index).unwrap();
        assert_eq!(discard_if_empty(&core, &ghost), Ok(true), "once the index heals it goes");
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
        assert!(load_messages(&core, &id).unwrap().is_none(), "a deleted transcript stays deleted");
        save_manifest(&core, &id, &crate::registry::RegistryManifest {
            version: 1,
            external_tools: Vec::new(),
            skills: Vec::new(),
            ext_fingerprint: 0,
            memory_files: None,
            memory_rows: None,
        });
        assert!(load_manifest(&core, &id).is_none(), "and so does its manifest");
        // All five session-scoped files, so this cannot regress one at a time.
        for suffix in ["messages.json", "manifest.json", "compaction.jsonl", "archive.jsonl", "usage.jsonl"] {
            let path = sessions_dir(&core).join(format!("{id}.{suffix}"));
            assert!(!path.exists(), "{suffix} came back after delete");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Same doctrine as the append/delete test above: racing two writers and
    /// hoping for the bad interleaving proves nothing, so this parks one
    /// writer mid-read-modify-write and asserts the other blocks. Two `Core`s
    /// on one data dir model two processes exactly - `sessions_lock` is per
    /// `Core`, so only the index flock can serialize them.
    #[test]
    fn a_parallel_process_cannot_erase_a_sibling_session() {
        let dir = std::env::temp_dir().join(format!("openmax-flock-{}", uuid::Uuid::new_v4()));
        let (core_a, _rx_a) = Core::new(dir.clone()).unwrap();
        let (core_b, _rx_b) = Core::new(dir.clone()).unwrap();

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let parked = {
            let core_a = core_a.clone();
            std::thread::spawn(move || {
                with_index(&core_a, |metas| {
                    entered_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    metas.push(SessionMeta {
                        id: "held-entry".into(),
                        project: "/tmp/p".into(),
                        title: "t".into(),
                        created_at: 1,
                        updated_at: 1,
                        resume_points: Vec::new(),
                    });
                })
                .unwrap();
            })
        };
        entered_rx.recv().unwrap();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let sibling = {
            let core_b = core_b.clone();
            std::thread::spawn(move || {
                let meta = create(&core_b, "/tmp/p".into()).unwrap();
                done_tx.send(()).unwrap();
                meta
            })
        };
        assert!(
            done_rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "a second process's create must block while another holds the index mid-write, \
             or its entry is saved over and the still-indexed gate drops its transcript"
        );
        release_tx.send(()).unwrap();
        parked.join().unwrap();
        let sibling_meta = sibling.join().unwrap();

        let ids: Vec<String> = load_index(&core_a).into_iter().map(|m| m.id).collect();
        assert!(ids.contains(&"held-entry".to_string()), "the parked writer's entry landed");
        assert!(ids.contains(&sibling_meta.id), "and the sibling's entry survived it");
        // The consequence that made this a data-loss bug and not a metadata
        // nit: the sibling's writes still land.
        append_usage(&core_b, &sibling_meta.id, &TokenUsage {
            ts: 1,
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: None,
        });
        assert_eq!(load_usage(&core_b, &sibling_meta.id).len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// One truncated read must not cascade into an empty store: the writer
    /// refuses a damaged index instead of "recovering" it to the default,
    /// and the transcript drop it forces is loud instead of silent.
    #[test]
    fn a_damaged_index_is_refused_not_replaced_with_the_empty_default() {
        let dir = std::env::temp_dir().join(format!("openmax-damaged-{}", uuid::Uuid::new_v4()));
        let (core, mut rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;

        let bad = "[{\"id\": \"trunc";
        std::fs::write(index_path(&core), bad).unwrap();

        touch(&core, &id);
        assert_eq!(
            std::fs::read_to_string(index_path(&core)).unwrap(),
            bad,
            "a metadata write must not save the empty default over a damaged index"
        );
        assert!(
            delete(&core, &id).is_err(),
            "deleting one session must not erase every other session's metadata"
        );
        assert_eq!(std::fs::read_to_string(index_path(&core)).unwrap(), bad);
        assert!(index_diagnostic(&core).is_some(), "recall's diagnostic still has its evidence");

        let mut persisted = 0usize;
        save_messages(&core, &id, &[ChatMessage::user("kept?")], &mut persisted, false);
        let warned = std::iter::from_fn(|| rx.try_recv().ok()).any(|envelope| {
            matches!(&envelope.event, AgentEvent::Error { message }
                if message.contains("does not parse"))
        });
        assert!(warned, "dropping a transcript over a damaged index must be loud");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Readers that answer "what history is there" name a damaged index
    /// instead of answering "none": `--continue` would otherwise report no
    /// previous session over a file full of them. Every refusal names the
    /// file, removals included (a frontend discards the empty session it
    /// leaves), and `--check` sees the same damage without a Core. The shared
    /// reason carries no repair: it also reaches live sessions, where moving
    /// the index aside leaves the session listed nowhere, its later saves
    /// landing where no lookup finds them or dropped without a word. A
    /// frontend's refusal points at `--check`, and only for the damage: a
    /// lock or write failure is not something `--check` repairs.
    #[test]
    fn a_damaged_index_is_named_to_readers_not_reported_as_empty() {
        let dir = std::env::temp_dir().join(format!("openmax-damaged-read-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        std::fs::write(index_path(&core), "[{").unwrap();
        let path = index_path(&core).display().to_string();

        let refusals = [
            latest(&core, "/tmp/p").unwrap_err(),
            list(&core, "/tmp/p").unwrap_err(),
            create(&core, "/tmp/p".into()).unwrap_err(),
            index_diagnostic(&core).unwrap(),
            discard_if_empty(&core, &id).unwrap_err(),
            delete(&core, &id).unwrap_err(),
        ];
        for reason in refusals {
            assert!(reason.contains(&path), "{reason}");
            assert!(!reason.contains("move it aside") && !reason.contains("--check"), "{reason}");
            let refusal = refusal_with_repair(&core, reason.clone());
            assert_eq!(refusal, format!("{reason}; run openmax --check for the repair"));
        }
        let unrelated = format!("cannot lock {path}: busy");
        assert_eq!(refusal_with_repair(&core, unrelated.clone()), unrelated);
        assert_eq!(index_damage(&dir).map(|(at, _)| at), Some(index_path(&core)));
        assert_eq!(std::fs::read_to_string(index_path(&core)).unwrap(), "[{");

        std::fs::remove_file(index_path(&core)).unwrap();
        assert!(latest(&core, "/tmp/p").unwrap().is_none(), "a missing index is an empty store");
        assert!(index_damage(&dir).is_none());
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
            extra_content: None,
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
    fn damaged_transcripts_refuse_loading_saving_and_compaction() {
        let dir = std::env::temp_dir().join(format!("openmax-damage-{}", uuid::Uuid::new_v4()));
        let (core, mut rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let path = messages_path(&core, &id);
        assert!(load_messages(&core, &id).unwrap().is_none());
        let good = b"{\"role\":\"user\",\"content\":\"keep\"}\n";
        let cases = [
            Vec::new(), b"{torn".to_vec(), vec![0xff],
            [good.as_slice(), b"{broken middle}\n", good.as_slice()].concat(),
            [good.as_slice(), b"{broken tail}\n"].concat(),
        ];
        for bytes in cases {
            std::fs::write(&path, &bytes).unwrap();
            let error = load_messages(&core, &id).unwrap_err();
            assert!(error.contains("transcript"), "{error}");
            if bytes.windows(6).any(|w| w == b"broken") { assert!(error.contains("line 2"), "{error}"); }
            for rewrite in [false, true] {
                let mut persisted = 0;
                assert!(!save_messages(&core, &id, &[ChatMessage::system("replacement")], &mut persisted, rewrite));
                assert_eq!(persisted, 0);
                assert_eq!(std::fs::read(&path).unwrap(), bytes);
            }
            assert!(commit_compaction(&core, &id, &[ChatMessage::system("replacement")], &[], None, 1, 2).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
        assert!(std::iter::from_fn(|| rx.try_recv().ok()).any(|e| matches!(e.event, AgentEvent::Error { .. })));
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(load_messages(&core, &id).unwrap_err().contains("unreadable"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_complete_final_record_without_newline_remains_resumable() {
        let dir = std::env::temp_dir().join(format!("openmax-line-end-{}", uuid::Uuid::new_v4()));
        let (core, _) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        std::fs::write(messages_path(&core, &id), r#"{"role":"user","content":"keep"}"#).unwrap();
        let mut messages = load_messages(&core, &id).unwrap().unwrap();
        let mut persisted = messages.len();
        messages.push(ChatMessage::user("next"));
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        assert_eq!(load_messages(&core, &id).unwrap().unwrap().len(), 2);
        // Damage after attachment must also survive a rewrite or compaction.
        std::fs::write(messages_path(&core, &id), b"{damage after loading").unwrap();
        assert!(!save_messages(&core, &id, &messages, &mut persisted, true));
        assert!(commit_compaction(&core, &id, &messages, &[], None, 2, 2).is_err());
        assert_eq!(std::fs::read(messages_path(&core, &id)).unwrap(), b"{damage after loading");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A crash or a power cut mid-append leaves the transcript ending in part
    /// of a record: no newline, the start of its JSON and nothing else,
    /// sometimes cut inside a multi-byte character or followed by zeros where
    /// a power cut left the end of the append unwritten. That fragment was
    /// never a message, so a resume drops it instead of refusing the session,
    /// and the next save cuts it instead of stranding it mid-file, where it
    /// would be damage. Only such a final line gets this: the same bytes
    /// anywhere else, a whole record that is not a message, or a final record
    /// that is closed but does not parse, are damage, still refused, and no
    /// save cuts them.
    #[test]
    fn a_torn_final_record_is_dropped_on_load_and_cut_by_the_next_save() {
        let dir = std::env::temp_dir().join(format!("openmax-torn-tail-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let path = messages_path(&core, &id);
        let line = |m: &ChatMessage| serde_json::to_string(m).unwrap() + "\n";
        let good = line(&ChatMessage::system("rules")) + &line(&ChatMessage::user("keep"));
        let torn: [&[u8]; 4] = [
            b"{\"role\":\"assistant\",\"content\":\"half", b"{\"role\":\"assistant\",\"content\":\"caf\xc3",
            b"{\"role\":\"assistant\",\"content\":\"half\0\0\0\0", b"\0\0\0\0",
        ];
        for fragment in torn {
            std::fs::write(&path, [good.as_bytes(), fragment].concat()).unwrap();
            let mut messages = load_messages(&core, &id).unwrap().expect("the complete records resume");
            assert_eq!(messages.len(), 2);
            let mut persisted = messages.len();
            messages.push(ChatMessage::user("next"));
            assert!(save_messages(&core, &id, &messages, &mut persisted, false));
            assert_eq!(std::fs::read_to_string(&path).unwrap(), good.clone() + &line(&messages[2]), "the fragment is cut, not stranded");
            assert_eq!(load_messages(&core, &id).unwrap().unwrap().len(), 3);
        }
        // A rewrite or a compaction replaces the file whole, fragment and all.
        let messages = vec![ChatMessage::system("rules"), ChatMessage::user("keep")];
        std::fs::write(&path, [good.as_bytes(), torn[0]].concat()).unwrap();
        assert!(save_messages(&core, &id, &messages, &mut 2, true));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), good);
        std::fs::write(&path, [good.as_bytes(), torn[0]].concat()).unwrap();
        commit_compaction(&core, &id, &messages, &[], None, 2, 2).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), good);

        let next = [messages.as_slice(), &[ChatMessage::user("next")]].concat();
        for damaged in [
            [good.as_bytes(), torn[0], b"\n", good.as_bytes()].concat(),
            [good.as_bytes(), b"{\"role\":5}"].concat(),
            [good.as_bytes(), b"{\"role\":\"user\",\"content\":\"edited\"]"].concat(),
        ] {
            std::fs::write(&path, &damaged).unwrap();
            assert!(load_messages(&core, &id).unwrap_err().contains("line 3"));
            assert!(!save_messages(&core, &id, &messages, &mut 2, true));
            assert_eq!(std::fs::read(&path).unwrap(), damaged);
            assert!(save_messages(&core, &id, &next, &mut 2, false));
            assert!(std::fs::read(&path).unwrap().starts_with(&damaged), "an append never cuts damage");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A transcript written before messages carried `reasoning_details` or a
    /// call's `extra_content` loads with neither, and saving it again writes
    /// the bytes it was read from: the additions are optional on disk and
    /// never rewrite an older file into a new shape.
    #[test]
    fn a_transcript_from_before_reasoning_details_loads_and_saves_unchanged() {
        let dir = std::env::temp_dir().join(format!("openmax-old-transcript-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let path = messages_path(&core, &id);
        let old = concat!(
            "{\"role\":\"system\",\"content\":\"rules\"}\n",
            "{\"role\":\"user\",\"content\":\"read a.txt\"}\n",
            "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}],\"reasoning_content\":\"read it first\",\"reasoning_origin\":\"0123456789abcdef\"}\n",
            "{\"role\":\"tool\",\"content\":\"alpha\",\"tool_call_id\":\"c1\"}\n",
            "{\"role\":\"assistant\",\"content\":\"done\"}\n",
        );
        std::fs::write(&path, old).unwrap();
        let messages = load_messages(&core, &id).unwrap().expect("an older transcript resumes");
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[2].reasoning_content.as_deref(), Some("read it first"));
        assert_eq!(messages[2].reasoning_origin.as_deref(), Some("0123456789abcdef"));
        assert!(messages.iter().all(|m| m.reasoning_details.is_none()));
        assert!(messages[2].tool_calls.as_ref().unwrap().iter().all(|c| c.extra_content.is_none()));
        assert!(save_messages(&core, &id, &messages, &mut 5, true));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), old);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A rename can reach the disk before the bytes it points at, so a power
    /// cut after an unsynced atomic write can leave an empty or partial file
    /// under the final name with the old contents already gone. The temp file
    /// is synced before the rename, and the directory after it, so the
    /// rename itself reaches the disk ahead of later writes too.
    #[test]
    fn an_atomic_write_syncs_its_bytes_before_the_rename_and_the_directory_after() {
        let dir = std::env::temp_dir().join(format!("openmax-sync-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.json");
        SYNCED.with(|synced| synced.take());
        write_atomic(&path, "[]").unwrap();
        let synced = SYNCED.with(|synced| synced.take());
        assert_eq!(synced.len(), if cfg!(unix) { 2 } else { 1 }, "{synced:?}");
        let name = synced[0].file_name().unwrap().to_string_lossy();
        assert!(synced[0].parent() == Some(dir.as_path()) && name.starts_with("index.json.") && name.ends_with(".tmp"), "{synced:?}");
        if cfg!(unix) {
            assert_eq!(synced[1], dir);
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[]");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The transcript rewrite deletes what the archive just gained, so the
    /// archive has to reach the disk first, or a power cut that keeps the
    /// rewrite loses those messages from both. That includes its name: the
    /// first compaction creates the archive, and a file's own sync does not
    /// cover its directory entry, so the directory is synced right after it.
    #[test]
    fn a_compaction_syncs_its_archive_before_the_transcript_rewrite() {
        let dir = std::env::temp_dir().join(format!("openmax-archive-sync-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let original = vec![ChatMessage::system("rules"), ChatMessage::user("first"), ChatMessage::user("second")];
        assert!(save_messages(&core, &id, &original, &mut 0, true));
        let candidate = vec![original[0].clone(), ChatMessage::user("digest")];
        SYNCED.with(|synced| synced.take());
        commit_compaction(&core, &id, &candidate, &original[1..], None, original.len(), 2).unwrap();
        let synced = SYNCED.with(|synced| synced.take());
        let archive = synced.iter().position(|p| *p == archive_path(&core, &id));
        let transcript = format!("{id}.messages.json.");
        let rewrite = synced.iter().position(|p| p.file_name().unwrap().to_string_lossy().starts_with(&transcript));
        assert!(matches!((archive, rewrite), (Some(a), Some(r)) if a < r), "{synced:?}");
        if cfg!(unix) {
            assert_eq!(archive.map(|a| &synced[a + 1]), Some(&sessions_dir(&core)), "{synced:?}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Some file systems cannot sync a directory at all: every directory sync
    /// fails with EINVAL, EBADF or ENOTSUP. Failing a compaction on that can
    /// never succeed later, so every turn past the compaction trigger would
    /// end in an error and the session could not go on. Those errors count
    /// as done; a real I/O error still fails the compaction before the
    /// transcript is touched.
    #[cfg(unix)]
    #[test]
    fn a_file_system_that_cannot_sync_directories_still_compacts() {
        let dir = std::env::temp_dir().join(format!("openmax-dir-sync-unsupported-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "project".into()).unwrap().id;
        let original = vec![ChatMessage::system("rules"), ChatMessage::user("first"), ChatMessage::user("second")];
        let candidate = vec![original[0].clone(), ChatMessage::user("digest")];
        let transcript = |core: &Core| serde_json::to_value(load_messages(core, &id).unwrap().unwrap()).unwrap();
        let commit_with = |code: i32| {
            assert!(save_messages(&core, &id, &original, &mut 0, true));
            FAIL_DIR_SYNC.with(|fail| fail.set(Some(code)));
            let committed = commit_compaction(&core, &id, &candidate, &original[1..], None, original.len(), 2);
            FAIL_DIR_SYNC.with(|fail| fail.set(None));
            committed
        };
        for code in [libc::EINVAL, libc::EBADF, libc::ENOTSUP, libc::EOPNOTSUPP] {
            assert_eq!(commit_with(code), Ok(()), "errno {code}");
            assert_eq!(transcript(&core), serde_json::to_value(&candidate).unwrap());
        }
        let error = commit_with(libc::EIO).unwrap_err();
        assert!(error.contains("cannot sync"), "{error}");
        assert_eq!(transcript(&core), serde_json::to_value(&original).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A session's first save creates its transcript. Appended, a crash or a
    /// power cut part way through leaves only a fragment or an empty file:
    /// no complete record to resume from, so the loader refuses it like any
    /// damage and the session can never be resumed. Written whole through a
    /// temp file and a rename, the same crash leaves no transcript at all,
    /// which is a fresh session. Later saves still append, without a sync.
    #[test]
    fn a_first_save_is_written_whole_so_a_crash_leaves_a_fresh_session() {
        let dir = std::env::temp_dir().join(format!("openmax-first-save-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let path = messages_path(&core, &id);
        for crashed in [&b""[..], b"{\"role\":\"system\",\"content\":\"ru"] {
            std::fs::write(&path, crashed).unwrap();
            assert!(load_messages(&core, &id).unwrap_err().contains("no complete record"));
        }
        std::fs::remove_file(&path).unwrap();
        // What an interrupted whole write leaves: its temp file, no transcript.
        std::fs::write(path.with_file_name(format!("{id}.messages.json.interrupted.tmp")), b"{\"ro").unwrap();
        assert!(load_messages(&core, &id).unwrap().is_none());

        let mut messages = vec![ChatMessage::system("rules"), ChatMessage::user("first")];
        let mut persisted = 0;
        SYNCED.with(|synced| synced.take());
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        let synced = SYNCED.with(|synced| synced.take());
        let transcript = format!("{id}.messages.json.");
        let name = synced.first().map(|p| p.file_name().unwrap().to_string_lossy().into_owned());
        assert!(name.is_some_and(|n| n.starts_with(&transcript) && n.ends_with(".tmp")), "{synced:?}");
        assert_eq!(load_messages(&core, &id).unwrap().unwrap().len(), 2);

        messages.push(ChatMessage::user("second"));
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        assert_eq!(SYNCED.with(|synced| synced.take()), Vec::<PathBuf>::new());
        assert_eq!(load_messages(&core, &id).unwrap().unwrap().len(), 3);
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

        let loaded = load_messages(&core, id).unwrap().unwrap();
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
        assert!(load_messages(&core, id).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_damaged_record_between_tool_call_and_reply_is_not_skipped() {
        let dir = std::env::temp_dir().join(format!("openmax-pair-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let bytes = concat!(
            "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"c\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{}\"}}]}\n",
            "{damaged reply}\n",
            "{\"role\":\"user\",\"content\":\"continue\"}\n",
        );
        std::fs::write(messages_path(&core, &id), bytes).unwrap();
        assert!(load_messages(&core, &id).unwrap_err().contains("line 2"));
        assert_eq!(std::fs::read_to_string(messages_path(&core, &id)).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn attachments_exclude_other_cores_until_detached() {
        let dir = std::env::temp_dir().join(format!("openmax-owner-{}", uuid::Uuid::new_v4()));
        let (first, _rx) = Core::new(dir.clone()).unwrap();
        let (second, _rx2) = Core::new(dir.clone()).unwrap();
        let id = create(&first, "/tmp/p".into()).unwrap().id;
        attach(&first, &id).unwrap();
        assert!(attach(&second, &id).unwrap_err().contains("another process"));
        assert!(delete(&second, &id).is_err());
        assert!(!save_messages(&second, &id, &[ChatMessage::user("foreign")], &mut 0, false));
        assert!(load_messages(&second, &id).unwrap().is_none(), "read-only access stays available");
        let other = create(&second, "/tmp/p".into()).unwrap().id;
        attach(&second, &other).unwrap();
        detach(&first, &id).unwrap();
        attach(&second, &id).unwrap();
        assert!(dir.join("sessions").join(format!("{id}.lock")).exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The lock protocol is part of the on-disk format: every released binary
    /// takes an exclusive flock(2) on these files, and an older and a newer
    /// binary often run side by side on one data dir. A raw flock stands in
    /// for the older binary, so a locking primitive that does not see flock
    /// locks (fcntl record locks on Linux) fails here instead of letting two
    /// versions write one session or one index at once.
    #[cfg(unix)]
    #[test]
    fn session_and_index_locks_exclude_a_raw_flock_holder() {
        let dir = std::env::temp_dir().join(format!("openmax-flock-compat-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let session_lock = sessions_dir(&core).join(format!("{id}.lock"));

        let older = raw_flock(&session_lock).expect("nothing holds the session yet");
        assert!(attach(&core, &id).unwrap_err().contains("another process"));
        drop(older);
        attach(&core, &id).unwrap();
        assert!(raw_flock(&session_lock).is_none(), "an older binary must not attach this session");

        let older = raw_flock(&sessions_dir(&core).join("index.lock")).expect("the index is idle");
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let creator = {
            let core = core.clone();
            std::thread::spawn(move || {
                create(&core, "/tmp/p".into()).unwrap();
                done_tx.send(()).unwrap();
            })
        };
        assert!(
            done_rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "an index write must wait while an older binary holds index.lock"
        );
        drop(older);
        creator.join().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Removing a session unlinks its lock file, which an older binary never
    /// does. That binary must still exclude the removal: unlinking a file it
    /// holds would delete the session it is writing and hand its name to the
    /// next claim. A raw flock stands in for it, as above.
    #[cfg(unix)]
    #[test]
    fn a_removal_is_refused_while_an_older_binary_holds_the_session() {
        let dir = std::env::temp_dir().join(format!("openmax-flock-remove-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        let session_lock = lock_path(&core, &id);

        let older = raw_flock(&session_lock).expect("nothing holds the session yet");
        assert!(delete(&core, &id).unwrap_err().contains("another process"));
        assert!(discard_if_empty(&core, &id).unwrap_err().contains("another process"));
        assert!(session_lock.exists(), "a lock file an older binary holds was unlinked");
        assert!(list(&core, "/tmp/p").unwrap().iter().any(|m| m.id == id), "a held session was removed");
        drop(older);
        assert_eq!(discard_if_empty(&core, &id), Ok(true));
        assert!(!session_lock.exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A released session must be free at once, not when the last copy of
    /// its lock descriptor closes. flock belongs to the open file
    /// description, and a child being spawned holds a copy of every open
    /// descriptor until it execs: a release that only closed the harness's
    /// copy left the session locked while any bash call, tool, or hook was
    /// mid-spawn, and reopening it then was refused as open in another
    /// process.
    #[cfg(unix)]
    #[test]
    fn a_released_session_reopens_while_a_child_is_spawning() {
        let dir = std::env::temp_dir().join(format!("openmax-spawn-session-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let (other, _other_rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        attach(&core, &id).unwrap();

        let spawning = crate::execution::PausedSpawn::start();
        detach(&core, &id).unwrap();
        attach(&core, &id).expect("a released session reopens while a child is spawning");
        detach(&core, &id).unwrap();
        attach(&other, &id).expect("another process takes a released session while a child is spawning");
        assert!(attach(&core, &id).unwrap_err().contains("another process"), "a held session stays exclusive");
        drop(spawning);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The index lock is released the same way, or every other process's
    /// index write waits on a child it never started.
    #[cfg(unix)]
    #[test]
    fn a_released_index_lock_is_free_while_a_child_is_spawning() {
        let dir = std::env::temp_dir().join(format!("openmax-spawn-index-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let store = lock_store(&core).unwrap();

        let spawning = crate::execution::PausedSpawn::start();
        drop(store);
        assert!(
            raw_flock(&sessions_dir(&core).join("index.lock")).is_some(),
            "a released index lock stayed held by a spawning child"
        );
        drop(spawning);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every claim creates `<id>.lock`. A deleted or discarded session is
    /// never claimed again, so a lock file it leaves behind is never reused
    /// or removed: one stray file for every such session, forever. A session
    /// that stays keeps its file, which its next claim reuses.
    #[test]
    fn deleting_or_discarding_a_session_leaves_no_lock_file() {
        let dir = std::env::temp_dir().join(format!("openmax-lockfile-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let lock = |id: &str| dir.join("sessions").join(format!("{id}.lock"));

        let deleted = create(&core, "/tmp/p".into()).unwrap().id;
        attach(&core, &deleted).unwrap();
        assert!(lock(&deleted).exists());
        delete(&core, &deleted).unwrap();
        assert!(!lock(&deleted).exists(), "delete left the session's lock file behind");

        let discarded = create(&core, "/tmp/p".into()).unwrap().id;
        attach(&core, &discarded).unwrap();
        assert_eq!(discard_if_empty(&core, &discarded), Ok(true));
        assert!(!lock(&discarded).exists(), "discard left the session's lock file behind");

        // Never claimed before: the claim delete makes for itself goes too.
        let unclaimed = create(&core, "/tmp/p".into()).unwrap().id;
        delete(&core, &unclaimed).unwrap();
        assert!(!lock(&unclaimed).exists(), "delete left the lock file its own claim made");

        let kept = create(&core, "/tmp/p".into()).unwrap().id;
        assert!(save_messages(&core, &kept, &[ChatMessage::system("sys")], &mut 0, false));
        assert_eq!(discard_if_empty(&core, &kept), Ok(false));
        assert!(lock(&kept).exists(), "a session that stays keeps its lock file");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The unlink is safe only because no claim can open the file before it
    /// and lock it after (see `SessionOwner`). Parked rather than raced, since
    /// a race that misses the bad interleaving proves nothing: while a
    /// deleter holds the store, a claim from another process must wait
    /// instead of opening the file.
    #[test]
    fn a_claim_cannot_interleave_with_the_unlink_of_its_lock_file() {
        let dir = std::env::temp_dir().join(format!("openmax-claim-{}", uuid::Uuid::new_v4()));
        let (deleter, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&deleter, "/tmp/p".into()).unwrap().id;
        attach(&deleter, &id).unwrap();

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let parked = {
            let deleter = deleter.clone();
            std::thread::spawn(move || {
                let _store = lock_store(&deleter).unwrap();
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
        };
        entered_rx.recv().unwrap();

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let claimer = {
            let (dir, id) = (dir.clone(), id.clone());
            std::thread::spawn(move || {
                let (other, _rx) = Core::new(dir).unwrap();
                let claimed = attach(&other, &id);
                done_tx.send(()).unwrap();
                claimed
            })
        };
        assert!(
            done_rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "a claim must wait while another process holds the store, or it can open \
             the lock file a delete is about to unlink"
        );
        release_tx.send(()).unwrap();
        parked.join().unwrap();
        let refused = claimer.join().unwrap().unwrap_err();
        assert!(refused.contains("another process"), "the deleter still owns it: {refused}");

        // With the file unlinked, claims that race for the name still end
        // with exactly one owner. Each core outlives the race, so no lock is
        // released early and handed to a later claim.
        delete(&deleter, &id).unwrap();
        let start = std::sync::Arc::new(std::sync::Barrier::new(4));
        let racers: Vec<_> = (0..4)
            .map(|_| {
                let (dir, id, start) = (dir.clone(), id.clone(), start.clone());
                std::thread::spawn(move || {
                    let (core, _rx) = Core::new(dir).unwrap();
                    start.wait();
                    let owned = attach(&core, &id).is_ok();
                    (owned, core)
                })
            })
            .collect();
        let results: Vec<_> = racers.into_iter().map(|r| r.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|(owned, _)| *owned).count(), 1, "one owner per session");
        drop(results);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A front end discards at exit, and a turn of its own can still be
    /// running then. Whether the session is empty is decided under the store
    /// lock every transcript write holds, so a save that lands while the
    /// discard waits for that lock keeps the session instead of being
    /// deleted with it.
    #[test]
    fn a_discard_keeps_a_session_that_gained_history_while_it_waited() {
        let dir = std::env::temp_dir().join(format!("openmax-discard-race-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        attach(&core, &id).unwrap();

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let writer = {
            let core = core.clone();
            std::thread::spawn(move || {
                let _store = lock_store(&core).unwrap();
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
        };
        entered_rx.recv().unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let discard = {
            let (core, id) = (core.clone(), id.clone());
            std::thread::spawn(move || {
                let discarded = discard_if_empty(&core, &id);
                done_tx.send(()).unwrap();
                discarded
            })
        };
        assert!(done_rx.recv_timeout(std::time::Duration::from_millis(300)).is_err());
        // The parked holder's save, landing while the discard waits.
        let line = serde_json::to_string(&ChatMessage::user("hello")).unwrap();
        std::fs::write(messages_path(&core, &id), format!("{line}\n")).unwrap();
        release_tx.send(()).unwrap();
        writer.join().unwrap();

        assert_eq!(discard.join().unwrap(), Ok(false), "a session with a transcript is history");
        assert_eq!(load_messages(&core, &id).unwrap().map(|m| m.len()), Some(1));
        assert!(list(&core, "/tmp/p").unwrap().iter().any(|m| m.id == id), "and it stays indexed");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The index lists every session of every project and is never pruned,
    /// so a write that reads it costs more with every session ever recorded.
    /// Once a claim has seen its session indexed, a model request's usage
    /// record, a transcript save, and a manifest save read none of it: no
    /// other process can remove a session this one holds, since a removal
    /// claims first. A removal by this process mid-turn is remembered, so the
    /// turn's late writes are still dropped, without reading the index either.
    #[test]
    fn a_steady_state_request_and_save_read_no_index_bytes() {
        let dir = std::env::temp_dir().join(format!("openmax-index-hot-path-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        for _ in 0..20 {
            create(&core, "/tmp/elsewhere".into()).unwrap();
        }
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        attach(&core, &id).unwrap();
        let read = || INDEX_BYTES_READ.with(|read| read.replace(0));
        read();

        let usage = |ts| TokenUsage { ts, prompt_tokens: 10, completion_tokens: 1, cached_tokens: None };
        let mut messages = vec![ChatMessage::system("rules"), ChatMessage::user("first")];
        let mut persisted = 0usize;
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        for turn in 0..3 {
            append_usage(&core, &id, &usage(turn));
            messages.push(ChatMessage::user(format!("message {turn}")));
            assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        }
        assert!(save_messages(&core, &id, &messages, &mut persisted, true), "a rewrite too");
        save_manifest(&core, &id, &crate::registry::Registry::builtin_only().to_manifest());
        assert_eq!(read(), 0, "a steady-state request or save read the session index");
        assert_eq!(load_usage(&core, &id).len(), 3);
        assert_eq!(load_messages(&core, &id).unwrap().unwrap().len(), messages.len());

        // Deleted while its turn runs, the session keeps its claim until the
        // turn settles, and what the turn still writes is dropped.
        core.running.lock().unwrap().insert(id.clone());
        delete(&core, &id).unwrap();
        read();
        append_usage(&core, &id, &usage(9));
        messages.push(ChatMessage::user("late"));
        assert!(save_messages(&core, &id, &messages, &mut persisted, false), "a deleted session's save is a silent no-op");
        save_manifest(&core, &id, &crate::registry::Registry::builtin_only().to_manifest());
        assert_eq!(read(), 0, "the deleted-session guard read the session index");
        for suffix in ["messages.json", "manifest.json", "usage.jsonl"] {
            assert!(!sessions_dir(&core).join(format!("{id}.{suffix}")).exists(), "{suffix} came back after delete");
        }
        core.running.lock().unwrap().remove(&id);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every turn start claims its session again (`attach`). Reparsing the
    /// whole transcript there made each turn cost more than the one before,
    /// though between turns an owned transcript changes only when something
    /// outside openmax writes it. So a turn start reads it again only once
    /// its length or modification time moved past what this process last
    /// validated or wrote, and damage written in between is still refused.
    #[test]
    fn a_turn_start_rereads_an_owned_transcript_only_after_it_changed() {
        let dir = std::env::temp_dir().join(format!("openmax-turn-start-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        attach(&core, &id).unwrap();
        let mut messages = vec![ChatMessage::system("rules"), ChatMessage::user("first")];
        let mut persisted = 0usize;
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        messages.push(ChatMessage::user("second"));
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        let reads = || TRANSCRIPT_READS.with(|reads| reads.replace(0));
        reads();

        attach(&core, &id).unwrap();
        attach(&core, &id).unwrap();
        assert_eq!(reads(), 0, "a turn start reparsed a transcript nothing had changed");

        let path = messages_path(&core, &id);
        let intact = std::fs::read(&path).unwrap();
        std::fs::write(&path, [intact.as_slice(), b"{damaged}\n"].concat()).unwrap();
        assert!(attach(&core, &id).unwrap_err().contains("damaged at line 4"));
        assert_eq!(reads(), 1);
        let edited = [intact.as_slice(), b"{\"role\":\"user\",\"content\":\"added outside\"}\n"].concat();
        std::fs::write(&path, edited).unwrap();
        attach(&core, &id).unwrap();
        attach(&core, &id).unwrap();
        assert_eq!(reads(), 1, "a repaired transcript is read once, then trusted again");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// An append does not read the history it extends, so it vouches for the
    /// file it leaves only when nothing else wrote it since this process last
    /// did. Damage written from outside during a turn, then appended past by
    /// that turn's own save, would otherwise be trusted by the next turn
    /// start and go unreported until a later sitting failed to load it.
    #[test]
    fn a_turn_start_rereads_a_transcript_written_from_outside_before_an_append() {
        let dir = std::env::temp_dir().join(format!("openmax-append-over-outside-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let id = create(&core, "/tmp/p".into()).unwrap().id;
        attach(&core, &id).unwrap();
        let mut messages = vec![ChatMessage::system("rules"), ChatMessage::user("first"), ChatMessage::user("second")];
        let mut persisted = 0usize;
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        let path = messages_path(&core, &id);
        let intact = std::fs::read(&path).unwrap();
        std::fs::write(&path, [intact.as_slice(), b"{damaged}\n"].concat()).unwrap();
        let reads = || TRANSCRIPT_READS.with(|reads| reads.replace(0));
        reads();

        messages.push(ChatMessage::user("third"));
        assert!(save_messages(&core, &id, &messages, &mut persisted, false));
        assert_eq!(reads(), 0, "the save appended rather than rewrote");
        let reason = attach(&core, &id).unwrap_err();
        assert!(reason.contains("damaged at line 4"), "{reason}");
        assert_eq!(reads(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The index keeps each project path as its frontend spelled it, while
    /// trust and history resolve it. A directory reached through a symlink,
    /// or moved and linked back, is still one project: its sessions list,
    /// and `--continue` takes the latest, from any spelling of the path.
    #[cfg(unix)]
    #[test]
    fn a_project_lists_its_sessions_from_any_spelling_of_its_path() {
        let dir = std::env::temp_dir().join(format!("openmax-spelling-{}", uuid::Uuid::new_v4()));
        let real = dir.join("real");
        let link = dir.join("link");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::create_dir_all(dir.join("other")).unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let (core, _rx) = Core::new(dir.join("data")).unwrap();
        let spell = |path: &Path| path.display().to_string();
        let through_link = create(&core, spell(&link)).unwrap().id;
        let direct = create(&core, spell(&real)).unwrap().id;
        create(&core, spell(&dir.join("other"))).unwrap();
        touch_at(&core, &through_link, 1_000);
        touch_at(&core, &direct, 2_000);

        for spelling in [real.clone(), link.clone(), std::fs::canonicalize(&real).unwrap()] {
            let listed: Vec<String> = list(&core, &spell(&spelling)).unwrap().into_iter().map(|m| m.id).collect();
            assert_eq!(listed, vec![direct.clone(), through_link.clone()], "{}", spelling.display());
        }
        touch_at(&core, &through_link, 3_000);
        assert_eq!(latest(&core, &spell(&real)).unwrap().unwrap().id, through_link);
        assert_eq!(
            std::fs::read_to_string(index_path(&core)).unwrap().matches(&spell(&link)).count(),
            1,
            "a lookup leaves every stored path as it was"
        );
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

        let original = crate::registry::Registry::build(&project.join("data"), &project);
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

        let loaded = load_messages(&core, id).unwrap().unwrap();
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
        let loaded = load_messages(&core, id).unwrap().unwrap();
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
