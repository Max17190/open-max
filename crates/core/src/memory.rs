//! Project memory: `.openmax/memory/<name>.md`, one durable fact per file,
//! written by the agent with the file tools it already has.
//!
//! The harness owns exactly three things here, all arithmetic: which memories
//! surface (an index line each in the frozen prompt, ranked by activation),
//! how activation is computed (recency and frequency of real use), and when a
//! memory is forgotten (deleted once it goes unused past a floor age). The
//! content, the writing, and any deliberate recall beyond the index (grep,
//! read_file) belong to the agent. No database, no daemon, no embedding: the
//! directory is the memory, and forgetting is a feature - an index that only
//! ever grows becomes a prompt tax, and stale facts poison context.
//!
//! Activation is ACT-R's base-level learning rule, the rational-analysis fit
//! to human forgetting (Anderson & Schooler 1991): each past access at age
//! `t` hours contributes `t^-0.5`, and activation is the log of the sum, so
//! recency and frequency trade off in one number and one use of an old memory
//! revives it. Activation only ranks, though; the floors gate on the age of
//! the most recent use, because the power-law sum lets `n` accesses outlast a
//! single-access activation floor for `n^2` floor-ages, and a fact leaned on
//! hourly for a week would otherwise spend decades in the index. Events come
//! from the file's mtime plus an append-only access log the turn loop feeds
//! (reads and writes of memory paths by the file tools). Everything is
//! computed lazily at scan time from timestamps; there is no background
//! process to schedule or crash.

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Project-relative home of memory files; global memories deliberately do not
/// exist (a fact useful across projects belongs in the project that proves it).
pub const MEMORY_DIR: &str = ".openmax/memory";
/// The access log lives beside the memories it scores, agent-inspectable like
/// everything else: `grep` it to see what you actually recall.
const ACCESS_LOG: &str = ".access.jsonl";

/// Index budget in the frozen prompt. Skills pay ~3000 bytes for capability
/// discovery; memory pays half that for fact discovery, and the bodies load
/// on demand via read_file.
pub const MAX_MEMORY_BYTES: usize = 1_500;
/// One line per memory: past this the first line is a summary, not an essay.
pub(crate) const MAX_DESCRIPTION_CHARS: usize = 160;
/// Names are slugs so index lines, log lines, and paths stay unambiguous.
const MAX_NAME_CHARS: usize = 64;

/// A memory whose most recent use is older than this many days drops out of
/// the index: still on disk, still greppable, no longer spending prompt
/// bytes. ~3 weeks without a use is where a human memory needs a cue again,
/// however well-worn it once was.
pub(crate) const INDEX_FLOOR_DAYS: f64 = 21.0;
/// Unused past this many days, the file itself is deleted (a `gc` log line
/// keeps its name, sha256, and description as the tombstone). Memory is not
/// an archive; the session transcripts are.
pub(crate) const GC_FLOOR_DAYS: f64 = 60.0;

/// ACT-R base-level activation: `ln(sum(t_hours^-d))` with d = 0.5. Ages are
/// clamped to one hour so a just-written memory contributes exactly 1.0 and
/// the power law starts after the first hour, not at a division by zero.
const DECAY_EXPONENT: f64 = 0.5;

fn activation(ages_hours: &[f64]) -> f64 {
    let sum: f64 = ages_hours.iter().map(|t| t.max(1.0).powf(-DECAY_EXPONENT)).sum();
    sum.ln()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccessRecord {
    pub name: String,
    pub ts: u64,
    /// `read` | `write` | `gc`
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MemoryEntry {
    pub name: String,
    pub description: String,
    /// Project-relative path, as the index line shows it.
    pub path: String,
    pub activation: f64,
    /// Age in hours of the most recent event; the index and GC floors gate
    /// on this, not on activation, so frequency raises rank, never lifespan.
    pub last_event_hours: u64,
    /// True when the entry made it into the injected index.
    pub in_index: bool,
}

impl MemoryEntry {
    /// True when the first line was longer than the index cap, so the line a
    /// future session reads is a cut of what the author wrote. Read off the
    /// description the scan already built (past the cap it is exactly the cap
    /// plus the ellipsis), never a second read of the file.
    pub fn description_clipped(&self) -> bool {
        self.description.chars().count() > MAX_DESCRIPTION_CHARS
    }
}

#[derive(Clone, Debug, Default)]
pub struct MemoryScan {
    /// Every valid memory on disk, strongest first.
    pub entries: Vec<MemoryEntry>,
    /// Valid memories the index excluded (faded or over the byte budget).
    pub omitted: usize,
}

fn memory_dir(project_root: &Path) -> PathBuf {
    project_root.join(MEMORY_DIR)
}

fn log_path(project_root: &Path) -> PathBuf {
    memory_dir(project_root).join(ACCESS_LOG)
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_CHARS
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// First non-empty line, leading markdown heading markers stripped, truncated
/// on a char boundary. The description is the memory's entire cost in future
/// prompts, so it is held to one line no matter what the body does.
fn description_of(text: &str) -> Option<String> {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    // The first line is author-controlled body text, and `str::lines` splits
    // only on `\n`: an interior carriage return, escape, or line separator
    // would otherwise ride the memory index line (one per fact) in the frozen
    // prompt and the receipt. Flatten it to one line, then strip a leading
    // markdown heading marker.
    let line = crate::text::one_line(line);
    let line = line.trim_start_matches('#').trim();
    if line.is_empty() {
        return None;
    }
    let mut out: String = line.chars().take(MAX_DESCRIPTION_CHARS).collect();
    if line.chars().count() > MAX_DESCRIPTION_CHARS {
        out.push('…');
    }
    Some(out)
}

/// Append access events, one line each, deduplicated by (name, kind) within
/// the call. Best-effort: memory scoring must never fail a turn.
pub fn record_accesses(project_root: &Path, events: &[(String, String)]) {
    if events.is_empty() {
        return;
    }
    let dir = memory_dir(project_root);
    if !dir.is_dir() {
        return;
    }
    let ts = unix_now();
    let mut lines = String::new();
    let mut seen: Vec<(&str, &str)> = Vec::new();
    for (name, kind) in events {
        if !valid_name(name) || seen.iter().any(|(n, k)| n == name && k == kind) {
            continue;
        }
        seen.push((name, kind));
        let record =
            AccessRecord { name: name.clone(), ts, kind: kind.clone(), sha256: None, description: None };
        if let Ok(line) = serde_json::to_string(&record) {
            lines.push_str(&line);
            lines.push('\n');
        }
    }
    if lines.is_empty() {
        return;
    }
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path(project_root))
        .and_then(|mut f| f.write_all(lines.as_bytes()));
}

/// The file tools' view of a call against the memory directory, for the turn
/// loop: `read_file` of a memory path is a recall, `write_file`/`edit_file`
/// is a reinforcement. Paths are matched the way the tools resolve them -
/// `subdir/../.openmax/memory/x.md` and an absolute path inside the project
/// are the same access - or an actively used memory would starve and fade.
pub fn access_of(tool: &str, path: &str, project_root: &Path) -> Option<(String, String)> {
    let kind = match tool {
        "read_file" => "read",
        "write_file" | "edit_file" => "write",
        _ => return None,
    };
    let normalized = normalize(path, project_root)?;
    let name = normalized.strip_prefix(".openmax/memory/")?.strip_suffix(".md")?;
    if !valid_name(name) {
        return None;
    }
    Some((name.to_string(), kind.to_string()))
}

/// Lexically resolve `.`/`..` and strip an absolute project prefix, without
/// touching the filesystem: the dispatcher classifies calls after the tool
/// already ran, it never re-resolves them. A path that walks above its root
/// is not a memory access (the tools reject it anyway).
fn normalize(path: &str, project_root: &Path) -> Option<String> {
    let p = Path::new(path);
    let rel = p.strip_prefix(project_root).unwrap_or(p);
    let mut parts: Vec<&std::ffi::OsStr> = Vec::new();
    for comp in rel.components() {
        match comp {
            std::path::Component::Normal(s) => parts.push(s),
            std::path::Component::ParentDir => {
                parts.pop()?;
            }
            std::path::Component::CurDir => {}
            _ => return None,
        }
    }
    let joined: PathBuf = parts.iter().collect();
    joined.to_str().map(str::to_string)
}

fn load_log(project_root: &Path) -> Vec<AccessRecord> {
    let Ok(text) = std::fs::read_to_string(log_path(project_root)) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Scan the memory directory and score every valid entry, strongest first
/// (ties by name for determinism). Pure with respect to the filesystem: no
/// writes, no deletions - `forget_faded` is the explicit destructive step.
/// Score one memory file into an index entry from ALREADY-READ content, or
/// None if it has no describable first line. Kept separate from the directory
/// walk so the fingerprint scan and the index scan can share ONE read of each
/// file: two independent reads could otherwise freeze one generation of a
/// file's index under another generation's fingerprint (Greptile).
fn entry_from(
    name: &str,
    text: &str,
    mtime: Option<std::time::SystemTime>,
    log: &[AccessRecord],
    now: u64,
) -> Option<MemoryEntry> {
    // A file with no describable first line is skipped, not guessed at:
    // openmax --check names it and the fix (write a first line).
    let description = description_of(text)?;
    // One physical act, one event: a write_file produces both an mtime and a
    // logged write at the same instant, and summing them would add a permanent
    // ln(2) of activation, outranking every honest single-event peer. Ages
    // bucket to whole hours (matching the one-hour clamp in `activation`), and
    // buckets deduplicate.
    let mut hour_buckets: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    if let Some(mtime) = mtime {
        let ts = mtime.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(now);
        hour_buckets.insert((now.saturating_sub(ts) / 3600).max(1));
    }
    // A gc tombstone is a horizon: a name reused after deletion starts fresh
    // instead of inheriting the dead namesake's access history.
    let horizon = log.iter().filter(|r| r.name == name && r.kind == "gc").map(|r| r.ts).max();
    for record in
        log.iter().filter(|r| r.name == name && r.kind != "gc" && horizon.is_none_or(|h| r.ts > h))
    {
        hour_buckets.insert((now.saturating_sub(record.ts) / 3600).max(1));
    }
    if hour_buckets.is_empty() {
        hour_buckets.insert(1);
    }
    let last_event_hours = hour_buckets.first().copied().unwrap_or(1);
    let ages: Vec<f64> = hour_buckets.into_iter().map(|h| h as f64).collect();
    Some(MemoryEntry {
        name: name.to_string(),
        description,
        path: format!("{MEMORY_DIR}/{name}.md"),
        activation: activation(&ages),
        last_event_hours,
        in_index: false,
    })
}

/// Rank entries strongest-first and greedily fill the index under the byte
/// budget, evicting the weakest surfaced entries until the lines plus the
/// omission trailer fit the cap.
fn fill_index(mut entries: Vec<MemoryEntry>) -> MemoryScan {
    entries.sort_by(|a, b| {
        b.activation.partial_cmp(&a.activation).unwrap_or(std::cmp::Ordering::Equal).then(a.name.cmp(&b.name))
    });
    // Entries unused past the floor age are never eligible, however high their
    // activation ranks them.
    let index_floor_hours = INDEX_FLOOR_DAYS * 24.0;
    let mut spent = 0usize;
    let mut omitted = 0usize;
    for entry in entries.iter_mut() {
        let line = index_line(entry);
        if entry.last_event_hours as f64 <= index_floor_hours
            && spent + line.len() <= MAX_MEMORY_BYTES
        {
            entry.in_index = true;
            spent += line.len();
        } else {
            omitted += 1;
        }
    }
    // The omission trailer spends the same budget the lines do: evict the
    // weakest surfaced entries until lines plus trailer fit the cap, so the
    // documented budget is what the prompt actually pays.
    while omitted > 0 && spent + trailer_line(omitted).len() > MAX_MEMORY_BYTES {
        let Some(last_in) = entries.iter_mut().rev().find(|e| e.in_index) else { break };
        let line_len = index_line(last_in).len();
        last_in.in_index = false;
        spent -= line_len;
        omitted += 1;
    }
    MemoryScan { entries, omitted }
}

pub fn scan(project_root: &Path, now: u64) -> MemoryScan {
    let Ok(read_dir) = std::fs::read_dir(memory_dir(project_root)) else {
        return MemoryScan::default();
    };
    let log = load_log(project_root);
    let mut entries: Vec<MemoryEntry> = Vec::new();
    for dirent in read_dir.flatten() {
        let path = dirent.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("md") || !valid_name(name) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let mtime = dirent.metadata().and_then(|m| m.modified()).ok();
        if let Some(entry) = entry_from(name, &text, mtime, &log, now) {
            entries.push(entry);
        }
    }
    fill_index(entries)
}

/// One read of the memory directory producing BOTH the fingerprint bytes and
/// the index selection from the SAME bytes, so a file replaced between two
/// separate scans can no longer freeze one generation's index under another
/// generation's fingerprint (Greptile). The fingerprint set is every
/// valid-named `.md` (a write to any refreezes); the index is the describable,
/// unfaded, in-budget subset.
pub struct MemoryFreeze {
    pub fingerprint_files: Vec<(PathBuf, Vec<u8>)>,
    pub section: IndexSection,
    pub identities: Vec<(String, u64)>,
}

/// Read a memory file's bytes and its mtime as ONE coherent generation. An
/// in-place write to the inode can replace the bytes between reading the mtime
/// and the body (either order), pairing one generation's content with
/// another's timestamp. Read mtime, then bytes, then mtime again; if it moved,
/// the file changed mid-read - retry so the pair always matches (Greptile).
/// Bounded: on persistent churn the last read is self-consistent bytes with
/// the mtime observed immediately after them.
fn read_coherent(path: &Path) -> Option<(Vec<u8>, Option<std::time::SystemTime>)> {
    for _ in 0..8 {
        let mut file = std::fs::File::open(path).ok()?;
        let before = file.metadata().and_then(|m| m.modified()).ok();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).ok()?;
        let after = file.metadata().and_then(|m| m.modified()).ok();
        if before == after {
            return Some((bytes, after));
        }
    }
    // The file is being rewritten faster than it can be read coherently
    // (pathological sub-read churn): SKIP it this freeze rather than return an
    // incoherent body/mtime pair (Greptile). It re-enters the fingerprint and
    // the index on the next freeze once writes settle - and because it was
    // absent from this fingerprint, that settling changes the fingerprint and
    // triggers the refreeze that indexes it.
    None
}

pub fn freeze_snapshot(project_root: &Path, now: u64) -> MemoryFreeze {
    let Ok(read_dir) = std::fs::read_dir(memory_dir(project_root)) else {
        return MemoryFreeze {
            fingerprint_files: Vec::new(),
            section: None,
            identities: Vec::new(),
        };
    };
    let log = load_log(project_root);
    let mut fingerprint_files: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    let mut entries: Vec<MemoryEntry> = Vec::new();
    for dirent in read_dir.flatten() {
        let path = dirent.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("md") || !valid_name(name) {
            continue;
        }
        // Bytes AND mtime describe the SAME generation of the file, so the
        // fingerprint (which hashes the bytes) and the index scoring (which
        // uses the mtime) can never be captured from two different generations
        // - a rename OR an in-place rewrite during the read would otherwise
        // pair one generation's bytes with another's mtime (Greptile).
        let Some((bytes, mtime)) = read_coherent(&path) else { continue };
        // The SAME bytes feed the fingerprint and the index: a describable,
        // UTF-8 file also enters the index; every valid-named file counts
        // toward the fingerprint regardless.
        if let Ok(text) = std::str::from_utf8(&bytes) {
            if let Some(entry) = entry_from(name, text, mtime, &log, now) {
                entries.push(entry);
            }
        }
        fingerprint_files.push((path, bytes));
    }
    fingerprint_files.sort_by(|a, b| a.0.cmp(&b.0));
    let scan = fill_index(entries);
    let (section, identities) = section_and_identities(&scan);
    MemoryFreeze { fingerprint_files, section, identities }
}

fn trailer_line(omitted: usize) -> String {
    format!("… {omitted} more (ls {MEMORY_DIR})\n")
}

pub fn index_line(entry: &MemoryEntry) -> String {
    format!("- {}: {} — {}\n", entry.name, entry.description, entry.path)
}

/// One scan producing BOTH the rendered index section (for the frozen prompt)
/// AND the receipt identities (for the refreeze receipt), so the two can
/// never disagree - a memory changed between two separate scans could make
/// the receipt claim a fact live that the next prompt then omits (Greptile).
/// (rendered index section, per-name byte breakdown) or None when empty.
pub type IndexSection = Option<(String, Vec<(String, usize)>)>;

/// The rendered index section and receipt identities for a completed scan.
/// Shared by `index_and_identities` (fresh scan) and `freeze_snapshot`
/// (fingerprint + index in one read) so all three outputs describe the same
/// selection.
fn section_and_identities(scan: &MemoryScan) -> (IndexSection, Vec<(String, u64)>) {
    use std::hash::{Hash, Hasher};
    let shown: Vec<&MemoryEntry> = scan.entries.iter().filter(|e| e.in_index).collect();
    let identities: Vec<(String, u64)> = shown
        .iter()
        .map(|e| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            index_line(e).hash(&mut h);
            (e.name.clone(), h.finish())
        })
        .collect();
    let section = if shown.is_empty() {
        None
    } else {
        let mut out = String::new();
        let mut breakdown = Vec::new();
        for entry in &shown {
            let line = index_line(entry);
            breakdown.push((entry.name.clone(), line.len()));
            out.push_str(&line);
        }
        if scan.omitted > 0 {
            out.push_str(&trailer_line(scan.omitted));
        }
        Some((out, breakdown))
    };
    (section, identities)
}

/// A fresh scan's section and identities, for tests that compare a
/// snapshot against what a plain scan would render.
#[cfg(test)]
fn index_and_identities(project_root: &Path, now: u64) -> (IndexSection, Vec<(String, u64)>) {
    section_and_identities(&scan(project_root, now))
}

/// The injected index section from a fresh scan, or None when nothing
/// qualifies so the zero-memory prompt stays byte-identical to a memoryless
/// build. The same renderer `freeze_snapshot` uses, so a registry that never
/// scanned renders exactly what a freeze would have.
pub fn index_section(project_root: &Path, now: u64) -> IndexSection {
    section_and_identities(&scan(project_root, now)).0
}

/// Delete memories unused past the GC floor age, logging a tombstone (name,
/// sha256, description) per deletion so what was forgotten stays sayable
/// even though the content is gone. Runs at session creation only: never
/// mid-session, never on resume, so a prune cannot yank a file the live
/// prompt still indexes.
pub fn forget_faded(project_root: &Path, now: u64) -> Vec<String> {
    let gc_floor_hours = GC_FLOOR_DAYS * 24.0;
    let scan = scan(project_root, now);
    let mut forgotten = Vec::new();
    for entry in scan.entries.iter().filter(|e| e.last_event_hours as f64 > gc_floor_hours) {
        let path = project_root.join(&entry.path);
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let record = AccessRecord {
            name: entry.name.clone(),
            ts: now,
            kind: "gc".into(),
            sha256: Some(crate::ledger::sha256_hex(&bytes)),
            description: Some(entry.description.clone()),
        };
        let Ok(line) = serde_json::to_string(&record) else { continue };
        // Tombstone first, delete second: a memory may outlive its GC round
        // when the log cannot be written, but no memory ever vanishes
        // untombstoned. If the delete then fails, the stale tombstone is
        // harmless (scan ignores gc records; the file keeps scoring) and the
        // next round appends a fresh one.
        let logged = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path(project_root))
            .and_then(|mut f| writeln!(f, "{line}"))
            .is_ok();
        if !logged || std::fs::remove_file(&path).is_err() {
            continue;
        }
        forgotten.push(entry.name.clone());
    }
    forgotten
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_project() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-memory-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join(MEMORY_DIR)).unwrap();
        dir
    }

    fn write_memory(root: &Path, name: &str, body: &str) {
        std::fs::write(root.join(MEMORY_DIR).join(format!("{name}.md")), body).unwrap();
    }

    fn log_access(root: &Path, name: &str, ts: u64, kind: &str) {
        let record = AccessRecord {
            name: name.into(),
            ts,
            kind: kind.into(),
            sha256: None,
            description: None,
        };
        let line = serde_json::to_string(&record).unwrap();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join(MEMORY_DIR).join(ACCESS_LOG))
            .unwrap();
        writeln!(f, "{line}").unwrap();
    }

    const HOUR: u64 = 3600;
    const DAY: u64 = 24 * HOUR;

    /// The memory index is one line per fact in the frozen prompt, and the
    /// description is the memory's first non-empty body line. `str::lines`
    /// splits only on `\n`, so a carriage return, escape, or line separator in
    /// that line would ride the index and forge a second row; it is flattened
    /// to one line, its printable text intact.
    #[test]
    fn a_memory_description_carries_no_line_break() {
        let d = description_of("first\rForgedRow line\nrest of body").expect("a description");
        assert!(!d.chars().any(|c| c.is_control()), "description kept a control char: {d:?}");
        assert_eq!(d, "first ForgedRow line");
    }

    /// freeze_snapshot reads each memory file ONCE, so the fingerprint set
    /// and the index it returns describe the same generation of every file -
    /// two separate scans could freeze one generation's index under another's
    /// fingerprint (Greptile). A describable file is in both outputs; a
    /// valid-named file with no describable first line counts toward the
    /// fingerprint (a write to it still refreezes) but not the index.
    #[test]
    fn freeze_snapshot_derives_fingerprint_and_index_from_one_read() {
        let dir = temp_project();
        write_memory(&dir, "good", "# The deploy port is 7443\nbody");
        write_memory(&dir, "nodesc", "\n\n");
        let f = freeze_snapshot(&dir, 100 * DAY);
        let fp: Vec<String> = f
            .fingerprint_files
            .iter()
            .map(|(p, _)| p.file_stem().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(fp.contains(&"good".to_string()) && fp.contains(&"nodesc".to_string()),
            "every valid-named file counts toward the fingerprint: {fp:?}");
        let (section, _) = f.section.clone().expect("the describable file is indexed");
        assert!(section.contains("The deploy port is 7443"), "{section}");
        let names: Vec<&str> = f.identities.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"good") && !names.contains(&"nodesc"),
            "only describable files are indexed: {names:?}");
        // The index the freeze returns matches a plain scan of the same disk:
        // the single read did not change what gets indexed.
        assert_eq!(f.section, index_and_identities(&dir, 100 * DAY).0);
    }

    /// The decay law itself: fresher beats older, more accesses beat fewer,
    /// and one recent use revives an old memory past a fresher-but-unused one.
    #[test]
    fn activation_is_monotone_in_recency_and_frequency() {
        assert!(activation(&[1.0]) > activation(&[100.0]));
        assert!(activation(&[100.0, 90.0]) > activation(&[100.0]));
        let revived_old = activation(&[24.0 * 50.0, 2.0]);
        let unused_newer = activation(&[24.0 * 10.0]);
        assert!(revived_old > unused_newer, "one recent recall must outweigh mere newness");
        // Ages under an hour clamp to the one-hour contribution.
        assert_eq!(activation(&[0.0]), activation(&[1.0]));
    }

    #[test]
    fn scan_ranks_by_activation_and_caps_the_index() {
        let root = temp_project();
        let now = unix_now();
        write_memory(&root, "fresh-fact", "# The deploy port is 7443\ndetails");
        write_memory(&root, "old-fact", "# Old decision nobody reads");
        // Both files share a fresh mtime; a distinct-hour logged read gives
        // fresh-fact a second event, which frequency must reward.
        log_access(&root, "fresh-fact", now - 2 * HOUR, "read");
        let scan = scan(&root, now);
        assert_eq!(scan.entries.len(), 2);
        assert_eq!(scan.entries[0].name, "fresh-fact");
        assert!(scan.entries[0].activation > scan.entries[1].activation);
        assert!(scan.entries.iter().all(|e| e.in_index), "both are fresh enough");
        assert_eq!(scan.entries[0].description, "The deploy port is 7443");

        // Files that are not valid memories are skipped entirely.
        std::fs::write(root.join(MEMORY_DIR).join("Bad Name.md"), "# x").unwrap();
        std::fs::write(root.join(MEMORY_DIR).join("notes.txt"), "# x").unwrap();
        write_memory(&root, "empty", "\n\n");
        assert_eq!(scan_len(&root, now), 2);
        let _ = std::fs::remove_dir_all(root);
    }

    fn scan_len(root: &Path, now: u64) -> usize {
        scan(root, now).entries.len()
    }

    /// One physical act must be one event: a write_file leaves both an mtime
    /// and a logged write at the same instant, and counting both would add a
    /// permanent ln(2) of activation, outranking every honest single-event
    /// peer. Found by the wire-level lifecycle eval, pinned here.
    #[test]
    fn same_hour_signals_collapse_to_one_event() {
        let root = temp_project();
        let now = unix_now();
        write_memory(&root, "written-once", "# a fact");
        write_memory(&root, "control", "# another fact");
        log_access(&root, "written-once", now, "write");
        let scan = scan(&root, now);
        let written = scan.entries.iter().find(|e| e.name == "written-once").unwrap();
        let control = scan.entries.iter().find(|e| e.name == "control").unwrap();
        assert_eq!(
            written.activation, control.activation,
            "an mtime and its logged write must not double-count"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// mtime alone cannot fade (editing the file is a use); fading needs the
    /// clock to outrun every event, which tests simulate with old log entries
    /// and a backdated file time via the log-only scoring of a deleted mtime.
    #[test]
    fn faded_memories_leave_the_index_and_gc_deletes_with_tombstone() {
        let root = temp_project();
        let now = unix_now();
        write_memory(&root, "stale-fact", "# A fact from another era");
        write_memory(&root, "live-fact", "# Still in use");
        // Simulate age: score at a future `now` where the only events (mtime
        // ~today, no log) are 30 and 90 days old respectively per entry.
        let idx_now = now + 30 * DAY;
        let scan30 = scan(&root, idx_now);
        assert!(
            scan30.entries.iter().all(|e| !e.in_index),
            "a 30-day-unused memory must fade from the index (floor {INDEX_FLOOR_DAYS} days)"
        );
        assert_eq!(scan30.omitted, 2);
        // A read revives it.
        log_access(&root, "live-fact", idx_now - HOUR, "read");
        let revived = scan(&root, idx_now);
        let live = revived.entries.iter().find(|e| e.name == "live-fact").unwrap();
        assert!(live.in_index, "one recall must restore index presence");

        // 90 days out, the unread one crosses the GC floor and is deleted.
        // The revival above is itself 60 days stale by then, so live-fact
        // needs a use inside the GC window: staleness is measured from the
        // last use, not bought off by history.
        let gc_now = now + 90 * DAY;
        log_access(&root, "live-fact", gc_now - DAY, "read");
        let forgotten = forget_faded(&root, gc_now);
        assert_eq!(forgotten, vec!["stale-fact".to_string()]);
        assert!(!root.join(MEMORY_DIR).join("stale-fact.md").exists());
        assert!(
            root.join(MEMORY_DIR).join("live-fact.md").exists(),
            "the recently used memory survives GC"
        );
        let tombstone = load_log(&root)
            .into_iter()
            .find(|r| r.kind == "gc")
            .expect("gc must log a tombstone");
        assert_eq!(tombstone.name, "stale-fact");
        assert!(tombstone.sha256.is_some());
        assert_eq!(tombstone.description.as_deref(), Some("A fact from another era"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// Frequency raises rank, never lifespan: a memory read hourly for most
    /// of a week and then abandoned must fade on the same clock as one read
    /// once. Against a single-access activation floor, 40 accesses would
    /// outlast the floor for 40^2 floor-ages (~92 years in the index,
    /// centuries on disk).
    #[test]
    fn a_frequent_memory_still_fades_once_it_goes_stale() {
        let root = temp_project();
        let now = unix_now();
        write_memory(&root, "hot-then-dropped", "# A fact leaned on hourly, then abandoned");
        for i in 0..40u64 {
            log_access(&root, "hot-then-dropped", now - i * HOUR, "read");
        }
        let stale_now = now + 180 * DAY;
        let stale = scan(&root, stale_now);
        let entry = stale.entries.iter().find(|e| e.name == "hot-then-dropped").unwrap();
        assert!(
            !entry.in_index,
            "40 accesses gone {INDEX_FLOOR_DAYS}+ days stale must leave the index \
             (activation {})",
            entry.activation
        );
        let forgotten = forget_faded(&root, stale_now);
        assert_eq!(forgotten, vec!["hot-then-dropped".to_string()], "and GC agrees: stale is stale");
        let _ = std::fs::remove_dir_all(root);
    }

    /// A name reused after GC starts from zero: the tombstone is a horizon,
    /// and the dead namesake's access history stays on its side of it.
    #[test]
    fn a_reborn_name_does_not_inherit_the_dead_namesakes_history() {
        let root = temp_project();
        let now = unix_now();
        // The dead namesake: twenty reads, then its gc tombstone.
        for i in 0..20u64 {
            log_access(&root, "fact", now - 40 * DAY - i * HOUR, "read");
        }
        log_access(&root, "fact", now - 30 * DAY, "gc");
        // The reborn file and an untouched control, written the same instant.
        write_memory(&root, "fact", "# A new fact under an old name");
        write_memory(&root, "control", "# Fresh either way");
        let scan = scan(&root, now);
        let reborn = scan.entries.iter().find(|e| e.name == "fact").unwrap();
        let control = scan.entries.iter().find(|e| e.name == "control").unwrap();
        assert_eq!(
            reborn.activation, control.activation,
            "a reborn name must score like the fresh write it is"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn index_section_respects_byte_budget_and_reports_omissions() {
        let root = temp_project();
        let now = unix_now();
        for i in 0..40 {
            write_memory(&root, &format!("fact-{i:02}"), &format!("# {}", "d".repeat(120)));
        }
        let (section, breakdown) = index_section(&root, now).expect("index exists");
        assert!(
            section.len() <= MAX_MEMORY_BYTES,
            "lines plus trailer must fit the documented budget, got {}",
            section.len()
        );
        assert!(breakdown.len() < 40, "not everything fits");
        assert!(section.contains("more (ls .openmax/memory)"), "{section}");
        let shown_bytes: usize = breakdown.iter().map(|(_, b)| b).sum();
        assert!(shown_bytes <= MAX_MEMORY_BYTES);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn no_memories_means_no_section() {
        let root = temp_project();
        assert!(index_section(&root, unix_now()).is_none());
        let empty = std::env::temp_dir().join(format!("openmax-nomem-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&empty).unwrap();
        assert!(index_section(&empty, unix_now()).is_none(), "no dir, no section");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn tool_calls_map_to_access_events() {
        let root = Path::new("/proj");
        let read = Some(("deploy-port".to_string(), "read".to_string()));
        let write = Some(("deploy-port".to_string(), "write".to_string()));
        assert_eq!(access_of("read_file", ".openmax/memory/deploy-port.md", root), read);
        assert_eq!(access_of("edit_file", "./.openmax/memory/deploy-port.md", root), write);
        // The tools resolve dot segments and project-absolute forms to the
        // same file; classification must too, or a used memory starves.
        assert_eq!(access_of("read_file", "subdir/../.openmax/memory/deploy-port.md", root), read);
        assert_eq!(access_of("read_file", "/proj/.openmax/memory/deploy-port.md", root), read);
        assert_eq!(access_of("read_file", "../elsewhere/.openmax/memory/deploy-port.md", root), None);
        assert_eq!(access_of("read_file", "/other/.openmax/memory/deploy-port.md", root), None);
        assert_eq!(access_of("read_file", "src/main.rs", root), None);
        assert_eq!(access_of("bash", ".openmax/memory/deploy-port.md", root), None);
        assert_eq!(access_of("read_file", ".openmax/memory/.access.jsonl", root), None);
        assert_eq!(access_of("read_file", ".openmax/memory/Bad.md", root), None);
    }

    #[test]
    fn record_accesses_dedupes_and_survives_reload() {
        let root = temp_project();
        write_memory(&root, "fact", "# f");
        record_accesses(
            &root,
            &[
                ("fact".into(), "read".into()),
                ("fact".into(), "read".into()),
                ("fact".into(), "write".into()),
            ],
        );
        let log = load_log(&root);
        assert_eq!(log.len(), 2, "same (name, kind) collapses within a turn");
        let _ = std::fs::remove_dir_all(root);
    }
}
