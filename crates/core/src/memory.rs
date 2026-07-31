//! Project memory: `.openmax/memory/<name>.md`, one durable fact per file,
//! written by the agent with the file tools it already has.
//!
//! The harness owns exactly three things here, all arithmetic: which memories
//! surface (an index line each in the frozen prompt, ranked by activation),
//! how activation is computed (recency and frequency of real use), and when a
//! memory is forgotten (deleted once its activation stays below a floor). The
//! content, the writing, and any deliberate recall beyond the index (grep,
//! read_file) belong to the agent. No database, no daemon, no embedding: the
//! directory is the memory, and forgetting is a feature - an index that only
//! ever grows becomes a prompt tax, and stale facts poison context.
//!
//! Activation is ACT-R's base-level learning rule, the rational-analysis fit
//! to human forgetting (Anderson & Schooler 1991): each past access at age
//! `t` hours contributes `t^-0.5`, and activation is the log of the sum, so
//! recency and frequency trade off in one number and one use of an old memory
//! revives it. Events come from the file's mtime plus an append-only access
//! log the turn loop feeds (reads and writes of memory paths by the file
//! tools). Everything is computed lazily at scan time from timestamps; there
//! is no background process to schedule or crash.

use std::io::Write as _;
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
const MAX_DESCRIPTION_CHARS: usize = 160;
/// Names are slugs so index lines, log lines, and paths stay unambiguous.
const MAX_NAME_CHARS: usize = 64;

/// A memory whose activation would equal one single access this many days
/// old drops out of the index: still on disk, still greppable, no longer
/// spending prompt bytes. ~3 weeks is where a single-shot human memory needs
/// a cue too.
const INDEX_FLOOR_DAYS: f64 = 21.0;
/// Below the activation of one access this many days old, the file itself is
/// deleted (a `gc` log line keeps its name, sha256, and description as the
/// tombstone). Memory is not an archive; the session transcripts are.
const GC_FLOOR_DAYS: f64 = 60.0;

/// ACT-R base-level activation: `ln(sum(t_hours^-d))` with d = 0.5. Ages are
/// clamped to one hour so a just-written memory contributes exactly 1.0 and
/// the power law starts after the first hour, not at a division by zero.
const DECAY_EXPONENT: f64 = 0.5;

fn activation(ages_hours: &[f64]) -> f64 {
    let sum: f64 = ages_hours.iter().map(|t| t.max(1.0).powf(-DECAY_EXPONENT)).sum();
    sum.ln()
}

/// The activation a single access of the given age would have; the floors are
/// defined through this so the constants above read in days, not in nepers.
fn floor_activation(days: f64) -> f64 {
    activation(&[days * 24.0])
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
    /// True when the entry made it into the injected index.
    pub in_index: bool,
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
/// is a reinforcement. Paths are project-relative as the prompt mandates.
pub fn access_of(tool: &str, path: &str) -> Option<(String, String)> {
    let rel = path.trim_start_matches("./");
    let name = rel.strip_prefix(".openmax/memory/")?.strip_suffix(".md")?;
    if !valid_name(name) {
        return None;
    }
    let kind = match tool {
        "read_file" => "read",
        "write_file" | "edit_file" => "write",
        _ => return None,
    };
    Some((name.to_string(), kind.to_string()))
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
pub fn scan(project_root: &Path, now: u64) -> MemoryScan {
    let dir = memory_dir(project_root);
    let Ok(read_dir) = std::fs::read_dir(&dir) else {
        return MemoryScan::default();
    };
    let log = load_log(project_root);
    let mut entries: Vec<MemoryEntry> = Vec::new();
    for entry in read_dir.flatten() {
        let path = entry.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("md") || !valid_name(name) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        // A file with no describable first line is skipped, not guessed at:
        // openmax --check names it and the fix (write a first line).
        let Some(description) = description_of(&text) else { continue };
        let mut ages: Vec<f64> = Vec::new();
        if let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) {
            let ts = mtime.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(now);
            ages.push(now.saturating_sub(ts) as f64 / 3600.0);
        }
        for record in log.iter().filter(|r| r.name == name && r.kind != "gc") {
            ages.push(now.saturating_sub(record.ts) as f64 / 3600.0);
        }
        if ages.is_empty() {
            ages.push(0.0);
        }
        entries.push(MemoryEntry {
            name: name.to_string(),
            description,
            path: format!("{MEMORY_DIR}/{name}.md"),
            activation: activation(&ages),
            in_index: false,
        });
    }
    entries.sort_by(|a, b| {
        b.activation.partial_cmp(&a.activation).unwrap_or(std::cmp::Ordering::Equal).then(a.name.cmp(&b.name))
    });

    // Greedy fill under the byte budget, faded entries never eligible.
    let index_floor = floor_activation(INDEX_FLOOR_DAYS);
    let mut spent = 0usize;
    let mut omitted = 0usize;
    for entry in entries.iter_mut() {
        let line = index_line(entry);
        if entry.activation >= index_floor && spent + line.len() <= MAX_MEMORY_BYTES {
            entry.in_index = true;
            spent += line.len();
        } else {
            omitted += 1;
        }
    }
    MemoryScan { entries, omitted }
}

pub fn index_line(entry: &MemoryEntry) -> String {
    format!("- {}: {} — {}\n", entry.name, entry.description, entry.path)
}

/// The injected index section, or None when nothing qualifies so the
/// zero-memory prompt stays byte-identical to a memoryless build.
pub fn index_section(project_root: &Path, now: u64) -> Option<(String, Vec<(String, usize)>)> {
    let scan = scan(project_root, now);
    let shown: Vec<&MemoryEntry> = scan.entries.iter().filter(|e| e.in_index).collect();
    if shown.is_empty() {
        return None;
    }
    let mut out = String::new();
    let mut breakdown = Vec::new();
    for entry in &shown {
        let line = index_line(entry);
        breakdown.push((entry.name.clone(), line.len()));
        out.push_str(&line);
    }
    if scan.omitted > 0 {
        out.push_str(&format!("… {} more (ls {MEMORY_DIR})\n", scan.omitted));
    }
    Some((out, breakdown))
}

/// Delete memories whose activation fell below the GC floor, logging a
/// tombstone (name, sha256, description) per deletion so what was forgotten
/// stays sayable even though the content is gone. Runs at session creation
/// only: never mid-session, never on resume, so a prune cannot yank a file
/// the live prompt still indexes.
pub fn forget_faded(project_root: &Path, now: u64) -> Vec<String> {
    let gc_floor = floor_activation(GC_FLOOR_DAYS);
    let scan = scan(project_root, now);
    let mut forgotten = Vec::new();
    for entry in scan.entries.iter().filter(|e| e.activation < gc_floor) {
        let path = project_root.join(&entry.path);
        let Ok(bytes) = std::fs::read(&path) else { continue };
        if std::fs::remove_file(&path).is_err() {
            continue;
        }
        let record = AccessRecord {
            name: entry.name.clone(),
            ts: now,
            kind: "gc".into(),
            sha256: Some(crate::ledger::sha256_hex(&bytes)),
            description: Some(entry.description.clone()),
        };
        if let Ok(line) = serde_json::to_string(&record) {
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path(project_root))
                .and_then(|mut f| writeln!(f, "{line}"));
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
        // Backdate old-fact's mtime signal by logging nothing and pushing the
        // file into the past via an old logged write plus filetime-free mtime:
        // mtime is now for both, so use log weight to rank fresh-fact higher.
        log_access(&root, "fresh-fact", now - HOUR, "read");
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
        let gc_now = now + 90 * DAY;
        let forgotten = forget_faded(&root, gc_now);
        assert_eq!(forgotten, vec!["stale-fact".to_string()]);
        assert!(!root.join(MEMORY_DIR).join("stale-fact.md").exists());
        assert!(
            root.join(MEMORY_DIR).join("live-fact.md").exists(),
            "the revived memory survives GC"
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

    #[test]
    fn index_section_respects_byte_budget_and_reports_omissions() {
        let root = temp_project();
        let now = unix_now();
        for i in 0..40 {
            write_memory(&root, &format!("fact-{i:02}"), &format!("# {}", "d".repeat(120)));
        }
        let (section, breakdown) = index_section(&root, now).expect("index exists");
        assert!(section.len() <= MAX_MEMORY_BYTES + 64, "section stays near budget");
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
        assert_eq!(
            access_of("read_file", ".openmax/memory/deploy-port.md"),
            Some(("deploy-port".into(), "read".into()))
        );
        assert_eq!(
            access_of("edit_file", "./.openmax/memory/deploy-port.md"),
            Some(("deploy-port".into(), "write".into()))
        );
        assert_eq!(access_of("read_file", "src/main.rs"), None);
        assert_eq!(access_of("bash", ".openmax/memory/deploy-port.md"), None);
        assert_eq!(access_of("read_file", ".openmax/memory/.access.jsonl"), None);
        assert_eq!(access_of("read_file", ".openmax/memory/Bad.md"), None);
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
