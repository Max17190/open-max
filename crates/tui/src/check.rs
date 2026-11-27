//! `--second-look`: an independent check of a print run's finished work.
//!
//! When the run's one turn ends with status 0, a second session in the same
//! project checks the work and may not change it. Its first turn plans the
//! checks from the request alone; its second sees the paths the author
//! changed, a capped diff and the author's final message, runs the checks and
//! appends its objections to a file. Then, with no model call, the harness
//! stops every process the checker started, puts every file of the project
//! back as the author left it, drops each objection that does not quote the
//! request or does not fail again when the harness runs its command itself,
//! and hands what is left to the author's own session once, with the output
//! the harness just got. The author's second end stands, and the run exits
//! with the author's status. A signal or a closed output stops the check as
//! it stops any other work, and the restore still runs.
//!
//! Permission rules cannot promise that the checker changes nothing, so the
//! restore does: every file and link of the project is copied when the author
//! ends, and after the check each one whose bytes, mode or target differ is
//! written back, and everything the check added is removed.
#![allow(dead_code, unused_imports)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use open_max_core::config::ApprovalMode;
use open_max_core::state::Core;
use open_max_core::types::AgentEventEnvelope;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Notify};

use crate::headless::{run_session, HeadlessArgs};

const PLAN: &str = include_str!("../check-plan.txt");
const RUN: &str = include_str!("../check-run.txt");

/// The most files a project may hold for its run to be checked: past it the
/// copy that guarantees the restore costs too much.
pub(crate) const MAX_FILES: usize = 20_000;
/// The most bytes the project's files may hold, for the same reason.
const MAX_BYTES: u64 = 512 << 20;
/// Files up to this size are read when the run starts, up to `TEXT_TOTAL`
/// in all, so the checker can be shown what the author changed in them.
const TEXT_FILE: u64 = 64 << 10;
const TEXT_TOTAL: u64 = 2 << 20;
/// The diff, and the author's final message, are cut at this many bytes.
const SHOWN_BYTES: usize = 4096;
/// The most paths the checker is shown.
const SHOWN_PATHS: usize = 200;
/// Requests each checker turn may make.
const PLAN_REQUESTS: usize = 6;
const RUN_REQUESTS: usize = 16;
/// How long the harness lets an objection's command run.
const RERUN_LIMIT: Duration = Duration::from_secs(60);
/// The most commands re-run, and the most objections handed back.
const RERUNS: usize = 10;
const KEPT: usize = 5;
/// Each command's output shown to the author: its head and its tail.
const OUTPUT_BYTES: usize = 1500;
/// Set in the environment of everything the run starts: to the check's id
/// for the checker and the re-run commands, which is how the harness finds
/// what to stop, and to that id and `-author` for the author, which is how
/// it learns whether the check ended something the author left running.
const MARKER: &str = "OPENMAX_CHECK";

#[derive(Clone, PartialEq)]
enum Kind {
    File { size: u64, mtime: Option<SystemTime>, mode: u32 },
    Link(PathBuf),
}

/// Every file, link and directory under a root, by path relative to it.
#[derive(Default)]
struct Tree {
    entries: BTreeMap<PathBuf, Kind>,
    dirs: BTreeSet<PathBuf>,
}

#[cfg(unix)]
fn mode(meta: &std::fs::Metadata) -> u32 {
    std::os::unix::fs::PermissionsExt::mode(&meta.permissions())
}

#[cfg(not(unix))]
fn mode(_: &std::fs::Metadata) -> u32 {
    0
}

/// The tree under `root`, `.git` included and `skip` left out; None past
/// `limit` files or when a directory or entry cannot be read. Links are
/// recorded, never followed.
fn scan(root: &Path, skip: &[PathBuf], limit: usize) -> Option<Tree> {
    let (mut tree, mut todo) = (Tree::default(), vec![root.to_path_buf()]);
    while let Some(dir) = todo.pop() {
        for entry in std::fs::read_dir(&dir).ok()? {
            let path = entry.ok()?.path();
            if skip.contains(&path) {
                continue;
            }
            let meta = std::fs::symlink_metadata(&path).ok()?;
            let rel = path.strip_prefix(root).ok()?.to_path_buf();
            if meta.is_dir() {
                tree.dirs.insert(rel);
                todo.push(path);
            } else if meta.file_type().is_symlink() {
                tree.entries.insert(rel, Kind::Link(std::fs::read_link(&path).ok()?));
            } else {
                tree.entries.insert(rel, Kind::File { size: meta.len(), mtime: meta.modified().ok(), mode: mode(&meta) });
            }
            if tree.entries.len() > limit {
                return None;
            }
        }
    }
    Some(tree)
}

/// The project as it was when the run started: its tree, and the text of
/// its small files.
struct Start {
    tree: Tree,
    text: HashMap<PathBuf, String>,
}

fn start(root: &Path, skip: &[PathBuf]) -> Option<Start> {
    let tree = scan(root, skip, MAX_FILES)?;
    let (mut text, mut read) = (HashMap::new(), 0);
    for (rel, kind) in &tree.entries {
        if let Kind::File { size, .. } = kind {
            if *size <= TEXT_FILE && read + size <= TEXT_TOTAL {
                if let Some(content) = std::fs::read(root.join(rel)).ok().and_then(|b| String::from_utf8(b).ok()) {
                    read += size;
                    text.insert(rel.clone(), content);
                }
            }
        }
    }
    Some(Start { tree, text })
}

/// The project's files copied when the author ended, the restore's source.
struct Snapshot {
    tree: Tree,
    copy: PathBuf,
}

/// Copy every file of `tree` under `copy`, or say why the check cannot
/// promise to put them back.
fn snapshot(root: &Path, tree: Tree, copy: PathBuf) -> Result<Snapshot, String> {
    let bytes: u64 = tree.entries.values().map(|k| if let Kind::File { size, .. } = k { *size } else { 0 }).sum();
    if bytes > MAX_BYTES {
        return Err(format!("the project's files hold {bytes} bytes, over {MAX_BYTES}"));
    }
    for (rel, kind) in &tree.entries {
        if let Kind::File { .. } = kind {
            let to = copy.join(rel);
            let made = to.parent().map_or(Ok(()), std::fs::create_dir_all);
            if let Err(e) = made.and_then(|()| std::fs::copy(root.join(rel), &to).map(drop)) {
                return Err(format!("{} could not be copied: {e}", rel.display()));
            }
        }
    }
    Ok(Snapshot { tree, copy })
}

/// Whether two files hold the same bytes.
fn same_bytes(a: &Path, b: &Path) -> bool {
    let (Ok(mut a), Ok(mut b)) = (std::fs::File::open(a), std::fs::File::open(b)) else { return false };
    let (mut x, mut y) = (vec![0; 64 << 10], vec![0; 64 << 10]);
    loop {
        let (Ok(n), Ok(m)) = (read_full(&mut a, &mut x), read_full(&mut b, &mut y)) else { return false };
        if n != m || x[..n] != y[..m] {
            return false;
        }
        if n == 0 {
            return true;
        }
    }
}

fn read_full(file: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// Remove whatever is at `path`, a directory with all it holds included,
/// without following a link.
fn remove(path: &Path) {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => drop(std::fs::remove_dir_all(path)),
        Ok(_) => drop(std::fs::remove_file(path)),
        Err(_) => {}
    }
}

/// Put the project back as `snap` holds it: directories first, parents
/// before children, so nothing is written through a link or a file put in a
/// directory's place; then everything the snapshot lacks goes, and every
/// file or link whose bytes, mode, time or target differ is written back.
/// The paths it changed, and those it could not put back.
fn restore(root: &Path, snap: &Snapshot, skip: &[PathBuf]) -> (Vec<String>, Vec<String>) {
    let (mut fixed, mut failed) = (BTreeSet::new(), Vec::new());
    let shown = |rel: &Path| rel.display().to_string();
    for dir in &snap.tree.dirs {
        let path = root.join(dir);
        if std::fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir()) {
            continue;
        }
        remove(&path);
        match std::fs::create_dir_all(&path) {
            Ok(()) => drop(fixed.insert(shown(dir))),
            Err(_) => failed.push(shown(dir)),
        }
    }
    // A listing that fails leaves what the check added, but every file the
    // author left is still compared and written back.
    let now = scan(root, skip, usize::MAX).unwrap_or_default();
    for rel in now.entries.keys().filter(|rel| !snap.tree.entries.contains_key(*rel)) {
        remove(&root.join(rel));
        fixed.insert(shown(rel));
    }
    for rel in now.dirs.iter().filter(|rel| !snap.tree.dirs.contains(*rel)) {
        if std::fs::symlink_metadata(root.join(rel)).is_ok() {
            remove(&root.join(rel));
            fixed.insert(shown(rel));
        }
    }
    for (rel, kind) in &snap.tree.entries {
        let path = root.join(rel);
        let meta = std::fs::symlink_metadata(&path).ok();
        match kind {
            Kind::Link(target) => {
                if std::fs::read_link(&path).ok().as_ref() == Some(target) {
                    continue;
                }
                remove(&path);
                #[cfg(unix)]
                let put = std::os::unix::fs::symlink(target, &path);
                #[cfg(not(unix))]
                let put: std::io::Result<()> = Err(std::io::ErrorKind::Unsupported.into());
                match put {
                    Ok(()) => drop(fixed.insert(shown(rel))),
                    Err(_) => failed.push(shown(rel)),
                }
            }
            Kind::File { mtime, mode: was, .. } => {
                let from = snap.copy.join(rel);
                let held = meta.as_ref().is_some_and(|m| m.is_file() && mode(m) == *was) && same_bytes(&path, &from);
                if held && meta.as_ref().and_then(|m| m.modified().ok()) == *mtime {
                    continue;
                }
                if !held {
                    if !meta.as_ref().is_some_and(|m| m.is_file()) || std::fs::copy(&from, &path).is_err() {
                        remove(&path);
                        if std::fs::copy(&from, &path).is_err() {
                            failed.push(shown(rel));
                            continue;
                        }
                    }
                    #[cfg(unix)]
                    let _ = std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(*was));
                    fixed.insert(shown(rel));
                }
                if let Some(mtime) = mtime {
                    let _ = std::fs::File::options().write(true).open(&path).and_then(|f| f.set_modified(*mtime));
                }
            }
        }
    }
    (fixed.into_iter().collect(), failed)
}

/// What marks the processes one part of the run starts, so the harness can
/// find them however they detach: `MARKER` set to `tag` in their
/// environment, and, where the OS shows no other process's environment once
/// it leaves its parent (macOS), `file`, which they inherit open.
#[derive(Clone)]
struct Mark {
    tag: String,
    file: PathBuf,
}

impl Mark {
    fn new(kept: &Path, tag: String) -> Self {
        Mark { file: kept.join(format!("{tag}.held")), tag }
    }

    /// Mark what this process starts from now until `release`.
    fn hold(&self) -> Option<std::fs::File> {
        std::env::set_var(MARKER, &self.tag);
        self.open()
    }

    fn release(held: Option<std::fs::File>) {
        drop(held);
        std::env::remove_var(MARKER);
    }

    /// The mark's file, open for the children this process starts.
    fn open(&self) -> Option<std::fs::File> {
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&self.file).ok()?;
        #[cfg(unix)]
        // SAFETY: clears close-on-exec on a descriptor this process owns, so children inherit it.
        unsafe { libc::fcntl(std::os::unix::io::AsRawFd::as_raw_fd(&file), libc::F_SETFD, 0) };
        Some(file)
    }

    /// The processes that carry the mark, this one aside.
    #[cfg(unix)]
    fn pids(&self) -> Vec<i32> {
        let (marker, me) = (format!("{MARKER}={}", self.tag), std::process::id() as i32);
        let pids: Vec<i32> = if cfg!(target_os = "linux") {
            std::fs::read_dir("/proc").into_iter().flatten().flatten()
                .filter_map(|e| e.file_name().to_str()?.parse().ok())
                .filter(|pid| std::fs::read(format!("/proc/{pid}/environ")).is_ok_and(|env| env.split(|b| *b == 0).any(|v| v == marker.as_bytes())))
                .collect()
        } else {
            let lines = |program: &str, args: &[&std::ffi::OsStr]| std::process::Command::new(program).args(args)
                .stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null()).output()
                .map(|out| String::from_utf8_lossy(&out.stdout).lines().map(str::to_string).collect::<Vec<_>>()).unwrap_or_default();
            let shown = lines("ps", &["axeww".as_ref(), "-o".as_ref(), "pid=".as_ref(), "-o".as_ref(), "command=".as_ref()]).into_iter()
                .filter(|line| line.split_whitespace().any(|word| word == marker))
                .filter_map(|line| line.split_whitespace().next()?.parse().ok());
            let holding = lines("lsof", &["-t".as_ref(), "--".as_ref(), self.file.as_os_str()]).into_iter().filter_map(|line| line.trim().parse().ok());
            shown.chain(holding).collect()
        };
        let mut pids: Vec<i32> = pids.into_iter().filter(|pid| *pid != me).collect();
        pids.sort_unstable();
        pids.dedup();
        pids
    }

    /// Stop every process that carries the mark, asking first and then
    /// killing, until none is left. The pids it stopped.
    #[cfg(unix)]
    fn stop(&self) -> Vec<i32> {
        let mut stopped = Vec::new();
        for round in 0..30 {
            let pids = self.pids();
            if pids.is_empty() {
                break;
            }
            for pid in pids {
                // SAFETY: a plain signal to a process that carries the check's mark.
                unsafe { libc::kill(pid, if round < 5 { libc::SIGTERM } else { libc::SIGKILL }) };
                if !stopped.contains(&pid) {
                    stopped.push(pid);
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        stopped
    }

    /// The processes with the mark that still run, each with its command line.
    #[cfg(unix)]
    fn running(&self) -> Vec<(i32, String)> {
        self.pids().into_iter().map(|pid| {
            let line = match cfg!(target_os = "linux") {
                true => std::fs::read(format!("/proc/{pid}/cmdline")).map(|b| String::from_utf8_lossy(&b).replace('\0', " ")).unwrap_or_default(),
                false => std::process::Command::new("ps").args(["-o", "command=", "-p", &pid.to_string()]).stderr(std::process::Stdio::null())
                    .output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default(),
            };
            (pid, line.trim().to_string())
        }).collect()
    }

    #[cfg(not(unix))]
    fn stop(&self) -> Vec<i32> {
        Vec::new()
    }

    #[cfg(not(unix))]
    fn running(&self) -> Vec<(i32, String)> {
        Vec::new()
    }
}

/// Whether process `pid` still runs.
#[cfg(unix)]
fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 only asks whether the process exists.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(not(unix))]
fn alive(_: i32) -> bool {
    true
}

/// What the checker is shown of the author's work: the paths changed, and
/// the diff of the text among them, each cut short where long.
fn changes(root: &Path, start: &Start, end: &Tree) -> (Vec<String>, String) {
    let mut paths = Vec::new();
    let mut diff = String::new();
    let under_git = |rel: &Path| rel.components().next().is_some_and(|c| c.as_os_str() == ".git");
    let (mut git, mut more) = (0, 0);
    let removed = start.tree.entries.keys().filter(|rel| !end.entries.contains_key(*rel)).map(|rel| (rel, "removed"));
    let touched = end.entries.iter().filter_map(|(rel, kind)| match start.tree.entries.get(rel) {
        None => Some((rel, "created")),
        Some(was) if was != kind => Some((rel, "changed")),
        Some(_) => None,
    });
    let mut listed: Vec<(&PathBuf, &str)> = touched.chain(removed).collect();
    listed.sort();
    for (rel, how) in listed {
        if under_git(rel) {
            git += 1;
            continue;
        }
        if paths.len() == SHOWN_PATHS {
            more += 1;
            continue;
        }
        paths.push(format!("{how}: {}", rel.display()));
        if how == "removed" || diff.len() >= SHOWN_BYTES {
            continue;
        }
        let old = match how {
            "created" => Some(""),
            _ => start.text.get(rel).map(String::as_str),
        };
        let new = std::fs::read(root.join(rel)).ok().and_then(|b| String::from_utf8(b).ok()).filter(|t| !t.contains('\0'));
        let name = rel.display().to_string();
        diff.push_str(&match (old, new) {
            (Some(old), Some(new)) => similar::TextDiff::configure().deadline(std::time::Instant::now() + Duration::from_millis(200))
                .diff_lines(old, &new).unified_diff().context_radius(3).header(&format!("a/{name}"), &format!("b/{name}")).to_string(),
            (None, Some(_)) => format!("{name}: changed; its earlier text was not kept\n"),
            _ => format!("{name}: not text\n"),
        });
    }
    if more > 0 {
        paths.push(format!("and {more} more"));
    }
    if git > 0 {
        paths.push(format!("and {git} under .git"));
    }
    (paths, cut(&diff, SHOWN_BYTES))
}

/// `text` cut at `max` bytes, on a character boundary, saying it was cut.
fn cut(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let end = (0..=max).rev().find(|i| text.is_char_boundary(*i)).unwrap_or(0);
    format!("{}\n[cut]\n", &text[..end])
}

/// A print run's settings held for one checker turn: at most `requests`.
fn hold_requests(core: &Core, requests: usize) -> usize {
    let mut settings = core.settings.lock().unwrap();
    let was = settings.max_agent_iterations;
    settings.max_agent_iterations = was.clamp(1, requests);
    was
}

fn turn(args: &HeadlessArgs, prompt: String, session: Option<String>) -> HeadlessArgs {
    HeadlessArgs {
        prompts: vec![prompt],
        project: args.project.clone(),
        continue_session: session.is_some(),
        session,
        json: args.json,
        second_look: false,
        quit: args.quit.clone(),
    }
}

/// What the author left, read once it ended: the tree's copy, the paths and
/// diff to show, the task text and the final message.
struct End {
    snap: Arc<Snapshot>,
    paths: Vec<String>,
    diff: String,
    task: String,
    message: String,
    running: Vec<(i32, String)>,
}

#[allow(clippy::too_many_arguments)]
fn end(core: &Core, root: &Path, skip: &[PathBuf], start: &Start, mark: &Mark, author: &str, prompt: &str, copy: PathBuf) -> Result<End, String> {
    let tree = scan(root, skip, MAX_FILES).ok_or(format!("the project holds more than {MAX_FILES} files, or one of its directories could not be read"))?;
    let (paths, diff) = changes(root, start, &tree);
    let snap = Arc::new(snapshot(root, tree, copy)?);
    let task = open_max_core::templates::expand_user_input(&core.data_dir, root, prompt);
    let messages = open_max_core::sessions::load_messages(core, author).ok().flatten().unwrap_or_default();
    let last = messages.iter().rev().find(|m| m.role == "assistant" && m.content.as_deref().is_some_and(|c| !c.trim().is_empty()));
    let message = cut(last.and_then(|m| m.content.as_deref()).unwrap_or(""), SHOWN_BYTES);
    Ok(End { snap, paths, diff, task, message, running: mark.running() })
}

/// The URLs named in the commands of session `id`, from its event log.
fn urls(core: &Core, id: &str) -> Vec<String> {
    let log = std::fs::read_to_string(core.data_dir.join("sessions").join(format!("{id}.events.jsonl"))).unwrap_or_default();
    let mut found = Vec::new();
    for line in log.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter(|l| l["type"] == "tool_start") {
        let text = line["args"].to_string();
        for (at, _) in text.match_indices("http") {
            let url: String = text[at..].chars().take_while(|c| !c.is_whitespace() && !matches!(c, '"' | '\'' | '\\' | '`' | ')' | '>')).collect();
            if (url.starts_with("http://") || url.starts_with("https://")) && !found.contains(&url) {
                found.push(url);
            }
        }
    }
    found
}

/// Stop what the check started and put the project back, to the end
/// whatever stops the run: this is what the check promises.
async fn guard(mark: &Mark, root: &Path, snap: &Arc<Snapshot>, skip: &[PathBuf], record: &mut Value, step: &str) {
    let (mark, root, snap, skip) = (mark.clone(), root.to_path_buf(), snap.clone(), skip.to_vec());
    let (stopped, (fixed, failed)) = tokio::task::spawn_blocking(move || (mark.stop(), restore(&root, &snap, &skip))).await.unwrap_or_default();
    if !stopped.is_empty() {
        eprintln!("openmax: check: stopped {} process{} the {step} started", stopped.len(), if stopped.len() == 1 { "" } else { "es" });
    }
    if !fixed.is_empty() {
        eprintln!("openmax: check: put back {} path{} the {step} changed: {}", fixed.len(), if fixed.len() == 1 { "" } else { "s" }, cut(&fixed.join(", "), 400).trim_end());
    }
    for path in &failed {
        eprintln!("openmax: check: could not put back {path}");
    }
    record[step] = json!({ "stopped": stopped, "put_back": fixed, "not_put_back": failed });
}

fn save(dir: &Path, record: &Value) {
    let _ = open_max_core::sessions::write_atomic(&dir.join("record.json"), serde_json::to_vec_pretty(record).unwrap_or_default());
}

/// The run's status once a stop ended the check: a signal's, else the
/// author's, whose output was all written.
fn stopped(code: i32) -> i32 {
    if crate::stopped_by_a_signal() { crate::quit_status() } else { code }
}

/// A print run with `--second-look`: the author's turn, then the checker's two.
pub(crate) async fn run(core: Arc<Core>, mut rx: mpsc::UnboundedReceiver<AgentEventEnvelope>, args: HeadlessArgs) -> i32 {
    let project = args.project.clone();
    let checks = std::env::temp_dir().join("openmax-check");
    let skip = vec![core.data_dir.clone(), checks.clone()];
    let mode = core.approval_mode(&project);
    let begun = match mode {
        ApprovalMode::Auto => {
            let (root, skip) = (project.clone(), skip.clone());
            match crate::until_stopped(args.quit.notified(), &mut None, move || start(&root, &skip)).await {
                Some(begun) if !crate::signalled() => begun,
                _ => return crate::quit_status(),
            }
        }
        _ => None,
    };
    let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let id = format!("{}-{nanos}", std::process::id());
    let (dir, kept) = (checks.join(&id), core.data_dir.join("checks").join(&id));
    let (authors, checkers) = (Mark::new(&kept, format!("{id}-author")), Mark::new(&kept, id.clone()));
    let held = begun.as_ref().and_then(|_| std::fs::create_dir_all(&kept).ok()).and_then(|()| authors.hold());
    let mut author = None;
    let code = run_session(&core, &mut rx, &args, &mut author).await;
    Mark::release(held);
    let (Some(author), true) = (author, code == 0 && !crate::signalled()) else { return code };
    let Some(begun) = begun else {
        match mode {
            ApprovalMode::Auto => eprintln!("openmax: check skipped: the project holds more than {MAX_FILES} files, or one of its directories could not be read"),
            mode => eprintln!("openmax: check skipped: the approval mode is {}, and the check runs commands only in auto", mode.as_str()),
        }
        return code;
    };

    let read = {
        let (core, root, skip, mark, author, prompt, copy) = (core.clone(), project.clone(), skip.clone(), authors.clone(), author.clone(), args.prompts.concat(), kept.join("author-end"));
        move || end(&core, &root, &skip, &begun, &mark, &author, &prompt, copy)
    };
    let left = match crate::until_stopped(args.quit.notified(), &mut None, read).await {
        Some(_) | None if crate::signalled() => return stopped(code),
        None => return stopped(code),
        Some(Err(why)) => {
            eprintln!("openmax: check skipped: {why}");
            return code;
        }
        Some(Ok(left)) => left,
    };
    let _ = std::fs::create_dir_all(&dir);
    eprintln!("openmax: check: a new session checks the work; its files are in {}, the harness's record in {}", dir.display(), kept.display());
    let mut record = json!({ "author_session": author, "paths": left.paths, "left_running": left.running });

    // The checker: a plan from the request alone, then the checks.
    let shown = dir.display().to_string();
    let held = checkers.hold();
    let (was, mut checker) = (hold_requests(&core, PLAN_REQUESTS), None);
    let plan = turn(&args, format!("{}{}", PLAN.replace("{dir}", &shown), left.task), None);
    let planned = run_session(&core, &mut rx, &plan, &mut checker).await;
    let _ = std::fs::copy(dir.join("plan.md"), kept.join("plan.md"));
    if let (Some(session), 0 | 4, false) = (&checker, planned, crate::signalled()) {
        hold_requests(&core, RUN_REQUESTS);
        let paths = if left.paths.is_empty() { "(none)".to_string() } else { left.paths.join("\n") };
        let text = RUN.replace("{dir}", &shown).replace("{paths}", &paths).replace("{diff}", left.diff.trim_end()).replace("{message}", left.message.trim_end());
        run_session(&core, &mut rx, &turn(&args, text, Some(session.clone())), &mut None).await;
    }
    core.settings.lock().unwrap().max_agent_iterations = was;
    Mark::release(held);
    guard(&checkers, &project, &left.snap, &skip, &mut record, "checker").await;
    record["checker_session"] = json!(checker);
    if let Some(session) = &checker {
        let named = urls(&core, session);
        for url in &named {
            eprintln!("openmax: check: the checker named {url}");
        }
        record["urls"] = json!(named);
    }
    if crate::signalled() {
        save(&kept, &record);
        return stopped(code);
    }

    save(&kept, &record);
    code
}
