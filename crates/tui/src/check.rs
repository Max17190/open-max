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
const HANDBACK: &str = include_str!("../check-handback.txt");

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
    File { size: u64, mode: u32 },
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
                tree.entries.insert(rel, Kind::File { size: meta.len(), mode: mode(&meta) });
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

/// Copy bytes without `std::fs::copy` (whose macOS copyfile/clonefile calls add
/// five libSystem imports past the launch-shape cap).
#[expect(clippy::disallowed_methods, reason = "copies into fresh snapshot paths or restores from the snapshot without copyfile imports")]
fn copy_file(from: &Path, to: &Path) -> std::io::Result<()> {
    std::io::copy(&mut std::fs::File::open(from)?, &mut std::fs::File::create(to)?)?;
    Ok(())
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
            if let Err(e) = made.and_then(|()| copy_file(&root.join(rel), &to)) {
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
            Kind::File { mode: was, .. } => {
                let from = snap.copy.join(rel);
                let held = meta.as_ref().is_some_and(|m| m.is_file() && mode(m) == *was) && same_bytes(&path, &from);
                if held {
                    continue;
                }
                if !meta.as_ref().is_some_and(|m| m.is_file()) || copy_file(&from, &path).is_err() {
                    remove(&path);
                    if copy_file(&from, &path).is_err() {
                        failed.push(shown(rel));
                        continue;
                    }
                }
                #[cfg(unix)]
                if let Ok(file) = std::fs::File::open(&path) {
                    let _ = file.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(*was));
                }
                fixed.insert(shown(rel));
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
            (Some(old), Some(new)) => line_diff(&name, old, &new),
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

/// Compact line diff between `old` and `new` with up to 3 lines of surrounding
/// context, trimmed at common prefix and suffix without linking `similar` into
/// the TUI binary.
#[cold]
#[inline(never)]
fn line_diff(name: &str, old: &str, new: &str) -> String {
    let (a, b): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    if pre == a.len() && pre == b.len() {
        return String::new();
    }
    let suf = a[pre..].iter().rev().zip(b[pre..].iter().rev()).take_while(|(x, y)| x == y).count();
    let (a_end, b_end) = (a.len() - suf, b.len() - suf);
    let ctx_pre = pre.saturating_sub(3);
    let ctx_suf = suf.min(3);
    let mut out = format!("--- a/{name}\n+++ b/{name}\n@@ -{},{} +{},{} @@\n", ctx_pre + 1, a_end - ctx_pre + ctx_suf, ctx_pre + 1, b_end - ctx_pre + ctx_suf);
    for line in &a[ctx_pre..pre] {
        out.push(' ');
        out.push_str(line);
        out.push('\n');
    }
    for line in &a[pre..a_end] {
        out.push('-');
        out.push_str(line);
        out.push('\n');
    }
    for line in &b[pre..b_end] {
        out.push('+');
        out.push_str(line);
        out.push('\n');
    }
    for line in &a[a_end..a_end + ctx_suf] {
        out.push(' ');
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `text` cut at `max` bytes, on a character boundary, saying it was cut.
fn cut(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let end = (0..=max).rev().find(|i| text.is_char_boundary(*i)).unwrap_or(0);
    format!("{}\n[cut]\n", &text[..end])
}

fn one_space(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `quote` is in `task`, whitespace aside: each part between
/// ellipses in turn, once its quotation marks are trimmed.
fn quoted(task: &str, quote: &str) -> bool {
    let (task, quote) = (one_space(task), one_space(quote.trim().trim_matches(|c| matches!(c, '"' | '\'' | '\u{201c}' | '\u{201d}'))));
    let parts: Vec<&str> = quote.split("...").flat_map(|p| p.split('\u{2026}')).map(str::trim).filter(|p| !p.is_empty()).collect();
    let mut from = 0;
    !parts.is_empty() && parts.iter().all(|part| match task[from..].find(part) {
        Some(at) => { from += at + part.len(); true }
        None => false,
    })
}

/// A field of an objection as text: a string as it is, any other value as JSON.
fn field(objection: &Value, key: &str) -> String {
    match &objection[key] {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// How an objection's command ran when the harness ran it.
enum Ran {
    Exited(Option<i32>, String),
    TimedOut,
    Stopped,
}

/// Run `command` with bash in the project, as the checker's own bash calls
/// run, with its output to `out`, for at most `RERUN_LIMIT`; a stop ends it.
async fn rerun(project: &Path, command: &str, mark: &Mark, out: &Path, quit: &Notify) -> Ran {
    let Ok(file) = std::fs::OpenOptions::new().create_new(true).write(true).open(out) else { return Ran::Exited(None, "the harness could not record the output".into()) };
    let Ok(err) = file.try_clone() else { return Ran::Exited(None, "the harness could not record the output".into()) };
    let mut run = tokio::process::Command::new("bash");
    run.args(["-c", command]).current_dir(project).stdin(std::process::Stdio::null()).stdout(file).stderr(err)
        .env(MARKER, &mark.tag).env("OPENMAX_SESSION", "1").env("GIT_TERMINAL_PROMPT", "0").env_remove("OPENMAX_HUMAN_ATTEST").kill_on_drop(true);
    #[cfg(unix)]
    run.process_group(0);
    let held = mark.open();
    let spawned = run.spawn();
    drop(held);
    let Ok(mut child) = spawned else { return Ran::Exited(None, "bash could not be started".into()) };
    let pid = child.id();
    let ran = tokio::select! {
        status = child.wait() => status.ok().map(|s| s.code()),
        _ = tokio::time::sleep(RERUN_LIMIT) => None,
        () = quit.notified() => None,
    };
    #[cfg(unix)]
    if let Some(pid) = pid {
        // SAFETY: the command's own process group, which it leads.
        unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
    }
    let _ = child.wait().await;
    match ran {
        _ if crate::signalled() => Ran::Stopped,
        None => Ran::TimedOut,
        Some(code) => {
            let bytes = std::fs::read(out).unwrap_or_default();
            let text = String::from_utf8_lossy(&bytes);
            let shown = if text.len() <= 2 * OUTPUT_BYTES { text.to_string() } else {
                let head = (0..=OUTPUT_BYTES).rev().find(|i| text.is_char_boundary(*i)).unwrap_or(0);
                let tail = (text.len() - OUTPUT_BYTES..text.len()).find(|i| text.is_char_boundary(*i)).unwrap_or(text.len());
                format!("{}\n[{} bytes not shown]\n{}", &text[..head], tail - head, &text[tail..])
            };
            Ran::Exited(code, shown)
        }
    }
}

/// Whether a command that ran shows its objection again: it failed, or it
/// printed what the checker observed.
fn reproduced(ran: &Ran, observed: &str) -> bool {
    match ran {
        Ran::Exited(code, output) => *code != Some(0) || (!observed.trim().is_empty() && output.contains(observed.trim())),
        _ => false,
    }
}

/// Whether the project's permission rules and `pre_tool_use` hooks let the
/// command run as a bash call, as the checker's own would have had to.
async fn allowed(core: &Core, project: &Path, session: &str, command: &str) -> bool {
    use open_max_core::{hooks, permissions};
    let args = json!({ "command": command });
    let rules = permissions::Permissions::discover_for_mode(project, &core.data_dir, ApprovalMode::Auto);
    if matches!(rules.evaluate("bash", &args), permissions::PermissionDecision::Deny { .. } | permissions::PermissionDecision::Ask) {
        return false;
    }
    let gates = hooks::Hooks::discover_for_mode(project, &core.data_dir, ApprovalMode::Auto);
    let cancel = Arc::new(open_max_core::state::CancelToken::default());
    matches!(gates.pre_tool_use(session, "bash", &args, project, &cancel).await, hooks::PreToolResult::Allow)
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

/// A print run with `--second-look`: the author's turn, then the check.
/// Kept cold behind a boxed trait object so runs without `--second-look`
/// never fault in the checker's future layout or poll code.
#[cold]
#[inline(never)]
pub(crate) fn run(
    core: Arc<Core>,
    rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    args: HeadlessArgs,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = i32>>> {
    Box::pin(run_inner(core, rx, args))
}

#[cold]
#[inline(never)]
async fn run_inner(core: Arc<Core>, mut rx: mpsc::UnboundedReceiver<AgentEventEnvelope>, args: HeadlessArgs) -> i32 {
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
    let code = run_session(&core, &mut rx, &args, Some(&mut author)).await;
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
    let planned = run_session(&core, &mut rx, &plan, Some(&mut checker)).await;
    let _ = copy_file(&dir.join("plan.md"), &kept.join("plan.md"));
    if let (Some(session), 0 | 4, false) = (&checker, planned, crate::signalled()) {
        hold_requests(&core, RUN_REQUESTS);
        let paths = if left.paths.is_empty() { "(none)".to_string() } else { left.paths.join("\n") };
        let text = RUN.replace("{dir}", &shown).replace("{paths}", &paths).replace("{diff}", left.diff.trim_end()).replace("{message}", left.message.trim_end());
        run_session(&core, &mut rx, &turn(&args, text, Some(session.clone())), None).await;
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

    let objections = std::fs::read_to_string(dir.join("objections.jsonl")).unwrap_or_default();
    let (failures, readings, reran) = filter(&core, &project, &author, &checkers, &left.task, &objections, &kept, &args.quit, &mut record).await;
    if reran {
        guard(&checkers, &project, &left.snap, &skip, &mut record, "re-run").await;
    }
    let ended: Vec<&(i32, String)> = left.running.iter().filter(|(pid, _)| !alive(*pid)).collect();
    record["ended_while_checked"] = json!(ended);
    if crate::signalled() {
        save(&kept, &record);
        return stopped(code);
    }
    if failures.is_empty() && ended.is_empty() {
        eprintln!("openmax: check: the work stands as the author left it");
        let core = core.clone();
        let _ = crate::until_stopped(args.quit.notified(), &mut None, move || open_max_core::sessions::touch(&core, &author)).await;
        save(&kept, &record);
        return code;
    }

    // The hand-back: once, in the author's own session.
    let message = handback(&failures, &readings, &ended);
    eprintln!("openmax: check: handing {} objection{} back to the author", failures.len(), if failures.len() == 1 { "" } else { "s" });
    record["handed_back"] = json!(message);
    let second = run_session(&core, &mut rx, &turn(&args, message, Some(author.clone())), None).await;
    if !crate::signalled() && !failures.is_empty() {
        last_look(&project, &skip, &checkers, &kept, &failures, &args.quit, &mut record).await;
    }
    save(&kept, &record);
    if crate::stopped_by_a_signal() { crate::quit_status() } else { second }
}

/// An objection that the harness reproduced: the objection, then how its
/// command exited and what it printed.
type Failure = (Value, Option<i32>, String);

/// Keep the objections in `lines` whose quote is the request's and whose
/// command fails again in the harness's hands, at most `KEPT`, and the
/// readings aside; each verdict goes to the record. Whether any command ran.
#[allow(clippy::too_many_arguments)]
async fn filter(core: &Core, project: &Path, author: &str, mark: &Mark, task: &str, lines: &str, kept: &Path, quit: &Notify, record: &mut Value) -> (Vec<Failure>, Vec<Value>, bool) {
    let (mut failures, mut readings, mut verdicts, mut reruns) = (Vec::new(), Vec::new(), Vec::new(), 0);
    for line in lines.lines().filter(|l| !l.trim().is_empty()) {
        let objection: Value = serde_json::from_str(line).unwrap_or(Value::Null);
        let (quote, command, kind) = (field(&objection, "quote"), field(&objection, "command"), field(&objection, "kind"));
        let verdict = match kind.as_str() {
            _ if !objection.is_object() => "dropped: not a JSON object".to_string(),
            "fails" | "missing" | "unrequested" | "reading" if kind != "unrequested" && !quoted(task, &quote) => "dropped: its quote is not in the request".into(),
            "reading" if readings.len() < KEPT => {
                readings.push(objection.clone());
                "aside: a reading".into()
            }
            "reading" => "dropped: past the readings kept".into(),
            "fails" | "missing" | "unrequested" if command.trim().is_empty() || command.contains('\n') => "dropped: its command is not one line".into(),
            "fails" | "missing" | "unrequested" if failures.len() >= KEPT || reruns >= RERUNS => "dropped: past the commands run again".into(),
            "fails" | "missing" | "unrequested" if !allowed(core, project, author, &command).await => "dropped: a permission rule or hook refuses its command".into(),
            "fails" | "missing" | "unrequested" => {
                reruns += 1;
                let ran = rerun(project, &command, mark, &kept.join(format!("rerun-{reruns}.out")), quit).await;
                match ran {
                    Ran::Stopped => break,
                    Ran::TimedOut => "dropped: its command did not finish".into(),
                    Ran::Exited(..) if !reproduced(&ran, &field(&objection, "observed")) => "dropped: its command passed".into(),
                    Ran::Exited(status, output) => {
                        failures.push((objection.clone(), status, output));
                        "kept".into()
                    }
                }
            }
            _ => "dropped: its kind is not one of the four".into(),
        };
        verdicts.push(json!({ "objection": objection, "verdict": verdict }));
    }
    eprintln!("openmax: check: {} of {} objection{} reproduced", failures.len(), verdicts.len(), if verdicts.len() == 1 { "" } else { "s" });
    record["objections"] = json!(verdicts);
    (failures, readings, reruns > 0)
}

/// The one message the author gets: each reproduced objection with the
/// output the harness got, the readings as information, and what the check
/// ended that the author had left running.
fn handback(failures: &[Failure], readings: &[Value], ended: &[&(i32, String)]) -> String {
    let parts: Vec<&str> = HANDBACK.split("\n%%\n").map(str::trim).collect();
    let mut message = String::new();
    if !failures.is_empty() {
        message.push_str(parts[0]);
        for (n, (objection, status, output)) in failures.iter().enumerate() {
            let status = status.map_or("was ended by a signal".to_string(), |s| format!("exited with status {s}"));
            let quote = one_space(&field(objection, "quote"));
            let says = if quote.is_empty() { String::new() } else { format!(" The request says: \"{quote}\".") };
            message.push_str(&format!("\n\n{}.{says}\n`{}` {status} and printed:\n```\n{}\n```\nThe check expected: {}",
                n + 1, field(objection, "command"), output.trim_end(), one_space(&field(objection, "expected"))));
        }
        if !readings.is_empty() {
            message.push_str(&format!("\n\n{}", parts[1]));
            for reading in readings {
                message.push_str(&format!("\n- The request says: \"{}\". The check read it as: {}", one_space(&field(reading, "quote")), one_space(&field(reading, "expected"))));
            }
        }
    }
    if !ended.is_empty() {
        message.push_str(&format!("\n\n{}", parts[2]));
        for (_, line) in ended {
            message.push_str(&format!("\n- `{line}`"));
        }
    }
    if !failures.is_empty() {
        message.push_str(&format!("\n\n{}", parts[3]));
    }
    message.trim().to_string()
}

/// At the author's second end, run the handed-back commands once more and
/// record which now pass, then put the tree back as the author left it.
async fn last_look(project: &Path, skip: &[PathBuf], mark: &Mark, kept: &Path, failures: &[Failure], quit: &Arc<Notify>, record: &mut Value) {
    let take = {
        let (root, skip, copy) = (project.to_path_buf(), skip.to_vec(), kept.join("second-end"));
        move || scan(&root, &skip, MAX_FILES).ok_or_else(|| "the tree could not be listed".to_string()).and_then(|tree| snapshot(&root, tree, copy))
    };
    let snap = match crate::until_stopped(quit.notified(), &mut None, take).await {
        Some(Ok(snap)) if !crate::signalled() => Arc::new(snap),
        Some(Err(why)) => return eprintln!("openmax: check: the commands were not run again: {why}"),
        _ => return,
    };
    let mut passing = Vec::new();
    for (n, (objection, ..)) in failures.iter().enumerate() {
        let ran = rerun(project, &field(objection, "command"), mark, &kept.join(format!("final-{}.out", n + 1)), quit).await;
        if matches!(ran, Ran::Stopped) {
            break;
        }
        passing.push(json!({ "command": field(objection, "command"), "passes": !reproduced(&ran, &field(objection, "observed")) }));
    }
    guard(mark, project, &snap, skip, record, "last re-run").await;
    let now = passing.iter().filter(|p| p["passes"] == true).count();
    eprintln!("openmax: check: after the author's second end, {now} of {} command{} pass", failures.len(), if failures.len() == 1 { "" } else { "s" });
    record["last_re_run"] = json!(passing);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quote_is_found_whitespace_and_ellipses_aside() {
        let task = "Write the two\nnames it prints,  one per line, to the file named below.";
        assert!(quoted(task, "two names it prints"));
        assert!(quoted(task, "\u{201c}the two names\u{201d}"));
        assert!(quoted(task, "two names ... one per line"));
        assert!(!quoted(task, "one per line ... two names"), "parts in another order are not a quote");
        assert!(!quoted(task, "three names"));
        assert!(!quoted(task, "  "), "an empty quote quotes nothing");
    }
}
