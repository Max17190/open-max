//! `openmax --subagent "<task>" --worktree`: one task, run by one child
//! session in a new git worktree of this repository.
//!
//! The child runs in this process, on print mode's path, so its events print
//! as `-p` prints them and nothing reaches any request: no tool, no prompt
//! line, no scheduler. The worktree is made under the project
//! (`.worktrees/<name>`), so the trust the human granted the project covers
//! it as it covers any subdirectory, and the child is held to the project's
//! approval mode or a stricter one. However the child ends, its tools are
//! stopped, its changes are committed to its branch, and the worktree is
//! removed.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use open_max_core::state::Core;
use open_max_core::text::one_line;
use open_max_core::types::AgentEventEnvelope;
use tokio::sync::{mpsc, Notify};

/// The directory under the project that holds sub-agent worktrees.
const DIR: &str = ".worktrees";

/// The pattern a run adds to the repository's own exclude file
/// (`info/exclude`, which is never committed) when it is not there yet: the
/// worktrees stay out of the project's `git status`, and out of its glob and
/// grep, which read the same file. It is never removed, so no run can take it
/// from another. Git and the walker stop at a worktree's own root, so the
/// child still sees its files.
const EXCLUDE: &str = ".worktrees/";

/// The variables that choose a repository, a working tree or an index for
/// git, as `git rev-parse --local-env-vars` names them: git clears them
/// itself when it works in another repository.
const REPOSITORY_VARIABLES: [&str; 15] = [
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
];

/// Clear `REPOSITORY_VARIABLES` from this process's environment. Set by a
/// caller (a git hook, a script run by git), they would point the sub-agent's
/// git commands, and its child's tools, at another repository, working tree
/// or index than the worktree's. Called before any thread starts, so the
/// change races no reader of the environment.
pub fn clear_repository_variables() {
    for name in REPOSITORY_VARIABLES {
        std::env::remove_var(name);
    }
}

/// Run `task` in a new worktree of the repository `project` is in, and
/// return the exit status: the child's, as print mode's, or 128 plus a
/// signal's number. `quit` is held until the worktree is dealt with, so a
/// signal then is read as the stop it is, and only a second one exits at
/// once.
pub async fn run(
    core: Arc<Core>,
    core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    project: &Path,
    task: String,
    json: bool,
    quit: Arc<Notify>,
) -> i32 {
    let worktree = match Worktree::create(project, &task) {
        Ok(worktree) => worktree,
        Err((code, reason)) => {
            eprintln!("openmax: {}", one_line(&reason));
            return if crate::signalled() { crate::quit_status() } else { code };
        }
    };
    eprintln!(
        "openmax: sub-agent working in {} on branch {}",
        worktree.root.display(),
        worktree.branch
    );
    // The project's own mode, or stricter: never a looser choice saved for
    // a path inside the worktree.
    core.hold_approval_mode(&worktree.project, core.approval_mode(project));
    // A signal while the worktree was made leaves no turn to run.
    let code = if crate::signalled() {
        crate::quit_status()
    } else {
        let args = crate::headless::HeadlessArgs {
            prompts: vec![task.clone()],
            project: worktree.project.clone(),
            continue_session: false,
            json,
            quit: quit.clone(),
        };
        crate::headless::run(core.clone(), core_rx, args).await
    };
    // A turn that did not settle in time still owns its tool groups, and no
    // tool may write in the worktree while it is committed and removed.
    open_max_core::execution::kill_process_groups();
    // A turn the stop's deadline left behind can have a file tool, which
    // runs in this process, still writing in the worktree: its work stays
    // where it is.
    let (kept, line) = match settled(&core) {
        true => worktree.finish(&task),
        false => (false, format!(
            "the sub-agent's turn did not stop in time, so its work stays as it is in {} on branch {}",
            worktree.root.display(),
            worktree.branch
        )),
    };
    eprintln!("openmax: {}", one_line(&line));
    drop(quit);
    if crate::signalled() {
        crate::quit_status()
    } else if !kept && code == 0 {
        1
    } else {
        code
    }
}

/// How long `settled` waits for the running list's lock: a turn that
/// settles holds it for the moment it takes to send its last event.
const SETTLE_CHECK: Duration = Duration::from_millis(500);

/// Whether the child's turn has settled. A turn takes itself off the
/// running list before its last event, so one still listed is one the stop's
/// deadline left behind. A turn still starting holds the list's lock while
/// it waits on the session index, for as long as another openmax keeps it,
/// so a lock still held after `SETTLE_CHECK` counts as a turn that has not
/// settled, and the check never waits on it.
fn settled(core: &Core) -> bool {
    use std::sync::TryLockError;
    let deadline = Instant::now() + SETTLE_CHECK;
    loop {
        match core.running.try_lock() {
            Ok(running) => return running.is_empty(),
            Err(TryLockError::Poisoned(running)) => return running.into_inner().is_empty(),
            Err(TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(TryLockError::WouldBlock) => return false,
        }
    }
}

struct Worktree {
    /// The canonical project the command ran in.
    parent: PathBuf,
    /// The worktree's root, in the parent's `.worktrees`.
    root: PathBuf,
    /// Where the child runs: the parent's place in the repository, inside
    /// the worktree.
    project: PathBuf,
    branch: String,
    /// The commit the worktree was made at.
    base: String,
}

impl Worktree {
    /// Make a worktree of HEAD on a new branch, or say why not, with the
    /// exit status: 2 outside a git repository, before its first commit, or
    /// where the worktree would have no copy of this directory inside the
    /// project; 1 when git or the filesystem fails.
    fn create(project: &Path, task: &str) -> Result<Self, (i32, String)> {
        let parent = std::fs::canonicalize(project)
            .map_err(|e| (1, format!("cannot resolve {}: {e}", project.display())))?;
        let prefix = git_output(&parent, &["rev-parse", "--show-prefix"]).map_err(|e| {
            (2, format!("--subagent --worktree runs its child in a git worktree, and {} is not in a git repository: {e}", parent.display()))
        })?;
        // This directory's path in the repository, byte for byte as git
        // wrote it: only the newline and the trailing slash are git's.
        let prefix = prefix.strip_suffix(b"\n").unwrap_or(&prefix);
        let prefix = path_from_bytes(prefix.strip_suffix(b"/").unwrap_or(prefix));
        let base = git(&parent, &["rev-parse", "--verify", "--quiet", "HEAD^{commit}"]).map_err(|_| {
            (2, "--subagent --worktree starts its child from HEAD, and this repository has no commit yet".to_string())
        })?;
        if launch_git(&parent, &["status", "--porcelain"]).is_ok_and(|status| !status.is_empty()) {
            eprintln!(
                "openmax: the sub-agent starts from HEAD ({}); uncommitted changes in this tree are not in its worktree",
                &base[..base.len().min(12)]
            );
        }
        let dir = parent.join(DIR);
        let name = name(task);
        let root = dir.join(&name);
        let project = match prefix.as_os_str().is_empty() {
            true => root.clone(),
            false => root.join(&prefix),
        };
        let worktree = Worktree { parent, root, project, branch: format!("subagent/{name}"), base };
        let add = [
            OsStr::new("worktree"),
            OsStr::new("add"),
            OsStr::new("--quiet"),
            OsStr::new("-b"),
            OsStr::new(&worktree.branch),
            worktree.root.as_os_str(),
            OsStr::new(&worktree.base),
        ];
        if let Err(e) = run_git(&worktree.parent, &add, Step::Checkout) {
            // A post-checkout hook that fails does so once the worktree and
            // its branch exist, with nothing in them yet.
            let left = match worktree.remove().is_ok() || !worktree.root.exists() {
                true => {
                    worktree.drop_branch();
                    String::new()
                }
                false => format!("; it stays at {} on branch {}", worktree.root.display(), worktree.branch),
            };
            return Err((1, format!("cannot make the sub-agent's worktree: {e}{left}")));
        }
        // The child's directory lies inside its worktree, which lies inside
        // the project: that place is what extends the project's trust to it,
        // and keeps its edits out of the project's tree. A link can point out
        // of either, as a `.worktrees` that is one does, or a directory HEAD
        // holds as one.
        let root = std::fs::canonicalize(&worktree.root);
        let reason = match (root, std::fs::canonicalize(&worktree.project)) {
            (Ok(root), Ok(path)) if root.starts_with(&worktree.parent) && path.starts_with(&root) => {
                // Only for a worktree that stays: a refusal leaves nothing.
                exclude(&worktree.parent);
                return Ok(worktree);
            }
            (_, Ok(path)) => format!(
                "the sub-agent's directory resolved to {}, outside its worktree in this project",
                path.display()
            ),
            (_, Err(_)) => format!(
                "HEAD has no {}, so the sub-agent's worktree has no copy of this directory; run --subagent from one HEAD has",
                prefix.display()
            ),
        };
        if worktree.remove().is_ok() {
            worktree.drop_branch();
        }
        Err((2, reason))
    }

    /// Commit what the child changed to its branch and remove the worktree.
    /// Returns whether all of that happened (the changes safe on the branch,
    /// or none, and the worktree gone), and the line that says where things
    /// are.
    fn finish(self, task: &str) -> (bool, String) {
        let branch = format!("refs/heads/{}", self.branch);
        // The snapshot goes on this branch or on a detached HEAD, never on a
        // branch the child switched to, which can be one of the user's.
        let detached = match git(&self.root, &["symbolic-ref", "-q", "HEAD"]) {
            Ok(head) if head == branch => Ok(()),
            _ => git(&self.root, &["update-ref", "--no-deref", "HEAD", "HEAD"]).map(drop),
        };
        let committed = detached.and_then(|()| git(&self.root, &["add", "--all"])).and_then(|_| {
            // Exits 1 when something is staged, a nested repository included
            // whatever the repository's settings ignore.
            if git(&self.root, &["diff", "--cached", "--quiet", "--ignore-submodules=none"]).is_ok() {
                return Ok(());
            }
            let message = format!("openmax sub-agent: {}", headline(task));
            // A snapshot of the child's work, made on the way out, possibly
            // under a stop: commit signing could wait on a person, so it is
            // off, as the repository's hooks are (`run_git`).
            let commit = ["-c", "commit.gpgsign=false", "commit", "--quiet", "-m"];
            git(&self.root, &[&commit[..], &[message.as_str()]].concat()).map(drop)
        });
        // A child that switched branches or detached HEAD still leaves its
        // work on this branch. Until it is there, the worktree is the only
        // place that holds it, so a step that fails keeps the worktree.
        let kept = committed
            .and_then(|()| git(&self.root, &["rev-parse", "HEAD"]))
            .and_then(|tip| git(&self.root, &["update-ref", branch.as_str(), tip.as_str()]).map(|_| tip));
        let tip = match kept {
            Ok(tip) => tip,
            Err(e) => {
                let line = format!("the sub-agent's changes are not on branch {} ({e}); they stay in {}", self.branch, self.root.display());
                return (false, line);
            }
        };
        let line = match self.unsaved(&tip) {
            Ok(left) if left.is_empty() => None,
            Ok(left) => Some(format!(
                "the sub-agent's work in {} is not on branch {}, as no commit here can hold it; it stays in {}",
                left.join(", "),
                self.branch,
                self.root.display()
            )),
            Err(e) => Some(format!(
                "the sub-agent's work may not all be on branch {} ({e}); it stays in {}",
                self.branch,
                self.root.display()
            )),
        };
        if let Some(line) = line {
            return (false, line);
        }
        let removed = self.remove();
        let unchanged = tip == self.base;
        let mut line = if unchanged {
            "the sub-agent changed nothing".to_string()
        } else {
            let stat = match git(&self.parent, &["diff", "--shortstat", self.base.as_str(), tip.as_str()]) {
                Ok(stat) if stat.is_empty() => ": no net change".to_string(),
                Ok(stat) => format!(": {stat}"),
                Err(_) => String::new(),
            };
            format!("the sub-agent's changes are on branch {}{stat}", self.branch)
        };
        match removed {
            Ok(()) if unchanged => {
                self.drop_branch();
                line.push_str(", so no branch was kept");
            }
            Ok(()) => {}
            Err(e) => {
                line.push_str(&format!("; its worktree stays at {}: {e}", self.root.display()));
                return (false, line);
            }
        }
        (true, line)
    }

    /// What the snapshot `tip` could not put on the branch, by path, which
    /// would go with the worktree: work inside a nested repository or a
    /// submodule, of which a commit here records only that repository's
    /// commit, checked whatever the repository's settings ignore; and a
    /// commit the child made in one, clean or not, which the branch records
    /// only by its id, unless a remote-tracking branch there holds it, as it
    /// does a commit the child fetched.
    fn unsaved(&self, tip: &str) -> Result<Vec<String>, String> {
        let status = git_output(&self.root, &["status", "--porcelain", "--ignore-submodules=none"])?;
        // Each line as git wrote it: a status code, a space, then the path.
        let status = String::from_utf8_lossy(&status);
        let mut left: Vec<String> = status.lines().map(|line| line.get(3..).unwrap_or(line).to_string()).collect();
        let changes = ["diff-tree", "-r", "-z", "--ignore-submodules=none", self.base.as_str(), tip];
        let changes = git_output(&self.root, &changes)?;
        // `:<mode> <mode> <id> <id> <status>`, then the path, each ending in
        // NUL; a repository's commit has mode 160000.
        let changes = String::from_utf8_lossy(&changes);
        let mut fields = changes.split('\0');
        while let (Some(change), Some(path)) = (fields.next(), fields.next()) {
            let [_, "160000", _, commit, _] = change.split(' ').collect::<Vec<_>>()[..] else {
                continue;
            };
            let refs = ["for-each-ref", "--count=1", "--contains", commit, "refs/remotes/"];
            let held = git(&self.root.join(path), &refs).is_ok_and(|held| !held.is_empty());
            if !held && !left.iter().any(|left| left == path) {
                left.push(path.to_string());
            }
        }
        Ok(left)
    }

    /// Remove the worktree, ignored files included, then `.worktrees` if it
    /// is empty.
    fn remove(&self) -> Result<(), String> {
        let remove = [OsStr::new("worktree"), OsStr::new("remove"), OsStr::new("--force"), self.root.as_os_str()];
        let removed = git(&self.parent, &remove);
        self.remove_dir();
        removed.map(drop)
    }

    /// Delete the branch, only while it still points at the base commit.
    fn drop_branch(&self) {
        let branch = format!("refs/heads/{}", self.branch);
        let _ = git(&self.parent, &["update-ref", "-d", branch.as_str(), self.base.as_str()]);
    }

    /// Remove `.worktrees` if it is empty: the call does nothing to one that
    /// another run or the user still keeps anything in.
    fn remove_dir(&self) {
        let _ = std::fs::remove_dir(self.parent.join(DIR));
    }
}

/// Add `EXCLUDE` to the repository's exclude file unless a line there is it
/// already. Best effort: a file that cannot be read or written leaves the
/// worktrees visible to the project's status, glob and grep, nothing worse.
fn exclude(parent: &Path) {
    use std::io::{Read, Write};
    let Ok(path) = git_output(parent, &["rev-parse", "--path-format=absolute", "--git-path", "info/exclude"]) else {
        return;
    };
    let path = path_from_bytes(path.strip_suffix(b"\n").unwrap_or(&path));
    let _ = path.parent().map(std::fs::create_dir_all);
    let Ok(mut file) = std::fs::OpenOptions::new().read(true).append(true).create(true).open(&path) else {
        return;
    };
    // Held while the file is read and written, so runs that start together
    // add the line once.
    let mut text = String::new();
    if file.lock().is_err() || file.read_to_string(&mut text).is_err() {
        return;
    }
    if text.lines().any(|line| line.trim() == EXCLUDE) {
        return;
    }
    let separator = if text.is_empty() || text.ends_with('\n') { "" } else { "\n" };
    let _ = file.write_all(format!("{separator}# openmax sub-agent worktrees\n{EXCLUDE}\n").as_bytes());
}

/// The worktree's directory and its branch's last part: the task's first
/// words, for a person reading `git branch`, then eight hex digits of the
/// process and the clock, which keep runs apart.
fn name(task: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut slug = String::new();
    for word in task.split(|c: char| !c.is_ascii_alphanumeric()).filter(|word| !word.is_empty()) {
        if slug.len() + word.len() > 32 {
            break;
        }
        slug.push_str(&word.to_ascii_lowercase());
        slug.push('-');
    }
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    (std::process::id(), std::time::SystemTime::now()).hash(&mut hash);
    format!("{slug}{:08x}", hash.finish() as u32)
}

/// The task's first line, short enough for a commit subject.
fn headline(task: &str) -> String {
    let line = one_line(task.trim().lines().next().unwrap_or_default());
    match line.char_indices().nth(60) {
        Some((cut, _)) => format!("{}…", &line[..cut]),
        None => line,
    }
}

/// How a git command of the run meets a stop and the repository's hooks.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// One a stop waits for, as the snapshot and the cleanup are, for up to
    /// `STOP_SETTLE`: it runs no hook and no file-system monitor of the
    /// repository's, which could hold it up, or veto the snapshot, but can
    /// run its filters (`git add` runs its clean filters), which can take as
    /// long as they like.
    Settled,
    /// One of the launch, before anything of the child's exists, which a
    /// stop ends instead of waiting for; no hook either.
    Launch,
    /// The launch's `git worktree add`, which a stop ends too, and which
    /// runs the repository's hooks, `post-checkout` included, as any new
    /// checkout does.
    Checkout,
}

/// `run_git`'s stdout as text, trimmed.
fn git<S: AsRef<OsStr>>(dir: &Path, args: &[S]) -> Result<String, String> {
    run_git(dir, args, Step::Settled).map(|stdout| String::from_utf8_lossy(&stdout).trim().to_string())
}

/// `run_git`'s stdout as git wrote it.
fn git_output<S: AsRef<OsStr>>(dir: &Path, args: &[S]) -> Result<Vec<u8>, String> {
    run_git(dir, args, Step::Settled)
}

/// As `git`, for a command of the launch, which a stop does not wait for.
fn launch_git<S: AsRef<OsStr>>(dir: &Path, args: &[S]) -> Result<String, String> {
    run_git(dir, args, Step::Launch).map(|stdout| String::from_utf8_lossy(&stdout).trim().to_string())
}

/// How often a git command looks for a stop.
const STOP_CHECK: Duration = Duration::from_millis(20);

/// How long a stopped command's group has between SIGTERM and SIGKILL.
const STOP_GRACE: Duration = Duration::from_millis(250);

/// Why a command of the launch failed when a stop ended it.
const STOPPED: &str = "a signal stopped it";

/// Run git in `dir` with its output captured, in a process group of its own
/// so Ctrl+C at the terminal reaches openmax, whose stop finishes the work,
/// and never a git command halfway. Returns stdout as git wrote it, or git's
/// own reason on one line.
///
/// A stop does not wait on git for long. A command of the launch
/// (`Step::Launch`, `Step::Checkout`) has made nothing to keep yet: none
/// starts once a signal has asked openmax to stop, and one that is running is
/// stopped (`stop_group`) at once, with a hook git runs, which can take as
/// long as it likes. One of the snapshot or the cleanup is stopped once it
/// has run `STOP_SETTLE` past the stop, and the worktree then stays.
fn run_git<S: AsRef<OsStr>>(dir: &Path, args: &[S], step: Step) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let grace = if step == Step::Settled { crate::STOP_SETTLE } else { Duration::ZERO };
    if grace.is_zero() && crate::signalled() {
        return Err(STOPPED.into());
    }
    let mut command = Command::new("git");
    command.arg("-C").arg(dir);
    if step != Step::Checkout {
        command.args(["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor="]);
    }
    command.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn().map_err(|e| format!("cannot run git: {e}"))?;
    // Each pipe drains on a thread of its own, so neither fills while git
    // writes the other, and this thread is free to look for a stop.
    let (sent, drained) = std::sync::mpsc::channel();
    let pipes: [Option<Box<dyn Read + Send>>; 2] =
        [child.stdout.take().map(|pipe| Box::new(pipe) as _), child.stderr.take().map(|pipe| Box::new(pipe) as _)];
    for (index, pipe) in pipes.into_iter().enumerate() {
        let sent = sent.clone();
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            let _ = sent.send((index, bytes));
        });
    }
    let mut output = [Vec::new(), Vec::new()];
    let mut open = output.len();
    let mut stop_at = None;
    while open > 0 {
        match drained.recv_timeout(STOP_CHECK) {
            Ok((index, bytes)) => {
                output[index] = bytes;
                open -= 1;
            }
            Err(_) if crate::signalled() => {
                if Instant::now() >= *stop_at.get_or_insert_with(|| Instant::now() + grace) {
                    stop_group(&mut child, &drained, open);
                    return Err(STOPPED.into());
                }
            }
            Err(_) => {}
        }
    }
    let status = child.wait().map_err(|e| format!("cannot run git: {e}"))?;
    let [stdout, stderr] = output;
    if !status.success() {
        return Err(one_line(String::from_utf8_lossy(&stderr).trim()));
    }
    Ok(stdout)
}

/// Stop a git command's process group, with whatever git started there:
/// SIGTERM, on which git removes what it had half made, then SIGKILL once
/// the group's pipes close or `STOP_GRACE` passes, for a hook that ignores
/// the first. Both reach the group before git is reaped, while its id can
/// name no other.
fn stop_group(child: &mut std::process::Child, drained: &std::sync::mpsc::Receiver<(usize, Vec<u8>)>, open: usize) {
    #[cfg(unix)]
    {
        if let Ok(group) = libc::pid_t::try_from(child.id()) {
            // SAFETY: kill only sends a signal, and the group's id is the pid
            // of this process's own unreaped child, so no other group has it.
            unsafe { libc::kill(-group, libc::SIGTERM) };
            let deadline = Instant::now() + STOP_GRACE;
            let mut open = open;
            while open > 0 && drained.recv_timeout(deadline.saturating_duration_since(Instant::now())).is_ok() {
                open -= 1;
            }
            // SAFETY: as above; git is reaped only below.
            unsafe { libc::kill(-group, libc::SIGKILL) };
        }
    }
    #[cfg(not(unix))]
    let _ = (drained, open);
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(unix)]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;
    PathBuf::from(OsStr::from_bytes(bytes))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A branch name is a slug of the task's first words (as many as fit in
    /// 32 characters) and eight hex digits, so it is a valid ref whatever
    /// the task holds.
    #[test]
    fn a_name_is_the_tasks_first_words_and_a_suffix() {
        let hex = |suffix: &str| suffix.len() == 8 && suffix.chars().all(|c| c.is_ascii_hexdigit());
        let name = name("Fix the flaky retry test in crates/core, then run `cargo test`!");
        let (slug, suffix) = name.rsplit_once('-').unwrap();
        assert_eq!(slug, "fix-the-flaky-retry-test-in", "adding crates would pass 32 characters");
        assert!(hex(suffix), "{name}");
        let bare = super::name("修正 ¿? ü");
        assert!(hex(&bare), "a task with no ascii word is the suffix alone: {bare}");
        let long = super::name(&"a".repeat(40));
        assert!(hex(&long), "a first word past the cap is left out: {long}");
        // The branch the recipe's example shows is the one its task gets.
        let example = super::name("make the flaky retry test deterministic");
        let recipe = open_max_core::spec::render("subagent").unwrap();
        assert!(recipe.contains(&format!("subagent/{}", &example[..example.len() - 8])), "{example}");
    }

    /// A turn still starting holds the running list's lock while it waits on
    /// the session index, as long as another openmax keeps it: the settle
    /// check counts a lock held past its bound as a turn that has not
    /// settled, instead of waiting with it, and waits out a lock held for a
    /// moment. A listed turn has not settled either.
    #[test]
    fn the_settle_check_never_waits_on_a_held_running_list() {
        let dir = crate::test_temp_dir("openmax-settled");
        let (core, _events) = Core::new(dir.clone()).unwrap();
        assert!(settled(&core), "no turn is listed");
        core.running.lock().unwrap().insert("s".into());
        assert!(!settled(&core), "a listed turn has not settled");
        core.running.lock().unwrap().clear();
        // A turn that settles holds the lock only to send its last event.
        let (taken, holding) = std::sync::mpsc::channel();
        let briefly = {
            let core = core.clone();
            std::thread::spawn(move || {
                let running = core.running.lock().unwrap();
                let _ = taken.send(());
                std::thread::sleep(Duration::from_millis(10));
                drop(running);
            })
        };
        holding.recv().unwrap();
        assert!(settled(&core), "a lock held for a moment is waited for");
        briefly.join().unwrap();
        let held = core.running.lock().unwrap();
        let (sent, verdict) = std::sync::mpsc::channel();
        let checking = core.clone();
        std::thread::spawn(move || sent.send(settled(&checking)));
        let verdict = verdict.recv_timeout(Duration::from_secs(2));
        drop(held);
        let _ = std::fs::remove_dir_all(dir);
        assert_eq!(verdict, Ok(false), "a list held past the bound counts as a turn not settled");
    }

    /// The commit subject is the task's first line on one line, cut at 60
    /// characters.
    #[test]
    fn a_headline_is_the_first_line_cut_short() {
        assert_eq!(headline("  fix it\nthen the rest"), "fix it");
        assert_eq!(headline(&"é".repeat(70)), format!("{}…", "é".repeat(60)));
        assert_eq!(headline(&"x".repeat(60)), "x".repeat(60));
    }
}
