//! Persistent trust decisions for project roots.
//!
//! An agent session can execute `bash`, project-local tools, and hooks with
//! the user's host authority. Trust is therefore resolved before any turn or
//! repository behavior starts, not after extension processes load.
//! Decisions are exact canonical paths in `~/.openmax/trust.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::config::ApprovalMode;

use serde::{Deserialize, Serialize};

const TRUST_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustFile {
    version: u32,
    #[serde(default)]
    projects: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    approval_modes: BTreeMap<PathBuf, ApprovalMode>,
}

impl Default for TrustFile {
    fn default() -> Self {
        Self {
            version: TRUST_VERSION,
            projects: Vec::new(),
            approval_modes: BTreeMap::new(),
        }
    }
}

fn trust_path(data_dir: &Path) -> PathBuf {
    data_dir.join("trust.json")
}

fn trust_lock_path(data_dir: &Path) -> PathBuf {
    data_dir.join("trust.lock")
}

fn canonical_project(project_root: &Path) -> Result<PathBuf, String> {
    std::fs::canonicalize(project_root).map_err(|e| {
        format!(
            "cannot resolve project root {} for trust: {e}",
            project_root.display()
        )
    })
}

fn load(data_dir: &Path) -> Result<TrustFile, String> {
    let path = trust_path(data_dir);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(TrustFile::default());
        }
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    let file: TrustFile =
        serde_json::from_str(&text).map_err(|e| format!("invalid {}: {e}", path.display()))?;
    if file.version != TRUST_VERSION {
        return Err(format!(
            "unsupported trust file version {} in {}",
            file.version,
            path.display()
        ));
    }
    Ok(file)
}

/// True only when the exact canonical project root was trusted previously.
/// Malformed trust state is an error so callers fail closed.
pub fn is_trusted(data_dir: &Path, project_root: &Path) -> Result<bool, String> {
    let canonical = canonical_project(project_root)?;
    // A trusted root covers its subtree: the agent already holds full
    // authority inside that root, so a worktree or subdirectory of it (the
    // delegation case) is not a widening. Comparison is component-wise on
    // canonical paths, so /a/b never covers /a/bc.
    Ok(load(data_dir)?
        .projects
        .iter()
        .any(|p| canonical == *p || canonical.starts_with(p)))
}

/// Persist trust for the exact canonical project root and record no mode, so
/// it runs under an enclosing root's saved choice or the settings value: the
/// state of a project trusted before trust grants recorded a mode. A human
/// trust grant goes through [`grant_trust`].
pub fn trust_project(data_dir: &Path, project_root: &Path) -> Result<PathBuf, String> {
    Ok(grant(data_dir, project_root, None)?.0)
}

/// The human trust grant (the interactive prompt and `--trust-project`):
/// persist trust for the exact canonical project root and, in the same
/// write, save `mode` as its approval mode. A settings default cannot carry
/// that choice, because every settings save writes the whole file and so
/// most settings files already say `ask` without the user choosing it.
///
/// Only a root that gains trust here gets a mode. One already covered by
/// trust keeps what governs it now (its own or an enclosing saved choice,
/// else the settings value): granting again must not overwrite a choice the
/// user made since, and a project trusted before grants recorded a mode keeps
/// its behavior. Returns the canonical root and the mode recorded, if any.
pub fn grant_trust(
    data_dir: &Path,
    project_root: &Path,
    mode: ApprovalMode,
) -> Result<(PathBuf, Option<ApprovalMode>), String> {
    grant(data_dir, project_root, Some(mode))
}

fn grant(
    data_dir: &Path,
    project_root: &Path,
    mode: Option<ApprovalMode>,
) -> Result<(PathBuf, Option<ApprovalMode>), String> {
    let canonical = canonical_project(project_root)?;
    let mut recorded = None;
    update(data_dir, |file| {
        let covered = file.projects.iter().any(|root| canonical.starts_with(root));
        if let Some(mode) = mode.filter(|_| !covered) {
            file.approval_modes.insert(canonical.clone(), mode);
            recorded = Some(mode);
        }
        file.projects.push(canonical.clone());
        file.projects.sort();
        file.projects.dedup();
        Ok(())
    })?;
    Ok((canonical, recorded))
}

/// Snapshot project choices at launch. They are never adopted from disk
/// during a turn; only a frontend's explicit selection changes the live mode.
pub(crate) fn approval_modes(data_dir: &Path) -> Result<BTreeMap<PathBuf, ApprovalMode>, String> {
    let file = load(data_dir)?;
    Ok(file.approval_modes.into_iter().filter(|(root, _)| {
        file.projects.iter().any(|trusted| root.starts_with(trusted))
    }).collect())
}

/// The choice governing `canonical`: its own saved entry, else the nearest
/// enclosing root's. Trust already covers a root's subtree, so a child started
/// in a worktree or subdirectory (the delegation case) must run under the
/// same mode, not fall back to the default where a headless child declines
/// every approval. A nested root's explicit choice still wins. Ancestors are
/// component-wise, so /a/bc never takes the choice saved for /a/b. `modes`
/// must come from [`approval_modes`], which keeps trusted roots only.
pub(crate) fn nearest_approval_mode(modes: &BTreeMap<PathBuf, ApprovalMode>, canonical: &Path) -> Option<ApprovalMode> {
    canonical.ancestors().find_map(|root| modes.get(root).copied())
}

/// Resolve a fresh process's mode, also used by the diagnostic CLI.
pub(crate) fn approval_mode(data_dir: &Path, project_root: &Path, default: ApprovalMode) -> Result<ApprovalMode, String> {
    let canonical = canonical_project(project_root)?;
    Ok(nearest_approval_mode(&approval_modes(data_dir)?, &canonical).unwrap_or(default))
}

/// Save an explicit choice for one canonical project, which its
/// subdirectories without a choice of their own follow, without changing the
/// default for other projects or granting any content approvals.
pub(crate) fn set_approval_mode(
    data_dir: &Path,
    project_root: &Path,
    mode: ApprovalMode,
) -> Result<PathBuf, String> {
    if std::env::var_os("OPENMAX_SESSION").is_some() {
        return Err("approval mode changes require a human-controlled frontend; this process was started by an agent".into());
    }
    let canonical = canonical_project(project_root)?;
    update(data_dir, |file| {
        if !file.projects.iter().any(|root| canonical.starts_with(root)) {
            return Err(format!("project {} is not trusted", canonical.display()));
        }
        file.approval_modes.insert(canonical.clone(), mode);
        Ok(())
    })?;
    Ok(canonical)
}

fn update(data_dir: &Path, change: impl FnOnce(&mut TrustFile) -> Result<(), String>) -> Result<(), String> {
    std::fs::create_dir_all(data_dir)
        .map_err(|e| format!("cannot create {}: {e}", data_dir.display()))?;
    let lock_path = trust_lock_path(data_dir);
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|e| format!("cannot open {}: {e}", lock_path.display()))?;
    lock.lock()
        .map_err(|e| format!("cannot lock {}: {e}", lock_path.display()))?;

    let mut file = load(data_dir)?;
    change(&mut file)?;
    let json = serde_json::to_vec_pretty(&file).map_err(|e| e.to_string())?;
    crate::sessions::write_atomic(&trust_path(data_dir), json)?;
    lock.unlock().map_err(|e| format!("cannot unlock {}: {e}", lock_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("openmax-trust-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A trusted root covers its worktrees and subdirectories, but never a
    /// sibling whose name merely extends it.
    #[test]
    fn subtree_of_a_trusted_root_is_trusted() {
        let data = temp_dir("subtree-data");
        let root = temp_dir("subtree-root");
        let child = root.join(".worktrees").join("task-a");
        std::fs::create_dir_all(&child).unwrap();
        let sibling = root
            .parent()
            .unwrap()
            .join(format!("{}-evil", root.file_name().unwrap().to_str().unwrap()));
        std::fs::create_dir_all(&sibling).unwrap();

        trust_project(&data, &root).unwrap();
        assert!(is_trusted(&data, &root).unwrap());
        assert!(is_trusted(&data, &child).unwrap(), "worktree under the root");
        assert!(!is_trusted(&data, &sibling).unwrap(), "path-prefix sibling must not ride along");
        let _ = std::fs::remove_dir_all(&data);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&sibling);
    }

    #[test]
    fn missing_store_is_untrusted_then_exact_path_persists() {
        let data = temp_dir("data");
        let project = temp_dir("project");
        let other = temp_dir("other");

        assert!(!is_trusted(&data, &project).unwrap());
        let canonical = trust_project(&data, &project).unwrap();
        assert_eq!(canonical, std::fs::canonicalize(&project).unwrap());
        assert!(is_trusted(&data, &project).unwrap());
        assert!(!is_trusted(&data, &other).unwrap());

        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(project);
        let _ = std::fs::remove_dir_all(other);
    }

    #[test]
    fn malformed_or_unsupported_store_fails_closed() {
        let data = temp_dir("bad-data");
        let project = temp_dir("bad-project");
        std::fs::write(data.join("trust.json"), r#"{"version":1,"projectz":[]}"#).unwrap();
        assert!(is_trusted(&data, &project).is_err());
        assert!(trust_project(&data, &project).is_err());

        std::fs::write(data.join("trust.json"), r#"{"version":2,"projects":[]}"#).unwrap();
        assert!(is_trusted(&data, &project).is_err());
        assert!(trust_project(&data, &project).is_err());
        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(project);
    }

    /// A child started in a subdirectory or worktree of a trusted project
    /// (the delegation case) must run under that project's saved mode: in
    /// ask, a headless child declines every approval and delegated work
    /// fails on its first write. A nested root's own choice still wins, and
    /// a mode saved on a root that is not trusted never reaches its subtree.
    #[test]
    fn approval_mode_comes_from_the_nearest_trusted_ancestor() {
        let data = temp_dir("inherit-data");
        let root = temp_dir("inherit-root");
        let sub = root.join("sub");
        let deep = sub.join("dir");
        std::fs::create_dir_all(&deep).unwrap();
        let sibling = root
            .parent()
            .unwrap()
            .join(format!("{}-evil", root.file_name().unwrap().to_str().unwrap()));
        std::fs::create_dir_all(&sibling).unwrap();

        trust_project(&data, &root).unwrap();
        set_approval_mode(&data, &root, ApprovalMode::Auto).unwrap();
        assert_eq!(approval_mode(&data, &deep, ApprovalMode::Ask).unwrap(), ApprovalMode::Auto, "inherits the project's choice");
        assert_eq!(approval_mode(&data, &sibling, ApprovalMode::Ask).unwrap(), ApprovalMode::Ask, "path-prefix sibling must not ride along");

        set_approval_mode(&data, &sub, ApprovalMode::Ask).unwrap();
        assert_eq!(approval_mode(&data, &deep, ApprovalMode::Readonly).unwrap(), ApprovalMode::Ask, "the nearest saved choice wins");
        assert_eq!(approval_mode(&data, &root, ApprovalMode::Readonly).unwrap(), ApprovalMode::Auto, "a nested choice never flows up");

        // Trust only the inner root; the outer root's saved mode is stale.
        let untrusted = temp_dir("inherit-untrusted-data");
        let outer = std::fs::canonicalize(&root).unwrap();
        let inner = std::fs::canonicalize(&sub).unwrap();
        update(&untrusted, |file| {
            file.projects.push(inner);
            file.approval_modes.insert(outer, ApprovalMode::Auto);
            Ok(())
        }).unwrap();
        assert_eq!(approval_mode(&untrusted, &deep, ApprovalMode::Ask).unwrap(), ApprovalMode::Ask, "an untrusted root's mode is ignored");

        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(untrusted);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(sibling);
    }

    /// A grant records its mode only for a root it newly trusts. Granting a
    /// root again, or a subdirectory of a trusted one, leaves the mode that
    /// governs it as it was: a choice saved since the first grant survives,
    /// and a project trusted before grants recorded a mode keeps resolving
    /// to the settings value.
    #[test]
    fn a_grant_records_its_mode_only_for_a_root_it_newly_trusts() {
        let data = temp_dir("grant-data");
        let fresh = temp_dir("grant-fresh");
        let earlier = temp_dir("grant-earlier");
        let sub = fresh.join("sub");
        std::fs::create_dir_all(&sub).unwrap();

        let (root, recorded) = grant_trust(&data, &fresh, ApprovalMode::Auto).unwrap();
        assert_eq!((root, recorded), (std::fs::canonicalize(&fresh).unwrap(), Some(ApprovalMode::Auto)));
        assert_eq!(approval_mode(&data, &fresh, ApprovalMode::Ask).unwrap(), ApprovalMode::Auto);

        set_approval_mode(&data, &fresh, ApprovalMode::Readonly).unwrap();
        assert_eq!(grant_trust(&data, &fresh, ApprovalMode::Auto).unwrap().1, None, "granting again records nothing");
        assert_eq!(approval_mode(&data, &fresh, ApprovalMode::Ask).unwrap(), ApprovalMode::Readonly, "a choice saved since survives");
        assert_eq!(grant_trust(&data, &sub, ApprovalMode::Ask).unwrap().1, None, "a covered subdirectory records nothing");
        assert!(is_trusted(&data, &sub).unwrap());
        assert_eq!(approval_mode(&data, &sub, ApprovalMode::Ask).unwrap(), ApprovalMode::Readonly, "it keeps its project's choice");

        trust_project(&data, &earlier).unwrap();
        assert_eq!(grant_trust(&data, &earlier, ApprovalMode::Auto).unwrap().1, None);
        assert_eq!(approval_mode(&data, &earlier, ApprovalMode::Ask).unwrap(), ApprovalMode::Ask, "an earlier trust keeps the settings value");

        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(fresh);
        let _ = std::fs::remove_dir_all(earlier);
    }

    #[test]
    fn concurrent_mode_selections_preserve_both_projects() {
        let data = temp_dir("modes-data");
        let a = temp_dir("mode-a");
        let b = temp_dir("mode-b");
        trust_project(&data, &a).unwrap();
        trust_project(&data, &b).unwrap();
        let handles: Vec<_> = [(a.clone(), ApprovalMode::Auto), (b.clone(), ApprovalMode::Readonly)]
            .into_iter().map(|(root, mode)| {
                let data = data.clone();
                std::thread::spawn(move || set_approval_mode(&data, &root, mode).unwrap())
            }).collect();
        for handle in handles { handle.join().unwrap(); }
        assert_eq!(approval_mode(&data, &a, ApprovalMode::Ask).unwrap(), ApprovalMode::Auto);
        assert_eq!(approval_mode(&data, &b, ApprovalMode::Ask).unwrap(), ApprovalMode::Readonly);
        let child = a.join("nested");
        std::fs::create_dir(&child).unwrap();
        assert_eq!(approval_mode(&data, &child, ApprovalMode::Ask).unwrap(), ApprovalMode::Auto, "a subdirectory runs under its project's choice");
        #[cfg(unix)] {
            let alias = data.join("alias");
            std::os::unix::fs::symlink(&a, &alias).unwrap();
            assert_eq!(approval_mode(&data, &alias, ApprovalMode::Ask).unwrap(), ApprovalMode::Auto);
        }
        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(a);
        let _ = std::fs::remove_dir_all(b);
    }

    #[test]
    fn concurrent_writers_do_not_lose_trust_entries() {
        let data = temp_dir("concurrent-data");
        let project_a = temp_dir("concurrent-a");
        let project_b = temp_dir("concurrent-b");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));

        let handles: Vec<_> = [project_a.clone(), project_b.clone()]
            .into_iter()
            .map(|project| {
                let data = data.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    trust_project(&data, &project).unwrap();
                })
            })
            .collect();
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }

        assert!(is_trusted(&data, &project_a).unwrap());
        assert!(is_trusted(&data, &project_b).unwrap());
        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(project_a);
        let _ = std::fs::remove_dir_all(project_b);
    }

    /// trust.lock speaks the same flock(2) protocol as the session locks: a
    /// write must wait while an older binary on this data dir holds it, or
    /// the two read-modify-writes interleave and one trust decision is lost.
    #[cfg(unix)]
    #[test]
    fn a_trust_write_waits_for_a_raw_flock_holder() {
        let data = temp_dir("flock-data");
        let project = temp_dir("flock-project");
        let older = crate::sessions::raw_flock(&trust_lock_path(&data)).expect("trust.lock is idle");
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let writer = {
            let (data, project) = (data.clone(), project.clone());
            std::thread::spawn(move || {
                trust_project(&data, &project).unwrap();
                done_tx.send(()).unwrap();
            })
        };
        assert!(
            done_rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "a trust write must wait while an older binary holds trust.lock"
        );
        drop(older);
        writer.join().unwrap();
        assert!(is_trusted(&data, &project).unwrap());
        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(project);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_alias_resolves_to_the_same_trust_identity() {
        use std::os::unix::fs::symlink;

        let data = temp_dir("link-data");
        let project = temp_dir("link-project");
        let alias_parent = temp_dir("link-parent");
        let alias = alias_parent.join("alias");
        symlink(&project, &alias).unwrap();

        trust_project(&data, &project).unwrap();
        assert!(is_trusted(&data, &alias).unwrap());

        let _ = std::fs::remove_dir_all(data);
        let _ = std::fs::remove_dir_all(alias_parent);
        let _ = std::fs::remove_dir_all(project);
    }
}
