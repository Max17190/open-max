//! Persistent trust decisions for project roots.
//!
//! An agent session can execute `bash`, project-local tools, and hooks with
//! the user's host authority. Trust is therefore resolved before any turn or
//! repository behavior starts, not after extension processes load.
//! Decisions are exact canonical paths in `~/.openmax/trust.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::config::ApprovalMode;

use fs2::FileExt;
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

/// Persist trust for the exact canonical project root.
pub fn trust_project(data_dir: &Path, project_root: &Path) -> Result<PathBuf, String> {
    let canonical = canonical_project(project_root)?;
    update(data_dir, |file| {
        file.projects.push(canonical.clone());
        file.projects.sort();
        file.projects.dedup();
        Ok(())
    })?;
    Ok(canonical)
}

/// Snapshot project choices at launch. They are never adopted from disk
/// during a turn; only a frontend's explicit selection changes the live mode.
pub(crate) fn approval_modes(data_dir: &Path) -> Result<BTreeMap<PathBuf, ApprovalMode>, String> {
    let file = load(data_dir)?;
    Ok(file.approval_modes.into_iter().filter(|(root, _)| {
        file.projects.iter().any(|trusted| root.starts_with(trusted))
    }).collect())
}

/// Resolve a fresh process's mode, also used by the diagnostic CLI.
pub(crate) fn approval_mode(data_dir: &Path, project_root: &Path, default: ApprovalMode) -> Result<ApprovalMode, String> {
    let canonical = canonical_project(project_root)?;
    Ok(approval_modes(data_dir)?.get(&canonical).copied().unwrap_or(default))
}

/// Save an explicit choice for one canonical project, without changing the
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
    lock.lock_exclusive()
        .map_err(|e| format!("cannot lock {}: {e}", lock_path.display()))?;

    let mut file = load(data_dir)?;
    change(&mut file)?;
    let json = serde_json::to_vec_pretty(&file).map_err(|e| e.to_string())?;
    crate::sessions::write_atomic(&trust_path(data_dir), json)?;
    FileExt::unlock(&lock).map_err(|e| format!("cannot unlock {}: {e}", lock_path.display()))?;
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
        assert_eq!(approval_mode(&data, &child, ApprovalMode::Ask).unwrap(), ApprovalMode::Ask);
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
