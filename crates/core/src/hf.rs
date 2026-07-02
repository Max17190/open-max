//! HuggingFace hub helpers: local cache inspection, repo sizing via the hub
//! API, and explicit model downloads with byte-accurate progress (total from
//! the API, done bytes from watching the cache directory grow).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};

use crate::state::{Core, CoreEvent, DownloadEvent};

/// An in-flight download; at most one at a time (weights are many GB).
pub struct DownloadProc {
    pub repo: String,
    pub child: tokio::process::Child,
}

#[derive(Clone, Debug)]
pub struct InstalledModel {
    pub repo: String,
    pub bytes: u64,
}

/// The hub cache: $HF_HOME/hub, defaulting to ~/.cache/huggingface/hub.
pub fn hub_cache_dir() -> PathBuf {
    if let Some(hf_home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join(".cache").join("huggingface").join("hub")
}

/// Cache directory for one repo: models--org--name.
pub fn repo_cache_dir(repo: &str) -> PathBuf {
    hub_cache_dir().join(format!("models--{}", repo.replace('/', "--")))
}

pub fn is_installed(repo: &str) -> bool {
    repo_cache_dir(repo).join("snapshots").is_dir()
}

/// Bytes on disk under `path`. Symlinks are not followed (snapshot dirs link
/// into blobs, which are counted once via the blobs dir itself).
pub fn dir_size(path: &PathBuf) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(path) else { return 0 };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total += dir_size(&entry.path());
        } else if meta.is_file() {
            total += meta.len();
        }
    }
    total
}

/// Models present in the local hub cache, largest first.
pub fn installed_models() -> Vec<InstalledModel> {
    let mut models = Vec::new();
    let Ok(entries) = std::fs::read_dir(hub_cache_dir()) else { return models };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(stripped) = name.strip_prefix("models--") else { continue };
        let repo = stripped.replacen("--", "/", 1);
        let bytes = dir_size(&entry.path());
        if bytes > 0 {
            models.push(InstalledModel { repo, bytes });
        }
    }
    models.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    models
}

pub fn delete_model(repo: &str) -> Result<(), String> {
    let dir = repo_cache_dir(repo);
    if !dir.exists() {
        return Err(format!("{repo} is not in the local cache"));
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("could not delete {repo}: {e}"))
}

/// Total size of a repo's files from the hub API.
pub async fn repo_total_bytes(repo: &str) -> Result<u64, String> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .map_err(|e| format!("hub API request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("hub API returned {} for {repo}", resp.status()));
    }
    let entries: Vec<serde_json::Value> = resp.json().await.map_err(|e| format!("bad hub API response: {e}"))?;
    Ok(entries.iter().filter_map(|e| e["size"].as_u64()).sum())
}

/// The downloader CLI inside the managed venv (`hf`, with the legacy name as
/// fallback).
fn downloader_bin(core: &Core) -> Option<PathBuf> {
    let bin_dir = core.data_dir.join("mlx-venv").join("bin");
    for name in ["hf", "huggingface-cli"] {
        let p = bin_dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Start downloading `repo` into the hub cache, emitting Progress events until
/// the underlying process exits. Errors if a download is already running or
/// the environment is not set up.
pub fn start_download(core: Arc<Core>, repo: String) -> Result<(), String> {
    let Some(bin) = downloader_bin(&core) else {
        return Err("the MLX environment is not set up yet; run setup first".into());
    };
    {
        let slot = core.download.lock().unwrap();
        if slot.is_some() {
            return Err("another download is already running".into());
        }
    }

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg("download")
        .arg(&repo)
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(true);
    let mut child = cmd.spawn().map_err(|e| format!("failed to start downloader: {e}"))?;

    // Keep the tail of the process output for the failure message.
    let tail: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    if let Some(out) = child.stdout.take() {
        capture_tail(out, tail.clone());
    }
    if let Some(err) = child.stderr.take() {
        capture_tail(err, tail.clone());
    }

    *core.download.lock().unwrap() = Some(DownloadProc { repo: repo.clone(), child });

    tokio::spawn(async move {
        let total = repo_total_bytes(&repo).await.unwrap_or(0);
        let dir = repo_cache_dir(&repo);
        loop {
            tokio::time::sleep(Duration::from_millis(700)).await;
            let exited = {
                let mut slot = core.download.lock().unwrap();
                match slot.as_mut() {
                    Some(proc) => proc.child.try_wait().ok().flatten(),
                    None => break, // cancelled and cleared elsewhere
                }
            };
            let done = dir_size(&dir);
            core.send(CoreEvent::Download(DownloadEvent::Progress {
                repo: repo.clone(),
                done_bytes: done,
                total_bytes: total,
            }));
            if let Some(status) = exited {
                *core.download.lock().unwrap() = None;
                let ok = status.success();
                let message = if ok {
                    format!("{repo} downloaded")
                } else {
                    let t = tail.lock().unwrap().join("\n");
                    format!("download failed (exit {:?})\n{t}", status.code())
                };
                core.send(CoreEvent::Download(DownloadEvent::Done { repo: repo.clone(), ok, message }));
                return;
            }
        }
    });

    Ok(())
}

fn capture_tail<R>(stream: R, tail: Arc<std::sync::Mutex<Vec<String>>>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let mut t = tail.lock().unwrap();
            t.push(line);
            if t.len() > 20 {
                t.remove(0);
            }
        }
    });
}

/// Kill the in-flight download, if any.
pub fn cancel_download(core: &Core) {
    let mut slot = core.download.lock().unwrap();
    if let Some(proc) = slot.as_mut() {
        let _ = proc.child.start_kill();
    }
    // The poll task notices the cleared slot or process exit and reports Done.
}
