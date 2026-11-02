use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use serde_json::{json, Map, Value};

const LIMIT: usize = 1024 * 1024;

fn regular_owned(file: &File) -> Result<(), String> {
    let metadata = file.metadata().map_err(|_| "cannot inspect provider catalog file")?;
    if !metadata.is_file() {
        return Err("provider catalog files must be regular files".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err("provider catalog files must be owned by this user".into());
        }
    }
    Ok(())
}

fn open(path: &Path, lock: bool) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(lock).create(lock);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    options.open(path)
}

fn read_document(dir: &Path) -> Result<Value, String> {
    let mut document = match open(&super::providers_path(dir), false) {
        Ok(file) => {
            regular_owned(&file).map_err(|error| format!("cannot read providers.json: {error}"))?;
            let mut bytes = Vec::new();
            file.take(LIMIT as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|_| "cannot read providers.json")?;
            if bytes.len() > LIMIT {
                return Err("provider catalog exceeds its size limit".into());
            }
            let text = std::str::from_utf8(&bytes).map_err(|_| "providers.json is invalid; catalog was not written")?;
            super::parse_providers_file(text).map_err(|_| "providers.json is invalid; catalog was not written")?;
            serde_json::from_str::<Value>(text).map_err(|_| "providers.json is invalid; catalog was not written")?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => json!({"providers":{}}),
        Err(_) => return Err("cannot read providers.json".into()),
    };
    let root = document
        .as_object_mut()
        .ok_or("providers.json must contain an object")?;
    root.entry("providers")
        .or_insert_with(|| json!({}))
        .as_object()
        .ok_or("providers.json lacks a providers object")?;
    Ok(document)
}

/// Read raw provider entries without creating files or acquiring a write lock.
pub fn read_catalog(dir: &Path) -> Result<Map<String, Value>, String> {
    read_document(dir)?["providers"]
        .as_object()
        .cloned()
        .ok_or_else(|| "providers.json lacks a providers object".into())
}

/// Edit one provider catalog under a shared process lock, preserving unrelated JSON.
pub async fn edit_catalog<T>(
    dir: &Path,
    edit: impl FnOnce(&mut Map<String, Value>) -> Result<T, String>,
) -> Result<T, String> {
    std::fs::create_dir_all(dir).map_err(|_| "cannot create provider directory")?;
    let file = open(&dir.join("providers.lock"), true).map_err(|_| "cannot open provider catalog lock")?;
    regular_owned(&file)?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let _lock = loop {
        let candidate = file.try_clone().map_err(|_| "cannot open provider catalog lock")?;
        match crate::sessions::FileLock::try_take(candidate) {
            Ok(lock) => break lock,
            Err(std::fs::TryLockError::WouldBlock) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            _ => return Err("provider catalog is busy; retry after the other operation finishes".into()),
        }
    };
    let path = super::providers_path(dir);
    let mut document = read_document(dir)?;
    let root = document
        .as_object_mut()
        .ok_or("providers.json must contain an object")?;
    let providers = root
        .entry("providers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or("providers.json lacks a providers object")?;
    let result = edit(providers)?;
    let bytes = serde_json::to_vec_pretty(&document).map_err(|_| "cannot encode provider catalog")?;
    if bytes.len() > LIMIT {
        return Err("provider catalog exceeds its size limit".into());
    }
    super::parse_providers_file(std::str::from_utf8(&bytes).map_err(|_| "cannot encode provider catalog")?)
        .map_err(|_| "provider catalog is invalid; catalog was not written")?;
    crate::sessions::replace_atomic(&path, &bytes, 0o600, |_| Ok(())).map_err(|_| "cannot save provider catalog")?;
    if let Ok(directory) = File::open(dir) {
        let _ = directory.sync_all();
    }
    super::invalidate_providers_cache();
    Ok(result)
}
