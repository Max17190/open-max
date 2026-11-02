use std::path::PathBuf;
use std::time::Duration;

use open_max_core::{chatgpt, providers};
use serde_json::{json, Value};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("openmax-provider-setup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_setup_and_account_catalog_preserve_unrelated_configuration() {
    let scratch = Scratch::new();
    let path = providers::providers_path(&scratch.0);
    let original = json!({"extra":{"version":1},"providers":{"existing":{
        "base_url":"https://fixture.invalid/v1","api_key":"fixture-key","future_field":[1,2]
    }}});
    std::fs::write(&path, original.to_string()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    let (locked, ready) = tokio::sync::oneshot::channel();
    let dir = scratch.0.clone();
    let setup = tokio::spawn(async move {
        providers::edit_catalog(&dir, |catalog| {
            locked.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            catalog.insert("custom".into(), json!({"base_url":"https://custom.invalid/v1"}));
            Ok(())
        })
        .await
        .unwrap();
    });
    let dir = scratch.0.clone();
    let account = tokio::spawn(async move {
        ready.await.unwrap();
        chatgpt::save_catalog(
            &dir,
            "main",
            &json!({"models":[{
                "slug":"model-one","visibility":"list","context_window":8192
            }]}),
        )
        .await
        .unwrap();
    });
    tokio::time::timeout(Duration::from_secs(2), async {
        setup.await.unwrap();
        account.await.unwrap();
    })
    .await
    .unwrap();
    let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(saved["extra"], original["extra"]);
    assert_eq!(saved["providers"]["existing"], original["providers"]["existing"]);
    assert_eq!(saved["providers"]["custom"]["base_url"], "https://custom.invalid/v1");
    assert_eq!(saved["providers"]["chatgpt-main"]["models"][0]["id"], "model-one");
    assert_eq!(providers::load_providers(&scratch.0).len(), 3);
    assert_eq!(
        providers::read_catalog(&scratch.0).unwrap(),
        saved["providers"].as_object().unwrap().clone()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    }
}

#[tokio::test]
async fn invalid_catalogs_and_rejected_edits_preserve_the_original_bytes() {
    let scratch = Scratch::new();
    let path = providers::providers_path(&scratch.0);
    assert!(providers::read_catalog(&scratch.0).unwrap().is_empty());
    assert_eq!(std::fs::read_dir(&scratch.0).unwrap().count(), 0);
    for original in [
        "{",
        "[]",
        r#"{"providers":[]}"#,
        r#"{"providers":{"bad":{"api_key":"private-fixture"}}}"#,
    ] {
        std::fs::write(&path, original).unwrap();
        let error = providers::edit_catalog(&scratch.0, |_| -> Result<(), String> {
            panic!("an invalid catalog reached the edit callback");
        })
        .await
        .unwrap_err();
        assert!(!error.contains("private-fixture"));
        assert!(providers::read_catalog(&scratch.0).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original.as_bytes());
    }
    let oversized = vec![b' '; 1_048_577];
    std::fs::write(&path, &oversized).unwrap();
    assert!(providers::read_catalog(&scratch.0).unwrap_err().contains("size limit"));
    assert!(providers::edit_catalog(&scratch.0, |_| Ok(())).await.is_err());
    assert_eq!(std::fs::read(&path).unwrap(), oversized);
    let original = br#"{"providers":{},"unknown":true}"#;
    std::fs::write(&path, original).unwrap();
    assert!(providers::edit_catalog(&scratch.0, |catalog| {
        catalog.insert("bad".into(), json!({"api_key":"private-fixture"}));
        Ok(())
    })
    .await
    .unwrap_err()
    .contains("invalid"));
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert!(providers::edit_catalog(&scratch.0, |catalog| {
        catalog.insert("valid".into(), json!({"base_url":"https://fixture.invalid/v1"}));
        Err::<(), _>("cancelled".into())
    })
    .await
    .unwrap_err()
    .contains("cancelled"));
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[cfg(unix)]
#[tokio::test]
async fn catalog_and_lock_symlinks_are_refused_without_changing_their_targets() {
    use std::os::unix::fs::symlink;
    for name in ["providers.json", "providers.lock"] {
        let scratch = Scratch::new();
        let target = scratch.0.join("target");
        let original = br#"{"providers":{}}"#;
        std::fs::write(&target, original).unwrap();
        symlink(&target, scratch.0.join(name)).unwrap();
        if name == "providers.json" {
            assert!(providers::read_catalog(&scratch.0).is_err());
        }
        assert!(providers::edit_catalog(&scratch.0, |_| Ok(())).await.is_err());
        assert_eq!(std::fs::read(&target).unwrap(), original);
        assert!(std::fs::symlink_metadata(scratch.0.join(name))
            .unwrap()
            .file_type()
            .is_symlink());
    }
}

#[tokio::test(start_paused = true)]
async fn a_busy_catalog_lock_has_a_bounded_wait() {
    let scratch = Scratch::new();
    let held = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(scratch.0.join("providers.lock"))
        .unwrap();
    held.lock().unwrap();
    let started = tokio::time::Instant::now();
    let error = providers::edit_catalog(&scratch.0, |_| Ok(())).await.unwrap_err();
    assert!(error.contains("busy"));
    assert_eq!(tokio::time::Instant::now() - started, Duration::from_secs(5));
    assert!(!providers::providers_path(&scratch.0).exists());
}

#[tokio::test]
async fn temperature_omission_requires_explicit_provider_compatibility() {
    let scratch = Scratch::new();
    providers::edit_catalog(&scratch.0, |catalog| {
        for name in ["original", "automatic"] {
            catalog.insert(
                name.into(),
                json!({"base_url":"https://fixture.invalid/v1",
                "models":[{"id":"model-one","context_tokens":8192}]}),
            );
        }
        catalog["automatic"]["compat"] = json!({"omit_temperature":true});
        Ok(())
    })
    .await
    .unwrap();
    let mut settings = open_max_core::config::Settings {
        provider: Some("original".into()),
        model: "model-one".into(),
        temperature: Some(0.25),
        ..Default::default()
    };
    assert_eq!(
        providers::resolve(&settings, &scratch.0).unwrap().temperature,
        Some(0.25)
    );
    settings.provider = Some("automatic".into());
    assert_eq!(providers::resolve(&settings, &scratch.0).unwrap().temperature, None);
    settings.base_url = "https://fixture.invalid/v1".into();
    settings.api_key = Some("fixture-fallback-key".into());
    providers::edit_catalog(&scratch.0, |catalog| {
        catalog.remove("automatic");
        Ok(())
    }).await.unwrap();
    assert!(matches!(providers::resolve(&settings, &scratch.0),
        Err(providers::ResolveError::UnknownProvider(name)) if name == "automatic"));
    settings.provider = Some("original".into());
    assert_eq!(providers::resolve(&settings, &scratch.0).unwrap().temperature, Some(0.25));
}
