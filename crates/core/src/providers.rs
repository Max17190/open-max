//! Named OpenAI-compatible providers (`~/.openmax/providers.json`).
//! Missing file is free: the flat `base_url` / `api_key` / `model` settings path
//! continues to work unchanged.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{Settings, DEFAULT_MLX_PORT};

/// Wire quirks for picky OpenAI-compatible servers.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CompatFlags {
    /// When true, send `max_completion_tokens` instead of `max_tokens`.
    pub use_max_completion_tokens: bool,
    /// When false, omit `stream_options` (some local servers reject unknown fields).
    pub send_stream_options: bool,
}

impl CompatFlags {
    fn defaults_for_missing() -> Self {
        Self {
            use_max_completion_tokens: false,
            send_stream_options: true,
        }
    }
}

// serde default for send_stream_options is false via Default; we want true when
// the field is omitted from JSON. Custom deserialize via Option merge on load.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct CompatFlagsFile {
    use_max_completion_tokens: Option<bool>,
    send_stream_options: Option<bool>,
}

impl From<CompatFlagsFile> for CompatFlags {
    fn from(f: CompatFlagsFile) -> Self {
        Self {
            use_max_completion_tokens: f.use_max_completion_tokens.unwrap_or(false),
            send_stream_options: f.send_stream_options.unwrap_or(true),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderModel {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub context_tokens: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct ProviderConfigFile {
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    /// Env var name, or list of names (first non-empty wins).
    #[serde(default)]
    api_key_env: Option<ApiKeyEnv>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    models: Vec<ProviderModel>,
    #[serde(default)]
    compat: CompatFlagsFile,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum ApiKeyEnv {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub api_key_env: Vec<String>,
    pub headers: BTreeMap<String, String>,
    pub models: Vec<ProviderModel>,
    pub compat: CompatFlags,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ProvidersFile {
    #[serde(default)]
    providers: BTreeMap<String, ProviderConfigFile>,
}

/// Resolved endpoint used for one completion request.
#[derive(Clone, Debug)]
pub struct ActiveEndpoint {
    pub provider: Option<String>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub headers: Vec<(String, String)>,
    pub model: String,
    pub context_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub compat: CompatFlags,
}

pub fn providers_path(data_dir: &Path) -> PathBuf {
    data_dir.join("providers.json")
}

use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

struct ProvidersCache {
    data_dir: PathBuf,
    /// None when the file was missing or unreadable at last load.
    mtime: Option<SystemTime>,
    map: BTreeMap<String, ProviderConfig>,
}

static PROVIDERS_CACHE: OnceLock<Mutex<ProvidersCache>> = OnceLock::new();

/// Drop cached providers so the next load re-reads disk (settings/provider edits).
pub fn invalidate_providers_cache() {
    if let Some(lock) = PROVIDERS_CACHE.get() {
        if let Ok(mut cache) = lock.lock() {
            cache.data_dir.clear();
            cache.mtime = None;
            cache.map.clear();
        }
    }
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn parse_providers_file(text: &str) -> BTreeMap<String, ProviderConfig> {
    let Ok(file) = serde_json::from_str::<ProvidersFile>(text) else {
        return BTreeMap::new();
    };
    file.providers
        .into_iter()
        .filter_map(|(name, raw)| {
            let base_url = raw.base_url.trim().to_string();
            if base_url.is_empty() {
                return None;
            }
            let api_key_env = match raw.api_key_env {
                Some(ApiKeyEnv::One(s)) => {
                    let s = s.trim().to_string();
                    if s.is_empty() { Vec::new() } else { vec![s] }
                }
                Some(ApiKeyEnv::Many(v)) => v
                    .into_iter()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                None => Vec::new(),
            };
            Some((
                name,
                ProviderConfig {
                    base_url,
                    api_key: raw.api_key,
                    api_key_env,
                    headers: raw.headers,
                    models: raw.models,
                    compat: raw.compat.into(),
                },
            ))
        })
        .collect()
}

/// Load named providers; empty map if missing or invalid.
/// Cached by data_dir + file mtime so multi-turn sessions do not re-parse disk.
pub fn load_providers(data_dir: &Path) -> BTreeMap<String, ProviderConfig> {
    let path = providers_path(data_dir);
    let mtime = file_mtime(&path);
    let lock = PROVIDERS_CACHE.get_or_init(|| {
        Mutex::new(ProvidersCache {
            data_dir: PathBuf::new(),
            mtime: None,
            map: BTreeMap::new(),
        })
    });
    let mut cache = lock.lock().unwrap_or_else(|e| e.into_inner());
    if cache.data_dir == data_dir && cache.mtime == mtime {
        return cache.map.clone();
    }
    let map = match std::fs::read_to_string(&path) {
        Ok(text) => parse_providers_file(&text),
        Err(_) => BTreeMap::new(),
    };
    cache.data_dir = data_dir.to_path_buf();
    cache.mtime = mtime;
    cache.map = map.clone();
    map
}

/// List provider names sorted for display.
pub fn list_provider_names(data_dir: &Path) -> Vec<String> {
    load_providers(data_dir).into_keys().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    /// Settings named a provider that is not in providers.json (or the file is bad).
    UnknownProvider(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnknownProvider(name) => write!(
                f,
                "unknown provider '{name}': add it to ~/.openmax/providers.json or clear settings.provider"
            ),
        }
    }
}

/// Resolve the active OpenAI-compatible endpoint from settings + providers.json.
///
/// When `settings.provider` is set, that name must exist. Silent fallback to
/// flat `base_url` would send traffic to the wrong endpoint.
pub fn resolve(settings: &Settings, data_dir: &Path) -> Result<ActiveEndpoint, ResolveError> {
    let providers = load_providers(data_dir);
    let provider_name = settings
        .provider
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(ref name) = provider_name {
        let Some(p) = providers.get(name) else {
            return Err(ResolveError::UnknownProvider(name.clone()));
        };
        let model_entry = p.models.iter().find(|m| m.id == settings.model);
        let context_tokens = model_entry
            .and_then(|m| m.context_tokens)
            .unwrap_or(settings.context_tokens)
            .max(1);
        let mut max_tokens = model_entry
            .and_then(|m| m.max_tokens)
            .unwrap_or(settings.max_tokens)
            .max(1);
        // Keep room for system + task history; never let max_tokens eat the window.
        let max_allowed = context_tokens.saturating_sub(2048).max(1);
        max_tokens = max_tokens.min(max_allowed);
        let api_key = resolve_api_key(
            p.api_key.as_deref(),
            &p.api_key_env,
            settings.api_key.as_deref(),
        );
        let headers = expand_headers(&p.headers);
        return Ok(ActiveEndpoint {
            provider: Some(name.clone()),
            base_url: p.base_url.clone(),
            api_key,
            headers,
            model: settings.model.clone(),
            context_tokens,
            max_tokens,
            temperature: settings.temperature,
            compat: p.compat.clone(),
        });
    }

    // Flat settings path when no provider is selected.
    let context_tokens = settings.context_tokens.max(1);
    let max_allowed = context_tokens.saturating_sub(2048).max(1);
    let max_tokens = settings.max_tokens.max(1).min(max_allowed);
    Ok(ActiveEndpoint {
        provider: None,
        base_url: settings.base_url.clone(),
        api_key: resolve_api_key(None, &[], settings.api_key.as_deref()),
        headers: Vec::new(),
        model: settings.model.clone(),
        context_tokens,
        max_tokens,
        temperature: settings.temperature,
        compat: CompatFlags::defaults_for_missing(),
    })
}

/// True when the resolved URL is the managed local MLX port (host is loopback).
pub fn is_managed_mlx(endpoint: &ActiveEndpoint, mlx_port: u16) -> bool {
    let port = if mlx_port == 0 { DEFAULT_MLX_PORT } else { mlx_port };
    let s = endpoint.base_url.trim();
    let rest = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))
        .unwrap_or(s);
    let authority = rest.split('/').next().unwrap_or("").split('@').next_back().unwrap_or("");
    authority.eq_ignore_ascii_case(&format!("127.0.0.1:{port}"))
        || authority.eq_ignore_ascii_case(&format!("localhost:{port}"))
        || authority.eq_ignore_ascii_case(&format!("[::1]:{port}"))
}

fn resolve_api_key(
    provider_key: Option<&str>,
    provider_env: &[String],
    settings_key: Option<&str>,
) -> Option<String> {
    if let Some(k) = provider_key {
        if let Some(v) = expand_secret(k) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    for name in provider_env {
        if let Ok(v) = std::env::var(name) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    if let Some(k) = settings_key {
        if let Some(v) = expand_secret(k) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    if let Ok(v) = std::env::var("OPENMAX_API_KEY") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return Some(v);
        }
    }
    None
}

/// Expand secrets:
/// - `$$...` → literal starting with `$` (escape)
/// - `$ENV_VAR` → environment value
/// - otherwise literal (trimmed). Empty → None.
fn expand_secret(raw: &str) -> Option<String> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(rest) = s.strip_prefix("$$") {
        return Some(format!("${rest}"));
    }
    if let Some(rest) = s.strip_prefix('$') {
        let name = rest.trim();
        if name.is_empty() {
            return None;
        }
        return std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    }
    Some(s.to_string())
}

fn expand_headers(map: &BTreeMap<String, String>) -> Vec<(String, String)> {
    map.iter()
        .filter_map(|(k, v)| {
            let key = k.trim();
            if key.is_empty() {
                return None;
            }
            // Skip headers whose secret env is unset rather than sending empty values.
            let val = expand_secret(v)?;
            if val.is_empty() {
                return None;
            }
            Some((key.to_string(), val))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Settings;

    fn write_providers(dir: &Path, json: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(providers_path(dir), json).unwrap();
    }

    #[test]
    fn missing_providers_file_uses_flat_settings() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        invalidate_providers_cache();
        let mut s = Settings::default();
        s.base_url = "http://127.0.0.1:11434/v1".into();
        s.model = "qwen".into();
        s.api_key = Some("k".into());
        let ep = resolve(&s, &dir).unwrap();
        assert_eq!(ep.base_url, "http://127.0.0.1:11434/v1");
        assert_eq!(ep.model, "qwen");
        assert_eq!(ep.api_key.as_deref(), Some("k"));
        assert!(ep.provider.is_none());
        assert!(ep.compat.send_stream_options);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn load_providers_cache_hits_second_call() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        write_providers(
            &dir,
            r#"{"providers":{"x":{"base_url":"http://x/v1","models":[{"id":"m"}]}}}"#,
        );
        invalidate_providers_cache();
        let a = load_providers(&dir);
        let b = load_providers(&dir);
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert!(a.contains_key("x") && b.contains_key("x"));
        // After invalidate, still correct.
        invalidate_providers_cache();
        let c = load_providers(&dir);
        assert!(c.contains_key("x"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn named_provider_overrides_base_url_and_headers() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        write_providers(
            &dir,
            r#"{
              "providers": {
                "or": {
                  "base_url": "https://openrouter.ai/api/v1",
                  "api_key": "sk-test",
                  "headers": { "X-Title": "Open Max" },
                  "models": [{ "id": "m1", "context_tokens": 64000, "max_tokens": 2048 }]
                }
              }
            }"#,
        );
        let mut s = Settings::default();
        s.provider = Some("or".into());
        s.model = "m1".into();
        s.base_url = "http://ignored".into();
        let ep = resolve(&s, &dir).unwrap();
        assert_eq!(ep.provider.as_deref(), Some("or"));
        assert_eq!(ep.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(ep.api_key.as_deref(), Some("sk-test"));
        assert_eq!(ep.context_tokens, 64000);
        assert_eq!(ep.max_tokens, 2048);
        assert!(ep.headers.iter().any(|(k, v)| k == "X-Title" && v == "Open Max"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn api_key_env_and_dollar_expansion() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        let var = format!("OPENMAX_TEST_KEY_{}", uuid::Uuid::new_v4().simple());
        std::env::set_var(&var, "from-env");
        write_providers(
            &dir,
            &format!(
                r#"{{
              "providers": {{
                "a": {{
                  "base_url": "http://a/v1",
                  "api_key_env": "{var}"
                }},
                "b": {{
                  "base_url": "http://b/v1",
                  "api_key": "${var}"
                }}
              }}
            }}"#
            ),
        );
        let mut s = Settings::default();
        s.provider = Some("a".into());
        let ep = resolve(&s, &dir).unwrap();
        assert_eq!(ep.api_key.as_deref(), Some("from-env"));
        s.provider = Some("b".into());
        let ep = resolve(&s, &dir).unwrap();
        assert_eq!(ep.api_key.as_deref(), Some("from-env"));
        std::env::remove_var(&var);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unknown_provider_errors() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        write_providers(&dir, r#"{"providers":{}}"#);
        let mut s = Settings::default();
        s.provider = Some("missing".into());
        s.base_url = "http://flat/v1".into();
        let err = resolve(&s, &dir).unwrap_err();
        assert!(matches!(err, ResolveError::UnknownProvider(_)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn dollar_escape_and_skip_empty_header_env() {
        assert_eq!(expand_secret("$$secret").as_deref(), Some("$secret"));
        let mut map = BTreeMap::new();
        map.insert("X-A".into(), "$NO_SUCH_OPENMAX_ENV_VAR_ZZZ".into());
        map.insert("X-B".into(), "ok".into());
        let headers = expand_headers(&map);
        assert_eq!(headers, vec![("X-B".into(), "ok".into())]);
    }

    #[test]
    fn managed_mlx_detection() {
        let ep = ActiveEndpoint {
            provider: None,
            base_url: format!("http://127.0.0.1:{DEFAULT_MLX_PORT}/v1"),
            api_key: None,
            headers: vec![],
            model: "m".into(),
            context_tokens: 1,
            max_tokens: 1,
            temperature: 0.0,
            compat: CompatFlags::defaults_for_missing(),
        };
        assert!(is_managed_mlx(&ep, DEFAULT_MLX_PORT));
        let mut remote = ep.clone();
        remote.base_url = "https://api.example.com/v1".into();
        assert!(!is_managed_mlx(&remote, DEFAULT_MLX_PORT));
        // Path must not trigger false positive.
        remote.base_url = "https://api.example.com/v1/127.0.0.1:8989".into();
        assert!(!is_managed_mlx(&remote, DEFAULT_MLX_PORT));
    }

    #[test]
    fn clamps_max_tokens_below_context() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        write_providers(
            &dir,
            r#"{
              "providers": {
                "tiny": {
                  "base_url": "http://t/v1",
                  "models": [{ "id": "m", "context_tokens": 2048, "max_tokens": 100000 }]
                }
              }
            }"#,
        );
        let mut s = Settings::default();
        s.provider = Some("tiny".into());
        s.model = "m".into();
        let ep = resolve(&s, &dir).unwrap();
        assert!(ep.max_tokens + 1024 < ep.context_tokens || ep.context_tokens <= 2048);
        assert!(ep.max_tokens <= ep.context_tokens.saturating_sub(2048).max(1));
        let _ = std::fs::remove_dir_all(dir);
    }
}
