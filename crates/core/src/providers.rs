//! Named OpenAI-compatible providers (`~/.openmax/providers.json`).
//! Missing file is free: the flat `base_url` / `api_key` / `model` settings path
//! continues to work unchanged.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::Settings;

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
struct ProvidersCache {
    data_dir: PathBuf,
    /// None when the file was missing or unreadable at last load.
    content_hash: Option<u64>,
    map: BTreeMap<String, ProviderConfig>,
    /// The parse error when the file existed but was not valid JSON. Kept so
    /// resolve() and receipts can say "the file is broken" instead of the
    /// misleading "unknown provider" an empty map produces.
    parse_error: Option<String>,
}

/// What the last providers.json read actually found, for receipts and
/// diagnostics: how many providers loaded, whether the file failed to
/// parse (and why), and the content identity of the bytes read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvidersStatus {
    pub count: usize,
    pub names: Vec<String>,
    pub parse_error: Option<String>,
    pub content_hash: Option<u64>,
}

static PROVIDERS_CACHE: OnceLock<Mutex<ProvidersCache>> = OnceLock::new();

/// Drop cached providers so the next load re-reads disk (settings/provider edits).
pub fn invalidate_providers_cache() {
    if let Some(lock) = PROVIDERS_CACHE.get() {
        if let Ok(mut cache) = lock.lock() {
            cache.data_dir.clear();
            cache.content_hash = None;
            cache.map.clear();
            cache.parse_error = None;
        }
    }
}

fn content_hash(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

fn parse_providers_file(text: &str) -> Result<BTreeMap<String, ProviderConfig>, String> {
    let file = match serde_json::from_str::<ProvidersFile>(text) {
        Ok(file) => file,
        // The runtime stays lax about unknown keys (--check names those),
        // but a file that is not JSON at all must keep its reason: an empty
        // map here surfaces later as "unknown provider", which sends the
        // repair in the wrong direction.
        Err(e) => return Err(e.to_string()),
    };
    // A provider with an empty base_url is kept, not dropped: it still shows
    // up in listings and fails loudly at resolve time (MissingEndpoint)
    // instead of silently vanishing from the picker.
    let map = file
        .providers
        .into_iter()
        .map(|(name, raw)| {
            let base_url = raw.base_url.trim().to_string();
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
            (
                name,
                ProviderConfig {
                    base_url,
                    api_key: raw.api_key,
                    api_key_env,
                    headers: raw.headers,
                    models: raw.models,
                    compat: raw.compat.into(),
                },
            )
        })
        .collect();
    Ok(map)
}

/// Diagnose `providers.json` for `openmax --check`. Runtime loading remains
/// deliberately quiet, while the validator reports syntax and values that
/// would otherwise produce an empty catalog or fail at request time. `Ok`
/// carries the provider count plus one warning per unknown key: serde's lax
/// deserialization keeps runtime loading tolerant, so a typo'd key
/// ("modles") configures nothing, and this is the one place that says so.
pub(crate) fn check_file(path: &Path) -> Option<Result<(usize, Vec<String>), String>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => return Some(Err(format!("cannot read: {e}"))),
    };
    let file = match serde_json::from_str::<ProvidersFile>(&text) {
        Ok(file) => file,
        Err(e) => return Some(Err(format!("invalid JSON: {e}"))),
    };
    for (name, provider) in &file.providers {
        if name.trim().is_empty() {
            return Some(Err("provider name cannot be empty".into()));
        }
        if name != name.trim() {
            return Some(Err(format!(
                "provider name '{name}' cannot have surrounding whitespace"
            )));
        }
        let base_url = provider.base_url.trim();
        if base_url.is_empty() {
            return Some(Err(format!("provider '{name}' has an empty base_url")));
        }
        match reqwest::Url::parse(base_url) {
            Ok(url) if matches!(url.scheme(), "http" | "https") => {}
            Ok(_) => {
                return Some(Err(format!(
                    "provider '{name}' base_url must use http or https"
                )))
            }
            Err(e) => {
                return Some(Err(format!(
                    "provider '{name}' has an invalid base_url: {e}"
                )))
            }
        }
        for (header, value) in &provider.headers {
            if reqwest::header::HeaderName::from_bytes(header.as_bytes()).is_err() {
                return Some(Err(format!(
                    "provider '{name}' has an invalid header name '{header}'"
                )));
            }
            if reqwest::header::HeaderValue::from_str(value).is_err() {
                return Some(Err(format!(
                    "provider '{name}' header '{header}' has an invalid value"
                )));
            }
        }
        for model in &provider.models {
            if model.id.trim().is_empty() {
                return Some(Err(format!(
                    "provider '{name}' contains a model with an empty id"
                )));
            }
            if model.id != model.id.trim() {
                return Some(Err(format!(
                    "provider '{name}' model id '{}' cannot have surrounding whitespace",
                    model.id
                )));
            }
            if model.context_tokens == Some(0) {
                return Some(Err(format!(
                    "provider '{name}' model '{}' has zero context_tokens",
                    model.id
                )));
            }
            if model.max_tokens == Some(0) {
                return Some(Err(format!(
                    "provider '{name}' model '{}' has zero max_tokens",
                    model.id
                )));
            }
        }
    }
    Some(Ok((file.providers.len(), unknown_key_warnings(&text))))
}

/// One warning per key the typed parse silently drops, at every level the
/// file has: top level, provider, model, and compat. Walked from the raw
/// JSON because serde discards the unknown keys it skips.
fn unknown_key_warnings(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return out;
    };
    let Some(top) = value.as_object() else { return out };
    unknown_keys(top, &["providers"], "at the top level", &mut out);
    let Some(providers) = top.get("providers").and_then(|v| v.as_object()) else {
        return out;
    };
    for (name, provider) in providers {
        let Some(provider) = provider.as_object() else { continue };
        unknown_keys(
            provider,
            &["base_url", "api_key", "api_key_env", "headers", "models", "compat"],
            &format!("in provider '{name}'"),
            &mut out,
        );
        if let Some(compat) = provider.get("compat").and_then(|v| v.as_object()) {
            unknown_keys(
                compat,
                &["use_max_completion_tokens", "send_stream_options"],
                &format!("in provider '{name}' compat"),
                &mut out,
            );
        }
        for model in provider.get("models").and_then(|v| v.as_array()).into_iter().flatten() {
            let Some(model) = model.as_object() else { continue };
            let id = model.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            unknown_keys(
                model,
                &["id", "name", "context_tokens", "max_tokens"],
                &format!("in provider '{name}' model '{id}'"),
                &mut out,
            );
        }
    }
    out
}

fn unknown_keys(
    obj: &serde_json::Map<String, serde_json::Value>,
    known: &[&str],
    place: &str,
    out: &mut Vec<String>,
) {
    for key in obj.keys() {
        if known.contains(&key.as_str()) {
            continue;
        }
        let hint = known
            .iter()
            .find(|k| crate::doctor::near(k, key))
            .map(|k| format!(", did you mean '{k}'"))
            .unwrap_or_default();
        out.push(format!("unknown key '{key}' {place} configures nothing{hint}"));
    }
}

/// Load named providers; empty map if missing or invalid.
/// Cached by data_dir + content hash so multi-turn sessions do not re-parse
/// disk. Keying on content (not mtime) means edits within one filesystem
/// timestamp tick still invalidate: endpoints and credentials must never be
/// served stale. The read happens under the cache mutex so a slow reader
/// holding older bytes can never publish over a newer snapshot.
pub fn load_providers(data_dir: &Path) -> BTreeMap<String, ProviderConfig> {
    load_snapshot(data_dir).0
}

/// One consistent read: the catalog AND its parse status from the same
/// cache generation, under one lock hold. resolve() uses this so the error
/// it reports describes the very bytes it looked the name up in - two
/// separate reads could straddle a file replacement and name the wrong
/// version.
fn load_snapshot(data_dir: &Path) -> (BTreeMap<String, ProviderConfig>, Option<String>) {
    let path = providers_path(data_dir);
    let lock = PROVIDERS_CACHE.get_or_init(|| {
        Mutex::new(ProvidersCache {
            data_dir: PathBuf::new(),
            content_hash: None,
            map: BTreeMap::new(),
            parse_error: None,
        })
    });
    let mut cache = lock.lock().unwrap_or_else(|e| e.into_inner());
    refresh_cache(&mut cache, data_dir, &path);
    (cache.map.clone(), cache.parse_error.clone())
}

/// Re-read disk into the cache when the content moved. Runs under the cache
/// mutex; both `load_providers` and `providers_status` go through here so
/// the catalog and its status can never disagree.
fn refresh_cache(cache: &mut ProvidersCache, data_dir: &Path, path: &Path) {
    let text = std::fs::read_to_string(path).ok();
    let hash = text.as_deref().map(content_hash);
    if cache.data_dir == data_dir && cache.content_hash == hash {
        return;
    }
    let (map, parse_error) = match text {
        None => (BTreeMap::new(), None),
        Some(t) => match parse_providers_file(&t) {
            Ok(map) => (map, None),
            Err(e) => (BTreeMap::new(), Some(e)),
        },
    };
    cache.data_dir = data_dir.to_path_buf();
    cache.content_hash = hash;
    cache.map = map;
    cache.parse_error = parse_error;
}

/// The current read's outcome, through the same cache (and mutex) as
/// [`load_providers`].
pub fn providers_status(data_dir: &Path) -> ProvidersStatus {
    let path = providers_path(data_dir);
    let lock = PROVIDERS_CACHE.get_or_init(|| {
        Mutex::new(ProvidersCache {
            data_dir: PathBuf::new(),
            content_hash: None,
            map: BTreeMap::new(),
            parse_error: None,
        })
    });
    let mut cache = lock.lock().unwrap_or_else(|e| e.into_inner());
    refresh_cache(&mut cache, data_dir, &path);
    ProvidersStatus {
        count: cache.map.len(),
        names: cache.map.keys().cloned().collect(),
        parse_error: cache.parse_error.clone(),
        content_hash: cache.content_hash,
    }
}

/// List provider names sorted for display.
pub fn list_provider_names(data_dir: &Path) -> Vec<String> {
    load_providers(data_dir).into_keys().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolveError {
    /// No endpoint configured anywhere; there is no default to fall back to.
    MissingEndpoint,
    /// No model id configured for the resolved endpoint.
    MissingModel,
    /// Settings named a provider that is not in providers.json.
    UnknownProvider(String),
    /// providers.json exists but is not valid JSON: every provider is
    /// unavailable until it parses. Distinct from UnknownProvider because
    /// "add it to providers.json" sends the repair in the wrong direction
    /// when the file itself is the problem.
    InvalidProvidersFile(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::MissingEndpoint => write!(
                f,
                "no model endpoint configured: set base_url in ~/.openmax/settings.json, or define one in ~/.openmax/providers.json and select it with settings.provider"
            ),
            ResolveError::MissingModel => write!(
                f,
                "no model configured: set model in ~/.openmax/settings.json"
            ),
            ResolveError::UnknownProvider(name) => write!(
                f,
                "unknown provider '{name}': add it to ~/.openmax/providers.json or clear settings.provider"
            ),
            ResolveError::InvalidProvidersFile(err) => write!(
                f,
                "~/.openmax/providers.json is invalid JSON: {err} — every provider is unavailable until it parses; fix the file (openmax --check names problems) or clear settings.provider"
            ),
        }
    }
}

/// Resolve the active OpenAI-compatible endpoint from settings + providers.json.
///
/// When `settings.provider` is set, that name must exist. Silent fallback to
/// flat `base_url` would send traffic to the wrong endpoint.
pub fn resolve(settings: &Settings, data_dir: &Path) -> Result<ActiveEndpoint, ResolveError> {
    let (providers, parse_error) = load_snapshot(data_dir);
    let provider_name = settings
        .provider
        .as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(ref name) = provider_name {
        let Some(p) = providers.get(name) else {
            // An empty catalog because the file failed to parse is the
            // file's fault, not the name's: say so - from the same snapshot
            // the lookup used.
            if let Some(err) = parse_error {
                return Err(ResolveError::InvalidProvidersFile(err));
            }
            return Err(ResolveError::UnknownProvider(name.clone()));
        };
        let base_url = p.base_url.trim();
        if base_url.is_empty() {
            return Err(ResolveError::MissingEndpoint);
        }
        let model = settings.model.trim();
        if model.is_empty() {
            return Err(ResolveError::MissingModel);
        }
        let model_entry = p.models.iter().find(|m| m.id == model);
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
            base_url: base_url.to_string(),
            api_key,
            headers,
            model: model.to_string(),
            context_tokens,
            max_tokens,
            temperature: settings.temperature,
            compat: p.compat.clone(),
        });
    }

    // Flat settings path when no provider is selected. No endpoint or model
    // means no request: there is no default server to guess at.
    let base_url = settings.base_url.trim();
    if base_url.is_empty() {
        return Err(ResolveError::MissingEndpoint);
    }
    let model = settings.model.trim();
    if model.is_empty() {
        return Err(ResolveError::MissingModel);
    }
    let context_tokens = settings.context_tokens.max(1);
    let max_allowed = context_tokens.saturating_sub(2048).max(1);
    let max_tokens = settings.max_tokens.max(1).min(max_allowed);
    Ok(ActiveEndpoint {
        provider: None,
        base_url: base_url.to_string(),
        api_key: resolve_api_key(None, &[], settings.api_key.as_deref()),
        headers: Vec::new(),
        model: model.to_string(),
        context_tokens,
        max_tokens,
        temperature: settings.temperature,
        compat: CompatFlags::defaults_for_missing(),
    })
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
        let s = Settings {
            base_url: "http://127.0.0.1:11434/v1".into(),
            model: "qwen".into(),
            api_key: Some("k".into()),
            ..Default::default()
        };
        let ep = resolve(&s, &dir).unwrap();
        assert_eq!(ep.base_url, "http://127.0.0.1:11434/v1");
        assert_eq!(ep.model, "qwen");
        assert_eq!(ep.api_key.as_deref(), Some("k"));
        assert!(ep.provider.is_none());
        assert!(ep.compat.send_stream_options);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn check_file_reports_valid_and_invalid_provider_documents() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("providers.json");
        std::fs::write(
            &path,
            r#"{"providers":{"local":{"base_url":"http://127.0.0.1:11434/v1","models":[{"id":"coder"}]}}}"#,
        )
        .unwrap();
        assert_eq!(check_file(&path).unwrap().unwrap(), (1, Vec::new()));

        std::fs::write(&path, r#"{"providers":{"local":{"base_url":"not a url"}}}"#)
            .unwrap();
        assert!(check_file(&path)
            .unwrap()
            .unwrap_err()
            .contains("invalid base_url"));

        std::fs::write(
            &path,
            r#"{"providers":{" local ":{"base_url":"http://localhost/v1"}}}"#,
        )
        .unwrap();
        assert!(check_file(&path)
            .unwrap()
            .unwrap_err()
            .contains("provider name ' local '"));

        std::fs::write(
            &path,
            r#"{"providers":{"local":{"base_url":"http://localhost/v1","models":[{"id":" coder "}]}}}"#,
        )
        .unwrap();
        assert!(check_file(&path)
            .unwrap()
            .unwrap_err()
            .contains("model id ' coder '"));

        std::fs::write(&path, "{not json").unwrap();
        assert!(check_file(&path)
            .unwrap()
            .unwrap_err()
            .contains("invalid JSON"));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Serde skips keys it does not know, so "modles" deserializes cleanly
    /// and the models list it was meant to be stays empty, with nothing
    /// anywhere saying why. The validator must name each ignored key, at
    /// every level of the file, with the near-miss it was probably meant
    /// to be - the same suggestion shape rules and hook filters get.
    #[test]
    fn check_file_names_unknown_keys_at_every_level() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("providers.json");

        std::fs::write(
            &path,
            r#"{"provider":{"local":{"base_url":"http://localhost/v1"}}}"#,
        )
        .unwrap();
        let (count, warnings) = check_file(&path).unwrap().unwrap();
        assert_eq!(count, 0, "the typo'd top-level key configures no providers");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("'provider'"), "{}", warnings[0]);
        assert!(warnings[0].contains("did you mean 'providers'"), "{}", warnings[0]);

        std::fs::write(
            &path,
            concat!(
                r#"{"providers":{"orl":{"base_url":"http://localhost/v1","modles":[],"#,
                r#""compat":{"use_max_completion_token":true},"#,
                r#""models":[{"id":"m","contex_tokens":9000}]}}}"#,
            ),
        )
        .unwrap();
        let (count, warnings) = check_file(&path).unwrap().unwrap();
        assert_eq!(count, 1);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert!(
            warnings.iter().any(|w| w.contains("'modles' in provider 'orl'")
                && w.contains("did you mean 'models'")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("'use_max_completion_token'")
                && w.contains("compat")
                && w.contains("did you mean 'use_max_completion_tokens'")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("'contex_tokens'")
                && w.contains("model 'm'")
                && w.contains("did you mean 'context_tokens'")),
            "{warnings:?}"
        );
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
    fn load_providers_detects_same_length_same_mtime_edit() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        write_providers(
            &dir,
            r#"{"providers":{"aa":{"base_url":"http://a/v1","models":[{"id":"m"}]}}}"#,
        );
        invalidate_providers_cache();
        let path = providers_path(&dir);
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let a = load_providers(&dir);
        assert!(a.contains_key("aa"));

        // Same byte length, pinned mtime: only the bytes differ. An
        // mtime-keyed cache would keep serving the old endpoint map.
        write_providers(
            &dir,
            r#"{"providers":{"bb":{"base_url":"http://b/v1","models":[{"id":"m"}]}}}"#,
        );
        let f = std::fs::File::options().write(true).open(&path).unwrap();
        f.set_modified(mtime).unwrap();
        drop(f);

        let b = load_providers(&dir);
        assert!(b.contains_key("bb"), "same-tick edit must invalidate the cache");
        assert!(!b.contains_key("aa"));
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
        let s = Settings {
            provider: Some("or".into()),
            model: "m1".into(),
            base_url: "http://ignored".into(),
            ..Default::default()
        };
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
        let mut s = Settings {
            provider: Some("a".into()),
            model: "m".into(),
            ..Default::default()
        };
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
        let s = Settings {
            provider: Some("missing".into()),
            base_url: "http://flat/v1".into(),
            ..Default::default()
        };
        let err = resolve(&s, &dir).unwrap_err();
        assert!(matches!(err, ResolveError::UnknownProvider(_)));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A providers.json that is not valid JSON parsed to an EMPTY catalog
    /// silently, so a named provider surfaced as "unknown provider" - a
    /// repair pointed at the wrong problem. The parse error must survive to
    /// both the status surface and the resolve error.
    #[test]
    fn a_malformed_providers_file_fails_loudly_not_as_unknown_provider() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        write_providers(&dir, r#"{"providers": {"xai": {"base_url": }"#);
        let status = providers_status(&dir);
        assert_eq!(status.count, 0);
        let err = status.parse_error.expect("the parse error is kept, not swallowed");
        assert!(!err.is_empty());

        let s = Settings {
            provider: Some("xai".into()),
            model: "m".into(),
            ..Default::default()
        };
        let resolved = resolve(&s, &dir).unwrap_err();
        assert!(
            matches!(resolved, ResolveError::InvalidProvidersFile(_)),
            "a broken file is the file's fault, not the name's: {resolved:?}"
        );
        assert!(resolved.to_string().contains("invalid JSON"), "{resolved}");

        // A genuinely unknown name in a VALID file keeps the old error.
        write_providers(&dir, r#"{"providers":{}}"#);
        let resolved = resolve(&s, &dir).unwrap_err();
        assert!(matches!(resolved, ResolveError::UnknownProvider(_)));
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
    fn empty_flat_settings_fail_closed() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = resolve(&Settings::default(), &dir).unwrap_err();
        assert!(matches!(err, ResolveError::MissingEndpoint));
        let s = Settings {
            base_url: "http://127.0.0.1:11434/v1".into(),
            ..Default::default()
        };
        let err = resolve(&s, &dir).unwrap_err();
        assert!(matches!(err, ResolveError::MissingModel));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn named_provider_without_endpoint_fails_closed() {
        let dir = std::env::temp_dir().join(format!("openmax-prov-{}", uuid::Uuid::new_v4()));
        write_providers(
            &dir,
            r#"{"providers":{"empty":{"base_url":"","models":[{"id":"m"}]}}}"#,
        );
        let settings = Settings {
            provider: Some("empty".into()),
            model: "m".into(),
            ..Default::default()
        };
        let err = resolve(&settings, &dir).unwrap_err();
        assert!(matches!(err, ResolveError::MissingEndpoint));
        let _ = std::fs::remove_dir_all(dir);
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
        let s = Settings {
            provider: Some("tiny".into()),
            model: "m".into(),
            ..Default::default()
        };
        let ep = resolve(&s, &dir).unwrap();
        assert!(ep.max_tokens + 1024 < ep.context_tokens || ep.context_tokens <= 2048);
        assert!(ep.max_tokens <= ep.context_tokens.saturating_sub(2048).max(1));
        let _ = std::fs::remove_dir_all(dir);
    }
}
