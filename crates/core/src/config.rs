//! `settings.json`: the resolved configuration for a data dir.
//!
//! Parsing is strict and fail-closed. An unknown key, an unparseable value, or
//! an unrecognized `approval_mode` is an error that stops the run, never a
//! silent fallback to a default. The reason is asymmetric risk: a typo in
//! `approval_mode` that quietly became `auto` would hand the agent authority
//! the user believed they had withheld, and they would not find out until
//! something ran.
//!
//! For the same reason there is no default endpoint. An empty `base_url` is a
//! hard resolve error rather than a guess at localhost, so a misconfigured
//! install fails at startup instead of silently talking to whatever happens to
//! be listening. There is no default context window either: the harness
//! never queries one, and a guess compacts a large model early or lets a
//! small one fail for length, so a missing `context_tokens` is a resolve
//! error that names where to set it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Execution policy for tools and extensions. Parsed strictly: an unrecognized value is a
/// configuration error, never a silent fallback, because a typo here would
/// otherwise weaken the approval gate the user asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    /// Skip confirmation and extension content grants; retain deny rules and validation.
    Auto,
    /// Prompt for mutating tools and unapproved external content.
    Ask,
    /// Block mutating tools and calls requiring approval.
    Readonly,
}

impl ApprovalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ApprovalMode::Auto => "auto",
            ApprovalMode::Ask => "ask",
            ApprovalMode::Readonly => "readonly",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "auto" => Some(ApprovalMode::Auto),
            "ask" => Some(ApprovalMode::Ask),
            "readonly" => Some(ApprovalMode::Readonly),
            _ => None,
        }
    }

    /// The mode a front end's step key (Shift+Tab in the TUI) moves to. It
    /// switches between `ask` and `auto`, the two modes a session changes
    /// its mind about, and it leaves `readonly` but never enters it: the key
    /// is easy to press by accident and the choice is saved per project, so
    /// one stray press that saved `readonly` would leave the project's next
    /// session refusing every write in a mode nobody chose on purpose.
    /// `readonly` is chosen on purpose: at the trust prompt, with
    /// `/approvals readonly`, or a frontend's explicit mode command. One
    /// source of truth so a front end cannot invent its own order.
    pub fn next(self) -> Self {
        match self {
            ApprovalMode::Readonly | ApprovalMode::Auto => ApprovalMode::Ask,
            ApprovalMode::Ask => ApprovalMode::Auto,
        }
    }

    /// Whichever of the two lets the agent do less: readonly, then ask, then
    /// auto. Spelled out rather than derived from the variant order, so
    /// reordering the variants cannot quietly loosen a running gate.
    pub fn stricter(self, other: Self) -> Self {
        match (self, other) {
            (ApprovalMode::Readonly, _) | (_, ApprovalMode::Readonly) => ApprovalMode::Readonly,
            (ApprovalMode::Ask, _) | (_, ApprovalMode::Ask) => ApprovalMode::Ask,
            _ => ApprovalMode::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// Named provider from `providers.json`. When set and found, supplies
    /// base_url, credentials, and headers; flat fields remain the fallback.
    #[serde(default)]
    pub provider: Option<String>,
    /// OpenAI-compatible base URL. There is no default endpoint: an empty
    /// value is a hard resolve error, never a silent localhost fallback.
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub approval_mode: ApprovalMode,
    /// The model's context window in tokens. No default: the harness never
    /// queries the server, and a guessed window is wrong in one direction or
    /// the other (too small compacts a large model's history early, too large
    /// lets a small model's requests fail for length). Required at resolve
    /// unless the named provider's model entry supplies it.
    pub context_tokens: Option<usize>,
    pub max_tokens: usize,
    /// Sent only when set; unset leaves sampling to the server's default.
    /// Some models accept nothing else: OpenAI's reasoning models reject any
    /// temperature but 1. Skipped when unset so a settings save never pins a
    /// value the user did not choose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Byte cap for bash/external tool output before truncation with
    /// spill-to-file. Unset means the tuned built-in default.
    pub max_output_bytes: Option<usize>,
    /// Compact when the estimated request (transcript plus frozen tool
    /// schemas) crosses this many tokens, instead of waiting for the
    /// window-derived budget. One-directional: it can only fire compaction
    /// earlier. Values above the budget, below the built-in floor, or
    /// unreachable under the frozen schemas fall back to the budget.
    pub compaction_tokens: Option<usize>,
    /// Ceiling on the tokens one turn may spend across the agent requests it
    /// makes, counted as a provider bills them: prompt plus completion, with
    /// cached prompt tokens at full value, because what this bounds is the
    /// work a turn may ask for and not what it happened to cost. Enforced at
    /// admission: a request goes out only when what the turn has spent plus
    /// the request's own estimated size still fits, so the ceiling refuses
    /// the request it cannot afford instead of the one after it. The reply
    /// side is not reserved. Output tokens and input-estimation error can
    /// exceed the cap; a request whose estimate cannot fit is withheld.
    /// Compaction summaries share this ceiling. When a summary cannot fit,
    /// compaction keeps its deterministic digest and makes no summary request.
    /// None means no ceiling.
    pub max_agent_tokens: Option<usize>,
    /// Cap on agent tool/model iterations per turn (main loop).
    #[serde(default = "default_max_agent_iterations")]
    pub max_agent_iterations: usize,
    /// Maximum number of read-only tool calls admitted concurrently.
    /// Runtime code clamps configured values to 1..=32.
    #[serde(default = "default_max_parallel_tools")]
    pub max_parallel_tools: usize,
}

fn default_max_agent_iterations() -> usize {
    50
}

fn default_max_parallel_tools() -> usize {
    4
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            provider: None,
            base_url: String::new(),
            api_key: None,
            model: String::new(),
            approval_mode: ApprovalMode::Ask,
            context_tokens: None,
            max_tokens: 4096,
            temperature: None,
            max_output_bytes: None,
            compaction_tokens: None,
            max_agent_tokens: None,
            max_agent_iterations: default_max_agent_iterations(),
            max_parallel_tools: default_max_parallel_tools(),
        }
    }
}

pub fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("settings.json")
}

/// Settings keys given on the command line. `--base-url`, `--api-key`,
/// `--model` and `--context-tokens` each name the key of the same name, and
/// `--provider` picks a providers.json endpoint. One rule decides what each
/// of the four does, so the first run configures an install with no file
/// written by hand, and no later run rewrites it by accident:
///
/// - a key settings.json leaves unset (no file, an empty `base_url` or
///   `model`, no `api_key` or `context_tokens`) is written to the file;
/// - a key the file sets to another value is overridden for this run only;
/// - a flag equal to the file's value changes nothing.
///
/// A key is written only when the run uses the endpoint settings.json
/// selects. `--base-url` names the flat endpoint, so beside a
/// `settings.provider` it sets that provider aside for the run, a
/// `--base-url` other than the one the file sets is another server, and a
/// `--provider` other than the file's picks another endpoint; then every
/// flag is for this run only, because a key written for one endpoint would
/// configure the other (a key for one server sent to another). An override stays the run's through every save the
/// session makes ([`RunOnly`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyFlags {
    pub provider: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub context_tokens: Option<usize>,
}

/// What [`KeyFlags::apply`] did, by settings key, in field order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// Keys the file left unset, now set in it: the caller saves it.
    pub written: Vec<&'static str>,
    /// Keys set for this run only, to a value the file does not hold.
    pub overridden: Vec<&'static str>,
    /// Whether this run sends no settings key: `--base-url` names another
    /// server than the file's `base_url`, whose credential the file's
    /// `api_key` is, and no `--api-key` gives the run its own. `api_key` is
    /// then among `overridden`, so no save drops the file's key.
    pub withheld_key: bool,
}

/// The provider `provider` names, or None for the flat endpoint.
fn selected(provider: &Option<String>) -> Option<&str> {
    provider.as_deref().map(str::trim).filter(|p| !p.is_empty())
}

impl KeyFlags {
    /// Why a value cannot be used, naming its flag, or None. Checked before
    /// anything runs, so a placeholder pasted unchanged from the installer
    /// or the first-run error (`--base-url URL --model MODEL --api-key
    /// '$MY_API_KEY' --context-tokens TOKENS`) is refused instead of written,
    /// and so is a key read from a variable this environment does not set,
    /// which would configure no key.
    pub fn problem(&self) -> Option<String> {
        if let Some(url) = &self.base_url {
            let http = reqwest::Url::parse(url.trim()).is_ok_and(|u| matches!(u.scheme(), "http" | "https"));
            if !http {
                return Some(format!(
                    "--base-url takes the http or https root of the model server's API, such as http://127.0.0.1:11434/v1 (got '{url}')"
                ));
            }
        }
        for (flag, value) in [("--provider", &self.provider), ("--api-key", &self.api_key), ("--model", &self.model)] {
            if value.as_deref().is_some_and(|v| v.trim().is_empty()) {
                return Some(format!("{flag} takes a non-empty value"));
            }
        }
        if self.model.as_deref() == Some("MODEL") {
            return Some("--model takes the model id the server expects, not the placeholder MODEL".into());
        }
        // Past the empty check above, a key expands to nothing only as a
        // `$NAME` whose variable is unset (a literal and a `$$` escape always
        // expand).
        if let Some(key) = self.api_key.as_deref().filter(|k| crate::providers::expand_secret(k).is_none()) {
            return Some(format!(
                "--api-key '{key}' reads an environment variable that is not set here: export it first, or leave --api-key out for a server that checks no key"
            ));
        }
        if self.context_tokens == Some(0) {
            return Some("--context-tokens takes the model's context window, a positive number of tokens".into());
        }
        None
    }

    /// Why these flags cannot configure the endpoint this run uses, given
    /// settings.json as read at launch, or None. `--api-key` sets the key
    /// for `base_url`, which a provider's requests never carry to another
    /// server, so beside the provider settings.json selects it is refused,
    /// as it is beside `--provider`, rather than written and never sent.
    pub fn conflict(&self, file: &Settings) -> Option<String> {
        if self.api_key.is_none() || self.base_url.is_some() {
            return None;
        }
        let provider = selected(&self.provider).or(selected(&file.provider))?;
        Some(format!(
            "--api-key is the key for settings.json's base_url, but this run uses provider '{provider}', whose key is in ~/.openmax/providers.json; pass --base-url as well to use the key with that server instead"
        ))
    }

    /// Apply the flags to `file`, settings.json as read at launch: fill in
    /// it each key it leaves unset, for the caller to save, and return the
    /// settings this run uses, which carry every flag.
    pub fn apply(&self, file: &mut Settings) -> (Settings, Applied) {
        let mut run = file.clone();
        let mut applied = Applied::default();
        match (&self.provider, &self.base_url) {
            (Some(name), _) => run.provider = Some(name.clone()),
            (None, Some(_)) => run.provider = None,
            (None, None) => {}
        }
        let same_provider = selected(&run.provider) == selected(&file.provider);
        if !same_provider {
            applied.overridden.push("provider");
        }
        let same_server = self.base_url.as_ref().is_none_or(|url| file.base_url.trim().is_empty() || file.base_url == *url);
        let write = same_provider && same_server;
        let blank = |v: &String| v.trim().is_empty();
        place(&mut applied, "base_url", self.base_url.clone(), write, blank, &mut file.base_url, &mut run.base_url);
        // The file's api_key is the credential for the server its base_url
        // names (scheme, host and port), so a --base-url for another server
        // does not carry it there. One just written is the file's server.
        let withheld = self.api_key.is_none()
            && file.api_key.as_deref().is_some_and(|k| !k.trim().is_empty())
            && self.base_url.as_ref().is_some_and(|url| !crate::providers::same_server(url, &file.base_url));
        applied.withheld_key = withheld;
        place(
            &mut applied,
            "api_key",
            self.api_key.clone().map(Some).or(withheld.then_some(None)),
            write,
            |v| v.as_deref().is_none_or(|k| k.trim().is_empty()),
            &mut file.api_key,
            &mut run.api_key,
        );
        place(&mut applied, "model", self.model.clone(), write, blank, &mut file.model, &mut run.model);
        place(
            &mut applied,
            "context_tokens",
            self.context_tokens.map(Some),
            write,
            Option::is_none,
            &mut file.context_tokens,
            &mut run.context_tokens,
        );
        (run, applied)
    }
}

/// One key of [`KeyFlags::apply`]: written to `file` when `write` and the
/// file leaves it `unset`, otherwise overridden in `run` when it differs.
fn place<T: Clone + PartialEq>(
    applied: &mut Applied,
    key: &'static str,
    flag: Option<T>,
    write: bool,
    unset: impl Fn(&T) -> bool,
    file: &mut T,
    run: &mut T,
) {
    let Some(value) = flag else { return };
    if write && unset(file) {
        *file = value.clone();
        applied.written.push(key);
    } else if *file != value {
        applied.overridden.push(key);
    }
    *run = value;
}

/// The keys the command line overrode for this run only, with settings.json
/// as launch left it and the settings the run started with. Every save
/// serializes Settings whole, so without this a `/model` or `/provider` save
/// would write the run's endpoint over the file's.
#[derive(Clone, Debug)]
pub struct RunOnly {
    file: Settings,
    run: Settings,
    keys: Vec<&'static str>,
}

impl RunOnly {
    /// The overrides [`KeyFlags::apply`] reported, or None when there are none.
    pub fn new(file: Settings, run: Settings, applied: &Applied) -> Option<Self> {
        (!applied.overridden.is_empty()).then(|| RunOnly { file, run, keys: applied.overridden.clone() })
    }

    /// What a save of `settings` writes, and the overrides that hold after
    /// it. An overridden key still at the run's value is written as the
    /// file holds it; one the session set to another value is written as
    /// the session set it, and is the session's from then on. While the run
    /// talks to another server than the file selects (the command line's
    /// provider, or with no provider its base_url), a model picked is that
    /// server's, so the file keeps its own model as well. A provider the
    /// session switched to is the session's endpoint, so its model is saved
    /// with it: the file's model may be one that provider does not serve.
    pub fn save(&self, settings: &Settings) -> (Settings, RunOnly) {
        let mut saved = settings.clone();
        let mut next = RunOnly { keys: Vec::new(), ..self.clone() };
        macro_rules! keep {
            ($($key:ident),*) => {$(
                if self.keys.contains(&stringify!($key)) && settings.$key == self.run.$key {
                    saved.$key = self.file.$key.clone();
                    next.keys.push(stringify!($key));
                }
            )*};
        }
        keep!(provider, base_url, api_key, model, context_tokens);
        let held = |key| next.keys.contains(&key);
        if held("provider") || (selected(&settings.provider).is_none() && held("base_url")) {
            saved.model = self.file.model.clone();
        }
        if selected(&settings.provider) != selected(&self.run.provider) {
            saved.model = settings.model.clone();
            next.keys.retain(|key| *key != "model");
        }
        (saved, next)
    }
}

/// Load settings. A missing file means defaults; a file that exists but does
/// not parse is a hard error. Falling back to defaults on malformed input
/// would silently discard the user's endpoint and approval policy.
pub fn load(data_dir: &Path) -> Result<Settings, String> {
    let path = settings_path(data_dir);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Settings::default());
        }
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    serde_json::from_str(&text)
        .map_err(|e| format!("invalid settings file {}: {e}", path.display()))
}

/// Write settings.json and return the exact bytes written. The one caller
/// is `Core::save_settings`, which fingerprints what it wrote rather than
/// re-reading a path something else may have replaced in the meantime, so a
/// session's own save is never mistaken for external drift. Nothing outside
/// the core writes settings.json.
pub(crate) fn save_bytes(data_dir: &Path, settings: &Settings) -> Result<Vec<u8>, String> {
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    let destination = settings_path(data_dir);
    crate::sessions::write_atomic(&destination, json.clone())?;
    // Endpoint resolution is cached; force a re-read after settings change.
    crate::providers::invalidate_providers_cache();
    Ok(json.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_iteration_cap_defaults_to_50() {
        let s: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(s.max_agent_iterations, 50);
    }

    #[test]
    fn default_settings_use_iteration_cap() {
        let s = Settings::default();
        assert_eq!(s.max_agent_iterations, 50);
        assert_eq!(s.max_parallel_tools, 4);
    }

    #[test]
    fn iteration_cap_round_trips_when_present() {
        let s: Settings = serde_json::from_str(r#"{"max_agent_iterations":3}"#).unwrap();
        assert_eq!(s.max_agent_iterations, 3);
    }

    #[test]
    fn parallel_tool_limit_defaults_and_round_trips() {
        let defaulted: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted.max_parallel_tools, 4);
        let configured: Settings = serde_json::from_str(r#"{"max_parallel_tools":7}"#).unwrap();
        assert_eq!(configured.max_parallel_tools, 7);
    }

    #[test]
    fn compaction_tokens_parses_and_defaults_to_none() {
        let defaulted: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted.compaction_tokens, None);
        let configured: Settings =
            serde_json::from_str(r#"{"compaction_tokens":150000}"#).unwrap();
        assert_eq!(configured.compaction_tokens, Some(150_000));
    }

    /// Absent from every settings.json that exists today, so the default has
    /// to be "no ceiling"; and the round trip has to survive `deny_unknown_fields`,
    /// which is what a serialized-but-unparsed key would fail.
    #[test]
    fn max_agent_tokens_defaults_to_none_and_round_trips() {
        let defaulted: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted.max_agent_tokens, None);
        assert_eq!(Settings::default().max_agent_tokens, None);
        let configured: Settings =
            serde_json::from_str(r#"{"max_agent_tokens":120000}"#).unwrap();
        assert_eq!(configured.max_agent_tokens, Some(120_000));
        let json = serde_json::to_string(&configured).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_agent_tokens, Some(120_000));
    }

    /// The TUI saves the whole struct, so an unset temperature has to stay
    /// out of the file: a written default would be indistinguishable from
    /// the user's choice and would go out on every request after.
    #[test]
    fn unset_temperature_is_not_written_by_a_settings_save() {
        assert_eq!(Settings::default().temperature, None);
        let defaulted: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted.temperature, None);
        let dir = temp_dir("temperature-unset");
        let bytes = save_bytes(&dir, &Settings { model: "m".into(), ..Settings::default() }).unwrap();
        let written: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(written.get("temperature").is_none(), "{written}");
        assert_eq!(load(&dir).unwrap().temperature, None);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A file written before temperature became optional carries the old
    /// default explicitly; it still parses and keeps sending that value.
    #[test]
    fn explicit_temperature_parses_and_round_trips() {
        let s: Settings = serde_json::from_str(r#"{"temperature": 0.2}"#).unwrap();
        assert_eq!(s.temperature, Some(0.2));
        let json: serde_json::Value = serde_json::to_value(&s).unwrap();
        assert!((json["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6, "{json}");
        let back: Settings = serde_json::from_value(json).unwrap();
        assert_eq!(back.temperature, Some(0.2));
    }

    #[test]
    fn load_missing_file_uses_iteration_default() {
        let dir = std::env::temp_dir().join(format!(
            "openmax-settings-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = load(&dir).unwrap();
        assert_eq!(s.max_agent_iterations, 50);
        let _ = std::fs::remove_dir_all(dir);
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "openmax-settings-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn malformed_settings_file_is_an_error_not_defaults() {
        let dir = temp_dir("malformed");
        std::fs::write(dir.join("settings.json"), "{\"model\": \"m\",}").unwrap();
        let err = load(&dir).unwrap_err();
        assert!(err.contains("invalid settings file"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unknown_approval_mode_is_an_error() {
        let dir = temp_dir("badmode");
        std::fs::write(
            dir.join("settings.json"),
            r#"{"approval_mode": "read-only"}"#,
        )
        .unwrap();
        assert!(load(&dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unknown_settings_key_is_an_error() {
        let dir = temp_dir("badkey");
        std::fs::write(
            dir.join("settings.json"),
            r#"{"approval_mod": "readonly"}"#,
        )
        .unwrap();
        assert!(load(&dir).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn flags(base_url: Option<&str>, model: Option<&str>) -> KeyFlags {
        KeyFlags { base_url: base_url.map(Into::into), model: model.map(Into::into), ..KeyFlags::default() }
    }

    /// A first run configures the install: every flag whose key the file
    /// leaves unset is written. A key the file sets is overridden for the run
    /// only, and a flag equal to the file's value does nothing at all.
    #[test]
    fn key_flags_write_what_the_file_leaves_unset_and_override_the_rest() {
        let all = KeyFlags {
            base_url: Some("http://127.0.0.1:1/v1".into()),
            api_key: Some("$KEY".into()),
            model: Some("m".into()),
            context_tokens: Some(8192),
            ..KeyFlags::default()
        };
        let mut file = Settings::default();
        let (run, applied) = all.apply(&mut file);
        assert_eq!(applied.written, ["base_url", "api_key", "model", "context_tokens"]);
        assert!(applied.overridden.is_empty(), "{applied:?}");
        assert_eq!((file.base_url.as_str(), file.api_key.as_deref(), file.model.as_str(), file.context_tokens), ("http://127.0.0.1:1/v1", Some("$KEY"), "m", Some(8192)));
        assert_eq!(serde_json::to_value(&run).unwrap(), serde_json::to_value(&file).unwrap());

        // An empty api_key is as unset as an absent one.
        let mut file = Settings { base_url: "http://a/v1".into(), api_key: Some(" ".into()), context_tokens: Some(1), ..Settings::default() };
        let (run, applied) = KeyFlags { api_key: Some("k".into()), context_tokens: Some(2), ..flags(None, Some("m")) }.apply(&mut file);
        assert_eq!((applied.written.as_slice(), applied.overridden.as_slice()), (&["api_key", "model"][..], &["context_tokens"][..]));
        assert_eq!((file.api_key.as_deref(), file.model.as_str(), file.context_tokens), (Some("k"), "m", Some(1)), "the file keeps its own window");
        assert_eq!(run.context_tokens, Some(2), "the run uses the flag's");

        let mut same = file.clone();
        let (_, applied) = flags(Some("http://a/v1"), Some("m")).apply(&mut same);
        assert_eq!(applied, Applied::default(), "a flag equal to the file changes nothing");
    }

    /// A key is written only for the endpoint settings.json selects. Beside a
    /// `settings.provider`, `--base-url` sets that provider aside for the
    /// run, and another `--provider` picks another endpoint, so each flag is
    /// for the run only; the file's own provider leaves writing as it is.
    #[test]
    fn key_flags_for_another_endpoint_write_nothing() {
        let mut file = Settings { provider: Some("p".into()), ..Settings::default() };
        let (run, applied) = flags(Some("http://b/v1"), Some("m")).apply(&mut file);
        assert_eq!(applied.written, Vec::<&str>::new());
        assert_eq!(applied.overridden, ["provider", "base_url", "model"]);
        assert_eq!((run.provider.as_deref(), file.provider.as_deref()), (None, Some("p")));
        assert!(file.base_url.is_empty() && file.model.is_empty(), "{file:?}");

        let mut file = Settings::default();
        let (run, applied) = KeyFlags { provider: Some("q".into()), ..flags(None, Some("m")) }.apply(&mut file);
        assert_eq!((applied.written.len(), applied.overridden.as_slice()), (0, &["provider", "model"][..]));
        assert_eq!((run.provider.as_deref(), run.model.as_str(), file.model.as_str()), (Some("q"), "m", ""));

        let mut file = Settings { provider: Some("q".into()), ..Settings::default() };
        let (_, applied) = KeyFlags { provider: Some("q".into()), ..flags(None, Some("m")) }.apply(&mut file);
        assert_eq!((applied.written.as_slice(), applied.overridden.len()), (&["model"][..], 0));
        assert_eq!(file.model, "m");

        // A base_url other than the file's is another server: a key written
        // for it would go to the file's server on the next plain run.
        let mut file = Settings { base_url: "http://a/v1".into(), ..Settings::default() };
        let (run, applied) = KeyFlags { api_key: Some("$B_KEY".into()), ..flags(Some("http://b/v1"), Some("m")) }.apply(&mut file);
        assert_eq!((applied.written.len(), applied.overridden.as_slice()), (0, &["base_url", "api_key", "model"][..]));
        assert_eq!((file.api_key.as_deref(), file.model.as_str()), (None, ""), "{file:?}");
        assert_eq!((run.base_url.as_str(), run.api_key.as_deref()), ("http://b/v1", Some("$B_KEY")));
        // The file's own base_url, or none yet, is the file's server.
        for base_url in ["http://a/v1", " "] {
            let mut file = Settings { base_url: base_url.into(), ..Settings::default() };
            let (_, applied) = flags(Some("http://a/v1"), Some("m")).apply(&mut file);
            assert!(applied.written.contains(&"model") && applied.overridden.is_empty(), "{base_url:?}: {applied:?}");
        }
    }

    /// The file's api_key is the credential for the server its base_url
    /// names, by scheme, host and port. A `--base-url` for another server
    /// does not carry it there: the run has no settings key unless
    /// `--api-key` gives one, and `api_key` is overridden, so no save drops
    /// the file's. Another path on the same server keeps the key.
    #[test]
    fn a_base_url_flag_for_another_server_withholds_the_files_key() {
        let file = Settings { base_url: "http://a:8080/v1".into(), api_key: Some("$A_KEY".into()), ..Settings::default() };
        for other in ["http://b:8080/v1", "https://a:8080/v1", "http://a:8081/v1"] {
            let (run, applied) = flags(Some(other), None).apply(&mut file.clone());
            assert_eq!((run.api_key, applied.withheld_key), (None, true), "{other}");
            assert_eq!(applied.overridden, ["base_url", "api_key"], "{other}");
        }
        let (run, applied) = flags(Some("http://a:8080/v2"), None).apply(&mut file.clone());
        assert_eq!((run.api_key.as_deref(), applied.withheld_key), (Some("$A_KEY"), false), "the same server");
        let own = KeyFlags { api_key: Some("$B_KEY".into()), ..flags(Some("http://b:8080/v1"), None) };
        let (run, applied) = own.apply(&mut file.clone());
        assert_eq!((run.api_key.as_deref(), applied.withheld_key), (Some("$B_KEY"), false), "the run's own key");
        let keyless = Settings { api_key: None, ..file.clone() };
        let (_, applied) = flags(Some("http://b:8080/v1"), None).apply(&mut keyless.clone());
        assert_eq!((applied.withheld_key, applied.overridden.as_slice()), (false, &["base_url"][..]), "no key to withhold");
        // A base_url the run writes makes its server the file's; one only
        // used for the run, beside the file's provider, does not.
        let unset = Settings { base_url: String::new(), ..file.clone() };
        let (run, applied) = flags(Some("http://b:8080/v1"), None).apply(&mut unset.clone());
        assert_eq!((run.api_key.as_deref(), applied.written.as_slice()), (Some("$A_KEY"), &["base_url"][..]));
        let beside = Settings { provider: Some("p".into()), ..unset };
        let (run, applied) = flags(Some("http://b:8080/v1"), None).apply(&mut beside.clone());
        assert_eq!((run.api_key, applied.withheld_key), (None, true));
    }

    /// Values are checked before anything runs, so a placeholder or a typo
    /// is refused, naming its flag, instead of written to settings.json:
    /// each placeholder of the first command the installer and the first-run
    /// error print, and a key read from a variable that is not set.
    #[test]
    fn key_flag_values_that_configure_nothing_are_refused() {
        for url in ["URL", "", "localhost:8080/v1", "ftp://host/v1"] {
            let problem = flags(Some(url), None).problem().unwrap_or_else(|| panic!("{url:?} must be refused"));
            assert!(problem.starts_with("--base-url takes") && problem.contains(&format!("'{url}'")), "{problem}");
        }
        assert_eq!(flags(Some("https://host/v1"), Some("m")).problem(), None);
        assert_eq!(flags(Some(" http://127.0.0.1:11434/v1 "), None).problem(), None);
        let empty = |flags: KeyFlags, flag: &str| {
            assert_eq!(flags.problem().as_deref(), Some(format!("{flag} takes a non-empty value").as_str()));
        };
        empty(flags(None, Some(" ")), "--model");
        empty(KeyFlags { api_key: Some(String::new()), ..KeyFlags::default() }, "--api-key");
        empty(KeyFlags { provider: Some(String::new()), ..KeyFlags::default() }, "--provider");
        let zero = KeyFlags { context_tokens: Some(0), ..KeyFlags::default() }.problem().unwrap();
        assert!(zero.starts_with("--context-tokens takes"), "{zero}");
        assert_eq!(KeyFlags { context_tokens: Some(1), ..KeyFlags::default() }.problem(), None);

        let placeholder = flags(None, Some("MODEL")).problem().unwrap();
        assert!(placeholder.starts_with("--model takes") && placeholder.contains("placeholder"), "{placeholder}");
        assert_eq!(flags(None, Some("model")).problem(), None, "only the placeholder itself");
        let key = |key: &str| KeyFlags { api_key: Some(key.into()), ..KeyFlags::default() }.problem();
        let unset = format!("$OMX_UNSET_{}", uuid::Uuid::new_v4().simple());
        let refused = key(&unset).unwrap_or_else(|| panic!("{unset} is not set, so it configures no key"));
        assert!(refused.starts_with(&format!("--api-key '{unset}' reads an environment variable that is not set")), "{refused}");
        assert!(key("$").is_some(), "a bare $ names no variable");
        // PATH is set wherever tests run; a literal and a $$ escape name none.
        for configured in ["$PATH", "sk-literal", "$$literal"] {
            assert_eq!(key(configured), None, "{configured}");
        }
    }

    /// `--api-key` sets the key for `base_url`, which a provider's requests
    /// never carry to another server: beside the provider this run uses,
    /// from the flag or from settings.json, it is refused naming that
    /// provider, and with `--base-url` the run is on the flat endpoint again.
    #[test]
    fn an_api_key_flag_beside_the_runs_provider_is_refused() {
        let key = KeyFlags { api_key: Some("$K".into()), ..KeyFlags::default() };
        let file = Settings { provider: Some("p".into()), ..Settings::default() };
        let reason = key.conflict(&file).expect("settings.json selects a provider");
        assert!(reason.starts_with("--api-key is the key for settings.json's base_url") && reason.contains("provider 'p'"), "{reason}");
        let named = KeyFlags { provider: Some("q".into()), ..key.clone() };
        assert!(named.conflict(&Settings::default()).is_some_and(|r| r.contains("provider 'q'")));
        assert_eq!(key.conflict(&Settings { provider: Some(" ".into()), ..Settings::default() }), None, "a blank provider is the flat endpoint");
        assert_eq!(KeyFlags { base_url: Some("http://h/v1".into()), ..key.clone() }.conflict(&file), None);
        assert_eq!(KeyFlags { api_key: None, ..named }.conflict(&file), None);
    }

    #[test]
    fn approval_mode_round_trips_as_lowercase_string() {
        let s: Settings = serde_json::from_str(r#"{"approval_mode": "readonly"}"#).unwrap();
        assert_eq!(s.approval_mode, ApprovalMode::Readonly);
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains(r#""approval_mode":"readonly""#), "{json}");
    }

    /// The step key reaches `ask` and `auto` from anywhere and never lands
    /// in `readonly`: a saved `readonly` from a stray press refuses every
    /// write in the project's next session, so that mode is only ever chosen
    /// on purpose.
    #[test]
    fn stepping_approval_modes_switches_ask_and_auto_and_never_enters_readonly() {
        assert_eq!(ApprovalMode::Ask.next(), ApprovalMode::Auto);
        assert_eq!(ApprovalMode::Auto.next(), ApprovalMode::Ask);
        assert_eq!(ApprovalMode::Readonly.next(), ApprovalMode::Ask, "the key leaves readonly");
        for mode in [ApprovalMode::Readonly, ApprovalMode::Ask, ApprovalMode::Auto] {
            assert_ne!(mode.next(), ApprovalMode::Readonly, "{mode:?} must not step into readonly");
        }
    }

    #[test]
    fn save_atomically_replaces_settings_without_leaving_temp_files() {
        let dir = std::env::temp_dir().join(format!(
            "openmax-settings-save-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let first = Settings { model: "one".into(), ..Settings::default() };
        save_bytes(&dir, &first).unwrap();
        let mut second = first.clone();
        second.model = "vendor/family/two".into();
        save_bytes(&dir, &second).unwrap();
        assert_eq!(load(&dir).unwrap().model, "vendor/family/two");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
