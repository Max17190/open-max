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
//! be listening.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Gate for mutating tools. Parsed strictly: an unrecognized value is a
/// configuration error, never a silent fallback, because a typo here would
/// otherwise weaken the approval gate the user asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    /// Run everything without prompting.
    Auto,
    /// Prompt before mutating tools.
    Ask,
    /// Block mutating tools entirely.
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

    /// The next mode when cycling through all three, ordered by how much the
    /// agent is allowed to do: readonly, ask, auto, and back. One source of
    /// truth so a front end offering a cycle key cannot invent its own order.
    pub fn next(self) -> Self {
        match self {
            ApprovalMode::Readonly => ApprovalMode::Ask,
            ApprovalMode::Ask => ApprovalMode::Auto,
            ApprovalMode::Auto => ApprovalMode::Readonly,
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
    pub context_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    /// Byte cap for bash/external tool output before tail-truncation with
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
    /// work a turn may ask for and not what it happened to cost. Checked
    /// before each request, so it can only end a turn early, never extend one.
    /// A compaction summary is not charged here: it is housekeeping that makes
    /// the next request smaller, and refusing a turn for it would spend the
    /// ceiling on the thing that was saving it. None means no ceiling (the
    /// behavior every existing settings.json has today).
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
            context_tokens: 16384,
            max_tokens: 4096,
            temperature: 0.2,
            max_output_bytes: None,
            compaction_tokens: None,
            max_agent_tokens: None,
            max_agent_iterations: default_max_agent_iterations(),
            max_parallel_tools: default_max_parallel_tools(),
        }
    }
}

fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("settings.json")
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

pub fn save(data_dir: &Path, settings: &Settings) -> Result<(), String> {
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    let destination = settings_path(data_dir);
    crate::sessions::write_atomic(&destination, json)?;
    // Endpoint resolution is cached; force a re-read after settings change.
    crate::providers::invalidate_providers_cache();
    Ok(())
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

    #[test]
    fn approval_mode_round_trips_as_lowercase_string() {
        let s: Settings = serde_json::from_str(r#"{"approval_mode": "readonly"}"#).unwrap();
        assert_eq!(s.approval_mode, ApprovalMode::Readonly);
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains(r#""approval_mode":"readonly""#), "{json}");
    }

    /// Cycling has to reach every mode and come back, or a front end that
    /// only offers the cycle key would strand the user in a subset of them.
    #[test]
    fn cycling_approval_modes_visits_each_one_and_returns() {
        let start = ApprovalMode::Ask;
        let mut seen = vec![start];
        let mut mode = start;
        for _ in 0..2 {
            mode = mode.next();
            assert!(!seen.contains(&mode), "cycle repeats before it closes");
            seen.push(mode);
        }
        assert_eq!(mode.next(), start, "cycle does not close");
        // Ordered by how much the agent may do, so the key reads as one
        // direction rather than an arbitrary rotation.
        assert_eq!(
            seen,
            vec![
                ApprovalMode::Ask,
                ApprovalMode::Auto,
                ApprovalMode::Readonly
            ],
        );
        assert_eq!(ApprovalMode::Readonly.next(), ApprovalMode::Ask);
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
        save(&dir, &first).unwrap();
        let mut second = first.clone();
        second.model = "vendor/family/two".into();
        save(&dir, &second).unwrap();
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
