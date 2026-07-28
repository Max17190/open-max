use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const DEFAULT_MLX_PORT: u16 = 8989;

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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// Named provider from `providers.json`. When set and found, supplies
    /// base_url, credentials, and headers; flat fields remain the fallback.
    #[serde(default)]
    pub provider: Option<String>,
    /// OpenAI-compatible base URL. Defaults to the optional managed local MLX
    /// port; any other compatible endpoint can be configured instead.
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub approval_mode: ApprovalMode,
    pub context_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    /// HuggingFace repo id served by the managed MLX server.
    pub mlx_model: String,
    pub mlx_port: u16,
    /// Draft model repo id for speculative decoding. Opt-in: payoff is
    /// hardware-dependent (and negative on MoE models), and setting it
    /// disables the server's continuous batching.
    pub draft_model: Option<String>,
    /// Tokens drafted per speculative step; only sent alongside draft_model.
    pub num_draft_tokens: Option<u32>,
    /// JSON object passed to the chat template, e.g. {"enable_thinking": false}
    /// to cut reasoning tokens on Qwen3-family models.
    pub chat_template_args: Option<String>,
    /// Byte cap for bash/external tool output before tail-truncation with
    /// spill-to-file. Unset means the tuned built-in default.
    pub max_output_bytes: Option<usize>,
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
            base_url: format!("http://127.0.0.1:{DEFAULT_MLX_PORT}/v1"),
            api_key: None,
            model: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit".into(),
            approval_mode: ApprovalMode::Ask,
            context_tokens: 16384,
            max_tokens: 4096,
            temperature: 0.2,
            mlx_model: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit".into(),
            mlx_port: DEFAULT_MLX_PORT,
            draft_model: None,
            num_draft_tokens: None,
            chat_template_args: None,
            max_output_bytes: None,
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
