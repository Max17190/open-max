use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const DEFAULT_MLX_PORT: u16 = 8989;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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
    /// "auto" (run everything), "ask" (approve writes/commands), "readonly".
    pub approval_mode: String,
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
}

fn default_max_agent_iterations() -> usize {
    50
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            provider: None,
            base_url: format!("http://127.0.0.1:{DEFAULT_MLX_PORT}/v1"),
            api_key: None,
            model: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit".into(),
            approval_mode: "ask".into(),
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
        }
    }
}

fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("settings.json")
}

pub fn load(data_dir: &Path) -> Settings {
    std::fs::read_to_string(settings_path(data_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(data_dir: &Path, settings: &Settings) -> Result<(), String> {
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(settings_path(data_dir), json).map_err(|e| e.to_string())?;
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
    }

    #[test]
    fn iteration_cap_round_trips_when_present() {
        let s: Settings = serde_json::from_str(r#"{"max_agent_iterations":3}"#).unwrap();
        assert_eq!(s.max_agent_iterations, 3);
    }

    #[test]
    fn unknown_max_task_iterations_key_is_ignored() {
        // Pre-removal settings may still carry this key; serde should not fail.
        let s: Settings = serde_json::from_str(
            r#"{"max_agent_iterations":7,"max_task_iterations":2}"#,
        )
        .unwrap();
        assert_eq!(s.max_agent_iterations, 7);
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
        let s = load(&dir);
        assert_eq!(s.max_agent_iterations, 50);
        let _ = std::fs::remove_dir_all(dir);
    }
}
