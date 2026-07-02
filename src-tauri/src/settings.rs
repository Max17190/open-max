use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

pub const DEFAULT_MLX_PORT: u16 = 8989;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// OpenAI-compatible base URL, e.g. http://127.0.0.1:8989/v1 (MLX),
    /// http://127.0.0.1:11434/v1 (Ollama), http://127.0.0.1:1234/v1 (LM Studio).
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    /// "auto" (run everything), "ask" (approve writes/commands), "readonly".
    pub approval_mode: String,
    pub context_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    /// HuggingFace repo id served by the managed MLX sidecar.
    pub mlx_model: String,
    pub mlx_port: u16,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            base_url: format!("http://127.0.0.1:{DEFAULT_MLX_PORT}/v1"),
            api_key: None,
            model: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit".into(),
            approval_mode: "ask".into(),
            context_tokens: 16384,
            max_tokens: 4096,
            temperature: 0.2,
            mlx_model: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit".into(),
            mlx_port: DEFAULT_MLX_PORT,
        }
    }
}

pub fn app_data_dir(app: &AppHandle) -> PathBuf {
    let dir = app
        .path()
        .app_data_dir()
        .expect("no app data dir available");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn settings_path(app: &AppHandle) -> PathBuf {
    app_data_dir(app).join("settings.json")
}

pub fn load(app: &AppHandle) -> Settings {
    std::fs::read_to_string(settings_path(app))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(app: &AppHandle, settings: &Settings) -> Result<(), String> {
    let json = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(settings_path(app), json).map_err(|e| e.to_string())
}
