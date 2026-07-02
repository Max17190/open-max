use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::settings::app_data_dir;
use crate::state::AppState;

const MAX_LOG_LINES: usize = 400;

/// State of the managed `mlx_lm.server` sidecar.
#[derive(Default)]
pub struct MlxProc {
    pub child: Option<tokio::process::Child>,
    pub model: Option<String>,
    pub port: u16,
    pub ready: bool,
    pub setting_up: bool,
    pub logs: VecDeque<String>,
}

#[derive(Clone, Serialize)]
pub struct MlxStatus {
    pub python_ok: bool,
    pub venv_ready: bool,
    pub setting_up: bool,
    pub server_running: bool,
    pub server_ready: bool,
    pub model: Option<String>,
    pub port: u16,
}

#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MlxEvent {
    SetupLog { line: String },
    SetupDone { ok: bool, message: String },
    ServerLog { line: String },
    ServerReady { model: String },
    ServerExit { code: i32 },
}

fn emit(app: &AppHandle, event: MlxEvent) {
    let _ = app.emit("mlx_event", event);
}

fn venv_dir(app: &AppHandle) -> PathBuf {
    app_data_dir(app).join("mlx-venv")
}

fn server_bin(app: &AppHandle) -> PathBuf {
    venv_dir(app).join("bin").join("mlx_lm.server")
}

fn push_log(app: &AppHandle, line: String, setup: bool) {
    let state = app.state::<AppState>();
    {
        let mut mlx = state.mlx.lock().unwrap();
        mlx.logs.push_back(line.clone());
        while mlx.logs.len() > MAX_LOG_LINES {
            mlx.logs.pop_front();
        }
    }
    if setup {
        emit(app, MlxEvent::SetupLog { line });
    } else {
        emit(app, MlxEvent::ServerLog { line });
    }
}

pub async fn status(app: &AppHandle) -> MlxStatus {
    let python_ok = tokio::process::Command::new("python3")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);

    let venv_ready = server_bin(app).exists();
    let state = app.state::<AppState>();
    let mut mlx = state.mlx.lock().unwrap();
    let server_running = match mlx.child.as_mut() {
        Some(child) => child.try_wait().map(|s| s.is_none()).unwrap_or(false),
        None => false,
    };
    if !server_running {
        mlx.ready = false;
    }
    MlxStatus {
        python_ok,
        venv_ready,
        setting_up: mlx.setting_up,
        server_running,
        server_ready: mlx.ready,
        model: mlx.model.clone(),
        port: mlx.port,
    }
}

/// Provision a private venv with mlx-lm in the app data dir. Idempotent.
pub fn setup(app: AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();
    {
        let mut mlx = state.mlx.lock().unwrap();
        if mlx.setting_up {
            return Err("setup is already running".into());
        }
        mlx.setting_up = true;
    }

    tauri::async_runtime::spawn(async move {
        let result = run_setup(&app).await;
        {
            let state = app.state::<AppState>();
            state.mlx.lock().unwrap().setting_up = false;
        }
        match result {
            Ok(()) => emit(&app, MlxEvent::SetupDone { ok: true, message: "mlx-lm installed".into() }),
            Err(e) => emit(&app, MlxEvent::SetupDone { ok: false, message: e }),
        }
    });
    Ok(())
}

async fn run_setup(app: &AppHandle) -> Result<(), String> {
    let venv = venv_dir(app);
    if !venv.join("bin").join("python").exists() {
        push_log(app, format!("Creating virtualenv at {}", venv.display()), true);
        run_logged(app, tokio::process::Command::new("python3").args(["-m", "venv"]).arg(&venv), true).await?;
    }
    push_log(app, "Installing mlx-lm (this can take a few minutes)…".into(), true);
    let pip = venv.join("bin").join("pip");
    run_logged(
        app,
        tokio::process::Command::new(&pip).args(["install", "--upgrade", "pip", "mlx-lm"]),
        true,
    )
    .await?;
    Ok(())
}

async fn run_logged(app: &AppHandle, cmd: &mut tokio::process::Command, setup: bool) -> Result<(), String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| format!("failed to start process: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (a, b) = (app.clone(), app.clone());
    let h1 = tauri::async_runtime::spawn(async move {
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&a, line, setup);
            }
        }
    });
    let h2 = tauri::async_runtime::spawn(async move {
        if let Some(err) = stderr {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&b, line, setup);
            }
        }
    });
    let status = child.wait().await.map_err(|e| e.to_string())?;
    let _ = h1.await;
    let _ = h2.await;
    if !status.success() {
        return Err(format!("process exited with {:?}", status.code()));
    }
    Ok(())
}

/// Start (or restart) the sidecar serving `model` on `port`. The model is
/// pulled from HuggingFace on first use; progress shows up in the server logs.
pub fn start(app: AppHandle, model: String, port: u16) -> Result<(), String> {
    let bin = server_bin(&app);
    if !bin.exists() {
        return Err("MLX environment is not set up yet — run setup first".into());
    }
    stop(&app);

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(["--model", &model, "--port", &port.to_string(), "--host", "127.0.0.1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(false);

    let mut child = cmd.spawn().map_err(|e| format!("failed to start mlx_lm.server: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let state = app.state::<AppState>();
        let mut mlx = state.mlx.lock().unwrap();
        mlx.child = Some(child);
        mlx.model = Some(model.clone());
        mlx.port = port;
        mlx.ready = false;
        mlx.logs.clear();
    }

    let (a, b) = (app.clone(), app.clone());
    tauri::async_runtime::spawn(async move {
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&a, line, false);
            }
        }
    });
    tauri::async_runtime::spawn(async move {
        if let Some(err) = stderr {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&b, line, false);
            }
        }
    });

    // Poll until the server answers /v1/models (the first start may download
    // many GB of weights, so be patient) or the process dies.
    tauri::async_runtime::spawn(async move {
        let url = format!("http://127.0.0.1:{port}/v1/models");
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            {
                let state = app.state::<AppState>();
                let mut mlx = state.mlx.lock().unwrap();
                match mlx.child.as_mut() {
                    Some(child) => {
                        if let Ok(Some(status)) = child.try_wait() {
                            mlx.child = None;
                            mlx.ready = false;
                            drop(mlx);
                            emit(&app, MlxEvent::ServerExit { code: status.code().unwrap_or(-1) });
                            return;
                        }
                    }
                    None => return, // stopped externally
                }
            }
            if let Ok(resp) = reqwest::get(&url).await {
                if resp.status().is_success() {
                    let state = app.state::<AppState>();
                    state.mlx.lock().unwrap().ready = true;
                    emit(&app, MlxEvent::ServerReady { model: model.clone() });
                    return;
                }
            }
        }
    });

    Ok(())
}

pub fn stop(app: &AppHandle) {
    let state = app.state::<AppState>();
    let mut mlx = state.mlx.lock().unwrap();
    if let Some(child) = mlx.child.as_mut() {
        let _ = child.start_kill();
    }
    mlx.child = None;
    mlx.ready = false;
    mlx.model = None;
}

pub fn logs(app: &AppHandle) -> Vec<String> {
    let state = app.state::<AppState>();
    let mlx = state.mlx.lock().unwrap();
    mlx.logs.iter().cloned().collect()
}
