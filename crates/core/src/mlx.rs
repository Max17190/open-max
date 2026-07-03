use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::state::Core;

const MAX_LOG_LINES: usize = 400;

/// State of the managed `mlx_lm.server` process.
#[derive(Default)]
pub struct MlxProc {
    pub child: Option<tokio::process::Child>,
    /// A server started by a previous run and adopted at startup.
    pub external_pid: Option<u32>,
    pub model: Option<String>,
    pub port: u16,
    pub ready: bool,
    pub setting_up: bool,
    pub logs: VecDeque<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MlxStatus {
    pub python_ok: bool,
    pub venv_ready: bool,
    pub setting_up: bool,
    pub server_running: bool,
    pub server_ready: bool,
    pub model: Option<String>,
    pub port: u16,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MlxEvent {
    SetupLog { line: String },
    SetupDone { ok: bool, message: String },
    ServerLog { line: String },
    ServerReady { model: String },
    ServerExit { code: i32 },
}

/// Persisted so the next launch can adopt a still-running server.
#[derive(Serialize, Deserialize)]
struct ServerInfo {
    pid: u32,
    model: String,
    port: u16,
}

fn venv_dir(core: &Core) -> PathBuf {
    core.data_dir.join("mlx-venv")
}

fn server_bin(core: &Core) -> PathBuf {
    venv_dir(core).join("bin").join("mlx_lm.server")
}

fn server_info_path(core: &Core) -> PathBuf {
    core.data_dir.join("mlx-server.json")
}

fn push_log(core: &Core, line: String, setup: bool) {
    {
        let mut mlx = core.mlx.lock().unwrap();
        mlx.logs.push_back(line.clone());
        while mlx.logs.len() > MAX_LOG_LINES {
            mlx.logs.pop_front();
        }
    }
    if setup {
        core.send_mlx(MlxEvent::SetupLog { line });
    } else {
        core.send_mlx(MlxEvent::ServerLog { line });
    }
}

/// Locate uv, checking PATH and the usual install locations.
fn find_uv() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            candidates.push(dir.join("uv"));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        candidates.push(home.join(".local/bin/uv"));
        candidates.push(home.join(".cargo/bin/uv"));
    }
    candidates.push(PathBuf::from("/opt/homebrew/bin/uv"));
    candidates.push(PathBuf::from("/usr/local/bin/uv"));
    candidates.into_iter().find(|p| p.is_file())
}

/// A working python3 cannot disappear mid-session, so a successful probe is
/// cached: status() runs on a ~2s tick while the models panel is open, and
/// spawning `python3 --version` each tick is pure waste. Failures re-probe,
/// so installing Python mid-session is still picked up.
async fn python3_ok() -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static PYTHON_OK: AtomicBool = AtomicBool::new(false);
    if PYTHON_OK.load(Ordering::Relaxed) {
        return true;
    }
    let ok = tokio::process::Command::new("python3")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        PYTHON_OK.store(true, Ordering::Relaxed);
    }
    ok
}

/// True if `pid` is alive and its command line looks like our server.
fn pid_is_mlx_server(pid: u32) -> bool {
    let out = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).contains("mlx_lm")
        }
        _ => false,
    }
}

pub async fn status(core: &Arc<Core>) -> MlxStatus {
    let python_ok = find_uv().is_some() || python3_ok().await;
    let venv_ready = server_bin(core).exists();
    let mut mlx = core.mlx.lock().unwrap();
    let server_running = match mlx.child.as_mut() {
        Some(child) => child.try_wait().map(|s| s.is_none()).unwrap_or(false),
        None => match mlx.external_pid {
            Some(pid) => pid_is_mlx_server(pid),
            None => false,
        },
    };
    if !server_running {
        mlx.ready = false;
        mlx.external_pid = None;
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

/// Adopt a server left running by a previous launch, if it still answers.
/// Returns true when a server was adopted.
pub async fn reattach(core: &Arc<Core>) -> bool {
    let Some(info) = std::fs::read_to_string(server_info_path(core))
        .ok()
        .and_then(|s| serde_json::from_str::<ServerInfo>(&s).ok())
    else {
        return false;
    };
    if !pid_is_mlx_server(info.pid) {
        let _ = std::fs::remove_file(server_info_path(core));
        return false;
    }
    let url = format!("http://127.0.0.1:{}/v1/models", info.port);
    let answers = matches!(
        reqwest::Client::new().get(&url).timeout(Duration::from_secs(2)).send().await,
        Ok(resp) if resp.status().is_success()
    );
    if !answers {
        return false;
    }
    {
        let mut mlx = core.mlx.lock().unwrap();
        mlx.external_pid = Some(info.pid);
        mlx.model = Some(info.model.clone());
        mlx.port = info.port;
        mlx.ready = true;
    }
    core.send_mlx(MlxEvent::ServerReady { model: info.model });
    true
}

/// Provision a private venv with mlx-lm under the data dir. Idempotent.
/// Prefers uv (which manages Python itself); falls back to python3 -m venv.
pub fn setup(core: Arc<Core>) -> Result<(), String> {
    {
        let mut mlx = core.mlx.lock().unwrap();
        if mlx.setting_up {
            return Err("setup is already running".into());
        }
        mlx.setting_up = true;
    }

    tokio::spawn(async move {
        let result = run_setup(&core).await;
        core.mlx.lock().unwrap().setting_up = false;
        match result {
            Ok(()) => core.send_mlx(MlxEvent::SetupDone { ok: true, message: "mlx-lm installed".into() }),
            Err(e) => core.send_mlx(MlxEvent::SetupDone { ok: false, message: e }),
        }
    });
    Ok(())
}

async fn run_setup(core: &Arc<Core>) -> Result<(), String> {
    let venv = venv_dir(core);
    let venv_python = venv.join("bin").join("python");

    if let Some(uv) = find_uv() {
        if !venv_python.exists() {
            push_log(core, format!("Creating environment at {} (uv)", venv.display()), true);
            run_logged(
                core,
                tokio::process::Command::new(&uv).arg("venv").arg(&venv).args(["--python", "3.12"]),
                true,
            )
            .await?;
        }
        push_log(core, "Installing mlx-lm (this can take a few minutes)…".into(), true);
        run_logged(
            core,
            tokio::process::Command::new(&uv)
                .args(["pip", "install", "--upgrade", "mlx-lm"])
                .arg("--python")
                .arg(&venv_python),
            true,
        )
        .await?;
        return Ok(());
    }

    if !python3_ok().await {
        return Err("neither uv nor python3 was found; install uv from https://docs.astral.sh/uv or Python 3.9+".into());
    }
    if !venv_python.exists() {
        push_log(core, format!("Creating virtualenv at {}", venv.display()), true);
        run_logged(core, tokio::process::Command::new("python3").args(["-m", "venv"]).arg(&venv), true).await?;
    }
    push_log(core, "Installing mlx-lm (this can take a few minutes)…".into(), true);
    let pip = venv.join("bin").join("pip");
    run_logged(
        core,
        tokio::process::Command::new(&pip).args(["install", "--upgrade", "pip", "mlx-lm"]),
        true,
    )
    .await?;
    Ok(())
}

async fn run_logged(core: &Arc<Core>, cmd: &mut tokio::process::Command, setup: bool) -> Result<(), String> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| format!("failed to start process: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (a, b) = (core.clone(), core.clone());
    let h1 = tokio::spawn(async move {
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&a, line, setup);
            }
        }
    });
    let h2 = tokio::spawn(async move {
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

/// Start (or restart) the server for `model` on `port`. The model is pulled
/// from HuggingFace on first use; progress shows up in the server logs.
pub fn start(core: Arc<Core>, model: String, port: u16) -> Result<(), String> {
    let bin = server_bin(&core);
    if !bin.exists() {
        return Err("the MLX environment is not set up yet; run setup first".into());
    }
    stop(&core);

    let mut cmd = tokio::process::Command::new(&bin);
    // WARNING log level: at INFO the server logs prompt-cache stats and
    // prompt-processing progress during every generation, and each line costs
    // a pipe read, a deque push, an event, and a TUI wakeup. Readiness is
    // probed over HTTP, not parsed from logs, so nothing needed is lost.
    cmd.args([
        "--model", &model,
        "--port", &port.to_string(),
        "--host", "127.0.0.1",
        "--log-level", "WARNING",
    ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        // The server deliberately outlives the TUI so relaunches reattach
        // instantly; stop() kills it explicitly.
        .kill_on_drop(false);

    let mut child = cmd.spawn().map_err(|e| format!("failed to start mlx_lm.server: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    if let Some(pid) = child.id() {
        let info = ServerInfo { pid, model: model.clone(), port };
        if let Ok(json) = serde_json::to_string(&info) {
            let _ = std::fs::write(server_info_path(&core), json);
        }
    }

    {
        let mut mlx = core.mlx.lock().unwrap();
        mlx.child = Some(child);
        mlx.external_pid = None;
        mlx.model = Some(model.clone());
        mlx.port = port;
        mlx.ready = false;
        mlx.logs.clear();
    }

    let (a, b) = (core.clone(), core.clone());
    tokio::spawn(async move {
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&a, line, false);
            }
        }
    });
    tokio::spawn(async move {
        if let Some(err) = stderr {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                push_log(&b, line, false);
            }
        }
    });

    // Poll until the server answers /v1/models (the first start may download
    // many GB of weights, so be patient) or the process dies.
    tokio::spawn(async move {
        let url = format!("http://127.0.0.1:{port}/v1/models");
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            {
                let mut mlx = core.mlx.lock().unwrap();
                match mlx.child.as_mut() {
                    Some(child) => {
                        if let Ok(Some(status)) = child.try_wait() {
                            mlx.child = None;
                            mlx.ready = false;
                            drop(mlx);
                            let _ = std::fs::remove_file(server_info_path(&core));
                            core.send_mlx(MlxEvent::ServerExit { code: status.code().unwrap_or(-1) });
                            return;
                        }
                    }
                    None => return, // stopped externally
                }
            }
            if let Ok(resp) = reqwest::get(&url).await {
                if resp.status().is_success() {
                    core.mlx.lock().unwrap().ready = true;
                    core.send_mlx(MlxEvent::ServerReady { model: model.clone() });
                    return;
                }
            }
        }
    });

    Ok(())
}

pub fn stop(core: &Arc<Core>) {
    let mut mlx = core.mlx.lock().unwrap();
    if let Some(child) = mlx.child.as_mut() {
        let _ = child.start_kill();
    } else if let Some(pid) = mlx.external_pid {
        // Adopted server from a previous run: verify identity before killing.
        if pid_is_mlx_server(pid) {
            let _ = std::process::Command::new("kill").arg(pid.to_string()).status();
        }
    }
    mlx.child = None;
    mlx.external_pid = None;
    mlx.ready = false;
    mlx.model = None;
    let _ = std::fs::remove_file(server_info_path(core));
}

pub fn logs(core: &Arc<Core>) -> Vec<String> {
    let mlx = core.mlx.lock().unwrap();
    mlx.logs.iter().cloned().collect()
}
