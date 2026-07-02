//! The event loop and all interaction logic. One inline viewport at the
//! bottom holds the live surface (streaming tail, composer, status line);
//! everything finished is pushed into terminal scrollback and stays native.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use open_max_core::mlx::MlxEvent;
use open_max_core::state::{Core, CoreEvent, DownloadEvent};
use open_max_core::types::AgentEvent;
use open_max_core::{agent, config, hf, mlx, sessions, tools};
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::Frame;
use tokio::sync::mpsc;

use crate::input::{Composer, ComposerAction};
use crate::theme;
use crate::ui::tool_card::{self, DiffText};
use crate::ui::transcript::{insert_block, wrap_lines, Term};
use crate::ui::{markdown, models};

const TICK: Duration = Duration::from_millis(120);
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub struct Args {
    pub continue_session: bool,
}

#[derive(PartialEq)]
enum Mode {
    Chat,
    Models,
}

pub struct App {
    core: Arc<Core>,
    project: PathBuf,
    session_id: Option<String>,
    mode: Mode,
    composer: Composer,
    models: models::ModelsState,

    running: bool,
    stream_text: String,
    thinking_chars: usize,
    thinking_tail: String,
    show_thinking: bool,
    turn_started: Option<Instant>,
    first_token: Option<Instant>,
    stream_chars: usize,
    running_tool: Option<(String, String)>,
    pending_approval: Option<(String, String, String)>,
    pending_diffs: HashMap<String, DiffText>,
    tool_meta: HashMap<String, (String, String)>,
    last_tool_output: Option<String>,
    budget: Option<(usize, usize)>,
    quit_armed: bool,
    spinner_i: usize,
    tick_i: u64,

    hf_tx: mpsc::UnboundedSender<(String, u64)>,
    should_quit: bool,
    needs_redraw: bool,
}

pub async fn run(
    mut terminal: Term,
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<CoreEvent>,
    args: Args,
) -> std::io::Result<()> {
    let (hf_tx, mut hf_rx) = mpsc::unbounded_channel();
    let ram = ram_bytes();
    let mut app = App {
        core: core.clone(),
        project: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        session_id: None,
        mode: Mode::Chat,
        composer: Composer::new(&core.data_dir),
        models: models::ModelsState::new(ram),
        running: false,
        stream_text: String::new(),
        thinking_chars: 0,
        thinking_tail: String::new(),
        show_thinking: false,
        turn_started: None,
        first_token: None,
        stream_chars: 0,
        running_tool: None,
        pending_approval: None,
        pending_diffs: HashMap::new(),
        tool_meta: HashMap::new(),
        last_tool_output: None,
        budget: None,
        quit_armed: false,
        spinner_i: 0,
        tick_i: 0,
        hf_tx,
        should_quit: false,
        needs_redraw: true,
    };

    app.startup(&mut terminal, &args).await?;

    let mut term_events = crossterm::event::EventStream::new();
    let mut tick = tokio::time::interval(TICK);

    loop {
        tokio::select! {
            ev = term_events.next() => {
                match ev {
                    Some(Ok(e)) => app.on_term_event(&mut terminal, e).await?,
                    Some(Err(_)) | None => app.should_quit = true,
                }
            }
            Some(ce) = core_rx.recv() => app.on_core_event(&mut terminal, ce).await?,
            Some((repo, bytes)) = hf_rx.recv() => {
                app.models.set_remote_size(&repo, bytes);
                app.needs_redraw = true;
            }
            _ = tick.tick() => app.on_tick().await,
        }
        if app.should_quit {
            break;
        }
        if app.needs_redraw {
            terminal.draw(|f| app.draw(f))?;
            app.needs_redraw = false;
        }
    }
    Ok(())
}

fn ram_bytes() -> u64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(0)
}

impl App {
    async fn startup(&mut self, terminal: &mut Term, args: &Args) -> std::io::Result<()> {
        let (model, version) = {
            let s = self.core.settings.lock().unwrap();
            (s.model.clone(), env!("CARGO_PKG_VERSION"))
        };
        let dir = self
            .project
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| self.project.display().to_string());
        insert_block(
            terminal,
            vec![
                Line::from(vec![
                    Span::styled("◆ open max", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  v{version}"), Style::default().fg(theme::DIM)),
                ]),
                Line::from(Span::styled(
                    format!("  {model} · {dir} · /help for commands"),
                    Style::default().fg(theme::DIM),
                )),
            ],
        )?;

        // Adopt a still-running server from a previous launch.
        if mlx::reattach(&self.core).await {
            self.models.status = Some(mlx::status(&self.core).await);
        }

        if args.continue_session {
            let project = self.project.display().to_string();
            match sessions::latest(&self.core, &project) {
                Some(meta) => {
                    self.session_id = Some(meta.id.clone());
                    self.replay(terminal, &meta.id)?;
                }
                None => self.note(terminal, "no previous session here; starting fresh")?,
            }
        }
        Ok(())
    }

    /// Re-render a persisted session compactly on --continue.
    fn replay(&mut self, terminal: &mut Term, session_id: &str) -> std::io::Result<()> {
        let Some(messages) = sessions::load_messages(&self.core, session_id) else {
            return Ok(());
        };
        for m in &messages {
            match m.role.as_str() {
                "user" => {
                    if let Some(text) = &m.content {
                        self.insert_user_block(terminal, text)?;
                    }
                }
                "assistant" => {
                    if let Some(text) = &m.content {
                        if !text.trim().is_empty() {
                            insert_block(terminal, markdown::render(text, markdown::highlighter()))?;
                        }
                    }
                    if let Some(calls) = &m.tool_calls {
                        for call in calls {
                            let args: serde_json::Value =
                                serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::Value::Null);
                            let summary = tools::summarize_call(&call.function.name, &args);
                            insert_block(
                                terminal,
                                vec![Line::from(vec![
                                    Span::styled("· ", Style::default().fg(theme::DIM)),
                                    Span::styled(call.function.name.clone(), Style::default().fg(theme::ACCENT)),
                                    Span::raw(" "),
                                    Span::styled(summary, Style::default().fg(theme::DIM)),
                                ])],
                            )?;
                        }
                    }
                }
                _ => {}
            }
        }
        self.note(terminal, "continuing previous session")?;
        Ok(())
    }

    // ---------- terminal events ----------

    async fn on_term_event(&mut self, terminal: &mut Term, event: TermEvent) -> std::io::Result<()> {
        match event {
            TermEvent::Key(key) if key.kind != KeyEventKind::Release => {
                self.on_key(terminal, key).await?;
            }
            TermEvent::Paste(text) => {
                if self.mode == Mode::Chat && self.pending_approval.is_none() {
                    self.composer.insert_str(&text);
                }
            }
            TermEvent::Resize(_, _) => {}
            _ => {}
        }
        self.needs_redraw = true;
        Ok(())
    }

    async fn on_key(&mut self, terminal: &mut Term, key: KeyEvent) -> std::io::Result<()> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        // Ctrl+C: cancel a running turn, otherwise quit on the second press.
        if ctrl && key.code == KeyCode::Char('c') {
            if self.running {
                if let Some(id) = &self.session_id {
                    self.core.cancel(id);
                }
                self.quit_armed = false;
            } else if self.quit_armed {
                self.should_quit = true;
            } else {
                self.quit_armed = true;
            }
            return Ok(());
        }
        self.quit_armed = false;

        if ctrl && key.code == KeyCode::Char('o') {
            if let Some(output) = self.last_tool_output.clone() {
                let lines = output
                    .lines()
                    .map(|l| Line::from(Span::styled(format!("  {l}"), Style::default().fg(theme::DIM))))
                    .collect();
                insert_block(terminal, lines)?;
            }
            return Ok(());
        }
        if ctrl && key.code == KeyCode::Char('t') {
            self.show_thinking = !self.show_thinking;
            return Ok(());
        }

        if self.mode == Mode::Models {
            self.on_models_key(terminal, key).await?;
            return Ok(());
        }

        // Approval prompt swallows keys until answered.
        if let Some((id, name, _)) = self.pending_approval.clone() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.core.respond_approval(&id, true);
                    self.pending_approval = None;
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.core.respond_approval(&id, false);
                    self.pending_approval = None;
                }
                KeyCode::Char('a') | KeyCode::Char('A') => {
                    self.core.settings.lock().unwrap().approval_mode = "auto".into();
                    self.core.respond_approval(&id, true);
                    self.pending_approval = None;
                    self.note(terminal, "approvals set to auto for this run (change with /approvals)")?;
                }
                _ => {
                    let _ = name;
                }
            }
            return Ok(());
        }

        if key.code == KeyCode::Esc {
            if self.running {
                if let Some(id) = &self.session_id {
                    self.core.cancel(id);
                }
            }
            return Ok(());
        }

        match self.composer.handle_key(key) {
            ComposerAction::Submit(text) => self.handle_submit(terminal, text).await?,
            ComposerAction::None => {}
        }
        Ok(())
    }

    async fn on_models_key(&mut self, terminal: &mut Term, key: KeyEvent) -> std::io::Result<()> {
        // Delete confirmation intercepts.
        if let Some(repo) = self.models.confirm_delete.clone() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    match hf::delete_model(&repo) {
                        Ok(()) => self.note(terminal, &format!("deleted {repo}"))?,
                        Err(e) => self.error(terminal, &e)?,
                    }
                    self.models.confirm_delete = None;
                    self.models.refresh();
                }
                _ => self.models.confirm_delete = None,
            }
            return Ok(());
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.mode = Mode::Chat,
            KeyCode::Up | KeyCode::Char('k') => {
                self.models.selected = self.models.selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.models.selected + 1 < self.models.items.len() {
                    self.models.selected += 1;
                }
            }
            KeyCode::Enter => {
                if let Some(repo) = self.models.selected_repo().map(str::to_string) {
                    let port = {
                        let mut s = self.core.settings.lock().unwrap();
                        s.model = repo.clone();
                        s.mlx_model = repo.clone();
                        s.base_url = format!("http://127.0.0.1:{}/v1", s.mlx_port);
                        let _ = config::save(&self.core.data_dir, &s);
                        s.mlx_port
                    };
                    match mlx::start(self.core.clone(), repo.clone(), port) {
                        Ok(()) => self.note(terminal, &format!("starting {repo} (first start downloads weights)"))?,
                        Err(e) => self.error(terminal, &e)?,
                    }
                    self.models.status = Some(mlx::status(&self.core).await);
                }
            }
            KeyCode::Char('d') => {
                if let Some(repo) = self.models.selected_repo().map(str::to_string) {
                    match hf::start_download(self.core.clone(), repo.clone()) {
                        Ok(()) => self.note(terminal, &format!("downloading {repo}"))?,
                        Err(e) => self.error(terminal, &e)?,
                    }
                }
            }
            KeyCode::Char('x') => {
                if let Some(repo) = self.models.selected_repo().map(str::to_string) {
                    if hf::is_installed(&repo) {
                        self.models.confirm_delete = Some(repo);
                    }
                }
            }
            KeyCode::Char('s') => {
                mlx::stop(&self.core);
                self.models.status = Some(mlx::status(&self.core).await);
                self.note(terminal, "model server stopped")?;
            }
            KeyCode::Char('u') => match mlx::setup(self.core.clone()) {
                Ok(()) => self.note(terminal, "setting up the MLX environment (watch /logs)")?,
                Err(e) => self.error(terminal, &e)?,
            },
            _ => {}
        }
        Ok(())
    }

    // ---------- submission and slash commands ----------

    async fn handle_submit(&mut self, terminal: &mut Term, text: String) -> std::io::Result<()> {
        if let Some(cmd) = text.strip_prefix('/') {
            return self.slash(terminal, cmd).await;
        }
        if self.running {
            self.note(terminal, "the agent is still working; esc cancels")?;
            return Ok(());
        }

        // Friendly gate: when pointed at the managed local server and it is
        // not serving yet, guide to /models instead of erroring.
        let (managed, ready) = {
            let s = self.core.settings.lock().unwrap();
            let managed = s.base_url.contains(&format!("127.0.0.1:{}", s.mlx_port));
            (managed, self.core.mlx.lock().unwrap().ready)
        };
        if managed && !ready {
            self.note(terminal, "no model is being served yet: open /models to set one up")?;
            return Ok(());
        }

        let session_id = match &self.session_id {
            Some(id) => id.clone(),
            None => {
                let meta = sessions::create(&self.core, self.project.display().to_string())
                    .map_err(std::io::Error::other)?;
                self.session_id = Some(meta.id.clone());
                meta.id
            }
        };

        self.insert_user_block(terminal, &text)?;
        match agent::start_turn(self.core.clone(), session_id, self.project.clone(), text) {
            Ok(()) => {
                self.running = true;
                self.turn_started = Some(Instant::now());
                self.first_token = None;
                self.stream_chars = 0;
                self.stream_text.clear();
                self.thinking_chars = 0;
                self.thinking_tail.clear();
            }
            Err(e) => self.error(terminal, &e)?,
        }
        Ok(())
    }

    async fn slash(&mut self, terminal: &mut Term, cmd: &str) -> std::io::Result<()> {
        let mut parts = cmd.split_whitespace();
        let head = parts.next().unwrap_or("");
        let rest: Vec<&str> = parts.collect();
        match head {
            "help" => {
                let lines = [
                    ("enter", "send · shift+enter or alt+enter for a newline"),
                    ("esc", "cancel the running turn"),
                    ("ctrl+o", "expand the last tool output"),
                    ("ctrl+t", "show or hide model thinking"),
                    ("ctrl+c ctrl+c", "quit (the model server keeps running)"),
                    ("/models", "manage and serve local models"),
                    ("/model <repo>", "use a specific model id"),
                    ("/approvals <auto|ask|readonly>", "how mutating tools are gated"),
                    ("/new", "start a fresh session"),
                    ("/status", "session and server state"),
                    ("/logs", "recent model server logs"),
                    ("/quit", "exit"),
                ];
                let block = lines
                    .iter()
                    .map(|(k, v)| {
                        Line::from(vec![
                            Span::styled(format!("  {k:<32}"), Style::default().fg(theme::ACCENT)),
                            Span::styled((*v).to_string(), Style::default().fg(theme::DIM)),
                        ])
                    })
                    .collect();
                insert_block(terminal, block)?;
            }
            "models" => {
                self.mode = Mode::Models;
                self.models.refresh();
                self.models.status = Some(mlx::status(&self.core).await);
                self.fetch_missing_sizes();
            }
            "model" => match rest.first() {
                Some(repo) => {
                    let repo = repo.to_string();
                    {
                        let mut s = self.core.settings.lock().unwrap();
                        s.model = repo.clone();
                        s.mlx_model = repo.clone();
                        let _ = config::save(&self.core.data_dir, &s);
                    }
                    self.note(terminal, &format!("model set to {repo} (serve it via /models if needed)"))?;
                }
                None => self.note(terminal, "usage: /model <huggingface repo id>")?,
            },
            "approvals" => match rest.first() {
                Some(&m @ ("auto" | "ask" | "readonly")) => {
                    {
                        let mut s = self.core.settings.lock().unwrap();
                        s.approval_mode = m.into();
                        let _ = config::save(&self.core.data_dir, &s);
                    }
                    self.note(terminal, &format!("approvals: {m}"))?;
                }
                _ => self.note(terminal, "usage: /approvals auto|ask|readonly")?,
            },
            "new" => {
                self.session_id = None;
                insert_block(
                    terminal,
                    vec![Line::from(Span::styled(
                        format!("── new session {}", "─".repeat(24)),
                        Style::default().fg(theme::DIM),
                    ))],
                )?;
            }
            "status" => {
                let s = self.core.settings.lock().unwrap().clone();
                let status = mlx::status(&self.core).await;
                let server = if status.server_ready {
                    format!("serving {} on :{}", status.model.as_deref().unwrap_or("?"), status.port)
                } else if status.server_running {
                    "starting".into()
                } else {
                    "stopped".into()
                };
                let ctx = self
                    .budget
                    .map(|(u, t)| format!("{}%", (u as f64 / t.max(1) as f64 * 100.0) as u32))
                    .unwrap_or_else(|| "0%".into());
                let block = vec![
                    kv("model", &s.model),
                    kv("endpoint", &s.base_url),
                    kv("server", &server),
                    kv("approvals", &s.approval_mode),
                    kv("context", &format!("{ctx} of {} tokens", s.context_tokens)),
                    kv("session", self.session_id.as_deref().unwrap_or("none yet")),
                    kv("project", &self.project.display().to_string()),
                    kv("data", &self.core.data_dir.display().to_string()),
                ];
                insert_block(terminal, block)?;
            }
            "logs" => {
                let logs = mlx::logs(&self.core);
                let tail: Vec<Line> = logs
                    .iter()
                    .rev()
                    .take(30)
                    .rev()
                    .map(|l| Line::from(Span::styled(format!("  {l}"), Style::default().fg(theme::DIM))))
                    .collect();
                if tail.is_empty() {
                    self.note(terminal, "no server logs yet")?;
                } else {
                    insert_block(terminal, tail)?;
                }
            }
            "quit" | "exit" => self.should_quit = true,
            other => self.note(terminal, &format!("unknown command: /{other} (see /help)"))?,
        }
        Ok(())
    }

    /// Fetch hub sizes for catalog entries that are not on disk yet.
    fn fetch_missing_sizes(&self) {
        for item in &self.models.items {
            if item.bytes.is_none() {
                let repo = item.repo.clone();
                let tx = self.hf_tx.clone();
                tokio::spawn(async move {
                    if let Ok(bytes) = hf::repo_total_bytes(&repo).await {
                        let _ = tx.send((repo, bytes));
                    }
                });
            }
        }
    }

    // ---------- core events ----------

    async fn on_core_event(&mut self, terminal: &mut Term, event: CoreEvent) -> std::io::Result<()> {
        match event {
            CoreEvent::Agent(env) => {
                if self.session_id.as_deref() != Some(env.session_id.as_str()) {
                    return Ok(());
                }
                self.on_agent_event(terminal, env.event)?;
            }
            CoreEvent::Mlx(ev) => self.on_mlx_event(terminal, ev).await?,
            CoreEvent::Download(ev) => match ev {
                DownloadEvent::Progress { repo, done_bytes, total_bytes } => {
                    self.models.download = Some((repo, done_bytes, total_bytes));
                }
                DownloadEvent::Done { repo, ok, message } => {
                    self.models.download = None;
                    self.models.refresh();
                    if ok {
                        self.note(terminal, &format!("{repo} is ready to serve"))?;
                    } else {
                        self.error(terminal, &message)?;
                    }
                }
            },
        }
        self.needs_redraw = true;
        Ok(())
    }

    fn on_agent_event(&mut self, terminal: &mut Term, event: AgentEvent) -> std::io::Result<()> {
        match event {
            AgentEvent::Token { text } => {
                self.first_token.get_or_insert_with(Instant::now);
                self.stream_chars += text.chars().count();
                self.stream_text.push_str(&text);
            }
            AgentEvent::Thinking { text } => {
                self.first_token.get_or_insert_with(Instant::now);
                self.stream_chars += text.chars().count();
                self.thinking_chars += text.chars().count();
                self.thinking_tail.push_str(&text);
                let overflow = self.thinking_tail.len().saturating_sub(600);
                if overflow > 0 {
                    let mut cut = overflow;
                    while !self.thinking_tail.is_char_boundary(cut) {
                        cut += 1;
                    }
                    self.thinking_tail.drain(..cut);
                }
            }
            AgentEvent::MessageDone { text } => {
                if !text.trim().is_empty() {
                    insert_block(terminal, markdown::render(&text, markdown::highlighter()))?;
                }
                self.stream_text.clear();
                self.thinking_tail.clear();
                self.thinking_chars = 0;
            }
            AgentEvent::Budget { used_tokens, context_tokens } => {
                self.budget = Some((used_tokens, context_tokens));
            }
            AgentEvent::ToolStart { call_id, name, args } => {
                let summary = tools::summarize_call(&name, &args);
                self.tool_meta.insert(call_id, (name.clone(), summary.clone()));
                self.running_tool = Some((name, summary));
            }
            AgentEvent::Diff { call_id, path, diff, added, removed } => {
                self.pending_diffs.insert(call_id, DiffText { path, diff, added, removed });
            }
            AgentEvent::ToolEnd { call_id, ok, output } => {
                let (name, summary) = self
                    .tool_meta
                    .remove(&call_id)
                    .unwrap_or_else(|| ("tool".into(), String::new()));
                let diff = self.pending_diffs.remove(&call_id);
                insert_block(terminal, tool_card::tool_block(&name, &summary, ok, &output, diff.as_ref()))?;
                self.last_tool_output = Some(output);
                self.running_tool = None;
            }
            AgentEvent::ApprovalRequest { approval_id, name, summary } => {
                self.pending_approval = Some((approval_id, name, summary));
            }
            AgentEvent::Done { stop_reason } => {
                self.running = false;
                self.running_tool = None;
                self.pending_approval = None;
                match stop_reason.as_str() {
                    "stop" | "tool_calls" => {}
                    "cancelled" => self.note(terminal, "cancelled")?,
                    "length" => self.note(terminal, "stopped: hit the response token limit")?,
                    "max_iterations" => self.note(terminal, "stopped: reached the tool-call limit for one turn (send a follow-up to continue)")?,
                    "error" => {}
                    other => self.note(terminal, &format!("stopped: {other}"))?,
                }
            }
            AgentEvent::Error { message } => {
                self.error(terminal, &message)?;
            }
        }
        Ok(())
    }

    async fn on_mlx_event(&mut self, terminal: &mut Term, event: MlxEvent) -> std::io::Result<()> {
        match event {
            MlxEvent::SetupDone { ok, message } => {
                if ok {
                    self.note(terminal, &message)?;
                } else {
                    self.error(terminal, &message)?;
                }
            }
            MlxEvent::ServerReady { model } => {
                self.note(terminal, &format!("{model} is serving"))?;
            }
            MlxEvent::ServerExit { code } => {
                self.error(terminal, &format!("model server exited with code {code} (see /logs)"))?;
            }
            MlxEvent::SetupLog { .. } | MlxEvent::ServerLog { .. } => {}
        }
        if self.mode == Mode::Models {
            self.models.status = Some(mlx::status(&self.core).await);
        }
        Ok(())
    }

    async fn on_tick(&mut self) {
        self.tick_i += 1;
        if self.running || self.models.download.is_some() {
            self.spinner_i = (self.spinner_i + 1) % SPINNER.len();
            self.needs_redraw = true;
        }
        // Refresh server status occasionally while the panel is open.
        if self.mode == Mode::Models && self.tick_i.is_multiple_of(16) {
            self.models.status = Some(mlx::status(&self.core).await);
            self.needs_redraw = true;
        }
    }

    // ---------- blocks ----------

    fn insert_user_block(&mut self, terminal: &mut Term, text: &str) -> std::io::Result<()> {
        let mut lines = Vec::new();
        for (i, l) in text.lines().enumerate() {
            let prefix = if i == 0 {
                Span::styled("❯ ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD))
            } else {
                Span::raw("  ")
            };
            lines.push(Line::from(vec![
                prefix,
                Span::styled(l.to_string(), Style::default().add_modifier(Modifier::BOLD)),
            ]));
        }
        insert_block(terminal, lines)
    }

    fn note(&mut self, terminal: &mut Term, text: &str) -> std::io::Result<()> {
        insert_block(
            terminal,
            vec![Line::from(Span::styled(
                text.to_string(),
                Style::default().fg(theme::DIM).add_modifier(Modifier::ITALIC),
            ))],
        )
    }

    fn error(&mut self, terminal: &mut Term, text: &str) -> std::io::Result<()> {
        let mut lines = Vec::new();
        for (i, l) in text.lines().enumerate() {
            let prefix = if i == 0 { "✗ " } else { "  " };
            lines.push(Line::from(Span::styled(
                format!("{prefix}{l}"),
                Style::default().fg(theme::ERR),
            )));
        }
        insert_block(terminal, lines)
    }

    // ---------- drawing ----------

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        if self.mode == Mode::Models {
            models::render(frame, area, &self.models);
            return;
        }

        // Clamp every band so the chrome never exceeds the viewport, even on
        // tiny terminals (rendering outside the buffer panics).
        let status_h = 1u16.min(area.height);
        let approval_h = if self.pending_approval.is_some() {
            1u16.min(area.height.saturating_sub(status_h))
        } else {
            0
        };
        let composer_h = self
            .composer
            .height()
            .min(area.height.saturating_sub(status_h + approval_h))
            .max(u16::from(area.height > status_h + approval_h));
        let chrome = status_h + approval_h + composer_h;
        let tail_h = area.height.saturating_sub(chrome);

        // Bottom-aligned stack: [tail][approval][composer][status].
        let mut y = area.y + tail_h;
        let tail_area = Rect { x: area.x, y: area.y, width: area.width, height: tail_h };
        let approval_area = Rect { x: area.x, y, width: area.width, height: approval_h };
        y += approval_h;
        let composer_area = Rect { x: area.x, y, width: area.width, height: composer_h };
        y += composer_h;
        let status_area = Rect { x: area.x, y, width: area.width, height: status_h };

        if tail_h > 0 && (self.running || !self.stream_text.is_empty()) {
            self.draw_tail(frame, tail_area);
        }
        if let Some((_, name, summary)) = &self.pending_approval {
            Paragraph::new(Line::from(vec![
                Span::styled("⚠ approve ", Style::default().fg(theme::WARN).add_modifier(Modifier::BOLD)),
                Span::styled(name.clone(), Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::styled(clip(summary, area.width.saturating_sub(40) as usize), Style::default()),
                Span::styled("  [y]es [n]o [a]lways", Style::default().fg(theme::DIM)),
            ]))
            .render(approval_area, frame.buffer_mut());
        }

        let (composer_lines, cx, cy) = self.composer.render(composer_h);
        Paragraph::new(composer_lines).render(composer_area, frame.buffer_mut());
        if self.pending_approval.is_none() {
            frame.set_cursor_position(Position::new(composer_area.x + cx, composer_area.y + cy));
        }

        self.draw_status(frame, status_area);
    }

    fn draw_tail(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();

        // Live text: thinking (optional) then content, most recent lines win.
        if self.show_thinking && !self.thinking_tail.is_empty() {
            let dim = Style::default().fg(theme::DIM).add_modifier(Modifier::ITALIC);
            for l in self.thinking_tail.lines() {
                lines.push(Line::from(Span::styled(l.to_string(), dim)));
            }
        }
        for l in self.stream_text.lines() {
            lines.push(Line::from(Span::raw(l.to_string())));
        }
        let mut wrapped = wrap_lines(lines, area.width);

        if let Some((name, summary)) = &self.running_tool {
            wrapped.push(tool_card::running_line(name, summary));
        }
        // Spinner meta line at the bottom of the tail.
        if self.running {
            let elapsed = self.turn_started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
            let toks = self.tok_per_sec();
            let mut meta = format!(" {}s", elapsed);
            if toks > 0.0 {
                meta.push_str(&format!(" · {toks:.0} tok/s"));
            }
            if self.thinking_chars > 0 && self.stream_text.is_empty() {
                meta.push_str(" · thinking (ctrl+t to peek)");
            }
            meta.push_str(" · esc to cancel");
            wrapped.push(Line::from(vec![
                Span::styled(SPINNER[self.spinner_i].to_string(), Style::default().fg(theme::ACCENT)),
                Span::styled(meta, Style::default().fg(theme::DIM)),
            ]));
        }

        let visible = area.height as usize;
        let first = wrapped.len().saturating_sub(visible);
        let shown: Vec<Line> = wrapped.into_iter().skip(first).collect();
        // Bottom-align within the tail area.
        let pad = area.height.saturating_sub(shown.len() as u16);
        let draw_area = Rect { x: area.x, y: area.y + pad, width: area.width, height: area.height - pad };
        Paragraph::new(shown).render(draw_area, frame.buffer_mut());
    }

    fn tok_per_sec(&self) -> f64 {
        match self.first_token {
            Some(t) => {
                let secs = t.elapsed().as_secs_f64();
                if secs < 0.5 {
                    0.0
                } else {
                    (self.stream_chars as f64 / 4.0) / secs
                }
            }
            None => 0.0,
        }
    }

    fn draw_status(&self, frame: &mut Frame, area: Rect) {
        let (model, approvals) = {
            let s = self.core.settings.lock().unwrap();
            (s.model.clone(), s.approval_mode.clone())
        };
        let ready = self.core.mlx.lock().unwrap().ready;
        let dot_color = if ready { theme::OK } else { theme::DIM };
        let ctx = self
            .budget
            .map(|(u, t)| format!(" · ctx {}%", (u as f64 / t.max(1) as f64 * 100.0) as u32))
            .unwrap_or_default();
        let short_model = model.rsplit('/').next().unwrap_or(&model).to_string();
        let right = if self.quit_armed { " · ctrl+c again to quit" } else { "" };
        let line = Line::from(vec![
            Span::styled("● ", Style::default().fg(dot_color)),
            Span::styled(short_model, Style::default().fg(theme::DIM)),
            Span::styled(format!("{ctx} · {approvals}{right}"), Style::default().fg(theme::DIM)),
        ]);
        Paragraph::new(line).render(area, frame.buffer_mut());
    }
}

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {k:<10}"), Style::default().fg(theme::ACCENT)),
        Span::raw(v.to_string()),
    ])
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max.max(8) {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max.max(8)).collect::<String>())
    }
}
