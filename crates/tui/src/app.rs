//! The event loop and all interaction logic. A fullscreen session on the
//! alternate screen: a pinned header at the top, the conversation anchored
//! to the bottom above the composer, and the shell restored intact on exit.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{
    Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
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
use crate::ui::transcript::{wrap_lines, StreamingWrap, Term, Transcript};
use crate::ui::{markdown, models};

const TICK: Duration = Duration::from_millis(120);
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const WHEEL_LINES: usize = 3;

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
    dir_label: String,
    session_id: Option<String>,
    mode: Mode,
    composer: Composer,
    models: models::ModelsState,
    transcript: Transcript,

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
    page_h: u16,

    hf_tx: mpsc::UnboundedSender<(String, u64)>,
    should_quit: bool,
    needs_redraw: bool,

    stream_wrap: StreamingWrap,
    thinking_wrapped: Vec<Line<'static>>,
    thinking_source: String,
    tail_width: u16,
    tail_content_len: usize,
    tail_stream_len: usize,
    tail_buf: Vec<Line<'static>>,
    chat_buf: Vec<Line<'static>>,
    status_model: String,
    status_approvals: String,
    status_ready: bool,
    status_budget: Option<(usize, usize)>,
    status_scrolled: bool,
    status_quit_armed: bool,
    status_line: Line<'static>,
}

pub async fn run(
    mut terminal: Term,
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<CoreEvent>,
    args: Args,
) -> std::io::Result<()> {
    let (hf_tx, mut hf_rx) = mpsc::unbounded_channel();
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let dir_label = project
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| project.display().to_string());
    let mut app = App {
        core: core.clone(),
        project,
        dir_label,
        session_id: None,
        mode: Mode::Chat,
        composer: Composer::new(&core.data_dir),
        models: models::ModelsState::empty(),
        transcript: Transcript::new(),
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
        page_h: 10,
        hf_tx,
        should_quit: false,
        needs_redraw: true,
        stream_wrap: StreamingWrap::default(),
        thinking_wrapped: Vec::new(),
        thinking_source: String::new(),
        tail_width: 0,
        tail_content_len: 0,
        tail_stream_len: 0,
        tail_buf: Vec::new(),
        chat_buf: Vec::new(),
        status_model: String::new(),
        status_approvals: String::new(),
        status_ready: false,
        status_budget: None,
        status_scrolled: false,
        status_quit_armed: false,
        status_line: Line::default(),
    };

    app.startup(&args).await;

    let mut term_events = crossterm::event::EventStream::new();
    let mut tick = tokio::time::interval(TICK);

    loop {
        tokio::select! {
            ev = term_events.next() => {
                match ev {
                    Some(Ok(e)) => app.on_term_event(e).await?,
                    Some(Err(_)) | None => app.should_quit = true,
                }
            }
            Some(ce) = core_rx.recv() => app.on_core_event(ce).await,
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
    async fn startup(&mut self, args: &Args) {
        // Adopt a still-running server from a previous launch.
        if mlx::reattach(&self.core).await {
            self.models.status = Some(mlx::status(&self.core).await);
        }

        if args.continue_session {
            let project = self.project.display().to_string();
            match sessions::latest(&self.core, &project) {
                Some(meta) => {
                    self.session_id = Some(meta.id.clone());
                    self.replay(&meta.id);
                }
                None => self.note("no previous session here; starting fresh"),
            }
        }
    }

    /// Re-render a persisted session compactly on --continue.
    fn replay(&mut self, session_id: &str) {
        let Some(messages) = sessions::load_messages(&self.core, session_id) else {
            return;
        };
        for m in &messages {
            match m.role.as_str() {
                "user" => {
                    if let Some(text) = &m.content {
                        self.insert_user_block(text);
                    }
                }
                "assistant" => {
                    if let Some(text) = &m.content {
                        if !text.trim().is_empty() {
                            self.transcript.push(markdown::render(text, markdown::highlighter()));
                        }
                    }
                    if let Some(calls) = &m.tool_calls {
                        for call in calls {
                            let args: serde_json::Value =
                                serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::Value::Null);
                            let summary = tools::summarize_call(&call.function.name, &args);
                            self.transcript.push(vec![Line::from(vec![
                                Span::styled("· ", Style::default().fg(theme::DIM)),
                                Span::styled(call.function.name.clone(), Style::default().fg(theme::ACCENT)),
                                Span::raw(" "),
                                Span::styled(summary, Style::default().fg(theme::DIM)),
                            ])]);
                        }
                    }
                }
                _ => {}
            }
        }
        self.note("continuing previous session");
    }

    // ---------- terminal events ----------

    async fn on_term_event(&mut self, event: TermEvent) -> std::io::Result<()> {
        match event {
            TermEvent::Key(key) if key.kind != KeyEventKind::Release => {
                self.on_key(key).await?;
                self.needs_redraw = true;
            }
            TermEvent::Paste(text) => {
                if self.mode == Mode::Chat && self.pending_approval.is_none() {
                    self.composer.insert_str(&text);
                    self.needs_redraw = true;
                }
            }
            TermEvent::Mouse(m) => {
                if self.mode == Mode::Chat {
                    match m.kind {
                        MouseEventKind::ScrollUp => {
                            self.transcript.scroll_up(WHEEL_LINES);
                            self.needs_redraw = true;
                        }
                        MouseEventKind::ScrollDown => {
                            self.transcript.scroll_down(WHEEL_LINES);
                            self.needs_redraw = true;
                        }
                        _ => {}
                    }
                }
            }
            TermEvent::Resize(_, _) => self.needs_redraw = true,
            _ => {}
        }
        Ok(())
    }

    async fn on_key(&mut self, key: KeyEvent) -> std::io::Result<()> {
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
                self.transcript.push(lines);
            }
            return Ok(());
        }
        if ctrl && key.code == KeyCode::Char('t') {
            self.show_thinking = !self.show_thinking;
            return Ok(());
        }

        if self.mode == Mode::Models {
            self.on_models_key(key).await;
            return Ok(());
        }

        // Transcript scrolling.
        match key.code {
            KeyCode::PageUp => {
                self.transcript.scroll_up(self.page_h.max(1) as usize);
                return Ok(());
            }
            KeyCode::PageDown => {
                self.transcript.scroll_down(self.page_h.max(1) as usize);
                return Ok(());
            }
            _ => {}
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
                    self.note("approvals set to auto for this run (change with /approvals)");
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
            } else if self.transcript.offset() > 0 {
                self.transcript.follow();
            }
            return Ok(());
        }

        match self.composer.handle_key(key) {
            ComposerAction::Submit(text) => self.handle_submit(text).await?,
            ComposerAction::None => {}
        }
        Ok(())
    }

    async fn on_models_key(&mut self, key: KeyEvent) {
        // Delete confirmation intercepts.
        if let Some(repo) = self.models.confirm_delete.clone() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    match hf::delete_model(&repo) {
                        Ok(()) => self.note(&format!("deleted {repo}")),
                        Err(e) => self.error(&e),
                    }
                    self.models.confirm_delete = None;
                    self.models.refresh();
                }
                _ => self.models.confirm_delete = None,
            }
            return;
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
                if self.models.download.is_some() {
                    self.note("download in progress");
                    return;
                }
                let Some(item) = self.models.selected_item().cloned() else {
                    return;
                };
                if item.installed {
                    self.serve_selected_model(item.repo).await;
                } else {
                    self.begin_model_download(&item);
                }
            }
            KeyCode::Char('d') => {
                if self.models.download.is_some() {
                    self.note("download in progress");
                    return;
                }
                if let Some(item) = self.models.selected_item().cloned() {
                    self.begin_model_download(&item);
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
                self.note("model server stopped");
            }
            KeyCode::Char('u') => match mlx::setup(self.core.clone()) {
                Ok(()) => self.note("setting up the MLX environment (watch /logs)"),
                Err(e) => self.error(&e),
            },
            _ => {}
        }
    }

    fn begin_model_download(&mut self, item: &models::ModelItem) {
        let total = item.bytes.unwrap_or(0);
        match hf::start_download(self.core.clone(), item.repo.clone()) {
            Ok(()) => {
                self.models.download = Some((item.repo.clone(), 0, total));
                self.models.footer = None;
            }
            Err(e) => self.error(&e),
        }
    }

    async fn serve_selected_model(&mut self, repo: String) {
        let port = {
            let mut s = self.core.settings.lock().unwrap();
            s.model = repo.clone();
            s.mlx_model = repo.clone();
            s.base_url = format!("http://127.0.0.1:{}/v1", s.mlx_port);
            let _ = config::save(&self.core.data_dir, &s);
            s.mlx_port
        };
        match mlx::start(self.core.clone(), repo.clone(), port) {
            Ok(()) => self.note(&format!("starting {repo}")),
            Err(e) => self.error(&e),
        }
        self.models.status = Some(mlx::status(&self.core).await);
    }

    // ---------- submission and slash commands ----------

    async fn handle_submit(&mut self, text: String) -> std::io::Result<()> {
        if let Some(cmd) = text.strip_prefix('/') {
            self.slash(cmd).await;
            return Ok(());
        }
        if self.running {
            self.note("the agent is still working; esc cancels");
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
            self.note("no model is being served yet: open /models to set one up");
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

        self.insert_user_block(&text);
        self.transcript.follow();
        match agent::start_turn(self.core.clone(), session_id, self.project.clone(), text) {
            Ok(()) => {
                self.running = true;
                self.turn_started = Some(Instant::now());
                self.first_token = None;
                self.stream_chars = 0;
                self.stream_text.clear();
                self.stream_wrap.clear();
                self.tail_stream_len = 0;
                self.thinking_chars = 0;
                self.thinking_tail.clear();
                self.thinking_source.clear();
                self.thinking_wrapped.clear();
            }
            Err(e) => self.error(&e),
        }
        Ok(())
    }

    async fn slash(&mut self, cmd: &str) {
        let mut parts = cmd.split_whitespace();
        let head = parts.next().unwrap_or("");
        let rest: Vec<&str> = parts.collect();
        match head {
            "help" => {
                let lines = [
                    ("enter", "send · shift+enter or alt+enter for a newline"),
                    ("esc", "cancel the running turn, or jump to the latest output"),
                    ("wheel · pgup/pgdn", "scroll the conversation"),
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
                self.transcript.push(block);
            }
            "models" => {
                self.mode = Mode::Models;
                self.models.ensure_loaded(ram_bytes());
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
                    self.note(&format!("model set to {repo} (serve it via /models if needed)"));
                }
                None => self.note("usage: /model <huggingface repo id>"),
            },
            "approvals" => match rest.first() {
                Some(&m @ ("auto" | "ask" | "readonly")) => {
                    {
                        let mut s = self.core.settings.lock().unwrap();
                        s.approval_mode = m.into();
                        let _ = config::save(&self.core.data_dir, &s);
                    }
                    self.note(&format!("approvals: {m}"));
                }
                _ => self.note("usage: /approvals auto|ask|readonly"),
            },
            "new" => {
                self.session_id = None;
                self.transcript.push(vec![Line::from(Span::styled(
                    format!("── new session {}", "─".repeat(24)),
                    Style::default().fg(theme::DIM),
                ))]);
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
                self.transcript.push(block);
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
                    self.note("no server logs yet");
                } else {
                    self.transcript.push(tail);
                }
            }
            "quit" | "exit" => self.should_quit = true,
            other => self.note(&format!("unknown command: /{other} (see /help)")),
        }
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

    async fn on_core_event(&mut self, event: CoreEvent) {
        match event {
            CoreEvent::Agent(env) => {
                if self.session_id.as_deref() != Some(env.session_id.as_str()) {
                    return;
                }
                self.on_agent_event(env.event);
            }
            CoreEvent::Mlx(ev) => self.on_mlx_event(ev).await,
            CoreEvent::Download(ev) => match ev {
                DownloadEvent::Progress { repo, done_bytes, total_bytes } => {
                    self.models.download = Some((repo, done_bytes, total_bytes));
                }
                DownloadEvent::Done { ok, message, .. } => {
                    self.models.download = None;
                    self.models.refresh();
                    if ok {
                        self.note(&message);
                    } else {
                        self.error(&message);
                    }
                }
            },
        }
        self.needs_redraw = true;
    }

    fn on_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Token { text } => {
                self.first_token.get_or_insert_with(Instant::now);
                self.stream_chars += text.len();
                self.stream_text.push_str(&text);
            }
            AgentEvent::Thinking { text } => {
                self.first_token.get_or_insert_with(Instant::now);
                self.stream_chars += text.len();
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
                    self.transcript.push(markdown::render(&text, markdown::highlighter()));
                }
                self.stream_text.clear();
                self.stream_wrap.clear();
                self.tail_stream_len = 0;
                self.thinking_tail.clear();
                self.thinking_source.clear();
                self.thinking_wrapped.clear();
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
                self.transcript
                    .push(tool_card::tool_block(&name, &summary, ok, &output, diff.as_ref()));
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
                    "cancelled" => self.note("cancelled"),
                    "length" => self.note("stopped: hit the response token limit"),
                    "max_iterations" => self.note("stopped: reached the tool-call limit for one turn (send a follow-up to continue)"),
                    "error" => {}
                    other => self.note(&format!("stopped: {other}")),
                }
            }
            AgentEvent::Error { message } => {
                self.error(&message);
            }
        }
    }

    async fn on_mlx_event(&mut self, event: MlxEvent) {
        match event {
            MlxEvent::SetupDone { ok, message } => {
                if ok {
                    self.note(&message);
                } else {
                    self.error(&message);
                }
            }
            MlxEvent::ServerReady { model } => {
                self.note(&format!("{model} is serving"));
            }
            MlxEvent::ServerExit { code } => {
                self.error(&format!("model server exited with code {code} (see /logs)"));
            }
            MlxEvent::SetupLog { .. } | MlxEvent::ServerLog { .. } => {}
        }
        if self.mode == Mode::Models {
            self.models.status = Some(mlx::status(&self.core).await);
        }
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

    fn insert_user_block(&mut self, text: &str) {
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
        self.transcript.push(lines);
    }

    fn note(&mut self, text: &str) {
        if self.mode == Mode::Models {
            self.models.footer = Some((text.to_string(), false));
        } else {
            self.transcript.push(vec![Line::from(Span::styled(
                text.to_string(),
                Style::default().fg(theme::DIM).add_modifier(Modifier::ITALIC),
            ))]);
        }
    }

    fn error(&mut self, text: &str) {
        if self.mode == Mode::Models {
            self.models.footer = Some((text.to_string(), true));
        } else {
            let mut lines = Vec::new();
            for (i, l) in text.lines().enumerate() {
                let prefix = if i == 0 { "✗ " } else { "  " };
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{l}"),
                    Style::default().fg(theme::ERR),
                )));
            }
            self.transcript.push(lines);
        }
    }

    // ---------- drawing ----------

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        if self.mode == Mode::Models {
            models::render(frame, area, &self.models);
            return;
        }

        // Clamp every band so the chrome never exceeds the screen, even on
        // tiny terminals (rendering outside the buffer panics).
        let status_h = 1u16.min(area.height);
        let header_h = 2u16.min(area.height.saturating_sub(status_h + 2));
        let approval_h = if self.pending_approval.is_some() {
            1u16.min(area.height.saturating_sub(header_h + status_h))
        } else {
            0
        };
        let composer_h = self
            .composer
            .height()
            .min(area.height.saturating_sub(header_h + status_h + approval_h))
            .max(u16::from(area.height > header_h + status_h + approval_h));
        let chrome = header_h + approval_h + composer_h + status_h;
        let chat_h = area.height.saturating_sub(chrome);
        self.page_h = chat_h.saturating_sub(1).max(1);

        // Top to bottom: [header][chat][approval][composer][status].
        let header_area = Rect { x: area.x, y: area.y, width: area.width, height: header_h };
        let chat_area = Rect { x: area.x, y: area.y + header_h, width: area.width, height: chat_h };
        let mut y = area.y + header_h + chat_h;
        let approval_area = Rect { x: area.x, y, width: area.width, height: approval_h };
        y += approval_h;
        let composer_area = Rect { x: area.x, y, width: area.width, height: composer_h };
        y += composer_h;
        let status_area = Rect { x: area.x, y, width: area.width, height: status_h };

        if header_h > 0 {
            self.draw_header(frame, header_area);
        }
        if chat_h > 0 {
            self.draw_chat(frame, chat_area);
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

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let version = env!("CARGO_PKG_VERSION");
        let line = Line::from(vec![
            Span::styled("◆ open max", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(
                " · the performance harness",
                Style::default().fg(theme::DIM).add_modifier(Modifier::ITALIC),
            ),
            Span::styled(
                format!("  v{version} · {} · /help", self.dir_label),
                Style::default().fg(theme::DIM),
            ),
        ]);
        // Second header row stays blank as breathing room above the chat.
        Paragraph::new(line).render(area, frame.buffer_mut());
    }

    /// Finished transcript plus the live tail, bottom anchored, honoring the
    /// scroll offset (0 follows the latest output).
    fn draw_chat(&mut self, frame: &mut Frame, area: Rect) {
        self.transcript.set_width(area.width);
        let tail_len = self.rebuild_tail(area.width);

        let total = self.transcript.len() + tail_len;
        let visible = area.height as usize;
        self.transcript.clamp_offset(total.saturating_sub(visible));
        let offset = self.transcript.offset();

        let end = total - offset;
        let start = end.saturating_sub(visible);
        self.chat_buf.clear();
        self.chat_buf.reserve(end.saturating_sub(start));
        for i in start..end {
            if i < self.transcript.len() {
                self.chat_buf.push(self.transcript.lines()[i].clone());
            } else {
                self.chat_buf
                    .push(self.tail_buf[i - self.transcript.len()].clone());
            }
        }

        // Bottom-align so the conversation grows upward from the composer.
        let pad = area.height.saturating_sub(self.chat_buf.len() as u16);
        let draw_area = Rect { x: area.x, y: area.y + pad, width: area.width, height: area.height - pad };
        Paragraph::new(self.chat_buf.as_slice()).render(draw_area, frame.buffer_mut());
    }

    /// Rebuild the live tail into `tail_buf`, reusing cached stream/thinking
    /// wraps when only the spinner meta line changes between ticks.
    fn rebuild_tail(&mut self, width: u16) -> usize {
        let width_changed = width != self.tail_width;
        if width_changed {
            self.tail_width = width;
            self.thinking_source.clear();
        }
        self.stream_wrap.update(&self.stream_text, width);

        let mut thinking_changed = false;
        if self.show_thinking && !self.thinking_tail.is_empty() {
            if self.thinking_tail != self.thinking_source {
                thinking_changed = true;
                self.thinking_source = self.thinking_tail.clone();
                let dim = Style::default().fg(theme::DIM).add_modifier(Modifier::ITALIC);
                let raw: Vec<Line<'static>> = self
                    .thinking_tail
                    .lines()
                    .map(|l| Line::from(Span::styled(l.to_string(), dim)))
                    .collect();
                self.thinking_wrapped = wrap_lines(&raw, width);
            }
        } else if !self.thinking_wrapped.is_empty() || !self.thinking_source.is_empty() {
            thinking_changed = true;
            self.thinking_wrapped.clear();
            self.thinking_source.clear();
        }

        let content_changed = width_changed
            || self.stream_text.len() != self.tail_stream_len
            || thinking_changed;

        if content_changed {
            self.tail_stream_len = self.stream_text.len();
            self.tail_buf.clear();
            self.tail_buf
                .extend(self.thinking_wrapped.iter().cloned());
            self.tail_buf
                .extend(self.stream_wrap.lines().cloned());
            self.tail_content_len = self.tail_buf.len();
        } else {
            self.tail_buf.truncate(self.tail_content_len);
        }

        if let Some((name, summary)) = &self.running_tool {
            self.tail_buf.push(tool_card::running_line(name, summary));
        }
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
            self.tail_buf.push(Line::from(vec![
                Span::styled(SPINNER[self.spinner_i].to_string(), Style::default().fg(theme::ACCENT)),
                Span::styled(meta, Style::default().fg(theme::DIM)),
            ]));
        }
        self.tail_buf.len()
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

    fn draw_status(&mut self, frame: &mut Frame, area: Rect) {
        let ready = self.core.mlx.lock().unwrap().ready;
        let (model, approvals) = {
            let s = self.core.settings.lock().unwrap();
            (s.model.clone(), s.approval_mode.clone())
        };
        let scrolled = self.transcript.offset() > 0;
        let needs_rebuild = model != self.status_model
            || approvals != self.status_approvals
            || ready != self.status_ready
            || self.status_budget != self.budget
            || self.status_scrolled != scrolled
            || self.status_quit_armed != self.quit_armed;

        if needs_rebuild {
            self.status_model = model;
            self.status_approvals = approvals;
            self.status_ready = ready;
            self.status_budget = self.budget;
            self.status_scrolled = scrolled;
            self.status_quit_armed = self.quit_armed;

            let dot_color = if ready { theme::OK } else { theme::DIM };
            let ctx = self
                .budget
                .map(|(u, t)| format!(" · ctx {}%", (u as f64 / t.max(1) as f64 * 100.0) as u32))
                .unwrap_or_default();
            let short_model = self
                .status_model
                .rsplit('/')
                .next()
                .unwrap_or(&self.status_model)
                .to_string();
            let scrolled_suffix = if scrolled { " · ↑ scrolled (esc to follow)" } else { "" };
            let right = if self.quit_armed { " · ctrl+c again to quit" } else { "" };
            self.status_line = Line::from(vec![
                Span::styled("● ", Style::default().fg(dot_color)),
                Span::styled(short_model, Style::default().fg(theme::DIM)),
                Span::styled(
                    format!("{ctx} · {}{scrolled_suffix}{right}", self.status_approvals),
                    Style::default().fg(theme::DIM),
                ),
            ]);
        }
        Paragraph::new(self.status_line.clone()).render(area, frame.buffer_mut());
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
