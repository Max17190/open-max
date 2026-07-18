//! The event loop and all interaction logic. A fullscreen session on the
//! alternate screen: a pinned header at the top, the conversation anchored
//! to the bottom above the composer, and the shell restored intact on exit.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{
    Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton,
    MouseEventKind,
};
use futures_util::StreamExt;
use open_max_core::mlx::MlxEvent;
use open_max_core::state::{Core, CoreEvent, DownloadEvent};
use open_max_core::types::AgentEvent;
use open_max_core::{agent, config, hf, mlx, prompt, registry, sessions};
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};
use ratatui::Frame;
use tokio::sync::mpsc;

use crate::clipboard;
use crate::completion;
use crate::input::{Composer, ComposerAction};
use crate::theme;
use crate::ui::sessions as sessions_ui;
use crate::ui::tool_card::{self, DiffText};
use crate::ui::transcript::{
    filter_matching_indices, wrap_lines, Term, Transcript,
};
use crate::ui::{context, extensions, markdown, model_picker, models};

/// Where keyboard focus lives in chat mode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Composer,
    Scrollback,
}

const TICK: Duration = Duration::from_millis(120);
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const WHEEL_LINES: usize = 3;
/// Paint-rate cap: coalesce redraw triggers into at most ~60 frames/s.
const MIN_DRAW_INTERVAL: Duration = Duration::from_millis(16);
/// A resize storm settles for this long before the transcript rewraps.
const RESIZE_DEBOUNCE: Duration = Duration::from_millis(16);
/// Core events drained per wake before painting once for the whole batch.
const CORE_DRAIN_MAX: usize = 32;

/// Fine-grained redraw reasons so spinner ticks can skip history rebuilds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Dirty {
    /// Finished transcript (blocks, scroll, selection, fold).
    chat: bool,
    /// Live stream / thinking / running-tool / spinner meta.
    tail: bool,
    /// Header, composer, status, popups, approval, models/sessions chrome.
    chrome: bool,
    /// Mouse text selection overlay only. It never invalidates cached lines.
    selection: bool,
}

impl Dirty {
    fn all() -> Self {
        Self {
            chat: true,
            tail: true,
            chrome: true,
            selection: true,
        }
    }

    fn any(self) -> bool {
        self.chat || self.tail || self.chrome || self.selection
    }

    fn mark_chat(&mut self) {
        self.chat = true;
        self.tail = true;
    }

    fn mark_tail(&mut self) {
        self.tail = true;
    }

    fn mark_chrome(&mut self) {
        self.chrome = true;
    }

    fn mark_selection(&mut self) {
        self.selection = true;
        self.chrome = true;
    }

    fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Viewport fingerprint for reusing the history portion of `chat_buf`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HistReuseKey {
    hist_len: usize,
    start: usize,
    hist_view_end: usize,
    sticky: bool,
    focus_scroll: bool,
    selected: Option<usize>,
    width: u16,
}

pub struct Args {
    pub continue_session: bool,
}

#[derive(PartialEq)]
enum Mode {
    Chat,
    ModelPicker,
    Models,
    Sessions,
}

struct ToolMeta {
    name: String,
    summary: String,
    started: Instant,
}

pub struct App {
    core: Arc<Core>,
    project: PathBuf,
    dir_label: String,
    session_id: Option<String>,
    mode: Mode,
    composer: Composer,
    models: models::ModelsState,
    model_picker: Option<model_picker::ModelPickerState>,
    sessions_panel: Option<sessions_ui::SessionsState>,
    transcript: Transcript,
    focus: Focus,
    completion: Option<completion::Popup>,
    /// Ctrl+R history search: filter text + selected index into matches.
    history_search: Option<(String, usize, Vec<String>)>,
    /// Ctrl+F scrollback find: query + selected match index + matching block indices.
    scroll_search: Option<(String, usize, Vec<usize>)>,
    /// Last find query so n/N can step matches after the popup closes.
    scroll_search_last: Option<String>,
    /// Project files for @-mentions; rescanned when a fresh `@` opens.
    file_index: Option<Arc<Vec<String>>>,
    file_index_pending: bool,
    /// Prompt templates as (name, description); rescanned when a fresh `/` opens.
    templates: Vec<(String, String)>,
    /// Messages typed while the agent works, sent in order after the turn.
    queued: Vec<String>,
    flush_queue: bool,
    /// Text of the in-flight user submit, kept so a `user_prompt_submit`
    /// block can restore the composer and drop the optimistic UI bubble.
    pending_submit: Option<String>,

    running: bool,
    stream_text: String,
    thinking_chars: usize,
    thinking_tail: String,
    show_thinking: bool,
    turn_started: Option<Instant>,
    first_token: Option<Instant>,
    stream_chars: usize,
    running_tool: Option<(String, String)>,
    /// Pending mutating-tool gate: id, tool name, summary, detail preview.
    pending_approval: Option<(String, String, String, String)>,
    pending_diffs: HashMap<String, DiffText>,
    tool_meta: HashMap<String, ToolMeta>,
    last_tool_output: Option<String>,
    last_assistant_response: Option<String>,
    budget: Option<(usize, usize)>,
    /// Prompt-cache hit rate of the last completion, from server usage.
    cache_pct: Option<u8>,
    quit_armed: bool,
    spinner_i: usize,
    tick_i: u64,
    page_h: u16,

    hf_tx: mpsc::UnboundedSender<(String, u64)>,
    files_tx: mpsc::UnboundedSender<Vec<String>>,
    should_quit: bool,
    dirty: Dirty,

    /// Live assistant stream, markdown-rendered and wrapped (matches final block).
    stream_wrapped: Vec<Line<'static>>,
    /// Incremental markdown highlighter for the live stream: completed lines are
    /// highlighted once, only the growing tail line re-renders per token, and a
    /// resize re-wraps without re-highlighting. Replaces the O(n)-per-refresh
    /// full re-render that scaled poorly on long code replies.
    stream_md: markdown::StreamingMarkdown,
    thinking_wrapped: Vec<Line<'static>>,
    thinking_source: String,
    tail_width: u16,
    tail_content_len: usize,
    tail_buf: Vec<Line<'static>>,
    chat_buf: Vec<Line<'static>>,
    /// Lines in `chat_buf` that are sticky + history (before live tail).
    hist_prefix_len: usize,
    hist_reuse_key: Option<HistReuseKey>,
    /// Absolute transcript line for each rendered row in `chat_buf`.
    chat_line_map: Vec<Option<usize>>,
    chat_draw_area: Rect,
    approval_hits: [Option<Rect>; 3],
    perf_layout_ms: f64,
    perf_selection_ms: f64,
    status_line: Line<'static>,
}

pub async fn run(
    mut terminal: Term,
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<CoreEvent>,
    args: Args,
) -> std::io::Result<()> {
    let (hf_tx, mut hf_rx) = mpsc::unbounded_channel();
    let (files_tx, mut files_rx) = mpsc::unbounded_channel();
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let dir_label = project
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| project.display().to_string());
    let mut app = App::new(core.clone(), project, dir_label, hf_tx, files_tx);

    app.startup(&args).await;

    // Terminal events are forwarded through a channel so the core-event arm
    // can be gated on `input_rx.is_empty()` — a token firehose must never
    // starve a keypress (crossterm's EventStream itself is not peekable).
    let (input_tx, mut input_rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut term_events = crossterm::event::EventStream::new();
        while let Some(ev) = term_events.next().await {
            let Ok(e) = ev else { break };
            if input_tx.send(e).is_err() {
                break;
            }
        }
        // Dropping input_tx closes the channel; the loop reads that as quit.
    });

    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Paint pacing: at most one frame per MIN_DRAW_INTERVAL. A redraw that
    // arrives too early is deferred to `draw_deadline` and coalesced with
    // everything else that lands before it (grok-build's cadence model).
    let mut last_draw = Instant::now() - MIN_DRAW_INTERVAL;
    let mut draw_deadline: Option<Instant> = None;

    loop {
        tokio::select! {
            biased;
            // Streaming sits above input but is gated on the input queue
            // being empty: input-first would let held keys starve redraws,
            // while the gate keeps cancel/quit ahead of the firehose.
            Some(ce) = core_rx.recv(), if input_rx.is_empty() => {
                app.on_core_event(ce).await;
                for _ in 1..CORE_DRAIN_MAX {
                    if !input_rx.is_empty() {
                        break;
                    }
                    match core_rx.try_recv() {
                        Ok(ce) => app.on_core_event(ce).await,
                        Err(_) => break,
                    }
                }
            }
            ev = input_rx.recv() => {
                match ev {
                    Some(TermEvent::Resize(_, _)) => {
                        // Terminals emit resize storms mid-drag; rewrapping
                        // the transcript on each one is wasted layout work.
                        app.dirty = Dirty::all();
                        draw_deadline = Some(Instant::now() + RESIZE_DEBOUNCE);
                    }
                    Some(e) => app.on_term_event(e).await?,
                    None => app.should_quit = true,
                }
            }
            Some((repo, bytes)) = hf_rx.recv() => {
                app.models.set_remote_size(&repo, bytes);
                app.dirty.mark_chrome();
            }
            Some(files) = files_rx.recv() => {
                app.file_index = Some(Arc::new(files));
                app.file_index_pending = false;
                app.sync_completion();
                app.dirty.mark_chrome();
            }
            _ = tick.tick(), if app.tick_armed() => app.on_tick().await,
            _ = tokio::time::sleep_until(
                draw_deadline.unwrap_or_else(Instant::now).into()
            ), if draw_deadline.is_some() => {}
        }
        if app.should_quit {
            break;
        }
        if app.dirty.any() {
            let now = Instant::now();
            let deferred = draw_deadline.is_some_and(|d| now < d);
            if !deferred && now.duration_since(last_draw) >= MIN_DRAW_INTERVAL {
                draw_frame(&mut terminal, &mut app)?;
                last_draw = now;
                draw_deadline = None;
                app.dirty.clear();
            } else if draw_deadline.is_none() {
                draw_deadline = Some(last_draw + MIN_DRAW_INTERVAL);
            }
        }
    }
    Ok(())
}

/// One frame, wrapped in a synchronized update so the terminal applies it
/// atomically — no half-painted frames under tmux or slow connections.
fn draw_frame(terminal: &mut Term, app: &mut App) -> std::io::Result<()> {
    use std::io::Write;
    let t0 = Instant::now();
    crossterm::queue!(terminal.backend_mut(), crossterm::terminal::BeginSynchronizedUpdate)?;
    terminal.draw(|f| app.draw(f))?;
    crossterm::queue!(terminal.backend_mut(), crossterm::terminal::EndSynchronizedUpdate)?;
    terminal.backend_mut().flush()?;
    if std::env::var_os("OPENMAX_PERF").is_some() {
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "openmax_perf draw_frame_ms={ms:.3} transcript_layout_ms={:.3} selection_overlay_ms={:.3}",
            app.perf_layout_ms, app.perf_selection_ms
        );
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
    fn new(
        core: Arc<Core>,
        project: PathBuf,
        dir_label: String,
        hf_tx: mpsc::UnboundedSender<(String, u64)>,
        files_tx: mpsc::UnboundedSender<Vec<String>>,
    ) -> Self {
        Self {
            composer: Composer::new(&core.data_dir),
            core,
            project,
            dir_label,
            session_id: None,
            pending_submit: None,
            mode: Mode::Chat,
            models: models::ModelsState::empty(),
            model_picker: None,
            sessions_panel: None,
            transcript: Transcript::new(),
            focus: Focus::Composer,
            completion: None,
            history_search: None,
            scroll_search: None,
            scroll_search_last: None,
            file_index: None,
            file_index_pending: false,
            templates: Vec::new(),
            queued: Vec::new(),
            flush_queue: false,
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
            last_assistant_response: None,
            budget: None,
            cache_pct: None,
            quit_armed: false,
            spinner_i: 0,
            tick_i: 0,
            page_h: 10,
            hf_tx,
            files_tx,
            should_quit: false,
            dirty: Dirty::all(),
            stream_wrapped: Vec::new(),
            stream_md: markdown::StreamingMarkdown::default(),
            thinking_wrapped: Vec::new(),
            thinking_source: String::new(),
            tail_width: 0,
            tail_content_len: 0,
            tail_buf: Vec::new(),
            chat_buf: Vec::new(),
            hist_prefix_len: 0,
            hist_reuse_key: None,
            chat_line_map: Vec::new(),
            chat_draw_area: Rect::default(),
            approval_hits: [None; 3],
            perf_layout_ms: 0.0,
            perf_selection_ms: 0.0,
            status_line: Line::default(),
        }
    }

    async fn startup(&mut self, args: &Args) {
        // Adopt a still-running server from a previous launch — in the
        // background: reattach spawns `ps` and probes HTTP with a 2s timeout,
        // and the first frame must never wait on that. ServerReady flips the
        // status dot when it resolves.
        let core = self.core.clone();
        tokio::spawn(async move {
            mlx::reattach(&core).await;
        });

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
        for (i, m) in messages.iter().enumerate() {
            match m.role.as_str() {
                "user" => {
                    if let Some(text) = &m.content {
                        self.insert_user_block(text);
                    }
                }
                "assistant" => {
                    if let Some(text) = &m.content {
                        if !text.trim().is_empty() {
                            self.last_assistant_response = Some(text.clone());
                            self.transcript.push_assistant(markdown::render(
                                text,
                                markdown::highlighter(),
                            ));
                        }
                    }
                    if let Some(calls) = &m.tool_calls {
                        // Tool results directly follow their assistant message;
                        // stop at the first non-tool message so a short turn
                        // (e.g. cancelled) never borrows a later turn's output.
                        let tool_results: Vec<_> = messages[i + 1..]
                            .iter()
                            .take_while(|tm| tm.role == "tool")
                            .take(calls.len())
                            .collect();
                        for (call, tool_msg) in calls.iter().zip(tool_results) {
                            let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
                                .unwrap_or(serde_json::Value::Null);
                            let summary = registry::summarize_call(&call.function.name, &args);
                            let content = tool_msg.content.as_deref().unwrap_or("");
                            let ok = !content.starts_with("Error:");
                            let compact = tool_card::tool_block(
                                &call.function.name,
                                &summary,
                                ok,
                                &truncate_replay_output(content),
                                None,
                            );
                            self.transcript.push_tool(compact, content.to_string());
                            self.last_tool_output = Some(content.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        self.note("continuing previous session");
    }

    /// Clear transcript and per-session UI state for `/new`.
    fn reset_for_new_session(&mut self) {
        if self.running {
            if let Some(id) = &self.session_id {
                self.core.cancel(id);
            }
        }
        self.session_id = None;
        self.transcript = Transcript::new();
        self.running = false;
        self.stream_text.clear();
        self.thinking_chars = 0;
        self.thinking_tail.clear();
        self.turn_started = None;
        self.first_token = None;
        self.stream_chars = 0;
        self.running_tool = None;
        self.pending_approval = None;
        self.model_picker = None;
        self.pending_diffs.clear();
        self.tool_meta.clear();
        self.last_tool_output = None;
        self.last_assistant_response = None;
        self.budget = None;
        self.cache_pct = None;
        self.completion = None;
        self.history_search = None;
        self.scroll_search = None;
        self.scroll_search_last = None;
        self.focus = Focus::Composer;
        self.queued.clear();
        self.flush_queue = false;
        self.stream_wrapped.clear();
        self.stream_md.clear();
        self.thinking_wrapped.clear();
        self.thinking_source.clear();
        self.tail_width = 0;
        self.tail_content_len = 0;
        self.tail_buf.clear();
        self.hist_prefix_len = 0;
        self.hist_reuse_key = None;
        self.chat_line_map.clear();
        self.chat_draw_area = Rect::default();
        self.approval_hits = [None; 3];
        self.transcript.follow();
        self.dirty.mark_chat();
        self.dirty.mark_chrome();
    }

    // ---------- terminal events ----------

    async fn on_term_event(&mut self, event: TermEvent) -> std::io::Result<()> {
        match event {
            TermEvent::Key(key) if key.kind != KeyEventKind::Release => {
                self.on_key(key).await?;
                // Keys can mutate many regions; mark specifically in handlers
                // when possible, otherwise fall back to a full redraw.
                if !self.dirty.any() {
                    self.dirty = Dirty::all();
                }
            }
            TermEvent::Paste(text) => {
                if self.mode == Mode::Chat && self.pending_approval.is_none() {
                    self.composer.insert_str(&text);
                    self.sync_completion();
                    self.dirty.mark_chrome();
                }
            }
            TermEvent::Mouse(m) => {
                if self.mode == Mode::Chat {
                    match m.kind {
                        MouseEventKind::Down(MouseButton::Left) => {
                            if let Some(choice) = self
                                .approval_hits
                                .iter()
                                .position(|hit| hit.is_some_and(|rect| rect_contains(rect, m.column, m.row)))
                            {
                                self.respond_approval_choice(choice);
                            } else if let Some((line, x)) =
                                self.transcript_position(m.column, m.row)
                            {
                                self.transcript.begin_text_selection_at(line, x);
                                self.dirty.mark_selection();
                            } else {
                                self.transcript.clear_text_selection();
                                self.dirty.mark_selection();
                            }
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            if let Some((line, x)) =
                                self.transcript_position(m.column, m.row)
                            {
                                self.transcript.update_text_selection_at(line, x);
                                self.dirty.mark_selection();
                            }
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            if let Some((line, x)) =
                                self.transcript_position(m.column, m.row)
                            {
                                self.transcript.update_text_selection_at(line, x);
                            }
                            self.transcript.finish_text_selection();
                            self.dirty.mark_selection();
                        }
                        MouseEventKind::ScrollUp => {
                            self.transcript.scroll_up(WHEEL_LINES);
                            self.dirty.mark_chat();
                        }
                        MouseEventKind::ScrollDown => {
                            self.transcript.scroll_down(WHEEL_LINES);
                            self.dirty.mark_chat();
                        }
                        _ => {}
                    }
                }
            }
            TermEvent::Resize(_, _) => self.dirty = Dirty::all(),
            _ => {}
        }
        Ok(())
    }

    async fn on_key(&mut self, key: KeyEvent) -> std::io::Result<()> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        if ctrl && shift && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C')) {
            let _ = self.copy_text_selection();
            return Ok(());
        }
        if key.code == KeyCode::Esc && self.transcript.clear_text_selection() {
            self.dirty.mark_selection();
            return Ok(());
        }

        // Ctrl+C: cancel a running turn, otherwise quit on the second press.
        if ctrl && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C')) {
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
            if self.transcript.expand_last_tool() {
                self.focus = Focus::Scrollback;
            } else if let Some(output) = self
                .transcript
                .last_tool_output()
                .map(str::to_string)
                .or_else(|| self.last_tool_output.clone())
            {
                let lines = output
                    .lines()
                    .map(|l| {
                        Line::from(Span::styled(
                            format!("  {l}"),
                            Style::default().fg(theme::DIM()),
                        ))
                    })
                    .collect();
                self.transcript.push(lines);
            }
            return Ok(());
        }
        if ctrl && key.code == KeyCode::Char('t') {
            self.show_thinking = !self.show_thinking;
            return Ok(());
        }
        if ctrl && key.code == KeyCode::Char('r') && self.mode == Mode::Chat {
            self.scroll_search = None;
            self.open_history_search();
            return Ok(());
        }
        if ctrl && key.code == KeyCode::Char('f') && self.mode == Mode::Chat {
            // Find in conversation (not prompt history). Skip while approval /
            // completion menus own the keyboard so those flows stay intact.
            if self.pending_approval.is_none() && self.completion.is_none() {
                self.history_search = None;
                self.open_scroll_search();
            }
            return Ok(());
        }

        if self.mode == Mode::Models {
            self.on_models_key(key).await;
            return Ok(());
        }
        if self.mode == Mode::ModelPicker {
            self.on_model_picker_key(key);
            return Ok(());
        }
        if self.mode == Mode::Sessions {
            self.on_sessions_key(key);
            return Ok(());
        }

        // History search overlay owns keys until Esc/Enter.
        if self.history_search.is_some() {
            self.on_history_search_key(key);
            return Ok(());
        }

        // Scrollback find overlay owns keys until Esc/Enter.
        if self.scroll_search.is_some() {
            self.on_scroll_search_key(key);
            return Ok(());
        }

        // Transcript scrolling always available in chat.
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
        if let Some((id, name, _, _)) = self.pending_approval.clone() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.core.respond_approval(&id, true);
                    // UI clears on ApprovalSettled from the agent.
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.core.respond_approval(&id, false);
                }
                KeyCode::Char('a') | KeyCode::Char('A') => {
                    self.core.settings.lock().unwrap().approval_mode = "auto".into();
                    self.core.respond_approval(&id, true);
                    self.note("approvals set to auto for this run (change with /approvals)");
                }
                _ => {
                    let _ = name;
                }
            }
            return Ok(());
        }

        if matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y'))
            && self.copy_text_selection()
        {
            return Ok(());
        }

        // Completion popup: navigation and acceptance take priority over the
        // composer; anything else falls through and refilters afterwards.
        if self.completion.is_some() {
            match key.code {
                KeyCode::Up | KeyCode::BackTab => {
                    if let Some(popup) = &mut self.completion {
                        popup.prev();
                    }
                    return Ok(());
                }
                KeyCode::Down => {
                    if let Some(popup) = &mut self.completion {
                        popup.next();
                    }
                    return Ok(());
                }
                KeyCode::Tab | KeyCode::Enter => {
                    let has_item = self
                        .completion
                        .as_ref()
                        .is_some_and(|p| p.selected_item().is_some());
                    if has_item {
                        if let Some(command) = self.accept_completion() {
                            self.handle_submit(command).await?;
                        }
                        return Ok(());
                    }
                    // "No matches": close and let Enter submit as typed.
                    self.completion = None;
                }
                KeyCode::Esc => {
                    self.completion = None;
                    return Ok(());
                }
                _ => {}
            }
        }

        // Dual focus: Tab toggles composer ↔ scrollback.
        if key.code == KeyCode::Tab && self.completion.is_none() {
            self.focus = match self.focus {
                Focus::Composer => Focus::Scrollback,
                Focus::Scrollback => Focus::Composer,
            };
            if self.focus == Focus::Composer {
                self.transcript.clear_selection();
            } else if self.transcript.selected().is_none() && self.transcript.block_count() > 0 {
                self.transcript.select_prev();
            }
            return Ok(());
        }

        // Scrollback-focused navigation.
        if self.focus == Focus::Scrollback {
            let shift = key.modifiers.contains(KeyModifiers::SHIFT);
            match key.code {
                // Turn jumps: `[`/`]` work on every terminal; Shift+arrows when
                // the terminal reports modifiers (many do not).
                KeyCode::Char('[') => {
                    self.transcript.select_prev_user();
                    return Ok(());
                }
                KeyCode::Char(']') => {
                    self.transcript.select_next_user();
                    return Ok(());
                }
                KeyCode::Up if shift => {
                    self.transcript.select_prev_user();
                    return Ok(());
                }
                KeyCode::Down if shift => {
                    self.transcript.select_next_user();
                    return Ok(());
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.transcript.select_prev();
                    return Ok(());
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.transcript.select_next();
                    return Ok(());
                }
                KeyCode::Char('g') => {
                    self.transcript.select_first();
                    return Ok(());
                }
                KeyCode::Char('G') => {
                    self.transcript.select_last_follow();
                    return Ok(());
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    let _ = self.transcript.toggle_fold_selected();
                    return Ok(());
                }
                KeyCode::Char('o') => {
                    let _ = self.transcript.expand_last_tool();
                    return Ok(());
                }
                KeyCode::Char('y') => {
                    if let Some(text) = self.transcript.selected_copy_text() {
                        if clipboard::copy_text(&text) {
                            self.note("copied block");
                        } else {
                            self.note("copy failed (terminal may block OSC 52)");
                        }
                    }
                    return Ok(());
                }
                // Continue last Ctrl+F query without reopening the find bar.
                KeyCode::Char('n') => {
                    self.step_last_scroll_search(1);
                    return Ok(());
                }
                KeyCode::Char('N') => {
                    self.step_last_scroll_search(-1);
                    return Ok(());
                }
                KeyCode::Esc | KeyCode::Char(' ') => {
                    self.focus = Focus::Composer;
                    self.transcript.clear_selection();
                    return Ok(());
                }
                _ => {}
            }
        }

        if key.code == KeyCode::Esc {
            if self.running {
                if let Some(id) = &self.session_id {
                    self.core.cancel(id);
                }
            } else if self.transcript.offset() > 0 {
                self.transcript.follow();
                self.focus = Focus::Composer;
            } else if self.focus == Focus::Scrollback {
                self.focus = Focus::Composer;
                self.transcript.clear_selection();
            }
            return Ok(());
        }

        // Typing returns focus to the composer.
        self.focus = Focus::Composer;
        match self.composer.handle_key(key) {
            ComposerAction::Submit(text) => {
                self.completion = None;
                self.handle_submit(text).await?;
            }
            ComposerAction::None => self.sync_completion(),
        }
        Ok(())
    }

    fn transcript_position(&self, column: u16, row: u16) -> Option<(usize, usize)> {
        if !rect_contains(self.chat_draw_area, column, row) {
            return None;
        }
        let rendered_row = row.saturating_sub(self.chat_draw_area.y) as usize;
        let line = self.chat_line_map.get(rendered_row).copied().flatten()?;
        let x = column.saturating_sub(self.chat_draw_area.x) as usize;
        Some((line, x))
    }

    fn copy_text_selection(&mut self) -> bool {
        let Some(text) = self.transcript.selected_text() else {
            return false;
        };
        if clipboard::copy_text(&text) {
            self.note("copied selection");
        } else {
            self.note("copy failed (terminal may block OSC 52)");
        }
        true
    }

    /// Approval hit regions use the fixed order allow once, allow for run,
    /// deny. Keyboard handling remains the authoritative path.
    fn respond_approval_choice(&mut self, choice: usize) {
        let Some((id, _, _, _)) = self.pending_approval.clone() else {
            return;
        };
        match choice {
            0 => self.core.respond_approval(&id, true),
            1 => {
                self.core.settings.lock().unwrap().approval_mode = "auto".into();
                self.core.respond_approval(&id, true);
                self.note("approvals set to auto for this run (change with /approvals)");
            }
            2 => self.core.respond_approval(&id, false),
            _ => {}
        }
    }

    fn on_model_picker_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.mode = Mode::Chat;
                self.model_picker = None;
            }
            KeyCode::Up | KeyCode::BackTab => {
                if let Some(picker) = &mut self.model_picker {
                    picker.prev();
                }
            }
            KeyCode::Down | KeyCode::Tab => {
                if let Some(picker) = &mut self.model_picker {
                    picker.next();
                }
            }
            KeyCode::Backspace => {
                if let Some(picker) = &mut self.model_picker {
                    picker.backspace();
                }
            }
            KeyCode::Enter => {
                let choice = self
                    .model_picker
                    .as_ref()
                    .and_then(model_picker::ModelPickerState::selected_choice)
                    .cloned();
                if let Some(choice) = choice {
                    self.mode = Mode::Chat;
                    self.model_picker = None;
                    self.persist_model_selection(choice.provider, choice.id);
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(picker) = &mut self.model_picker {
                    picker.push(c);
                }
            }
            _ => {}
        }
        self.dirty.mark_chrome();
    }

    fn persist_model_selection(&mut self, provider: Option<String>, model: String) {
        let current = self.core.settings.lock().unwrap().clone();
        match save_model_selection(
            &self.core.data_dir,
            &current,
            provider.clone(),
            model.clone(),
        ) {
            Ok(next) => {
                *self.core.settings.lock().unwrap() = next;
                let source = provider
                    .map(|name| format!(" from {name}"))
                    .unwrap_or_default();
                self.note(&format!("model set to {model}{source}"));
            }
            Err(error) => self.error(&format!("could not save model selection: {error}")),
        }
    }

    fn open_history_search(&mut self) {
        let entries = self.composer.history_entries();
        if entries.is_empty() {
            self.note("no prompt history yet");
            return;
        }
        let matches = entries;
        let selected = matches.len().saturating_sub(1);
        self.history_search = Some((String::new(), selected, matches));
        self.completion = None;
    }

    fn on_history_search_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.history_search = None;
            }
            KeyCode::Enter => {
                let pick = self
                    .history_search
                    .as_ref()
                    .and_then(|(q, sel, all)| {
                        let _ = q;
                        all.get(*sel).cloned()
                    });
                if let Some(text) = pick {
                    self.composer.load(&text);
                }
                self.history_search = None;
                self.focus = Focus::Composer;
            }
            KeyCode::Up => {
                if let Some((_, sel, _)) = &mut self.history_search {
                    if *sel > 0 {
                        *sel -= 1;
                    }
                }
            }
            KeyCode::Down => {
                if let Some((_, sel, all)) = &mut self.history_search {
                    if *sel + 1 < all.len() {
                        *sel += 1;
                    }
                }
            }
            KeyCode::Backspace => {
                if let Some((query, _, _)) = &mut self.history_search {
                    query.pop();
                }
                self.refilter_history_search();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some((query, _, _)) = &mut self.history_search {
                    query.push(c);
                }
                self.refilter_history_search();
            }
            _ => {}
        }
    }

    fn refilter_history_search(&mut self) {
        let entries = self.composer.history_entries();
        let Some((query, selected, matches)) = &mut self.history_search else {
            return;
        };
        let q = query.to_ascii_lowercase();
        *matches = entries
            .into_iter()
            .filter(|e| q.is_empty() || e.to_ascii_lowercase().contains(&q))
            .collect();
        if matches.is_empty() {
            *selected = 0;
        } else {
            *selected = (*selected).min(matches.len() - 1);
        }
    }

    fn open_scroll_search(&mut self) {
        if self.transcript.block_count() == 0 {
            self.note("no conversation yet");
            return;
        }
        let n = self.transcript.block_count();
        let matches: Vec<usize> = (0..n).collect();
        let selected = matches.len().saturating_sub(1);
        self.scroll_search = Some((String::new(), selected, matches));
        self.completion = None;
        self.focus_scroll_match();
    }

    fn on_scroll_search_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                if let Some((q, _, _)) = &self.scroll_search {
                    if !q.is_empty() {
                        self.scroll_search_last = Some(q.clone());
                    }
                }
                self.scroll_search = None;
            }
            KeyCode::Enter => {
                if let Some((q, _, _)) = &self.scroll_search {
                    if !q.is_empty() {
                        self.scroll_search_last = Some(q.clone());
                    }
                }
                self.focus_scroll_match();
                self.scroll_search = None;
                self.focus = Focus::Scrollback;
            }
            // Next / previous match while the find bar is open.
            // (n/N step the last query after Enter, from scrollback focus, so
            // the letter n stays typeable in the query.)
            KeyCode::Up => {
                self.step_scroll_match(-1);
            }
            KeyCode::Down => {
                self.step_scroll_match(1);
            }
            KeyCode::Backspace => {
                if let Some((query, _, _)) = &mut self.scroll_search {
                    query.pop();
                }
                self.refilter_scroll_search();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some((query, _, _)) = &mut self.scroll_search {
                    query.push(c);
                }
                self.refilter_scroll_search();
            }
            _ => {}
        }
    }

    fn refilter_scroll_search(&mut self) {
        self.refilter_scroll_search_inner(true);
    }

    /// Refresh matches when the transcript grows under an open find bar.
    /// Keeps the current block selected if it still matches.
    fn refilter_scroll_search_live(&mut self) {
        if self.scroll_search.is_none() {
            return;
        }
        self.refilter_scroll_search_inner(false);
    }

    fn refilter_scroll_search_inner(&mut self, prefer_latest: bool) {
        let texts = self.transcript.all_block_search_texts();
        {
            let Some((query, selected, matches)) = &mut self.scroll_search else {
                return;
            };
            let prev_bi = matches.get(*selected).copied();
            *matches = filter_matching_indices(&texts, query);
            if matches.is_empty() {
                *selected = 0;
            } else if prefer_latest {
                *selected = matches.len() - 1;
            } else if let Some(bi) = prev_bi {
                *selected = matches
                    .iter()
                    .position(|&m| m == bi)
                    .unwrap_or(matches.len() - 1);
            } else {
                *selected = matches.len() - 1;
            }
        }
        self.focus_scroll_match();
    }

    /// Scroll the currently highlighted scroll-search match into view.
    fn focus_scroll_match(&mut self) {
        let Some((_, sel, matches)) = &self.scroll_search else {
            return;
        };
        if matches.is_empty() {
            self.transcript.clear_selection();
            return;
        }
        if let Some(&bi) = matches.get(*sel) {
            self.transcript.select_find_match(bi);
            self.focus = Focus::Scrollback;
        }
    }

    /// Step match selection by `delta` (-1 prev, +1 next), wrapping.
    fn step_scroll_match(&mut self, delta: i32) {
        {
            let Some((_, sel, matches)) = &mut self.scroll_search else {
                return;
            };
            if matches.is_empty() {
                return;
            }
            let n = matches.len() as i32;
            *sel = (*sel as i32 + delta).rem_euclid(n) as usize;
        }
        self.focus_scroll_match();
    }

    /// n/N after find closed: jump using the last query from scrollback focus.
    fn step_last_scroll_search(&mut self, delta: i32) {
        let Some(query) = self.scroll_search_last.clone() else {
            return;
        };
        if query.is_empty() {
            return;
        }
        let texts = self.transcript.all_block_search_texts();
        let matches = filter_matching_indices(&texts, &query);
        if matches.is_empty() {
            self.note("no matches");
            return;
        }
        let current = self.transcript.selected();
        let n = matches.len() as i32;
        let next = match current.and_then(|c| matches.iter().position(|&bi| bi == c)) {
            Some(pos) => (pos as i32 + delta).rem_euclid(n) as usize,
            None if delta > 0 => matches
                .iter()
                .position(|&bi| current.is_none_or(|c| bi >= c))
                .unwrap_or(0),
            None => matches
                .iter()
                .rposition(|&bi| current.is_none_or(|c| bi <= c))
                .unwrap_or(matches.len() - 1),
        };
        self.transcript.select_find_match(matches[next]);
        self.focus = Focus::Scrollback;
    }

    /// Accept the selected completion into the composer. Returns a command to
    /// submit immediately for no-argument slash commands.
    fn accept_completion(&mut self) -> Option<String> {
        let popup = self.completion.take()?;
        let item = popup.selected_item()?.clone();
        self.composer.replace_token(popup.token_start, popup.token_len, &item.insert);
        self.sync_completion();
        if item.submits {
            Some(self.composer.take())
        } else {
            None
        }
    }

    /// Open, refilter, or close the popup from the token under the cursor.
    fn sync_completion(&mut self) {
        if self.mode != Mode::Chat || self.pending_approval.is_some() {
            self.completion = None;
            return;
        }
        let (row, col, line) = self.composer.cursor_context();
        let line = line.to_string();
        match completion::trigger(&line, col, row == 0) {
            None => self.completion = None,
            Some((kind, token_start, query)) => {
                let token_len = query.chars().count() + 1;
                if kind == completion::Kind::File && self.completion.is_none() {
                    // A fresh `@` rescans in the background so files the agent
                    // just wrote show up; the old index serves meanwhile.
                    self.refresh_file_index();
                }
                if kind == completion::Kind::Slash && self.completion.is_none() {
                    // A fresh `/` rescans templates (two small dirs) so one
                    // the agent just wrote shows up immediately.
                    self.templates = open_max_core::templates::discover(&self.project)
                        .into_iter()
                        .map(|t| (t.name, t.description))
                        .collect();
                }
                let items = match kind {
                    completion::Kind::Slash => completion::slash_items(&query, &self.templates),
                    completion::Kind::File => match &self.file_index {
                        Some(files) => completion::file_items(files, &query),
                        None => Vec::new(),
                    },
                };
                self.completion =
                    Some(completion::Popup { kind, items, selected: 0, token_start, token_len });
            }
        }
    }

    fn refresh_file_index(&mut self) {
        if self.file_index_pending {
            return;
        }
        self.file_index_pending = true;
        let root = self.project.clone();
        let tx = self.files_tx.clone();
        tokio::task::spawn_blocking(move || {
            let _ = tx.send(completion::scan_files(&root));
        });
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

    fn on_sessions_key(&mut self, key: KeyEvent) {
        let Some(panel) = &mut self.sessions_panel else {
            self.mode = Mode::Chat;
            return;
        };

        // Delete confirmation intercepts.
        if let Some(id) = panel.confirm_delete.clone() {
            panel.confirm_delete = None;
            if matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y')) {
                match sessions::delete(&self.core, &id) {
                    Ok(()) => {
                        panel.items.retain(|s| s.id != id);
                        panel.selected = panel.selected.min(panel.items.len().saturating_sub(1));
                        if self.session_id.as_deref() == Some(id.as_str()) {
                            self.session_id = None;
                        }
                        if panel.items.is_empty() {
                            self.mode = Mode::Chat;
                            self.sessions_panel = None;
                            self.note("no sessions left in this project");
                        }
                    }
                    Err(e) => self.error(&e),
                }
            }
            return;
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.mode = Mode::Chat;
                self.sessions_panel = None;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                panel.selected = panel.selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if panel.selected + 1 < panel.items.len() {
                    panel.selected += 1;
                }
            }
            KeyCode::Char('x') => {
                if let Some(item) = panel.selected_item() {
                    panel.confirm_delete = Some(item.id.clone());
                }
            }
            KeyCode::Enter => {
                if let Some(id) = panel.selected_item().map(|s| s.id.clone()) {
                    self.sessions_panel = None;
                    self.mode = Mode::Chat;
                    if self.session_id.as_deref() == Some(id.as_str()) {
                        self.note("already in this session");
                        return;
                    }
                    self.reset_for_new_session();
                    self.session_id = Some(id.clone());
                    self.replay(&id);
                }
            }
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
        let text = if let Some(cmd) = text.strip_prefix('/') {
            let head = cmd.split_whitespace().next().unwrap_or("");
            let builtin = head == "exit"
                || completion::COMMANDS.iter().any(|spec| spec.name == head);
            // Built-ins win; anything else may be a prompt template, whose
            // expansion submits as a normal user message.
            let expanded = if builtin {
                None
            } else {
                open_max_core::templates::expand_invocation(&self.project, cmd)
            };
            match expanded {
                Some(expanded) => expanded,
                None => {
                    self.slash(cmd).await;
                    return Ok(());
                }
            }
        } else {
            text
        };
        if self.running {
            // Queue instead of refusing: the message goes out, in order, as
            // soon as the current turn finishes. Esc cancels and hands the
            // queue back to the composer.
            self.queued.push(text);
            self.transcript.follow();
            self.dirty.mark_chat();
            self.dirty.mark_chrome();
            return Ok(());
        }

        // Friendly gate: when the resolved endpoint is the managed local MLX
        // server and it is not serving yet, guide to /models instead of erroring.
        let (managed, ready) = {
            let s = self.core.settings.lock().unwrap();
            let managed = match open_max_core::providers::resolve(&s, &self.core.data_dir) {
                Ok(ep) => open_max_core::providers::is_managed_mlx(&ep, s.mlx_port),
                Err(_) => false,
            };
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

        // Painted optimistically; rolled back on `stop_reason: "blocked"` if
        // user_prompt_submit rejects before the text enters the core transcript.
        self.insert_user_block(&text);
        self.transcript.follow();
        match agent::start_turn(
            self.core.clone(),
            session_id,
            self.project.clone(),
            text.clone(),
        ) {
            Ok(()) => {
                self.running = true;
                self.pending_submit = Some(text);
                self.turn_started = Some(Instant::now());
                self.first_token = None;
                self.stream_chars = 0;
                self.stream_text.clear();
                self.stream_wrapped.clear();
                self.stream_md.clear();
                self.thinking_chars = 0;
                self.thinking_tail.clear();
                self.thinking_source.clear();
                self.thinking_wrapped.clear();
                self.dirty.mark_chat();
                self.dirty.mark_chrome();
            }
            Err(e) => {
                let _ = self.transcript.pop_last_user();
                self.error(&e);
            }
        }
        Ok(())
    }

    async fn slash(&mut self, cmd: &str) {
        let (head, raw_rest) = command_parts(cmd);
        let rest: Vec<&str> = raw_rest.split_whitespace().collect();
        match head {
            "help" => {
                let mut block: Vec<Line<'static>> = HELP_KEYS
                    .iter()
                    .map(|(k, v)| {
                        Line::from(vec![
                            Span::styled(format!("  {k:<32}"), Style::default().fg(theme::ACCENT())),
                            Span::styled((*v).to_string(), Style::default().fg(theme::DIM())),
                        ])
                    })
                    .collect();
                block.push(Line::default());
                block.extend(completion::COMMANDS.iter().map(|spec| {
                    Line::from(vec![
                        Span::styled(
                            format!("  {:<32}", spec.usage()),
                            Style::default().fg(theme::ACCENT()),
                        ),
                        Span::styled(
                            spec.description.to_string(),
                            Style::default().fg(theme::DIM()),
                        ),
                    ])
                }));
                self.transcript.push(block);
                self.dirty.mark_chat();
            }
            "theme" => {
                match rest.first().map(|s| s.to_ascii_lowercase()).as_deref() {
                    Some("light" | "day") => {
                        theme::apply(theme::ThemeId::Light);
                        self.note("theme: light");
                    }
                    Some("dark" | "night") => {
                        theme::apply(theme::ThemeId::Dark);
                        self.note("theme: dark");
                    }
                    Some("catppuccin" | "mocha" | "cat") => {
                        theme::apply(theme::ThemeId::Catppuccin);
                        self.note("theme: catppuccin");
                    }
                    Some("mono" | "bw") => {
                        theme::set_tokens(theme::Tokens::mono());
                        self.note("theme: mono");
                    }
                    _ => self.note("usage: /theme dark|light|mono|catppuccin"),
                }
                self.transcript.invalidate_styles();
                self.hist_reuse_key = None;
                self.tail_width = 0;
                self.thinking_source.clear();
                self.dirty = Dirty::all();
            }
            "models" => {
                self.mode = Mode::Models;
                self.models.ensure_loaded(ram_bytes());
                self.models.refresh();
                self.models.status = Some(mlx::status(&self.core).await);
                self.fetch_missing_sizes();
                self.dirty.mark_chrome();
            }
            "model" if raw_rest.is_empty() => {
                let settings = self.core.settings.lock().unwrap().clone();
                self.model_picker = Some(model_picker::ModelPickerState::load(
                    &self.core.data_dir,
                    settings.provider.as_deref(),
                    &settings.model,
                ));
                self.transcript.clear_text_selection();
                self.completion = None;
                self.mode = Mode::ModelPicker;
                self.dirty.mark_chrome();
            }
            "model" => {
                let provider = self.core.settings.lock().unwrap().provider.clone();
                self.persist_model_selection(provider, raw_rest.to_string());
            }
            "copy" => {
                if let Some(text) = self
                    .last_assistant_response
                    .clone()
                    .or_else(|| self.transcript.last_assistant_text())
                {
                    if clipboard::copy_text(&text) {
                        self.note("copied latest assistant response");
                    } else {
                        self.note("copy failed (terminal may block OSC 52)");
                    }
                } else {
                    self.note("no assistant response to copy");
                }
            }
            "provider" => {
                let names = open_max_core::providers::list_provider_names(&self.core.data_dir);
                match rest.first() {
                    None => {
                        if names.is_empty() {
                            self.note(
                                "no providers in ~/.openmax/providers.json (flat base_url still works)",
                            );
                        } else {
                            let active = self
                                .core
                                .settings
                                .lock()
                                .unwrap()
                                .provider
                                .clone()
                                .unwrap_or_default();
                            let mut lines = vec![Line::from(Span::styled(
                                "  providers".to_string(),
                                Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD),
                            ))];
                            for name in &names {
                                let mark = if *name == active { "*" } else { " " };
                                lines.push(Line::from(Span::styled(
                                    format!("  {mark} {name}"),
                                    Style::default().fg(theme::DIM()),
                                )));
                            }
                            lines.push(Line::from(Span::styled(
                                "  use /provider <name> to switch".to_string(),
                                Style::default().fg(theme::DIM()),
                            )));
                            self.transcript.push(lines);
                        }
                    }
                    Some(name) => {
                        let name = name.to_string();
                        let providers = open_max_core::providers::load_providers(&self.core.data_dir);
                        let Some(p) = providers.get(&name) else {
                            self.note(&format!(
                                "unknown provider '{name}'; define it in ~/.openmax/providers.json"
                            ));
                            return;
                        };
                        {
                            let mut s = self.core.settings.lock().unwrap();
                            s.provider = Some(name.clone());
                            // Keep the model only if it is still in this catalog;
                            // otherwise switch to the first listed model.
                            if !p.models.is_empty() && !p.models.iter().any(|m| m.id == s.model) {
                                s.model = p.models[0].id.clone();
                                s.mlx_model = s.model.clone();
                            }
                            let _ = config::save(&self.core.data_dir, &s);
                        }
                        let ep = {
                            let s = self.core.settings.lock().unwrap();
                            open_max_core::providers::resolve(&s, &self.core.data_dir)
                        };
                        match ep {
                            Ok(ep) => self.note(&format!(
                                "provider {name} → {} ({})",
                                extensions::display_base_url(&ep.base_url),
                                ep.model
                            )),
                            Err(e) => self.note(&e.to_string()),
                        }
                    }
                }
            }
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
            "resume" => {
                let items = sessions::list(&self.core, &self.project.display().to_string());
                if items.is_empty() {
                    self.note("no sessions in this project yet");
                } else {
                    self.sessions_panel = Some(sessions_ui::SessionsState::new(items));
                    self.completion = None;
                    self.mode = Mode::Sessions;
                }
            }
            "reload" => match &self.session_id {
                None => self.note("no session yet; a new session always freezes the current config"),
                Some(id) => {
                    let id = id.clone();
                    match agent::reload_session(&self.core, &id, &self.project).await {
                        Ok((tools, skills)) => self.note(&format!(
                            "re-frozen: {tools} tools, {skills} skills (prompt cache will re-prefill once)"
                        )),
                        Err(e) => self.error(&e),
                    }
                }
            },
            "new" => {
                let old_id = self.session_id.clone();
                self.reset_for_new_session();
                if let Some(id) = old_id {
                    if let Ok(mut sessions) = self.core.sessions.try_lock() {
                        sessions.remove(&id);
                    }
                }
                self.note("new session");
            }
            "context" => {
                // A hydrated session shows its frozen breakdown; before any
                // turn runs (or with no session), preview what the next new
                // session would freeze from today's config.
                let frozen = match &self.session_id {
                    Some(id) => self
                        .core
                        .sessions
                        .lock()
                        .await
                        .get(id)
                        .map(|data| data.prompt_breakdown.as_ref().clone()),
                    None => None,
                };
                let (breakdown, is_frozen) = match frozen {
                    Some(b) => (b, true),
                    None => {
                        let project = self.project.clone();
                        let registry = tokio::task::spawn_blocking({
                            let project = project.clone();
                            move || registry::Registry::build(&project)
                        })
                        .await
                        .unwrap_or_else(|_| registry::Registry::builtin_only());
                        let (_, b) = prompt::system_prompt_with_breakdown(&project, &registry);
                        (b, false)
                    }
                };
                self.transcript.push(context::context_block(
                    &breakdown,
                    is_frozen,
                    self.budget,
                    self.cache_pct,
                ));
            }
            "tools" => {
                // Hold the sessions lock only long enough to format a frozen
                // registry. Disk preview runs after the lock is released so a
                // slow tools/skills tree cannot stall other session access.
                let frozen = {
                    let sessions = self.core.sessions.lock().await;
                    self.session_id
                        .as_ref()
                        .and_then(|id| sessions.get(id))
                        .map(|data| extensions::tools_block(&data.registry, true))
                };
                let lines = if let Some(lines) = frozen {
                    lines
                } else {
                    let project = self.project.clone();
                    let reg = tokio::task::spawn_blocking(move || registry::Registry::build(&project))
                        .await
                        .unwrap_or_else(|_| registry::Registry::builtin_only());
                    extensions::tools_block(&reg, false)
                };
                self.transcript.push(lines);
            }
            "skills" => {
                let frozen = {
                    let sessions = self.core.sessions.lock().await;
                    self.session_id.as_ref().and_then(|id| {
                        sessions.get(id).map(|data| {
                            extensions::skills_block(&data.registry.skills, &self.project, true)
                        })
                    })
                };
                let lines = if let Some(lines) = frozen {
                    lines
                } else {
                    let project = self.project.clone();
                    let reg = tokio::task::spawn_blocking(move || registry::Registry::build(&project))
                        .await
                        .unwrap_or_else(|_| registry::Registry::builtin_only());
                    extensions::skills_block(&reg.skills, &self.project, false)
                };
                self.transcript.push(lines);
            }
            "status" => {
                let s = self.core.settings.lock().unwrap().clone();
                let ep = open_max_core::providers::resolve(&s, &self.core.data_dir);
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
                let cache = self
                    .cache_pct
                    .map(|percent| format!("{percent}% prompt tokens"))
                    .unwrap_or_else(|| "not reported".into());
                let ttft = match (self.turn_started, self.first_token) {
                    (Some(started), Some(first)) => {
                        format!("{} ms", first.saturating_duration_since(started).as_millis())
                    }
                    _ => "not available".into(),
                };
                let throughput = {
                    let rate = self.tok_per_sec();
                    if rate > 0.0 {
                        format!("{rate:.1} tokens/s")
                    } else {
                        "not available".into()
                    }
                };
                let (provider, model, endpoint, host, context_tokens) = match &ep {
                    Ok(ep) => {
                        let endpoint = extensions::display_base_url(&ep.base_url);
                        let host = extensions::endpoint_host(&ep.base_url)
                            .unwrap_or_else(|| endpoint.clone());
                        (
                            ep.provider.as_deref().unwrap_or("(flat base_url)").to_string(),
                            ep.model.clone(),
                            endpoint,
                            host,
                            ep.context_tokens,
                        )
                    }
                    Err(e) => (
                        format!("error: {e}"),
                        s.model.clone(),
                        extensions::display_base_url(&s.base_url),
                        extensions::endpoint_host(&s.base_url).unwrap_or_else(|| s.base_url.clone()),
                        s.context_tokens,
                    ),
                };
                let block = vec![
                    kv("provider", &provider),
                    kv("model", &model),
                    kv("endpoint", &endpoint),
                    kv("host", &host),
                    kv("server", &server),
                    kv("approvals", &s.approval_mode),
                    kv("context", &format!("{ctx} of {} tokens", context_tokens)),
                    kv("cache", &cache),
                    kv("ttft", &ttft),
                    kv("throughput", &throughput),
                    kv("session", self.session_id.as_deref().unwrap_or("none yet")),
                    kv("project", &self.project.display().to_string()),
                    kv("data", &self.core.data_dir.display().to_string()),
                    Line::from(Span::styled(
                        "  network".to_string(),
                        Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD),
                    )),
                    kv("  dest", &endpoint),
                    kv(
                        "  also",
                        "Hugging Face only when you use /models to download or serve",
                    ),
                    kv(
                        "  privacy",
                        "no telemetry · sessions stay local · external tools may use the network",
                    ),
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
                    .map(|l| Line::from(Span::styled(format!("  {l}"), Style::default().fg(theme::DIM()))))
                    .collect();
                if tail.is_empty() {
                    self.note("no server logs yet");
                } else {
                    self.transcript.push(tail);
                }
            }
            "quit" | "exit" => self.should_quit = true,
            other => self.note(&format!(
                "unknown command: /{other} (see /help; prompt templates live in .agents/prompts/)"
            )),
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
            CoreEvent::Mlx(ev) => {
                // Log lines are pull-only (/logs). Repainting per line would
                // turn the server's own logging into a render load while the
                // model generates.
                if matches!(ev, MlxEvent::ServerLog { .. } | MlxEvent::SetupLog { .. }) {
                    return;
                }
                self.on_mlx_event(ev).await
            }
            CoreEvent::Download(ev) => match ev {
                DownloadEvent::Progress { repo, done_bytes, total_bytes } => {
                    self.models.download = Some((repo, done_bytes, total_bytes));
                    self.dirty.mark_chrome();
                }
                DownloadEvent::Done { ok, message, .. } => {
                    self.models.download = None;
                    self.models.refresh();
                    self.dirty.mark_chrome();
                    if ok {
                        self.note(&message);
                    } else {
                        self.error(&message);
                    }
                }
            },
        }
        // Send the next queued message once the turn has fully settled.
        if self.flush_queue {
            self.flush_queue = false;
            if !self.running && self.pending_approval.is_none() && !self.queued.is_empty() {
                let text = self.queued.remove(0);
                let _ = self.handle_submit(text.clone()).await;
                if !self.running {
                    // The turn never started (server stopped, start error):
                    // nothing will drain the rest, so hand it all back.
                    self.queued.insert(0, text);
                    self.return_queue_to_composer();
                }
            }
        }
    }

    /// Move queued messages into the composer (in front of any draft) so
    /// nothing typed during a turn is lost when the turn cannot continue.
    fn return_queue_to_composer(&mut self) {
        if self.queued.is_empty() {
            return;
        }
        let mut text = self.queued.join("\n");
        self.queued.clear();
        if !self.composer.is_empty() {
            text.push('\n');
            text.push_str(&self.composer.text());
        }
        self.composer.load(&text);
        self.dirty.mark_chrome();
        self.note("queued input returned to the composer");
    }

    /// A dim one-line record of a finished turn, kept only when the turn was
    /// long enough for the numbers to mean something.
    fn push_turn_stats(&mut self) {
        let Some(started) = self.turn_started else { return };
        let secs = started.elapsed().as_secs();
        if secs < 5 {
            return;
        }
        let mut meta = format!("◦ {secs}s");
        let toks = self.tok_per_sec();
        if toks > 0.0 {
            meta.push_str(&format!(" · {toks:.0} tok/s"));
        }
        if let Some(pct) = self.cache_pct {
            meta.push_str(&format!(" · cache {pct}%"));
        }
        self.transcript.push(vec![Line::from(Span::styled(
            meta,
            Style::default().fg(theme::DIM()),
        ))]);
        self.dirty.mark_chat();
    }

    fn on_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::Token { text } => {
                self.first_token.get_or_insert_with(Instant::now);
                self.stream_chars += text.len();
                self.stream_text.push_str(&text);
                self.dirty.mark_tail();
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
                self.dirty.mark_tail();
            }
            AgentEvent::MessageDone { text } => {
                if !text.trim().is_empty() {
                    self.last_assistant_response = Some(text.clone());
                    self.transcript.push_assistant(markdown::render(
                        &text,
                        markdown::highlighter(),
                    ));
                    self.refilter_scroll_search_live();
                }
                self.stream_text.clear();
                self.stream_wrapped.clear();
                self.stream_md.clear();
                self.thinking_tail.clear();
                self.thinking_source.clear();
                self.thinking_wrapped.clear();
                self.thinking_chars = 0;
                // Caches above are already empty, so rebuild_tail would not see a
                // stream_changed edge; drop the stitched tail content here so the
                // finished assistant body is not still painted under the spinner.
                self.tail_content_len = 0;
                self.tail_buf.clear();
                self.dirty.mark_chat();
            }
            AgentEvent::Budget { used_tokens, context_tokens } => {
                self.budget = Some((used_tokens, context_tokens));
                self.dirty.mark_chrome();
            }
            AgentEvent::Usage { prompt_tokens, cached_tokens, .. } => {
                self.cache_pct = match cached_tokens {
                    Some(c) if prompt_tokens > 0 => {
                        Some(((c as f64 / prompt_tokens as f64) * 100.0).round() as u8)
                    }
                    _ => None,
                };
                self.dirty.mark_chrome();
            }
            AgentEvent::ToolStart { call_id, name, args } => {
                let summary = registry::summarize_call(&name, &args);
                self.tool_meta.insert(
                    call_id,
                    ToolMeta {
                        name: name.clone(),
                        summary: summary.clone(),
                        started: Instant::now(),
                    },
                );
                self.running_tool = Some((name, summary));
                self.dirty.mark_tail();
            }
            AgentEvent::Diff { call_id, path, diff, added, removed } => {
                self.pending_diffs.insert(call_id, DiffText { path, diff, added, removed });
            }
            AgentEvent::ToolEnd { call_id, ok, output } => {
                let meta = self
                    .tool_meta
                    .remove(&call_id)
                    .unwrap_or_else(|| ToolMeta {
                        name: "tool".into(),
                        summary: String::new(),
                        started: Instant::now(),
                    });
                let duration = meta.started.elapsed();
                let diff = self.pending_diffs.remove(&call_id);
                let compact = tool_card::tool_block_timed(
                    &meta.name,
                    &meta.summary,
                    ok,
                    &output,
                    diff.as_ref(),
                    Some(duration),
                );
                self.transcript.push_tool(compact, output.clone());
                self.last_tool_output = Some(output);
                self.running_tool = None;
                self.refilter_scroll_search_live();
                self.dirty.mark_chat();
            }
            AgentEvent::ApprovalRequest {
                approval_id,
                name,
                summary,
                detail,
            } => {
                self.pending_approval = Some((approval_id, name, summary, detail));
                self.completion = None;
                self.dirty.mark_chrome();
            }
            AgentEvent::ApprovalSettled {
                approval_id,
                outcome,
            } => {
                if self
                    .pending_approval
                    .as_ref()
                    .is_some_and(|(id, _, _, _)| id == &approval_id)
                {
                    self.pending_approval = None;
                    self.dirty.mark_chrome();
                    match outcome.as_str() {
                        "timed_out" => self.note("approval timed out · declined"),
                        "cancelled" => self.note("approval cancelled"),
                        _ => {}
                    }
                }
            }
            AgentEvent::Refrozen { tools, skills } => {
                self.note(&format!(
                    "extensions changed on disk: re-frozen with {tools} tools, {skills} skills"
                ));
            }
            AgentEvent::Done { stop_reason } => {
                self.running = false;
                self.running_tool = None;
                self.pending_approval = None;
                // Spinner/status clear plus any transcript note/stats.
                self.dirty.mark_chat();
                self.dirty.mark_chrome();
                match stop_reason.as_str() {
                    "stop" | "tool_calls" => {
                        self.pending_submit = None;
                        self.push_turn_stats();
                    }
                    "cancelled" => {
                        self.pending_submit = None;
                        self.note("cancelled");
                    }
                    "length" => {
                        self.pending_submit = None;
                        self.note("stopped: hit the response token limit");
                    }
                    "max_iterations" => {
                        self.pending_submit = None;
                        self.note(
                            "stopped: reached the tool-call limit for one turn (send a follow-up to continue)",
                        );
                    }
                    "error" => {
                        self.pending_submit = None;
                    }
                    "blocked" => {
                        // Core never accepted the text: drop the optimistic
                        // user bubble and restore it to the composer.
                        let _ = self.transcript.pop_last_user();
                        if let Some(text) = self.pending_submit.take() {
                            if self.composer.is_empty() {
                                self.composer.load(&text);
                            } else {
                                // Preserve any draft the user started typing
                                // while the block was in flight.
                                let mut combined = text;
                                combined.push('\n');
                                combined.push_str(&self.composer.text());
                                self.composer.load(&combined);
                            }
                        }
                        self.dirty.mark_chat();
                        self.dirty.mark_chrome();
                    }
                    other => {
                        self.pending_submit = None;
                        self.note(&format!("stopped: {other}"));
                    }
                }
                if !self.queued.is_empty() {
                    // An interrupted or failed turn returns the queue to the
                    // composer instead of firing blind into a broken state.
                    if matches!(stop_reason.as_str(), "cancelled" | "error" | "blocked") {
                        self.return_queue_to_composer();
                    } else {
                        self.flush_queue = true;
                    }
                }
            }
            AgentEvent::Error { message } => {
                self.pending_approval = None;
                self.dirty.mark_chrome();
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
            self.dirty.mark_chrome();
        }
    }

    fn tick_armed(&self) -> bool {
        self.running || self.models.download.is_some() || self.mode == Mode::Models
    }

    async fn on_tick(&mut self) {
        self.tick_i += 1;
        if self.running {
            self.spinner_i = (self.spinner_i + 1) % SPINNER.len();
            // Spinner lives in the live tail; history stays reusable.
            self.dirty.mark_tail();
        } else if self.models.download.is_some() {
            self.spinner_i = (self.spinner_i + 1) % SPINNER.len();
            self.dirty.mark_chrome();
        }
        // Refresh server status occasionally while the panel is open.
        if self.mode == Mode::Models && self.tick_i.is_multiple_of(16) {
            self.models.status = Some(mlx::status(&self.core).await);
            self.dirty.mark_chrome();
        }
    }

    // ---------- blocks ----------

    fn insert_user_block(&mut self, text: &str) {
        let lines = text
            .lines()
            .map(|line| {
                Line::from(Span::styled(
                    line.to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                ))
            })
            .collect();
        self.transcript.push_user(lines);
        self.dirty.mark_chat();
    }

    fn note(&mut self, text: &str) {
        if self.mode == Mode::Models {
            self.models.footer = Some((text.to_string(), false));
            self.dirty.mark_chrome();
        } else {
            self.transcript.push(vec![Line::from(vec![
                Span::styled("• ", Style::default().fg(theme::ACCENT())),
                Span::styled(
                    text.to_string(),
                    Style::default()
                        .fg(theme::DIM())
                        .add_modifier(Modifier::ITALIC),
                ),
            ])]);
            self.dirty.mark_chat();
        }
    }

    fn error(&mut self, text: &str) {
        if self.mode == Mode::Models {
            self.models.footer = Some((text.to_string(), true));
            self.dirty.mark_chrome();
        } else {
            let mut lines = Vec::new();
            for (i, l) in text.lines().enumerate() {
                let prefix = if i == 0 { "✗ " } else { "  " };
                lines.push(Line::from(Span::styled(
                    format!("{prefix}{l}"),
                    Style::default().fg(theme::ERR()),
                )));
            }
            self.transcript.push(lines);
            self.dirty.mark_chat();
        }
    }

    // ---------- drawing ----------

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        self.approval_hits = [None; 3];
        if self.mode == Mode::ModelPicker {
            if let Some(picker) = &self.model_picker {
                model_picker::render(frame, area, picker);
            }
            return;
        }
        if self.mode == Mode::Models {
            models::render(frame, area, &self.models);
            return;
        }
        if self.mode == Mode::Sessions {
            if let Some(panel) = &self.sessions_panel {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                sessions_ui::render(frame, area, panel, now);
            }
            return;
        }

        // The composer and approval card share one input band. Approvals
        // temporarily own that space instead of stacking more chrome.
        let status_h = 1u16.min(area.height);
        let header_h = 1u16.min(area.height.saturating_sub(status_h + 2));
        let input_h = if self.pending_approval.is_some() {
            5
        } else {
            self.composer.height().saturating_add(2).max(3)
        }
        .min(area.height.saturating_sub(header_h + status_h));
        let queue_h = if self.queued.is_empty() {
            0
        } else {
            (self.queued.len() as u16)
                .min(3)
                .min(area.height.saturating_sub(header_h + status_h + input_h))
        };
        let hist_lines = self.history_search.as_ref().map(|(q, sel, items)| {
            history_search_lines(q, *sel, items, area.width)
        });
        let find_lines = if hist_lines.is_some() {
            None
        } else {
            self.scroll_search.as_ref().map(|(q, sel, matches)| {
                scroll_search_lines(q, *sel, matches, &self.transcript, area.width)
            })
        };
        let popup_lines = if hist_lines.is_some() || find_lines.is_some() {
            None
        } else {
            self.completion.as_ref().map(|p| {
                let indexing = p.kind == completion::Kind::File && self.file_index.is_none();
                completion::render_lines(p, area.width, indexing)
            })
        };
        let popup_h = hist_lines
            .as_ref()
            .or(find_lines.as_ref())
            .or(popup_lines.as_ref())
            .map(|l| l.len() as u16)
            .unwrap_or(0)
            .min(area.height.saturating_sub(header_h + status_h + input_h + queue_h));
        let chrome = header_h + queue_h + popup_h + input_h + status_h;
        let chat_h = area.height.saturating_sub(chrome);
        self.page_h = chat_h.saturating_sub(1).max(1);

        // Top to bottom: header, transcript, queued input, popup, input, status.
        let header_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: header_h,
        };
        let chat_area = Rect {
            x: area.x,
            y: area.y + header_h,
            width: area.width,
            height: chat_h,
        };
        let mut y = area.y + header_h + chat_h;
        let queue_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: queue_h,
        };
        y += queue_h;
        let popup_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: popup_h,
        };
        y += popup_h;
        let input_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: input_h,
        };
        y += input_h;
        let status_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: status_h,
        };

        if header_h > 0 {
            self.draw_header(frame, header_area);
        }
        if chat_h > 0 {
            self.draw_chat(frame, chat_area);
        }

        if queue_h > 0 {
            let mut qlines: Vec<Line> = self
                .queued
                .iter()
                .take(queue_h as usize)
                .map(|q| {
                    Line::from(vec![
                        Span::styled("↳ ", Style::default().fg(theme::ACCENT())),
                        Span::styled(
                            clip(&q.replace('\n', " "), area.width.saturating_sub(4) as usize),
                            Style::default()
                                .fg(theme::DIM())
                                .add_modifier(Modifier::ITALIC),
                        ),
                    ])
                })
                .collect();
            if self.queued.len() as u16 > queue_h {
                qlines.pop();
                qlines.push(Line::from(Span::styled(
                    format!("↳ … {} more queued", self.queued.len() as u16 - queue_h + 1),
                    Style::default().fg(theme::DIM()),
                )));
            }
            Paragraph::new(qlines).render(queue_area, frame.buffer_mut());
        }

        if let Some(lines) = hist_lines.or(find_lines).or(popup_lines) {
            if popup_h > 0 {
                Paragraph::new(lines).render(popup_area, frame.buffer_mut());
            }
        }

        if self.pending_approval.is_some() {
            self.draw_approval(frame, input_area);
        } else if input_h > 0 {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme::BORDER()))
                .style(Style::default().bg(theme::COMPOSER_BG()));
            let inner = block.inner(input_area);
            block.render(input_area, frame.buffer_mut());
            let (composer_lines, cx, cy) = self.composer.render(inner.height);
            Paragraph::new(composer_lines)
                .style(Style::default().bg(theme::COMPOSER_BG()))
                .render(inner, frame.buffer_mut());
            if self.focus == Focus::Composer
                && self.history_search.is_none()
                && self.scroll_search.is_none()
            {
                frame.set_cursor_position(Position::new(inner.x + cx, inner.y + cy));
            }
        }
        self.draw_status(frame, status_area);
    }

    fn draw_approval(&mut self, frame: &mut Frame, area: Rect) {
        let Some((_, name, summary, detail)) = self.pending_approval.as_ref() else {
            return;
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme::WARN()))
            .title(Span::styled(
                " Approval ",
                Style::default()
                    .fg(theme::WARN())
                    .add_modifier(Modifier::BOLD),
            ))
            .style(Style::default().bg(theme::SURFACE()));
        let inner = block.inner(area);
        block.render(area, frame.buffer_mut());
        let lines = approval_card_lines(name, summary, detail, inner.width);
        Paragraph::new(lines)
            .style(Style::default().bg(theme::SURFACE()))
            .render(inner, frame.buffer_mut());

        if inner.height >= 3 {
            self.approval_hits = approval_hit_regions(inner);
        }
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let line = Line::from(vec![
            Span::styled(
                "◆ Open Max",
                Style::default()
                    .fg(theme::ACCENT())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {}", clip(&self.dir_label, area.width.saturating_sub(13) as usize)),
                Style::default().fg(theme::DIM()),
            ),
        ]);
        Paragraph::new(line).render(area, frame.buffer_mut());
    }

    /// Finished transcript plus the live tail, bottom anchored, honoring the
    /// scroll offset (0 follows the latest output).
    ///
    /// When only the live tail is dirty (spinner / tokens), the history prefix
    /// of `chat_buf` is reused and the tail is re-stitched.
    fn draw_chat(&mut self, frame: &mut Frame, area: Rect) {
        let layout_started = Instant::now();
        let content_w = area.width.saturating_sub(1);
        if content_w == 0 || area.height == 0 {
            self.chat_draw_area = Rect::default();
            self.chat_line_map.clear();
            self.perf_layout_ms = layout_started.elapsed().as_secs_f64() * 1000.0;
            self.perf_selection_ms = 0.0;
            return;
        }
        let chat_dirty = self.dirty.chat;

        self.transcript.set_width(content_w);
        let tail_len = self.rebuild_tail(content_w);

        let hist_len = self.transcript.len();
        let total = hist_len + tail_len;
        let visible = area.height as usize;
        self.transcript.clamp_offset(total.saturating_sub(visible));
        let offset = self.transcript.offset();

        let end = total - offset;
        let start = end.saturating_sub(visible);

        // Fingerprint sticky presence without cloning spans; clone only if we rebuild.
        let has_sticky = offset > 0 && self.transcript.has_sticky_user(start);
        let focus_scroll = self.focus == Focus::Scrollback;
        let selected = self.transcript.selected();
        let hist_view_end = end.min(hist_len);
        let reuse_key = HistReuseKey {
            hist_len,
            start,
            hist_view_end,
            sticky: has_sticky,
            focus_scroll,
            selected,
            width: content_w,
        };

        let rebuild_hist = chat_dirty
            || self.hist_reuse_key != Some(reuse_key)
            || self.hist_prefix_len > self.chat_buf.len();

        if rebuild_hist {
            self.chat_buf.clear();
            self.chat_line_map.clear();
            // One clone of sticky spans: take ownership and insert the gutter.
            if has_sticky {
                if let Some(mut s) = self.transcript.sticky_user_line(start) {
                    s.spans
                        .insert(0, Span::styled("┊ ", Style::default().fg(theme::DIM())));
                    self.chat_buf.push(s);
                    self.chat_line_map.push(None);
                }
            }
            let budget = visible.saturating_sub(self.chat_buf.len());
            let view_end = start.saturating_add(budget).min(hist_view_end);
            let selected_bi = if focus_scroll { selected } else { None };
            // Single clone per viewport history line (reuse path skips this).
            self.transcript
                .fill_viewport(&mut self.chat_buf, start, view_end, selected_bi);
            self.chat_line_map.extend((start..view_end).map(Some));
            self.hist_prefix_len = self.chat_buf.len();
            self.hist_reuse_key = Some(reuse_key);
        } else {
            self.chat_buf.truncate(self.hist_prefix_len);
            self.chat_line_map.truncate(self.hist_prefix_len);
        }

        // Stitch visible tail after the history prefix.
        let budget = visible.saturating_sub(self.chat_buf.len());
        let mut idx = start.max(hist_len);
        let mut taken = 0usize;
        while taken < budget && idx < end {
            let ti = idx - hist_len;
            if ti < self.tail_buf.len() {
                self.chat_buf.push(self.tail_buf[ti].clone());
                self.chat_line_map.push(None);
            }
            idx += 1;
            taken += 1;
        }

        let pad = area.height.saturating_sub(self.chat_buf.len() as u16);
        let draw_area = Rect {
            x: area.x,
            y: area.y + pad,
            width: content_w,
            height: area.height - pad,
        };
        self.chat_draw_area = draw_area;
        Paragraph::new(self.chat_buf.as_slice()).render(draw_area, frame.buffer_mut());
        self.perf_layout_ms = layout_started.elapsed().as_secs_f64() * 1000.0;

        let selection_started = Instant::now();
        paint_text_selection(
            frame.buffer_mut(),
            &mut self.transcript,
            &self.chat_line_map,
            draw_area,
        );
        self.perf_selection_ms = selection_started.elapsed().as_secs_f64() * 1000.0;

        // Thin scrollbar: thumb position from bottom-based offset.
        if total > visible && area.width > 0 {
            let track_h = area.height as usize;
            let thumb_h = ((visible * track_h) / total).max(1);
            let max_off = total - visible;
            let from_top = max_off.saturating_sub(offset);
            let thumb_y = if max_off == 0 {
                0
            } else {
                (from_top * track_h.saturating_sub(thumb_h)) / max_off
            };
            for row_i in 0..track_h {
                let on = row_i >= thumb_y && row_i < thumb_y + thumb_h;
                if let Some(cell) = frame.buffer_mut().cell_mut((
                    area.x + area.width.saturating_sub(1),
                    area.y + row_i as u16,
                )) {
                    cell.set_symbol(if on { "▐" } else { " " });
                    cell.set_style(if on {
                        Style::default().fg(theme::DIM())
                    } else {
                        Style::default()
                    });
                }
            }
        }
    }

    /// Rebuild the live tail into `tail_buf`, reusing cached stream/thinking
    /// wraps when only the spinner meta line changes between ticks.
    fn rebuild_tail(&mut self, width: u16) -> usize {
        let width_changed = width != self.tail_width;
        let prose_width = width.saturating_sub(2).max(8);
        if width_changed {
            self.tail_width = width;
            self.thinking_source.clear();
        }
        // Incremental markdown: completed lines are highlighted exactly once
        // and a resize only re-wraps, so a long streamed code block stays O(n)
        // over the reply instead of re-highlighting the whole buffer on every
        // newline. `stream_changed` gates the tail_buf rebuild below.
        let mut stream_changed = false;
        if self.stream_text.is_empty() {
            if self.stream_md.text_len() != 0 || !self.stream_wrapped.is_empty() {
                self.stream_md.clear();
                self.stream_wrapped.clear();
                stream_changed = true;
            }
        } else if width_changed || self.stream_md.text_len() != self.stream_text.len() {
            self.stream_md.update(&self.stream_text, prose_width);
            self.stream_md.copy_into(&mut self.stream_wrapped);
            stream_changed = true;
        }

        let mut thinking_changed = false;
        if self.show_thinking && !self.thinking_tail.is_empty() {
            if self.thinking_tail != self.thinking_source {
                thinking_changed = true;
                self.thinking_source = self.thinking_tail.clone();
                let dim = Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC);
                let raw: Vec<Line<'static>> = self
                    .thinking_tail
                    .lines()
                    .map(|l| Line::from(Span::styled(l.to_string(), dim)))
                    .collect();
                self.thinking_wrapped = wrap_lines(&raw, prose_width);
            }
        } else if !self.thinking_wrapped.is_empty() || !self.thinking_source.is_empty() {
            thinking_changed = true;
            self.thinking_wrapped.clear();
            self.thinking_source.clear();
        }

        let content_changed = stream_changed || thinking_changed;

        if content_changed {
            self.tail_buf.clear();
            self.tail_buf.extend(self.thinking_wrapped.iter().cloned().map(|mut line| {
                line.spans.insert(
                    0,
                    Span::styled("◌ ", Style::default().fg(theme::DIM())),
                );
                line
            }));
            self.tail_buf.extend(self.stream_wrapped.iter().cloned().map(|mut line| {
                line.spans.insert(
                    0,
                    Span::styled("│ ", Style::default().fg(theme::BORDER())),
                );
                line
            }));
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
                Span::styled(SPINNER[self.spinner_i].to_string(), Style::default().fg(theme::ACCENT())),
                Span::styled(meta, Style::default().fg(theme::DIM())),
            ]));
        }
        // Queued messages render in dedicated chrome above the composer.
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
        let (model, approvals) = {
            let s = self.core.settings.lock().unwrap();
            (s.model.clone(), s.approval_mode.clone())
        };
        let width = area.width as usize;
        let left = format!(" {}", self.status_hint());
        let short_model = extensions::short_model(&model);
        let context = self
            .budget
            .map(|(used, total)| {
                format!(
                    "ctx {}%",
                    (used as f64 / total.max(1) as f64 * 100.0) as u32
                )
            })
            .unwrap_or_else(|| "ctx 0%".into());
        let right = if width >= 78 {
            format!("● {short_model} · {context} · {approvals} ")
        } else if width >= 54 {
            format!("● {short_model} · {approvals} ")
        } else if width < 4 {
            "●".into()
        } else {
            let model_width = width
                .saturating_sub(3)
                .min(width.saturating_div(2).max(8));
            format!("● {} ", clip(short_model, model_width))
        };
        let right_len = right.chars().count().min(width);
        let left_max = width.saturating_sub(right_len + 1);
        let left = clip(&left, left_max);
        let padding = width.saturating_sub(left.chars().count() + right_len);
        let dot_color = if self.running {
            theme::WARN()
        } else {
            theme::OK()
        };
        let right_tail = right.strip_prefix('●').unwrap_or(&right).to_string();
        self.status_line = Line::from(vec![
            Span::styled(left, Style::default().fg(theme::DIM())),
            Span::raw(" ".repeat(padding)),
            Span::styled("●", Style::default().fg(dot_color)),
            Span::styled(right_tail, Style::default().fg(theme::DIM())),
        ]);
        Paragraph::new(self.status_line.clone()).render(area, frame.buffer_mut());
    }

    fn status_hint(&self) -> &'static str {
        if self.transcript.has_text_selection() {
            "y copy selection · esc clear"
        } else if self.pending_approval.is_some() {
            "y allow once · a allow for run · n deny"
        } else if self.history_search.is_some() {
            "↑↓ pick · enter insert · esc close"
        } else if self.scroll_search.is_some() {
            "↑↓ match · enter jump · esc close"
        } else if self.completion.is_some() {
            "↑↓ select · tab accept · esc close"
        } else if self.focus == Focus::Scrollback {
            "j/k block · enter fold · y copy"
        } else if self.running {
            "enter queue · esc cancel"
        } else if self.transcript.offset() > 0 {
            "esc follow · pgup/pgdn scroll"
        } else if self.quit_armed {
            "ctrl+c again to quit"
        } else {
            "/ commands · @ files · drag to select"
        }
    }
}

fn command_parts(command: &str) -> (&str, &str) {
    let command = command.trim();
    let head_end = command
        .find(char::is_whitespace)
        .unwrap_or(command.len());
    let head = &command[..head_end];
    let rest = command[head_end..].trim();
    (head, rest)
}

fn save_model_selection(
    data_dir: &std::path::Path,
    current: &config::Settings,
    provider: Option<String>,
    model: String,
) -> Result<config::Settings, String> {
    let mut next = current.clone();
    next.provider = provider;
    next.model = model;
    config::save(data_dir, &next)?;
    Ok(next)
}

/// Single source of truth for `/help` and onboarding copy.
const HELP_KEYS: &[(&str, &str)] = &[
    ("enter", "send · shift+enter or alt+enter for a newline"),
    ("enter while working", "queue the message for after this turn"),
    ("tab", "focus conversation ↔ composer"),
    ("↑↓ / j k in history", "select a block · enter fold · y copy"),
    ("[ ] in history", "jump to previous or next user turn (shift+↑↓ too)"),
    ("g / G in history", "top of scrollback · follow latest"),
    ("/ at the start", "command menu · tab or enter completes"),
    ("@", "mention a project file (fuzzy search)"),
    ("ctrl+r", "search prompt history"),
    ("ctrl+f", "find in conversation"),
    ("n / N after find", "next or previous match in scrollback"),
    ("esc", "cancel turn · follow latest · return to composer"),
    ("wheel · pgup/pgdn", "scroll the conversation"),
    ("mouse drag", "select transcript text · y or ctrl+shift+c copies"),
    ("ctrl+o / o", "expand the last tool block"),
    ("ctrl+t", "show or hide model thinking"),
    ("ctrl+c ctrl+c", "quit (the model server keeps running)"),
    ("/<template> [args]", "run a prompt template from .agents/prompts/<name>.md"),
];

fn history_search_lines(
    query: &str,
    selected: usize,
    items: &[String],
    width: u16,
) -> Vec<Line<'static>> {
    let width = width as usize;
    let mut lines = vec![Line::from(vec![
        Span::styled("⌕ ", Style::default().fg(theme::ACCENT())),
        Span::styled(
            if query.is_empty() {
                "history…".to_string()
            } else {
                query.to_string()
            },
            if query.is_empty() {
                Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC)
            } else {
                Style::default()
            },
        ),
    ])];
    if items.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no matches",
            Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
        )));
        return lines;
    }
    let visible = items.len().min(6);
    let first = selected.saturating_sub(visible - 1).min(items.len() - visible);
    for (i, item) in items.iter().enumerate().skip(first).take(visible) {
        let on = i == selected;
        let marker = if on {
            Span::styled("▸ ", Style::default().fg(theme::ACCENT()))
        } else {
            Span::raw("  ")
        };
        let style = if on {
            Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::DIM())
        };
        let one_line = item.replace('\n', " ");
        lines.push(Line::from(vec![
            marker,
            Span::styled(clip(&one_line, width.saturating_sub(4)), style),
        ]));
    }
    lines
}

fn scroll_search_lines(
    query: &str,
    selected: usize,
    matches: &[usize],
    transcript: &Transcript,
    width: u16,
) -> Vec<Line<'static>> {
    let width = width as usize;
    let count = if matches.is_empty() {
        "0/0".to_string()
    } else {
        format!("{}/{}", selected + 1, matches.len())
    };
    let mut lines = vec![Line::from(vec![
        Span::styled("⌕ ", Style::default().fg(theme::ACCENT())),
        Span::styled(
            if query.is_empty() {
                "find in conversation…".to_string()
            } else {
                query.to_string()
            },
            if query.is_empty() {
                Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC)
            } else {
                Style::default()
            },
        ),
        Span::raw("  "),
        Span::styled(count, Style::default().fg(theme::DIM())),
    ])];
    if matches.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no matches",
            Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
        )));
        return lines;
    }
    let visible = matches.len().min(6);
    let first = selected
        .saturating_sub(visible - 1)
        .min(matches.len() - visible);
    for (i, &bi) in matches.iter().enumerate().skip(first).take(visible) {
        let on = i == selected;
        let marker = if on {
            Span::styled("▸ ", Style::default().fg(theme::ACCENT()))
        } else {
            Span::raw("  ")
        };
        let style = if on {
            Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::DIM())
        };
        let preview = transcript
            .block_preview(bi, query)
            .unwrap_or_else(|| format!("block {bi}"));
        let one_line = preview.replace('\n', " ");
        lines.push(Line::from(vec![
            marker,
            Span::styled(clip(&one_line, width.saturating_sub(4)), style),
        ]));
    }
    lines
}

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {k:<10}"), Style::default().fg(theme::ACCENT())),
        Span::raw(v.to_string()),
    ])
}

fn clip(s: &str, max: usize) -> String {
    if max == 0 {
        String::new()
    } else if s.chars().count() <= max {
        s.to_string()
    } else if max == 1 {
        "…".into()
    } else {
        format!("{}…", s.chars().take(max - 1).collect::<String>())
    }
}

/// Replay shows a short tool-output preview, not the full persisted payload.
fn truncate_replay_output(output: &str) -> String {
    const MAX_LINES: usize = 10;
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= MAX_LINES {
        output.to_string()
    } else {
        format!("{}\n…", lines[..MAX_LINES].join("\n"))
    }
}

fn paint_text_selection(
    buffer: &mut ratatui::buffer::Buffer,
    transcript: &mut Transcript,
    line_map: &[Option<usize>],
    area: Rect,
) {
    for (row, line_idx) in line_map.iter().copied().enumerate() {
        let Some(line_idx) = line_idx else {
            continue;
        };
        let Some((start_col, end_col)) = transcript.selection_columns(line_idx) else {
            continue;
        };
        let max_col = end_col.min(area.width as usize);
        for column in start_col.min(max_col)..max_col {
            if let Some(cell) =
                buffer.cell_mut((area.x + column as u16, area.y + row as u16))
            {
                cell.set_bg(theme::SELECT());
            }
        }
    }
}

const APPROVAL_LABELS: [&str; 3] =
    ["▸ [y] Allow once", "   [a] Allow for run", "   [n] Deny"];

fn approval_card_lines(
    name: &str,
    summary: &str,
    detail: &str,
    width: u16,
) -> Vec<Line<'static>> {
    let width = width as usize;
    let body = if detail.is_empty() { summary } else { detail };
    vec![
        Line::from(vec![
            Span::styled(
                tool_card::human_name(name),
                Style::default()
                    .fg(theme::WARN())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                clip(summary, width.saturating_sub(name.len() + 2)),
                Style::default(),
            ),
        ]),
        Line::from(Span::styled(
            clip(body, width),
            Style::default().fg(theme::DIM()),
        )),
        Line::from(vec![
            Span::styled(
                APPROVAL_LABELS[0],
                Style::default()
                    .fg(theme::WARN())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(APPROVAL_LABELS[1], Style::default().fg(theme::DIM())),
            Span::styled(APPROVAL_LABELS[2], Style::default().fg(theme::DIM())),
        ]),
    ]
}

fn approval_hit_regions(inner: Rect) -> [Option<Rect>; 3] {
    let mut hits = [None; 3];
    if inner.height < 3 {
        return hits;
    }
    let mut x = inner.x;
    for (index, label) in APPROVAL_LABELS.iter().enumerate() {
        let width = label
            .chars()
            .count()
            .min(inner.right().saturating_sub(x) as usize);
        if width > 0 {
            hits[index] = Some(Rect {
                x,
                y: inner.y + 2,
                width: width as u16,
                height: 1,
            });
        }
        x = x.saturating_add(width as u16);
    }
    hits
}

fn rect_contains(rect: Rect, x: u16, y: u16) -> bool {
    x >= rect.x && x < rect.right() && y >= rect.y && y < rect.bottom()
}

#[cfg(test)]
mod tests {
    use super::{
        approval_card_lines, approval_hit_regions, command_parts, paint_text_selection,
        rect_contains, save_model_selection, App, Dirty,
    };
    use open_max_core::config;
    use open_max_core::state::Core;
    use open_max_core::types::AgentEvent;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::Modifier;
    use ratatui::text::Line;
    use ratatui::widgets::{Paragraph, Widget};
    use ratatui::Terminal;
    use serde_json::json;
    use std::fs;
    use tokio::sync::mpsc;

    use crate::theme;
    use crate::ui::transcript::Transcript;

    fn app_fixture() -> (App, std::path::PathBuf) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("openmax-app-render-{nonce}"));
        let (core, _rx) = Core::new(dir.clone());
        let (hf_tx, _hf_rx) = mpsc::unbounded_channel();
        let (files_tx, _files_rx) = mpsc::unbounded_channel();
        let app = App::new(core, dir.clone(), "sample-project".into(), hf_tx, files_tx);
        (app, dir)
    }

    fn render_app(app: &mut App, width: u16, height: u16) -> Buffer {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        terminal.backend().buffer().clone()
    }

    fn rows(buffer: &Buffer) -> Vec<String> {
        (buffer.area.y..buffer.area.bottom())
            .map(|y| {
                (buffer.area.x..buffer.area.right())
                    .map(|x| buffer[(x, y)].symbol())
                    .collect()
            })
            .collect()
    }

    fn buffer_text(buffer: &Buffer) -> String {
        rows(buffer).join("\n")
    }

    #[test]
    fn dirty_default_is_clean() {
        let d = Dirty::default();
        assert!(!d.any());
        assert!(!d.chat && !d.tail && !d.chrome && !d.selection);
    }

    #[test]
    fn dirty_all_sets_every_region() {
        let d = Dirty::all();
        assert!(d.any());
        assert!(d.chat && d.tail && d.chrome && d.selection);
    }

    #[test]
    fn mark_chat_also_marks_tail() {
        let mut d = Dirty::default();
        d.mark_chat();
        assert!(d.chat);
        assert!(d.tail);
        assert!(!d.chrome);
        assert!(d.any());
    }

    #[test]
    fn mark_tail_is_isolated() {
        let mut d = Dirty::default();
        d.mark_tail();
        assert!(!d.chat);
        assert!(d.tail);
        assert!(!d.chrome);
    }

    #[test]
    fn mark_chrome_is_isolated() {
        let mut d = Dirty::default();
        d.mark_chrome();
        assert!(!d.chat);
        assert!(!d.tail);
        assert!(d.chrome);
    }

    #[test]
    fn mark_selection_redraws_overlay_and_status_without_rebuilding_chat() {
        let mut d = Dirty::default();
        d.mark_selection();
        assert!(!d.chat);
        assert!(!d.tail);
        assert!(d.chrome);
        assert!(d.selection);
        assert!(d.any());
    }

    #[test]
    fn clear_resets_all_flags() {
        let mut d = Dirty::all();
        d.clear();
        assert!(!d.any());
        assert_eq!(d, Dirty::default());
    }

    #[test]
    fn any_true_when_only_one_region_set() {
        let mut d = Dirty::default();
        d.mark_tail();
        assert!(d.any());
        d.clear();
        d.mark_chrome();
        assert!(d.any());
        d.clear();
        d.mark_chat();
        assert!(d.any());
    }

    #[test]
    fn model_selection_persists_provider_and_complete_id() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("openmax-model-save-{nonce}"));
        fs::create_dir_all(&dir).unwrap();
        let current = config::Settings::default();
        let exact = "openrouter/vendor/family/model".to_string();
        let saved = save_model_selection(
            &dir,
            &current,
            Some("openrouter".into()),
            exact.clone(),
        )
        .unwrap();
        assert_eq!(saved.provider.as_deref(), Some("openrouter"));
        assert_eq!(saved.model, exact);
        assert_eq!(saved.mlx_model, current.mlx_model);
        let disk = config::load(&dir);
        assert_eq!(disk.provider.as_deref(), Some("openrouter"));
        assert_eq!(disk.model, exact);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn command_parser_preserves_the_complete_trimmed_model_id() {
        assert_eq!(
            command_parts("  model   openrouter/vendor/family/model  "),
            ("model", "openrouter/vendor/family/model")
        );
        assert_eq!(command_parts("model"), ("model", ""));
    }

    #[test]
    fn model_selection_failure_leaves_current_settings_unchanged() {
        let current = config::Settings::default();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let missing = std::env::temp_dir()
            .join(format!("openmax-model-save-missing-{nonce}"))
            .join("nested");
        let result = save_model_selection(
            &missing,
            &current,
            Some("other".into()),
            "other/model".into(),
        );
        assert!(result.is_err());
        assert!(current.provider.is_none());
        assert_eq!(current.model, config::Settings::default().model);
    }

    #[test]
    fn approval_card_has_focused_default_and_clipped_detail() {
        let lines = approval_card_lines(
            "bash",
            "run tests",
            "cargo test with a deliberately long trailing argument",
            24,
        );
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].spans[0].content.as_ref(), "Shell");
        assert!(lines[0].spans[0].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(lines[2].spans[0].content.as_ref(), "▸ [y] Allow once");
        let detail: String = lines[1]
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(detail.chars().count() <= 25);
    }

    #[test]
    fn approval_hit_regions_match_each_visible_choice_and_clip_narrowly() {
        let wide = approval_hit_regions(Rect::new(2, 3, 60, 3));
        for hit in wide.into_iter().flatten() {
            assert!(rect_contains(hit, hit.x, hit.y));
            assert!(!rect_contains(hit, hit.right(), hit.y));
        }
        let narrow = approval_hit_regions(Rect::new(0, 0, 12, 3));
        assert!(narrow[0].is_some());
        assert!(narrow[1].is_none());
        assert!(narrow[2].is_none());
    }

    #[test]
    fn selection_overlay_changes_only_selected_transcript_cells() {
        let mut transcript = Transcript::new();
        transcript.set_width(20);
        transcript.push_user(vec![Line::from("hello")]);
        assert!(transcript.begin_text_selection_at(0, 2));
        assert!(transcript.update_text_selection_at(0, 7));
        transcript.finish_text_selection();

        let mut lines = Vec::new();
        transcript.fill_viewport(&mut lines, 0, 1, None);
        let area = Rect::new(0, 0, 20, 1);
        let mut buffer = Buffer::empty(area);
        Paragraph::new(lines).render(area, &mut buffer);
        paint_text_selection(&mut buffer, &mut transcript, &[Some(0)], area);

        assert_eq!(buffer[(1, 0)].bg, theme::USER_BG());
        assert_eq!(buffer[(2, 0)].bg, theme::SELECT());
        assert_eq!(buffer[(6, 0)].bg, theme::SELECT());
        assert_eq!(buffer[(7, 0)].bg, theme::USER_BG());
    }

    #[test]
    fn idle_layout_is_restrained_and_has_bordered_composer() {
        let (mut app, dir) = app_fixture();
        let buffer = render_app(&mut app, 96, 18);
        let text = buffer_text(&buffer);
        assert!(text.contains("◆ Open Max  sample-project"));
        assert!(text.contains("Describe a task"));
        assert!(text.contains("╭"));
        assert!(text.contains("/ commands · @ files · drag to select"));
        assert!(!text.contains("no telemetry"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn exchange_and_tool_states_have_separate_visual_planes() {
        let (mut app, dir) = app_fixture();
        app.insert_user_block("please test this");
        app.on_agent_event(AgentEvent::MessageDone {
            text: "I will inspect it.".into(),
        });
        assert_eq!(
            app.last_assistant_response.as_deref(),
            Some("I will inspect it.")
        );
        app.on_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".into(),
            name: "bash".into(),
            args: json!({"command":"cargo test"}),
        });
        let running = render_app(&mut app, 96, 20);
        let running_text = buffer_text(&running);
        assert!(running_text.contains("❯ please test this"));
        assert!(running_text.contains("│ I will inspect it."));
        assert!(running_text.contains("Shell cargo test"));
        let rendered_rows = rows(&running);
        let user_y = rendered_rows
            .iter()
            .position(|row| row.contains("❯ please test this"))
            .unwrap() as u16;
        assert_eq!(running[(0, user_y)].bg, theme::USER_BG());

        app.on_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".into(),
            ok: true,
            output: "test one ok\ntest two ok\ntest three ok\ntest four ok".into(),
        });
        let complete = render_app(&mut app, 96, 22);
        let complete_text = buffer_text(&complete);
        assert!(complete_text.contains("✓ Shell"));
        assert!(complete_text.contains("test three ok"));
        assert!(complete_text.contains("1 more line"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn approval_replaces_composer_then_restores_its_draft() {
        let (mut app, dir) = app_fixture();
        app.composer.load("keep this draft");
        app.on_agent_event(AgentEvent::ApprovalRequest {
            approval_id: "approval-1".into(),
            name: "bash".into(),
            summary: "install dependencies".into(),
            detail: "cargo fetch".into(),
        });
        let pending = render_app(&mut app, 88, 16);
        let pending_text = buffer_text(&pending);
        assert!(pending_text.contains("Approval"));
        assert!(pending_text.contains("[y] Allow once"));
        assert!(pending_text.contains("[a] Allow for run"));
        assert!(!pending_text.contains("keep this draft"));
        assert!(app.approval_hits.iter().all(Option::is_some));

        app.on_agent_event(AgentEvent::ApprovalSettled {
            approval_id: "approval-1".into(),
            outcome: "approved".into(),
        });
        let settled = render_app(&mut app, 88, 16);
        assert!(buffer_text(&settled).contains("keep this draft"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn completion_popup_and_narrow_layout_render_without_duplicate_hints() {
        let (mut app, dir) = app_fixture();
        app.composer.load("/");
        app.sync_completion();
        let wide = render_app(&mut app, 88, 18);
        let text = buffer_text(&wide);
        assert!(text.contains("/model"));
        let wide_rows = rows(&wide);
        let selected_y = wide_rows
            .iter()
            .position(|row| row.contains("/help"))
            .unwrap() as u16;
        assert_eq!(wide[(0, selected_y)].bg, theme::SURFACE());

        app.composer.load("/co");
        app.sync_completion();
        let copy_popup = render_app(&mut app, 88, 18);
        assert!(buffer_text(&copy_popup).contains("/copy"));

        app.composer.load("");
        app.sync_completion();
        let narrow = render_app(&mut app, 34, 8);
        let narrow_text = buffer_text(&narrow);
        assert!(narrow_text.contains("Open Max"));
        assert!(narrow_text.contains("Describe a task"));
        let tiny = render_app(&mut app, 12, 5);
        assert!(buffer_text(&tiny).contains("Open Max"));
        fs::remove_dir_all(dir).unwrap();
    }
}
