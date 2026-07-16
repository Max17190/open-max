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
use open_max_core::{agent, config, hf, mlx, prompt, registry, sessions};
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::Frame;
use tokio::sync::mpsc;

use crate::clipboard;
use crate::completion;
use crate::input::{Composer, ComposerAction};
use crate::theme;
use crate::ui::sessions as sessions_ui;
use crate::ui::tool_card::{self, DiffText};
use crate::ui::transcript::{wrap_lines, StreamingWrap, Term, Transcript};
use crate::ui::{context, extensions, markdown, models};

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

pub struct Args {
    pub continue_session: bool,
}

#[derive(PartialEq)]
enum Mode {
    Chat,
    Models,
    Sessions,
}

pub struct App {
    core: Arc<Core>,
    project: PathBuf,
    dir_label: String,
    session_id: Option<String>,
    mode: Mode,
    composer: Composer,
    models: models::ModelsState,
    sessions_panel: Option<sessions_ui::SessionsState>,
    transcript: Transcript,
    focus: Focus,
    completion: Option<completion::Popup>,
    /// Ctrl+R history search: filter text + selected index into matches.
    history_search: Option<(String, usize, Vec<String>)>,
    /// Project files for @-mentions; rescanned when a fresh `@` opens.
    file_index: Option<Arc<Vec<String>>>,
    file_index_pending: bool,
    /// Messages typed while the agent works, sent in order after the turn.
    queued: Vec<String>,
    flush_queue: bool,

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
    /// Prompt-cache hit rate of the last completion, from server usage.
    cache_pct: Option<u8>,
    quit_armed: bool,
    spinner_i: usize,
    tick_i: u64,
    page_h: u16,

    hf_tx: mpsc::UnboundedSender<(String, u64)>,
    files_tx: mpsc::UnboundedSender<Vec<String>>,
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
    status_cache: Option<u8>,
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
    let (files_tx, mut files_rx) = mpsc::unbounded_channel();
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
        sessions_panel: None,
        transcript: Transcript::new(),
        focus: Focus::Composer,
        completion: None,
        history_search: None,
        file_index: None,
        file_index_pending: false,
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
        budget: None,
        cache_pct: None,
        quit_armed: false,
        spinner_i: 0,
        tick_i: 0,
        page_h: 10,
        hf_tx,
        files_tx,
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
        status_cache: None,
        status_scrolled: false,
        status_quit_armed: false,
        status_line: Line::default(),
    };

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
                        app.needs_redraw = true;
                        draw_deadline = Some(Instant::now() + RESIZE_DEBOUNCE);
                    }
                    Some(e) => app.on_term_event(e).await?,
                    None => app.should_quit = true,
                }
            }
            Some((repo, bytes)) = hf_rx.recv() => {
                app.models.set_remote_size(&repo, bytes);
                app.needs_redraw = true;
            }
            Some(files) = files_rx.recv() => {
                app.file_index = Some(Arc::new(files));
                app.file_index_pending = false;
                app.sync_completion();
                app.needs_redraw = true;
            }
            _ = tick.tick(), if app.tick_armed() => app.on_tick().await,
            _ = tokio::time::sleep_until(
                draw_deadline.unwrap_or_else(Instant::now).into()
            ), if draw_deadline.is_some() => {}
        }
        if app.should_quit {
            break;
        }
        if app.needs_redraw {
            let now = Instant::now();
            let deferred = draw_deadline.is_some_and(|d| now < d);
            if !deferred && now.duration_since(last_draw) >= MIN_DRAW_INTERVAL {
                draw_frame(&mut terminal, &mut app)?;
                last_draw = now;
                draw_deadline = None;
                app.needs_redraw = false;
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
    crossterm::queue!(terminal.backend_mut(), crossterm::terminal::BeginSynchronizedUpdate)?;
    terminal.draw(|f| app.draw(f))?;
    crossterm::queue!(terminal.backend_mut(), crossterm::terminal::EndSynchronizedUpdate)?;
    terminal.backend_mut().flush()?;
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
        } else {
            self.note(
                "your endpoint · /tools · /skills · /context · type while the agent works to queue",
            );
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
        self.pending_diffs.clear();
        self.tool_meta.clear();
        self.last_tool_output = None;
        self.budget = None;
        self.cache_pct = None;
        self.completion = None;
        self.history_search = None;
        self.focus = Focus::Composer;
        self.queued.clear();
        self.flush_queue = false;
        self.stream_wrap.clear();
        self.thinking_wrapped.clear();
        self.thinking_source.clear();
        self.tail_width = 0;
        self.tail_content_len = 0;
        self.tail_stream_len = 0;
        self.tail_buf.clear();
        self.transcript.follow();
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
                    self.sync_completion();
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
            self.open_history_search();
            return Ok(());
        }

        if self.mode == Mode::Models {
            self.on_models_key(key).await;
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
                let items = match kind {
                    completion::Kind::Slash => completion::slash_items(&query),
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
        if let Some(cmd) = text.strip_prefix('/') {
            self.slash(cmd).await;
            return Ok(());
        }
        if self.running {
            // Queue instead of refusing: the message goes out, in order, as
            // soon as the current turn finishes. Esc cancels and hands the
            // queue back to the composer.
            self.queued.push(text);
            self.transcript.follow();
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
                let block = HELP_KEYS
                    .iter()
                    .map(|(k, v)| {
                        Line::from(vec![
                            Span::styled(format!("  {k:<32}"), Style::default().fg(theme::ACCENT())),
                            Span::styled((*v).to_string(), Style::default().fg(theme::DIM())),
                        ])
                    })
                    .collect();
                self.transcript.push(block);
            }
            "theme" => match rest.first().map(|s| s.to_ascii_lowercase()).as_deref() {
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
            },
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
                let endpoint = extensions::display_base_url(&s.base_url);
                let host = extensions::endpoint_host(&s.base_url)
                    .unwrap_or_else(|| endpoint.clone());
                let block = vec![
                    kv("model", &s.model),
                    kv("endpoint", &endpoint),
                    kv("host", &host),
                    kv("server", &server),
                    kv("approvals", &s.approval_mode),
                    kv("context", &format!("{ctx} of {} tokens", s.context_tokens)),
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
        self.needs_redraw = true;
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
                    self.transcript.push_assistant(markdown::render(
                        &text,
                        markdown::highlighter(),
                    ));
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
            AgentEvent::Usage { prompt_tokens, cached_tokens, .. } => {
                self.cache_pct = match cached_tokens {
                    Some(c) if prompt_tokens > 0 => {
                        Some(((c as f64 / prompt_tokens as f64) * 100.0).round() as u8)
                    }
                    _ => None,
                };
            }
            AgentEvent::SubagentProgress { call_id: _, kind, tool, step } => {
                // Update the running tool card breadcrumb without adding transcript weight.
                if let Some((name, summary)) = self.running_tool.as_mut() {
                    if name == "task" {
                        *summary = format!("{kind} · step {step}: {tool}");
                        self.needs_redraw = true;
                    }
                }
            }
            AgentEvent::ToolStart { call_id, name, args } => {
                let summary = registry::summarize_call(&name, &args);
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
                let compact =
                    tool_card::tool_block(&name, &summary, ok, &output, diff.as_ref());
                self.transcript.push_tool(compact, output.clone());
                self.last_tool_output = Some(output);
                self.running_tool = None;
            }
            AgentEvent::ApprovalRequest { approval_id, name, summary } => {
                self.pending_approval = Some((approval_id, name, summary));
                self.completion = None;
            }
            AgentEvent::Done { stop_reason } => {
                self.running = false;
                self.running_tool = None;
                self.pending_approval = None;
                match stop_reason.as_str() {
                    "stop" | "tool_calls" => self.push_turn_stats(),
                    "cancelled" => self.note("cancelled"),
                    "length" => self.note("stopped: hit the response token limit"),
                    "max_iterations" => self.note("stopped: reached the tool-call limit for one turn (send a follow-up to continue)"),
                    "error" => {}
                    other => self.note(&format!("stopped: {other}")),
                }
                if !self.queued.is_empty() {
                    // An interrupted or failed turn returns the queue to the
                    // composer instead of firing blind into a broken state.
                    if matches!(stop_reason.as_str(), "cancelled" | "error") {
                        self.return_queue_to_composer();
                    } else {
                        self.flush_queue = true;
                    }
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

    fn tick_armed(&self) -> bool {
        self.running || self.models.download.is_some() || self.mode == Mode::Models
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
                Span::styled("❯ ", Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD))
            } else {
                Span::raw("  ")
            };
            lines.push(Line::from(vec![
                prefix,
                Span::styled(l.to_string(), Style::default().add_modifier(Modifier::BOLD)),
            ]));
        }
        self.transcript.push_user(lines);
    }

    fn note(&mut self, text: &str) {
        if self.mode == Mode::Models {
            self.models.footer = Some((text.to_string(), false));
        } else {
            self.transcript.push(vec![Line::from(Span::styled(
                text.to_string(),
                Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
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
                    Style::default().fg(theme::ERR()),
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

        // Clamp every band so the chrome never exceeds the screen, even on
        // tiny terminals (rendering outside the buffer panics).
        let status_h = 1u16.min(area.height);
        let hints_h = 1u16.min(area.height.saturating_sub(status_h));
        let header_h = 1u16.min(area.height.saturating_sub(status_h + hints_h + 2));
        let approval_h = if self.pending_approval.is_some() {
            1u16.min(area.height.saturating_sub(header_h + status_h + hints_h))
        } else {
            0
        };
        let queue_h = if self.queued.is_empty() {
            0
        } else {
            (self.queued.len() as u16)
                .min(3)
                .min(area.height.saturating_sub(header_h + status_h + hints_h + approval_h + 2))
        };
        let hist_lines = self.history_search.as_ref().map(|(q, sel, items)| {
            history_search_lines(q, *sel, items, area.width)
        });
        let popup_lines = if hist_lines.is_some() {
            None
        } else {
            self.completion.as_ref().map(|p| {
                let indexing = p.kind == completion::Kind::File && self.file_index.is_none();
                completion::render_lines(p, area.width, indexing)
            })
        };
        let popup_h = hist_lines
            .as_ref()
            .or(popup_lines.as_ref())
            .map(|l| l.len() as u16)
            .unwrap_or(0)
            .min(area.height.saturating_sub(header_h + status_h + hints_h + approval_h + queue_h + 2));
        let composer_h = self
            .composer
            .height()
            .min(area.height.saturating_sub(
                header_h + status_h + hints_h + approval_h + queue_h + popup_h,
            ))
            .max(u16::from(
                area.height > header_h + status_h + hints_h + approval_h + queue_h + popup_h,
            ));
        let chrome =
            header_h + approval_h + queue_h + popup_h + composer_h + hints_h + status_h;
        let chat_h = area.height.saturating_sub(chrome);
        self.page_h = chat_h.saturating_sub(1).max(1);

        // Top→bottom: [header][chat][approval][queue][popup][composer][hints][status].
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
        let approval_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: approval_h,
        };
        y += approval_h;
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
        let composer_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: composer_h,
        };
        y += composer_h;
        let hints_area = Rect {
            x: area.x,
            y,
            width: area.width,
            height: hints_h,
        };
        y += hints_h;
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
        if let Some((_, name, summary)) = &self.pending_approval {
            let budget = area.width.saturating_sub(36) as usize;
            Paragraph::new(Line::from(vec![
                Span::styled("▎", Style::default().fg(theme::WARN())),
                Span::styled(
                    " approve ",
                    Style::default().fg(theme::WARN()).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    name.clone(),
                    Style::default()
                        .fg(theme::WARN())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(clip(summary, budget), Style::default().fg(theme::DIM())),
                Span::styled("  [y]es [n]o [a]lways", Style::default().fg(theme::DIM())),
            ]))
            .render(approval_area, frame.buffer_mut());
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

        if let Some(lines) = hist_lines.or(popup_lines) {
            if popup_h > 0 {
                Paragraph::new(lines).render(popup_area, frame.buffer_mut());
            }
        }

        let (composer_lines, cx, cy) = self.composer.render(composer_h);
        Paragraph::new(composer_lines).render(composer_area, frame.buffer_mut());
        if self.pending_approval.is_none()
            && self.focus == Focus::Composer
            && self.history_search.is_none()
        {
            frame.set_cursor_position(Position::new(composer_area.x + cx, composer_area.y + cy));
        }

        if hints_h > 0 {
            Paragraph::new(self.contextual_hints())
                .render(hints_area, frame.buffer_mut());
        }
        self.draw_status(frame, status_area);
    }

    fn contextual_hints(&self) -> Line<'static> {
        let text = if self.pending_approval.is_some() {
            "y approve · n deny · a always"
        } else if self.history_search.is_some() {
            "↑↓ pick · enter insert · esc cancel"
        } else if self.completion.is_some() {
            "↑↓ select · tab/enter accept · esc close"
        } else if self.focus == Focus::Scrollback {
            "j/k block · [/] turn · g/G top/end · enter fold · y copy"
        } else if self.running {
            "enter queues · esc cancel · tab history · ctrl+r search"
        } else if self.transcript.offset() > 0 {
            "esc follow · tab browse · pgup/pgdn scroll"
        } else {
            "enter send · /tools · /skills · @ file · tab history"
        };
        Line::from(Span::styled(
            format!(" {text}"),
            Style::default().fg(theme::DIM()),
        ))
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let version = env!("CARGO_PKG_VERSION");
        let (model, base_url, approvals) = {
            let s = self.core.settings.lock().unwrap();
            (s.model.clone(), s.base_url.clone(), s.approval_mode.clone())
        };
        let model = extensions::short_model(&model);
        let host = extensions::endpoint_host(&base_url).unwrap_or_else(|| "endpoint".into());
        let endpoint = format!("{model}@{host}");
        let max_ep = (area.width as usize / 3).clamp(12, 36);
        let endpoint = clip(&endpoint, max_ep);
        let line = Line::from(vec![
            Span::styled(
                "◆ openmax",
                Style::default()
                    .fg(theme::ACCENT())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  v{version} · {} · {endpoint} · {approvals}", self.dir_label),
                Style::default().fg(theme::DIM()),
            ),
        ]);
        Paragraph::new(line).render(area, frame.buffer_mut());
    }

    /// Finished transcript plus the live tail, bottom anchored, honoring the
    /// scroll offset (0 follows the latest output).
    fn draw_chat(&mut self, frame: &mut Frame, area: Rect) {
        let content_w = area.width.saturating_sub(1).max(8);
        self.transcript.set_width(content_w);
        let tail_len = self.rebuild_tail(content_w);

        let hist_len = self.transcript.len();
        let total = hist_len + tail_len;
        let visible = area.height as usize;
        self.transcript.clamp_offset(total.saturating_sub(visible));
        let offset = self.transcript.offset();

        let end = total - offset;
        let start = end.saturating_sub(visible);

        let sticky = if offset > 0 {
            self.transcript.sticky_user_line(start)
        } else {
            None
        };
        let focus_scroll = self.focus == Focus::Scrollback;

        // Collect indices first, then clone lines (avoids borrow fights).
        self.chat_buf.clear();
        if let Some(s) = sticky {
            let mut spans = vec![Span::styled("┊ ", Style::default().fg(theme::DIM()))];
            spans.extend(s.spans.iter().cloned());
            self.chat_buf.push(Line::from(spans));
        }
        let budget = visible.saturating_sub(self.chat_buf.len());
        let mut idx = start;
        let mut taken = 0usize;
        while taken < budget && idx < end {
            if idx < hist_len {
                let selected = focus_scroll && self.transcript.is_selected_block_for_line(idx);
                let mut line = self.transcript.lines()[idx].clone();
                if selected {
                    line.spans
                        .insert(0, Span::styled("▌", Style::default().fg(theme::ACCENT())));
                }
                self.chat_buf.push(line);
            } else {
                let ti = idx - hist_len;
                if ti < self.tail_buf.len() {
                    self.chat_buf.push(self.tail_buf[ti].clone());
                }
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
        Paragraph::new(self.chat_buf.as_slice()).render(draw_area, frame.buffer_mut());

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
                let dim = Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC);
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
        let ready = self.core.mlx.lock().unwrap().ready;
        let (model, approvals) = {
            let s = self.core.settings.lock().unwrap();
            (s.model.clone(), s.approval_mode.clone())
        };
        let scrolled = self.transcript.offset() > 0;

        // Status is a single line; rebuild every paint (cheap).
        {
            self.status_model = model;
            self.status_approvals = approvals;
            self.status_ready = ready;
            self.status_budget = self.budget;
            self.status_cache = self.cache_pct;
            self.status_scrolled = scrolled;
            self.status_quit_armed = self.quit_armed;

            let dot_color = if self.running {
                theme::WARN()
            } else if ready {
                theme::OK()
            } else {
                theme::DIM()
            };
            let mut ctx = self
                .budget
                .map(|(u, t)| format!(" · ctx {}%", (u as f64 / t.max(1) as f64 * 100.0) as u32))
                .unwrap_or_default();
            if let Some(pct) = self.cache_pct {
                ctx.push_str(&format!(" · cache {pct}%"));
            }
            if !self.queued.is_empty() {
                ctx.push_str(&format!(" · q:{}", self.queued.len()));
            }
            if self.running {
                if let Some(started) = self.turn_started {
                    let secs = started.elapsed().as_secs();
                    ctx.push_str(&format!(" · {secs}s"));
                }
                if let (Some(started), Some(first)) = (self.turn_started, self.first_token) {
                    let ttft = first.saturating_duration_since(started).as_millis();
                    ctx.push_str(&format!(" · ttft {ttft}ms"));
                }
            }
            let short_model = extensions::short_model(&self.status_model).to_string();
            let focus = if self.focus == Focus::Scrollback {
                " · hist"
            } else {
                ""
            };
            let scrolled_suffix = if scrolled { " · ↑" } else { "" };
            // Harness does not phone home; external tools may still use the network.
            let privacy = if self.running || self.quit_armed {
                ""
            } else {
                " · no telemetry"
            };
            let right = if self.quit_armed {
                " · ctrl+c again to quit"
            } else {
                ""
            };
            self.status_line = Line::from(vec![
                Span::styled("● ", Style::default().fg(dot_color)),
                Span::styled(short_model, Style::default().fg(theme::DIM())),
                Span::styled(
                    format!(
                        "{ctx} · {}{scrolled_suffix}{focus}{privacy}{right}",
                        self.status_approvals
                    ),
                    Style::default().fg(theme::DIM()),
                ),
            ]);
        }
        Paragraph::new(self.status_line.clone()).render(area, frame.buffer_mut());
    }
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
    ("esc", "cancel turn · follow latest · return to composer"),
    ("wheel · pgup/pgdn", "scroll the conversation"),
    ("ctrl+o / o", "expand the last tool block"),
    ("ctrl+t", "show or hide model thinking"),
    ("ctrl+c ctrl+c", "quit (the model server keeps running)"),
    ("/models", "manage and serve local models"),
    ("/model <repo>", "use a specific model id"),
    ("/theme dark|light|mono|catppuccin", "switch appearance"),
    ("/approvals <auto|ask|readonly>", "how mutating tools are gated"),
    ("/new", "start a fresh session"),
    ("/resume", "pick an earlier session in this project"),
    ("/tools", "list tools frozen for this session"),
    ("/skills", "list skills frozen for this session"),
    ("/context", "prompt token costs, cache hits, and budget"),
    ("/status", "session, endpoint, and network destinations"),
    ("/logs", "recent model server logs"),
    ("/quit", "exit"),
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

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {k:<10}"), Style::default().fg(theme::ACCENT())),
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
