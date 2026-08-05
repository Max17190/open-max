//! The event loop and all interaction logic. A fullscreen session on the
//! alternate screen: transient idle branding, a scrollable conversation, and
//! a composer fixed to the terminal bottom.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{
    Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton,
    MouseEventKind,
};
use futures_util::StreamExt;
use open_max_core::state::Core;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use open_max_core::{agent, config, prompt, registry, sessions};
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
use crate::ui::{context, extensions, markdown, model_picker, ready};

/// Where keyboard focus lives in chat mode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Composer,
    Scrollback,
}

const TICK: Duration = Duration::from_millis(120);
/// Faster tick for the silent wait before the first token, where the spinner
/// and elapsed counter are the only signs of life. 50 ms reads as fluid
/// animation; once content streams, paints follow the token cadence and the
/// relaxed tick suffices. Draws in that state are tail-only and sub-ms, so
/// the cost is a fraction of a percent of one core, and only while waiting.
const WAIT_TICK: Duration = Duration::from_millis(50);
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const WHEEL_LINES: usize = 3;

/// How close together two presses on a cell count as one multi-click gesture.
const MULTI_CLICK_WINDOW: Duration = Duration::from_millis(400);
/// Paint-rate cap for high-refresh terminals. Five and a half milliseconds
/// leaves normal scheduler overhead inside a 144 Hz display interval without
/// busy-spinning. The loop remains event-driven, so idle produces no frames.
const MIN_DRAW_INTERVAL: Duration = Duration::from_micros(5_500);
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
        // Scroll and transcript changes can alter the status hint. Keeping
        // this invariant lets draw_status reuse its line on token-only frames.
        self.chrome = true;
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

/// Chat-mode geometry. The input owns the terminal's bottom edge; every
/// transient surface grows upward into the conversation plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConversationLayout {
    header: Rect,
    chat: Rect,
    queue: Rect,
    popup: Rect,
    status: Rect,
    input: Rect,
}

fn conversation_layout(
    area: Rect,
    show_header: bool,
    desired_input_h: u16,
    desired_queue_h: u16,
    desired_popup_h: u16,
) -> ConversationLayout {
    // The composer is the primary interaction surface. On short terminals it
    // keeps its requested height before the status or idle wordmark get rows.
    let input_h = desired_input_h.min(area.height);
    let mut remaining = area.height.saturating_sub(input_h);
    // A keyboard-modal surface must never be invisible. Reserve its first row
    // before passive status chrome, then let the normal layout fill it out.
    let popup_min_h = u16::from(desired_popup_h > 0 && remaining > 0);
    remaining = remaining.saturating_sub(popup_min_h);
    let status_h = u16::from(remaining > 0);
    remaining = remaining.saturating_sub(status_h);
    // The path header is persistent chrome, but on degenerate heights the
    // last row belongs to the conversation, not the address line.
    let header_h = u16::from(show_header && remaining > 1);
    remaining = remaining.saturating_sub(header_h);
    // Active completion and search surfaces own keyboard input, so they must
    // stay visible before passive queued-message previews receive rows.
    let popup_extra_h = desired_popup_h
        .saturating_sub(popup_min_h)
        .min(remaining);
    let popup_h = popup_min_h + popup_extra_h;
    remaining = remaining.saturating_sub(popup_extra_h);
    let queue_h = desired_queue_h.min(remaining);
    let chat_h = remaining.saturating_sub(queue_h);

    let header = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: header_h,
    };
    let chat = Rect {
        x: area.x,
        y: header.bottom(),
        width: area.width,
        height: chat_h,
    };
    let queue = Rect {
        x: area.x,
        y: chat.bottom(),
        width: area.width,
        height: queue_h,
    };
    let popup = Rect {
        x: area.x,
        y: queue.bottom(),
        width: area.width,
        height: popup_h,
    };
    let status = Rect {
        x: area.x,
        y: popup.bottom(),
        width: area.width,
        height: status_h,
    };
    let input = Rect {
        x: area.x,
        y: status.bottom(),
        width: area.width,
        height: input_h,
    };

    ConversationLayout {
        header,
        chat,
        queue,
        popup,
        status,
        input,
    }
}

pub struct Args {
    pub continue_session: bool,
}

#[derive(PartialEq)]
enum Mode {
    Chat,
    ModelPicker,
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
    session_id: Option<String>,
    mode: Mode,
    composer: Composer,
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
    /// The running work is a forced `/compact`, not a turn: it settles on
    /// `Compacted` (or `Error`), never on `Done`, so those arms own the
    /// state clearing a turn gets from its terminator.
    compacting: bool,
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

    files_tx: mpsc::UnboundedSender<Vec<String>>,
    should_quit: bool,
    dirty: Dirty,

    /// Live assistant stream, markdown-rendered and wrapped (matches final block).
    stream_wrapped: Vec<Line<'static>>,
    /// Complete wrapped stream lines already copied into `stream_wrapped`.
    /// The partial line after this prefix is replaced on each token.
    stream_stable_len: usize,
    /// Incremental markdown renderer for the live stream: completed lines are
    /// rendered once, only the growing tail line re-renders per token, and a
    /// resize re-wraps without re-rendering. Replaces the O(n)-per-refresh
    /// full re-render that scaled poorly on long code replies.
    stream_md: markdown::StreamingMarkdown,
    thinking_wrapped: Vec<Line<'static>>,
    thinking_source: String,
    tail_width: u16,
    tail_content_len: usize,
    /// Thinking plus complete stream lines at the front of `tail_buf`.
    /// Content after this prefix is the changing partial line and transient
    /// running metadata, so a token only rebuilds that suffix.
    tail_stable_len: usize,
    tail_buf: Vec<Line<'static>>,
    chat_buf: Vec<Line<'static>>,
    /// Lines in `chat_buf` that are sticky + history (before live tail).
    hist_prefix_len: usize,
    hist_reuse_key: Option<HistReuseKey>,
    /// Absolute transcript line for each rendered row in `chat_buf`.
    chat_line_map: Vec<Option<usize>>,
    chat_draw_area: Rect,
    /// Where the composer text last painted, so the wheel and the mouse can
    /// tell the prompt apart from the conversation above it.
    composer_draw_area: Rect,
    /// Cell, time, and running count of the last left press. Terminals report
    /// presses, never click counts, so double and triple clicks are derived
    /// here rather than delivered.
    last_click: Option<(u16, u16, Instant, u8)>,
    /// Whether the previous frame reserved the right-hand scrollbar column.
    /// Sticky so a steadily overflowing transcript wraps at one width per
    /// frame instead of re-deciding (and re-wrapping all history twice) on
    /// every paint.
    scrollbar_reserved: bool,
    /// Live-tail line count and content width at the previous frame, used to
    /// keep a scrolled-up viewport stationary as the tail grows or collapses.
    last_tail_len: usize,
    last_content_w: u16,
    /// Last state written into the terminal title (idle / working / needs
    /// approval); writes are edge-triggered.
    presence: Presence,
    approval_hits: [Option<Rect>; 3],
    perf_layout_ms: f64,
    perf_selection_ms: f64,
    header_line: Line<'static>,
    header_width: u16,
    status_line: Line<'static>,
    status_width: u16,
}

pub async fn run(
    mut terminal: Term,
    core: Arc<Core>,
    mut core_rx: mpsc::UnboundedReceiver<AgentEventEnvelope>,
    args: Args,
) -> std::io::Result<()> {
    let (files_tx, mut files_rx) = mpsc::unbounded_channel();
    let project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut app = App::new(core.clone(), project, files_tx);

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
    let mut tick_period = TICK;

    // Paint pacing: at most one frame per MIN_DRAW_INTERVAL. A redraw that
    // arrives too early is deferred to `draw_deadline` and coalesced with
    // everything else that lands before it (grok-build's cadence model).
    // An idle app has no armed tick and may receive no terminal event after
    // entering the alternate screen. Paint once before waiting so first launch
    // can never sit on a blank frame until the user presses a key.
    let mut last_draw = Instant::now();
    let mut draw_deadline: Option<Instant> = None;
    draw_frame(&mut terminal, &mut app, MIN_DRAW_INTERVAL)?;
    app.dirty.clear();
    // State the initial presence in the title; transitions are edge-driven.
    app.emit_presence_title();

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
        // Animation cadence follows state; the interval is recreated only on
        // transitions (turn start, first token, turn end), not per loop.
        let desired_tick = app.tick_period();
        if desired_tick != tick_period {
            tick_period = desired_tick;
            tick = tokio::time::interval(desired_tick);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        }
        if app.dirty.any() {
            let now = Instant::now();
            let deferred = draw_deadline.is_some_and(|d| now < d);
            if !deferred && now.duration_since(last_draw) >= MIN_DRAW_INTERVAL {
                draw_frame(&mut terminal, &mut app, now.duration_since(last_draw))?;
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
fn draw_frame(
    terminal: &mut Term,
    app: &mut App,
    frame_interval: Duration,
) -> std::io::Result<()> {
    use std::io::Write;
    let t0 = Instant::now();
    crossterm::queue!(terminal.backend_mut(), crossterm::terminal::BeginSynchronizedUpdate)?;
    terminal.draw(|f| app.draw(f))?;
    crossterm::queue!(terminal.backend_mut(), crossterm::terminal::EndSynchronizedUpdate)?;
    terminal.backend_mut().flush()?;
    if std::env::var_os("OPENMAX_PERF").is_some() {
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let interval_ms = frame_interval.as_secs_f64() * 1000.0;
        eprintln!(
            "openmax_perf frame_interval_ms={interval_ms:.3} draw_frame_ms={ms:.3} transcript_layout_ms={:.3} selection_overlay_ms={:.3}",
            app.perf_layout_ms, app.perf_selection_ms
        );
    }
    Ok(())
}


impl App {
    fn new(
        core: Arc<Core>,
        project: PathBuf,
        files_tx: mpsc::UnboundedSender<Vec<String>>,
    ) -> Self {
        Self {
            composer: Composer::new(&core.data_dir),
            core,
            project,
            session_id: None,
            pending_submit: None,
            mode: Mode::Chat,
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
            compacting: false,
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
            files_tx,
            should_quit: false,
            dirty: Dirty::all(),
            stream_wrapped: Vec::new(),
            stream_stable_len: 0,
            stream_md: markdown::StreamingMarkdown::default(),
            thinking_wrapped: Vec::new(),
            thinking_source: String::new(),
            tail_width: 0,
            tail_content_len: 0,
            tail_stable_len: 0,
            tail_buf: Vec::new(),
            chat_buf: Vec::new(),
            hist_prefix_len: 0,
            hist_reuse_key: None,
            chat_line_map: Vec::new(),
            chat_draw_area: Rect::default(),
            composer_draw_area: Rect::default(),
            last_click: None,
            scrollbar_reserved: false,
            last_tail_len: 0,
            last_content_w: 0,
            presence: Presence::Idle,
            approval_hits: [None; 3],
            perf_layout_ms: 0.0,
            perf_selection_ms: 0.0,
            header_line: Line::default(),
            header_width: u16::MAX,
            status_line: Line::default(),
            status_width: u16::MAX,
        }
    }

    async fn startup(&mut self, args: &Args) {
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
        // This sitting is a new boundary; earlier boundaries render below.
        sessions::record_resume_point(&self.core, session_id, messages.len() as u64);
        let boundaries: std::collections::HashSet<u64> = sessions::meta(&self.core, session_id)
            .map(|meta| meta.resume_points.into_iter().collect())
            .unwrap_or_default();
        for (i, m) in messages.iter().enumerate() {
            if boundaries.contains(&(i as u64)) {
                self.transcript.push(vec![Line::from(Span::styled(
                    "• resumed",
                    Style::default().fg(theme::DIM()),
                ))]);
            }
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
                            self.transcript.push_assistant(markdown::render(text));
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
                            // Diff events are not persisted, but the result
                            // text carries the counts: a replayed edit card
                            // keeps its +N −N badge instead of demoting to a
                            // bare checkmark.
                            // Only tools that actually mutate files may wear
                            // a diff badge: a read whose CONTENT happens to
                            // contain "(+N −N)" must never present fabricated
                            // modification evidence.
                            let mutating = matches!(
                                call.function.name.as_str(),
                                "write_file" | "edit_file"
                            );
                            let path = args["path"].as_str().unwrap_or("");
                            let badge = if !mutating || path.is_empty() {
                                None
                            } else {
                                parse_change_counts(content).map(|(added, removed)| {
                                    tool_card::DiffText {
                                        path: path.to_string(),
                                        diff: String::new(),
                                        added,
                                        removed,
                                    }
                                })
                            };
                            let compact = tool_card::tool_block(
                                &call.function.name,
                                &summary,
                                ok,
                                &truncate_replay_output(content),
                                badge.as_ref(),
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
        // Session-scoped like `running`: the old session's receipt is
        // filtered out once the id changes, so a flag left armed here would
        // misroute the next session's first Error into the compaction branch.
        self.compacting = false;
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
        // Typed-ahead work was never bound to the old session; the careful
        // esc path restores it, and silently clearing it here was the one
        // path that destroyed it.
        self.return_queue_to_composer();
        self.flush_queue = false;
        self.stream_wrapped.clear();
        self.stream_stable_len = 0;
        self.stream_md.clear();
        self.thinking_wrapped.clear();
        self.thinking_source.clear();
        self.tail_width = 0;
        self.tail_content_len = 0;
        self.tail_stable_len = 0;
        self.tail_buf.clear();
        self.hist_prefix_len = 0;
        self.hist_reuse_key = None;
        self.chat_line_map.clear();
        self.chat_draw_area = Rect::default();
        self.composer_draw_area = Rect::default();
        self.scrollbar_reserved = false;
        self.last_tail_len = 0;
        self.last_content_w = 0;
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
                            let clicks = self.count_click(m.column, m.row);
                            if let Some(choice) = self
                                .approval_hits
                                .iter()
                                .position(|hit| hit.is_some_and(|rect| rect_contains(rect, m.column, m.row)))
                            {
                                self.respond_approval_choice(choice);
                            } else if let Some((cell, row)) =
                                self.composer_position(m.column, m.row)
                            {
                                let area = self.composer_draw_area;
                                self.focus = Focus::Composer;
                                self.transcript.clear_text_selection();
                                self.transcript.clear_selection();
                                let (w, h) = (area.width, area.height);
                                match clicks {
                                    2 => {
                                        self.composer.select_word_at(w, h, cell, row);
                                    }
                                    3 => {
                                        self.composer.select_line_at(w, h, cell, row);
                                    }
                                    _ => self.composer.click_at(w, h, cell, row),
                                }
                                self.dirty.mark_chrome();
                                self.dirty.mark_selection();
                            } else if let Some((line, x)) =
                                self.transcript_position(m.column, m.row)
                            {
                                self.focus = Focus::Scrollback;
                                self.composer.clear_selection();
                                match clicks {
                                    2 => {
                                        self.transcript.select_word_at(line, x);
                                    }
                                    3 => {
                                        self.transcript.select_line_at(line, x);
                                    }
                                    _ => {
                                        self.transcript.begin_text_selection_at(line, x);
                                    }
                                }
                                self.dirty.mark_selection();
                            } else {
                                self.transcript.clear_text_selection();
                                self.dirty.mark_selection();
                            }
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            if self.composer.is_dragging() {
                                let area = self.composer_draw_area;
                                let (cell, row) = self.composer_drag_target(m.column, m.row);
                                self.composer.drag_to(area.width, area.height, cell, row);
                                self.dirty.mark_chrome();
                            } else if let Some((line, x)) =
                                self.transcript_position(m.column, m.row)
                            {
                                self.transcript.update_text_selection_at(line, x);
                                self.dirty.mark_selection();
                            }
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            if self.composer.is_dragging() {
                                self.composer.finish_selection();
                                self.dirty.mark_chrome();
                            } else {
                                if let Some((line, x)) =
                                    self.transcript_position(m.column, m.row)
                                {
                                    // The release of a gesture's own click
                                    // must not shrink the word or line it
                                    // picked; end_text_selection_at knows.
                                    self.transcript.end_text_selection_at(line, x);
                                } else {
                                    self.transcript.finish_text_selection();
                                }
                                self.dirty.mark_selection();
                            }
                        }
                        MouseEventKind::ScrollUp => {
                            if self.composer_position(m.column, m.row).is_some() {
                                let area = self.composer_draw_area;
                                self.composer.scroll_by(area.width, area.height, -1);
                                self.dirty.mark_chrome();
                            } else {
                                self.transcript.scroll_up(WHEEL_LINES);
                                self.dirty.mark_chat();
                            }
                        }
                        MouseEventKind::ScrollDown => {
                            if self.composer_position(m.column, m.row).is_some() {
                                let area = self.composer_draw_area;
                                self.composer.scroll_by(area.width, area.height, 1);
                                self.dirty.mark_chrome();
                            } else {
                                self.transcript.scroll_down(WHEEL_LINES);
                                self.dirty.mark_chat();
                            }
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

        // Non-short-circuiting `|`: Esc clears both selections, not whichever
        // one happens to be checked first.
        if key.code == KeyCode::Esc
            && (self.composer.clear_selection() | self.transcript.clear_text_selection())
        {
            self.dirty.mark_chrome();
            self.dirty.mark_selection();
            return Ok(());
        }

        // Ctrl+C copies a live selection, then cancels a running turn, and
        // quits on the second press. Ctrl+Shift+C is indistinguishable from
        // Ctrl+C unless the terminal speaks the kitty keyboard protocol, so
        // copy cannot hang off the shift alone: the selection is what makes
        // the press mean copy. Copying drops it, so the next press cancels.
        if ctrl && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C')) {
            if self.copy_selection() {
                self.quit_armed = false;
                return Ok(());
            }
            // On a terminal that does report the shift, Ctrl+Shift+C is a
            // distinct key that has only ever meant copy. With nothing
            // selected it stays a no-op instead of becoming the quit binding.
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                return Ok(());
            }
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
            self.dirty.mark_tail();
            // Between turns there is no thinking on screen to appear or
            // vanish, so the toggle has to say what it did.
            if !self.running {
                self.note(if self.show_thinking {
                    "thinking shown while the model streams (ctrl+t to hide)"
                } else {
                    "thinking hidden (ctrl+t to show)"
                });
            }
            return Ok(());
        }
        if ctrl && key.code == KeyCode::Char('r') && self.mode == Mode::Chat {
            if self.pending_approval.is_none() && self.completion.is_none() {
                self.scroll_search = None;
                self.open_history_search();
            }
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
                    self.core.set_run_approval_mode(config::ApprovalMode::Auto);
                    self.core.respond_approval(&id, true);
                    self.note("approvals set to auto for this run (change with /approvals)");
                }
                _ => {
                    let _ = name;
                }
            }
            return Ok(());
        }

        if self.focus == Focus::Scrollback
            && matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y'))
            && self.copy_text_selection()
        {
            return Ok(());
        }

        // Completion popup: navigation and acceptance take priority over the
        // composer; anything else falls through and refilters afterwards.
        if self.completion.is_some() {
            // Shift+Tab steps back through the list here rather than cycling
            // approval modes: an open popup owns the keyboard, and the reverse
            // of "Tab picks the next item" is the only reading of it on screen.
            if is_shift_tab(&key) {
                if let Some(popup) = &mut self.completion {
                    popup.prev();
                }
                return Ok(());
            }
            match key.code {
                KeyCode::Up => {
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
                KeyCode::Tab | KeyCode::Enter
                    if key.code == KeyCode::Tab
                        || !key
                            .modifiers
                            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
                {
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

        // Shift+Tab cycles how much the agent may do without asking. It lands
        // here, after every modal surface has had the key, because those own
        // the keyboard while they are open.
        if is_shift_tab(&key) {
            self.cycle_approval_mode();
            return Ok(());
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
                    // G follows the bottom without selecting a block; y must
                    // still mean "copy the block I am looking at".
                    if self.transcript.selected().is_none() {
                        self.transcript.select_prev();
                    }
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
            if self.transcript.offset() > 0 {
                // Reading comes first: from a scrolled-up view Esc returns
                // to the live tail, never destroys the running turn the
                // user was still reading. A second Esc at the bottom
                // cancels.
                self.transcript.follow();
                self.focus = Focus::Composer;
            } else if self.running {
                if let Some(id) = &self.session_id {
                    self.core.cancel(id);
                }
            } else if self.focus == Focus::Scrollback {
                self.focus = Focus::Composer;
                self.transcript.clear_selection();
            }
            return Ok(());
        }

        // Up in an empty composer pulls the newest queued message back for
        // editing, mirroring up-to-edit-history: the only other way to
        // amend a queued message is cancelling the whole turn.
        if key.code == KeyCode::Up && self.composer.is_empty() && !self.queued.is_empty() {
            if let Some(text) = self.queued.pop() {
                self.composer.load(&text);
            }
            self.dirty.mark_chrome();
            return Ok(());
        }

        // Typing returns focus to the composer.
        self.focus = Focus::Composer;
        match self.composer.handle_key(key) {
            ComposerAction::Submit(text) => {
                self.completion = None;
                self.handle_submit(text).await?;
            }
            ComposerAction::None => {
                self.sync_completion();
                // Composer edits and completion refilters only touch the
                // bottom chrome plane. Avoid rebuilding wrapped transcript
                // history on every printable key.
                self.dirty.mark_chrome();
            }
        }
        Ok(())
    }

    /// How many times this cell has been clicked in quick succession: 1, 2, or
    /// 3, cycling so a fourth rapid click starts over rather than sticking.
    ///
    /// A one-cell tolerance keeps a slightly drifting hand on the same word.
    fn count_click(&mut self, column: u16, row: u16) -> u8 {
        let count = match self.last_click {
            Some((cx, cy, at, n))
                if cy == row
                    && cx.abs_diff(column) <= 1
                    && at.elapsed() <= MULTI_CLICK_WINDOW =>
            {
                (n % 3) + 1
            }
            _ => 1,
        };
        self.last_click = Some((column, row, Instant::now(), count));
        count
    }

    /// Cell and row of `(column, row)` inside the composer text area, or
    /// `None` when the pointer is anywhere else (including its border).
    fn composer_position(&self, column: u16, row: u16) -> Option<(u16, u16)> {
        if !rect_contains(self.composer_draw_area, column, row) {
            return None;
        }
        Some((
            column - self.composer_draw_area.x,
            row - self.composer_draw_area.y,
        ))
    }

    /// Drag target clamped into the composer, so a selection started in the
    /// prompt keeps tracking the pointer once it leaves the box.
    fn composer_drag_target(&self, column: u16, row: u16) -> (u16, u16) {
        let area = self.composer_draw_area;
        let cell = column
            .max(area.x)
            .min(area.right().saturating_sub(1))
            .saturating_sub(area.x);
        let row = row
            .max(area.y)
            .min(area.bottom().saturating_sub(1))
            .saturating_sub(area.y);
        (cell, row)
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

    /// Copy whichever text selection is live, prompt first. Returns false when
    /// nothing is selected, so the caller can fall through to its own binding.
    fn copy_selection(&mut self) -> bool {
        if let Some(text) = self.composer.selected_text() {
            self.composer.clear_selection();
            self.dirty.mark_chrome();
            self.note_copied(&text);
            return true;
        }
        self.copy_text_selection()
    }

    fn copy_text_selection(&mut self) -> bool {
        let Some(text) = self.transcript.selected_text() else {
            return false;
        };
        // Dropping the highlight is what keeps a selection from swallowing
        // the next Ctrl+C: press once to copy, again to cancel or quit.
        self.transcript.clear_text_selection();
        self.dirty.mark_selection();
        self.note_copied(&text);
        true
    }

    fn note_copied(&mut self, text: &str) {
        if clipboard::copy_text(text) {
            self.note("copied selection");
        } else {
            self.note("copy failed (terminal may block OSC 52)");
        }
    }

    /// Cycle how much the agent may do without asking, for this run.
    ///
    /// Deliberately not persisted. `/approvals` is the path that writes
    /// settings.json, because widening the trust boundary for every future
    /// session in a project should cost a typed command. A key one slip away
    /// from `auto` must not do it quietly, so the change lives and dies with
    /// the run and the acknowledgement says exactly that. This mirrors the
    /// approval card's "allow for run", which is already run-scoped.
    fn cycle_approval_mode(&mut self) {
        let mode = self.core.approval_mode().next();
        self.core.set_run_approval_mode(mode);
        // Terse on purpose: cycling to the mode you want costs one line per
        // press, and the status line already carries the live value. It stays
        // a transcript note because below 54 columns the status line drops the
        // mode entirely, and a widened trust boundary must never be silent.
        self.note(&format!(
            "approvals: {} for this run (/approvals persists)",
            mode.as_str(),
        ));
        // The status line carries the live mode; nothing else repaints.
        self.dirty.mark_chrome();
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
                self.core.set_run_approval_mode(config::ApprovalMode::Auto);
                self.core.respond_approval(&id, true);
                self.note("approvals set to auto for this run (change with /approvals)");
            }
            2 => self.core.respond_approval(&id, false),
            _ => {}
        }
    }

    fn on_model_picker_key(&mut self, key: KeyEvent) {
        // Before the Tab arm below: where the terminal reports Shift+Tab as a
        // shifted Tab, matching on the code alone would step forward.
        if is_shift_tab(&key) {
            if let Some(picker) = &mut self.model_picker {
                picker.prev();
            }
            return;
        }
        match key.code {
            KeyCode::Esc => {
                self.mode = Mode::Chat;
                self.model_picker = None;
            }
            KeyCode::Up => {
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
                let has_match = self
                    .scroll_search
                    .as_ref()
                    .is_some_and(|(_, _, matches)| !matches.is_empty());
                if let Some((q, _, _)) = &self.scroll_search {
                    if !q.is_empty() {
                        self.scroll_search_last = Some(q.clone());
                    }
                }
                if has_match {
                    self.focus_scroll_match();
                    self.scroll_search = None;
                    self.focus = Focus::Scrollback;
                } else {
                    // Nothing to jump to: land back at the prompt. Dropping
                    // into scrollback focus after a failed search left the
                    // next typed prompt feeding silent nav bindings.
                    self.scroll_search = None;
                    self.focus = Focus::Composer;
                }
            }
            KeyCode::Tab => {
                // The explicit way OUT of the find bar and into typing:
                // every other printable key feeds the query, so a prompt
                // typed here would vanish into it.
                if let Some((q, _, _)) = &self.scroll_search {
                    if !q.is_empty() {
                        self.scroll_search_last = Some(q.clone());
                    }
                }
                self.scroll_search = None;
                self.focus = Focus::Composer;
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
        if item.submits {
            Some(self.composer.take())
        } else {
            self.sync_completion();
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
                    self.templates = open_max_core::templates::discover(&self.core.data_dir, &self.project)
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

    // ---------- submission and slash commands ----------

    async fn handle_submit(&mut self, text: String) -> std::io::Result<()> {
        let text = if let Some(cmd) = text.strip_prefix('/') {
            let head = cmd.split_whitespace().next().unwrap_or("");
            let builtin = head == "exit"
                || completion::COMMANDS.iter().any(|spec| spec.name == head);
            // Built-ins win; anything else may be a prompt template, whose
            // expansion submits as a normal user message. The expansion itself
            // is the same helper --print and --stdio submit through, so the
            // three front ends cannot drift.
            let expanded = if builtin {
                None
            } else {
                open_max_core::templates::expand_slash_line(&self.core.data_dir, &self.project, &text)
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
                self.set_presence(Presence::Working);
                self.pending_submit = Some(text);
                self.turn_started = Some(Instant::now());
                self.first_token = None;
                self.stream_chars = 0;
                self.stream_text.clear();
                self.stream_wrapped.clear();
                self.stream_stable_len = 0;
                self.stream_md.clear();
                self.tail_content_len = 0;
                self.tail_stable_len = 0;
                self.tail_buf.clear();
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
                    .map(|(key, description)| help_line(key, description))
                    .collect();
                block.push(Line::default());
                block.extend(
                    completion::COMMANDS
                        .iter()
                        .map(|spec| help_line(&spec.usage(), spec.description)),
                );
                // The user's own templates are commands too; a help screen
                // that omits them hides exactly the capability this harness
                // exists to grow. Same shadowing rule as the popup: a
                // template that collides with a built-in name never
                // dispatches, so advertising it here would be a lie.
                let templates = self.templates.iter().filter(|(name, _)| {
                    !completion::COMMANDS.iter().any(|spec| spec.name == *name)
                });
                let mut first = true;
                for (name, desc) in templates {
                    if first {
                        block.push(Line::default());
                        first = false;
                    }
                    let usage = format!("/{name}");
                    let desc: &str = if desc.is_empty() {
                        "prompt template"
                    } else {
                        desc
                    };
                    block.push(help_line(&usage, desc));
                }
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
            "approvals" => match rest.first().and_then(|m| config::ApprovalMode::parse(m)) {
                Some(mode) => {
                    {
                        let mut s = self.core.settings.lock().unwrap();
                        s.approval_mode = mode;
                        let _ = config::save(&self.core.data_dir, &s);
                    }
                    // An explicit persisted choice outranks a run override,
                    // which would otherwise keep masking it.
                    self.core.clear_run_approval_mode();
                    self.note(&format!("approvals: {}", mode.as_str()));
                }
                None => self.note("usage: /approvals auto|ask|readonly"),
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
                        Ok((tools, skills, _changes)) => self.note(&format!(
                            "re-frozen: {}, {} (prompt cache will re-prefill once)",
                            plural(tools, "tool"),
                            plural(skills, "skill")
                        )),
                        Err(e) => self.error(&e),
                    }
                }
            },
            "compact" => match &self.session_id {
                None => self.note("no session yet; nothing to compact"),
                Some(id) => {
                    let id = id.clone();
                    // Spawned by the core: the summary upgrade is a real model
                    // request, and the event loop must keep painting under it.
                    // Marked running like a turn so prompts queue instead of
                    // being refused and Esc cancels; the receipt (Compacted,
                    // or Error) settles the state Done would for a turn.
                    match agent::compact_session(&self.core, &id, &self.project) {
                        Ok(()) => {
                            self.running = true;
                            self.compacting = true;
                            self.set_presence(Presence::Working);
                            self.dirty.mark_chrome();
                            self.note("compacting…");
                        }
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
                        let dd = self.core.data_dir.clone();
                        let registry = tokio::task::spawn_blocking({
                            let project = project.clone();
                            move || registry::Registry::build(&dd, &project)
                        })
                        .await
                        .unwrap_or_else(|_| registry::Registry::builtin_only());
                        let (_, b) = prompt::system_prompt_with_breakdown(&project, &registry);
                        (b, false)
                    }
                };
                let session_cache = self
                    .session_id
                    .as_deref()
                    .map(|id| open_max_core::sessions::load_usage(&self.core, id))
                    .as_deref()
                    .and_then(open_max_core::sessions::cache_hit_totals);
                self.transcript.push(context::context_block(
                    &breakdown,
                    is_frozen,
                    self.budget,
                    self.cache_pct,
                    session_cache,
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
                    let dd = self.core.data_dir.clone();
                    let reg = tokio::task::spawn_blocking(move || registry::Registry::build(&dd, &project))
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
                    let dd = self.core.data_dir.clone();
                    let reg = tokio::task::spawn_blocking(move || registry::Registry::build(&dd, &project))
                        .await
                        .unwrap_or_else(|_| registry::Registry::builtin_only());
                    extensions::skills_block(&reg.skills, &self.project, false)
                };
                self.transcript.push(lines);
            }
            "status" => {
                let s = self.core.settings.lock().unwrap().clone();
                let ep = open_max_core::providers::resolve(&s, &self.core.data_dir);
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
                    kv("approvals", self.core.approval_mode().as_str()),
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
                        "  privacy",
                        "no telemetry · sessions stay local · external tools may use the network",
                    ),
                ];
                self.transcript.push(block);
            }
            "quit" | "exit" => self.should_quit = true,
            other => self.note(&format!(
                "unknown command: /{other} (see /help; prompt templates live in .agents/prompts/)"
            )),
        }
    }

    // ---------- core events ----------

    async fn on_core_event(&mut self, env: AgentEventEnvelope) {
        if self.session_id.as_deref() == Some(env.session_id.as_str()) {
            self.on_agent_event(env.event);
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
                    let rendered = if self.stream_text == text {
                        self.stream_md.finish(&text)
                    } else {
                        markdown::render(&text)
                    };
                    self.transcript.push_assistant(rendered);
                    self.refilter_scroll_search_live();
                }
                self.stream_text.clear();
                self.stream_wrapped.clear();
                self.stream_stable_len = 0;
                self.stream_md.clear();
                self.thinking_tail.clear();
                self.thinking_source.clear();
                self.thinking_wrapped.clear();
                self.thinking_chars = 0;
                // Caches above are already empty, so rebuild_tail would not see a
                // stream_changed edge; drop the stitched tail content here so the
                // finished assistant body is not still painted under the spinner.
                self.tail_content_len = 0;
                self.tail_stable_len = 0;
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
                let expanded = diff
                    .as_ref()
                    .map(|change| change.diff.clone())
                    .unwrap_or_else(|| output.clone());
                self.transcript.push_tool(compact, expanded.clone());
                self.last_tool_output = Some(expanded);
                self.running_tool = None;
                self.refilter_scroll_search_live();
                self.dirty.mark_chat();
            }
            AgentEvent::ApprovalRequest {
                approval_id,
                name,
                summary,
                detail,
                reason,
                source_path,
                source_sha,
            } => {
                // Interactive approvals show provenance at the moment it
                // matters: capability content no human has approved - the
                // tool's .toml, or the project code it runs, which can be
                // rewritten after the .toml was blessed - named by the file
                // and the exact bytes being approved.
                let detail = if reason == "unapproved_source" {
                    let mut head = "unapproved tool content (its .toml or the code it runs)".to_string();
                    if !source_path.is_empty() {
                        head.push_str(&format!(" · {source_path} ({source_sha})"));
                    }
                    if detail.is_empty() {
                        head
                    } else {
                        format!("{head} · {detail}")
                    }
                } else {
                    detail
                };
                self.pending_approval = Some((approval_id, name, summary, detail));
                self.set_presence(Presence::NeedsApproval);
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
                    // The turn resumes; the needs-you state is over.
                    self.set_presence(Presence::Working);
                    self.dirty.mark_chrome();
                    match outcome.as_str() {
                        "timed_out" => self.note("approval timed out · declined"),
                        "cancelled" => self.note("approval cancelled"),
                        _ => {}
                    }
                }
            }
            AgentEvent::Refrozen { tools, skills, changes } => {
                // The receipt: what changed and who changed it, so the action
                // space never mutates silently (a poisoned skill arriving via
                // git pull is announced, not slipped in).
                let detail = if changes.is_empty() {
                    String::new()
                } else {
                    let head: Vec<&str> =
                        changes.iter().take(3).map(String::as_str).collect();
                    let more = changes.len().saturating_sub(3);
                    if more > 0 {
                        format!(" - {} (+{more} more)", head.join(", "))
                    } else {
                        format!(" - {}", head.join(", "))
                    }
                };
                self.note(&format!(
                    "toolbox changed: re-frozen with {}, {}{detail}",
                    plural(tools, "tool"),
                    plural(skills, "skill"),
                ));
            }
            AgentEvent::Compacted { tokens_before, tokens_after, compacted_messages } => {
                if self.compacting {
                    self.compacting = false;
                    self.running = false;
                    self.set_presence(Presence::Idle);
                    self.dirty.mark_chrome();
                    if !self.queued.is_empty() {
                        self.flush_queue = true;
                    }
                }
                if compacted_messages == 0 {
                    self.note(&format!(
                        "already compact: ~{tokens_before} tokens is at or under the prune target"
                    ));
                } else {
                    self.note(&format!(
                        "compacted: ~{tokens_before} to ~{tokens_after} tokens, {} archived \
                         (prompt cache will re-prefill once)",
                        plural(compacted_messages, "message"),
                    ));
                }
            }
            AgentEvent::SchemasOverBudget { schema_tokens, budget_tokens } => {
                // Says what it costs and what to do, not how compaction reacts:
                // that depends on whether any room is left at all.
                self.note(&format!(
                    "installed tool schemas cost ~{schema_tokens} tokens of the ~{budget_tokens} \
                     this context window can spend: little is left for the conversation, so history \
                     is compacted early and turns may not fit at all. Remove tools (/tools, \
                     openmax --spec usage) or raise context_tokens"
                ));
            }
            AgentEvent::HookFailed { hook, event, detail } => {
                // Observe hooks are fail-open; the note keeps them honest.
                self.note(&format!("hook '{hook}' failed on {event}: {detail}"));
            }
            AgentEvent::Done { stop_reason } => {
                self.running = false;
                self.set_presence(Presence::Idle);
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
                    if matches!(stop_reason.as_str(), "cancelled" | "error" | "blocked" | "truncated") {
                        self.return_queue_to_composer();
                    } else {
                        self.flush_queue = true;
                    }
                }
            }
            AgentEvent::Error { message } => {
                if self.compacting {
                    // A failed compaction has no Done to settle it; mirror
                    // the failed-turn policy, queue back to the composer.
                    self.compacting = false;
                    self.running = false;
                    self.set_presence(Presence::Idle);
                    self.return_queue_to_composer();
                }
                self.pending_approval = None;
                self.dirty.mark_chrome();
                self.error(&message);
            }
        }
    }

    fn tick_armed(&self) -> bool {
        self.running
    }

    /// Announce a presence change in the terminal title (edge-triggered),
    /// with a bell on the needs-you edge. Best-effort raw writes, same as
    /// the OSC 52 clipboard path; never part of a frame.
    fn set_presence(&mut self, presence: Presence) {
        if self.presence == presence {
            return;
        }
        self.presence = presence;
        self.emit_presence_title();
    }

    fn emit_presence_title(&self) {
        use std::io::Write;
        let title = presence_title(self.presence, &self.project);
        let mut seq = format!("\x1b]0;{title}\x07");
        if self.presence == Presence::NeedsApproval {
            seq.push('\x07');
        }
        let mut out = std::io::stdout();
        let _ = out.write_all(seq.as_bytes()).and_then(|_| out.flush());
    }

    /// Animation cadence: fluid while the user has nothing but the spinner
    /// to watch, relaxed once content is flowing or the app is idle.
    fn tick_period(&self) -> Duration {
        if self.running && self.first_token.is_none() {
            WAIT_TICK
        } else {
            TICK
        }
    }

    async fn on_tick(&mut self) {
        self.tick_i += 1;
        if self.running {
            self.spinner_i = (self.spinner_i + 1) % SPINNER.len();
            // Spinner lives in the live tail; history stays reusable.
            self.dirty.mark_tail();
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

    fn error(&mut self, text: &str) {
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

        // The composer and approval card share the bottom band. Approvals
        // temporarily own it instead of stacking more chrome.
        // The composer soft-wraps inside its border, so how many rows it wants
        // depends on the width it will get: the area less the two border cells.
        let desired_input_h = if self.pending_approval.is_some() {
            5
        } else {
            self.composer
                .height(area.width.saturating_sub(2))
                .saturating_add(2)
                .max(3)
        };
        let desired_queue_h = if self.queued.is_empty() {
            0
        } else {
            (self.queued.len() as u16).min(3)
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
        let desired_popup_h = hist_lines
            .as_ref()
            .or(find_lines.as_ref())
            .or(popup_lines.as_ref())
            .map(|l| l.len() as u16)
            .unwrap_or(0);
        let layout = conversation_layout(
            area,
            true,
            desired_input_h,
            desired_queue_h,
            desired_popup_h,
        );
        self.page_h = layout.chat.height.saturating_sub(1).max(1);

        // Top to bottom: project path, transcript, transient overlays,
        // status, and the bottom-fixed input surface.
        if layout.header.height > 0 {
            self.draw_header(frame, layout.header);
        }
        if layout.chat.height > 0 {
            self.draw_chat(frame, layout.chat);
        }

        if layout.queue.height > 0 {
            let mut qlines: Vec<Line> = self
                .queued
                .iter()
                .take(layout.queue.height as usize)
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
            if self.queued.len() as u16 > layout.queue.height {
                qlines.pop();
                qlines.push(Line::from(Span::styled(
                    format!(
                        "↳ … {} more queued",
                        self.queued.len() as u16 - layout.queue.height + 1
                    ),
                    Style::default().fg(theme::DIM()),
                )));
            }
            Paragraph::new(qlines).render(layout.queue, frame.buffer_mut());
        }

        if let Some(lines) = hist_lines.or(find_lines).or(popup_lines) {
            if layout.popup.height > 0 {
                Paragraph::new(lines).render(layout.popup, frame.buffer_mut());
            }
        }

        self.composer_draw_area = Rect::default();
        if self.pending_approval.is_some() {
            self.draw_approval(frame, layout.input);
        } else if layout.input.height >= 3 {
            let border_color = if self.focus == Focus::Composer {
                theme::ACCENT()
            } else {
                theme::BORDER()
            };
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(border_color))
                .style(Style::default().bg(theme::COMPOSER_BG()));
            let inner = block.inner(layout.input);
            block.render(layout.input, frame.buffer_mut());
            let (composer_lines, cx, cy) = self.composer.render(inner.width, inner.height);
            self.composer_draw_area = inner;
            Paragraph::new(composer_lines)
                .style(Style::default().bg(theme::COMPOSER_BG()))
                .render(inner, frame.buffer_mut());
            if self.focus == Focus::Composer
                && self.history_search.is_none()
                && self.scroll_search.is_none()
            {
                frame.set_cursor_position(Position::new(inner.x + cx, inner.y + cy));
            }
        } else if layout.input.height > 0 {
            // Below three rows a box has no interior. Keep the prompt usable
            // by dropping only the border and bottom-aligning its visible rows.
            let (composer_lines, cx, cy) =
                self.composer.render(layout.input.width, layout.input.height);
            let composer_h = (composer_lines.len() as u16).min(layout.input.height);
            let composer_area = Rect {
                y: layout.input.bottom().saturating_sub(composer_h),
                height: composer_h,
                ..layout.input
            };
            self.composer_draw_area = composer_area;
            Paragraph::new(composer_lines)
                .style(Style::default().bg(theme::COMPOSER_BG()))
                .render(composer_area, frame.buffer_mut());
            if self.focus == Focus::Composer
                && self.history_search.is_none()
                && self.scroll_search.is_none()
                && composer_area.width > 0
                && composer_area.height > 0
            {
                frame.set_cursor_position(Position::new(
                    (composer_area.x + cx).min(composer_area.right().saturating_sub(1)),
                    (composer_area.y + cy).min(composer_area.bottom().saturating_sub(1)),
                ));
            }
        }
        self.draw_status(frame, layout.status);
    }

    fn draw_approval(&mut self, frame: &mut Frame, area: Rect) {
        let Some((_, name, summary, detail)) = self.pending_approval.as_ref() else {
            return;
        };
        if area.height < 5 {
            let lines = compact_approval_lines(name, summary, detail, area.width, area.height);
            let height = (lines.len() as u16).min(area.height);
            let draw_area = Rect {
                y: area.bottom().saturating_sub(height),
                height,
                ..area
            };
            Paragraph::new(lines)
                .style(Style::default().bg(theme::SURFACE()))
                .render(draw_area, frame.buffer_mut());
            if height > 0 {
                self.approval_hits = approval_choice_hit_regions(Rect {
                    x: draw_area.x,
                    y: draw_area.bottom() - 1,
                    width: draw_area.width,
                    height: 1,
                });
            }
            return;
        }
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

    fn draw_header(&mut self, frame: &mut Frame, area: Rect) {
        if self.dirty.chrome || self.header_width != area.width {
            self.header_width = area.width;
            self.header_line = header_path_line(&self.project, area.width as usize);
        }
        (&self.header_line).render(area, frame.buffer_mut());
    }

    /// Finished transcript plus the live tail, bottom anchored, honoring the
    /// scroll offset (0 follows the latest output).
    ///
    /// When only the live tail is dirty (spinner / tokens), the history prefix
    /// of `chat_buf` is reused and the tail is re-stitched.
    fn draw_chat(&mut self, frame: &mut Frame, area: Rect) {
        let layout_started = Instant::now();
        let mut content_w = area.width;
        if content_w == 0 || area.height == 0 {
            self.chat_draw_area = Rect::default();
            self.chat_line_map.clear();
            self.perf_layout_ms = layout_started.elapsed().as_secs_f64() * 1000.0;
            self.perf_selection_ms = 0.0;
            return;
        }
        let chat_dirty = self.dirty.chat;

        // Start from the previous frame's scrollbar decision. Re-deciding
        // from the full width on every paint re-wrapped the entire
        // transcript twice per frame once history overflowed (the width
        // oscillated between W and W-1), so frame cost grew with session
        // length exactly when a long reply was streaming.
        if self.scrollbar_reserved && area.width > 1 {
            content_w = area.width - 1;
        }
        // A width change re-wraps every block, and the bottom-anchored
        // offset would resolve to different content afterward. Anchor the
        // history line at the viewport bottom by content and restore it
        // after the re-wrap; positions inside the live tail keep the
        // numeric offset, which the tail compensation self-corrects.
        let resize_anchor = {
            let from_bottom = self.transcript.offset().saturating_sub(self.last_tail_len);
            if content_w != self.last_content_w && from_bottom > 0 {
                self.transcript.anchor_at(from_bottom)
            } else {
                None
            }
        };
        self.transcript.set_width(content_w);
        let mut tail_len = self.rebuild_tail(content_w);

        let mut hist_len = self.transcript.len();
        let mut total = hist_len + tail_len;
        let visible = area.height as usize;
        if total > visible && content_w == area.width && area.width > 1 {
            // Overflow began: rewrap once with a dedicated one-cell track.
            self.scrollbar_reserved = true;
            content_w = area.width - 1;
            self.transcript.set_width(content_w);
            tail_len = self.rebuild_tail(content_w);
            hist_len = self.transcript.len();
            total = hist_len + tail_len;
        } else if total <= visible && content_w < area.width {
            // Fits again at the narrowed width, so it also fits at the full
            // width (a wider wrap never yields more lines): reclaim the
            // scrollbar column for content.
            self.scrollbar_reserved = false;
            content_w = area.width;
            self.transcript.set_width(content_w);
            tail_len = self.rebuild_tail(content_w);
            hist_len = self.transcript.len();
            total = hist_len + tail_len;
        }
        // Keep a scrolled-up reader stationary as the live tail changes.
        // History pushes already bump the offset; the tail below history
        // must be compensated the same way, or streaming drags the view
        // forward and a collapsing tail flings it to the top. Skipped on a
        // width change, where the whole line mapping is rebuilt anyway.
        if content_w == self.last_content_w {
            self.transcript
                .compensate_tail_delta(tail_len as isize - self.last_tail_len as isize);
        } else if let Some(anchor) = resize_anchor {
            let from_bottom = self.transcript.resolve_anchor(anchor);
            self.transcript.set_offset(from_bottom + tail_len);
        }
        self.last_content_w = content_w;
        self.last_tail_len = tail_len;
        if total == 0 && self.pending_approval.is_none() {
            self.chat_buf.clear();
            self.chat_line_map.clear();
            self.hist_prefix_len = 0;
            self.hist_reuse_key = None;
            self.chat_draw_area = Rect::default();
            self.perf_layout_ms = layout_started.elapsed().as_secs_f64() * 1000.0;
            self.perf_selection_ms = 0.0;
            if self.composer.is_empty() {
                ready::render(
                    Rect {
                        width: content_w,
                        ..area
                    },
                    frame.buffer_mut(),
                );
            }
            return;
        }
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
                        .insert(0, Span::styled("❯ ", Style::default().fg(theme::DIM())));
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

        // One positional marker communicates scroll state without recreating
        // a tall barcode rail when history barely exceeds the viewport.
        if total > visible && area.width > 0 {
            let track_h = area.height as usize;
            let max_off = total - visible;
            let from_top = max_off.saturating_sub(offset);
            let marker_y =
                (from_top * track_h.saturating_sub(1)).checked_div(max_off).unwrap_or(0);
            if let Some(cell) = frame.buffer_mut().cell_mut((
                area.x + area.width.saturating_sub(1),
                area.y + marker_y as u16,
            )) {
                cell.set_symbol("▐");
                cell.set_style(Style::default().fg(theme::DIM()));
            }
        }
    }

    /// Rebuild the live tail into `tail_buf`, reusing cached stream/thinking
    /// wraps when only the spinner meta line changes between ticks.
    fn rebuild_tail(&mut self, width: u16) -> usize {
        let width_changed = width != self.tail_width;
        let prose_width = width.saturating_sub(2).max(8);
        let previous_stream_stable = self.stream_stable_len;
        let previous_thinking_len = self.thinking_wrapped.len();
        if width_changed {
            self.tail_width = width;
            self.thinking_source.clear();
        }
        // Incremental markdown: completed lines are rendered exactly once
        // and a resize only re-wraps, so a long streamed code block stays O(n)
        // over the reply instead of re-rendering the whole buffer on every
        // newline. `stream_changed` gates the tail_buf rebuild below.
        let mut stream_changed = false;
        let mut stream_reset = false;
        if self.stream_text.is_empty() {
            if self.stream_md.text_len() != 0 || !self.stream_wrapped.is_empty() {
                self.stream_md.clear();
                self.stream_wrapped.clear();
                self.stream_stable_len = 0;
                stream_changed = true;
                stream_reset = true;
            }
        } else if width_changed || self.stream_md.text_len() != self.stream_text.len() {
            stream_reset = width_changed || self.stream_text.len() < self.stream_md.text_len();
            self.stream_md.update(&self.stream_text, prose_width);
            let stable = if stream_reset {
                0
            } else {
                self.stream_stable_len
            };
            self.stream_stable_len = self
                .stream_md
                .sync_into(&mut self.stream_wrapped, stable);
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
            // The completed stream prefix is append-only at a fixed width.
            // Preserve it in both buffers and replace only the partial suffix,
            // rather than cloning the whole response for every token.
            let can_sync_stream_suffix = stream_changed
                && !thinking_changed
                && !stream_reset
                && self.tail_stable_len == previous_thinking_len + previous_stream_stable
                && self.tail_buf.len() >= self.tail_stable_len;
            if can_sync_stream_suffix {
                self.tail_buf.truncate(self.tail_stable_len);
                self.tail_buf.extend(
                    self.stream_wrapped[previous_stream_stable..]
                        .iter()
                        .cloned(),
                );
            } else {
                self.tail_buf.clear();
                self.tail_buf.extend(self.thinking_wrapped.iter().cloned().map(|mut line| {
                    line.spans.insert(
                        0,
                        Span::styled("◌ ", Style::default().fg(theme::DIM())),
                    );
                    line
                }));
                self.tail_buf.extend(self.stream_wrapped.iter().cloned());
            }
            self.tail_stable_len = self.thinking_wrapped.len() + self.stream_stable_len;
            self.tail_content_len = self.tail_buf.len();
        } else {
            self.tail_buf.truncate(self.tail_content_len);
        }

        if let Some((name, summary)) = &self.running_tool {
            self.tail_buf.push(tool_card::running_line(name, summary));
        }
        if self.running {
            let elapsed = self.turn_started.map(|t| t.elapsed()).unwrap_or_default();
            let toks = self.tok_per_sec();
            let mut meta = format!(" {}", elapsed_label(elapsed));
            if toks > 0.0 {
                meta.push_str(&format!(" · {toks:.0} tok/s"));
            }
            if self.thinking_chars > 0 && self.stream_text.is_empty() {
                meta.push_str(if self.show_thinking {
                    " · thinking (ctrl+t to hide)"
                } else {
                    " · thinking (ctrl+t to peek)"
                });
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
        if self.dirty.chrome || self.status_width != area.width {
            self.status_width = area.width;
            // Read the mode before taking the settings lock: the accessor
            // takes it too, and this mutex is not reentrant.
            let approvals = self.core.approval_mode().as_str().to_string();
            let model = self.core.settings.lock().unwrap().model.clone();
            let width = area.width as usize;
            let hint = self.status_hint();
            let left = if hint.is_empty() {
                String::new()
            } else {
                format!(" {hint}")
            };
            let short_model = extensions::short_model(&model);
            let right = if width >= 78 {
                match self.budget {
                    Some((used, total)) => format!(
                        "{short_model}  {}%  {approvals} ",
                        (used as f64 / total.max(1) as f64 * 100.0) as u32
                    ),
                    None => format!("{short_model}  {approvals} "),
                }
            } else if width >= 54 {
                format!("{short_model}  {approvals} ")
            } else if width < 4 {
                String::new()
            } else {
                let model_width = width
                    .saturating_sub(1)
                    .min(width.saturating_div(2).max(8));
                format!("{} ", clip(short_model, model_width))
            };
            let right_len = crate::ui::text::width(&right).min(width);
            let left_max = width.saturating_sub(right_len + 1);
            let left = clip(&left, left_max);
            let padding = width.saturating_sub(crate::ui::text::width(&left) + right_len);
            self.status_line = Line::from(vec![
                Span::styled(left, Style::default().fg(theme::DIM())),
                Span::raw(" ".repeat(padding)),
                Span::styled(right, Style::default().fg(theme::DIM())),
            ]);
        }
        (&self.status_line).render(area, frame.buffer_mut());
    }

    fn status_hint(&self) -> &'static str {
        if self.composer.has_selection() {
            "ctrl+c copy selection · esc clear"
        } else if self.transcript.has_text_selection() && self.focus == Focus::Scrollback {
            "y copy selection · esc clear"
        } else if self.transcript.has_text_selection() {
            "ctrl+c copy selection · esc clear"
        } else if self.pending_approval.is_some() {
            "y allow once · a allow for run · n deny"
        } else if self.history_search.is_some() {
            "↑↓ pick · enter insert · esc close"
        } else if self.scroll_search.is_some() {
            "↑↓ match · enter jump · esc close"
        } else if self.completion.is_some() {
            "↑↓ select · enter/tab accept · esc close"
        } else if self.focus == Focus::Scrollback {
            "j/k block · enter fold · y copy"
        } else if self.transcript.offset() > 0 {
            // Scrolled-up wins over running: while reading, Esc follows
            // (it does not cancel), and the hint must say so.
            "esc follow · pgup/pgdn scroll"
        } else if self.running {
            "enter queue · esc cancel"
        } else if self.quit_armed {
            "ctrl+c again to quit"
        } else {
            ""
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
    ("shift+tab", "cycle approvals for this run: ask → auto → readonly"),
    ("↑↓ / j k in history", "select a block · enter fold · y copy"),
    ("[ ] in history", "jump to previous or next user turn (shift+↑↓ too)"),
    ("g / G in history", "top of scrollback · follow latest"),
    ("/ at the start", "command menu · tab or enter completes"),
    ("@", "mention a project file (fuzzy search)"),
    ("ctrl+r", "search prompt history"),
    ("ctrl+f", "find in conversation"),
    ("n / N after find", "next or previous match in scrollback"),
    ("esc", "follow latest · cancel turn · return to composer"),
    ("wheel · pgup/pgdn", "scroll the conversation · over the prompt, the draft"),
    ("click in the prompt", "put the cursor there in a wrapped draft"),
    ("mouse drag", "select transcript or prompt text · y or ctrl+c copies"),
    ("double / triple click", "select the word · the whole line"),
    ("ctrl+o / o", "expand the last tool block"),
    ("ctrl+t", "show or hide model thinking"),
    ("ctrl+c ctrl+c", "quit (the model server keeps running)"),
    ("/<template> [args]", "run a prompt template from .agents/prompts/<name>.md"),
];

/// What this session needs from the world, stated in the terminal title so
/// tmux window lists, tab bars, and pane supervisors can read it without an
/// orchestrator attached. The bell rings on the idle-hands edge (an approval
/// arriving), which tmux monitor-bell and terminal urgency hints surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Presence {
    Idle,
    Working,
    NeedsApproval,
}

fn presence_title(presence: Presence, project: &std::path::Path) -> String {
    let base = project
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "openmax".into());
    match presence {
        Presence::Idle => format!("{base} · openmax"),
        Presence::Working => format!("{base} · openmax · working"),
        Presence::NeedsApproval => format!("{base} · openmax · needs approval"),
    }
}

/// Elapsed-time label for the live tail: tenths below ten seconds so the
/// silent wait visibly advances between whole seconds, whole seconds after.
/// The boundary sits at exactly 10.0 so the display is monotonic: 9.9s,
/// then 10.0s (the tenths branch rounds up), then 10s, never backward.
fn elapsed_label(elapsed: Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 10.0 {
        format!("{secs:.1}s")
    } else {
        format!("{}s", elapsed.as_secs())
    }
}

fn help_line(key: &str, description: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {key:<32} "),
            Style::default().fg(theme::ACCENT()),
        ),
        Span::styled(description.to_string(), Style::default().fg(theme::DIM())),
    ])
}

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

/// The one persistent line of top chrome: the project path, parent dimmed
/// and basename bright, shortened from the left so the basename survives
/// narrow terminals. `$HOME` collapses to `~`.
fn header_path_line(project: &std::path::Path, width: usize) -> Line<'static> {
    if width < 2 {
        return Line::default();
    }
    let home = std::env::var("HOME").ok();
    let display = home_shortened(&project.display().to_string(), home.as_deref());
    let (parent, base) = match display.rfind('/') {
        Some(i) if i + 1 < display.len() => display.split_at(i + 1),
        _ => ("", display.as_str()),
    };
    let base_style = Style::default()
        .fg(theme::ACCENT())
        .add_modifier(Modifier::BOLD);
    let parent_w = crate::ui::text::width(parent);
    let base_w = crate::ui::text::width(base);
    if parent_w + base_w <= width {
        return Line::from(vec![
            Span::styled(parent.to_string(), Style::default().fg(theme::DIM())),
            Span::styled(base.to_string(), base_style),
        ]);
    }
    if base_w + 2 <= width {
        return Line::from(vec![
            Span::styled("…/", Style::default().fg(theme::DIM())),
            Span::styled(base.to_string(), base_style),
        ]);
    }
    Line::from(Span::styled(clip(base, width), base_style))
}

fn home_shortened(path: &str, home: Option<&str>) -> String {
    let Some(home) = home else {
        return path.to_string();
    };
    if home.is_empty() || home == "/" {
        return path.to_string();
    }
    if path == home {
        return "~".to_string();
    }
    match path.strip_prefix(home) {
        Some(rest) if rest.starts_with('/') => format!("~{rest}"),
        _ => path.to_string(),
    }
}

fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else {
        format!("{n} {word}s")
    }
}

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {k:<11}"), Style::default().fg(theme::ACCENT())),
        Span::raw(v.to_string()),
    ])
}

fn clip(s: &str, max: usize) -> String {
    crate::ui::text::clip(s, max)
}

/// Replay shows a short tool-output preview, not the full persisted payload.
/// (+N −N) change counts from a persisted write/edit result, so a replayed
/// card keeps its badge even though Diff events are not persisted.
fn parse_change_counts(content: &str) -> Option<(usize, usize)> {
    let open = content.rfind("(+")?;
    let rest = &content[open + 2..];
    let digits = rest.find(|c: char| !c.is_ascii_digit())?;
    let added: usize = rest[..digits].parse().ok()?;
    let rest = rest[digits..].strip_prefix(" −")?;
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    let removed: usize = rest[..end].parse().ok()?;
    rest[end..].strip_prefix(')')?;
    Some((added, removed))
}

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

fn approval_choice_line() -> Line<'static> {
    Line::from(vec![
        Span::styled(
            APPROVAL_LABELS[0],
            Style::default()
                .fg(theme::WARN())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(APPROVAL_LABELS[1], Style::default().fg(theme::DIM())),
        Span::styled(APPROVAL_LABELS[2], Style::default().fg(theme::DIM())),
    ])
}

fn compact_approval_lines(
    name: &str,
    summary: &str,
    detail: &str,
    width: u16,
    height: u16,
) -> Vec<Line<'static>> {
    if height == 0 {
        return Vec::new();
    }
    let width = width as usize;
    if height == 1 {
        return vec![approval_choice_line()];
    }

    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{}  ", tool_card::human_name(name)),
            Style::default()
                .fg(theme::WARN())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(clip(summary, width.saturating_sub(name.len() + 2))),
    ])];
    if height >= 3 {
        let body = if detail.is_empty() { summary } else { detail };
        lines.push(Line::from(Span::styled(
            clip(body, width),
            Style::default().fg(theme::DIM()),
        )));
    }
    lines.push(approval_choice_line());
    lines
}

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
        approval_choice_line(),
    ]
}

fn approval_hit_regions(inner: Rect) -> [Option<Rect>; 3] {
    if inner.height < 3 {
        return [None; 3];
    }
    approval_choice_hit_regions(Rect {
        x: inner.x,
        y: inner.y + 2,
        width: inner.width,
        height: 1,
    })
}

fn approval_choice_hit_regions(row: Rect) -> [Option<Rect>; 3] {
    let mut hits = [None; 3];
    if row.height == 0 {
        return hits;
    }
    let mut x = row.x;
    for (index, label) in APPROVAL_LABELS.iter().enumerate() {
        let width = label
            .chars()
            .count()
            .min(row.right().saturating_sub(x) as usize);
        if width > 0 {
            hits[index] = Some(Rect {
                x,
                y: row.y,
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

/// Whether a key event is Shift+Tab.
///
/// Terminals disagree on the wire: most send CSI Z, which crossterm reports as
/// `BackTab` with no modifier, while the kitty keyboard protocol reports a Tab
/// carrying an explicit shift. Both are the same keystroke to the person at
/// the keyboard, so every binding has to read them the same way.
fn is_shift_tab(key: &KeyEvent) -> bool {
    key.code == KeyCode::BackTab
        || (key.code == KeyCode::Tab && key.modifiers.contains(KeyModifiers::SHIFT))
}

#[cfg(test)]
mod tests {
    use super::{
        approval_card_lines, approval_hit_regions, command_parts, compact_approval_lines,
        conversation_layout, elapsed_label, header_path_line, help_line, home_shortened, kv,
        is_shift_tab, paint_text_selection, parse_change_counts, plural, presence_title,
        rect_contains, save_model_selection,
        App, Dirty, Focus, Presence, TermEvent, MIN_DRAW_INTERVAL, TICK, WAIT_TICK,
    };
    use std::time::Duration;
    use crossterm::event::{
        KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
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
        let dir = crate::test_temp_dir("openmax-app-render");
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        let (files_tx, _files_rx) = mpsc::unbounded_channel();
        let app = App::new(core, dir.clone(), files_tx);
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

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    /// A /new issued mid-compaction must not leave the compaction flag
    /// armed: the old session's receipt is filtered out after the id
    /// changes, and a stale flag would misroute the next session's first
    /// Error into the compaction branch, clearing `running` while the core
    /// still owns a turn (whose next prompt then skips the queue).
    #[test]
    fn a_session_reset_clears_the_compaction_flag_with_the_running_one() {
        let (mut app, dir) = app_fixture();
        app.session_id = Some("old".into());
        app.running = true;
        app.compacting = true;
        app.reset_for_new_session();
        assert!(!app.running);
        assert!(!app.compacting, "compaction state is session-scoped, like running");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dirty_default_is_clean() {
        let d = Dirty::default();
        assert!(!d.any());
        assert!(!d.chat && !d.tail && !d.chrome && !d.selection);
    }

    #[test]
    fn paint_budget_supports_144_hz_with_scheduler_headroom() {
        let display_144hz = std::time::Duration::from_nanos(1_000_000_000 / 144);
        assert!(MIN_DRAW_INTERVAL < display_144hz);
        assert!(MIN_DRAW_INTERVAL < std::time::Duration::from_millis(6));
    }

    #[test]
    fn conversation_layout_pins_input_to_the_terminal_bottom() {
        let area = Rect::new(2, 3, 80, 18);
        let plain = conversation_layout(area, true, 3, 0, 0);
        let busy = conversation_layout(area, false, 3, 3, 7);
        let multiline = conversation_layout(area, false, 6, 3, 7);

        assert_eq!(plain.input.bottom(), area.bottom());
        assert_eq!(busy.input, plain.input);
        assert_eq!(busy.status.bottom(), busy.input.y);
        assert_eq!(busy.popup.bottom(), busy.status.y);
        assert_eq!(multiline.input.bottom(), area.bottom());
        assert!(multiline.input.y < busy.input.y);
    }

    #[test]
    fn short_layout_yields_brand_and_status_before_prompt_height() {
        let four_rows = conversation_layout(Rect::new(0, 0, 64, 4), true, 3, 0, 0);
        assert_eq!(four_rows.input.height, 3);
        assert_eq!(four_rows.status.height, 1);
        assert_eq!(four_rows.header.height, 0);
        assert_eq!(four_rows.input.bottom(), 4);

        let three_rows = conversation_layout(Rect::new(0, 0, 64, 3), true, 3, 0, 0);
        assert_eq!(three_rows.input.height, 3);
        assert_eq!(three_rows.status.height, 0);
        assert_eq!(three_rows.header.height, 0);

        // With exactly one spare row the conversation wins it, not the header.
        let five_rows = conversation_layout(Rect::new(0, 0, 64, 5), true, 3, 0, 0);
        assert_eq!(five_rows.header.height, 0);
        assert_eq!(five_rows.chat.height, 1);

        let six_rows = conversation_layout(Rect::new(0, 0, 64, 6), true, 3, 0, 0);
        assert_eq!(six_rows.header.height, 1);
        assert_eq!(six_rows.chat.height, 1);
    }

    #[test]
    fn active_overlay_has_priority_over_passive_queue_rows() {
        let layout = conversation_layout(Rect::new(0, 0, 64, 8), false, 3, 3, 6);

        assert_eq!(layout.popup.height, 4);
        assert_eq!(layout.queue.height, 0);
        assert_eq!(layout.popup.bottom(), layout.status.y);
        assert_eq!(layout.input.bottom(), 8);
    }

    #[test]
    fn active_overlay_stays_visible_before_status_on_four_rows() {
        let layout = conversation_layout(Rect::new(0, 0, 64, 4), false, 3, 0, 6);

        assert_eq!(layout.popup.height, 1);
        assert_eq!(layout.status.height, 0);
        assert_eq!(layout.popup.bottom(), layout.input.y);
        assert_eq!(layout.input.bottom(), 4);
    }

    #[test]
    fn chrome_invalidation_rebuilds_header_cache_at_the_same_width() {
        let (mut app, dir) = app_fixture();
        let _ = render_app(&mut app, 80, 24);
        app.header_line = Line::from("stale theme");
        app.dirty.mark_chrome();

        let buffer = render_app(&mut app, 80, 24);

        assert!(rows(&buffer)[0].contains("openmax-app-render"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn status_key_value_rows_keep_a_separator_after_long_keys() {
        let line = kv("throughput", "not available");
        let text: String = line.spans.iter().map(|span| span.content.as_ref()).collect();
        assert_eq!(text, "  throughput not available");
    }

    #[test]
    fn dirty_all_sets_every_region() {
        let d = Dirty::all();
        assert!(d.any());
        assert!(d.chat && d.tail && d.chrome && d.selection);
    }

    #[test]
    fn mark_chat_also_marks_tail_and_status() {
        let mut d = Dirty::default();
        d.mark_chat();
        assert!(d.chat);
        assert!(d.tail);
        assert!(d.chrome);
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

    #[tokio::test]
    async fn printable_key_invalidates_chrome_without_rebuilding_transcript() {
        let (mut app, dir) = app_fixture();
        app.dirty.clear();

        app.on_term_event(TermEvent::Key(KeyEvent::new(
            KeyCode::Char('x'),
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();

        assert_eq!(app.composer.text(), "x");
        assert!(!app.dirty.chat);
        assert!(!app.dirty.tail);
        assert!(app.dirty.chrome);
        fs::remove_dir_all(dir).unwrap();
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
        let dir = crate::test_temp_dir("openmax-model-save");
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
        let disk = config::load(&dir).unwrap();
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
        let missing = crate::test_temp_dir("openmax-model-save-missing").join("nested");
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

    /// What is highlighted is exactly what a copy carries, on the real buffer
    /// and not just in the spans. #132 made that true of the transcript; the
    /// prompt was still dropping the character under the release cell.
    #[tokio::test]
    async fn the_prompt_highlight_is_exactly_what_a_copy_carries() {
        let (mut app, dir) = app_fixture();
        app.composer.load("hello world");
        render_app(&mut app, 60, 24);
        let area = app.composer_draw_area;

        for release in [3u16, 5, 7, 12] {
            app.composer.click_at(area.width, area.height, 2, 0);
            app.composer.drag_to(area.width, area.height, release, 0);
            app.composer.finish_selection();
            let buffer = render_app(&mut app, 60, 24);

            let row = area.y;
            let highlighted: String = (area.x..area.right())
                .filter(|x| buffer[(*x, row)].bg == theme::SELECT())
                .map(|x| buffer[(x, row)].symbol().to_string())
                .collect();
            assert_eq!(
                highlighted,
                app.composer.selected_text().unwrap_or_default(),
                "highlight and copy disagree at release cell {release}",
            );
            // The release cell itself is carried, never dropped.
            assert_eq!(buffer[(area.x + release, row)].bg, theme::SELECT());
        }
        fs::remove_dir_all(dir).unwrap();
    }

    /// Terminals report presses, never click counts, so the gesture only
    /// works if the derivation does. Driven through the real event path.
    #[tokio::test]
    async fn repeated_presses_become_word_and_line_gestures() {
        let (mut app, dir) = app_fixture();
        app.composer.load("see crates/tui/src/app.rs:42 now");
        render_app(&mut app, 60, 24);
        let area = app.composer_draw_area;
        // Over the "crates/..." token.
        let (col, row) = (area.x + 2 + 8, area.y);
        let press = |c, r| mouse(MouseEventKind::Down(MouseButton::Left), c, r);
        let release = |c, r| mouse(MouseEventKind::Up(MouseButton::Left), c, r);

        app.on_term_event(press(col, row)).await.unwrap();
        app.on_term_event(release(col, row)).await.unwrap();
        assert_eq!(app.composer.selected_text(), None, "one press selects nothing");

        app.on_term_event(press(col, row)).await.unwrap();
        app.on_term_event(release(col, row)).await.unwrap();
        assert_eq!(
            app.composer.selected_text().as_deref(),
            Some("crates/tui/src/app.rs:42"),
            "the second press did not become a word gesture",
        );

        app.on_term_event(press(col, row)).await.unwrap();
        app.on_term_event(release(col, row)).await.unwrap();
        assert_eq!(
            app.composer.selected_text().as_deref(),
            Some("see crates/tui/src/app.rs:42 now"),
            "the third press did not become a line gesture",
        );

        // A fourth starts over rather than sticking on the line.
        app.on_term_event(press(col, row)).await.unwrap();
        app.on_term_event(release(col, row)).await.unwrap();
        assert_eq!(app.composer.selected_text(), None);
        fs::remove_dir_all(dir).unwrap();
    }

    /// Presses far apart in space are separate gestures even back to back.
    #[tokio::test]
    async fn presses_on_different_cells_do_not_compound() {
        let (mut app, dir) = app_fixture();
        app.composer.load("alpha beta gamma");
        render_app(&mut app, 60, 24);
        let area = app.composer_draw_area;
        let press = |c, r| mouse(MouseEventKind::Down(MouseButton::Left), c, r);

        app.on_term_event(press(area.x + 2, area.y)).await.unwrap();
        app.on_term_event(press(area.x + 10, area.y)).await.unwrap();
        app.on_term_event(mouse(
            MouseEventKind::Up(MouseButton::Left),
            area.x + 10,
            area.y,
        ))
        .await
        .unwrap();
        assert_eq!(
            app.composer.selected_text(),
            None,
            "two presses on different cells became a word gesture",
        );
        fs::remove_dir_all(dir).unwrap();
    }

    fn shift_tab_events() -> [KeyEvent; 2] {
        [
            // What most terminals send (CSI Z), reported with no modifier.
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE),
            // What the kitty keyboard protocol sends.
            KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT),
        ]
    }

    #[test]
    fn shift_tab_is_recognised_in_both_terminal_encodings() {
        for key in shift_tab_events() {
            assert!(is_shift_tab(&key), "not recognised: {key:?}");
        }
        // A plain Tab is the focus toggle, not the cycle.
        assert!(!is_shift_tab(&KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
        assert!(!is_shift_tab(&KeyEvent::new(
            KeyCode::Tab,
            KeyModifiers::CONTROL
        )));
    }

    /// Both encodings have to drive the same cycle, or the binding works on
    /// one half of the terminals and silently does something else on the rest.
    #[tokio::test]
    async fn shift_tab_cycles_approvals_in_both_encodings() {
        for key in shift_tab_events() {
            let (mut app, dir) = app_fixture();
            let mode = |app: &App| app.core.approval_mode();
            assert_eq!(mode(&app), config::ApprovalMode::Ask, "fixture default");

            app.on_key(key).await.unwrap();
            assert_eq!(mode(&app), config::ApprovalMode::Auto);
            app.on_key(key).await.unwrap();
            assert_eq!(mode(&app), config::ApprovalMode::Readonly);
            app.on_key(key).await.unwrap();
            assert_eq!(mode(&app), config::ApprovalMode::Ask, "cycle did not close");
            fs::remove_dir_all(dir).unwrap();
        }
    }

    /// The trust boundary only widens for every future session through a typed
    /// command. A keystroke must not write settings.json.
    #[tokio::test]
    async fn shift_tab_does_not_persist_the_widened_mode() {
        let (mut app, dir) = app_fixture();
        let on_disk = {
            let settings = app.core.settings.lock().unwrap().clone();
            config::save(&dir, &settings).unwrap();
            config::load(&dir).unwrap().approval_mode
        };
        assert_eq!(on_disk, config::ApprovalMode::Ask);

        app.on_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(
            app.core.approval_mode(),
            config::ApprovalMode::Auto,
            "the run should see the new mode",
        );
        assert_eq!(
            config::load(&dir).unwrap().approval_mode,
            config::ApprovalMode::Ask,
            "a keystroke persisted a wider approval mode",
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// Modal surfaces own the keyboard while they are open, so Shift+Tab means
    /// "previous item" there, never a change to the trust boundary behind it.
    #[tokio::test]
    async fn open_surfaces_keep_shift_tab_for_themselves() {
        for key in shift_tab_events() {
            // Completion popup: steps back through the list.
            let (mut app, dir) = app_fixture();
            app.composer.load("/");
            app.sync_completion();
            let count = app.completion.as_ref().unwrap().items.len();
            assert!(count > 1, "need a list to step through");

            app.on_key(key).await.unwrap();
            assert_eq!(
                app.completion.as_ref().unwrap().selected,
                count - 1,
                "shift+tab did not wrap to the last item",
            );
            assert_eq!(
                app.core.approval_mode(),
                config::ApprovalMode::Ask,
                "an open popup let the key through to the trust boundary",
            );
            fs::remove_dir_all(dir).unwrap();

            // A pending approval swallows keys until it is answered.
            let (mut app, dir) = app_fixture();
            app.pending_approval = Some((
                "id".into(),
                "write_file".into(),
                "summary".into(),
                "detail".into(),
            ));
            app.on_key(key).await.unwrap();
            assert_eq!(
                app.core.approval_mode(),
                config::ApprovalMode::Ask,
                "shift+tab changed approvals out from under a pending card",
            );
            fs::remove_dir_all(dir).unwrap();

        }
    }

    /// The picker matched `Down | Tab` for "next", so on a terminal that
    /// reports Shift+Tab as a shifted Tab it stepped the wrong way.
    #[tokio::test]
    async fn shift_tab_steps_back_in_the_model_picker() {
        for key in shift_tab_events() {
            let (mut app, dir) = app_fixture();
            fs::write(
                dir.join("providers.json"),
                r#"{"providers":{"alpha":{"base_url":"http://alpha/v1",
                   "models":[{"id":"one"},{"id":"two"},{"id":"three"}]}}}"#,
            )
            .unwrap();
            // No cache invalidation needed: the providers cache is keyed on
            // (data_dir, content hash) and this fixture's dir is unique, so
            // the read always misses. Touching global state from a test is
            // how the older races here started.
            app.model_picker = Some(crate::ui::model_picker::ModelPickerState::load(
                &dir,
                Some("alpha"),
                "one",
            ));
            let listed = app.model_picker.as_ref().unwrap().filtered.len();
            assert!(listed > 1, "need a list to step through");

            app.on_model_picker_key(key);

            assert_eq!(
                app.model_picker.as_ref().unwrap().selected,
                listed - 1,
                "shift+tab stepped forward instead of back",
            );
            fs::remove_dir_all(dir).unwrap();
        }
    }

    /// The leak this design exists to prevent: every save path serializes the
    /// whole `Settings`, so a run-scoped mode kept there would ride along on
    /// the next unrelated write and outlive the run.
    #[tokio::test]
    async fn an_unrelated_save_cannot_persist_the_run_scoped_mode() {
        let (mut app, dir) = app_fixture();
        {
            let settings = app.core.settings.lock().unwrap().clone();
            config::save(&dir, &settings).unwrap();
        }

        app.on_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
            .await
            .unwrap();
        assert_eq!(app.core.approval_mode(), config::ApprovalMode::Auto);

        // Any later write of the shared settings: picking a model.
        app.persist_model_selection(Some("alpha".into()), "some/model".into());

        assert_eq!(
            config::load(&dir).unwrap().approval_mode,
            config::ApprovalMode::Ask,
            "a model save carried the run-scoped approval mode to disk",
        );
        assert_eq!(
            app.core.approval_mode(),
            config::ApprovalMode::Auto,
            "the run should still see its own mode",
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// The approval card's "allow for run" is run-scoped for the same reason.
    #[tokio::test]
    async fn allow_for_run_does_not_reach_disk_either() {
        let (mut app, dir) = app_fixture();
        {
            let settings = app.core.settings.lock().unwrap().clone();
            config::save(&dir, &settings).unwrap();
        }
        app.pending_approval =
            Some(("id".into(), "bash".into(), "sum".into(), "detail".into()));

        app.on_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
            .await
            .unwrap();
        assert_eq!(app.core.approval_mode(), config::ApprovalMode::Auto);

        app.persist_model_selection(None, "some/model".into());
        assert_eq!(
            config::load(&dir).unwrap().approval_mode,
            config::ApprovalMode::Ask,
            "allow-for-run reached disk through an unrelated save",
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// A typed, persisted choice has to outrank a run override, or it would
    /// stay masked for the rest of the session.
    #[tokio::test]
    async fn typed_approvals_command_outranks_a_run_override() {
        let (mut app, dir) = app_fixture();
        app.on_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
            .await
            .unwrap();
        assert_eq!(app.core.approval_mode(), config::ApprovalMode::Auto);

        app.handle_submit("/approvals readonly".into()).await.unwrap();

        assert_eq!(
            app.core.approval_mode(),
            config::ApprovalMode::Readonly,
            "the run override kept masking the typed choice",
        );
        assert_eq!(
            config::load(&dir).unwrap().approval_mode,
            config::ApprovalMode::Readonly,
        );
        fs::remove_dir_all(dir).unwrap();
    }

    /// Plain Tab keeps its own job.
    #[tokio::test]
    async fn plain_tab_still_toggles_focus() {
        let (mut app, dir) = app_fixture();
        assert!(app.focus == Focus::Composer);
        app.on_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
            .await
            .unwrap();
        assert!(app.focus == Focus::Scrollback);
        assert_eq!(app.core.approval_mode(), config::ApprovalMode::Ask);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn retained_text_selection_does_not_swallow_composer_y() {
        let (mut app, dir) = app_fixture();
        app.transcript.set_width(20);
        app.transcript.push_user(vec![Line::from("selected text")]);
        assert!(app.transcript.begin_text_selection_at(0, 2));
        assert!(app.transcript.update_text_selection_at(0, 10));
        app.transcript.finish_text_selection();
        app.focus = Focus::Composer;

        app.on_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.composer.text(), "y");
        assert!(app.transcript.has_text_selection());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn y_after_g_copies_the_last_block_instead_of_no_op() {
        let (mut app, dir) = app_fixture();
        app.transcript.set_width(40);
        app.transcript.push_user(vec![Line::from("a question")]);
        app.transcript
            .push_assistant(vec![Line::from("the final answer")]);
        app.focus = Focus::Scrollback;

        // G follows the bottom and clears any block selection; y right
        // after must still copy the block at the bottom, not no-op.
        app.on_key(KeyEvent::new(KeyCode::Char('G'), KeyModifiers::SHIFT))
            .await
            .unwrap();
        app.on_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
            .await
            .unwrap();

        let rendered = buffer_text(&render_app(&mut app, 60, 14));
        assert!(rendered.contains("copied"), "{rendered}");
        fs::remove_dir_all(dir).unwrap();
    }

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> TermEvent {
        TermEvent::Mouse(MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        })
    }

    /// Ctrl+Shift+C only reaches the app on terminals that speak the kitty
    /// keyboard protocol; everywhere else it arrives as a plain Ctrl+C. A live
    /// selection is what makes the press a copy, and copying clears it so the
    /// press after it still cancels or quits.
    #[tokio::test]
    async fn ctrl_c_copies_a_live_selection_before_it_arms_quit() {
        let (mut app, dir) = app_fixture();
        app.composer.load("copy me please");
        render_app(&mut app, 60, 24);
        let area = app.composer_draw_area;
        app.composer.click_at(area.width, area.height, 2, 0);
        app.composer.drag_to(area.width, area.height, 5, 0);
        app.composer.finish_selection();
        assert_eq!(app.composer.selected_text().as_deref(), Some("copy"));

        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        app.on_key(ctrl_c).await.unwrap();
        assert!(!app.composer.has_selection(), "copy left a stale highlight");
        assert!(!app.quit_armed, "copying must not arm quit");

        app.on_key(ctrl_c).await.unwrap();
        assert!(app.quit_armed, "a copy must not disarm the quit binding");
        fs::remove_dir_all(dir).unwrap();
    }

    /// Terminals that do report the shift send a distinct Ctrl+Shift+C, which
    /// has only ever meant copy. With nothing selected it must stay a no-op.
    #[tokio::test]
    async fn ctrl_shift_c_never_becomes_the_quit_binding() {
        let (mut app, dir) = app_fixture();

        app.on_key(KeyEvent::new(
            KeyCode::Char('C'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        ))
        .await
        .unwrap();
        assert!(!app.quit_armed);

        // A plain Ctrl+C is still the cancel and quit binding.
        app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .await
            .unwrap();
        assert!(app.quit_armed);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn ctrl_c_copies_a_transcript_selection_too() {
        let (mut app, dir) = app_fixture();
        app.transcript.set_width(20);
        app.transcript.push_user(vec![Line::from("selected text")]);
        assert!(app.transcript.begin_text_selection_at(0, 2));
        assert!(app.transcript.update_text_selection_at(0, 10));
        app.transcript.finish_text_selection();

        app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .await
            .unwrap();
        assert!(!app.transcript.has_text_selection());
        assert!(!app.quit_armed);
        fs::remove_dir_all(dir).unwrap();
    }

    /// The wheel belongs to whatever is under the pointer: a long draft
    /// scrolls inside the prompt, the conversation scrolls above it.
    #[tokio::test]
    async fn the_wheel_scrolls_whichever_surface_is_under_the_pointer() {
        let (mut app, dir) = app_fixture();
        app.transcript.set_width(58);
        for i in 0..40 {
            app.transcript.push_user(vec![Line::from(format!("turn {i}"))]);
        }
        let mut draft = String::new();
        for i in 0..20 {
            draft.push_str(&format!("line {i:02}\n"));
        }
        app.composer.load(&draft);
        render_app(&mut app, 60, 24);

        let prompt = app.composer_draw_area;
        let first_visible = |app: &mut App| {
            let area = app.composer_draw_area;
            let (lines, _, _) = app.composer.render(area.width, area.height);
            line_text(&lines[0])
        };
        let tail = first_visible(&mut app);

        app.on_term_event(mouse(MouseEventKind::ScrollUp, prompt.x, prompt.y))
            .await
            .unwrap();
        assert_ne!(first_visible(&mut app), tail, "the draft did not scroll");
        assert_eq!(app.transcript.offset(), 0, "the conversation moved instead");

        // Over the conversation the wheel goes back to the transcript.
        let scrolled = first_visible(&mut app);
        app.on_term_event(mouse(MouseEventKind::ScrollUp, 1, app.chat_draw_area.y))
            .await
            .unwrap();
        assert!(app.transcript.offset() > 0, "the conversation did not scroll");
        assert_eq!(first_visible(&mut app), scrolled, "the draft moved instead");
        fs::remove_dir_all(dir).unwrap();
    }

    /// A click in the prompt is an edit gesture, not a transcript selection.
    #[tokio::test]
    async fn clicking_the_prompt_moves_the_cursor_and_takes_focus() {
        let (mut app, dir) = app_fixture();
        app.transcript.set_width(58);
        app.transcript.push_user(vec![Line::from("earlier turn")]);
        app.composer.load("hello world");
        app.focus = Focus::Scrollback;
        render_app(&mut app, 60, 24);

        let prompt = app.composer_draw_area;
        app.on_term_event(mouse(
            MouseEventKind::Down(MouseButton::Left),
            prompt.x + 4,
            prompt.y,
        ))
        .await
        .unwrap();

        assert!(app.focus == Focus::Composer);
        assert_eq!(app.composer.cursor_context().1, 2);
        assert!(!app.transcript.has_text_selection());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn submitting_exact_slash_completion_closes_the_popup() {
        let (mut app, dir) = app_fixture();
        app.composer.load("/status");
        app.sync_completion();
        assert!(app.completion.is_some());

        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .await
            .unwrap();

        assert!(app.completion.is_none());
        assert!(app.composer.is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn modified_enter_inserts_newline_instead_of_accepting_completion() {
        for modifiers in [KeyModifiers::SHIFT, KeyModifiers::ALT] {
            let (mut app, dir) = app_fixture();
            app.composer.load("/status");
            app.sync_completion();
            assert!(app.completion.is_some());

            app.on_key(KeyEvent::new(KeyCode::Enter, modifiers))
                .await
                .unwrap();

            assert_eq!(app.composer.text(), "/status\n");
            assert!(app.completion.is_none());
            assert_eq!(app.transcript.block_count(), 0);
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[tokio::test]
    async fn approval_remains_modal_when_history_search_is_requested() {
        let (mut app, dir) = app_fixture();
        app.composer.load("previous prompt");
        let _ = app.composer.take();
        app.on_agent_event(AgentEvent::ApprovalRequest {
            reason: "gate".into(),
            approval_id: "approval-history".into(),
            name: "bash".into(),
            summary: "run tests".into(),
            detail: "cargo test".into(),
            source_path: String::new(),
            source_sha: String::new(),
        });

        app.on_key(KeyEvent::new(
            KeyCode::Char('r'),
            KeyModifiers::CONTROL,
        ))
        .await
        .unwrap();

        assert!(app.pending_approval.is_some());
        assert!(app.history_search.is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn idle_layout_is_restrained_and_has_bordered_composer() {
        let (mut app, dir) = app_fixture();
        app.core.settings.lock().unwrap().model = "provider/test-model".into();
        let buffer = render_app(&mut app, 96, 18);
        let text = buffer_text(&buffer);
        assert!(rows(&buffer)[0].contains("openmax-app-render"));
        assert!(rows(&buffer)[1].starts_with("READY"));
        assert!(text.contains("READY"));
        assert!(text.contains("Describe a task"));
        assert!(text.contains("╭"));
        assert!(text.contains("test-model  ask"));
        assert!(!text.contains("sample-project"));
        assert!(!text.contains("small core"));
        assert!(!text.contains("skills · tools"));
        assert!(!text.contains("/ commands"));
        assert!(!text.contains("●"));
        assert!(!text.contains("ctx 0%"));
        assert!(rows(&buffer).last().unwrap().starts_with('╰'));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn prompt_stays_visible_at_the_bottom_of_short_terminals() {
        let (mut app, dir) = app_fixture();

        let four_rows = rows(&render_app(&mut app, 64, 4));
        assert!(!four_rows[0].contains("openmax-app-render"));
        assert!(four_rows[2].contains("Describe a task"));
        assert!(four_rows[3].starts_with('╰'));

        let two_rows = rows(&render_app(&mut app, 64, 2));
        assert!(two_rows[1].contains("Describe a task"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn project_path_header_persists_through_active_conversation() {
        let (mut app, dir) = app_fixture();
        app.insert_user_block("inspect the current layout");
        let buffer = render_app(&mut app, 72, 12);
        let rendered = rows(&buffer);

        assert!(rendered[0].contains("openmax-app-render"));
        assert!(rendered.iter().any(|row| row.contains("inspect the current layout")));
        assert!(rendered.last().unwrap().starts_with('╰'));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn header_path_dims_parent_and_keeps_basename_on_narrow_widths() {
        let path = std::path::Path::new("/work/deep/nested/project");

        let wide = header_path_line(path, 40);
        assert_eq!(line_text(&wide), "/work/deep/nested/project");
        assert_eq!(wide.spans.len(), 2);
        assert_eq!(wide.spans[1].content.as_ref(), "project");

        let narrow = header_path_line(path, 12);
        assert_eq!(line_text(&narrow), "…/project");

        let tiny = header_path_line(path, 5);
        assert_eq!(line_text(&tiny), "proj…");
        for width in 0..30 {
            let text = line_text(&header_path_line(path, width));
            assert!(crate::ui::text::width(&text) <= width);
        }
    }

    #[test]
    fn home_prefix_collapses_to_tilde_without_eating_sibling_dirs() {
        let home = Some("/Users/max");
        assert_eq!(home_shortened("/Users/max/code/app", home), "~/code/app");
        assert_eq!(home_shortened("/Users/max", home), "~");
        assert_eq!(home_shortened("/Users/maxine/code", home), "/Users/maxine/code");
        assert_eq!(home_shortened("/srv/data", home), "/srv/data");
        assert_eq!(home_shortened("/srv/data", None), "/srv/data");
        assert_eq!(home_shortened("/srv/data", Some("/")), "/srv/data");
    }

    #[test]
    fn streaming_output_grows_above_the_fixed_prompt() {
        let (mut app, dir) = app_fixture();
        app.running = true;
        app.turn_started = Some(std::time::Instant::now());
        app.on_agent_event(AgentEvent::Token {
            text: "first streamed line\nsecond streamed line".into(),
        });
        let rendered = rows(&render_app(&mut app, 64, 8));

        assert!(rendered.iter().any(|row| row.contains("second streamed line")));
        assert!(rendered.iter().any(|row| row.contains("esc to cancel")));
        assert!(rendered.last().unwrap().starts_with('╰'));
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn tab_leaves_the_find_bar_at_the_composer() {
        let (mut app, dir) = app_fixture();
        app.transcript.push(vec![Line::from("some history")]);
        app.scroll_search = Some(("hist".into(), 0, vec![0]));

        app.on_term_event(TermEvent::Key(KeyEvent::new(
            KeyCode::Tab,
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();
        assert!(app.scroll_search.is_none());
        assert!(matches!(app.focus, Focus::Composer));

        // The next keystroke is a prompt in the composer, not a swallowed
        // nav binding or a find-query character.
        app.on_term_event(TermEvent::Key(KeyEvent::new(
            KeyCode::Char('g'),
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();
        assert_eq!(app.composer.text(), "g");
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn find_with_no_matches_closes_at_the_composer() {
        let (mut app, dir) = app_fixture();
        app.transcript.push(vec![Line::from("some history")]);
        app.scroll_search = Some(("zzznotfound".into(), 0, vec![]));

        app.on_term_event(TermEvent::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();
        assert!(app.scroll_search.is_none());
        assert!(matches!(app.focus, Focus::Composer));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn presence_titles_state_the_project_and_the_need() {
        let project = std::path::Path::new("/home/max/things/open-max");
        assert_eq!(
            presence_title(Presence::Idle, project),
            "open-max · openmax"
        );
        assert_eq!(
            presence_title(Presence::Working, project),
            "open-max · openmax · working"
        );
        assert_eq!(
            presence_title(Presence::NeedsApproval, project),
            "open-max · openmax · needs approval"
        );
    }

    #[test]
    fn presence_follows_the_turn_lifecycle() {
        let (mut app, dir) = app_fixture();
        assert_eq!(app.presence, Presence::Idle);

        app.on_agent_event(AgentEvent::ApprovalRequest {
            approval_id: "a1".into(),
            name: "write_file".into(),
            summary: "write x".into(),
            detail: String::new(),
            reason: "gate".into(),
            source_path: String::new(),
            source_sha: String::new(),
        });
        assert_eq!(app.presence, Presence::NeedsApproval);

        app.on_agent_event(AgentEvent::ApprovalSettled {
            approval_id: "a1".into(),
            outcome: "allowed".into(),
        });
        assert_eq!(app.presence, Presence::Working);

        app.on_agent_event(AgentEvent::Done {
            stop_reason: "stop".into(),
        });
        assert_eq!(app.presence, Presence::Idle);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn replay_never_badges_a_read_only_tool() {
        let (mut app, dir) = app_fixture();
        let meta = open_max_core::sessions::create(
            &app.core,
            app.project.display().to_string(),
        )
        .unwrap();
        let messages = vec![
            open_max_core::types::ChatMessage::user("read the changelog"),
            open_max_core::types::ChatMessage {
                role: "assistant".into(),
                content: Some(String::new()),
                tool_calls: Some(vec![open_max_core::types::ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: open_max_core::types::ToolCallFunction {
                        name: "read_file".into(),
                        arguments: "{\"path\":\"CHANGELOG.md\"}".into(),
                    },
                }]),
                tool_call_id: None,
            },
            // File content that happens to look like an edit summary.
            open_max_core::types::ChatMessage {
                role: "tool".into(),
                content: Some("1 release notes (+3 −0) overall".into()),
                tool_calls: None,
                tool_call_id: Some("c1".into()),
            },
        ];
        let mut persisted = 0usize;
        open_max_core::sessions::save_messages(&app.core, &meta.id, &messages, &mut persisted, false);
        app.replay(&meta.id);
        let text = buffer_text(&render_app(&mut app, 100, 30));
        assert!(text.contains("CHANGELOG.md"), "{text}");
        assert!(!text.contains("+3"), "read card wears a diff badge: {text}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn change_counts_parse_from_persisted_results() {
        assert_eq!(parse_change_counts("wrote notes.md (+3 −0)"), Some((3, 0)));
        assert_eq!(
            parse_change_counts("edited a.rs (+2 −1) · first change at line 4"),
            Some((2, 1))
        );
        assert_eq!(parse_change_counts("hello world"), None);
        assert_eq!(parse_change_counts("odd (+x −1)"), None);
    }

    #[test]
    fn replay_keeps_diff_badges_and_marks_sittings() {
        let (mut app, dir) = app_fixture();
        let meta = open_max_core::sessions::create(
            &app.core,
            app.project.display().to_string(),
        )
        .unwrap();
        let messages = vec![
            open_max_core::types::ChatMessage::user("write the note"),
            open_max_core::types::ChatMessage {
                role: "assistant".into(),
                content: Some(String::new()),
                tool_calls: Some(vec![open_max_core::types::ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: open_max_core::types::ToolCallFunction {
                        name: "write_file".into(),
                        arguments: "{\"path\":\"notes.md\",\"content\":\"x\"}".into(),
                    },
                }]),
                tool_call_id: None,
            },
            open_max_core::types::ChatMessage {
                role: "tool".into(),
                content: Some("wrote notes.md (+3 −0)".into()),
                tool_calls: None,
                tool_call_id: Some("c1".into()),
            },
            open_max_core::types::ChatMessage {
                role: "assistant".into(),
                content: Some("done".into()),
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let mut persisted = 0usize;
        open_max_core::sessions::save_messages(&app.core, &meta.id, &messages, &mut persisted, false);
        // An earlier sitting ended after the first two messages.
        open_max_core::sessions::record_resume_point(&app.core, &meta.id, 2);

        app.replay(&meta.id);
        let text = buffer_text(&render_app(&mut app, 100, 30));
        // The replayed write card keeps its evidence.
        assert!(text.contains("notes.md"), "{text}");
        assert!(text.contains("+3"), "{text}");
        assert!(text.contains("−0"), "{text}");
        // The sitting boundary renders as a divider.
        assert!(text.contains("• resumed"), "{text}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn new_session_returns_queued_input_to_the_composer() {
        let (mut app, dir) = app_fixture();
        app.queued = vec!["first queued".into(), "second queued".into()];
        app.reset_for_new_session();
        assert_eq!(app.composer.text(), "first queued\nsecond queued");
        assert!(app.queued.is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn up_in_an_empty_composer_pulls_back_the_newest_queued_message() {
        let (mut app, dir) = app_fixture();
        app.running = true;
        app.queued = vec!["first".into(), "second".into()];

        app.on_term_event(TermEvent::Key(KeyEvent::new(
            KeyCode::Up,
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();

        assert_eq!(app.composer.text(), "second");
        assert_eq!(app.queued, vec!["first".to_string()]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn elapsed_label_shows_tenths_only_while_short() {
        assert_eq!(elapsed_label(Duration::from_millis(400)), "0.4s");
        assert_eq!(elapsed_label(Duration::from_millis(3940)), "3.9s");
        assert_eq!(elapsed_label(Duration::from_millis(9940)), "9.9s");
        // Around the boundary the display must never move backward:
        // 9.9s, 10.0s, 10s.
        assert_eq!(elapsed_label(Duration::from_millis(9960)), "10.0s");
        assert_eq!(elapsed_label(Duration::from_millis(10_400)), "10s");
        assert_eq!(elapsed_label(Duration::from_secs(12)), "12s");
    }

    #[test]
    fn tick_runs_fluid_only_during_the_pre_token_wait() {
        let (mut app, dir) = app_fixture();
        // Idle: relaxed cadence.
        assert_eq!(app.tick_period(), TICK);
        // Waiting on the model with nothing streamed yet: the spinner is
        // the only sign of life, so animation runs fluid.
        app.running = true;
        assert_eq!(app.tick_period(), WAIT_TICK);
        // Content is flowing: paints follow tokens, cadence relaxes.
        app.first_token = Some(std::time::Instant::now());
        assert_eq!(app.tick_period(), TICK);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn composer_focus_is_the_only_bright_border() {
        let (mut app, dir) = app_fixture();
        let focused = render_app(&mut app, 64, 12);
        let border_y = rows(&focused)
            .iter()
            .position(|row| row.starts_with('╭'))
            .unwrap() as u16;
        assert_eq!(focused[(0, border_y)].fg, theme::ACCENT());

        app.focus = Focus::Scrollback;
        let unfocused = render_app(&mut app, 64, 12);
        assert_eq!(unfocused[(0, border_y)].fg, theme::BORDER());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ready_state_disappears_when_conversation_or_live_activity_starts() {
        let (mut app, dir) = app_fixture();
        app.insert_user_block("inspect this project");
        let conversation = buffer_text(&render_app(&mut app, 80, 16));
        assert!(!conversation.contains("READY"));

        app.reset_for_new_session();
        app.running = true;
        app.turn_started = Some(std::time::Instant::now());
        let running = buffer_text(&render_app(&mut app, 80, 16));
        assert!(!running.contains("READY"));
        assert!(running.contains("esc to cancel"));
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
        assert!(running_text.contains("I will inspect it."));
        assert!(!running_text.contains("│ I will inspect it."));
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
        assert!(!complete_text.contains("test one ok"));
        assert_eq!(
            app.last_tool_output.as_deref(),
            Some("test one ok\ntest two ok\ntest three ok\ntest four ok")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn assistant_code_has_one_structural_edge_without_prose_rails() {
        let (mut app, dir) = app_fixture();
        app.on_agent_event(AgentEvent::MessageDone {
            text: "Plain response.\n```rust\nfn main() {}\n```\nDone.".into(),
        });
        let text = buffer_text(&render_app(&mut app, 64, 12));

        assert!(text.contains("Plain response."));
        assert!(text.contains("│ fn main() {}"));
        assert!(!text.contains("│ Plain response."));
        assert!(!text.contains("│ │"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn non_overflowing_transcript_reclaims_scrollbar_column() {
        let (mut app, dir) = app_fixture();
        let full_width = "1234567890123456789012345678901234";
        app.on_agent_event(AgentEvent::MessageDone {
            text: full_width.into(),
        });
        let rendered = rows(&render_app(&mut app, 34, 8));

        assert!(rendered.iter().any(|row| row == full_width));
        assert!(rendered.iter().all(|row| !row.ends_with('▐')));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn overflowing_transcript_uses_one_position_marker_not_a_rail() {
        let (mut app, dir) = app_fixture();
        for index in 0..8 {
            app.transcript
                .push(vec![Line::from(format!("history line {index}"))]);
        }
        let rendered = rows(&render_app(&mut app, 40, 10));
        let marker_count = rendered
            .iter()
            .map(|row| row.matches('▐').count())
            .sum::<usize>();

        assert_eq!(marker_count, 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn steady_overflow_frames_never_rewrap_history() {
        let (mut app, dir) = app_fixture();
        for index in 0..12 {
            app.transcript
                .push(vec![Line::from(format!("history line {index}"))]);
        }
        // The first frame discovers overflow and re-wraps once to reserve
        // the scrollbar column.
        let first = rows(&render_app(&mut app, 40, 6));
        assert!(first.iter().any(|row| row.contains('▐')));
        let settled = app.transcript.rewraps;

        // Steady state: replies land, frames repaint, history never
        // re-wraps. Before the sticky scrollbar decision every paint
        // re-wrapped the whole transcript twice (W → W-1 oscillation).
        for index in 0..5 {
            app.on_agent_event(AgentEvent::MessageDone {
                text: format!("reply {index}"),
            });
            render_app(&mut app, 40, 6);
        }
        assert_eq!(app.transcript.rewraps, settled);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn growing_taller_releases_the_scrollbar_column_with_one_rewrap() {
        let (mut app, dir) = app_fixture();
        for index in 0..12 {
            app.transcript
                .push(vec![Line::from(format!("history line {index}"))]);
        }
        render_app(&mut app, 40, 6);
        let reserved = app.transcript.rewraps;

        // Tall enough to fit: the column is released with exactly one
        // re-wrap and later frames stay settled at the full width.
        let tall = rows(&render_app(&mut app, 40, 30));
        assert!(tall.iter().all(|row| !row.contains('▐')));
        assert_eq!(app.transcript.rewraps, reserved + 1);
        render_app(&mut app, 40, 30);
        assert_eq!(app.transcript.rewraps, reserved + 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streaming_appends_keep_a_scrolled_view_pinned() {
        let (mut app, dir) = app_fixture();
        for index in 0..30 {
            app.transcript
                .push(vec![Line::from(format!("history line {index:02}"))]);
        }
        render_app(&mut app, 40, 10);
        app.transcript.scroll_up(12);
        let before = rows(&render_app(&mut app, 40, 10));

        // The live tail grows line by line below history; the content the
        // reader is anchored on must not move.
        for index in 0..8 {
            app.on_agent_event(AgentEvent::Token {
                text: format!("stream line {index}\n"),
            });
            render_app(&mut app, 40, 10);
        }
        let after = rows(&render_app(&mut app, 40, 10));
        assert_eq!(top_history_row(&before), top_history_row(&after));
        fs::remove_dir_all(dir).unwrap();
    }

    /// First visible transcript row (skips header/chrome rows), the line a
    /// scrolled-up reader is anchored on.
    fn top_history_row(rendered: &[String]) -> String {
        rendered
            .iter()
            .find(|row| row.contains("history line"))
            .cloned()
            .expect("no history row visible")
    }

    #[test]
    fn finished_tail_does_not_fling_a_scrolled_view_to_the_top() {
        let (mut app, dir) = app_fixture();
        for index in 0..30 {
            app.transcript
                .push(vec![Line::from(format!("history line {index:02}"))]);
        }
        render_app(&mut app, 40, 10);
        let reply = (0..10)
            .map(|i| format!("reply line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        app.on_agent_event(AgentEvent::Token { text: reply.clone() });
        render_app(&mut app, 40, 10);
        app.transcript.scroll_up(20);
        let before = rows(&render_app(&mut app, 40, 10));

        // The turn ends: the tail collapses into a history block. The view
        // must stay where the reader was, not jump to the transcript top.
        app.on_agent_event(AgentEvent::MessageDone { text: reply });
        let after = rows(&render_app(&mut app, 40, 10));
        assert_eq!(top_history_row(&before), top_history_row(&after));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resize_keeps_the_scrolled_view_on_the_same_content() {
        let (mut app, dir) = app_fixture();
        for index in 0..30 {
            app.transcript.push(vec![Line::from(format!(
                "history line {index:02} padded so it wraps at the narrow width"
            ))]);
        }
        render_app(&mut app, 40, 10);
        app.transcript.scroll_up(20);
        let before = rows(&render_app(&mut app, 40, 10));

        // Growing the terminal re-wraps every block (two rows collapse to
        // one). The block at the viewport bottom must stay the same block,
        // not whatever a stale numeric offset happens to land on.
        let after = rows(&render_app(&mut app, 90, 10));
        assert_eq!(bottom_history_label(&before), bottom_history_label(&after));
        fs::remove_dir_all(dir).unwrap();
    }

    /// Label ("history line NN") of the last visible transcript row that
    /// carries one, the content a scrolled reader is anchored on.
    fn bottom_history_label(rendered: &[String]) -> String {
        rendered
            .iter()
            .rev()
            .find_map(|row| {
                let start = row.find("history line")?;
                Some(row[start..start + 15].to_string())
            })
            .expect("no history row visible")
    }

    #[tokio::test]
    async fn esc_while_scrolled_follows_the_live_view_instead_of_cancelling() {
        let (mut app, dir) = app_fixture();
        for index in 0..30 {
            app.transcript
                .push(vec![Line::from(format!("history line {index:02}"))]);
        }
        render_app(&mut app, 40, 10);
        app.running = true;
        app.transcript.scroll_up(12);

        app.on_term_event(TermEvent::Key(KeyEvent::new(
            KeyCode::Esc,
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();

        assert_eq!(app.transcript.offset(), 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn scrolled_hint_outranks_the_running_hint() {
        let (mut app, dir) = app_fixture();
        for index in 0..30 {
            app.transcript
                .push(vec![Line::from(format!("history line {index:02}"))]);
        }
        render_app(&mut app, 40, 10);
        app.running = true;
        app.transcript.scroll_up(5);
        assert_eq!(app.status_hint(), "esc follow · pgup/pgdn scroll");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn message_done_falls_back_to_the_provider_text_when_the_stream_differs() {
        let (mut app, dir) = app_fixture();
        app.stream_text = "stale streamed draft".into();
        app.rebuild_tail(40);

        app.on_agent_event(AgentEvent::MessageDone {
            text: "canonical provider response".into(),
        });

        assert_eq!(
            app.transcript.last_assistant_text().as_deref(),
            Some("canonical provider response")
        );
        assert!(app.stream_text.is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn approval_replaces_composer_then_restores_its_draft() {
        let (mut app, dir) = app_fixture();
        app.composer.load("keep this draft");
        app.on_agent_event(AgentEvent::ApprovalRequest {
            reason: "gate".into(),
            approval_id: "approval-1".into(),
            name: "bash".into(),
            summary: "install dependencies".into(),
            detail: "cargo fetch".into(),
            source_path: String::new(),
            source_sha: String::new(),
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
    fn compact_approval_keeps_keyboard_choices_visible() {
        let lines = compact_approval_lines("bash", "run tests", "cargo test", 64, 4);
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");
        assert!(text.contains("Shell"));
        assert!(text.contains("cargo test"));
        assert!(text.contains("[y] Allow once"));
        assert!(text.contains("[a] Allow for run"));
        assert!(text.contains("[n] Deny"));
    }

    #[test]
    fn compact_approval_keeps_mouse_targets_aligned_with_choices() {
        let (mut app, dir) = app_fixture();
        app.on_agent_event(AgentEvent::ApprovalRequest {
            reason: "gate".into(),
            approval_id: "approval-compact".into(),
            name: "bash".into(),
            summary: "run tests".into(),
            detail: "cargo test".into(),
            source_path: String::new(),
            source_sha: String::new(),
        });
        let buffer = render_app(&mut app, 64, 4);
        let rendered = rows(&buffer);
        let choices_y = rendered
            .iter()
            .position(|row| row.contains("[y] Allow once"))
            .unwrap() as u16;

        assert!(app.approval_hits.iter().all(Option::is_some));
        assert!(app
            .approval_hits
            .iter()
            .flatten()
            .all(|region| region.y == choices_y));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn help_columns_always_separate_long_usage_from_description() {
        let line = help_line(
            "/theme dark|light|mono|catppuccin",
            "switch appearance",
        );
        assert!(line_text(&line).contains("catppuccin switch appearance"));
    }

    #[tokio::test]
    async fn help_lists_the_users_own_templates() {
        let (mut app, dir) = app_fixture();
        app.templates = vec![
            ("deploy".to_string(), "ship it".to_string()),
            // Shadowed by the /new built-in: dispatch would never reach it,
            // so help must not advertise it.
            ("new".to_string(), "never invocable".to_string()),
        ];
        app.slash("help").await;
        let text = buffer_text(&render_app(&mut app, 100, 30));
        assert!(text.contains("/deploy"));
        assert!(text.contains("ship it"));
        assert!(!text.contains("never invocable"));
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
        assert!(!text.contains("READY"));
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
        assert!(narrow_text.contains("openmax-app-render"));
        assert!(narrow_text.contains("READY"));
        assert!(narrow_text.contains("Describe a task"));
        assert!(!narrow_text.contains("small core"));
        assert!(!narrow_text.contains("skills · tools"));
        // At five rows the header yields its row to the conversation plane.
        let tiny = render_app(&mut app, 12, 5);
        assert!(buffer_text(&tiny).contains("READY"));
        assert!(!buffer_text(&tiny).contains("openmax-app-render"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn completion_remains_visible_through_tiny_resize_round_trip() {
        let (mut app, dir) = app_fixture();
        app.composer.load("/");
        app.sync_completion();

        let wide_before = buffer_text(&render_app(&mut app, 96, 18));
        let tiny = buffer_text(&render_app(&mut app, 64, 4));
        let wide_after = buffer_text(&render_app(&mut app, 96, 18));

        assert!(wide_before.contains("/help"));
        assert!(tiny.contains("/help"));
        assert!(wide_after.contains("/help"));
        assert!(app.completion.is_some());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn live_tail_suffix_sync_matches_full_rebuild_across_tokens_and_resize() {
        let (mut incremental, incremental_dir) = app_fixture();
        let (mut oracle, oracle_dir) = app_fixture();
        let samples = [
            ("intro\n```rust\nlet first = 1;", 48),
            ("intro\n```rust\nlet first = 1; let second = 2;", 48),
            (
                "intro\n```rust\nlet first = 1; let second = 2;\nlet third = 3;",
                48,
            ),
            (
                "intro\n```rust\nlet first = 1; let second = 2;\nlet third = 3;",
                28,
            ),
            (
                "intro\n```rust\nlet first = 1; let second = 2;\nlet third = 3;\n```\ndone",
                28,
            ),
        ];

        for (text, width) in samples {
            incremental.stream_text = text.to_string();
            incremental.rebuild_tail(width);

            oracle.stream_text = text.to_string();
            oracle.tail_width = 0; // force the full-rebuild path
            oracle.rebuild_tail(width);

            assert_eq!(incremental.tail_buf, oracle.tail_buf);
            assert_eq!(
                incremental.tail_stable_len,
                incremental.thinking_wrapped.len() + incremental.stream_stable_len
            );
        }

        fs::remove_dir_all(incremental_dir).unwrap();
        fs::remove_dir_all(oracle_dir).unwrap();
    }

    #[test]
    fn counts_read_naturally_in_singular() {
        assert_eq!(plural(1, "skill"), "1 skill");
        assert_eq!(plural(0, "skill"), "0 skills");
        assert_eq!(plural(8, "tool"), "8 tools");
    }

}
