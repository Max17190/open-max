//! What the chat app does with each key, click, search, panel and agent
//! event, asserted where the person at the terminal meets it: the focus, the
//! selected block, the draft, the notes and the painted rows. They sit under
//! `app` beside its other tests because the binary has no library target.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use open_max_core::state::CancelToken;
use open_max_core::types::{AgentEvent, AgentEventEnvelope};
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::Terminal;

use super::tests::app_fixture;
use super::{
    approval_body_line, compact_approval_lines, elapsed_label, header_path_line,
    history_search_lines, rect_contains, scroll_search_lines, wide_status_right,
    wrap_str_preserving, App, Dirty, Focus, Mode, TermEvent, SPINNER,
};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn ctrl(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
}

fn shift(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::SHIFT)
}

async fn press(app: &mut App, key: KeyEvent) {
    app.on_key(key).await.unwrap();
}

async fn click(app: &mut App, column: u16, row: u16) {
    app.on_term_event(TermEvent::Mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }))
    .await
    .unwrap();
}

/// One frame of `app` at `width` x `height`: its rows, and the backend
/// holding the cursor the frame left.
fn frame(app: &mut App, width: u16, height: u16) -> (Vec<String>, TestBackend) {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| app.draw(frame)).unwrap();
    let backend = terminal.backend().clone();
    let buffer = backend.buffer();
    let rows = (0..height)
        .map(|y| (0..width).map(|x| buffer[(x, y)].symbol()).collect())
        .collect();
    (rows, backend)
}

fn line_text(line: &Line<'_>) -> String {
    line.spans.iter().map(|span| span.content.as_ref()).collect()
}

fn lines_text(lines: &[Line<'_>]) -> Vec<String> {
    lines.iter().map(line_text).collect()
}

/// Five blocks, prompts at 0, 2 and 4.
fn conversation(app: &mut App) {
    for turn in 1..=3 {
        app.insert_user_block(&format!("question {turn}"));
        if turn < 3 {
            app.transcript.push_assistant(vec![Line::from(format!("answer {turn}"))]);
        }
    }
}

/// The terminal's out-of-band writes (titles, bells, notifications,
/// clipboard), kept instead of thrown away.
#[derive(Clone, Default)]
struct Written(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Written {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Written {
    fn take(&self) -> String {
        String::from_utf8(std::mem::take(&mut *self.0.lock().unwrap())).unwrap()
    }
}

fn approval(id: &str, name: &str) -> AgentEvent {
    AgentEvent::ApprovalRequest {
        approval_id: id.into(),
        name: name.into(),
        summary: "run the tests".into(),
        detail: "cargo test".into(),
        reason: "gate".into(),
        source_path: String::new(),
        source_sha: String::new(),
        env: vec![],
    }
}

/// An answer channel for approval `id`, as a waiting turn registers one.
fn waiting_turn(app: &App, id: &str) -> tokio::sync::oneshot::Receiver<bool> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.core.approvals.lock().unwrap().insert(id.into(), tx);
    rx
}

#[tokio::test]
async fn scrollback_keys_walk_blocks_and_prompts_without_typing() {
    let (mut app, dir) = app_fixture();
    conversation(&mut app);
    press(&mut app, key(KeyCode::Tab)).await;
    assert!(app.focus == Focus::Scrollback);
    assert_eq!(app.transcript.selected(), Some(4), "focus lands on the newest block");

    // Each key, the block it selects. Plain arrows step one block; shifted
    // arrows and brackets jump between prompts.
    let steps = [
        (key(KeyCode::Up), 3),
        (key(KeyCode::Char('j')), 4),
        (key(KeyCode::Char('k')), 3),
        (key(KeyCode::Down), 4),
        (shift(KeyCode::Up), 2),
        (key(KeyCode::Down), 3),
        (key(KeyCode::Char('[')), 2),
        (key(KeyCode::Char(']')), 4),
        (key(KeyCode::Char('[')), 2),
        (shift(KeyCode::Down), 4),
        (key(KeyCode::Char('g')), 0),
        (key(KeyCode::Char('o')), 0),
        (key(KeyCode::Char('n')), 0),
    ];
    for (step, (key, selected)) in steps.into_iter().enumerate() {
        press(&mut app, key).await;
        assert_eq!(app.transcript.selected(), Some(selected), "step {step}: {:?}", key.code);
        assert!(app.focus == Focus::Scrollback, "step {step} left the scrollback");
        assert_eq!(app.composer.text(), "", "step {step} typed into the draft");
    }

    press(&mut app, key(KeyCode::Char(' '))).await;
    assert!(app.focus == Focus::Composer);
    assert_eq!(app.transcript.selected(), None);
    assert_eq!(app.composer.text(), "", "space leaves the scrollback, it types nothing");

    // Tab keeps a block another surface selected, and Tab back clears it.
    app.transcript.select_block(1);
    press(&mut app, key(KeyCode::Tab)).await;
    assert_eq!(app.transcript.selected(), Some(1));
    press(&mut app, key(KeyCode::Tab)).await;
    assert!(app.focus == Focus::Composer);
    assert_eq!(app.transcript.selected(), None);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn page_keys_scroll_the_transcript_one_page() {
    let (mut app, dir) = app_fixture();
    for i in 0..40 {
        app.note(&format!("line {i}"));
    }
    frame(&mut app, 40, 12);
    let page = app.page_h as usize;
    assert!(page > 1);
    press(&mut app, key(KeyCode::PageUp)).await;
    assert_eq!(app.transcript.offset(), page);
    press(&mut app, key(KeyCode::PageUp)).await;
    assert_eq!(app.transcript.offset(), 2 * page);
    press(&mut app, key(KeyCode::PageDown)).await;
    assert_eq!(app.transcript.offset(), page);
    assert_eq!(app.composer.text(), "");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn approval_keys_and_clicks_answer_the_waiting_turn() {
    for (answer, approved) in [
        (key(KeyCode::Char('y')), true),
        (key(KeyCode::Char('Y')), true),
        (key(KeyCode::Char('n')), false),
        (key(KeyCode::Char('N')), false),
        (key(KeyCode::Esc), false),
    ] {
        let (mut app, dir) = app_fixture();
        let mut turn = waiting_turn(&app, "ap");
        app.on_agent_event(approval("ap", "bash"));
        press(&mut app, answer).await;
        assert_eq!(turn.try_recv(), Ok(approved), "{:?}", answer.code);
        let _ = std::fs::remove_dir_all(dir);
    }

    // A click on a choice the card painted answers as its key does.
    for (label, approved) in [("Allow once", true), ("Deny", false)] {
        let (mut app, dir) = app_fixture();
        let mut turn = waiting_turn(&app, "ap");
        app.on_agent_event(approval("ap", "bash"));
        let (rows, _) = frame(&mut app, 60, 20);
        let (row, text) = rows
            .iter()
            .enumerate()
            .find(|(_, row)| row.contains(label))
            .expect("the card paints its choices");
        let column = text[..text.find(label).unwrap()].chars().count() as u16;
        click(&mut app, column, row as u16).await;
        assert_eq!(turn.try_recv(), Ok(approved), "{label}");
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[tokio::test]
async fn ctrl_t_flips_thinking_and_says_so_only_between_turns() {
    let (mut app, dir) = app_fixture();
    let shown = app.show_thinking;
    press(&mut app, ctrl('t')).await;
    assert_eq!(app.show_thinking, !shown);
    let first = app.transcript.export_text();
    press(&mut app, ctrl('t')).await;
    assert_eq!(app.show_thinking, shown);
    let text = app.transcript.export_text();
    assert!(text.contains("thinking shown while the model streams"), "{text}");
    assert!(text.contains("thinking hidden (ctrl+t to show)"), "{text}");
    assert!(first.lines().count() < text.lines().count());

    app.running = true;
    let before = app.transcript.block_count();
    press(&mut app, ctrl('t')).await;
    assert_eq!(app.transcript.block_count(), before, "a running turn shows the change itself");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn ctrl_o_opens_the_paste_under_the_cursor_unless_an_approval_waits() {
    let pasted: String = (0..12).map(|i| format!("pasted line {i}\n")).collect();
    let (mut app, dir) = app_fixture();
    app.composer.insert_paste(&pasted);
    assert!(app.composer.text().contains("pasted line 11"));
    let collapsed = app.composer.cursor_context().2.to_string();
    assert!(collapsed.contains("[pasted #1"), "{collapsed}");
    press(&mut app, ctrl('o')).await;
    assert!(app.composer.cursor_context().0 > 0, "the paste opened into its own lines");

    // While an approval waits, Ctrl+O is the tool block's, never the draft's.
    let (mut app, dir2) = app_fixture();
    app.composer.insert_paste(&pasted);
    app.transcript.push_tool(vec![Line::from("ran a tool")], "full output".into(), true);
    app.on_agent_event(approval("ap", "bash"));
    press(&mut app, ctrl('o')).await;
    assert_eq!(app.composer.cursor_context().0, 0, "the paste stayed collapsed");
    assert!(app.focus == Focus::Scrollback, "the tool block opened");
    assert_eq!(app.transcript.selected(), Some(0));
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(dir2);
}

#[tokio::test]
async fn find_and_history_search_open_only_when_nothing_else_owns_the_keys() {
    let (mut app, dir) = app_fixture();
    conversation(&mut app);
    app.composer.load("an earlier prompt");
    let _ = app.composer.handle_key(key(KeyCode::Enter));
    press(&mut app, ctrl('r')).await;
    assert!(app.history_search.is_some(), "ctrl+r opens history search");
    press(&mut app, key(KeyCode::Esc)).await;
    press(&mut app, ctrl('f')).await;
    assert!(app.scroll_search.is_some(), "ctrl+f opens find");
    press(&mut app, key(KeyCode::Esc)).await;

    // Another ctrl chord opens neither.
    press(&mut app, ctrl('e')).await;
    assert!(app.history_search.is_none() && app.scroll_search.is_none());

    // Neither opens over the model picker, and neither over an approval.
    app.mode = Mode::ModelPicker;
    press(&mut app, ctrl('r')).await;
    press(&mut app, ctrl('f')).await;
    assert!(app.history_search.is_none() && app.scroll_search.is_none());
    app.mode = Mode::Chat;
    app.on_agent_event(approval("ap", "bash"));
    press(&mut app, ctrl('f')).await;
    assert!(app.scroll_search.is_none(), "find opened over an approval");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn esc_clears_selections_before_it_follows_or_cancels() {
    let (mut app, dir) = app_fixture();
    for i in 0..40 {
        app.note(&format!("line {i}"));
    }
    app.composer.load("alpha beta");
    let (rows, _) = frame(&mut app, 40, 12);
    let area = app.composer_draw_area;
    let draft = &rows[area.y as usize];
    let beta = draft[..draft.find("beta").unwrap()].chars().count() as u16;
    // Double-click a word of the draft, then a third click takes the line.
    click(&mut app, beta, area.y).await;
    click(&mut app, beta, area.y).await;
    assert_eq!(app.composer.selected_text().as_deref(), Some("beta"));
    click(&mut app, beta, area.y).await;
    assert_eq!(app.composer.selected_text().as_deref(), Some("alpha beta"));

    app.transcript.scroll_up(3);
    press(&mut app, key(KeyCode::Esc)).await;
    assert!(!app.composer.has_selection());
    assert_eq!(app.transcript.offset(), 3, "clearing a selection is all Esc did");

    // So does a selection in the transcript.
    let row = rows.iter().position(|row| row.contains("line ")).unwrap() as u16;
    for _ in 0..3 {
        click(&mut app, 5, row).await;
    }
    assert!(app.transcript.has_text_selection());
    press(&mut app, key(KeyCode::Esc)).await;
    assert!(!app.transcript.has_text_selection());
    assert_eq!(app.transcript.offset(), 3);

    // The click focused the scrollback: Esc leaves it, the next one follows
    // the live view, and at the bottom of a running turn Esc cancels it.
    press(&mut app, key(KeyCode::Esc)).await;
    assert!(app.focus == Focus::Composer);
    assert_eq!(app.transcript.offset(), 3);
    press(&mut app, key(KeyCode::Esc)).await;
    assert_eq!(app.transcript.offset(), 0, "Esc follows the live view");
    let token = Arc::new(CancelToken::default());
    app.core.cancel_flags.lock().unwrap().insert("s".into(), token.clone());
    app.session_id = Some("s".into());
    app.running = true;
    press(&mut app, key(KeyCode::Esc)).await;
    assert!(token.is_cancelled());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn shift_or_alt_enter_in_a_half_typed_command_is_a_newline() {
    for modifiers in [KeyModifiers::SHIFT, KeyModifiers::ALT] {
        let (mut app, dir) = app_fixture();
        app.composer.load("/sta");
        app.sync_completion();
        assert!(app.completion.is_some());
        press(&mut app, KeyEvent::new(KeyCode::Enter, modifiers)).await;
        assert_eq!(app.composer.text(), "/sta\n", "{modifiers:?}");
        assert_eq!(app.transcript.block_count(), 0, "nothing ran");
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[tokio::test]
async fn terminal_events_reach_only_the_surface_they_belong_to() {
    let (mut app, dir) = app_fixture();
    // A key's release is not another press.
    let release = KeyEvent {
        code: KeyCode::Char('x'),
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Release,
        state: KeyEventState::NONE,
    };
    app.on_term_event(TermEvent::Key(release)).await.unwrap();
    assert_eq!(app.composer.text(), "");
    app.on_term_event(TermEvent::Paste("pasted".into())).await.unwrap();
    assert_eq!(app.composer.text(), "pasted");

    // A waiting approval owns the keyboard, pastes included.
    app.on_agent_event(approval("ap", "bash"));
    app.on_term_event(TermEvent::Paste(" more".into())).await.unwrap();
    assert_eq!(app.composer.text(), "pasted");

    app.dirty = Dirty::default();
    app.on_term_event(TermEvent::Resize(80, 24)).await.unwrap();
    assert_eq!(app.dirty, Dirty::all(), "a resize repaints everything");
    app.on_term_event(TermEvent::FocusLost).await.unwrap();
    assert_eq!(app.term_focus, Some(false));
    app.on_term_event(TermEvent::FocusGained).await.unwrap();
    assert_eq!(app.term_focus, Some(true));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn model_picker_keys_move_filter_and_close() {
    let (mut app, dir) = app_fixture();
    std::fs::write(
        dir.join("providers.json"),
        r#"{"providers":{"alpha":{"base_url":"http://alpha/v1",
           "models":[{"id":"one"},{"id":"two"},{"id":"three"}]}}}"#,
    )
    .unwrap();
    app.model_picker = Some(crate::ui::model_picker::ModelPickerState::load(&dir, Some("alpha"), "one"));
    app.mode = Mode::ModelPicker;
    let selected = |app: &App| app.model_picker.as_ref().unwrap().selected;
    let query = |app: &App| app.model_picker.as_ref().unwrap().query.clone();
    let start = selected(&app);
    press(&mut app, key(KeyCode::Down)).await;
    assert_eq!(selected(&app), (start + 1) % 3);
    press(&mut app, key(KeyCode::Tab)).await;
    assert_eq!(selected(&app), (start + 2) % 3);
    press(&mut app, key(KeyCode::Up)).await;
    assert_eq!(selected(&app), (start + 1) % 3);

    press(&mut app, key(KeyCode::Char('t'))).await;
    assert_eq!(query(&app), "t");
    press(&mut app, ctrl('w')).await;
    assert_eq!(query(&app), "t", "a ctrl chord is not filter text");
    press(&mut app, key(KeyCode::Backspace)).await;
    assert_eq!(query(&app), "");
    press(&mut app, key(KeyCode::Esc)).await;
    assert!(app.mode == Mode::Chat && app.model_picker.is_none());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn history_search_steps_within_its_matches_and_filters_as_typed() {
    let (mut app, dir) = app_fixture();
    for prompt in ["alpha one", "beta two", "alpha three"] {
        app.composer.load(prompt);
        let _ = app.composer.handle_key(key(KeyCode::Enter));
    }
    press(&mut app, ctrl('r')).await;
    let selected = |app: &App| app.history_search.as_ref().unwrap().1;
    assert_eq!(selected(&app), 2, "the newest prompt first");
    for expected in [1, 0, 0] {
        press(&mut app, key(KeyCode::Up)).await;
        assert_eq!(selected(&app), expected);
    }
    for expected in [1, 2, 2] {
        press(&mut app, key(KeyCode::Down)).await;
        assert_eq!(selected(&app), expected);
    }

    press(&mut app, key(KeyCode::Char('a'))).await;
    press(&mut app, key(KeyCode::Char('l'))).await;
    press(&mut app, ctrl('k')).await;
    let (query, at, matches) = app.history_search.clone().unwrap();
    assert_eq!(query, "al");
    assert_eq!(matches, ["alpha one", "alpha three"]);
    assert_eq!(at, 1, "the pick stays inside the narrower list");
    press(&mut app, key(KeyCode::Backspace)).await;
    assert_eq!(app.history_search.as_ref().unwrap().0, "a");
    assert_eq!(app.history_search.as_ref().unwrap().2.len(), 3);

    press(&mut app, key(KeyCode::Esc)).await;
    assert!(app.history_search.is_none());
    assert_eq!(app.composer.text(), "", "Esc inserts nothing");
    let _ = std::fs::remove_dir_all(dir);
}

/// Blocks 0 to 6; "apple" in 1, 3 and 5.
fn fruit(app: &mut App) {
    for text in ["intro", "apple one", "banana", "apple two", "cherry", "apple three", "outro"] {
        app.transcript.push_assistant(vec![Line::from(text)]);
    }
}

#[tokio::test]
async fn find_steps_through_matches_and_n_continues_after_it_closes() {
    let (mut app, dir) = app_fixture();
    fruit(&mut app);
    press(&mut app, ctrl('f')).await;
    for c in "apple".chars() {
        press(&mut app, key(KeyCode::Char(c))).await;
    }
    press(&mut app, ctrl('a')).await;
    let (query, _, matches) = app.scroll_search.clone().unwrap();
    assert_eq!((query.as_str(), matches.as_slice()), ("apple", &[1, 3, 5][..]));
    assert_eq!(app.transcript.selected(), Some(5), "typing lands on the newest match");
    press(&mut app, key(KeyCode::Up)).await;
    assert_eq!(app.transcript.selected(), Some(3));
    press(&mut app, key(KeyCode::Up)).await;
    assert_eq!(app.transcript.selected(), Some(1));
    press(&mut app, key(KeyCode::Down)).await;
    assert_eq!(app.transcript.selected(), Some(3));
    press(&mut app, key(KeyCode::Esc)).await;
    assert!(app.scroll_search.is_none());
    assert_eq!(app.scroll_search_last.as_deref(), Some("apple"));

    // n and N step the closed search from the selected block, wrapping.
    app.focus = Focus::Scrollback;
    for (step, expected) in [('n', 5), ('n', 1), ('N', 5), ('N', 3)] {
        press(&mut app, key(KeyCode::Char(step))).await;
        assert_eq!(app.transcript.selected(), Some(expected), "{step}");
    }
    // From a block that is not a match: the next match after it, or the
    // last one before it, wrapping past either end.
    for (from, step, expected) in [(4, 'n', 5), (4, 'N', 3), (0, 'N', 5), (6, 'n', 1), (2, 'n', 3)] {
        app.transcript.select_block(from);
        press(&mut app, key(KeyCode::Char(step))).await;
        assert_eq!(app.transcript.selected(), Some(expected), "{step} from {from}");
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn find_keeps_its_last_query_however_it_closes() {
    for close in [KeyCode::Enter, KeyCode::Tab] {
        let (mut app, dir) = app_fixture();
        fruit(&mut app);
        press(&mut app, ctrl('f')).await;
        press(&mut app, key(KeyCode::Char('b'))).await;
        press(&mut app, key(KeyCode::Backspace)).await;
        press(&mut app, key(KeyCode::Char('c'))).await;
        assert_eq!(app.scroll_search.as_ref().unwrap().0, "c");
        press(&mut app, key(close)).await;
        assert!(app.scroll_search.is_none());
        assert_eq!(app.scroll_search_last.as_deref(), Some("c"), "{close:?}");
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[tokio::test]
async fn an_open_find_takes_in_new_blocks_and_keeps_its_place() {
    let (mut app, dir) = app_fixture();
    fruit(&mut app);
    press(&mut app, ctrl('f')).await;
    for c in "apple".chars() {
        press(&mut app, key(KeyCode::Char(c))).await;
    }
    press(&mut app, key(KeyCode::Up)).await;
    assert_eq!(app.transcript.selected(), Some(3));
    app.on_agent_event(AgentEvent::MessageDone { text: "apple four".into() });
    let (_, at, matches) = app.scroll_search.clone().unwrap();
    assert_eq!(matches, [1, 3, 5, 7]);
    assert_eq!((at, app.transcript.selected()), (1, Some(3)), "the reader keeps their place");

    // A query that matched nothing takes the first block that does.
    let (mut app, dir2) = app_fixture();
    fruit(&mut app);
    press(&mut app, ctrl('f')).await;
    for c in "kiwi".chars() {
        press(&mut app, key(KeyCode::Char(c))).await;
    }
    assert!(app.scroll_search.as_ref().unwrap().2.is_empty());
    app.on_agent_event(AgentEvent::MessageDone { text: "kiwi".into() });
    assert_eq!(app.transcript.selected(), Some(7));
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_dir_all(dir2);
}

#[tokio::test]
async fn sessions_panel_moves_deletes_and_closes() {
    use open_max_core::sessions;
    let (mut app, dir) = app_fixture();
    let project = dir.display().to_string();
    let ids: Vec<String> =
        (0..3).map(|_| sessions::create(&app.core, project.clone()).unwrap().id).collect();
    app.handle_submit("/resume".into()).await.unwrap();
    assert!(app.mode == Mode::Sessions);
    let at = |app: &App| app.sessions_panel.as_ref().unwrap().selected;
    for expected in [1, 2, 2] {
        press(&mut app, key(KeyCode::Down)).await;
        assert_eq!(at(&app), expected);
    }
    press(&mut app, key(KeyCode::Up)).await;
    assert_eq!(at(&app), 1);
    press(&mut app, key(KeyCode::Char('k'))).await;
    assert_eq!(at(&app), 0);

    // Deleting the open session leaves the app without one.
    let doomed = app.sessions_panel.as_ref().unwrap().items[0].id.clone();
    let kept = app.sessions_panel.as_ref().unwrap().items[1].id.clone();
    app.session_id = Some(doomed.clone());
    press(&mut app, key(KeyCode::Char('x'))).await;
    assert_eq!(app.sessions_panel.as_ref().unwrap().confirm_delete.as_deref(), Some(doomed.as_str()));
    press(&mut app, key(KeyCode::Char('y'))).await;
    let left: Vec<String> =
        app.sessions_panel.as_ref().unwrap().items.iter().map(|s| s.id.clone()).collect();
    assert_eq!(left.len(), 2);
    assert!(!left.contains(&doomed) && left.iter().all(|id| ids.contains(id)));
    assert_eq!(app.session_id, None);

    // Deleting another session keeps the open one.
    app.session_id = Some(kept.clone());
    let other = left.iter().find(|id| **id != kept).unwrap().clone();
    let row = left.iter().position(|id| *id == other).unwrap();
    app.sessions_panel.as_mut().unwrap().selected = row;
    press(&mut app, key(KeyCode::Char('x'))).await;
    press(&mut app, key(KeyCode::Char('y'))).await;
    assert_eq!(app.session_id.as_deref(), Some(kept.as_str()));

    press(&mut app, key(KeyCode::Char('q'))).await;
    assert!(app.mode == Mode::Chat && app.sessions_panel.is_none());
    app.handle_submit("/resume".into()).await.unwrap();
    press(&mut app, key(KeyCode::Esc)).await;
    assert!(app.mode == Mode::Chat && app.sessions_panel.is_none());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn thinking_keeps_only_its_last_600_bytes_cut_on_a_character() {
    let (mut app, dir) = app_fixture();
    let text = format!("{}x", "é".repeat(350));
    app.on_agent_event(AgentEvent::Thinking { text: text.clone() });
    // 701 bytes, 101 over: the cut moves up to the next character, 102.
    assert_eq!(app.thinking_tail, text[102..]);
    assert_eq!(app.thinking_chars, 351);
    assert_eq!(app.stream_chars, 701);
    app.on_agent_event(AgentEvent::Thinking { text: "y".into() });
    assert_eq!(app.thinking_tail.len(), 600);
    assert!(app.thinking_tail.ends_with("xy"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_running_line_reports_rate_and_thinking_only_when_there_is_some() {
    let (mut app, dir) = app_fixture();
    app.running = true;
    app.turn_started = Some(Instant::now());
    let running_row = |app: &mut App| {
        let (rows, _) = frame(app, 100, 20);
        rows.into_iter().find(|row| row.contains("esc to cancel")).unwrap()
    };
    let row = running_row(&mut app);
    assert!(!row.contains("tok/s") && !row.contains("thinking"), "{row}");

    app.on_agent_event(AgentEvent::Thinking { text: "hmm".into() });
    let row = running_row(&mut app);
    assert!(row.contains("thinking (ctrl+t to peek)") && !row.contains("tok/s"), "{row}");

    // Two seconds of output at four characters a token: 100 tokens/s.
    app.first_token = Some(Instant::now() - Duration::from_secs(2));
    app.on_agent_event(AgentEvent::Token { text: "x".repeat(797) });
    app.dirty = Dirty::all();
    let row = running_row(&mut app);
    assert!(row.contains("· 100 tok/s"), "{row}");
    assert!(!row.contains("thinking"), "the answer streaming replaces the thinking label: {row}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn usage_without_prompt_tokens_reports_no_cache_rate() {
    let (mut app, dir) = app_fixture();
    app.on_agent_event(AgentEvent::Usage { prompt_tokens: 200, completion_tokens: 7, cached_tokens: Some(50) });
    assert_eq!(app.cache_pct, Some(25));
    app.on_agent_event(AgentEvent::Usage { prompt_tokens: 0, completion_tokens: 3, cached_tokens: Some(0) });
    assert_eq!(app.cache_pct, None);
    assert_eq!(app.generated_tokens, 10);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_unapproved_source_names_its_file_before_the_detail() {
    let (mut app, dir) = app_fixture();
    let request = |reason: &str, source_path: &str| AgentEvent::ApprovalRequest {
        approval_id: "ap".into(),
        name: "danger".into(),
        summary: "danger".into(),
        detail: "{\"count\":3}".into(),
        reason: reason.into(),
        source_path: source_path.into(),
        source_sha: "abc123".into(),
        env: vec![],
    };
    app.on_agent_event(request("unapproved_source", ".openmax/tools/danger.toml"));
    assert_eq!(
        app.pending_approval.as_ref().unwrap().3,
        "unapproved tool content (its .toml or the code it runs) · .openmax/tools/danger.toml (abc123) · {\"count\":3}"
    );
    app.on_agent_event(request("unapproved_source", ""));
    assert_eq!(
        app.pending_approval.as_ref().unwrap().3,
        "unapproved tool content (its .toml or the code it runs) · {\"count\":3}"
    );
    app.on_agent_event(request("gate", ".openmax/tools/danger.toml"));
    assert_eq!(app.pending_approval.as_ref().unwrap().3, "{\"count\":3}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_settled_approval_says_why_it_ended() {
    for (outcome, note) in [("timed_out", Some("approval timed out · declined")), ("cancelled", Some("approval cancelled")), ("approved", None)] {
        let (mut app, dir) = app_fixture();
        app.on_agent_event(approval("ap", "bash"));
        let before = app.transcript.block_count();
        app.on_agent_event(AgentEvent::ApprovalSettled { approval_id: "ap".into(), outcome: outcome.into() });
        assert!(app.pending_approval.is_none());
        match note {
            Some(note) => assert!(app.transcript.export_text().contains(note), "{outcome}"),
            None => assert_eq!(app.transcript.block_count(), before, "{outcome}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn a_refreeze_names_three_changes_and_counts_the_rest() {
    let (mut app, dir) = app_fixture();
    let refrozen = |n: usize| AgentEvent::Refrozen {
        tools: 1,
        skills: 0,
        changes: (0..n).map(|i| format!("c{i}.toml")).collect(),
        runs: vec![],
    };
    app.on_agent_event(refrozen(5));
    app.on_agent_event(refrozen(3));
    let text = app.transcript.export_text();
    assert!(text.contains(" - c0.toml, c1.toml, c2.toml (+2 more)"), "{text}");
    assert!(text.trim_end().ends_with(" - c0.toml, c1.toml, c2.toml"), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_compaction_receipt_ends_the_compaction_and_sends_the_queue() {
    for queued in [vec![], vec!["next".to_string()]] {
        let (mut app, dir) = app_fixture();
        app.compacting = true;
        app.running = true;
        app.queued = queued.clone();
        app.on_agent_event(AgentEvent::Compacted { tokens_before: 900, tokens_after: 900, compacted_messages: 0 });
        assert!(!app.running && !app.compacting);
        assert_eq!(app.flush_queue, !queued.is_empty());
        assert!(app.transcript.export_text().contains("nothing to prune (~900 tokens)"));
        app.on_agent_event(AgentEvent::Compacted { tokens_before: 900, tokens_after: 300, compacted_messages: 2 });
        assert!(app.transcript.export_text().contains("compacted: ~900 to ~300 tokens, 2 messages archived"));
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn every_line_of_an_error_after_the_first_is_indented() {
    let (mut app, dir) = app_fixture();
    app.error("first\nsecond\nthird");
    let text = app.transcript.export_text();
    assert!(text.contains("> ✗ first\n>   second\n>   third\n"), "{text:?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_finished_turn_records_its_time_and_rate_only_when_long_enough() {
    let stats = |secs: u64, chars: usize, first_token: Option<u64>| {
        let (mut app, dir) = app_fixture();
        app.turn_started = Some(Instant::now() - Duration::from_secs(secs));
        app.first_token = first_token.map(|ago| Instant::now() - Duration::from_secs(ago));
        app.stream_chars = chars;
        app.running = true;
        app.on_agent_event(AgentEvent::Done { stop_reason: "stop".into() });
        let text = app.transcript.export_text();
        let _ = std::fs::remove_dir_all(dir);
        text
    };
    assert!(!stats(4, 0, None).contains('◦'), "a short turn leaves no record");
    assert!(stats(5, 0, None).trim_end().ends_with("> ◦ 5s"));
    assert!(stats(6, 800, Some(2)).trim_end().ends_with("> ◦ 6s · 100 tok/s"));
}

#[tokio::test]
async fn a_settled_turn_sends_the_queue_and_returns_what_cannot_start() {
    let envelope = || AgentEventEnvelope {
        session_id: "elsewhere".into(),
        event: AgentEvent::Done { stop_reason: "stop".into() },
    };
    // A turn still running keeps the queue waiting.
    let (mut app, dir) = app_fixture();
    app.queued = vec!["next question".into()];
    app.flush_queue = true;
    app.running = true;
    app.on_core_event(envelope()).await;
    assert_eq!(app.queued, ["next question"]);
    assert_eq!(app.composer.text(), "");

    // Settled: the queue goes out. A session another turn still holds
    // refuses the next one before it starts, so the message comes back to
    // the draft instead of being lost.
    app.session_id = Some("busy".into());
    app.core.running.lock().unwrap().insert("busy".into());
    app.running = false;
    app.flush_queue = true;
    app.on_core_event(envelope()).await;
    assert!(app.queued.is_empty());
    assert_eq!(app.composer.text(), "next question");
    assert!(app.transcript.export_text().contains("queued input returned to the composer"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn presence_titles_ring_for_an_approval_and_notify_a_finished_turn() {
    let (mut app, dir) = app_fixture();
    let written = Written::default();
    app.term_out = Box::new(written.clone());
    let project = super::project_label(&app.project);
    app.on_focus_change(false);
    app.on_agent_event(approval("ap", "bash"));
    assert_eq!(written.take(), format!("\x1b]0;{project} · openmax · needs approval\x07\x07"));
    app.on_agent_event(AgentEvent::ApprovalSettled { approval_id: "ap".into(), outcome: "approved".into() });
    assert_eq!(written.take(), format!("\x1b]0;{project} · openmax · working\x07"));
    app.on_agent_event(AgentEvent::Done { stop_reason: "stop".into() });
    assert_eq!(
        written.take(),
        format!("\x1b]0;{project} · openmax\x07\x1b]9;{project} · turn complete\x07\x07")
    );

    // A copied selection goes to the clipboard and says so.
    app.composer.load("copy me");
    let (_, _) = frame(&mut app, 40, 10);
    let area = app.composer_draw_area;
    for _ in 0..3 {
        click(&mut app, area.x + 3, area.y).await;
    }
    press(&mut app, ctrl('c')).await;
    assert!(written.take().starts_with("\x1b]52;c;"));
    assert!(app.transcript.export_text().contains("copied selection"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn the_spinner_turns_only_while_a_turn_runs() {
    let (mut app, dir) = app_fixture();
    assert!(!app.tick_armed());
    app.on_tick().await;
    assert_eq!(app.spinner_i, 0);
    app.running = true;
    assert!(app.tick_armed());
    app.on_tick().await;
    assert_eq!(app.spinner_i, 1);
    for _ in 1..SPINNER.len() {
        app.on_tick().await;
    }
    assert_eq!(app.spinner_i, 0, "the spinner wraps after its last frame");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_approval_body_gives_the_env_grant_the_width_first() {
    let env = vec!["TOKEN".to_string()];
    let text = |detail: &str, env: &[String], width| line_text(&approval_body_line(detail, env, width));
    assert_eq!(text("cargo test", &[], 40), "cargo test");
    assert_eq!(text("", &[], 40), "");
    assert_eq!(text("cargo test", &env, 40), "receives env: TOKEN  cargo test");
    // "receives env: TOKEN" is 19 cells: two more leave room for the gap
    // only, three for one cell of the detail.
    assert_eq!(text("cargo test", &env, 22), "receives env: TOKEN  …");
    assert_eq!(text("cargo test", &env, 21), "receives env: TOKEN");
    assert_eq!(text("cargo test", &env, 20), "receives env: TOKEN");
    assert_eq!(text("cargo test", &env, 19), "receives env: TOKEN");
}

#[test]
fn a_two_row_approval_card_shows_the_grant_before_the_call() {
    let env = vec!["TOKEN".to_string()];
    let lines = |env: &[String], height| lines_text(&compact_approval_lines("bash", "ls", "", env, 60, height));
    assert_eq!(lines(&env, 2), ["receives env: TOKEN  ls", "▸ [y] Allow once [a] Auto for project [n] Deny"]);
    assert_eq!(lines(&[], 2)[1], "▸ [y] Allow once [a] Auto for project [n] Deny");
    assert!(lines(&[], 2)[0].ends_with("ls"));
    assert!(!lines(&[], 2)[0].contains("receives env"));
    let three = lines(&env, 3);
    assert_eq!(three.len(), 3);
    assert!(!three[0].contains("receives env") && three[1] == "receives env: TOKEN  ls", "{three:?}");
}

#[test]
fn wrapping_a_grant_splits_long_words_and_keeps_every_character() {
    assert_eq!(wrap_str_preserving("abcdefgh", 3), ["abc", "def", "gh"]);
    assert_eq!(wrap_str_preserving("ab abcdefgh", 3), ["ab", "abc", "def", "gh"]);
    assert_eq!(wrap_str_preserving("ab cd", 5), ["ab cd"]);
}

#[test]
fn small_pure_rules_hold_at_their_edges() {
    let rect = Rect::new(2, 3, 4, 2);
    assert!(rect_contains(rect, 2, 3) && rect_contains(rect, 5, 4));
    assert!(!rect_contains(rect, 6, 3) && !rect_contains(rect, 2, 5));
    assert!(!rect_contains(rect, 1, 3) && !rect_contains(rect, 2, 2));
    assert_eq!(elapsed_label(Duration::from_millis(9_940)), "9.9s");
    assert_eq!(elapsed_label(Duration::from_secs(10)), "10s");
    assert_eq!(wide_status_right("model", None, None, 0, "ask", 100), "model  ask ");
    assert_eq!(wide_status_right("model", None, None, 1, "ask", 100), "model  out 1  ask ");
}

#[test]
fn the_header_keeps_a_trailing_slash_and_needs_two_cells() {
    let text = |path: &str, width| line_text(&header_path_line(std::path::Path::new(path), width));
    let line = header_path_line(std::path::Path::new("/srv/proj/"), 80);
    assert_eq!(line.spans.len(), 2);
    assert_eq!(line.spans[0].content, "");
    assert_eq!(line.spans[1].content, "/srv/proj/");
    assert_eq!(text("/srv/proj", 80), "/srv/proj");
    assert_eq!(text("/", 80), "/");
    assert_eq!(text("/srv/proj", 1), "");
    assert!(!text("/srv/proj", 2).is_empty());
}

#[test]
fn history_search_shows_six_rows_around_the_pick() {
    let items: Vec<String> = (0..10).map(|i| format!("p{i}")).collect();
    let rows = |selected| lines_text(&history_search_lines("p", selected, &items, 40));
    let picked = |selected| rows(selected).iter().filter(|row| row.starts_with("▸ ")).cloned().collect::<Vec<_>>();
    assert_eq!(rows(0)[1..7], ["▸ p0", "  p1", "  p2", "  p3", "  p4", "  p5"]);
    assert_eq!(rows(7)[1..7], ["  p2", "  p3", "  p4", "  p5", "  p6", "▸ p7"]);
    assert_eq!(rows(9)[1..7], ["  p4", "  p5", "  p6", "  p7", "  p8", "▸ p9"]);
    assert_eq!(picked(9), ["▸ p9"]);
    assert_eq!(rows(9)[7], "  … 4 more (keep typing)");
    let six = &items[..6];
    assert_eq!(history_search_lines("p", 0, six, 40).len(), 7, "all six shown, nothing more to say");
}

#[test]
fn find_shows_six_previews_around_the_pick() {
    let (mut app, dir) = app_fixture();
    for i in 0..10 {
        app.transcript.push_assistant(vec![Line::from(format!("block {i}"))]);
    }
    let matches: Vec<usize> = (0..10).collect();
    let mut previews = std::collections::HashMap::new();
    let mut rows = |selected| {
        previews.clear();
        lines_text(&scroll_search_lines("block", selected, &matches, &app.transcript, &mut previews, 40))
    };
    assert!(rows(0)[0].ends_with("1/10"));
    assert_eq!(rows(7)[1..7], ["  block 2", "  block 3", "  block 4", "  block 5", "  block 6", "▸ block 7"]);
    assert_eq!(rows(9)[1..7], ["  block 4", "  block 5", "  block 6", "  block 7", "  block 8", "▸ block 9"]);
    assert_eq!(rows(9).len(), 7);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_status_row_fits_each_width_and_redraws_when_it_changes() {
    let (mut app, dir) = app_fixture();
    app.core.settings.lock().unwrap().model = "test-model".into();
    app.generated_tokens = 1_500;
    let status = |app: &mut App, width| {
        let (rows, _) = frame(app, width, 12);
        let border = rows.iter().position(|row| row.starts_with('╭')).unwrap();
        rows[border - 1].clone()
    };
    let wide = status(&mut app, 100);
    assert!(wide.ends_with("test-model  out 1.5k  ask "), "{wide:?}");
    assert!(wide.starts_with(" shift+tab effort · /permissions"), "{wide:?}");

    // A narrower frame with nothing else changed still lays the row out
    // again for its own width.
    app.dirty = Dirty::default();
    let medium = status(&mut app, 78);
    assert!(medium.ends_with("test-model  ask "), "{medium:?}");
    assert_eq!(medium.chars().count(), 78);
    app.dirty = Dirty::default();
    let narrow = status(&mut app, 54);
    assert!(narrow.ends_with("test-model  ask "), "{narrow:?}");
    // Below 54 cells only the model is left, at most half the row.
    app.dirty = Dirty::default();
    let tight = status(&mut app, 30);
    assert_eq!(tight, " shift+tab effort… test-model ");
    app.dirty = Dirty::default();
    assert_eq!(status(&mut app, 4), "te… ", "four cells hold a clipped model");
    app.dirty = Dirty::default();
    assert_eq!(status(&mut app, 3), " … ", "three hold no model");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_terminal_cursor_sits_at_the_drafts_caret() {
    let (mut app, dir) = app_fixture();
    app.composer.load("ab\ncd");
    let (_, backend) = frame(&mut app, 40, 12);
    let area = app.composer_draw_area;
    // Two cells of prompt mark, then the caret after "cd" on the second row.
    assert_eq!(area.x, 1);
    assert_eq!(backend.cursor_position(), ratatui::layout::Position::new(5, area.y + 1));
    assert!(backend.cursor_visible());

    // Too short for a box: the draft keeps its rows, bottom aligned.
    for (draft, area_rows, cursor) in [("ab", (1, 1), (4, 1)), ("ab\ncd", (0, 2), (4, 1))] {
        let (mut app, dir2) = app_fixture();
        app.composer.load(draft);
        let (rows, backend) = frame(&mut app, 20, 2);
        let area = app.composer_draw_area;
        assert_eq!((area.y, area.height), area_rows, "{rows:?}");
        assert_eq!(backend.cursor_position(), ratatui::layout::Position::new(cursor.0, cursor.1), "{rows:?}");
        assert!(backend.cursor_visible());
        let _ = std::fs::remove_dir_all(dir2);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// One step of a turn as the screen sees it.
#[derive(Clone)]
enum Step {
    Event(AgentEvent),
    ToggleThinking,
    Width(u16),
}

async fn apply(app: &mut App, step: &Step) {
    match step {
        Step::Event(event) => app.on_agent_event(event.clone()),
        Step::ToggleThinking => press(app, ctrl('t')).await,
        Step::Width(_) => {}
    }
}

/// The live tail keeps what it already wrapped and rebuilds only what
/// changed. Whatever it reused, each frame must be the frame an app that saw
/// the same events and never drew before paints.
#[tokio::test]
async fn a_streamed_tail_paints_what_one_fresh_frame_paints() {
    let token = |text: &str| Step::Event(AgentEvent::Token { text: text.into() });
    let thinking = |text: &str| Step::Event(AgentEvent::Thinking { text: text.into() });
    let steps = [
        thinking("weighing the first option\n"),
        Step::ToggleThinking,
        thinking("and then the second"),
        token("Here is"),
        token(" the answer, long enough to wrap at this width.\n\nA second paragraph"),
        token(" that goes on.\n- one\n- two"),
        Step::Width(30),
        token("\n- three\n"),
        Step::ToggleThinking,
        Step::Event(AgentEvent::MessageDone { text: "Here is the answer.".into() }),
        token("A next message"),
        Step::Width(44),
        thinking("more thought"),
        Step::ToggleThinking,
        token(" after a tool."),
    ];
    let start = |app: &mut App| {
        app.running = true;
        app.turn_started = None;
    };
    let (mut live, dir) = app_fixture();
    start(&mut live);
    let mut width = 44;
    for (i, step) in steps.iter().enumerate() {
        apply(&mut live, step).await;
        if let Step::Width(w) = step {
            width = *w;
        }
        let (painted, _) = frame(&mut live, width, 16);

        let (mut fresh, fresh_dir) = app_fixture();
        start(&mut fresh);
        for step in &steps[..=i] {
            apply(&mut fresh, step).await;
        }
        let (expected, _) = frame(&mut fresh, width, 16);
        // Row 0 is the header, which names each fixture's own directory.
        assert_eq!(painted[1..], expected[1..], "step {i}");
        let _ = std::fs::remove_dir_all(fresh_dir);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// The chat rows of a 30x14 frame: between the header row and the status
/// row above the composer.
fn chat_rows(app: &mut App) -> Vec<String> {
    let (rows, _) = frame(app, 30, 14);
    let border = rows.iter().position(|row| row.starts_with('╭')).unwrap();
    rows[1..border - 1].to_vec()
}

#[test]
fn short_history_sits_on_the_composer_and_long_history_scrolls_with_one_marker() {
    let (mut app, dir) = app_fixture();
    app.insert_user_block("question");
    app.transcript.push_assistant((1..=3).map(|i| Line::from(format!("answer line {i}"))).collect());
    // Six rows (the prompt, the reply, a blank row after each) sit at the
    // bottom of the view, blank rows above them.
    let rows = chat_rows(&mut app);
    let visible = rows.len();
    assert!(rows[..visible - 6].iter().all(|row| row.trim().is_empty()), "{rows:#?}");
    assert!(rows[visible - 6].starts_with("❯ question"), "{rows:#?}");
    assert!(rows[visible - 2].starts_with("answer line 3"), "{rows:#?}");
    assert!(rows.iter().all(|row| !row.ends_with('▐')), "nothing to scroll: {rows:#?}");

    // Twice the view: the marker sits on the bottom row while following and
    // on the top row at the top of history.
    app.transcript.push_assistant((4..=2 * visible).map(|i| Line::from(format!("answer line {i}"))).collect());
    let rows = chat_rows(&mut app);
    let marked: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].ends_with('▐')).collect();
    assert_eq!(marked, [visible - 1], "{rows:#?}");
    app.transcript.scroll_up(10 * visible);
    let rows = chat_rows(&mut app);
    let marked: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].ends_with('▐')).collect();
    assert_eq!(marked, [0], "{rows:#?}");

    // Scrolled into the reply, its prompt stays pinned on the first row.
    app.transcript.follow();
    app.transcript.scroll_up(4);
    let rows = chat_rows(&mut app);
    assert!(rows[0].starts_with("❯ question"), "{rows:#?}");
    assert!(rows[1].contains("answer line"), "{rows:#?}");
    assert!(rows.iter().filter(|row| row.contains("question")).count() == 1, "{rows:#?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn history_that_exactly_fills_the_view_shows_no_marker() {
    let (mut app, dir) = app_fixture();
    let visible = chat_rows(&mut app).len();
    // A reply takes its lines and a blank row after them.
    app.transcript.push_assistant((1..visible).map(|i| Line::from(format!("line {i}"))).collect());
    let rows = chat_rows(&mut app);
    assert!(rows[0].starts_with("line 1 ") && rows[visible - 2].starts_with(&format!("line {} ", visible - 1)), "{rows:#?}");
    assert!(rows.iter().all(|row| !row.ends_with('▐')), "{rows:#?}");
    app.note("one row more");
    let rows = chat_rows(&mut app);
    assert!(rows[visible - 1].ends_with('▐'), "{rows:#?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn only_a_scrollback_focus_highlights_the_selected_block() {
    let (mut app, dir) = app_fixture();
    conversation(&mut app);
    app.transcript.select_block(3);
    // A selected block is painted on the border color's surface.
    let highlighted = |app: &mut App| {
        let mut terminal = Terminal::new(TestBackend::new(30, 14)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..14).any(|y| (0..30).any(|x| buffer[(x, y)].bg == crate::theme::BORDER()))
    };
    assert!(!highlighted(&mut app), "the composer has focus");
    app.focus = Focus::Scrollback;
    assert!(highlighted(&mut app), "the scrollback has focus");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_popup_gets_no_row_the_composer_needs() {
    let full = super::conversation_layout(Rect::new(0, 0, 40, 3), true, 3, 0, 2);
    assert_eq!((full.popup.height, full.status.height, full.chat.height), (0, 0, 0));
    let one_spare = super::conversation_layout(Rect::new(0, 0, 40, 4), true, 3, 0, 2);
    assert_eq!((one_spare.popup.height, one_spare.status.height), (1, 0));
}

#[tokio::test]
async fn a_resumed_session_replays_its_prompts_and_leaves_nothing_to_discard() {
    use open_max_core::sessions;
    use open_max_core::types::ChatMessage;
    let (mut app, dir) = app_fixture();
    let core = app.core.clone();
    let id = sessions::create(&core, dir.display().to_string()).unwrap().id;
    sessions::attach(&core, &id).unwrap();
    let history = vec![
        ChatMessage::system("rules"),
        ChatMessage::user("the first question"),
        ChatMessage::assistant(Some("the first answer".into()), None),
    ];
    assert!(sessions::save_messages(&core, &id, &history, &mut 0, true));
    sessions::end_turn(&core, &id, &history);
    sessions::detach(&core, &id).unwrap();
    app.left_running = vec![id.clone(), "another".into()];
    app.handle_submit("/resume".into()).await.unwrap();
    press(&mut app, key(KeyCode::Enter)).await;
    assert_eq!(app.session_id.as_deref(), Some(id.as_str()));
    let shown = app.transcript.export_text();
    assert!(shown.contains("the first question") && shown.contains("the first answer"), "{shown}");
    assert_eq!(app.left_running, ["another"], "resumed, it is no longer the app's to discard");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn resume_with_no_sessions_says_so_instead_of_opening_an_empty_list() {
    let (mut app, dir) = app_fixture();
    app.handle_submit("/resume".into()).await.unwrap();
    assert!(app.mode == Mode::Chat && app.sessions_panel.is_none());
    assert!(app.transcript.export_text().contains("no sessions in this project yet"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn built_in_commands_win_over_templates_of_the_same_name() {
    let (mut app, dir) = app_fixture();
    let prompts = dir.join(".agents").join("prompts");
    std::fs::create_dir_all(&prompts).unwrap();
    for (name, body) in [("greet", "Say hello to the team"), ("status", "Template status body"), ("exit", "Template exit body")] {
        std::fs::write(prompts.join(format!("{name}.md")), body).unwrap();
    }
    app.handle_submit("/status".into()).await.unwrap();
    let shown = app.transcript.export_text();
    assert!(shown.contains("throughput") && !shown.contains("Template status body"), "{shown}");
    app.handle_submit("/greet".into()).await.unwrap();
    assert!(app.transcript.export_text().contains("Say hello to the team"), "a template expands into a prompt");
    app.handle_submit("/exit".into()).await.unwrap();
    assert!(app.should_quit);
    assert!(!app.transcript.export_text().contains("Template exit body"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn status_reports_throughput_only_once_there_is_some() {
    let throughput = |app: &App| {
        let shown = app.transcript.export_text();
        shown.lines().rev().find(|line| line.contains("throughput")).unwrap().to_string()
    };
    let (mut app, dir) = app_fixture();
    app.handle_submit("/status".into()).await.unwrap();
    assert!(throughput(&app).ends_with("throughput not available"), "{}", throughput(&app));
    app.first_token = Some(Instant::now() - Duration::from_secs(2));
    app.stream_chars = 800;
    app.handle_submit("/status".into()).await.unwrap();
    assert!(throughput(&app).ends_with(" tokens/s"), "{}", throughput(&app));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_header_is_laid_out_again_for_a_new_width() {
    let (mut app, dir) = app_fixture();
    app.project = std::path::PathBuf::from("/a/long/parent/directory/proj");
    let (rows, _) = frame(&mut app, 60, 12);
    assert!(rows[0].starts_with("/a/long/parent/directory/proj"), "{:?}", rows[0]);
    app.dirty = Dirty::default();
    let (rows, _) = frame(&mut app, 14, 12);
    assert!(rows[0].trim_end().ends_with("proj"), "{:?}", rows[0]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn queued_messages_show_three_rows_at_most_and_count_the_rest() {
    let queue_rows = |n: usize| {
        let (mut app, dir) = app_fixture();
        app.queued = (1..=n).map(|i| format!("queued {i}")).collect();
        let (rows, _) = frame(&mut app, 40, 16);
        let _ = std::fs::remove_dir_all(dir);
        rows.into_iter().filter(|row| row.starts_with("↳ ")).map(|row| row.trim_end().to_string()).collect::<Vec<_>>()
    };
    assert_eq!(queue_rows(2), ["↳ queued 1", "↳ queued 2"]);
    assert_eq!(queue_rows(3), ["↳ queued 1", "↳ queued 2", "↳ queued 3"]);
    assert_eq!(queue_rows(5), ["↳ queued 1", "↳ queued 2", "↳ … 3 more queued"]);
}

#[test]
fn a_scrolled_view_pins_its_prompt_and_places_the_marker_by_position() {
    let (mut app, dir) = app_fixture();
    app.insert_user_block("question");
    // Rows: the prompt, a blank, eighteen reply lines, a blank: 21 in a
    // view of nine.
    app.transcript.push_assistant((1..=18).map(|i| Line::from(format!("answer line {i}"))).collect());
    let rows = chat_rows(&mut app);
    assert_eq!(rows.len(), 9);
    assert!(!rows[0].starts_with("❯ question"), "following, nothing is pinned: {rows:#?}");

    // Four rows up: the view ends on reply line 14 and the prompt takes the
    // first row from line 6.
    app.transcript.scroll_up(4);
    let (all, _) = frame(&mut app, 30, 14);
    let rows = chat_rows(&mut app);
    assert!(rows[0].starts_with("❯ question"), "{rows:#?}");
    assert!(rows[1].starts_with("answer line 7 "), "{rows:#?}");
    assert!(rows[8].starts_with("answer line 14 "), "{rows:#?}");
    let border = all.iter().position(|row| row.starts_with('╭')).unwrap();
    assert!(all[border - 1].contains("esc follow"), "the status row is intact: {all:#?}");
    // Six of twelve rows from the top: the marker is mid-track.
    app.transcript.scroll_up(2);
    let rows = chat_rows(&mut app);
    let marked: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].ends_with('▐')).collect();
    assert_eq!(marked, [4], "{rows:#?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn pointer_positions_map_to_the_draft_and_drags_extend_the_selection() {
    let (mut app, dir) = app_fixture();
    app.composer.load("alpha beta");
    let (rows, _) = frame(&mut app, 40, 12);
    let area = app.composer_draw_area;
    let draft = &rows[area.y as usize];
    let alpha = draft[..draft.find("alpha").unwrap()].chars().count() as u16;
    // The last letter of a word picks that word, not the next one.
    click(&mut app, alpha + 4, area.y).await;
    click(&mut app, alpha + 4, area.y).await;
    assert_eq!(app.composer.selected_text().as_deref(), Some("alpha"));

    // A press then a drag past the end of the draft takes everything after
    // the press, however far the pointer goes.
    let (_, _) = frame(&mut app, 40, 12);
    let drag = |kind, column| TermEvent::Mouse(MouseEvent { kind, column, row: area.y, modifiers: KeyModifiers::NONE });
    app.last_click = None;
    app.on_term_event(drag(MouseEventKind::Down(MouseButton::Left), alpha + 6)).await.unwrap();
    app.on_term_event(drag(MouseEventKind::Drag(MouseButton::Left), 39)).await.unwrap();
    app.on_term_event(drag(MouseEventKind::Up(MouseButton::Left), 39)).await.unwrap();
    assert_eq!(app.composer.selected_text().as_deref(), Some("beta"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn a_transcript_selection_paints_on_the_row_it_selected() {
    let (mut app, dir) = app_fixture();
    for i in 0..6 {
        app.note(&format!("line {i}"));
    }
    let (rows, _) = frame(&mut app, 30, 14);
    let row = rows.iter().position(|row| row.contains("line 3")).unwrap();
    for _ in 0..3 {
        click(&mut app, 4, row as u16).await;
    }
    let mut terminal = Terminal::new(TestBackend::new(30, 14)).unwrap();
    terminal.draw(|frame| app.draw(frame)).unwrap();
    let buffer = terminal.backend().buffer();
    let selected: Vec<u16> =
        (0..14).filter(|&y| (0..30).any(|x| buffer[(x, y)].bg == crate::theme::SELECT())).collect();
    assert_eq!(selected, [row as u16]);
    let _ = std::fs::remove_dir_all(dir);
}
