//! The composer: a small multiline input with history. Enter submits;
//! Shift+Enter (kitty protocol terminals) or Alt+Enter inserts a newline.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;

use crate::theme;

const MAX_HISTORY: usize = 200;

pub enum ComposerAction {
    None,
    Submit(String),
}

pub struct Composer {
    lines: Vec<String>,
    /// Cursor as (row, char column).
    row: usize,
    col: usize,
    history: Vec<String>,
    hist_idx: Option<usize>,
    stash: String,
    history_path: PathBuf,
}

impl Composer {
    pub fn new(data_dir: &std::path::Path) -> Self {
        let history_path = data_dir.join("history.json");
        let history: Vec<String> = std::fs::read_to_string(&history_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self {
            lines: vec![String::new()],
            row: 0,
            col: 0,
            history,
            hist_idx: None,
            stash: String::new(),
            history_path,
        }
    }

    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub fn is_empty(&self) -> bool {
        self.lines.len() == 1 && self.lines[0].is_empty()
    }

    /// Snapshot of prompt history (oldest first) for Ctrl+R search.
    pub fn history_entries(&self) -> Vec<String> {
        self.history.clone()
    }

    /// Cursor as (row, char column) plus the current row's text, for the
    /// completion popup to inspect the token being typed.
    pub fn cursor_context(&self) -> (usize, usize, &str) {
        (self.row, self.col, &self.lines[self.row])
    }

    /// Replace `len` chars starting at char index `start` in the current row
    /// and leave the cursor after the replacement. Used to accept completions.
    pub fn replace_token(&mut self, start: usize, len: usize, replacement: &str) {
        let line = &mut self.lines[self.row];
        let from = char_to_byte(line, start);
        let to = char_to_byte(line, start + len);
        line.replace_range(from..to, replacement);
        self.col = start + replacement.chars().count();
    }

    /// Load text into the composer (e.g. queued messages handed back after a
    /// cancel), leaving the cursor at the end.
    pub fn load(&mut self, text: &str) {
        self.set_text(text);
        self.hist_idx = None;
    }

    /// Take the trimmed text and clear, remembering it in history. Used when
    /// a completion accept submits a command directly.
    pub fn take(&mut self) -> String {
        let text = self.text().trim().to_string();
        if !text.is_empty() {
            self.remember(&text);
        }
        self.clear();
        text
    }

    pub fn height(&self) -> u16 {
        self.lines.len().min(6) as u16
    }

    pub fn insert_str(&mut self, s: &str) {
        for c in s.chars() {
            if c == '\n' {
                self.newline();
            } else if c != '\r' {
                self.insert_char(c);
            }
        }
    }

    fn insert_char(&mut self, c: char) {
        let line = &mut self.lines[self.row];
        let byte = char_to_byte(line, self.col);
        line.insert(byte, c);
        self.col += 1;
    }

    fn newline(&mut self) {
        let line = &mut self.lines[self.row];
        let byte = char_to_byte(line, self.col);
        let rest = line.split_off(byte);
        self.lines.insert(self.row + 1, rest);
        self.row += 1;
        self.col = 0;
    }

    fn clear(&mut self) {
        self.lines = vec![String::new()];
        self.row = 0;
        self.col = 0;
        self.hist_idx = None;
    }

    fn set_text(&mut self, text: &str) {
        self.lines = text.split('\n').map(str::to_string).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.row = self.lines.len() - 1;
        self.col = self.lines[self.row].chars().count();
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> ComposerAction {
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        match key.code {
            KeyCode::Enter if shift || alt => {
                self.newline();
            }
            KeyCode::Enter => {
                let text = self.text().trim().to_string();
                if text.is_empty() {
                    return ComposerAction::None;
                }
                self.remember(&text);
                self.clear();
                return ComposerAction::Submit(text);
            }
            KeyCode::Char('u') if ctrl => {
                self.lines[self.row].clear();
                self.col = 0;
                // Collapse the emptied row from any cursor position so
                // every press makes visible progress: without this a
                // pasted multi-line draft was a dead end (ctrl+u on an
                // empty line did nothing, including on the first row).
                if self.lines.len() > 1 {
                    self.lines.remove(self.row);
                    if self.row >= self.lines.len() {
                        self.row = self.lines.len() - 1;
                    }
                    self.col = self.lines[self.row].chars().count();
                }
            }
            KeyCode::Char('w') if ctrl => self.delete_word(),
            KeyCode::Char('a') if ctrl => self.col = 0,
            KeyCode::Char('e') if ctrl => self.col = self.lines[self.row].chars().count(),
            KeyCode::Char('k') if ctrl => {
                let byte = char_to_byte(&self.lines[self.row], self.col);
                self.lines[self.row].truncate(byte);
            }
            KeyCode::Char(c) if !ctrl => {
                self.hist_idx = None;
                self.insert_char(c);
            }
            KeyCode::Backspace => {
                if self.col > 0 {
                    let start = previous_grapheme_boundary(&self.lines[self.row], self.col);
                    let from = char_to_byte(&self.lines[self.row], start);
                    let to = char_to_byte(&self.lines[self.row], self.col);
                    self.lines[self.row].replace_range(from..to, "");
                    self.col = start;
                } else if self.row > 0 {
                    let removed = self.lines.remove(self.row);
                    self.row -= 1;
                    self.col = self.lines[self.row].chars().count();
                    self.lines[self.row].push_str(&removed);
                }
            }
            KeyCode::Delete => {
                let len = self.lines[self.row].chars().count();
                if self.col < len {
                    let end = next_grapheme_boundary(&self.lines[self.row], self.col);
                    let from = char_to_byte(&self.lines[self.row], self.col);
                    let to = char_to_byte(&self.lines[self.row], end);
                    self.lines[self.row].replace_range(from..to, "");
                } else if self.row + 1 < self.lines.len() {
                    let next = self.lines.remove(self.row + 1);
                    self.lines[self.row].push_str(&next);
                }
            }
            KeyCode::Left => {
                if self.col > 0 {
                    self.col = previous_grapheme_boundary(&self.lines[self.row], self.col);
                } else if self.row > 0 {
                    self.row -= 1;
                    self.col = self.lines[self.row].chars().count();
                }
            }
            KeyCode::Right => {
                if self.col < self.lines[self.row].chars().count() {
                    self.col = next_grapheme_boundary(&self.lines[self.row], self.col);
                } else if self.row + 1 < self.lines.len() {
                    self.row += 1;
                    self.col = 0;
                }
            }
            KeyCode::Up => {
                if self.row > 0 {
                    self.row -= 1;
                    self.col = floor_grapheme_boundary(&self.lines[self.row], self.col);
                } else {
                    self.history_prev();
                }
            }
            KeyCode::Down => {
                if self.row + 1 < self.lines.len() {
                    self.row += 1;
                    self.col = floor_grapheme_boundary(&self.lines[self.row], self.col);
                } else {
                    self.history_next();
                }
            }
            KeyCode::Home => self.col = 0,
            KeyCode::End => self.col = self.lines[self.row].chars().count(),
            _ => {}
        }
        ComposerAction::None
    }

    fn delete_word(&mut self) {
        let line = &mut self.lines[self.row];
        let graphemes: Vec<&str> = line.graphemes(true).collect();
        let boundaries = grapheme_boundaries(line);
        let mut i = boundaries
            .iter()
            .position(|boundary| *boundary == self.col)
            .unwrap_or_else(|| {
                boundaries
                    .iter()
                    .rposition(|boundary| *boundary < self.col)
                    .unwrap_or(0)
            });
        while i > 0 && graphemes[i - 1].chars().all(char::is_whitespace) {
            i -= 1;
        }
        while i > 0 && !graphemes[i - 1].chars().all(char::is_whitespace) {
            i -= 1;
        }
        let start_col = boundaries[i];
        let start = char_to_byte(line, start_col);
        let end = char_to_byte(line, self.col);
        line.replace_range(start..end, "");
        self.col = start_col;
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.hist_idx {
            None => {
                self.stash = self.text();
                self.hist_idx = Some(self.history.len() - 1);
            }
            Some(0) => return,
            Some(i) => self.hist_idx = Some(i - 1),
        }
        if let Some(i) = self.hist_idx {
            let entry = self.history[i].clone();
            self.set_text(&entry);
        }
    }

    fn history_next(&mut self) {
        let Some(i) = self.hist_idx else { return };
        if i + 1 < self.history.len() {
            self.hist_idx = Some(i + 1);
            let entry = self.history[i + 1].clone();
            self.set_text(&entry);
        } else {
            self.hist_idx = None;
            let stash = self.stash.clone();
            self.set_text(&stash);
        }
    }

    fn remember(&mut self, text: &str) {
        if self.history.last().map(|l| l == text).unwrap_or(false) {
            return;
        }
        self.history.push(text.to_string());
        if self.history.len() > MAX_HISTORY {
            let overflow = self.history.len() - MAX_HISTORY;
            self.history.drain(..overflow);
        }
        if let Ok(json) = serde_json::to_string(&self.history) {
            let _ = std::fs::write(&self.history_path, json);
        }
    }

    /// Lines to draw plus the cursor position (x, y) within them. `max_h` is
    /// the height actually granted by the caller, which can be smaller than
    /// `height()` on tiny terminals.
    pub fn render(&self, max_h: u16) -> (Vec<Line<'static>>, u16, u16) {
        let mut out = Vec::new();
        // Show the last rows that fit the height budget, sliding up when the
        // cursor moves into rows that would otherwise be scrolled out.
        let visible = (self.height().min(max_h.max(1))) as usize;
        let first = self.lines.len().saturating_sub(visible).min(self.row);
        for (i, line) in self.lines.iter().enumerate().skip(first).take(visible) {
            let prefix = if i == 0 {
                Span::styled(
                    "❯ ",
                    Style::default()
                        .fg(theme::ACCENT())
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled("… ", Style::default().fg(theme::DIM()))
            };
            if i == 0 && self.is_empty() {
                out.push(Line::from(vec![
                    prefix,
                    Span::styled(
                        "Describe a task",
                        Style::default()
                            .fg(theme::DIM())
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
            } else {
                out.push(Line::from(vec![prefix, Span::raw(line.clone())]));
            }
        }
        let cursor_y = (self.row - first) as u16;
        let cursor_byte = char_to_byte(&self.lines[self.row], self.col);
        let cursor_x =
            2 + crate::ui::text::width(&self.lines[self.row][..cursor_byte]) as u16;
        (out, cursor_x, cursor_y)
    }
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

fn grapheme_boundaries(s: &str) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(s.graphemes(true).count() + 1);
    boundaries.push(0);
    let mut chars = 0usize;
    for grapheme in s.graphemes(true) {
        chars += grapheme.chars().count();
        boundaries.push(chars);
    }
    boundaries
}

fn previous_grapheme_boundary(s: &str, col: usize) -> usize {
    grapheme_boundaries(s)
        .into_iter()
        .rfind(|boundary| *boundary < col)
        .unwrap_or(0)
}

fn next_grapheme_boundary(s: &str, col: usize) -> usize {
    grapheme_boundaries(s)
        .into_iter()
        .find(|boundary| *boundary > col)
        .unwrap_or_else(|| s.chars().count())
}

fn floor_grapheme_boundary(s: &str, col: usize) -> usize {
    grapheme_boundaries(s)
        .into_iter()
        .rfind(|boundary| *boundary <= col)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(line: &Line<'static>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn empty_composer_uses_single_purposeful_placeholder() {
        let composer = Composer::new(&std::env::temp_dir());
        let (lines, cursor_x, cursor_y) = composer.render(3);
        assert_eq!(lines.len(), 1);
        assert_eq!(plain(&lines[0]), "❯ Describe a task");
        assert_eq!((cursor_x, cursor_y), (2, 0));
    }

    #[test]
    fn multiline_composer_keeps_distinct_continuation_gutter() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("first\nsecond");
        let (lines, cursor_x, cursor_y) = composer.render(3);
        assert_eq!(plain(&lines[0]), "❯ first");
        assert_eq!(plain(&lines[1]), "… second");
        assert_eq!((cursor_x, cursor_y), (8, 1));
    }

    #[test]
    fn composer_edits_complete_graphemes() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("e\u{301}👩‍💻");

        composer.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(composer.col, 2);
        composer.handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert_eq!(composer.text(), "e\u{301}");

        composer.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(composer.text(), "");
        assert_eq!(composer.col, 0);
    }

    #[test]
    fn composer_cursor_uses_terminal_cell_width() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("漢e\u{301}👩‍💻");
        let (_, cursor_x, cursor_y) = composer.render(3);
        assert_eq!((cursor_x, cursor_y), (7, 0));
    }

    #[test]
    fn ctrl_u_clears_a_multi_line_draft_from_any_row() {
        let ctrl_u = KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL);

        // From the last row: presses walk the draft upward to empty.
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("first\nsecond\nthird");
        composer.handle_key(ctrl_u);
        assert_eq!(composer.text(), "first\nsecond");
        composer.handle_key(ctrl_u);
        assert_eq!(composer.text(), "first");
        composer.handle_key(ctrl_u);
        assert_eq!(composer.text(), "");
        // One more press on the empty draft stays a no-op, never a panic.
        composer.handle_key(ctrl_u);
        assert_eq!(composer.text(), "");

        // From the top row: presses consume downward, no stuck state.
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("first\nsecond\nthird");
        composer.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        composer.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(composer.row, 0);
        composer.handle_key(ctrl_u);
        assert_eq!(composer.text(), "second\nthird");
        composer.handle_key(ctrl_u);
        assert_eq!(composer.text(), "third");
        composer.handle_key(ctrl_u);
        assert_eq!(composer.text(), "");
    }
}
