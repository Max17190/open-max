//! The composer: a small multiline input with history. Enter submits;
//! Shift+Enter (kitty protocol terminals) or Alt+Enter inserts a newline.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

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
                    let byte = char_to_byte(&self.lines[self.row], self.col - 1);
                    self.lines[self.row].remove(byte);
                    self.col -= 1;
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
                    let byte = char_to_byte(&self.lines[self.row], self.col);
                    self.lines[self.row].remove(byte);
                } else if self.row + 1 < self.lines.len() {
                    let next = self.lines.remove(self.row + 1);
                    self.lines[self.row].push_str(&next);
                }
            }
            KeyCode::Left => {
                if self.col > 0 {
                    self.col -= 1;
                } else if self.row > 0 {
                    self.row -= 1;
                    self.col = self.lines[self.row].chars().count();
                }
            }
            KeyCode::Right => {
                if self.col < self.lines[self.row].chars().count() {
                    self.col += 1;
                } else if self.row + 1 < self.lines.len() {
                    self.row += 1;
                    self.col = 0;
                }
            }
            KeyCode::Up => {
                if self.row > 0 {
                    self.row -= 1;
                    self.col = self.col.min(self.lines[self.row].chars().count());
                } else {
                    self.history_prev();
                }
            }
            KeyCode::Down => {
                if self.row + 1 < self.lines.len() {
                    self.row += 1;
                    self.col = self.col.min(self.lines[self.row].chars().count());
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
        let chars: Vec<char> = line.chars().collect();
        let mut i = self.col;
        while i > 0 && chars[i - 1] == ' ' {
            i -= 1;
        }
        while i > 0 && chars[i - 1] != ' ' {
            i -= 1;
        }
        let start = char_to_byte(line, i);
        let end = char_to_byte(line, self.col);
        line.replace_range(start..end, "");
        self.col = i;
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
                Span::styled("❯ ", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD))
            } else {
                Span::styled("… ", Style::default().fg(theme::DIM))
            };
            out.push(Line::from(vec![prefix, Span::raw(line.clone())]));
        }
        let cursor_y = (self.row - first) as u16;
        let cursor_x = 2 + self.lines[self.row]
            .chars()
            .take(self.col)
            .map(|c| unicode_width::UnicodeWidthChar::width(c).unwrap_or(0))
            .sum::<usize>() as u16;
        (out, cursor_x, cursor_y)
    }
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map(|(b, _)| b).unwrap_or(s.len())
}
