//! The composer: a small multiline input with history. Enter submits;
//! Shift+Enter (kitty protocol terminals) or Alt+Enter inserts a newline.
//!
//! Logical lines are the edit model; what you see is a soft-wrapped view of
//! them. A draft longer than the box scrolls inside it, the wheel and the
//! mouse reach every row, and dragging selects text for [`Composer::selected_text`].

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;

use crate::theme;
use crate::ui::text;

const MAX_HISTORY: usize = 200;

/// Cells the gutter takes from every rendered row: `❯ ` on the first row of
/// the draft, `… ` on a top row that is scrolled off its start, blank on
/// every other row.
const GUTTER: usize = 2;

/// Rows the composer may occupy before a draft starts scrolling inside it.
const MAX_VISIBLE_ROWS: usize = 6;

/// A paste collapses to a marker at either bound: enough lines to bury the
/// draft (the composer shows six rows), or enough characters that one
/// logical line still drowns it (a minified blob, a stack trace one-liner).
const PASTE_COLLAPSE_LINES: usize = 10;
const PASTE_COLLAPSE_CHARS: usize = 1000;

/// The draft stand-in for collapsed paste `index` (1-based). Content-derived
/// and index-stamped, so two identical pastes still get distinct markers and
/// a marker matches only while it is intact.
fn paste_marker(index: usize, text: &str) -> String {
    let lines = text.lines().count();
    if lines > 1 {
        format!("[pasted #{index}: {lines} lines]")
    } else {
        format!("[pasted #{index}: {} chars]", text.chars().count())
    }
}

/// A cursor or selection endpoint as (logical line, char column).
type Point = (usize, usize);

pub enum ComposerAction {
    None,
    Submit(String),
}

/// A mouse selection over the draft, endpoints inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Selection {
    anchor: Point,
    head: Point,
    /// Set by a word or line gesture, which picks a real range even when it is
    /// one character wide. With inclusive endpoints a drag that has not moved
    /// looks identical to a one-character selection; this tells them apart and
    /// keeps a plain click from selecting.
    explicit: bool,
    /// The inclusive range the press picked: the word or line under a
    /// gesture, the character under a plain press. A drag extends from it
    /// but never shrinks the selection below it, matching the transcript.
    origin: (Point, Point),
}

impl Selection {
    fn is_empty(&self) -> bool {
        !self.explicit && self.anchor == self.head
    }
}

/// One display row: a soft-wrapped slice `start..end` (char columns) of
/// logical line `line`. `last` marks the row that ends the logical line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    line: usize,
    start: usize,
    end: usize,
    last: bool,
}

pub struct Composer {
    lines: Vec<String>,
    /// Cursor as (row, char column).
    row: usize,
    col: usize,
    /// First visible display row.
    scroll: usize,
    /// Whether the viewport tracks the cursor. The wheel detaches it; the
    /// next edit or cursor move re-attaches, so typing is never lost offscreen.
    follow: bool,
    /// Mouse selection in logical coordinates. Both endpoints are inclusive,
    /// matching the transcript: what is highlighted is exactly what a copy
    /// carries, down to the character under the release cell.
    selection: Option<Selection>,
    /// Whether the mouse button is still down on that selection.
    dragging: bool,
    /// Ticks on every entry point that can change the text, via `touch`, so a
    /// wrap cached against a generation cannot outlive the text it describes.
    /// A `cfg(test)` check in `ensure_rows` proves no mutation path skips it.
    generation: u64,
    /// Wrapped rows for one `(text width, generation)` pair.
    wrap_cache: Option<(usize, u64, Vec<Row>)>,
    /// Cache misses, i.e. how many times the draft was actually re-wrapped.
    /// The oracle for "moving the cursor must not re-wrap".
    #[cfg(test)]
    wraps: usize,
    /// Single-char insertions. The oracle for "a paste splices whole
    /// segments": per-char insertion re-scans the line prefix each time,
    /// which is quadratic over a single-line paste.
    #[cfg(test)]
    char_inserts: usize,
    history: Vec<String>,
    hist_idx: Option<usize>,
    stash: String,
    history_path: PathBuf,
    /// Collapsed pastes, indexed by the `#N` in their draft markers. The
    /// draft holds `[pasted #N: … lines]` while the bytes wait here; every
    /// reader of the whole draft (`text`, and so submit, queue, history) gets
    /// the expansion, so a marker never leaves the composer. An entry whose
    /// marker was hand-edited or deleted simply never matches again; the
    /// vector drains with `clear`, so nothing outlives its draft. `None` is
    /// a spent reference: an in-place expansion consumed the bytes, and the
    /// occupied index keeps sibling markers' numbers stable.
    pastes: Vec<Option<String>>,
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
            scroll: 0,
            follow: true,
            selection: None,
            dragging: false,
            generation: 0,
            wrap_cache: None,
            #[cfg(test)]
            wraps: 0,
            #[cfg(test)]
            char_inserts: 0,
            history,
            hist_idx: None,
            stash: String::new(),
            history_path,
            pastes: Vec::new(),
        }
    }

    pub fn text(&self) -> String {
        self.expand_pastes(&self.lines.join("\n"))
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
        self.on_input();
        self.touch();
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

    /// Rows the draft needs at `width` cells, capped so the composer never
    /// eats the conversation. `width` is the full render width, gutter included.
    pub fn height(&mut self, width: u16) -> u16 {
        self.ensure_rows(text_width(width));
        self.cached_rows().len().clamp(1, MAX_VISIBLE_ROWS) as u16
    }

    pub fn insert_str(&mut self, s: &str) {
        self.on_input();
        self.touch();
        // One splice per logical line, not one per char: inserting char by
        // char re-scans the line prefix for every char, which is quadratic
        // over a single-line paste (a 100 KB minified blob froze the UI for
        // seconds). `\r` is dropped, matching the char path it replaces.
        for (i, seg) in s.split('\n').enumerate() {
            if i > 0 {
                self.newline();
            }
            let seg = if seg.contains('\r') {
                std::borrow::Cow::Owned(seg.replace('\r', ""))
            } else {
                std::borrow::Cow::Borrowed(seg)
            };
            if seg.is_empty() {
                continue;
            }
            let line = &mut self.lines[self.row];
            let byte = char_to_byte(line, self.col);
            line.insert_str(byte, &seg);
            self.col += seg.chars().count();
        }
    }

    /// Insert a paste, collapsing a big one to a numbered marker so the
    /// draft stays editable instead of drowning under it. The bytes live in
    /// `pastes` and rejoin the draft when the whole text is read out (see
    /// the field doc); `expand_paste_at_cursor` splices them back in place
    /// for editing. Collapse and expand ship together on purpose: a
    /// collapsed paste with no way back is worse than no collapse.
    pub fn insert_paste(&mut self, text: &str) {
        // Match what `insert_str` would have produced so expansion is
        // byte-identical to the never-collapsed draft.
        let text = text.replace('\r', "");
        if text.lines().count() < PASTE_COLLAPSE_LINES
            && text.chars().count() < PASTE_COLLAPSE_CHARS
        {
            self.insert_str(&text);
            return;
        }
        let marker = paste_marker(self.pastes.len() + 1, &text);
        self.pastes.push(Some(text));
        self.insert_str(&marker);
    }

    fn expand_pastes(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (i, paste) in self.pastes.iter().enumerate() {
            let Some(paste) = paste else {
                // Spent by an in-place expansion: any surviving duplicate of
                // its marker is literal text, never a second delivery.
                continue;
            };
            let marker = paste_marker(i + 1, paste);
            if out.contains(&marker) {
                // Each insertion wrote its marker exactly once, so exactly
                // one occurrence is the paste: a hand-typed duplicate of the
                // marker's spelling stays literal text instead of injecting
                // the hidden bytes a second time. A duplicate placed BEFORE
                // the genuine marker therefore receives the expansion in its
                // stead, deterministically; the cursor-anchored ctrl+o
                // splice is how a human resolves which occurrence they mean.
                // Span tracking through arbitrary edits would pin WHICH
                // occurrence and is not worth its machinery for a forged
                // marker.
                out = out.replacen(&marker, paste, 1);
            }
        }
        out
    }

    /// Splice the collapsed paste whose marker sits under the cursor back
    /// into the draft, for editing it in place. Returns false when the
    /// cursor's row holds no intact marker under the cursor.
    pub fn expand_paste_at_cursor(&mut self) -> bool {
        let row_text = self.lines[self.row].clone();
        for i in 0..self.pastes.len() {
            let marker = match self.pastes[i].as_ref() {
                Some(paste) => paste_marker(i + 1, paste),
                None => continue,
            };
            // The occurrence UNDER the cursor, never the first in the row: a
            // byte-identical duplicate earlier in the row must neither
            // swallow the gesture nor receive the splice.
            let Some(pos) = row_text.match_indices(&marker).map(|(p, _)| p).find(|&p| {
                let start = row_text[..p].chars().count();
                let end = start + marker.chars().count();
                self.col >= start && self.col <= end
            }) else {
                continue;
            };
            // take() both yields the bytes and leaves the tombstone: the
            // reference is SPENT, so a surviving duplicate is literal text
            // on every later path instead of a second delivery, while the
            // occupied index keeps sibling markers' numbers stable.
            let Some(paste) = self.pastes[i].take() else {
                continue;
            };
            self.on_input();
            self.touch();
            let before = row_text[..pos].to_string();
            let after = row_text[pos + marker.len()..].to_string();
            let segs: Vec<&str> = paste.split('\n').collect();
            let mut rows: Vec<String> = Vec::with_capacity(segs.len());
            match segs.as_slice() {
                [only] => rows.push(format!("{before}{only}{after}")),
                [first, mid @ .., last] => {
                    rows.push(format!("{before}{first}"));
                    rows.extend(mid.iter().map(|s| s.to_string()));
                    rows.push(format!("{last}{after}"));
                }
                [] => rows.push(format!("{before}{after}")),
            }
            // Cursor lands at the end of the spliced content.
            let last_row = self.row + rows.len() - 1;
            let last_col = rows
                .last()
                .map(|r| r.chars().count() - after.chars().count())
                .unwrap_or(0);
            self.lines.splice(self.row..=self.row, rows);
            self.row = last_row;
            self.col = last_col;
            return true;
        }
        false
    }

    /// A new input gesture: drop the mouse selection and re-attach the
    /// viewport to the cursor.
    fn on_input(&mut self) {
        self.selection = None;
        self.dragging = false;
        self.follow = true;
    }

    /// The text may have changed, so the wrap cached against the current
    /// generation no longer describes it.
    fn touch(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Whether a key only moves the cursor. Erring toward invalidation is
    /// safe, so this lists what is known harmless rather than trying to
    /// enumerate every edit; `set_text` covers the history keys, which do
    /// replace the draft from row 0.
    fn is_cursor_only(key: &KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down => true,
            KeyCode::Home | KeyCode::End => true,
            KeyCode::Char('a') | KeyCode::Char('e') if ctrl => true,
            _ => false,
        }
    }

    fn insert_char(&mut self, c: char) {
        #[cfg(test)]
        {
            self.char_inserts += 1;
        }
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
        self.scroll = 0;
        self.hist_idx = None;
        self.pastes.clear();
        self.on_input();
        self.touch();
    }

    fn set_text(&mut self, text: &str) {
        self.lines = text.split('\n').map(str::to_string).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.row = self.lines.len() - 1;
        self.col = self.lines[self.row].chars().count();
        self.scroll = 0;
        self.on_input();
        self.touch();
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> ComposerAction {
        self.on_input();
        // Navigating a long pasted draft must not cost a re-wrap of it.
        if !Self::is_cursor_only(&key) {
            self.touch();
        }
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

    // ---------- display layout ----------

    /// Cache the wrap for `text_width`, reusing it while the text and width
    /// hold. Wrapping is O(draft), and a frame asks for it twice; a 100 KB
    /// paste costs about 3 ms to wrap, so recomputing it per paint would put
    /// the composer alone over the whole frame budget.
    fn ensure_rows(&mut self, text_width: usize) {
        let fresh = self
            .wrap_cache
            .as_ref()
            .is_some_and(|(w, rev, _)| *w == text_width && *rev == self.generation);
        if !fresh {
            let rows = self.wrap(text_width);
            self.wrap_cache = Some((text_width, self.generation, rows));
            #[cfg(test)]
            {
                self.wraps += 1;
            }
        }
        // Release test runs (the measurement path) skip this; every ordinary
        // debug test run pays for it.
        #[cfg(all(test, debug_assertions))]
        {
            // Prove the generation actually covers every mutation instead of
            // trusting that it does.
            assert_eq!(
                self.cached_rows(),
                self.wrap(text_width).as_slice(),
                "composer wrap cache went stale",
            );
        }
    }

    fn cached_rows(&self) -> &[Row] {
        self.wrap_cache
            .as_ref()
            .map(|(_, _, rows)| rows.as_slice())
            .unwrap_or_default()
    }

    /// Soft-wrap every logical line into display rows at `text_width` cells.
    fn wrap(&self, text_width: usize) -> Vec<Row> {
        let mut out = Vec::new();
        for (line, text) in self.lines.iter().enumerate() {
            let cols = wrap_cols(text, text_width);
            let last = cols.len() - 1;
            out.extend(cols.into_iter().enumerate().map(|(i, (start, end))| Row {
                line,
                start,
                end,
                last: i == last,
            }));
        }
        out
    }

    /// Display row holding the cursor. A wrap point belongs to the row that
    /// starts there, so typing at a boundary shows up where the text will land.
    fn cursor_row(&self, rows: &[Row]) -> usize {
        rows.iter()
            .position(|r| {
                r.line == self.row
                    && self.col >= r.start
                    && (self.col < r.end || (r.last && self.col <= r.end))
            })
            .unwrap_or_else(|| rows.len().saturating_sub(1))
    }

    /// Rows shown at once: the granted height, never more than the cap.
    fn visible(rows: usize, max_h: u16) -> usize {
        rows.min(max_h.max(1) as usize).clamp(1, MAX_VISIBLE_ROWS)
    }

    /// Lines to draw plus the cursor position (x, y) within them. `width` and
    /// `max_h` are the render area actually granted by the caller, which can
    /// be smaller than [`Self::height`] asked for on tiny terminals.
    pub fn render(&mut self, width: u16, max_h: u16) -> (Vec<Line<'static>>, u16, u16) {
        self.ensure_rows(text_width(width));
        let total = self.cached_rows().len();
        let visible = Self::visible(total, max_h);
        let cursor_row = self.cursor_row(self.cached_rows());

        if self.follow {
            if cursor_row < self.scroll {
                self.scroll = cursor_row;
            } else if cursor_row >= self.scroll + visible {
                self.scroll = cursor_row + 1 - visible;
            }
        }
        self.scroll = self.scroll.min(total.saturating_sub(visible));

        let selection = self.selection_bounds();
        let rows = self.cached_rows();
        let mut out = Vec::with_capacity(visible);
        for (seen, row) in rows.iter().skip(self.scroll).take(visible).enumerate() {
            // The caret marks the true start of the draft. Every other row of
            // the same draft pays the gutter in blank cells: a glyph per
            // continuation row is noise on every multi-line prompt, and the
            // transcript already renders a wrapped user message this way. The
            // one row that earns a glyph is a top row that is not the start of
            // the draft, where `…` is the only signal that the view is
            // scrolled and there is more text above.
            let prefix = if row.line == 0 && row.start == 0 {
                Span::styled(
                    "❯ ",
                    Style::default()
                        .fg(theme::ACCENT())
                        .add_modifier(Modifier::BOLD),
                )
            } else if seen == 0 {
                Span::styled("… ", Style::default().fg(theme::DIM()))
            } else {
                Span::raw("  ")
            };
            if self.is_empty() {
                out.push(Line::from(vec![
                    prefix,
                    Span::styled(
                        "Describe a task",
                        Style::default()
                            .fg(theme::DIM())
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
                continue;
            }
            let mut spans = vec![prefix];
            let text = &self.lines[row.line];
            let selected = selection.and_then(|(lo, hi)| row_selection(lo, hi, row));
            let (sel_start, sel_end) = selected.unwrap_or((row.start, row.start));
            for (from, to, style) in [
                (row.start, sel_start, Style::default()),
                (sel_start, sel_end, Style::default().bg(theme::SELECT())),
                (sel_end, row.end, Style::default()),
            ] {
                if from >= to {
                    continue;
                }
                let slice = &text[char_to_byte(text, from)..char_to_byte(text, to)];
                spans.push(Span::styled(slice.to_string(), style));
            }
            out.push(Line::from(spans));
        }

        // Clamp into the box: while the wheel holds the view away from the
        // cursor there is no on-screen cell for it, and a terminal cursor
        // outside the composer would land on unrelated chrome.
        let cursor_y = cursor_row.saturating_sub(self.scroll).min(visible - 1) as u16;
        let line = &self.lines[self.row];
        let anchor = rows.get(cursor_row).map(|r| r.start).unwrap_or(0);
        let to = char_to_byte(line, self.col);
        let from = char_to_byte(line, anchor).min(to);
        let cursor_x = (GUTTER + text::width(&line[from..to])) as u16;
        (out, cursor_x.min(width.saturating_sub(1)), cursor_y)
    }

    // ---------- mouse ----------

    /// Move the cursor to a click at `(cell, row)` relative to the render
    /// area, and start a selection there.
    pub fn click_at(&mut self, width: u16, max_h: u16, cell: u16, row: u16) {
        let point = self.point_at(width, max_h, cell, row);
        self.row = point.0;
        self.col = point.1;
        self.follow = true;
        self.hist_idx = None;
        self.selection = Some(Selection {
            anchor: point,
            head: point,
            explicit: false,
            origin: (point, point),
        });
        self.dragging = true;
    }

    /// Extend an in-progress click selection to `(cell, row)`.
    pub fn drag_to(&mut self, width: u16, max_h: u16, cell: u16, row: u16) {
        let Some(current) = self.selection.filter(|_| self.dragging) else {
            return;
        };
        let point = self.point_at(width, max_h, cell, row);
        // The selection grows from the origin range and never shrinks below
        // it: past either edge the far edge anchors and the pointer leads;
        // inside it the selection is exactly the range the press picked.
        let (lo, hi) = current.origin;
        let (anchor, head) = if point > hi {
            (lo, point)
        } else if point < lo {
            (hi, point)
        } else {
            (lo, hi)
        };
        self.selection = Some(Selection { anchor, head, ..current });
        self.row = head.0;
        self.col = head.1;
        self.follow = true;
    }

    /// Select the word under a double-click in the draft.
    pub fn select_word_at(&mut self, width: u16, max_h: u16, cell: u16, row: u16) -> bool {
        self.select_range_at(width, max_h, cell, row, text::word_bounds)
    }

    /// Select the logical line under a triple-click, even where it wraps.
    pub fn select_line_at(&mut self, width: u16, max_h: u16, cell: u16, row: u16) -> bool {
        self.select_range_at(width, max_h, cell, row, text::line_bounds)
    }

    fn select_range_at(
        &mut self,
        width: u16,
        max_h: u16,
        cell: u16,
        row: u16,
        bounds: fn(&str, usize) -> (usize, usize),
    ) -> bool {
        let point = self.point_at(width, max_h, cell, row);
        // A press replaces whatever was selected, including with nothing. An
        // empty line has no word and no line to take, and leaving the old
        // highlight standing would make it lie about what a copy carries, so
        // this falls back to exactly what a plain press does.
        self.row = point.0;
        self.col = point.1;
        self.follow = true;
        self.hist_idx = None;
        self.selection = Some(Selection {
            anchor: point,
            head: point,
            explicit: false,
            origin: (point, point),
        });
        self.dragging = true;

        let (start, end) = bounds(&self.lines[point.0], point.1);
        if start >= end {
            return false;
        }
        self.col = end;
        let (lo, hi) = ((point.0, start), (point.0, end - 1));
        self.selection = Some(Selection {
            anchor: lo,
            head: hi,
            explicit: true,
            origin: (lo, hi),
        });
        true
    }

    /// Release the mouse button. A click that never moved leaves no highlight,
    /// only a cursor, so the next drag in the transcript is not swallowed.
    pub fn finish_selection(&mut self) {
        self.dragging = false;
        if self.selection_bounds().is_none() {
            self.selection = None;
        }
    }

    /// Whether a click started in the prompt and the button is still down.
    pub fn is_dragging(&self) -> bool {
        self.dragging
    }

    /// Scroll the draft inside the box without moving the cursor.
    pub fn scroll_by(&mut self, width: u16, max_h: u16, delta: i32) {
        self.ensure_rows(text_width(width));
        let rows = self.cached_rows().len();
        let max_scroll = rows.saturating_sub(Self::visible(rows, max_h)) as i32;
        self.scroll = (self.scroll as i32 + delta).clamp(0, max_scroll) as usize;
        self.follow = false;
    }

    /// Logical position under `(cell, row)` of the render area. Rows below the
    /// draft clamp to its end; columns past a row's text clamp to that row so
    /// a click never jumps the cursor to a line you did not point at.
    fn point_at(&mut self, width: u16, max_h: u16, cell: u16, row: u16) -> Point {
        self.ensure_rows(text_width(width));
        let rows = self.cached_rows();
        let visible = Self::visible(rows.len(), max_h);
        let scroll = self.scroll.min(rows.len().saturating_sub(visible));
        let index = (scroll + row as usize).min(rows.len().saturating_sub(1));
        let Some(target) = rows.get(index) else {
            return (0, 0);
        };
        let text = &self.lines[target.line];
        let cell = (cell as usize).saturating_sub(GUTTER);
        let mut col = target.start;
        let mut used = 0usize;
        let from = char_to_byte(text, target.start);
        let to = char_to_byte(text, target.end);
        for grapheme in text[from..to].graphemes(true) {
            let cells = text::width(grapheme);
            if used + cells > cell {
                break;
            }
            used += cells;
            col += grapheme.chars().count();
        }
        let end = if target.last {
            target.end
        } else {
            target.end.saturating_sub(1).max(target.start)
        };
        (target.line, col.min(end))
    }

    // ---------- selection ----------

    pub fn has_selection(&self) -> bool {
        self.selection_bounds().is_some()
    }

    pub fn clear_selection(&mut self) -> bool {
        self.dragging = false;
        self.selection.take().is_some()
    }

    /// Selection as (low, high) logical points, or `None` when it is empty.
    fn selection_bounds(&self) -> Option<(Point, Point)> {
        let selection = self.selection?;
        if selection.is_empty() {
            return None;
        }
        Some(if selection.anchor <= selection.head {
            (selection.anchor, selection.head)
        } else {
            (selection.head, selection.anchor)
        })
    }

    /// The highlighted text, newline-joined across logical lines.
    pub fn selected_text(&self) -> Option<String> {
        let (lo, hi) = self.selection_bounds()?;
        if lo.0 >= self.lines.len() {
            return None;
        }
        let mut out = String::new();
        for index in lo.0..=hi.0.min(self.lines.len() - 1) {
            let line = &self.lines[index];
            let chars = line.chars().count();
            let from = if index == lo.0 { lo.1.min(chars) } else { 0 };
            // Inclusive of the character under the release cell: a drag that
            // ends ON the closing brace must copy the brace.
            let to = if index == hi.0 {
                hi.1.saturating_add(1).min(chars)
            } else {
                chars
            };
            if index > lo.0 {
                out.push('\n');
            }
            out.push_str(&line[char_to_byte(line, from)..char_to_byte(line, to)]);
        }
        Some(out)
    }
}

/// Cells left for text once the gutter is paid for.
fn text_width(width: u16) -> usize {
    (width as usize).saturating_sub(GUTTER).max(1)
}

/// Soft-wrap one logical line into `(start, end)` char-column ranges of at
/// most `width` cells, breaking at the last space that fits and hard-breaking
/// words too long for a row. Always returns at least one range.
fn wrap_cols(line: &str, width: usize) -> Vec<(usize, usize)> {
    let width = width.max(1);
    // (first char column, cells, is a break opportunity) per grapheme cluster.
    let mut marks: Vec<(usize, usize, bool)> = Vec::new();
    let mut chars = 0usize;
    for grapheme in line.graphemes(true) {
        marks.push((
            chars,
            text::width(grapheme),
            grapheme.chars().all(char::is_whitespace),
        ));
        chars += grapheme.chars().count();
    }
    if marks.is_empty() {
        return vec![(0, 0)];
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    while i < marks.len() {
        let mut used = 0usize;
        let mut j = i;
        let mut last_break: Option<usize> = None;
        while j < marks.len() {
            if used + marks[j].1 > width {
                break;
            }
            used += marks[j].1;
            j += 1;
            if marks[j - 1].2 {
                last_break = Some(j);
            }
        }
        let cut = if j == marks.len() {
            j
        } else {
            match last_break {
                Some(brk) if brk > i => brk,
                _ => j.max(i + 1),
            }
        };
        let end = marks.get(cut).map(|m| m.0).unwrap_or(chars);
        out.push((marks[i].0, end));
        i = cut;
    }
    out
}

/// Where a selection crosses one display row, as char columns of its line.
fn row_selection(lo: Point, hi: Point, row: &Row) -> Option<(usize, usize)> {
    // `hi` is inclusive; painting wants a half-open range, so the highlight
    // covers exactly the characters a copy carries.
    let hi = (hi.0, hi.1.saturating_add(1));
    let start = lo.max((row.line, row.start));
    let end = hi.min((row.line, row.end));
    if start >= end {
        return None;
    }
    Some((start.1, end.1))
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

fn grapheme_boundaries(s: &str) -> Vec<usize> {
    // Vec::new, not with_capacity(graphemes().count()): sizing the Vec by
    // segmenting the whole string first costs a second full pass per call.
    let mut boundaries = vec![0];
    let mut chars = 0usize;
    for grapheme in s.graphemes(true) {
        chars += grapheme.chars().count();
        boundaries.push(chars);
    }
    boundaries
}

/// Char-index grapheme boundaries of `s`, lazily: 0, then one entry per
/// grapheme. The cursor helpers below walk this only as far as the cursor,
/// so a keystroke costs O(cursor), never a whole-line table allocation.
fn boundary_iter(s: &str) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(0).chain(s.graphemes(true).scan(0usize, |chars, grapheme| {
        *chars += grapheme.chars().count();
        Some(*chars)
    }))
}

fn previous_grapheme_boundary(s: &str, col: usize) -> usize {
    let mut prev = 0;
    for boundary in boundary_iter(s) {
        if boundary >= col {
            break;
        }
        prev = boundary;
    }
    prev
}

fn next_grapheme_boundary(s: &str, col: usize) -> usize {
    // Falls through to the final boundary, which is the char count of `s`.
    let mut last = 0;
    for boundary in boundary_iter(s) {
        if boundary > col {
            return boundary;
        }
        last = boundary;
    }
    last
}

fn floor_grapheme_boundary(s: &str, col: usize) -> usize {
    let mut prev = 0;
    for boundary in boundary_iter(s) {
        if boundary > col {
            break;
        }
        prev = boundary;
    }
    prev
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
    fn large_paste_collapses_to_a_marker_and_reads_back_expanded() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let blob: String = (1..=12).map(|i| format!("line {i}\n")).collect();
        let blob = blob.trim_end().to_string();
        composer.insert_str("before ");
        composer.insert_paste(&blob);
        // The draft shows one marker row, not twelve pasted rows.
        assert_eq!(composer.lines.len(), 1);
        assert_eq!(composer.lines[0], "before [pasted #1: 12 lines]");
        // Every whole-draft reader gets the real bytes: submit, queueing,
        // and history must never carry the marker.
        assert_eq!(composer.text(), format!("before {blob}"));
        let taken = composer.take();
        assert_eq!(taken, format!("before {blob}"));
        assert!(composer.pastes.is_empty(), "clear drains the paste store");
    }

    /// Ctrl+O expands the occurrence under the cursor, never the first in
    /// the row: a byte-identical duplicate typed earlier in the row must
    /// neither swallow the gesture nor receive the splice, and the spent
    /// reference means the surviving duplicate is literal text on every
    /// later path, never a second delivery.
    #[test]
    fn ctrl_o_expands_the_marker_under_the_cursor_not_the_first() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let blob: String = (1..=12).map(|i| format!("line {i}\n")).collect();
        let blob = blob.trim_end().to_string();
        composer.insert_paste(&blob);
        // Type a byte-identical duplicate BEFORE the genuine marker.
        composer.row = 0;
        composer.col = 0;
        composer.insert_str("[pasted #1: 12 lines] ");
        // Cursor onto the genuine marker, just past the duplicate.
        composer.row = 0;
        composer.col = "[pasted #1: 12 lines] ".chars().count() + 1;
        assert!(
            composer.expand_paste_at_cursor(),
            "the genuine marker under the cursor expands"
        );
        assert!(
            composer.lines[0].starts_with("[pasted #1: 12 lines] line 1"),
            "the duplicate stays and the genuine marker became content: {}",
            composer.lines[0]
        );
        let text = composer.text();
        assert_eq!(text.matches("line 12").count(), 1, "one delivery: {text}");
        assert!(
            text.starts_with("[pasted #1: 12 lines] "),
            "the duplicate is literal on the submit path too: {text}"
        );
    }

    /// A hand-typed duplicate of a marker is text, not a second handle on
    /// the hidden bytes: each insertion wrote its marker exactly once, so
    /// exactly one occurrence expands and the duplicate stays literal
    /// instead of injecting the paste twice.
    #[test]
    fn a_typed_marker_duplicate_stays_literal() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let blob: String = (1..=12).map(|i| format!("line {i}\n")).collect();
        let blob = blob.trim_end().to_string();
        composer.insert_paste(&blob);
        // The user types the marker's exact spelling after the real one.
        composer.insert_str(" [pasted #1: 12 lines]");
        let text = composer.text();
        assert_eq!(
            text.matches("line 12").count(),
            1,
            "the hidden bytes expand once: {text}"
        );
        assert!(
            text.ends_with(" [pasted #1: 12 lines]"),
            "the typed duplicate stays literal: {text}"
        );
    }

    #[test]
    fn small_pastes_splice_literally() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_paste("just\nthree\nlines");
        assert_eq!(composer.lines.len(), 3);
        assert_eq!(composer.text(), "just\nthree\nlines");
        assert!(composer.pastes.is_empty());
    }

    #[test]
    fn a_single_line_blob_collapses_on_the_char_bound() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let blob = "x".repeat(2000);
        composer.insert_paste(&blob);
        assert_eq!(composer.lines[0], "[pasted #1: 2000 chars]");
        assert_eq!(composer.text(), blob);
    }

    #[test]
    fn expand_paste_at_cursor_splices_the_bytes_back_for_editing() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let blob: String = (1..=12)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        composer.insert_str("head ");
        composer.insert_paste(&blob);
        composer.insert_str(" tail");
        // The cursor sits after " tail", off the marker: no expansion.
        assert!(!composer.expand_paste_at_cursor());
        // Move onto the marker (its closing bracket) and expand in place.
        for _ in 0..5 {
            composer.col -= 1;
        }
        assert!(composer.expand_paste_at_cursor());
        assert_eq!(composer.lines.len(), 12);
        assert_eq!(composer.lines[0], "head line 1");
        assert_eq!(composer.lines[11], "line 12 tail");
        assert_eq!(composer.text(), format!("head {blob} tail"));
        assert_eq!((composer.row, composer.col), (11, "line 12".chars().count()));
        // Idempotent: the marker is gone, nothing expands twice.
        assert!(!composer.expand_paste_at_cursor());
    }

    #[test]
    fn an_edited_marker_never_expands() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let blob: String = (1..=12).map(|i| format!("l{i}\n")).collect();
        composer.insert_paste(blob.trim_end());
        // Damage the marker itself; a char merely appended after the closing
        // bracket leaves the marker intact and expanding.
        composer.lines[0].insert(1, '!');
        composer.col += 1;
        assert!(!composer.expand_paste_at_cursor());
        assert_eq!(composer.text(), "[!pasted #1: 12 lines]");
    }

    #[test]
    fn empty_composer_uses_single_purposeful_placeholder() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let (lines, cursor_x, cursor_y) = composer.render(40, 3);
        assert_eq!(lines.len(), 1);
        assert_eq!(plain(&lines[0]), "❯ Describe a task");
        assert_eq!((cursor_x, cursor_y), (2, 0));
    }

    /// A multi-line draft pays the gutter in blank cells, not in a glyph per
    /// row: the text stays aligned under the caret and nothing competes with
    /// it for the eye. The transcript renders a wrapped user message the same
    /// way, so a draft and its echo line up column for column.
    #[test]
    fn multiline_composer_indents_continuation_rows_with_blank_cells() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("first\nsecond");
        let (lines, cursor_x, cursor_y) = composer.render(40, 3);
        assert_eq!(plain(&lines[0]), "❯ first");
        assert_eq!(plain(&lines[1]), "  second");
        assert_eq!((cursor_x, cursor_y), (8, 1));
    }

    /// The one row that still earns a glyph: a top row that is not the start
    /// of the draft, where `…` is the only thing saying the view is scrolled
    /// and there is text above. Rows under it stay blank.
    #[test]
    fn only_a_scrolled_top_row_marks_the_gutter() {
        let mut composer = Composer::new(&std::env::temp_dir());
        for i in 0..10 {
            composer.insert_str(&format!("line {i:02}\n"));
        }
        let (width, height) = (20, 4);
        let rows: Vec<String> = composer.render(width, height).0.iter().map(plain).collect();
        assert_eq!(rows[0], "… line 07");
        assert!(
            rows[1..].iter().all(|row| row.starts_with("  ")),
            "a row below the top should not repeat the scroll mark: {rows:?}",
        );
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

    /// A paste must splice whole segments, never insert char by char:
    /// per-char insertion re-scans the line prefix for every char, which is
    /// quadratic over a single-line paste and froze the UI for seconds on a
    /// 100 KB minified blob.
    #[test]
    fn a_paste_splices_segments_instead_of_inserting_per_char() {
        let mut composer = Composer::new(&std::env::temp_dir());
        let blob = "x".repeat(10_000);
        composer.insert_str(&blob);
        assert_eq!(
            composer.char_inserts, 0,
            "paste fell back to per-char insertion"
        );
        assert_eq!(composer.text(), blob);
        // Typing still lands one insert per char.
        composer.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE));
        assert_eq!(composer.char_inserts, 1);
    }

    #[test]
    fn a_crlf_paste_drops_carriage_returns_and_splits_lines() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("first\r\nsecond\rhalf\r\n");
        assert_eq!(composer.text(), "first\nsecondhalf\n");
        assert_eq!((composer.row, composer.col), (2, 0));
    }

    #[test]
    fn a_mid_line_paste_lands_at_the_cursor() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("head tail");
        for _ in 0..4 {
            composer.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        }
        composer.insert_str("mid\nnext ");
        assert_eq!(composer.text(), "head mid\nnext tail");
        assert_eq!((composer.row, composer.col), (1, 5));
    }

    /// The lazy cursor helpers must agree with the materialized boundary
    /// table at every column, including past-the-end columns.
    #[test]
    fn lazy_grapheme_boundaries_match_the_materialized_table() {
        for s in ["", "abc", "e\u{301}👩‍💻x", "漢字 mix e\u{301}", "👩‍💻👩‍💻"] {
            let table = grapheme_boundaries(s);
            let chars = s.chars().count();
            for col in 0..=chars + 1 {
                assert_eq!(
                    previous_grapheme_boundary(s, col),
                    table.iter().copied().rfind(|b| *b < col).unwrap_or(0),
                    "previous at {col} in {s:?}"
                );
                assert_eq!(
                    next_grapheme_boundary(s, col),
                    table.iter().copied().find(|b| *b > col).unwrap_or(chars),
                    "next at {col} in {s:?}"
                );
                assert_eq!(
                    floor_grapheme_boundary(s, col),
                    table.iter().copied().rfind(|b| *b <= col).unwrap_or(0),
                    "floor at {col} in {s:?}"
                );
            }
        }
    }

    #[test]
    fn composer_cursor_uses_terminal_cell_width() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("漢e\u{301}👩‍💻");
        let (_, cursor_x, cursor_y) = composer.render(40, 3);
        assert_eq!((cursor_x, cursor_y), (7, 0));
    }

    /// A pasted paragraph is one logical line. Without soft wrapping it drew
    /// as a single clipped row and the cursor ran off past the box edge, so
    /// most of the draft was neither visible nor reachable.
    #[test]
    fn a_long_line_wraps_into_rows_instead_of_running_past_the_box() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str(&"word ".repeat(20)); // 100 cells on one line
        let width = 20;

        assert_eq!(composer.height(width), 6);
        let (lines, cursor_x, cursor_y) = composer.render(width, 6);
        assert_eq!(lines.len(), 6);
        for line in &lines {
            assert!(
                text::width(&plain(line)) <= width as usize,
                "row overflows the box: {:?}",
                plain(line),
            );
        }
        // The cursor is at the end of the draft, inside the box on both axes.
        assert!(cursor_x < width, "cursor escaped to x={cursor_x}");
        assert_eq!(cursor_y, 5);
    }

    /// Walking a long pasted draft with the cursor keys must not re-wrap it:
    /// the wrap only depends on the text and the width, neither of which a
    /// cursor move touches.
    #[test]
    fn moving_the_cursor_does_not_re_wrap_the_draft() {
        // A unique directory that is never created: the composer only reads
        // history.json, so this test is isolated from every other one's history.
        let mut composer = Composer::new(&crate::test_temp_dir("openmax-composer-wraps"));
        composer.insert_str(&"word ".repeat(200));
        composer.render(40, 6);
        let base = composer.wraps;
        assert!(base > 0, "the first paint has to wrap once");

        for code in [
            KeyCode::Left,
            KeyCode::Right,
            KeyCode::Up,
            KeyCode::Down,
            KeyCode::Home,
            KeyCode::End,
        ] {
            composer.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
            composer.render(40, 6);
        }
        for code in [KeyCode::Char('a'), KeyCode::Char('e')] {
            composer.handle_key(KeyEvent::new(code, KeyModifiers::CONTROL));
            composer.render(40, 6);
        }
        assert_eq!(composer.wraps, base, "cursor movement re-wrapped the draft");

        // A width change and an edit both still do.
        composer.render(30, 6);
        assert_eq!(composer.wraps, base + 1);
        composer.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        composer.render(30, 6);
        assert_eq!(composer.wraps, base + 2);
    }

    /// Wrapping is by whole words, matching how the transcript above wraps.
    #[test]
    fn wrapping_breaks_on_words_and_hard_breaks_only_unbreakable_ones() {
        assert_eq!(wrap_cols("hello world foo", 10), vec![(0, 6), (6, 15)]);
        assert_eq!(wrap_cols("", 10), vec![(0, 0)]);
        // A token longer than the row still has to fit somewhere.
        assert_eq!(wrap_cols("abcdefgh", 4), vec![(0, 4), (4, 8)]);
        // Double-width cells are counted as cells, not as chars.
        assert_eq!(wrap_cols("漢字漢字", 4), vec![(0, 2), (2, 4)]);
    }

    /// The wheel reaches rows scrolled above the window, and the next key
    /// press snaps back so typing never happens off screen.
    #[test]
    fn the_wheel_reaches_rows_above_the_window_and_typing_snaps_back() {
        let mut composer = Composer::new(&std::env::temp_dir());
        for i in 0..20 {
            composer.insert_str(&format!("line {i:02}\n"));
        }
        let (width, height) = (20, 6);

        let visible = |c: &mut Composer| {
            c.render(width, height)
                .0
                .iter()
                .map(plain)
                .collect::<Vec<_>>()
        };
        // The tail is what a fresh draft shows.
        assert_eq!(visible(&mut composer)[0], "… line 15");

        composer.scroll_by(width, height, -8);
        assert_eq!(visible(&mut composer)[0], "… line 07");

        // Scrolling stops at the top rather than running off the draft.
        composer.scroll_by(width, height, -50);
        assert_eq!(visible(&mut composer)[0], "❯ line 00");

        composer.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(visible(&mut composer)[0], "… line 15");
    }

    /// Clicking lands the cursor on the character pointed at, in the wrapped
    /// row pointed at, not on whatever logical line happens to be last.
    #[test]
    fn a_click_lands_the_cursor_on_the_row_and_column_pointed_at() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("hello world foo");
        let (width, height) = (12, 6);
        // Rows are "hello " and "world foo"; the gutter costs two cells.
        composer.click_at(width, height, GUTTER as u16 + 3, 1);
        assert_eq!((composer.row, composer.col), (0, 9));

        // Past the end of a wrapped row, the cursor stays on that row.
        composer.click_at(width, height, 60, 0);
        assert_eq!((composer.row, composer.col), (0, 5));

        // Below the draft it clamps to the end, never out of bounds.
        composer.click_at(width, height, 60, 40);
        assert_eq!((composer.row, composer.col), (0, 15));
    }

    #[test]
    fn a_click_respects_double_width_cells() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("漢字ab");
        // Cell 2 is the second half of 漢: the cursor goes before it.
        composer.click_at(40, 6, GUTTER as u16 + 1, 0);
        assert_eq!(composer.col, 0);
        composer.click_at(40, 6, GUTTER as u16 + 2, 0);
        assert_eq!(composer.col, 1);
    }

    /// Dragging over the prompt yields exactly the source text under it,
    /// across wrapped rows and across logical lines.
    #[test]
    fn dragging_selects_the_source_text_under_the_pointer() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("hello world foo");
        let (width, height) = (12, 6);

        composer.click_at(width, height, GUTTER as u16, 0);
        composer.drag_to(width, height, GUTTER as u16 + 4, 1);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("hello world"));

        // Across logical lines the newline comes with it.
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("alpha\nbeta");
        composer.click_at(40, height, GUTTER as u16 + 2, 0);
        composer.drag_to(40, height, GUTTER as u16 + 2, 1);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("pha\nbet"));
    }

    /// Double-click grabs the whole token a developer meant, including the
    /// one-character case that an inclusive endpoint alone cannot express.
    #[test]
    fn double_click_selects_the_word_under_the_pointer() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("see crates/tui/src/app.rs:42 now");
        let (width, height) = (60, 6);

        let at = |c: &mut Composer, cell: usize| {
            c.select_word_at(width, height, (GUTTER + cell) as u16, 0);
            c.finish_selection();
            c.selected_text()
        };
        assert_eq!(at(&mut composer, 1).as_deref(), Some("see"));
        assert_eq!(
            at(&mut composer, 10).as_deref(),
            Some("crates/tui/src/app.rs:42"),
        );

        // A one-character word is a real selection, not an empty one.
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("let x = 1");
        assert_eq!(at(&mut composer, 4).as_deref(), Some("x"));
    }

    #[test]
    fn triple_click_selects_the_logical_line_even_where_it_wraps() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("alpha\nhello world foo bar\nomega");
        // Width 12 wraps the middle line across rows; a triple-click on any of
        // them still means the whole logical line.
        let (width, height) = (12, 6);
        for row in [0u16, 1] {
            let mut c = Composer::new(&std::env::temp_dir());
            c.insert_str("hello world foo bar");
            c.select_line_at(width, height, GUTTER as u16, row);
            c.finish_selection();
            assert_eq!(
                c.selected_text().as_deref(),
                Some("hello world foo bar"),
                "row {row}",
            );
        }
        // And it stops at the logical line, never running into the next.
        composer.select_line_at(60, height, GUTTER as u16, 1);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("hello world foo bar"));
    }

    /// A gesture that finds nothing has to clear, not leave the last
    /// highlight standing: the highlight would then lie about what a copy
    /// carries, which is the one thing this surface promises.
    #[test]
    fn a_gesture_on_an_empty_line_clears_the_previous_selection() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("alpha\n\nbeta");
        let (width, height) = (40, 6);

        composer.select_word_at(width, height, GUTTER as u16, 0);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("alpha"));

        // Row 1 is the empty logical line.
        assert!(!composer.select_word_at(width, height, GUTTER as u16, 1));
        assert_eq!(composer.selected_text(), None, "stale word selection");
        assert!(!composer.has_selection());

        composer.select_word_at(width, height, GUTTER as u16, 0);
        composer.finish_selection();
        assert!(composer.has_selection());
        assert!(!composer.select_line_at(width, height, GUTTER as u16, 1));
        assert_eq!(composer.selected_text(), None, "stale line selection");
    }

    /// A drag extends a gesture from the range it picked and never shrinks
    /// the selection below it: dragging inside the double-clicked word keeps
    /// the word, dragging past it leads with the pointer. The transcript
    /// clamps the same way.
    #[test]
    fn a_composer_gesture_drag_never_shrinks_below_the_word() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("alpha beta gamma");
        let (width, height) = (40, 6);
        let cell = |col: usize| (GUTTER + col) as u16;

        // Word gesture on "beta", then a drag that stays inside the word.
        composer.select_word_at(width, height, cell(7), 0);
        composer.drag_to(width, height, cell(8), 0);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("beta"));

        // A drag past the word extends from its far edge, inclusive.
        composer.select_word_at(width, height, cell(7), 0);
        composer.drag_to(width, height, cell(12), 0);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("beta ga"));

        // A drag before the word anchors on its end.
        composer.select_word_at(width, height, cell(7), 0);
        composer.drag_to(width, height, cell(2), 0);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("pha beta"));
    }

    /// The guarantee #132 established for the transcript, now true here too.
    #[test]
    fn the_release_cell_is_carried_not_dropped() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("fn main() {}");
        composer.click_at(40, 6, GUTTER as u16, 0);
        composer.drag_to(40, 6, GUTTER as u16 + 11, 0);
        composer.finish_selection();
        assert_eq!(composer.selected_text().as_deref(), Some("fn main() {}"));
    }

    /// A click that never moved leaves a cursor, not a highlight, so it can
    /// never swallow the Ctrl+C that follows it.
    #[test]
    fn a_click_without_a_drag_leaves_no_selection() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("hello");
        composer.click_at(40, 6, GUTTER as u16 + 2, 0);
        assert!(composer.is_dragging());
        composer.finish_selection();
        assert!(!composer.is_dragging());
        assert!(!composer.has_selection());
        assert_eq!(composer.selected_text(), None);
    }

    #[test]
    fn only_the_selected_cells_are_highlighted() {
        let mut composer = Composer::new(&std::env::temp_dir());
        composer.insert_str("alpha\nbeta");
        composer.click_at(40, 6, GUTTER as u16 + 2, 0);
        composer.drag_to(40, 6, GUTTER as u16 + 2, 1);
        composer.finish_selection();

        let highlighted = |line: &Line<'static>| -> String {
            line.spans
                .iter()
                .filter(|span| span.style.bg == Some(theme::SELECT()))
                .map(|span| span.content.as_ref())
                .collect()
        };
        let (lines, _, _) = composer.render(40, 6);
        assert_eq!(plain(&lines[0]), "❯ alpha");
        assert_eq!(highlighted(&lines[0]), "pha");
        assert_eq!(highlighted(&lines[1]), "bet");
        // The highlight is exactly the text a copy carries.
        assert_eq!(
            format!("{}\n{}", highlighted(&lines[0]), highlighted(&lines[1])),
            composer.selected_text().unwrap(),
        );

        // Editing drops the highlight rather than leaving a stale one.
        composer.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        let (lines, _, _) = composer.render(40, 6);
        assert!(lines.iter().all(|line| highlighted(line).is_empty()));
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

    /// Insertion cost of a paste, the case the segment splice in
    /// `insert_str` exists for. Not a correctness test; run with:
    ///   cargo test -p open-max-tui --bin openmax --release -- --ignored --nocapture measure_paste
    #[test]
    #[ignore]
    fn measure_paste_cost() {
        use std::time::Instant;

        for (label, text) in [
            ("single-line-10k", "x".repeat(10_000)),
            ("single-line-100k", "x".repeat(100_000)),
            ("multi-line-100k", "let value = compute(input);\n".repeat(3_600)),
        ] {
            let t0 = Instant::now();
            let mut composer = Composer::new(&std::env::temp_dir());
            composer.insert_str(&text);
            let paste_ms = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(composer.text());
            println!("{label}: paste {paste_ms:.3} ms");
        }
    }

    /// Wrap cost for a draft the size of a real paste. Not a correctness
    /// test; run with:
    ///   cargo test -p open-max-tui --bin openmax --release -- --ignored --nocapture measure_wrap
    #[test]
    #[ignore]
    fn measure_wrap_cost_per_frame() {
        use std::time::Instant;

        for (label, text) in [
            ("one-line-prompt", "explain the ledger reconciliation".to_string()),
            ("paste-10k", "let value = compute(input, &config).unwrap_or_default();\n".repeat(180)),
            ("paste-100k", "let value = compute(input, &config).unwrap_or_default();\n".repeat(1800)),
        ] {
            let mut composer = Composer::new(&std::env::temp_dir());
            composer.insert_str(&text);
            // A frame asks twice: once to size the box, once to fill it.
            let frame = |c: &mut Composer| {
                std::hint::black_box(c.height(80));
                std::hint::black_box(c.render(80, 6));
            };
            frame(&mut composer);

            let t0 = Instant::now();
            for _ in 0..200 {
                frame(&mut composer);
            }
            let cached_us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;

            let arrow = KeyEvent::new(KeyCode::Left, KeyModifiers::NONE);
            let t0 = Instant::now();
            for _ in 0..50 {
                composer.handle_key(arrow);
                frame(&mut composer);
            }
            let move_us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;

            let key = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
            let t0 = Instant::now();
            for _ in 0..50 {
                composer.handle_key(key);
                frame(&mut composer);
            }
            let edit_us = t0.elapsed().as_secs_f64() * 1e6 / 50.0;

            eprintln!(
                "MEASURE {label} bytes={} cached_frame_us={cached_us:.1} frame_after_cursor_move_us={move_us:.1} frame_after_edit_us={edit_us:.1}",
                text.len(),
            );
        }
    }

}
