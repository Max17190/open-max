//! The conversation transcript. Finished blocks are stored as styled lines,
//! wrapped once per width, and rendered bottom anchored above the composer.
//! A scroll offset (in wrapped lines from the bottom) supports the mouse
//! wheel and PageUp/PageDown; offset 0 follows the latest output.

use ratatui::prelude::CrosstermBackend;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::Terminal;
use unicode_width::UnicodeWidthChar;

pub type Term = Terminal<CrosstermBackend<std::io::Stdout>>;

#[derive(Default)]
pub struct Transcript {
    raw: Vec<Line<'static>>,
    wrapped: Vec<Line<'static>>,
    width: u16,
    offset: usize,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a finished block followed by one blank spacer line.
    pub fn push(&mut self, mut lines: Vec<Line<'static>>) {
        lines.push(Line::default());
        if self.width > 0 {
            let added = wrap_lines(&lines, self.width);
            // While scrolled up, keep the visible text anchored in place.
            if self.offset > 0 {
                self.offset += added.len();
            }
            self.wrapped.extend(added);
        }
        self.raw.extend(lines);
    }

    /// Rewrap the whole transcript when the terminal width changes.
    pub fn set_width(&mut self, width: u16) {
        if width != self.width {
            self.width = width;
            self.wrapped = wrap_lines(&self.raw, width);
            self.offset = self.offset.min(self.wrapped.len());
        }
    }

    pub fn lines(&self) -> &[Line<'static>] {
        &self.wrapped
    }

    pub fn len(&self) -> usize {
        self.wrapped.len()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn scroll_up(&mut self, n: usize) {
        self.offset = self.offset.saturating_add(n);
    }

    pub fn scroll_down(&mut self, n: usize) {
        self.offset = self.offset.saturating_sub(n);
    }

    /// Jump back to following the latest output.
    pub fn follow(&mut self) {
        self.offset = 0;
    }

    /// Called once per frame with the maximum meaningful offset.
    pub fn clamp_offset(&mut self, max: usize) {
        self.offset = self.offset.min(max);
    }
}

/// Span-preserving word wrap. Greedy, breaking at the last space that fits;
/// hard-breaks tokens longer than the width.
pub fn wrap_lines(lines: &[Line<'static>], width: u16) -> Vec<Line<'static>> {
    let width = width.max(8) as usize;
    let mut out = Vec::new();
    for line in lines {
        // Flatten into styled chars.
        let chars: Vec<(char, Style)> = line
            .spans
            .iter()
            .flat_map(|s| s.content.chars().map(move |c| (c, s.style)))
            .collect();
        if chars.is_empty() {
            out.push(Line::default());
            continue;
        }
        let mut start = 0;
        while start < chars.len() {
            let mut used = 0usize;
            let mut end = start;
            let mut last_space: Option<usize> = None;
            while end < chars.len() {
                let w = chars[end].0.width().unwrap_or(0);
                if used + w > width {
                    break;
                }
                if chars[end].0 == ' ' {
                    last_space = Some(end);
                }
                used += w;
                end += 1;
            }
            let cut = if end == chars.len() {
                end
            } else {
                match last_space {
                    // Break after the space so continuation lines stay flush.
                    Some(s) if s > start => s + 1,
                    _ => end.max(start + 1),
                }
            };
            out.push(rebuild(&chars[start..cut]));
            start = cut;
        }
    }
    out
}

/// Reassemble styled chars into a Line, merging adjacent same-style runs.
fn rebuild(chars: &[(char, Style)]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut style: Option<Style> = None;
    for (c, s) in chars {
        match style {
            Some(current) if current == *s => buf.push(*c),
            Some(current) => {
                spans.push(Span::styled(std::mem::take(&mut buf), current));
                buf.push(*c);
                style = Some(*s);
            }
            None => {
                buf.push(*c);
                style = Some(*s);
            }
        }
    }
    if let Some(current) = style {
        if !buf.is_empty() {
            spans.push(Span::styled(buf, current));
        }
    }
    Line::from(spans)
}

/// Incrementally wraps growing streaming text. Only the current incomplete
/// line is re-wrapped on each token; completed lines are wrapped once.
#[derive(Default)]
pub struct StreamingWrap {
    width: u16,
    text_len: usize,
    complete_newlines: usize,
    complete_wrapped: Vec<Line<'static>>,
    partial_raw: String,
    partial_wrapped: Vec<Line<'static>>,
}

impl StreamingWrap {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn lines(&self) -> impl DoubleEndedIterator<Item = &Line<'static>> {
        self.complete_wrapped.iter().chain(self.partial_wrapped.iter())
    }

    pub fn update(&mut self, text: &str, width: u16) {
        if width != self.width || text.len() < self.text_len {
            self.rewrap_all(text, width);
            return;
        }
        if text.len() == self.text_len {
            return;
        }

        let delta = &text[self.text_len..];
        if delta.matches('\n').count() > 1 {
            self.rewrap_all(text, width);
            return;
        }

        let new_newlines = text.as_bytes().iter().filter(|&&b| b == b'\n').count();
        if new_newlines > self.complete_newlines {
            let completed = if let Some((before, _)) = delta.split_once('\n') {
                format!("{}{before}", self.partial_raw)
            } else {
                self.partial_raw.clone()
            };
            // Empty completed lines are paragraph breaks; keep them.
            let line = Line::from(Span::raw(completed));
            self.complete_wrapped.extend(wrap_lines(&[line], width));
            self.complete_newlines = new_newlines;
        }

        let partial = match text.rsplit_once('\n') {
            Some((_, p)) => p,
            None => text,
        };
        if partial != self.partial_raw {
            self.partial_raw = partial.to_string();
            self.partial_wrapped = if partial.is_empty() {
                Vec::new()
            } else {
                wrap_lines(&[Line::from(Span::raw(self.partial_raw.clone()))], width)
            };
        }
        self.text_len = text.len();
    }

    fn rewrap_all(&mut self, text: &str, width: u16) {
        self.width = width;
        self.text_len = text.len();
        self.complete_newlines = text.as_bytes().iter().filter(|&&b| b == b'\n').count();
        self.complete_wrapped.clear();
        self.partial_wrapped.clear();
        self.partial_raw.clear();

        if text.is_empty() {
            return;
        }

        if let Some((complete, partial)) = text.rsplit_once('\n') {
            // split('\n') rather than lines(): blank lines are paragraph
            // breaks and must survive, including one before the last newline.
            let raw: Vec<Line<'static>> = complete
                .split('\n')
                .map(|l| Line::from(Span::raw(l.to_string())))
                .collect();
            self.complete_wrapped = wrap_lines(&raw, width);
            self.partial_raw = partial.to_string();
            if !partial.is_empty() {
                self.partial_wrapped =
                    wrap_lines(&[Line::from(Span::raw(self.partial_raw.clone()))], width);
            }
        } else {
            self.partial_raw = text.to_string();
            self.partial_wrapped =
                wrap_lines(&[Line::from(Span::raw(self.partial_raw.clone()))], width);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    /// Reference semantics: every '\n' separates a line (blank lines are
    /// paragraph breaks and survive); a trailing empty partial renders nothing.
    fn full_wrap(input: &str, width: u16) -> Vec<String> {
        let mut parts: Vec<&str> = input.split('\n').collect();
        if parts.last() == Some(&"") {
            parts.pop();
        }
        let raw: Vec<Line<'static>> = parts
            .iter()
            .map(|l| Line::from(Span::raw(l.to_string())))
            .collect();
        text(&wrap_lines(&raw, width))
    }

    fn assert_incremental_matches(chunks: &[&str], width: u16) {
        let mut inc = StreamingWrap::default();
        let mut full = String::new();
        for chunk in chunks {
            full.push_str(chunk);
            inc.update(&full, width);
            assert_eq!(
                inc.lines().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>()).collect::<Vec<_>>(),
                full_wrap(&full, width),
                "after {chunk:?}"
            );
        }
    }

    #[test]
    fn streaming_wrap_matches_full_rewrap() {
        assert_incremental_matches(&["fn ", "main() ", "{\n", "    ok\n", "}"], 20);
    }

    #[test]
    fn streaming_wrap_keeps_paragraph_breaks() {
        // Blank line arriving token by token, and as one multi-newline delta.
        assert_incremental_matches(&["para1", "\n", "\n", "para2"], 20);
        assert_incremental_matches(&["para1\n\npara2", " more"], 20);
    }

    #[test]
    fn streaming_wrap_clears_on_reset() {
        let mut w = StreamingWrap::default();
        w.update("hello world", 10);
        assert!(w.lines().next().is_some());
        w.update("", 10);
        assert!(w.lines().next().is_none());
    }

    #[test]
    fn wraps_at_word_boundaries() {
        let lines = vec![Line::from("the quick brown fox jumps over the lazy dog")];
        let wrapped = wrap_lines(&lines, 16);
        for l in text(&wrapped) {
            assert!(l.chars().count() <= 16, "line too long: {l:?}");
        }
        // The wrapper never drops characters; it only chooses break points.
        assert_eq!(text(&wrapped).join(""), "the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn hard_breaks_long_tokens() {
        let lines = vec![Line::from("abcdefghijklmnopqrstuvwxyz")];
        let wrapped = wrap_lines(&lines, 10);
        assert_eq!(text(&wrapped), vec!["abcdefghij", "klmnopqrst", "uvwxyz"]);
    }

    #[test]
    fn empty_line_survives() {
        let wrapped = wrap_lines(&[Line::default()], 10);
        assert_eq!(wrapped.len(), 1);
    }

    #[test]
    fn transcript_appends_wrapped_blocks_with_spacer() {
        let mut t = Transcript::new();
        t.set_width(10);
        t.push(vec![Line::from("hello world wide")]);
        // "hello world wide" wraps to two lines at width 10, plus the spacer.
        assert_eq!(t.len(), 3);
        assert_eq!(text(t.lines())[0], "hello ");
    }

    #[test]
    fn transcript_rewraps_on_width_change() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push(vec![Line::from("abcdefghijklmnopqrst")]);
        assert_eq!(t.len(), 2);
        t.set_width(10);
        assert_eq!(t.len(), 3); // two wrapped lines plus the spacer
    }

    #[test]
    fn scrolled_view_stays_anchored_when_new_blocks_arrive() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push(vec![Line::from("one")]);
        t.scroll_up(2);
        assert_eq!(t.offset(), 2);
        t.push(vec![Line::from("two")]); // adds 2 wrapped lines
        assert_eq!(t.offset(), 4);
        t.follow();
        assert_eq!(t.offset(), 0);
    }
}
