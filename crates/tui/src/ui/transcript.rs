//! Block-oriented conversation scrollback.
//!
//! Finished turns are typed blocks (user, assistant, tool, system) with
//! per-block wrap caches. Tools fold by default; selection and sticky user
//! headers support dual-focus navigation. Scroll offset is in wrapped lines
//! from the bottom; 0 follows the latest output.

use ratatui::prelude::CrosstermBackend;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::Terminal;
use unicode_width::UnicodeWidthChar;

use crate::theme;

pub type Term = Terminal<CrosstermBackend<std::io::Stdout>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BlockKind {
    User,
    Assistant,
    Tool,
    Thinking,
    System,
}

struct Block {
    kind: BlockKind,
    /// Full content when expanded (or the only content when not foldable).
    raw: Vec<Line<'static>>,
    /// Compact body when foldable; shown while `folded`.
    compact: Option<Vec<Line<'static>>>,
    folded: bool,
    /// Full tool output for expand / copy.
    full_output: Option<String>,
    cache_width: u16,
    cache_folded: bool,
    cache: Vec<Line<'static>>,
}

impl Block {
    fn new(kind: BlockKind, raw: Vec<Line<'static>>) -> Self {
        Self {
            kind,
            raw,
            compact: None,
            folded: false,
            full_output: None,
            cache_width: 0,
            cache_folded: false,
            cache: Vec::new(),
        }
    }

    fn tool(compact: Vec<Line<'static>>, full_output: String) -> Self {
        let header = compact
            .first()
            .cloned()
            .unwrap_or_else(|| Line::from("tool"));
        let mut full_lines = vec![header];
        for line in full_output.lines().take(80) {
            full_lines.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(theme::DIM()),
            )));
        }
        let total = full_output.lines().count();
        if total > 80 {
            full_lines.push(Line::from(Span::styled(
                format!("  … {} more lines", total - 80),
                Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
            )));
        }
        Self {
            kind: BlockKind::Tool,
            raw: full_lines,
            compact: Some(compact),
            folded: true,
            full_output: Some(full_output),
            cache_width: 0,
            cache_folded: true,
            cache: Vec::new(),
        }
    }

    fn source_lines(&self) -> &[Line<'static>] {
        if self.folded {
            if let Some(c) = &self.compact {
                return c;
            }
        }
        &self.raw
    }

    fn ensure_cache(&mut self, width: u16) {
        if self.cache_width == width && self.cache_folded == self.folded && !self.cache.is_empty() {
            return;
        }
        self.cache_width = width;
        self.cache_folded = self.folded;
        let mut wrapped = wrap_lines(self.source_lines(), width);
        if self.folded && self.compact.is_some() {
            wrapped.push(Line::from(Span::styled(
                "  ⋯ enter expand · y copy",
                Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
            )));
        }
        wrapped.push(Line::default());
        self.cache = wrapped;
    }

    fn invalidate(&mut self) {
        self.cache_width = 0;
        self.cache.clear();
    }
}

#[derive(Default)]
pub struct Transcript {
    blocks: Vec<Block>,
    wrapped: Vec<Line<'static>>,
    line_block: Vec<usize>,
    block_starts: Vec<usize>,
    width: u16,
    offset: usize,
    selected: Option<usize>,
    dirty: bool,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append styled lines as a system/notice block.
    pub fn push(&mut self, lines: Vec<Line<'static>>) {
        self.push_kind(BlockKind::System, lines);
    }

    pub fn push_kind(&mut self, kind: BlockKind, mut lines: Vec<Line<'static>>) {
        while lines
            .last()
            .is_some_and(|l| l.spans.is_empty() || line_is_blank(l))
        {
            lines.pop();
        }
        if lines.is_empty() {
            return;
        }
        let prev_len = if self.width > 0 {
            self.ensure_flat();
            self.wrapped.len()
        } else {
            0
        };
        self.blocks.push(Block::new(kind, lines));
        self.dirty = true;
        self.ensure_flat();
        if self.offset > 0 {
            let added = self.wrapped.len().saturating_sub(prev_len);
            self.offset = self.offset.saturating_add(added);
        }
    }

    pub fn push_user(&mut self, lines: Vec<Line<'static>>) {
        self.push_kind(BlockKind::User, lines);
    }

    pub fn push_assistant(&mut self, lines: Vec<Line<'static>>) {
        self.push_kind(BlockKind::Assistant, lines);
    }

    pub fn push_tool(&mut self, compact: Vec<Line<'static>>, full_output: String) {
        let prev_len = if self.width > 0 {
            self.ensure_flat();
            self.wrapped.len()
        } else {
            0
        };
        self.blocks.push(Block::tool(compact, full_output));
        self.dirty = true;
        self.ensure_flat();
        if self.offset > 0 {
            let added = self.wrapped.len().saturating_sub(prev_len);
            self.offset = self.offset.saturating_add(added);
        }
    }

    pub fn set_width(&mut self, width: u16) {
        if width != self.width {
            self.width = width;
            for b in &mut self.blocks {
                b.invalidate();
            }
            self.dirty = true;
            self.ensure_flat();
            self.offset = self.offset.min(self.wrapped.len());
        }
    }

    fn rebuild_flat(&mut self) {
        self.wrapped.clear();
        self.line_block.clear();
        self.block_starts.clear();
        if self.width == 0 {
            self.dirty = false;
            return;
        }
        for (bi, block) in self.blocks.iter_mut().enumerate() {
            self.block_starts.push(self.wrapped.len());
            block.ensure_cache(self.width);
            for line in &block.cache {
                self.wrapped.push(line.clone());
                self.line_block.push(bi);
            }
        }
        self.dirty = false;
    }

    fn ensure_flat(&mut self) {
        if self.dirty || (self.wrapped.is_empty() && !self.blocks.is_empty() && self.width > 0) {
            self.rebuild_flat();
        }
    }

    pub fn lines(&mut self) -> &[Line<'static>] {
        self.ensure_flat();
        &self.wrapped
    }

    pub fn len(&mut self) -> usize {
        self.ensure_flat();
        self.wrapped.len()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    #[allow(dead_code)]
    pub fn is_following(&self) -> bool {
        self.offset == 0
    }

    pub fn scroll_up(&mut self, n: usize) {
        self.offset = self.offset.saturating_add(n);
    }

    pub fn scroll_down(&mut self, n: usize) {
        self.offset = self.offset.saturating_sub(n);
    }

    pub fn follow(&mut self) {
        self.offset = 0;
        self.selected = None;
    }

    pub fn clamp_offset(&mut self, max: usize) {
        self.offset = self.offset.min(max);
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn select_next(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        let next = match self.selected {
            None => self.blocks.len() - 1,
            Some(i) if i + 1 < self.blocks.len() => i + 1,
            Some(i) => i,
        };
        self.selected = Some(next);
        self.scroll_to_block(next);
    }

    pub fn select_prev(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        let prev = match self.selected {
            None => self.blocks.len().saturating_sub(1),
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.selected = Some(prev);
        self.scroll_to_block(prev);
    }

    fn scroll_to_block(&mut self, bi: usize) {
        self.ensure_flat();
        let Some(&start) = self.block_starts.get(bi) else {
            return;
        };
        let end = self
            .block_starts
            .get(bi + 1)
            .copied()
            .unwrap_or(self.wrapped.len());
        let total = self.wrapped.len();
        self.offset = total.saturating_sub(end);
        let _ = start;
    }

    pub fn toggle_fold_selected(&mut self) -> bool {
        let Some(i) = self.selected else {
            return false;
        };
        self.toggle_fold_at(i)
    }

    /// Ctrl+O: expand the last tool block (or toggle if already selected).
    pub fn expand_last_tool(&mut self) -> bool {
        let i = self
            .blocks
            .iter()
            .rposition(|b| b.kind == BlockKind::Tool && b.compact.is_some());
        let Some(i) = i else {
            return false;
        };
        self.selected = Some(i);
        if self.blocks[i].folded {
            self.blocks[i].folded = false;
            self.blocks[i].invalidate();
            self.dirty = true;
            self.ensure_flat();
            true
        } else {
            self.toggle_fold_at(i)
        }
    }

    fn toggle_fold_at(&mut self, i: usize) -> bool {
        let Some(block) = self.blocks.get_mut(i) else {
            return false;
        };
        if block.compact.is_none() {
            return false;
        }
        block.folded = !block.folded;
        block.invalidate();
        self.dirty = true;
        self.ensure_flat();
        true
    }

    pub fn selected_copy_text(&self) -> Option<String> {
        let i = self.selected?;
        let b = self.blocks.get(i)?;
        if let Some(out) = &b.full_output {
            return Some(out.clone());
        }
        Some(lines_to_plain(&b.raw))
    }

    pub fn last_tool_output(&self) -> Option<&str> {
        self.blocks
            .iter()
            .rev()
            .find_map(|b| b.full_output.as_deref())
    }

    /// First line of the nearest user block above the viewport.
    pub fn sticky_user_line(&mut self, view_start_line: usize) -> Option<Line<'static>> {
        self.ensure_flat();
        if view_start_line == 0 || self.blocks.is_empty() {
            return None;
        }
        let bi = self.line_block.get(view_start_line).copied().unwrap_or(0);
        for i in (0..=bi).rev() {
            if self.blocks[i].kind == BlockKind::User {
                let start = self.block_starts[i];
                if start < view_start_line {
                    return self.blocks[i].source_lines().first().cloned();
                }
            }
        }
        None
    }

    pub fn is_selected_block_for_line(&self, line_idx: usize) -> bool {
        let Some(sel) = self.selected else {
            return false;
        };
        self.line_block.get(line_idx) == Some(&sel)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    /// Total wrapped height and current thumb position for a 1-col scrollbar.
    #[allow(dead_code)]
    pub fn scrollbar(&mut self, viewport: usize) -> Option<(usize, usize, usize)> {
        self.ensure_flat();
        let total = self.wrapped.len();
        if total <= viewport {
            return None;
        }
        // (total, viewport, offset_from_bottom)
        Some((total, viewport, self.offset))
    }
}

fn line_is_blank(l: &Line<'_>) -> bool {
    l.spans.iter().all(|s| s.content.trim().is_empty())
}

fn lines_to_plain(lines: &[Line<'static>]) -> String {
    lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Span-preserving word wrap. Greedy, breaking at the last space that fits;
/// hard-breaks tokens longer than the width.
pub fn wrap_lines(lines: &[Line<'static>], width: u16) -> Vec<Line<'static>> {
    let width = width.max(8) as usize;
    let mut out = Vec::new();
    for line in lines {
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
        self.complete_wrapped
            .iter()
            .chain(self.partial_wrapped.iter())
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
                wrap_lines(
                    &[Line::from(Span::raw(self.partial_raw.clone()))],
                    width,
                )
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
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

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
                inc.lines()
                    .map(|l| {
                        l.spans
                            .iter()
                            .map(|s| s.content.as_ref())
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>(),
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
        assert_eq!(
            text(&wrapped).join(""),
            "the quick brown fox jumps over the lazy dog"
        );
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
        assert!(t.len() >= 2);
        assert_eq!(text(t.lines())[0], "hello ");
    }

    #[test]
    fn transcript_rewraps_on_width_change() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push(vec![Line::from("abcdefghijklmnopqrst")]);
        let wide = t.len();
        t.set_width(10);
        assert!(t.len() >= wide);
    }

    #[test]
    fn scrolled_view_stays_anchored_when_new_blocks_arrive() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push(vec![Line::from("one")]);
        t.scroll_up(2);
        assert_eq!(t.offset(), 2);
        t.push(vec![Line::from("two")]);
        assert!(t.offset() >= 2);
        t.follow();
        assert_eq!(t.offset(), 0);
    }

    #[test]
    fn tool_block_folds_by_default() {
        let mut t = Transcript::new();
        t.set_width(80);
        let compact = vec![Line::from("✓ read_file foo"), Line::from("  preview")];
        t.push_tool(compact, "line1\nline2\nline3\nline4".into());
        assert_eq!(t.block_count(), 1);
        // Folded height is smaller than expanded.
        let folded = t.len();
        t.selected = Some(0);
        assert!(t.toggle_fold_selected());
        let expanded = t.len();
        assert!(expanded >= folded);
        assert!(t.toggle_fold_selected());
    }
}
