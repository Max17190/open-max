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
    cache_maps: Vec<Option<CachedLineMap>>,
    selectable: String,
    selectable_chars: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedLineMap {
    /// Character offsets into `Block::selectable_text`.
    start: usize,
    end: usize,
    /// Terminal column where selectable content starts after UI gutters.
    x_offset: usize,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextPoint {
    pub block: usize,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TextSelection {
    anchor: TextPoint,
    head: TextPoint,
    dragging: bool,
}

impl Block {
    fn new(kind: BlockKind, raw: Vec<Line<'static>>) -> Self {
        let selectable = lines_to_plain(&raw);
        let selectable_chars = selectable.chars().count();
        Self {
            kind,
            raw,
            compact: None,
            folded: false,
            full_output: None,
            cache_width: 0,
            cache_folded: false,
            cache: Vec::new(),
            cache_maps: Vec::new(),
            selectable,
            selectable_chars,
        }
    }

    fn tool(compact: Vec<Line<'static>>, full_output: String) -> Self {
        let selectable = lines_to_plain(&compact);
        let selectable_chars = selectable.chars().count();
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
                Style::default()
                    .fg(theme::DIM())
                    .add_modifier(Modifier::ITALIC),
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
            cache_maps: Vec::new(),
            selectable,
            selectable_chars,
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
        self.selectable = lines_to_plain(self.source_lines());
        self.selectable_chars = self.selectable.chars().count();
        let gutter = match self.kind {
            BlockKind::User | BlockKind::Assistant | BlockKind::Tool => 2,
            BlockKind::Thinking | BlockKind::System => 0,
        };
        let content_width = width.saturating_sub(gutter).max(8);
        let (wrapped, maps) = wrap_lines_mapped(self.source_lines(), content_width);
        self.cache.clear();
        self.cache_maps.clear();
        for (i, (line, map)) in wrapped.into_iter().zip(maps).enumerate() {
            let (line, x_offset) = decorate_line(self.kind, line, i == 0, width);
            self.cache.push(line);
            self.cache_maps.push(map.map(|mut map| {
                map.x_offset = x_offset;
                map
            }));
        }
        if self.folded && self.compact.is_some() {
            self.cache.push(surface_line(
                Line::from(Span::styled(
                    "  ⋯ enter expand · y copy",
                    Style::default()
                        .fg(theme::DIM())
                        .add_modifier(Modifier::ITALIC),
                )),
                width,
                theme::SURFACE(),
            ));
            self.cache_maps.push(None);
        }
        self.cache.push(Line::default());
        self.cache_maps.push(None);
    }

    fn invalidate(&mut self) {
        self.cache_width = 0;
        self.cache.clear();
        self.cache_maps.clear();
    }

    fn selectable_text(&self) -> &str {
        &self.selectable
    }
}

#[derive(Default)]
pub struct Transcript {
    blocks: Vec<Block>,
    wrapped: Vec<Line<'static>>,
    line_block: Vec<usize>,
    line_maps: Vec<Option<CachedLineMap>>,
    block_starts: Vec<usize>,
    width: u16,
    offset: usize,
    selected: Option<usize>,
    text_selection: Option<TextSelection>,
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
        if self.width == 0 {
            self.blocks.push(Block::new(kind, lines));
            self.dirty = true;
            return;
        }
        if self.dirty {
            self.ensure_flat();
        }
        let prev_len = self.wrapped.len();
        self.blocks.push(Block::new(kind, lines));
        let bi = self.blocks.len() - 1;
        self.append_block_flat(bi);
        if self.offset > 0 {
            let added = self.wrapped.len().saturating_sub(prev_len);
            self.offset = self.offset.saturating_add(added);
        }
    }

    pub fn push_user(&mut self, lines: Vec<Line<'static>>) {
        self.push_kind(BlockKind::User, lines);
    }

    /// Drop the most recent user block (e.g. a submit that was blocked by
    /// `user_prompt_submit` before it entered the core transcript). Returns
    /// true when a user block was removed.
    pub fn pop_last_user(&mut self) -> bool {
        let Some(i) = self.blocks.iter().rposition(|b| b.kind == BlockKind::User) else {
            return false;
        };
        self.blocks.remove(i);
        // Selection / sticky indices past the removed block must retreat.
        if let Some(sel) = self.selected {
            if sel == i {
                self.selected = None;
            } else if sel > i {
                self.selected = Some(sel - 1);
            }
        }
        if let Some(selection) = self.text_selection {
            let (start, end) = normalized_selection(selection);
            if i >= start.block && i <= end.block {
                self.text_selection = None;
            } else {
                let retreat = |point: TextPoint| TextPoint {
                    block: point.block.saturating_sub(usize::from(point.block > i)),
                    offset: point.offset,
                };
                self.text_selection = Some(TextSelection {
                    anchor: retreat(selection.anchor),
                    head: retreat(selection.head),
                    dragging: selection.dragging,
                });
            }
        }
        self.dirty = true;
        self.ensure_flat();
        self.offset = self.offset.min(self.wrapped.len());
        true
    }

    pub fn push_assistant(&mut self, lines: Vec<Line<'static>>) {
        self.push_kind(BlockKind::Assistant, lines);
    }

    pub fn push_tool(&mut self, compact: Vec<Line<'static>>, full_output: String) {
        if self.width == 0 {
            self.blocks.push(Block::tool(compact, full_output));
            self.dirty = true;
            return;
        }
        if self.dirty {
            self.ensure_flat();
        }
        let prev_len = self.wrapped.len();
        self.blocks.push(Block::tool(compact, full_output));
        let bi = self.blocks.len() - 1;
        self.append_block_flat(bi);
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

    /// Theme changes affect cached line surfaces even when content and width
    /// stay constant.
    pub fn invalidate_styles(&mut self) {
        for block in &mut self.blocks {
            block.invalidate();
        }
        self.dirty = true;
    }

    fn rebuild_flat(&mut self) {
        self.wrapped.clear();
        self.line_block.clear();
        self.line_maps.clear();
        self.block_starts.clear();
        if self.width == 0 {
            self.dirty = false;
            return;
        }
        for (bi, block) in self.blocks.iter_mut().enumerate() {
            self.block_starts.push(self.wrapped.len());
            block.ensure_cache(self.width);
            for (line, map) in block.cache.iter().zip(&block.cache_maps) {
                self.wrapped.push(line.clone());
                self.line_block.push(bi);
                self.line_maps.push(map.clone());
            }
        }
        self.dirty = false;
    }

    /// Incrementally append one newly pushed block to the flat tables.
    /// `bi` must be the last block; width is unchanged and tables are current.
    fn append_block_flat(&mut self, bi: usize) {
        self.blocks[bi].ensure_cache(self.width);
        self.block_starts.push(self.wrapped.len());
        // Clone out of the block cache so we can extend disjoint flat tables.
        let lines = self.blocks[bi].cache.clone();
        let maps = self.blocks[bi].cache_maps.clone();
        for (line, map) in lines.into_iter().zip(maps) {
            self.wrapped.push(line);
            self.line_block.push(bi);
            self.line_maps.push(map);
        }
    }

    fn ensure_flat(&mut self) {
        if self.dirty || (self.wrapped.is_empty() && !self.blocks.is_empty() && self.width > 0) {
            self.rebuild_flat();
        }
    }

    /// Test-support oracle access: `draw_chat` paints via `fill_viewport`.
    #[cfg(test)]
    pub fn lines(&mut self) -> &[Line<'static>] {
        self.ensure_flat();
        &self.wrapped
    }

    /// Borrow a single wrapped history line without cloning the full buffer.
    /// Test-support oracle access.
    #[cfg(test)]
    pub fn line_at(&mut self, idx: usize) -> Option<&Line<'static>> {
        self.ensure_flat();
        self.wrapped.get(idx)
    }

    /// Clone history lines `[start, end)` into `out` once each.
    ///
    /// When `selected_bi` is `Some`, lines belonging to that block receive a
    /// quiet background. Text selection is painted later as a buffer overlay.
    pub fn fill_viewport(
        &mut self,
        out: &mut Vec<Line<'static>>,
        start: usize,
        end: usize,
        selected_bi: Option<usize>,
    ) {
        self.ensure_flat();
        let end = end.min(self.wrapped.len());
        let start = start.min(end);
        out.reserve(end - start);
        for idx in start..end {
            let mut line = self.wrapped[idx].clone();
            if selected_bi.is_some_and(|bi| self.line_block.get(idx) == Some(&bi)) {
                line = surface_line(line, self.width, theme::BORDER());
            }
            out.push(line);
        }
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

    /// Jump to the next user-turn block after the current selection.
    pub fn select_next_user(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        let start = self.selected.map(|i| i + 1).unwrap_or(0);
        if let Some(i) =
            (start..self.blocks.len()).find(|&i| self.blocks[i].kind == BlockKind::User)
        {
            self.selected = Some(i);
            self.scroll_to_block(i);
        }
    }

    /// Jump to the previous user-turn block before the current selection.
    pub fn select_prev_user(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        let end = self
            .selected
            .unwrap_or_else(|| self.blocks.len().saturating_sub(1));
        if let Some(i) = (0..=end)
            .rev()
            .find(|&i| self.blocks[i].kind == BlockKind::User && Some(i) != self.selected)
        {
            self.selected = Some(i);
            self.scroll_to_block(i);
            return;
        }
        // If nothing before selection, land on the nearest user at or before end.
        if let Some(i) = (0..=end)
            .rev()
            .find(|&i| self.blocks[i].kind == BlockKind::User)
        {
            self.selected = Some(i);
            self.scroll_to_block(i);
        }
    }

    /// Select the first block and scroll toward the top of history.
    /// Offset is lines-from-bottom; a large value is clamped to the top in draw.
    pub fn select_first(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        self.selected = Some(0);
        self.ensure_flat();
        self.offset = self.wrapped.len();
    }

    /// Follow the latest output and clear selection (bottom of scrollback).
    pub fn select_last_follow(&mut self) {
        self.follow();
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
            if self
                .text_selection
                .is_some_and(|selection| selection_contains_block(selection, i))
            {
                self.text_selection = None;
            }
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
        if self
            .text_selection
            .is_some_and(|selection| selection_contains_block(selection, i))
        {
            self.text_selection = None;
        }
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

    /// Plain text for scrollback find (user/assistant/tool/system).
    /// Tool blocks include the compact header and full output.
    pub fn block_search_text(&self, i: usize) -> Option<String> {
        let b = self.blocks.get(i)?;
        Some(block_search_text(b))
    }

    /// Plain search text for every block, in order.
    pub fn all_block_search_texts(&self) -> Vec<String> {
        self.blocks.iter().map(block_search_text).collect()
    }

    /// Select a block by index and scroll it into view. Test-support only:
    /// the app drives selection through the focus/scroll key handlers.
    #[cfg(test)]
    pub fn select_block(&mut self, bi: usize) {
        if bi >= self.blocks.len() {
            return;
        }
        self.selected = Some(bi);
        self.scroll_to_block(bi);
    }

    /// Select a find match: expand folded tools so full output is visible,
    /// then scroll into view.
    pub fn select_find_match(&mut self, bi: usize) {
        if bi >= self.blocks.len() {
            return;
        }
        if self.blocks[bi].kind == BlockKind::Tool
            && self.blocks[bi].folded
            && self.blocks[bi].compact.is_some()
        {
            self.blocks[bi].folded = false;
            self.blocks[bi].invalidate();
            if self
                .text_selection
                .is_some_and(|selection| selection_contains_block(selection, bi))
            {
                self.text_selection = None;
            }
            self.dirty = true;
            self.ensure_flat();
        }
        self.selected = Some(bi);
        self.scroll_to_block(bi);
    }

    /// One-line preview for find UI. Prefer a line containing `query` when set.
    pub fn block_preview(&self, i: usize, query: &str) -> Option<String> {
        let text = self.block_search_text(i)?;
        let q = query.trim();
        if !q.is_empty() {
            let q_low = q.to_lowercase();
            if let Some(line) = text.lines().find(|l| l.to_lowercase().contains(&q_low)) {
                return Some(line.trim().to_string());
            }
        }
        let line = text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("")
            .to_string();
        Some(line)
    }

    pub fn last_tool_output(&self) -> Option<&str> {
        self.blocks
            .iter()
            .rev()
            .find_map(|b| b.full_output.as_deref())
    }

    pub fn last_assistant_text(&self) -> Option<String> {
        self.blocks
            .iter()
            .rev()
            .find(|block| block.kind == BlockKind::Assistant)
            .map(|block| lines_to_plain(&block.raw))
    }

    pub fn has_text_selection(&self) -> bool {
        self.text_selection
            .is_some_and(|selection| selection.anchor != selection.head)
    }

    pub fn clear_text_selection(&mut self) -> bool {
        self.text_selection.take().is_some()
    }

    pub fn begin_text_selection_at(&mut self, line_idx: usize, x: usize) -> bool {
        let Some(point) = self.hit_test(line_idx, x) else {
            return false;
        };
        self.text_selection = Some(TextSelection {
            anchor: point,
            head: point,
            dragging: true,
        });
        true
    }

    pub fn update_text_selection_at(&mut self, line_idx: usize, x: usize) -> bool {
        let Some(point) = self.hit_test(line_idx, x) else {
            return false;
        };
        let Some(selection) = &mut self.text_selection else {
            return false;
        };
        selection.head = point;
        true
    }

    pub fn finish_text_selection(&mut self) {
        if let Some(selection) = &mut self.text_selection {
            selection.dragging = false;
            if selection.anchor == selection.head {
                self.text_selection = None;
            }
        }
    }

    fn hit_test(&mut self, line_idx: usize, x: usize) -> Option<TextPoint> {
        self.ensure_flat();
        let block = *self.line_block.get(line_idx)?;
        let map = self.line_maps.get(line_idx)?.as_ref()?;
        let relative = x.saturating_sub(map.x_offset);
        let offset = map.start + display_column_to_char_offset(&map.text, relative);
        Some(TextPoint {
            block,
            offset: offset.min(map.end),
        })
    }

    /// Highlight columns for an absolute wrapped history line.
    pub fn selection_columns(&mut self, line_idx: usize) -> Option<(usize, usize)> {
        self.ensure_flat();
        let selection = self.text_selection?;
        let (start, end) = normalized_selection(selection);
        if start == end {
            return None;
        }
        let block = *self.line_block.get(line_idx)?;
        if block < start.block || block > end.block {
            return None;
        }
        let map = self.line_maps.get(line_idx)?.as_ref()?;
        let block_len = self.blocks.get(block)?.selectable_chars;
        let range_start = if block == start.block {
            start.offset
        } else {
            0
        };
        let range_end = if block == end.block {
            end.offset
        } else {
            block_len
        };
        let local_start = range_start.max(map.start).min(map.end);
        let local_end = range_end.min(map.end).max(map.start);
        if local_start >= local_end {
            return None;
        }
        let start_col =
            map.x_offset + char_offset_to_display_column(&map.text, local_start - map.start);
        let end_col =
            map.x_offset + char_offset_to_display_column(&map.text, local_end - map.start);
        Some((start_col, end_col.max(start_col + 1)))
    }

    pub fn selected_text(&self) -> Option<String> {
        let selection = self.text_selection?;
        let (start, end) = normalized_selection(selection);
        if start == end || start.block >= self.blocks.len() || end.block >= self.blocks.len() {
            return None;
        }
        let mut parts = Vec::new();
        for block_index in start.block..=end.block {
            let block = &self.blocks[block_index];
            let text = block.selectable_text();
            let char_len = block.selectable_chars;
            let from = if block_index == start.block {
                start.offset.min(char_len)
            } else {
                0
            };
            let to = if block_index == end.block {
                end.offset.min(char_len)
            } else {
                char_len
            };
            parts.push(slice_chars(text, from, to));
        }
        Some(parts.join("\n\n"))
    }

    /// Index of the nearest user block whose start is above `view_start_line`.
    fn sticky_user_block_idx(&mut self, view_start_line: usize) -> Option<usize> {
        self.ensure_flat();
        if view_start_line == 0 || self.blocks.is_empty() {
            return None;
        }
        let bi = self.line_block.get(view_start_line).copied().unwrap_or(0);
        (0..=bi).rev().find(|&i| {
            self.blocks[i].kind == BlockKind::User && self.block_starts[i] < view_start_line
        })
    }

    /// Whether a sticky user header should render for this viewport start.
    pub fn has_sticky_user(&mut self, view_start_line: usize) -> bool {
        self.sticky_user_block_idx(view_start_line).is_some()
    }

    /// First line of the nearest user block above the viewport.
    pub fn sticky_user_line(&mut self, view_start_line: usize) -> Option<Line<'static>> {
        let i = self.sticky_user_block_idx(view_start_line)?;
        self.blocks[i].source_lines().first().cloned()
    }

    /// Test-support oracle for the selection marker `fill_viewport` paints.
    #[cfg(test)]
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

fn block_search_text(b: &Block) -> String {
    match &b.full_output {
        Some(out) => {
            let header = b
                .compact
                .as_ref()
                .map(|c| lines_to_plain(c))
                .unwrap_or_default();
            if header.is_empty() {
                out.clone()
            } else {
                format!("{header}\n{out}")
            }
        }
        None => lines_to_plain(&b.raw),
    }
}

/// Case-insensitive substring filter over block plain texts.
/// Returns indices into `texts` whose content matches `query`.
/// An empty query matches every block (browse-all mode).
pub fn filter_matching_indices(texts: &[String], query: &str) -> Vec<usize> {
    if query.is_empty() {
        return (0..texts.len()).collect();
    }
    let q = query.to_lowercase();
    texts
        .iter()
        .enumerate()
        .filter(|(_, t)| t.to_lowercase().contains(&q))
        .map(|(i, _)| i)
        .collect()
}

/// Span-preserving word wrap. Greedy, breaking at the last space that fits;
/// hard-breaks tokens longer than the width.
pub fn wrap_lines(lines: &[Line<'static>], width: u16) -> Vec<Line<'static>> {
    wrap_lines_mapped(lines, width).0
}

fn wrap_lines_mapped(
    lines: &[Line<'static>],
    width: u16,
) -> (Vec<Line<'static>>, Vec<Option<CachedLineMap>>) {
    let width = width.max(8) as usize;
    let mut out = Vec::new();
    let mut maps = Vec::new();
    let mut logical_start = 0usize;
    for line in lines {
        let chars: Vec<(char, Style)> = line
            .spans
            .iter()
            .flat_map(|s| s.content.chars().map(move |c| (c, s.style)))
            .collect();
        if chars.is_empty() {
            out.push(Line::default());
            maps.push(Some(CachedLineMap {
                start: logical_start,
                end: logical_start,
                x_offset: 0,
                text: String::new(),
            }));
            logical_start += 1;
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
            maps.push(Some(CachedLineMap {
                start: logical_start + start,
                end: logical_start + cut,
                x_offset: 0,
                text: chars[start..cut].iter().map(|(ch, _)| ch).collect(),
            }));
            start = cut;
        }
        logical_start += chars.len() + 1;
    }
    (out, maps)
}

fn decorate_line(
    kind: BlockKind,
    mut line: Line<'static>,
    first: bool,
    width: u16,
) -> (Line<'static>, usize) {
    match kind {
        BlockKind::User => {
            let prefix = if first { "❯ " } else { "  " };
            line.spans.insert(
                0,
                Span::styled(
                    prefix,
                    Style::default()
                        .fg(theme::USER())
                        .bg(theme::USER_BG())
                        .add_modifier(Modifier::BOLD),
                ),
            );
            (surface_line(line, width, theme::USER_BG()), 2)
        }
        BlockKind::Assistant => {
            line.spans
                .insert(0, Span::styled("│ ", Style::default().fg(theme::BORDER())));
            (line, 2)
        }
        BlockKind::Tool => {
            line.spans.insert(
                0,
                Span::styled(
                    "│ ",
                    Style::default().fg(theme::ACCENT()).bg(theme::SURFACE()),
                ),
            );
            (surface_line(line, width, theme::SURFACE()), 2)
        }
        BlockKind::Thinking | BlockKind::System => (line, 0),
    }
}

fn surface_line(
    mut line: Line<'static>,
    width: u16,
    background: ratatui::style::Color,
) -> Line<'static> {
    line.style = line.style.bg(background);
    for span in &mut line.spans {
        span.style = span.style.bg(background);
    }
    let used = line.width();
    if used < width as usize {
        line.spans.push(Span::styled(
            " ".repeat(width as usize - used),
            Style::default().bg(background),
        ));
    }
    line
}

fn normalized_selection(selection: TextSelection) -> (TextPoint, TextPoint) {
    if selection.anchor <= selection.head {
        (selection.anchor, selection.head)
    } else {
        (selection.head, selection.anchor)
    }
}

fn selection_contains_block(selection: TextSelection, block: usize) -> bool {
    let (start, end) = normalized_selection(selection);
    block >= start.block && block <= end.block
}

fn display_column_to_char_offset(text: &str, column: usize) -> usize {
    let mut used = 0usize;
    for (index, ch) in text.chars().enumerate() {
        let width = ch.width().unwrap_or(0);
        if used + width > column {
            return index;
        }
        used += width;
    }
    text.chars().count()
}

fn char_offset_to_display_column(text: &str, offset: usize) -> usize {
    text.chars()
        .take(offset)
        .map(|ch| ch.width().unwrap_or(0))
        .sum()
}

fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
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
    fn pop_last_user_removes_optimistic_bubble() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push_user(vec![Line::from("first")]);
        t.push_assistant(vec![Line::from("reply")]);
        t.push_user(vec![Line::from("blocked later")]);
        assert!(t.pop_last_user());
        assert_eq!(t.blocks.len(), 2);
        assert_eq!(t.blocks[0].kind, BlockKind::User);
        assert_eq!(t.blocks[1].kind, BlockKind::Assistant);
        // No user left at the end: second pop still finds the first user.
        assert!(t.pop_last_user());
        assert_eq!(t.blocks.len(), 1);
        assert_eq!(t.blocks[0].kind, BlockKind::Assistant);
        assert!(!t.pop_last_user());
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

    #[test]
    fn filter_matching_indices_empty_query_matches_all() {
        let texts = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(filter_matching_indices(&texts, ""), vec![0, 1, 2]);
    }

    #[test]
    fn filter_matching_indices_case_insensitive_substring() {
        let texts = vec![
            "Error: file not found".into(),
            "ok done".into(),
            "ERROR in path /tmp/foo".into(),
            "nothing here".into(),
        ];
        assert_eq!(filter_matching_indices(&texts, "error"), vec![0, 2]);
        assert_eq!(filter_matching_indices(&texts, "PATH"), vec![2]);
        assert_eq!(filter_matching_indices(&texts, "zzz"), Vec::<usize>::new());
    }

    #[test]
    fn block_search_text_includes_tool_full_output() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push_user(vec![Line::from("you: check logs")]);
        t.push_assistant(vec![Line::from("assistant: looking")]);
        t.push_tool(
            vec![Line::from("✓ read_file app.rs")],
            "secret_token_xyz\nline2".into(),
        );
        let texts = t.all_block_search_texts();
        assert_eq!(texts.len(), 3);
        assert!(texts[0].contains("check logs"));
        assert!(texts[1].contains("looking"));
        assert!(texts[2].contains("read_file"));
        assert!(texts[2].contains("secret_token_xyz"));
        let hits = filter_matching_indices(&texts, "secret_token");
        assert_eq!(hits, vec![2]);
    }

    #[test]
    fn select_block_sets_selection() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push_user(vec![Line::from("one")]);
        t.push_assistant(vec![Line::from("two")]);
        t.select_block(1);
        assert_eq!(t.selected(), Some(1));
        t.select_block(99);
        assert_eq!(t.selected(), Some(1));
    }

    #[test]
    fn append_matches_rebuild_oracle() {
        const W: u16 = 40;
        let mut incremental = Transcript::new();
        incremental.set_width(W);

        let kind_pushes: Vec<(BlockKind, Vec<Line<'static>>)> = vec![
            (
                BlockKind::User,
                vec![Line::from("hello from the user side")],
            ),
            (
                BlockKind::Assistant,
                vec![Line::from(
                    "a longer assistant reply that will wrap at this width",
                )],
            ),
            (BlockKind::System, vec![Line::from("notice")]),
            (
                BlockKind::User,
                vec![Line::from("second question with more words than fit")],
            ),
            (BlockKind::Assistant, vec![Line::from("short ok")]),
        ];

        for (i, (kind, lines)) in kind_pushes.iter().enumerate() {
            incremental.push_kind(*kind, lines.clone());

            // Oracle: push with width 0 (blocks only), then set_width forces rebuild_flat.
            let mut oracle = Transcript::new();
            for (kind, lines) in kind_pushes.iter().take(i + 1) {
                oracle.push_kind(*kind, lines.clone());
            }
            oracle.set_width(W);

            assert_eq!(
                text(incremental.lines()),
                text(oracle.lines()),
                "mismatch after {} kind blocks",
                i + 1
            );
        }

        // Also exercise push_tool against rebuild oracle.
        let tool_compact = vec![Line::from("✓ tool"), Line::from("  preview line")];
        let tool_out = "out1\nout2\nout3".to_string();
        incremental.push_tool(tool_compact.clone(), tool_out.clone());

        let mut oracle = Transcript::new();
        for (kind, lines) in &kind_pushes {
            oracle.push_kind(*kind, lines.clone());
        }
        oracle.push_tool(tool_compact, tool_out);
        oracle.set_width(W);

        assert_eq!(text(incremental.lines()), text(oracle.lines()));
    }

    #[test]
    fn sticky_user_line_stable_after_many_appends() {
        let mut t = Transcript::new();
        t.set_width(80);
        for i in 0..25 {
            t.push_user(vec![Line::from(format!("user turn {i}"))]);
            t.push_assistant(vec![Line::from(format!("assistant reply {i}"))]);
        }
        let total = t.len();
        assert!(total > 10);
        // Viewport starts past the first few blocks.
        let view_start = total / 3;
        assert!(t.has_sticky_user(view_start));
        let sticky = t
            .sticky_user_line(view_start)
            .expect("sticky above mid viewport");
        let plain: String = sticky.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            plain.starts_with("user turn "),
            "sticky should be a user line, got {plain:?}"
        );

        // More appends (with scroll offset so history stays put) must not break maps.
        t.scroll_up(4);
        for i in 25..40 {
            t.push_user(vec![Line::from(format!("user turn {i}"))]);
            t.push_assistant(vec![Line::from(format!("assistant reply {i}"))]);
        }
        assert!(t.has_sticky_user(view_start));
        let sticky2 = t
            .sticky_user_line(view_start)
            .expect("sticky after appends");
        assert_eq!(text(&[sticky]), text(&[sticky2]));
        // Absolute line maps for earlier history stay valid after appends.
        let bi_at = t.line_block[view_start];
        assert!(bi_at < t.block_count());
        assert!(t.line_at(view_start).is_some());
        let n = t.len();
        assert!(t.line_at(n).is_none());
    }

    #[test]
    fn selection_index_stable_after_many_appends() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push_user(vec![Line::from("select me")]);
        t.push_assistant(vec![Line::from("first answer")]);
        t.selected = Some(0);
        assert_eq!(t.selected(), Some(0));
        assert!(t.is_selected_block_for_line(0));

        for i in 0..30 {
            t.push_user(vec![Line::from(format!("u{i}"))]);
            t.push_assistant(vec![Line::from(format!("a{i}"))]);
        }
        assert_eq!(t.selected(), Some(0));
        assert!(t.is_selected_block_for_line(0));
        // Last history line belongs to a later block.
        let last = t.len() - 1;
        assert!(!t.is_selected_block_for_line(last));
        assert_eq!(t.line_block[last], t.block_count() - 1);
    }

    #[test]
    fn fill_viewport_clones_range_and_styles_block_selection() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push_user(vec![Line::from("hello")]);
        t.push_assistant(vec![Line::from("world")]);
        let n = t.len();
        assert!(n >= 2);

        let mut out = Vec::new();
        t.fill_viewport(&mut out, 0, n, None);
        assert_eq!(out.len(), n);
        assert_eq!(text(&out), text(t.lines()));

        t.selected = Some(0);
        let mut marked = Vec::new();
        t.fill_viewport(&mut marked, 0, n, Some(0));
        // User block lines receive the block-selection surface without
        // shifting text geometry.
        for (idx, line) in marked.iter().enumerate() {
            let selected_surface = line.style.bg == Some(theme::BORDER());
            assert_eq!(selected_surface, t.is_selected_block_for_line(idx));
            assert_eq!(selected_surface, t.line_block[idx] == 0);
        }
    }

    #[test]
    fn user_and_assistant_blocks_have_distinct_gutters_and_surfaces() {
        let mut t = Transcript::new();
        t.set_width(24);
        t.push_user(vec![Line::from("hello")]);
        t.push_assistant(vec![Line::from("world")]);
        let rendered = text(t.lines());
        assert!(rendered[0].starts_with("❯ hello"));
        assert!(rendered[2].starts_with("│ world"));
        assert_eq!(t.lines()[0].style.bg, Some(theme::USER_BG()));
        assert_ne!(t.lines()[2].style.bg, Some(theme::USER_BG()));
    }

    #[test]
    fn mouse_selection_uses_character_offsets_for_unicode() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push_user(vec![Line::from("héllo world")]);
        assert!(t.begin_text_selection_at(0, 2));
        assert!(t.update_text_selection_at(0, 7));
        t.finish_text_selection();
        assert_eq!(t.selected_text().as_deref(), Some("héllo"));
        assert_eq!(t.selection_columns(0), Some((2, 7)));
    }

    #[test]
    fn text_selection_survives_rewrap_and_crosses_blocks() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push_user(vec![Line::from("alpha beta")]);
        t.push_assistant(vec![Line::from("gamma delta")]);
        assert!(t.begin_text_selection_at(0, 8));
        assert!(t.update_text_selection_at(2, 7));
        t.finish_text_selection();
        assert_eq!(t.selected_text().as_deref(), Some("beta\n\ngamma"));
        t.set_width(10);
        assert_eq!(t.selected_text().as_deref(), Some("beta\n\ngamma"));
        assert!(t.has_text_selection());
    }

    #[test]
    fn reverse_drag_normalizes_text_and_block_removal_clears_range() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push_user(vec![Line::from("alpha")]);
        t.push_assistant(vec![Line::from("beta")]);
        t.push_user(vec![Line::from("gamma")]);
        t.push_assistant(vec![Line::from("delta")]);
        assert!(t.begin_text_selection_at(6, 7));
        assert!(t.update_text_selection_at(0, 2));
        t.finish_text_selection();
        assert_eq!(
            t.selected_text().as_deref(),
            Some("alpha\n\nbeta\n\ngamma\n\ndelta")
        );
        assert!(t.pop_last_user());
        assert!(!t.has_text_selection());
    }

    #[test]
    fn folding_a_selected_tool_clears_text_selection() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_tool(vec![Line::from("✓ Read file")], "line one\nline two".into());
        assert!(t.begin_text_selection_at(0, 2));
        assert!(t.update_text_selection_at(0, 5));
        t.finish_text_selection();
        assert!(t.has_text_selection());
        t.selected = Some(0);
        assert!(t.toggle_fold_selected());
        assert!(!t.has_text_selection());
    }

    #[test]
    fn latest_assistant_text_ignores_newer_system_and_tool_blocks() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(vec![Line::from("first")]);
        t.push_assistant(vec![Line::from("latest"), Line::from("response")]);
        t.push_tool(vec![Line::from("✓ Shell")], "done".into());
        t.push(vec![Line::from("notice")]);
        assert_eq!(t.last_assistant_text().as_deref(), Some("latest\nresponse"));
    }
}
