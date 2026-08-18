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
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::theme;

/// Frames go through one large buffer per flush: bare `Stdout` is
/// line-buffered at 1 KiB, which turns a busy streaming frame into dozens of
/// write(2) calls. `FrameWriter` also keeps a mid-panic partial frame from
/// flushing over the restored shell.
pub type Term = Terminal<CrosstermBackend<crate::FrameWriter<std::io::Stdout>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    User,
    Assistant,
    Tool,
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
    /// Lowercase plain text for find, built once at push: the find bar
    /// filters on every keystroke and previews on every list rebuild, and
    /// rebuilding + lowercasing every block's text made both O(session)
    /// allocation per key. Fold-independent (it reads compact/full_output/
    /// raw, never the folded view), so no invalidation path exists.
    search_lower: String,
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
    /// Set by a word or line gesture, which picks a real range even when it is
    /// one character wide. Endpoints are inclusive, so a drag that has not
    /// moved yet is indistinguishable from a one-character selection; this is
    /// what tells them apart, and it keeps a plain click from selecting.
    explicit: bool,
    /// The inclusive range the press picked: the word or line under a
    /// gesture, the single character under a plain press. A drag extends
    /// from it but never shrinks the selection below it, so a gesture's own
    /// release (and a drag that wanders back inside the range) keeps the
    /// word or line the user was shown.
    origin: (TextPoint, TextPoint),
}

impl TextSelection {
    /// Whether this covers any text at all.
    fn is_empty(&self) -> bool {
        !self.explicit && self.anchor == self.head
    }
}

/// The one place a block's search text is lowered, so the test oracle can
/// count builds: pushing a block builds exactly one, and no find keystroke
/// may build any.
fn lower_for_search(text: &str) -> String {
    #[cfg(test)]
    SEARCH_BUILDS.with(|counter| counter.set(counter.get() + 1));
    text.to_lowercase()
}

#[cfg(test)]
thread_local! {
    static SEARCH_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl Block {
    fn new(kind: BlockKind, raw: Vec<Line<'static>>) -> Self {
        let selectable = lines_to_plain(&raw);
        let selectable_chars = selectable.chars().count();
        let search_lower = lower_for_search(&selectable);
        Self {
            kind,
            raw,
            compact: None,
            folded: false,
            full_output: None,
            search_lower,
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
        // Search covers the compact header plus the whole output, matching
        // what `search_text_line` reads back out; the folded view is not
        // part of it.
        let search_lower = if selectable.is_empty() {
            lower_for_search(&full_output)
        } else {
            lower_for_search(&format!("{selectable}\n{full_output}"))
        };
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
            search_lower,
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
            BlockKind::User => 2,
            BlockKind::Assistant | BlockKind::Tool | BlockKind::System => 0,
        };
        let content_width = width.saturating_sub(gutter).max(8);
        let (wrapped, maps) = wrap_lines_mapped(self.source_lines(), content_width);
        self.cache.clear();
        self.cache_maps.clear();
        for (i, (line, map)) in wrapped.into_iter().zip(maps).enumerate() {
            let (line, x_offset) = decorate_line(self.kind, line, i == 0, width);
            self.cache.push(line);
            self.cache_maps.push(map.map(|mut map| {
                // The wrapper may already have shifted continuation rows by a
                // hanging indent; the gutter stacks on top of that.
                map.x_offset += x_offset;
                map
            }));
        }
        self.cache.push(Line::default());
        self.cache_maps.push(None);
    }

    fn invalidate(&mut self) {
        self.cache_width = 0;
        self.cache.clear();
        self.cache_maps.clear();
    }
}

/// A content position in the wrapped transcript that survives re-wraps.
/// See [`Transcript::anchor_at`].
#[derive(Clone, Copy, Debug)]
pub struct WrapAnchor {
    block: usize,
    lines_into_block: usize,
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
    /// Full re-wraps performed (every block invalidated by a width change).
    /// Oracle for the draw path's promise that steady-state frames never
    /// re-wrap history.
    #[cfg(test)]
    pub(crate) rewraps: u64,
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
                    explicit: selection.explicit,
                    origin: (retreat(selection.origin.0), retreat(selection.origin.1)),
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
            #[cfg(test)]
            {
                self.rewraps += 1;
            }
            self.width = width;
            for b in &mut self.blocks {
                b.invalidate();
            }
            self.dirty = true;
            self.ensure_flat();
            self.offset = self.offset.min(self.wrapped.len());
        }
    }

    /// Capture the content at `lines_from_bottom` so it can be found again
    /// after a re-wrap. Wrapped-line indices are meaningless across widths;
    /// the block plus the distance into it survives, approximate across
    /// widths but bounded by one block's height. None when history is empty
    /// or the position is not above the bottom.
    pub fn anchor_at(&mut self, lines_from_bottom: usize) -> Option<WrapAnchor> {
        self.ensure_flat();
        if lines_from_bottom == 0 || self.wrapped.is_empty() {
            return None;
        }
        let index = self
            .wrapped
            .len()
            .saturating_sub(lines_from_bottom)
            .min(self.wrapped.len() - 1);
        let block = *self.line_block.get(index)?;
        Some(WrapAnchor {
            block,
            lines_into_block: index - self.block_starts[block],
        })
    }

    /// Wrapped-line distance from the bottom of history to the anchored
    /// content, after any re-wraps since capture.
    pub fn resolve_anchor(&mut self, anchor: WrapAnchor) -> usize {
        self.ensure_flat();
        let Some(&start) = self.block_starts.get(anchor.block) else {
            return 0;
        };
        let end = self
            .block_starts
            .get(anchor.block + 1)
            .copied()
            .unwrap_or(self.wrapped.len());
        let index = (start + anchor.lines_into_block).min(end.saturating_sub(1));
        self.wrapped.len() - index
    }

    /// Place the view at an absolute distance from the bottom; the draw
    /// path clamps it against the current total as usual.
    pub fn set_offset(&mut self, lines_from_bottom: usize) {
        self.offset = lines_from_bottom;
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

    /// Shift a scrolled-up offset by the live-tail line delta so the visible
    /// content stays stationary. The offset is measured from the bottom, so
    /// a growing tail would otherwise drag the view forward line by line
    /// while the reader is anchored on older content, and a collapsing tail
    /// (cancel, message done) would fling the view to the top once the stale
    /// offset hits the clamp. No-op while following.
    pub fn compensate_tail_delta(&mut self, delta: isize) {
        if self.offset == 0 || delta == 0 {
            return;
        }
        if delta > 0 {
            self.offset = self.offset.saturating_add(delta as usize);
        } else {
            self.offset = self.offset.saturating_sub(delta.unsigned_abs());
        }
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
            // Tool output is already the exact bytes; never transform it.
            return Some(out.clone());
        }
        Some(copy_text_without_chrome(&b.raw))
    }

    /// Indices of blocks whose search text contains `query`,
    /// case-insensitive; an empty query matches every block. Runs against
    /// the per-block cache: one lowercase of the query, no per-block
    /// allocation, so a find keystroke costs a scan, not a session rebuild.
    pub fn filter_matches(&self, query: &str) -> Vec<usize> {
        if query.is_empty() {
            return (0..self.blocks.len()).collect();
        }
        let q = query.to_lowercase();
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.search_lower.contains(&q))
            .map(|(i, _)| i)
            .collect()
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

    /// One-line preview for find UI. Prefer a line containing `query` when
    /// set. The match is located in the cached lowercase text and only the
    /// one line it names is read back in original case: `to_lowercase`
    /// never maps a char to or from `\n`, so line indices agree between the
    /// cache and the original.
    pub fn block_preview(&self, i: usize, query: &str) -> Option<String> {
        let b = self.blocks.get(i)?;
        let q = query.trim();
        if !q.is_empty() {
            let q_low = q.to_lowercase();
            if let Some(pos) = b.search_lower.find(&q_low) {
                let line_idx =
                    b.search_lower[..pos].bytes().filter(|&c| c == b'\n').count();
                if let Some(line) = search_text_line(b, line_idx) {
                    return Some(line.trim().to_string());
                }
            }
        }
        let first_content = b
            .search_lower
            .split('\n')
            .position(|l| !l.trim().is_empty());
        let line = first_content
            .and_then(|idx| search_text_line(b, idx))
            .map(|l| l.trim().to_string())
            .unwrap_or_default();
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
            .is_some_and(|selection| !selection.is_empty())
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
            explicit: false,
            origin: (point, point),
        });
        true
    }

    /// Select the word under a double-click.
    pub fn select_word_at(&mut self, line_idx: usize, x: usize) -> bool {
        self.select_range_at(line_idx, x, crate::ui::text::word_bounds)
    }

    /// Select the logical line under a triple-click, even where it wraps.
    pub fn select_line_at(&mut self, line_idx: usize, x: usize) -> bool {
        self.select_range_at(line_idx, x, crate::ui::text::line_bounds)
    }

    fn select_range_at(
        &mut self,
        line_idx: usize,
        x: usize,
        bounds: fn(&str, usize) -> (usize, usize),
    ) -> bool {
        // A press replaces whatever was selected, including with nothing. An
        // empty line has no word and no line to take, and leaving the old
        // highlight standing would make it lie about what a copy carries.
        self.text_selection = None;
        let Some(point) = self.hit_test(line_idx, x) else {
            return false;
        };
        // Fall back to what a plain press does, so the gesture still places an
        // anchor a drag can extend from.
        self.text_selection = Some(TextSelection {
            anchor: point,
            head: point,
            dragging: true,
            explicit: false,
            origin: (point, point),
        });
        let Some(block) = self.blocks.get(point.block) else {
            return false;
        };
        let (start, end) = bounds(&block.selectable, point.offset);
        if start >= end {
            return false;
        }
        let lo = TextPoint { block: point.block, offset: start };
        // Endpoints are inclusive, matching what a copy carries.
        let hi = TextPoint { block: point.block, offset: end - 1 };
        self.text_selection = Some(TextSelection {
            anchor: lo,
            head: hi,
            dragging: true,
            explicit: true,
            origin: (lo, hi),
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
        // The selection grows from the origin range and never shrinks below
        // it: past either edge the far edge anchors and the pointer leads;
        // inside it the selection is exactly the range the press picked.
        let (lo, hi) = selection.origin;
        if point > hi {
            selection.anchor = lo;
            selection.head = point;
        } else if point < lo {
            selection.anchor = hi;
            selection.head = point;
        } else {
            selection.anchor = lo;
            selection.head = hi;
        }
        true
    }

    /// The button came up at `(line_idx, x)`: the same extension as a drag
    /// (release-inclusive, so a release whose motion the terminal coalesced
    /// away still reaches its cell), then the drag ends. The origin clamp in
    /// [`Self::update_text_selection_at`] is what keeps a gesture's own
    /// release, which lands back inside the picked range, from shrinking the
    /// word or line to a fragment.
    pub fn end_text_selection_at(&mut self, line_idx: usize, x: usize) {
        self.update_text_selection_at(line_idx, x);
        self.finish_text_selection();
    }

    pub fn finish_text_selection(&mut self) {
        if let Some(selection) = &mut self.text_selection {
            selection.dragging = false;
            // A click that never moved is not a selection; a word or line
            // gesture is one even at a single character.
            if selection.is_empty() {
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
        if selection.is_empty() {
            return None;
        }
        let (start, end) = normalized_selection(selection);
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
            // Inclusive of the release cell, matching selected_text: the
            // highlight must cover exactly what a copy would carry.
            end.offset.saturating_add(1).min(block_len)
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
        if selection.is_empty() {
            return None;
        }
        let (start, end) = normalized_selection(selection);
        if start.block >= self.blocks.len() || end.block >= self.blocks.len() {
            return None;
        }
        let mut parts = Vec::new();
        for block_index in start.block..=end.block {
            let block = &self.blocks[block_index];
            let char_len = block.selectable_chars;
            let from = if block_index == start.block {
                start.offset.min(char_len)
            } else {
                0
            };
            let to = if block_index == end.block {
                // Inclusive of the character under the release cell: a drag
                // that ends ON the closing brace must copy the brace.
                end.offset.saturating_add(1).min(char_len)
            } else {
                char_len
            };
            parts.push(slice_block_chars(block, from, to));
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

}

fn line_is_blank(l: &Line<'_>) -> bool {
    l.spans.iter().all(|s| s.content.trim().is_empty())
}

/// A renderer-added fence gutter is exactly a line whose FIRST span is the
/// two-character "│ " glyph followed by the code span (markdown::render_line
/// builds fence lines that way). Literal content that merely begins with the
/// same characters arrives as a single span and is never treated as chrome,
/// so the clipboard keeps it.
fn is_rail_line(line: &Line<'static>) -> bool {
    line.spans.len() >= 2 && line.spans[0].content.as_ref() == "│ "
}

/// Character slice of a block's selectable text, identical to slicing the
/// joined text except that the two gutter characters at the start of a fence
/// line are dropped when the slice covers them. The clipboard carries the
/// code, never the chrome.
fn slice_block_chars(block: &Block, from: usize, to: usize) -> String {
    let rails: Vec<bool> = block.source_lines().iter().map(is_rail_line).collect();
    let mut out = String::new();
    let mut pos = 0usize;
    for (i, line) in block.selectable.split('\n').enumerate() {
        if i > 0 {
            // The separating newline occupies one character position.
            if pos >= from && pos < to {
                out.push('\n');
            }
            pos += 1;
        }
        let rail = rails.get(i).copied().unwrap_or(false);
        let mut count = 0usize;
        for (j, ch) in line.chars().enumerate() {
            count += 1;
            let p = pos + j;
            if p >= from && p < to && !(rail && j < 2) {
                out.push(ch);
            }
        }
        pos += count;
    }
    out
}

/// Full-content copy with renderer chrome (the fence gutter span) excluded.
/// The clipboard must carry exact bytes: pasting code with a rail character
/// on every line makes the paste useless.
fn copy_text_without_chrome(lines: &[Line<'static>]) -> String {
    lines
        .iter()
        .map(|line| {
            let skip = usize::from(is_rail_line(line));
            line.spans
                .iter()
                .skip(skip)
                .map(|span| span.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
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

/// Line `line_idx` of a block's search text in original case, without
/// rebuilding the whole text: the compact header is at most a couple of
/// lines and the full output is read by reference. Composed exactly like
/// the lowercase cache in `Block::new` / `Block::tool`, which is what
/// keeps line indices between the two in agreement.
fn search_text_line(b: &Block, line_idx: usize) -> Option<String> {
    // `split('\n')`, not `.lines()`: indices come from counting `\n` bytes
    // in the cache, and `.lines()` would drop a trailing empty slot.
    match &b.full_output {
        Some(out) => {
            let header = b
                .compact
                .as_ref()
                .map(|c| lines_to_plain(c))
                .unwrap_or_default();
            if header.is_empty() {
                return out.split('\n').nth(line_idx).map(str::to_string);
            }
            // The composed text is `header\nout`, so the header occupies
            // its `\n` count plus one line slots before `out` begins.
            let header_slots = header.matches('\n').count() + 1;
            if line_idx < header_slots {
                header.split('\n').nth(line_idx).map(str::to_string)
            } else {
                out.split('\n').nth(line_idx - header_slots).map(str::to_string)
            }
        }
        None => lines_to_plain(&b.raw)
            .split('\n')
            .nth(line_idx)
            .map(str::to_string),
    }
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
        let indent = hanging_indent(&chars, width);
        // A fence line's continuation keeps the rail glyph itself, in the
        // rail's own style, so a wrapped code block still reads as one block
        // at narrow widths instead of dissolving into prose. Rail detection
        // is structural (the renderer's dedicated gutter span), the same
        // rule the copy paths use: literal content that merely begins with
        // the rail characters arrives as a single span and gets the plain
        // space indent like any other line. The rail is non-ASCII, so only
        // the Unicode path below ever sees one.
        let rail = is_rail_line(line);
        if chars.iter().all(|(ch, _)| ch.is_ascii()) {
            // Coding-agent transcripts are overwhelmingly ASCII. Preserve the
            // original scalar loop here so Unicode safety has no tax on the
            // common resize and streaming paths.
            let mut start = 0usize;
            let mut first_row = true;
            while start < chars.len() {
                let avail = if first_row { width } else { width - indent };
                let mut used = 0usize;
                let mut end = start;
                let mut last_space: Option<usize> = None;
                while end < chars.len() {
                    let width_here = chars[end].0.width().unwrap_or(0);
                    if used + width_here > avail {
                        break;
                    }
                    if chars[end].0 == ' ' {
                        last_space = Some(end);
                    }
                    used += width_here;
                    end += 1;
                }
                let cut = if end == chars.len() {
                    end
                } else {
                    match last_space {
                        Some(space) if space > start => space + 1,
                        _ => end.max(start + 1),
                    }
                };
                let mut row = rebuild(&chars[start..cut]);
                if !first_row && indent > 0 {
                    row.spans.insert(0, Span::raw(" ".repeat(indent)));
                }
                out.push(row);
                maps.push(Some(CachedLineMap {
                    start: logical_start + start,
                    end: logical_start + cut,
                    x_offset: if first_row { 0 } else { indent },
                    text: chars[start..cut].iter().map(|(ch, _)| ch).collect(),
                }));
                start = cut;
                first_row = false;
            }
        } else {
            let plain: String = chars.iter().map(|(ch, _)| ch).collect();
            let mut graphemes = Vec::new();
            let mut char_start = 0usize;
            for grapheme in plain.graphemes(true) {
                let char_end = char_start + grapheme.chars().count();
                graphemes.push((
                    char_start,
                    grapheme.width(),
                    grapheme.chars().all(char::is_whitespace),
                ));
                char_start = char_end;
            }

            let mut start = 0usize;
            let mut first_row = true;
            while start < graphemes.len() {
                let avail = if first_row { width } else { width - indent };
                let mut used = 0usize;
                let mut end = start;
                let mut last_space: Option<usize> = None;
                while end < graphemes.len() {
                    let (_, width_here, is_space) = graphemes[end];
                    if used + width_here > avail {
                        break;
                    }
                    if is_space {
                        last_space = Some(end);
                    }
                    used += width_here;
                    end += 1;
                }
                let cut = if end == graphemes.len() {
                    end
                } else {
                    match last_space {
                        Some(space) if space > start => space + 1,
                        _ => end.max(start + 1),
                    }
                };
                let from_char = graphemes[start].0;
                let to_char = if cut == graphemes.len() {
                    chars.len()
                } else {
                    graphemes[cut].0
                };
                let mut row = rebuild(&chars[from_char..to_char]);
                if !first_row && indent > 0 {
                    if rail && indent == 2 {
                        row.spans.insert(0, Span::styled("│ ", line.spans[0].style));
                    } else {
                        row.spans.insert(0, Span::raw(" ".repeat(indent)));
                    }
                }
                out.push(row);
                maps.push(Some(CachedLineMap {
                    start: logical_start + from_char,
                    end: logical_start + to_char,
                    x_offset: if first_row { 0 } else { indent },
                    text: chars[from_char..to_char]
                        .iter()
                        .map(|(ch, _)| ch)
                        .collect(),
                }));
                start = cut;
                first_row = false;
            }
        }
        logical_start += chars.len() + 1;
    }
    (out, maps)
}

/// Continuation indent for a wrapped line: its leading whitespace plus any
/// list marker. Without it the second row of a long bullet starts in the
/// marker column and reads as a separate item.
fn hanging_indent(chars: &[(char, Style)], width: usize) -> usize {
    let mut i = 0;
    while i < chars.len() && chars[i].0 == ' ' {
        i += 1;
    }
    let marker = match chars.get(i).map(|c| c.0) {
        Some('•' | '-' | '*') if chars.get(i + 1).map(|c| c.0) == Some(' ') => 2,
        // The code-fence rail: continuations stay inside the fence column.
        Some('│') if chars.get(i + 1).map(|c| c.0) == Some(' ') => 2,
        Some(d) if d.is_ascii_digit() => {
            let mut j = i;
            while j < chars.len() && chars[j].0.is_ascii_digit() {
                j += 1;
            }
            match (chars.get(j).map(|c| c.0), chars.get(j + 1).map(|c| c.0)) {
                (Some('.' | ')'), Some(' ')) => j + 2 - i,
                _ => 0,
            }
        }
        _ => 0,
    };
    let indent = i + marker;
    // An indent that swallows half the line would wrap worse than none.
    if indent >= width / 2 {
        0
    } else {
        indent
    }
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
        BlockKind::Assistant => (line, 0),
        BlockKind::Tool => (surface_line(line, width, theme::SURFACE()), 0),
        BlockKind::System => (line, 0),
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
    let mut chars = 0usize;
    for grapheme in text.graphemes(true) {
        let width = grapheme.width();
        if used + width > column {
            return chars;
        }
        used += width;
        chars += grapheme.chars().count();
    }
    chars
}

fn char_offset_to_display_column(text: &str, offset: usize) -> usize {
    let byte = text
        .char_indices()
        .nth(offset)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len());
    text[..byte].width()
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

    #[test]
    fn anchor_finds_the_same_block_after_a_rewrap() {
        let mut t = Transcript::new();
        t.set_width(30);
        for i in 0..20 {
            t.push_user(vec![Line::from(format!(
                "block {i:02} body long enough to wrap at thirty columns"
            ))]);
        }
        let anchor = t.anchor_at(10).unwrap();
        assert_eq!(t.resolve_anchor(anchor), 10);
        let block = t.line_block[t.wrapped.len() - 10];

        // Wider wrap: every block collapses to fewer lines, all indices
        // shift, but the anchor still resolves into the same block.
        t.set_width(120);
        let from_bottom = t.resolve_anchor(anchor);
        assert!(from_bottom >= 1);
        let index = t.wrapped.len() - from_bottom;
        assert_eq!(t.line_block[index], block);

        // And back: still the same block after a narrow re-wrap.
        t.set_width(24);
        let index = t.wrapped.len() - t.resolve_anchor(anchor);
        assert_eq!(t.line_block[index], block);
    }

    #[test]
    fn literal_rail_content_wraps_without_fabricated_fence_chrome() {
        let mut t = Transcript::new();
        t.set_width(24);
        // A single-span line that merely begins with the rail characters is
        // content: its continuations get the plain space indent, never an
        // injected rail that would present prose as fenced code.
        t.push_assistant(vec![Line::from(
            "│ a literal border line long enough to wrap at this width",
        )]);
        let rows = text(t.lines());
        let wrapped: Vec<&String> = rows.iter().filter(|r| !r.trim().is_empty()).collect();
        assert!(wrapped.len() >= 2, "expected a wrap: {rows:?}");
        assert!(wrapped[0].starts_with("│ "));
        for row in &wrapped[1..] {
            assert!(
                !row.trim_start().starts_with('│'),
                "fabricated rail on literal content: {row:?}"
            );
        }
    }

    #[test]
    fn wrapped_code_lines_keep_the_fence_rail() {
        let mut t = Transcript::new();
        t.set_width(30);
        t.push_assistant(crate::ui::markdown::render(
            "```rust\nlet value = a_deliberately_long_identifier_that_wraps + another_long_identifier;\n```",
        ));
        let rows = text(t.lines());
        let code_rows: Vec<&String> = rows
            .iter()
            .filter(|row| row.contains("identifier") || row.contains("let value"))
            .collect();
        assert!(code_rows.len() >= 2, "expected a wrapped fence: {rows:?}");
        for row in code_rows {
            assert!(
                row.starts_with("│ "),
                "a fence row escaped the rail: {row:?}"
            );
        }
    }

    #[test]
    fn literal_rail_prefixed_content_is_not_chrome() {
        let mut t = Transcript::new();
        t.set_width(60);
        // A single-span line that merely begins with the same characters as
        // the fence gutter is content, not renderer chrome; the clipboard
        // must keep it intact on both copy paths.
        t.push_assistant(vec![Line::from("│ literal table border")]);
        t.select_prev();
        assert_eq!(
            t.selected_copy_text().as_deref(),
            Some("│ literal table border")
        );

        t.ensure_flat();
        assert!(t.begin_text_selection_at(0, 0));
        assert!(t.update_text_selection_at(0, 21));
        t.finish_text_selection();
        assert_eq!(t.selected_text().as_deref(), Some("│ literal table border"));
    }

    #[test]
    fn copied_block_carries_no_fence_rails() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(crate::ui::markdown::render(
            "Intro.\n\n```rust\nfn a() -> u32 { 1 }\nfn b() -> u32 { 2 }\n```\n\nOutro.",
        ));
        t.select_prev();
        let copied = t.selected_copy_text().unwrap();
        assert!(copied.contains("fn a() -> u32 { 1 }"), "{copied}");
        assert!(copied.contains("Outro."));
        assert!(!copied.contains('│'), "rail leaked into clipboard: {copied}");
    }

    #[test]
    fn mouse_selection_is_release_inclusive_and_rail_free() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(crate::ui::markdown::render("```rust\nfn a() -> u32 { 1 }\n```"));
        t.ensure_flat();
        let li = text(&t.wrapped)
            .iter()
            .position(|l| l.contains("fn a"))
            .unwrap();

        // Drag from the first code character to the closing brace: the
        // character under the release cell is part of the selection.
        assert!(t.begin_text_selection_at(li, 2));
        assert!(t.update_text_selection_at(li, 20));
        t.finish_text_selection();
        assert_eq!(t.selected_text().unwrap(), "fn a() -> u32 { 1 }");

        // Drag from column zero: the fence rail is display chrome and must
        // not reach the clipboard.
        assert!(t.begin_text_selection_at(li, 0));
        assert!(t.update_text_selection_at(li, 20));
        t.finish_text_selection();
        assert_eq!(t.selected_text().unwrap(), "fn a() -> u32 { 1 }");
    }

    /// A terminal double- or triple-click is Down, Up, Down, Up: the second
    /// Down picks the word or line, and the second Up lands back on the
    /// pressed cell. That release must not drag the head to the click point,
    /// or the copy carries a fragment of what the highlight promised.
    #[test]
    fn a_gesture_survives_its_own_release() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(vec![Line::from("alpha beta gamma")]);
        t.ensure_flat();

        // Double-click on the middle of "beta", release on the same cell.
        assert!(t.select_word_at(0, 7));
        t.end_text_selection_at(0, 7);
        assert_eq!(t.selected_text().unwrap(), "beta");

        // Triple-click mid-line, release on the same cell.
        assert!(t.select_line_at(0, 7));
        t.end_text_selection_at(0, 7);
        assert_eq!(t.selected_text().unwrap(), "alpha beta gamma");
    }

    /// A gesture the user then actually drags is a drag: the release cell
    /// joins the selection, exactly like any other drag.
    #[test]
    fn a_dragged_gesture_still_takes_the_release_cell() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(vec![Line::from("alpha beta gamma")]);
        t.ensure_flat();

        assert!(t.select_word_at(0, 7));
        assert!(t.update_text_selection_at(0, 12));
        t.end_text_selection_at(0, 12);
        // Endpoints are inclusive: the release cell's character comes along.
        assert_eq!(t.selected_text().unwrap(), "beta ga");
    }

    /// A gesture pressed on one cell and released on another is motion even
    /// when the terminal coalesced away every Drag event in between: the
    /// release cell joins the selection exactly as if the drag had been
    /// delivered.
    #[test]
    fn a_coalesced_gesture_drag_still_reaches_the_release_cell() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(vec![Line::from("alpha beta gamma")]);
        t.ensure_flat();

        // Word gesture on "beta", release over "gamma" with no Drag events.
        assert!(t.select_word_at(0, 7));
        t.end_text_selection_at(0, 12);
        assert_eq!(t.selected_text().unwrap(), "beta ga");
    }

    /// A drag that wanders away and comes back to the pressed cell releases
    /// into the word the gesture picked, not a fragment of it.
    #[test]
    fn a_gesture_drag_returning_to_the_press_cell_keeps_the_word() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(vec![Line::from("alpha beta gamma")]);
        t.ensure_flat();

        assert!(t.select_word_at(0, 7));
        assert!(t.update_text_selection_at(0, 12));
        assert!(t.update_text_selection_at(0, 7));
        t.end_text_selection_at(0, 7);
        assert_eq!(t.selected_text().unwrap(), "beta");
    }

    /// A plain press whose motion the terminal coalesced away still selects
    /// press-to-release: with no gesture in play the release cell always
    /// carries the head.
    #[test]
    fn a_plain_press_selects_to_the_release_cell_without_drag_events() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push_assistant(vec![Line::from("alpha beta gamma")]);
        t.ensure_flat();

        assert!(t.begin_text_selection_at(0, 0));
        t.end_text_selection_at(0, 9);
        assert_eq!(t.selected_text().unwrap(), "alpha beta");
    }

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
    fn wrapped_list_items_hang_under_their_marker() {
        let wrapped = text(&wrap_lines(&[Line::from("• alpha beta gamma delta")], 16));
        assert_eq!(wrapped[0], "• alpha beta ");
        for row in &wrapped[1..] {
            assert!(row.starts_with("  "), "continuation not indented: {row:?}");
            assert!(row.chars().count() <= 16, "line too long: {row:?}");
        }
        // Dropping the injected indent recovers the original text exactly.
        let rejoined: String = std::iter::once(wrapped[0].clone())
            .chain(wrapped[1..].iter().map(|r| r[2..].to_string()))
            .collect();
        assert_eq!(rejoined, "• alpha beta gamma delta");

        let numbered = text(&wrap_lines(&[Line::from("12. alpha beta gamma delta")], 16));
        assert!(numbered[1].starts_with("    "), "{:?}", numbered[1]);

        // Plain prose keeps flush continuations.
        let prose = text(&wrap_lines(&[Line::from("alpha beta gamma delta")], 12));
        assert!(!prose[1].starts_with(' '), "{:?}", prose[1]);
    }

    #[test]
    fn hanging_indent_applies_to_the_unicode_path_too() {
        let wrapped = text(&wrap_lines(&[Line::from("• héllo wörld again now")], 14));
        assert!(wrapped.len() > 1);
        for row in &wrapped[1..] {
            assert!(row.starts_with("  "), "continuation not indented: {row:?}");
        }
    }

    /// The indent shifts continuation rows on screen, so the column map that
    /// drives mouse selection has to shift with them. Without this, clicking
    /// the start of the indented row selects two characters too far in.
    #[test]
    fn mouse_selection_accounts_for_the_hanging_indent() {
        let mut t = Transcript::new();
        t.set_width(20);
        t.push(vec![Line::from("• alpha beta gamma delta")]);
        let rendered = text(t.lines());
        let row = rendered
            .iter()
            .position(|r| r.trim_start() == "delta")
            .expect("wrapped continuation row");
        assert_eq!(rendered[row], "  delta");

        assert!(t.begin_text_selection_at(row, 2));
        assert!(t.update_text_selection_at(row, 7));
        t.finish_text_selection();
        assert_eq!(t.selected_text().as_deref(), Some("delta"));
        assert_eq!(t.selection_columns(row), Some((2, 7)));
    }

    /// Double-click in the transcript is how a developer lifts a path out of
    /// tool output; it has to come back whole.
    #[test]
    fn double_click_selects_the_whole_token() {
        let mut t = Transcript::new();
        t.set_width(60);
        t.push(vec![Line::from("edited crates/tui/src/app.rs:42 ok")]);
        let row = 0;
        let col = "edited cr".len();

        assert!(t.select_word_at(row, col));
        t.finish_text_selection();
        assert_eq!(
            t.selected_text().as_deref(),
            Some("crates/tui/src/app.rs:42"),
        );
        // The highlight covers exactly those columns.
        let (from, to) = t.selection_columns(row).expect("highlight");
        assert_eq!(to - from, "crates/tui/src/app.rs:42".chars().count());
    }

    /// A one-character word survives the release-cell-inclusive endpoint,
    /// which on its own cannot tell it from a click that never moved.
    #[test]
    fn double_click_selects_a_one_character_word() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push(vec![Line::from("let x = 1")]);
        assert!(t.select_word_at(0, 4));
        t.finish_text_selection();
        assert!(t.has_text_selection());
        assert_eq!(t.selected_text().as_deref(), Some("x"));
    }

    /// Triple-click means the logical line, even where it wraps on screen.
    #[test]
    fn triple_click_selects_the_logical_line_across_wraps() {
        let mut t = Transcript::new();
        t.set_width(16);
        t.push(vec![
            Line::from("first line here"),
            Line::from("second line that wraps across rows"),
        ]);
        let rendered = text(t.lines());
        let row = rendered
            .iter()
            .position(|r| r.contains("across"))
            .expect("a continuation row of the second line");
        assert!(row > 1, "the second line must actually wrap");

        assert!(t.select_line_at(row, 2));
        t.finish_text_selection();
        assert_eq!(
            t.selected_text().as_deref(),
            Some("second line that wraps across rows"),
        );
    }

    /// Same rule in the transcript: a gesture with nothing to take clears.
    #[test]
    fn a_gesture_on_a_blank_line_clears_the_previous_selection() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push(vec![
            Line::from("alpha beta"),
            Line::from(""),
            Line::from("gamma"),
        ]);
        assert!(t.select_word_at(0, 0));
        t.finish_text_selection();
        assert_eq!(t.selected_text().as_deref(), Some("alpha"));

        assert!(!t.select_line_at(1, 0));
        assert_eq!(t.selected_text(), None, "stale selection after a blank line");
        assert!(!t.has_text_selection());
    }

    /// A plain click still selects nothing, or every click would copy.
    #[test]
    fn a_click_that_never_moved_is_not_a_selection() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push(vec![Line::from("hello there")]);
        assert!(t.begin_text_selection_at(0, 3));
        t.finish_text_selection();
        assert!(!t.has_text_selection());
        assert_eq!(t.selected_text(), None);
    }

    #[test]
    fn hard_breaks_long_tokens() {
        let lines = vec![Line::from("abcdefghijklmnopqrstuvwxyz")];
        let wrapped = wrap_lines(&lines, 10);
        assert_eq!(text(&wrapped), vec!["abcdefghij", "klmnopqrst", "uvwxyz"]);
    }

    #[test]
    fn wrapping_never_splits_emoji_or_combining_graphemes() {
        let wrapped = wrap_lines(&[Line::from("1234567👩‍💻x")], 9);
        assert_eq!(text(&wrapped), vec!["1234567👩‍💻", "x"]);

        let wrapped = wrap_lines(&[Line::from("1234567e\u{301}x")], 8);
        assert_eq!(text(&wrapped), vec!["1234567e\u{301}", "x"]);
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

    // Release-only diagnostic for the synchronous resize path. It is ignored
    // in normal CI because elapsed-time assertions are machine-dependent.
    // Run with:
    //   cargo test -p open-max-tui --release -- --ignored --nocapture measure_transcript_resize
    #[test]
    #[ignore]
    fn measure_transcript_resize_cost() {
        use std::time::Instant;

        let mut transcript = Transcript::new();
        transcript.set_width(100);
        for turn in 0..2_500 {
            transcript.push_user(vec![Line::from(format!(
                "request {turn}: inspect the rendering path and preserve exact terminal behavior"
            ))]);
            transcript.push_assistant(vec![
                Line::from("The implementation keeps completed history in stable block caches."),
                Line::from("Resize changes wrapping width while preserving the styled source spans."),
                Line::from("Normal streaming and typing should never rebuild this finished prefix."),
            ]);
        }
        let source_lines = 10_000usize;

        for width in [72, 120, 88, 100] {
            let started = Instant::now();
            transcript.set_width(width);
            std::hint::black_box(transcript.len());
            eprintln!(
                "MEASURE transcript_resize source_lines={source_lines} width={width} elapsed_ms={:.3}",
                started.elapsed().as_secs_f64() * 1e3
            );
        }
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
    fn filter_matches_empty_query_matches_all() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push_user(vec![Line::from("a")]);
        t.push_assistant(vec![Line::from("b")]);
        t.push(vec![Line::from("c")]);
        assert_eq!(t.filter_matches(""), vec![0, 1, 2]);
    }

    #[test]
    fn filter_matches_case_insensitive_substring() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push(vec![Line::from("Error: file not found")]);
        t.push(vec![Line::from("ok done")]);
        t.push(vec![Line::from("ERROR in path /tmp/foo")]);
        t.push(vec![Line::from("nothing here")]);
        assert_eq!(t.filter_matches("error"), vec![0, 2]);
        assert_eq!(t.filter_matches("PATH"), vec![2]);
        assert_eq!(t.filter_matches("zzz"), Vec::<usize>::new());
    }

    #[test]
    fn find_covers_tool_full_output_beyond_the_folded_view() {
        let mut t = Transcript::new();
        t.set_width(80);
        t.push_user(vec![Line::from("you: check logs")]);
        t.push_assistant(vec![Line::from("assistant: looking")]);
        t.push_tool(
            vec![Line::from("✓ read_file app.rs")],
            "secret_token_xyz\nline2".into(),
        );
        assert_eq!(t.filter_matches("check logs"), vec![0]);
        assert_eq!(t.filter_matches("looking"), vec![1]);
        assert_eq!(t.filter_matches("read_file"), vec![2]);
        assert_eq!(t.filter_matches("secret_token"), vec![2]);
    }

    /// A find keystroke may not rebuild any block's search text: the cache
    /// is built exactly once per pushed block, and filtering plus previews
    /// run entirely against it.
    #[test]
    fn find_keystrokes_reuse_cached_search_text() {
        let mut t = Transcript::new();
        t.set_width(80);
        let before = SEARCH_BUILDS.with(|c| c.get());
        t.push_user(vec![Line::from("check the ledger")]);
        t.push_assistant(vec![Line::from("looking at it")]);
        t.push_tool(vec![Line::from("✓ bash")], "a very long output\nledger ok".into());
        assert_eq!(
            SEARCH_BUILDS.with(|c| c.get()) - before,
            3,
            "one build per pushed block"
        );
        let filtered = SEARCH_BUILDS.with(|c| c.get());
        for query in ["l", "le", "led", "ledger", "LEDGER", "zzz"] {
            std::hint::black_box(t.filter_matches(query));
            std::hint::black_box(t.block_preview(2, query));
        }
        assert_eq!(
            SEARCH_BUILDS.with(|c| c.get()),
            filtered,
            "a find keystroke rebuilt a block's search text"
        );
    }

    /// Find keystroke cost at session scale. Not a correctness test; run with:
    ///   cargo test -p open-max-tui --bin openmax --release -- --ignored --nocapture measure_find
    #[test]
    #[ignore]
    fn measure_find_filter_cost() {
        use std::time::Instant;
        let mut t = Transcript::new();
        t.set_width(120);
        let output = "a log line about work being done in the harness\n".repeat(40);
        for i in 0..500 {
            t.push_user(vec![Line::from(format!("user turn {i}: check the ledger"))]);
            t.push_assistant(vec![Line::from(format!("assistant reply {i} with prose"))]);
            t.push_tool(vec![Line::from("✓ bash")], output.clone());
        }

        let n = 200;
        let t0 = Instant::now();
        for _ in 0..n {
            std::hint::black_box(t.filter_matches("ledger"));
        }
        let cached_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;

        // The old shape: rebuild and lowercase every block's text per call.
        let t0 = Instant::now();
        for _ in 0..n {
            let texts: Vec<String> = t
                .blocks
                .iter()
                .map(|b| match &b.full_output {
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
                })
                .collect();
            let q = "ledger";
            std::hint::black_box(
                texts
                    .iter()
                    .enumerate()
                    .filter(|(_, text)| text.to_lowercase().contains(q))
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>(),
            );
        }
        let rebuilt_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!(
            "1500 blocks (~1 MB): cached {cached_ms:.3} ms per keystroke, rebuild {rebuilt_ms:.3} ms"
        );
    }

    /// The cached preview path must return exactly what the uncached
    /// implementation returned: rebuild the full text, lowercase per line,
    /// prefer the first line containing the query, else the first non-empty
    /// line. Includes unicode whose lowercase changes byte lengths.
    #[test]
    fn preview_matches_the_uncached_reference() {
        fn reference(t: &TestBlockText, query: &str) -> String {
            let text = &t.0;
            let q = query.trim();
            if !q.is_empty() {
                let q_low = q.to_lowercase();
                if let Some(line) = text.lines().find(|l| l.to_lowercase().contains(&q_low)) {
                    return line.trim().to_string();
                }
            }
            text.lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("")
                .to_string()
        }
        struct TestBlockText(String);

        let mut t = Transcript::new();
        t.set_width(80);
        t.push_user(vec![Line::from("İstanbul Σ ΟΔΟΣ mixed")]);
        t.push_assistant(vec![Line::from("first"), Line::from(""), Line::from("  Second Line  ")]);
        t.push_tool(
            vec![Line::from("✓ read_file notes.md")],
            "Header MATCH here\n\n漢字テスト row\ntail line".into(),
        );
        let texts = [
            TestBlockText("İstanbul Σ ΟΔΟΣ mixed".into()),
            TestBlockText("first\n\n  Second Line  ".into()),
            TestBlockText("✓ read_file notes.md\nHeader MATCH here\n\n漢字テスト row\ntail line".into()),
        ];
        for (bi, text) in texts.iter().enumerate() {
            for query in ["", "istanbul", "σ", "second", "match", "漢字", "tail", "zzz", "  match  "] {
                assert_eq!(
                    t.block_preview(bi, query).unwrap(),
                    reference(text, query),
                    "diverged on block {bi}, query {query:?}"
                );
            }
        }
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
    fn user_prompt_and_assistant_prose_have_distinct_structure() {
        let mut t = Transcript::new();
        t.set_width(24);
        t.push_user(vec![Line::from("hello")]);
        t.push_assistant(vec![Line::from("world")]);
        let rendered = text(t.lines());
        assert!(rendered[0].starts_with("❯ hello"));
        assert!(rendered[2].starts_with("world"));
        assert!(!rendered[2].starts_with('│'));
        assert_eq!(t.lines()[0].style.bg, Some(theme::USER_BG()));
        assert_ne!(t.lines()[2].style.bg, Some(theme::USER_BG()));
    }

    #[test]
    fn mouse_selection_uses_character_offsets_for_unicode() {
        let mut t = Transcript::new();
        t.set_width(40);
        t.push_user(vec![Line::from("héllo world")]);
        assert!(t.begin_text_selection_at(0, 2));
        // Release ON the final 'o': the cell under the cursor is included.
        assert!(t.update_text_selection_at(0, 6));
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
        // Release ON the final 'a' of gamma; the release cell is included.
        assert!(t.update_text_selection_at(2, 4));
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

