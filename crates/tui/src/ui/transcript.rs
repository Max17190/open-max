//! Writes finished blocks into real terminal scrollback via insert_before.
//! The live viewport only ever holds transient state; once content is final
//! it becomes ordinary terminal history: selectable, searchable, scrollable.

use ratatui::prelude::CrosstermBackend;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::Terminal;
use unicode_width::UnicodeWidthChar;

pub type Term = Terminal<CrosstermBackend<std::io::Stdout>>;

/// Insert a block above the viewport, wrapped to the terminal width, followed
/// by one blank spacer line.
pub fn insert_block(terminal: &mut Term, lines: Vec<Line<'static>>) -> std::io::Result<()> {
    let width = terminal.size()?.width.max(20);
    let mut wrapped = wrap_lines(lines, width);
    wrapped.push(Line::default());
    // insert_before takes u16; feed very long blocks in chunks.
    for chunk in wrapped.chunks(1024) {
        let chunk_vec: Vec<Line<'static>> = chunk.to_vec();
        terminal.insert_before(chunk_vec.len() as u16, |buf| {
            Paragraph::new(chunk_vec).render(buf.area, buf);
        })?;
    }
    Ok(())
}

/// Span-preserving word wrap. Greedy, breaking at the last space that fits;
/// hard-breaks tokens longer than the width.
pub fn wrap_lines(lines: Vec<Line<'static>>, width: u16) -> Vec<Line<'static>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    #[test]
    fn wraps_at_word_boundaries() {
        let lines = vec![Line::from("the quick brown fox jumps over the lazy dog")];
        let wrapped = wrap_lines(lines, 16);
        for l in text(&wrapped) {
            assert!(l.chars().count() <= 16, "line too long: {l:?}");
        }
        // The wrapper never drops characters; it only chooses break points.
        assert_eq!(text(&wrapped).join(""), "the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn hard_breaks_long_tokens() {
        let lines = vec![Line::from("abcdefghijklmnopqrstuvwxyz")];
        let wrapped = wrap_lines(lines, 10);
        assert_eq!(text(&wrapped), vec!["abcdefghij", "klmnopqrst", "uvwxyz"]);
    }

    #[test]
    fn empty_line_survives() {
        let wrapped = wrap_lines(vec![Line::default()], 10);
        assert_eq!(wrapped.len(), 1);
    }
}
