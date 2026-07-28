//! Restrained orientation for an empty chat session.
//!
//! This is transient chrome, not transcript content. It disappears as soon as
//! a turn produces history or a live tail, so it never reaches saved sessions
//! or the model context.

use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::theme;

const FULL_MIN_WIDTH: u16 = 52;
const FULL_MIN_HEIGHT: u16 = 7;
const COMPACT_MIN_WIDTH: u16 = 28;
const COMPACT_MIN_HEIGHT: u16 = 3;

pub fn render(area: Rect, buf: &mut Buffer) {
    let lines = lines(area.width, area.height);
    if lines.is_empty() {
        return;
    }

    let height = lines.len() as u16;
    // Place the mark in the upper third. It feels anchored to the conversation
    // plane while leaving a clear visual path down to the composer.
    let top_pad = area.height.saturating_sub(height) / 3;
    let draw_area = Rect {
        y: area.y + top_pad,
        height: height.min(area.height),
        ..area
    };
    Paragraph::new(lines)
        .alignment(Alignment::Center)
        .render(draw_area, buf);
}

fn lines(width: u16, height: u16) -> Vec<Line<'static>> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let title = Line::from(Span::styled(
        "◆ READY",
        Style::default()
            .fg(theme::ACCENT())
            .add_modifier(Modifier::BOLD),
    ));
    let dim = Style::default().fg(theme::DIM());
    if width >= FULL_MIN_WIDTH && height >= FULL_MIN_HEIGHT {
        vec![
            title,
            Line::default(),
            Line::from(Span::styled("A small core, shaped by your workflow.", dim)),
            Line::default(),
            Line::from(Span::styled("skills · tools · hooks · prompts", dim)),
        ]
    } else if width >= COMPACT_MIN_WIDTH && height >= COMPACT_MIN_HEIGHT {
        vec![
            title,
            Line::from(Span::styled("small core · your workflow", dim)),
            Line::from(Span::styled("skills · tools · hooks", dim)),
        ]
    } else if width >= 10 {
        vec![Line::from(Span::styled(
            "◆ ready",
            Style::default().fg(theme::DIM()),
        ))]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn full_state_orients_without_repeating_the_help_screen() {
        assert_eq!(
            plain(&lines(80, 12)),
            [
                "◆ READY",
                "",
                "A small core, shaped by your workflow.",
                "",
                "skills · tools · hooks · prompts",
            ]
        );
    }

    #[test]
    fn compact_state_preserves_readiness_and_product_orientation() {
        assert_eq!(
            plain(&lines(34, 3)),
            [
                "◆ READY",
                "small core · your workflow",
                "skills · tools · hooks",
            ]
        );
    }

    #[test]
    fn tiny_state_never_overflows_its_width() {
        assert_eq!(plain(&lines(12, 1)), ["◆ ready"]);
        assert!(lines(8, 1).is_empty());
        assert!(lines(80, 0).is_empty());
    }
}
