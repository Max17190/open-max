//! Restrained orientation for an empty chat session.
//!
//! This is transient chrome, not transcript content. It disappears as soon as
//! a turn produces history or a live tail, so it never reaches saved sessions
//! or the model context.

use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::theme;

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
    if width < 5 || height == 0 {
        return Vec::new();
    }

    vec![Line::from(Span::styled(
        "READY",
        Style::default().fg(theme::DIM()),
    ))]
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
    fn ready_state_is_one_quiet_signal_at_every_supported_size() {
        assert_eq!(plain(&lines(80, 12)), ["READY"]);
        assert_eq!(plain(&lines(20, 2)), ["READY"]);
    }

    #[test]
    fn tiny_state_never_overflows_its_width_or_height() {
        assert!(lines(4, 1).is_empty());
        assert!(lines(80, 0).is_empty());
    }
}
