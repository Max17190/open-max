//! The /resume picker: past sessions in this project, newest first, rendered
//! as an overlay mode like the models panel.

use open_max_core::sessions::SessionMeta;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::Frame;

use crate::theme;

pub struct SessionsState {
    pub items: Vec<SessionMeta>,
    pub selected: usize,
    /// Session id pending delete confirmation.
    pub confirm_delete: Option<String>,
}

impl SessionsState {
    pub fn new(items: Vec<SessionMeta>) -> Self {
        Self { items, selected: 0, confirm_delete: None }
    }

    pub fn selected_item(&self) -> Option<&SessionMeta> {
        self.items.get(self.selected)
    }
}

/// "just now", "5m ago", "3h ago", "2d ago" from epoch seconds.
pub fn age(updated_at: u64, now: u64) -> String {
    let secs = now.saturating_sub(updated_at);
    match secs {
        0..=59 => "just now".into(),
        60..=3_599 => format!("{}m ago", secs / 60),
        3_600..=86_399 => format!("{}h ago", secs / 3_600),
        _ => format!("{}d ago", secs / 86_400),
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &SessionsState, now: u64) {
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("resume", Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("  {} session{} in this project", state.items.len(), if state.items.len() == 1 { "" } else { "s" }),
            Style::default().fg(theme::DIM()),
        ),
    ]));

    let rows_budget = area.height.saturating_sub(3) as usize;
    let first = state.selected.saturating_sub(rows_budget.saturating_sub(1));
    for (i, item) in state.items.iter().enumerate().skip(first).take(rows_budget) {
        let marker = if i == state.selected {
            Span::styled("▸ ", Style::default().fg(theme::ACCENT()))
        } else {
            Span::raw("  ")
        };
        let title_style = if i == state.selected {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            marker,
            Span::styled(format!("{:<52}", clip(&item.title, 50)), title_style),
            Span::styled(age(item.updated_at, now), Style::default().fg(theme::DIM())),
        ]));
    }

    if let Some(id) = &state.confirm_delete {
        let title = state
            .items
            .iter()
            .find(|s| &s.id == id)
            .map(|s| s.title.clone())
            .unwrap_or_else(|| id.clone());
        lines.push(Line::from(vec![
            Span::styled("delete ", Style::default().fg(theme::ERR()).add_modifier(Modifier::BOLD)),
            Span::raw(format!("\u{201c}{}\u{201d}?  ", clip(&title, 50))),
            Span::styled("[y] yes  [n] no", Style::default().fg(theme::DIM())),
        ]));
    } else {
        lines.push(Line::from(Span::styled(
            "↑/↓ navigate · enter resume · x delete · esc close",
            Style::default().fg(theme::DIM()),
        )));
    }

    Paragraph::new(lines).render(area, frame.buffer_mut());
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ages_read_naturally() {
        assert_eq!(age(1_000, 1_030), "just now");
        assert_eq!(age(1_000, 1_000 + 5 * 60), "5m ago");
        assert_eq!(age(1_000, 1_000 + 3 * 3_600), "3h ago");
        assert_eq!(age(1_000, 1_000 + 2 * 86_400), "2d ago");
    }
}
