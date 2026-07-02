//! Compact one-line tool cards with a short output preview, plus colored
//! unified diffs. Full output is available on demand (Ctrl+O).

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;

/// Lines for a finished tool call: status glyph, name, summary, then a short
/// dim preview of the output (more of it when the call failed).
pub fn tool_block(
    name: &str,
    summary: &str,
    ok: bool,
    output: &str,
    diff: Option<&DiffText>,
) -> Vec<Line<'static>> {
    let (glyph, glyph_style) = if ok {
        ("✓", Style::default().fg(theme::OK))
    } else {
        ("✗", Style::default().fg(theme::ERR))
    };
    let mut header = vec![
        Span::styled(glyph.to_string(), glyph_style),
        Span::raw(" "),
        Span::styled(name.to_string(), Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ];
    match diff {
        // For edits the path plus change counts say it all; the summary would
        // repeat the path.
        Some(d) => header.extend([
            Span::styled(clip(&d.path, 90), Style::default().fg(theme::DIM)),
            Span::raw("  "),
            Span::styled(format!("+{}", d.added), Style::default().fg(theme::OK)),
            Span::raw(" "),
            Span::styled(format!("−{}", d.removed), Style::default().fg(theme::ERR)),
        ]),
        None => header.push(Span::styled(clip(summary, 90), Style::default().fg(theme::DIM))),
    }
    let mut lines = vec![Line::from(header)];

    match diff {
        Some(d) => lines.extend(diff_lines(&d.diff)),
        None => {
            let preview_lines = if ok { 2 } else { 5 };
            for line in output.lines().take(preview_lines) {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(clip(line, 110), Style::default().fg(theme::DIM)),
                ]));
            }
            let total = output.lines().count();
            if total > preview_lines {
                lines.push(Line::from(Span::styled(
                    format!("  … {} more lines (ctrl+o to expand)", total - preview_lines),
                    Style::default().fg(theme::DIM).add_modifier(Modifier::ITALIC),
                )));
            }
        }
    }
    lines
}

/// A running tool, shown in the live viewport while it executes.
pub fn running_line(name: &str, summary: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled("⚙ ", Style::default().fg(theme::WARN)),
        Span::styled(name.to_string(), Style::default().fg(theme::ACCENT)),
        Span::raw(" "),
        Span::styled(clip(summary, 90), Style::default().fg(theme::DIM)),
    ])
}

pub struct DiffText {
    pub path: String,
    pub diff: String,
    pub added: usize,
    pub removed: usize,
}

/// Unified diff with the conventional coloring, gutter-indented.
pub fn diff_lines(diff: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for raw in diff.lines() {
        let style = if raw.starts_with("+++") || raw.starts_with("---") || raw.starts_with("@@") {
            Style::default().fg(theme::DIM)
        } else if raw.starts_with('+') {
            Style::default().fg(theme::OK)
        } else if raw.starts_with('-') {
            Style::default().fg(theme::ERR)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![Span::raw("  "), Span::styled(raw.to_string(), style)]));
    }
    lines
}

fn clip(s: &str, max: usize) -> String {
    let clean = s.replace(['\n', '\r'], " ");
    if clean.chars().count() <= max {
        clean
    } else {
        let cut: String = clean.chars().take(max).collect();
        format!("{cut}…")
    }
}
