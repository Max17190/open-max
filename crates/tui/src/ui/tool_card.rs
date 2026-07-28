//! Compact one-line tool results. Full output and diffs are available on
//! demand (Ctrl+O); failures retain one diagnostic line in the transcript.

use std::time::Duration;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;

/// Lines for a finished tool call: status glyph, name, summary, and an
/// optional first failure line.
pub fn tool_block(
    name: &str,
    summary: &str,
    ok: bool,
    output: &str,
    diff: Option<&DiffText>,
) -> Vec<Line<'static>> {
    tool_block_timed(name, summary, ok, output, diff, None)
}

pub fn tool_block_timed(
    name: &str,
    summary: &str,
    ok: bool,
    output: &str,
    diff: Option<&DiffText>,
    duration: Option<Duration>,
) -> Vec<Line<'static>> {
    let (glyph, glyph_style) = if ok {
        ("✓", Style::default().fg(theme::OK()))
    } else {
        ("✗", Style::default().fg(theme::ERR()))
    };
    let mut header = vec![
        Span::styled(glyph.to_string(), glyph_style),
        Span::raw(" "),
        Span::styled(
            human_name(name),
            Style::default()
                .fg(theme::ACCENT())
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ];
    match diff {
        // For edits the path plus change counts say it all; the summary would
        // repeat the path.
        Some(d) => header.extend([
            Span::styled(clip(&d.path, 90), Style::default().fg(theme::DIM())),
            Span::raw("  "),
            Span::styled(format!("+{}", d.added), Style::default().fg(theme::OK())),
            Span::raw(" "),
            Span::styled(format!("−{}", d.removed), Style::default().fg(theme::ERR())),
        ]),
        None => header.push(Span::styled(
            clip(summary, 90),
            Style::default().fg(theme::DIM()),
        )),
    }
    if let Some(duration) = duration {
        header.extend([
            Span::raw("  "),
            Span::styled(format_duration(duration), Style::default().fg(theme::DIM())),
        ]);
    }
    let mut lines = vec![Line::from(header)];
    if !ok {
        if let Some(reason) = output.lines().find(|line| !line.trim().is_empty()) {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(clip(reason, 110), Style::default().fg(theme::ERR())),
            ]));
        }
    }
    lines
}

/// A running tool, shown in the live viewport while it executes.
pub fn running_line(name: &str, summary: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled("⚙ ", Style::default().fg(theme::WARN())),
        Span::styled(human_name(name), Style::default().fg(theme::ACCENT())),
        Span::raw(" "),
        Span::styled(clip(summary, 90), Style::default().fg(theme::DIM())),
    ])
}

pub fn human_name(name: &str) -> String {
    match name {
        "read_file" => "Read".into(),
        "write_file" => "Write".into(),
        "edit_file" => "Edit".into(),
        "list_dir" => "List".into(),
        "grep" => "Search".into(),
        "glob" => "Find".into(),
        "bash" => "Shell".into(),
        other => {
            let mut words = other.split('_');
            let first = words.next().unwrap_or("Tool");
            let mut label = first.to_string();
            if let Some(initial) = label.get_mut(0..1) {
                initial.make_ascii_uppercase();
            }
            for word in words {
                label.push(' ');
                label.push_str(word);
            }
            label
        }
    }
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.1}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

pub struct DiffText {
    pub path: String,
    pub diff: String,
    pub added: usize,
    pub removed: usize,
}

/// Unified diff with the conventional coloring, gutter-indented.
fn clip(s: &str, max: usize) -> String {
    let clean = s.replace(['\n', '\r'], " ");
    super::text::clip(&clean, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn successful_tool_is_one_line_with_human_label_and_duration() {
        let lines = tool_block_timed(
            "read_file",
            "src/main.rs",
            true,
            "one\ntwo\nthree\nfour",
            None,
            Some(Duration::from_millis(42)),
        );
        let text = plain(&lines);
        assert!(text.contains("✓ Read"));
        assert!(text.contains("42ms"));
        assert!(!text.contains("one"));
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn failure_keeps_only_the_first_diagnostic_line() {
        let output = "permission denied\n2\n3\n4\n5\n6\n7";
        let lines = tool_block_timed("bash", "cargo test", false, output, None, None);
        assert_eq!(lines[1].spans[1].style.fg, Some(theme::ERR()));
        assert!(plain(&lines).contains("permission denied"));
        assert!(!plain(&lines).contains("\n  2"));
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn edit_summary_and_diff_preview_are_bounded() {
        let diff = (0..20)
            .map(|index| format!("+line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let edit = DiffText {
            path: "src/app.rs".into(),
            diff,
            added: 20,
            removed: 3,
        };
        let lines = tool_block_timed("edit_file", "ignored", true, "", Some(&edit), None);
        let text = plain(&lines);
        assert!(text.contains("src/app.rs"));
        assert!(text.contains("+20"));
        assert!(text.contains("−3"));
        assert_eq!(lines.len(), 1);
    }
}
