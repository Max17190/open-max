//! `/diff`: the files this session's write and edit tools changed, and the
//! diff they made, read from the transcript's tool blocks. A shell command's
//! changes reach no such block, so `git diff` is named for those.

use std::collections::HashMap;

use ratatui::style::Style;
use ratatui::text::{Line, Span};

use super::text::strip_escapes;
use super::transcript::FileChange;
use crate::theme;

/// The limit, stated wherever /diff answers.
const LIMIT: &str = "edit and write tools only; git diff shows what shell commands changed";

/// One file's changes, in the order the session first touched it.
struct File<'a> {
    path: &'a str,
    added: usize,
    removed: usize,
    changes: Vec<(&'a FileChange, Option<&'a str>)>,
}

fn by_file<'a>(changes: impl Iterator<Item = (&'a FileChange, Option<&'a str>)>) -> Vec<File<'a>> {
    let mut files: Vec<File<'a>> = Vec::new();
    let mut at: HashMap<&'a str, usize> = HashMap::new();
    for (change, diff) in changes {
        let i = *at.entry(change.path.as_str()).or_insert_with(|| {
            files.push(File { path: &change.path, added: 0, removed: 0, changes: Vec::new() });
            files.len() - 1
        });
        let file = &mut files[i];
        file.added += change.added;
        file.removed += change.removed;
        file.changes.push((change, diff));
    }
    files
}

fn dim(text: String) -> Line<'static> {
    Line::from(Span::styled(text, Style::default().fg(theme::DIM)))
}

/// The list `/diff` prints: each file with its added and removed lines,
/// counts first so a long path wraps alone at a narrow width.
pub fn summary<'a>(changes: impl Iterator<Item = (&'a FileChange, Option<&'a str>)>) -> Vec<Line<'static>> {
    let files = by_file(changes);
    if files.is_empty() {
        return vec![dim(format!("no file changes in this session yet. /diff lists {LIMIT}"))];
    }
    let added_width = files.iter().map(|f| f.added.to_string().len()).max().unwrap_or(1);
    let removed_width = files.iter().map(|f| f.removed.to_string().len()).max().unwrap_or(1);
    let mut lines = vec![Line::from(Span::styled("Files the edit and write tools changed", theme::WARN_BOLD))];
    for file in &files {
        let mut row = vec![
            Span::raw("  "),
            Span::styled(format!("+{:<added_width$}", file.added), Style::default().fg(theme::OK)),
            Span::raw(" "),
            Span::styled(format!("−{:<removed_width$}", file.removed), Style::default().fg(theme::ERR)),
            Span::raw("  "),
            Span::raw(file.path.to_string()),
        ];
        if file.changes.len() > 1 {
            row.push(Span::styled(format!(" ({} changes)", file.changes.len()), Style::default().fg(theme::DIM)));
        }
        lines.push(Line::from(row));
    }
    let (added, removed) = files.iter().fold((0, 0), |(a, r), f| (a + f.added, r + f.removed));
    let noun = if files.len() == 1 { "file" } else { "files" };
    lines.push(dim(format!("  {} {noun}, +{added} −{removed} · /diff all opens the combined diff", files.len())));
    lines.push(dim(format!("  {LIMIT}")));
    lines
}

/// The combined diff `/diff all` prints: each file's diffs in the order the
/// tools made them, unindented so a copy of the block is the diff itself.
pub fn combined<'a>(changes: impl Iterator<Item = (&'a FileChange, Option<&'a str>)>) -> Vec<Line<'static>> {
    let files = by_file(changes);
    if files.is_empty() {
        return vec![dim(format!("no file changes in this session yet. /diff lists {LIMIT}"))];
    }
    let mut lines = Vec::new();
    for file in &files {
        for (change, diff) in &file.changes {
            match diff {
                Some(diff) => lines.extend(diff.lines().map(diff_line)),
                None => lines.push(dim(format!(
                    "{}: +{} −{}, diff not kept across a resume",
                    file.path, change.added, change.removed
                ))),
            }
        }
    }
    lines.push(dim(format!("/diff lists {LIMIT}")));
    lines
}

fn diff_line(line: &str) -> Line<'static> {
    let line = strip_escapes(line).into_owned();
    let style = if line.starts_with("+++") || line.starts_with("---") {
        theme::ACCENT_BOLD
    } else if line.starts_with("@@") {
        Style::default().fg(theme::DIM)
    } else if line.starts_with('+') {
        Style::default().fg(theme::OK)
    } else if line.starts_with('-') {
        Style::default().fg(theme::ERR)
    } else {
        Style::default()
    };
    Line::from(Span::styled(line, style))
}
