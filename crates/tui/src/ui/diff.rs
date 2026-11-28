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

impl File<'_> {
    /// Whether a change to this file was cut off at its diff's deadline,
    /// so its counts may include unchanged lines.
    fn upper_bound(&self) -> bool {
        self.changes.iter().any(|(change, _)| change.upper_bound)
    }
}

/// The session's changes by file, and the paths written with no line
/// changed (a file rewritten as it was, or created empty), which count as
/// no change.
fn by_file<'a>(changes: impl Iterator<Item = (&'a FileChange, Option<&'a str>)>) -> (Vec<File<'a>>, Vec<&'a str>) {
    let mut files: Vec<File<'a>> = Vec::new();
    let mut at: HashMap<&'a str, usize> = HashMap::new();
    let mut unchanged: Vec<&'a str> = Vec::new();
    for (change, diff) in changes {
        if change.added == 0 && change.removed == 0 {
            if !unchanged.contains(&change.path.as_str()) {
                unchanged.push(&change.path);
            }
            continue;
        }
        let i = *at.entry(change.path.as_str()).or_insert_with(|| {
            files.push(File { path: &change.path, added: 0, removed: 0, changes: Vec::new() });
            files.len() - 1
        });
        let file = &mut files[i];
        file.added += change.added;
        file.removed += change.removed;
        file.changes.push((change, diff));
    }
    unchanged.retain(|path| !at.contains_key(path));
    (files, unchanged)
}

/// The paths written with no line changed, said apart from the list.
fn unchanged_line(unchanged: &[&str], indent: &str) -> Option<Line<'static>> {
    (!unchanged.is_empty()).then(|| dim(format!("{indent}written with no line changed: {}", unchanged.join(", "))))
}

fn dim(text: String) -> Line<'static> {
    Line::from(Span::styled(text, Style::default().fg(theme::DIM)))
}

/// The list `/diff` prints: each file with its added and removed lines,
/// counts first so a long path wraps alone at a narrow width.
pub fn summary<'a>(
    changes: impl Iterator<Item = (&'a FileChange, Option<&'a str>)>,
    compacted: bool,
) -> Vec<Line<'static>> {
    let (files, unchanged) = by_file(changes);
    if files.is_empty() && unchanged.is_empty() {
        return none_yet(compacted);
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
        if file.upper_bound() {
            row.push(Span::styled(" · counts are an upper bound", Style::default().fg(theme::DIM)));
        }
        lines.push(Line::from(row));
    }
    lines.extend(unchanged_line(&unchanged, "  "));
    let (added, removed) = files.iter().fold((0, 0), |(a, r), f| (a + f.added, r + f.removed));
    let noun = if files.len() == 1 { "file" } else { "files" };
    let bound = if files.iter().any(File::upper_bound) { "at most " } else { "" };
    lines.push(dim(format!("  {} {noun}, {bound}+{added} −{removed} · /diff all opens the combined diff", files.len())));
    lines.push(dim(format!("  {LIMIT}")));
    if compacted {
        lines.push(dim(format!("  {COMPACTED}")));
    }
    lines
}

/// Said when the session was compacted before it was resumed: the edits
/// the compaction dropped left no tool block to read.
const COMPACTED: &str = "edits from before this session's compaction are not listed";

fn none_yet(compacted: bool) -> Vec<Line<'static>> {
    let mut lines = vec![dim(format!("no file changes in this session yet. /diff lists {LIMIT}"))];
    if compacted {
        lines = vec![dim(format!("no file changes listed. /diff lists {LIMIT}")), dim(COMPACTED.to_string())];
    }
    lines
}

/// The combined diff `/diff all` prints: each file's diffs in the order the
/// tools made them, unindented so a copy of the block is the diff itself.
pub fn combined<'a>(
    changes: impl Iterator<Item = (&'a FileChange, Option<&'a str>)>,
    compacted: bool,
) -> Vec<Line<'static>> {
    let (files, unchanged) = by_file(changes);
    if files.is_empty() && unchanged.is_empty() {
        return none_yet(compacted);
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
    lines.extend(unchanged_line(&unchanged, ""));
    lines.push(dim(format!("/diff lists {LIMIT}")));
    if compacted {
        lines.push(dim(COMPACTED.to_string()));
    }
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
