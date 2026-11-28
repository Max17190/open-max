//! The /memory block: this project's memories as the core scans them, each
//! with how often the file tools touched it, how long ago, and whether a new
//! session's index carries it. Read-only; the files stay the record.

use open_max_core::memory::{MemoryScan, INDEX_FLOOR_DAYS, MAX_MEMORY_BYTES, MEMORY_DIR};
use ratatui::style::Style;
use ratatui::text::{Line, Span};

use super::extensions::aside;
use crate::theme;

/// Where a memory stands for the index a new session freezes.
fn state(in_index: bool, last_event_hours: u64) -> &'static str {
    if in_index {
        "indexed"
    } else if last_event_hours as f64 > INDEX_FLOOR_DAYS * 24.0 {
        "faded"
    } else {
        "over cap"
    }
}

/// The scan buckets ages to whole hours and clamps them to one, so an age of
/// one hour means anything under two.
fn age(hours: u64) -> String {
    match hours {
        0 | 1 => "<2h ago".into(),
        h if h < 48 => format!("{h}h ago"),
        h => format!("{}d ago", h / 24),
    }
}

fn uses(n: usize) -> String {
    if n == 1 { "1 use".into() } else { format!("{n} uses") }
}

pub fn memory_block(scan: &MemoryScan) -> Vec<Line<'static>> {
    let mut lines = vec![aside(format!(
        "memory ({MEMORY_DIR}/; indexed means a new session's prompt lists it)"
    ))];
    let recall = aside("  search everything kept, sessions included: openmax --recall \"<query>\"");
    if scan.entries.is_empty() {
        lines.push(aside(format!(
            "  (none yet) the agent writes one fact per file under {MEMORY_DIR}/ as it learns this project"
        )));
        lines.push(recall);
        return lines;
    }
    for entry in &scan.entries {
        lines.push(Line::from(vec![
            Span::styled(format!("  {}", super::text::pad_right(&entry.name, 20)), theme::ACCENT_BOLD),
            Span::styled(
                format!("{:<10}{:<10}{:<9}", state(entry.in_index, entry.last_event_hours), uses(entry.uses), age(entry.last_event_hours)),
                Style::default().fg(theme::DIM),
            ),
            Span::styled(super::text::clip(&entry.description, 56), Style::default().fg(theme::DIM)),
        ]));
    }
    let count = |label| scan.entries.iter().filter(|e| state(e.in_index, e.last_event_hours) == label).count();
    let mut summary = format!("  {} on disk · {} indexed", scan.entries.len(), count("indexed"));
    for (label, why) in [
        ("faded", format!("unused for {INDEX_FLOOR_DAYS} days")),
        ("over cap", format!("past the {MAX_MEMORY_BYTES}-byte index")),
    ] {
        let n = count(label);
        if n > 0 {
            summary.push_str(&format!(" · {n} {label} ({why}; still on disk)"));
        }
    }
    lines.push(aside(summary));
    lines.push(aside("  uses: reads and writes by the agent's file tools · age: since the last one or the file's change"));
    lines.push(recall);
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state names each reason a memory is or is not in the index, and
    /// ages read in the unit a person would say them in.
    #[test]
    fn states_and_ages_say_what_the_scan_decided() {
        let floor = (INDEX_FLOOR_DAYS * 24.0) as u64;
        assert_eq!(state(true, floor + 1), "indexed");
        assert_eq!(state(false, floor + 1), "faded");
        assert_eq!(state(false, floor), "over cap");
        assert_eq!(age(1), "<2h ago");
        assert_eq!(age(47), "47h ago");
        assert_eq!(age(72), "3d ago");
        assert_eq!(uses(1), "1 use");
        assert_eq!(uses(0), "0 uses");
    }
}
