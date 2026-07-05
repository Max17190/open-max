//! The /context block: where the session's frozen prompt prefix spends its
//! tokens (pi-token-burden style), plus the live cache-hit and budget state.
//! Every line here is a token the model re-prefills on every single turn,
//! so the point is to make the cost of each component visible.

use open_max_core::prompt::PromptBreakdown;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;

/// Same heuristic as the budget estimator: ~4 chars per token.
fn tok(chars: usize) -> usize {
    chars / 4
}

fn row(label: &str, tokens: usize, detail: &str) -> Line<'static> {
    let mut spans = vec![
        Span::styled(format!("  {label:<22}"), Style::default().fg(theme::ACCENT)),
        Span::raw(format!("~{tokens:>5} tok")),
    ];
    if !detail.is_empty() {
        spans.push(Span::styled(format!("   {detail}"), Style::default().fg(theme::DIM)));
    }
    Line::from(spans)
}

pub fn context_block(
    breakdown: &PromptBreakdown,
    frozen: bool,
    budget: Option<(usize, usize)>,
    cache_pct: Option<u8>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let header = if frozen {
        "context (frozen at session creation; config changes apply to /new sessions)"
    } else {
        "context (preview of the next new session)"
    };
    lines.push(Line::from(Span::styled(
        header.to_string(),
        Style::default().fg(theme::DIM).add_modifier(Modifier::ITALIC),
    )));

    let mut total_chars = 0usize;
    for (label, chars) in &breakdown.components {
        total_chars += chars;
        lines.push(row(label, tok(*chars), ""));
    }

    let builtin_chars: usize = breakdown.tools.iter().filter(|t| !t.2).map(|t| t.1).sum();
    let builtin_count = breakdown.tools.iter().filter(|t| !t.2).count();
    total_chars += builtin_chars;
    lines.push(row(&format!("tools ({builtin_count} built-in)"), tok(builtin_chars), ""));

    let externals: Vec<&(String, usize, bool)> = breakdown.tools.iter().filter(|t| t.2).collect();
    if !externals.is_empty() {
        let chars: usize = externals.iter().map(|t| t.1).sum();
        total_chars += chars;
        let detail = externals
            .iter()
            .map(|(name, c, _)| format!("{name} {}", tok(*c)))
            .collect::<Vec<_>>()
            .join(" · ");
        lines.push(row(&format!("external tools ({})", externals.len()), tok(chars), &detail));
    }

    if !breakdown.skills.is_empty() {
        // Skill lines are already counted inside the "skills index" prompt
        // component; this row just itemizes them.
        let detail = breakdown
            .skills
            .iter()
            .map(|(name, c)| format!("{name} {}", tok(*c)))
            .collect::<Vec<_>>()
            .join(" · ");
        lines.push(row(&format!("skills ({})", breakdown.skills.len()), 0, &detail));
    }

    lines.push(Line::from(Span::styled(
        format!("  {}", "─".repeat(34)),
        Style::default().fg(theme::DIM),
    )));
    lines.push(row("total prompt prefix", tok(total_chars), ""));

    if let Some(pct) = cache_pct {
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<22}", "cache hit (last turn)"), Style::default().fg(theme::ACCENT)),
            Span::raw(format!("{pct:>6} %")),
        ]));
    }
    if let Some((used, total)) = budget {
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<22}", "context used"), Style::default().fg(theme::ACCENT)),
            Span::raw(format!("~{used:>5} tok")),
            Span::styled(
                format!("   of {total} ({}%)", (used as f64 / total.max(1) as f64 * 100.0) as u32),
                Style::default().fg(theme::DIM),
            ),
        ]));
    }
    lines
}
