//! The /context block: where the session's frozen prompt prefix spends its
//! tokens, plus the live cache-hit and budget state.
//! Every line here is a token the model re-prefills on every single turn,
//! so the point is to make the cost of each component visible.

use open_max_core::prompt::PromptBreakdown;
use open_max_core::types::estimate_tokens as tok;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;

/// Label column width: one wider than the longest label the core mints
/// ("system prompt (persisted)", 25), so every token column aligns.
const LABEL_W: usize = 26;

fn row(label: &str, tokens: usize, detail: &str) -> Line<'static> {
    let mut spans = vec![
        Span::styled(format!("  {label:<LABEL_W$}"), Style::default().fg(theme::ACCENT())),
        Span::raw(format!("~{tokens:>5} tok")),
    ];
    if !detail.is_empty() {
        spans.push(Span::styled(format!("   {detail}"), Style::default().fg(theme::DIM())));
    }
    Line::from(spans)
}

/// What the header may truthfully claim about the numbers below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// A hydrated session's own frozen breakdown.
    Frozen,
    /// No session yet: what the next new session would freeze from today's
    /// config.
    NewPreview,
    /// A resumed session before its next message: its frozen breakdown is
    /// not loaded yet, so the numbers below are still today's-config
    /// preview, not the session's own.
    ResumedPending,
}

pub fn context_block(
    breakdown: &PromptBreakdown,
    provenance: Provenance,
    budget: Option<(usize, usize)>,
    cache_pct: Option<u8>,
    // Cached and total prompt tokens over the whole session, when the
    // endpoint reported them. The last turn alone cannot show a broken
    // prefix: one cold request looks exactly like one evicted cache.
    session_cache: Option<(u64, u64)>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let header = match provenance {
        Provenance::Frozen => {
            "context (frozen at session creation; config changes apply to /new sessions)"
        }
        Provenance::NewPreview => "context (preview of the next new session)",
        // A resumed session continues with its OWN frozen prompt, which is
        // not loaded until its next message runs; claiming these numbers
        // preview "the next new session" would describe nothing that can
        // happen from here.
        Provenance::ResumedPending => {
            "context (preview from today's config; the resumed session's frozen context loads with its next message)"
        }
    };
    lines.push(Line::from(Span::styled(
        header.to_string(),
        Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
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

    if !breakdown.memory.is_empty() {
        // Memory lines are already counted inside the "memory index" prompt
        // component; this row just itemizes them.
        let detail = breakdown
            .memory
            .iter()
            .map(|(name, c)| format!("{name} {}", tok(*c)))
            .collect::<Vec<_>>()
            .join(" · ");
        lines.push(row(&format!("memory ({})", breakdown.memory.len()), 0, &detail));
    }

    lines.push(Line::from(Span::styled(
        format!("  {}", "─".repeat(38)),
        Style::default().fg(theme::DIM()),
    )));
    lines.push(row("total prompt prefix", tok(total_chars), ""));

    if let Some(pct) = cache_pct {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<LABEL_W$}", "cache hit (last turn)"),
                Style::default().fg(theme::ACCENT()),
            ),
            Span::raw(format!("{pct:>6} %")),
        ]));
    }
    if let Some((cached, prompt)) = session_cache.filter(|(_, p)| *p > 0) {
        let pct = (cached as f64 / prompt as f64 * 100.0).round() as u64;
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<LABEL_W$}", "cache hit (session)"),
                Style::default().fg(theme::ACCENT()),
            ),
            Span::raw(format!("{pct:>6} %")),
            Span::styled(
                format!("   {cached} of {prompt} prompt tok"),
                Style::default().fg(theme::DIM()),
            ),
        ]));
    }
    if let Some((used, total)) = budget {
        // Transcript plus the tool schemas above, the same total the budget
        // enforces, so this row and compaction never disagree.
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<LABEL_W$}", "context used"),
                Style::default().fg(theme::ACCENT()),
            ),
            Span::raw(format!("~{used:>5} tok")),
            Span::styled(
                format!("   of {total} ({}%)", (used as f64 / total.max(1) as f64 * 100.0) as u32),
                Style::default().fg(theme::DIM()),
            ),
        ]));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(line: &Line<'_>) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// "system prompt (persisted)" is the longest label the core mints and
    /// the one every hydrated session shows; its token column must sit in
    /// the same column as every other row's.
    #[test]
    fn the_longest_label_keeps_its_column() {
        let a = plain(&row("base rules", 100, ""));
        let b = plain(&row("system prompt (persisted)", 100, ""));
        assert_eq!(a.find('~'), b.find('~'), "{a:?} vs {b:?}");
    }

    /// Three provenances, three true sentences. A resumed session's next
    /// message continues THAT session, so its header may not promise a
    /// preview of "the next new session", and it must say the numbers are
    /// not the session's own yet.
    #[test]
    fn the_header_tells_the_resumed_session_the_truth() {
        let bd = PromptBreakdown::default();
        let heads: Vec<String> =
            [Provenance::Frozen, Provenance::NewPreview, Provenance::ResumedPending]
                .into_iter()
                .map(|p| plain(&context_block(&bd, p, None, None, None)[0]))
                .collect();
        assert!(heads[2].contains("resumed session"), "{:?}", heads[2]);
        assert!(
            !heads[2].contains("next new session"),
            "the resumed header still promises a new-session preview: {:?}",
            heads[2]
        );
        assert_ne!(heads[0], heads[2]);
        assert_ne!(heads[1], heads[2]);
    }
}
