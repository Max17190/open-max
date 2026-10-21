//! The /context block: where each request of the session spends its tokens
//! (the frozen prompt and tools, then any trailing note of changes sent after
//! the history), plus the live cache-hit and budget state. Every line here is
//! sent on every request, so the point is to make the cost of each component
//! visible.

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
            "context (frozen at session creation; later changes ride a trailing note until /reload)"
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

    if !breakdown.note.is_empty() {
        lines.push(Line::from(Span::styled(
            "  trailing note (changes since the prompt, sent after the history)".to_string(),
            Style::default().fg(theme::DIM()),
        )));
        for (label, chars) in &breakdown.note {
            total_chars += chars;
            lines.push(row(label, tok(*chars), ""));
        }
    }

    lines.push(Line::from(Span::styled(
        format!("  {}", "─".repeat(38)),
        Style::default().fg(theme::DIM()),
    )));
    lines.push(row("total per request", tok(total_chars), ""));

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

    /// The note's components are rows of their own after the prompt's and
    /// count toward the per-request total once, as a tool file's entry in the
    /// tools array does; a skill's and a memory's index lines are already
    /// inside the prompt's sections, so their rows only itemize them, and are
    /// absent when there are none.
    #[test]
    fn the_note_rows_count_once_toward_the_total() {
        let bd = PromptBreakdown {
            components: vec![("base rules".into(), 400)],
            note: vec![("note header".into(), 52), ("tool files changes".into(), 100)],
            tools: vec![("bash".into(), 200, false), ("wordfreq".into(), 60, true)],
            skills: vec![("release".into(), 40)],
            memory: vec![("build".into(), 32)],
        };
        let lines: Vec<String> = context_block(&bd, Provenance::Frozen, None, None, None).iter().map(plain).collect();
        let at = |label: &str| lines.iter().position(|l| l.trim_start().starts_with(label)).unwrap_or_else(|| panic!("no {label} row: {lines:#?}"));
        assert!(at("memory (1)") < at("trailing note") && at("trailing note") < at("note header"), "{lines:#?}");
        assert!(lines[at("tool files changes")].contains("~   25 tok"), "{lines:#?}");
        assert!(lines[at("external tools (1)")].contains("~   15 tok"), "{lines:#?}");
        assert!(lines[at("total per request")].contains(&format!("~{:>5} tok", (400 + 200 + 60 + 52 + 100) / 4)), "{lines:#?}");
        assert!(lines[at("skills (1)")].contains("~    0 tok") && lines[at("skills (1)")].contains("release 10"), "{lines:#?}");
        assert!(lines[at("memory (1)")].contains("~    0 tok") && lines[at("memory (1)")].contains("build 8"), "{lines:#?}");
        let none = PromptBreakdown { skills: Vec::new(), memory: Vec::new(), ..bd.clone() };
        let lines: Vec<String> = context_block(&none, Provenance::Frozen, None, None, None).iter().map(plain).collect();
        assert!(!lines.iter().any(|l| l.contains("skills (") || l.contains("memory (")), "no rows for none: {lines:#?}");
        let without = PromptBreakdown { note: Vec::new(), ..bd };
        let lines: Vec<String> = context_block(&without, Provenance::Frozen, None, None, None).iter().map(plain).collect();
        assert!(!lines.iter().any(|l| l.contains("trailing note (changes")), "no note, no note heading: {lines:#?}");
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
