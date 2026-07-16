//! A deliberately small markdown renderer producing ratatui Lines: headings,
//! emphasis, inline code, lists, blockquotes, rules, and syntect-highlighted
//! fenced code. Enough for model output without pulling in a full parser.

use std::str::FromStr;
use std::sync::OnceLock;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{
    Color as SynColor, ScopeSelectors, StyleModifier, Theme, ThemeItem, ThemeSettings,
};
use syntect::parsing::SyntaxSet;

use crate::theme;

pub struct Highlighter {
    syntaxes: SyntaxSet,
    theme: Theme,
}

/// Loading syntect's syntax dump costs tens of milliseconds, so it happens
/// lazily on the first rendered code fence, not at startup.
pub fn highlighter() -> &'static Highlighter {
    static HL: OnceLock<Highlighter> = OnceLock::new();
    HL.get_or_init(Highlighter::default)
}

impl Default for Highlighter {
    fn default() -> Self {
        // Keep default-syntaxes: we have no vendored language subset, so the
        // full packdump is still the honest choice for fence language coverage.
        // Theme is a single in-memory palette (no ThemeSet / default-themes dump).
        let syntaxes = SyntaxSet::load_defaults_newlines();
        let theme = code_theme();
        Self { syntaxes, theme }
    }
}

/// Compact base16-eighties-inspired palette for fenced code only.
/// Avoids embedding syntect's multi-theme `default.themedump` and the plist
/// crate that a vendored `.tmTheme` would need.
fn code_theme() -> Theme {
    let fg = rgb(0xd3, 0xd0, 0xc8);
    let comment = rgb(0x74, 0x73, 0x69);
    let red = rgb(0xf2, 0x77, 0x7a);
    let orange = rgb(0xf9, 0x91, 0x57);
    let yellow = rgb(0xff, 0xcc, 0x66);
    let green = rgb(0x99, 0xcc, 0x99);
    let cyan = rgb(0x66, 0xcc, 0xcc);
    let blue = rgb(0x66, 0x99, 0xcc);
    let magenta = rgb(0xcc, 0x99, 0xcc);

    let mut scopes = Vec::new();
    let mut rule = |selector: &str, color: SynColor| {
        scopes.push(ThemeItem {
            scope: ScopeSelectors::from_str(selector).expect("static scope selector"),
            style: StyleModifier {
                foreground: Some(color),
                background: None,
                font_style: None,
            },
        });
    };

    rule("comment, punctuation.definition.comment", comment);
    rule("string, punctuation.definition.string", green);
    rule("constant.numeric, constant.language, constant.character", orange);
    rule("keyword, storage, storage.type, storage.modifier", magenta);
    rule("entity.name.function, support.function, meta.function-call", blue);
    rule("entity.name.type, entity.name.class, support.type, support.class", yellow);
    rule("variable, variable.language, variable.parameter", red);
    rule("keyword.operator", cyan);
    rule("entity.name.tag", red);
    rule("entity.other.attribute-name", orange);

    Theme {
        name: Some("open-max-code".into()),
        author: None,
        settings: ThemeSettings {
            foreground: Some(fg),
            background: Some(rgb(0x2d, 0x2d, 0x2d)),
            ..ThemeSettings::default()
        },
        scopes,
    }
}

const fn rgb(r: u8, g: u8, b: u8) -> SynColor {
    SynColor { r, g, b, a: 0xff }
}

/// Render markdown to styled lines. Code fences are highlighted and prefixed
/// with a dim gutter bar; everything else is line-oriented markdown.
pub fn render(text: &str, hl: &Highlighter) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut in_fence = false;
    let mut highlighter: Option<HighlightLines> = None;

    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            if in_fence {
                in_fence = false;
                highlighter = None;
            } else {
                in_fence = true;
                let fence_lang = trimmed.trim_start_matches('`').trim();
                let syntax = hl
                    .syntaxes
                    .find_syntax_by_token(fence_lang)
                    .unwrap_or_else(|| hl.syntaxes.find_syntax_plain_text());
                highlighter = Some(HighlightLines::new(syntax, &hl.theme));
            }
            continue;
        }

        if in_fence {
            let mut spans = vec![Span::styled("│ ", Style::default().fg(theme::DIM()))];
            match highlighter.as_mut() {
                Some(h) => match h.highlight_line(raw, &hl.syntaxes) {
                    Ok(ranges) => {
                        for (style, piece) in ranges {
                            let fg = style.foreground;
                            spans.push(Span::styled(
                                piece.to_string(),
                                Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b)),
                            ));
                        }
                    }
                    Err(_) => spans.push(Span::raw(raw.to_string())),
                },
                None => spans.push(Span::raw(raw.to_string())),
            }
            out.push(Line::from(spans));
            continue;
        }

        // Headings.
        if let Some(rest) = strip_heading(trimmed) {
            out.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        // Horizontal rule.
        if trimmed == "---" || trimmed == "***" {
            out.push(Line::from(Span::styled("─".repeat(24), Style::default().fg(theme::DIM()))));
            continue;
        }
        // Blockquote.
        if let Some(rest) = trimmed.strip_prefix("> ") {
            let mut spans = vec![Span::styled("▎", Style::default().fg(theme::DIM()))];
            spans.extend(inline(rest, Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC)));
            out.push(Line::from(spans));
            continue;
        }
        // Bullets keep their indent.
        let indent_len = raw.len() - trimmed.len();
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            let mut spans = vec![Span::raw(" ".repeat(indent_len)), Span::styled("• ", Style::default().fg(theme::ACCENT()))];
            spans.extend(inline(&trimmed[2..], Style::default()));
            out.push(Line::from(spans));
            continue;
        }

        let mut spans = Vec::new();
        if indent_len > 0 {
            spans.push(Span::raw(" ".repeat(indent_len)));
        }
        spans.extend(inline(trimmed, Style::default()));
        out.push(Line::from(spans));
    }
    out
}

fn strip_heading(line: &str) -> Option<&str> {
    for prefix in ["#### ", "### ", "## ", "# "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

/// Inline markdown within one line: `code`, **bold**, *italic*.
fn inline(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    let flush = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), base));
        }
    };

    while i < chars.len() {
        // `code`
        if chars[i] == '`' {
            if let Some(close) = find(&chars, i + 1, "`") {
                flush(&mut buf, &mut spans);
                let code: String = chars[i + 1..close].iter().collect();
                spans.push(Span::styled(code, base.fg(theme::CODE())));
                i = close + 1;
                continue;
            }
        }
        // **bold**
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(close) = find(&chars, i + 2, "**") {
                flush(&mut buf, &mut spans);
                let inner: String = chars[i + 2..close].iter().collect();
                spans.push(Span::styled(inner, base.add_modifier(Modifier::BOLD)));
                i = close + 2;
                continue;
            }
        }
        // *italic*
        if chars[i] == '*' {
            if let Some(close) = find(&chars, i + 1, "*") {
                if close > i + 1 {
                    flush(&mut buf, &mut spans);
                    let inner: String = chars[i + 1..close].iter().collect();
                    spans.push(Span::styled(inner, base.add_modifier(Modifier::ITALIC)));
                    i = close + 1;
                    continue;
                }
            }
        }
        buf.push(chars[i]);
        i += 1;
    }
    flush(&mut buf, &mut spans);
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

/// Find `needle` starting at char index `from`; returns the char index.
fn find(chars: &[char], from: usize, needle: &str) -> Option<usize> {
    let n: Vec<char> = needle.chars().collect();
    if n.is_empty() || from >= chars.len() {
        return None;
    }
    let mut i = from;
    while i + n.len() <= chars.len() {
        if chars[i..i + n.len()] == n[..] {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    #[test]
    fn renders_headings_bullets_and_code_fence() {
        let hl = Highlighter::default();
        let lines = render("# Title\n- item one\n```rust\nfn main() {}\n```\ndone", &hl);
        let texts = plain(&lines);
        assert_eq!(texts[0], "Title");
        assert_eq!(texts[1], "• item one");
        assert_eq!(texts[2], "│ fn main() {}");
        assert_eq!(texts[3], "done");
    }

    #[test]
    fn inline_styles_do_not_lose_text() {
        let hl = Highlighter::default();
        let lines = render("mix of `code`, **bold**, *italic*, and plain", &hl);
        assert_eq!(plain(&lines)[0], "mix of code, bold, italic, and plain");
    }

    #[test]
    fn unclosed_markers_stay_literal() {
        let hl = Highlighter::default();
        let lines = render("a * lone star and `tick", &hl);
        assert_eq!(plain(&lines)[0], "a * lone star and `tick");
    }

    #[test]
    fn rust_fence_applies_syntect_colors() {
        let hl = Highlighter::default();
        let lines = render(
            "```rust\nfn main() { let x = 1; println!(\"hi\"); }\n```",
            &hl,
        );
        assert!(!lines.is_empty());
        let code = &lines[0];
        // Gutter bar + highlighted pieces; more than a single plain span.
        assert!(
            code.spans.len() > 2,
            "expected multi-span highlight, got {}",
            code.spans.len()
        );
        let texts: String = code.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(texts.contains("fn main()"));

        let fg_colors: Vec<_> = code
            .spans
            .iter()
            .skip(1) // skip gutter
            .filter_map(|s| match s.style.fg {
                Some(Color::Rgb(r, g, b)) => Some((r, g, b)),
                _ => None,
            })
            .collect();
        assert!(
            fg_colors.len() >= 2,
            "expected RGB foregrounds from syntect theme, got {fg_colors:?}"
        );
        let distinct: std::collections::HashSet<_> = fg_colors.into_iter().collect();
        assert!(
            distinct.len() >= 2,
            "expected more than one highlight color for keywords/idents, got {distinct:?}"
        );
    }
}
