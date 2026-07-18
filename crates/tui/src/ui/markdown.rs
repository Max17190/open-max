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
use crate::ui::transcript::wrap_lines;

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

/// Per-line markdown state carried across source lines: fence status and the
/// active syntect highlighter inside a fence. Both the batch [`render`] and the
/// incremental [`StreamingMarkdown`] drive lines through [`render_line`] with
/// this state, so their per-line output is identical by construction.
#[derive(Default)]
pub struct LineState<'a> {
    in_fence: bool,
    code: Option<HighlightLines<'a>>,
}

impl LineState<'_> {
    /// A throwaway state seeded from `in_fence` for rendering an uncommitted
    /// trailing line without disturbing (or cloning) the committed highlighter.
    fn detached(in_fence: bool) -> Self {
        Self {
            in_fence,
            code: None,
        }
    }
}

/// Render one source line, advancing `st`. Returns `None` for fence-marker
/// lines (```` ``` ````), which emit nothing but toggle fence state. A line
/// inside a fence with no active highlighter (`st.code` is `None`) renders as
/// plain code under the gutter — the streaming path uses this for the
/// still-growing partial line, which the committed highlighter colors once the
/// line ends.
pub fn render_line<'a>(raw: &str, st: &mut LineState<'a>, hl: &'a Highlighter) -> Option<Line<'static>> {
    let trimmed = raw.trim_start();
    if trimmed.starts_with("```") {
        if st.in_fence {
            st.in_fence = false;
            st.code = None;
        } else {
            st.in_fence = true;
            let fence_lang = trimmed.trim_start_matches('`').trim();
            let syntax = hl
                .syntaxes
                .find_syntax_by_token(fence_lang)
                .unwrap_or_else(|| hl.syntaxes.find_syntax_plain_text());
            st.code = Some(HighlightLines::new(syntax, &hl.theme));
        }
        return None;
    }

    if st.in_fence {
        let mut spans = vec![Span::styled("│ ", Style::default().fg(theme::DIM()))];
        match st.code.as_mut() {
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
        return Some(Line::from(spans));
    }

    // Headings.
    if let Some(rest) = strip_heading(trimmed) {
        return Some(Line::from(Span::styled(
            rest.to_string(),
            Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD),
        )));
    }
    // Horizontal rule.
    if trimmed == "---" || trimmed == "***" {
        return Some(Line::from(Span::styled(
            "─".repeat(24),
            Style::default().fg(theme::DIM()),
        )));
    }
    // Blockquote.
    if let Some(rest) = trimmed.strip_prefix("> ") {
        let mut spans = vec![Span::styled("▎", Style::default().fg(theme::DIM()))];
        spans.extend(inline(rest, Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC)));
        return Some(Line::from(spans));
    }
    // Bullets keep their indent.
    let indent_len = raw.len() - trimmed.len();
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        let mut spans = vec![
            Span::raw(" ".repeat(indent_len)),
            Span::styled("• ", Style::default().fg(theme::ACCENT())),
        ];
        spans.extend(inline(&trimmed[2..], Style::default()));
        return Some(Line::from(spans));
    }

    let mut spans = Vec::new();
    if indent_len > 0 {
        spans.push(Span::raw(" ".repeat(indent_len)));
    }
    spans.extend(inline(trimmed, Style::default()));
    Some(Line::from(spans))
}

/// Render markdown to styled lines. Code fences are highlighted and prefixed
/// with a dim gutter bar; everything else is line-oriented markdown.
pub fn render(text: &str, hl: &Highlighter) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut st = LineState::default();
    for raw in text.lines() {
        if let Some(line) = render_line(raw, &mut st, hl) {
            out.push(line);
        }
    }
    out
}

/// Incremental markdown renderer for the live assistant stream.
///
/// The batch [`render`] re-highlights the whole message on every refresh, so a
/// long streamed code block costs O(n) per newline and O(n²) over the reply —
/// the exact shape of a coding agent's output. This keeps completed source
/// lines highlighted once (append-only, syntect's `HighlightLines` carries
/// fence state across lines) and re-renders only the growing trailing line each
/// token. Highlighting is width-independent, so a resize re-wraps the cached
/// lines without re-highlighting. Committing a line yields output identical to
/// batch [`render`]; the uncommitted partial line inside a fence shows plain
/// until its newline lands.
#[derive(Default)]
pub struct StreamingMarkdown {
    width: u16,
    /// Bytes of the source consumed on the last `update`.
    text_len: usize,
    /// Bytes committed as complete lines; always at a `\n` boundary (or 0).
    committed_bytes: usize,
    /// Committed fence/highlighter state at `committed_bytes`.
    state: LineState<'static>,
    /// Highlighted, unwrapped, one per committed non-marker source line.
    complete_md: Vec<Line<'static>>,
    /// `complete_md` wrapped for `width`.
    complete_wrapped: Vec<Line<'static>>,
    /// Count of `complete_md` already folded into `complete_wrapped`.
    wrapped_md: usize,
    /// The trailing (uncommitted) line, wrapped; empty if none / a marker.
    partial_wrapped: Vec<Line<'static>>,
}

impl StreamingMarkdown {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Source length processed by the last `update`; the caller compares this
    /// against the current stream length to skip no-op refreshes.
    pub fn text_len(&self) -> usize {
        self.text_len
    }

    /// Advance to `text` at `width`. `text` is expected to grow by appends; any
    /// shrink (a new turn cleared the stream) triggers a full rebuild.
    pub fn update(&mut self, text: &str, width: u16) {
        if text.len() < self.text_len {
            self.clear();
        }
        let width_changed = width != self.width;
        self.width = width;

        // Commit any lines that ended since the last update (append-only).
        if let Some(last_nl) = text.rfind('\n') {
            let commit_end = last_nl + 1;
            if commit_end > self.committed_bytes {
                let hl = highlighter();
                let newly = &text[self.committed_bytes..commit_end];
                for raw in newly.split_inclusive('\n') {
                    let line = raw.strip_suffix('\n').unwrap_or(raw);
                    if let Some(rendered) = render_line(line, &mut self.state, hl) {
                        self.complete_md.push(rendered);
                    }
                }
                self.committed_bytes = commit_end;
            }
        }

        // Wrapping is width-dependent; highlighting is not. On resize re-wrap
        // every cached line; otherwise wrap only the freshly committed ones.
        if width_changed {
            self.complete_wrapped = wrap_lines(&self.complete_md, width);
            self.wrapped_md = self.complete_md.len();
        } else if self.wrapped_md < self.complete_md.len() {
            let fresh = wrap_lines(&self.complete_md[self.wrapped_md..], width);
            self.complete_wrapped.extend(fresh);
            self.wrapped_md = self.complete_md.len();
        }

        // The uncommitted trailing line: render on a detached state so the
        // committed highlighter is untouched (and never cloned).
        self.partial_wrapped.clear();
        let partial = &text[self.committed_bytes..];
        if !partial.is_empty() {
            let mut tmp = LineState::detached(self.state.in_fence);
            if let Some(line) = render_line(partial, &mut tmp, highlighter()) {
                self.partial_wrapped = wrap_lines(&[line], width);
            }
        }

        self.text_len = text.len();
    }

    /// Fill `out` with the current wrapped lines (clones).
    pub fn copy_into(&self, out: &mut Vec<Line<'static>>) {
        out.clear();
        out.reserve(self.complete_wrapped.len() + self.partial_wrapped.len());
        out.extend(self.complete_wrapped.iter().cloned());
        out.extend(self.partial_wrapped.iter().cloned());
    }
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

    // ---- StreamingMarkdown: incremental output must match batch render ----

    /// Text + full style per span, per line: a strong equality projection that
    /// does not depend on `Line: PartialEq`.
    fn sig(lines: &[Line]) -> Vec<Vec<(String, Style)>> {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| (s.content.to_string(), s.style))
                    .collect()
            })
            .collect()
    }

    /// Feed `text` to a fresh `StreamingMarkdown` in `chunk`-byte steps
    /// (respecting UTF-8 boundaries), mirroring token-by-token arrival.
    fn feed(text: &str, chunk: usize, width: u16) -> StreamingMarkdown {
        let mut sm = StreamingMarkdown::default();
        let mut end = 0;
        while end < text.len() {
            let mut next = (end + chunk.max(1)).min(text.len());
            while !text.is_char_boundary(next) {
                next += 1;
            }
            sm.update(&text[..next], width);
            end = next;
        }
        if text.is_empty() {
            sm.update("", width);
        }
        sm
    }

    #[test]
    fn streaming_commit_matches_batch_for_any_chunking() {
        let hl = highlighter();
        let samples = [
            "hello world this is a fairly long line that should wrap somewhere nice\n",
            "# Title\n\n- one\n- two\n\nsome **bold** and `code` and *italic* text here\n",
            "intro\n```rust\nfn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n```\ndone\n",
            "```\nplain block line one\nno language given here\n```\ntrailing prose\n",
            "> a quoted line that is long enough to wrap at a narrow width for sure\n",
        ];
        for text in samples {
            let batch = wrap_lines(&render(text, hl), 30);
            for chunk in [1usize, 3, 7, text.len()] {
                let sm = feed(text, chunk, 30);
                let mut out = Vec::new();
                sm.copy_into(&mut out);
                assert_eq!(
                    sig(&out),
                    sig(&batch),
                    "streaming != batch for text={text:?} chunk={chunk}"
                );
            }
        }
    }

    #[test]
    fn streaming_partial_code_line_is_visible_then_commits() {
        let hl = highlighter();
        let mut sm = StreamingMarkdown::default();
        // Partial code line (no trailing newline yet) must still be shown.
        sm.update("```rust\nlet x = 1;", 40);
        let mut out = Vec::new();
        sm.copy_into(&mut out);
        let joined: String = out
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(joined.contains("let x = 1;"), "partial line hidden: {joined:?}");

        // Once the newline lands, output matches batch (syntect-highlighted).
        let full = "```rust\nlet x = 1;\n";
        sm.update(full, 40);
        let mut out2 = Vec::new();
        sm.copy_into(&mut out2);
        assert_eq!(sig(&out2), sig(&wrap_lines(&render(full, hl), 40)));
    }

    #[test]
    fn streaming_resize_rewraps_and_keeps_highlight() {
        let hl = highlighter();
        let text = "```rust\nfn main() { let a = 1; let b = 2; let c = 3; done(a, b, c); }\n```\n";
        let mut sm = StreamingMarkdown::default();
        sm.update(text, 80);
        sm.update(text, 24); // narrow resize: re-wrap without re-highlighting
        let mut out = Vec::new();
        sm.copy_into(&mut out);
        assert_eq!(sig(&out), sig(&wrap_lines(&render(text, hl), 24)));
        let has_rgb = out
            .iter()
            .flat_map(|l| l.spans.iter())
            .any(|s| matches!(s.style.fg, Some(Color::Rgb(..))));
        assert!(has_rgb, "resize dropped syntect colors");
    }

    #[test]
    fn streaming_resize_midstream_then_continue_matches_batch() {
        let hl = highlighter();
        let text = "intro line\n```rust\nfn a() { one(); }\nfn b() { two(); }\nfn c() { three(); }\n```\ntrailing prose that is long enough to wrap narrowly\n";
        let mid = text.find("fn c").unwrap();
        let mut sm = StreamingMarkdown::default();
        sm.update(&text[..mid], 80); // stream some at wide width
        sm.update(&text[..mid], 30); // resize narrower mid-stream
        sm.update(text, 30); // keep streaming to completion at new width
        let mut out = Vec::new();
        sm.copy_into(&mut out);
        assert_eq!(sig(&out), sig(&wrap_lines(&render(text, hl), 30)));
    }

    #[test]
    fn streaming_clears_on_reset() {
        let mut sm = StreamingMarkdown::default();
        sm.update("hello there\n", 20);
        assert_ne!(sm.text_len(), 0);
        sm.clear();
        let mut out = Vec::new();
        sm.copy_into(&mut out);
        assert!(out.is_empty());
        assert_eq!(sm.text_len(), 0);
        // A shrink (new turn) inside update also resets cleanly.
        sm.update("longer text again\n", 20);
        sm.update("short\n", 20);
        let mut out2 = Vec::new();
        sm.copy_into(&mut out2);
        assert_eq!(sig(&out2), sig(&wrap_lines(&render("short\n", highlighter()), 20)));
    }

    // Streaming-cost comparison for a long code reply (the coding-agent hot
    // case). Not a correctness test; run with:
    //   cargo test -p open-max-tui --release -- --ignored --nocapture measure_stream
    #[test]
    #[ignore]
    fn measure_stream_render_cost() {
        use std::time::Instant;

        let hl = highlighter();
        let w: u16 = 100;
        // ~240-line rust reply, streamed one source line at a time.
        let mut reply = String::from("Here is the implementation you asked for.\n\n```rust\n");
        for i in 0..220 {
            reply.push_str(&format!(
                "    let value_{i} = compute(input_{i}, &config).map(|v| v * {i}).unwrap_or_default();\n"
            ));
        }
        reply.push_str("```\n\nThat should cover every case cleanly.\n");
        let lines: Vec<&str> = reply.split_inclusive('\n').collect();

        // OLD: full markdown re-render + re-wrap on each completed line (what
        // the `boundary` trigger did) — O(n) per line, O(n^2) over the reply.
        let mut acc = String::new();
        let t0 = Instant::now();
        for l in &lines {
            acc.push_str(l);
            let md = render(&acc, hl);
            let wrapped = wrap_lines(&md, w);
            std::hint::black_box(&wrapped);
        }
        let old_ms = t0.elapsed().as_secs_f64() * 1e3;

        // NEW: incremental — completed lines highlight once, tail re-renders.
        let mut acc = String::new();
        let mut sm = StreamingMarkdown::default();
        let mut buf = Vec::new();
        let t0 = Instant::now();
        for l in &lines {
            acc.push_str(l);
            sm.update(&acc, w);
            sm.copy_into(&mut buf);
            std::hint::black_box(&buf);
        }
        let new_ms = t0.elapsed().as_secs_f64() * 1e3;

        eprintln!("MEASURE stream_lines={}", lines.len());
        eprintln!("MEASURE old_full_rerender_ms={old_ms:.3}");
        eprintln!("MEASURE new_incremental_ms={new_ms:.3}");
        eprintln!("MEASURE speedup={:.1}x", old_ms / new_ms.max(1e-6));
    }
}
