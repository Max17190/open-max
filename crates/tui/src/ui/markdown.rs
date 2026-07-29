//! A deliberately small markdown renderer producing ratatui Lines: headings,
//! emphasis, inline code, lists, blockquotes, rules, and fenced code behind a
//! dim gutter. Enough for model output without pulling in a full parser.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;
use crate::ui::transcript::wrap_lines;

/// Per-line markdown state carried across source lines: fence status. Both the
/// batch [`render`] and the incremental [`StreamingMarkdown`] drive lines
/// through [`render_line`] with this state, so their per-line output is
/// identical by construction.
#[derive(Default)]
pub struct LineState {
    in_fence: bool,
}

impl LineState {
    /// A throwaway state seeded from `in_fence` for rendering an uncommitted
    /// trailing line without disturbing the committed state.
    fn detached(in_fence: bool) -> Self {
        Self { in_fence }
    }
}

/// Render one source line, advancing `st`. Returns `None` for fence-marker
/// lines (```` ``` ````), which emit nothing but toggle fence state. Lines
/// inside a fence render as plain code behind a dim gutter bar.
pub fn render_line(raw: &str, st: &mut LineState) -> Option<Line<'static>> {
    let trimmed = raw.trim_start();
    if trimmed.starts_with("```") {
        st.in_fence = !st.in_fence;
        return None;
    }

    if st.in_fence {
        return Some(Line::from(vec![
            Span::styled("│ ", Style::default().fg(theme::DIM())),
            Span::styled(raw.to_string(), Style::default().fg(theme::CODE())),
        ]));
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

/// Render markdown to styled lines. Code fences render behind a dim gutter
/// bar; everything else is line-oriented markdown.
pub fn render(text: &str) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut st = LineState::default();
    for raw in text.lines() {
        if let Some(line) = render_line(raw, &mut st) {
            out.push(line);
        }
    }
    out
}

/// Incremental markdown renderer for the live assistant stream.
///
/// The batch [`render`] re-renders the whole message on every refresh, so a
/// long streamed reply costs O(n) per newline and O(n²) over the reply — the
/// exact shape of a coding agent's output. This renders completed source lines
/// once (append-only, fence state carries across lines) and re-renders only
/// the growing trailing line each token. Rendering is width-independent, so a
/// resize re-wraps the cached lines without re-rendering. Committing a line
/// yields output identical to batch [`render`].
#[derive(Default)]
pub struct StreamingMarkdown {
    width: u16,
    /// Bytes of the source consumed on the last `update`.
    text_len: usize,
    /// Bytes committed as complete lines; always at a `\n` boundary (or 0).
    committed_bytes: usize,
    /// Committed fence state at `committed_bytes`.
    state: LineState,
    /// Rendered, unwrapped, one per committed non-marker source line.
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
        self.commit_complete_lines(text);

        // Wrapping is width-dependent; rendering is not. On resize re-wrap
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
        // committed state is untouched.
        self.partial_wrapped.clear();
        let partial = &text[self.committed_bytes..];
        if !partial.is_empty() {
            let mut tmp = LineState::detached(self.state.in_fence);
            if let Some(line) = render_line(partial, &mut tmp) {
                self.partial_wrapped = wrap_lines(&[line], width);
            }
        }

        self.text_len = text.len();
    }

    fn commit_complete_lines(&mut self, text: &str) {
        // Search only the uncommitted suffix. Completed lines can no longer
        // change, so the scan bound is independent of accumulated history.
        let uncommitted = &text[self.committed_bytes..];
        let Some(last_nl) = uncommitted.rfind('\n') else {
            return;
        };
        let commit_end = self.committed_bytes + last_nl + 1;

        let newly = &text[self.committed_bytes..commit_end];
        for raw in newly.split_inclusive('\n') {
            let line = raw
                .strip_suffix('\n')
                .map(|line| line.strip_suffix('\r').unwrap_or(line))
                .unwrap_or(raw);
            if let Some(rendered) = render_line(line, &mut self.state) {
                self.complete_md.push(rendered);
            }
        }
        self.committed_bytes = commit_end;
    }

    /// Synchronize `out` while preserving its already-copied complete prefix.
    ///
    /// `stable_complete` is the value returned by the previous call at the
    /// same width. Completed lines are append-only, so only newly completed
    /// lines and the changing partial tail need cloning. Pass `0` after a
    /// resize or source reset.
    pub fn sync_into(
        &self,
        out: &mut Vec<Line<'static>>,
        stable_complete: usize,
    ) -> usize {
        let stable_complete = stable_complete
            .min(self.complete_wrapped.len())
            .min(out.len());
        out.truncate(stable_complete);
        out.reserve(
            self.complete_wrapped.len() - stable_complete + self.partial_wrapped.len(),
        );
        out.extend(self.complete_wrapped[stable_complete..].iter().cloned());
        let complete = self.complete_wrapped.len();
        out.extend(self.partial_wrapped.iter().cloned());
        complete
    }

    /// Promote the current stream into an unwrapped finished message.
    ///
    /// `text` must be the exact append-only source used to build this stream.
    /// The caller compares its token buffer with the provider's final message
    /// before taking this path. A final delta may arrive after the last paint,
    /// so catch up here, then commit the trailing partial line with the real
    /// fence state instead of re-rendering the whole response.
    pub fn finish(&mut self, text: &str) -> Vec<Line<'static>> {
        if text.len() < self.text_len {
            self.clear();
        }
        self.commit_complete_lines(text);

        let partial = &text[self.committed_bytes..];
        if !partial.is_empty() {
            if let Some(line) = render_line(partial, &mut self.state) {
                self.complete_md.push(line);
            }
        }

        let finished = std::mem::take(&mut self.complete_md);
        self.clear();
        finished
    }

    /// Fill `out` from scratch. Tests and one-shot callers use this; the live
    /// TUI uses [`Self::sync_into`] to avoid cloning stable history per token.
    #[cfg(test)]
    pub fn copy_into(&self, out: &mut Vec<Line<'static>>) {
        out.clear();
        let _ = self.sync_into(out, 0);
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
                spans.push(Span::styled(
                    code,
                    base.fg(theme::CODE()).add_modifier(Modifier::REVERSED),
                ));
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
        let lines = render("# Title\n- item one\n```rust\nfn main() {}\n```\ndone");
        let texts = plain(&lines);
        assert_eq!(texts[0], "Title");
        assert_eq!(texts[1], "• item one");
        assert_eq!(texts[2], "│ fn main() {}");
        assert_eq!(texts[3], "done");
    }

    #[test]
    fn inline_styles_do_not_lose_text() {
        let lines = render("mix of `code`, **bold**, *italic*, and plain");
        assert_eq!(plain(&lines)[0], "mix of code, bold, italic, and plain");
        assert!(lines[0]
            .spans
            .iter()
            .find(|span| span.content == "code")
            .unwrap()
            .style
            .add_modifier
            .contains(Modifier::REVERSED));
    }

    #[test]
    fn unclosed_markers_stay_literal() {
        let lines = render("a * lone star and `tick");
        assert_eq!(plain(&lines)[0], "a * lone star and `tick");
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
        let samples = [
            "hello world this is a fairly long line that should wrap somewhere nice\n",
            "windows first line\r\nwindows second line\r\n",
            "# Title\n\n- one\n- two\n\nsome **bold** and `code` and *italic* text here\n",
            "intro\n```rust\nfn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n```\ndone\n",
            "```\nplain block line one\nno language given here\n```\ntrailing prose\n",
            "> a quoted line that is long enough to wrap at a narrow width for sure\n",
        ];
        for text in samples {
            let batch = wrap_lines(&render(text), 30);
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

        // Once the newline lands, output matches batch.
        let full = "```rust\nlet x = 1;\n";
        sm.update(full, 40);
        let mut out2 = Vec::new();
        sm.copy_into(&mut out2);
        assert_eq!(sig(&out2), sig(&wrap_lines(&render(full), 40)));
    }

    #[test]
    fn streaming_sync_reuses_complete_prefix_and_replaces_partial_suffix() {
        let mut sm = StreamingMarkdown::default();
        let mut out = Vec::new();

        sm.update("first complete line\npartial", 40);
        let stable = sm.sync_into(&mut out, 0);
        assert_eq!(stable, 1);
        let complete_prefix = sig(&out[..stable]);

        sm.update("first complete line\npartial grows", 40);
        let stable = sm.sync_into(&mut out, stable);
        assert_eq!(stable, 1);
        assert_eq!(sig(&out[..stable]), complete_prefix);
        assert_eq!(
            sig(&out),
            sig(&wrap_lines(&render("first complete line\npartial grows"), 40)),
        );

        sm.update("first complete line\npartial grows\nnext", 40);
        let stable = sm.sync_into(&mut out, stable);
        assert_eq!(stable, 2);
        assert_eq!(
            sig(&out),
            sig(&wrap_lines(&render("first complete line\npartial grows\nnext"), 40)),
        );
    }

    #[test]
    fn streaming_resize_rewraps_and_matches_batch() {
        let text = "```rust\nfn main() { let a = 1; let b = 2; let c = 3; done(a, b, c); }\n```\n";
        let mut sm = StreamingMarkdown::default();
        sm.update(text, 80);
        sm.update(text, 24); // narrow resize: re-wrap the cached lines
        let mut out = Vec::new();
        sm.copy_into(&mut out);
        assert_eq!(sig(&out), sig(&wrap_lines(&render(text), 24)));
    }

    #[test]
    fn streaming_resize_midstream_then_continue_matches_batch() {
        let text = "intro line\n```rust\nfn a() { one(); }\nfn b() { two(); }\nfn c() { three(); }\n```\ntrailing prose that is long enough to wrap narrowly\n";
        let mid = text.find("fn c").unwrap();
        let mut sm = StreamingMarkdown::default();
        sm.update(&text[..mid], 80); // stream some at wide width
        sm.update(&text[..mid], 30); // resize narrower mid-stream
        sm.update(text, 30); // keep streaming to completion at new width
        let mut out = Vec::new();
        sm.copy_into(&mut out);
        assert_eq!(sig(&out), sig(&wrap_lines(&render(text), 30)));
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
        assert_eq!(sig(&out2), sig(&wrap_lines(&render("short\n"), 20)));
    }

    #[test]
    fn finishing_promotes_incremental_markdown_and_commits_the_unpainted_tail() {
        let complete = "intro\n```rust\nfn main() {\n    println!(\"done\");\n}\n```\nfinal prose";
        let painted = complete.find("    println!").unwrap();
        let mut sm = StreamingMarkdown::default();
        sm.update(&complete[..painted], 24);

        let finished = sm.finish(complete);

        assert_eq!(sig(&finished), sig(&render(complete)));
        assert_eq!(sm.text_len(), 0);
        let mut stream = Vec::new();
        sm.copy_into(&mut stream);
        assert!(stream.is_empty());
    }

    // Streaming-cost comparison for a long code reply (the coding-agent hot
    // case). Not a correctness test; run with:
    //   cargo test -p open-max-tui --release -- --ignored --nocapture measure_stream
    #[test]
    #[ignore]
    fn measure_stream_render_cost() {
        use std::time::Instant;

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
        // Provider deltas are much smaller than source lines. This fixture is
        // ASCII, so byte chunks are also valid UTF-8 boundaries.
        let deltas: Vec<&str> = reply
            .as_bytes()
            .chunks(8)
            .map(|chunk| std::str::from_utf8(chunk).unwrap())
            .collect();

        // OLD: full markdown re-render + re-wrap on each completed line (what
        // the `boundary` trigger did) — O(n) per line, O(n^2) over the reply.
        let mut acc = String::new();
        let t0 = Instant::now();
        for l in &lines {
            acc.push_str(l);
            let md = render(&acc);
            let wrapped = wrap_lines(&md, w);
            std::hint::black_box(&wrapped);
        }
        let old_ms = t0.elapsed().as_secs_f64() * 1e3;

        // PREVIOUS: highlighting is incremental, but every update clones the
        // complete accumulated output into the destination buffer.
        let mut acc = String::new();
        let mut sm = StreamingMarkdown::default();
        let mut buf = Vec::new();
        let t0 = Instant::now();
        for delta in &deltas {
            acc.push_str(delta);
            sm.update(&acc, w);
            sm.copy_into(&mut buf);
            std::hint::black_box(&buf);
        }
        let clone_all_ms = t0.elapsed().as_secs_f64() * 1e3;

        // NEW: preserve the completed prefix and replace only the changing
        // partial suffix, matching the live TUI buffer path.
        let mut acc = String::new();
        let mut sm = StreamingMarkdown::default();
        let mut buf = Vec::new();
        let mut stable = 0;
        let t0 = Instant::now();
        for delta in &deltas {
            acc.push_str(delta);
            sm.update(&acc, w);
            stable = sm.sync_into(&mut buf, stable);
            std::hint::black_box(&buf);
        }
        let suffix_sync_ms = t0.elapsed().as_secs_f64() * 1e3;

        // OLD completion: discard the incremental state and highlight the
        // finished response from scratch.
        let t0 = Instant::now();
        let batch_finished = render(&reply);
        let batch_finish_ms = t0.elapsed().as_secs_f64() * 1e3;

        // NEW completion: promote the already-highlighted stream and commit
        // only a possible trailing partial line. Include the exact source
        // comparison that guards the live promotion path.
        let provider_finished = reply.clone();
        let t0 = Instant::now();
        assert_eq!(
            std::hint::black_box(reply.as_str()),
            std::hint::black_box(provider_finished.as_str())
        );
        let promoted = sm.finish(&provider_finished);
        let promote_finish_ms = t0.elapsed().as_secs_f64() * 1e3;
        assert_eq!(sig(&promoted), sig(&batch_finished));

        eprintln!("MEASURE stream_lines={}", lines.len());
        eprintln!("MEASURE stream_deltas={}", deltas.len());
        eprintln!("MEASURE old_full_rerender_ms={old_ms:.3}");
        eprintln!("MEASURE incremental_clone_all_ms={clone_all_ms:.3}");
        eprintln!("MEASURE incremental_suffix_sync_ms={suffix_sync_ms:.3}");
        eprintln!("MEASURE completion_batch_render_ms={batch_finish_ms:.3}");
        eprintln!("MEASURE completion_promote_ms={promote_finish_ms:.3}");
        eprintln!(
            "MEASURE suffix_sync_speedup={:.1}x",
            clone_all_ms / suffix_sync_ms.max(1e-6)
        );
        eprintln!(
            "MEASURE total_speedup={:.1}x",
            old_ms / suffix_sync_ms.max(1e-6)
        );
        eprintln!(
            "MEASURE completion_speedup={:.1}x",
            batch_finish_ms / promote_finish_ms.max(1e-6)
        );
    }
}
