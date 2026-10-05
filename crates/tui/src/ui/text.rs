//! Terminal-cell aware text primitives shared by compact TUI surfaces.

use std::borrow::Cow;

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Cells a tab occupies in the transcript. A fixed width rather than tab
/// stops: the transcript's gutters (the tool output indent, the fence rail)
/// and the line numbers `read_file` prints shift content off any stop grid,
/// which would draw the first level of indentation narrower than the rest.
pub const TAB_WIDTH: usize = 4;

pub fn width(text: &str) -> usize {
    text.width()
}

/// Cells one grapheme occupies once painted. The renderer drops any grapheme
/// holding a control character, so a tab would vanish (and Unicode width
/// counts every other control as one cell it never gets); the wrapper draws a
/// tab as [`TAB_WIDTH`] spaces instead and every other control as nothing.
pub fn cell_width(grapheme: &str) -> usize {
    if grapheme == "\t" {
        TAB_WIDTH
    } else if grapheme.contains(char::is_control) {
        0
    } else {
        grapheme.width()
    }
}

/// One line of captured program output without its complete CSI (`ESC [`,
/// colors and cursor moves), OSC (`ESC ]`, titles and hyperlinks, ended by
/// BEL or `ESC \`), and charset designation (`ESC (` and kin, as in the
/// `ESC ( B ESC [ m` a terminfo color reset emits) escape sequences. The
/// renderer drops the ESC byte and paints the rest, so a red `error` would
/// read `[31merror[0m`. An incomplete sequence is left as it is, and none
/// spans a newline: removing text the program did not clearly mark as a
/// sequence would hide output.
pub fn strip_escapes(text: &str) -> Cow<'_, str> {
    if !text.contains('\u{1b}') {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut kept = 0usize;
    let mut at = 0usize;
    while let Some(found) = text[at..].find('\u{1b}') {
        let esc = at + found;
        match escape_len(&text.as_bytes()[esc..]) {
            Some(len) => {
                out.push_str(&text[kept..esc]);
                at = esc + len;
                kept = at;
            }
            None => at = esc + 1,
        }
    }
    out.push_str(&text[kept..]);
    Cow::Owned(out)
}

/// Byte length of the complete CSI, OSC, or charset designation sequence
/// `seq` starts with (its first byte is ESC). Every byte that can end one is
/// ASCII, so the length always lands on a char boundary.
fn escape_len(seq: &[u8]) -> Option<usize> {
    match seq.get(1)? {
        // Designates a character set: exactly one final byte follows. Kept
        // this narrow because a wider escape rule would also eat an ESC
        // followed by a space and a letter.
        b'(' | b')' | b'*' | b'+' => (0x30..=0x7e).contains(seq.get(2)?).then_some(3),
        b'[' => {
            // Parameter and intermediate bytes, then one final byte.
            let end = 2 + seq[2..].iter().position(|b| !(0x20..=0x3f).contains(b))?;
            (0x40..=0x7e).contains(&seq[end]).then_some(end + 1)
        }
        b']' => {
            for (i, &b) in seq.iter().enumerate().skip(2) {
                match b {
                    0x07 => return Some(i + 1),
                    0x1b => return (seq.get(i + 1) == Some(&b'\\')).then_some(i + 2),
                    b'\n' => return None,
                    _ => {}
                }
            }
            None
        }
        _ => None,
    }
}

/// Clip to `max` terminal cells without splitting a grapheme cluster.
pub fn clip(text: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if text.width() <= max {
        return text.to_string();
    }

    let ellipsis = "…";
    let content_max = max.saturating_sub(ellipsis.width());
    let mut out = String::new();
    let mut used = 0usize;
    for grapheme in text.graphemes(true) {
        let next = grapheme.width();
        if used + next > content_max {
            break;
        }
        out.push_str(grapheme);
        used += next;
    }
    out.push_str(ellipsis);
    out
}

/// Pad to exactly `target` terminal cells after clipping.
pub fn pad_right(text: &str, target: usize) -> String {
    let mut out = clip(text, target);
    out.push_str(&" ".repeat(target.saturating_sub(out.width())));
    out
}

/// Whether `c` belongs to a word for double-click selection.
///
/// Deliberately wider than alphanumeric. What a developer double-clicks in a
/// coding transcript is a path, a flag, or a citation, so `crates/tui/src/app.rs:42`,
/// `--recall`, `snake_case`, and `~/.openmax` each have to come back whole.
/// `path:line` is the form Open Max's own recall citations use.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '~' | '@' | '+')
}

/// Char range `[start, end)` of the word containing char index `offset`.
///
/// Off a word, the run of like characters is taken instead: whitespace
/// expands over whitespace, and any other single character selects itself, so
/// a double-click always yields something rather than nothing. Never crosses a
/// newline.
pub fn word_bounds(text: &str, offset: usize) -> (usize, usize) {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return (0, 0);
    }
    let at = offset.min(chars.len() - 1);
    if chars[at] == '\n' {
        return (at, at);
    }
    let class = |c: char| {
        if is_word_char(c) {
            0
        } else if c.is_whitespace() {
            1
        } else {
            2
        }
    };
    let here = class(chars[at]);
    // A lone punctuation run is not a word; selecting the whole run would
    // swallow a whole `))));` for one click on it.
    if here == 2 {
        return (at, at + 1);
    }
    let mut start = at;
    while start > 0 && chars[start - 1] != '\n' && class(chars[start - 1]) == here {
        start -= 1;
    }
    let mut end = at + 1;
    while end < chars.len() && chars[end] != '\n' && class(chars[end]) == here {
        end += 1;
    }
    (start, end)
}

/// Char range `[start, end)` of the logical line containing `offset`, newline
/// excluded. Blocks hold their lines newline-joined, so this is the unit a
/// triple-click means even where the line wraps on screen.
pub fn line_bounds(text: &str, offset: usize) -> (usize, usize) {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return (0, 0);
    }
    let at = offset.min(chars.len() - 1);
    let mut start = at;
    while start > 0 && chars[start - 1] != '\n' {
        start -= 1;
    }
    let mut end = at;
    while end < chars.len() && chars[end] != '\n' {
        end += 1;
    }
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_obeys_cell_width_and_grapheme_boundaries() {
        assert_eq!(clip("abcdef", 4), "abc…");
        assert_eq!(clip("漢字ab", 4), "漢…");
        assert_eq!(clip("👩‍💻abc", 3), "👩‍💻…");
        assert_eq!(clip("e\u{301}abc", 2), "e\u{301}…");
        assert_eq!(clip("abc", 0), "");
        for max in 0..8 {
            assert!(width(&clip("漢👩‍💻e\u{301}abcdef", max)) <= max);
        }
    }

    /// The point of a wide word class: one click has to return the whole
    /// token a developer meant to grab.
    #[test]
    fn a_word_is_the_whole_token_a_developer_would_copy() {
        let line = "see crates/tui/src/app.rs:42 and --recall or ~/.openmax now";
        let word_at = |needle: &str, within: usize| {
            let at = line.find(needle).unwrap() + within;
            let (s, e) = word_bounds(line, at);
            line[s..e].to_string()
        };
        assert_eq!(word_at("crates", 3), "crates/tui/src/app.rs:42");
        assert_eq!(word_at("--recall", 4), "--recall");
        assert_eq!(word_at("~/.openmax", 2), "~/.openmax");
        assert_eq!(word_at("see", 1), "see");
    }

    #[test]
    fn word_bounds_never_cross_a_newline_and_always_yield_something() {
        let text = "alpha\nbeta gamma";
        // Last char of the first line stays on it.
        let (s, e) = word_bounds(text, 4);
        assert_eq!(&text[s..e], "alpha");
        // First char of the second line stays on it.
        let (s, e) = word_bounds(text, 6);
        assert_eq!(&text[s..e], "beta");
        // On whitespace, the whitespace run.
        let (s, e) = word_bounds("a   b", 2);
        assert_eq!(e - s, 3);
        // On punctuation, exactly that character, never the whole run.
        let (s, e) = word_bounds("f(x));;", 5);
        assert_eq!(e - s, 1);
        // Empty text and an out-of-range offset are not panics.
        assert_eq!(word_bounds("", 0), (0, 0));
        assert_eq!(word_bounds("ab", 99), (0, 2));
    }

    #[test]
    fn line_bounds_take_the_logical_line_without_its_newline() {
        let text = "one\ntwo three\nfour";
        let (s, e) = line_bounds(text, 5);
        assert_eq!(&text[s..e], "two three");
        let (s, e) = line_bounds(text, 0);
        assert_eq!(&text[s..e], "one");
        let (s, e) = line_bounds(text, text.len() - 1);
        assert_eq!(&text[s..e], "four");
        // An empty line selects nothing rather than panicking.
        let (s, e) = line_bounds("a\n\nb", 2);
        assert_eq!(s, e);
        assert_eq!(line_bounds("", 0), (0, 0));
    }

    #[test]
    fn strip_escapes_removes_complete_sequences_and_keeps_the_rest() {
        let plain = "no escapes\there";
        assert!(matches!(strip_escapes(plain), Cow::Borrowed(_)));
        // SGR colors, a cursor move, and private-mode parameters.
        assert_eq!(
            strip_escapes("\u{1b}[1;31merror\u{1b}[0m: \u{1b}[2Kdone\u{1b}[?25h"),
            "error: done"
        );
        // OSC 8 hyperlinks under both terminators, around non-ASCII text.
        assert_eq!(
            strip_escapes("\u{1b}]8;;https://e.x/é\u{7}café\u{1b}]8;;\u{1b}\\!"),
            "café!"
        );
        // The terminfo color reset designates the ASCII charset first.
        assert_eq!(
            strip_escapes("\u{1b}[31merror\u{1b}(B\u{1b}[m: failed"),
            "error: failed"
        );
        // Incomplete and other sequences stay: they are not clearly markup.
        for kept in [
            "cut \u{1b}[31",
            "cut \u{1b}",
            "cut \u{1b}(",
            "\u{1b} B",
            "title \u{1b}]0;never ended",
            "title \u{1b}]0;ended\non the next line\u{7}",
        ] {
            assert_eq!(strip_escapes(kept), kept);
        }
        // A broken sequence does not swallow the complete one after it.
        assert_eq!(strip_escapes("bad \u{1b}[3\u{1b}[0m"), "bad \u{1b}[3");
    }

    #[test]
    fn cell_width_matches_what_the_renderer_paints() {
        assert_eq!(cell_width("\t"), TAB_WIDTH);
        assert_eq!(cell_width("\u{1b}"), 0);
        assert_eq!(cell_width("a"), 1);
        assert_eq!(cell_width("漢"), 2);
        assert_eq!(cell_width("👩‍💻"), 2);
    }

    #[test]
    fn pad_right_uses_terminal_cells() {
        let padded = pad_right("漢", 4);
        assert_eq!(width(&padded), 4);
        assert_eq!(padded, "漢  ");
    }
}
