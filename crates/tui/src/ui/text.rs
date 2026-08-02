//! Terminal-cell aware text primitives shared by compact TUI surfaces.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

pub fn width(text: &str) -> usize {
    text.width()
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
    fn pad_right_uses_terminal_cells() {
        let padded = pad_right("漢", 4);
        assert_eq!(width(&padded), 4);
        assert_eq!(padded, "漢  ");
    }
}
