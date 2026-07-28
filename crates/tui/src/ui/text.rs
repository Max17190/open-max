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

    #[test]
    fn pad_right_uses_terminal_cells() {
        let padded = pad_right("漢", 4);
        assert_eq!(width(&padded), 4);
        assert_eq!(padded, "漢  ");
    }
}
