//! Neutralizing author-controlled text before it renders into a trusted,
//! line- or clause-structured surface.
//!
//! A capability file's declared `name`, its `description`, a parse-failure
//! reason quoting either, and filesystem paths are all bytes the agent or a
//! third party wrote (a cloned repo's filenames feed the layout map; its
//! `.agents/skills` and `.openmax/tools` feed the indices and their receipts).
//! Those strings are interpolated into surfaces the model and the human trust
//! to describe what is installed: the frozen system prompt (one entry per
//! line), refreeze receipts and policy notices (user-role messages the model
//! reads as the human's own words), and `openmax --check` rows (one verdict
//! per line). A raw newline, carriage return, or escape sequence in any of
//! them lets the content forge a second line, a second clause, or a terminal
//! repaint, so a file can make the harness lie about itself. That is the exact
//! failure the separation of capability from authority exists to prevent, so
//! the value has to be neutralized at the boundary, not trusted to be clean.
//!
//! This is the one place that decision lives, so every authored-text
//! interpolation can route through it and a new surface inherits the rule
//! instead of reintroducing the class.

/// Collapse every character that could break out of a single line of display
/// to one space: the C0 and C1 control characters (newline, carriage return,
/// tab, ESC, and the rest of Unicode category Cc, which `char::is_control`
/// reports), plus the Unicode line and paragraph separators U+2028 and U+2029,
/// which are line breaks to a renderer but are not control characters. Every
/// printable character and the string's length are preserved, so a validator
/// downstream (a charset or length check, e.g. the Agent Skills portability
/// warning) still sees a faithful, if flattened, value. The result is exactly
/// one line: an interpolated value can never leave its own line, its own
/// receipt clause, or its own `--check` row.
pub fn one_line(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_control() || c == '\u{2028}' || c == '\u{2029}' {
                ' '
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_line_breaking_character_becomes_a_space() {
        // The whole point is that the output is one line, so no interpolation
        // of it can forge a second line, clause, or row.
        for brk in ['\n', '\r', '\u{0085}', '\u{2028}', '\u{2029}'] {
            let out = one_line(&format!("before{brk}after"));
            assert!(!out.contains(brk), "{brk:?} survived: {out:?}");
            assert_eq!(out, "before after");
        }
    }

    #[test]
    fn escape_and_other_controls_are_flattened_too() {
        // An ESC (or any C0/C1 control) in a --check row or a receipt would
        // repaint the terminal; it must not survive either.
        let out = one_line("a\u{1b}[31mb\tc");
        assert_eq!(out, "a [31mb c");
        assert!(!out.chars().any(|c| c.is_control()));
    }

    #[test]
    fn printable_text_and_length_are_preserved() {
        // A charset or length check downstream must still see a faithful value,
        // so nothing printable is dropped and the char count is unchanged.
        let s = "code-review: runs the checks — .agents/skills/x/SKILL.md";
        assert_eq!(one_line(s), s);
        let mixed = "a\nb\rc";
        assert_eq!(one_line(mixed).chars().count(), mixed.chars().count());
    }
}
