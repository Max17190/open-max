//! Tokens for the TUI: one monochrome palette, the terminal's own foreground
//! and background (`Color::Reset`) with the palette's gray (slot 8) for
//! secondary text, selection, and every border except the focused
//! composer's and the approval card's. Nothing names White or Black, and
//! nothing fills a surface: slots 15 and 0 are the backgrounds of common
//! light and dark themes, where text in them vanishes and a fill in them
//! reads as a box. Rows, cards and the composer are told apart by weight,
//! glyphs and borders instead. There is nothing to switch and nothing to
//! persist, and nothing here carries a hue.

use ratatui::style::{Color, Modifier, Style};

pub const ACCENT: Color = Color::Reset;
pub const DIM: Color = Color::DarkGray;
pub const CODE: Color = Color::Reset;
pub const OK: Color = Color::Reset;
pub const ERR: Color = Color::Reset;
pub const WARN: Color = Color::Reset;
pub const BORDER: Color = Color::DarkGray;
pub const USER: Color = Color::Reset;
pub const SELECT: Color = Color::DarkGray;

/// The emphasis styles the chrome repeats: placeholders and footers, the
/// selected or titled row, and the approval card's warnings.
pub const DIM_ITALIC: Style = Style::new().fg(DIM).add_modifier(Modifier::ITALIC);
pub const ACCENT_BOLD: Style = Style::new().fg(ACCENT).add_modifier(Modifier::BOLD);
pub const WARN_BOLD: Style = Style::new().fg(WARN).add_modifier(Modifier::BOLD);

#[cfg(test)]
mod tests {
    use super::*;

    /// `Reset` follows the terminal's theme, light or dark.
    #[test]
    fn tokens_are_the_terminals_own_colors_or_the_chrome_gray() {
        let colors = [ACCENT, DIM, CODE, OK, ERR, WARN, BORDER, USER, SELECT];
        for color in colors {
            assert!(matches!(color, Color::Reset | Color::DarkGray), "{color:?}");
        }
    }
}
