//! Tokens for the TUI: one monochrome palette, the terminal's own foreground
//! and background (`Color::Reset`) with the palette's gray (slot 8) for
//! secondary text, borders and selection. Nothing names White or Black, and
//! nothing fills a surface: slots 15 and 0 are the backgrounds of common
//! light and dark themes, where text in them vanishes and a fill in them
//! reads as a box. Rows, cards and the composer are told apart by weight,
//! glyphs and borders instead. There is nothing to switch and nothing to
//! persist, and `NO_COLOR` is honored by construction because nothing here
//! carries a hue.

use ratatui::style::Color;

// Call-site names match the old consts so a simple rename keeps working.
#[allow(non_snake_case)]
pub fn ACCENT() -> Color {
    Color::Reset
}
#[allow(non_snake_case)]
pub fn DIM() -> Color {
    Color::DarkGray
}
#[allow(non_snake_case)]
pub fn CODE() -> Color {
    Color::Reset
}
#[allow(non_snake_case)]
pub fn OK() -> Color {
    Color::Reset
}
#[allow(non_snake_case)]
pub fn ERR() -> Color {
    Color::Reset
}
#[allow(non_snake_case)]
pub fn WARN() -> Color {
    Color::Reset
}
#[allow(non_snake_case)]
pub fn BORDER() -> Color {
    Color::DarkGray
}
#[allow(non_snake_case)]
pub fn USER() -> Color {
    Color::Reset
}
#[allow(non_snake_case)]
pub fn SELECT() -> Color {
    Color::DarkGray
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Reset` follows the terminal's theme, light or dark.
    #[test]
    fn tokens_are_the_terminals_own_colors_or_the_chrome_gray() {
        let colors = [
            ACCENT(),
            DIM(),
            CODE(),
            OK(),
            ERR(),
            WARN(),
            BORDER(),
            USER(),
            SELECT(),
        ];
        for color in colors {
            assert!(matches!(color, Color::Reset | Color::DarkGray), "{color:?}");
        }
    }
}
