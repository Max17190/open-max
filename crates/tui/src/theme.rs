//! Tokens for the TUI: one monochrome palette, the terminal's own foreground
//! and background with dim gray for chrome. There is nothing to switch and
//! nothing to persist, and `NO_COLOR` is honored by construction because
//! nothing here carries a hue.

use ratatui::style::Color;

// Call-site names match the old consts so a simple rename keeps working.
#[allow(non_snake_case)]
pub fn ACCENT() -> Color {
    Color::White
}
#[allow(non_snake_case)]
pub fn DIM() -> Color {
    Color::DarkGray
}
#[allow(non_snake_case)]
pub fn CODE() -> Color {
    Color::White
}
#[allow(non_snake_case)]
pub fn OK() -> Color {
    Color::White
}
#[allow(non_snake_case)]
pub fn ERR() -> Color {
    Color::White
}
#[allow(non_snake_case)]
pub fn WARN() -> Color {
    Color::White
}
#[allow(non_snake_case)]
pub fn BORDER() -> Color {
    Color::DarkGray
}
#[allow(non_snake_case)]
pub fn USER() -> Color {
    Color::White
}
#[allow(non_snake_case)]
pub fn SELECT() -> Color {
    Color::DarkGray
}
#[allow(non_snake_case)]
pub fn SURFACE() -> Color {
    Color::Black
}
#[allow(non_snake_case)]
pub fn USER_BG() -> Color {
    Color::Black
}
#[allow(non_snake_case)]
pub fn COMPOSER_BG() -> Color {
    Color::Black
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_achromatic(color: Color) -> bool {
        match color {
            Color::Black | Color::White | Color::Gray | Color::DarkGray | Color::Reset => true,
            Color::Rgb(r, g, b) => r == g && g == b,
            _ => false,
        }
    }

    /// The one palette carries no hue, which is what makes `NO_COLOR` a
    /// no-op and every terminal background safe.
    #[test]
    fn tokens_contain_no_hue() {
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
            SURFACE(),
            USER_BG(),
            COMPOSER_BG(),
        ];
        assert!(colors.into_iter().all(is_achromatic));
        assert_eq!(SURFACE(), Color::Black);
        assert_eq!(USER_BG(), Color::Black);
        assert_eq!(COMPOSER_BG(), Color::Black);
    }
}
