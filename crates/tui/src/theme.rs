//! Theme tokens for the TUI. One accent identity; everything else stays
//! neutral. `/theme dark|light|mono|catppuccin` switches at runtime;
//! `NO_COLOR` forces mono.

use ratatui::style::Color;
use std::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorLevel {
    Truecolor,
    Ansi256,
    Ansi16,
    Mono,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThemeId {
    Dark,
    Light,
    Catppuccin,
}

#[derive(Clone, Copy, Debug)]
pub struct Tokens {
    pub accent: Color,
    pub dim: Color,
    pub code: Color,
    pub ok: Color,
    pub err: Color,
    pub warn: Color,
    pub border: Color,
    pub user: Color,
    pub select: Color,
    pub surface: Color,
    pub user_bg: Color,
    pub composer_bg: Color,
}

impl Tokens {
    pub fn dark() -> Self {
        Self {
            accent: Color::Rgb(0x63, 0xe0, 0xbd),
            dim: Color::DarkGray,
            code: Color::Rgb(0xd7, 0xba, 0x7d),
            ok: Color::Green,
            err: Color::Red,
            warn: Color::Yellow,
            border: Color::DarkGray,
            user: Color::Rgb(0x63, 0xe0, 0xbd),
            select: Color::Rgb(0x3a, 0x5a, 0x52),
            surface: Color::Rgb(0x20, 0x25, 0x29),
            user_bg: Color::Rgb(0x24, 0x2d, 0x2c),
            composer_bg: Color::Rgb(0x1d, 0x22, 0x25),
        }
    }

    pub fn light() -> Self {
        Self {
            accent: Color::Rgb(0x0b, 0x7a, 0x62),
            dim: Color::Gray,
            code: Color::Rgb(0x8a, 0x6d, 0x1a),
            ok: Color::Green,
            err: Color::Red,
            warn: Color::Rgb(0xb5, 0x89, 0x00),
            border: Color::Gray,
            user: Color::Rgb(0x0b, 0x7a, 0x62),
            select: Color::Rgb(0xd0, 0xec, 0xe4),
            surface: Color::Rgb(0xee, 0xf0, 0xf2),
            user_bg: Color::Rgb(0xe3, 0xf2, 0xee),
            composer_bg: Color::Rgb(0xf5, 0xf6, 0xf7),
        }
    }

    /// Catppuccin Mocha-inspired palette for a softer dark session.
    pub fn catppuccin() -> Self {
        Self {
            accent: Color::Rgb(0x89, 0xb4, 0xfa), // blue
            dim: Color::Rgb(0x6c, 0x70, 0x86),    // overlay0
            code: Color::Rgb(0xf9, 0xe2, 0xaf),   // yellow
            ok: Color::Rgb(0xa6, 0xe3, 0xa1),     // green
            err: Color::Rgb(0xf3, 0x8b, 0xa8),    // red
            warn: Color::Rgb(0xfa, 0xb3, 0x87),   // peach
            border: Color::Rgb(0x58, 0x5b, 0x70), // surface2
            user: Color::Rgb(0x89, 0xb4, 0xfa),
            select: Color::Rgb(0x45, 0x47, 0x5a),
            surface: Color::Rgb(0x1e, 0x1e, 0x2e),
            user_bg: Color::Rgb(0x25, 0x27, 0x3a),
            composer_bg: Color::Rgb(0x18, 0x18, 0x25),
        }
    }

    pub fn mono() -> Self {
        Self {
            accent: Color::White,
            dim: Color::DarkGray,
            code: Color::White,
            ok: Color::White,
            err: Color::White,
            warn: Color::White,
            border: Color::DarkGray,
            user: Color::White,
            select: Color::DarkGray,
            surface: Color::Reset,
            user_bg: Color::Reset,
            composer_bg: Color::Reset,
        }
    }
}

fn store() -> &'static RwLock<Tokens> {
    static STORE: RwLock<Tokens> = RwLock::new(Tokens {
        accent: Color::Rgb(0x63, 0xe0, 0xbd),
        dim: Color::DarkGray,
        code: Color::Rgb(0xd7, 0xba, 0x7d),
        ok: Color::Green,
        err: Color::Red,
        warn: Color::Yellow,
        border: Color::DarkGray,
        user: Color::Rgb(0x63, 0xe0, 0xbd),
        select: Color::Rgb(0x3a, 0x5a, 0x52),
        surface: Color::Rgb(0x20, 0x25, 0x29),
        user_bg: Color::Rgb(0x24, 0x2d, 0x2c),
        composer_bg: Color::Rgb(0x1d, 0x22, 0x25),
    });
    &STORE
}

pub fn set_tokens(t: Tokens) {
    if let Ok(mut g) = store().write() {
        *g = t;
    }
}

pub fn init() {
    let level = detect_color_level();
    let tokens = match level {
        ColorLevel::Mono => Tokens::mono(),
        _ => Tokens::dark(),
    };
    set_tokens(tokens);
}

pub fn detect_color_level() -> ColorLevel {
    if std::env::var_os("NO_COLOR").is_some() {
        return ColorLevel::Mono;
    }
    let colorterm = std::env::var("COLORTERM").unwrap_or_default();
    if colorterm.eq_ignore_ascii_case("truecolor") || colorterm.eq_ignore_ascii_case("24bit") {
        return ColorLevel::Truecolor;
    }
    let term = std::env::var("TERM").unwrap_or_default();
    if term.contains("256color") || term.contains("truecolor") {
        return ColorLevel::Ansi256;
    }
    if term == "dumb" || term.is_empty() {
        return ColorLevel::Ansi16;
    }
    ColorLevel::Ansi256
}

pub fn apply(id: ThemeId) {
    if matches!(detect_color_level(), ColorLevel::Mono) {
        set_tokens(Tokens::mono());
        return;
    }
    set_tokens(match id {
        ThemeId::Dark => Tokens::dark(),
        ThemeId::Light => Tokens::light(),
        ThemeId::Catppuccin => Tokens::catppuccin(),
    });
}

fn t() -> Tokens {
    store()
        .read()
        .map(|g| *g)
        .unwrap_or_else(|_| Tokens::dark())
}

// Call-site names match the old consts so a simple rename keeps working.
#[allow(non_snake_case)]
pub fn ACCENT() -> Color {
    t().accent
}
#[allow(non_snake_case)]
pub fn DIM() -> Color {
    t().dim
}
#[allow(non_snake_case)]
pub fn CODE() -> Color {
    t().code
}
#[allow(non_snake_case)]
pub fn OK() -> Color {
    t().ok
}
#[allow(non_snake_case)]
pub fn ERR() -> Color {
    t().err
}
#[allow(non_snake_case)]
pub fn WARN() -> Color {
    t().warn
}
#[allow(non_snake_case)]
pub fn BORDER() -> Color {
    t().border
}
#[allow(non_snake_case)]
pub fn USER() -> Color {
    t().user
}
#[allow(non_snake_case)]
pub fn SELECT() -> Color {
    t().select
}
#[allow(non_snake_case)]
pub fn SURFACE() -> Color {
    t().surface
}
#[allow(non_snake_case)]
pub fn USER_BG() -> Color {
    t().user_bg
}
#[allow(non_snake_case)]
pub fn COMPOSER_BG() -> Color {
    t().composer_bg
}
