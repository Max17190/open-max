//! Neutral dark, one accent — an evil purple. Everything else stays on
//! terminal defaults so Open Max looks native in any dark terminal.

use ratatui::style::Color;

/// The single brand accent: electric violet.
pub const ACCENT: Color = Color::Rgb(0xa8, 0x6e, 0xff);
/// Shadow purple: the mascot's horns and body shading.
pub const ACCENT_DEEP: Color = Color::Rgb(0x6d, 0x3f, 0xc4);
/// Hot magenta: the mascot's eyes.
pub const EYES: Color = Color::Rgb(0xff, 0x3d, 0x81);
/// De-emphasized chrome: gutters, hints, timestamps.
pub const DIM: Color = Color::DarkGray;
/// Inline code.
pub const CODE: Color = Color::Rgb(0xd7, 0xba, 0x7d);
pub const OK: Color = Color::Green;
pub const ERR: Color = Color::Red;
pub const WARN: Color = Color::Yellow;
