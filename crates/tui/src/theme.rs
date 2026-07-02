//! Neutral dark, one accent — an evil purple. Everything else stays on
//! terminal defaults so Open Max looks native in any dark terminal.

use ratatui::style::Color;

/// The single brand accent: electric violet.
pub const ACCENT: Color = Color::Rgb(0xa8, 0x6e, 0xff);
/// De-emphasized chrome: gutters, hints, timestamps.
pub const DIM: Color = Color::DarkGray;
/// Inline code.
pub const CODE: Color = Color::Rgb(0xd7, 0xba, 0x7d);
pub const OK: Color = Color::Green;
pub const ERR: Color = Color::Red;
pub const WARN: Color = Color::Yellow;
