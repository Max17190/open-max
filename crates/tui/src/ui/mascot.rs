//! The Open Max imp: a tiny evil-purple pixel creature that lingers in the
//! header and reacts to what the agent is doing. Pure quadrant-block art from
//! `&'static str` spans — no timers, no threads, and no background colors, so
//! it sits directly on the terminal's own background.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

use crate::theme;

/// A lighter magenta the eyes pulse toward while the agent works.
const EYE_GLOW: Color = Color::Rgb(0xff, 0x8a, 0xc4);

/// What the imp is feeling, derived from existing app state at draw time.
/// No new state machinery: moods map 1:1 onto flags the app already tracks.
#[derive(Clone, Copy, PartialEq)]
pub enum Mood {
    /// Nothing happening; calm stare with a rare blink.
    Idle,
    /// Agent turn or download in flight; eyes pulse.
    Working,
    /// Approval pending; eyes go wide.
    Waiting,
    /// An error was just pushed; eyes narrow into a glare.
    Error,
}

// Header imp, one row of four cells: two horn shoulders flanking the eyes.
//   ▙▀▀▟
const HORN_L: &str = "▙";
const HORN_R: &str = "▟";

/// The imp's single header row for a given mood and tick. The tick drives
/// the idle blink and the working pulse; both are pure functions of it, so
/// the caller decides when a redraw is worth it.
pub fn line(mood: Mood, tick: u64) -> Line<'static> {
    let horn = Style::default().fg(theme::ACCENT);
    let (eye_glyph, eye_style) = eyes(mood, tick);
    Line::from(vec![
        Span::styled(HORN_L, horn),
        Span::styled(eye_glyph, eye_style),
        Span::styled(HORN_R, horn),
    ])
}

/// Eye glyphs and style per mood. Every variant is exactly two cells wide so
/// the sprite never shifts the header text. A blink dims the eyes into the
/// horn shading instead of painting a background.
fn eyes(mood: Mood, tick: u64) -> (&'static str, Style) {
    match mood {
        Mood::Idle => {
            if tick % 32 < 2 {
                ("▀▀", Style::default().fg(theme::ACCENT_DEEP))
            } else {
                ("▀▀", Style::default().fg(theme::EYES))
            }
        }
        Mood::Working => {
            let glow = if (tick / 2).is_multiple_of(2) { theme::EYES } else { EYE_GLOW };
            ("▀▀", Style::default().fg(glow))
        }
        Mood::Waiting => ("██", Style::default().fg(theme::EYES)),
        Mood::Error => ("▄▄", Style::default().fg(theme::EYES)),
    }
}

/// One-line micro imp for inline headings (e.g. /help).
pub fn micro() -> Span<'static> {
    Span::styled("▝▙▟▘", Style::default().fg(theme::ACCENT_DEEP))
}

/// The launch splash: a larger imp beside the version and a hint line.
/// Pushed once into the transcript at startup and scrolls away naturally.
/// Eye cells are bare half-blocks — the sockets beneath them stay on the
/// terminal background.
pub fn splash(version: &str) -> Vec<Line<'static>> {
    let deep = Style::default().fg(theme::ACCENT_DEEP);
    let body = Style::default().fg(theme::ACCENT);
    let eyes = Style::default().fg(theme::EYES);
    let dim = Style::default().fg(theme::DIM);
    let title = Style::default()
        .fg(theme::ACCENT)
        .add_modifier(ratatui::style::Modifier::BOLD);

    vec![
        Line::default(),
        Line::from(vec![Span::styled("▝▙▖     ▗▟▘", deep)]),
        Line::from(vec![
            Span::styled(" ▜▙", deep),
            Span::styled("▄▄▄▄▄", body),
            Span::styled("▟▛ ", deep),
            Span::raw("   "),
            Span::styled(format!("open max v{version}"), title),
        ]),
        Line::from(vec![
            Span::styled(" ▐██", body),
            Span::styled("▀", eyes),
            Span::styled("█", body),
            Span::styled("▀", eyes),
            Span::styled("██▌ ", body),
            Span::raw("   "),
            Span::styled("a minimal harness for local models", dim),
        ]),
        Line::from(vec![
            Span::styled(" ▝▜█████▛▘ ", body),
            Span::raw("   "),
            Span::styled("/models to serve one · /help for commands", dim),
        ]),
        Line::default(),
    ]
}
