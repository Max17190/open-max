//! The Open Max imp: a tiny evil-purple pixel creature that lingers in the
//! header and reacts to what the agent is doing. Pure quadrant-block art from
//! `&'static str` rows — no timers, no threads, no allocation beyond the
//! spans ratatui already builds per draw.

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

// Header imp, 7 cols x 2 rows of quadrant pixels:
//   ▝▙▄▄▄▟▘   horns curling out of the head top
//    █?█?█    face; `?` cells are the eyes, styled per mood
const HORN_L: &str = "▝▙";
const HEAD_TOP: &str = "▄▄▄";
const HORN_R: &str = "▟▘";

/// The imp's two header rows for a given mood and tick. The tick drives the
/// idle blink and the working pulse; both are pure functions of it, so the
/// caller decides when a redraw is worth it.
pub fn lines(mood: Mood, tick: u64) -> [Line<'static>; 2] {
    let deep = Style::default().fg(theme::ACCENT_DEEP);
    let body = Style::default().fg(theme::ACCENT);
    let (eye_glyph, eye_style) = eye(mood, tick);
    let top = Line::from(vec![
        Span::styled(HORN_L, deep),
        Span::styled(HEAD_TOP, body),
        Span::styled(HORN_R, deep),
    ]);
    let face = Line::from(vec![
        Span::raw(" "),
        Span::styled("█", body),
        Span::styled(eye_glyph, eye_style),
        Span::styled("█", body),
        Span::styled(eye_glyph, eye_style),
        Span::styled("█", body),
        Span::raw(" "),
    ]);
    [top, face]
}

/// Eye glyph and style per mood. Half-block eyes sit on a face-colored
/// background so the head stays solid; a blink simply melts them into it.
fn eye(mood: Mood, tick: u64) -> (&'static str, Style) {
    let face = theme::ACCENT;
    match mood {
        Mood::Idle => {
            if tick % 32 < 2 {
                ("█", Style::default().fg(face))
            } else {
                ("▀", Style::default().fg(theme::EYES).bg(face))
            }
        }
        Mood::Working => {
            let glow = if (tick / 2).is_multiple_of(2) { theme::EYES } else { EYE_GLOW };
            ("▀", Style::default().fg(glow).bg(face))
        }
        Mood::Waiting => ("█", Style::default().fg(theme::EYES)),
        Mood::Error => ("▄", Style::default().fg(theme::EYES).bg(face)),
    }
}

/// One-line micro imp for inline headings (e.g. /help).
pub fn micro() -> Span<'static> {
    Span::styled("▝▙▟▘", Style::default().fg(theme::ACCENT_DEEP))
}

/// The launch splash: a larger imp beside the version and a hint line.
/// Pushed once into the transcript at startup and scrolls away naturally.
pub fn splash(version: &str) -> Vec<Line<'static>> {
    let deep = Style::default().fg(theme::ACCENT_DEEP);
    let body = Style::default().fg(theme::ACCENT);
    let eyes = Style::default().fg(theme::EYES).bg(theme::ACCENT);
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
