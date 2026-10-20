//! Instrument for the transcript repaint metric: frame time with 100,000
//! transcript lines live. It sits under `app` rather than in `tests/`
//! because the binary has no library target, and a frame needs the `App`
//! the draw path renders.

use std::time::Instant;

use ratatui::backend::TestBackend;
use ratatui::Terminal;

use super::tests::{app_fixture, push_long_history};
use super::App;

const LINES: usize = 100_000;
/// Blocks pushed per fill step; the fill stops at the first step that
/// reaches `LINES` rows.
const FILL_STEP: usize = 1_000;
/// Frames timed per phase: 1,000 in release, where each p99 is held to the
/// target; 101 in debug, where only each median is held to a sanity bound,
/// so the debug suite stays fast.
const FRAMES: usize = if cfg!(debug_assertions) { 101 } else { 1_000 };
const WIDTH: u16 = 140;
const HEIGHT: u16 = 45;

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Transcript frame time at 100,000 lines, idle and while scrolling.
///
/// Method: an `App` on a 140x45 `TestBackend` holds a long mixed history
/// (user prompts, wrapping markdown replies with a list and a code fence,
/// folded and failed tool cards, notices), pushed 1,000 blocks at a time
/// until the transcript counts at least 100,000 rows. A frame is drawn
/// before each count, since a transcript has no width, and so counts no
/// rows, until a draw sizes it. That count is the transcript's own at the
/// drawn width, a lower bound, since a block not yet wrapped is estimated
/// at no more rows than its wrap. After 50 warmup frames, 1,000 frames
/// (101 in debug) are timed with the transcript marked dirty and the view
/// still (idle), then as many that each move the view one row toward older
/// output and draw (scrolling). A scrolling sample includes `scroll_up`,
/// which wraps the rows it crosses, so blocks entering the view pay their
/// wrap inside the frame that shows them. Each sample is that plus one
/// `Terminal::draw` of `App::draw`, the buffer diff included and no
/// terminal write. The test binary installs a counting global allocator,
/// so every sample also pays a counter per allocation, which leans the
/// numbers high. Prints n, median and p99 per phase.
///
/// Target 0.5 ms. Measured on this tree on an Apple-silicon laptop with
/// other builds running, where the fill takes 25,000 blocks for 100,000
/// rows. Five release runs: idle median 0.073 to 0.076 ms, p99 0.078 to
/// 0.204 ms; scrolling median 0.082 to 0.084 ms, p99 0.091 to 0.145 ms.
/// Debug, 101 frames inside the parallel unit suite: idle median 0.837 ms,
/// p99 0.958 ms; scrolling median 0.878 ms, p99 1.028 ms. A release build
/// asserts the target on each phase's p99. A debug build asserts a 20x
/// sanity bound on each phase's median, not its p99: over 101 frames the
/// p99 is the second-slowest frame, which two scheduler preemptions in the
/// parallel unit suite would decide.
#[test]
fn transcript_frame_p99_at_100k_lines() {
    const TARGET_MS: f64 = 0.5;
    let (mut app, dir) = app_fixture();
    let mut terminal = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
    // A transcript counts no rows until a draw gives it a width, so every
    // fill step draws before the next count.
    let mut blocks = 0;
    loop {
        terminal.draw(|f| app.draw(f)).unwrap();
        if app.transcript.len() >= LINES {
            break;
        }
        push_long_history(&mut app, FILL_STEP);
        blocks += FILL_STEP;
    }
    let rows = app.transcript.len();
    let mut frame = |app: &mut App, scroll: bool| {
        let started = Instant::now();
        if scroll {
            app.transcript.scroll_up(1);
        }
        app.dirty.mark_chat();
        terminal.draw(|f| app.draw(f)).unwrap();
        started.elapsed().as_secs_f64() * 1e3
    };
    for _ in 0..50 {
        frame(&mut app, false);
    }
    let mut idle: Vec<f64> = (0..FRAMES).map(|_| frame(&mut app, false)).collect();
    let mut scrolling: Vec<f64> = (0..FRAMES).map(|_| frame(&mut app, true)).collect();
    assert_eq!(app.transcript.offset(), FRAMES, "every scrolling frame moved the view one row");
    let build = if cfg!(debug_assertions) { "debug" } else { "release" };
    let mut stats = Vec::new();
    for (phase, samples) in [("idle", &mut idle), ("scrolling", &mut scrolling)] {
        samples.sort_by(f64::total_cmp);
        let (median, p99) = (quantile(samples, 0.5), quantile(samples, 0.99));
        println!(
            "MEASURE transcript_frame_p99_at_100k_lines {phase} n={} median={median:.3}ms p99={p99:.3}ms ({blocks} blocks, {rows} rows; target {TARGET_MS}ms, {build})",
            samples.len()
        );
        stats.push((phase, median, p99));
    }
    for (phase, median, p99) in stats {
        if cfg!(debug_assertions) {
            assert!(median < TARGET_MS * 20.0, "debug {phase} frame median {median:.3}ms is past the 20x sanity bound");
        } else {
            assert!(p99 <= TARGET_MS, "{phase} frame p99 {p99:.3}ms at {rows} lines is past the {TARGET_MS}ms target");
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}
