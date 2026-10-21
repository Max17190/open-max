//! The first-frame thesis test's baseline: a process that writes the end of
//! a frame, the close of a synchronized update, to its terminal and exits.
//! `cargo test` builds it beside `openmax`, with the same profile, so the
//! test times both on the same pseudo-terminal and divides the first frame
//! by this, which leaves out what the machine spends starting any release
//! binary and hearing from it: see `first_frame_holds_its_ratio_to_a_baseline_binary`
//! in crates/tui/tests/thesis.rs.

use std::io::Write;

fn main() {
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(b"\x1b[?2026l");
    let _ = stdout.flush();
}
