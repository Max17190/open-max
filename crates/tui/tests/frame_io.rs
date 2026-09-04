//! Measures the frame write path: bare `Stdout` is a 1 KiB line buffer, so a
//! frame's escape stream leaves the process as many small write(2) calls; the
//! TUI's 256 KiB `BufWriter` turns the same frame into one flush.
//!
//! The fast check asserts the call-count contract against a counting sink.
//! The ignored run prints timings for realistic frame sizes:
//!   cargo test -p openmax --test frame_io -- --ignored --nocapture

use std::io::{LineWriter, Write};

/// Counts how many times the OS-facing writer is invoked. Each call models
/// one write(2) on the terminal fd.
#[derive(Default)]
struct CountingSink {
    calls: usize,
    bytes: usize,
}

impl Write for CountingSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.calls += 1;
        self.bytes += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A frame the way crossterm emits it: many small queued pieces (cursor
/// moves, SGR runs, cell text), no newlines.
fn frame_chunks(total_bytes: usize) -> Vec<Vec<u8>> {
    let piece: &[u8] = b"\x1b[38;2;200;200;200m\x1b[12;40Hstreamed cell run";
    let mut chunks = Vec::new();
    let mut emitted = 0;
    while emitted < total_bytes {
        let take = piece.len().min(total_bytes - emitted);
        chunks.push(piece[..take].to_vec());
        emitted += take;
    }
    chunks
}

fn os_writes_line_buffered(chunks: &[Vec<u8>]) -> usize {
    // Stdout's internals: a LineWriter over a 1 KiB buffer.
    let mut w = LineWriter::with_capacity(1024, CountingSink::default());
    for c in chunks {
        w.write_all(c).unwrap();
    }
    w.flush().unwrap();
    w.get_ref().calls
}

fn os_writes_frame_buffered(chunks: &[Vec<u8>]) -> usize {
    let mut w = std::io::BufWriter::with_capacity(256 * 1024, CountingSink::default());
    for c in chunks {
        w.write_all(c).unwrap();
    }
    w.flush().unwrap();
    w.get_ref().calls
}

#[test]
fn one_flush_per_frame_instead_of_one_write_per_kilobyte() {
    for frame_bytes in [4 * 1024, 32 * 1024, 128 * 1024] {
        let chunks = frame_chunks(frame_bytes);
        let line_buffered = os_writes_line_buffered(&chunks);
        let frame_buffered = os_writes_frame_buffered(&chunks);

        assert!(
            line_buffered >= frame_bytes / 1024,
            "{frame_bytes}B frame: expected ≥{} line-buffered writes, saw {line_buffered}",
            frame_bytes / 1024
        );
        assert_eq!(
            frame_buffered, 1,
            "{frame_bytes}B frame should leave in one buffered flush"
        );
    }
}

#[test]
#[ignore = "timing measurement; run with --ignored --nocapture"]
fn frame_flush_timing() {
    for frame_bytes in [4 * 1024, 32 * 1024, 128 * 1024] {
        let chunks = frame_chunks(frame_bytes);
        let devnull = || std::fs::OpenOptions::new().write(true).open("/dev/null").unwrap();
        const ROUNDS: usize = 2000;

        let started = std::time::Instant::now();
        let mut w = LineWriter::with_capacity(1024, devnull());
        for _ in 0..ROUNDS {
            for c in &chunks {
                w.write_all(c).unwrap();
            }
            w.flush().unwrap();
        }
        let line_buffered = started.elapsed();

        let started = std::time::Instant::now();
        let mut w = std::io::BufWriter::with_capacity(256 * 1024, devnull());
        for _ in 0..ROUNDS {
            for c in &chunks {
                w.write_all(c).unwrap();
            }
            w.flush().unwrap();
        }
        let frame_buffered = started.elapsed();

        println!(
            "frame {:>6}B x{ROUNDS}: line-buffered {:>8.2?}  frame-buffered {:>8.2?}  ({:.1}x)",
            frame_bytes,
            line_buffered,
            frame_buffered,
            line_buffered.as_secs_f64() / frame_buffered.as_secs_f64().max(f64::EPSILON),
        );
    }
}
