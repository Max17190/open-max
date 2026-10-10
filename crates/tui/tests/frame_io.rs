//! Measures the buffer under the frame write path: bare `Stdout` is a 1 KiB
//! line buffer, so a frame's escape stream leaves the process as many small
//! write(2) calls; the TUI's 256 KiB `FrameWriter` buffer turns the same frame
//! into one flush. That the real draw path then writes each frame once, with
//! ratatui's own mid-frame flushes held, is checked in `app.rs` by
//! `each_frame_reaches_the_terminal_in_one_write`.
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

/// The first frame against a real pseudo-terminal. Asking the terminal
/// whether it speaks the kitty keyboard protocol costs a round trip (a
/// network one over ssh), and a terminal that never answers leaves crossterm
/// waiting out its 2 s timeout, so the first frame must not wait on it.
#[cfg(unix)]
mod first_frame {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    const FRAME_END: &[u8] = b"\x1b[?2026l";
    const KEYBOARD_QUERY: &[u8] = b"\x1b[?u";
    const KEYBOARD_PUSH: &[u8] = b"\x1b[>1u";
    const KEYBOARD_POP: &[u8] = b"\x1b[<1u";
    const LEAVE_ALT_SCREEN: &[u8] = b"\x1b[?1049l";
    /// A terminal that supports the protocol with no flags pushed yet, then
    /// its primary device attributes, which every terminal sends.
    const ANSWER: &[u8] = b"\x1b[?0u\x1b[?62;22c";
    /// How long crossterm waits for the answer before giving up.
    const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
    const CTRL_C_TWICE: &[u8] = b"\x03\x03";

    /// openmax on a pseudo-terminal that is its controlling terminal, the way
    /// a terminal emulator starts it: crossterm reads the size through
    /// /dev/tty, which would otherwise be whatever terminal runs the tests,
    /// and a resize signals only the foreground process group of a terminal
    /// that some session controls.
    struct Session {
        controller: File,
        child: Child,
        out: Vec<u8>,
        /// (bytes read so far, when that read returned), one per read.
        reads: Vec<(usize, Instant)>,
        started: Instant,
        dir: PathBuf,
    }

    impl Session {
        fn spawn(tag: &str, cols: u16, rows: u16) -> Session {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir()
                .join(format!("openmax-first-frame-{tag}-{}-{nonce}", std::process::id()));
            let (home, project) = (dir.join("home"), dir.join("project"));
            std::fs::create_dir_all(home.join(".openmax")).unwrap();
            std::fs::create_dir_all(&project).unwrap();
            std::fs::write(
                home.join(".openmax").join("settings.json"),
                r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#,
            )
            .unwrap();
            let (controller, terminal) = pseudo_terminal();
            set_size(&terminal, cols, rows);
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_openmax"));
            cmd.arg("--trust-project")
                .current_dir(&project)
                .env("HOME", &home)
                .env("TERM", "xterm-256color")
                .env_remove("OPENMAX_API_KEY")
                .env_remove("OPENMAX_SESSION")
                // --trust-project is a human's grant; tests attest to it.
                .env("OPENMAX_HUMAN_ATTEST", "1")
                .stdin(terminal.try_clone().unwrap())
                .stdout(terminal.try_clone().unwrap())
                .stderr(Stdio::null());
            // SAFETY: setsid and ioctl are async-signal-safe, and the closure
            // touches nothing else between fork and exec.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::setsid() == -1 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            let started = Instant::now();
            let child = cmd.spawn().unwrap();
            // Only the child holds the terminal end now, so its exit reads
            // as the end of output.
            drop(cmd);
            drop(terminal);
            Session { controller, child, out: Vec::new(), reads: Vec::new(), started, dir }
        }

        /// Read what arrives within `wait`. False once the child closed the
        /// terminal.
        fn pump(&mut self, wait: Duration) -> bool {
            let mut fd = libc::pollfd {
                fd: self.controller.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one valid pollfd for the duration of the call.
            if unsafe { libc::poll(&mut fd, 1, wait.as_millis() as libc::c_int) } <= 0 {
                return true;
            }
            let mut buf = [0u8; 64 * 1024];
            match self.controller.read(&mut buf) {
                Ok(n) if n > 0 => {
                    self.out.extend_from_slice(&buf[..n]);
                    self.reads.push((self.out.len(), Instant::now()));
                    true
                }
                // EIO on Linux, EOF on macOS: the child is gone.
                _ => false,
            }
        }

        /// Read until `needle` appears at or after `from`, and return where
        /// it starts; None if `within` passes first.
        fn wait_for(&mut self, needle: &[u8], from: usize, within: Duration) -> Option<usize> {
            let deadline = Instant::now() + within;
            loop {
                if let Some(at) = find(&self.out, needle, from) {
                    return Some(at);
                }
                let left = deadline.checked_duration_since(Instant::now())?;
                if !self.pump(left.min(Duration::from_millis(20))) {
                    return find(&self.out, needle, from);
                }
            }
        }

        /// Keep reading for `wait`, so answers go out on schedule while the
        /// child keeps writing.
        fn pump_for(&mut self, wait: Duration) {
            let deadline = Instant::now() + wait;
            while let Some(left) = deadline.checked_duration_since(Instant::now()) {
                if !self.pump(left.min(Duration::from_millis(5))) {
                    return;
                }
            }
        }

        /// When the read holding the byte at `at` returned.
        fn arrived(&self, at: usize) -> Instant {
            self.reads.iter().find(|(end, _)| *end > at).expect("byte was read").1
        }

        fn write(&mut self, bytes: &[u8]) -> Instant {
            self.controller.write_all(bytes).unwrap();
            Instant::now()
        }

        /// Wait for the child to exit on its own, reading as it writes.
        fn wait_exit(&mut self, within: Duration) -> Option<std::process::ExitStatus> {
            let deadline = Instant::now() + within;
            while Instant::now() < deadline {
                if let Some(status) = self.child.try_wait().unwrap() {
                    // What it wrote on the way out is still in the terminal.
                    self.pump_for(Duration::from_millis(200));
                    return Some(status);
                }
                self.pump(Duration::from_millis(20));
            }
            None
        }

        fn context(&self) -> String {
            format!("{:?}", String::from_utf8_lossy(&self.out))
        }
    }

    impl Drop for Session {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A pseudo-terminal: the controller end the test keeps, and the terminal
    /// end the child gets. Both are opened close-on-exec, so neither leaks
    /// into a child that a test running alongside spawns.
    fn pseudo_terminal() -> (File, File) {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::OpenOptionsExt;
        // ptsname answers in one static buffer, which a test running
        // alongside would overwrite before it is copied out, and then both
        // would open the same terminal.
        static PTSNAME: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let open = |path: &std::path::Path| {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_NOCTTY)
                .open(path)
                .unwrap()
        };
        // What posix_openpt opens, but through std, which sets close-on-exec
        // atomically.
        let controller = open("/dev/ptmx".as_ref());
        let fd = controller.as_raw_fd();
        let path = {
            let _only = PTSNAME.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: fd stays open for the whole block, and ptsname's
            // buffer is copied out before the lock is released.
            unsafe {
                assert_eq!(libc::grantpt(fd), 0, "grantpt: {}", std::io::Error::last_os_error());
                assert_eq!(libc::unlockpt(fd), 0, "unlockpt: {}", std::io::Error::last_os_error());
                let name = libc::ptsname(fd);
                assert!(!name.is_null(), "ptsname: {}", std::io::Error::last_os_error());
                std::ffi::OsStr::from_bytes(std::ffi::CStr::from_ptr(name).to_bytes()).to_owned()
            }
        };
        (controller, open(path.as_ref()))
    }

    /// Set the window size; on the controller end this also signals the
    /// child the way a terminal emulator's resize does.
    fn set_size(fd: &File, cols: u16, rows: u16) {
        let size = libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
        // SAFETY: TIOCSWINSZ reads one winsize from the pointer.
        let rc = unsafe { libc::ioctl(fd.as_raw_fd(), libc::TIOCSWINSZ as _, &size) };
        assert_eq!(rc, 0, "TIOCSWINSZ: {}", std::io::Error::last_os_error());
    }

    fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
        hay.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|i| from + i)
    }

    /// The lowest row any cursor move (CSI row ; col H) in `out` reaches.
    fn lowest_row(out: &[u8]) -> u16 {
        let number = |s: &[u8]| {
            let n = s.iter().take_while(|b| b.is_ascii_digit()).count();
            (std::str::from_utf8(&s[..n]).ok().and_then(|d| d.parse::<u16>().ok()), n)
        };
        let (mut lowest, mut from) = (0, 0);
        while let Some(at) = find(out, b"\x1b[", from) {
            from = at + 2;
            let (Some(row), n) = number(&out[from..]) else { continue };
            let rest = &out[from + n..];
            if rest.first() != Some(&b';') {
                continue;
            }
            let (col, m) = number(&rest[1..]);
            if col.is_some() && rest.get(1 + m) == Some(&b'H') {
                lowest = lowest.max(row);
            }
        }
        lowest
    }

    #[test]
    fn a_terminal_that_never_answers_gets_its_first_frame_at_once() {
        let mut s = Session::spawn("silent", 80, 24);
        let frame = s.wait_for(FRAME_END, 0, Duration::from_secs(10)).unwrap_or_else(|| {
            panic!("no first frame: {}", s.context())
        });
        let painted = s.arrived(frame) - s.started;
        let query = s
            .wait_for(KEYBOARD_QUERY, 0, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no keyboard query: {}", s.context()));
        assert!(
            frame < query,
            "the first frame waited on the keyboard query (painted after {painted:?}): {}",
            s.context()
        );
        assert!(painted < PROBE_TIMEOUT / 2, "first frame after {painted:?}");

        // A resize while the probe still waits is held, not lost: once the
        // probe gives up, the screen is repainted at the new size.
        let resized = s.out.len();
        set_size(&s.controller, 100, 30);
        let deadline = Instant::now() + PROBE_TIMEOUT + Duration::from_secs(10);
        while lowest_row(&s.out[resized..]) < 30 && Instant::now() < deadline {
            s.pump(Duration::from_millis(20));
        }
        assert_eq!(lowest_row(&s.out[resized..]), 30, "not repainted: {}", s.context());
        assert!(find(&s.out, KEYBOARD_PUSH, 0).is_none(), "{}", s.context());
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
    }

    #[test]
    fn a_slow_answer_still_enables_the_protocol_and_keeps_keys_typed_meanwhile() {
        let mut s = Session::spawn("slow", 80, 24);
        let query = s
            .wait_for(KEYBOARD_QUERY, 0, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no keyboard query: {}", s.context()));
        let asked = s.arrived(query);
        // A remote terminal: the user types before the answer gets back.
        s.pump_for(Duration::from_millis(100).saturating_sub(asked.elapsed()));
        s.write(CTRL_C_TWICE);
        s.pump_for(Duration::from_millis(300).saturating_sub(asked.elapsed()));
        let answered = s.write(ANSWER);
        let answered_at = s.out.len();

        let frame = find(&s.out, FRAME_END, 0);
        assert!(
            frame.is_some_and(|f| s.arrived(f) < answered),
            "the first frame waited on the answer: {}",
            s.context()
        );
        let push = s.wait_for(KEYBOARD_PUSH, 0, Duration::from_secs(10));
        assert!(
            push.is_some_and(|p| p >= answered_at),
            "flags not pushed after the answer: {}",
            s.context()
        );
        // The keys typed before the answer reach the session: it quits.
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
        let pop = find(&s.out, KEYBOARD_POP, answered_at);
        let leave = find(&s.out, LEAVE_ALT_SCREEN, answered_at);
        assert!(
            pop.is_some() && pop < leave,
            "flags not popped on the alternate screen: {}",
            s.context()
        );
    }

    #[test]
    fn a_signal_during_the_probe_keeps_the_terminal_raw_until_the_answer() {
        let mut s = Session::spawn("signal", 80, 24);
        let query = s
            .wait_for(KEYBOARD_QUERY, 0, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no keyboard query: {}", s.context()));
        let asked = s.arrived(query);
        s.pump_for(Duration::from_millis(100).saturating_sub(asked.elapsed()));
        // SAFETY: kill only sends a signal to the child, which is unreaped.
        assert_eq!(unsafe { libc::kill(s.child.id() as libc::pid_t, libc::SIGTERM) }, 0);
        s.pump_for(Duration::from_millis(300).saturating_sub(asked.elapsed()));
        assert!(s.child.try_wait().unwrap().is_none(), "ended before the answer: {}", s.context());
        let answered_at = s.out.len();
        s.write(ANSWER);

        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
        // A cooked terminal echoes the answer (its device attributes are the
        // tail of it) and leaves it queued as input for the shell.
        assert!(
            find(&s.out, b"?62;22c", answered_at).is_none(),
            "the answer reached a restored terminal as typed text: {}",
            s.context()
        );
        let push = find(&s.out, KEYBOARD_PUSH, answered_at);
        let pop = find(&s.out, KEYBOARD_POP, answered_at);
        let leave = find(&s.out, LEAVE_ALT_SCREEN, answered_at);
        assert!(
            push.is_some() && push < pop && pop < leave,
            "flags not pushed on the answer and popped before leaving: {}",
            s.context()
        );
    }
}
