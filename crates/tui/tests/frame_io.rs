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
//!
//! `first_frame` then runs the real binary on a pseudo-terminal: the first
//! frame, its colors, and what a turn or `/reload` leaves on screen.

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

/// openmax against a real pseudo-terminal: the first frame, its colors, and
/// what a turn or `/reload` leaves on screen. Asking the terminal whether it
/// speaks the kitty keyboard protocol costs a round trip (a network one over
/// ssh), and a terminal that never answers leaves crossterm waiting out its
/// 2 s timeout, so the first frame must not wait on it. The colors it selects
/// are checked as the bytes a terminal receives. What a turn or `/reload`
/// leaves is read from `screen`, the grid those bytes paint, with a stub
/// completions endpoint (`endpoint`) answering the turn.
#[cfg(unix)]
mod first_frame {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
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
            Session::spawn_with(
                tag,
                cols,
                rows,
                r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#,
            )
        }

        fn spawn_with(tag: &str, cols: u16, rows: u16, settings: &str) -> Session {
            Session::spawn_home(tag, cols, rows, |dir, cmd, _, _| {
                std::fs::write(dir.join("settings.json"), settings).unwrap();
                cmd.arg("--trust-project");
            })
        }

        /// `spawn` with `~/.openmax` (the `Path`), the command's arguments
        /// and its stderr set up by `setup` instead. The `File`s are the
        /// child's terminal end and the controller, where what is written
        /// before the spawn waits as input typed ahead.
        fn spawn_home(
            tag: &str,
            cols: u16,
            rows: u16,
            setup: impl FnOnce(&Path, &mut Command, &File, &mut File),
        ) -> Session {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir()
                .join(format!("openmax-first-frame-{tag}-{}-{nonce}", std::process::id()));
            let (home, project) = (dir.join("home"), dir.join("project"));
            std::fs::create_dir_all(home.join(".openmax")).unwrap();
            std::fs::create_dir_all(&project).unwrap();
            let (mut controller, terminal) = pseudo_terminal();
            set_size(&terminal, cols, rows);
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_openmax"));
            cmd.current_dir(&project)
                .env("HOME", &home)
                .env("TERM", "xterm-256color")
                .env_remove("OPENMAX_API_KEY")
                .env_remove("OPENMAX_SESSION")
                // --trust-project is a human's grant; tests attest to it.
                .env("OPENMAX_HUMAN_ATTEST", "1")
                .stdin(terminal.try_clone().unwrap())
                .stdout(terminal.try_clone().unwrap())
                .stderr(Stdio::null());
            setup(&home.join(".openmax"), &mut cmd, &terminal, &mut controller);
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

        /// A title can arrive in an earlier read than the editable prompt.
        /// Match its footer after that title, never one from an older redraw.
        fn wait_for_prompt(&mut self, title: &[u8], from: usize, within: Duration) -> Option<usize> {
            let deadline = Instant::now() + within;
            let at = self.wait_for(title, from, within)?;
            let left = deadline.checked_duration_since(Instant::now())?;
            self.wait_for(b"Enter confirms; Esc cancels\r\n", at + title.len(), left)?;
            Some(at)
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

        /// Keep large paste writes bounded even when a failing child stops reading stdin.
        fn write_bounded(&mut self, mut bytes: &[u8]) {
            struct RestoreFlags(libc::c_int, libc::c_int);
            impl Drop for RestoreFlags {
                fn drop(&mut self) { unsafe { libc::fcntl(self.0, libc::F_SETFL, self.1); } }
            }
            let fd = self.controller.as_raw_fd();
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            assert!(flags >= 0);
            let _restore = RestoreFlags(fd, flags);
            assert_eq!(unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) }, 0);
            let deadline = Instant::now() + Duration::from_secs(5);
            while !bytes.is_empty() {
                assert!(Instant::now() < deadline, "setup stopped consuming pasted input");
                assert!(self.child.try_wait().unwrap().is_none(), "setup exited while receiving pasted input");
                match self.controller.write(bytes) {
                    Ok(0) => panic!("terminal stopped accepting pasted input"),
                    Ok(n) => bytes = &bytes[n..],
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(self.pump(Duration::from_millis(5)), "setup terminal closed during paste");
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => panic!("terminal paste failed: {error}"),
                }
            }
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
            // A killed child cannot finish exiting while output it wrote sits
            // unread in its terminal, so a test that panics before reading
            // everything would hang in `wait` instead of failing. Drain until
            // it is reaped or the terminal closes, bounded.
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.child.try_wait().ok().flatten().is_none() && Instant::now() < deadline {
                if !self.pump(Duration::from_millis(20)) {
                    break;
                }
            }
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

    /// The parameters of every SGR in `out` that selects a foreground or
    /// background other than the terminal's own (39, 49) or palette slot 8,
    /// in the 16-color, 256-color and truecolor forms a terminal backend
    /// writes.
    fn foreign_colors(out: &[u8]) -> Vec<String> {
        let mut hits = Vec::new();
        let mut from = 0;
        while let Some(at) = find(out, b"\x1b[", from) {
            from = at + 2;
            let len = out[from..].iter().take_while(|b| b.is_ascii_digit() || **b == b';').count();
            if out.get(from + len) != Some(&b'm') {
                continue;
            }
            let params = std::str::from_utf8(&out[from..from + len]).unwrap();
            let p: Vec<u16> = params.split(';').map(|n| n.parse().unwrap_or(0)).collect();
            let mut i = 0;
            while i < p.len() {
                let foreign = match (p[i], p.get(i + 1)) {
                    (38 | 48, Some(5)) => {
                        i += 2;
                        p.get(i) != Some(&8)
                    }
                    (38 | 48, Some(2)) => {
                        i += 4;
                        true
                    }
                    // 90 and 100 are slot 8 in its 16-color form.
                    (30..=37 | 40..=47 | 91..=97 | 101..=107, _) => true,
                    _ => false,
                };
                if foreign {
                    hits.push(params.to_string());
                }
                i += 1;
            }
        }
        hits
    }

    #[test]
    fn foreign_colors_allow_only_the_terminals_own_colors_and_slot_8() {
        for own in ["39;49", "38;5;8;48;5;8", "90", "100", "0;1;22"] {
            let sgr = format!("\x1b[{own}m");
            assert!(foreign_colors(sgr.as_bytes()).is_empty(), "{own}");
        }
        for foreign in ["38;5;15", "1;48;5;0", "97", "40", "31", "38;2;8;8;8"] {
            let sgr = format!("\x1b[{foreign}m");
            assert_eq!(foreign_colors(sgr.as_bytes()), vec![foreign], "{foreign}");
        }
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

    /// A named pipe at `path` that nothing writes yet, so whoever opens it to
    /// read waits; `serve` lets the first reader through.
    fn unwritten(path: &Path) {
        use std::os::unix::ffi::OsStrExt;
        let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: mkfifo reads one NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0, "{}", std::io::Error::last_os_error());
    }

    /// Give the first reader of the pipe at `path` `content`, and put a file
    /// holding it in the pipe's place before that reader can see the end, so
    /// every later read gets the file.
    fn serve(path: PathBuf, content: String) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut pipe = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            pipe.write_all(content.as_bytes()).unwrap();
            let file = path.with_extension("served");
            std::fs::write(&file, &content).unwrap();
            std::fs::rename(&file, &path).unwrap();
        })
    }

    /// The first frame waits on no file. With trust.json and settings.json
    /// pipes that nothing writes yet, reading either waits, and the frame
    /// still arrives: the project path, the empty conversation and the
    /// composer. Once they are written, both are read behind it, and the
    /// session goes on to show the model the settings name and the mode the
    /// trust records.
    #[test]
    fn the_first_frame_waits_on_neither_trust_nor_settings() {
        let mut s = Session::spawn_home("unread", 80, 24, |dir, cmd, _, _| {
            unwritten(&dir.join("trust.json"));
            unwritten(&dir.join("settings.json"));
            cmd.arg("--trust-project");
        });
        let frame = s.wait_for(FRAME_END, 0, Duration::from_secs(10)).unwrap_or_else(|| {
            panic!("no first frame while trust.json and settings.json are unread: {}", s.context())
        });
        let shown = screen(&s.out[..frame + FRAME_END.len()], 80, 24).join("\n");
        assert!(shown.contains("project") && shown.contains("READY") && shown.contains("❯"), "{shown}");

        let dir = s.dir.join("home/.openmax");
        let root = std::fs::canonicalize(s.dir.join("project")).unwrap();
        let trust = serde_json::json!({"version": 1, "projects": [root], "approval_modes": {root.to_str().unwrap(): "auto"}});
        let served = [
            serve(dir.join("trust.json"), trust.to_string()),
            serve(
                dir.join("settings.json"),
                r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#.into(),
            ),
        ];
        let shown = screen_when(&mut s, 80, 24, Duration::from_secs(10), |shown| {
            shown.contains("stub-model  auto")
        });
        assert!(shown.contains("stub-model  auto"), "{shown}\n{}", s.context());
        s.write(ANSWER);
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
        for server in served {
            server.join().unwrap();
        }
    }

    /// A project that is not trusted is asked about after the first frame, on
    /// the shell: the session paints, hands the terminal back and asks there,
    /// and on `y` takes the terminal again, paints the whole screen, which
    /// entering it again blanked, and goes on in auto. Ctrl+C at the question
    /// ends the run at once, as at any shell prompt, with nothing trusted. A
    /// `y` typed before the question shows answers it, as when the question
    /// came first: the session reads no input before the question does, so
    /// the `y` is neither lost nor left in the prompt's draft.
    #[test]
    fn the_trust_question_comes_after_the_first_frame_on_the_shell() {
        let asked = |tag: &str, typed_ahead: &[u8]| {
            let mut s = Session::spawn_home(tag, 80, 24, |dir, cmd, terminal, controller| {
                std::fs::write(
                    dir.join("settings.json"),
                    r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#,
                )
                .unwrap();
                cmd.stderr(terminal.try_clone().unwrap());
                controller.write_all(typed_ahead).unwrap();
            });
            let frame = s.wait_for(FRAME_END, 0, Duration::from_secs(10));
            let handed_back = frame.and_then(|at| s.wait_for(LEAVE_ALT_SCREEN, at, Duration::from_secs(10)));
            let question = handed_back.and_then(|at| s.wait_for(b"Trust project", at, Duration::from_secs(10)));
            assert!(question.is_some(), "no first frame, then the question on the shell: {}", s.context());
            s.pump_for(Duration::from_millis(50));
            s
        };
        let trusted = |s: &Session| {
            open_max_core::trust::is_trusted(&s.dir.join("home/.openmax"), &s.dir.join("project")).unwrap()
        };

        let mut s = asked("ask-cancel", b"");
        s.write(b"\x03");
        let status = s.wait_exit(Duration::from_secs(5));
        assert_eq!(status.and_then(|e| e.code()), Some(130), "{status:?}: {}", s.context());
        assert!(!trusted(&s), "{}", s.context());

        let mut s = asked("ask-yes", b"");
        s.write(b"y\r");
        let shown = screen_when(&mut s, 80, 24, Duration::from_secs(10), |shown| {
            shown.contains("stub-model  auto")
        });
        assert!(shown.contains("stub-model  auto"), "{shown}\n{}", s.context());
        assert!(
            shown.contains("project") && shown.contains("READY") && shown.contains("❯"),
            "the screen taken again shows only what the session changed: {shown}\n{}",
            s.context()
        );
        assert!(trusted(&s), "{}", s.context());
        s.write(ANSWER);
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());

        let mut s = asked("ask-typed-ahead", b"y\r");
        let shown = screen_when(&mut s, 80, 24, Duration::from_secs(10), |shown| {
            shown.contains("stub-model  auto")
        });
        assert!(shown.contains("stub-model  auto"), "the typed-ahead y did not answer: {shown}\n{}", s.context());
        assert!(trusted(&s), "{}", s.context());
        s.write(ANSWER);
        s.pump_for(Duration::from_millis(200));
        let shown = screen(&s.out, 80, 24).join("\n");
        assert!(shown.contains("❯ Describe a task"), "the draft is not empty: {shown}");
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
    }

    #[test]
    fn provider_setup_hides_keys_and_returns_input_to_the_same_tui() {
        use std::os::unix::fs::PermissionsExt;
        fn begin(s: &mut Session, command: &str, prompt: &[u8]) -> usize {
            let start = s.out.len();
            s.write(format!("{command}\r").as_bytes());
            let found = if prompt == LEAVE_ALT_SCREEN {
                s.wait_for(prompt, start, Duration::from_secs(5))
            } else {
                s.wait_for_prompt(prompt, start, Duration::from_secs(5))
            };
            found.unwrap_or_else(|| panic!("setup prompt missing for {command}: {}", s.context()))
        }
        // The terminal is asked about the keyboard protocol once a run: setup
        // hands it back with the flags its one answer allows pushed again,
        // then the chat paints.
        fn resume(s: &mut Session, start: usize) {
            let push = s.wait_for(KEYBOARD_PUSH, start, Duration::from_secs(5)).unwrap_or_else(|| panic!("setup did not return to chat: {}", s.context()));
            s.wait_for(FRAME_END, push, Duration::from_secs(5)).unwrap_or_else(|| panic!("chat did not paint after setup: {}", s.context()));
        }
        fn close_picker(s: &mut Session) {
            let start = s.out.len();
            s.write(b"\x1b");
            s.wait_for(FRAME_END, start, Duration::from_secs(5)).expect("model picker did not close");
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let key = format!("$fixture-hidden-key{}", "x".repeat(4096 - "$fixture-hidden-key".len()));
        let expected_key = key.clone();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut socket = loop {
                match listener.accept() {
                    Ok((socket, _)) => break socket,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "setup did not request the model catalog");
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("{error}"),
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut request = Vec::new();
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let mut bytes = [0; 4096];
                let read = socket.read(&mut bytes).unwrap();
                assert!(read > 0);
                request.extend_from_slice(&bytes[..read]);
            }
            let request = String::from_utf8(request).unwrap();
            assert!(request.starts_with("GET /v1/models "));
            assert!(request.to_ascii_lowercase().contains(&format!("authorization: bearer {expected_key}\r\n")));
            let body = r#"{"data":[{"id":"fixture-model","context_window":32768},{"id":"other-model","context_window":65536},{"id":"text-embedding-fixture","context_window":8192}]}"#;
            write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        let mut s = Session::spawn_home("provider-setup", 80, 24, |dir, cmd, terminal, _| {
            std::fs::write(dir.join("settings.json"), r#"{"provider":"fixture","model":"previous-model","context_tokens":16384}"#).unwrap();
            std::fs::write(dir.join("providers.json"), serde_json::json!({"providers":{
                "fixture":{"base_url":base,"models":[{"id":"previous-model","context_tokens":16384}]},
                "chatgpt":{"base_url":open_max_core::chatgpt::API,"wire_api":"responses","chatgpt_account":"work",
                    "models":[{"id":"work-model","context_tokens":16384}]},
                "plan-work":{"base_url":open_max_core::chatgpt::API,"wire_api":"responses","chatgpt_account":"work"}
            }}).to_string()).unwrap();
            let credentials = dir.join("chatgpt.json");
            std::fs::write(&credentials, serde_json::json!({"host":"fixture","accounts":{
                "main":{"client_id":"main-client","subject":"main-subject","email":"","tokens":null},
                "work":{"client_id":"work-client","subject":"work-subject","email":"","tokens":null}
            }}).to_string()).unwrap();
            std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o600)).unwrap();
            cmd.arg("--trust-project").env("OPENAI_API_KEY", "fixture-env-key")
                .env("ANTHROPIC_API_KEY", "fixture-env-key").stderr(terminal.try_clone().unwrap());
        });
        let probe = s.wait_for(KEYBOARD_QUERY, 0, Duration::from_secs(10)).expect("initial keyboard probe");
        s.write(ANSWER);
        s.wait_for(KEYBOARD_PUSH, probe, Duration::from_secs(5)).unwrap_or_else(|| panic!("chat keyboard probe did not finish: {}", s.context()));
        let dir = s.dir.join("home/.openmax");
        let original_settings = std::fs::read(dir.join("settings.json")).unwrap();
        let start = s.out.len();
        s.write(b"/login fixture\r");
        let restored = s.wait_for(LEAVE_ALT_SCREEN, start, Duration::from_secs(5)).unwrap_or_else(|| panic!("setup did not restore terminal: {}", s.context()));
        s.wait_for_prompt(b"API key (hidden)", restored, Duration::from_secs(5)).unwrap_or_else(|| panic!("setup did not receive input: {}", s.context()));
        // Ordinary c/d are text, not cancellation. Other modified or invalid input is inert.
        s.write(b"cd\x7f\x7f\x1b[120;1:3u\x01\x1bqX");
        s.write_bounded(format!("\x1b[200~{key}\x1b[201~\x7f\x1b[200~bad\npaste\x1b[201~\x1b[200~{key}\x1b[201~z\r").as_bytes());
        let model = s.wait_for_prompt(b"Choose a model", restored, Duration::from_secs(5)).unwrap_or_else(|| panic!("catalog was not loaded: {}", s.context()));
        s.write(b"x");
        let filtered = s.wait_for(b"> x\r\n", model, Duration::from_secs(5)).expect("filter text was not shown");
        s.wait_for(b"Enter confirms; Esc cancels", filtered, Duration::from_secs(5)).expect("filter redraw did not finish");
        let filter = s.out.len();
        s.write(b"\x7f");
        let choice = s.wait_for(b"> fixture-model", filter, Duration::from_secs(5)).expect("backspace did not restore choices");
        s.wait_for(b"Enter confirms; Esc cancels", choice, Duration::from_secs(5)).unwrap();
        let shown: Vec<_> = screen(&s.out[model..], 80, 24).into_iter().filter(|row| !row.is_empty()).collect();
        assert_eq!(shown, ["Choose a model (type to filter)", ">", "> fixture-model", "  other-model", "  text-embedding-fixture", "Enter confirms; Esc cancels"]);
        s.write(b"\x1b[B\x1b[A\r");
        let push = s.wait_for(KEYBOARD_PUSH, model, Duration::from_secs(5)).unwrap_or_else(|| panic!("setup did not resume the TUI: {}", s.context()));
        assert!(s.out[..push].windows(b"$fixture-hidden-key".len()).all(|window| window != b"$fixture-hidden-key"), "secret appeared on terminal");
        let shown = screen_when(&mut s, 80, 24, Duration::from_secs(5), |shown| shown.contains("fixture-model"));
        assert!(shown.contains("fixture-model"), "model picker did not refresh: {shown}");
        assert_eq!(std::fs::read(dir.join("settings.json")).unwrap(), original_settings, "login selected a default before the user picked it");
        let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("providers.json")).unwrap()).unwrap();
        assert_eq!(catalog["providers"]["fixture"]["api_key"], format!("${key}"));
        let selected = catalog["providers"]["fixture"]["models"].as_array().unwrap().iter()
            .find(|model| model["id"] == "fixture-model").unwrap();
        assert_eq!(selected["context_tokens"], 32768,
            "the selected model keeps its own context window");
        close_picker(&mut s);
        let start = s.out.len();
        s.write(b"/logout fixture\r");
        let question = s.wait_for_prompt(b"Disconnect fixture?", start, Duration::from_secs(5)).unwrap_or_else(|| panic!("logout did not get keyboard input: {}", s.context()));
        s.write(b"\x1b[B\r");
        s.wait_for(KEYBOARD_PUSH, question, Duration::from_secs(5)).unwrap_or_else(|| panic!("logout did not resume TUI: {}", s.context()));
        let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("providers.json")).unwrap()).unwrap();
        assert!(catalog["providers"].get("fixture").is_none());
        assert_eq!(std::fs::read(dir.join("settings.json")).unwrap(), original_settings);
        for target in ["chatgpt", "plan-work"] {
            let start = s.out.len();
            s.write(format!("/logout {target}\r").as_bytes());
            let push = s.wait_for(KEYBOARD_PUSH, start, Duration::from_secs(5)).unwrap_or_else(|| panic!("account logout did not resume TUI: {}", s.context()));
            let output = String::from_utf8_lossy(&s.out[start..push]);
            assert!(output.contains("account 'work' signed out"), "{output}");
            assert!(!output.contains("account 'main' signed out"), "{output}");
        }
        let credentials: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("chatgpt.json")).unwrap()).unwrap();
        assert_eq!(credentials["accounts"]["main"]["subject"], "main-subject");
        assert_eq!(credentials["accounts"]["work"]["subject"], "work-subject");
        let original_catalog = std::fs::read(dir.join("providers.json")).unwrap();
        let question = begin(&mut s, "/login custom", b"Connection name");
        s.write(b"plan-work\r");
        resume(&mut s, question);
        assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), original_catalog);
        let start = s.out.len();
        s.write(b"/login custom\r");
        let question = s.wait_for_prompt(b"Connection name", start, Duration::from_secs(5)).unwrap_or_else(|| panic!("custom setup did not request a name: {}", s.context()));
        s.write(b"openai\r");
        let push = s.wait_for(KEYBOARD_PUSH, question, Duration::from_secs(5)).unwrap_or_else(|| panic!("reserved name was not rejected: {}", s.context()));
        assert!(!String::from_utf8_lossy(&s.out[question..push]).contains("API key (hidden)"));
        assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), original_catalog);
        for (name, valid) in [(String::new(), false), ("x".repeat(64), true), ("x".repeat(65), false),
            ("x".repeat(66), false), ("dash-name".into(), true), ("under_score".into(), true), ("bad name".into(), false)] {
            let question = begin(&mut s, "/login custom", b"Connection name");
            s.write(format!("\x1b[200~{name}\x1b[201~\r").as_bytes());
            if valid {
                s.wait_for_prompt(b"API base URL", question, Duration::from_secs(5)).unwrap_or_else(|| panic!("valid connection name was refused: {}", s.context()));
                s.write(b"\x1b");
            }
            resume(&mut s, question);
            assert_eq!(std::fs::read(dir.join("providers.json")).unwrap(), original_catalog);
        }
        let question = begin(&mut s, "/login unknown-fixture", LEAVE_ALT_SCREEN);
        resume(&mut s, question);
        let mut entries: serde_json::Value = serde_json::from_slice(&original_catalog).unwrap();
        entries["providers"]["broken"] = serde_json::Value::Null;
        std::fs::write(dir.join("providers.json"), entries.to_string()).unwrap();
        let question = begin(&mut s, "/login broken", LEAVE_ALT_SCREEN);
        resume(&mut s, question);
        std::fs::write(dir.join("providers.json"), &original_catalog).unwrap();
        for (name, env) in [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")] {
            let start = s.out.len();
            let question = begin(&mut s, &format!("/login {name}"), format!("Enter copies {env}").as_bytes());
            let output = String::from_utf8_lossy(&s.out[start..]);
            assert_eq!(output.contains("native Messages API"), name == "anthropic");
            assert!(!output.contains("uses the compatible API"), "native and unrelated connections must not be mislabeled: {output}");
            s.write(b"\x1b");
            resume(&mut s, question);
        }
        let mut legacy: serde_json::Value = serde_json::from_slice(&original_catalog).unwrap();
        legacy["providers"]["anthropic"] = serde_json::json!({"base_url":"https://api.anthropic.com/v1","wire_api":"chat_completions"});
        std::fs::write(dir.join("providers.json"), legacy.to_string()).unwrap();
        let start = s.out.len();
        let question = begin(&mut s, "/login anthropic", b"Enter copies ANTHROPIC_API_KEY");
        let output = String::from_utf8_lossy(&s.out[start..]);
        assert!(output.contains("uses the compatible API") && !output.contains("native Messages API"), "legacy connection protocol was mislabeled: {output}");
        s.write(b"\x1b");
        resume(&mut s, question);
        std::fs::write(dir.join("providers.json"), &original_catalog).unwrap();
        for (url, accepted) in [(open_max_core::chatgpt::API, true), ("https://example.invalid/v1", false)] {
            let mut entries: serde_json::Value = serde_json::from_slice(&original_catalog).unwrap();
            entries["providers"]["openai"] = serde_json::json!({"base_url":url});
            std::fs::write(dir.join("providers.json"), entries.to_string()).unwrap();
            let question = begin(&mut s, "/login openai", LEAVE_ALT_SCREEN);
            if accepted {
                s.wait_for_prompt(b"Enter copies OPENAI_API_KEY", question, Duration::from_secs(5)).expect("matching preset endpoint refused");
                s.write(b"\x1b");
            }
            resume(&mut s, question);
        }
        std::fs::write(dir.join("providers.json"), &original_catalog).unwrap();
        let manual = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        manual.set_nonblocking(true).unwrap();
        let mut entries: serde_json::Value = serde_json::from_slice(&original_catalog).unwrap();
        entries["providers"]["manual"] = serde_json::json!({"base_url":format!("http://{}", manual.local_addr().unwrap())});
        std::fs::write(dir.join("providers.json"), entries.to_string()).unwrap();
        let manual_server = std::thread::spawn(move || {
            for request in 0..6 {
                let deadline = Instant::now() + Duration::from_secs(10);
                let mut socket = loop {
                    match manual.accept() {
                        Ok((socket, _)) => break socket,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "manual model discovery did not run");
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("{error}"),
                    }
                };
                socket.set_nonblocking(false).unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
                let mut header = Vec::new();
                while !header.windows(4).any(|w| w == b"\r\n\r\n") {
                    let mut buffer = [0; 2048];
                    let n = socket.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    header.extend_from_slice(&buffer[..n]);
                }
                let (status, body) = if request < 5 { (404, "") } else {
                    (200, r#"{"data":[{"id":"reported-zero","context_window":0}]}"#)
                };
                write!(socket, "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        let question = begin(&mut s, "/login manual", b"API key (hidden)");
        s.write(b"\r");
        resume(&mut s, question);
        for (model, window, success) in [("m".repeat(257), None, false), ("m".repeat(258), None, false),
            (String::new(), None, false), ("m".repeat(256), Some("1"), true),
            ("zero-window".into(), Some("0"), false), ("reported-zero".into(), Some("1"), true)] {
            let question = begin(&mut s, "/login manual", b"API key (hidden)");
            s.write(b"fixture-manual-key\r");
            if model == "reported-zero" {
                s.wait_for_prompt(b"Choose a model", question, Duration::from_secs(5)).expect("reported model picker missing");
                s.write(b"\r");
            } else {
                s.wait_for_prompt(b"Model ID", question, Duration::from_secs(5)).unwrap_or_else(||
                    panic!("manual model prompt missing (model bytes {}, window {window:?}): {}", model.len(), s.context()));
                s.write(format!("\x1b[200~{model}\x1b[201~\r").as_bytes());
            }
            if let Some(window) = window {
                s.wait_for_prompt(b"Context window in tokens", question, Duration::from_secs(5)).unwrap_or_else(|| panic!("context window prompt missing: {}", s.context()));
                s.write(format!("{window}\r").as_bytes());
            }
            let error_from = s.out.len();
            resume(&mut s, question);
            if window == Some("0") {
                assert!(screen(&s.out[error_from..], 80, 24).join("\n").contains("positive number of tokens"), "context-window refusal was not shown");
            }
            if success {
                close_picker(&mut s);
                let entries: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("providers.json")).unwrap()).unwrap();
                assert!(entries["providers"]["manual"]["models"].as_array().unwrap().iter()
                    .any(|entry| entry["id"] == model && entry["context_tokens"] == 1));
            }
            assert_eq!(std::fs::read(dir.join("settings.json")).unwrap(), original_settings);
        }
        manual_server.join().unwrap();
        let start = s.out.len();
        s.write(b"/login\r");
        let question = s.wait_for_prompt(b"Connect a provider", start, Duration::from_secs(5)).unwrap_or_else(|| panic!("second login did not get input: {}", s.context()));
        s.wait_for(b"manual (configured)", question, Duration::from_secs(5)).expect("configured provider was not appended");
        assert!(String::from_utf8_lossy(&s.out[question..]).contains("OpenAI API"));
        assert!(String::from_utf8_lossy(&s.out[question..]).contains("Anthropic API"));
        s.write(b"manual\r");
        let key_prompt = s.wait_for_prompt(b"API key (hidden)", question, Duration::from_secs(5)).expect("configured provider filter did not select it");
        s.write(b"\x04");
        resume(&mut s, key_prompt);
        let question = begin(&mut s, "/login", b"Connect a provider");
        s.write(b"\x03");
        s.wait_for(KEYBOARD_PUSH, question, Duration::from_secs(5)).unwrap_or_else(|| panic!("cancel did not resume TUI: {}", s.context()));
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(5));
        assert!(status.is_some_and(|status| status.success()), "{status:?}: {}", s.context());
        assert!(!s.out.windows(b"$fixture-hidden-key".len()).any(|window| window == b"$fixture-hidden-key"));
        let asked = s.out.windows(KEYBOARD_QUERY.len()).filter(|window| *window == KEYBOARD_QUERY).count();
        assert_eq!(asked, 1, "the terminal was asked about the keyboard protocol again: {}", s.context());
        for entry in std::fs::read_dir(dir.join("history")).unwrap() {
            assert!(!std::fs::read_to_string(entry.unwrap().path()).unwrap().contains("$fixture-hidden-key"));
        }
        server.join().unwrap();
    }

    /// What a session that painted first has to say before it runs lands on
    /// the shell, after the terminal is handed back (the alternate screen
    /// left), never on the screen the session leaves behind. A
    /// `--trust-project` grant reports once the session ends, and ahead of
    /// the refusal when settings that do not parse then refuse the session
    /// (exit 2). A trust file that does not parse and a grant asked for from
    /// an agent session refuse it too (exit 3). On a first run, the line that
    /// says which keys the settings flags wrote prints after the grant's, and
    /// a flag refused against settings.json refuses the session before the
    /// grant (exit 2).
    #[test]
    fn a_grant_and_a_refusal_behind_the_first_frame_print_on_the_shell() {
        // The run's exit code and what it printed once it left the
        // alternate screen; `go_on` answers a session that starts.
        let on_the_shell = |tag: &str, setup: &dyn Fn(&Path, &mut Command), go_on: bool| {
            let mut s = Session::spawn_home(tag, 80, 24, |dir, cmd, terminal, _| {
                setup(dir, cmd);
                cmd.stderr(terminal.try_clone().unwrap());
            });
            if go_on {
                // The query follows the session's own first frame, once its
                // loop reads keys, so Ctrl+C is a key there.
                let query = s.wait_for(KEYBOARD_QUERY, 0, Duration::from_secs(10));
                assert!(query.is_some(), "no keyboard query: {}", s.context());
                s.write(ANSWER);
                s.write(CTRL_C_TWICE);
            }
            let status = s.wait_exit(Duration::from_secs(10));
            let left = find(&s.out, FRAME_END, 0).and_then(|at| find(&s.out, LEAVE_ALT_SCREEN, at));
            let left = left.unwrap_or_else(|| panic!("no first frame, then the shell: {}", s.context()));
            let shell = String::from_utf8_lossy(&s.out[left..]).into_owned();
            (status.and_then(|e| e.code()), shell)
        };
        let settings = |dir: &Path| {
            std::fs::write(
                dir.join("settings.json"),
                r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#,
            )
            .unwrap()
        };

        let (code, shell) = on_the_shell(
            "grant-reported",
            &|dir, cmd| {
                settings(dir);
                cmd.arg("--trust-project");
            },
            true,
        );
        assert_eq!(code, Some(0), "{shell:?}");
        assert!(shell.contains("openmax: trusted project"), "{shell:?}");

        let (code, shell) = on_the_shell(
            "grant-then-bad-settings",
            &|dir, cmd| {
                std::fs::write(dir.join("settings.json"), "{").unwrap();
                cmd.arg("--trust-project");
            },
            false,
        );
        assert_eq!(code, Some(2), "{shell:?}");
        let grant = shell.find("openmax: trusted project");
        let refusal = shell.find("openmax: invalid settings file");
        assert!(grant.is_some() && grant < refusal, "the grant's line, then the refusal: {shell:?}");

        let (code, shell) = on_the_shell(
            "bad-trust",
            &|dir, _| {
                settings(dir);
                std::fs::write(dir.join("trust.json"), "{").unwrap();
            },
            false,
        );
        assert_eq!(code, Some(3), "{shell:?}");
        assert!(shell.contains("openmax: invalid ") && shell.contains("trust.json"), "{shell:?}");

        let (code, shell) = on_the_shell(
            "agent-grant",
            &|dir, cmd| {
                settings(dir);
                cmd.arg("--trust-project").env("OPENMAX_SESSION", "1");
            },
            false,
        );
        assert_eq!(code, Some(3), "{shell:?}");
        assert!(shell.contains("openmax: trust grants are human actions"), "{shell:?}");

        let first_run = ["--trust-project", "--base-url", "http://127.0.0.1:9/v1", "-m", "stub-model", "--context-tokens", "16384"];
        let (code, shell) = on_the_shell("first-run", &|_, cmd| _ = cmd.args(first_run), true);
        assert_eq!(code, Some(0), "{shell:?}");
        let grant = shell.find("openmax: trusted project");
        let wrote = shell.find("openmax: wrote base_url, model, context_tokens to ");
        assert!(grant.is_some() && grant < wrote, "the grant's line, then the keys written: {shell:?}");

        let (code, shell) = on_the_shell(
            "key-beside-provider",
            &|dir, cmd| {
                std::fs::write(dir.join("settings.json"), r#"{"provider":"p","model":"m"}"#).unwrap();
                cmd.args(["--trust-project", "--api-key", "sk-k"]);
            },
            false,
        );
        assert_eq!(code, Some(2), "{shell:?}");
        assert!(shell.contains("openmax: --api-key is the key for settings.json's base_url"), "{shell:?}");
        assert!(!shell.contains("openmax: trusted project"), "refused before the grant: {shell:?}");
    }

    /// Hold the pipe at `path` open to write once its reader has it open, so
    /// that reader waits in its read: a write end opened without blocking is
    /// refused until a reader is there.
    fn reader_waits(path: &Path, within: Duration) -> File {
        use std::os::unix::fs::OpenOptionsExt;
        let deadline = Instant::now() + within;
        loop {
            match std::fs::OpenOptions::new().write(true).custom_flags(libc::O_NONBLOCK).open(path) {
                Ok(pipe) => return pipe,
                Err(e) if e.raw_os_error() == Some(libc::ENXIO) && Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(e) => panic!("nothing reads {}: {e}", path.display()),
            }
        }
    }

    /// Start a `--trust-project` session whose read of the file `stalled`
    /// names in `~/.openmax` waits on a pipe nothing writes, as on storage
    /// that stalls; once the first frame is up and the read waits, stop the
    /// run with `stop`. It must end at once with `code`, the terminal handed
    /// back.
    fn stopped_while_reading(tag: &str, stalled: impl Fn(&Path) -> PathBuf, stop: impl FnOnce(&mut Session), code: i32) {
        let mut pipe = PathBuf::new();
        let mut s = Session::spawn_home(tag, 80, 24, |dir, cmd, _, _| {
            std::fs::write(
                dir.join("settings.json"),
                r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#,
            )
            .unwrap();
            pipe = stalled(dir);
            std::fs::create_dir_all(pipe.parent().unwrap()).unwrap();
            unwritten(&pipe);
            cmd.arg("--trust-project");
        });
        let frame = s.wait_for(FRAME_END, 0, Duration::from_secs(10));
        let frame = frame.unwrap_or_else(|| panic!("{tag}: no first frame: {}", s.context()));
        let _held = reader_waits(&pipe, Duration::from_secs(10));
        stop(&mut s);
        let status = s.wait_exit(Duration::from_secs(5));
        assert_eq!(status.and_then(|e| e.code()), Some(code), "{tag}: {status:?}: {}", s.context());
        assert!(find(&s.out, LEAVE_ALT_SCREEN, frame).is_some(), "{tag}: {}", s.context());
    }

    /// Until the session's loop reads the quit, the first signal ends the run
    /// at once, with its status and the terminal handed back, as it did when
    /// trust and settings were read before the session took the terminal,
    /// and Ctrl+C at the terminal raises SIGINT, as it did before raw mode.
    /// Both hold while trust.json is read, in front of the session, and while
    /// the prompt history is read, as the session starts behind the first
    /// frame.
    #[test]
    fn a_signal_or_ctrl_c_before_the_loop_runs_ends_the_run_at_once() {
        let trust = |dir: &Path| dir.join("trust.json");
        let history = |dir: &Path| {
            let project = dir.parent().unwrap().parent().unwrap().join("project");
            dir.join("history").join(format!("{}.json", open_max_core::ledger::project_key(&project)))
        };
        let term = |s: &mut Session| {
            // SAFETY: kill sends one signal to one pid.
            assert_eq!(unsafe { libc::kill(s.child.id() as libc::pid_t, libc::SIGTERM) }, 0);
        };
        let ctrl_c = |s: &mut Session| {
            s.write(b"\x03");
        };
        stopped_while_reading("stalled-trust-term", trust, term, 128 + libc::SIGTERM);
        stopped_while_reading("stalled-trust-ctrl-c", trust, ctrl_c, 128 + libc::SIGINT);
        stopped_while_reading("stalled-history-term", history, term, 128 + libc::SIGTERM);
        stopped_while_reading("stalled-history-ctrl-c", history, ctrl_c, 128 + libc::SIGINT);
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

        // The session ends as a quit would, then reports the signal.
        let status = s.wait_exit(Duration::from_secs(10));
        assert_eq!(status.and_then(|e| e.code()), Some(143), "{status:?}: {}", s.context());
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

    /// An absolute palette slot is the background of some terminal theme:
    /// slot 15 text vanishes on a light terminal and a slot 0 fill is a box
    /// there. The first frames as the terminal receives them (the shell's
    /// header and focused composer with its border, then the session's
    /// status line) select only the terminal's own colors and the palette's
    /// gray, whatever the tokens map to. The keyboard query follows the
    /// session's first frame.
    #[test]
    fn the_first_frame_uses_only_the_terminals_own_colors_and_its_gray() {
        let mut s = Session::spawn("palette", 80, 24);
        let query = s.wait_for(KEYBOARD_QUERY, 0, Duration::from_secs(10)).unwrap_or_else(|| {
            panic!("no keyboard query: {}", s.context())
        });
        let foreign = foreign_colors(&s.out[..query]);
        assert!(foreign.is_empty(), "{foreign:?}: {}", s.context());
        s.write(ANSWER);
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
    }

    /// The screen a terminal shows after `out`, row by row: cursor moves,
    /// erases and printed characters applied to a `cols` by `rows` grid, and
    /// every other sequence skipped. A frame is a diff against the one before
    /// it, so the bytes of one line need not arrive together; a person reads
    /// whole frames, so `out` ends where a frame does.
    fn screen(out: &[u8], cols: usize, rows: usize) -> Vec<String> {
        let blank = || vec![vec![' '; cols]; rows];
        let mut grid = blank();
        let (mut row, mut col) = (0usize, 0usize);
        let text = String::from_utf8_lossy(out);
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\x1b' => match chars.next() {
                    Some('[') => {
                        let mut params = String::new();
                        let mut last = None;
                        for c in chars.by_ref() {
                            if ('\x40'..='\x7e').contains(&c) {
                                last = Some(c);
                                break;
                            }
                            params.push(c);
                        }
                        let n: Vec<usize> = params.split(';').map(|p| p.parse().unwrap_or(1)).collect();
                        match last {
                            Some('H') => {
                                row = n.first().copied().unwrap_or(1).max(1) - 1;
                                col = n.get(1).copied().unwrap_or(1).max(1) - 1;
                            }
                            Some('A') => row = row.saturating_sub(n.first().copied().unwrap_or(1).max(1)),
                            Some('G') => col = n.first().copied().unwrap_or(1).max(1) - 1,
                            Some('J') if params.is_empty() || params == "0" => {
                                if let Some(line) = grid.get_mut(row) { line[col.min(cols)..].fill(' '); }
                                for line in grid.iter_mut().skip(row + 1) { line.fill(' '); }
                            }
                            Some('J') if params == "2" || params == "3" => grid = blank(),
                            // Entering the alternate screen blanks it.
                            Some('h') if params == "?1049" => grid = blank(),
                            Some('K') => {
                                if let Some(line) = grid.get_mut(row) {
                                    line[col.min(cols)..].fill(' ');
                                }
                            }
                            _ => {}
                        }
                    }
                    // An OSC runs to BEL or to ESC \.
                    Some(']') => {
                        while let Some(c) = chars.next() {
                            if c == '\x07' || (c == '\x1b' && chars.next_if_eq(&'\\').is_some()) {
                                break;
                            }
                        }
                    }
                    _ => {}
                },
                '\r' => col = 0,
                '\n' => row += 1,
                c if c.is_control() => {}
                c => {
                    if let Some(cell) = grid.get_mut(row).and_then(|line| line.get_mut(col)) {
                        *cell = c;
                    }
                    col += 1;
                }
            }
        }
        grid.into_iter().map(|line| line.into_iter().collect::<String>().trim_end().to_string()).collect()
    }

    /// A completions endpoint on its own thread: each request, read whole,
    /// gets the next reply as one JSON body. Returns the base URL.
    fn endpoint(replies: Vec<String>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for reply in replies {
                let Ok((mut stream, _)) = listener.accept() else { return };
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && stream.read(&mut byte).is_ok_and(|n| n == 1) {
                    head.push(byte[0]);
                }
                let length = String::from_utf8_lossy(&head)
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                let mut body = vec![0u8; length];
                let _ = stream.read_exact(&mut body);
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        base_url
    }

    /// The screen as of the last whole frame, once `done` holds for it or
    /// when `within` passes. The TUI sends a frame in one write, but the
    /// pseudo-terminal can hand it to the reader in pieces, and a note in the
    /// bottom-anchored transcript rewrites every row above the composer, top
    /// to bottom: part of a frame can show a new row above an old one, which a
    /// terminal that honours synchronized updates never does.
    fn screen_when(
        s: &mut Session,
        cols: usize,
        rows: usize,
        within: Duration,
        done: impl Fn(&str) -> bool,
    ) -> String {
        let deadline = Instant::now() + within;
        loop {
            let end =
                s.out.windows(FRAME_END.len()).rposition(|w| w == FRAME_END).map_or(0, |at| at + FRAME_END.len());
            let shown = screen(&s.out[..end], cols, rows).join("\n");
            if done(&shown) || Instant::now() >= deadline {
                return shown;
            }
            s.pump(Duration::from_millis(50));
        }
    }

    #[test]
    fn permissions_picker_owns_explicit_project_changes_while_shift_tab_cycles_effort() {
        let (cols, rows) = (160, 40);
        let mut s = Session::spawn("permissions-picker", cols as u16, rows as u16);
        s.wait_for(FRAME_END, 0, Duration::from_secs(10)).expect("first frame");
        s.write(ANSWER);
        let data = s.dir.join("home/.openmax");
        let project = s.dir.join("project");
        let mode = || open_max_core::state::Core::new(data.clone()).unwrap().0.approval_mode(&project);
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("stub-model"));
        assert!(shown.contains("stub-model"), "{shown}");
        assert_eq!(mode(), open_max_core::config::ApprovalMode::Auto, "new trust grants already default to Auto");
        s.write(b"\x1b[Z");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested none"));
        assert!(shown.contains("requested none"), "{shown}");
        assert_eq!(mode(), open_max_core::config::ApprovalMode::Auto);
        s.write(b"/permissions\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("new project default"));
        assert!(shown.contains("new project default"), "{shown}");
        s.write(b"\x1b[Z\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("permissions: readonly"));
        assert!(shown.contains("permissions: readonly"), "{shown}");
        assert_eq!(mode(), open_max_core::config::ApprovalMode::Readonly);
        assert_eq!(open_max_core::config::load(&data).unwrap().reasoning_effort, Some(open_max_core::config::ReasoningEffort::None));
        s.write(b"\x1b[Z");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested minimal"));
        assert!(shown.contains("requested minimal"), "{shown}");
        assert_eq!(mode(), open_max_core::config::ApprovalMode::Readonly);
        s.write(b"/permissions ask\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("permissions: ask"));
        assert!(shown.contains("permissions: ask"), "{shown}");
        assert_eq!(mode(), open_max_core::config::ApprovalMode::Ask);
        s.write(b"/approvals auto\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("permissions: auto"));
        assert!(shown.contains("permissions: auto"), "{shown}");
        assert_eq!(mode(), open_max_core::config::ApprovalMode::Auto);
        s.write(b"/permissions invalid\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("usage: /permissions"));
        assert!(shown.contains("usage: /permissions"), "{shown}");
        assert_eq!(mode(), open_max_core::config::ApprovalMode::Auto);
        assert_eq!(open_max_core::config::load(&data).unwrap().approval_mode, open_max_core::config::ApprovalMode::Ask,
            "project selections never change the legacy global fallback");
        s.write(CTRL_C_TWICE);
        assert!(s.wait_exit(Duration::from_secs(5)).is_some_and(|status| status.success()), "{}", s.context());
    }

    #[test]
    fn ctrl_c_discards_the_prompt_then_quits_on_a_second_press() {
        let (cols, rows) = (100, 30);
        let mut s = Session::spawn_with("clear-prompt", cols as u16, rows as u16,
            r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#);
        let probe = s.wait_for(KEYBOARD_QUERY, 0, Duration::from_secs(10)).expect("initial keyboard probe");
        s.write(ANSWER);
        s.wait_for(KEYBOARD_PUSH, probe, Duration::from_secs(5)).expect("keyboard negotiation");
        // A multiline paste must be discarded without becoming submitted history.
        s.write("\x1b[200~discard this draft\nUnicode λ\x1b[201~".as_bytes());
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("discard this draft"));
        assert!(shown.contains("discard this draft"), "{shown}");
        s.write(b"\x03");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("ctrl+c again to quit"));
        assert!(shown.contains("ctrl+c again to quit") && !shown.contains("discard this draft") && !shown.contains("Unicode"), "{shown}");
        assert!(s.child.try_wait().unwrap().is_none(), "first press quit the app");
        // Editing disarms quit; Up must not resurrect the discarded draft.
        s.write(b"\x1b[Anew draft");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("new draft"));
        assert!(shown.contains("new draft") && !shown.contains("discard this draft") && !shown.contains("ctrl+c again to quit"), "{shown}");
        s.write(b"\x03");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("ctrl+c again to quit"));
        assert!(shown.contains("ctrl+c again to quit") && !shown.contains("new draft"), "{shown}");
        // Clearing also releases the completion popup that owned the keys.
        s.write(b"/sta");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("/status"));
        assert!(shown.contains("/status"), "completion did not open: {shown}");
        s.write(b"\x03");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("ctrl+c again to quit"));
        assert!(shown.contains("ctrl+c again to quit") && !shown.contains("/status") && !shown.contains("/sta"), "{shown}");
        // A collapsed paste has backing content beyond its visible marker.
        let paste = "collapsed Unicode λ draft\n".repeat(12);
        s.write(format!("\x1b[200~{paste}\x1b[201~").as_bytes());
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("[pasted #"));
        assert!(shown.contains("[pasted #") && !shown.contains("ctrl+c again to quit"), "{shown}");
        s.write(b"\x03");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("ctrl+c again to quit"));
        assert!(shown.contains("ctrl+c again to quit") && !shown.contains("[pasted #"), "{shown}");
        s.write(b"\x1b[Ausable after paste");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("usable after paste"));
        assert!(shown.contains("usable after paste") && !shown.contains("[pasted #") && !shown.contains("collapsed Unicode"), "{shown}");
        s.write(CTRL_C_TWICE);
        assert!(s.wait_exit(Duration::from_secs(5)).is_some_and(|status| status.success()), "{}", s.context());
    }

    #[test]
    fn effort_picker_selects_cancels_clears_and_refuses_ignored_settings() {
        let (cols, rows) = (160, 40);
        let mut s = Session::spawn_home("effort-picker", cols as u16, rows as u16, |dir, cmd, _, _| {
            std::fs::write(dir.join("settings.json"),
                r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384,"reasoning_effort":"high"}"#).unwrap();
            cmd.args(["--trust-project", "--effort", "default", "--model", "transient-model", "--context-tokens", "8192"]);
        });
        s.wait_for(FRAME_END, 0, Duration::from_secs(10)).expect("first frame");
        s.write(ANSWER);
        s.write(b"/effort default\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested: default"));
        assert!(shown.contains("requested: default"), "{shown}");
        let initial: serde_json::Value = serde_json::from_slice(&std::fs::read(s.dir.join("home/.openmax/settings.json")).unwrap()).unwrap();
        assert!(initial.get("reasoning_effort").is_none(), "explicit same-value Default clears the file");
        assert_eq!(initial["model"], "stub-model");
        assert_eq!(initial["context_tokens"], 16384, "unrelated overrides remain run-only");
        s.write(b"/effort\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("effort default"));
        assert!(shown.lines().any(|line| line.contains("effort default") && line.contains("omit the request setting")), "{shown}");
        s.write(b"\x1b[B\x1b[B\x1b[B\x1b[B\x1b[B\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested: high"));
        assert!(shown.contains("requested: high") && shown.contains("requested high"), "{shown}");
        let settings_path = s.dir.join("home/.openmax/settings.json");
        let saved = || serde_json::from_slice::<serde_json::Value>(&std::fs::read(&settings_path).unwrap()).unwrap();
        assert_eq!(saved()["reasoning_effort"], "high");
        s.write(b"/effort\r");
        screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("effort high"));
        s.write(b"\x1b");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| !shown.contains("model validates support"));
        assert!(!shown.contains("model validates support"), "{shown}");
        assert_eq!(saved()["reasoning_effort"], "high");
        s.write(b"/effort\r");
        screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("model validates support"));
        s.write(b"\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested: default") && saved().get("reasoning_effort").is_none());
        assert!(shown.contains("requested: default"), "{shown}");
        assert!(saved().get("reasoning_effort").is_none());
        s.write(b"/effort none\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested: none"));
        assert!(shown.contains("requested: none"), "{shown}");
        assert_eq!(saved()["reasoning_effort"], "none");
        s.write(b"/effort turbo\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("invalid reasoning effort"));
        assert!(shown.contains("invalid reasoning effort"), "{shown}");
        assert_eq!(saved()["reasoning_effort"], "none");
        s.write(CTRL_C_TWICE);
        assert!(s.wait_exit(Duration::from_secs(5)).is_some_and(|status| status.success()), "{}", s.context());

        let mut s = Session::spawn_home("same-effort", cols as u16, rows as u16, |dir, cmd, _, _| {
            std::fs::write(dir.join("settings.json"),
                r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384,"reasoning_effort":"high"}"#).unwrap();
            cmd.args(["--trust-project", "--effort", "low"]);
        });
        s.wait_for(FRAME_END, 0, Duration::from_secs(10)).expect("first frame");
        s.write(ANSWER);
        s.write(b"/effort low\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested: low"));
        assert!(shown.contains("requested: low"), "{shown}");
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(s.dir.join("home/.openmax/settings.json")).unwrap()).unwrap();
        assert_eq!(saved["reasoning_effort"], "low", "same-value selection becomes persistent");
        s.write(CTRL_C_TWICE);
        assert!(s.wait_exit(Duration::from_secs(5)).is_some_and(|status| status.success()), "{}", s.context());

        let mut s = Session::spawn_home("scoped-effort", cols as u16, rows as u16, |dir, cmd, _, _| {
            std::fs::write(dir.join("settings.json"),
                r#"{"provider":"saved","model":"stub-model","context_tokens":16384}"#).unwrap();
            std::fs::write(dir.join("providers.json"),
                r#"{"providers":{"saved":{"base_url":"https://api.anthropic.com/v1"},"temporary":{"base_url":"http://127.0.0.1:9/v1"}}}"#).unwrap();
            cmd.args(["--trust-project", "--provider", "temporary"]);
        });
        s.wait_for(FRAME_END, 0, Duration::from_secs(10)).expect("first frame");
        s.write(ANSWER);
        s.write(b"/effort high\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested: high"));
        assert!(shown.contains("requested: high") && shown.contains("for this run only"), "{shown}");
        let data = s.dir.join("home/.openmax");
        let saved = open_max_core::config::load(&data).unwrap();
        assert_eq!(saved.provider.as_deref(), Some("saved"));
        assert!(saved.reasoning_effort.is_none(), "temporary endpoint effort must not poison the saved endpoint");
        assert!(open_max_core::providers::resolve(&saved, &data).is_ok());
        s.write(b"/provider saved\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("ignores reasoning effort"));
        assert!(shown.contains("ignores reasoning effort"), "{shown}");
        assert!(open_max_core::config::load(&data).unwrap().reasoning_effort.is_none());
        s.write(b"/model stub-model\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("model set to stub-model"));
        assert!(shown.contains("model set to stub-model"), "{shown}");
        assert!(open_max_core::config::load(&data).unwrap().reasoning_effort.is_none(), "later saves preserve run-only effort");
        s.write(b"/effort default\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("requested: default"));
        assert!(shown.contains("requested: default"), "{shown}");
        assert!(open_max_core::providers::resolve(&open_max_core::config::load(&data).unwrap(), &data).is_ok());
        s.write(CTRL_C_TWICE);
        assert!(s.wait_exit(Duration::from_secs(5)).is_some_and(|status| status.success()), "{}", s.context());

        let mut s = Session::spawn_with("ignored-effort", cols as u16, rows as u16,
            r#"{"base_url":"https://api.anthropic.com/v1","model":"fixture","context_tokens":16384}"#);
        s.wait_for(FRAME_END, 0, Duration::from_secs(10)).expect("first frame");
        s.write(ANSWER);
        s.write(b"/effort high\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(5), |shown| shown.contains("ignores reasoning effort"));
        assert!(shown.contains("ignores reasoning effort"), "{shown}");
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(s.dir.join("home/.openmax/settings.json")).unwrap()).unwrap();
        assert!(saved.get("reasoning_effort").is_none());
        s.write(CTRL_C_TWICE);
        assert!(s.wait_exit(Duration::from_secs(5)).is_some_and(|status| status.success()), "{}", s.context());
    }

    /// In auto a tool the agent writes mid-turn is callable on its next step
    /// with no card, so the screen is the one place a person can see what it
    /// runs before it does: when the tool lands, the transcript names it and
    /// the exact command line its manifest declares.
    #[test]
    fn a_tool_written_mid_turn_shows_the_command_it_runs() {
        let manifest = "name = \"wordfreq\"\ndescription = \"count words\"\ncommand = \"/bin/echo\"\nargs = [\"release-7f3a\"]\n";
        let write = serde_json::json!({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null, "tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "write_file",
                "arguments": serde_json::json!({"path": ".openmax/tools/wordfreq.toml", "content": manifest}).to_string(),
            }}],
        }}]});
        let done = serde_json::json!({"choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "tool ready",
        }}]});
        let base_url = endpoint(vec![write.to_string(), done.to_string()]);
        let settings =
            serde_json::json!({"base_url": base_url, "model": "stub-model", "context_tokens": 16384}).to_string();
        let (cols, rows) = (160, 40);
        // --trust-project records auto for the project, so nothing stops for a card.
        let mut s = Session::spawn_with("refreeze", cols as u16, rows as u16, &settings);
        s.wait_for(FRAME_END, 0, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no first frame: {}", s.context()));
        s.write(ANSWER);
        s.write(b"add the tool\r");
        // Wait for the turn to settle, not only for its last message: the
        // reply is drawn before the turn ends, and Ctrl+C on a running turn
        // cancels instead of arming the quit. The hint that Esc cancels is
        // gone once the turn has ended.
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(30), |shown| {
            shown.contains("tool ready") && !shown.contains("esc cancel")
        });
        assert!(shown.contains("tool ready") && !shown.contains("esc cancel"), "the turn did not settle:\n{shown}");
        assert!(
            shown.contains("tool wordfreq runs: '/bin/echo' 'release-7f3a'"),
            "the new tool's command line is not on screen:\n{shown}"
        );
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
    }

    /// A tool that lands between turns (through a pull, or written by a turn
    /// that was cancelled before its refreeze) becomes callable, with no card
    /// in auto, at the next turn start, or on `/reload` if that runs first.
    /// The reload's note names the tool and the exact command line its
    /// manifest declares.
    #[test]
    fn a_tool_picked_up_by_reload_shows_the_command_it_runs() {
        let reply = serde_json::json!({"choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "hello back",
        }}]});
        let base_url = endpoint(vec![reply.to_string()]);
        let settings =
            serde_json::json!({"base_url": base_url, "model": "stub-model", "context_tokens": 16384}).to_string();
        let (cols, rows) = (160, 40);
        let mut s = Session::spawn_with("reload", cols as u16, rows as u16, &settings);
        s.wait_for(FRAME_END, 0, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no first frame: {}", s.context()));
        s.write(ANSWER);
        // A first turn creates the session that /reload re-freezes; the hint
        // that Esc cancels is gone once the turn has settled.
        s.write(b"hello\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(30), |shown| {
            shown.contains("hello back") && !shown.contains("esc cancel")
        });
        assert!(shown.contains("hello back") && !shown.contains("esc cancel"), "the turn did not settle:\n{shown}");
        let tools = s.dir.join("project/.openmax/tools");
        std::fs::create_dir_all(&tools).unwrap();
        std::fs::write(
            tools.join("stamp.toml"),
            "name = \"stamp\"\ndescription = \"stamp a build\"\ncommand = \"/bin/echo\"\nargs = [\"build-91c2\"]\n",
        )
        .unwrap();
        s.write(b"/reload\r");
        let shown = screen_when(&mut s, cols, rows, Duration::from_secs(30), |shown| shown.contains("re-frozen"));
        assert!(shown.contains("re-frozen"), "the reload did not finish:\n{shown}");
        assert!(
            shown.contains("tool stamp runs: '/bin/echo' 'build-91c2'"),
            "the reloaded tool's command line is not on screen:\n{shown}"
        );
        s.write(CTRL_C_TWICE);
        let status = s.wait_exit(Duration::from_secs(10));
        assert!(status.is_some_and(|e| e.success()), "{status:?}: {}", s.context());
    }

    /// The session at a terminal starts behind its first frame, apart from
    /// every other run, and records its tool groups as they do: a session
    /// killed outright, its watcher first, leaves its tool's tree running and
    /// recorded, and the next session at a terminal stops it once its first
    /// frame is up and removes the record, before it checks its settings
    /// flags against settings.json, asks for trust or writes settings: here a
    /// session whose flag settings.json refuses (exit 2) still stops it. The
    /// tree ignores SIGTERM, so only the SIGKILL after the grace stops it.
    /// The watcher is the process whose command line names the record
    /// (`openmax --lifeline <record> <start>`).
    #[test]
    fn a_session_at_a_terminal_stops_the_tools_a_killed_session_left_running() {
        let script = "trap '' TERM; echo $$ >> ../pids; sleep 6841 & echo $! >> ../pids; exec sleep 7841";
        let call = serde_json::json!({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": null, "tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "bash",
                "arguments": serde_json::json!({"command": script}).to_string(),
            }}],
        }}]});
        let base_url = endpoint(vec![call.to_string()]);
        let settings =
            serde_json::json!({"base_url": base_url, "model": "stub-model", "context_tokens": 16384}).to_string();
        let mut killed = Session::spawn_with("reap-killed", 120, 30, &settings);
        killed
            .wait_for(FRAME_END, 0, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no first frame: {}", killed.context()));
        killed.write(ANSWER);
        killed.write(b"start the tree\r");
        let home = killed.dir.join("home");
        let record = home.join(".openmax").join("run").join(format!("{}.groups", killed.child.id()));
        let deadline = Instant::now() + Duration::from_secs(10);
        let pids = loop {
            let listed = std::fs::read_to_string(killed.dir.join("pids")).unwrap_or_default();
            let pids: Vec<i32> = listed.lines().filter_map(|l| l.trim().parse().ok()).collect();
            let lines = std::fs::read_to_string(&record).unwrap_or_default().lines().count();
            if (pids.len() == 2 && lines >= 2) || Instant::now() >= deadline {
                break pids;
            }
            killed.pump(Duration::from_millis(20));
        };
        // Started by the record's first write, so waited for.
        let deadline = Instant::now() + Duration::from_secs(10);
        let watcher = loop {
            let out = Command::new("ps").args(["-A", "-ww", "-o", "pid=", "-o", "args="]).output().expect("ps runs");
            let (listing, needle) = (String::from_utf8_lossy(&out.stdout), record.to_string_lossy());
            let watcher: Option<i32> =
                listing.lines().filter(|line| line.contains(&*needle)).find_map(|line| line.split_whitespace().next()?.parse().ok());
            if watcher.is_some() || Instant::now() >= deadline {
                break watcher;
            }
            killed.pump(Duration::from_millis(20));
        };
        if let Some(watcher) = watcher {
            unsafe { libc::kill(watcher, libc::SIGKILL) };
        }
        let _ = killed.child.kill();
        let deadline = Instant::now() + Duration::from_secs(10);
        while killed.child.try_wait().unwrap().is_none() && Instant::now() < deadline {
            killed.pump(Duration::from_millis(20));
        }
        let alive = |pid: &i32| unsafe { libc::kill(*pid, 0) == 0 };
        let survived_the_kill = pids.len() == 2 && pids.iter().all(alive);
        let recorded = record.exists();
        // An --api-key beside the provider settings.json selects is refused
        // before the trust grant.
        std::fs::write(home.join(".openmax").join("settings.json"), r#"{"provider":"p","model":"m"}"#).unwrap();
        let mut next = Session::spawn_home("reap-next", 120, 30, |_, cmd, _, _| {
            cmd.env("HOME", &home).args(["--trust-project", "--api-key", "sk-k"]);
        });
        next.wait_for(FRAME_END, 0, Duration::from_secs(10))
            .unwrap_or_else(|| panic!("no first frame: {}", next.context()));
        let status = next.wait_exit(Duration::from_secs(10));
        let deadline = Instant::now() + Duration::from_secs(5);
        while pids.iter().any(alive) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let survivors: Vec<i32> = pids.iter().copied().filter(alive).collect();
        for pid in &survivors {
            unsafe { libc::kill(*pid, libc::SIGKILL) };
        }
        let record_kept = record.exists();
        assert!(watcher.is_some(), "the session's first tool started no watcher: {}", killed.context());
        assert!(survived_the_kill, "the tool tree did not outlive SIGKILL, so this proves nothing: {pids:?}");
        assert!(recorded, "the killed session left no record of its tool groups at {}", record.display());
        assert_eq!(status.and_then(|e| e.code()), Some(2), "the flag was not refused: {}", next.context());
        assert!(survivors.is_empty(), "the next session left the tool tree running: {survivors:?}");
        assert!(!record_kept, "the record outlived the session that stopped its groups");
    }
}
