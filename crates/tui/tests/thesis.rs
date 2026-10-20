//! The thesis targets, one test each, named after the target it holds.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The lines of one job in ci.yml, from its key up to the next key at the
/// jobs' indentation or less.
fn ci_job(name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(repo().join(".github/workflows/ci.yml")).unwrap();
    let key = format!("  {name}:");
    let mut lines = text.lines().skip_while(|line| line.trim_end() != key);
    assert!(lines.next().is_some(), "ci.yml has no `{name}` job");
    lines
        .take_while(|line| {
            let body = line.trim_start();
            body.is_empty() || body.starts_with('#') || line.len() - body.len() > 2
        })
        .map(str::to_string)
        .collect()
}

/// The whole suite runs in 5 seconds after compilation. No test inside the
/// suite can time the suite, so CI does: its test job runs
/// scripts/check-suite-time.sh on the suite it just built, on every runner of
/// its matrix, with that runner's multiplier, which the script explains, in a
/// step with a timeout. This test holds that wiring, and holds the gate to
/// its job by driving it with a stand-in `cargo` that lists stand-in test
/// binaries: it runs them at once (four of them each wait for the other
/// three to start), each the way `cargo test` would (from its package
/// directory, with Ctrl+C not ignored), passes a suite inside 5 seconds
/// times the multiplier, and fails one over it or with a binary that fails
/// again when run alone, naming the binary and showing its failures, and
/// passes one whose binary failed only beside the others. Under a perl
/// without a module it needs, it stops before cargo and names the module.
#[test]
fn whole_suite_runs_in_five_seconds() {
    let job = ci_job("test");
    let step = |run: &str| {
        job.iter()
            .position(|line| line.trim() == format!("run: {run}"))
            .unwrap_or_else(|| panic!("the test job does not run `{run}`"))
    };
    let gate_run = "scripts/check-suite-time.sh ${{ matrix.suite_time_multiplier }}";
    let (tests, gate) = (step("cargo test --workspace --locked"), step(gate_run));
    assert!(tests < gate, "the suite time gate runs before the suite is built");
    // The gate's own step, from its first key, whichever that is, to the
    // next step: a condition there skips the gate on some runner, and so
    // passes the pull request that slowed the suite on it.
    let first = job[..gate].iter().rposition(|line| line.trim().starts_with("- ")).unwrap();
    let end = job[gate..].iter().position(|line| line.trim().starts_with("- ")).map_or(job.len(), |n| gate + n);
    let keys: Vec<&str> = job[first..end].iter().map(|line| line.trim().trim_start_matches("- ")).collect();
    let soft: Vec<&&str> =
        keys.iter().filter(|key| ["if:", "continue-on-error:"].iter().any(|soft| key.starts_with(soft))).collect();
    assert!(soft.is_empty(), "the suite time gate can be skipped or ignored: {soft:?}");
    assert!(
        keys.iter().any(|key| key.starts_with("timeout-minutes:")),
        "a test that hangs beside the others holds the suite time gate until the job's own limit"
    );
    // Each runner of the matrix has its own multiplier, and the script gives
    // the reason for each one: a multiplier raised without one passes a pull
    // request that slowed the suite on that runner.
    let script = std::fs::read_to_string(repo().join("scripts/check-suite-time.sh")).unwrap();
    let runners: Vec<&str> = job
        .iter()
        .find_map(|line| line.trim().strip_prefix("os: [")?.strip_suffix(']'))
        .expect("the test job names no runners")
        .split(", ")
        .collect();
    let mut multipliers = Vec::new();
    let mut entry = None;
    for line in job.iter().map(|line| line.trim()) {
        if let Some(os) = line.strip_prefix("- os: ") {
            entry = Some(os);
        } else if let Some(multiplier) = line.strip_prefix("suite_time_multiplier: ") {
            multipliers.push((entry.take().expect("a multiplier outside a runner's entry"), multiplier));
        }
    }
    for runner in runners {
        let (_, multiplier) = multipliers
            .iter()
            .find(|(os, _)| *os == runner)
            .unwrap_or_else(|| panic!("the suite time gate has no multiplier on {runner}"));
        assert!(
            script.contains(&format!("a multiplier of {multiplier} on {runner}")),
            "the script does not give the reason for the multiplier of {multiplier} on {runner}"
        );
    }

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("openmax-suite-gate-{}-{nonce}", std::process::id()));
    let (bin, package, started) = (dir.join("bin"), dir.join("package"), dir.join("started"));
    for made in [&bin, &package, &started] {
        std::fs::create_dir_all(made).unwrap();
    }
    const QUICK: usize = 4;
    let package = package.canonicalize().unwrap();
    let executable = |path: &Path, body: &str| {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(path, body).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    };
    // `cargo test --no-run --message-format=json` as cargo prints it: one
    // artifact per line, the suite's test binaries among others.
    executable(&bin.join("cargo"), "#!/bin/sh\ncat \"$ARTIFACTS\"\n");
    // A test binary that checks it was run as cargo runs one, then waits
    // for the others named `quick-*` to start (which only binaries started
    // at once ever do) and sleeps for `sleep` seconds.
    let probe = |name: &str, sleep: &str| {
        let path = dir.join(name);
        let body = format!(
            "#!/bin/sh\n\
             [ \"$1\" = --test-threads ] || {{ echo \"no thread pool given: $*\"; exit 1; }}\n\
             [ \"$(pwd -P)\" = '{package}' ] || {{ echo \"run from $(pwd -P)\"; exit 1; }}\n\
             perl -e 'exit($SIG{{INT}} eq \"IGNORE\")' || {{ echo 'run with SIGINT ignored'; exit 1; }}\n\
             : > '{started}/{name}'\n\
             waited=0\n\
             case {name} in quick-*) while [ \"$(ls '{started}' | grep -c '^quick-')\" -lt {quick} ]; do\n\
               waited=$((waited + 1)); [ $waited -lt 1000 ] || {{ echo 'the other binaries never started'; exit 1; }}; sleep 0.01\n\
             done ;; esac\n\
             sleep {sleep}\n",
            package = package.display(),
            started = started.display(),
            quick = QUICK,
        );
        executable(&path, &body);
        path
    };
    let failing = dir.join("failing");
    executable(&failing, "#!/bin/sh\necho 'test boom ... FAILED'\nprintf 'failures:\\n\\n---- boom stdout ----\\nit broke\\n'\nexit 101\n");
    // A binary that misses a deadline beside the others and passes alone,
    // as a test with a deadline can on a small runner: it fails its first
    // run (not counting the `--list` before the clock) and passes the next.
    let contended = dir.join("contended");
    let ran = dir.join("contended-ran");
    executable(&contended, &format!("#!/bin/sh\n[ \"$1\" = --list ] && exit 0\n[ -e '{0}' ] && exit 0\n: > '{0}'\necho 'test race ... FAILED'\nexit 101\n", ran.display()));
    let artifact = |name: &str, exe: Option<&Path>| {
        format!(
            r#"{{"reason":"compiler-artifact","manifest_path":"{}/Cargo.toml","target":{{"name":"{name}","kind":["test"]}},"profile":{{"test":{}}},"executable":{}}}"#,
            package.display(),
            exe.is_some(),
            exe.map_or("null".to_string(), |exe| format!("\"{}\"", exe.display())),
        )
    };
    // A perl that cannot load JSON::PP, as on a minimal Linux image, for the
    // gate run that looks for its modules here first.
    let no_json = dir.join("no-json-pp");
    std::fs::create_dir_all(no_json.join("JSON")).unwrap();
    std::fs::write(no_json.join("JSON/PP.pm"), "die \"Can't locate JSON/PP.pm\\n\";\n").unwrap();
    // One run of the gate on the binaries listed for it, its perl looking
    // for modules in `perl_lib` first when one is given.
    let run_gate = |label: &str, multiplier: &str, binaries: &[(&str, Option<&Path>)], perl_lib: Option<&Path>| {
        let listed = dir.join(format!("{label}.jsonl"));
        let lines: Vec<String> = binaries.iter().map(|(name, exe)| artifact(name, *exe)).collect();
        std::fs::write(&listed, lines.join("\n") + "\n").unwrap();
        let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap_or_default());
        let mut gate = Command::new(repo().join("scripts/check-suite-time.sh"));
        gate.arg(multiplier).env("PATH", path).env("ARTIFACTS", &listed).env_remove("GITHUB_ACTIONS");
        if let Some(lib) = perl_lib {
            gate.env("PERL5LIB", lib);
        }
        let out = gate.output().unwrap();
        (out.status.code(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
    };

    let quick: Vec<PathBuf> = (0..QUICK).map(|i| probe(&format!("quick-{i}"), "0")).collect();
    let slow = probe("slow", "0.7");
    let (
        (passed, report),
        (over, over_report),
        (failed, failed_report),
        (usage, _),
        (retried, retried_report),
        (unequipped, unequipped_report),
    ) = std::thread::scope(|scope| {
        let passed = scope.spawn(|| {
            let mut binaries: Vec<(&str, Option<&Path>)> =
                ["a", "b", "c", "d"].iter().zip(&quick).map(|(name, exe)| (*name, Some(exe.as_path()))).collect();
            binaries.push(("built but no test", None));
            run_gate("passed", "1", &binaries, None)
        });
        // A tenth of the target is half a second, which the slow binary
        // outlasts on any machine.
        let over = scope.spawn(|| run_gate("over", "0.1", &[("slow", Some(slow.as_path()))], None));
        let failed = scope.spawn(|| run_gate("failed", "1", &[("broken", Some(failing.as_path()))], None));
        let usage = scope.spawn(|| run_gate("usage", "twice", &[], None));
        let retried = scope.spawn(|| run_gate("retried", "1", &[("contended", Some(contended.as_path()))], None));
        let unequipped = scope.spawn(|| run_gate("unequipped", "1", &[], Some(&no_json)));
        (
            passed.join().unwrap(),
            over.join().unwrap(),
            failed.join().unwrap(),
            usage.join().unwrap(),
            retried.join().unwrap(),
            unequipped.join().unwrap(),
        )
    });
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(passed, Some(0), "binaries that pass, run at once, fail the gate: {report}");
    assert!(report.contains("of its 5.00s budget"), "the target is not 5 seconds: {report}");
    for name in ["a (test)", "b (test)", "c (test)", "d (test)"] {
        assert!(report.contains(name), "{name} is not reported: {report}");
    }
    assert!(!report.contains("built but no test"), "an artifact that is not a test binary was run: {report}");
    assert_eq!(over, Some(1), "a suite over its budget passes the gate: {over_report}");
    assert!(over_report.contains("over its 0.50s budget"), "the budget is not the target times the multiplier: {over_report}");
    assert_eq!(failed, Some(1), "a failing test binary passes the gate: {failed_report}");
    assert!(failed_report.contains("broken (test) FAILED (exit 101)"), "{failed_report}");
    assert!(failed_report.contains("---- boom stdout ----\nit broke"), "the failure is not shown: {failed_report}");
    assert_eq!(usage, Some(2), "a multiplier that is not a number is not refused");
    assert_eq!(retried, Some(0), "a binary that passes alone fails the gate: {retried_report}");
    assert!(
        retried_report.contains("contended (test) failed beside the other binaries and passed alone"),
        "the failure beside the others is not reported: {retried_report}"
    );
    assert_eq!(unequipped, Some(2), "a perl without JSON::PP is not refused: {unequipped_report}");
    assert!(
        unequipped_report.contains("needs perl with its core modules JSON::PP and Time::HiRes"),
        "the gate does not name what it is missing: {unequipped_report}"
    );
}

/// Cold start to the first painted frame under one millisecond.
///
/// Method: 21 launches, after one warmup launch, each into a fresh fixture
/// HOME (settings only, the project not yet trusted, so `--trust-project`
/// grants it) on a fresh 80x24 pseudo-terminal. A launch is timed from just
/// before the spawn to the return of the read that brings the end of the
/// first frame, the close of its synchronized update. The spawn is
/// `posix_spawn` (no `pre_exec`), so the clock does not include forking the
/// test process. The launches run in a session of their own, which has no
/// controlling terminal, so each paints at its pseudo-terminal's 80x24
/// wherever the tests run (`first_frame::alone`). After the frame
/// the launch answers the keyboard protocol query, quits with Ctrl+C twice,
/// and keeps reading until the child is reaped, so it never blocks on unread
/// output. The median is the measurement, printed with the 25th and 75th
/// percentiles. Beside it, the same binary's process start, timed the same
/// way, one run in front of each launch so a change in the machine's load
/// moves both alike: spawn to the first byte that `--version`, printed by
/// the argument parser as it reaches the flag, writes.
///
/// A release build asserts the ratchet, then the target. The ratchet holds
/// the harness's own work in front of the frame, the frame's median beyond
/// the process start's, under three times that work's median measured
/// below, 0.39 ms: room for a busy machine (0.30 ms measured), not for one
/// so loaded that its process start doubles (up to 0.70 ms measured).
/// Painting the frame behind the grant again breaks it: 1.0 to 1.4 ms in
/// five batches. The runtime's start (a median of 0.18 ms) or a settings
/// read (0.10 ms) in front of the frame stays within the spread of this
/// change's own batches (0.00 to 0.31 ms), so this cannot see it. Work in
/// front of argument parsing delays `--version` as much as the frame, so
/// the ratchet cannot see that either; the target can. The target assertion
/// fails by design until the process start itself is under 1 ms: alone it
/// measures 1.41 to 1.53 ms on this platform (below). A debug build prints
/// the measurement and holds the frame to 20 times the target, which only a
/// hang or a stall many times today's breaks; its own timings swing with
/// the machine's load, so it asserts neither the ratchet nor the target.
///
/// Measured in a release build on macOS, five batches of 21 launches each,
/// quoting the batch with the median harness work: the frame painted before
/// any file is read or the runtime starts, first frame 1.66 ms, process
/// start 1.53 ms, so 0.13 ms of harness work (0.08 to 0.14 ms over the
/// batches). With the frame painted behind the grant (trust.json written
/// with its disk syncs), settings, the session and the runtime: 3.20 ms,
/// process start 1.51 ms, so 1.70 ms (1.61 to 1.78 ms, and up to 7.70 ms on
/// a busy machine, as the disk syncs wait on whatever else the disk is
/// doing), which leaves out that build's runtime start because its
/// `--version` paid it too. A project already trusted, launched without the
/// flag: 0.11 ms of harness work, against 0.28 ms behind trust, settings,
/// the session and the runtime.
#[cfg(unix)]
#[test]
fn first_frame_is_under_one_millisecond() {
    use std::time::Duration;
    const TARGET: Duration = Duration::from_millis(1);
    // Three times the harness work measured in front of the frame (above).
    const RATCHET: Duration = Duration::from_micros(3 * 130);
    const LAUNCHES: usize = 21;
    if !first_frame::alone("first_frame_is_under_one_millisecond") {
        return;
    }
    first_frame::version(0);
    first_frame::launch(0);
    let (mut starts, mut times) = (Vec::new(), Vec::new());
    for n in 1..=LAUNCHES {
        starts.push(first_frame::version(n));
        times.push(first_frame::launch(n));
    }
    let summary = |times: &mut [Duration]| {
        times.sort();
        let quartiles = format!(
            "median {:?} (p25 {:?}, p75 {:?})",
            times[LAUNCHES / 2],
            times[LAUNCHES / 4],
            times[3 * LAUNCHES / 4]
        );
        (times[LAUNCHES / 2], quartiles)
    };
    let (start, start_quartiles) = summary(&mut starts);
    let (frame, frame_quartiles) = summary(&mut times);
    let report = format!(
        "first frame: {frame_quartiles} over {LAUNCHES} launches; process start alone: {start_quartiles}"
    );
    println!("{report}");
    if cfg!(debug_assertions) {
        assert!(frame < TARGET * 20, "in a debug build, {report}: {times:?}");
        return;
    }
    let ahead = frame.saturating_sub(start);
    assert!(
        ahead < RATCHET,
        "the harness spends {ahead:?} in front of the first frame, over its ratchet of {RATCHET:?}; {report}: {times:?}"
    );
    assert!(frame < TARGET, "{report}: {times:?}");
}

#[cfg(unix)]
mod first_frame {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    const FRAME_END: &[u8] = b"\x1b[?2026l";
    const KEYBOARD_QUERY: &[u8] = b"\x1b[?u";
    /// A terminal that speaks the keyboard protocol, then its primary device
    /// attributes; then Ctrl+C twice, which quits.
    const ANSWER_AND_QUIT: &[u8] = b"\x1b[?0u\x1b[?62;22c\x03\x03";

    /// Whether this is the run of the test `name` in a session of its own;
    /// if not, run it again in one and pass on how that run ends. crossterm
    /// reads the size through /dev/tty, the session's controlling terminal,
    /// ahead of the terminal on stdout, so a launch in the test's session
    /// paints at the size of whatever terminal runs the tests: from a 250x70
    /// terminal, 0.2 ms more harness work in front of the frame than at
    /// 80x24. A new session has no controlling terminal, so each launch
    /// paints at its pseudo-terminal's size. The rerun forks to start the
    /// session (`pre_exec`), outside every clock, and the launches it times
    /// are `posix_spawn`ed into that session.
    pub fn alone(name: &str) -> bool {
        use std::os::unix::process::CommandExt;
        const ALONE: &str = "OPENMAX_THESIS_ALONE";
        if std::env::var_os(ALONE).is_some() {
            return true;
        }
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", name, "--nocapture", "--test-threads=1"]).env(ALONE, "1");
        // SAFETY: setsid is async-signal-safe, and the closure touches
        // nothing else between fork and exec.
        unsafe {
            cmd.pre_exec(|| match libc::setsid() {
                -1 => Err(std::io::Error::last_os_error()),
                _ => Ok(()),
            });
        }
        let out = cmd.output().unwrap();
        let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        print!("{stdout}");
        assert!(out.status.success(), "in a session of its own: {stdout}{stderr}");
        false
    }

    /// Spawn to the first byte of `openmax --version` on a pseudo-terminal.
    pub fn version(_: usize) -> Duration {
        let (mut controller, terminal) = pseudo_terminal(80, 24);
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_openmax"));
        cmd.arg("--version").stdin(Stdio::null()).stdout(terminal).stderr(Stdio::null());
        let started = Instant::now();
        let mut child = cmd.spawn().unwrap();
        drop(cmd);
        let (mut out, mut first) = (Vec::new(), None);
        let deadline = started + Duration::from_secs(10);
        while first.is_none() && Instant::now() < deadline {
            match read_some(&mut controller, &mut out) {
                Some(read) if !out.is_empty() => first = Some(read - started),
                Some(_) => {}
                None => break,
            }
        }
        while read_some(&mut controller, &mut out).is_some() {}
        child.wait().unwrap();
        first.unwrap_or_else(|| panic!("--version printed nothing: {:?}", String::from_utf8_lossy(&out)))
    }

    /// One launch: spawn to the first frame's end.
    pub fn launch(n: usize) -> Duration {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir()
            .join(format!("openmax-thesis-first-frame-{}-{nonce}-{n}", std::process::id()));
        let (home, project) = (dir.join("home"), dir.join("project"));
        std::fs::create_dir_all(home.join(".openmax")).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(
            home.join(".openmax/settings.json"),
            r#"{"base_url":"http://127.0.0.1:9/v1","model":"stub-model","context_tokens":16384}"#,
        )
        .unwrap();
        let (mut controller, terminal) = pseudo_terminal(80, 24);
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_openmax"));
        cmd.arg("--trust-project")
            .current_dir(&project)
            .env("HOME", &home)
            .env("TERM", "xterm-256color")
            .env_remove("OPENMAX_API_KEY")
            .env_remove("OPENMAX_SESSION")
            .env_remove("OPENMAX_PERF")
            // --trust-project is a human's grant; the test attests to it.
            .env("OPENMAX_HUMAN_ATTEST", "1")
            .stdin(terminal.try_clone().unwrap())
            .stdout(terminal)
            .stderr(Stdio::null());
        let started = Instant::now();
        let mut child = cmd.spawn().unwrap();
        // Only the child holds the terminal end now.
        drop(cmd);
        let (mut out, mut painted, mut answered) = (Vec::new(), None, false);
        let deadline = started + Duration::from_secs(10);
        while Instant::now() < deadline && !answered {
            let Some(read) = read_some(&mut controller, &mut out) else { break };
            if painted.is_none() && find(&out, FRAME_END).is_some() {
                painted = Some(read - started);
            }
            if painted.is_some() && find(&out, KEYBOARD_QUERY).is_some() {
                controller.write_all(ANSWER_AND_QUIT).unwrap();
                answered = true;
            }
        }
        // Reap while reading: a child cannot finish exiting while what it
        // wrote sits unread in its terminal.
        let reap_by = Instant::now() + Duration::from_secs(10);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() > reap_by {
                let _ = child.kill();
            }
            if read_some(&mut controller, &mut Vec::new()).is_none() {
                let _ = child.wait();
                break;
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        painted.unwrap_or_else(|| panic!("no first frame: {:?}", String::from_utf8_lossy(&out)))
    }

    /// Append what the controller has within 20 ms to `out`, and return when
    /// the read returned; None once the child closed the terminal.
    fn read_some(controller: &mut File, out: &mut Vec<u8>) -> Option<Instant> {
        let mut fd = libc::pollfd { fd: controller.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one valid pollfd for the duration of the call.
        if unsafe { libc::poll(&mut fd, 1, 20) } <= 0 {
            return Some(Instant::now());
        }
        let mut buf = [0u8; 64 * 1024];
        match controller.read(&mut buf) {
            Ok(n) if n > 0 => {
                let read = Instant::now();
                out.extend_from_slice(&buf[..n]);
                Some(read)
            }
            // EIO on Linux, EOF on macOS: the child is gone.
            _ => None,
        }
    }

    fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    /// A pseudo-terminal of `cols` by `rows`: the controller end the test
    /// keeps and the terminal end the child gets, both close-on-exec.
    fn pseudo_terminal(cols: u16, rows: u16) -> (File, File) {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::OpenOptionsExt;
        // ptsname answers in one static buffer.
        static PTSNAME: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let open = |path: &std::path::Path| {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_NOCTTY)
                .open(path)
                .unwrap()
        };
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
        let terminal = open(path.as_ref());
        let size = libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
        // SAFETY: TIOCSWINSZ reads one winsize from the pointer.
        let rc = unsafe { libc::ioctl(terminal.as_raw_fd(), libc::TIOCSWINSZ as _, &size) };
        assert_eq!(rc, 0, "TIOCSWINSZ: {}", std::io::Error::last_os_error());
        (controller, terminal)
    }
}

/// A local OpenAI-compatible endpoint that answers every request with the
/// same finished one-line completion, for as long as the test runs.
fn spawn_answering_server() -> String {
    const ANSWER: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"first answer\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            // Headers, then exactly Content-Length body bytes.
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => break,
                }
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
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{ANSWER}",
                    ANSWER.len()
                )
                .as_bytes(),
            );
        }
    });
    format!("http://{addr}/v1")
}

/// A new, empty HOME and project directory: a fresh install.
fn fresh_install(tag: &str, run: usize) -> (PathBuf, PathBuf) {
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("openmax-thesis-{tag}-{}-{run}-{nonce}", std::process::id()));
    let (project, home) = (base.join("project"), base.join("home"));
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    (project, home)
}

/// One command from a fresh install to a first real answer in under 30
/// seconds, carrying the trust grant, the endpoint and the first turn. Nearly
/// all of those 30 seconds are the download and the model's own latency, so
/// what this gates is the harness's share: the wall time of
/// `openmax --trust-project --base-url URL --model ID --api-key '$NAME'
/// --context-tokens N -p TASK` on an empty HOME against a local endpoint that
/// answers at once, from spawn to exit. Method: one warmup run, then the
/// median of 11 runs, each on its own empty HOME and project, each required
/// to exit 0 with the answer on stdout and the endpoint written to
/// settings.json. Release builds assert the median is under 1 second; debug
/// builds print it and assert under 20 seconds.
#[test]
fn one_command_from_a_fresh_install_answers_in_under_thirty_seconds() {
    let base_url = spawn_answering_server();
    let mut runs = Vec::new();
    for run in 0..12 {
        let (project, home) = fresh_install("first-answer", run);
        let started = Instant::now();
        let out = Command::new(env!("CARGO_BIN_EXE_openmax"))
            .current_dir(&project)
            .env("HOME", &home)
            .env_remove("OPENMAX_SESSION")
            .env_remove("OPENMAX_API_KEY")
            // Test automation is a human act with no terminal; the grant
            // needs it said.
            .env("OPENMAX_HUMAN_ATTEST", "1")
            .env("THESIS_KEY", "k")
            .args(["--trust-project", "--base-url", &base_url, "--model", "m", "--api-key", "$THESIS_KEY"])
            .args(["--context-tokens", "32768", "-p", "first task"])
            .output()
            .unwrap();
        let elapsed = started.elapsed();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "{stderr}");
        assert!(String::from_utf8_lossy(&out.stdout).contains("first answer"), "{stderr}");
        let settings = std::fs::read_to_string(home.join(".openmax/settings.json")).unwrap();
        assert!(settings.contains(&base_url), "{settings}");
        let _ = std::fs::remove_dir_all(project.parent().unwrap());
        if run > 0 {
            runs.push(elapsed);
        }
    }
    runs.sort();
    let median = runs[runs.len() / 2];
    let target = Duration::from_secs(1);
    let build = if cfg!(debug_assertions) { "debug" } else { "release" };
    eprintln!("first answer from a fresh install: median {median:?} over {} runs ({build} build)", runs.len());
    if cfg!(debug_assertions) {
        assert!(median < target * 20, "median {median:?}, runs {runs:?}");
    } else {
        assert!(median < target, "median {median:?}, runs {runs:?}");
    }
}
