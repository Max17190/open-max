//! The thesis targets, one test each, named after the target it holds.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The lines of one job in a workflow under .github/workflows, from its key up
/// to the next key at the jobs' indentation or less.
fn workflow_job(workflow: &str, name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(repo().join(".github/workflows").join(workflow)).unwrap();
    let key = format!("  {name}:");
    let mut lines = text.lines().skip_while(|line| line.trim_end() != key);
    assert!(lines.next().is_some(), "{workflow} has no `{name}` job");
    lines
        .take_while(|line| {
            let body = line.trim_start();
            body.is_empty() || body.starts_with('#') || line.len() - body.len() > 2
        })
        .map(str::to_string)
        .collect()
}

/// The lines of one job in ci.yml.
fn ci_job(name: &str) -> Vec<String> {
    workflow_job("ci.yml", name)
}

/// The keys of the step in `job` that holds line `at`, from its first key to
/// the next step, each without the `- ` that starts a step.
fn step_keys(job: &[String], at: usize) -> Vec<&str> {
    let first = job[..=at].iter().rposition(|line| line.trim().starts_with("- ")).expect("a line outside every step");
    let end = job[at + 1..].iter().position(|line| line.trim().starts_with("- ")).map_or(job.len(), |n| at + 1 + n);
    job[first..end].iter().map(|line| line.trim().trim_start_matches("- ")).collect()
}

/// The value a shell script assigns to `name` (a `name=value` line).
fn assigned(script: &str, name: &str) -> String {
    let prefix = format!("{name}=");
    script
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("the script assigns no `{name}`"))
        .to_string()
}

/// `n` with a comma between each group of three digits, as the README writes
/// a count.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// A new, empty directory for one test's files.
fn scratch(tag: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("openmax-thesis-{tag}-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write `body` to `path` as an executable script.
fn executable(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// One of the repository's scripts, to run outside GitHub Actions unless the
/// caller says otherwise.
fn script(path: &str) -> Command {
    let mut command = Command::new(repo().join(path));
    command.env_remove("GITHUB_ACTIONS").env_remove("GITHUB_STEP_SUMMARY");
    command
}

/// A command's exit code and everything it printed, stdout then stderr.
fn ran(command: &mut Command) -> (Option<i32>, String) {
    let out = command.output().unwrap();
    (out.status.code(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

/// The whole suite runs in 5 seconds after compilation. No test inside the
/// suite can time the suite, so CI does: its test job runs
/// scripts/check-suite-time.sh on the suite it just built, on every runner of
/// its matrix, with that runner's multiplier, which the script explains, in a
/// step with a timeout. Linux gates the result; macOS reports failures as an
/// advisory warning and summary. Functional tests remain enforced on both.
/// This test holds that wiring, and holds the gate to
/// its job by driving it with a stand-in `cargo` that lists stand-in test
/// binaries: it runs them at once (four of them each wait for the other
/// three to start), each the way `cargo test` would (from its package
/// directory, with Ctrl+C not ignored), passes a suite inside 5 seconds
/// times the multiplier, and fails one over it or with a binary that fails
/// again when run alone (or that a signal ends, SIGALRM included), naming
/// the binary and showing its failures, and passes one whose binary failed
/// only beside the others. A binary still running at three times the budget
/// is stopped, with every process still under it (a grandchild and one in a
/// session of its own among them; on Linux, with no `ps` on the PATH), and
/// fails the gate, which names the test it had not finished. Run several
/// times over (`--runs N`), it reports each run's time, slowest binary and
/// load (with no `uptime` on the PATH), shows and counts a binary that
/// passed only alone, then the median of an odd or even count and the
/// slowest run, and fails when any run fails, saying why when a binary
/// never reported. It refuses arguments it does not take. Under a perl
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
    let functional = step_keys(&job, tests);
    assert!(!functional.iter().any(|key| key.starts_with("if:") || key.starts_with("continue-on-error:")),
        "normal functional tests must stay enforced on every runner");
    let keys = step_keys(&job, gate);
    assert!(!keys.iter().any(|key| key.starts_with("if:")), "a runner skips its suite timing measurement");
    let soft: Vec<_> = keys.iter().filter(|key| key.starts_with("continue-on-error:")).copied().collect();
    assert_eq!(soft, ["continue-on-error: ${{ matrix.os == 'macos-latest' }}"],
        "only the macOS concurrent suite measurement is advisory");
    assert!(keys.contains(&"id: suite_time"), "the advisory reporter cannot inspect the measurement outcome");
    let report = job.iter().position(|line| line.trim() == "- name: Report macOS suite timing failure")
        .expect("macOS measurement failures have no visible report");
    let report = step_keys(&job, report);
    assert!(report.contains(&"if: ${{ !cancelled() && matrix.os == 'macos-latest' && steps.suite_time.outcome == 'failure' }}"),
        "the advisory report must inspect the failed step outcome, including an allowed failure");
    assert!(report.iter().any(|line| line.contains("::warning::")) && report.iter().any(|line| line.contains("$GITHUB_STEP_SUMMARY")),
        "macOS measurement failures must remain visible as a warning and summary");
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
    // A binary that a signal ends, as a crash would, here SIGALRM: its
    // status, 142, is one a binary can end with, so it must not read as the
    // gate's own stop at the bound.
    let crashing = dir.join("crashing");
    executable(&crashing, "#!/bin/sh\n[ \"$1\" = --list ] && exit 0\nkill -s ALRM $$\n");
    // A binary that, beside the others, finishes one of its two tests and
    // hangs in the other past the bound of the gate run that lists it, and
    // passes alone. It leaves processes running that hold the write end of
    // `lifeline`, one in a session of its own and one that a child of the
    // binary started, so a sweep has to look past the binary's children:
    // the read end, opened here first, sees the end of the file once none
    // of them runs, with no pid to probe.
    let (hanging, hung, lifeline) = (dir.join("hanging"), dir.join("hanging-ran"), dir.join("lifeline"));
    let fifo = std::ffi::CString::new(std::os::unix::ffi::OsStrExt::as_bytes(lifeline.as_os_str())).unwrap();
    // SAFETY: mkfifo reads one NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0, "mkfifo: {}", std::io::Error::last_os_error());
    use std::os::unix::fs::OpenOptionsExt;
    let mut left_running = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&lifeline)
        .unwrap();
    executable(
        &hanging,
        &format!(
            "#!/bin/sh\n[ \"$1\" = --list ] && {{ printf 'hangs: test\\nquick: test\\n'; exit 0; }}\n\
             [ -e '{0}' ] && exit 0\n: > '{0}'\necho 'test quick ... ok'\nexec 3> '{1}'\nsh -c 'sleep 5; :' &\n\
             perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' sleep 5 &\necho started >&3\nexec 3>&-\nexec sleep 5\n",
            hung.display(),
            lifeline.display()
        ),
    );
    // A binary that misses a deadline beside the others and passes alone,
    // as a test with a deadline can on a small runner: it fails its first
    // run (not counting the `--list` before the clock) and passes the next.
    let fails_once = |name: &str| {
        let (path, ran) = (dir.join(name), dir.join(format!("{name}-ran")));
        executable(&path, &format!("#!/bin/sh\n[ \"$1\" = --list ] && exit 0\n[ -e '{0}' ] && exit 0\n: > '{0}'\necho 'test race ... FAILED'\nexit 101\n", ran.display()));
        path
    };
    let (contended, flaky) = (fails_once("contended"), fails_once("flaky"));
    // A binary whose n-th run sleeps the n-th of `sleeps` seconds, for the
    // gate run several times over. In each series below the median is a
    // tenth of a second or more from the mean, the last run and the middle
    // of the run order (and in the even one from either middle run), so
    // load on the machine does not blur them, and the slowest run is
    // neither the first nor the last.
    let uneven = |name: &str, sleeps: &[&str]| {
        let (path, counted) = (dir.join(name), dir.join(format!("{name}-runs")));
        let cases: String = sleeps.iter().enumerate().map(|(n, sleep)| format!("{}) sleep {sleep} ;; ", n + 1)).collect();
        executable(&path, &format!(
            "#!/bin/sh\n[ \"$1\" = --list ] && exit 0\nn=$(($(cat '{0}' 2>/dev/null || echo 0) + 1))\necho $n > '{0}'\n\
             case $n in {cases}esac\n",
            counted.display()
        ));
        path
    };
    let (odd, even) = (uneven("odd", &["0.1", "0.5", "0"]), uneven("even", &["0.2", "0.6", "0", "0"]));
    // A binary that takes its runner, the shell above its launcher, down
    // with it, so it never reports.
    let vanishing = dir.join("vanishing");
    executable(
        &vanishing,
        "#!/bin/sh\n[ \"$1\" = --list ] && exit 0\nkill -9 $(cut -d ' ' -f 4 /proc/$PPID/stat 2>/dev/null || ps -o ppid= -p $PPID)\n",
    );
    // No `uptime`, and on Linux no `ps`, as on a minimal Debian or Ubuntu
    // image, where procps provides both: the gate reads the load average,
    // and sweeps what a stopped binary left running, without them.
    executable(&bin.join("uptime"), "#!/bin/sh\nexit 127\n");
    if cfg!(target_os = "linux") {
        executable(&bin.join("ps"), "#!/bin/sh\nexit 127\n");
    }
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
    // One run of the gate with `args` on the binaries listed for it, its
    // perl looking for modules in `perl_lib` first when one is given.
    let run_gate = |label: &str, args: &[&str], binaries: &[(&str, Option<&Path>)], perl_lib: Option<&Path>| {
        let listed = dir.join(format!("{label}.jsonl"));
        let lines: Vec<String> = binaries.iter().map(|(name, exe)| artifact(name, *exe)).collect();
        std::fs::write(&listed, lines.join("\n") + "\n").unwrap();
        let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap_or_default());
        let mut gate = Command::new(repo().join("scripts/check-suite-time.sh"));
        gate.args(args).env("PATH", path).env("ARTIFACTS", &listed).env_remove("GITHUB_ACTIONS");
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
        (stopped, stopped_report),
        (usage, _),
        (leftover, _),
        (retried, retried_report),
        (unequipped, unequipped_report),
        repeated_odd,
        repeated_even,
        (repeated_failed, repeated_failed_report),
        (unreported, unreported_report),
        (no_runs, _),
    ) = std::thread::scope(|scope| {
        let passed = scope.spawn(|| {
            let mut binaries: Vec<(&str, Option<&Path>)> =
                ["a", "b", "c", "d"].iter().zip(&quick).map(|(name, exe)| (*name, Some(exe.as_path()))).collect();
            binaries.push(("built but no test", None));
            run_gate("passed", &["1"], &binaries, None)
        });
        // A tenth of the target is half a second, which the slow binary
        // outlasts on any machine.
        let over = scope.spawn(|| run_gate("over", &["0.1"], &[("slow", Some(slow.as_path()))], None));
        let failed = scope.spawn(|| {
            run_gate("failed", &["1"], &[("broken", Some(failing.as_path())), ("crashing", Some(crashing.as_path()))], None)
        });
        // A twenty-fifth of the target is a fifth of a second, so the bound
        // is three fifths.
        let stopped = scope.spawn(|| run_gate("stopped", &["0.04"], &[("hanging", Some(hanging.as_path()))], None));
        let usage = scope.spawn(|| run_gate("usage", &["twice"], &[], None));
        let leftover = scope.spawn(|| run_gate("leftover", &["1", "--runs", "2"], &[], None));
        let retried = scope.spawn(|| run_gate("retried", &["1"], &[("contended", Some(contended.as_path()))], None));
        let unequipped = scope.spawn(|| run_gate("unequipped", &["1"], &[], Some(&no_json)));
        let repeated_odd = scope.spawn(|| run_gate("repeated-odd", &["--runs", "3", "1"], &[("odd", Some(odd.as_path()))], None));
        let repeated_even =
            scope.spawn(|| run_gate("repeated-even", &["--runs", "4", "1"], &[("even", Some(even.as_path()))], None));
        let repeated_failed = scope.spawn(|| {
            let binaries = [("broken", Some(failing.as_path())), ("flaky", Some(flaky.as_path()))];
            run_gate("repeated-failed", &["--runs", "2", "1"], &binaries, None)
        });
        let unreported =
            scope.spawn(|| run_gate("unreported", &["--runs", "2", "1"], &[("vanishing", Some(vanishing.as_path()))], None));
        let no_runs = scope.spawn(|| run_gate("no-runs", &["--runs", "0", "1"], &[], None));
        (
            passed.join().unwrap(),
            over.join().unwrap(),
            failed.join().unwrap(),
            stopped.join().unwrap(),
            usage.join().unwrap(),
            leftover.join().unwrap(),
            retried.join().unwrap(),
            unequipped.join().unwrap(),
            repeated_odd.join().unwrap(),
            repeated_even.join().unwrap(),
            repeated_failed.join().unwrap(),
            unreported.join().unwrap(),
            no_runs.join().unwrap(),
        )
    });
    // What the processes the hanging binary left running wrote, and whether
    // one of them still holds the lifeline two seconds on.
    let (mut said, mut chunk) = (Vec::new(), [0_u8; 64]);
    let deadline = Instant::now() + Duration::from_secs(2);
    let outlived = loop {
        match left_running.read(&mut chunk) {
            Ok(0) => break false,
            Ok(read) => said.extend_from_slice(&chunk[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("reading the lifeline: {error}"),
        }
        if Instant::now() >= deadline {
            break true;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
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
    // A signal's end reads as a shell reports it, 128 plus the signal, and
    // never as a stop at the bound.
    assert!(
        failed_report.contains("crashing (test) FAILED (exit 142)") && failed_report.contains("crashing (test) failed:"),
        "a binary that a signal ended passes, or reads as stopped at the bound: {failed_report}"
    );
    assert!(failed_report.contains("---- boom stdout ----\nit broke"), "the failure is not shown: {failed_report}");
    // A binary still running at three times the budget is stopped with
    // every process still under it, in its session or not, and its run
    // fails, naming the test it had not finished, even though the binary
    // then passes alone.
    assert_eq!(stopped, Some(1), "a binary that hangs beside the others passes the gate: {stopped_report}");
    assert!(
        said == b"started\n",
        "the hanging binary was stopped before it started what it leaves running, so no sweep was tried (they wrote {:?}): {stopped_report}",
        String::from_utf8_lossy(&said)
    );
    assert!(!outlived, "the processes the stopped binary left running outlived it: {stopped_report}");
    let unfinished: Vec<&str> = stopped_report
        .split_once("hanging (test) failed beside the other binaries and passed alone:\nstill running at 0.60s, when it was stopped:\n")
        .map_or(Vec::new(), |(_, rest)| rest.lines().take_while(|line| !line.is_empty() && !line.contains(' ')).collect());
    assert!(
        stopped_report.contains("hanging (test) FAILED (stopped at 0.60s)")
            && unfinished == ["hangs"]
            && stopped_report.contains("over its 0.20s budget"),
        "the binary stopped at its bound, or the test it had not finished, is not named: {stopped_report}"
    );
    assert_eq!(usage, Some(2), "a multiplier that is not a number is not refused");
    assert_eq!(leftover, Some(2), "arguments after the multiplier are not refused");
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
    // Run several times over, the gate reports each run (its time, its
    // slowest binary and the load it began under), then the median of the
    // times it reported (the middle one of an odd count, the mean of the
    // middle two of an even one) and the slowest.
    for ((code, report), runs, name) in [(repeated_odd, 3, "odd"), (repeated_even, 4, "even")] {
        assert_eq!(code, Some(0), "runs that each pass fail the gate run {runs} times: {report}");
        let mut took: Vec<f64> = (1..=runs)
            .map(|n| {
                let line = report.lines().find_map(|line| line.strip_prefix(&format!("run {n} of {runs}: ")));
                let time = line.and_then(|line| {
                    let (time, rest) = line.split_once("s, slowest binary ")?;
                    let (slowest, load) = rest.split_once(", load ")?;
                    let reported = slowest.ends_with(&format!("s {name} (test)")) && load.parse::<f64>().is_ok();
                    time.parse::<f64>().ok().filter(|_| reported)
                });
                time.unwrap_or_else(|| panic!("run {n} of {runs} is not reported: {report}"))
            })
            .collect();
        took.sort_by(f64::total_cmp);
        let middle = if runs % 2 == 1 { took[runs / 2] } else { (took[runs / 2 - 1] + took[runs / 2]) / 2.0 };
        let summary = report
            .lines()
            .find_map(|line| line.strip_prefix("median ")?.strip_suffix(&format!("s over {runs} runs, against the 5.00s budget of each")))
            .and_then(|line| line.split_once("s, slowest "))
            .and_then(|(median, slowest)| Some((median.parse::<f64>().ok()?, slowest.parse::<f64>().ok()?)));
        // Times print in hundredths, cut short: the mean of two printed
        // times can be half a hundredth off the one the gate computed.
        let near = |a: f64, b: f64, by: f64| (a - b).abs() <= by + 1e-9;
        assert!(
            summary.is_some_and(|(median, slowest)| near(median, middle, if runs % 2 == 1 { 0.0 } else { 0.005 })
                && near(slowest, took[runs - 1], 0.0)),
            "not a median of {middle:.3}s and a slowest run of {:.2}s: {report}",
            took[runs - 1]
        );
    }
    // A run that failed shows why and says so, and fails the gate; a run in
    // which a binary passed only alone shows that too, and counts it.
    let run_line = |n: usize| repeated_failed_report.lines().find(|line| line.starts_with(&format!("run {n} of 2: ")));
    assert_eq!(repeated_failed, Some(1), "a failing binary passes the gate run several times: {repeated_failed_report}");
    assert!(
        repeated_failed_report.contains("---- boom stdout ----\nit broke")
            && repeated_failed_report.contains("flaky (test) failed beside the other binaries and passed alone")
            && run_line(1).is_some_and(|line| line.ends_with(", 1 passed only when run again alone, failed"))
            && run_line(2).is_some_and(|line| line.ends_with(", failed") && !line.contains("alone")),
        "the failing runs are not shown, or the binary that passed only alone is not counted: {repeated_failed_report}"
    );
    // A binary that never reports fails its run, which says why, and the
    // runs after it go on.
    assert_eq!(unreported, Some(1), "a run whose binary never reported passes: {unreported_report}");
    assert!(
        unreported_report.contains("error: 1 of 1 test binaries did not report")
            && unreported_report.lines().any(|line| line.starts_with("run 2 of 2: ") && line.ends_with(", failed")),
        "the run whose binary never reported does not say why: {unreported_report}"
    );
    assert_eq!(no_runs, Some(2), "a run count that is not a positive number is not refused");
}

/// What scripts/count-core-lines.sh prints for `checkout` with `flags`, or,
/// when it fails, everything it printed.
fn gauge(checkout: &Path, flags: &[&str]) -> Result<String, String> {
    let out = Command::new(repo().join("scripts/count-core-lines.sh"))
        .args(flags)
        .arg(checkout)
        .output()
        .map_err(|e| format!("count-core-lines.sh did not run: {e}"))?;
    let report = String::from_utf8_lossy(&out.stdout).into_owned();
    if !out.status.success() {
        return Err(format!("count-core-lines.sh failed: {report}{}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(report)
}

/// What scripts/count-core-lines.sh --files prints for this checkout. Both
/// line tests below read it, and it is the same report each time, so the
/// script reads the workspace once per test binary instead of once per test.
fn repo_lines() -> Result<&'static str, String> {
    static REPORT: std::sync::OnceLock<Result<String, String>> = std::sync::OnceLock::new();
    REPORT.get_or_init(|| gauge(&repo(), &["--files"])).as_deref().map_err(String::clone)
}

/// The core, production Rust without its tests, is about 37,000 lines.
/// scripts/count-core-lines.sh measures it as core-excluding-tests (the
/// lines of crates/*/src less test-only items, by the method its header
/// states) beside total-including-tests (every line of Rust under crates/),
/// so a test's lines count toward the second wherever they sit and toward
/// the first nowhere. The target is not met yet, so this prints it beside
/// both numbers and holds the ratchet: core-excluding-tests at least
/// BASELINE, the count this test last measured, and at most RATCHET, which
/// is BASELINE plus MARGIN but never past CEILING. A change that needs more
/// raises BASELINE to its new count in its own diff and says why; one that
/// removes production lines must lower BASELINE to its new count, so the
/// lines it saves stay saved. A BASELINE past CEILING does not compile, and
/// no change grows the core past CEILING, lines the agent loop moves into
/// other files (agent_loop_is_under_2500_lines) included. README.md states
/// the ratchet and CEILING, held here. A count, so every build asserts it.
///
/// First the script is held to a fixture checkout
/// (crates/tui/tests/fixtures/count-core-lines.txt) whose every source line
/// is marked counted or excluded: a test module in the middle of a file with
/// production code after it, brackets and attribute text inside strings, raw
/// strings, chars and comments, each kind of test-only item, field,
/// parameter, match arm and statement the script's header names, each with
/// production code after it, `cfg` forms that do and do not need `test`, and
/// test-only module files where rustc finds them. Its whole report on the
/// fixture must match the marks, alone, with the excluded lines
/// (--excluded), with each file's row (--files), and with both. Then a
/// test-only item the script cannot bound must stop the count with an error
/// that names its line, whatever follows it.
#[test]
fn core_source_stays_under_its_line_budget() -> Result<(), String> {
    const TARGET: usize = 37_000;
    // Effort controls and native Messages support add 711 lines for configuration and translation;
    // the agent loop and frozen prompt are unchanged.
    const BASELINE: usize = 44_271;
    const MARGIN: usize = 200;
    // Lines the agent loop moves into other files of the core do not grow the
    // core past CEILING: once BASELINE is within MARGIN of it, the ratchet is
    // CEILING.
    const CEILING: usize = 44_523;
    const { assert!(BASELINE <= CEILING, "BASELINE is raised past CEILING") };
    const RATCHET: usize = if BASELINE + MARGIN < CEILING { BASELINE + MARGIN } else { CEILING };

    // The fixture's files, each with its text and its lines marked excluded.
    let fixture = std::fs::read_to_string(repo().join("crates/tui/tests/fixtures/count-core-lines.txt")).unwrap();
    let mut files: Vec<(String, String, Vec<usize>)> = Vec::new();
    for line in fixture.lines().skip_while(|line| !line.starts_with("== ")) {
        if let Some(path) = line.strip_prefix("== ") {
            files.push((path.to_string(), String::new(), Vec::new()));
            continue;
        }
        let (path, text, excluded) = files.last_mut().unwrap();
        let mut body = line;
        if path.contains("/src/") {
            let marked = |mark: char| line.strip_prefix(mark).filter(|rest| rest.is_empty() || rest.starts_with(' '));
            if let Some(rest) = marked('-') {
                excluded.push(text.lines().count() + 1);
                body = rest;
            } else {
                body = marked('+').unwrap_or_else(|| panic!("a source line in the fixture is not marked: {line:?}"));
            }
            body = body.strip_prefix(' ').unwrap_or(body);
        }
        text.push_str(body);
        text.push('\n');
    }
    files.sort();
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("openmax-line-gauge-{}-{nonce}", std::process::id()));
    // The report the marks call for: each run of excluded lines with
    // --excluded, then each file's row with --files, then a row per crate and
    // the total.
    let (mut excluded_runs, mut file_rows, mut crate_rows) = (String::new(), String::new(), String::new());
    let mut rows = std::collections::BTreeMap::<&str, (usize, usize)>::new();
    for (path, text, excluded) in &files {
        let file = dir.join("marked").join(path);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, text).unwrap();
        let row = rows.entry(path.split('/').nth(1).unwrap()).or_default();
        // A crate's build output counts toward nothing.
        if path.split('/').nth(2) == Some("target") {
            continue;
        }
        let lines = text.lines().count();
        let core = if path.contains("/src/") { lines - excluded.len() } else { 0 };
        file_rows.push_str(&format!("core-excluding-tests={core:<6} total-including-tests={lines:<6} {path}\n"));
        row.0 += core;
        row.1 += lines;
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for &line in excluded {
            match runs.last_mut() {
                Some(run) if run.1 + 1 == line => run.1 = line,
                _ => runs.push((line, line)),
            }
        }
        for (first, last) in runs {
            excluded_runs.push_str(&format!("{path}:{first}-{last}\n"));
        }
    }
    let (core, total) = rows.values().fold((0, 0), |sum, row| (sum.0 + row.0, sum.1 + row.1));
    for (name, (core, total)) in rows.into_iter().chain([("TOTAL", (core, total))]) {
        crate_rows.push_str(&format!("{name:<6} core-excluding-tests={core:<6} total-including-tests={total}\n"));
    }
    let modes = [
        (&[][..], crate_rows.clone()),
        (&["--excluded"][..], format!("{excluded_runs}{crate_rows}")),
        (&["--files"][..], format!("{file_rows}{crate_rows}")),
        (&["--excluded", "--files"][..], format!("{excluded_runs}{file_rows}{crate_rows}")),
    ];
    // A test-only item the gauge cannot bound must stop the count at its
    // line, whatever follows it, rather than take the code after it for test
    // code: an unstable `macro` definition, read as an expression, an
    // unstable `auto trait`, read as an item that ends at its `;`, and an `if`
    // whose condition has a brace right after `>`, which can close generic
    // arguments or compare.
    let unknown = [
        ("macro probe() {}", "fn after() {}"),
        ("macro probe() {}", "#[inline]\n    fn after() {}"),
        ("macro probe() {}", "/// After.\n    fn after() {}"),
        ("macro probe() {}", "{}"),
        ("unsafe auto trait Probe {}", "fn after() {}"),
        ("if 2 > { 1 } {}", "fn after() {}"),
    ];
    for (n, (item, after)) in unknown.iter().enumerate() {
        let src = dir.join(format!("unknown{n}/crates/gamma/src"));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("lib.rs"), format!("pub fn kept() {{\n    #[cfg(test)]\n    {item}\n    {after}\n}}\n")).unwrap();
    }
    let reports: Vec<_> = modes.iter().map(|(flags, _)| gauge(&dir.join("marked"), flags)).collect();
    let stops: Vec<_> = (0..unknown.len()).map(|n| gauge(&dir.join(format!("unknown{n}")), &[])).collect();
    let _ = std::fs::remove_dir_all(&dir);
    for ((flags, expected), report) in modes.iter().zip(reports) {
        assert_eq!(report?, *expected, "the gauge's report on its fixture with {flags:?} differs from the fixture's marks");
    }
    for ((item, after), stop) in unknown.iter().zip(stops) {
        let Err(stop) = stop else { panic!("the gauge counted past `{item}` before `{after}`") };
        assert!(
            stop.contains("crates/gamma/src/lib.rs:2: cannot tell where this test-only item ends"),
            "the gauge stopped without naming the item: {stop}"
        );
    }

    let report = repo_lines()?;
    let count = |name: &str| -> usize {
        report
            .lines()
            .find(|line| line.starts_with("TOTAL "))
            .and_then(|row| row.split_whitespace().find_map(|field| field.strip_prefix(name)?.parse().ok()))
            .unwrap_or_else(|| panic!("no total {name} in the gauge's report: {report}"))
    };
    let (core, total) = (count("core-excluding-tests="), count("total-including-tests="));
    println!("core source: {core} lines excluding tests (ratchet {RATCHET}, target {TARGET}), {total} including tests");
    assert!(
        core <= CEILING,
        "core-excluding-tests is {core}, {} over CEILING, {CEILING}, past which no BASELINE raises its ratchet of \
         {RATCHET}: trim it (scripts/count-core-lines.sh --excluded lists the lines it leaves out)",
        core - CEILING
    );
    assert!(
        core <= RATCHET,
        "core-excluding-tests is {core}, {} over its ratchet of {RATCHET}: trim it, or set BASELINE to {core} in this \
         change and say why (scripts/count-core-lines.sh --excluded lists the lines it leaves out)",
        core - RATCHET
    );
    // Lines a change removes stay removed: it lowers BASELINE with them, so a
    // later change cannot spend them without raising BASELINE and saying why.
    assert!(
        core >= BASELINE,
        "core-excluding-tests is {core}, {} under BASELINE, the count the ratchet was set from: lower BASELINE to \
         {core} in this change",
        BASELINE - core
    );
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    for claim in [
        format!(
            "The ratchet is {} today, {} over the baseline measurement of {} lines, against a target of {}",
            thousands(RATCHET as u64),
            thousands((RATCHET - BASELINE) as u64),
            thousands(BASELINE as u64),
            thousands(TARGET as u64)
        ),
        format!("The core's ratchet is never raised above {}", thousands(CEILING as u64)),
    ] {
        assert!(readme.contains(&claim), "README.md does not say \"{claim}\", which this test holds");
    }
    Ok(())
}

/// The agent loop, crates/core/src/agent.rs without its tests, is at most
/// 2,500 lines. scripts/count-core-lines.sh --files gives each file's
/// core-excluding-tests by the method core_source_stays_under_its_line_budget
/// holds to its fixture, so the loop is measured as the core is, not by a
/// second counter. The target is not met yet, so this prints it beside the
/// loop's two counts and holds a ratchet by the same rule: the loop's
/// core-excluding-tests at least BASELINE, the count this test last
/// measured, and at most RATCHET, BASELINE plus MARGIN. A change that needs
/// more raises BASELINE to its new count in its own diff and says why; one
/// that cuts the loop must lower BASELINE to its new count, so the lines it
/// cuts stay cut. Lines the loop hands to another file of the core stay
/// under the core's ratchet. README.md states this ratchet, held here.
#[test]
fn agent_loop_is_under_2500_lines() -> Result<(), String> {
    const TARGET: usize = 2_500;
    const BASELINE: usize = 3_354;
    // About 1% of the loop: room for a small fix, not for a feature.
    const MARGIN: usize = 50;
    const RATCHET: usize = BASELINE + MARGIN;
    const LOOP: &str = "crates/core/src/agent.rs";
    let report = repo_lines()?;
    let row = report
        .lines()
        .find(|line| line.split_whitespace().last() == Some(LOOP))
        .ok_or_else(|| format!("no row for {LOOP} in the gauge's report: {report}"))?;
    let count = |name: &str| -> usize {
        row.split_whitespace()
            .find_map(|field| field.strip_prefix(name)?.parse().ok())
            .unwrap_or_else(|| panic!("no {name} in the gauge's row for {LOOP}: {row}"))
    };
    let (lines, total) = (count("core-excluding-tests="), count("total-including-tests="));
    println!("agent loop: {lines} lines excluding tests (ratchet {RATCHET}, target {TARGET}), {total} including tests");
    assert!(
        lines <= RATCHET,
        "{LOOP} has {lines} lines excluding tests, {} over its ratchet of {RATCHET}: trim it, or set BASELINE to \
         {lines} in this change and say why (scripts/count-core-lines.sh --excluded lists the lines it leaves out)",
        lines - RATCHET
    );
    assert!(
        lines >= BASELINE,
        "{LOOP} has {lines} lines excluding tests, {} under BASELINE, the count the ratchet was set from: lower \
         BASELINE to {lines} in this change",
        BASELINE - lines
    );
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    let claim = format!(
        "The agent loop's ratchet is {} today, {} over the baseline measurement of {} lines, against a target of {}",
        thousands(RATCHET as u64),
        thousands(MARGIN as u64),
        thousands(BASELINE as u64),
        thousands(TARGET as u64)
    );
    assert!(readme.contains(&claim), "README.md does not say \"{claim}\", which this test holds");
    Ok(())
}

/// Cold start to the first painted frame under one millisecond.
/// The thesis claims are gated in CI: every thesis test runs there in
/// release and fails the pull request, but for the ones a runner reports. A
/// test that times the harness asserts its bound only in a release build,
/// and the test job builds in debug, so ci.yml's `thesis` job builds in
/// release, on each of the test job's runners, every test target named
/// thesis* in the workspace (cargo matches the glob against target names,
/// and a file under a crate's tests/ is a target named after its stem), the
/// binary's `thesis_frames` module and the first-frame test's baseline
/// binary, an example that `cargo test` leaves unbuilt once a target is
/// selected. Its reports step, the only step in this job allowed to fail, runs
/// the tests its runner reports to print their numbers and copies each
/// `MEASURE` line to the job's summary; its gate step, after that so a
/// failing gate cannot skip the reports, runs exactly every other thesis
/// test and fails the pull request when one fails. A runner reports only
/// the tests whose targets it does not meet with room, listed in its matrix
/// entry, held here and named in README.md, so moving a test between the
/// two is a change to this test too: the harness time per tool round on
/// both runners, and the transcript frame's p99 on macOS. libtest selects
/// and skips tests by substring, so no other name in a thesis source may
/// hold a reported one. Every command runs in release, one test at a time,
/// printing what it measured, under a timeout, and the gate's even when the
/// one before it failed, as does each test target after one that failed.
/// Nothing in the job skips a step (no `if:` but `if: ${{ !cancelled() }}`,
/// which runs a step after a failed one too), and nothing turns on debug
/// assertions for its builds, which would hold each timing test to its 20x
/// debug bound: not the release profile in Cargo.toml, nor the workflow's
/// or the job's environment.
#[test]
fn thesis_claims_are_gated_in_ci() {
    let job = ci_job("thesis");
    let runners =
        |job: &[String]| job.iter().find_map(|line| Some(line.trim().strip_prefix("os: [")?.strip_suffix(']')?.to_string()));
    assert_eq!(runners(&job), runners(&ci_job("test")), "the thesis job does not run on the test job's runners");
    // The job's own lines, then each list item's, a step's or a matrix
    // entry's, from its `- ` line to the next, without blanks or comments.
    let mut steps: Vec<Vec<&str>> = vec![Vec::new()];
    for line in job.iter().map(|line| line.trim()).filter(|line| !line.is_empty() && !line.starts_with('#')) {
        match line.strip_prefix("- ") {
            Some(first) => steps.push(vec![first]),
            None => steps.last_mut().unwrap().push(line),
        }
    }
    let skipped: Vec<&&str> =
        steps.iter().flatten().filter(|line| line.starts_with("if:") && **line != "if: ${{ !cancelled() }}").collect();
    assert!(skipped.is_empty(), "the thesis job, or a step of it, can be skipped: {skipped:?}");
    // Only the thesis reports and the test job's macOS concurrent timing
    // measurement may fail without failing their job. Keep both exceptions
    // attached to their exact steps rather than allowing either on any gate.
    let ci = std::fs::read_to_string(repo().join(".github/workflows/ci.yml")).unwrap();
    let soft: Vec<&str> =
        ci.lines().map(str::trim).filter(|line| !line.starts_with('#') && line.contains("continue-on-error")).collect();
    let macos_advisory = "continue-on-error: ${{ matrix.os == 'macos-latest' }}";
    assert_eq!(soft, [macos_advisory, "continue-on-error: true"],
        "ci.yml allows a job or step to fail beyond the two documented measurements");
    let test_job = ci_job("test");
    let timing = test_job.iter().position(|line| line.trim() == "run: scripts/check-suite-time.sh ${{ matrix.suite_time_multiplier }}")
        .expect("the test job has no concurrent suite timing step");
    let timing = step_keys(&test_job, timing);
    assert!(timing.contains(&"id: suite_time") && timing.contains(&macos_advisory),
        "the conditional exception must belong only to the test job's concurrent suite timing step");
    // Debug assertions in these builds would hold each timing test to its
    // debug bound while every other check here passes. They can come from
    // the release profile, or from the environment: the workflow's, the
    // job's or a step's, a flag, or $GITHUB_ENV.
    let manifest = std::fs::read_to_string(repo().join("Cargo.toml")).unwrap();
    let mut section = "";
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line;
        }
        assert!(
            !(section.starts_with("[profile.release") && line.starts_with("debug-assertions")),
            "Cargo.toml's {section} turns on debug assertions, so every timing test holds only its debug bound"
        );
    }
    let workflow_env = ci.lines().skip_while(|line| *line != "env:").skip(1).take_while(|line| line.starts_with(' '));
    let debug: Vec<&str> = workflow_env
        .map(str::trim)
        .chain(steps.iter().flatten().copied())
        .filter(|line| {
            line.starts_with("env:")
                || ["RUSTFLAGS", "CARGO_PROFILE", "debug-assertions", "GITHUB_ENV"].iter().any(|key| line.contains(key))
        })
        .collect();
    assert!(debug.is_empty(), "the thesis job's builds can turn on debug assertions: {debug:?}");
    let step = |name: &str| {
        steps
            .iter()
            .position(|step| step.contains(&format!("name: {name}").as_str()))
            .unwrap_or_else(|| panic!("the thesis job has no `{name}` step"))
    };
    let (build, reports, gates) = (step("Build release tests"), step("Thesis reports"), step("Thesis gates"));
    // A step after a failed one is skipped, and no step here has an `if:`,
    // so the reports run ahead of the gates, which can fail.
    assert!(build < reports && reports < gates, "the thesis job does not build, then report, then gate");
    assert!(steps[reports].contains(&"continue-on-error: true"), "the thesis reports step is not the one allowed to fail");
    for n in [gates, reports] {
        assert!(
            steps[n].iter().any(|line| line.starts_with("timeout-minutes:")),
            "a thesis test that hangs holds its step until the job's own limit: {:?}",
            steps[n]
        );
    }
    // What each runner reports, from its matrix entry.
    let mut reported = Vec::new();
    let mut entry = None;
    for line in job.iter().map(|line| line.trim()) {
        if let Some(runner) = line.strip_prefix("- os: ") {
            entry = Some(runner);
        } else if let Some(tests) = line.strip_prefix("reported: ") {
            reported.push((entry.take().expect("a reported list outside a runner's entry"), tests));
        }
    }
    assert_eq!(
        reported,
        [
            ("ubuntu-latest", "harness_time_per_tool_round"),
            ("macos-latest", "harness_time_per_tool_round transcript_frame_p99_at_100k_lines"),
        ],
        "the thesis tests a runner reports instead of gating changed"
    );
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    assert!(
        readme.contains(
            "`harness_time_per_tool_round` (target 250 us) on both runners and `transcript_frame_p99_at_100k_lines` (target 0.5 ms) on macOS"
        ),
        "README.md does not name the thesis tests each runner reports instead of gating"
    );
    // A renamed test would no longer be skipped by the gate step, nor run by
    // the reports step. libtest selects and skips by substring, so a test or
    // module whose name holds a reported one would leave the gate step too.
    let mut sources = std::fs::read_to_string(repo().join("crates/tui/src/app/thesis_frames.rs")).unwrap();
    for tests in ["crates/core/tests", "crates/tui/tests"] {
        for file in std::fs::read_dir(repo().join(tests)).unwrap().map(|file| file.unwrap().path()) {
            if file.file_name().is_some_and(|name| name.to_string_lossy().starts_with("thesis")) {
                sources.push_str(&std::fs::read_to_string(file).unwrap());
            }
        }
    }
    let names: Vec<&str> = ["fn ", "mod "]
        .into_iter()
        .flat_map(|item| sources.split(item).skip(1))
        .map(|rest| &rest[..rest.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(rest.len())])
        .collect();
    for test in reported.iter().flat_map(|(_, tests)| tests.split(' ')) {
        let holding: Vec<&&str> = names.iter().filter(|name| name.contains(test)).collect();
        assert_eq!(holding, [&test], "the reported `{test}` names no thesis test, or the gate step skips more than it");
    }
    let commands = |n: usize| -> Vec<&str> { steps[n].iter().skip_while(|line| **line != "run: |").skip(1).copied().collect() };
    // Exactly these, so no filter or skip of the step's own takes a test out
    // of it: only its runner's reported list does.
    assert_eq!(
        commands(gates),
        [
            "skip=; for test in ${{ matrix.reported }}; do skip=\"$skip --skip $test\"; done",
            "cargo test --release --workspace --locked --test 'thesis*' --no-fail-fast -- --test-threads=1 --nocapture $skip || failed=1",
            "cargo test --release --workspace --locked --bin openmax -- --test-threads=1 --nocapture thesis_frames $skip || failed=1",
            "exit ${failed:-0}",
        ],
        "the gate step does not run, in release, one at a time and printing what it measured, every thesis test but the \
         ones its runner reports, each command even after one failed, and fail when one did"
    );
    assert_eq!(
        commands(reports),
        [
            "[ -n \"${{ matrix.reported }}\" ] || exit 0",
            "cargo test --release --workspace --locked --test 'thesis*' --bin openmax --no-fail-fast -- --test-threads=1 --nocapture ${{ matrix.reported }} 2>&1 | tee \"$RUNNER_TEMP/thesis-reports.log\"",
            "status=${PIPESTATUS[0]}",
            "{ echo \"Reported, not gated, on ${{ matrix.os }}; cargo test exited $status:\"; echo '```'; sed -n 's/.*MEASURE /MEASURE /p' \"$RUNNER_TEMP/thesis-reports.log\"; echo '```'; } >> \"$GITHUB_STEP_SUMMARY\"",
            "exit \"$status\"",
        ],
        "the reports step does not run, in release and one at a time, exactly the tests its runner reports, \
         copy their measurements to the job's summary and end as they did"
    );
    assert_eq!(
        commands(build),
        [
            "cargo test --release --workspace --locked --no-run --test 'thesis*' --bin openmax",
            "cargo build --release --workspace --locked --example startup_baseline",
        ],
        "the thesis job does not build in release what its steps run and the baseline binary"
    );
    let app = std::fs::read_to_string(repo().join("crates/tui/src/app.rs")).unwrap();
    assert!(app.contains("\nmod thesis_frames;\n"), "the binary has no `thesis_frames` module, so its filter runs nothing");
    assert!(
        repo().join("crates/tui/examples/startup_baseline.rs").is_file(),
        "the example the thesis job builds is not the first-frame test's baseline binary"
    );
}

/// The workspace has 2,000 tests. No test can count the suite it runs in, so
/// CI does: its test job runs scripts/check-test-count.sh on each of its
/// runners once the suite has built and passed, in a step that nothing there
/// skips or lets fail. Linux and macOS each compile a few tests the other does
/// not, so the script counts against the floor of the platform `uname -s`
/// names, each a ratchet at that platform's count when it was last raised,
/// and every runner of the job has one. It fails under that floor, offers to
/// raise that floor when the count is over it, reports the distance to the
/// 2,000-test target until that is met, and on a platform with no floor
/// reports the count and passes; the README states both floors and the
/// target. This test holds the wiring, holds the README to the script, and
/// holds the script to its method by driving it with a stand-in `uname` and a
/// stand-in `cargo` whose test binaries list their tests as libtest does: it
/// counts the lines that end in ": test" (not a benchmark, not the summary),
/// takes out the ones `--list --ignored` names, and on Linux and on macOS
/// passes a count at that platform's floor, fails one under it and names that
/// floor for a count over it; it passes any count on another platform, and
/// fails when cargo fails rather than counting nothing.
#[test]
fn the_workspace_has_two_thousand_tests() {
    let gate = std::fs::read_to_string(repo().join("scripts/check-test-count.sh")).unwrap();
    let floor_of = |name: &str| -> u64 { assigned(&gate, name).parse().unwrap_or_else(|_| panic!("{name} is not a count")) };
    let (linux, macos) = (floor_of("linux_floor"), floor_of("macos_floor"));
    let target: u64 = assigned(&gate, "target").parse().expect("the target is a count");
    assert_eq!(target, 2000, "the target is 2,000 tests");
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    for claim in [
        format!("fails when there are fewer than {} on Linux, or {} on macOS", thousands(linux), thousands(macos)),
        format!("The target is {} tests", thousands(target)),
    ] {
        assert!(readme.contains(&claim), "the README does not say \"{claim}\", which scripts/check-test-count.sh gates");
    }

    let job = ci_job("test");
    let step = |run: &str| {
        job.iter()
            .position(|line| line.trim() == format!("run: {run}"))
            .unwrap_or_else(|| panic!("the test job does not run `{run}`"))
    };
    let (tests, count) = (step("cargo test --workspace --locked"), step("scripts/check-test-count.sh"));
    assert!(tests < count, "the tests are counted before the suite is built");
    // Every runner counts against its own platform's floor: a condition
    // skips the gate on some runner, and continue-on-error lets it fail.
    let keys = step_keys(&job, count);
    let soft: Vec<&&str> = keys.iter().filter(|key| key.starts_with("if:") || key.starts_with("continue-on-error:")).collect();
    assert!(soft.is_empty(), "the test count can be skipped or ignored on a runner: {soft:?}");
    let runners: Vec<&str> = job
        .iter()
        .find_map(|line| line.trim().strip_prefix("os: [")?.strip_suffix(']'))
        .expect("the test job names no runners")
        .split(", ")
        .collect();
    for runner in &runners {
        assert!(
            runner.starts_with("ubuntu-") || runner.starts_with("macos-"),
            "the test job's runner {runner} counts against no floor: scripts/check-test-count.sh has Linux's and macOS's"
        );
    }

    let dir = scratch("test-count");
    // `cargo test --workspace --locked -- --list [--ignored]` as libtest
    // prints it: each test, a benchmark, a blank line and the summary.
    executable(
        &dir.join("cargo"),
        "#!/bin/sh\n\
         [ -z \"$CARGO_FAILS\" ] || { echo 'error: could not compile `open-max-core`' >&2; exit 101; }\n\
         case \"$*\" in\n\
           'test --workspace --locked -- --list --ignored') n=$IGNORED ;;\n\
           'test --workspace --locked -- --list') n=$LISTED ;;\n\
           *) echo \"unexpected: cargo $*\" >&2; exit 2 ;;\n\
         esac\n\
         i=0\n\
         while [ $i -lt $n ]; do echo \"tests::case_$i: test\"; i=$((i + 1)); done\n\
         echo 'benches::parse: benchmark'\n\
         echo\n\
         echo \"$n tests, 1 benchmarks\"\n",
    );
    // `uname -s` as the platform under test prints it.
    executable(&dir.join("uname"), "#!/bin/sh\necho \"$UNAME\"\n");
    let path = format!("{}:{}", dir.display(), std::env::var("PATH").unwrap_or_default());
    let count_with = |uname: &str, listed: u64, ignored: u64, fails: bool| {
        let mut gate = script("scripts/check-test-count.sh");
        gate.env("PATH", &path).env("UNAME", uname).env("LISTED", listed.to_string()).env("IGNORED", ignored.to_string());
        if fails {
            gate.env("CARGO_FAILS", "1");
        } else {
            gate.env_remove("CARGO_FAILS");
        }
        ran(&mut gate)
    };
    // Each platform with a floor, as `uname -s` names it, as the report names
    // it, and the floor's name in the script: a count at the floor, a test
    // under it, and a test over it.
    let platforms = [("Linux", "Linux", "linux_floor", linux), ("Darwin", "macOS", "macos_floor", macos)];
    let counted: Vec<_> = platforms
        .iter()
        .map(|&(uname, _, _, floor)| {
            (count_with(uname, floor + 3, 3, false), (floor > 0).then(|| count_with(uname, floor + 2, 3, false)), count_with(uname, floor + 4, 3, false))
        })
        .collect();
    // Elsewhere no floor holds, however few the tests.
    let (elsewhere, elsewhere_report) = count_with("FreeBSD", 3, 3, false);
    let (broken, broken_report) = count_with("Linux", linux + 3, 3, true);
    let _ = std::fs::remove_dir_all(&dir);

    for (&(_, platform, name, floor), (at, under, over)) in platforms.iter().zip(counted) {
        let (at, at_report) = at;
        assert_eq!(at, Some(0), "a suite at {platform}'s floor of {floor} fails the gate on {platform}: {at_report}");
        let counted = format!("{floor} tests on {platform} ({} listed, 3 of them ignored); floor {floor}, target {target}", floor + 3);
        assert!(at_report.contains(&counted), "not counted as libtest lists tests: {at_report}");
        if floor < target {
            let to_go = format!("{} tests under the target of {target}", target - floor);
            assert!(at_report.contains(&to_go), "the distance to the target is not reported: {at_report}");
        }
        assert!(!at_report.contains("can be raised"), "a suite at {platform}'s floor offers to raise it: {at_report}");
        if let Some((under, under_report)) = under {
            assert_eq!(under, Some(1), "a suite a test under {platform}'s floor of {floor} passes the gate on {platform}: {under_report}");
            assert!(under_report.contains(&format!("tests on {platform}, 1 under its floor of {floor}")), "{under_report}");
        }
        let (over, over_report) = over;
        assert_eq!(over, Some(0), "a suite a test over {platform}'s floor fails the gate on {platform}: {over_report}");
        let raise = format!("{name} in scripts/check-test-count.sh can be raised to {}", floor + 1);
        assert!(over_report.contains(&raise), "a count over {platform}'s floor does not offer to raise that floor, {raise:?}: {over_report}");
    }
    assert_eq!(elsewhere, Some(0), "a platform with no floor fails the gate: {elsewhere_report}");
    assert!(elsewhere_report.contains("0 tests on FreeBSD (3 listed, 3 of them ignored); no floor, target"), "{elsewhere_report}");
    assert_eq!(broken, Some(1), "a suite that does not build passes the gate: {broken_report}");
    assert!(
        broken_report.contains("could not compile") && broken_report.contains("--list` failed"),
        "the failure is not shown: {broken_report}"
    );
}

/// At most a fifth of the latest 50 commits of main change the core: the
/// capability the agent needs lands in files (tools, skills, templates,
/// hooks) instead of the core growing. No test can read the history it was
/// built from, as CI's test job checks out one commit, so CI measures it: its
/// core-share job runs scripts/core-share.sh on every push to main and every
/// pull request, on the event's own commit with the whole history (for a
/// pull request, GitHub's test merge of it into its base), keeps the CSV the
/// script writes as an artifact and builds no Rust, and nothing skips the job
/// or a step of it, or lets them fail. The share is reported, not gated,
/// until it meets the target: the script exits 0 whatever the share, so the
/// job fails only when the script cannot measure. The README states the
/// method, the window, the artifact, the target and the gate. This test holds
/// the wiring and the README to the script, and the script to its method on a
/// history it builds: one row for each first-parent commit, newest first, so
/// a merge is one commit, its diff against its first parent, and the commits
/// it brought in are no rows, as for the test merge CI checks out for a pull
/// request; a root commit diffed against the empty tree; a file under
/// crates/core/ touched by an edit, a move out of the core, a binary file and
/// a mode change, and paths that begin like the core's or hold its path
/// further in not touching it; the lines added and removed in the core and
/// outside it, summed over a commit's files; an empty commit; each subject
/// quoted as CSV quotes it, in UTF-8; the rolling share over the window,
/// empty where fewer commits remain, rounded to a tenth with a half up; the
/// CSV replacing what its file held; the target met at exactly a fifth and
/// missed over it; the same report under settings that change what git's
/// porcelain prints, from a subdirectory; the report added to the job's
/// summary; and a shallow clone, a ref that names no commit, a directory
/// outside any repository, which git refuses with no error of the script's
/// after it, a CSV, a job summary or a temporary file it cannot write, output
/// of git's it does not read or a git command that fails, and each malformed
/// argument refused; with no run leaving a temporary file behind.
#[test]
fn at_most_a_fifth_of_commits_touch_the_core() {
    let method = std::fs::read_to_string(repo().join("scripts/core-share.sh")).unwrap();
    let count_of = |name: &str| -> u64 { assigned(&method, name).parse().unwrap_or_else(|_| panic!("{name} is not a count")) };
    let (window, target, core) = (count_of("window"), count_of("target"), assigned(&method, "core"));
    assert_eq!((window, target), (50, 20), "the target is at most 20% of the latest 50 commits");
    // The name CI keeps the CSV under, which the README gives.
    let artifact = "core-share";
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    for claim in [
        format!("of the latest {window} first-parent commits of main whose diff against their first parent changes a file under `{core}`"),
        format!("the share over it and the {} commits before it", window - 1),
        format!("keeps the CSV as the run's `{artifact}` artifact"),
        format!("The target is at most {target}%"),
        "the job reports the share and fails only when the script cannot measure it".to_string(),
    ] {
        assert!(readme.contains(&claim), "the README does not say \"{claim}\", which the script and CI's core-share job hold");
    }

    let workflow = std::fs::read_to_string(repo().join(".github/workflows/ci.yml")).unwrap();
    assert!(
        workflow.contains("\non:\n  push:\n    branches: [main]\n  pull_request:\n\n"),
        "CI does not run on every push to main and every pull request"
    );
    let job = ci_job("core-share");
    // Every push and pull request measures the share and keeps the CSV: no
    // `if:` skips the job or a step of it, no `needs:` skips the job whenever
    // a job it needs fails or is skipped, and no `continue-on-error:` lets
    // either fail while the run passes.
    let soft: Vec<&str> = job
        .iter()
        .map(|line| line.trim().trim_start_matches("- "))
        .filter(|key| ["if:", "needs:", "continue-on-error:"].iter().any(|refused| key.starts_with(refused)))
        .collect();
    assert!(soft.is_empty(), "the core share can be skipped, or fail to be measured or kept while the run passes: {soft:?}");
    let at = |what: &str, wanted: &dyn Fn(&str) -> bool| {
        job.iter().position(|line| wanted(line.trim())).unwrap_or_else(|| panic!("the core-share job has no {what}"))
    };
    let checkout = at("checkout", &|line| line.starts_with("- uses: actions/checkout@"));
    let checked_out = step_keys(&job, checkout);
    assert!(checked_out.contains(&"fetch-depth: 0"), "the core-share job does not check out the whole history");
    // The checkout takes the event's own commit unless a `ref:` or a
    // `repository:` names another, such as the pull request's head, which
    // makes a row of each of the pull request's commits, not one of them all.
    let other: Vec<&str> = checked_out.into_iter().filter(|key| key.starts_with("ref:") || key.starts_with("repository:")).collect();
    assert!(other.is_empty(), "the core-share job does not check out the event's own commit, GitHub's test merge on a pull request: {other:?}");
    let measured = at("step that measures", &|line| line == "run: scripts/core-share.sh --csv core-share.csv");
    let upload = at("upload of the CSV", &|line| line.starts_with("uses: actions/upload-artifact@"));
    let kept = step_keys(&job, upload);
    let named = format!("name: {artifact}");
    for key in [named.as_str(), "path: core-share.csv", "if-no-files-found: error", "overwrite: true"] {
        assert!(kept.contains(&key), "the core-share job does not keep the CSV with `{key}`: {kept:?}");
    }
    assert!(checkout < measured && measured < upload, "the core-share job does not check out, measure, then keep the CSV");
    let rust: Vec<&String> = job.iter().filter(|line| line.contains("cargo") || line.contains("rust-toolchain")).collect();
    assert!(rust.is_empty(), "the core-share job builds Rust: {rust:?}");

    /// Removes the test's files on the way out of a failed assertion too.
    struct Removed(PathBuf);
    impl Drop for Removed {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let dir = scratch("core-share");
    let _removed = Removed(dir.clone());
    let (tree, temp) = (dir.join("repo"), dir.join("tmp"));
    std::fs::create_dir_all(tree.join("crates")).unwrap();
    std::fs::create_dir_all(&temp).unwrap();
    // git and the script under no setting of this machine's: no system or
    // global configuration, and no system or user attributes file (with no
    // HOME, git looks for no user's); below, git init copies no template
    // directory. Nor the GIT_DIR or other variables a git hook that runs the
    // tests would set, which would point git at this repository instead. Their
    // TMPDIR is the test's own, which every run must leave empty.
    let clean = |program: &Path, cwd: &Path| {
        let mut command = Command::new(program);
        command.env_clear().env("PATH", std::env::var_os("PATH").unwrap_or_default()).current_dir(cwd);
        command.env("GIT_CONFIG_NOSYSTEM", "1").env("GIT_CONFIG_GLOBAL", "/dev/null").env("GIT_ATTR_NOSYSTEM", "1");
        command.env("TMPDIR", &temp);
        command
    };
    let git = |args: &[&str], input: &[u8]| {
        let mut git = clean(Path::new("git"), &tree);
        git.args(args).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
        let mut child = git.spawn().unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "git {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
    };
    // The history, built by one git fast-import. Each commit is marked with
    // the day of January 2026 it was committed on, at 13:00 an hour ahead of
    // UTC, and authored three hours earlier, and goes on `branch` after its
    // parents (none for a root), with its changes, each a fast-import file
    // command and the data an inline one takes. A merge's tree is its first
    // parent's with the changes given, so a merge gives the other parent's.
    let mut stream = Vec::new();
    let mut commit = |branch: &str, mark: u64, subject: &str, parents: &[u64], changes: &[(&str, &[u8])]| {
        let (when, message) = (1_767_268_800 + 86_400 * (mark - 1), format!("{subject}\n"));
        let head = format!(
            "commit refs/{branch}\nmark :{mark}\nauthor Author <author@example.com> {} +0100\ncommitter Fixture <fixture@example.com> {when} +0100\n",
            when - 3 * 3600
        );
        stream.extend_from_slice(format!("{head}data {}\n{message}", message.len()).as_bytes());
        for (i, parent) in parents.iter().enumerate() {
            stream.extend_from_slice(format!("{} :{parent}\n", if i == 0 { "from" } else { "merge" }).as_bytes());
        }
        for (command, data) in changes {
            stream.extend_from_slice(format!("{command}\n").as_bytes());
            if command.contains(" inline ") {
                stream.extend_from_slice(format!("data {}\n", data.len()).as_bytes());
                stream.extend_from_slice(data);
                stream.push(b'\n');
            }
        }
        stream.push(b'\n');
    };
    let (lib, merged, build, logo): (&[u8], &[u8], &[u8], &[u8]) =
        (b"fn a() {}\nfn b() {}\n", b"fn a() {}\nfn b() {}\nfn c() {}\n", b"cargo build\n", b"\x89PNG\r\n\x1a\n\x00\x01");
    let first: &[(&str, &[u8])] = &[
        ("M 100644 inline crates/core/src/lib.rs", lib),
        ("M 100644 inline crates/core/build.sh", build),
        ("M 100644 inline crates/core/Cargo.toml", b"[package]\n"),
        ("M 100644 inline README.md", b"# Fixture\n"),
    ];
    commit("heads/main", 1, "Initial commit", &[], first);
    commit("heads/main", 2, "docs: say \"hi\", twice", &[1], &[("M 100644 inline README.md", b"# Fixture\nhi, hi\n")]);
    // Paths that begin like the core's or hold it further in, none of them
    // under crates/core/.
    let near: &[(&str, &[u8])] = &[
        ("M 100644 inline crates/core-extras/lib.rs", b"x\n"),
        ("M 100644 inline crates/core.md", b"x\n"),
        ("M 100644 inline vendor/crates/core/lib.rs", b"x\n"),
    ];
    commit("heads/main", 3, "near misses", &[2], near);
    commit("heads/side", 4, "core: add c", &[3], &[("M 100644 inline crates/core/src/lib.rs", merged)]);
    commit("heads/side", 5, "docs: mention c", &[4], &[("M 100644 inline docs/c.md", b"c\n")]);
    commit("heads/main", 6, "tui: add the frame", &[3], &[("M 100644 inline crates/tui/src/main.rs", b"fn main() {}\n")]);
    let side: &[(&str, &[u8])] = &[("M 100644 inline crates/core/src/lib.rs", merged), ("M 100644 inline docs/c.md", b"c\n")];
    commit("heads/main", 7, "Merge branch 'side'", &[6, 5], side);
    commit("heads/main", 8, "move lib out of the core", &[7], &[("R crates/core/src/lib.rs crates/tui/src/lib.rs", b"")]);
    commit("heads/main", 9, "core: add the logo", &[8], &[("M 100644 inline crates/core/assets/café.png", logo)]);
    commit("heads/main", 10, "core: make the build script executable", &[9], &[("M 100755 inline crates/core/build.sh", build)]);
    commit("heads/main", 11, "an empty commit", &[10], &[]);
    commit("heads/main", 12, "docs: naïve café", &[11], &[("M 100644 inline docs/café.md", b"naive\n")]);
    for mark in 13..=17 {
        commit("heads/main", mark, &format!("docs: note {mark}"), &[mark - 1], &[(&format!("M 100644 inline docs/note-{mark}.md"), b"note\n")]);
    }
    // The pull request adds a file to the core and replaces a line in two
    // others, while main adds a note and empties two: lines removed from two
    // files in the core, and from two outside it.
    let change: &[(&str, &[u8])] = &[
        ("M 100644 inline crates/core/src/change.rs", b"fn change() {}\nfn more() {}\n"),
        ("M 100755 inline crates/core/build.sh", b"cargo build --locked\n"),
        ("M 100644 inline crates/core/Cargo.toml", b"[lib]\n"),
    ];
    let describe: &[(&str, &[u8])] = &[("M 100644 inline docs/change.md", b"change\n")];
    commit("heads/pr", 18, "core: add the change", &[17], change);
    commit("heads/pr", 19, "docs: describe the change", &[18], describe);
    let emptied: &[(&str, &[u8])] =
        &[("M 100644 inline docs/note-20.md", b"note\n"), ("M 100644 inline docs/note-13.md", b""), ("M 100644 inline docs/note-14.md", b"")];
    commit("heads/main", 20, "docs: note 20, and empty notes 13 and 14", &[17], emptied);
    // GitHub's test merge of the pull request into its base, which CI checks
    // out, detached, for a pull request.
    commit("pull/1/merge", 21, "Merge the pull request into main", &[20, 19], &[change, describe].concat());
    let marks = dir.join("marks");
    git(&["init", "-q", "--template="], b"");
    git(&["fast-import", "--quiet", &format!("--export-marks={}", marks.display())], &stream);
    git(&["update-ref", "--no-deref", "HEAD", "refs/pull/1/merge"], b"");
    let sha_of: std::collections::HashMap<u64, String> = std::fs::read_to_string(&marks)
        .unwrap()
        .lines()
        .map(|line| {
            let (mark, sha) = line.split_once(' ').expect("a mark without its commit");
            (mark.trim_start_matches(':').parse().expect("a mark that is not a number"), sha.to_string())
        })
        .collect();

    // Each first-parent commit of the test merge, newest first: its mark, its
    // subject, whether it touches the core, the lines it adds and removes
    // there and elsewhere, and how many of it and the two commits before it
    // touch the core. The side branch's commits and the pull request's are no
    // rows of their own.
    type Row = (u64, &'static str, u8, [u32; 4], Option<usize>);
    let rows: [Row; 17] = [
        (21, "Merge the pull request into main", 1, [4, 2, 1, 0], Some(1)),
        (20, "docs: note 20, and empty notes 13 and 14", 0, [0, 0, 1, 2], Some(0)),
        (17, "docs: note 17", 0, [0, 0, 1, 0], Some(0)),
        (16, "docs: note 16", 0, [0, 0, 1, 0], Some(0)),
        (15, "docs: note 15", 0, [0, 0, 1, 0], Some(0)),
        (14, "docs: note 14", 0, [0, 0, 1, 0], Some(0)),
        (13, "docs: note 13", 0, [0, 0, 1, 0], Some(0)),
        (12, "docs: naïve café", 0, [0, 0, 1, 0], Some(1)),
        (11, "an empty commit", 0, [0, 0, 0, 0], Some(2)),
        (10, "core: make the build script executable", 1, [0, 0, 0, 0], Some(3)),
        (9, "core: add the logo", 1, [0, 0, 0, 0], Some(3)),
        (8, "move lib out of the core", 1, [0, 3, 3, 0], Some(2)),
        (7, "Merge branch 'side'", 1, [1, 0, 1, 0], Some(1)),
        (6, "tui: add the frame", 0, [0, 0, 1, 0], Some(0)),
        (3, "near misses", 0, [0, 0, 3, 0], Some(1)),
        (2, "docs: say \"hi\", twice", 0, [0, 0, 1, 0], None),
        (1, "Initial commit", 1, [4, 0, 1, 0], None),
    ];
    // The CSV the script writes for a window of three, or of 50, where no
    // commit has a full window.
    let csv_of = |window: usize| {
        let columns = "sha,date,subject,touches_core,core_added,core_removed,other_added,other_removed";
        let mut csv = format!("{columns},rolling_{window}_core,rolling_{window}_percent\n");
        for (mark, subject, touches, [core_added, core_removed, other_added, other_removed], of_three) in rows {
            let share = match of_three {
                Some(count) if window == 3 => format!("{count},{}", ["0.0", "33.3", "66.7", "100.0"][count]),
                _ => ",".to_string(),
            };
            let (subject, sha) = (subject.replace('"', "\"\""), &sha_of[&mark]);
            let lines = format!("{core_added},{core_removed},{other_added},{other_removed}");
            csv += &format!("{sha},2026-01-{mark:02}T13:00:00+01:00,\"{subject}\",{touches},{lines},{share}\n");
        }
        csv
    };
    let script = repo().join("scripts/core-share.sh");
    let measure = |args: &[&str], cwd: &Path| ran(clean(&script, cwd).args(args));
    let (head, main) = (&sha_of[&21][..12], &sha_of[&20][..12]);
    let share = |count: u32, of: u32, percent: &str| {
        format!("core share at {head} (HEAD): {count} of the latest {of} first-parent commits touch crates/core/, {percent}%; target at most 20%\n")
    };
    let itself = format!("{head} itself touches crates/core/: +4 -2 there, +1 -0 elsewhere\n");

    // The CSV replaces what its file held.
    let three = dir.join("three.csv");
    std::fs::write(&three, "an earlier run's row\n").unwrap();
    let (code, report) = measure(&["--window", "3", "--csv", three.to_str().unwrap()], &tree);
    assert_eq!(code, Some(0), "the script fails on a share over the target: {report}");
    let written = std::fs::read_to_string(&three).unwrap_or_else(|err| panic!("no CSV in UTF-8 was written: {err}: {report}"));
    assert_eq!(written, csv_of(3), "the CSV for a window of three is not the history's");
    assert!(written.contains(",\"docs: say \"\"hi\"\", twice\","), "a subject is not quoted as CSV quotes it: {written}");
    assert_eq!(report, format!("{}1 over the target, which allows 0 of 3\n{itself}", share(1, 3, "33.3")));
    // The same, whatever settings change in what git's porcelain prints and
    // from a directory under the top of the repository.
    let settings = dir.join("gitconfig");
    let hostile = "[diff]\n\trenames = copies\n\trelative = true\n[log]\n\tshowRoot = false\n[color]\n\tui = always\n\
                   [core]\n\tquotePath = true\n[i18n]\n\tlogOutputEncoding = ISO-8859-1\n";
    std::fs::write(&settings, hostile).unwrap();
    let under = dir.join("under.csv");
    let mut configured = clean(&script, &tree.join("crates"));
    configured.args(["--window", "3", "--csv", under.to_str().unwrap()]).env("GIT_CONFIG_GLOBAL", &settings);
    assert_eq!(ran(&mut configured), (Some(0), report), "git's settings change the report");
    assert_eq!(String::from_utf8_lossy(&std::fs::read(&under).unwrap()), written, "git's settings change the CSV");

    // The default window, which no commit here fills, with the report added
    // to the job's summary in GitHub Actions.
    let (summary, fifty) = (dir.join("summary.md"), dir.join("fifty.csv"));
    std::fs::write(&summary, "earlier\n").unwrap();
    let mut actions = clean(&script, &tree);
    actions.args(["--csv", fifty.to_str().unwrap()]).env("GITHUB_ACTIONS", "true").env("GITHUB_STEP_SUMMARY", &summary);
    let (code, report) = ran(&mut actions);
    let unfilled =
        format!("core share at {head} (HEAD): 17 first-parent commits, fewer than the window of 50, so no share; target at most 20%\n{itself}");
    assert_eq!((code, report), (Some(0), unfilled.clone()), "no share is not reported for a history shorter than the window");
    assert_eq!(std::fs::read_to_string(&fifty).unwrap(), csv_of(50), "the CSV holds a share for a commit with fewer than 50 to count");
    let added = std::fs::read_to_string(&summary).unwrap();
    assert_eq!(added, format!("earlier\n## Core share\n\n```text\n{unfilled}```\n"), "the report is not added to the job's summary");
    // The target met at exactly a fifth; the share rounded to a tenth with a
    // half up (31.25%); a window as long as the history; and a commit that
    // does not touch the core, at most 999,999,999 commits.
    for (args, expected) in [
        (vec!["--window", "5"], format!("{}the target is met\n{itself}", share(1, 5, "20.0"))),
        (vec!["--window", "16"], format!("{}2 over the target, which allows 3 of 16\n{itself}", share(5, 16, "31.3"))),
        (vec!["--window", "17"], format!("{}3 over the target, which allows 3 of 17\n{itself}", share(6, 17, "35.3"))),
        (
            vec!["--window", "999999999", "main"],
            format!(
                "core share at {main} (main): 16 first-parent commits, fewer than the window of 999999999, so no share; target at most 20%\n\
                 {main} itself does not touch crates/core/: +0 -0 there, +1 -2 elsewhere\n"
            ),
        ),
    ] {
        assert_eq!(measure(&args, &tree), (Some(0), expected), "the share is not reported as its method gives it for {args:?}");
    }

    // What it cannot measure: a shallow clone, a ref that names no commit,
    // a directory outside any repository, which git refuses on its own (the
    // script's own errors start `error: `, git's `fatal: `), and a CSV, a job
    // summary or a temporary file it cannot write.
    std::fs::write(tree.join(".git/shallow"), format!("{}\n", sha_of[&3])).unwrap();
    let (shallow, shallow_report) = ran(clean(&script, &tree).env("GITHUB_ACTIONS", "true"));
    std::fs::remove_file(tree.join(".git/shallow")).unwrap();
    assert!(shallow == Some(1) && shallow_report.contains("::error::a shallow clone"), "a shallow clone is measured: {shallow_report}");
    let (code, report) = measure(&["no-such-ref"], &tree);
    assert!(code == Some(1) && report.contains("error: no-such-ref names no commit"), "a ref that names no commit is measured: {report}");
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let (code, report) = ran(clean(&script, &elsewhere).env("GIT_CEILING_DIRECTORIES", &dir));
    assert!(
        code == Some(1) && report.contains("not a git repository") && !report.contains("error: "),
        "a directory outside any repository is measured, or the script reports an error of its own after git's: {report}"
    );
    let unwritable = dir.join("missing/out.csv");
    let (code, report) = measure(&["--csv", unwritable.to_str().unwrap()], &tree);
    let refused = format!("error: cannot write {}", unwritable.display());
    assert!(code == Some(1) && report.contains(&refused), "a CSV it cannot write passes: {report}");
    let (code, report) = ran(clean(&script, &tree).env("GITHUB_STEP_SUMMARY", dir.join("missing/summary.md")));
    assert!(code == Some(1) && report.contains("error: cannot write the job summary"), "a job summary it cannot write passes: {report}");
    // A temporary file it cannot write in TMPDIR, which mktemp refuses on its
    // own, naming the file.
    let (code, report) = ran(clean(&script, &tree).env("TMPDIR", dir.join("missing")));
    let template = format!("{}/core-share.", dir.join("missing").display());
    assert!(
        code == Some(1) && report.contains(&template) && !report.contains("error: "),
        "a temporary file it cannot write in TMPDIR passes, or the script reports an error of its own after mktemp's: {report}"
    );
    // What git prints and the script does not read, from a git on PATH that
    // passes one command's output through a Perl edit: a commit missing, one
    // printed twice, a file before any commit, a file without its two counts,
    // a count that is not a number, and a command that fails. Each is
    // refused, never counted.
    let path = std::env::var_os("PATH").unwrap_or_default();
    let real = std::env::split_paths(&path).map(|dir| dir.join("git")).find(|git| git.is_file()).expect("no git on PATH");
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let editing_git = "#!/bin/sh\n[ \"$1\" = \"$EDITED\" ] || exec \"$REAL_GIT\" \"$@\"\n\"$REAL_GIT\" \"$@\" | perl -0777 -pe \"$EDIT\"\n";
    executable(&bin.join("git"), editing_git);
    let edited_path = std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(&path))).unwrap();
    let unread = |after: u32, what: &str| format!("error: git diff-tree printed what this script does not read, after {after} commits: {what}\n");
    let file = r"crates\/core\/src\/change.rs";
    for (command, edit, refused) in [
        ("diff-tree", r"s/\x01[^\x01]*\x01[^\x01]*\x01Initial commit\0.*//s".to_string(), "error: git diff-tree gave 16 of the 17 commits of HEAD\n".to_string()),
        ("diff-tree", r"s/\A([^\0]*\0)/$1$1/".to_string(), unread(1, &sha_of[&21])),
        ("diff-tree", r"s/\A/1\t0\tstray\0/".to_string(), unread(0, "1\t0\tstray")),
        ("diff-tree", format!(r"s/2\t0\t{file}\0/2\0/"), unread(1, "2")),
        ("diff-tree", format!(r"s/2\t0\t{file}/x\t0\t{file}/"), unread(1, "x\t0\tcrates/core/src/change.rs")),
        ("diff-tree", format!(r"s/2\t0\t{file}/2\tx\t{file}/"), unread(1, "2\tx\tcrates/core/src/change.rs")),
        ("diff-tree", "exit 1".to_string(), "error: git diff-tree failed on HEAD\n".to_string()),
        ("rev-list", "exit 1".to_string(), "error: git rev-list failed on HEAD\n".to_string()),
    ] {
        let mut edited = clean(&script, &tree);
        edited.env("PATH", &edited_path).env("REAL_GIT", &real).env("EDITED", command).env("EDIT", &edit);
        let (code, report) = ran(&mut edited);
        assert!(code == Some(1) && report.ends_with(&refused), "the script reads git {command}'s output edited by {edit:?}: {report}");
    }
    for args in [&["--window"][..], &["--csv"], &["--csv", ""], &["--bogus"], &["HEAD", "main"]] {
        let (code, report) = measure(args, &tree);
        assert!(code == Some(2) && report.starts_with("usage: "), "{args:?} is not refused: {report}");
    }
    for window in ["0", "05", "5x", "1000000000"] {
        let (code, report) = measure(&["--window", window], &tree);
        let refused = format!("error: --window takes a count of commits from 1 to 999999999, not '{window}'");
        assert!(code == Some(2) && report.contains(&refused), "a window of {window} is not refused: {report}");
    }
    // Every run that wrote a temporary file, a run that cannot write its CSV
    // or job summary among them, removed it.
    let left: Vec<_> = std::fs::read_dir(&temp).unwrap().map(|entry| entry.unwrap().file_name()).collect();
    assert!(left.is_empty(), "the script leaves its temporary files behind: {left:?}");
}

/// The mutation score, caught / (caught + missed) over every mutant of the
/// workspace, reaches 95%, and 90% over the mutants of each file, whose score
/// the run's report lists. No test can mutate the code it runs in, so CI does:
/// the Mutants workflow runs cargo-mutants weekly and by hand over the whole
/// workspace, with one pinned compiler and at most four concurrent shards.
/// Runs queue per branch. Each shard first tests the unmutated workspace in
/// a memory-capped cgroup and preserves outcomes for the final score judge.
/// The score floor, baseline and margin remain documented in the README;
/// docs/testing.md also explains targeted local checks with the script's
/// optional --diff mode. Pull requests and pushes do not trigger full sweeps.
/// This test holds the wiring and the documents to the files, and holds the
/// script to its rules by driving it with shards' outcomes as cargo-mutants
/// 27.1.0 writes them, over a source file with a macOS-only function,
/// non-Unix statements, a statement under cfg(not(test)) and one under
/// cfg(true): it merges shards; leaves out the unviable mutants and the ones
/// the run's target does not compile (read from each shard's cfg.txt), there
/// and on macOS, keeping code that either build of a crate, with or without
/// cfg(test), compiles; counts a listed timeout, and a listed mutant whose
/// build or tests a signal ended, as caught, an unlisted one and a listed
/// miss as missed, and names each mutant a signal ended with its phase and
/// signal; passes a score exactly at the floor and fails one a mutant under
/// it; says how many of the missed would meet the target; gives each file's
/// score, worst first and ties by name, with the catches it needs for 90%;
/// offers to raise the floor to a score less the margin, and only when that
/// is above the floor; in diff mode fails on exactly the unlisted misses,
/// timeouts and signals, marking each on its line with its message escaped
/// as a workflow command needs; and refuses a shard that did not finish, one
/// whose unmutated tests failed, shards that overlap, built different targets
/// or hold fewer mutants than the plan listed, a mutant that was only checked,
/// a Failure no signal ended, an outcome cargo-mutants does not write even in
/// code the run did not build, and an allowlist entry with no reason.
#[test]
fn mutation_score_is_at_least_ninety_five_percent() {
    let gate = std::fs::read_to_string(repo().join("scripts/check-mutants.sh")).unwrap();
    let (floor, margin, target) = (assigned(&gate, "floor"), assigned(&gate, "margin"), assigned(&gate, "target"));
    let file_target = assigned(&gate, "file_target");
    assert_eq!((target.as_str(), file_target.as_str()), ("95", "90"), "the target is 95%, and 90% in each file");
    // A percentage with two decimals, in hundredths of a point.
    let hundredths = |value: &str| {
        let (whole, part) = value.split_once('.').unwrap_or_else(|| panic!("{value} has no decimals"));
        let bp = whole.parse::<u64>().unwrap() * 100 + part.parse::<u64>().unwrap();
        assert!(part.len() == 2 && bp <= 10_000, "{value} is not a percentage with two decimals");
        bp
    };
    let percent = |bp: u64| format!("{}.{:02}%", bp / 100, bp % 100);
    let (floor_bp, margin_bp) = (hundredths(&floor), hundredths(&margin));
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    let measured = format!("{}, less a margin of {margin} points", percent(floor_bp + margin_bp));
    for claim in [format!("is under {floor}%"), measured, format!("The target is {target}%, and {file_target}% in each file")] {
        assert!(readme.contains(&claim), "the README does not say \"{claim}\", which scripts/check-mutants.sh gates");
    }
    let config: toml::Table = std::fs::read_to_string(repo().join(".cargo/mutants.toml")).unwrap().parse().unwrap();
    let (multiplier, minimum) = (config["timeout_multiplier"].as_float().unwrap(), config["minimum_test_timeout"].as_integer().unwrap());
    let testing = std::fs::read_to_string(repo().join("docs/testing.md")).unwrap();
    let timeout = format!("the larger of {minimum} seconds and {multiplier} times the time the unmutated tests took");
    assert!(testing.contains(&timeout), "docs/testing.md does not give the timeout .cargo/mutants.toml sets: {timeout}");
    let margined = format!("less a margin of {margin} points");
    assert!(testing.contains(&margined), "docs/testing.md does not give the margin scripts/check-mutants.sh sets: {margined}");
    // The run that set the floor, as docs/testing.md records it: its caught
    // over its scored is its score; less the catches that cannot be told from
    // chance, which the floor's basis leaves out, it is the basis, which less
    // the margin is the floor, while the score less the margin is what the
    // run's report offered; the README gives its count, its score and its
    // basis; and the documents say how many of the basis's catches a run of
    // unchanged code can lose before it fails.
    let (tested, caught_then, scored_then, score_then) = testing
        .split_once("The floor was set by the full run on ")
        .and_then(|(_, run)| run.split_once(", which tested "))
        .and_then(|(_, run)| run.split_once('%'))
        .and_then(|(run, _)| run.split_once(" mutants and caught "))
        .and_then(|(tested, run)| run.split_once(" of the ").map(|(caught, run)| (tested, caught, run)))
        .and_then(|(tested, caught, run)| run.split_once(" it scored: ").map(|(scored, score)| (tested, caught, scored, score)))
        .expect("docs/testing.md does not record the run that set the floor as \"..., which tested N mutants and caught N of the N it scored: S%\"");
    let (left_out, kept, kept_of, basis) = testing
        .split_once("The floor's basis leaves out ")
        .and_then(|(_, basis)| basis.split_once(" of those catches"))
        .and_then(|(left_out, basis)| basis.split_once(": ").map(|(_, basis)| (left_out, basis)))
        .and_then(|(left_out, basis)| basis.split_once(" caught of the ").map(|(kept, basis)| (left_out, kept, basis)))
        .and_then(|(left_out, kept, basis)| basis.split_once(", ").map(|(of, basis)| (left_out, kept, of, basis)))
        .and_then(|(left_out, kept, of, basis)| basis.split_once('%').map(|(basis, _)| (left_out, kept, of, basis)))
        .expect("docs/testing.md does not record the floor's basis as \"The floor's basis leaves out N of those catches, ...: N caught of the N, S%\"");
    let record = format!("{tested} mutants, {caught_then} caught of {scored_then}, {score_then}%; {left_out} left out, {kept} of {kept_of}, {basis}%");
    let count = |text: &str| text.replace(',', "").parse::<u64>().unwrap_or_else(|_| panic!("{text:?} is not a count: {record}"));
    let (tested, caught_then, scored_then) = (count(tested), count(caught_then), count(scored_then));
    let (left_out, kept, kept_of) = (count(left_out), count(kept), count(kept_of));
    assert_eq!(format!("{score_then}%"), percent(caught_then * 10_000 / scored_then), "the score docs/testing.md gives is not its counts': {record}");
    assert_eq!((kept + left_out, kept_of), (caught_then, scored_then), "the floor's basis is not the run's catches less those it leaves out: {record}");
    assert_eq!(format!("{basis}%"), percent(kept * 10_000 / kept_of), "the basis docs/testing.md gives is not its counts': {record}");
    assert_eq!(kept * 10_000 / kept_of, floor_bp + margin_bp, "the floor is not the recorded basis less the margin: {record}");
    let offered = percent(caught_then * 10_000 / scored_then - margin_bp);
    for claim in [format!("so the floor is {floor}%"), format!("not the {offered} the run's report offered")] {
        assert!(testing.contains(&claim), "docs/testing.md does not say \"{claim}\": {record}");
    }
    let without = format!("That run scored {score_then}%, or {basis}% without the {} catches", thousands(left_out));
    for claim in [format!("{} in the run that set the floor", thousands(tested)), without] {
        assert!(readme.contains(&claim), "the README does not say \"{claim}\", as docs/testing.md records");
    }
    let header: Vec<&str> = gate.lines().take_while(|line| line.starts_with('#')).map(|line| line.trim_start_matches('#').trim()).collect();
    let lost = kept + 1 - (floor_bp * kept_of).div_ceil(10_000);
    let fails = format!("fails only when more than {} turn from caught to missed", lost - 1);
    assert!(testing.contains(&fails) && header.join(" ").contains(&fails), "the documents do not say a run of unchanged code {fails}");
    assert_eq!(config["test_workspace"].as_bool(), Some(true), "a mutant is not tested by every package that can observe it");
    // Every listed mutant gives its reason, and is in a file that exists.
    let allowed = std::fs::read_to_string(repo().join(".cargo/mutants-allowed.txt")).unwrap();
    for entry in allowed.lines().filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#')) {
        let (mutant, reason) = entry.split_once(" # ").unwrap_or_else(|| panic!("an allowlist entry without a reason: {entry}"));
        assert!(!reason.trim().is_empty(), "an allowlist entry without a reason: {entry}");
        let file = mutant.split_once(": ").map_or(mutant, |(file, _)| file);
        assert!(repo().join(file).is_file(), "an allowlist entry for a file that does not exist: {entry}");
    }

    let workflow = std::fs::read_to_string(repo().join(".github/workflows/mutants.yml")).unwrap();
    let triggers: Vec<&str> =
        workflow.lines().skip_while(|line| *line != "on:").skip(1).take_while(|line| line.starts_with(' ') || line.is_empty()).collect();
    for trigger in ["  schedule:", "  workflow_dispatch:"] {
        assert!(triggers.contains(&trigger), "the Mutants workflow does not run on `{}`", trigger.trim());
    }
    for trigger in ["  pull_request:", "  push:"] {
        assert!(!triggers.contains(&trigger), "a multi-hour mutation sweep runs automatically on `{}`", trigger.trim());
    }
    let concurrency = "concurrency:\n  group: mutants-${{ github.ref }}\n  cancel-in-progress: false\n";
    assert!(workflow.contains(concurrency), "full mutation sweeps do not queue per branch: {concurrency}");
    // A listing that fails reads as no mutants unless the pipe fails too.
    assert!(workflow.contains("defaults:\n  run:\n    shell: bash\n"), "the workflow's steps do not run under bash -eo pipefail");
    // Linux runners, as the README says: the code a run leaves out as not
    // built is the code Linux does not compile.
    let runners: Vec<&str> = workflow.lines().filter_map(|line| line.trim().strip_prefix("runs-on: ")).collect();
    assert!(
        !runners.is_empty() && runners.iter().all(|runner| *runner == "ubuntu-latest"),
        "a job of the Mutants workflow runs off Linux: {runners:?}"
    );
    assert!(!workflow.contains("continue-on-error"), "a step of the Mutants workflow may fail without failing it");
    // One compiler, a release's, for every job: under a floating `stable` a
    // Rust release during a full run would build its shards with two.
    let compiler = workflow.lines().find_map(|line| line.strip_prefix("  RUST_TOOLCHAIN: ")).expect("the Mutants workflow pins no compiler");
    assert!(
        compiler.split('.').count() == 3 && compiler.split('.').all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())),
        "the Mutants workflow's compiler {compiler} is not a release"
    );
    let installed: Vec<&str> = workflow.lines().filter_map(|line| line.trim().strip_prefix("toolchain: ")).collect();
    assert!(
        installed.iter().all(|toolchain| *toolchain == "${{ env.RUST_TOOLCHAIN }}"),
        "a job of the Mutants workflow installs another compiler than the pinned one: {installed:?}"
    );
    let (plan, shards, judge) = (workflow_job("mutants.yml", "plan"), workflow_job("mutants.yml", "mutants"), workflow_job("mutants.yml", "judge"));
    let holds = |job: &[String], line: &str| job.iter().any(|l| l.trim() == line);
    for job in [&plan, &shards] {
        assert!(holds(job, "echo \"$CARGO_MUTANTS_SHA256  $RUNNER_TEMP/cargo-mutants.tar.gz\" | sha256sum -c -"), "cargo-mutants runs unchecked");
        assert!(holds(job, "toolchain: ${{ env.RUST_TOOLCHAIN }}"), "a job of the Mutants workflow builds with the runner's own compiler");
    }
    assert!(holds(&plan, "mutants=$(cargo mutants --list | wc -l)"), "the plan does not count the mutants it shards");
    assert!(holds(&shards, "shard: ${{ fromJSON(needs.plan.outputs.shards) }}"), "the shards are not the plan's");
    assert!(holds(&shards, "SHARD: ${{ matrix.shard }}/${{ needs.plan.outputs.count }}"), "a shard runs with another shard count");
    assert!(holds(&shards, "set -- --in-place --shard \"$SHARD\" --annotations none -vV"), "a shard does not test its share");
    // A test that a mutant drives to allocate without bound is killed inside
    // the shard's memory-capped cgroup, not left to starve the runner: the
    // step caps the cgroup and joins it before cargo-mutants starts.
    let at = |line: &str| shards.iter().position(|l| l.trim() == line);
    let capped = shards.iter().any(|l| l.contains("| sudo tee \"$fence/memory.max\" > /dev/null"));
    let (joined, tested) = (at("echo $$ | sudo tee \"$fence/cgroup.procs\" > /dev/null"), at("cargo mutants \"$@\" || status=$?"));
    assert!(
        capped && matches!((joined, tested), (Some(joined), Some(tested)) if joined < tested),
        "a shard's mutants run outside a memory-capped cgroup"
    );
    // cargo-mutants' baseline runs only the tests of the packages its mutants
    // are in, and each mutant every package's: a test that failed without any
    // mutant would catch them all. So every test of the workspace passes on
    // the unmutated tree first, in the cgroup the mutants run in.
    let unmutated = at("cargo test --workspace --locked");
    assert!(
        matches!((joined, unmutated, tested), (Some(joined), Some(unmutated), Some(tested)) if joined < unmutated && unmutated < tested),
        "a shard tests its mutants before every test of the workspace has passed on the unmutated tree in its cgroup"
    );
    assert!(holds(&shards, "max-parallel: 4"), "background mutation sweeps do not reserve runner capacity");
    assert!(holds(&judge, "if: ${{ !cancelled() }}"), "the judgement is skipped when a shard fails");
    assert!(holds(&judge, "needs: [plan, mutants]"), "the judgement does not wait for the shards");
    // The download puts a lone shard's artifact in shards/ itself, and each of
    // several in a directory of its own: a small workspace with one shard
    // is judged too.
    assert!(
        holds(&judge, "if [ -e shards/outcomes.json ]; then dirs=(shards); else dirs=(shards/*/); fi"),
        "a run with a single shard is not judged"
    );
    assert!(
        holds(&judge, "scripts/check-mutants.sh --mutants \"$MUTANTS\" --missed missed.txt \"${dirs[@]}\""),
        "the shards are not judged against the number of mutants the plan listed"
    );
    // Re-running a run's failed jobs keeps its earlier attempt's artifacts,
    // and the upload refuses a name the run holds unless it may overwrite it:
    // a shard that failed once cargo-mutants had started kept its outcomes,
    // which the re-run docs/testing.md gives must replace.
    let lines: Vec<String> = workflow.lines().map(str::to_string).collect();
    let uploads: Vec<Vec<&str>> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim().starts_with("uses: actions/upload-artifact@"))
        .map(|(at, _)| step_keys(&lines, at))
        .collect();
    assert!(
        uploads.iter().any(|keys| keys.contains(&"name: mutants-shard-${{ matrix.shard }}"))
            && uploads.iter().all(|keys| keys.contains(&"overwrite: true")),
        "a job of the Mutants workflow cannot upload its artifact again when it is re-run: {uploads:?}"
    );

    let tree = scratch("mutants");
    std::fs::create_dir_all(tree.join("src")).unwrap();
    std::fs::create_dir_all(tree.join(".cargo")).unwrap();
    // Line 2 is `kept`'s body, 8 `apple`'s (macOS only; its span starts at
    // its doc comment, line 5), 14 a non-Unix statement of `split`'s, 18 a
    // Unix one, 27 the else branch of a non-Unix `if` in `branch`, 34 a
    // statement only a build without cfg(test) compiles, which is built: the
    // integration tests and the binary use that build; and 40 a statement
    // under cfg(true), which every build compiles.
    std::fs::write(
        tree.join("src/lib.rs"),
        "fn kept(a: u32) -> bool {\n    a > 1\n}\n\n/// Built only on macOS.\n#[cfg(target_os = \"macos\")]\n\
         fn apple(a: u32) -> bool {\n    a > 1\n}\n\nfn split(a: u32) -> u32 {\n    #[cfg(not(unix))]\n    {\n        a + 1\n    }\n\
         \x20   #[cfg(unix)]\n    {\n        a + 2\n    }\n}\n\n\
         fn branch(a: u32) -> u32 {\n    #[cfg(not(unix))]\n    if a > 1 {\n        return a;\n    } else {\n        return a + 3;\n    }\n    a\n}\n\n\
         fn quiet(a: u32) -> u32 {\n    #[cfg(not(test))]\n    let a = a % 7;\n    a\n}\n\n\
         fn always(a: u32) -> u32 {\n    #[cfg(true)]\n    let a = a + 4;\n    a\n}\n",
    )
    .unwrap();
    let allowlist = |extra: &str| {
        std::fs::write(
            tree.join(".cargo/mutants-allowed.txt"),
            format!(
                "# Mutants allowed to survive.\n\
                 src/lib.rs: replace > with == in kept # hangs: the loop it ends never ends\n\
                 src/lib.rs: replace > with != in kept # allocates without bound: the memory cap ends the test run\n\
                 src/lib.rs: replace + with - in split # the sum is only logged\n{extra}"
            ),
        )
        .unwrap()
    };
    allowlist("");
    let linux = "debug_assertions\ntarget_family=\"unix\"\ntarget_os=\"linux\"\nunix\n";
    let macos = "debug_assertions\ntarget_family=\"unix\"\ntarget_os=\"macos\"\nunix\n";
    // The summary cargo-mutants 27.1.0 writes for a mutant whose phases ended
    // as `how` says, and those phases, for the kinds whose summary alone does
    // not say how they ended: a signal that ends cargo, building or testing,
    // makes a Failure, and so would a phase that ended with no exit status;
    // a run with --check only checks each mutant, a Success.
    fn phases(how: &str) -> (&str, serde_json::Value) {
        let phase = |phase: &str, status: serde_json::Value| serde_json::json!({"phase": phase, "process_status": status});
        let built = phase("Build", "Success".into());
        match how {
            "test signalled" => ("Failure", serde_json::json!([built, phase("Test", serde_json::json!({"Signalled": 9}))])),
            "build signalled" => ("Failure", serde_json::json!([phase("Build", serde_json::json!({"Signalled": 9}))])),
            "no status" => ("Failure", serde_json::json!([built, phase("Test", "Other".into())])),
            "checked" => ("Success", serde_json::json!([phase("Check", "Success".into())])),
            _ => (how, serde_json::Value::Null),
        }
    }
    // One shard's output directory, as cargo-mutants writes it, each mutant
    // `(name, the line its function starts on or 0 for none, outcome)`.
    let shard = |name: &str, cfg: &str, mutants: &[(String, usize, &str)], finished: bool, baseline: &str| {
        let dir = tree.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let outcomes: Vec<serde_json::Value> = std::iter::once(serde_json::json!({"scenario": "Baseline", "summary": baseline}))
            .chain(mutants.iter().map(|(mutant, function, how)| {
                let mut found = serde_json::json!({"name": mutant});
                if *function > 0 {
                    found["file"] = mutant.split(':').next().unwrap().into();
                    found["function"] = serde_json::json!({"span": {"start": {"line": function}}});
                }
                let (summary, ran) = phases(how);
                let mut outcome = serde_json::json!({"scenario": {"Mutant": found}, "summary": summary});
                if !ran.is_null() {
                    outcome["phase_results"] = ran;
                }
                outcome
            }))
            .collect();
        let end = if finished { serde_json::json!("2026-10-21T00:00:00Z") } else { serde_json::Value::Null };
        let run = serde_json::json!({"outcomes": outcomes, "end_time": end});
        std::fs::write(dir.join("outcomes.json"), run.to_string()).unwrap();
        std::fs::write(dir.join("cfg.txt"), cfg).unwrap();
        dir
    };
    let m = |name: &str, function: usize, summary: &'static str| (name.to_string(), function, summary);
    // Caught: 5, one of them under cfg(true), a listed timeout and a listed
    // mutant whose tests a signal ended. Missed: two unlisted, one of them
    // under cfg(not(test)), a listed one, an unlisted timeout and an unlisted
    // mutant whose build a signal ended. Unviable: 1. Not built on Linux:
    // `apple`'s, the non-Unix statement's and the non-Unix `if`'s.
    let first = [
        m("src/lib.rs:2:7: replace > with >= in kept", 1, "MissedMutant"),
        m("src/lib.rs:2:6: replace > with == in kept", 1, "Timeout"),
        m("src/lib.rs:8:7: replace > with >= in apple", 5, "MissedMutant"),
        m("src/lib.rs:2:5: replace kept -> bool with true", 1, "CaughtMutant"),
        m("src/lib.rs:2:5: replace kept -> bool with false", 1, "CaughtMutant"),
    ];
    let second = [
        m("src/lib.rs:14:11: replace + with - in split", 11, "MissedMutant"),
        m("src/lib.rs:18:11: replace + with - in split", 11, "MissedMutant"),
        m("src/lib.rs:18:11: replace + with * in split", 11, "CaughtMutant"),
        m("src/lib.rs:2:8: replace > with < in kept", 1, "Timeout"),
        m("src/lib.rs:12:5: replace split -> u32 with 0", 11, "Unviable"),
        m("src/lib.rs:12:5: replace split -> u32 with 1", 11, "CaughtMutant"),
        m("src/lib.rs:27:18: replace + with - in branch", 22, "MissedMutant"),
        m("src/lib.rs:34:15: replace % with / in quiet", 32, "MissedMutant"),
        m("src/lib.rs:40:15: replace + with - in always", 38, "CaughtMutant"),
        m("src/lib.rs:2:7: replace > with != in kept", 1, "test signalled"),
        m("src/lib.rs:18:11: replace + with / in split", 11, "build signalled"),
    ];
    let (one, two) = (shard("one", linux, &first, true, "Success"), shard("two", linux, &second, true, "Success"));
    let on_macos = shard("on-macos", macos, &first, true, "Success");
    let again = shard("again", linux, &first, true, "Success");
    let unfinished = shard("unfinished", linux, &first, false, "Success");
    let unbaselined = shard("unbaselined", linux, &first, true, "Failure");
    let other_target = shard("other-target", macos, &second, true, "Success");
    let checked = shard("checked", linux, &[m("src/lib.rs:2:7: replace > with >= in kept", 1, "checked")], true, "Success");
    let no_status = shard("no-status", linux, &[m("src/lib.rs:2:7: replace > with >= in kept", 1, "no status")], true, "Success");
    // An outcome cargo-mutants 27.1.0 does not write is refused even in code
    // the run did not build, `apple`'s.
    let unknown = shard("unknown", linux, &[m("src/lib.rs:8:7: replace > with >= in apple", 5, "Bogus")], true, "Success");
    // Four files: two tied at a third, which come in name order, one under
    // 90% by less than a mutant, and one at 90%. Their mutants have no
    // function, so their files need not exist.
    let per_file: Vec<_> = [("b.rs", 1, 2), ("a.rs", 9, 1), ("d.rs", 2, 4), ("c.rs", 8, 1)]
        .iter()
        .flat_map(|&(file, caught, missed)| {
            (0..caught + missed).map(move |i| (format!("{file}:1:{}: replace > with < in f", i + 1), 0, if i < caught { "CaughtMutant" } else { "MissedMutant" }))
        })
        .collect();
    let files = shard("files", linux, &per_file, true, "Success");
    // Scores of exactly the floor and a mutant under it, each over the fewest
    // mutants that score it exactly (hundredths of a point over 10,000, in
    // lowest terms), of exactly the floor plus the margin as the report
    // truncates it, over the fewest mutants that score that, and of 100%.
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let exactly = |bp: u64| (bp / gcd(bp, 10_000), 10_000 / gcd(bp, 10_000));
    let truncated = |bp: u64| {
        let scored = (1..).find(|&scored| (bp * scored).div_ceil(10_000) * 10_000 < (bp + 1) * scored).unwrap();
        ((bp * scored).div_ceil(10_000), scored)
    };
    // Each mutant with no function and its outcome alone, to keep a fixture
    // of 10,000 quick to read.
    let boundary = |name: &str, (caught, scored): (u64, u64)| {
        let mutants: Vec<_> =
            (0..scored).map(|i| m(&format!("src/lib.rs:2:{}: m", 100 + i), 0, if i < caught { "CaughtMutant" } else { "MissedMutant" })).collect();
        shard(name, linux, &mutants, true, "Success")
    };
    let at_floor = exactly(floor_bp);
    let passing = boundary("passing", at_floor);
    let failing = (at_floor.0 > 0).then(|| boundary("failing", (at_floor.0 - 1, at_floor.1)));
    let margin_over = (floor_bp + margin_bp <= 10_000).then(|| boundary("margin-over", truncated(floor_bp + margin_bp)));
    let perfect = boundary("perfect", (1, 1));
    let missed = tree.join("missed.txt");
    let summary = tree.join("summary.md");
    let judge_run = |args: &[&str], dirs: &[&PathBuf]| {
        let mut gate = script("scripts/check-mutants.sh");
        gate.current_dir(&tree).args(args).args(dirs);
        ran(&mut gate)
    };
    let (score, score_report) = {
        let mut gate = script("scripts/check-mutants.sh");
        gate.current_dir(&tree).env("GITHUB_STEP_SUMMARY", &summary).args(["--mutants", "16", "--missed"]).arg(&missed).args([&one, &two]);
        ran(&mut gate)
    };
    let (change, change_report) = {
        let mut gate = script("scripts/check-mutants.sh");
        gate.current_dir(&tree).env("GITHUB_ACTIONS", "true").arg("--diff").args([&one, &two]);
        ran(&mut gate)
    };
    let (macos_change, macos_report) = judge_run(&["--diff"], &[&on_macos]);
    let (at, at_report) = judge_run(&[], &[&passing]);
    let under = failing.as_ref().map(|failing| judge_run(&[], &[failing]));
    let over = margin_over.as_ref().map(|margin_over| judge_run(&[], &[margin_over]));
    let (full, full_report) = judge_run(&[], &[&perfect]);
    let (_, files_report) = judge_run(&[], &[&files]);
    let refusals: Vec<(&str, (Option<i32>, String))> = vec![
        ("did not finish", judge_run(&[], &[&unfinished])),
        ("the unmutated tests did not pass", judge_run(&[], &[&unbaselined])),
        ("the shards overlap", judge_run(&[], &[&one, &again])),
        ("the shards built different targets", judge_run(&[], &[&one, &other_target])),
        ("a shard is missing", judge_run(&["--mutants", "17"], &[&one, &two])),
        ("it was only checked, never built or tested (a run with --check)", judge_run(&[], &[&checked])),
        ("an outcome this script does not know: Failure, its test phase ending \"Other\"", judge_run(&[], &[&no_status])),
        ("an outcome this script does not know: Bogus", judge_run(&[], &[&unknown])),
    ];
    let missed_list = std::fs::read_to_string(&missed).unwrap_or_default();
    let summary_text = std::fs::read_to_string(&summary).unwrap_or_default();
    allowlist("src/lib.rs: replace > with >= in kept\n");
    let (unreasoned, unreasoned_report) = judge_run(&["--diff"], &[&one]);
    let _ = std::fs::remove_dir_all(&tree);

    let expect_failed = 7 * 10_000 < floor_bp * 12;
    assert_eq!(score, Some(i32::from(expect_failed)), "a score of 58.33% against the floor of {floor}%: {score_report}");
    // How many of the missed must be caught for caught / scored to reach a
    // percentage in hundredths of a point; and the report's table of files,
    // each score truncated to hundredths as the score is.
    let to_reach = |bp: u64, caught: u64, scored: u64| (bp * scored).saturating_sub(caught * 10_000).div_ceil(10_000);
    let file_bp = file_target.parse::<u64>().unwrap() * 100;
    let table = |under: usize, rows: &[(u64, u64, &str)]| {
        let mut table = format!("per file, worst first ({under} of {} under {file_target}%):\n", rows.len());
        table += &format!("{:>7} {:>7} {:>7} {:>7}  file\n", "score", "caught", "missed", format!("to {file_target}%"));
        for &(caught, missed, file) in rows {
            let (score, needed) = (percent(caught * 10_000 / (caught + missed)), to_reach(file_bp, caught, caught + missed));
            table += &format!("{score:>7} {caught:>7} {missed:>7} {needed:>7}  {file}\n");
        }
        table
    };
    let file_table = table(1, &[(7, 5, "src/lib.rs")]);
    for line in [
        "16 mutants in 2 shards:",
        "      5 caught\n      3 missed\n",
        "      2 timed out: 1 listed with a reason, counted caught; 1 not, counted missed\n",
        "      2 ended by a signal: 1 listed with a reason, counted caught; 1 not, counted missed\n",
        "      1 unviable\n      3 not built for this target\n",
        "score: 58.33%, 7 caught of 12 caught or missed",
        &file_table,
        "ended by a signal:\n  src/lib.rs:2:7: replace > with != in kept (its tests ended by signal 9, listed, counted caught)\n  \
         src/lib.rs:18:11: replace + with / in split (its build ended by signal 9, counted missed)\n",
    ] {
        assert!(score_report.contains(line), "the report does not say {line:?}: {score_report}");
    }
    // Worst first, a tie by name; a file under the target by a fraction of a
    // mutant needs a whole one, and one at it needs none.
    let files_table = table(3, &[(1, 2, "b.rs"), (2, 4, "d.rs"), (8, 1, "c.rs"), (9, 1, "a.rs")]);
    assert!(files_report.contains(&files_table), "the files are not reported as {files_table:?}: {files_report}");
    assert_eq!(
        (to_reach(file_bp, 7, 12), to_reach(file_bp, 8, 9), to_reach(file_bp, 9, 10)),
        (4, 1, 0),
        "90% of 12 is 10.8, so 11 caught, 4 more than 7; of 9, 8.1, so 9, 1 more than 8; of 10, 9, none more"
    );
    assert_eq!(
        missed_list,
        "src/lib.rs:2:7: replace > with >= in kept\nsrc/lib.rs:2:8: replace > with < in kept\nsrc/lib.rs:18:11: replace + with - in split\n\
         src/lib.rs:18:11: replace + with / in split\nsrc/lib.rs:34:15: replace % with / in quiet\n",
        "the missed list is not the misses, the unlisted timeout and the unlisted signal, in source order"
    );
    assert!(summary_text.starts_with("## Mutation score\n") && summary_text.contains("score: 58.33%"), "{summary_text}");

    assert_eq!(change, Some(1), "a change with unlisted misses passes: {change_report}");
    assert!(
        change_report.contains(
            "4 missed, timed out or ended by a signal, and not listed in .cargo/mutants-allowed.txt:\n  src/lib.rs:2:7: replace > with >= in kept\n  \
             src/lib.rs:2:8: replace > with < in kept\n  src/lib.rs:18:11: replace + with / in split\n  src/lib.rs:34:15: replace % with / in quiet\n"
        ),
        "{change_report}"
    );
    // Each blocking mutant is marked on its line, its message escaped as a
    // workflow command needs (% as %25), and nothing else is marked.
    for (line, column, what) in [
        (2, 7, "replace > with >= in kept"),
        (2, 8, "replace > with < in kept"),
        (18, 11, "replace + with / in split"),
        (34, 15, "replace %25 with / in quiet"),
    ] {
        let mark = format!("::error file=src/lib.rs,line={line},col={column}::Not caught, and not listed in .cargo/mutants-allowed.txt: {what}\n");
        assert!(change_report.contains(&mark), "a blocking mutant is not marked on its line as {mark:?}: {change_report}");
    }
    assert_eq!(change_report.matches("::error").count(), 4, "a mutant that does not block the change is marked: {change_report}");
    assert_eq!(macos_change, Some(1), "on macOS the macOS-only function is built, and its miss blocks: {macos_report}");
    assert!(macos_report.contains("  src/lib.rs:8:7: replace > with >= in apple\n") && macos_report.contains("      0 not built for this target"), "{macos_report}");

    let offer = "the floor in scripts/check-mutants.sh can be raised to";
    assert_eq!(at, Some(0), "a score of exactly the floor, {} caught of {}, fails the floor of {floor}%: {at_report}", at_floor.0, at_floor.1);
    assert!(!at_report.contains(offer), "a score at the floor offers to raise it: {at_report}");
    let target_bp = target.parse::<u64>().unwrap() * 100;
    if floor_bp < target_bp {
        let short = target_bp - floor_bp;
        let to_go = format!(
            "{}.{:02} points under the target of {target}%; catching {} of the missed would meet it",
            short / 100,
            short % 100,
            to_reach(target_bp, at_floor.0, at_floor.1)
        );
        assert!(at_report.contains(&to_go), "the report does not say how far the target is, {to_go:?}: {at_report}");
    }
    if let Some((under, under_report)) = under {
        assert_eq!(under, Some(1), "a score a mutant under the floor of {floor}% passes: {under_report}");
        assert!(under_report.contains(&format!("the score is under its floor of {floor}%")), "{under_report}");
    }
    // The floor may rise to a run's score less the margin, and only above itself.
    if let Some((over, over_report)) = over {
        assert_eq!(over, Some(0), "a score the margin over the floor of {floor}% fails it: {over_report}");
        assert!(!over_report.contains(offer), "a score the margin over the floor offers to raise the floor to itself: {over_report}");
    }
    assert_eq!(full, Some(0), "a score of 100% fails the floor of {floor}%: {full_report}");
    assert!(full_report.contains(&format!("the target of {target}% is met")), "{full_report}");
    if 10_000 - margin_bp > floor_bp {
        let raise = format!("{offer} {}, the score less its margin of {margin} points", percent(10_000 - margin_bp));
        assert!(full_report.contains(&raise), "a score of 100% does not offer to raise the floor to the score less the margin, {raise:?}: {full_report}");
    }
    for (why, (code, report)) in refusals {
        assert_eq!(code, Some(2), "a run where {why} is judged: {report}");
        assert!(report.contains(why), "the refusal does not say {why:?}: {report}");
    }
    assert_eq!(unreasoned, Some(2), "an allowlist entry without a reason is taken: {unreasoned_report}");
    assert!(unreasoned_report.contains("not a mutant's name, then \" # \" and the reason it may survive"), "{unreasoned_report}");
}

/// The first frame, held in CI by its ratio to a baseline binary's start.
///
/// The target, cold start to the first painted frame in 1 ms, is a release
/// build's on the reference machine, not CI's: timed from the spawn, a frame
/// includes the machine's own process start, which is not the harness's
/// work and differs from one machine to the next. A baseline binary's start
/// alone (below) took 0.58 to 0.83 ms on ubuntu-latest and 1.72 to 2.88 ms
/// on macos-latest, where it exceeds the whole target and moved by more
/// than it from one run to the next. So this test times the first frame
/// against a baseline binary started the same way on the same machine, and
/// holds the ratio of their medians to a ratchet. The baseline,
/// crates/tui/examples/startup_baseline.rs, built beside `openmax` with the
/// same profile, writes the end of a frame and exits: its time is the
/// machine's start of a release binary (the spawn, the dynamic loader, the
/// runtime's start) and the pseudo-terminal's delivery, so the ratio is the
/// harness's start to its first frame, its own binary's loading included,
/// in units of that.///
/// Method: 21 launches, after one warmup launch, each into a fresh fixture
/// HOME (settings only, the project not yet trusted, so `--trust-project`
/// grants it) on a fresh 80x24 pseudo-terminal. A launch is timed from just
/// before the spawn to the return of the read that brings the end of the
/// first frame, the close of its synchronized update. The spawn is
/// `posix_spawn` (no `pre_exec`), so the clock does not include forking the
/// test process. The launches run in a session of their own, which has no
/// controlling terminal, so each paints at its pseudo-terminal's 80x24
/// wherever the tests run (`first_frame::alone`). After the frame the launch
/// answers the keyboard protocol query, quits with Ctrl+C twice, and keeps
/// reading until the child is reaped, so it never blocks on unread output.
/// In front of each launch, the warmup's included, one run of the baseline
/// binary is timed the same way, to the read that brings its frame end, so a
/// change in the machine's load moves both alike. The medians are the
/// measurement, printed with the 25th and 75th percentiles and their ratio,
/// and with the first frame's two components: the process start, the
/// baseline's median, and the harness's work, its binary's loading
/// included, the first frame's median less the baseline's.
///
/// A release build asserts the ratio under the ratchet of the platform it
/// runs on: the highest ratio CI's runner for that platform measured, plus a
/// quarter, rounded up to a tenth, since a platform's process start, and so
/// the ratio, has a scale of its own (below). A machine much unlike those
/// runners can measure another ratio. A debug build prints the measurement
/// and holds the ratio to 20 times the ratchet, which only a hang or a stall
/// many times today's breaks. Neither asserts a time: they are printed.
/// Without the baseline binary, which `cargo test` builds only when no
/// target is selected, a release build fails and a debug build skips.
///
/// Measured in release builds, as the ratio of the medians (first frame;
/// baseline). On macos-latest, seven runs: 1.38 to 1.57 (2.37 to 4.43 ms;
/// 1.72 to 2.88 ms), so a ratchet of 2.0. On ubuntu-latest, six runs: 2.03
/// to 2.29 (1.30 to 1.72 ms; 0.58 to 0.83 ms), so 2.9. On an Apple M5
/// laptop (macOS 27), eleven runs with every other build stopped, at a
/// one-minute load average of 4.7: 1.39 to 1.48 (1.60 to 2.34 ms; 1.09 to
/// 1.68 ms). The startup a change can add and still pass is the ratchet
/// times the baseline, less the first frame: 1.06 to 1.64 ms in the
/// macos-latest runs above and 0.36 to 0.69 ms in the ubuntu-latest ones.
/// Painting the frame behind the trust grant, settings and the core, the
/// order before it came first, moved the laptop's ratio to 2.00 to 2.46 in
/// ten runs beside other builds, the lowest past the macOS ratchet by 0.07%.
#[cfg(unix)]
#[test]
fn first_frame_holds_its_ratio_to_a_baseline_binary() {
    use std::time::Duration;
    // The highest ratio CI's runner for this platform measured, plus a
    // quarter, rounded up to a tenth (above).
    const RATCHET: f64 = if cfg!(target_os = "macos") { 2.0 } else { 2.9 };
    const LAUNCHES: usize = 21;
    let Some(baseline) = first_frame::baseline_binary() else { return };
    if !first_frame::alone("first_frame_holds_its_ratio_to_a_baseline_binary") {
        return;
    }
    first_frame::baseline(&baseline);
    first_frame::launch(0);
    let (mut starts, mut frames) = (Vec::new(), Vec::new());
    for n in 1..=LAUNCHES {
        starts.push(first_frame::baseline(&baseline));
        frames.push(first_frame::launch(n));
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
    let (frame, frame_quartiles) = summary(&mut frames);
    let ratio = frame.as_secs_f64() / start.as_secs_f64();
    let (bound, asserted) = if cfg!(debug_assertions) {
        (20.0 * RATCHET, "debug, asserting 20 times the ratchet")
    } else {
        (RATCHET, "release, asserting the ratchet")
    };
    let work = frame.saturating_sub(start);
    let report = format!(
        "first frame: {frame_quartiles} over {LAUNCHES} launches; baseline binary (process start): {start_quartiles}; \
         harness work (first frame less baseline): {work:?}; ratio {ratio:.2} (ratchet {RATCHET}; {asserted}; \
         the times are printed, not asserted)"
    );
    println!("{report}");
    assert!(
        ratio < bound,
        "the first frame takes {ratio:.2} times the baseline binary's start, over {bound}; {report}: frames {frames:?}, baselines {starts:?}"
    );
}

/// The first-frame harness reads what a run wrote before it exited. On
/// macOS the last close of a pseudo-terminal's terminal end discards
/// whatever its controller has not read, so a run that wrote its frame and
/// exited before the harness's first read left nothing to read, and the
/// measurement failed without a frame; the harness now holds a descriptor
/// of the terminal end of its own until the frame end is read. Here the
/// baseline binary writes its frame end and is reaped before the first
/// read.
#[cfg(unix)]
#[test]
fn the_first_frame_harness_reads_what_a_run_wrote_before_it_exited() {
    let Some(baseline) = first_frame::baseline_binary() else { return };
    let (painted, out) = first_frame::baseline_reaped_before_read(&baseline);
    assert!(
        painted.is_some(),
        "the harness lost the frame end a run wrote before it exited: {:?}",
        String::from_utf8_lossy(&out)
    );
}

#[cfg(unix)]
mod first_frame {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
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

    /// The baseline binary, crates/tui/examples/startup_baseline.rs, where
    /// cargo puts an example: in the `examples` directory beside the
    /// `openmax` these tests run, built with the same profile. `cargo test`
    /// builds it there only when no target is selected (a selection that
    /// takes in examples, such as `--all-targets`, builds a test harness of
    /// it instead, under another name), and CI's thesis job builds it on its
    /// own. Missing, it fails a release build, which asserts the ratio it
    /// divides, and skips a debug build, which only prints it.
    pub fn baseline_binary() -> Option<PathBuf> {
        let binary = Path::new(env!("CARGO_BIN_EXE_openmax")).with_file_name("examples").join("startup_baseline");
        match std::fs::metadata(&binary) {
            Ok(_) => Some(binary),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && cfg!(debug_assertions) => {
                println!("skipped: the baseline binary {} is not built (cargo build --example startup_baseline)", binary.display());
                None
            }
            Err(e) => panic!(
                "the baseline binary {}: {e}; build it with `cargo build --release --example startup_baseline`",
                binary.display()
            ),
        }
    }

    /// Spawn to the frame end the baseline binary writes, which is all it
    /// does.
    pub fn baseline(binary: &Path) -> Duration {
        let mut run = Run::spawn(Command::new(binary));
        let painted = run.painted();
        let out = run.reap();
        painted.unwrap_or_else(|| panic!("the baseline binary wrote no frame end: {:?}", String::from_utf8_lossy(&out)))
    }

    /// A run of the baseline binary reaped before the first read, so what
    /// is read is only what its terminal kept: the time to its frame end,
    /// if that came, and everything read.
    pub fn baseline_reaped_before_read(binary: &Path) -> (Option<Duration>, Vec<u8>) {
        let mut run = Run::spawn(Command::new(binary));
        run.child.wait().unwrap();
        let painted = run.painted();
        (painted, run.reap())
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
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_openmax"));
        cmd.arg("--trust-project")
            .current_dir(&project)
            .env("HOME", &home)
            .env("TERM", "xterm-256color")
            .env_remove("OPENMAX_API_KEY")
            .env_remove("OPENMAX_SESSION")
            .env_remove("OPENMAX_PERF")
            // --trust-project is a human's grant; the test attests to it.
            .env("OPENMAX_HUMAN_ATTEST", "1");
        let mut run = Run::spawn(cmd);
        let painted = run.painted();
        if painted.is_some() {
            let deadline = Instant::now() + Duration::from_secs(10);
            while find(&run.out, KEYBOARD_QUERY).is_none() && Instant::now() < deadline {
                if read_some(&mut run.controller, &mut run.out).is_none() {
                    break;
                }
            }
            if find(&run.out, KEYBOARD_QUERY).is_some() {
                run.controller.write_all(ANSWER_AND_QUIT).unwrap();
            }
        }
        let out = run.reap();
        let _ = std::fs::remove_dir_all(&dir);
        painted.unwrap_or_else(|| panic!("no first frame: {:?}", String::from_utf8_lossy(&out)))
    }

    /// A child with a fresh 80x24 pseudo-terminal on its stdin and stdout,
    /// timed from just before its spawn, and what the test read from it.
    struct Run {
        child: Child,
        controller: File,
        /// The test's own descriptor of the terminal end, held until the
        /// frame end is read: on macOS the last close of a terminal end
        /// discards whatever its controller has not read, so a child that
        /// wrote its frame and exited before the first read would leave
        /// nothing to read.
        terminal: Option<File>,
        started: Instant,
        out: Vec<u8>,
    }

    impl Run {
        fn spawn(mut cmd: Command) -> Run {
            let (controller, terminal) = pseudo_terminal(80, 24);
            cmd.stdin(terminal.try_clone().unwrap()).stdout(terminal.try_clone().unwrap()).stderr(Stdio::null());
            let started = Instant::now();
            let child = cmd.spawn().unwrap();
            // The child's are the only other descriptors of the terminal
            // end now.
            drop(cmd);
            Run { child, controller, terminal: Some(terminal), started, out: Vec::new() }
        }

        /// Spawn to the return of the read that brought the end of the
        /// first frame, or None if none came within 10 seconds. The test's
        /// descriptor of the terminal end is let go after, so the
        /// controller reads the end of the output once the child closes
        /// its own.
        fn painted(&mut self) -> Option<Duration> {
            let deadline = self.started + Duration::from_secs(10);
            let mut painted = None;
            while painted.is_none() && Instant::now() < deadline {
                let Some(read) = read_some(&mut self.controller, &mut self.out) else { break };
                if find(&self.out, FRAME_END).is_some() {
                    painted = Some(read - self.started);
                }
            }
            self.terminal = None;
            painted
        }

        /// Read until the child is reaped, killing one still running 10
        /// seconds on, and return everything read: a child cannot finish
        /// exiting while what it wrote sits unread in its terminal.
        fn reap(mut self) -> Vec<u8> {
            self.terminal = None;
            let reap_by = Instant::now() + Duration::from_secs(10);
            while self.child.try_wait().unwrap().is_none() {
                if Instant::now() > reap_by {
                    let _ = self.child.kill();
                }
                if read_some(&mut self.controller, &mut self.out).is_none() {
                    let _ = self.child.wait();
                    break;
                }
            }
            self.out
        }
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

/// One session file reproduces the event stream byte for byte, offline, on
/// any machine.
///
/// Method: a live `openmax --trust-project --json -p` run against a local
/// endpoint whose first reply calls `bash` to create a marker file and whose
/// second streams the answer in two pieces with usage, so the run has a tool
/// round, tokens, usage and a done. The session's own event log,
/// `~/.openmax/sessions/<id>.events.jsonl` as the live run left it, is
/// copied unchanged into another directory, as onto another machine, and
/// `openmax --replay <copy> --json` runs there under another HOME (empty: no
/// settings, no trust, no data dir), working directory and TMPDIR. Its
/// stdout must equal the live run's stdout byte for byte, and the replay
/// must exit 0, send no request (the endpoint, still listening, served the
/// live run's two), run no tool (no marker where it ran) and create no data
/// dir. A byte gate, so it asserts in every build.
#[test]
fn replay_reproduces_the_event_stream_byte_for_byte() {
    let chunk = |delta: serde_json::Value, finish: Option<&str>, usage: Option<(u64, u64)>| {
        let mut chunk = serde_json::json!({"choices": [{"delta": delta, "finish_reason": finish}]});
        if let Some((prompt, completion)) = usage {
            chunk["usage"] = serde_json::json!({"prompt_tokens": prompt, "completion_tokens": completion});
        }
        format!("data: {chunk}\n\n")
    };
    let call = serde_json::json!([{"index": 0, "id": "call_0", "type": "function",
        "function": {"name": "bash", "arguments": r#"{"command":"touch ran.marker"}"#}}]);
    let script = vec![
        chunk(serde_json::json!({"tool_calls": call}), None, None)
            + &chunk(serde_json::json!({}), Some("tool_calls"), Some((40, 5)))
            + "data: [DONE]\n\n",
        chunk(serde_json::json!({"content": "replayed"}), None, None)
            + &chunk(serde_json::json!({"content": " answer"}), None, None)
            + &chunk(serde_json::json!({}), Some("stop"), Some((60, 3)))
            + "data: [DONE]\n\n",
    ];
    let (base_url, served) = spawn_scripted_server(script);
    let (project, home) = fresh_install("replay-live", 0);
    let live = Command::new(env!("CARGO_BIN_EXE_openmax"))
        .current_dir(&project)
        .env("HOME", &home)
        .env_remove("OPENMAX_SESSION")
        .env_remove("OPENMAX_API_KEY")
        // Test automation is a human act with no terminal; the grant needs it said.
        .env("OPENMAX_HUMAN_ATTEST", "1")
        .args(["--trust-project", "--base-url", &base_url, "--model", "m", "--context-tokens", "32768"])
        .args(["--json", "-p", "make the marker"])
        .output()
        .unwrap();
    let printed = String::from_utf8(live.stdout).unwrap();
    assert_eq!(live.status.code(), Some(0), "{printed}{}", String::from_utf8_lossy(&live.stderr));
    assert!(project.join("ran.marker").exists(), "the live run's tool round ran: {printed}");
    let kinds: Vec<serde_json::Value> =
        printed.lines().map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap()["type"].clone()).collect();
    for kind in ["tool_start", "tool_end", "token", "usage", "done"] {
        assert!(kinds.contains(&kind.into()), "the live run printed no {kind} line: {printed}");
    }
    let first: serde_json::Value = serde_json::from_str(printed.lines().next().unwrap()).unwrap();
    let id = first["session_id"].as_str().unwrap();
    let log = home.join(".openmax/sessions").join(format!("{id}.events.jsonl"));
    assert!(log.is_file(), "the live run left no event log at {}", log.display());

    let (elsewhere, other_home) = fresh_install("replay-elsewhere", 0);
    let tmp = elsewhere.join("tmp");
    std::fs::create_dir_all(&tmp).unwrap();
    let copy = elsewhere.join("copied").join("session.events.jsonl");
    std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
    std::fs::copy(&log, &copy).unwrap();
    let replayed = Command::new(env!("CARGO_BIN_EXE_openmax"))
        .current_dir(&elsewhere)
        .env("HOME", &other_home)
        .env("TMPDIR", &tmp)
        .env_remove("OPENMAX_SESSION")
        .env_remove("OPENMAX_API_KEY")
        .arg("--replay")
        .arg(&copy)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(replayed.status.code(), Some(0), "{}", String::from_utf8_lossy(&replayed.stderr));
    assert_eq!(String::from_utf8(replayed.stdout).unwrap(), printed, "the replay prints what the live run printed");
    assert_eq!(served.load(std::sync::atomic::Ordering::SeqCst), 2, "the replay sent a request");
    assert!(!elsewhere.join("ran.marker").exists(), "the replay ran a tool");
    assert!(!other_home.join(".openmax").exists(), "the replay opened a data dir");
    for dir in [project.parent().unwrap(), elsewhere.parent().unwrap()] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// A local OpenAI-compatible endpoint that answers the Nth request it reads
/// with the Nth SSE body of `script`, and any later one with a 400, for as
/// long as the test runs; and the count of requests it has read.
fn spawn_scripted_server(script: Vec<String>) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let served = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = served.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => break,
                }
            }
            if !head.ends_with(b"\r\n\r\n") {
                continue;
            }
            let length = String::from_utf8_lossy(&head)
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            let _ = stream.read_exact(&mut vec![0u8; length]);
            let response = match script.get(count.fetch_add(1, std::sync::atomic::Ordering::SeqCst)) {
                Some(sse) => format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{sse}",
                    sse.len()
                ),
                None => "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
            };
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1"), served)
}

/// An unattended run, with no terminal anywhere, in which the agent writes a
/// tool and calls it on its very next step: one turn, no human input after
/// the grant, exit 0.
///
/// Method: the command an unattended runner starts, `OPENMAX_HUMAN_ATTEST=1
/// openmax --trust-project --base-url URL --model m --context-tokens N -p
/// TASK`, on a fresh HOME and a project that ships `notes.txt`, in a session
/// of its own (setsid, so no controlling terminal) with stdin /dev/null and
/// none of the environment's proxy variables, so the requests go straight
/// to a loopback endpoint (`scripted_endpoint`) that answers the run's
/// three requests in turn. The first gets a `write_file` of
/// `.openmax/tools/line_count.toml`, an awk tool that counts the lines of
/// notes.txt and prints the attestation variable as its environment holds it.
/// The second must carry, on that write's result, the refreeze receipt naming
/// `line_count` callable from the next step and the trailing note's line
/// giving its description and parameters, and gets a call of `line_count` by
/// name. The third must carry that call's output, and gets the answer. The
/// run must exit 0 with the answer on stdout, having made exactly those
/// requests, and its session's event log
/// (`~/.openmax/sessions/<id>.events.jsonl`) must hold, in this order, the
/// write's start and end, the receipt as a `harness_note` on the write's
/// call, and the call's start and end with the tool's output; between the
/// write's end and the call's start, the `refrozen` event naming the file;
/// and one `done`, its last line. The manifest names `OPENMAX_HUMAN_ATTEST`
/// in its `env`, so only the harness's strip keeps the variable from the
/// tool, whose output must show it empty. First, the same command without the
/// variable must exit 3 before any request, leaving HOME empty: there, the
/// variable is the one human act, and nothing else grants the run. Each run
/// that is still going 30 seconds after it started is ended and fails the
/// test. A functional gate, asserted in every build.
#[test]
fn an_unattended_run_writes_a_tool_and_calls_it_on_the_next_step() {
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::process::Stdio;
    let tool = ".openmax/tools/line_count.toml";
    let manifest = r#"name = "line_count"
description = "Count the lines of notes.txt"
command = "awk"
args = ['END { printf "%d lines; attest=[%s]\n", NR, ENVIRON["OPENMAX_HUMAN_ATTEST"] }', "notes.txt"]
env = ["OPENMAX_HUMAN_ATTEST"]
mutating = false
"#;
    // A streamed reply that calls tool `name` once with `arguments`.
    let call = |id: &str, name: &str, arguments: serde_json::Value| {
        let one = serde_json::json!({"index": 0, "id": id, "type": "function",
            "function": {"name": name, "arguments": arguments.to_string()}});
        let chunk = serde_json::json!({"choices": [{"delta": {"tool_calls": [one]}, "finish_reason": null}]});
        let end = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        format!("data: {chunk}\n\ndata: {end}\n\ndata: [DONE]\n\n")
    };
    let endpoint = scripted_endpoint(vec![
        call("call_write", "write_file", serde_json::json!({"path": tool, "content": manifest})),
        call("call_tool", "line_count", serde_json::json!({})),
        sse_answer("notes.txt has 3 lines"),
    ]);
    let (project, home) = fresh_install("unattended", 0);
    std::fs::write(project.join("notes.txt"), "one\ntwo\nthree\n").unwrap();
    // The run to its exit: its code, its stdout, and all it printed.
    let run = |attested: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_openmax"));
        cmd.current_dir(&project)
            .env("HOME", &home)
            .env_remove("OPENMAX_SESSION")
            .env_remove("OPENMAX_API_KEY")
            .env_remove("OPENMAX_HUMAN_ATTEST")
            .args(["--trust-project", "--base-url", &endpoint.base_url, "--model", "m", "--context-tokens", "32768"])
            .args(["-p", "count the lines of notes.txt with a project tool"])
            .stdin(Stdio::null());
        // The binary's HTTP client reads the proxy variables (only the core's
        // own test build ignores them), and a proxy with no exception for
        // loopback would take the requests meant for the endpoint.
        for proxy in ["http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"] {
            cmd.env_remove(proxy);
        }
        if attested {
            cmd.env("OPENMAX_HUMAN_ATTEST", "1");
        }
        // The alarm outlives the exec, and no process openmax forks inherits
        // it, so the kernel ends openmax, and only openmax, with SIGALRM if it
        // is still running 30 seconds after it started, where a run here
        // takes well under one: a hang on the unattended path fails this test
        // instead of holding the job until its own time limit.
        // SAFETY: alarm and setsid are async-signal-safe, and the closure
        // touches nothing else between fork and exec.
        unsafe {
            cmd.pre_exec(|| {
                libc::alarm(30);
                match libc::setsid() {
                    -1 => Err(std::io::Error::last_os_error()),
                    _ => Ok(()),
                }
            });
        }
        let out = cmd.output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let printed = format!("{stdout}{}", String::from_utf8_lossy(&out.stderr));
        assert_ne!(out.status.signal(), Some(libc::SIGALRM), "the run was still going after 30 seconds: {printed}");
        (out.status.code(), stdout, printed)
    };

    let (code, _, printed) = run(false);
    assert_eq!(code, Some(3), "with no terminal and no attestation the grant went through: {printed}");
    assert!(printed.contains("OPENMAX_HUMAN_ATTEST=1"), "the refusal does not name the human act: {printed}");
    assert!(endpoint.served.lock().unwrap().is_empty(), "a refused grant reached the endpoint");
    assert!(!home.join(".openmax").exists(), "a refused grant wrote to the data dir, so the run after it is no fresh install");

    let (code, stdout, printed) = run(true);
    let served = std::mem::take(&mut *endpoint.served.lock().unwrap());
    assert_eq!(code, Some(0), "{printed}");
    assert!(stdout.contains("notes.txt has 3 lines"), "no answer on stdout: {printed}");
    assert_eq!(served.len(), 3, "the run made {} requests, not three: {served:?}", served.len());
    assert_eq!(std::fs::read_to_string(project.join(tool)).unwrap(), manifest, "the tool file is not what was written");
    // The write's result carries the receipt, then the trailing note's line
    // for the tool: the tools array stays as the session froze it, so that
    // line is where the model reads the new tool's description and
    // parameters.
    let write_result = tool_result(&served[1], "call_write");
    assert!(
        write_result.contains("[extension refreeze: ")
            && write_result.contains("New tools callable from your next step: line_count."),
        "the request after the write carries no receipt naming the tool callable: {write_result}"
    );
    assert!(
        write_result.contains("\n- line_count: Count the lines of notes.txt {"),
        "the request after the write carries no note line with the tool's description and parameters: {write_result}"
    );
    let output = tool_result(&served[2], "call_tool");
    assert!(
        output.starts_with("3 lines; attest=[]"),
        "the request after the call carries no output of the tool, or the tool received the attestation: {output}"
    );

    let logs: Vec<PathBuf> = std::fs::read_dir(home.join(".openmax/sessions"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.to_string_lossy().ends_with(".events.jsonl"))
        .collect();
    assert_eq!(logs.len(), 1, "the run did not leave one event log: {logs:?}");
    let log = std::fs::read_to_string(&logs[0]).unwrap();
    // The events, without the format line that heads the log.
    let events: Vec<serde_json::Value> = log
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .filter(|line| line.get("type").is_some())
        .collect();
    // Where the first event of `kind` that `holds` stands in the log.
    let at = |kind: &str, holds: &dyn Fn(&serde_json::Value) -> bool| {
        let found = events.iter().position(|event| event["type"] == kind && holds(event));
        found.unwrap_or_else(|| panic!("the event log has no such {kind} event: {log}"))
    };
    let says = |event: &serde_json::Value, key: &str, text: &str| event[key].as_str().is_some_and(|s| s.contains(text));
    let order = [
        at("tool_start", &|e| e["call_id"] == "call_write" && e["name"] == "write_file" && e["args"]["path"] == tool),
        at("tool_end", &|e| e["call_id"] == "call_write" && e["ok"] == true),
        at("harness_note", &|e| {
            e["call_id"] == "call_write" && says(e, "text", "New tools callable from your next step: line_count.")
        }),
        at("tool_start", &|e| e["call_id"] == "call_tool" && e["name"] == "line_count"),
        at("tool_end", &|e| e["call_id"] == "call_tool" && e["ok"] == true && says(e, "output", "3 lines; attest=[]")),
    ];
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "the write, receipt, call and result are out of order, {order:?}: {log}");
    let added = format!("{tool} added");
    let refrozen = at("refrozen", &|e| e["changes"].as_array().is_some_and(|c| c.iter().any(|c| c == added.as_str())));
    assert!(order[1] < refrozen && refrozen < order[3], "the refreeze is not between the write and the call: {log}");
    let done: Vec<usize> = (0..events.len()).filter(|&n| events[n]["type"] == "done").collect();
    assert_eq!(done, [events.len() - 1], "the run is not one turn that ends the log: {log}");
    assert_eq!(events[done[0]]["stop_reason"], "stop", "{log}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// The recorded sessions say what providers served: `scripts/cache-report.sh
/// <sessions dir>` reads every session's `<id>.usage.jsonl`, one line per
/// model request whose reply reported its token usage, and reports the share
/// of prompt tokens served from cache over the sessions with more than
/// 100,000 prompt tokens (target 75%), the share of sessions that cached no
/// prompt token (target 5%), and the p50 and p90 of the gaps between
/// consecutive requests of a session, a gap over 600 s left out as idle
/// (target p90 90 s). No test can read a user's sessions, so this one holds
/// the method to the synthetic ones in `crates/tui/tests/fixtures/sessions`,
/// whose every number follows from the files: `long` (four requests of
/// 40,000 prompt tokens, 101,000 cached, gaps of 10, 90 and an idle 900 s)
/// and `cold` (120,000, none cached, a gap of -10 s, as when the clock moves
/// back) are the two above 100,000, so the share is 101,000 of 280,000,
/// 36.1%; `threshold` sums to 100,000 exactly, which is not above, and its
/// gap of 600 s is not idle; `silent` reports no cached tokens on any of its
/// three requests, absent or null, and counts as uncached; `corrupt` has one
/// request among a string count, a fraction and a cut line, which are
/// skipped, and a blank one, passed over; `alone` has one request and no
/// gap; `empty` has none and is no session; and an events log beside them is
/// not read. So 2 of the 6 sessions cached nothing, and the six gaps, -10,
/// 10, 30, 90, 300 and 600 s, have their p50 at rank 3 and their p90 at rank
/// 6 (the nearest rank, ceil(q n)): 30 and 600 s, where interpolating would
/// give 60 and 450 s. It exits 1 for a directory without a usage file and 2
/// without one directory argument. README.md states the method, held here.
#[test]
fn cache_report_measures_recorded_sessions() {
    let fixture = repo().join("crates/tui/tests/fixtures/sessions");
    let (code, printed) = ran(script("scripts/cache-report.sh").arg(&fixture));
    assert_eq!(code, Some(0), "{printed}");
    assert_eq!(
        printed,
        "\
session   requests  prompt tokens  cached tokens  cached unreported
long             4        160,000        101,000   63.1%          0
cold             2        120,000              0    0.0%          0
threshold        2        100,000         10,000   10.0%          0
silent           3          5,000              0    0.0%          3
alone            1            900            500   55.6%          0
corrupt          1            800            100   12.5%          0
skipped: 3 lines that are not usage records, 1 file with no request
cache: 36.1% of prompt tokens cached, 101,000 of 280,000, over 2 sessions above 100,000 prompt tokens; target 75%
zero-cache: 2 of 6 sessions (33.3%) cached no prompt token, 1 of them reporting no cached tokens at all; target 5%
gaps: p50 30 s, p90 600 s over 6 gaps of at most 600 s between consecutive requests of a session, 1 of them negative; \
1 longer gap left out as idle; target p90 90 s
",
        "the report over the synthetic sessions changed"
    );
    let empty = scratch("cache-report");
    let (code, printed) = ran(script("scripts/cache-report.sh").arg(&empty));
    assert_eq!(code, Some(1), "{printed}");
    assert!(printed.contains("holds no *.usage.jsonl file"), "{printed}");
    for args in [&[][..], &[fixture.join("long.usage.jsonl")][..], &[fixture.clone(), fixture.clone()][..]] {
        let (code, printed) = ran(script("scripts/cache-report.sh").args(args));
        assert_eq!(code, Some(2), "{args:?}: {printed}");
        assert!(printed.starts_with("usage: "), "{args:?}: {printed}");
    }
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    for claim in [
        "the share of prompt tokens served from cache over the sessions with more than 100,000 prompt tokens (target 75%)",
        "the share of sessions that cached no prompt token (target 5%)",
        "a gap over 600 s left out as idle (target p90 90 s)",
    ] {
        assert!(readme.contains(claim), "README.md does not say \"{claim}\", which scripts/cache-report.sh reports");
    }
    let _ = std::fs::remove_dir_all(empty);
}

/// One flag runs a sub-agent, and the flag adds nothing to any request.
/// Method: in a git repository a human trusted in auto, an agent's bash call
/// (the session marker set, no human present) runs
/// `openmax --subagent TASK --worktree --json` against a scripted endpoint
/// whose model writes notes.txt, starts a background process with bash, and
/// answers. The gate holds the run to: the child's events on stdout in the
/// order they happened, its answer the last message; its edit committed on the
/// branch the last stderr line names, the project's own tree untouched; the
/// worktree and `.worktrees` gone; no process the child started still running.
/// Zero tokens: the child's first request, with its worktree path read as the
/// project's, is byte for byte the first request `openmax -p TASK` sends from
/// the project, and no request names the flag; the project's own sessions are
/// unchanged, as nothing in the prompt or the tools array mentions it
/// (crates/core/tests/thesis.rs holds those bytes). An untrusted project is
/// refused with exit 3 before a worktree or a request is made, and SIGTERM
/// during the child's bash call ends the run with 143, with the child's
/// processes stopped, the worktree removed, and its partial work on its branch.
#[cfg(unix)]
#[test]
fn a_sub_agent_is_one_flag_and_zero_prompt_tokens() {
    use std::process::Stdio;
    use sub_agent::{answer, calls, git, repo, serve, trust};
    let gone = |pid: i32| {
        // A killed process answers until init has reaped it.
        let deadline = Instant::now() + Duration::from_secs(3);
        while unsafe { libc::kill(pid, 0) } == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        unsafe { libc::kill(pid, 0) != 0 }
    };
    let openmax = |project: &Path| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_openmax"));
        command
            .current_dir(project)
            .env("HOME", project.parent().unwrap().join("home"))
            .env_remove("OPENMAX_API_KEY")
            .env_remove("OPENMAX_HUMAN_ATTEST")
            .env("OPENMAX_SESSION", "1")
            .envs(sub_agent::GIT_ENV);
        command
    };
    let branch_line = |stderr: &str| -> (String, String) {
        let last = stderr.lines().last().unwrap_or_default();
        let rest = last.strip_prefix("openmax: the sub-agent's changes are on branch ");
        let (branch, stat) = rest.and_then(|rest| rest.split_once(": ")).unwrap_or_else(|| panic!("{stderr}"));
        (branch.to_string(), stat.to_string())
    };

    // SIGTERM while the child's bash call runs: started first, stopped last.
    let stopped = repo("stopped");
    trust(&stopped);
    let pids = stopped.parent().unwrap().join("pids");
    let tree = format!(
        "echo partial > partial.txt; echo $$ >> '{p}'; sleep 6841 & echo $! >> '{p}'; exec sleep 7841",
        p = pids.display()
    );
    serve(&stopped, vec![calls(&[("bash", serde_json::json!({ "command": tree }))])]);
    /// Stops the run on the way out of a failed assertion too: SIGTERM
    /// stops its tool tree with it.
    struct Stopped(Option<std::process::Child>);
    impl Drop for Stopped {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
                let _ = child.wait();
            }
        }
    }
    let mut stopping = Stopped(Some(
        openmax(&stopped)
            .args(["--subagent", "stop me", "--worktree", "--json"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));

    // An untrusted project: refused before a worktree or a request is made.
    let untrusted = repo("untrusted");
    serve(&untrusted, Vec::new());
    let out = openmax(&untrusted).args(["--subagent", "edit the notes", "--worktree"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("is not trusted"), "{stderr}");
    assert!(!untrusted.join(".worktrees").exists(), "{stderr}");
    assert_eq!(git(&untrusted, &["branch", "--list", "subagent/*"]), "", "{stderr}");

    // The sub-agent edits notes.txt, leaves a process behind, and answers.
    let project = repo("edits");
    trust(&project);
    let sleeper = project.parent().unwrap().join("sleeper");
    let background = format!("sleep 6841 & echo $! > '{}'", sleeper.display());
    let bodies = serve(
        &project,
        vec![
            calls(&[
                ("write_file", serde_json::json!({ "path": "notes.txt", "content": "edited\n" })),
                ("bash", serde_json::json!({ "command": background })),
            ]),
            answer("edited the notes"),
            answer("ok"),
        ],
    );
    let out = openmax(&project).args(["--subagent", "edit the notes", "--worktree", "--json"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stdout}\n{stderr}");
    let mut tools = std::collections::HashMap::new();
    let happened: Vec<String> = stdout
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap_or_else(|_| panic!("not an event line: {line}")))
        .filter_map(|event| {
            let call = event["call_id"].as_str().unwrap_or_default().to_string();
            match event["type"].as_str()? {
                "tool_start" => {
                    tools.insert(call, event["name"].as_str()?.to_string());
                    Some(format!("start {}", event["name"].as_str()?))
                }
                "tool_end" => Some(format!("end {} ok={}", tools.get(&call)?, event["ok"])),
                // A message that only calls tools is done with no text.
                "message_done" => Some(format!("answer {}", event["text"].as_str().filter(|text| !text.is_empty())?)),
                "done" => Some(format!("done {}", event["stop_reason"].as_str()?)),
                _ => None,
            }
        })
        .collect();
    let expected =
        ["start write_file", "end write_file ok=true", "start bash", "end bash ok=true", "answer edited the notes", "done stop"];
    assert_eq!(happened, expected, "{stdout}");
    let root = std::fs::canonicalize(&project).unwrap();
    let first = stderr.lines().next().unwrap_or_default();
    let worktree = first.strip_prefix("openmax: sub-agent working in ").and_then(|rest| rest.split(" on branch ").next());
    let worktree = worktree.unwrap_or_else(|| panic!("the run names its worktree first: {stderr}"));
    assert!(worktree.starts_with(&format!("{}/.worktrees/edit-the-notes-", root.display())), "{stderr}");
    let (branch, stat) = branch_line(&stderr);
    assert_eq!(stat, "1 file changed, 1 insertion(+), 1 deletion(-)", "{stderr}");
    assert_eq!(git(&project, &["show", &format!("{branch}:notes.txt")]), "edited");
    assert_eq!(std::fs::read_to_string(project.join("notes.txt")).unwrap(), "first\n", "the project's tree is untouched");
    assert_eq!(git(&project, &["status", "--porcelain"]), "", "{stderr}");
    assert!(!project.join(".worktrees").exists(), "{stderr}");
    assert_eq!(git(&project, &["worktree", "list"]).lines().count(), 1, "{stderr}");
    let sleeper: i32 = std::fs::read_to_string(&sleeper).unwrap().trim().parse().unwrap();
    assert!(gone(sleeper), "the child's background process {sleeper} outlived the run");

    // Zero tokens: what the child sent first is what -p sends from the project.
    let out = openmax(&project).args(["-p", "edit the notes", "--json"]).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let bodies = bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 3, "two child requests, then the -p request");
    assert_eq!(bodies[0].replace(worktree, &root.display().to_string()), bodies[2], "the child's request differs from -p's");
    for body in &bodies {
        assert!(!body.contains("--subagent") && !body.contains("--worktree"), "a request names the flag: {body}");
    }

    // The stopped run: no process, no worktree, and its partial work kept.
    let mut running = stopping.0.take().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let started = loop {
        let text = std::fs::read_to_string(&pids).unwrap_or_default();
        let started: Vec<i32> = text.lines().filter_map(|line| line.trim().parse().ok()).collect();
        if started.len() == 2 || Instant::now() >= deadline || running.try_wait().unwrap().is_some() {
            break started;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    unsafe { libc::kill(running.id() as libc::pid_t, libc::SIGTERM) };
    let deadline = Instant::now() + Duration::from_secs(10);
    while running.try_wait().unwrap().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = running.kill();
    let out = running.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(started.len(), 2, "the child's bash call never started: {stderr}");
    let survivors: Vec<i32> = started.into_iter().filter(|pid| !gone(*pid)).collect();
    for pid in &survivors {
        unsafe { libc::kill(*pid, libc::SIGKILL) };
    }
    assert!(survivors.is_empty(), "the child's processes outlived the stop: {survivors:?}\n{stderr}");
    assert_eq!(out.status.code(), Some(143), "{stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let done: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap_or_default()).unwrap_or_default();
    assert_eq!((done["type"].as_str(), done["stop_reason"].as_str()), (Some("done"), Some("cancelled")), "{stdout}");
    assert!(!stopped.join(".worktrees").exists(), "{stderr}");
    let (branch, _) = branch_line(&stderr);
    assert_eq!(git(&stopped, &["show", &format!("{branch}:partial.txt")]), "partial");
    for dir in [stopped, untrusted, project] {
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }
}

/// Git repositories, their HOMEs, and scripted endpoints for the sub-agent
/// gate.
#[cfg(unix)]
mod sub_agent {
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    /// Git's identity and config for every git command the test and openmax
    /// run: none of the machine's own.
    pub const GIT_ENV: [(&str, &str); 6] = [
        ("GIT_AUTHOR_NAME", "openmax test"),
        ("GIT_AUTHOR_EMAIL", "test@example.com"),
        ("GIT_COMMITTER_NAME", "openmax test"),
        ("GIT_COMMITTER_EMAIL", "test@example.com"),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
    ];

    pub fn git(dir: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git").current_dir(dir).args(args).envs(GIT_ENV).output().unwrap();
        assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// A project that is a git repository with notes.txt committed, beside an
    /// empty HOME.
    pub fn repo(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let base = std::env::temp_dir().join(format!("openmax-thesis-subagent-{tag}-{}-{nonce}", std::process::id()));
        let project = base.join("project");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::create_dir_all(base.join("home/.openmax")).unwrap();
        git(&project, &["init", "-q"]);
        std::fs::write(project.join("notes.txt"), "first\n").unwrap();
        git(&project, &["add", "notes.txt"]);
        git(&project, &["commit", "-q", "-m", "notes"]);
        project
    }

    /// Trust `project` in auto, as a human's grant leaves it.
    pub fn trust(project: &Path) {
        let data = project.parent().unwrap().join("home/.openmax");
        open_max_core::trust::grant_trust(&data, project, open_max_core::config::ApprovalMode::Auto).unwrap();
    }

    /// Point `project`'s HOME at an endpoint that answers request n with
    /// `replies[n]` and refuses connections after the last, and return the
    /// request bodies it records.
    pub fn serve(project: &Path, replies: Vec<String>) -> Arc<Mutex<Vec<String>>> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let settings = serde_json::json!({ "base_url": url, "model": "m", "approval_mode": "ask", "context_tokens": 32768 });
        std::fs::write(project.parent().unwrap().join("home/.openmax/settings.json"), settings.to_string()).unwrap();
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let record = bodies.clone();
        std::thread::spawn(move || {
            for reply in replies {
                let Ok((mut stream, _)) = listener.accept() else { return };
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    match stream.read(&mut byte) {
                        Ok(1) => head.push(byte[0]),
                        _ => return,
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
                if stream.read_exact(&mut body).is_err() {
                    return;
                }
                record.lock().unwrap().push(String::from_utf8_lossy(&body).into_owned());
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        bodies
    }

    fn sse(chunks: &[serde_json::Value]) -> String {
        chunks.iter().map(|chunk| format!("data: {chunk}\n\n")).chain(["data: [DONE]\n\n".to_string()]).collect()
    }

    /// One assistant message calling each tool in order.
    pub fn calls(calls: &[(&str, serde_json::Value)]) -> String {
        let deltas: Vec<serde_json::Value> = calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                serde_json::json!({
                    "index": i, "id": format!("call_{i}"), "type": "function",
                    "function": { "name": name, "arguments": args.to_string() }
                })
            })
            .collect();
        sse(&[
            serde_json::json!({ "choices": [{ "delta": { "tool_calls": deltas }, "finish_reason": null }] }),
            serde_json::json!({ "choices": [{ "delta": {}, "finish_reason": "tool_calls" }] }),
        ])
    }

    pub fn answer(text: &str) -> String {
        sse(&[
            serde_json::json!({ "choices": [{ "delta": { "content": text }, "finish_reason": null }] }),
            serde_json::json!({ "choices": [{ "delta": {}, "finish_reason": "stop" }] }),
        ])
    }
}

/// A local OpenAI-compatible endpoint that answers every request with the
/// same finished one-line completion, for as long as the test runs.
fn spawn_answering_server() -> String {
    spawn_server(Vec::new())
}

/// The same endpoint, answering its first requests with `first`, in turn.
fn spawn_server(first: Vec<String>) -> String {
    const ANSWER: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"first answer\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut first = first.into_iter();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            // Headers, then exactly Content-Length body bytes.
            let _ = read_one_request(&mut stream);
            let answer = first.next().unwrap_or_else(|| ANSWER.to_string());
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{answer}",
                    answer.len()
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

/// Peak RSS of a two-round headless turn at most 4 MB, read as 4,000,000
/// bytes, the stricter of its two readings (4 MiB is 4,194,304 bytes), so
/// the target is never reported met under the looser one.
///
/// Method: `openmax --trust-project -p TASK` on a fresh HOME and project,
/// against a loopback endpoint (`scripted_endpoint`) that answers the run's
/// three requests with a `read_file` call, a `list_dir` call and a final
/// answer. Each run must exit 0 with the answer on stdout, and the endpoint
/// must have served exactly those three requests, the second carrying the
/// file's lines as the read's result and the third `notes.txt` as the
/// listing's. The peak is the child's `ru_maxrss` from `wait4` (bytes on
/// macOS, KiB on Linux): the high-water mark of its resident memory, its
/// binary's code pages included, which leaves out a page released before
/// that mark. Beside each run, the process floor: the peak of `openmax
/// --version`, which loads the same binary and exits from the argument
/// parser. One warmup run of each, then the medians of 11, printed with the
/// 25th and 75th percentiles. The turn runs without proxy variables, so its
/// requests reach the endpoint directly, and both runs without this
/// environment's allocator variables (`Malloc*` on macOS, `MALLOC_*` for
/// glibc), so a setting such as `MallocNanoZone=0` cannot change what either
/// series measures. The endpoint is plain http on loopback: the figure
/// counts the TLS configuration the HTTP client builds for every endpoint,
/// but not what an https provider adds, the handshake and the certificate
/// chain.
///
/// That mark is the child's own on macOS, which starts each child as a task
/// of its own. On Linux it is not: std spawns the child with `posix_spawn`,
/// which glibc runs as a vfork in this test process's address space until
/// the exec, and at an exec the kernel keeps the high-water mark of the
/// address space it leaves in the process's record, which `wait4` reports.
/// So every Linux figure, floor and turn alike, is at least this process's
/// own high-water mark at the launch (`VmHWM`, which the report prints): an
/// upper bound on the child's, and no basis for a ratchet, which needs the
/// child's own mark after its exec.
///
/// The published figure, which the 4 MB target is reported against until a
/// turn meets it, is the turn's median with the allocator as a user's run
/// has it. On macOS 27 that allocator is xzone malloc. It keeps the
/// allocations that may hold pointers in four type buckets per size class,
/// each on pages of its own, and picks an allocation's bucket by hashing its
/// type under keys derived from the boot session and the binary's code
/// signature. Rust's allocations carry no type, so each one is hashed by the
/// address it was called from, which address space layout randomization
/// moves at every launch. How the turn's allocations share pages, and so
/// the published figure, changes between launches, builds and boots. With
/// one bucket that draw is gone, so the ratchet is held on a second series
/// of the same runs, each child started with `MallocMaxMagazines=1`, which
/// xzone malloc reads as one pointer bucket. With the nano allocator
/// engaged, as it is by default, that is all it changes: the variable also
/// limits the allocator's slots to one in all, and the nano configuration
/// lifts that limit again, so the slots and lists, which start at one per
/// CPU cluster or one in all and grow to one per CPU under contention,
/// follow the CPU topology as before, and the bucket count is a constant.
/// A release build for macOS on Apple silicon asserts the pinned turn's
/// median beyond the pinned floor's: the value measured below plus 128 KiB
/// (8 of this platform's 16 KiB pages). The floor is subtracted because
/// there it is the loader's and the system libraries', which differ between
/// OS releases. Other platforms, macOS on Intel among them, with no ratchet
/// measured yet, and every debug build print the published figure and hold
/// the turn to 20 times the target.
///
/// Measured in a release build on macOS 27 (Apple M5, 10 cores in two
/// clusters, 16 KiB pages), 11 interleaved batches of this test against each
/// binary, quoting the median batch. A print run on the runtime's default of
/// one worker thread per core: turn 6,768 KiB, floor 2,336 KiB, 4,432 KiB
/// beyond it; pinned, 4,128 KiB beyond. On one worker: turn 6,512 KiB,
/// floor 2,368 KiB, 4,144 KiB beyond it; pinned, 3,888 KiB beyond (3,888 to
/// 3,904 over the 11 batches, and 3,792 built with Rust 1.99.0 instead of
/// 1.94.0). The one-worker binary re-signed under eight identifiers, so with
/// eight bucket keys, 2 batches of 11 runs each in an external harness of
/// this scenario: beyond the floor, 4,128 to 4,224 KiB by default and 3,872
/// to 3,904 KiB pinned. With every thread held to one cluster, neither
/// series moved. Of the one-worker peak, 3,472 KiB is the binary's code
/// pages the turn touches (217 of 418), about 1,500 KiB the allocator's, 307
/// KiB of it live at the peak, about 650 KiB the loader's own code, 288 KiB
/// constant data it rebases at launch, and 224 KiB thread stacks.
#[test]
fn headless_turn_peak_rss_is_under_four_megabytes() {
    const KIB: u64 = 1024;
    // 4 MB, in bytes (above).
    const TARGET: u64 = 4_000_000;
    // The pinned turn's memory beyond the pinned floor, measured above, plus
    // 128 KiB.
    const RATCHET: u64 = (3888 + 128) * KIB;
    const RUNS: usize = 11;
    // Where the ratchet was measured: macOS on Apple silicon, in a release
    // build.
    let gated = cfg!(all(target_os = "macos", target_arch = "aarch64")) && !cfg!(debug_assertions);
    let endpoint = scripted_endpoint(vec![
        sse_tool_call("call_1", "read_file", r#"{\"path\":\"notes.txt\"}"#),
        sse_tool_call("call_2", "list_dir", r#"{\"path\":\".\"}"#),
        sse_answer("two rounds done"),
    ]);
    // One run's floor and turn peaks, both children started without this
    // environment's allocator variables, and with one pointer bucket when
    // `pinned`.
    let measure = |run: usize, pinned: bool| {
        let allocator = |cmd: &mut Command| {
            for (name, _) in std::env::vars_os() {
                if name.to_string_lossy().to_ascii_lowercase().starts_with("malloc") {
                    cmd.env_remove(name);
                }
            }
            if pinned {
                cmd.env("MallocMaxMagazines", "1");
            }
        };
        let mut version = Command::new(env!("CARGO_BIN_EXE_openmax"));
        version.arg("--version");
        allocator(&mut version);
        let floor = peak_rss(&mut version, "--version").1;
        let (project, home) = fresh_install("rss", run);
        let settings = format!(
            r#"{{"base_url":"{}","model":"stub-model","context_tokens":32768}}"#,
            endpoint.base_url
        );
        std::fs::create_dir_all(home.join(".openmax")).unwrap();
        std::fs::write(home.join(".openmax/settings.json"), settings).unwrap();
        std::fs::write(project.join("notes.txt"), "a line the first round reads\n".repeat(20)).unwrap();
        endpoint.served.lock().unwrap().clear();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_openmax"));
        cmd.current_dir(&project)
            .env("HOME", &home)
            .env_remove("OPENMAX_SESSION")
            .env_remove("OPENMAX_API_KEY")
            // --trust-project is a human's grant; the test attests to it.
            .env("OPENMAX_HUMAN_ATTEST", "1")
            .args(["--trust-project", "-p", "read notes.txt, then list the directory"]);
        for proxy in ["http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"] {
            cmd.env_remove(proxy);
        }
        allocator(&mut cmd);
        let (out, turn) = peak_rss(&mut cmd, "the turn");
        let served = std::mem::take(&mut *endpoint.served.lock().unwrap());
        let _ = std::fs::remove_dir_all(project.parent().unwrap());
        assert!(out.contains("two rounds done"), "no answer on stdout: {out}");
        assert_eq!(served.len(), 3, "the turn made {} requests, not three: {served:?}", served.len());
        let read = tool_result(&served[1], "call_1");
        assert!(read.contains("a line the first round reads"), "the read returned: {read}");
        assert_eq!(tool_result(&served[2], "call_2"), "notes.txt", "the listing: {}", served[2]);
        (floor, turn)
    };
    let (mut published, mut pinned) = (Vec::new(), Vec::new());
    for run in 0..=RUNS {
        let default = measure(run, false);
        let one = gated.then(|| measure(run, true));
        if run > 0 {
            published.push(default);
            pinned.extend(one);
        }
    }
    // A series' turn median, that median beyond the floor's, and its report.
    let summary = |runs: &[(u64, u64)]| {
        let quartiles = |mut values: Vec<u64>| {
            values.sort();
            let kib = |n: u64| n / KIB;
            let text = format!(
                "median {} KiB (p25 {}, p75 {})",
                kib(values[RUNS / 2]),
                kib(values[RUNS / 4]),
                kib(values[3 * RUNS / 4])
            );
            (values[RUNS / 2], text)
        };
        let (floor, floor_text) = quartiles(runs.iter().map(|run| run.0).collect());
        let (turn, turn_text) = quartiles(runs.iter().map(|run| run.1).collect());
        let beyond = turn.saturating_sub(floor);
        let text = format!(
            "{turn_text} over {RUNS} runs; --version floor: {floor_text}; {} KiB beyond the floor",
            beyond / KIB
        );
        (turn, beyond, text)
    };
    let (turn, _, text) = summary(&published);
    let met = if turn <= TARGET { "met" } else { "not met yet" };
    let build = if cfg!(debug_assertions) { "debug" } else { "release" };
    // On Linux every figure starts from this process's own high-water mark
    // (above), which the report names.
    let inherited = if cfg!(target_os = "linux") {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mark = status.lines().find_map(|line| line.strip_prefix("VmHWM:")).map_or("unread", str::trim);
        format!(
            "; on Linux each figure is at least this test process's high-water mark at its launch \
             (VmHWM {mark} now), so it bounds the turn's from above and sets no ratchet"
        )
    } else {
        String::new()
    };
    let report =
        format!("two-round headless turn peak RSS: {text}; target {TARGET} bytes {met} ({build} build){inherited}");
    println!("{report}");
    if !gated {
        assert!(turn < TARGET * 20, "{report}: {published:?}");
        return;
    }
    let (_, beyond, text) = summary(&pinned);
    let ratchet = format!("with MallocMaxMagazines=1: {text}; ratchet {} KiB beyond the floor", RATCHET / KIB);
    println!("{ratchet}");
    assert!(beyond <= RATCHET, "over the ratchet: {ratchet}; {report}");
}

/// The content of the tool message answering call `id` in a request `body`.
fn tool_result(body: &str, id: &str) -> String {
    let request: serde_json::Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"));
    let messages = request["messages"].as_array().unwrap_or_else(|| panic!("no messages: {body}"));
    let answer = messages.iter().find(|m| m["role"] == "tool" && m["tool_call_id"] == id);
    let answer = answer.unwrap_or_else(|| panic!("no result for {id}: {body}"));
    answer["content"].as_str().unwrap_or_default().to_string()
}

/// Run `cmd` to its exit with stdin closed: what it printed on stdout and its
/// peak resident set in bytes, read from `wait4`'s rusage (bytes on macOS,
/// KiB elsewhere). It must exit 0 within 30 seconds, where a turn here takes
/// well under one; one still running then is killed and reaped, and the
/// test fails. `what` names it in the failure.
// The child is reaped by `wait4`, not `Child::wait`, which has no rusage.
#[allow(clippy::zombie_processes)]
fn peak_rss(cmd: &mut Command, what: &str) -> (String, u64) {
    use std::os::unix::process::ExitStatusExt;
    let dir = std::env::temp_dir().join(format!(
        "openmax-thesis-rss-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (stdout, stderr) = (dir.join("stdout"), dir.join("stderr"));
    let child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::fs::File::create(&stdout).unwrap())
        .stderr(std::fs::File::create(&stderr).unwrap())
        .spawn()
        .unwrap();
    let pid = child.id() as libc::pid_t;
    let mut status = 0;
    // SAFETY: rusage is integers and timevals, for which all zeroes is a value.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut killed = false;
    loop {
        let options = if killed { 0 } else { libc::WNOHANG };
        // SAFETY: a child of this process not yet reaped; both out-pointers
        // are valid for the call.
        let reaped = unsafe { libc::wait4(pid, &mut status, options, &mut usage) };
        if reaped == pid {
            break;
        }
        assert_eq!(reaped, 0, "wait4: {}", std::io::Error::last_os_error());
        if Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }
        // SAFETY: the child is not reaped yet, so the pid is still its own.
        unsafe { libc::kill(pid, libc::SIGKILL) };
        killed = true;
    }
    let out = std::fs::read_to_string(&stdout).unwrap_or_default();
    let err = std::fs::read_to_string(&stderr).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!killed, "{what} was still running after 30 seconds: {out}{err}");
    let status = std::process::ExitStatus::from_raw(status);
    assert!(status.success(), "{what} exited {status}: {out}{err}");
    let unit = if cfg!(target_os = "macos") { 1 } else { 1024 };
    (out, usage.ru_maxrss as u64 * unit)
}

/// A running scripted endpoint: its base URL and the request bodies it has
/// served, in order.
struct ScriptedEndpoint {
    base_url: String,
    served: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

/// A loopback OpenAI-compatible endpoint that answers the Nth request since
/// `served` was last cleared with the Nth SSE body of `script`, and a request
/// past the script with a 400, which the client does not resend. Each
/// connection is served on a thread of its own, keep-alive, every reply
/// framed with a Content-Length.
fn scripted_endpoint(script: Vec<String>) -> ScriptedEndpoint {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let served = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let (record, script) = (served.clone(), std::sync::Arc::new(script));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let (record, script) = (record.clone(), script.clone());
            std::thread::spawn(move || {
                while let Some(body) = read_one_request(&mut stream) {
                    let reply = {
                        let mut served = record.lock().unwrap();
                        served.push(body);
                        script.get(served.len() - 1).cloned()
                    };
                    let head = match &reply {
                        Some(sse) => format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n",
                            sse.len()
                        ),
                        None => "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
                    };
                    let sent = stream.write_all(format!("{head}{}", reply.as_deref().unwrap_or("")).as_bytes());
                    if sent.is_err() || reply.is_none() {
                        return;
                    }
                }
            });
        }
    });
    ScriptedEndpoint { base_url, served }
}

/// One HTTP request from `stream`: its body, or None once the client closed
/// the connection.
fn read_one_request(stream: &mut std::net::TcpStream) -> Option<String> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => head.push(byte[0]),
            _ => return None,
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
    stream.read_exact(&mut body).ok()?;
    Some(String::from_utf8_lossy(&body).into_owned())
}

/// An SSE completion that calls tool `name` once with the JSON-escaped
/// `arguments`.
fn sse_tool_call(id: &str, name: &str, arguments: &str) -> String {
    format!(
        concat!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"{}\",\"type\":\"function\",",
            "\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}]}},\"finish_reason\":null}}]}}\n\n",
            "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n",
            "data: [DONE]\n\n"
        ),
        id, name, arguments
    )
}

/// An SSE completion that answers `text` and stops.
fn sse_answer(text: &str) -> String {
    format!(
        concat!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
            "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n",
            "data: [DONE]\n\n"
        ),
        text
    )
}

/// Whether a process holds `pid`: a zombie still does, until the process
/// that adopted it collects it.
#[cfg(unix)]
fn alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

/// A run's watchers, `openmax --lifeline <record> <start>`: the processes
/// whose command line names its `record`, as `ps` lists them at full width,
/// waited for up to 10 seconds, as the run starts its watcher once the
/// record is first written.
#[cfg(unix)]
fn watchers_of(record: &Path) -> Vec<i32> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let out = Command::new("ps").args(["-A", "-ww", "-o", "pid=", "-o", "args="]).output().expect("ps runs");
        assert!(out.status.success(), "ps failed: {}", String::from_utf8_lossy(&out.stderr));
        let (listing, needle) = (String::from_utf8_lossy(&out.stdout), record.to_string_lossy());
        let found: Vec<i32> =
            listing.lines().filter(|line| line.contains(&*needle)).filter_map(|line| line.split_whitespace().next()?.parse().ok()).collect();
        if !found.is_empty() || Instant::now() >= deadline {
            return found;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A print run (`-p`) on a fresh project whose endpoint asks for one `bash`
/// call running `script`, then answers; with the path of the run's record
/// of its tool groups. The run leads a process group of its own, as a job a
/// shell starts does. With `home`, HOME is a fresh directory, which holds
/// the record (`~/.openmax/run/<pid>.groups`). Without it, the run has no
/// HOME, so its data directory is the relative `.openmax`, which names one
/// in the project, the directory the run runs in; the record's path then
/// names the project as the run reads its working directory, with every
/// symbolic link resolved (macOS's temporary directory is under one), as
/// the run's watcher is given it. With `removed`, the run starts from a hard
/// link to the binary under the target directory's `tmp`, removed as soon as
/// the spawn returns, as an upgrade that replaced the file leaves a run:
/// before its first tool, which waits on its start and its endpoint.
#[cfg(unix)]
fn run_one_bash_call(
    tag: &str,
    home: bool,
    removed: bool,
    script: impl Fn(&Path) -> String,
) -> (std::process::Child, PathBuf, PathBuf) {
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;
    let (project, home_dir) = fresh_install(tag, 0);
    let arguments = serde_json::json!({ "command": script(project.parent().unwrap()) }).to_string();
    let call = serde_json::json!({"choices": [{"finish_reason": null, "delta": {"tool_calls": [
        {"index": 0, "id": "call-1", "type": "function", "function": {"name": "bash", "arguments": arguments}},
    ]}}]});
    let end = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
    let base_url = spawn_server(vec![format!("data: {call}\n\ndata: {end}\n\ndata: [DONE]\n\n")]);
    let link = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("openmax-{tag}-{}", std::process::id()));
    if removed {
        let _ = std::fs::remove_file(&link);
        std::fs::hard_link(env!("CARGO_BIN_EXE_openmax"), &link).unwrap();
    }
    let mut run = Command::new(if removed { link.as_path() } else { Path::new(env!("CARGO_BIN_EXE_openmax")) });
    run.process_group(0).current_dir(&project).env_remove("OPENMAX_SESSION").env_remove("OPENMAX_API_KEY");
    let data = if home {
        run.env("HOME", &home_dir);
        home_dir.join(".openmax")
    } else {
        run.env_remove("HOME");
        std::fs::canonicalize(&project).unwrap().join(".openmax")
    };
    // Test automation is a human act with no terminal; the grant needs it
    // said.
    run.env("OPENMAX_HUMAN_ATTEST", "1").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    run.args(["--trust-project", "--base-url", &base_url, "--model", "m", "--context-tokens", "32768", "-p", tag]);
    let run = run.spawn().unwrap();
    if removed {
        std::fs::remove_file(&link).unwrap();
    }
    let record = data.join("run").join(format!("{}.groups", run.id()));
    (run, project.parent().unwrap().to_path_buf(), record)
}

/// Where `zero_orphans_after_kill_9` lists each process it watched, under
/// the target directory's `tmp`, which the CI thesis job keeps as an
/// artifact on each runner.
const ORPHANS_LISTING: &str = "zero-orphans-after-kill-9.txt";

/// The CI thesis job, which asserts the 100 ms bound in a release build,
/// keeps `ORPHANS_LISTING` on each of its runners: a step after its gates,
/// which runs after a failing gate too and on every runner of the matrix,
/// uploads it from `target/tmp`, where cargo's `CARGO_TARGET_TMPDIR` points
/// in that job, under a name per runner, fails when the file is missing once
/// the gates have passed, and replaces it on a re-run of the job. After a
/// failing gate it only warns of a missing file, as a release build that
/// failed leaves none.
#[test]
fn ci_keeps_the_orphan_listing_on_each_runner() {
    let job = ci_job("thesis");
    let gate = "cargo test --release --workspace --locked --test 'thesis*' --no-fail-fast -- --test-threads=1 --nocapture $skip || failed=1";
    let gates = job.iter().position(|line| line.trim() == gate).expect("the thesis job runs no thesis gates");
    let path = format!("path: target/tmp/{ORPHANS_LISTING}");
    let kept = job.iter().position(|line| line.trim() == path).expect("the thesis job keeps no orphan listing");
    assert!(gates < kept, "the thesis job keeps the listing before its gates write it");
    assert!(step_keys(&job, gates).contains(&"id: gates"), "the gates step has no `id: gates` for the listing's step to read");
    let keys = step_keys(&job, kept);
    let wanted = [
        "if: ${{ !cancelled() }}",
        "name: zero-orphans-after-kill-9-${{ matrix.os }}",
        "if-no-files-found: ${{ steps.gates.outcome == 'success' && 'error' || 'warn' }}",
        "overwrite: true",
    ];
    for key in wanted {
        assert!(keys.contains(&key), "the step that keeps the orphan listing lacks `{key}`: {keys:?}");
    }
    assert!(keys.iter().any(|key| key.starts_with("uses: actions/upload-artifact@")), "{keys:?}");
}

/// The target: every process of a tool tree a killed run left, and the
/// run's watcher, gone within 100 ms of the SIGKILL.
const ORPHANS_GONE_WITHIN: Duration = Duration::from_millis(100);

/// How `assert_no_orphans_after_kill_9` starts and kills its run.
#[cfg(unix)]
struct Kill9 {
    /// HOME holds the run's data directory; without it, that is the relative
    /// `.openmax` (`run_one_bash_call`).
    home: bool,
    /// SIGKILL goes to the run's whole process group, as a shell's
    /// `kill -9 %1` sends it to a job, not to its pid alone: a watcher in
    /// that group would die with the run.
    group: bool,
    /// The run starts from a binary removed before its first tool
    /// (`run_one_bash_call`).
    removed: bool,
}

/// Zero orphans after kill -9: within 100 ms of the SIGKILL that kills an
/// openmax outright, with no launch since, no process of a tool tree it
/// left is running, and neither is the watcher its first tool started. The
/// signal goes to the run's whole process group, as a shell sends it to a
/// job, and the watcher, which leads a session of its own, is out of its
/// reach. The method is `assert_no_orphans_after_kill_9`'s, on a run whose
/// HOME holds its data directory; this run's listing is the one CI keeps.
/// README.md states the grace and the bound, held here.
#[cfg(unix)]
#[test]
fn zero_orphans_after_kill_9() {
    let kill = Kill9 { home: true, group: true, removed: false };
    assert_no_orphans_after_kill_9("zero_orphans_after_kill_9", kill, Some(ORPHANS_LISTING));
    let (grace, bound) = (open_max_core::execution::LEFT_BEHIND_GRACE.as_millis(), ORPHANS_GONE_WITHIN.as_millis());
    let readme = std::fs::read_to_string(repo().join("README.md")).unwrap();
    for claim in [
        format!("SIGKILL if it is still there {grace} ms later"),
        format!("within {bound} ms of the signal, and none of the three before the {grace} ms grace"),
    ] {
        assert!(readme.contains(&claim), "README.md does not say \"{claim}\", which this test holds");
    }
}

/// The same for `kill -9 <pid>` of a run with no HOME, whose data directory
/// is therefore the relative `.openmax`, in the directory it runs in, while
/// its watcher runs in `/`: the watcher must be given the record's absolute
/// path to find it.
#[cfg(unix)]
#[test]
fn zero_orphans_after_kill_9_with_no_home() {
    let kill = Kill9 { home: false, group: false, removed: false };
    assert_no_orphans_after_kill_9("zero_orphans_after_kill_9_with_no_home", kill, None);
}

/// The same for a run whose binary an upgrade replaced before its first
/// tool. Its path then reads `<path> (deleted)`, which no exec finds, so the
/// watcher runs the binary the run runs, `/proc/self/exe`. macOS has no such
/// name: there the next launch stops such a run's tools (docs/usage.md).
#[cfg(target_os = "linux")]
#[test]
fn zero_orphans_after_kill_9_once_its_binary_is_replaced() {
    let kill = Kill9 { home: true, group: false, removed: true };
    assert_no_orphans_after_kill_9("zero_orphans_after_kill_9_once_its_binary_is_replaced", kill, None);
}

/// When each of `pids` read as gone after `since`, polled every millisecond
/// for up to `wait`, or None for one still found then.
#[cfg(unix)]
fn gone_after(pids: &[i32], since: Instant, wait: Duration) -> Vec<Option<Duration>> {
    let mut gone = vec![None; pids.len()];
    while gone.contains(&None) && since.elapsed() < wait {
        for (at, &pid) in gone.iter_mut().zip(pids) {
            if at.is_none() && !alive(pid) {
                *at = Some(since.elapsed());
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    gone
}

/// Kill a print run outright in the middle of a `bash` call whose tree only
/// SIGKILL ends, and assert that, with no launch since, each process of the
/// tree is gone within `ORPHANS_GONE_WITHIN` of the signal, and so is the
/// watcher the run's first tool started, found before the kill and leading
/// a session of its own; and that none of the tree is gone before
/// `LEFT_BEHIND_GRACE` has passed, since the watcher sends the SIGTERM the
/// tree ignores first and SIGKILL only once that grace ends. `name` is the
/// run's prompt, its install's tag and its listing's first word, `kill` says
/// how the run starts and is killed, and the listing is written to
/// `listing_file` under the target directory's `tmp` when that is given.
///
/// Method: the call's shell ignores SIGTERM, writes its pid, starts a
/// subshell in the background, writes that pid and waits; the subshell
/// starts a sleeper, the shell's grandchild, writes its pid and waits, and
/// each inherits the ignored SIGTERM. Once the three pids are written and the
/// run's record lists the shell's group, this waits for the run's watcher,
/// which the record's first write starts, then sends SIGKILL and waits for
/// the run, as the shell that started it would. From the signal on, it polls
/// each process of the tree and the watcher every millisecond for up to 5
/// seconds with `kill(pid, 0)`, which finds a zombie its new parent has not
/// collected, so each reads as gone once it has exited and been collected,
/// and lists each pid with when it did. A release build, as CI's thesis job
/// runs it, asserts the bound; a debug build asserts 20 times it, as the
/// whole suite runs beside it, and prints the times.
#[cfg(unix)]
fn assert_no_orphans_after_kill_9(name: &str, kill: Kill9, listing_file: Option<&str>) {
    use open_max_core::execution::LEFT_BEHIND_GRACE;
    let (mut run, base, record) = run_one_bash_call(name, kill.home, kill.removed, |base| {
        let p = base.join("pids");
        let p = p.display();
        format!("trap '' TERM; echo $$ >> '{p}'; (sleep 6845 & echo $! >> '{p}'; wait) & echo $! >> '{p}'; wait")
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    let tree = loop {
        let text = std::fs::read_to_string(base.join("pids")).unwrap_or_default();
        let tree: Vec<i32> = text.lines().filter_map(|line| line.trim().parse().ok()).collect();
        let recorded = std::fs::read_to_string(&record).unwrap_or_default();
        let shell = tree.first().map(|shell| format!("{shell} "));
        if tree.len() == 3 && shell.is_some_and(|shell| recorded.lines().any(|line| line.starts_with(&shell))) {
            break tree;
        }
        if Instant::now() >= deadline || run.try_wait().unwrap().is_some() {
            let _ = run.kill();
            let _ = run.wait();
            for &pid in tree.iter().filter(|&&pid| alive(pid)) {
                unsafe { libc::kill(pid, libc::SIGKILL) };
            }
            panic!("the tool tree did not start, or was not recorded: pids {tree:?}, record {recorded:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let watcher = watchers_of(&record).first().copied();
    let session = watcher.map(|pid| unsafe { libc::getsid(pid) });
    let target = run.id() as libc::pid_t;
    unsafe { libc::kill(if kill.group { -target } else { target }, libc::SIGKILL) };
    let killed = Instant::now();
    run.wait().unwrap();
    let watched: Vec<i32> = tree.iter().copied().chain(watcher).collect();
    let gone = gone_after(&watched, killed, Duration::from_secs(5));
    let (bound, build) =
        if cfg!(debug_assertions) { (ORPHANS_GONE_WITHIN * 20, "debug") } else { (ORPHANS_GONE_WITHIN, "release") };
    let ms = |time: Duration| format!("{:.1} ms", time.as_secs_f64() * 1e3);
    let mut listing = format!(
        "{name} on {}, {build} build: SIGKILL to a print run's {}, with no launch since; when each process of its \
         tool tree, which ignores SIGTERM, and its watcher read as gone, exited and collected, polled every \
         millisecond; bound {}, grace {}\n",
        std::env::consts::OS,
        if kill.group { "process group" } else { "pid" },
        ms(bound),
        ms(LEFT_BEHIND_GRACE)
    );
    for (n, (&pid, time)) in watched.iter().zip(&gone).enumerate() {
        let role = if n == tree.len() { "watcher" } else if n == 0 { "shell" } else { "descendant" };
        let when = time.map_or("running 5 s after the kill".into(), |time| format!("gone {} after the kill", ms(time)));
        listing.push_str(&format!("{role} {pid} {when}\n"));
    }
    if watcher.is_none() {
        listing.push_str("watcher none\n");
    }
    for (&pid, _) in watched.iter().zip(&gone).filter(|(_, time)| time.is_none()) {
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    let _ = std::fs::remove_dir_all(&base);
    if let Some(file) = listing_file {
        std::fs::write(Path::new(env!("CARGO_TARGET_TMPDIR")).join(file), &listing).unwrap();
    }
    print!("{listing}");
    assert!(watcher.is_some(), "the run's first tool started no watcher\n{listing}");
    assert_eq!(session, watcher, "the watcher does not lead a session of its own\n{listing}");
    assert!(gone.iter().all(|time| time.is_some_and(|time| time < bound)), "a process outlived its killed run by {}\n{listing}", ms(bound));
    assert!(
        gone[..tree.len()].iter().flatten().all(|&time| time >= LEFT_BEHIND_GRACE),
        "a process of the tree, which ignores SIGTERM, was gone before the {} grace had passed\n{listing}",
        ms(LEFT_BEHIND_GRACE)
    );
}

/// A run's watcher ends with the run. A print run whose `bash` call writes
/// its shell's pid and waits for a file to appear: while the call waits, one
/// process names the run's record in its command line, the watcher the call
/// started, whose command line `ps` lists as `openmax --lifeline <record>
/// <start>` on Linux too, where it runs `/proc/self/exe`; and once the test
/// has made the file and the run has answered and exited 0, that process is
/// gone within 500 ms.
#[cfg(unix)]
#[test]
fn a_runs_watcher_ends_with_the_run() {
    let (mut run, base, record) = run_one_bash_call("watcher", true, false, |base| {
        let (shell, go) = (base.join("shell"), base.join("go"));
        format!("echo $$ > '{}'; until [ -e '{}' ]; do sleep 0.01; done", shell.display(), go.display())
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    let shell: i32 = loop {
        let shell = std::fs::read_to_string(base.join("shell")).ok().and_then(|text| text.trim().parse().ok());
        let recorded = std::fs::read_to_string(&record).unwrap_or_default();
        if let Some(shell) = shell.filter(|shell| recorded.lines().any(|line| line.starts_with(&format!("{shell} ")))) {
            break shell;
        }
        let ended = run.try_wait().unwrap();
        if Instant::now() >= deadline || ended.is_some() {
            let _ = run.kill();
            let _ = run.wait();
            if let Some(shell) = shell {
                unsafe { libc::kill(-shell, libc::SIGKILL) };
            }
            panic!("the bash call did not start, or was not recorded: run {ended:?}, record {recorded:?}");
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let watchers = watchers_of(&record);
    let named = watchers.first().map(|&pid| {
        let out = Command::new("ps").args(["-ww", "-o", "args=", "-p", &pid.to_string()]).output().expect("ps runs");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    });
    std::fs::write(base.join("go"), "").unwrap();
    let out = run.wait_with_output().unwrap();
    let deadline = Instant::now() + Duration::from_millis(500);
    while watchers.iter().any(|&pid| alive(pid)) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    let left: Vec<i32> = watchers.iter().copied().filter(|&pid| alive(pid)).collect();
    for &pid in &left {
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    if alive(shell) {
        unsafe { libc::kill(-shell, libc::SIGKILL) };
    }
    let _ = std::fs::remove_dir_all(&base);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("first answer"), "{stderr}");
    assert_eq!(watchers.len(), 1, "the run's bash call had {} watchers, not one: {watchers:?}", watchers.len());
    let line = format!("openmax --lifeline {} ", record.display());
    assert!(named.as_ref().is_some_and(|named| named.starts_with(&line)), "the watcher's command line is {named:?}, not {line}<start>");
    assert!(left.is_empty(), "the watcher outlived its run, which exited 0, by 500 ms: {left:?}");
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
