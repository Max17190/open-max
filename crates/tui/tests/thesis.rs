//! The thesis gates, each named for what it asserts: one whose name states a
//! bound prints a `MEASURE` line keyed by its name that gives that bound
//! (every_thesis_gate_named_for_a_bound_prints_its_measurement_and_bound).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The lines of one job in a workflow under .github/workflows, from its key up
/// to the first line after it at the jobs' indentation or less that is not
/// blank: the next job's key, or the comment above that job, which is not
/// this one's.
fn workflow_job(workflow: &str, name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(repo().join(".github/workflows").join(workflow)).unwrap();
    let key = format!("  {name}:");
    let mut lines = text.lines().skip_while(|line| line.trim_end() != key);
    assert!(lines.next().is_some(), "{workflow} has no `{name}` job");
    lines
        .take_while(|line| {
            let body = line.trim_start();
            body.is_empty() || line.len() - body.len() > 2
        })
        .map(str::to_string)
        .collect()
}

/// The lines of one job in ci.yml.
fn ci_job(name: &str) -> Vec<String> {
    workflow_job("ci.yml", name)
}

/// The test job's step that runs every test of the workspace, under nextest.
const NEXTEST_RUN: &str = "cargo nextest run --workspace --locked --profile ci";
/// The test job's step that runs the doc tests, which nextest does not run.
const DOC_TESTS: &str = "cargo test --workspace --locked --doc";

/// The events a workflow under .github/workflows runs on: each key of its
/// `on:` block, in order, with the lines nested under it (its filters),
/// blank lines and comments left out.
fn triggers(workflow: &str) -> Vec<(String, Vec<String>)> {
    let text = std::fs::read_to_string(repo().join(".github/workflows").join(workflow)).unwrap();
    let mut events: Vec<(String, Vec<String>)> = Vec::new();
    let mut block = text.lines().skip_while(|line| *line != "on:");
    assert!(block.next().is_some(), "{workflow} has no `on:` block");
    for line in block {
        let body = line.trim_start();
        if body.is_empty() || body.starts_with('#') {
            continue;
        }
        match line.len() - body.len() {
            0 => break,
            2 => events.push((body.trim_end_matches(':').to_string(), Vec::new())),
            _ => events.last_mut().expect("a filter outside every event").1.push(body.to_string()),
        }
    }
    events
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

/// `n` with a comma between each group of three digits, as the docs write
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

/// The suite fails over its time budget: 5 seconds after compilation, the
/// target, times the multiplier of the machine it runs on. No test inside
/// the suite can time the suite, so CI does: its Suite time job builds the
/// suite and runs scripts/check-suite-time.sh on it with ubuntu-latest's
/// multiplier, which the script explains, in a step with a timeout, and
/// fails the pull request over that budget. It is a job of its own, so no
/// step of the test job runs on one of its runners only, and it restores the
/// test job's cache, saving none. Neither it nor the report gives cargo
/// another profile than Cargo.toml's, so each times the build `cargo test`
/// makes. macos-latest does not meet its budget, so
/// the Reports workflow runs the script there every night instead, five
/// times after one build with that runner's multiplier, and reports what it
/// prints, failing only when the five runs did not finish. This test holds
/// that wiring, and holds the gate to
/// its job by driving it with a stand-in `cargo` that lists stand-in test
/// binaries: it runs them at once (four of them each wait for the other
/// three to start), each the way `cargo test` would (from its package
/// directory, with Ctrl+C not ignored), passes a suite inside 5 seconds
/// times the multiplier, and fails one over it or with a binary that fails
/// again when run alone (or that a signal ends, SIGALRM included), naming
/// the binary and showing its failures, and passes one whose binary failed
/// only beside the others. A test the script runs alone, as it times its
/// own work, is left out of its binary's run beside the others and then runs
/// alone once they have ended, on the clock (a run names it when it is
/// the slowest), and fails the gate when it fails there, with no second
/// run; each such test is one the suite holds.
/// A binary still running at three times the budget
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
fn the_suite_fails_over_its_time_budget() {
    // The gate: a job of its own on one runner, which builds the suite and
    // then times it under a timeout, and restores the test job's cache.
    let job = ci_job("suite-time");
    let at = |job: &[String], line: &str| job.iter().position(|l| l.trim() == line);
    assert!(at(&job, "runs-on: ubuntu-latest").is_some(), "the suite time gate does not run on ubuntu-latest");
    let (built, gate) =
        (at(&job, "run: cargo test --workspace --locked --no-run"), at(&job, "run: scripts/check-suite-time.sh 3"));
    let (Some(built), Some(gate)) = (built, gate) else {
        panic!("the suite time job does not build the suite and then check it with a multiplier of 3: {job:?}")
    };
    assert!(built < gate, "the suite time gate builds under its own timeout");
    assert!(
        step_keys(&job, gate).iter().any(|key| key.starts_with("timeout-minutes:")),
        "a test that hangs beside the others holds the suite time gate until the job's own limit"
    );
    for line in ["shared-key: test", "save-if: false"] {
        assert!(at(&job, line).is_some(), "the suite time job does not restore the test job's cache without saving one: {line}");
    }
    let test_job = ci_job("test");
    assert!(
        !test_job.iter().any(|line| line.contains("check-suite-time")),
        "the test job times the suite on every runner of its matrix, or on one of them only"
    );
    // The report: macos-latest, five runs, its own multiplier, run as outside
    // GitHub Actions so a run over budget is a line and not an error, and a
    // failure only when the five runs did not finish.
    let report = workflow_job("reports.yml", "suite-time");
    assert!(at(&report, "runs-on: macos-latest").is_some(), "the suite time report does not run on macos-latest");
    assert!(at(&report, "run: cargo test --workspace --locked --no-run").is_some(), "the report builds under its timeout");
    for line in ["shared-key: test", "save-if: false"] {
        assert!(at(&report, line).is_some(), "the suite time report does not restore the test job's cache without saving one: {line}");
    }
    // The gate and the report time the build `cargo test` makes from
    // Cargo.toml: neither job gives cargo another profile, by a variable, a
    // flag or a config file, as the test job does to build unoptimized.
    for (name, lines) in [("gate", &job), ("report", &report)] {
        let overrides: Vec<&str> = lines
            .iter()
            .map(|line| line.trim())
            .filter(|line| !line.starts_with('#'))
            .filter(|line| *line == "env:" || ["CARGO_", "RUSTFLAGS", "GITHUB_ENV", "--config", "config.toml"].iter().any(|key| line.contains(key)))
            .collect();
        assert!(overrides.is_empty(), "the suite time {name} can build another profile than Cargo.toml's: {overrides:?}");
    }
    let run = at(&report, "- name: Report the suite time").expect("the Reports workflow has no suite time report");
    assert!(
        step_keys(&report, run).iter().any(|key| key.starts_with("timeout-minutes:")),
        "a test that hangs holds the suite time report until the job's own limit"
    );
    let commands: Vec<&str> =
        report[run..].iter().map(|line| line.trim()).skip_while(|line| *line != "run: |").skip(1).take_while(|line| !line.is_empty()).collect();
    assert_eq!(
        commands,
        [
            "env -u GITHUB_ACTIONS scripts/check-suite-time.sh --runs 5 4 2>&1 | tee \"$RUNNER_TEMP/suite-time.log\"",
            "status=${PIPESTATUS[0]}",
            "{ echo \"Reported, not gated, on macos-latest; the check exited $status:\"; echo '```'; sed -n -E '/^(run [0-9]+ of 5: |median )/p' \"$RUNNER_TEMP/suite-time.log\"; echo '```'; } >> \"$GITHUB_STEP_SUMMARY\"",
            "grep -q '^median .* over 5 runs,' \"$RUNNER_TEMP/suite-time.log\" || { echo \"::error::the suite did not run 5 times, so it measured nothing\"; exit 1; }",
            "if grep -E 'FAILED \\(stopped at |test binaries did not report' \"$RUNNER_TEMP/suite-time.log\"; then echo \"::error::a run did not finish, so its time is not the suite's\"; exit 1; fi",
        ],
        "the suite time report does not run the suite five times on macos-latest, report each run and the median, \
         and fail only when the runs did not finish"
    );
    // Each runner's multiplier, the gate's and the report's, has its reason
    // in the script: a multiplier raised without one passes a pull request
    // that slowed the suite on that runner.
    let script = std::fs::read_to_string(repo().join("scripts/check-suite-time.sh")).unwrap();
    for (multiplier, runner) in [("3", "ubuntu-latest"), ("4", "macos-latest")] {
        assert!(
            script.contains(&format!("a multiplier of {multiplier} on {runner}")),
            "the script does not give the reason for the multiplier of {multiplier} on {runner}"
        );
    }
    // The tests the script runs alone are tests of the suite under those
    // names: one renamed would run beside the others again and time the
    // runner.
    let named = script.lines().find_map(|line| line.strip_prefix("alone_tests=\"")?.strip_suffix('"')).unwrap_or("");
    assert!(!named.is_empty(), "the script names no test that runs alone");
    let sources: String = ["crates/core/tests", "crates/tui/tests"]
        .iter()
        .flat_map(|tests| std::fs::read_dir(repo().join(tests)).unwrap())
        .map(|entry| std::fs::read_to_string(entry.unwrap().path()).unwrap_or_default())
        .collect();
    for test in named.split(' ') {
        assert!(sources.contains(&format!("fn {test}()")), "the script runs {test} alone, which no test is named");
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
    // A binary that holds a test the gate runs alone, and a binary beside it
    // that marks when it ended: beside the others the first must be run
    // without that test, then alone with only it, with libtest's own pool,
    // once every other binary has ended, once a run. Alone, its test takes a
    // second the first time, four fifths of a second longer than the binary
    // beside it (room for either to start late on a busy runner and still
    // keep their order), and no time after that. It ends with `code`.
    let held_alone = |name: &str, code: u8| {
        let (path, ended, runs) = (dir.join(name), dir.join(format!("{name}-lagging-ended")), dir.join(format!("{name}-alone-runs")));
        executable(
            &path,
            &format!(
                "#!/bin/sh\n[ \"$1\" = --list ] && {{ printf 'beside: test\\nharness_time_per_tool_round: test\\n'; exit 0; }}\n\
                 case \"$*\" in\n\
                 \"--test-threads \"*\" --exact --skip harness_time_per_tool_round\") exit 0 ;;\n\
                 \"--exact harness_time_per_tool_round\") echo run >> '{runs}' ;;\n\
                 *) echo \"run with $*\"; exit 1 ;;\n\
                 esac\n\
                 [ -e '{ended}' ] || {{ echo 'run alone beside another binary'; exit 1; }}\n\
                 [ \"$(wc -l < '{runs}')\" -gt 1 ] || sleep 1\n\
                 [ {code} = 0 ] || printf 'failures:\\n\\n---- harness_time_per_tool_round stdout ----\\npast its bound\\n'\n\
                 exit {code}\n",
                runs = runs.display(),
                ended = ended.display(),
            ),
        );
        let lagging = dir.join(format!("{name}-lagging"));
        executable(&lagging, &format!("#!/bin/sh\n[ \"$1\" = --list ] && exit 0\nrm -f '{0}'\nsleep 0.2\n: > '{0}'\n", ended.display()));
        (path, lagging, runs)
    };
    let (timed, timed_lagging, timed_runs) = held_alone("timed", 0);
    let (slow_alone, slow_lagging, slow_runs) = held_alone("slow-alone", 101);
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
        (alone, alone_report),
        (alone_failed, alone_failed_report),
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
        // A tenth of the target is half a second, so the bound is a second
        // and a half: room for the hanging binary to start what it leaves
        // running on a busy runner, beside a dozen other gates, before the
        // bound stops it, while its `sleep 5` still outlasts the bound.
        let stopped = scope.spawn(|| run_gate("stopped", &["0.1"], &[("hanging", Some(hanging.as_path()))], None));
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
        let alone = scope.spawn(|| {
            let binaries = [("timed", Some(timed.as_path())), ("lagging", Some(timed_lagging.as_path()))];
            run_gate("alone", &["--runs", "2", "1"], &binaries, None)
        });
        let alone_failed = scope.spawn(|| {
            let binaries = [("slow-alone", Some(slow_alone.as_path())), ("lagging", Some(slow_lagging.as_path()))];
            run_gate("alone-failed", &["1"], &binaries, None)
        });
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
            alone.join().unwrap(),
            alone_failed.join().unwrap(),
        )
    });
    let ran_alone = |runs: &Path| std::fs::read_to_string(runs).map_or(0, |text| text.lines().count());
    let (timed_ran, slow_ran) = (ran_alone(&timed_runs), ran_alone(&slow_runs));
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
        .split_once("hanging (test) failed beside the other binaries and passed alone:\nstill running at 1.50s, when it was stopped:\n")
        .map_or(Vec::new(), |(_, rest)| rest.lines().take_while(|line| !line.is_empty() && !line.contains(' ')).collect());
    assert!(
        stopped_report.contains("hanging (test) FAILED (stopped at 1.50s)")
            && unfinished == ["hangs"]
            && stopped_report.contains("over its 0.50s budget"),
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
    // A test that times its own work runs alone: beside the others its
    // binary runs without it, then it runs alone, once every other binary
    // has ended, in every run, and its time counts toward the run's, which
    // names it when it is the slowest. Alone, its failure fails the gate,
    // with no second run.
    let first_run = alone_report
        .lines()
        .find_map(|line| line.strip_prefix("run 1 of 2: ")?.split_once("s, slowest binary "))
        .and_then(|(took, slowest)| Some((took.parse::<f64>().ok()?, slowest.split_once(", load ")?.0)));
    assert!(
        alone == Some(0)
            && timed_ran == 2
            && alone_report.lines().any(|line| line.starts_with("run 2 of 2: "))
            && first_run.is_some_and(|(took, slowest)| {
                took >= 1.2 && slowest.ends_with("s harness_time_per_tool_round in timed (test), alone")
            }),
        "a test that runs alone did not run once a run, alone, after the others and on the clock, \
         or the run did not name it as its slowest: {alone_report}"
    );
    assert!(
        alone_failed == Some(1)
            && alone_failed_report.contains("harness_time_per_tool_round in slow-alone (test), alone FAILED (exit 101)")
            && alone_failed_report.contains(
                "harness_time_per_tool_round in slow-alone (test), alone failed:\nfailures:\n\n---- harness_time_per_tool_round stdout ----\npast its bound"
            )
            && !alone_failed_report.contains("passed alone")
            && slow_ran == 1,
        "a test that fails alone passes the gate, or runs a second time: {alone_failed_report}"
    );
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

/// The core, production Rust without its tests, stays under its line
/// ratchet, against a target of about 37,000 lines.
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
/// other files (agent_loop_stays_under_its_line_ratchet) included.
/// docs/development.md states the ratchet and CEILING, held here. A count, so every
/// build asserts it.
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
    // Several replacements per edit_file call, and an excerpt of the
    // changed lines in its result, so an edit takes one round, not one per
    // replacement plus a read-back.
    const BASELINE: usize = 44_259;
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
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
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
        assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which this test holds");
    }
    Ok(())
}

/// The agent loop, crates/core/src/agent.rs without its tests, stays under
/// its line ratchet, against a target of 2,000 lines.
/// scripts/count-core-lines.sh --files gives each file's
/// core-excluding-tests by the method core_source_stays_under_its_line_budget
/// holds to its fixture, so the loop is measured as the core is, not by a
/// second counter. This prints the target beside the loop's two counts and
/// holds a ratchet by the same rule: the loop's
/// core-excluding-tests at least BASELINE, the count this test last
/// measured, and at most RATCHET, BASELINE plus MARGIN. A change that needs
/// more raises BASELINE to its new count in its own diff and says why; one
/// that cuts the loop must lower BASELINE to its new count, so the lines it
/// cuts stay cut. Lines the loop hands to another file of the core stay
/// under the core's ratchet. docs/development.md states this ratchet, held here.
#[test]
fn agent_loop_stays_under_its_line_ratchet() -> Result<(), String> {
    const TARGET: usize = 2_000;
    const BASELINE: usize = 1_715;
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
    println!(
        "MEASURE agent_loop_stays_under_its_line_ratchet {lines} lines excluding tests (ratchet {RATCHET}, target {TARGET}), \
         {total} including tests"
    );
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
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    let claim = format!(
        "The agent loop's ratchet is {} today, {} over the baseline measurement of {} lines, against a target of {}",
        thousands(RATCHET as u64),
        thousands(MARGIN as u64),
        thousands(BASELINE as u64),
        thousands(TARGET as u64)
    );
    assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which this test holds");
    Ok(())
}

/// The thesis gates run in release on both runners: every thesis test runs
/// in CI's thesis job in release and fails the pull request, but for the
/// ones a runner reports instead. A
/// test that times the harness asserts its bound only in a release build,
/// and the test job builds in debug, so ci.yml's `thesis` job builds in
/// release, on each of the test job's runners, every test target named
/// thesis* in the workspace (cargo matches the glob against target names,
/// and a file under a crate's tests/ is a target named after its stem), the
/// binary's `thesis_frames` module where the runner gates one of its tests,
/// and the first-frame test's baseline binary, an example that `cargo test`
/// leaves unbuilt once a target is selected. The module's tests run from
/// the binary's own unit-test harness, one more whole-program LTO unit, so
/// a runner whose matrix entry reports them all (`frames: false`) neither
/// builds nor runs it. Its gate step runs exactly every thesis test but the
/// ones its runner reports, and fails the pull request when one fails. The Reports
/// workflow runs the reported tests instead, every night, in release and one
/// at a time on the same runners, from a matrix that lists the same tests;
/// it copies each `MEASURE` line to its job's summary and fails only when a
/// reported test printed none, so it never runs on a pull request, whose
/// gate holds only steps that can fail it. A runner reports only
/// the tests whose targets it does not meet with room, listed in its matrix
/// entry, held here and named in docs/development.md, so moving a test between the
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
fn the_thesis_gates_run_in_release_on_both_runners() {
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
    // No step of ci.yml may fail without failing the gate:
    // pull_request_workflows_skip_no_step_and_let_none_fail holds that.
    let ci = std::fs::read_to_string(repo().join(".github/workflows/ci.yml")).unwrap();
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
    let (build, gates) = (step("Build release tests"), step("Thesis gates"));
    assert!(build < gates, "the thesis job does not build before it gates");
    assert!(
        steps[gates].iter().any(|line| line.starts_with("timeout-minutes:")),
        "a thesis test that hangs holds the gate step until the job's own limit: {:?}",
        steps[gates]
    );
    // A key of each runner's matrix entry, one value each: `reported`, what
    // the runner reports, in the thesis job, whose gate step skips them, and
    // in the Reports workflow, which runs them; and the thesis job's
    // `frames`.
    let by_runner = |job: &[String], key: &str| {
        let mut values = Vec::new();
        let mut entry = None;
        for line in job.iter().map(|line| line.trim().to_string()) {
            if let Some(runner) = line.strip_prefix("- os: ") {
                entry = Some(runner.to_string());
            } else if let Some(value) = line.strip_prefix(&format!("{key}: ")) {
                values.push((entry.take().unwrap_or_else(|| panic!("a `{key}` outside a runner's entry")), value.to_string()));
            }
        }
        values
    };
    let reported_by = |job: &[String]| by_runner(job, "reported");
    let reports_job = workflow_job("reports.yml", "thesis-reports");
    assert_eq!(runners(&reports_job), runners(&job), "the thesis reports do not run on the thesis job's runners");
    assert_eq!(
        reported_by(&reports_job),
        reported_by(&job),
        "the Reports workflow does not run exactly the thesis tests each runner's gate step skips"
    );
    let reported = reported_by(&job);
    let reported: Vec<(&str, &str)> = reported.iter().map(|(os, tests)| (os.as_str(), tests.as_str())).collect();
    assert_eq!(
        reported,
        [
            ("ubuntu-latest", "harness_time_per_tool_round"),
            ("macos-latest", "harness_time_per_tool_round transcript_frame_p99_at_100k_lines transcript_frame_p99_at_1m_lines"),
        ],
        "the thesis tests a runner reports instead of gating changed"
    );
    // The binary's thesis_frames tests run from its own unit-test harness,
    // one more whole-program LTO unit of the release build: a runner builds
    // and runs it exactly when it gates one of them, a test built there, not
    // ignored and not reported. So no runner skips one it gates, nor builds
    // a harness its gate step would run no test from.
    let frames_tests = tests_in(&std::fs::read_to_string(repo().join("crates/tui/src/app/thesis_frames.rs")).unwrap());
    assert!(!frames_tests.is_empty(), "the binary's thesis_frames module holds no test");
    let gates_frames = |os: &str, reported: &str| {
        let platform = if os.starts_with("macos") { "macos" } else { "linux" };
        frames_tests.iter().any(|(name, attributes)| {
            let cfgs = attributes.iter().filter_map(|line| line.strip_prefix("#[cfg(")?.strip_suffix(")]"));
            !reported.split(' ').any(|test| test == name.as_str())
                && !attributes.iter().any(|line| line.starts_with("#[ignore"))
                && cfgs.into_iter().all(|cfg| cfg_holds(cfg, platform))
        })
    };
    let frames: Vec<(&str, bool)> = reported.iter().map(|(os, tests)| (*os, gates_frames(os, tests))).collect();
    let listed = by_runner(&job, "frames");
    let listed: Vec<(&str, bool)> = listed
        .iter()
        .map(|(os, value)| {
            (os.as_str(), value.parse::<bool>().unwrap_or_else(|_| panic!("{os}'s `frames` is not true or false: {value}")))
        })
        .collect();
    assert_eq!(listed, frames, "a runner's `frames` is not whether it gates a test of the binary's thesis_frames module");
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    assert!(
        stated.contains(
            "`harness_time_per_tool_round` (target 250 us) on both runners and `transcript_frame_p99_at_100k_lines` (target 0.5 ms) and `transcript_frame_p99_at_1m_lines` (target 0.25 ms) on macOS"
        ),
        "docs/development.md does not name the thesis tests each runner reports instead of gating"
    );
    // A renamed test would no longer be skipped by the gate step, nor run by
    // the reports step. libtest selects and skips by substring, so a test or
    // module whose name holds a reported one would leave the gate step too.
    let sources: String = thesis_sources().into_iter().map(|(_, text)| text).collect();
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
            "if ${{ matrix.frames }}; then",
            "cargo test --release --workspace --locked --bin openmax -- --test-threads=1 --nocapture thesis_frames $skip || failed=1",
            "fi",
            "exit ${failed:-0}",
        ],
        "the gate step does not run, in release, one at a time and printing what it measured, every thesis test but the \
         ones its runner reports, each command even after one failed, and fail when one did"
    );
    // The Reports workflow's report: in release and one at a time, exactly
    // the tests its runner reports, every MEASURE line to the job's summary,
    // and a failure only for a reported test that printed none.
    let report = reports_job
        .iter()
        .position(|line| line.trim() == "- name: Report the thesis tests this runner does not gate")
        .expect("the thesis reports job has no report step");
    assert!(
        step_keys(&reports_job, report).iter().any(|key| key.starts_with("timeout-minutes:")),
        "a reported test that hangs holds its report until the job's own limit"
    );
    let report: Vec<&str> = reports_job[report..]
        .iter()
        .map(|line| line.trim())
        .skip_while(|line| *line != "run: |")
        .skip(1)
        .take_while(|line| !line.is_empty() && !line.starts_with("- "))
        .collect();
    assert_eq!(
        report,
        [
            "cargo test --release --workspace --locked --test 'thesis*' --bin openmax --no-fail-fast -- --test-threads=1 --nocapture ${{ matrix.reported }} 2>&1 | tee \"$RUNNER_TEMP/reports.log\"",
            "status=${PIPESTATUS[0]}",
            "sed -n 's/.*MEASURE /MEASURE /p' \"$RUNNER_TEMP/reports.log\" > \"$RUNNER_TEMP/measured.txt\"",
            "{ echo \"Reported, not gated, on ${{ matrix.os }}; cargo test exited $status:\"; echo '```'; cat \"$RUNNER_TEMP/measured.txt\"; echo '```'; } >> \"$GITHUB_STEP_SUMMARY\"",
            "for test in ${{ matrix.reported }}; do",
            "grep -q \"^MEASURE $test \" \"$RUNNER_TEMP/measured.txt\" || { echo \"::error::$test printed no MEASURE line, so it measured nothing\"; missing=1; }",
            "done",
            "exit ${missing:-0}",
        ],
        "the Reports workflow does not run, in release and one at a time, exactly the tests its runner reports, \
         copy their measurements to the job's summary, and fail only when one measured nothing"
    );
    let built = "run: cargo test --release --workspace --locked --no-run --test 'thesis*' --bin openmax";
    assert!(reports_job.iter().any(|line| line.trim() == built), "the thesis reports build what they run under its timeout");
    assert_eq!(
        commands(build),
        [
            "cargo test --release --workspace --locked --no-run --test 'thesis*' ${{ matrix.frames && '--bin openmax' || '' }}",
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

/// Whether `script`, a step's `run`, swallows a failure: a `||` followed by
/// `true`, `:` or `exit 0`, read as shell words, so no whitespace between
/// them, a line break included, no braces or parentheses around the rest,
/// and no `;` after them hides one; or a `set` command whose flags turn
/// `errexit` or `pipefail` off (`set +e`, `set +euo pipefail`,
/// `set +o errexit`), shell keywords and command prefixes before it
/// (`then set +e`, `builtin set +e`) skipped. A `+` word of another command
/// (`cargo +stable test`) is no such flag.
fn swallows_a_failure(script: &str) -> bool {
    let spaced = script.replace("||", " || ");
    let words: Vec<&str> = spaced
        .split_whitespace()
        .map(|word| word.trim_matches(['(', ')', '{', '}', ';']))
        .filter(|word| !word.is_empty())
        .collect();
    let turns_off = |command: &str| {
        let words: Vec<&str> = command
            .split_whitespace()
            .map(|word| word.trim_matches(['(', ')', '{', '}']))
            .filter(|word| !word.is_empty())
            .skip_while(|word| matches!(*word, "!" | "if" | "then" | "else" | "elif" | "while" | "until" | "do" | "time" | "builtin" | "command"))
            .collect();
        words.first() == Some(&"set")
            && words.iter().enumerate().any(|(at, word)| {
                let flags = word.strip_prefix('+').filter(|flags| !flags.is_empty() && flags.chars().all(|c| c.is_ascii_alphabetic()));
                flags.is_some_and(|flags| flags.contains('e') || (flags.ends_with('o') && matches!(words.get(at + 1), Some(&("pipefail" | "errexit")))))
            })
    };
    words.windows(2).any(|pair| pair[0] == "||" && matches!(pair[1], "true" | ":"))
        || words.windows(3).any(|three| three == ["||", "exit", "0"])
        || script.split(['\n', ';', '&', '|']).any(turns_off)
}

/// No step of a workflow that runs on pull requests can be skipped in a run
/// that decides the gate, or fail without failing it, by the means this
/// reads, each key read as YAML reads it, quoted or with space before its
/// colon: no `continue-on-error` anywhere; no `if:` but
/// `if: ${{ !cancelled() }}`, which runs a step after a failed one too and
/// skips it only in a run that was cancelled; no script that swallows a
/// failure (swallows_a_failure); and every script runs under bash with
/// `pipefail`, the shell the workflow's `defaults` names and no job or step
/// replaces, so no pipe loses the failure of a command in it, wherever the
/// command stands. A measurement no runner meets with room leaves the pull
/// request's gate for the Reports workflow, which runs every night and by
/// hand, on no pull request and no push, and fails only when it cannot
/// measure. Of the workflows GitHub runs, `.yml` and `.yaml` files, ci.yml is
/// the one that runs on pull requests.
#[test]
fn pull_request_workflows_skip_no_step_and_let_none_fail() {
    for (script, swallows) in [
        ("cargo test || true", true),
        ("cargo test ||true", true),
        ("cargo test ||  true", true),
        ("cargo test ||\n  true", true),
        ("cargo test || :", true),
        ("cargo test || exit 0; echo done", true),
        ("cargo test || { true; }", true),
        ("cargo test || failed=1", false),
        ("cargo test || exit 1", false),
        ("x=\"${a:-b}\" || exit", false),
        ("set +o pipefail\ncargo test | tee log", true),
        ("set +e; cargo test", true),
        ("set +euo pipefail", true),
        ("set +o errexit", true),
        ("set -euo pipefail\nchmod +x scripts/a.sh", false),
        ("cargo +stable test", false),
        ("cargo +nightly test || exit 1", false),
        ("cargo build && set +e", true),
        ("{ set +o pipefail; }", true),
        ("if true; then set +e; fi", true),
        ("while read -r line; do builtin set +o pipefail; done", true),
        ("if cargo test; then echo set +e; fi", false),
    ] {
        assert_eq!(swallows_a_failure(script), swallows, "whether {script:?} swallows a failure");
    }
    let mut workflows: Vec<String> = std::fs::read_dir(repo().join(".github/workflows"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".yml") || name.ends_with(".yaml"))
        .collect();
    workflows.sort();
    let gating: Vec<&String> =
        workflows.iter().filter(|workflow| triggers(workflow).iter().any(|(event, _)| event.starts_with("pull_request"))).collect();
    assert_eq!(gating, ["ci.yml"], "the workflows that run on pull requests changed");
    let mut soft = Vec::new();
    for workflow in gating {
        let text = std::fs::read_to_string(repo().join(".github/workflows").join(workflow)).unwrap();
        if !text.contains("\ndefaults:\n  run:\n    shell: bash\n") {
            soft.push(format!("{workflow} names no `defaults: run: shell: bash`, so its pipes run without pipefail"));
        }
        let lines: Vec<String> = text.lines().map(str::to_string).collect();
        let unquoted = |text: &str| text.trim().trim_matches(['"', '\'']).to_string();
        for (at, line) in lines.iter().enumerate() {
            let body = line.trim().trim_start_matches("- ");
            if body.starts_with('#') {
                continue;
            }
            let (key, value) = body.split_once(':').map_or((String::new(), String::new()), |(key, value)| (unquoted(key), unquoted(value)));
            if key == "continue-on-error" || (key == "if" && value != "${{ !cancelled() }}") || (key == "shell" && value != "bash") {
                soft.push(format!("{workflow} line {}: {body}", at + 1));
            }
            if key != "run" {
                continue;
            }
            // The script: the value with every line under it indented past
            // the key, which continue a plain value as they make a `|` or
            // `>` value's block.
            let indent = line.len() - line.trim_start().len() + if line.trim_start().starts_with("- ") { 2 } else { 0 };
            let block = lines[at + 1..].iter().take_while(|next| next.trim().is_empty() || next.len() - next.trim_start().len() > indent);
            let script = std::iter::once(value.as_str()).chain(block.map(String::as_str)).collect::<Vec<_>>().join("\n");
            if swallows_a_failure(&script) {
                soft.push(format!("{workflow} line {}: the step's script swallows a failure", at + 1));
            }
        }
    }
    assert!(
        soft.is_empty(),
        "a step or job of the pull request's gate can be skipped, or fail without failing it:\n{}",
        soft.join("\n")
    );
    let reports: Vec<String> = triggers("reports.yml").into_iter().map(|(event, _)| event).collect();
    assert_eq!(reports, ["schedule", "workflow_dispatch"], "the Reports workflow runs on other events than these");
}

/// The thesis sources, each with its path: every test target named thesis*
/// under a crate's tests/, and the binary's `thesis_frames` module.
fn thesis_sources() -> Vec<(String, String)> {
    let mut paths = vec!["crates/tui/src/app/thesis_frames.rs".to_string()];
    for tests in ["crates/core/tests", "crates/tui/tests"] {
        let mut found: Vec<String> = std::fs::read_dir(repo().join(tests))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with("thesis") && name.ends_with(".rs"))
            .map(|name| format!("{tests}/{name}"))
            .collect();
        found.sort();
        paths.extend(found);
    }
    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(repo().join(&path)).unwrap();
            (path, text)
        })
        .collect()
}

/// Each test in `source`: its name, and its attribute lines, from the first
/// one above its `#[test]` or `#[tokio::test]` to its `fn` line.
fn tests_in(source: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = source.lines().collect();
    let attribute = |line: &&str| line.trim_start().starts_with("#[");
    let mut tests = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        let line = line.trim_start();
        if line != "#[test]" && !line.starts_with("#[tokio::test") {
            continue;
        }
        let first = lines[..at].iter().rposition(|line| !attribute(line)).map_or(0, |above| above + 1);
        let item = at + lines[at..].iter().position(|line| !attribute(line)).expect("a test attribute on no item");
        let name = lines[item]
            .trim_start()
            .trim_start_matches("async ")
            .strip_prefix("fn ")
            .and_then(|rest| rest.split('(').next())
            .unwrap_or_else(|| panic!("a test that is not a function: {}", lines[item]));
        tests.push((name.to_string(), lines[first..item].iter().map(|line| line.trim().to_string()).collect()));
    }
    tests
}

/// The function `name` in `source`, from its `fn` line to the line that
/// closes it at that line's indentation.
fn function_in(source: &str, name: &str) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    let at = lines.iter().position(|line| {
        let item = line.trim_start().trim_start_matches("pub ").trim_start_matches("async ");
        item.strip_prefix("fn ").and_then(|rest| rest.strip_prefix(name)).is_some_and(|rest| rest.starts_with(['(', '<']))
    })?;
    let close = format!("{}}}", &lines[at][..lines[at].len() - lines[at].trim_start().len()]);
    let end = lines[at..].iter().position(|line| *line == close)?;
    Some(lines[at..=at + end].join("\n"))
}

/// Where `needle` first stands in `code` as code: a `//` before it on its
/// own line comments it out, so a print commented out prints nothing. A
/// block comment is not read.
fn code_find(code: &str, needle: &str) -> Option<usize> {
    let commented = |at: usize| code[code[..at].rfind('\n').map_or(0, |n| n + 1)..at].contains("//");
    code.match_indices(needle).map(|(at, _)| at).find(|&at| !commented(at))
}

/// The string literal whose opening quote is at byte `open` of `code`, as
/// rustc reads it: a `\` that ends a line is taken out with the whitespace
/// that starts the next, and every other escape is kept as written.
fn literal_at(code: &str, open: usize) -> String {
    let mut text = String::new();
    let mut chars = code[open + 1..].chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' if chars.peek() == Some(&'\n') => {
                while chars.peek().is_some_and(|c| c.is_whitespace()) {
                    chars.next();
                }
            }
            '\\' => {
                text.push(c);
                text.extend(chars.next());
            }
            c => text.push(c),
        }
    }
    text
}

/// The number that starts at `words[at]` of a test's name, as the count of
/// words it takes and its value: in digits, in words (`two_thousand`,
/// `ninety_five`), or `a` before a unit (`a_second`, 1) or a fraction
/// (`a_fifth`, the 20 percent it is).
fn number_at(words: &[&str], at: usize) -> Option<(usize, u64)> {
    const ONES: [&str; 20] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ];
    const TENS: [&str; 8] = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    const UNITS: [&str; 14] = [
        "second", "minute", "hour", "millisecond", "microsecond", "byte", "kilobyte", "megabyte", "gigabyte", "line",
        "token", "test", "commit", "request",
    ];
    const FRACTIONS: [(&str, u64); 5] = [("half", 50), ("third", 33), ("quarter", 25), ("fifth", 20), ("tenth", 10)];
    let first = *words.get(at)?;
    if let Ok(value) = first.parse::<u64>() {
        return Some((1, value));
    }
    if first == "a" {
        let next = *words.get(at + 1)?;
        let fraction = FRACTIONS.iter().find(|(word, _)| *word == next).map(|(_, percent)| (2, *percent));
        return fraction.or(UNITS.contains(&next).then_some((2, 1)));
    }
    let (mut taken, mut total, mut group) = (0, 0, 0);
    while let Some(word) = words.get(at + taken) {
        if let Some(n) = ONES.iter().position(|one| one == word) {
            group += n as u64;
        } else if let Some(n) = TENS.iter().position(|ten| ten == word) {
            group += 20 + 10 * n as u64;
        } else if *word == "hundred" && taken > 0 {
            group *= 100;
        } else if (*word == "thousand" || *word == "million") && taken > 0 {
            total += group * if *word == "thousand" { 1_000 } else { 1_000_000 };
            group = 0;
        } else {
            break;
        }
        taken += 1;
    }
    (taken > 0).then_some((taken, total + group))
}

/// The bound a test's name states, as the number its MEASURE line must
/// give: a number (number_at) after a comparator (`under`, `over`, `below`,
/// `above`, `within`, `at_least`, `at_most`, `fewer_than`, `less_than`,
/// `more_than`, `has`, `in`), a number before `percent`, or 0 for `zero`.
fn stated_bound(name: &str) -> Option<u64> {
    let words: Vec<&str> = name.split('_').collect();
    for at in 0..words.len() {
        if words[at] == "zero" {
            return Some(0);
        }
        let pair = words.get(at + 1).map(|next| format!("{}_{next}", words[at]));
        let comparator = if ["under", "over", "below", "above", "within", "has", "in"].contains(&words[at]) {
            Some(1)
        } else if pair.is_some_and(|pair| ["at_least", "at_most", "fewer_than", "less_than", "more_than"].contains(&pair.as_str())) {
            Some(2)
        } else {
            None
        };
        if let Some((_, bound)) = comparator.and_then(|words_taken| number_at(&words, at + words_taken)) {
            return Some(bound);
        }
        if words[at] == "percent" {
            let before = (0..at).find_map(|from| number_at(&words, from).filter(|(taken, _)| from + taken == at));
            if let Some((_, bound)) = before {
                return Some(bound);
            }
        }
    }
    None
}

/// Whether the predicate of a `#[cfg(...)]`, `cfg`, holds in the build CI's
/// thesis job makes on `os`, `linux` or `macos`: `unix` and `test` hold,
/// `target_os = "..."` holds for its own os, `debug_assertions` does not
/// hold in a release build, and `not`, `all` and `any` combine as rustc's
/// do. Any other predicate (a feature, an architecture) is taken not to
/// hold, so a cfg this cannot read counts as keeping a test off.
fn cfg_holds(cfg: &str, os: &str) -> bool {
    let cfg = cfg.trim();
    let Some((combinator, inner)) = cfg.strip_suffix(')').and_then(|cfg| cfg.split_once('(')) else {
        let (key, value) = cfg.split_once('=').map_or((cfg, ""), |(key, value)| (key.trim(), value.trim().trim_matches('"')));
        return matches!(key, "unix" | "test") || (key == "target_os" && value == os);
    };
    let (mut parts, mut depth, mut from) = (Vec::new(), 0, 0);
    for (at, c) in inner.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&inner[from..at]);
                from = at + 1;
            }
            _ => {}
        }
    }
    parts.push(&inner[from..]);
    let mut parts = parts.into_iter().filter(|part| !part.trim().is_empty()).map(|part| cfg_holds(part, os));
    match combinator.trim() {
        "not" => !parts.next().unwrap_or(false),
        "all" => parts.all(|holds| holds),
        "any" => parts.any(|holds| holds),
        _ => false,
    }
}

/// Whether a test with these attribute lines is built on one of CI's
/// runners at least, in a release build: rustc builds an item only where
/// every `#[cfg(...)]` on it holds, so all of them must hold on the same
/// runner (cfg_holds), and two that each hold on another keep it off both.
fn built_on_a_runner(attributes: &[impl AsRef<str>]) -> bool {
    let cfgs: Vec<&str> =
        attributes.iter().filter_map(|attribute| attribute.as_ref().strip_prefix("#[cfg(")?.strip_suffix(")]")).collect();
    ["linux", "macos"].iter().any(|os| cfgs.iter().all(|cfg| cfg_holds(cfg, os)))
}

/// Every thesis test whose name states a bound prints what it measured
/// beside that bound, and CI gates it. A name states a bound with a number
/// after a comparator (`_under_700_`, `_at_least_ninety_five_`,
/// `_in_five_seconds`, `_has_two_thousand_`), a number before `_percent`, or
/// `zero_`, in digits or in words (stated_bound gives the grammar). Each
/// such test prints, with `println!` or `eprintln!`, which CI's thesis job
/// shows, a `MEASURE` line keyed by its own name, from its body or from the
/// function it passes its name to first, which keys the line by that
/// parameter; a line or call that a `//` comments out does not count
/// (code_find). The line's first number is formatted from a local of the
/// test's own, named inline (`{orphans}`, not `{}` or `{0}`), neither
/// written in the literal nor a constant of its code or its file, and the
/// line gives the bound the name states as a number of
/// its own (not the 10 of `p10`), in its literal or through a constant it
/// formats, read from the function that prints the line, then the test,
/// then the file, so another test's constant of the same name does not
/// answer for it. And CI's thesis job gates the test: no runner reports it
/// instead, it is not ignored, and its cfg attributes all hold on one of
/// CI's runners in a release build (built_on_a_runner). What the test
/// asserts against the bound is its own, and so is whether the value it
/// prints is a measurement of what its name states rather than a value
/// read from elsewhere: no reading of its source settles that, and its red
/// record shows it failing past the bound. This holds what a test's name,
/// its line and CI say to each other, so a test named for a bound that prints
/// nothing keyed by its name, as `mutation_score_is_at_least_ninety_five_percent`
/// did, or one that prints the bound where a measurement belongs, fails
/// here, and so does a test that takes back one of the seven names retired
/// for claiming a bound their tests never asserted, whatever its line says.
#[test]
fn every_thesis_gate_named_for_a_bound_prints_its_measurement_and_bound() {
    for (cfg, linux, macos) in [
        ("unix", true, true),
        ("not(unix)", false, false),
        ("windows", false, false),
        ("target_os = \"linux\"", true, false),
        ("not(target_os = \"macos\")", true, false),
        ("any(target_os = \"linux\", target_os = \"macos\")", true, true),
        ("all(unix, not(debug_assertions))", true, true),
        ("all(unix, feature = \"slow\")", false, false),
    ] {
        assert_eq!((cfg_holds(cfg, "linux"), cfg_holds(cfg, "macos")), (linux, macos), "where #[cfg({cfg})] holds");
    }
    for (attributes, built) in [
        (&["#[test]"][..], true),
        (&["#[cfg(unix)]", "#[cfg(target_os = \"linux\")]", "#[test]"][..], true),
        (&["#[cfg(target_os = \"linux\")]", "#[cfg(target_os = \"macos\")]", "#[test]"][..], false),
    ] {
        assert_eq!(built_on_a_runner(attributes), built, "whether a test under {attributes:?} is built on one of CI's runners");
    }
    for (code, at) in [("// println!(\"M\");\n    println!(\"M\");", Some(31)), ("let a = 1; // println!(\"M\");", None)] {
        assert_eq!(code_find(code, "\"M\""), at, "where {code:?} prints \"M\" as code");
    }
    // The names seven gates had while they claimed a bound they never
    // asserted, each with that bound. A gate that took one back would claim
    // it again whatever its line said, so none may.
    let retired = [
        ("mutation_score_is_at_least_ninety_five_percent", 95),
        ("the_workspace_has_two_thousand_tests", 2_000),
        ("whole_suite_runs_in_five_seconds", 5),
        ("at_most_a_fifth_of_commits_touch_the_core", 20),
        ("agent_loop_is_under_2500_lines", 2_500),
        ("headless_turn_peak_rss_is_under_four_megabytes", 4),
        ("one_command_from_a_fresh_install_answers_in_under_thirty_seconds", 30),
    ];
    // The grammar, on those names and names the thesis sources have.
    for (name, bound) in retired.iter().map(|&(name, bound)| (name, Some(bound))).chain([
        ("one_command_from_a_fresh_install_answers_locally_in_under_a_second", Some(1)),
        ("frozen_prefix_is_under_700_tokens", Some(700)),
        ("a_session_prompt_is_75_percent_cacheable", Some(75)),
        ("zero_orphans_after_kill_9", Some(0)),
        ("every_test_runs_under_a_checksummed_nextest_that_retries_none", None),
        ("every_request_in_a_session_carries_the_same_prefix_bytes", None),
        ("transcript_frame_p99_at_100k_lines", None),
        ("core_source_stays_under_its_line_budget", None),
        ("the_thesis_gates_run_in_release_on_both_runners", None),
    ]) {
        assert_eq!(stated_bound(name), bound, "the bound `{name}` states");
    }
    let reported: Vec<String> = ci_job("thesis")
        .iter()
        .filter_map(|line| line.trim().strip_prefix("reported: ").map(str::to_string))
        .flat_map(|tests| tests.split(' ').map(str::to_string).collect::<Vec<_>>())
        .collect();
    assert!(!reported.is_empty(), "no runner reports a thesis test, so this holds nothing about the reported ones");
    let (mut bounded, mut problems) = (Vec::new(), Vec::new());
    for (path, source) in thesis_sources() {
        for (name, attributes) in tests_in(&source) {
            let Some(bound) = stated_bound(&name) else { continue };
            bounded.push(format!("{name} ({bound})"));
            let mut problem = |what: String| problems.push(format!("{path}: {name}: {what}"));
            if retired.iter().any(|&(old, _)| old == name) {
                problem("it takes back a name retired for claiming a bound its test never asserted".to_string());
            }
            if reported.contains(&name) {
                problem("a runner reports it, so no pull request gates the bound its name states".to_string());
            }
            if attributes.iter().any(|attribute| attribute.contains("ignore")) {
                problem("it is ignored, so no CI job runs it".to_string());
            }
            if !built_on_a_runner(&attributes[..]) {
                let cfgs: Vec<&str> = attributes.iter().map(String::as_str).filter(|attribute| attribute.starts_with("#[cfg(")).collect();
                problem(format!("{} keeps it off both of CI's runners", cfgs.join(" with ")));
            }
            let body = function_in(&source, &name).unwrap_or_else(|| panic!("{path}: {name} has no end"));
            let keyed = code_find(&body, &format!("\"MEASURE {name} ")).map(|at| (body.clone(), at)).or_else(|| {
                // The function the test passes its name to first.
                let call = code_find(&body, &format!("(\"{name}\""))?;
                let callee = &body[..call];
                let callee = &callee[callee.rfind(|c: char| !c.is_alphanumeric() && c != '_').map_or(0, |n| n + 1)..];
                let helper = function_in(&source, callee)?;
                let parameter = helper.split_once('(')?.1.split(':').next()?.trim().to_string();
                let at = code_find(&helper, &format!("\"MEASURE {{{parameter}}} "))?;
                Some((helper, at))
            });
            let Some((code, at)) = keyed else {
                problem("prints no MEASURE line keyed by its name".to_string());
                continue;
            };
            if !["println!(", "eprintln!("].iter().any(|print| code[..at].trim_end().ends_with(print)) {
                problem("its MEASURE line is not printed with println! or eprintln!".to_string());
            }
            // A constant the line formats, as its value: a number, or the
            // count a Duration's constructor takes. It is read from the
            // function that prints the line, then the test, then the file.
            let constant = |name: &str| {
                let named = !name.is_empty() && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
                [code.as_str(), body.as_str(), source.as_str()].into_iter().filter(|_| named).find_map(|scope| {
                    scope.lines().find_map(|line| {
                        let typed = line.trim_start().strip_prefix("const ")?.strip_prefix(name)?.strip_prefix(':')?;
                        let value = typed.split_once('=')?.1.trim().strip_suffix(';')?;
                        let value = value.strip_prefix("Duration::from_").and_then(|call| call.split_once('(')?.1.strip_suffix(')')).unwrap_or(value);
                        let value = value.replace('_', "");
                        value.parse::<f64>().is_ok().then_some(value)
                    })
                })
            };
            let literal = literal_at(&code, at);
            let line = literal.splitn(3, ' ').nth(2).unwrap_or_default();
            let first = line.find('{').filter(|value| line.find(|c: char| c.is_ascii_digit()).is_none_or(|digit| *value < digit));
            let first = first.map(|open| line[open + 1..].split(['}', ':']).next().unwrap_or_default());
            // A local, named inline as a format string names one: a
            // positional `{}`, `{0}` or `{:?}` leaves the value to the call's
            // arguments, which this does not read, and an upper-case name is
            // a constant's or a static's.
            let local = |value: &str| {
                value.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
                    && value.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            };
            match first {
                None => problem(format!("its MEASURE line writes its first number instead of formatting a value: {literal}")),
                Some(value) if constant(value).is_some() => {
                    problem(format!("its MEASURE line's first number is the constant {value}, not a value it measured: {literal}"))
                }
                Some(value) if !local(value) => problem(format!(
                    "its MEASURE line's first value, {{{value}}}, is not a local named inline, so the line cannot show it is measured: {literal}"
                )),
                Some(_) => {}
            }
            // The line with each constant it formats written out, and every
            // other value left out.
            let mut said = String::new();
            let mut rest = line;
            while let Some(open) = rest.find('{') {
                said.push_str(&rest[..open]);
                let close = rest[open..].find('}').map_or(rest.len(), |n| open + n);
                let value = rest[open + 1..close].split(':').next().unwrap_or_default();
                said.push_str(&constant(value).unwrap_or_else(|| " ".to_string()));
                rest = &rest[(close + 1).min(rest.len())..];
            }
            said.push_str(rest);
            // Each number the line says, but the digits that go on a word or
            // a number (`p10`, `x86_64`, the 0 of `85.0`).
            let numbers = said.char_indices().filter(|(at, c)| {
                c.is_ascii_digit() && said[..*at].chars().next_back().is_none_or(|before| !before.is_alphanumeric() && !"_.".contains(before))
            });
            let mut numbers = numbers.map(|(at, _)| {
                let digits = said[at..].find(|c: char| !c.is_ascii_digit() && c != '.').map_or(&said[at..], |end| &said[at..at + end]);
                digits.trim_end_matches('.').parse::<f64>().unwrap_or(f64::NAN)
            });
            if !numbers.any(|number| number == bound as f64) {
                problem(format!("its MEASURE line does not give the bound its name states, {bound}: {literal}"));
            }
        }
    }
    println!("thesis gates whose names state a bound: {}", bounded.join(", "));
    assert!(
        bounded.iter().any(|gate| gate == "frozen_prefix_is_under_700_tokens (700)"),
        "the frozen prompt's gate is not among the gates found to state a bound, so the parser finds less than it should: {bounded:?}"
    );
    assert!(problems.is_empty(), "a thesis gate's name states what it does not measure:\n{}", problems.join("\n"));
}

/// The test count fails under its platform floor, and reports how far it is
/// from the 2,000-test target. No test can count the suite it runs in, so
/// CI does: its test job runs scripts/check-test-count.sh on each of its
/// runners once the suite has built and passed, in a step that nothing there
/// skips or lets fail. Linux and macOS each compile a few tests the other does
/// not, so the script counts against the floor of the platform `uname -s`
/// names, each a ratchet at that platform's count when it was last raised,
/// and every runner of the job has one. It fails under that floor, offers to
/// raise that floor when the count is over it, reports the distance to the
/// 2,000-test target until that is met, and on a platform with no floor
/// reports the count and passes; docs/development.md states both floors and the
/// target. This test holds the wiring, holds that page to the script, and
/// holds the script to its method by driving it with a stand-in `uname` and a
/// stand-in `cargo` whose test binaries list their tests as libtest does: it
/// counts the lines that end in ": test" (not a benchmark, not the summary),
/// takes out the ones `--list --ignored` names, and on Linux and on macOS
/// passes a count at that platform's floor, fails one under it and names that
/// floor for a count over it; it passes any count on another platform, and
/// fails when cargo fails rather than counting nothing.
#[test]
fn the_test_count_fails_under_its_platform_floor() {
    let gate = std::fs::read_to_string(repo().join("scripts/check-test-count.sh")).unwrap();
    let floor_of = |name: &str| -> u64 { assigned(&gate, name).parse().unwrap_or_else(|_| panic!("{name} is not a count")) };
    let (linux, macos) = (floor_of("linux_floor"), floor_of("macos_floor"));
    let target: u64 = assigned(&gate, "target").parse().expect("the target is a count");
    assert_eq!(target, 2000, "the target is 2,000 tests");
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    for claim in [
        format!("fails when there are fewer than {} on Linux, or {} on macOS", thousands(linux), thousands(macos)),
        format!("The target is {} tests", thousands(target)),
    ] {
        assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which scripts/check-test-count.sh gates");
    }

    let job = ci_job("test");
    let step = |run: &str| {
        job.iter()
            .position(|line| line.trim() == format!("run: {run}"))
            .unwrap_or_else(|| panic!("the test job does not run `{run}`"))
    };
    let (tests, count) = (step(NEXTEST_RUN), step("scripts/check-test-count.sh"));
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

/// The core share reports how many of the latest 50 commits of main change
/// the core, against a target of at most a fifth: the capability the agent
/// needs lands in files (tools, skills, templates, hooks) instead of the
/// core growing. No test can read the history it was built from, as CI's
/// test job checks out one commit, so CI measures it: its
/// core-share job runs scripts/core-share.sh on every push to main and every
/// pull request, on the event's own commit with the whole history (for a
/// pull request, GitHub's test merge of it into its base), keeps the CSV the
/// script writes as an artifact and builds no Rust, and nothing skips the job
/// or a step of it, or lets them fail. The share is reported, not gated,
/// until it meets the target: the script exits 0 whatever the share, so the
/// job fails only when the script cannot measure. docs/development.md states the
/// method, the window, the artifact, the target and the gate. This test holds
/// the wiring and that page to the script, and the script to its method on a
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
fn core_share_reports_the_share_of_commits_that_change_the_core() {
    let method = std::fs::read_to_string(repo().join("scripts/core-share.sh")).unwrap();
    let count_of = |name: &str| -> u64 { assigned(&method, name).parse().unwrap_or_else(|_| panic!("{name} is not a count")) };
    let (window, target, core) = (count_of("window"), count_of("target"), assigned(&method, "core"));
    assert_eq!((window, target), (50, 20), "the target is at most 20% of the latest 50 commits");
    // The name CI keeps the CSV under, which docs/development.md gives.
    let artifact = "core-share";
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    for claim in [
        format!("of the latest {window} first-parent commits of main whose diff against their first parent changes a file under `{core}`"),
        format!("the share over it and the {} commits before it", window - 1),
        format!("keeps the CSV as the run's `{artifact}` artifact"),
        format!("The target is at most {target}%"),
        "the job reports the share and fails only when the script cannot measure it".to_string(),
    ] {
        assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which the script and CI's core-share job hold");
    }

    let events = triggers("ci.yml");
    let on = |event: &str| events.iter().find(|(name, _)| name == event).map(|(_, filters)| filters.as_slice());
    assert!(
        on("push") == Some(&["branches: [main]".to_string()][..]) && on("pull_request") == Some(&[][..]),
        "CI does not run on every push to main and every pull request: {events:?}"
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

/// The pull-request gate is the CI workflow's jobs, and they start at once:
/// none needs another, so the gate takes as long as its slowest job, and
/// Clippy, which builds with a compiler of its own, pinned to a release, is a
/// job of its own instead of steps after the tests. The jobs are the tests,
/// the suite time, Clippy, the thesis gates, the core share and dist's plan
/// of the release;
/// the six release builds are not among them (crates/tui/tests/release.rs
/// holds their workflow and the plan). CI runs on every push to main, every
/// pull request, every merge group of a merge queue and every night. A push
/// to a pull request cancels the run still testing its previous head: the
/// concurrency group is the pull request's number, and every other run's
/// group is its own run id, so no other run is ever cancelled. Every cache
/// step of a workflow that builds saves only on main: a pull request's copy
/// would serve only that pull request's later runs, which main's copy serves
/// as well unless the pull request changes Cargo.lock, since rust-cache keeps
/// no workspace crate, and it would take the repository's cache space. A job
/// that builds a ci.yml job's dependencies again saves none, and restores
/// that job's cache by naming it as its shared key.
#[test]
fn the_ci_jobs_start_at_once_and_a_newer_push_cancels_its_run() {
    let events = triggers("ci.yml");
    let names: Vec<&str> = events.iter().map(|(event, _)| event.as_str()).collect();
    assert_eq!(names, ["push", "pull_request", "merge_group", "schedule"], "CI runs on other events than these: {events:?}");
    let filters = |event: &str| events.iter().find(|(name, _)| name == event).map(|(_, filters)| filters.clone()).unwrap();
    assert_eq!(filters("push"), ["branches: [main]"], "CI does not run on every push to main");
    assert!(
        filters("pull_request").is_empty() && filters("merge_group").is_empty(),
        "a filter keeps CI off some pull requests or merge groups: {events:?}"
    );
    let schedule = filters("schedule");
    assert!(schedule.len() == 1 && schedule[0].starts_with("- cron: "), "CI does not run every night: {schedule:?}");

    let ci = std::fs::read_to_string(repo().join(".github/workflows/ci.yml")).unwrap();
    let concurrency = "\nconcurrency:\n  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.run_id }}\n  cancel-in-progress: true\n";
    assert!(
        ci.contains(concurrency),
        "a push to a pull request does not cancel the run of its previous head, or other runs can be cancelled: {concurrency}"
    );
    let jobs: Vec<&str> = ci
        .lines()
        .skip_while(|line| *line != "jobs:")
        .skip(1)
        .filter_map(|line| line.strip_prefix("  ")?.strip_suffix(':').filter(|key| !key.starts_with([' ', '#'])))
        .collect();
    assert_eq!(jobs, ["test", "suite-time", "clippy", "thesis", "core-share", "release-plan"], "the gate's jobs changed");
    for job in &jobs {
        let waits: Vec<String> = ci_job(job).into_iter().filter(|line| line.trim().starts_with("needs:")).collect();
        assert!(waits.is_empty(), "the {job} job waits for another, so the gate takes the two one after the other: {waits:?}");
    }

    let clippy = ci_job("clippy");
    let toolchain = clippy
        .iter()
        .find_map(|line| line.trim().strip_prefix("toolchain: "))
        .expect("the clippy job installs no toolchain");
    assert!(
        toolchain.split('.').count() == 3 && toolchain.split('.').all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())),
        "Clippy's compiler {toolchain} is not a release, so a Rust release can fail an unrelated pull request"
    );
    let lint = format!("run: cargo +{toolchain} clippy --workspace --all-targets --locked -- -D warnings");
    for line in ["components: clippy", lint.as_str()] {
        assert!(clippy.iter().any(|l| l.trim() == line), "the clippy job does not hold `{line}`: {clippy:?}");
    }
    let soft: Vec<&String> = clippy
        .iter()
        .filter(|line| {
            let key = line.trim().trim_start_matches("- ");
            key.starts_with("if:") || key.starts_with("continue-on-error:")
        })
        .collect();
    assert!(soft.is_empty(), "Clippy can be skipped or fail while the gate passes: {soft:?}");

    // ci.yml's jobs whose cache steps save on main, by job: the keys another
    // job's cache step may restore.
    let saved: Vec<&str> =
        jobs.iter().copied().filter(|job| ci_job(job).iter().any(|line| line.trim() == "save-if: ${{ github.ref == 'refs/heads/main' }}")).collect();
    for workflow in ["ci.yml", "release-build.yml", "mutants.yml", "reports.yml"] {
        let lines: Vec<String> = std::fs::read_to_string(repo().join(".github/workflows").join(workflow))
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        let caches: Vec<usize> =
            (0..lines.len()).filter(|&at| lines[at].trim().starts_with("uses: Swatinem/rust-cache@")).collect();
        assert!(!caches.is_empty(), "{workflow} caches nothing, so this test holds nothing there");
        for at in caches {
            let keys = step_keys(&lines, at);
            let shared = keys.iter().find_map(|key| key.strip_prefix("shared-key: "));
            let saves_on_main = keys.contains(&"save-if: ${{ github.ref == 'refs/heads/main' }}");
            let restores = keys.contains(&"save-if: false") && shared.is_some_and(|job| saved.contains(&job));
            assert!(
                (saves_on_main && shared.is_none()) || restores,
                "a cache step of {workflow} saves on other branches too, or saves a second copy of a ci.yml job's: {keys:?}"
            );
        }
    }
}

/// Every test of the workspace runs under cargo-nextest in CI's test job:
/// each test in a process of its own, as many at once as the runner has
/// cores, where `cargo test` runs the test binaries one after another. The
/// job installs nextest's release binary for its runner, pinned to a version
/// and checked against the sha256 its release publishes before it is
/// unpacked, for each runner of its matrix, and fails on a runner it pinned
/// none for. It runs nextest under `.config/nextest.toml`'s ci profile: every
/// failure shown rather than the first, none retried, and a hung test stopped
/// after a bounded number of its slow periods; and no setting anywhere in
/// that file retries a test. Nor does a failure on one runner cancel the
/// job on the other, which would hide its failures. A lock inside one test
/// binary keeps nothing apart across the processes nextest starts, so the
/// harness-time tests, which thesis_rounds holds apart with a lock for their
/// CPU-bound timing, reserve every test thread instead. The job builds the
/// workspace's own crates unoptimized, the build each debug bound was set on,
/// by two settings in cargo's home config that override Cargo.toml's test
/// profile, written before any step builds. The doc tests, which
/// nextest does not run, run in a step of their own. docs/development.md states this.
#[test]
fn every_test_runs_under_a_checksummed_nextest_that_retries_none() {
    let job = ci_job("test");
    let at = |what: &str, wanted: &dyn Fn(&str) -> bool| {
        job.iter().position(|line| wanted(line.trim())).unwrap_or_else(|| panic!("the test job has no {what}"))
    };
    let install = at("step that installs cargo-nextest", &|line| line == "- name: Install cargo-nextest");
    let run = at("step that runs nextest", &|line| line == format!("run: {NEXTEST_RUN}"));
    at("step that runs the doc tests", &|line| line == format!("run: {DOC_TESTS}"));
    assert!(install < run, "the tests run before nextest is installed");
    let unoptimized = at("step that builds the workspace's crates unoptimized", &|line| {
        line.starts_with("printf ")
            && ["[profile.test.package.open-max-core]\\nopt-level = 0", "[profile.test.package.openmax]\\nopt-level = 0", "/config.toml"]
                .iter()
                .all(|part| line.contains(part))
    });
    assert!(unoptimized < run, "the tests are built before the workspace's crates are set unoptimized");
    let keys = step_keys(&job, install);
    let version = keys.iter().find_map(|key| key.strip_prefix("NEXTEST_VERSION: ")).expect("nextest's version is not pinned");
    assert!(
        version.split('.').count() == 3 && version.split('.').all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit())),
        "nextest is not pinned to a release: {version}"
    );
    // Each runner's asset and its hash, then the line for any other runner.
    let pins: Vec<(&str, &str, &str)> = keys
        .iter()
        .filter_map(|key| {
            let (runner, rest) = key.split_once(") asset=")?;
            let (asset, rest) = rest.split_once(" sha256=")?;
            Some((runner, asset, rest.strip_suffix(" ;;")?))
        })
        .collect();
    let pinned: Vec<(&str, &str)> = pins.iter().map(|(runner, asset, _)| (*runner, *asset)).collect();
    assert_eq!(
        pinned,
        [("Linux-X64", "x86_64-unknown-linux-gnu"), ("macOS-*", "universal-apple-darwin")],
        "nextest is not pinned for each runner of the test job, or for another one"
    );
    let runners = job.iter().find_map(|line| line.trim().strip_prefix("os: [")?.strip_suffix(']')).expect("the test job names no runners");
    assert_eq!(runners, "ubuntu-latest, macos-latest", "the test job runs on a runner whose nextest is not pinned");
    for (runner, _, sha256) in &pins {
        assert!(
            sha256.len() == 64 && sha256.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "{runner}'s nextest hash is not a sha256: {sha256}"
        );
    }
    assert!(
        keys.iter().any(|key| key.starts_with("*) ") && key.ends_with("exit 1 ;;")),
        "a runner with no pinned nextest installs one unchecked: {keys:?}"
    );
    let line = |start: &str| {
        keys.iter().position(|key| key.starts_with(start)).unwrap_or_else(|| panic!("the install step has no `{start}...`: {keys:?}"))
    };
    let fetched = line("curl -sSfL --retry 3 \"https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-$NEXTEST_VERSION/cargo-nextest-$NEXTEST_VERSION-$asset.tar.gz\" -o \"$RUNNER_TEMP/cargo-nextest.tar.gz\"");
    let checked = line("echo \"$sha256  $RUNNER_TEMP/cargo-nextest.tar.gz\" | shasum -a 256 --check --strict -");
    let unpacked = line("tar -xzf \"$RUNNER_TEMP/cargo-nextest.tar.gz\" ");
    assert!(fetched < checked && checked < unpacked, "nextest is unpacked before its hash is checked");
    let strategy = at("strategy", &|line| line == "strategy:");
    assert!(
        job[strategy + 1..].iter().take_while(|line| line.starts_with("      ")).any(|line| line.trim() == "fail-fast: false"),
        "a failure on one runner cancels the test job on the other, so its failures go unseen"
    );

    let config: toml::Table = std::fs::read_to_string(repo().join(".config/nextest.toml")).unwrap().parse().unwrap();
    let profiles = config.get("profile").and_then(toml::Value::as_table).expect(".config/nextest.toml has no profiles");
    let ci = profiles.get("ci").and_then(toml::Value::as_table).expect("there is no ci profile");
    assert_eq!(ci.get("fail-fast").and_then(toml::Value::as_bool), Some(false), "a run stops at its first failure");
    assert_eq!(ci.get("retries").and_then(toml::Value::as_integer), Some(0), "the ci profile does not say it retries no test");
    let slow = ci.get("slow-timeout").and_then(toml::Value::as_table).expect("the ci profile never stops a hung test");
    assert!(
        slow.get("period").and_then(toml::Value::as_str).is_some()
            && slow.get("terminate-after").and_then(toml::Value::as_integer).is_some_and(|periods| periods >= 1),
        "a hung test is never stopped: {slow:?}"
    );
    let mut overrides = Vec::new();
    for (name, profile) in profiles {
        let profile = profile.as_table().unwrap_or_else(|| panic!("the {name} profile is not a table"));
        assert!(
            profile.get("retries").is_none_or(|retries| retries.as_integer() == Some(0)),
            "the {name} profile retries failed tests"
        );
        for each in profile.get("overrides").and_then(toml::Value::as_array).into_iter().flatten() {
            let each = each.as_table().unwrap_or_else(|| panic!("an override of the {name} profile is not a table"));
            assert!(each.get("retries").is_none(), "an override of the {name} profile retries the tests it selects: {each:?}");
            overrides.push((name.as_str(), each));
        }
    }
    let alone = overrides.iter().any(|(profile, each)| {
        *profile == "default"
            && each.get("filter").and_then(toml::Value::as_str) == Some("binary_id(=open-max-core::thesis_rounds)")
            && each.get("threads-required").and_then(toml::Value::as_str) == Some("num-test-threads")
    });
    assert!(alone, "the harness-time tests run beside others under nextest: {overrides:?}");
    let rounds = std::fs::read_to_string(repo().join("crates/core/tests/thesis_rounds.rs")).unwrap();
    assert!(
        rounds.contains("static SERIAL: tokio::sync::Mutex<()>"),
        "thesis_rounds no longer holds its tests apart, the reason they reserve every thread under nextest"
    );
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    let claim = "CI's test job runs every test under cargo-nextest, each in a process of its own and none retried (`.config/nextest.toml`)";
    assert!(stated.contains(claim), "docs/development.md does not say \"{claim}\", which this test holds");
}

/// The pull-request gate turns green in under 90 seconds at the median and
/// under 2 minutes at the 99th percentile; it does not yet, so this is a
/// report. No test can time the run it is part of, so CI measures the runs
/// before: its Gate time workflow runs scripts/gate-time.sh on every push to
/// main, every night and on demand, never on a pull request, with read access
/// only, and nothing skips its job or lets it fail. The script reads GitHub's
/// record of the CI workflow's latest pull-request runs with `gh api`, and
/// for each of the latest N (20 by default) whose first attempt passed takes
/// the wall clock from its first job's start to its last job's end, a skipped
/// job counting at neither end, a job that began and ended in one second
/// counting, and a new year between the two read as any other second. It
/// prints a line for each, then the p50, p90 and p99 by the nearest rank and
/// the slowest; then how many runs as recent failed at their first attempt (a
/// failed job outranks a cancelled one, and a run with no job failed), how
/// many of those passed when run again, and how many were cancelled; then the
/// target, met only when the median is under 90 s and the p99 under 120 s
/// over all N runs asked for and none as recent failed at its first attempt,
/// so the N are consecutive. It reads a second page of runs only
/// after a full first one, reads no run past the N-th that passed, says when
/// it found fewer, adds its report to the job's summary, and names its
/// repository by --repo, else GITHUB_REPOSITORY. It exits 0 whatever it
/// measures, 1 when it cannot measure (gh fails or prints what the script
/// does not read, or no run passed), and 2 on a malformed argument or a perl
/// without its modules. docs/development.md states the method and the target. This
/// test drives the script with a stand-in `gh` serving runs it makes up.
#[test]
fn gate_time_reports_the_p50_p90_and_p99_of_green_pull_request_runs() {
    let gate_script = std::fs::read_to_string(repo().join("scripts/gate-time.sh")).unwrap();
    let default_runs: u64 = assigned(&gate_script, "runs").parse().expect("the default sample is a count");
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    for claim in [
        "`scripts/gate-time.sh` (bash, gh and perl) measures the pull-request gate".to_string(),
        format!("the latest {default_runs} pull-request runs of the CI workflow that passed at their first attempt"),
        "from the start of its first job to the end of its last".to_string(),
        format!("The target is a p50 under 90 seconds and a p99 under 120 seconds over {default_runs} consecutive runs"),
        "The Gate time workflow runs it on every push to main and every night".to_string(),
        "the job reports the times and fails only when the script cannot measure them".to_string(),
    ] {
        assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which scripts/gate-time.sh and its workflow hold");
    }
    let events = triggers("gate-time.yml");
    let names: Vec<&str> = events.iter().map(|(event, _)| event.as_str()).collect();
    assert_eq!(names, ["push", "schedule", "workflow_dispatch"], "the gate time is measured on other events: {events:?}");
    assert_eq!(events[0].1, ["branches: [main]"], "the gate time is not measured on every push to main");
    let workflow = std::fs::read_to_string(repo().join(".github/workflows/gate-time.yml")).unwrap();
    assert!(
        workflow.contains("\npermissions:\n  actions: read\n  contents: read\n"),
        "the gate-time workflow can do more than read the runs"
    );
    let job = workflow_job("gate-time.yml", "gate-time");
    let soft: Vec<&str> = job
        .iter()
        .map(|line| line.trim().trim_start_matches("- "))
        .filter(|key| ["if:", "needs:", "continue-on-error:"].iter().any(|refused| key.starts_with(refused)))
        .collect();
    assert!(soft.is_empty(), "the gate time can go unmeasured while the run passes: {soft:?}");
    let measured = job.iter().position(|line| line.trim() == "run: scripts/gate-time.sh").expect("the gate-time job does not run the script");
    assert!(
        step_keys(&job, measured).contains(&"GH_TOKEN: ${{ github.token }}"),
        "the script runs without the run's token, so gh cannot read the runs"
    );
    assert!(job.iter().any(|line| line.trim().starts_with("timeout-minutes:")), "a stalled gh holds the job to GitHub's own limit");

    let dir = scratch("gate-time");
    let (bin, served) = (dir.join("bin"), dir.join("served"));
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::create_dir_all(&served).unwrap();
    // `gh api PATH` as GitHub answers it: the file named after PATH, each
    // character a file name should not hold made `_`, or a 404. It logs the
    // paths it is asked for.
    executable(
        &bin.join("gh"),
        "#!/bin/sh\n\
         [ \"$1\" = api ] && [ $# -eq 2 ] || { echo \"unexpected: gh $*\" >&2; exit 2; }\n\
         printf '%s\\n' \"$2\" >> \"$GH_LOG\"\n\
         f=\"$GH_SERVED/$(printf '%s' \"$2\" | tr -c 'A-Za-z0-9.-' '_')\"\n\
         [ -f \"$f\" ] || { echo \"gh: Not Found (HTTP 404)\" >&2; exit 1; }\n\
         cat \"$f\"\n",
    );
    let file = |path: &str| -> String {
        path.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect()
    };
    let page = |repo: &str, n: usize| format!("repos/{repo}/actions/workflows/ci.yml/runs?event=pull_request&status=completed&per_page=100&page={n}");
    let jobs_of = |repo: &str, id: u64| format!("repos/{repo}/actions/runs/{id}/attempts/1/jobs?per_page=100");
    let serve = |path: &str, body: &str| std::fs::write(served.join(file(path)), body).unwrap();
    let sha = |id: u64| format!("{id:08}{}", "a".repeat(32));
    // A job: its name, conclusion, and the second of an hour it started and
    // ended at, or neither.
    let job = |name: &str, conclusion: &str, span: Option<(u32, u32)>| {
        let at = |second: u32| format!("2026-11-06T10:{:02}:{:02}Z", second / 60, second % 60);
        serde_json::json!({"name": name, "conclusion": conclusion, "started_at": span.map(|(a, _)| at(a)), "completed_at": span.map(|(_, b)| at(b))})
    };
    // A run: its id, its conclusion after its last attempt, and the jobs of
    // its first.
    type Run<'a> = (u64, &'a str, Vec<serde_json::Value>);
    // A page of runs, newest first.
    let serve_page = |repo: &str, n: usize, runs: &[Run]| {
        let listed: Vec<serde_json::Value> = runs
            .iter()
            .map(|(id, conclusion, _)| serde_json::json!({"id": id, "head_sha": sha(*id), "conclusion": conclusion}))
            .collect();
        serve(&page(repo, n), &serde_json::json!({"total_count": runs.len(), "workflow_runs": listed}).to_string());
        for (id, _, jobs) in runs {
            serve(&jobs_of(repo, *id), &serde_json::json!({"total_count": jobs.len(), "jobs": jobs}).to_string());
        }
    };
    let green = |id: u64, span: (u32, u32)| (id, "success", vec![job("Test (ubuntu-latest)", "success", Some(span))]);

    // Newest first: a pass whose skipped job, at neither end, would widen it;
    // a failure that passed when run again; a failure whose matrix cancelled
    // the job beside it; a run cancelled with none failed; four passes, one
    // ending on a job whose name holds spaces and one with a job that began
    // and ended in the same second; a run with no job; and a fifth pass that
    // a sample of five never reads.
    serve_page(
        "o/main",
        1,
        &[
            (101, "success", vec![
                job("Test (macos-latest)", "success", Some((3, 150))),
                job("Thesis (macos-latest)", "success", Some((8, 213))),
                job("Release plan", "skipped", Some((0, 300))),
            ]),
            (102, "success", vec![job("Test (ubuntu-latest)", "failure", Some((0, 100))), job("Clippy", "success", Some((0, 50)))]),
            (103, "failure", vec![
                job("Test (ubuntu-latest)", "failure", Some((0, 50))),
                job("Test (macos-latest)", "cancelled", Some((0, 51))),
            ]),
            (104, "cancelled", vec![job("Test (ubuntu-latest)", "success", Some((0, 40))), job("Clippy", "cancelled", Some((0, 41)))]),
            (105, "success", vec![job("Core share", "success", Some((100, 107))), job("Thesis (macos-latest)", "success", Some((100, 161)))]),
            (106, "startup_failure", vec![]),
            green(107, (0, 89)),
            green(108, (10, 100)),
            (109, "success", vec![job("Test (ubuntu-latest)", "success", Some((0, 119))), job("Core share", "success", Some((5, 5)))]),
            green(110, (0, 120)),
        ],
    );
    // Twenty passes of 1 to 20 seconds, where each percentile is a rank of
    // its own; and a pass across the turn of a year.
    let twenty: Vec<Run> = (1..=20).map(|k| green(400 + k, (0, k as u32))).collect();
    serve_page("o/twenty", 1, &twenty);
    let new_year = serde_json::json!({
        "name": "Test (ubuntu-latest)", "conclusion": "success",
        "started_at": "2025-12-31T23:59:30Z", "completed_at": "2026-01-01T00:00:30Z",
    });
    serve_page("o/newyear", 1, &[(431, "success", vec![new_year])]);
    serve_page("o/met", 1, &[green(201, (0, 61)), green(202, (0, 89)), green(203, (0, 119))]);
    serve_page("o/p50", 1, &[green(211, (0, 90)), green(212, (0, 100))]);
    serve_page("o/p99", 1, &[green(221, (0, 60)), green(222, (0, 120))]);
    serve_page("o/one", 1, &[(231, "failure", vec![job("Clippy", "failure", Some((0, 30)))]), green(232, (0, 45))]);
    // A full page of cancelled runs, then a page with one that passed.
    let cancelled: Vec<Run> =
        (1000..1100).map(|id| (id, "cancelled", vec![job("Test (ubuntu-latest)", "cancelled", Some((0, 5)))])).collect();
    serve_page("o/paged", 1, &cancelled);
    serve_page("o/paged", 2, &[green(2000, (0, 30))]);
    serve_page("o/nogreen", 1, &[(241, "failure", vec![job("Clippy", "failure", Some((0, 30)))]), cancelled[0].clone()]);
    // What the script must refuse rather than misread.
    serve(&page("o/garbled", 1), "<html>rate limited</html>");
    serve(&page("o/nolist", 1), "{\"total_count\": 0}");
    // Each with one field wrong, so that without its check the script would
    // read jobs no stand-in serves and say something else.
    serve(&page("o/badrun", 1), &format!("{{\"total_count\": 1, \"workflow_runs\": [{{\"id\": 0, \"head_sha\": \"{}\"}}]}}", sha(301)));
    serve(&page("o/badsha", 1), "{\"total_count\": 1, \"workflow_runs\": [{\"id\": 351, \"head_sha\": \"abc\"}]}");
    serve_page("o/partial", 1, &[green(311, (0, 30))]);
    serve(&jobs_of("o/partial", 311), &serde_json::json!({"total_count": 2, "jobs": [job("Clippy", "success", Some((0, 30)))]}).to_string());
    serve_page("o/badjob", 1, &[(321, "success", vec![serde_json::json!({"conclusion": "success"})])]);
    serve_page("o/noverdict", 1, &[(361, "success", vec![serde_json::json!({"name": "Clippy"})])]);
    serve_page("o/notime", 1, &[(331, "success", vec![job("Clippy", "success", None)])]);
    serve_page("o/backwards", 1, &[green(341, (30, 29))]);
    // A perl that cannot load one of the modules the script needs.
    let broken = |module: &str| {
        let lib = dir.join(format!("no-{}", module.replace("::", "-")));
        let file = lib.join(module.replace("::", "/") + ".pm");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, format!("die \"Can't locate {module}\\n\";\n")).unwrap();
        lib.display().to_string()
    };
    let (no_json, no_time) = (broken("JSON::PP"), broken("Time::Local"));
    let summary = dir.join("summary.md");

    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap_or_default());
    // One run of the script with `args` and `env`: its exit code, what it
    // printed, and the paths it asked gh for.
    let gate = |tag: &str, args: &[&str], env: &[(&str, String)]| {
        let log = dir.join(format!("asked-{tag}"));
        let mut command = script("scripts/gate-time.sh");
        command.args(args).env("PATH", &path).env("GH_SERVED", &served).env("GH_LOG", &log).env_remove("GITHUB_REPOSITORY");
        for (key, value) in env {
            command.env(key, value);
        }
        let (code, report) = ran(&mut command);
        (code, report, std::fs::read_to_string(&log).unwrap_or_default())
    };
    let summarize = [("GITHUB_STEP_SUMMARY", summary.display().to_string())];
    // A run of the script: its tag, its arguments and the environment it adds.
    type Case<'a> = (&'a str, Vec<&'a str>, Vec<(&'a str, String)>);
    let cases: Vec<Case> = vec![
        ("five", vec!["--repo", "o/main", "--runs", "5"], vec![]),
        ("default", vec!["--repo", "o/main"], vec![]),
        ("twenty", vec!["--repo", "o/twenty"], vec![]),
        ("newyear", vec!["--repo", "o/newyear", "--runs", "1"], vec![]),
        ("unsummarized", vec!["--repo", "o/met", "--runs", "3"], vec![("GITHUB_STEP_SUMMARY", dir.join("gone/summary.md").display().to_string())]),
        ("met", vec!["--runs", "3"], vec![("GITHUB_REPOSITORY", "o/met".to_string()), summarize[0].clone()]),
        ("over", vec!["--repo", "o/met", "--runs", "3"], vec![("GITHUB_REPOSITORY", "o/nowhere".to_string())]),
        ("short", vec!["--repo", "o/met"], vec![]),
        ("p50", vec!["--repo", "o/p50", "--runs", "2"], vec![]),
        ("p99", vec!["--repo", "o/p99", "--runs", "2"], vec![]),
        ("one", vec!["--repo", "o/one", "--runs", "1"], vec![]),
        ("paged", vec!["--repo", "o/paged", "--runs", "1"], vec![]),
        ("nogreen", vec!["--repo", "o/nogreen"], vec![]),
        ("missing", vec!["--repo", "o/missing"], vec![]),
        ("annotated", vec!["--repo", "o/missing"], vec![("GITHUB_ACTIONS", "true".to_string())]),
        ("garbled", vec!["--repo", "o/garbled"], vec![]),
        ("nolist", vec!["--repo", "o/nolist"], vec![]),
        ("badrun", vec!["--repo", "o/badrun"], vec![]),
        ("badsha", vec!["--repo", "o/badsha"], vec![]),
        ("partial", vec!["--repo", "o/partial"], vec![]),
        ("badjob", vec!["--repo", "o/badjob"], vec![]),
        ("noverdict", vec!["--repo", "o/noverdict"], vec![]),
        ("notime", vec!["--repo", "o/notime"], vec![]),
        ("backwards", vec!["--repo", "o/backwards"], vec![]),
        ("no-json", vec!["--repo", "o/met"], vec![("PERL5LIB", no_json.clone())]),
        ("no-time", vec!["--repo", "o/met"], vec![("PERL5LIB", no_time.clone())]),
        ("no-repo", vec![], vec![]),
        ("bad-repo", vec!["--repo", "o/met/x"], vec![]),
        ("runs-0", vec!["--repo", "o/met", "--runs", "0"], vec![]),
        ("runs-07", vec!["--repo", "o/met", "--runs", "07"], vec![]),
        ("runs-1000", vec!["--repo", "o/met", "--runs", "1000"], vec![]),
        ("runs-x", vec!["--repo", "o/met", "--runs", "x"], vec![]),
        ("runs-none", vec!["--repo", "o/met", "--runs"], vec![]),
        ("repo-none", vec!["--repo"], vec![]),
        ("unknown", vec!["--repo", "o/met", "--json"], vec![]),
    ];
    let results: Vec<(Option<i32>, String, String)> = std::thread::scope(|scope| {
        let gate = &gate;
        let running: Vec<_> = cases.iter().map(|(tag, args, env)| scope.spawn(move || gate(tag, args, env))).collect();
        running.into_iter().map(|run| run.join().unwrap()).collect()
    });
    let outcome = |tag: &str| &results[cases.iter().position(|(name, _, _)| *name == tag).unwrap()];
    let summary_written = std::fs::read_to_string(&summary).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);

    let (code, report, asked) = outcome("five");
    assert_eq!(*code, Some(0), "{report}");
    assert_eq!(
        report.as_str(),
        "gate time of the latest 5 runs of ci.yml in o/main on a pull request that passed at the first attempt, from the start of its first job to the end of its last:\n\
         run 101 at 00000101: 210 s, Thesis (macos-latest) last\n\
         run 105 at 00000105: 61 s, Thesis (macos-latest) last\n\
         run 107 at 00000107: 89 s, Test (ubuntu-latest) last\n\
         run 108 at 00000108: 90 s, Test (ubuntu-latest) last\n\
         run 109 at 00000109: 119 s, Test (ubuntu-latest) last\n\
         p50 90 s, p90 210 s, p99 210 s, slowest 210 s over 5 runs\n\
         since the oldest of them, 3 runs failed at the first attempt (1 of them passed when run again) and 1 was cancelled\n\
         target p50 under 90 s and p99 under 120 s over 5 consecutive runs that passed at the first attempt: not met\n",
        "the gate time of five runs is not measured by its method"
    );
    let asked_for: Vec<&str> = asked.lines().collect();
    let mut expected = vec![page("o/main", 1)];
    expected.extend((101..=109).map(|id| jobs_of("o/main", id)));
    assert_eq!(asked_for, expected, "the script reads other runs than up to the fifth that passed, or reads them otherwise");

    // Every run read: the six that passed, fewer than the 20 asked for, and
    // no second page after a first that was not full.
    let (code, report, asked) = outcome("default");
    assert_eq!(*code, Some(0), "{report}");
    assert!(
        report.starts_with("gate time of the latest 6 runs of ci.yml in o/main on a pull request that passed at the first attempt, fewer than the 20 asked for, from")
            && report.contains("\nrun 110 at 00000110: 120 s, Test (ubuntu-latest) last\n")
            && report.contains("\np50 90 s, p90 210 s, p99 210 s, slowest 210 s over 6 runs\n")
            && report.ends_with("\nsince the oldest of them, 3 runs failed at the first attempt (1 of them passed when run again) and 1 was cancelled\ntarget p50 under 90 s and p99 under 120 s over 20 consecutive runs that passed at the first attempt: not met\n"),
        "a sample smaller than the one asked for is not reported as one: {report}"
    );
    assert!(asked.lines().count() == 11 && !asked.contains("page=2"), "{asked}");
    let (code, report, _) = outcome("twenty");
    assert!(
        *code == Some(0)
            && report.starts_with("gate time of the latest 20 runs of ci.yml in o/twenty on a pull request that passed at the first attempt, from")
            && report.contains("\np50 10 s, p90 18 s, p99 20 s, slowest 20 s over 20 runs\n")
            && report.ends_with("\ntarget p50 under 90 s and p99 under 120 s over 20 consecutive runs that passed at the first attempt: met\n"),
        "twenty runs' percentiles are not their nearest ranks: {report}"
    );
    let (code, report, _) = outcome("newyear");
    assert!(*code == Some(0) && report.contains("\nrun 431 at 00000431: 60 s, Test (ubuntu-latest) last\n"), "a run across a new year: {report}");

    // The target: met under 90 s at the median and 120 s at the p99 over
    // every run asked for, none as recent failed; missed at either bound,
    // with a failure among them, or with fewer runs than asked for.
    let (code, report, _) = outcome("met");
    assert_eq!(*code, Some(0), "{report}");
    assert!(
        report.contains("\np50 89 s, p90 119 s, p99 119 s, slowest 119 s over 3 runs\n")
            && report.ends_with("\nsince the oldest of them, 0 runs failed at the first attempt (0 of them passed when run again) and 0 were cancelled\ntarget p50 under 90 s and p99 under 120 s over 3 consecutive runs that passed at the first attempt: met\n")
            && !report.contains("asked for"),
        "a gate under both bounds does not meet the target: {report}"
    );
    assert_eq!(summary_written, format!("## Gate time\n\n```text\n{}```\n", report), "the report is not the job's summary");
    let (code, over_report, _) = outcome("over");
    assert!(*code == Some(0) && over_report == report, "--repo does not name the repository over GITHUB_REPOSITORY: {over_report}");
    for (tag, line) in [
        ("p50", "\np50 90 s, p90 100 s, p99 100 s, slowest 100 s over 2 runs\n"),
        ("p99", "\np50 60 s, p90 120 s, p99 120 s, slowest 120 s over 2 runs\n"),
    ] {
        let (code, report, _) = outcome(tag);
        assert!(
            *code == Some(0)
                && report.contains(line)
                && report.ends_with("\ntarget p50 under 90 s and p99 under 120 s over 2 consecutive runs that passed at the first attempt: not met\n"),
            "a gate at the {tag} bound meets the target: {report}"
        );
    }
    let (code, report, _) = outcome("short");
    assert!(
        *code == Some(0)
            && report.contains(", fewer than the 20 asked for, ")
            && report.contains("\np50 89 s, p90 119 s, p99 119 s, slowest 119 s over 3 runs\n")
            && report.ends_with(" and 0 were cancelled\ntarget p50 under 90 s and p99 under 120 s over 20 consecutive runs that passed at the first attempt: not met\n"),
        "fewer runs than asked for meet the target: {report}"
    );
    let (code, report, _) = outcome("one");
    assert!(
        *code == Some(0)
            && report.starts_with("gate time of the latest 1 run of ci.yml in o/one on a pull request")
            && report.contains("\np50 45 s, p90 45 s, p99 45 s, slowest 45 s over 1 run\n")
            && report.ends_with("\nsince the oldest of them, 1 run failed at the first attempt (0 of them passed when run again) and 0 were cancelled\ntarget p50 under 90 s and p99 under 120 s over 1 consecutive run that passed at the first attempt: not met\n"),
        "a failed first attempt among the runs leaves the target met: {report}"
    );
    let (code, report, asked) = outcome("paged");
    assert!(
        *code == Some(0)
            && report.contains("\nrun 2000 at 00002000: 30 s, Test (ubuntu-latest) last\n")
            && report.contains(" and 100 were cancelled\n")
            && asked.lines().last() == Some(jobs_of("o/paged", 2000).as_str())
            && asked.contains(&page("o/paged", 2))
            && !asked.contains("page=3"),
        "a second page is not read after a full first one: {report}"
    );
    // A cancelled run gave no verdict, so it breaks no run of them.
    assert!(
        report.ends_with(" and 100 were cancelled\ntarget p50 under 90 s and p99 under 120 s over 1 consecutive run that passed at the first attempt: met\n"),
        "a cancelled run breaks the consecutive runs that passed: {report}"
    );

    // Each refusal: its exit code and what it says.
    for (tag, wanted, said) in [
        ("nogreen", 1, "error: no pull-request run of ci.yml in o/nogreen passed at its first attempt".to_string()),
        ("missing", 1, format!("gh: Not Found (HTTP 404)\nerror: gh api {} failed", page("o/missing", 1))),
        ("annotated", 1, format!("::error::gh api {} failed", page("o/missing", 1))),
        ("garbled", 1, format!("error: gh api {} printed what this script does not read", page("o/garbled", 1))),
        ("nolist", 1, format!("error: gh api {} printed no list of runs", page("o/nolist", 1))),
        ("badrun", 1, format!("error: gh api {} listed a run this script does not read", page("o/badrun", 1))),
        ("badsha", 1, format!("error: gh api {} listed a run this script does not read", page("o/badsha", 1))),
        ("partial", 1, format!("error: gh api {} printed no whole list of jobs", jobs_of("o/partial", 311))),
        ("badjob", 1, format!("error: gh api {} listed a job this script does not read", jobs_of("o/badjob", 321))),
        ("noverdict", 1, format!("error: gh api {} listed a job this script does not read", jobs_of("o/noverdict", 361))),
        ("unsummarized", 1, "error: cannot write the job summary".to_string()),
        ("notime", 1, format!("error: gh api {} gave Clippy no start and end this script reads", jobs_of("o/notime", 331))),
        ("backwards", 1, format!("error: gh api {} gave Test (ubuntu-latest) no start and end this script reads", jobs_of("o/backwards", 341))),
        ("no-json", 2, "needs perl with its core modules JSON::PP and Time::Local".to_string()),
        ("no-time", 2, "needs perl with its core modules JSON::PP and Time::Local".to_string()),
        ("no-repo", 2, "error: name the repository as OWNER/NAME with --repo (or GITHUB_REPOSITORY), not ''".to_string()),
        ("bad-repo", 2, "error: name the repository as OWNER/NAME with --repo (or GITHUB_REPOSITORY), not 'o/met/x'".to_string()),
        ("runs-0", 2, "error: --runs takes a count of runs from 1 to 999, not '0'".to_string()),
        ("runs-07", 2, "error: --runs takes a count of runs from 1 to 999, not '07'".to_string()),
        ("runs-1000", 2, "error: --runs takes a count of runs from 1 to 999, not '1000'".to_string()),
        ("runs-x", 2, "error: --runs takes a count of runs from 1 to 999, not 'x'".to_string()),
        ("runs-none", 2, "usage: ".to_string()),
        ("repo-none", 2, "usage: ".to_string()),
        ("unknown", 2, "usage: ".to_string()),
    ] {
        let (code, report, _) = outcome(tag);
        assert!(*code == Some(wanted) && report.contains(&said), "{tag}: not refused with {wanted} and \"{said}\": {code:?} {report}");
    }
}

/// A full mutation run fails when the mutation score, caught / (caught +
/// missed) over every mutant of the workspace, is under the floor
/// scripts/check-mutants.sh keeps, a ratchet toward the 95% target; the
/// run's report also gives each file's score against a target of 90%.
/// No test sees a score: none can mutate the code it runs in, so CI does:
/// the Mutants workflow runs cargo-mutants weekly and by hand over the whole
/// workspace, with one pinned compiler and at most four concurrent shards.
/// Runs queue per branch. Each shard first tests the unmutated workspace in
/// a memory-capped cgroup and preserves outcomes for the final score judge.
/// The score floor, baseline and margin remain documented in docs/development.md;
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
/// signal; passes a score exactly at the floor and fails one a hundredth of a
/// point under it; says how many of the missed would meet the target; gives
/// each file's score, worst first and ties by name, with the catches it needs
/// for 90%; offers to raise the floor to a score less the margin, and only
/// when that is above the floor; in diff mode fails on exactly the unlisted
/// misses, timeouts and signals, marking each on its line with its message
/// escaped as a workflow command needs; and refuses a shard that did not
/// finish, one whose unmutated tests failed, shards that overlap, built
/// different targets or hold fewer mutants than the plan listed, a mutant
/// that was only checked, a Failure no signal ended, an outcome cargo-mutants
/// does not write even in code the run did not build, and an allowlist entry
/// with no reason.
#[test]
fn a_full_mutation_run_fails_under_its_floor() {
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
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    let measured = format!("{}, less a margin of {margin} points", percent(floor_bp + margin_bp));
    for claim in [format!("is under {floor}%"), measured, format!("The target is {target}%, and {file_target}% in each file")] {
        assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which scripts/check-mutants.sh gates");
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
    // run's report offered; docs/development.md gives its count, its score and its
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
        assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", as docs/testing.md records");
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
    // Linux runners, as docs/development.md says: the code a run leaves out as not
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
    assert!(
        shards.iter().any(|l| l.trim().starts_with("printf ")
            && l.contains("[profile.test.package.open-max-core]\\nopt-level = 0")
            && l.contains("[profile.test.package.openmax]\\nopt-level = 0")),
        "a shard builds the workspace's crates as the test job does not, so its mutants are judged by another build's tests"
    );
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
    // Scores of exactly the floor and a hundredth of a point under it, each
    // over the fewest mutants that score it exactly (hundredths of a point
    // over 10,000, in lowest terms), of exactly the floor plus the margin as
    // the report truncates it, over the fewest mutants that score that, and
    // of 100%.
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
    let failing = (floor_bp > 0).then(|| boundary("failing", exactly(floor_bp - 1)));
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
        assert_eq!(under, Some(1), "a score a hundredth of a point under the floor of {floor}% passes: {under_report}");
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

/// The first frame, held in CI by its ratio to a baseline binary's start,
/// and on the reference machine by the harness's work in front of it.
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
/// in units of that.
///
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
/// runners can measure another ratio. On a machine of the reference
/// machine's class, an Apple M5 (`first_frame::reference_machine`; CI's
/// runners are not), a release build times 61 launches instead of 21 and
/// also asserts the harness's work, the first frame's median less the
/// baseline's, under its ratchet. The ratchet reaches a measurement whose
/// baseline median is within the 90th percentile of the baseline medians
/// in the first measurement of each of the reference machine's runs,
/// rounded up to 10 us. Past that the machine is slower than the one the
/// ratchet was set on, so where each measurement is slower the work is
/// reported and not asserted. The ratchet is the median work of the
/// reference machine's measurements within that reach, plus a tenth,
/// rounded up to 10 us. It takes the 61 so that the noise alone seldom puts
/// a measurement over the ratchet, in none of the 20 runs below. A burst of
/// other work on the machine slows a measurement or two, while a change
/// that adds work slows every one, so a measurement over the ratchet, or
/// past its reach, is taken again, up to three in all, and the work fails
/// only when each one the ratchet reaches is over. The work's
/// target, 100 us, is not met: it is printed, met or not, in a `MEASURE`
/// line. A debug build prints the measurement, holds the ratio to 20 times
/// the ratchet, which only a hang or a stall many times today's breaks,
/// and asserts no time. Without the baseline binary, which `cargo test`
/// builds only when no target is selected, a release build fails and a
/// debug build skips.
///
/// Measured in release builds, as the ratio of the medians (first frame;
/// baseline). On macos-latest, seven runs: 1.38 to 1.57 (2.37 to 4.43 ms;
/// 1.72 to 2.88 ms), so a ratchet of 2.0. On ubuntu-latest, six runs: 2.03
/// to 2.29 (1.30 to 1.72 ms; 0.58 to 0.83 ms), so 2.9. The startup a change
/// can add and still pass is the ratchet times the baseline, less the first
/// frame: 1.06 to 1.64 ms in the macos-latest runs above and 0.36 to 0.69
/// ms in the ubuntu-latest ones. Painting the frame behind the trust grant,
/// settings and the core, the order before it came first, moved an Apple
/// M5's ratio to 2.00 to 2.46 in ten runs beside other builds, the lowest
/// past the macOS ratchet by 0.07%.
///
/// On the reference machine (Apple M5, macOS 27), 20 runs of 61 launches at
/// one-minute load averages of 7.8 to 9.1. The baseline medians of their
/// first measurements came to 1.174 ms at the 90th percentile, so the
/// ratchet reaches 1.18 ms. Within that reach, in 20 of the 23
/// measurements the runs took: the harness's work 384 to 439 us, median
/// 408 (ratio 1.34 to 1.39; first frame 1.50 to 1.58 ms; baseline 1.10 to
/// 1.17 ms), so a ratchet of 450 us. Interleaved with them, 20 runs of the
/// build before it, which loaded libiconv, walked rebase opcodes and
/// started the signal watcher's thread and runtime in front of the frame
/// (crates/tui/build.rs, `hold_quit_signals`), within the reach in 50 of
/// their 60 measurements: 491 to 539 us, median 515 (ratio 1.42 to 1.47;
/// first frame 1.59 to 1.71 ms), over the ratchet in each run.
///
/// Most of what is left is the dynamic loader's, before `main` runs. It
/// binds every symbol the binary imports at launch, lazy ones included, and
/// finds each by searching libSystem's 39 re-exported libraries in order, so
/// a symbol costs more the later its library comes (libsystem_c is 14th,
/// libsystem_kernel 32nd). It binds 183 for a release build, all from
/// libSystem: the 182 the binary's code calls or reads, all of them used,
/// and `__tlv_bootstrap`, which the linker adds for thread-local variables
/// (`the_binary_names_only_libsystem_caps_its_imports_and_chains_its_fixups`).
/// It binds 69 for the baseline, and 115 of the binary's 183 not for it,
/// each a function but one variable. Timed the same way, to the first byte the
/// process writes, over 1,001 launches of each copy interleaved, with the
/// instructions each process retired read before it was reaped: a copy of
/// the baseline built with this profile's options and chained fixups wrote
/// 276 to 297 us later, in three such runs, given those 115 imports and
/// calling none, and retired 6.46M more instructions, with no more page
/// faults; given 100 functions of libsystem_kernel instead, 302 us and 7.01M
/// more, and 100 of libsystem_c, 120 us and 2.87M. `openmax --version`,
/// which prints from its argument parsing, wrote 304 to 344 us after the
/// baseline, in two such runs, with 6.76M more instructions, so the
/// binary's loading alone is past the work's target with the linking the
/// platform supports; the link options (crates/tui/build.rs) took 57 to 62
/// us, 1.42M to 1.43M instructions and 30 to 31 page faults off that
/// `--version`. The build before them rebased 13,636 pointers by
/// opcodes before `main`: 13,455 in `__DATA_CONST,__const`, one in `__got`
/// and its 180 lazy symbol pointers (`llvm-objdump --macho --rebase`). This
/// one has no lazy symbol pointers, and chains the other 13,456
/// (`dyld_info -fixups`).
#[cfg(unix)]
#[test]
fn first_frame_holds_its_ratio_to_a_baseline_binary() {
    use std::time::Duration;
    // The highest ratio CI's runner for each platform measured, plus a
    // quarter, rounded up to a tenth (above).
    const MACOS_RATCHET: f64 = 2.0;
    const LINUX_RATCHET: f64 = 2.9;
    const RATCHET: f64 = if cfg!(target_os = "macos") { MACOS_RATCHET } else { LINUX_RATCHET };
    // The harness's work, its target on the reference machine, and the
    // ratchet a release build of that machine's class asserts until the
    // target is met, from what that machine measured, in microseconds
    // (above): the ratchet reaches the 90th percentile of the runs' first
    // baseline medians, rounded up to 10 us, and is the median work of the
    // measurements within that reach, plus a tenth, rounded up to 10 us.
    const WORK_TARGET: Duration = Duration::from_micros(100);
    const WORK_MEASURED: u64 = 408;
    const BASELINE_MEASURED: u64 = 1174;
    const WORK_RATCHET: Duration = Duration::from_micros((WORK_MEASURED * 11).div_ceil(10).next_multiple_of(10));
    const WORK_RATCHET_BASELINE: Duration = Duration::from_micros(BASELINE_MEASURED.next_multiple_of(10));
    // The measurements that build takes before the work fails, each one
    // over the ratchet (above).
    const WORK_MEASUREMENTS: usize = 3;
    // Launches, and where the work is asserted more of them, so its median
    // holds still under a ratchet with less room (above).
    const LAUNCHES: usize = 21;
    const ASSERTED_LAUNCHES: usize = 61;
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    for claim in [
        format!("under a ratchet of {MACOS_RATCHET:.1} on macOS and {LINUX_RATCHET:.1} on Linux"),
        format!("it times {ASSERTED_LAUNCHES} launches instead of {LAUNCHES}"),
        format!("under a ratchet of {} us", WORK_RATCHET.as_micros()),
        format!("no slower than {:.2} ms", WORK_RATCHET_BASELINE.as_secs_f64() * 1e3),
        format!("each of the reference machine's runs ({:.3} ms)", BASELINE_MEASURED as f64 / 1e3),
        format!("its measurements within the reach ({WORK_MEASURED} us)"),
        format!("up to {WORK_MEASUREMENTS} measurements in all"),
        format!("The work's target of {} us is not met", WORK_TARGET.as_micros()),
    ] {
        assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which this test holds");
    }
    let Some(baseline) = first_frame::baseline_binary() else { return };
    if !first_frame::alone("first_frame_holds_its_ratio_to_a_baseline_binary") {
        return;
    }
    let reference = first_frame::reference_machine();
    let work_asserted = reference && !cfg!(debug_assertions);
    let launches = if work_asserted { ASSERTED_LAUNCHES } else { LAUNCHES };
    // One measurement: the baseline's starts and the first frames, each
    // launch after one of the baseline, sorted.
    let measure = || {
        first_frame::baseline(&baseline);
        first_frame::launch(0);
        let (mut starts, mut frames) = (Vec::new(), Vec::new());
        for n in 1..=launches {
            starts.push(first_frame::baseline(&baseline));
            frames.push(first_frame::launch(n));
        }
        starts.sort();
        frames.sort();
        (starts, frames)
    };
    let median = |times: &[Duration]| times[launches / 2];
    let work = |(starts, frames): &(Vec<Duration>, Vec<Duration>)| median(frames).saturating_sub(median(starts));
    // Whether the ratchet reaches a measurement, by its baseline's median,
    // and whether the work there is within it.
    let reached = |(starts, _): &(Vec<Duration>, Vec<Duration>)| median(starts) <= WORK_RATCHET_BASELINE;
    let holds = |measurement: &(Vec<Duration>, Vec<Duration>)| reached(measurement) && work(measurement) <= WORK_RATCHET;
    let mut measurements = vec![measure()];
    while work_asserted && measurements.len() < WORK_MEASUREMENTS && !measurements.iter().any(holds) {
        measurements.push(measure());
    }
    // The measurement reported: one the ratchet holds, else the least work
    // it reaches, else the last.
    let (starts, frames) = measurements
        .iter()
        .find(|measurement| holds(measurement))
        .or_else(|| measurements.iter().filter(|measurement| reached(measurement)).min_by_key(|measurement| work(measurement)))
        .unwrap_or_else(|| measurements.last().unwrap());
    let quartiles = |times: &[Duration]| {
        format!("median {:?} (p25 {:?}, p75 {:?})", times[launches / 2], times[launches / 4], times[3 * launches / 4])
    };
    let (start, frame) = (median(starts), median(frames));
    let ratio = frame.as_secs_f64() / start.as_secs_f64();
    let (bound, asserted) = if cfg!(debug_assertions) {
        (20.0 * RATCHET, "debug, asserting 20 times the ratchet")
    } else {
        (RATCHET, "release, asserting the ratchet")
    };
    let report = format!(
        "first frame: {} over {launches} launches; baseline binary (process start): {}; ratio {ratio:.2} (ratchet \
         {RATCHET}; {asserted})",
        quartiles(frames),
        quartiles(starts)
    );
    println!("{report}");
    let works: Vec<Duration> = measurements.iter().map(work).collect();
    let baselines: Vec<Duration> = measurements.iter().map(|(starts, _)| median(starts)).collect();
    let reaches = measurements.iter().any(reached);
    println!(
        "MEASURE first_frame_harness_work {:?} (first frame median less baseline median; target {WORK_TARGET:?} on the \
         reference machine, {}; ratchet {WORK_RATCHET:?} where the baseline's median is within {WORK_RATCHET_BASELINE:?}; \
         {}; work {works:?} at baselines {baselines:?})",
        frame.saturating_sub(start),
        if frame.saturating_sub(start) <= WORK_TARGET { "met" } else { "not met" },
        match (reference, work_asserted, reaches) {
            (true, true, true) => "asserted: the reference machine's class in a release build",
            (true, true, false) => "not asserted: each baseline median slower than the ratchet's",
            (true, false, _) => "not asserted: a debug build",
            (false, _, _) => "not asserted: not the reference machine's class",
        }
    );
    assert!(
        ratio < bound,
        "the first frame takes {ratio:.2} times the baseline binary's start, over {bound}; {report}: frames {frames:?}, baselines {starts:?}"
    );
    assert!(
        !work_asserted || !reaches || measurements.iter().any(holds),
        "the harness's work before the first frame is over its ratchet of {WORK_RATCHET:?} in each measurement the \
         ratchet reaches: work {works:?} at baselines {baselines:?}; {report}: frames {frames:?}, baselines {starts:?}"
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

/// What the dynamic loader does for the binary before `main`, so before the
/// first frame (crates/tui/build.rs): on macOS it opens the libraries the
/// binary names, binds the symbols it imports and fixes up its pointers.
/// The binary names libSystem alone, the one library it binds symbols from,
/// and on Apple silicon its fixups are chained, so the loader walks no
/// rebase opcodes. Below a deployment target of macOS 12 the chains must be
/// in DYLD_CHAINED_PTR_64, the pointer format the linker itself writes for
/// such a target, as it writes DYLD_CHAINED_PTR_64_OFFSET only from macOS
/// 12. That the macOS 11 loader reads them rests on that choice of the
/// linker's: no launch on macOS 11 is tested. On Apple silicon a release
/// build binds at most 183 imports, the count in its chained fixups' header
/// today, built with rustc 1.94.0 or 1.99.1 alike (whose std calls `readdir`
/// where 1.94.0's calls `readdir_r`). The loader searches for each one at
/// every launch, and those the baseline binary does not bind are the larger
/// part of the harness's work before the first frame
/// (`first_frame_holds_its_ratio_to_a_baseline_binary`), so one more fails
/// here, whether a change or a toolchain's std adds it. A debug build binds
/// two more, `__memcpy_chk` and `__memset_chk`, which ring's C code calls
/// when it is not optimized, so it prints its count without asserting it.
/// Read from the Mach-O load commands and the chained fixups' header; a
/// structure gate, so it asserts the rest in every build.
#[cfg(target_os = "macos")]
#[test]
fn the_binary_names_only_libsystem_caps_its_imports_and_chains_its_fixups() {
    use std::io::{Read, Seek, SeekFrom};
    const LC_REQ_DYLD: u32 = 0x8000_0000;
    const LIBRARIES: [u32; 5] = [0xc, 0x18 | LC_REQ_DYLD, 0x1f | LC_REQ_DYLD, 0x20, 0x23 | LC_REQ_DYLD];
    const LC_BUILD_VERSION: u32 = 0x32;
    const LC_DYLD_INFO_ONLY: u32 = 0x22 | LC_REQ_DYLD;
    const LC_DYLD_CHAINED_FIXUPS: u32 = 0x34 | LC_REQ_DYLD;
    const DYLD_CHAINED_PTR_64: u16 = 2;
    // The imports a release build binds on Apple silicon (above).
    const IMPORTS: u32 = 183;
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    let claim = format!("binds at most {IMPORTS} libSystem imports in a release build on Apple silicon");
    assert!(stated.contains(&claim), "docs/development.md does not say \"{claim}\", which this test holds");
    let path = env!("CARGO_BIN_EXE_openmax");
    let mut file = std::fs::File::open(path).unwrap();
    let mut header = [0u8; 32];
    file.read_exact(&mut header).unwrap();
    let word = |bytes: &[u8], at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    assert_eq!(word(&header, 0), 0xfeed_facf, "{path} is not a 64-bit Mach-O");
    let mut commands = vec![0u8; word(&header, 20) as usize];
    file.read_exact(&mut commands).unwrap();
    let (mut at, mut libraries, mut fixups, mut opcodes, mut minos) = (0, Vec::new(), None, false, 0);
    for _ in 0..word(&header, 16) {
        let (command, size) = (word(&commands, at), word(&commands, at + 4) as usize);
        if LIBRARIES.contains(&command) {
            let name = &commands[at + word(&commands, at + 8) as usize..at + size];
            libraries.push(String::from_utf8_lossy(name.split(|&b| b == 0).next().unwrap()).into_owned());
        }
        if command == LC_DYLD_CHAINED_FIXUPS {
            fixups = Some((word(&commands, at + 8), word(&commands, at + 12) as usize));
        }
        if command == LC_BUILD_VERSION {
            minos = word(&commands, at + 12);
        }
        // A dyld_info_command's rebase opcodes, by their size.
        opcodes |= command == LC_DYLD_INFO_ONLY && word(&commands, at + 12) > 0;
        at += size;
    }
    // Each segment's pointer format: dyld_chained_starts_in_segment's, in
    // the chained fixups' starts in image; and the imports the loader binds,
    // the header's imports_count.
    let (formats, imports): (Vec<u16>, Option<u32>) = fixups.map_or_else(
        || (Vec::new(), None),
        |(offset, size)| {
            let mut data = vec![0u8; size];
            file.seek(SeekFrom::Start(offset.into())).unwrap();
            file.read_exact(&mut data).unwrap();
            let starts = word(&data, 4) as usize;
            let formats = (0..word(&data, starts) as usize)
                .map(|segment| word(&data, starts + 4 + 4 * segment) as usize)
                .filter(|&info| info != 0)
                .map(|info| u16::from_le_bytes(data[starts + info + 6..starts + info + 8].try_into().unwrap()))
                .collect();
            (formats, Some(word(&data, 16)))
        },
    );
    let (chained, macos) = (fixups.is_some(), format!("{}.{}", minos >> 16, (minos >> 8) & 0xff));
    println!(
        "MEASURE binary_launch_shape libraries={libraries:?} chained_fixups={chained} pointer_formats={formats:?} \
         imports={imports:?} rebase_opcodes={opcodes} minimum_macos={macos}"
    );
    assert_eq!(libraries, ["/usr/lib/libSystem.B.dylib"], "{path} names a library the loader opens for nothing");
    if cfg!(target_arch = "aarch64") {
        assert!(chained && !opcodes, "{path} has rebase opcodes for the loader to walk instead of chained fixups");
        assert!(
            minos >> 16 >= 12 || (!formats.is_empty() && formats.iter().all(|&format| format == DYLD_CHAINED_PTR_64)),
            "{path} names macOS {macos} but chains its fixups in pointer formats {formats:?}, not {DYLD_CHAINED_PTR_64} \
             (DYLD_CHAINED_PTR_64), the one the linker writes for that release"
        );
        assert!(
            cfg!(debug_assertions) || imports.is_some_and(|count| count <= IMPORTS),
            "{path} binds {} imports, over the {IMPORTS} this test holds; the loader searches for each one at every \
             launch",
            imports.unwrap_or_default()
        );
    }
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

    /// Whether this machine is of the reference machine's class, where the
    /// harness's work is asserted: its CPU is an Apple M5 (the brand macOS
    /// reports is "Apple M5", "Apple M5 Pro" or "Apple M5 Max"). CI's
    /// runners are not.
    pub fn reference_machine() -> bool {
        #[cfg(target_os = "macos")]
        {
            let mut brand = [0u8; 64];
            let mut len = brand.len();
            // SAFETY: the buffer and its length outlive the call, which
            // writes at most `len` bytes and stores how many it wrote.
            let rc = unsafe {
                libc::sysctlbyname(
                    c"machdep.cpu.brand_string".as_ptr(),
                    brand.as_mut_ptr().cast(),
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            let brand = std::ffi::CStr::from_bytes_until_nul(&brand[..len.min(brand.len())]);
            rc == 0 && brand.is_ok_and(|b| matches!(b.to_bytes(), b"Apple M5" | b"Apple M5 Pro" | b"Apple M5 Max"))
        }
        #[cfg(not(target_os = "macos"))]
        false
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
/// `line_count` callable from the next step and the stored note's line
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
    // The write's result carries the receipt, then the stored note's line
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
/// without one directory argument. docs/development.md states the method, held here.
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
    let stated = std::fs::read_to_string(repo().join("docs/development.md")).unwrap();
    for claim in [
        "the share of prompt tokens served from cache over the sessions with more than 100,000 prompt tokens (target 75%)",
        "the share of sessions that cached no prompt token (target 5%)",
        "a gap over 600 s left out as idle (target p90 90 s)",
    ] {
        assert!(stated.contains(claim), "docs/development.md does not say \"{claim}\", which scripts/cache-report.sh reports");
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
    let child = bodies[0].replace(worktree, &root.display().to_string());
    // The bytes between the two requests' longest common prefix and suffix,
    // so one byte inserted counts as one, not as every byte after it.
    let (theirs, own) = (child.as_bytes(), bodies[2].as_bytes());
    let same = theirs.iter().zip(own).take_while(|(theirs, own)| theirs == own).count();
    let tail = theirs[same..].iter().rev().zip(own[same..].iter().rev()).take_while(|(theirs, own)| theirs == own).count();
    let differ = theirs.len().max(own.len()) - same - tail;
    println!(
        "MEASURE a_sub_agent_is_one_flag_and_zero_prompt_tokens {differ} bytes differ between the child's first request, its \
         worktree read as the project, and the first request -p sends ({} and {} bytes, the first {same} the same); target 0",
        theirs.len(),
        own.len()
    );
    assert_eq!(child, bodies[2], "the child's request differs from -p's");
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

/// Peak RSS of a two-round headless turn stays under its ratchet, and is
/// reported against a target of 4 MB, read as 4,000,000 bytes, the stricter
/// of its two readings (4 MiB is 4,194,304 bytes), so the target is never
/// reported met under the looser one.
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
fn headless_turn_peak_rss_stays_under_its_ratchet() {
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
    println!("MEASURE headless_turn_peak_rss_stays_under_its_ratchet {report}");
    if !gated {
        assert!(turn < TARGET * 20, "{report}: {published:?}");
        return;
    }
    let (_, beyond, text) = summary(&pinned);
    let ratchet = format!("with MallocMaxMagazines=1: {text}; ratchet {} KiB beyond the floor", RATCHET / KIB);
    println!("MEASURE headless_turn_peak_rss_stays_under_its_ratchet {ratchet}");
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
/// docs/execution.md states the grace and the bound, held here.
#[cfg(unix)]
#[test]
fn zero_orphans_after_kill_9() {
    let kill = Kill9 { home: true, group: true, removed: false };
    assert_no_orphans_after_kill_9("zero_orphans_after_kill_9", kill, Some(ORPHANS_LISTING));
    let (grace, bound) = (open_max_core::execution::LEFT_BEHIND_GRACE.as_millis(), ORPHANS_GONE_WITHIN.as_millis());
    let stated = std::fs::read_to_string(repo().join("docs/execution.md")).unwrap();
    for claim in [
        format!("SIGKILL if it is still there {grace} ms later"),
        format!("within {bound} ms of the signal, and none of the three before the {grace} ms grace"),
    ] {
        assert!(stated.contains(&claim), "docs/execution.md does not say \"{claim}\", which this test holds");
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

/// When each of `pids` read as gone after `since`, and when `file`, where one
/// is given, read as removed, polled every millisecond for up to `wait`; None
/// for one still found then. The polls wait on the watcher's own timer, which
/// a QoS clamp on this test does not stretch as it would a plain sleep.
#[cfg(unix)]
fn gone_after(pids: &[i32], file: Option<&Path>, since: Instant, wait: Duration) -> (Vec<Option<Duration>>, Option<Duration>) {
    let mut gone = vec![None; pids.len()];
    let mut removed = None;
    while (gone.contains(&None) || (file.is_some() && removed.is_none())) && since.elapsed() < wait {
        for (at, &pid) in gone.iter_mut().zip(pids) {
            if at.is_none() && !alive(pid) {
                *at = Some(since.elapsed());
            }
        }
        if removed.is_none() && file.is_some_and(|file| !file.exists()) {
            removed = Some(since.elapsed());
        }
        open_max_core::execution::sleep_on_time(Duration::from_millis(1));
    }
    (gone, removed)
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
/// which the record's first write starts, then reads the clock, sends
/// SIGKILL and waits for the run, as the shell that started it would. From
/// that reading on, it polls each process of the tree and the watcher every
/// millisecond for up to 5 seconds with `kill(pid, 0)`, which finds a zombie
/// its new parent has not collected, so each reads as gone once it has
/// exited and been collected, and lists each pid with when it did. It also
/// lists when the run was collected, an upper bound on the time the kernel
/// took to end it, which the watcher waits for, and when the record was
/// removed, which the watcher does just after its SIGKILL, so a listing over
/// the bound shows which part took the time: the kernel's end of the run, the
/// watcher's stop, or the end of the tree its SIGKILL began. A release
/// build, as CI's thesis job runs it, asserts the bound; a debug build
/// asserts 20 times it, as the whole suite runs beside it, and prints the
/// times. Its `MEASURE` line, keyed by `name`, gives the count it asserts is
/// zero: the processes still found at the bound. The clock is read before
/// the signal, not once the call returns:
/// the watcher's SIGTERM follows the run's death, which follows the call's
/// start, and its SIGKILL comes no sooner than the grace after its SIGTERM,
/// so no process of the tree can read as gone within the grace of that
/// reading. A reading taken once the call returns can fall after the
/// SIGTERM, as when this thread runs again late, and the grace bound would
/// then fail on correct code.
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
    let killed = Instant::now();
    unsafe { libc::kill(if kill.group { -target } else { target }, libc::SIGKILL) };
    run.wait().unwrap();
    let collected = killed.elapsed();
    let watched: Vec<i32> = tree.iter().copied().chain(watcher).collect();
    let (gone, removed) = gone_after(&watched, Some(&record), killed, Duration::from_secs(5));
    let (bound, build) =
        if cfg!(debug_assertions) { (ORPHANS_GONE_WITHIN * 20, "debug") } else { (ORPHANS_GONE_WITHIN, "release") };
    let ms = |time: Duration| format!("{:.1} ms", time.as_secs_f64() * 1e3);
    let mut listing = format!(
        "{name} on {}, {build} build: SIGKILL to a print run's {}, with no launch since; when the run was collected, \
         when its record was removed, which its watcher does just after its SIGKILL, and when each process of its \
         tool tree, which ignores SIGTERM, and its watcher read as gone, exited and collected, polled every \
         millisecond; bound {}, grace {}\n",
        std::env::consts::OS,
        if kill.group { "process group" } else { "pid" },
        ms(bound),
        ms(LEFT_BEHIND_GRACE)
    );
    listing.push_str(&format!("run {target} collected {} after the kill\n", ms(collected)));
    let removed = removed.map_or("kept 5 s after the kill".into(), |time| format!("removed {} after the kill", ms(time)));
    listing.push_str(&format!("record {removed}\n"));
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
    // The orphans: each process of the tree, and the watcher, still found
    // at the bound. The gate's name states their count, zero, and that
    // count is what it prints and asserts.
    let orphans = gone.iter().filter(|time| time.is_none_or(|time| time >= bound)).count();
    println!(
        "MEASURE {name} {orphans} of {} processes, the tool tree's and its watcher, outlived the killed run by {} \
         (target 0, {build} build)",
        watched.len(),
        ms(bound)
    );
    assert!(watcher.is_some(), "the run's first tool started no watcher\n{listing}");
    assert_eq!(session, watcher, "the watcher does not lead a session of its own\n{listing}");
    assert_eq!(orphans, 0, "a process outlived its killed run by {}\n{listing}", ms(bound));
    assert!(
        gone[..tree.len()].iter().flatten().all(|&time| time >= LEFT_BEHIND_GRACE),
        "a process of the tree, which ignores SIGTERM, was gone before the {} grace had passed\n{listing}",
        ms(LEFT_BEHIND_GRACE)
    );
}

/// A run's watcher ends with the run, at once. A print run whose `bash` call
/// writes its shell's pid and waits for a file to appear: while the call
/// waits, one process names the run's record in its command line, the
/// watcher the call started, whose command line `ps` lists as `openmax
/// --lifeline <record> <start>` on Linux too, where it runs
/// `/proc/self/exe`; and once the test has made the file and the run has
/// answered, exited 0 and been collected, that process reads as gone,
/// exited and collected, polled every millisecond from then, within
/// `LEFT_BEHIND_GRACE` in a release build. So it waited out neither a
/// stop's grace nor its wait for an owner still exiting, which is longer
/// and, measured from the run's collection, could still read just under its
/// own length. A debug build asserts 20 times the bound, as the whole suite
/// runs beside it; each prints the time. docs/execution.md states the bound, held
/// here.
#[cfg(unix)]
#[test]
fn a_runs_watcher_ends_with_the_run() {
    use open_max_core::execution::LEFT_BEHIND_GRACE;
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
    let (gone, _) = gone_after(&watchers, None, Instant::now(), Duration::from_secs(5));
    for (&pid, _) in watchers.iter().zip(&gone).filter(|(_, time)| time.is_none()) {
        unsafe { libc::kill(pid, libc::SIGKILL) };
    }
    if alive(shell) {
        unsafe { libc::kill(-shell, libc::SIGKILL) };
    }
    let _ = std::fs::remove_dir_all(&base);
    let (bound, build) =
        if cfg!(debug_assertions) { (LEFT_BEHIND_GRACE * 20, "debug") } else { (LEFT_BEHIND_GRACE, "release") };
    let ms = |time: Duration| format!("{:.1} ms", time.as_secs_f64() * 1e3);
    for (&pid, time) in watchers.iter().zip(&gone) {
        let when = time.map_or("running 5 s after it".into(), |time| format!("gone {} after it", ms(time)));
        let os = std::env::consts::OS;
        println!("a_runs_watcher_ends_with_the_run on {os}, {build} build: watcher {pid} of a run that exited 0, {when}; bound {}", ms(bound));
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("first answer"), "{stderr}");
    assert_eq!(watchers.len(), 1, "the run's bash call had {} watchers, not one: {watchers:?}", watchers.len());
    let line = format!("openmax --lifeline {} ", record.display());
    assert!(named.as_ref().is_some_and(|named| named.starts_with(&line)), "the watcher's command line is {named:?}, not {line}<start>");
    assert!(gone.iter().all(|time| time.is_some_and(|time| time < bound)), "the watcher outlived its run, which exited 0, by {} or more: {gone:?}", ms(bound));
    let claim = format!("which it outlives by less than {} ms in a release build", LEFT_BEHIND_GRACE.as_millis());
    let stated = std::fs::read_to_string(repo().join("docs/execution.md")).unwrap();
    assert!(stated.contains(&claim), "docs/execution.md does not say \"{claim}\", which this test holds");
}

/// One command from a fresh install answers a local endpoint in under a
/// second, carrying the trust grant, the endpoint and the first turn: the
/// harness's share of the 30 seconds a first real answer is meant to take,
/// nearly all of which are the download and the model's own latency, which
/// no test here measures. The share is the wall time of
/// `openmax --trust-project --base-url URL --model ID --api-key '$NAME'
/// --context-tokens N -p TASK` on an empty HOME against a local endpoint that
/// answers at once, from spawn to exit. Method: one warmup run, then the
/// median of 11 runs, each on its own empty HOME and project, each required
/// to exit 0 with the answer on stdout and the endpoint written to
/// settings.json. Release builds assert the median is under 1 second; debug
/// builds print it and assert under 20 seconds.
#[test]
fn one_command_from_a_fresh_install_answers_locally_in_under_a_second() {
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
    const TARGET: Duration = Duration::from_secs(1);
    let (bound, build) = if cfg!(debug_assertions) { (TARGET * 20, "debug") } else { (TARGET, "release") };
    println!(
        "MEASURE one_command_from_a_fresh_install_answers_locally_in_under_a_second {median:?} median over {} runs \
         ({build} build); bound {bound:?}, the target {TARGET:?} in release",
        runs.len()
    );
    assert!(median < bound, "median {median:?}, runs {runs:?}");
}
