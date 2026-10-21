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

/// The thesis claims are gated in CI: every thesis test runs there in
/// release and fails the pull request, but for the ones a runner reports. A
/// test that times the harness asserts its bound only in a release build,
/// and the test job builds in debug, so ci.yml's `thesis` job builds in
/// release, on each of the test job's runners, every test target named
/// thesis* in the workspace (cargo matches the glob against target names,
/// and a file under a crate's tests/ is a target named after its stem), the
/// binary's `thesis_frames` module and the first-frame test's baseline
/// binary, an example that `cargo test` leaves unbuilt once a target is
/// selected. Its reports step, the only step in ci.yml allowed to fail, runs
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
/// Nothing in the job skips a step (no `if:`), and nothing turns on debug
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
    let skipped: Vec<&&str> = steps.iter().flatten().filter(|line| line.starts_with("if:")).collect();
    assert!(skipped.is_empty(), "the thesis job, or a step of it, can be skipped: {skipped:?}");
    // Anywhere but the reports step, this turns a gate, this job's or the
    // test job's, into a warning.
    let ci = std::fs::read_to_string(repo().join(".github/workflows/ci.yml")).unwrap();
    let soft: Vec<&str> =
        ci.lines().map(str::trim).filter(|line| !line.starts_with('#') && line.contains("continue-on-error")).collect();
    assert_eq!(soft, ["continue-on-error: true"], "ci.yml has a step or job allowed to fail besides the thesis reports step");
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
