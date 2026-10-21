//! What CI has to hold for a release tag to ship, checked against the files
//! that define it. A tag builds every target in `[workspace.metadata.dist]`,
//! and one failed build stops the whole release, so each target has to build,
//! and fit its size budget, on the pull request that would break it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    std::fs::read_to_string(repo().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// One table of the workspace Cargo.toml, by its dotted path. Parsed as TOML
/// rather than matched as text: cargo-dist reads every valid spelling of an
/// entry (any spacing, quoting or comment), and one this test failed to match
/// would be skipped without a failure.
fn manifest_table(path: &str) -> toml::Table {
    let mut table: toml::Table = read("Cargo.toml").parse().unwrap_or_else(|e| panic!("Cargo.toml: {e}"));
    for key in path.split('.') {
        table = match table.remove(key) {
            Some(toml::Value::Table(inner)) => inner,
            _ => panic!("Cargo.toml has no [{path}]"),
        };
    }
    table
}

/// Every target a release tag builds and publishes.
fn dist_targets() -> Vec<String> {
    let dist = manifest_table("workspace.metadata.dist");
    let targets: Vec<String> = dist
        .get("targets")
        .and_then(toml::Value::as_array)
        .expect("[workspace.metadata.dist] lists its targets")
        .iter()
        .map(|target| target.as_str().expect("each dist target is a string").to_string())
        .collect();
    assert!(!targets.is_empty(), "no targets in [workspace.metadata.dist]");
    targets
}

/// The lines of one job in a workflow, from its key up to the next key at the
/// jobs' indentation or less (the next job, or the end of `jobs:`).
fn workflow_job(workflow: &str, name: &str) -> Vec<String> {
    let text = read(workflow);
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
    workflow_job(".github/workflows/ci.yml", name)
}

/// `(target, runner)` for each entry of CI's release-build matrix.
fn ci_release_matrix() -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::new();
    for line in ci_job("release-build").iter().map(|line| line.trim()) {
        if let Some(target) = line.strip_prefix("- target: ") {
            entries.push((target.to_string(), String::new()));
        } else if let Some(runner) = line.strip_prefix("runner: ") {
            if let Some(entry) = entries.last_mut() {
                entry.1 = runner.to_string();
            }
        }
    }
    entries
}

/// The cargo-dist release the runners below were read from.
const DIST_VERSION: &str = "0.32.0";

/// The runner `dist plan` assigns each target when the manifest's
/// `github-custom-runners` does not choose one. They live in the dist binary,
/// not in any file here, and a dist bump can move a target to a runner whose
/// OS image or C library differs from the one CI builds it on, so they are
/// recorded here against [`DIST_VERSION`] and re-read from
/// `dist plan --output-format=json` on a bump.
const DIST_DEFAULT_RUNNERS: &[(&str, &str)] = &[
    ("aarch64-apple-darwin", "macos-14"),
    ("x86_64-apple-darwin", "macos-15-intel"),
    ("aarch64-unknown-linux-gnu", "ubuntu-22.04-arm"),
    ("x86_64-unknown-linux-gnu", "ubuntu-22.04"),
    ("aarch64-unknown-linux-musl", "ubuntu-22.04-arm"),
    ("x86_64-unknown-linux-musl", "ubuntu-22.04"),
];

/// The runner a `github-custom-runners` entry releases `target` on.
/// cargo-dist reads a runner name and a table that sets only `runner` the
/// same way, and ignores keys it does not know. A `host` or `container` can
/// also change how the release builds (a cross-compile, or another image and C
/// library), which ci.yml does not mirror, so an entry that sets either fails
/// rather than passing on its runner alone.
fn custom_runner<'a>(target: &str, entry: &'a toml::Value) -> &'a str {
    let runner = match entry {
        toml::Value::Table(table) => {
            for key in ["host", "container"] {
                assert!(
                    !table.contains_key(key),
                    "{target}'s custom runner sets `{key}`, which ci.yml does not mirror: {entry:?}"
                );
            }
            table.get("runner")
        }
        name => Some(name),
    };
    runner
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("{target}'s custom runner names no runner: {entry:?}"))
}

/// CI builds exactly the targets a tag publishes, each on the runner the
/// release builds it on, and gates every binary's size, on every run, with
/// the profile the published binaries are built with.
#[test]
fn ci_builds_and_size_gates_every_release_target() {
    let matrix = ci_release_matrix();
    let mut built: Vec<&str> = matrix.iter().map(|(target, _)| target.as_str()).collect();
    built.sort_unstable();
    let mut published = dist_targets();
    published.sort_unstable();
    assert_eq!(built, published, "CI's release-build matrix is not the set of targets a tag publishes");

    let dist = manifest_table("workspace.metadata.dist");
    assert_eq!(
        dist.get("cargo-dist-version").and_then(toml::Value::as_str),
        Some(DIST_VERSION),
        "cargo-dist changed, and with it maybe the runner a target is released from: re-read DIST_DEFAULT_RUNNERS from `dist plan --output-format=json` and re-sync ci.yml's release-build matrix"
    );
    let custom = dist
        .get("github-custom-runners")
        .map(|runners| runners.as_table().expect("github-custom-runners is a table of target = runner"));
    for (target, runner) in &matrix {
        let released = match custom.and_then(|runners| runners.get(target)) {
            Some(entry) => custom_runner(target, entry),
            None => DIST_DEFAULT_RUNNERS
                .iter()
                .find(|(t, _)| t == target)
                .map(|(_, r)| *r)
                .unwrap_or_else(|| panic!("no release runner recorded for {target}: add the one `dist plan` assigns it to DIST_DEFAULT_RUNNERS")),
        };
        assert_eq!(runner, released, "the release builds {target} on {released}, but CI builds it on {runner}");
    }

    let job = ci_job("release-build");
    let gate = "run: scripts/check-binary-size.sh ${{ matrix.target }} target/${{ matrix.target }}/release/openmax";
    assert!(
        job.iter().any(|line| line.trim() == gate),
        "the release-build job does not gate each binary's size: `{gate}`"
    );
    // A condition on the job, its build step or its gate step skips the gate,
    // so the pull request that broke the target passes. Only the musl-tools
    // install is conditional.
    let conditions: Vec<&str> = job
        .iter()
        .map(|line| line.trim().trim_start_matches("- "))
        .filter(|line| {
            let key = line.split_once(':').map_or("", |(key, _)| key.trim());
            key.trim_matches(|c| c == '"' || c == '\'') == "if"
        })
        .collect();
    assert_eq!(
        conditions,
        ["if: endsWith(matrix.target, '-musl')"],
        "an `if:` on the release-build job or its build or gate step skips the gate, which passes the pull request that broke the target"
    );
    // On the gate's step or its job, this lets an over-budget binary pass.
    let soft = job.iter().map(|line| line.trim()).find(|line| !line.starts_with('#') && line.contains("continue-on-error"));
    assert_eq!(soft, None, "a continue-on-error step or job turns the size gate back into a warning");
    // The gate measures the release profile; dist publishes with its own
    // `dist` profile, so that profile must be release, unchanged.
    let release = toml::Table::from_iter([("inherits".to_string(), toml::Value::from("release"))]);
    assert_eq!(
        manifest_table("profile.dist"),
        release,
        "[profile.dist] overrides the release profile the gate measures"
    );
}

/// The gate passes a binary exactly at its target's budget and fails one a
/// byte over it, for every published target, and fails a target it has no
/// budget for rather than passing it unmeasured.
#[test]
fn the_size_gate_fails_a_binary_over_its_budget() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("openmax-size-gate-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("openmax");
    let gate = |target: &str, len: u64| {
        // Sparse, so a budget-sized file costs no disk.
        std::fs::File::create(&binary).unwrap().set_len(len).unwrap();
        let out = Command::new("sh")
            .arg(repo().join("scripts/check-binary-size.sh"))
            .arg(target)
            .arg(&binary)
            .output()
            .unwrap();
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code(), report)
    };
    // The budget is read from the gate's own report, so this pins the
    // boundary rather than any particular size.
    let budget_in = |report: &str| -> Option<u64> {
        let end = report.find("-byte budget")?;
        let start = report[..end].rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
        report[start..end].parse().ok()
    };

    for target in dist_targets() {
        let (code, report) = gate(&target, 0);
        assert_eq!(code, Some(0), "{target}: an empty binary fails the gate: {report}");
        let budget = budget_in(&report).unwrap_or_else(|| panic!("{target}: no budget in {report:?}"));
        let (code, report) = gate(&target, budget);
        assert_eq!(code, Some(0), "{target}: a binary at its {budget}-byte budget fails: {report}");
        let (code, report) = gate(&target, budget + 1);
        assert_eq!(code, Some(1), "{target}: a binary over its {budget}-byte budget passes: {report}");
        assert!(
            report.contains(&target) && report.contains(&format!("{budget}-byte budget")),
            "{target}: the failure does not name the target and its budget: {report}"
        );
    }

    let (code, report) = gate("riscv64gc-unknown-linux-gnu", 0);
    assert_eq!(code, Some(1), "a target with no budget passes: {report}");
    assert!(report.contains("no size budget for riscv64gc-unknown-linux-gnu"), "{report}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every action CI runs, in ci.yml and in the Mutants workflow, is pinned to
/// a full commit SHA, with the release it corresponds to in a trailing
/// comment. A tag can be moved to other code after review, and these jobs run
/// with the repository's token; a commit cannot change under its SHA.
#[test]
fn ci_pins_every_action_to_a_commit() {
    for workflow in [".github/workflows/ci.yml", ".github/workflows/mutants.yml"] {
        let text = read(workflow);
        let actions: Vec<&str> = text
            .lines()
            .filter_map(|line| line.trim().trim_start_matches("- ").strip_prefix("uses: "))
            .collect();
        assert!(!actions.is_empty(), "{workflow} runs no actions, so this test pins nothing");
        let unpinned: Vec<&str> = actions
            .iter()
            .copied()
            .filter(|action| {
                let Some((_, pin)) = action.split_once('@') else { return true };
                let (sha, version) = pin.split_once(" # ").unwrap_or((pin, ""));
                let full_sha = sha.len() == 40 && sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
                !full_sha || version.trim().is_empty()
            })
            .collect();
        assert!(
            unpinned.is_empty(),
            "{workflow}: actions not pinned to a full commit SHA with their version in a trailing comment: {unpinned:?}"
        );
    }
}

/// The release patches the shell installer after dist builds it and before it
/// is uploaded, on every run of the job. dist's own installer skips the
/// archive check wherever `sha256sum` is missing, which is every macOS before
/// 15, so a release that uploaded it unpatched would install whatever archive
/// it downloaded there; and it ends without the first command, so its user
/// would have to find the endpoint flags in the docs.
#[test]
fn the_release_patches_the_installer_it_uploads() {
    let job = workflow_job(".github/workflows/release.yml", "build-global-artifacts");
    let step = |what: &str, wanted: &dyn Fn(&str) -> bool| {
        job.iter()
            .position(|line| wanted(line.trim()))
            .unwrap_or_else(|| panic!("release.yml's build-global-artifacts job has no {what}"))
    };
    let build = step("dist build of the installer", &|line| {
        line.starts_with("dist build ") && line.contains("--artifacts=global")
    });
    let upload = step("upload step", &|line| line == r#"- name: "Upload artifacts""#);
    for script in ["patch-installer-checksum.sh", "patch-installer-first-command.sh"] {
        let run = format!("run: scripts/{script} target/distrib/openmax-installer.sh");
        let patch = step(&format!("step that runs {script} on the installer"), &|line| line == run);
        assert!(
            build < patch && patch < upload,
            "{script} patches the installer at line {patch} of the job, not between its build (line {build}) and its upload (line {upload})"
        );
    }
    // Either one lets the job upload dist's installer unpatched.
    let soft: Vec<&str> = job
        .iter()
        .map(|line| line.trim().trim_start_matches("- "))
        .filter(|line| {
            let key = line.split_once(':').map_or("", |(key, _)| key.trim());
            matches!(key.trim_matches(|c| c == '"' || c == '\''), "if" | "continue-on-error")
        })
        .collect();
    assert!(soft.is_empty(), "build-global-artifacts can skip the installer patch: {soft:?}");
}

/// The installer a release publishes ends with the first run: a line that
/// names the binary by the absolute path it was installed to, because the
/// PATH change it made reaches only a new shell, and carries the trust grant,
/// the endpoint flags and a first task. The end of install() is dist's own,
/// as the v2026.10.1 installer shipped it, patched by the script the release
/// runs on it. The patch applies exactly once, and fails leaving the file as
/// it was otherwise. The line's flags are ones the binary takes: pasted
/// unchanged, or with any one placeholder left as it is, it is refused before
/// anything is written, and with values for them all (the key's variable
/// exported) it gets as far as the trust grant.
#[test]
fn the_installer_ends_with_the_first_command() {
    let fixture = "crates/tui/tests/fixtures/dist-installer-finish.sh";
    let dist = read(fixture);
    let extracted_from = dist
        .lines()
        .next()
        .and_then(|line| line.split_once("cargo-dist "))
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or_else(|| panic!("{fixture}'s first line names no cargo-dist version"));
    assert_eq!(
        manifest_table("workspace.metadata.dist").get("cargo-dist-version").and_then(toml::Value::as_str),
        Some(extracted_from),
        "cargo-dist changed, and with it maybe the end of its installer: re-extract {fixture} from an installer the new version generates, name that version on its first line, and update scripts/patch-installer-first-command.sh to match it"
    );
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("openmax-installer-finish-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let installer = dir.join("installer.sh");
    let patch = || {
        let out = Command::new("sh")
            .arg(repo().join("scripts/patch-installer-first-command.sh"))
            .arg(&installer)
            .output()
            .unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    std::fs::write(&installer, dist.repeat(2)).unwrap();
    let (patched, report) = patch();
    assert!(!patched && report.contains("more than one end of install()"), "{report}");
    assert_eq!(std::fs::read_to_string(&installer).unwrap(), dist.repeat(2), "a patch that failed changed the installer");
    std::fs::write(&installer, &dist).unwrap();
    let (patched, report) = patch();
    assert!(patched, "the patch does not apply to dist's installer: {report}");
    let once = std::fs::read_to_string(&installer).unwrap();
    let (patched, report) = patch();
    assert!(!patched && report.contains("no end of install()"), "the patch applies twice: {report}");
    assert_eq!(std::fs::read_to_string(&installer).unwrap(), once, "a patch that failed changed the installer");

    // The installer's last line, installed to `bin`, run from `dir`.
    let finish = |bin: &std::path::Path| {
        let out = Command::new("sh")
            .arg("-c")
            .arg(r#". "$1"; _install_dir=$2; _bins=openmax; PRINT_QUIET=0; finish_install"#)
            .arg("sh")
            .arg(&installer)
            .arg(bin)
            .current_dir(&dir)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "{stdout}{}", String::from_utf8_lossy(&out.stderr));
        stdout
    };
    let bin = dir.join("home/.openmax/bin");
    let stdout = finish(&bin);
    let last = stdout.lines().last().unwrap_or_default();
    let command = format!(
        "{}/openmax --trust-project --base-url URL --model MODEL --api-key '$MY_API_KEY' --context-tokens TOKENS -p 'first task'",
        bin.display()
    );
    assert_eq!(last.trim_start(), command, "{stdout}");
    // A relative OPENMAX_INSTALL_DIR installs under the directory the
    // installer ran in, and the line names that absolute path, since the
    // first command runs in the project.
    let stdout = finish(std::path::Path::new("rel/.openmax/bin"));
    let resolved = format!("{}/rel/.openmax/bin/openmax --trust-project ", dir.canonicalize().unwrap().display());
    assert!(stdout.lines().last().unwrap_or_default().trim_start().starts_with(&resolved), "{stdout}");

    // The line's own words, split by a shell, reach the binary built here.
    let run = |tag: &str, line: &str, key: Option<&str>| {
        let home = dir.join(format!("run-{tag}"));
        std::fs::create_dir_all(&home).unwrap();
        let words = line.split_once("/openmax ").unwrap().1;
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(format!(r#"exec "$0" {words}"#))
            .arg(env!("CARGO_BIN_EXE_openmax"))
            .current_dir(&dir)
            .env("HOME", &home)
            .env_remove("OPENMAX_SESSION")
            .env_remove("OPENMAX_HUMAN_ATTEST")
            .env_remove("MY_API_KEY")
            .stdin(std::process::Stdio::null());
        if let Some(key) = key {
            command.env("MY_API_KEY", key);
        }
        let out = command.output().unwrap();
        (out.status.code(), String::from_utf8_lossy(&out.stderr).into_owned(), home.join(".openmax/settings.json").exists())
    };
    let (code, stderr, written) = run("unchanged", last, None);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(!written, "a placeholder reached settings.json: {stderr}");
    // Each word in capitals, filled with a value, and the key by exporting
    // the variable it names.
    let values = [("URL", "http://127.0.0.1:9/v1"), ("MODEL", "m"), ("TOKENS", "8192")];
    let fill_line = |line: &str, left: &str| {
        values
            .iter()
            .filter(|(placeholder, _)| *placeholder != left)
            .fold(line.to_string(), |line, (placeholder, value)| line.replace(&format!(" {placeholder} "), &format!(" {value} ")))
    };
    let fill = |left: &str| fill_line(last, left);
    for (left, _) in values {
        let (code, stderr, written) = run(left, &fill(left), Some("k"));
        assert_eq!(code, Some(2), "{left} left as it is must be refused: {stderr}");
        assert!(!written, "{left} reached settings.json: {stderr}");
    }
    let (code, stderr, written) = run("key-unset", &fill(""), None);
    assert_eq!(code, Some(2), "'$MY_API_KEY' with the variable unset must be refused: {stderr}");
    assert!(stderr.contains("--api-key '$MY_API_KEY' reads an environment variable that is not set") && !written, "{stderr}");
    let (code, stderr, written) = run("filled", &fill(""), Some("k"));
    assert_eq!(code, Some(3), "every flag parses, so the run reaches the trust grant, which needs a terminal: {stderr}");
    assert!(stderr.contains("a trust grant is a human action") && !written, "{stderr}");

    // An install path a shell would split or expand is printed single-quoted,
    // and a backslash in it as it is, so the whole line, pasted as it is,
    // runs the binary installed there.
    let odd = dir.join("home dir/it's $HOME/team\\tools/.openmax/bin");
    std::fs::create_dir_all(&odd).unwrap();
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_openmax"), odd.join("openmax")).unwrap();
    let stdout = finish(&odd);
    let quoted = stdout.lines().last().unwrap_or_default().trim_start();
    let expected = format!("'{}/openmax' --trust-project ", odd.display().to_string().replace('\'', r"'\''"));
    assert!(quoted.starts_with(&expected), "{stdout}");
    let home = dir.join("run-quoted");
    std::fs::create_dir_all(&home).unwrap();
    let out = Command::new("sh")
        .arg("-c")
        .arg(fill_line(quoted, ""))
        .current_dir(&dir)
        .env("HOME", &home)
        .env("MY_API_KEY", "k")
        .env_remove("OPENMAX_SESSION")
        .env_remove("OPENMAX_HUMAN_ATTEST")
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "the quoted path is one word naming the binary: {stderr}");
    assert!(stderr.contains("a trust grant is a human action"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Where `tool` is on this machine, if anywhere: on PATH, or in /usr/sbin or
/// /sbin, where macOS keeps `sha256sum` and a test's PATH may not look.
fn system_tool(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .chain(["/usr/sbin", "/sbin"].map(PathBuf::from))
        .map(|dir| dir.join(tool))
        .find(|candidate| candidate.is_file())
}

/// The installer a release publishes checks the archive it downloads with
/// `sha256sum`, or with `shasum -a 256` where there is no `sha256sum` (macOS
/// 14 and earlier ship only `shasum`), and refuses one whose hash differs. It
/// skips the check, saying so, only on a machine with neither. The check is
/// dist's own, as the v2026.10.1 installer shipped it, patched by the script
/// the release runs on it, and each run sees only the tools named on its PATH.
#[test]
fn the_installer_checks_the_archive_with_sha256sum_or_shasum() {
    let fixture = "crates/tui/tests/fixtures/dist-installer-checksum.sh";
    let dist = read(fixture);
    // Checked against the version the fixture names on its first line, not
    // DIST_VERSION, which a dist bump updates for the runners: only
    // re-extracting the fixture clears this.
    let extracted_from = dist
        .lines()
        .next()
        .and_then(|line| line.split_once("cargo-dist "))
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or_else(|| panic!("{fixture}'s first line names no cargo-dist version"));
    assert_eq!(
        manifest_table("workspace.metadata.dist").get("cargo-dist-version").and_then(toml::Value::as_str),
        Some(extracted_from),
        "cargo-dist changed, and with it maybe the installer's check: re-extract {fixture} from an installer the new version generates, name that version on its first line, and update scripts/patch-installer-checksum.sh to match it"
    );
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("openmax-installer-check-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let installer = dir.join("installer.sh");
    // Patching only the first of two checks would leave the second skipping,
    // so an installer with two fails as one with none does.
    std::fs::write(&installer, dist.repeat(2)).unwrap();
    let patch = || {
        let out = Command::new("sh")
            .arg(repo().join("scripts/patch-installer-checksum.sh"))
            .arg(&installer)
            .output()
            .unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    let (patched, report) = patch();
    assert!(!patched, "the patch applies to an installer with two sha256 checks");
    assert!(report.contains("more than one sha256 check"), "{report}");
    assert_eq!(std::fs::read_to_string(&installer).unwrap(), dist.repeat(2), "a patch that failed changed the installer");

    std::fs::write(&installer, &dist).unwrap();
    let (patched, report) = patch();
    assert!(patched, "the patch does not apply to dist's installer: {report}");
    // The patched installer no longer has dist's check, so a second run is
    // what a release built by a dist whose check changed would see: it fails
    // and leaves the installer as it was.
    let once = std::fs::read_to_string(&installer).unwrap();
    let (patched, report) = patch();
    assert!(!patched, "the patch applies twice, so it would apply to a check it was not written for");
    assert!(report.contains("no sha256 check"), "{report}");
    assert_eq!(std::fs::read_to_string(&installer).unwrap(), once, "a patch that failed changed the installer");

    let archive = dir.join("input.tar.xz");
    std::fs::write(&archive, "openmax archive\n").unwrap();
    // The SHA-256 of the archive's contents above.
    let right = "8c334d8230bd31265cd8f1d920694c85b6e148cbfa496a59386c0ca6655e3911";
    let wrong = "0".repeat(64);
    let check = |tools: &str, hashers: &[&str], want: &str| {
        let bin = dir.join(tools);
        if !bin.exists() {
            std::fs::create_dir(&bin).unwrap();
            for tool in hashers.iter().copied().chain(["awk"]) {
                let found = system_tool(tool)
                    .unwrap_or_else(|| panic!("no `{tool}` on this machine, and this test runs the installer's check with it"));
                std::os::unix::fs::symlink(found, bin.join(tool)).unwrap();
            }
        }
        let out = Command::new("sh")
            .arg("-c")
            .arg(r#". "$1"; PATH=$2; verify_checksum "$3" sha256 "$4""#)
            .arg("sh")
            .arg(&installer)
            .arg(&bin)
            .arg(&archive)
            .arg(want)
            .env("PRINT_QUIET", "0")
            .output()
            .unwrap();
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code(), report)
    };

    // CI has both hashers and runs both cases. A machine can lack either
    // (macOS 14 and earlier have no `sha256sum`, a Linux without perl no
    // `shasum`), and there the case it cannot run is skipped, saying so.
    let under_ci = std::env::var_os("CI").is_some();
    for hasher in ["sha256sum", "shasum"] {
        if system_tool(hasher).is_none() {
            assert!(!under_ci, "no `{hasher}` on this CI runner, so the installer's check with it goes untested");
            eprintln!("skipping the installer's check with only `{hasher}`: there is no `{hasher}` on this machine");
            continue;
        }
        let tools = format!("only-{hasher}");
        let (code, report) = check(&tools, &[hasher], right);
        assert_eq!(code, Some(0), "with only {hasher}, the published archive fails the check: {report}");
        assert!(!report.contains("skipping"), "with only {hasher}, the check is skipped: {report}");
        let (code, report) = check(&tools, &[hasher], &wrong);
        assert_eq!(code, Some(1), "with only {hasher}, an archive with another hash is accepted: {report}");
        assert!(
            report.contains("checksum mismatch") && report.contains(right),
            "with only {hasher}, the refusal does not name the hash it computed: {report}"
        );
    }
    let (code, report) = check("neither", &[], &wrong);
    assert_eq!(code, Some(0), "with neither hasher, the install stops: {report}");
    assert!(
        report.contains("skipping sha256 checksum verification"),
        "with neither hasher, the check is skipped without saying so: {report}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
