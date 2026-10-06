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

/// The lines of one job in ci.yml, from its key up to the next key at the
/// jobs' indentation or less (the next job, or the end of `jobs:`).
fn ci_job(name: &str) -> Vec<String> {
    let ci = read(".github/workflows/ci.yml");
    let key = format!("  {name}:");
    let mut lines = ci.lines().skip_while(|line| line.trim_end() != key);
    assert!(lines.next().is_some(), "ci.yml has no `{name}` job");
    lines
        .take_while(|line| {
            let body = line.trim_start();
            body.is_empty() || body.starts_with('#') || line.len() - body.len() > 2
        })
        .map(str::to_string)
        .collect()
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
            // A table here also sets a host or container, which ci.yml does
            // not mirror, so it fails rather than passing on the runner alone.
            Some(custom) => custom
                .as_str()
                .unwrap_or_else(|| panic!("{target}'s custom runner is not a runner name, which ci.yml cannot mirror: {custom:?}")),
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
    let ci = read(".github/workflows/ci.yml");
    let soft = ci.lines().map(str::trim).find(|line| !line.starts_with('#') && line.contains("continue-on-error"));
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

/// Every action CI runs is pinned to a full commit SHA, with the release it
/// corresponds to in a trailing comment. A tag can be moved to other code
/// after review, and these jobs run with the repository's token; a commit
/// cannot change under its SHA.
#[test]
fn ci_pins_every_action_to_a_commit() {
    let ci = read(".github/workflows/ci.yml");
    let actions: Vec<&str> = ci
        .lines()
        .filter_map(|line| line.trim().trim_start_matches("- ").strip_prefix("uses: "))
        .collect();
    assert!(!actions.is_empty(), "ci.yml runs no actions, so this test pins nothing");
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
        "actions not pinned to a full commit SHA with their version in a trailing comment: {unpinned:?}"
    );
}
