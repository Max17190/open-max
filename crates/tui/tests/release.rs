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

/// The body of one `[header]` table in the workspace Cargo.toml, up to the
/// next table.
fn manifest_table(header: &str) -> String {
    let manifest = read("Cargo.toml");
    let start = manifest
        .find(&format!("\n[{header}]\n"))
        .unwrap_or_else(|| panic!("Cargo.toml has no [{header}]"));
    let body = &manifest[start + header.len() + 4..];
    body[..body.find("\n[").unwrap_or(body.len())].to_string()
}

/// Every target a release tag builds and publishes.
fn dist_targets() -> Vec<String> {
    let dist = manifest_table("workspace.metadata.dist");
    let list = dist
        .split_once("\ntargets = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .expect("[workspace.metadata.dist] lists its targets")
        .0;
    let targets: Vec<String> = list.split('"').skip(1).step_by(2).map(str::to_string).collect();
    assert!(!targets.is_empty(), "no targets in [workspace.metadata.dist]");
    targets
}

/// `(target, runner)` for each entry of CI's release-build matrix.
fn ci_release_matrix() -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::new();
    for line in read(".github/workflows/ci.yml").lines().map(str::trim) {
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

/// CI builds exactly the targets a tag publishes, on the runner the release
/// builds each one on where the manifest chooses it, and gates every binary's
/// size with the profile the published binaries are built with.
#[test]
fn ci_builds_and_size_gates_every_release_target() {
    let matrix = ci_release_matrix();
    let mut built: Vec<&str> = matrix.iter().map(|(target, _)| target.as_str()).collect();
    built.sort_unstable();
    let mut published = dist_targets();
    published.sort_unstable();
    assert_eq!(built, published, "CI's release-build matrix is not the set of targets a tag publishes");

    for line in manifest_table("workspace.metadata.dist.github-custom-runners").lines() {
        let Some((target, runner)) = line.split_once(" = ").filter(|_| !line.starts_with('#')) else {
            continue;
        };
        let runner = runner.trim_matches('"');
        let ci = matrix.iter().find(|(t, _)| t == target).map(|(_, r)| r.as_str());
        assert_eq!(ci, Some(runner), "the release builds {target} on {runner}");
    }

    let ci = read(".github/workflows/ci.yml");
    let gate = "run: scripts/check-binary-size.sh ${{ matrix.target }} target/${{ matrix.target }}/release/openmax";
    assert!(
        ci.lines().any(|line| line.trim() == gate),
        "the release-build job does not gate each binary's size: `{gate}`"
    );
    // On the gate's step or its job, this lets an over-budget binary pass.
    let soft = ci.lines().map(str::trim).find(|line| !line.starts_with('#') && line.contains("continue-on-error"));
    assert_eq!(soft, None, "a continue-on-error step or job turns the size gate back into a warning");
    // The gate measures the release profile; dist publishes with its own
    // `dist` profile, so that profile must be release, unchanged.
    let dist_profile: Vec<String> = manifest_table("profile.dist")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect();
    assert_eq!(dist_profile, ["inherits = \"release\""], "[profile.dist] overrides the release profile the gate measures");
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
