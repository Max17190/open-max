use std::path::Path;
use std::sync::Arc;

use open_max_core::state::CancelToken;
use open_max_core::tools::{execute, OutputCaps, ToolOutcome};
use serde_json::json;

async fn edit(root: &Path, old: &str, new: &str) -> ToolOutcome {
    execute(
        "edit_file",
        &json!({"path": "report.rs", "old_string": old, "new_string": new}),
        root,
        root,
        OutputCaps::default(),
        Arc::new(CancelToken::default()),
    )
    .await
}

#[tokio::test]
async fn a_multiline_edit_miss_shows_the_changed_line_for_a_direct_retry() {
    let root = std::env::temp_dir().join(format!("openmax-edit-evidence-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("report.rs");
    let original = "struct Report {\n    baseline: usize,\n    observed: usize,\n}\n";
    std::fs::write(&file, original).unwrap();
    let miss = edit(
        &root,
        "struct Report {\n    baseline: usize,\n    expected: usize,\n}",
        "struct Summary {}",
    )
    .await;
    assert!(!miss.ok, "{}", miss.output);
    assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    let exact_line = miss
        .output
        .lines()
        .find_map(|line| line.strip_prefix("3 "))
        .unwrap_or_else(|| panic!("expected the actual line at the mismatch: {}", miss.output));
    let retry = edit(&root, exact_line, "    measured: usize,").await;
    assert!(retry.ok, "{}", retry.output);
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        original.replace("observed", "measured")
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn edit_miss_evidence_is_bounded_and_requires_a_unique_anchor() {
    let root = std::env::temp_dir().join(format!("openmax-edit-evidence-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let file = root.join("report.rs");
    let long_line = format!(" {}\n", "é".repeat(1_000));
    let original = format!(
        "struct Report {{\n{}omitted line\n}}\n",
        long_line.repeat(3)
    );
    let stale = "struct Report {\n    expected: usize,\n}";
    std::fs::write(&file, &original).unwrap();
    let miss = edit(&root, stale, "struct Summary {}").await;
    assert!(!miss.ok, "{}", miss.output);
    assert!(miss
        .output
        .starts_with("old_string not found. First difference at line 2."));
    assert_eq!(miss.output.matches("[line clipped;").count(), 3);
    assert!(miss.output.len() < 1_700, "{}", miss.output.len());
    assert!(!miss.output.contains("omitted line"));
    assert_eq!(std::fs::read_to_string(&file).unwrap(), original);

    let duplicate = original.repeat(2);
    std::fs::write(&file, &duplicate).unwrap();
    let miss = edit(&root, stale, "struct Summary {}").await;
    assert!(
        !miss.ok && miss.output.contains("Closest match is at line 1"),
        "{}",
        miss.output
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), duplicate);

    std::fs::remove_dir_all(root).unwrap();
}
