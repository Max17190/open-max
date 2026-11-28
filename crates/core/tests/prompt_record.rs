//! The committed record of the built-in prompt and tool schemas.
//!
//! `crates/core/tests/fixtures/built-in-prompt/` holds the bytes every
//! session sends before anything of the project's own: the base rules of the
//! system prompt, with the project path written as `{root}`, and the built-in
//! tools array as the Chat Completions wire carries it, once as sent to an
//! endpoint without image input and once as sent to one with it. The bash
//! tool's description differs between Linux and macOS, so each has its own
//! tools records, and the test on each checks its own. The test fails when
//! the code builds other bytes, so a change to them updates the record in the
//! same pull request, where its diff shows what changed, and
//! `scripts/prompt-record.sh` compares the record between two releases for
//! the release notes.
//!
//! To update this platform's records after a deliberate change:
//! `OPENMAX_WRITE_PROMPT_RECORD=1 cargo test -p open-max-core --test prompt_record`.
//! The failure prints the bytes it built, for the other platform's.
//!
//! This file is its own test binary because it sets `tools::IMAGE_INPUT`,
//! a process-wide switch other tests read.
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use open_max_core::prompt::{system_prompt_with_breakdown, MIN_AGENTS_MD_BYTES};
use open_max_core::registry::Registry;
use open_max_core::tools::IMAGE_INPUT;

const WRITE: &str = "OPENMAX_WRITE_PROMPT_RECORD";

fn record_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/built-in-prompt")
}

/// The base rules a new session freezes in an empty project, its path
/// written as `{root}`.
fn base_rules() -> String {
    let dir = std::env::temp_dir().join(format!("omx-prompt-record-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let registry = Registry::builtin_only();
    let (prompt, breakdown) = system_prompt_with_breakdown(&dir, &registry, MIN_AGENTS_MD_BYTES);
    let _ = std::fs::remove_dir_all(&dir);
    let (name, len) = &breakdown.components[0];
    assert_eq!(name, "base rules", "the prompt opens with its base rules");
    let root = dir.to_string_lossy();
    let base = &prompt[..*len];
    assert_eq!(base.matches(&*root).count(), 1, "the base rules name the project path once:\n{base}");
    base.replace(&*root, "{root}")
}

/// The built-in tools array's wire bytes, as sent with image input `images`.
fn tools(images: bool) -> String {
    IMAGE_INPUT.store(images, Ordering::Relaxed);
    let wire = Registry::builtin_only().tool_schemas_wire().to_string();
    IMAGE_INPUT.store(false, Ordering::Relaxed);
    wire
}

#[test]
fn the_built_in_prompt_and_tools_match_their_committed_record() {
    let platform = if cfg!(target_os = "linux") { "linux" } else { "macos" };
    let built = [
        ("system.txt".to_string(), base_rules()),
        (format!("tools-{platform}.json"), tools(false)),
        (format!("tools-{platform}-image-input.json"), tools(true)),
    ];
    let dir = record_dir();
    if std::env::var_os(WRITE).is_some() {
        std::fs::create_dir_all(&dir).unwrap();
        for (file, bytes) in &built {
            std::fs::write(dir.join(file), bytes).unwrap();
        }
    }
    let mut stale = Vec::new();
    for (file, bytes) in &built {
        let recorded = std::fs::read_to_string(dir.join(file)).unwrap_or_default();
        if &recorded != bytes {
            stale.push(format!("{file}: recorded {} bytes, built {} bytes:\n{bytes}", recorded.len(), bytes.len()));
        }
    }
    assert!(
        stale.is_empty(),
        "the built-in prompt or tool schemas differ from their record in {}:\n{}\nIf the change is \
         deliberate, update the record in the same change with \
         `{WRITE}=1 cargo test -p open-max-core --test prompt_record` and commit it.",
        dir.display(),
        stale.join("\n"),
    );
}
