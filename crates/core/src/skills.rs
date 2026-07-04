//! Skills: pi-style progressive disclosure. Only a skill's name and one-line
//! description are ever resident in the system prompt (~15 tokens each); the
//! model reads the full SKILL.md body on demand with read_file. Discovery is
//! implemented alongside the prompt section; this module starts with the spec
//! type shared with the session manifest.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillSpec {
    pub name: String,
    pub description: String,
    /// SKILL.md location, so the model can read_file the full body.
    pub path: PathBuf,
}

/// Discover skills for a project. Placeholder until the skills step lands:
/// returns no skills, which keeps the prompt byte-identical to a build
/// without the extensibility layer.
pub fn discover(_project_root: &std::path::Path) -> Vec<SkillSpec> {
    Vec::new()
}
