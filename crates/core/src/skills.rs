//! Skills: pi-style progressive disclosure. Only a skill's name and one-line
//! description are ever resident in the system prompt (~15 tokens each); the
//! model reads the full SKILL.md body on demand with read_file. This is the
//! structural answer to adding capability without taxing every prefill.
//!
//! Discovery: `~/.openmax/skills/<name>/SKILL.md` (global), then the
//! project's `.agents/skills/<name>/SKILL.md` — the emerging cross-harness
//! convention — with the project winning on name collision. A SKILL.md
//! carries `---`-delimited frontmatter with `name:` and `description:`;
//! only those two scalar keys are read, so no YAML dependency is needed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Skills beyond this count stop being "a small index" and start being a
/// prompt tax; the sorted head wins and the trailer says what was dropped.
pub const MAX_SKILLS: usize = 50;
pub const MAX_SKILL_DESC_CHARS: usize = 200;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillSpec {
    pub name: String,
    pub description: String,
    /// SKILL.md location, so the model can read_file the full body.
    pub path: PathBuf,
}

/// Discover skills for a project: global first, project overwrites on name
/// collision. Malformed skill files are skipped, never fatal.
pub fn discover(project_root: &Path) -> Vec<SkillSpec> {
    discover_in(&[
        crate::state::default_data_dir().join("skills"),
        project_root.join(".agents").join("skills"),
    ])
}

pub(crate) fn discover_in(dirs: &[PathBuf]) -> Vec<SkillSpec> {
    let mut by_name: HashMap<String, SkillSpec> = HashMap::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut skill_files: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path().join("SKILL.md"))
            .filter(|p| p.is_file())
            .collect();
        skill_files.sort();
        for path in skill_files {
            if let Some(spec) = parse_skill_md(&path) {
                by_name.insert(spec.name.clone(), spec);
            }
        }
    }
    let mut skills: Vec<SkillSpec> = by_name.into_values().collect();
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills.truncate(MAX_SKILLS);
    skills
}

/// Pull `name:` and `description:` out of the frontmatter block. Values may
/// be bare or double-quoted; anything more exotic belongs in the body.
fn parse_skill_md(path: &Path) -> Option<SkillSpec> {
    let text = std::fs::read_to_string(path).ok()?;
    let body = text.strip_prefix("---")?;
    let end = body.find("\n---")?;
    let mut name = None;
    let mut description = None;
    for line in body[..end].lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(v.trim().trim_matches('"').to_string());
        } else if let Some(v) = line.strip_prefix("description:") {
            description = Some(v.trim().trim_matches('"').to_string());
        }
    }
    let name = name.filter(|n| !n.is_empty())?;
    let mut description = description.unwrap_or_default().replace(['\n', '\r'], " ");
    if description.chars().count() > MAX_SKILL_DESC_CHARS {
        description = description.chars().take(MAX_SKILL_DESC_CHARS).collect::<String>() + "…";
    }
    Some(SkillSpec { name, description, path: path.to_path_buf() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-skill-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_skill(root: &Path, dir_name: &str, frontmatter: &str) {
        let dir = root.join(dir_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), format!("---\n{frontmatter}\n---\nFull body here.\n")).unwrap();
    }

    #[test]
    fn parses_minimal_frontmatter() {
        let root = temp_dir("min");
        write_skill(&root, "review", "name: code-review\ndescription: reviews a diff");
        let skills = discover_in(&[root.clone()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "code-review");
        assert_eq!(skills[0].description, "reviews a diff");
        assert!(skills[0].path.ends_with("review/SKILL.md"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn project_skill_wins_and_malformed_is_skipped() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        write_skill(&global, "deploy", "name: deploy\ndescription: global variant");
        write_skill(&project, "deploy", "name: deploy\ndescription: project variant");
        // No frontmatter at all: skipped without failing discovery.
        let broken = project.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("SKILL.md"), "just prose, no frontmatter").unwrap();
        let skills = discover_in(&[global.clone(), project.clone()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "project variant");
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn skills_sorted_capped_and_descriptions_clamped() {
        let root = temp_dir("cap");
        for i in 0..(MAX_SKILLS + 5) {
            write_skill(&root, &format!("s{i:03}"), &format!("name: skill-{i:03}\ndescription: {}", "x".repeat(400)));
        }
        let skills = discover_in(&[root.clone()]);
        assert_eq!(skills.len(), MAX_SKILLS);
        let names: Vec<_> = skills.iter().map(|s| s.name.clone()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted, "must be sorted for deterministic prompts");
        assert!(skills[0].description.chars().count() <= MAX_SKILL_DESC_CHARS + 1);
        let _ = std::fs::remove_dir_all(root);
    }
}
