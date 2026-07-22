//! `openmax --check`: validate every extension surface and say why a file
//! would be ignored, instead of the silent skip the agent loop does. This is
//! how the agent verifies its own self-modifications (run it via bash after
//! writing an extension file) and how a human debugs a hook that "does
//! nothing" or a permissions file that fails closed.

use std::path::{Path, PathBuf};

use crate::tools;

#[derive(Debug)]
pub struct Finding {
    /// Surface: tool, skill, template, hook, permissions, or providers.
    pub kind: &'static str,
    pub path: PathBuf,
    /// Ok holds a short summary (name, event, rule count); Err the reason
    /// the agent loop ignores or fails closed on this file.
    pub status: Result<String, String>,
}

pub fn has_errors(findings: &[Finding]) -> bool {
    findings.iter().any(|f| f.status.is_err())
}

/// Validate all extension files for a project (global + project dirs).
/// Missing dirs and files contribute nothing; an empty report means an empty
/// (and healthy) configuration.
pub fn check(project_root: &Path) -> Vec<Finding> {
    check_at(project_root, &crate::state::default_data_dir())
}

fn check_at(project_root: &Path, data_dir: &Path) -> Vec<Finding> {
    let mut findings = Vec::new();

    for dir in crate::registry::external_tool_dirs(project_root) {
        for path in files_with_extension(&dir, "toml") {
            let status = crate::registry::parse_tool_file(&path).and_then(|spec| {
                if tools::TOOL_NAMES.contains(&spec.name.as_str()) {
                    Err(format!("'{}' shadows a built-in tool and is ignored", spec.name))
                } else {
                    Ok(format!("tool '{}'", spec.name))
                }
            });
            findings.push(Finding { kind: "tool", path, status });
        }
    }

    for dir in crate::skills::skill_dirs(project_root) {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut paths: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path().join("SKILL.md"))
            .filter(|p| p.is_file())
            .collect();
        paths.sort();
        for path in paths {
            let status = crate::skills::parse_skill_md(&path).map(|s| format!("skill '{}'", s.name));
            findings.push(Finding { kind: "skill", path, status });
        }
    }

    for dir in crate::templates::template_dirs(project_root) {
        for path in files_with_extension(&dir, "md") {
            let status =
                crate::templates::parse_template(&path).map(|t| format!("template /{}", t.name));
            findings.push(Finding { kind: "template", path, status });
        }
    }

    for dir in crate::hooks::hook_dirs(project_root) {
        for path in files_with_extension(&dir, "toml") {
            let status = crate::hooks::parse_hook_file(&path)
                .map(|h| format!("hook on {}", h.event.as_str()));
            findings.push(Finding { kind: "hook", path, status });
        }
    }

    for path in crate::permissions::permission_files(project_root) {
        if let Some(result) = crate::permissions::check_file(&path) {
            findings.push(Finding {
                kind: "permissions",
                path,
                status: result.map(|n| format!("{n} rules")),
            });
        }
    }

    let path = crate::providers::providers_path(data_dir);
    if let Some(result) = crate::providers::check_file(&path) {
        findings.push(Finding {
            kind: "providers",
            path,
            status: result.map(|n| format!("{n} providers")),
        });
    }

    findings
}

fn files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == ext) && p.is_file())
        .collect();
    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_project() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-doctor-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reports_valid_invalid_and_shadowing_files() {
        let root = temp_project();
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        std::fs::create_dir_all(root.join(".openmax/hooks")).unwrap();
        std::fs::create_dir_all(root.join(".agents/skills/good")).unwrap();
        std::fs::create_dir_all(root.join(".agents/prompts")).unwrap();

        std::fs::write(
            root.join(".openmax/tools/good.toml"),
            "name = \"deploy\"\ndescription = \"d\"\ncommand = \"/bin/true\"\n",
        )
        .unwrap();
        std::fs::write(root.join(".openmax/tools/broken.toml"), "name = [not toml").unwrap();
        std::fs::write(
            root.join(".openmax/tools/shadow.toml"),
            "name = \"bash\"\ndescription = \"d\"\ncommand = \"/bin/true\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join(".agents/skills/good/SKILL.md"),
            "---\nname: good\ndescription: fine\n---\nbody\n",
        )
        .unwrap();
        std::fs::write(root.join(".agents/prompts/review.md"), "Review the diff.\n").unwrap();
        std::fs::write(
            root.join(".openmax/hooks/bad-event.toml"),
            "event = \"on_fire\"\ncommand = \"/bin/true\"\n",
        )
        .unwrap();
        std::fs::write(root.join(".openmax/permissions.toml"), "[[rule]]\n").unwrap();

        let findings = check(&root);
        assert!(has_errors(&findings));
        let get = |name: &str| {
            findings
                .iter()
                .find(|f| f.path.to_string_lossy().contains(name))
                .unwrap_or_else(|| panic!("no finding for {name}"))
        };
        assert!(get("good.toml").status.as_ref().unwrap().contains("deploy"));
        assert!(get("broken.toml").status.as_ref().unwrap_err().contains("invalid TOML"));
        assert!(get("shadow.toml").status.as_ref().unwrap_err().contains("shadows a built-in"));
        assert!(get("SKILL.md").status.is_ok());
        assert!(get("review.md").status.as_ref().unwrap().contains("/review"));
        assert!(get("bad-event.toml").status.as_ref().unwrap_err().contains("unknown event"));
        assert!(
            get("permissions.toml").status.as_ref().unwrap_err().contains("malformed"),
            "typo'd [[rule]] must be reported as the fail-closed reason"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn empty_project_reports_nothing() {
        let root = temp_project();
        // Global dirs may hold the developer's real extensions; only assert
        // that nothing from this project root shows up.
        let root_str = root.to_string_lossy().to_string();
        assert!(!check(&root).iter().any(|f| f.path.to_string_lossy().contains(&root_str)));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn reports_global_provider_configuration() {
        let root = temp_project();
        let data = temp_project();
        std::fs::write(
            data.join("providers.json"),
            r#"{"providers":{"local":{"base_url":"http://127.0.0.1:11434/v1"}}}"#,
        )
        .unwrap();

        let findings = check_at(&root, &data);
        let provider = findings.iter().find(|f| f.kind == "providers").unwrap();
        assert_eq!(provider.status.as_ref().unwrap(), "1 providers");

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }
}
