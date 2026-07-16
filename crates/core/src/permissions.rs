//! Optional declarative permission rules from `permissions.toml`.
//! Empty discovery is free: missing files mean zero behavior change.
//! Order: hooks pre → permissions → approval_mode → execute → hooks post.

use std::path::{Path, PathBuf};

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

use crate::tools;

/// Result of evaluating permission rules against a tool call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PermissionDecision {
    /// No rule matched; existing approval_mode logic applies.
    Default,
    Allow,
    Deny { reason: String },
    Ask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Effect {
    Allow,
    Deny,
    Ask,
}

#[derive(Clone, Debug)]
struct Rule {
    effect: Effect,
    tool: String,
    /// Compiled optional arg filter. Invalid patterns are dropped at load.
    arg_regex: Option<Regex>,
}

/// Permission rules for the current project. Loaded once per agent turn.
#[derive(Clone, Debug, Default)]
pub struct Permissions {
    rules: Vec<Rule>,
}

#[derive(Deserialize)]
struct PermissionsFile {
    #[serde(default)]
    rules: Vec<RuleFile>,
}

#[derive(Deserialize)]
struct RuleFile {
    effect: String,
    tool: String,
    #[serde(default)]
    arg_regex: Option<String>,
}

impl Permissions {
    /// Discover rules under project `.openmax/permissions.toml` then global
    /// `~/.openmax/permissions.toml`. Project rules are listed first so they win.
    pub fn discover(project_root: &Path) -> Self {
        Self::from_files(&permission_files(project_root))
    }

    fn from_files(paths: &[PathBuf]) -> Self {
        let mut rules = Vec::new();
        for path in paths {
            if let Some(mut loaded) = load_file(path) {
                rules.append(&mut loaded);
            }
        }
        Self { rules }
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// First matching rule wins. Missing rules → [`PermissionDecision::Default`].
    pub fn evaluate(&self, tool: &str, args: &Value) -> PermissionDecision {
        let haystack = arg_haystack(tool, args);
        for rule in &self.rules {
            if rule.tool != tool {
                continue;
            }
            if let Some(re) = &rule.arg_regex {
                if !re.is_match(&haystack) {
                    continue;
                }
            }
            return match rule.effect {
                Effect::Allow => PermissionDecision::Allow,
                Effect::Deny => PermissionDecision::Deny {
                    reason: format!("permission rule denied tool {tool}"),
                },
                Effect::Ask => PermissionDecision::Ask,
            };
        }
        PermissionDecision::Default
    }
}

fn permission_files(project_root: &Path) -> Vec<PathBuf> {
    let mut files = vec![project_root.join(".openmax").join("permissions.toml")];
    if let Some(home) = std::env::var_os("HOME") {
        files.push(PathBuf::from(home).join(".openmax").join("permissions.toml"));
    }
    files
}

fn load_file(path: &Path) -> Option<Vec<Rule>> {
    if !path.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    let file: PermissionsFile = toml::from_str(&text).ok()?;
    let mut rules = Vec::with_capacity(file.rules.len());
    for raw in file.rules {
        let tool = raw.tool.trim().to_string();
        if tool.is_empty() {
            continue;
        }
        let effect = match raw.effect.trim() {
            "allow" => Effect::Allow,
            "deny" => Effect::Deny,
            "ask" => Effect::Ask,
            _ => continue,
        };
        let arg_regex = match raw.arg_regex.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            None => None,
            Some(pat) => match Regex::new(pat) {
                Ok(re) => Some(re),
                // Invalid regex: skip the rule (do not panic; do not match).
                Err(_) => continue,
            },
        };
        rules.push(Rule {
            effect,
            tool,
            arg_regex,
        });
    }
    Some(rules)
}

/// Primary argument string used for optional `arg_regex` matching.
fn arg_haystack(tool: &str, args: &Value) -> String {
    match tool {
        "bash" => args["command"].as_str().unwrap_or("").to_string(),
        "write_file" | "edit_file" | "read_file" | "list_dir" => {
            args["path"].as_str().unwrap_or("").to_string()
        }
        "glob" | "grep" => args["pattern"].as_str().unwrap_or("").to_string(),
        "task" => tools::summarize_call("task", args),
        _ => serde_json::to_string(args).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write_perms(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-perms-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_file_is_default() {
        let tmp = tempfile_dir();
        let perms = Permissions::discover(&tmp);
        assert!(perms.is_empty());
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "rm -rf /"})),
            PermissionDecision::Default
        );
    }

    #[test]
    fn deny_bash_rm_rf() {
        let tmp = tempfile_dir();
        write_perms(
            &tmp.join(".openmax").join("permissions.toml"),
            r#"
[[rules]]
effect = "deny"
tool = "bash"
arg_regex = "rm\\s+-rf"
"#,
        );
        let perms = Permissions::discover(&tmp);
        match perms.evaluate("bash", &json!({"command": "rm -rf /tmp/foo"})) {
            PermissionDecision::Deny { reason } => {
                assert!(reason.contains("bash"), "{reason}");
            }
            other => panic!("expected Deny, got {other:?}"),
        }
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "ls"})),
            PermissionDecision::Default
        );
    }

    #[test]
    fn allow_cargo_test() {
        let tmp = tempfile_dir();
        write_perms(
            &tmp.join(".openmax").join("permissions.toml"),
            r#"
[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^cargo (test|check|build)"
"#,
        );
        let perms = Permissions::discover(&tmp);
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "cargo test -p foo"})),
            PermissionDecision::Allow
        );
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "cargo publish"})),
            PermissionDecision::Default
        );
    }

    #[test]
    fn first_match_project_before_global() {
        let tmp = tempfile_dir();
        let project = tmp.join("project-permissions.toml");
        let global = tmp.join("global-permissions.toml");
        write_perms(
            &project,
            r#"
[[rules]]
effect = "deny"
tool = "bash"
arg_regex = "cargo"
"#,
        );
        write_perms(
            &global,
            r#"
[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "cargo"
"#,
        );

        // Same merge order as discover: project file first, then global.
        let perms = Permissions::from_files(&[project, global]);
        match perms.evaluate("bash", &json!({"command": "cargo test"})) {
            PermissionDecision::Deny { .. } => {}
            other => panic!("project deny should win over global allow, got {other:?}"),
        }
    }

    #[test]
    fn invalid_regex_skipped() {
        let tmp = tempfile_dir();
        write_perms(
            &tmp.join(".openmax").join("permissions.toml"),
            r#"
[[rules]]
effect = "deny"
tool = "bash"
arg_regex = "(unclosed"

[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^ls"
"#,
        );
        let perms = Permissions::discover(&tmp);
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "ls -la"})),
            PermissionDecision::Allow
        );
        // The invalid deny rule must not match everything.
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "echo hi"})),
            PermissionDecision::Default
        );
    }

    #[test]
    fn tool_only_rule_matches_any_args() {
        let tmp = tempfile_dir();
        write_perms(
            &tmp.join(".openmax").join("permissions.toml"),
            r#"
[[rules]]
effect = "ask"
tool = "write_file"
"#,
        );
        let perms = Permissions::discover(&tmp);
        assert_eq!(
            perms.evaluate("write_file", &json!({"path": "a.rs", "content": "x"})),
            PermissionDecision::Ask
        );
        assert_eq!(
            perms.evaluate("write_file", &json!({"path": "b.rs"})),
            PermissionDecision::Ask
        );
        assert_eq!(
            perms.evaluate("read_file", &json!({"path": "a.rs"})),
            PermissionDecision::Default
        );
    }
}
