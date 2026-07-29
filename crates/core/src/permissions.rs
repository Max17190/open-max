//! Optional declarative permission rules from `permissions.toml`.
//! Empty discovery is free: missing files mean zero behavior change.
//! Order: hooks pre → permissions → approval_mode → execute → hooks post.

use std::path::{Path, PathBuf};

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

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
    /// True when an existing permissions file could not be parsed. Evaluate
    /// then denies every tool so a broken policy cannot fail open.
    fail_closed: bool,
    fail_closed_reason: Option<String>,
    /// The file that failed to parse, and the root that project-relative tool
    /// paths resolve against. Rewriting exactly that file stays reachable so a
    /// single bad rule cannot lock every repair out of the session.
    invalid_path: Option<PathBuf>,
    project_root: PathBuf,
}

/// Reject unknown top-level keys so `[rule]` / `[[rule]]` typos cannot load as empty policy.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PermissionsFile {
    #[serde(default)]
    rules: Vec<RuleFile>,
}

/// Reject unknown keys so a misspelled `arg_regex` cannot silently widen an allow.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleFile {
    effect: String,
    tool: String,
    #[serde(default)]
    arg_regex: Option<String>,
}

enum FileLoad {
    Missing,
    Ok(Vec<Rule>),
    /// File exists but is unusable; caller must fail closed.
    Invalid(String),
}

impl Permissions {
    /// Discover rules under project `.openmax/permissions.toml` then global
    /// `~/.openmax/permissions.toml`. Project rules are listed first so they win.
    pub fn discover(project_root: &Path) -> Self {
        Self::from_files(project_root, &permission_files(project_root))
    }

    fn from_files(project_root: &Path, paths: &[PathBuf]) -> Self {
        let mut rules = Vec::new();
        for path in paths {
            match load_file(path) {
                FileLoad::Missing => {}
                FileLoad::Ok(mut loaded) => rules.append(&mut loaded),
                FileLoad::Invalid(reason) => {
                    return Self {
                        rules: Vec::new(),
                        fail_closed: true,
                        fail_closed_reason: Some(reason),
                        invalid_path: Some(path.clone()),
                        project_root: project_root.to_path_buf(),
                    };
                }
            }
        }
        Self {
            rules,
            fail_closed: false,
            fail_closed_reason: None,
            invalid_path: None,
            project_root: project_root.to_path_buf(),
        }
    }

    /// True when this call is a rewrite of the very file that is failing
    /// closed. The broken file expresses no enforceable policy, and the agent
    /// is told to write this file, so leaving one path open keeps a typo from
    /// being unrecoverable from inside the session. It grants nothing on its
    /// own: the caller still applies `approval_mode`, so an interactive user
    /// approves the repair and `readonly` still refuses it.
    ///
    /// Only the project file is reachable this way, and deliberately so. The
    /// global file lives outside the project root, where the built-in file
    /// tools cannot write even under a healthy policy; reaching it needs
    /// `bash`, and exempting `bash` here would leave no gate at all. A broken
    /// global policy is a user-authored config error, so it is fixed the same
    /// way it was written: from the shell, guided by the path in the deny
    /// reason and by `openmax --check`.
    fn repairs_invalid_policy(&self, tool: &str, args: &Value) -> bool {
        if !matches!(tool, "write_file" | "edit_file") {
            return false;
        }
        let Some(invalid) = &self.invalid_path else {
            return false;
        };
        let Some(raw) = args["path"].as_str() else {
            return false;
        };
        // Compare resolved paths so `./x/../permissions.toml` and symlinked
        // roots can neither dodge nor spoof the match, and require the result
        // to sit inside the project, so `../` cannot walk out to the global
        // file that the file tools would refuse to write anyway.
        let (Ok(candidate), Ok(invalid), Ok(root)) = (
            self.project_root.join(raw).canonicalize(),
            invalid.canonicalize(),
            self.project_root.canonicalize(),
        ) else {
            return false;
        };
        candidate == invalid && candidate.starts_with(&root)
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty() && !self.fail_closed
    }

    /// First matching rule wins. Missing rules → [`PermissionDecision::Default`].
    pub fn evaluate(&self, tool: &str, args: &Value) -> PermissionDecision {
        if self.fail_closed {
            if self.repairs_invalid_policy(tool, args) {
                return PermissionDecision::Default;
            }
            return PermissionDecision::Deny {
                reason: self.fail_closed_reason.clone().unwrap_or_else(|| {
                    "permissions.toml is malformed; failing closed".into()
                }),
            };
        }
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

/// Diagnose one permissions file for `openmax --check`: None when the file
/// does not exist, Ok(the tool each rule names, in file order) when it loads,
/// Err(reason) when the agent loop would fail closed because of it. The names
/// come back rather than just a count because matching is exact, so a rule
/// naming a tool that does not exist is a rule that silently never fires.
pub(crate) fn check_file(path: &Path) -> Option<Result<Vec<String>, String>> {
    match load_file(path) {
        FileLoad::Missing => None,
        FileLoad::Ok(rules) => Some(Ok(rules.into_iter().map(|r| r.tool).collect())),
        FileLoad::Invalid(reason) => Some(Err(reason)),
    }
}

pub(crate) fn permission_files(project_root: &Path) -> Vec<PathBuf> {
    let mut files = vec![project_root.join(".openmax").join("permissions.toml")];
    if let Some(home) = std::env::var_os("HOME") {
        files.push(PathBuf::from(home).join(".openmax").join("permissions.toml"));
    }
    files
}

fn load_file(path: &Path) -> FileLoad {
    if !path.is_file() {
        return FileLoad::Missing;
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            return FileLoad::Invalid(format!(
                "permissions file {} unreadable ({e}); failing closed",
                path.display()
            ));
        }
    };
    // Empty file is an intentional no-op, not a parse failure.
    if text.trim().is_empty() {
        return FileLoad::Ok(Vec::new());
    }
    let file: PermissionsFile = match toml::from_str(&text) {
        Ok(f) => f,
        Err(e) => {
            return FileLoad::Invalid(format!(
                "permissions file {} is malformed ({e}); failing closed",
                path.display()
            ));
        }
    };
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
            other => {
                return FileLoad::Invalid(format!(
                    "permissions file {} has unknown effect {other:?}; failing closed",
                    path.display()
                ));
            }
        };
        let arg_regex = match raw.arg_regex.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            None => None,
            Some(pat) => match Regex::new(pat) {
                Ok(re) => Some(re),
                Err(e) => {
                    return FileLoad::Invalid(format!(
                        "permissions file {} has invalid arg_regex ({e}); failing closed",
                        path.display()
                    ));
                }
            },
        };
        rules.push(Rule {
            effect,
            tool,
            arg_regex,
        });
    }
    FileLoad::Ok(rules)
}

/// Primary argument string used for optional `arg_regex` matching.
fn arg_haystack(tool: &str, args: &Value) -> String {
    match tool {
        "bash" => args["command"].as_str().unwrap_or("").to_string(),
        "write_file" | "edit_file" | "read_file" | "list_dir" => {
            args["path"].as_str().unwrap_or("").to_string()
        }
        "glob" | "grep" => args["pattern"].as_str().unwrap_or("").to_string(),
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

    /// A broken policy file must stay repairable from inside the session.
    /// Everything else still fails closed, and the repair itself is only
    /// exempted from the rules, not from `approval_mode` (Default, not Allow).
    #[test]
    fn invalid_policy_file_stays_repairable() {
        let tmp = tempfile_dir();
        let perms_path = tmp.join(".openmax").join("permissions.toml");
        // The typo an agent actually writes: `[[rule]]` instead of `[[rules]]`.
        write_perms(&perms_path, "[[rule]]\neffect = \"deny\"\ntool = \"bash\"\n");
        let perms = Permissions::discover(&tmp);

        assert_eq!(
            perms.evaluate("write_file", &json!({"path": ".openmax/permissions.toml"})),
            PermissionDecision::Default,
            "rewriting the broken file must remain possible"
        );
        assert_eq!(
            perms.evaluate("edit_file", &json!({"path": "./.openmax/../.openmax/permissions.toml"})),
            PermissionDecision::Default,
            "path resolution must not depend on spelling"
        );

        // Nothing else is exempt.
        for (tool, args) in [
            ("bash", json!({"command": "cat .openmax/permissions.toml"})),
            ("read_file", json!({"path": ".openmax/permissions.toml"})),
            ("write_file", json!({"path": "src/main.rs"})),
        ] {
            assert!(
                matches!(perms.evaluate(tool, &args), PermissionDecision::Deny { .. }),
                "{tool} must still fail closed"
            );
        }
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// The repair exemption stops at the project boundary. A malformed global
    /// file is only reachable through `bash`, and exempting `bash` while
    /// failing closed would leave no gate, so it stays a shell-side fix.
    #[test]
    fn invalid_global_policy_is_not_repairable_from_the_project() {
        let tmp = tempfile_dir();
        let global = tmp.join("home").join(".openmax").join("permissions.toml");
        write_perms(&global, "[[rule]]\neffect = \"deny\"\ntool = \"bash\"\n");
        let project = tmp.join("project");
        std::fs::create_dir_all(&project).unwrap();
        let perms = Permissions::from_files(&project, std::slice::from_ref(&global));

        for args in [
            json!({ "path": ".openmax/permissions.toml" }),
            json!({ "path": global.to_str().unwrap() }),
            json!({ "path": "../home/.openmax/permissions.toml" }),
        ] {
            assert!(
                matches!(perms.evaluate("write_file", &args), PermissionDecision::Deny { .. }),
                "global policy must not be repairable from inside the project: {args}"
            );
        }
        let _ = std::fs::remove_dir_all(tmp);
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
        let perms = Permissions::from_files(&tmp, &[project, global]);
        match perms.evaluate("bash", &json!({"command": "cargo test"})) {
            PermissionDecision::Deny { .. } => {}
            other => panic!("project deny should win over global allow, got {other:?}"),
        }
    }

    #[test]
    fn invalid_regex_fails_closed() {
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
        // Broken policy must not drop remaining rules and fail open.
        match perms.evaluate("bash", &json!({"command": "ls -la"})) {
            PermissionDecision::Deny { reason } => {
                assert!(reason.contains("failing closed"), "{reason}");
            }
            other => panic!("expected fail-closed Deny, got {other:?}"),
        }
    }

    #[test]
    fn malformed_toml_fails_closed() {
        let tmp = tempfile_dir();
        write_perms(
            &tmp.join(".openmax").join("permissions.toml"),
            "this is not valid toml [[[",
        );
        let perms = Permissions::discover(&tmp);
        match perms.evaluate("bash", &json!({"command": "echo hi"})) {
            PermissionDecision::Deny { reason } => {
                assert!(reason.contains("malformed") || reason.contains("failing closed"), "{reason}");
            }
            other => panic!("expected fail-closed Deny, got {other:?}"),
        }
    }

    #[test]
    fn unknown_rule_field_fails_closed() {
        let tmp = tempfile_dir();
        // Misspelled filter key must not become an unconditional allow.
        write_perms(
            &tmp.join(".openmax").join("permissions.toml"),
            r#"
[[rules]]
effect = "allow"
tool = "bash"
args_regex = "^cargo test"
"#,
        );
        let perms = Permissions::discover(&tmp);
        match perms.evaluate("bash", &json!({"command": "rm -rf /"})) {
            PermissionDecision::Deny { reason } => {
                assert!(reason.contains("failing closed") || reason.contains("malformed"), "{reason}");
            }
            other => panic!("expected fail-closed Deny, got {other:?}"),
        }
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

    #[test]
    fn misspelled_top_level_rules_fails_closed() {
        let tmp = tempfile_dir();
        // `[[rule]]` instead of `[[rules]]` must not load as empty/default policy.
        write_perms(
            &tmp.join(".openmax").join("permissions.toml"),
            r#"
[[rule]]
effect = "deny"
tool = "bash"
arg_regex = "rm"
"#,
        );
        let perms = Permissions::discover(&tmp);
        match perms.evaluate("bash", &json!({"command": "rm -rf /"})) {
            PermissionDecision::Deny { reason } => {
                assert!(
                    reason.contains("failing closed") || reason.contains("malformed"),
                    "{reason}"
                );
            }
            other => panic!("expected fail-closed Deny, got {other:?}"),
        }
    }
}
