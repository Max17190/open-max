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
    /// Allow rules that were dropped because the file granting them sits
    /// inside the project and no human approved its content, as one line per
    /// file. `openmax --check` reports these.
    inert_allows: Vec<String>,
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
    pub fn discover(project_root: &Path, data_dir: &Path) -> Self {
        Self::from_files(project_root, &permission_files(project_root), data_dir)
    }

    fn from_files(project_root: &Path, paths: &[PathBuf], data_dir: &Path) -> Self {
        let mut rules = Vec::new();
        let mut inert_allows = Vec::new();
        for path in paths {
            match load_file(path) {
                FileLoad::Missing => {}
                FileLoad::Ok(mut loaded) => {
                    if let Some(dropped) = drop_unapproved_allows(
                        &mut loaded,
                        path,
                        project_root,
                        data_dir,
                    ) {
                        inert_allows.push(dropped);
                    }
                    rules.append(&mut loaded);
                }
                FileLoad::Invalid(reason) => {
                    return Self {
                        rules: Vec::new(),
                        fail_closed: true,
                        fail_closed_reason: Some(reason),
                        invalid_path: Some(path.clone()),
                        project_root: project_root.to_path_buf(),
                        inert_allows: Vec::new(),
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
            inert_allows,
        }
    }

    /// Allow rules that are not in effect because the file granting them is
    /// agent-writable and unapproved. Not an error: the call simply falls
    /// through to `approval_mode`, so the human is asked - which is how this
    /// state announces itself without a channel of its own.
    pub fn notices(&self) -> &[String] {
        &self.inert_allows
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

/// Strip `allow` rules a human never approved, returning what was dropped.
///
/// `allow` is the only effect that takes a gate away: it sets the force-allow
/// path, so the call skips the approval prompt outright - even a mutating tool
/// in ask mode, even an external tool whose content nobody blessed. `deny` and
/// `ask` only ever add friction, so they need no approval: an agent writing
/// them costs the session availability, which an agent holding `bash` has
/// anyway.
///
/// The project file is an ordinary project path the agent writes, and
/// `--spec permissions` tells it exactly how to write one, so an unapproved
/// `allow` there is the agent handing itself the gate the human was standing
/// at. Requiring the approval is what makes "rules the agent cannot change"
/// true of the direction that matters. The global file lives outside the
/// project root, where the confined file tools cannot write - the same
/// boundary `trust.json` and the ledger sit at - so it needs no approval;
/// containment is judged rather than assumed, because a `$HOME` inside the
/// project root really is agent-writable.
///
/// A dropped rule is inert, not fatal: evaluation continues to the next rule,
/// so what remains is the same policy with its relaxations removed. Losing an
/// `allow` can only cause a prompt, never an unguarded call, which is why this
/// needs none of the fail-closed machinery a revoked gate does.
fn drop_unapproved_allows(
    rules: &mut Vec<Rule>,
    path: &Path,
    project_root: &Path,
    data_dir: &Path,
) -> Option<String> {
    if !rules.iter().any(|r| r.effect == Effect::Allow) {
        return None;
    }
    if !agent_writable(path, project_root) {
        return None;
    }
    // Any failure to verify reads as unapproved, the same direction every
    // other approval check fails in.
    let approved = std::fs::read(path).is_ok_and(|bytes| {
        crate::ledger::is_approved(data_dir, project_root, &crate::ledger::sha256_hex(&bytes))
    });
    if approved {
        return None;
    }
    let before = rules.len();
    rules.retain(|r| r.effect != Effect::Allow);
    let dropped = before - rules.len();
    Some(format!(
        "{}: {dropped} allow rule(s) are inert because they skip the approval prompt and this file sits inside the project, where the agent writes; calls fall through to approval_mode until a human approves this exact content with `openmax --approve {}`",
        path.display(),
        path.display()
    ))
}

/// Whether the agent's own file tools can write this path: it resolves inside
/// the project root. Judged through symlinks as well as lexically, so neither
/// spelling decides the question differently.
fn agent_writable(path: &Path, project_root: &Path) -> bool {
    let resolved = path.canonicalize();
    let candidate = resolved.as_deref().unwrap_or(path);
    let root = project_root.canonicalize();
    let root = root.as_deref().unwrap_or(project_root);
    candidate.starts_with(root) || path.starts_with(project_root)
}

/// Why this file's `allow` rules are not authority, for `openmax --check`.
/// None when it has none, when it is out of the agent's reach, or when a human
/// approved it.
pub(crate) fn inert_allow_reason(
    path: &Path,
    project_root: &Path,
    data_dir: &Path,
) -> Option<String> {
    let FileLoad::Ok(mut rules) = load_file(path) else { return None };
    drop_unapproved_allows(&mut rules, path, project_root, data_dir)
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
        let perms = Permissions::discover(&tmp, &tmp.join("data"));

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
        let perms = Permissions::from_files(&project, std::slice::from_ref(&global), &tmp.join("data"));

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

    /// `allow` is the only effect that takes a gate away: it sets the
    /// force-allow path, so the call skips the approval prompt entirely, even
    /// for a mutating tool in ask mode. The project file that can grant it is
    /// an ordinary project path the agent writes, and `--spec permissions`
    /// tells the agent exactly how to write one - so without a human in the
    /// act, the agent can hand itself the gate the human was standing at, and
    /// can do it over the top of a global rule the human wrote.
    #[test]
    fn a_project_allow_rule_is_inert_until_a_human_approves_the_file() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        // The global file has to sit outside the project root, which is the
        // whole reason it is treated differently.
        let root = tmp.join("project");
        std::fs::create_dir_all(&root).unwrap();
        let project = root.join(".openmax").join("permissions.toml");
        let global = tmp.join("home").join(".openmax").join("permissions.toml");
        write_perms(
            &global,
            "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \"rm\"\n",
        );
        // Exactly what the spec's own example teaches the agent to write.
        let body = r#"
[[rules]]
effect = "allow"
tool = "bash"

[[rules]]
effect = "deny"
tool = "write_file"
arg_regex = "^src/"
"#;
        write_perms(&project, body);

        let files = [project.clone(), global.clone()];
        let perms = Permissions::from_files(&root, &files, &data);
        assert!(
            matches!(
                perms.evaluate("bash", &json!({"command": "rm -rf /"})),
                PermissionDecision::Deny { .. }
            ),
            "an unapproved allow must not override the human's global deny"
        );
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "cargo test"})),
            PermissionDecision::Default,
            "an unapproved allow falls through to approval_mode, granting nothing"
        );
        // Tightening needs no approval: a rule that only adds friction costs
        // availability, which an agent holding bash has anyway.
        assert!(matches!(
            perms.evaluate("write_file", &json!({"path": "src/main.rs"})),
            PermissionDecision::Deny { .. }
        ));
        // Inert is not silent.
        let notices = perms.notices();
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert!(notices[0].contains("openmax --approve"), "{}", notices[0]);

        // A human approves that exact content, and the allow is authority.
        let sha = crate::ledger::sha256_hex(&std::fs::read(&project).unwrap());
        crate::ledger::approve_capability(&data, &root, &project, &[sha]).unwrap();
        let perms = Permissions::from_files(&root, &files, &data);
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "cargo test"})),
            PermissionDecision::Allow
        );
        assert!(perms.notices().is_empty(), "{:?}", perms.notices());

        // Any edit revokes it: the new bytes are bytes nobody approved.
        write_perms(&project, &format!("{body}# one more comment\n"));
        let perms = Permissions::from_files(&root, &files, &data);
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "cargo test"})),
            PermissionDecision::Default
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    /// The global file lives outside the project root, where the confined file
    /// tools cannot write - the same boundary trust.json and the ledger sit
    /// at. Requiring an approval there would ask a human to bless their own
    /// hand-written config for no gain.
    #[test]
    fn a_global_allow_rule_needs_no_approval() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let root = tmp.join("project");
        std::fs::create_dir_all(&root).unwrap();
        let global = tmp.join("home").join(".openmax").join("permissions.toml");
        write_perms(
            &global,
            "[[rules]]\neffect = \"allow\"\ntool = \"bash\"\narg_regex = \"^cargo test\"\n",
        );
        let perms = Permissions::from_files(&root, std::slice::from_ref(&global), &data);
        assert_eq!(
            perms.evaluate("bash", &json!({"command": "cargo test -p core"})),
            PermissionDecision::Allow
        );
        assert!(perms.notices().is_empty());
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn missing_file_is_default() {
        let tmp = tempfile_dir();
        let perms = Permissions::discover(&tmp, &tmp.join("data"));
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
        let perms = Permissions::discover(&tmp, &tmp.join("data"));
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

    /// The feature's headline use: a human lets the agent run the test suite
    /// without a prompt each time. It takes one approval of the file, because
    /// the file sits where the agent writes.
    #[test]
    fn allow_cargo_test() {
        let tmp = tempfile_dir();
        let data = tmp.join("data");
        let path = tmp.join(".openmax").join("permissions.toml");
        write_perms(
            &path,
            r#"
[[rules]]
effect = "allow"
tool = "bash"
arg_regex = "^cargo (test|check|build)"
"#,
        );
        let sha = crate::ledger::sha256_hex(&std::fs::read(&path).unwrap());
        crate::ledger::approve_capability(&data, &tmp, &path, &[sha]).unwrap();
        let perms = Permissions::discover(&tmp, &data);
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
        let perms = Permissions::from_files(&tmp, &[project, global], &tmp.join("data"));
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
        let perms = Permissions::discover(&tmp, &tmp.join("data"));
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
        let perms = Permissions::discover(&tmp, &tmp.join("data"));
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
        let perms = Permissions::discover(&tmp, &tmp.join("data"));
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
        let perms = Permissions::discover(&tmp, &tmp.join("data"));
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
        let perms = Permissions::discover(&tmp, &tmp.join("data"));
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
