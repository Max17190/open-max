//! The session-frozen tool registry: the seven built-in tools plus any
//! external tools configured under `.openmax/tools/*.toml` (project) and
//! `~/.openmax/tools/*.toml` (global), plus discovered skills.
//!
//! Built exactly once per session and never rebuilt: the serialized tool
//! schema array is part of the prompt prefix the server's KV cache keys on,
//! so it must stay byte-stable for the session's whole lifetime. Config
//! changes apply to new sessions only.
//!
//! With no external tools or skills installed, the schema JSON is
//! byte-identical to the built-in `tools::tool_schemas()` array and the
//! prompt gains nothing: extensibility costs zero tokens by default.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::skills::{self, SkillSpec};
use crate::state::CancelToken;
use crate::tools::{self, ToolOutcome};

/// External tool descriptions ride in the prompt prefix of every request, so
/// they are capped hard; authors link a README for anything longer.
pub const MAX_EXTERNAL_DESC_CHARS: usize = 200;

#[derive(Clone, Debug)]
pub enum ToolKind {
    Builtin,
    External(ExternalTool),
}

/// How to run one external tool: spawn `command args...`, write the call's
/// JSON arguments to stdin, read the result from stdout.
#[derive(Clone, Debug)]
pub struct ExternalTool {
    pub command: String,
    pub args: Vec<String>,
    pub timeout_secs: u64,
    /// Where the definition came from, for actionable error messages.
    pub source_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON-schema object for the tool's parameters, as sent to the model.
    pub parameters: Value,
    /// Mutating tools go through approval gating.
    pub mutating: bool,
    pub kind: ToolKind,
}

/// Frozen at session creation; immutable afterwards.
pub struct Registry {
    /// Built-ins first in their fixed order, then external tools sorted by
    /// name — deterministic so two builds serialize identically.
    pub tools: Vec<ToolSpec>,
    pub skills: Vec<SkillSpec>,
    /// Schema array serialized once at freeze time.
    schemas: Value,
    by_name: HashMap<String, usize>,
}

impl Registry {
    /// Discover external tools and skills for a project and freeze the
    /// registry. Called once, at session creation.
    pub fn build(project_root: &Path) -> Self {
        Self::assemble(discover_external(project_root), skills::discover(project_root))
    }

    /// A registry with built-ins only: used for sessions that predate the
    /// extensibility layer, so their behavior never changes retroactively.
    pub fn builtin_only() -> Self {
        Self::assemble(Vec::new(), Vec::new())
    }

    pub(crate) fn assemble(mut external: Vec<ToolSpec>, skills: Vec<SkillSpec>) -> Self {
        // Built-ins come straight from the canonical schema literals so the
        // registry can never drift from what tools.rs implements.
        let mut tools_list = builtin_specs();
        // Built-in names win over external ones: shadowing a built-in would
        // silently change core behavior mid-workflow.
        external.retain(|t| !tools::TOOL_NAMES.contains(&t.name.as_str()));
        external.sort_by(|a, b| a.name.cmp(&b.name));
        tools_list.extend(external);

        let mut schemas = tools::tool_schemas().clone();
        if let Some(arr) = schemas.as_array_mut() {
            for spec in tools_list.iter().filter(|s| !matches!(s.kind, ToolKind::Builtin)) {
                arr.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    }
                }));
            }
        }

        let by_name = tools_list
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.clone(), i))
            .collect();
        Self { tools: tools_list, skills, schemas, by_name }
    }

    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.by_name.get(name).map(|&i| &self.tools[i])
    }

    pub fn is_mutating(&self, name: &str) -> bool {
        self.get(name).map(|s| s.mutating).unwrap_or(false)
    }

    /// Every tool name, in frozen order. Feeds the fallback call parser.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|s| s.name.clone()).collect()
    }

    /// The OpenAI-format tool schema array, serialized once at freeze time.
    pub fn tool_schemas_json(&self) -> &Value {
        &self.schemas
    }

    /// Ephemeral read-only registry for a `task` subagent. Mutating tools and
    /// the `task` meta-tool are always excluded so a child cannot change the
    /// workspace or recurse.
    pub fn scoped(&self, allowed: &[&str]) -> Registry {
        let tools: Vec<ToolSpec> = self
            .tools
            .iter()
            .filter(|s| allowed.contains(&s.name.as_str()))
            .filter(|s| !s.mutating && s.name != tools::TASK_TOOL)
            .cloned()
            .collect();
        let schemas = serialize_specs(&tools);
        let by_name = tools.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
        Registry { tools, skills: Vec::new(), schemas, by_name }
    }

    pub async fn execute(
        &self,
        name: &str,
        args: &Value,
        root: &Path,
        caps: tools::OutputCaps,
        cancel: Arc<CancelToken>,
    ) -> ToolOutcome {
        match self.get(name).map(|s| s.kind.clone()) {
            Some(ToolKind::Builtin) => tools::execute(name, args, root, caps, cancel).await,
            Some(ToolKind::External(tool)) => spawn_external(name, &tool, args, root, caps, cancel).await,
            None => ToolOutcome::err(format!(
                "unknown tool: {name}; the available tools are {}",
                self.tool_names().join(", ")
            )),
        }
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::builtin_only()
    }
}

/// The persisted record of what a session's registry froze at creation:
/// enough to rebuild the exact same schemas on resume without re-reading
/// any config from disk, so a session never changes shape retroactively.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RegistryManifest {
    pub version: u32,
    pub external_tools: Vec<ExternalToolManifest>,
    pub skills: Vec<SkillSpec>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ExternalToolManifest {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub mutating: bool,
    pub command: String,
    pub args: Vec<String>,
    pub timeout_secs: u64,
    pub source_path: PathBuf,
}

impl Registry {
    pub fn to_manifest(&self) -> RegistryManifest {
        let external_tools = self
            .tools
            .iter()
            .filter_map(|spec| match &spec.kind {
                ToolKind::Builtin => None,
                ToolKind::External(t) => Some(ExternalToolManifest {
                    name: spec.name.clone(),
                    description: spec.description.clone(),
                    parameters: spec.parameters.clone(),
                    mutating: spec.mutating,
                    command: t.command.clone(),
                    args: t.args.clone(),
                    timeout_secs: t.timeout_secs,
                    source_path: t.source_path.clone(),
                }),
            })
            .collect();
        RegistryManifest { version: 1, external_tools, skills: self.skills.clone() }
    }

    pub fn from_manifest(manifest: RegistryManifest) -> Self {
        let external = manifest
            .external_tools
            .into_iter()
            .map(|t| ToolSpec {
                name: t.name,
                description: t.description,
                parameters: t.parameters,
                mutating: t.mutating,
                kind: ToolKind::External(ExternalTool {
                    command: t.command,
                    args: t.args,
                    timeout_secs: t.timeout_secs,
                    source_path: t.source_path,
                }),
            })
            .collect();
        Self::assemble(external, manifest.skills)
    }

    /// True when the registry carries anything beyond the built-ins; an
    /// all-builtin session needs no manifest file at all.
    pub fn has_extensions(&self) -> bool {
        !self.skills.is_empty()
            || self.tools.iter().any(|s| !matches!(s.kind, ToolKind::Builtin))
    }
}

/// One-line human summary of a call, for approval prompts and tool cards.
/// Registry-free on purpose: built-in names summarize by their known argument
/// shapes, every other name by the external heuristic — exactly what a
/// registry lookup would produce, without threading session state into the UI.
pub fn summarize_call(name: &str, args: &Value) -> String {
    if tools::TOOL_NAMES.contains(&name) {
        tools::summarize_call(name, args)
    } else {
        summarize_external(args)
    }
}

/// Serialize tool specs into the OpenAI schema array. Used for ephemeral
/// (subagent) registries via `scoped`, which do not need byte-identity with
/// the built-in schema literals.
fn serialize_specs(tools: &[ToolSpec]) -> Value {
    Value::Array(
        tools
            .iter()
            .map(|spec| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    }
                })
            })
            .collect(),
    )
}

/// Built-in tool specs derived from the canonical `tools::tool_schemas()`
/// literals, so name/description/parameters have a single source of truth.

fn builtin_specs() -> Vec<ToolSpec> {
    let schemas = tools::tool_schemas();
    schemas
        .as_array()
        .expect("builtin schemas are an array")
        .iter()
        .map(|entry| {
            let f = &entry["function"];
            let name = f["name"].as_str().expect("builtin schema has a name").to_string();
            ToolSpec {
                mutating: tools::is_mutating(&name),
                description: f["description"].as_str().unwrap_or("").to_string(),
                parameters: f["parameters"].clone(),
                name,
                kind: ToolKind::Builtin,
            }
        })
        .collect()
}

/// External tools have arbitrary parameter names, so summaries fall back to
/// the most path/command-looking argument available.
fn summarize_external(args: &Value) -> String {
    for key in ["command", "path", "pattern"] {
        if let Some(v) = args[key].as_str() {
            return v.to_string();
        }
    }
    args.as_object()
        .and_then(|o| o.values().find_map(|v| v.as_str()))
        .unwrap_or("")
        .to_string()
}

/// The TOML shape of one tool definition file.
#[derive(serde::Deserialize)]
struct ExternalToolFile {
    name: String,
    description: String,
    /// JSON-schema object for the parameters; defaults to "no parameters".
    #[serde(default)]
    params: Option<Value>,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
    #[serde(default)]
    mutating: bool,
}

fn default_timeout() -> u64 {
    60
}

/// Discover external tool definitions: global `~/.openmax/tools/*.toml`
/// first, then the project's `.openmax/tools/*.toml`, which wins on name
/// collision. Malformed files are skipped, never fatal.
fn discover_external(project_root: &Path) -> Vec<ToolSpec> {
    discover_external_in(&[
        crate::state::default_data_dir().join("tools"),
        project_root.join(".openmax").join("tools"),
    ])
}

fn discover_external_in(dirs: &[PathBuf]) -> Vec<ToolSpec> {
    let mut by_name: HashMap<String, ToolSpec> = HashMap::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .collect();
        // Deterministic within a dir; later dirs (the project) win overall.
        paths.sort();
        for path in paths {
            if let Some(spec) = parse_tool_file(&path) {
                by_name.insert(spec.name.clone(), spec);
            }
        }
    }
    by_name.into_values().collect()
}

fn parse_tool_file(path: &Path) -> Option<ToolSpec> {
    let text = std::fs::read_to_string(path).ok()?;
    let file: ExternalToolFile = toml::from_str(&text).ok()?;
    let name = file.name.trim().to_string();
    // Boring, model-friendly names only; anything else is a config mistake.
    let name_ok = !name.is_empty()
        && name.len() <= 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !name_ok || file.command.trim().is_empty() {
        return None;
    }
    let mut description = file.description.trim().replace(['\n', '\r'], " ");
    if description.chars().count() > MAX_EXTERNAL_DESC_CHARS {
        description = description.chars().take(MAX_EXTERNAL_DESC_CHARS).collect::<String>() + "…";
    }
    let parameters = match file.params {
        Some(p) if p.is_object() => p,
        Some(_) => return None,
        None => serde_json::json!({ "type": "object", "properties": {} }),
    };
    Some(ToolSpec {
        name,
        description,
        parameters,
        mutating: file.mutating,
        kind: ToolKind::External(ExternalTool {
            command: file.command.trim().to_string(),
            args: file.args,
            timeout_secs: file.timeout_secs.clamp(1, 300),
            source_path: path.to_path_buf(),
        }),
    })
}

/// Run one external tool: spawn the command in the project root, hand it the
/// call's JSON arguments on stdin, and treat stdout as the result. Same
/// output caps and spill-to-file behavior as bash. One process per call,
/// nothing stays resident.
async fn spawn_external(
    name: &str,
    tool: &ExternalTool,
    args: &Value,
    root: &Path,
    caps: tools::OutputCaps,
    cancel: Arc<CancelToken>,
) -> ToolOutcome {
    let stdin_json = args.to_string();
    let mut cmd = tokio::process::Command::new(&tool.command);
    cmd.args(&tool.args)
        .current_dir(root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return ToolOutcome::err(format!(
                "failed to start external tool '{name}' (command '{}', defined in {}): {e}",
                tool.command,
                tool.source_path.display()
            ));
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        // A tool that exits without reading stdin closes the pipe; that is fine, not an error.
        let _ = stdin.write_all(stdin_json.as_bytes()).await;
    }

    let mut child_slot = Some(child);
    tokio::select! {
        _ = cancel.cancelled() => {
            if let Some(mut c) = child_slot.take() {
                let _ = c.kill().await;
            }
            ToolOutcome::err(format!("external tool '{name}' cancelled by user"))
        }
        _ = tokio::time::sleep(Duration::from_secs(tool.timeout_secs)) => {
            if let Some(mut c) = child_slot.take() {
                let _ = c.kill().await;
            }
            ToolOutcome::err(format!(
                "external tool '{name}' timed out after {}s",
                tool.timeout_secs
            ))
        }
        output = async {
            child_slot.take().expect("child taken twice").wait_with_output().await
        } => {
            match output {
                Err(e) => ToolOutcome::err(format!("external tool '{name}' failed: {e}")),
                Ok(output) => {
                    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if !stderr.trim().is_empty() {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str("[stderr]\n");
                        text.push_str(&stderr);
                    }
                    if text.len() > caps.command_bytes {
                        text = tools::truncate_command_output(&text, caps.command_bytes);
                    }
                    if text.trim().is_empty() {
                        text = "(no output)".into();
                    }
                    let code = output.status.code().unwrap_or(-1);
                    if output.status.success() {
                        ToolOutcome::ok(text)
                    } else {
                        ToolOutcome { ok: false, output: format!("exit code {code}\n{text}"), diff: None }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::CancelToken;

    fn no_cancel() -> Arc<CancelToken> {
        Arc::new(CancelToken::default())
    }

    #[test]
    fn builtin_only_schemas_are_byte_identical_to_static() {
        let registry = Registry::builtin_only();
        // Byte identity, not just structural: this array is part of the
        // prompt-cache prefix.
        assert_eq!(
            registry.tool_schemas_json().to_string(),
            tools::tool_schemas().to_string()
        );
    }

    #[test]
    fn build_with_no_config_matches_builtin_only() {
        let dir = std::env::temp_dir().join(format!("omx-reg-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = Registry::build(&dir);
        assert_eq!(
            registry.tool_schemas_json().to_string(),
            Registry::builtin_only().tool_schemas_json().to_string()
        );
        assert!(registry.skills.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn schemas_are_deterministic_across_builds() {
        let a = Registry::builtin_only();
        let b = Registry::builtin_only();
        assert_eq!(a.tool_schemas_json().to_string(), b.tool_schemas_json().to_string());
        assert_eq!(a.tool_names(), b.tool_names());
    }

    #[test]
    fn scoped_excludes_mutating_and_task() {
        let reg = Registry::builtin_only();
        let scoped = reg.scoped(&["list_dir", "read_file", "glob", "grep", "bash", "task"]);
        let names: Vec<_> = scoped.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["list_dir", "read_file", "glob", "grep"]);
        assert!(scoped.get("bash").is_none());
        assert!(scoped.get(tools::TASK_TOOL).is_none());
        assert!(scoped.get("write_file").is_none());
    }

    #[test]
    fn builtin_lookups_match_tools_module() {
        let registry = Registry::builtin_only();
        assert_eq!(
            registry.tool_names(),
            tools::TOOL_NAMES.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        for name in tools::TOOL_NAMES {
            assert_eq!(registry.is_mutating(name), tools::is_mutating(name), "{name}");
        }
        assert!(!registry.is_mutating("nope"));
    }

    #[tokio::test]
    async fn unknown_tool_error_lists_names() {
        let registry = Registry::builtin_only();
        let out = registry
            .execute("nope", &serde_json::json!({}), Path::new("."), tools::OutputCaps::default(), no_cancel())
            .await;
        assert!(!out.ok);
        assert!(out.output.contains("bash"), "should list valid tools: {}", out.output);
    }

    // ---------- external tools ----------

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tool(dir: &Path, file: &str, body: &str) {
        std::fs::write(dir.join(file), body).unwrap();
    }

    fn write_script(dir: &Path, file: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(file);
        std::fs::write(&path, body).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn registry_from(global: &Path, project: &Path) -> Registry {
        Registry::assemble(
            discover_external_in(&[global.to_path_buf(), project.to_path_buf()]),
            Vec::new(),
        )
    }

    #[test]
    fn project_tool_wins_over_global_on_collision() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        write_tool(&global, "hello.toml", "name = \"hello\"\ndescription = \"global\"\ncommand = \"/bin/false\"\n");
        write_tool(&project, "hello.toml", "name = \"hello\"\ndescription = \"project\"\ncommand = \"/bin/true\"\n");
        let registry = registry_from(&global, &project);
        let spec = registry.get("hello").expect("hello discovered");
        assert_eq!(spec.description, "project");
        match &spec.kind {
            ToolKind::External(t) => assert_eq!(t.command, "/bin/true"),
            _ => panic!("expected external"),
        }
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn externals_sort_after_builtins_and_schemas_stay_deterministic() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        write_tool(&project, "zz.toml", "name = \"zz_tool\"\ndescription = \"z\"\ncommand = \"/bin/true\"\n");
        write_tool(&project, "aa.toml", "name = \"aa_tool\"\ndescription = \"a\"\ncommand = \"/bin/true\"\n");
        let a = registry_from(&global, &project);
        let b = registry_from(&global, &project);
        let names = a.tool_names();
        let builtin_count = tools::TOOL_NAMES.len();
        assert_eq!(&names[..builtin_count], tools::TOOL_NAMES);
        assert_eq!(&names[builtin_count..], &["aa_tool", "zz_tool"]);
        assert_eq!(
            a.tool_schemas_json().to_string(),
            b.tool_schemas_json().to_string(),
            "schema serialization must be deterministic"
        );
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[test]
    fn external_description_capped_and_builtin_shadowing_rejected() {
        let global = temp_dir("glob");
        let project = temp_dir("proj");
        let long_desc = "d".repeat(500);
        write_tool(&project, "long.toml", &format!("name = \"long_tool\"\ndescription = \"{long_desc}\"\ncommand = \"/bin/true\"\n"));
        // Shadowing a built-in would silently change core behavior: dropped.
        write_tool(&project, "bash.toml", "name = \"bash\"\ndescription = \"evil\"\ncommand = \"/bin/true\"\n");
        // Malformed files are skipped, never fatal.
        write_tool(&project, "broken.toml", "name = \"broken\ncommand=");
        let registry = registry_from(&global, &project);
        let spec = registry.get("long_tool").unwrap();
        assert!(spec.description.chars().count() <= MAX_EXTERNAL_DESC_CHARS + 1);
        match &registry.get("bash").unwrap().kind {
            ToolKind::Builtin => {}
            _ => panic!("built-in bash must not be shadowed"),
        }
        assert!(registry.get("broken").is_none());
        let _ = std::fs::remove_dir_all(global);
        let _ = std::fs::remove_dir_all(project);
    }

    #[tokio::test]
    async fn external_tool_round_trips_args_over_stdin() {
        let project = temp_dir("proj");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        let script = write_script(&project, "echo_args.sh", "#!/bin/sh\nread -r line\necho \"got: $line\"\n");
        write_tool(
            &tools_dir,
            "echo.toml",
            &format!(
                "name = \"echo_args\"\ndescription = \"echoes\"\ncommand = \"{}\"\n\n[params]\ntype = \"object\"\n[params.properties.message]\ntype = \"string\"\n",
                script.display()
            ),
        );
        let registry = Registry::assemble(discover_external_in(&[tools_dir]), Vec::new());
        let out = registry
            .execute("echo_args", &serde_json::json!({"message": "hi"}), &project, tools::OutputCaps::default(), no_cancel())
            .await;
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("got: {\"message\":\"hi\"}"), "{}", out.output);
        let _ = std::fs::remove_dir_all(project);
    }

    #[tokio::test]
    async fn external_tool_timeout_and_failure_shape() {
        let project = temp_dir("proj");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        let slow = write_script(&project, "slow.sh", "#!/bin/sh\nsleep 5\n");
        let fail = write_script(&project, "fail.sh", "#!/bin/sh\necho oops >&2\nexit 3\n");
        write_tool(&tools_dir, "slow.toml", &format!("name = \"slow\"\ndescription = \"s\"\ncommand = \"{}\"\ntimeout_secs = 1\n", slow.display()));
        write_tool(&tools_dir, "fail.toml", &format!("name = \"fail\"\ndescription = \"f\"\ncommand = \"{}\"\n", fail.display()));
        let registry = Registry::assemble(discover_external_in(&[tools_dir]), Vec::new());

        let out = registry.execute("slow", &serde_json::json!({}), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok);
        assert!(out.output.contains("timed out after 1s"), "{}", out.output);

        let out = registry.execute("fail", &serde_json::json!({}), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok);
        assert!(out.output.starts_with("exit code 3"), "{}", out.output);
        assert!(out.output.contains("[stderr]") && out.output.contains("oops"), "{}", out.output);

        let out = registry.execute("missing_binary", &serde_json::json!({}), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok && out.output.contains("unknown tool"));
        let _ = std::fs::remove_dir_all(project);
    }

    #[tokio::test]
    async fn external_tool_spawn_failure_is_actionable() {
        let project = temp_dir("proj");
        let tools_dir = project.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        write_tool(&tools_dir, "ghost.toml", "name = \"ghost\"\ndescription = \"g\"\ncommand = \"/nonexistent/binary\"\n");
        let registry = Registry::assemble(discover_external_in(&[tools_dir.clone()]), Vec::new());
        let out = registry.execute("ghost", &serde_json::json!({}), &project, tools::OutputCaps::default(), no_cancel()).await;
        assert!(!out.ok);
        assert!(out.output.contains("ghost") && out.output.contains("/nonexistent/binary"), "{}", out.output);
        assert!(out.output.contains("ghost.toml"), "must point at the defining file: {}", out.output);
        let _ = std::fs::remove_dir_all(project);
    }
}
