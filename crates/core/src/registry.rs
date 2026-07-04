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

use serde_json::Value;

use crate::skills::{self, SkillSpec};
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

    fn assemble(mut external: Vec<ToolSpec>, skills: Vec<SkillSpec>) -> Self {
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

    pub async fn execute(&self, name: &str, args: &Value, root: &Path) -> ToolOutcome {
        match self.get(name).map(|s| s.kind.clone()) {
            Some(ToolKind::Builtin) => tools::execute(name, args, root).await,
            Some(ToolKind::External(tool)) => spawn_external(name, &tool, args, root).await,
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

/// Discover `.openmax/tools/*.toml` definitions. Placeholder until the
/// external-tools step lands: no discovery, zero external tools.
fn discover_external(_project_root: &Path) -> Vec<ToolSpec> {
    Vec::new()
}

async fn spawn_external(name: &str, tool: &ExternalTool, _args: &Value, _root: &Path) -> ToolOutcome {
    // Filled in by the external-tools step; unreachable until discovery
    // can produce ToolKind::External entries.
    ToolOutcome::err(format!(
        "external tool '{name}' ({}) is not runnable yet",
        tool.command
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            .execute("nope", &serde_json::json!({}), Path::new("."))
            .await;
        assert!(!out.ok);
        assert!(out.output.contains("bash"), "should list valid tools: {}", out.output);
    }
}
