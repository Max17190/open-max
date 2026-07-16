use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::state::CancelToken;

use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};

use crate::client::truncate;

const MAX_RESULTS: usize = 200;
/// Grep lines run up to ~300 chars: 200 results could inject ~60KB (≈15k
/// tokens) into a 16k window in one call, and every one of those tokens is
/// re-prefilled on every subsequent turn. 50 is plenty to act on.
const MAX_GREP_RESULTS: usize = 50;
const MAX_OUTPUT_BYTES: usize = 30_000;
const MAX_READ_LINES: usize = 1500;
const MAX_READ_BYTES: usize = 24_000;
const MAX_DIR_ENTRIES: usize = 200;

/// Output limits threaded into command-running tools (bash and external
/// tools). Settings can widen or tighten the command cap; everything else
/// keeps the tuned constants above.
#[derive(Clone, Copy)]
pub struct OutputCaps {
    pub command_bytes: usize,
}

impl Default for OutputCaps {
    fn default() -> Self {
        Self { command_bytes: MAX_OUTPUT_BYTES }
    }
}

impl OutputCaps {
    pub fn from_settings(settings: &crate::config::Settings) -> Self {
        Self { command_bytes: settings.max_output_bytes.unwrap_or(MAX_OUTPUT_BYTES).max(1_000) }
    }
}
const MAX_LINE_CHARS: usize = 500;
const MAX_FILE_BYTES: u64 = 1_500_000;

#[derive(Clone, serde::Serialize)]
pub struct DiffInfo {
    pub path: String,
    pub diff: String,
    pub added: usize,
    pub removed: usize,
}

#[derive(Clone)]
pub struct ToolOutcome {
    pub ok: bool,
    pub output: String,
    pub diff: Option<DiffInfo>,
}

impl ToolOutcome {
    pub(crate) fn ok(output: String) -> Self {
        Self { ok: true, output, diff: None }
    }
    pub(crate) fn err(output: impl Into<String>) -> Self {
        Self { ok: false, output: output.into(), diff: None }
    }
}

/// True for tools that can change state and therefore go through approval.
/// `task` is read-only (it spawns a read-only subagent), so it never gates.
pub fn is_mutating(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file" | "bash")
}

/// The meta-tool that spawns a read-only subagent. It is a built-in for schema
/// purposes but is dispatched by the agent loop, never by `execute` below.
pub const TASK_TOOL: &str = "task";

/// Every tool name exposed by the harness. `task` comes last so the frozen
/// schema array and `TOOL_NAMES` stay in the same order the registry builds.
pub const TOOL_NAMES: &[&str] =
    &["list_dir", "read_file", "write_file", "edit_file", "glob", "grep", "bash", TASK_TOOL];

pub fn tool_names() -> Vec<String> {
    TOOL_NAMES.iter().map(|s| s.to_string()).collect()
}

/// One-line human summary of a call, shown in approval prompts and tool cards.
pub fn summarize_call(name: &str, args: &Value) -> String {
    match name {
        "bash" => args["command"].as_str().unwrap_or("?").to_string(),
        "write_file" | "edit_file" | "read_file" | "list_dir" => {
            args["path"].as_str().unwrap_or("?").to_string()
        }
        "glob" | "grep" => args["pattern"].as_str().unwrap_or("?").to_string(),
        "task" => {
            let kind = args["subagent"].as_str().unwrap_or("explore");
            let prompt = args["prompt"].as_str().unwrap_or("?");
            format!("{kind}: {prompt}")
        }
        _ => String::new(),
    }
}

/// Tool schemas in the OpenAI `tools` wire format. Kept deliberately small and
/// strict — small local models do much better with fewer, simpler parameters.
pub fn tool_schemas() -> &'static Value {
    static SCHEMAS: OnceLock<Value> = OnceLock::new();
    SCHEMAS.get_or_init(|| {
        json!([
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List a directory. Path \".\" is the project root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file as numbered lines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "offset": { "type": "integer", "description": "1-based start line" },
                        "limit": { "type": "integer", "description": "Max lines" }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file; parent dirs are created.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string", "description": "Full file content" }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace old_string with new_string in a file. Read it first; old_string must match exactly and be unique unless replace_all.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "old_string": { "type": "string" },
                        "new_string": { "type": "string" },
                        "replace_all": { "type": "boolean", "description": "Replace every occurrence (default false)" }
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Find files by glob pattern, e.g. \"**/*.rs\"; newest first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Regex-search file contents; returns path:line: text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Rust regex; no lookahead/backrefs" },
                        "path": { "type": "string", "description": "Directory to search (default \".\")" },
                        "glob": { "type": "string", "description": "Only files matching, e.g. \"*.rs\"" }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the project root (builds, tests, git).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "timeout_secs": { "type": "integer", "description": "Default 60, max 300" }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "task",
                "description": "Delegate a read-only investigation to a subagent with its own context; it returns only a summary. Cannot edit or run commands. Use for broad questions like \"where is X handled\".",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent": {
                            "type": "string",
                            "enum": ["explore", "search", "plan"],
                            "description": "explore=investigate; search=locate code; plan=step-by-step plan"
                        },
                        "prompt": { "type": "string", "description": "What to find and what to report back" }
                    },
                    "required": ["subagent", "prompt"]
                }
            }
        }
    ])
    })
}

/// Resolve a model-supplied relative path, refusing escapes from the project root.
fn resolve(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim().trim_start_matches('/');
    let joined = if rel.is_empty() || rel == "." { root.to_path_buf() } else { root.join(rel) };
    // Canonicalize the deepest existing ancestor so traversal via `..` is caught
    // even for paths that don't exist yet (e.g. write_file targets).
    let mut probe = joined.clone();
    let mut tail = Vec::new();
    while !probe.exists() {
        match (probe.file_name(), probe.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name.to_os_string());
                probe = parent.to_path_buf();
            }
            _ => return Err("invalid path".into()),
        }
    }
    let mut canon = probe.canonicalize().map_err(|e| format!("cannot resolve path: {e}"))?;
    for part in tail.iter().rev() {
        canon.push(part);
    }
    let root_canon = root.canonicalize().map_err(|e| format!("cannot resolve project root: {e}"))?;
    if !canon.starts_with(&root_canon) {
        return Err(format!("path escapes the project root: {rel}"));
    }
    Ok(canon)
}

fn rel_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

pub async fn execute(
    name: &str,
    args: &Value,
    root: &Path,
    caps: OutputCaps,
    cancel: Arc<CancelToken>,
) -> ToolOutcome {
    if name == TASK_TOOL {
        // The subagent meta-tool is intercepted in the agent loop; it must never
        // reach plain tool execution (and never appears in a subagent registry).
        return ToolOutcome::err("task is dispatched by the agent loop, not executable here");
    }
    if name == "bash" {
        return bash_tool(root, args, caps, cancel).await;
    }
    if cancel.is_cancelled() {
        return ToolOutcome::err("tool cancelled by user");
    }
    // The file tools are synchronous fs/walk work; run them off the async
    // workers so a big grep or read never stalls streaming and the UI.
    // Esc stops waiting immediately; the blocking task may finish in the pool.
    let name = name.to_string();
    let args = args.clone();
    let root = root.to_path_buf();
    tokio::select! {
        _ = cancel.cancelled() => ToolOutcome::err("tool cancelled by user"),
        result = tokio::task::spawn_blocking(move || match name.as_str() {
            "list_dir" => list_dir(&root, &args),
            "read_file" => read_file(&root, &args),
            "write_file" => write_file(&root, &args),
            "edit_file" => edit_file(&root, &args),
            "glob" => glob_tool(&root, &args),
            "grep" => grep_tool(&root, &args),
            other => ToolOutcome::err(format!(
                "unknown tool: {other}; the available tools are {}",
                TOOL_NAMES.join(", ")
            )),
        }) => result.unwrap_or_else(|e| ToolOutcome::err(format!("tool execution failed: {e}"))),
    }
}

fn list_dir(root: &Path, args: &Value) -> ToolOutcome {
    let rel = args["path"].as_str().unwrap_or(".");
    let dir = match resolve(root, rel) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) => return ToolOutcome::err(format!("cannot list {rel}: {e}")),
    };
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == ".git" || name == "node_modules" || name == ".DS_Store" {
            continue;
        }
        if entry.path().is_dir() {
            dirs.push(format!("{name}/"));
        } else {
            files.push(name);
        }
    }
    dirs.sort();
    files.sort();
    dirs.extend(files);
    if dirs.is_empty() {
        return ToolOutcome::ok("(empty directory)".into());
    }
    let total = dirs.len();
    let shown: Vec<String> = dirs.into_iter().take(MAX_DIR_ENTRIES).collect();
    let mut output = shown.join("\n");
    if total > MAX_DIR_ENTRIES {
        output.push_str(&format!(
            "\n… {} more entries not shown (use glob to find specific files)",
            total - MAX_DIR_ENTRIES
        ));
    }
    ToolOutcome::ok(output)
}

fn read_file(root: &Path, args: &Value) -> ToolOutcome {
    let rel = args["path"].as_str().unwrap_or_default();
    let path = match resolve(root, rel) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    match std::fs::metadata(&path) {
        Ok(m) if m.len() > MAX_FILE_BYTES => {
            return ToolOutcome::err(format!("file too large ({} bytes); use grep or read with offset/limit", m.len()))
        }
        Err(e) => return ToolOutcome::err(format!("cannot read {rel}: {e}")),
        _ => {}
    }
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return ToolOutcome::err(format!("{rel} is not a UTF-8 text file")),
    };
    let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(MAX_READ_LINES as u64) as usize;
    let limit = limit.min(MAX_READ_LINES);
    let total = text.lines().count();
    let mut out = String::new();
    let mut stopped_by_bytes = false;
    let mut byte_cap_line = 0usize;
    for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
        let line = if line.len() > MAX_LINE_CHARS { &line[..floor_char(line, MAX_LINE_CHARS)] } else { line };
        let formatted = format!("{:>5} {}\n", i + 1, line);
        if out.len() + formatted.len() > MAX_READ_BYTES {
            stopped_by_bytes = true;
            byte_cap_line = i + 1;
            break;
        }
        out.push_str(&formatted);
    }
    if stopped_by_bytes {
        out.push_str(&format!(
            "… output limit reached at line {byte_cap_line} (file has {total} lines; continue with offset={})\n",
            byte_cap_line + 1
        ));
    } else if total > offset - 1 + limit {
        out.push_str(&format!("… {} more lines (file has {total} lines; continue with offset={})\n", total - (offset - 1 + limit), offset + limit));
    }
    if out.is_empty() {
        out = "(empty file)".into();
    }
    ToolOutcome::ok(out)
}

fn floor_char(s: &str, mut idx: usize) -> usize {
    while !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn make_diff(root: &Path, path: &Path, old: &str, new: &str) -> DiffInfo {
    diff_strings(&rel_display(root, path), old, new)
}

/// Unified diff between two versions of a file. Shared with the cumulative
/// thread diff command in lib.rs.
pub fn diff_strings(rel: &str, old: &str, new: &str) -> DiffInfo {
    let text_diff = TextDiff::from_lines(old, new);
    let mut added = 0;
    let mut removed = 0;
    for change in text_diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => added += 1,
            ChangeTag::Delete => removed += 1,
            ChangeTag::Equal => {}
        }
    }
    let diff = text_diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{rel}"), &format!("b/{rel}"))
        .to_string();
    DiffInfo { path: rel.to_string(), diff: truncate(&diff, 40_000), added, removed }
}

fn write_file(root: &Path, args: &Value) -> ToolOutcome {
    let rel = args["path"].as_str().unwrap_or_default();
    let Some(content) = args["content"].as_str() else {
        return ToolOutcome::err("missing required argument: content");
    };
    let path = match resolve(root, rel) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    let old = std::fs::read_to_string(&path).unwrap_or_default();
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return ToolOutcome::err(format!("cannot create directories: {e}"));
        }
    }
    if let Err(e) = std::fs::write(&path, content) {
        return ToolOutcome::err(format!("cannot write {rel}: {e}"));
    }
    let diff = make_diff(root, &path, &old, content);
    let summary = format!("wrote {} (+{} −{})", diff.path, diff.added, diff.removed);
    ToolOutcome { ok: true, output: summary, diff: Some(diff) }
}

fn leading_whitespace(s: &str) -> &str {
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if matches!(c, ' ' | '\t') {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    &s[..end]
}

fn line_similarity(a: &str, b: &str) -> f64 {
    let a = a.trim();
    let b = b.trim();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let diff = TextDiff::from_chars(a, b);
    let mut equal = 0usize;
    for change in diff.iter_all_changes() {
        if change.tag() == ChangeTag::Equal {
            equal += change.value().chars().count();
        }
    }
    let total = a.chars().count() + b.chars().count();
    if total == 0 {
        0.0
    } else {
        2.0 * equal as f64 / total as f64
    }
}

fn closest_line_hint(content: &str, old_string: &str) -> String {
    let needle = old_string.lines().next().unwrap_or(old_string);
    let mut best_idx = 0usize;
    let mut best_score = 0.0f64;
    for (i, line) in content.lines().enumerate() {
        let score = line_similarity(line, needle);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    let closest = content.lines().nth(best_idx).unwrap_or("");
    format!(
        "old_string not found. Closest match is at line {}: '{}'. Read the file around that line and retry with the exact text.",
        best_idx + 1,
        truncate(closest, 120)
    )
}

fn find_trimmed_runs(file_lines: &[&str], old_lines: &[&str]) -> Vec<(usize, usize)> {
    if old_lines.is_empty() {
        return Vec::new();
    }
    let n = old_lines.len();
    if file_lines.len() < n {
        return Vec::new();
    }
    let mut runs = Vec::new();
    for start in 0..=file_lines.len() - n {
        if (0..n).all(|i| file_lines[start + i].trim() == old_lines[i].trim()) {
            runs.push((start, start + n));
        }
    }
    runs
}

fn reindent_new_string(new_string: &str, old_string: &str, file_first_matched_line: &str) -> String {
    let old_base = leading_whitespace(old_string.lines().next().unwrap_or(""));
    let file_base = leading_whitespace(file_first_matched_line);
    new_string
        .lines()
        .map(|line| {
            let content = line.trim_start();
            if content.is_empty() && line.is_empty() {
                return String::new();
            }
            let new_ws = leading_whitespace(line);
            let rel = if new_ws.len() >= old_base.len() { &new_ws[old_base.len()..] } else { "" };
            format!("{file_base}{rel}{content}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn replace_line_range(content: &str, start: usize, count: usize, replacement: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let had_trailing_nl = content.ends_with('\n');
    let mut out: Vec<&str> = lines[..start].to_vec();
    out.extend(replacement.lines());
    out.extend_from_slice(&lines[start + count..]);
    let mut result = out.join("\n");
    if had_trailing_nl && !result.is_empty() {
        result.push('\n');
    }
    result
}

fn edit_file(root: &Path, args: &Value) -> ToolOutcome {
    let rel = args["path"].as_str().unwrap_or_default();
    let (Some(old_string), Some(new_string)) = (args["old_string"].as_str(), args["new_string"].as_str()) else {
        return ToolOutcome::err("missing required arguments: old_string and new_string");
    };
    if old_string == new_string {
        return ToolOutcome::err("old_string and new_string are identical");
    }
    let replace_all = args["replace_all"].as_bool().unwrap_or(false);
    let path = match resolve(root, rel) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    let old = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return ToolOutcome::err(format!("cannot read {rel}: {e}")),
    };

    let mut fuzzy_match = false;
    let new = if old.contains(old_string) {
        let count = old.matches(old_string).count();
        if count > 1 && !replace_all {
            return ToolOutcome::err(format!(
                "old_string matches {count} times; provide a longer unique string or set replace_all to true"
            ));
        }
        if replace_all {
            old.replace(old_string, new_string)
        } else {
            old.replacen(old_string, new_string, 1)
        }
    } else {
        let old_lines: Vec<&str> = old_string.lines().collect();
        let file_lines: Vec<&str> = old.lines().collect();
        let runs = find_trimmed_runs(&file_lines, &old_lines);
        if runs.is_empty() {
            return ToolOutcome::err(closest_line_hint(&old, old_string));
        }
        if runs.len() > 1 && !replace_all {
            return ToolOutcome::err(format!(
                "old_string matches {} locations with whitespace normalization; provide a longer unique string or set replace_all to true",
                runs.len()
            ));
        }
        fuzzy_match = true;
        let mut updated = old.clone();
        for (start, end) in runs.iter().rev() {
            let reindented = reindent_new_string(new_string, old_string, file_lines[*start]);
            updated = replace_line_range(&updated, *start, end - start, &reindented);
        }
        updated
    };

    if let Err(e) = std::fs::write(&path, &new) {
        return ToolOutcome::err(format!("cannot write {rel}: {e}"));
    }
    let diff = make_diff(root, &path, &old, &new);
    let suffix = if fuzzy_match { " [matched with whitespace normalization]" } else { "" };
    let summary = format!("edited {} (+{} −{}){}", diff.path, diff.added, diff.removed, suffix);
    ToolOutcome { ok: true, output: summary, diff: Some(diff) }
}

fn project_walk(root: &Path) -> ignore::Walk {
    ignore::WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .max_depth(Some(24))
        .build()
}

/// The subtree a glob can possibly match: its literal prefix up to the last
/// '/' before the first metacharacter (`src/**/*.rs` → `src/`). Agent-issued
/// globs are almost always prefix-scoped, and walking only that subtree
/// instead of the whole project dominates the tool's latency.
///
/// Only plain relative prefixes narrow the walk; absolute or `..`-carrying
/// prefixes fall back to the full project walk, where the matcher (which only
/// ever sees root-relative paths) filters exactly as before. Deliberately no
/// canonicalization here: it would resolve symlinks and break relative
/// display against the un-canonicalized root.
fn glob_walk_root(root: &Path, pattern: &str) -> PathBuf {
    let literal_end = pattern.find(['*', '?', '[', '{']).unwrap_or(pattern.len());
    let prefix = match pattern[..literal_end].rfind('/') {
        Some(i) => &pattern[..i],
        None => return root.to_path_buf(),
    };
    let p = Path::new(prefix);
    let plain_relative =
        !p.is_absolute() && p.components().all(|c| matches!(c, std::path::Component::Normal(_)));
    if plain_relative {
        root.join(p)
    } else {
        root.to_path_buf()
    }
}

fn glob_tool(root: &Path, args: &Value) -> ToolOutcome {
    let Some(pattern) = args["pattern"].as_str() else {
        return ToolOutcome::err("missing required argument: pattern");
    };
    let matcher = match globset::GlobBuilder::new(pattern).literal_separator(false).build() {
        Ok(g) => g.compile_matcher(),
        Err(e) => return ToolOutcome::err(format!("invalid glob: {e}")),
    };
    let walk_root = glob_walk_root(root, pattern);
    let mut hits: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in project_walk(&walk_root).flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let rel = rel_display(root, path);
        if matcher.is_match(&rel) {
            let mtime = entry.metadata().ok().and_then(|m| m.modified().ok()).unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            hits.push((mtime, rel));
        }
    }
    hits.sort_by(|a, b| b.0.cmp(&a.0));
    let total = hits.len();
    let listed: Vec<String> = hits.into_iter().take(MAX_RESULTS).map(|(_, p)| p).collect();
    if listed.is_empty() {
        return ToolOutcome::ok("no files matched".into());
    }
    let mut out = listed.join("\n");
    if total > MAX_RESULTS {
        out.push_str(&format!("\n… {} more matches not shown", total - MAX_RESULTS));
    }
    ToolOutcome::ok(out)
}

fn grep_tool(root: &Path, args: &Value) -> ToolOutcome {
    let Some(pattern) = args["pattern"].as_str() else {
        return ToolOutcome::err("missing required argument: pattern");
    };
    let re = match regex::RegexBuilder::new(pattern).size_limit(1 << 20).build() {
        Ok(r) => r,
        Err(e) => return ToolOutcome::err(format!("invalid regex: {e}")),
    };
    let search_root = match resolve(root, args["path"].as_str().unwrap_or(".")) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    let file_matcher = match args["glob"].as_str() {
        Some(g) => match globset::Glob::new(g) {
            Ok(m) => Some(m.compile_matcher()),
            Err(e) => return ToolOutcome::err(format!("invalid glob: {e}")),
        },
        None => None,
    };
    // Full-corpus scans (rare or no matches) dominate this tool's latency, so
    // walk and scan in parallel. Hits are collected per file and sorted before
    // the cap so the output order is deterministic across runs.
    use std::sync::atomic::{AtomicBool, Ordering};
    let hits: std::sync::Mutex<Vec<(String, usize, String)>> = std::sync::Mutex::new(Vec::new());
    let enough = AtomicBool::new(false);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12);
    ignore::WalkBuilder::new(&search_root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .max_depth(Some(24))
        .threads(threads)
        .build_parallel()
        .run(|| {
            Box::new(|entry| {
                use ignore::WalkState;
                if enough.load(Ordering::Relaxed) {
                    return WalkState::Quit;
                }
                let Ok(entry) = entry else { return WalkState::Continue };
                let path = entry.path();
                if !path.is_file() {
                    return WalkState::Continue;
                }
                if let Some(m) = &file_matcher {
                    let name_match =
                        path.file_name().map(|n| m.is_match(n.as_ref() as &Path)).unwrap_or(false);
                    if !name_match && !m.is_match(rel_display(root, path)) {
                        return WalkState::Continue;
                    }
                }
                if entry.metadata().map(|m| m.len() > MAX_FILE_BYTES).unwrap_or(true) {
                    return WalkState::Continue;
                }
                let Ok(text) = std::fs::read_to_string(path) else {
                    return WalkState::Continue;
                };
                let rel = rel_display(root, path);
                let mut file_hits = Vec::new();
                for (i, line) in text.lines().enumerate() {
                    if re.is_match(line) {
                        file_hits.push((rel.clone(), i + 1, truncate(line.trim(), 300)));
                    }
                }
                if !file_hits.is_empty() {
                    let mut all = hits.lock().unwrap();
                    all.extend(file_hits);
                    if all.len() >= MAX_GREP_RESULTS {
                        enough.store(true, Ordering::Relaxed);
                    }
                }
                WalkState::Continue
            })
        });

    let mut hits = hits.into_inner().unwrap();
    if hits.is_empty() {
        return ToolOutcome::ok("no matches".into());
    }
    hits.sort();
    let capped = hits.len() >= MAX_GREP_RESULTS;
    hits.truncate(MAX_GREP_RESULTS);
    let mut out = String::new();
    for (rel, line_no, line) in hits {
        out.push_str(&format!("{rel}:{line_no}: {line}\n"));
    }
    if capped {
        out.push_str("… result limit reached; refine the pattern\n");
    }
    ToolOutcome::ok(out)
}

/// Truncate oversized command output keeping the TAIL: build and test failures
/// live at the end, and losing them forces the model to re-run the command.
/// The full output is spilled to a log file the model can grep or tail.
pub(crate) fn truncate_command_output(text: &str, max_bytes: usize) -> String {
    let spilled = spill_command_output(text);
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    // Start the kept tail at a line boundary when one is close by.
    if let Some(nl) = text[start..].find('\n') {
        if nl < 200 {
            start += nl + 1;
        }
    }
    let note = match spilled {
        Some(path) => format!(
            "[start of output truncated; full output saved to {} — tail or grep it with bash]",
            path.display()
        ),
        None => "[start of output truncated]".to_string(),
    };
    format!("{note}\n…{}", &text[start..])
}

/// Write the complete output of an oversized command to the data dir so it
/// stays inspectable after truncation. Best effort: None if it cannot be saved.
fn spill_command_output(text: &str) -> Option<PathBuf> {
    let dir = crate::state::default_data_dir().join("cmd-logs");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("cmd-{}.log", uuid::Uuid::new_v4()));
    std::fs::write(&path, text).ok()?;
    Some(path)
}

async fn bash_tool(root: &Path, args: &Value, caps: OutputCaps, cancel: Arc<CancelToken>) -> ToolOutcome {
    let Some(command) = args["command"].as_str() else {
        return ToolOutcome::err("missing required argument: command");
    };
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(60).clamp(1, 300);
    // Prefer zsh (macOS default), then bash, then sh for portable Linux CI/hosts.
    let shell = ["/bin/zsh", "/bin/bash", "/bin/sh"]
        .into_iter()
        .find(|p| Path::new(p).exists())
        .unwrap_or("/bin/sh");
    let mut cmd = tokio::process::Command::new(shell);
    cmd.arg("-lc")
        .arg(command)
        .current_dir(root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return ToolOutcome::err(format!("failed to spawn shell: {e}")),
    };
    let mut child_slot = Some(child);
    tokio::select! {
        _ = cancel.cancelled() => {
            if let Some(mut c) = child_slot.take() {
                let _ = c.kill().await;
            }
            ToolOutcome::err("command cancelled by user")
        }
        _ = tokio::time::sleep(Duration::from_secs(timeout_secs)) => {
            if let Some(mut c) = child_slot.take() {
                let _ = c.kill().await;
            }
            ToolOutcome::err(format!("command timed out after {timeout_secs}s"))
        }
        output = async {
            child_slot.take().expect("child taken twice").wait_with_output().await
        } => {
            match output {
                Err(e) => ToolOutcome::err(format!("command failed: {e}")),
                Ok(output) => {
                    let mut text = String::new();
                    text.push_str(&String::from_utf8_lossy(&output.stdout));
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if !stderr.trim().is_empty() {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str("[stderr]\n");
                        text.push_str(&stderr);
                    }
                    if text.len() > caps.command_bytes {
                        text = truncate_command_output(&text, caps.command_bytes);
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
    use serde_json::json;

    fn temp_project() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-tools-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // macOS temp dirs live behind a symlink; the tools compare walked
        // paths against the root, so hand them the physical path.
        let dir = dir.canonicalize().unwrap();
        std::fs::create_dir_all(dir.join("src/deep")).unwrap();
        std::fs::create_dir_all(dir.join("docs")).unwrap();
        std::fs::write(dir.join("src/a.rs"), "fn alpha() {}\nfn alpha_two() {}\n").unwrap();
        std::fs::write(dir.join("src/deep/b.rs"), "fn alpha_three() {}\n").unwrap();
        std::fs::write(dir.join("docs/c.md"), "alpha in prose\n").unwrap();
        dir
    }

    #[test]
    fn glob_walk_root_uses_literal_prefix() {
        let root = temp_project();
        assert_eq!(glob_walk_root(&root, "src/**/*.rs"), root.join("src"));
        assert_eq!(glob_walk_root(&root, "src/deep/*.rs"), root.join("src/deep"));
        // No literal directory prefix: the whole project.
        assert_eq!(glob_walk_root(&root, "**/*.rs"), root);
        assert_eq!(glob_walk_root(&root, "README.md"), root);
        // Escaping or absolute prefixes fall back to the full (safe) walk.
        assert_eq!(glob_walk_root(&root, "../elsewhere/*.rs"), root);
        assert_eq!(glob_walk_root(&root, "/etc/*.conf"), root);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn glob_scoped_pattern_finds_nested_files() {
        let root = temp_project();
        let out = glob_tool(&root, &json!({"pattern": "src/**/*.rs"}));
        assert!(out.ok);
        assert!(out.output.contains("src/a.rs"), "{}", out.output);
        assert!(out.output.contains("src/deep/b.rs"), "{}", out.output);
        assert!(!out.output.contains("docs/c.md"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn grep_output_is_sorted_and_complete() {
        let root = temp_project();
        let out = grep_tool(&root, &json!({"pattern": "alpha"}));
        assert!(out.ok);
        let lines: Vec<&str> = out.output.lines().collect();
        assert_eq!(lines.len(), 4, "{}", out.output);
        let mut sorted = lines.clone();
        sorted.sort();
        assert_eq!(lines, sorted, "results must be deterministic (path, line) order");
        assert!(lines[0].starts_with("docs/c.md:1:"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn grep_caps_results_with_notice() {
        let root = temp_project();
        let mut big = String::new();
        for i in 0..(MAX_GREP_RESULTS + 20) {
            big.push_str(&format!("alpha line {i}\n"));
        }
        std::fs::write(root.join("big.txt"), big).unwrap();
        let out = grep_tool(&root, &json!({"pattern": "alpha", "glob": "*.txt"}));
        assert!(out.ok);
        assert!(out.output.contains("result limit reached"), "{}", out.output);
        let hits = out.output.lines().filter(|l| l.contains("big.txt")).count();
        assert_eq!(hits, MAX_GREP_RESULTS, "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn command_truncation_keeps_the_tail() {
        let mut text = String::new();
        for i in 0..4000 {
            text.push_str(&format!("line number {i} with some padding text\n"));
        }
        assert!(text.len() > MAX_OUTPUT_BYTES);
        let kept = truncate_command_output(&text, MAX_OUTPUT_BYTES);
        assert!(kept.len() < text.len());
        assert!(kept.contains("line number 3999"), "the end of the output must survive");
        assert!(!kept.contains("line number 0 "), "the head is what gets dropped");
        assert!(kept.starts_with("[start of output truncated"), "{}", &kept[..120]);
    }

    #[tokio::test]
    async fn bash_failure_preserves_tail_of_output() {
        let root = temp_project();
        // 40k+ bytes of output with the failure marker at the very end.
        let cmd = "for i in $(seq 1 2000); do echo \"noise line $i padded out a bit\"; done; echo THE_REAL_FAILURE; exit 3";
        let out = bash_tool(
            &root,
            &json!({"command": cmd}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(!out.ok);
        assert!(out.output.starts_with("exit code 3"), "{}", &out.output[..60]);
        assert!(out.output.contains("THE_REAL_FAILURE"), "tail must survive truncation");
        assert!(!out.output.contains("noise line 1 "), "head should be dropped");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn read_file_stops_at_byte_cap() {
        let root = std::env::temp_dir().join(format!("openmax-read-bytes-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let long_line = "x".repeat(400);
        let mut content = String::new();
        for _ in 0..100 {
            content.push_str(&long_line);
            content.push('\n');
        }
        std::fs::write(root.join("big.txt"), &content).unwrap();
        let out = read_file(&root, &json!({"path": "big.txt"}));
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("output limit reached at line"), "{}", out.output);
        assert!(out.output.contains("continue with offset="), "{}", out.output);
        assert!(out.output.len() <= MAX_READ_BYTES + 200, "{}", out.output.len());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn list_dir_caps_entries() {
        let root = std::env::temp_dir().join(format!("openmax-listdir-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        for i in 0..MAX_DIR_ENTRIES + 50 {
            std::fs::write(root.join(format!("file{i:03}.txt")), "x").unwrap();
        }
        let out = list_dir(&root, &json!({"path": "."}));
        assert!(out.ok, "{}", out.output);
        let lines: Vec<&str> = out.output.lines().collect();
        assert_eq!(lines.len(), MAX_DIR_ENTRIES + 1, "{}", out.output);
        assert!(out.output.contains("more entries not shown"), "{}", out.output);
        assert!(out.output.contains("use glob to find specific files"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn edit_file_tier2_whitespace_match_preserves_indent() {
        let root = std::env::temp_dir().join(format!("openmax-edit-fuzzy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        std::fs::write(root.join("src.rs"), "fn outer() {\n    fn inner() {\n        old_value\n    }\n}\n").unwrap();
        let out = edit_file(
            &root,
            &json!({
                "path": "src.rs",
                "old_string": "fn inner() {\n    old_value\n}",
                "new_string": "fn inner() {\n    new_value\n}"
            }),
        );
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("[matched with whitespace normalization]"), "{}", out.output);
        let content = std::fs::read_to_string(root.join("src.rs")).unwrap();
        assert!(content.contains("        new_value\n"), "indent must be preserved: {content:?}");
        assert!(!content.contains("old_value"), "{}", content);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn edit_file_tier2_ambiguity_error() {
        let root = std::env::temp_dir().join(format!("openmax-edit-ambig-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        std::fs::write(
            root.join("dup.rs"),
            "    fn foo() {\n        a\n    }\nfn bar() {}\n    fn foo() {\n        a\n    }\n",
        )
        .unwrap();
        let out = edit_file(
            &root,
            &json!({
                "path": "dup.rs",
                "old_string": "fn foo() {\n    a\n}",
                "new_string": "fn foo() {\n    b\n}"
            }),
        );
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("whitespace normalization"), "{}", out.output);
        assert!(out.output.contains("2 locations"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn edit_file_closest_match_hint_in_error() {
        let root = std::env::temp_dir().join(format!("openmax-edit-hint-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        std::fs::write(root.join("hint.rs"), "fn almost_match() {}\nfn unrelated() {}\n").unwrap();
        let out = edit_file(
            &root,
            &json!({
                "path": "hint.rs",
                "old_string": "fn almost_matched() {}",
                "new_string": "fn almost_matched() { /* x */ }"
            }),
        );
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("Closest match is at line 1"), "{}", out.output);
        assert!(out.output.contains("almost_match"), "{}", out.output);
        assert!(out.output.contains("Read the file around that line"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn readonly_execute_returns_immediately_when_cancelled() {
        use std::sync::Arc;

        use crate::state::CancelToken;

        let cancel = Arc::new(CancelToken::default());
        cancel.cancel();
        let root = temp_project();
        let out = execute("glob", &json!({"pattern": "**/*.rs"}), &root, OutputCaps::default(), cancel).await;
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("cancelled"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }
}
