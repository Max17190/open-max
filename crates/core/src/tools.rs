//! The seven built-in tools and their wire schemas.
//!
//! `TOOL_NAMES` and `tool_schemas()` are fixed and asserted in lockstep by
//! `registry`, because this array is serialized into every request for the
//! life of a session: a tool added here is paid for by every user on every
//! request, forever. That is the bar for entry, and it is why the set stops at
//! reading, writing, searching, and running commands. Optional capability goes
//! on the tool-file surface instead, where only the projects that install it
//! pay.
//!
//! Schemas are kept deliberately small and strict. Fewer, simpler parameters
//! measurably help smaller models, and every character is prompt cost.
//!
//! Output is bounded rather than trusted: a command that prints a gigabyte is
//! captured to a cap, tail-first, and the remainder spills to a file under the
//! session's data dir with a breadcrumb in the result. Truncation always says
//! so, so the model can tell a short answer from a clipped one.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::execution::{
    self, CaptureSpec, ProcessError, ProcessOutput, ProcessRequest, StdinMode, Termination,
};
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
#[derive(Default)]
pub struct ToolOutcome {
    pub ok: bool,
    pub output: String,
    pub diff: Option<DiffInfo>,
    /// Bytes the underlying process produced, when the tool ran one. `output`
    /// is a bounded rendering of that, so the two differ for a noisy command.
    /// None for tools that are not a process (file and search built-ins).
    pub process_bytes: Option<u64>,
    /// True when `output` dropped part of what the process produced.
    pub process_truncated: bool,
}

impl ToolOutcome {
    pub(crate) fn ok(output: String) -> Self {
        Self { ok: true, output, ..Self::default() }
    }
    pub(crate) fn err(output: impl Into<String>) -> Self {
        Self { ok: false, output: output.into(), ..Self::default() }
    }
    /// A result that carries none of the process output: the user cancelled
    /// the call, so what it printed is deliberately not spent back into the
    /// context. The fields still record that the output happened, so a hook
    /// can tell a quiet command from a silenced one.
    pub(crate) fn from_killed_process(output: impl Into<String>, process: &ProcessOutput) -> Self {
        let produced = process.stdout.total_bytes.saturating_add(process.stderr.total_bytes);
        Self {
            ok: false,
            output: output.into(),
            diff: None,
            process_bytes: Some(produced),
            process_truncated: produced > 0,
        }
    }

    /// Record what the process behind this result produced, so a
    /// `post_tool_use` hook can tell a quiet command from a clipped one
    /// without parsing the truncation notice out of the text.
    pub(crate) fn from_process(
        ok: bool,
        output: String,
        process: &ProcessOutput,
        truncated: bool,
    ) -> Self {
        Self {
            ok,
            output,
            diff: None,
            process_bytes: Some(
                process.stdout.total_bytes.saturating_add(process.stderr.total_bytes),
            ),
            process_truncated: truncated,
        }
    }
}

/// True for tools that can change state and therefore go through approval.
pub fn is_mutating(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file" | "bash")
}

/// Every built-in tool name. Order matches `tool_schemas()` so the frozen
/// schema array and registry stay in lockstep.
pub const TOOL_NAMES: &[&str] =
    &["list_dir", "read_file", "write_file", "edit_file", "glob", "grep", "bash"];

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
        }
    ])
    })
}

/// Resolve a model-supplied path, refusing escapes from the project root.
///
/// A ROOT-ABSOLUTE path names itself, not a path relative to the root.
/// Stripping its leading `/` re-rooted it: under root `/app`, `/app/x` became
/// `/app/app/x`, and the call still reported success, so `read_file` said "No
/// such file" for a file that existed and `write_file` landed bytes at a path
/// the model never named. `Path::join` already does the right thing here (an
/// absolute argument replaces the root), and the escape check below is what
/// decides the path is allowed: absolute paths under the root resolve to
/// themselves, and one outside is refused as an escape instead of being
/// silently rewritten into the project.
fn resolve(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim();
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
    data_dir: &Path,
    root: &Path,
    caps: OutputCaps,
    cancel: Arc<CancelToken>,
) -> ToolOutcome {
    if name == "bash" {
        return bash_tool(data_dir, root, args, caps, cancel).await;
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
    // An offset past the end must not read like an empty file: the model
    // would conclude the content is gone rather than that its offset is
    // stale.
    if offset > total && total > 0 {
        return ToolOutcome::err(format!(
            "offset {offset} is past the end of {rel} ({total} lines); retry with a smaller offset"
        ));
    }
    let mut out = String::new();
    let mut stopped_by_bytes = false;
    let mut byte_cap_line = 0usize;
    for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
        // A clipped line must say so: silently dropping its tail sends the
        // model into edit_file with an old_string that can never match.
        let formatted = if line.len() > MAX_LINE_CHARS {
            let end = floor_char(line, MAX_LINE_CHARS);
            format!("{:>5} {}… [line clipped; {} more bytes]\n", i + 1, &line[..end], line.len() - end)
        } else {
            format!("{:>5} {}\n", i + 1, line)
        };
        if out.len() + formatted.len() > MAX_READ_BYTES {
            stopped_by_bytes = true;
            byte_cap_line = i + 1;
            break;
        }
        out.push_str(&formatted);
    }
    if stopped_by_bytes {
        // `byte_cap_line` is the first line that did NOT fit, so the
        // continuation resumes exactly there. The former `+ 1` skipped one
        // line per capped read, and pointed past EOF when the cap landed on
        // the final line.
        out.push_str(&format!(
            "… output limit reached at line {byte_cap_line} (file has {total} lines; continue with offset={byte_cap_line})\n"
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
    ToolOutcome { ok: true, output: summary, diff: Some(diff), ..Default::default() }
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
    ToolOutcome { ok: true, output: summary, diff: Some(diff), ..Default::default() }
}

/// Hidden files are searchable: the agent's own extension surface lives in
/// dot-directories (`.openmax/tools`, `.agents`, `.github`), and a walker
/// that skips them makes the agent blind to the capabilities it wrote.
/// `.git` alone is excluded by name; gitignore rules still apply.
fn project_walk(root: &Path) -> ignore::Walk {
    ignore::WalkBuilder::new(root)
        .hidden(false)
        .filter_entry(|e| e.file_name() != std::ffi::OsStr::new(".git"))
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

/// True when `rel` (a root-relative path) names `.git` or anything inside it.
fn touches_git(rel: &Path) -> bool {
    rel.components().any(|c| c.as_os_str() == std::ffi::OsStr::new(".git"))
}

/// Model-issued patterns routinely arrive scoped `./like/this` or
/// `/like/this`. Matching runs against root-relative paths, so either prefix
/// makes a pattern that can never match anything; both mean
/// project-root-relative here.
///
/// This is deliberately NOT what `resolve()` does with a path argument. A glob
/// is matched against root-relative candidates, so there is no such thing as an
/// absolute pattern to honor, while a root-absolute path argument names a real
/// location and resolves to itself.
fn normalize_pattern(pattern: &str) -> &str {
    let mut p = pattern;
    loop {
        let trimmed = p.trim_start_matches('/').trim_start_matches("./");
        if trimmed == p {
            return trimmed;
        }
        p = trimmed;
    }
}

fn glob_tool(root: &Path, args: &Value) -> ToolOutcome {
    let Some(pattern) = args["pattern"].as_str() else {
        return ToolOutcome::err("missing required argument: pattern");
    };
    let pattern = normalize_pattern(pattern);
    // "", "/", "./" and friends all normalize to nothing. An empty glob can
    // never match, and answering "no files matched" would read as a fact
    // about the project rather than about the pattern.
    if pattern.is_empty() {
        return ToolOutcome::err("empty glob pattern; give a pattern like \"**/*.rs\"");
    }
    let matcher = match globset::GlobBuilder::new(pattern).literal_separator(false).build() {
        Ok(g) => g.compile_matcher(),
        Err(e) => return ToolOutcome::err(format!("invalid glob: {e}")),
    };
    let walk_root = glob_walk_root(root, pattern);
    // The walker's filter skips entries named .git during descent, but never
    // the walk root itself, so a pattern scoped at or under .git would start
    // inside the excluded tree. Canonicalizing catches a symlinked prefix
    // that aliases .git without naming it (walk roots are followed even
    // though the walk itself never follows links).
    if walk_root.as_path() != root {
        let scoped_into_git =
            walk_root.strip_prefix(root).map(touches_git).unwrap_or(true);
        let aliases_git = match (walk_root.canonicalize(), root.canonicalize()) {
            (Ok(canon), Ok(root_canon)) => {
                canon.strip_prefix(&root_canon).map(touches_git).unwrap_or(true)
            }
            // A nonexistent prefix walks nothing; let the normal path answer.
            _ => false,
        };
        if scoped_into_git || aliases_git {
            return ToolOutcome::err(".git is excluded from search");
        }
    }
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
    hits.sort_by_key(|h| std::cmp::Reverse(h.0));
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
    // resolve() canonicalized, so a path (or a symlink) that lands inside
    // .git names it here even when the argument never did. The walker's
    // filter cannot help once .git is the walk root.
    let inside_git = match root.canonicalize() {
        Ok(root_canon) => search_root.strip_prefix(&root_canon).map(touches_git).unwrap_or(false),
        Err(_) => false,
    };
    if inside_git {
        return ToolOutcome::err(".git is excluded from search");
    }
    let file_matcher = match args["glob"].as_str() {
        Some(g) => {
            let g = normalize_pattern(g);
            // An empty filter matches nothing; "no matches" would blame the
            // regex when the filter excluded every file up front.
            if g.is_empty() {
                return ToolOutcome::err("empty glob filter; give a pattern like \"*.rs\"");
            }
            match globset::Glob::new(g) {
                Ok(m) => Some(m.compile_matcher()),
                Err(e) => return ToolOutcome::err(format!("invalid glob: {e}")),
            }
        }
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
        // Same visibility contract as `project_walk`: hidden files are
        // searchable, `.git` alone is excluded by name.
        .hidden(false)
        .filter_entry(|e| e.file_name() != std::ffi::OsStr::new(".git"))
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

/// Truncate text for command rendering while keeping its tail. The process
/// supervisor owns any bounded spill log; this helper never writes files.
fn truncate_rendered_command_output(text: &str, max_bytes: usize, log_path: Option<&PathBuf>) -> String {
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
    let note = match log_path {
        Some(path) => format!(
            "[start of output truncated; bounded output log saved to {}; tail or grep it with bash]",
            path.display()
        ),
        None => "[start of output truncated]".to_string(),
    };
    format!("{note}\n…{}", &text[start..])
}

fn captured_text(stream: &execution::CapturedStream) -> String {
    String::from_utf8_lossy(&stream.rendered_bytes()).into_owned()
}

fn stream_was_truncated(stream: &execution::CapturedStream) -> bool {
    stream.total_bytes > stream.head.len().saturating_add(stream.tail.len()) as u64
}

/// Format native-process output identically for bash and external tools.
/// The supervisor has already bounded each stream and owns any spill log.
pub(crate) fn render_process_output(output: &ProcessOutput, max_bytes: usize) -> (String, bool) {
    let mut text = captured_text(&output.stdout);
    let stderr = captured_text(&output.stderr);
    if !stderr.trim().is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("[stderr]\n");
        text.push_str(&stderr);
    }

    let needs_notice = output.log_truncated
        || stream_was_truncated(&output.stdout)
        || stream_was_truncated(&output.stderr)
        || text.len() > max_bytes;
    if text.len() > max_bytes {
        text = truncate_rendered_command_output(&text, max_bytes, output.log_path.as_ref());
    } else if needs_notice {
        if let Some(path) = &output.log_path {
            text = format!(
                "[start of output truncated; bounded output log saved to {}; tail or grep it with bash]\n…{text}",
                path.display()
            );
        } else {
            text = format!("[start of output truncated]\n…{text}");
        }
    }
    if text.trim().is_empty() {
        ("(no output)".into(), needs_notice)
    } else {
        (text, needs_notice)
    }
}

async fn bash_tool(
    data_dir: &Path,
    root: &Path,
    args: &Value,
    caps: OutputCaps,
    cancel: Arc<CancelToken>,
) -> ToolOutcome {
    let Some(command) = args["command"].as_str() else {
        return ToolOutcome::err("missing required argument: command");
    };
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(60).clamp(1, 300);
    // Prefer zsh (macOS default), then bash, then sh for portable Linux CI/hosts.
    let shell = ["/bin/zsh", "/bin/bash", "/bin/sh"]
        .into_iter()
        .find(|p| Path::new(p).exists())
        .unwrap_or("/bin/sh");
    let request = ProcessRequest {
        program: shell.into(),
        args: vec!["-lc".into(), command.into()],
        cwd: root.to_path_buf(),
        stdin: StdinMode::Null,
        timeout: std::time::Duration::from_secs(timeout_secs),
        capture: CaptureSpec {
            head_bytes: 0,
            tail_bytes: caps.command_bytes,
            spill_dir: Some(data_dir.join("cmd-logs")),
            spill_bytes_per_stream: 16 * 1024 * 1024,
        },
        sandbox: None,
        env_allowlist: None,
    };
    match execution::run_process(request, cancel).await {
        Err(ProcessError::Spawn(e)) => ToolOutcome::err(format!("failed to spawn shell: {e}")),
        Err(ProcessError::Wait(e)) => ToolOutcome::err(format!("command failed: {e}")),
        // bash never runs sandboxed (request.sandbox is None above); keep
        // the honest message should that ever change.
        Err(e @ ProcessError::SandboxUnavailable(_)) => ToolOutcome::err(e.to_string()),
        Ok(output) => match &output.termination {
            Termination::Cancelled => {
                ToolOutcome::from_killed_process("command cancelled by user", &output)
            }
            // A hung command's last output is the diagnostic: which test was
            // running, what it was waiting on. The tail is already captured
            // when the timeout fires, so dropping it would turn a measurable
            // failure into a guess.
            Termination::TimedOut => {
                let (text, truncated) = render_process_output(&output, caps.command_bytes);
                ToolOutcome::from_process(
                    false,
                    format!("command timed out after {timeout_secs}s; output until the kill:\n{text}"),
                    &output,
                    truncated,
                )
            }
            Termination::Exited(status) => {
                // The note is the harness speaking, not captured output, but it
                // still has to fit the caller's cap, which `max_output_bytes`
                // can set as low as 1000 bytes. Reserve its length instead of
                // letting a fixed annotation push the result past the limit.
                let reserved = match output.background_terminated {
                    true => BACKGROUND_TERMINATED_NOTE.len() + 1,
                    false => 0,
                };
                let budget = caps.command_bytes.saturating_sub(reserved).max(1);
                let (text, truncated) = render_process_output(&output, budget);
                let (ok, text) = match status.success() {
                    true => (true, text),
                    false => (false, format!("{}\n{text}", describe_exit(status))),
                };
                // A backgrounded server dies with the call and the exit status
                // is still 0, so without this the next step is a request to a
                // port nothing is listening on.
                let text = match output.background_terminated {
                    true => format!("{text}\n{BACKGROUND_TERMINATED_NOTE}"),
                    false => text,
                };
                ToolOutcome::from_process(ok, text, &output, truncated)
            }
        },
    }
}

/// Each bash call runs in its own process group, and the group is terminated
/// when the call returns, so a backgrounded process does not outlive it. The
/// exit status is still the shell's, which means a caller that started a server
/// sees success and an absent server, with nothing connecting the two.
///
/// `setsid` is named as a conditional, not a recipe: it is util-linux and does
/// not exist on macOS, where the harness also runs. A named tmux session is the
/// answer that holds on both, and is what this project already documents for
/// durable background work.
pub(crate) const BACKGROUND_TERMINATED_NOTE: &str = concat!(
    "[openmax: this command left running background processes, and they were ",
    "terminated when it returned. Every bash call runs in its own process group ",
    "and that group is cleaned up on exit, so `&`, `nohup` and `disown` do not ",
    "survive the call. To keep something running, start it in a named tmux ",
    "session you can inspect and reattach, or, on Linux only, detach it from the ",
    "group with `setsid`.]"
);

/// Describe a non-success exit honestly. A signal kill has no exit code, and
/// the former "exit code -1" pointed diagnosis at a code nothing returned;
/// naming the signal turns a segfault or an OOM kill into a readable fact.
pub(crate) fn describe_exit(status: &std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            // Only numbers that are identical on Linux and macOS get a name;
            // platform-divergent ones (e.g. SIGBUS: 7 vs 10) stay numeric
            // rather than risk a wrong label.
            let name = match signal {
                1 => " (SIGHUP)",
                2 => " (SIGINT)",
                4 => " (SIGILL)",
                6 => " (SIGABRT)",
                8 => " (SIGFPE)",
                9 => " (SIGKILL)",
                11 => " (SIGSEGV)",
                13 => " (SIGPIPE)",
                14 => " (SIGALRM)",
                15 => " (SIGTERM)",
                24 => " (SIGXCPU)",
                _ => "",
            };
            return format!("killed by signal {signal}{name}");
        }
    }
    format!("exit code {}", status.code().unwrap_or(-1))
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

    /// Models routinely scope patterns "./like/this" or "/like/this"; both
    /// must mean project-root-relative rather than silently matching nothing.
    #[test]
    fn scoped_pattern_prefixes_are_normalized() {
        let root = temp_project();
        let out = glob_tool(&root, &json!({"pattern": "./src/**/*.rs"}));
        assert!(out.ok && out.output.contains("src/a.rs"), "{}", out.output);
        let out = glob_tool(&root, &json!({"pattern": "/src/*.rs"}));
        assert!(out.ok && out.output.contains("src/a.rs"), "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "alpha", "glob": "./src/*.rs"}));
        assert!(out.ok && out.output.contains("src/a.rs:1:"), "{}", out.output);
        // Normalization happens before the .git refusal, not instead of it.
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/config"), "x\n").unwrap();
        let out = glob_tool(&root, &json!({"pattern": "./.git/config"}));
        assert!(!out.ok && out.output.contains("excluded from search"), "{}", out.output);
        // The normalizer's own shape.
        assert_eq!(normalize_pattern("././x"), "x");
        assert_eq!(normalize_pattern(".//x"), "x");
        assert_eq!(normalize_pattern(".git/x"), ".git/x");
        assert_eq!(normalize_pattern("**/*.rs"), "**/*.rs");
        // A pattern that is nothing but scope prefixes cannot match anything;
        // saying "no files matched" would read as a fact about the project.
        for empty in ["", "/", "./", "/./", ".//"] {
            let out = glob_tool(&root, &json!({"pattern": empty}));
            assert!(!out.ok, "{empty:?}: {}", out.output);
            assert!(out.output.contains("empty glob pattern"), "{empty:?}: {}", out.output);
            let out = grep_tool(&root, &json!({"pattern": "alpha", "glob": empty}));
            assert!(!out.ok, "{empty:?}: {}", out.output);
            assert!(out.output.contains("empty glob filter"), "{empty:?}: {}", out.output);
        }
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

    /// The extension surface lives in dot-directories. A search that skips
    /// them makes the agent blind to the capabilities it wrote, so hidden
    /// files must be visible to glob and grep while `.git` never is.
    #[test]
    fn glob_and_grep_see_hidden_files_but_never_git() {
        let root = temp_project();
        std::fs::create_dir_all(root.join(".github/workflows")).unwrap();
        std::fs::write(root.join(".github/workflows/ci.yml"), "name: ci\n").unwrap();
        std::fs::create_dir_all(root.join(".openmax/tools")).unwrap();
        std::fs::write(root.join(".openmax/tools/fetch.toml"), "name = \"fetch_page\"\n").unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/config"), "fetch_page in git internals\n").unwrap();

        let out = glob_tool(&root, &json!({"pattern": "**/*.yml"}));
        assert!(out.output.contains(".github/workflows/ci.yml"), "{}", out.output);
        let out = glob_tool(&root, &json!({"pattern": "**/*.toml"}));
        assert!(out.output.contains(".openmax/tools/fetch.toml"), "{}", out.output);
        let out = glob_tool(&root, &json!({"pattern": "**/*"}));
        assert!(!out.output.contains(".git/"), "{}", out.output);

        let out = grep_tool(&root, &json!({"pattern": "fetch_page"}));
        assert!(out.output.contains(".openmax/tools/fetch.toml:1:"), "{}", out.output);
        assert!(!out.output.contains(".git/config"), "{}", out.output);

        // Scoping a search at or under .git must not sidestep the walker's
        // filter, which never sees the walk root itself.
        let out = glob_tool(&root, &json!({"pattern": ".git/config"}));
        assert!(!out.ok && out.output.contains("excluded from search"), "{}", out.output);
        let out = glob_tool(&root, &json!({"pattern": ".git/**"}));
        assert!(!out.ok && out.output.contains("excluded from search"), "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "fetch_page", "path": ".git"}));
        assert!(!out.ok && out.output.contains("excluded from search"), "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "fetch_page", "path": ".git/hooks"}));
        assert!(!out.ok && out.output.contains("excluded from search"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A symlink inside the project can alias .git without naming it; the
    /// canonical path is the authority for the exclusion.
    #[cfg(unix)]
    #[test]
    fn scoped_symlink_to_git_is_still_excluded() {
        let root = temp_project();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/config"), "fetch_page in git internals\n").unwrap();
        std::os::unix::fs::symlink(root.join(".git"), root.join("gitlink")).unwrap();

        let out = grep_tool(&root, &json!({"pattern": "fetch_page", "path": "gitlink"}));
        assert!(!out.ok && out.output.contains("excluded from search"), "{}", out.output);
        let out = glob_tool(&root, &json!({"pattern": "gitlink/*"}));
        assert!(!out.ok && out.output.contains("excluded from search"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn command_truncation_keeps_the_tail() {
        let mut text = String::new();
        for i in 0..4000 {
            text.push_str(&format!("line number {i} with some padding text\n"));
        }
        assert!(text.len() > MAX_OUTPUT_BYTES);
        let kept = truncate_rendered_command_output(&text, MAX_OUTPUT_BYTES, None);
        assert!(kept.len() < text.len());
        assert!(kept.contains("line number 3999"), "the end of the output must survive");
        assert!(!kept.contains("line number 0 "), "the head is what gets dropped");
        assert!(kept.starts_with("[start of output truncated"), "{}", &kept[..120]);
    }

    fn rendered_output(
        stdout: execution::CapturedStream,
        stderr: execution::CapturedStream,
        log_path: Option<PathBuf>,
        log_truncated: bool,
    ) -> ProcessOutput {
        let status = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("exit 0")
            .status()
            .unwrap();
        ProcessOutput {
            termination: Termination::Exited(status),
            stdout,
            stderr,
            log_path,
            log_truncated,
            // This helper builds outputs for rendering tests; none of them
            // background anything.
            background_terminated: false,
        }
    }

    fn stream(total_bytes: u64, head: &[u8], tail: &[u8]) -> execution::CapturedStream {
        execution::CapturedStream {
            total_bytes,
            head: head.to_vec(),
            tail: tail.to_vec(),
        }
    }

    #[test]
    fn process_renderer_does_not_duplicate_overlapping_head_and_tail() {
        let output = rendered_output(stream(3, b"ab", b"abc"), stream(0, b"", b""), None, false);
        assert_eq!(render_process_output(&output, 100).0, "abc");
    }

    #[test]
    fn process_renderer_labels_stderr_after_stdout() {
        let output = rendered_output(
            stream(6, b"", b"stdout"),
            stream(6, b"", b"stderr"),
            None,
            false,
        );
        assert_eq!(render_process_output(&output, 100).0, "stdout\n[stderr]\nstderr");
    }

    #[test]
    fn process_renderer_marks_truncated_capture_without_log() {
        let output = rendered_output(stream(100, b"", b"tail"), stream(0, b"", b""), None, false);
        let (text, _) = render_process_output(&output, 100);
        assert!(text.starts_with("[start of output truncated]"), "{text}");
        assert!(text.ends_with("tail"), "{text}");
    }

    #[test]
    fn process_renderer_points_to_bounded_log_when_available() {
        let path = PathBuf::from("/tmp/openmax-command.log");
        let output = rendered_output(
            stream(100, b"", b"tail"),
            stream(0, b"", b""),
            Some(path),
            false,
        );
        let (text, _) = render_process_output(&output, 100);
        assert!(
            text.contains("bounded output log saved to /tmp/openmax-command.log"),
            "{text}"
        );
    }

    /// A command that succeeds reports its size just as a failing one does.
    /// Only a process-backed tool can, so the file tools stay empty.
    #[tokio::test]
    async fn a_successful_command_reports_what_it_produced() {
        let root = temp_project();
        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "for i in $(seq 1 2000); do echo \"noise line $i padded out a bit\"; done"}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(out.ok, "{}", out.output);
        let produced = out.process_bytes.expect("a command reports its own size");
        assert!(produced > out.output.len() as u64, "the result is a bounded rendering");
        assert!(out.process_truncated, "and it says so, without parsing the notice");

        let quiet = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "printf 'hi\\n'"}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert_eq!(quiet.process_bytes, Some(3));
        assert!(!quiet.process_truncated, "a short command is distinguishable from a clipped one");

        let read = read_file(&root, &json!({"path": "src/a.rs"}));
        assert!(read.process_bytes.is_none(), "no process ran behind a file read");
        assert!(!read.process_truncated);

        let _ = std::fs::remove_dir_all(root);
    }

    /// A hung command's last output is the diagnostic: which test was
    /// running, what it was waiting on. The tail is captured before the kill,
    /// so the result must carry it instead of reporting only that time ran
    /// out.
    ///
    /// The timeout has to outlast shell startup by a wide margin: on a loaded
    /// CI runner a one-second budget can expire before `echo` ever runs, and
    /// then there is nothing captured for the result to carry.
    #[tokio::test]
    async fn a_timed_out_command_reports_the_tail_it_captured() {
        let root = temp_project();
        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "echo before-the-timeout; sleep 30", "timeout_secs": 5}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;

        assert!(!out.ok);
        assert!(out.output.contains("timed out after 5s"), "{}", out.output);
        assert!(
            out.output.contains("before-the-timeout"),
            "the captured tail must survive the kill: {}",
            out.output
        );
        assert_eq!(
            out.process_bytes,
            Some("before-the-timeout\n".len() as u64),
            "the bytes it managed to print still happened"
        );
        assert!(!out.process_truncated, "everything printed made it into the result");
        let _ = std::fs::remove_dir_all(root);
    }

    /// A segfault or an OOM kill has no exit code; reporting "exit code -1"
    /// pointed diagnosis at a code nothing returned. The signal is the fact.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_signal_killed_command_names_the_signal() {
        let root = temp_project();
        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "echo about-to-die; kill -SEGV $$"}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(!out.ok);
        assert!(
            out.output.starts_with("killed by signal 11 (SIGSEGV)"),
            "{}",
            out.output
        );
        assert!(out.output.contains("about-to-die"), "output before the kill survives: {}", out.output);
        assert!(!out.output.contains("exit code"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A backgrounded process does not outlive the call that started it: the
    /// process group is cleaned up on exit while the exit status stays the
    /// shell's zero. A caller that started a server was handed success and no
    /// server, with nothing in the result connecting the two, so it retried and
    /// then reported success against a port nothing was listening on.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_terminated_background_process_is_reported() {
        let root = temp_project();
        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "sleep 30 & echo started"}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        // The command succeeded. That is precisely why the note has to exist:
        // success is what the caller would otherwise act on.
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("started"), "{}", out.output);
        assert!(
            out.output.contains("were terminated when it returned"),
            "a killed background process must be reported: {}",
            out.output
        );
        // The escape it names has to exist on this platform; `setsid` is
        // util-linux and absent on macOS, so tmux is the one always offered.
        assert!(out.output.contains("tmux"), "{}", out.output);

        // The note is reserved out of the cap, not added on top of it: a small
        // `max_output_bytes` must still bound the whole result.
        let tight = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "for i in $(seq 1 500); do echo padding-line-$i; done; sleep 30 &"}),
            OutputCaps { command_bytes: 1_000 },
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(
            tight.output.contains("were terminated when it returned"),
            "{}",
            tight.output
        );
        // `render_process_output` adds its own truncation notice outside
        // max_bytes by design, so the cap was never an exact total; what this
        // pins is that OUR note is carved out of the budget rather than added
        // on top of it. Without the reservation this lands near 1_600.
        assert!(
            tight.output.len() <= 1_300,
            "note must be reserved from the cap, not appended past it: {} bytes",
            tight.output.len()
        );

        // A command that leaves nothing behind says nothing, or the note
        // becomes noise on every call and stops being read.
        let plain = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "echo plain"}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(plain.ok, "{}", plain.output);
        assert!(
            !plain.output.contains("were terminated when it returned"),
            "no background children, no note: {}",
            plain.output
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn bash_failure_preserves_tail_of_output() {
        let root = temp_project();
        // 40k+ bytes of output with the failure marker at the very end.
        let cmd = "for i in $(seq 1 2000); do echo \"noise line $i padded out a bit\"; done; echo THE_REAL_FAILURE; exit 3";
        let out = bash_tool(
            &root.join("data"),
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

    /// A root-absolute path names itself. Stripping its leading `/` joined it
    /// to the root a second time, so under root `/app` the path `/app/x`
    /// became `/app/app/x` while the call still reported success: bytes landed
    /// where the model never named them, and reading back the path it did name
    /// said the file was missing.
    #[test]
    fn a_root_absolute_path_resolves_to_itself() {
        let root = std::env::temp_dir().join(format!("openmax-abs-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();

        let abs = root.join("eigen.py");
        let abs_str = abs.to_str().unwrap().to_string();

        let wrote = write_file(&root, &json!({"path": abs_str, "content": "print(1)\n"}));
        assert!(wrote.ok, "{}", wrote.output);
        assert!(abs.exists(), "write must land at the path the model named: {abs_str}");
        // The re-rooted twin is what the old code produced; it must not exist.
        let re_rooted = root.join(abs_str.trim_start_matches('/'));
        assert!(!re_rooted.exists(), "path was re-rooted to {}", re_rooted.display());

        let read = read_file(&root, &json!({"path": abs_str}));
        assert!(read.ok, "{}", read.output);
        assert!(read.output.contains("print(1)"), "{}", read.output);

        // An absolute path outside the root is still refused, and refused as an
        // escape rather than quietly rewritten into the project.
        let outside = read_file(&root, &json!({"path": "/etc/hosts"}));
        assert!(!outside.ok, "reading outside the root must fail: {}", outside.output);
        assert!(
            outside.output.contains("escapes the project root"),
            "{}",
            outside.output
        );

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

        // The continuation must resume at the first omitted line: an
        // off-by-one silently skips one line per capped read.
        let hint: usize = out.output.split("continue with offset=").nth(1).unwrap()
            .chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap();
        let shown = out.output.lines().filter(|l| l.contains("xxxx")).count();
        assert_eq!(hint, shown + 1, "hint must name the first omitted line: {}", out.output);
        let next = read_file(&root, &json!({"path": "big.txt", "offset": hint}));
        assert!(next.ok, "{}", next.output);
        assert!(
            next.output.trim_start().starts_with(&format!("{hint} ")),
            "continuation starts at the omitted line: {}",
            next.output
        );

        // When the cap lands on the final line, the hint must still name a
        // readable line rather than pointing past EOF.
        let mut exact = String::new();
        for _ in 0..hint {
            exact.push_str(&long_line);
            exact.push('\n');
        }
        std::fs::write(root.join("exact.txt"), &exact).unwrap();
        let out = read_file(&root, &json!({"path": "exact.txt"}));
        let hint2: usize = out.output.split("continue with offset=").nth(1)
            .expect("a file one line past the cap still gets a continuation")
            .chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap();
        assert_eq!(hint2, hint, "the final line is the first omitted one: {}", out.output);
        let last = read_file(&root, &json!({"path": "exact.txt", "offset": hint2}));
        assert!(last.ok, "the hint must be followable: {}", last.output);
        let _ = std::fs::remove_dir_all(root);
    }

    /// A clipped line must say so. The silent version sends the model into
    /// edit_file with an old_string that can never match: the read shows a
    /// 500-byte prefix, the edit fails, and the closest-match hint points at
    /// the very line the model just read.
    #[test]
    fn read_file_marks_clipped_long_lines() {
        let root = std::env::temp_dir().join(format!("openmax-read-clip-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let long = format!("prefix {}suffix-END", "x".repeat(600));
        std::fs::write(root.join("long.txt"), format!("{long}\nshort\n")).unwrap();
        let out = read_file(&root, &json!({"path": "long.txt"}));
        assert!(out.ok, "{}", out.output);
        let first = out.output.lines().next().unwrap();
        assert!(first.contains("[line clipped;"), "{first}");
        assert!(first.contains("more bytes]"), "{first}");
        assert!(!first.contains("suffix-END"), "the tail is dropped, not hidden: {first}");
        assert!(out.output.contains("    2 short"), "ordinary lines stay unmarked: {}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    /// An offset past the end must not read like an empty file: the model
    /// would conclude the content is gone rather than that its offset is
    /// stale.
    #[test]
    fn read_file_offset_past_eof_is_an_error_not_an_empty_file() {
        let root = std::env::temp_dir().join(format!("openmax-read-eof-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        std::fs::write(root.join("small.txt"), "one\ntwo\nthree\n").unwrap();
        let out = read_file(&root, &json!({"path": "small.txt", "offset": 50}));
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("past the end"), "{}", out.output);
        assert!(out.output.contains("3 lines"), "{}", out.output);
        // The last line is still reachable, and a truly empty file keeps its
        // own message.
        let out = read_file(&root, &json!({"path": "small.txt", "offset": 3}));
        assert!(out.ok && out.output.contains("three"), "{}", out.output);
        std::fs::write(root.join("empty.txt"), "").unwrap();
        let out = read_file(&root, &json!({"path": "empty.txt", "offset": 5}));
        assert!(out.ok && out.output.contains("(empty file)"), "{}", out.output);
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
        let out = execute("glob", &json!({"pattern": "**/*.rs"}), &root.join("data"), &root, OutputCaps::default(), cancel).await;
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("cancelled"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }
}
