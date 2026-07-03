use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

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
const MAX_LINE_CHARS: usize = 500;
const MAX_FILE_BYTES: u64 = 1_500_000;

#[derive(Clone, serde::Serialize)]
pub struct DiffInfo {
    pub path: String,
    pub diff: String,
    pub added: usize,
    pub removed: usize,
}

pub struct ToolOutcome {
    pub ok: bool,
    pub output: String,
    pub diff: Option<DiffInfo>,
}

impl ToolOutcome {
    fn ok(output: String) -> Self {
        Self { ok: true, output, diff: None }
    }
    fn err(output: impl Into<String>) -> Self {
        Self { ok: false, output: output.into(), diff: None }
    }
}

/// True for tools that can change state and therefore go through approval.
pub fn is_mutating(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file" | "bash")
}

/// Every tool name exposed by the harness.
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
                "description": "List files and directories at a path inside the project. Use path \".\" for the project root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Directory path relative to the project root" }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file. Returns numbered lines. Large files are paginated via offset/limit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to the project root" },
                        "offset": { "type": "integer", "description": "1-based line to start from (optional)" },
                        "limit": { "type": "integer", "description": "Max lines to return (optional)" }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file with the given content. Parent directories are created automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to the project root" },
                        "content": { "type": "string", "description": "Full new file content" }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace an exact string in a file. old_string must match the file exactly (including whitespace) and must be unique unless replace_all is true. Read the file first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to the project root" },
                        "old_string": { "type": "string", "description": "Exact text to replace" },
                        "new_string": { "type": "string", "description": "Replacement text" },
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
                "description": "Find files by glob pattern, e.g. \"**/*.ts\" or \"src/**/test_*.py\". Returns paths sorted by modification time, newest first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Glob pattern matched against paths relative to the project root" }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search file contents with a regular expression. Returns matching lines as path:line: text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Rust-flavored regex" },
                        "path": { "type": "string", "description": "Directory to search, relative to project root (optional, default \".\")" },
                        "glob": { "type": "string", "description": "Only search files matching this glob, e.g. \"*.rs\" (optional)" }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the project root and return its output. Use for builds, tests, git, and anything the other tools can't do.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "The shell command to run" },
                        "timeout_secs": { "type": "integer", "description": "Timeout in seconds (default 60, max 300)" }
                    },
                    "required": ["command"]
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

pub async fn execute(name: &str, args: &Value, root: &Path) -> ToolOutcome {
    match name {
        "list_dir" => list_dir(root, args),
        "read_file" => read_file(root, args),
        "write_file" => write_file(root, args),
        "edit_file" => edit_file(root, args),
        "glob" => glob_tool(root, args),
        "grep" => grep_tool(root, args),
        "bash" => bash_tool(root, args).await,
        other => ToolOutcome::err(format!(
            "unknown tool: {other}; the available tools are {}",
            TOOL_NAMES.join(", ")
        )),
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
    ToolOutcome::ok(dirs.join("\n"))
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
    for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
        let line = if line.len() > MAX_LINE_CHARS { &line[..floor_char(line, MAX_LINE_CHARS)] } else { line };
        out.push_str(&format!("{:>5} {}\n", i + 1, line));
    }
    if total > offset - 1 + limit {
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
    let count = old.matches(old_string).count();
    if count == 0 {
        return ToolOutcome::err("old_string not found in file. Read the file and retry with the exact text, including whitespace.");
    }
    if count > 1 && !replace_all {
        return ToolOutcome::err(format!(
            "old_string matches {count} times; provide a longer unique string or set replace_all to true"
        ));
    }
    let new = if replace_all {
        old.replace(old_string, new_string)
    } else {
        old.replacen(old_string, new_string, 1)
    };
    if let Err(e) = std::fs::write(&path, &new) {
        return ToolOutcome::err(format!("cannot write {rel}: {e}"));
    }
    let diff = make_diff(root, &path, &old, &new);
    let summary = format!("edited {} (+{} −{})", diff.path, diff.added, diff.removed);
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
fn truncate_command_output(text: &str) -> String {
    let spilled = spill_command_output(text);
    let mut start = text.len() - MAX_OUTPUT_BYTES;
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

async fn bash_tool(root: &Path, args: &Value) -> ToolOutcome {
    let Some(command) = args["command"].as_str() else {
        return ToolOutcome::err("missing required argument: command");
    };
    let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(60).clamp(1, 300);
    let mut cmd = tokio::process::Command::new("/bin/zsh");
    cmd.arg("-lc")
        .arg(command)
        .current_dir(root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let fut = async {
        let child = cmd.spawn().map_err(|e| format!("failed to spawn shell: {e}"))?;
        child.wait_with_output().await.map_err(|e| format!("command failed: {e}"))
    };
    match tokio::time::timeout(Duration::from_secs(timeout_secs), fut).await {
        Err(_) => ToolOutcome::err(format!("command timed out after {timeout_secs}s")),
        Ok(Err(e)) => ToolOutcome::err(e),
        Ok(Ok(output)) => {
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
            if text.len() > MAX_OUTPUT_BYTES {
                text = truncate_command_output(&text);
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
        let kept = truncate_command_output(&text);
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
        let out = bash_tool(&root, &json!({"command": cmd})).await;
        assert!(!out.ok);
        assert!(out.output.starts_with("exit code 3"), "{}", &out.output[..60]);
        assert!(out.output.contains("THE_REAL_FAILURE"), "tail must survive truncation");
        assert!(!out.output.contains("noise line 1 "), "head should be dropped");
        let _ = std::fs::remove_dir_all(root);
    }
}
