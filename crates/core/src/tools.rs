use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};
use similar::{ChangeTag, TextDiff};

use crate::client::truncate;

const MAX_RESULTS: usize = 200;
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

/// Names of every tool the harness exposes, for the fallback call parser.
pub fn tool_names() -> Vec<String> {
    ["list_dir", "read_file", "write_file", "edit_file", "glob", "grep", "bash"]
        .iter()
        .map(|s| s.to_string())
        .collect()
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
pub fn tool_schemas() -> Value {
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
        other => ToolOutcome::err(format!("unknown tool: {other}")),
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

fn glob_tool(root: &Path, args: &Value) -> ToolOutcome {
    let Some(pattern) = args["pattern"].as_str() else {
        return ToolOutcome::err("missing required argument: pattern");
    };
    let matcher = match globset::GlobBuilder::new(pattern).literal_separator(false).build() {
        Ok(g) => g.compile_matcher(),
        Err(e) => return ToolOutcome::err(format!("invalid glob: {e}")),
    };
    let mut hits: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in project_walk(root).flatten() {
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
    let mut out = String::new();
    let mut count = 0usize;
    'files: for entry in project_walk(&search_root).flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(m) = &file_matcher {
            let name_match = path.file_name().map(|n| m.is_match(n.as_ref() as &Path)).unwrap_or(false);
            if !name_match && !m.is_match(rel_display(root, path)) {
                continue;
            }
        }
        if entry.metadata().map(|m| m.len() > MAX_FILE_BYTES).unwrap_or(true) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else { continue };
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) {
                let rel = rel_display(root, path);
                out.push_str(&format!("{rel}:{}: {}\n", i + 1, truncate(line.trim(), 300)));
                count += 1;
                if count >= MAX_RESULTS {
                    out.push_str("… result limit reached; refine the pattern\n");
                    break 'files;
                }
            }
        }
    }
    if count == 0 {
        return ToolOutcome::ok("no matches".into());
    }
    ToolOutcome::ok(out)
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
                let cut = floor_char(&text, MAX_OUTPUT_BYTES);
                text = format!("{}\n… output truncated", &text[..cut]);
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
