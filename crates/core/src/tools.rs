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
    /// A result that carries none of the process output, because the call was
    /// killed before it could be rendered. What the process managed to print
    /// still happened, and the result dropped all of it.
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
pub const TOOL_NAMES: &[&str] = &[
    "list_dir", "read_file", "write_file", "edit_file", "glob", "grep", "bash", "web_search",
];

pub fn tool_names() -> Vec<String> {
    TOOL_NAMES.iter().map(|s| s.to_string()).collect()
}

/// One-line human summary of a call, shown in approval prompts and tool cards.
pub fn summarize_call(name: &str, args: &Value) -> String {
    match name {
        "bash" => args["command"].as_str().unwrap_or("?").to_string(),
        "web_search" => args["query"].as_str().unwrap_or("?").to_string(),
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
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web (Firecrawl); returns titles, urls, snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "integer", "description": "1-10, default 5" }
                    },
                    "required": ["query"]
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
    if name == "bash" {
        return bash_tool(root, args, caps, cancel).await;
    }
    if name == "web_search" {
        return web_search_tool(args, cancel).await;
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

/// Where `web_search` sends queries: a self-hosted Firecrawl when
/// FIRECRAWL_API_URL is set, otherwise the keyless cloud endpoint. Public so
/// /status can disclose the exact network destination.
pub fn web_search_base() -> String {
    std::env::var("FIRECRAWL_API_URL")
        .ok()
        .map(|u| u.trim().trim_end_matches('/').to_string())
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| "https://api.firecrawl.dev".to_string())
}

/// True when a FIRECRAWL_API_KEY is set; /status reports keyless vs keyed.
pub fn web_search_has_key() -> bool {
    std::env::var("FIRECRAWL_API_KEY").map(|k| !k.trim().is_empty()).unwrap_or(false)
}

/// Search the public web via Firecrawl's search API.
///
/// Zero setup by design: the cloud endpoint answers without credentials
/// (Firecrawl Keyless), so a fresh install can ground itself immediately.
/// Two plain env vars are the entire upgrade surface: FIRECRAWL_API_KEY for
/// higher limits, FIRECRAWL_API_URL for a self-hosted instance. Read-only,
/// so it runs without an approval prompt like grep; permissions rules can
/// still deny or gate it by name, and /status discloses the destination.
/// The request carries the query and nothing else.
async fn web_search_tool(args: &Value, cancel: Arc<CancelToken>) -> ToolOutcome {
    let Some(query) = args["query"].as_str().map(str::trim).filter(|q| !q.is_empty()) else {
        return ToolOutcome::err("web_search needs a non-empty query");
    };
    let limit = args["limit"].as_u64().unwrap_or(5).clamp(1, 10);

    // One pooled client for the process lifetime, same reason as ChatClient:
    // rebuilding the pool would redo the TLS handshake on every search.
    static HTTP: OnceLock<reqwest::Client> = OnceLock::new();
    let http = HTTP.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("failed to build http client")
    });

    let mut request = http
        .post(format!("{}/v2/search", web_search_base()))
        .json(&json!({ "query": query, "limit": limit }));
    if web_search_has_key() {
        if let Ok(key) = std::env::var("FIRECRAWL_API_KEY") {
            request = request.bearer_auth(key.trim());
        }
    }

    // One cancellation race over the WHOLE exchange: headers can arrive
    // quickly while the body trickles, and Esc must win over the body read
    // too, not just over connect and send.
    let exchange = async {
        let response = request
            .send()
            .await
            .map_err(|e| format!("web search failed: {e}"))?;
        let status = response.status();
        if status.as_u16() == 429 {
            return Err(
                "web search rate-limited (keyless tier); retry later or set FIRECRAWL_API_KEY for higher limits"
                    .to_string(),
            );
        }
        let body: Value = response
            .json()
            .await
            .map_err(|e| format!("web search returned unreadable JSON (HTTP {status}): {e}"))?;
        if !status.is_success() || body["success"] != Value::Bool(true) {
            let detail = body["error"].as_str().unwrap_or("unknown error");
            return Err(format!("web search failed (HTTP {status}): {detail}"));
        }
        Ok(body)
    };
    let body = tokio::select! {
        _ = cancel.cancelled() => return ToolOutcome::err("tool cancelled by user"),
        r = exchange => match r {
            Ok(body) => body,
            Err(e) => return ToolOutcome::err(e),
        },
    };
    match format_web_results(&body["data"]["web"]) {
        Ok(text) => ToolOutcome::ok(text),
        Err(e) => ToolOutcome::err(e),
    }
}

/// Compact plain-text rendering of search results: rank, title, url, and a
/// whitespace-collapsed snippet. Plain lines instead of the provider's JSON
/// because every byte here is re-prefilled on each later turn.
fn format_web_results(web: &Value) -> Result<String, String> {
    let Some(results) = web.as_array() else {
        return Err("web search response had no results array".into());
    };
    if results.is_empty() {
        return Ok("no results".into());
    }
    let mut out = String::new();
    // Every field here is provider-controlled bytes headed for re-prefilled
    // history: cap all of them, and never render more entries than the tool
    // can be asked for, whatever the response claims.
    for (i, r) in results.iter().take(10).enumerate() {
        let title = collapse_snippet(r["title"].as_str().unwrap_or("(untitled)"), 120);
        let url = cap_chars(r["url"].as_str().unwrap_or("").trim(), 300);
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!("{}. {title}\n   {url}\n", i + 1));
        let snippet = collapse_snippet(r["description"].as_str().unwrap_or(""), 240);
        if !snippet.is_empty() {
            out.push_str(&format!("   {snippet}\n"));
        }
    }
    Ok(out)
}

/// Char-boundary cap without whitespace collapsing (URLs have none).
fn cap_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{cut}…")
}

/// Collapse runs of whitespace to single spaces and cap the length on a char
/// boundary, so a markdown-y description becomes one honest line.
fn collapse_snippet(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let cut: String = collapsed.chars().take(max_chars).collect();
    format!("{cut}…")
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
    let request = ProcessRequest {
        program: shell.into(),
        args: vec!["-lc".into(), command.into()],
        cwd: root.to_path_buf(),
        stdin: StdinMode::Null,
        timeout: std::time::Duration::from_secs(timeout_secs),
        capture: CaptureSpec {
            head_bytes: 0,
            tail_bytes: caps.command_bytes,
            spill_dir: Some(crate::state::default_data_dir().join("cmd-logs")),
            spill_bytes_per_stream: 16 * 1024 * 1024,
        },
    };
    match execution::run_process(request, cancel).await {
        Err(ProcessError::Spawn(e)) => ToolOutcome::err(format!("failed to spawn shell: {e}")),
        Err(ProcessError::Wait(e)) => ToolOutcome::err(format!("command failed: {e}")),
        Ok(output) => match &output.termination {
            Termination::Cancelled => {
                ToolOutcome::from_killed_process("command cancelled by user", &output)
            }
            Termination::TimedOut => ToolOutcome::from_killed_process(
                format!("command timed out after {timeout_secs}s"),
                &output,
            ),
            Termination::Exited(status) => {
                let (text, truncated) = render_process_output(&output, caps.command_bytes);
                let (ok, text) = match status.success() {
                    true => (true, text),
                    false => {
                        let code = status.code().unwrap_or(-1);
                        (false, format!("exit code {code}\n{text}"))
                    }
                };
                ToolOutcome::from_process(ok, text, &output, truncated)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn web_search_is_read_only_and_named() {
        assert!(!is_mutating("web_search"));
        assert!(TOOL_NAMES.contains(&"web_search"));
        assert_eq!(
            summarize_call("web_search", &json!({"query": "ratatui rendering"})),
            "ratatui rendering"
        );
    }

    #[test]
    fn web_results_format_compactly_from_the_recorded_shape() {
        // Shape recorded from a live keyless POST to /v2/search.
        let web = json!([
            {
                "url": "https://ratatui.rs/concepts/rendering/",
                "title": "Rendering",
                "description": "# Rendering\nemploys the immediate mode rendering approach for TUI development.\n\n## What is",
                "position": 1
            },
            {"url": "https://example.com/two", "title": "Two", "position": 2}
        ]);
        let text = format_web_results(&web).unwrap();
        assert_eq!(
            text,
            "1. Rendering\n   https://ratatui.rs/concepts/rendering/\n   # Rendering employs the immediate mode rendering approach for TUI development. ## What is\n\n2. Two\n   https://example.com/two\n"
        );
        assert_eq!(format_web_results(&json!([])).unwrap(), "no results");
        assert!(format_web_results(&json!(null)).is_err());
    }

    #[test]
    fn snippets_collapse_and_cap_on_char_boundaries() {
        assert_eq!(collapse_snippet("a\n\n  b\tc", 100), "a b c");
        assert_eq!(collapse_snippet("héllo wörld", 5), "héllo…");
        assert_eq!(collapse_snippet("", 10), "");
    }

    #[test]
    fn hostile_result_metadata_is_bounded() {
        // Provider-controlled bytes: a degenerate response with giant
        // fields and more entries than the tool can be asked for must
        // still render bounded.
        let giant = "x".repeat(50_000);
        let entry = json!({
            "url": format!("https://e.example/{giant}"),
            "title": giant,
            "description": giant,
            "position": 1
        });
        let many: Vec<_> = (0..50).map(|_| entry.clone()).collect();
        let text = format_web_results(&json!(many)).unwrap();
        assert!(text.len() < 10_000, "unbounded output: {} bytes", text.len());
        assert!(!text.contains("11. "), "more entries than the request cap");
        assert!(text.contains("10. "));
    }

    /// One canned Firecrawl endpoint serving a single response; the receiver
    /// yields the captured raw request (head plus body) so auth headers and
    /// payload can be asserted from the wire, not from intent.
    fn canned_firecrawl(
        status: &'static str,
        body: &'static str,
    ) -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let mut read = 0usize;
            // Read until the JSON body closes; requests here are small.
            loop {
                let n = stream.read(&mut buf[read..]).unwrap_or(0);
                if n == 0 {
                    break;
                }
                read += n;
                let text = String::from_utf8_lossy(&buf[..read]);
                if let Some(body_at) = text.find("\r\n\r\n") {
                    let header = &text[..body_at];
                    let length = header
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length: "))
                        .or_else(|| {
                            header.lines().find_map(|l| l.strip_prefix("Content-Length: "))
                        })
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if read >= body_at + 4 + length {
                        break;
                    }
                }
            }
            let _ = tx.send(String::from_utf8_lossy(&buf[..read]).to_string());
            let reply = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(reply.as_bytes());
        });
        (base, rx)
    }

    #[tokio::test]
    async fn web_search_speaks_keyless_then_keyed_and_reports_limits() {
        // Env is process-global: this one test owns both FIRECRAWL vars and
        // restores them before returning, so no sibling can race it.
        let saved_url = std::env::var("FIRECRAWL_API_URL").ok();
        let saved_key = std::env::var("FIRECRAWL_API_KEY").ok();
        let root = temp_project();
        let ok_body = r#"{"success":true,"data":{"web":[{"url":"https://a.example","title":"A","description":"alpha  beta","position":1}]}}"#;

        // Keyless: no Authorization header leaves the process.
        let (base, rx) = canned_firecrawl("200 OK", ok_body);
        std::env::set_var("FIRECRAWL_API_URL", &base);
        std::env::remove_var("FIRECRAWL_API_KEY");
        let out = execute(
            "web_search",
            &json!({"query": "q", "limit": 3}),
            &root,
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(out.ok, "{}", out.output);
        assert_eq!(out.output, "1. A\n   https://a.example\n   alpha beta\n");
        let request = rx.recv().unwrap();
        assert!(!request.to_lowercase().contains("authorization:"), "{request}");
        assert!(request.contains("\"query\":\"q\""));
        assert!(request.contains("\"limit\":3"));

        // Keyed: the same call carries the bearer token.
        let (base, rx) = canned_firecrawl("200 OK", ok_body);
        std::env::set_var("FIRECRAWL_API_URL", &base);
        std::env::set_var("FIRECRAWL_API_KEY", "fc-test-key");
        let out = execute(
            "web_search",
            &json!({"query": "q"}),
            &root,
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(out.ok, "{}", out.output);
        let request = rx.recv().unwrap();
        assert!(
            request.contains("authorization: Bearer fc-test-key")
                || request.contains("Authorization: Bearer fc-test-key"),
            "{request}"
        );

        // Rate limit: the error names the keyless tier and the upgrade path.
        let (base, _rx) = canned_firecrawl("429 Too Many Requests", "{}");
        std::env::set_var("FIRECRAWL_API_URL", &base);
        std::env::remove_var("FIRECRAWL_API_KEY");
        let out = execute(
            "web_search",
            &json!({"query": "q"}),
            &root,
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(!out.ok);
        assert!(out.output.contains("FIRECRAWL_API_KEY"), "{}", out.output);

        // Provider-reported failure is surfaced verbatim, never invented.
        let err_body = r#"{"success":false,"error":"query too long"}"#;
        let (base, _rx) = canned_firecrawl("200 OK", err_body);
        std::env::set_var("FIRECRAWL_API_URL", &base);
        let out = execute(
            "web_search",
            &json!({"query": "q"}),
            &root,
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(!out.ok);
        assert!(out.output.contains("query too long"), "{}", out.output);

        match saved_url {
            Some(v) => std::env::set_var("FIRECRAWL_API_URL", v),
            None => std::env::remove_var("FIRECRAWL_API_URL"),
        }
        match saved_key {
            Some(v) => std::env::set_var("FIRECRAWL_API_KEY", v),
            None => std::env::remove_var("FIRECRAWL_API_KEY"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

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

    /// A command killed for running too long still printed something, and a
    /// hook that is told a process ran no bytes would draw the wrong
    /// conclusion about the turn.
    ///
    /// The timeout has to outlast shell startup by a wide margin: on a loaded
    /// CI runner a one-second budget can expire before `echo` ever runs, and
    /// then the harness correctly reports the zero bytes that were printed.
    #[tokio::test]
    async fn a_timed_out_command_still_reports_what_it_printed() {
        let root = temp_project();
        let out = bash_tool(
            &root,
            &json!({"command": "echo before-the-timeout; sleep 30", "timeout_secs": 5}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;

        assert!(!out.ok);
        assert!(out.output.contains("timed out"), "{}", out.output);
        assert_eq!(
            out.process_bytes,
            Some("before-the-timeout\n".len() as u64),
            "the bytes it managed to print still happened"
        );
        assert!(out.process_truncated, "and the result carries none of them");
        let _ = std::fs::remove_dir_all(root);
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
