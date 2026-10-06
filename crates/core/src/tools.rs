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
use std::time::{Duration, Instant};

use crate::execution::{
    self, CaptureSpec, ProcessError, ProcessOutput, ProcessRequest, StdinMode, Termination,
};
use crate::state::CancelToken;

use ignore::gitignore::Gitignore;
use serde_json::{json, Value};
use similar::{ChangeTag, DiffTag, TextDiff};

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
    let (canon, names_dir) = canonical_target(joined.clone())?;
    let root_canon = root.canonicalize().map_err(|e| format!("cannot resolve project root: {e}"))?;
    if !canon.starts_with(&root_canon) {
        return Err(format!("path escapes the project root: {rel}"));
    }
    // The OS refuses `notes.txt/`, or a link whose target is `notes.txt/`,
    // because the separator demands a directory. The resolved path has lost
    // that separator (and canonicalization ignores it on some platforms), so
    // without this a write would reach `notes.txt` itself. The OS's own
    // lookup decides for an entry that exists; the flag covers one that a
    // write would create.
    let os_refuses = std::fs::metadata(&joined).is_err_and(|e| e.kind() == std::io::ErrorKind::NotADirectory);
    if os_refuses || (names_dir && !canon.is_dir()) {
        return Err(format!("not a directory: {rel}"));
    }
    Ok(canon)
}

/// Where a file operation on `path` lands, for paths that may not exist yet
/// (write_file targets): the deepest existing ancestor canonicalized, plus the
/// missing tail, so traversal via `..` is caught.
///
/// An ancestor counts as existing when its directory holds the entry, even as
/// a dangling symlink. `Path::exists` follows links and is false for a
/// dangling one, so its name used to join the unresolved tail and pass the
/// root check, and the write then followed it and created the target outside
/// the project. A dangling link is resolved to its target instead, which is
/// where anything created through it lands.
///
/// A trailing separator or `.` makes lstat follow the final link, so a link
/// probed as `link/` looked missing and its bare name joined the tail. Each
/// hop probes the lexically normalized path, which also covers link targets
/// that end in a separator. Normalizing drops the separator's demand for a
/// directory, so the returned flag reports whether any hop made it.
fn canonical_target(mut path: PathBuf) -> Result<(PathBuf, bool), String> {
    // Bounds a chain or loop of dangling links, as the OS bounds link hops.
    const MAX_LINK_HOPS: usize = 40;
    let mut names_dir = false;
    for _ in 0..MAX_LINK_HOPS {
        // Every hop's final component is the operation's final entry, so a
        // separator on any of them binds the entry the path ends at.
        names_dir |= demands_directory(&path);
        let mut probe: PathBuf = path.components().collect();
        let mut tail = Vec::new();
        while probe.symlink_metadata().is_err() {
            match (probe.file_name(), probe.parent()) {
                (Some(name), Some(parent)) => {
                    tail.push(name.to_os_string());
                    probe = parent.to_path_buf();
                }
                _ => return Err("invalid path".into()),
            }
        }
        let mut canon = match probe.canonicalize() {
            Ok(canon) => canon,
            Err(_) if probe.is_symlink() => {
                let target = std::fs::read_link(&probe).map_err(|e| format!("cannot resolve path: {e}"))?;
                path = probe.parent().map(|dir| dir.join(&target)).unwrap_or(target);
                path.extend(tail.iter().rev());
                continue;
            }
            Err(e) => return Err(format!("cannot resolve path: {e}")),
        };
        canon.extend(tail.iter().rev());
        return Ok((canon, names_dir));
    }
    Err("cannot resolve path: too many levels of symbolic links".into())
}

/// Whether `path` ends in a separator or a `.` component, either of which
/// makes the OS require the entry it names to be a directory.
fn demands_directory(path: &Path) -> bool {
    let bytes = path.as_os_str().as_encoded_bytes();
    let bytes = bytes.strip_suffix(b".").unwrap_or(bytes);
    bytes.last().is_some_and(|&b| std::path::is_separator(char::from(b)))
}

fn rel_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string())
}

/// The argument `key` read by `native`, or parsed from a string. Models often
/// write a number or a boolean as a string (`"5"`, `"true"`), and reading the
/// JSON type alone dropped it without a word and ran the default instead.
fn arg<T: std::str::FromStr>(args: &Value, key: &str, native: fn(&Value) -> Option<T>) -> Option<T> {
    let value = &args[key];
    native(value).or_else(|| value.as_str()?.trim().to_ascii_lowercase().parse().ok())
}

/// The file a read, write or edit names. A blank path names none: resolved,
/// it reached the project root, and the error that followed named nothing.
fn file_path_arg(args: &Value) -> Option<&str> {
    args["path"].as_str().filter(|p| !p.trim().is_empty())
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
    // Cancellation may stop waiting for reads. Started mutations must settle
    // before the turn releases ownership of the files they can still change.
    let mutating = is_mutating(name);
    let name = name.to_string();
    let args = args.clone();
    let root = root.to_path_buf();
    let task_cancel = cancel.clone();
    let task = tokio::task::spawn_blocking(move || match name.as_str() {
            "list_dir" => list_dir(&root, &args),
            "read_file" => read_file(&root, &args, &task_cancel),
            "write_file" => write_file(&root, &args),
            "edit_file" => edit_file(&root, &args),
            "glob" => glob_tool(&root, &args),
            "grep" => grep_tool(&root, &args),
            other => ToolOutcome::err(format!(
                "unknown tool: {other}; the available tools are {}",
                TOOL_NAMES.join(", ")
            )),
        });
    finish_file_task(task, mutating, cancel).await
}

async fn finish_file_task(
    task: tokio::task::JoinHandle<ToolOutcome>,
    mutating: bool,
    cancel: Arc<CancelToken>,
) -> ToolOutcome {
    if mutating {
        return task.await.unwrap_or_else(|e| ToolOutcome::err(format!("tool execution failed: {e}")));
    }
    tokio::select! {
        _ = cancel.cancelled() => ToolOutcome::err("tool cancelled by user"),
        result = task => result.unwrap_or_else(|e| ToolOutcome::err(format!("tool execution failed: {e}"))),
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

fn read_file(root: &Path, args: &Value, cancel: &CancelToken) -> ToolOutcome {
    let Some(rel) = file_path_arg(args) else {
        return ToolOutcome::err("missing required argument: path");
    };
    let path = match resolve(root, rel) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    let offset = arg(args, "offset", Value::as_u64);
    let limit = arg(args, "limit", Value::as_u64);
    let asked_for_window = offset.is_some() || limit.is_some();
    let mut window = ReadWindow {
        offset: offset.unwrap_or(1).max(1) as usize,
        limit: limit.unwrap_or(MAX_READ_LINES as u64).min(MAX_READ_LINES as u64) as usize,
        out: String::new(),
        cut_at: None,
    };
    let read_error = |e: std::io::Error| match e.kind() {
        std::io::ErrorKind::InvalidData => ToolOutcome::err(format!("{rel} is not a UTF-8 text file")),
        _ => ToolOutcome::err(format!("cannot read {rel}: {e}")),
    };
    let total = match std::fs::metadata(&path) {
        // A directory fails to read like any unreadable path; calling it a
        // file that is not UTF-8 sends the model after an encoding problem.
        Ok(m) if m.is_dir() => {
            return ToolOutcome::err(format!("{rel} is a directory; use list_dir to see its entries"))
        }
        // The cap keeps a whole-file read of a huge file out of memory. A
        // window of one streams instead: this refusal tells the model to
        // ask for one, and refusing that too leaves it nothing to retry. It
        // names nothing else, because grep skips a file this large.
        Ok(m) if m.len() > MAX_FILE_BYTES && !asked_for_window => {
            return ToolOutcome::err(format!(
                "file too large ({} bytes); read a range of lines with offset and limit",
                m.len()
            ))
        }
        Ok(m) if m.len() > MAX_FILE_BYTES => match stream_window(&path, &mut window, cancel) {
            Ok(total) => total,
            Err(_) if cancel.is_cancelled() => return ToolOutcome::err("tool cancelled by user"),
            Err(e) => return read_error(e),
        },
        Ok(_) => {
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => return read_error(e),
            };
            for (i, line) in text.lines().enumerate().skip(window.offset - 1).take(window.limit) {
                if !window.push(i + 1, line, line.len()) {
                    break;
                }
            }
            text.lines().count()
        }
        Err(e) => return ToolOutcome::err(format!("cannot read {rel}: {e}")),
    };
    window.finish(rel, total)
}

/// The numbered lines one read_file call returns.
struct ReadWindow {
    /// First line to show, 1-based.
    offset: usize,
    /// Most lines to show.
    limit: usize,
    out: String,
    /// The first line that did not fit under `MAX_READ_BYTES`, if one did not.
    cut_at: Option<usize>,
}

impl ReadWindow {
    /// Whether line `n` (1-based) is still to be shown.
    fn wants(&self, n: usize) -> bool {
        self.cut_at.is_none() && n >= self.offset && n - self.offset < self.limit
    }

    /// Show line `n`, whose full length is `len` bytes; `line` may hold only
    /// the first `MAX_LINE_CHARS` bytes of a longer one. False once the byte
    /// cap is reached and the read should stop.
    fn push(&mut self, n: usize, line: &str, len: usize) -> bool {
        // A clipped line must say so: silently dropping its tail sends the
        // model into edit_file with an old_string that can never match.
        let formatted = if len > MAX_LINE_CHARS {
            let end = floor_char(line, MAX_LINE_CHARS);
            format!("{n:>5} {}… [line clipped; {} more bytes]\n", &line[..end], len - end)
        } else {
            format!("{n:>5} {line}\n")
        };
        if self.out.len() + formatted.len() > MAX_READ_BYTES {
            self.cut_at = Some(n);
            return false;
        }
        self.out.push_str(&formatted);
        true
    }

    /// The result for a file of `total` lines.
    fn finish(self, rel: &str, total: usize) -> ToolOutcome {
        let Self { offset, limit, mut out, cut_at } = self;
        // An offset past the end must not read like an empty file: the model
        // would conclude the content is gone rather than that its offset is
        // stale.
        if offset > total && total > 0 {
            return ToolOutcome::err(format!(
                "offset {offset} is past the end of {rel} ({total} lines); retry with a smaller offset"
            ));
        }
        if let Some(cut_at) = cut_at {
            // `cut_at` is the first line that did NOT fit, so the
            // continuation resumes exactly there. The former `+ 1` skipped
            // one line per capped read, and pointed past EOF when the cap
            // landed on the final line.
            out.push_str(&format!(
                "… output limit reached at line {cut_at} (file has {total} lines; continue with offset={cut_at})\n"
            ));
        } else if total > offset - 1 + limit {
            out.push_str(&format!("… {} more lines (file has {total} lines; continue with offset={})\n", total - (offset - 1 + limit), offset + limit));
        }
        if out.is_empty() {
            out = "(empty file)".into();
        }
        ToolOutcome::ok(out)
    }
}

/// Fill `window` from a file too large to load whole, and return its line
/// count. Lines are read one at a time and only those in the window are
/// decoded, each held to the prefix that `ReadWindow::push` can show, so
/// memory stays bounded however long the file or any of its lines runs.
///
/// The count reads to the end of a file of any size, so the read stops once
/// the call is cancelled: the caller has stopped waiting for it by then.
fn stream_window(path: &Path, window: &mut ReadWindow, cancel: &CancelToken) -> std::io::Result<usize> {
    let file = Cancellable { file: std::fs::File::open(path)?, cancel };
    let mut reader = std::io::BufReader::new(file);
    let mut kept = Vec::new();
    let mut n = 0;
    loop {
        let keep = if window.wants(n + 1) { MAX_LINE_CHARS } else { 0 };
        let Some(len) = next_line(&mut reader, &mut kept, keep)? else {
            return Ok(n);
        };
        n += 1;
        if keep == 0 {
            continue;
        }
        let line = match std::str::from_utf8(&kept) {
            Ok(line) => line,
            // The kept prefix of a clipped line can end inside a character.
            Err(e) if e.error_len().is_none() && len > kept.len() => {
                std::str::from_utf8(&kept[..e.valid_up_to()]).unwrap_or_default()
            }
            Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        };
        window.push(n, line, len);
    }
}

/// A file whose reads fail once the call reading it is cancelled.
struct Cancellable<'a> {
    file: std::fs::File,
    cancel: &'a CancelToken,
}

impl std::io::Read for Cancellable<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(std::io::Error::other("tool cancelled by user"));
        }
        std::io::Read::read(&mut self.file, buf)
    }
}

/// Read the next line into `kept`, keeping at most its first `keep` bytes,
/// and return its full length; None at the end of the file. Lines end as
/// `str::lines` ends them: at `\n`, with one `\r` before it dropped.
fn next_line(reader: &mut impl std::io::BufRead, kept: &mut Vec<u8>, keep: usize) -> std::io::Result<Option<usize>> {
    kept.clear();
    let mut len = 0;
    let mut last = None;
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            return Ok((len > 0).then_some(len));
        }
        let newline = chunk.iter().position(|&b| b == b'\n');
        let body = &chunk[..newline.unwrap_or(chunk.len())];
        kept.extend_from_slice(&body[..body.len().min(keep.saturating_sub(kept.len()))]);
        len += body.len();
        last = body.last().copied().or(last);
        let used = newline.map_or(chunk.len(), |i| i + 1);
        reader.consume(used);
        if newline.is_some() {
            if last == Some(b'\r') {
                len -= 1;
                kept.truncate(len);
            }
            return Ok(Some(len));
        }
    }
}

fn floor_char(s: &str, mut idx: usize) -> usize {
    while !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Longest an edit spends on a closest-line hint or a display diff. Both are
/// quadratic at worst (a minified line, a whole-file rewrite) and run inside
/// a call that cancellation waits for, so each reports what it has by then.
const DIFF_BUDGET: Duration = Duration::from_millis(200);

/// A finished write or edit: the diff for the UI and the summary the model
/// reads. Counts from a diff cut off at its deadline can include unchanged
/// lines, so the summary gives them as an upper bound, or the model takes an
/// estimate for the size of its edit. The caveat follows the "(+N −M)"
/// group rather than going inside it: session replay reads the counts back
/// from that exact group, and a replayed edit card would lose its badge.
fn changed_file(verb: &str, root: &Path, path: &Path, old: &str, new: &str) -> ToolOutcome {
    let (diff, complete) = diff_strings(&rel_display(root, path), old, new);
    let mut summary = format!("{verb} {} (+{} −{})", diff.path, diff.added, diff.removed);
    if !complete {
        summary.push_str(" · counts are an upper bound");
    }
    ToolOutcome { ok: true, output: summary, diff: Some(diff), ..Default::default() }
}

/// Unified diff between two versions of a file, and whether it finished.
fn diff_strings(rel: &str, old: &str, new: &str) -> (DiffInfo, bool) {
    let deadline = Instant::now() + DIFF_BUDGET;
    let text_diff = TextDiff::configure().deadline(deadline).diff_lines(old, new);
    // Past its deadline the diff stops searching and writes each region left
    // as one deletion and one insertion: a valid edit but not the smallest,
    // so it would list unchanged lines as rewritten and its counts can run
    // high.
    let gave_up = Instant::now() > deadline;
    // Counted per op, not per line, so a cut-off diff of a huge rewrite is
    // not walked line by line after its deadline. A deletion has an empty
    // new range and an insertion an empty old range.
    let mut added = 0;
    let mut removed = 0;
    for op in text_diff.ops() {
        if op.tag() != DiffTag::Equal {
            removed += op.old_range().len();
            added += op.new_range().len();
        }
    }
    let diff = if gave_up {
        format!(
            "--- a/{rel}\n+++ b/{rel}\n(diff not shown: the change was too large to compare in time, so +{added} −{removed} may overcount)\n"
        )
    } else {
        text_diff
            .unified_diff()
            .context_radius(3)
            .header(&format!("a/{rel}"), &format!("b/{rel}"))
            .to_string()
    };
    (DiffInfo { path: rel.to_string(), diff: truncate(&diff, 40_000), added, removed }, !gave_up)
}

fn write_file(root: &Path, args: &Value) -> ToolOutcome {
    let Some(rel) = file_path_arg(args) else {
        return ToolOutcome::err("missing required argument: path");
    };
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
    changed_file("wrote", root, &path, &old, content)
}

/// The part of a line the closest-match hint compares: trimmed, and no more
/// of it than read_file shows. A character diff is quadratic in line length,
/// so scoring a whole minified line takes minutes, and the shown prefix is
/// all of a clipped line the model can have copied.
fn hint_text(line: &str) -> &str {
    let line = line.trim();
    &line[..floor_char(line, line.len().min(MAX_LINE_CHARS))]
}

fn line_similarity(a: &str, b: &str) -> f64 {
    let a = hint_text(a);
    let b = hint_text(b);
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

fn byte_counts(text: &str) -> [u32; 256] {
    let mut counts = [0u32; 256];
    for b in text.bytes() {
        counts[usize::from(b)] += 1;
    }
    counts
}

/// An upper bound on `line_similarity` from one pass over the line: two texts
/// cannot share more characters than they share bytes.
fn similarity_ceiling(text: &str, key: &str, key_bytes: &[u32; 256]) -> f64 {
    let mut left = *key_bytes;
    let mut shared = 0usize;
    for b in text.bytes() {
        let count = &mut left[usize::from(b)];
        if *count > 0 {
            *count -= 1;
            shared += 1;
        }
    }
    let total = text.chars().count() + key.chars().count();
    if total == 0 {
        1.0
    } else {
        2.0 * shared as f64 / total as f64
    }
}

fn closest_line_hint(content: &str, old_string: &str) -> String {
    closest_line_hint_until(content, old_string, Instant::now() + DIFF_BUDGET)
}

fn closest_line_hint_until(content: &str, old_string: &str, deadline: Instant) -> String {
    let needle = old_string.lines().next().unwrap_or(old_string);
    let key = hint_text(needle);
    let key_bytes = byte_counts(key);
    let mut best_idx = 0usize;
    let mut best_score = 0.0f64;
    let mut cut_at = None;
    for (i, line) in content.lines().enumerate() {
        // Every line before `i` has been compared, by its diff or by a ceiling
        // that ruled it out, so the hint always names the best of at least
        // one. Waiting for a diff instead never stops a scan in which no
        // line shares a byte with the copied text.
        if i > 0 && Instant::now() > deadline {
            cut_at = Some(i);
            break;
        }
        // A line whose ceiling cannot beat the best score would not replace
        // it, so skipping its diff leaves the result unchanged.
        if similarity_ceiling(hint_text(line), key, &key_bytes) <= best_score {
            continue;
        }
        let score = line_similarity(line, needle);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    let closest = content.lines().nth(best_idx).unwrap_or("");
    // A scan cut short must say so, or the model reads the best of the lines
    // compared as the best of the file.
    let scope = cut_at.map(|n| format!(" in lines 1-{n}")).unwrap_or_default();
    format!(
        "old_string not found. Closest match{scope} is at line {}: '{}'. Read the file around that line and retry with the exact text.",
        best_idx + 1,
        truncate(closest, 120)
    )
}

/// `text` with every bare LF written as CRLF; existing CRLFs are kept.
fn to_crlf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\n', "\r\n")
}

fn edit_file(root: &Path, args: &Value) -> ToolOutcome {
    let Some(rel) = file_path_arg(args) else {
        return ToolOutcome::err("missing required argument: path");
    };
    let (Some(old_string), Some(new_string)) = (args["old_string"].as_str(), args["new_string"].as_str()) else {
        return ToolOutcome::err("missing required arguments: old_string and new_string");
    };
    // An empty old_string matches between every pair of characters, so the
    // ambiguity error would steer the model to replace_all, which splices
    // new_string between every character of the file.
    if old_string.is_empty() {
        return ToolOutcome::err(
            "old_string is empty; give the exact text to replace, or use write_file to create or replace a whole file",
        );
    }
    if old_string == new_string {
        return ToolOutcome::err("old_string and new_string are identical");
    }
    let replace_all = arg(args, "replace_all", Value::as_bool).unwrap_or(false);
    let path = match resolve(root, rel) {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(e),
    };
    let old = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return ToolOutcome::err(format!("cannot read {rel}: {e}")),
    };

    // read_file and grep show lines without their \r, so text copied from
    // either has bare LF breaks and can never match a CRLF file. When every
    // line break in the file is CRLF, read both strings as written with CRLF:
    // the match is otherwise exact, and the edit keeps the file's endings
    // instead of splicing LF lines into it. A mixed file has no single ending
    // to assume, so it stays exact.
    let crlf = old.matches("\r\n").count();
    let lf = old.matches('\n').count();
    let (old_string, new_string) = if crlf > 0 && crlf == lf {
        (to_crlf(old_string), to_crlf(new_string))
    } else {
        (old_string.to_string(), new_string.to_string())
    };
    // Strings that differ only in line endings are the same edit once read
    // as CRLF; writing it would report a change that never reached the file.
    if old_string == new_string {
        return ToolOutcome::err(format!(
            "old_string and new_string differ only in line endings; {rel} uses CRLF throughout and edit_file keeps it, so use write_file to change line endings"
        ));
    }

    let new = if old.contains(&old_string) {
        let count = old.matches(&old_string).count();
        if count > 1 && !replace_all {
            return ToolOutcome::err(format!(
                "old_string matches {count} times; provide a longer unique string or set replace_all to true"
            ));
        }
        if replace_all {
            old.replace(&old_string, &new_string)
        } else {
            old.replacen(&old_string, &new_string, 1)
        }
    } else {
        let mut hint = closest_line_hint(&old, &old_string);
        if crlf > 0 && crlf < lf && old_string.contains('\n') {
            hint.push_str(&format!(
                " Note: {rel} mixes CRLF and LF line endings, which read_file does not show, so a multi-line old_string must match each line ending exactly; edit one line at a time or rewrite the file with write_file."
            ));
        }
        return ToolOutcome::err(hint);
    };

    if let Err(e) = std::fs::write(&path, &new) {
        return ToolOutcome::err(format!("cannot write {rel}: {e}"));
    }
    changed_file("edited", root, &path, &old, &new)
}

/// The walk glob and grep share, from `walk_root` in the project whose
/// canonical root is `root_canon`.
///
/// Hidden files are searchable: the agent's own extension surface lives in
/// dot-directories (`.openmax/tools`, `.agents`, `.github`), and a walker
/// that skips them makes the agent blind to the capabilities it wrote.
/// `.git` alone is excluded by name; gitignore rules still apply.
///
/// The walker reads .gitignore only inside a repository by default, which in
/// a project without one sent glob and grep through the node_modules and
/// build output it ignores. Inside a repository the default stays: it reads
/// them up to the repository's root and no further, so a .gitignore above
/// it, such as a home directory kept in git that ignores `*`, cannot hide
/// the project. Outside one nothing marks where to stop: told to read them
/// anyway, the walker reads every .gitignore up to the filesystem root,
/// ones git never applies there, and the user's global excludes, which git
/// applies only inside a repository. So outside a repository the walker
/// reads .ignore files alone, from every ancestor as it always has, and
/// `ProjectGitignores` applies the project's .gitignore files.
fn project_walker(root_canon: &Path, walk_root: &Path) -> ignore::WalkBuilder {
    // The markers the walker itself checks, on the canonical ancestors.
    let canon = walk_root.canonicalize().unwrap_or_else(|_| walk_root.to_path_buf());
    let in_repository = canon.ancestors().any(|dir| dir.join(".git").exists() || dir.join(".jj").exists());
    let gitignores = (!in_repository).then(|| ProjectGitignores {
        root: root_canon.to_path_buf(),
        walk_root: walk_root.to_path_buf(),
        canon,
        dirs: Default::default(),
    });
    let mut walker = ignore::WalkBuilder::new(walk_root);
    walker
        .hidden(false)
        .filter_entry(move |e| {
            e.file_name() != std::ffi::OsStr::new(".git") && !gitignores.as_ref().is_some_and(|g| g.drops(e))
        })
        .git_ignore(in_repository)
        .git_exclude(in_repository)
        .git_global(in_repository)
        .max_depth(Some(24));
    walker
}

/// The .gitignore files of a project outside a repository, applied to what
/// the walker passes.
///
/// They apply from the project's root down, and rank as the walker ranks the
/// files it reads: the nearest file that names an entry decides, and every
/// .ignore, from any ancestor, outranks every .gitignore. The walker has
/// applied the .ignore files before an entry gets here, so none of them
/// ignores it, but one may whitelist it, and then no .gitignore drops it.
/// Pruning on a .gitignore the walker read itself, before any .ignore above
/// the walk root was consulted, made a scoped glob or grep miss a file the
/// unscoped one found.
struct ProjectGitignores {
    /// The project's canonical root; no .gitignore above it applies.
    root: PathBuf,
    /// The walk root as the walker reports its entries.
    walk_root: PathBuf,
    /// The walk root, canonical.
    canon: PathBuf,
    /// Each directory's rules by canonical path, read once per walk: every
    /// entry asks about every directory above it.
    dirs: std::sync::RwLock<std::collections::HashMap<PathBuf, Arc<DirRules>>>,
}

/// One directory's ignore files, and its parent's.
struct DirRules {
    dir: PathBuf,
    /// Empty above the project's root.
    gitignore: Gitignore,
    /// Read only once a .gitignore drops an entry below it.
    ignore: OnceLock<Gitignore>,
    parent: Option<Arc<DirRules>>,
}

impl ProjectGitignores {
    /// Whether the project's .gitignore files drop `entry`.
    fn drops(&self, entry: &ignore::DirEntry) -> bool {
        // Canonical, like the directories the rules are rooted at, as the
        // walker's own paths already are when it starts from a canonical root.
        let path: std::borrow::Cow<Path> = if self.walk_root == self.canon {
            entry.path().into()
        } else {
            self.canon.join(entry.path().strip_prefix(&self.walk_root).unwrap_or(entry.path())).into()
        };
        let Some(dir) = path.parent() else {
            return false;
        };
        let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
        let rules = self.rules(dir);
        let chain = || std::iter::successors(Some(&*rules), |d| d.parent.as_deref());
        // A .ignore that whitelists the entry outranks every .gitignore.
        let whitelisted = || {
            let verdict = |d: &DirRules| {
                ignore_verdict(d.ignore.get_or_init(|| read_rules(&d.dir, ".ignore")), &path, is_dir)
            };
            chain().find_map(verdict) == Some(false)
        };
        chain().find_map(|d| ignore_verdict(&d.gitignore, &path, is_dir)) == Some(true) && !whitelisted()
    }

    /// The rules in `dir` and in every directory above it.
    fn rules(&self, dir: &Path) -> Arc<DirRules> {
        if let Some(rules) = self.dirs.read().unwrap().get(dir) {
            return rules.clone();
        }
        let rules = Arc::new(DirRules {
            dir: dir.to_path_buf(),
            gitignore: if dir.starts_with(&self.root) { read_rules(dir, ".gitignore") } else { Gitignore::empty() },
            ignore: OnceLock::new(),
            parent: dir.parent().map(|parent| self.rules(parent)),
        });
        self.dirs.write().unwrap().entry(dir.to_path_buf()).or_insert(rules).clone()
    }
}

/// The rules in the ignore file `name` in `dir`; none when it is missing.
fn read_rules(dir: &Path, name: &str) -> Gitignore {
    Gitignore::new(dir.join(name)).0
}

/// Whether `rules` ignore `path` (`Some(true)`), whitelist it
/// (`Some(false)`), or say nothing about it (`None`).
fn ignore_verdict(rules: &Gitignore, path: &Path, is_dir: bool) -> Option<bool> {
    let matched = rules.matched(path, is_dir);
    (!matched.is_none()).then(|| matched.is_ignore())
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

/// Walks never descend through a link, but they do yield links, and
/// `is_file` and `read_to_string` follow them: a link in the project to a
/// file outside it would be listed by glob and read by grep. A link counts as
/// a project file only when its target is inside the canonical root.
fn link_stays_in_root(entry: &ignore::DirEntry, root_canon: &Path) -> bool {
    !entry.path_is_symlink() || entry.path().canonicalize().is_ok_and(|p| p.starts_with(root_canon))
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
    let root_canon = match root.canonicalize() {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(format!("cannot resolve project root: {e}")),
    };
    let walk_root = glob_walk_root(root, pattern);
    // The walker's filter skips entries named .git during descent, but never
    // the walk root itself, so a pattern scoped at or under .git would start
    // inside the excluded tree. Walk roots are also followed even though the
    // walk itself never follows links, so a symlinked prefix walks wherever
    // it points. The canonical prefix is the authority for both: it catches a
    // prefix that aliases .git without naming it, and one that leaves the
    // project.
    if walk_root.as_path() != root {
        // A nonexistent prefix walks nothing; let the normal path answer.
        let canon = walk_root.canonicalize().ok();
        if canon.as_ref().is_some_and(|c| !c.starts_with(&root_canon)) {
            return ToolOutcome::err(format!("path escapes the project root: {pattern}"));
        }
        let scoped_into_git =
            walk_root.strip_prefix(root).map(touches_git).unwrap_or(true);
        let aliases_git =
            canon.as_deref().and_then(|c| c.strip_prefix(&root_canon).ok()).is_some_and(touches_git);
        if scoped_into_git || aliases_git {
            return ToolOutcome::err(".git is excluded from search");
        }
    }
    let mut hits: Vec<(std::time::SystemTime, String)> = Vec::new();
    for entry in project_walker(&root_canon, &walk_root).build().flatten() {
        let path = entry.path();
        if !path.is_file() || !link_stays_in_root(&entry, &root_canon) {
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
    let root_canon = match root.canonicalize() {
        Ok(p) => p,
        Err(e) => return ToolOutcome::err(format!("cannot resolve project root: {e}")),
    };
    // resolve() canonicalized, so a path (or a symlink) that lands inside
    // .git names it here even when the argument never did. The walker's
    // filter cannot help once .git is the walk root.
    if search_root.strip_prefix(&root_canon).map(touches_git).unwrap_or(false) {
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let hits: std::sync::Mutex<Vec<(String, usize, String)>> = std::sync::Mutex::new(Vec::new());
    let enough = AtomicBool::new(false);
    // Files passed over, by reason: a skipped file can hold the match, so a
    // search that skips any must say so rather than answer "no matches".
    let too_large = AtomicUsize::new(0);
    let not_text = AtomicUsize::new(0);
    let unreadable = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12);
    project_walker(&root_canon, &search_root)
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
                if !path.is_file() || !link_stays_in_root(&entry, &root_canon) {
                    return WalkState::Continue;
                }
                if let Some(m) = &file_matcher {
                    let name_match =
                        path.file_name().map(|n| m.is_match(n.as_ref() as &Path)).unwrap_or(false);
                    if !name_match && !m.is_match(rel_display(root, path)) {
                        return WalkState::Continue;
                    }
                }
                // The size of what the read below gets: through a link, its
                // target, not the link itself.
                let text = match std::fs::metadata(path) {
                    Ok(m) if m.len() > MAX_FILE_BYTES => Err(&too_large),
                    Ok(_) => std::fs::read_to_string(path).map_err(|e| match e.kind() {
                        std::io::ErrorKind::InvalidData => &not_text,
                        _ => &unreadable,
                    }),
                    Err(_) => Err(&unreadable),
                };
                let text = match text {
                    Ok(text) => text,
                    Err(skipped) => {
                        skipped.fetch_add(1, Ordering::Relaxed);
                        return WalkState::Continue;
                    }
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

    let skipped = skipped_note(too_large.into_inner(), not_text.into_inner(), unreadable.into_inner());
    let mut hits = hits.into_inner().unwrap();
    if hits.is_empty() {
        return ToolOutcome::ok(match skipped {
            Some(note) => format!("no matches\n{note}"),
            None => "no matches".into(),
        });
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
    if let Some(note) = skipped {
        out.push_str(&note);
        out.push('\n');
    }
    ToolOutcome::ok(out)
}

/// One line counting the files grep passed over, by reason, or None when it
/// searched every file it walked.
fn skipped_note(too_large: usize, not_text: usize, unreadable: usize) -> Option<String> {
    let files = |n: usize| if n == 1 { "1 file".to_string() } else { format!("{n} files") };
    let mut reasons = Vec::new();
    if too_large > 0 {
        reasons.push(format!("{} over {} MB", files(too_large), MAX_FILE_BYTES as f64 / 1e6));
    }
    if not_text > 0 {
        reasons.push(format!("{} not UTF-8 text", files(not_text)));
    }
    if unreadable > 0 {
        reasons.push(format!("{} unreadable", files(unreadable)));
    }
    (!reasons.is_empty()).then(|| format!("… not searched: {}", reasons.join(", ")))
}

/// The tail of `text` within `max_bytes`, starting at a line boundary when
/// one is close by.
fn kept_tail(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    if let Some(nl) = text[start..].find('\n') {
        if nl < 200 {
            start += nl + 1;
        }
    }
    &text[start..]
}

/// Heads the result when any part of the output was dropped. The process
/// supervisor owns any bounded spill log; rendering never writes files.
fn truncation_notice(log_path: Option<&PathBuf>) -> String {
    match log_path {
        Some(path) => format!(
            "[start of output truncated; bounded output log saved to {}; tail or grep it with bash]",
            path.display()
        ),
        None => "[start of output truncated]".to_string(),
    }
}

fn captured_text(stream: &execution::CapturedStream) -> String {
    String::from_utf8_lossy(&stream.rendered_bytes()).into_owned()
}

fn stream_was_truncated(stream: &execution::CapturedStream) -> bool {
    stream.total_bytes > stream.head.len().saturating_add(stream.tail.len()) as u64
}

/// Format native-process output identically for bash and external tools.
/// The supervisor has already bounded each stream and owns any spill log.
///
/// Over the cap, each stream keeps its own tail, so a flood of warnings on
/// stderr cannot evict all of stdout, which is usually the result the command
/// ran for. A stream that needs less than half of the cap keeps all of it and
/// leaves the rest to the other, and a `…` marks each stream whose start was
/// dropped.
pub(crate) fn render_process_output(output: &ProcessOutput, max_bytes: usize) -> (String, bool) {
    let stdout = captured_text(&output.stdout);
    let stderr = captured_text(&output.stderr);
    let (label, stderr) = match (stderr.trim().is_empty(), stdout.is_empty()) {
        (true, _) => ("", ""),
        (false, true) => ("[stderr]\n", stderr.as_str()),
        (false, false) => ("\n[stderr]\n", stderr.as_str()),
    };
    let over_cap = stdout.len() + label.len() + stderr.len() > max_bytes;
    let needs_notice = output.log_truncated
        || stream_was_truncated(&output.stdout)
        || stream_was_truncated(&output.stderr)
        || over_cap;

    let budget = max_bytes.saturating_sub(label.len());
    let stdout_share = stdout.len().min((budget / 2).max(budget.saturating_sub(stderr.len())));
    let elide = |text: &str, share: usize, captured_whole: bool| {
        let kept = kept_tail(text, share);
        match captured_whole && kept.len() == text.len() {
            true => kept.to_string(),
            false => format!("…{kept}"),
        }
    };
    let mut text = elide(&stdout, stdout_share, !stream_was_truncated(&output.stdout));
    if !label.is_empty() {
        text.push_str(label);
        text.push_str(&elide(
            stderr,
            budget - stdout_share,
            !stream_was_truncated(&output.stderr),
        ));
    }
    if needs_notice {
        text = format!("{}\n{text}", truncation_notice(output.log_path.as_ref()));
    }
    if text.trim().is_empty() {
        ("(no output)".into(), needs_notice)
    } else {
        (text, needs_notice)
    }
}

/// The shell the bash tool runs. The tool is named bash and models write
/// bash, so run bash wherever it is installed: zsh's defaults break common
/// bash (an unmatched glob is an error, `$var` is not word-split, arrays
/// start at 1). Fixed paths, not a PATH lookup, so a PATH entry inside the
/// project cannot swap in an agent-written shell under every approved
/// command. Run with `-c`, never as a login shell: the child already
/// inherits the launching environment, and re-sourcing profiles costs time
/// per call and rewrites PATH (Debian's /etc/profile resets it outright,
/// dropping a container's `ENV PATH` toolchains; elsewhere an activated
/// virtualenv stops coming first).
fn bash_shell() -> &'static str {
    [
        "/bin/bash",
        "/usr/bin/bash",
        "/usr/local/bin/bash",
        "/run/current-system/sw/bin/bash",
    ]
    .into_iter()
    .find(|p| Path::new(p).exists())
    .unwrap_or("/bin/sh")
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
    let timeout_secs = arg(args, "timeout_secs", Value::as_u64).unwrap_or(60).clamp(1, 300);
    let request = ProcessRequest {
        program: bash_shell().into(),
        args: vec!["-c".into(), command.into()],
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
        self_link: true,
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
            Termination::Exited(status) => exited_outcome(&output, status, caps.command_bytes),
        },
    }
}

/// The result of a bash or external tool process that exited on its own.
/// Both run in their own process group with the same cleanup on exit, so
/// they share this rendering: a copy that drops the note below hands the
/// caller a clean exit and a helper that is already gone.
pub(crate) fn exited_outcome(
    output: &ProcessOutput,
    status: &std::process::ExitStatus,
    cap: usize,
) -> ToolOutcome {
    // The note is the harness speaking, not captured output, but it
    // still has to fit the caller's cap, which `max_output_bytes`
    // can set as low as 1000 bytes. Reserve its length instead of
    // letting a fixed annotation push the result past the limit.
    let reserved = match output.background_terminated {
        true => BACKGROUND_TERMINATED_NOTE.len() + 1,
        false => 0,
    };
    let budget = cap.saturating_sub(reserved).max(1);
    let (text, truncated) = render_process_output(output, budget);
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
    ToolOutcome::from_process(ok, text, output, truncated)
}

/// Each bash call and each external tool call runs in its own process group,
/// and the group is terminated when the call returns, so a backgrounded
/// process does not outlive it. The exit status is still the command's, which
/// means a caller that started a server sees success and an absent server,
/// with nothing connecting the two.
///
/// `setsid` is named as a conditional, not a recipe: it is util-linux and does
/// not exist on macOS, where the harness also runs. A named tmux session is the
/// answer that holds on both, and is what this project already documents for
/// durable background work.
pub(crate) const BACKGROUND_TERMINATED_NOTE: &str = concat!(
    "[openmax: this command left running background processes, and they were ",
    "terminated when it returned. Every bash or external tool call runs in its ",
    "own process group and that group is cleaned up on exit, so `&`, `nohup` ",
    "and `disown` do not survive the call. To keep something running, start it ",
    "in a named tmux session you can inspect and reattach, or, on Linux only, ",
    "detach it from the group with `setsid`.]"
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

    #[tokio::test]
    async fn a_started_file_mutation_settles_before_cancellation_returns() {
        let root = temp_project();
        let path = root.join("finished.txt");
        let written = path.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let task = tokio::task::spawn_blocking(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            std::fs::write(written, "settled").unwrap();
            ToolOutcome::ok("written".into())
        });
        let cancel = Arc::new(CancelToken::default());
        let mut settled = tokio::spawn(finish_file_task(task, true, cancel.clone()));
        started_rx.await.unwrap();
        cancel.cancel();
        let early = tokio::time::timeout(std::time::Duration::from_millis(50), &mut settled).await;
        let waited = early.is_err();
        release_tx.send(()).unwrap();
        let outcome = match early {
            Ok(result) => result.unwrap(),
            Err(_) => settled.await.unwrap(),
        };
        assert!(waited, "cancellation must not detach a started mutation");
        assert!(outcome.ok, "{}", outcome.output);
        assert_eq!(std::fs::read_to_string(path).unwrap(), "settled");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn edits_require_exact_whitespace_without_overlapping_fallbacks() {
        let root = temp_project();
        for (content, needle) in [
            ("if ready:\n    run()\n", "if ready:\nrun()"),
            ("items:\n  - first\n", "items:\n- first"),
            ("\tfirst\n\tsecond\n", "first\nsecond"),
            // CRLF files read LF needles as CRLF, but that never stacks with
            // indentation fuzzing, and a mixed-ending file stays exact.
            ("\tfirst\r\n\tsecond\r\n", "first\nsecond"),
            ("first\r\nsecond\nthird\r\n", "first\nsecond\nthird"),
            (" a\n a\n a\n a\n", "a\na\na"),
        ] {
            std::fs::write(root.join("exact.txt"), content).unwrap();
            let out = edit_file(&root, &json!({
                "path": "exact.txt", "old_string": needle, "new_string": "changed", "replace_all": true
            }));
            assert!(!out.ok, "non-exact edit unexpectedly succeeded: {content:?}");
            assert_eq!(std::fs::read_to_string(root.join("exact.txt")).unwrap(), content);
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// An empty old_string matches between every pair of characters. The
    /// ambiguity error then steers the model to replace_all, which splices
    /// new_string between every character and corrupts the whole file.
    #[test]
    fn an_empty_old_string_is_refused_and_leaves_the_file_untouched() {
        let root = temp_project();
        let original = "abc\r\ndef\n";
        std::fs::write(root.join("keep.txt"), original).unwrap();
        for replace_all in [true, false] {
            let out = edit_file(&root, &json!({
                "path": "keep.txt", "old_string": "", "new_string": "X", "replace_all": replace_all
            }));
            assert!(!out.ok, "an empty old_string must be refused (replace_all={replace_all}): {}", out.output);
            assert!(out.output.contains("old_string is empty"), "{}", out.output);
            assert!(out.output.contains("write_file"), "the refusal names the whole-file tool: {}", out.output);
            assert!(!out.output.contains("replace_all"), "the refusal must not steer to replace_all: {}", out.output);
            assert_eq!(std::fs::read(root.join("keep.txt")).unwrap(), original.as_bytes());
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// read_file shows lines without their \r, so text copied from it has
    /// bare LF breaks. On a CRLF file that copy must still match, and the
    /// replacement must keep the file's CRLF endings rather than splice LF
    /// lines into it.
    #[test]
    fn an_edit_copied_from_read_file_matches_a_crlf_file_and_keeps_crlf() {
        let root = temp_project();
        std::fs::write(root.join("win.txt"), "alpha\r\nbeta\r\ngamma\r\n").unwrap();
        let read = read_file(&root, &json!({"path": "win.txt"}));
        assert!(read.ok, "{}", read.output);
        let shown: Vec<&str> = read
            .output
            .lines()
            .map(|l| l.trim_start().split_once(' ').unwrap().1)
            .collect();
        let old_string = shown[..2].join("\n");
        assert_eq!(old_string, "alpha\nbeta", "read_file hides the \\r: {:?}", read.output);
        let out = edit_file(&root, &json!({
            "path": "win.txt", "old_string": old_string, "new_string": "alpha\nBETA\nbeta two"
        }));
        assert!(out.ok, "a two-line edit copied from read_file must match a CRLF file: {}", out.output);
        assert_eq!(
            std::fs::read_to_string(root.join("win.txt")).unwrap(),
            "alpha\r\nBETA\r\nbeta two\r\ngamma\r\n"
        );

        // A single-line match whose replacement adds lines keeps CRLF too.
        let out = edit_file(&root, &json!({
            "path": "win.txt", "old_string": "gamma", "new_string": "gamma\ndelta"
        }));
        assert!(out.ok, "{}", out.output);
        assert_eq!(
            std::fs::read_to_string(root.join("win.txt")).unwrap(),
            "alpha\r\nBETA\r\nbeta two\r\ngamma\r\ndelta\r\n"
        );

        // Uniqueness is judged on the CRLF form, exactly as before.
        std::fs::write(root.join("dup.txt"), "x\r\ny\r\nx\r\ny\r\n").unwrap();
        let out = edit_file(&root, &json!({"path": "dup.txt", "old_string": "x\ny", "new_string": "z"}));
        assert!(!out.ok && out.output.contains("matches 2 times"), "{}", out.output);
        assert_eq!(std::fs::read_to_string(root.join("dup.txt")).unwrap(), "x\r\ny\r\nx\r\ny\r\n");
        let out = edit_file(&root, &json!({
            "path": "dup.txt", "old_string": "x\ny", "new_string": "z\nw", "replace_all": true
        }));
        assert!(out.ok, "{}", out.output);
        assert_eq!(std::fs::read_to_string(root.join("dup.txt")).unwrap(), "z\r\nw\r\nz\r\nw\r\n");

        // Strings that differ only in line endings become the same edit once
        // read as CRLF, so they are refused like identical strings instead of
        // reporting a line-ending change that was never written.
        std::fs::write(root.join("ends.txt"), "a\r\nb\r\n").unwrap();
        for (old_string, new_string) in [("a\r\nb", "a\nb"), ("a\nb", "a\r\nb")] {
            let out = edit_file(&root, &json!({
                "path": "ends.txt", "old_string": old_string, "new_string": new_string
            }));
            assert!(!out.ok, "a line-ending-only edit must not report success: {}", out.output);
            assert!(out.output.contains("differ only in line endings"), "{}", out.output);
            assert_eq!(std::fs::read(root.join("ends.txt")).unwrap(), b"a\r\nb\r\n");
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// A file mixing CRLF and LF has no single ending to assume, and
    /// read_file shows neither, so a multi-line edit stays exact and the
    /// failure says why instead of pointing at a line that looks identical.
    #[test]
    fn a_mixed_ending_file_stays_exact_and_the_miss_says_why() {
        let root = temp_project();
        let original = "one\r\ntwo\nthree\r\n";
        std::fs::write(root.join("mixed.txt"), original).unwrap();
        let out = edit_file(&root, &json!({"path": "mixed.txt", "old_string": "one\ntwo", "new_string": "1\n2"}));
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("mixes CRLF and LF line endings"), "{}", out.output);
        assert_eq!(std::fs::read_to_string(root.join("mixed.txt")).unwrap(), original);
        // The text that does match exactly still edits, untouched otherwise.
        let out = edit_file(&root, &json!({"path": "mixed.txt", "old_string": "two\nthree", "new_string": "2\n3"}));
        assert!(out.ok, "{}", out.output);
        assert_eq!(std::fs::read_to_string(root.join("mixed.txt")).unwrap(), "one\r\n2\n3\r\n");
        let _ = std::fs::remove_dir_all(root);
    }

    /// read_file in a call nothing cancels.
    fn read_file(root: &Path, args: &Value) -> ToolOutcome {
        super::read_file(root, args, &CancelToken::default())
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

    /// A file grep passes over can hold the match, so a search that skipped
    /// it must say so instead of answering "no matches".
    #[test]
    fn grep_counts_the_files_it_could_not_search() {
        let root = temp_project();
        let filler = "padding line without the word\n";
        let big = filler.repeat(MAX_FILE_BYTES as usize / filler.len() + 1) + "needle\n";
        std::fs::write(root.join("docs/huge.log"), big).unwrap();
        std::fs::write(root.join("docs/latin1.txt"), b"needle caf\xe9\n").unwrap();

        let out = grep_tool(&root, &json!({"pattern": "needle"}));
        assert!(out.ok, "{}", out.output);
        assert_eq!(
            out.output,
            "no matches\n… not searched: 1 file over 1.5 MB, 1 file not UTF-8 text",
            "skipped files are counted by reason"
        );

        // The note follows the hits, and counts every file of a reason.
        std::fs::write(root.join("docs/latin1-2.txt"), b"caf\xe9\n").unwrap();
        std::fs::write(root.join("src/needle.rs"), "// needle\n").unwrap();
        let out = grep_tool(&root, &json!({"pattern": "needle"}));
        assert!(out.ok, "{}", out.output);
        assert_eq!(
            out.output,
            "src/needle.rs:1: // needle\n… not searched: 1 file over 1.5 MB, 2 files not UTF-8 text\n"
        );

        // A search that read every candidate carries no note.
        let out = grep_tool(&root, &json!({"pattern": "needle", "path": "src"}));
        assert_eq!(out.output, "src/needle.rs:1: // needle\n");
        let _ = std::fs::remove_dir_all(root);
    }

    /// The ignore crate reads .gitignore only inside a git repository by
    /// default, so in a project without one grep and glob walked
    /// node_modules and build output the project had ignored.
    #[test]
    fn grep_and_glob_honor_gitignore_without_a_git_repository() {
        let root = temp_project();
        std::fs::write(root.join(".gitignore"), "node_modules/\nbuild/\n").unwrap();
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::write(root.join("node_modules/pkg/index.js"), "alpha_vendored\n").unwrap();
        std::fs::create_dir_all(root.join("build")).unwrap();
        std::fs::write(root.join("build/out.js"), "alpha_vendored\n").unwrap();
        std::fs::write(root.join("src/app.js"), "alpha_vendored\n").unwrap();

        let out = glob_tool(&root, &json!({"pattern": "**/*.js"}));
        assert_eq!(out.output, "src/app.js", "ignored directories are not listed");
        let out = grep_tool(&root, &json!({"pattern": "alpha_vendored"}));
        assert_eq!(out.output, "src/app.js:1: alpha_vendored\n", "ignored directories are not searched");

        // A walk that starts below the root still applies the root's rules,
        // and a path named outright is searched even though they ignore it.
        std::fs::create_dir_all(root.join("src/build")).unwrap();
        std::fs::write(root.join("src/build/gen.js"), "alpha_vendored\n").unwrap();
        let out = glob_tool(&root, &json!({"pattern": "src/**/*.js"}));
        assert_eq!(out.output, "src/app.js", "a scoped glob applies the root's .gitignore");
        let out = grep_tool(&root, &json!({"pattern": "alpha_vendored", "path": "src"}));
        assert_eq!(out.output, "src/app.js:1: alpha_vendored\n", "a scoped grep applies the root's .gitignore");
        let out = grep_tool(&root, &json!({"pattern": "alpha_vendored", "path": "build"}));
        assert_eq!(out.output, "build/out.js:1: alpha_vendored\n", "a named path is searched");

        // Inside a repository the rules stop at its root as before: a
        // .gitignore above it, such as a home directory's "*", must not hide
        // the project.
        let outer = outside_dir();
        std::fs::create_dir_all(outer.join(".git")).unwrap();
        std::fs::write(outer.join(".gitignore"), "*\n").unwrap();
        let project = outer.join("project");
        std::fs::create_dir_all(project.join(".git")).unwrap();
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::write(project.join("src/lib.rs"), "fn alpha() {}\n").unwrap();
        let out = glob_tool(&project, &json!({"pattern": "**/*.rs"}));
        assert_eq!(out.output, "src/lib.rs");
        let out = grep_tool(&project, &json!({"pattern": "alpha"}));
        assert_eq!(out.output, "src/lib.rs:1: fn alpha() {}\n");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outer);
    }

    /// Outside a repository nothing marks where git's rules stop, so a walker
    /// told to read .gitignore there read every one up to the filesystem
    /// root. git applies none of them, and one that ignores `*` (a home
    /// directory whose dotfiles are tracked from elsewhere) hid the whole
    /// project: glob and grep answered as if it were empty.
    #[test]
    fn a_gitignore_above_a_project_without_git_cannot_hide_it() {
        let outer = outside_dir();
        std::fs::write(outer.join(".gitignore"), "*\n").unwrap();
        // A .ignore file applies from any ancestor, as it always has.
        std::fs::write(outer.join(".ignore"), "*.secret\n").unwrap();
        let project = outer.join("project");
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::write(project.join("src/lib.rs"), "fn alpha() {}\n").unwrap();
        std::fs::write(project.join("src/key.secret"), "alpha\n").unwrap();

        for pattern in ["**/*", "src/*"] {
            let out = glob_tool(&project, &json!({"pattern": pattern}));
            assert_eq!(out.output, "src/lib.rs", "glob {pattern}");
        }
        for path in [".", "src"] {
            let out = grep_tool(&project, &json!({"pattern": "alpha", "path": path}));
            assert_eq!(out.output, "src/lib.rs:1: fn alpha() {}\n", "grep in {path}");
        }
        let _ = std::fs::remove_dir_all(outer);
    }

    /// The walker passes an entry when no ignore file it read ignores it,
    /// though one may whitelist it, and a nearer file outranks a farther
    /// one. A rule from above a scoped walk pruned such an entry on its own,
    /// so a glob or grep scoped below the root answered "no matches" for a
    /// file the unscoped one found.
    #[test]
    fn a_scoped_walk_ranks_ignore_files_as_a_whole_project_walk_does() {
        let outer = outside_dir();
        // Every .ignore outranks every .gitignore, however near.
        std::fs::write(outer.join(".ignore"), "*.secret\n").unwrap();
        let project = outer.join("project");
        std::fs::create_dir_all(project.join("src/gen")).unwrap();
        std::fs::write(project.join(".gitignore"), "*.gen\n").unwrap();
        // drop.gen stays ignored: nothing nearer than the root's rule names it.
        std::fs::write(project.join("src/.gitignore"), "!keep.gen\n!key.secret\n").unwrap();
        for file in ["keep.gen", "drop.gen", "key.secret"] {
            std::fs::write(project.join("src/gen").join(file), "needle\n").unwrap();
        }

        for pattern in ["**/*.{gen,secret}", "src/**/*.{gen,secret}", "src/gen/*.{gen,secret}"] {
            let out = glob_tool(&project, &json!({"pattern": pattern}));
            assert_eq!(out.output, "src/gen/keep.gen", "glob {pattern}");
        }
        for path in [".", "src", "src/gen"] {
            let out = grep_tool(&project, &json!({"pattern": "needle", "path": path}));
            assert_eq!(out.output, "src/gen/keep.gen:1: needle\n", "grep in {path}");
        }
        let _ = std::fs::remove_dir_all(outer);
    }

    /// A .ignore that whitelists an entry outranks a .gitignore that ignores
    /// it, wherever each file sits. The walker pruned on a .gitignore inside
    /// the walk before a .ignore above the walk root could keep the entry,
    /// so a scoped glob or grep missed a file the unscoped one found, and a
    /// .ignore above the project root lost to the project's own .gitignore.
    #[test]
    fn an_ignore_file_above_the_walk_outranks_a_gitignore_inside_it() {
        let outer = outside_dir();
        std::fs::write(outer.join(".ignore"), "!keep.out\n").unwrap();
        let project = outer.join("project");
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::write(project.join(".ignore"), "!keep.gen\n").unwrap();
        std::fs::write(project.join("src/.gitignore"), "*.gen\n*.out\n").unwrap();
        for file in ["keep.gen", "drop.gen", "keep.out", "drop.out"] {
            std::fs::write(project.join("src").join(file), format!("{file}\n")).unwrap();
        }

        for ext in ["gen", "out"] {
            for pattern in [format!("**/*.{ext}"), format!("src/*.{ext}")] {
                let out = glob_tool(&project, &json!({"pattern": pattern}));
                assert_eq!(out.output, format!("src/keep.{ext}"), "glob {pattern}");
            }
            for path in [".", "src"] {
                let out = grep_tool(&project, &json!({"pattern": format!("^(keep|drop)\\.{ext}$"), "path": path}));
                assert_eq!(out.output, format!("src/keep.{ext}:1: keep.{ext}\n"), "grep {ext} in {path}");
            }
        }
        let _ = std::fs::remove_dir_all(outer);
    }

    /// git applies the user's global excludes only inside a repository. A
    /// walker told to read .gitignore outside one applied them there too, so
    /// a project without git lost every file they name (`*.log`, `.env`)
    /// from glob and grep without a word.
    #[cfg(unix)]
    #[test]
    fn global_git_excludes_apply_only_inside_a_repository() {
        const CHILD: &str = "OPENMAX_TEST_GLOBAL_EXCLUDES";
        if std::env::var_os(CHILD).is_none() {
            // The excludes file is found through HOME and XDG_CONFIG_HOME,
            // which every test in this process shares: run this test in a
            // child whose HOME and XDG_CONFIG_HOME lead to one naming *.log.
            let home = outside_dir();
            std::fs::create_dir_all(home.join(".config/git")).unwrap();
            std::fs::write(home.join(".config/git/ignore"), "*.log\n").unwrap();
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "tools::tests::global_git_excludes_apply_only_inside_a_repository"])
                .env(CHILD, "1")
                .env("HOME", &home)
                .env("XDG_CONFIG_HOME", home.join(".config"))
                .output()
                .unwrap();
            let _ = std::fs::remove_dir_all(&home);
            let stdout = String::from_utf8_lossy(&child.stdout);
            // "1 passed" rules out a filter that matched nothing and exited 0.
            assert!(
                child.status.success() && stdout.contains("1 passed"),
                "{stdout}{}",
                String::from_utf8_lossy(&child.stderr)
            );
            return;
        }
        let project = outside_dir();
        std::fs::write(project.join("notes.log"), "alpha\n").unwrap();
        let out = glob_tool(&project, &json!({"pattern": "**/*.log"}));
        assert_eq!(out.output, "notes.log", "glob outside a repository");
        let out = grep_tool(&project, &json!({"pattern": "alpha"}));
        assert_eq!(out.output, "notes.log:1: alpha\n", "grep outside a repository");
        // Inside one they apply, as git applies them.
        std::fs::create_dir_all(project.join(".git")).unwrap();
        let out = glob_tool(&project, &json!({"pattern": "**/*.log"}));
        assert_eq!(out.output, "no files matched", "glob inside a repository");
        let out = grep_tool(&project, &json!({"pattern": "alpha"}));
        assert_eq!(out.output, "no matches", "grep inside a repository");
        let _ = std::fs::remove_dir_all(project);
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

    fn outside_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openmax-outside-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    /// A dangling link does not "exist" to `Path::exists`, so its name used to
    /// pass the root check as a not-yet-created file, and the write then
    /// followed it and created the target outside the project. A cloned
    /// repository can ship such a link.
    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_cannot_carry_a_write_outside_the_root() {
        let root = temp_project();
        let outside = outside_dir();
        std::os::unix::fs::symlink(outside.join("planted.txt"), root.join("notes.txt")).unwrap();
        std::os::unix::fs::symlink(outside.join("missing"), root.join("cache")).unwrap();
        // A link to a dangling link is the same escape one hop later.
        std::os::unix::fs::symlink("notes.txt", root.join("relay.txt")).unwrap();
        // A trailing separator or "." on the path, or on a link's target,
        // makes lstat follow the final link: its name must not pass as a
        // file that does not exist yet.
        std::os::unix::fs::symlink("notes.txt/", root.join("trail.txt")).unwrap();

        let paths =
            ["notes.txt", "cache/planted.txt", "relay.txt", "src/../notes.txt", "notes.txt/", "notes.txt/.", "trail.txt"];
        for path in paths {
            let out = write_file(&root, &json!({"path": path, "content": "x\n"}));
            assert!(!out.ok, "{path}: write through a dangling link succeeded: {}", out.output);
            assert!(out.output.contains("path escapes the project root"), "{path}: {}", out.output);
        }
        let out = edit_file(&root, &json!({"path": "notes.txt", "old_string": "a", "new_string": "b"}));
        assert!(!out.ok, "{}", out.output);
        assert!(!outside.join("planted.txt").exists(), "a write landed outside the root");
        assert!(!outside.join("missing").exists(), "a directory was created outside the root");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    /// `is_file` and `read_to_string` follow links, so a link inside the
    /// project to a file outside it used to be read by grep and listed by
    /// glob. A glob prefix that leaves the project through a link must be
    /// refused as the escape it is.
    #[cfg(unix)]
    #[test]
    fn search_tools_do_not_follow_symlinks_out_of_the_root() {
        let root = temp_project();
        let outside = outside_dir();
        std::fs::write(outside.join("secret.txt"), "alpha outside the project\n").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("vendor")).unwrap();
        std::os::unix::fs::symlink(outside.join("secret.txt"), root.join("docs/leak.txt")).unwrap();

        let out = grep_tool(&root, &json!({"pattern": "outside the project"}));
        assert!(out.ok && out.output == "no matches", "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "outside the project", "path": "docs"}));
        assert!(out.ok && out.output == "no matches", "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "alpha", "path": "vendor"}));
        assert!(!out.ok && out.output.contains("path escapes the project root"), "{}", out.output);

        let out = glob_tool(&root, &json!({"pattern": "**/*.txt"}));
        assert!(!out.output.contains("leak.txt"), "{}", out.output);
        // The refusal names the escape; it is not a .git exclusion.
        for pattern in ["vendor/**", "vendor/*.txt"] {
            let out = glob_tool(&root, &json!({"pattern": pattern}));
            assert!(!out.output.contains("secret.txt"), "{pattern}: {}", out.output);
            assert!(!out.ok && out.output.contains("path escapes the project root"), "{pattern}: {}", out.output);
        }

        let out = list_dir(&root, &json!({"path": "vendor"}));
        assert!(!out.ok && out.output.contains("path escapes the project root"), "{}", out.output);
        let out = read_file(&root, &json!({"path": "docs/leak.txt"}));
        assert!(!out.ok && out.output.contains("path escapes the project root"), "{}", out.output);
        // A trailing separator or "." must not hide the link's final hop.
        for path in ["docs/leak.txt/", "docs/leak.txt/."] {
            let out = read_file(&root, &json!({"path": path}));
            assert!(!out.ok && out.output.contains("path escapes the project root"), "{path}: {}", out.output);
            let out = edit_file(&root, &json!({"path": path, "old_string": "alpha", "new_string": "beta"}));
            assert!(!out.ok && out.output.contains("path escapes the project root"), "{path}: {}", out.output);
            let out = grep_tool(&root, &json!({"pattern": "alpha", "path": path}));
            assert!(!out.ok && out.output.contains("path escapes the project root"), "{path}: {}", out.output);
        }
        assert_eq!(std::fs::read_to_string(outside.join("secret.txt")).unwrap(), "alpha outside the project\n");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    /// Confinement judges where a link lands, not that it is a link: links
    /// that stay in the project work in every file tool, dangling or not.
    #[cfg(unix)]
    #[test]
    fn symlinks_that_stay_in_the_root_keep_working() {
        let root = temp_project();
        std::os::unix::fs::symlink(root.join("src"), root.join("alias")).unwrap();
        std::os::unix::fs::symlink("src/a.rs", root.join("a_link.rs")).unwrap();
        std::os::unix::fs::symlink(root.join("docs/new.md"), root.join("pending.md")).unwrap();

        let out = glob_tool(&root, &json!({"pattern": "alias/**/*.rs"}));
        assert!(out.ok && out.output.contains("alias/deep/b.rs"), "{}", out.output);
        let out = glob_tool(&root, &json!({"pattern": "*.rs"}));
        assert!(out.ok && out.output.contains("a_link.rs"), "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "alpha_two"}));
        assert!(out.ok && out.output.contains("a_link.rs:2:"), "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "alpha_three", "path": "alias"}));
        assert!(out.ok && out.output.contains("deep/b.rs:1:"), "{}", out.output);
        let out = list_dir(&root, &json!({"path": "alias"}));
        assert!(out.ok && out.output.contains("a.rs"), "{}", out.output);
        let out = read_file(&root, &json!({"path": "a_link.rs"}));
        assert!(out.ok && out.output.contains("alpha_two"), "{}", out.output);

        let out = write_file(&root, &json!({"path": "pending.md", "content": "landed\n"}));
        assert!(out.ok, "{}", out.output);
        assert_eq!(std::fs::read_to_string(root.join("docs/new.md")).unwrap(), "landed\n");
        let out = write_file(&root, &json!({"path": "alias/fresh.rs", "content": "fn f() {}\n"}));
        assert!(out.ok, "{}", out.output);
        assert!(root.join("src/fresh.rs").exists());
        let out = edit_file(&root, &json!({"path": "a_link.rs", "old_string": "alpha_two", "new_string": "beta"}));
        assert!(out.ok, "{}", out.output);
        assert!(std::fs::read_to_string(root.join("src/a.rs")).unwrap().contains("beta"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// A trailing separator or `.`, on the path or on a link's target, makes
    /// the OS require a directory: opening a link to `important.txt/` fails.
    /// Resolution drops the separator to find the entry, so without keeping
    /// that requirement a write through such a link overwrote `important.txt`.
    #[cfg(unix)]
    #[test]
    fn a_trailing_separator_still_requires_a_directory() {
        let root = temp_project();
        std::fs::write(root.join("important.txt"), "original\n").unwrap();
        std::os::unix::fs::symlink("important.txt/", root.join("link.txt")).unwrap();
        std::os::unix::fs::symlink("fresh.txt/", root.join("pending.txt")).unwrap();

        for path in ["link.txt", "important.txt/", "important.txt/.", "pending.txt", "fresh.txt/"] {
            let out = write_file(&root, &json!({"path": path, "content": "overwritten\n"}));
            assert!(!out.ok && out.output.contains("not a directory"), "{path}: {}", out.output);
        }
        for path in ["link.txt", "important.txt/"] {
            let out = read_file(&root, &json!({"path": path}));
            assert!(!out.ok && out.output.contains("not a directory"), "{path}: {}", out.output);
            let out = edit_file(&root, &json!({"path": path, "old_string": "original", "new_string": "edited"}));
            assert!(!out.ok && out.output.contains("not a directory"), "{path}: {}", out.output);
        }
        assert_eq!(std::fs::read_to_string(root.join("important.txt")).unwrap(), "original\n");
        assert!(!root.join("fresh.txt").exists(), "a file was created where a directory was named");
        // A directory named with a trailing separator still resolves.
        let out = list_dir(&root, &json!({"path": "src/"}));
        assert!(out.ok && out.output.contains("a.rs"), "{}", out.output);
        let out = grep_tool(&root, &json!({"pattern": "alpha_three", "path": "src/deep/."}));
        assert!(out.ok && out.output.contains("b.rs:1:"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn command_truncation_keeps_the_tail() {
        let mut text = String::new();
        for i in 0..4000 {
            text.push_str(&format!("line number {i} with some padding text\n"));
        }
        assert!(text.len() > MAX_OUTPUT_BYTES);
        let output = rendered_output(
            stream(text.len() as u64, b"", text.as_bytes()),
            stream(0, b"", b""),
            None,
            false,
        );
        let (kept, _) = render_process_output(&output, MAX_OUTPUT_BYTES);
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

    /// When both streams overflow, neither is starved: each keeps its own
    /// tail, and each cut is marked where it happened.
    #[test]
    fn process_renderer_splits_the_cap_when_both_streams_overflow() {
        let lines = |tag: &str| (0..100).map(|i| format!("{tag}-{i:03}\n")).collect::<String>();
        let (out, err) = (lines("out"), lines("err"));
        let output = rendered_output(
            stream(out.len() as u64, b"", out.as_bytes()),
            stream(err.len() as u64, b"", err.as_bytes()),
            None,
            false,
        );
        let cap = 200;
        let (text, truncated) = render_process_output(&output, cap);
        assert!(truncated);
        let (notice, body) = text.split_once('\n').unwrap();
        assert_eq!(notice, "[start of output truncated]");
        let (kept_out, kept_err) = body.split_once("\n[stderr]\n").expect(&text);
        assert!(kept_out.starts_with('…') && kept_out.ends_with("out-099\n"), "{text}");
        assert!(kept_err.starts_with('…') && kept_err.ends_with("err-099\n"), "{text}");
        assert!(!body.contains("out-000") && !body.contains("err-000"), "{text}");
        // An even split, give or take the line each cut snaps forward to.
        let line = "out-000\n".len();
        for kept in [kept_out, kept_err] {
            assert!(kept.len() + line >= (cap - "\n[stderr]\n".len()) / 2, "{text}");
        }
        assert!(body.replace('…', "").len() <= cap, "{text}");
    }

    /// The prompt, receipts, and --check rows tell the agent to run a bare
    /// `openmax` through bash, and bash inherits PATH: an older install
    /// earlier on it would answer with claims this build has retracted,
    /// under the same version string. A bare `openmax` must resolve to a
    /// link to the running executable, from the first PATH entry, so no
    /// inherited install can come before it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_bare_openmax_in_bash_runs_the_running_binary() {
        // Read from inside the shell instead of moving this process's PATH:
        // tests run in parallel and share one environment.
        let root = temp_project();
        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "command -v openmax && printf '%s\\n' \"${PATH%%:*}\""}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        let _ = std::fs::remove_dir_all(&root);
        assert!(out.ok, "{}", out.output);
        let mut lines = out.output.lines().map(|line| PathBuf::from(line.trim()));
        let found = lines.next().unwrap_or_default();
        let first = lines.next().unwrap_or_default();
        let exe = std::env::current_exe().unwrap().canonicalize().unwrap();
        assert_eq!(
            std::fs::read_link(&found).ok(),
            Some(exe),
            "`openmax` in bash resolved to {}, not a link to the running executable",
            found.display()
        );
        assert_eq!(
            found.parent(),
            Some(first.as_path()),
            "the link's directory must be first on PATH, ahead of any inherited install"
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

    /// Commands ran under zsh wherever it was installed, so on every Mac a
    /// model's bash met 1-indexed arrays and unmatched globs as errors, and as
    /// a login shell that re-sourced profiles on every call. `${arr[1]}` naming
    /// the second element with `login_shell` off holds for non-login bash only.
    /// A host with no bash at all runs the `/bin/sh` fallback, so there the
    /// test proves a portable command still runs instead of passing blind.
    #[tokio::test]
    async fn the_bash_tool_runs_bash_as_a_non_login_shell() {
        let root = temp_project();
        let (command, expected) = if bash_shell() == "/bin/sh" {
            ("echo portable", "portable")
        } else {
            ("arr=(zero one); echo \"${arr[1]}\"; if shopt -q login_shell; then echo login; fi", "one")
        };
        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": command}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(out.ok, "{}", out.output);
        assert_eq!(out.output.trim(), expected, "{} as a non-login shell: {}", bash_shell(), out.output);
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

    /// Compiler and test warnings go to stderr by the thousand while the
    /// result the command ran for is on stdout. Cutting the tail of the
    /// joined text let stderr alone fill the cap and evict every byte of
    /// stdout, so the caller saw warnings and never the result.
    #[tokio::test]
    async fn a_stderr_flood_does_not_evict_stdout() {
        let root = temp_project();
        let cap = 1_000;
        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "echo RESULT; for i in $(seq 1 3000); do echo warn-$i >&2; done"}),
            OutputCaps { command_bytes: cap },
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(out.ok, "{}", out.output);
        let stderr_bytes = out.process_bytes.unwrap() - "RESULT\n".len() as u64;
        assert!(stderr_bytes > cap as u64, "stderr alone must overflow the cap: {stderr_bytes}");
        assert!(out.output.contains("RESULT"), "stdout must survive a stderr flood: {}", out.output);
        assert!(out.output.contains("\n[stderr]\n"), "{}", out.output);
        assert!(out.output.ends_with("warn-3000\n"), "stderr keeps its tail: {}", out.output);
        assert!(!out.output.contains("warn-1\n"), "and drops its head: {}", out.output);
        // The notice and the elision marks sit outside the cap, as they always
        // have; the captured text they frame stays within it.
        let (notice, body) = out.output.split_once('\n').unwrap();
        assert!(notice.contains("bounded output log saved to"), "{notice}");
        assert!(
            body.replace('…', "").len() <= cap,
            "both streams share one cap: {} bytes",
            body.len()
        );
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

    /// A file over the size cap was refused before offset and limit were
    /// read, while the refusal told the model to retry with offset and
    /// limit: a retry that could never work. A window of it streams, and
    /// shows exactly what the same lines show in a file read whole.
    #[test]
    fn read_file_reads_a_window_of_a_file_too_large_to_load_whole() {
        let root = temp_project();
        let mut lines: Vec<String> = Vec::new();
        let mut bytes = 0;
        while bytes <= MAX_FILE_BYTES as usize {
            let line = format!("row {} of a large log\n", lines.len() + 1);
            bytes += line.len();
            lines.push(line);
        }
        let total = lines.len();
        let offset = total / 2;
        // In the window: a long line whose clip point falls inside a
        // three-byte character, and a CRLF line.
        lines[offset] = format!("{}\n", "€".repeat(400));
        lines[offset + 1] = "carriage return\r\n".into();
        std::fs::write(root.join("big.log"), lines.concat()).unwrap();
        assert!(std::fs::metadata(root.join("big.log")).unwrap().len() > MAX_FILE_BYTES);

        let out = read_file(&root, &json!({"path": "big.log", "offset": offset, "limit": 4}));
        assert!(out.ok, "a window of a large file must be readable: {}", out.output);
        let shown: Vec<&str> = out.output.lines().collect();
        assert_eq!(shown.len(), 5, "{}", out.output);
        assert_eq!(
            shown[4],
            format!("… {} more lines (file has {total} lines; continue with offset={})", total - (offset + 3), offset + 4)
        );
        // The same lines in a file small enough to read whole render the same.
        std::fs::write(root.join("small.log"), lines[..offset + 3].concat()).unwrap();
        let small = read_file(&root, &json!({"path": "small.log", "offset": offset, "limit": 4}));
        assert!(small.ok, "{}", small.output);
        assert_eq!(shown[..4], small.output.lines().collect::<Vec<_>>()[..], "{}", out.output);
        assert!(shown[1].contains("[line clipped; 702 more bytes]"), "{}", shown[1]);
        assert!(shown[2].ends_with(" carriage return"), "{:?}", shown[2]);

        // The final line is reachable, and an offset past it says so.
        let out = read_file(&root, &json!({"path": "big.log", "offset": total}));
        assert_eq!(out.output, format!("{total:>5} row {total} of a large log\n"));
        let out = read_file(&root, &json!({"path": "big.log", "offset": total + 1}));
        assert!(!out.ok && out.output.contains(&format!("past the end of big.log ({total} lines)")), "{}", out.output);
        // A whole-file read is still refused, and the refusal names only the
        // window above: grep skips a file this large, so pointing the model
        // at it cost a turn on a search that never looked inside.
        let len = std::fs::metadata(root.join("big.log")).unwrap().len();
        let out = read_file(&root, &json!({"path": "big.log"}));
        let advice = format!("file too large ({len} bytes); read a range of lines with offset and limit");
        assert_eq!((out.ok, out.output.as_str()), (false, advice.as_str()));
        let _ = std::fs::remove_dir_all(root);
    }

    /// The caller stops waiting for a read once the call is cancelled, but a
    /// window of a large file went on reading to the end of the file to count
    /// its lines, however large the file.
    #[test]
    fn a_cancelled_read_of_a_large_file_stops_reading() {
        let root = temp_project();
        let line = "row of a large log\n";
        std::fs::write(root.join("big.log"), line.repeat(MAX_FILE_BYTES as usize / line.len() + 1)).unwrap();
        let cancel = CancelToken::default();
        cancel.cancel();
        let out = super::read_file(&root, &json!({"path": "big.log", "offset": 1, "limit": 1}), &cancel);
        assert_eq!((out.ok, out.output.as_str()), (false, "tool cancelled by user"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// A streamed window must number and split lines exactly as a file read
    /// whole does, wherever the reader's buffer boundaries fall.
    #[test]
    fn streamed_lines_split_like_str_lines() {
        for text in ["", "a", "a\n", "a\n\n", "a\r\nb", "a\r", "a\r\r\nb\n", "\r\n", "a\rb\n", "héllo\r\nwörld\n"] {
            for capacity in [1, 2, 3, 64] {
                let mut reader = std::io::BufReader::with_capacity(capacity, text.as_bytes());
                let mut kept = Vec::new();
                let mut lines = Vec::new();
                while let Some(len) = next_line(&mut reader, &mut kept, usize::MAX).unwrap() {
                    assert_eq!(len, kept.len(), "{text:?} at capacity {capacity}");
                    lines.push(String::from_utf8(kept.clone()).unwrap());
                }
                assert_eq!(lines, text.lines().collect::<Vec<_>>(), "{text:?} at capacity {capacity}");
            }
        }
        // A long line keeps only its prefix but reports its whole length.
        let mut reader = std::io::BufReader::with_capacity(2, "abcdef\r\nxy".as_bytes());
        let mut kept = Vec::new();
        assert_eq!(next_line(&mut reader, &mut kept, 2).unwrap(), Some(6));
        assert_eq!(kept, b"ab");
        assert_eq!(next_line(&mut reader, &mut kept, 0).unwrap(), Some(2));
        assert!(kept.is_empty());
        assert_eq!(next_line(&mut reader, &mut kept, 2).unwrap(), None);
    }

    /// read_file discarded the read error and called every failure "not a
    /// UTF-8 text file", so a directory or a call with no path sent the
    /// model looking for an encoding problem.
    #[test]
    fn file_tools_name_a_directory_and_a_missing_path() {
        let root = temp_project();
        let out = read_file(&root, &json!({"path": "src"}));
        assert!(!out.ok, "{}", out.output);
        assert_eq!(out.output, "src is a directory; use list_dir to see its entries");
        let out = read_file(&root, &json!({"path": "."}));
        assert!(!out.ok && out.output.starts_with(". is a directory"), "{}", out.output);

        let out = read_file(&root, &json!({}));
        assert_eq!((out.ok, out.output.as_str()), (false, "missing required argument: path"));
        let out = write_file(&root, &json!({"content": "x\n"}));
        assert_eq!((out.ok, out.output.as_str()), (false, "missing required argument: path"));
        let out = edit_file(&root, &json!({"old_string": "a", "new_string": "b"}));
        assert_eq!((out.ok, out.output.as_str()), (false, "missing required argument: path"));
        // A blank path names no file either; resolved, it reached the
        // project root and produced an error about a nameless directory.
        for blank in ["", "  "] {
            let out = read_file(&root, &json!({"path": blank}));
            assert_eq!((out.ok, out.output.as_str()), (false, "missing required argument: path"), "{blank:?}");
            let out = write_file(&root, &json!({"path": blank, "content": "x\n"}));
            assert_eq!((out.ok, out.output.as_str()), (false, "missing required argument: path"), "{blank:?}");
            let out = edit_file(&root, &json!({"path": blank, "old_string": "a", "new_string": "b"}));
            assert_eq!((out.ok, out.output.as_str()), (false, "missing required argument: path"), "{blank:?}");
        }

        // A file that is not UTF-8 keeps its own message.
        std::fs::write(root.join("latin1.txt"), b"caf\xe9\n").unwrap();
        let out = read_file(&root, &json!({"path": "latin1.txt"}));
        assert_eq!((out.ok, out.output.as_str()), (false, "latin1.txt is not a UTF-8 text file"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// Models often write a number or a boolean as a string. Those
    /// arguments were read as JSON numbers and booleans only, so `"5"` and
    /// `"true"` were dropped without a word and the defaults ran instead.
    #[tokio::test]
    async fn numeric_and_boolean_arguments_written_as_strings_are_honored() {
        let root = temp_project();
        std::fs::write(root.join("three.txt"), "one\ntwo\nthree\n").unwrap();
        let out = read_file(&root, &json!({"path": "three.txt", "offset": "2", "limit": " 1 "}));
        assert!(out.ok, "{}", out.output);
        assert!(out.output.starts_with("    2 two\n"), "offset as a string: {}", out.output);
        assert!(!out.output.contains("three\n"), "limit as a string: {}", out.output);

        std::fs::write(root.join("twice.txt"), "x\nx\n").unwrap();
        let out = edit_file(&root, &json!({"path": "twice.txt", "old_string": "x", "new_string": "y", "replace_all": "false"}));
        assert!(!out.ok && out.output.contains("matches 2 times"), "{}", out.output);
        let out = edit_file(&root, &json!({"path": "twice.txt", "old_string": "x", "new_string": "y", "replace_all": "true"}));
        assert!(out.ok, "{}", out.output);
        assert_eq!(std::fs::read_to_string(root.join("twice.txt")).unwrap(), "y\ny\n");

        let out = bash_tool(
            &root.join("data"),
            &root,
            &json!({"command": "sleep 30", "timeout_secs": "2"}),
            OutputCaps::default(),
            Arc::new(CancelToken::default()),
        )
        .await;
        assert!(!out.ok && out.output.contains("timed out after 2s"), "{}", out.output);

        // The helper's own shape: anything else still falls back.
        assert_eq!(arg(&json!({"n": 7}), "n", Value::as_u64), Some(7));
        assert_eq!(arg(&json!({"n": "True"}), "n", Value::as_bool), Some(true));
        for unusable in [json!({"n": "-1"}), json!({"n": "five"}), json!({"n": null}), json!({})] {
            assert_eq!(arg(&unusable, "n", Value::as_u64), None, "{unusable}");
        }
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
    fn an_exact_edit_preserves_surrounding_indentation() {
        let root = std::env::temp_dir().join(format!("openmax-edit-fuzzy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        std::fs::write(root.join("src.rs"), "fn outer() {\n    fn inner() {\n        old_value\n    }\n}\n").unwrap();
        let out = edit_file(
            &root,
            &json!({
                "path": "src.rs",
                "old_string": "    fn inner() {\n        old_value\n    }",
                "new_string": "    fn inner() {\n        new_value\n    }"
            }),
        );
        assert!(out.ok, "{}", out.output);
        let content = std::fs::read_to_string(root.join("src.rs")).unwrap();
        assert!(content.contains("        new_value\n"), "indent must be preserved: {content:?}");
        assert!(!content.contains("old_value"), "{}", content);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_exact_edit_rejects_ambiguous_matches() {
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
                "old_string": "    fn foo() {\n        a\n    }",
                "new_string": "    fn foo() {\n        b\n    }"
            }),
        );
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("matches 2 times"), "{}", out.output);
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

    /// A character diff is quadratic in line length, so scoring a whole
    /// megabyte line on an edit miss runs for minutes inside a call that
    /// cancellation waits for. Each line is scored on a bounded prefix,
    /// which is also all of a clipped line the model could have copied.
    #[test]
    fn the_closest_match_hint_scores_a_bounded_prefix_of_long_lines() {
        let root = temp_project();
        let table = format!("pub const TABLE: &[u32] = &[{}];", "1234, ".repeat(200_000));
        std::fs::write(root.join("table.rs"), format!("fn main() {{}}\n{table}\n")).unwrap();
        // The start of the long line as read_file shows it, followed by a
        // line the file does not have.
        let copied = &table[..400];
        let out = edit_file(&root, &json!({
            "path": "table.rs", "old_string": format!("{copied}\n// missing"), "new_string": "x"
        }));
        assert!(!out.ok, "{}", out.output);
        assert!(
            out.output.contains("Closest match is at line 2"),
            "a line that starts with the copied text is its closest match however long it runs: {}",
            out.output
        );

        // A miss against a file that is one megabyte line returns, and still
        // names the line.
        let blob = "var mixing=0;".repeat(80_000);
        assert!(hint_text(&blob).len() <= MAX_LINE_CHARS, "the diff sees a bounded prefix");
        std::fs::write(root.join("blob.js"), &blob).unwrap();
        let out = edit_file(&root, &json!({
            "path": "blob.js", "old_string": "function missing() {}", "new_string": "x"
        }));
        assert!(!out.ok, "{}", out.output);
        assert!(out.output.contains("Closest match is at line 1"), "{}", out.output);
        let _ = std::fs::remove_dir_all(root);
    }

    /// Past its deadline the scan stops at the next line once it has compared
    /// one, and the hint names the lines it compared so a partial best is not
    /// read as the file's best.
    #[test]
    fn a_closest_match_scan_past_its_deadline_names_the_lines_it_compared() {
        let content = "fn alpha() {}\nfn beta() {}\nfn almost_match() {}\n";
        let needle = "fn almost_matched() {}";
        let hint = closest_line_hint_until(content, needle, Instant::now() + Duration::from_secs(60));
        assert!(hint.contains("Closest match is at line 3"), "{hint}");
        let expired = Instant::now() - Duration::from_millis(1);
        let hint = closest_line_hint_until(content, needle, expired);
        assert!(hint.contains("Closest match in lines 1-1 is at line 1"), "{hint}");
        assert!(hint.contains("Read the file around that line"), "{hint}");
    }

    /// A line that shares no byte with the copied text is ruled out by its
    /// ceiling without a diff, so the deadline cannot wait for a diff: a scan
    /// in which no line shares a byte (a copied blank first line, non-ASCII
    /// text against an ASCII file) would walk the whole file past it.
    #[test]
    fn a_closest_match_scan_stops_at_its_deadline_when_no_line_shares_a_byte() {
        let content: String = (0..1_000).map(|i| format!("fn f{i}() {{}}\n")).collect();
        let expired = Instant::now() - Duration::from_millis(1);
        for needle in ["   \nfn missing() {}", "日本語"] {
            let hint = closest_line_hint_until(&content, needle, expired);
            assert!(hint.contains("Closest match in lines 1-1 is at line 1"), "{needle:?}: {hint}");
        }
    }

    /// The scan skips a line's diff when its ceiling cannot beat the best
    /// score; that is only sound while the ceiling never undercuts the score.
    #[test]
    fn the_similarity_ceiling_never_undercuts_the_score() {
        let long = format!("let table = [{}];", "7, ".repeat(400));
        for (line, needle) in [
            ("fn alpha() {}", "fn almost_matched() {}"),
            ("", ""),
            ("   ", "x"),
            ("é è ë", "è é"),
            ("naïve café", "naive cafe"),
            ("    let x = 1;", "let x = 2;"),
            (long.as_str(), "let table = [7, 7, 8];"),
        ] {
            let key = hint_text(needle);
            let ceiling = similarity_ceiling(hint_text(line), key, &byte_counts(key));
            let score = line_similarity(line, needle);
            assert!(ceiling >= score, "{line:?} vs {needle:?}: ceiling {ceiling} under score {score}");
        }
    }

    /// A full rewrite shares no lines with the file it replaces, so the line
    /// diff for the UI is quadratic in the file's length and a mutating
    /// call cannot be cancelled while it runs. A diff cut off at its deadline
    /// can count unchanged lines as rewritten, so the model is told its
    /// counts are an upper bound rather than the size of the edit, in a
    /// summary whose counts a replayed session can still read.
    #[test]
    fn a_full_rewrite_of_a_large_file_reports_a_bounded_diff() {
        let root = temp_project();
        std::fs::write(root.join("small.txt"), "one\ntwo\nthree\n").unwrap();
        let out = write_file(&root, &json!({"path": "small.txt", "content": "one\nTWO\nthree\n"}));
        assert_eq!(out.output, "wrote small.txt (+1 −1)", "a finished diff reports exact counts");
        let diff = out.diff.expect("a write reports its diff");
        assert_eq!((diff.added, diff.removed), (1, 1), "{}", diff.diff);
        assert!(diff.diff.starts_with("--- a/small.txt\n+++ b/small.txt\n"), "{}", diff.diff);
        assert!(diff.diff.contains("-two\n+TWO\n"), "an ordinary edit keeps its full diff: {}", diff.diff);

        let lines = 40_000;
        let old: String = (0..lines).map(|i| format!("old line {i}\n")).collect();
        let new: String = (0..lines).map(|i| format!("new line {i}\n")).collect();
        std::fs::write(root.join("big.txt"), old).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let dir = root.clone();
        std::thread::spawn(move || {
            let _ = tx.send(write_file(&dir, &json!({"path": "big.txt", "content": new})));
        });
        // A guard far above the bounded diff's cost, so a regression fails
        // here instead of hanging the suite.
        let out = rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the diff for a full rewrite must stop at its deadline");
        assert!(out.ok, "{}", out.output);
        // The "(+N −M)" group stays intact ahead of the caveat: session
        // replay reads the counts back from it to badge the edit card.
        assert_eq!(
            out.output,
            format!("wrote big.txt (+{lines} −{lines}) · counts are an upper bound"),
            "a diff cut off at its deadline reports its counts as an upper bound"
        );
        let diff = out.diff.expect("a write reports its diff");
        assert_eq!((diff.added, diff.removed), (lines, lines), "{}", out.output);
        assert!(diff.diff.contains("diff not shown"), "{}", diff.diff);
        assert!(diff.diff.len() < 300, "the fallback is a summary, not a listing: {}", diff.diff);
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
