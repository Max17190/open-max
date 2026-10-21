//! Native child-process supervision shared by agent-dispatched commands.
//!
//! By default this module deliberately preserves the host environment: it
//! owns lifecycle and bounded capture, and the session's tools, hooks, and
//! bash all run with `sandbox: None`. The one exception is an explicit
//! [`SandboxPolicy`] on a request - probe runs of unapproved tool code -
//! which is argv surgery plus an env scrub, still the same lifecycle.

use std::{
    ffi::OsString,
    io,
    path::{Path, PathBuf},
    process::ExitStatus,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
};

use crate::state::CancelToken;

const READ_CHUNK_BYTES: usize = 8 * 1024;
const TERMINATION_GRACE: Duration = Duration::from_millis(250);
/// How often a group given [`TERMINATION_GRACE`] is checked for members
/// still running.
const GROUP_EXIT_POLL: Duration = Duration::from_millis(2);
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_secs(1);
/// The longest a cancelled command takes to settle: a SIGTERM grace for its
/// shell and one for the rest of its group, one for a stdin writer the
/// command never drained, then both output drain windows for a descendant
/// that left the group still holding a pipe.
pub const CANCEL_SETTLE: Duration = TERMINATION_GRACE
    .saturating_mul(4)
    .saturating_add(OUTPUT_DRAIN_GRACE);
/// How long a spilled command log stays on disk. Long enough that a resumed
/// session can still tail a log its transcript points at; short enough that
/// the directory stays bounded (unpruned, one machine reached 95 MB in a
/// month).
const SPILL_LOG_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// The timeout a bash call or an external tool asks for in whole seconds,
/// which is what the bash schema and tool manifests count in. In tests a
/// second lasts what `short_seconds` set for the thread: a real second,
/// unless a test that waits a timeout out asked for less.
pub(crate) fn timeout_secs(secs: u64) -> Duration {
    #[cfg(not(test))]
    let second = Duration::from_secs(1);
    #[cfg(test)]
    let second = TEST_SECOND.with(std::cell::Cell::get);
    second.saturating_mul(u32::try_from(secs).unwrap_or(u32::MAX))
}

#[cfg(test)]
thread_local! {
    static TEST_SECOND: std::cell::Cell<Duration> = const { std::cell::Cell::new(Duration::from_secs(1)) };
}

/// The lengths a test that waits a timeout out tries for one second of it,
/// shortest first. A command the timeout stopped before it got as far as the
/// test needs (a loaded runner can take longer than the short ones to start
/// a shell) is run again with the next, up to a real second.
#[cfg(test)]
pub(crate) const TRIAL_SECONDS: [Duration; 4] =
    [Duration::from_millis(10), Duration::from_millis(50), Duration::from_millis(250), Duration::from_secs(1)];

/// Makes [`timeout_secs`] count `second` per second on this thread until
/// the guard drops. A `#[tokio::test]` runs on its own thread, so only that
/// test's timeouts change.
#[cfg(test)]
pub(crate) fn short_seconds(second: Duration) -> ShortSeconds {
    ShortSeconds(TEST_SECOND.with(|cell| cell.replace(second)))
}

/// Restores the second [`short_seconds`] replaced.
#[cfg(test)]
pub(crate) struct ShortSeconds(Duration);

#[cfg(test)]
impl Drop for ShortSeconds {
    fn drop(&mut self) {
        TEST_SECOND.with(|cell| cell.set(self.0));
    }
}

/// A single native process invocation. The caller supplies exact argv and cwd;
/// no shell interpretation is introduced by the supervisor.
pub(crate) struct ProcessRequest {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
    pub stdin: StdinMode,
    pub timeout: Duration,
    pub capture: CaptureSpec,
    /// OS-level containment for this one spawn. `None` is the default and
    /// preserves the module's stated posture exactly (host environment,
    /// no sandbox policy). `Some` exists for probe runs of code no human
    /// has approved yet: see [`SandboxPolicy`].
    pub sandbox: Option<SandboxPolicy>,
    /// Env var NAMES forwarded from the parent environment. `None` = the
    /// full host environment (bash, hooks: the user's shell) with git's
    /// terminal prompt turned off. `Some(names)`
    /// = scrub, then a baseline (PATH, HOME, LANG, TERM from the parent)
    /// plus exactly the named variables - external tools, whose manifest
    /// declares the list, making the credential grant part of the bytes a
    /// human approves. Ignored under `sandbox`: a probe stays fully
    /// scrubbed whatever its manifest asks for.
    pub env_allowlist: Option<Vec<String>>,
    /// Put the self link, a bare `openmax` naming the running harness, first
    /// on the child's PATH. Only the agent's shell sets it. The agent's
    /// account can write that directory, so on a hook's or tool's PATH a
    /// name planted there while it runs would replace the system binary the
    /// human approved, and the approval binding hashes no code outside the
    /// project. Ignored under `sandbox`.
    pub self_link: bool,
    /// Variables set for this child on top of the environment the fields
    /// above give it, in every mode. The agent's shell takes its command text
    /// this way, so the text never sits in the shell's argument list.
    pub env: Vec<(OsString, OsString)>,
}

/// Containment for a probe run: no network, filesystem reads allowed, writes
/// confined to `rw_scratch`, environment scrubbed to a minimal set with
/// `HOME` pointed at the scratch dir. This is not a general sandbox and
/// deliberately not applied to bash, hooks, or approved tools - it exists so
/// an agent can iterate on a tool it just wrote (and a human can see "the
/// example passed in a sandbox") before anyone grants the tool real host
/// authority. Reads are not cut because the agent already holds unconfined
/// bash: the marginal authority a probe must not have is the network (the
/// probe exfiltrating on its own) and mutation outside its scratch.
///
/// Backends: macOS `sandbox-exec` (deny network* + deny file-write* outside
/// scratch, on an allow-default base - the shape Bazel and Nix ride), Linux
/// bubblewrap (`--unshare-all`, read-only root bind, scratch bound
/// read-write). No backend, or a backend that cannot be probed alive, is
/// [`ProcessError::SandboxUnavailable`]: callers fail closed to their
/// pre-sandbox refusal, never silently unsandboxed.
#[derive(Clone, Debug)]
pub(crate) struct SandboxPolicy {
    /// The one directory the probe may write; also its HOME. Reads are
    /// broadly allowed (see above); the cwd and inputs live in the project
    /// root the caller already passes to the spawn.
    pub rw_scratch: PathBuf,
}

pub(crate) enum StdinMode {
    Null,
    Bytes(Vec<u8>),
}

impl StdinMode {
    /// A JSON payload delivered as one newline-terminated line. JSON parsers
    /// ignore the trailing newline; line-oriented consumers require it - a
    /// script reading with `read -r` under `set -e` hits EOF on an
    /// unterminated stream, exits nonzero, and (as a gate hook) blocks every
    /// call it guards even though the payload was fully delivered.
    pub(crate) fn json_line(value: &serde_json::Value) -> Self {
        let mut bytes = value.to_string().into_bytes();
        bytes.push(b'\n');
        Self::Bytes(bytes)
    }
}

/// Bounded in-memory capture plus an optional, bounded raw-output log.
#[derive(Clone)]
pub(crate) struct CaptureSpec {
    pub head_bytes: usize,
    pub tail_bytes: usize,
    pub spill_dir: Option<PathBuf>,
    pub spill_bytes_per_stream: usize,
    /// Keep only the final text of a line redrawn with carriage returns
    /// (see [`Redraws`]). Applied as bytes arrive, before the head,
    /// the tail, the spill log and `total_bytes` see them, so a progress bar
    /// costs the cap one line instead of every frame it drew.
    pub redraws: bool,
}

pub(crate) enum Termination {
    Exited(ExitStatus),
    Cancelled,
    TimedOut,
}

pub(crate) struct CapturedStream {
    pub total_bytes: u64,
    pub head: Vec<u8>,
    pub tail: Vec<u8>,
}

impl CapturedStream {
    /// Reconstruct small output exactly. Larger output has an explicit gap
    /// between its bounded prefix and suffix.
    pub(crate) fn rendered_bytes(&self) -> Vec<u8> {
        let total = usize::try_from(self.total_bytes).unwrap_or(usize::MAX);
        let retained = self.head.len().saturating_add(self.tail.len());
        if total <= retained {
            if total <= self.head.len() {
                return self.head[..total].to_vec();
            }
            if total <= self.tail.len() {
                return self.tail[self.tail.len() - total..].to_vec();
            }
            let overlap = retained - total;
            let mut rendered = self.head.clone();
            rendered.extend_from_slice(&self.tail[overlap..]);
            return rendered;
        }
        if self.head.is_empty() {
            return self.tail.clone();
        }
        if self.tail.is_empty() {
            return self.head.clone();
        }
        let mut rendered = self.head.clone();
        rendered.extend_from_slice(b"\n[openmax: output truncated]\n");
        rendered.extend_from_slice(&self.tail);
        rendered
    }
}

pub(crate) struct ProcessOutput {
    pub termination: Termination,
    pub stdout: CapturedStream,
    pub stderr: CapturedStream,
    /// Combined raw log, stdout followed by a labeled stderr section.
    pub log_path: Option<PathBuf>,
    /// True when either stream exceeded its configured spill cap.
    pub log_truncated: bool,
    /// True when the command exited normally but left process-group members
    /// behind that this call then terminated. Only a NORMAL exit sets it: a
    /// cancel or a timeout kills the group by design and already says so.
    /// Callers must surface it, because the caller who backgrounded a server
    /// otherwise sees a clean exit and no server.
    pub background_terminated: bool,
}

#[derive(Debug)]
pub(crate) enum ProcessError {
    Spawn(io::Error),
    Wait(io::Error),
    /// A sandboxed run was requested and no working backend exists on this
    /// host. Distinct so callers can fall back to their pre-sandbox refusal:
    /// the one wrong answer is running the probe unsandboxed anyway.
    SandboxUnavailable(String),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(error) => write!(f, "failed to spawn process: {error}"),
            Self::Wait(error) => write!(f, "failed while supervising process: {error}"),
            Self::SandboxUnavailable(reason) => {
                write!(f, "no sandbox backend available on this host: {reason}")
            }
        }
    }
}

impl std::error::Error for ProcessError {}

struct StreamCapture {
    stream: CapturedStream,
    /// Set when the capture keeps only the final text of a redrawn line,
    /// with the buffer each chunk's kept bytes pass through.
    redraws: Option<(Redraws, Vec<u8>)>,
    prefix: Vec<u8>,
    spill_path: Option<PathBuf>,
    spill_file: Option<tokio::fs::File>,
    written_to_spill: usize,
    omitted_from_spill: u64,
    spill_disabled: bool,
    capture: CaptureSpec,
    stream_name: &'static str,
}

impl StreamCapture {
    fn new(capture: CaptureSpec, stream_name: &'static str) -> Self {
        Self {
            stream: CapturedStream {
                total_bytes: 0,
                head: Vec::with_capacity(capture.head_bytes),
                tail: Vec::with_capacity(capture.tail_bytes),
            },
            redraws: capture.redraws.then(Default::default),
            prefix: Vec::with_capacity(capture.head_bytes.max(capture.tail_bytes)),
            spill_path: None,
            spill_file: None,
            written_to_spill: 0,
            omitted_from_spill: 0,
            spill_disabled: false,
            capture,
            stream_name,
        }
    }

    fn threshold(&self) -> usize {
        self.capture
            .head_bytes
            .saturating_add(self.capture.tail_bytes)
    }

    async fn push(&mut self, chunk: &[u8]) {
        let Some((mut redraws, mut shown)) = self.redraws.take() else {
            return self.keep(chunk).await;
        };
        shown.clear();
        redraws.push(chunk, &mut shown);
        self.keep(&shown).await;
        self.redraws = Some((redraws, shown));
    }

    async fn keep(&mut self, chunk: &[u8]) {
        self.stream.total_bytes = self.stream.total_bytes.saturating_add(chunk.len() as u64);

        let head_remaining = self
            .capture
            .head_bytes
            .saturating_sub(self.stream.head.len());
        self.stream
            .head
            .extend_from_slice(&chunk[..chunk.len().min(head_remaining)]);
        push_tail(&mut self.stream.tail, self.capture.tail_bytes, chunk);

        if self.spill_file.is_none() {
            let available = self.threshold().saturating_sub(self.prefix.len());
            self.prefix
                .extend_from_slice(&chunk[..chunk.len().min(available)]);
            if self.stream.total_bytes <= self.threshold() as u64 {
                return;
            }
            self.start_spill().await;
        }

        // `prefix` was written when spilling started. Only append the portion
        // of this chunk that was not already retained in it.
        let already_buffered = self
            .prefix
            .len()
            .saturating_sub(self.stream.total_bytes.saturating_sub(chunk.len() as u64) as usize);
        if already_buffered < chunk.len() {
            self.write_spill(&chunk[already_buffered..]).await;
        }
    }

    async fn start_spill(&mut self) {
        if self.spill_disabled {
            return;
        }
        let Some(dir) = self.capture.spill_dir.as_ref() else {
            self.spill_disabled = true;
            return;
        };
        if tokio::fs::create_dir_all(dir).await.is_err() {
            self.spill_disabled = true;
            return;
        }
        let path = dir.join(format!(
            ".openmax-{}-{}.tmp",
            self.stream_name,
            uuid::Uuid::new_v4()
        ));
        let Ok(mut file) = tokio::fs::File::create(&path).await else {
            self.spill_disabled = true;
            return;
        };
        let prefix = self.prefix.clone();
        if self.write_spill_to(&mut file, &prefix).await.is_err() {
            let _ = tokio::fs::remove_file(&path).await;
            self.spill_disabled = true;
            return;
        }
        self.spill_path = Some(path);
        self.spill_file = Some(file);
    }

    async fn write_spill(&mut self, bytes: &[u8]) {
        if let Some(mut file) = self.spill_file.take() {
            if self.write_spill_to(&mut file, bytes).await.is_ok() {
                self.spill_file = Some(file);
            } else {
                if let Some(path) = self.spill_path.take() {
                    let _ = tokio::fs::remove_file(path).await;
                }
                self.spill_disabled = true;
            }
        }
    }

    async fn write_spill_to(&mut self, file: &mut tokio::fs::File, bytes: &[u8]) -> io::Result<()> {
        let available = self
            .capture
            .spill_bytes_per_stream
            .saturating_sub(self.written_to_spill);
        let kept = bytes.len().min(available);
        if kept > 0 {
            file.write_all(&bytes[..kept]).await?;
            self.written_to_spill += kept;
        }
        self.omitted_from_spill = self
            .omitted_from_spill
            .saturating_add((bytes.len().saturating_sub(kept)) as u64);
        Ok(())
    }

    async fn finish(mut self) -> io::Result<FinishedStream> {
        if let Some((redraws, _)) = self.redraws.take() {
            self.keep(&redraws.finish()).await;
        }
        if let Some(mut file) = self.spill_file.take() {
            if self.omitted_from_spill > 0 {
                let _ = file
                    .write_all(
                        format!(
                            "\n[openmax: {} bytes omitted from {} output log]\n",
                            self.omitted_from_spill, self.stream_name
                        )
                        .as_bytes(),
                    )
                    .await;
            }
            let _ = file.flush().await;
        }
        Ok(FinishedStream {
            stream: self.stream,
            spill_path: self.spill_path,
            omitted: self.omitted_from_spill > 0,
        })
    }
}

struct FinishedStream {
    stream: CapturedStream,
    spill_path: Option<PathBuf>,
    omitted: bool,
}

/// Longest stretch of one line [`Redraws`] holds back in case a later `\r`
/// replaces it. A real terminal wraps a line this long over many rows and a
/// `\r` returns only to the start of the last one, so once a line's text
/// since its last redraw passes it, that text and the rest of the line up to
/// its next `\r` or `\n` are passed on as they arrive: memory stays bounded,
/// a later redraw never replaces what was passed on, and where the pipe's
/// reads end changes nothing. The `\r` that ends such text is passed on
/// with it, so the frame drawn next does not run on from it.
const REDRAW_LINE_BYTES: usize = 8 * 1024;

/// The final text of a line that output redraws: a `\r` not followed by
/// `\n` starts the line over, and the text after it replaces the whole line
/// so far. Package managers, build tools and downloaders draw progress this
/// way, and every frame they drew reached the model although a person at a
/// terminal sees only the last. A terminal overwrites in place, so a shorter
/// frame leaves the end of a longer one on screen; that leftover is dropped
/// too. A line is held until its `\n` (or the end of the stream) so a later
/// `\r` can still replace it; `\r\n` stays a line ending, byte for byte, a
/// run of `\r` before a `\n` becomes one `\r\n`, and a `\r` with nothing
/// after it on its line replaces nothing, so one at the end of the stream is
/// dropped (`last\r` ends as `last`).
#[derive(Default)]
struct Redraws {
    /// The current line's text since its last redraw.
    line: Vec<u8>,
    /// A `\r` came after the line's last text, and nothing yet after it.
    returned: bool,
    /// The line's text since its last redraw passed [`REDRAW_LINE_BYTES`]
    /// and was passed on. The rest of it passes straight on until its next
    /// `\r` or `\n`, so no later `\r` replaces part of a line already
    /// passed on, whatever the reads. It stays set across that `\r`, so the
    /// `\r` is passed on ahead of the next frame's text.
    passing: bool,
}

impl Redraws {
    fn push(&mut self, mut rest: &[u8], out: &mut Vec<u8>) {
        // Most output redraws nothing: its whole lines pass on in one copy
        // instead of a walk over every line ending.
        if !self.returned && !rest.contains(&b'\r') {
            if let Some(end) = rest.iter().rposition(|&b| b == b'\n') {
                out.append(&mut self.line);
                out.extend_from_slice(&rest[..=end]);
                self.passing = false;
                rest = &rest[end + 1..];
            }
            return self.text(rest, out);
        }
        let mut start = 0;
        for (at, &byte) in rest.iter().enumerate() {
            if byte != b'\r' && byte != b'\n' {
                continue;
            }
            self.text(&rest[start..at], out);
            if byte == b'\n' {
                out.append(&mut self.line);
                if std::mem::take(&mut self.returned) {
                    out.push(b'\r');
                }
                out.push(b'\n');
                self.passing = false;
            } else {
                self.returned = true;
            }
            start = at + 1;
        }
        self.text(&rest[start..], out);
    }

    fn text(&mut self, text: &[u8], out: &mut Vec<u8>) {
        if text.is_empty() {
            return;
        }
        if std::mem::take(&mut self.returned) {
            self.line.clear();
            if std::mem::take(&mut self.passing) {
                out.push(b'\r');
            }
        }
        if self.passing || self.line.len() + text.len() > REDRAW_LINE_BYTES {
            out.append(&mut self.line);
            out.extend_from_slice(text);
            self.passing = true;
        } else {
            self.line.extend_from_slice(text);
        }
    }

    /// The end of the stream ends the line it was drawing: what is left to
    /// pass on.
    fn finish(self) -> Vec<u8> {
        self.line
    }
}

fn push_tail(tail: &mut Vec<u8>, limit: usize, bytes: &[u8]) {
    if limit == 0 {
        return;
    }
    if bytes.len() >= limit {
        tail.clear();
        tail.extend_from_slice(&bytes[bytes.len() - limit..]);
        return;
    }
    let excess = tail.len().saturating_add(bytes.len()).saturating_sub(limit);
    if excess > 0 {
        tail.drain(..excess);
    }
    tail.extend_from_slice(bytes);
}

async fn drain_stream<R>(
    reader: R,
    capture: CaptureSpec,
    name: &'static str,
    stop: Arc<CancelToken>,
) -> io::Result<FinishedStream>
where
    R: AsyncRead + Unpin,
{
    let mut reader = reader;
    let mut buffered = StreamCapture::new(capture, name);
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    loop {
        tokio::select! {
            count = reader.read(&mut chunk) => {
                let count = count?;
                if count == 0 {
                    return buffered.finish().await;
                }
                buffered.push(&chunk[..count]).await;
            }
            _ = stop.cancelled() => {
                return buffered.finish().await;
            }
        }
    }
}

/// Rewrite `(program, args)` to run under this host's sandbox backend, or
/// say why no containment is possible. Pure argv surgery: the supervisor's
/// lifecycle, capture, and process-group handling see one program like any
/// other (the wrapper is the group leader, so cancel and timeout kill the
/// whole tree unchanged).
fn apply_sandbox(
    program: &OsString,
    args: &[OsString],
    policy: &SandboxPolicy,
) -> Result<(OsString, Vec<OsString>), ProcessError> {
    #[cfg(target_os = "macos")]
    {
        let exec = Path::new("/usr/bin/sandbox-exec");
        if !exec.exists() {
            return Err(ProcessError::SandboxUnavailable(
                "/usr/bin/sandbox-exec is missing".into(),
            ));
        }
        // Allow-default with explicit cuts: network entirely, writes outside
        // the scratch (plus the null device). The kernel evaluates canonical
        // paths, and /var and /tmp are symlinks into /private on macOS, so
        // the subpath rule must name the canonical spelling or a scratch
        // under the temp dir denies its own writes. Seatbelt profile strings
        // quote paths; a quote inside a path cannot be expressed safely, so
        // refuse.
        let canonical = policy
            .rw_scratch
            .canonicalize()
            .unwrap_or_else(|_| policy.rw_scratch.clone());
        let scratch = canonical.to_string_lossy();
        if scratch.contains('"') {
            return Err(ProcessError::SandboxUnavailable(
                "scratch path contains a quote, unrepresentable in a sandbox profile".into(),
            ));
        }
        let profile = format!(
            "(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n\
             (allow file-write* (subpath \"{scratch}\"))\n\
             (allow file-write-data (literal \"/dev/null\"))\n\
             (allow file-write-data (literal \"/dev/dtracehelper\"))\n"
        );
        let mut wrapped: Vec<OsString> = vec!["-p".into(), profile.into(), program.clone()];
        wrapped.extend(args.iter().cloned());
        Ok((exec.as_os_str().to_os_string(), wrapped))
    }
    #[cfg(target_os = "linux")]
    {
        let Some(bwrap) = ["/usr/bin/bwrap", "/usr/local/bin/bwrap", "/bin/bwrap"]
            .iter()
            .map(Path::new)
            .find(|p| p.exists())
        else {
            return Err(ProcessError::SandboxUnavailable(
                "bubblewrap (bwrap) is not installed".into(),
            ));
        };
        // Read-only root, scratch bound writable, all namespaces unshared
        // (which cuts the network). If user namespaces are disabled the run
        // fails at spawn with bwrap's own diagnostic - a loud refusal, not a
        // silent unsandboxed run.
        let mut wrapped: Vec<OsString> = vec![
            "--die-with-parent".into(),
            "--unshare-all".into(),
            "--ro-bind".into(),
            "/".into(),
            "/".into(),
            "--dev".into(),
            "/dev".into(),
            "--proc".into(),
            "/proc".into(),
            "--tmpfs".into(),
            "/tmp".into(),
            "--bind".into(),
            policy.rw_scratch.as_os_str().to_os_string(),
            policy.rw_scratch.as_os_str().to_os_string(),
            "--".into(),
            program.clone(),
        ];
        wrapped.extend(args.iter().cloned());
        Ok((bwrap.as_os_str().to_os_string(), wrapped))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (program, args, policy);
        Err(ProcessError::SandboxUnavailable(
            "no sandbox backend for this platform".into(),
        ))
    }
}

/// The directory whose one entry, `openmax`, links to the executable running
/// this harness; the agent's shell gets it first on PATH. None leaves PATH as
/// inherited, which is never a wrong link.
///
/// Lifecycle: the name derives from the account and the canonical executable
/// path, so whichever process writes the directory writes the same link.
/// Concurrent harnesses on one build share it, different builds never touch
/// each other's, and nothing needs removing on exit: a crash leaves nothing
/// a later run could misread, and the OS temp cleaner reaps the directories
/// of builds no longer run. The directory is checked on every spawn: the
/// link is renamed back into place when something removed or replaced it,
/// any other entry is removed, and a link whose binary has since been
/// deleted keeps the directory off PATH. A build installed over this one at
/// the same path is what the link then runs, as `$OPENMAX_BIN` names it: the
/// replaced bytes have no path left to link to, and dropping the link would
/// hand a bare `openmax` back to whichever install comes first on the
/// inherited PATH, possibly an older one.
#[cfg(unix)]
fn self_link_dir() -> Option<PathBuf> {
    static TARGET: std::sync::OnceLock<Option<(PathBuf, PathBuf)>> = std::sync::OnceLock::new();
    let (exe, dir) = TARGET
        .get_or_init(|| {
            // Canonical, so a harness started through the link lands on the
            // same directory instead of linking to a link.
            let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
            let digest = crate::ledger::sha256_hex(exe.as_os_str().as_encoded_bytes());
            let uid = unsafe { libc::geteuid() };
            // A per-account runtime directory, where one exists, so another
            // account cannot take the name first the way it can in /tmp.
            let base = std::env::var_os("XDG_RUNTIME_DIR")
                .map(PathBuf::from)
                .filter(|dir| dir.is_absolute() && dir.is_dir())
                .unwrap_or_else(std::env::temp_dir);
            let dir = base.join(format!("openmax-{uid}-{}", &digest[..16]));
            Some((exe, dir))
        })
        .as_ref()?;
    ensure_self_link(dir, exe).ok()?;
    Some(dir.clone())
}

#[cfg(not(unix))]
fn self_link_dir() -> Option<PathBuf> {
    None
}

#[cfg(unix)]
fn ensure_self_link(dir: &Path, exe: &Path) -> io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    if let Err(error) = std::fs::DirBuilder::new().mode(0o700).create(dir) {
        if error.kind() != io::ErrorKind::AlreadyExists {
            return Err(error);
        }
    }
    // The temp dir can be shared between accounts (/tmp on Linux), and an
    // entry first on PATH runs under every bare `openmax`: only a directory
    // this account owns and nobody else can write may hold it, or another
    // account could create the name first and swap the link mid-session.
    let meta = std::fs::symlink_metadata(dir)?;
    if !meta.is_dir() || meta.uid() != unsafe { libc::geteuid() } || meta.mode() & 0o022 != 0 {
        return Err(io::Error::other("self link directory is not private to this account"));
    }
    // The agent's own account can write here too, and this directory comes
    // first on the PATH of every bash command: a `python3` planted beside the
    // link by one command would replace the system binary in every later
    // one, including a command a human read and approved. Only the link, and
    // another harness's staged link mid-rename, may stay; anything that
    // cannot be removed keeps the directory off PATH.
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "openmax" || is_staged_link(&name) {
            continue;
        }
        let removed = match entry.file_type() {
            Ok(kind) if kind.is_dir() => std::fs::remove_dir_all(entry.path()),
            _ => std::fs::remove_file(entry.path()),
        };
        match removed {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
    }
    let link = dir.join("openmax");
    if std::fs::read_link(&link).ok().as_deref() != Some(exe) {
        // Staged, then renamed over: a concurrent harness on the same build
        // never sees the name missing.
        let staged = dir.join(format!(".openmax-{}", uuid::Uuid::new_v4()));
        std::os::unix::fs::symlink(exe, &staged)?;
        if let Err(error) = std::fs::rename(&staged, &link) {
            let _ = std::fs::remove_file(&staged);
            return Err(error);
        }
    }
    // Follows the link: a binary deleted since (an uninstall, a cleaned
    // target dir) must not leave a dangling entry first on PATH.
    std::fs::metadata(&link).map(|_| ())
}

/// The name `ensure_self_link` stages a link under before renaming it into
/// place; no command is ever run by such a name.
#[cfg(unix)]
fn is_staged_link(name: &std::ffi::OsStr) -> bool {
    name.to_str()
        .and_then(|name| name.strip_prefix(".openmax-"))
        .is_some_and(|id| uuid::Uuid::parse_str(id).is_ok())
}

/// The PATH the agent's shell gets: the self link directory first, then the
/// harness's own entries. None when the harness has no PATH: a PATH holding
/// only the link would hide every system binary from the shell, which falls
/// back to its own default search path only while PATH is unset.
fn child_path() -> Option<OsString> {
    let inherited = std::env::var_os("PATH")?;
    self_link_dir()
        .and_then(|dir| path_with_self_link(&dir, &inherited))
        .or(Some(inherited))
}

/// The child's PATH: the self link directory first, then the inherited
/// entries without it, so a harness started from a harness's child does not
/// stack another copy per level.
fn path_with_self_link(dir: &Path, inherited: &std::ffi::OsStr) -> Option<OsString> {
    let rest = std::env::split_paths(inherited).filter(|entry| entry != dir);
    std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(rest)).ok()
}

/// Execute one native process with concurrent bounded output capture.
pub(crate) async fn run_process(
    request: ProcessRequest,
    cancel: Arc<CancelToken>,
) -> Result<ProcessOutput, ProcessError> {
    let (program, args) = match &request.sandbox {
        None => (request.program.clone(), request.args.clone()),
        Some(policy) => apply_sandbox(&request.program, &request.args, policy)?,
    };
    let mut command = Command::new(&program);
    command
        .args(&args)
        .current_dir(&request.cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    if let Some(policy) = &request.sandbox {
        // A probe holds no ambient secrets: scrubbed environment, HOME at
        // the scratch. PATH keeps the system directories so interpreters
        // resolve; project-local scripts are reachable via cwd as usual.
        // Deliberately ignores env_allowlist: unapproved code gets nothing.
        command.env_clear();
        command.env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin");
        command.env("HOME", &policy.rw_scratch);
        // Temp files land in the scratch too: without TMPDIR a shell writes
        // heredocs and process substitutions under /tmp, which the sandbox
        // denies - a false negative on correct code, which costs a working
        // script a rewrite to appease a sandbox production never has.
        command.env("TMPDIR", &policy.rw_scratch);
        command.env("TERM", "dumb");
        command.env("LANG", "C.UTF-8");
    } else if let Some(names) = &request.env_allowlist {
        // The manifest-declared grant: baseline from the parent so
        // interpreters and tools behave, plus exactly the named variables.
        // Everything else - API keys included - stays with the harness.
        command.env_clear();
        for baseline in ["PATH", "HOME", "LANG", "TERM"] {
            if let Some(value) = std::env::var_os(baseline) {
                command.env(baseline, value);
            }
        }
        for name in names {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
    } else {
        // Bash and hooks keep the user's whole environment, but git must not
        // prompt for credentials: the child has no terminal to answer on
        // (see `configure_process_group`). Turned off, git fails at once and
        // names the cause. A credential helper or askpass program still
        // answers first.
        command.env("GIT_TERMINAL_PROMPT", "0");
    }
    for (name, value) in &request.env {
        command.env(name, value);
    }
    // Mark every native child as agent-spawned. Trust grants are human
    // actions: the CLI refuses --trust-project (and the interactive trust
    // prompt) when this marker is present, so an agent cannot launder a
    // trust grant through a child process it starts. Applied after any
    // env_clear so the marker survives the scrub.
    command.env("OPENMAX_SESSION", "1");
    // The human attestation must never reach an agent-spawned child: a
    // session launched under an attested shell (CI, test automation) would
    // otherwise hand every bash call a ready-made bypass - unset the marker,
    // inherit the attestation, grant authority. Stripped unconditionally;
    // no child of the harness is the human.
    command.env_remove("OPENMAX_HUMAN_ATTEST");
    // Point the child at the binary that is running this session. The
    // prompt, receipts, and every spec tell the agent to shell out to a bare
    // `openmax`, and the first one on the inherited PATH can be an older
    // build than the harness hosting the session, teaching claims that build
    // has since retracted - both print the same version string. So in the
    // agent's shell a bare `openmax` resolves through the self link first;
    // hooks and tools keep the inherited PATH (see `self_link`), and
    // `$OPENMAX_BIN` names the same build in every one of them. Not applied
    // under a sandbox: a probe gets no handle to the harness.
    if request.sandbox.is_none() {
        if request.self_link {
            if let Some(path) = child_path() {
                command.env("PATH", path);
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            command.env("OPENMAX_BIN", exe);
        }
    }
    match request.stdin {
        StdinMode::Null => {
            command.stdin(std::process::Stdio::null());
        }
        StdinMode::Bytes(_) => {
            command.stdin(std::process::Stdio::piped());
        }
    }
    // A sandboxed probe keeps the group it has always had: it starts only
    // from `openmax --check --run-examples`, never from a session the TUI
    // draws.
    configure_process_group(&mut command, request.sandbox.is_none());
    #[cfg(all(test, unix))]
    hold_before_exec(&mut command);

    let (mut child, group) = GroupGuard::spawn(&mut command)?;
    let pid = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ProcessError::Wait(io::Error::other("stdout pipe unavailable")))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ProcessError::Wait(io::Error::other("stderr pipe unavailable")))?;

    let drain_stop = Arc::new(CancelToken::default());
    let stdout_task = tokio::spawn(drain_stream(
        stdout,
        request.capture.clone(),
        "stdout",
        drain_stop.clone(),
    ));
    let stderr_task = tokio::spawn(drain_stream(
        stderr,
        request.capture.clone(),
        "stderr",
        drain_stop.clone(),
    ));
    let stdin_task = match request.stdin {
        StdinMode::Null => None,
        StdinMode::Bytes(bytes) => child.stdin.take().map(|mut stdin| {
            tokio::spawn(async move {
                // Closing stdin is automatic when this task ends. A process may
                // legitimately exit before consuming its input.
                match stdin.write_all(&bytes).await {
                    Ok(()) => Ok(()),
                    Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
                    Err(error) => Err(error),
                }
            })
        }),
    };

    let (termination, background_terminated) =
        supervise_child(&mut child, pid, request.timeout, &cancel).await?;
    group.settled();

    if let Some(task) = stdin_task {
        finish_stdin_task(task).await;
    }
    let (stdout, stderr) = join_streams(stdout_task, stderr_task, drain_stop).await?;
    let log_truncated = stdout.omitted || stderr.omitted;
    let retained_limit = request
        .capture
        .head_bytes
        .saturating_add(request.capture.tail_bytes) as u64;
    let combined_total = stdout
        .stream
        .total_bytes
        .saturating_add(stderr.stream.total_bytes);
    let force_combined_log = stdout.spill_path.is_none()
        && stderr.spill_path.is_none()
        && stdout.stream.total_bytes <= retained_limit
        && stderr.stream.total_bytes <= retained_limit
        && combined_total > retained_limit;
    let log_path = combine_logs(
        &request.capture.spill_dir,
        stdout.spill_path,
        stderr.spill_path,
        &stdout.stream,
        &stderr.stream,
        force_combined_log,
    )
    .await?;

    Ok(ProcessOutput {
        termination,
        stdout: stdout.stream,
        stderr: stderr.stream,
        log_path,
        log_truncated,
        background_terminated,
    })
}

async fn finish_stdin_task(mut task: tokio::task::JoinHandle<io::Result<()>>) {
    // A detached descendant can inherit the read end without consuming it.
    // Once the supervised process is gone, stdin is best effort and must not
    // strand the invocation.
    if tokio::time::timeout(TERMINATION_GRACE, &mut task)
        .await
        .is_err()
    {
        task.abort();
        let _ = task.await;
    }
}

async fn join_streams(
    mut stdout: tokio::task::JoinHandle<io::Result<FinishedStream>>,
    mut stderr: tokio::task::JoinHandle<io::Result<FinishedStream>>,
    stop: Arc<CancelToken>,
) -> Result<(FinishedStream, FinishedStream), ProcessError> {
    // One join future carries both drains through both grace windows. A
    // `JoinHandle` yields its output once and panics when polled again, so a
    // drain that finished inside the first window must not be joined afresh
    // in the second: the pinned join remembers which side is already done.
    let joined = {
        let mut both = std::pin::pin!(async { tokio::join!(&mut stdout, &mut stderr) });
        match tokio::time::timeout(OUTPUT_DRAIN_GRACE, &mut both).await {
            Ok(joined) => Some(joined),
            Err(_) => {
                // A descendant can escape the invocation's process group while
                // retaining an inherited pipe. Stop reading so that it cannot
                // strand the agent after the supervised process has terminated.
                stop.cancel();
                tokio::time::timeout(TERMINATION_GRACE, &mut both).await.ok()
            }
        }
    };
    let Some((stdout_result, stderr_result)) = joined else {
        stdout.abort();
        stderr.abort();
        return Err(ProcessError::Wait(io::Error::other(
            "output drains did not stop after process termination",
        )));
    };
    let finished = |result: Result<io::Result<FinishedStream>, tokio::task::JoinError>| {
        result
            .map_err(|error| ProcessError::Wait(io::Error::other(error)))?
            .map_err(ProcessError::Wait)
    };
    Ok((finished(stdout_result)?, finished(stderr_result)?))
}

async fn combine_logs(
    spill_dir: &Option<PathBuf>,
    stdout: Option<PathBuf>,
    stderr: Option<PathBuf>,
    stdout_stream: &CapturedStream,
    stderr_stream: &CapturedStream,
    force: bool,
) -> Result<Option<PathBuf>, ProcessError> {
    if stdout.is_none() && stderr.is_none() && !force {
        return Ok(None);
    }
    let Some(dir) = spill_dir else {
        return Ok(None);
    };
    if tokio::fs::create_dir_all(dir).await.is_err() {
        discard_spills(stdout, stderr).await;
        return Ok(None);
    }
    // The one moment the directory grows is the right moment to shrink it:
    // quiet commands never pay for a scan, and the files this invocation just
    // wrote are fresh, so age-based pruning cannot touch them.
    prune_spill_dir(dir).await;
    let path = dir.join(format!("cmd-{}.log", uuid::Uuid::new_v4()));
    let Ok(mut target) = tokio::fs::File::create(&path).await else {
        discard_spills(stdout, stderr).await;
        return Ok(None);
    };
    let copied = async {
        if let Some(stdout_path) = stdout.as_ref() {
            append_file(&mut target, stdout_path).await?;
        } else {
            target.write_all(&stdout_stream.rendered_bytes()).await?;
        }
        if let Some(stderr_path) = stderr.as_ref() {
            target.write_all(b"\n[stderr]\n").await?;
            append_file(&mut target, stderr_path).await?;
        } else if stderr_stream.total_bytes > 0 {
            target.write_all(b"\n[stderr]\n").await?;
            target.write_all(&stderr_stream.rendered_bytes()).await?;
        }
        target.flush().await
    }
    .await;
    if copied.is_err() {
        drop(target);
        let _ = tokio::fs::remove_file(&path).await;
        discard_spills(stdout, stderr).await;
        return Ok(None);
    }
    discard_spills(stdout, stderr).await;
    Ok(Some(path))
}

/// Best-effort age-based cleanup of the spill directory. Only the two names
/// this module writes are candidates (`cmd-*.log` and orphaned
/// `.openmax-*.tmp` from a crashed spill); anything else is left alone, and
/// every error is ignored because pruning must never fail the command.
///
/// A live spill can never age past the threshold: every caller that sets a
/// spill dir clamps its timeout to at most 300 s, so an in-flight
/// invocation's tmpfiles are always orders of magnitude fresher than the
/// seven-day cutoff. An unbounded-timeout caller would break that invariant.
async fn prune_spill_dir(dir: &Path) {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return;
    };
    let now = std::time::SystemTime::now();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let ours = (name.starts_with("cmd-") && name.ends_with(".log"))
            || (name.starts_with(".openmax-") && name.ends_with(".tmp"));
        if !ours {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let Ok(modified) = meta.modified() else {
            continue;
        };
        if now.duration_since(modified).is_ok_and(|age| age > SPILL_LOG_RETENTION) {
            let _ = tokio::fs::remove_file(entry.path()).await;
        }
    }
}

async fn discard_spills(stdout: Option<PathBuf>, stderr: Option<PathBuf>) {
    if let Some(stdout_path) = stdout {
        let _ = tokio::fs::remove_file(stdout_path).await;
    }
    if let Some(stderr_path) = stderr {
        let _ = tokio::fs::remove_file(stderr_path).await;
    }
}

async fn append_file(target: &mut tokio::fs::File, path: &Path) -> io::Result<()> {
    let mut source = tokio::fs::File::open(path).await?;
    tokio::io::copy(&mut source, target).await?;
    Ok(())
}

async fn supervise_child(
    child: &mut Child,
    pid: Option<u32>,
    timeout: Duration,
    cancel: &CancelToken,
) -> Result<(Termination, bool), ProcessError> {
    tokio::select! {
        result = child.wait() => {
            let status = result.map_err(ProcessError::Wait)?;
            // A shell can exit while ordinary background descendants retain the
            // group. Those belong to this invocation, so clean them up too.
            //
            // This is the rule that makes `some-server &` look like it worked
            // and then serve nothing: the shell returns, the group is killed,
            // and the caller is handed a successful exit. Report it so the
            // caller learns the rule from the result instead of from a failing
            // request later.
            let background_terminated = terminate_remaining_group(pid).await;
            Ok((Termination::Exited(status), background_terminated))
        }
        _ = cancel.cancelled() => {
            terminate_process_group(child, pid).await?;
            Ok((Termination::Cancelled, false))
        }
        _ = tokio::time::sleep(timeout) => {
            terminate_process_group(child, pid).await?;
            Ok((Termination::TimedOut, false))
        }
    }
}

async fn terminate_process_group(child: &mut Child, pid: Option<u32>) -> Result<(), ProcessError> {
    send_termination(child, pid);
    let reaped = tokio::time::timeout(TERMINATION_GRACE, child.wait()).await;
    match reaped {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => return Err(ProcessError::Wait(error)),
        Err(_) => {
            send_kill(child, pid);
            child.wait().await.map_err(ProcessError::Wait)?;
        }
    }
    let _ = terminate_remaining_group(pid).await;
    Ok(())
}

/// Returns true when group members outlived the leader and were terminated
/// here, which is the caller's only signal that a backgrounded child is gone.
async fn terminate_remaining_group(pid: Option<u32>) -> bool {
    // The leader may already have exited. Give remaining group members the
    // same grace period, then kill the group if any of them is still there.
    // The wait ends when the last one exits: a fixed sleep held every call
    // that left a background process for the whole grace, though one that
    // exits on SIGTERM is gone within milliseconds.
    if !process_group_exists(pid) {
        return false;
    }
    send_termination_group(pid);
    let deadline = tokio::time::Instant::now() + TERMINATION_GRACE;
    while process_group_exists(pid) {
        if tokio::time::Instant::now() >= deadline {
            send_kill_group(pid);
            break;
        }
        tokio::time::sleep(GROUP_EXIT_POLL).await;
    }
    true
}

/// Make the child lead its own process group, the unit every termination
/// path signals: a cancel or a timeout stops the whole tree, and members left
/// behind after a normal exit are found and stopped.
///
/// With `new_session` the child also leads a new session, which has no
/// controlling terminal. A child in the harness's session can open the
/// terminal the TUI draws on: a git credential prompt, an ssh host-key
/// question, or sudo then writes over the screen, and its read stops the
/// child (a background group may not read its terminal) until the call times
/// out. In a session of its own, opening /dev/tty fails at once and the
/// command reports why. A session leader's group id is its own pid, so the
/// group signals and the check for members left behind are unchanged.
#[cfg(unix)]
fn configure_process_group(command: &mut Command, new_session: bool) {
    use std::os::unix::process::CommandExt;
    if !new_session {
        command.as_std_mut().process_group(0);
        return;
    }
    // Runs between fork and exec, where only async-signal-safe calls are
    // allowed, and setsid is one. A failure fails the spawn rather than
    // running the command on the terminal. std has no stable way to ask
    // posix_spawn for a new session, so these spawns take its fork path.
    unsafe {
        command.as_std_mut().pre_exec(|| match libc::setsid() {
            -1 => Err(io::Error::last_os_error()),
            _ => Ok(()),
        });
    }
}

#[cfg(not(unix))]
fn configure_process_group(_: &mut Command, _: bool) {}

/// Leaders of the process groups commands started and have not settled yet.
/// Nothing else stops such a group once the harness is going away:
/// `kill_on_drop` reaches only the leader, and `std::process::exit` runs no
/// destructor at all.
static LIVE_GROUPS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Held by a command from just before its spawn until its group is
/// registered, so the exit's kill can wait for a spawn in flight, for at
/// most `EXIT_SPAWN_WAIT`, instead of missing the group it is about to start.
static SPAWNING: Mutex<()> = Mutex::new(());

/// Set by [`kill_process_groups`]: the process is about to exit, so a
/// command whose spawn has not begun is refused. One already spawning is not
/// stopped by it (see `EXIT_SPAWN_WAIT`).
static EXITING: AtomicBool = AtomicBool::new(false);

/// How long the exit's kill waits for a command caught between fork and exec.
/// A child can stall there (exec on a hung mount, a slow exec check), and a
/// second signal must still end the process. A spawn still in flight when
/// the wait ends is not stopped: every caller exits as soon as the kill
/// returns, so the spawning thread dies before it registers the group, and
/// the command runs on after the exit.
const EXIT_SPAWN_WAIT: Duration = TERMINATION_GRACE;

fn live_groups() -> std::sync::MutexGuard<'static, Vec<u32>> {
    LIVE_GROUPS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Kill every process group a running command started, and refuse to start
/// another, right before the process exits. This is for an exit that cannot
/// wait out the cancel path's grace, a second signal or a turn that never
/// settled; once the harness has exited, nothing would stop them. A spawn in
/// flight is waited for at most `EXIT_SPAWN_WAIT`, and one still in flight
/// then is not stopped: its command runs after the exit.
pub fn kill_process_groups() {
    EXITING.store(true, Ordering::SeqCst);
    let deadline = std::time::Instant::now() + EXIT_SPAWN_WAIT;
    while !matches!(SPAWNING.try_lock(), Ok(_) | Err(std::sync::TryLockError::Poisoned(_)))
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(5));
    }
    for pid in live_groups().iter() {
        send_kill_group(Some(*pid));
    }
    // Every group listed was just killed, so the record lists none a later
    // launch must stop, and the exit removes it.
    change_record(RecordChange::Clear);
}

/// Keeps a command's group in `LIVE_GROUPS` while it runs, and kills the
/// group when the command is dropped before it settles: the runtime shutting
/// down under a TUI that quit, or an aborted task. A destructor cannot wait
/// out a SIGTERM grace, so this is SIGKILL, as `kill_on_drop` is for the
/// leader.
struct GroupGuard(Option<u32>);

impl GroupGuard {
    /// Start `command` and register its group while holding `SPAWNING`, so
    /// the exit's kill waits for this spawn and finds the group, or comes
    /// first and the command never starts. A thread paused between the two
    /// steps would otherwise leave a group the kill never saw. The kill
    /// waits at most `EXIT_SPAWN_WAIT`, so a spawn that outlasts it is not
    /// stopped: the process exits while this thread is still spawning, and
    /// the command runs on after the exit.
    fn spawn(command: &mut Command) -> Result<(Child, Self), ProcessError> {
        let _spawning = SPAWNING.lock().unwrap_or_else(|e| e.into_inner());
        if EXITING.load(Ordering::SeqCst) {
            return Err(ProcessError::Spawn(io::Error::other("openmax is exiting")));
        }
        let child = command.spawn().map_err(ProcessError::Spawn)?;
        let pid = child.id();
        live_groups().extend(pid);
        if let Some(pid) = pid {
            record(pid);
        }
        Ok((child, Self(pid)))
    }

    /// Supervision stopped the group or found it empty. It is not signalled
    /// again: once its last member is gone, the id can name another group.
    fn settled(mut self) {
        if let Some(pid) = self.0.take() {
            forget(&mut live_groups(), pid);
            unrecord(pid);
        }
    }
}

fn forget(live: &mut Vec<u32>, pid: u32) {
    if let Some(at) = live.iter().position(|leader| *leader == pid) {
        live.swap_remove(at);
    }
}

impl Drop for GroupGuard {
    fn drop(&mut self) {
        if let Some(pid) = self.0.take() {
            // Forgotten and killed under one hold: an exit's kill between
            // the two would find the group gone from the list and still
            // running.
            let mut live = live_groups();
            forget(&mut live, pid);
            send_kill_group(Some(pid));
            unrecord(pid);
        }
    }
}

/// The run directory this process records its tool groups in, set by
/// [`reap_left_behind_groups`]; unset, nothing is recorded.
static RUN_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Where the changes to this process's record of the tool groups it started
/// and has not settled go: the record's thread, started by the first tool
/// or hook the process runs ([`record`]), so a launch that runs none spends
/// nothing on it, and `None` if it could not start. That thread owns the
/// record and makes every write to it ([`keep_record`]), so no tool's spawn
/// or settle, and no exit, waits on the home directory: a home whose file
/// server stopped answering stalls that thread alone. A harness that dies
/// without its exit path (SIGKILL, a crash that runs no handler) leaves the
/// file behind, and the next launch stops the groups it names.
static RECORDER: std::sync::OnceLock<Option<std::sync::mpsc::Sender<RecordChange>>> = std::sync::OnceLock::new();

/// A change to the record, made by its thread in the order sent.
enum RecordChange {
    Add(u32, u64),
    Strike(u32),
    /// Every group listed was just killed.
    Clear,
    /// The process is exiting: remove the record once it lists no group,
    /// then answer.
    Close(std::sync::mpsc::Sender<()>),
}

fn change_record(change: RecordChange) {
    if let Some(Some(recorder)) = RECORDER.get() {
        let _ = recorder.send(change);
    }
}

/// The record's thread: each change in the order sent, until the exit
/// closes the record.
fn keep_record(mut record: GroupRecord, changes: std::sync::mpsc::Receiver<RecordChange>) {
    for change in changes {
        match change {
            RecordChange::Add(pid, start) => record.add(pid, start),
            RecordChange::Strike(pid) => record.strike(pid),
            RecordChange::Clear => record.live.clear(),
            RecordChange::Close(closed) => {
                close(record);
                let _ = closed.send(());
                return;
            }
        }
    }
}

/// Once the record has grown past this many bytes, the next strike that
/// leaves no group listed cuts it back to its first line. Every group ends
/// with its tool call, so that happens once a turn's calls have ended, and
/// the file stays near this size.
const RECORD_CUT_BYTES: u64 = 4096;

/// `{data_dir}/run/{pid}.groups`, created by the first tool a launch starts and
/// then held open by the record's thread, so a tool round costs it two appends,
/// not a rewrite of the file. The first line is the owner's start time, then on
/// Linux this boot's id; each `<leader> <start>` line adds a group, and each
/// `-<leader> <start>` line strikes one. Every line is one write, which a
/// SIGKILL can still cut short, as Linux ends a write at a page boundary of the
/// file once one is pending. Only the last line can be cut, and the reader then
/// skips it or finds in it nothing to signal: a start time cut short is
/// earlier than that of any process that has had the pid since, and a strike
/// cut short leaves listed a group that was stopped, or found empty, before
/// the strike was sent. A write that fails, as on a full disk, can leave part
/// of a line too, which the next line would run into, so the next change
/// rewrites the file whole.
struct GroupRecord {
    path: PathBuf,
    own_start: u64,
    file: Option<std::fs::File>,
    /// The file's length, where the next line goes.
    len: u64,
    /// The first line's length, its newline included, once it is written.
    first_len: u64,
    live: Vec<(u32, u64)>,
}

impl GroupRecord {
    fn new(path: PathBuf, own_start: u64) -> Self {
        Self { path, own_start, file: None, len: 0, first_len: 0, live: Vec::new() }
    }

    fn add(&mut self, pid: u32, start: u64) {
        self.live.push((pid, start));
        self.write(&format!("{pid} {start}\n"));
    }

    fn strike(&mut self, pid: u32) {
        let Some(at) = self.live.iter().position(|(leader, _)| *leader == pid) else { return };
        let (pid, start) = self.live.swap_remove(at);
        if self.live.is_empty() && self.len > RECORD_CUT_BYTES {
            use std::io::Seek;
            let first_line = self.first_len;
            if let Some(mut file) = self.file.as_ref() {
                if file.set_len(first_line).and_then(|()| file.seek(io::SeekFrom::Start(first_line))).is_ok() {
                    self.len = first_line;
                    return;
                }
            }
        }
        self.write(&format!("-{pid} {start}\n"));
    }

    fn write(&mut self, line: &str) {
        use std::io::Write;
        if let Some(mut file) = self.file.as_ref() {
            if file.write_all(line.as_bytes()).is_ok() {
                self.len += line.len() as u64;
                return;
            }
            // A write cut short can leave part of a line, which the next
            // one would run into and hide: rewrite the file whole instead.
            self.file = None;
        }
        // Not open yet, or a write failed: the file starts with the first
        // line and every group listed, in one write. The run directory is
        // created by the first tool any launch starts.
        let first = first_line(self.own_start, boot_id().as_deref());
        let mut text = format!("{first}\n");
        for (pid, start) in &self.live {
            text.push_str(&format!("{pid} {start}\n"));
        }
        let create = || std::fs::File::create(&self.path);
        let created = create().or_else(|_| {
            std::fs::create_dir_all(self.path.parent().unwrap_or(Path::new(".")))?;
            create()
        });
        if let Ok(mut file) = created {
            if file.write_all(text.as_bytes()).is_ok() {
                self.len = text.len() as u64;
                self.first_len = first.len() as u64 + 1;
                self.file = Some(file);
            }
        }
    }
}

/// Read the leader's start time here, while its pid surely names it (one
/// that has exited keeps its pid until it is waited for), and leave the
/// line to the record's thread, which the first tool or hook starts.
fn record(pid: u32) {
    let Some(run) = RUN_DIR.get() else { return };
    if let (Some(recorder), Some(start)) = (RECORDER.get_or_init(|| start_recording(run)), start_time(pid)) {
        let _ = recorder.send(RecordChange::Add(pid, start));
    }
}

/// Start the record's thread on `{run}/{pid}.groups`, and have the exit
/// close the record.
fn start_recording(run: &Path) -> Option<std::sync::mpsc::Sender<RecordChange>> {
    let pid = std::process::id();
    let record = GroupRecord::new(run.join(format!("{pid}.groups")), start_time(pid)?);
    let (recorder, changes) = std::sync::mpsc::channel();
    std::thread::Builder::new().name("record".into()).spawn(move || keep_record(record, changes)).ok()?;
    // Run by `exit`, which a return from `main` reaches too, after the
    // runtime's shutdown has killed each group a dropped command owned.
    #[cfg(unix)]
    unsafe { libc::atexit(close_groups_record) };
    Some(recorder)
}

fn unrecord(pid: u32) {
    change_record(RecordChange::Strike(pid));
}

/// Stop the tool groups a harness that died without its exit path left
/// running, then have this process record its own, from the first tool or
/// hook it runs. Every launch that can start a turn or a tool calls it
/// first. A record whose owner is gone (no process of this user has its
/// pid, or the process that has it has exited or started at another time
/// than the record says) names groups nothing else will stop: each whose
/// leader still has the start time recorded for it gets SIGTERM, then
/// SIGKILL after `TERMINATION_GRACE`, and the record is removed, with the
/// run directory once no record is left there. A record from another boot
/// is removed with nothing signalled, and one a running openmax owns is
/// left alone. A process that left its tool's group, and a group whose
/// leader has exited and been waited for, are not covered. The group this
/// launch runs in is not signalled, and while its leader runs, its record
/// is kept for a launch outside it.
#[cfg(unix)]
pub fn reap_left_behind_groups(data_dir: &Path) {
    let run = data_dir.join("run");
    reap(&run);
    let _ = RUN_DIR.set(run);
}

#[cfg(not(unix))]
pub fn reap_left_behind_groups(_: &Path) {}

/// Have the record's thread close the record, and wait for it at most
/// `RECORD_CLOSE_WAIT`.
#[cfg(unix)]
extern "C" fn close_groups_record() {
    let (closed, done) = std::sync::mpsc::channel();
    change_record(RecordChange::Close(closed));
    let _ = done.recv_timeout(RECORD_CLOSE_WAIT);
}

/// How long an exit waits for the record's thread to remove the record. A
/// home whose file server stopped answering stalls the thread, and an exit
/// stalled with it would never end, a second signal's included. Past the
/// wait the process exits, and the record is left for the next launch to
/// remove.
const RECORD_CLOSE_WAIT: Duration = Duration::from_millis(100);

/// At exit, remove the record once it lists no group, and the run directory
/// once no other openmax keeps a record there, so the next launch reads
/// nothing. A group still listed is one no exit path stopped: its record
/// stays for the next launch.
#[cfg(unix)]
fn close(record: GroupRecord) {
    if record.file.is_some() && record.live.is_empty() {
        let _ = std::fs::remove_file(&record.path);
        if let Some(run) = record.path.parent() {
            let _ = std::fs::remove_dir(run);
        }
    }
}

#[cfg(unix)]
fn reap(run: &Path) {
    // Most launches find no run directory, and read no boot id.
    if run.exists() {
        reap_in_boot(run, boot_id().as_deref());
    }
}

/// [`reap`], in the boot whose id is `boot`.
#[cfg(unix)]
fn reap_in_boot(run: &Path, boot: Option<&str>) {
    let own_group = unsafe { libc::getpgrp() } as u32;
    let mut groups = Vec::new();
    let mut stale = Vec::new();
    for entry in std::fs::read_dir(run).into_iter().flatten().flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Some(owner) = name.to_str().and_then(|n| n.strip_suffix(".groups")).and_then(positive_pid) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let mut lines = text.lines();
        let first = lines.next();
        if written_in(boot, first) {
            if owner_is_running(owner, first) {
                continue;
            }
            let listed = recorded_groups(lines).into_iter().filter(|&(pid, start)| start_time(pid) == Some(start));
            let (own, others): (Vec<_>, Vec<_>) = listed.partition(|&(pid, _)| pid == own_group);
            groups.extend(others);
            // The group this launch runs in, as an openmax that a left-behind
            // tool's script starts does: a signal to it would end this launch
            // as it starts. It is left running, with its record, for the next
            // launch outside it to stop.
            if !own.is_empty() {
                continue;
            }
        }
        stale.push(path);
    }
    stop_groups(&groups, is_recorded_group);
    if stale.is_empty() {
        return;
    }
    for path in stale {
        let _ = std::fs::remove_file(path);
    }
    // And the run directory, once no record is left in it.
    let _ = std::fs::remove_dir(run);
}

/// The groups a record's lines leave listed: each one added and not struck
/// by a later line.
#[cfg(unix)]
fn recorded_groups<'a>(lines: impl Iterator<Item = &'a str>) -> Vec<(u32, u64)> {
    let mut groups = Vec::new();
    for line in lines {
        let (struck, line) = match line.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, line),
        };
        let Some((pid, start)) = line.split_once(' ') else { continue };
        let (Some(pid), Ok(start)) = (leader_pid(pid), start.parse::<u64>()) else { continue };
        if !struck {
            groups.push((pid, start));
        } else if let Some(at) = groups.iter().position(|group| *group == (pid, start)) {
            groups.swap_remove(at);
        }
    }
    groups
}

/// A pid that names one process: kill takes 0 and negative values as
/// groups.
#[cfg(unix)]
fn positive_pid(text: &str) -> Option<u32> {
    text.parse::<libc::pid_t>().ok().filter(|pid| *pid > 0).map(|pid| pid as u32)
}

/// A recorded group leader: never 1, whose group signal, kill(-1), reaches
/// every process this user may signal. The group this launch runs in is
/// left to [`reap_in_boot`].
#[cfg(unix)]
fn leader_pid(text: &str) -> Option<u32> {
    positive_pid(text).filter(|pid| *pid > 1)
}

/// Whether the openmax that wrote a record still runs: its pid is held by a
/// process this user may signal, which has not exited and started when the
/// first field of the record's first line says. An openmax runs as the user
/// that reads its record, so a pid no such process holds (ESRCH, or EPERM
/// for another user's process) is a harness that is gone, as is one that
/// has exited, which holds its pid until its parent waits for it, as a
/// killed harness does. A start time that cannot be read, or a first line
/// not yet written, is taken as still running.
#[cfg(unix)]
fn owner_is_running(pid: u32, recorded_start: Option<&str>) -> bool {
    if unsafe { libc::kill(pid as libc::pid_t, 0) } != 0 {
        return false;
    }
    match (started(pid), recorded_start.and_then(|line| line.split(' ').next()?.parse::<u64>().ok())) {
        (Some((_, true)), _) => false,
        (Some((start, false)), Some(recorded)) => start == recorded,
        _ => true,
    }
}

/// SIGTERM each recorded group, then SIGKILL each still there once the
/// grace ends. `recorded` ([`is_recorded_group`]) tells whether a group is
/// still the one recorded, and each is dropped the first time it is not,
/// never to be signalled again: once a group has ended, its id can name a
/// new group, which reads as recorded when its own leader has exited.
#[cfg(unix)]
fn stop_groups(groups: &[(u32, u64)], mut recorded: impl FnMut(u32, u64) -> bool) {
    let mut stopping: Vec<(u32, u64)> = groups.iter().copied().filter(|&(pid, start)| recorded(pid, start)).collect();
    for (pid, _) in &stopping {
        send_termination_group(Some(*pid));
    }
    let deadline = std::time::Instant::now() + TERMINATION_GRACE;
    loop {
        stopping.retain(|&(pid, start)| recorded(pid, start));
        if stopping.is_empty() || std::time::Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    for (pid, _) in stopping {
        send_kill_group(Some(pid));
    }
}

/// Whether the group led by `pid` is still the one a record names, given
/// that it has not read as gone since the record was read: it exists, and
/// its leader runs with the recorded start time or has exited. No process
/// gets a pid that is a live group's ID (POSIX fork: "The child process ID
/// also shall not match any active process group ID"), so a group that has
/// existed throughout and whose leader is gone is still the recorded one,
/// while a leader pid that reads another start time is a process that took
/// the pid after the recorded group ended, and may lead a group of its own.
#[cfg(unix)]
fn is_recorded_group(pid: u32, start: u64) -> bool {
    process_group_exists(Some(pid)) && start_time(pid).is_none_or(|now| now == start)
}

/// A record's first line: the owner's start time, then the boot's id where
/// the system gives one.
fn first_line(own_start: u64, boot: Option<&str>) -> String {
    match boot {
        Some(boot) => format!("{own_start} {boot}"),
        None => own_start.to_string(),
    }
}

/// Whether a record's first line was written in the boot whose id is
/// `boot`. Linux start times count clock ticks from boot, so after a restart
/// a pid and start a record names can belong to a new process: a record
/// from another boot names processes that are all gone, and none of its
/// groups may be signalled. A first line or a system with no boot id is
/// taken as this boot's.
#[cfg(unix)]
fn written_in(boot: Option<&str>, first_line: Option<&str>) -> bool {
    match (boot, first_line.and_then(|line| line.split_once(' '))) {
        (Some(boot), Some((_, recorded))) => recorded == boot,
        _ => true,
    }
}

/// This boot's id: "a UUID generated the first time this is retrieved, and
/// unvarying after that" (the kernel's sysctl documentation for
/// `/proc/sys/kernel/random/boot_id`), and a new one each boot.
#[cfg(target_os = "linux")]
fn boot_id() -> Option<String> {
    let id = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let id = id.trim();
    (!id.is_empty() && !id.contains(' ')).then(|| id.to_owned())
}

/// Elsewhere none is needed: macOS start times count from the epoch, so a
/// pid and start a record names never meet a process of a later boot.
#[cfg(not(target_os = "linux"))]
fn boot_id() -> Option<String> {
    None
}

/// When the process `pid` started, in microseconds since the epoch, and
/// whether it has exited: the `pbi_start_tvsec`, `pbi_start_tvusec` and
/// `pbi_status` of `proc_pidinfo`'s PROC_PIDTBSDINFO. The start time is fixed
/// for the life of a process, so a pid that now names another process reads
/// another value. A process that has exited holds its pid until its parent
/// waits for it, and reads only with a nonzero argument, as XNU's
/// `proc_pidinfo` then looks among those too; its status is then `SZOMB`.
#[cfg(target_os = "macos")]
fn started(pid: u32) -> Option<(u64, bool)> {
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    let read = unsafe {
        libc::proc_pidinfo(pid as libc::c_int, libc::PROC_PIDTBSDINFO, 1, (&raw mut info).cast(), size)
    };
    (read == size).then(|| (info.pbi_start_tvsec * 1_000_000 + info.pbi_start_tvusec, info.pbi_status == libc::SZOMB))
}

/// When the process `pid` started, in clock ticks after boot, and whether it
/// has exited, from `/proc/<pid>/stat`, which a process that has exited has
/// until its parent waits for it.
#[cfg(target_os = "linux")]
fn started(pid: u32) -> Option<(u64, bool)> {
    stat_started(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn started(_: u32) -> Option<(u64, bool)> {
    None
}

/// When the process `pid` started, whether or not it has exited.
fn start_time(pid: u32) -> Option<u64> {
    started(pid).map(|(start, _)| start)
}

/// Field 22, starttime, of a `/proc/<pid>/stat` line, and whether field 3,
/// the state, is Z, a process that has exited. Fields are counted after the
/// command name's closing parenthesis, since the name itself can hold spaces
/// and parentheses: the first field after it is field 3.
#[cfg(any(target_os = "linux", test))]
fn stat_started(stat: &str) -> Option<(u64, bool)> {
    let (_, fields) = stat.rsplit_once(')')?;
    let mut fields = fields.split_whitespace();
    let exited = fields.next()? == "Z";
    Some((fields.nth(22 - 4)?.parse().ok()?, exited))
}

#[cfg(all(test, unix))]
thread_local! {
    /// Set by [`PausedSpawn`] for the next spawn on its thread: the pipe the
    /// child reports its fork on, the pipe it waits on before exec, and that
    /// pipe's write end.
    static PAUSE_BEFORE_EXEC: std::cell::Cell<Option<[std::os::fd::RawFd; 3]>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(all(test, unix))]
fn hold_before_exec(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    let Some([forked, resume, resume_writer]) = PAUSE_BEFORE_EXEC.take() else { return };
    // SAFETY: close, write, and read are async-signal-safe, the only kind of
    // call allowed between fork and exec.
    unsafe {
        command.as_std_mut().pre_exec(move || {
            // The child's own copy of the write end would keep it waiting
            // forever if the test let go without writing.
            libc::close(resume_writer);
            let mut byte = 0_u8;
            libc::write(forked, (&raw const byte).cast(), 1);
            libc::read(resume, (&raw mut byte).cast(), 1);
            Ok(())
        });
    }
}

/// A command started through [`run_process`] and held between fork and exec,
/// the window in which a spawning child holds a copy of every descriptor the
/// harness has open. Dropping it lets the child exec and waits for the
/// command to finish.
#[cfg(all(test, unix))]
pub(crate) struct PausedSpawn {
    resume: Option<std::io::PipeWriter>,
    spawner: Option<std::thread::JoinHandle<()>>,
}

#[cfg(all(test, unix))]
impl PausedSpawn {
    pub(crate) fn start() -> Self {
        use std::io::Read;
        use std::os::fd::AsRawFd;
        let (mut forked, forked_writer) = std::io::pipe().unwrap();
        let (resume_reader, resume) = std::io::pipe().unwrap();
        let fds = [forked_writer.as_raw_fd(), resume_reader.as_raw_fd(), resume.as_raw_fd()];
        let spawner = std::thread::spawn(move || {
            PAUSE_BEFORE_EXEC.set(Some(fds));
            let request = ProcessRequest {
                program: "/bin/echo".into(),
                args: Vec::new(),
                cwd: std::env::temp_dir(),
                stdin: StdinMode::Null,
                timeout: Duration::from_secs(30),
                capture: CaptureSpec { head_bytes: 64, tail_bytes: 64, spill_dir: None, spill_bytes_per_stream: 1024, redraws: false },
                sandbox: None,
                env_allowlist: None,
                self_link: false,
                env: Vec::new(),
            };
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let output = runtime.block_on(run_process(request, Arc::new(CancelToken::default())));
            // Held until the spawn returns, so a spawn that never forked
            // reads as end of file instead of a wait that never ends.
            drop((forked_writer, resume_reader));
            assert!(output.is_ok(), "the held command failed once resumed");
        });
        assert_eq!(forked.read(&mut [0_u8]).unwrap(), 1, "the command never reached its fork");
        Self { resume: Some(resume), spawner: Some(spawner) }
    }
}

#[cfg(all(test, unix))]
impl Drop for PausedSpawn {
    fn drop(&mut self) {
        use std::io::Write;
        if let Some(mut resume) = self.resume.take() {
            let _ = resume.write_all(&[0]);
        }
        if let Some(spawner) = self.spawner.take() {
            let finished = spawner.join();
            if !std::thread::panicking() {
                finished.unwrap();
            }
        }
    }
}

#[cfg(unix)]
fn send_termination(_: &mut Child, pid: Option<u32>) {
    signal_group(pid, libc::SIGTERM);
}

#[cfg(not(unix))]
fn send_termination(child: &mut Child, _: Option<u32>) {
    let _ = child.start_kill();
}

#[cfg(unix)]
fn send_kill(_: &mut Child, pid: Option<u32>) {
    signal_group(pid, libc::SIGKILL);
}

#[cfg(not(unix))]
fn send_kill(child: &mut Child, _: Option<u32>) {
    let _ = child.start_kill();
}

#[cfg(unix)]
fn send_kill_group(pid: Option<u32>) {
    signal_group(pid, libc::SIGKILL);
}

#[cfg(not(unix))]
fn send_kill_group(_: Option<u32>) {}

#[cfg(unix)]
fn send_termination_group(pid: Option<u32>) {
    signal_group(pid, libc::SIGTERM);
}

#[cfg(not(unix))]
fn send_termination_group(_: Option<u32>) {}

#[cfg(unix)]
fn process_group_exists(pid: Option<u32>) -> bool {
    let Some(pid) = pid else {
        return false;
    };
    // kill(..., 0) never changes process state. This harness owns the group,
    // so a failure is treated as absent and keeps the normal path fast.
    unsafe { libc::kill(-(pid as libc::pid_t), 0) == 0 }
}

#[cfg(not(unix))]
fn process_group_exists(_: Option<u32>) -> bool {
    false
}

#[cfg(unix)]
fn signal_group(pid: Option<u32>, signal: libc::c_int) {
    if let Some(pid) = pid {
        // A missing group simply means every process has already exited.
        unsafe { libc::kill(-(pid as libc::pid_t), signal) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(program: &str, args: &[&str]) -> ProcessRequest {
        ProcessRequest {
            program: program.into(),
            args: args.iter().map(|arg| (*arg).into()).collect(),
            cwd: std::env::temp_dir(),
            stdin: StdinMode::Null,
            timeout: Duration::from_secs(5),
            capture: CaptureSpec {
                head_bytes: 4,
                tail_bytes: 4,
                spill_dir: None,
                spill_bytes_per_stream: 1024,
                redraws: false,
            },
            sandbox: None,
            env_allowlist: None,
            self_link: false,
            env: Vec::new(),
        }
    }

    /// One drain finished, the other outlived the grace: the shape of a
    /// detached descendant that kept an inherited pipe open. The second wait
    /// must not poll the finished drain again: a `JoinHandle` polled after
    /// completion panics, taking the whole turn down with it.
    #[tokio::test(start_paused = true)]
    async fn a_finished_drain_is_not_polled_again_when_the_other_hangs() {
        let finished = |bytes: &'static [u8]| FinishedStream {
            stream: CapturedStream { total_bytes: bytes.len() as u64, head: bytes.to_vec(), tail: Vec::new() },
            spill_path: None,
            omitted: false,
        };
        let stdout = tokio::spawn(async move { Ok(finished(b"out")) });
        let stop = Arc::new(CancelToken::default());
        let stop_for_stderr = stop.clone();
        let stderr = tokio::spawn(async move {
            stop_for_stderr.cancelled().await;
            Ok(finished(b"err"))
        });
        let (out, err) = join_streams(stdout, stderr, stop).await.expect("both drains settle");
        assert_eq!(out.stream.head, b"out");
        assert_eq!(err.stream.head, b"err");
    }

    #[tokio::test]
    async fn spawned_processes_carry_the_session_marker() {
        let request = ProcessRequest {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "printf %s \"$OPENMAX_SESSION\"".into()],
            cwd: std::env::temp_dir(),
            stdin: StdinMode::Null,
            timeout: Duration::from_secs(5),
            capture: CaptureSpec {
                head_bytes: 1024,
                tail_bytes: 1024,
                spill_dir: None,
                spill_bytes_per_stream: 0,
                redraws: false,
            },
            sandbox: None,
            env_allowlist: None,
            self_link: false,
            env: Vec::new(),
        };
        let output = run_process(request, Arc::new(CancelToken::default())).await.unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout.head), "1");
    }

    /// A command's group leaves the exit's kill list once the command
    /// settles: the group is gone by then, and by the time the exit's kill
    /// runs its id can name another process's group.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_settled_command_leaves_no_group_for_the_exit_to_kill() {
        let mut request = request("/bin/sh", &["-c", "echo $$"]);
        request.capture.head_bytes = 64;
        let output = run_process(request, Arc::new(CancelToken::default())).await.unwrap();
        let leader: u32 = String::from_utf8_lossy(&output.stdout.head).trim().parse().unwrap();
        assert!(!live_groups().contains(&leader), "the settled group {leader} is still listed");
    }

    /// Forgetting a settled group keeps every other on the list, so the
    /// exit's kill still reaches the groups of commands still running. The
    /// list is shared by every command in the process, so this is checked
    /// on a list of its own.
    #[test]
    fn forgetting_a_group_keeps_the_others() {
        let mut live = vec![11, 22, 33];
        forget(&mut live, 22);
        live.sort_unstable();
        assert_eq!(live, [11, 33]);
        forget(&mut live, 44);
        assert_eq!(live, [11, 33], "forgetting an unlisted group changed the list");
    }

    /// The start time is field 22 of /proc/<pid>/stat, and the state field 3,
    /// counted after the command name, which can itself hold spaces and
    /// parentheses.
    #[test]
    fn the_linux_start_time_is_field_22_counted_after_the_command_name() {
        let stat = "4242 (a) b (c) S 1 4242 4242 0 -1 4194560 100 0 0 0 0 0 0 0 20 0 1 0 98765 1000 7";
        assert_eq!(stat_started(stat), Some((98765, false)));
        assert_eq!(stat_started(&stat.replace(") S ", ") Z ")), Some((98765, true)), "a state of Z is an exited process");
        assert_eq!(stat_started("4242 (cut"), None);
    }

    /// A process that has exited holds its pid until its parent waits for
    /// it, and its start time still reads until then, with the exit told
    /// apart: so a tool group whose leader exits before its spawn returns is
    /// recorded, and a killed harness its parent has not waited for reads as
    /// gone, not as running. waitid with WNOWAIT returns once the child has
    /// exited and leaves it unwaited for.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn an_exited_process_reads_its_start_time_until_it_is_waited_for() {
        let mut child = std::process::Command::new("/bin/sh")
            .args(["-c", "read line"])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let running = started(child.id());
        drop(child.stdin.take());
        let (mut info, options) = (unsafe { std::mem::zeroed::<libc::siginfo_t>() }, libc::WEXITED | libc::WNOWAIT);
        let exited = match unsafe { libc::waitid(libc::P_PID, child.id() as libc::id_t, &mut info, options) } {
            0 => Ok(started(child.id())),
            _ => Err(io::Error::last_os_error()),
        };
        child.wait().unwrap();
        let start = running.expect("a running process has a start time").0;
        assert_eq!(running, Some((start, false)), "a running process read as exited");
        assert_eq!(exited.unwrap(), Some((start, true)), "an exited process its parent has not waited for");
    }

    /// The boot's id is what tells a record from an earlier boot apart: it
    /// reads, holds no space (the first line splits on one), and stays the
    /// same within a boot.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_boot_id_reads_and_holds_for_the_boot() {
        let id = boot_id().expect("Linux gives every boot an id");
        assert_eq!(boot_id().as_deref(), Some(id.as_str()), "the boot id changed within a boot");
        assert!(!id.is_empty() && !id.contains(' '), "{id:?}");
        assert_eq!(first_line(5, Some(&id)), format!("5 {id}"));
    }

    /// A process in a group of its own, standing in for a tool a killed
    /// harness left running, with its start time as a record names it.
    #[cfg(unix)]
    fn left_behind_group() -> (std::process::Child, u64) {
        use std::os::unix::process::CommandExt;
        let child = std::process::Command::new("sleep").arg("300").process_group(0).spawn().unwrap();
        let start = start_time(child.id()).expect("a running child has a start time");
        (child, start)
    }

    /// The same, running `setup` under perl first, and returned only once it
    /// has, so a signal handler it installs is in place.
    #[cfg(unix)]
    fn left_behind_perl(setup: &str) -> (std::process::Child, u64) {
        use std::io::BufRead;
        use std::os::unix::process::CommandExt;
        let mut child = std::process::Command::new("perl")
            .args(["-e", &format!("$| = 1; {setup}; print qq(ready\\n); sleep 300")])
            .stdout(std::process::Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let mut ready = String::new();
        std::io::BufReader::new(child.stdout.as_mut().unwrap()).read_line(&mut ready).unwrap();
        assert_eq!(ready, "ready\n");
        let start = start_time(child.id()).expect("a running child has a start time");
        (child, start)
    }

    /// A pid no process holds, the owner of a record whose harness is gone.
    #[cfg(unix)]
    fn dead_pid() -> u32 {
        let mut child = std::process::Command::new("/bin/sh").args(["-c", "exit 0"]).spawn().unwrap();
        child.wait().unwrap();
        child.id()
    }

    #[cfg(unix)]
    fn run_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-reap-{tag}-{}", uuid::Uuid::new_v4())).join("run");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(unix)]
    fn stopped_within(child: &mut std::process::Child, wait: Duration) -> bool {
        let deadline = std::time::Instant::now() + wait;
        while std::time::Instant::now() < deadline {
            if child.try_wait().unwrap().is_some() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    /// A dead harness's pid can since be held by another user's process,
    /// which this user may not signal, as with pid 1. No openmax this user
    /// runs has that pid, so the record is reaped like any other a dead
    /// harness left, even when its first line is that process's start time.
    #[cfg(unix)]
    #[test]
    fn a_record_whose_owner_pid_another_user_holds_is_reaped() {
        let (mut group, start) = left_behind_group();
        let run = run_dir("other-user");
        let record = run.join("1.groups");
        // pid 1's own start time, so only the refused signal tells that no
        // openmax of this user holds it. Where this user may signal pid 1
        // (as root, or in a container whose init it runs), name another.
        let owner_start = match start_time(1) {
            Some(start) if unsafe { libc::kill(1, 0) } == 0 => start + 1,
            Some(start) => start,
            None => 0,
        };
        std::fs::write(&record, format!("{owner_start}\n{} {start}\n", group.id())).unwrap();
        reap(&run);
        let stopped = stopped_within(&mut group, Duration::from_millis(500));
        let _ = group.kill();
        let kept = record.exists();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert!(stopped, "a record whose owner pid another user holds kept its group running");
        assert!(!kept, "a record whose owner pid another user holds was kept");
    }

    /// A group that outlives SIGTERM gets SIGKILL once the grace ends, while
    /// one that ends on SIGTERM is given the grace to: it is not killed
    /// outright.
    #[cfg(unix)]
    #[test]
    fn a_group_that_outlives_sigterm_is_killed_once_the_grace_ends() {
        use std::os::unix::process::ExitStatusExt;
        let (mut stubborn, stubborn_start) = left_behind_perl(r#"$SIG{TERM} = "IGNORE""#);
        let (mut graceful, graceful_start) = left_behind_perl("$SIG{TERM} = sub { exit 7 }");
        let run = run_dir("grace");
        let record = run.join(format!("{}.groups", dead_pid()));
        let lines = format!("1\n{} {stubborn_start}\n{} {graceful_start}\n", stubborn.id(), graceful.id());
        std::fs::write(&record, lines).unwrap();
        reap(&run);
        let stubborn_stopped = stopped_within(&mut stubborn, Duration::from_millis(500));
        let graceful_stopped = stopped_within(&mut graceful, Duration::from_millis(500));
        let _ = stubborn.kill();
        let _ = graceful.kill();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert!(stubborn_stopped, "a group that ignores SIGTERM outlived the reap");
        assert_eq!(stubborn.wait().unwrap().signal(), Some(libc::SIGKILL));
        assert!(graceful_stopped, "a group that ends on SIGTERM outlived the reap");
        let graceful = graceful.wait().unwrap();
        assert_eq!(graceful.code(), Some(7), "a group that ends on SIGTERM was not given the grace: {graceful:?}");
    }

    /// The grace is waited out only for a group still running: once every
    /// signalled group is gone the reap returns, so a launch that stops a
    /// left-behind tool which ends on SIGTERM does not start 250 ms late.
    /// A thread reaps the process the moment it exits, as init does for a
    /// tool whose harness died, since a zombie still counts as its group.
    #[cfg(unix)]
    #[test]
    fn a_reap_whose_groups_end_on_sigterm_does_not_wait_out_the_grace() {
        let (mut graceful, start) = left_behind_perl("$SIG{TERM} = sub { exit 7 }");
        let pid = graceful.id();
        let reaped = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while std::time::Instant::now() < deadline {
                if let Some(status) = graceful.try_wait().unwrap() {
                    return Some(status);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            let _ = graceful.kill();
            let _ = graceful.wait();
            None
        });
        let run = run_dir("prompt");
        std::fs::write(run.join(format!("{}.groups", dead_pid())), format!("1\n{pid} {start}\n")).unwrap();
        let began = std::time::Instant::now();
        reap(&run);
        let took = began.elapsed();
        let status = reaped.join().unwrap();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert_eq!(status.and_then(|status| status.code()), Some(7), "the group did not end on the reap's SIGTERM");
        assert!(took < TERMINATION_GRACE, "a reap whose group ended on SIGTERM took {took:?}, the whole grace");
    }

    /// Each signal of a stop goes only to a group still the one recorded. A
    /// leader pid that reads another start time is a process that took the
    /// pid once the recorded group ended, as can happen within the grace: it
    /// is never signalled. A leader that ends on SIGTERM while a member of
    /// its group ignores it leaves the group the recorded one, since no
    /// process gets a pid a live group bears, so the member is killed once
    /// the grace ends. A thread reaps the leader as init would.
    #[cfg(unix)]
    #[test]
    fn a_stop_signals_only_a_group_still_the_one_recorded() {
        use std::io::BufRead;
        use std::os::unix::process::{CommandExt, ExitStatusExt};
        let (mut earlier, earlier_start) = left_behind_group();
        std::thread::sleep(Duration::from_millis(20));
        let (mut bystander, _) = left_behind_group();
        let (mut leader, leader_start) = left_behind_perl("$SIG{TERM} = sub { exit 0 }");
        let leader_pid = leader.id();
        let mut member = std::process::Command::new("perl")
            .args(["-e", r#"$| = 1; $SIG{TERM} = "IGNORE"; print qq(ready\n); sleep 300"#])
            .stdout(std::process::Stdio::piped())
            .process_group(leader_pid as libc::pid_t)
            .spawn()
            .unwrap();
        let mut ready = String::new();
        std::io::BufReader::new(member.stdout.as_mut().unwrap()).read_line(&mut ready).unwrap();
        assert_eq!(ready, "ready\n");
        let reaped = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while std::time::Instant::now() < deadline {
                if let Some(status) = leader.try_wait().unwrap() {
                    return Some(status);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            let _ = leader.kill();
            let _ = leader.wait();
            None
        });
        stop_groups(&[(bystander.id(), earlier_start), (leader_pid, leader_start)], is_recorded_group);
        let member_stopped = stopped_within(&mut member, Duration::from_millis(500));
        let bystander_alive = !stopped_within(&mut bystander, Duration::from_millis(50));
        let _ = member.kill();
        let _ = bystander.kill();
        let _ = earlier.kill();
        let leader_status = reaped.join().unwrap();
        assert!(bystander_alive, "a process that took a recorded leader's pid was signalled");
        assert_eq!(leader_status.and_then(|status| status.code()), Some(0), "the leader did not end on SIGTERM");
        assert!(member_stopped, "a group whose leader ended on SIGTERM kept a member running");
        assert_eq!(member.wait().unwrap().signal(), Some(libc::SIGKILL));
    }

    /// Once a group has ended, its id can name a new group that reads as the
    /// one recorded: a process that took the leader's pid led a group of its
    /// own and has exited, leaving members. So a stop drops each group the
    /// first time it reads as gone, and never signals it again. The group
    /// stood in for here reads as recorded at the SIGTERM, which it ignores,
    /// as gone at the first look within the grace, and as recorded after;
    /// another that ignores SIGTERM holds the grace open to its end.
    #[cfg(unix)]
    #[test]
    fn a_group_that_read_as_gone_is_never_signalled_again() {
        let (mut taken, _) = left_behind_perl(r#"$SIG{TERM} = "IGNORE""#);
        let (mut stubborn, stubborn_start) = left_behind_perl(r#"$SIG{TERM} = "IGNORE""#);
        let taken_pid = taken.id();
        let mut looks = 0;
        stop_groups(&[(taken_pid, 0), (stubborn.id(), stubborn_start)], |pid, start| {
            if pid != taken_pid {
                return is_recorded_group(pid, start);
            }
            looks += 1;
            looks != 2
        });
        let stubborn_stopped = stopped_within(&mut stubborn, Duration::from_millis(500));
        let taken_alive = !stopped_within(&mut taken, Duration::from_millis(50));
        let _ = taken.kill();
        let _ = stubborn.kill();
        assert!(stubborn_stopped, "a group that ignores SIGTERM outlived the stop");
        assert!(taken_alive, "a group id that read as gone within the grace was signalled once it read as recorded");
    }

    /// A record write can stall, as on a home whose file server stopped
    /// answering. One thread makes every write to the record, so a tool's
    /// spawn and settle, and the exit's kill and its close of the record,
    /// never wait on one, and the exit waits at most `RECORD_CLOSE_WAIT` for
    /// that thread. A FIFO no process reads stands at the record's path, so
    /// the thread's first write, the file's creation, blocks for good. The
    /// kill marks the process as exiting, and the probe exits, so it runs in
    /// a child, which exits with 42 once a tool round and its exit are
    /// through.
    #[cfg(unix)]
    #[test]
    fn a_stalled_record_write_holds_neither_a_tool_nor_the_exit() {
        use std::os::unix::ffi::OsStrExt;
        const CHILD: &str = "OPENMAX_TEST_STALLED_RECORD";
        if let Some(data) = std::env::var_os(CHILD) {
            let data = PathBuf::from(data);
            reap_left_behind_groups(&data);
            let run = data.join("run");
            std::fs::create_dir_all(&run).unwrap();
            let path = run.join(format!("{}.groups", std::process::id()));
            let fifo = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0, "{}", io::Error::last_os_error());
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let round = runtime.block_on(run_process(request("/bin/echo", &[]), Arc::new(CancelToken::default())));
            assert!(round.is_ok(), "the tool round failed");
            kill_process_groups();
            std::process::exit(42);
        }
        let data = std::env::temp_dir().join(format!("omx-stalled-{}", uuid::Uuid::new_v4()));
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "execution::tests::a_stalled_record_write_holds_neither_a_tool_nor_the_exit"])
            .env(CHILD, &data)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let exited = stopped_within(&mut child, Duration::from_secs(10));
        let _ = child.kill();
        let status = child.wait().unwrap();
        let mut stderr = String::new();
        let _ = std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut stderr);
        let _ = std::fs::remove_dir_all(&data);
        assert!(exited, "a tool round or the exit waited on a stalled record write\n{stderr}");
        assert_eq!(status.code(), Some(42), "{status:?}\n{stderr}");
    }

    /// A leader pid that now names a process started at another time is
    /// someone else's: it is not signalled, though its stale record goes.
    /// The record names the bystander's pid with the start time of another
    /// process that really ran, as a group whose leader exited and whose pid
    /// was taken would, so only start times that tell two processes apart
    /// keep it running. Linux counts them in clock ticks, 10 ms each, so the
    /// bystander starts a tick after the other.
    #[cfg(unix)]
    #[test]
    fn a_reused_leader_pid_is_not_signalled() {
        let (mut earlier, earlier_start) = left_behind_group();
        std::thread::sleep(Duration::from_millis(20));
        let (mut bystander, bystander_start) = left_behind_group();
        let run = run_dir("reused-pid");
        let record = run.join(format!("{}.groups", dead_pid()));
        std::fs::write(&record, format!("1\n{} {earlier_start}\n", bystander.id())).unwrap();
        reap(&run);
        // A signalled process can still be exiting once its group reads as
        // gone: give it the time to.
        let alive = !stopped_within(&mut bystander, Duration::from_millis(50));
        let _ = bystander.kill();
        let _ = earlier.kill();
        let record_kept = record.exists();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert!(alive, "a process the record does not name was killed");
        assert!(!record_kept, "the stale record outlived the reap");
        assert_ne!(earlier_start, bystander_start, "two processes started apart read one start time");
    }

    /// Linux start times count clock ticks from boot, so once the machine has
    /// restarted, a pid and start a record names can belong to a new process.
    /// A record whose first line names another boot is removed with nothing
    /// signalled; the same record from this boot is reaped.
    #[cfg(unix)]
    #[test]
    fn a_record_from_another_boot_is_removed_without_signalling() {
        let (mut group, start) = left_behind_group();
        let run = run_dir("boot");
        let record = run.join(format!("{}.groups", dead_pid()));
        std::fs::write(&record, format!("1 an-earlier-boot\n{} {start}\n", group.id())).unwrap();
        reap_in_boot(&run, Some("this-boot"));
        let alive = !stopped_within(&mut group, Duration::from_millis(50));
        let earlier_kept = record.exists();
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(&record, format!("1 this-boot\n{} {start}\n", group.id())).unwrap();
        reap_in_boot(&run, Some("this-boot"));
        let stopped = stopped_within(&mut group, Duration::from_millis(500));
        let _ = group.kill();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert!(alive, "a record from another boot had its group signalled");
        assert!(!earlier_kept, "a record from another boot outlived the reap");
        assert!(stopped, "a record from this boot kept its group running");
    }

    /// The pid-reuse guard rests on a start time being when the process
    /// started. On macOS that is microseconds since the epoch, read here
    /// against the wall clock around the spawn; on Linux, clock ticks after
    /// boot, read against `/proc/uptime`. A constant, or a value in other
    /// units, falls outside the window and would let a process that took a
    /// recorded pid pass as the one recorded.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn a_start_time_is_when_the_process_started() {
        #[cfg(target_os = "macos")]
        let (clock, slack) = (
            || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as u64,
            1_000,
        );
        #[cfg(target_os = "linux")]
        let (clock, slack) = (
            || {
                let uptime = std::fs::read_to_string("/proc/uptime").unwrap();
                let seconds: f64 = uptime.split_whitespace().next().unwrap().parse().unwrap();
                (seconds * unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64) as u64
            },
            2,
        );
        let before = clock();
        let (mut child, start) = left_behind_group();
        let after = clock();
        let _ = child.kill();
        let _ = child.wait();
        assert!(
            before - slack <= start && start <= after + slack,
            "a process started between {before} and {after} read {start}"
        );
    }

    /// kill(-1) signals every process this user may signal, and kill(0)
    /// this launch's own group: no record line can name either as a leader.
    #[cfg(unix)]
    #[test]
    fn a_record_never_names_pid_1_or_0_as_a_leader() {
        for text in ["1", "0", "-1", "-42", "x"] {
            assert_eq!(leader_pid(text), None, "{text} was taken as a group leader");
        }
        assert_eq!(leader_pid("42"), Some(42));
        assert_eq!(positive_pid("0"), None, "0, which kill takes as a group, was taken as one process");
    }

    /// A record whose owner pid runs and started when the record says is a
    /// live openmax's: left alone, groups and file, as is one whose first
    /// line is not written yet. The same pid with another start time is a
    /// process that took a dead harness's pid. The first line is the
    /// owner's start time, then on Linux the boot's id as the kernel gives
    /// it, and the reap reads it back as written.
    #[cfg(unix)]
    #[test]
    fn a_running_owner_keeps_its_record_and_a_reused_owner_pid_does_not() {
        let own = (std::process::id(), start_time(std::process::id()).unwrap());
        let (mut group, start) = left_behind_group();
        let run = run_dir("live-owner");
        let mut record = GroupRecord::new(run.join("sub").join(format!("{}.groups", own.0)), own.1);
        record.add(group.id(), start);
        let text = std::fs::read_to_string(&record.path).unwrap();
        let first = match std::fs::read_to_string("/proc/sys/kernel/random/boot_id") {
            Ok(boot) => format!("{} {}", own.1, boot.trim()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => own.1.to_string(),
            Err(error) => panic!("the boot id: {error}"),
        };
        assert_eq!(text, format!("{first}\n{} {start}\n", group.id()));
        reap(record.path.parent().unwrap());
        assert!(!stopped_within(&mut group, Duration::from_millis(50)), "a live owner's group was stopped");
        assert!(record.path.exists(), "a live owner's record was removed");
        let creating = run.join("new").join(format!("{}.groups", own.0));
        std::fs::create_dir_all(creating.parent().unwrap()).unwrap();
        std::fs::write(&creating, "").unwrap();
        reap(creating.parent().unwrap());
        assert!(creating.exists(), "a record its owner may still be writing was removed");

        let mut reused = GroupRecord::new(run.join(format!("{}.groups", own.0)), own.1 + 1);
        reused.add(group.id(), start);
        reap(&run);
        let stopped = stopped_within(&mut group, Duration::from_millis(500));
        let _ = group.kill();
        assert!(stopped, "a record whose owner pid was reused kept its group running");
        assert!(!reused.path.exists(), "the reused owner's record outlived the reap");
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
    }

    /// The record lists a group from its spawn until it settles. It grows by
    /// a line per change, so a small record is never cut. Past its bound,
    /// the strike that leaves no group listed cuts it back to its first line,
    /// kept whole, and lines written after the cut still read. While a group
    /// is listed it is never cut, and a struck group is never signalled, even
    /// while its leader's pid still names the process it was.
    #[cfg(unix)]
    #[test]
    fn the_record_strikes_settled_groups_and_is_cut_back_once_none_is_listed() {
        let (mut running, running_start) = left_behind_group();
        let (mut settled, settled_start) = left_behind_group();
        let run = run_dir("strike");
        let path = run.join(format!("{}.groups", dead_pid()));
        let mut record = GroupRecord::new(path.clone(), 1);
        let round = "42 7\n-42 7\n".len() as u64;
        let rounds = |record: &mut GroupRecord| {
            for _ in 0..=RECORD_CUT_BYTES / round {
                record.add(42, 7);
                record.strike(42);
            }
        };
        for _ in 0..10 {
            record.add(42, 7);
            record.strike(42);
        }
        let small = std::fs::metadata(&path).unwrap().len();
        let first = first_line(1, boot_id().as_deref()).len() as u64 + 1;
        assert_eq!(small, first + 10 * round, "a small record was cut");
        rounds(&mut record);
        let cut = std::fs::metadata(&path).unwrap().len();
        assert!(cut < RECORD_CUT_BYTES, "a record past its bound with no group listed was not cut: {cut} bytes");
        let kept = std::fs::read_to_string(&path).unwrap();
        let owner = first_line(1, boot_id().as_deref());
        assert_eq!(kept.lines().next(), Some(owner.as_str()), "a cut did not keep the first line whole");
        record.add(running.id(), running_start);
        rounds(&mut record);
        record.add(settled.id(), settled_start);
        record.strike(settled.id());
        reap(&run);
        let running_stopped = stopped_within(&mut running, Duration::from_millis(500));
        let settled_alive = !stopped_within(&mut settled, Duration::from_millis(50));
        let _ = running.kill();
        let _ = settled.kill();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert!(running_stopped, "a group listed past a cut outlived the reap of its dead owner's record");
        assert!(settled_alive, "a group struck from the record was signalled");
    }

    /// A write that fails, as on a full disk, can leave part of a line, and
    /// the next line would run into it and hide both. The next change
    /// rewrites the file whole instead, and later lines append to that.
    #[cfg(unix)]
    #[test]
    fn a_failed_write_is_repaired_by_rewriting_the_record() {
        use std::io::Write;
        let run = run_dir("rewrite");
        let path = run.join("1.groups");
        let mut record = GroupRecord::new(path.clone(), 1);
        record.add(42, 7);
        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(b"4").unwrap();
        // A handle every write fails on.
        record.file = Some(std::fs::File::open(&path).unwrap());
        record.add(43, 8);
        let rewritten = std::fs::read_to_string(&path).unwrap();
        record.strike(42);
        let appended = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        let first = first_line(1, boot_id().as_deref());
        assert_eq!(rewritten, format!("{first}\n42 7\n43 8\n"), "a failed write left the record unreadable");
        assert_eq!(appended, format!("{first}\n42 7\n43 8\n-42 7\n"), "lines after the rewrite");
        assert_eq!(recorded_groups(appended.lines().skip(1)), [(43, 8)]);
    }

    /// At exit a record that lists no group is removed, and the run
    /// directory with the last record in it. One that still lists a group
    /// names a tool no exit path stopped, so it stays for the next launch.
    #[cfg(unix)]
    #[test]
    fn an_exit_removes_the_record_only_once_it_lists_no_group() {
        let run = run_dir("close");
        let listed = run.join("11.groups");
        let mut record = GroupRecord::new(listed.clone(), 1);
        record.add(42, 7);
        close(record);
        let listed_kept = listed.exists();
        let settled = |name: &str| {
            let mut record = GroupRecord::new(run.join(name), 1);
            record.add(42, 7);
            record.strike(42);
            close(record);
            run.join(name).exists()
        };
        let beside_kept = settled("12.groups");
        let run_kept_beside = run.exists();
        let _ = std::fs::remove_file(&listed);
        let last_kept = settled("13.groups");
        let run_kept = run.exists();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert!(listed_kept, "an exit removed a record that still lists a group");
        assert!(!beside_kept, "an exit kept a record that lists no group");
        assert!(run_kept_beside, "an exit removed the run directory with another record in it");
        assert!(!last_kept, "an exit kept the last record");
        assert!(!run_kept, "an exit kept the run directory its last record left empty");
    }

    /// Recording a tool group is harness time in every tool round: reading
    /// the leader's start time and sending its line to the record's thread
    /// at spawn, sending its strike at settle; the thread writes them
    /// meanwhile. The gate is a tenth of the 0.25 ms a whole tool round may
    /// cost. Method: the median of 101 rounds after 20 warmup rounds, on a
    /// record in the temp directory, which the thread then closes, every
    /// change written; a release build asserts the gate, and a debug build
    /// prints the median and checks only 20 times the gate.
    #[cfg(unix)]
    #[test]
    fn recording_a_tool_group_costs_under_a_tenth_of_a_tool_round() {
        let (mut group, _) = left_behind_group();
        let run = run_dir("cost");
        let path = run.join("1.groups");
        let (recorder, changes) = std::sync::mpsc::channel();
        let record = GroupRecord::new(path.clone(), 1);
        let keeper = std::thread::spawn(move || keep_record(record, changes));
        let pid = group.id();
        let round = || {
            let began = std::time::Instant::now();
            let start = start_time(pid).unwrap();
            recorder.send(RecordChange::Add(pid, start)).unwrap();
            recorder.send(RecordChange::Strike(pid)).unwrap();
            began.elapsed()
        };
        for _ in 0..20 {
            round();
        }
        let mut rounds: Vec<Duration> = (0..101).map(|_| round()).collect();
        rounds.sort();
        let median = rounds[rounds.len() / 2];
        let (closed, done) = std::sync::mpsc::channel();
        recorder.send(RecordChange::Close(closed)).unwrap();
        done.recv().unwrap();
        keeper.join().unwrap();
        let kept = path.exists();
        let _ = group.kill();
        let _ = std::fs::remove_dir_all(run.parent().unwrap());
        assert!(!kept, "the record's thread did not keep up with every change and remove the record");
        let gate = Duration::from_micros(25);
        eprintln!("recording a tool group: median {median:?} over {} rounds, gate {gate:?}", rounds.len());
        let bound = if cfg!(debug_assertions) { gate * 20 } else { gate };
        assert!(median < bound, "recording a tool group took {median:?} a round, over {bound:?}");
    }

    /// The human attestation never reaches an agent-spawned child, even when
    /// the harness itself was launched under one: otherwise a bash call
    /// could unset the session marker and inherit a ready-made bypass.
    #[tokio::test]
    async fn spawned_processes_never_inherit_the_human_attestation() {
        std::env::set_var("OPENMAX_HUMAN_ATTEST", "1");
        let request = ProcessRequest {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "printf '[%s]' \"$OPENMAX_HUMAN_ATTEST\"".into()],
            cwd: std::env::temp_dir(),
            stdin: StdinMode::Null,
            timeout: Duration::from_secs(5),
            capture: CaptureSpec { head_bytes: 64, tail_bytes: 0, spill_dir: None, spill_bytes_per_stream: 0, redraws: false },
            sandbox: None,
            env_allowlist: None,
            self_link: false,
            env: Vec::new(),
        };
        let output = run_process(request, Arc::new(CancelToken::default())).await.unwrap();
        std::env::remove_var("OPENMAX_HUMAN_ATTEST");
        assert_eq!(String::from_utf8_lossy(&output.stdout.head), "[]");
    }

    /// Every unsandboxed child learns which binary is running the session:
    /// `$OPENMAX_BIN` names this executable, so `openmax --check` in a tool
    /// or bash can be pinned to the same build. A sandboxed probe gets no
    /// such handle.
    #[tokio::test]
    async fn spawned_processes_learn_the_hosting_binary() {
        let request = ProcessRequest {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "printf %s \"$OPENMAX_BIN\"".into()],
            cwd: std::env::temp_dir(),
            stdin: StdinMode::Null,
            timeout: Duration::from_secs(5),
            capture: CaptureSpec { head_bytes: 4096, tail_bytes: 0, spill_dir: None, spill_bytes_per_stream: 0, redraws: false },
            sandbox: None,
            env_allowlist: None,
            self_link: false,
            env: Vec::new(),
        };
        let output = run_process(request, Arc::new(CancelToken::default())).await.unwrap();
        let got = String::from_utf8_lossy(&output.stdout.head).to_string();
        let expected = std::env::current_exe().unwrap().to_string_lossy().to_string();
        assert_eq!(got, expected, "OPENMAX_BIN must name the running executable");
    }

    /// The self link directory is shared and never cleaned up, so every
    /// spawn re-checks it: a replaced link is put back, anything planted
    /// beside it is removed, a deleted build keeps the directory off PATH
    /// instead of leaving a dangling first entry, a directory another account
    /// can write is never used, and a nested harness does not stack a second
    /// copy of the entry.
    #[cfg(unix)]
    #[test]
    fn the_self_link_is_rechecked_and_never_left_dangling_on_path() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("omx-selflink-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let exe = base.join("build");
        std::fs::write(&exe, "#!/bin/sh\n").unwrap();
        let dir = base.join("links");
        let link = dir.join("openmax");
        ensure_self_link(&dir, &exe).expect("a fresh directory gets the link");
        assert_eq!(std::fs::read_link(&link).unwrap(), exe);

        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(base.join("elsewhere"), &link).unwrap();
        ensure_self_link(&dir, &exe).expect("a replaced link is put back");
        assert_eq!(std::fs::read_link(&link).unwrap(), exe);

        // Only the link and a concurrent harness's staged link may stay.
        let staged = dir.join(format!(".openmax-{}", uuid::Uuid::new_v4()));
        std::os::unix::fs::symlink(&exe, &staged).unwrap();
        std::fs::write(dir.join("python3"), "#!/bin/sh\n").unwrap();
        std::fs::create_dir_all(dir.join("bin/nested")).unwrap();
        ensure_self_link(&dir, &exe).expect("planted entries are cleared");
        let mut left: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        left.sort();
        assert_eq!(left, vec![staged.file_name().unwrap().to_os_string(), "openmax".into()]);

        std::fs::remove_file(&exe).unwrap();
        assert!(ensure_self_link(&dir, &exe).is_err(), "a link to a deleted build must stay off PATH");

        std::fs::write(&exe, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(ensure_self_link(&dir, &exe).is_err(), "a directory others can write is never used");

        let inherited = std::env::join_paths([dir.clone(), "/usr/bin".into(), dir.clone()]).unwrap();
        let path = path_with_self_link(&dir, &inherited).unwrap();
        assert_eq!(std::env::split_paths(&path).collect::<Vec<_>>(), vec![dir.clone(), "/usr/bin".into()]);
        let _ = std::fs::remove_dir_all(base);
    }

    /// The agent's shell gets the self link directory first on PATH, and the
    /// agent's own account can write it: a `python3` planted there by one
    /// command would replace the system binary in every later one, including
    /// a command a human read and approved. So each such spawn clears
    /// everything but the link first.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_name_planted_beside_the_self_link_never_runs() {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let dir = self_link_dir().expect("this host can place the self link");
        let name = format!("omx-planted-{}", uuid::Uuid::new_v4().simple());
        let planted = dir.join(&name);
        // Created executable in one step: a concurrent spawn may clear the
        // name at any moment, and a separate chmod would then fail.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o755)
            .open(&planted)
            .unwrap();
        file.write_all(b"#!/bin/sh\necho planted-ran\n").unwrap();
        drop(file);
        let mut request = request(&name, &[]);
        request.capture.head_bytes = 4096;
        request.self_link = true;
        let ran = run_process(request, Arc::new(CancelToken::default())).await;
        let _ = std::fs::remove_file(&planted);
        let stdout = ran
            .map(|output| String::from_utf8_lossy(&output.stdout.head).into_owned())
            .unwrap_or_default();
        assert!(!stdout.contains("planted-ran"), "a name planted beside the self link ran");
    }

    /// A bash heredoc (and process substitution) writes a temp file; a
    /// probe whose sandbox denies /tmp turned that into a false negative on
    /// correct code. TMPDIR is the scratch, so ordinary shell idioms work.
    #[tokio::test]
    async fn a_sandboxed_probe_can_use_bash_heredocs() {
        let base = std::env::temp_dir().join(format!("omx-sbx-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let script = "bash -c 'cat <<EOF\nheredoc-ok\nEOF\n' && printf %s \"$TMPDIR\"";
        let Some(output) = run_sandboxed(sandboxed_request(&base, script)).await else {
            return;
        };
        let stdout = String::from_utf8_lossy(&output.stdout.head).to_string();
        assert!(stdout.contains("heredoc-ok"), "heredoc must work in the probe: {stdout} / {}", String::from_utf8_lossy(&output.stderr.head));
        assert!(
            stdout.contains(base.file_name().unwrap().to_str().unwrap()),
            "TMPDIR points into the scratch: {stdout}"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    fn sandboxed_request(scratch: &Path, script: &str) -> ProcessRequest {
        ProcessRequest {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            cwd: scratch.to_path_buf(),
            stdin: StdinMode::Null,
            timeout: Duration::from_secs(10),
            capture: CaptureSpec {
                head_bytes: 4096,
                tail_bytes: 4096,
                spill_dir: None,
                spill_bytes_per_stream: 0,
                redraws: false,
            },
            sandbox: Some(SandboxPolicy { rw_scratch: scratch.to_path_buf() }),
            env_allowlist: None,
            self_link: false,
            env: Vec::new(),
        }
    }

    /// Run a sandboxed request, or - on a host with no backend - assert the
    /// fail-closed error and skip the behavioral half. Never silently green.
    async fn run_sandboxed(request: ProcessRequest) -> Option<ProcessOutput> {
        match run_process(request, Arc::new(CancelToken::default())).await {
            Ok(output) => Some(output),
            Err(ProcessError::SandboxUnavailable(reason)) => {
                eprintln!("sandbox backend unavailable here ({reason}); fail-closed path verified");
                None
            }
            Err(other) => panic!("sandboxed spawn failed unexpectedly: {other}"),
        }
    }

    /// The probe's write authority ends at its scratch: an escape attempt
    /// fails and leaves nothing behind, while a scratch write succeeds.
    #[tokio::test]
    async fn a_sandboxed_probe_writes_only_inside_its_scratch() {
        let base = std::env::temp_dir().join(format!("omx-sbx-{}", uuid::Uuid::new_v4()));
        let scratch = base.join("scratch");
        let outside = base.join("outside");
        std::fs::create_dir_all(&scratch).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let script = format!(
            "echo probed > inside.txt; echo escaped > {}/escape.txt",
            outside.display()
        );
        let Some(output) = run_sandboxed(sandboxed_request(&scratch, &script)).await else {
            return;
        };
        assert!(scratch.join("inside.txt").exists(), "scratch write must succeed");
        assert!(
            !outside.join("escape.txt").exists(),
            "a write outside the scratch must be denied"
        );
        match output.termination {
            Termination::Exited(status) => assert!(!status.success(), "the escape must fail loudly"),
            _ => panic!("probe should run to completion"),
        }
        let _ = std::fs::remove_dir_all(base);
    }

    /// The probe has no network: a loopback connect to a live listener fails
    /// under the sandbox. The listener is provably alive (bound in-process),
    /// so a pass can only mean the sandbox cut the network.
    #[tokio::test]
    async fn a_sandboxed_probe_has_no_network() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let base = std::env::temp_dir().join(format!("omx-sbx-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let script = format!(
            "python3 -c 'import socket; socket.create_connection((\"127.0.0.1\", {port}), 2); print(\"CONNECTED\")'"
        );
        let Some(output) = run_sandboxed(sandboxed_request(&base, &script)).await else {
            return;
        };
        let stdout = String::from_utf8_lossy(&output.stdout.head).to_string();
        assert!(
            !stdout.contains("CONNECTED"),
            "a sandboxed probe reached the network: {stdout}"
        );
        drop(listener);
        let _ = std::fs::remove_dir_all(base);
    }

    /// The probe's environment is scrubbed - a secret in the parent env never
    /// reaches it - while the agent-spawned marker survives the scrub and
    /// HOME points into the scratch.
    #[tokio::test]
    async fn a_sandboxed_probe_env_is_scrubbed() {
        // Set on the test process; an unsandboxed child would inherit it.
        std::env::set_var("OPENMAX_TEST_SECRET_ZZZ", "leak");
        let base = std::env::temp_dir().join(format!("omx-sbx-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let script = "printf '%s|%s|%s' \"$OPENMAX_TEST_SECRET_ZZZ\" \"$OPENMAX_SESSION\" \"$HOME\"";
        let Some(output) = run_sandboxed(sandboxed_request(&base, script)).await else {
            std::env::remove_var("OPENMAX_TEST_SECRET_ZZZ");
            return;
        };
        std::env::remove_var("OPENMAX_TEST_SECRET_ZZZ");
        let stdout = String::from_utf8_lossy(&output.stdout.head).to_string();
        let parts: Vec<&str> = stdout.split('|').collect();
        assert_eq!(parts[0], "", "parent env must not leak into a probe: {stdout}");
        assert_eq!(parts[1], "1", "the session marker survives the scrub: {stdout}");
        assert!(
            parts[2].contains(base.file_name().unwrap().to_str().unwrap()),
            "HOME points into the scratch: {stdout}"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn captures_head_tail_and_both_streams() {
        let output = run_process(
            request("/bin/sh", &["-c", "printf abcdefghij; printf KLMNOP 1>&2"]),
            Arc::new(CancelToken::default()),
        )
        .await
        .unwrap();
        assert!(matches!(output.termination, Termination::Exited(status) if status.success()));
        assert_eq!(output.stdout.total_bytes, 10);
        assert_eq!(output.stdout.head, b"abcd");
        assert_eq!(output.stdout.tail, b"ghij");
        assert_eq!(output.stderr.head, b"KLMN");
        assert_eq!(output.stderr.tail, b"MNOP");
    }

    #[test]
    fn reconstructs_overlapping_head_and_tail_without_duplication() {
        let stream = CapturedStream {
            total_bytes: 5,
            head: b"abcd".to_vec(),
            tail: b"bcde".to_vec(),
        };
        assert_eq!(stream.rendered_bytes(), b"abcde");
    }

    /// A pipe read ends wherever the kernel cut the output, so a redraw has
    /// to read the same whatever the cut: a `\r` that ends one read and the
    /// `\n` that starts the next are still one `\r\n` line ending (as is a
    /// run of `\r` before a `\n`), a frame split over two reads is still one
    /// frame, a `\r` at the end of the stream is dropped, and a line past the
    /// 8 KiB limit is passed on whole, with the `\r` that ends it, however
    /// its reads were cut.
    #[test]
    fn a_redraw_reads_the_same_wherever_a_read_ends() {
        let input = b"get 1%\rget 50%\rget 100%\nkept\r\ntwo\r\r\n\r\nx\r\ry\rlast\r";
        let shown = b"get 100%\nkept\r\ntwo\r\n\r\nlast";
        let run = |reads: &[&[u8]]| {
            let (mut redraws, mut out) = (Redraws::default(), Vec::new());
            for read in reads {
                redraws.push(read, &mut out);
            }
            out.extend(redraws.finish());
            out
        };
        for cut in 0..=input.len() {
            assert_eq!(run(&[&input[..cut], &input[cut..]]), shown, "cut at {cut}");
        }
        let bytes: Vec<&[u8]> = input.chunks(1).collect();
        assert_eq!(run(&bytes), shown);

        // Text up to the 8 KiB the configuration docs state can still be
        // replaced; past it, it is passed on as it arrives instead of held
        // without bound, and the `\r` after it stays between it and the next
        // frame instead of gluing the two together.
        let held = "h".repeat(8 * 1024);
        assert_eq!(run(&[format!("{held}\rnext\n").as_bytes()]), b"next\n");
        let long = "l".repeat(8 * 1024 + 1);
        assert_eq!(run(&[format!("{long}\rnext\n").as_bytes()]), format!("{long}\rnext\n").as_bytes());

        // A line longer than the limit keeps all of it whatever the reads,
        // including reads of the capture's own size; one `\r` follows it
        // however many came, and the next redraw or line is held again.
        let (l, m) = ("L".repeat(20_000), "M".repeat(9_000));
        let input = format!("{l}\r\rmid\rnext\n{m}\nshort\rlast\n");
        let shown = format!("{l}\rnext\n{m}\nlast\n");
        assert_eq!(run(&[input.as_bytes()]), shown.as_bytes());
        for size in [1, 7, 4096, READ_CHUNK_BYTES, READ_CHUNK_BYTES + 1] {
            let reads: Vec<&[u8]> = input.as_bytes().chunks(size).collect();
            assert_eq!(run(&reads), shown.as_bytes(), "reads of {size}");
        }
    }

    /// A short timeout races process startup on a loaded runner (a 25ms
    /// window flaked the head assertion on macOS), so the command marks that
    /// it wrote, and a run stopped before that is tried with a longer one.
    #[tokio::test]
    async fn times_out_and_drains_output() {
        let wrote = std::env::temp_dir().join(format!("openmax-wrote-{}", uuid::Uuid::new_v4()));
        let script = format!("printf before; : > '{}'; sleep 10", wrote.display());
        let mut output = None;
        for second in TRIAL_SECONDS {
            let mut request = request("/bin/sh", &["-c", &script]);
            request.timeout = second / 2;
            output = Some(run_process(request, Arc::new(CancelToken::default())).await.unwrap());
            if wrote.exists() {
                break;
            }
        }
        let output = output.unwrap();
        assert!(wrote.exists(), "the command never wrote before its timeout");
        let _ = std::fs::remove_file(&wrote);
        assert!(matches!(output.termination, Termination::TimedOut));
        assert_eq!(output.stdout.head, b"befo");
    }

    #[tokio::test]
    async fn cancellation_terminates_the_child() {
        let cancel = Arc::new(CancelToken::default());
        let pending = run_process(request("/bin/sh", &["-c", "sleep 10"]), cancel.clone());
        let cancel_task = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            cancel.cancel();
        });
        let output = pending.await.unwrap();
        cancel_task.await.unwrap();
        assert!(matches!(output.termination, Termination::Cancelled));
    }

    #[tokio::test]
    async fn writes_stdin_and_drains_concurrent_floods() {
        let mut stdin_request = request("/bin/sh", &["-c", "cat"]);
        stdin_request.stdin = StdinMode::Bytes(b"hello stdin".to_vec());
        stdin_request.capture.head_bytes = 32;
        stdin_request.capture.tail_bytes = 32;
        let stdin_output = run_process(stdin_request, Arc::new(CancelToken::default()))
            .await
            .unwrap();
        assert_eq!(stdin_output.stdout.rendered_bytes(), b"hello stdin");

        let mut flood = request(
            "/bin/sh",
            &[
                "-c",
                "yes x | head -c 524288 & yes y | head -c 524288 >&2 & wait",
            ],
        );
        flood.timeout = Duration::from_secs(5);
        flood.capture.head_bytes = 128;
        flood.capture.tail_bytes = 128;
        let output = run_process(flood, Arc::new(CancelToken::default()))
            .await
            .unwrap();
        assert!(matches!(output.termination, Termination::Exited(status) if status.success()));
        assert_eq!(output.stdout.total_bytes, 512 * 1024);
        assert_eq!(output.stderr.total_bytes, 512 * 1024);
        assert!(output.stdout.head.len() <= 128);
        assert!(output.stderr.tail.len() <= 128);
    }

    #[tokio::test]
    async fn spills_lazily_and_bounds_the_log() {
        let dir = std::env::temp_dir().join(format!("openmax-execution-{}", uuid::Uuid::new_v4()));
        let mut request = request("/bin/sh", &["-c", "printf 1234567890"]);
        request.capture.spill_dir = Some(dir.clone());
        request.capture.spill_bytes_per_stream = 6;
        let output = run_process(request, Arc::new(CancelToken::default()))
            .await
            .unwrap();
        let path = output.log_path.unwrap();
        let log = tokio::fs::read(&path).await.unwrap();
        assert!(output.log_truncated);
        assert!(String::from_utf8_lossy(&log).contains("4 bytes omitted"));
        assert!(log.starts_with(b"123456"));
        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    /// Unpruned, this directory grows without bound: tens of megabytes of
    /// spill logs accumulate in a month because nothing ever deletes them. Writing a new log is the moment the
    /// directory grows, so it is also the moment old logs age out; files the
    /// harness did not write are never touched.
    #[cfg(unix)]
    #[tokio::test]
    async fn writing_a_spill_log_prunes_the_aged_ones() {
        fn age_by_eight_days(path: &Path) {
            use std::os::unix::ffi::OsStrExt;
            let c = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
            let old = unsafe { libc::time(std::ptr::null_mut()) } - 8 * 24 * 60 * 60;
            let times =
                [libc::timeval { tv_sec: old, tv_usec: 0 }, libc::timeval { tv_sec: old, tv_usec: 0 }];
            assert_eq!(unsafe { libc::utimes(c.as_ptr(), times.as_ptr()) }, 0);
        }

        let dir = std::env::temp_dir().join(format!("openmax-prune-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let aged_log = dir.join("cmd-ancient.log");
        let aged_orphan = dir.join(".openmax-stdout-orphan.tmp");
        let not_ours = dir.join("keepme.txt");
        let fresh_log = dir.join("cmd-fresh.log");
        for path in [&aged_log, &aged_orphan, &not_ours, &fresh_log] {
            tokio::fs::write(path, b"x").await.unwrap();
        }
        age_by_eight_days(&aged_log);
        age_by_eight_days(&aged_orphan);
        age_by_eight_days(&not_ours);

        let mut request = request("/bin/sh", &["-c", "printf 1234567890"]);
        request.capture.spill_dir = Some(dir.clone());
        request.capture.spill_bytes_per_stream = 6;
        let output = run_process(request, Arc::new(CancelToken::default()))
            .await
            .unwrap();
        assert!(output.log_path.is_some(), "the command itself must still spill");

        assert!(!aged_log.exists(), "an aged log must be pruned");
        assert!(!aged_orphan.exists(), "an aged orphaned spill must be pruned");
        assert!(not_ours.exists(), "files the harness did not write are never touched");
        assert!(fresh_log.exists(), "fresh logs are kept");
        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn stderr_only_spill_creates_a_combined_log() {
        let dir = std::env::temp_dir().join(format!("openmax-execution-{}", uuid::Uuid::new_v4()));
        let mut request = request("/bin/sh", &["-c", "printf stdout; printf 1234567890 >&2"]);
        request.capture.spill_dir = Some(dir.clone());
        let output = run_process(request, Arc::new(CancelToken::default()))
            .await
            .unwrap();
        let path = output.log_path.expect("stderr spill must be retained");
        let log = tokio::fs::read(&path).await.unwrap();
        assert!(log.starts_with(b"stdout\n[stderr]\n1234567890"));
        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn combined_overflow_creates_log_when_each_stream_fits() {
        let dir = std::env::temp_dir().join(format!("openmax-execution-{}", uuid::Uuid::new_v4()));
        let mut request = request("/bin/sh", &["-c", "printf 1234; printf 5678 >&2"]);
        request.capture.head_bytes = 0;
        request.capture.tail_bytes = 6;
        request.capture.spill_dir = Some(dir.clone());
        let output = run_process(request, Arc::new(CancelToken::default()))
            .await
            .unwrap();
        let path = output
            .log_path
            .expect("combined overflow must remain inspectable");
        let log = tokio::fs::read(path).await.unwrap();
        assert_eq!(log, b"1234\n[stderr]\n5678");
        let _ = tokio::fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn drain_stop_returns_partial_capture_for_an_open_pipe() {
        let (mut writer, reader) = tokio::io::duplex(64);
        let stop = Arc::new(CancelToken::default());
        let task = tokio::spawn(drain_stream(
            reader,
            CaptureSpec {
                head_bytes: 16,
                tail_bytes: 16,
                spill_dir: None,
                spill_bytes_per_stream: 0,
                redraws: false,
            },
            "stdout",
            stop.clone(),
        ));
        writer.write_all(b"partial").await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        stop.cancel();
        let captured = task.await.unwrap().unwrap();
        assert_eq!(captured.stream.rendered_bytes(), b"partial");
    }

    #[tokio::test(start_paused = true)]
    async fn blocked_stdin_writer_has_bounded_shutdown() {
        let (mut writer, _reader) = tokio::io::duplex(1);
        let task = tokio::spawn(async move { writer.write_all(&vec![b'x'; 64 * 1024]).await });
        tokio::time::timeout(Duration::from_secs(1), finish_stdin_task(task))
            .await
            .expect("stdin writer shutdown exceeded its bound");
    }

    #[tokio::test]
    async fn inaccessible_spill_directory_does_not_fail_the_command() {
        let mut request = request("/bin/sh", &["-c", "printf 1234567890"]);
        request.capture.spill_dir = Some(PathBuf::from("/proc/openmax-cmd-logs"));
        let output = run_process(request, Arc::new(CancelToken::default()))
            .await
            .unwrap();
        assert!(matches!(output.termination, Termination::Exited(status) if status.success()));
        assert!(output.log_path.is_none());
        assert_eq!(output.stdout.tail, b"7890");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cleans_background_descendants_after_normal_exit() {
        let mut lifeline = Lifeline::new();
        let pid_file =
            std::env::temp_dir().join(format!("openmax-descendant-{}", uuid::Uuid::new_v4()));
        let script = format!("{}; sleep 6844 & exec 3>&-; echo $! > '{}'", lifeline.open_in_shell(), pid_file.display());
        let output = run_process(
            request("/bin/sh", &["-c", &script]),
            Arc::new(CancelToken::default()),
        )
        .await
        .unwrap();
        assert!(matches!(output.termination, Termination::Exited(status) if status.success()));
        let pid = recorded_pid(&pid_file);
        let _ = std::fs::remove_file(&pid_file);
        assert!(!lifeline.survived(pid), "background descendant survived process cleanup");
    }

    /// A process group whose leader has exited and left `member` running in
    /// it: the shape a command leaves behind when it backgrounds a child. The
    /// member is the test's own child, so the test reaps it as it exits, and
    /// a host whose init does not reap orphans cannot keep it in the group as
    /// a zombie.
    #[cfg(unix)]
    async fn group_whose_leader_exited(member: &[&str]) -> (Option<u32>, Child) {
        use std::os::unix::process::CommandExt;
        let mut leader = Command::new("sleep");
        leader.arg("6846").kill_on_drop(true);
        leader.as_std_mut().process_group(0);
        let mut leader = leader.spawn().unwrap();
        let group = leader.id().unwrap();
        let mut command = Command::new(member[0]);
        command
            .args(&member[1..])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        command.as_std_mut().process_group(i32::try_from(group).unwrap());
        let member = command.spawn().unwrap();
        leader.start_kill().unwrap();
        leader.wait().await.unwrap();
        (Some(group), member)
    }

    /// A background child that exits on SIGTERM ends the cleanup as soon as
    /// it is gone. A fixed wait held every call that left one behind for the
    /// whole grace, though such a child is gone within milliseconds. Only the
    /// cleanup is timed, from a group whose leader has already exited: on a
    /// loaded machine, starting a shell alone can take longer than the grace,
    /// and that is not the wait. The fastest of five runs is the one measured.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_background_child_that_stops_on_sigterm_does_not_hold_the_call() {
        use std::os::unix::process::ExitStatusExt;
        let mut fastest = Duration::MAX;
        for _ in 0..5 {
            let (group, mut member) = group_whose_leader_exited(&["sleep", "6845"]).await;
            let started = std::time::Instant::now();
            let (found, exited) = tokio::join!(terminate_remaining_group(group), member.wait());
            fastest = fastest.min(started.elapsed());
            assert!(found, "the background child was not found");
            assert_eq!(exited.unwrap().signal(), Some(libc::SIGTERM), "the background child did not stop on SIGTERM");
            if fastest < TERMINATION_GRACE {
                break;
            }
        }
        assert!(fastest < TERMINATION_GRACE, "the cleanup waited {fastest:?} for a child SIGTERM stopped at once");
    }

    /// A background child that is still there after SIGTERM, cleaning up or
    /// ignoring it, gets the whole grace: the group's SIGKILL waits for the
    /// deadline, not for the first checks that find the child still there.
    /// The child ignores SIGTERM and says so before the cleanup starts, and
    /// the time is a lower bound, which a loaded machine only lengthens.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_background_child_that_outlasts_sigterm_gets_the_whole_grace() {
        use std::os::unix::process::ExitStatusExt;
        let (group, mut member) =
            group_whose_leader_exited(&["/bin/sh", "-c", "trap '' TERM; echo ready; exec sleep 6847"]).await;
        let mut ready = [0; 1];
        let said = member.stdout.take().unwrap().read(&mut ready).await.unwrap();
        assert_eq!(said, 1, "the background child never set its trap");
        let started = std::time::Instant::now();
        assert!(terminate_remaining_group(group).await, "the background child was not found");
        let waited = started.elapsed();
        let exited = tokio::time::timeout(Duration::from_secs(5), member.wait())
            .await
            .expect("the group was never killed")
            .unwrap();
        assert!(waited >= TERMINATION_GRACE, "the group was killed {waited:?} into its {TERMINATION_GRACE:?} grace");
        assert_eq!(exited.signal(), Some(libc::SIGKILL), "the background child was not killed at the deadline");
    }

    /// The pid a command wrote to `path` before the call returned.
    #[cfg(unix)]
    fn recorded_pid(path: &Path) -> libc::pid_t {
        let recorded = std::fs::read_to_string(path).unwrap_or_default();
        recorded.trim().parse().unwrap_or_else(|_| panic!("no pid was recorded: {recorded:?}"))
    }

    /// A FIFO that the processes a test command starts hold open for
    /// writing, on descriptor 3 (see [`Lifeline::open_in_shell`]). Its read
    /// end reports end of file once every one of them has exited, so a test
    /// sees them end without probing a pid, which a reaped process gives up
    /// for reuse, and without counting a zombie that no init reaps, which
    /// holds no descriptors, as a survivor.
    #[cfg(unix)]
    struct Lifeline {
        path: PathBuf,
        reader: std::fs::File,
    }

    #[cfg(unix)]
    impl Lifeline {
        fn new() -> Self {
            use std::os::unix::{ffi::OsStrExt, fs::OpenOptionsExt};
            let path = std::env::temp_dir().join(format!("openmax-lifeline-{}", uuid::Uuid::new_v4()));
            let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
            // SAFETY: mkfifo reads one NUL-terminated path.
            assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0, "mkfifo: {}", std::io::Error::last_os_error());
            // Opened first: a shell's open for writing waits for a reader.
            let reader = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(&path).unwrap();
            Self { path, reader }
        }

        /// The shell command that opens the write end as descriptor 3, which
        /// every process the shell starts after it inherits. The script
        /// closes it (`exec 3>&-`) once it has started the processes the test
        /// follows.
        fn open_in_shell(&self) -> String {
            format!("exec 3> '{}'", self.path.display())
        }

        /// Whether a process still holds the write end two seconds on. One
        /// that was stopped lets go as it exits, so the answer comes at once
        /// unless one survived. The survivor is the process recorded as
        /// `pid`, still running, so the pid is still its own: it is killed,
        /// and a failing test leaks nothing.
        fn survived(&mut self, pid: libc::pid_t) -> bool {
            use std::io::Read;
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            let mut buf = [0; 64];
            loop {
                match self.reader.read(&mut buf) {
                    Ok(0) => return false,
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(error) => panic!("reading the lifeline: {error}"),
                }
                if std::time::Instant::now() >= deadline {
                    // SAFETY: kill only sends a signal.
                    unsafe { libc::kill(pid, libc::SIGKILL) };
                    return true;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    #[cfg(unix)]
    impl Drop for Lifeline {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// A timeout stops everything the command started, not just the shell:
    /// a background descendant that ignores SIGTERM is still killed once the
    /// grace runs out, so nothing keeps running after the call reported its
    /// timeout.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_timeout_stops_every_process_the_command_started() {
        let pid_file = std::env::temp_dir().join(format!("openmax-tree-{}", uuid::Uuid::new_v4()));
        // The descendant ignores SIGTERM and records its pid once it does;
        // the shell waits for that, so the timeout stops a tree and not a
        // lone shell. A run stopped before then is tried with a longer one.
        let mut tried = None;
        for second in TRIAL_SECONDS {
            let _ = std::fs::remove_file(&pid_file);
            let lifeline = Lifeline::new();
            let script = format!(
                "{}; sh -c \"trap '' TERM; echo \\$\\$ > '{1}'; exec sleep 6843\" & exec 3>&-; while [ ! -s '{1}' ]; do sleep 0.01; done; sleep 30",
                lifeline.open_in_shell(),
                pid_file.display()
            );
            let mut request = request("/bin/sh", &["-c", &script]);
            request.timeout = second;
            let output = run_process(request, Arc::new(CancelToken::default())).await.unwrap();
            tried = Some((output, lifeline));
            if std::fs::read_to_string(&pid_file).is_ok_and(|pid| pid.ends_with('\n')) {
                break;
            }
        }
        let (output, mut lifeline) = tried.unwrap();
        assert!(matches!(output.termination, Termination::TimedOut));
        let pid = recorded_pid(&pid_file);
        let _ = std::fs::remove_file(&pid_file);
        assert!(!lifeline.survived(pid), "a descendant outlived the timeout that stopped its command");
    }

    /// A command dropped mid-run (the runtime shutting down under a TUI that
    /// quit, an aborted task) stops its whole process group. `kill_on_drop`
    /// reaches only the shell, so a background child kept running with
    /// nobody left to stop it.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_dropped_command_stops_its_whole_process_group() {
        let pid_file = std::env::temp_dir().join(format!("openmax-dropped-{}", uuid::Uuid::new_v4()));
        let script = format!("sleep 6842 & echo $! > '{}'; wait", pid_file.display());
        let mut request = request("/bin/sh", &["-c", &script]);
        request.timeout = Duration::from_secs(30);
        let running = tokio::spawn(run_process(request, Arc::new(CancelToken::default())));
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let pid: libc::pid_t = loop {
            let recorded = std::fs::read_to_string(&pid_file).unwrap_or_default();
            if let Ok(pid) = recorded.trim().parse() {
                break pid;
            }
            assert!(std::time::Instant::now() < deadline, "the background child never started");
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        running.abort();
        let _ = running.await;
        let alive = || unsafe { libc::kill(pid, 0) == 0 };
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while alive() && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let survived = alive();
        if survived {
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
        let _ = std::fs::remove_file(&pid_file);
        assert!(!survived, "a background child outlived the command that started it");
    }

    /// The kill an exit makes waits for a command caught between its spawn
    /// and its registration, but only for `EXIT_SPAWN_WAIT`: a child stalled
    /// before exec (a hung mount, a slow exec check) must not hold a second
    /// signal's exit. A command whose spawn begins after the kill is
    /// refused. This does not show that the held command is stopped, and it
    /// is not: its spawn outlasts the wait, so it runs once released, as it
    /// would after a real exit. The kill marks the process as exiting, so
    /// the probe runs in a child.
    #[cfg(unix)]
    #[test]
    fn the_exit_kill_waits_a_bounded_time_for_a_command_caught_mid_spawn() {
        const CHILD: &str = "OPENMAX_TEST_KILL_MID_SPAWN";
        if std::env::var_os(CHILD).is_some() {
            let spawning = PausedSpawn::start();
            let started = std::time::Instant::now();
            kill_process_groups();
            let waited = started.elapsed();
            assert!(waited >= EXIT_SPAWN_WAIT, "the kill did not wait for the spawn ({waited:?})");
            assert!(waited < EXIT_SPAWN_WAIT * 8, "the kill waited on a stalled spawn ({waited:?})");
            drop(spawning);
            let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let late = runtime.block_on(run_process(request("/bin/echo", &[]), Arc::new(CancelToken::default())));
            assert!(matches!(late, Err(ProcessError::Spawn(_))), "a command started after the kill");
            return;
        }
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "execution::tests::the_exit_kill_waits_a_bounded_time_for_a_command_caught_mid_spawn"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&child.stdout);
        // "1 passed" rules out a filter that matched nothing and exited 0.
        assert!(
            child.status.success() && stdout.contains("1 passed"),
            "{:?}: {stdout}{}",
            child.status,
            String::from_utf8_lossy(&child.stderr)
        );
    }

    /// A command that prompts on the terminal (a git credential prompt, an
    /// ssh host-key question, sudo) has nobody to answer it: the TUI owns the
    /// screen and the keyboard. A child in the harness's session could open
    /// that terminal, so the prompt drew over the TUI, and the read stopped
    /// the child (a background group may not read its terminal) until the
    /// call timed out. A child that leads its own session has no terminal to
    /// open: the command fails at once, and nothing reaches the screen.
    ///
    /// The harness side must hold a controlling terminal for this to prove
    /// anything, and the shared test process must not be given one, so the
    /// probe runs in a child whose terminal is a fresh pseudo-terminal.
    #[cfg(unix)]
    #[test]
    fn a_command_that_prompts_on_the_terminal_fails_at_once() {
        use std::io::Read;
        use std::os::unix::process::CommandExt;
        const CHILD: &str = "OPENMAX_TEST_TERMINAL_PROMPT";
        if std::env::var_os(CHILD).is_some() {
            assert!(
                std::fs::File::open("/dev/tty").is_ok(),
                "the harness side must hold a controlling terminal"
            );
            let mut request = request(
                "/bin/sh",
                &["-c", "printf 'Password: ' > /dev/tty && read -r answer < /dev/tty"],
            );
            request.timeout = Duration::from_secs(10);
            let runtime =
                tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let output = runtime
                .block_on(run_process(request, Arc::new(CancelToken::default())))
                .unwrap();
            assert!(
                matches!(output.termination, Termination::Exited(status) if !status.success()),
                "a command that prompted on the terminal waited there until its timeout"
            );
            return;
        }

        let (mut controller, terminal) = pseudo_terminal();
        let mut probe = std::process::Command::new(std::env::current_exe().unwrap());
        probe
            .args(["--exact", "execution::tests::a_command_that_prompts_on_the_terminal_fails_at_once"])
            .env(CHILD, "1")
            .stdin(terminal)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        // A new session whose controlling terminal is the pseudo-terminal on
        // stdin: what a TUI started from an interactive shell holds.
        unsafe {
            probe.pre_exec(|| {
                if libc::setsid() == -1 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = probe.spawn().unwrap();
        // Only the probe holds the terminal end now, so the controller reads
        // end-of-file once it exits. It is read while the probe runs: a
        // session leader's exit can wait for its terminal's output to drain.
        drop(probe);
        let (sent, screen) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut chunk = [0_u8; 1024];
            while let Ok(read @ 1..) = controller.read(&mut chunk) {
                if sent.send(chunk[..read].to_vec()).is_err() {
                    break;
                }
            }
        });
        let out = child.wait_with_output().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut shown = Vec::new();
        while let Ok(bytes) = screen.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now())) {
            shown.extend(bytes);
        }
        let shown = String::from_utf8_lossy(&shown);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(shown.is_empty(), "the prompt reached the terminal: {shown:?}\n{stdout}");
        // "1 passed" rules out a filter that matched nothing and exited 0.
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// A pseudo-terminal: the controller end the test reads, and the terminal
    /// end a child takes as its controlling terminal.
    #[cfg(unix)]
    fn pseudo_terminal() -> (std::fs::File, std::fs::File) {
        use std::os::fd::FromRawFd;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::OpenOptionsExt;
        // SAFETY: posix_openpt returns a new descriptor that the File then
        // owns, and ptsname's static buffer is copied out before any other
        // pty call.
        let (controller, path) = unsafe {
            let fd = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
            assert!(fd >= 0, "posix_openpt: {}", io::Error::last_os_error());
            let controller = std::fs::File::from_raw_fd(fd);
            assert_eq!(libc::grantpt(fd), 0, "grantpt: {}", io::Error::last_os_error());
            assert_eq!(libc::unlockpt(fd), 0, "unlockpt: {}", io::Error::last_os_error());
            let name = libc::ptsname(fd);
            assert!(!name.is_null(), "ptsname: {}", io::Error::last_os_error());
            let path = std::ffi::OsStr::from_bytes(std::ffi::CStr::from_ptr(name).to_bytes());
            (controller, path.to_owned())
        };
        let terminal = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOCTTY)
            .open(path)
            .unwrap();
        (controller, terminal)
    }
}
