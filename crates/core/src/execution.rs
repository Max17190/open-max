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
    sync::Arc,
    time::Duration,
};

use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
};

use crate::state::CancelToken;

const READ_CHUNK_BYTES: usize = 8 * 1024;
const TERMINATION_GRACE: Duration = Duration::from_millis(250);
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_secs(1);
/// How long a spilled command log stays on disk. Long enough that a resumed
/// session can still tail a log its transcript points at; short enough that
/// the directory stays bounded (unpruned, one machine reached 95 MB in a
/// month).
const SPILL_LOG_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

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
    /// full host environment (bash, hooks: the user's shell). `Some(names)`
    /// = scrub, then a baseline (PATH, HOME, LANG, TERM from the parent)
    /// plus exactly the named variables - external tools, whose manifest
    /// declares the list, making the credential grant part of the bytes a
    /// human approves. Ignored under `sandbox`: a probe stays fully
    /// scrubbed whatever its manifest asks for.
    pub env_allowlist: Option<Vec<String>>,
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
    /// The project root the probe may read (documentation of intent; reads
    /// are broadly allowed - see above - but the cwd and inputs live here).
    #[allow(dead_code)]
    pub ro_root: PathBuf,
    /// The one directory the probe may write; also its HOME.
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
    }
    // Mark every native child as agent-spawned. Trust grants are human
    // actions: the CLI refuses --trust-project (and the interactive trust
    // prompt) when this marker is present, so an agent cannot launder a
    // trust grant through a child process it starts. Applied after any
    // env_clear so the marker survives the scrub.
    command.env("OPENMAX_SESSION", "1");
    // The human attestation must never reach an agent-spawned child: a
    // session launched under an attested shell (CI, an eval rig) would
    // otherwise hand every bash call a ready-made bypass - unset the marker,
    // inherit the attestation, grant authority. Stripped unconditionally;
    // no child of the harness is the human.
    command.env_remove("OPENMAX_HUMAN_ATTEST");
    // Name the binary that is running this session. Every spec tells the
    // agent to shell out to `openmax --check` / `--spec`, and round-4
    // dogfooding hit a PATH `openmax` twelve days older than the harness
    // hosting the session, teaching claims that build had since retracted -
    // both printed the same version string. `$OPENMAX_BIN` is the same
    // build by construction. Not applied under a sandbox: a probe gets no
    // handle to the harness.
    if request.sandbox.is_none() {
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
    configure_process_group(&mut command);

    let mut child = command.spawn().map_err(ProcessError::Spawn)?;
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

    let termination = supervise_child(&mut child, pid, request.timeout, &cancel).await?;

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
    let wait_for_both = async {
        let (stdout_result, stderr_result) = tokio::join!(&mut stdout, &mut stderr);
        (stdout_result, stderr_result)
    };
    let joined = match tokio::time::timeout(OUTPUT_DRAIN_GRACE, wait_for_both).await {
        Ok(joined) => joined,
        Err(_) => {
            // A descendant can escape the invocation's process group while
            // retaining an inherited pipe. Stop reading so that it cannot
            // strand the agent after the supervised process has terminated.
            stop.cancel();
            match tokio::time::timeout(TERMINATION_GRACE, async {
                tokio::join!(&mut stdout, &mut stderr)
            })
            .await
            {
                Ok(joined) => joined,
                Err(_) => {
                    stdout.abort();
                    stderr.abort();
                    return Err(ProcessError::Wait(io::Error::other(
                        "output drains did not stop after process termination",
                    )));
                }
            }
        }
    };
    let stdout = joined
        .0
        .map_err(|error| ProcessError::Wait(io::Error::other(error)))?
        .map_err(ProcessError::Wait)?;
    let stderr = joined
        .1
        .map_err(|error| ProcessError::Wait(io::Error::other(error)))?
        .map_err(ProcessError::Wait)?;
    Ok((stdout, stderr))
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
) -> Result<Termination, ProcessError> {
    tokio::select! {
        result = child.wait() => {
            let status = result.map_err(ProcessError::Wait)?;
            // A shell can exit while ordinary background descendants retain the
            // group. Those belong to this invocation, so clean them up too.
            terminate_remaining_group(pid).await;
            Ok(Termination::Exited(status))
        }
        _ = cancel.cancelled() => {
            terminate_process_group(child, pid).await?;
            Ok(Termination::Cancelled)
        }
        _ = tokio::time::sleep(timeout) => {
            terminate_process_group(child, pid).await?;
            Ok(Termination::TimedOut)
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
    terminate_remaining_group(pid).await;
    Ok(())
}

async fn terminate_remaining_group(pid: Option<u32>) {
    // The leader may already have exited. Give remaining group members the
    // same grace period, then kill the group unconditionally if still present.
    if !process_group_exists(pid) {
        return;
    }
    send_termination_group(pid);
    tokio::time::sleep(TERMINATION_GRACE).await;
    send_kill_group(pid);
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.as_std_mut().process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_: &mut Command) {}

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
            },
            sandbox: None,
            env_allowlist: None,
        }
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
            },
            sandbox: None,
            env_allowlist: None,
        };
        let output = run_process(request, Arc::new(CancelToken::default())).await.unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout.head), "1");
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
            capture: CaptureSpec { head_bytes: 64, tail_bytes: 0, spill_dir: None, spill_bytes_per_stream: 0 },
            sandbox: None,
            env_allowlist: None,
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
            capture: CaptureSpec { head_bytes: 4096, tail_bytes: 0, spill_dir: None, spill_bytes_per_stream: 0 },
            sandbox: None,
            env_allowlist: None,
        };
        let output = run_process(request, Arc::new(CancelToken::default())).await.unwrap();
        let got = String::from_utf8_lossy(&output.stdout.head).to_string();
        let expected = std::env::current_exe().unwrap().to_string_lossy().to_string();
        assert_eq!(got, expected, "OPENMAX_BIN must name the running executable");
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
            },
            sandbox: Some(SandboxPolicy {
                ro_root: scratch.to_path_buf(),
                rw_scratch: scratch.to_path_buf(),
            }),
            env_allowlist: None,
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

    #[tokio::test]
    async fn times_out_and_drains_output() {
        let mut request = request("/bin/sh", &["-c", "printf before; sleep 10"]);
        request.timeout = Duration::from_millis(25);
        let output = run_process(request, Arc::new(CancelToken::default()))
            .await
            .unwrap();
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

    /// One dogfooding machine reached 95 MB of spill logs in a month because
    /// nothing ever deleted them. Writing a new log is the moment the
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

    #[tokio::test]
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
        let marker =
            std::env::temp_dir().join(format!("openmax-descendant-{}", uuid::Uuid::new_v4()));
        let script = format!("(sleep 1; touch '{}') &", marker.display());
        let output = run_process(
            request("/bin/sh", &["-c", &script]),
            Arc::new(CancelToken::default()),
        )
        .await
        .unwrap();
        assert!(matches!(output.termination, Termination::Exited(status) if status.success()));
        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert!(
            !marker.exists(),
            "background descendant survived process cleanup"
        );
    }
}
