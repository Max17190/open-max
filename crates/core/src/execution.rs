//! Native child-process supervision shared by agent-dispatched commands.
//!
//! This module deliberately preserves the host environment. It owns lifecycle
//! and bounded capture, not sandbox policy.

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

/// A single native process invocation. The caller supplies exact argv and cwd;
/// no shell interpretation is introduced by the supervisor.
pub(crate) struct ProcessRequest {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
    pub stdin: StdinMode,
    pub timeout: Duration,
    pub capture: CaptureSpec,
}

pub(crate) enum StdinMode {
    Null,
    Bytes(Vec<u8>),
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
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(error) => write!(f, "failed to spawn process: {error}"),
            Self::Wait(error) => write!(f, "failed while supervising process: {error}"),
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

/// Execute one native process with concurrent bounded output capture.
pub(crate) async fn run_process(
    request: ProcessRequest,
    cancel: Arc<CancelToken>,
) -> Result<ProcessOutput, ProcessError> {
    let mut command = Command::new(&request.program);
    command
        .args(&request.args)
        .current_dir(&request.cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
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
        // Preserve the historical caller contract: a child may exit without
        // consuming stdin, and stdin transport errors do not replace its
        // actual exit status or captured diagnostics.
        let _ = task.await;
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
        }
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
