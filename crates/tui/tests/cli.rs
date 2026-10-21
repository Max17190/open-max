//! End-to-end tests of the built binary: trust gating, --check exit codes,
//! the --stdio handshake, and a full print-mode turn against a stub
//! OpenAI-compatible server. These are the contracts scripts and frontends
//! build on, exercised exactly as a user's shell would.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use open_max_core::config::ApprovalMode;

fn openmax_bin() -> &'static str {
    env!("CARGO_BIN_EXE_openmax")
}

/// Bound a regression's child lifetime even if a broken admission path hangs.
fn finish_with_deadline(mut child: std::process::Child) -> std::process::Output {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if child.try_wait().unwrap().is_some() { return child.wait_with_output().unwrap(); }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!("child did not finish: {}", String::from_utf8_lossy(&output.stderr));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn an_idle_stdio_owner_excludes_continuation_until_process_exit() {
    let (project, home) = fresh_dirs("session-owner");
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    server.set_nonblocking(true).unwrap();
    write_settings(&home, &format!("http://{}/v1", server.local_addr().unwrap()));
    let mut owner = cmd(&project, &home).args(["--trust-project", "--stdio"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let stdout = owner.stdout.take().unwrap();
    let (hello_tx, hello_rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut line = String::new();
        BufReader::new(stdout).read_line(&mut line).unwrap();
        let _ = hello_tx.send(line);
    });
    let hello = match hello_rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(line) => line,
        Err(e) => { let _ = owner.kill(); let _ = owner.wait(); panic!("missing handshake: {e}"); }
    };
    reader.join().unwrap();
    let hello: serde_json::Value = serde_json::from_str(&hello).unwrap();
    let id = hello["session_id"].as_str().unwrap();
    let transcript = home.join(".openmax/sessions").join(format!("{id}.messages.json"));
    let competitors = [vec!["--continue", "-p", "must not run"], vec!["--continue", "--stdio"]];
    for args in competitors {
        let output = finish_with_deadline(cmd(&project, &home).args(args)
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("already open in another process"));
        assert!(!transcript.exists());
    }
    assert_eq!(server.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "a refused attachment cannot contact the model");
    let independent = finish_with_deadline(cmd(&project, &home).arg("--stdio")
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
    assert!(independent.status.success(), "a separate session remains usable");
    // An abrupt exit must release the OS lock without a cleanup command.
    owner.kill().unwrap();
    owner.wait().unwrap();
    let resumed = finish_with_deadline(cmd(&project, &home).args(["--continue", "--stdio"])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
    assert!(resumed.status.success(), "{}", String::from_utf8_lossy(&resumed.stderr));
    let resumed: serde_json::Value = serde_json::from_slice(resumed.stdout.split(|b| *b == b'\n').next().unwrap()).unwrap();
    assert_eq!(resumed["session_id"], id);
    assert_eq!(resumed["continued"], true);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

#[test]
fn damaged_continuation_preserves_bytes_before_hooks_or_provider_requests() {
    let (project, home) = fresh_dirs("damaged-continuation");
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    server.set_nonblocking(true).unwrap();
    write_settings_with_mode(&home, &format!("http://{}/v1", server.local_addr().unwrap()), "auto");
    let data = home.join(".openmax");
    let canonical = std::fs::canonicalize(&project).unwrap();
    let (core, _) = open_max_core::state::Core::new(data.clone()).unwrap();
    let id = open_max_core::sessions::create(&core, canonical.display().to_string()).unwrap().id;
    drop(core);
    let transcript = data.join("sessions").join(format!("{id}.messages.json"));
    let bytes = b"{\"role\":\"user\",\"content\":\"preserve\"}\n{damaged record}\n";
    std::fs::write(&transcript, bytes).unwrap();
    std::fs::create_dir_all(project.join(".openmax/hooks")).unwrap();
    std::fs::write(project.join(".openmax/hooks/submit.toml"),
        "event = \"user_prompt_submit\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch should-not-run\"]\n").unwrap();
    for args in [vec!["--trust-project", "--continue", "-p", "hello"], vec!["--trust-project", "--continue", "--stdio"]] {
        let output = finish_with_deadline(cmd(&project, &home).args(args)
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("line 2") && error.contains("preserved"), "{error}");
        assert_eq!(std::fs::read(&transcript).unwrap(), bytes);
        assert!(!project.join("should-not-run").exists());
    }
    assert_eq!(server.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A session is recorded under its project's resolved path. When that
/// directory moves and a symlink to another project takes its place, the
/// path leads to the other project, which inherits none of the history
/// recorded there: run from it, `--recall` finds nothing and `--continue`
/// has no session to resume.
#[cfg(unix)]
#[test]
fn a_symlink_that_takes_a_projects_place_does_not_carry_its_history() {
    let (project, home) = fresh_dirs("retargeted-project");
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    server.set_nonblocking(true).unwrap();
    write_settings(&home, &format!("http://{}/v1", server.local_addr().unwrap()));
    let earlier = project.parent().unwrap().join("earlier");
    std::fs::create_dir(&earlier).unwrap();
    let recorded = std::fs::canonicalize(&earlier).unwrap();
    let (core, _) = open_max_core::state::Core::new(home.join(".openmax")).unwrap();
    let id = open_max_core::sessions::create(&core, recorded.display().to_string()).unwrap().id;
    let messages = [
        open_max_core::types::ChatMessage::system("rules"),
        open_max_core::types::ChatMessage::user("the quokkaberry rollout plan"),
    ];
    assert!(open_max_core::sessions::save_messages(&core, &id, &messages, &mut 0, false));
    drop(core);
    std::fs::rename(&recorded, project.parent().unwrap().join("moved")).unwrap();
    std::os::unix::fs::symlink(&project, &recorded).unwrap();

    let recalled = cmd(&project, &home).args(["--recall", "quokkaberry", "--json"]).output().unwrap();
    assert!(recalled.status.success(), "{}", String::from_utf8_lossy(&recalled.stderr));
    let report: serde_json::Value = serde_json::from_slice(&recalled.stdout).unwrap();
    assert_eq!(report["sessions_scanned"], 0, "{report}");
    assert!(report["hits"].as_array().unwrap().is_empty(), "{report}");

    let resumed = finish_with_deadline(cmd(&project, &home).args(["--trust-project", "--continue", "--stdio"])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
    let stderr = String::from_utf8_lossy(&resumed.stderr);
    assert_eq!(resumed.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("no prior session in this directory"), "{stderr}");
    assert_eq!(server.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A damaged session index is history openmax cannot read, not an empty past:
/// `--continue` names the file instead of reporting that no prior session
/// exists, a new session is refused the same way, and `--check` lists it as
/// an error with the repair. Each refusal points at `--check` rather than
/// leaving the user with a bare path to move while another openmax may still
/// be saving to it. The damaged bytes stay as they were and no request
/// reaches the model.
#[test]
fn a_damaged_session_index_is_named_by_continue_and_check() {
    let (project, home) = fresh_dirs("damaged-index");
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    server.set_nonblocking(true).unwrap();
    write_settings(&home, &format!("http://{}/v1", server.local_addr().unwrap()));
    let index = home.join(".openmax/sessions/index.json");
    std::fs::create_dir_all(index.parent().unwrap()).unwrap();
    std::fs::write(&index, "[{").unwrap();
    let path = index.display().to_string();
    for args in [
        vec!["--trust-project", "--continue", "-p", "hello"],
        vec!["--trust-project", "--continue", "--stdio"],
        vec!["--trust-project", "-p", "hello"],
        vec!["--trust-project", "--stdio"],
    ] {
        let output = finish_with_deadline(cmd(&project, &home).args(&args)
            .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(&path), "{args:?} must name the damaged index: {error}");
        assert!(error.contains("run openmax --check for the repair"), "{args:?} must point at the repair: {error}");
        assert_eq!(output.status.code(), Some(1), "{args:?}: {error}");
        assert!(output.stdout.is_empty(), "{args:?} must fail before any output");
        assert_eq!(std::fs::read(&index).unwrap(), b"[{");
    }
    let check = cmd(&project, &home).arg("--check").output().unwrap();
    let report = String::from_utf8_lossy(&check.stdout);
    let row = report.lines().find(|line| line.contains(&path))
        .unwrap_or_else(|| panic!("--check must list the damaged index: {report}"));
    assert!(row.starts_with("err") && row.contains("mv "), "{row}");
    assert_eq!(check.status.code(), Some(1), "{report}");
    assert_eq!(std::fs::read(&index).unwrap(), b"[{");
    assert_eq!(server.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A fresh project dir plus a fresh HOME, so trust and settings never leak
/// between tests or into the developer's real ~/.openmax.
fn fresh_dirs(tag: &str) -> (PathBuf, PathBuf) {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!("openmax-e2e-{tag}-{}-{nonce}", std::process::id()));
    let project = base.join("project");
    let home = base.join("home");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    (project, home)
}

fn write_settings(home: &Path, base_url: &str) {
    write_settings_with_mode(home, base_url, "ask");
}

/// `auto` is what an unattended run uses: mutating tools execute without a
/// human. Tests that must prove a call was refused for its own reason (not
/// because the approval gate declined it) run in that mode.
fn write_settings_with_mode(home: &Path, base_url: &str, approval_mode: &str) {
    let dir = home.join(".openmax");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("settings.json"),
        format!(
            r#"{{"base_url":"{base_url}","model":"stub-model","approval_mode":"{approval_mode}","context_tokens":16384}}"#
        ),
    )
    .unwrap();
}

/// Trust `project` as a human who answers `a` at the trust prompt does. For
/// tests of the ask gates: `--trust-project` records auto, which skips them.
fn trust_in_ask(project: &Path, home: &Path) {
    let grant = open_max_core::trust::grant_trust(&home.join(".openmax"), project, ApprovalMode::Ask);
    assert_eq!(grant.unwrap().1, Some(ApprovalMode::Ask));
}

/// Approval records in this home's ledger logs, read from disk.
fn approval_records(home: &Path) -> usize {
    let Ok(dirs) = std::fs::read_dir(home.join(".openmax").join("ledger")) else {
        return 0;
    };
    dirs.flatten()
        .filter_map(|dir| std::fs::read_to_string(dir.path().join("log.jsonl")).ok())
        .map(|log| log.lines().filter(|line| line.contains("\"kind\":\"approval\"")).count())
        .sum()
}

fn cmd(project: &Path, home: &Path) -> Command {
    let mut c = Command::new(openmax_bin());
    c.current_dir(project);
    c.env("HOME", home);
    c.env_remove("OPENMAX_API_KEY");
    // A developer may run cargo test from inside a session;
    // the harness marks such children and trust would refuse (#83).
    c.env_remove("OPENMAX_SESSION");
    // Tests are human-run automation with no terminal: attest it, so
    // --approve / --trust-project (which now require a tty otherwise) run.
    c.env("OPENMAX_HUMAN_ATTEST", "1");
    c
}

/// The authority-granting commands refuse a caller with no terminal and no
/// attestation, even with the session marker absent: `env -u
/// OPENMAX_SESSION openmax --approve` from an agent's bash (piped stdio, no
/// tty) is exactly this shape, and it is the first thing an agent reaches
/// for when the in-session refusal blocks it.
#[test]
fn approve_and_trust_refuse_without_a_terminal_or_attestation() {
    let (project, home) = fresh_dirs("noterminal");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(project.join("gate.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(project.join("gate.sh"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
    }
    std::fs::write(
        hooks.join("gate.toml"),
        "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n",
    )
    .unwrap();
    let bare = |args: &[&str]| {
        let mut c = cmd(&project, &home);
        c.env_remove("OPENMAX_HUMAN_ATTEST");
        c.stdin(std::process::Stdio::null());
        c.args(args).output().unwrap()
    };
    let out = bare(&["--approve", ".openmax/hooks/gate.toml"]);
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no terminal"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The printed repair is pasted into a shell, so a path with a space must
    // come back quoted, not split into two arguments.
    let out = bare(&["--approve", ".openmax/hooks/my gate.toml"]);
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("`openmax --approve '.openmax/hooks/my gate.toml'`"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = bare(&["--trust-project", "-p", "hi"]);
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    // The attestation (what cmd() sets) is what lets test automation through.
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
}

/// What a user's shell, a CI job, or a frontend actually runs: none of the
/// attestation `cmd` adds for test convenience.
fn plain_cmd(project: &Path, home: &Path) -> Command {
    let mut c = cmd(project, home);
    c.env_remove("OPENMAX_HUMAN_ATTEST");
    c
}

/// A pseudo-terminal: the controller end the test keeps, and the terminal end
/// a child gets as stdin, which is what an interactive shell hands openmax.
/// The controller must outlive the child.
fn pseudo_terminal() -> (std::fs::File, std::fs::File) {
    use std::os::fd::FromRawFd;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    // SAFETY: posix_openpt returns a new descriptor that the File then owns,
    // and ptsname's static buffer is copied out before any other pty call.
    let (controller, path) = unsafe {
        let fd = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
        assert!(fd >= 0, "posix_openpt: {}", std::io::Error::last_os_error());
        let controller = std::fs::File::from_raw_fd(fd);
        assert_eq!(libc::grantpt(fd), 0, "grantpt: {}", std::io::Error::last_os_error());
        assert_eq!(libc::unlockpt(fd), 0, "unlockpt: {}", std::io::Error::last_os_error());
        let name = libc::ptsname(fd);
        assert!(!name.is_null(), "ptsname: {}", std::io::Error::last_os_error());
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

/// Every line of a ```sh block that runs `--trust-project`.
fn documented_trust_commands(markdown: &str) -> Vec<String> {
    let mut shell_block = None;
    let mut lines = Vec::new();
    for line in markdown.lines().map(str::trim) {
        if let Some(lang) = line.strip_prefix("```") {
            shell_block = match shell_block {
                Some(_) => None,
                None => Some(matches!(lang, "sh" | "bash")),
            };
        } else if shell_block == Some(true) && line.contains("--trust-project") {
            lines.push(line.to_string());
        }
    }
    lines
}

/// The words and trailing comment of one documented shell line, split the
/// way sh splits the forms the docs use: bare words, single- and
/// double-quoted strings (neither expands anything the docs write in them),
/// and a `# comment`.
fn shell_words(line: &str) -> (Vec<String>, String) {
    let (mut words, mut word, mut quote) = (Vec::new(), None::<String>, None::<char>);
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match (c, quote) {
            ('"' | '\'', None) => {
                quote = Some(c);
                word.get_or_insert_with(String::new);
            }
            (c, Some(open)) if c == open => quote = None,
            ('#', None) if word.is_none() => return (words, chars.as_str().trim().to_string()),
            (c, None) if c.is_whitespace() => words.extend(word.take()),
            (c, _) => word.get_or_insert_with(String::new).push(c),
        }
    }
    assert!(quote.is_none(), "unterminated quote in a documented command: {line}");
    words.extend(word);
    (words, String::new())
}

/// Read the stdio handshake, then quit through `input`. The child's stdout
/// closing first (a refusal) is reported with its exit and stderr, and a
/// child that stays alive without a handshake is killed at a deadline rather
/// than hanging the test.
fn stdio_handshake(
    mut child: std::process::Child,
    input: &mut dyn Write,
) -> Result<std::process::Output, String> {
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut hello = String::new();
        stdout.read_line(&mut hello).unwrap();
        let _ = tx.send((hello, stdout));
    });
    // The reader comes back with the line and is held until the child exits,
    // so nothing it writes after the handshake meets a closed pipe.
    let (hello, _stdout) = match rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(read) => read,
        Err(e) => {
            let _ = child.kill();
            let out = child.wait_with_output().unwrap();
            return Err(format!(
                "no stdio handshake ({e}): {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
    };
    if !hello.contains("\"hello\"") {
        let out = finish_with_deadline(child);
        return Err(format!(
            "no stdio handshake (exit {:?}): {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    input.write_all(b"{\"cmd\":\"quit\"}\n").unwrap();
    Ok(finish_with_deadline(child))
}

/// Run one documented trust command exactly as written, in a fresh project,
/// in the context the docs give it: a line commented `from a terminal` gets a
/// terminal on stdin, any other line gets the pipe a frontend or a CI job
/// hands it and, as in CI, no controlling terminal. No variable is set that
/// the line does not set itself, except the variable an `--api-key '$NAME'`
/// reads, which the docs tell the user to export first. Success is the
/// documented result: the run
/// itself works, the project is trusted in auto although settings.json says
/// ask, and a frontend's plain `openmax --stdio` then starts with no flag.
/// A line that carries `--base-url` is a first run: it runs on an install
/// with no settings.json, against the stub, whose address stands in for the
/// documented server's, and must leave the endpoint written.
fn run_documented_trust_command(line: &str) -> Result<(), String> {
    let (mut words, comment) = shell_words(line);
    let at = words.iter().position(|w| w == "openmax").ok_or("no openmax invocation")?;
    let (project, home) = fresh_dirs("documented-trust");
    let (base_url, _requests, _server) = spawn_stub_server();
    let first_run = words.iter().position(|w| w == "--base-url");
    match first_run {
        Some(flag) => *words.get_mut(flag + 1).ok_or("--base-url without a value")? = base_url.clone(),
        None => write_settings(&home, &base_url),
    }
    let mut command = plain_cmd(&project, &home);
    let key = words.iter().position(|w| w == "--api-key").and_then(|flag| words.get(flag + 1));
    if let Some(name) = key.and_then(|key| key.strip_prefix('$')) {
        command.env(name, "documented-key");
    }
    for assignment in &words[..at] {
        let (name, value) = assignment
            .split_once('=')
            .ok_or_else(|| format!("unexpected word before openmax: {assignment}"))?;
        command.env(name, value);
    }
    let args = &words[at + 1..];
    command.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut controller = None;
    if comment.contains("from a terminal") {
        let (pty, terminal) = pseudo_terminal();
        controller = Some(pty);
        command.stdin(terminal);
    } else {
        // What a CI job gets: no controlling terminal either.
        without_controlling_terminal(command.stdin(Stdio::piped()));
    }
    let mut child = command.spawn().unwrap();
    let out = if args.iter().any(|a| a == "--stdio") {
        let mut pipe = child.stdin.take();
        let input: &mut dyn Write = match (&mut controller, &mut pipe) {
            (Some(pty), _) => pty,
            (None, Some(pipe)) => pipe,
            (None, None) => unreachable!("stdin is piped when no terminal is given"),
        };
        stdio_handshake(child, input)?
    } else if args.iter().any(|a| a == "-p") || controller.is_none() {
        // A print run ends with its turn, and a grant alone with no terminal
        // ends once it is recorded.
        finish_with_deadline(child)
    } else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("a grant alone from a terminal starts a session: teach this test the documented result".into());
    };
    drop(controller);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if out.status.code() != Some(0) {
        return Err(format!("exit {:?}: {}", out.status.code(), stderr.trim()));
    }
    if args.iter().any(|a| a == "-p") && !stdout.contains("stub says hi") {
        return Err(format!("the turn did not reach stdout: {stdout}\n{stderr}"));
    }
    let settings = std::fs::read_to_string(home.join(".openmax/settings.json")).unwrap_or_default();
    if first_run.is_some() && !settings.contains(&base_url) {
        return Err(format!("the first run did not write its endpoint: {settings:?}\n{stderr}"));
    }
    if open_max_core::trust::is_trusted(&home.join(".openmax"), &project) != Ok(true) {
        return Err("the project is not trusted afterwards".into());
    }
    let mode = approval_mode(&home, &project);
    if mode != ApprovalMode::Auto {
        return Err(format!("the grant left the project in {}, not auto", mode.as_str()));
    }
    let mut frontend = plain_cmd(&project, &home)
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = frontend.stdin.take().unwrap();
    let out = stdio_handshake(frontend, &mut pipe).map_err(|e| format!("frontend afterwards: {e}"))?;
    if out.status.code() != Some(0) {
        return Err(format!("frontend afterwards exited {:?}", out.status.code()));
    }
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    Ok(())
}

/// Every trust command the README and the guides show runs verbatim and gets
/// its documented result. A frontend spawns its command with stdin as the
/// protocol pipe and CI has no terminal at all, so a command that only works
/// when a human types it must say so, and the ones meant for a frontend or
/// automation must work without one.
#[test]
fn documented_trust_commands_work_as_written() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut docs = vec![root.join("README.md")];
    docs.extend(
        std::fs::read_dir(root.join("docs"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "md")),
    );
    let (mut checked, mut failures) = (0, Vec::new());
    for doc in docs {
        let text = std::fs::read_to_string(&doc).unwrap();
        for line in documented_trust_commands(&text) {
            checked += 1;
            if let Err(why) = run_documented_trust_command(&line) {
                failures.push(format!("{}: `{line}`: {why}", doc.display()));
            }
        }
    }
    assert!(checked > 0, "the docs show no trust command, so this test pins nothing");
    assert!(
        failures.is_empty(),
        "documented trust commands that do not work as written:\n{}",
        failures.join("\n")
    );
}

/// A frontend's stdin is its protocol pipe, never a terminal, so it cannot
/// grant trust, and the refusal it gets on an untrusted project must name a
/// grant that works for it rather than the flag that cannot. The grant the
/// no-terminal refusal prints must paste into a shell as written, from a
/// project whose path a shell would split.
#[test]
fn a_frontend_cannot_grant_trust_and_is_told_where_it_comes_from() {
    let (base, home) = fresh_dirs("frontend-trust");
    let project = base.join("my proj");
    std::fs::create_dir_all(&project).unwrap();
    write_settings(&home, "http://127.0.0.1:9/v1");
    let spawn = |args: &[&str]| {
        finish_with_deadline(
            plain_cmd(&project, &home)
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        )
    };
    let out = spawn(&["--trust-project", "--stdio"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("no terminal"), "{stderr}");
    let repair = format!(
        "`cd {} && openmax --trust-project`",
        open_max_core::doctor::shell_quote(&std::fs::canonicalize(&project).unwrap())
    );
    assert!(stderr.contains(&repair), "the printed grant must paste as written: {stderr}");
    assert!(out.stdout.is_empty(), "a refused grant must not start a session");
    assert_eq!(open_max_core::trust::is_trusted(&home.join(".openmax"), &project), Ok(false));

    // The frontend's user pastes this into a terminal that is usually not in
    // the project, and a grant covers its subtree, so the repair must carry
    // the project rather than trust wherever that terminal happens to be.
    let out = spawn(&["--stdio"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("not trusted") && stderr.contains("from a terminal"), "{stderr}");
    assert!(stderr.contains(&repair), "the grant must name the refused project: {stderr}");
    let _ = std::fs::remove_dir_all(base.parent().unwrap());
}

/// The mode a fresh launch in `project` runs under: the project's saved
/// choice or an enclosing one, else the settings value.
fn approval_mode(home: &Path, project: &Path) -> ApprovalMode {
    open_max_core::state::Core::new(home.join(".openmax")).unwrap().0.approval_mode(project)
}

/// `--trust-project` records auto for the project in the same trust.json
/// write as the trust itself, so a newly trusted project runs in auto even
/// where settings.json says ask (as most files do without the user choosing
/// it: every settings save writes the whole file). A project trusted before
/// grants recorded a mode is not granted again: it keeps resolving to the
/// settings value. An agent-spawned process still can neither trust nor
/// record a mode.
#[test]
fn trust_project_records_auto_for_a_project_it_newly_trusts() {
    let (project, home) = fresh_dirs("trust-mode");
    // An endpoint that answers: one that refuses holds each run for the
    // whole retry backoff, which this test does not look at.
    let (base_url, _requests, _server) = spawn_stub_server();
    write_settings(&home, &base_url);
    let out = cmd(&project, &home).args(["--trust-project", "-p", "hi"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(approval_mode(&home, &project), ApprovalMode::Auto, "{stderr}");
    assert!(stderr.contains("auto") && stderr.contains("/approvals"), "the grant must say what it recorded: {stderr}");

    let earlier = project.parent().unwrap().join("earlier");
    std::fs::create_dir_all(&earlier).unwrap();
    open_max_core::trust::trust_project(&home.join(".openmax"), &earlier).unwrap();
    let out = cmd(&earlier, &home).args(["--trust-project", "-p", "hi"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(approval_mode(&home, &earlier), ApprovalMode::Ask, "no retroactive mode: {stderr}");
    assert!(stderr.contains("already trusted") && stderr.contains("unchanged"), "{stderr}");

    let spawned = project.parent().unwrap().join("spawned");
    std::fs::create_dir_all(&spawned).unwrap();
    let out = cmd(&spawned, &home)
        .env("OPENMAX_SESSION", "parent")
        .args(["--trust-project", "-p", "hi"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(open_max_core::trust::is_trusted(&home.join(".openmax"), &spawned), Ok(false));
    let store = std::fs::read_to_string(home.join(".openmax/trust.json")).unwrap();
    let spawned = std::fs::canonicalize(&spawned).unwrap();
    assert!(!store.contains(spawned.to_str().unwrap()), "an agent-spawned grant recorded state: {store}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Start `command` with no controlling terminal, as in a CI job. A run that
/// goes on to the interactive session then fails to open /dev/tty instead of
/// putting the terminal the tests run in into raw mode and reading its keys.
fn without_controlling_terminal(command: &mut Command) -> &mut Command {
    use std::os::unix::process::CommandExt;
    // SAFETY: setsid is async-signal-safe, and the closure touches nothing
    // else between fork and exec.
    unsafe {
        command.pre_exec(|| match libc::setsid() {
            -1 => Err(std::io::Error::last_os_error()),
            _ => Ok(()),
        })
    }
}

/// A grant where nothing gives it a terminal, as in a CI job (what the
/// attestation is for), ends the run once it is recorded: `--trust-project`
/// with no `-p` or `--stdio` records the grant, says what it recorded and
/// that no session started, and exits 0. The interactive session it
/// otherwise goes on to draws on stdout, and starting it there failed after
/// the grant was recorded, so a grant that succeeded exited 1 with a terminal
/// error. A second grant keeps the mode the project has by then (here one
/// changed to ask since) and exits 0 the same way.
/// With `--continue` the run asked for that session, which did not start, so
/// it exits 1 after the grant instead of as if the session had run.
#[test]
fn a_grant_with_no_terminal_exits_0_once_it_is_recorded() {
    let (project, home) = fresh_dirs("grant-only");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let grant = |session: &[&str]| {
        let mut command = cmd(&project, &home);
        command
            .arg("--trust-project")
            .args(session)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        finish_with_deadline(without_controlling_terminal(&mut command).spawn().unwrap())
    };
    let root = std::fs::canonicalize(&project).unwrap();
    let out = grant(&[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(
        stderr.contains(&format!("openmax: trusted project {} in auto mode", root.display())),
        "the run must say what it recorded: {stderr}"
    );
    assert!(stderr.contains("no session was started"), "{stderr}");
    assert!(!stderr.contains("failed to initialize terminal"), "{stderr}");
    assert!(out.stdout.is_empty(), "{}", String::from_utf8_lossy(&out.stdout));
    let trust_file = home.join(".openmax/trust.json");
    let read_store =
        || serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(&trust_file).unwrap()).unwrap();
    let mut store = read_store();
    let root = root.to_str().unwrap();
    assert_eq!(store["projects"], serde_json::json!([root]), "{store}");
    assert_eq!(store["approval_modes"][root], "auto", "{store}");

    // A grant records auto, so the second one starts from ask: had it
    // recorded its mode again, the project would read auto afterwards.
    store["approval_modes"][root] = "ask".into();
    std::fs::write(&trust_file, store.to_string()).unwrap();
    let out = grant(&[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(stderr.contains("already trusted") && stderr.contains("no session was started"), "{stderr}");
    assert!(stderr.contains("its approval mode is unchanged"), "{stderr}");
    assert!(!stderr.contains("failed to initialize terminal"), "{stderr}");
    let store = read_store();
    assert_eq!(store["approval_modes"][root], "ask", "{store}");

    let out = grant(&["--continue"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "the session --continue asks for did not start: {stderr}");
    assert!(stderr.contains("already trusted") && stderr.contains("no session was started"), "{stderr}");
    assert!(!stderr.contains("failed to initialize terminal"), "{stderr}");
    assert!(out.stdout.is_empty(), "{}", String::from_utf8_lossy(&out.stdout));
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A launch with `--trust-project` on a project it already trusts writes
/// nothing: trust.json keeps its bytes and its inode, where it used to be
/// replaced (temp file, rename, two disk syncs) with identical bytes on the
/// way to the first frame. The run still exits 0 and says the approval mode
/// is unchanged.
#[test]
fn a_grant_on_an_already_trusted_project_leaves_trust_json_untouched() {
    use std::os::unix::fs::MetadataExt;
    let (project, home) = fresh_dirs("grant-again");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let grant = || {
        let mut command = cmd(&project, &home);
        command.arg("--trust-project").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        finish_with_deadline(without_controlling_terminal(&mut command).spawn().unwrap())
    };
    let out = grant();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let trust_file = home.join(".openmax/trust.json");
    let store = || (std::fs::metadata(&trust_file).unwrap().ino(), std::fs::read(&trust_file).unwrap());
    let first = store();

    let out = grant();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(stderr.contains("already trusted; its approval mode is unchanged"), "{stderr}");
    assert_eq!(store(), first, "a grant that changes nothing must not rewrite trust.json");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

#[test]
fn an_untrusted_project_fails_closed_with_exit_3() {
    let (project, home) = fresh_dirs("trust");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let out = cmd(&project, &home)
        .args(["-p", "hello"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("trust"), "{stderr}");
}

#[test]
fn check_exit_codes_follow_findings() {
    let (project, home) = fresh_dirs("check");
    // Healthy config: a real tool file. --check needs no trust and no endpoint.
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(
        tools.join("ok.toml"),
        "name = \"ok\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n",
    )
    .unwrap();
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stdout));

    std::fs::write(tools.join("broken.toml"), "name = [not toml").unwrap();
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("broken.toml"), "{stdout}");
}

/// An agent runs `--check` after every extension file it writes, and the
/// report stays in its transcript, re-sent with every later request. So the
/// default report prints only rows to act on (warn and err, plus a hook's row,
/// in auto mode the one place its shape is reported), names files relative to
/// the project, and counts the files with nothing to report on one line.
/// `--all` lists every row; the JSON face and the exit codes are unchanged.
#[test]
fn check_prints_rows_to_act_on_and_counts_the_files_that_loaded() {
    let (project, home) = fresh_dirs("check-diet");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let grant = open_max_core::trust::grant_trust(&home.join(".openmax"), &project, ApprovalMode::Auto);
    assert_eq!(grant.unwrap().1, Some(ApprovalMode::Auto));
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(tools.join("probe.toml"), "name = \"probe\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n")
        .unwrap();
    let skill = project.join(".agents").join("skills").join("notes");
    std::fs::create_dir_all(&skill).unwrap();
    std::fs::write(skill.join("SKILL.md"), "---\nname: notes\ndescription: Take notes\n---\nBody.\n").unwrap();
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(hooks.join("watch.toml"), "event = \"turn_end\"\ncommand = \"/bin/echo\"\n").unwrap();
    let root = std::fs::canonicalize(&project).unwrap().display().to_string();
    let roots = [project.display().to_string(), root.clone()];
    let check = |project: &Path, args: &[&str]| {
        let out = cmd(project, &home).args(args).output().unwrap();
        (out.status.code(), String::from_utf8_lossy(&out.stdout).into_owned())
    };

    let (code, report) = check(&project, &["--check"]);
    assert_eq!(code, Some(0), "{report}");
    assert!(!report.lines().any(|l| l.starts_with("ok   tool") || l.starts_with("ok   skill")), "{report}");
    assert!(roots.iter().all(|root| !report.contains(root.as_str())), "absolute path printed:\n{report}");
    let summary: Vec<&str> = report.lines().filter(|l| l.starts_with("ok: ")).collect();
    assert_eq!(summary.len(), 1, "{report}");
    assert!(summary[0].contains("1 tool,") && summary[0].contains("1 skill"), "{report}");
    assert!(summary[0].contains("openmax --check --all"), "{report}");
    // A turn_end hook without `blocking` observes only, and in auto mode its
    // row is the only report of that: it is not folded into the count.
    assert!(
        report.lines().any(|l| l.starts_with("ok   hook        .openmax/hooks/watch.toml") && l.contains("observer")),
        "{report}"
    );

    let (code, all) = check(&project, &["--check", "--all"]);
    assert_eq!(code, Some(0), "{all}");
    assert!(all.lines().any(|l| l.starts_with("ok   tool        .openmax/tools/probe.toml  (tool 'probe'")), "{all}");
    assert!(all.lines().any(|l| l.starts_with("ok   skill       .agents/skills/notes/SKILL.md  (skill 'notes')")), "{all}");
    assert!(all.lines().any(|l| l.starts_with("ok   settings    ")), "{all}");
    assert!(!all.lines().any(|l| l.starts_with("ok: ")), "{all}");

    // A message prints as written: the parse error quotes the broken line,
    // which names this very file by its quoted absolute path, and only the
    // row's path column is relative.
    let quoted = format!("'{root}/.openmax/tools/broken.toml'");
    std::fs::write(tools.join("broken.toml"), format!("name = \"broken\"\ncommand = \"cat {quoted}\" oops\n")).unwrap();
    let (code, report) = check(&project, &["--check"]);
    assert_eq!(code, Some(1), "{report}");
    let out = cmd(&project, &home).args(["--check", "--json"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let rows: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = rows.as_array().unwrap();
    // The machine face still carries every finding, at its absolute path.
    assert!(rows.iter().any(|r| r["status"] == "ok" && r["surface"] == "tool"), "{rows:?}");
    assert!(rows.iter().all(|r| Path::new(r["path"].as_str().unwrap()).is_absolute()), "{rows:?}");
    let reason = rows
        .iter()
        .find(|r| r["path"].as_str().unwrap().ends_with("broken.toml"))
        .and_then(|r| r["message"].as_str())
        .unwrap();
    assert!(reason.contains(&quoted), "the parse error quotes the line: {reason}");
    let row = report
        .lines()
        .find(|l| l.starts_with("err  tool        .openmax/tools/broken.toml  "))
        .unwrap_or_else(|| panic!("the broken manifest's row:\n{report}"));
    assert!(row.ends_with(reason), "the row carries the reason as written {reason:?}: {row}");

    // In ask, an unapproved tool's row carries the approval command, naming
    // the file relative to the project; pasted there, it approves the tool,
    // which then has nothing to report.
    let (project, _) = fresh_dirs("check-diet-ask");
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(tools.join("probe.toml"), "name = \"probe\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n")
        .unwrap();
    let (code, report) = check(&project, &["--check"]);
    assert_eq!(code, Some(0), "{report}");
    let command = report
        .split("(openmax --approve ")
        .nth(1)
        .and_then(|rest| rest.split(", or").next())
        .unwrap_or_else(|| panic!("no approval command:\n{report}"));
    assert_eq!(command, "'.openmax/tools/probe.toml'", "{report}");
    let approve = Command::new("/bin/sh")
        .args(["-c", &format!("\"$0\" --approve {command}"), openmax_bin()])
        .current_dir(&project)
        .env("HOME", &home)
        .env_remove("OPENMAX_SESSION")
        .env("OPENMAX_HUMAN_ATTEST", "1")
        .output()
        .unwrap();
    assert_eq!(approve.status.code(), Some(0), "{}", String::from_utf8_lossy(&approve.stderr));
    let (code, report) = check(&project, &["--check"]);
    assert_eq!(code, Some(0), "{report}");
    assert!(report.starts_with("ok: 1 tool"), "{report}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    let _ = std::fs::remove_dir_all(home.parent().unwrap());
}

/// `--check --run-examples` is the one path that executes project code, so it
/// carries a session's gates: trust, then content approval. The JSON face
/// reports the same verdicts, because the consumer most likely to parse it is
/// an agent verifying a tool it just wrote.
#[test]
fn run_examples_is_gated_and_reported_through_json() {
    let (project, home) = fresh_dirs("examples");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    // Echoes its stdin JSON back; its example proves the payload arrived.
    std::fs::write(
        tools.join("prover.toml"),
        "name = \"prover\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"cat\"]\n\n[example]\nexpect_regex = \"hello\"\n[example.args]\nmsg = \"hello\"\n",
    )
    .unwrap();
    std::fs::write(
        tools.join("failer.toml"),
        "name = \"failer\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"echo boom >&2; exit 3\"]\n\n[example]\n",
    )
    .unwrap();

    let json = |out: &std::process::Output| -> serde_json::Value {
        serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stdout)))
    };
    let messages = |value: &serde_json::Value| -> String {
        value
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["surface"] == "example")
            .map(|row| format!("{} {}", row["status"], row["message"]))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Untrusted: plain --check still passes (it only reads), examples do not.
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stdout));
    let out = cmd(&project, &home)
        .args(["--check", "--json", "--run-examples"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(messages(&json(&out)).contains("not trusted"), "{}", messages(&json(&out)));

    // Trust it in ask: the sandbox and approval gates below are ask's.
    trust_in_ask(&project, &home);

    // Trusted but unapproved: each example probes in a sandbox with zero
    // host authority instead of refusing flat. The harmless prover passes
    // (marked sandboxed, with the approve pointer), the broken failer fails
    // with its own diagnostic, and nothing is blessed by any of it. On a
    // host with no sandbox backend, the fall-back refusal keeps the old
    // unapproved-source wording; both are exit 1 here (failer always fails).
    // A sandboxed probe can PROVE a tool passes, but a non-pass is
    // inconclusive: the sandbox denies the network and non-scratch writes, so
    // a tool that needs either cannot pass one, and a failure there is not
    // proof the tool is broken (a passing probe approves nothing, so a failing
    // one condemns nothing). So an unapproved sandbox non-pass is a `warn` that
    // does NOT fail the check - the honest verdict is the approved host run
    // below, where the failer errs. Exit is 0 because nothing ran with host
    // authority. (The failer here has no network, so on a backend host it runs
    // and exits nonzero: still inconclusive by this contract, warned not erred.)
    let out = cmd(&project, &home)
        .args(["--check", "--json", "--run-examples"])
        .output()
        .unwrap();
    let value = json(&out);
    let reported = messages(&value);
    assert_eq!(
        out.status.code(),
        Some(0),
        "unapproved sandbox non-pass must not fail the check: {reported}"
    );
    assert!(
        value
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["surface"] == "example")
            .all(|row| row["sandboxed"] == true),
        "unapproved probes must be labeled: {value}"
    );
    if reported.contains("cannot sandbox a probe") {
        // No sandbox backend on this host: both probes are inconclusive warns.
        assert_eq!(reported.matches("\"warn\"").count(), 2, "{reported}");
        assert!(reported.contains("--approve"), "{reported}");
    } else {
        assert!(reported.contains("ran in a sandbox"), "{reported}"); // prover passed
        assert!(reported.contains("could not prove"), "{reported}"); // failer inconclusive
        assert!(reported.contains("\"warn\""), "{reported}");
    }

    for tool in ["prover.toml", "failer.toml"] {
        let out = cmd(&project, &home)
            .args(["--approve", &format!(".openmax/tools/{tool}")])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    }

    // Approved: host runs now, unlabeled - the passing example passes, the
    // failing one fails the run and brings its diagnostic with it.
    let out = cmd(&project, &home)
        .args(["--check", "--json", "--run-examples"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let value = json(&out);
    let reported = messages(&value);
    assert!(reported.contains("ok"), "{reported}");
    assert!(reported.contains("boom"), "{reported}");
    assert!(reported.contains("exit code 3"), "{reported}");
    assert!(
        value
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["surface"] == "example")
            .all(|row| row["sandboxed"] == false),
        "approved content keeps unlabeled host runs: {value}"
    );

    // Without --check the flag would be silently swallowed; that reads as
    // success for work that never ran.
    let out = cmd(&project, &home).arg("--run-examples").output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("--run-examples requires --check"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--approve` blesses a manifest and the code it runs in one act, and says
/// so: a human cannot approve bytes they were not shown. A named command that
/// does not exist is refused rather than half-approved.
#[test]
fn approve_names_every_file_it_blesses() {
    let (project, home) = fresh_dirs("approve");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(
        hooks.join("gate.toml"),
        "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n",
    )
    .unwrap();

    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "a missing command must not be half-approved");
    assert!(stderr.contains("gate.sh"), "{stderr}");

    std::fs::write(project.join("gate.sh"), "#!/bin/sh\nexit 1\n").unwrap();
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("approved .openmax/hooks/gate.toml"), "{stdout}");
    assert!(stdout.contains("gate.sh"), "the code it runs must be named: {stdout}");
    assert!(
        stdout.contains("this records a hook on pre_tool_use"),
        "the receipt must say what shape was activated: {stdout}"
    );

    // The shape a human most needs to see is the one an agent most often
    // mis-describes: a turn_end file without `blocking` handed over as a
    // completion gate. The receipt says what it will not do.
    std::fs::write(
        hooks.join("watch.toml"),
        "event = \"turn_end\"\ncommand = \"./gate.sh\"\n",
    )
    .unwrap();
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/watch.toml"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    assert!(
        stdout.contains("observes only") && stdout.contains("`blocking = true`"),
        "approving a turn_end observer must say exit status is ignored: {stdout}"
    );

    // The pair is live, and rewriting the script alone revokes it.
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stdout));
    std::fs::write(project.join("gate.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{stdout}");
    assert!(stdout.contains("gate.sh"), "{stdout}");
}

/// An inert allow names the command that approves it, and `--approve` records
/// the approval for the directory it runs in. The global file's path names no
/// project, so a bare `openmax --approve ~/.openmax/permissions.toml` pasted
/// into a terminal opened elsewhere (often $HOME) printed "approved" for that
/// directory and left the project prompting. The printed command has to work
/// from wherever it is pasted.
#[test]
fn the_printed_global_allow_approval_works_from_another_directory() {
    let (project, home) = fresh_dirs("global-allow");
    trust_in_ask(&project, &home);
    std::fs::write(
        home.join(".openmax").join("permissions.toml"),
        "[[rules]]\neffect = \"allow\"\ntool = \"bash\"\n",
    )
    .unwrap();
    let inert = || -> Option<String> {
        let out = cmd(&project, &home).args(["--check", "--json"]).output().unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stdout)));
        rows.as_array()
            .unwrap()
            .iter()
            .filter_map(|row| row["message"].as_str())
            .find(|message| message.contains("are inert"))
            .map(str::to_string)
    };

    let notice = inert().expect("an unapproved global allow is inert in ask");
    let command = notice.rsplit('`').nth(1).unwrap_or_else(|| panic!("no command in {notice}"));
    let bin_dir = Path::new(openmax_bin()).parent().unwrap();
    let out = Command::new("/bin/sh")
        .args(["-c", command])
        .current_dir(&home)
        .env("HOME", &home)
        .env("PATH", format!("{}:/usr/bin:/bin", bin_dir.display()))
        .env_remove("OPENMAX_SESSION")
        .env("OPENMAX_HUMAN_ATTEST", "1")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(out.status.success(), "{command}: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(inert(), None, "`{command}` run from {} must approve it for the project", home.display());
}

/// A permissions file is read through a link that resolves to a regular
/// file. Once the link stops resolving, it fails closed instead of reading
/// as no file, which would drop its rules without a word.
#[cfg(unix)]
#[test]
fn a_linked_permissions_file_loads_and_fails_closed_once_it_dangles() {
    let (project, home) = fresh_dirs("linked-permissions");
    std::fs::create_dir_all(project.join(".openmax")).unwrap();
    std::fs::write(
        project.join("rules.toml"),
        "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\n",
    )
    .unwrap();
    std::os::unix::fs::symlink("../rules.toml", project.join(".openmax/permissions.toml")).unwrap();
    let check = || -> (Option<i32>, String) {
        let out = cmd(&project, &home).args(["--check", "--json"]).output().unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stdout)));
        let row = rows
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["surface"] == "permissions")
            .unwrap_or_else(|| panic!("no permissions row: {rows}"));
        let (status, message) = (row["status"].as_str().unwrap(), row["message"].as_str().unwrap());
        (out.status.code(), format!("{status} {message}"))
    };

    assert_eq!(check(), (Some(0), "ok 1 rules".to_string()));
    std::fs::remove_file(project.join("rules.toml")).unwrap();
    assert_eq!(check(), (Some(1), "err not a readable regular file; failing closed".to_string()));
}

/// Deleting an approved hook fails every tool call closed, so a human who
/// meant the removal needs a way to say so. `--forget` is that way, and it is
/// guarded harder than `--approve` because it removes a policy instead of
/// adding one: an agent session is refused, and so is any run without an
/// interactive terminal - which is what a `bash` subprocess inside a turn and
/// this test harness both look like. Neither check is a sandbox; see the
/// residual stated at the call site.
#[test]
fn forget_refuses_without_a_human_at_a_terminal() {
    let (project, home) = fresh_dirs("forget");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(
        hooks.join("gate.toml"),
        "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n",
    )
    .unwrap();
    cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();

    std::fs::remove_file(hooks.join("gate.toml")).unwrap();
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "a deleted gate must not read as a clean project: {stdout}");
    assert!(stdout.contains("deleted"), "{stdout}");

    // Agent-spawned processes cannot retire a human's approval.
    let out = cmd(&project, &home)
        .env("OPENMAX_SESSION", "1")
        .args(["--forget", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&out.stderr).contains("human actions"));

    // And neither can anything else without a terminal, marker or not: the
    // marker is one `unset` away from any shell the agent already has.
    let out = cmd(&project, &home)
        .args(["--forget", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "stdout: {}", String::from_utf8_lossy(&out.stdout));
    assert!(stderr.contains("interactive terminal"), "{stderr}");
    // The refusal has to leave the human a way forward that does not need one.
    assert!(stderr.contains("restore the file"), "{stderr}");

    // Refused means refused: the gate is still expected, so tools still fail
    // closed and --check still reports it.
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stdout));

    // The path the refusal names does work: the file itself is the record, and
    // restoring it is the repair the harness prefers.
    std::fs::write(
        hooks.join("gate.toml"),
        "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n",
    )
    .unwrap();
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "restoring the approved bytes must clear the fail-closed state: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// A ledger nobody can verify refuses to be extended and names the way
/// back - which is a human at an interactive terminal: agent sessions
/// and terminal-less runs are both refused, and the fail-closed state
/// survives the refusal. The quarantine itself is proven at the unit level,
/// where no confirmation prompt stands in the way.
#[test]
fn an_unverifiable_ledger_is_refused_and_repairable() {
    let (project, home) = fresh_dirs("ledger-repair");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    // /usr/bin/true exists on both CI platforms; /bin/true is Linux-only.
    std::fs::write(
        hooks.join("gate.toml"),
        "event = \"pre_tool_use\"\ncommand = \"/usr/bin/true\"\n",
    )
    .unwrap();
    // An approval is a ledger record, so this is also what writes the chain.
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));

    let ledger = home.join(".openmax").join("ledger");
    let dir = std::fs::read_dir(&ledger)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a ledger directory");
    // The easiest tamper there is: delete the pin.
    std::fs::remove_file(dir.join("chain-head")).unwrap();

    // Approving again cannot extend a chain nobody can verify, and the
    // refusal names the way back.
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("--ledger-repair"), "the way back must be named: {stderr}");

    // The gate it approved is revoked while the chain cannot be trusted.
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stdout));

    let out = cmd(&project, &home)
        .arg("--ledger-repair")
        .env("OPENMAX_SESSION", "s-1")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3), "repair is a human action");

    // The marker is one `unset` away from any shell the agent already has,
    // so a terminal stands behind it - but the stakes still print first, so
    // even a refused run says what repair would set aside.
    let out = cmd(&project, &home).arg("--ledger-repair").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stdout}");
    assert!(stderr.contains("interactive terminal"), "{stderr}");
    assert!(stdout.contains("set aside"), "the stakes must print before the refusal: {stdout}");

    // Refused means refused: the chain is still unverifiable and --check
    // still fails closed.
    let out = cmd(&project, &home).arg("--check").output().unwrap();
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stdout));
    assert!(
        !std::fs::read_dir(&dir).unwrap().flatten().any(|e| {
            e.file_name().to_string_lossy().starts_with("log.jsonl.unverified-")
        }),
        "a refused repair must move nothing"
    );
}

/// Capability-file history is no longer recorded, so `--ledger` has nothing
/// to print. It stays accepted so a script that calls it keeps working: one
/// line says so and names where earlier records and objects stay, and
/// nothing there is touched.
#[test]
fn ledger_is_deprecated_and_leaves_existing_records_in_place() {
    let (project, home) = fresh_dirs("ledger-deprecated");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(hooks.join("gate.toml"), "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n")
        .unwrap();
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(approval_records(&home), 1);
    let dir = std::fs::read_dir(home.join(".openmax").join("ledger"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a ledger directory");
    // An object an earlier build stored.
    std::fs::create_dir_all(dir.join("objects")).unwrap();
    std::fs::write(dir.join("objects").join("a".repeat(64)), "earlier bytes").unwrap();
    let log = std::fs::read(dir.join("log.jsonl")).unwrap();

    let out = cmd(&project, &home).arg("--ledger").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(stdout.contains("no longer recorded"), "{stdout}");
    assert!(stdout.contains(&dir.display().to_string()), "where the records stay: {stdout}");
    assert_eq!(
        std::fs::read_to_string(dir.join("objects").join("a".repeat(64))).unwrap(),
        "earlier bytes",
        "stored objects are kept"
    );
    assert_eq!(std::fs::read(dir.join("log.jsonl")).unwrap(), log, "the log is kept as it was");
}

/// Each early-exit operation runs and exits before the next one is
/// considered, so a second operation on the same command line was dropped
/// without a word: `--check --ledger` printed history and exited 0 having
/// validated nothing. Any two operations are refused before either runs, with
/// the usage exit and a first line that names both.
#[test]
fn two_operations_on_one_command_line_are_refused_naming_both() {
    let (project, home) = fresh_dirs("two-operations");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(
        hooks.join("gate.toml"),
        "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n",
    )
    .unwrap();
    let cases: [(&[&str], [&str; 2]); 3] = [
        (&["--check", "--ledger"], ["--check", "--ledger"]),
        (&["-p", "x", "--approve", ".openmax/hooks/gate.toml"], ["--print", "--approve"]),
        (&["--stdio", "--forget", ".openmax/hooks/gate.toml"], ["--stdio", "--forget"]),
    ];
    let mut wrong = Vec::new();
    for (args, named) in cases {
        let out = cmd(&project, &home).args(args).stdin(Stdio::null()).output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        // The usage text that follows a refusal names every flag, so only the
        // first line shows whether the refusal named this conflict.
        let first = stderr.lines().next().unwrap_or_default();
        if out.status.code() != Some(2) || !named.iter().all(|flag| first.contains(flag)) {
            wrong.push(format!("{args:?} exited {:?}: {first}", out.status.code()));
        }
    }
    assert!(wrong.is_empty(), "each pair must exit 2 naming both:\n{}", wrong.join("\n"));
    // Refused before either ran: the approval was never recorded.
    assert!(
        !home.join(".openmax").join("ledger").exists(),
        "a refused --approve must record nothing"
    );
}

#[test]
fn stdio_handshake_speaks_the_contract() {
    let (project, home) = fresh_dirs("stdio");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let mut child = cmd(&project, &home)
        .args(["--trust-project", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    let mut hello = String::new();
    reader.read_line(&mut hello).unwrap();
    let hello: serde_json::Value = serde_json::from_str(&hello).unwrap();
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["proto"], "openmax-stdio/8");
    assert_eq!(hello["protocol_version"], 8);
    assert!(hello["session_id"].is_string());

    writeln!(stdin, r#"{{"cmd":"quit"}}"#).unwrap();
    drop(stdin);
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(0));
}

/// Read one HTTP request (headers, then exactly Content-Length body bytes)
/// off `stream`, returning the body. None when the peer went away mid-request.
fn read_request(stream: &mut std::net::TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => buf.push(byte[0]),
            _ => return None,
        }
    }
    let headers = String::from_utf8_lossy(&buf).to_string();
    let content_length: usize = headers
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse().ok())?
        })
        .unwrap_or(0);
    let mut body = vec![0u8; content_length];
    if content_length > 0 && stream.read_exact(&mut body).is_err() {
        return None;
    }
    Some(String::from_utf8_lossy(&body).to_string())
}

/// One canned SSE completion per request, in script order. `finished` says
/// whether the body is framed with a Content-Length (a complete response) or
/// simply cut off by closing the socket, which is what a provider dying
/// mid-answer looks like on the wire: a well-formed transfer whose completion
/// signal never arrives. Every request body is captured so a test can assert
/// what the model was actually sent.
fn spawn_scripted_server(
    script: Vec<(String, bool)>,
) -> (String, Arc<Mutex<Vec<String>>>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let requests: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    let handle = std::thread::spawn(move || {
        for (sse, finished) in script {
            let Ok((mut stream, _)) = listener.accept() else { return };
            let Some(body) = read_request(&mut stream) else { return };
            seen.lock().unwrap().push(body);
            let response = if finished {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{sse}",
                    sse.len(),
                )
            } else {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse}"
                )
            };
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1"), requests, handle)
}

/// A finished one-line answer.
const HELLO_SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"stub says hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":3}}\n\n",
    "data: [DONE]\n\n",
);

/// One syntactically complete `write_file` call: nothing is half-written, so
/// only the missing completion signal distinguishes it from a real request.
const WRITE_CALL_SSE: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\":\\\"side-effect.txt\\\",\\\"content\\\":\\\"written\\\"}\"}}]}}]}\n\n";

const TOOL_CALLS_TERMINATOR: &str = concat!(
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// Minimal OpenAI-compatible streaming endpoint: the same finished completion
/// for a few requests, enough for a whole print-mode turn.
fn spawn_stub_server() -> (String, Arc<Mutex<Vec<String>>>, std::thread::JoinHandle<()>) {
    spawn_scripted_server(vec![(HELLO_SSE.to_string(), true); 4])
}

#[test]
fn a_print_turn_against_a_stub_server_reaches_stdout() {
    let (project, home) = fresh_dirs("turn");
    let (base_url, _requests, _server) = spawn_stub_server();
    write_settings(&home, &base_url);

    let out = cmd(&project, &home)
        .args(["--trust-project", "-p", "say hi"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("stub says hi"), "stdout: {stdout}\nstderr: {stderr}");
}

/// The first error a local-model user meets is a server not started yet.
/// Print mode says so in lines a human acts on: each retry names the server
/// and the cause and counts to the attempt the request really ends on, and
/// the error says what to do, once. It used to print reqwest's whole source
/// chain three times, count retries to a budget of 8 that a refused address
/// never reaches, and repeat the failure as "stopped (error)".
///
/// Each run waits out the real backoff, so the three runs go at once.
#[test]
fn refused_connection_names_the_host_and_the_cause() {
    let (project, home) = fresh_dirs("refused");
    let addr = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    write_settings(&home, &format!("http://{addr}/v1"));
    let start = |env: &[(&str, &str)]| {
        let mut command = cmd(&project, &home);
        for var in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"] {
            command.env_remove(var);
        }
        command.envs(env.iter().copied());
        command.args(["--trust-project", "-p", "say hi"]).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap()
    };
    let finish = |child: std::process::Child| {
        let out = finish_with_deadline(child);
        assert_eq!(out.status.code(), Some(1), "stderr: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8_lossy(&out.stderr).into_owned()
    };
    // Through a proxy that refuses, the refusal is the proxy's; a proxy
    // that NO_PROXY exempts the endpoint from never sees the request.
    let proxy = format!("http://{}", TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap());
    let direct = start(&[]);
    let proxied = start(&[("HTTP_PROXY", &proxy)]);
    let exempted = start(&[("HTTP_PROXY", &proxy), ("NO_PROXY", "127.0.0.1")]);

    let reason = format!("cannot connect to {addr} (connection refused)");
    let expected = [
        format!("openmax: {reason}; retrying (2 of 3)"),
        format!("openmax: {reason}; retrying (3 of 3)"),
        format!("openmax: error: {reason}: nothing is listening there; start the server or fix base_url (openmax --spec settings)"),
    ];
    let failure = |stderr: &str| -> Vec<String> {
        stderr.lines().skip_while(|line| !line.contains(&reason)).map(str::to_string).collect()
    };
    let stderr = finish(direct);
    assert_eq!(failure(&stderr), expected, "stderr: {stderr}");

    // Naming the endpoint and advising to start it would send the user the
    // wrong way when the refusal is the proxy's, so the error keeps the
    // client's own account.
    let stderr = finish(proxied);
    assert!(!stderr.contains("cannot connect to") && !stderr.contains("start the server"), "stderr: {stderr}");
    assert!(stderr.contains("openmax: error: request failed: error sending request"), "stderr: {stderr}");

    // The common setup for a local server behind a proxy.
    let stderr = finish(exempted);
    assert_eq!(failure(&stderr), expected, "stderr: {stderr}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Prompt templates are a harness feature, not a TUI one: the delegate
/// pattern (`openmax -p` in a child process) must send the model the template
/// body, never the literal slash line.
#[test]
fn a_print_turn_expands_a_prompt_template() {
    let (project, home) = fresh_dirs("template");
    let (base_url, requests, _server) = spawn_stub_server();
    write_settings(&home, &base_url);
    let prompts = project.join(".agents").join("prompts");
    std::fs::create_dir_all(&prompts).unwrap();
    std::fs::write(prompts.join("greet.md"), "MARKER: greet $ARGUMENTS\n").unwrap();

    let out = cmd(&project, &home)
        .args(["--trust-project", "-p", "/greet world"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr: {stderr}");

    let sent = requests.lock().unwrap().join("\n");
    assert!(sent.contains("MARKER: greet world"), "the model must get the body: {sent}");
    assert!(!sent.contains("/greet world"), "the raw slash line must not be sent: {sent}");
}

/// One scripted SSE completion per request, and every request body appended to
/// `record`, so a test can assert both what openmax did and what the model was
/// told afterwards. Records to a file (not memory) so a test can read the wire
/// after the run without holding the server handle.
fn spawn_recording_server(
    bodies: Vec<String>,
    record: PathBuf,
) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        for sse in bodies {
            let Ok((mut stream, _)) = listener.accept() else { return };
            let mut buf = Vec::new();
            let mut byte = [0u8; 1];
            while !buf.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => buf.push(byte[0]),
                    _ => return,
                }
            }
            let headers = String::from_utf8_lossy(&buf).to_string();
            let content_length: usize = headers
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse().ok())?
                })
                .unwrap_or(0);
            let mut body = vec![0u8; content_length];
            if content_length > 0 && stream.read_exact(&mut body).is_err() {
                return;
            }
            let mut log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&record)
                .unwrap();
            log.write_all(&body).unwrap();
            log.write_all(b"\n").unwrap();

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{sse}",
                sse.len(),
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1"), handle)
}

fn sse(chunks: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str(&format!("data: {chunk}\n\n"));
    }
    out.push_str("data: [DONE]\n\n");
    out
}

fn sse_tool_call(name: &str, args: serde_json::Value) -> String {
    sse_tool_calls(&[(name, args)])
}

/// One assistant message carrying several tool calls, the shape that routes
/// consecutive read-only calls into the concurrent batch path.
fn sse_tool_calls(calls: &[(&str, serde_json::Value)]) -> String {
    let deltas: Vec<serde_json::Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (name, args))| {
            serde_json::json!({
                "index": i, "id": format!("call_{i}"), "type": "function",
                "function": {"name": name, "arguments": args.to_string()}
            })
        })
        .collect();
    sse(&[
        serde_json::json!({"choices":[{"delta":{"tool_calls":deltas},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
    ])
}

fn sse_text(text: &str) -> String {
    sse(&[
        serde_json::json!({"choices":[{"delta":{"content":text},"finish_reason":null}]}),
        serde_json::json!({"choices":[{"delta":{},"finish_reason":"stop"}]}),
    ])
}

/// Emitting the same unapproved tool twice in one message is the bypass:
/// consecutive external non-mutating calls are routed to the concurrent batch
/// path, which has no approval UI. The gate has to survive that routing, so
/// both calls must land on the serial path and prompt.
#[test]
fn two_calls_to_an_unapproved_tool_cannot_batch_past_the_gate() {
    let (project, home) = fresh_dirs("unapproved-batch");
    let record = project.parent().unwrap().join("requests.jsonl");
    let (base_url, _server) = spawn_recording_server(
        vec![
            sse_tool_calls(&[
                ("peek", serde_json::json!({"count": 1})),
                ("peek", serde_json::json!({"count": 2})),
            ]),
            sse_text("blocked"),
        ],
        record,
    );
    write_settings_with_mode(&home, &base_url, "ask");

    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(
        tools.join("peek.toml"),
        "name = \"peek\"\ndescription = \"look something up\"\ncommand = \"/bin/sh\"\n\
         args = [\"-c\", \"cat >/dev/null; echo ran >> peeked.txt; echo looked\"]\nmutating = false\n\
         \n[params]\ntype = \"object\"\n[params.properties.count]\ntype = \"number\"\n",
    )
    .unwrap();

    trust_in_ask(&project, &home);
    let out = cmd(&project, &home)
        .args(["--json", "-p", "peek twice"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        !project.join("peeked.txt").exists(),
        "batching must not run unapproved host code\nstdout: {stdout}\nstderr: {stderr}"
    );

    let events: Vec<serde_json::Value> =
        stdout.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let prompts: Vec<&serde_json::Value> =
        events.iter().filter(|e| e["type"] == "approval_request").collect();
    assert_eq!(prompts.len(), 2, "each call takes the serial path: {stdout}");
    for prompt in prompts {
        assert_eq!(prompt["reason"], "unapproved_source");
        assert_eq!(prompt["source_path"], ".openmax/tools/peek.toml");
    }
}

/// The human content boundary covers every agent-written tool, not only the
/// ones that declare `mutating` - that field is written by the agent, while the
/// call itself is a native host process. The refusal must also be actionable:
/// the event, the operator's stderr, and the model's own tool result all have
/// to name the file and the command that unblocks it.
#[test]
fn a_read_only_agent_written_tool_is_gated_until_a_human_approves_it() {
    let (project, home) = fresh_dirs("unapproved-tool");
    let record = project.parent().unwrap().join("requests.jsonl");
    let (base_url, _server) = spawn_recording_server(
        vec![
            sse_tool_call("peek", serde_json::json!({"count": 3})),
            sse_text("blocked"),
            sse_tool_call("peek", serde_json::json!({"count": 3})),
            sse_text("ran it"),
        ],
        record.clone(),
    );
    // The tool declares itself non-mutating, so only its content gate asks.
    write_settings_with_mode(&home, &base_url, "ask");

    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(
        tools.join("peek.toml"),
        // Declares itself read-only and takes no string arguments: the exact
        // shape that used to bypass the gate and summarize as "".
        "name = \"peek\"\ndescription = \"look something up\"\ncommand = \"/bin/sh\"\n\
         args = [\"-c\", \"cat >/dev/null; echo ran > peeked.txt; echo looked\"]\nmutating = false\n\
         \n[params]\ntype = \"object\"\n[params.properties.count]\ntype = \"number\"\n",
    )
    .unwrap();

    trust_in_ask(&project, &home);
    let out = cmd(&project, &home)
        .args(["--json", "-p", "peek at it"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        !project.join("peeked.txt").exists(),
        "unapproved host code must not have run\nstdout: {stdout}\nstderr: {stderr}"
    );

    let events: Vec<serde_json::Value> =
        stdout.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let request = events
        .iter()
        .find(|e| e["type"] == "approval_request")
        .unwrap_or_else(|| panic!("a read-only external tool must still ask: {stdout}"));
    assert_eq!(request["reason"], "unapproved_source");
    assert_eq!(request["source_path"], ".openmax/tools/peek.toml");
    assert_eq!(request["source_sha"].as_str().unwrap().len(), 12);
    assert_eq!(request["summary"], "peek", "a summary must never be empty");

    // The operator running headless gets a command, not a placeholder, and
    // the path is quoted the way every other printed --approve line is, so
    // a file the agent named survives the paste.
    assert!(
        stderr.contains("openmax --approve '.openmax/tools/peek.toml'"),
        "stderr must name the real file, quoted: {stderr}"
    );
    assert!(!stderr.contains("<its .toml>"), "{stderr}");

    // And so does the model: the harness enforced a boundary, no user declined.
    let sent = std::fs::read_to_string(&record).unwrap();
    let last: serde_json::Value =
        serde_json::from_str(sent.lines().nth(1).expect("a second request")).unwrap();
    let tool_result = last["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the declined call is reported back")["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        tool_result.contains("openmax --approve .openmax/tools/peek.toml"),
        "the agent must be able to relay the exact command: {tool_result}"
    );
    assert!(!tool_result.contains("The user declined"), "{tool_result}");

    // A human approves the exact bytes; the same call then runs unprompted.
    let approve = cmd(&project, &home)
        .args(["--approve", ".openmax/tools/peek.toml"])
        .output()
        .unwrap();
    assert_eq!(approve.status.code(), Some(0), "{}", String::from_utf8_lossy(&approve.stderr));

    let out = cmd(&project, &home)
        .args(["--json", "-p", "peek at it"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("approval_request"),
        "approved content must not ask again: {stdout}"
    );
    assert!(project.join("peeked.txt").exists(), "the approved tool must run: {stdout}");
}

#[test]
fn a_json_print_turn_emits_valid_envelopes_ending_in_done() {
    let (project, home) = fresh_dirs("json");
    let (base_url, _requests, _server) = spawn_stub_server();
    write_settings(&home, &base_url);

    let out = cmd(&project, &home)
        .args(["--trust-project", "--json", "-p", "say hi"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("every stdout line is JSON"))
        .collect();
    assert!(!lines.is_empty());
    for line in &lines {
        assert!(line["session_id"].is_string(), "{line}");
        assert!(line["type"].is_string(), "{line}");
    }
    let last = lines.last().unwrap();
    assert_eq!(last["type"], "done", "the stream must end in done: {stdout}");
    assert!(
        lines.iter().any(|l| l["type"] == "message_done" && l["text"] == "stub says hi"),
        "{stdout}"
    );
}

/// A harness note is addressed to the model, which reads it in its own
/// transcript, so print mode keeps it off stderr. A parent agent that runs
/// `openmax -p` reads the child's stderr in its bash result, and the child's
/// turn-start execution policy note used to arrive there on every call.
/// Tool progress stays on stderr, and `--json` still carries the note as a
/// `harness_note` line.
#[test]
fn a_print_run_keeps_harness_notes_off_stderr() {
    let (project, home) = fresh_dirs("notes-off-stderr");
    std::fs::write(project.join("a.txt"), "alpha\n").unwrap();
    let (base_url, requests, _server) = spawn_scripted_server(vec![
        (sse_tool_call("read_file", serde_json::json!({"path": "a.txt"})), true),
        (sse_text("read it"), true),
        (sse_text("read it"), true),
    ]);
    write_settings(&home, &base_url);
    // Trusted before the run, as a parent's project is: `--trust-project` on
    // the run itself prints its own grant line.
    open_max_core::trust::grant_trust(&home.join(".openmax"), &project, ApprovalMode::Auto).unwrap();

    let out = cmd(&project, &home).args(["-p", "read a.txt"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stderr: {stderr}");
    let progress = stderr.lines().filter(|l| l.starts_with("→ ") || l.starts_with("← ")).count();
    assert_eq!(progress, 2, "the call's progress still reaches stderr: {stderr}");
    assert_eq!(stderr.lines().count(), progress, "stderr carries tool progress only: {stderr}");
    let first = requests.lock().unwrap()[0].clone();
    assert!(first.contains("[execution policy: auto"), "the model still reads the note: {first}");

    let out = cmd(&project, &home).args(["--json", "-p", "say hi"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}");
    let note = stdout
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("every stdout line is JSON"))
        .find(|l| l["type"] == "harness_note")
        .unwrap_or_else(|| panic!("--json carries the note: {stdout}"));
    assert!(note["text"].as_str().unwrap().starts_with("[execution policy: auto"), "{note}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// `--model` names the model a run asks for, over the one settings.json
/// names: the request a print run sends carries the flag's.
#[test]
fn the_model_flag_names_the_model_a_run_asks_for() {
    let (project, home) = fresh_dirs("model-flag");
    let (base_url, requests, _server) = spawn_scripted_server(vec![(sse_text("hi"), true)]);
    write_settings(&home, &base_url);
    open_max_core::trust::grant_trust(&home.join(".openmax"), &project, ApprovalMode::Auto).unwrap();

    let out = cmd(&project, &home).args(["-m", "flag-model", "-p", "say hi"]).output().unwrap();
    assert_eq!(out.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let sent: serde_json::Value = serde_json::from_str(&requests.lock().unwrap()[0]).unwrap();
    assert_eq!(sent["model"], "flag-model", "settings.json names stub-model: {sent}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Run `openmax` with `args` and `commands` on stdin, then EOF, which quits a
/// stdio session once its turn in flight drains, and hand back its stdout
/// lines with the session's id, taken from `hello`, replaced by `S`.
fn stdio_lines(project: &Path, home: &Path, args: &[&str], commands: &[&str]) -> Vec<String> {
    let mut child = cmd(project, home)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for command in commands {
        writeln!(stdin, "{command}").unwrap();
    }
    drop(stdin);
    let out = finish_with_deadline(child);
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(0), "{stdout}{}", String::from_utf8_lossy(&out.stderr));
    let hello: serde_json::Value = serde_json::from_str(stdout.lines().next().unwrap()).unwrap();
    let id = hello["session_id"].as_str().unwrap();
    stdout.lines().map(|line| line.replace(id, "S")).collect()
}

/// `--stdio` and `--print --json` render one event stream, and
/// docs/stdio-protocol.md gives every event line as `session_id`, then
/// `type`, then the fields in declaration order. So one turn is framed in
/// the same bytes by both, session id aside, and the lines stdio writes
/// itself carry `type` first, exactly as the doc shows each one. The stdio
/// frames used to come out with their keys sorted, so a client comparing or
/// hashing lines, or reading a captured print run with its stdio parser, met
/// two different wires.
#[test]
fn stdio_frames_a_turn_in_the_bytes_print_json_prints() {
    let (project, home) = fresh_dirs("wire-order");
    std::fs::write(project.join("a.txt"), "hello\n").unwrap();
    let turn = [
        (sse_tool_call("read_file", serde_json::json!({"path": "a.txt"})), true),
        (sse_text("stub says hi"), true),
    ];
    let (base_url, _requests, _server) = spawn_scripted_server([turn.clone(), turn].concat());
    write_settings_with_mode(&home, &base_url, "auto");

    let (code, lines, stdout) = json_turn(&project, &home, "read it");
    assert_eq!(code, Some(0), "{stdout}");
    let id = lines[0]["session_id"].as_str().unwrap();
    let printed: Vec<String> = stdout.lines().map(|line| line.replace(id, "S")).collect();
    // The doc's example event, as print mode writes it.
    assert!(
        printed.iter().any(|line| line
            == r#"{"session_id":"S","type":"tool_start","call_id":"call_0","name":"read_file","args":{"path":"a.txt"}}"#),
        "{printed:#?}"
    );

    let framed = stdio_lines(&project, &home, &["--stdio"], &[r#"{"cmd":"user","text":"read it"}"#]);
    let hello: serde_json::Value = serde_json::from_str(&framed[0]).unwrap();
    assert_eq!(
        framed[0],
        format!(
            r#"{{"type":"hello","proto":"openmax-stdio/8","protocol_version":8,"session_id":"S","version":"{}","project":{},"continued":false}}"#,
            env!("CARGO_PKG_VERSION"),
            hello["project"],
        )
    );
    assert_eq!(framed[1..], printed[..], "stdio and print mode framed one turn differently");

    let resumed = stdio_lines(&project, &home, &["--continue", "--stdio"], &[]);
    assert_eq!(resumed.len(), 2, "{resumed:#?}");
    let history: serde_json::Value = serde_json::from_str(&resumed[1]).unwrap();
    let messages = history["messages"].as_array().unwrap();
    assert_eq!(messages[messages.len() - 2..], [
        serde_json::json!({"role": "user", "content": "read it"}),
        serde_json::json!({"role": "assistant", "content": "stub says hi"}),
    ]);
    let messages: Vec<String> = messages
        .iter()
        .map(|m| format!(r#"{{"role":{},"content":{}}}"#, m["role"], m["content"]))
        .collect();
    assert_eq!(
        resumed[1],
        format!(
            r#"{{"type":"transcript","session_id":"S","messages":[{}],"truncated":false}}"#,
            messages.join(",")
        )
    );

    let own = stdio_lines(
        &project,
        &home,
        &["--stdio"],
        &[r#"{"cmd":"approval_mode","mode":"auto"}"#, r#"{"cmd":"reload"}"#, r#"{"cmd":"user","text":" "}"#],
    );
    let refrozen: serde_json::Value = serde_json::from_str(&own[2]).unwrap();
    assert_eq!(
        own[1..],
        [
            r#"{"type":"approval_mode","mode":"auto"}"#.to_string(),
            format!(
                r#"{{"session_id":"S","type":"refrozen","tools":{},"skills":{},"changes":[]}}"#,
                refrozen["tools"], refrozen["skills"],
            ),
            r#"{"type":"protocol_error","message":"user text is empty"}"#.to_string(),
            r#"{"session_id":"S","type":"done","stop_reason":"refused"}"#.to_string(),
        ]
    );
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Drive one `--stdio` process, started with `args` as well, a step at a
/// time: send a command, then read stdout through the line that settles it
/// (`done` for a turn, `forked` for a branch), so no turn is refused for
/// overlapping the one before it. Every line read comes back parsed, `hello`
/// first.
fn stdio_steps(project: &Path, home: &Path, args: &[&str], commands: &[&str]) -> Vec<serde_json::Value> {
    let mut child = cmd(project, home)
        .args(["--trust-project", "--stdio"])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut lines: Vec<serde_json::Value> = Vec::new();
    let mut next = |lines: &mut Vec<serde_json::Value>| {
        let mut line = String::new();
        assert!(stdout.read_line(&mut line).unwrap() > 0, "stdio ended early after {lines:#?}");
        lines.push(serde_json::from_str(&line).unwrap());
        lines.last().unwrap()["type"].as_str().unwrap_or("").to_string()
    };
    next(&mut lines);
    for command in commands {
        writeln!(stdin, "{command}").unwrap();
        while !matches!(next(&mut lines).as_str(), "done" | "forked" | "protocol_error") {}
    }
    drop(stdin);
    let out = finish_with_deadline(child);
    assert_eq!(out.status.code(), Some(0), "{lines:#?}{}", String::from_utf8_lossy(&out.stderr));
    lines
}

/// The user and assistant text of a message list, in order: what a request
/// body or a transcript carries once the system prompt, tool traffic, and
/// the bracketed notes the harness writes ahead of a prompt are set aside.
fn conversation(messages: &[serde_json::Value]) -> Vec<String> {
    messages
        .iter()
        .filter(|m| m["role"] == "user" || m["role"] == "assistant")
        .map(|m| m["content"].as_str().unwrap_or("").to_string())
        .filter(|text| !text.starts_with('['))
        .collect()
}

fn transcript_on_disk(home: &Path, id: &str) -> Vec<serde_json::Value> {
    let path = home.join(".openmax").join("sessions").join(format!("{id}.messages.json"));
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// `rewind` branches a session at the end of a turn and `fork` at its end,
/// and each new session's next request carries exactly the history it was
/// branched with. The session left behind keeps every byte of its
/// transcript, so the two agree up to the fork point and differ only after
/// it. The rewind runs in a later sitting (`--continue`), so the original
/// was saved by one process and branched by another.
#[test]
fn rewind_and_fork_branch_a_session_and_the_next_request_carries_only_the_branch() {
    let (project, home) = fresh_dirs("fork-rewind");
    let record = project.parent().unwrap().join("requests.jsonl");
    let replies = ["r1", "r2", "r3", "r4", "r5"].map(sse_text).to_vec();
    let (base_url, _server) = spawn_recording_server(replies, record.clone());
    write_settings_with_mode(&home, &base_url, "auto");

    let mut lines = stdio_steps(&project, &home, &[], &[
        r#"{"cmd":"user","text":"one"}"#,
        r#"{"cmd":"user","text":"two"}"#,
        r#"{"cmd":"user","text":"three"}"#,
    ]);
    let parent = lines[0]["session_id"].as_str().unwrap().to_string();
    let transcript_path = home.join(".openmax").join("sessions").join(format!("{parent}.messages.json"));
    let saved = std::fs::read(&transcript_path).unwrap();
    lines.extend(stdio_steps(&project, &home, &["--continue"], &[
        r#"{"cmd":"rewind","turn":1}"#,
        r#"{"cmd":"user","text":"four"}"#,
        r#"{"cmd":"fork"}"#,
        r#"{"cmd":"user","text":"five"}"#,
    ]));
    assert_eq!(std::fs::read(&transcript_path).unwrap(), saved, "the original keeps every byte");
    let forks: Vec<&serde_json::Value> = lines.iter().filter(|l| l["type"] == "forked").collect();
    assert_eq!(forks.len(), 2, "{lines:#?}");
    let rewound = forks[0]["session_id"].as_str().unwrap().to_string();
    let forked = forks[1]["session_id"].as_str().unwrap().to_string();
    assert_eq!(forks[0]["parent_id"], parent.as_str());
    assert_eq!(forks[1]["parent_id"], rewound.as_str());
    let done_ids: Vec<&str> = lines
        .iter()
        .filter(|l| l["type"] == "done")
        .map(|l| l["session_id"].as_str().unwrap())
        .collect();
    assert_eq!(done_ids, [&parent, &parent, &parent, &rewound, &forked].map(String::as_str));

    let requests: Vec<Vec<String>> = std::fs::read_to_string(&record)
        .unwrap()
        .lines()
        .map(|body| conversation(serde_json::from_str::<serde_json::Value>(body).unwrap()["messages"].as_array().unwrap()))
        .collect();
    assert_eq!(requests.len(), 5, "{requests:#?}");
    assert_eq!(requests[3], ["one", "r1", "four"], "a rewind to turn 1 sends only turn 1");
    assert_eq!(requests[4], ["one", "r1", "four", "r4", "five"], "a fork sends all it was branched with");

    let (p, r, f) = (transcript_on_disk(&home, &parent), transcript_on_disk(&home, &rewound), transcript_on_disk(&home, &forked));
    assert_eq!(conversation(&p), ["one", "r1", "two", "r2", "three", "r3"], "the parent is untouched");
    assert_eq!(conversation(&r), ["one", "r1", "four", "r4"], "a fork is untouched by its own fork");
    assert_eq!(conversation(&f), ["one", "r1", "four", "r4", "five", "r5"]);
    let at = forks[0]["message_index"].as_u64().unwrap() as usize;
    assert_eq!(p[..at], r[..at], "the branches agree up to the fork point");
    assert_ne!(p[at], r[at], "and differ right after it");
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(home.join(".openmax").join("sessions").join("index.json")).unwrap(),
    ).unwrap();
    let entry = |id: &str| index.as_array().unwrap().iter().find(|m| m["id"] == id).unwrap().clone();
    assert_eq!(entry(&rewound)["forked_from"], serde_json::json!({"parent_id": parent, "message_index": at}));
    assert!(entry(&parent).get("forked_from").is_none(), "a session that is no fork writes no field");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A completion gate's refusal reaches the model as a user message right
/// after the reply it refused, and belongs to that turn: `rewind` counts
/// the turns that sent a `done`, so turn 1 ends after the reply the gate let
/// through, and a session of two prompts has two turns whatever the gate
/// wrote between them.
#[test]
fn a_rewind_counts_a_turn_a_completion_gate_refused_as_one_turn() {
    use std::os::unix::fs::PermissionsExt;
    let (project, home) = fresh_dirs("fork-gate");
    let record = project.parent().unwrap().join("requests.jsonl");
    let replies = ["r1", "r1 checked", "r2", "r3"].map(sse_text).to_vec();
    let (base_url, _server) = spawn_recording_server(replies, record.clone());
    write_settings_with_mode(&home, &base_url, "auto");
    // Refuses the first end attempt and lets every later one through.
    let gate = project.join("gate.sh");
    let passed = project.parent().unwrap().join("passed");
    std::fs::write(
        &gate,
        format!(
            "#!/bin/sh\nif [ -f '{0}' ]; then exit 0; fi\n: > '{0}'\necho 'the build does not compile yet'\nexit 1\n",
            passed.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&gate, std::fs::Permissions::from_mode(0o755)).unwrap();
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(
        hooks.join("gate.toml"),
        format!("event = \"turn_end\"\nblocking = true\ncommand = \"{}\"\n", gate.display()),
    )
    .unwrap();

    let lines = stdio_steps(&project, &home, &[], &[
        r#"{"cmd":"user","text":"one"}"#,
        r#"{"cmd":"user","text":"two"}"#,
        r#"{"cmd":"rewind","turn":3}"#,
        r#"{"cmd":"rewind","turn":1}"#,
        r#"{"cmd":"user","text":"three"}"#,
    ]);
    assert!(lines.iter().any(|l| l["type"] == "turn_refused"), "the gate never refused: {lines:#?}");
    let refusal = lines.iter().find(|l| l["type"] == "protocol_error").expect("rewind 3 was not refused");
    assert_eq!(refusal["message"], "no turn 3 to rewind to; the session has 2", "{lines:#?}");
    assert!(lines.iter().any(|l| l["type"] == "forked"), "{lines:#?}");

    let requests: Vec<Vec<String>> = std::fs::read_to_string(&record)
        .unwrap()
        .lines()
        .map(|body| conversation(serde_json::from_str::<serde_json::Value>(body).unwrap()["messages"].as_array().unwrap()))
        .collect();
    assert_eq!(requests.len(), 4, "{requests:#?}");
    assert_eq!(
        requests[3],
        ["one", "r1", "the build does not compile yet", "r1 checked", "three"],
        "turn 1 ends where its done was sent, past the refusal"
    );
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// The original goes on after a fork just as the fork does: each next
/// request carries its own session's history alone, and the two transcripts
/// agree up to the fork point and differ only after it. Before the original
/// continues, its index entry is rewritten to the exact shape a binary that
/// predates forks wrote (no `forked_from`), dated after the fork so
/// `--continue` picks it, and it still loads with its history.
#[test]
fn a_fork_and_its_original_both_continue_and_diverge_only_after_the_fork_point() {
    let (project, home) = fresh_dirs("fork-both");
    let record = project.parent().unwrap().join("requests.jsonl");
    let replies = ["r1", "r2", "r3"].map(sse_text).to_vec();
    let (base_url, _server) = spawn_recording_server(replies, record.clone());
    write_settings_with_mode(&home, &base_url, "auto");

    let first = stdio_steps(&project, &home, &[], &[
        r#"{"cmd":"user","text":"one"}"#,
        r#"{"cmd":"fork"}"#,
        r#"{"cmd":"user","text":"two"}"#,
    ]);
    let original = first[0]["session_id"].as_str().unwrap().to_string();
    let forked = first.iter().find(|l| l["type"] == "forked").expect("no forked line");
    let (fork, at) = (forked["session_id"].as_str().unwrap().to_string(), forked["message_index"].as_u64().unwrap() as usize);

    let index_path = home.join(".openmax").join("sessions").join("index.json");
    let mut index: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let entry = index.iter_mut().find(|m| m["id"] == original.as_str()).unwrap();
    *entry = serde_json::json!({
        "id": entry["id"], "project": entry["project"], "title": entry["title"],
        "created_at": entry["created_at"], "updated_at": 4_102_444_800u64,
    });
    std::fs::write(&index_path, serde_json::to_string(&index).unwrap()).unwrap();

    let second = stdio_steps(&project, &home, &["--continue"], &[r#"{"cmd":"user","text":"three"}"#]);
    assert_eq!(second[0]["session_id"], original.as_str(), "--continue loads an entry with no forked_from");
    assert_eq!(second[0]["continued"], true);

    let requests: Vec<Vec<String>> = std::fs::read_to_string(&record)
        .unwrap()
        .lines()
        .map(|body| conversation(serde_json::from_str::<serde_json::Value>(body).unwrap()["messages"].as_array().unwrap()))
        .collect();
    assert_eq!(requests.len(), 3, "{requests:#?}");
    assert_eq!(requests[1], ["one", "r1", "two"], "the fork continues from what it copied");
    assert_eq!(requests[2], ["one", "r1", "three"], "the original continues from its own history alone");

    let (o, f) = (transcript_on_disk(&home, &original), transcript_on_disk(&home, &fork));
    assert_eq!(conversation(&o), ["one", "r1", "three", "r3"]);
    assert_eq!(conversation(&f), ["one", "r1", "two", "r2"]);
    assert_eq!(o[..at], f[..at], "the two agree up to the fork point");
    assert_eq!(conversation(&o[..at]), ["one", "r1"], "and the fork point is the end of turn one");
    // The older entry's first turn under this binary records the ends its
    // messages imply, so its two turns stay two and the second ends last.
    let index: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let ends = index.iter().find(|m| m["id"] == original.as_str()).unwrap()["turn_ends"].clone();
    assert_eq!(ends, serde_json::json!([at, o.len()]), "{index:#?}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Run one print-mode turn as JSON and hand back the exit code, the parsed
/// event lines, and the raw stdout for assertion messages.
fn json_turn(
    project: &Path,
    home: &Path,
    prompt: &str,
) -> (Option<i32>, Vec<serde_json::Value>, String) {
    let out = cmd(project, home)
        .args(["--trust-project", "--json", "-p", prompt])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let lines = stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("every stdout line is JSON"))
        .collect();
    (out.status.code(), lines, stdout)
}

/// Everything this HOME persisted about its sessions, as one blob.
fn session_dump(home: &Path) -> String {
    std::fs::read_dir(home.join(".openmax").join("sessions"))
        .expect("sessions dir")
        .filter_map(|e| e.ok())
        .map(|e| std::fs::read_to_string(e.path()).unwrap_or_default())
        .collect()
}

/// However a truncated turn ends up truncated, it ends the same way: nonzero
/// exit, one `error` naming the incomplete reply, and exactly one `done` (the
/// terminator guarantee) carrying stop_reason `truncated` as the last line.
fn assert_truncated_turn(code: Option<i32>, lines: &[serde_json::Value], stdout: &str) {
    assert_eq!(code, Some(1), "a cut-off answer must not exit 0: {stdout}");
    assert_eq!(
        lines.iter().filter(|l| l["type"] == "done").count(),
        1,
        "exactly one done per turn: {stdout}"
    );
    let last = lines.last().expect("at least one line");
    assert_eq!(last["type"], "done", "{stdout}");
    assert_eq!(last["stop_reason"], "truncated", "{stdout}");
    assert!(
        lines.iter().any(|l| l["type"] == "error"
            && l["message"].as_str().is_some_and(|m| m.contains("incomplete"))),
        "the truncation must be reported as an error: {stdout}"
    );
}

/// A stream the provider abandons must never read as a finished answer: the
/// partial text is kept (so the session resumes), but the turn reports an
/// error, ends with stop_reason `truncated`, and print mode exits nonzero.
#[test]
fn a_truncated_stream_reports_truncation_instead_of_a_clean_stop() {
    let (project, home) = fresh_dirs("truncated");
    let partial =
        "data: {\"choices\":[{\"delta\":{\"content\":\"half an ans\"},\"finish_reason\":null}]}\n\n";
    let (base_url, _requests, _server) = spawn_scripted_server(vec![(partial.to_string(), false)]);
    write_settings(&home, &base_url);

    let (code, lines, stdout) = json_turn(&project, &home, "say hi");
    assert_truncated_turn(code, &lines, &stdout);
    assert!(
        lines.iter().any(|l| l["type"] == "message_done" && l["text"] == "half an ans"),
        "the partial answer must still be delivered: {stdout}"
    );

    // ...and it must survive on disk, or a resume would lose the partial turn.
    let saved = session_dump(&home);
    assert!(saved.contains("half an ans"), "partial reply must be persisted: {saved}");
}

/// The dangerous half of the same bug: the stream dies *after* a complete tool
/// call, so the arguments parse and nothing looks broken. A stream with no
/// completion signal is not a response the model asked to act on (more calls
/// may have been coming, or this one may still have been under revision), so
/// the call must not run. Reply text streams first, so this is the
/// interruption the client does not start over.
#[test]
fn a_truncated_stream_never_runs_the_tool_call_it_carried() {
    let (project, home) = fresh_dirs("truncated-native-call");
    let prose = "data: {\"choices\":[{\"delta\":{\"content\":\"writing it\"},\"finish_reason\":null}]}\n\n";
    let (base_url, _requests, _server) =
        spawn_scripted_server(vec![(format!("{prose}{WRITE_CALL_SSE}"), false)]);
    // auto, so a refusal here is the truncation and not the approval gate.
    write_settings_with_mode(&home, &base_url, "auto");

    let (code, lines, stdout) = json_turn(&project, &home, "write the file");
    assert_truncated_turn(code, &lines, &stdout);
    // The property that matters: no side effect.
    assert!(
        !project.join("side-effect.txt").exists(),
        "a call from an unterminated stream must not run: {stdout}"
    );
    assert!(
        !lines.iter().any(|l| l["type"] == "tool_start"),
        "no tool may even be dispatched: {stdout}"
    );
    assert!(
        lines.iter().any(|l| l["type"] == "error"
            && l["message"].as_str().is_some_and(|m| m.contains("did not run"))),
        "the error must say the call was refused: {stdout}"
    );
    // The refused call was persisted before it was refused, so it needs a
    // tool reply: an unanswered tool_call id breaks chat-template replay
    // and would make the session unresumable.
    let saved = session_dump(&home);
    assert!(
        saved.contains("\"role\":\"tool\"")
            && saved.contains("The provider stream ended before this call could run"),
        "the refused call id must be answered on disk: {saved}"
    );
}

/// A provider that fails after its 200 has gone out says so inside the
/// stream. That used to read as a clean end: the turn stopped with nothing
/// reported, and a tool call that streamed before the failure ran. The turn
/// fails with the provider's own message and runs nothing.
#[test]
fn a_provider_error_inside_the_stream_is_reported_and_runs_no_tool_call() {
    let (project, home) = fresh_dirs("stream-error");
    let failure = concat!(
        "data: {\"error\":{\"code\":400,\"message\":\"upstream rejected the request\"},",
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"error\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (format!("{WRITE_CALL_SSE}{failure}"), true),
        (HELLO_SSE.to_string(), true),
    ]);
    // auto, so a call that is not run was refused for this reason and not by
    // the approval gate.
    write_settings_with_mode(&home, &base_url, "auto");

    let (code, lines, stdout) = json_turn(&project, &home, "write the file");
    assert_eq!(code, Some(1), "a failed reply must not exit 0: {stdout}");
    assert!(
        !project.join("side-effect.txt").exists(),
        "a call from a stream the provider failed must not run: {stdout}"
    );
    assert!(!lines.iter().any(|l| l["type"] == "tool_start"), "{stdout}");
    assert!(
        lines.iter().any(|l| l["type"] == "error"
            && l["message"].as_str().is_some_and(|m| m.contains("upstream rejected the request"))),
        "the provider's message must be reported: {stdout}"
    );
    let last = lines.last().expect("at least one line");
    assert_eq!(last["type"], "done", "{stdout}");
    assert_eq!(last["stop_reason"], "error", "{stdout}");
}

/// Control for the refusals above: in the same unattended mode, that exact
/// call does run once the stream finishes. Without this, the refusal test
/// could pass for the wrong reason, such as an approval gate.
#[test]
fn a_finished_stream_still_runs_the_same_write_call() {
    let (project, home) = fresh_dirs("finished-native-call");
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (format!("{WRITE_CALL_SSE}{TOOL_CALLS_TERMINATOR}"), true),
        (HELLO_SSE.to_string(), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");

    let (code, lines, stdout) = json_turn(&project, &home, "write the file");
    assert_eq!(code, Some(0), "{stdout}");
    assert!(
        lines.iter().any(|l| l["type"] == "tool_start" && l["name"] == "write_file"),
        "the finished call must be dispatched: {stdout}"
    );
    assert_eq!(
        std::fs::read_to_string(project.join("side-effect.txt")).unwrap_or_default(),
        "written",
        "the finished call must run"
    );
}

/// A write that fails partway, as on a full disk, over a quota or past a file
/// size limit, must leave the file as it was. Writing in place truncated the
/// file first, so the failure left a prefix of the new contents under the old
/// name while the tool said only that it could not write. A file with other
/// hard links is still written in place, so its old contents are written
/// back. A file that did not exist is not created, and the result says which.
#[cfg(unix)]
#[test]
fn a_write_that_fails_partway_leaves_the_file_as_it_was() {
    use std::os::unix::process::CommandExt;
    let (project, home) = fresh_dirs("failed-write");
    let original: String = (0..1000).map(|i| format!("record {i:06} value=ORIGINAL\n")).collect();
    let longer = "ORIGINAL_BUT_MUCH_LONGER_VALUE";
    let grown = original.replace("ORIGINAL", longer);
    std::fs::write(project.join("data.txt"), &original).unwrap();
    std::fs::write(project.join("linked.txt"), &original).unwrap();
    std::fs::hard_link(project.join("linked.txt"), project.join("alias.txt")).unwrap();
    let grow = |path: &str| {
        serde_json::json!({"path": path, "old_string": "ORIGINAL", "new_string": longer, "replace_all": true})
    };
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (
            sse_tool_calls(&[
                ("edit_file", grow("data.txt")),
                ("edit_file", grow("linked.txt")),
                ("write_file", serde_json::json!({"path": "fresh.txt", "content": grown})),
            ]),
            true,
        ),
        (HELLO_SSE.to_string(), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");

    // Halfway between the two sizes: the old contents fit under the limit
    // and the new ones cannot.
    let limit = ((original.len() + grown.len()) / 2) as libc::rlim_t;
    let mut command = cmd(&project, &home);
    command.args(["--trust-project", "--json", "-p", "grow the values"]);
    // SAFETY: the closure makes two system calls, allocating nothing and
    // taking no lock, and they change only the child. With SIGXFSZ ignored,
    // a write past the limit fails with EFBIG instead of killing the process.
    unsafe {
        command.pre_exec(move || {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
            let cap = libc::rlimit { rlim_cur: limit, rlim_max: limit };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &cap) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
    // The limit is the process's, so it also caps the session transcript
    // under HOME once the turn appends the assistant message that carries
    // the new contents. That save fails too; the turn reports it as a
    // warning, runs the calls anyway, and exits 1 for it, so the exit code
    // says nothing about the writes. The tool results and the files do. The
    // output is read while the child runs: the tool_start lines carry the
    // whole new contents, more than a pipe holds unread.
    let out = command.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let ends: Vec<serde_json::Value> = stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|l| l["type"] == "tool_end")
        .collect();
    assert_eq!(ends.len(), 3, "every write must run: {stdout}");
    for name in ["data.txt", "linked.txt", "alias.txt"] {
        let after = std::fs::read_to_string(project.join(name)).unwrap();
        assert!(after == original, "a failed write must leave {name} as it was, not {} bytes", after.len());
    }
    assert!(!project.join("fresh.txt").exists(), "a failed write must not leave a partial new file");
    let said: Vec<&str> = ends.iter().map(|end| end["output"].as_str().unwrap_or_default()).collect();
    assert!(ends.iter().all(|end| end["ok"] == false), "{stdout}");
    assert!(said[0].starts_with("cannot write data.txt") && said[0].ends_with("data.txt is unchanged"), "{said:?}");
    assert!(said[1].starts_with("cannot write linked.txt") && said[1].ends_with("old contents were written back"), "{said:?}");
    assert!(said[2].starts_with("cannot write fresh.txt") && said[2].ends_with("fresh.txt was not created"), "{said:?}");
    let names: Vec<_> = std::fs::read_dir(&project).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(names.iter().all(|n| !n.to_string_lossy().ends_with(".tmp")), "nothing left behind: {names:?}");
}

/// Replacing a file with a new one must not change what makes it the same
/// file to everything else: its mode, its group, the other names a hard link
/// gives it, the symlink that points at it, and whether it can be written at
/// all. A file in a directory that does not let a new file be created next
/// to it is still writable in place, and a name near the length limit still
/// leaves room for the temp file beside it. A hard-linked file this user may
/// write but not read is left unchanged, and the result says so: written in
/// place for its second name, it could not have its old contents put back if
/// that write failed. That refusal is new; the rest held when every write
/// was made in place. A new file gets 0666 less the umask, as one created in
/// place did, and a replaced file loses setuid, as a write by anyone but the
/// superuser clears it.
#[cfg(unix)]
#[test]
fn a_write_keeps_the_files_mode_links_and_permissions() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    use std::os::unix::process::CommandExt;
    let (project, home) = fresh_dirs("write-identity");
    // 240 of the 255 bytes a name may have.
    let long = format!("{}.txt", "n".repeat(236));
    std::fs::write(project.join(&long), "old\n").unwrap();
    let long_new = format!("{}.txt", "m".repeat(236));
    // A group this user is in that a new file here would not get.
    let grouped = project.join("grouped.txt");
    std::fs::write(&grouped, "old\n").unwrap();
    let mut groups = [0 as libc::gid_t; 64];
    // SAFETY: getgroups writes at most 64 entries into the buffer it is given.
    let count = unsafe { libc::getgroups(64, groups.as_mut_ptr()) };
    let own = std::fs::metadata(&grouped).unwrap().gid();
    let other = groups[..count.max(0) as usize].iter().copied().find(|&gid| gid != own);
    if let Some(gid) = other {
        std::os::unix::fs::chown(&grouped, None, Some(gid)).unwrap();
    }
    let script = project.join("run.sh");
    std::fs::write(&script, "#!/bin/sh\necho old\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o750)).unwrap();
    // This user owns it, so it may carry setuid.
    let setuid = project.join("setuid.sh");
    std::fs::write(&setuid, "#!/bin/sh\necho old\n").unwrap();
    std::fs::set_permissions(&setuid, std::fs::Permissions::from_mode(0o4755)).unwrap();
    assert_eq!(std::fs::metadata(&setuid).unwrap().mode() & 0o7777, 0o4755);
    std::fs::write(project.join("shared.txt"), "old\n").unwrap();
    std::fs::hard_link(project.join("shared.txt"), project.join("alias.txt")).unwrap();
    std::fs::write(project.join("real.txt"), "old\n").unwrap();
    std::os::unix::fs::symlink("real.txt", project.join("link.txt")).unwrap();
    let readonly = project.join("readonly.txt");
    std::fs::write(&readonly, "old\n").unwrap();
    std::fs::set_permissions(&readonly, std::fs::Permissions::from_mode(0o444)).unwrap();
    // The superuser may write it anyway, in place or not.
    let refused = std::fs::OpenOptions::new().write(true).open(&readonly).is_err();
    // Written in place for its second name, with nothing to put back if
    // that write failed, since it cannot be read: left alone instead.
    let writeonly = project.join("writeonly.txt");
    std::fs::write(&writeonly, "old\n").unwrap();
    std::fs::hard_link(&writeonly, project.join("writeonly-alias.txt")).unwrap();
    std::fs::set_permissions(&writeonly, std::fs::Permissions::from_mode(0o200)).unwrap();
    let unreadable = std::fs::File::open(&writeonly).is_err();
    let locked = project.join("locked");
    std::fs::create_dir(&locked).unwrap();
    std::fs::write(locked.join("inside.txt"), "old\n").unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();

    let edit = |path: &str| serde_json::json!({"path": path, "old_string": "old", "new_string": "new"});
    let calls = [
        ("write_file", serde_json::json!({"path": "run.sh", "content": "#!/bin/sh\necho new\n"})),
        ("edit_file", edit("shared.txt")),
        ("edit_file", edit("link.txt")),
        ("edit_file", edit("readonly.txt")),
        ("edit_file", edit("locked/inside.txt")),
        ("edit_file", edit(&long)),
        ("write_file", serde_json::json!({"path": long_new, "content": "new\n"})),
        ("edit_file", edit("grouped.txt")),
        ("edit_file", edit("setuid.sh")),
        ("write_file", serde_json::json!({"path": "writeonly.txt", "content": "new\n"})),
    ];
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (sse_tool_calls(&calls), true),
        (HELLO_SSE.to_string(), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");
    // Set for the child alone, so no other test's files are created under
    // it, and unlike the usual 022, so a new file's mode shows it.
    let umask: libc::mode_t = 0o027;
    let mut command = cmd(&project, &home);
    command.args(["--trust-project", "--json", "-p", "update the files"]);
    // SAFETY: the closure makes one system call, allocating nothing and
    // taking no lock, and it changes only the child.
    unsafe {
        command.pre_exec(move || {
            libc::umask(umask);
            Ok(())
        });
    }
    let out = command.output().unwrap();
    let code = out.status.code();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let lines: Vec<serde_json::Value> =
        stdout.lines().map(|l| serde_json::from_str(l).expect("every stdout line is JSON")).collect();
    // Writable again before any assertion, so a failure leaves nothing that
    // cannot be deleted.
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::set_permissions(&writeonly, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(code, Some(0), "{stdout}");
    assert_eq!(lines.iter().filter(|l| l["type"] == "tool_end").count(), calls.len(), "{stdout}");

    assert_eq!(std::fs::read_to_string(&script).unwrap(), "#!/bin/sh\necho new\n");
    assert_eq!(std::fs::metadata(&script).unwrap().mode() & 0o7777, 0o750, "the mode must survive");
    assert_eq!(std::fs::read_to_string(project.join("alias.txt")).unwrap(), "new\n", "both names must see the edit");
    assert_eq!(std::fs::metadata(project.join("shared.txt")).unwrap().nlink(), 2, "the link must survive");
    assert_eq!(std::fs::read_to_string(project.join("real.txt")).unwrap(), "new\n");
    assert!(project.join("link.txt").is_symlink(), "the symlink must stay a symlink");
    assert_eq!(std::fs::read_to_string(locked.join("inside.txt")).unwrap(), "new\n", "{stdout}");
    assert_eq!(std::fs::read_to_string(project.join(&long)).unwrap(), "new\n", "{stdout}");
    assert_eq!(std::fs::read_to_string(project.join(&long_new)).unwrap(), "new\n", "{stdout}");
    let created = std::fs::metadata(project.join(&long_new)).unwrap().mode() & 0o7777;
    assert_eq!(created, 0o640, "a new file gets 0666 less the umask of 027");
    assert_eq!(std::fs::read_to_string(&setuid).unwrap(), "#!/bin/sh\necho new\n");
    assert_eq!(std::fs::metadata(&setuid).unwrap().mode() & 0o7777, 0o755, "a replacement drops setuid");
    assert_eq!(std::fs::read_to_string(&grouped).unwrap(), "new\n");
    // The superuser may read and write any file, and a user in one group
    // has no second one to keep, so those checks cannot run for them. Said
    // aloud, so such a run does not pass for one that made every check.
    if let Some(gid) = other {
        assert_eq!(std::fs::metadata(&grouped).unwrap().gid(), gid, "the group must survive");
    } else {
        eprintln!("skipped: this user is in no second group, so grouped.txt's group is not checked");
    }
    if refused {
        assert_eq!(std::fs::read_to_string(&readonly).unwrap(), "old\n", "a file it may not write is not replaced");
    } else {
        eprintln!("skipped: this user may write the 0444 readonly.txt, so its refusal is not checked");
    }
    if unreadable {
        assert_eq!(std::fs::read_to_string(&writeonly).unwrap(), "old\n", "{stdout}");
        let said = lines.iter().rfind(|l| l["type"] == "tool_end").unwrap()["output"].clone();
        assert!(said.as_str().is_some_and(|s| s.ends_with("writeonly.txt is unchanged")), "{said}");
    } else {
        eprintln!("skipped: this user may read the 0200 writeonly.txt, so the refusal this test was added for is not checked; run it as a user who is not the superuser");
    }
    let names: Vec<_> = std::fs::read_dir(&project).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(names.iter().all(|n| !n.to_string_lossy().ends_with(".tmp")), "nothing left behind: {names:?}");
}

#[test]
fn assistant_text_never_dispatches_a_tool() {
    let call = r#"<tool_call>{"name":"write_file","arguments":{"path":"side-effect.txt","content":"written"}}</tool_call>"#;
    for text in [call.to_string(), format!("Document `{call}`; do not run it."),
        "```tool_call\n{\"name\":\"write_file\",\"arguments\":{\"path\":\"side-effect.txt\",\"content\":\"written\"}}\n```".into()] {
        let (project, home) = fresh_dirs("text-is-not-a-call");
        let delta = serde_json::json!({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]});
        let (base_url, _requests, _server) = spawn_scripted_server(vec![
            (format!("data: {delta}\n\ndata: [DONE]\n\n"), true),
            (HELLO_SSE.to_string(), true),
        ]);
        write_settings_with_mode(&home, &base_url, "auto");
        let (code, lines, stdout) = json_turn(&project, &home, "explain the call syntax");
        assert_eq!(code, Some(0), "{stdout}");
        assert!(!project.join("side-effect.txt").exists(), "prose executed: {stdout}");
        assert!(!lines.iter().any(|l| l["type"] == "tool_start"), "{stdout}");
        assert!(lines.iter().any(|l| l["type"] == "message_done" && l["text"] == text), "{stdout}");
        assert!(!session_dump(&home).contains("\"role\":\"tool\""));
    }
}

/// Opening a named pipe waits for a writer, and reading a pipe takes
/// whatever arrives for as long as it keeps coming, so a file tool sent to
/// one could wait indefinitely: Esc could not end the turn, because a
/// started mutation is waited for, and /quit could not end the process,
/// which waits on the blocked thread at exit. Each file tool names what the
/// path is and leaves it unopened: a writer waiting on the pipe would wake
/// to an open and lose its reader at the close.
#[cfg(unix)]
#[test]
fn file_tools_name_a_pipe_or_socket_and_leave_it_unopened() {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
    // A socket's path is capped at about a hundred bytes, so the directory
    // name is kept short.
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("om-{:x}-{nonce:x}", std::process::id()));
    // Take the pipe and socket away even when an assertion fails.
    struct RemoveOnDrop(PathBuf);
    impl Drop for RemoveOnDrop {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = RemoveOnDrop(base.clone());
    let (project, home) = (base.join("project"), base.join("home"));
    std::fs::create_dir_all(&project).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    let pipe = std::ffi::CString::new(project.join("pipe").as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(pipe.as_ptr(), 0o644) }, 0);
    let _socket = std::os::unix::net::UnixListener::bind(project.join("sock")).unwrap();
    // Opening a pipe to write waits for a reader to open it, so a tool that
    // opened this one would let the writer through.
    let (writer_tx, writer_rx) = std::sync::mpsc::channel();
    let writer = {
        let pipe = project.join("pipe");
        std::thread::spawn(move || {
            let _ = writer_tx.send("waiting");
            let file = std::fs::OpenOptions::new().write(true).open(pipe);
            let _ = writer_tx.send(if file.is_ok() { "opened" } else { "failed" });
        })
    };
    assert_eq!(writer_rx.recv(), Ok("waiting"));
    let script = vec![
        (sse_tool_call("read_file", serde_json::json!({"path": "pipe"})), true),
        (sse_tool_call("edit_file", serde_json::json!({"path": "pipe", "old_string": "a", "new_string": "b"})), true),
        (sse_tool_call("write_file", serde_json::json!({"path": "pipe", "content": "x"})), true),
        (sse_tool_call("read_file", serde_json::json!({"path": "sock"})), true),
        (sse_text("done"), true),
    ];
    let (base_url, _requests, _server) = spawn_scripted_server(script);
    write_settings_with_mode(&home, &base_url, "auto");

    let child = cmd(&project, &home)
        .args(["--trust-project", "--json", "-p", "use the pipe"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let out = finish_with_deadline(child);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    let ends: Vec<(bool, String)> = stdout
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("every stdout line is JSON"))
        .filter(|l| l["type"] == "tool_end")
        .map(|l| (l["ok"] == true, l["output"].as_str().unwrap_or_default().to_string()))
        .collect();
    let expected = [
        "pipe is a named pipe, not a regular file",
        "pipe is a named pipe, not a regular file",
        "pipe is a named pipe, not a regular file",
        "sock is a socket, not a regular file",
    ];
    assert_eq!(ends.len(), expected.len(), "every call must end: {stdout}");
    for ((ok, output), expected) in ends.iter().zip(expected) {
        assert!(!ok && output == expected, "{output:?} should be {expected:?}: {stdout}");
    }
    assert!(
        std::fs::symlink_metadata(project.join("pipe")).unwrap().file_type().is_fifo(),
        "write_file must not replace the pipe"
    );
    // An open by any tool let the writer through well before the run ended.
    let early = writer_rx.recv_timeout(std::time::Duration::from_millis(200));
    assert_eq!(early, Err(std::sync::mpsc::RecvTimeoutError::Timeout), "no file tool may open the pipe");
    let _reader = std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(project.join("pipe"));
    assert_eq!(writer_rx.recv_timeout(std::time::Duration::from_secs(5)), Ok("opened"), "the writer was waiting");
    writer.join().unwrap();
}

/// A settings file this process will never act on must not be able to hide
/// the project's own history. `--recall` reads no settings - it reaches an
/// endpoint never and spends nothing - so it answers, and says plainly that
/// the file is broken. The paths that do spend still refuse.
#[test]
fn recall_reads_history_when_settings_are_unreadable() {
    let (project, home) = fresh_dirs("recall-bad-settings");
    std::fs::create_dir_all(home.join(".openmax")).unwrap();
    // A key from a newer build is the case that prompted this: real, valid
    // JSON that this binary's schema does not know.
    std::fs::write(
        home.join(".openmax").join("settings.json"),
        "{\n  \"model\": \"m\",\n  \"reasoning_effort\": \"high\"\n}\n",
    )
    .unwrap();

    let searched = Command::new(openmax_bin())
        .args(["--recall", "anything at all"])
        .current_dir(&project)
        .env("HOME", &home)
        .output()
        .unwrap();
    assert!(
        searched.status.success(),
        "recall must still answer: {}",
        String::from_utf8_lossy(&searched.stderr)
    );
    let warned = String::from_utf8_lossy(&searched.stderr);
    assert!(
        warned.contains("reasoning_effort") && warned.contains("searching history anyway"),
        "the broken file must be reported, not swallowed: {warned}"
    );

    // Trusted, so the refusal below can only be the settings file: the trust
    // gate runs first and would otherwise mask what this is asserting.
    let refused = Command::new(openmax_bin())
        .args(["--trust-project", "-p", "hello"])
        .current_dir(&project)
        .env("HOME", &home)
        // Trust is a human act; this test stands in for the human.
        .env("OPENMAX_HUMAN_ATTEST", "1")
        .output()
        .unwrap();
    assert!(!refused.status.success(), "a turn must still fail closed on unreadable settings");
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("invalid settings file"),
        "and for that reason: {}",
        String::from_utf8_lossy(&refused.stderr)
    );
}

/// The prompt prefix only ever grows within a turn.
///
/// Prefix caching keys on the token sequence the server renders, so the single
/// most valuable cost property this harness has is that successive requests
/// share a byte-identical leading prompt: cached input is an order of
/// magnitude cheaper than uncached, and cache traffic dominates a coding
/// agent's bill. A regression is silent by construction - it changes no
/// output, only the invoice and the latency - so nothing but an assertion
/// will catch it. A timestamp in the system prompt, a reordered tool schema,
/// or a cwd rendered into the prefix would each fail here.
#[test]
fn the_prompt_prefix_only_grows_within_a_turn() {
    let (project, home) = fresh_dirs("prefix-stable");
    // Three tool round trips then an answer, so one turn spans four requests
    // and every tool result has to append rather than rewrite.
    let script = vec![
        (sse_tool_calls(&[("list_dir", serde_json::json!({"path": "."}))]), true),
        (sse_tool_calls(&[("list_dir", serde_json::json!({"path": "."}))]), true),
        (sse_tool_calls(&[("list_dir", serde_json::json!({"path": "."}))]), true),
        (sse_text("done"), true),
    ];
    let (base_url, requests, _server) = spawn_scripted_server(script);
    write_settings_with_mode(&home, &base_url, "auto");

    let out = cmd(&project, &home).args(["--trust-project", "-p", "look around"]).output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bodies = requests.lock().unwrap().clone();
    assert!(bodies.len() >= 3, "expected several requests in one turn, got {}", bodies.len());
    let parsed: Vec<serde_json::Value> =
        bodies.iter().map(|b| serde_json::from_str(b).expect("request body is JSON")).collect();

    for pair in parsed.windows(2) {
        let (prev, cur) = (&pair[0], &pair[1]);
        // Tools serialize into the prefix ahead of the conversation, so any
        // change to them invalidates everything after.
        assert_eq!(prev["tools"], cur["tools"], "tool schemas must not move within a turn");
        let (pm, cm) = (
            prev["messages"].as_array().expect("messages"),
            cur["messages"].as_array().expect("messages"),
        );
        assert!(
            cm.len() > pm.len(),
            "a request must add to the conversation, not replace it: {} -> {}",
            pm.len(),
            cm.len()
        );
        for (i, old) in pm.iter().enumerate() {
            assert_eq!(
                old, &cm[i],
                "message {i} was rewritten mid-turn; everything after it re-prefills"
            );
        }
    }
}

/// Two sessions in the same project start from a byte-identical prefix, so
/// the second one opens against a warm cache instead of paying to prefill a
/// system prompt and tool schemas the provider already holds. Anything
/// session-scoped rendered into the prompt - an id, a timestamp, a clock -
/// would break this and cost a full prefill on every new session.
#[test]
fn a_new_session_reuses_the_previous_session_prefix() {
    let (project, home) = fresh_dirs("prefix-cross");
    let (base_url, requests, _server) =
        spawn_scripted_server(vec![(sse_text("one").to_string(), true); 2]);
    write_settings(&home, &base_url);

    for prompt in ["first session", "second session"] {
        let out = cmd(&project, &home).args(["--trust-project", "-p", prompt]).output().unwrap();
        assert_eq!(
            out.status.code(),
            Some(0),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let bodies = requests.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2, "one request per session");
    let a: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
    let b: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    assert_eq!(a["tools"], b["tools"], "tool schemas must be identical across sessions");
    assert_eq!(
        a["messages"][0], b["messages"][0],
        "the system prompt must be identical across sessions, or every new session \
         pays a full prefill"
    );
}

/// A session launched under an attested shell must not
/// hand its bash children a bypass. The child unsets the marker; the
/// attestation must already be gone; --approve refuses for lack of a
/// terminal.
#[test]
fn an_attested_parent_does_not_let_a_bash_child_approve() {
    let (project, home) = fresh_dirs("attest-inherit");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    std::fs::write(project.join("gate.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(project.join("gate.sh"), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    std::fs::write(hooks.join("gate.toml"), "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n").unwrap();
    // The scripted turn: one bash call of exactly that shape, from a parent
    // that IS attested (cmd() sets it).
    let bypass = format!(
        "env -u OPENMAX_SESSION {} --approve .openmax/hooks/gate.toml < /dev/null; echo \"exit=$?\"",
        openmax_bin()
    );
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (sse_tool_call("bash", serde_json::json!({ "command": bypass })), true),
        (sse_text("done"), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");
    let out = cmd(&project, &home).args(["--trust-project", "-p", "go"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("exit=3"), "the child's --approve must refuse: {stderr}");
    assert_eq!(approval_records(&home), 0, "no approval may land from an agent's bash child");
}

/// A process the harness itself spawned has no human on its client: an agent
/// that launches `openmax --stdio` from bash receives its own
/// unapproved_source card over the pipe, answers it, and lands a standing
/// grant in the ledger as a human act. The card is never raised in
/// an agent-spawned process: the call is declined with the reason, an eager
/// client's `approve` finds nothing to answer, and the ledger stays empty.
#[test]
fn a_nested_stdio_session_cannot_answer_its_own_content_card() {
    let (project, home) = fresh_dirs("nested-stdio");
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    let marker = project.join("SELF-APPROVED");
    std::fs::write(
        tools.join("selfapprove.toml"),
        format!(
            "name = \"selfapprove\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\n",
            marker.display()
        ),
    )
    .unwrap();
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (sse_tool_call("selfapprove", serde_json::json!({})), true),
        (sse_text("done"), true),
    ]);
    write_settings_with_mode(&home, &base_url, "ask");
    // Trust as the human first, in ask, then run the nested session the way
    // an agent's bash would: session marker set, piped stdio, no tty.
    trust_in_ask(&project, &home);
    let mut child = cmd(&project, &home)
        .env("OPENMAX_SESSION", "1")
        .env_remove("OPENMAX_HUMAN_ATTEST")
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut reader = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert!(line.contains("\"hello\""), "{line}");
    writeln!(stdin, r#"{{"cmd":"user","text":"call selfapprove"}}"#).unwrap();
    let mut saw_card = false;
    let mut decline = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let ev: serde_json::Value = serde_json::from_str(&line).unwrap();
        match ev["type"].as_str() {
            Some("approval_request") => {
                saw_card = true;
                // The eager client answers yes - it must change nothing.
                writeln!(stdin, r#"{{"cmd":"approve","approval_id":"{}","approved":true}}"#, ev["approval_id"].as_str().unwrap()).unwrap();
            }
            Some("tool_end") => decline = ev["output"].as_str().unwrap_or("").to_string(),
            Some("done") => break,
            _ => {}
        }
    }
    writeln!(stdin, r#"{{"cmd":"quit"}}"#).unwrap();
    drop(stdin);
    let _ = child.wait();
    assert!(!saw_card, "no card may be raised in an agent-spawned process");
    assert!(decline.contains("no human is on this client"), "the refusal says why: {decline}");
    assert!(decline.contains("openmax --approve"), "{decline}");
    assert!(!marker.exists(), "the unapproved tool must not have run");
    assert_eq!(approval_records(&home), 0, "the ledger must record no grant");
}

/// A sandboxed probe cannot prove a tool that needs the network or a write
/// outside its scratch dir, so a probe non-pass is inconclusive, not proof the
/// tool is broken: `openmax --check --run-examples` reports it as `warn`, not
/// `err`, and does NOT exit nonzero on it (a passing probe approves nothing, so
/// a failing one condemns nothing). The honest signal is the approved host run.
/// Guards the false-`err` that failed CI and misled agents on the largest tool
/// family (anything network-shaped). Terminal (non-JSON) path.
#[test]
fn an_unapproved_sandbox_non_pass_warns_and_does_not_fail_the_check() {
    let (project, home) = fresh_dirs("probe-warn");
    write_settings_with_mode(&home, "http://127.0.0.1:9/v1", "ask");
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    // Its example always exits nonzero. Unapproved, it runs only as a sandboxed
    // probe; the probe cannot vouch for it, but that is not the tool being
    // broken as far as an UNAPPROVED check is concerned - the verdict waits for
    // the approved host run.
    std::fs::write(
        tools.join("nonpass.toml"),
        "name = \"nonpass\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"echo boom >&2; exit 3\"]\n\n[example]\n",
    )
    .unwrap();
    // Trust in ask so examples run unapproved, as sandboxed probes.
    trust_in_ask(&project, &home);

    let out = cmd(&project, &home).args(["--check", "--run-examples"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "an unapproved sandbox non-pass must not fail --check --run-examples:\n{stdout}"
    );
    assert!(
        !stdout.contains("err  example"),
        "a sandbox non-pass is not an err:\n{stdout}"
    );
    assert!(
        stdout.contains("warn example     nonpass"),
        "the sandbox non-pass is reported as a warn on the tool:\n{stdout}"
    );
}

/// `openmax --spec usage` must never answer "zero extension cost" while the
/// frozen prompt is actually paying for something. A memory note rides the
/// frozen index (prompt.rs), so a project holding only a memory file pays real
/// bytes on every request; the old short-circuit printed a false zero and
/// named nothing. The cost surface now prints the frozen-prefix breakdown and
/// lists the memory (Judges D/E).
#[test]
fn spec_usage_names_memory_cost_and_never_claims_zero() {
    let (project, home) = fresh_dirs("usage-memory");
    let mem = project.join(".openmax").join("memory");
    std::fs::create_dir_all(&mem).unwrap();
    std::fs::write(
        mem.join("staging-port.md"),
        "# The staging deploy port is 7443 (set 2026-07-31)\nSet in infra/nginx.conf.\n",
    )
    .unwrap();

    let out = cmd(&project, &home).args(["--spec", "usage"]).output().unwrap();
    assert!(out.status.success(), "--spec usage exits 0");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !text.contains("zero extension cost"),
        "the cost surface must not claim zero while a memory is in the prompt:\n{text}"
    );
    assert!(
        text.contains("frozen prompt prefix"),
        "the frozen-prefix breakdown is always shown:\n{text}"
    );
    assert!(
        text.contains("staging-port") && text.contains("memory"),
        "the installed memory note is named with its cost:\n{text}"
    );
}

#[test]
fn saved_project_auto_applies_to_headless_and_cannot_be_changed_by_a_child() {
    let (project, home) = fresh_dirs("saved-auto");
    std::fs::create_dir_all(project.join(".openmax/tools")).unwrap();
    std::fs::write(project.join(".openmax/tools/create.toml"),
        "name = \"create\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"echo saved-auto > result\"]\nmutating = true\n"
    ).unwrap();
    let (base_url, _, _server) = spawn_scripted_server(vec![
        (sse_tool_call("create", serde_json::json!({})), true),
        (sse_text("done"), true),
    ]);
    write_settings(&home, &base_url);
    // Trusted in ask, so the auto below is the frontend's selection, not the grant's.
    trust_in_ask(&project, &home);
    let mut select = cmd(&project, &home).arg("--stdio")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    select.stdin.take().unwrap().write_all(b"{\"cmd\":\"approval_mode\",\"mode\":\"auto\"}\n{\"cmd\":\"quit\"}\n").unwrap();
    let selected = select.wait_with_output().unwrap();
    assert!(selected.status.success(), "{}", String::from_utf8_lossy(&selected.stderr));
    let selected = String::from_utf8(selected.stdout).unwrap();
    assert!(selected.contains("\"type\":\"approval_mode\""), "{selected}");

    let mut child = cmd(&project, &home).arg("--stdio").env("OPENMAX_SESSION", "parent")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(b"{\"cmd\":\"approval_mode\",\"mode\":\"ask\"}\n{\"cmd\":\"quit\"}\n").unwrap();
    let denied = child.wait_with_output().unwrap();
    let denied = String::from_utf8(denied.stdout).unwrap();
    assert!(denied.contains("human-controlled frontend"), "{denied}");

    let output = cmd(&project, &home).args(["-p", "run create"]).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(std::fs::read_to_string(project.join("result")).unwrap(), "saved-auto\n");
    let settings: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(home.join(".openmax/settings.json")).unwrap()).unwrap();
    assert_eq!(settings["approval_mode"], "ask", "the global default must not change");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Every frontend on a data dir saves to one trust store. A run that started
/// in auto takes up ask that another frontend saved for the project while it
/// ran: its next mutating call waits for approval, which a headless run
/// declines, instead of running unasked. The bash call holds the turn open
/// until the other frontend has saved, and for about 30 seconds at most.
#[test]
fn ask_saved_by_another_frontend_reaches_a_running_session() {
    /// Ends the held run when the test fails before releasing it: the
    /// release file ends the bash call and the run is killed.
    struct Held(std::path::PathBuf, Option<std::process::Child>);
    impl Drop for Held {
        fn drop(&mut self) {
            let _ = std::fs::write(self.0.join("release"), "");
            if let Some(mut running) = self.1.take() {
                let _ = running.kill();
                let _ = running.wait();
            }
        }
    }
    let (project, home) = fresh_dirs("sibling-ask");
    let hold = "touch ready; i=0; while [ ! -e release ] && [ $i -lt 600 ]; do sleep 0.05; i=$((i+1)); done";
    let (base_url, _, _server) = spawn_scripted_server(vec![
        (sse_tool_call("bash", serde_json::json!({"command": hold})), true),
        (sse_tool_call("write_file", serde_json::json!({"path": "side-effect.txt", "content": "written"})), true),
        (sse_text("done"), true),
    ]);
    write_settings(&home, &base_url);
    let grant = open_max_core::trust::grant_trust(&home.join(".openmax"), &project, ApprovalMode::Auto);
    assert_eq!(grant.unwrap().1, Some(ApprovalMode::Auto));

    let running = cmd(&project, &home).args(["--json", "-p", "go"])
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut held = Held(project.clone(), Some(running));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !project.join("ready").exists() {
        assert!(std::time::Instant::now() < deadline, "the run never reached its bash call");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let mut select = cmd(&project, &home).arg("--stdio")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    select.stdin.take().unwrap().write_all(b"{\"cmd\":\"approval_mode\",\"mode\":\"ask\"}\n{\"cmd\":\"quit\"}\n").unwrap();
    let selected = finish_with_deadline(select);
    let selected = String::from_utf8_lossy(&selected.stdout);
    assert!(selected.contains("\"type\":\"approval_mode\""), "{selected}");
    std::fs::write(project.join("release"), "").unwrap();

    let output = finish_with_deadline(held.1.take().unwrap());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !project.join("side-effect.txt").exists(),
        "the write ran unasked after another frontend saved ask: {stdout}"
    );
    let asked = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .any(|event| event["type"] == "approval_request" && event["name"] == "write_file");
    assert!(asked, "the write must wait for approval: {stdout}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A run whose model answered with one bash call that leaves a tree behind,
/// the shape of a build or a dev server the agent started: the shell (which
/// then becomes a long foreground command) and a background child each
/// record their pid, and both outlive any test unless something stops them.
struct ToolTree {
    child: std::process::Child,
    /// Held open: a stdio client that has not quit, so only a signal ends it.
    _stdin: Option<std::process::ChildStdin>,
    pids: Vec<i32>,
    base: PathBuf,
}

/// Start `args` (a print or a stdio run) and return once the bash call's
/// shell and its background child are both running. `preamble` runs first in
/// that shell.
fn start_tool_tree(tag: &str, args: &[&str], preamble: &str) -> ToolTree {
    let (project, home) = fresh_dirs(tag);
    let pid_file = project.parent().unwrap().join("pids");
    let script = format!(
        "{preamble}echo $$ >> '{p}'; sleep 6841 & echo $! >> '{p}'; exec sleep 7841",
        p = pid_file.display()
    );
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (sse_tool_call("bash", serde_json::json!({ "command": script })), true),
        (sse_text("done"), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");
    let stdio = args.contains(&"--stdio");
    let mut child = cmd(&project, &home)
        .arg("--trust-project")
        .args(args)
        .stdin(if stdio { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take();
    if let Some(stdin) = stdin.as_mut() {
        writeln!(stdin, r#"{{"cmd":"user","text":"start the tree"}}"#).unwrap();
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let pids = loop {
        let recorded = std::fs::read_to_string(&pid_file).unwrap_or_default();
        let pids: Vec<i32> = recorded.lines().filter_map(|l| l.trim().parse().ok()).collect();
        if pids.len() == 2 {
            break pids;
        }
        if std::time::Instant::now() >= deadline || child.try_wait().unwrap().is_some() {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!("the tool tree never started: {}", String::from_utf8_lossy(&output.stderr));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    ToolTree { child, _stdin: stdin, pids, base: project.parent().unwrap().to_path_buf() }
}

/// Send each signal after its delay, wait for the run to end, and return its
/// output and every recorded pid still alive once the tree had time to go.
/// Survivors are killed before returning, so a failing assertion leaks
/// nothing.
fn signal_tool_tree(
    tree: ToolTree,
    signals: &[(std::time::Duration, libc::c_int)],
) -> (std::process::Output, Vec<i32>) {
    let ToolTree { child, _stdin, pids, base } = tree;
    for (delay, signal) in signals {
        std::thread::sleep(*delay);
        unsafe { libc::kill(child.id() as libc::pid_t, *signal) };
    }
    let output = finish_with_deadline(child);
    let alive = |pid: &i32| unsafe { libc::kill(*pid, 0) == 0 };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while pids.iter().any(alive) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let survivors: Vec<i32> = pids.iter().copied().filter(alive).collect();
    for pid in &survivors {
        unsafe { libc::kill(*pid, libc::SIGKILL) };
    }
    let _ = std::fs::remove_dir_all(base);
    (output, survivors)
}

fn last_event(stdout: &[u8]) -> serde_json::Value {
    let stdout = String::from_utf8_lossy(stdout);
    let last = stdout.lines().last().unwrap_or_default();
    serde_json::from_str(last).unwrap_or_else(|_| panic!("the stream must end in an event: {stdout}"))
}

/// SIGTERM is how a supervisor, a CI runner, or a frontend shutting down
/// stops a run. Every tool runs in a session of its own, so the signal
/// reaches openmax alone, and dying on it left the shell and its children
/// changing the project with nobody left to report to. The run stops the
/// tree first, ends its stream with the cancelled turn's done event, and
/// exits 143, the status of a job a SIGTERM stopped.
#[test]
fn sigterm_stops_a_print_run_and_its_tool_tree() {
    let tree = start_tool_tree("sigterm-print", &["--json", "-p", "start the tree"], "");
    let (output, survivors) =
        signal_tool_tree(tree, &[(std::time::Duration::ZERO, libc::SIGTERM)]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(survivors.is_empty(), "tool processes outlived the run: {survivors:?}\n{stderr}");
    assert_eq!(output.status.code(), Some(143), "{:?}\n{stderr}", output.status);
    let done = last_event(&output.stdout);
    assert_eq!((done["type"].as_str(), done["stop_reason"].as_str()), (Some("done"), Some("cancelled")), "{done}");
}

/// Ctrl+C at a shell sends SIGINT to openmax's process group, which no tool
/// is in. A stdio client gets the terminator a cancel gives it, then the 130
/// exit.
#[test]
fn sigint_stops_a_stdio_session_and_its_tool_tree() {
    let tree = start_tool_tree("sigint-stdio", &["--stdio"], "");
    let (output, survivors) =
        signal_tool_tree(tree, &[(std::time::Duration::ZERO, libc::SIGINT)]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(survivors.is_empty(), "tool processes outlived the session: {survivors:?}\n{stderr}");
    assert_eq!(output.status.code(), Some(130), "{:?}\n{stderr}", output.status);
    let done = last_event(&output.stdout);
    assert_eq!((done["type"].as_str(), done["stop_reason"].as_str()), (Some("done"), Some("cancelled")), "{done}");
}

/// A terminal that closes sends SIGHUP. A text run says the turn stopped and
/// exits 129.
#[test]
fn sighup_stops_a_text_print_run_and_its_tool_tree() {
    let tree = start_tool_tree("sighup-print", &["-p", "start the tree"], "");
    let (output, survivors) =
        signal_tool_tree(tree, &[(std::time::Duration::ZERO, libc::SIGHUP)]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(survivors.is_empty(), "tool processes outlived the run: {survivors:?}\n{stderr}");
    assert_eq!(output.status.code(), Some(129), "{:?}\n{stderr}", output.status);
    assert!(stderr.contains("openmax: stopped (cancelled)"), "{stderr}");
}

/// SIGQUIT (Ctrl+\ at a shell) ends a process by default with a core
/// dump and no cleanup, leaving the tool tree running under pid 1. The run
/// stops it as SIGTERM does instead and exits 131, a status and not a
/// death by signal, so no core is written.
#[test]
fn sigquit_stops_a_print_run_and_its_tool_tree() {
    let tree = start_tool_tree("sigquit-print", &["--json", "-p", "start the tree"], "");
    let (output, survivors) =
        signal_tool_tree(tree, &[(std::time::Duration::ZERO, libc::SIGQUIT)]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(survivors.is_empty(), "tool processes outlived the run: {survivors:?}\n{stderr}");
    assert_eq!(output.status.code(), Some(128 + libc::SIGQUIT), "{:?}\n{stderr}", output.status);
    let done = last_event(&output.stdout);
    assert_eq!((done["type"].as_str(), done["stop_reason"].as_str()), (Some("done"), Some("cancelled")), "{done}");
}

/// SIGUSR1 and SIGUSR2 end a process outright by default, so a supervisor
/// that sends one to stop a run left its tool tree running. Each stops the
/// run and exits 128 plus its number, which differs between macOS and Linux.
#[test]
fn sigusr1_stops_a_stdio_session_and_its_tool_tree() {
    let tree = start_tool_tree("sigusr1-stdio", &["--stdio"], "");
    let (output, survivors) =
        signal_tool_tree(tree, &[(std::time::Duration::ZERO, libc::SIGUSR1)]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(survivors.is_empty(), "tool processes outlived the session: {survivors:?}\n{stderr}");
    assert_eq!(output.status.code(), Some(128 + libc::SIGUSR1), "{:?}\n{stderr}", output.status);
    let done = last_event(&output.stdout);
    assert_eq!((done["type"].as_str(), done["stop_reason"].as_str()), (Some("done"), Some("cancelled")), "{done}");
}

#[test]
fn sigusr2_stops_a_text_print_run_and_its_tool_tree() {
    let tree = start_tool_tree("sigusr2-print", &["-p", "start the tree"], "");
    let (output, survivors) =
        signal_tool_tree(tree, &[(std::time::Duration::ZERO, libc::SIGUSR2)]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(survivors.is_empty(), "tool processes outlived the run: {survivors:?}\n{stderr}");
    assert_eq!(output.status.code(), Some(128 + libc::SIGUSR2), "{:?}\n{stderr}", output.status);
    assert!(stderr.contains("openmax: stopped (cancelled)"), "{stderr}");
}

/// A second signal while the first is still stopping the tools exits
/// without waiting for that stop. Exiting runs no destructor, so the tool
/// groups must still be killed on the way out. This tree ignores SIGTERM, so
/// the cancel's grace is still running when the second signal lands, and
/// only that kill stops it.
#[test]
fn a_second_signal_kills_the_tool_tree_on_the_way_out() {
    let tree = start_tool_tree("sigterm-twice", &["--json", "-p", "start the tree"], "trap '' TERM; ");
    let (output, survivors) = signal_tool_tree(
        tree,
        &[
            (std::time::Duration::ZERO, libc::SIGTERM),
            (std::time::Duration::from_millis(100), libc::SIGTERM),
        ],
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(survivors.is_empty(), "tool processes outlived the run: {survivors:?}\n{stderr}");
    assert_eq!(output.status.code(), Some(143), "{:?}\n{stderr}", output.status);
}

/// A run that ends with its tools stopped leaves nothing for the next launch
/// to read. A tool round that finished strikes its group from the record as
/// it ends, which the record of a stdio session that is still running
/// shows: one line adds the group and the next strikes it. The second
/// signal's kill clears each group it killed. Either exit then removes the
/// record and the run directory it leaves empty.
#[test]
fn a_run_that_stops_its_tools_leaves_no_record_of_them() {
    let (project, home) = fresh_dirs("record-finished");
    // A leader that has already exited when its spawn returns still has its
    // start time read, so even a call this short is recorded.
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (sse_tool_call("bash", serde_json::json!({ "command": "true" })), true),
        (sse_text("done"), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");
    let mut session = cmd(&project, &home)
        .args(["--trust-project", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = session.stdin.take().unwrap();
    let mut reader = BufReader::new(session.stdout.take().unwrap());
    writeln!(stdin, r#"{{"cmd":"user","text":"run it"}}"#).unwrap();
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap() > 0 && !line.contains(r#""type":"done""#) {
        line.clear();
    }
    let run = home.join(".openmax").join("run");
    // The record's own thread writes it, which can be after the turn's end
    // reaches stdout.
    let record = run.join(format!("{}.groups", session.id()));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let during = loop {
        let during = std::fs::read_to_string(&record).unwrap_or_default();
        if during.lines().count() >= 3 || std::time::Instant::now() >= deadline {
            break during;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    drop(stdin);
    let finished = finish_with_deadline(session);
    let finished_left = run.exists();
    let _ = std::fs::remove_dir_all(project.parent().unwrap());

    let ToolTree { child, _stdin, pids, base } =
        start_tool_tree("record-killed", &["--json", "-p", "start the tree"], "trap '' TERM; ");
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    std::thread::sleep(std::time::Duration::from_millis(100));
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    let killed = finish_with_deadline(child);
    let killed_left = base.join("home").join(".openmax").join("run").exists();
    for pid in &pids {
        unsafe { libc::kill(*pid, libc::SIGKILL) };
    }
    let _ = std::fs::remove_dir_all(&base);
    let stderr = String::from_utf8_lossy(&finished.stderr);
    assert!(finished.status.success(), "{:?}\n{stderr}", finished.status);
    let lines: Vec<&str> = during.lines().collect();
    assert_eq!(lines.len(), 3, "a running session's record after one tool round: {during:?}\n{stderr}");
    assert_eq!(lines[2], format!("-{}", lines[1]), "a finished tool round left its group listed: {during:?}");
    assert!(!finished_left, "a run whose tool round finished left its record\n{stderr}");
    let stderr = String::from_utf8_lossy(&killed.stderr);
    assert_eq!(killed.status.code(), Some(143), "{:?}\n{stderr}", killed.status);
    assert!(!killed_left, "a run whose second signal killed its tools left their record\n{stderr}");
}

/// SIGKILL, or a crash that runs no handler, gives openmax no chance to stop
/// its tools, and the tool tree runs on under pid 1. The run keeps a record
/// of its live tool groups in `~/.openmax/run`, and the next launch (here
/// `--check`) stops each group a dead run recorded and removes the record.
#[test]
fn the_next_launch_stops_the_tool_tree_a_killed_run_left_behind() {
    launch_after_a_kill("sigkill-reap", None, &["--check"]);
}

/// A print run and a stdio session stop the tree as well, and before they
/// start a session, so before any tool of their own: each here is refused
/// for a settings flag (`--api-key` beside the provider settings.json
/// selects), which ends it before any session, and has stopped the tree all
/// the same.
#[test]
fn a_print_run_stops_the_tool_tree_a_killed_run_left_behind_before_its_session() {
    let provider = Some(r#"{"provider":"p","model":"m"}"#);
    let print = launch_after_a_kill("sigkill-reap-print", provider, &["--api-key", "sk-k", "-p", "go"]);
    let stderr = String::from_utf8_lossy(&print.stderr);
    assert_eq!(print.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--api-key is the key for settings.json's base_url"), "{stderr}");
}

#[test]
fn a_stdio_session_stops_the_tool_tree_a_killed_run_left_behind_before_it_starts() {
    let provider = Some(r#"{"provider":"p","model":"m"}"#);
    let stdio = launch_after_a_kill("sigkill-reap-stdio", provider, &["--api-key", "sk-k", "--stdio"]);
    let stderr = String::from_utf8_lossy(&stdio.stderr);
    assert_eq!(stdio.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--api-key is the key for settings.json's base_url"), "{stderr}");
}

/// Kill a print run outright once its record lists the group of its `bash`
/// call, whose tree ignores SIGTERM, so only the SIGKILL after the grace
/// stops it. Then launch openmax with `next`, on the same home and project
/// and with `settings` as its settings.json if given, before the killed
/// run's parent, this test, has waited for it: until then the killed run
/// holds its pid and its start time, as an exited process. Asserts that the
/// launch stopped the tree and removed the record and the run directory,
/// and returns the launch's output.
fn launch_after_a_kill(tag: &str, settings: Option<&str>, next: &[&str]) -> std::process::Output {
    let ToolTree { mut child, _stdin, pids, base } =
        start_tool_tree(tag, &["--json", "-p", "start the tree"], "trap '' TERM; ");
    let home = base.join("home");
    let record = home.join(".openmax").join("run").join(format!("{}.groups", child.id()));
    // The group is recorded just after its spawn returns, which can be after
    // the tree has written its pids.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::fs::read_to_string(&record).unwrap_or_default().lines().count() < 2
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGKILL) };
    // Returns once the run has exited, and leaves it unwaited for.
    let (mut info, options) = (unsafe { std::mem::zeroed::<libc::siginfo_t>() }, libc::WEXITED | libc::WNOWAIT);
    let exited = match unsafe { libc::waitid(libc::P_PID, child.id() as libc::id_t, &mut info, options) } {
        0 => Ok(()),
        _ => Err(std::io::Error::last_os_error()),
    };
    let alive = |pid: &i32| unsafe { libc::kill(*pid, 0) == 0 };
    let survived_the_kill = pids.iter().all(alive);
    let recorded = record.exists();
    if let Some(settings) = settings {
        std::fs::write(home.join(".openmax").join("settings.json"), settings).unwrap();
    }
    let launch = cmd(&base.join("project"), &home).args(next).output().unwrap();
    let _ = child.wait();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    while pids.iter().any(alive) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let survivors: Vec<i32> = pids.iter().copied().filter(alive).collect();
    for pid in &survivors {
        unsafe { libc::kill(*pid, libc::SIGKILL) };
    }
    let record_kept = record.exists();
    let run_kept = record.parent().unwrap().exists();
    let _ = std::fs::remove_dir_all(&base);
    let stderr = String::from_utf8_lossy(&launch.stderr);
    assert!(exited.is_ok(), "waitid: {exited:?}");
    assert!(survived_the_kill, "the tool tree did not outlive SIGKILL, so this proves nothing: {pids:?}");
    assert!(recorded, "the killed run left no record of its tool groups at {}", record.display());
    assert!(survivors.is_empty(), "the next launch left the tool tree running: {survivors:?}\n{stderr}");
    assert!(!record_kept, "the record outlived the launch that stopped its groups");
    assert!(!run_kept, "the run directory outlived the launch that emptied it");
    launch
}

/// A pid in a dead run's record can since name another process, which the
/// record's start time tells apart: the next launch leaves it running and
/// only removes the stale record.
#[test]
fn the_next_launch_never_stops_a_process_that_took_a_recorded_pid() {
    use std::os::unix::process::CommandExt;
    let (project, home) = fresh_dirs("reap-reused-pid");
    let mut bystander = Command::new("sleep").arg("300").process_group(0).spawn().unwrap();
    let mut gone = Command::new("/bin/sh").args(["-c", "exit 0"]).spawn().unwrap();
    gone.wait().unwrap();
    let run = home.join(".openmax").join("run");
    std::fs::create_dir_all(&run).unwrap();
    let record = run.join(format!("{}.groups", gone.id()));
    std::fs::write(&record, format!("1\n{} 1\n", bystander.id())).unwrap();
    let check = cmd(&project, &home).arg("--check").output().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let alive = bystander.try_wait().unwrap().is_none();
    let _ = bystander.kill();
    let _ = bystander.wait();
    let record_kept = record.exists();
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    let stderr = String::from_utf8_lossy(&check.stderr);
    assert!(alive, "a process the record does not name was stopped\n{stderr}");
    assert!(!record_kept, "the stale record outlived the launch");
}

/// An openmax that a left-behind tool's script starts runs in that tool's
/// group, which the killed run's record lists. A launch never signals the
/// group it runs in, which would end it as it starts: it runs to its end and
/// leaves the tool's group running with its record in place, so the next
/// launch outside the group stops it and removes the record.
#[test]
fn a_launch_from_a_left_behind_tool_does_not_stop_its_own_group() {
    let preamble = format!(
        "(until [ -f ../go ]; do sleep 0.02; done; HOME=\"$(cd ../home && pwd)\" '{bin}' --check \
         > /dev/null 2>&1; echo $? > ../status) & ",
        bin = openmax_bin()
    );
    let ToolTree { mut child, _stdin, pids, base } =
        start_tool_tree("reap-own-group", &["--json", "-p", "start the tree"], &preamble);
    let home = base.join("home");
    let record = home.join(".openmax").join("run").join(format!("{}.groups", child.id()));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::fs::read_to_string(&record).unwrap_or_default().lines().count() < 2
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGKILL) };
    let _ = child.wait();
    let recorded = record.exists();
    std::fs::write(base.join("go"), "").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let status = loop {
        let status = std::fs::read_to_string(base.join("status")).unwrap_or_default();
        if status.ends_with('\n') || std::time::Instant::now() >= deadline {
            break status;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let alive = |pid: &i32| unsafe { libc::kill(*pid, 0) == 0 };
    let tree_alive = pids.iter().all(alive);
    let record_kept = record.exists();
    let outside = cmd(&base.join("project"), &home).arg("--check").output().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    while pids.iter().any(alive) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let survivors: Vec<i32> = pids.iter().copied().filter(alive).collect();
    for pid in &survivors {
        unsafe { libc::kill(*pid, libc::SIGKILL) };
    }
    let record_left = record.exists();
    let _ = std::fs::remove_dir_all(&base);
    let stderr = String::from_utf8_lossy(&outside.stderr);
    assert!(recorded, "the killed run left no record of its tool groups at {}", record.display());
    assert_eq!(status.trim(), outside.status.code().unwrap().to_string(), "the launch inside the tool's group");
    assert!(tree_alive, "a launch stopped the group it runs in: {pids:?}");
    assert!(record_kept, "the launch inside the tool's group removed the record that lists it");
    assert!(survivors.is_empty(), "the launch outside the group left the tool tree running: {survivors:?}\n{stderr}");
    assert!(!record_left, "the record outlived the launch that stopped its groups");
}

/// A consumer that stops reading leaves the run blocked in a write to a full
/// pipe, so the first signal never reaches the stopped turn: the run stays
/// up. The second must still end it: waiting for stdout on the way out
/// would hold the exit until something drained the pipe, and only SIGKILL,
/// which stops no tool, would end it.
#[test]
fn a_second_signal_exits_while_output_is_stuck_on_a_full_pipe() {
    use std::os::fd::AsRawFd;
    let (project, home) = fresh_dirs("sigterm-stuck-pipe");
    let (base_url, _requests, _server) =
        spawn_scripted_server(vec![(sse_text(&"x".repeat(1 << 20)), true)]);
    write_settings(&home, &base_url);
    let mut child = cmd(&project, &home)
        .args(["--trust-project", "--json", "-p", "say a lot"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Held and never read until the run has ended. The answer is sixteen
    // times what a pipe holds, so once the bytes waiting in the pipe stop
    // growing, the run is blocked writing the rest.
    let mut stdout = child.stdout.take().unwrap();
    let queued = || {
        let mut n: libc::c_int = 0;
        // SAFETY: FIONREAD writes one int, the bytes waiting in the pipe.
        unsafe { libc::ioctl(stdout.as_raw_fd(), libc::FIONREAD, &mut n) };
        n
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let (mut last, mut steady) = (0, 0);
    while steady < 3 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let now = queued();
        steady = if now > 0 && now == last { steady + 1 } else { 0 };
        last = now;
    }
    let pid = child.id() as libc::pid_t;
    let signal = || unsafe { libc::kill(pid, libc::SIGTERM) };
    signal();
    std::thread::sleep(std::time::Duration::from_millis(300));
    let stuck = child.try_wait().unwrap().is_none();
    signal();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let mut written = Vec::new();
    let _ = stdout.read_to_end(&mut written);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    assert!(steady >= 3, "the output never stalled on the full pipe");
    // A precondition, not the behavior under test: while the writer is
    // blocked, the run cannot act on the first signal. A change that lets it
    // must stall this run another way to keep covering the second signal.
    assert!(
        stuck,
        "precondition: the first signal ended a run whose writer was blocked, so this \
         test no longer reaches the second signal's exit; stall the run another way"
    );
    let status = status.expect("the second signal did not end a run stuck writing its output");
    assert_eq!(status.code(), Some(143), "{status:?}");
    let written = String::from_utf8_lossy(&written);
    assert!(!written.contains(r#""type":"done""#), "the stuck run still finished its stream");
}

/// A reload reads every extension manifest, and a read can stall: here the
/// manifest is a pipe nobody writes. No turn runs during a reload, so a
/// signal must end the session at once; waiting the read out held the stop
/// until a second signal.
#[test]
fn a_signal_ends_a_stdio_session_stalled_in_a_reload() {
    use std::os::unix::ffi::OsStrExt;
    let (project, home) = fresh_dirs("sigterm-reload");
    write_settings(&home, "http://127.0.0.1:9/v1");
    let mut child = cmd(&project, &home)
        .args(["--trust-project", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    // A session that never says hello would hold the read below: past the
    // wait, the run is killed, which ends its stdout.
    let (said_hello, watch) = std::sync::mpsc::channel::<()>();
    let pid = child.id() as libc::pid_t;
    std::thread::spawn(move || {
        let wait = std::time::Duration::from_secs(10);
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = watch.recv_timeout(wait) {
            unsafe { libc::kill(pid, libc::SIGKILL) };
        }
    });
    let mut hello = String::new();
    let _ = BufReader::new(child.stdout.take().unwrap()).read_line(&mut hello);
    let _ = said_hello.send(());
    if !hello.contains(r#""type":"hello""#) {
        let _ = child.kill();
        let output = child.wait_with_output().unwrap();
        panic!("no hello: {hello}{}", String::from_utf8_lossy(&output.stderr));
    }
    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    let pipe = tools.join("stalled.toml");
    let path = std::ffi::CString::new(pipe.as_os_str().as_bytes()).unwrap();
    // SAFETY: mkfifo reads one NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o644) }, 0, "{}", std::io::Error::last_os_error());
    writeln!(stdin, r#"{{"cmd":"reload"}}"#).unwrap();
    // A write end opens without blocking only once the reload is reading the
    // pipe. Held open and never written, it keeps that read waiting.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let _writer = loop {
        use std::os::unix::fs::OpenOptionsExt;
        let opened =
            std::fs::OpenOptions::new().write(true).custom_flags(libc::O_NONBLOCK).open(&pipe);
        if let Ok(writer) = opened {
            break writer;
        }
        if std::time::Instant::now() >= deadline || child.try_wait().unwrap().is_some() {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!("the reload never read the manifest: {}", String::from_utf8_lossy(&output.stderr));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    drop(stdin);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    let status = status.expect("SIGTERM did not end a session stalled in a reload");
    assert_eq!(status.code(), Some(143), "{status:?}");
}

/// When the test takes the session index lock, or for one case the trust
/// store's, standing in for another openmax process on the same data dir
/// whose write takes as long as it likes.
#[derive(Clone, Copy, PartialEq, Debug)]
enum IndexHeld {
    /// From before the run starts: its session setup waits on the lock.
    FromLaunch,
    /// As `FromLaunch`, but the lock is freed a moment after the signal, so
    /// the session opens inside the stop's deadline. No turn ran: the stop
    /// must remove the session it opened, not leave it indexed.
    OpenFreedAfterTheSignal,
    /// Once a stdio session has said hello, before its first turn: starting
    /// that turn claims the session, which waits on the lock.
    BeforeTheTurn,
    /// As `BeforeTheTurn`, but the lock is freed a moment after the signal,
    /// so the turn starts inside the stop's deadline: the stop must cancel
    /// that turn and end its stream with the cancelled `done`, not exit
    /// past a turn it never stopped.
    FreedAfterTheSignal,
    /// Once a stdio session has said hello, before a `reload`: the reload
    /// claims the session, which waits on the lock. No turn runs, so the
    /// stop must end the session without an answer to the reload, and the
    /// empty session it leaves stays indexed.
    BeforeAReload,
    /// Once a stdio session has said hello, the trust store's lock instead
    /// of the index, before an `approval_mode`: saving the mode waits on it.
    /// The stop must end the session without an answer to the command, and
    /// with the index free it removes the empty session.
    TrustBeforeAnApprovalMode,
    /// While a fresh run's first turn waits on a model that never answers:
    /// nothing is saved, so the stop removes the empty session, and that
    /// removal waits on the lock.
    DuringTheTurn,
    /// Once the model hung up on a fresh run's first request and stopped
    /// listening, so the turn is retrying. The cancelled turn's save then
    /// waits on the lock inside a runtime worker, and that worker can be the
    /// one left to drive the runtime's timers: no runtime timer fires until
    /// the lock is free.
    AfterTheModelHungUp,
    /// While a stdio session's second turn waits on a model that never
    /// answers, once its first turn finished and saved history. The session
    /// is not empty, so the stop must keep it without a warning, even when
    /// the deadline passes before the empty-session check gets its turn.
    AfterASavedTurn,
}

/// A signal must end a print or stdio run within the stop's deadline
/// wherever the run, or its cancelled turn, is waiting on the session index
/// lock or the trust store's. Each of these waits could hold the first
/// signal unanswered for as long as the other process kept the lock, so that
/// only a second signal ended the run. The run must exit
/// 143 within the deadline, plus half of it for the exit itself, and say so
/// when the empty session it created stays indexed, and only then.
fn stop_with_the_session_index_held(tag: &str, args: &[&str], held: IndexHeld) {
    // The binary's STOP_SETTLE: a cancelled command's settle, then a second
    // for the rest of the turn.
    let bound = open_max_core::execution::CANCEL_SETTLE + std::time::Duration::from_secs(1);
    let margin = bound / 2;
    let (project, home) = fresh_dirs(tag);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    write_settings(&home, &format!("http://{}/v1", listener.local_addr().unwrap()));
    // An endpoint that takes one request and stops listening. Unless it
    // hangs up, the turn stays in that first model request, so nothing of
    // it is saved. After a saved turn, the request it takes is the second:
    // the first gets a finished answer.
    let (asked, request) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        if held == IndexHeld::AfterASavedTurn {
            let Ok((mut stream, _)) = listener.accept() else { return };
            let _ = read_request(&mut stream);
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{HELLO_SSE}",
                    HELLO_SSE.len()
                )
                .as_bytes(),
            );
        }
        let Ok((mut stream, _)) = listener.accept() else { return };
        let _ = read_request(&mut stream);
        let _ = asked.send(stream);
    });
    let sessions = home.join(".openmax").join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    let lock_path = if held == IndexHeld::TrustBeforeAnApprovalMode {
        home.join(".openmax").join("trust.lock")
    } else {
        sessions.join("index.lock")
    };
    let mut held_lock = Some(
        std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(lock_path).unwrap(),
    );
    let take_lock = |lock: &Option<std::fs::File>| {
        lock.as_ref().unwrap().try_lock().expect("nothing else holds the lock yet")
    };
    if matches!(held, IndexHeld::FromLaunch | IndexHeld::OpenFreedAfterTheSignal) {
        take_lock(&held_lock);
    }
    let stdio = args.contains(&"--stdio");
    let mut child = cmd(&project, &home)
        .arg("--trust-project")
        .args(args)
        .stdin(if stdio { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Held open: a stdio client that has not quit, so only the signal ends it.
    let mut stdin = child.stdin.take();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let (line_tx, lines) = std::sync::mpsc::channel::<String>();
    let stderr = {
        let stderr = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            let mut all = String::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let _ = line_tx.send(line.clone());
                all.push_str(&line);
                all.push('\n');
            }
            all
        })
    };
    let wait = std::time::Duration::from_secs(10);
    // The model's connection, held open while the turn waits on it.
    let mut endpoint = None;
    let fail = |child: &mut std::process::Child, what: &str| -> ! {
        let _ = child.kill();
        let _ = child.wait();
        // The run is gone, so its stderr ends: what the reader still holds
        // says why it fell short.
        let said: Vec<String> = std::iter::from_fn(|| lines.recv_timeout(wait).ok()).collect();
        panic!("{tag}: {what}\n{}", said.join("\n"));
    };
    match held {
        IndexHeld::FromLaunch | IndexHeld::OpenFreedAfterTheSignal => {
            // The trust line comes just before the signal watcher starts, and
            // the setup waits on the lock just after it: opening the settings
            // and registering the handlers take milliseconds.
            if !std::iter::from_fn(|| lines.recv_timeout(wait).ok()).any(|l| l.contains("trusted project")) {
                fail(&mut child, "the run never reached its session setup");
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        IndexHeld::BeforeTheTurn
        | IndexHeld::FreedAfterTheSignal
        | IndexHeld::BeforeAReload
        | IndexHeld::TrustBeforeAnApprovalMode => {
            // A session that never says hello would hold the read below:
            // past the wait, the run is killed, which ends its stdout.
            let (said_hello, watch) = std::sync::mpsc::channel::<()>();
            let pid = child.id() as libc::pid_t;
            std::thread::spawn(move || {
                if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = watch.recv_timeout(wait) {
                    unsafe { libc::kill(pid, libc::SIGKILL) };
                }
            });
            let mut hello = String::new();
            let _ = stdout.read_line(&mut hello);
            let _ = said_hello.send(());
            if !hello.contains(r#""type":"hello""#) {
                fail(&mut child, &format!("no hello: {hello}"));
            }
            take_lock(&held_lock);
            let command = match held {
                IndexHeld::BeforeAReload => r#"{"cmd":"reload"}"#,
                IndexHeld::TrustBeforeAnApprovalMode => r#"{"cmd":"approval_mode","mode":"ask"}"#,
                _ => r#"{"cmd":"user","text":"wait"}"#,
            };
            writeln!(stdin.as_mut().unwrap(), "{command}").unwrap();
            // Time for the session to read the command and start on it.
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        IndexHeld::AfterASavedTurn => {
            let stdin = stdin.as_mut().unwrap();
            writeln!(stdin, r#"{{"cmd":"user","text":"one"}}"#).unwrap();
            // A first turn that never ends would hold the read below: past
            // the wait, the run is killed, which ends its stdout.
            let (turn_ended, watch) = std::sync::mpsc::channel::<()>();
            let pid = child.id() as libc::pid_t;
            std::thread::spawn(move || {
                if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = watch.recv_timeout(wait) {
                    unsafe { libc::kill(pid, libc::SIGKILL) };
                }
            });
            let mut line = String::new();
            loop {
                line.clear();
                if stdout.read_line(&mut line).unwrap_or(0) == 0 {
                    fail(&mut child, "the stream ended before the first turn's done");
                }
                if line.contains(r#""type":"done""#) {
                    break;
                }
            }
            let _ = turn_ended.send(());
            if !line.contains(r#""stop_reason":"stop""#) {
                fail(&mut child, &format!("the first turn did not finish: {line}"));
            }
            writeln!(stdin, r#"{{"cmd":"user","text":"wait"}}"#).unwrap();
            match request.recv_timeout(wait) {
                Ok(stream) => endpoint = Some(stream),
                Err(_) => fail(&mut child, "the second turn never asked the endpoint"),
            }
            take_lock(&held_lock);
        }
        IndexHeld::DuringTheTurn | IndexHeld::AfterTheModelHungUp => {
            if let Some(stdin) = stdin.as_mut() {
                writeln!(stdin, r#"{{"cmd":"user","text":"wait"}}"#).unwrap();
            }
            match request.recv_timeout(wait) {
                Ok(stream) if held == IndexHeld::DuringTheTurn => endpoint = Some(stream),
                Ok(hung_up) => drop(hung_up),
                Err(_) => fail(&mut child, "the turn never asked the endpoint"),
            }
            take_lock(&held_lock);
        }
    }
    let signalled = std::time::Instant::now();
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    if matches!(held, IndexHeld::FreedAfterTheSignal | IndexHeld::OpenFreedAfterTheSignal) {
        std::thread::sleep(bound / 6);
        held_lock = None;
    }
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if signalled.elapsed() >= bound + margin {
            let _ = child.kill();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let elapsed = signalled.elapsed();
    let _ = child.wait();
    drop((held_lock, endpoint));
    drop(stdin);
    let stderr = stderr.join().unwrap();
    let mut rest = String::new();
    let _ = stdout.read_to_string(&mut rest);
    let index = std::fs::read_to_string(sessions.join("index.json")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    eprintln!("{tag}: {elapsed:?} from SIGTERM to {status:?}");
    let status = status.unwrap_or_else(|| {
        panic!("{tag}: still running {elapsed:?} after SIGTERM, past the {bound:?} stop deadline: {stderr}")
    });
    assert_eq!(status.code(), Some(143), "{tag}: {status:?}\n{stderr}");
    match held {
        IndexHeld::FromLaunch => {}
        IndexHeld::OpenFreedAfterTheSignal => {
            assert!(!stderr.contains("stays indexed"), "{tag}: {stderr}");
            let listed: Vec<serde_json::Value> = serde_json::from_str(&index).unwrap_or_else(|e| panic!("{tag}: {e}: {index}"));
            assert!(listed.is_empty(), "{tag}: the session opened under the signal stayed indexed: {index}");
        }
        IndexHeld::AfterASavedTurn => {
            assert!(!stderr.contains("stays indexed"), "{tag}: a session with history was reported empty: {stderr}");
            let listed: Vec<serde_json::Value> = serde_json::from_str(&index).unwrap_or_else(|e| panic!("{tag}: {e}: {index}"));
            assert_eq!(listed.len(), 1, "{tag}: the session with history left the index: {index}");
        }
        IndexHeld::BeforeAReload => {
            assert!(rest.is_empty(), "{tag}: the stopped session wrote on after its hello: {rest}");
            assert!(stderr.contains("stays indexed"), "{tag}: the kept empty session went unreported: {stderr}");
        }
        IndexHeld::TrustBeforeAnApprovalMode => {
            assert!(rest.is_empty(), "{tag}: the stopped session wrote on after its hello: {rest}");
            assert!(!stderr.contains("stays indexed"), "{tag}: {stderr}");
            let listed: Vec<serde_json::Value> = serde_json::from_str(&index).unwrap_or_else(|e| panic!("{tag}: {e}: {index}"));
            assert!(listed.is_empty(), "{tag}: the empty session stayed indexed: {index}");
        }
        IndexHeld::FreedAfterTheSignal => {
            let done = last_event(rest.as_bytes());
            assert_eq!(
                (done["type"].as_str(), done["stop_reason"].as_str()),
                (Some("done"), Some("cancelled")),
                "{tag}: the turn that started under the signal was not stopped: {rest}\n{stderr}"
            );
        }
        _ => assert!(stderr.contains("stays indexed"), "{tag}: the kept empty session went unreported: {stderr}"),
    }
}

#[test]
fn a_signal_ends_a_fresh_stdio_session_on_time_while_the_session_index_is_held() {
    stop_with_the_session_index_held("index-held-stdio", &["--stdio"], IndexHeld::DuringTheTurn);
}

#[test]
fn a_signal_ends_a_fresh_print_run_on_time_while_the_session_index_is_held() {
    stop_with_the_session_index_held("index-held-print", &["--json", "-p", "wait"], IndexHeld::DuringTheTurn);
}

#[test]
fn a_signal_ends_a_fresh_stdio_session_on_time_after_the_model_hung_up_while_the_session_index_is_held() {
    stop_with_the_session_index_held("index-held-stdio-hung-up", &["--stdio"], IndexHeld::AfterTheModelHungUp);
}

#[test]
fn a_signal_ends_a_fresh_print_run_on_time_after_the_model_hung_up_while_the_session_index_is_held() {
    stop_with_the_session_index_held(
        "index-held-print-hung-up",
        &["--json", "-p", "wait"],
        IndexHeld::AfterTheModelHungUp,
    );
}

#[test]
fn a_signal_keeps_a_stdio_session_with_history_without_a_warning_while_the_session_index_is_held() {
    stop_with_the_session_index_held("index-held-stdio-saved", &["--stdio"], IndexHeld::AfterASavedTurn);
}

#[test]
fn a_signal_ends_a_stdio_session_waiting_on_the_session_index_to_open() {
    stop_with_the_session_index_held("index-held-stdio-open", &["--stdio"], IndexHeld::FromLaunch);
}

#[test]
fn a_signal_ends_a_print_run_waiting_on_the_session_index_to_open() {
    stop_with_the_session_index_held("index-held-print-open", &["--json", "-p", "wait"], IndexHeld::FromLaunch);
}

#[test]
fn a_stdio_session_that_opens_after_the_signal_is_removed_and_ends() {
    stop_with_the_session_index_held("index-freed-stdio-open", &["--stdio"], IndexHeld::OpenFreedAfterTheSignal);
}

#[test]
fn a_print_run_whose_session_opens_after_the_signal_removes_it_and_ends() {
    stop_with_the_session_index_held(
        "index-freed-print-open",
        &["--json", "-p", "wait"],
        IndexHeld::OpenFreedAfterTheSignal,
    );
}

#[test]
fn a_signal_ends_a_stdio_session_waiting_on_the_session_index_to_start_a_turn() {
    stop_with_the_session_index_held("index-held-stdio-turn", &["--stdio"], IndexHeld::BeforeTheTurn);
}

#[test]
fn a_turn_that_starts_after_the_signal_is_cancelled_and_ends_its_stream() {
    stop_with_the_session_index_held("index-freed-stdio-turn", &["--stdio"], IndexHeld::FreedAfterTheSignal);
}

#[test]
fn a_signal_ends_a_stdio_session_waiting_on_the_session_index_to_reload() {
    stop_with_the_session_index_held("index-held-stdio-reload", &["--stdio"], IndexHeld::BeforeAReload);
}

#[test]
fn a_signal_ends_a_stdio_session_waiting_on_the_trust_store_to_save_an_approval_mode() {
    stop_with_the_session_index_held(
        "trust-held-stdio-approval-mode",
        &["--stdio"],
        IndexHeld::TrustBeforeAnApprovalMode,
    );
}

/// A signal the run inherited as ignored stays ignored: `nohup` keeps a run
/// alive past its terminal's SIGHUP, and a shell starts a script's
/// background job with SIGINT ignored so Ctrl+C stops only the foreground.
/// A handler for it would cancel the turn the parent meant to keep.
#[test]
fn a_signal_the_run_inherited_as_ignored_stays_ignored() {
    use std::os::unix::process::CommandExt;
    let (project, home) = fresh_dirs("sighup-ignored");
    let pid_file = project.parent().unwrap().join("pids");
    let script = format!("echo $$ >> '{}'; sleep 1", pid_file.display());
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (sse_tool_call("bash", serde_json::json!({ "command": script })), true),
        (sse_text("done"), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");
    let mut command = cmd(&project, &home);
    command
        .args(["--trust-project", "--json", "-p", "outlive the terminal"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // SAFETY: sigaction is async-signal-safe, and nothing else runs between
    // fork and exec.
    unsafe {
        command.pre_exec(|| {
            let mut ignore: libc::sigaction = std::mem::zeroed();
            ignore.sa_sigaction = libc::SIG_IGN;
            match libc::sigaction(libc::SIGHUP, &ignore, std::ptr::null_mut()) {
                0 => Ok(()),
                _ => Err(std::io::Error::last_os_error()),
            }
        });
    }
    let mut child = command.spawn().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::fs::read_to_string(&pid_file).unwrap_or_default().trim().is_empty() {
        if std::time::Instant::now() >= deadline || child.try_wait().unwrap().is_some() {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!("the tool never started: {}", String::from_utf8_lossy(&output.stderr));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGHUP) };
    let output = finish_with_deadline(child);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{:?}\n{stderr}", output.status);
    let done = last_event(&output.stdout);
    assert_eq!((done["type"].as_str(), done["stop_reason"].as_str()), (Some("done"), Some("stop")), "{done}");
}

/// A supervisor stops a TUI session with a signal. The session ends through
/// its normal exit path, which hands the terminal back, and then exits 128
/// plus the signal's number: a wrapper that read 0 took the stop for a
/// /quit. It returns through main, so the runtime's shutdown drops the
/// command still running, and the command's group guard kills the tool tree
/// it left behind, which `kill_on_drop` alone would leave running.
#[test]
fn a_signal_ends_the_tui_with_its_status_and_stops_the_tool_tree() {
    let (project, home) = fresh_dirs("sigterm-tui");
    let pid_file = project.parent().unwrap().join("pids");
    let script = format!(
        "echo $$ >> '{p}'; sleep 6841 & echo $! >> '{p}'; exec sleep 7841",
        p = pid_file.display()
    );
    let (base_url, _requests, _server) = spawn_scripted_server(vec![
        (sse_tool_call("bash", serde_json::json!({ "command": script })), true),
        (sse_text("done"), true),
    ]);
    write_settings_with_mode(&home, &base_url, "auto");
    let (pty, terminal) = pseudo_terminal();
    let size = libc::winsize { ws_row: 24, ws_col: 80, ws_xpixel: 0, ws_ypixel: 0 };
    // SAFETY: TIOCSWINSZ reads one winsize from a live descriptor.
    unsafe { libc::ioctl(std::os::fd::AsRawFd::as_raw_fd(&terminal), libc::TIOCSWINSZ, &size) };
    let mut child = cmd(&project, &home)
        .arg("--trust-project")
        .env("TERM", "xterm-256color")
        .stdin(terminal.try_clone().unwrap())
        .stdout(terminal.try_clone().unwrap())
        .stderr(terminal)
        .spawn()
        .unwrap();
    // A terminal nobody reads stops the writer, so the screen is drained on
    // a thread for as long as the session holds it.
    let screen = Arc::new(Mutex::new(Vec::new()));
    let mut reader = pty.try_clone().unwrap();
    let drained = screen.clone();
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        while let Ok(n) = reader.read(&mut buf) {
            if n == 0 {
                break;
            }
            drained.lock().unwrap().extend_from_slice(&buf[..n]);
        }
    });
    let shown = |bytes: &[u8]| screen.lock().unwrap().windows(bytes.len()).any(|w| w == bytes);
    let wait_for = |what: &str, done: &dyn Fn() -> bool, child: &mut std::process::Child| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !done() {
            if std::time::Instant::now() >= deadline || child.try_wait().unwrap().is_some() {
                let _ = child.kill();
                let _ = child.wait();
                panic!("{what}: {}", String::from_utf8_lossy(&screen.lock().unwrap()));
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    };
    // The keyboard protocol query goes out behind the first frame. Answered
    // as a terminal without the protocol, input starts at once.
    wait_for("the TUI never asked about the keyboard", &|| shown(b"\x1b[?u"), &mut child);
    let mut keys = pty.try_clone().unwrap();
    keys.write_all(b"\x1b[?62cstart the tree\r").unwrap();
    let pids = || -> Vec<i32> {
        let recorded = std::fs::read_to_string(&pid_file).unwrap_or_default();
        recorded.lines().filter_map(|l| l.trim().parse().ok()).collect()
    };
    wait_for("the tool tree never started", &|| pids().len() == 2, &mut child);
    let pids = pids();
    let restored_before = shown(b"\x1b[?1049l");

    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let alive = |pid: &i32| unsafe { libc::kill(*pid, 0) == 0 };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while pids.iter().any(alive) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let survivors: Vec<i32> = pids.iter().copied().filter(alive).collect();
    for pid in &survivors {
        unsafe { libc::kill(*pid, libc::SIGKILL) };
    }
    let screen_text = String::from_utf8_lossy(&screen.lock().unwrap()).into_owned();
    drop(pty);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
    let status = status.expect("SIGTERM did not end the TUI");
    assert!(!restored_before, "the alternate screen was left before the signal");
    assert!(screen_text.contains("\x1b[?1049l"), "the terminal was not handed back: {screen_text:?}");
    assert_eq!(status.code(), Some(143), "{status:?}");
    assert!(survivors.is_empty(), "tool processes outlived the session: {survivors:?}");
}

/// A fresh install has no endpoint, model or window. One run must name all
/// three with an example, because settings are read at launch: reported one
/// at a time, each field costs an edit and a restart.
#[test]
fn a_fresh_install_is_told_every_missing_setting_at_once() {
    let (project, home) = fresh_dirs("fresh-install");
    let out = cmd(&project, &home).args(["--trust-project", "-p", "hi"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("no endpoint, model or context window configured"), "{stderr}");
    for key in ["\"base_url\"", "\"model\"", "\"context_tokens\"", "\"api_key\""] {
        assert!(stderr.contains(key), "{key} missing from {stderr}");
    }
    assert!(stderr.contains("read at launch"), "{stderr}");
    // The error is the documentation of the first run: it names the flags
    // that set each key as well as the keys.
    for flag in ["--base-url", "--model", "--context-tokens", "--api-key"] {
        assert!(stderr.contains(flag), "{flag} missing from {stderr}");
    }
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// The whole first run is one command. On a fresh install, with no
/// settings.json and no trust.json, `--trust-project` with `--base-url`,
/// `--model`, `--api-key`, `--context-tokens` and `-p` records the project in
/// auto, writes the four keys to settings.json before the turn, and prints
/// the answer on stdout: nothing written by hand, nothing restarted. The next
/// plain run needs no flag, and the same command run again changes nothing
/// and says nothing about settings. The key is given as `$NAME` and written
/// as given, so settings.json names the variable and never holds the secret
/// the variable carries, which only the request does. The command with an
/// empty task is a usage error, refused before it grants trust or writes a
/// key.
///
/// The thesis target is under 30 seconds from a fresh install to a first
/// real answer. Nearly all of that is the download and the model's own
/// latency, so the harness's share, measured here as the wall time of the
/// one command against a local endpoint, is held under one second in a
/// release build (debug builds print it and hold a 20 second sanity bound).
/// The median gate over repeated runs is in tests/thesis.rs.
#[test]
fn a_fresh_install_answers_in_one_command() {
    let (project, home) = fresh_dirs("one-command");
    let (base_url, headers) = spawn_header_server();
    let one_command = [
        "--trust-project",
        "--base-url",
        base_url.as_str(),
        "--model",
        "stub-model",
        "--api-key",
        "$FIRST_RUN_KEY",
        "--context-tokens",
        "16384",
        "-p",
        "say hi",
    ];
    let secret = "first-run-secret";
    let mut empty_task = one_command;
    *empty_task.last_mut().unwrap() = "";
    let out = cmd(&project, &home).env("FIRST_RUN_KEY", secret).args(empty_task).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.starts_with("openmax: each --print requires a non-empty prompt"), "{stderr}");
    assert!(!home.join(".openmax").exists(), "a usage error wrote nothing: {stderr}");

    let started = std::time::Instant::now();
    let out = cmd(&project, &home).env("FIRST_RUN_KEY", secret).args(one_command).output().unwrap();
    let elapsed = started.elapsed();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("stub says hi"), "stdout: {stdout}\nstderr: {stderr}");
    let settings_path = home.join(".openmax/settings.json");
    let written = std::fs::read_to_string(&settings_path).unwrap();
    let settings: serde_json::Value = serde_json::from_str(&written).unwrap();
    assert_eq!(settings["base_url"], base_url.as_str(), "{written}");
    assert_eq!(settings["model"], "stub-model", "{written}");
    assert_eq!(settings["api_key"], "$FIRST_RUN_KEY", "{written}");
    assert!(!written.contains(secret), "{written}");
    assert_eq!(settings["context_tokens"], 16384, "{written}");
    let sent = headers.lock().unwrap().concat();
    let authorization = sent.lines().find_map(|l| {
        let (name, value) = l.split_once(':')?;
        name.eq_ignore_ascii_case("authorization").then(|| value.trim())
    });
    assert_eq!(authorization, Some("Bearer first-run-secret"), "the key the variable carries reaches the server: {sent}");
    assert!(
        stderr.contains(&format!("openmax: wrote base_url, api_key, model, context_tokens to {}", settings_path.display())),
        "the run must say what it wrote: {stderr}"
    );
    assert_eq!(approval_mode(&home, &project), ApprovalMode::Auto, "{stderr}");
    if cfg!(debug_assertions) {
        eprintln!("one-command first run: {elapsed:?} (debug build; the release build holds it under 1 s)");
        assert!(elapsed < std::time::Duration::from_secs(20), "{elapsed:?}");
    } else {
        assert!(elapsed < std::time::Duration::from_secs(1), "the harness's share of a first answer: {elapsed:?}");
    }

    let out = cmd(&project, &home).args(["-p", "say hi"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "the configured install runs with no flag: {stderr}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("stub says hi"), "{stderr}");

    let out = cmd(&project, &home).env("FIRST_RUN_KEY", secret).args(one_command).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(!stderr.contains("settings.json"), "flags equal to the file change nothing and say nothing: {stderr}");
    assert_eq!(std::fs::read_to_string(&settings_path).unwrap(), written);
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// On an install that already sets a key, the flag for it applies to this
/// run only: the turn goes to the flag's endpoint, the run says which keys
/// the command line overrode, and settings.json keeps its bytes, so a typo
/// in a one-off flag can never break the next plain run. A `--base-url`
/// other than the file's is another server, so a key the file leaves unset
/// is not written either: the next plain run would send it to the file's.
#[test]
fn flags_override_a_configured_install_for_this_run_only() {
    let (project, home) = fresh_dirs("override");
    let (base_url, _requests, _server) = spawn_stub_server();
    write_settings_with_mode(&home, "http://127.0.0.1:9/v1", "auto");
    let settings_path = home.join(".openmax/settings.json");
    let before = std::fs::read(&settings_path).unwrap();
    let out = cmd(&project, &home)
        .args(["--trust-project", "--base-url", &base_url, "--api-key", "sk-flag", "--context-tokens", "32768", "-p", "say hi"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("stub says hi"), "the turn must use the flag's endpoint: {stderr}");
    assert!(
        stderr.contains(&format!(
            "openmax: for this run only, the command line overrides base_url, api_key, context_tokens in {}; the file is unchanged, so set them there to keep them",
            settings_path.display()
        )),
        "{stderr}"
    );
    assert!(!stderr.contains("wrote"), "{stderr}");
    assert_eq!(std::fs::read(&settings_path).unwrap(), before, "a configured install is never rewritten by a flag");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// A stub endpoint answering like `spawn_stub_server` that records each
/// request's header block, for what a request carries besides its body.
fn spawn_header_server() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => break,
                }
            }
            let head = String::from_utf8_lossy(&head).into_owned();
            let length = head
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            let mut body = vec![0u8; length];
            let _ = stream.read_exact(&mut body);
            log.lock().unwrap().push(head);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{HELLO_SSE}",
                HELLO_SSE.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1"), seen)
}

/// settings.json's api_key is the credential for the server its base_url
/// names. A one-off `--base-url` for another server is sent no settings
/// key: its requests carry no Authorization header, the run says why and
/// how to send one, and the file keeps its key. With `--api-key` the run
/// sends its own.
#[test]
fn a_base_url_flag_for_another_server_is_not_sent_the_files_key() {
    let (project, home) = fresh_dirs("withheld-key");
    let dir = home.join(".openmax");
    std::fs::create_dir_all(&dir).unwrap();
    let settings_path = dir.join("settings.json");
    std::fs::write(
        &settings_path,
        r#"{"base_url":"http://127.0.0.1:9/v1","api_key":"sk-file-secret","model":"stub-model","approval_mode":"auto","context_tokens":16384}"#,
    )
    .unwrap();
    let before = std::fs::read(&settings_path).unwrap();
    for (key, sent) in [(None, None), (Some("sk-flag"), Some("bearer sk-flag"))] {
        let (base_url, headers) = spawn_header_server();
        let mut args = vec!["--trust-project", "--base-url", base_url.as_str()];
        args.extend(key.map(|k| ["--api-key", k]).into_iter().flatten());
        let out = cmd(&project, &home).args(args).args(["-p", "say hi"]).output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
        assert!(stdout.contains("stub says hi"), "{stderr}");
        let headers = headers.lock().unwrap().join("").to_ascii_lowercase();
        assert!(!headers.contains("sk-file-secret"), "the file's key reached another server: {headers}");
        let authorization = headers.lines().find_map(|l| l.strip_prefix("authorization: ")).map(str::trim);
        assert_eq!(authorization, sent, "{headers}");
        let withheld = format!(
            "openmax: this run sends no api_key: the one in {} is for the server its base_url names, and --base-url names another; pass --api-key to send one",
            settings_path.display()
        );
        assert_eq!(stderr.contains(&withheld), key.is_none(), "{stderr}");
        assert_eq!(std::fs::read(&settings_path).unwrap(), before, "the file keeps its key");
    }
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Where settings.json selects a provider, its providers.json entry gives
/// the key and may size the model, so the flags for those say what the run
/// does with them. `--api-key` is the key for `base_url`, never sent to a
/// provider on another server, so it is refused naming the provider, before
/// the trust grant, with nothing written. `--context-tokens` is the window
/// for a model the entry does not size: it is written while settings.json
/// leaves it unset, and where the entry sizes the model the run says that
/// window wins.
#[test]
fn settings_flags_beside_a_selected_provider_say_what_the_run_uses() {
    let (project, home) = fresh_dirs("provider-flags");
    let (base_url, _requests, _server) = spawn_stub_server();
    let dir = home.join(".openmax");
    std::fs::create_dir_all(&dir).unwrap();
    let catalog = serde_json::json!({"providers": {"stub": {"base_url": base_url, "models": [{"id": "stub-model", "context_tokens": 16384}]}}});
    std::fs::write(dir.join("providers.json"), catalog.to_string()).unwrap();
    let settings_path = dir.join("settings.json");
    std::fs::write(&settings_path, r#"{"provider":"stub","model":"stub-model","approval_mode":"auto"}"#).unwrap();
    let before = std::fs::read(&settings_path).unwrap();

    let out = cmd(&project, &home).args(["--trust-project", "--api-key", "sk-literal", "-p", "say hi"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--api-key is the key for settings.json's base_url, but this run uses provider 'stub'"), "{stderr}");
    assert_eq!(std::fs::read(&settings_path).unwrap(), before, "a refused flag writes nothing");
    assert_eq!(open_max_core::trust::is_trusted(&dir, &project), Ok(false), "a refused flag grants no trust: {stderr}");

    let out = cmd(&project, &home).args(["--trust-project", "--context-tokens", "9000", "-p", "say hi"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("stub says hi"), "{stderr}");
    assert!(stderr.contains(&format!("openmax: wrote context_tokens to {}", settings_path.display())), "{stderr}");
    assert!(
        stderr.contains("openmax: --context-tokens 9000 is not this run's window: provider 'stub' gives model stub-model 16384 tokens in ~/.openmax/providers.json, which wins"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}

/// Each request a window server read: its token count, and whether it was
/// refused.
type WindowLog = Arc<Mutex<Vec<(usize, bool)>>>;

/// A provider that counts a token per three request bytes (denser than the
/// harness's bytes/4 estimate, as on JSON or a small-vocabulary model) and
/// refuses, as a window does, every request whose count plus `max_tokens`
/// is over `window`. A prompt starting with `BULK` is answered with ~18 KB of
/// text, anything else with `ok`; answers report their prompt tokens. Each
/// request's count and whether it was refused is recorded.
fn spawn_window_server(window: Arc<std::sync::atomic::AtomicUsize>) -> (String, WindowLog) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen: WindowLog = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let Some(body) = read_request(&mut stream) else { continue };
            let request: serde_json::Value = serde_json::from_str(&body).unwrap();
            let tokens = body.len() / 3;
            let reserve = request["max_tokens"].as_u64().or(request["max_completion_tokens"].as_u64()).unwrap_or(0);
            let limit = window.load(std::sync::atomic::Ordering::SeqCst);
            let refused = tokens + reserve as usize > limit;
            log.lock().unwrap().push((tokens, refused));
            let response = if refused {
                let error = serde_json::json!({"error": {
                    "message": format!("This model's maximum context length is {limit} tokens. However, you requested {} tokens.", tokens + reserve as usize),
                    "type": "invalid_request_error",
                }})
                .to_string();
                format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{error}",
                    error.len()
                )
            } else {
                let prompt = request["messages"].as_array().and_then(|m| m.last()).and_then(|m| m["content"].as_str()).unwrap_or("");
                let text = if prompt.starts_with("BULK") { "lorem ipsum ".repeat(1_500) } else { "ok".to_string() };
                let sse = sse(&[
                    serde_json::json!({"choices":[{"delta":{"content":text},"finish_reason":null}]}),
                    serde_json::json!({"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":tokens,"completion_tokens":1}}),
                ]);
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{sse}",
                    sse.len()
                )
            };
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1"), seen)
}

/// Read stdio events through the turn's `done`, returning its stop reason
/// and every event line.
fn stdio_turn(reader: &mut impl BufRead) -> (String, Vec<serde_json::Value>) {
    let mut events = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        assert!(reader.read_line(&mut line).unwrap() > 0, "the session ended mid-turn: {events:?}");
        let event: serde_json::Value = serde_json::from_str(&line).unwrap();
        if event["type"] == "done" {
            return (event["stop_reason"].as_str().unwrap_or("").to_string(), events);
        }
        events.push(event);
    }
}

/// A session continued onto a smaller window starts on the plain bytes/4
/// estimate, which a dense tokenizer outruns: the first request is under the
/// budget by the estimate and over the window by the provider's count, so it
/// is refused. A refusal reports no usage, so nothing calibrated and every
/// later turn sent the same transcript and was refused the same way. The
/// refusal now proves a floor on the ratio. This provider counts about a
/// third more than the estimate, so the refused request is well under 1.5
/// times what the window leaves for it, and the next turn compacts and is
/// answered.
#[test]
fn a_refusal_for_length_lets_the_next_turn_compact_and_be_answered() {
    let (project, home) = fresh_dirs("length-refusal");
    let window = Arc::new(std::sync::atomic::AtomicUsize::new(200_000));
    let (base_url, requests) = spawn_window_server(window.clone());
    let settings = |context_tokens: usize| {
        let dir = home.join(".openmax");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("settings.json"),
            format!(r#"{{"base_url":"{base_url}","model":"stub-model","approval_mode":"auto","context_tokens":{context_tokens}}}"#),
        )
        .unwrap();
    };

    settings(200_000);
    for args in [vec!["--trust-project", "-p", "BULK one"], vec!["--trust-project", "--continue", "-p", "BULK two"]] {
        let out = cmd(&project, &home).args(&args).output().unwrap();
        assert_eq!(out.status.code(), Some(0), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
    assert!(requests.lock().unwrap().iter().all(|&(_, refused)| !refused));

    settings(16_384);
    window.store(16_384, std::sync::atomic::Ordering::SeqCst);
    let before = requests.lock().unwrap().len();
    let mut child = cmd(&project, &home)
        .args(["--trust-project", "--continue", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut reader = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    assert!(line.contains("\"hello\""), "{line}");
    line.clear();
    reader.read_line(&mut line).unwrap();
    assert!(line.contains("\"transcript\""), "{line}");

    writeln!(stdin, r#"{{"cmd":"user","text":"summarize both"}}"#).unwrap();
    let (first, events) = stdio_turn(&mut reader);
    assert_eq!(first, "error", "{events:?}");
    let budget = events.iter().find(|e| e["type"] == "budget").expect("a budget line");
    assert!(budget["used_tokens"].as_u64().unwrap() <= 16_384 - (4_096 + 1_024), "the estimate let it through: {budget}");
    assert!(
        events.iter().any(|e| e["type"] == "error" && e["message"].as_str().unwrap_or("").contains("maximum context length")),
        "{events:?}"
    );

    writeln!(stdin, r#"{{"cmd":"user","text":"try again"}}"#).unwrap();
    let (second, events) = stdio_turn(&mut reader);
    writeln!(stdin, r#"{{"cmd":"quit"}}"#).unwrap();
    drop(stdin);
    let out = finish_with_deadline(child);
    assert_eq!(second, "stop", "the next turn must fit: {events:?}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(events.iter().any(|e| e["type"] == "message_done" && e["text"] == "ok"), "{events:?}");

    let phase = requests.lock().unwrap()[before..].to_vec();
    assert!(phase[0].1, "the continued session's first request is refused: {phase:?}");
    let answered = phase.last().unwrap();
    assert!(!answered.1 && answered.0 + 4_096 <= 16_384, "{phase:?}");
    let _ = std::fs::remove_dir_all(project.parent().unwrap());
}
