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
/// way sh splits the forms the docs use: bare words, double-quoted strings,
/// and a `# comment`.
fn shell_words(line: &str) -> (Vec<String>, String) {
    let (mut words, mut word, mut quoted) = (Vec::new(), None::<String>, false);
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                quoted = !quoted;
                word.get_or_insert_with(String::new);
            }
            '#' if !quoted && word.is_none() => return (words, chars.as_str().trim().to_string()),
            c if c.is_whitespace() && !quoted => words.extend(word.take()),
            c => word.get_or_insert_with(String::new).push(c),
        }
    }
    assert!(!quoted, "unterminated quote in a documented command: {line}");
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
/// hands it. No variable is set that the line does not set itself. Success is
/// the documented result: the run itself works, the project is trusted in
/// auto although settings.json says ask, and a frontend's plain
/// `openmax --stdio` then starts with no flag.
fn run_documented_trust_command(line: &str) -> Result<(), String> {
    let (words, comment) = shell_words(line);
    let at = words.iter().position(|w| w == "openmax").ok_or("no openmax invocation")?;
    let (project, home) = fresh_dirs("documented-trust");
    let (base_url, _requests, _server) = spawn_stub_server();
    write_settings(&home, &base_url);
    let mut command = plain_cmd(&project, &home);
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
        command.stdin(Stdio::piped());
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
    } else if args.iter().any(|a| a == "-p") {
        finish_with_deadline(child)
    } else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("neither -p nor --stdio: teach this test the documented result".into());
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
    write_settings(&home, "http://127.0.0.1:9/v1");
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
    assert_eq!(hello["proto"], "openmax-stdio/6");
    assert_eq!(hello["protocol_version"], 6);
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
    assert!(stuck, "the first signal ended a run whose writer was blocked");
    let status = status.expect("the second signal did not end a run stuck writing its output");
    assert_eq!(status.code(), Some(143), "{status:?}");
    let written = String::from_utf8_lossy(&written);
    assert!(!written.contains(r#""type":"done""#), "the stuck run still finished its stream");
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

/// A supervisor stopping a session, or a terminal closing under it, ends the
/// TUI with a signal. The session ends through its normal exit path, which
/// hands the terminal back, and then exits 128 plus the signal's number: a
/// wrapper that read 0 took the stop for a /quit. Exiting there runs no
/// destructor, so a tool tree the turn left running must be killed on the
/// way out, or it outlives the session as it would under a plain exit.
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
