//! End-to-end tests of the built binary: trust gating, --check exit codes,
//! the --stdio handshake, and a full print-mode turn against a stub
//! OpenAI-compatible server. These are the contracts scripts and frontends
//! build on, exercised exactly as a user's shell would.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

fn openmax_bin() -> &'static str {
    env!("CARGO_BIN_EXE_openmax")
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
            r#"{{"base_url":"{base_url}","model":"stub-model","approval_mode":"{approval_mode}"}}"#
        ),
    )
    .unwrap();
}

fn cmd(project: &Path, home: &Path) -> Command {
    let mut c = Command::new(openmax_bin());
    c.current_dir(project);
    c.env("HOME", home);
    c.env_remove("OPENMAX_API_KEY");
    // A developer dogfooding openmax runs cargo test from inside a session;
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
/// tty) is exactly this shape, and round-4 dogfooding watched an agent
/// reach for it on its first attempt.
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
    let out = bare(&["--trust-project", "-p", "hi"]);
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    // The attestation (what cmd() sets) is what lets test automation through.
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
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

    // Trust it (the endpoint is dead, so the turn fails after trust is stored).
    cmd(&project, &home).args(["--trust-project", "-p", "hi"]).output().unwrap();

    // Trusted but unapproved: each example probes in a sandbox with zero
    // host authority instead of refusing flat. The harmless prover passes
    // (marked sandboxed, with the approve pointer), the broken failer fails
    // with its own diagnostic, and nothing is blessed by any of it. On a
    // host with no sandbox backend, the fall-back refusal keeps the old
    // unapproved-source wording; both are exit 1 here (failer always fails).
    let out = cmd(&project, &home)
        .args(["--check", "--json", "--run-examples"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let value = json(&out);
    let reported = messages(&value);
    if reported.contains("cannot sandbox a probe") {
        assert_eq!(reported.matches("cannot sandbox a probe").count(), 2, "{reported}");
        assert!(reported.contains("--approve"), "{reported}");
    } else {
        assert!(reported.contains("ran in a sandbox"), "{reported}");
        assert!(reported.contains("openmax --approve"), "{reported}");
        assert!(reported.contains("boom"), "{reported}");
        assert!(
            value
                .as_array()
                .unwrap()
                .iter()
                .filter(|row| row["surface"] == "example")
                .all(|row| row["sandboxed"] == true),
            "unapproved probes must be labeled: {value}"
        );
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

/// An approval store inherited from a release that kept them in a plain file
/// is not authority until a human says so: nothing in it takes effect, it
/// survives being read, and adopting it is guarded like every other command
/// that moves authority. The adoption itself is proven at the unit level,
/// where no confirmation prompt stands in the way.
#[test]
fn an_inherited_approval_store_waits_for_a_human() {
    use open_max_core::ledger::sha256_hex;
    let (project, home) = fresh_dirs("adopt");
    let hooks = project.join(".openmax").join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    let gate = hooks.join("gate.toml");
    std::fs::write(&gate, "event = \"pre_tool_use\"\ncommand = \"/usr/bin/true\"\n").unwrap();
    let sha = sha256_hex(&std::fs::read(&gate).unwrap());

    // Seed this project's ledger so the test knows which directory is its
    // own, then rewrite it into what the released build would have left: a
    // v1 chained log plus a plain store beside it.
    let out = cmd(&project, &home)
        .args(["--approve", ".openmax/hooks/gate.toml"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let dir = std::fs::read_dir(home.join(".openmax").join("ledger"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("log.jsonl").exists())
        .expect("a seeded ledger");
    let record = format!(
        "{{\"v\":1,\"ts\":1,\"path\":{},\"sha256\":\"{sha}\",\"actor\":\"initial\",\"prev\":\"\"}}",
        serde_json::to_string(&gate.display().to_string()).unwrap()
    );
    std::fs::write(dir.join("log.jsonl"), format!("{record}\n")).unwrap();
    std::fs::write(dir.join("chain-head"), sha256_hex(record.as_bytes())).unwrap();
    let _ = std::fs::remove_file(dir.join("chain-head.pending"));
    std::fs::write(
        dir.join("approved.json"),
        format!("{{\"version\":1,\"hashes\":[\"{sha}\"],\"paths\":[]}}"),
    )
    .unwrap();

    let out = cmd(&project, &home).arg("--check").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert_eq!(out.status.code(), Some(1), "an unadopted store is not in effect: {stdout}");
    assert!(stdout.contains("--adopt-approvals"), "the way in must be named: {stdout}");
    assert!(dir.join("approved.json").exists(), "a read must not consume it");

    let out = cmd(&project, &home)
        .arg("--adopt-approvals")
        .env("OPENMAX_SESSION", "s-1")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(3), "an agent session must not adopt");

    let out = cmd(&project, &home).arg("--adopt-approvals").output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(3), "no terminal, no adoption: {stdout}");
    assert!(stdout.contains("nothing in it is in effect"), "{stdout}");
    assert!(dir.join("approved.json").exists(), "a refusal must not consume it either");
}

/// A ledger nobody can verify refuses to be read as history and names the
/// way back - which is a human at an interactive terminal: agent sessions
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

    let out = cmd(&project, &home).arg("--ledger").output().unwrap();
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
    assert_eq!(hello["proto"], "openmax-stdio/4");
    assert_eq!(hello["protocol_version"], 4);
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

/// The same call as markup leaking into `content`, the shape the fallback
/// parser recovers (it deliberately accepts an unclosed final tag).
const WRITE_CALL_MARKUP_SSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"<tool_call>{\\\"name\\\":\\\"write_file\\\",\\\"arguments\\\":{\\\"path\\\":\\\"side-effect.txt\\\",\\\"content\\\":\\\"written\\\"}}</tool_call>\"}}]}\n\n";

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
    write_settings_with_mode(&home, &base_url, "auto");

    let tools = project.join(".openmax").join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    std::fs::write(
        tools.join("peek.toml"),
        "name = \"peek\"\ndescription = \"look something up\"\ncommand = \"/bin/sh\"\n\
         args = [\"-c\", \"cat >/dev/null; echo ran >> peeked.txt; echo looked\"]\nmutating = false\n\
         \n[params]\ntype = \"object\"\n[params.properties.count]\ntype = \"number\"\n",
    )
    .unwrap();

    let out = cmd(&project, &home)
        .args(["--trust-project", "--json", "-p", "peek twice"])
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
    // auto mode: nothing but the content boundary can produce a prompt here.
    write_settings_with_mode(&home, &base_url, "auto");

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

    let out = cmd(&project, &home)
        .args(["--trust-project", "--json", "-p", "peek at it"])
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

    // The operator running headless gets a command, not a placeholder.
    assert!(
        stderr.contains("openmax --approve .openmax/tools/peek.toml"),
        "stderr must name the real file: {stderr}"
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
/// the call must not run. Both routes into `tool_calls` are covered: native
/// deltas, and markup recovered from content by the fallback parser.
#[test]
fn a_truncated_stream_never_runs_the_tool_call_it_carried() {
    for (tag, body) in [("native", WRITE_CALL_SSE), ("markup", WRITE_CALL_MARKUP_SSE)] {
        let (project, home) = fresh_dirs(&format!("truncated-call-{tag}"));
        let (base_url, _requests, _server) = spawn_scripted_server(vec![(body.to_string(), false)]);
        // auto, so a refusal here is the truncation and not the approval gate.
        write_settings_with_mode(&home, &base_url, "auto");

        let (code, lines, stdout) = json_turn(&project, &home, "write the file");
        assert_truncated_turn(code, &lines, &stdout);
        // The property that matters: no side effect.
        assert!(
            !project.join("side-effect.txt").exists(),
            "{tag}: a call from an unterminated stream must not run: {stdout}"
        );
        assert!(
            !lines.iter().any(|l| l["type"] == "tool_start"),
            "{tag}: no tool may even be dispatched: {stdout}"
        );
        assert!(
            lines.iter().any(|l| l["type"] == "error"
                && l["message"].as_str().is_some_and(|m| m.contains("did not run"))),
            "{tag}: the error must say the call was refused: {stdout}"
        );
        // The refused call was persisted before it was refused, so it needs a
        // tool reply: an unanswered tool_call id breaks chat-template replay
        // and would make the session unresumable.
        let saved = session_dump(&home);
        assert!(
            saved.contains("\"role\":\"tool\"")
                && saved.contains("The provider stream ended before this call could run"),
            "{tag}: the refused call id must be answered on disk: {saved}"
        );
    }
}

/// Control for the refusals above: in the same unattended mode, that exact
/// call does run once the stream finishes. Without this, the refusal test
/// could pass for the wrong reason (a gate, or markup nothing recognized).
#[test]
fn a_finished_stream_still_runs_the_same_write_call() {
    for (tag, body) in [("native", WRITE_CALL_SSE), ("markup", WRITE_CALL_MARKUP_SSE)] {
        let (project, home) = fresh_dirs(&format!("finished-call-{tag}"));
        let (base_url, _requests, _server) = spawn_scripted_server(vec![
            (format!("{body}{TOOL_CALLS_TERMINATOR}"), true),
            (HELLO_SSE.to_string(), true),
        ]);
        write_settings_with_mode(&home, &base_url, "auto");

        let (code, lines, stdout) = json_turn(&project, &home, "write the file");
        assert_eq!(code, Some(0), "{tag}: {stdout}");
        assert!(
            lines.iter().any(|l| l["type"] == "tool_start" && l["name"] == "write_file"),
            "{tag}: the finished call must be dispatched: {stdout}"
        );
        assert_eq!(
            std::fs::read_to_string(project.join("side-effect.txt")).unwrap_or_default(),
            "written",
            "{tag}: the finished call must run"
        );
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
