use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn server(
    status: u16,
    headers: &str,
    body: String,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let headers = headers.to_owned();
    let task = tokio::spawn(async move {
        tokio::time::timeout(std::time::Duration::from_secs(5), async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 2048];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut buffer).await.unwrap();
                assert!(n > 0);
                request.extend_from_slice(&buffer[..n]);
            }
            let response = format!("HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}", body.len());
            // A size-limit refusal may close the socket before the fixture finishes sending.
            let _ = socket.write_all(response.as_bytes()).await;
            String::from_utf8(request).unwrap()
        }).await.expect("model-discovery fixture did not finish")
    });
    (base, task)
}

#[tokio::test]
async fn model_discovery_sends_selected_credentials_and_preserves_reported_limits() {
    let (base, task) = server(
        200,
        "",
        json!({"data":[{"id":"fixture-model", "max_input_tokens":65536}]}).to_string(),
    )
    .await;
    let headers = json!({"anthropic-version":"2023-06-01","anthropic-workspace-id":"fixture-workspace","x-organization":"$$fixture-org"});
    let catalog = discover(&base, "fixture-secret", headers.as_object().unwrap(), false)
        .await
        .unwrap();
    assert_eq!(catalog[0]["max_input_tokens"], 65536);
    let request = task.await.unwrap().to_ascii_lowercase();
    assert!(request.starts_with("get /models http/1.1"));
    assert!(request.contains("authorization: bearer fixture-secret\r\n"));
    assert!(request.contains("anthropic-workspace-id: fixture-workspace\r\n"));
    assert!(request.contains("anthropic-version: 2023-06-01\r\n"));
    assert!(request.contains("x-organization: $fixture-org\r\n"));

    let general = ["gpt-4.1", "gpt-4o", "o3", "fixture-future-chat",
        "ft:gpt-4.1:fixture:custom:id", "gpt-realtimehelper"];
    let specialized = ["text-embedding-3-small", "text-embedding-3-large", "omni-moderation-latest",
        "text-moderation-latest", "tts-1", "tts-1-hd", "dall-e-3", "gpt-image-1",
        "gpt-realtime", "gpt-realtime-mini", "gpt-live-fixture", "gpt-live-transcribe-fixture",
        "gpt-transcribe-fixture", "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
        "gpt-4o-transcribe-diarize", "gpt-4o-mini-tts", "whisper-1",
        "ft:gpt-4o-mini-transcribe:fixture:specialized:id"];
    let all: Vec<_> = general.iter().chain(&specialized).copied().collect();
    let body = json!({"data":all.iter().map(|id| json!({"id":id,"context_window":65536})).collect::<Vec<_>>()}).to_string();
    for openai in [true, false] {
        let (base, task) = server(200, "", body.clone()).await;
        let catalog = discover(&base, "fixture-secret", &serde_json::Map::new(), openai).await.unwrap();
        let ids: Vec<_> = catalog.iter().map(|model| model["id"].as_str().unwrap()).collect();
        assert_eq!(ids, if openai { general.as_slice() } else { &all });
        assert!(catalog.iter().all(|model| model["context_window"] == 65536));
        task.await.unwrap();
    }
    let only_specialized = json!({"data":[{"id":"text-embedding-3-small"}]}).to_string();
    assert!(parse_catalog(only_specialized.as_bytes(), true).unwrap_err().contains("no available models"));
    assert_eq!(parse_catalog(only_specialized.as_bytes(), false).unwrap().len(), 1);
}

#[tokio::test]
async fn model_discovery_refuses_redirects_and_does_not_echo_error_bodies() {
    let trap = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let location = format!("Location: http://{}/stolen\r\n", trap.local_addr().unwrap());
    for status in [302, 401] {
        let (base, task) = server(
            status,
            &location,
            "fixture-secret must never be shown".into(),
        )
        .await;
        let error = discover(&base, "fixture-secret", &serde_json::Map::new(), false)
            .await
            .unwrap_err();
        assert!(error.contains(&format!("HTTP {status}")));
        assert!(!error.contains("fixture-secret"));
        task.await.unwrap();
    }
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(30), trap.accept())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn model_discovery_bounds_remote_data_and_rejects_invalid_or_empty_catalogs() {
    for (body, expected) in [
        ("fixture-secret".into(), "invalid model catalog"),
        (json!({"data":[]}).to_string(), "no available models"),
        (json!({"unexpected":[]}).to_string(), "no model catalog"),
        ("x".repeat(1_048_577), "exceeds its size limit"),
    ] {
        let (base, task) = server(200, "", body).await;
        let error = discover(&base, "fixture-secret", &serde_json::Map::new(), false)
            .await
            .unwrap_err();
        assert!(error.contains(expected), "{error}");
        assert!(!error.contains("fixture-secret"));
        task.await.unwrap();
    }
    // The inclusive byte limit accepts a valid catalog even when padded to its boundary.
    for length in [1_048_575, 1_048_576, 1_048_577] {
        let mut body = json!({"data":[{"id":"fixture-model"}]}).to_string();
        body.extend(std::iter::repeat_n(' ', length - body.len()));
        let (base, task) = server(200, "", body).await;
        let result = discover(&base, "fixture-secret", &serde_json::Map::new(), false).await;
        if length <= 1_048_576 {
            assert_eq!(result.unwrap()[0]["id"], "fixture-model");
        } else {
            assert!(result.unwrap_err().contains("exceeds its size limit"));
        }
        task.await.unwrap();
    }
    // Invalid IDs must never become selectable, while the exact length boundary is valid.
    let longest = "m".repeat(256);
    let body = json!({"data":[{"id":""},{"id":"m".repeat(257)},
        {"id":"fixture\nmodel"},{"id":42},{"id":longest},{"id":"fixture-model"}]}).to_string();
    let (base, task) = server(200, "", body).await;
    let catalog = discover(&base, "fixture-secret", &serde_json::Map::new(), false).await.unwrap();
    assert_eq!(catalog, vec![json!({"id":longest}), json!({"id":"fixture-model"})]);
    task.await.unwrap();
    for status in [404, 405] {
        let (base, task) = server(status, "", String::new()).await;
        assert!(discover(&base, "fixture-secret", &serde_json::Map::new(), false)
            .await
            .unwrap()
            .is_empty());
        task.await.unwrap();
    }
    for header in [
        "Cookie",
        "X-Auth-Token",
        "X-Secret",
        "X-Password",
        "X-Credential",
    ] {
        let mut headers = serde_json::Map::new();
        headers.insert(header.into(), json!("fixture-old-secret"));
        let error = discover("http://127.0.0.1:9", "fixture-secret", &headers, false)
            .await
            .unwrap_err();
        assert!(error.contains("additional credentials"));
        assert!(!error.contains("fixture-old-secret"));
    }
    let headers = json!({"x-organization":42});
    assert!(discover("http://127.0.0.1:9", "fixture-secret", headers.as_object().unwrap(), false)
        .await.unwrap_err().contains("headers must contain strings"));
    for base in [
        "http://example.com/v1",
        "https://user:fixture-secret@example.com/v1",
        "https://user@example.com/v1",
        "https://:fixture-secret@example.com/v1",
        "https://example.com/v1?key=fixture-secret",
        "https://example.com/v1#fixture-secret",
        "file:///tmp/fixture-secret",
        "ftp://localhost/v1",
    ] {
        let error = validate_url(base).unwrap_err();
        assert!(error.contains("use an HTTPS API URL"));
        assert!(!error.contains("fixture-secret"));
    }
    for base in ["https://example.com/v1", "http://localhost/v1", "http://127.0.0.1/v1", "http://[::1]/v1"] {
        validate_url(base).unwrap();
    }
}

#[test]
fn reconnecting_preserves_provider_compatibility_and_wire_protocol() {
    // These defaults belong to new presets, never to a credential rotation.
    for name in ["openai", "anthropic", "fixture"] {
        for compat in [None, Some(json!({"omit_temperature":false,"other":true})), Some(json!({"omit_temperature":true}))] {
            let mut entry = json!({"models":[{"id":"existing"}]});
            if let Some(compat) = compat { entry["compat"] = compat; }
            let original = entry.clone();
            api_defaults(&mut entry, name, false).unwrap();
            assert_eq!(entry, original, "reconnecting {name} changed implicit defaults");
            entry["wire_api"] = json!("chat_completions");
            let original = entry.clone();
            api_defaults(&mut entry, name, false).unwrap();
            assert_eq!(entry, original, "reconnecting {name} changed explicit settings");
        }
        let mut new_entry = json!({});
        api_defaults(&mut new_entry, name, true).unwrap();
        assert_eq!(new_entry["wire_api"], if name == "openai" { "responses" } else { "chat_completions" });
        assert_eq!(new_entry["compat"]["omit_temperature"], if name == "fixture" { Value::Null } else { json!(true) });
    }
}

#[cfg(unix)]
#[test]
fn forced_setup_exit_restores_terminal_input() {
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::process::CommandExt;
    use std::process::Command;
    const TEST: &str = "setup::tests::forced_setup_exit_restores_terminal_input";
    const ROLE: &str = "OPENMAX_SETUP_SIGNAL_PROBE";
    const FLAGS: libc::tcflag_t = libc::ECHO | libc::ICANON;
    fn mode() -> libc::tcflag_t {
        let mut term = std::mem::MaybeUninit::<libc::termios>::uninit();
        assert_eq!(unsafe { libc::tcgetattr(0, term.as_mut_ptr()) }, 0);
        unsafe { term.assume_init().c_lflag }
    }
    match std::env::var(ROLE).as_deref() {
        Ok(role @ ("worker" | "worker-tui")) => {
            let _quit = crate::watch_quit_signals(role == "worker-tui");
            let _input = InputMode::enter().unwrap();
            println!("SETUP_READY={}", std::process::id());
            loop { std::thread::park(); }
        }
        Ok("owner") => {
            let original = mode();
            let input = InputMode::enter().unwrap();
            assert_eq!(mode() & FLAGS, 0);
            drop(input);
            assert_eq!(mode() & FLAGS, original & FLAGS, "normal prompt drop left raw input");
            for role in ["worker", "worker-tui"] {
                let mut worker = Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", TEST, "--nocapture"]).env(ROLE, role)
                    .stdout(std::process::Stdio::piped()).spawn().unwrap();
                let mut ready = std::io::BufReader::new(worker.stdout.take().unwrap());
                loop {
                    let mut line = String::new();
                    assert!(std::io::BufRead::read_line(&mut ready, &mut line).unwrap() > 0);
                    if line.contains("SETUP_READY=") {
                        print!("{line}");
                        std::io::stdout().flush().unwrap();
                        break;
                    }
                }
                let status = worker.wait().unwrap();
                assert!(status.code().is_some_and(|code| [128 + libc::SIGTERM, 128 + libc::SIGUSR1].contains(&code)));
                assert_eq!(mode() & FLAGS, original & FLAGS, "forced setup exit left raw input");
            }
            return;
        }
        _ => {}
    }
    let (mut master, mut slave) = (-1, -1);
    assert_eq!(unsafe { libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) }, 0);
    let (mut master, slave) = unsafe { (std::fs::File::from_raw_fd(master), std::fs::File::from_raw_fd(slave)) };
    for fd in [master.as_raw_fd(), slave.as_raw_fd()] {
        assert_eq!(unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) }, 0);
    }
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args(["--exact", TEST, "--nocapture"]).env(ROLE, "owner")
        .stdin(slave.try_clone().unwrap()).stdout(slave.try_clone().unwrap()).stderr(slave);
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut owner = command.spawn().unwrap();
    drop(command);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let (mut output, mut worker) = (Vec::new(), None);
    let (mut status, mut closed) = (None, false);
    while std::time::Instant::now() < deadline {
        let mut fd = libc::pollfd { fd: master.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        if !closed && unsafe { libc::poll(&mut fd, 1, 10) } > 0 {
            let mut bytes = [0; 8192];
            match master.read(&mut bytes) {
                Ok(n) if n > 0 => output.extend_from_slice(&bytes[..n]),
                _ => closed = true,
            }
        }
        if closed { std::thread::sleep(std::time::Duration::from_millis(5)); }
        let text = String::from_utf8_lossy(&output);
        if let Some((_, line)) = text.rsplit_once("SETUP_READY=") {
            if let Some((pid, _)) = line.split_once('\n') {
                let pid = pid.trim().parse::<libc::pid_t>().unwrap();
                if worker != Some(pid) {
                    worker = Some(pid);
                    assert_eq!(unsafe { libc::kill(pid, libc::SIGTERM) }, 0);
                    assert_eq!(unsafe { libc::kill(pid, libc::SIGUSR1) }, 0);
                }
            }
        }
        status = owner.try_wait().unwrap();
        if status.is_some() {
            while !closed {
                if unsafe { libc::poll(&mut fd, 1, 0) } <= 0 { break; }
                let mut bytes = [0; 8192];
                match master.read(&mut bytes) {
                    Ok(n) if n > 0 => output.extend_from_slice(&bytes[..n]),
                    _ => closed = true,
                }
            }
            break;
        }
    }
    if status.is_none() {
        // The owner and its worker share this test's isolated process group.
        unsafe { libc::kill(-(owner.id() as libc::pid_t), libc::SIGKILL); }
        drop(master);
    }
    let _ = owner.wait();
    assert!(status.is_some_and(|status| status.success()), "terminal probe failed: {}", String::from_utf8_lossy(&output));
    assert_eq!(output.windows(b"\x1b[?2004l".len()).filter(|bytes| *bytes == b"\x1b[?2004l").count(), 3,
        "normal, forced CLI, and forced TUI exits must each disable bracketed paste");
}
