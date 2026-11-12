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
    for (native, explicit) in [(false, false), (false, true), (true, false), (true, true)] {
        let (base, task) = server(200, "", json!({"data":[{"id":"fixture-model"}]}).to_string()).await;
        let mut headers = serde_json::Map::new();
        if explicit { headers.insert("AnThRoPiC-VeRsIoN".into(), json!("fixture-version")); }
        discover(&base, "fixture-secret", &headers, false, native).await.unwrap();
        let request = task.await.unwrap();
        let versions: Vec<_> = request.lines().filter_map(|line| line.split_once(':'))
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-version"))
            .map(|(_, value)| value.trim()).collect();
        let expected: &[&str] = if explicit { &["fixture-version"] } else if native { &["2023-06-01"] } else { &[] };
        assert_eq!(versions, expected, "discovery must preserve an explicit version exactly once and default only native requests");
    }
    for native in [false, true] {
    let (base, task) = server(
        200,
        "",
        json!({"data":[{"id":"fixture-model", "max_input_tokens":65536}]}).to_string(),
    )
    .await;
    let headers = json!({"anthropic-version":"2023-06-01","anthropic-workspace-id":"fixture-workspace","x-organization":"$$fixture-org"});
    let catalog = discover(&base, "fixture-secret", headers.as_object().unwrap(), false, native)
        .await
        .unwrap();
    assert_eq!(catalog[0]["max_input_tokens"], 65536);
    let request = task.await.unwrap().to_ascii_lowercase();
    assert!(request.starts_with("get /models http/1.1"));
    assert_eq!(request.contains("authorization: bearer fixture-secret\r\n"), !native);
    assert_eq!(request.contains("x-api-key: fixture-secret\r\n"), native);
    assert!(request.contains("anthropic-workspace-id: fixture-workspace\r\n"));
    assert!(request.contains("anthropic-version: 2023-06-01\r\n"));
    assert!(request.contains("x-organization: $fixture-org\r\n"));

    }
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
        let catalog = discover(&base, "fixture-secret", &serde_json::Map::new(), openai, false).await.unwrap();
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
        let error = discover(&base, "fixture-secret", &serde_json::Map::new(), false, false)
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
        let error = discover(&base, "fixture-secret", &serde_json::Map::new(), false, false)
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
        let result = discover(&base, "fixture-secret", &serde_json::Map::new(), false, false).await;
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
    let catalog = discover(&base, "fixture-secret", &serde_json::Map::new(), false, false).await.unwrap();
    assert_eq!(catalog, vec![json!({"id":longest}), json!({"id":"fixture-model"})]);
    task.await.unwrap();
    for status in [404, 405] {
        let (base, task) = server(status, "", String::new()).await;
        assert!(discover(&base, "fixture-secret", &serde_json::Map::new(), false, false)
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
        let error = discover("http://127.0.0.1:9", "fixture-secret", &headers, false, false)
            .await
            .unwrap_err();
        assert!(error.contains("additional credentials"));
        assert!(!error.contains("fixture-old-secret"));
    }
    let headers = json!({"x-organization":42});
    assert!(discover("http://127.0.0.1:9", "fixture-secret", headers.as_object().unwrap(), false, false)
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

/// On a connection whose wire is not Chat Completions, setup saves the output
/// maximum a catalog reports, under each name a catalog uses for it, capped at
/// a quarter of the window because the limit is also reserved out of the input
/// budget, never under the default; it guesses nothing when the catalog is
/// silent and keeps a limit already set.
#[test]
fn setup_saves_the_reply_limit_a_catalog_reports() {
    let saved = |metadata: Value, window: u64| {
        let mut catalog = vec![json!({"id":"other","max_tokens":1})];
        record_model(&mut catalog, "m", window, &metadata, false);
        assert_eq!(catalog[0], json!({"id":"other","max_tokens":1}), "other models are untouched");
        assert_eq!(catalog[1]["context_tokens"], window);
        catalog[1].get("max_tokens").map(|n| n.as_u64().unwrap())
    };
    for metadata in [json!({"max_completion_tokens":32768}), json!({"max_output_tokens":32768}),
        json!({"max_tokens":32768}), json!({"top_provider":{"max_completion_tokens":32768}})] {
        assert_eq!(saved(metadata.clone(), 1_048_576), Some(32768), "the model's maximum is under a quarter of the window: {metadata}");
        assert_eq!(saved(metadata.clone(), 65536), Some(16384), "a quarter of the window is under the maximum: {metadata}");
        assert_eq!(saved(metadata.clone(), 8192), Some(4096), "never under the default: {metadata}");
    }
    assert_eq!(saved(json!({"max_tokens":2048}), 131072), Some(2048), "a model's own lower maximum is what it accepts");
    assert_eq!(saved(json!({"context_length":131072}), 131072), None, "a silent catalog gets no guess");
    assert_eq!(saved(json!({"max_tokens":0}), 131072), None);
    let mut catalog = vec![json!({"id":"m","context_tokens":1,"max_tokens":2048})];
    record_model(&mut catalog, "m", 131072, &json!({"max_tokens":65536}), false);
    assert_eq!(catalog, vec![json!({"id":"m","context_tokens":131072,"max_tokens":2048})], "a limit the user set is kept");
}

/// A Chat Completions connection continues a reply cut at its limit, so setup
/// saves the model's window and writes no reply limit when the catalog reports
/// an output maximum at or over the default; the default stays in force. A
/// maximum under the default is what the model accepts, so it is saved. A
/// `max_tokens` already on the entry, and every other model, is kept as it was.
#[test]
fn setup_saves_the_window_and_keeps_the_reply_limit_it_finds() {
    for most in [32768, 4096] {
        let mut catalog = vec![json!({"id":"other","max_tokens":1})];
        record_model(&mut catalog, "m", 131072, &json!({"max_completion_tokens":most}), true);
        assert_eq!(catalog, vec![json!({"id":"other","max_tokens":1}), json!({"id":"m","context_tokens":131072})]);
    }
    let mut catalog = Vec::new();
    record_model(&mut catalog, "m", 131072, &json!({"max_tokens":2048}), true);
    assert_eq!(catalog, vec![json!({"id":"m","context_tokens":131072,"max_tokens":2048})], "a lower maximum is what the model accepts");
    let mut catalog = vec![json!({"id":"m","context_tokens":1,"max_tokens":2048})];
    record_model(&mut catalog, "m", 131072, &json!({"max_tokens":65536}), true);
    assert_eq!(catalog, vec![json!({"id":"m","context_tokens":131072,"max_tokens":2048})], "a limit the user set is kept");
}

/// A Chat Completions model saved by setup from a catalog that reports a large
/// output maximum still runs at the default reply limit: the entry resolves to
/// it and a request carries it as `max_tokens`.
#[tokio::test]
async fn a_model_setup_saved_sends_the_default_reply_limit() {
    let (catalog_base, catalog_task) = server(200, "", json!({"data":[{"id":"fixture-model",
        "context_length":262144,"top_provider":{"max_completion_tokens":65536}}]}).to_string()).await;
    let models = discover(&catalog_base, "fixture-secret", &serde_json::Map::new(), false, false).await.unwrap();
    catalog_task.await.unwrap();
    let mut saved = Vec::new();
    record_model(&mut saved, "fixture-model", models[0]["context_length"].as_u64().unwrap(), &models[0], true);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let dir = crate::test_temp_dir("openmax-setup-reply-limit");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("providers.json"), json!({"providers":{"fixture":{"base_url":base,"models":saved}}}).to_string()).unwrap();
    let settings = open_max_core::config::Settings { provider: Some("fixture".into()), model: "fixture-model".into(), ..Default::default() };
    let endpoint = providers::resolve(&settings, &dir).unwrap();
    assert_eq!((endpoint.context_tokens, endpoint.max_tokens), (262144, 4096));
    let request = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        let body = loop {
            let n = socket.read(&mut buffer).await.unwrap();
            assert!(n > 0, "request ended early");
            request.extend_from_slice(&buffer[..n]);
            let text = String::from_utf8_lossy(&request).into_owned();
            let Some((head, body)) = text.split_once("\r\n\r\n") else { continue };
            let length = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()));
            if body.len() >= length.unwrap_or(0) { break body.to_owned(); }
        };
        let reply = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let _ = socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len()).as_bytes()).await;
        body
    });
    let client = open_max_core::client::ChatClient::from_endpoint(&endpoint);
    let messages = [open_max_core::types::ChatMessage::user("fixture task")];
    client.stream_chat(&messages, "[]", Default::default(), |_| {}).await.unwrap();
    let body: Value = serde_json::from_str(&request.await.unwrap()).unwrap();
    assert_eq!(body["max_tokens"], 4096);
    let _ = std::fs::remove_dir_all(dir);
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
        assert_eq!(new_entry["wire_api"], match name { "openai" => "responses", "anthropic" => "anthropic", _ => "chat_completions" });
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
