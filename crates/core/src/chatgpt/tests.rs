use super::*;
use std::sync::{Arc, Mutex};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("openmax-oauth-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fields(query: &str) -> BTreeMap<String, String> {
    reqwest::Url::parse(&format!("http://127.0.0.1/?{query}")).unwrap().query_pairs().into_owned().collect()
}

// Artificial signing key used only by the local identity-provider fixture.
fn signed(claims: &Value) -> String {
    let key = signature::RsaKeyPair::from_pkcs8(include_bytes!("test-key.der")).unwrap();
    let input = format!("{}.{}", B64.encode(br#"{"alg":"RS256","kid":"fixture"}"#), B64.encode(claims.to_string()));
    let mut sig = vec![0; key.public().modulus_len()];
    key.sign(&signature::RSA_PKCS1_SHA256, &SystemRandom::new(), input.as_bytes(), &mut sig).unwrap();
    format!("{input}.{}", B64.encode(sig))
}

fn keys() -> Value {
    let hex = "C0A906F336EC6E65DA1306CA7D59739CC140EB5A5C6A093AAAC6CE4085D77EE0A9FA52D3CBA69CC2D4287CF98195CBE5E752AE142AC30591865E0F0644C83AC998582C5A302EF5488495757EF2216C134A4D005A04A3FE1BB65EC80366E2D301DCD1643CE029AF51B814EAB625F71887EFCF5D8EFB73DCF92E7A465723E491D44795F54A573EBD699B1A892D79119692D87FF1977ABF15231CE2CB7A6865895502777D8C57A7D4FEE9F47C89BDC768DD0F11F15D5CB569A710DA57E509237AF82C5A418F8B24DC02B30C314BF5183A157736B9253768BD1F4614CC5FC5EEBB3CC8099D243B0B671359C9B53C55307FB2B9A57AD2A81B912F10857BB8E6016043";
    let n: Vec<_> = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect();
    json!({"keys":[{"kty":"RSA","kid":"fixture","alg":"RS256","use":"sig","n":B64.encode(n),"e":"AQAB"}]})
}

fn claims(nonce: &str) -> Value {
    json!({"iss":ISSUER,"aud":"issued-client","sub":"user-1","iat":now(),"exp":now()+3600,"nonce":nonce,"email":"fixture@example.invalid"})
}

fn token_body(id: &str) -> Value {
    json!({"token_type":"Bearer","access_token":"new-access","refresh_token":"rotated-refresh",
        "id_token":id,"expires_in":3600,"scope":SCOPES})
}

async fn serve(
    count: usize,
    reply: impl Fn(&str, &str, &str) -> (u16, Value) + Send + 'static,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let origin = base.clone();
    let job = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(10), async move {
            for _ in 0..count {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (head, body) = loop {
                    let mut buf = [0; 4096];
                    let n = socket.read(&mut buf).await.unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buf[..n]);
                    if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
                        let length: usize = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break (head, String::from_utf8(bytes[end + 4..end + 4 + length].to_vec()).unwrap());
                        }
                    }
                };
                let path = head.split_whitespace().nth(1).unwrap();
                let (status, body) = reply(path, &body, &origin);
                let body = body.to_string();
                socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
        }).await.expect("identity-provider fixture did not finish");
    });
    (base, job)
}

pub(crate) fn seeded(dir: &Path) -> Connection {
    let tokens = Tokens {
        access: "old-access".into(),
        refresh: "old-refresh".into(),
        id: "old-id".into(),
        scopes: SCOPES.into(),
        expires: 0,
    };
    let account = Account {
        client_id: "issued-client".into(),
        subject: "user-1".into(),
        email: "fixture@example.invalid".into(),
        tokens: Some(tokens),
    };
    save(dir, &Store { host: "urn:uuid:fixture".into(), accounts: BTreeMap::from([("main".into(), account)]) })
        .unwrap();
    connection(dir, "main").unwrap()
}

#[tokio::test]
async fn logout_revokes_or_reports_failure_and_keeps_the_registration() {
    for succeeds in [true, false] {
        let scratch = Scratch::new();
        let auth = seeded(&scratch.0);
        // Keep the runtime ready so paused time cannot advance while loopback I/O is pending.
        let clock_guard = succeeds.then(|| tokio::spawn(async {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while std::time::Instant::now() < deadline {
                tokio::task::yield_now().await;
            }
            panic!("initial revocation did not run without a delay");
        }));
        let revocations = Arc::new(Mutex::new(None::<std::time::Instant>));
        let discovered = Arc::new(Mutex::new(None::<tokio::time::Instant>));
        let (base, server) = serve(if succeeds { 2 } else { 3 }, move |path, body, origin| match path {
            "/discovery" => {
                if succeeds {
                    tokio::time::pause();
                    *discovered.lock().unwrap() = Some(tokio::time::Instant::now());
                }
                (200, json!({"issuer":ISSUER,"revocation_endpoint":format!("{origin}/revoke")}))
            }
            "/revoke" => {
                if succeeds {
                    assert_eq!(tokio::time::Instant::now(), discovered.lock().unwrap().unwrap());
                    tokio::time::resume();
                }
                let mut last = revocations.lock().unwrap();
                if let Some(previous) = *last {
                    assert!(std::time::Instant::now().duration_since(previous) >= Duration::from_millis(250));
                }
                *last = Some(std::time::Instant::now());
                let form = fields(body);
                assert_eq!(form["token"], "old-refresh");
                assert_eq!(form["token_type_hint"], "refresh_token");
                assert_eq!(form["client_id"], "issued-client");
                (if succeeds { 200 } else { 503 }, json!({}))
            }
            _ => panic!("unexpected revocation request"),
        })
        .await;
        let message = logout_at(&scratch.0, "main", &format!("{base}/discovery")).await.unwrap();
        if let Some(clock_guard) = clock_guard {
            clock_guard.abort();
            assert!(clock_guard.await.unwrap_err().is_cancelled());
        }
        assert_eq!(message.contains("not confirmed"), !succeeds);
        let store = load(&scratch.0).unwrap();
        assert!(store.accounts["main"].tokens.is_none());
        assert_eq!(identity(&store.accounts["main"]), auth.identity);
        assert_eq!(store.host, "urn:uuid:fixture");
        assert!(logout(&scratch.0, "main").await.unwrap().contains("signed out"));
        server.await.unwrap();
    }
}

#[tokio::test]
async fn credential_lock_stops_retrying_at_its_deadline() {
    let scratch = Scratch::new();
    let _held = lock(&scratch.0).await.unwrap();
    tokio::time::pause();
    let mut waiting = Box::pin(lock(&scratch.0));
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    assert!(std::future::Future::poll(waiting.as_mut(), &mut context).is_pending());
    tokio::time::advance(Duration::from_secs(35)).await;
    match std::future::Future::poll(waiting.as_mut(), &mut context) {
        std::task::Poll::Ready(Err(error)) => assert!(error.contains("credentials are busy")),
        _ => panic!("credential lock retried past its deadline"),
    }
}

#[tokio::test]
async fn cancelling_a_refresh_waiter_does_not_discard_rotated_credentials() {
    let scratch = Scratch::new();
    let auth = seeded(&scratch.0);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (received_tx, received) = tokio::sync::oneshot::channel();
    let (release, release_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        loop {
            let mut buf = [0; 2048];
            let n = socket.read(&mut buf).await.unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&buf[..n]);
            if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        received_tx.send(()).unwrap();
        release_rx.await.unwrap();
        let body = token_body("renewed-id").to_string();
        socket
            .write_all(
                format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
                    .as_bytes(),
            )
            .await
            .unwrap();
    });
    // The harness's own client, with its deadlines: 10 seconds to connect and
    // 30 for a request. No wait below has a deadline of the test's own: a
    // refresh request that stalls ends at the client's, a waiter that ends
    // before its refresh reaches the server fails the test at once, and the
    // second refresh waits on the credential lock, which gives up after 35
    // seconds. 2-second deadlines here timed the runner instead, failing
    // beside every other test binary on CI.
    let client = http().unwrap();
    let (other, other_http, other_url) = (auth.clone(), client.clone(), url.clone());
    let mut waiter = tokio::spawn(async move { other.access_at(&other_http, &other_url).await });
    tokio::select! {
        reached = received => reached.unwrap(),
        ended = &mut waiter => panic!("the waiter ended before its refresh reached the server: {ended:?}"),
    }
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    release.send(()).unwrap();
    assert_eq!(auth.access_at(&client, &url).await.unwrap(), "new-access");
    assert_eq!(load(&scratch.0).unwrap().accounts["main"].tokens.as_ref().unwrap().refresh, "rotated-refresh");
    server.await.unwrap();
}

#[cfg(unix)]
#[test]
#[ignore = "child process for refresh_shutdown_waits_for_rotated_credentials"]
fn refresh_shutdown_child() {
    let dir = PathBuf::from(std::env::var_os("OPENMAX_REFRESH_TEST_DIR").unwrap());
    let url = std::env::var("OPENMAX_REFRESH_TEST_URL").unwrap();
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        let mut signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()).unwrap();
        let auth = seeded(&dir);
        let http = reqwest::Client::builder().no_proxy().build().unwrap();
        let waiter = tokio::spawn(async move { auth.access_at(&http, &url).await });
        signal.recv().await;
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        finish_refreshes().await;
        assert!(connection(&dir, "main").unwrap().access().await.unwrap_err().contains("shutting down"));
        std::process::exit(0);
    });
}

#[cfg(unix)]
#[tokio::test]
async fn refresh_shutdown_waits_for_rotated_credentials() {
    let scratch = Scratch::new();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "chatgpt::tests::refresh_shutdown_child", "--ignored"])
        .env("OPENMAX_REFRESH_TEST_DIR", &scratch.0)
        .env("OPENMAX_REFRESH_TEST_URL", url)
        .stdout(std::process::Stdio::null())
        .spawn().unwrap();
    let check = tokio::time::timeout(Duration::from_secs(5), async {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        loop {
            let mut buf = [0; 2048];
            let n = socket.read(&mut buf).await.unwrap();
            assert!(n > 0);
            bytes.extend_from_slice(&buf[..n]);
            if bytes.windows(b"old-refresh".len()).any(|w| w == b"old-refresh") { break; }
        }
        assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(child.try_wait().unwrap().is_none(), "exit discarded an in-flight rotation");
        let body = token_body("renewed-id").to_string();
        socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        loop {
            if let Some(status) = child.try_wait().unwrap() { assert!(status.success()); break; }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(load(&scratch.0).unwrap().accounts["main"].tokens.as_ref().unwrap().refresh, "rotated-refresh");
    }).await;
    let _ = child.kill();
    let _ = child.wait();
    check.expect("refresh shutdown did not settle");
}

#[cfg(unix)]
#[test]
fn credentials_refuse_links_and_readable_by_others() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let scratch = Scratch::new();
    seeded(&scratch.0);
    let path = scratch.0.join("chatgpt.json");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(connection(&scratch.0, "main").unwrap_err().contains("mode 600"));
    std::fs::rename(&path, scratch.0.join("moved")).unwrap();
    symlink(scratch.0.join("moved"), &path).unwrap();
    assert!(connection(&scratch.0, "main").unwrap_err().contains("cannot read"));
    assert!(auth_endpoint("https://wrong.invalid/revoke", DISCOVERY).is_err());
    assert!(auth_endpoint("http://auth.openai.com/revoke", DISCOVERY).is_err());
    assert!(auth_endpoint("https://auth.openai.com:8443/revoke", DISCOVERY).is_err());
    assert!(auth_endpoint("https://user@auth.openai.com/revoke", DISCOVERY).is_err());
    assert!(auth_endpoint("https://auth.openai.com/revoke#fragment", DISCOVERY).is_err());
}

#[test]
fn credentials_and_account_labels_enforce_size_and_path_boundaries() {
    let scratch = Scratch::new();
    for account in ["", "../main", "with space", &"x".repeat(65)] {
        assert!(connection(&scratch.0, account).unwrap_err().contains("account label"));
    }
    for account in ["valid-label", "valid_label", &"x".repeat(64)] {
        assert!(connection(&scratch.0, account).unwrap_err().contains("signed out"));
    }
    let mut store = Store::default();
    let limit = 1_048_576;
    store.host = "x".repeat(limit - serde_json::to_vec(&store).unwrap().len());
    save(&scratch.0, &store).unwrap();
    let exact = std::fs::read(scratch.0.join("chatgpt.json")).unwrap();
    assert_eq!(exact.len(), limit);
    assert_eq!(load(&scratch.0).unwrap().host, store.host);
    store.host.push('x');
    assert!(save(&scratch.0, &store).unwrap_err().contains("size limit"));
    assert_eq!(std::fs::read(scratch.0.join("chatgpt.json")).unwrap(), exact);
    let mut oversized = exact;
    oversized.push(b' ');
    std::fs::write(scratch.0.join("chatgpt.json"), oversized).unwrap();
    assert!(load(&scratch.0).err().unwrap().contains("size limit"));
}

#[tokio::test]
async fn browser_callback_exchange_and_returning_login_validate_and_preserve_registration() {
    let scratch = Scratch::new();
    let authorization = Arc::new(Mutex::new(BTreeMap::<String, String>::new()));
    let captured = authorization.clone();
    let exchanges = std::sync::atomic::AtomicUsize::new(0);
    let (base, server) = serve(9, move |path, body, origin| {
        (
            200,
            match path {
                "/token" => {
                    let form = fields(body);
                    let auth = captured.lock().unwrap();
                    assert_eq!(form["client_id"], "issued-client");
                    assert_eq!(form["grant_type"], "authorization_code");
                    assert_eq!(form["redirect_uri"], auth["redirect_uri"]);
                    assert_eq!(form["resource"], API);
                    assert_eq!(auth["code_challenge"], B64.encode(digest::digest(&digest::SHA256, form["code_verifier"].as_bytes())));
                    assert!(!form.contains_key("client_secret"));
                    {
                        let mut body = token_body(&signed(&claims(&auth["nonce"])));
                        if exchanges.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1 {
                            body["scope"] = json!("openid profile email offline_access");
                        }
                        body
                    }
                }
                "/discovery" => json!({"issuer":ISSUER,"jwks_uri":format!("{origin}/keys")}),
                "/keys" => keys(),
                _ => panic!("unexpected fixture path {path}"),
            },
        )
    })
    .await;
    let mut host = String::new();
    for step in 0..3 {
        let returning = step > 0;
        let callback_task = Arc::new(Mutex::new(None));
        let task = callback_task.clone();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            login_at(
                &scratch.0,
                "main",
                |url| {
                    let url = reqwest::Url::parse(url).unwrap();
                    let auth = fields(url.query().unwrap());
                    assert_eq!(auth["scope"], SCOPES);
                    assert_eq!(auth["code_challenge_method"], "S256");
                    assert_eq!(auth.get("prompt").map(String::as_str), if step == 2 { Some("consent") } else { None });
                    assert_eq!(auth["client_id"], if returning { "issued-client" } else { "dynamic_agent_client" });
                    if returning {
                        assert_eq!(host, auth["ext_agent_host_id"]);
                    } else {
                        host = auth["ext_agent_host_id"].clone();
                    }
                    *authorization.lock().unwrap() = auth.clone();
                    *task.lock().unwrap() = Some(tokio::spawn(async move {
                        let client = reqwest::Client::builder().no_proxy().build().unwrap();
                        let mut redirect = reqwest::Url::parse(&auth["redirect_uri"]).unwrap();
                        assert_eq!(redirect.host_str(), Some("127.0.0.1"));
                        if step == 0 {
                            let mut oversized = tokio::net::TcpStream::connect(("127.0.0.1", redirect.port().unwrap())).await.unwrap();
                            oversized.write_all(&vec![b'x'; 17000]).await.unwrap();
                            let mut reply = [0; 128];
                            let n = tokio::time::timeout(Duration::from_secs(1), oversized.read(&mut reply)).await.unwrap().unwrap();
                            assert!(String::from_utf8_lossy(&reply[..n]).starts_with("HTTP/1.1 400"));
                        }
                        redirect.query_pairs_mut().extend_pairs([
                            ("state", "wrong-state"),
                            ("code", "discarded"),
                            ("client_id", "issued-client"),
                        ]);
                        assert_eq!(client.get(redirect.clone()).send().await.unwrap().status(), 400);
                        redirect.set_query(None);
                        redirect.query_pairs_mut().extend_pairs([
                            ("state", auth["state"].as_str()),
                            ("code", "valid-code"),
                            ("client_id", "issued-client"),
                        ]);
                        if step == 0 {
                            let mut request = format!("GET {}?{} HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Padding: ", redirect.path(), redirect.query().unwrap());
                            request.push_str(&" ".repeat(16384 - request.len() - 4));
                            request.push_str("\r\n\r\n");
                            let mut socket = tokio::net::TcpStream::connect(("127.0.0.1", redirect.port().unwrap())).await.unwrap();
                            let mut reply = [0; 256];
                            let (head, ending) = request.as_bytes().split_at(request.len() - 4);
                            socket.write_all(head).await.unwrap();
                            assert!(tokio::time::timeout(Duration::from_millis(50), socket.read(&mut reply)).await.is_err());
                            socket.write_all(ending).await.unwrap();
                            let n = socket.read(&mut reply).await.unwrap();
                            assert!(String::from_utf8_lossy(&reply[..n]).starts_with("HTTP/1.1 200"));
                        } else {
                            assert_eq!(client.get(redirect).send().await.unwrap().status(), 200);
                        }
                    }));
                },
                &format!("{base}/token"),
                &format!("{base}/discovery"),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(result.message.contains("ChatGPT plan"));
        assert_eq!(result.plan_enabled, step != 1);
        if step == 1 {
            assert!(connection(&scratch.0, "main").unwrap_err().contains("disabled"));
            assert!(status(&scratch.0).unwrap()[0].contains("plan disabled"));
        }
        let task = callback_task.lock().unwrap().take().unwrap();
        task.await.unwrap();
    }
    server.await.unwrap();
    let store = load(&scratch.0).unwrap();
    assert_eq!(store.host, host);
    assert_eq!(store.accounts["main"].client_id, "issued-client");
    assert_eq!(store.accounts["main"].tokens.as_ref().unwrap().access, "new-access");
    assert!(!status(&scratch.0).unwrap().join(" ").contains("new-access"));
    assert!(!scratch.0.join("settings.json").exists());
    assert!(!scratch.0.join("sessions").exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(scratch.0.join("chatgpt.json")).unwrap().permissions().mode() & 0o777, 0o600);
    }
}

#[test]
fn invalid_identity_and_callback_cannot_select_an_account() {
    let valid = claims("nonce");
    assert!(validate_id(&signed(&valid), &keys(), "issued-client", "nonce").is_ok());
    for (field, value) in [
        ("iss", json!("https://wrong.invalid")),
        ("aud", json!("wrong-client")),
        ("nonce", json!("wrong")),
        ("exp", json!(now() - 1)),
        ("sub", json!("")),
        ("iat", json!(now() + 120)),
        ("iat", Value::Null),
        ("exp", Value::Null),
        ("nbf", json!(now() + 60)),
        ("nbf", json!("invalid")),
        ("azp", json!("wrong-client")),
        ("aud", json!(["issued-client", "other-client"])),
        ("aud", json!(["other-client"])),
    ] {
        let mut bad = valid.clone();
        bad[field] = value;
        assert!(
            validate_id(&signed(&bad), &keys(), "issued-client", "nonce").is_err(),
            "{field}"
        );
    }
    let mut multiple = valid.clone();
    multiple["aud"] = json!(["issued-client"]);
    assert!(validate_id(&signed(&multiple), &keys(), "issued-client", "nonce").is_ok());
    multiple["aud"] = json!(["issued-client", "other-client"]);
    multiple["azp"] = json!("issued-client");
    assert!(validate_id(&signed(&multiple), &keys(), "issued-client", "nonce").is_ok());
    let mut checked_clock_boundary = false;
    for _ in 0..3 {
        let second = now();
        let mut skewed = valid.clone();
        skewed["iat"] = json!(second + 60);
        skewed["nbf"] = json!(second);
        let result = validate_id(&signed(&skewed), &keys(), "issued-client", "nonce");
        if now() == second {
            assert!(result.is_ok(), "the permitted clock boundary was rejected");
            checked_clock_boundary = true;
            break;
        }
    }
    assert!(checked_clock_boundary, "clock changed during every validation attempt");
    for (field, value) in [
        ("kty", "EC"),
        ("kid", "unknown"),
        ("use", "enc"),
        ("alg", "RS512"),
    ] {
        let mut bad_keys = keys();
        bad_keys["keys"][0][field] = json!(value);
        assert!(
            validate_id(&signed(&valid), &bad_keys, "issued-client", "nonce").is_err(),
            "{field}"
        );
    }
    let mut tampered = signed(&valid).into_bytes();
    let last = tampered.len() - 10;
    tampered[last] = if tampered[last] == b'A' { b'B' } else { b'A' };
    assert!(
        validate_id(
            std::str::from_utf8(&tampered).unwrap(),
            &keys(),
            "issued-client",
            "nonce"
        )
        .is_err()
    );
    for query in [
        "state=x&code=c&client_id=issued-client",
        "state=s&state=s&code=c&client_id=issued-client",
        "state=s&code=c&code=other&client_id=issued-client",
        "state=s&code=c&client_id=issued-client&client_id=issued-client",
        "state=s&code=&client_id=issued-client",
        "state=s&code=c&client_id=dynamic_agent_client",
        "state=s&code=c&client_id=other-client",
        "state=s&error=access_denied",
    ] {
        assert!(
            callback(
                &format!("/auth/callback?{query}"),
                "s",
                Some("issued-client")
            )
            .is_err()
        );
    }
    assert!(callback("/auth/callback?state=s&code=c", "s", None).is_err());
    for client in ["", "dynamic_agent_client"] {
        assert!(callback(&format!("/auth/callback?state=s&code=c&client_id={client}"), "s", None).is_err());
    }
    assert_eq!(
        callback("/auth/callback?state=s&code=c", "s", Some("issued-client"))
            .unwrap()
            .1,
        "issued-client"
    );
}

#[tokio::test]
async fn failed_returning_login_preserves_the_signed_in_account() {
    for scenario in ["wrong-nonce", "changed-subject", "declined"] {
        let scratch = Scratch::new();
        seeded(&scratch.0);
        let before = std::fs::read(scratch.0.join("chatgpt.json")).unwrap();
        let authorization = Arc::new(Mutex::new(BTreeMap::<String, String>::new()));
        let captured = authorization.clone();
        let (base, server) = serve(if scenario == "declined" { 0 } else { 3 }, move |path, _, origin| {
            (
                200,
                match path {
                    "/token" => {
                        let auth = captured.lock().unwrap();
                        let mut identity = claims(if scenario == "changed-subject" {
                            &auth["nonce"]
                        } else {
                            "wrong-nonce"
                        });
                        if scenario == "changed-subject" {
                            identity["sub"] = json!("other-user");
                        }
                        token_body(&signed(&identity))
                    }
                    "/discovery" => json!({"issuer":ISSUER,"jwks_uri":format!("{origin}/keys")}),
                    "/keys" => keys(),
                    _ => panic!("unexpected identity validation request"),
                },
            )
        })
        .await;
        let callback_task = Arc::new(Mutex::new(None));
        let task = callback_task.clone();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            login_at(
                &scratch.0,
                "main",
                |url| {
                    let url = reqwest::Url::parse(url).unwrap();
                    let auth = fields(url.query().unwrap());
                    *authorization.lock().unwrap() = auth.clone();
                    *task.lock().unwrap() = Some(tokio::spawn(async move {
                        let mut redirect = reqwest::Url::parse(&auth["redirect_uri"]).unwrap();
                        redirect.query_pairs_mut().append_pair("state", &auth["state"]);
                        if scenario == "declined" {
                            redirect.query_pairs_mut().append_pair("error", "access_denied");
                        } else {
                            redirect.query_pairs_mut().append_pair("code", "valid-code");
                        }
                        assert_eq!(
                            reqwest::Client::builder()
                                .no_proxy()
                                .build()
                                .unwrap()
                                .get(redirect)
                                .send()
                                .await
                                .unwrap()
                                .status(),
                            if scenario == "declined" { 400 } else { 200 }
                        );
                    }));
                },
                &format!("{base}/token"),
                &format!("{base}/discovery"),
            ),
        )
        .await
        .unwrap();
        let error = result.err().expect("invalid identity was accepted");
        assert!(error.contains(match scenario {
            "changed-subject" => "differs from the selected account",
            "declined" => "authorization was declined",
            _ => "identity validation",
        }));
        assert_eq!(
            std::fs::read(scratch.0.join("chatgpt.json")).unwrap(),
            before
        );
        let task = callback_task.lock().unwrap().take().unwrap();
        task.await.unwrap();
        server.await.unwrap();
    }
}

#[tokio::test]
async fn refresh_without_plan_scope_persists_rotation_and_blocks_further_use() {
    let scratch = Scratch::new();
    let auth = seeded(&scratch.0);
    let (url, server) = serve(1, |_, _, _| {
        let mut body = token_body("renewed-id");
        body["scope"] = json!("openid profile email offline_access");
        (200, body)
    })
    .await;
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    assert_eq!(
        auth.access_at(&client, &url).await.unwrap_err(),
        PLAN_DISABLED
    );
    let store = load(&scratch.0).unwrap();
    let credentials = store.accounts["main"].tokens.as_ref().unwrap();
    assert_eq!(credentials.refresh, "rotated-refresh");
    assert_eq!(credentials.access, "new-access");
    server.await.unwrap();
    assert_eq!(
        auth.access_at(&client, &url).await.unwrap_err(),
        PLAN_DISABLED
    );
    assert_eq!(connection(&scratch.0, "main").unwrap_err(), PLAN_DISABLED);
}

#[tokio::test]
async fn concurrent_refresh_rotates_once_and_failure_retains_or_clears_as_required() {
    let scratch = Scratch::new();
    let auth = seeded(&scratch.0);
    let mut store = load(&scratch.0).unwrap();
    store.accounts.get_mut("main").unwrap().tokens.as_mut().unwrap().expires = now() + 60;
    save(&scratch.0, &store).unwrap();
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let (url, server) = serve(1, |_, body, _| {
        let form = fields(body);
        assert_eq!(form["refresh_token"], "old-refresh");
        assert_eq!(form["client_id"], "issued-client");
        assert_eq!(form["resource"], API);
        (200, token_body("renewed-id"))
    })
    .await;
    let (a, b, c) =
        tokio::join!(auth.access_at(&client, &url), auth.access_at(&client, &url), auth.access_at(&client, &url));
    for access in [a, b, c] {
        assert_eq!(access.unwrap(), "new-access");
    }
    assert_eq!(auth.access().await.unwrap(), "new-access");
    server.await.unwrap();
    assert_eq!(load(&scratch.0).unwrap().accounts["main"].tokens.as_ref().unwrap().refresh, "rotated-refresh");
    let expires = load(&scratch.0).unwrap().accounts["main"].tokens.as_ref().unwrap().expires;
    assert!(expires > SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
    let mut store = load(&scratch.0).unwrap();
    store.accounts.get_mut("main").unwrap().subject = "other-user".into();
    save(&scratch.0, &store).unwrap();
    assert!(auth.access().await.unwrap_err().contains("account changed"));
    for (status_code, code, retains) in [(503, "temporarily_unavailable", true), (400, "invalid_grant", false)] {
        let auth = seeded(&scratch.0);
        let (url, server) =
            serve(1, move |_, _, _| (status_code, json!({"error":code,"description":"secret-should-not-leak"}))).await;
        let error = auth.access_at(&client, &url).await.unwrap_err();
        assert!(!error.contains("secret-should-not-leak"));
        assert_eq!(load(&scratch.0).unwrap().accounts["main"].tokens.is_some(), retains);
        server.await.unwrap();
    }
}

#[tokio::test]
async fn invalid_refresh_responses_preserve_the_previous_credentials() {
    for scenario in ["zero-lifetime", "long-lifetime", "exact-limit", "oversized"] {
        let scratch = Scratch::new();
        let auth = seeded(&scratch.0);
        let before = std::fs::read(scratch.0.join("chatgpt.json")).unwrap();
        let (url, server) = serve(1, move |_, _, _| {
            let mut body = token_body("renewed-id");
            match scenario {
                "zero-lifetime" => body["expires_in"] = json!(0),
                "long-lifetime" => body["expires_in"] = json!(86401),
                "exact-limit" => {
                    body["padding"] = json!("");
                    body["padding"] = json!("x".repeat(LIMIT - body.to_string().len()));
                }
                _ => body["padding"] = json!("x".repeat(LIMIT)),
            }
            (200, body)
        }).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let result = auth.access_at(&client, &url).await;
        if scenario == "exact-limit" {
            assert_eq!(result.unwrap(), "new-access");
        } else {
            let error = result.unwrap_err();
            assert!(error.contains(if scenario == "oversized" { "size limit" } else { "invalid token lifetime" }));
            assert_eq!(std::fs::read(scratch.0.join("chatgpt.json")).unwrap(), before);
        }
        server.await.unwrap();
    }
}

#[tokio::test]
async fn catalog_creates_a_provider_and_preserves_configured_model_limits() {
    let scratch = Scratch::new();
    let catalog = json!({"models":[
        {"slug":"zero-window","visibility":"list","context_window":0},
        {"slug":"positive-window","visibility":"list","context_window":1},
        {"slug":"saved-window","visibility":"list"},
        {"slug":"empty-window","visibility":"list"}
    ]});
    let models = save_catalog(&scratch.0, "main", &catalog).await.unwrap();
    assert_eq!(models[0].context_tokens, None);
    assert_eq!(models[1].context_tokens, Some(1));
    let path = scratch.0.join("providers.json");
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let saved = &mut file["providers"]["chatgpt-main"]["models"];
    saved[2]["context_tokens"] = json!(4096);
    saved[2]["max_tokens"] = json!(1);
    saved[2]["compaction_tokens"] = json!(120_000);
    saved[3]["context_tokens"] = json!(0);
    saved[3]["max_tokens"] = json!(0);
    std::fs::write(&path, file.to_string()).unwrap();
    let models = save_catalog(&scratch.0, "main", &catalog).await.unwrap();
    assert_eq!(models[2].context_tokens, Some(4096));
    assert_eq!(models[2].max_tokens, Some(1));
    assert_eq!(models[2].compaction_tokens, Some(120_000));
    assert_eq!(models[3].context_tokens, None);
    assert_eq!(models[3].max_tokens, None);
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(save_catalog(&scratch.0, "main", &catalog).await.unwrap_err().contains("cannot read providers.json"));
}

#[tokio::test]
async fn catalog_and_provider_resolution_keep_credentials_and_defaults_separate() {
    use crate::{config::Settings, providers};
    let scratch = Scratch::new();
    seeded(&scratch.0);
    let unrelated = json!({"base_url":"http://127.0.0.1:9000","api_key":"separate-key","models":[{"id":"local"}]});
    std::fs::write(scratch.0.join("providers.json"), json!({"providers":{"local":unrelated}}).to_string()).unwrap();
    let catalog = json!({"models":[{"slug":"available-model","display_name":"Available","visibility":"list"},
        {"slug":"hidden-model","visibility":"hide"}]});
    let models = save_catalog(&scratch.0, "main", &catalog).await.unwrap();
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(scratch.0.join("providers.json")).unwrap().permissions().mode() & 0o777, 0o600);
    }
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].context_tokens, None);
    let bytes = std::fs::read_to_string(scratch.0.join("providers.json")).unwrap();
    assert!(!bytes.contains("old-access") && !bytes.contains("old-refresh"));
    let file: Value = serde_json::from_str(&bytes).unwrap();
    assert_eq!(file["providers"]["local"], unrelated);
    let mut settings = Settings {
        provider: Some("chatgpt-main".into()),
        model: "available-model".into(),
        api_key: Some("must-not-inherit".into()),
        context_tokens: Some(32768),
        ..Settings::default()
    };
    let endpoint = providers::resolve(&settings, &scratch.0).unwrap();
    assert!(endpoint.api_key.is_none());
    assert!(endpoint.chatgpt.is_some());
    let mut store = load(&scratch.0).unwrap();
    store.accounts.get_mut("main").unwrap().tokens.as_mut().unwrap().expires = now() + 3600;
    save(&scratch.0, &store).unwrap();
    for guard in ["headers", "api_key", "base_url", "wire_api"] {
        let mut endpoint = endpoint.clone();
        if guard == "wire_api" { endpoint.wire_api = providers::WireApi::ChatCompletions; }
        let mut client = crate::client::ChatClient::from_endpoint(&endpoint);
        match guard {
            "headers" => client.headers.push(("X-Test".into(), "fixture".into())),
            "api_key" => client.api_key = Some("fixture-key".into()),
            "base_url" => client.base_url = "https://wrong.invalid/v1".into(),
            _ => {}
        }
        let cancelled = Arc::new(crate::state::CancelToken::default());
        cancelled.cancel();
        let error = client.stream_chat(&[], client.reply_reserve, "[]", cancelled, |_| {}).await.err().unwrap();
        assert!(error.contains("fixed OpenAI endpoint"), "{guard}: {error}");
    }
    settings.provider = Some("local".into());
    let endpoint = providers::resolve(&settings, &scratch.0).unwrap();
    assert!(endpoint.chatgpt.is_none());
    assert_eq!(endpoint.api_key.as_deref(), Some("separate-key"));
    assert_eq!(endpoint.wire_api, providers::WireApi::ChatCompletions);
    let mut file = file;
    file["providers"]["chatgpt-main"]["models"][0]["context_tokens"] = json!(123456);
    std::fs::write(scratch.0.join("providers.json"), file.to_string()).unwrap();
    assert_eq!(save_catalog(&scratch.0, "main", &catalog).await.unwrap()[0].context_tokens, Some(123456));
    for (field, value) in [
        ("chatgpt_account", json!("other-account")),
        ("base_url", json!("https://wrong.invalid")),
        ("headers", json!({"X-Test":"test"})),
        ("api_key", json!("key")),
        ("api_key_env", json!("TEST_KEY")),
        ("wire_api", json!("chat_completions")),
    ] {
        let mut bad = file.clone();
        bad["providers"]["chatgpt-main"][field] = value;
        std::fs::write(scratch.0.join("providers.json"), bad.to_string()).unwrap();
        if matches!(field, "chatgpt_account" | "base_url" | "wire_api") {
            assert!(save_catalog(&scratch.0, "main", &catalog).await.is_err(), "{field}");
            let unchanged: Value = serde_json::from_slice(&std::fs::read(scratch.0.join("providers.json")).unwrap()).unwrap();
            assert_eq!(unchanged, bad);
        }
        settings.provider = Some("chatgpt-main".into());
        assert!(providers::resolve(&settings, &scratch.0).is_err(), "{field}");
    }
}
