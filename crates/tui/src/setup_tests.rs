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
    let catalog = discover(&base, "fixture-secret", headers.as_object().unwrap())
        .await
        .unwrap();
    assert_eq!(catalog[0]["max_input_tokens"], 65536);
    let request = task.await.unwrap().to_ascii_lowercase();
    assert!(request.starts_with("get /models http/1.1"));
    assert!(request.contains("authorization: bearer fixture-secret\r\n"));
    assert!(request.contains("anthropic-workspace-id: fixture-workspace\r\n"));
    assert!(request.contains("anthropic-version: 2023-06-01\r\n"));
    assert!(request.contains("x-organization: $fixture-org\r\n"));
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
        let error = discover(&base, "fixture-secret", &serde_json::Map::new())
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
    for body in [
        "fixture-secret".into(),
        json!({"data":[]}).to_string(),
        "x".repeat(LIMIT + 1),
    ] {
        let (base, task) = server(200, "", body).await;
        let error = discover(&base, "fixture-secret", &serde_json::Map::new())
            .await
            .unwrap_err();
        assert!(!error.contains("fixture-secret"));
        task.await.unwrap();
    }
    for status in [404, 405] {
        let (base, task) = server(status, "", String::new()).await;
        assert!(discover(&base, "fixture-secret", &serde_json::Map::new())
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
        let error = discover("http://127.0.0.1:9", "fixture-secret", &headers)
            .await
            .unwrap_err();
        assert!(error.contains("additional credentials"));
        assert!(!error.contains("fixture-old-secret"));
    }
    for base in [
        "http://example.com/v1",
        "https://user:fixture-secret@example.com/v1",
        "https://example.com/v1?key=fixture-secret",
        "file:///tmp/fixture-secret",
    ] {
        let error = discover(base, "fixture-secret", &serde_json::Map::new())
            .await
            .unwrap_err();
        assert!(!error.contains("fixture-secret"));
    }
}
