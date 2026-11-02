use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use open_max_core::{
    client::{ChatClient, StreamDelta},
    config::{ReasoningEffort, Settings},
    providers::{self, WireApi},
    state::CancelToken,
    types::ChatMessage,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("openmax-effort-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Request {
    path: String,
    body: Vec<u8>,
}

async fn serve(replies: Vec<(u16, String)>) -> (String, tokio::task::JoinHandle<Vec<Request>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let job = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(10), async move {
            let mut records = Vec::new();
            for (status, reply) in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut buffer = [0; 4096];
                    let count = socket.read(&mut buffer).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&buffer[..count]);
                    if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..end]);
                        let length: usize = head.lines().find_map(|line| {
                            line.to_ascii_lowercase().strip_prefix("content-length:")
                                .map(|value| value.trim().parse().unwrap())
                        }).unwrap();
                        if bytes.len() >= end + 4 + length {
                            records.push(Request {
                                path: head.split_whitespace().nth(1).unwrap().into(),
                                body: bytes[end + 4..end + 4 + length].to_vec(),
                            });
                            break;
                        }
                    }
                }
                socket.write_all(format!(
                    "HTTP/1.1 {status} Fixture\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len(),
                ).as_bytes()).await.unwrap();
            }
            records
        }).await.expect("request fixture did not finish")
    });
    (base, job)
}

fn completed(wire: WireApi) -> String {
    match wire {
        WireApi::ChatCompletions => format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({
                "choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]
            })
        ),
        WireApi::Responses => format!(
            "event: response.completed\ndata: {}\n\n",
            json!({
                "type":"response.completed","response":{"status":"completed","output":[{
                    "type":"message","role":"assistant","status":"completed",
                    "content":[{"type":"output_text","text":"ok"}]
                }]}
            })
        ),
    }
}

fn tools() -> String {
    json!([{"type":"function","function":{"name":"inspect","description":"Read a path",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},
            "required":["path"],"additionalProperties":false}}}])
    .to_string()
}

fn client(dir: &Path, base: &str, wire: WireApi, effort: Option<ReasoningEffort>) -> ChatClient {
    let mut settings = Settings {
        base_url: base.into(),
        api_key: Some("fixture-only".into()),
        model: "fixture-model".into(),
        context_tokens: Some(8192),
        max_tokens: 128,
        temperature: Some(0.25),
        reasoning_effort: effort,
        ..Settings::default()
    };
    if wire == WireApi::Responses {
        std::fs::write(
            dir.join("providers.json"),
            json!({"providers":{"fixture":{
                "base_url":base,"api_key":"fixture-only","wire_api":"responses"
            }}})
            .to_string(),
        )
        .unwrap();
        settings.provider = Some("fixture".into());
    }
    let endpoint = providers::resolve(&settings, dir).unwrap();
    assert_eq!(endpoint.reasoning_effort, effort);
    let client = ChatClient::from_endpoint(&endpoint);
    assert_eq!(client.reasoning_effort, effort);
    client
}

#[tokio::test]
async fn explicit_effort_uses_each_protocol_without_changing_other_request_fields() {
    for wire in [WireApi::ChatCompletions, WireApi::Responses] {
        let scratch = Scratch::new();
        let levels = [
            None,
            Some("none"),
            Some("minimal"),
            Some("low"),
            Some("medium"),
            Some("high"),
            Some("xhigh"),
            Some("max"),
        ];
        let (base, server) = serve(levels.iter().map(|_| (200, completed(wire))).collect()).await;
        let messages = [
            ChatMessage::system("fixture instructions"),
            ChatMessage::user("fixture task"),
        ];
        for level in levels {
            let effort = level.map(|value| serde_json::from_value(json!(value)).unwrap());
            let client = client(&scratch.0, &base, wire, effort);
            let result = tokio::time::timeout(
                Duration::from_secs(3),
                client.stream_chat(&messages, &tools(), Arc::new(CancelToken::default()), |_| {}),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(result.content, "ok");
        }
        let records = server.await.unwrap();
        let mut default = None;
        for (request, level) in records.into_iter().zip(levels) {
            let mut body: Value = serde_json::from_slice(&request.body).unwrap();
            assert_eq!(body["temperature"], 0.25);
            assert_eq!(body["model"], "fixture-model");
            match wire {
                WireApi::ChatCompletions => {
                    assert_eq!(request.path, "/chat/completions");
                    assert!(body.get("reasoning").is_none());
                    if let Some(level) = level {
                        assert_eq!(body["reasoning_effort"], level);
                    } else {
                        assert!(body.get("reasoning_effort").is_none());
                    }
                    body.as_object_mut().unwrap().remove("reasoning_effort");
                    assert_eq!(
                        body["messages"],
                        json!([
                            {"role":"system","content":"fixture instructions"},
                            {"role":"user","content":"fixture task"}
                        ])
                    );
                    assert_eq!(body["tools"], serde_json::from_str::<Value>(&tools()).unwrap());
                    assert_eq!(body["tool_choice"], "auto");
                }
                WireApi::Responses => {
                    assert_eq!(request.path, "/responses");
                    assert!(body.get("reasoning_effort").is_none());
                    if let Some(level) = level {
                        assert_eq!(body["reasoning"], json!({"effort":level}));
                    } else {
                        assert!(body.get("reasoning").is_none());
                    }
                    body.as_object_mut().unwrap().remove("reasoning");
                    assert_eq!(
                        body["input"][0],
                        json!({"role":"developer","content":"fixture instructions"})
                    );
                    assert_eq!(body["input"][1]["type"], "additional_tools");
                    assert_eq!(body["input"][1]["tools"][0]["name"], "inspect");
                    assert_eq!(body["input"][2], json!({"role":"user","content":"fixture task"}));
                }
            }
            if let Some(default) = &default {
                assert_eq!(&body, default);
            } else {
                default = Some(body);
            }
        }
    }
}

#[tokio::test]
async fn retry_preserves_the_exact_request_body_and_explicit_effort() {
    for wire in [WireApi::ChatCompletions, WireApi::Responses] {
        let scratch = Scratch::new();
        let (base, server) = serve(vec![
            (503, json!({"error":{"message":"temporarily unavailable"}}).to_string()),
            (200, completed(wire)),
        ])
        .await;
        let client = client(&scratch.0, &base, wire, Some(ReasoningEffort::High));
        let mut retries = 0;
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            client.stream_chat(
                &[ChatMessage::user("fixture task")],
                &tools(),
                Arc::new(CancelToken::default()),
                |delta| {
                    if matches!(delta, StreamDelta::Retry { .. }) {
                        retries += 1;
                    }
                },
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result.content, "ok");
        assert_eq!(retries, 1);
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].body, requests[1].body);
    }
}

#[tokio::test]
async fn incompatible_endpoint_refuses_explicit_effort_before_request_serialization() {
    let scratch = Scratch::new();
    let mut settings = Settings {
        base_url: "https://api.anthropic.com/v1".into(),
        api_key: Some("fixture-only".into()),
        model: "fixture-model".into(),
        context_tokens: Some(8192),
        ..Settings::default()
    };
    assert!(providers::resolve(&settings, &scratch.0).is_ok());
    settings.reasoning_effort = Some(ReasoningEffort::High);
    assert!(matches!(
        providers::resolve(&settings, &scratch.0),
        Err(providers::ResolveError::ReasoningEffort(_))
    ));
    let mut client = ChatClient::new(settings.base_url, settings.api_key, settings.model, None, 128);
    client.reasoning_effort = settings.reasoning_effort;
    // Invalid tools also fail before transport if the endpoint preflight regresses.
    let error = client
        .stream_chat(
            &[ChatMessage::user("fixture task")],
            "not-json",
            Arc::new(CancelToken::default()),
            |_| {},
        )
        .await
        .err()
        .unwrap();
    assert!(error.contains("reasoning"), "{error}");
}

#[tokio::test]
async fn custom_endpoint_rejections_keep_the_backend_error_without_changing_effort() {
    for wire in [WireApi::ChatCompletions, WireApi::Responses] {
        let scratch = Scratch::new();
        let message = "fixture model does not support the requested effort";
        let (base, server) = serve(vec![(400, json!({"error":{"message":message}}).to_string())]).await;
        let client = client(&scratch.0, &base, wire, Some(ReasoningEffort::High));
        let error = tokio::time::timeout(
            Duration::from_secs(3),
            client.stream_chat(
                &[ChatMessage::user("fixture task")],
                "[]",
                Arc::new(CancelToken::default()),
                |_| {},
            ),
        )
        .await
        .unwrap()
        .err()
        .unwrap();
        assert!(error.contains("400") && error.contains(message), "{error}");
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            if wire == WireApi::Responses {
                &body["reasoning"]["effort"]
            } else {
                &body["reasoning_effort"]
            },
            "high"
        );
    }
}
