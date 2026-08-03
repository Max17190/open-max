//! The HTTP client for OpenAI-compatible chat completions.
//!
//! One streaming call, `stream_chat`, plus the compat knobs real gateways
//! need: `max_tokens` versus `max_completion_tokens`, optional
//! `stream_options`, and provider-specific headers. The request body is
//! serialized once before the retry loop so a retry resends identical bytes.
//!
//! Retries are deliberately narrow: connect failures, timeouts, and 429 only,
//! and only BEFORE the first token arrives. Once a stream has produced output,
//! retrying would duplicate text the caller already saw, so a mid-stream
//! failure is reported instead.
//!
//! There is no overall request timeout, only a connect timeout. A local or
//! slow endpoint can legitimately take minutes to generate, and a deadline
//! here would look like a bug in the model rather than a policy in the client.
//! A stream that ends without `[DONE]` and without a `finish_reason` is
//! reported as truncated rather than treated as a complete reply.

use std::sync::Arc;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;

use crate::types::{ChatMessage, ToolCall, ToolCallFunction};

/// OpenAI-compatible `/chat/completions` request body. Built once per
/// `stream_chat` call and serialized to bytes before the retry loop so retries
/// do not re-walk the transcript into a `serde_json::Value`.
///
/// Token limit and stream_options fields follow provider compat flags so
/// gateways that reject `max_tokens` or unknown `stream_options` stay happy.
/// Tools are injected as `RawValue` so a frozen registry wire form is embedded
/// byte-for-byte without re-serializing the tools array.
#[derive(Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<usize>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// Serialize the chat-completion request body once. Honors multi-provider
/// compat: `max_completion_tokens` vs `max_tokens`, optional `stream_options`,
/// and tools/`tool_choice` only when `tools_wire` is a non-empty JSON array.
///
/// `tools_wire` is the frozen registry schema string (exact bytes). It is
/// injected via `RawValue` so multi-iteration turns keep schema identity for
/// the server's KV cache without re-walking a `Value` tree.
fn serialize_chat_request_body(
    model: &str,
    messages: &[ChatMessage],
    temperature: f32,
    max_tokens: usize,
    use_max_completion_tokens: bool,
    send_stream_options: bool,
    tools_wire: &str,
) -> Result<Vec<u8>, String> {
    let include_tools = !tools_wire.is_empty() && tools_wire != "[]";
    // Borrow the frozen wire as RawValue: validation only, no Value rebuild.
    // Serialize writes these exact bytes into the body.
    let tools_raw: Option<&RawValue> = if include_tools {
        Some(
            serde_json::from_str(tools_wire)
                .map_err(|e| format!("invalid tools wire JSON: {e}"))?,
        )
    } else {
        None
    };
    let req = ChatCompletionRequest {
        model,
        messages,
        temperature,
        max_tokens: if use_max_completion_tokens {
            None
        } else {
            Some(max_tokens)
        },
        max_completion_tokens: if use_max_completion_tokens {
            Some(max_tokens)
        } else {
            None
        },
        stream: true,
        stream_options: if send_stream_options {
            Some(StreamOptions { include_usage: true })
        } else {
            None
        },
        tools: tools_raw,
        tool_choice: if include_tools { Some("auto") } else { None },
    };
    serde_json::to_vec(&req).map_err(|e| format!("failed to serialize chat request: {e}"))
}

/// Incremental output from a streaming completion.
pub enum StreamDelta {
    Content(String),
    Reasoning(String),
}

/// `finish_reason` for a stream the server never terminated: no `[DONE]` line
/// and no finish_reason chunk, just EOF part way through the answer. Reported
/// instead of `stop` so a cut-off reply cannot pass for a finished one.
pub const TRUNCATED: &str = "truncated";

pub struct CompletionResult {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// The server's reason, or `cancelled` (we stopped reading) or
    /// [`TRUNCATED`] (the server stopped writing without ever finishing).
    pub finish_reason: String,
    /// Server-reported token accounting, when the backend provides it.
    pub usage: Option<Usage>,
}

/// Ground-truth token usage from the server. `cached_tokens` is the number of
/// prompt tokens served from the prompt cache (servers report it under
/// `prompt_tokens_details`): if it stays near zero across turns, the harness
/// broke prefix stability and every step is paying a full re-prefill.
#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: Option<u64>,
}

/// Minimal client for any OpenAI-compatible /v1/chat/completions endpoint
/// (Ollama, LM Studio, vLLM, llama.cpp, cloud gateways, private proxies).
pub struct ChatClient {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: usize,
    pub headers: Vec<(String, String)>,
    pub use_max_completion_tokens: bool,
    pub send_stream_options: bool,
    http: reqwest::Client,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
    // Sent on the final chunk when the request asks for it via stream_options.
    usage: Option<UsageJson>,
}

#[derive(Deserialize)]
struct UsageJson {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize)]
struct PromptTokensDetails {
    cached_tokens: Option<u64>,
}

impl UsageJson {
    fn into_usage(self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_tokens.unwrap_or(0),
            completion_tokens: self.completion_tokens.unwrap_or(0),
            cached_tokens: self.prompt_tokens_details.and_then(|d| d.cached_tokens),
        }
    }
}

#[derive(Deserialize)]
struct StreamChoice {
    finish_reason: Option<String>,
    // Some servers omit `delta` entirely on the final finish_reason chunk.
    #[serde(default)]
    delta: StreamDeltaJson,
}

#[derive(Deserialize, Default)]
struct StreamDeltaJson {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    index: Option<u64>,
    id: Option<String>,
    function: Option<ToolCallFnDelta>,
}

#[derive(Deserialize)]
struct ToolCallFnDelta {
    name: Option<String>,
    arguments: Option<String>,
}

impl ChatClient {
    pub fn new(base_url: String, api_key: Option<String>, model: String, temperature: f32, max_tokens: usize) -> Self {
        Self::with_options(
            base_url,
            api_key,
            model,
            temperature,
            max_tokens,
            Vec::new(),
            false,
            true,
        )
    }

    /// Build a client from a resolved multi-provider endpoint.
    pub fn from_endpoint(ep: &crate::providers::ActiveEndpoint) -> Self {
        Self::with_options(
            ep.base_url.clone(),
            ep.api_key.clone(),
            ep.model.clone(),
            ep.temperature,
            ep.max_tokens,
            ep.headers.clone(),
            ep.compat.use_max_completion_tokens,
            ep.compat.send_stream_options,
        )
    }

    // One argument per endpoint option; the call sites build it straight from
    // a resolved Endpoint, so a builder would add ceremony without clarity.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        base_url: String,
        api_key: Option<String>,
        model: String,
        temperature: f32,
        max_tokens: usize,
        headers: Vec<(String, String)>,
        use_max_completion_tokens: bool,
        send_stream_options: bool,
    ) -> Self {
        // One client (and connection pool) for the process lifetime: ChatClient
        // is rebuilt every turn, and rebuilding the pool with it would redo the
        // TCP/TLS handshake per turn on remote endpoints.
        static HTTP: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
        let http = HTTP
            .get_or_init(|| {
                reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(10))
                    // No overall timeout: local generations can legitimately take minutes.
                    .build()
                    .expect("failed to build http client")
            })
            .clone();
        Self {
            base_url,
            api_key,
            model,
            temperature,
            max_tokens,
            headers,
            use_max_completion_tokens,
            send_stream_options,
            http,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Stream a chat completion, invoking `on_delta` for each token. Returns the
    /// fully accumulated message. If the server replies with plain JSON instead
    /// of an SSE stream, the response is parsed in one shot.
    ///
    /// `tools_wire` is the registry's frozen schema array JSON (`tool_schemas_wire`).
    /// Pass the same slice every iteration of a turn so the tools field stays
    /// byte-identical for prompt-cache stability.
    pub async fn stream_chat(
        &self,
        messages: &[ChatMessage],
        tools_wire: &str,
        cancelled: Arc<crate::state::CancelToken>,
        mut on_delta: impl FnMut(StreamDelta),
    ) -> Result<CompletionResult, String> {
        // Serialize once before retries: cloning bytes is cheap; re-walking a
        // long transcript into Value (and re-serializing) on every attempt is not.
        // Tools ride as RawValue from the frozen registry wire form.
        // Compat flags (max_completion_tokens / stream_options) are baked in.
        let body = serialize_chat_request_body(
            &self.model,
            messages,
            self.temperature,
            self.max_tokens,
            self.use_max_completion_tokens,
            self.send_stream_options,
            tools_wire,
        )?;

        // Retry only pre-stream transport failures (connect/timeout) and 429
        // (request rejected before work starts). Do not retry 502/503/504: a
        // proxy may already have forwarded the POST and started a completion,
        // so a second attempt can duplicate backend work with no idempotency key.
        // Once SSE bytes start, failures fail cleanly without a second prefill.
        const MAX_ATTEMPTS: u32 = 3;
        let mut attempt = 0u32;
        let resp = loop {
            attempt += 1;
            let mut req = self
                .http
                .post(self.endpoint())
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.clone());
            if let Some(key) = &self.api_key {
                if !key.is_empty() {
                    req = req.bearer_auth(key);
                }
            }
            for (name, value) in &self.headers {
                req = req.header(name.as_str(), value.as_str());
            }
            // An endpoint can spend a long time in prompt processing before the
            // first byte arrives, and a large prompt makes that worse wherever
            // it runs; keep cancellation responsive throughout.
            let send_result = tokio::select! {
                r = req.send() => r,
                _ = cancelled.cancelled() => {
                    return Ok(CompletionResult {
                        content: String::new(),
                        tool_calls: Vec::new(),
                        finish_reason: "cancelled".into(),
                        usage: None,
                    });
                }
            };
            let resp = match send_result {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("request failed: {}", describe_transport(&e));
                    if attempt < MAX_ATTEMPTS && is_transient_transport(&e) {
                        backoff_sleep(attempt).await;
                        if cancelled.is_cancelled() {
                            return Ok(CompletionResult {
                                content: String::new(),
                                tool_calls: Vec::new(),
                                finish_reason: "cancelled".into(),
                                usage: None,
                            });
                        }
                        continue;
                    }
                    return Err(msg);
                }
            };
            let status = resp.status();
            if status.is_success() {
                break resp;
            }
            let code = status.as_u16();
            let text = resp.text().await.unwrap_or_default();
            let err = format!("backend returned {status}: {}", describe_backend(&text));
            if attempt < MAX_ATTEMPTS && is_retryable_status(code) {
                backoff_sleep(attempt).await;
                if cancelled.is_cancelled() {
                    return Ok(CompletionResult {
                        content: String::new(),
                        tool_calls: Vec::new(),
                        finish_reason: "cancelled".into(),
                        usage: None,
                    });
                }
                continue;
            }
            return Err(err);
        };
        let is_json = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("application/json"));

        // Some servers ignore `stream` and return a complete JSON body.
        if is_json {
            let v: Value = resp.json().await.map_err(|e| format!("bad JSON response: {e}"))?;
            return parse_complete_response(&v, &mut on_delta);
        }

        let mut content = String::new();
        let mut partials: Vec<PartialToolCall> = Vec::new();
        let mut finish_reason = String::from("stop");
        // Did the server ever say it was done (a `[DONE]` line or a
        // finish_reason chunk)? Without one, the stream ending means the
        // connection dropped mid-answer, not that the model finished.
        let mut saw_terminator = false;
        let mut usage: Option<Usage> = None;
        // Byte buffer: chunks can split multi-byte UTF-8 sequences, so text
        // conversion only happens on complete lines ('\n' is never part of a
        // multi-byte sequence).
        let mut buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();

        'outer: loop {
            let next = tokio::select! {
                c = stream.next() => c,
                _ = cancelled.cancelled() => {
                    finish_reason = "cancelled".into();
                    break;
                }
            };
            let Some(chunk) = next else { break };
            let chunk = chunk.map_err(|e| format!("stream error: {e}"))?;
            buf.extend_from_slice(&chunk);

            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let rest = buf.split_off(pos + 1);
                let consumed = std::mem::replace(&mut buf, rest);
                let line = trim_bytes(&consumed[..pos]);
                if line.is_empty() || line.first() == Some(&b':') {
                    continue;
                }
                let data = strip_data_prefix(line);
                if data == b"[DONE]" {
                    saw_terminator = true;
                    break 'outer;
                }
                let Ok(chunk) = serde_json::from_slice::<StreamChunk>(data) else { continue };
                if let Some(u) = chunk.usage {
                    usage = Some(u.into_usage());
                }
                // The usage-bearing final chunk has an empty choices array.
                let Some(choice) = chunk.choices.into_iter().next() else { continue };

                if let Some(reason) = choice.finish_reason {
                    // Servers that end a stream here and never send `[DONE]`
                    // are still finished: this is a terminator too.
                    saw_terminator = true;
                    finish_reason = reason;
                }
                let delta = choice.delta;
                if let Some(text) = delta.content {
                    if !text.is_empty() {
                        content.push_str(&text);
                        on_delta(StreamDelta::Content(text));
                    }
                }
                // Reasoning models surface thinking under different keys.
                if let Some(text) = delta.reasoning_content {
                    if !text.is_empty() {
                        on_delta(StreamDelta::Reasoning(text));
                    }
                } else if let Some(text) = delta.reasoning {
                    if !text.is_empty() {
                        on_delta(StreamDelta::Reasoning(text));
                    }
                }
                if let Some(calls) = delta.tool_calls {
                    for tc in calls {
                        let idx = tc.index.unwrap_or(0) as usize;
                        while partials.len() <= idx {
                            partials.push(PartialToolCall::default());
                        }
                        if let Some(id) = tc.id {
                            partials[idx].id.push_str(&id);
                        }
                        if let Some(function) = tc.function {
                            if let Some(name) = function.name {
                                partials[idx].name.push_str(&name);
                            }
                            if let Some(args) = function.arguments {
                                partials[idx].arguments.push_str(&args);
                            }
                        }
                    }
                }
            }
        }

        // The stream ran out without the server ever finishing it: report the
        // truncation rather than the default "stop", which would make a
        // cut-off answer indistinguishable from a complete one. Cancellation
        // ends the stream from this side, so it keeps its own reason.
        if !saw_terminator && finish_reason != "cancelled" {
            finish_reason = TRUNCATED.into();
        }
        let tool_calls = finalize_tool_calls(partials);
        if !tool_calls.is_empty() && finish_reason == "stop" {
            finish_reason = "tool_calls".into();
        }
        Ok(CompletionResult { content, tool_calls, finish_reason, usage })
    }
}

fn trim_bytes(mut s: &[u8]) -> &[u8] {
    while s.first().is_some_and(|b| b.is_ascii_whitespace()) {
        s = &s[1..];
    }
    while s.last().is_some_and(|b| b.is_ascii_whitespace()) {
        s = &s[..s.len() - 1];
    }
    s
}

fn strip_data_prefix(line: &[u8]) -> &[u8] {
    const PREFIX: &[u8] = b"data:";
    if line.starts_with(PREFIX) {
        trim_bytes(&line[PREFIX.len()..])
    } else {
        line
    }
}

fn parse_complete_response(
    v: &Value,
    on_delta: &mut impl FnMut(StreamDelta),
) -> Result<CompletionResult, String> {
    let Some(choice) = v["choices"].get(0) else {
        return Err(format!("response had no choices: {}", truncate(&v.to_string(), 400)));
    };
    let msg = &choice["message"];
    let content = msg["content"].as_str().unwrap_or("").to_string();
    if !content.is_empty() {
        on_delta(StreamDelta::Content(content.clone()));
    }
    let mut partials = Vec::new();
    if let Some(calls) = msg["tool_calls"].as_array() {
        for tc in calls {
            partials.push(PartialToolCall {
                id: tc["id"].as_str().unwrap_or("").to_string(),
                name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: tc["function"]["arguments"].as_str().unwrap_or("").to_string(),
            });
        }
    }
    let tool_calls = finalize_tool_calls(partials);
    let finish_reason = choice["finish_reason"]
        .as_str()
        .unwrap_or(if tool_calls.is_empty() { "stop" } else { "tool_calls" })
        .to_string();
    let usage = serde_json::from_value::<UsageJson>(v["usage"].clone())
        .ok()
        .map(UsageJson::into_usage);
    Ok(CompletionResult { content, tool_calls, finish_reason, usage })
}

fn finalize_tool_calls(partials: Vec<PartialToolCall>) -> Vec<ToolCall> {
    partials
        .into_iter()
        .enumerate()
        .filter(|(_, p)| !p.name.is_empty())
        .map(|(i, p)| ToolCall {
            // Some local servers omit ids; synthesize one so tool replies can refer back.
            id: if p.id.is_empty() { format!("call_{i}") } else { p.id },
            kind: "function".into(),
            function: ToolCallFunction { name: p.name, arguments: p.arguments },
        })
        .collect()
}

/// reqwest's `Display` stops at "error sending request for url (...)" and
/// hides the cause underneath, which is the only part a user can act on:
/// "connection refused" means nothing is listening at base_url. Walk the
/// source chain so that line survives into the transcript.
fn describe_transport(e: &reqwest::Error) -> String {
    use std::error::Error;
    let mut out = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        let text = cause.to_string();
        // Wrappers often restate their child verbatim; keep the chain short.
        if !out.contains(&text) {
            out.push_str(": ");
            out.push_str(&text);
        }
        source = cause.source();
    }
    truncate(&out, 600)
}

/// OpenAI-compatible servers wrap failures in `{"error": {"message": ...}}`.
/// Show that message; fall back to the raw body for servers that do not.
fn describe_backend(body: &str) -> String {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message").or(Some(e)))
                .and_then(|m| m.as_str().map(str::to_string))
        })
        .map(|m| truncate(&m, 600))
        .unwrap_or_else(|| truncate(body, 600))
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

/// Only statuses that mean the server rejected the request before doing work.
/// 5xx from a proxy is not safe to retry without an idempotency key.
fn is_retryable_status(code: u16) -> bool {
    code == 429
}

fn is_transient_transport(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout() || err.is_request()
}

async fn backoff_sleep(attempt: u32) {
    // ~100ms, ~200ms (capped); keep total retry budget small for local UX.
    let ms = 100u64.saturating_mul(1u64 << (attempt.saturating_sub(1).min(2)));
    tokio::time::sleep(std::time::Duration::from_millis(ms.min(400))).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn trim_bytes_strips_whitespace() {
        assert_eq!(trim_bytes(b"  hello \r"), b"hello");
    }

    #[test]
    fn backend_errors_surface_the_message_not_the_envelope() {
        let body = r#"{"error":{"message":"model \"qwen\" not found","type":"not_found"}}"#;
        assert_eq!(describe_backend(body), "model \"qwen\" not found");
        // Servers that answer with plain text or HTML still show something.
        assert_eq!(describe_backend("upstream timeout"), "upstream timeout");
        assert_eq!(describe_backend(""), "");
    }

    #[test]
    fn backend_error_without_a_message_falls_back_to_the_error_value() {
        assert_eq!(
            describe_backend(r#"{"error":"quota exceeded"}"#),
            "quota exceeded"
        );
        // An object with no message is not a string; keep the raw body.
        let body = r#"{"error":{"code":42}}"#;
        assert_eq!(describe_backend(body), body);
    }

    #[test]
    fn strip_data_prefix_bytes() {
        assert_eq!(super::strip_data_prefix(b"data: {\"x\":1}"), b"{\"x\":1}");
        assert_eq!(super::strip_data_prefix(b"{\"x\":1}"), b"{\"x\":1}");
    }

    #[test]
    fn parse_sse_line_extracts_content() {
        let line = br#"data: {"choices":[{"delta":{"content":"hi"}}]}"#;
        let data = super::strip_data_prefix(trim_bytes(line));
        let chunk: StreamChunk = serde_json::from_slice(data).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hi"));
    }

    /// One-shot endpoint that answers with a close-delimited SSE body (no
    /// Content-Length, so the body ends at EOF) and then drops the connection.
    /// That is exactly what a provider dying mid-stream looks like on the wire:
    /// the transfer is well-formed, only the completion signal is missing.
    fn spawn_sse_once(sse: &'static str) -> String {
        use std::io::{Read as _, Write as _};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else { return };
            // Read headers, then the request body, so the client never sees a
            // reset while it is still writing.
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
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse}"
                )
                .as_bytes(),
            );
            // Dropping the socket ends the body: the client sees a plain EOF.
        });
        format!("http://{addr}/v1")
    }

    async fn stream_once(sse: &'static str) -> CompletionResult {
        let client = ChatClient::new(spawn_sse_once(sse), None, "m".into(), 0.0, 64);
        client
            .stream_chat(
                &[ChatMessage::user("hi")],
                "[]",
                Arc::new(crate::state::CancelToken::default()),
                |_| {},
            )
            .await
            .expect("a close-delimited body is not a transport error")
    }

    /// The bug this guards: a server that dies mid-answer sends neither
    /// `[DONE]` nor a finish_reason, and the partial reply used to come back
    /// as a normal "stop": a cut-off answer no client could tell from a
    /// finished one.
    #[tokio::test]
    async fn a_stream_that_ends_with_no_terminator_reports_truncation() {
        let result = stream_once(
            "data: {\"choices\":[{\"delta\":{\"content\":\"half an ans\"},\"finish_reason\":null}]}\n\n",
        )
        .await;
        assert_eq!(result.finish_reason, TRUNCATED);
        // The partial text still comes back: it lands in the transcript so the
        // session stays resumable.
        assert_eq!(result.content, "half an ans");
    }

    /// Plenty of servers close right after the finish_reason chunk and never
    /// send `[DONE]`. That is a finished answer, not a truncation.
    #[tokio::test]
    async fn an_explicit_finish_reason_is_a_clean_stop_without_done() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        ))
        .await;
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.content, "all of it");
    }

    #[tokio::test]
    async fn done_without_a_finish_reason_chunk_is_a_clean_stop() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"}}]}\n\n",
            "data: [DONE]\n\n",
        ))
        .await;
        assert_eq!(result.finish_reason, "stop");
    }

    /// The client reports what arrived without hiding it: the calls come back
    /// alongside the truncation, and refusing to dispatch them is the agent
    /// loop's decision (a call from a stream the model never finished may not
    /// be the call it meant to make).
    #[tokio::test]
    async fn a_truncated_stream_still_returns_the_tool_calls_it_carried() {
        let result = stream_once(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}]}}]}\n\n",
        )
        .await;
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "read_file");
        assert_eq!(result.finish_reason, TRUNCATED, "an unfinished stream is not a clean tool_calls stop");
    }

    #[test]
    fn retryable_status_codes() {
        assert!(is_retryable_status(429));
        // Proxy 5xx may have already started work; do not retry.
        assert!(!is_retryable_status(502));
        assert!(!is_retryable_status(503));
        assert!(!is_retryable_status(504));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(404));
        assert!(!is_retryable_status(500));
    }

    #[test]
    fn serialize_chat_request_body_includes_expected_fields_and_messages() {
        let messages = vec![
            ChatMessage::system("you are helpful"),
            ChatMessage::user("list files"),
            ChatMessage::assistant(Some("calling a tool".into()), None),
        ];
        let tools_wire = json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } }
                }
            }
        }])
        .to_string();
        // Default compat: max_tokens + stream_options (local OpenAI-compatible).
        let bytes = serialize_chat_request_body(
            "test-model",
            &messages,
            0.2,
            1024,
            false,
            true,
            &tools_wire,
        )
        .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(v["model"], "test-model");
        assert!((v["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-5);
        assert_eq!(v["max_tokens"], 1024);
        assert!(v.get("max_completion_tokens").is_none());
        assert_eq!(v["stream"], true);
        assert_eq!(v["stream_options"]["include_usage"], true);
        assert_eq!(v["tool_choice"], "auto");
        assert!(v["tools"].as_array().is_some_and(|a| a.len() == 1));
        assert_eq!(v["tools"][0]["function"]["name"], "read_file");
        // RawValue must embed the frozen wire bytes exactly.
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(
            body.contains(&tools_wire),
            "body must contain exact tools wire substring"
        );

        let msgs = v["messages"].as_array().expect("messages array");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "you are helpful");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "list files");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "calling a tool");
    }

    #[test]
    fn serialize_chat_request_body_omits_tools_when_empty() {
        let messages = vec![ChatMessage::user("hi")];
        for tools_wire in ["", "[]"] {
            let bytes =
                serialize_chat_request_body("m", &messages, 0.0, 64, false, true, tools_wire)
                    .unwrap();
            let v: Value = serde_json::from_slice(&bytes).unwrap();
            assert!(
                v.get("tools").is_none(),
                "tools should be omitted for {tools_wire:?}"
            );
            assert!(
                v.get("tool_choice").is_none(),
                "tool_choice should be omitted for {tools_wire:?}"
            );
            assert_eq!(v["messages"][0]["content"], "hi");
            assert_eq!(v["stream"], true);
            assert_eq!(v["stream_options"]["include_usage"], true);
        }
    }

    #[test]
    fn serialize_chat_request_body_is_byte_stable_for_retries() {
        let messages = vec![ChatMessage::user("stable body")];
        let tools_wire = json!([{
            "type": "function",
            "function": { "name": "bash", "parameters": { "type": "object" } }
        }])
        .to_string();
        let a = serialize_chat_request_body("m", &messages, 0.5, 256, false, true, &tools_wire)
            .unwrap();
        let b = serialize_chat_request_body("m", &messages, 0.5, 256, false, true, &tools_wire)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn serialize_respects_max_completion_tokens_and_omits_stream_options() {
        let messages = vec![ChatMessage::user("compat")];
        let bytes = serialize_chat_request_body(
            "gpt-style",
            &messages,
            0.1,
            512,
            true,  // use_max_completion_tokens
            false, // send_stream_options
            "",    // no tools
        )
        .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["max_completion_tokens"], 512);
        assert!(v.get("max_tokens").is_none());
        assert!(v.get("stream_options").is_none());
        assert_eq!(v["stream"], true);
        assert!(v.get("tools").is_none());
    }

    #[test]
    fn registry_tools_wire_embeds_exact_substring_in_body() {
        let registry = crate::registry::Registry::builtin_only();
        assert_eq!(
            registry.tool_schemas_wire(),
            registry.tool_schemas_json().to_string()
        );
        let messages = vec![ChatMessage::user("hi")];
        let wire = registry.tool_schemas_wire();
        let bytes =
            serialize_chat_request_body("m", &messages, 0.0, 64, false, true, wire).unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(
            body.contains(wire),
            "HTTP body must embed the frozen tools wire bytes exactly"
        );
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["tool_choice"], "auto");
        assert!(v["tools"].as_array().is_some_and(|a| !a.is_empty()));
    }
}
