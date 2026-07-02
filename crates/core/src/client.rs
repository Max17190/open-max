use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::types::{ChatMessage, ToolCall, ToolCallFunction};

/// Incremental output from a streaming completion.
pub enum StreamDelta {
    Content(String),
    Reasoning(String),
}

pub struct CompletionResult {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
}

/// Minimal client for any OpenAI-compatible /v1/chat/completions endpoint
/// (mlx-lm server, Ollama, LM Studio, vLLM, llama.cpp, cloud gateways...).
pub struct ChatClient {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: usize,
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
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            // No overall timeout: local generations can legitimately take minutes.
            .build()
            .expect("failed to build http client");
        Self { base_url, api_key, model, temperature, max_tokens, http }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Stream a chat completion, invoking `on_delta` for each token. Returns the
    /// fully accumulated message. If the server replies with plain JSON instead
    /// of an SSE stream, the response is parsed in one shot.
    pub async fn stream_chat(
        &self,
        messages: &[ChatMessage],
        tools: &Value,
        cancelled: Arc<AtomicBool>,
        mut on_delta: impl FnMut(StreamDelta),
    ) -> Result<CompletionResult, String> {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": true,
        });
        if tools.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            body["tools"] = tools.clone();
        }

        let mut req = self.http.post(self.endpoint()).json(&body);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
        }

        // Local models can spend a long time in prompt processing before the
        // first byte arrives; keep cancellation responsive throughout.
        let resp = tokio::select! {
            r = req.send() => r.map_err(|e| format!("request failed: {e}"))?,
            _ = wait_for(&cancelled) => {
                return Ok(CompletionResult { content: String::new(), tool_calls: Vec::new(), finish_reason: "cancelled".into() });
            }
        };
        let status = resp.status();
        let is_json = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("application/json"));

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("backend returned {status}: {}", truncate(&text, 600)));
        }

        // Some servers ignore `stream` and return a complete JSON body.
        if is_json {
            let v: Value = resp.json().await.map_err(|e| format!("bad JSON response: {e}"))?;
            return parse_complete_response(&v, &mut on_delta);
        }

        let mut content = String::new();
        let mut partials: Vec<PartialToolCall> = Vec::new();
        let mut finish_reason = String::from("stop");
        // Byte buffer: chunks can split multi-byte UTF-8 sequences, so text
        // conversion only happens on complete lines ('\n' is never part of a
        // multi-byte sequence).
        let mut buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();

        'outer: loop {
            let next = tokio::select! {
                c = stream.next() => c,
                _ = wait_for(&cancelled) => {
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
                    break 'outer;
                }
                let Ok(chunk) = serde_json::from_slice::<StreamChunk>(data) else { continue };
                let Some(choice) = chunk.choices.into_iter().next() else { continue };

                if let Some(reason) = choice.finish_reason {
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

        let tool_calls = finalize_tool_calls(partials);
        if !tool_calls.is_empty() && finish_reason == "stop" {
            finish_reason = "tool_calls".into();
        }
        Ok(CompletionResult { content, tool_calls, finish_reason })
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
    Ok(CompletionResult { content, tool_calls, finish_reason })
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

/// Resolves once `cancelled` becomes true; polled at 100ms.
async fn wait_for(cancelled: &AtomicBool) {
    loop {
        if cancelled.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_bytes_strips_whitespace() {
        assert_eq!(trim_bytes(b"  hello \r"), b"hello");
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
}
