//! The HTTP client for OpenAI-compatible chat completions.
//!
//! One streaming call, `stream_chat`, plus the compat knobs real gateways
//! need: `max_tokens` versus `max_completion_tokens`, optional
//! `stream_options`, and provider-specific headers. The request body is
//! serialized once before the retry loop so a retry resends identical bytes.
//!
//! Retries cover what can end one request before the reply exists: a
//! transport failure on send, a rate limit or transient server error (429,
//! 500, 502, 503, 504, 529), and a stream that died before any reply text
//! arrived. Each attempt resends the same bytes after an exponential backoff,
//! stretched to a server's Retry-After, and tells the caller through
//! [`StreamDelta::Retry`]. A 429 for an exhausted quota is reported at once:
//! no wait would let it succeed.
//! Once reply text has streamed, a retry would duplicate what the caller
//! already showed, so a failure after that point is reported as a
//! truncation instead. Reasoning deltas do not count: a retried attempt's
//! reasoning is void, and the result carries only the final attempt's.
//! A stream this client ends on purpose (the size cap, an out-of-range tool
//! index, cancellation) is never retried: the next attempt would end the
//! same way. A reply carrying only tool calls has no reply text, so a server
//! that never sends a completion signal costs the whole budget on such a
//! reply before its truncation is reported.
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
    messages: WireMessages<'a>,
    // Omitted unless configured: some models reject any value but their own
    // default, and a server fills in its default when the field is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
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

/// The transcript as one endpoint receives it, borrowed so a request never
/// copies it. Each message goes out as it serializes on disk, except that
/// reasoning goes only to the endpoint that produced it (`origin` matches
/// the message's `reasoning_origin`), and the stamp itself never goes.
struct WireMessages<'a> {
    messages: &'a [ChatMessage],
    origin: &'a str,
}

impl Serialize for WireMessages<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.messages.iter().map(|m| {
            let replay = m.reasoning_origin.as_deref() == Some(self.origin);
            WireMessage {
                role: &m.role,
                content: m.content.as_deref(),
                tool_calls: m.tool_calls.as_deref(),
                tool_call_id: m.tool_call_id.as_deref(),
                reasoning_content: m.reasoning_content.as_deref().filter(|_| replay),
                reasoning: m.reasoning.as_deref().filter(|_| replay),
            }
        }))
    }
}

/// One message on the wire. Fields, order, and skip rules follow
/// `ChatMessage`, so a message without reasoning keeps its exact bytes.
#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<&'a [ToolCall]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a str>,
}

/// Serialize the chat-completion request body once. Honors multi-provider
/// compat: `max_completion_tokens` vs `max_tokens`, optional `stream_options`,
/// and tools/`tool_choice` only when `tools_wire` is a non-empty JSON array.
///
/// `tools_wire` is the frozen registry schema string (exact bytes). It is
/// injected via `RawValue` so multi-iteration turns keep schema identity for
/// the server's KV cache without re-walking a `Value` tree. `origin` is the
/// receiving endpoint's `ChatClient::origin`, which decides whose reasoning
/// goes out.
// One argument per request field, as in `with_options`.
#[allow(clippy::too_many_arguments)]
fn serialize_chat_request_body(
    model: &str,
    messages: &[ChatMessage],
    origin: &str,
    temperature: Option<f32>,
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
        messages: WireMessages { messages, origin },
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
    /// The request is being resent: `attempt` is the one about to go out of
    /// the `max_attempts` budget, after `reason` ended the previous one. It
    /// is announced before the backoff wait, which is what the caller is
    /// waiting through; a cancellation during the wait ends the request
    /// instead, and the attempt never goes out. Any reasoning streamed for
    /// the failed attempt is void; no content was. A request gives up before
    /// the budget when [`UNREACHED_ATTEMPTS`] in a row never reached the
    /// endpoint.
    Retry { attempt: u32, max_attempts: u32, reason: String },
}

/// `finish_reason` for a stream the server never terminated: no `[DONE]` line
/// and no finish_reason chunk, just EOF part way through the answer. Reported
/// instead of `stop` so a cut-off reply cannot pass for a finished one.
pub const TRUNCATED: &str = "truncated";

// Bound wire input, including reasoning, unterminated lines, and tool arguments.
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOOL_CALLS: usize = 128;

pub struct CompletionResult {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// The reply's reasoning under the key the server sent it in, for the
    /// caller to put on the assistant message stamped with this client's
    /// `origin` (see `ChatMessage`). At most one is set, possibly to an empty
    /// string (`reasoning_fields` says which).
    pub reasoning_content: Option<String>,
    pub reasoning: Option<String>,
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
    pub temperature: Option<f32>,
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
    pub fn new(base_url: String, api_key: Option<String>, model: String, temperature: Option<f32>, max_tokens: usize) -> Self {
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
    fn with_options(
        base_url: String,
        api_key: Option<String>,
        model: String,
        temperature: Option<f32>,
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

    /// The stamp for reasoning this endpoint produces: the first 16 hex chars
    /// of sha256 over the whole route, base_url, model, credential, and
    /// headers. The model is part of it because a server can reject reasoning
    /// for one model it accepts for another; the credential and headers
    /// because two providers sharing a URL and model can still route to
    /// different backends or accounts, and reasoning one produced must not
    /// reach the other (encrypted reasoning only decrypts for the account
    /// that made it). Trailing slashes are dropped as `endpoint` drops them,
    /// since the request goes to the same URL. A hash, so a session file never
    /// holds the URL, the key, or a header value.
    pub fn origin(&self) -> String {
        let mut key = format!("{}\n{}\n{}", self.base_url.trim_end_matches('/'), self.model, self.api_key.as_deref().unwrap_or(""));
        for (name, value) in &self.headers {
            key.push_str(&format!("\n{}:{value}", name.to_ascii_lowercase()));
        }
        let mut origin = crate::ledger::sha256_hex(key.as_bytes());
        origin.truncate(16);
        origin
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
            &self.origin(),
            self.temperature,
            self.max_tokens,
            self.use_max_completion_tokens,
            self.send_stream_options,
            tools_wire,
        )?;

        let mut attempt = 0u32;
        // Attempts in a row that never reached the endpoint; any other
        // outcome, including a later transport fault, resets it.
        let mut unreached = 0u32;
        loop {
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
                _ = cancelled.cancelled() => return Ok(cancelled_response()),
            };
            let resp = match send_result {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("request failed: {}", describe_transport(&e));
                    unreached = if e.is_connect() || e.is_timeout() { unreached + 1 } else { 0 };
                    if attempt < MAX_ATTEMPTS && unreached < UNREACHED_ATTEMPTS && is_transient_transport(&e) {
                        if !retry_after(attempt, &msg, &cancelled, &mut on_delta).await {
                            return Ok(cancelled_response());
                        }
                        continue;
                    }
                    return Err(msg);
                }
            };
            unreached = 0;
            let status = resp.status();
            if !status.is_success() {
                let code = status.as_u16();
                let asked = retry_after_secs(resp.headers());
                let body = read_body(resp, &cancelled).await
                    .map_err(|e| format!("backend returned {status}: {e}"))?;
                let Some(body) = body else { return Ok(cancelled_response()); };
                let text = String::from_utf8_lossy(&body);
                let message = describe_backend(&text);
                let mut err = format!("backend returned {status}: {message}");
                if let Some(hint) = temperature_hint(self.temperature, &message) {
                    err.push_str(&hint);
                }
                if attempt < MAX_ATTEMPTS && is_retryable_status(code) && !quota_exhausted(&text) {
                    if !resend_after(attempt, &err, backoff(attempt, asked), &cancelled, &mut on_delta).await {
                        return Ok(cancelled_response());
                    }
                    continue;
                }
                return Err(err);
            }
            let is_json = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|ct| ct.contains("application/json"));

            // Some servers ignore `stream` and return a complete JSON body.
            if is_json {
                let Some(body) = read_body(resp, &cancelled).await? else { return Ok(cancelled_response()); };
                let v: Value = serde_json::from_slice(&body).map_err(|e| format!("bad JSON response: {e}"))?;
                return parse_complete_response(&v, &mut on_delta);
            }

            let (reply, interrupted) = read_sse(resp, &cancelled, &mut on_delta).await;
            match interrupted {
                Some(reason) if reply.content.is_empty() && attempt < MAX_ATTEMPTS => {
                    if !retry_after(attempt, &reason, &cancelled, &mut on_delta).await {
                        return Ok(cancelled_response());
                    }
                }
                _ => return Ok(reply),
            }
        }
    }
}

/// Parse one SSE stream to its end. The second value names the interruption
/// when the network ended the stream before the server did: EOF with no
/// terminator, or a read error. It is `None` for a finished reply and for a
/// stream this client cut itself, which a fresh attempt would cut the same way.
async fn read_sse(
    resp: reqwest::Response,
    cancelled: &crate::state::CancelToken,
    on_delta: &mut impl FnMut(StreamDelta),
) -> (CompletionResult, Option<String>) {
    let mut content = String::new();
    // One buffer per key, so the reply goes back under the key that carried
    // it. `None` until the server sends that key as a string, even an empty one.
    let mut reasoning_content: Option<String> = None;
    let mut reasoning: Option<String> = None;
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
    let mut received = 0usize;
    let mut scanned = 0usize;
    // Set when the network, not the server or this client, ended the
    // stream: the one truncation a fresh attempt can undo.
    let mut interrupted: Option<String> = None;

    'outer: loop {
        let next = tokio::select! {
            c = stream.next() => c,
            _ = cancelled.cancelled() => {
                finish_reason = "cancelled".into();
                break;
            }
        };
        let Some(chunk) = next else {
            if !saw_terminator {
                interrupted = Some("the stream ended before the reply finished".into());
            }
            break;
        };
        let chunk = match chunk {
            Ok(chunk) if chunk.len() <= MAX_RESPONSE_BYTES - received => chunk,
            Ok(_) => {
                saw_terminator = false;
                finish_reason = TRUNCATED.into();
                break;
            }
            Err(e) => {
                saw_terminator = false;
                finish_reason = TRUNCATED.into();
                // Every error this stream yields is the connection failing
                // under the body: a chunked or sized body cut short surfaces
                // here, not as a clean end, and this client applies no content
                // decoding that could fail on its own. So each one is worth a
                // fresh attempt.
                interrupted = Some(describe_transport(&e));
                break;
            }
        };
        received += chunk.len();
        buf.extend_from_slice(&chunk);

        while let Some(rel) = buf[scanned..].iter().position(|&b| b == b'\n') {
            let pos = scanned + rel;
            scanned = 0;
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
            if let Some(text) = &delta.reasoning_content {
                reasoning_content.get_or_insert_with(String::new).push_str(text);
            }
            if let Some(text) = &delta.reasoning {
                reasoning.get_or_insert_with(String::new).push_str(text);
            }
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
                    if idx >= MAX_TOOL_CALLS {
                        saw_terminator = false;
                        finish_reason = TRUNCATED.into();
                        break 'outer;
                    }
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
        scanned = buf.len();
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
    let (reasoning_content, reasoning) = reasoning_fields(reasoning_content, reasoning);
    (CompletionResult { content, tool_calls, reasoning_content, reasoning, finish_reason, usage }, interrupted)
}

/// Which key one reply's reasoning goes back under: the one the server sent
/// it in. Presence decides, not text: DeepSeek wants the key back on every
/// later assistant message, and an empty string satisfies it, so a key sent
/// as a string is kept even when empty; a null or absent key is not. A reply
/// that used both keys keeps the one carrying text, and `reasoning_content`
/// (the key the display path also prefers) when both or neither do. The
/// other key is dropped rather than merged, so the server gets its thinking
/// back once, under a key it emits. Neither key present sets neither, so the
/// message keeps its old shape.
fn reasoning_fields(
    reasoning_content: Option<String>,
    reasoning: Option<String>,
) -> (Option<String>, Option<String>) {
    let has_text = |key: &Option<String>| key.as_deref().is_some_and(|t| !t.is_empty());
    if has_text(&reasoning_content) || (reasoning_content.is_some() && !has_text(&reasoning)) {
        (reasoning_content, None)
    } else {
        (None, reasoning)
    }
}

fn cancelled_response() -> CompletionResult {
    CompletionResult {
        content: String::new(),
        tool_calls: Vec::new(),
        reasoning_content: None,
        reasoning: None,
        finish_reason: "cancelled".into(),
        usage: None,
    }
}

async fn read_body(
    resp: reqwest::Response,
    cancelled: &crate::state::CancelToken,
) -> Result<Option<Vec<u8>>, String> {
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    loop {
        let next = tokio::select! {
            _ = cancelled.cancelled() => return Ok(None),
            next = stream.next() => next,
        };
        let Some(chunk) = next else { return Ok(Some(body)); };
        let chunk = chunk.map_err(|e| format!("response body failed: {e}"))?;
        if chunk.len() > MAX_RESPONSE_BYTES - body.len() {
            return Err(format!("response exceeded {MAX_RESPONSE_BYTES} bytes"));
        }
        body.extend_from_slice(&chunk);
    }
}

/// Strip a leaked leading reasoning block from assistant content. Serving
/// layers normally split reasoning into `reasoning_content`, but when template
/// coverage lags a model the raw `<think>…</think>` block arrives in
/// `content`. Persisting it would re-prefill dead reasoning tokens on every
/// subsequent turn. Handles an unterminated block (stream cut mid-thought) by
/// treating the rest of the message as reasoning. Returns None when there is
/// nothing to strip.
pub(crate) fn strip_leading_think(content: &str) -> Option<String> {
    let trimmed = content.trim_start();
    for (open, close) in [("<think>", "</think>"), ("<thinking>", "</thinking>")] {
        if let Some(body) = trimmed.strip_prefix(open) {
            let rest = match body.find(close) {
                Some(i) => &body[i + close.len()..],
                None => "",
            };
            return Some(rest.trim_start().to_string());
        }
    }
    None
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
        if calls.len() > MAX_TOOL_CALLS {
            return Err(format!("response exceeded {MAX_TOOL_CALLS} tool calls"));
        }
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
    let text = |key: &str| msg[key].as_str().map(str::to_string);
    let (reasoning_content, reasoning) = reasoning_fields(text("reasoning_content"), text("reasoning"));
    Ok(CompletionResult { content, tool_calls, reasoning_content, reasoning, finish_reason, usage })
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

/// What to do when a server refuses the temperature it was sent. Versions
/// before temperature became optional wrote `"temperature": 0.2` into
/// settings.json on every save, so a user who never chose a value can still
/// be sending one, and reasoning models refuse every request that carries
/// it. Dropping a saved 0.2 on load would also discard a value someone did
/// choose, so the refusal names the setting and the fix instead. Only the
/// server's own error message is read, never the raw body: a backend that
/// echoes the request beside an unrelated error would otherwise blame a
/// valid setting for, say, an exhausted quota.
fn temperature_hint(sent: Option<f32>, message: &str) -> Option<String> {
    let sent = sent?;
    message.to_ascii_lowercase().contains("temperature").then(|| {
        format!(
            " (settings.json sets temperature {sent}; older versions wrote 0.2 there on every save, so if you did not choose it, delete the key and the server's default applies)"
        )
    })
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

/// Attempts per request. With [`BACKOFF_UNIT`] doubling up to
/// [`BACKOFF_CAP`], the last attempt goes out about a minute after the
/// first, later only when a server's Retry-After asks for longer waits
/// (see [`backoff`]). A transient fault on the path to an endpoint (a reset
/// or a TLS alert on send, a stream cut mid-reply) can recur for minutes and
/// clears within seconds most times and within a minute at worst, so
/// attempts packed into one second only ever observe the fault and end a
/// turn the next connection would have completed. Every wait is announced
/// and cancellable.
const MAX_ATTEMPTS: u32 = 8;
/// Attempts in a row that may fail before any connection exists (refused,
/// unresolvable, a connect timeout, a failed TLS handshake) before the
/// request gives up early. That is an address with nothing listening far
/// more often than a blip, and a user who forgot to start a local server
/// should hear so after the two short waits, not a minute. A refused
/// address fails in about three seconds; one that drops packets pays the
/// connect timeout on each attempt as well.
const UNREACHED_ATTEMPTS: u32 = 3;

/// One second: the step of the backoff and the unit a Retry-After header
/// counts in.
#[cfg(not(test))]
const BACKOFF_UNIT: std::time::Duration = std::time::Duration::from_secs(1);
/// Tests run every wait in milliseconds: the attempt count and which wait
/// applies, never seconds of wall clock.
#[cfg(test)]
const BACKOFF_UNIT: std::time::Duration = std::time::Duration::from_millis(1);
const BACKOFF_CAP: std::time::Duration = std::time::Duration::from_secs(16);
/// The longest wait a Retry-After header can ask for, in seconds: a minute,
/// the whole window of a per-minute rate limit. A longer ask is a limit that
/// resets hours away, and holding the turn that long helps no one; the
/// attempt goes out after the minute, announced and cancellable like every
/// other.
const RETRY_AFTER_CAP_SECS: u64 = 60;

/// Statuses that say the server could not take the request just now: a rate
/// limit (429), or an upstream that failed or shed load (500, 502, 503, 504,
/// and the 529 an overloaded provider sends). Resending is safe: a chat
/// completion changes nothing, tools run only once the agent loop holds a
/// finished reply, and a refused status carries no reply text the caller
/// could show twice. A 429 for an exhausted quota is excluded by
/// [`quota_exhausted`].
fn is_retryable_status(code: u16) -> bool {
    matches!(code, 429 | 500 | 502 | 503 | 504 | 529)
}

/// Whether an error body reports an exhausted quota (`insufficient_quota` as
/// the error's code or type) rather than a rate limit. Both arrive as 429,
/// but an account out of credit stays out on every attempt, and retrying
/// would only hold back the message that says so for the whole budget.
fn quota_exhausted(body: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(body) else { return false };
    let Some(error) = v.get("error") else { return false };
    ["code", "type"].iter().any(|key| error.get(key).and_then(Value::as_str) == Some("insufficient_quota"))
}

/// A Retry-After header in delta-seconds. The HTTP-date form is left to the
/// backoff: honoring it means trusting this clock to agree with the server's.
fn retry_after_secs(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?.trim().parse().ok()
}

/// The wait after `attempt` (1-based, the one that just failed): the
/// doubling backoff, or the server's Retry-After (`asked`, in seconds) when
/// that is longer, up to [`RETRY_AFTER_CAP_SECS`]. A shorter ask never
/// undercuts the backoff, so a server answering 0 to every client at once
/// cannot pull this one into a tight loop.
fn backoff(attempt: u32, asked: Option<u64>) -> std::time::Duration {
    let floor = BACKOFF_UNIT.saturating_mul(1u32 << attempt.saturating_sub(1).min(8)).min(BACKOFF_CAP);
    let asked = BACKOFF_UNIT.saturating_mul(asked.unwrap_or(0).min(RETRY_AFTER_CAP_SECS) as u32);
    floor.max(asked)
}

fn is_transient_transport(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout() || err.is_request()
}

/// Announce the retry, then wait out the backoff for `attempt` (1-based, the
/// one that just failed). Returns false when cancelled during the wait.
async fn retry_after(
    attempt: u32,
    reason: &str,
    cancelled: &crate::state::CancelToken,
    on_delta: &mut impl FnMut(StreamDelta),
) -> bool {
    resend_after(attempt, reason, backoff(attempt, None), cancelled, on_delta).await
}

/// [`retry_after`] with the wait chosen by the caller: a refused status
/// passes the server's Retry-After through [`backoff`]. Returns false when
/// cancelled during the wait: one can reach [`RETRY_AFTER_CAP_SECS`], and a
/// user who cancels must not sit through it.
async fn resend_after(
    attempt: u32,
    reason: &str,
    wait: std::time::Duration,
    cancelled: &crate::state::CancelToken,
    on_delta: &mut impl FnMut(StreamDelta),
) -> bool {
    on_delta(StreamDelta::Retry { attempt: attempt + 1, max_attempts: MAX_ATTEMPTS, reason: reason.to_string() });
    tokio::select! {
        _ = tokio::time::sleep(wait) => true,
        _ = cancelled.cancelled() => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn leading_think_block_is_stripped() {
        assert_eq!(
            strip_leading_think("<think>hmm, let me see</think>\nThe answer is 4.").as_deref(),
            Some("The answer is 4.")
        );
        assert_eq!(
            strip_leading_think("  <thinking>pondering</thinking>done").as_deref(),
            Some("done")
        );
    }

    #[test]
    fn unterminated_think_block_consumes_the_rest() {
        assert_eq!(strip_leading_think("<think>cut off mid-").as_deref(), Some(""));
    }

    #[test]
    fn think_tag_mid_message_is_left_alone() {
        assert!(strip_leading_think("The `<think>` tag is used by Qwen3.").is_none());
        assert!(strip_leading_think("plain answer").is_none());
    }

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

    /// A settings.json saved before temperature became optional still holds
    /// the old 0.2 default, which reasoning models refuse. The refusal must
    /// name the setting and the fix; nothing is said when no temperature was
    /// sent or the refusal is about something else.
    #[tokio::test]
    async fn a_refused_temperature_names_the_setting_to_remove() {
        let refusal = r#"{"error":{"message":"Unsupported value: 'temperature' does not support 0.2 with this model. Only the default (1) value is supported."}}"#;
        let refuse = || spawn_response_once(
            format!("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{refusal}", refusal.len()),
            None,
        );
        let err = ChatClient::new(refuse(), None, "m".into(), Some(0.2), 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |_| {})
            .await
            .err()
            .expect("a 400 is an error");
        assert!(err.contains("does not support 0.2"), "{err}");
        assert!(err.contains("settings.json sets temperature 0.2") && err.contains("delete the key"), "{err}");

        assert!(temperature_hint(None, refusal).is_none(), "nothing was sent, so nothing to remove");
        assert!(temperature_hint(Some(0.2), "model not found").is_none());

        // A backend that echoes the request beside an unrelated error says
        // nothing about temperature in its message: no advice to delete it.
        let echoed = r#"{"error":{"message":"You exceeded your current quota"},"request":{"model":"m","temperature":0.2}}"#;
        let refuse = || spawn_response_once(
            format!("HTTP/1.1 422 Unprocessable Entity\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{echoed}", echoed.len()),
            None,
        );
        let err = ChatClient::new(refuse(), None, "m".into(), Some(0.2), 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |_| {})
            .await
            .err()
            .expect("a 422 is an error");
        assert!(err.contains("exceeded your current quota"), "{err}");
        assert!(!err.contains("settings.json"), "{err}");
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
    fn spawn_sse_once(sse: &str) -> String {
        spawn_response_once(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse}"), None)
    }

    fn spawn_response_once(response: String, hold: Option<std::sync::mpsc::Receiver<()>>) -> String {
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
            let _ = stream.write_all(response.as_bytes());
            if let Some(hold) = hold { let _ = hold.recv(); }
            // Dropping the socket ends the body: the client sees a plain EOF.
        });
        format!("http://{addr}/v1")
    }

    async fn stream_once(sse: &str) -> CompletionResult {
        let client = ChatClient::new(spawn_sse_once(sse), None, "m".into(), None, 64);
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

    #[tokio::test]
    async fn response_bodies_observe_cancellation() {
        for status in ["200 OK", "400 Bad Request"] {
            let (release, hold) = std::sync::mpsc::channel();
            let url = spawn_response_once(format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: 10000\r\n\r\n{{"), Some(hold));
            let cancelled = Arc::new(crate::state::CancelToken::default());
            let task_cancel = cancelled.clone();
            let mut task = tokio::spawn(async move {
                ChatClient::new(url, None, "m".into(), None, 64)
                    .stream_chat(&[ChatMessage::user("hi")], "[]", task_cancel, |_| {}).await
            });
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            cancelled.cancel();
            let result = tokio::time::timeout(std::time::Duration::from_millis(200), &mut task).await;
            let _ = release.send(());
            assert_eq!(result.expect("body cancellation must finish promptly").unwrap().unwrap().finish_reason, "cancelled");
        }
    }

    #[tokio::test]
    async fn a_transport_error_keeps_partial_text() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"keep this\"}}]}\n\n";
        let url = spawn_response_once(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 10000\r\n\r\n{body}"), None);
        let result = ChatClient::new(url, None, "m".into(), None, 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |_| {}).await.unwrap();
        assert_eq!(result.content, "keep this");
        assert_eq!(result.finish_reason, TRUNCATED);
    }

    #[tokio::test]
    async fn an_oversized_stream_is_refused() {
        let body = format!("{}\ndata: [DONE]\n\n", "x".repeat(17 * 1024 * 1024));
        assert_eq!(stream_once(&body).await.finish_reason, TRUNCATED);
    }

    #[tokio::test]
    async fn an_out_of_bounds_tool_index_is_refused() {
        let body = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":128,\"id\":\"c1\",\"function\":{\"name\":\"bash\",\"arguments\":\"{}\"}}]}}]}\n\ndata: [DONE]\n\n";
        assert_eq!(stream_once(body).await.finish_reason, TRUNCATED);
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

    /// DeepSeek's thinking mode streams its reasoning as `reasoning_content`
    /// and answers the next request carrying tools with a 400 unless it comes
    /// back, so the result keeps the whole of it, under that key.
    #[tokio::test]
    async fn streamed_reasoning_content_is_kept_for_the_next_request() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me \"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"check\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ))
        .await;
        assert_eq!(result.reasoning_content.as_deref(), Some("let me check"));
        assert_eq!(result.reasoning, None);
        assert_eq!(result.content, "done");
    }

    /// Reasoning goes back under the key the server sent it in and no other:
    /// a strict server rejects a message property it never emits.
    #[tokio::test]
    async fn reasoning_goes_back_under_the_key_the_server_used() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning\":\"hmm\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
        ))
        .await;
        assert_eq!(result.reasoning.as_deref(), Some("hmm"));
        assert_eq!(result.reasoning_content, None);
        let origin = ChatClient::new("http://a/v1".into(), None, "m".into(), None, 64).origin();
        let mut reply = ChatMessage::assistant(Some(result.content), None);
        reply.reasoning_content = result.reasoning_content;
        reply.reasoning = result.reasoning;
        reply.reasoning_origin = Some(origin.clone());
        assert_eq!(
            wire(std::slice::from_ref(&reply), &origin),
            r#"[{"role":"assistant","content":"ok","reasoning":"hmm"}]"#
        );
        // It rides every later request, so the context budget counts it.
        assert!(reply.estimated_tokens() > ChatMessage::assistant(Some("ok".into()), None).estimated_tokens());

        // A server that uses both keys gets one back, the display's choice.
        let both = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"same\",\"reasoning\":\"same\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
        ))
        .await;
        assert_eq!(both.reasoning_content.as_deref(), Some("same"));
        assert_eq!(both.reasoning, None);
        // Unless only the other one carries text: the thinking is not lost.
        let both = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"\",\"reasoning\":\"real\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
        ))
        .await;
        assert_eq!(both.reasoning_content, None);
        assert_eq!(both.reasoning.as_deref(), Some("real"));
    }

    /// A server that ignores `stream` and answers in one JSON body keeps its
    /// reasoning too: the next request owes it back all the same.
    #[tokio::test]
    async fn a_one_shot_json_reply_keeps_its_reasoning() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"ok","reasoning_content":"thought"},"finish_reason":"stop"}]}"#;
        let url = spawn_response_once(
            format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()),
            None,
        );
        let result = ChatClient::new(url, None, "m".into(), None, 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |_| {}).await.unwrap();
        assert_eq!(result.content, "ok");
        assert_eq!(result.reasoning_content.as_deref(), Some("thought"));
        assert_eq!(result.reasoning, None);

        let reply = json!({"choices":[{"message":{"content":"ok","reasoning":"aside"}}]});
        let result = parse_complete_response(&reply, &mut |_| {}).unwrap();
        assert_eq!(result.reasoning.as_deref(), Some("aside"));
        assert_eq!(result.reasoning_content, None);
        let reply = json!({"choices":[{"message":{"content":"ok"}}]});
        let result = parse_complete_response(&reply, &mut |_| {}).unwrap();
        assert!(result.reasoning_content.is_none() && result.reasoning.is_none());
    }

    /// An endpoint that answers successive connections with successive
    /// bodies, each close-delimited, then stops listening. An empty body
    /// closes the connection after reading the request without answering:
    /// the shape of a transport fault on send. Returns the URL and the
    /// number of connections it served.
    fn spawn_sse_sequence(bodies: Vec<String>) -> (String, Arc<std::sync::Mutex<usize>>) {
        use std::io::{Read as _, Write as _};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let served = Arc::new(std::sync::Mutex::new(0usize));
        let count = served.clone();
        std::thread::spawn(move || {
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
                *count.lock().unwrap() += 1;
                if sse.is_empty() {
                    continue;
                }
                // A body prefixed with RAW: is the whole response, status
                // line and headers included (see `status_response`).
                if let Some(response) = sse.strip_prefix("RAW:") {
                    let _ = stream.write_all(response.as_bytes());
                    continue;
                }
                // A body prefixed with CHUNKED: is framed the way real servers
                // frame a stream, `Transfer-Encoding: chunked`, and the socket
                // closes after the first chunk with no terminating chunk: the
                // cut arrives as a body error, not as a clean end.
                if let Some(payload) = sse.strip_prefix("CHUNKED:") {
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{payload}\r\n",
                            payload.len()
                        )
                        .as_bytes(),
                    );
                    continue;
                }
                let _ = stream.write_all(
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{sse}").as_bytes(),
                );
            }
        });
        (format!("http://{addr}/v1"), served)
    }

    async fn stream_sequence(bodies: Vec<String>) -> (CompletionResult, Vec<String>, usize) {
        let (url, served) = spawn_sse_sequence(bodies);
        let mut deltas: Vec<String> = Vec::new();
        let result = ChatClient::new(url, None, "m".into(), None, 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |d| {
                deltas.push(match d {
                    StreamDelta::Content(t) => format!("content:{t}"),
                    StreamDelta::Reasoning(t) => format!("reasoning:{t}"),
                    StreamDelta::Retry { attempt, max_attempts, reason } => format!("retry:{attempt}/{max_attempts}:{reason}"),
                })
            })
            .await
            .unwrap();
        let served = *served.lock().unwrap();
        (result, deltas, served)
    }

    /// The bug this guards: a connection that died while the model was still
    /// reasoning ended the whole turn with the task half done, on a fault the
    /// next connection did not see. Nothing the caller keeps had arrived, so
    /// the client starts the reply over, announces the retry after the
    /// reasoning it voids, and the result is the finished reply alone.
    #[tokio::test]
    async fn a_stream_interrupted_before_any_reply_text_is_started_over() {
        let (result, deltas, served) = stream_sequence(vec![
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me\"},\"finish_reason\":null}]}\n\n".into(),
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            )
            .into(),
        ])
        .await;
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.content, "all of it");
        assert_eq!(served, 2);
        assert_eq!(
            deltas,
            vec![
                "reasoning:let me".to_string(),
                format!("retry:2/{MAX_ATTEMPTS}:the stream ended before the reply finished"),
                "content:all of it".to_string(),
            ]
        );
    }

    /// The same interruption as a real server delivers it. Streams are framed
    /// chunked, and a connection cut mid-body arrives as a read error rather
    /// than a clean end. The bug this guards: that arm never restarted the
    /// reply, so the fault it was written for still ended the turn.
    #[tokio::test]
    async fn a_stream_cut_mid_chunk_is_started_over() {
        let (result, deltas, served) = stream_sequence(vec![
            "CHUNKED:data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me\"},\"finish_reason\":null}]}\n\n".into(),
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            )
            .into(),
        ])
        .await;
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.content, "all of it");
        assert_eq!(served, 2, "the cut stream was started over on a second connection");
        assert_eq!(deltas.len(), 3, "{deltas:?}");
        assert_eq!(deltas[0], "reasoning:let me");
        assert!(deltas[1].starts_with(&format!("retry:2/{MAX_ATTEMPTS}:")), "{}", deltas[1]);
        assert_eq!(deltas[2], "content:all of it");
    }

    /// A connection the far side drops before answering is a transport fault
    /// after the connection existed: each one is resent after a backoff,
    /// announced, and the turn goes on with the reply that arrives.
    #[tokio::test]
    async fn a_request_dropped_on_send_is_resent_until_a_reply_arrives() {
        let good = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (result, deltas, served) =
            stream_sequence(vec![String::new(), String::new(), String::new(), good.into()]).await;
        assert_eq!(served, 4);
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.content, "all of it");
        assert_eq!(deltas.len(), 4);
        for (i, delta) in deltas.iter().take(3).enumerate() {
            assert!(
                delta.starts_with(&format!("retry:{}/{MAX_ATTEMPTS}:request failed: ", i + 2)),
                "resend {} is announced with its cause: {delta}",
                i + 2
            );
        }
        assert_eq!(deltas[3], "content:all of it");
    }

    /// Nothing listening is not a fault worth a minute of backoff: after
    /// [`UNREACHED_ATTEMPTS`] in a row the request gives up early. The port
    /// was bound and released by this test, so nothing answers on it; the
    /// deadline keeps a stray listener from turning that into a hang.
    #[tokio::test]
    async fn an_address_that_never_answers_gives_up_before_the_budget() {
        let url = {
            let released = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}/v1", released.local_addr().unwrap())
        };
        let mut deltas: Vec<String> = Vec::new();
        let client = ChatClient::new(url, None, "m".into(), None, 64);
        let messages = [ChatMessage::user("hi")];
        let request = client.stream_chat(&messages, "[]", Arc::new(crate::state::CancelToken::default()), |d| {
            if let StreamDelta::Retry { attempt, max_attempts, .. } = d {
                deltas.push(format!("{attempt}/{max_attempts}"));
            }
        });
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), request)
            .await
            .expect("a released port answers with a refusal, not silence");
        let Err(err) = result else { panic!("no listener is a failed request") };
        assert!(err.starts_with("request failed: "), "{err}");
        assert_eq!(deltas, vec![format!("2/{MAX_ATTEMPTS}"), format!("3/{MAX_ATTEMPTS}")]);
    }

    /// The client reports what arrived without hiding it: once the retry
    /// budget is spent, the calls come back alongside the truncation, and
    /// refusing to dispatch them is the agent loop's decision (a call from a
    /// stream the model never finished may not be the call it meant to make).
    #[tokio::test]
    async fn a_truncated_stream_still_returns_the_tool_calls_it_carried() {
        let body = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}]}}]}\n\n";
        let (result, deltas, served) = stream_sequence(vec![body.to_string(); MAX_ATTEMPTS as usize]).await;
        assert_eq!(served, MAX_ATTEMPTS as usize, "every attempt in the budget was spent first");
        assert_eq!(deltas.len(), MAX_ATTEMPTS as usize - 1, "each resend was announced");
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "read_file");
        assert_eq!(result.finish_reason, TRUNCATED, "an unfinished stream is not a clean tool_calls stop");
    }

    /// Reply text the caller has already shown is never duplicated: a stream
    /// that dies after content streamed is a truncation, not a retry.
    #[tokio::test]
    async fn a_stream_interrupted_after_reply_text_is_not_retried() {
        let (result, deltas, served) = stream_sequence(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"half an ans\"},\"finish_reason\":null}]}\n\n".into(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"unreached\"},\"finish_reason\":\"stop\"}]}\n\n".into(),
        ])
        .await;
        assert_eq!(served, 1);
        assert_eq!(result.finish_reason, TRUNCATED);
        assert_eq!(result.content, "half an ans");
        assert_eq!(deltas, vec!["content:half an ans".to_string()]);
    }

    /// The reasoning an interrupted attempt streamed is void once the reply
    /// starts over: the result carries the attempt that finished, never the
    /// two run together.
    #[tokio::test]
    async fn a_retried_reply_keeps_only_the_final_attempts_reasoning() {
        let (result, _, served) = stream_sequence(vec![
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"abandoned\"},\"finish_reason\":null}]}\n\n".into(),
            concat!(
                "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"fresh\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            )
            .into(),
        ])
        .await;
        assert_eq!(served, 2);
        assert_eq!(result.content, "all of it");
        assert_eq!(result.reasoning_content.as_deref(), Some("fresh"));
    }

    /// A refusal for [`spawn_sse_sequence`] to send verbatim: `status`, any
    /// extra `headers` (each ending in CRLF), and a JSON error body.
    fn status_response(status: &str, headers: &str, body: &str) -> String {
        format!(
            "RAW:HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    const FINISHED: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    /// The bug this guards: an overloaded or failing upstream ended the turn
    /// on a refusal the next attempt would not have seen. A refused status
    /// carries no reply text, and a chat completion has no side effect before
    /// the agent loop dispatches tools, so it is resent like a 429.
    #[tokio::test]
    async fn a_transient_server_error_is_resent_until_a_reply_arrives() {
        for status in ["500 Internal Server Error", "502 Bad Gateway", "503 Service Unavailable", "504 Gateway Timeout", "529 Overloaded"] {
            let refused = status_response(status, "", r#"{"error":{"message":"upstream overloaded"}}"#);
            let (result, deltas, served) = stream_sequence(vec![refused, FINISHED.into()]).await;
            assert_eq!(served, 2, "{status} was resent");
            assert_eq!(result.content, "all of it", "{status}");
            assert_eq!(deltas.len(), 2, "{status}: {deltas:?}");
            assert!(
                deltas[0].starts_with(&format!("retry:2/{MAX_ATTEMPTS}:backend returned {}", &status[..3]))
                    && deltas[0].ends_with("upstream overloaded"),
                "the resend names the refusal: {}",
                deltas[0]
            );
            assert_eq!(deltas[1], "content:all of it");
        }
    }

    /// A server that names its wait is waited for. Before, Retry-After was
    /// ignored and the resend went out after the bare backoff, into the same
    /// limit. The header counts seconds, which are [`BACKOFF_UNIT`]s outside
    /// tests; here a unit is a millisecond, so the wait stays measurable.
    #[tokio::test]
    async fn a_retry_after_header_sets_the_wait_before_the_resend() {
        let limited = status_response("429 Too Many Requests", "Retry-After: 40\r\n", r#"{"error":{"message":"rate limited"}}"#);
        let started = std::time::Instant::now();
        let (result, deltas, served) = stream_sequence(vec![limited, FINISHED.into()]).await;
        let waited = started.elapsed();
        assert_eq!(served, 2);
        assert_eq!(result.content, "all of it");
        assert_eq!(deltas[0], format!("retry:2/{MAX_ATTEMPTS}:backend returned 429 Too Many Requests: rate limited"));
        assert!(waited >= BACKOFF_UNIT * 40, "resent after {waited:?}, before the 40 units the server asked for");
    }

    /// An exhausted account also answers 429, but no wait makes it succeed:
    /// the bug this guards spent the whole retry budget, about a minute, on
    /// such a refusal before showing it. The server's message comes back
    /// after the one request.
    #[tokio::test]
    async fn an_exhausted_quota_is_reported_without_a_retry() {
        let exhausted = status_response(
            "429 Too Many Requests",
            "",
            r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#,
        );
        let (url, served) = spawn_sse_sequence(vec![exhausted, FINISHED.into()]);
        let mut retries = 0;
        let result = ChatClient::new(url, None, "m".into(), None, 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |d| {
                if let StreamDelta::Retry { .. } = d {
                    retries += 1;
                }
            })
            .await;
        let Err(err) = result else { panic!("an exhausted quota is an error, not a resend") };
        assert_eq!(
            err,
            "backend returned 429 Too Many Requests: You exceeded your current quota, please check your plan and billing details."
        );
        assert_eq!(retries, 0);
        assert_eq!(*served.lock().unwrap(), 1);
    }

    #[test]
    fn retryable_status_codes() {
        // A rate limit, or an upstream that cannot serve the request right now.
        for code in [429, 500, 502, 503, 504, 529] {
            assert!(is_retryable_status(code), "{code}");
        }
        // A refusal of the request itself: every attempt gets the same answer.
        for code in [400, 401, 403, 404, 413, 422, 501] {
            assert!(!is_retryable_status(code), "{code}");
        }
    }

    /// Retry-After stretches the backoff up to the cap and never shortens
    /// it; a value the header cannot hold as whole seconds is left to the
    /// backoff.
    #[test]
    fn retry_after_stretches_the_backoff_but_never_shortens_it() {
        assert_eq!(backoff(1, None), BACKOFF_UNIT);
        assert_eq!(backoff(4, None), BACKOFF_UNIT * 8);
        assert_eq!(backoff(1, Some(40)), BACKOFF_UNIT * 40);
        assert_eq!(backoff(1, Some(u64::MAX)), BACKOFF_UNIT * RETRY_AFTER_CAP_SECS as u32);
        assert_eq!(backoff(4, Some(0)), BACKOFF_UNIT * 8);
        assert_eq!(backoff(4, Some(3)), BACKOFF_UNIT * 8);

        let header = |value: &str| {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
            retry_after_secs(&headers)
        };
        assert_eq!(header("40"), Some(40));
        assert_eq!(header(" 7 "), Some(7));
        assert_eq!(header("99999999999999"), Some(99_999_999_999_999));
        assert_eq!(header("Wed, 21 Oct 2026 07:28:00 GMT"), None);
        assert_eq!(header("1.5"), None);
        assert_eq!(header("-1"), None);
        assert_eq!(retry_after_secs(&reqwest::header::HeaderMap::new()), None);
    }

    /// Only an exhausted quota stops the retry: a rate limit says when it
    /// lifts, and a body that is not an error envelope says nothing.
    #[test]
    fn only_an_exhausted_quota_is_not_worth_a_retry() {
        assert!(quota_exhausted(r#"{"error":{"message":"out of credit","type":"insufficient_quota","code":"insufficient_quota"}}"#));
        assert!(quota_exhausted(r#"{"error":{"message":"out of credit","code":"insufficient_quota"}}"#));
        assert!(quota_exhausted(r#"{"error":{"message":"out of credit","type":"insufficient_quota","code":null}}"#));
        assert!(!quota_exhausted(r#"{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}"#));
        assert!(!quota_exhausted("insufficient_quota"));
        assert!(!quota_exhausted(""));
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
            "o",
            Some(0.2),
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

    /// A message with no reasoning keeps the exact bytes it had before the
    /// reasoning fields existed, on the wire and on disk, so a server that
    /// never sends reasoning sees no change, and a session file from before
    /// them still loads. Reasoning withheld from another endpoint leaves the
    /// same bytes behind.
    #[test]
    fn a_message_without_reasoning_serializes_as_before() {
        let messages = vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant(Some("calling".into()), Some(vec![bash_call()])),
            ChatMessage::tool("c1", "out"),
        ];
        let old_shape = concat!(
            r#"[{"role":"user","content":"hi"},"#,
            r#"{"role":"assistant","content":"calling","tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]},"#,
            r#"{"role":"tool","content":"out","tool_call_id":"c1"}]"#,
        );
        assert_eq!(wire(&messages, "o"), old_shape);
        assert_eq!(serde_json::to_string(&messages).unwrap(), old_shape);
        let bytes = serialize_chat_request_body("m", &messages, "o", None, 64, false, false, "").unwrap();
        let body = String::from_utf8(bytes).unwrap();
        assert!(body.contains(&format!(r#""messages":{old_shape}"#)), "{body}");

        let mut foreign = messages.clone();
        foreign[1].reasoning_content = Some("thought".into());
        foreign[1].reasoning_origin = Some("elsewhere".into());
        assert_eq!(wire(&foreign, "o"), old_shape);

        let old: ChatMessage = serde_json::from_str(r#"{"role":"assistant","content":"hi"}"#).unwrap();
        assert!(old.reasoning_content.is_none() && old.reasoning.is_none() && old.reasoning_origin.is_none());
    }

    /// The bug this guards: a session that ran on DeepSeek and then moved to
    /// OpenAI or Groq with `/model` sent DeepSeek's reasoning along, and both
    /// answer a message property their model never emits with a 400. The
    /// reasoning goes back to the endpoint that produced it and nowhere else,
    /// and the stamp that decides it never leaves the session file.
    #[test]
    fn reasoning_goes_only_to_the_endpoint_that_produced_it() {
        let client = |base: &str, model: &str| ChatClient::new(base.into(), None, model.into(), None, 64);
        let deepseek = client("https://api.deepseek.com", "deepseek-reasoner");
        let mut reply = ChatMessage::assistant(None, Some(vec![bash_call()]));
        reply.reasoning_content = Some("thought".into());
        reply.reasoning_origin = Some(deepseek.origin());
        let messages = vec![ChatMessage::user("hi"), reply, ChatMessage::tool("c1", "out")];

        // The same endpoint, however its base_url ends, gets it back.
        let same = client("https://api.deepseek.com/", "deepseek-reasoner");
        assert_eq!(same.origin(), deepseek.origin());
        assert_eq!(deepseek.origin().len(), 16);
        let body = request_body(&same, &messages);
        assert!(body.contains(r#""tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}],"reasoning_content":"thought"}"#), "{body}");

        // A `/model` switch to another base_url, or to another model on the
        // same one, sends the transcript without it.
        for other in [client("https://api.openai.com/v1", "gpt-5"), client("https://api.deepseek.com", "deepseek-chat")] {
            assert_ne!(other.origin(), deepseek.origin());
            let body = request_body(&other, &messages);
            assert!(!body.contains("reasoning_content") && !body.contains(r#""reasoning""#), "{body}");
        }

        // Two providers on one URL and model are still two routes when the
        // credential or a routing header differs: neither gets the other's.
        let mut keyed = client("https://api.deepseek.com", "deepseek-reasoner");
        keyed.api_key = Some("sk-other-account".into());
        let mut routed = client("https://api.deepseek.com", "deepseek-reasoner");
        routed.headers = vec![("X-Route".into(), "backend-b".into())];
        for other in [keyed, routed] {
            assert_ne!(other.origin(), deepseek.origin());
            assert!(!request_body(&other, &messages).contains("reasoning_content"));
        }

        // Reasoning no endpoint stamped goes back to none.
        let mut unstamped = messages.clone();
        unstamped[1].reasoning_origin = None;
        assert!(!request_body(&deepseek, &unstamped).contains("reasoning_content"));

        // The stamp itself is never on the wire, though the session file keeps it.
        let body = request_body(&deepseek, &messages);
        assert!(!body.contains("reasoning_origin") && !body.contains(&deepseek.origin()), "{body}");
        let mut secret = client("https://api.deepseek.com", "deepseek-reasoner");
        secret.api_key = Some("sk-live-secret".into());
        assert!(!secret.origin().contains("sk-"), "the stamp is a hash, never the key");
        assert!(serde_json::to_string(&messages[1]).unwrap().contains(&deepseek.origin()));
    }

    /// DeepSeek wants the key back on every later assistant message, and an
    /// empty string satisfies it: a key the server sent as a string is kept
    /// and echoed even when it carried no text. A null key is no key.
    #[tokio::test]
    async fn an_empty_reasoning_key_is_echoed_as_an_empty_string() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"\",\"reasoning\":null},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
        ))
        .await;
        assert_eq!(result.reasoning_content.as_deref(), Some(""));
        assert_eq!(result.reasoning, None);
        let origin = ChatClient::new("http://a/v1".into(), None, "m".into(), None, 64).origin();
        let mut reply = ChatMessage::assistant(Some(result.content), None);
        reply.reasoning_content = result.reasoning_content;
        reply.reasoning_origin = Some(origin.clone());
        assert_eq!(wire(&[reply], &origin), r#"[{"role":"assistant","content":"ok","reasoning_content":""}]"#);

        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":null},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
        ))
        .await;
        assert!(result.reasoning_content.is_none() && result.reasoning.is_none());

        let reply = json!({"choices":[{"message":{"content":"ok","reasoning_content":""}}]});
        let result = parse_complete_response(&reply, &mut |_| {}).unwrap();
        assert_eq!(result.reasoning_content.as_deref(), Some(""));
        let reply = json!({"choices":[{"message":{"content":"ok","reasoning_content":null}}]});
        let result = parse_complete_response(&reply, &mut |_| {}).unwrap();
        assert!(result.reasoning_content.is_none() && result.reasoning.is_none());
    }

    fn bash_call() -> ToolCall {
        ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: "bash".into(), arguments: "{}".into() },
        }
    }

    /// The messages array exactly as `origin` would receive it.
    fn wire(messages: &[ChatMessage], origin: &str) -> String {
        serde_json::to_string(&WireMessages { messages, origin }).unwrap()
    }

    /// The whole request body `client` would send for `messages`.
    fn request_body(client: &ChatClient, messages: &[ChatMessage]) -> String {
        let bytes = serialize_chat_request_body(&client.model, messages, &client.origin(), None, 64, false, false, "").unwrap();
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn serialize_chat_request_body_omits_tools_when_empty() {
        let messages = vec![ChatMessage::user("hi")];
        for tools_wire in ["", "[]"] {
            let bytes =
                serialize_chat_request_body("m", &messages, "o", None, 64, false, true, tools_wire)
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
        let a = serialize_chat_request_body("m", &messages, "o", Some(0.5), 256, false, true, &tools_wire)
            .unwrap();
        let b = serialize_chat_request_body("m", &messages, "o", Some(0.5), 256, false, true, &tools_wire)
            .unwrap();
        assert_eq!(a, b);
    }

    /// Unset means absent, not a placeholder: a model that accepts only its
    /// own default refuses the request when the field is present at all.
    #[test]
    fn serialize_omits_temperature_unless_configured() {
        let messages = vec![ChatMessage::user("hi")];
        let unset = serialize_chat_request_body("m", &messages, "o", None, 64, false, true, "").unwrap();
        let v: Value = serde_json::from_slice(&unset).unwrap();
        assert!(v.get("temperature").is_none(), "{v}");
        let set = serialize_chat_request_body("m", &messages, "o", Some(0.7), 64, false, true, "").unwrap();
        let v: Value = serde_json::from_slice(&set).unwrap();
        assert!((v["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-5, "{v}");
    }

    #[test]
    fn serialize_respects_max_completion_tokens_and_omits_stream_options() {
        let messages = vec![ChatMessage::user("compat")];
        let bytes = serialize_chat_request_body(
            "gpt-style",
            &messages,
            "o",
            Some(0.1),
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
            serialize_chat_request_body("m", &messages, "o", None, 64, false, true, wire).unwrap();
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
