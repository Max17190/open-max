//! The HTTP client for OpenAI-compatible chat completions.
//!
//! One streaming call, `stream_chat`, plus the compat knobs real gateways
//! need: `max_tokens` versus `max_completion_tokens`, optional
//! `stream_options`, and provider-specific headers. The request body is
//! serialized once before the retry loop so a retry resends identical bytes.
//!
//! Retries cover what can end one request before the reply exists: a
//! transport failure on send, a rate limit or transient server error (429,
//! 500, 502, 503, 504, 529), and a stream that died or went silent, or a
//! stream or reply the server failed with a rate limit, an overload, or a
//! server fault, before any reply text arrived. Each attempt resends the
//! same bytes after an exponential backoff, stretched to a server's
//! Retry-After, and tells the caller through [`StreamDelta::Retry`]. A 429
//! for an exhausted quota, and a Retry-After longer than a minute, are
//! reported at once: no wait the turn would take lets the request succeed.
//! Once reply text has streamed, a retry would duplicate what the caller
//! already showed, so a cut after that point is reported as a truncation
//! instead. A failure the server reports inside a 200 response (an `error`
//! object on the chunk or reply or on its choice, or finish_reason `error`)
//! is never a reply, whatever partial output came with it: unless it is
//! retried, it is an error carrying the server's own message. A cut after
//! the server finished its reply leaves the reply finished. Reasoning deltas
//! do not count: a retried attempt's reasoning is void, and the result
//! carries only the final attempt's. A stream this client ends on purpose
//! (the size cap, an out-of-range tool index, cancellation) is never
//! retried: the next attempt would end the same way. A reply carrying only
//! tool calls has no reply text, so a server that never sends a completion
//! signal costs the whole budget on such a reply before its truncation is
//! reported.
//!
//! There is no overall request timeout. A local or slow endpoint can
//! legitimately take minutes to generate, and a deadline here would look like
//! a bug in the model rather than a policy in the client. Silence ends an
//! attempt instead: an endpoint that sends nothing at all (no response
//! headers, no reply bytes, not even an SSE keepalive comment) for
//! [`IDLE_TIMEOUT`], or its provider's `idle_timeout_secs`, is a connection
//! that failed, under the rules above. A one-shot JSON reply or a refusal
//! whose body goes silent is an error, as a cut there is.
//! A stream that ends without `[DONE]` and without a `finish_reason` is
//! reported as truncated rather than treated as a complete reply.
//!
//! A credential never crosses plain http to another machine: a request that
//! would carry the key or an Authorization header that way is refused
//! before it is sent.

use std::collections::HashMap;
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
/// reasoning, `reasoning_details`, and each tool call's `extra_content` go
/// only to the endpoint that produced them (`origin` matches the message's
/// `reasoning_origin`), and the stamp itself never goes. What does go is
/// the stored bytes, unchanged.
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
                tool_calls: m.tool_calls.as_deref().map(|calls| WireToolCalls { calls, replay }),
                tool_call_id: m.tool_call_id.as_deref(),
                reasoning_content: m.reasoning_content.as_deref().filter(|_| replay),
                reasoning: m.reasoning.as_deref().filter(|_| replay),
                reasoning_details: m.reasoning_details.as_deref().filter(|_| replay),
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
    tool_calls: Option<WireToolCalls<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_details: Option<&'a RawValue>,
}

/// One message's tool calls on the wire, each carrying its `extra_content`
/// only when `replay` says the receiver produced it.
struct WireToolCalls<'a> {
    calls: &'a [ToolCall],
    replay: bool,
}

impl Serialize for WireToolCalls<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.calls.iter().map(|c| WireToolCall {
            id: &c.id,
            kind: &c.kind,
            function: &c.function,
            extra_content: c.extra_content.as_deref().filter(|_| self.replay),
        }))
    }
}

/// One tool call on the wire. Fields, order, and skip rules follow
/// `ToolCall`, so a call without `extra_content` keeps its exact bytes.
#[derive(Serialize)]
struct WireToolCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    function: &'a ToolCallFunction,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_content: Option<&'a RawValue>,
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
/// How long an endpoint may send nothing at all (no response headers, no
/// reply bytes, no keepalive comment) before the attempt ends as a transport
/// fault. Generous, because a local server can spend minutes on a long
/// prompt before its first byte; a provider's `idle_timeout_secs` sets
/// another. Without one, an endpoint that took the request and went quiet
/// held the turn forever.
const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

pub struct CompletionResult {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// The reply's reasoning under the key the server sent it in, for the
    /// caller to put on the assistant message stamped with this client's
    /// `origin` (see `ChatMessage`). At most one is set, possibly to an empty
    /// string (`reasoning_fields` says which).
    pub reasoning_content: Option<String>,
    pub reasoning: Option<String>,
    /// The reply's `reasoning_details`, for the same message under the same
    /// stamp: set when the server sent the key as an array with entries. An
    /// empty one carries no signature, and OpenRouter sends it on every delta
    /// of a model that does not reason, so it counts as absent: keeping it
    /// would change the bytes of every such message on disk and on the wire.
    pub reasoning_details: Option<Box<RawValue>>,
    /// The server's reason, or `cancelled` (we stopped reading) or
    /// [`TRUNCATED`] (the server stopped writing without ever finishing).
    pub finish_reason: String,
    /// Server-reported token accounting, when the backend provides it.
    pub usage: Option<Usage>,
}

/// Ground-truth token usage from the server. `cached_tokens` is the number of
/// prompt tokens served from the prompt cache (servers report it under
/// `prompt_tokens_details` or as `prompt_cache_hit_tokens`): if it stays
/// near zero across turns, the harness broke prefix stability and every
/// step is paying a full re-prefill.
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
    /// How long the endpoint may send nothing before an attempt ends.
    idle_timeout: std::time::Duration,
    /// Why no key was sent though settings configure one, for a 401 to
    /// report (see `ActiveEndpoint::key_hint`).
    key_hint: Option<String>,
    http: reqwest::Client,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
    extra_content: Option<Box<RawValue>>,
}

#[derive(Deserialize)]
struct StreamChunk {
    // A failure line may carry no choices at all.
    #[serde(default)]
    choices: Vec<StreamChoice>,
    // Sent on the final chunk when the request asks for it via stream_options.
    usage: Option<UsageJson>,
    // A failure the server reports after its 200 has gone out: alone on its
    // line, or beside a choice that ends with finish_reason `error`.
    error: Option<Value>,
}

#[derive(Deserialize)]
struct UsageJson {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    prompt_tokens_details: Option<PromptTokensDetails>,
    // Some servers report cache hits here instead of (or beside)
    // `prompt_tokens_details.cached_tokens`. Kept untyped so a value of the
    // wrong type in this fallback cannot fail the whole usage object, or a
    // streamed chunk's content with it.
    prompt_cache_hit_tokens: Option<Value>,
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
            cached_tokens: self
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .or(self.prompt_cache_hit_tokens.and_then(|v| v.as_u64())),
        }
    }
}

#[derive(Deserialize)]
struct StreamChoice {
    finish_reason: Option<String>,
    // Some servers omit `delta` entirely on the final finish_reason chunk.
    #[serde(default)]
    delta: StreamDeltaJson,
    // A failure can ride the choice instead of the chunk.
    error: Option<Value>,
}

#[derive(Deserialize, Default)]
struct StreamDeltaJson {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    // Kept untyped, so an entry of a shape this client does not expect
    // cannot fail the whole chunk and the content beside it.
    reasoning_details: Option<Value>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    index: Option<u64>,
    id: Option<String>,
    function: Option<ToolCallFnDelta>,
    // Opaque: any JSON value parses, so it cannot fail the chunk either.
    extra_content: Option<Box<RawValue>>,
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
        Self {
            idle_timeout: ep.idle_timeout_secs.map_or(IDLE_TIMEOUT, |secs| std::time::Duration::from_secs(secs.max(1))),
            key_hint: ep.key_hint.clone(),
            ..Self::with_options(
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
                    // No overall timeout: local generations can legitimately
                    // take minutes. `stream_chat` ends an attempt on silence.
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
            idle_timeout: IDLE_TIMEOUT,
            key_hint: None,
            http,
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Whether a request carries a credential: the key, or an Authorization
    /// header among the provider's headers.
    fn sends_credential(&self) -> bool {
        self.api_key.as_deref().is_some_and(|key| !key.is_empty())
            || self.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
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
        if self.sends_credential() {
            if let Some(refusal) = plain_http_refusal(&self.base_url) {
                return Err(refusal);
            }
        }
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
            // it runs; keep cancellation responsive throughout. Only silence
            // for the whole idle interval ends the wait.
            let send_result = tokio::select! {
                r = tokio::time::timeout(self.idle_timeout, req.send()) => r,
                _ = cancelled.cancelled() => return Ok(cancelled_response()),
            };
            let resp = match send_result {
                Ok(Ok(r)) => r,
                // The endpoint took the request and sent nothing back, not
                // even a status line (a wedged server, a proxy holding a dead
                // upstream). Waiting on would hold the turn for good. Nothing
                // has reached the caller, so it is resent like any transport
                // fault after the connection existed.
                Err(_) => {
                    unreached = 0;
                    let msg = format!("request failed: {}", silence(self.idle_timeout));
                    if attempt < MAX_ATTEMPTS {
                        if !retry_after(attempt, &msg, &cancelled, &mut on_delta).await {
                            return Ok(cancelled_response());
                        }
                        continue;
                    }
                    return Err(msg);
                }
                Ok(Err(e)) => {
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
            // The server's wait governs a resend whether it refused the
            // request by status or failed it inside a 200. An ask past the
            // cap outlasts any wait this turn takes: resending would only
            // meet the same refusal.
            let asked = retry_after_secs(resp.headers());
            let beyond_cap = asked.filter(|&secs| secs > RETRY_AFTER_CAP_SECS);
            if !status.is_success() {
                let code = status.as_u16();
                let body = read_body(resp, &cancelled, self.idle_timeout).await
                    .map_err(|e| format!("backend returned {status}: {e}"))?;
                let Some(body) = body else { return Ok(cancelled_response()); };
                let text = String::from_utf8_lossy(&body);
                let message = describe_backend(&text);
                let mut err = format!("backend returned {status}: {message}");
                if let Some(hint) = temperature_hint(self.temperature, &message) {
                    err.push_str(&hint);
                }
                if code == 401 {
                    if let Some(hint) = &self.key_hint {
                        err.push_str(hint);
                    }
                }
                if attempt < MAX_ATTEMPTS && is_retryable_status(code) && !quota_exhausted(&text) && beyond_cap.is_none() {
                    if !resend_after(attempt, &err, backoff(attempt, asked), &cancelled, &mut on_delta).await {
                        return Ok(cancelled_response());
                    }
                    continue;
                }
                return Err(naming_wait(err, beyond_cap));
            }
            let is_json = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|ct| ct.contains("application/json"));

            // Some servers ignore `stream` and return a complete JSON body.
            if is_json {
                let Some(body) = read_body(resp, &cancelled, self.idle_timeout).await? else { return Ok(cancelled_response()); };
                let v: Value = serde_json::from_slice(&body).map_err(|e| format!("bad JSON response: {e}"))?;
                // A 200 can still carry a failure in place of the reply: an
                // error at the top level, or one on the choice beside
                // finish_reason `error` and partial output that must not
                // stand as the reply. None of it has reached the caller, so
                // the restart rules hold and a retry repeats nothing.
                let choice = &v["choices"][0];
                let error = [&v["error"], &choice["error"]].into_iter().find(|e| !e.is_null());
                if error.is_some() || choice["finish_reason"] == "error" {
                    let failure = server_failure(error.unwrap_or(&Value::Null), &String::from_utf8_lossy(&body));
                    if failure.retryable && attempt < MAX_ATTEMPTS && beyond_cap.is_none() {
                        if !resend_after(attempt, &failure.message, backoff(attempt, asked), &cancelled, &mut on_delta).await {
                            return Ok(cancelled_response());
                        }
                        continue;
                    }
                    return Err(naming_wait(failure.message, beyond_cap));
                }
                return parse_complete_response(&v, &mut on_delta);
            }

            let (reply, unfinished) = read_sse(resp, &cancelled, self.idle_timeout, &mut on_delta).await;
            // The restart rules: nothing the caller keeps has streamed yet,
            // and the budget has an attempt left.
            let restartable = reply.content.is_empty() && attempt < MAX_ATTEMPTS;
            let (reason, wait) = match unfinished {
                // The network cut it: the server's headers never spoke to that.
                Some(Unfinished::Interrupted(reason)) if restartable => (reason, backoff(attempt, None)),
                Some(Unfinished::Failed(failure)) if failure.retryable && restartable && beyond_cap.is_none() => {
                    (failure.message, backoff(attempt, asked))
                }
                // Never a truncated reply: the server said this one failed,
                // and a tool call it carried must not run.
                Some(Unfinished::Failed(failure)) => return Err(naming_wait(failure.message, beyond_cap)),
                _ => return Ok(reply),
            };
            if !resend_after(attempt, &reason, wait, &cancelled, &mut on_delta).await {
                return Ok(cancelled_response());
            }
        }
    }
}

/// Why a stream ended without the reply it promised.
enum Unfinished {
    /// The network ended the stream before the server did: EOF with no
    /// terminator, or a read error before one.
    Interrupted(String),
    /// The server reported a failure inside the stream.
    Failed(ServerFailure),
}

/// A failure a server reported inside a 200 response rather than as its
/// status, which is how a gateway fails once the stream has begun.
struct ServerFailure {
    /// The error to return, or the reason to give for a retry.
    message: String,
    /// A rate limit, an overloaded server, or a server fault: what a fresh
    /// attempt can outlast, on the terms [`is_retryable_status`] resends a
    /// refused status. A refused request, an exhausted quota, or a filtered
    /// reply would fail the same way again.
    retryable: bool,
}

/// Read an `error` value (an object, or a bare message string) so the
/// message is the provider's own, wherever in the response it sat. `raw`,
/// the response text that carried it, stands in only when there is no
/// message to read (or no error, only finish_reason `error`).
fn server_failure(error: &Value, raw: &str) -> ServerFailure {
    let code = &error["code"];
    // A numeric code, sometimes sent as a string, is an HTTP status.
    let status = code.as_u64().or_else(|| code.as_str()?.parse().ok());
    // Only the code and type tell an exhausted quota from a rate limit: a
    // per-minute limit's message can say "quota" too, and is worth a retry.
    let kinds = || [code, &error["type"]].into_iter().filter_map(Value::as_str);
    let retryable = !kinds().any(|kind| kind.contains("quota"))
        && match status {
            // Resent exactly when that status would be: a 501 inside a 200
            // is no more worth a retry than a 501 status.
            Some(status) => u16::try_from(status).is_ok_and(is_retryable_status),
            None => kinds().any(|kind| ["rate_limit", "overloaded", "server_error"].iter().any(|k| kind.contains(k))),
        };
    let code = match code {
        Value::Null => String::new(),
        // Without the quotes a JSON string would print with.
        Value::String(code) => format!(" ({code})"),
        code => format!(" ({code})"),
    };
    let message = error["message"].as_str().or(error.as_str()).map_or_else(|| describe_backend(raw), |m| truncate(m, 600));
    ServerFailure { message: format!("backend reported an error{code}: {message}"), retryable }
}

/// Parse one SSE stream to its end. The second value says why the stream
/// ended unfinished: the network cut it, or the server reported a failure
/// inside it. It is `None` for a finished reply and for a stream this client
/// cut itself, which a fresh attempt would cut the same way.
async fn read_sse(
    resp: reqwest::Response,
    cancelled: &crate::state::CancelToken,
    idle: std::time::Duration,
    on_delta: &mut impl FnMut(StreamDelta),
) -> (CompletionResult, Option<Unfinished>) {
    let mut content = String::new();
    // One buffer per key, so the reply goes back under the key that carried
    // it. `None` until the server sends that key as a string, even an empty one.
    let mut reasoning_content: Option<String> = None;
    let mut reasoning: Option<String> = None;
    // Empty until the server sends `reasoning_details` with an entry in it.
    let mut reasoning_details: Vec<Value> = Vec::new();
    let mut open_details = OpenDetails::new();
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
    // Set when something outside this client ended the stream unfinished:
    // the network cut it, or the server reported a failure in it.
    let mut unfinished: Option<Unfinished> = None;

    'outer: loop {
        let next = tokio::select! {
            c = tokio::time::timeout(idle, stream.next()) => c,
            _ = cancelled.cancelled() => {
                finish_reason = "cancelled".into();
                break;
            }
        };
        // Not a byte, not even a keepalive comment, for the whole idle
        // interval: the connection failed as surely as a cut, and ends the
        // same way. After the server finished, the reply stands.
        let Ok(next) = next else {
            if !saw_terminator {
                finish_reason = TRUNCATED.into();
                unfinished = Some(Unfinished::Interrupted(silence(idle)));
            }
            break;
        };
        let Some(chunk) = next else {
            if !saw_terminator {
                unfinished = Some(Unfinished::Interrupted("the stream ended before the reply finished".into()));
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
            // A cut after the server finished (while the usage chunk or
            // `[DONE]` was still due) costs nothing of the reply: it stands,
            // rather than coming back truncated or being generated and
            // billed a second time.
            Err(_) if saw_terminator => break,
            Err(e) => {
                finish_reason = TRUNCATED.into();
                // Every error this stream yields is the connection failing
                // under the body: a chunked or sized body cut short surfaces
                // here, not as a clean end, and this client applies no content
                // decoding that could fail on its own. So each one is worth a
                // fresh attempt.
                unfinished = Some(Unfinished::Interrupted(describe_transport(&e)));
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
            let mut choice = chunk.choices.into_iter().next();
            let error = chunk.error.or_else(|| choice.as_mut()?.error.take());
            // finish_reason `error` ends the stream, but not as a reply: it
            // is a failure even when the server sends no error to go with it.
            if error.is_some() || choice.as_ref().is_some_and(|c| c.finish_reason.as_deref() == Some("error")) {
                let error = error.unwrap_or(Value::Null);
                unfinished = Some(Unfinished::Failed(server_failure(&error, &String::from_utf8_lossy(data))));
                break 'outer;
            }
            // The usage-bearing final chunk has an empty choices array.
            let Some(choice) = choice else { continue };

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
            if let Some(Value::Array(parts)) = delta.reasoning_details {
                for part in parts {
                    merge_reasoning_detail(&mut reasoning_details, &mut open_details, part);
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
                    if let Some(extra) = tc.extra_content {
                        partials[idx].extra_content = Some(extra);
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
    let reasoning_details = Some(reasoning_details).filter(|d| !d.is_empty()).and_then(|d| serde_json::value::to_raw_value(&d).ok());
    (CompletionResult { content, tool_calls, reasoning_content, reasoning, reasoning_details, finish_reason, usage }, unfinished)
}

/// Where each text or summary block of a streaming reply's
/// `reasoning_details` has its latest entry, keyed by the block's `index` and
/// the field its text streams in, so a part finds the entry it continues in
/// one lookup. The server decides how many parts a stream has: a scan of the
/// entries kept so far would make a stream of parts that each start a new
/// block cost time quadratic in its length.
type OpenDetails = HashMap<(u64, &'static str), usize>;

/// Fold one streamed `reasoning_details` part into the reply's entries.
/// OpenRouter streams a text or summary block in parts that share its
/// `index` and `type`: the text arrives in pieces, and fields such as the
/// signature arrive null at first and set in a later part. Such a part
/// continues the latest entry with the same `index` and `type`: its text is
/// appended, and a value it carries fills a field still null or empty. A
/// part whose `id` conflicts with that entry's starts a new entry instead,
/// which later parts of its block continue. Any other part (an encrypted
/// blob, a type this client does not know, an entry without an index)
/// arrives whole and is kept as its own entry, since joining two such blobs
/// would corrupt both.
fn merge_reasoning_detail(details: &mut Vec<Value>, open: &mut OpenDetails, part: Value) {
    let text_key = match part["type"].as_str() {
        Some("reasoning.text") => "text",
        Some("reasoning.summary") => "summary",
        _ => {
            details.push(part);
            return;
        }
    };
    let Some(index) = part["index"].as_u64() else {
        details.push(part);
        return;
    };
    // The entry's `index` and `type` are never null or empty, so merging
    // never changes them: the entry stays under the key it was opened with.
    let latest = open.get(&(index, text_key)).map(|&at| &mut details[at]);
    let continued = latest.filter(|entry| entry["id"].is_null() || part["id"].is_null() || entry["id"] == part["id"]);
    let Some(Value::Object(entry)) = continued else {
        open.insert((index, text_key), details.len());
        details.push(part);
        return;
    };
    let Value::Object(part) = part else { return };
    for (key, value) in part {
        match entry.get_mut(&key) {
            Some(Value::String(held)) if key == text_key => {
                if let Value::String(more) = value {
                    held.push_str(&more);
                }
            }
            Some(held) if !held.is_null() && held.as_str() != Some("") => {}
            _ => {
                entry.insert(key, value);
            }
        }
    }
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
        reasoning_details: None,
        finish_reason: "cancelled".into(),
        usage: None,
    }
}

async fn read_body(
    resp: reqwest::Response,
    cancelled: &crate::state::CancelToken,
    idle: std::time::Duration,
) -> Result<Option<Vec<u8>>, String> {
    let mut body = Vec::new();
    let mut stream = resp.bytes_stream();
    loop {
        let next = tokio::select! {
            _ = cancelled.cancelled() => return Ok(None),
            next = tokio::time::timeout(idle, stream.next()) => next,
        };
        let Ok(next) = next else { return Err(format!("response body failed: {}", silence(idle))) };
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
                extra_content: opaque(&tc["extra_content"]),
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
    let reasoning_details = Some(&msg["reasoning_details"]).filter(|d| d.as_array().is_some_and(|a| !a.is_empty())).and_then(opaque);
    Ok(CompletionResult { content, tool_calls, reasoning_content, reasoning, reasoning_details, finish_reason, usage })
}

/// A value from a one-shot reply, kept to send back; null is absent. It is
/// encoded once, here, from the parsed reply, and goes out as stored.
fn opaque(value: &Value) -> Option<Box<RawValue>> {
    if value.is_null() {
        return None;
    }
    serde_json::value::to_raw_value(value).ok()
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
            extra_content: p.extra_content,
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
/// first, later only when a server's Retry-After asks for longer waits, up
/// to [`RETRY_AFTER_CAP_SECS`] each (see [`backoff`]); a longer ask is
/// reported instead of retried. A transient fault on the path to an
/// endpoint (a reset or a TLS alert on send, a stream cut mid-reply) can
/// recur for minutes and clears within seconds most times and within a
/// minute at worst, so attempts packed into one second only ever observe the
/// fault and end a turn the next connection would have completed. Every
/// wait is announced and cancellable.
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
/// The longest Retry-After this client waits out, in seconds: a minute, the
/// whole window of a per-minute rate limit. A longer ask is a limit that
/// resets hours away (a daily quota, a maintenance window). No resend before
/// then can succeed and holding the turn that long helps no one, so the
/// refusal is reported at once with the server's wait.
const RETRY_AFTER_CAP_SECS: u64 = 60;

/// Statuses that say the server could not take the request just now: a rate
/// limit (429), or an upstream that failed or shed load (500, 502, 503, 504,
/// and the 529 an overloaded provider sends). Resending is safe: a chat
/// completion changes nothing, tools run only once the agent loop holds a
/// finished reply, and a refused status carries no reply text the caller
/// could show twice. A gateway's 500 or 502 can follow work the upstream
/// already did, so a resend may pay for that work twice. The chat
/// completions API has no idempotency key to collapse the two, and failing
/// the turn saves nothing: it ends with no reply, and the only way to one
/// is the same request sent again. A stream cut before any reply text, and
/// a failure a server reports inside a 200 before any reply text with one
/// of these codes (see [`server_failure`]), are resent on the same terms. A
/// 429 for an exhausted quota is excluded by [`quota_exhausted`].
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

/// Why an attempt ended when the endpoint sent nothing for `idle`.
fn silence(idle: std::time::Duration) -> String {
    format!("the endpoint sent nothing for {}s", idle.as_secs_f64())
}

/// Why a request to `base_url` must not carry a credential, or None when it
/// may. Plain http to another machine puts the key on the network
/// unencrypted, readable anywhere along the path. Loopback never leaves this
/// machine, and neither does the unspecified address (0.0.0.0 or ::), which
/// a local server often prints as its own and which connects here. A
/// base_url that does not parse is left to the request to fail.
fn plain_http_refusal(base_url: &str) -> Option<String> {
    let url = reqwest::Url::parse(base_url.trim()).ok()?;
    if url.scheme() != "http" {
        return None;
    }
    let host = url.host_str()?;
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let this_machine = bare.eq_ignore_ascii_case("localhost")
        || bare.parse::<std::net::IpAddr>().is_ok_and(|ip| {
            let ip = ip.to_canonical();
            ip.is_loopback() || ip.is_unspecified()
        });
    (!this_machine).then(|| {
        format!(
            "refusing to send credentials over plain http to {host}, where anyone on the network path can read them: use an https base_url, or a loopback address (127.0.0.1, ::1, or localhost) for a server on this machine. A server that needs no key works over http once none is configured for it (api_key, api_key_env, OPENMAX_API_KEY, or an Authorization header)"
        )
    })
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

/// `err`, naming the server's wait when that is past
/// [`RETRY_AFTER_CAP_SECS`]: why it was not resent, and when to try again.
fn naming_wait(mut err: String, beyond_cap: Option<u64>) -> String {
    if let Some(secs) = beyond_cap {
        err.push_str(&format!(" (the server asks for a retry after {secs}s)"));
    }
    err
}

/// [`retry_after`] with the wait chosen by the caller: a refused status, or
/// a failure inside a 200, passes the server's Retry-After through
/// [`backoff`]. Returns false when cancelled during the wait: one can reach
/// [`RETRY_AFTER_CAP_SECS`], and a user who cancels must not sit through it.
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
        serve_once(response, hold, None)
    }

    /// [`spawn_response_once`] that also hands back the request head it read
    /// (request line and headers), to show what the client sent.
    fn spawn_recording_once(response: String) -> (String, std::sync::mpsc::Receiver<String>) {
        let (head, heads) = std::sync::mpsc::channel();
        (serve_once(response, None, Some(head)), heads)
    }

    fn serve_once(
        response: String,
        hold: Option<std::sync::mpsc::Receiver<()>>,
        head: Option<std::sync::mpsc::Sender<String>>,
    ) -> String {
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
            if let Some(head) = head {
                let _ = head.send(headers.clone());
            }
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

    /// Some servers report prompt cache hits only as a top-level
    /// `prompt_cache_hit_tokens`. Reading `prompt_tokens_details` alone shows
    /// their hits as not reported, so a working prompt cache looks absent.
    #[tokio::test]
    async fn cache_hits_reported_as_prompt_cache_hit_tokens_are_counted() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"prompt_cache_hit_tokens\":80,\"prompt_cache_miss_tokens\":20}}\n\n",
            "data: [DONE]\n\n",
        ))
        .await;
        let usage = result.usage.expect("the final chunk carries usage");
        assert_eq!((usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens), (100, 5, Some(80)));

        let usage_of = |usage: Value| {
            let reply = json!({"choices":[{"message":{"content":"ok"}}],"usage":usage});
            parse_complete_response(&reply, &mut |_| {}).unwrap().usage.expect("usage is parsed")
        };
        assert_eq!(usage_of(json!({"prompt_tokens":100,"prompt_cache_hit_tokens":64})).cached_tokens, Some(64));
        // The details field wins when a server sends both.
        let both = json!({"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":70},"prompt_cache_hit_tokens":64});
        assert_eq!(usage_of(both).cached_tokens, Some(70));
        let null_details = json!({"prompt_tokens":100,"prompt_tokens_details":null,"prompt_cache_hit_tokens":64});
        assert_eq!(usage_of(null_details).cached_tokens, Some(64));
        assert_eq!(usage_of(json!({"prompt_tokens_details":{"cached_tokens":70}})).cached_tokens, Some(70));
        // A server that reports neither still reads as not reported, not zero.
        assert_eq!(usage_of(json!({"prompt_tokens":100})).cached_tokens, None);
    }

    /// `prompt_cache_hit_tokens` is only a fallback. A value of the wrong
    /// type there must not fail the whole usage object (or, when streamed,
    /// the whole chunk with its content): that would lose token counts and
    /// text that parsed fine before the field was read at all.
    #[tokio::test]
    async fn an_unusable_prompt_cache_hit_tokens_keeps_the_rest_of_usage() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],",
            "\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,",
            "\"prompt_tokens_details\":{\"cached_tokens\":70},\"prompt_cache_hit_tokens\":\"80\"}}\n\n",
            "data: [DONE]\n\n",
        ))
        .await;
        assert_eq!((result.content.as_str(), result.finish_reason.as_str()), ("ok", "stop"));
        let usage = result.usage.expect("usage beside an unusable fallback still parses");
        assert_eq!((usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens), (100, 5, Some(70)));

        let usage_of = |usage: Value| {
            let reply = json!({"choices":[{"message":{"content":"ok"}}],"usage":usage});
            parse_complete_response(&reply, &mut |_| {}).unwrap().usage.expect("usage is parsed")
        };
        let with_details = json!({"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":70},"prompt_cache_hit_tokens":"80"});
        assert_eq!(usage_of(with_details).cached_tokens, Some(70));
        // With nothing usable to fall back on, cached tokens read as not reported.
        for bad in [json!("80"), json!(-1), json!(1.5), json!({})] {
            let usage = usage_of(json!({"prompt_tokens":100,"prompt_cache_hit_tokens":bad}));
            assert_eq!((usage.prompt_tokens, usage.cached_tokens), (100, None));
        }
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
            // Connections a SILENT or STALL: body holds open, until the
            // sequence ends.
            let mut held = Vec::new();
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
                // SILENT takes the request and answers nothing, not even a
                // status line, while keeping the connection open.
                if sse == "SILENT" {
                    held.push(stream);
                    continue;
                }
                // A body prefixed with STALL: is sent as one chunk of a
                // chunked stream, which then neither continues nor ends.
                if let Some(payload) = sse.strip_prefix("STALL:") {
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{payload}\r\n",
                            payload.len()
                        )
                        .as_bytes(),
                    );
                    held.push(stream);
                    continue;
                }
                // A body prefixed with KEEPALIVE: is preceded by SSE comments,
                // each sent well inside [`SILENT_FOR`] and all of them well
                // past it: a server holding its stream open through a long
                // prompt.
                if let Some(payload) = sse.strip_prefix("KEEPALIVE:") {
                    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
                    for _ in 0..12 {
                        std::thread::sleep(SILENT_FOR / 5);
                        let _ = stream.write_all(b": keepalive\n\n");
                    }
                    let _ = stream.write_all(payload.as_bytes());
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
                // A body prefixed with JSON: is a whole JSON reply, the
                // shape of a server that ignores `stream`.
                if let Some(json) = sse.strip_prefix("JSON:") {
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{json}",
                            json.len()
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
        let (result, deltas, served) = try_stream_sequence(bodies).await;
        (result.unwrap(), deltas, served)
    }

    async fn try_stream_sequence(bodies: Vec<String>) -> (Result<CompletionResult, String>, Vec<String>, usize) {
        try_stream_sequence_idle(bodies, IDLE_TIMEOUT).await
    }

    /// [`try_stream_sequence`] with the client giving up on an endpoint
    /// after `idle` of silence.
    async fn try_stream_sequence_idle(
        bodies: Vec<String>,
        idle: std::time::Duration,
    ) -> (Result<CompletionResult, String>, Vec<String>, usize) {
        let (url, served) = spawn_sse_sequence(bodies);
        let mut deltas: Vec<String> = Vec::new();
        let mut client = ChatClient::new(url, None, "m".into(), None, 64);
        client.idle_timeout = idle;
        let result = client
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |d| {
                deltas.push(match d {
                    StreamDelta::Content(t) => format!("content:{t}"),
                    StreamDelta::Reasoning(t) => format!("reasoning:{t}"),
                    StreamDelta::Retry { attempt, max_attempts, reason } => format!("retry:{attempt}/{max_attempts}:{reason}"),
                })
            })
            .await;
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

    /// How a gateway reports a failure once its 200 has gone out: an `error`
    /// object beside a choice that ends with finish_reason `error`.
    fn gateway_error_chunk(code: Value, message: &str) -> String {
        let chunk = json!({
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "error": {"code": code, "message": message},
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
        });
        format!("data: {chunk}\n\ndata: [DONE]\n\n")
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

    /// The bug this guards: finish_reason `error` read as a clean end, so a
    /// failure the server reported inside its stream ended the turn with
    /// nothing said. It is an error carrying the provider's own message, and
    /// one that would fail the same way again is not resent.
    #[tokio::test]
    async fn an_error_inside_the_stream_is_reported_with_the_providers_message() {
        let (result, deltas, served) = try_stream_sequence(vec![
            gateway_error_chunk(json!(400), "the prompt is longer than this endpoint accepts"),
            FINISHED.into(),
        ])
        .await;
        let err = result.err().expect("a failure the server reported is an error");
        assert!(err.contains("the prompt is longer than this endpoint accepts"), "{err}");
        assert_eq!(served, 1, "a refused request is not resent");
        assert!(deltas.is_empty(), "{deltas:?}");

        // A retryable failure after reply text streamed is reported, not
        // resent: the caller has already shown that text.
        let (result, deltas, served) = try_stream_sequence(vec![
            format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"half an ans\"}},\"finish_reason\":null}}]}}\n\n{}",
                gateway_error_chunk(json!("server_error"), "upstream disconnected unexpectedly")
            ),
            FINISHED.into(),
        ])
        .await;
        let err = result.err().expect("a failure after reply text is still an error");
        assert!(err.contains("upstream disconnected unexpectedly"), "{err}");
        assert_eq!(served, 1);
        assert_eq!(deltas, vec!["content:half an ans".to_string()]);
    }

    /// A rate limit, an overloaded upstream, or a server fault reported
    /// before any reply text is what a fresh attempt outlasts, as the same
    /// failure sent as a status is: the reply starts over under the same
    /// rules as a dropped stream, and the retry names the provider's reason.
    #[tokio::test]
    async fn a_retryable_error_inside_the_stream_is_started_over_before_reply_text() {
        for (failure, reason) in [
            (
                gateway_error_chunk(json!(429), "rate limit reached upstream"),
                "backend reported an error (429): rate limit reached upstream",
            ),
            (
                "data: {\"error\":{\"type\":\"rate_limit_error\",\"message\":\"too many requests upstream\"}}\n\n".to_string(),
                "backend reported an error: too many requests upstream",
            ),
            (
                "data: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"upstream overloaded\"}}\n\n".to_string(),
                "backend reported an error: upstream overloaded",
            ),
            (
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\",\"error\":{\"code\":\"server_error\",\"message\":\"upstream reset\"}}]}\n\n".to_string(),
                "backend reported an error (server_error): upstream reset",
            ),
            // The failure can ride the choice instead of the chunk. Read only
            // at the top level, this one was a failure with no code (never
            // retried) and the raw chunk stood in for its message.
            (
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"error\",\"error\":{\"code\":429,\"message\":\"Rate limit exceeded\"}}]}\n\n".to_string(),
                "backend reported an error (429): Rate limit exceeded",
            ),
        ] {
            let (result, deltas, served) = try_stream_sequence(vec![failure, FINISHED.into()]).await;
            let result = result.expect("the second attempt finished");
            assert_eq!(served, 2);
            assert_eq!(result.finish_reason, "stop");
            assert_eq!(result.content, "all of it");
            assert_eq!(
                deltas,
                vec![format!("retry:2/{MAX_ATTEMPTS}:{reason}"), "content:all of it".to_string()]
            );
        }
    }

    /// The bug this guards: a status code inside a 200 was resent whenever it
    /// was a 5xx, so a 501 or 505, which no attempt can get past and which
    /// the same refusal as a status reports at once, cost the whole retry
    /// budget, about a minute, before its message was shown. A code inside a
    /// 200 is resent exactly when that status would be.
    #[tokio::test]
    async fn a_status_inside_a_200_is_resent_on_the_terms_of_that_status() {
        for code in [429, 500, 502, 503, 504, 529, 501, 505] {
            let failure = gateway_error_chunk(json!(code), "upstream failed");
            let (result, deltas, served) = try_stream_sequence(vec![failure, FINISHED.into()]).await;
            if is_retryable_status(code) {
                assert_eq!(result.expect("the second attempt finished").content, "all of it", "{code}");
                assert_eq!(served, 2, "{code} is resent as its status is");
                assert_eq!(deltas[0], format!("retry:2/{MAX_ATTEMPTS}:backend reported an error ({code}): upstream failed"));
            } else {
                assert_eq!(result.err(), Some(format!("backend reported an error ({code}): upstream failed")));
                assert_eq!(served, 1, "{code} is reported at once, as its status is");
                assert!(deltas.is_empty(), "{code}: {deltas:?}");
            }
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

    /// Stream against a server that sends `refusal` and then a finished
    /// reply. Returns the outcome, the resends announced, and the requests
    /// served.
    async fn stream_after_refusal(refusal: String) -> (Result<CompletionResult, String>, usize, usize) {
        let (url, served) = spawn_sse_sequence(vec![refusal, FINISHED.into()]);
        let mut retries = 0;
        let result = ChatClient::new(url, None, "m".into(), None, 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |d| {
                if let StreamDelta::Retry { .. } = d {
                    retries += 1;
                }
            })
            .await;
        let served = *served.lock().unwrap();
        (result, retries, served)
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
        let (result, retries, served) = stream_after_refusal(exhausted).await;
        let Err(err) = result else { panic!("an exhausted quota is an error, not a resend") };
        assert_eq!(
            err,
            "backend returned 429 Too Many Requests: You exceeded your current quota, please check your plan and billing details."
        );
        assert_eq!(retries, 0);
        assert_eq!(served, 1);
    }

    /// A Retry-After past [`RETRY_AFTER_CAP_SECS`] is a limit that lifts
    /// long after any wait a turn can take (a daily token limit, a
    /// maintenance window). The bug this guards cut such an ask to the cap
    /// and resent anyway on every attempt: minutes of waits, each answered
    /// with the same refusal. It comes back at once with the server's wait,
    /// so the user knows when to try again.
    #[tokio::test]
    async fn a_retry_after_past_the_cap_is_reported_without_a_retry() {
        for status in ["429 Too Many Requests", "503 Service Unavailable"] {
            let refused = status_response(status, "Retry-After: 3600\r\n", r#"{"error":{"message":"daily limit reached"}}"#);
            let (result, retries, served) = stream_after_refusal(refused).await;
            let Err(err) = result else { panic!("{status}: an ask past the cap is an error, not a resend") };
            assert_eq!(err, format!("backend returned {status}: daily limit reached (the server asks for a retry after 3600s)"));
            assert_eq!(retries, 0, "{status}");
            assert_eq!(served, 1, "{status}");
        }
    }

    /// A failure inside a 200 is resent on the terms of a refused status,
    /// and those include the server's Retry-After. The bug this guards read
    /// the header only on a refused status: a rate limit a gateway put
    /// inside a 200 with `Retry-After: 3600` was resent on the bare backoff
    /// into a limit the server said would hold for an hour, and the error
    /// came only after the whole budget, without the server's wait.
    #[tokio::test]
    async fn a_failure_inside_a_200_waits_as_the_server_asks() {
        let ok = |content_type: &str, retry_after: u64, body: &str| {
            format!(
                "RAW:HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nRetry-After: {retry_after}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        };
        let json_failure = r#"{"error":{"code":429,"message":"rate limit reached upstream"}}"#.to_string();
        let sse_failure = gateway_error_chunk(json!(429), "rate limit reached upstream");
        for (content_type, body) in [("application/json", json_failure), ("text/event-stream", sse_failure)] {
            let (result, retries, served) = stream_after_refusal(ok(content_type, 3600, &body)).await;
            assert_eq!(
                result.err().as_deref(),
                Some("backend reported an error (429): rate limit reached upstream (the server asks for a retry after 3600s)"),
                "{content_type}"
            );
            assert_eq!((retries, served), (0, 1), "{content_type}");

            let started = std::time::Instant::now();
            let (result, retries, served) = stream_after_refusal(ok(content_type, 40, &body)).await;
            let waited = started.elapsed();
            assert_eq!(result.expect("the second attempt finished").content, "all of it", "{content_type}");
            assert_eq!((retries, served), (1, 2), "{content_type}");
            assert!(waited >= BACKOFF_UNIT * 40, "{content_type}: resent after {waited:?}, before the 40 units the server asked for");
        }
    }

    /// Some servers send the failure alone on its line, with no choices. That
    /// line failed to parse and was skipped, so the stream read as one that
    /// died with nothing said, and was resent until the budget ran out.
    #[tokio::test]
    async fn a_bare_error_line_is_reported_with_the_providers_message() {
        for line in [
            r#"data: {"error":{"message":"insufficient credits for this request","code":402}}"#,
            r#"data: {"error":"insufficient credits for this request"}"#,
        ] {
            let (result, _, served) = try_stream_sequence(vec![format!("{line}\n\n"), FINISHED.into()]).await;
            let err = result.err().expect("a bare error line is an error");
            assert!(err.contains("insufficient credits for this request"), "{err}");
            assert_eq!(served, 1, "a refused request is not resent");
        }

        // finish_reason `error` with no error beside it still fails the
        // reply, and says what the server sent.
        let (result, _, served) = try_stream_sequence(vec![
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"error\"}]}\n\ndata: [DONE]\n\n".into(),
            FINISHED.into(),
        ])
        .await;
        let err = result.err().expect("finish_reason error is an error");
        assert!(err.contains(r#""finish_reason":"error""#), "{err}");
        assert_eq!(served, 1);
    }

    /// A server that ignores `stream` can put the same failure in its JSON
    /// reply. It reads the same way: the provider's message, and a retry when
    /// a fresh attempt can outlast it.
    #[tokio::test]
    async fn a_failure_in_a_json_reply_is_reported_like_one_in_a_stream() {
        let (result, deltas, served) = try_stream_sequence(vec![
            r#"JSON:{"error":{"code":503,"message":"upstream overloaded"}}"#.into(),
            FINISHED.into(),
        ])
        .await;
        assert_eq!(result.expect("the second attempt finished").content, "all of it");
        assert_eq!(served, 2);
        assert!(deltas[0].starts_with(&format!("retry:2/{MAX_ATTEMPTS}:")) && deltas[0].contains("upstream overloaded"), "{deltas:?}");

        let (result, _, served) = try_stream_sequence(vec![
            r#"JSON:{"error":{"type":"invalid_request_error","message":"unknown parameter: tools"}}"#.into(),
            FINISHED.into(),
        ])
        .await;
        let err = result.err().expect("a refused request is an error");
        assert_eq!(err, "backend reported an error: unknown parameter: tools");
        assert_eq!(served, 1);

        // A failed JSON reply can instead carry its error on the choice,
        // beside finish_reason `error` and the partial output that came
        // first. That read as a finished reply: the partial text was kept
        // and its tool call ran. None of it has reached the caller yet, so
        // a failure a fresh attempt can outlast is resent.
        let (result, deltas, served) = try_stream_sequence(vec![
            failed_json_reply(json!({"code": 502, "message": "Provider disconnected mid-stream"})),
            FINISHED.into(),
        ])
        .await;
        assert_eq!(result.expect("the second attempt finished").content, "all of it");
        assert_eq!(served, 2);
        assert_eq!(
            deltas,
            vec![
                format!("retry:2/{MAX_ATTEMPTS}:backend reported an error (502): Provider disconnected mid-stream"),
                "content:all of it".to_string(),
            ]
        );

        // One that would fail the same way again is an error, and neither
        // the partial text nor the tool call comes back.
        let (result, deltas, served) = try_stream_sequence(vec![
            failed_json_reply(json!({"code": 403, "message": "output flagged by a content filter"})),
            FINISHED.into(),
        ])
        .await;
        let err = result.err().expect("a failed reply is an error");
        assert_eq!(err, "backend reported an error (403): output flagged by a content filter");
        assert_eq!(served, 1);
        assert!(deltas.is_empty(), "{deltas:?}");

        // finish_reason `error` with no error beside it fails the reply too.
        let (result, deltas, served) = try_stream_sequence(vec![failed_json_reply(Value::Null), FINISHED.into()]).await;
        let err = result.err().expect("finish_reason error is an error");
        assert!(err.contains(r#""finish_reason":"error""#), "{err}");
        assert_eq!(served, 1);
        assert!(deltas.is_empty(), "{deltas:?}");
    }

    /// A JSON reply a gateway failed partway: partial output, a tool call,
    /// and finish_reason `error` with `error` on the choice.
    fn failed_json_reply(error: Value) -> String {
        let reply = json!({"choices": [{
            "message": {
                "role": "assistant",
                "content": "partial output",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{\"path\":\"a.rs\",\"content\":\"x\"}"},
                }],
            },
            "finish_reason": "error",
            "error": error,
        }]});
        format!("JSON:{reply}")
    }

    #[test]
    fn server_failures_a_retry_can_outlast() {
        let retryable = |error: Value| server_failure(&error, "").retryable;
        // A status code is retried exactly when that status would be.
        for code in [json!(429), json!(500), json!(502), json!(503), json!(504), json!(529), json!("429"), json!("503")] {
            assert!(retryable(json!({"code": code, "message": "m"})), "{code}");
        }
        for kind in ["rate_limit_exceeded", "rate_limit_error", "server_error", "overloaded_error"] {
            assert!(retryable(json!({"code": kind})), "{kind}");
            assert!(retryable(json!({"type": kind})), "{kind}");
        }
        // A per-minute limit can name its quota in the message; only the
        // code and type mark one as exhausted.
        assert!(retryable(json!({"code": 429, "message": "Quota exceeded for requests per minute"})));
        for error in [
            // A refusal no attempt gets past, as a status or inside a 200.
            json!({"code": 501}),
            json!({"code": 505}),
            json!({"code": "501"}),
            json!({"code": 400}),
            json!({"code": 401}),
            json!({"code": 402}),
            json!({"code": 404}),
            json!({"code": "insufficient_quota", "type": "insufficient_quota"}),
            // An exhausted quota fails the same way however it is coded.
            json!({"code": 429, "type": "insufficient_quota"}),
            json!({"type": "invalid_request_error"}),
            json!({"message": "no code at all"}),
            json!("a bare message"),
            Value::Null,
        ] {
            assert!(!retryable(error.clone()), "{error}");
        }

        let raw = r#"{"error":{"code":429,"message":"slow down"}}"#;
        let error: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(server_failure(&error["error"], raw).message, "backend reported an error (429): slow down");
        let raw = r#"{"error":{"code":"server_error","message":"upstream reset"}}"#;
        let error: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(server_failure(&error["error"], raw).message, "backend reported an error (server_error): upstream reset");
    }

    /// The bug this guards: a connection cut after the server finished its
    /// reply threw that finish away. A reply with text came back truncated,
    /// and one carrying only tool calls has no text, so it was generated and
    /// billed a second time. The finish the server sent stands.
    #[tokio::test]
    async fn a_finished_reply_survives_a_connection_cut_after_it() {
        let (result, deltas, served) = stream_sequence(vec![
            concat!(
                "CHUNKED:data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            )
            .into(),
            FINISHED.into(),
        ])
        .await;
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.content, "all of it");
        assert_eq!(served, 1);
        assert_eq!(deltas, vec!["content:all of it".to_string()]);

        let call = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        );
        let (result, deltas, served) = stream_sequence(vec![format!("CHUNKED:{call}"), call.into()]).await;
        assert_eq!(served, 1, "the finished reply is generated once");
        assert!(deltas.is_empty(), "{deltas:?}");
        assert_eq!(result.finish_reason, "tool_calls");
        assert_eq!(result.tool_calls.len(), 1);
    }

    /// How long a test endpoint stays quiet before the client gives up on
    /// it: the idle interval, injected in place of the minutes a real
    /// endpoint gets.
    const SILENT_FOR: std::time::Duration = std::time::Duration::from_millis(250);

    /// [`try_stream_sequence_idle`] at [`SILENT_FOR`], under a deadline that
    /// turns a hang into a failure.
    async fn stream_with_silence(bodies: Vec<String>) -> (Result<CompletionResult, String>, Vec<String>, usize) {
        tokio::time::timeout(std::time::Duration::from_secs(10), try_stream_sequence_idle(bodies, SILENT_FOR))
            .await
            .expect("a silent endpoint must not hold the request forever")
    }

    /// The bug this guards: only connecting had a deadline, so an endpoint
    /// that took the request and then sent nothing (a wedged server, a proxy
    /// holding a dead upstream) held the turn forever, and a headless run
    /// never exited. Silence for the whole idle interval ends the attempt as
    /// a transport fault: before any reply text it is resent, whether the
    /// endpoint never answered or its stream stopped part way.
    #[tokio::test]
    async fn a_silent_endpoint_is_resent_before_reply_text() {
        let silent = format!("the endpoint sent nothing for {}s", SILENT_FOR.as_secs_f64());
        let (result, deltas, served) = stream_with_silence(vec!["SILENT".into(), FINISHED.into()]).await;
        assert_eq!(result.expect("the second attempt finished").content, "all of it");
        assert_eq!(served, 2);
        assert_eq!(deltas, vec![format!("retry:2/{MAX_ATTEMPTS}:request failed: {silent}"), "content:all of it".to_string()]);

        let thinking = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me\"},\"finish_reason\":null}]}\n\n";
        let (result, deltas, served) = stream_with_silence(vec![format!("STALL:{thinking}"), FINISHED.into()]).await;
        assert_eq!(result.expect("the second attempt finished").content, "all of it");
        assert_eq!(served, 2);
        assert_eq!(
            deltas,
            vec!["reasoning:let me".to_string(), format!("retry:2/{MAX_ATTEMPTS}:{silent}"), "content:all of it".to_string()]
        );
    }

    /// Silence after reply text is a truncation, as a cut there is: the
    /// caller has already shown that text. Silence after the server finished
    /// leaves the reply finished.
    #[tokio::test]
    async fn a_stream_silent_after_reply_text_or_its_finish_is_not_resent() {
        let half = "data: {\"choices\":[{\"delta\":{\"content\":\"half an ans\"},\"finish_reason\":null}]}\n\n";
        let (result, deltas, served) = stream_with_silence(vec![format!("STALL:{half}"), FINISHED.into()]).await;
        let result = result.expect("a truncated reply is still a reply");
        assert_eq!((result.content.as_str(), result.finish_reason.as_str()), ("half an ans", TRUNCATED));
        assert_eq!(served, 1);
        assert_eq!(deltas, vec!["content:half an ans".to_string()]);

        let done = "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"},\"finish_reason\":\"stop\"}]}\n\n";
        let (result, _, served) = stream_with_silence(vec![format!("STALL:{done}"), FINISHED.into()]).await;
        let result = result.expect("a finished reply");
        assert_eq!((result.content.as_str(), result.finish_reason.as_str()), ("all of it", "stop"));
        assert_eq!(served, 1);
    }

    /// Any byte is activity, an SSE comment too: a server that keeps its
    /// stream alive with comments through a long prompt is waited for, however
    /// long that takes in all.
    #[tokio::test]
    async fn keepalive_comments_hold_off_the_idle_timeout() {
        let started = std::time::Instant::now();
        let (result, deltas, served) = stream_with_silence(vec![format!("KEEPALIVE:{FINISHED}")]).await;
        assert_eq!(result.expect("a kept-alive stream finishes").content, "all of it");
        assert_eq!((served, deltas), (1, vec!["content:all of it".to_string()]));
        assert!(started.elapsed() > SILENT_FOR * 2, "the stream outlasted the idle interval");
    }

    /// A reply for [`spawn_recording_once`]: the finished stream.
    fn finished_response() -> String {
        format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{FINISHED}")
    }

    /// The bug this guards: a provider without a key of its own was sent the
    /// key settings.json holds for settings.base_url, whatever host the
    /// provider named. It inherits that key only when it is the same server
    /// (scheme, host, and port); any other gets no key, and the 401 that
    /// follows says how to give it its own.
    #[tokio::test]
    async fn a_settings_key_reaches_only_its_own_host() {
        let dir = std::env::temp_dir().join(format!("openmax-keys-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let provider_at = |url: &str| {
            let catalog = json!({"providers": {"local": {"base_url": url, "models": [{"id": "m", "context_tokens": 8192}]}}});
            std::fs::write(crate::providers::providers_path(&dir), catalog.to_string()).unwrap();
        };
        let mut settings = crate::config::Settings {
            provider: Some("local".into()),
            model: "m".into(),
            base_url: "https://api.example.com/v1".into(),
            api_key: Some("sk-settings".into()),
            ..Default::default()
        };
        let request = |endpoint: &crate::providers::ActiveEndpoint| {
            let client = ChatClient::from_endpoint(endpoint);
            async move {
                client
                    .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |_| {})
                    .await
            }
        };

        let refusal = r#"{"error":{"message":"missing credentials"}}"#;
        let (url, heads) = spawn_recording_once(format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{refusal}",
            refusal.len()
        ));
        provider_at(&url);
        let endpoint = crate::providers::resolve(&settings, &dir).unwrap();
        let err = request(&endpoint).await.err().expect("a 401 is an error");
        let head = heads.recv().unwrap().to_ascii_lowercase();
        assert!(!head.contains("authorization") && !head.contains("sk-settings"), "{head}");
        assert!(err.starts_with("backend returned 401 Unauthorized: missing credentials"), "{err}");
        assert!(err.contains("provider 'local'") && err.contains("api_key_env"), "{err}");

        // The same scheme, host, and port is the same server, whatever the path.
        let (url, heads) = spawn_recording_once(finished_response());
        provider_at(&url);
        settings.base_url = url.trim_end_matches("/v1").into();
        let endpoint = crate::providers::resolve(&settings, &dir).unwrap();
        assert_eq!(request(&endpoint).await.unwrap().content, "all of it");
        let head = heads.recv().unwrap().to_ascii_lowercase();
        assert!(head.contains("authorization: bearer sk-settings"), "{head}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The bug this guards: the key went out as a bearer header over plain
    /// http to any host, readable anywhere on the path. A request that would
    /// carry a credential (the key, or an Authorization header) over http to
    /// another machine is refused before anything is sent, naming the fix;
    /// over loopback, to a server on this machine, it goes out.
    #[tokio::test]
    async fn a_key_never_crosses_plain_http_to_another_machine() {
        for (key, headers) in [
            (Some("sk-test".to_string()), Vec::new()),
            (None, vec![("Authorization".to_string(), "Bearer sk-test".to_string())]),
        ] {
            let mut client = ChatClient::new("http://models.example.invalid/v1".into(), key, "m".into(), None, 64);
            client.headers = headers;
            let mut retries = 0;
            let err = client
                .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |_| retries += 1)
                .await
                .err()
                .expect("a key over plain http to another machine is refused");
            assert!(err.starts_with("refusing to send"), "{err}");
            assert!(err.contains("models.example.invalid") && err.contains("https") && err.contains("127.0.0.1"), "{err}");
            assert_eq!(retries, 0, "nothing was sent, so nothing was resent");
        }

        let (url, heads) = spawn_recording_once(finished_response());
        let result = ChatClient::new(url, Some("sk-local".into()), "m".into(), None, 64)
            .stream_chat(&[ChatMessage::user("hi")], "[]", Arc::new(crate::state::CancelToken::default()), |_| {})
            .await
            .expect("loopback http carries a key");
        assert_eq!(result.content, "all of it");
        let head = heads.recv().unwrap().to_ascii_lowercase();
        assert!(head.contains("authorization: bearer sk-local"), "{head}");
    }

    /// https goes anywhere; plain http only to this machine, however its
    /// address is spelled.
    #[test]
    fn only_https_or_this_machine_may_carry_a_key() {
        for url in [
            "https://api.example.com/v1",
            "http://127.0.0.1:11434/v1",
            "http://127.8.9.10/v1",
            "http://localhost:1234/v1",
            "http://LOCALHOST/v1",
            "http://[::1]:8080/v1",
            "http://[::ffff:127.0.0.1]/v1",
            "http://0.0.0.0:8000/v1",
        ] {
            assert_eq!(plain_http_refusal(url), None, "{url}");
        }
        for url in [
            "http://api.example.com/v1",
            "http://192.168.1.5:8000/v1",
            "http://10.0.0.2/v1",
            "http://[2001:db8::1]/v1",
            "http://localhost.example.com/v1",
        ] {
            assert!(plain_http_refusal(url).is_some(), "{url}");
        }
    }

    /// A provider's `idle_timeout_secs` replaces the default interval for
    /// its requests; a provider without one keeps the default.
    #[test]
    fn a_providers_idle_timeout_secs_sets_its_interval() {
        let dir = std::env::temp_dir().join(format!("openmax-idle-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let catalog = json!({"providers": {
            "slow": {"base_url": "http://127.0.0.1:8080/v1", "idle_timeout_secs": 1800, "models": [{"id": "m", "context_tokens": 8192}]},
            "plain": {"base_url": "http://127.0.0.1:8081/v1", "models": [{"id": "m", "context_tokens": 8192}]},
        }});
        std::fs::write(crate::providers::providers_path(&dir), catalog.to_string()).unwrap();
        let idle = |provider: &str| {
            let settings = crate::config::Settings { provider: Some(provider.into()), model: "m".into(), ..Default::default() };
            ChatClient::from_endpoint(&crate::providers::resolve(&settings, &dir).unwrap()).idle_timeout
        };
        assert_eq!(idle("slow"), std::time::Duration::from_secs(1800));
        assert_eq!(idle("plain"), IDLE_TIMEOUT);
        let _ = std::fs::remove_dir_all(dir);
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
        foreign[1].reasoning_details = Some(RawValue::from_string(r#"[{"type":"reasoning.encrypted","data":"x"}]"#.into()).unwrap());
        foreign[1].tool_calls.as_mut().unwrap()[0].extra_content = Some(RawValue::from_string(r#"{"google":{"thought_signature":"s"}}"#.into()).unwrap());
        foreign[1].reasoning_origin = Some("elsewhere".into());
        assert_eq!(wire(&foreign, "o"), old_shape);

        let old: ChatMessage = serde_json::from_str(r#"{"role":"assistant","content":"hi"}"#).unwrap();
        assert!(old.reasoning_content.is_none() && old.reasoning.is_none() && old.reasoning_origin.is_none());
        assert!(old.reasoning_details.is_none());
        let old: ToolCall = serde_json::from_str(r#"{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}"#).unwrap();
        assert!(old.extra_content.is_none());
    }

    /// Gemini streams each call whole, without an `index`, and puts its
    /// thought signature in the call's `extra_content`; its next request
    /// fails without it. The value comes back on that call exactly as it
    /// arrived, through the session file too, to the endpoint that produced
    /// it and to no other. A call beside it without one keeps its old bytes.
    #[tokio::test]
    async fn a_calls_extra_content_goes_back_as_it_came_to_its_endpoint_only() {
        let result = stream_once(concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"extra_content\": {\"google\": {\"thought_signature\": \"sig/A==\"}},\"function\":{\"arguments\":\"{}\",\"name\":\"bash\"},\"id\":\"c1\",\"type\":\"function\"}]},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c2\",\"function\":{\"name\":\"bash\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"stop\"}]}\n\n",
        ))
        .await;
        assert_eq!(result.finish_reason, "tool_calls");
        let signature = r#"{"google": {"thought_signature": "sig/A=="}}"#;
        assert_eq!(result.tool_calls[0].extra_content.as_deref().map(RawValue::get), Some(signature));
        assert!(result.tool_calls[1].extra_content.is_none());

        let client = |base: &str| ChatClient::new(base.into(), None, "m".into(), None, 64);
        let (signing, other) = (client("http://signing/v1"), client("http://other/v1"));
        let mut reply = ChatMessage::assistant(None, Some(result.tool_calls));
        reply.reasoning_origin = Some(signing.origin());
        let signed = concat!(
            r#"[{"role":"assistant","tool_calls":["#,
            r#"{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"},"extra_content":{"google": {"thought_signature": "sig/A=="}}},"#,
            r#"{"id":"c2","type":"function","function":{"name":"bash","arguments":"{}"}}]}]"#,
        );
        let unsigned = concat!(
            r#"[{"role":"assistant","tool_calls":["#,
            r#"{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}},"#,
            r#"{"id":"c2","type":"function","function":{"name":"bash","arguments":"{}"}}]}]"#,
        );
        assert_eq!(wire(std::slice::from_ref(&reply), &signing.origin()), signed);
        assert_eq!(wire(std::slice::from_ref(&reply), &other.origin()), unsigned);

        // The session file holds the same bytes, and a resumed session sends them.
        let line = serde_json::to_string(&reply).unwrap();
        assert!(line.contains(&format!(r#""extra_content":{signature}"#)), "{line}");
        let resumed: ChatMessage = serde_json::from_str(&line).unwrap();
        assert_eq!(wire(&[resumed], &signing.origin()), signed);
        // Unstamped, it goes nowhere; it rides every request, so it is counted.
        let mut unstamped = reply.clone();
        unstamped.reasoning_origin = None;
        assert_eq!(wire(std::slice::from_ref(&unstamped), &signing.origin()), unsigned);
        unstamped.tool_calls.as_mut().unwrap()[0].extra_content = None;
        assert!(reply.estimated_tokens() > unstamped.estimated_tokens());

        // A one-shot JSON reply keeps it too; a null is no value.
        let one_shot = json!({"choices":[{"message":{"tool_calls":[
            {"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig"}}},
            {"id":"c2","type":"function","function":{"name":"bash","arguments":"{}"},"extra_content":null},
        ]},"finish_reason":"tool_calls"}]});
        let calls = parse_complete_response(&one_shot, &mut |_| {}).unwrap().tool_calls;
        assert_eq!(calls[0].extra_content.as_deref().map(RawValue::get), Some(r#"{"google":{"thought_signature":"sig"}}"#));
        assert!(calls[1].extra_content.is_none());
    }

    /// OpenRouter streams a signed model's `reasoning_details` in parts: a
    /// block's text in pieces under one `index`, its signature null until a
    /// later part. Those parts come back as one entry per block, with the
    /// signature; encrypted blobs, and a block with another `id`, stay
    /// separate entries even under a shared index, since joining them would
    /// corrupt them. An array the server sent empty carries no signature and
    /// is no details at all, as if absent; a key of another shape is ignored
    /// without losing the chunk.
    #[tokio::test]
    async fn streamed_reasoning_details_are_merged_by_index() {
        let detail = |parts: &str| format!("data: {{\"choices\":[{{\"delta\":{{\"reasoning_details\":{parts}}},\"finish_reason\":null}}]}}\n\n");
        let sse = [
            detail(r#"[{"type":"reasoning.summary","summary":"plan ","index":0},{"type":"reasoning.text","text":"think ","signature":null,"id":"t1","index":1}]"#),
            detail(r#"[{"type":"reasoning.summary","summary":"done","index":0},{"type":"reasoning.text","text":"more","signature":null,"index":1}]"#),
            detail(r#"[{"type":"reasoning.text","text":"","signature":"sig1","index":1},{"type":"reasoning.encrypted","data":"blobA","id":"e1","index":2},{"type":"reasoning.encrypted","data":"blobB","id":"e2","index":2}]"#),
            detail(r#"[{"type":"reasoning.text","text":"second","signature":"sig2","id":"t2","index":1}]"#),
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\",\"reasoning_details\":{\"odd\":true}},\"finish_reason\":\"stop\"}]}\n\n".into(),
        ]
        .concat();
        let result = stream_once(&sse).await;
        assert_eq!(result.content, "ok", "a key of another shape must not cost the chunk its content");
        let details: Value = serde_json::from_str(result.reasoning_details.as_deref().expect("the server sent details").get()).unwrap();
        assert_eq!(
            details,
            json!([
                {"type": "reasoning.summary", "summary": "plan done", "index": 0},
                {"type": "reasoning.text", "text": "think more", "signature": "sig1", "id": "t1", "index": 1},
                {"type": "reasoning.encrypted", "data": "blobA", "id": "e1", "index": 2},
                {"type": "reasoning.encrypted", "data": "blobB", "id": "e2", "index": 2},
                {"type": "reasoning.text", "text": "second", "signature": "sig2", "id": "t2", "index": 1},
            ])
        );

        // OpenRouter sends an empty array on every delta of a model that
        // does not reason.
        let empty = stream_once(&[detail("[]"), detail("[]"), "data: {\"choices\":[{\"delta\":{\"content\":\"ok\",\"reasoning\":null,\"reasoning_details\":[]},\"finish_reason\":\"stop\"}]}\n\n".into()].concat()).await;
        assert_eq!(empty.content, "ok");
        assert!(empty.reasoning_details.is_none(), "{:?}", empty.reasoning_details);
        let absent = stream_once("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n").await;
        assert!(absent.reasoning_details.is_none());

        // A one-shot JSON reply keeps its array as sent, unless it is empty.
        let one_shot = |details: Value| {
            let reply = json!({"choices":[{"message":{"content":"ok","reasoning_details":details}}]});
            parse_complete_response(&reply, &mut |_| {}).unwrap().reasoning_details.map(|d| d.get().to_string())
        };
        assert_eq!(one_shot(json!([{"data": "x", "type": "reasoning.encrypted"}])).as_deref(), Some(r#"[{"data":"x","type":"reasoning.encrypted"}]"#));
        assert_eq!(one_shot(json!([])), None);
        assert_eq!(one_shot(Value::Null), None);
        assert_eq!(one_shot(json!("text")), None);

        // They go back, as stored, beside the plain text, to their endpoint only.
        let origin = ChatClient::new("http://a/v1".into(), None, "m".into(), None, 64).origin();
        let mut reply = ChatMessage::assistant(Some("ok".into()), None);
        reply.reasoning = Some("think more".into());
        reply.reasoning_details = result.reasoning_details;
        reply.reasoning_origin = Some(origin.clone());
        let stored = reply.reasoning_details.as_deref().unwrap().get().to_string();
        assert_eq!(
            wire(std::slice::from_ref(&reply), &origin),
            format!(r#"[{{"role":"assistant","content":"ok","reasoning":"think more","reasoning_details":{stored}}}]"#)
        );
        assert_eq!(wire(std::slice::from_ref(&reply), "elsewhere"), r#"[{"role":"assistant","content":"ok"}]"#);
        let mut without = reply.clone();
        without.reasoning_details = None;
        assert!(reply.estimated_tokens() > without.estimated_tokens(), "they ride every request, so they are counted");
    }

    /// The server decides how many `reasoning_details` parts a stream has, so
    /// a part must find the entry it continues without scanning every entry
    /// kept so far: a stream of parts that each start a new block (a fresh
    /// `index`, or none) would otherwise cost time quadratic in its length
    /// and stall the turn. Each such part stays its own entry, and a late
    /// part of an early block still finds that block.
    #[test]
    fn many_reasoning_detail_blocks_each_keep_their_own_entry() {
        let blocks = 10_000u64;
        let (mut details, mut open) = (Vec::new(), OpenDetails::new());
        for index in 0..blocks {
            merge_reasoning_detail(&mut details, &mut open, json!({"type": "reasoning.text", "text": "a", "index": index}));
            merge_reasoning_detail(&mut details, &mut open, json!({"type": "reasoning.summary", "summary": "s"}));
        }
        merge_reasoning_detail(&mut details, &mut open, json!({"type": "reasoning.text", "text": "b", "signature": "sig", "index": 0}));
        let mut expected: Vec<Value> = (0..blocks)
            .flat_map(|index| [json!({"type": "reasoning.text", "text": "a", "index": index}), json!({"type": "reasoning.summary", "summary": "s"})])
            .collect();
        expected[0] = json!({"type": "reasoning.text", "text": "ab", "signature": "sig", "index": 0});
        assert_eq!(details.len(), expected.len(), "a part with a new index, or none, starts its own entry");
        assert!(details == expected, "each block keeps its own entry, and a late part continues its block");
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
            extra_content: None,
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
