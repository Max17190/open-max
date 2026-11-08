//! Native Messages translation; tool execution and trust stay in the agent loop.
use super::*;
use serde_json::json;
use ring::digest::{Context, SHA256};

pub(super) fn serialize(
    client: &ChatClient,
    messages: &[ChatMessage],
    tools: &str,
) -> Result<Vec<u8>, String> {
    let origin = client.origin();
    let schemas: Vec<Value> = serde_json::from_str(tools).map_err(|_| "invalid frozen tools")?;
    let mut functions = Vec::new();
    for schema in schemas {
        let function = &schema["function"];
        let name = function["name"]
            .as_str()
            .filter(|name| valid_name(name))
            .ok_or("invalid Messages tool name")?;
        if !function["parameters"].is_object() {
            return Err("invalid Messages tool schema".into());
        }
        let mut tool = json!({"name":name,"input_schema":function["parameters"]});
        if let Some(description) = function.get("description") {
            tool["description"] = description.clone();
        }
        functions.push(tool);
    }
    let system: Vec<Value> = messages.iter().filter(|m| m.role == "system")
        .filter_map(|m| m.content.as_deref().filter(|text| !text.is_empty()))
        .map(|text| json!({"type":"text","text":text})).collect();
    let mut prefix = prefix_seed(&json!(system), &json!(functions));
    let mut input: Vec<Value> = Vec::new();
    for message in messages {
        let text = message.content.clone().unwrap_or_default();
        if message.role == "system" { continue; }
        let role = if message.role == "tool" {
            "user"
        } else {
            &message.role
        };
        if !matches!(role, "user" | "assistant") {
            return Err("unsupported Messages role".into());
        }
        let mut blocks = Vec::new();
        if role == "assistant" && message.reasoning_origin.as_deref() == Some(&origin) {
            if let Some(raw) = &message.reasoning_details {
                let stored: Value = serde_json::from_str(raw.get()).map_err(|_| "invalid stored Messages content")?;
                // The running hash excludes the last message, which user/tool
                // grouping can still extend. Hash that one only in the snapshot.
                let mut at = prefix.clone();
                if let Some(last) = input.last() { hash_message(&mut at, last); }
                if stored["prefix"] == crate::ledger::hex(at.finish().as_ref()) {
                    if let Some(content) = stored["content"].as_array().filter(|content| visible_matches(content, message)) {
                        validate_blocks(content)?;
                        blocks = content.clone();
                    }
                }
            }
        }
        if blocks.is_empty() {
            if message.role == "tool" {
                let id = message
                    .tool_call_id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .ok_or("Messages tool result lacks call ID")?;
                blocks.push(json!({"type":"tool_result","tool_use_id":id,"content":text}));
            } else {
                if !text.is_empty() {
                    blocks.push(json!({"type":"text","text":text}));
                }
                if let Some(calls) = &message.tool_calls {
                    for call in calls {
                        let arguments: Value = serde_json::from_str(&call.function.arguments)
                            .map_err(|_| "invalid Messages tool arguments")?;
                        if !arguments.is_object() {
                            return Err("Messages tool arguments must be an object".into());
                        }
                        blocks.push(json!({"type":"tool_use","id":call.id,"name":call.function.name,"input":arguments}));
                    }
                }
            }
        }
        if !blocks.is_empty() {
            if role == "user" && input.last().is_some_and(|m| m["role"] == role) {
                input.last_mut().unwrap()["content"]
                    .as_array_mut()
                    .unwrap()
                    .extend(blocks);
            } else {
                if let Some(last) = input.last() { hash_message(&mut prefix, last); }
                input.push(json!({"role":role,"content":blocks}));
            }
        }
    }
    // The cache is asked for at two markers: the end of the system prompt
    // (tools and system, the prefix every request and session shares) and
    // the newest block, so each request reads what the one before it wrote.
    // Thinking blocks take no marker.
    let tail = input.last_mut().and_then(|m| m["content"].as_array_mut()?.last_mut());
    if let Some(block) = tail.filter(|b| matches!(b["type"].as_str(), Some("text" | "tool_use" | "tool_result"))) {
        block["cache_control"] = json!({"type":"ephemeral"});
    }
    let mut body =
        json!({"model":client.model,"max_tokens":client.max_tokens,"stream":true,"messages":input});
    if !system.is_empty() {
        body["system"] = json!(system);
        body["system"][system.len() - 1]["cache_control"] = json!({"type":"ephemeral"});
    }
    if !functions.is_empty() {
        body["tools"] = json!(functions);
        body["tool_choice"] = json!({"type":"auto"});
    }
    if let Some(temperature) = client.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(effort) = client.reasoning_effort {
        body["output_config"] = json!({"effort":effort});
    }
    serde_json::to_vec(&body).map_err(|_| "cannot serialize Messages request".into())
}


// Compact JSON contains no literal newlines, so each framed value is unambiguous.
fn hash_message(hash: &mut Context, value: &Value) {
    hash.update(value.to_string().as_bytes());
    hash.update(b"\n");
}

fn prefix_seed(system: &Value, tools: &Value) -> Context {
    let mut hash = Context::new(&SHA256);
    hash.update(b"native-prefix-v1\n");
    hash_message(&mut hash, system);
    hash_message(&mut hash, tools);
    hash
}

pub(super) fn stamp(reply: &mut CompletionResult, request: &[u8]) -> Result<(), String> {
    let Some(content) = reply.reasoning_details.take() else { return Ok(()); };
    let mut body: Value = serde_json::from_slice(request).map_err(|_| "invalid Messages request")?;
    // Cache markers are request metadata, not content, and the next request
    // moves the tail one: the hash that binds signed thinking leaves them out.
    unmark(body.get_mut("system"));
    for message in body["messages"].as_array_mut().ok_or("invalid Messages request history")? { unmark(message.get_mut("content")); }
    let empty = json!([]);
    let mut hash = prefix_seed(body.get("system").unwrap_or(&empty), body.get("tools").unwrap_or(&empty));
    for message in body["messages"].as_array().ok_or("invalid Messages request history")? { hash_message(&mut hash, message); }
    let content: Value = serde_json::from_str(content.get()).map_err(|_| "invalid Messages content")?;
    reply.reasoning_details = Some(serde_json::value::to_raw_value(&json!({"prefix":crate::ledger::hex(hash.finish().as_ref()),"content":content}))
        .map_err(|_| "cannot preserve Messages content")?);
    Ok(())
}

fn unmark(blocks: Option<&mut Value>) {
    for block in blocks.and_then(Value::as_array_mut).into_iter().flatten().filter_map(Value::as_object_mut) {
        block.remove("cache_control");
    }
}

fn visible_matches(blocks: &[Value], message: &ChatMessage) -> bool {
    let text: String = blocks.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect();
    let calls: Vec<_> = blocks.iter().filter(|b| b["type"] == "tool_use").collect();
    let expected = message.tool_calls.as_deref().unwrap_or(&[]);
    text == message.content.as_deref().unwrap_or("") && calls.len() == expected.len()
        && calls.iter().zip(expected).all(|(block, call)| block["id"] == call.id && block["name"] == call.function.name
            && serde_json::from_str::<Value>(&call.function.arguments).is_ok_and(|arguments| arguments == block["input"]))
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn validate_blocks(blocks: &[Value]) -> Result<(), String> {
    let mut ids = std::collections::HashSet::new();
    for block in blocks {
        let valid = match block["type"].as_str() {
            Some("text") => block["text"].is_string(),
            Some("thinking") => {
                block["thinking"].is_string()
                    && block["signature"].as_str().is_some_and(|s| !s.is_empty())
            }
            Some("redacted_thinking") => block["data"].as_str().is_some_and(|s| !s.is_empty()),
            Some("tool_use") => {
                block["id"]
                    .as_str()
                    .is_some_and(|id| !id.is_empty() && ids.insert(id))
                    && block["name"].as_str().is_some_and(valid_name)
                    && block["input"].is_object()
            }
            _ => false,
        };
        if !valid {
            return Err("invalid or unsupported Messages content block".into());
        }
    }
    if ids.len() > MAX_TOOL_CALLS {
        return Err("Messages exceeded the tool-call limit".into());
    }
    Ok(())
}

fn complete(blocks: Vec<Value>, reason: &str, usage: Value) -> Result<CompletionResult, String> {
    validate_blocks(&blocks)?;
    let mut reply = cancelled_response();
    reply.finish_reason = match reason {
        "end_turn" | "stop_sequence" | "refusal" => "stop",
        "tool_use" => "tool_calls",
        "max_tokens" | "model_context_window_exceeded" => "length",
        _ => return Err("Messages returned an unsupported stop reason".into()),
    }
    .into();
    for block in &blocks {
        match block["type"].as_str() {
            Some("text") => reply.content.push_str(block["text"].as_str().unwrap()),
            Some("tool_use") => reply.tool_calls.push(ToolCall {
                id: block["id"].as_str().unwrap().into(),
                kind: "function".into(),
                function: ToolCallFunction {
                    name: block["name"].as_str().unwrap().into(),
                    arguments: block["input"].to_string(),
                },
                extra_content: None,
            }),
            _ => {}
        }
    }
    if reason == "tool_use" && reply.tool_calls.is_empty() {
        return Err("Messages stopped for tool use without a call".into());
    }
    if reason != "tool_use" && !reply.tool_calls.is_empty() {
        return Err("Messages returned tool calls without a tool-use stop".into());
    }
    if blocks
        .iter()
        .any(|b| matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking")))
    {
        reply.reasoning_details = Some(
            serde_json::value::to_raw_value(&blocks).map_err(|_| "invalid Messages reasoning")?,
        );
    }
    reply.usage = parse_usage(&usage);
    Ok(reply)
}

struct Block {
    value: Value,
    arguments: String,
    closed: bool,
}

pub(super) async fn read(
    resp: reqwest::Response,
    cancelled: &crate::state::CancelToken,
    idle: std::time::Duration,
    on_delta: &mut impl FnMut(StreamDelta),
) -> (CompletionResult, Option<Unfinished>) {
    let mut reply = cancelled_response();
    reply.finish_reason = TRUNCATED.into();
    let request_id = resp
        .headers()
        .get("request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let failed = |message: String| {
        Some(Unfinished::Failed(ServerFailure {
            message,
            retryable: false,
        }))
    };
    let failure = |error: &Value| {
        let api_error = error["type"] == "api_error";
        let mut error = server_failure(error, &error.to_string());
        error.retryable |= api_error;
        if let Some(id) = &request_id {
            error
                .message
                .push_str(&format!(" (request ID: {})", truncate(id, 120)));
        }
        Some(Unfinished::Failed(error))
    };
    if resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("application/json"))
    {
        return match read_body(resp, cancelled, idle).await {
            Ok(None) => (cancelled_response(), None),
            Err(error) => (reply, failed(error)),
            Ok(Some(bytes)) => {
                let value: Value = match serde_json::from_slice(&bytes) {
                    Ok(v) => v,
                    Err(_) => return (reply, failed("invalid Messages JSON".into())),
                };
                if let Some(error) = value.get("error") {
                    return (reply, failure(error));
                }
                if value["type"] != "message" || value["role"] != "assistant" {
                    return (reply, failed("invalid Messages response envelope".into()));
                }
                let result = value["content"]
                    .as_array()
                    .ok_or_else(|| "Messages response lacks content".into())
                    .and_then(|blocks| {
                        complete(
                            blocks.clone(),
                            value["stop_reason"].as_str().unwrap_or(""),
                            value["usage"].clone(),
                        )
                    });
                match result {
                    Ok(done) => {
                        if !done.content.is_empty() {
                            on_delta(StreamDelta::Content(done.content.clone()));
                        }
                        if cancelled.is_cancelled() {
                            reply.content = done.content;
                            reply.finish_reason = "cancelled".into();
                            return (reply, None);
                        }
                        (done, None)
                    }
                    Err(error) => (reply, failed(error)),
                }
            }
        };
    }
    let mut stream = resp.bytes_stream();
    let mut buffer = Vec::new();
    let mut data = Vec::new();
    let mut received = 0;
    let mut started = false;
    let mut blocks: Vec<Block> = Vec::new();
    let mut reason = None;
    let mut usage = json!({});
    loop {
        let next = tokio::select! {
            next = tokio::time::timeout(idle, stream.next()) => next,
            _ = cancelled.cancelled() => { reply.finish_reason = "cancelled".into(); return (reply, None); }
        };
        let chunk = match next {
            Ok(Some(Ok(chunk))) => chunk,
            _ => {
                return (
                    reply,
                    Some(Unfinished::Interrupted(
                        "Messages stream ended before message_stop".into(),
                    )),
                )
            }
        };
        if chunk.len() > MAX_RESPONSE_BYTES - received {
            return (reply, None);
        }
        received += chunk.len();
        buffer.extend_from_slice(&chunk);
        let complete_end = buffer.iter().rposition(|b| *b == b'\n').map_or(0, |at| at + 1);
        for line in buffer[..complete_end].split_inclusive(|b| *b == b'\n') {
            let line = trim_bytes(line);
            if !line.is_empty() {
                if let Some(bytes) = data_line(line) {
                    if !data.is_empty() {
                        data.push(b'\n');
                    }
                    data.extend_from_slice(bytes);
                }
                continue;
            }
            if data.is_empty() {
                continue;
            }
            let event: Value = match serde_json::from_slice(&data) {
                Ok(v) => v,
                Err(_) => return (reply, failed("invalid Messages event".into())),
            };
            data.clear();
            let applied: Result<(), String> = (|| {
                match event["type"].as_str() {
                    Some("message_start") => {
                        if started
                            || event["message"]["role"] != "assistant"
                            || event["message"]["content"]
                                .as_array()
                                .is_none_or(|a| !a.is_empty())
                        {
                            return Err("invalid Messages message_start".into());
                        }
                        started = true;
                        usage = event["message"]["usage"].clone();
                    }
                    Some("content_block_start") => {
                        if !started
                            || reason.is_some()
                            || event["index"].as_u64() != Some(blocks.len() as u64)
                            || blocks.last().is_some_and(|b| !b.closed)
                            || blocks.len() >= MAX_TOOL_CALLS * 4
                        {
                            return Err("invalid Messages block index or order".into());
                        }
                        let block = event["content_block"].clone();
                        match block["type"].as_str() {
                            Some("text") => {
                                let text = block["text"].as_str().ok_or("invalid Messages text")?;
                                reply.content.push_str(text);
                                if !text.is_empty() {
                                    on_delta(StreamDelta::Content(text.into()));
                                }
                            }
                            Some("thinking") => {
                                if !block["thinking"].is_string() {
                                    return Err("invalid Messages thinking".into());
                                }
                            }
                            Some("redacted_thinking" | "tool_use") => {}
                            _ => return Err("unsupported Messages content block".into()),
                        }
                        blocks.push(Block {
                            value: block,
                            arguments: String::new(),
                            closed: false,
                        });
                    }
                    Some("content_block_delta" | "content_block_stop") => {
                        let index = event["index"]
                            .as_u64()
                            .and_then(|n| usize::try_from(n).ok())
                            .ok_or("invalid Messages block index")?;
                        let block = blocks
                            .get_mut(index)
                            .filter(|b| !b.closed)
                            .ok_or("Messages delta for absent or closed block")?;
                        if event["type"] == "content_block_stop" {
                            if block.value["type"] == "tool_use" && !block.arguments.is_empty() {
                                block.value["input"] = serde_json::from_str(&block.arguments)
                                    .map_err(|_| "invalid Messages tool JSON")?;
                            }
                            validate_blocks(std::slice::from_ref(&block.value))?;
                            block.closed = true;
                        } else {
                            let delta = &event["delta"];
                            let (key, value) =
                                match (block.value["type"].as_str(), delta["type"].as_str()) {
                                    (Some("text"), Some("text_delta")) => {
                                        ("text", delta["text"].as_str())
                                    }
                                    (Some("thinking"), Some("thinking_delta")) => {
                                        ("thinking", delta["thinking"].as_str())
                                    }
                                    (Some("thinking"), Some("signature_delta")) => {
                                        ("signature", delta["signature"].as_str())
                                    }
                                    (Some("tool_use"), Some("input_json_delta")) => {
                                        ("input", delta["partial_json"].as_str())
                                    }
                                    _ => return Err("unsupported Messages block delta".into()),
                                };
                            let value = value.ok_or("invalid Messages delta value")?;
                            if key == "input" {
                                block.arguments.push_str(value);
                            } else {
                                if block.value[key].is_null() { block.value[key] = json!(""); }
                                let Value::String(text) = &mut block.value[key] else { return Err("invalid Messages block field".into()); };
                                text.push_str(value);
                                if key == "text" {
                                    reply.content.push_str(value);
                                    on_delta(StreamDelta::Content(value.into()));
                                }
                                if key == "thinking" {
                                    on_delta(StreamDelta::Reasoning(value.into()));
                                }
                            }
                        }
                    }
                    Some("message_delta") => {
                        if !started || blocks.iter().any(|b| !b.closed) {
                            return Err("Messages ended with an open block".into());
                        }
                        if let Some(stop) = event["delta"]["stop_reason"].as_str() {
                            reason = Some(stop.to_owned());
                        }
                        if let Some(counts) = event["usage"].as_object() {
                            if !usage.is_object() {
                                usage = json!({});
                            }
                            usage.as_object_mut().unwrap().extend(counts.clone());
                        }
                    }
                    _ => {}
                }
                Ok(())
            })();
            if cancelled.is_cancelled() {
                reply.finish_reason = "cancelled".into();
                return (reply, None);
            }
            if let Err(error) = applied {
                return (reply, failed(error));
            }
            if event["type"] == "error" {
                return (reply, failure(&event["error"]));
            }
            if event["type"] == "message_stop" {
                if !started || blocks.iter().any(|b| !b.closed) {
                    return (
                        reply,
                        failed("Messages stopped before blocks completed".into()),
                    );
                }
                return match complete(
                    blocks.into_iter().map(|b| b.value).collect(),
                    reason.as_deref().unwrap_or(""),
                    usage,
                ) {
                    Ok(done) => (done, None),
                    Err(error) => (reply, failed(error)),
                };
            }
        }
        buffer.drain(..complete_end);
    }
}
