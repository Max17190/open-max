//! Responses wire translation. Tool execution stays in the existing agent loop.
use super::*;
use crate::types::ToolCallFunction;
use serde_json::json;

pub(super) fn serialize(
    client: &ChatClient,
    messages: &[ChatMessage],
    note: Option<&str>,
    tools: &str,
) -> Result<Vec<u8>, String> {
    let origin = client.origin();
    let mut input = Vec::new();
    let last = messages.len().saturating_sub(1);
    let fold_note = messages.last().is_some_and(|m| m.role == "user" || m.role == "tool");
    for (index, m) in messages.iter().enumerate() {
        let mut content = m.content.clone().unwrap_or_default();
        if index == last && fold_note {
            if let Some(note) = note {
                content.push_str(NOTE_SEPARATOR);
                content.push_str(note);
            }
        }
        if m.role == "assistant" && m.reasoning_origin.as_deref() == Some(&origin) {
            if let Some(raw) = &m.reasoning_details {
                let items: Vec<Value> =
                    serde_json::from_str(raw.get()).map_err(|_| "invalid stored Responses reasoning")?;
                for item in items {
                    if item["type"] != "reasoning" {
                        return Err("invalid Responses reasoning item".into());
                    }
                    input.push(item);
                }
            }
        }
        if m.role == "tool" {
            input.push(json!({"type":"function_call_output","call_id":m.tool_call_id,"output":content}));
        } else if !content.is_empty() {
            input.push(json!({"role":if m.role == "system" { "developer" } else { &m.role },"content":content}));
        }
        if let Some(calls) = &m.tool_calls {
            for call in calls {
                input.push(json!({"type":"function_call","call_id":call.id,"name":call.function.name,
                    "arguments":call.function.arguments}));
            }
        }
    }
    if !fold_note {
        if let Some(note) = note {
            input.push(json!({"role":"user","content":note}));
        }
    }
    let schemas: Vec<Value> = serde_json::from_str(tools).map_err(|_| "invalid frozen tools")?;
    let mut functions = Vec::new();
    for schema in schemas {
        let mut function = schema["function"].as_object().ok_or("invalid function schema")?.clone();
        function.insert("type".into(), json!("function"));
        // Responses normalizes schemas to strict by default. That is lossless
        // for required scalar fields; optional or complex arguments must opt out.
        let parameters = &function["parameters"];
        let required = parameters["required"].as_array();
        let required_scalars = parameters["properties"].as_object().is_some_and(|properties| {
            properties.iter().all(|(name, property)| {
                required.is_some_and(|required| required.iter().any(|r| r == name))
                    && matches!(property["type"].as_str(), Some("string" | "number" | "integer" | "boolean"))
            })
        }) && parameters.get("additionalProperties").is_none_or(|v| v == false);
        if !required_scalars {
            function.insert("strict".into(), json!(false));
        }
        functions.push(Value::Object(function));
    }
    // This supported input item avoids an extra namespace and leaves tool names
    // unchanged. It joins the frozen prefix immediately after the instructions.
    if !functions.is_empty() {
        let at = usize::from(input.first().is_some_and(|item| item["role"] == "developer"));
        input.insert(at, json!({"type":"additional_tools","role":"developer","tools":functions}));
    }
    let mut body = json!({"model":client.model,"input":input,"store":false,"stream":true,
        "include":["reasoning.encrypted_content"]});
    // Subscription Responses rejects output-token and sampling parameters. The
    // harness's local context reservation and turn-spend admission still apply.
    if client.chatgpt.is_none() {
        body["max_output_tokens"] = json!(client.max_tokens);
        if let Some(t) = client.temperature {
            body["temperature"] = json!(t);
        }
    }
    serde_json::to_vec(&body).map_err(|_| "cannot serialize Responses request".into())
}

pub(super) fn usage_hint(message: String, raw: &str) -> String {
    if raw.contains("subscription_sharing_usage_limit_exceeded") {
        format!("{message}. ChatGPT plan or app limit reached. Manage usage: https://chatgpt.com/settings/usage")
    } else {
        message
    }
}

pub(super) fn plan_limit(raw: &str) -> bool {
    serde_json::from_str::<Value>(raw).is_ok_and(|v| v["error"]["code"] == "subscription_sharing_usage_limit_exceeded")
}

fn failure(value: &Value) -> ServerFailure {
    let raw = value.to_string();
    let mut failure = server_failure(value, &raw);
    if value["code"] == "subscription_sharing_usage_limit_exceeded" {
        failure.retryable = false;
    }
    if matches!(
        value["code"].as_str(),
        Some("subscription_sharing_usage_unavailable" | "subscription_sharing_user_unavailable")
    ) {
        failure.retryable = true;
    }
    failure.message = usage_hint(failure.message, &raw);
    if let Some(param) = value["param"].as_str() {
        failure.message.push_str(&format!(" (param: {})", truncate(param, 120)));
    }
    failure
}

pub(super) fn http_failure(raw: &str, request_id: Option<&str>) -> String {
    let message = match serde_json::from_str::<Value>(raw) {
        Ok(value) if value.get("error").is_some() => failure(&value["error"]).message,
        _ => usage_hint(describe_backend(raw), raw),
    };
    match request_id {
        Some(id) => format!("{message} (request ID: {})", truncate(id, 120)),
        None => message,
    }
}

fn complete(value: &Value) -> Result<CompletionResult, String> {
    if value["status"] != "completed" {
        return Err("Responses did not complete successfully".into());
    }
    let output = value["output"].as_array().ok_or("Responses completed without output")?;
    let mut result = cancelled_response();
    result.finish_reason = "stop".into();
    let mut reasoning = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for item in output {
        if item.get("status").is_some_and(|status| !status.is_null() && status != "completed") {
            return Err("Responses returned an unfinished output item".into());
        }
        match item["type"].as_str() {
            Some("message") => {
                for part in item["content"].as_array().ok_or("invalid Responses message")? {
                    let text = match part["type"].as_str() {
                        Some("output_text") => part["text"].as_str(),
                        Some("refusal") => part["refusal"].as_str(),
                        _ => None,
                    };
                    if let Some(text) = text {
                        result.content.push_str(text);
                    }
                }
            }
            Some("reasoning") => reasoning.push(item.clone()),
            Some("function_call") => {
                if result.tool_calls.len() >= MAX_TOOL_CALLS {
                    return Err("Responses exceeded the tool-call limit".into());
                }
                let field = |key| item[key].as_str().filter(|s| !s.is_empty()).ok_or("invalid Responses function call");
                let id = field("call_id")?;
                if !ids.insert(id) || item.get("namespace").is_some_and(|n| !n.is_null()) {
                    return Err("Responses returned a duplicate call or unknown tool namespace".into());
                }
                result.tool_calls.push(ToolCall {
                    id: id.into(),
                    kind: "function".into(),
                    function: ToolCallFunction { name: field("name")?.into(), arguments: field("arguments")?.into() },
                    extra_content: None,
                });
            }
            _ => return Err("Responses returned an unsupported output item".into()),
        }
    }
    if !result.tool_calls.is_empty() {
        result.finish_reason = "tool_calls".into();
    }
    if !reasoning.is_empty() {
        result.reasoning_details = Some(
            RawValue::from_string(serde_json::to_string(&reasoning).map_err(|_| "invalid reasoning")?)
                .map_err(|_| "invalid reasoning")?,
        );
    }
    if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
        result.usage = Some(Usage {
            prompt_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
            completion_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
            cached_tokens: usage["input_tokens_details"]["cached_tokens"].as_u64(),
        });
    }
    Ok(result)
}

/// Completion is an explicit terminal event. Incomplete or cut streams never dispatch tools.
pub(super) async fn read(
    resp: reqwest::Response,
    cancelled: &crate::state::CancelToken,
    idle: std::time::Duration,
    on_delta: &mut impl FnMut(StreamDelta),
) -> (CompletionResult, Option<Unfinished>) {
    let mut result = cancelled_response();
    result.finish_reason = TRUNCATED.into();
    let request_id = resp.headers().get("x-request-id").and_then(|v| v.to_str().ok()).map(str::to_string);
    let failed = |value: &Value| {
        let mut failure = failure(value);
        if let Some(id) = &request_id {
            failure.message.push_str(&format!(" (request ID: {})", truncate(id, 120)));
        }
        Some(Unfinished::Failed(failure))
    };
    let mut stream = resp.bytes_stream();
    let mut buffer = Vec::new();
    let mut data = Vec::new();
    let mut output = std::collections::BTreeMap::new();
    let mut received = 0;
    loop {
        let next = tokio::select! {
            chunk = tokio::time::timeout(idle, stream.next()) => chunk,
            _ = cancelled.cancelled() => { result.finish_reason = "cancelled".into(); return (result, None); }
        };
        let chunk = match next {
            Ok(Some(Ok(chunk))) => chunk,
            _ => {
                return (
                    result,
                    Some(Unfinished::Interrupted("Responses stream ended before response.completed".into())),
                )
            }
        };
        if chunk.len() > MAX_RESPONSE_BYTES - received {
            return (result, None);
        }
        received += chunk.len();
        buffer.extend_from_slice(&chunk);
        let mut consumed = 0;
        while let Some(at) = buffer[consumed..].iter().position(|b| *b == b'\n') {
            let end = consumed + at;
            let line = trim_bytes(&buffer[consumed..end]);
            consumed = end + 1;
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
            let mut event: Value = match serde_json::from_slice(&data) {
                Ok(v) => v,
                Err(_) => return (result, Some(Unfinished::Interrupted("invalid Responses event".into()))),
            };
            data.clear();
            match event["type"].as_str() {
                Some("response.output_text.delta" | "response.refusal.delta") => {
                    if let Some(text) = event["delta"].as_str() {
                        result.content.push_str(text);
                        on_delta(StreamDelta::Content(text.into()));
                    }
                }
                Some("response.reasoning_summary_text.delta") => {
                    if let Some(text) = event["delta"].as_str() {
                        on_delta(StreamDelta::Reasoning(text.into()));
                    }
                }
                Some("response.output_item.done") => {
                    let Some(index) = event["output_index"].as_u64() else {
                        return (result, failed(&json!({"message":"Responses output item lacks an index"})));
                    };
                    if output.insert(index, event["item"].take()).is_some() {
                        return (result, failed(&json!({"message":"Responses returned a duplicate output index"})));
                    }
                }
                Some("response.completed") => {
                    let mut response = event["response"].take();
                    // Some streaming endpoints omit output from the terminal event.
                    // Finished items remain inert until successful completion arrives.
                    if response["output"].as_array().is_none_or(Vec::is_empty) && !output.is_empty() {
                        if output.keys().enumerate().any(|(expected, actual)| expected as u64 != *actual) {
                            return (result, failed(&json!({"message":"Responses completed with missing output items"})));
                        }
                        response["output"] = Value::Array(output.into_values().collect());
                    }
                    return match complete(&response) {
                        Ok(done) => {
                            if result.content.is_empty() && !done.content.is_empty() {
                                on_delta(StreamDelta::Content(done.content.clone()));
                            }
                            (done, None)
                        }
                        Err(message) => (result, Some(Unfinished::Failed(ServerFailure { message, retryable: false }))),
                    }
                }
                Some("response.failed") => return (result, failed(&event["response"]["error"])),
                Some("error") => return (result, failed(&event)),
                Some("response.incomplete") => return (result, None),
                _ => {}
            }
        }
        if consumed > 0 {
            buffer.drain(..consumed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscription_request_omits_unsupported_options_without_changing_key_requests() {
        let dir = std::env::temp_dir().join(format!("openmax-response-body-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut client = ChatClient::new(crate::chatgpt::API.into(), None, "fixture".into(), Some(0.5), 4096);
        client.wire_api = crate::providers::WireApi::Responses;
        client.chatgpt = Some(crate::chatgpt::tests::seeded(&dir));
        assert!(client.sends_credential());
        let messages = [ChatMessage::system("rules"), ChatMessage::user("task")];
        let body: Value = serde_json::from_slice(&serialize(&client, &messages, None, "[]").unwrap()).unwrap();
        for key in ["max_output_tokens", "temperature", "previous_response_id", "conversation", "background", "tools"] {
            assert!(body.get(key).is_none(), "{key}");
        }
        assert_eq!(body["input"][0]["role"], "developer");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        let origin = client.origin();
        client.chatgpt = None;
        assert_ne!(client.origin(), origin);
        let body: Value = serde_json::from_slice(&serialize(&client, &messages, None, "[]").unwrap()).unwrap();
        assert_eq!(body["max_output_tokens"], 4096);
        assert_eq!(body["temperature"], 0.5);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
