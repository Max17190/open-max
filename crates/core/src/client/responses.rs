//! Responses wire translation. Tool execution stays in the existing agent loop.
use super::*;
use crate::types::ToolCallFunction;
use serde_json::json;

pub(super) fn serialize(
    client: &ChatClient,
    messages: &[ChatMessage],
    tools: &str,
) -> Result<Vec<u8>, String> {
    let (origin, (effort, changes)) = (client.origin(), client.effort_plan(messages));
    let mut input = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let content = m.content.as_deref().unwrap_or_default();
        if let Some((_, level)) = changes.iter().find(|(at, _)| *at == i) {
            input.push(json!({"type":"configuration_update","reasoning":{"effort":level}}));
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
    // A server whose input defines no such item refuses the whole request
    // over it (`providers::TOOLS_FIELD`): there the tools go in `tools`.
    if !functions.is_empty() && !crate::providers::TOOLS_FIELD.iter().any(|(host, prefix)| client.serves(host, prefix)) {
        let at = usize::from(input.first().is_some_and(|item| item["role"] == "developer"));
        input.insert(at, json!({"type":"additional_tools","role":"developer","tools":std::mem::take(&mut functions)}));
    }
    let mut body = json!({"model":client.model,"input":input,"store":false,"stream":true,
        "include":["reasoning.encrypted_content"]});
    if !functions.is_empty() {
        body["tools"] = json!(functions);
    }
    if let Some(effort) = effort {
        body["reasoning"] = json!({"effort": effort});
    }
    // Both OpenAI routes take it: the subscription one refuses
    // `prompt_cache_retention` and `user`, not this.
    if let Some(key) = &client.cache_key {
        body["prompt_cache_key"] = json!(key);
    }
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
                    let key = match part["type"].as_str() { Some("output_text") => "text", Some("refusal") => "refusal", _ => continue };
                    result.content.push_str(part[key].as_str().unwrap_or_default());
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
    result.reasoning_details =
        (!reasoning.is_empty()).then(|| serde_json::value::to_raw_value(&reasoning)).transpose().map_err(|_| "invalid reasoning")?;
    result.usage = parse_usage(&value["usage"]);
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
        let Ok(Some(Ok(chunk))) = next else {
            return (result, Some(Unfinished::Interrupted("Responses stream ended before response.completed".into())));
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
            let Ok(mut event) = serde_json::from_slice::<Value>(&data) else {
                return (result, Some(Unfinished::Interrupted("invalid Responses event".into())));
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
        buffer.drain(..consumed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn endpoint(replies: Vec<(u16, String)>) -> (ChatClient, Arc<std::sync::Mutex<Vec<Vec<u8>>>>, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = ChatClient::new(format!("http://{}", listener.local_addr().unwrap()), None, "fixture".into(), None, 64);
        client.wire_api = crate::providers::WireApi::Responses;
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let job = tokio::spawn(async move {
            for (status, body) in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut buf = [0; 8192];
                    let n = socket.read(&mut buf).await.unwrap();
                    assert_ne!(n, 0);
                    bytes.extend_from_slice(&buf[..n]);
                    if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let len: usize = String::from_utf8_lossy(&bytes[..end]).lines().find_map(|l| {
                            l.to_ascii_lowercase().strip_prefix("content-length:").map(|s| s.trim().parse().unwrap())
                        }).unwrap();
                        if bytes.len() >= end + 4 + len {
                            recorded.lock().unwrap().push(bytes[end + 4..end + 4 + len].to_vec());
                            break;
                        }
                    }
                }
                let header = format!("HTTP/1.1 {status} fixture\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = socket.write_all(header.as_bytes()).await;
                let _ = socket.write_all(body.as_bytes()).await;
            }
        });
        (client, requests, job)
    }

    fn completed(text: &str) -> String {
        format!("data: {}\n\n", json!({"type":"response.completed","response":{"status":"completed","output":[
            {"type":"message","status":"completed","content":[{"type":"output_text","text":text}]}]}}))
    }

    #[tokio::test]
    async fn http_rate_limits_retry_within_budget_and_plan_limits_stop_immediately() {
        for (replies, expected_requests, expected_error) in [
            (vec![(429, json!({"error":{"code":"rate_limit_exceeded","message":"busy"}}).to_string()),
                (200, completed("recovered"))], 2, None),
            (vec![(429, json!({"error":{"code":"subscription_sharing_usage_limit_exceeded","message":"app limit"}}).to_string()),
                (200, completed("unexpected"))], 1, Some("https://chatgpt.com/settings/usage")),
            (vec![(429, json!({"error":{"code":"rate_limit_exceeded","message":"busy"}}).to_string()); 9], 8, Some("backend reported an error (rate_limit_exceeded): busy")),
            (vec![(400, json!({"message":"request rejected"}).to_string())], 1, Some("request rejected")),
            (vec![(200, format!("data: {}\n\n", json!({"type":"error","code":"server_error","message":"busy"}))),
                (200, completed("recovered"))], 2, None),
            (vec![(200, format!("data: {}\n\n", json!({"type":"error","code":"server_error","message":"busy"}))); 9],
                8, Some("backend reported an error (server_error): busy")),
        ] {
            let streamed_error = replies[0].0 == 200;
            let (client, requests, job) = endpoint(replies).await;
            let mut retries = Vec::new();
            let mut reasons = Vec::new();
            let mut visible = String::new();
            let result = client.stream_chat(&[ChatMessage::user("go")], "[]", Arc::new(crate::state::CancelToken::default()), |delta| {
                if let StreamDelta::Retry { attempt, max_attempts, ref reason } = delta {
                    retries.push((attempt, max_attempts));
                    reasons.push(reason.clone());
                }
                if let StreamDelta::Content(text) = delta { visible.push_str(&text); }
            }).await;
            job.abort();
            match expected_error {
                Some(message) => assert!(result.err().unwrap().contains(message)),
                None => {
                    assert_eq!(result.unwrap().content, "recovered");
                    assert_eq!(visible, "recovered");
                }
            }
            let requests = requests.lock().unwrap();
            assert_eq!(requests.len(), expected_requests);
            assert!(requests.windows(2).all(|pair| pair[0] == pair[1]));
            assert_eq!(retries, (2..=expected_requests as u32).map(|attempt| (attempt, 8)).collect::<Vec<_>>());
            if streamed_error { assert!(reasons[0].contains("server_error")); }
        }
        let (client, requests, job) = endpoint(vec![(200, String::new()); 9]).await;
        let mut retries = Vec::new();
        let result = client.stream_chat(&[ChatMessage::user("go")], "[]", Arc::new(crate::state::CancelToken::default()), |delta| {
            if let StreamDelta::Retry { attempt, .. } = delta { retries.push(attempt); }
        }).await.unwrap();
        job.abort();
        assert_eq!(result.finish_reason, TRUNCATED);
        assert_eq!(requests.lock().unwrap().len(), 8);
        assert_eq!(retries, vec![2, 3, 4, 5, 6, 7, 8]);
        for reply in [
            (200, String::new()),
            (200, format!("data: {}\n\n", json!({"type":"error","code":"server_error","message":"busy"}))),
            (429, json!({"error":{"code":"rate_limit_exceeded","message":"busy"}}).to_string()),
        ] {
            let (client, requests, job) = endpoint(vec![reply; 2]).await;
            let cancelled = Arc::new(crate::state::CancelToken::default());
            let result = client.stream_chat(&[ChatMessage::user("go")], "[]", cancelled.clone(), |delta| {
                if matches!(delta, StreamDelta::Retry { .. }) { cancelled.cancel(); }
            }).await.unwrap();
            job.abort();
            assert_eq!(result.finish_reason, "cancelled");
            assert_eq!(requests.lock().unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn streamed_byte_limit_rejects_a_late_completed_call() {
        let first = format!("data: {}\n\n", json!({"type":"response.output_item.done","output_index":0,
            "item":{"type":"function_call","status":"completed","call_id":"call-1","name":"write_file","arguments":"{}"}}));
        let terminal = format!("data: {}\n\n", json!({"type":"response.completed","response":{"status":"completed"}}));
        let comment = format!(":{}\n\n", "x".repeat(4093));
        for excess in [0, 1] {
            let padding = MAX_RESPONSE_BYTES + excess - first.len() - terminal.len();
            let frames = (padding - 3) / comment.len();
            let mut body = first.clone() + &comment.repeat(frames);
            body.push_str(&format!(":{}\n\n", "x".repeat(padding - frames * comment.len() - 3)));
            body.push_str(&terminal);
            assert_eq!(body.len(), MAX_RESPONSE_BYTES + excess);
            let (client, _, job) = endpoint(vec![(200, body)]).await;
            let result = client.stream_chat(&[ChatMessage::user("go")], "[]", Arc::new(crate::state::CancelToken::default()), |_| {}).await.unwrap();
            assert_eq!(result.finish_reason, if excess == 0 { "tool_calls" } else { TRUNCATED });
            assert_eq!(result.tool_calls.len(), usize::from(excess == 0));
            job.await.unwrap();
        }
    }

    #[tokio::test]
    async fn multiline_sse_data_preserves_json_token_boundaries() {
        for (separator, valid) in [("\ndata: ", true), ("1\ndata: 2", false)] {
            let body = format!("data: {}\n\n", json!({"type":"response.completed","response":{"status":"completed",
                "output":[{"type":"message","status":"completed","content":[{"type":"refusal","refusal":"Declined"}]},
                {"type":"function_call","call_id":"call-1","name":"write_file","arguments":"{}"}],
                "usage":{"input_tokens":12,"output_tokens":1}}}));
            let body = if valid {
                body.replace("\"response\":", &format!("\"response\":{separator}"))
            } else {
                body.replace("\"input_tokens\":12", &format!("\"input_tokens\":{separator}"))
            };
            let (client, _, job) = endpoint(vec![(200, body)]).await;
            let response = reqwest::Client::new().post(client.endpoint()).body("{}").send().await.unwrap();
            let (result, unfinished) = read(response, &crate::state::CancelToken::default(), std::time::Duration::from_secs(5), &mut |_| {}).await;
            if valid {
                assert!(unfinished.is_none());
                assert_eq!(result.tool_calls.len(), 1);
                assert_eq!(result.finish_reason, "tool_calls");
                assert_eq!(result.content, "Declined");
                assert_eq!(result.usage.unwrap().prompt_tokens, 12);
            } else {
                assert!(matches!(unfinished, Some(Unfinished::Interrupted(_))));
                assert!(result.tool_calls.is_empty());
            }
            job.await.unwrap();
        }
        let (client, _, job) = endpoint(vec![(200, format!("data: {}\n\n", json!({"type":"response.incomplete"})))]).await;
        let response = reqwest::Client::new().post(client.endpoint()).body("{}").send().await.unwrap();
        let (result, unfinished) = read(response, &crate::state::CancelToken::default(), std::time::Duration::from_secs(5), &mut |_| {}).await;
        assert!(unfinished.is_none());
        assert_eq!(result.finish_reason, TRUNCATED);
        job.await.unwrap();
    }

    #[test]
    fn each_message_goes_as_its_stored_text_and_complex_tool_schemas_keep_their_shape() {
        let client = ChatClient::new("http://127.0.0.1".into(), None, "fixture".into(), None, 64);
        // A prompt cache that ends its entry at the newest user message or
        // tool output holds the stored text, as the next request sends it.
        for (last, key) in [
            (ChatMessage::user("task"), "content"),
            (ChatMessage::tool("call-1", "result"), "output"),
            (ChatMessage::assistant(Some("answer".into()), None), "content"),
        ] {
            let messages = [ChatMessage::system("rules"), last.clone()];
            let body: Value = serde_json::from_slice(&serialize(&client, &messages, "[]").unwrap()).unwrap();
            assert_eq!(body["input"][0]["content"], "rules");
            let input = body["input"].as_array().unwrap();
            assert_eq!(input.len(), 2);
            assert_eq!(input[1][key], last.content.unwrap());
        }
        for parameters in [
            json!({"type":"object","properties":{"value":{"type":"string"}},"required":[],"additionalProperties":false}),
            json!({"type":"object","properties":{"value":{"type":"array","items":{"type":"string"}}},"required":["value"],"additionalProperties":false}),
            json!({"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":true}),
        ] {
            let tools = json!([{"type":"function","function":{"name":"fixture","parameters":parameters}}]);
            let body: Value = serde_json::from_slice(&serialize(&client, &[], &tools.to_string()).unwrap()).unwrap();
            assert_eq!(body["input"][0]["tools"][0]["strict"], false);
            assert_eq!(body["input"][0]["tools"][0]["parameters"], parameters);
        }
        for same_origin in [true, false] {
            let mut message = ChatMessage::assistant(Some("answer".into()), None);
            message.reasoning_details = Some(RawValue::from_string(json!([{"type":"reasoning","encrypted_content":"opaque"}]).to_string()).unwrap());
            message.reasoning_origin = Some(if same_origin { client.origin() } else { "different-origin".into() });
            let body = serialize(&client, &[message], "[]").unwrap();
            assert_eq!(String::from_utf8(body).unwrap().contains("opaque"), same_origin);
        }
    }

    /// The table is pinned: a server joins it only with its host and model
    /// prefix. A request to that host, or for one of its models at any host
    /// (a proxy or gateway in front of it), carries the tools in `tools` and
    /// no `additional_tools` item; any other carries the item and no `tools`.
    /// Either way the same functions go, and none when there are no tools.
    #[test]
    fn a_server_without_the_tools_item_gets_the_tools_field() {
        assert_eq!(crate::providers::TOOLS_FIELD, [("api.x.ai", "grok-")]);
        let fixture = json!([{"type":"function","function":{"name":"fixture","parameters":{"type":"object"}}}]).to_string();
        let body = |url: &str, model: &str, tools: &str| -> Value {
            let client = ChatClient::new(url.into(), None, model.into(), None, 64);
            let messages = [ChatMessage::system("rules"), ChatMessage::user("task")];
            serde_json::from_slice(&serialize(&client, &messages, tools).unwrap()).unwrap()
        };
        let openai = body("https://api.openai.com/v1", "gpt-5.6", &fixture);
        assert_eq!(openai["input"][1]["type"], "additional_tools");
        let functions = &openai["input"][1]["tools"];
        assert_eq!(functions.as_array().unwrap().len(), 1);
        let messages = json!([{"role":"developer","content":"rules"},{"role":"user","content":"task"}]);
        for (url, model) in [("https://api.x.ai/v1", "m"), (" https://API.x.ai/v1/ ", "m"), ("http://127.0.0.1:8080/v1", "grok-4.7"),
            ("https://openrouter.ai/api/v1", "x-ai/Grok-4.7")] {
            let field = body(url, model, &fixture);
            assert_eq!((&field["tools"], &field["input"]), (functions, &messages), "{model} at {url}");
            assert_eq!(body(url, model, "[]").get("tools"), None, "{model} at {url}");
        }
        for (url, model) in [("http://127.0.0.1:8080/v1", "m"), ("https://api.x.ai.example/v1", "x-grok-4.7"), ("not a url", "grok"),
            ("https://openrouter.ai/api/v1", "grok/m")] {
            let item = body(url, model, &fixture);
            assert_eq!((&item["input"][1]["tools"], item.get("tools")), (functions, None), "{model} at {url}");
            assert_eq!(body(url, model, "[]")["input"], messages, "{model} at {url}");
        }
    }

    /// A GPT-6 conversation keeps the effort it opened with at the top of
    /// every request and sends each later choice as a `configuration_update`
    /// just ahead of the prompt it was made for, so a change appends to what
    /// the requests before it sent. Only the prompts this route ran and a
    /// reply answered count, and the newest. Where the model, the wire or the
    /// route documents no such item, or the prompts in the history do not end
    /// on the current effort or return to the default, the top names the
    /// current effort and no item goes, as before.
    #[test]
    fn an_effort_change_rides_the_history_where_the_model_takes_it() {
        use crate::config::ReasoningEffort::{High, Medium};
        let history = |client: &ChatClient, efforts: &[&str]| {
            let mut messages = vec![ChatMessage::system("rules")];
            for (turn, effort) in efforts.iter().enumerate() {
                messages.push(ChatMessage::user("note"));
                messages.push(ChatMessage { effort: Some(format!("{effort} {}", client.origin())), ..ChatMessage::user(format!("task {turn}")) });
                messages.push(ChatMessage::assistant(Some(format!("done {turn}")), None));
            }
            messages
        };
        let body = |client: &ChatClient, messages: &[ChatMessage]| -> Value {
            serde_json::from_slice(&serialize(client, messages, "[]").unwrap()).unwrap()
        };
        let update = |effort: &str| json!({"type":"configuration_update","reasoning":{"effort":effort}});
        let mut client = ChatClient::new("https://api.openai.com/v1".into(), None, "gpt-6.1-sol".into(), None, 64);
        client.wire_api = crate::providers::WireApi::Responses;
        client.reasoning_effort = Some(High);
        let sent = body(&client, &history(&client, &["default", "high"]));
        assert!(sent.get("reasoning").is_none(), "the top keeps the default the conversation opened with: {sent}");
        assert_eq!(sent["input"][5], update("high"));
        assert_eq!(sent["input"][6], json!({"role":"user","content":"task 1"}), "the change goes just ahead of its prompt");
        let messages = history(&client, &["high", "medium", "high"]);
        let sent = body(&client, &messages);
        assert_eq!(sent["reasoning"], json!({"effort":"high"}));
        let updates: Vec<&Value> = sent["input"].as_array().unwrap().iter().filter(|item| item["type"] == "configuration_update").collect();
        assert_eq!(updates, [&update("medium"), &update("high")]);
        assert_eq!(client.effort_plan(&messages), (Some(High), vec![(5, Medium), (8, High)]));
        assert_eq!(client.effort_plan(&messages[..9]), (Some(High), vec![(5, Medium), (8, High)]), "the newest prompt counts before its reply");
        assert_eq!(client.effort_plan(&history(&client, &["high", "high"])), (Some(High), vec![]), "an unchanged effort sends no item");
        let mut refused = history(&client, &["none", "medium", "high"]);
        refused.remove(3);
        assert_eq!(client.effort_plan(&refused), (Some(Medium), vec![(7, High)]), "an effort no reply answered never goes again");
        let mut terra = ChatClient::new("https://api.openai.com/v1".into(), None, "gpt-6.1-terra".into(), None, 64);
        terra.wire_api = crate::providers::WireApi::Responses;
        let mut switched = history(&terra, &["none"]);
        switched.extend(history(&client, &["medium", "high"]).into_iter().skip(1));
        assert_eq!(client.effort_plan(&switched), (Some(Medium), vec![(8, High)]), "an effort another model took never goes again");
        client.model = "openai/GPT-6.1-sol".into();
        assert_eq!(client.effort_plan(&history(&client, &["high", "medium", "high"])).1.len(), 2, "a gateway's vendor prefix names the same model");
        let current = (Some(High), vec![]);
        assert_eq!(client.effort_plan(&history(&client, &["high", "medium"])), current, "the history does not end on the current effort");
        assert_eq!(client.effort_plan(&messages[..1]), current, "no prompt says the effort");
        assert_eq!(client.effort_plan(&history(&client, &["default", "high", "default", "high"])), current, "a return to the default stays unnamed");
        client.reasoning_effort = None;
        assert_eq!(client.effort_plan(&history(&client, &["high", "default"])), (None, vec![]), "no item names the default");
        client.reasoning_effort = Some(High);
        for (model, wire) in [("gpt-5.6", crate::providers::WireApi::Responses), ("gpt-6.1-sol", crate::providers::WireApi::ChatCompletions)] {
            (client.model, client.wire_api) = (model.into(), wire);
            assert_eq!(client.effort_plan(&history(&client, &["high", "medium", "high"])), current, "{model} over {wire:?}");
        }
        client.wire_api = crate::providers::WireApi::Responses;
        let dir = std::env::temp_dir().join(format!("openmax-effort-plan-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        client.chatgpt = Some(crate::chatgpt::tests::seeded(&dir));
        let sent = body(&client, &history(&client, &["high", "medium", "high"]));
        assert_eq!(sent["reasoning"], json!({"effort":"high"}));
        assert!(!sent.to_string().contains("configuration_update"), "a ChatGPT plan's route documents no such item: {sent}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn subscription_request_omits_unsupported_options_without_changing_key_requests() {
        let dir = std::env::temp_dir().join(format!("openmax-response-body-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut client = ChatClient::new(crate::chatgpt::API.into(), None, "fixture".into(), Some(0.5), 4096).for_conversation("session");
        client.wire_api = crate::providers::WireApi::Responses;
        client.chatgpt = Some(crate::chatgpt::tests::seeded(&dir));
        assert!(client.sends_credential());
        let messages = [ChatMessage::system("rules"), ChatMessage::user("task")];
        let body: Value = serde_json::from_slice(&serialize(&client, &messages, "[]").unwrap()).unwrap();
        for key in ["max_output_tokens", "temperature", "previous_response_id", "conversation", "background", "tools",
            "prompt_cache_retention", "user"] {
            assert!(body.get(key).is_none(), "{key}");
        }
        let cache_key = body["prompt_cache_key"].as_str().expect("the subscription route takes prompt_cache_key").to_owned();
        assert_eq!(body["input"][0]["role"], "developer");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        let origin = client.origin();
        assert!(body.get("reasoning").is_none());
        client.reasoning_effort = Some(crate::config::ReasoningEffort::High);
        let explicit: Value = serde_json::from_slice(&serialize(&client, &messages, "[]").unwrap()).unwrap();
        assert_eq!(explicit["reasoning"], json!({"effort":"high"}));
        assert!(explicit.get("temperature").is_none());
        assert!(explicit.get("max_output_tokens").is_none());
        assert_eq!(client.origin(), origin, "effort does not change opaque-reasoning ownership");
        client.chatgpt = None;
        assert_ne!(client.origin(), origin);
        let body: Value = serde_json::from_slice(&serialize(&client, &messages, "[]").unwrap()).unwrap();
        assert_eq!(body["max_output_tokens"], 4096);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["prompt_cache_key"], cache_key.as_str());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
