//! Native provider translation through the real HTTP client and stream reader.
use std::{path::PathBuf, sync::Arc, time::Duration};
use open_max_core::{client::{ChatClient, StreamDelta}, config::{ReasoningEffort, Settings}, providers, state::CancelToken, types::ChatMessage};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("openmax-native-messages-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap(); Self(path)
    }
}
impl Drop for Scratch { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

struct Request { head: String, body: Vec<u8> }
async fn serve(replies: Vec<(u16, String)>) -> (String, tokio::task::JoinHandle<Vec<Request>>) {
    serve_chunks(replies, 37).await
}
async fn serve_chunks(replies: Vec<(u16, String)>, chunk_size: usize) -> (String, tokio::task::JoinHandle<Vec<Request>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(15), async move {
            let mut requests = Vec::new();
            for (status, reply) in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                loop {
                    let mut buffer = [0;4096]; let n = socket.read(&mut buffer).await.unwrap(); assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                    if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..end]).into_owned();
                        let length: usize = head.lines().find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap();
                        if bytes.len() >= end+4+length { requests.push(Request { head, body:bytes[end+4..end+4+length].to_vec() }); break; }
                    }
                }
                let content_type=if reply.starts_with('{') {"application/json"} else {"text/event-stream"};
                let header = format!("HTTP/1.1 {status} Fixture\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",reply.len());
                socket.write_all(header.as_bytes()).await.unwrap();
                // Fragment frames across reads, including event and JSON boundaries.
                for chunk in reply.as_bytes().chunks(chunk_size) { if socket.write_all(chunk).await.is_err() { break; } }
            }
            requests
        }).await.expect("native request fixture deadline")
    });
    (base,task)
}
fn event(value: Value) -> String { format!("event: {}\ndata: {}\n\n",value["type"].as_str().unwrap(),value) }
fn start() -> String { event(json!({"type":"message_start","message":{"id":"msg-fixture","type":"message","role":"assistant","model":"fixture-model","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}}})) }
fn end(reason: &str) -> String {
    event(json!({"type":"message_delta","delta":{"stop_reason":reason,"stop_sequence":null},"usage":{"output_tokens":7}})) + &event(json!({"type":"message_stop"}))
}
fn text_response() -> String {
    start() + &event(json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}))
        + &event(json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"verified"}}))
        + &event(json!({"type":"content_block_stop","index":0})) + &end("end_turn")
}
fn tool_response() -> String {
    start() + &event(json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}))
        + &event(json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"inspect first"}}))
        + &event(json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"fixture-signature"}}))
        + &event(json!({"type":"content_block_stop","index":0}))
        + &event(json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-fixture","name":"inspect","input":{}}}))
        + &event(json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}))
        + &event(json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"note.txt\"}"}}))
        + &event(json!({"type":"content_block_stop","index":1})) + &end("tool_use")
}
fn tools() -> String { json!([{"type":"function","function":{"name":"inspect","description":"Read a path","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]).to_string() }
fn client(dir: &Scratch, base: &str, effort: Option<ReasoningEffort>) -> ChatClient {
    std::fs::write(dir.0.join("providers.json"),json!({"providers":{"native":{"base_url":base,"api_key":"fixture-native-key","wire_api":"anthropic"}}}).to_string()).unwrap();
    let settings = Settings { provider:Some("native".into()),model:"fixture-model".into(),context_tokens:Some(8192),max_tokens:128,temperature:Some(0.25),reasoning_effort:effort,..Settings::default() };
    ChatClient::from_endpoint(&providers::resolve(&settings,&dir.0).unwrap())
}
async fn run(client: &ChatClient, messages: &[ChatMessage]) -> Result<open_max_core::client::CompletionResult,String> {
    tokio::time::timeout(Duration::from_secs(5),client.stream_chat(messages,&tools(),Arc::new(CancelToken::default()), |_|{})).await.expect("native completion deadline")
}

#[tokio::test]
async fn native_effort_auth_and_default_omission_preserve_request_content() {
    let scratch=Scratch::new();
    let levels=[None,Some(ReasoningEffort::Low),Some(ReasoningEffort::Medium),Some(ReasoningEffort::High),Some(ReasoningEffort::XHigh),Some(ReasoningEffort::Max)];
    let (base,server)=serve(levels.iter().map(|_|(200,text_response())).collect()).await;
    for effort in levels {
        let client=client(&scratch,&base,effort).for_conversation("native-session"); let mut shown=String::new();
        let result=client.stream_chat_with_note(&[ChatMessage::system("instructions"),ChatMessage::user("task")],Some("turn note"),&tools(),Arc::new(CancelToken::default()),|delta| if let StreamDelta::Content(text)=delta {shown.push_str(&text)}).await.unwrap();
        assert_eq!(result.content,"verified"); assert_eq!(shown,"verified"); assert_eq!(result.finish_reason,"stop"); assert!(result.tool_calls.is_empty());
        let usage=result.usage.unwrap();assert_eq!((usage.prompt_tokens,usage.completion_tokens,usage.cached_tokens),(17,7,Some(4)));
    }
    let mut default=None;
    for (request,effort) in server.await.unwrap().into_iter().zip(levels) {
        assert!(request.head.starts_with("POST /v1/messages "));
        let headers=request.head.to_ascii_lowercase();assert!(headers.contains("x-api-key: fixture-native-key\r\n"));assert!(headers.contains("anthropic-version: 2023-06-01\r\n"));assert!(!headers.contains("authorization:"));
        let mut body:Value=serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["model"],"fixture-model");assert_eq!(body["max_tokens"],128);assert_eq!(body["temperature"],0.25);assert_eq!(body["stream"],true);
        assert!(body.get("reasoning_effort").is_none());assert!(body.get("thinking").is_none());assert!(body.get("prompt_cache_key").is_none());
        if let Some(effort)=effort {assert_eq!(body["output_config"]["effort"],effort.as_str());} else {assert!(body.get("output_config").is_none());}
        assert!(body["system"].to_string().contains("instructions"));
        assert_eq!(body["messages"][0]["role"],"user");assert!(body["messages"][0]["content"].to_string().contains("turn note"));
        assert_eq!(body["tools"][0]["name"],"inspect");assert_eq!(body["tools"][0]["input_schema"],serde_json::from_str::<Value>(&tools()).unwrap()[0]["function"]["parameters"]);
        body.as_object_mut().unwrap().remove("output_config");
        if let Some(default)=&default {assert_eq!(&body,default);} else {default=Some(body);}
    }
}

#[tokio::test]
async fn native_tool_round_trip_preserves_signed_blocks_only_for_the_same_origin() {
    let scratch=Scratch::new();let(base,server)=serve(vec![(200,tool_response()),(200,text_response()),(200,text_response())]).await;
    let mut client=client(&scratch,&base,Some(ReasoningEffort::High));
    let mut messages=vec![ChatMessage::system("instructions"),ChatMessage::user("inspect")];
    let mut thinking=Vec::new();
    let first=client.stream_chat(&messages,&tools(),Arc::new(CancelToken::default()),|delta| {
        if let StreamDelta::Reasoning(text)=delta {thinking.push(text);}
    }).await.unwrap();
    assert_eq!(thinking,["inspect first"]);
    assert_eq!(first.finish_reason,"tool_calls");assert_eq!(first.tool_calls.len(),1);
    assert_eq!(first.tool_calls[0].id,"call-fixture");assert_eq!(first.tool_calls[0].function.name,"inspect");assert_eq!(serde_json::from_str::<Value>(&first.tool_calls[0].function.arguments).unwrap(),json!({"path":"note.txt"}));
    let mut assistant=ChatMessage::assistant(Some(first.content),Some(first.tool_calls));assistant.reasoning_details=first.reasoning_details;assistant.reasoning_origin=Some(client.origin());
    assert!(assistant.reasoning_details.is_some());messages.push(assistant);messages.push(ChatMessage::tool("call-fixture","file contents"));
    run(&client,&messages).await.unwrap();client.model="different-model".into();run(&client,&messages).await.unwrap();
    let records=server.await.unwrap();let second:Value=serde_json::from_slice(&records[1].body).unwrap();
    assert_eq!(second["messages"][1]["content"],json!([{"type":"thinking","thinking":"inspect first","signature":"fixture-signature"},{"type":"tool_use","id":"call-fixture","name":"inspect","input":{"path":"note.txt"}}]));
    assert_eq!(second["messages"][2]["content"][0],json!({"type":"tool_result","tool_use_id":"call-fixture","content":"file contents"}));
    let switched=String::from_utf8(records[2].body.clone()).unwrap();assert!(!switched.contains("fixture-signature"));assert!(!switched.contains("inspect first"));assert!(switched.contains("call-fixture"));
}

#[tokio::test]
async fn native_invalid_and_incomplete_streams_never_return_executable_tools() {
    let valid=tool_response();
    let truncated=valid.replace(&event(json!({"type":"message_stop"})),"");
    let malformed=valid.replace("note.txt","note\\\".txt");
    let bad_stop=valid.replace("\"stop_reason\":\"tool_use\"","\"stop_reason\":\"max_tokens\"");
    let error=valid.replace(&end("tool_use"),&event(json!({"type":"error","error":{"type":"invalid_request_error","message":"fixture rejected"}})));
    let duplicate=valid.replace(&end("tool_use"),&(event(json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-fixture","name":"inspect","input":{}}}))+&end("tool_use")));
    let missing_signature=valid.replace("fixture-signature", "");
    let bad_name=valid.replace("\"name\":\"inspect\"", "\"name\":\"bad name\"");
    let unknown=valid.replace("\"type\":\"tool_use\"", "\"type\":\"unknown_tool\"");
    let duplicate_id=valid.replace(&end("tool_use"),&(event(json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call-fixture","name":"inspect","input":{}}}))+&event(json!({"type":"content_block_stop","index":2}))+&end("tool_use")));
    for response in [truncated,malformed,bad_stop,error,duplicate,missing_signature,bad_name,unknown,duplicate_id] {
        let scratch=Scratch::new(); let(base,server)=serve(vec![(200,response)]).await;
        let client=client(&scratch,&base,None);let cancel=Arc::new(CancelToken::default());
        let result=tokio::time::timeout(Duration::from_secs(5),client.stream_chat(&[ChatMessage::user("inspect")],&tools(),cancel.clone(),|delta| {
            if matches!(delta,StreamDelta::Retry{..}) {cancel.cancel();}
        })).await.expect("invalid stream deadline");
        match result {Err(_)=>{},Ok(result)=>{assert!(result.tool_calls.is_empty());assert_ne!(result.finish_reason,"tool_calls");}}
        assert_eq!(server.await.unwrap().len(),1);
    }
}

#[tokio::test]
async fn native_retry_keeps_effort_and_body_identical_and_model_errors_stay_errors() {
    let scratch=Scratch::new();let(base,server)=serve(vec![(503,json!({"error":{"type":"overloaded_error","message":"fixture busy"}}).to_string()),(200,text_response()),(400,json!({"error":{"type":"invalid_request_error","message":"unsupported effort for this model"}}).to_string())]).await;
    let client=client(&scratch,&base,Some(ReasoningEffort::Max));let messages=[ChatMessage::user("task")];
    assert_eq!(run(&client,&messages).await.unwrap().content,"verified");
    let error=run(&client,&messages).await.err().expect("400 must stay an error");assert!(error.contains("unsupported effort"),"{error}");
    let records=server.await.unwrap();assert_eq!(records.len(),3);assert_eq!(records[0].body,records[1].body);assert_eq!(records[1].body,records[2].body);
}

#[test]
fn native_unsupported_effort_is_rejected_before_a_request() {
    let scratch=Scratch::new();
    std::fs::write(scratch.0.join("providers.json"),json!({"providers":{"native":{"base_url":"https://api.anthropic.com/v1","wire_api":"anthropic"}}}).to_string()).unwrap();
    for effort in [ReasoningEffort::None,ReasoningEffort::Minimal] {
        let settings=Settings{provider:Some("native".into()),model:"fixture-model".into(),context_tokens:Some(8192),reasoning_effort:Some(effort),..Settings::default()};
        assert!(providers::resolve(&settings,&scratch.0).is_err());
    }
}

#[tokio::test]
async fn native_json_fallback_and_cancellation_keep_the_same_tool_contract() {
    let scratch=Scratch::new();
    let complete=json!({"type":"message","role":"assistant","model":"fixture-model","content":[
        {"type":"redacted_thinking","data":"fixture-opaque"},
        {"type":"text","text":"read next"},
        {"type":"tool_use","id":"json-call","name":"inspect","input":{"path":"note.txt"}}
    ],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":7}}).to_string();
    let(base,server)=serve_chunks(vec![(200,complete.clone()),(200,tool_response()),(200,complete)],usize::MAX).await;
    let client=client(&scratch,&base,None);
    let mut shown=String::new();
    let first=client.stream_chat(&[ChatMessage::user("task")],&tools(),Arc::new(CancelToken::default()),|delta| {
        if let StreamDelta::Content(text)=delta {shown.push_str(&text);}
    }).await.unwrap();
    assert_eq!(shown,"read next");
    assert_eq!(first.content,"read next");assert_eq!(first.finish_reason,"tool_calls");assert_eq!(first.tool_calls[0].id,"json-call");
    assert!(first.reasoning_details.unwrap().get().contains("fixture-opaque"));
    let cancel=Arc::new(CancelToken::default());
    let second=client.stream_chat(&[ChatMessage::user("task")],&tools(),cancel.clone(),|delta| {
        if matches!(delta,StreamDelta::Reasoning(_)) {cancel.cancel();}
    }).await.unwrap();
    assert_eq!(second.finish_reason,"cancelled");assert!(second.tool_calls.is_empty());
    let cancel=Arc::new(CancelToken::default());
    let third=client.stream_chat(&[ChatMessage::user("task")],&tools(),cancel.clone(),|delta| {
        if matches!(delta,StreamDelta::Content(_)) {cancel.cancel();}
    }).await.unwrap();
    assert_eq!(third.finish_reason,"cancelled");assert!(third.tool_calls.is_empty());assert!(third.reasoning_details.is_none());assert_eq!(third.content,"read next");
    assert_eq!(server.await.unwrap().len(),3);
}

fn json_message(blocks: Vec<Value>, reason: &str) -> String {
    json!({"type":"message","role":"assistant","content":blocks,"stop_reason":reason,"usage":{"input_tokens":1,"output_tokens":2}}).to_string()
}

#[tokio::test]
async fn native_custom_headers_and_legacy_credentials_keep_their_protocol() {
    let scratch=Scratch::new(); let(base,server)=serve(vec![(200,text_response())]).await;
    let mut native=client(&scratch,&base,None);
    native.api_key=None;
    native.headers=vec![("X-Api-Key".into(),"fixture-header-key".into()),("Anthropic-Version".into(),"2023-06-01".into())];
    run(&native,&[ChatMessage::user("task")]).await.unwrap();
    let head=server.await.unwrap().remove(0).head.to_ascii_lowercase();
    assert_eq!(head.matches("anthropic-version:").count(),1);
    assert_eq!(head.matches("x-api-key:").count(),1);
    assert!(head.contains("x-api-key: fixture-header-key"));assert!(!head.contains("authorization:"));
    native.api_key=Some("another-fixture-key".into());
    assert!(run(&native,&[ChatMessage::user("task")]).await.err().unwrap().contains("one API key source"));

    let legacy=ChatClient::new("https://legacy.example/v1".into(),Some("fixture-old-key".into()),"fixture".into(),None,64);
    // A persisted transcript's stamp predates the native protocol and must stay valid.
    assert_eq!(legacy.origin(),"4749a7f612ed0d7a");
    let reply="data: {\"choices\":[{\"delta\":{\"content\":\"legacy\"},\"finish_reason\":\"stop\"}]}\n\n".to_owned();
    let(base,server)=serve(vec![(200,reply)]).await;
    let legacy=ChatClient::new(base,Some("fixture-old-key".into()),"fixture".into(),None,64);
    assert_eq!(run(&legacy,&[ChatMessage::user("task")]).await.unwrap().content,"legacy");
    let head=server.await.unwrap().remove(0).head.to_ascii_lowercase();
    assert!(head.contains("authorization: bearer fixture-old-key"));
    assert!(!head.contains("anthropic-version:"));assert!(!head.contains("x-api-key:"));
}

#[tokio::test]
async fn native_history_groups_tool_results_and_places_notes_at_the_tail() {
    use open_max_core::types::{ToolCall,ToolCallFunction};
    let scratch=Scratch::new();let(base,server)=serve(vec![(200,text_response());3]).await;
    let native=client(&scratch,&base,None);
    let call=|id:&str| ToolCall {id:id.into(),kind:"function".into(),function:ToolCallFunction{name:"inspect".into(),arguments:"{}".into()},extra_content:None};
    let history=[ChatMessage::system(""),ChatMessage::system("instructions"),ChatMessage::user("first"),ChatMessage::user("second"),
        ChatMessage::assistant(Some("checking".into()),Some(vec![call("one"),call("two")])),ChatMessage::tool("one","result one"),ChatMessage::tool("two","result two")];
    native.stream_chat_with_note(&history,Some("fresh note"),"[]",Arc::new(CancelToken::default()), |_|{}).await.unwrap();
    native.stream_chat_with_note(&history[..5],Some("after assistant"),"[]",Arc::new(CancelToken::default()), |_|{}).await.unwrap();
    native.stream_chat(&history[..2],"[]",Arc::new(CancelToken::default()), |_|{}).await.unwrap();
    let records=server.await.unwrap();let body:Value=serde_json::from_slice(&records[0].body).unwrap();
    assert!(body.get("tools").is_none());assert!(body.get("tool_choice").is_none());
    assert_eq!(body["system"],json!([{"type":"text","text":"instructions"}]));
    assert_eq!(body["messages"][0]["content"],json!([{"type":"text","text":"first"},{"type":"text","text":"second"}]));
    assert_eq!(body["messages"][1]["content"][1]["input"],json!({}));
    assert_eq!(body["messages"][2]["content"][0]["content"],"result one");
    assert!(body["messages"][2]["content"][1]["content"].as_str().unwrap().starts_with("result two"));
    assert!(body["messages"][2]["content"][1]["content"].as_str().unwrap().ends_with("fresh note"));
    let body:Value=serde_json::from_slice(&records[1].body).unwrap();
    assert_eq!(body["messages"].as_array().unwrap().last().unwrap(),&json!({"role":"user","content":[{"type":"text","text":"after assistant"}]}));
    let body:Value=serde_json::from_slice(&records[2].body).unwrap();assert_eq!(body["messages"],json!([]));
}

#[tokio::test]
async fn native_tool_and_block_limits_accept_the_boundary_and_refuse_overflow() {
    let name=format!("a_-{}","b".repeat(125));assert_eq!(name.len(),128);
    let calls=|count:usize| (0..count).map(|i|json!({"type":"tool_use","id":format!("call-{i}"),"name":name,"input":{}})).collect();
    let blocks=|count:usize| {
        let mut wire=start();
        for i in 0..count { wire+=&event(json!({"type":"content_block_start","index":i,"content_block":{"type":"text","text":"x"}}));wire+=&event(json!({"type":"content_block_stop","index":i})); }
        wire+&end("end_turn")
    };
    let empty_tool=start()+&event(json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"empty-call","name":"inspect","input":{}}}))
        +&event(json!({"type":"content_block_stop","index":0}))+&end("tool_use");
    let scratch=Scratch::new();let(base,server)=serve_chunks(vec![(200,json_message(calls(128),"tool_use")),(200,json_message(calls(129),"tool_use")),
        (200,blocks(512)),(200,blocks(513)),(200,json_message(vec![json!({"type":"text","text":"capped"})],"max_tokens")),(200,empty_tool)],4096).await;
    let native=client(&scratch,&base,None);
    let reply=run(&native,&[ChatMessage::user("task")]).await.unwrap();assert_eq!(reply.tool_calls.len(),128);assert_eq!(reply.tool_calls[0].function.name,name);assert_eq!(reply.tool_calls[127].function.arguments,"{}");
    assert!(run(&native,&[ChatMessage::user("task")]).await.is_err());
    assert_eq!(run(&native,&[ChatMessage::user("task")]).await.unwrap().content,"x".repeat(512));
    assert!(run(&native,&[ChatMessage::user("task")]).await.is_err());
    let capped=run(&native,&[ChatMessage::user("task")]).await.unwrap();assert_eq!(capped.finish_reason,"length");assert_eq!(capped.content,"capped");assert!(capped.tool_calls.is_empty());
    let empty=run(&native,&[ChatMessage::user("task")]).await.unwrap();assert_eq!(empty.tool_calls.len(),1);assert_eq!(empty.tool_calls[0].function.arguments,"{}");
    assert_eq!(server.await.unwrap().len(),6);
}

#[tokio::test]
async fn native_stream_faults_are_reported_and_retryable_api_errors_restart() {
    let error=event(json!({"type":"error","error":{"type":"api_error","message":"temporary fixture fault"}}));
    let recovered=text_response().replace("\"text\":\"\"","\"text\":\"start:\"").replace("\"usage\":{","\"usage\":\ndata: {");
    let scratch=Scratch::new();let(base,server)=serve(vec![(200,error),(200,recovered)]).await;
    let native=client(&scratch,&base,None);let mut retries=0;let mut content=Vec::new();
    let result=native.stream_chat(&[ChatMessage::user("task")],&tools(),Arc::new(CancelToken::default()),|delta| match delta {
        StreamDelta::Retry{..}=>retries+=1,StreamDelta::Content(text)=>content.push(text),_=>{},
    }).await.unwrap();
    assert_eq!(retries,1);assert_eq!(result.content,"start:verified");assert_eq!(content,["start:","verified"]);let requests=server.await.unwrap();assert_eq!(requests[0].body,requests[1].body);

    let valid=text_response();
    let begin=|index:usize,id:&str| event(json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":id,"name":"inspect","input":{}}}));
    let close=|index:usize| event(json!({"type":"content_block_stop","index":index}));
    let interleaved=start()+&begin(0,"first")+&begin(1,"second")+&close(0)+&close(1)+&end("tool_use");
    let wrong_index=start()+&begin(9,"first")+&close(0)+&end("tool_use");
    let after_finish=start()+&begin(0,"first")+&close(0)+&event(json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}))
        +&begin(1,"second")+&close(1)+&event(json!({"type":"message_stop"}));
    let early_finish=start()+&begin(0,"first")+&event(json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}))
        +&close(0)+&event(json!({"type":"message_stop"}));
    let invalid=[interleaved,wrong_index,after_finish,early_finish,valid.replace("\"input_tokens\":10","\"input_tokens\":1\ndata: 0"),valid.replacen(&start(),"",1),valid.replacen(&start(),&(start()+&start()),1),
        valid.replace("\"role\":\"assistant\"","\"role\":\"user\""),valid.replace("\"content\":[]","\"content\":[{\"type\":\"text\",\"text\":\"unexpected\"}]"),
        valid.replace(&event(json!({"type":"content_block_stop","index":0})),""),valid.replace("\"stop_reason\":\"end_turn\"","\"stop_reason\":\"pause_turn\""),
        json_message(vec![json!({"type":"tool_use","id":"","name":"inspect","input":{}})],"tool_use"),
        json_message(vec![json!({"type":"tool_use","id":"call","name":"a".repeat(129),"input":{}})],"tool_use"),
        json_message(vec![json!({"type":"tool_use","id":"call","name":"inspect","input":[]})],"tool_use"),
        json_message(vec![json!({"type":"redacted_thinking","data":""})],"end_turn"),
        json_message(vec![json!({"type":"thinking","thinking":false,"signature":"signature"})],"end_turn"),
        json_message(vec![],"tool_use"),json_message(vec![],"end_turn").replace("\"role\":\"assistant\"","\"role\":\"user\"")];
    for body in invalid {
        let scratch=Scratch::new();let(base,server)=serve_chunks(vec![(200,body)],usize::MAX).await;let native=client(&scratch,&base,None);
        assert!(run(&native,&[ChatMessage::user("task")]).await.is_err());server.await.unwrap();
    }
}

#[tokio::test]
async fn native_stream_byte_budget_keeps_a_boundary_reply_and_refuses_late_tools() {
    const LIMIT:usize=16*1024*1024;
    let body=|overflow:usize| {
        let tail=tool_response();let mut remaining=LIMIT+overflow-tail.len();let unit=format!(":{}\n\n","x".repeat(4093));
        let mut wire=String::with_capacity(LIMIT+overflow);
        while remaining>4099 {wire.push_str(&unit);remaining-=unit.len();}
        wire.push(':');wire.push_str(&"x".repeat(remaining-3));wire.push_str("\n\n");wire.push_str(&tail);wire
    };
    let scratch=Scratch::new();let(base,server)=serve_chunks(vec![(200,body(0)),(200,body(1))],65536).await;
    let native=client(&scratch,&base,None);
    assert_eq!(run(&native,&[ChatMessage::user("task")]).await.unwrap().tool_calls.len(),1);
    let reply=run(&native,&[ChatMessage::user("task")]).await.unwrap();assert_eq!(reply.finish_reason,open_max_core::client::TRUNCATED);assert!(reply.tool_calls.is_empty());
    assert_eq!(server.await.unwrap().len(),2);
}

#[tokio::test]
async fn native_signed_thinking_requires_unchanged_prefix_and_assistant_payload() {
    let scratch=Scratch::new();
    let marker=event(json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call-fixture","name":"inspect","input":{}}}));
    let text=event(json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":"planning"}}))+&event(json!({"type":"content_block_stop","index":1}));
    let signed=tool_response().replace("\"index\":1","\"index\":2").replace(&marker,&(text+&marker));
    let mut replies=vec![(200,signed.clone())];replies.extend(vec![(200,text_response());9]);
    replies.push((200,signed));replies.push((200,text_response()));
    let(base,server)=serve(replies).await;let mut native=client(&scratch,&base,None);
    let opening=vec![ChatMessage::system("instructions"),ChatMessage::user("earlier task"),ChatMessage::assistant(Some("earlier answer".into()),None),ChatMessage::user("inspect")];
    let result=run(&native,&opening).await.unwrap();
    let mut assistant=ChatMessage::assistant(Some(result.content),Some(result.tool_calls));
    assistant.reasoning_details=result.reasoning_details;assistant.reasoning_origin=Some(native.origin());
    let mut history=opening.clone();history.push(assistant.clone());history.push(ChatMessage::tool("call-fixture","result"));
    let resumed:Vec<ChatMessage>=serde_json::from_slice(&serde_json::to_vec(&history).unwrap()).unwrap();
    run(&native,&resumed).await.unwrap();
    native.reasoning_effort=Some(ReasoningEffort::Low);native.max_tokens=96;native.temperature=None;
    run(&native,&history).await.unwrap();
    let mut changed=history.clone();changed[0]=ChatMessage::system("changed instructions");run(&native,&changed).await.unwrap();
    let mut changed_tools:Value=serde_json::from_str(&tools()).unwrap();changed_tools[0]["function"]["description"]=json!("changed description");
    native.stream_chat(&history,&changed_tools.to_string(),Arc::new(CancelToken::default()), |_|{}).await.unwrap();
    let mut changed=history.clone();changed[1]=ChatMessage::user("edited earlier task");run(&native,&changed).await.unwrap();
    let mut compacted=history.clone();compacted.drain(1..3);compacted.insert(1,ChatMessage::user("[context note: earlier exchange summarized]"));
    run(&native,&compacted).await.unwrap();
    let mut changed=history.clone();changed[4].content=Some("edited assistant text".into());run(&native,&changed).await.unwrap();
    let mut changed=history.clone();changed[4].tool_calls.as_mut().unwrap()[0].function.arguments="{\"path\":\"edited.txt\"}".into();run(&native,&changed).await.unwrap();
    let mut changed=history.clone();changed[4].tool_calls.as_mut().unwrap()[0].id="edited-call".into();changed[5].tool_call_id=Some("edited-call".into());run(&native,&changed).await.unwrap();
    let noted=native.stream_chat_with_note(&opening,Some("transient capability note"),&tools(),Arc::new(CancelToken::default()), |_|{}).await.unwrap();
    assistant.reasoning_details=noted.reasoning_details;
    let mut noted_history=opening;noted_history.push(assistant);noted_history.push(ChatMessage::tool("call-fixture","result"));
    run(&native,&noted_history).await.unwrap();
    let records=server.await.unwrap();assert_eq!(records.len(),12);
    for index in [1,2] {assert!(String::from_utf8_lossy(&records[index].body).contains("fixture-signature"),"request {index}");}
    for index in [3,4,5,6,7,8,9,11] {assert!(!String::from_utf8_lossy(&records[index].body).contains("fixture-signature"),"request {index}");}
    let edited:Value=serde_json::from_slice(&records[7].body).unwrap();assert_eq!(edited["messages"][3]["content"][0]["text"],"edited assistant text");
    let edited:Value=serde_json::from_slice(&records[8].body).unwrap();assert_eq!(edited["messages"][3]["content"].as_array().unwrap().iter().find(|b| b["type"] == "tool_use").unwrap()["input"]["path"],"edited.txt");
    assert!(records.iter().all(|r| !String::from_utf8_lossy(&r.body).contains("reasoning_origin")));
}
