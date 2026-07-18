//! Pure translation between line-delimited JSON-RPC 2.0 (the host side) and
//! the `openmax-stdio/1` JSONL contract (the child side). No I/O lives here:
//! the binary owns the pipes, this module owns the mapping, so every case is
//! unit-tested without spawning a process. The child contract this speaks is
//! the one pinned in README under "stdio protocol".

use serde_json::{json, Value};

/// A request the host asked the bridge to perform, parsed from a JSON-RPC
/// method plus params independent of transport.
#[derive(Debug, PartialEq)]
pub enum HostRequest {
    /// Handshake; answered from the child's hello line.
    Initialize,
    /// Start a turn with this text.
    Prompt { text: String },
    /// Cancel the running turn.
    Cancel,
    /// Answer a pending approval.
    Approve { approval_id: String, approved: bool },
    /// Drain the in-flight turn, then exit.
    Shutdown,
}

impl HostRequest {
    /// Parse a JSON-RPC `method` and `params`. An unknown method returns a
    /// reason starting with `method not found` so the caller can pick the
    /// right JSON-RPC error code; malformed params return `invalid params:`.
    pub fn parse(method: &str, params: &Value) -> Result<HostRequest, String> {
        match method {
            "initialize" => Ok(HostRequest::Initialize),
            "prompt" => {
                let text = params
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or("invalid params: prompt needs string 'text'")?;
                if text.trim().is_empty() {
                    return Err("invalid params: prompt 'text' is empty".into());
                }
                Ok(HostRequest::Prompt { text: text.to_string() })
            }
            "cancel" => Ok(HostRequest::Cancel),
            "approve" => {
                let approval_id = params
                    .get("approval_id")
                    .and_then(Value::as_str)
                    .ok_or("invalid params: approve needs string 'approval_id'")?;
                let approved = params
                    .get("approved")
                    .and_then(Value::as_bool)
                    .ok_or("invalid params: approve needs bool 'approved'")?;
                Ok(HostRequest::Approve { approval_id: approval_id.to_string(), approved })
            }
            "shutdown" => Ok(HostRequest::Shutdown),
            other => Err(format!("method not found: {other}")),
        }
    }
}

/// A JSON-RPC 2.0 request or notification must be an object carrying
/// `"jsonrpc":"2.0"`. Reject anything else before dispatch, so a malformed or
/// 1.0 envelope cannot reach a state-changing method.
pub fn check_jsonrpc_version(msg: &Value) -> Result<(), String> {
    match msg.get("jsonrpc").and_then(Value::as_str) {
        Some("2.0") => Ok(()),
        _ => Err("invalid request: expected jsonrpc \"2.0\"".to_string()),
    }
}

/// The `openmax-stdio` command line a request maps to, if any. `Initialize`
/// has no child command (it is answered from the stored hello).
pub fn child_command(req: &HostRequest) -> Option<Value> {
    match req {
        HostRequest::Prompt { text } => Some(json!({"cmd": "user", "text": text})),
        HostRequest::Cancel => Some(json!({"cmd": "cancel"})),
        HostRequest::Approve { approval_id, approved } => {
            Some(json!({"cmd": "approve", "approval_id": approval_id, "approved": approved}))
        }
        HostRequest::Shutdown => Some(json!({"cmd": "quit"})),
        HostRequest::Initialize => None,
    }
}

/// A classified line from the child's stdout.
#[derive(Debug, PartialEq)]
pub enum ChildLine {
    /// The handshake line; captured to answer `initialize`.
    Hello(Value),
    /// The terminal `done` event, carrying its stop_reason, which resolves the
    /// in-flight `prompt` request.
    Done { stop_reason: String },
    /// Any other event or protocol_error, forwarded verbatim as an update.
    Update(Value),
}

/// Classify one child stdout line. A non-JSON or type-less line is a contract
/// violation the bridge surfaces rather than silently drops.
pub fn classify_child_line(line: &str) -> Result<ChildLine, String> {
    let value: Value =
        serde_json::from_str(line).map_err(|e| format!("child emitted non-JSON: {e}"))?;
    let ty = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or("child line has no 'type'")?;
    match ty {
        "hello" => Ok(ChildLine::Hello(value)),
        "done" => {
            let stop_reason = value
                .get("stop_reason")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(ChildLine::Done { stop_reason })
        }
        _ => Ok(ChildLine::Update(value)),
    }
}

// JSON-RPC 2.0 message builders (line-delimited on the wire).

pub fn rpc_result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

pub fn rpc_notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "method": method, "params": params})
}

/// The `initialize` result, drawn from the child's hello line so the host
/// learns the protocol version and session it is now driving.
pub fn initialize_result(hello: &Value) -> Value {
    json!({
        "server": "openmax",
        "protocol_version": hello.get("protocol_version").cloned().unwrap_or(Value::Null),
        "version": hello.get("version").cloned().unwrap_or(Value::Null),
        "session_id": hello.get("session_id").cloned().unwrap_or(Value::Null),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_method() {
        assert_eq!(HostRequest::parse("initialize", &Value::Null).unwrap(), HostRequest::Initialize);
        assert_eq!(
            HostRequest::parse("prompt", &json!({"text": "hi"})).unwrap(),
            HostRequest::Prompt { text: "hi".into() }
        );
        assert_eq!(HostRequest::parse("cancel", &Value::Null).unwrap(), HostRequest::Cancel);
        assert_eq!(
            HostRequest::parse("approve", &json!({"approval_id": "a1", "approved": true})).unwrap(),
            HostRequest::Approve { approval_id: "a1".into(), approved: true }
        );
        assert_eq!(HostRequest::parse("shutdown", &Value::Null).unwrap(), HostRequest::Shutdown);
    }

    #[test]
    fn rejects_non_2_0_envelopes() {
        assert!(check_jsonrpc_version(&json!({"jsonrpc": "2.0", "method": "prompt"})).is_ok());
        assert!(check_jsonrpc_version(&json!({"jsonrpc": "1.0", "method": "prompt"})).is_err());
        assert!(check_jsonrpc_version(&json!({"method": "prompt"})).is_err());
        assert!(check_jsonrpc_version(&json!(["not", "an", "object"])).is_err());
    }

    #[test]
    fn rejects_bad_methods_and_params() {
        assert!(HostRequest::parse("reboot", &Value::Null)
            .unwrap_err()
            .starts_with("method not found"));
        assert!(HostRequest::parse("prompt", &json!({})).unwrap_err().starts_with("invalid params"));
        assert!(HostRequest::parse("prompt", &json!({"text": "   "})).is_err());
        assert!(HostRequest::parse("approve", &json!({"approval_id": "a1"})).is_err());
        assert!(HostRequest::parse("approve", &json!({"approved": true})).is_err());
    }

    #[test]
    fn maps_requests_to_child_commands() {
        assert_eq!(
            child_command(&HostRequest::Prompt { text: "hi".into() }).unwrap(),
            json!({"cmd": "user", "text": "hi"})
        );
        assert_eq!(child_command(&HostRequest::Cancel).unwrap(), json!({"cmd": "cancel"}));
        assert_eq!(child_command(&HostRequest::Shutdown).unwrap(), json!({"cmd": "quit"}));
        assert_eq!(
            child_command(&HostRequest::Approve { approval_id: "a1".into(), approved: false })
                .unwrap(),
            json!({"cmd": "approve", "approval_id": "a1", "approved": false})
        );
        assert_eq!(child_command(&HostRequest::Initialize), None);
    }

    #[test]
    fn classifies_child_lines() {
        let hello = r#"{"type":"hello","proto":"openmax-stdio/1","protocol_version":1,"session_id":"s","version":"0.2.0","project":"/p"}"#;
        assert!(matches!(classify_child_line(hello).unwrap(), ChildLine::Hello(_)));

        assert_eq!(
            classify_child_line(r#"{"session_id":"s","type":"done","stop_reason":"stop"}"#).unwrap(),
            ChildLine::Done { stop_reason: "stop".into() }
        );

        match classify_child_line(r#"{"session_id":"s","type":"token","text":"hi"}"#).unwrap() {
            ChildLine::Update(v) => assert_eq!(v["type"], "token"),
            other => panic!("expected update, got {other:?}"),
        }
        // protocol_error is forwarded as an update, not dropped.
        match classify_child_line(r#"{"type":"protocol_error","message":"x"}"#).unwrap() {
            ChildLine::Update(v) => assert_eq!(v["type"], "protocol_error"),
            other => panic!("expected update, got {other:?}"),
        }

        assert!(classify_child_line("not json").is_err());
        assert!(classify_child_line(r#"{"no":"type"}"#).is_err());
    }

    #[test]
    fn initialize_result_carries_handshake() {
        let hello = json!({
            "type": "hello", "proto": "openmax-stdio/1", "protocol_version": 1,
            "session_id": "s1", "version": "0.2.0", "project": "/p"
        });
        let result = initialize_result(&hello);
        assert_eq!(result["server"], "openmax");
        assert_eq!(result["protocol_version"], 1);
        assert_eq!(result["session_id"], "s1");
        assert_eq!(result["version"], "0.2.0");
    }

    #[test]
    fn rpc_builders_are_well_formed() {
        assert_eq!(
            rpc_result(json!(1), json!({"ok": true})),
            json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}})
        );
        assert_eq!(
            rpc_error(json!(2), -32601, "nope"),
            json!({"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "nope"}})
        );
        assert_eq!(
            rpc_notification("update", json!({"type": "token"})),
            json!({"jsonrpc": "2.0", "method": "update", "params": {"type": "token"}})
        );
    }
}
