use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string, as in the OpenAI wire format.
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

fn function_type() -> String {
    "function".to_string()
}

/// A message in the OpenAI chat wire format. `content` is optional because
/// assistant messages that only carry tool calls have no content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: Some(content.into()), tool_calls: None, tool_call_id: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: Some(content.into()), tool_calls: None, tool_call_id: None }
    }
    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self { role: "assistant".into(), content, tool_calls, tool_call_id: None }
    }
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: Some(content.into()), tool_calls: None, tool_call_id: Some(tool_call_id.into()) }
    }

    /// Rough size estimate used for context budgeting (~4 chars per token).
    pub fn estimated_tokens(&self) -> usize {
        let mut chars = self.content.as_deref().map(str::len).unwrap_or(0);
        if let Some(calls) = &self.tool_calls {
            for c in calls {
                chars += c.function.name.len() + c.function.arguments.len() + 16;
            }
        }
        chars / 4 + 8
    }
}

/// Events streamed from the agent loop to the frontend. `Deserialize` makes
/// the wire contract round-trippable: a frontend, an interop adapter, or the
/// `openmax --check --stdio` conformance validator can parse events back into
/// this type, so the `--stdio` protocol has one authoritative schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Token { text: String },
    Thinking { text: String },
    /// One assistant message finalized (tool-call markup already stripped).
    /// A run of Token deltas always ends in exactly one MessageDone.
    MessageDone { text: String },
    /// Estimated context usage at the start of a completion.
    Budget { used_tokens: usize, context_tokens: usize },
    /// Server-reported token accounting for one completion. `cached_tokens`
    /// near zero on a long session means prefix stability broke and every
    /// step is paying a full prompt re-prefill.
    Usage { prompt_tokens: u64, completion_tokens: u64, cached_tokens: Option<u64> },
    ToolStart { call_id: String, name: String, args: serde_json::Value },
    ToolEnd { call_id: String, ok: bool, output: String },
    Diff { call_id: String, path: String, diff: String, added: usize, removed: usize },
    /// Mutating tool waiting on the user. `detail` is a short args preview
    /// (paths, command head) for the TUI card; may be empty.
    ApprovalRequest {
        approval_id: String,
        name: String,
        summary: String,
        detail: String,
    },
    /// The approval waiter closed (approve, deny, timeout, cancel, or drop).
    /// Frontends must clear any pending approval UI matching `approval_id`.
    ApprovalSettled {
        approval_id: String,
        /// `approved` | `declined` | `timed_out` | `cancelled`
        outcome: String,
    },
    /// The session's tools, skills, and system prompt were re-frozen from
    /// current config (extension files changed, or the user forced /reload).
    Refrozen { tools: usize, skills: usize },
    Done { stop_reason: String },
    Error { message: String },
}

#[derive(Clone, Debug, Serialize)]
pub struct AgentEventEnvelope {
    pub session_id: String,
    #[serde(flatten)]
    pub event: AgentEvent,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn env(event: AgentEvent) -> String {
        let e = AgentEventEnvelope { session_id: "s1".into(), event };
        serde_json::to_string(&e).unwrap()
    }

    /// Golden wire format for every `AgentEvent`, wrapped in its envelope
    /// exactly as `--stdio` and `--print --json` emit it. These strings are
    /// the `openmax-stdio/1` contract: session_id first, then the `type`
    /// discriminator, then variant fields in declaration order. A change here
    /// is a protocol break and must bump `PROTO_VERSION`.
    #[test]
    fn event_envelope_wire_is_stable() {
        assert_eq!(
            env(AgentEvent::Token { text: "hi".into() }),
            r#"{"session_id":"s1","type":"token","text":"hi"}"#
        );
        assert_eq!(
            env(AgentEvent::Thinking { text: "mm".into() }),
            r#"{"session_id":"s1","type":"thinking","text":"mm"}"#
        );
        assert_eq!(
            env(AgentEvent::MessageDone { text: "done".into() }),
            r#"{"session_id":"s1","type":"message_done","text":"done"}"#
        );
        assert_eq!(
            env(AgentEvent::Budget { used_tokens: 10, context_tokens: 8000 }),
            r#"{"session_id":"s1","type":"budget","used_tokens":10,"context_tokens":8000}"#
        );
        assert_eq!(
            env(AgentEvent::Usage {
                prompt_tokens: 100,
                completion_tokens: 20,
                cached_tokens: Some(80),
            }),
            r#"{"session_id":"s1","type":"usage","prompt_tokens":100,"completion_tokens":20,"cached_tokens":80}"#
        );
        assert_eq!(
            env(AgentEvent::Usage {
                prompt_tokens: 100,
                completion_tokens: 20,
                cached_tokens: None,
            }),
            r#"{"session_id":"s1","type":"usage","prompt_tokens":100,"completion_tokens":20,"cached_tokens":null}"#
        );
        assert_eq!(
            env(AgentEvent::ToolStart {
                call_id: "c1".into(),
                name: "read_file".into(),
                args: json!({"path": "a.rs"}),
            }),
            r#"{"session_id":"s1","type":"tool_start","call_id":"c1","name":"read_file","args":{"path":"a.rs"}}"#
        );
        assert_eq!(
            env(AgentEvent::ToolEnd { call_id: "c1".into(), ok: true, output: "ok".into() }),
            r#"{"session_id":"s1","type":"tool_end","call_id":"c1","ok":true,"output":"ok"}"#
        );
        assert_eq!(
            env(AgentEvent::Diff {
                call_id: "c1".into(),
                path: "a.rs".into(),
                diff: "+x".into(),
                added: 1,
                removed: 0,
            }),
            r#"{"session_id":"s1","type":"diff","call_id":"c1","path":"a.rs","diff":"+x","added":1,"removed":0}"#
        );
        assert_eq!(
            env(AgentEvent::ApprovalRequest {
                approval_id: "ap1".into(),
                name: "bash".into(),
                summary: "run".into(),
                detail: "ls".into(),
            }),
            r#"{"session_id":"s1","type":"approval_request","approval_id":"ap1","name":"bash","summary":"run","detail":"ls"}"#
        );
        assert_eq!(
            env(AgentEvent::ApprovalSettled {
                approval_id: "ap1".into(),
                outcome: "approved".into(),
            }),
            r#"{"session_id":"s1","type":"approval_settled","approval_id":"ap1","outcome":"approved"}"#
        );
        assert_eq!(
            env(AgentEvent::Refrozen { tools: 7, skills: 2 }),
            r#"{"session_id":"s1","type":"refrozen","tools":7,"skills":2}"#
        );
        assert_eq!(
            env(AgentEvent::Done { stop_reason: "stop".into() }),
            r#"{"session_id":"s1","type":"done","stop_reason":"stop"}"#
        );
        assert_eq!(
            env(AgentEvent::Error { message: "boom".into() }),
            r#"{"session_id":"s1","type":"error","message":"boom"}"#
        );
    }

    /// The contract is round-trippable: every event deserializes from its own
    /// wire form back into an identical value. This is what lets an interop
    /// adapter or the conformance validator parse events authoritatively.
    #[test]
    fn events_round_trip_through_wire() {
        let samples = [
            AgentEvent::Token { text: "hi".into() },
            AgentEvent::Usage { prompt_tokens: 1, completion_tokens: 2, cached_tokens: None },
            AgentEvent::ToolStart {
                call_id: "c1".into(),
                name: "grep".into(),
                args: json!({"pattern": "fn"}),
            },
            AgentEvent::Diff {
                call_id: "c1".into(),
                path: "a".into(),
                diff: "d".into(),
                added: 1,
                removed: 2,
            },
            AgentEvent::Done { stop_reason: "cancelled".into() },
        ];
        for ev in samples {
            let wire = serde_json::to_string(&ev).unwrap();
            let back: AgentEvent = serde_json::from_str(&wire).unwrap();
            assert_eq!(serde_json::to_string(&back).unwrap(), wire);
        }
    }

    /// An event line carries a flattened `session_id` alongside the tag; the
    /// enum must still parse when that sibling field is present (the validator
    /// deserializes the bare event out of a full envelope line).
    #[test]
    fn event_parses_with_session_id_sibling() {
        let line = r#"{"session_id":"s1","type":"token","text":"hi"}"#;
        let ev: AgentEvent = serde_json::from_str(line).unwrap();
        assert!(matches!(ev, AgentEvent::Token { text } if text == "hi"));
    }
}
