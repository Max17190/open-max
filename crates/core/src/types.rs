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

/// Events streamed from the agent loop to the frontend.
#[derive(Clone, Debug, Serialize)]
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
