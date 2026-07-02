//! Client-side extraction of tool calls from raw assistant text.
//!
//! Serving layers turn model-specific tool markup into structured
//! `tool_calls` using per-model parsers, and parser coverage lags newly
//! released models. When that happens the model emits correct markup that
//! leaks into `content` verbatim. This module recognizes the common shapes
//! so the agent keeps working regardless of the serving layer:
//!
//! - `<tool_call>{...}</tool_call>` blocks (Qwen template family), including
//!   an unclosed final tag from a truncated stream
//! - fenced ```tool_call / ```tool_code / ```tool blocks with a JSON body
//! - fenced ```json blocks, accepted only when `name` matches a known tool,
//!   because models legitimately write JSON code blocks in prose
//!
//! Only consulted when a completion carries no native tool calls.

use serde_json::Value;

use crate::types::{ToolCall, ToolCallFunction};

const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";

/// Try to pull tool calls out of raw assistant text. Returns the text with
/// call markup removed plus the synthesized calls, or None if nothing valid
/// was found.
pub fn extract_tool_calls(content: &str, known_tools: &[String]) -> Option<(String, Vec<ToolCall>)> {
    let mut spans: Vec<(usize, usize, ToolCallFunction)> = Vec::new();
    collect_tagged(content, &mut spans);
    collect_fenced(content, known_tools, &mut spans);
    if spans.is_empty() {
        return None;
    }

    // Document order; drop any span that overlaps an earlier one (e.g. a tag
    // quoted inside a fenced block).
    spans.sort_by_key(|s| s.0);
    let mut kept: Vec<(usize, usize, ToolCallFunction)> = Vec::new();
    for span in spans {
        if kept.last().map(|k| span.0 >= k.1).unwrap_or(true) {
            kept.push(span);
        }
    }

    let mut cleaned = String::with_capacity(content.len());
    let mut cursor = 0;
    let mut calls = Vec::new();
    for (i, (start, end, function)) in kept.into_iter().enumerate() {
        cleaned.push_str(&content[cursor..start]);
        cursor = end;
        calls.push(ToolCall {
            id: format!("call_fb_{i}"),
            kind: "function".into(),
            function,
        });
    }
    cleaned.push_str(&content[cursor..]);
    Some((tidy(&cleaned), calls))
}

/// `<tool_call>{json}</tool_call>` blocks. A final unclosed tag is tolerated:
/// streams can end mid-markup while the JSON body is already complete.
fn collect_tagged(content: &str, spans: &mut Vec<(usize, usize, ToolCallFunction)>) {
    let mut from = 0;
    while let Some(rel) = content[from..].find(OPEN_TAG) {
        let start = from + rel;
        let body_start = start + OPEN_TAG.len();
        let (body_end, end) = match content[body_start..].find(CLOSE_TAG) {
            Some(rel_close) => (body_start + rel_close, body_start + rel_close + CLOSE_TAG.len()),
            None => (content.len(), content.len()),
        };
        if let Some(function) = parse_call(content[body_start..body_end].trim(), None) {
            spans.push((start, end, function));
        }
        from = end.max(body_start);
        if from >= content.len() {
            break;
        }
    }
}

/// Fenced code blocks that carry a call. The `json` info string additionally
/// requires a known tool name to avoid eating ordinary JSON examples.
fn collect_fenced(content: &str, known_tools: &[String], spans: &mut Vec<(usize, usize, ToolCallFunction)>) {
    let mut from = 0;
    while let Some(rel) = content[from..].find("```") {
        let fence_start = from + rel;
        let info_start = fence_start + 3;
        let Some(nl) = content[info_start..].find('\n') else { break };
        let info = content[info_start..info_start + nl].trim().to_ascii_lowercase();
        let body_start = info_start + nl + 1;
        let Some(close_rel) = content[body_start..].find("```") else { break };
        let body_end = body_start + close_rel;
        let end = body_end + 3;

        let required = match info.as_str() {
            "tool_call" | "tool_code" | "tool" => None,
            "json" => Some(known_tools),
            _ => {
                from = end;
                continue;
            }
        };
        if let Some(function) = parse_call(content[body_start..body_end].trim(), required) {
            spans.push((fence_start, end, function));
        }
        from = end;
    }
}

/// Parse one candidate JSON body into a call. `required_names`, when given,
/// rejects names outside the known tool set.
fn parse_call(body: &str, required_names: Option<&[String]>) -> Option<ToolCallFunction> {
    let v: Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    if let Some(known) = required_names {
        if !known.iter().any(|k| k == &name) {
            return None;
        }
    }
    let args = v.get("arguments").or_else(|| v.get("parameters")).cloned().unwrap_or(Value::Object(Default::default()));
    // The OpenAI wire format carries arguments as a JSON-encoded string; some
    // models pre-encode it themselves.
    let arguments = match args {
        Value::String(s) => s,
        other => other.to_string(),
    };
    Some(ToolCallFunction { name, arguments })
}

/// Collapse the whitespace holes left where markup was cut out.
fn tidy(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank_run = 0;
    for line in s.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known() -> Vec<String> {
        crate::tools::tool_names()
    }

    #[test]
    fn qwen_single_call_with_prose() {
        let text = "I'll check the directory first.\n<tool_call>\n{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}\n</tool_call>";
        let (clean, calls) = extract_tool_calls(text, &known()).unwrap();
        assert_eq!(clean, "I'll check the directory first.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_dir");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap()["path"],
            "."
        );
    }

    #[test]
    fn qwen_multiple_calls_in_order() {
        let text = "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}</tool_call>\n<tool_call>{\"name\": \"grep\", \"arguments\": {\"pattern\": \"fn main\"}}</tool_call>";
        let (clean, calls) = extract_tool_calls(text, &known()).unwrap();
        assert!(clean.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[1].function.name, "grep");
        assert_eq!(calls[0].id, "call_fb_0");
        assert_eq!(calls[1].id, "call_fb_1");
    }

    #[test]
    fn unclosed_final_tag_is_tolerated() {
        let text = "Running it now.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"cargo test\"}}";
        let (clean, calls) = extract_tool_calls(text, &known()).unwrap();
        assert_eq!(clean, "Running it now.");
        assert_eq!(calls[0].function.name, "bash");
    }

    #[test]
    fn fenced_tool_call_block() {
        let text = "```tool_call\n{\"name\": \"glob\", \"arguments\": {\"pattern\": \"**/*.rs\"}}\n```";
        let (_, calls) = extract_tool_calls(text, &known()).unwrap();
        assert_eq!(calls[0].function.name, "glob");
    }

    #[test]
    fn fenced_json_requires_known_tool() {
        let call = "Some explanation.\n```json\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"todo\"}}\n```";
        let (clean, calls) = extract_tool_calls(call, &known()).unwrap();
        assert_eq!(clean, "Some explanation.");
        assert_eq!(calls[0].function.name, "grep");

        let prose = "Here is a config example:\n```json\n{\"name\": \"my-app\", \"arguments\": {\"port\": 3000}}\n```";
        assert!(extract_tool_calls(prose, &known()).is_none());
    }

    #[test]
    fn plain_json_fence_without_call_shape_is_ignored() {
        let text = "```json\n{\"dependencies\": {\"serde\": \"1\"}}\n```";
        assert!(extract_tool_calls(text, &known()).is_none());
    }

    #[test]
    fn malformed_json_in_tag_is_skipped() {
        let text = "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \n</tool_call>";
        assert!(extract_tool_calls(text, &known()).is_none());
    }

    #[test]
    fn pre_encoded_string_arguments_pass_through() {
        let text = "<tool_call>{\"name\": \"read_file\", \"arguments\": \"{\\\"path\\\": \\\"b.rs\\\"}\"}</tool_call>";
        let (_, calls) = extract_tool_calls(text, &known()).unwrap();
        assert_eq!(calls[0].function.arguments, "{\"path\": \"b.rs\"}");
    }

    #[test]
    fn parameters_key_variant() {
        let text = "<tool_call>{\"name\": \"list_dir\", \"parameters\": {\"path\": \"src\"}}</tool_call>";
        let (_, calls) = extract_tool_calls(text, &known()).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap()["path"],
            "src"
        );
    }

    #[test]
    fn text_without_markup_returns_none() {
        assert!(extract_tool_calls("All done. The tests pass.", &known()).is_none());
    }

    #[test]
    fn tag_quoted_inside_fence_not_double_counted() {
        let text = "```tool_call\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo <tool_call>\"}}\n```";
        let (clean, calls) = extract_tool_calls(text, &known()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert!(clean.is_empty());
    }
}
