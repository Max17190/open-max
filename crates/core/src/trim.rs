//! Price-aware history trimming (`Settings::cache_price_ratio`): a view of the next request. The stored history keeps
//! everything. Where cached input costs a large share of fresh input, re-sent old output costs more than it helps.
use crate::types::ChatMessage;
use std::borrow::Cow;

/// At a cached-to-fresh price ratio of 0.1 or more, replies older than the newest 8 lose their reasoning and their
/// tool results of 400 bytes or more become a one-line stub naming the tool and its path or command, followed by any
/// harness notes the result carried. The trim point
/// moves in steps of 8 replies, so the cached prefix holds between steps.
pub(crate) fn view(ratio: Option<f64>, messages: &[ChatMessage]) -> Cow<'_, [ChatMessage]> {
    let cut = messages.iter().filter(|m| m.role == "assistant").count().saturating_sub(8) / 8 * 8;
    if cut == 0 || ratio.is_none_or(|ratio| ratio < 0.1) { return Cow::Borrowed(messages); }
    let (mut out, mut seen, mut calls) = (messages.to_vec(), 0, None);
    for (message, sent) in messages.iter().zip(out.iter_mut()) {
        if message.role == "assistant" { seen += 1; calls = (seen <= cut).then(|| message.tool_calls.as_deref().unwrap_or_default()); }
        if message.role == "assistant" && seen <= cut { (sent.reasoning_content, sent.reasoning, sent.reasoning_details) = (None, None, None); }
        // A result matches a call of the reply it follows: call ids need not be unique across replies.
        let len = message.content.as_ref().map_or(0, String::len);
        let Some(call) = calls.filter(|_| len >= 400).and_then(|calls| calls.iter().find(|call| message.tool_call_id.as_deref() == Some(&call.id))) else { continue };
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap_or_default();
        let key: String = ["path", "command", "pattern"].iter().find_map(|key| args[key].as_str()).unwrap_or_default().chars().take(160).collect();
        let notes = notes(message.content.as_deref().unwrap_or_default());
        sent.content = Some(format!("[Old output trimmed: {} {key:?}, {} bytes. Call it again if you need it.]{notes}", call.function.name, len - notes.len()));
        sent.images = None;
    }
    Cow::Owned(out)
}

/// The harness notes a tool result ends with, kept verbatim through a trim: one-line `[...]` notes after a newline
/// (`notes::append_and_emit_note`), and the changes note after `NOTE_SEPARATOR` (`notes::store_note`).
fn notes(content: &str) -> &str {
    let mut cut = content.find(crate::prompt::NOTE_HEADER).map_or(content.len(), |at| content[..at].trim_end_matches('\n').len());
    while let Some(line) = content[..cut].rfind('\n').filter(|&nl| content[nl + 1..cut].starts_with('[') && content[..cut].ends_with(']')) {
        cut = line;
    }
    &content[cut..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ToolCall, ToolCallFunction};

    fn reply(n: usize, args: &str) -> ChatMessage {
        let call = ToolCall { id: format!("c{n}"), kind: "function".into(), function: ToolCallFunction { name: "bash".into(), arguments: args.into() }, extra_content: None };
        ChatMessage { reasoning_content: Some(format!("thought {n}")), ..ChatMessage::assistant(None, Some(vec![call])) }
    }

    /// A system prompt, a user prompt, then `replies` rounds of one bash call with a long result.
    fn history(replies: usize) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system("sys"), ChatMessage::user("task")];
        for n in 0..replies {
            messages.push(reply(n, &format!(r#"{{"command":"cat f{n}"}}"#)));
            messages.push(ChatMessage::tool(format!("c{n}"), "x".repeat(500)));
        }
        messages
    }

    fn trimmed(messages: &[ChatMessage]) -> usize {
        messages.iter().filter(|m| m.content.as_deref().is_some_and(|c| c.starts_with("[Old output trimmed"))).count()
    }

    #[test]
    fn trims_only_at_or_above_the_price_ratio() {
        let messages = history(30);
        for ratio in [None, Some(0.03), Some(0.099)] {
            assert!(matches!(view(ratio, &messages), Cow::Borrowed(_)), "{ratio:?}");
        }
        assert_eq!(trimmed(&view(Some(0.2), &messages)), 16);
        assert_eq!(messages.iter().filter(|m| m.reasoning_content.is_some()).count(), 30, "the stored history keeps all");
    }

    #[test]
    fn trims_in_blocks_so_the_prefix_holds_between_trim_points() {
        let counts: Vec<usize> = (1..=33).map(|n| trimmed(&view(Some(0.2), &history(n)))).collect();
        assert!(counts[..15].iter().all(|&c| c == 0), "{counts:?}");
        assert!(counts[15..23].iter().all(|&c| c == 8), "{counts:?}");
        assert!(counts[23..31].iter().all(|&c| c == 16), "{counts:?}");
        assert_eq!(counts[31], 24);
        // Within a block, each request's history starts with the previous request's history, byte for byte.
        let (a, b) = (view(Some(0.2), &history(20)).into_owned(), view(Some(0.2), &history(21)).into_owned());
        let wire = |m: &[ChatMessage]| serde_json::to_string(m).unwrap();
        assert_eq!(wire(&a), wire(&b[..a.len()]));
    }

    #[test]
    fn stubs_name_the_call_and_keep_recent_reasoning_and_short_results() {
        let mut messages = history(16);
        messages[3].content = Some("short".into());
        let sent = view(Some(0.2), &messages);
        assert_eq!(sent[3].content.as_deref(), Some("short"));
        assert_eq!(sent[5].content.as_deref(), Some(r#"[Old output trimmed: bash "cat f1", 500 bytes. Call it again if you need it.]"#));
        let reasoning: Vec<bool> = sent.iter().filter(|m| m.role == "assistant").map(|m| m.reasoning_content.is_some()).collect();
        assert_eq!(reasoning, [vec![false; 8], vec![true; 8]].concat());
        assert_eq!(sent.last().unwrap().content.as_deref().map(str::len), Some(500));
    }

    #[test]
    fn notes_on_a_trimmed_result_survive_verbatim() {
        let mut messages = history(16);
        let budget = "[harness: request budget: 30 of 60 requests used (50%), 30 left. Prioritize finishing and verifying the task.]";
        let changes = format!("{}{}\n- mode: auto", crate::prompt::NOTE_SEPARATOR, crate::prompt::NOTE_HEADER);
        let tail = format!("\n{budget}\n[harness: identical call, identical output, 3 times in a row.]{changes}");
        messages[5].content = Some(format!("{}\n[not a note, the last line of output]\nend{tail}", "x".repeat(500)));
        messages[7].content = Some(format!("{}{tail}", "y".repeat(500)));
        let sent = view(Some(0.2), &messages);
        let len = messages[5].content.as_ref().unwrap().len() - tail.len();
        assert_eq!(sent[5].content.as_deref(), Some(format!("[Old output trimmed: bash \"cat f1\", {len} bytes. Call it again if you need it.]{tail}").as_str()));
        assert!(sent[7].content.as_deref().unwrap().ends_with(&tail) && !sent[7].content.as_deref().unwrap().contains("yyy"));
    }
}
