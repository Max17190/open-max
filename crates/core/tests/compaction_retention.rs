mod common;

use common::{drive_turn, recording_endpoint};
use open_max_core::{sessions, state::Core, types::AgentEvent};
use serde_json::{json, Value};

#[tokio::test]
async fn fresh_tool_batch_survives_a_soft_compaction_trigger() {
    fresh_tool_batch_survives(6, 65_536, false).await;
}

#[tokio::test]
async fn fresh_tool_batch_survives_a_hard_context_limit() {
    fresh_tool_batch_survives(8, 21_504, true).await;
}

async fn fresh_tool_batch_survives(call_count: usize, context_tokens: usize, exceeds_window: bool) {
    const TASK: &str = "Read the files and report their end markers.";
    const REFUSAL: &str = "prompt exceeds the context window";
    let dir =
        std::env::temp_dir().join(format!("openmax-compaction-batch-{}", uuid::Uuid::new_v4()));
    let data = dir.join("data");
    let project = dir.join("project");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&project).unwrap();
    let project = project.canonicalize().unwrap();

    let calls: Vec<_> = (0..call_count)
        .map(|i| {
            let path = format!("part-{i}.txt");
            let body = "source text to inspect while checking files on disk.\n".repeat(265)
                + &format!("END_MARKER_{i}\n");
            std::fs::write(project.join(&path), body).unwrap();
            json!({
                "id": format!("call-{i}"),
                "type": "function",
                "function": {"name": "read_file", "arguments": json!({"path": path}).to_string()}
            })
        })
        .collect();
    let answer = json!({
        "choices": [{"message": {"role": "assistant", "content": "Results recorded."}, "finish_reason": "stop"}]
    });
    let reply = if exceeds_window {
        json!({"error": {"code": 400, "message": REFUSAL}})
    } else {
        answer.clone()
    };
    // The third response lets the fixture still capture the main request if
    // a regression inserts an auxiliary summary request first.
    let (url, requests) = recording_endpoint(vec![
        json!({
            "choices": [{"message": {"role": "assistant", "tool_calls": calls}, "finish_reason": "tool_calls"}]
        }),
        reply,
        answer,
    ]).await;
    std::fs::write(
        data.join("settings.json"),
        json!({
            "base_url": url,
            "model": "fixture",
            "context_tokens": context_tokens,
            "max_tokens": 512,
            "compaction_tokens": 20_000
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        data.join("trust.json"),
        json!({
            "version": 1,
            "projects": [project]
        })
        .to_string(),
    )
    .unwrap();

    let (core, mut events) = Core::new(data).unwrap();
    let id = sessions::create(&core, project.display().to_string())
        .unwrap()
        .id;
    let events = drive_turn(&core, &mut events, &id, &project, TASK).await;
    let archive_exists = std::path::Path::new(&sessions::archive_display(&core, &id)).exists();
    std::fs::remove_dir_all(dir).unwrap();

    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolEnd { ok: true, .. }))
            .count(),
        call_count
    );
    let requests = requests.lock().unwrap();
    let main: Vec<Value> = requests
        .iter()
        .map(|request| serde_json::from_str(request).unwrap())
        .filter(|request: &Value| {
            request["tools"]
                .as_array()
                .is_some_and(|tools| !tools.is_empty())
        })
        .collect();
    assert_eq!(
        main.len(),
        2,
        "the turn must send a request after reading the files"
    );
    let messages = main[1]["messages"].as_array().unwrap();
    let results: Vec<_> = messages
        .iter()
        .filter(|message| message["role"] == "tool")
        .collect();
    assert_eq!(
        results.len(),
        call_count,
        "executed tool results in the next main request; {} total requests",
        requests.len()
    );
    assert_eq!(
        requests.len(),
        2,
        "keeping the active exchange needs no summary request"
    );
    assert!(!archive_exists);
    assert!(messages.iter().any(|message| message["content"] == TASK));
    let batch = messages
        .iter()
        .find_map(|message| message["tool_calls"].as_array())
        .unwrap();
    assert_eq!(batch.len(), call_count);
    for (i, (call, result)) in batch.iter().zip(results).enumerate() {
        assert_eq!(call["id"], format!("call-{i}"));
        assert_eq!(result["tool_call_id"], call["id"]);
        let body = result["content"].as_str().unwrap();
        assert!(
            body.contains(&format!("END_MARKER_{i}")),
            "each result retains its body"
        );
        assert!(!body.contains("older tool output truncated"));
    }
    let used = events
        .iter()
        .rev()
        .find_map(|event| match event {
            AgentEvent::Budget { used_tokens, .. } => Some(*used_tokens),
            _ => None,
        })
        .unwrap();
    assert!(
        used > 20_000,
        "the batch crosses the configured soft trigger"
    );
    assert_eq!(used > context_tokens - 512 - 1024, exceeds_window);
    let errors: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Error { message } => Some(message),
            _ => None,
        })
        .collect();
    if exceeds_window {
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains(REFUSAL));
        assert!(errors[0].contains("this request was over budget before it was sent"));
    } else {
        assert!(errors.is_empty(), "{errors:?}");
    }
}
