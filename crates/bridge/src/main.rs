//! `openmax-bridge`: a line-delimited JSON-RPC 2.0 server over stdio that
//! drives a child `openmax --stdio` session, so any host that speaks JSON-RPC
//! can run openmax without knowing its native JSONL envelope. It is a client
//! of the pinned `openmax-stdio/1` contract, not a change to the harness core.
//!
//! Host methods (requests carry `id`, notifications omit it):
//!   initialize                       -> { server, protocol_version, version, session_id }
//!   prompt   { text }                -> streams `update` notifications, resolves at `done`
//!   cancel                           -> { ok: true }
//!   approve  { approval_id, approved}-> answers a pending approval
//!   shutdown                         -> { ok: true }, then the bridge drains and exits
//!
//! Bridge notifications to the host:
//!   update   { params: <AgentEvent> }   one per child event (approvals included)
//!
//! The child binary is `openmax` on PATH, or `$OPENMAX_BIN` when set.

mod translate;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;

use serde_json::{json, Value};
use translate::{
    child_command, classify_child_line, initialize_result, rpc_error, rpc_notification, rpc_result,
    ChildLine, HostRequest,
};

enum Incoming {
    Host(String),
    Child(String),
    HostEof,
    ChildEof,
}

fn main() {
    let bin = std::env::var("OPENMAX_BIN").unwrap_or_else(|_| "openmax".to_string());
    let mut child = match Command::new(&bin)
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("openmax-bridge: cannot spawn '{bin} --stdio': {e}");
            std::process::exit(1);
        }
    };
    let mut child_stdin = child.stdin.take().expect("child stdin piped");
    let mut child_reader = BufReader::new(child.stdout.take().expect("child stdout piped"));

    // Read the hello line synchronously so `initialize` can be answered without
    // racing the child's first write.
    let mut hello: Option<Value> = None;
    {
        let mut first = String::new();
        if child_reader.read_line(&mut first).unwrap_or(0) > 0 {
            match classify_child_line(first.trim()) {
                Ok(ChildLine::Hello(v)) => hello = Some(v),
                _ => eprintln!("openmax-bridge: child did not open with a hello line"),
            }
        }
    }

    let (tx, rx) = mpsc::channel::<Incoming>();

    let tx_host = tx.clone();
    thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    if tx_host.send(Incoming::Host(l)).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx_host.send(Incoming::HostEof);
    });

    let tx_child = tx.clone();
    thread::spawn(move || {
        for line in child_reader.lines() {
            match line {
                Ok(l) => {
                    if tx_child.send(Incoming::Child(l)).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx_child.send(Incoming::ChildEof);
    });
    drop(tx);

    let mut stdout = std::io::stdout();
    // JSON-RPC id of the prompt awaiting its terminal `done`, if any.
    let mut pending_prompt: Option<Value> = None;

    for msg in rx {
        match msg {
            Incoming::Child(line) => {
                if line.trim().is_empty() {
                    continue;
                }
                match classify_child_line(&line) {
                    Ok(ChildLine::Hello(v)) => hello = Some(v),
                    Ok(ChildLine::Done { stop_reason }) => {
                        if let Some(id) = pending_prompt.take() {
                            emit(&mut stdout, &rpc_result(id, json!({"stop_reason": stop_reason})));
                        }
                    }
                    Ok(ChildLine::Update(v)) => {
                        emit(&mut stdout, &rpc_notification("update", v));
                    }
                    Err(reason) => {
                        emit(
                            &mut stdout,
                            &rpc_notification("update", json!({"type": "bridge_error", "message": reason})),
                        );
                    }
                }
            }
            Incoming::Host(line) => {
                if line.trim().is_empty() {
                    continue;
                }
                handle_host_line(&line, &mut child_stdin, &mut stdout, &hello, &mut pending_prompt);
            }
            Incoming::HostEof => {
                // Mirror the child's EOF-is-quit rule: drain the in-flight turn.
                let _ = writeln!(child_stdin, r#"{{"cmd":"quit"}}"#);
                let _ = child_stdin.flush();
            }
            Incoming::ChildEof => break,
        }
    }

    let _ = child.wait();
}

fn handle_host_line(
    line: &str,
    child_stdin: &mut impl Write,
    stdout: &mut impl Write,
    hello: &Option<Value>,
    pending_prompt: &mut Option<Value>,
) {
    let value: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            emit(stdout, &rpc_error(Value::Null, -32700, &format!("parse error: {e}")));
            return;
        }
    };
    let id = value.get("id").cloned().unwrap_or(Value::Null);
    let method = value.get("method").and_then(Value::as_str).unwrap_or("");
    let params = value.get("params").cloned().unwrap_or(Value::Null);

    let req = match HostRequest::parse(method, &params) {
        Ok(r) => r,
        Err(reason) => {
            // -32601 method not found, -32602 invalid params (JSON-RPC 2.0).
            let code = if reason.starts_with("method not found") { -32601 } else { -32602 };
            if !id.is_null() {
                emit(stdout, &rpc_error(id, code, &reason));
            }
            return;
        }
    };

    match &req {
        HostRequest::Initialize => {
            let result = match hello {
                Some(h) => initialize_result(h),
                None => json!({"server": "openmax"}),
            };
            emit(stdout, &rpc_result(id, result));
        }
        HostRequest::Prompt { .. } => {
            if pending_prompt.is_some() {
                emit(stdout, &rpc_error(id, -32000, "a prompt is already in flight"));
                return;
            }
            forward(child_stdin, &req);
            // A request (has id) is resolved at `done`; a notification streams
            // updates only.
            if !id.is_null() {
                *pending_prompt = Some(id);
            }
        }
        HostRequest::Cancel | HostRequest::Shutdown | HostRequest::Approve { .. } => {
            forward(child_stdin, &req);
            if !id.is_null() {
                emit(stdout, &rpc_result(id, json!({"ok": true})));
            }
        }
    }
}

fn forward(child_stdin: &mut impl Write, req: &HostRequest) {
    if let Some(cmd) = child_command(req) {
        let _ = writeln!(child_stdin, "{cmd}");
        let _ = child_stdin.flush();
    }
}

fn emit(out: &mut impl Write, value: &Value) {
    if let Ok(s) = serde_json::to_string(value) {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}
