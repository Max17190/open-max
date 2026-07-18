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
    check_jsonrpc_version, child_command, classify_child_line, initialize_result, rpc_error,
    rpc_notification, rpc_result, ChildLine, HostRequest,
};

enum Incoming {
    Host(String),
    Child(String),
    HostEof,
    ChildEof,
}

/// The at-most-one turn in flight. `active` gates concurrent prompts even when
/// the prompt was a notification (no `id`); `id` is the request to resolve at
/// `done`, absent for a notification prompt.
#[derive(Default)]
struct Turn {
    active: bool,
    id: Option<Value>,
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
    let mut turn = Turn::default();

    for msg in rx {
        match msg {
            Incoming::Child(line) => {
                if line.trim().is_empty() {
                    continue;
                }
                match classify_child_line(&line) {
                    Ok(ChildLine::Hello(v)) => hello = Some(v),
                    Ok(ChildLine::Done { stop_reason }) => {
                        turn.active = false;
                        if let Some(id) = turn.id.take() {
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
                handle_host_line(&line, &mut child_stdin, &mut stdout, &hello, &mut turn);
            }
            Incoming::HostEof => {
                // Mirror the child's EOF-is-quit rule: drain the in-flight turn.
                let _ = writeln!(child_stdin, r#"{{"cmd":"quit"}}"#);
                let _ = child_stdin.flush();
            }
            Incoming::ChildEof => {
                // The child died: fail any request still waiting on `done`
                // rather than closing the transport on an unresolved call.
                if let Some(id) = turn.id.take() {
                    emit(
                        &mut stdout,
                        &rpc_error(id, -32001, "child exited before the turn completed"),
                    );
                }
                break;
            }
        }
    }

    let _ = child.wait();
}

fn handle_host_line(
    line: &str,
    child_stdin: &mut impl Write,
    stdout: &mut impl Write,
    hello: &Option<Value>,
    turn: &mut Turn,
) {
    let value: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            emit(stdout, &rpc_error(Value::Null, -32700, &format!("parse error: {e}")));
            return;
        }
    };
    let id = value.get("id").cloned().unwrap_or(Value::Null);

    // A malformed or non-2.0 envelope must never reach a state-changing method.
    if let Err(reason) = check_jsonrpc_version(&value) {
        emit(stdout, &rpc_error(id, -32600, &reason));
        return;
    }

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
            // A notification (no id) gets no response.
            if id.is_null() {
                return;
            }
            match hello {
                Some(h) => emit(stdout, &rpc_result(id, initialize_result(h))),
                // No valid handshake means compatibility is unknown: fail
                // rather than report a partial success without a version.
                None => emit(stdout, &rpc_error(id, -32002, "child did not hand shake")),
            }
        }
        HostRequest::Prompt { .. } => {
            if turn.active {
                if !id.is_null() {
                    emit(stdout, &rpc_error(id, -32000, "a prompt is already in flight"));
                }
                return;
            }
            // Only arm the turn once the command actually reached the child; a
            // failed write would otherwise strand the request with no `done`.
            if let Err(e) = forward(child_stdin, &req) {
                if !id.is_null() {
                    emit(stdout, &rpc_error(id, -32001, &format!("failed to send prompt: {e}")));
                }
                return;
            }
            turn.active = true;
            turn.id = if id.is_null() { None } else { Some(id) };
        }
        HostRequest::Cancel | HostRequest::Shutdown | HostRequest::Approve { .. } => {
            let outcome = forward(child_stdin, &req);
            if id.is_null() {
                return;
            }
            match outcome {
                Ok(()) => emit(stdout, &rpc_result(id, json!({"ok": true}))),
                Err(e) => emit(stdout, &rpc_error(id, -32001, &format!("failed to reach child: {e}"))),
            }
        }
    }
}

fn forward(child_stdin: &mut impl Write, req: &HostRequest) -> std::io::Result<()> {
    if let Some(cmd) = child_command(req) {
        writeln!(child_stdin, "{cmd}")?;
        child_stdin.flush()?;
    }
    Ok(())
}

fn emit(out: &mut impl Write, value: &Value) {
    if let Ok(s) = serde_json::to_string(value) {
        let _ = writeln!(out, "{s}");
        let _ = out.flush();
    }
}
