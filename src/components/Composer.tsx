import { useRef, useState } from "react";
import { useStore } from "../store";

export function Composer() {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const activeProjectId = useStore((s) => s.activeProjectId);
  const activeThreadId = useStore((s) => s.activeThreadId);
  const running = useStore((s) => s.running);
  const approvals = useStore((s) => s.approvals);
  const changedFiles = useStore((s) => s.changedFiles);
  const send = useStore((s) => s.send);
  const stop = useStore((s) => s.stop);
  const respondApproval = useStore((s) => s.respondApproval);
  const openPanel = useStore((s) => s.openPanel);

  const approval = activeThreadId ? approvals[activeThreadId] : undefined;
  const changed = activeThreadId ? (changedFiles[activeThreadId] ?? []) : [];
  const isRunning = activeThreadId ? !!running[activeThreadId] : false;
  const disabled = !activeProjectId;

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || isRunning || disabled) return;
    setText("");
    void send(trimmed);
    requestAnimationFrame(() => resize());
  }

  function resize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }

  return (
    <div className="composer-area">
      {changed.length > 0 && (
        <div className="changes-bar">
          {changed.map((f) => (
            <button
              key={f.path}
              className="change-chip"
              title="Review diff"
              onClick={() => openPanel("diff", f.path)}
            >
              <span className="change-name">{f.path.split("/").pop()}</span>
              <span className="added">+{f.added}</span>
              <span className="removed">−{f.removed}</span>
            </button>
          ))}
        </div>
      )}
      {approval && (
        <div className="approval-banner">
          <div className="approval-text">
            <span className="approval-kind">{approval.name}</span>
            <code className="approval-summary">{approval.summary}</code>
          </div>
          <div className="approval-actions">
            <button className="btn btn-ghost" onClick={() => respondApproval(false)}>
              Deny
            </button>
            <button className="btn btn-primary" onClick={() => respondApproval(true)}>
              Approve
            </button>
          </div>
        </div>
      )}
      <div className={`composer ${disabled ? "composer-disabled" : ""}`}>
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          placeholder={disabled ? "Add a project" : "Describe a task"}
          disabled={disabled}
          onChange={(e) => {
            setText(e.target.value);
            resize();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        {isRunning ? (
          <button className="btn btn-stop" onClick={stop} title="Stop">
            ■
          </button>
        ) : (
          <button
            className="btn btn-send"
            onClick={submit}
            disabled={disabled || !text.trim()}
            title="Send"
          >
            ↑
          </button>
        )}
      </div>
    </div>
  );
}
