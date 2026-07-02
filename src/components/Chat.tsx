import { memo, useEffect, useRef, useState } from "react";
import { useStore } from "../store";
import type { ChatItem } from "../types";
import { Markdown } from "./Markdown";
import { ToolCard } from "./ToolCard";

export function Chat() {
  const activeProjectId = useStore((s) => s.activeProjectId);
  const activeThreadId = useStore((s) => s.activeThreadId);
  const chats = useStore((s) => s.chats);
  const running = useStore((s) => s.running);
  const changedFiles = useStore((s) => s.changedFiles);

  const items = activeThreadId ? (chats[activeThreadId] ?? []) : [];
  const isRunning = activeThreadId ? !!running[activeThreadId] : false;
  const changedPaths = activeThreadId
    ? (changedFiles[activeThreadId] ?? []).map((f) => f.path)
    : undefined;

  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinnedRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [items]);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }

  if (!activeProjectId) {
    return (
      <div className="chat-empty">
        <span className="glyph glyph-lg" />
        <p>Add a project to begin</p>
      </div>
    );
  }

  return (
    <div className="chat" ref={scrollRef} onScroll={onScroll}>
      <div className="chat-inner">
        {items.length === 0 && <div className="chat-hint">What should we build?</div>}
        {items.map((item) => (
          <div className="chat-item" key={item.id}>
            <ItemView item={item} changedPaths={changedPaths} />
          </div>
        ))}
        {isRunning && items[items.length - 1]?.kind === "user" && (
          <div className="msg-pending">
            <span className="spinner" />
          </div>
        )}
      </div>
    </div>
  );
}

const ItemView = memo(function ItemView({
  item,
  changedPaths,
}: {
  item: ChatItem;
  changedPaths?: string[];
}) {
  switch (item.kind) {
    case "user":
      return <div className="msg-user">{item.text}</div>;
    case "assistant":
      return (
        <div className="msg-assistant">
          {item.thinking && <Thinking text={item.thinking} done={!item.streaming} />}
          <Markdown text={item.text} filePaths={changedPaths} />
          {item.streaming && <span className="caret" />}
        </div>
      );
    case "tool":
      return <ToolCard item={item} />;
    case "error":
      return <div className="msg-error">{item.text}</div>;
  }
});

function Thinking({ text, done }: { text: string; done: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="thinking">
      <button className="thinking-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} {done ? "thought" : "thinking"}
      </button>
      {open && <pre className="thinking-body">{text}</pre>}
    </div>
  );
}
