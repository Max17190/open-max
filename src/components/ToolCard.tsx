import { memo, useState } from "react";
import { useStore } from "../store";
import type { ChatItem } from "../types";

const TOOL_ICONS: Record<string, string> = {
  read_file: "≡",
  write_file: "✎",
  edit_file: "✎",
  list_dir: "▤",
  glob: "✱",
  grep: "⌕",
  bash: "❯",
};

type ToolItem = Extract<ChatItem, { kind: "tool" }>;

/** Quiet one-line row for a tool call; the diff (if any) is the loud part. */
export const ToolCard = memo(function ToolCard({ item }: { item: ToolItem }) {
  const [expanded, setExpanded] = useState(false);
  const openPanel = useStore((s) => s.openPanel);
  const icon = TOOL_ICONS[item.name] ?? "⚙";

  return (
    <div className={`tool-line status-${item.status}`}>
      <div className="tool-row" onClick={() => setExpanded((e) => !e)}>
        <span className="tool-icon">{icon}</span>
        <span className="tool-name">{item.name}</span>
        <span className="tool-summary">{item.summary}</span>
        <span className="tool-status">
          {item.status === "running" ? <span className="spinner" /> : item.status === "ok" ? "✓" : "✗"}
        </span>
      </div>
      {expanded && item.output && <pre className="tool-output">{item.output}</pre>}
      {item.diff && (
        <div
          className="diff-chip"
          title="Review diff"
          onClick={() => openPanel("diff", item.diff!.path)}
        >
          <span className="diff-chip-path">{item.diff.path}</span>
          <span className="diff-stats">
            <span className="added">+{item.diff.added}</span>{" "}
            <span className="removed">−{item.diff.removed}</span>
          </span>
        </div>
      )}
    </div>
  );
});
