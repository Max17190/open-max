import DOMPurify from "dompurify";
import { marked } from "marked";
import { memo, useMemo } from "react";
import { useStore } from "../store";

marked.setOptions({ gfm: true, breaks: true });

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Renders assistant markdown. Inline code spans that name a file the thread
 * changed become clickable links into the code panel (delegated click handler,
 * no per-node listeners).
 */
export const Markdown = memo(function Markdown({
  text,
  filePaths,
}: {
  text: string;
  filePaths?: string[];
}) {
  const openPanel = useStore((s) => s.openPanel);

  const html = useMemo(() => {
    let out = DOMPurify.sanitize(marked.parse(text, { async: false }) as string);
    if (filePaths?.length) {
      for (const p of filePaths) {
        const esc = escapeHtml(p);
        out = out.replaceAll(
          `<code>${esc}</code>`,
          `<code class="file-link" data-path="${esc}">${esc}</code>`,
        );
      }
    }
    return out;
  }, [text, filePaths]);

  return (
    <div
      className="md"
      onClick={(e) => {
        const el = (e.target as HTMLElement).closest("code.file-link");
        const path = el?.getAttribute("data-path");
        if (path) openPanel("diff", path);
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});
