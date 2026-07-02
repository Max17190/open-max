import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { highlightFile } from "../highlight";
import { useStore } from "../store";
import type { ChatItem } from "../types";
import { DiffView } from "./DiffView";
import { FileTree } from "./FileTree";

export function CodePanel() {
  const panel = useStore((s) => s.panel);
  const panelTreeOpen = useStore((s) => s.panelTreeOpen);
  const projects = useStore((s) => s.projects);
  const activeProjectId = useStore((s) => s.activeProjectId);
  const activeThreadId = useStore((s) => s.activeThreadId);
  const changedFiles = useStore((s) => s.changedFiles);
  const openPanel = useStore((s) => s.openPanel);
  const closePanel = useStore((s) => s.closePanel);
  const togglePanelTree = useStore((s) => s.togglePanelTree);

  const project = projects.find((p) => p.id === activeProjectId);
  const path = panel?.path ?? "";
  const changed =
    activeThreadId && path
      ? (changedFiles[activeThreadId] ?? []).some((f) => f.path === path)
      : false;
  const view = panel?.view === "diff" && changed ? "diff" : "code";

  const [content, setContent] = useState("");
  const [diff, setDiff] = useState("");

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") closePanel();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [closePanel]);

  useEffect(() => {
    if (!project || !path) return;
    if (view === "code") {
      api
        .readProjectFile(project.path, path)
        .then(setContent)
        .catch((e) => setContent(`Cannot preview: ${String(e)}`));
    } else if (activeThreadId) {
      api
        .threadFileDiff(activeThreadId, project.path, path)
        .then((d) => setDiff(d.diff))
        .catch(() => {
          // Session not in memory (e.g. after restart): fall back to the last
          // inline diff recorded in the transcript.
          const items = useStore.getState().chats[activeThreadId] ?? [];
          const last = [...items]
            .reverse()
            .find(
              (it): it is Extract<ChatItem, { kind: "tool" }> =>
                it.kind === "tool" && it.diff?.path === path,
            );
          if (last?.diff) setDiff(last.diff.diff);
          else openPanel("code", path);
        });
    }
  }, [project, path, view, activeThreadId, openPanel]);

  const highlighted = useMemo(
    () => (view === "code" && path ? highlightFile(path, content) : ""),
    [view, path, content],
  );

  if (!panel || !project) return null;

  return (
    <div className="panel">
      <div className="panel-head">
        <button
          className={`icon-btn ${panelTreeOpen ? "icon-btn-on" : ""}`}
          title="Files"
          onClick={togglePanelTree}
        >
          ☰
        </button>
        <span className="panel-path">{path || project.name}</span>
        {path && (
          <div className="seg seg-small">
            <button className={view === "code" ? "seg-on" : ""} onClick={() => openPanel("code", path)}>
              Code
            </button>
            <button
              className={view === "diff" ? "seg-on" : ""}
              disabled={!changed}
              title={changed ? "Changes in this thread" : "No changes in this thread"}
              onClick={() => openPanel("diff", path)}
            >
              Diff
            </button>
          </div>
        )}
        <button className="icon-btn" onClick={closePanel} title="Close (esc)">
          ×
        </button>
      </div>
      <div className="panel-body">
        {panelTreeOpen && (
          <div className="panel-tree">
            <FileTree project={project} onFile={(rel) => openPanel("code", rel)} />
          </div>
        )}
        <div className="panel-view">
          {!path ? (
            <div className="panel-empty">Select a file</div>
          ) : view === "code" ? (
            <pre className="viewer-body">
              <code dangerouslySetInnerHTML={{ __html: highlighted }} />
            </pre>
          ) : (
            <DiffView diff={diff} />
          )}
        </div>
      </div>
    </div>
  );
}
