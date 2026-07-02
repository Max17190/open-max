import { open } from "@tauri-apps/plugin-dialog";
import { useState } from "react";
import { useStore } from "../store";

export function Sidebar() {
  const projects = useStore((s) => s.projects);
  const threads = useStore((s) => s.threads);
  const activeProjectId = useStore((s) => s.activeProjectId);
  const activeThreadId = useStore((s) => s.activeThreadId);
  const running = useStore((s) => s.running);
  const approvals = useStore((s) => s.approvals);
  const selectProject = useStore((s) => s.selectProject);
  const selectThread = useStore((s) => s.selectThread);
  const createThread = useStore((s) => s.createThread);
  const deleteThread = useStore((s) => s.deleteThread);
  const addProject = useStore((s) => s.addProject);
  const removeProject = useStore((s) => s.removeProject);

  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  function toggleCollapsed(id: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function pickProject() {
    const path = await open({ directory: true, multiple: false, title: "Add project" });
    if (typeof path === "string") await addProject(path);
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-head" data-tauri-drag-region>
        <span className="wordmark" data-tauri-drag-region>
          <span className="glyph" />
          openmax
        </span>
      </div>

      <div className="sidebar-body">
        {projects.map((p) => {
          const isCollapsed = collapsed.has(p.id);
          const projectThreads = threads[p.id] ?? [];
          return (
            <div key={p.id} className="project-group">
              <div
                className={`project-row ${p.id === activeProjectId ? "active" : ""}`}
                title={p.path}
                onClick={() => {
                  if (p.id === activeProjectId) toggleCollapsed(p.id);
                  else void selectProject(p.id);
                }}
              >
                <span className="project-chevron">{isCollapsed ? "▸" : "▾"}</span>
                <span className="project-name">{p.name}</span>
                <button
                  className="icon-btn row-action"
                  title="New thread"
                  onClick={(e) => {
                    e.stopPropagation();
                    void createThread(p.id);
                  }}
                >
                  +
                </button>
                <button
                  className="icon-btn row-action"
                  title="Remove project"
                  onClick={(e) => {
                    e.stopPropagation();
                    void removeProject(p.id);
                  }}
                >
                  ×
                </button>
              </div>

              {!isCollapsed &&
                projectThreads.map((t) => {
                  const isRunning = !!running[t.id];
                  const needsApproval = !!approvals[t.id];
                  return (
                    <div
                      key={t.id}
                      className={`thread-row ${t.id === activeThreadId ? "active" : ""}`}
                      onClick={() => void selectThread(p.id, t.id)}
                    >
                      <span
                        className={`thread-dot ${
                          needsApproval ? "dot-approval" : isRunning ? "dot-running" : ""
                        }`}
                      />
                      <span className="thread-title">{t.title}</span>
                      <button
                        className="icon-btn row-action"
                        title="Delete thread"
                        onClick={(e) => {
                          e.stopPropagation();
                          void deleteThread(p.id, t.id);
                        }}
                      >
                        ×
                      </button>
                    </div>
                  );
                })}
            </div>
          );
        })}

        <button className="add-project" onClick={pickProject}>
          + Add project
        </button>
      </div>
    </aside>
  );
}
