import { useEffect } from "react";
import { Chat } from "./components/Chat";
import { CodePanel } from "./components/CodePanel";
import { Composer } from "./components/Composer";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar } from "./components/Sidebar";
import { useStore } from "./store";

export default function App() {
  const init = useStore((s) => s.init);
  const projects = useStore((s) => s.projects);
  const threads = useStore((s) => s.threads);
  const activeProjectId = useStore((s) => s.activeProjectId);
  const activeThreadId = useStore((s) => s.activeThreadId);
  const settings = useStore((s) => s.settings);
  const mlx = useStore((s) => s.mlx);
  const panel = useStore((s) => s.panel);
  const settingsOpen = useStore((s) => s.settingsOpen);
  const setSettingsOpen = useStore((s) => s.setSettingsOpen);
  const togglePanel = useStore((s) => s.togglePanel);

  useEffect(() => {
    void init();
  }, [init]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "e") {
        e.preventDefault();
        togglePanel();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePanel]);

  const project = projects.find((p) => p.id === activeProjectId);
  const thread = activeProjectId
    ? (threads[activeProjectId] ?? []).find((t) => t.id === activeThreadId)
    : undefined;
  const modelName = settings?.model.split("/").pop() ?? "no model";
  const usingMlx = settings ? settings.base_url.includes(`127.0.0.1:${settings.mlx_port}`) : false;
  const dotClass = !usingMlx
    ? "dot-neutral"
    : mlx?.server_ready
      ? "dot-ok"
      : mlx?.server_running
        ? "dot-warn"
        : "dot-off";

  return (
    <div className="app">
      <Sidebar />
      <main className="main">
        <header className="topbar" data-tauri-drag-region>
          <span className="breadcrumb" data-tauri-drag-region>
            {project && <span className="crumb-project">{project.name}</span>}
            {project && thread && <span className="crumb-sep">/</span>}
            {thread && <span className="crumb-thread">{thread.title}</span>}
          </span>
          <div className="topbar-actions">
            {project && (
              <button
                className={`btn btn-ghost btn-small ${panel ? "btn-on" : ""}`}
                onClick={togglePanel}
                title="Toggle code panel (⌘E)"
              >
                Code
              </button>
            )}
            <button className="model-pill" onClick={() => setSettingsOpen(true)} title={settings?.base_url}>
              <span className={`dot ${dotClass}`} />
              {modelName}
            </button>
          </div>
        </header>
        <Chat />
        <Composer />
      </main>
      {panel && <CodePanel />}
      {settingsOpen && <SettingsModal />}
    </div>
  );
}
