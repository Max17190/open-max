import { listen } from "@tauri-apps/api/event";
import { create } from "zustand";
import { api } from "./api";
import type {
  AgentEvent,
  ChangedFile,
  ChatItem,
  MlxEvent,
  MlxStatus,
  PanelState,
  PendingApproval,
  Project,
  Settings,
  ThreadMeta,
} from "./types";

const MAX_LOG_LINES = 400;
const SAVE_DEBOUNCE_MS = 500;

interface Store {
  projects: Project[];
  threads: Record<string, ThreadMeta[]>; // by project id
  activeProjectId: string | null;
  activeThreadId: string | null;

  chats: Record<string, ChatItem[]>; // by thread id
  running: Record<string, boolean>;
  approvals: Record<string, PendingApproval>;
  changedFiles: Record<string, ChangedFile[]>;

  panel: PanelState | null;
  panelTreeOpen: boolean;

  settings: Settings | null;
  mlx: MlxStatus | null;
  mlxLogs: string[];
  settingsOpen: boolean;

  init: () => Promise<void>;
  addProject: (path: string) => Promise<void>;
  removeProject: (id: string) => Promise<void>;
  selectProject: (id: string) => Promise<void>;
  selectThread: (projectId: string, threadId: string) => Promise<void>;
  createThread: (projectId: string) => Promise<void>;
  deleteThread: (projectId: string, threadId: string) => Promise<void>;

  send: (text: string) => Promise<void>;
  stop: () => void;
  respondApproval: (approved: boolean) => void;

  openPanel: (view: PanelState["view"], path: string) => void;
  togglePanel: () => void;
  closePanel: () => void;
  togglePanelTree: () => void;

  setSettingsOpen: (open: boolean) => void;
  saveSettings: (s: Settings) => Promise<void>;
  refreshMlx: () => Promise<void>;
}

function updateChat(
  chats: Record<string, ChatItem[]>,
  threadId: string,
  fn: (items: ChatItem[]) => ChatItem[],
): Record<string, ChatItem[]> {
  return { ...chats, [threadId]: fn(chats[threadId] ?? []) };
}

export const useStore = create<Store>((set, get) => {
  const loaded = new Set<string>();
  const saveTimers = new Map<string, number>();

  function persistThread(threadId: string) {
    const prev = saveTimers.get(threadId);
    if (prev !== undefined) window.clearTimeout(prev);
    saveTimers.set(
      threadId,
      window.setTimeout(() => {
        saveTimers.delete(threadId);
        const s = get();
        void api
          .saveThreadItems(threadId, {
            items: s.chats[threadId] ?? [],
            changedFiles: s.changedFiles[threadId] ?? [],
          })
          .catch(() => {});
      }, SAVE_DEBOUNCE_MS),
    );
  }

  async function ensureThreadLoaded(threadId: string) {
    if (loaded.has(threadId)) return;
    loaded.add(threadId);
    try {
      const file = await api.loadThreadItems(threadId);
      if (file && Array.isArray(file.items)) {
        set((s) => ({
          chats: { ...s.chats, [threadId]: file.items },
          changedFiles: { ...s.changedFiles, [threadId]: file.changedFiles ?? [] },
        }));
      }
    } catch {
      // a brand-new thread has no items file yet
    }
  }

  async function ensureThreadsListed(projectId: string) {
    if (get().threads[projectId]) return;
    const list = await api.listThreads(projectId);
    set((s) => ({ threads: { ...s.threads, [projectId]: list } }));
  }

  function handleAgentEvent(ev: AgentEvent) {
    const tid = ev.session_id;
    switch (ev.type) {
      case "token":
      case "thinking": {
        set((s) => ({
          chats: updateChat(s.chats, tid, (items) => {
            const last = items[items.length - 1];
            if (last?.kind === "assistant" && last.streaming) {
              const updated =
                ev.type === "token"
                  ? { ...last, text: last.text + ev.text }
                  : { ...last, thinking: last.thinking + ev.text };
              return [...items.slice(0, -1), updated];
            }
            return [
              ...items,
              {
                kind: "assistant",
                id: crypto.randomUUID(),
                text: ev.type === "token" ? ev.text : "",
                thinking: ev.type === "thinking" ? ev.text : "",
                streaming: true,
              },
            ];
          }),
        }));
        break;
      }
      case "tool_start": {
        const summary = summarizeArgs(ev.name, ev.args);
        set((s) => ({
          chats: updateChat(s.chats, tid, (items) => [
            ...items.map((it) =>
              it.kind === "assistant" && it.streaming ? { ...it, streaming: false } : it,
            ),
            {
              kind: "tool",
              id: crypto.randomUUID(),
              callId: ev.call_id,
              name: ev.name,
              summary,
              status: "running",
              output: "",
            },
          ]),
        }));
        break;
      }
      case "tool_end": {
        set((s) => ({
          chats: updateChat(s.chats, tid, (items) =>
            items.map((it) =>
              it.kind === "tool" && it.callId === ev.call_id && it.status === "running"
                ? { ...it, status: ev.ok ? "ok" : "error", output: ev.output }
                : it,
            ),
          ),
        }));
        break;
      }
      case "diff": {
        set((s) => {
          const list = s.changedFiles[tid] ?? [];
          const entry: ChangedFile = { path: ev.path, added: ev.added, removed: ev.removed };
          const next = list.some((f) => f.path === ev.path)
            ? list.map((f) => (f.path === ev.path ? entry : f))
            : [...list, entry];
          return {
            changedFiles: { ...s.changedFiles, [tid]: next },
            chats: updateChat(s.chats, tid, (items) =>
              items.map((it) =>
                it.kind === "tool" && it.callId === ev.call_id
                  ? { ...it, diff: { path: ev.path, diff: ev.diff, added: ev.added, removed: ev.removed } }
                  : it,
              ),
            ),
          };
        });
        break;
      }
      case "approval_request": {
        set((s) => ({
          approvals: {
            ...s.approvals,
            [tid]: { approvalId: ev.approval_id, name: ev.name, summary: ev.summary },
          },
        }));
        break;
      }
      case "error": {
        set((s) => ({
          chats: updateChat(s.chats, tid, (items) => [
            ...items,
            { kind: "error", id: crypto.randomUUID(), text: ev.message },
          ]),
        }));
        break;
      }
      case "done": {
        set((s) => {
          const approvals = { ...s.approvals };
          delete approvals[tid];
          return {
            running: { ...s.running, [tid]: false },
            approvals,
            chats: updateChat(s.chats, tid, (items) =>
              items.map((it) =>
                it.kind === "assistant" && it.streaming ? { ...it, streaming: false } : it,
              ),
            ),
          };
        });
        persistThread(tid);
        break;
      }
    }
  }

  function handleMlxEvent(ev: MlxEvent) {
    if (ev.type === "setup_log" || ev.type === "server_log") {
      set((s) => ({ mlxLogs: [...s.mlxLogs, ev.line].slice(-MAX_LOG_LINES) }));
      return;
    }
    const line =
      ev.type === "setup_done"
        ? `[setup ${ev.ok ? "finished" : "failed"}] ${ev.message}`
        : ev.type === "server_ready"
          ? `[server ready] ${ev.model}`
          : `[server exited with code ${ev.code}]`;
    set((s) => ({ mlxLogs: [...s.mlxLogs, line].slice(-MAX_LOG_LINES) }));
    void get().refreshMlx();
  }

  let initialized = false;

  return {
    projects: [],
    threads: {},
    activeProjectId: null,
    activeThreadId: null,
    chats: {},
    running: {},
    approvals: {},
    changedFiles: {},
    panel: null,
    panelTreeOpen: true,
    settings: null,
    mlx: null,
    mlxLogs: [],
    settingsOpen: false,

    init: async () => {
      if (initialized) return;
      initialized = true;
      await listen<AgentEvent>("agent_event", (e) => handleAgentEvent(e.payload));
      await listen<MlxEvent>("mlx_event", (e) => handleMlxEvent(e.payload));
      const [projects, settings] = await Promise.all([api.listProjects(), api.getSettings()]);
      set({ projects, settings });
      if (projects[0]) await get().selectProject(projects[0].id);
      void get().refreshMlx();
    },

    addProject: async (path) => {
      const projects = await api.addProject(path);
      set({ projects });
      const added = projects.find((p) => p.path === path);
      if (added) await get().selectProject(added.id);
    },

    removeProject: async (id) => {
      const projects = await api.removeProject(id);
      set((s) => ({
        projects,
        activeProjectId: s.activeProjectId === id ? null : s.activeProjectId,
        activeThreadId: s.activeProjectId === id ? null : s.activeThreadId,
      }));
      if (!get().activeProjectId && projects[0]) await get().selectProject(projects[0].id);
    },

    selectProject: async (id) => {
      set({ activeProjectId: id, panel: null });
      await ensureThreadsListed(id);
      const first = get().threads[id]?.[0];
      if (first) {
        await get().selectThread(id, first.id);
      } else {
        set({ activeThreadId: null });
      }
    },

    selectThread: async (projectId, threadId) => {
      set({ activeProjectId: projectId, activeThreadId: threadId, panel: null });
      await ensureThreadLoaded(threadId);
    },

    createThread: async (projectId) => {
      const meta = await api.createThread(projectId);
      loaded.add(meta.id);
      set((s) => ({
        threads: { ...s.threads, [projectId]: [meta, ...(s.threads[projectId] ?? [])] },
        activeProjectId: projectId,
        activeThreadId: meta.id,
        panel: null,
      }));
    },

    deleteThread: async (projectId, threadId) => {
      await api.deleteThread(threadId);
      loaded.delete(threadId);
      set((s) => {
        const list = (s.threads[projectId] ?? []).filter((t) => t.id !== threadId);
        return {
          threads: { ...s.threads, [projectId]: list },
          activeThreadId: s.activeThreadId === threadId ? (list[0]?.id ?? null) : s.activeThreadId,
        };
      });
      const next = get().activeThreadId;
      if (next) await ensureThreadLoaded(next);
    },

    send: async (text) => {
      const { activeProjectId, projects } = get();
      const project = projects.find((p) => p.id === activeProjectId);
      if (!project) return;
      if (!get().activeThreadId) await get().createThread(project.id);
      const threadId = get().activeThreadId;
      if (!threadId) return;

      set((s) => ({
        chats: updateChat(s.chats, threadId, (items) => [
          ...items,
          { kind: "user", id: crypto.randomUUID(), text },
        ]),
        running: { ...s.running, [threadId]: true },
        // Mirror the backend's auto-title locally.
        threads: {
          ...s.threads,
          [project.id]: (s.threads[project.id] ?? []).map((t) =>
            t.id === threadId && t.title === "New thread" ? { ...t, title: text.slice(0, 48) } : t,
          ),
        },
      }));
      try {
        await api.sendMessage(threadId, project.path, text);
      } catch (e) {
        set((s) => ({
          running: { ...s.running, [threadId]: false },
          chats: updateChat(s.chats, threadId, (items) => [
            ...items,
            { kind: "error", id: crypto.randomUUID(), text: String(e) },
          ]),
        }));
      }
    },

    stop: () => {
      const id = get().activeThreadId;
      if (id) void api.stopAgent(id);
    },

    respondApproval: (approved) => {
      const tid = get().activeThreadId;
      if (!tid) return;
      const approval = get().approvals[tid];
      if (!approval) return;
      set((s) => {
        const approvals = { ...s.approvals };
        delete approvals[tid];
        return { approvals };
      });
      void api.respondApproval(approval.approvalId, approved);
    },

    openPanel: (view, path) => set({ panel: { view, path } }),
    togglePanel: () => {
      const { panel, changedFiles, activeThreadId } = get();
      if (panel) {
        set({ panel: null });
        return;
      }
      const changed = activeThreadId ? (changedFiles[activeThreadId] ?? []) : [];
      const last = changed[changed.length - 1];
      set({
        panel: last ? { view: "diff", path: last.path } : { view: "code", path: "" },
        panelTreeOpen: true,
      });
    },
    closePanel: () => set({ panel: null }),
    togglePanelTree: () => set((s) => ({ panelTreeOpen: !s.panelTreeOpen })),

    setSettingsOpen: (open) => set({ settingsOpen: open }),

    saveSettings: async (s) => {
      await api.setSettings(s);
      set({ settings: s });
    },

    refreshMlx: async () => {
      try {
        const mlx = await api.mlxStatus();
        set({ mlx });
      } catch {
        // status is best-effort
      }
    },
  };
});

function summarizeArgs(name: string, args: unknown): string {
  if (typeof args !== "object" || args === null) return "";
  const a = args as Record<string, unknown>;
  switch (name) {
    case "bash":
      return String(a.command ?? "");
    case "read_file":
    case "write_file":
    case "edit_file":
    case "list_dir":
      return String(a.path ?? "");
    case "glob":
    case "grep":
      return String(a.pattern ?? "");
    default:
      return "";
  }
}
