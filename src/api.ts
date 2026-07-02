import { invoke } from "@tauri-apps/api/core";
import type {
  DiffInfo,
  MlxStatus,
  Project,
  Settings,
  ThreadItemsFile,
  ThreadMeta,
  TreeEntry,
} from "./types";

export const api = {
  getSettings: () => invoke<Settings>("get_settings"),
  setSettings: (newSettings: Settings) => invoke<void>("set_settings", { newSettings }),

  listProjects: () => invoke<Project[]>("list_projects"),
  addProject: (path: string) => invoke<Project[]>("add_project", { path }),
  removeProject: (id: string) => invoke<Project[]>("remove_project", { id }),
  fileTree: (projectPath: string, rel: string) =>
    invoke<TreeEntry[]>("file_tree", { projectPath, rel }),
  readProjectFile: (projectPath: string, rel: string) =>
    invoke<string>("read_project_file", { projectPath, rel }),

  listThreads: (projectId: string) => invoke<ThreadMeta[]>("list_threads", { projectId }),
  createThread: (projectId: string) => invoke<ThreadMeta>("create_thread", { projectId }),
  deleteThread: (threadId: string) => invoke<void>("delete_thread", { threadId }),
  loadThreadItems: (threadId: string) =>
    invoke<ThreadItemsFile | null>("load_thread_items", { threadId }),
  saveThreadItems: (threadId: string, items: ThreadItemsFile) =>
    invoke<void>("save_thread_items", { threadId, items }),
  threadFileDiff: (threadId: string, projectPath: string, rel: string) =>
    invoke<DiffInfo>("thread_file_diff", { threadId, projectPath, rel }),

  sendMessage: (sessionId: string, projectPath: string, text: string) =>
    invoke<void>("send_message", { sessionId, projectPath, text }),
  stopAgent: (sessionId: string) => invoke<void>("stop_agent", { sessionId }),
  respondApproval: (approvalId: string, approved: boolean) =>
    invoke<void>("respond_approval", { approvalId, approved }),

  mlxStatus: () => invoke<MlxStatus>("mlx_status"),
  mlxSetup: () => invoke<void>("mlx_setup"),
  mlxStart: (model: string, port: number) => invoke<void>("mlx_start", { model, port }),
  mlxStop: () => invoke<void>("mlx_stop"),
  mlxLogs: () => invoke<string[]>("mlx_logs"),
};
