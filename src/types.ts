export interface Project {
  id: string;
  name: string;
  path: string;
}

export interface ThreadMeta {
  id: string;
  project_id: string;
  title: string;
  created_at: number;
  updated_at: number;
}

export interface TreeEntry {
  name: string;
  rel_path: string;
  is_dir: boolean;
}

export interface Settings {
  base_url: string;
  api_key: string | null;
  model: string;
  approval_mode: "auto" | "ask" | "readonly";
  context_tokens: number;
  max_tokens: number;
  temperature: number;
  mlx_model: string;
  mlx_port: number;
}

export interface MlxStatus {
  python_ok: boolean;
  venv_ready: boolean;
  setting_up: boolean;
  server_running: boolean;
  server_ready: boolean;
  model: string | null;
  port: number;
}

export interface DiffInfo {
  path: string;
  diff: string;
  added: number;
  removed: number;
}

export interface ChangedFile {
  path: string;
  added: number;
  removed: number;
}

/** Mirror of the Rust AgentEvent enum (serde snake_case tagged). */
export type AgentEvent = { session_id: string } & (
  | { type: "token"; text: string }
  | { type: "thinking"; text: string }
  | { type: "tool_start"; call_id: string; name: string; args: unknown }
  | { type: "tool_end"; call_id: string; ok: boolean; output: string }
  | ({ type: "diff"; call_id: string } & DiffInfo)
  | { type: "approval_request"; approval_id: string; name: string; summary: string }
  | { type: "done"; stop_reason: string }
  | { type: "error"; message: string }
);

export type MlxEvent =
  | { type: "setup_log"; line: string }
  | { type: "setup_done"; ok: boolean; message: string }
  | { type: "server_log"; line: string }
  | { type: "server_ready"; model: string }
  | { type: "server_exit"; code: number };

export type ChatItem =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string; thinking: string; streaming: boolean }
  | {
      kind: "tool";
      id: string;
      callId: string;
      name: string;
      summary: string;
      status: "running" | "ok" | "error";
      output: string;
      diff?: DiffInfo;
    }
  | { kind: "error"; id: string; text: string };

/** Persisted per-thread UI state ({thread}.items.json). */
export interface ThreadItemsFile {
  items: ChatItem[];
  changedFiles: ChangedFile[];
}

export interface PendingApproval {
  approvalId: string;
  name: string;
  summary: string;
}

export interface PanelState {
  view: "code" | "diff";
  path: string;
}
