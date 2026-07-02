import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { MLX_MODELS } from "../catalog";
import { useStore } from "../store";
import type { Settings } from "../types";

export function SettingsModal() {
  const settings = useStore((s) => s.settings);
  const mlx = useStore((s) => s.mlx);
  const mlxLogs = useStore((s) => s.mlxLogs);
  const setSettingsOpen = useStore((s) => s.setSettingsOpen);
  const saveSettings = useStore((s) => s.saveSettings);
  const refreshMlx = useStore((s) => s.refreshMlx);

  const [form, setForm] = useState<Settings | null>(settings);
  const [mode, setMode] = useState<"mlx" | "custom">(() =>
    settings && settings.base_url.includes(`127.0.0.1:${settings.mlx_port}`) ? "mlx" : "custom",
  );
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logsRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    void refreshMlx();
    const t = setInterval(() => void refreshMlx(), 4000);
    return () => clearInterval(t);
  }, [refreshMlx]);

  useEffect(() => {
    logsRef.current?.scrollTo({ top: logsRef.current.scrollHeight });
  }, [mlxLogs]);

  if (!form) return null;

  function patch(p: Partial<Settings>) {
    setForm((f) => (f ? { ...f, ...p } : f));
  }

  async function run(label: string, fn: () => Promise<void>) {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
      void refreshMlx();
    }
  }

  async function save() {
    if (!form) return;
    const next: Settings =
      mode === "mlx"
        ? {
            ...form,
            base_url: `http://127.0.0.1:${form.mlx_port}/v1`,
            model: form.mlx_model,
            api_key: null,
          }
        : form;
    await run("save", () => saveSettings(next));
    setSettingsOpen(false);
  }

  const inCatalog = MLX_MODELS.some((m) => m.id === form.mlx_model);

  return (
    <div className="modal-backdrop" onClick={() => setSettingsOpen(false)}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Settings</h2>
          <button className="icon-btn" onClick={() => setSettingsOpen(false)}>
            ×
          </button>
        </div>

        <div className="modal-body">
          <section>
            <h3>Model backend</h3>
            <div className="seg">
              <button className={mode === "mlx" ? "seg-on" : ""} onClick={() => setMode("mlx")}>
                Local MLX
              </button>
              <button
                className={mode === "custom" ? "seg-on" : ""}
                onClick={() => setMode("custom")}
              >
                Custom endpoint
              </button>
            </div>

            {mode === "mlx" ? (
              <div className="stack">
                {mlx && !mlx.python_ok && (
                  <p className="warn">python3 was not found on PATH — install Python 3.9+ first.</p>
                )}
                {mlx && !mlx.venv_ready && (
                  <div className="row-between">
                    <span className="muted">
                      {mlx.setting_up
                        ? "Installing mlx-lm into a private environment…"
                        : "MLX runtime is not installed yet."}
                    </span>
                    <button
                      className="btn btn-primary"
                      disabled={mlx.setting_up || !mlx.python_ok}
                      onClick={() => run("setup", () => api.mlxSetup())}
                    >
                      {mlx.setting_up ? "Installing…" : "Install mlx-lm"}
                    </button>
                  </div>
                )}

                <label className="field">
                  <span>Model</span>
                  <select
                    value={inCatalog ? form.mlx_model : "__custom__"}
                    onChange={(e) => {
                      if (e.target.value !== "__custom__") patch({ mlx_model: e.target.value });
                    }}
                  >
                    {MLX_MODELS.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.label} · {m.ram}
                      </option>
                    ))}
                    <option value="__custom__">Custom HuggingFace repo…</option>
                  </select>
                </label>
                {!inCatalog && (
                  <label className="field">
                    <span>HuggingFace repo id</span>
                    <input
                      value={form.mlx_model}
                      placeholder="mlx-community/…"
                      onChange={(e) => patch({ mlx_model: e.target.value })}
                    />
                  </label>
                )}
                {inCatalog && (
                  <p className="muted small">{MLX_MODELS.find((m) => m.id === form.mlx_model)?.note}</p>
                )}

                <div className="row-between">
                  <span className="muted">
                    {mlx?.server_ready
                      ? `Serving ${mlx.model ?? ""} on :${mlx.port}`
                      : mlx?.server_running
                        ? "Starting (downloads the model on first run)…"
                        : "Server stopped"}
                    <span
                      className={`dot ${mlx?.server_ready ? "dot-ok" : mlx?.server_running ? "dot-warn" : "dot-off"}`}
                    />
                  </span>
                  {mlx?.server_running ? (
                    <button className="btn btn-ghost" onClick={() => run("stop", () => api.mlxStop())}>
                      Stop server
                    </button>
                  ) : (
                    <button
                      className="btn btn-primary"
                      disabled={!mlx?.venv_ready || busy === "start"}
                      onClick={() =>
                        run("start", () => api.mlxStart(form.mlx_model, form.mlx_port))
                      }
                    >
                      Start server
                    </button>
                  )}
                </div>

                {mlxLogs.length > 0 && (
                  <pre className="mlx-logs" ref={logsRef}>
                    {mlxLogs.join("\n")}
                  </pre>
                )}
              </div>
            ) : (
              <div className="stack">
                <label className="field">
                  <span>Base URL</span>
                  <input
                    value={form.base_url}
                    placeholder="http://127.0.0.1:11434/v1"
                    onChange={(e) => patch({ base_url: e.target.value })}
                  />
                </label>
                <p className="muted small">
                  Any OpenAI-compatible server: Ollama (:11434), LM Studio (:1234), vLLM, llama.cpp…
                </p>
                <label className="field">
                  <span>Model</span>
                  <input
                    value={form.model}
                    placeholder="qwen3-coder:30b"
                    onChange={(e) => patch({ model: e.target.value })}
                  />
                </label>
                <label className="field">
                  <span>API key (optional)</span>
                  <input
                    type="password"
                    value={form.api_key ?? ""}
                    onChange={(e) => patch({ api_key: e.target.value || null })}
                  />
                </label>
              </div>
            )}
          </section>

          <section>
            <h3>Agent</h3>
            <label className="field">
              <span>Approvals</span>
              <select
                value={form.approval_mode}
                onChange={(e) => patch({ approval_mode: e.target.value as Settings["approval_mode"] })}
              >
                <option value="ask">Ask before edits & commands</option>
                <option value="auto">Run everything automatically</option>
                <option value="readonly">Read-only (never modify files)</option>
              </select>
            </label>
            <div className="field-grid">
              <label className="field">
                <span>Context tokens</span>
                <input
                  type="number"
                  value={form.context_tokens}
                  min={2048}
                  onChange={(e) => patch({ context_tokens: Number(e.target.value) || 16384 })}
                />
              </label>
              <label className="field">
                <span>Max output tokens</span>
                <input
                  type="number"
                  value={form.max_tokens}
                  min={256}
                  onChange={(e) => patch({ max_tokens: Number(e.target.value) || 4096 })}
                />
              </label>
              <label className="field">
                <span>Temperature</span>
                <input
                  type="number"
                  step={0.1}
                  min={0}
                  max={2}
                  value={form.temperature}
                  onChange={(e) => patch({ temperature: Number(e.target.value) })}
                />
              </label>
            </div>
          </section>

          {error && <p className="warn">{error}</p>}
        </div>

        <div className="modal-foot">
          <button className="btn btn-ghost" onClick={() => setSettingsOpen(false)}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={busy !== null} onClick={() => void save()}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
