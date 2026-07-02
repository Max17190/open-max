import { useEffect, useState } from "react";
import { api } from "../api";
import type { Project, TreeEntry } from "../types";

export function FileTree({ project, onFile }: { project: Project; onFile: (rel: string) => void }) {
  const [children, setChildren] = useState<Record<string, TreeEntry[]>>({});
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    setChildren({});
    setExpanded(new Set());
    api
      .fileTree(project.path, ".")
      .then((entries) => setChildren({ ".": entries }))
      .catch(() => setChildren({ ".": [] }));
  }, [project.id, project.path]);

  async function toggleDir(rel: string) {
    const next = new Set(expanded);
    if (next.has(rel)) {
      next.delete(rel);
    } else {
      next.add(rel);
      if (!children[rel]) {
        try {
          const entries = await api.fileTree(project.path, rel);
          setChildren((c) => ({ ...c, [rel]: entries }));
        } catch {
          setChildren((c) => ({ ...c, [rel]: [] }));
        }
      }
    }
    setExpanded(next);
  }

  function renderLevel(rel: string, depth: number) {
    const entries = children[rel];
    if (!entries) return null;
    return entries.map((e) => (
      <div key={e.rel_path}>
        <div
          className={`tree-row ${e.is_dir ? "tree-dir" : "tree-file"}`}
          style={{ paddingLeft: 14 + depth * 14 }}
          onClick={() => (e.is_dir ? void toggleDir(e.rel_path) : onFile(e.rel_path))}
        >
          <span className="tree-chevron">
            {e.is_dir ? (expanded.has(e.rel_path) ? "▾" : "▸") : ""}
          </span>
          <span className="tree-name">{e.name}</span>
        </div>
        {e.is_dir && expanded.has(e.rel_path) && renderLevel(e.rel_path, depth + 1)}
      </div>
    ));
  }

  return <div className="file-tree">{renderLevel(".", 0)}</div>;
}
