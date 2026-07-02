export interface CatalogModel {
  id: string;
  label: string;
  ram: string;
  note: string;
}

/**
 * Curated MLX models that work well as coding agents. All are mlx-community
 * quantizations pulled from HuggingFace on first start. Any other repo id can
 * be entered manually in settings.
 */
export const MLX_MODELS: CatalogModel[] = [
  {
    id: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    label: "Qwen3 Coder 30B (MoE)",
    ram: "~18 GB",
    note: "Strongest local coding agent; fast for its size (3.3B active params)",
  },
  {
    id: "mlx-community/gpt-oss-20b-MXFP4-Q8",
    label: "gpt-oss 20B (MoE)",
    ram: "~12 GB",
    note: "Very reliable tool calling; reasoning model",
  },
  {
    id: "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    label: "Qwen2.5 Coder 14B",
    ram: "~9 GB",
    note: "Solid mid-size coder",
  },
  {
    id: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    label: "Qwen2.5 Coder 7B",
    ram: "~4.5 GB",
    note: "Light and fast; good starter",
  },
];
