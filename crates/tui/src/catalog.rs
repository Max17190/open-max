/// Curated MLX models that work well as coding agents, all 4-bit community
/// quantizations pulled from HuggingFace on first use. Any other repo id can
/// be entered with /model. RAM figures are informational; the models panel
/// shows exact sizes from the hub.
pub struct CatalogModel {
    pub id: &'static str,
    pub label: &'static str,
    pub ram: &'static str,
    pub note: &'static str,
}

pub const MLX_MODELS: &[CatalogModel] = &[
    CatalogModel {
        id: "mlx-community/Qwen3.6-35B-A3B-4bit",
        label: "Qwen3.6 35B A3B",
        ram: "~19 GB",
        note: "MoE flagship: fast agentic coding (3B active params)",
    },
    CatalogModel {
        // Repo casing is inconsistent across the Gemma 4 uploads; these ids
        // are copied verbatim from the hub.
        id: "mlx-community/gemma-4-31b-it-4bit",
        label: "Gemma 4 31B",
        ram: "~19 GB",
        note: "Flagship dense Gemma 4 instruct",
    },
    CatalogModel {
        id: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        label: "Qwen3 Coder 30B A3B",
        ram: "~18 GB",
        note: "Agentic MoE coder, strong tool use",
    },
    CatalogModel {
        id: "mlx-community/Qwen3.6-27B-4bit",
        label: "Qwen3.6 27B",
        ram: "~16 GB",
        note: "Best dense coder at consumer scale",
    },
    CatalogModel {
        id: "mlx-community/gemma-4-26b-a4b-it-4bit",
        label: "Gemma 4 26B A4B",
        ram: "~16 GB",
        note: "MoE Gemma 4: fast (4B active params)",
    },
    CatalogModel {
        id: "mlx-community/gpt-oss-20b-MXFP4-Q8",
        label: "gpt-oss 20B",
        ram: "~12 GB",
        note: "Reliable tool calling; adjustable reasoning",
    },
    CatalogModel {
        id: "mlx-community/gemma-4-12B-it-qat-4bit",
        label: "Gemma 4 12B",
        ram: "~11 GB",
        note: "Unified 12B; QAT holds quality at 4-bit",
    },
    CatalogModel {
        id: "mlx-community/gemma-4-e4b-it-4bit",
        label: "Gemma 4 E4B",
        ram: "~5.5 GB",
        note: "Efficient small Gemma 4",
    },
    CatalogModel {
        id: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        label: "Qwen2.5 Coder 7B",
        ram: "~4.5 GB",
        note: "Light and fast starter",
    },
];
