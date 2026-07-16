//! The model manager panel: curated catalog plus whatever is in the local
//! hub cache, with live sizes, RAM-fit dots, download progress, and server
//! control. Rendered inside the live viewport as an overlay mode.

use std::collections::HashMap;

use open_max_core::hf;
use open_max_core::mlx::MlxStatus;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::Frame;

use crate::catalog::MLX_MODELS;
use crate::theme;

#[derive(Clone)]
pub struct ModelItem {
    pub repo: String,
    pub label: String,
    pub note: String,
    /// Exact bytes: from disk when installed, from the hub API otherwise.
    pub bytes: Option<u64>,
    /// Catalog RAM estimate shown until exact bytes are known.
    pub ram_hint: Option<&'static str>,
    pub installed: bool,
}

pub struct ModelsState {
    pub selected: usize,
    pub items: Vec<ModelItem>,
    /// Repo pending delete confirmation.
    pub confirm_delete: Option<String>,
    /// (repo, done, total) while a download runs.
    pub download: Option<(String, u64, u64)>,
    /// User visible message in the panel footer (text, is_error).
    pub footer: Option<(String, bool)>,
    pub ram_bytes: u64,
    pub status: Option<MlxStatus>,
    /// Hub sizes already fetched this run, so refresh() never regresses an
    /// exact size back to a placeholder.
    remote_sizes: HashMap<String, u64>,
    loaded: bool,
}

impl ModelsState {
    /// Empty shell; disk and RAM probing happen on first `/models`.
    pub fn empty() -> Self {
        Self {
            selected: 0,
            items: Vec::new(),
            confirm_delete: None,
            download: None,
            footer: None,
            ram_bytes: 0,
            status: None,
            remote_sizes: HashMap::new(),
            loaded: false,
        }
    }

    /// Load the catalog and local cache the first time the panel opens.
    pub fn ensure_loaded(&mut self, ram_bytes: u64) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        self.ram_bytes = ram_bytes;
        self.refresh();
    }

    /// Rebuild the item list from the catalog and the local cache.
    pub fn refresh(&mut self) {
        let installed = hf::installed_models();
        let mut items: Vec<ModelItem> = MLX_MODELS
            .iter()
            .map(|m| {
                let disk = installed.iter().find(|i| i.repo == m.id);
                ModelItem {
                    repo: m.id.to_string(),
                    label: m.label.to_string(),
                    note: m.note.to_string(),
                    bytes: disk.map(|d| d.bytes).or_else(|| self.remote_sizes.get(m.id).copied()),
                    ram_hint: Some(m.ram),
                    installed: disk.is_some(),
                }
            })
            .collect();
        for i in installed {
            if !items.iter().any(|it| it.repo == i.repo) {
                items.push(ModelItem {
                    repo: i.repo.clone(),
                    label: i.repo.clone(),
                    note: "local cache".into(),
                    bytes: Some(i.bytes),
                    ram_hint: None,
                    installed: true,
                });
            }
        }
        self.items = items;
        self.selected = self.selected.min(self.items.len().saturating_sub(1));
    }

    pub fn selected_repo(&self) -> Option<&str> {
        self.items.get(self.selected).map(|i| i.repo.as_str())
    }

    pub fn selected_item(&self) -> Option<&ModelItem> {
        self.items.get(self.selected)
    }

    /// Record a live size fetched from the hub for a not-installed repo.
    pub fn set_remote_size(&mut self, repo: &str, bytes: u64) {
        self.remote_sizes.insert(repo.to_string(), bytes);
        if let Some(item) = self.items.iter_mut().find(|i| i.repo == repo) {
            if item.bytes.is_none() {
                item.bytes = Some(bytes);
            }
        }
    }
}

pub fn human_bytes(bytes: u64) -> String {
    let gb = bytes as f64 / 1e9;
    if gb >= 1.0 {
        format!("{gb:.1} GB")
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

/// Green under 70% of RAM, yellow under 85%, red above.
fn fit_dot(bytes: Option<u64>, ram: u64) -> Span<'static> {
    let Some(b) = bytes else {
        return Span::styled("· ", Style::default().fg(theme::DIM()));
    };
    let ratio = b as f64 / ram.max(1) as f64;
    let color = if ratio <= 0.70 {
        theme::OK()
    } else if ratio <= 0.85 {
        theme::WARN()
    } else {
        theme::ERR()
    };
    Span::styled("● ", Style::default().fg(color))
}

fn download_label(state: &ModelsState, repo: &str) -> String {
    state
        .items
        .iter()
        .find(|i| i.repo == repo)
        .map(|i| i.label.clone())
        .unwrap_or_else(|| repo.to_string())
}

fn render_download_line(state: &ModelsState, repo: &str, done: u64, total: u64) -> Line<'static> {
    let label = download_label(state, repo);
    let width = 20usize;
    if total > 0 {
        let pct = (done as f64 / total as f64 * 100.0).min(100.0);
        let filled = (pct / 100.0 * width as f64) as usize;
        Line::from(vec![
            Span::styled("pulling ", Style::default().fg(theme::ACCENT())),
            Span::raw(format!("{label}  ")),
            Span::styled(
                format!("▕{}{}▏", "█".repeat(filled), "░".repeat(width.saturating_sub(filled))),
                Style::default().fg(theme::ACCENT()),
            ),
            Span::styled(
                format!("  {pct:>3.0}%  {} / {}", human_bytes(done), human_bytes(total)),
                Style::default().fg(theme::DIM()),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("pulling ", Style::default().fg(theme::ACCENT())),
            Span::raw(format!("{label}  ")),
            Span::styled(format!("▕{}▏", "░".repeat(width)), Style::default().fg(theme::ACCENT())),
            Span::styled(format!("  …  {}", human_bytes(done)), Style::default().fg(theme::DIM())),
        ])
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &ModelsState) {
    let mut lines: Vec<Line> = Vec::new();

    // Header.
    let server = match &state.status {
        Some(s) if s.server_ready => format!(
            "serving {} :{}",
            s.model.as_deref().unwrap_or("?"),
            s.port
        ),
        Some(s) if s.server_running => "server starting…".into(),
        Some(s) if !s.venv_ready => "environment not set up (press u)".into(),
        _ => "server stopped".into(),
    };
    lines.push(Line::from(vec![
        Span::styled("models", Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("  ram {}  ·  {}", human_bytes(state.ram_bytes), server),
            Style::default().fg(theme::DIM()),
        ),
    ]));

    // Rows, windowed around the selection.
    let rows_budget = area.height.saturating_sub(3) as usize;
    let first = state.selected.saturating_sub(rows_budget.saturating_sub(1));
    for (i, item) in state.items.iter().enumerate().skip(first).take(rows_budget) {
        let marker = if i == state.selected {
            Span::styled("▸ ", Style::default().fg(theme::ACCENT()))
        } else {
            Span::raw("  ")
        };
        let serving = state
            .status
            .as_ref()
            .map(|s| s.server_running && s.model.as_deref() == Some(item.repo.as_str()))
            .unwrap_or(false);
        let size = item
            .bytes
            .map(human_bytes)
            .or_else(|| item.ram_hint.map(str::to_string))
            .unwrap_or_else(|| "…".into());
        let mut spans = vec![
            marker,
            fit_dot(item.bytes, state.ram_bytes),
            Span::styled(
                format!("{:<22}", item.label),
                if i == state.selected {
                    Style::default().add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                },
            ),
            Span::styled(format!("{size:>9}  "), Style::default().fg(theme::DIM())),
        ];
        if serving {
            spans.push(Span::styled("serving  ", Style::default().fg(theme::OK())));
        } else if item.installed {
            spans.push(Span::styled("installed  ", Style::default().fg(theme::DIM())));
        }
        spans.push(Span::styled(clip(&item.note, 46), Style::default().fg(theme::DIM())));
        lines.push(Line::from(spans));
    }

    // Download progress, delete confirmation, status message, or key hints.
    if let Some((repo, done, total)) = &state.download {
        lines.push(render_download_line(state, repo, *done, *total));
    } else if let Some(repo) = &state.confirm_delete {
        lines.push(Line::from(vec![
            Span::styled("delete ", Style::default().fg(theme::ERR()).add_modifier(Modifier::BOLD)),
            Span::raw(format!("{repo} from disk?  ")),
            Span::styled("[y] yes  [n] no", Style::default().fg(theme::DIM())),
        ]));
    } else if let Some((msg, is_err)) = &state.footer {
        let style = if *is_err {
            Style::default().fg(theme::ERR())
        } else {
            Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC)
        };
        lines.push(Line::from(Span::styled(msg.clone(), style)));
    } else {
        let oversized = state
            .items
            .get(state.selected)
            .and_then(|i| i.bytes)
            .map(|b| b as f64 / state.ram_bytes.max(1) as f64 > 0.85)
            .unwrap_or(false);
        let hint = if oversized {
            "tight fit: raising the gpu wired limit can help (sudo sysctl iogpu.wired_limit_mb=...)".to_string()
        } else {
            "↑/↓ navigate · enter download or serve · x delete · s stop · u setup · esc close".to_string()
        };
        lines.push(Line::from(Span::styled(hint, Style::default().fg(theme::DIM()))));
    }

    Paragraph::new(lines).render(area, frame.buffer_mut());
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}
