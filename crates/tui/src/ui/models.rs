//! The model manager panel: curated catalog plus whatever is in the local
//! hub cache, with live sizes, RAM-fit dots, download progress, and server
//! control. Rendered inside the live viewport as an overlay mode.

use open_max_core::hf;
use open_max_core::mlx::MlxStatus;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::Frame;

use crate::catalog::MLX_MODELS;
use crate::theme;

pub struct ModelItem {
    pub repo: String,
    pub label: String,
    pub note: String,
    /// Exact bytes: from disk when installed, from the hub API otherwise.
    pub bytes: Option<u64>,
    pub installed: bool,
}

pub struct ModelsState {
    pub selected: usize,
    pub items: Vec<ModelItem>,
    /// Repo pending delete confirmation.
    pub confirm_delete: Option<String>,
    /// (repo, done, total) while a download runs.
    pub download: Option<(String, u64, u64)>,
    pub ram_bytes: u64,
    pub status: Option<MlxStatus>,
}

impl ModelsState {
    pub fn new(ram_bytes: u64) -> Self {
        let mut s = Self {
            selected: 0,
            items: Vec::new(),
            confirm_delete: None,
            download: None,
            ram_bytes,
            status: None,
        };
        s.refresh();
        s
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
                    bytes: disk.map(|d| d.bytes),
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

    /// Record a live size fetched from the hub for a not-installed repo.
    pub fn set_remote_size(&mut self, repo: &str, bytes: u64) {
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
        return Span::styled("· ", Style::default().fg(theme::DIM));
    };
    let ratio = b as f64 / ram.max(1) as f64;
    let color = if ratio <= 0.70 {
        theme::OK
    } else if ratio <= 0.85 {
        theme::WARN
    } else {
        theme::ERR
    };
    Span::styled("● ", Style::default().fg(color))
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
        Span::styled("models", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("  ram {}  ·  {}", human_bytes(state.ram_bytes), server),
            Style::default().fg(theme::DIM),
        ),
    ]));

    // Rows, windowed around the selection.
    let rows_budget = area.height.saturating_sub(3) as usize;
    let first = state.selected.saturating_sub(rows_budget.saturating_sub(1));
    for (i, item) in state.items.iter().enumerate().skip(first).take(rows_budget) {
        let marker = if i == state.selected {
            Span::styled("▸ ", Style::default().fg(theme::ACCENT))
        } else {
            Span::raw("  ")
        };
        let serving = state
            .status
            .as_ref()
            .map(|s| s.server_running && s.model.as_deref() == Some(item.repo.as_str()))
            .unwrap_or(false);
        let size = item.bytes.map(human_bytes).unwrap_or_else(|| "…".into());
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
            Span::styled(format!("{size:>9}  "), Style::default().fg(theme::DIM)),
        ];
        if serving {
            spans.push(Span::styled("serving  ", Style::default().fg(theme::OK)));
        } else if item.installed {
            spans.push(Span::styled("installed  ", Style::default().fg(theme::DIM)));
        }
        spans.push(Span::styled(clip(&item.note, 46), Style::default().fg(theme::DIM)));
        lines.push(Line::from(spans));
    }

    // Download progress, delete confirmation, or key hints.
    if let Some((repo, done, total)) = &state.download {
        let pct = if *total > 0 { (*done as f64 / *total as f64 * 100.0).min(100.0) } else { 0.0 };
        let width = 24usize;
        let filled = (pct / 100.0 * width as f64) as usize;
        lines.push(Line::from(vec![
            Span::styled("⇣ ", Style::default().fg(theme::ACCENT)),
            Span::raw(format!("{repo}  ")),
            Span::styled(
                format!("{}{}", "█".repeat(filled), "░".repeat(width - filled)),
                Style::default().fg(theme::ACCENT),
            ),
            Span::styled(
                format!(" {:.0}%  {} / {}", pct, human_bytes(*done), human_bytes(*total)),
                Style::default().fg(theme::DIM),
            ),
        ]));
    } else if let Some(repo) = &state.confirm_delete {
        lines.push(Line::from(vec![
            Span::styled("delete ", Style::default().fg(theme::ERR).add_modifier(Modifier::BOLD)),
            Span::raw(format!("{repo} from disk?  ")),
            Span::styled("[y] yes  [n] no", Style::default().fg(theme::DIM)),
        ]));
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
            "enter serve · d download · x delete · s stop server · u set up env · esc close".to_string()
        };
        lines.push(Line::from(Span::styled(hint, Style::default().fg(theme::DIM))));
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
