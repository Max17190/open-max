//! Searchable `/model` picker backed only by user-owned provider catalogs.

use std::path::Path;

use open_max_core::providers;
use ratatui::layout::{Position, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

use crate::{completion, theme};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelChoice {
    pub provider: Option<String>,
    pub id: String,
    pub name: String,
    pub active: bool,
}

impl ModelChoice {
    fn search_text(&self) -> String {
        format!(
            "{} {} {}",
            self.provider.as_deref().unwrap_or("direct"),
            self.name,
            self.id
        )
    }
}

pub struct ModelPickerState {
    pub items: Vec<ModelChoice>,
    pub filtered: Vec<usize>,
    pub selected: usize,
    pub query: String,
}

impl ModelPickerState {
    pub fn load(data_dir: &Path, active_provider: Option<&str>, active_model: &str) -> Self {
        let catalogs = providers::load_providers(data_dir);
        let mut items = Vec::new();
        for (provider, config) in catalogs {
            for model in config.models {
                let active = active_provider == Some(provider.as_str()) && model.id == active_model;
                items.push(ModelChoice {
                    provider: Some(provider.clone()),
                    name: model.name.unwrap_or_else(|| model.id.clone()),
                    id: model.id,
                    active,
                });
            }
        }

        let current_present = items
            .iter()
            .any(|item| item.provider.as_deref() == active_provider && item.id == active_model);
        if !current_present {
            items.push(ModelChoice {
                provider: active_provider.map(str::to_string),
                id: active_model.to_string(),
                name: active_model.to_string(),
                active: true,
            });
        }
        items.sort_by(|a, b| {
            b.active.cmp(&a.active).then_with(|| {
                a.provider
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.provider.as_deref().unwrap_or(""))
            })
        });

        let filtered = (0..items.len()).collect();
        Self {
            items,
            filtered,
            selected: 0,
            query: String::new(),
        }
    }

    pub fn selected_choice(&self) -> Option<&ModelChoice> {
        self.filtered
            .get(self.selected)
            .and_then(|index| self.items.get(*index))
    }

    pub fn push(&mut self, c: char) {
        self.query.push(c);
        self.refilter();
    }

    pub fn backspace(&mut self) {
        self.query.pop();
        self.refilter();
    }

    pub fn next(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = (self.selected + 1) % self.filtered.len();
        }
    }

    pub fn prev(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = (self.selected + self.filtered.len() - 1) % self.filtered.len();
        }
    }

    fn refilter(&mut self) {
        let query = self.query.trim();
        let mut scored: Vec<(i32, usize)> = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                completion::fuzzy_score(&item.search_text(), query).map(|score| (score, index))
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| self.items[b.1].active.cmp(&self.items[a.1].active))
                .then_with(|| a.1.cmp(&b.1))
        });
        self.filtered = scored.into_iter().map(|(_, index)| index).collect();
        self.selected = 0;
    }
}

pub fn render(frame: &mut Frame, area: Rect, state: &ModelPickerState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(" Model ", theme::ACCENT_BOLD));
    let inner = block.inner(area);
    block.render(area, frame.buffer_mut());
    let mut lines = vec![Line::from(vec![
        Span::styled("⌕ ", Style::default().fg(theme::ACCENT)),
        Span::styled(
            if state.query.is_empty() {
                "type to filter".to_string()
            } else {
                state.query.clone()
            },
            if state.query.is_empty() {
                theme::DIM_ITALIC
            } else {
                Style::default()
            },
        ),
    ])];

    let rows = inner.height.saturating_sub(2) as usize;
    let visible = state.filtered.len().min(rows);
    let first = state
        .selected
        .saturating_sub(visible.saturating_sub(1))
        .min(state.filtered.len().saturating_sub(visible));
    for (row, item_index) in state.filtered.iter().enumerate().skip(first).take(visible) {
        let item = &state.items[*item_index];
        let selected = row == state.selected;
        let provider = item.provider.as_deref().unwrap_or("direct");
        let marker = if selected { "▸" } else { " " };
        let active = if item.active { "●" } else { " " };
        let base = if selected { theme::ACCENT_BOLD } else { Style::default() };
        let row_width = inner.width as usize;
        let name_width = if row_width >= 72 {
            24
        } else if row_width >= 48 {
            18
        } else {
            10
        };
        let provider_width = if row_width >= 72 {
            14
        } else if row_width >= 40 {
            10
        } else {
            8
        };
        let id_width = row_width.saturating_sub(4 + name_width + provider_width);
        lines.push(Line::from(vec![
            Span::styled(format!("{marker} {active} "), base),
            Span::styled(
                super::text::pad_right(&clip(&item.name, name_width.saturating_sub(1)), name_width),
                base,
            ),
            Span::styled(
                super::text::pad_right(
                    &clip(provider, provider_width.saturating_sub(1)),
                    provider_width,
                ),
                base.fg(theme::DIM),
            ),
            Span::styled(clip(&item.id, id_width), base.fg(theme::DIM)),
        ]));
    }
    if state.filtered.is_empty() {
        lines.push(Line::from(Span::styled("  no matching configured models", theme::DIM_ITALIC)));
    }
    lines.push(Line::from(Span::styled(
        "↑/↓ navigate · enter select · type to filter · esc close",
        Style::default().fg(theme::DIM),
    )));
    Paragraph::new(lines).render(inner, frame.buffer_mut());
    if inner.width > 2 && inner.height > 0 {
        let cursor_x = inner
            .x
            .saturating_add(2)
            .saturating_add(state.query.as_str().width() as u16)
            .min(inner.right().saturating_sub(1));
        frame.set_cursor_position(Position::new(cursor_x, inner.y));
    }
}

fn clip(s: &str, max: usize) -> String {
    super::text::clip(s, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Modifier;
    use std::fs;

    fn fixture() -> std::path::PathBuf {
        let dir = crate::test_temp_dir("openmax-model-picker");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("providers.json"),
            r#"{
              "providers": {
                "beta": {
                  "base_url": "http://beta/v1",
                  "models": [{"id":"org/model/2","name":"Model Two"},{"id":"shared"}]
                },
                "alpha": {
                  "base_url": "http://alpha/v1",
                  "models": [{"id":"one","name":"Model One"},{"id":"shared"}]
                }
              }
            }"#,
        )
        .unwrap();
        providers::invalidate_providers_cache();
        dir
    }

    #[test]
    fn active_choice_is_first_and_slashes_survive() {
        let dir = fixture();
        let state = ModelPickerState::load(&dir, Some("beta"), "org/model/2");
        let active = &state.items[0];
        assert!(active.active);
        assert_eq!(active.provider.as_deref(), Some("beta"));
        assert_eq!(active.id, "org/model/2");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_active_model_is_still_selectable() {
        let dir = fixture();
        let state = ModelPickerState::load(&dir, Some("alpha"), "custom/not-listed");
        assert_eq!(state.items[0].id, "custom/not-listed");
        assert!(state.items[0].active);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn filter_matches_provider_name_and_id() {
        let dir = fixture();
        let mut state = ModelPickerState::load(&dir, Some("alpha"), "one");
        for c in "beta".chars() {
            state.push(c);
        }
        assert_eq!(state.filtered.len(), 2);
        assert!(state
            .filtered
            .iter()
            .all(|index| { state.items[*index].provider.as_deref() == Some("beta") }));
        state.query.clear();
        state.refilter();
        for c in "two".chars() {
            state.push(c);
        }
        assert_eq!(state.selected_choice().unwrap().name, "Model Two");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn duplicate_ids_remain_distinct_and_provider_order_is_stable() {
        let dir = fixture();
        let state = ModelPickerState::load(&dir, None, "direct/current");
        let shared: Vec<_> = state
            .items
            .iter()
            .filter(|item| item.id == "shared")
            .map(|item| item.provider.as_deref().unwrap())
            .collect();
        assert_eq!(shared, vec!["alpha", "beta"]);
        let alpha: Vec<_> = state
            .items
            .iter()
            .filter(|item| item.provider.as_deref() == Some("alpha"))
            .map(|item| item.id.as_str())
            .collect();
        assert_eq!(alpha, vec!["one", "shared"]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn picker_renders_in_a_narrow_buffer_with_selected_style() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let dir = fixture();
        let state = ModelPickerState::load(&dir, Some("alpha"), "one");
        let backend = TestBackend::new(32, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render(frame, frame.area(), &state))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let text: String = buffer.content().iter().map(|cell| cell.symbol()).collect();
        assert!(text.contains("Model"));
        assert!(text.contains("Model One"));
        assert_eq!(buffer[(1, 2)].symbol(), "▸");
        assert!(buffer[(1, 2)].modifier.contains(Modifier::BOLD));
        let _ = fs::remove_dir_all(dir);
    }

    /// An active model a catalog lists is not added a second time, and
    /// stepping through the list wraps around at both ends.
    #[test]
    fn a_listed_active_model_appears_once_and_the_selection_wraps() {
        let dir = fixture();
        let mut state = ModelPickerState::load(&dir, Some("alpha"), "one");
        assert_eq!(state.items.len(), 4);
        let mut stepped = Vec::new();
        for _ in 0..4 {
            state.next();
            stepped.push(state.selected);
        }
        assert_eq!(stepped, [1, 2, 3, 0]);
        state.prev();
        assert_eq!(state.selected, 3);
        let _ = fs::remove_dir_all(dir);
    }

    /// Each row lays out marker, name, provider and id in columns whose
    /// widths follow the picker's width, and the cursor sits in the query
    /// line only while the picker has room for it.
    #[test]
    fn rows_lay_out_columns_by_width_and_the_cursor_needs_room() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let dir = fixture();
        let state = ModelPickerState::load(&dir, Some("alpha"), "one");
        let draw = |width: u16, height: u16| {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| render(frame, frame.area(), &state)).unwrap();
            let backend = terminal.backend();
            let row: String = if height > 2 {
                (0..width).map(|x| backend.buffer()[(x, 2)].symbol().to_string()).collect()
            } else {
                String::new()
            };
            let cursor = backend.cursor_visible().then(|| backend.cursor_position());
            (row, cursor)
        };
        // Inside the borders: 78 cells give the name 24 and the provider 14,
        // and 60 give them 18 and 10; the id takes what is left.
        let (wide, cursor) = draw(80, 10);
        assert_eq!(wide, format!("│▸ ● {:<24}{:<14}{:<36}│", "Model One", "alpha", "one"));
        assert_eq!(cursor, Some(Position::new(3, 1)));
        let (medium, _) = draw(62, 10);
        assert_eq!(medium, format!("│▸ ● {:<18}{:<10}{:<28}│", "Model One", "alpha", "one"));
        // Two cells wide or no rows high, there is nowhere to put it.
        assert_eq!(draw(4, 10).1, None);
        assert_eq!(draw(20, 2).1, None);
        let _ = fs::remove_dir_all(dir);
    }
}
