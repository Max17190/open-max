//! Inline completion for the composer: slash commands and @-file mentions.
//! A popup opens while the token under the cursor looks completable; the
//! composer keeps owning the text, this module only proposes replacements.

use std::path::Path;
use std::sync::Arc;

use ignore::WalkBuilder;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;

/// Popup height cap: enough choices without hiding the conversation.
pub const MAX_VISIBLE: usize = 6;
/// File-index cap. Beyond this the popup still works on what was scanned;
/// gitignore pruning keeps real projects far below it.
const MAX_FILES: usize = 20_000;

/// One command registry drives dispatch recognition, completion, and `/help`.
/// Dispatch itself stays explicit in `App::slash`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub args: &'static str,
    pub description: &'static str,
    /// Accepting the completion immediately runs the command.
    pub submits: bool,
}

const fn command(
    name: &'static str,
    args: &'static str,
    description: &'static str,
    submits: bool,
) -> CommandSpec {
    CommandSpec {
        name,
        args,
        description,
        submits,
    }
}

pub const COMMANDS: &[CommandSpec] = &[
    command("help", "", "keybindings and commands", true),
    command("models", "", "manage and serve local models", true),
    command(
        "model",
        "[id]",
        "pick a configured model or use an exact id",
        true,
    ),
    command(
        "provider",
        "[name]",
        "list or select a named provider",
        false,
    ),
    command(
        "approvals",
        "auto|ask|readonly",
        "how mutating tools are gated",
        false,
    ),
    command("new", "", "start a fresh session", true),
    command(
        "resume",
        "",
        "pick an earlier session in this project",
        true,
    ),
    command("copy", "", "copy the latest assistant response", true),
    command("tools", "", "list tools frozen for this session", true),
    command("skills", "", "list skills frozen for this session", true),
    command(
        "reload",
        "",
        "re-freeze tools, skills, and prompt from current config",
        true,
    ),
    command(
        "context",
        "",
        "prompt token costs, cache hits, and budget",
        true,
    ),
    command(
        "status",
        "",
        "endpoint, cache, performance, privacy, and network details",
        true,
    ),
    command("logs", "", "recent model server logs", true),
    command(
        "theme",
        "dark|light|mono|catppuccin",
        "switch appearance",
        false,
    ),
    command("quit", "", "exit", true),
];

impl CommandSpec {
    pub fn usage(self) -> String {
        if self.args.is_empty() {
            format!("/{}", self.name)
        } else {
            format!("/{} {}", self.name, self.args)
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Kind {
    Slash,
    File,
}

#[derive(Clone)]
pub struct Item {
    /// Text that replaces the token (including its `/` or `@` sigil).
    pub insert: String,
    pub label: String,
    pub detail: String,
    /// Slash commands that take no argument submit on accept.
    pub submits: bool,
}

pub struct Popup {
    pub kind: Kind,
    pub items: Vec<Item>,
    pub selected: usize,
    /// Char index in the composer row where the token (sigil included) starts.
    pub token_start: usize,
    /// Char length of the token being replaced.
    pub token_len: usize,
}

impl Popup {
    pub fn next(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1) % self.items.len();
        }
    }

    pub fn prev(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + self.items.len() - 1) % self.items.len();
        }
    }

    pub fn selected_item(&self) -> Option<&Item> {
        self.items.get(self.selected)
    }
}

/// The token under the cursor, if it can drive a completion. Slash commands
/// complete only as the first token of the message; @-files complete anywhere.
pub fn trigger(line: &str, col: usize, first_row: bool) -> Option<(Kind, usize, String)> {
    let chars: Vec<char> = line.chars().collect();
    let col = col.min(chars.len());
    let mut start = col;
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let token: String = chars[start..col].iter().collect();
    if first_row && start == 0 {
        if let Some(query) = token.strip_prefix('/') {
            // Past the command name (a space would end the token) argument
            // hints take over; no completion inside arguments. `-`/`_` cover
            // prompt-template names like fix-issue.
            if query
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Some((Kind::Slash, start, query.to_string()));
            }
        }
    }
    if let Some(query) = token.strip_prefix('@') {
        return Some((Kind::File, start, query.to_string()));
    }
    None
}

/// Filtered slash-command items for `query` (text after the `/`): built-in
/// commands first, then prompt templates (name, description). A template that
/// shadows a built-in name is dropped; built-ins always win at dispatch too.
pub fn slash_items(query: &str, templates: &[(String, String)]) -> Vec<Item> {
    let mut items: Vec<Item> = COMMANDS
        .iter()
        .filter(|spec| spec.name.starts_with(query))
        .map(|spec| Item {
            insert: if spec.submits {
                format!("/{}", spec.name)
            } else {
                format!("/{} ", spec.name)
            },
            label: format!("/{}", spec.name),
            detail: if spec.args.is_empty() {
                spec.description.to_string()
            } else {
                format!("{} · {}", spec.args, spec.description)
            },
            submits: spec.submits,
        })
        .collect();
    for (name, desc) in templates {
        if !name.starts_with(query) || COMMANDS.iter().any(|spec| spec.name == name) {
            continue;
        }
        items.push(Item {
            // Templates may take arguments, so accepting never auto-submits.
            insert: format!("/{name} "),
            label: format!("/{name}"),
            detail: if desc.is_empty() {
                "prompt template".to_string()
            } else {
                format!("{desc} · template")
            },
            submits: false,
        });
    }
    items
}

/// Fuzzy-filtered file items for `query` (text after the `@`).
pub fn file_items(files: &Arc<Vec<String>>, query: &str) -> Vec<Item> {
    let mut scored: Vec<(i32, &String)> = files
        .iter()
        .filter_map(|path| fuzzy_score(path, query).map(|s| (s, path)))
        .collect();
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.len().cmp(&b.1.len()))
            .then_with(|| a.1.cmp(b.1))
    });
    scored
        .into_iter()
        .take(MAX_VISIBLE * 3)
        .map(|(_, path)| Item {
            insert: format!("@{path} "),
            label: path.clone(),
            detail: String::new(),
            submits: false,
        })
        .collect()
}

/// Case-insensitive subsequence match. Higher is better: filename hits beat
/// directory hits, consecutive runs and segment starts beat scattered chars,
/// shorter paths win ties (via the sort above).
pub fn fuzzy_score(path: &str, query: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let hay: Vec<char> = path.chars().map(|c| c.to_ascii_lowercase()).collect();
    let needle: Vec<char> = query.chars().map(|c| c.to_ascii_lowercase()).collect();
    let name_start = path
        .rfind('/')
        .map(|i| path[..=i].chars().count())
        .unwrap_or(0);

    let mut score = 0i32;
    let mut hi = 0usize;
    let mut prev_hit: Option<usize> = None;
    for &nc in &needle {
        let mut found = None;
        while hi < hay.len() {
            if hay[hi] == nc {
                found = Some(hi);
                break;
            }
            hi += 1;
        }
        let at = found?;
        score += 1;
        if at >= name_start {
            score += 8;
        }
        if prev_hit == Some(at.wrapping_sub(1)) {
            score += 6;
        }
        if at == 0
            || matches!(
                hay.get(at.wrapping_sub(1)),
                Some('/') | Some('_') | Some('-') | Some('.')
            )
        {
            score += 4;
        }
        prev_hit = Some(at);
        hi = at + 1;
    }
    Some(score)
}

/// Project files, gitignore-aware, relative paths with `/` separators,
/// shallowest-first so the popup's empty-query view starts at the root.
pub fn scan_files(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let walker = WalkBuilder::new(root)
        .hidden(true)
        .follow_links(false)
        .build();
    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        if let Ok(rel) = entry.path().strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
            if out.len() >= MAX_FILES {
                break;
            }
        }
    }
    out.sort_by_key(|p| (p.matches('/').count(), p.clone()));
    out
}

/// Render the popup as full-width rows, selection marked and windowed.
pub fn render_lines(popup: &Popup, width: u16, indexing: bool) -> Vec<Line<'static>> {
    let width = width as usize;
    if indexing {
        return vec![Line::from(Span::styled(
            "  indexing files…",
            Style::default()
                .fg(theme::DIM())
                .add_modifier(Modifier::ITALIC),
        ))];
    }
    if popup.items.is_empty() {
        return vec![Line::from(Span::styled(
            "  no matches",
            Style::default()
                .fg(theme::DIM())
                .add_modifier(Modifier::ITALIC),
        ))];
    }
    let visible = popup.items.len().min(MAX_VISIBLE);
    // Window keeps the selection in view, pinned to the edges at the ends.
    let first = popup
        .selected
        .saturating_sub(visible - 1)
        .min(popup.items.len() - visible);
    let mut lines = Vec::with_capacity(visible);
    for (i, item) in popup.items.iter().enumerate().skip(first).take(visible) {
        let selected = i == popup.selected;
        let marker = if selected {
            Span::styled("▸ ", Style::default().fg(theme::ACCENT()))
        } else {
            Span::raw("  ")
        };
        let label_style = if selected {
            Style::default()
                .fg(theme::ACCENT())
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let mut spans = vec![
            marker,
            Span::styled(clip(&item.label, width.saturating_sub(4)), label_style),
        ];
        if !item.detail.is_empty() {
            let room = width.saturating_sub(item.label.chars().count() + 6);
            if room > 4 {
                spans.push(Span::styled(
                    format!("  {}", clip(&item.detail, room)),
                    Style::default().fg(theme::DIM()),
                ));
            }
        }
        let mut line = Line::from(spans);
        if selected {
            line.style = Style::default().bg(theme::SURFACE());
            for span in &mut line.spans {
                span.style = span.style.bg(theme::SURFACE());
            }
            let used = line.width();
            if used < width {
                line.spans.push(Span::styled(
                    " ".repeat(width - used),
                    Style::default().bg(theme::SURFACE()),
                ));
            }
        }
        lines.push(line);
    }
    if popup.items.len() > visible {
        lines.push(Line::from(Span::styled(
            format!("  … {} more (keep typing)", popup.items.len() - visible),
            Style::default()
                .fg(theme::DIM())
                .add_modifier(Modifier::ITALIC),
        )));
    }
    lines
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max.max(4) {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max.max(4) - 1).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_triggers_only_at_message_start() {
        assert!(matches!(trigger("/mo", 3, true), Some((Kind::Slash, 0, q)) if q == "mo"));
        assert!(trigger("/mo", 3, false).is_none());
        assert!(trigger("say /mo", 7, true).is_none());
        // Inside an argument the popup stays closed.
        assert!(trigger("/model foo", 10, true).is_none());
    }

    #[test]
    fn at_triggers_anywhere() {
        let got = trigger("look at @src/ma", 15, true);
        assert!(matches!(got, Some((Kind::File, 8, q)) if q == "src/ma"));
        assert!(matches!(trigger("@", 1, false), Some((Kind::File, 0, q)) if q.is_empty()));
        // A mid-word @ (an email address) never opens the popup.
        assert!(trigger("email me a@b", 12, true).is_none());
    }

    #[test]
    fn slash_items_filter_by_prefix() {
        let items = slash_items("mo", &[]);
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, vec!["/models", "/model"]);
        // Bare /model is a real action that opens the picker. A user can type
        // a space after it to use the raw-id escape hatch.
        assert!(items[0].submits);
        assert_eq!(items[1].insert, "/model");
        assert!(items[1].submits);
    }

    #[test]
    fn slash_items_append_templates_but_never_shadow_builtins() {
        let templates = vec![
            ("fix-issue".to_string(), "fix a GitHub issue".to_string()),
            ("new".to_string(), "shadowed by the builtin".to_string()),
        ];
        let items = slash_items("", &templates);
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"/fix-issue"));
        assert_eq!(labels.iter().filter(|l| **l == "/new").count(), 1);
        let tmpl = items.iter().find(|i| i.label == "/fix-issue").unwrap();
        assert_eq!(tmpl.insert, "/fix-issue ");
        assert!(!tmpl.submits);
        assert!(tmpl.detail.contains("template"));
        // Prefix filtering applies to templates too.
        assert!(slash_items("fix", &templates)
            .iter()
            .any(|i| i.label == "/fix-issue"));
        assert!(!slash_items("zz", &templates)
            .iter()
            .any(|i| i.label == "/fix-issue"));
    }

    #[test]
    fn fuzzy_prefers_filename_and_runs() {
        let files = Arc::new(vec![
            "src/app.rs".to_string(),
            "crates/tui/src/main.rs".to_string(),
            "assets/apple.png".to_string(),
        ]);
        let items = file_items(&files, "app");
        assert_eq!(items[0].label, "src/app.rs");
        assert_eq!(items[0].insert, "@src/app.rs ");
    }

    #[test]
    fn fuzzy_rejects_non_subsequences() {
        assert!(fuzzy_score("src/app.rs", "zzz").is_none());
        assert!(fuzzy_score("src/app.rs", "sar").is_some());
    }

    #[test]
    fn popup_selection_wraps() {
        let mut p = Popup {
            kind: Kind::Slash,
            items: slash_items("", &[]),
            selected: 0,
            token_start: 0,
            token_len: 1,
        };
        p.prev();
        assert_eq!(p.selected, p.items.len() - 1);
        p.next();
        assert_eq!(p.selected, 0);
    }

    #[test]
    fn render_windows_around_selection() {
        let mut p = Popup {
            kind: Kind::Slash,
            items: slash_items("", &[]),
            selected: 0,
            token_start: 0,
            token_len: 1,
        };
        p.selected = p.items.len() - 1;
        let lines = render_lines(&p, 80, false);
        // Cap plus the "more" hint at most.
        assert!(lines.len() <= MAX_VISIBLE + 1);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(text.contains("/quit"));
        assert!(lines
            .iter()
            .any(|line| line.style.bg == Some(theme::SURFACE())));
    }

    #[test]
    fn command_registry_is_unique_and_drives_copy_and_model_metadata() {
        let mut names: Vec<_> = COMMANDS.iter().map(|spec| spec.name).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), before);
        let model = COMMANDS.iter().find(|spec| spec.name == "model").unwrap();
        assert_eq!(model.usage(), "/model [id]");
        assert!(model.submits);
        assert!(COMMANDS.iter().any(|spec| spec.name == "copy"));
    }
}
