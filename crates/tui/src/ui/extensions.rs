//! Read-only listings of the session-frozen (or next-session preview)
//! tool registry and skills. Surfaces what the harness already loaded
//! without changing freeze semantics.

use std::path::Path;

use open_max_core::registry::{Registry, ToolKind};
use open_max_core::skills::SkillSpec;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme;

fn clip(s: &str, max: usize) -> String {
    let clean = s.replace(['\n', '\r'], " ");
    if clean.chars().count() <= max {
        clean
    } else {
        format!("{}…", clean.chars().take(max).collect::<String>())
    }
}

/// Host portion of an OpenAI-compatible base URL for compact chrome.
/// Returns `None` when the URL has no usable host.
pub fn endpoint_host(base_url: &str) -> Option<String> {
    let s = base_url.trim();
    if s.is_empty() {
        return None;
    }
    let rest = s
        .strip_prefix("https://")
        .or_else(|| s.strip_prefix("http://"))
        .unwrap_or(s);
    let hostport = rest.split('/').next().unwrap_or("").trim();
    if hostport.is_empty() || hostport.contains(' ') {
        return None;
    }
    // Prefer host without default ports noise; keep non-default ports.
    let host = match hostport.rsplit_once(':') {
        Some((h, port)) if port.chars().all(|c| c.is_ascii_digit()) => {
            if port == "80" || port == "443" {
                h
            } else {
                hostport
            }
        }
        _ => hostport,
    };
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

/// Short model id for header/status (last path segment of a HF-style id).
pub fn short_model(model: &str) -> &str {
    model.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(model)
}

pub fn tools_block(registry: &Registry, frozen: bool) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let header = if frozen {
        "tools (frozen at session creation; config changes apply to /new)"
    } else {
        "tools (preview of the next new session)"
    };
    lines.push(Line::from(Span::styled(
        header.to_string(),
        Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
    )));

    for spec in &registry.tools {
        let kind = match &spec.kind {
            ToolKind::Builtin => "built-in",
            ToolKind::External(_) => "external",
        };
        let mut flags = kind.to_string();
        if spec.mutating {
            flags.push_str(" · mut");
        }
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<14}", clip(&spec.name, 14)),
                Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{flags:<16}"),
                Style::default().fg(theme::DIM()),
            ),
            Span::styled(
                clip(&spec.description, 60),
                Style::default().fg(theme::DIM()),
            ),
        ]));
    }

    let external_n = registry
        .tools
        .iter()
        .filter(|t| !matches!(t.kind, ToolKind::Builtin))
        .count();
    lines.push(Line::from(Span::styled(
        format!(
            "  {} tools · {} external · drop TOML in .openmax/tools/ or ~/.openmax/tools/",
            registry.tools.len(),
            external_n
        ),
        Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
    )));
    lines.push(Line::from(Span::styled(
        "  token cost: /context".to_string(),
        Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
    )));
    lines
}

pub fn skills_block(skills: &[SkillSpec], project_root: &Path, frozen: bool) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let header = if frozen {
        "skills (frozen at session creation; config changes apply to /new)"
    } else {
        "skills (preview of the next new session)"
    };
    lines.push(Line::from(Span::styled(
        header.to_string(),
        Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
    )));

    if skills.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (none loaded)".to_string(),
            Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
        )));
    } else {
        let project_skills = project_root.join(".agents").join("skills");
        for skill in skills {
            let scope = if skill.path.starts_with(&project_skills) {
                "project"
            } else {
                "global"
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<16}", clip(&skill.name, 16)),
                    Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{scope:<9}"),
                    Style::default().fg(theme::DIM()),
                ),
                Span::styled(
                    clip(&skill.description, 56),
                    Style::default().fg(theme::DIM()),
                ),
            ]));
        }
    }

    lines.push(Line::from(Span::styled(
        format!(
            "  {} skills · SKILL.md under .agents/skills/ or ~/.openmax/skills/",
            skills.len()
        ),
        Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
    )));
    lines.push(Line::from(Span::styled(
        "  token cost: /context".to_string(),
        Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC),
    )));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use open_max_core::registry::Registry;

    #[test]
    fn endpoint_host_parses_common_urls() {
        assert_eq!(
            endpoint_host("http://127.0.0.1:11434/v1").as_deref(),
            Some("127.0.0.1:11434")
        );
        assert_eq!(
            endpoint_host("https://api.example.com/v1").as_deref(),
            Some("api.example.com")
        );
        assert_eq!(
            endpoint_host("https://api.example.com:443/v1/chat").as_deref(),
            Some("api.example.com")
        );
        assert_eq!(endpoint_host("not a url"), None);
        assert_eq!(endpoint_host(""), None);
    }

    #[test]
    fn short_model_takes_last_segment() {
        assert_eq!(short_model("Qwen/Qwen2.5-Coder-7B"), "Qwen2.5-Coder-7B");
        assert_eq!(short_model("gpt-oss-20b"), "gpt-oss-20b");
    }

    #[test]
    fn tools_block_lists_builtins() {
        let reg = Registry::builtin_only();
        let lines = tools_block(&reg, false);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("built-in") || text.contains("read_file") || text.contains("bash"));
        assert!(text.contains("preview") || text.contains("tools"));
    }

    #[test]
    fn skills_block_empty_is_quiet() {
        let lines = skills_block(&[], Path::new("/tmp/proj"), true);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("");
        assert!(text.contains("none loaded"));
        assert!(text.contains("frozen"));
    }
}
