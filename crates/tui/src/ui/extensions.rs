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
    super::text::clip(&clean, max)
}

/// Split an OpenAI-compatible base URL into (normalized scheme, authority, path).
/// Scheme matching is case-insensitive (`HTTPS://` works). Userinfo is not
/// stripped here; callers that render must drop it from `authority`.
fn split_base_url(base_url: &str) -> Option<(&str, &str, &str)> {
    let s = base_url.trim();
    if s.is_empty() {
        return None;
    }
    let (scheme, rest) = if let Some(i) = s.find("://") {
        let scheme_raw = &s[..i];
        if scheme_raw.eq_ignore_ascii_case("https") {
            ("https://", &s[i + 3..])
        } else if scheme_raw.eq_ignore_ascii_case("http") {
            ("http://", &s[i + 3..])
        } else {
            // Unknown scheme: treat whole string as opaque authority/path.
            ("", s)
        }
    } else {
        ("", s)
    };
    if rest.is_empty() {
        return None;
    }
    let (authority, path) = match rest.split_once('/') {
        Some((a, _)) => (a, &rest[a.len()..]), // keeps leading '/'
        None => (rest, ""),
    };
    Some((scheme, authority, path))
}

/// Drop `user:pass@` from an authority so chrome never renders secrets.
fn strip_userinfo(authority: &str) -> &str {
    authority
        .rsplit_once('@')
        .map(|(_, host)| host)
        .unwrap_or(authority)
}

/// Authority host[:port] from a base URL, with userinfo stripped so credentials
/// never appear in the header or status chrome.
pub fn endpoint_host(base_url: &str) -> Option<String> {
    let (_, authority, _) = split_base_url(base_url)?;
    let authority = authority.trim();
    if authority.is_empty() || authority.contains(' ') {
        return None;
    }
    let hostport = strip_userinfo(authority);
    if hostport.is_empty() {
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

/// Full base URL with userinfo redacted for `/status` and similar listings.
pub fn display_base_url(base_url: &str) -> String {
    let Some((scheme, authority, path)) = split_base_url(base_url) else {
        return String::new();
    };
    let hostport = strip_userinfo(authority);
    format!("{scheme}{hostport}{path}")
}

/// Short model id for header/status (last path segment of a HF-style id).
pub fn short_model(model: &str) -> &str {
    model.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(model)
}

pub fn tools_block(registry: &Registry, frozen: bool) -> Vec<Line<'static>> {
    let mut lines = vec![aside(if frozen {
        "tools (loaded now; the frozen tools array takes a new one at /reload)"
    } else {
        "tools (preview of the next new session)"
    })];
    for spec in &registry.tools {
        let kind = match &spec.kind {
            ToolKind::Builtin => "built-in",
            ToolKind::External(_) => "external",
        };
        let flags = if spec.mutating { format!("{kind} · mut") } else { kind.to_string() };
        lines.push(entry(&spec.name, 14, &flags, 16, &spec.description, 60));
    }
    let external_n = registry.tools.iter().filter(|t| !matches!(t.kind, ToolKind::Builtin)).count();
    lines.push(aside(format!(
        "  {} tools · {external_n} external · drop TOML in .openmax/tools/ or ~/.openmax/tools/",
        registry.tools.len(),
    )));
    lines.push(aside("  token cost: /context"));
    lines
}

pub fn skills_block(skills: &[SkillSpec], project_root: &Path, frozen: bool) -> Vec<Line<'static>> {
    let mut lines = vec![aside(if frozen {
        "skills (loaded now; the frozen prompt takes a new one at /reload)"
    } else {
        "skills (preview of the next new session)"
    })];
    if skills.is_empty() {
        lines.push(aside("  (none loaded)"));
    }
    let project_skills = project_root.join(".agents").join("skills");
    for skill in skills {
        let scope = if skill.path.starts_with(&project_skills) { "project" } else { "global" };
        lines.push(entry(&skill.name, 16, scope, 9, &skill.description, 56));
    }
    lines.push(aside(format!(
        "  {} skills · SKILL.md under .agents/skills/ or ~/.openmax/skills/",
        skills.len()
    )));
    lines.push(aside("  token cost: /context"));
    lines
}

/// A listing's dim italic line: its header, an empty state, its footer.
fn aside(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(text.into(), Style::default().fg(theme::DIM()).add_modifier(Modifier::ITALIC)))
}

/// A listing's row: the name padded to `name_w`, a tag padded to `tag_w`,
/// and the description clipped to `max`.
fn entry(name: &str, name_w: usize, tag: &str, tag_w: usize, description: &str, max: usize) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {}", super::text::pad_right(name, name_w)),
            Style::default().fg(theme::ACCENT()).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("{tag:<tag_w$}"), Style::default().fg(theme::DIM())),
        Span::styled(clip(description, max), Style::default().fg(theme::DIM())),
    ])
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
        assert_eq!(
            endpoint_host("https://user:secret@api.example.com:8443/v1").as_deref(),
            Some("api.example.com:8443")
        );
        assert_eq!(endpoint_host("not a url"), None);
        assert_eq!(endpoint_host(""), None);
    }

    #[test]
    fn display_base_url_strips_userinfo() {
        assert_eq!(
            display_base_url("https://user:secret@api.example.com/v1"),
            "https://api.example.com/v1"
        );
        assert_eq!(
            display_base_url("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1"
        );
        // Schemes are case-insensitive; mixed case must still redact.
        assert_eq!(
            display_base_url("HTTPS://alice:secret@api.example.com/v1"),
            "https://api.example.com/v1"
        );
        assert_eq!(
            endpoint_host("HtTpS://alice:secret@api.example.com:8443/v1").as_deref(),
            Some("api.example.com:8443")
        );
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
        // Only external tools count as external.
        assert!(text.contains(&format!("{} tools · 0 external", reg.tools.len())), "{text}");
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
