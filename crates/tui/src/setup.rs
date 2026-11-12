//! Human-only provider setup, outside the composer and agent transcript.
use std::io::{IsTerminal, Write};
use std::path::Path;

use crossterm::{
    cursor,
    event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers},
    terminal,
};
use futures_util::StreamExt;
use open_max_core::{providers, text::one_line};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Login,
    Logout,
}

const OPENAI: &str = "https://api.openai.com/v1";
const ANTHROPIC: &str = "https://api.anthropic.com/v1";
const LIMIT: usize = 1024 * 1024;

/// Setup's keys: a session's reader (`app::TerminalInput`), else its own stream.
pub(crate) type Keys<'a> = Box<dyn futures_util::Stream<Item = std::io::Result<Event>> + Unpin + 'a>;

pub async fn run(dir: &Path, action: Action, target: Option<&str>, keys: Option<Keys<'_>>) -> Result<(), String> {
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return Err("provider setup needs an interactive terminal".into());
    }
    let snapshot = providers::read_catalog(dir)?;
    let mut prompts = Prompts { events: keys.unwrap_or_else(|| Box::new(EventStream::new())) };
    let name = match target.filter(|s| !s.is_empty()) {
        Some(name) => name.to_owned(),
        None => {
            let mut choices: Vec<(String, String)> = if action == Action::Login {
                vec![
                    (
                        "chatgpt".into(),
                        "OpenAI: ChatGPT plan (browser sign-in)".into(),
                    ),
                    (
                        "openai".into(),
                        "OpenAI API (API key, separate billing)".into(),
                    ),
                    (
                        "anthropic".into(),
                        "Anthropic API (API key, separate billing)".into(),
                    ),
                    ("custom".into(), "Other compatible API".into()),
                ]
            } else {
                Vec::new()
            };
            for name in snapshot.keys() {
                if !choices.iter().any(|(id, _)| id == name) {
                    choices.push((name.clone(), format!("{name} (configured)")));
                }
            }
            prompts
                .pick(
                    if action == Action::Login {
                        "Connect a provider"
                    } else {
                        "Disconnect a provider"
                    },
                    &choices,
                )
                .await?
        }
    };
    if name == "chatgpt"
        && snapshot
            .get(&name)
            .is_some_and(|p| !p["chatgpt_account"].is_string())
    {
        return Err("the reserved chatgpt alias conflicts with an existing connection; rename that connection in providers.json".into());
    }
    if name == "chatgpt"
        || snapshot
            .get(&name)
            .is_some_and(|p| p["chatgpt_account"].is_string())
    {
        let account = snapshot
            .get(&name)
            .and_then(|entry| entry["chatgpt_account"].as_str())
            .unwrap_or("main");
        return crate::chatgpt::run(
            dir,
            if action == Action::Login {
                "login"
            } else {
                "logout"
            },
            account,
        )
        .await;
    }
    if action == Action::Logout {
        let previous = snapshot
            .get(&name)
            .ok_or("provider is not configured")?
            .clone();
        prompts
            .pick(
                &format!(
                    "Disconnect {}? Its local connection and catalog will be removed.",
                    one_line(&name)
                ),
                &[
                    ("cancel".into(), "Cancel".into()),
                    ("disconnect".into(), "Disconnect".into()),
                ],
            )
            .await
            .and_then(|choice| {
                if choice == "disconnect" {
                    Ok(())
                } else {
                    Err("Cancelled".into())
                }
            })?;
        providers::edit_catalog(dir, |entries| {
            unchanged(entries.get(&name), Some(&previous))?;
            entries.remove(&name);
            Ok(())
        })
        .await?;
        eprintln!("Disconnected {} locally. The API key remains valid at the provider; revoke it there if needed.", one_line(&name));
        return Ok(());
    }
    let custom = name == "custom";
    let name = if custom {
        prompts
            .ask(
                "Connection name (letters, digits, hyphens or underscores)",
                false,
            )
            .await?
    } else {
        name
    };
    if custom && matches!(name.as_str(), "chatgpt" | "openai" | "anthropic" | "custom") {
        return Err(
            "choose a connection name other than chatgpt, openai, anthropic, or custom".into(),
        );
    }
    let previous = snapshot.get(&name).cloned();
    if previous.is_none()
        && (name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
    {
        return Err(
            "connection name must be 1 to 64 letters, digits, hyphens or underscores".into(),
        );
    }
    let preset = match name.as_str() {
        "openai" => Some(OPENAI),
        "anthropic" => Some(ANTHROPIC),
        _ => None,
    };
    if preset.is_none() && previous.is_none() && target.is_some_and(|t| t != "custom") {
        return Err("unknown provider; choose openai, chatgpt, anthropic, custom, or a configured connection".into());
    }
    let mut entry = previous.clone().unwrap_or_else(|| json!({}));
    if !entry.is_object() || entry["chatgpt_account"].is_string() {
        return Err("this connection cannot be replaced with API-key authentication".into());
    }
    if let (Some(base), Some(old)) = (preset, previous.as_ref()) {
        if old["base_url"].as_str().map(|s| s.trim_end_matches('/')) != Some(base) {
            return Err(
                "a different endpoint already uses this connection name; choose a custom name"
                    .into(),
            );
        }
    }
    let base = match preset.or_else(|| entry["base_url"].as_str()) {
        Some(base) => base.to_owned(),
        None => {
            prompts
                .ask("API base URL (including /v1 when required)", false)
                .await?
        }
    };
    let base = base.trim_end_matches('/').to_owned();
    validate_url(&base)?;
    eprintln!(
        "Connecting {} to {}. API usage is billed by this provider.",
        one_line(&name),
        one_line(&base)
    );
    let native = entry["wire_api"] == "anthropic" || (name == "anthropic" && previous.is_none());
    if native {
        eprintln!("This connection uses the native Messages API with model-dependent reasoning effort.");
    } else if name == "anthropic" {
        eprintln!(
            "This connection uses the compatible API, which does not provide prompt caching."
        );
    }
    let env = match name.as_str() {
        "openai" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        _ => None,
    };
    let prompt = match env.filter(|key| std::env::var(key).is_ok_and(|v| !v.trim().is_empty())) {
        Some(key) => format!("API key (hidden; Enter copies {key} into this connection)"),
        None => "API key (hidden)".into(),
    };
    let typed = prompts.ask(&prompt, true).await?;
    let key = if typed.is_empty() {
        env.and_then(|key| std::env::var(key).ok())
            .unwrap_or_default()
    } else {
        typed
    };
    if key.is_empty() || reqwest::header::HeaderValue::from_str(&key).is_err() {
        return Err("a non-empty valid API key is required".into());
    }
    let headers = entry
        .as_object_mut()
        .unwrap()
        .entry("headers")
        .or_insert_with(|| json!({}));
    let headers = headers
        .as_object_mut()
        .ok_or("provider headers must be an object")?;
    if headers
        .keys()
        .any(|k| open_max_core::client::is_credential_header(k))
    {
        return Err(
            "this provider uses credential headers; manage its authentication in providers.json"
                .into(),
        );
    }
    if name == "anthropic" {
        headers.insert("anthropic-version".into(), json!("2023-06-01"));
        let workspace = prompts
            .ask(
                "Workspace ID (leave empty for a workspace-scoped key)",
                false,
            )
            .await?;
        if workspace.is_empty() {
            headers.remove("anthropic-workspace-id");
        } else {
            headers.insert("anthropic-workspace-id".into(), json!(workspace));
        }
    }
    eprintln!("Checking access and loading available models...");
    let models = discover(&base, &key, headers, base == OPENAI, native).await?;
    let model = if models.is_empty() {
        eprintln!("This endpoint does not expose a model catalog, so access could not be verified. Enter its exact model ID.");
        prompts.ask("Model ID", false).await?
    } else {
        let choices = models
            .iter()
            .filter_map(|m| Some((m["id"].as_str()?.to_owned(), m["id"].as_str()?.to_owned())))
            .collect::<Vec<_>>();
        prompts
            .pick("Choose a model (type to filter)", &choices)
            .await?
    };
    if model.is_empty() || model.len() > 256 {
        return Err("a model ID is required".into());
    }
    let metadata = models
        .iter()
        .find(|m| m["id"] == model)
        .unwrap_or(&Value::Null);
    let window = metadata["context_window"]
        .as_u64()
        .or_else(|| metadata["context_length"].as_u64())
        .or_else(|| metadata["max_input_tokens"].as_u64())
        .filter(|n| *n > 0 && usize::try_from(*n).is_ok());
    let window = match window {
        Some(n) => n,
        None => {
            eprintln!("The API did not report this model's context window. Use the value from its documentation.");
            prompts
                .ask("Context window in tokens", false)
                .await?
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or("context window must be a positive number of tokens")? as u64
        }
    };
    let catalog = entry
        .as_object_mut()
        .unwrap()
        .entry("models")
        .or_insert_with(|| json!([]));
    let catalog = catalog
        .as_array_mut()
        .ok_or("provider models must be an array")?;
    record_model(catalog, &model, window);
    entry["base_url"] = json!(base);
    entry["api_key"] = json!(if key.starts_with('$') {
        format!("${key}")
    } else {
        key
    });
    entry.as_object_mut().unwrap().remove("api_key_env");
    api_defaults(&mut entry, &name, previous.is_none())?;
    providers::edit_catalog(dir, |entries| {
        unchanged(entries.get(&name), previous.as_ref())?;
        entries.insert(name.clone(), entry);
        Ok(())
    })
    .await?;
    eprintln!(
        "Saved connection {}. Credentials are stored locally with owner-only permissions.",
        one_line(&name)
    );
    eprintln!(
        "Choose it with /model, or run openmax --provider {} --model {}",
        open_max_core::doctor::shell_quote(Path::new(&name)),
        open_max_core::doctor::shell_quote(Path::new(&model))
    );
    Ok(())
}

/// Saves the model's window. A reply limit is never written: the default stays
/// in force, and a `max_tokens` already on the entry is kept.
fn record_model(catalog: &mut Vec<Value>, model: &str, window: u64) {
    match catalog.iter_mut().find(|m| m["id"] == model) {
        Some(old) => old["context_tokens"] = json!(window),
        None => catalog.push(json!({"id":model,"context_tokens":window})),
    }
}

fn api_defaults(entry: &mut Value, name: &str, new: bool) -> Result<(), String> {
    if !new { return Ok(()); }
    if matches!(name, "openai" | "anthropic") {
        let compat = entry
            .as_object_mut()
            .unwrap()
            .entry("compat")
            .or_insert_with(|| json!({}));
        compat
            .as_object_mut()
            .ok_or("provider compat must be an object")?
            .insert("omit_temperature".into(), json!(true));
    }
    if entry.get("wire_api").is_none() {
        entry["wire_api"] = json!(if name == "openai" {
            "responses"
        } else if name == "anthropic" {
            "anthropic"
        } else {
            "chat_completions"
        });
    }
    Ok(())
}

fn unchanged(current: Option<&Value>, previous: Option<&Value>) -> Result<(), String> {
    if current == previous {
        Ok(())
    } else {
        Err("connection changed during setup; retry without overwriting the other change".into())
    }
}

fn validate_url(base: &str) -> Result<(), String> {
    let url = reqwest::Url::parse(base).map_err(|_| "invalid API base URL")?;
    let local = url.host_str().is_some_and(|h| {
        h == "localhost"
            || h.trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || (url.scheme() == "http" && !local)
    {
        return Err("use an HTTPS API URL without credentials, query, or fragment; plain HTTP is allowed only on loopback".into());
    }
    Ok(())
}

async fn discover(
    base: &str,
    key: &str,
    headers: &serde_json::Map<String, Value>,
    openai: bool,
    native: bool,
) -> Result<Vec<Value>, String> {
    validate_url(base)?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|_| "cannot initialize model discovery")?;
    let request = client.get(format!("{base}/models"));
    let mut request = if native { request.header("x-api-key", key) } else { request.bearer_auth(key) };
    if native && !headers.keys().any(|name| name.eq_ignore_ascii_case("anthropic-version")) {
        request = request.header("anthropic-version", "2023-06-01");
    }
    if base == ANTHROPIC {
        request = request.query(&[("limit", "1000")]);
    }
    let raw: std::collections::BTreeMap<_, _> = headers.iter().map(|(name, value)| {
        if open_max_core::client::is_credential_header(name) {
            return Err("provider defines additional credentials; manage its authentication in providers.json");
        }
        Ok((name.clone(), value.as_str().ok_or("provider headers must contain strings")?.to_owned()))
    }).collect::<Result<_, _>>()?;
    for (name, value) in providers::expand_headers(&raw) {
        request = request.header(name, value);
    }
    let mut response = request
        .send()
        .await
        .map_err(|_| "could not reach the provider; connection was not saved")?;
    if matches!(response.status().as_u16(), 404 | 405) {
        return Ok(Vec::new());
    }
    if !response.status().is_success() {
        return Err(format!(
            "model discovery returned HTTP {}; connection was not saved",
            response.status().as_u16()
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "model catalog read failed")?
    {
        if bytes.len().saturating_add(chunk.len()) > LIMIT {
            return Err("model catalog exceeds its size limit".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    parse_catalog(&bytes, openai)
}

fn parse_catalog(bytes: &[u8], openai: bool) -> Result<Vec<Value>, String> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|_| "provider returned an invalid model catalog")?;
    let models = value["data"]
        .as_array()
        .ok_or("provider returned no model catalog")?;
    let models: Vec<_> = models
        .iter()
        .filter(|m| {
            m["id"].as_str().is_some_and(|id| {
                !id.is_empty() && id.len() <= 256 && !id.chars().any(char::is_control)
                    && (!openai || openai_response_model(id))
            })
        })
        .cloned()
        .collect();
    if models.is_empty() {
        return Err("provider returned no available models; connection was not saved".into());
    }
    Ok(models)
}

fn openai_response_model(id: &str) -> bool {
    let base = id.strip_prefix("ft:").unwrap_or(id).split(':').next().unwrap_or(id);
    ![
        "text-embedding", "omni-moderation", "text-moderation", "tts", "dall-e",
        "gpt-image", "gpt-realtime", "gpt-live", "gpt-transcribe",
        "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-mini-tts", "whisper",
    ].iter().any(|family| {
        base == *family || base.strip_prefix(family).is_some_and(|suffix| suffix.starts_with('-'))
    })
}

#[cfg(test)]
#[path = "setup_tests.rs"]
mod tests;

static INPUT_ACTIVE: std::sync::Mutex<bool> = std::sync::Mutex::new(false);

/// Keep the guard until forced process exit so another prompt cannot enter behind it.
pub(crate) fn restore_input() -> std::sync::MutexGuard<'static, bool> {
    let mut active = INPUT_ACTIVE.lock().unwrap_or_else(|error| error.into_inner());
    if *active {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(std::io::stderr(), crossterm::event::DisableBracketedPaste);
        *active = false;
    }
    active
}

struct InputMode;
impl InputMode {
    fn enter() -> Result<Self, String> {
        let prepared = {
            let mut active = INPUT_ACTIVE.lock().unwrap_or_else(|error| error.into_inner());
            terminal::enable_raw_mode().map_err(|_| "cannot read terminal input")?;
            *active = true;
            crossterm::execute!(std::io::stderr(), crossterm::event::EnableBracketedPaste)
                .map_err(|_| "cannot prepare terminal input")
        };
        if let Err(error) = prepared {
            drop(restore_input());
            return Err(error.into());
        }
        Ok(Self)
    }
}
impl Drop for InputMode {
    fn drop(&mut self) {
        drop(restore_input());
    }
}

struct Prompts<'a> {
    events: Keys<'a>,
}
impl Prompts<'_> {
    async fn ask(&mut self, title: &str, secret: bool) -> Result<String, String> {
        self.input(title, secret, None).await
    }
    async fn pick(&mut self, title: &str, choices: &[(String, String)]) -> Result<String, String> {
        if choices.is_empty() {
            return Err("no configured providers; use /login or openmax --login first".into());
        }
        self.input(title, false, Some(choices)).await
    }

    async fn input(
        &mut self,
        title: &str,
        secret: bool,
        choices: Option<&[(String, String)]>,
    ) -> Result<String, String> {
        let _mode = InputMode::enter()?;
        eprint!("{}\r\n", one_line(title));
        let (mut text, mut selected, mut rows) = (String::new(), 0usize, 0u16);
        loop {
            let filtered: Vec<_> = choices
                .into_iter()
                .flatten()
                .filter(|(id, label)| {
                    format!("{id} {label}")
                        .to_lowercase()
                        .contains(&text.to_lowercase())
                })
                .collect();
            selected = selected.min(filtered.len().saturating_sub(1));
            let mut out = std::io::stderr();
            if rows > 0 {
                crossterm::execute!(out, cursor::MoveUp(rows))
                    .map_err(|_| "terminal write failed")?;
            }
            crossterm::execute!(
                out,
                cursor::MoveToColumn(0),
                terminal::Clear(terminal::ClearType::FromCursorDown)
            )
            .map_err(|_| "terminal write failed")?;
            let width = terminal::size()
                .map(|(w, _)| w as usize)
                .unwrap_or(80)
                .saturating_sub(5);
            let shown = if secret {
                "*".repeat(text.chars().count().min(width))
            } else {
                one_line(&text).chars().take(width).collect()
            };
            write!(out, "> {shown}\r\n").map_err(|_| "terminal write failed")?;
            rows = 1;
            if choices.is_some() {
                let first = selected.saturating_sub(6);
                for (index, (_, label)) in filtered.iter().enumerate().skip(first).take(7) {
                    writeln!(
                        out,
                        "{} {}\r",
                        if index == selected { ">" } else { " " },
                        one_line(label).chars().take(width).collect::<String>()
                    )
                    .map_err(|_| "terminal write failed")?;
                    rows += 1;
                }
            }
            write!(out, "Enter confirms; Esc cancels\r\n").map_err(|_| "terminal write failed")?;
            rows += 1;
            out.flush().map_err(|_| "terminal write failed")?;
            let event = self
                .events
                .next()
                .await
                .ok_or("terminal input closed")?
                .map_err(|_| "terminal input failed")?;
            match event {
                Event::Key(key) if key.kind != KeyEventKind::Release => match key.code {
                    KeyCode::Esc => {
                        return Err("Cancelled".into());
                    }
                    KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Err("Cancelled".into());
                    }
                    KeyCode::Enter => {
                        if choices.is_none() {
                            return Ok(text.trim().to_owned());
                        }
                        if let Some((id, _)) = filtered.get(selected) {
                            return Ok(id.to_string());
                        }
                    }
                    KeyCode::Up => selected = selected.saturating_sub(1),
                    KeyCode::Down => {
                        selected = (selected + 1).min(filtered.len().saturating_sub(1))
                    }
                    KeyCode::Backspace => {
                        text.pop();
                        selected = 0;
                    }
                    KeyCode::Char(c)
                        if !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                            && !c.is_control()
                            && text.len() < 4096 =>
                    {
                        text.push(c);
                        selected = 0;
                    }
                    _ => {}
                },
                Event::Paste(value)
                    if text.len() + value.trim().len() <= 4096
                        && !value.trim().chars().any(char::is_control) =>
                {
                    text.push_str(value.trim());
                    selected = 0;
                }
                _ => {}
            }
        }
    }
}
