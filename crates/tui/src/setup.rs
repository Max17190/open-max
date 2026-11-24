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

pub(crate) type Selection = (String, String);

pub(crate) fn has_terminal() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Ok with the connection and model an API login saved, else Ok(None).
pub async fn run(
    dir: &Path,
    action: Action,
    target: Option<&str>,
    keys: Option<Keys<'_>>,
) -> Result<Option<Selection>, String> {
    if !has_terminal() {
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
        .await
        .map(|()| None);
    }
    if action == Action::Logout {
        let previous = snapshot
            .get(&name)
            .ok_or("provider is not configured")?
            .clone();
        let question = format!(
            "Disconnect {}? Its local connection and catalog will be removed.",
            one_line(&name)
        );
        let choices = [
            ("cancel".into(), "Cancel".into()),
            ("disconnect".into(), "Disconnect".into()),
        ];
        if prompts.pick(&question, &choices).await? != "disconnect" {
            return Err("Cancelled".into());
        }
        providers::edit_catalog(dir, |entries| {
            unchanged(entries.get(&name), Some(&previous))?;
            entries.remove(&name);
            Ok(())
        })
        .await?;
        eprintln!("Disconnected {} locally.", one_line(&name));
        let nonempty = |value: &Value| value.as_str().is_some_and(|s| !s.trim().is_empty());
        let env = &previous["api_key_env"];
        if nonempty(&previous["api_key"])
            || nonempty(env)
            || env.as_array().is_some_and(|names| names.iter().any(nonempty))
            || previous["headers"].as_object().is_some_and(|headers| {
                headers.iter().any(|(name, value)| open_max_core::client::is_credential_header(name) && nonempty(value))
            })
        {
            eprintln!("The credentials remain valid at the provider; revoke them there if needed.");
        }
        return Ok(None);
    }
    let name = if name == "custom" {
        prompts
            .ask_until("Connection name (letters, digits, hyphens or underscores)", false, |name| {
                if matches!(name.as_str(), "chatgpt" | "openai" | "anthropic" | "custom") {
                    Err("choose a connection name other than chatgpt, openai, anthropic, or custom")
                } else if snapshot.contains_key(&name)
                    || (!name.is_empty()
                        && name.len() <= 64
                        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'))
                {
                    Ok(name)
                } else {
                    Err("a connection name is 1 to 64 letters, digits, hyphens or underscores")
                }
            })
            .await?
    } else {
        name
    };
    let previous = snapshot.get(&name).cloned();
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
        Some(base) => base.trim_end_matches('/').to_owned(),
        None => {
            prompts
                .ask_until("API base URL (including /v1 when required)", false, |base| {
                    let base = base.trim_end_matches('/').to_owned();
                    validate_url(&base).map(|_| base)
                })
                .await?
        }
    };
    let local = validate_url(&base)?;
    let billed = if local { "" } else { " API usage is billed by this provider." };
    eprintln!(
        "Connecting {} to {}.{billed}",
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
        None if local => "API key (hidden; leave empty if the server does not require one)".into(),
        None => "API key (hidden)".into(),
    };
    let mut key = prompts.ask_key(&prompt, env, local).await?;
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
    let models = loop {
        eprintln!("Checking access and loading available models...");
        match discover(&base, &key, headers, base == OPENAI, native).await {
            Ok(models) => break models,
            Err(DiscoveryError::Rejected(status)) => {
                eprintln!("The provider rejected access (HTTP {status}). Enter the API key again, or press Esc to cancel.");
                key = prompts.ask_key("API key (hidden)", None, local).await?;
            }
            Err(error) => return Err(error.to_string()),
        }
    };
    let model = if models.is_empty() {
        eprintln!("This endpoint does not expose a model catalog, so access could not be verified. Enter its exact model ID.");
        prompts
            .ask_until("Model ID", false, |id| {
                if id.is_empty() || id.len() > 256 {
                    Err("a model ID is 1 to 256 characters")
                } else {
                    Ok(id)
                }
            })
            .await?
    } else {
        let choices = models
            .iter()
            .filter_map(|m| Some((m["id"].as_str()?.to_owned(), m["id"].as_str()?.to_owned())))
            .collect::<Vec<_>>();
        prompts
            .pick("Choose a model (type to filter)", &choices)
            .await?
    };
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
                .ask_until("Context window in tokens (such as 128000 or 128k)", false, |text| {
                    context_window(&text).ok_or("a context window is a positive number of tokens, such as 128000 or 128k")
                })
                .await?
        }
    };
    api_defaults(&mut entry, &name, previous.is_none())?;
    let inherited = entry.get("wire_api").is_none_or(|w| w == "chat_completions").then(|| open_max_core::config::load(dir).map_or(4096, |s| s.max_tokens) as u64);
    let catalog = entry
        .as_object_mut()
        .unwrap()
        .entry("models")
        .or_insert_with(|| json!([]));
    let catalog = catalog
        .as_array_mut()
        .ok_or("provider models must be an array")?;
    record_model(catalog, &model, window, metadata, inherited);
    entry["base_url"] = json!(base);
    let keyless = key.is_empty();
    if keyless {
        entry.as_object_mut().unwrap().remove("api_key");
    } else {
        entry["api_key"] = json!(if key.starts_with('$') {
            format!("${key}")
        } else {
            key
        });
    }
    entry.as_object_mut().unwrap().remove("api_key_env");
    providers::edit_catalog(dir, |entries| {
        unchanged(entries.get(&name), previous.as_ref())?;
        entries.insert(name.clone(), entry);
        Ok(())
    })
    .await?;
    if keyless {
        eprintln!("Saved connection {} without an API key.", one_line(&name));
    } else {
        eprintln!(
            "Saved connection {}. Credentials are stored locally with owner-only permissions.",
            one_line(&name)
        );
    }
    Ok(Some((name, model)))
}

pub fn has_selection(settings: &open_max_core::config::Settings) -> bool {
    settings.provider.as_deref().is_some_and(|p| !p.trim().is_empty()) || !settings.base_url.trim().is_empty()
}

/// After a login from the shell: when settings select no endpoint, the
/// connection just made becomes the default through the save `/model`
/// makes, so the next `openmax` runs on it; otherwise the selection stays.
pub fn select_unless_chosen(dir: &Path, provider: &str, model: &str) -> Result<(), String> {
    let mut settings = open_max_core::config::load(dir)?;
    if has_selection(&settings) {
        eprintln!(
            "Choose it with /model, or run openmax --provider {} --model {}",
            open_max_core::doctor::shell_quote(Path::new(provider)),
            open_max_core::doctor::shell_quote(Path::new(model))
        );
        return Ok(());
    }
    settings.provider = Some(provider.into());
    settings.model = model.into();
    open_max_core::config::save(dir, &settings)?;
    eprintln!("{} from {} is now the default model; change it with /model.", one_line(model), one_line(provider));
    Ok(())
}

/// A context window as typed: a number of tokens, or thousands or millions
/// of them with a k or m after it (128k is 128,000).
fn context_window(text: &str) -> Option<u64> {
    let text = text.to_ascii_lowercase();
    let (digits, scale) = if let Some(digits) = text.strip_suffix('k') {
        (digits, 1_000)
    } else if let Some(digits) = text.strip_suffix('m') {
        (digits, 1_000_000)
    } else {
        (text.as_str(), 1)
    };
    digits.trim().parse::<u64>().ok()?.checked_mul(scale)
        .filter(|n| *n > 0 && usize::try_from(*n).is_ok())
}

/// Saves the model's window and the output maximum a catalog reports as the reply
/// limit; on Chat Completions only one under `inherited`, the limit an entry without
/// one runs at. It is capped at a quarter of the window (not below the 4096 default)
/// so no request asks for a longer reply. An entry's limit is kept; none reported, none guessed.
fn record_model(catalog: &mut Vec<Value>, model: &str, window: u64, metadata: &Value, inherited: Option<u64>) {
    let most = ["max_completion_tokens", "max_output_tokens", "max_tokens"].iter().find_map(|k| metadata[*k].as_u64())
        .or_else(|| metadata["top_provider"]["max_completion_tokens"].as_u64()).filter(|n| *n > 0 && inherited.is_none_or(|limit| *n < limit));
    let i = catalog.iter().position(|m| m["id"] == model).unwrap_or_else(|| { catalog.push(json!({"id":model})); catalog.len() - 1 });
    catalog[i]["context_tokens"] = json!(window);
    if let (Some(most), None) = (most, catalog[i].get("max_tokens")) { catalog[i]["max_tokens"] = json!(most.min((window / 4).max(4096))); }
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

/// Whether `base` is on this machine, once it is a URL setup may save.
fn validate_url(base: &str) -> Result<bool, String> {
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
    Ok(local)
}

#[derive(Debug)]
enum DiscoveryError {
    Rejected(u16),
    Other(String),
}

impl From<&str> for DiscoveryError {
    fn from(error: &str) -> Self {
        Self::Other(error.into())
    }
}

impl std::fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(status) => write!(f, "model discovery returned HTTP {status}; connection was not saved"),
            Self::Other(error) => f.write_str(error),
        }
    }
}

async fn discover(
    base: &str,
    key: &str,
    headers: &serde_json::Map<String, Value>,
    openai: bool,
    native: bool,
) -> Result<Vec<Value>, DiscoveryError> {
    validate_url(base).map_err(DiscoveryError::Other)?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|_| "cannot initialize model discovery")?;
    let request = client.get(format!("{base}/models"));
    let mut request = match (key.is_empty(), native) {
        (true, _) => request,
        (false, true) => request.header("x-api-key", key),
        (false, false) => request.bearer_auth(key),
    };
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
    if matches!(response.status().as_u16(), 401 | 403) {
        return Err(DiscoveryError::Rejected(response.status().as_u16()));
    }
    if !response.status().is_success() {
        return Err(DiscoveryError::Other(format!(
            "model discovery returned HTTP {}; connection was not saved",
            response.status().as_u16()
        )));
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
    parse_catalog(&bytes, openai).map_err(DiscoveryError::Other)
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
    /// Ask until `check` takes the answer, printing why it refused each
    /// one before asking again; Esc or the end of input still cancels.
    async fn ask_until<T, E: std::fmt::Display>(
        &mut self,
        title: &str,
        secret: bool,
        check: impl Fn(String) -> Result<T, E>,
    ) -> Result<T, String> {
        loop {
            match check(self.ask(title, secret).await?) {
                Ok(answer) => return Ok(answer),
                Err(why) => eprintln!("{why}"),
            }
        }
    }
    async fn ask_key(&mut self, prompt: &str, env: Option<&str>, local: bool) -> Result<String, String> {
        self.ask_until(prompt, true, |typed| {
            let key = if typed.is_empty() {
                env.and_then(|key| std::env::var(key).ok()).unwrap_or_default()
            } else {
                typed
            };
            match reqwest::header::HeaderValue::from_str(&key) {
                Ok(_) if !key.is_empty() || local => Ok(key),
                _ => Err("enter a valid API key; only a server on this machine may take none"),
            }
        }).await
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
