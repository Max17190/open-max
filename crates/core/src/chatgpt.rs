//! Local ChatGPT credentials. Only an explicitly selected provider reads them.
//! OAuth refuses redirects; inference keeps credentials on its configured origin.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use futures_util::StreamExt;
use ring::{
    digest,
    rand::{SecureRandom, SystemRandom},
    signature,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const API: &str = "https://api.openai.com/v1";
const ISSUER: &str = "https://auth.openai.com";
const AUTHORIZE: &str = "https://auth.openai.com/api/accounts/authorize";
const TOKEN: &str = "https://auth.openai.com/api/accounts/oauth/token";
const DISCOVERY: &str = "https://auth.openai.com/.well-known/openid-configuration";
const SCOPES: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
const DIRECT: &str = "chatgpt.tokens.use.direct";
const LIMIT: usize = 1024 * 1024;

static STOPPING: AtomicBool = AtomicBool::new(false);
static PENDING: AtomicUsize = AtomicUsize::new(0);
static SETTLED: tokio::sync::Notify = tokio::sync::Notify::const_new();
struct Pending;
impl Drop for Pending {
    fn drop(&mut self) {
        PENDING.fetch_sub(1, Ordering::SeqCst);
        SETTLED.notify_waiters();
    }
}

/// Final process shutdown: prevent new refreshes and persist any rotation in flight.
/// With no ChatGPT request this is only an atomic check, with no file or network work.
pub async fn finish_refreshes() {
    STOPPING.store(true, Ordering::SeqCst);
    while PENDING.load(Ordering::SeqCst) != 0 {
        let settled = SETTLED.notified();
        tokio::pin!(settled);
        settled.as_mut().enable();
        if PENDING.load(Ordering::SeqCst) == 0 { return; }
        settled.await;
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Store {
    host: String,
    accounts: BTreeMap<String, Account>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct Account {
    client_id: String,
    subject: String,
    email: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tokens: Option<Tokens>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct Tokens {
    access: String,
    refresh: String,
    id: String,
    scopes: String,
    expires: u64,
}

/// Contains no token. Identity is stable across renewal, and changes across registrations.
#[derive(Clone, Debug)]
pub struct Connection {
    dir: PathBuf,
    account: String,
    pub identity: String,
}

pub struct Login {
    pub message: String,
    pub plan_enabled: bool,
}

fn plan_enabled(tokens: &Tokens) -> bool {
    tokens.scopes.split_whitespace().any(|s| s == DIRECT)
}
const PLAN_DISABLED: &str = "ChatGPT plan use is disabled; run openmax --chatgpt login with this account label to enable it, or select an API-key provider";

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn http() -> Result<reqwest::Client, String> {
    static HTTP: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    if let Some(client) = HTTP.get() {
        return Ok(client.clone());
    }
    let builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30));
    #[cfg(test)]
    let builder = builder.no_proxy();
    let client = builder.build().map_err(|_| "cannot initialize ChatGPT authentication client")?;
    let _ = HTTP.set(client.clone());
    Ok(client)
}

fn load(dir: &Path) -> Result<Store, String> {
    let path = dir.join("chatgpt.json");
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = match options.open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Store::default()),
        Err(_) => return Err("cannot read ChatGPT credential file".into()),
    };
    let meta = file.metadata().map_err(|_| "cannot inspect ChatGPT credential file")?;
    if !meta.is_file() {
        return Err("ChatGPT credentials must be a regular file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if meta.permissions().mode() & 0o077 != 0 || meta.uid() != unsafe { libc::geteuid() } {
            return Err("ChatGPT credentials must be owned by this user with mode 600".into());
        }
    }
    let mut bytes = Vec::new();
    file.take(LIMIT as u64 + 1).read_to_end(&mut bytes).map_err(|_| "cannot read ChatGPT credentials")?;
    if bytes.len() > LIMIT {
        return Err("ChatGPT credential file exceeds its size limit".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "invalid ChatGPT credential file".into())
}

fn save(dir: &Path, store: &Store) -> Result<(), String> {
    let bytes = serde_json::to_vec(store).map_err(|_| "cannot encode ChatGPT credentials")?;
    if bytes.len() > LIMIT {
        return Err("ChatGPT credentials exceed their size limit".into());
    }
    crate::sessions::replace_atomic(&dir.join("chatgpt.json"), &bytes, 0o600, |_| Ok(()))
        .map_err(|_| "cannot save ChatGPT credentials".to_string())?;
    if let Ok(dir) = File::open(dir) {
        let _ = dir.sync_all();
    }
    Ok(())
}

async fn lock(dir: &Path) -> Result<File, String> {
    std::fs::create_dir_all(dir).map_err(|_| "cannot create credential directory")?;
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(dir.join("chatgpt.lock")).map_err(|_| "cannot open credential lock")?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(35);
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await
            }
            _ => return Err("ChatGPT credentials are busy; retry after the other operation finishes".into()),
        }
    }
}

fn label(account: &str) -> Result<(), String> {
    if account.is_empty()
        || account.len() > 64
        || !account.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("account label must be 1 to 64 letters, digits, hyphens or underscores".into());
    }
    Ok(())
}

fn identity(account: &Account) -> String {
    format!("{}:{}", account.client_id, account.subject)
}

pub fn connection(dir: &Path, account: &str) -> Result<Connection, String> {
    label(account)?;
    let store = load(dir)?;
    let entry = store.accounts.get(account).filter(|a| a.tokens.is_some()).ok_or_else(|| {
        format!("ChatGPT account '{account}' is signed out; run openmax --chatgpt login --account {account}")
    })?;
    if !entry.tokens.as_ref().is_some_and(plan_enabled) {
        return Err(PLAN_DISABLED.into());
    }
    Ok(Connection { dir: dir.into(), account: account.into(), identity: identity(entry) })
}

async fn read_json(response: reqwest::Response) -> Result<Value, String> {
    let status = response.status();
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| "ChatGPT response interrupted")?;
        if chunk.len() > LIMIT.saturating_sub(bytes.len()) {
            return Err("ChatGPT response exceeds its size limit".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|_| format!("ChatGPT returned HTTP {status} without valid JSON"))?;
    if !status.is_success() {
        // Never relay an arbitrary token endpoint body into logs or a transcript.
        let code = value["error"].as_str().or_else(|| value["error"]["code"].as_str()).unwrap_or("");
        let terminal = matches!(
            code,
            "invalid_grant"
                | "invalid_refresh_token"
                | "token_expired"
                | "refresh_token_expired"
                | "refresh_token_invalidated"
                | "refresh_token_reused"
        );
        return Err(if terminal {
            "ChatGPT session expired or was revoked; sign in again".into()
        } else {
            format!("ChatGPT request failed (HTTP {status}); check account access and retry")
        });
    }
    Ok(value)
}

async fn token_request(http: &reqwest::Client, url: &str, fields: &[(&str, &str)]) -> Result<Value, String> {
    let response =
        http.post(url).form(fields).send().await.map_err(|_| "ChatGPT token request failed; credentials retained")?;
    read_json(response).await
}

fn tokens(value: &Value, previous_id: Option<&str>) -> Result<Tokens, String> {
    let field = |key: &str| {
        value[key]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .ok_or_else(|| format!("ChatGPT token response lacks {key}"))
    };
    let scopes = field("scope")?;
    if !value["token_type"].as_str().is_some_and(|s| s.eq_ignore_ascii_case("Bearer")) {
        return Err("ChatGPT returned an unsupported token type".into());
    }
    let lifetime = value["expires_in"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 86400)
        .ok_or("ChatGPT returned an invalid token lifetime")?;
    Ok(Tokens {
        access: field("access_token")?,
        refresh: field("refresh_token")?,
        id: value["id_token"].as_str().or(previous_id).ok_or("ChatGPT response lacks ID token")?.into(),
        scopes,
        expires: now().saturating_add(lifetime),
    })
}

impl Connection {
    pub async fn access(&self) -> Result<String, String> {
        self.access_at(&http()?, TOKEN).await
    }

    async fn access_at(&self, http: &reqwest::Client, token_url: &str) -> Result<String, String> {
        // Finish a rotating refresh even if its waiting turn is cancelled.
        // Dropping the join handle detaches it; the file lock still covers persistence.
        let (connection, http, url) = (self.clone(), http.clone(), token_url.to_string());
        PENDING.fetch_add(1, Ordering::SeqCst);
        let pending = Pending;
        if STOPPING.load(Ordering::SeqCst) { return Err("ChatGPT authentication is shutting down".into()); }
        tokio::spawn(async move {
            let _pending = pending;
            connection.renew(&http, &url).await
        })
            .await
            .map_err(|_| "ChatGPT credential refresh task failed")?
    }

    async fn renew(&self, http: &reqwest::Client, token_url: &str) -> Result<String, String> {
        let _lock = lock(&self.dir).await?;
        let mut store = load(&self.dir)?;
        let account = store.accounts.get_mut(&self.account).ok_or("ChatGPT account no longer exists")?;
        if identity(account) != self.identity {
            return Err("ChatGPT account changed; start a new turn".into());
        }
        let old = account.tokens.as_ref().ok_or("ChatGPT account is signed out")?;
        if !plan_enabled(old) {
            return Err(PLAN_DISABLED.into());
        }
        if old.expires > now() + 60 {
            return Ok(old.access.clone());
        }
        let result = token_request(
            http,
            token_url,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", &account.client_id),
                ("refresh_token", &old.refresh),
                ("resource", API),
            ],
        )
        .await;
        let value = match result {
            Ok(v) => v,
            Err(e) => {
                if e.contains("expired or was revoked") {
                    account.tokens = None;
                    save(&self.dir, &store)?;
                }
                return Err(e);
            }
        };
        let renewed = tokens(&value, Some(&old.id))?;
        let enabled = plan_enabled(&renewed);
        let access = renewed.access.clone();
        account.tokens = Some(renewed);
        save(&self.dir, &store)?;
        if !enabled {
            return Err(PLAN_DISABLED.into());
        }
        Ok(access)
    }
}

fn random() -> Result<String, String> {
    let mut bytes = [0; 32];
    SystemRandom::new().fill(&mut bytes).map_err(|_| "secure random source unavailable")?;
    Ok(B64.encode(bytes))
}

fn auth_endpoint<'a>(url: &'a str, discovery_url: &str) -> Result<&'a str, String> {
    let parsed = reqwest::Url::parse(url).map_err(|_| "invalid authentication endpoint")?;
    let origin = reqwest::Url::parse(discovery_url).map_err(|_| "invalid discovery endpoint")?.origin();
    if parsed.origin() != origin
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        return Err("authentication endpoint is outside the trusted issuer".into());
    }
    Ok(url)
}

async fn discovery(http: &reqwest::Client, url: &str) -> Result<Value, String> {
    let response = http.get(url).send().await.map_err(|_| "cannot load OpenAI identity configuration")?;
    let value = read_json(response).await?;
    if value["issuer"] != ISSUER {
        return Err("unexpected OpenAI identity issuer".into());
    }
    Ok(value)
}

fn validate_id(token: &str, keys: &Value, client: &str, nonce: &str) -> Result<Value, String> {
    let invalid = || "ChatGPT ID token failed identity validation".to_string();
    let parts: Vec<_> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(invalid());
    }
    let decode = |part: &str| B64.decode(part).map_err(|_| invalid());
    let header: Value = serde_json::from_slice(&decode(parts[0])?).map_err(|_| invalid())?;
    if header["alg"] != "RS256" {
        return Err(invalid());
    }
    let kid = header["kid"].as_str().ok_or_else(invalid)?;
    let key = keys["keys"]
        .as_array()
        .and_then(|keys| keys.iter().find(|k| k["kid"] == kid && k["kty"] == "RSA"))
        .ok_or_else(invalid)?;
    if key.get("use").is_some_and(|u| u != "sig") || key.get("alg").is_some_and(|a| a != "RS256") {
        return Err(invalid());
    }
    let n = decode(key["n"].as_str().ok_or_else(invalid)?)?;
    let e = decode(key["e"].as_str().ok_or_else(invalid)?)?;
    let signed = format!("{}.{}", parts[0], parts[1]);
    signature::RsaPublicKeyComponents { n: &n, e: &e }
        .verify(&signature::RSA_PKCS1_2048_8192_SHA256, signed.as_bytes(), &decode(parts[2])?)
        .map_err(|_| invalid())?;
    let claims: Value = serde_json::from_slice(&decode(parts[1])?).map_err(|_| invalid())?;
    let audience = claims["aud"] == client || claims["aud"].as_array().is_some_and(|a| a.iter().any(|v| v == client));
    let multiple = claims["aud"].as_array().is_some_and(|a| a.len() > 1);
    if claims["iss"] != ISSUER
        || !audience
        || claims["nonce"] != nonce
        || claims["exp"].as_u64().is_none_or(|exp| exp <= now())
        || claims["iat"].as_u64().is_none_or(|iat| iat > now() + 60)
        || claims.get("nbf").is_some_and(|v| v.as_u64().is_none_or(|n| n > now()))
        || claims["sub"].as_str().is_none_or(str::is_empty)
        || ((multiple || claims.get("azp").is_some()) && claims["azp"] != client)
    {
        return Err(invalid());
    }
    Ok(claims)
}

fn callback(target: &str, state: &str, saved_client: Option<&str>) -> Result<(String, String), String> {
    if !target.starts_with("/auth/callback?") {
        return Err("invalid callback path".into());
    }
    let url = reqwest::Url::parse(&format!("http://127.0.0.1{target}")).map_err(|_| "invalid callback")?;
    if url.path() != "/auth/callback" {
        return Err("invalid callback path".into());
    }
    let pairs: Vec<_> = url.query_pairs().collect();
    let get = |key: &str| -> Result<Option<String>, String> {
        let values: Vec<_> = pairs.iter().filter(|(k, _)| k == key).collect();
        if values.len() > 1 {
            return Err("duplicate OAuth callback parameter".into());
        }
        Ok(values.first().map(|(_, v)| v.to_string()))
    };
    if get("state")?.as_deref() != Some(state) {
        return Err("OAuth state mismatch".into());
    }
    if get("error")?.is_some() {
        return Err("ChatGPT authorization was declined".into());
    }
    let issued = get("client_id")?;
    if saved_client.is_some_and(|saved| issued.as_deref().is_some_and(|id| id != saved)) {
        return Err("ChatGPT registration changed during sign-in".into());
    }
    let client = issued
        .or_else(|| saved_client.map(str::to_string))
        .filter(|s| !s.is_empty() && s != "dynamic_agent_client")
        .ok_or("callback lacks issued client ID")?;
    let code = get("code")?.filter(|s| !s.is_empty()).ok_or("callback lacks authorization code")?;
    Ok((code, client))
}

/// Runs before any project is loaded. The caller owns presentation and browser launch.
pub async fn login(dir: &Path, name: &str, announce: impl FnOnce(&str)) -> Result<Login, String> {
    login_at(dir, name, announce, TOKEN, DISCOVERY).await
}

async fn login_at(
    dir: &Path,
    name: &str,
    announce: impl FnOnce(&str),
    token_url: &str,
    discovery_url: &str,
) -> Result<Login, String> {
    label(name)?;
    let (host, previous) = {
        let _lock = lock(dir).await?;
        let mut store = load(dir)?;
        if store.host.is_empty() {
            store.host = format!("urn:uuid:{}", uuid::Uuid::new_v4());
            save(dir, &store)?;
        }
        (store.host, store.accounts.get(name).cloned())
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.map_err(|_| "cannot start sign-in callback")?;
    let redirect = format!(
        "http://127.0.0.1:{}/auth/callback",
        listener.local_addr().map_err(|_| "cannot read callback address")?.port()
    );
    let verifier = random()?;
    let state = random()?;
    let nonce = random()?;
    let challenge = B64.encode(digest::digest(&digest::SHA256, verifier.as_bytes()));
    let mut url = reqwest::Url::parse(AUTHORIZE).expect("static OAuth URL");
    let client = previous.as_ref().map(|a| a.client_id.as_str());
    url.query_pairs_mut().extend_pairs([
        ("client_id", client.unwrap_or("dynamic_agent_client")),
        ("ext_agent_host_id", &host),
        ("response_type", "code"),
        ("redirect_uri", &redirect),
        ("resource", API),
        ("scope", SCOPES),
        ("state", &state),
        ("nonce", &nonce),
        ("code_challenge", &challenge),
        ("code_challenge_method", "S256"),
    ]);
    if client.is_none() {
        url.query_pairs_mut().append_pair("agent_name_hint", "Open Max");
    }
    if previous.as_ref().and_then(|a| a.tokens.as_ref()).is_some_and(|t| !plan_enabled(t)) {
        url.query_pairs_mut().append_pair("prompt", "consent");
    }
    announce(url.as_str());
    let received = tokio::time::timeout(Duration::from_secs(300), async {
        loop {
            let (mut socket, _) = listener.accept().await.map_err(|_| "sign-in callback failed")?;
            let request = tokio::time::timeout(Duration::from_secs(3), async {
                let mut bytes = Vec::new();
                loop {
                    let mut buf = [0; 1024];
                    let n = socket.read(&mut buf).await.map_err(|_| "callback read failed")?;
                    if n == 0 || bytes.len() + n > 16384 {
                        return Err("invalid callback request");
                    }
                    bytes.extend_from_slice(&buf[..n]);
                    if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                        return Ok(bytes);
                    }
                }
            })
            .await;
            let result = match request {
                Ok(Ok(bytes)) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut words = text.lines().next().unwrap_or("").split_whitespace();
                    if words.next() == Some("GET") {
                        callback(words.next().unwrap_or(""), &state, client)
                    } else {
                        Err("invalid callback method".into())
                    }
                }
                _ => Err("invalid callback request".into()),
            };
            let ok = result.is_ok();
            let message = if ok {
                "Sign-in received. Return to the terminal to complete validation."
            } else {
                "Sign-in callback rejected. Return to the terminal."
            };
            let reply = format!(
                "HTTP/1.1 {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{message}",
                if ok { "200 OK" } else { "400 Bad Request" },
                message.len()
            );
            let _ = tokio::time::timeout(Duration::from_secs(1), socket.write_all(reply.as_bytes())).await;
            match result {
                Ok(v) => return Ok(v),
                Err(e) if e == "ChatGPT authorization was declined" => return Err(e),
                _ => {}
            }
        }
    })
    .await
    .map_err(|_| "ChatGPT sign-in timed out; run login again")??;
    let http = http()?;
    let value = token_request(
        &http,
        token_url,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", &received.1),
            ("code", &received.0),
            ("code_verifier", &verifier),
            ("redirect_uri", &redirect),
            ("resource", API),
        ],
    )
    .await?;
    let credentials = tokens(&value, None)?;
    let enabled = plan_enabled(&credentials);
    let discovery = discovery(&http, discovery_url).await?;
    // Production discovery is fixed; keys and revocation stay on that issuer.
    let jwks = auth_endpoint(discovery["jwks_uri"].as_str().ok_or("missing identity keys endpoint")?, discovery_url)?;
    let keys = read_json(http.get(jwks).send().await.map_err(|_| "cannot load OpenAI identity keys")?).await?;
    let claims = validate_id(&credentials.id, &keys, &received.1, &nonce)?;
    let subject = claims["sub"].as_str().ok_or("missing account identity")?.to_string();
    if previous.as_ref().is_some_and(|a| a.subject != subject) {
        return Err("signed-in identity differs from the selected account; use another account label".into());
    }
    let _lock = lock(dir).await?;
    let mut store = load(dir)?;
    if store.accounts.get(name) != previous.as_ref() {
        return Err("account changed during sign-in; retry".into());
    }
    store.accounts.insert(
        name.into(),
        Account {
            client_id: received.1,
            subject,
            email: claims["email"].as_str().unwrap_or("").into(),
            tokens: Some(credentials),
        },
    );
    save(dir, &store)?;
    let message = if enabled {
        format!("ChatGPT account '{name}' connected. Eligible requests use your ChatGPT plan. Manage usage: https://chatgpt.com/settings/usage")
    } else {
        format!("ChatGPT account '{name}' signed in. {PLAN_DISABLED}")
    };
    Ok(Login { message, plan_enabled: enabled })
}

pub fn status(dir: &Path) -> Result<Vec<String>, String> {
    Ok(load(dir)?
        .accounts
        .iter()
        .map(|(name, a)| {
            format!(
                "{}: {}{}",
                crate::text::one_line(name),
                match &a.tokens {
                    Some(t) if plan_enabled(t) => "signed in",
                    Some(_) => "signed in; ChatGPT plan disabled",
                    None => "signed out",
                },
                if a.email.is_empty() { String::new() } else { format!(" ({})", crate::text::one_line(&a.email)) }
            )
        })
        .collect())
}

pub async fn logout(dir: &Path, name: &str) -> Result<String, String> {
    logout_at(dir, name, DISCOVERY).await
}

async fn logout_at(dir: &Path, name: &str, discovery_url: &str) -> Result<String, String> {
    label(name)?;
    let _lock = lock(dir).await?;
    let mut store = load(dir)?;
    let account = store.accounts.get_mut(name).ok_or("unknown ChatGPT account")?;
    let revoked = if let Some(tokens) = &account.tokens {
        async {
            let http = http()?;
            let doc = discovery(&http, discovery_url).await?;
            let url =
                auth_endpoint(doc["revocation_endpoint"].as_str().ok_or("revocation unavailable")?, discovery_url)?;
            for attempt in 0..2 {
                if attempt > 0 {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                if http
                    .post(url)
                    .form(&[
                        ("token", tokens.refresh.as_str()),
                        ("token_type_hint", "refresh_token"),
                        ("client_id", account.client_id.as_str()),
                    ])
                    .send()
                    .await
                    .is_ok_and(|r| r.status().as_u16() == 200)
                {
                    return Ok::<_, String>(());
                }
            }
            Err("revocation unconfirmed".into())
        }
        .await
        .is_ok()
    } else {
        true
    };
    account.tokens = None;
    save(dir, &store)?;
    Ok(if revoked {
        format!("ChatGPT account '{name}' signed out")
    } else {
        format!("ChatGPT account '{name}' signed out locally. Remote revocation was not confirmed; disconnect the app in ChatGPT Settings.")
    })
}

/// Refresh the account catalog without selecting a provider or changing settings.
pub async fn models(dir: &Path, name: &str) -> Result<Value, String> {
    let connection = connection(dir, name)?;
    let token = connection.access().await?;
    read_json(
        http()?
            .get(format!("{API}/models"))
            .bearer_auth(token)
            .send()
            .await
            .map_err(|_| "ChatGPT model discovery failed")?,
    )
    .await
}

/// Only this explicitly requested operation writes a catalog; other providers keep their bytes as JSON values.
pub async fn save_catalog(
    dir: &Path,
    name: &str,
    catalog: &Value,
) -> Result<Vec<crate::providers::ProviderModel>, String> {
    label(name)?;
    let models: Vec<_> = catalog["models"]
        .as_array()
        .ok_or("ChatGPT returned no model catalog")?
        .iter()
        .filter(|m| m["visibility"] == "list")
        .filter_map(|m| {
            Some(crate::providers::ProviderModel {
                id: m["slug"].as_str()?.into(),
                name: m["display_name"].as_str().map(str::to_string),
                context_tokens: m["context_window"].as_u64().and_then(|n| usize::try_from(n).ok()).filter(|n| *n > 0),
                max_tokens: None,
            })
        })
        .collect();
    crate::providers::edit_catalog(dir, |providers| {
        let key = format!("chatgpt-{name}");
        if let Some(existing) = providers.get(&key) {
            if existing["chatgpt_account"] != name || existing["base_url"] != API || existing["wire_api"] != "responses" {
                return Err(format!("provider '{key}' already exists; catalog was not overwritten"));
            }
        }
        let entry =
            providers.entry(key).or_insert_with(|| json!({"base_url":API,"wire_api":"responses","chatgpt_account":name}));
        // Preserve explicitly configured windows when the service does not report one.
        let previous = entry["models"].as_array().cloned().unwrap_or_default();
        let models: Vec<_> = models
            .into_iter()
            .map(|mut m| {
                if let Some(old) = previous.iter().find(|old| old["id"] == m.id) {
                    m.context_tokens = m.context_tokens.or_else(|| {
                        old["context_tokens"].as_u64().and_then(|n| usize::try_from(n).ok()).filter(|n| *n > 0)
                    });
                    m.max_tokens = old["max_tokens"].as_u64().and_then(|n| usize::try_from(n).ok()).filter(|n| *n > 0);
                }
                m
            })
            .collect();
        entry["models"] = json!(models);
        Ok(models)
    }).await
}

#[cfg(test)]
pub(crate) mod tests;
