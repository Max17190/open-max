//! Update notices and `openmax --update`.
//!
//! The interactive session checks for a newer release at most once a day, in
//! the background, and never installs anything itself: Homebrew, npm, bun and
//! the install script each own the binary they put in place, and a version
//! that changes under a running eval or session would make its results
//! unreproducible. A newer release is named once, at the start of the next
//! session, with the one command that updates this install, and
//! `openmax --update` runs that command.
//!
//! Only an install made by one of those four is checked: a source build or a
//! binary copied by hand makes no request. `OPENMAX_NO_UPDATE_CHECK=1` turns
//! the check off, and print, stdio, sub-agent and check runs never make it.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The latest published release. Prereleases and drafts are not "latest".
pub const RELEASES_URL: &str = "https://api.github.com/repos/Max17190/open-max/releases/latest";
const INSTALL_SCRIPT: &str = "https://useopenmax.dev/install.sh";
const INSTALL_DOCS: &str = "https://github.com/Max17190/open-max/blob/main/docs/install.md";
/// How old a cached answer may be before a session asks again.
const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CURRENT: &str = env!("CARGO_PKG_VERSION");

/// How this binary was installed, which decides the command that updates it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Install {
    Homebrew,
    Npm,
    Bun,
    Pnpm,
    /// The install script, into this directory.
    Script { dir: PathBuf, default_dir: bool },
    /// A source build, a copied binary, or anything else: never checked.
    Unknown,
}

impl Install {
    /// The install this process runs from, from its executable's real path.
    pub fn current() -> Self {
        let Some(exe) = std::env::current_exe().ok().and_then(|p| p.canonicalize().ok()) else {
            return Self::Unknown;
        };
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let receipt_dir = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|h| h.join(".config")))
            .and_then(|config| receipt_prefix(&config.join("openmax/openmax-receipt.json")));
        Self::detect(&exe, home.as_deref(), receipt_dir.as_deref())
    }

    /// `exe` is the canonical path of the binary; `receipt_dir` the directory
    /// the install script's receipt says it installed into.
    pub fn detect(exe: &Path, home: Option<&Path>, receipt_dir: Option<&Path>) -> Self {
        let path = exe.to_string_lossy();
        if path.contains("/Cellar/openmax/") {
            return Self::Homebrew;
        }
        // The npm package's launcher fetches the binary into its own
        // `node_modules/.bin_real`. A one-off `npx` run lives in npm's
        // cache and is replaced by the next one, so it is not checked.
        if path.contains("/node_modules/openmax-cli/") && !path.contains("/_npx/") {
            return if path.contains("/.bun/") {
                Self::Bun
            } else if path.contains("/pnpm/") || path.contains("/.pnpm/") {
                Self::Pnpm
            } else {
                Self::Npm
            };
        }
        let Some(dir) = exe.parent() else { return Self::Unknown };
        let default = home.map(|h| h.join(".openmax/bin"));
        let same = |other: &Path| other.canonicalize().is_ok_and(|other| other == dir);
        if default.as_deref().is_some_and(same) {
            return Self::Script { dir: dir.to_path_buf(), default_dir: true };
        }
        if receipt_dir.is_some_and(same) {
            return Self::Script { dir: dir.to_path_buf(), default_dir: false };
        }
        Self::Unknown
    }

    /// The command that updates this install, as a user would type it.
    pub fn command(&self) -> Option<String> {
        Some(match self {
            Self::Homebrew => "brew upgrade openmax".into(),
            Self::Npm => "npm install -g openmax-cli@latest".into(),
            Self::Bun => "bun install -g openmax-cli@latest".into(),
            Self::Pnpm => "pnpm add -g openmax-cli@latest".into(),
            Self::Script { default_dir: true, .. } => format!("curl -fsSL {INSTALL_SCRIPT} | sh"),
            Self::Script { dir, default_dir: false } => {
                format!("curl -fsSL {INSTALL_SCRIPT} | OPENMAX_INSTALL_DIR={} sh", shell_quote(&dir.to_string_lossy()))
            }
            Self::Unknown => return None,
        })
    }

    /// The process that runs `command`. The install script leaves the
    /// shell profile alone: the first install already put the directory on
    /// `PATH`.
    fn process(&self) -> Option<std::process::Command> {
        let words: &[&str] = match self {
            Self::Homebrew => &["brew", "upgrade", "openmax"],
            Self::Npm => &["npm", "install", "-g", "openmax-cli@latest"],
            Self::Bun => &["bun", "install", "-g", "openmax-cli@latest"],
            Self::Pnpm => &["pnpm", "add", "-g", "openmax-cli@latest"],
            Self::Script { dir, .. } => {
                let mut command = std::process::Command::new("sh");
                command
                    .args(["-c", &format!("curl -fsSL {INSTALL_SCRIPT} | sh")])
                    .env("OPENMAX_NO_MODIFY_PATH", "1")
                    .env("OPENMAX_INSTALL_DIR", dir);
                return Some(command);
            }
            Self::Unknown => return None,
        };
        let mut command = std::process::Command::new(words[0]);
        command.args(&words[1..]);
        Some(command)
    }
}

fn shell_quote(word: &str) -> String {
    if !word.is_empty() && word.bytes().all(|b| b.is_ascii_alphanumeric() || b"/._-~".contains(&b)) {
        word.to_string()
    } else {
        format!("'{}'", word.replace('\'', r"'\''"))
    }
}

/// The `install_prefix` the install script's receipt records.
fn receipt_prefix(receipt: &Path) -> Option<PathBuf> {
    #[derive(Deserialize)]
    struct Receipt {
        install_prefix: PathBuf,
    }
    let text = std::fs::read_to_string(receipt).ok()?;
    serde_json::from_str::<Receipt>(&text).ok().map(|r| r.install_prefix)
}

/// Whether `latest` is a later release than `current`. Versions are dotted
/// numbers (`2026.11.3`); anything else is never newer.
pub fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |v: &str| v.trim().trim_start_matches('v').split('.').map(|p| p.parse::<u64>().ok()).collect::<Option<Vec<_>>>();
    match (parse(latest), parse(current)) {
        (Some(latest), Some(current)) => latest > current,
        _ => false,
    }
}

/// Whether the background check may run: `OPENMAX_NO_UPDATE_CHECK` unset
/// (or `0`), and an install it knows how to update.
pub fn check_enabled(install: &Install) -> bool {
    let off = std::env::var("OPENMAX_NO_UPDATE_CHECK").is_ok_and(|v| !v.is_empty() && v != "0");
    !off && *install != Install::Unknown
}

/// The last answer, kept in `~/.openmax/update-check.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cache {
    /// Unix seconds of the last attempt, answered or not.
    pub checked_at: u64,
    /// The latest release the last answered check saw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest: Option<String>,
}

pub fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("update-check.json")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Cache {
    pub fn read(data_dir: &Path) -> Self {
        std::fs::read_to_string(cache_path(data_dir))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// Written to a temporary file and renamed, so a session reading it
    /// never sees half of one.
    fn write(&self, data_dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(data_dir)?;
        let tmp = data_dir.join(format!("update-check.json.{}", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec(self).unwrap_or_default())?;
        std::fs::rename(&tmp, cache_path(data_dir))
    }

    /// Due for another check at `now`: never checked, older than a day, or
    /// stamped in the future by a clock that moved back.
    pub fn due(&self, now: u64) -> bool {
        now.saturating_sub(self.checked_at) >= CHECK_INTERVAL.as_secs() || self.checked_at > now
    }
}

/// The latest release's version from a releases API answer at `url`.
pub async fn fetch_latest(url: &str) -> Result<String, String> {
    #[derive(Deserialize)]
    struct Release {
        tag_name: String,
    }
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .user_agent(concat!("openmax/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| e.to_string())?;
    let release: Release = client
        .get(url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let version = release.tag_name.trim_start_matches('v').to_string();
    if is_newer(&version, "0") {
        Ok(version)
    } else {
        Err(format!("unexpected release tag {:?}", release.tag_name))
    }
}

/// Refresh the cache from `url`. A failed check still stamps the time, so an
/// offline machine asks once a day rather than on every launch.
pub async fn refresh(data_dir: &Path, url: &str) -> Cache {
    let previous = Cache::read(data_dir);
    let latest = fetch_latest(url).await.ok().or(previous.latest);
    let cache = Cache { checked_at: now(), latest };
    let _ = cache.write(data_dir);
    cache
}

/// The session's start: the notice the cached answer gives, if any, and a
/// background refresh when the answer is a day old. The refresh never
/// delays the session; what it finds is shown at the next one.
pub fn on_session_start(data_dir: &Path) -> Option<String> {
    let install = Install::current();
    if !check_enabled(&install) {
        return None;
    }
    let cache = Cache::read(data_dir);
    if cache.due(now()) {
        let data_dir = data_dir.to_path_buf();
        tokio::spawn(async move { refresh(&data_dir, RELEASES_URL).await });
    }
    notice(&cache, &install)
}

/// The one line a session shows when the cache names a newer release.
pub fn notice(cache: &Cache, install: &Install) -> Option<String> {
    let latest = cache.latest.as_deref().filter(|latest| is_newer(latest, CURRENT))?;
    let command = install.command()?;
    Some(format!("openmax {latest} is available (this is {CURRENT}); update with `openmax --update` or `{command}`"))
}

/// What `/status` says about update checks.
pub fn status() -> String {
    let install = Install::current();
    if !check_enabled(&install) {
        return match install {
            Install::Unknown => "off (not a Homebrew, npm, bun or install-script install)".into(),
            _ => "off (OPENMAX_NO_UPDATE_CHECK)".into(),
        };
    }
    "daily check of api.github.com for a newer release; nothing installs itself".into()
}

/// `openmax --update`: say whether a newer release exists, then run the
/// command that updates this install. A failed check still runs it: the
/// package manager has the last word. Returns the exit code.
pub async fn run(data_dir: &Path) -> i32 {
    let install = Install::current();
    let Some(mut process) = install.process() else {
        let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default();
        eprintln!(
            "openmax: {exe} was not installed by Homebrew, npm, bun or the install script, so it cannot update itself; \
             see {INSTALL_DOCS}"
        );
        return 1;
    };
    match fetch_latest(RELEASES_URL).await {
        Ok(latest) => {
            let _ = Cache { checked_at: now(), latest: Some(latest.clone()) }.write(data_dir);
            if !is_newer(&latest, CURRENT) {
                println!("openmax {CURRENT} is the latest release");
                return 0;
            }
            println!("updating openmax {CURRENT} to {latest}");
        }
        Err(e) => eprintln!("openmax: could not check for a newer release ({e}); updating anyway"),
    }
    let command = install.command().unwrap_or_default();
    println!("$ {command}");
    match tokio::task::spawn_blocking(move || process.status()).await {
        Ok(Ok(status)) if status.success() => 0,
        Ok(Ok(status)) => {
            eprintln!("openmax: `{command}` failed ({status})");
            status.code().unwrap_or(1)
        }
        Ok(Err(e)) => {
            eprintln!("openmax: could not run `{command}`: {e}");
            1
        }
        Err(e) => {
            eprintln!("openmax: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_compares_dotted_numbers() {
        assert!(is_newer("2026.11.4", "2026.11.3"));
        assert!(is_newer("v2026.12.0", "2026.11.30"));
        assert!(is_newer("2026.11.3.1", "2026.11.3"));
        assert!(!is_newer("2026.11.3", "2026.11.3"));
        assert!(!is_newer("2026.10.9", "2026.11.0"));
        assert!(!is_newer("2026.11.4-rc.1", "2026.11.3"));
        assert!(!is_newer("garbage", "2026.11.3"));
    }

    #[test]
    fn detects_each_install() {
        let scratch = std::env::temp_dir().join(format!("openmax-update-{}", std::process::id()));
        let home = scratch.join("home");
        let default_bin = home.join(".openmax/bin");
        let custom_bin = scratch.join("tools");
        std::fs::create_dir_all(&default_bin).unwrap();
        std::fs::create_dir_all(&custom_bin).unwrap();
        let (default_bin, custom_bin) = (default_bin.canonicalize().unwrap(), custom_bin.canonicalize().unwrap());
        let detect = |exe: &Path, receipt: Option<&Path>| Install::detect(exe, Some(&home), receipt);

        assert_eq!(detect(Path::new("/opt/homebrew/Cellar/openmax/2026.11.3/bin/openmax"), None), Install::Homebrew);
        assert_eq!(
            detect(Path::new("/usr/lib/node_modules/openmax-cli/node_modules/.bin_real/openmax"), None),
            Install::Npm
        );
        assert_eq!(
            detect(Path::new("/u/.bun/install/global/node_modules/openmax-cli/node_modules/.bin_real/openmax"), None),
            Install::Bun
        );
        assert_eq!(
            detect(Path::new("/u/.local/share/pnpm/global/5/node_modules/openmax-cli/node_modules/.bin_real/openmax"), None),
            Install::Pnpm
        );
        assert_eq!(
            detect(Path::new("/u/.npm/_npx/ab12/node_modules/openmax-cli/node_modules/.bin_real/openmax"), None),
            Install::Unknown
        );
        assert_eq!(
            detect(&default_bin.join("openmax"), None),
            Install::Script { dir: default_bin.clone(), default_dir: true }
        );
        assert_eq!(
            detect(&custom_bin.join("openmax"), Some(&custom_bin)),
            Install::Script { dir: custom_bin.clone(), default_dir: false }
        );
        assert_eq!(detect(&custom_bin.join("openmax"), None), Install::Unknown);
        assert_eq!(detect(Path::new("/src/open-max/target/release/openmax"), None), Install::Unknown);
        let _ = std::fs::remove_dir_all(&scratch);
    }

    #[test]
    fn commands_match_each_install() {
        assert_eq!(Install::Homebrew.command().unwrap(), "brew upgrade openmax");
        assert_eq!(Install::Npm.command().unwrap(), "npm install -g openmax-cli@latest");
        assert_eq!(Install::Bun.command().unwrap(), "bun install -g openmax-cli@latest");
        let script = Install::Script { dir: "/home/u/.openmax/bin".into(), default_dir: true };
        assert_eq!(script.command().unwrap(), "curl -fsSL https://useopenmax.dev/install.sh | sh");
        let custom = Install::Script { dir: "/opt/my tools".into(), default_dir: false };
        assert_eq!(
            custom.command().unwrap(),
            "curl -fsSL https://useopenmax.dev/install.sh | OPENMAX_INSTALL_DIR='/opt/my tools' sh"
        );
        assert_eq!(Install::Unknown.command(), None);
    }

    #[test]
    fn a_check_is_due_daily() {
        let day = CHECK_INTERVAL.as_secs();
        assert!(Cache::default().due(1_000_000));
        assert!(!Cache { checked_at: 1_000_000, latest: None }.due(1_000_000 + day - 1));
        assert!(Cache { checked_at: 1_000_000, latest: None }.due(1_000_000 + day));
        assert!(Cache { checked_at: 1_000_000 + 60, latest: None }.due(1_000_000));
    }

    #[test]
    fn the_notice_names_a_newer_release_and_its_command() {
        let newer = Cache { checked_at: 0, latest: Some("9999.1.0".into()) };
        let line = notice(&newer, &Install::Homebrew).unwrap();
        assert!(line.starts_with("openmax 9999.1.0 is available"), "{line}");
        assert!(line.contains("`brew upgrade openmax`"), "{line}");
        assert_eq!(notice(&newer, &Install::Unknown), None);
        assert_eq!(notice(&Cache { checked_at: 0, latest: Some(CURRENT.into()) }, &Install::Npm), None);
        assert_eq!(notice(&Cache::default(), &Install::Npm), None);
    }

    /// One HTTP answer on a local port: `body` with `status`.
    async fn serve_once(status: &'static str, body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/releases/latest", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0; 4096];
            let _ = socket.read(&mut request).await;
            let reply = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
        });
        url
    }

    #[tokio::test]
    async fn refresh_caches_the_latest_release_and_keeps_it_through_a_failure() {
        let data = std::env::temp_dir().join(format!("openmax-update-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        let url = serve_once("200 OK", r#"{"tag_name":"v2099.1.2","name":"x"}"#).await;
        let cache = refresh(&data, &url).await;
        assert_eq!(cache.latest.as_deref(), Some("2099.1.2"));
        assert_eq!(Cache::read(&data), cache);

        let url = serve_once("403 Forbidden", r#"{"message":"rate limited"}"#).await;
        let after = refresh(&data, &url).await;
        assert_eq!(after.latest.as_deref(), Some("2099.1.2"), "a failed check keeps the last answer");
        assert!(after.checked_at >= cache.checked_at);
        assert!(!after.due(now()), "a failed check still waits a day");
        let _ = std::fs::remove_dir_all(&data);
    }
}
