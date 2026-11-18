//! Update notices and `openmax --update` (docs/install.md#updating). A session
//! of an installed release asks GitHub for the latest release at most daily,
//! in the background, and names a newer one at the next start. Nothing ever
//! installs itself: the package manager owns the binary, and evals need it
//! pinned. Source builds and copied binaries are never checked.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const RELEASES_URL: &str = "https://api.github.com/repos/Max17190/open-max/releases/latest";
const SCRIPT: &str = "curl -fsSL https://useopenmax.dev/install.sh |";
const DAY: u64 = 24 * 60 * 60;
const CURRENT: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Install { Homebrew, Npm, Bun, Pnpm, Script { dir: PathBuf, default_dir: bool }, Unknown }

impl Install {
    pub fn current() -> Self {
        let Some(exe) = std::env::current_exe().ok().and_then(|p| p.canonicalize().ok()) else { return Self::Unknown };
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let config = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| home.as_ref().map(|h| h.join(".config")));
        // The install script's receipt records the directory it installed into.
        let receipt = config.and_then(|c| std::fs::read_to_string(c.join("openmax/openmax-receipt.json")).ok());
        let prefix = receipt.and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok())
            .and_then(|r| r["install_prefix"].as_str().map(PathBuf::from));
        Self::detect(&exe, home.as_deref(), prefix.as_deref())
    }

    /// From the binary's canonical path: the npm launcher keeps it in its own
    /// `node_modules/.bin_real`; an `npx` copy is replaced by the next run.
    pub fn detect(exe: &Path, home: Option<&Path>, receipt_dir: Option<&Path>) -> Self {
        let path = exe.to_string_lossy();
        if path.contains("/Cellar/openmax/") { return Self::Homebrew }
        if path.contains("/node_modules/openmax-cli/") && !path.contains("/_npx/") {
            return if path.contains("/.bun/") { Self::Bun } else if path.contains("pnpm/") { Self::Pnpm } else { Self::Npm };
        }
        let Some(dir) = exe.parent() else { return Self::Unknown };
        let same = |other: &Path| other.canonicalize().is_ok_and(|other| other == dir);
        match (home.map(|h| h.join(".openmax/bin")).as_deref().is_some_and(same), receipt_dir.is_some_and(same)) {
            (false, false) => Self::Unknown,
            (default_dir, _) => Self::Script { dir: dir.to_path_buf(), default_dir },
        }
    }

    /// The command that updates this install, as a user would type it.
    pub fn command(&self) -> Option<String> {
        Some(match self {
            Self::Homebrew => "brew upgrade openmax".into(),
            Self::Npm => "npm install -g openmax-cli@latest".into(),
            Self::Bun => "bun install -g openmax-cli@latest".into(),
            Self::Pnpm => "pnpm add -g openmax-cli@latest".into(),
            Self::Script { default_dir: true, .. } => format!("{SCRIPT} sh"),
            Self::Script { dir, .. } => format!("{SCRIPT} OPENMAX_INSTALL_DIR={} sh", open_max_core::doctor::shell_quote(dir)),
            Self::Unknown => return None,
        })
    }
}

/// Whether `latest` is a later dotted-number version than `current`.
pub fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |v: &str| v.trim_start_matches('v').split('.').map(|p| p.parse::<u64>().ok()).collect::<Option<Vec<_>>>();
    matches!((parse(latest), parse(current)), (Some(l), Some(c)) if l > c)
}

fn check_enabled(install: &Install) -> bool {
    *install != Install::Unknown && std::env::var("OPENMAX_NO_UPDATE_CHECK").map_or(true, |v| v.is_empty() || v == "0")
}

/// `~/.openmax/update-check.json`: the last attempt (answered or not, so an
/// offline machine asks daily, not every launch) and the last answer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cache { pub checked_at: u64, pub latest: Option<String> }

fn now() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) }

impl Cache {
    pub fn read(data_dir: &Path) -> Self {
        std::fs::read(data_dir.join("update-check.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    fn write(&self, data_dir: &Path) {
        let tmp = data_dir.join(format!("update-check.json.{}", std::process::id()));
        let _ = std::fs::create_dir_all(data_dir).and_then(|_| std::fs::write(&tmp, serde_json::to_vec(self).unwrap_or_default()))
            .and_then(|_| std::fs::rename(&tmp, data_dir.join("update-check.json")));
    }

    /// A day old, never checked, or stamped ahead by a clock that moved back.
    pub fn due(&self, now: u64) -> bool { now.saturating_sub(self.checked_at) >= DAY || self.checked_at > now }
}

pub async fn fetch_latest(url: &str) -> Result<String, String> {
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(10))
        .user_agent(concat!("openmax/", env!("CARGO_PKG_VERSION"))).build().map_err(|e| e.to_string())?;
    let reply = client.get(url).header("Accept", "application/vnd.github+json").send().await.and_then(|r| r.error_for_status());
    let release: serde_json::Value = reply.map_err(|e| e.to_string())?.json().await.map_err(|e| e.to_string())?;
    let version = release["tag_name"].as_str().unwrap_or_default().trim_start_matches('v');
    if is_newer(version, "0") { Ok(version.to_string()) } else { Err(format!("unexpected release {}", release["tag_name"])) }
}

pub async fn refresh(data_dir: &Path, url: &str) -> Cache {
    let latest = fetch_latest(url).await.ok().or(Cache::read(data_dir).latest);
    let cache = Cache { checked_at: now(), latest };
    cache.write(data_dir);
    cache
}

/// At a session's start: the notice the cached answer gives, and a refresh
/// in the background when it is a day old, shown at the next start.
pub fn on_session_start(data_dir: &Path) -> Option<String> {
    let install = Install::current();
    if !check_enabled(&install) { return None }
    let cache = Cache::read(data_dir);
    if cache.due(now()) {
        let data_dir = data_dir.to_path_buf();
        tokio::spawn(async move { refresh(&data_dir, RELEASES_URL).await });
    }
    notice(&cache, &install)
}

pub fn notice(cache: &Cache, install: &Install) -> Option<String> {
    let latest = cache.latest.as_deref().filter(|latest| is_newer(latest, CURRENT))?;
    Some(format!("openmax {latest} is available (this is {CURRENT}); type /update, or run `{}`", install.command()?))
}

pub fn status() -> String {
    match Install::current() {
        Install::Unknown => "off (not installed by Homebrew, npm, bun or the install script)".into(),
        install if !check_enabled(&install) => "off (OPENMAX_NO_UPDATE_CHECK)".into(),
        _ => "daily check of api.github.com for a newer release; nothing installs itself".into(),
    }
}

const NOT_INSTALLED: &str = "this binary was not installed by Homebrew, npm, bun or the install script, so it cannot update itself; \
    see https://github.com/Max17190/open-max/blob/main/docs/install.md";

/// The session `/update` leaves: Some once asked for, holding its id if any.
static REQUESTED: std::sync::Mutex<Option<Option<String>>> = std::sync::Mutex::new(None);

/// Whether `/update` should leave the session: an install it can update, and
/// a newer release or a failed check (the package manager then decides).
pub async fn prepare() -> Result<(), String> {
    Install::current().command().ok_or(NOT_INSTALLED)?;
    match fetch_latest(RELEASES_URL).await {
        Ok(latest) if !is_newer(&latest, CURRENT) => Err(format!("openmax {CURRENT} is the latest release")),
        _ => Ok(()),
    }
}

pub fn request(session: Option<String>) {
    *REQUESTED.lock().unwrap_or_else(|e| e.into_inner()) = Some(session);
}

/// Run this install's update command on the restored terminal; Ok when it succeeded.
fn install(data_dir: &Path) -> Result<(), i32> {
    let command = Install::current().command().ok_or_else(|| { eprintln!("openmax: {NOT_INSTALLED}"); 1 })?;
    Cache { checked_at: now(), latest: None }.write(data_dir);
    println!("$ {command}");
    // The first install already put the directory on PATH.
    match std::process::Command::new("sh").args(["-c", &command]).env("OPENMAX_NO_MODIFY_PATH", "1").status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => { eprintln!("openmax: `{command}` failed ({status})"); Err(status.code().unwrap_or(1)) }
        Err(e) => { eprintln!("openmax: could not run `{command}`: {e}"); Err(1) }
    }
}

/// After a session `/update` ended, once its terminal is restored: update,
/// then replace this process with the new binary resuming that session.
/// None when no `/update` was asked for; else the exit code of a failure.
pub fn relaunch(core: &open_max_core::state::Core) -> Option<i32> {
    let session = REQUESTED.lock().unwrap_or_else(|e| e.into_inner()).take()?;
    // A session discarded on exit for being empty has nothing to resume.
    let resume = session.filter(|id| open_max_core::sessions::meta(core, id).is_some());
    let hint = if resume.is_some() { "openmax --continue" } else { "openmax" };
    if let Err(code) = install(&core.data_dir) {
        eprintln!("openmax: continue with `{hint}`");
        return Some(code);
    }
    // The name it was started by, so a PATH lookup finds the new version.
    let program = std::env::args_os().next().unwrap_or_else(|| "openmax".into());
    let error = std::os::unix::process::CommandExt::exec(std::process::Command::new(program).args(resume.map(|_| "--continue")));
    eprintln!("openmax: updated, but could not start it again ({error}); continue with `{hint}`");
    Some(1)
}

/// `openmax --update`: run this install's update command when a newer
/// release exists, or when the check fails (the package manager decides).
pub async fn run(data_dir: &Path) -> i32 {
    if Install::current().command().is_none() {
        eprintln!("openmax: {NOT_INSTALLED}");
        return 1;
    }
    match fetch_latest(RELEASES_URL).await {
        Ok(latest) if !is_newer(&latest, CURRENT) => { println!("openmax {CURRENT} is the latest release"); return 0 }
        Ok(latest) => println!("updating openmax {CURRENT} to {latest}"),
        Err(e) => eprintln!("openmax: could not check for a newer release ({e}); updating anyway"),
    }
    install(data_dir).err().unwrap_or(0)
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
        let day = DAY;
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
