//! `openmax --check`: validate every extension surface and say why a file
//! would be ignored, instead of the silent skip the agent loop does. This is
//! how the agent verifies its own self-modifications (run it via bash after
//! writing an extension file) and how a human debugs a hook that "does
//! nothing" or a permissions file that fails closed.
//!
//! A file is ignored for more reasons than being unparseable. It can sit at a
//! path nothing reads, lose its name to a file in another tier, or name a tool
//! that does not exist. Those are reported as warnings rather than errors:
//! each is legitimate in some project, so none of them fails the run.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::tools;

#[derive(Debug)]
pub enum Status {
    /// Loaded and live. Holds a short summary (name, event, rule count).
    Ok(String),
    /// Loads, but will not do what it looks like: a file at a path nothing
    /// reads, a definition another tier overrides, a rule that cannot match.
    Warn(String),
    /// The agent loop ignores this file or fails closed on it.
    Err(String),
}

impl Status {
    pub fn summary(&self) -> &str {
        match self {
            Status::Ok(s) | Status::Warn(s) | Status::Err(s) => s,
        }
    }

    /// Stable machine-readable discriminator for `--check --json`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Ok(_) => "ok",
            Status::Warn(_) => "warn",
            Status::Err(_) => "err",
        }
    }
}

#[derive(Debug)]
pub struct Finding {
    /// Surface: tool, skill, template, hook, permissions, providers, or path.
    pub kind: &'static str,
    pub path: PathBuf,
    pub status: Status,
}

pub fn has_errors(findings: &[Finding]) -> bool {
    findings.iter().any(|f| matches!(f.status, Status::Err(_)))
}

pub fn has_warnings(findings: &[Finding]) -> bool {
    findings.iter().any(|f| matches!(f.status, Status::Warn(_)))
}

/// Validate all extension files for a project (global + project dirs).
/// Missing dirs and files contribute nothing; an empty report means an empty
/// (and healthy) configuration.
pub fn check(project_root: &Path) -> Vec<Finding> {
    check_at(project_root, &crate::state::default_data_dir())
}

/// A finding plus the identity a loader would file it under, used to work out
/// which of two files with the same identity actually wins.
type Entry = (Finding, Option<String>);

pub(crate) fn check_at(project_root: &Path, data_dir: &Path) -> Vec<Finding> {
    let mut findings = Vec::new();

    let mut tools_found: Vec<Entry> = Vec::new();
    #[allow(unused_assignments)]
    let mut external_names: Vec<String> = Vec::new();
    for dir in crate::registry::external_tool_dirs(project_root) {
        for path in files_with_extension(&dir, "toml") {
            let parsed = crate::registry::parse_tool_file(&path);
            let mut id = None;
            let status = match parsed {
                Ok(spec) if tools::TOOL_NAMES.contains(&spec.name.as_str()) => {
                    Status::Err(format!("'{}' shadows a built-in tool and is ignored", spec.name))
                }
                Ok(spec) => {
                    id = Some(spec.name.clone());
                    external_names.push(spec.name.clone());
                    let command = match &spec.kind {
                        crate::registry::ToolKind::External(ext) => Some(ext.command.clone()),
                        crate::registry::ToolKind::Builtin => None,
                    };
                    match command.and_then(|c| missing_command_reason(&c, project_root)) {
                        Some(reason) => Status::Warn(reason),
                        None => Status::Ok(format!("tool '{}'", spec.name)),
                    }
                }
                Err(reason) => Status::Err(reason),
            };
            tools_found.push((Finding { kind: "tool", path, status }, id));
        }
    }
    // Later directories overwrite earlier ones by name, so the last file to
    // claim a name is the live one.
    mark_shadowed(&mut tools_found, false);
    mark_beyond_cap(
        &mut tools_found,
        crate::registry::MAX_EXTERNAL_TOOLS,
        |_| true,
        "tool cap",
    );
    // Rules and hook filters resolve against what actually loads, so the
    // known-tool set is the live entries after shadowing and the cap - a
    // rule naming a beyond-cap tool is as dead as one naming a typo.
    external_names = tools_found
        .iter()
        .filter(|(f, id)| id.is_some() && matches!(f.status, Status::Ok(_)))
        .filter_map(|(_, id)| id.clone())
        .collect();
    findings.extend(tools_found.into_iter().map(|(f, _)| f));

    let mut skills_found: Vec<Entry> = Vec::new();
    for dir in crate::skills::skill_dirs(project_root) {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        dirs.sort();
        for entry in dirs {
            if entry.is_file() {
                findings.push(Finding {
                    kind: "path",
                    path: entry,
                    status: Status::Warn(
                        "a skill is a directory holding a SKILL.md, so this file is not read"
                            .into(),
                    ),
                });
                continue;
            }
            let path = entry.join("SKILL.md");
            if !path.is_file() {
                if !dir_is_empty(&entry) {
                    // Case-insensitive filesystems resolve skill.md here and
                    // case-sensitive ones do not, so the same repo loads this
                    // skill on one machine and silently drops it on another.
                    let hint = match miscased_skill_file(&entry) {
                        Some(found) => format!(", but {} is",
                            found.file_name().unwrap_or_default().to_string_lossy()),
                        None => String::new(),
                    };
                    findings.push(Finding {
                        kind: "path",
                        path: entry,
                        status: Status::Warn(format!(
                            "no skill loads from here: SKILL.md is not spelled exactly{hint}"
                        )),
                    });
                }
                continue;
            }
            let mut id = None;
            let status = match crate::skills::parse_skill_md(&path) {
                Ok(s) => {
                    id = Some(s.name.clone());
                    Status::Ok(format!("skill '{}'", s.name))
                }
                Err(reason) => Status::Err(reason),
            };
            skills_found.push((Finding { kind: "skill", path, status }, id));
        }
    }
    mark_shadowed(&mut skills_found, false);
    findings.extend(skills_found.into_iter().map(|(f, _)| f));

    let mut templates_found: Vec<Entry> = Vec::new();
    for dir in crate::templates::template_dirs(project_root) {
        for path in files_with_extension(&dir, "md") {
            let mut id = None;
            let status = match crate::templates::parse_template(&path) {
                Ok(t) => {
                    id = Some(t.name.clone());
                    Status::Ok(format!("template /{}", t.name))
                }
                Err(reason) => Status::Err(reason),
            };
            templates_found.push((Finding { kind: "template", path, status }, id));
        }
    }
    mark_shadowed(&mut templates_found, false);
    findings.extend(templates_found.into_iter().map(|(f, _)| f));

    let known_tools = known_tool_names(&external_names);
    let mut hooks_found: Vec<Entry> = Vec::new();
    // Aligned with hooks_found: the parsed event of each Ok entry.
    let mut hook_events: Vec<Option<&'static str>> = Vec::new();
    let mut hook_extras: Vec<Finding> = Vec::new();
    for dir in crate::hooks::hook_dirs(project_root) {
        for path in files_with_extension(&dir, "toml") {
            // Hooks resolve by file stem, and the first stem to appear claims
            // it whether or not the file parses.
            let id = path.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string());
            let status = match crate::hooks::parse_hook_file(&path) {
                Ok(h) => {
                    if let Some(filter) = h.tool_filter.as_deref() {
                        if let Some(reason) = unknown_tool_reason(
                            filter,
                            &known_tools,
                            &path,
                            project_root,
                            "this hook never runs",
                        ) {
                            hook_extras.push(Finding {
                                kind: "hook",
                                path: path.clone(),
                                status: Status::Warn(reason),
                            });
                        }
                    }
                    hook_events.push(Some(h.event.as_str()));
                    if !crate::ledger::is_approved(data_dir, project_root, &h.source_sha256) {
                        // Hooks run with host authority and no per-call gate,
                        // so unapproved content is inert, and this is where
                        // that inertness stops being silent.
                        Status::Err(format!(
                            "unapproved and inert: a human must approve this exact content with `openmax --approve {}` (an in-session write approval also counts)",
                            path.display()
                        ))
                    } else {
                        match missing_command_reason(&h.command, project_root) {
                            Some(reason) => Status::Warn(reason),
                            None => Status::Ok(format!("hook on {}", h.event.as_str())),
                        }
                    }
                }
                Err(reason) => {
                    hook_events.push(None);
                    Status::Err(reason)
                }
            };
            hooks_found.push((Finding { kind: "hook", path, status }, id));
        }
    }
    // First stem wins, and a shadowed file is never loaded: the runtime does
    // not fail closed on one, so neither does this.
    mark_shadowed(&mut hooks_found, true);
    for event in crate::hooks::HookEvent::ALL {
        let event = event.as_str();
        mark_beyond_cap(
            &mut hooks_found,
            crate::hooks::MAX_HOOKS_PER_EVENT,
            |i| hook_events.get(i).copied().flatten() == Some(event),
            &format!("{event} hook cap"),
        );
    }
    findings.extend(hooks_found.into_iter().map(|(f, _)| f));
    findings.extend(hook_extras);

    for path in crate::permissions::permission_files(project_root) {
        let Some(result) = crate::permissions::check_file(&path) else { continue };
        match result {
            Ok(rule_tools) => {
                for (i, tool) in rule_tools.iter().enumerate() {
                    if let Some(reason) = unknown_tool_reason(
                        tool,
                        &known_tools,
                        &path,
                        project_root,
                        &format!("rule {} never matches", i + 1),
                    ) {
                        findings.push(Finding {
                            kind: "permissions",
                            path: path.clone(),
                            status: Status::Warn(reason),
                        });
                    }
                }
                findings.push(Finding {
                    kind: "permissions",
                    path,
                    status: Status::Ok(format!("{} rules", rule_tools.len())),
                });
            }
            Err(reason) => {
                findings.push(Finding { kind: "permissions", path, status: Status::Err(reason) })
            }
        }
    }

    let path = crate::providers::providers_path(data_dir);
    if let Some(result) = crate::providers::check_file(&path) {
        findings.push(Finding {
            kind: "providers",
            path,
            status: match result {
                Ok(n) => Status::Ok(format!("{n} providers")),
                Err(reason) => Status::Err(reason),
            },
        });
    }

    findings.extend(unread_paths(project_root));
    findings
}

/// Every tool name this project can actually call: the built-ins plus every
/// external tool that parsed, in either tier.
fn known_tool_names(external: &[String]) -> Vec<String> {
    let mut names: Vec<String> = tools::TOOL_NAMES.iter().map(|s| s.to_string()).collect();
    names.extend(external.iter().cloned());
    names
}

/// Why a `tool` field naming `tool` will never match, if it will not.
///
/// Rules are matched by exact name, so a typo is silent: a deny that never
/// fires reads exactly like a deny that never had to. Only project files are
/// checked. A global file is shared across projects and may legitimately name
/// a tool defined in a different one, and a project file may legitimately name
/// a tool the agent has not written yet, which is why this warns.
fn unknown_tool_reason(
    tool: &str,
    known: &[String],
    path: &Path,
    project_root: &Path,
    consequence: &str,
) -> Option<String> {
    if !path.starts_with(project_root) || known.iter().any(|k| k == tool) {
        return None;
    }
    let hint = known
        .iter()
        .find(|k| near(k, tool))
        .map(|k| format!(", did you mean '{k}'"))
        .unwrap_or_default();
    Some(format!(
        "no tool named '{tool}' exists in this project, so {consequence}{hint}"
    ))
}

/// Why `command` will not spawn from this checkout, if it will not. A path
/// (contains '/') resolves against the project root, exactly as the runtime
/// spawns it; a bare name resolves on PATH. This warns rather than errors:
/// check-time and run-time environments legitimately differ (CI without the
/// tool installed, a script the agent writes next).
fn missing_command_reason(command: &str, project_root: &Path) -> Option<String> {
    let command = command.trim();
    if command.is_empty() {
        return None; // the parser already errors on this
    }
    if command.contains('/') {
        let path = if Path::new(command).is_absolute() {
            PathBuf::from(command)
        } else {
            project_root.join(command)
        };
        if !path.is_file() {
            return Some(format!("command '{command}' does not exist from the project root"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let executable = std::fs::metadata(&path)
                .map(|m| m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false);
            if !executable {
                return Some(format!("command '{command}' exists but is not executable"));
            }
        }
        return None;
    }
    let found = std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| dir.join(command).is_file())
    });
    (!found).then(|| format!("command '{command}' is not on PATH"))
}

/// Mark files a loader never reaches because another file claims the same
/// identity. `first_wins` mirrors the loader: hooks keep the first file to
/// claim a stem, while tools, skills, and templates let a later directory
/// overwrite an earlier one. Both orders resolve to the project tier, since
/// each surface lists its directories so the project file is the winner.
fn mark_shadowed(entries: &mut [Entry], first_wins: bool) {
    let mut winner: HashMap<String, usize> = HashMap::new();
    for (i, (_, id)) in entries.iter().enumerate() {
        let Some(id) = id else { continue };
        if first_wins && winner.contains_key(id) {
            continue;
        }
        winner.insert(id.clone(), i);
    }
    let shadowed: Vec<(usize, String, PathBuf)> = entries
        .iter()
        .enumerate()
        .filter_map(|(i, (_, id))| {
            let id = id.as_ref()?;
            let w = *winner.get(id)?;
            (w != i).then(|| (i, id.clone(), entries[w].0.path.clone()))
        })
        .collect();
    for (i, id, winner_path) in shadowed {
        let kind = entries[i].0.kind;
        entries[i].0.status = Status::Warn(format!(
            "shadowed by {}, where {kind} '{id}' resolves",
            winner_path.display()
        ));
    }
}

/// Mark live (Ok, unshadowed) entries the loader's cap drops. The loader keeps
/// the identity-sorted head, so ranking live identities reproduces exactly
/// which files never load. `in_scope` restricts the ranking (hooks cap per
/// event); `what` names the cap in the message.
fn mark_beyond_cap(
    entries: &mut [Entry],
    cap: usize,
    in_scope: impl Fn(usize) -> bool,
    what: &str,
) {
    let mut live: Vec<(String, usize)> = entries
        .iter()
        .enumerate()
        .filter(|(i, (f, id))| {
            id.is_some() && matches!(f.status, Status::Ok(_)) && in_scope(*i)
        })
        .map(|(i, (_, id))| (id.clone().unwrap(), i))
        .collect();
    live.sort();
    for (id, i) in live.into_iter().skip(cap) {
        entries[i].0.status = Status::Err(format!(
            "'{id}' is beyond the {cap}-file {what} and never loads: consolidate or delete files"
        ));
    }
}

/// Directories the project tier is actually read from, as (parent, child).
const PROJECT_DIRS: &[(&str, &str)] = &[
    (".openmax", "tools"),
    (".openmax", "hooks"),
    (".agents", "skills"),
    (".agents", "prompts"),
];

/// Files that legitimately sit directly under a project config directory.
const LOOSE_FILES: &[(&str, &str)] = &[(".openmax", "permissions.toml")];

/// Config-shaped extensions that suggest a file was meant to be read. Scripts
/// are deliberately absent: a tool TOML normally sits beside the program it
/// runs, and flagging that would be noise.
const CONFIG_EXTENSIONS: &[&str] = &["yaml", "yml", "json", "ini", "cfg", "conf", "txt"];

/// Extension files sitting where nothing reads them. Without this, a project
/// whose every extension is misplaced reports a clean bill of health, which is
/// the most misleading answer `--check` can give.
fn unread_paths(project_root: &Path) -> Vec<Finding> {
    let mut out = Vec::new();
    for parent_name in [".openmax", ".agents"] {
        let parent = project_root.join(parent_name);
        let Ok(rd) = std::fs::read_dir(&parent) else { continue };
        let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        entries.sort();
        for path in entries {
            let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
                continue;
            };
            let name = name.as_str();
            let canonical = PROJECT_DIRS.iter().any(|(p, c)| *p == parent_name && *c == name);
            if path.is_file() {
                if !LOOSE_FILES.iter().any(|(p, f)| *p == parent_name && *f == name)
                    && path.extension().is_some_and(|e| e == "toml" || e == "md")
                {
                    out.push(Finding {
                        kind: "path",
                        path,
                        status: Status::Warn(format!(
                            "nothing reads files directly in {parent_name}/; \
                             see openmax --spec for where each surface lives"
                        )),
                    });
                }
                continue;
            }
            if canonical {
                out.extend(wrong_extension_files(&path, parent_name, name));
                continue;
            }
            if dir_is_empty(&path) {
                continue;
            }
            // A real directory of the right name under the wrong parent is the
            // likeliest mistake, so it gets the exact path instead of a guess.
            if let Some((other, _)) = PROJECT_DIRS.iter().find(|(p, c)| *c == name && *p != parent_name)
            {
                out.push(Finding {
                    kind: "path",
                    path,
                    status: Status::Warn(format!("not read; {name} live in {other}/{name}/")),
                });
            } else if let Some((_, c)) =
                PROJECT_DIRS.iter().find(|(p, c)| *p == parent_name && near(c, name))
            {
                out.push(Finding {
                    kind: "path",
                    path,
                    status: Status::Warn(format!("not read; did you mean {parent_name}/{c}/")),
                });
            }
        }
    }
    out
}

/// Files in a directory that is read, but with an extension that is not.
fn wrong_extension_files(dir: &Path, parent: &str, child: &str) -> Vec<Finding> {
    let want = match (parent, child) {
        (".openmax", "tools") | (".openmax", "hooks") => "toml",
        (".agents", "prompts") => "md",
        _ => return Vec::new(),
    };
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    paths.sort();
    paths
        .into_iter()
        .filter(|p| p.is_file())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e != want && CONFIG_EXTENSIONS.contains(&e))
        })
        .map(|path| Finding {
            kind: "path",
            path,
            status: Status::Warn(format!("not read; {parent}/{child}/ is read as .{want} only")),
        })
        .collect()
}

/// A SKILL.md that only differs by case, which no filesystem guarantees to
/// resolve the same way.
fn miscased_skill_file(dir: &Path) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    rd.flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n != "SKILL.md" && n.eq_ignore_ascii_case("SKILL.md"))
        })
}

fn dir_is_empty(dir: &Path) -> bool {
    std::fs::read_dir(dir).map(|mut rd| rd.next().is_none()).unwrap_or(true)
}

/// One edit apart, counting an adjacent swap as one edit: a dropped plural, a
/// doubled letter, a single mistyped or transposed character.
fn near(a: &str, b: &str) -> bool {
    if a == b {
        return false;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let (long, short) = if a.len() >= b.len() { (&a, &b) } else { (&b, &a) };
    if long.len() - short.len() > 1 {
        return false;
    }
    let (mut i, mut j, mut edited) = (0, 0, false);
    while i < long.len() && j < short.len() {
        if long[i] == short[j] {
            i += 1;
            j += 1;
            continue;
        }
        if edited {
            return false;
        }
        edited = true;
        if long.len() == short.len() {
            if i + 1 < long.len() && long[i] == short[j + 1] && long[i + 1] == short[j] {
                i += 2;
                j += 2;
                continue;
            }
            i += 1;
            j += 1;
        } else {
            i += 1;
        }
    }
    true
}

fn files_with_extension(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == ext) && p.is_file())
        .collect();
    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_project() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omx-doctor-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: PathBuf, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    fn tool_toml(name: &str) -> String {
        format!("name = \"{name}\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n")
    }

    /// Findings for paths under this project only: the developer's real global
    /// extensions must not leak into an assertion.
    fn local(root: &Path) -> Vec<Finding> {
        let root = root.to_string_lossy().to_string();
        check(Path::new(&root))
            .into_iter()
            .filter(|f| f.path.to_string_lossy().starts_with(&root))
            .collect()
    }

    fn find<'a>(findings: &'a [Finding], needle: &str) -> &'a Finding {
        findings
            .iter()
            .find(|f| f.path.to_string_lossy().contains(needle))
            .unwrap_or_else(|| panic!("no finding for {needle}"))
    }

    #[test]
    fn reports_valid_invalid_and_shadowing_files() {
        let root = temp_project();
        write(root.join(".openmax/tools/good.toml"), &tool_toml("deploy"));
        write(root.join(".openmax/tools/broken.toml"), "name = [not toml");
        write(root.join(".openmax/tools/shadow.toml"), &tool_toml("bash"));
        write(
            root.join(".agents/skills/good/SKILL.md"),
            "---\nname: good\ndescription: fine\n---\nbody\n",
        );
        write(root.join(".agents/prompts/review.md"), "Review the diff.\n");
        write(
            root.join(".openmax/hooks/bad-event.toml"),
            "event = \"on_fire\"\ncommand = \"/bin/sh\"\n",
        );
        write(root.join(".openmax/permissions.toml"), "[[rule]]\n");

        let findings = local(&root);
        assert!(has_errors(&findings));
        assert!(matches!(find(&findings, "good.toml").status, Status::Ok(_)));
        assert!(find(&findings, "good.toml").status.summary().contains("deploy"));
        assert!(find(&findings, "broken.toml").status.summary().contains("invalid TOML"));
        assert!(find(&findings, "shadow.toml").status.summary().contains("shadows a built-in"));
        assert!(matches!(find(&findings, "SKILL.md").status, Status::Ok(_)));
        assert!(find(&findings, "review.md").status.summary().contains("/review"));
        assert!(find(&findings, "bad-event.toml").status.summary().contains("unknown event"));
        assert!(
            matches!(find(&findings, "permissions.toml").status, Status::Err(ref r) if r.contains("malformed")),
            "typo'd [[rule]] must be reported as the fail-closed reason"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn empty_project_reports_nothing() {
        let root = temp_project();
        assert!(local(&root).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn files_beyond_loader_caps_are_errors_not_silence() {
        let root = temp_project();
        let data = temp_project();
        let tools_dir = root.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        // /bin/sh exists everywhere: the command probe must stay quiet so the
        // cap ranking (which counts live entries) is what gets exercised.
        for i in 0..(crate::registry::MAX_EXTERNAL_TOOLS + 1) {
            std::fs::write(
                tools_dir.join(format!("tool-{i:03}.toml")),
                format!("name = \"tool-{i:03}\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n"),
            )
            .unwrap();
        }
        let hooks_dir = root.join(".openmax").join("hooks");
        std::fs::create_dir_all(&hooks_dir).unwrap();
        for i in 0..(crate::hooks::MAX_HOOKS_PER_EVENT + 1) {
            std::fs::write(
                hooks_dir.join(format!("hook-{i:03}.toml")),
                "event = \"post_tool_use\"\ncommand = \"/bin/sh\"\n",
            )
            .unwrap();
        }
        // Approve the (identical) hook content so the cap ranking, not the
        // approval gate, is what this test exercises.
        crate::ledger::approve_hash(
            &data,
            &root,
            &crate::ledger::sha256_hex(b"event = \"post_tool_use\"\ncommand = \"/bin/sh\"\n"),
        )
        .unwrap();

        let findings = check_at(&root, &data);
        let over_tool = findings
            .iter()
            .find(|f| f.kind == "tool" && f.path.ends_with(format!(
                "tool-{:03}.toml",
                crate::registry::MAX_EXTERNAL_TOOLS
            )))
            .unwrap();
        assert!(
            over_tool.status.summary().contains("never loads"),
            "{}",
            over_tool.status.summary()
        );
        let over_hook = findings
            .iter()
            .find(|f| f.kind == "hook" && f.path.ends_with(format!(
                "hook-{:03}.toml",
                crate::hooks::MAX_HOOKS_PER_EVENT
            )))
            .unwrap();
        assert!(
            over_hook.status.summary().contains("never loads"),
            "{}",
            over_hook.status.summary()
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn missing_tool_command_warns_instead_of_looking_healthy() {
        let root = temp_project();
        let data = temp_project();
        let tools_dir = root.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        std::fs::write(
            tools_dir.join("ghost.toml"),
            "name = \"ghost\"\ndescription = \"d\"\ncommand = \"./scripts/does-not-exist.sh\"\n",
        )
        .unwrap();
        std::fs::write(
            tools_dir.join("real.toml"),
            "name = \"real\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n",
        )
        .unwrap();

        let findings = check_at(&root, &data);
        let ghost = findings.iter().find(|f| f.path.ends_with("ghost.toml")).unwrap();
        assert!(matches!(ghost.status, Status::Warn(_)), "{:?}", ghost.status);
        assert!(ghost.status.summary().contains("does not exist"), "{}", ghost.status.summary());
        let real = findings.iter().find(|f| f.path.ends_with("real.toml")).unwrap();
        assert!(matches!(real.status, Status::Ok(_)), "{:?}", real.status);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn reports_global_provider_configuration() {
        let root = temp_project();
        let data = temp_project();
        std::fs::write(
            data.join("providers.json"),
            r#"{"providers":{"local":{"base_url":"http://127.0.0.1:11434/v1"}}}"#,
        )
        .unwrap();

        let findings = check_at(&root, &data);
        let provider = findings.iter().find(|f| f.kind == "providers").unwrap();
        assert_eq!(provider.status.summary(), "1 providers");

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// The report a project gets when every extension it wrote is at a path
    /// nothing reads. This used to be "no extension files found", exit 0.
    #[test]
    fn misplaced_files_are_reported_instead_of_looking_healthy() {
        let root = temp_project();
        write(root.join(".openmax/tool/deploy.toml"), &tool_toml("deploy"));
        write(root.join(".openmax/skills/helper/SKILL.md"), "---\nname: h\n---\n");
        write(root.join(".agents/prompt/review.md"), "Review.\n");
        write(root.join(".openmax/deploy.toml"), &tool_toml("stray"));

        let findings = local(&root);
        assert!(!findings.is_empty(), "a project of misplaced files must not look empty");
        assert!(!has_errors(&findings), "a guess about intent must not fail the run");
        assert!(has_warnings(&findings));

        assert!(find(&findings, ".openmax/tool").status.summary().contains(".openmax/tools/"));
        assert!(
            find(&findings, ".openmax/skills").status.summary().contains(".agents/skills/"),
            "skills live under .agents, and saying so beats guessing"
        );
        assert!(find(&findings, ".agents/prompt").status.summary().contains(".agents/prompts/"));
        assert!(find(&findings, ".openmax/deploy.toml").status.summary().contains("directly in"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_empty_misspelled_directory_is_not_worth_reporting() {
        let root = temp_project();
        std::fs::create_dir_all(root.join(".openmax/tool")).unwrap();
        assert!(local(&root).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_wrong_extension_is_reported_but_a_sibling_script_is_not() {
        let root = temp_project();
        write(root.join(".openmax/tools/deploy.toml"), &tool_toml("deploy"));
        write(root.join(".openmax/tools/deploy.sh"), "#!/bin/sh\n");
        write(root.join(".openmax/tools/other.yaml"), "name: other\n");

        let findings = local(&root);
        assert!(find(&findings, "other.yaml").status.summary().contains(".toml only"));
        assert!(
            !findings.iter().any(|f| f.path.to_string_lossy().contains("deploy.sh")),
            "a tool TOML sits beside the program it runs; flagging that is noise"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_skill_directory_without_a_skill_md_is_reported() {
        let root = temp_project();
        write(root.join(".agents/skills/helper/notes.md"), "not a skill\n");
        let findings = local(&root);
        assert!(find(&findings, "helper").status.summary().contains("SKILL.md is not spelled"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_rule_naming_a_tool_that_does_not_exist_is_reported() {
        let root = temp_project();
        write(root.join(".openmax/tools/deploy.toml"), &tool_toml("deploy"));
        write(
            root.join(".openmax/permissions.toml"),
            "[[rules]]\neffect = \"deny\"\ntool = \"bash\"\n\n\
             [[rules]]\neffect = \"deny\"\ntool = \"bahs\"\n\n\
             [[rules]]\neffect = \"allow\"\ntool = \"deploy\"\n",
        );

        let findings = local(&root);
        let warn = findings
            .iter()
            .find(|f| f.kind == "permissions" && matches!(f.status, Status::Warn(_)))
            .expect("a rule that can never match must be reported");
        assert!(warn.status.summary().contains("bahs"), "{}", warn.status.summary());
        assert!(warn.status.summary().contains("rule 2"), "the rule must be locatable");
        assert!(warn.status.summary().contains("did you mean 'bash'"));
        assert!(
            !has_errors(&findings),
            "the tool may simply not be written yet, so this cannot fail the run"
        );
        // The built-in and the project's own tool are both known.
        assert_eq!(
            findings
                .iter()
                .filter(|f| f.kind == "permissions" && matches!(f.status, Status::Warn(_)))
                .count(),
            1
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_hook_filtering_on_an_unknown_tool_is_reported() {
        let root = temp_project();
        write(
            root.join(".openmax/hooks/gate.toml"),
            "event = \"pre_tool_use\"\ncommand = \"/bin/sh\"\ntool = \"write-file\"\n",
        );
        let findings = local(&root);
        let warn = findings
            .iter()
            .find(|f| f.kind == "hook" && matches!(f.status, Status::Warn(_)))
            .expect("a filter that never matches must be reported");
        assert!(warn.status.summary().contains("write-file"));
        assert!(warn.status.summary().contains("did you mean 'write_file'"));
        let _ = std::fs::remove_dir_all(root);
    }

    fn entry(kind: &'static str, path: &str, status: Status, id: &str) -> Entry {
        (Finding { kind, path: PathBuf::from(path), status }, Some(id.to_string()))
    }

    /// A file nothing loads cannot fail closed, so it must not fail the check
    /// either: `--check` and the runtime have to agree about what is live.
    /// Hooks keep the first file to claim a stem, whatever state it is in.
    #[test]
    fn a_shadowed_hook_is_inert_even_when_it_is_broken() {
        let mut hooks = vec![
            entry("hook", "/project/.openmax/hooks/gate.toml", Status::Ok("hook on pre_tool_use".into()), "gate"),
            entry("hook", "/home/.openmax/hooks/gate.toml", Status::Err("unknown event 'on_fire'".into()), "gate"),
        ];
        mark_shadowed(&mut hooks, true);

        assert!(matches!(hooks[0].0.status, Status::Ok(_)), "the project file wins");
        assert!(
            matches!(hooks[1].0.status, Status::Warn(_)),
            "a broken file the loop never reads must not fail the run"
        );
        assert!(hooks[1].0.status.summary().contains("/project/.openmax/hooks/gate.toml"));
    }

    /// Tools, skills, and templates let a later directory overwrite an earlier
    /// one, so the winner is the last file to claim the name, not the first.
    #[test]
    fn a_later_definition_wins_for_overwriting_surfaces() {
        let mut tools = vec![
            entry("tool", "/home/.openmax/tools/deploy.toml", Status::Ok("tool 'deploy'".into()), "deploy"),
            entry("tool", "/project/.openmax/tools/deploy.toml", Status::Ok("tool 'deploy'".into()), "deploy"),
        ];
        mark_shadowed(&mut tools, false);

        assert!(matches!(tools[0].0.status, Status::Warn(_)));
        assert!(tools[0].0.status.summary().contains("/project/.openmax/tools/deploy.toml"));
        assert!(matches!(tools[1].0.status, Status::Ok(_)), "the project file wins");
    }

    #[test]
    fn near_matches_one_edit_and_nothing_further() {
        assert!(near("tools", "tool"));
        assert!(near("hooks", "hook"));
        assert!(near("prompts", "promts"));
        assert!(near("bash", "bahs"), "an adjacent swap is one typo");
        assert!(near("write_file", "write-file"));
        assert!(!near("tools", "tools"));
        assert!(!near("tools", "skills"));
        assert!(!near("write_file", "read_file"));
    }
}
