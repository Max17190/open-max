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
    let mut tool_meta: Vec<(String, PathBuf)> = Vec::new();
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
                    tool_meta.push((spec.name.clone(), path.clone()));
                    let external = match &spec.kind {
                        crate::registry::ToolKind::External(ext) => Some(ext.clone()),
                        crate::registry::ToolKind::Builtin => None,
                    };
                    let missing = external
                        .as_ref()
                        .and_then(|ext| missing_command_reason(&ext.command, project_root));
                    match (missing, external) {
                        (Some(reason), _) => Status::Warn(reason),
                        (None, Some(ext)) => match stale_code_reason(
                            data_dir,
                            project_root,
                            &ext.source_sha256,
                            &ext.command,
                            &ext.args,
                        ) {
                            // The tool still runs; it asks again first, which
                            // is the point. Silence here is what let a swapped
                            // script read as a healthy tool.
                            Some(changed) => Status::Warn(format!(
                                "{changed}, so the next call asks for approval again"
                            )),
                            None => Status::Ok(format!("tool '{}'", spec.name)),
                        },
                        (None, None) => Status::Ok(format!("tool '{}'", spec.name)),
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
    let mut skill_meta: Vec<(String, String, PathBuf)> = Vec::new();
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
                    skill_meta.push((s.name.clone(), s.description.clone(), path.clone()));
                    Status::Ok(format!("skill '{}'", s.name))
                }
                Err(reason) => Status::Err(reason),
            };
            skills_found.push((Finding { kind: "skill", path, status }, id));
        }
    }
    mark_shadowed(&mut skills_found, false);
    mark_beyond_cap(&mut skills_found, crate::skills::MAX_SKILLS, |_| true, "skill cap");
    // The index byte cap drops whole lines from the frozen prompt: a skill
    // past it parses fine, but the model never sees its name, so nothing can
    // ever invoke it. Reproduce the exact accounting the prompt uses, and
    // name each dropped skill rather than let it read as healthy.
    let indexed = crate::skills::discover(project_root);
    for (name, chars) in crate::prompt::skill_index_costs(project_root, &indexed) {
        if chars > 0 {
            continue;
        }
        for (f, id) in skills_found.iter_mut() {
            if id.as_deref() == Some(name.as_str()) && matches!(f.status, Status::Ok(_)) {
                f.status = Status::Warn(format!(
                    "'{name}' parses but is not in the frozen skills index: the {}-byte index budget fills before it, so the model cannot see or invoke it; shorten earlier descriptions or delete skills",
                    crate::prompt::MAX_SKILLS_BYTES
                ));
            }
        }
    }
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
                    // Hooks run with host authority and no per-call gate, so
                    // content no human approved - the file or the code it runs
                    // - is inert, and this is where that stops being silent.
                    let unapproved = stale_code_reason(
                        data_dir,
                        project_root,
                        &h.source_sha256,
                        &h.command,
                        &h.args,
                    )
                    .or_else(|| {
                        (!crate::ledger::is_approved(data_dir, project_root, &h.source_sha256))
                            .then(|| "its content is not approved".to_string())
                    });
                    match unapproved {
                        Some(mut reason) => {
                            let approvals = crate::ledger::approvals(data_dir, project_root).ok();
                            let was_live =
                                approvals.as_ref().is_some_and(|a| a.was_live(&path));
                            let approved =
                                approvals.as_ref().and_then(|a| a.approved_hook(&path));
                            // The loop classifies a modified hook by the event
                            // a human approved, so --check has to ask the same
                            // question or it would report a demoted gate as a
                            // harmless observer.
                            let was_gate = approved.map(|a| a.is_gate()).unwrap_or(true);
                            if let Some(approved) = approved {
                                if approved.is_gate() && !h.event.is_gate() {
                                    reason = format!(
                                        "an approved {} gate was rewritten as a {} hook, which would stop it gating",
                                        approved.event(),
                                        h.event.as_str()
                                    );
                                }
                            }
                            if was_live && was_gate {
                                Status::Err(format!(
                                    "{reason}; this gate was live, so every tool call fails closed until the approved content is restored or a human re-approves it: `openmax --approve {}`",
                                    path.display()
                                ))
                            } else {
                                Status::Err(format!(
                                    "inert because {reason}: a human must approve this exact content with `openmax --approve {}` (an in-session write approval also counts)",
                                    path.display()
                                ))
                            }
                        }
                        None => match missing_command_reason(&h.command, project_root) {
                            Some(reason) => Status::Warn(reason),
                            None => Status::Ok(format!("hook on {}", h.event.as_str())),
                        },
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
    // A deleted hook file leaves nothing on disk to report against, so the
    // approved paths are the source of truth for what should be there. This
    // is the same reconciliation the loop fails closed on; --check is where a
    // human finds out which file it means.
    if let Ok(approvals) = crate::ledger::approvals(data_dir, project_root) {
        for path in approvals.live_paths() {
            if path.exists() || !crate::hooks::is_hook_manifest(path, project_root) {
                continue;
            }
            findings.push(Finding {
                kind: "hook",
                path: path.clone(),
                status: Status::Err(format!(
                    "an approved hook file was deleted; every tool call fails closed until it is restored or retired with `openmax --forget {}`",
                    path.display()
                )),
            });
        }
    }

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

    // An inherited approval store explains, in one line, why a capability that
    // worked yesterday now asks or fails closed. Without it the hook findings
    // above read as "its content changed" for content nobody touched.
    if let Some(pending) = crate::ledger::pending_legacy(data_dir, project_root) {
        let status = if pending.malformed {
            Status::Err(
                "an inherited approval store that does not parse: nothing in it is in effect, and it cannot be adopted until it is fixed or deleted".into(),
            )
        } else {
            Status::Err(format!(
                "an approval store inherited from an older release: {} hash(es) and {} remembered hook shape(s) are NOT in effect, so approvals here ask again and a capability it says was installed fails closed. `openmax --adopt-approvals` inherits it after showing you what it claims; deleting the file discards it",
                pending.hashes, pending.shapes
            ))
        };
        findings.push(Finding { kind: "approvals", path: pending.path, status });
    }

    findings.extend(inline_program_findings(project_root));
    findings.extend(memory_findings(project_root));
    findings.extend(unread_paths(project_root));
    findings.extend(hygiene_findings(project_root, data_dir, &tool_meta, &skill_meta));
    findings
}

/// Warn where approval's reach ends: a manifest that hands an interpreter a
/// program on the command line binds that text, but not the project file the
/// text opens while it runs. Only flagged when the inline program actually
/// names a file that exists here - a warning that fired on every `sh -c` would
/// teach authors to skip warnings.
fn inline_program_findings(project_root: &Path) -> Vec<Finding> {
    let mut out = Vec::new();
    let mut warn = |kind: &'static str, path: PathBuf, command: &str, args: &[String]| {
        if let Some(read) = crate::ledger::inline_program_read(command, args, project_root) {
            let named = read.strip_prefix(project_root).unwrap_or(&read).display().to_string();
            out.push(Finding {
                kind,
                path,
                status: Status::Warn(format!(
                    "its inline program reads {named} at runtime, and approval does not cover that file: only this manifest's text is bound. move the program into {named} and name it in `args` so its bytes are approved too"
                )),
            });
        }
    };
    for dir in crate::registry::external_tool_dirs(project_root) {
        for path in files_with_extension(&dir, "toml") {
            if let Ok(spec) = crate::registry::parse_tool_file(&path) {
                if let crate::registry::ToolKind::External(ext) = &spec.kind {
                    warn("tool", path.clone(), &ext.command, &ext.args);
                }
            }
        }
    }
    for dir in crate::hooks::hook_dirs(project_root) {
        for path in files_with_extension(&dir, "toml") {
            if let Ok(hook) = crate::hooks::parse_hook_file(&path) {
                warn("hook", path.clone(), &hook.command, &hook.args);
            }
        }
    }
    out
}

/// Memory files are data, not capabilities, so nothing here is an Err: a file
/// the index ignores is a Warn naming the reason and the fix, and a live one
/// is an Ok saying whether the index currently shows it. The check answers
/// the question the agent actually has after writing a memory: will a future
/// session see this?
fn memory_findings(project_root: &Path) -> Vec<Finding> {
    let mut findings = Vec::new();
    let dir = project_root.join(crate::memory::MEMORY_DIR);
    let Ok(read_dir) = std::fs::read_dir(&dir) else {
        return findings;
    };
    let scan = crate::memory::scan(project_root, crate::memory::unix_now());
    for entry in read_dir.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default().to_string();
        if path.is_dir() || name.starts_with('.') {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
        if let Some(memory) = scan.entries.iter().find(|e| e.name == stem) {
            let visibility = if memory.in_index {
                "in the session index".to_string()
            } else {
                "faded from the index (unused; a read_file revives it)".to_string()
            };
            findings.push(Finding {
                kind: "memory",
                path,
                status: Status::Ok(format!("memory '{stem}' — {visibility}")),
            });
            continue;
        }
        let reason = if path.extension().and_then(|e| e.to_str()) != Some("md") {
            "only .md files are memories; this file is never indexed".to_string()
        } else if !stem.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            || stem.is_empty()
            || stem.len() > 64
        {
            "memory names are 1-64 chars of [a-z0-9-]; rename it to be indexed".to_string()
        } else {
            "no describable first line; give it one non-empty line to be indexed".to_string()
        };
        findings.push(Finding { kind: "memory", path, status: Status::Warn(reason) });
    }
    findings
}

/// One declared example and what running it proved. `path` is the tool file
/// that declared it, so a refusal names the file to fix or approve.
#[derive(Debug)]
pub struct ExampleVerdict {
    pub tool: String,
    pub path: PathBuf,
    pub result: Result<(), String>,
}

/// Newline-separated stack of project roots whose examples are already
/// running. Every process an example spawns inherits it, so a tool whose
/// example runs `openmax --check --run-examples` (its cwd is the project
/// root, so it re-finds itself) refuses at the second level instead of
/// forking a third.
const RUN_EXAMPLES_STACK: &str = "OPENMAX_RUN_EXAMPLES";

/// Characters of a failing example's output kept in its verdict.
const EXAMPLE_DETAIL_CHARS: usize = 2_000;

/// The gates a turn applies to a tool call, resolved once for the whole run.
/// Nothing here can prompt, so a gate that would put the call in front of a
/// person refuses it instead.
struct ExampleGates {
    hooks: crate::hooks::Hooks,
    permissions: crate::permissions::Permissions,
    approval_mode: crate::config::ApprovalMode,
    /// True when the agent loop, not a person, started this process.
    agent_spawned: bool,
}

impl ExampleGates {
    /// The documented call order (hooks pre → permissions → approval_mode →
    /// execute), evaluated with the same predicates the agent loop calls.
    ///
    /// Every decision is matched exhaustively rather than tested for the one
    /// variant that refuses: a gate whose default is "run it" turns the next
    /// variant someone adds into a silent bypass.
    async fn admit(
        &self,
        spec: &crate::registry::ToolSpec,
        ext: &crate::registry::ExternalTool,
        args: &serde_json::Value,
        project_root: &Path,
        data_dir: &Path,
        cancel: &std::sync::Arc<crate::state::CancelToken>,
    ) -> Result<(), String> {
        use crate::config::ApprovalMode;
        use crate::hooks::PreToolResult;
        use crate::permissions::PermissionDecision;

        match self
            .hooks
            .pre_tool_use("check", &spec.name, args, project_root, cancel)
            .await
        {
            PreToolResult::Allow => {}
            PreToolResult::Block { reason } => {
                return Err(format!("blocked by pre_tool_use: {reason}"))
            }
            PreToolResult::Cancelled => return Err("cancelled".into()),
        }
        match self.permissions.evaluate(&spec.name, args) {
            PermissionDecision::Deny { reason } => return Err(reason),
            // A rule that singles this tool out for a prompt cannot be honored
            // in a batch, and a rule the user wrote for one tool is too
            // specific to answer with "a human typed the command".
            PermissionDecision::Ask => {
                return Err(
                    "permission rule requires human approval of this tool; examples cannot prompt (change the rule to effect = \"allow\" to run it unattended)"
                        .into(),
                )
            }
            // Allow is a user's "do not put this one in front of me" for the
            // prompt a turn would raise. It deliberately does not answer the
            // question below - whether a person is attached to this process -
            // because permissions.toml is a file the agent can write for
            // itself; trust and the content approval are the human decisions.
            PermissionDecision::Allow | PermissionDecision::Default => {}
        }
        // One exhaustive read of approval_mode, in the turn's precedence:
        // readonly is a hard block, ask needs a person, auto needs neither.
        let needs_person = if spec.mutating {
            match self.approval_mode {
                ApprovalMode::Readonly => {
                    return Err("approval_mode is readonly; mutating tools are disabled".into())
                }
                ApprovalMode::Ask => true,
                ApprovalMode::Auto => false,
            }
        } else {
            false
        };
        // Running an example runs the file's command with host authority and
        // no human on the other end of a prompt, so the exact bytes must be
        // approved first. Content-bound, exactly like the in-session gate: any
        // edit to the tool file revokes the approval.
        if !crate::ledger::is_approved(data_dir, project_root, &ext.source_sha256) {
            return Err(format!(
                "unapproved source; run openmax --approve {}",
                ext.source_path.display()
            ));
        }
        // A turn in `ask` mode puts every mutating call in front of a person.
        // The human who typed this command is that person; an agent-spawned
        // process has nobody, so it refuses rather than running unattended.
        if needs_person && self.agent_spawned {
            return Err(
                "approval_mode is ask and this process was started from an agent session; ask the user to run openmax --check --run-examples"
                    .into(),
            );
        }
        Ok(())
    }
}

/// `--check --run-examples`: prove each tool with an `[example]` actually
/// runs, through the exact spawn path a session uses (stdin JSON, timeout,
/// output caps) and behind the exact gates a session applies. `Err` means
/// nothing ran at all. This executes project commands: opt-in per invocation,
/// never part of plain `--check`.
pub async fn run_examples(
    project_root: &Path,
    report: impl FnMut(&ExampleVerdict),
) -> Result<Vec<ExampleVerdict>, String> {
    run_examples_at(project_root, &crate::state::default_data_dir(), report).await
}

pub(crate) async fn run_examples_at(
    project_root: &Path,
    data_dir: &Path,
    report: impl FnMut(&ExampleVerdict),
) -> Result<Vec<ExampleVerdict>, String> {
    let stack = std::env::var(RUN_EXAMPLES_STACK).unwrap_or_default();
    run_examples_within(project_root, data_dir, &stack, report).await
}

/// `stack` is taken as an argument rather than read here so the recursion
/// guard is testable without racing other tests over one process-wide var.
async fn run_examples_within(
    project_root: &Path,
    data_dir: &Path,
    stack: &str,
    mut report: impl FnMut(&ExampleVerdict),
) -> Result<Vec<ExampleVerdict>, String> {
    use crate::registry::{Registry, ToolKind};

    let root_key = std::fs::canonicalize(project_root)
        .unwrap_or_else(|_| project_root.to_path_buf())
        .display()
        .to_string();
    if stack.lines().any(|root| root == root_key) {
        return Err(format!(
            "examples for {root_key} are already running in a parent process; an example must not invoke openmax --run-examples"
        ));
    }
    // Examples execute repository code with the user's authority, exactly like
    // a turn does, so they need the same trust decision. Plain --check only
    // reads files and stays trust-free.
    if !crate::trust::is_trusted(data_dir, project_root)? {
        return Err(format!(
            "project {} is not trusted; inspect it, then rerun with --trust-project",
            project_root.display()
        ));
    }
    // Fail closed like a session start: a malformed settings file is a
    // configuration error, not a silent reset to defaults (and it carries the
    // approval_mode this run has to honor).
    let settings = crate::config::load(data_dir)?;
    let caps = crate::tools::OutputCaps::from_settings(&settings);
    let gates = ExampleGates {
        hooks: crate::hooks::Hooks::discover(project_root, data_dir),
        permissions: crate::permissions::Permissions::discover(project_root),
        approval_mode: settings.approval_mode,
        agent_spawned: std::env::var_os("OPENMAX_SESSION").is_some(),
    };

    let registry = Registry::build(project_root);
    let cancel = std::sync::Arc::new(crate::state::CancelToken::default());
    watch_cancel_signals(cancel.clone());
    // Published before the first spawn so every descendant inherits it, and
    // restored when this level finishes: the marker names the levels that are
    // running, and children captured it at spawn time.
    let outer = std::env::var_os(RUN_EXAMPLES_STACK);
    std::env::set_var(
        RUN_EXAMPLES_STACK,
        match stack.is_empty() {
            true => root_key,
            false => format!("{stack}\n{root_key}"),
        },
    );

    let mut results = Vec::new();
    for spec in &registry.tools {
        let ToolKind::External(ext) = &spec.kind else { continue };
        let Some(example) = &ext.example else { continue };
        let result = match gates
            .admit(spec, ext, &example.args, project_root, data_dir, &cancel)
            .await
        {
            Ok(()) => {
                let outcome = registry
                    .execute(&spec.name, &example.args, project_root, caps, cancel.clone())
                    .await;
                example_verdict(&outcome, example)
            }
            Err(reason) => Err(reason),
        };
        let verdict = ExampleVerdict {
            tool: spec.name.clone(),
            path: ext.source_path.clone(),
            result,
        };
        report(&verdict);
        results.push(verdict);
        // A signal cancelled the run: the in-flight example's process group is
        // already terminated and nothing new may start.
        if cancel.is_cancelled() {
            break;
        }
    }
    match outer {
        Some(value) => std::env::set_var(RUN_EXAMPLES_STACK, value),
        None => std::env::remove_var(RUN_EXAMPLES_STACK),
    }
    Ok(results)
}

fn example_verdict(
    outcome: &crate::tools::ToolOutcome,
    example: &crate::registry::ToolExample,
) -> Result<(), String> {
    if !outcome.ok {
        return Err(format!("example run failed; got:\n{}", detail(&outcome.output)));
    }
    let Some(pattern) = &example.expect_regex else {
        return Ok(());
    };
    // Tool parsing rejects an invalid expect_regex, so a mismatch is the only
    // failure reachable here.
    if regex::Regex::new(pattern).is_ok_and(|re| re.is_match(&outcome.output)) {
        return Ok(());
    }
    Err(format!(
        "want expect_regex {pattern:?}, got:\n{}",
        detail(&outcome.output)
    ))
}

/// The tail of a failing example's output, indented under its verdict line.
/// A nonzero exit renders as `exit code N` followed by the output, so a
/// first-line summary is structurally guaranteed to drop the diagnostic.
fn detail(output: &str) -> String {
    let text = output.trim_end();
    if text.is_empty() {
        return "        (no output)".to_string();
    }
    let start = text
        .char_indices()
        .rev()
        .take(EXAMPLE_DETAIL_CHARS)
        .last()
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut lines = Vec::new();
    if start > 0 {
        lines.push("        ...".to_string());
    }
    lines.extend(text[start..].lines().map(|line| format!("        {line}")));
    lines.join("\n")
}

/// A signal killing the checker must not orphan the example's process tree:
/// `kill_on_drop` never runs when the parent dies from a signal, and every
/// child is in its own process group. Cancelling instead routes the kill
/// through the normal termination path, which signals the whole group.
#[cfg(unix)]
fn watch_cancel_signals(cancel: std::sync::Arc<crate::state::CancelToken>) {
    use tokio::signal::unix::{signal, SignalKind};
    let (Ok(mut interrupt), Ok(mut terminate)) =
        (signal(SignalKind::interrupt()), signal(SignalKind::terminate()))
    else {
        return;
    };
    tokio::spawn(async move {
        let mut fired = false;
        loop {
            tokio::select! {
                _ = interrupt.recv() => {}
                _ = terminate.recv() => {}
            }
            if fired {
                // The operator insisting: stop waiting for the group to die.
                std::process::exit(130);
            }
            fired = true;
            cancel.cancel();
        }
    });
}

#[cfg(not(unix))]
fn watch_cancel_signals(_: std::sync::Arc<crate::state::CancelToken>) {}

/// Skill-library hygiene: extensions the usage record says are pure prompt
/// tax, and near-duplicate skill descriptions that shadow each other in the
/// index. Warnings only - the agent (or human) judges and deletes; nothing
/// is pruned automatically.
fn hygiene_findings(
    project_root: &Path,
    data_dir: &Path,
    tools: &[(String, PathBuf)],
    skills: &[(String, String, PathBuf)],
) -> Vec<Finding> {
    let mut findings = Vec::new();
    /// Enough recorded calls to judge non-use as signal rather than youth.
    const MIN_SIGNAL_CALLS: u64 = 50;
    if let Ok(usage) = crate::ledger::load_usage(data_dir, project_root) {
        if usage.total_calls >= MIN_SIGNAL_CALLS {
            for (name, path) in tools {
                if usage.tools.get(name).map(|e| e.calls).unwrap_or(0) == 0 {
                    findings.push(Finding {
                        kind: "tool",
                        path: path.clone(),
                        status: Status::Warn(format!(
                            "'{name}' was never called across {} recorded calls; its schema is pure prompt tax - consider deleting it",
                            usage.total_calls
                        )),
                    });
                }
            }
            for (name, _, path) in skills {
                if usage.skills.get(name).map(|e| e.calls).unwrap_or(0) == 0 {
                    findings.push(Finding {
                        kind: "skill",
                        path: path.clone(),
                        status: Status::Warn(format!(
                            "'{name}' was never read across {} recorded calls; consider deleting or merging it",
                            usage.total_calls
                        )),
                    });
                }
            }
        }
    }
    // Near-duplicate descriptions make the model pick between look-alikes
    // (skill shadowing); flag the later name of each pair.
    for (i, (name_a, desc_a, _)) in skills.iter().enumerate() {
        for (name_b, desc_b, path_b) in skills.iter().skip(i + 1) {
            if name_a == name_b {
                continue; // cross-tier shadowing is reported elsewhere
            }
            if description_similarity(desc_a, desc_b) > 0.6 {
                findings.push(Finding {
                    kind: "skill",
                    path: path_b.clone(),
                    status: Status::Warn(format!(
                        "'{name_b}' describes nearly the same thing as '{name_a}'; look-alike skills degrade selection - consider merging"
                    )),
                });
            }
        }
    }
    findings
}

/// Word-set Jaccard similarity, lowercase: cheap, deterministic, no model.
fn description_similarity(a: &str, b: &str) -> f64 {
    let words = |s: &str| -> std::collections::HashSet<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(str::to_string)
            .collect()
    };
    let (a, b) = (words(a), words(b));
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(&b).count() as f64;
    let union = a.union(&b).count() as f64;
    intersection / union
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

/// Why the code this manifest runs is not the code a human approved, if it is
/// not. Only meaningful once the manifest itself is approved: before that the
/// whole definition is unapproved, which each surface reports its own way.
fn stale_code_reason(
    data_dir: &Path,
    project_root: &Path,
    manifest_sha: &str,
    command: &str,
    args: &[String],
) -> Option<String> {
    let approvals = crate::ledger::approvals(data_dir, project_root).ok()?;
    if !approvals.contains(manifest_sha) {
        return None;
    }
    let problem = crate::ledger::bound_code(command, args, project_root)
        .iter()
        .find_map(|c| c.problem(&approvals))?;
    Some(format!("the code it runs, {problem}"))
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

    /// A capability whose script was rewritten after approval must not read
    /// as healthy: `--check` is where a human looks to find out whether what
    /// they blessed is what is installed.
    #[test]
    fn a_swapped_script_is_reported_for_both_surfaces() {
        let root = temp_project();
        let data = root.join("data");
        write(
            root.join(".openmax/tools/deploy.toml"),
            "name = \"deploy\"\ndescription = \"d\"\ncommand = \"./deploy.sh\"\nmutating = true\n",
        );
        write(
            root.join(".openmax/hooks/gate.toml"),
            "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n",
        );
        for script in ["deploy.sh", "gate.sh"] {
            let path = root.join(script);
            std::fs::write(&path, "#!/bin/sh\ntrue\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
        }
        let approve = |rel: &str| {
            let manifest = root.join(rel);
            let mut shas =
                vec![crate::ledger::sha256_hex(&std::fs::read(&manifest).unwrap())];
            shas.extend(
                crate::ledger::manifest_code(&manifest, &root)
                    .into_iter()
                    .filter_map(|c| c.sha256),
            );
            crate::ledger::approve_capability(&data, &root, &manifest, &shas).unwrap();
        };
        approve(".openmax/tools/deploy.toml");
        approve(".openmax/hooks/gate.toml");

        let findings: Vec<Finding> = check_at(&root, &data)
            .into_iter()
            .filter(|f| f.path.starts_with(&root))
            .collect();
        assert!(matches!(find(&findings, "deploy.toml").status, Status::Ok(_)));
        assert!(matches!(find(&findings, "gate.toml").status, Status::Ok(_)));

        std::fs::write(root.join("deploy.sh"), "#!/bin/sh\necho PWNED\n").unwrap();
        std::fs::write(root.join("gate.sh"), "#!/bin/sh\nexit 0\n").unwrap();
        let findings: Vec<Finding> = check_at(&root, &data)
            .into_iter()
            .filter(|f| f.path.starts_with(&root))
            .collect();
        // The tool still runs, but it asks again first, so this is a warning.
        match &find(&findings, "deploy.toml").status {
            Status::Warn(reason) => {
                assert!(reason.contains("deploy.sh"), "{reason}");
                assert!(reason.contains("asks for approval again"), "{reason}");
            }
            other => panic!("expected a warning about the swapped script: {other:?}"),
        }
        // The gate was live and is not: that is an error, and it fails closed.
        match &find(&findings, "gate.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("gate.sh"), "{reason}");
                assert!(reason.contains("fails closed"), "{reason}");
            }
            other => panic!("expected an error about the revoked gate: {other:?}"),
        }
        assert!(has_errors(&findings));

        // A gate rewritten into an observe hook is still judged as the gate a
        // human approved, and --check has to say which of the two happened.
        write(
            root.join(".openmax/hooks/gate.toml"),
            "event = \"post_tool_use\"\ncommand = \"./gate.sh\"\n",
        );
        let findings: Vec<Finding> = check_at(&root, &data)
            .into_iter()
            .filter(|f| f.path.starts_with(&root))
            .collect();
        match &find(&findings, "gate.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("rewritten as a post_tool_use hook"), "{reason}");
                assert!(reason.contains("fails closed"), "{reason}");
            }
            other => panic!("a demoted gate must not read as an inert observer: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// A deleted hook file has nothing on disk to report against, so it is
    /// found by reconciling the approved paths. Without that, `--check` says
    /// "no extension files found" about a project whose gate was removed.
    #[test]
    fn a_deleted_approved_hook_is_reported_and_can_be_retired() {
        let root = temp_project();
        let data = root.join("data");
        let hook = root.join(".openmax/hooks/gate.toml");
        // A command that really exists on every platform the harness runs on:
        // one that does not is now uncovered on purpose, which is a different
        // finding from the one under test.
        write(hook.clone(), "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n");
        let sha = crate::ledger::sha256_hex(&std::fs::read(&hook).unwrap());
        crate::ledger::approve_capability(&data, &root, &hook, &[sha]).unwrap();
        assert!(!has_errors(&check_at(&root, &data)), "{:?}", check_at(&root, &data));

        std::fs::remove_file(&hook).unwrap();
        let findings = check_at(&root, &data);
        match &find(&findings, "gate.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("deleted"), "{reason}");
                assert!(reason.contains("--forget"), "the way out must be named: {reason}");
            }
            other => panic!("a deleted approved hook must be an error: {other:?}"),
        }

        // Retiring it is the human saying the removal was intended.
        assert!(crate::ledger::forget_capability(&data, &root, &hook).unwrap());
        assert!(!has_errors(&check_at(&root, &data)));
        assert!(
            !crate::ledger::forget_capability(&data, &root, &hook).unwrap(),
            "forgetting twice reports that nothing was recorded"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A skill the model cannot see must not read as healthy: past the count
    /// cap it never loads (an error, like tools), and past the index byte cap
    /// the model cannot name it to invoke it (a warning naming the budget
    /// that filled first). Before this, both read as `ok` while the model
    /// silently lacked them.
    #[test]
    fn skills_past_either_cap_are_named() {
        let root = temp_project();
        let total = crate::skills::MAX_SKILLS + 2;
        for i in 0..total {
            // `aaa-` sorts ahead of any global skill a developer machine may
            // have, so the cap ranking below is deterministic everywhere.
            write(
                root.join(".agents/skills").join(format!("s{i:03}")).join("SKILL.md"),
                &format!("---\nname: aaa-skill-{i:03}\ndescription: {}\n---\nbody\n", "d".repeat(120)),
            );
        }
        let findings = local(&root);
        match &find(&findings, "s000/SKILL.md").status {
            Status::Ok(_) => {}
            other => panic!("a carried skill stays healthy: {other:?}"),
        }
        match &find(&findings, "s025/SKILL.md").status {
            Status::Warn(reason) => {
                assert!(reason.contains("frozen skills index"), "{reason}");
                assert!(reason.contains("cannot see or invoke"), "{reason}");
            }
            other => panic!("a byte-capped skill must warn: {other:?}"),
        }
        match &find(&findings, &format!("s{:03}/SKILL.md", total - 1)).status {
            Status::Err(reason) => assert!(reason.contains("skill cap"), "{reason}"),
            other => panic!("a count-capped skill must be an error: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
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

    /// Write one tool file and return its path.
    fn tool_file(root: &Path, file: &str, body: &str) -> PathBuf {
        let path = root.join(".openmax").join("tools").join(file);
        write(path.clone(), body);
        path
    }

    /// The state a human establishes before any example may run: the project
    /// is trusted and the named tool files are approved by exact content.
    fn approved_data_dir(root: &Path, tools: &[&Path]) -> PathBuf {
        let data = temp_project();
        crate::trust::trust_project(&data, root).unwrap();
        for tool in tools {
            let bytes = std::fs::read(tool).unwrap();
            crate::ledger::approve_hash(&data, root, &crate::ledger::sha256_hex(&bytes)).unwrap();
        }
        data
    }

    async fn examples(root: &Path, data: &Path) -> Result<Vec<ExampleVerdict>, String> {
        run_examples_at(root, data, |_| {}).await
    }

    fn verdict<'a>(results: &'a [ExampleVerdict], tool: &str) -> &'a ExampleVerdict {
        results
            .iter()
            .find(|v| v.tool == tool)
            .unwrap_or_else(|| panic!("no verdict for {tool}"))
    }

    #[tokio::test]
    async fn run_examples_proves_tools_through_the_real_spawn_path() {
        let root = temp_project();
        // Echoes its stdin JSON back: the example asserts the payload arrived.
        let echoer = tool_file(
            &root,
            "echoer.toml",
            "name = \"echoer\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"cat\"]\n\n[example]\nexpect_regex = \"hello\"\n[example.args]\nmsg = \"hello\"\n",
        );
        let broken = tool_file(
            &root,
            "broken.toml",
            "name = \"broken\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"exit 3\"]\n\n[example]\n",
        );
        let data = approved_data_dir(&root, &[&echoer, &broken]);

        let results = examples(&root, &data).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(verdict(&results, "echoer").result.is_ok());
        let failure = verdict(&results, "broken").result.as_ref().unwrap_err();
        assert!(failure.contains("example run failed"), "{failure}");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// An example is the tool's command with host authority, so the content
    /// gate that makes same-turn self-extension safe applies here too: an
    /// unapproved tool file must not spawn, and must say how to approve it.
    #[tokio::test]
    async fn an_unapproved_tool_file_never_runs_its_example() {
        let root = temp_project();
        let touched = root.join("side-effect");
        let approved = tool_file(
            &root,
            "approved.toml",
            "name = \"approved\"\ndescription = \"d\"\ncommand = \"/bin/echo\"\n\n[example]\n",
        );
        tool_file(
            &root,
            "pwn.toml",
            &format!(
                "name = \"pwn\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\nmutating = true\n\n[example]\n",
                touched.display()
            ),
        );
        let data = approved_data_dir(&root, &[&approved]);

        let results = examples(&root, &data).await.unwrap();
        assert!(verdict(&results, "approved").result.is_ok());
        let refusal = verdict(&results, "pwn").result.as_ref().unwrap_err();
        assert!(refusal.contains("unapproved source"), "{refusal}");
        assert!(refusal.contains("--approve"), "{refusal}");
        assert!(!touched.exists(), "the refused example must not have run");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// A cloned repository executes nothing under a validation flag: examples
    /// need the same trust decision a turn needs.
    #[tokio::test]
    async fn an_untrusted_project_runs_no_examples() {
        let root = temp_project();
        let touched = root.join("side-effect");
        let tool = tool_file(
            &root,
            "pwn.toml",
            &format!(
                "name = \"pwn\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\n\n[example]\n",
                touched.display()
            ),
        );
        // Approved content, but nobody trusted the project it came with.
        let data = temp_project();
        let bytes = std::fs::read(&tool).unwrap();
        crate::ledger::approve_hash(&data, &root, &crate::ledger::sha256_hex(&bytes)).unwrap();

        let refusal = examples(&root, &data).await.unwrap_err();
        assert!(refusal.contains("not trusted"), "{refusal}");
        assert!(refusal.contains("--trust-project"), "{refusal}");
        assert!(!touched.exists(), "nothing may run in an untrusted project");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// Permission rules and approval_mode bind examples exactly as they bind
    /// the same call inside a turn, including the malformed-file deny.
    #[tokio::test]
    async fn permissions_and_readonly_mode_refuse_examples() {
        let root = temp_project();
        let touched = root.join("side-effect");
        let tool = tool_file(
            &root,
            "writer.toml",
            &format!(
                "name = \"writer\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\nmutating = true\n\n[example]\n",
                touched.display()
            ),
        );
        let data = approved_data_dir(&root, &[&tool]);

        write(
            root.join(".openmax/permissions.toml"),
            "[[rules]]\neffect = \"deny\"\ntool = \"writer\"\n",
        );
        let denied = examples(&root, &data).await.unwrap();
        assert!(verdict(&denied, "writer").result.is_err());

        // A malformed policy denies every tool; an example is not an exception.
        write(root.join(".openmax/permissions.toml"), "not toml [[[\n");
        let failed_closed = examples(&root, &data).await.unwrap();
        let reason = verdict(&failed_closed, "writer").result.as_ref().unwrap_err();
        assert!(reason.contains("malformed"), "{reason}");

        std::fs::remove_file(root.join(".openmax/permissions.toml")).unwrap();
        std::fs::write(
            data.join("settings.json"),
            r#"{"approval_mode":"readonly"}"#,
        )
        .unwrap();
        let readonly = examples(&root, &data).await.unwrap();
        let reason = verdict(&readonly, "writer").result.as_ref().unwrap_err();
        assert!(reason.contains("readonly"), "{reason}");

        assert!(!touched.exists(), "no refused example may have run");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// `effect = "ask"` singles a tool out for a prompt. A batch run has
    /// nothing to prompt with, so it refuses - for a human-started run too,
    /// since the CLI cannot stop mid-report to ask. The escape hatch is the
    /// rule the user already knows how to write.
    #[tokio::test]
    async fn an_ask_permission_rule_refuses_the_example() {
        let root = temp_project();
        let touched = root.join("side-effect");
        let tool = tool_file(
            &root,
            "writer.toml",
            &format!(
                "name = \"writer\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\n\n[example]\n",
                touched.display()
            ),
        );
        let data = approved_data_dir(&root, &[&tool]);
        write(
            root.join(".openmax/permissions.toml"),
            "[[rules]]\neffect = \"ask\"\ntool = \"writer\"\n",
        );

        // This process carries no OPENMAX_SESSION, so it is the human case.
        let results = examples(&root, &data).await.unwrap();
        let reason = verdict(&results, "writer").result.as_ref().unwrap_err();
        assert!(reason.contains("cannot prompt"), "{reason}");
        assert!(reason.contains("effect = \"allow\""), "{reason}");
        assert!(!touched.exists(), "an ask rule must not reach a spawn");

        // The rule the message names does let it run.
        write(
            root.join(".openmax/permissions.toml"),
            "[[rules]]\neffect = \"allow\"\ntool = \"writer\"\n",
        );
        let allowed = examples(&root, &data).await.unwrap();
        assert!(verdict(&allowed, "writer").result.is_ok());
        assert!(touched.exists());
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// `auto` is the mode that runs mutating tools without a prompt, so an
    /// approved mutating example runs unattended there and nowhere else.
    #[tokio::test]
    async fn auto_mode_runs_an_approved_mutating_example() {
        let root = temp_project();
        let touched = root.join("side-effect");
        let tool = tool_file(
            &root,
            "writer.toml",
            &format!(
                "name = \"writer\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\nmutating = true\n\n[example]\n",
                touched.display()
            ),
        );
        let data = approved_data_dir(&root, &[&tool]);
        std::fs::write(data.join("settings.json"), r#"{"approval_mode":"auto"}"#).unwrap();

        let results = examples(&root, &data).await.unwrap();
        assert!(verdict(&results, "writer").result.is_ok());
        assert!(touched.exists());
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// An approved `pre_tool_use` gate is a gate for examples too.
    #[tokio::test]
    async fn a_pre_tool_use_hook_can_block_an_example() {
        let root = temp_project();
        let touched = root.join("side-effect");
        let tool = tool_file(
            &root,
            "writer.toml",
            &format!(
                "name = \"writer\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\n\n[example]\n",
                touched.display()
            ),
        );
        let hook = root.join(".openmax/hooks/deny.toml");
        write(
            hook.clone(),
            "event = \"pre_tool_use\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"echo blocked by policy >&2; exit 2\"]\n",
        );
        let data = approved_data_dir(&root, &[&tool, &hook]);

        let results = examples(&root, &data).await.unwrap();
        let reason = verdict(&results, "writer").result.as_ref().unwrap_err();
        assert!(reason.contains("pre_tool_use"), "{reason}");
        assert!(!touched.exists(), "a blocked example must not have run");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// The verdict has to carry the reason the tool failed. On a nonzero exit
    /// the first line is always `exit code N`, and a regex miss is useless
    /// without the output it was matched against.
    #[tokio::test]
    async fn a_failing_example_reports_the_diagnostic_and_what_was_wanted() {
        let root = temp_project();
        let failer = tool_file(
            &root,
            "failer.toml",
            "name = \"failer\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"echo 'ERROR: config.yaml missing on line 42' >&2; exit 3\"]\n\n[example]\n",
        );
        let misser = tool_file(
            &root,
            "misser.toml",
            "name = \"misser\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"echo actual-output\"]\n\n[example]\nexpect_regex = \"WANTED\"\n",
        );
        let data = approved_data_dir(&root, &[&failer, &misser]);

        let results = examples(&root, &data).await.unwrap();
        let failed = verdict(&results, "failer").result.as_ref().unwrap_err();
        assert!(failed.contains("exit code 3"), "{failed}");
        assert!(failed.contains("config.yaml missing on line 42"), "{failed}");
        let missed = verdict(&results, "misser").result.as_ref().unwrap_err();
        assert!(missed.contains("want expect_regex \"WANTED\""), "{missed}");
        assert!(missed.contains("actual-output"), "{missed}");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// A tool whose example runs the checker re-finds this project and forks a
    /// level deeper every time. The inherited stack stops it at the second.
    #[tokio::test]
    async fn an_example_run_inside_an_example_run_is_refused() {
        let root = temp_project();
        let touched = root.join("side-effect");
        let tool = tool_file(
            &root,
            "recurse.toml",
            &format!(
                "name = \"recurse\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"touch {}\"]\n\n[example]\n",
                touched.display()
            ),
        );
        let data = approved_data_dir(&root, &[&tool]);
        let stack = std::fs::canonicalize(&root).unwrap().display().to_string();

        let refusal = run_examples_within(&root, &data, &stack, |_| {})
            .await
            .unwrap_err();
        assert!(refusal.contains("already running"), "{refusal}");
        assert!(!touched.exists(), "the recursive level must not spawn");
        // An unrelated project on the stack is not this project.
        let other = format!("{}/elsewhere", stack);
        assert!(run_examples_within(&root, &data, &other, |_| {}).await.is_ok());
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn an_invalid_example_regex_is_a_parse_error() {
        let text = "name = \"t\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n\n[example]\nexpect_regex = \"(\"\n";
        let err = crate::registry::parse_tool_file_from_text_for_tests(text).unwrap_err();
        assert!(err.contains("expect_regex"), "{err}");
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
    fn hygiene_flags_unused_tools_and_lookalike_skills() {
        let root = temp_project();
        let data = temp_project();
        let tools_dir = root.join(".openmax").join("tools");
        std::fs::create_dir_all(&tools_dir).unwrap();
        std::fs::write(
            tools_dir.join("ghost.toml"),
            "name = \"ghost\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n",
        )
        .unwrap();
        for (name, desc) in [
            ("deploy-app", "Deploy the application to the production cluster safely"),
            ("ship-app", "Deploy the application to the production cluster with checks"),
        ] {
            let dir = root.join(".agents").join("skills").join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join("SKILL.md"),
                format!("---\nname: {name}\ndescription: {desc}\n---\nbody\n"),
            )
            .unwrap();
        }
        // Enough recorded signal to judge non-use.
        let mut delta = crate::ledger::UsageDelta::default();
        for _ in 0..50 {
            delta.tools.push(("something-else".into(), true));
        }
        crate::ledger::record_usage(&data, &root, &delta).unwrap();

        let findings = check_at(&root, &data);
        assert!(
            findings.iter().any(|f| f.path.ends_with("ghost.toml")
                && f.status.summary().contains("never called")),
            "unused tool must be flagged"
        );
        assert!(
            findings.iter().any(|f| matches!(f.status, Status::Warn(_))
                && f.status.summary().contains("nearly the same thing")),
            "look-alike skills must be flagged"
        );
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

    /// After writing a memory the agent's question is "will a future session
    /// see this?" - so every file in the directory gets an answer: indexed,
    /// faded, or ignored with the reason and the fix. Ignored files are
    /// warnings, never errors: memory is data, and nothing here may fail a
    /// check run.
    #[test]
    fn memory_files_report_index_visibility_and_ignore_reasons() {
        let root = temp_project();
        write(root.join(".openmax/memory/deploy-port.md"), "# port 7443\nbody");
        write(root.join(".openmax/memory/Bad_Name.md"), "# x");
        write(root.join(".openmax/memory/blank.md"), "\n\n");
        write(root.join(".openmax/memory/notes.txt"), "# not markdown");
        write(root.join(".openmax/memory/.access.jsonl"), "");

        let findings = local(&root);
        let live = find(&findings, "deploy-port.md");
        assert!(matches!(live.status, Status::Ok(_)), "{:?}", live.status);
        assert!(live.status.summary().contains("in the session index"));

        let bad = find(&findings, "Bad_Name.md");
        assert!(matches!(bad.status, Status::Warn(_)));
        assert!(bad.status.summary().contains("[a-z0-9-]"));

        let blank = find(&findings, "blank.md");
        assert!(matches!(blank.status, Status::Warn(_)));
        assert!(blank.status.summary().contains("first line"));

        let txt = find(&findings, "notes.txt");
        assert!(matches!(txt.status, Status::Warn(_)));
        assert!(txt.status.summary().contains(".md"));

        assert!(
            !findings.iter().any(|f| f.path.to_string_lossy().contains(".access.jsonl")),
            "the access log is not a memory and gets no finding"
        );
        assert!(!has_errors(&findings), "memory findings never fail a check");
        let _ = std::fs::remove_dir_all(root);
    }
}
