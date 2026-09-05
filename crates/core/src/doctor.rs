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
    let mut tool_meta: Vec<(String, PathBuf)> = Vec::new();
    // Deferred to after shadowing and the cap: a file the loader never reaches
    // has no runtime behavior to describe. Indices into `tools_found`.
    let mut tool_clamps: Vec<(usize, String)> = Vec::new();
    for dir in crate::registry::external_tool_dirs(data_dir, project_root) {
        for path in files_with_extension(&dir, "toml") {
            // One read per file. Every claim in this file's findings has to
            // come from one generation of its bytes: a second read for the
            // clamp diagnostic is an interval a rewrite can split, and the
            // report would then describe two files that never existed at once.
            let source = std::fs::read_to_string(&path);
            let parsed = match &source {
                Ok(text) => crate::registry::parse_tool_source(&path, text),
                Err(e) => Err(format!("unreadable: {e}")),
            };
            let mut id = None;
            let status = match parsed {
                Ok(spec) if tools::TOOL_NAMES.contains(&spec.name.as_str()) => {
                    Status::Err(format!("'{}' shadows a built-in tool and is ignored", spec.name))
                }
                Ok(spec) => {
                    id = Some(spec.name.clone());
                    tool_meta.push((spec.name.clone(), path.clone()));
                    if let Some(reason) = source.as_deref().ok().and_then(|text| {
                        clamped_timeout_reason(
                            text,
                            "tool",
                            crate::registry::MIN_TIMEOUT_SECS,
                            crate::registry::MAX_TIMEOUT_SECS,
                        )
                    }) {
                        tool_clamps.push((tools_found.len(), reason));
                    }
                    if let Some(reason) = source
                        .as_deref()
                        .ok()
                        .and_then(|text| tool_description_gap(text, &spec.name))
                    {
                        tool_clamps.push((tools_found.len(), reason));
                    }
                    let external = match &spec.kind {
                        crate::registry::ToolKind::External(ext) => Some(ext.clone()),
                        crate::registry::ToolKind::Builtin => None,
                    };
                    let missing = external.as_ref().and_then(|ext| {
                        missing_command_reason(&ext.command, project_root)
                            .or_else(|| missing_script_reason(&ext.command, &ext.args, project_root))
                    });
                    match (missing, external) {
                        (Some((_, reason)), _) => Status::Warn(reason),
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
                            // Loaded but never approved: the same state a
                            // hook reports as inert. A tool is not inert -
                            // its first call prompts - but "ok" read as
                            // "callable now" to weaker models, opposite
                            // the hook line for the same gate. Say so.
                            None if !crate::ledger::is_approved(
                                data_dir,
                                project_root,
                                &ext.source_sha256,
                            ) =>
                            {
                                Status::Warn(format!(
                                    "tool '{}' loads; no human has approved its content, so its \
                                     first call stops for approval (openmax --approve {}, or \
                                     approve the card it raises in a session; prove it first with \
                                     openmax --check --run-examples, which probes unapproved \
                                     tools in a sandbox)",
                                    spec.name,
                                    shell_quote(&ext.source_path)
                                ))
                            }
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
    let tool_shadows = mark_shadowed(&mut tools_found, false);
    let tool_capped = mark_beyond_cap(
        &mut tools_found,
        crate::registry::MAX_EXTERNAL_TOOLS,
        &tool_shadows,
        |_| true,
        "tool cap",
    );
    // Rules and hook filters resolve against what actually loads, so the
    // known-tool set is the entries the loader keeps: a rule naming a
    // beyond-cap tool is as dead as one naming a typo, but a warned entry is
    // still live - a tool whose command does not exist yet, or whose code
    // awaits re-approval, loads all the same, so a rule naming it matches.
    let external_names: Vec<String> = tools_found
        .iter()
        .filter(|(f, id)| id.is_some() && !matches!(f.status, Status::Err(_)))
        .filter_map(|(_, id)| id.clone())
        .collect();
    let tool_extras = clamp_findings("tool", &tools_found, tool_clamps, &tool_shadows, &tool_capped);
    findings.extend(tools_found.into_iter().map(|(f, _)| f));
    findings.extend(tool_extras);

    let mut skills_found: Vec<Entry> = Vec::new();
    let mut skill_meta: Vec<(String, String, PathBuf)> = Vec::new();
    // Deferred like the tool clamps, for the same reason: what the index line
    // fails to say only matters for a skill that has an index line at all.
    // Indices into `skills_found`.
    let mut skill_notes: Vec<(usize, String)> = Vec::new();
    for dir in crate::skills::skill_dirs(data_dir, project_root) {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        dirs.sort();
        for entry in dirs {
            if entry.is_file() {
                // A dotfile can never be a skill and no loader reads it, and
                // Finder drops .DS_Store into every directory it opens, so
                // warning about one would put permanent noise in every report.
                // Judged through the lossy rendering, not to_str: a hidden
                // name with invalid UTF-8 after the dot is still hidden.
                let hidden = entry
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'));
                if hidden {
                    continue;
                }
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
                    let status = if let Some(found) = miscased_skill_file(&entry) {
                        Status::Warn(format!(
                            "no skill loads from here: SKILL.md is not spelled exactly, but {} is",
                            found.file_name().unwrap_or_default().to_string_lossy()
                        ))
                    } else if let Some(deep) = nested_skill_md(&entry) {
                        // A correctly-spelled SKILL.md sits below, not at the
                        // top. The registry joins SKILL.md one level down only,
                        // so calling this a spelling mistake sends the author
                        // renaming a file that is already named right.
                        Status::Warn(format!(
                            "no skill loads from here: a skill's SKILL.md sits at the top of its \
                             directory, but the only one is deeper, at {}",
                            deep.display()
                        ))
                    } else {
                        Status::Warn(
                            "no skill loads from here: SKILL.md is not spelled exactly".into(),
                        )
                    };
                    findings.push(Finding { kind: "path", path: entry, status });
                }
                continue;
            }
            let mut id = None;
            // One read per file, like the tool loop: the description the spec
            // clamps and the description the author wrote have to come from
            // one generation of the bytes, or the report compares two files.
            let source = std::fs::read_to_string(&path);
            let parsed = match &source {
                Ok(text) => crate::skills::parse_skill_source(&path, text),
                Err(e) => Err(format!("unreadable: {e}")),
            };
            let status = match parsed {
                Ok(s) => {
                    id = Some(s.name.clone());
                    skill_meta.push((s.name.clone(), s.description.clone(), path.clone()));
                    if let Some(reason) = source
                        .as_deref()
                        .ok()
                        .and_then(|text| skill_description_gap(text, &s.name))
                    {
                        skill_notes.push((skills_found.len(), reason));
                    }
                    // The dir listing, not the resolved path: on a
                    // case-folding filesystem `SKILL.md` resolves to a file
                    // spelled otherwise and the skill loads here, so the
                    // miscase branch below - the only warning about it - can
                    // never fire on exactly the machines where the skill
                    // still works. It vanishes on the first case-sensitive
                    // checkout instead.
                    if let Some(found) = miscased_skill_file(&entry) {
                        let found =
                            found.file_name().unwrap_or_default().to_string_lossy().to_string();
                        let exact = std::fs::read_dir(&entry)
                            .ok()
                            .is_some_and(|rd| rd.flatten().any(|e| e.file_name() == "SKILL.md"));
                        let reason = if exact {
                            format!(
                                "'{}' also holds {found}, which is never read and collides with SKILL.md on a case-folding filesystem; keep exactly one",
                                s.name
                            )
                        } else {
                            format!(
                                "'{}' loads from {found} because this filesystem folds case; a case-sensitive checkout drops the skill silently, so rename it SKILL.md",
                                s.name
                            )
                        };
                        skill_notes.push((skills_found.len(), reason));
                    }
                    if let Some(reason) = agent_skills_name_issue(&s.name) {
                        skill_notes.push((skills_found.len(), reason));
                    }
                    Status::Ok(format!("skill '{}'", s.name))
                }
                Err(reason) => Status::Err(reason),
            };
            skills_found.push((Finding { kind: "skill", path, status }, id));
        }
    }
    let skill_shadows = mark_shadowed(&mut skills_found, false);
    mark_beyond_cap(&mut skills_found, crate::skills::MAX_SKILLS, &skill_shadows, |_| true, "skill cap");
    // The index byte cap drops whole lines from the frozen prompt: a skill
    // past it parses fine, but the model never sees its name, so nothing can
    // ever invoke it. Reproduce the exact accounting the prompt uses, and
    // name each dropped skill rather than let it read as healthy.
    let indexed = crate::skills::discover(data_dir, project_root);
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
    // Last word to the checks above: they decide whether the model ever sees
    // an index line for this skill, and these notes only describe one it does.
    for (i, reason) in skill_notes {
        let (finding, _) = &mut skills_found[i];
        if matches!(finding.status, Status::Ok(_)) {
            finding.status = Status::Warn(reason);
        }
    }
    findings.extend(skills_found.into_iter().map(|(f, _)| f));

    let mut templates_found: Vec<Entry> = Vec::new();
    // Deferred like the skill notes, for the same reason: what the popup line
    // fails to say only matters for a template whose file the loader keeps.
    let mut template_notes: Vec<(usize, String)> = Vec::new();
    for dir in crate::templates::template_dirs(data_dir, project_root) {
        for path in files_with_extension(&dir, "md") {
            let mut id = None;
            // One read per file, like the tool and skill loops: the clamp
            // diagnostic and the parse must describe one generation of bytes.
            let source = std::fs::read_to_string(&path);
            let parsed = match &source {
                Ok(text) => crate::templates::parse_template_source(&path, text),
                Err(e) => Err(format!("unreadable: {e}")),
            };
            let status = match parsed {
                Ok(t) => {
                    id = Some(t.name.clone());
                    if let Some(reason) = source
                        .as_deref()
                        .ok()
                        .and_then(|text| template_description_gap(text, &t.name))
                    {
                        template_notes.push((templates_found.len(), reason));
                    }
                    Status::Ok(format!("template /{}", t.name))
                }
                Err(reason) => Status::Err(reason),
            };
            templates_found.push((Finding { kind: "template", path, status }, id));
        }
    }
    mark_shadowed(&mut templates_found, false);
    // Last word to the shadowing above: a note only describes a file whose
    // line the popup will actually show.
    for (i, reason) in template_notes {
        let (finding, _) = &mut templates_found[i];
        if matches!(finding.status, Status::Ok(_)) {
            finding.status = Status::Warn(reason);
        }
    }
    findings.extend(templates_found.into_iter().map(|(f, _)| f));

    let known_tools = known_tool_names(&external_names);
    let mut hooks_found: Vec<Entry> = Vec::new();
    // Aligned with hooks_found: the parsed event of each Ok entry.
    let mut hook_events: Vec<Option<&'static str>> = Vec::new();
    let mut hook_extras: Vec<Finding> = Vec::new();
    // Indices into `hooks_found`, emitted only for the entries that load.
    let mut hook_clamps: Vec<(usize, String)> = Vec::new();
    for dir in crate::hooks::hook_dirs(project_root) {
        for path in files_with_extension(&dir, "toml") {
            // Hooks resolve by file stem, and the first stem to appear claims
            // it whether or not the file parses.
            let id = path.file_stem().and_then(|s| s.to_str()).map(|s| s.to_string());
            // One read, for the same reason as the tool loop above.
            let source = std::fs::read_to_string(&path);
            let parsed = match &source {
                Ok(text) => crate::hooks::parse_hook_source(&path, text),
                Err(e) => Err(format!("unreadable: {e}")),
            };
            let status = match parsed {
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
                    if let Some(mut reason) = source.as_deref().ok().and_then(|text| {
                        clamped_timeout_reason(
                            text,
                            "hook",
                            crate::hooks::MIN_TIMEOUT_SECS,
                            crate::hooks::MAX_TIMEOUT_SECS,
                        )
                    }) {
                        // A gate that times out blocks, so the clamp decides
                        // when this file starts refusing calls. Asked of the
                        // hook, not of its event: a blocking turn_end refuses
                        // on the same terms.
                        if h.gates() {
                            reason.push_str("; a gate that times out blocks");
                        }
                        hook_clamps.push((hooks_found.len(), reason));
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
                                if approved.is_gate() && !h.gates() {
                                    reason = format!(
                                        "an approved {} gate was rewritten as a {} hook, which would stop it gating",
                                        crate::hooks::shape_name(approved.event(), approved.blocking()),
                                        crate::hooks::shape_name(h.event.as_str(), h.blocking)
                                    );
                                }
                            }
                            // A live gate is judged on its own state first:
                            // the session is refusing every tool call, and
                            // `reason` already names the file that is missing
                            // when that is what revoked it.
                            if was_live && was_gate {
                                Status::Err(format!(
                                    "{reason}; this gate was live, so every tool call fails closed until the approved content is restored or a human re-approves it: `openmax --approve {}`",
                                    shell_quote(&path)
                                ))
                            } else if let Some((problem, missing)) =
                                missing_command_reason(&h.command, project_root).or_else(|| {
                                    missing_script_reason(&h.command, &h.args, project_root)
                                })
                            {
                                // Approval is not the next step when the
                                // command does not resolve: `openmax
                                // --approve` answers a manifest whose code it
                                // cannot read with that diagnosis instead. Each
                                // resolution failure gets the repair that fits
                                // it, since "create it" is wrong advice for a
                                // file that exists.
                                Status::Err(format!(
                                    "inert because {missing}: {}",
                                    problem.hook_repair(h.command.trim(), &path)
                                ))
                            } else {
                                // Only `openmax --approve`, from outside a
                                // session, activates a hook: the in-session
                                // write card shows a clipped preview, and a
                                // preview is not shown bytes. For a
                                // non-blocking turn_end the shape rides
                                // along: this is the line an author
                                // read before handing an observer to a human
                                // as a completion gate, three out of three
                                // weak-tier runs.
                                let shape_note = if h.event == crate::hooks::HookEvent::TurnEnd
                                    && !h.blocking
                                {
                                    ". as written it observes only: exit status is ignored at turn end, and `blocking = true` is what gates completion"
                                } else {
                                    ""
                                };
                                Status::Err(format!(
                                    "inert because {reason}: a human must approve this exact content with `openmax --approve {}`, run outside a session (an in-session write approval approves the write and nothing more){shape_note}",
                                    shell_quote(&path)
                                ))
                            }
                        }
                        None => match missing_command_reason(&h.command, project_root)
                            .or_else(|| missing_script_reason(&h.command, &h.args, project_root))
                        {
                            Some((_, reason)) => Status::Warn(reason),
                            // The shape, not just the event: `turn_end` alone
                            // says nothing about whether this file can end a
                            // turn, and that is the whole of what it does. For
                            // the one event with two shapes the ok line says
                            // which one loaded: all three weak-tier authors in
                            // an author wrote an observer while stating the
                            // harness would block on nonzero exit, and this
                            // line is where they looked.
                            None => {
                                if h.event == crate::hooks::HookEvent::TurnEnd && !h.blocking {
                                    Status::Ok(
                                        "hook on turn_end (observer: exit status is ignored; `blocking = true` gates completion)"
                                            .to_string(),
                                    )
                                } else {
                                    Status::Ok(format!(
                                        "hook on {}",
                                        crate::hooks::shape_name(h.event.as_str(), h.blocking)
                                    ))
                                }
                            }
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
    let hook_shadows = mark_shadowed(&mut hooks_found, true);
    let mut hook_capped = std::collections::HashSet::new();
    for event in crate::hooks::HookEvent::ALL {
        let event = event.as_str();
        hook_capped.extend(mark_beyond_cap(
            &mut hooks_found,
            crate::hooks::MAX_HOOKS_PER_EVENT,
            &hook_shadows,
            |i| hook_events.get(i).copied().flatten() == Some(event),
            &format!("{event} hook cap"),
        ));
    }
    // Both of the above are ordinary bookkeeping for a tool or a skill: the
    // file does not load, and nothing else changes. For a gate a human
    // approved it is the loop failing every tool call closed, so --check has
    // to say that rather than leave a human reading "never loads" while their
    // session refuses to run.
    if let Ok(approvals) = crate::ledger::approvals(data_dir, project_root) {
        mark_displaced_gates(&mut hooks_found, |path| {
            approvals
                .approved_hook(path)
                .map(|a| a.is_gate())
                .unwrap_or_else(|| approvals.was_live(path))
        });
        // An unparseable file that held its stem and was approved once is the
        // loop failing every tool call closed (`invalid` in hooks.rs, which
        // blocks whatever the approved shape was, because broken bytes name
        // no event): its parse error alone reads as bookkeeping while the
        // session refuses to run. After shadow marking, because a broken file
        // an earlier valid stem shadows never runs and must not claim to be
        // blocking anything.
        for (i, (finding, _)) in hooks_found.iter_mut().enumerate() {
            if hook_shadows.contains(&i)
                || hook_capped.contains(&i)
                || hook_events.get(i).copied().flatten().is_some()
                || !approvals.was_live(&finding.path)
            {
                continue;
            }
            if let Status::Err(reason) = &finding.status {
                finding.status = Status::Err(format!(
                    "{reason}; this file was live, so every tool call fails closed until the approved content is restored or a human re-approves it: `openmax --approve {}`",
                    shell_quote(&finding.path)
                ));
            }
        }
    }
    hook_extras.extend(clamp_findings(
        "hook",
        &hooks_found,
        hook_clamps,
        &hook_shadows,
        &hook_capped,
    ));
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
                    shell_quote(path)
                )),
            });
        }
    }

    for path in crate::permissions::permission_files(project_root) {
        let Some(result) = crate::permissions::check_file(&path, project_root, data_dir) else {
            continue;
        };
        match result {
            Ok((rule_tools, inert_verdict)) => {
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
                // An inert allow reads as a healthy rule count from the file
                // alone: the rule is well-formed, it just is not authority.
                // The summary row counts the inert ones too - it prints after
                // the warn, so "(N rules)" alone would close the file's story
                // claiming every rule is in force (reproduced).
                // The verdict came from the SAME read as the rule list.
                let mut inert = 0;
                if let Some((reason, dropped)) = inert_verdict {
                    inert = dropped;
                    findings.push(Finding {
                        kind: "permissions",
                        path: path.clone(),
                        status: Status::Warn(reason),
                    });
                }
                let summary = if inert > 0 {
                    format!("{} rules, {inert} inert until approved", rule_tools.len())
                } else {
                    format!("{} rules", rule_tools.len())
                };
                findings.push(Finding {
                    kind: "permissions",
                    path,
                    status: Status::Ok(summary),
                });
            }
            Err(reason) => {
                findings.push(Finding { kind: "permissions", path, status: Status::Err(reason) })
            }
        }
    }

    let path = crate::providers::providers_path(data_dir);
    if let Some(result) = crate::providers::check_file(&path) {
        match result {
            Ok((n, unknown_keys)) => {
                // A typo'd key deserializes cleanly and configures nothing,
                // which is exactly the silence this command exists to break.
                for reason in unknown_keys {
                    findings.push(Finding {
                        kind: "providers",
                        path: path.clone(),
                        status: Status::Warn(reason),
                    });
                }
                findings.push(Finding {
                    kind: "providers",
                    path,
                    status: Status::Ok(format!("{n} providers")),
                });
            }
            Err(reason) => {
                findings.push(Finding { kind: "providers", path, status: Status::Err(reason) })
            }
        }
    }

    // settings.json is launch-read and fail-closed: a file that will refuse
    // the next launch (exit 2) must be named by the one command whose job is
    // to say what is wrong before a launch finds out. The drift receipt tells
    // the agent to fix it; this is how the agent (or a human) confirms it did.
    let settings_path = crate::config::settings_path(data_dir);
    if settings_path.exists() {
        match crate::config::load(data_dir) {
            // A file that parses can still refuse every turn: no endpoint, no
            // model, or no context window resolves. Say so here, where the
            // reader is looking, rather than at the first prompt.
            Ok(settings) => findings.push(Finding {
                kind: "settings",
                path: settings_path,
                status: match crate::providers::resolve(&settings, data_dir) {
                    Ok(endpoint) => Status::Ok(format!(
                        "model {} via {}, {}-token window; read at launch, not hot (openmax --spec settings)",
                        endpoint.model,
                        endpoint.provider.as_deref().unwrap_or("base_url"),
                        endpoint.context_tokens,
                    )),
                    Err(reason) => Status::Warn(format!(
                        "parses, but every turn will refuse until this resolves: {reason}; read at launch, not hot (openmax --spec settings)"
                    )),
                },
            }),
            Err(reason) => findings.push(Finding {
                kind: "settings",
                path: settings_path,
                status: Status::Err(format!(
                    "{reason}; the next launch will refuse to start (exit 2) until this parses"
                )),
            }),
        }
    }

    findings.extend(inline_program_findings(data_dir, project_root));
    findings.extend(memory_findings(project_root));
    findings.extend(unread_paths(project_root));
    findings.extend(global_unread_paths(data_dir));
    findings.extend(hygiene_findings(project_root, data_dir, &tool_meta, &skill_meta));
    findings
}

/// Warn where approval's reach ends: a manifest that hands an interpreter a
/// program on the command line binds that text, but not the project file the
/// text opens while it runs. Only flagged when the inline program actually
/// names a file that exists here - a warning that fired on every `sh -c` would
/// teach authors to skip warnings.
/// Wrap a path for a copyable shell command. Several `--check` diagnostics
/// print `openmax --approve <path>` for a human to paste, and the path comes
/// from a file the agent named: a hook called `gate$(cmd).toml` is a legal
/// file the agent can create, approve once, then break, and an unquoted path
/// in the pasted repair command would run `$(cmd)` instead of naming the
/// file. POSIX single-quoting neutralizes every metacharacter, including an
/// embedded single quote (closed, escaped, reopened).
pub fn shell_quote(path: &std::path::Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

fn inline_program_findings(data_dir: &Path, project_root: &Path) -> Vec<Finding> {
    let mut out = Vec::new();
    let mut warn = |kind: &'static str, path: PathBuf, command: &str, args: &[String]| {
        if let Some(read) = crate::ledger::inline_program_read(command, args, project_root) {
            let named = read.strip_prefix(project_root).unwrap_or(&read).display().to_string();
            out.push(Finding {
                kind,
                path,
                status: Status::Warn(format!(
                    "its inline program reads {named} at runtime, and approval does not cover that file: only this manifest's text is bound. put the program in a project file and name it in `args` so its bytes are approved too"
                )),
            });
        }
    };
    for dir in crate::registry::external_tool_dirs(data_dir, project_root) {
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
/// is an Ok saying whether the index currently shows it (a Warn when it shows
/// it clipped). The check answers the question the agent actually has after
/// A directory entry is hidden when its name begins with a dot byte. Uses
/// to_string_lossy, not to_str, so a name whose bytes after the dot are not
/// valid UTF-8 is still recognized as hidden rather than treated as visible,
/// matching the skill scan's `.DS_Store` exemption.
fn name_is_hidden(name: &std::ffi::OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}

/// Any `.md` file at any depth under `dir`. A memory subdirectory that holds
/// one is notes the flat scan will never index, which is worth naming; an
/// empty or note-free subdir is left alone as it costs nothing.
fn dir_has_md(dir: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    for entry in rd.flatten() {
        let path = entry.path();
        // Skip hidden entries the way the skill scan skips `.DS_Store`: a
        // markdown file inside an editor's `.obsidian/` cache or a nested
        // `.git` is not a memory note the author lost.
        if path.file_name().is_some_and(name_is_hidden) {
            continue;
        }
        if path.is_dir() {
            if dir_has_md(&path) {
                return true;
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            return true;
        }
    }
    false
}

/// writing a memory: will a future session see this, as written?
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
        if path.is_dir() {
            // Memory is flat: the scan indexes only top-level `.md` files, so
            // notes tucked into a subfolder are silently never read. Name that
            // rather than pass it in a clean bill of health. A dotfile dir (a
            // stray `.git`, an editor cache) is noise and stays skipped.
            if !name.starts_with('.') && dir_has_md(&path) {
                findings.push(Finding {
                    kind: "memory",
                    path,
                    status: Status::Warn(format!(
                        "memory is flat: .md notes under {name}/ are never indexed; \
                         move them directly into .openmax/memory/"
                    )),
                });
            }
            continue;
        }
        if name.starts_with('.') {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
        if let Some(memory) = scan.entries.iter().find(|e| e.name == stem) {
            let visibility = if memory.in_index {
                "indexed at the next prompt freeze (session start, /reload, or any refreeze); a running session sees it once its prefix rebuilds".to_string()
            } else {
                "faded from the index (unused; a read_file revives it)".to_string()
            };
            // Indexed under a first line the index had to cut is still a
            // degraded index line: the future session it was written for reads
            // the clip, not the sentence, so the cut is worth a warning.
            let status = if memory.description_clipped() {
                Status::Warn(format!(
                    "memory '{stem}' is {visibility}, but its first line runs past the {}-char index cap: the line a future session reads is clipped there, so keep it a one-line summary and put the rest in the body",
                    crate::memory::MAX_DESCRIPTION_CHARS
                ))
            } else {
                Status::Ok(format!("memory '{stem}' — {visibility}"))
            };
            findings.push(Finding { kind: "memory", path, status });
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
    /// True when this run was a sandboxed probe of UNAPPROVED content: it
    /// executed with no network, writes confined to a scratch dir, and a
    /// scrubbed environment - and granted nothing. In-session calls still
    /// prompt until a human runs `openmax --approve`.
    pub sandboxed: bool,
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

/// How an example may run: with host authority (approved content behind the
/// full gate set), or as a sandboxed probe (unapproved content, no network,
/// writes confined, scrubbed env - zero host authority granted).
enum Admission {
    Host,
    Sandboxed,
}

/// A passing sandboxed probe leaves a receipt keyed on the manifest sha,
/// recording the FULL sha vector the probe ran (manifest + bound code).
/// Advisory evidence for the approval card, deliberately not a ledger
/// record: it grants nothing, and any edit to either half changes the
/// vector and orphans the receipt.
fn record_probe_receipt(data_dir: &Path, tool: &str, shas: &[String]) {
    let Some(manifest_sha) = shas.first() else { return };
    let dir = data_dir.join("probes");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let unix_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let receipt = serde_json::json!({
        "tool": tool,
        "shas": shas,
        "verdict": "example passed in sandbox (no network, writes confined)",
        "unix_time": unix_time,
    });
    // Atomic publish: a concurrent card lookup sees either the previous
    // complete receipt or this one, never a truncated document it would
    // read as "no evidence".
    let _ = crate::sessions::write_atomic(&dir.join(format!("{manifest_sha}.json")), receipt.to_string());
}

/// Whether a passing probe receipt exists for exactly this sha vector. Used
/// by the in-session approval card as evidence; a mismatch on any element
/// (edited manifest or script) reads as no receipt.
pub(crate) fn probe_passed(data_dir: &Path, shas: &[String]) -> bool {
    let Some(manifest_sha) = shas.first() else { return false };
    let path = data_dir.join("probes").join(format!("{manifest_sha}.json"));
    let Ok(text) = std::fs::read_to_string(&path) else { return false };
    let Ok(receipt) = serde_json::from_str::<serde_json::Value>(&text) else { return false };
    let recorded: Vec<&str> = receipt["shas"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    recorded == shas.iter().map(String::as_str).collect::<Vec<_>>()
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
    ) -> Result<Admission, String> {
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
        // Unapproved content probes in a sandbox instead of refusing. The
        // flat refusal protected nothing an agent could not already do with
        // bash - it only prevented the verified, receipt-producing path -
        // while forcing write -> approve -> fail -> edit -> re-approve loops
        // on the human. A probe grants zero host authority (no network,
        // writes confined to scratch, scrubbed env), so it is exempt from
        // the mode/person gates below, which exist to guard host authority;
        // the mode gates still guard every approved (host) run, unchanged.
        // The in-session content gate (unapproved_capability) is untouched:
        // a passing probe approves nothing.
        if !crate::ledger::is_approved(data_dir, project_root, &ext.source_sha256) {
            return Ok(Admission::Sandboxed);
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
        // A turn in `ask` mode puts every mutating call in front of a person.
        // The human who typed this command is that person; an agent-spawned
        // process has nobody, so it refuses rather than running unattended.
        if needs_person && self.agent_spawned {
            return Err(
                "approval_mode is ask and this process was started from an agent session; ask the user to run openmax --check --run-examples"
                    .into(),
            );
        }
        Ok(Admission::Host)
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
        permissions: crate::permissions::Permissions::discover(project_root, data_dir),
        approval_mode: settings.approval_mode,
        agent_spawned: std::env::var_os("OPENMAX_SESSION").is_some(),
    };

    let registry = Registry::build(data_dir, project_root);
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
        let admission = gates
            .admit(spec, ext, &example.args, project_root, data_dir, &cancel)
            .await;
        let mut sandboxed = false;
        let result = match admission {
            Ok(Admission::Host) => {
                let outcome = registry
                    .execute(&spec.name, &example.args, data_dir, project_root, caps, cancel.clone())
                    .await;
                example_verdict(&outcome, example)
            }
            Ok(Admission::Sandboxed) => {
                sandboxed = true;
                // The vector the probe is ABOUT to run: captured before the
                // spawn, so a script rewritten while the probe runs cannot
                // earn a passing receipt for bytes that were never probed.
                let mut probed_shas = vec![ext.source_sha256.clone()];
                probed_shas.extend(
                    crate::ledger::bound_code(&ext.command, &ext.args, project_root)
                        .into_iter()
                        .filter_map(|c| c.sha256),
                );
                let scratch = data_dir
                    .join("probes-scratch")
                    .join(uuid::Uuid::new_v4().to_string());
                match std::fs::create_dir_all(&scratch) {
                    Err(e) => Err(format!("could not create the probe scratch dir: {e}")),
                    Ok(()) => {
                        let outcome = registry
                            .execute_example_sandboxed(
                                &spec.name,
                                &example.args,
                                data_dir,
                                project_root,
                                caps,
                                cancel.clone(),
                                &scratch,
                            )
                            .await;
                        let _ = std::fs::remove_dir_all(&scratch);
                        // No backend: fall back to the pre-sandbox refusal,
                        // with the reason - never a silent unsandboxed run.
                        if !outcome.ok && outcome.output.contains("no sandbox backend available") {
                            Err(format!(
                                "unapproved source and this host cannot sandbox a probe ({}); a human can run `openmax --approve {}`, or make the tool's first call in a session and approve the card it raises",
                                outcome.output.trim(),
                                shell_quote(&ext.source_path)
                            ))
                        } else {
                            let verdict = example_verdict(&outcome, example);
                            if verdict.is_ok() {
                                // Evidence for the approval card: the exact
                                // pre-spawn vector, so any edit - before OR
                                // during the run - orphans the receipt.
                                record_probe_receipt(data_dir, &spec.name, &probed_shas);
                            }
                            verdict
                        }
                    }
                }
            }
            Err(reason) => Err(reason),
        };
        let verdict = ExampleVerdict {
            tool: spec.name.clone(),
            path: ext.source_path.clone(),
            result,
            sandboxed,
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

/// The three ways a `command` fails to resolve. They are three different
/// repairs, and for a hook they are also three different answers to "can this
/// be approved at all": `openmax --approve` blesses bytes, so a command that
/// resolves to no file binds nothing it can read, while an unexecutable script
/// approves fine and still cannot spawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CommandProblem {
    Absent,
    NotExecutable,
    NotOnPath,
}

impl CommandProblem {
    /// What repairs it, phrased for a hook that is inert until approved: each
    /// case names its own fix, and whether approval is even reachable yet.
    fn hook_repair(self, command: &str, manifest: &Path) -> String {
        let manifest = shell_quote(manifest);
        let command_q = shell_quote(Path::new(command));
        match self {
            Self::Absent => format!(
                "create it, then approve the hook and the code it runs together with `openmax --approve {manifest}`"
            ),
            Self::NotExecutable => format!(
                "make it executable (chmod +x {command_q}), then approve the hook and the code it runs together with `openmax --approve {manifest}`"
            ),
            Self::NotOnPath => format!(
                "install it or point `command` at a script inside the project; a name that resolves to nothing binds no code, so `openmax --approve {manifest}` refuses it as it stands"
            ),
        }
    }
}

/// Why `command` will not spawn from this checkout, if it will not. A path
/// (contains '/') resolves against the project root, exactly as the runtime
/// spawns it; a bare name resolves on PATH. This warns rather than errors:
/// check-time and run-time environments legitimately differ (CI without the
/// tool installed, a script the agent writes next).
fn missing_command_reason(
    command: &str,
    project_root: &Path,
) -> Option<(CommandProblem, String)> {
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
            return Some((
                CommandProblem::Absent,
                format!("command '{command}' does not exist from the project root"),
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let executable = std::fs::metadata(&path)
                .map(|m| m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false);
            if !executable {
                return Some((
                    CommandProblem::NotExecutable,
                    format!("command '{command}' exists but is not executable"),
                ));
            }
        }
        return None;
    }
    let found = std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| dir.join(command).is_file())
    });
    (!found).then(|| {
        (
            CommandProblem::NotOnPath,
            format!("command '{command}' is not on PATH"),
        )
    })
}

/// Why the script an interpreter-style command names will not run, if it
/// will not. Judged the way `bound_code` binds it: the first positional
/// argument, and only when it is shaped like a script file, so a module name
/// (`python3 -m pytest`) or a data argument never draws a warning about a
/// file it never was. The repair is `Absent`'s: the file has to exist before
/// anything else about it can be true.
fn missing_script_reason(
    command: &str,
    args: &[String],
    project_root: &Path,
) -> Option<(CommandProblem, String)> {
    let script = crate::ledger::interpreter_script(command, args)?;
    let path = if Path::new(script).is_absolute() {
        PathBuf::from(script)
    } else {
        project_root.join(script)
    };
    (!path.is_file()).then(|| {
        (
            CommandProblem::Absent,
            format!(
                "command '{}' runs '{script}', which does not exist from the project root",
                command.trim()
            ),
        )
    })
}

/// The clamp warnings for the entries that survived shadowing and the cap. A
/// file the loader never reaches describes no runtime behavior: it is reported
/// as shadowed or capped, and a second line about a timeout it will never
/// serve would read as if it ran.
fn clamp_findings(
    kind: &'static str,
    entries: &[Entry],
    clamps: Vec<(usize, String)>,
    shadowed: &std::collections::HashSet<usize>,
    capped: &std::collections::HashSet<usize>,
) -> Vec<Finding> {
    clamps
        .into_iter()
        .filter(|(i, _)| !shadowed.contains(i) && !capped.contains(i))
        .map(|(i, reason)| Finding {
            kind,
            path: entries[i].0.path.clone(),
            status: Status::Warn(reason),
        })
        .collect()
}

/// Why the `timeout_secs` this file asks for is not the one it gets, if it is
/// not. Both loaders clamp out-of-range values and say nothing, which for a
/// gate is a policy change rather than a detail: a gate that times out blocks,
/// so an author who wrote 600 is enforcing a 60-second budget. Takes the bytes
/// the caller already parsed, never a path: the parsed spec carries only the
/// clamped number, and reading the file again to recover the written one would
/// let a rewrite between the two reads produce a report of two generations.
fn clamped_timeout_reason(text: &str, what: &str, min: u64, max: u64) -> Option<String> {
    let written = toml::from_str::<toml::Value>(text)
        .ok()?
        .as_table()?
        .get("timeout_secs")?
        .as_integer()?;
    // A value outside i64, or below zero, never parses into the spec at all,
    // so anything reaching here is a number the loader accepted and clamped.
    let effective = written.clamp(min as i64, max as i64);
    (written != effective).then(|| {
        format!(
            "timeout_secs = {written} is outside the documented {min}..{max} range, so this {what} is clamped to {effective} seconds, not {written}"
        )
    })
}

/// What a skill's line in the frozen index will not say. The description is
/// the only text of a skill that ever reaches the model, so a skill without
/// one is indexed under a bare name that cannot say when to use it, and one
/// written past the cap is cut there, usually losing the "when" at the end.
/// Both load, and neither is visible in the file, which is why `--check` has
/// to say it. Takes the bytes the caller already parsed, for the same reason
/// the timeout clamp does.
/// Why the description this tool wrote is not the one the schema shows, when
/// it is not. Same contract as `skill_description_gap`: the schema line is
/// the model's only knowledge of an external tool, and both the clamp and an
/// empty string decide silently what that line can say.
fn tool_description_gap(text: &str, name: &str) -> Option<String> {
    let written = crate::registry::raw_description(text)?;
    if written.trim().is_empty() {
        return Some(format!(
            "'{name}' has an empty `description`: the schema line is the model's only knowledge of this tool, and it says nothing"
        ));
    }
    let cap = crate::registry::MAX_EXTERNAL_DESC_CHARS;
    let chars = written.chars().count();
    (chars > cap).then(|| {
        format!(
            "'{name}' has a {chars}-char `description`, past the {cap}-char cap: the schema shows the first {cap} chars and cuts the rest, so whatever the tail says never reaches the model"
        )
    })
}

/// The same gap for a template's popup line. An absent description is legal
/// here - the human invoking /name already knows what they installed - so
/// only the silent clamp is worth a line.
fn template_description_gap(text: &str, name: &str) -> Option<String> {
    let written = crate::templates::raw_description(text)?;
    let cap = crate::templates::MAX_TEMPLATE_DESC_CHARS;
    let chars = written.chars().count();
    (chars > cap).then(|| {
        format!(
            "'/{name}' has a {chars}-char `description:`, past the {cap}-char cap: the completion popup shows the first {cap} chars and cuts the rest"
        )
    })
}

fn skill_description_gap(text: &str, name: &str) -> Option<String> {
    let written = crate::skills::raw_description(text);
    if written.as_deref().is_none_or(|w| w.trim().is_empty()) {
        // A key that is present but empty (a bare `description:`, or a
        // `>`/`|` block header with nothing indented under it) reads the
        // same to the index as no key at all, and is named as what it is.
        let how = if written.is_some() { "an empty" } else { "no" };
        return Some(format!(
            "'{name}' has {how} `description:`, so its line in the frozen skills index is a bare name: the index line cannot say when to use it, and the model has nothing else to go on"
        ));
    }
    let written = written.unwrap_or_default();
    let cap = crate::skills::MAX_SKILL_DESC_CHARS;
    let chars = written.chars().count();
    (chars > cap).then(|| {
        format!(
            "'{name}' has a {chars}-char `description:`, past the {cap}-char cap: the frozen skills index shows the first {cap} chars and cuts the rest, so whatever the tail says (often the \"when\") never reaches the model"
        )
    })
}

/// Mark files a loader never reaches because another file claims the same
/// identity. `first_wins` mirrors the loader: hooks keep the first file to
/// claim a stem, while tools, skills, and templates let a later directory
/// overwrite an earlier one. Both orders resolve to the project tier, since
/// each surface lists its directories so the project file is the winner.
/// Returns the shadowed indices: the loader deduplicates before it caps, so
/// the cap ranking below needs to know which entries never held a slot.
fn mark_shadowed(entries: &mut [Entry], first_wins: bool) -> std::collections::HashSet<usize> {
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
    let mut indices = std::collections::HashSet::new();
    for (i, id, winner_path) in shadowed {
        indices.insert(i);
        let kind = entries[i].0.kind;
        entries[i].0.status = Status::Warn(format!(
            "shadowed by {}, where {kind} '{id}' resolves",
            winner_path.display()
        ));
    }
    indices
}

/// Mark entries the loader's cap drops. The loader counts every parsed,
/// deduplicated definition against its cap - a tool whose command is missing
/// still occupies a slot - so the ranking is the parsed winners whatever
/// their status, and it reproduces exactly which files never load. `shadowed`
/// excludes the entries deduplication already dropped; `in_scope` restricts
/// the ranking further (hooks cap per event, and only parsed files hold an
/// event); `what` names the cap in the message. Returns the indices it marked,
/// because nothing else --check says about a file that never loads is true.
fn mark_beyond_cap(
    entries: &mut [Entry],
    cap: usize,
    shadowed: &std::collections::HashSet<usize>,
    in_scope: impl Fn(usize) -> bool,
    what: &str,
) -> std::collections::HashSet<usize> {
    let mut live: Vec<(String, usize)> = entries
        .iter()
        .enumerate()
        .filter(|(i, (_, id))| id.is_some() && !shadowed.contains(i) && in_scope(*i))
        .map(|(i, (_, id))| (id.clone().unwrap(), i))
        .collect();
    live.sort();
    let mut marked = std::collections::HashSet::new();
    for (id, i) in live.into_iter().skip(cap) {
        marked.insert(i);
        entries[i].0.status = Status::Err(format!(
            "'{id}' is beyond the {cap}-file {what} and never loads: consolidate or delete files"
        ));
    }
    marked
}

/// Restate what displacement means for a gate a human approved. A shadowed or
/// beyond-cap tool simply does not load; a gate in that position still holds
/// the bytes a human blessed and is not gating, which the loop answers by
/// failing every tool call closed. Two approved hooks on one stem are exempt:
/// a project hook overriding a global one is precedence a human built, and the
/// loop reads it the same way.
fn mark_displaced_gates(entries: &mut [Entry], approved_gate: impl Fn(&Path) -> bool) {
    let mut winner: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, (_, id)) in entries.iter().enumerate() {
        if let Some(id) = id {
            winner.entry(id.as_str()).or_insert(i);
        }
    }
    let displaced: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(i, (finding, id))| {
            if !approved_gate(&finding.path) {
                return false;
            }
            match &finding.status {
                // The cap ranks approved entries only, so reaching here means
                // a hook a human blessed lost the ranking.
                Status::Err(message) => message.contains("never loads"),
                Status::Warn(message) => {
                    message.starts_with("shadowed by")
                        && id
                            .as_deref()
                            .and_then(|id| winner.get(id))
                            .is_some_and(|w| *w != *i && !matches!(entries[*w].0.status, Status::Ok(_)))
                }
                Status::Ok(_) => false,
            }
        })
        .map(|(i, _)| i)
        .collect();
    for i in displaced {
        let reason = entries[i].0.status.summary().to_string();
        entries[i].0.status = Status::Err(format!(
            "{reason}; it still holds the content a human approved, so every tool call fails closed until it can run again"
        ));
    }
}

/// Directories the project tier is actually read from, as (parent, child).
/// `memory` is the lone singular name here; its four siblings are plural, so
/// the natural miswrite is the plural `memories`, which `near` cannot reach
/// (see `is_regular_plural`).
const PROJECT_DIRS: &[(&str, &str)] = &[
    (".openmax", "tools"),
    (".openmax", "hooks"),
    (".openmax", "memory"),
    (".agents", "skills"),
    (".agents", "prompts"),
];

/// Files that legitimately sit directly under a project config directory.
const LOOSE_FILES: &[(&str, &str)] = &[(".openmax", "permissions.toml")];

/// The global tier is flat: every surface sits directly under the data dir
/// (`~/.openmax/{tools,hooks,skills,prompts}`), with no `.openmax`/`.agents`
/// split and no memory tier. The data dir also holds legitimate non-extension
/// files and dirs (settings.json, ledger/, sessions/), so the global net flags
/// only a near-miss of these names or a wrong-extension file inside one, never
/// a loose file.
const GLOBAL_DIRS: &[&str] = &["tools", "hooks", "skills", "prompts"];

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
                             run openmax --spec tools (or skills, prompts, hooks, \
                             permissions, memory) for where each surface lives"
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
            } else if let Some((_, c)) = PROJECT_DIRS
                .iter()
                .find(|(p, c)| *p == parent_name && (near(c, name) || is_regular_plural(c, name)))
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
        // `.openmax/memory` is canonical but not here: memory_findings already
        // names a non-.md file in it ("only .md files are memories"), and that
        // message is more specific than this generic one. Listing it here too
        // would report the same file twice.
        _ => return Vec::new(),
    };
    wrong_extension_in(dir, want, &format!("{parent}/{child}/"))
}

/// Files in a read directory whose extension is a recognizable config format
/// other than `want`, or a near-miss of it (.tml, .tomll): meant to be read,
/// silently never read. The caller names the directory in the message, so the
/// project and global tiers share one scan.
fn wrong_extension_in(dir: &Path, want: &str, dir_label: &str) -> Vec<Finding> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    paths.sort();
    paths
        .into_iter()
        .filter(|p| p.is_file())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                // A config format in the wrong dialect (.yaml where .toml is
                // read), a near-miss of the right one (.tml), or the harness's
                // other structured format `.toml` where `.md` is read (a prompt
                // written as a tool). `.md` where `.toml` is read is left out on
                // purpose: it is as likely a README as a misplaced surface file.
                e != want && (CONFIG_EXTENSIONS.contains(&e) || e == "toml" || near(e, want))
            })
        })
        .map(|path| Finding {
            kind: "path",
            path,
            status: Status::Warn(format!("not read; {dir_label} is read as .{want} only")),
        })
        .collect()
}

/// The global tier's version of `unread_paths`. The data dir is flat and also
/// holds legitimate files and dirs, so this flags only two things: a directory
/// whose name is a near-miss of a canonical surface, and a wrong-extension file
/// inside a canonical one. A loose file or an unrelated dir (ledger/, sessions/)
/// is left alone. Every canonical global name is plural, so `near` alone covers
/// the singular and one-typo misses; there is no singular name to need the
/// `is_regular_plural` bridge the project tier gives `memory`.
fn global_unread_paths(data_dir: &Path) -> Vec<Finding> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(data_dir) else { return out };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        if let Some(canon) = GLOBAL_DIRS.iter().find(|c| **c == name) {
            let want = match *canon {
                "tools" | "hooks" => "toml",
                "prompts" => "md",
                // A skill is a directory holding SKILL.md, not a flat file; the
                // skill loop already validates skills across both tiers.
                _ => continue,
            };
            out.extend(wrong_extension_in(&path, want, &format!("the global {canon}/")));
            continue;
        }
        if dir_is_empty(&path) {
            continue;
        }
        if let Some(canon) = GLOBAL_DIRS.iter().find(|c| near(c, name)) {
            out.push(Finding {
                kind: "path",
                path,
                status: Status::Warn(format!("not read; did you mean the global {canon}/")),
            });
        }
    }
    out
}

/// How a skill's frontmatter `name` departs from the Agent Skills naming rules
/// (agentskills.io/specification: 1 to 64 characters, lowercase letters,
/// digits, and hyphens only, no leading or trailing hyphen, no consecutive
/// hyphens). Open Max indexes by the frontmatter name and loads the skill
/// regardless; a standard skills consumer will not load a name that breaks
/// these, so a violation is worth a portability advisory. The spec's
/// name-matches-parent-directory rule is deliberately NOT checked: Open Max
/// indexes by the frontmatter name by design, so name != dir is a first-class
/// shape here (its own fixtures use it), and warning on it would fight the
/// documented model rather than flag a portability defect.
fn agent_skills_name_issue(name: &str) -> Option<String> {
    let mut issues: Vec<&str> = Vec::new();
    let len = name.chars().count();
    if len == 0 || len > 64 {
        issues.push("must be 1 to 64 characters");
    }
    if !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
        issues.push("may hold only lowercase letters, digits, and hyphens");
    }
    if name.starts_with('-') || name.ends_with('-') {
        issues.push("must not start or end with a hyphen");
    }
    if name.contains("--") {
        issues.push("must not contain consecutive hyphens");
    }
    if issues.is_empty() {
        return None;
    }
    Some(format!(
        "skill name '{name}' is not a portable Agent Skills name ({}); Open Max loads it, \
         but a standard skills consumer will not",
        issues.join("; ")
    ))
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

/// A correctly-spelled `SKILL.md` nested below `dir` (the caller has already
/// ruled out one at the top). The registry joins `SKILL.md` exactly one level
/// under the skills root, so a deeper one is a skill placed a level too far
/// down, not a misspelling. The path is returned relative to `dir` for the
/// message, and dot-prefixed entries are skipped so an editor cache never
/// stands in for the real file.
fn nested_skill_md(dir: &Path) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    for entry in rd.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            if let Some(found) = nested_skill_md(&path) {
                return Some(PathBuf::from(name).join(found));
            }
        } else if name == "SKILL.md" {
            return Some(PathBuf::from(name));
        }
    }
    None
}

/// One edit apart, counting an adjacent swap as one edit: a dropped plural, a
/// doubled letter, a single mistyped or transposed character. Shared with the
/// providers validator, which suggests near-miss keys the same way.
pub(crate) fn near(a: &str, b: &str) -> bool {
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

/// Is `candidate` the regular English plural of the singular directory name
/// `singular`? `near` already pairs a singular canonical dir with a simple
/// `+s` plural, because that is one trailing edit (tools/tool). It cannot
/// bridge `y -> ies`, which is three edits, so a `memory` dir written
/// `memories` (the plural its four sibling surfaces all use) would otherwise
/// slip through with no diagnostic. This names that one class of miswrite.
fn is_regular_plural(singular: &str, candidate: &str) -> bool {
    let plural = match singular.strip_suffix('y') {
        // English pluralizes `y` to `ies` only after a consonant (memory ->
        // memories, not day -> daies); every other `y` word just takes `s`.
        Some(stem) if stem.ends_with(|c: char| !"aeiou".contains(c)) => format!("{stem}ies"),
        _ if singular.ends_with(['s', 'x', 'z']) || singular.ends_with("ch") || singular.ends_with("sh") => {
            format!("{singular}es")
        }
        _ => format!("{singular}s"),
    };
    candidate == plural
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

    /// An approved gate can stop running without its file changing at all:
    /// it lands past the per-event cap. The loop fails every tool call closed
    /// on that, so `--check` has to describe that state rather than the milder
    /// one it looks like from the directory listing ("never loads").
    #[test]
    fn an_approved_gate_past_the_cap_reads_as_fail_closed() {
        let root = temp_project();
        let data = root.join("data");
        let body = "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n";
        for i in 0..(crate::hooks::MAX_HOOKS_PER_EVENT + 1) {
            let path = root.join(format!(".openmax/hooks/gate-{i:03}.toml"));
            write(path.clone(), body);
            let sha = crate::ledger::sha256_hex(&std::fs::read(&path).unwrap());
            crate::ledger::approve_capability(&data, &root, &path, &[sha]).unwrap();
        }

        let findings = check_at(&root, &data);
        let over = find(
            &findings,
            &format!("gate-{:03}.toml", crate::hooks::MAX_HOOKS_PER_EVENT),
        );
        match &over.status {
            Status::Err(reason) => {
                assert!(reason.contains("never loads"), "{reason}");
                assert!(reason.contains("fails closed"), "{reason}");
            }
            other => panic!("a gate past the cap must read as fail-closed: {other:?}"),
        }
        // The hooks that do run are still reported as healthy.
        assert!(matches!(find(&findings, "gate-000.toml").status, Status::Ok(_)));
        let _ = std::fs::remove_dir_all(root);
    }

    /// The other way a gate is displaced is occupation: a same-stem file in a
    /// higher-precedence directory. Judged here on the entries themselves,
    /// because the two tiers are the project and `$HOME` - and pointing the
    /// process at a different HOME to build the case would leak into every
    /// other test in this binary that reads it. Two approved hooks on one
    /// stem stay the precedence a human built, exactly as the loop reads it.
    #[test]
    fn displacement_marks_a_gate_only_when_what_holds_its_stem_is_not_live() {
        let entry = |path: &str, status: Status| {
            (Finding { kind: "hook", path: PathBuf::from(path), status }, Some("gate".to_string()))
        };
        let shadowed = || {
            Status::Warn("shadowed by /p/.openmax/hooks/gate.toml, where hook 'gate' resolves".into())
        };

        // The occupier does not load, so the human's gate is not gating.
        let mut entries = vec![
            entry("/p/.openmax/hooks/gate.toml", Status::Err("inert because its content is not approved".into())),
            entry("/h/.openmax/hooks/gate.toml", shadowed()),
        ];
        mark_displaced_gates(&mut entries, |_| true);
        match &entries[1].0.status {
            Status::Err(reason) => {
                assert!(reason.contains("shadowed by"), "{reason}");
                assert!(reason.contains("fails closed"), "{reason}");
            }
            other => panic!("a shadowed approved gate must read as fail-closed: {other:?}"),
        }

        // The occupier is live: a project hook overriding a global one.
        let mut entries = vec![
            entry("/p/.openmax/hooks/gate.toml", Status::Ok("hook on pre_tool_use".into())),
            entry("/h/.openmax/hooks/gate.toml", shadowed()),
        ];
        mark_displaced_gates(&mut entries, |_| true);
        assert!(
            matches!(entries[1].0.status, Status::Warn(_)),
            "an approved override is precedence, not displacement: {:?}",
            entries[1].0.status
        );

        // Nothing a human approved is at stake, so nothing is upgraded.
        let mut entries = vec![
            entry("/p/.openmax/hooks/gate.toml", Status::Err("inert".into())),
            entry("/h/.openmax/hooks/gate.toml", shadowed()),
        ];
        mark_displaced_gates(&mut entries, |_| false);
        assert!(matches!(entries[1].0.status, Status::Warn(_)));
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

    /// The description is the whole of a skill's presence in the prompt, so a
    /// skill missing one, or writing one longer than the index shows, is not
    /// the healthy file `ok` claims: both were silent before this, and the
    /// degradation is invisible in the file itself. The skill still loads, so
    /// both are warnings.
    #[test]
    fn skill_descriptions_that_the_index_cannot_carry_are_named() {
        let root = temp_project();
        // A scoped data dir, so a global skill on the developer's machine
        // cannot spend the index budget these four are measured against.
        let data = temp_project();
        write(
            root.join(".agents/skills/silent/SKILL.md"),
            "---\nname: silent\n---\nbody\n",
        );
        write(
            root.join(".agents/skills/blank/SKILL.md"),
            "---\nname: blank\ndescription:   \n---\nbody\n",
        );
        let long = "w".repeat(crate::skills::MAX_SKILL_DESC_CHARS + 40);
        write(
            root.join(".agents/skills/wordy/SKILL.md"),
            &format!("---\nname: wordy\ndescription: {long}\n---\nbody\n"),
        );
        write(
            root.join(".agents/skills/good/SKILL.md"),
            "---\nname: good\ndescription: cuts a release, when the changelog is ready\n---\nbody\n",
        );
        // A block scalar with nothing under it: the key is there, the value
        // is not, and before block scalars were read the index carried `>`.
        write(
            root.join(".agents/skills/hollow/SKILL.md"),
            "---\nname: hollow\ndescription: >\n---\nbody\n",
        );
        // The multi-line spelling third-party packages ship: folded to one
        // index line, and healthy.
        write(
            root.join(".agents/skills/folded/SKILL.md"),
            "---\nname: folded\ndescription: >\n  Manages stacked branches.\n  Use when a stack is checked out.\nmetadata:\n  author: someone\n---\nbody\n",
        );

        let findings = check_at(&root, &data);
        for (dir, name, how) in [("silent", "silent", "no"), ("blank", "blank", "an empty"), ("hollow", "hollow", "an empty")] {
            match &find(&findings, &format!("{dir}/SKILL.md")).status {
                Status::Warn(reason) => {
                    assert!(reason.contains(&format!("'{name}' has {how} `description:`")), "{reason}");
                    assert!(reason.contains("when to use it"), "{reason}");
                }
                other => panic!("a skill with no description must warn: {other:?}"),
            }
        }
        match &find(&findings, "folded/SKILL.md").status {
            Status::Ok(summary) => assert!(
                summary.contains("skill 'folded'"),
                "a folded description is a healthy skill: {summary}"
            ),
            other => panic!("a folded block scalar description is healthy: {other:?}"),
        }
        match &find(&findings, "wordy/SKILL.md").status {
            Status::Warn(reason) => {
                assert!(
                    reason.contains(&format!("{}-char cap", crate::skills::MAX_SKILL_DESC_CHARS)),
                    "the cap must be named: {reason}"
                );
                assert!(reason.contains("cuts the rest"), "{reason}");
            }
            other => panic!("an over-cap description must warn: {other:?}"),
        }
        match &find(&findings, "good/SKILL.md").status {
            Status::Ok(_) => {}
            other => panic!("a describable skill stays healthy: {other:?}"),
        }
        assert!(!has_errors(&findings), "none of this stops the skill loading");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// A template whose frontmatter never closes expands its own fence and
    /// keys into the user's message. SKILL.md refuses that by name; the
    /// template parser now does too, and the report has to agree with the
    /// parser or one of them is lying about what `/name` will send.
    #[test]
    fn a_template_with_unclosed_frontmatter_is_an_error() {
        let root = temp_project();
        write(
            root.join(".agents/prompts/half-open.md"),
            "---\ndescription: fix a bug\nFind the bug in $1 and fix it.\n",
        );
        match &find(&local(&root), "half-open.md").status {
            Status::Err(reason) => assert!(reason.contains("frontmatter never closes"), "{reason}"),
            other => panic!("an unclosed frontmatter block must be an error: {other:?}"),
        }
        assert!(
            crate::templates::expand_invocation(&root.join("data"), &root, "half-open now")
                .is_none(),
            "the report says err, so the runtime must refuse it too"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// Every finding for one path: clamp-style warnings ride as extra lines
    /// beside the entry's own status, so asserting on the first match alone
    /// would read past them.
    fn statuses_of<'a>(findings: &'a [Finding], needle: &str) -> Vec<&'a Status> {
        findings
            .iter()
            .filter(|f| f.path.to_string_lossy().contains(needle))
            .map(|f| &f.status)
            .collect()
    }

    fn some_warn_contains(findings: &[Finding], needle: &str, text: &str) -> bool {
        statuses_of(findings, needle)
            .iter()
            .any(|s| matches!(s, Status::Warn(reason) if reason.contains(text)))
    }

    /// The schema clamps an overlong tool description with no report, so the
    /// author believes the model reads the whole line; same silent-surface
    /// class as the skill description gap, one surface over.
    #[test]
    fn an_overlong_tool_description_warns_what_the_schema_cuts() {
        let root = temp_project();
        let long = "word ".repeat(60);
        write(
            root.join(".openmax/tools/big.toml"),
            &format!("name = \"big\"\ndescription = \"{long}\"\ncommand = \"/bin/sh\"\n"),
        );
        let findings = local(&root);
        assert!(
            some_warn_contains(&findings, "big.toml", "past the 200-char cap"),
            "an overlong tool description must warn: {findings:?}"
        );

        write(
            root.join(".openmax/tools/mute.toml"),
            "name = \"mute\"\ndescription = \"\"\ncommand = \"/bin/sh\"\n",
        );
        let findings = local(&root);
        assert!(
            some_warn_contains(&findings, "mute.toml", "empty `description`"),
            "an empty tool description must warn: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// The popup clamps a template description the same way; a template with
    /// a clipped line read as plain ok.
    #[test]
    fn an_overlong_template_description_warns_what_the_popup_cuts() {
        let root = temp_project();
        let long = "word ".repeat(60);
        write(
            root.join(".agents/prompts/verbose.md"),
            &format!("---\ndescription: {long}\n---\nDo the thing to $1.\n"),
        );
        let findings = local(&root);
        assert!(
            some_warn_contains(&findings, "verbose.md", "past the 200-char cap"),
            "an overlong template description must warn: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// `.tml` is one edit from the one extension the dir is read as, which is
    /// precisely the typo an author makes; it was silently never read while
    /// `.yaml` in the same dir warned.
    #[test]
    fn a_near_miss_extension_in_a_manifest_dir_is_named() {
        let root = temp_project();
        write(root.join(".openmax/tools/notes.tml"), "name = \"notes\"\n");
        let findings = local(&root);
        assert!(
            some_warn_contains(&findings, "notes.tml", "read as .toml only"),
            "a near-miss extension must be named: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A SKILL.md spelled skill.md must draw a warning on every filesystem.
    /// Case-sensitive systems never load it (the existing miscase branch);
    /// case-folding systems load it fine and previously said nothing, which
    /// is backwards: the machine where it works is the machine that needs the
    /// warning, because the first case-sensitive checkout drops the skill
    /// silently.
    #[test]
    fn a_miscased_skill_file_is_named_on_every_filesystem() {
        let root = temp_project();
        write(
            root.join(".agents/skills/howto/skill.md"),
            "---\nname: howto\ndescription: how to do the thing\n---\nbody\n",
        );
        let findings = local(&root);
        assert!(
            some_warn_contains(&findings, "howto", "skill.md"),
            "the miscased file must be named whether or not it loaded here: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A skill whose frontmatter name is not a valid Agent Skills name
    /// (uppercase, spaces, bad hyphens, or too long) loads in Open Max but will
    /// not port to a standard skills consumer, so --check advises it.
    #[test]
    fn a_nonportable_skill_name_is_advised() {
        let root = temp_project();
        write(
            root.join(".agents/skills/pdftools/SKILL.md"),
            "---\nname: PDF Tools\ndescription: work with pdfs, when the user mentions a pdf\n---\nbody\n",
        );
        let findings = local(&root);
        let warn = find(&findings, "pdftools");
        assert!(
            warn.status.summary().contains("not a portable Agent Skills name")
                && warn.status.summary().contains("lowercase"),
            "a non-portable name must be advised: {}",
            warn.status.summary()
        );
        assert!(!has_errors(&findings), "an advisory must not fail the check");
        let _ = std::fs::remove_dir_all(root);
    }

    /// A conformant name draws no advisory, and a name that merely differs from
    /// its directory is NOT flagged: Open Max indexes by the frontmatter name,
    /// so name != dir is a first-class shape here, not a portability defect.
    #[test]
    fn a_conformant_name_that_differs_from_its_dir_is_not_advised() {
        let root = temp_project();
        write(
            root.join(".agents/skills/review/SKILL.md"),
            "---\nname: code-review\ndescription: reviews a diff, when a branch is ready\n---\nbody\n",
        );
        let findings = local(&root);
        let f = find(&findings, "review/SKILL.md");
        assert!(
            matches!(f.status, Status::Ok(_)),
            "a conformant name that differs from its dir is not advised: {:?}",
            f.status
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn agent_skills_name_issue_matches_the_spec_rules() {
        for ok in ["pdf-processing", "code-review", "data42", "a"] {
            assert!(agent_skills_name_issue(ok).is_none(), "{ok} conforms");
        }
        for bad in ["PDF-Processing", "pdf tools", "-pdf", "pdf-", "pdf--processing", ""] {
            assert!(agent_skills_name_issue(bad).is_some(), "{bad:?} violates a rule");
        }
        assert!(agent_skills_name_issue(&"a".repeat(65)).is_some(), "65 chars is too long");
        assert!(agent_skills_name_issue(&"a".repeat(64)).is_none(), "64 chars is the limit");
    }

    /// A correctly-spelled SKILL.md placed a level too deep is a depth mistake,
    /// not a spelling one. Calling it misspelled sends the author renaming a
    /// file that is already right; --check must say it is too deep and where.
    #[test]
    fn a_skill_md_nested_too_deep_is_named_as_depth_not_spelling() {
        let root = temp_project();
        write(
            root.join(".agents/skills/pdf/tools/SKILL.md"),
            "---\nname: pdf\ndescription: work with pdfs\n---\nbody\n",
        );
        let findings = local(&root);
        let warn = find(&findings, ".agents/skills/pdf");
        assert!(
            warn.status.summary().contains("deeper")
                && warn.status.summary().contains("tools/SKILL.md"),
            "a too-deep SKILL.md must be named as depth and located: {}",
            warn.status.summary()
        );
        assert!(
            !warn.status.summary().contains("not spelled exactly"),
            "the file is spelled exactly; the reason must not say otherwise: {}",
            warn.status.summary()
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// An interpreter command whose argv names a script that does not exist
    /// fails at every call (tool) or every fire (hook), and --check said
    /// nothing: `missing_command_reason` judged only `command`, and `python3`
    /// resolves fine.
    #[test]
    fn a_missing_interpreter_script_is_named_for_tools_and_hooks() {
        let root = temp_project();
        write(
            root.join(".openmax/tools/runner.toml"),
            "name = \"runner\"\ndescription = \"runs the generator\"\ncommand = \"python3\"\nargs = [\"gen.py\"]\n",
        );
        write(
            root.join(".openmax/hooks/audit.toml"),
            "event = \"post_tool_use\"\ncommand = \"python3\"\nargs = [\"audit.py\"]\n",
        );
        let findings = local(&root);
        assert!(
            some_warn_contains(&findings, "runner.toml", "runs 'gen.py', which does not exist"),
            "the tool's missing script must be named: {findings:?}"
        );
        let hook_says = statuses_of(&findings, "audit.toml").iter().any(|s| match s {
            Status::Err(reason) | Status::Warn(reason) => {
                reason.contains("runs 'audit.py', which does not exist")
                    && reason.contains("create it")
            }
            _ => false,
        });
        assert!(hook_says, "the hook's missing script must be named with its repair: {findings:?}");

        // A module name is not a script file: `-m pytest` must stay silent.
        write(
            root.join(".openmax/tools/mod.toml"),
            "name = \"modrun\"\ndescription = \"runs a module\"\ncommand = \"python3\"\nargs = [\"-m\", \"pytest\"]\n",
        );
        // An option can consume or redefine the operand behind it: `node -p`
        // evaluates the next token as expression text, and `sh -s` reads the
        // program from stdin and keeps the token as $1. Neither names a file
        // the command opens, so neither may draw a missing-script warning.
        write(
            root.join(".openmax/tools/nodep.toml"),
            "name = \"nodep\"\ndescription = \"prints an expression\"\ncommand = \"node\"\nargs = [\"-p\", \"result.js\"]\n",
        );
        write(
            root.join(".openmax/tools/shs.toml"),
            "name = \"shs\"\ndescription = \"stdin program\"\ncommand = \"sh\"\nargs = [\"-s\", \"script.sh\"]\n",
        );
        let findings = local(&root);
        assert!(
            !some_warn_contains(&findings, "mod.toml", "does not exist"),
            "a module argument must not draw a script warning: {findings:?}"
        );
        assert!(
            !some_warn_contains(&findings, "nodep.toml", "does not exist"),
            "an option operand is not a script: {findings:?}"
        );
        assert!(
            !some_warn_contains(&findings, "shs.toml", "does not exist"),
            "sh -s keeps the token as $1, not a program file: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A memory whose first line is longer than the index cap is indexed under
    /// a clipped line: the future session it was written for reads the clip.
    /// Reporting a plain `ok` left the author believing the whole sentence
    /// carried over.
    #[test]
    fn a_memory_first_line_past_the_index_cap_says_it_is_clipped() {
        let root = temp_project();
        let long = "e".repeat(crate::memory::MAX_DESCRIPTION_CHARS + 30);
        write(root.join(".openmax/memory/verbose.md"), &format!("# {long}\nbody\n"));
        write(root.join(".openmax/memory/terse.md"), "# The deploy port is 7443\nbody\n");

        let findings = local(&root);
        match &find(&findings, "verbose.md").status {
            Status::Warn(reason) => {
                assert!(reason.contains("indexed at the next prompt freeze"), "still indexed: {reason}");
                assert!(
                    reason.contains(&format!("{}-char index cap", crate::memory::MAX_DESCRIPTION_CHARS)),
                    "the cap must be named: {reason}"
                );
                assert!(reason.contains("clipped"), "{reason}");
            }
            other => panic!("a clipped memory description must warn: {other:?}"),
        }
        assert!(matches!(find(&findings, "terse.md").status, Status::Ok(_)));
        assert!(!has_errors(&findings), "memory findings never fail a check");
        let _ = std::fs::remove_dir_all(root);
    }

    /// Memory is flat: the scan indexes only top-level `.md` files, so a note
    /// tucked into a subfolder is silently never read. --check must name it
    /// rather than report a clean bill of health.
    #[test]
    fn a_memory_note_nested_in_a_subdirectory_is_named() {
        let root = temp_project();
        write(root.join(".openmax/memory/adr/frozen-prefix.md"), "# a durable fact\nbody\n");
        // A valid top-level note and the internal access log must not be
        // dragged into the warning.
        write(root.join(".openmax/memory/deploy-port.md"), "# The deploy port is 7443\nb\n");
        write(root.join(".openmax/memory/.access.jsonl"), "{}\n");

        let findings = local(&root);
        let warn = find(&findings, ".openmax/memory/adr");
        assert!(
            warn.status.summary().contains("never indexed"),
            "a nested memory note must be named: {}",
            warn.status.summary()
        );
        assert!(matches!(warn.status, Status::Warn(_)), "it is a warning, not an error");
        assert!(!has_errors(&findings), "memory findings never fail a check");
        assert!(
            findings.iter().any(|f| f.kind == "memory" && matches!(f.status, Status::Ok(_))),
            "the healthy top-level note still reports Ok: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A markdown file that exists only inside a hidden descendant (an editor
    /// cache, a nested `.git`) of a memory subdir is tooling, not a lost note,
    /// so it must not raise the flat-memory warning, the same exemption the
    /// skill scan gives `.DS_Store`.
    #[test]
    fn a_memory_note_hidden_in_editor_metadata_is_not_a_lost_note() {
        let root = temp_project();
        write(root.join(".openmax/memory/vault/.obsidian/cache.md"), "# editor state\n");

        let findings = local(&root);
        assert!(
            !findings.iter().any(|f| f.path.to_string_lossy().contains("vault")),
            "a note only inside a hidden descendant must not warn: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// The hidden-descendant exemption keys on the leading byte, not on the
    /// name being valid UTF-8: a name that begins with a dot but carries
    /// invalid bytes after it is still hidden tooling (a `to_str` check would
    /// drop the whole name to None and traverse it). Filesystems that reject
    /// such names never create the directory, so the fs-level test is
    /// effectively Linux-only; the check itself is exercised here in memory.
    #[cfg(unix)]
    #[test]
    fn a_non_utf8_dot_name_is_recognized_as_hidden() {
        use std::os::unix::ffi::OsStrExt;
        assert!(name_is_hidden(std::ffi::OsStr::from_bytes(b".\xffcache")));
        assert!(name_is_hidden(std::ffi::OsStr::new(".obsidian")));
        assert!(!name_is_hidden(std::ffi::OsStr::new("vault")));
        // A non-dot leading byte is visible even if invalid UTF-8 follows.
        assert!(!name_is_hidden(std::ffi::OsStr::from_bytes(b"v\xffault")));
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
        // Parses and loads; approval state (Ok vs the unapproved Warn) is
        // the ledger's business, asserted elsewhere.
        assert!(matches!(
            find(&findings, "good.toml").status,
            Status::Ok(_) | Status::Warn(_)
        ));
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
    async fn an_unapproved_tool_probes_sandboxed_and_cannot_touch_the_host() {
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
        let ok = verdict(&results, "approved");
        assert!(ok.result.is_ok());
        assert!(!ok.sandboxed, "approved content keeps its host run");
        // Unapproved content now probes in a sandbox instead of refusing
        // flat: the guarantee the refusal used to provide - no host side
        // effects - is enforced by the sandbox itself. The write-outside-
        // scratch is denied, so the probe fails loudly and the project stays
        // untouched. On a host with no backend, the fall-back refusal keeps
        // the old wording.
        let pwn = verdict(&results, "pwn");
        match &pwn.result {
            Err(reason) if reason.contains("cannot sandbox a probe") => {
                assert!(reason.contains("--approve"), "{reason}");
            }
            Err(reason) => {
                assert!(pwn.sandboxed, "unapproved content runs only as a probe");
                assert!(reason.contains("example run failed"), "{reason}");
            }
            Ok(()) => panic!("a write outside the scratch must fail the probe"),
        }
        assert!(!touched.exists(), "the probe must not touch the project");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// A passing probe leaves an exact-vector receipt; editing the manifest
    /// orphans it. The receipt is evidence, never authority: the ledger
    /// still records no approval.
    #[tokio::test]
    async fn a_passing_probe_leaves_a_receipt_bound_to_the_exact_bytes() {
        let root = temp_project();
        let manifest = tool_file(
            &root,
            "probe.toml",
            "name = \"probe\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"cat\"]\n\n[example]\nexpect_regex = \"hello\"\n[example.args]\nmsg = \"hello\"\n",
        );
        let data = approved_data_dir(&root, &[]);

        let results = examples(&root, &data).await.unwrap();
        let v = verdict(&results, "probe");
        if let Err(reason) = &v.result {
            assert!(reason.contains("cannot sandbox a probe"), "{reason}");
            let _ = std::fs::remove_dir_all(root);
            let _ = std::fs::remove_dir_all(data);
            return;
        }
        let bytes = std::fs::read(&manifest).unwrap();
        let shas = vec![crate::ledger::sha256_hex(&bytes)];
        assert!(probe_passed(&data, &shas), "the passing probe must leave a receipt");
        assert!(
            !crate::ledger::is_approved(&data, &root, &shas[0]),
            "a receipt is evidence, not an approval"
        );

        // Any edit to the manifest changes the vector and orphans the receipt.
        std::fs::write(
            &manifest,
            "name = \"probe\"\ndescription = \"edited\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"cat\"]\n",
        )
        .unwrap();
        let edited = std::fs::read(&manifest).unwrap();
        let edited_shas = vec![crate::ledger::sha256_hex(&edited)];
        assert!(!probe_passed(&data, &edited_shas), "edited bytes have no receipt");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// A script rewritten WHILE its probe runs must not earn a passing
    /// receipt for the rewritten bytes: the receipt names the vector that
    /// was actually probed (captured before the spawn), and the rewritten
    /// vector has no receipt.
    #[tokio::test]
    async fn a_receipt_names_the_bytes_that_were_probed_not_the_bytes_after() {
        let root = temp_project();
        let script = root.join("slow.sh");
        write(
            script.clone(),
            "#!/bin/sh\nsleep 1\nprintf hello\n",
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let manifest = tool_file(
            &root,
            "slow.toml",
            "name = \"slow\"\ndescription = \"d\"\ncommand = \"./slow.sh\"\n\n[example]\nexpect_regex = \"hello\"\n",
        );
        let data = approved_data_dir(&root, &[]);
        let original_script_sha = crate::ledger::sha256_hex(&std::fs::read(&script).unwrap());
        let manifest_sha = crate::ledger::sha256_hex(&std::fs::read(&manifest).unwrap());

        // Rewrite the script mid-run: the probe sleeps 1s, we swap at ~200ms.
        let rewriter = {
            let script = script.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                // Same length and same output as the original, so the
                // shell (which reads scripts incrementally) still exits 0
                // and prints hello - only the BYTES differ. That is the
                // precise shape where a post-run hash would lie.
                std::fs::write(&script, "#!/bin/sh\nsleep 1\nprintf hello\n#x").unwrap();
            })
        };
        let results = examples(&root, &data).await.unwrap();
        let _ = rewriter.await;
        let v = verdict(&results, "slow");
        if let Err(reason) = &v.result {
            assert!(reason.contains("cannot sandbox a probe"), "{reason}");
            let _ = std::fs::remove_dir_all(root);
            let _ = std::fs::remove_dir_all(data);
            return;
        }
        let probed = vec![manifest_sha.clone(), original_script_sha];
        assert!(probe_passed(&data, &probed), "the receipt names the bytes that actually ran");
        let rewritten_sha = crate::ledger::sha256_hex(&std::fs::read(&script).unwrap());
        let rewritten = vec![manifest_sha, rewritten_sha];
        assert!(!probe_passed(&data, &rewritten), "the rewritten bytes earned no receipt");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// The fluidity half of the probe contract: an agent can prove a
    /// freshly written, harmless tool end-to-end before any human approves
    /// it - the probe runs, its verdict is marked sandboxed, and nothing is
    /// blessed by the pass (the ledger still has no approval).
    #[tokio::test]
    async fn an_unapproved_read_only_tool_probe_passes_in_the_sandbox() {
        let root = temp_project();
        let unapproved = tool_file(
            &root,
            "probe.toml",
            "name = \"probe\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\nargs = [\"-c\", \"cat\"]\n\n[example]\nexpect_regex = \"hello\"\n[example.args]\nmsg = \"hello\"\n",
        );
        let data = approved_data_dir(&root, &[]);

        let results = examples(&root, &data).await.unwrap();
        let v = verdict(&results, "probe");
        match &v.result {
            Err(reason) if reason.contains("cannot sandbox a probe") => {
                // No backend on this host: fail-closed verified instead.
                assert!(reason.contains("--approve"), "{reason}");
            }
            Err(reason) => panic!("a harmless probe must pass in the sandbox: {reason}"),
            Ok(()) => {
                assert!(v.sandboxed);
                let bytes = std::fs::read(&unapproved).unwrap();
                assert!(
                    !crate::ledger::is_approved(&data, &root, &crate::ledger::sha256_hex(&bytes)),
                    "a passing probe must approve nothing"
                );
            }
        }
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

    /// The loader counts every parsed tool against the cap, whatever state
    /// its command is in, so the cap ranking must too - in both directions.
    /// A warned tool sorted past the cap never loads, so its name is dead for
    /// rules; a warned tool inside the cap pushes the last healthy tool out,
    /// which must not keep reading as healthy.
    #[test]
    fn the_cap_ranking_counts_warned_tools_like_the_loader() {
        let cap = crate::registry::MAX_EXTERNAL_TOOLS;
        let healthy = |root: &Path| {
            for i in 0..cap {
                write(
                    root.join(".openmax/tools").join(format!("tool-{i:03}.toml")),
                    &format!("name = \"tool-{i:03}\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\n"),
                );
            }
        };

        // Sorts after every healthy tool: the loader's sorted head drops it.
        let root = temp_project();
        let data = temp_project();
        healthy(&root);
        write(
            root.join(".openmax/tools/zzz-ghost.toml"),
            "name = \"zzz-ghost\"\ndescription = \"d\"\ncommand = \"./missing.sh\"\n",
        );
        write(
            root.join(".openmax/permissions.toml"),
            "[[rules]]\neffect = \"deny\"\ntool = \"zzz-ghost\"\n",
        );
        let findings = check_at(&root, &data);
        match &find(&findings, "zzz-ghost.toml").status {
            Status::Err(reason) => assert!(reason.contains("never loads"), "{reason}"),
            other => panic!("a capped-out tool is dead, not merely command-less: {other:?}"),
        }
        assert!(
            findings.iter().any(|f| f.status.summary().contains("no tool named 'zzz-ghost'")),
            "a rule naming a capped-out tool never matches and must say so: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);

        // Sorts before every healthy tool: it takes a slot, and the last
        // healthy tool is the one the loader drops.
        let root = temp_project();
        let data = temp_project();
        healthy(&root);
        write(
            root.join(".openmax/tools/aaa-ghost.toml"),
            "name = \"aaa-ghost\"\ndescription = \"d\"\ncommand = \"./missing.sh\"\n",
        );
        let findings = check_at(&root, &data);
        match &find(&findings, "aaa-ghost.toml").status {
            Status::Warn(reason) => assert!(reason.contains("does not exist"), "{reason}"),
            other => panic!("a loaded tool with a missing command warns: {other:?}"),
        }
        match &find(&findings, &format!("tool-{:03}.toml", cap - 1)).status {
            Status::Err(reason) => assert!(reason.contains("never loads"), "{reason}"),
            other => panic!("the tool the warned one displaced must not read healthy: {other:?}"),
        }
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
        // A loadable but never-approved tool is not "ok": its first call
        // stops for approval, and --check says so (the same disclosure a
        // hook gets, previously missing for tools).
        let real = findings.iter().find(|f| f.path.ends_with("real.toml")).unwrap();
        assert!(matches!(real.status, Status::Warn(_)), "{:?}", real.status);
        assert!(real.status.summary().contains("no human has approved"), "{}", real.status.summary());
        assert!(real.status.summary().contains("--approve"), "{}", real.status.summary());
        // Approve the exact bytes and it reads healthy.
        let bytes = std::fs::read(tools_dir.join("real.toml")).unwrap();
        crate::ledger::approve_hash(&data, &root, &crate::ledger::sha256_hex(&bytes)).unwrap();
        let findings = check_at(&root, &data);
        let real = findings.iter().find(|f| f.path.ends_with("real.toml")).unwrap();
        assert!(matches!(real.status, Status::Ok(_)), "{:?}", real.status);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// Findings for files under this project, for the tests that pass an
    /// explicit data dir: `$HOME/.openmax/hooks` is a real directory on a
    /// developer's machine and its files are not part of any assertion here.
    fn local_at(root: &Path, data: &Path) -> Vec<Finding> {
        check_at(root, data).into_iter().filter(|f| f.path.starts_with(root)).collect()
    }

    /// The repair command a diagnostic prints is copyable, and the path in it
    /// comes from a file the agent named. A hook called `gate$(cmd).toml` is a
    /// legal file: approved once, then broken, its fail-closed message prints
    /// `openmax --approve <path>`, and an unquoted path would run `$(cmd)`
    /// when pasted. Every copyable command in --check quotes the path.
    #[test]
    fn copyable_repair_commands_shell_quote_the_path() {
        let root = temp_project();
        let data = root.join("data");
        let evil = ".openmax/hooks/gate$(touch pwned).toml";
        let hook = root.join(evil);
        write(hook.clone(), "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n");
        let sha = crate::ledger::sha256_hex(&std::fs::read(&hook).unwrap());
        crate::ledger::approve_capability(&data, &root, &hook, &[sha]).unwrap();
        write(hook.clone(), "event = \"pre_tool_use\ncommand = broken");
        let reason = match &find(&local_at(&root, &data), "gate$(touch pwned)").status {
            Status::Err(reason) => reason.clone(),
            other => panic!("a broken live hook must err: {other:?}"),
        };
        // The metacharacters are inside single quotes, so a paste is inert.
        assert!(
            reason.contains("'") && reason.contains("$(touch pwned)"),
            "the path must appear, single-quoted: {reason}"
        );
        assert!(
            !reason.contains("approve /") && !reason.contains(".toml`"),
            "the raw unquoted path must not sit in a copyable command: {reason}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// The inline-program warning's repair must name a destination that makes
    /// sense: testing caught it prescribing "move the program into
    /// data/sales.csv" - the data file the program reads - because the repair
    /// clause reused the read target as the destination.
    #[test]
    fn the_inline_program_repair_names_a_project_file_not_the_read_target() {
        let root = temp_project();
        write(root.join("data/sales.csv"), "a,b\n1,2\n");
        write(
            root.join(".openmax/tools/inline.toml"),
            "name = \"inline\"\ndescription = \"d\"\ncommand = \"python3\"\nargs = [\"-c\", \"print(open('data/sales.csv').read())\"]\n",
        );
        let findings = local(&root);
        let warn = statuses_of(&findings, "inline.toml")
            .into_iter()
            .find_map(|s| match s {
                Status::Warn(reason) if reason.contains("inline program") => Some(reason.clone()),
                _ => None,
            })
            .expect("the inline read must warn");
        assert!(
            warn.contains("put the program in a project file"),
            "the repair must name a sensible destination: {warn}"
        );
        assert!(
            !warn.contains("move the program into"),
            "the repair must not name the read target as the destination: {warn}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A hook file that was approved and live and then stops parsing is the
    /// runtime failing every tool call closed (`invalid` in hooks.rs).
    /// Reporting the parse error alone reads as ordinary bookkeeping while
    /// the session refuses to run every call; the same broken bytes at a path
    /// no human approved really are ordinary, and must not borrow the claim.
    #[test]
    fn a_broken_once_live_hook_file_names_its_consequence() {
        let root = temp_project();
        let data = root.join("data");
        let hook = root.join(".openmax/hooks/gate.toml");
        write(hook.clone(), "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n");
        let sha = crate::ledger::sha256_hex(&std::fs::read(&hook).unwrap());
        crate::ledger::approve_capability(&data, &root, &hook, &[sha]).unwrap();
        // Broken in place: the bytes no longer parse, the approval stands.
        write(hook.clone(), "event = \"pre_tool_use\ncommand = broken");
        let findings = local_at(&root, &data);
        match &find(&findings, "gate.toml").status {
            Status::Err(reason) => {
                assert!(
                    reason.contains("every tool call fails closed"),
                    "the report must say what the session is doing: {reason}"
                );
                assert!(reason.contains("--approve"), "{reason}");
            }
            other => panic!("a broken live hook must err with its consequence: {other:?}"),
        }

        // The same broken bytes never approved never loaded, so they block
        // nothing and the finding must not claim they do.
        write(
            root.join(".openmax/hooks/scratch.toml"),
            "event = \"pre_tool_use\ncommand = broken",
        );
        let findings = local_at(&root, &data);
        match &find(&findings, "scratch.toml").status {
            Status::Err(reason) => {
                assert!(!reason.contains("fails closed"), "an inert file must stay ordinary: {reason}")
            }
            other => panic!("broken bytes are still an error: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// Only `openmax --approve`, run outside a session, activates a hook:
    /// `retain_approved` drops everything else, and the in-session write card
    /// shows a preview rather than the bytes. Offering that write approval as
    /// an alternative would send a human to an act that leaves the hook inert.
    #[test]
    fn an_unapproved_hook_names_the_only_approval_that_activates_it() {
        let root = temp_project();
        let data = root.join("data");
        write(
            root.join(".openmax/hooks/gate.toml"),
            "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\n",
        );

        let findings = local_at(&root, &data);
        match &find(&findings, "gate.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("openmax --approve"), "{reason}");
                assert!(reason.contains("run outside a session"), "{reason}");
                assert!(
                    reason.contains("approves the write and nothing more"),
                    "an in-session write approval never activates a hook: {reason}"
                );
            }
            other => panic!("an unapproved hook must read as inert: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// A hook whose command does not resolve cannot be approved at all:
    /// `openmax --approve` refuses a manifest whose code it cannot read. So
    /// the report has to name the file, or it prescribes the one command that
    /// answers with a different diagnosis. The three ways a command fails to
    /// resolve are three different repairs, and only one of them is "create
    /// it": telling a human to create a file that already exists is the same
    /// wrong instruction in a new place.
    #[test]
    fn a_hook_whose_command_does_not_resolve_names_the_repair_that_fits() {
        let root = temp_project();
        let data = root.join("data");
        write(
            root.join(".openmax/hooks/gate.toml"),
            "event = \"pre_tool_use\"\ncommand = \"./gate.sh\"\n",
        );
        write(
            root.join(".openmax/hooks/unexec.toml"),
            "event = \"pre_tool_use\"\ncommand = \"./unexec.sh\"\n",
        );
        write(
            root.join(".openmax/hooks/ghost.toml"),
            "event = \"pre_tool_use\"\ncommand = \"openmax-nonexistent-binary\"\n",
        );
        // Readable, so `--approve` would take it; not executable, so it still
        // cannot spawn.
        write(root.join("unexec.sh"), "#!/bin/sh\ntrue\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                root.join("unexec.sh"),
                std::fs::Permissions::from_mode(0o644),
            )
            .unwrap();
        }
        // The premise for the other two: no bytes to bless, which is exactly
        // what `--approve` refuses on.
        for command in ["./gate.sh", "openmax-nonexistent-binary"] {
            assert!(
                crate::ledger::bound_code(command, &[], &root)
                    .iter()
                    .any(|c| c.sha256.is_none()),
                "a command that resolves to nothing must bind no approvable code: {command}"
            );
        }

        let findings = local_at(&root, &data);
        match &find(&findings, "gate.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("./gate.sh"), "{reason}");
                assert!(reason.contains("does not exist from the project root"), "{reason}");
                assert!(reason.contains("create it"), "{reason}");
                assert!(
                    !reason.contains("its content is not approved"),
                    "approval is not the missing step here: {reason}"
                );
            }
            other => panic!("a hook with no code must name the missing file: {other:?}"),
        }
        // Executability is a unix mode bit; nothing else reads one.
        #[cfg(unix)]
        match &find(&findings, "unexec.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("exists but is not executable"), "{reason}");
                assert!(reason.contains("chmod +x './unexec.sh'"), "{reason}");
                assert!(
                    !reason.contains("create it"),
                    "the file exists; creating it is not the repair: {reason}"
                );
            }
            other => panic!("an unexecutable script must name its own repair: {other:?}"),
        }
        match &find(&findings, "ghost.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("is not on PATH"), "{reason}");
                assert!(reason.contains("install it"), "{reason}");
                assert!(reason.contains("refuses it as it stands"), "{reason}");
                assert!(
                    !reason.contains("create it"),
                    "a PATH name is not created in the project: {reason}"
                );
            }
            other => panic!("a command missing from PATH must name its own repair: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// Both loaders clamp `timeout_secs` silently, and for a gate the clamp is
    /// policy: it decides when the hook starts blocking calls. A report that
    /// says "ok" leaves the author believing the budget they wrote.
    #[test]
    fn a_timeout_outside_the_documented_range_is_reported_not_silently_clamped() {
        let root = temp_project();
        let data = root.join("data");
        write(
            root.join(".openmax/tools/slow.toml"),
            "name = \"slow\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\ntimeout_secs = 600\n",
        );
        write(
            root.join(".openmax/hooks/gate.toml"),
            "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\ntimeout_secs = 600\n",
        );
        write(
            root.join(".openmax/hooks/watch.toml"),
            "event = \"turn_end\"\ncommand = \"/bin/echo\"\ntimeout_secs = 5\n",
        );

        let findings = local_at(&root, &data);
        let clamped = |needle: &str| {
            findings.iter().find(|f| {
                f.path.to_string_lossy().contains(needle)
                    && f.status.summary().contains("timeout_secs")
            })
        };
        match clamped("slow.toml").map(|f| &f.status) {
            Some(Status::Warn(reason)) => {
                assert!(reason.contains("timeout_secs = 600"), "{reason}");
                assert!(reason.contains("1..300"), "{reason}");
                assert!(reason.contains("clamped to 300 seconds"), "{reason}");
            }
            other => panic!("a tool's clamped timeout must be reported: {other:?}"),
        }
        match clamped("gate.toml").map(|f| &f.status) {
            Some(Status::Warn(reason)) => {
                assert!(reason.contains("timeout_secs = 600"), "{reason}");
                assert!(reason.contains("1..60"), "{reason}");
                assert!(reason.contains("clamped to 60 seconds"), "{reason}");
                assert!(reason.contains("a gate that times out blocks"), "{reason}");
            }
            other => panic!("a gate's clamped timeout must be reported: {other:?}"),
        }
        // A value the loader honours is not worth a line.
        assert!(clamped("watch.toml").is_none(), "{findings:?}");
        let _ = std::fs::remove_dir_all(root);
    }

    /// `--check` classifies hooks so a human can see which files can refuse
    /// something. A `turn_end` hook with `blocking = true` can, so every place
    /// the report tells a gate from an observer has to ask the hook and not
    /// its event: the shape it names, the clamp that decides when it starts
    /// refusing, and the fail-closed line it earns once a human has approved
    /// it and the content moves.
    #[test]
    fn check_names_a_blocking_turn_end_hook_as_a_gate() {
        let root = temp_project();
        let data = root.join("data");
        let gate = root.join(".openmax/hooks/verify.toml");
        let gate_body =
            "event = \"turn_end\"\nblocking = true\ncommand = \"/bin/echo\"\ntimeout_secs = 600\n";
        write(gate.clone(), gate_body);
        write(
            root.join(".openmax/hooks/watch.toml"),
            "event = \"turn_end\"\ncommand = \"/bin/echo\"\ntimeout_secs = 600\n",
        );
        crate::ledger::approve_capability(
            &data,
            &root,
            &gate,
            &[crate::ledger::sha256_hex(gate_body.as_bytes())],
        )
        .unwrap();

        let findings = local_at(&root, &data);
        let clamped = |needle: &str| {
            findings
                .iter()
                .find(|f| {
                    f.path.to_string_lossy().contains(needle)
                        && f.status.summary().contains("timeout_secs")
                })
                .map(|f| f.status.summary())
                .unwrap_or_else(|| panic!("no clamp finding for {needle}"))
        };
        assert!(
            clamped("verify.toml").contains("a gate that times out blocks"),
            "{}",
            clamped("verify.toml")
        );
        assert!(
            !clamped("watch.toml").contains("a gate that times out blocks"),
            "an observer's clamp is not a policy change: {}",
            clamped("watch.toml")
        );
        // The shape it is, not the event it rides on.
        assert!(
            matches!(&find(&findings, "verify.toml").status, Status::Ok(s) if s == "hook on blocking turn_end"),
            "{:?}",
            find(&findings, "verify.toml").status
        );
        // The observer shape says the one thing separating it from the gate
        // its author may have meant: weaker models consistently
        // wrote exactly this file while stating nonzero exit would block. It
        // must ride the message on both sides of approval, because the
        // authoring model only ever reads the unapproved one.
        assert!(
            matches!(&find(&findings, "watch.toml").status,
                Status::Err(s) if s.contains("exit status is ignored") && s.contains("blocking = true")),
            "{:?}",
            find(&findings, "watch.toml").status
        );
        let watch = root.join(".openmax/hooks/watch.toml");
        let watch_body = std::fs::read(&watch).unwrap();
        crate::ledger::approve_capability(
            &data,
            &root,
            &watch,
            &[crate::ledger::sha256_hex(&watch_body)],
        )
        .unwrap();
        assert!(
            matches!(&find(&local_at(&root, &data), "watch.toml").status,
                Status::Ok(s) if s.contains("exit status is ignored") && s.contains("blocking = true")),
            "{:?}",
            find(&local_at(&root, &data), "watch.toml").status
        );

        // Approved, then rewritten without the word: a gate a human installed
        // that no longer gates, which is the loop refusing every tool call.
        write(gate.clone(), "event = \"turn_end\"\ncommand = \"/bin/echo\"\n");
        match &find(&local_at(&root, &data), "verify.toml").status {
            Status::Err(reason) => {
                assert!(reason.contains("blocking turn_end"), "{reason}");
                assert!(reason.contains("stop it gating"), "{reason}");
                assert!(reason.contains("every tool call fails closed"), "{reason}");
            }
            other => panic!("a demoted turn_end gate must read as a live gate: {other:?}"),
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// A clamp warning is a claim about runtime behavior, so it belongs only
    /// to the definitions the loader actually keeps. A shadowed file and a
    /// file past the cap are reported as shadowed and capped; a second line
    /// about the timeout they will never serve would read as if they ran.
    #[test]
    fn a_clamp_warning_is_never_reported_for_a_file_that_never_loads() {
        let root = temp_project();
        let data = root.join("data");
        // Same tool name in both tiers: the project file wins, so the global
        // one is shadowed and its timeout is nothing the loader ever sees.
        write(
            data.join("tools/dup.toml"),
            "name = \"dup\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\ntimeout_secs = 600\n",
        );
        write(
            root.join(".openmax/tools/dup.toml"),
            "name = \"dup\"\ndescription = \"d\"\ncommand = \"/bin/sh\"\ntimeout_secs = 10\n",
        );
        // One hook past the per-event cap, and the rest inside it.
        for i in 0..(crate::hooks::MAX_HOOKS_PER_EVENT + 1) {
            write(
                root.join(format!(".openmax/hooks/gate-{i:03}.toml")),
                "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\ntimeout_secs = 600\n",
            );
        }

        let findings = local_at(&root, &data);
        let clamp_line = |needle: &str| {
            findings.iter().find(|f| {
                f.path.to_string_lossy().contains(needle)
                    && f.status.summary().contains("timeout_secs")
            })
        };
        let shadowed = find(&findings, "data/tools/dup.toml");
        assert!(
            matches!(&shadowed.status, Status::Warn(r) if r.contains("shadowed by")),
            "{:?}",
            shadowed.status
        );
        assert!(
            clamp_line("data/tools/dup.toml").is_none(),
            "a shadowed tool serves no timeout: {:?}",
            clamp_line("data/tools/dup.toml").map(|f| f.status.summary())
        );
        let capped = format!("gate-{:03}.toml", crate::hooks::MAX_HOOKS_PER_EVENT);
        assert!(
            matches!(&find(&findings, &capped).status, Status::Err(r) if r.contains("never loads")),
            "{:?}",
            find(&findings, &capped).status
        );
        assert!(
            clamp_line(&capped).is_none(),
            "a hook past the cap serves no timeout: {:?}",
            clamp_line(&capped).map(|f| f.status.summary())
        );
        // The files that do load still say what they were going to say.
        assert!(
            clamp_line(".openmax/hooks/gate-000.toml").is_some(),
            "a hook inside the cap still reports its clamp"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// The clamp diagnostic describes the bytes it was handed, which are the
    /// bytes the finding's own parse used. Recovering the written value from a
    /// second read of the path would leave an interval a rewrite can land in,
    /// and one finding would then describe two generations of a file.
    #[test]
    fn a_clamp_diagnostic_describes_the_bytes_it_was_given() {
        let root = temp_project();
        let path = root.join(".openmax/hooks/gate.toml");
        // What is on disk now: the generation a rewrite left behind.
        write(
            path.clone(),
            "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\ntimeout_secs = 5\n",
        );
        // What the caller parsed a moment earlier.
        let parsed = "event = \"pre_tool_use\"\ncommand = \"/bin/echo\"\ntimeout_secs = 600\n";

        let reason = clamped_timeout_reason(
            parsed,
            "hook",
            crate::hooks::MIN_TIMEOUT_SECS,
            crate::hooks::MAX_TIMEOUT_SECS,
        )
        .expect("the parsed bytes ask for 600");
        assert!(reason.contains("timeout_secs = 600"), "{reason}");
        assert!(reason.contains("clamped to 60 seconds"), "{reason}");
        // And the generation on disk, judged on its own bytes, says nothing.
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert!(
            clamped_timeout_reason(
                &on_disk,
                "hook",
                crate::hooks::MIN_TIMEOUT_SECS,
                crate::hooks::MAX_TIMEOUT_SECS,
            )
            .is_none(),
            "an in-range timeout is not a finding"
        );
        let _ = std::fs::remove_dir_all(root);
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

    /// settings.json is launch-read and fail-closed, so --check must name a
    /// file that will refuse the next launch - it is how the drift receipt's
    /// "fix it now" gets confirmed. The gap this pins: a bricking file
    /// got a silent exit 0 from --check while -p exited 2 on it.
    #[test]
    fn check_names_a_settings_file_that_would_brick_the_next_launch() {
        let root = temp_project();
        let data = temp_project();
        std::fs::write(data.join("settings.json"), "{ invalid json broken").unwrap();
        let findings = check_at(&root, &data);
        let s = findings.iter().find(|f| f.kind == "settings").expect("settings is checked");
        assert!(matches!(s.status, Status::Err(_)), "{:?}", s.status);
        assert!(s.status.summary().contains("exit 2"), "{}", s.status.summary());

        std::fs::write(
            data.join("settings.json"),
            r#"{"base_url":"http://x/v1","model":"m","approval_mode":"auto","context_tokens":32768}"#,
        )
        .unwrap();
        let findings = check_at(&root, &data);
        let s = findings.iter().find(|f| f.kind == "settings").unwrap();
        assert!(matches!(s.status, Status::Ok(_)), "{:?}", s.status);
        assert!(s.status.summary().contains("read at launch"), "{}", s.status.summary());
        assert!(s.status.summary().contains("32768-token window"), "{}", s.status.summary());

        // Parses, launches, and then refuses every turn: the window is not
        // guessed, so a file without one is named here rather than at the
        // first prompt.
        std::fs::write(
            data.join("settings.json"),
            r#"{"base_url":"http://x/v1","model":"m","approval_mode":"auto"}"#,
        )
        .unwrap();
        let findings = check_at(&root, &data);
        let s = findings.iter().find(|f| f.kind == "settings").unwrap();
        assert!(matches!(s.status, Status::Warn(_)), "{:?}", s.status);
        assert!(s.status.summary().contains("context_tokens"), "{}", s.status.summary());
        assert!(s.status.summary().contains("every turn will refuse"), "{}", s.status.summary());
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

    /// A typo'd providers.json key deserializes cleanly and configures
    /// nothing; before this it read as "ok (1 providers)" while the models
    /// list it was meant to be stayed silently empty.
    #[test]
    fn unknown_provider_keys_surface_as_warnings_not_health() {
        let root = temp_project();
        let data = temp_project();
        std::fs::write(
            data.join("providers.json"),
            r#"{"providers":{"local":{"base_url":"http://127.0.0.1:11434/v1","modles":[]}}}"#,
        )
        .unwrap();

        let findings = check_at(&root, &data);
        let warn = findings
            .iter()
            .find(|f| f.kind == "providers" && matches!(f.status, Status::Warn(_)))
            .expect("an ignored key must be reported");
        assert!(warn.status.summary().contains("'modles'"), "{}", warn.status.summary());
        assert!(warn.status.summary().contains("did you mean 'models'"), "{}", warn.status.summary());
        // The file still loads, so the count line stays and nothing errors.
        assert!(findings
            .iter()
            .any(|f| f.kind == "providers" && f.status.summary() == "1 providers"));
        assert!(!has_errors(&findings));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }

    /// The report a project gets when every extension it wrote is at a path
    /// nothing reads. This used to be "no extension files found", exit 0.
    #[test]
    fn an_inert_allow_file_does_not_summarize_as_live_rules() {
        // The warn names the inert allow, but the Ok summary prints after it
        // and "(2 rules)" alone closed the file's story claiming both rules
        // are in force; the truth is one deny in force, one allow inert.
        let root = temp_project();
        let data = root.join("data");
        write(
            root.join(".openmax/permissions.toml"),
            "[[rules]]\neffect = \"allow\"\ntool = \"bash\"\narg_regex = \"^git status\"\n\n[[rules]]\neffect = \"deny\"\ntool = \"bash\"\narg_regex = \"rm -rf\"\n",
        );
        let findings = check_at(&root, &data);
        let summary = findings
            .iter()
            .filter(|f| f.kind == "permissions")
            .find_map(|f| match &f.status {
                Status::Ok(s) => Some(s.clone()),
                _ => None,
            })
            .expect("the file draws a summary row");
        assert_eq!(summary, "2 rules, 1 inert until approved", "the summary counts what is in force");
        assert!(
            findings.iter().any(|f| matches!(&f.status, Status::Warn(w) if w.contains("inert"))),
            "the warn still names the inert allow"
        );
        let _ = std::fs::remove_dir_all(root);
    }

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
        // The repair pointer is runnable as printed: bare `openmax --spec`
        // errors with the full help dump, so a surface token must follow
        // (the sibling warnings already name exact targets).
        assert!(
            find(&findings, ".openmax/deploy.toml")
                .status
                .summary()
                .contains("openmax --spec tools"),
            "the pointer names a runnable command"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_empty_misspelled_directory_is_not_worth_reporting() {
        let root = temp_project();
        std::fs::create_dir_all(root.join(".openmax/tool")).unwrap();
        assert!(local(&root).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    /// The global `~/.openmax/` tier gets the same misplaced-file net as the
    /// project tier: a wrong-extension file in a canonical dir and a near-miss
    /// dir are named, while the data dir's legitimate files and dirs (a sibling
    /// script, ledger/, sessions/, a loose file) stay silent. Uses a custom
    /// data dir so the real `~/.openmax` is never read.
    #[test]
    fn check_covers_the_global_tier() {
        let root = temp_project();
        let data = root.join("data");
        write(data.join("tools/gh.yaml"), "name: gh\n");
        write(data.join("prompt/review.md"), "Review.\n");
        write(data.join("tools/deploy.sh"), "#!/bin/sh\ntrue\n");
        write(data.join("ledger/log.jsonl"), "{}\n");
        write(data.join("sessions/index.json"), "{}\n");
        write(data.join("notes.txt"), "scratch\n");

        let findings: Vec<Finding> = check_at(&root, &data)
            .into_iter()
            .filter(|f| f.path.starts_with(&data))
            .collect();
        assert!(
            find(&findings, "gh.yaml").status.summary().contains(".toml only"),
            "a wrong-extension global tool must be named: {}",
            find(&findings, "gh.yaml").status.summary()
        );
        assert!(
            find(&findings, "data/prompt").status.summary().contains("prompts/"),
            "a near-miss global dir must point at the real one: {}",
            find(&findings, "data/prompt").status.summary()
        );
        for legit in ["deploy.sh", "/ledger", "/sessions", "notes.txt"] {
            assert!(
                !findings.iter().any(|f| f.path.to_string_lossy().contains(legit)),
                "{legit} is legitimate and must not warn: {findings:?}"
            );
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// Memory is the one singular surface among four plural siblings, so the
    /// natural miswrite is the plural `memories`. That is three edits from
    /// `memory`, past what `near` reaches, so before this it was the lone
    /// misplaced surface `--check` reported as a clean bill of health: the
    /// note never loads and nothing says why.
    #[test]
    fn a_memory_dir_written_with_the_plural_name_is_named() {
        let root = temp_project();
        write(root.join(".openmax/memories/deploy-port.md"), "# The deploy port is 7443\nbody\n");

        let findings = local(&root);
        let warn = find(&findings, ".openmax/memories");
        assert!(
            warn.status.summary().contains(".openmax/memory/"),
            "the plural must be pointed at the real dir: {}",
            warn.status.summary()
        );
        assert!(!has_errors(&findings), "a guess about intent must not fail the run");
        let _ = std::fs::remove_dir_all(root);
    }

    /// A single-character memory typo, and memory under the wrong parent, both
    /// resolve to the real dir now that memory is a recognized surface.
    #[test]
    fn a_memory_typo_and_a_wrong_parent_both_point_at_the_real_dir() {
        let root = temp_project();
        write(root.join(".openmax/memry/a.md"), "# fact\nb\n");
        write(root.join(".agents/memory/b.md"), "# fact\nb\n");

        let findings = local(&root);
        assert!(find(&findings, ".openmax/memry").status.summary().contains(".openmax/memory/"));
        assert!(find(&findings, ".agents/memory").status.summary().contains(".openmax/memory/"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// Making memory a recognized surface must not turn a healthy memory dir
    /// into noise: a valid note reports only its own `memory` finding, and the
    /// internal `.access.jsonl` bookkeeping file is never flagged as unread.
    #[test]
    fn a_healthy_memory_dir_is_not_reported_as_unread() {
        let root = temp_project();
        write(root.join(".openmax/memory/deploy-port.md"), "# The deploy port is 7443\nbody\n");
        write(root.join(".openmax/memory/.access.jsonl"), "{}\n");

        let findings = local(&root);
        assert!(
            !findings
                .iter()
                .any(|f| f.kind == "path" && f.path.to_string_lossy().contains("memory")),
            "a healthy memory dir must not draw an unread-path warning: {findings:?}"
        );
        assert!(
            !findings.iter().any(|f| matches!(f.status, Status::Warn(_) | Status::Err(_))
                && f.path.to_string_lossy().contains(".access.jsonl")),
            "the internal access log is not a memory note and must not warn: {findings:?}"
        );
        assert!(
            findings.iter().any(|f| f.kind == "memory" && matches!(f.status, Status::Ok(_))),
            "the valid note should still report as a healthy memory: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn regular_plural_bridges_the_ies_gap_near_cannot() {
        // The case near() misses (three edits), which is why this exists.
        assert!(is_regular_plural("memory", "memories"));
        assert!(!near("memory", "memories"));
        // Regular +s / +es, and non-plurals.
        assert!(is_regular_plural("hook", "hooks"));
        assert!(is_regular_plural("box", "boxes"));
        assert!(!is_regular_plural("memory", "memory"));
        assert!(!is_regular_plural("memory", "memoried"));
        // A vowel before `y` takes `s`, not `ies`, so this is not a plural.
        assert!(!is_regular_plural("day", "daies"));
        assert!(is_regular_plural("day", "days"));
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

    /// The harness reads two formats, so a stray `.toml` where `.md` is read (a
    /// prompt written as a tool) is named. A stray `.md` where `.toml` is read
    /// is left alone: a README belongs in a tool directory as easily as a
    /// misplaced surface file would.
    #[test]
    fn a_toml_where_md_is_read_is_named_but_a_readme_is_not() {
        let root = temp_project();
        write(root.join(".agents/prompts/review.toml"), "x = 1\n");
        write(root.join(".openmax/tools/README.md"), "# docs\n");

        let findings = local(&root);
        assert!(
            find(&findings, "review.toml").status.summary().contains(".md only"),
            "a prompt written as .toml must be named: {}",
            find(&findings, "review.toml").status.summary()
        );
        assert!(
            !findings.iter().any(|f| f.path.to_string_lossy().contains("README.md")),
            "a README in a tool dir must not warn: {findings:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// Finder drops .DS_Store into every directory it opens; a dotfile can
    /// never be a skill, so reporting one as a misplaced skill is permanent
    /// noise on every macOS machine. A visible stray file still warns: that
    /// one may really be a skill written to the wrong place.
    #[test]
    fn a_skill_dir_dotfile_is_not_a_finding_but_a_visible_stray_is() {
        let root = temp_project();
        write(root.join(".agents/skills/.DS_Store"), "junk");
        write(root.join(".agents/skills/README.md"), "docs\n");

        let findings = local(&root);
        assert!(
            !findings.iter().any(|f| f.path.to_string_lossy().contains(".DS_Store")),
            "{findings:?}"
        );
        assert!(find(&findings, "README.md").status.summary().contains("SKILL.md"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// The exemption keys on the leading byte, not on the name being valid
    /// UTF-8: `.` followed by invalid bytes is still a dotfile. Filesystems
    /// that refuse such names skip the scenario by construction.
    #[cfg(unix)]
    #[test]
    fn a_non_utf8_dotfile_is_still_exempt() {
        use std::os::unix::ffi::OsStrExt;
        let root = temp_project();
        let dir = root.join(".agents/skills");
        std::fs::create_dir_all(&dir).unwrap();
        let name = std::ffi::OsStr::from_bytes(b".\xffhidden");
        if std::fs::write(dir.join(name), "junk").is_ok() {
            let findings = local(&root);
            assert!(
                !findings.iter().any(|f| f.path.to_string_lossy().contains("hidden")),
                "{findings:?}"
            );
        }
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
        // The built-in and the project's own tool are both known. The file's
        // `allow` rule is unapproved, so it warns too; that is the only other
        // warning this file may produce.
        let warnings: Vec<&str> = findings
            .iter()
            .filter(|f| f.kind == "permissions" && matches!(f.status, Status::Warn(_)))
            .map(|f| f.status.summary())
            .collect();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(
            warnings.iter().filter(|w| w.contains("allow rule(s) are inert")).count() == 1,
            "{warnings:?}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    /// A tool that parses loads into the registry whether or not its command
    /// exists yet (the missing script only matters at spawn time), so a rule
    /// or hook filter naming it matches at runtime. Before this, --check said
    /// "no tool named 'ghost' exists" about a tool `--spec usage` listed as
    /// installed, inviting the user to delete a live deny rule.
    #[test]
    fn a_warned_tool_is_still_a_known_name_for_rules_and_filters() {
        let root = temp_project();
        write(
            root.join(".openmax/tools/ghost.toml"),
            "name = \"ghost\"\ndescription = \"d\"\ncommand = \"./scripts/ghost.sh\"\n",
        );
        write(
            root.join(".openmax/permissions.toml"),
            "[[rules]]\neffect = \"deny\"\ntool = \"ghost\"\n",
        );
        write(
            root.join(".openmax/hooks/gate.toml"),
            "event = \"pre_tool_use\"\ncommand = \"/bin/sh\"\ntool = \"ghost\"\n",
        );

        let findings = local(&root);
        // The tool itself still warns about the command it cannot spawn.
        assert!(matches!(find(&findings, "ghost.toml").status, Status::Warn(_)));
        assert!(
            !findings.iter().any(|f| f.status.summary().contains("no tool named 'ghost'")),
            "a live tool's name must resolve for rules and hook filters: {findings:?}"
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
        assert!(live.status.summary().contains("indexed at the next prompt freeze"));

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
