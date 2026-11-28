//! `/gate`: the project completion gate of `examples/done-gate`, installed,
//! shown and removed from the terminal UI. The command writes that example's
//! two files, its script byte for byte and its manifest with the check in the
//! `args` line, so the example and the command are one mechanism.

use std::path::{Path, PathBuf};

use open_max_core::config::ApprovalMode;
use ratatui::style::Style;
use ratatui::text::{Line, Span};

use crate::theme;

/// The hook's name, as the refusal note and `turn_verified` give it.
pub const NAME: &str = "done-gate";
const MANIFEST: &str = ".openmax/hooks/done-gate.toml";
const SCRIPT_PATH: &str = ".openmax/hooks/done-gate.sh";
const SCRIPT: &str = include_str!("../../../examples/done-gate/done-gate.sh");
const EXAMPLE: &str = include_str!("../../../examples/done-gate/done-gate.toml");
/// The example's check, which the command replaces with the one it is given.
const EXAMPLE_ARGS: &str = "args = [\".openmax/hooks/done-gate.sh\", \"cargo test\"]";
const ARGS: &str = "args = [\".openmax/hooks/done-gate.sh\", ";
/// What the command's manifest says in place of the example's opening
/// comment, which tells a reader to copy the files by hand. It is also how the
/// command knows a manifest is its own.
const HEADER: &str = "\
# A completion gate: Open Max refuses the end of a turn while the check fails.
# /gate wrote this file and done-gate.sh from examples/done-gate. /gate <check>
# replaces the check below, and /gate off removes both files. The check runs
# with sh -c in the project root.
";

/// What `/gate` needs from the app, read before the work leaves the event
/// loop for a blocking thread.
pub struct Context {
    pub project: PathBuf,
    pub data_dir: PathBuf,
    pub mode: ApprovalMode,
    /// The session a person approves the gate in, if one exists yet.
    pub session: Option<String>,
    /// Whether a person may approve here: the walls `openmax --approve`
    /// stands behind, or why this process cannot.
    pub human: Result<(), String>,
    /// How the gate judged this session's latest end of a turn, if it did.
    pub last: Option<&'static str>,
}

/// The manifest `/gate <check>` writes: the example's, under the command's
/// own header, with `check` in place of the example's.
pub fn manifest(check: &str) -> String {
    let body: String = EXAMPLE.lines().skip_while(|line| line.starts_with('#')).map(|line| format!("{line}\n")).collect();
    // A JSON string is a TOML basic string for every value serde_json writes.
    let quoted = serde_json::to_string(check).unwrap_or_default();
    format!("{HEADER}{}", body.replacen(EXAMPLE_ARGS, &format!("{ARGS}{quoted}]"), 1))
}

/// The check of a manifest the command wrote, None for any other bytes: a
/// file someone wrote or edited by hand is theirs, and the command leaves it.
fn check_of(text: &str) -> Option<String> {
    let line = text.lines().find(|line| line.starts_with(ARGS))?;
    let check: String = serde_json::from_str(line.strip_prefix(ARGS)?.strip_suffix(']')?).ok()?;
    (manifest(&check) == text).then_some(check)
}

/// The command a person runs in a terminal to approve the gate.
pub fn approve_command(project: &Path) -> String {
    format!("cd {} && openmax --approve {MANIFEST}", open_max_core::doctor::shell_quote(project))
}

fn forget_command(project: &Path) -> String {
    format!("cd {} && openmax --forget {MANIFEST}", open_max_core::doctor::shell_quote(project))
}

/// Run `/gate` with `arg`: show the gate, install one for a check, or
/// remove the one the command installed.
pub fn run(ctx: &Context, arg: &str) -> Vec<Line<'static>> {
    let result = match arg {
        "" => Ok(status(ctx)),
        "off" => off(ctx),
        check => install(ctx, check),
    };
    match result {
        Ok(lines) => note(&lines),
        Err(why) => vec![Line::from(Span::styled(format!("✗ {why}"), Style::default().fg(theme::ERR)))],
    }
}

fn note(lines: &[String]) -> Vec<Line<'static>> {
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let mark = if i == 0 { "• " } else { "  " };
            Line::from(vec![
                Span::styled(mark, Style::default().fg(theme::ACCENT)),
                Span::styled(line.clone(), theme::DIM_ITALIC),
            ])
        })
        .collect()
}

/// What is at `path`: None when nothing is, the bytes when they read.
fn read(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("cannot read {}: {e}", path.display())),
    }
}

fn not_ours(path: &str) -> String {
    format!("{path} is a hook file /gate did not write, or one edited since; /gate leaves it as it is (edit it, or remove it and run /gate again)")
}

/// Whether a person approved a hook at `path`, so that outside auto its
/// absence or a change to it fails closed.
fn was_approved(ctx: &Context, path: &Path) -> Result<bool, String> {
    open_max_core::ledger::approvals(&ctx.data_dir, &ctx.project)
        .map(|approvals| approvals.was_live(path))
        .map_err(|why| format!("cannot read the capability ledger, so /gate cannot tell what changing the gate would do; nothing was changed: {why}"))
}

fn install(ctx: &Context, check: &str) -> Result<Vec<String>, String> {
    let (manifest_path, script_path) = (ctx.project.join(MANIFEST), ctx.project.join(SCRIPT_PATH));
    let text = manifest(check);
    let current = read(&manifest_path)?;
    if current.as_deref().is_some_and(|current| check_of(current).is_none()) {
        return Err(not_ours(MANIFEST));
    }
    // An approved gate whose bytes change fails closed outside auto, so it
    // is changed only where the change can be approved in the same act.
    let changes = current.as_deref().is_some_and(|current| current != text);
    if let (Err(why), true) = (&ctx.human, ctx.mode != ApprovalMode::Auto && changes) {
        if was_approved(ctx, &manifest_path)? {
            return Err(format!("the gate here is approved and this process cannot approve a change to it, which would make every tool call fail closed; nothing was changed: {why}"));
        }
    }
    let script = read(&script_path)?;
    if script.as_deref().is_some_and(|text| text != SCRIPT) {
        return Err(not_ours(SCRIPT_PATH));
    }
    let hooks = manifest_path.parent().unwrap_or(&ctx.project);
    std::fs::create_dir_all(hooks).map_err(|e| format!("cannot create {}: {e}", hooks.display()))?;
    // The script first, so the manifest never names a file that is not there.
    if script.is_none() {
        open_max_core::sessions::write_atomic(&script_path, SCRIPT)?;
    }
    open_max_core::sessions::write_atomic(&manifest_path, &text)?;
    let mut lines = vec![
        format!("completion gate set: each turn's end now waits for `{check}` to pass"),
        format!("{MANIFEST} and done-gate.sh run it with sh -c in the project root; while it fails, its output goes back to the model, up to 3 times in a turn (max_refusals)"),
    ];
    if ctx.mode == ApprovalMode::Auto {
        lines.push("in auto it runs from the next turn without approval; /gate off removes it".into());
        return Ok(lines);
    }
    // Outside auto a hook is inert until a person approves its bytes. Typing
    // /gate here is that act when `openmax --approve` would take it as one,
    // and the approval goes through the same code.
    let approved = ctx.human.clone().and_then(|()| {
        let vouched = [open_max_core::ledger::sha256_hex(SCRIPT.as_bytes())];
        crate::approve_file(&ctx.data_dir, &ctx.project, &manifest_path, text.as_bytes(), Some(&vouched), ctx.session.as_deref())
    });
    match approved {
        Ok(approved) => {
            lines.push("approved its exact content, as openmax --approve would; it runs from the next turn".into());
            lines.extend(approved);
        }
        Err(why) => {
            lines.push(format!("not approved: {why}"));
            lines.push(format!("in {} it is inert until a person runs `{}` in a terminal", ctx.mode.as_str(), approve_command(&ctx.project)));
        }
    }
    Ok(lines)
}

fn off(ctx: &Context) -> Result<Vec<String>, String> {
    let (manifest_path, script_path) = (ctx.project.join(MANIFEST), ctx.project.join(SCRIPT_PATH));
    let Some(text) = read(&manifest_path)? else {
        return Ok(vec!["no completion gate from /gate in this project; nothing was removed".into()]);
    };
    let Some(check) = check_of(&text) else {
        return Err(not_ours(MANIFEST));
    };
    let approved = was_approved(ctx, &manifest_path)?;
    // A deleted gate a person approved fails closed outside auto: every tool
    // call is refused and every turn ends unverified until its approval is
    // retired, which only `openmax --forget` does.
    if approved && ctx.mode != ApprovalMode::Auto {
        return Err(format!(
            "the gate's approval is on record, and deleting an approved gate makes every tool call fail closed in {}; retire it first with `{}` in a terminal, then run /gate off",
            ctx.mode.as_str(),
            forget_command(&ctx.project)
        ));
    }
    std::fs::remove_file(&manifest_path).map_err(|e| format!("cannot remove {MANIFEST}: {e}"))?;
    let mut lines = vec![format!("completion gate removed: turns from the next one end without `{check}`")];
    // The script goes too unless another hook here still runs it.
    let named_elsewhere = std::fs::read_dir(manifest_path.parent().unwrap_or(&ctx.project))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "toml"))
        .any(|entry| std::fs::read_to_string(entry.path()).is_ok_and(|text| text.contains("done-gate.sh")));
    match read(&script_path)? {
        Some(script) if script == SCRIPT && !named_elsewhere => {
            std::fs::remove_file(&script_path).map_err(|e| format!("cannot remove {SCRIPT_PATH}: {e}"))?;
            lines.push(format!("removed {MANIFEST} and done-gate.sh"));
        }
        Some(_) => lines.push(format!("removed {MANIFEST}; {SCRIPT_PATH} stays, since another hook runs it or it was edited")),
        None => lines.push(format!("removed {MANIFEST}")),
    }
    if approved {
        lines.push(format!(
            "its approval is still on record: outside auto every tool call would fail closed until `{}` retires it",
            forget_command(&ctx.project)
        ));
    }
    Ok(lines)
}

fn status(ctx: &Context) -> Vec<String> {
    let manifest_path = ctx.project.join(MANIFEST);
    let text = match read(&manifest_path) {
        Ok(Some(text)) => text,
        Ok(None) => return vec!["no completion gate from /gate in this project; /gate <check> installs one, such as /gate cargo test".into()],
        Err(why) => return vec![why],
    };
    let mut lines = match check_of(&text) {
        Some(check) => vec![format!("completion gate: each turn's end waits for `{check}` to pass ({MANIFEST})")],
        None => vec![not_ours(MANIFEST)],
    };
    // The state `openmax --check` reports for the file, word for word.
    let same = |path: &Path| path.canonicalize().ok() == manifest_path.canonicalize().ok();
    let state = open_max_core::doctor::check(&ctx.project, &ctx.data_dir)
        .into_iter()
        .find(|finding| same(&finding.path))
        .map(|finding| format!("{}: {}", finding.status.as_str(), finding.status.summary()));
    lines.push(format!("mode {}; openmax --check reports it as {}", ctx.mode.as_str(), state.as_deref().unwrap_or("nothing (no hook row)")));
    if let Some(last) = ctx.last {
        lines.push(last.into());
    }
    lines.push("/gate <check> replaces the check, /gate off removes the gate".into());
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The command writes the example's manifest: the same keys and values,
    /// with the check in the one line that names it.
    #[test]
    fn the_command_writes_the_examples_manifest_with_its_check() {
        assert_eq!(EXAMPLE.matches(EXAMPLE_ARGS).count(), 1, "{EXAMPLE}");
        // The note the command prints names the example's cap.
        assert!(EXAMPLE.contains("\nmax_refusals = 3\n"), "{EXAMPLE}");
        let written = manifest("make test && echo \"ok\"");
        let keys = |text: &str| text.lines().filter(|line| !line.starts_with('#') && !line.starts_with(ARGS)).map(str::to_string).collect::<Vec<_>>();
        assert_eq!(keys(&written), keys(EXAMPLE));
        assert!(written.contains(r#"args = [".openmax/hooks/done-gate.sh", "make test && echo \"ok\""]"#), "{written}");
        assert_eq!(check_of(&written).as_deref(), Some("make test && echo \"ok\""));
        // The example itself, copied by hand, is not the command's.
        assert_eq!(check_of(EXAMPLE), None);
        assert_eq!(check_of(&written.replace("max_refusals = 3", "max_refusals = 5")), None);
    }

    fn context(mode: ApprovalMode) -> Context {
        let root = crate::test_temp_dir("openmax-gate");
        let (project, data_dir) = (root.join("project"), root.join("data"));
        std::fs::create_dir_all(&project).unwrap();
        let human = Err("this process was started from an agent session".to_string());
        Context { project, data_dir, mode, session: None, human, last: None }
    }

    fn text(lines: &[Line<'_>]) -> String {
        lines.iter().map(|line| line.spans.iter().map(|span| span.content.as_ref()).collect::<String>() + "\n").collect()
    }

    /// A hook file the command did not write is the person's: a copy of the
    /// example edited by hand is neither replaced nor removed, and neither is
    /// a script that is not the example's.
    #[test]
    fn gate_never_replaces_or_removes_a_hook_file_it_did_not_write() {
        let ctx = context(ApprovalMode::Auto);
        let hooks = ctx.project.join(".openmax/hooks");
        std::fs::create_dir_all(&hooks).unwrap();
        let own = EXAMPLE.replace("\"cargo test\"", "\"make check\"");
        open_max_core::sessions::write_atomic(&hooks.join("done-gate.toml"), &own).unwrap();
        for arg in ["npm test", "off"] {
            let shown = text(&run(&ctx, arg));
            assert!(shown.starts_with("✗ .openmax/hooks/done-gate.toml is a hook file /gate did not write"), "{shown}");
        }
        assert!(text(&run(&ctx, "")).contains("/gate did not write"));
        assert_eq!(std::fs::read_to_string(hooks.join("done-gate.toml")).unwrap(), own);
        assert!(!hooks.join("done-gate.sh").exists());
        std::fs::remove_file(hooks.join("done-gate.toml")).unwrap();
        open_max_core::sessions::write_atomic(&hooks.join("done-gate.sh"), "exit 0\n").unwrap();
        assert!(text(&run(&ctx, "npm test")).contains("done-gate.sh is a hook file /gate did not write"));
        assert!(!hooks.join("done-gate.toml").exists());
        std::fs::remove_dir_all(ctx.project.parent().unwrap()).unwrap();
    }

    /// /gate off removes the manifest it wrote, and the script only when no
    /// other hook here still runs it.
    #[test]
    fn gate_off_keeps_the_script_another_hook_runs() {
        let ctx = context(ApprovalMode::Auto);
        let hooks = ctx.project.join(".openmax/hooks");
        assert!(text(&run(&ctx, "npm test")).contains("completion gate set"));
        let other = "event = \"turn_end\"\nblocking = true\ncommand = \"sh\"\nargs = [\".openmax/hooks/done-gate.sh\", \"make lint\"]\n";
        open_max_core::sessions::write_atomic(&hooks.join("lint-gate.toml"), other).unwrap();
        let shown = text(&run(&ctx, "off"));
        assert!(shown.contains("done-gate.sh stays, since another hook runs it"), "{shown}");
        assert!(!hooks.join("done-gate.toml").exists() && hooks.join("done-gate.sh").exists());
        assert!(text(&run(&ctx, "off")).contains("nothing was removed"));
        std::fs::remove_dir_all(ctx.project.parent().unwrap()).unwrap();
    }

    /// Outside auto, where no person can approve (a session an agent
    /// started), the gate is written inert with the command that approves it,
    /// and a gate a person approved is not changed, since a change no one can
    /// approve fails every tool call closed.
    #[test]
    fn without_a_person_the_gate_is_written_inert_and_an_approved_one_is_left_alone() {
        let ctx = context(ApprovalMode::Ask);
        let manifest_path = ctx.project.join(MANIFEST);
        let shown = text(&run(&ctx, "npm test"));
        assert!(shown.contains("not approved: this process was started from an agent session"), "{shown}");
        assert!(shown.contains(&format!("in ask it is inert until a person runs `{}`", approve_command(&ctx.project))), "{shown}");
        let approvals = open_max_core::ledger::approvals(&ctx.data_dir, &ctx.project).unwrap();
        assert!(!approvals.was_live(&manifest_path), "nothing may be approved here");
        // A person approves it at a terminal; this process still cannot.
        let script = open_max_core::ledger::sha256_hex(SCRIPT.as_bytes());
        let written = std::fs::read(&manifest_path).unwrap();
        let shas = [open_max_core::ledger::sha256_hex(&written), script];
        open_max_core::ledger::approve_capability(&ctx.data_dir, &ctx.project, &manifest_path, &shas).unwrap();
        let shown = text(&run(&ctx, "true"));
        assert!(shown.contains("the gate here is approved and this process cannot approve a change"), "{shown}");
        assert_eq!(std::fs::read(&manifest_path).unwrap(), written);
        // Nor is it deleted, which would fail closed the same way.
        let shown = text(&run(&ctx, "off"));
        assert!(shown.contains("retire it first with `cd ") && shown.contains("openmax --forget .openmax/hooks/done-gate.toml"), "{shown}");
        assert!(manifest_path.exists());
        std::fs::remove_dir_all(ctx.project.parent().unwrap()).unwrap();
    }
}
