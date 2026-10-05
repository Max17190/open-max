mod app;
mod clipboard;
mod completion;
mod headless;
mod input;
mod stdio;
mod theme;
mod ui;

use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicU8, Ordering};

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::{execute, queue};
use open_max_core::state::{default_data_dir, Core};

const HELP: &str = "openmax: a barebones high-performance agent harness

usage: openmax [options] [prompt...]

options:
  -c, --continue         resume the latest session in this directory
  -m, --model <id>       use this model id for the run
      --provider <name>  use a named provider from ~/.openmax/providers.json
  -p, --print            headless: run one turn and exit (prompt required;
                         repeat -p for multi-turn on the same session)
      --json             with --print, emit AgentEvent envelopes as JSONL;
                         with --check, emit findings as one JSON array
      --stdio            bidirectional JSONL session: commands on stdin
                         ({\"cmd\":\"user\"|\"approve\"|\"approval_mode\"|
                         \"reload\"|\"cancel\"|\"quit\"}), AgentEvent envelopes
                         on stdout; the custom-frontend protocol
      --recall <query>   search this project's past sessions, archives,
                         compaction digests, and memories; prints ranked
                         excerpts, each cited as file:line so the full record
                         reads back exactly. Query syntax: plain terms plus
                         path:<substr>, session:<id-prefix>, k:<n>,
                         budget:<tokens>, excerpt:<chars>; --json for
                         structured output. Full contract:
                         --spec recall
      --ledger           print the capability-file history for this project
      --ledger-repair    quarantine an unverifiable ledger log (nothing is
                         deleted) and start a new chain; approvals in the
                         quarantined log must be granted again
      --approve <path>   approve the exact current content of a capability file
                         and of the project-local code it runs
      --forget <path>    stop expecting an approved HOOK file to exist (after
                         deliberately deleting one; a missing approved hook
                         fails closed). Tools never fail closed: a deleted
                         approved tool needs nothing forgotten
      --run-examples     with --check, execute each tool's [example] once, in
                         a trusted project, under the same permission rules
                         and pre_tool_use hooks as a turn. In auto, a valid
                         tool then runs on the host without content
                         approval. In ask or readonly, an approved tool runs
                         on the host under approval_mode, and an unapproved
                         one is probed in a sandbox (no network, writes
                         confined to a scratch dir); a passing probe
                         approves nothing
      --check            validate extension files (tools, skills, templates,
                         hooks, permissions, providers, memory) and exit;
                         nonzero if any is broken.
                         with --stdio, validate a JSONL protocol stream on
                         stdin against the openmax-stdio contract instead
      --spec <surface>   print the authoring contract for one surface and
                         exit (tools, skills, prompts, hooks, permissions,
                         providers, settings, memory, recall, stdio, usage)
      --trust-project    persist trust for this exact project root, then run
  -V, --version          print the version
  -h, --help             this help

point at any OpenAI-compatible endpoint via settings.json base_url, or register
named providers in ~/.openmax/providers.json and switch with --provider.
run inside a project directory; /help lists in-session commands.

examples:
  openmax
  openmax --provider ollama -m qwen2.5-coder:7b
  openmax -p \"summarize this repo\"
  openmax -p --json \"list top-level files\"
  openmax -p \"list crates\" -p \"summarize the first one\"";

struct CliArgs {
    continue_session: bool,
    model: Option<String>,
    provider: Option<String>,
    print: bool,
    json: bool,
    stdio: bool,
    check: bool,
    run_examples: bool,
    approve: Option<String>,
    forget: Option<String>,
    ledger: bool,
    /// Query string for `--recall`: ranked search over this project's own
    /// session history and memories.
    recall: Option<String>,
    /// Quarantine an unverifiable ledger log and start a new chain.
    ledger_repair: bool,
    /// Surface name whose authoring contract should be printed (`--spec`).
    spec: Option<String>,
    trust_project: bool,
    /// One prompt string per headless turn (tokens between repeated -p flags
    /// are joined with spaces into a single turn).
    prompts: Vec<String>,
}

fn parse_args() -> Result<CliArgs, lexopt::Error> {
    parse_args_from(std::env::args_os().skip(1))
}

fn parse_args_from<I, T>(args: I) -> Result<CliArgs, lexopt::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    use lexopt::prelude::*;
    let mut out = CliArgs {
        continue_session: false,
        model: None,
        provider: None,
        print: false,
        json: false,
        stdio: false,
        check: false,
        run_examples: false,
        approve: None,
        forget: None,
        ledger: false,
        recall: None,
        ledger_repair: false,
        spec: None,
        trust_project: false,
        prompts: Vec::new(),
    };
    // Tokens for the current -p group; flushed into prompts on the next -p or end.
    let mut current: Vec<String> = Vec::new();
    let mut parser = lexopt::Parser::from_args(args);
    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') | Long("continue") => out.continue_session = true,
            Short('m') | Long("model") => out.model = Some(parser.value()?.string()?),
            Long("provider") => out.provider = Some(parser.value()?.string()?),
            Short('p') | Long("print") => {
                if out.print {
                    // Subsequent -p closes the previous prompt; empty is an error.
                    flush_prompt_tokens(&mut out.prompts, &mut current)?;
                }
                out.print = true;
            }
            Long("json") => out.json = true,
            Long("stdio") => out.stdio = true,
            Long("check") => out.check = true,
            Long("run-examples") => out.run_examples = true,
            Long("approve") => set_once(&mut out.approve, "--approve", &mut parser)?,
            Long("forget") => set_once(&mut out.forget, "--forget", &mut parser)?,
            Long("ledger") => out.ledger = true,
            Long("recall") => set_once(&mut out.recall, "--recall", &mut parser)?,
            Long("ledger-repair") => out.ledger_repair = true,
            Long("spec") => set_once(&mut out.spec, "--spec", &mut parser)?,
            Long("trust-project") => out.trust_project = true,
            Short('V') | Long("version") => {
                println!("openmax {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            Short('h') | Long("help") => {
                println!("{HELP}");
                std::process::exit(0);
            }
            Value(v) => current.push(v.string()?),
            _ => return Err(arg.unexpected()),
        }
    }
    if out.print {
        flush_prompt_tokens(&mut out.prompts, &mut current)?;
    } else if !current.is_empty() {
        // Freeform args without --print are rejected in main; still clear cleanly.
        flush_prompt_tokens(&mut out.prompts, &mut current)?;
    }
    Ok(out)
}

/// Store the value of an operation that acts on one thing. A second
/// occurrence would replace the first, which was then dropped without a word
/// (`--approve a --approve b` approved only b and exited 0), so it is refused.
fn set_once(
    slot: &mut Option<String>,
    flag: &str,
    parser: &mut lexopt::Parser,
) -> Result<(), lexopt::Error> {
    use lexopt::ValueExt;
    if slot.is_some() {
        return Err(lexopt::Error::from(format!("{flag} given twice; run one at a time")));
    }
    *slot = Some(parser.value()?.string()?);
    Ok(())
}

fn flush_prompt_tokens(
    prompts: &mut Vec<String>,
    current: &mut Vec<String>,
) -> Result<(), lexopt::Error> {
    if current.is_empty() {
        return Err(lexopt::Error::from(
            "each --print requires a non-empty prompt".to_string(),
        ));
    }
    let joined = current.join(" ");
    current.clear();
    if joined.trim().is_empty() {
        return Err(lexopt::Error::from(
            "each --print requires a non-empty prompt".to_string(),
        ));
    }
    prompts.push(joined);
    Ok(())
}

/// Whether a human is standing behind an authority-GRANTING command
/// (`--approve`, `--trust-project`). Two walls, both required to pass:
///
/// 1. Not agent-spawned: the `OPENMAX_SESSION` marker every child carries.
///    That marker alone is one `env -u` away: an agent hitting the
///    in-session refusal reaches for `env -u OPENMAX_SESSION
///    openmax --approve` immediately, and the ledger records the
///    result as a human act.
/// 2. An interactive terminal on stdin - the same wall `--forget` and
///    `--ledger-repair` already stand behind. A `bash` tool call has piped
///    stdio and no tty; the marker can be unset, a terminal cannot be
///    conjured without allocating a pty (the documented ~20-line ceiling,
///    now the SAME ceiling for granting authority as for removing it).
///
/// Human automation (CI, test automation, `cargo test`) has no terminal and is
/// still a human act: `OPENMAX_HUMAN_ATTEST=1` states that explicitly. It
/// is honored only when the session marker is ABSENT, and it is a variable
/// a human sets on purpose - never one a session exports to its children -
/// so an agent must both unset the marker AND forge the attestation, which
/// is a deliberate act with a name, not a stumbled-into one-liner.
fn require_human(what: &str, repair: &str) {
    if std::env::var_os("OPENMAX_SESSION").is_some() {
        eprintln!(
            "openmax: {what} is a human action: this process was started from an agent session; ask the user to run `{repair}`"
        );
        std::process::exit(3);
    }
    let attested = std::env::var_os("OPENMAX_HUMAN_ATTEST").is_some_and(|v| v == "1");
    if !attested && !std::io::stdin().is_terminal() {
        eprintln!(
            "openmax: {what} is a human action and this process has no terminal; run `{repair}` from an interactive shell (automation a human runs may set OPENMAX_HUMAN_ATTEST=1)"
        );
        std::process::exit(3);
    }
}

/// Every printed `openmax --approve` quotes its path (core's shell_quote
/// contract): manifest paths are agent-chosen filenames, and a space or
/// metacharacter in one turned the pastable repair into a command that
/// failed on a path fragment (reproduced).
fn approve_command(path: &std::path::Path) -> String {
    format!("openmax --approve {}", open_max_core::doctor::shell_quote(path))
}

/// Why a command line is refused before anything runs, or None.
///
/// `main` performs one operation, and every operation but a session exits
/// when it is done, before the next one is considered. A second operation,
/// or an option only another operation reads, was dropped without a word and
/// the run exited as if that work had happened: `--check --ledger` printed
/// history and exited 0 having validated nothing. Every combination of
/// different flags is decided here, and each refusal names both sides of the
/// conflict (one operation given twice is refused while parsing, by
/// `set_once`). The destructure below is exhaustive, so a new `CliArgs` field
/// stops compiling here until it is placed among the operations or options.
fn refusal(cli: &CliArgs) -> Option<String> {
    let CliArgs {
        continue_session,
        model,
        provider,
        print,
        json,
        stdio,
        check,
        run_examples,
        approve,
        forget,
        ledger,
        recall,
        ledger_repair,
        spec,
        trust_project,
        prompts,
    } = cli;
    // Each operation, named by its flag. `--check --stdio` is one operation
    // of its own: validating a protocol stream on stdin.
    let operations: Vec<&str> = [
        (spec.is_some(), "--spec"),
        (recall.is_some(), "--recall"),
        (*ledger, "--ledger"),
        (*ledger_repair, "--ledger-repair"),
        (approve.is_some(), "--approve"),
        (forget.is_some(), "--forget"),
        (*check, if *stdio { "--check --stdio" } else { "--check" }),
        (*stdio && !*check, "--stdio"),
        (*print, "--print"),
    ]
    .into_iter()
    .filter_map(|(on, flag)| on.then_some(flag))
    .collect();
    if let [first, second, ..] = operations.as_slice() {
        return Some(format!("{first} and {second} are separate operations; run them one at a time"));
    }
    // None is the interactive session.
    let operation = operations.first().copied();
    // Each option and the operations that read it.
    const SESSIONS: &[Option<&str>] = &[None, Some("--stdio"), Some("--print")];
    let options: [(bool, &str, &[Option<&str>]); 6] = [
        (*trust_project, "--trust-project", SESSIONS),
        (*continue_session, "--continue", SESSIONS),
        (model.is_some(), "--model", SESSIONS),
        (provider.is_some(), "--provider", SESSIONS),
        (*json, "--json", &[Some("--print"), Some("--check"), Some("--recall")]),
        (*run_examples, "--run-examples", &[Some("--check")]),
    ];
    for (on, flag, readers) in options {
        if !on || readers.contains(&operation) {
            continue;
        }
        return Some(match operation {
            Some(operation) => format!("{operation} does not use {flag}; run them separately"),
            None => {
                let readers: Vec<&str> = readers.iter().flatten().copied().collect();
                let readers = match readers.split_last() {
                    Some((last, rest)) if !rest.is_empty() => format!("{}, or {last}", rest.join(", ")),
                    _ => readers.concat(),
                };
                format!("{flag} requires {readers}")
            }
        });
    }
    if !prompts.is_empty() && operation != Some("--print") {
        // With no operation named, a stray word is most likely a prompt
        // missing its -p, so that refusal points at --print; beside a named
        // operation it was most likely meant for that operation (a path
        // after --check), so the refusal names the word instead.
        let words = prompts.join(" ");
        return Some(match operation {
            None => "unexpected arguments (use --print for headless)".to_string(),
            Some(operation) => {
                let stdin = match operation {
                    "--stdio" => "; it reads commands on stdin",
                    "--check --stdio" => "; it reads the stream on stdin",
                    _ => "",
                };
                format!("{operation} takes no other arguments (got '{words}'){stdin}")
            }
        });
    }
    None
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let cli = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("openmax: {e}\n\n{HELP}");
            std::process::exit(2);
        }
    };

    if let Some(reason) = refusal(&cli) {
        eprintln!("openmax: {reason}\n\n{HELP}");
        std::process::exit(2);
    }

    if let Some(surface) = &cli.spec {
        // `usage` is the one dynamic surface: it joins each extension's
        // frozen-prompt cost (paid on every request) with its recorded use,
        // so the agent can see which of its own creations are pure tax and
        // prune them - the deletion roadmap applied to the agent's toolbox.
        if surface == "usage" {
            print_usage_economics();
            std::process::exit(0);
        }
        // Pure print: no session, no endpoint, no state dir, no trust. The
        // error path lists every valid surface so a wrong guess self-corrects.
        match open_max_core::spec::render(surface) {
            Some(text) => {
                println!("{text}");
                std::process::exit(0);
            }
            None => {
                eprintln!(
                    "openmax: unknown spec surface '{surface}'; available: {}",
                    open_max_core::spec::SURFACES.join(", ")
                );
                std::process::exit(2);
            }
        }
    }

    if let Some(query) = &cli.recall {
        // Read-only introspection, like --ledger: no session, no endpoint,
        // no trust gate. Recall only ever surfaces this project's own history
        // (the session index is keyed by project), and the project key is the
        // same raw current_dir form session creation stores.
        let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        // Settings are how a turn reaches a provider and what it may spend;
        // recall reads neither, so a settings file this process will never act
        // on must not be able to hide the project's own history. Reported on
        // stderr, so a `--json` consumer still gets clean output on stdout.
        let (core, _rx, unreadable_settings) =
            open_max_core::state::Core::read_only(default_data_dir());
        if let Some(reason) = unreadable_settings {
            eprintln!(
                "openmax: {reason}\n  searching history anyway: recall never reads settings, \
                 but every other command will refuse until this is fixed"
            );
        }
        match open_max_core::recall::recall(&core, &project, query) {
            Ok(report) => {
                if cli.json {
                    println!("{}", serde_json::to_string(&report).unwrap_or_else(|_| "{}".into()));
                } else {
                    print!("{}", open_max_core::recall::render(&report));
                }
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("openmax: {e}");
                std::process::exit(2);
            }
        }
    }

    if cli.ledger {
        // Read-only history, like --check: no session, no endpoint, no trust.
        let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let data_dir = default_data_dir();
        match open_max_core::ledger::read(&data_dir, &project) {
            Ok(history) if history.records.is_empty() => {
                println!("no capability-file history for this project yet");
                std::process::exit(0);
            }
            Ok(history) => {
                use open_max_core::ledger::{Kind, ObjectState};
                let objects = open_max_core::ledger::project_dir(&data_dir, &project).join("objects");
                // Approvals are recorded by content hash; a path is carried
                // when the caller knew one, and otherwise resolved from the
                // change record that observed the same bytes.
                let mut path_of: std::collections::HashMap<&str, &std::path::Path> =
                    std::collections::HashMap::new();
                for r in &history.records {
                    if let (Kind::Change, Some(sha)) = (r.kind, &r.sha256) {
                        path_of.insert(sha.as_str(), r.path.as_path());
                    }
                }
                let mut states: std::collections::HashMap<&str, ObjectState> =
                    std::collections::HashMap::new();
                let mut damaged = 0usize;
                let mut restorable = 0usize;
                // Each intact object with the project path it belongs at, so the
                // footer can print a `cp` command that actually runs. The
                // history lines above abbreviate the sha to 12 chars for
                // reading, but an object's filename is the full 64, so the bare
                // `cp <objects>/<sha> <path>` template was never executable
                // with what the history printed (Judge F).
                let mut restore: Vec<(String, std::path::PathBuf)> = Vec::new();
                for r in &history.records {
                    let short = r.sha256.as_deref().map(|s| &s[..12.min(s.len())]);
                    let where_ = match (r.path.as_os_str().is_empty(), r.sha256.as_deref()) {
                        (false, _) => r.path.display().to_string(),
                        (true, Some(sha)) => path_of
                            .get(sha)
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "(file not in this ledger)".to_string()),
                        (true, None) => String::new(),
                    };
                    // Objects are the bytes rollback copies, so a rewritten
                    // one is a backdoor with a documented delivery route:
                    // verify on read, not only on write.
                    let note = match (r.kind, r.sha256.as_deref()) {
                        (Kind::Change, Some(sha)) => {
                            restorable += 1;
                            let state = *states.entry(sha).or_insert_with(|| {
                                open_max_core::ledger::object_state(&data_dir, &project, sha)
                            });
                            match state {
                                ObjectState::Intact => "",
                                ObjectState::Missing => {
                                    damaged += 1;
                                    "  (object missing: cannot restore)"
                                }
                                ObjectState::Corrupt => {
                                    damaged += 1;
                                    "  (object CORRUPT: does not hash to its name - do not restore it)"
                                }
                            }
                        }
                        _ => "",
                    };
                    let session = r
                        .session_id
                        .as_deref()
                        .map(|s| format!("  session {s}"))
                        .unwrap_or_default();
                    // One approval act can bless a manifest and the code it
                    // runs; the audit has to show that it covered both - and
                    // whether their bytes are actually stored. A bound object
                    // that is missing or corrupt has nothing to restore, and
                    // the footer's recipe must not imply otherwise.
                    let bound = match r.also.len() {
                        0 => String::new(),
                        n => {
                            let stored = r
                                .also
                                .iter()
                                .filter(|sha| {
                                    matches!(
                                        *states.entry(sha.as_str()).or_insert_with(|| {
                                            open_max_core::ledger::object_state(&data_dir, &project, sha)
                                        }),
                                        ObjectState::Intact
                                    )
                                })
                                .count();
                            // Each non-intact bound object is damage too, or
                            // the footer prints an unqualified `restore with cp`
                            // recipe while this approval cannot be fully
                            // restored - same accounting the manifest path does.
                            damaged += n - stored;
                            let plural = if n == 1 { "" } else { "s" };
                            if stored == n {
                                format!("  (+{n} bound file{plural})")
                            } else {
                                format!(
                                    "  (+{n} bound file{plural}, {} with no intact object)",
                                    n - stored
                                )
                            }
                        }
                    };
                    // The manifest object matters as much as the bound ones:
                    // a damaged store can keep intact bound objects while the
                    // PRIMARY manifest object is missing or corrupt, so the
                    // row must not read as fully restorable.
                    // Checked from the full sha, not the short
                    // display form.
                    let manifest_note =
                        if open_max_core::ledger::approval_manifest_missing(&data_dir, &project, r) {
                            // Count it as damaged too, or the footer prints an
                            // unqualified `restore with cp` recipe while this
                            // very approval cannot be fully restored.
                            damaged += 1;
                            "  (manifest bytes not stored: cannot restore the manifest)"
                        } else {
                            ""
                        };
                    let what = match (r.kind, short) {
                        (Kind::Change, Some(sha)) => format!("change   {sha} {where_}"),
                        (Kind::Change, None) => format!("removed  {:12} {where_}", ""),
                        (Kind::Approval, Some(sha)) => {
                            format!("approved {sha} {where_}{bound}{manifest_note}")
                        }
                        (Kind::Approval, None) => {
                            format!("approved {:12} {where_} (path only)", "")
                        }
                        (Kind::PathRetired, _) => {
                            format!("retired  {:12} {where_} (path no longer expected)", "")
                        }
                    };
                    // The row carries the record's stored path, which is a
                    // capability file's own name. One record is one row, the
                    // same rule check_row applies to a finding.
                    println!(
                        "{}",
                        open_max_core::text::one_line(&format!(
                            "{} {:8} {what}{note}{session}",
                            open_max_core::ledger::format_ts(r.ts),
                            r.actor.as_str(),
                        ))
                    );
                    // Record the runnable restore targets for the footer. A
                    // change record names its own path and hash; an approval
                    // names its manifest plus, per vouched hash in `also`, the
                    // file the approved bytes bound it to - the second file
                    // the deleted-hook recovery needs and that the old footer
                    // never named. approval_restore_targets pairs them from
                    // the record's own path list (hooks) or the stored
                    // manifest object (tools record none), and refuses rather
                    // than guesses when a card-skipped file leaves the lists
                    // unequal. Only intact objects are offered; a missing or
                    // corrupt one is already counted as damage above.
                    match r.kind {
                        Kind::Change => {
                            if let Some(sha) = &r.sha256 {
                                if states.get(sha.as_str()).copied() == Some(ObjectState::Intact) {
                                    restore.push((sha.clone(), r.path.clone()));
                                }
                            }
                        }
                        Kind::Approval => {
                            if let Some(sha) = &r.sha256 {
                                if !r.path.as_os_str().is_empty()
                                    && !open_max_core::ledger::approval_manifest_missing(
                                        &data_dir, &project, r,
                                    )
                                {
                                    restore.push((sha.clone(), r.path.clone()));
                                }
                            }
                            restore.extend(open_max_core::ledger::approval_restore_targets(
                                &data_dir, &project, r,
                            ));
                        }
                        _ => {}
                    }
                }
                if history.interrupted_write {
                    let authority = history.records[history.pinned..]
                        .iter()
                        .filter(|r| r.kind != open_max_core::ledger::Kind::Change)
                        .count();
                    if authority > 0 {
                        println!(
                            "\nnote: {authority} record(s) past the chain-head pin grant or retire authority; nobody's pin vouches for them, so they are inert until `openmax --ledger-repair` sets them aside"
                        );
                    } else {
                        println!(
                            "\nnote: the last append landed but its chain-head pin did not (an interrupted write); nothing was removed, and the next capability change re-pins it"
                        );
                    }
                }
                if let Some(summary) =
                    ledger_objects_summary(objects.is_dir(), &restore, restorable, damaged, &objects)
                {
                    println!("{summary}");
                }
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("openmax: {e}");
                std::process::exit(1);
            }
        }
    }

    if cli.ledger_repair {
        use open_max_core::ledger::RepairPlan;
        // Repair rewrites the record of what happened, so it is a human
        // action for the same reason approval is - and guarded the same way
        // as --forget: the marker is one `unset` away from any shell the
        // agent already has, so a terminal and a typed word stand behind it.
        // The same honest ceiling applies; see the --forget comment.
        if std::env::var_os("OPENMAX_SESSION").is_some() {
            eprintln!(
                "openmax: ledger repair is a human action: this process was started from an agent session; ask the user to run `openmax --ledger-repair`"
            );
            std::process::exit(3);
        }
        let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let plan = open_max_core::ledger::repair_plan(&default_data_dir(), &project);
        // The word the human must type is the operation they just read, so a
        // confirmation proves they saw which repair this is.
        let word = match &plan {
            RepairPlan::Nothing => None,
            RepairPlan::Repin => {
                println!(
                    "the last append landed but its chain-head pin did not (an interrupted write)."
                );
                println!("re-pinning keeps every record; nothing gains authority.");
                Some("repin")
            }
            RepairPlan::QuarantineTail { tail } => {
                println!(
                    "{} record(s) sit past the pinned chain head, and some grant or retire authority:",
                    tail.len()
                );
                for r in tail {
                    let sha = r.sha256.as_deref().unwrap_or("-");
                    println!(
                        "  {}  {:<16} {}  {}",
                        open_max_core::ledger::format_ts(r.ts),
                        r.kind.as_str(),
                        &sha[..12.min(sha.len())],
                        r.path.display(),
                    );
                }
                println!(
                    "nobody's pin vouches for these lines, so repair sets them aside (nothing is deleted)."
                );
                println!(
                    "if an approval above is one you performed, re-run `openmax --approve <path>` afterwards."
                );
                Some("quarantine")
            }
            RepairPlan::Quarantine { records, approvals } => {
                println!(
                    "the ledger does not verify: {records} record(s), {approvals} of them approval-grade, will be set aside (nothing is deleted)."
                );
                println!("a new chain starts at the next capability change; every approval must be granted again.");
                Some("quarantine")
            }
        };
        if let Some(word) = word {
            if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
                eprintln!(
                    "openmax: --ledger-repair changes what this project's history vouches for, so it only runs at an interactive terminal"
                );
                std::process::exit(3);
            }
            print!("type `{word}` to confirm: ");
            let _ = std::io::stdout().flush();
            let mut answer = String::new();
            if std::io::stdin().read_line(&mut answer).is_err() || answer.trim() != word {
                eprintln!("openmax: confirmation did not match; nothing was changed");
                std::process::exit(1);
            }
        }
        match open_max_core::ledger::repair(&default_data_dir(), &project) {
            Ok(outcome) => {
                match outcome.quarantined {
                    Some(path) => {
                        println!(
                            "quarantined {} unverifiable record(s) to {}",
                            outcome.records,
                            path.display()
                        );
                        println!("a new chain starts at the next capability change; nothing was deleted, and the objects for rollback are untouched");
                        if outcome.approvals > 0 {
                            println!(
                                "{} approval(s) went with it: re-approve each file you still trust with `openmax --approve <path>`",
                                outcome.approvals
                            );
                        }
                    }
                    None if outcome.repinned => {
                        println!("re-pinned the chain head after an interrupted write; no records were lost")
                    }
                    None => println!("ledger verifies; nothing to repair"),
                }
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("openmax: {e}");
                std::process::exit(1);
            }
        }
    }

    if let Some(path) = &cli.approve {
        // Approval is a human action, exactly like trust; see require_human
        // for the two walls and why the marker alone was not one.
        require_human("approval", &format!("openmax --approve {path}"));
        let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let file = std::path::Path::new(path);
        let bytes = match std::fs::read(file) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("openmax: cannot read {path}: {e}");
                std::process::exit(1);
            }
        };
        let sha = open_max_core::ledger::sha256_hex(&bytes);
        // A manifest is a pointer: the file it names is the code that runs,
        // and it sits at an ordinary project path the agent writes freely. So
        // approving the manifest approves that code in the same act - and
        // prints it, because a human cannot bless bytes they were not shown.
        // The code list comes from the bytes just hashed, not a second read:
        // two reads are an interval a concurrent write can split, putting one
        // file's hash on record next to another file's code.
        let code = std::str::from_utf8(&bytes)
            .map(|text| open_max_core::ledger::manifest_code_source(file, text, &project))
            .unwrap_or_default();
        let mut shas = vec![sha.clone()];
        for entry in &code {
            let Some(code_sha) = &entry.sha256 else {
                eprintln!(
                    "openmax: {path} runs {}, which cannot be read; create it first, then approve them together",
                    entry.path.display()
                );
                std::process::exit(1);
            };
            shas.push(code_sha.clone());
        }
        match open_max_core::ledger::approve_capability(&default_data_dir(), &project, file, &shas) {
            Ok(()) => {
                // The one screen a human reads before vouching for bytes, so
                // no path it lists may forge a line and claim a file the
                // approval does not cover.
                println!(
                    "{}",
                    open_max_core::text::one_line(&format!("approved {path} ({})", &sha[..12]))
                );
                for entry in &code {
                    let code_sha = entry.sha256.clone().unwrap_or_default();
                    println!(
                        "{}",
                        open_max_core::text::one_line(&format!(
                            "  and the code it runs: {} ({})",
                            entry.path.display(),
                            &code_sha[..12.min(code_sha.len())]
                        ))
                    );
                }
                if !code.is_empty() {
                    println!("editing either revokes this approval; re-run --approve after any change");
                }
                // Absolute for the dir comparison: the hook-dir check
                // matches parents against absolute discovery dirs, and the
                // CLI accepts a project-relative path.
                let shape_path =
                    if file.is_absolute() { file.to_path_buf() } else { project.join(file) };
                if let Some(line) =
                    open_max_core::hooks::approved_shape_line(&shape_path, &project, &default_data_dir(), &bytes)
                {
                    println!("{line}");
                }
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("openmax: {e}");
                std::process::exit(1);
            }
        }
    }

    if let Some(path) = &cli.forget {
        // Retiring an approval REMOVES a gate, which is why it is guarded
        // harder than granting one. `--approve` and `--trust-project` add
        // authority and leave an auditable artifact behind: a content hash
        // pinned to bytes a human can read afterwards. This leaves nothing,
        // so the session marker alone is too thin a wall - an agent holding
        // `bash` clears an environment variable with `unset`.
        //
        // What the terminal check buys is cost, not impossibility, and the
        // measured numbers are worth stating rather than implying. It removes
        // the one-liner: `unset OPENMAX_SESSION; openmax --forget <path>` now
        // exits 3. It does not stop an attacker who allocates a pty and
        // answers this prompt - roughly twenty lines of Python, verified to
        // still retire the approval. What is left for `bash` to do outright
        // is destroy ledger state, and destruction now fails closed: a
        // removed or rewritten chain reads as tamper, surrenders every
        // approval, and waits for `--ledger-repair`. The one memory an
        // attacker can still erase wholesale is the ledger directory itself,
        // which forgets that a gate ever existed - documented, detected by
        // nothing short of an OS sandbox.
        //
        // So this is a speed bump on the supported route, not a boundary.
        // Without an OS sandbox that is the honest ceiling, the same one
        // trust lives at; the marker check stays as the cheap first line.
        if std::env::var_os("OPENMAX_SESSION").is_some() {
            eprintln!(
                "openmax: approvals are human actions: this process was started from an agent session; ask the user to run `openmax --forget {path}`"
            );
            std::process::exit(3);
        }
        let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let file = std::path::Path::new(path);
        let target = if file.is_absolute() { file.to_path_buf() } else { project.join(file) };
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            eprintln!(
                "openmax: --forget retires a human-installed policy, so it only runs at an interactive terminal.\n\
                 without one, restore the file instead: `openmax --ledger` names the object holding its approved bytes"
            );
            std::process::exit(3);
        }
        println!("about to retire the approval recorded at {}", target.display());
        println!(
            "if that file was a gate, the policy it enforced stops being expected and stops being enforced."
        );
        print!("type the path exactly as given to confirm: ");
        let _ = std::io::stdout().flush();
        let mut answer = String::new();
        if std::io::stdin().read_line(&mut answer).is_err() {
            eprintln!("openmax: could not read a confirmation; nothing was changed");
            std::process::exit(1);
        }
        if !forget_confirmed(&answer, path, &target) {
            eprintln!("openmax: confirmation did not match; nothing was changed");
            std::process::exit(1);
        }
        match open_max_core::ledger::forget_capability(&default_data_dir(), &project, &target) {
            Ok(true) => {
                println!("forgot {path}; the harness no longer expects a capability file there");
                if target.exists() {
                    println!(
                        "note: the file is still on disk and its content approval still stands"
                    );
                }
                std::process::exit(0);
            }
            Ok(false) => {
                eprintln!("openmax: no approved capability is recorded at {path}");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("openmax: {e}");
                std::process::exit(1);
            }
        }
    }

    if cli.check {
        // With --stdio, validate a JSONL protocol stream on stdin instead of
        // the filesystem: no session, no endpoint, no state dir either.
        if cli.stdio {
            std::process::exit(stdio::run_conformance());
        }
        // Pure filesystem validation: no session, no endpoint, no state dir.
        let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let findings = open_max_core::doctor::check(&project);
        if cli.json {
            // Machine face of the same report: the agent parses this in-turn.
            let mut array: Vec<serde_json::Value> = findings
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "surface": f.kind,
                        "path": f.path.display().to_string(),
                        "status": f.status.as_str(),
                        "message": f.status.summary(),
                    })
                })
                .collect();
            let mut failed = open_max_core::doctor::has_errors(&findings);
            if cli.run_examples {
                // Examples belong in the machine report too: the consumer most
                // likely to gate on it is an agent verifying a tool it just
                // wrote, and all-green for a proof that never ran is a lie.
                let (rows, failures) = tool_example_rows(&project).await;
                array.extend(rows);
                failed = failed || failures > 0;
            }
            println!("{}", serde_json::Value::Array(array));
            std::process::exit(if failed { 1 } else { 0 });
        }
        // With --run-examples there is still a verdict to report (or a refusal
        // to explain), and both faces have to agree on the exit code.
        if findings.is_empty() && !cli.run_examples {
            println!(
                "no extension files found (tools, skills, templates, hooks, permissions, providers, settings)"
            );
            std::process::exit(0);
        }
        for f in &findings {
            println!("{}", check_row(f));
        }
        let mut example_failures = 0usize;
        if cli.run_examples {
            example_failures = run_tool_examples(&project, &findings).await;
        }
        // Warnings do not fail the run: a shadowed global default and a rule
        // written before its tool are both normal.
        if open_max_core::doctor::has_warnings(&findings) {
            println!(
                "\nwarn lines are files the agent loop never reads, or reads but cannot act on \
                 as written. They do not fail this check."
            );
        }
        let failed = open_max_core::doctor::has_errors(&findings) || example_failures > 0;
        std::process::exit(if failed { 1 } else { 0 });
    }

    let data_dir = default_data_dir();
    let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    if let Err(e) = ensure_project_trust(&cli, &data_dir, &project) {
        eprintln!("openmax: {e}");
        std::process::exit(3);
    }

    let (core, core_rx) = match Core::new(data_dir) {
        Ok(pair) => pair,
        // Fail closed: a malformed settings file is a configuration error,
        // not a silent reset to defaults.
        Err(e) => {
            eprintln!("openmax: {e}");
            std::process::exit(2);
        }
    };
    {
        let mut s = core.settings.lock().unwrap();
        if let Some(provider) = &cli.provider {
            s.provider = Some(provider.clone());
        }
        if let Some(model) = &cli.model {
            s.model = model.clone();
        }
        // Headless and stdio runs fail fast on an unresolvable endpoint: there
        // is no interface behind them to fix it in. The interactive TUI starts
        // anyway - /model and /provider are exactly how a first run gets
        // configured, and every turn surfaces the same actionable error.
        if let Err(e) = open_max_core::providers::resolve(&s, &core.data_dir) {
            if cli.stdio || cli.print {
                eprintln!("openmax: {e}");
                std::process::exit(2);
            }
        }
    }

    if cli.stdio {
        let code = stdio::run(
            core,
            core_rx,
            stdio::StdioArgs { continue_session: cli.continue_session },
        )
        .await;
        std::process::exit(code);
    }

    if cli.print {
        if cli.prompts.is_empty() || cli.prompts.iter().all(|p| p.trim().is_empty()) {
            eprintln!("openmax: --print requires a prompt\n\n{HELP}");
            std::process::exit(2);
        }
        let code = headless::run(
            core,
            core_rx,
            headless::HeadlessArgs {
                prompts: cli.prompts,
                continue_session: cli.continue_session,
                json: cli.json,
            },
        )
        .await;
        std::process::exit(code);
    }

    // Signals are routed before the terminal changes, so none of them can
    // fall back to its default action and exit with every mode left on.
    let quit = watch_quit_signals();

    // Fullscreen session on the alternate screen: openmax owns the whole
    // terminal while it runs, and your shell (prompt, history, scrollback)
    // reappears untouched on exit.
    let terminal = match init_terminal() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("openmax: failed to initialize terminal: {e}");
            std::process::exit(1);
        }
    };

    // Kitty keyboard protocol makes Shift+Enter distinct; Alt+Enter stays as
    // the fallback everywhere else. Bracketed paste for sane multiline paste.
    if crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false) {
        enable_mode(
            MODE_KEYBOARD,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES),
        );
    }
    enable_mode(MODE_PASTE, crossterm::event::EnableBracketedPaste);
    // Mouse capture for wheel scrolling of the transcript. Terminals still
    // allow text selection with the usual modifier (Option on macOS).
    enable_mode(MODE_MOUSE, EnableMouseCapture);
    // Focus reports gate the turn-done ring on the user being away.
    enable_mode(MODE_FOCUS, crossterm::event::EnableFocusChange);

    let result = app::run(
        terminal,
        core,
        core_rx,
        app::Args { continue_session: cli.continue_session, quit },
    )
    .await;

    restore_terminal();
    result
}

/// Terminal modes the fullscreen session has switched on, one bit each.
/// Every exit path (quit, a signal, a failed init, a UI panic) ends in
/// `restore_terminal`, which undoes exactly these. A mode left on outlives
/// the process: mouse reporting alone turns every later mouse move into
/// escape bytes typed at the shell prompt.
static TERM_MODES: AtomicU8 = AtomicU8::new(0);
const MODE_RAW: u8 = 1;
const MODE_TITLE: u8 = 1 << 1;
const MODE_ALT_SCREEN: u8 = 1 << 2;
const MODE_KEYBOARD: u8 = 1 << 3;
const MODE_PASTE: u8 = 1 << 4;
const MODE_MOUSE: u8 = 1 << 5;
const MODE_FOCUS: u8 = 1 << 6;

/// XTWINOPS title stack pop; see `push_title`.
const POP_TITLE: &[u8] = b"\x1b[23;0t";

/// Switch one mode on and record it for `restore_terminal`. The bit is set
/// first: a write that fails partway may still have reached the terminal,
/// and undoing a mode that never took is harmless.
fn enable_mode(mode: u8, command: impl crossterm::Command) {
    TERM_MODES.fetch_or(mode, Ordering::SeqCst);
    let _ = execute!(std::io::stdout(), command);
}

/// Hand the terminal back to the shell. Every exit path calls it, and only
/// the first call after a mode was enabled does anything.
fn restore_terminal() {
    // Held until raw mode is off too: a concurrent caller (the forced exit
    // on a second signal) blocks here instead of finding the modes claimed
    // but not yet undone, exiting, and leaving the shell raw.
    let mut out = std::io::stdout().lock();
    let restored = write_restore(&mut out, &TERM_MODES);
    // Last, so whatever prints next (an error, a panic report) lands on a
    // cooked terminal.
    if restored & MODE_RAW != 0 {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// Write the sequences that undo each mode recorded in `modes`, and return
/// the modes it claimed. The swap claims them all at once, so a second call
/// writes nothing and two exit paths can never both restore.
fn write_restore<W: Write>(out: &mut W, modes: &AtomicU8) -> u8 {
    use crossterm::cursor::Show;
    use crossterm::event::{DisableBracketedPaste, DisableFocusChange};
    use crossterm::terminal::{EndSynchronizedUpdate, LeaveAlternateScreen};
    let on = modes.swap(0, Ordering::SeqCst);
    // One write per mode: a failed write must not keep the next mode on.
    if on & MODE_ALT_SCREEN != 0 {
        // A frame cut short by a panic can leave a synchronized update open
        // (the terminal then holds every later byte) and the cursor hidden.
        let _ = queue!(out, EndSynchronizedUpdate);
        let _ = queue!(out, Show);
    }
    if on & MODE_FOCUS != 0 {
        let _ = queue!(out, DisableFocusChange);
    }
    if on & MODE_MOUSE != 0 {
        let _ = queue!(out, DisableMouseCapture);
    }
    if on & MODE_PASTE != 0 {
        let _ = queue!(out, DisableBracketedPaste);
    }
    // Terminals keep one keyboard flag stack per screen, so the pop has to
    // land while the alternate screen is still current.
    if on & MODE_KEYBOARD != 0 {
        let _ = queue!(out, PopKeyboardEnhancementFlags);
    }
    if on & MODE_ALT_SCREEN != 0 {
        let _ = queue!(out, LeaveAlternateScreen);
    }
    if on & MODE_TITLE != 0 {
        let _ = out.write_all(POP_TITLE);
    }
    let _ = out.flush();
    on
}

/// Restore the terminal ahead of the panic report, but only for a panic on
/// `ui`, the thread that draws. The hook is process-global, and the core
/// survives panics on its own worker threads (a turn task, a tool) by
/// reporting them as errors while the TUI keeps drawing; restoring there
/// would drop the live session out of the alternate screen and leave it
/// painting over the shell.
fn install_panic_restore(ui: std::thread::ThreadId, restore: impl Fn() + Send + Sync + 'static) {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if std::thread::current().id() == ui {
            restore();
        }
        hook(info);
    }));
}

/// SIGTERM, SIGHUP (the terminal went away), and SIGINT sent from outside
/// (raw mode makes Ctrl+C a key) each end the process on the spot by
/// default, with every terminal mode still on. While the event loop runs,
/// the first one reaches it as a quit, so the session ends through the
/// normal exit path; a second means that path is stuck, so the terminal is
/// restored here and the process exits. Once the loop has ended, nothing
/// reads a quit again, so the first signal takes that exit itself.
///
/// The watcher runs on its own thread and runtime. When main returns, the
/// main runtime drops its tasks and then waits on blocking work still
/// running (a grep, a scan), while the handlers stay installed; a watcher
/// on that runtime would leave every signal through that wait swallowed.
#[cfg(unix)]
fn watch_quit_signals() -> std::sync::Arc<tokio::sync::Notify> {
    use futures_util::future::select_all;
    use tokio::signal::unix::{signal, SignalKind};
    let quit = std::sync::Arc::new(tokio::sync::Notify::new());
    // Weak: main and then the event loop own the quit, so it is gone once
    // the loop has ended, and a signal from then on exits at once instead
    // of waiting on a loop that will never read it.
    let event_loop = std::sync::Arc::downgrade(&quit);
    let (registered, ready) = std::sync::mpsc::channel();
    let _ = std::thread::Builder::new().name("signals".into()).spawn(move || {
        let Ok(runtime) = tokio::runtime::Builder::new_current_thread().enable_io().build() else {
            return;
        };
        runtime.block_on(async move {
            let mut signals: Vec<_> =
                [SignalKind::terminate(), SignalKind::hangup(), SignalKind::interrupt()]
                    .into_iter()
                    .filter_map(|kind| Some((kind.as_raw_value(), signal(kind).ok()?)))
                    .collect();
            let _ = registered.send(());
            if signals.is_empty() {
                return;
            }
            let (_, first, _) =
                select_all(signals.iter_mut().map(|(_, s)| Box::pin(s.recv()))).await;
            let last = match event_loop.upgrade() {
                Some(quit) => {
                    quit.notify_one();
                    drop(quit);
                    select_all(signals.iter_mut().map(|(_, s)| Box::pin(s.recv()))).await.1
                }
                None => first,
            };
            // Held until exit (the lock is reentrant, so the restore still
            // writes): the event loop must not paint over the restored shell.
            let _stdout = std::io::stdout().lock();
            restore_terminal();
            std::process::exit(128 + signals[last].0);
        });
    });
    // Return only once the handlers are registered, so a signal that lands
    // during startup still ends the session cleanly. A watcher that never
    // started drops its sender, which ends this wait too.
    let _ = ready.recv();
    quit
}

#[cfg(not(unix))]
fn watch_quit_signals() -> std::sync::Arc<tokio::sync::Notify> {
    std::sync::Arc::new(tokio::sync::Notify::new())
}

/// XTWINOPS title stack: save the shell's tab title on entry, restore it on
/// every exit path. Terminals without the stack ignore both writes and are
/// left with the last presence title, which at exit reads "project · openmax".
fn push_title() {
    TERM_MODES.fetch_or(MODE_TITLE, Ordering::SeqCst);
    let mut out = std::io::stdout();
    let _ = out.write_all(b"\x1b[22;0t").and_then(|_| out.flush());
}

/// `ratatui::init` with one change: frame output goes through a 256 KiB
/// buffer so each flush is one write(2) instead of the dozens that `Stdout`'s
/// built-in 1 KiB line buffer produces on token-streaming frames.
/// `restore_terminal` is the counterpart on every exit path; it writes to the
/// shared stdout fd, and every completed frame ends fully flushed.
fn init_terminal() -> std::io::Result<ui::transcript::Term> {
    use crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};
    // Hook first: any panic past raw mode must restore the shell. This thread
    // runs the event loop and draws every frame.
    install_panic_restore(std::thread::current().id(), restore_terminal);
    enter_raw_mode(&TERM_MODES, enable_raw_mode)?;
    // The session states its presence in the tab title while it runs (see
    // app::Presence); save the shell's title and hand it back on every exit
    // path. Pushed only after raw mode succeeds, so an early error cannot
    // leave an orphaned entry on the terminal's title stack.
    push_title();
    let init = || -> std::io::Result<ui::transcript::Term> {
        let mut out = FrameWriter::new(std::io::stdout(), 256 * 1024);
        TERM_MODES.fetch_or(MODE_ALT_SCREEN, Ordering::SeqCst);
        execute!(out, EnterAlternateScreen)?;
        out.flush()?;
        ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(out))
    };
    init().inspect_err(|_| restore_terminal())
}

/// Switch to raw mode and record it as one step under the stdout lock, which
/// `restore_terminal` holds from claiming the modes until raw mode is off.
/// Raw mode is termios state, not bytes on stdout, so nothing else orders it
/// against a concurrent restore: switched outside the lock, a forced exit
/// landing between the switch and the record would find nothing to undo and
/// exit with the shell raw. Under the lock, that exit either runs first and
/// keeps stdout until the process ends, so raw mode is never entered, or
/// finds raw mode recorded and leaves it.
fn enter_raw_mode(
    modes: &AtomicU8,
    enable: impl FnOnce() -> std::io::Result<()>,
) -> std::io::Result<()> {
    let _out = std::io::stdout().lock();
    enable()?;
    modes.fetch_or(MODE_RAW, Ordering::SeqCst);
    Ok(())
}

/// A frame-sized `BufWriter` that discards, rather than writes, its bytes
/// while the thread is panicking. The panic hook has already restored the
/// normal screen by the time unwinding drops the terminal, and that drop
/// still flushes (ratatui shows the cursor on the way out), so writing then
/// would land a partial frame, with its synchronized update still open, on
/// the user's shell.
pub struct FrameWriter<W: Write>(Option<std::io::BufWriter<W>>);

impl<W: Write> FrameWriter<W> {
    fn new(inner: W, capacity: usize) -> Self {
        Self(Some(std::io::BufWriter::with_capacity(capacity, inner)))
    }

    fn buf(&mut self) -> &mut std::io::BufWriter<W> {
        self.0.as_mut().expect("writer present until drop")
    }
}

impl<W: Write> Write for FrameWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if std::thread::panicking() {
            return Ok(buf.len());
        }
        self.buf().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if std::thread::panicking() {
            if let Some(w) = self.0.take() {
                // into_parts hands the buffer back without writing it.
                let capacity = w.capacity();
                let (inner, _) = w.into_parts();
                self.0 = Some(std::io::BufWriter::with_capacity(capacity, inner));
            }
            return Ok(());
        }
        self.buf().flush()
    }
}

impl<W: Write> Drop for FrameWriter<W> {
    fn drop(&mut self) {
        if let Some(w) = self.0.take() {
            if std::thread::panicking() {
                // into_parts hands the buffer back without writing it.
                drop(w.into_parts());
            } else {
                drop(w);
            }
        }
    }
}

/// Print `openmax --check --run-examples` results; returns the failure count.
async fn run_tool_examples(
    project: &std::path::Path,
    findings: &[open_max_core::doctor::Finding],
) -> usize {
    use open_max_core::doctor::Status;
    let mut failures = 0usize;
    let mut ran = 0usize;
    // Each verdict prints as it lands: examples run serially with per-tool
    // timeouts, so batching the report would look like a hang.
    let result = open_max_core::doctor::run_examples(project, |verdict| {
        ran += 1;
        if !verdict.sandboxed && verdict.result.is_err() {
            failures += 1;
        }
        println!("{}", example_row(verdict));
        let _ = std::io::stdout().flush();
    })
    .await;
    match result {
        Err(reason) => {
            println!(
                "{}",
                open_max_core::text::one_line(&format!(
                    "err  example     {}  {reason}",
                    project.display()
                ))
            );
            failures += 1;
        }
        Ok(_) if ran == 0 => {
            let unloadable = findings
                .iter()
                .filter(|f| f.kind == "tool" && matches!(f.status, Status::Err(_)))
                .count();
            match unloadable {
                0 => println!("no tool declares an [example]"),
                n => println!(
                    "no tool declares a runnable [example] ({n} tool file(s) failed to load; see the err lines above)"
                ),
            }
        }
        Ok(_) => {}
    }
    failures
}

/// The same example verdicts as JSON rows, for `--check --json`. Returns the
/// rows and the failure count.
async fn tool_example_rows(project: &std::path::Path) -> (Vec<serde_json::Value>, usize) {
    let mut rows = Vec::new();
    let mut failures = 0usize;
    match open_max_core::doctor::run_examples(project, |_| {}).await {
        Ok(verdicts) => {
            for verdict in verdicts {
                let (status, message) = match &verdict.result {
                    Ok(()) if verdict.sandboxed => (
                        "ok",
                        format!(
                            "example for '{}' ran in a sandbox (unapproved content: no network, writes confined; approve with {})",
                            verdict.tool,
                            approve_command(&verdict.path)
                        ),
                    ),
                    Ok(()) => ("ok", format!("example for '{}' ran", verdict.tool)),
                    // Inconclusive, not a failure: a sandboxed probe cannot
                    // prove a tool that needs the network or a non-scratch
                    // write, so it does not fail the check (see
                    // run_tool_examples). The real run after approval is the
                    // honest signal.
                    Err(reason) if verdict.sandboxed => (
                        "warn",
                        format!(
                            "could not prove '{}' in the sandbox (network and non-scratch writes are denied): {reason}",
                            verdict.tool
                        ),
                    ),
                    Err(reason) => {
                        failures += 1;
                        ("err", reason.clone())
                    }
                };
                rows.push(serde_json::json!({
                    "surface": "example",
                    "path": verdict.path.display().to_string(),
                    "status": status,
                    "message": message,
                    "sandboxed": verdict.sandboxed,
                }));
            }
        }
        Err(reason) => {
            failures += 1;
            rows.push(serde_json::json!({
                "surface": "example",
                "path": project.display().to_string(),
                "status": "err",
                "message": reason,
            }));
        }
    }
    (rows, failures)
}

/// Whether the typed answer retires exactly the capability that was named.
/// Either spelling the human has in front of them counts - the argument they
/// passed, or the resolved path the prompt printed - and nothing else does:
/// the point is that a person read which policy is going away, so "y" is not
/// enough.
/// The trailing objects-store summary for `--ledger`. `restore` is every
/// intact object paired with the path it belongs at, rendered as a runnable,
/// shell-quoted `cp` line so the documented recovery ("an ordinary cp from the
/// objects directory") can be copied and run - the old bare template named a
/// 12-char sha prefix the history printed, but object filenames are the full
/// 64, so it never worked (Judge F). `restorable` still counts the change
/// records that named stored bytes: with none, a missing store is the normal
/// state of an approvals-only ledger, not damage, so warning that "no version
/// can be restored" would cry wolf over a ledger holding nothing restorable.
fn ledger_objects_summary(
    store_exists: bool,
    restore: &[(String, std::path::PathBuf)],
    restorable: usize,
    damaged: usize,
    objects: &std::path::Path,
) -> Option<String> {
    if store_exists {
        let mut out = format!("\nobjects: {}", objects.display());
        if !restore.is_empty() {
            out.push_str(
                "\nrestore a file to a stored version by copying its object back into place:",
            );
            let mut seen = std::collections::HashSet::new();
            for (sha, path) in restore {
                if seen.insert((sha.as_str(), path.as_path())) {
                    out.push_str(&format!(
                        "\n  cp {} {}",
                        open_max_core::doctor::shell_quote(&objects.join(sha)),
                        open_max_core::doctor::shell_quote(path)
                    ));
                }
            }
        }
        if damaged > 0 {
            out.push_str(&format!(
                "\nwarning: {damaged} record(s) have no trustworthy object; those bytes cannot be restored from this ledger"
            ));
        }
        Some(out)
    } else if restorable > 0 {
        Some(format!(
            "\nobjects: {} is gone, so no version above can be restored from this ledger",
            objects.display()
        ))
    } else {
        None
    }
}

fn forget_confirmed(answer: &str, given: &str, resolved: &std::path::Path) -> bool {
    let answer = answer.trim();
    !answer.is_empty() && (answer == given.trim() || answer == resolved.display().to_string())
}

/// `openmax --spec usage`: per-extension prompt cost joined with lifetime
/// usage and approval state. Read-only; requires no trust or endpoint.
fn print_usage_economics() {
    use open_max_core::registry::ToolKind;
    let project = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let data_dir = default_data_dir();
    let registry = open_max_core::registry::Registry::build(&data_dir, &project);
    let usage = open_max_core::ledger::load_usage(&data_dir, &project).unwrap_or_default();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ago = |ts: u64| -> String {
        if ts == 0 {
            return "never".into();
        }
        match (now.saturating_sub(ts)) / 86_400 {
            0 => "today".into(),
            d => format!("{d}d ago"),
        }
    };

    let externals: Vec<_> = registry
        .tools
        .iter()
        .filter_map(|spec| match &spec.kind {
            ToolKind::External(ext) => Some((spec, ext)),
            ToolKind::Builtin => None,
        })
        .collect();
    // The frozen prompt prefix is paid on every request whether or not any
    // tool or skill is installed: the base rules, the self-extension guide,
    // AGENTS.md, the memory index, and the layout map all ride it. Report the
    // real breakdown here so "what does my prompt cost" is answered truthfully
    // and never as a false zero - a project with only an AGENTS.md or a memory
    // note pays real bytes the old "zero extension cost" line hid.
    let (_, breakdown) = open_max_core::prompt::system_prompt_with_breakdown(&project, &registry);
    // The whole serialized array is what rides every request, including the
    // brackets and commas between schemas, so measure the wire bytes rather
    // than summing each schema alone (that undercounts the array overhead).
    let schema_chars = registry.tool_schemas_wire().len();
    let component_chars: usize = breakdown.components.iter().map(|(_, c)| *c).sum();
    let frozen_chars = component_chars + schema_chars;
    println!("frozen prompt prefix (paid on every request):");
    for (name, chars) in &breakdown.components {
        println!("  {name:<26} {chars:>7} chars  ~{} tok", chars / 4);
    }
    println!("  {:<26} {schema_chars:>7} chars  ~{} tok", "tool schemas", schema_chars / 4);
    println!("  {:<26} {frozen_chars:>7} chars  ~{} tok  total", "", frozen_chars / 4);
    println!();

    if externals.is_empty() && registry.skills.is_empty() && breakdown.memory.is_empty() {
        // `breakdown.memory` is what the frozen index currently carries, not
        // what exists on disk: a note that faded past the 21-day index floor,
        // or one dropped by the byte cap, is still a file. So report the index
        // state, not an absolute "nothing installed".
        println!(
            "no tool or skill extensions installed, and no memory note is in the frozen index; the frozen prefix above is the harness base plus any AGENTS.md and layout map (faded or byte-capped memory notes may still be on disk: openmax --recall searches them)."
        );
        return;
    }
    println!(
        "{:<24} {:<6} {:>12} {:>7} {:>5} {:>5} {:>10}  approved",
        "extension", "kind", "prompt_chars", "calls", "ok", "err", "last_used"
    );
    for (spec, ext) in &externals {
        let chars = serde_json::json!({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
        })
        .to_string()
        .len();
        let entry = usage.tools.get(&spec.name).cloned().unwrap_or_default();
        // Approved means the whole definition: the manifest and the code it
        // runs. A tool whose script was rewritten after approval is not
        // approved, and must not read as if it were.
        let approvals =
            open_max_core::ledger::approvals(&data_dir, &project).unwrap_or_default();
        let approved = if approvals.contains(&ext.source_sha256)
            && approvals.covers_code(&open_max_core::ledger::bound_code(
                &ext.command,
                &ext.args,
                &project,
            )) {
            "yes"
        } else {
            "no"
        };
        println!(
            "{:<24} {:<6} {:>12} {:>7} {:>5} {:>5} {:>10}  {approved}",
            spec.name, "tool", chars, entry.calls, entry.ok, entry.err, ago(entry.last_used)
        );
    }
    // The same accounting the frozen prompt uses: a skill the index byte cap
    // dropped costs zero, and pricing it as carried would aim pruning at
    // tokens nobody is paying.
    let mut capped_out = 0usize;
    for (skill, (_, chars)) in
        registry.skills.iter().zip(open_max_core::prompt::skill_index_costs(&project, &registry.skills))
    {
        if chars == 0 {
            capped_out += 1;
        }
        let entry = usage.skills.get(&skill.name).cloned().unwrap_or_default();
        println!(
            "{:<24} {:<6} {:>12} {:>7} {:>5} {:>5} {:>10}  -",
            skill.name, "skill", chars, entry.calls, entry.ok, entry.err, ago(entry.last_used)
        );
    }
    // Memory notes ride the same frozen index as skills, so a human pricing
    // the prompt should see each one's index-line cost beside the skills.
    // Access counts live in the activation log (openmax --recall), not the
    // call counters, so the call columns stay blank.
    for (name, chars) in &breakdown.memory {
        println!(
            "{:<24} {:<6} {:>12} {:>7} {:>5} {:>5} {:>10}  -",
            name, "memory", chars, "-", "-", "-", "-"
        );
    }
    println!(
        "\nprompt_chars are paid on every request while the extension is installed.\n{} recorded calls total. Delete what you do not use; openmax --ledger keeps the history restorable.",
        usage.total_calls
    );
    if capped_out > 0 {
        println!(
            "{capped_out} skill(s) show 0 prompt_chars: past the index byte cap, so not in the frozen prompt and free until others shrink; the agent cannot see them either (openmax --check names them)."
        );
    }
    if registry.skills_omitted > 0 {
        println!(
            "{} more skill(s) were discovered beyond the {}-skill index cap: not listed, not indexed, and costing nothing.",
            registry.skills_omitted,
            open_max_core::skills::MAX_SKILLS
        );
    }
}

fn ensure_project_trust(
    cli: &CliArgs,
    data_dir: &std::path::Path,
    project: &std::path::Path,
) -> Result<(), String> {
    // Trust grants are human actions. Any process the agent loop spawns
    // carries OPENMAX_SESSION, and both the flag and the interactive prompt
    // refuse under it: a session must not be able to grant itself (or a
    // child) trust in a new directory.
    let agent_spawned = std::env::var_os("OPENMAX_SESSION").is_some();
    if cli.trust_project {
        if agent_spawned {
            return Err(format!(
                "trust grants are human actions: this process was started from an agent session; ask the user to run `openmax --trust-project` in {}",
                project.display()
            ));
        }
        // Same second wall as --approve: a human at a terminal, or an
        // explicit attestation for automation a human runs.
        require_human(
            "a trust grant",
            &format!("openmax --trust-project (in {})", project.display()),
        );
        let trusted = open_max_core::trust::trust_project(data_dir, project)?;
        eprintln!("openmax: trusted project {}", trusted.display());
        return Ok(());
    }
    if open_max_core::trust::is_trusted(data_dir, project)? {
        return Ok(());
    }
    if agent_spawned || cli.print || cli.stdio || !std::io::stdin().is_terminal() {
        return Err(format!(
            "project {} is not trusted; inspect it, then rerun with --trust-project",
            project.display()
        ));
    }

    eprint!(
        "Open Max can execute repository code with your user authority.\nTrust project {}? [y/N] ",
        project.display()
    );
    std::io::stderr().flush().map_err(|e| e.to_string())?;
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| e.to_string())?;
    if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        return Err("project remains untrusted".into());
    }
    open_max_core::trust::trust_project(data_dir, project)?;
    Ok(())
}

/// A data directory no other test can be handed. Tests delete their directory
/// when they finish, so a name two of them share makes one delete the other's
/// fixture mid-run. A timestamp alone does not separate them: `SystemTime`
/// advances in microsecond steps on macOS, and parallel test threads routinely
/// read the same value. The counter is what actually guarantees uniqueness;
/// pid and clock only keep leftovers from earlier runs out of the way.
/// One row of the `--check` human report. The whole row rides one terminal
/// line: a file's own name is author-controlled bytes (write_file only trims
/// the ends of a path), so a control character in it could forge a second,
/// clean-looking row. One space per control byte, the same rule hook-authored
/// stderr text gets. The `--json` face serializes through serde and needs no
/// counterpart.
fn check_row(f: &open_max_core::doctor::Finding) -> String {
    use open_max_core::doctor::Status;
    let path = f.path.display();
    let row = match &f.status {
        Status::Ok(summary) => format!("ok   {:<11} {}  ({summary})", f.kind, path),
        Status::Warn(reason) => format!("warn {:<11} {}  {reason}", f.kind, path),
        Status::Err(reason) => format!("err  {:<11} {}  {reason}", f.kind, path),
    };
    open_max_core::text::one_line(&row)
}

/// One verdict row of `--check --run-examples`, one-lined by the same rule:
/// the tool name is the manifest's own declaration and a failure reason often
/// carries the example's captured stderr, both author-controlled bytes.
fn example_row(verdict: &open_max_core::doctor::ExampleVerdict) -> String {
    let badge = match verdict.sandboxed {
        // Loud by design: the probe ran UNAPPROVED content with zero
        // host authority; nothing was blessed by it running.
        true => format!(
            "  [sandboxed probe: unapproved content ran with no network, writes confined; in-session calls still prompt until: {}]",
            approve_command(&verdict.path)
        ),
        false => String::new(),
    };
    let row = match &verdict.result {
        Ok(()) => format!("ok   example     {}{badge}", verdict.tool),
        // A sandboxed probe denies the network and any write outside its
        // scratch dir (ADR-0011), so a tool that legitimately needs either
        // cannot pass one - and a failure here is inconclusive, not proof
        // the tool is broken. A passing probe approves nothing; a failing
        // one condemns nothing. Report it as a warning and let the real
        // run after approval be the honest signal, rather than failing the
        // check on the largest tool family (anything that reaches the
        // network).
        Err(reason) if verdict.sandboxed => format!(
            "warn example     {}  could not be proven in the sandbox (a tool that needs the network or a write outside its scratch dir cannot): {reason}{badge}",
            verdict.tool
        ),
        // Approved content ran with the host's authority: a failure is real.
        Err(reason) => format!("err  example     {}  {reason}{badge}", verdict.tool),
    };
    open_max_core::text::one_line(&row)
}

#[cfg(test)]
pub fn test_temp_dir(prefix: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nonce}-{seq}", std::process::id()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn approve_commands_quote_metacharacter_paths() {
        // The path is agent-chosen; the printed command is pastable.
        assert_eq!(
            super::approve_command(std::path::Path::new("/tmp/my probe dir/gate$(x).toml")),
            "openmax --approve '/tmp/my probe dir/gate$(x).toml'"
        );
    }

    /// The path column is the file's own name, author-controlled bytes: a
    /// control character in it (or in a diagnostic built from file content)
    /// must not forge a second, clean-looking report row or restyle the line.
    #[test]
    fn a_check_row_is_one_line_whatever_the_file_is_named() {
        use open_max_core::doctor::{Finding, Status};
        let forged = Finding {
            kind: "tool",
            path: std::path::PathBuf::from(
                ".openmax/tools/a\nok   tool        forged.toml  (schema ok)\u{1b}[31m.toml",
            ),
            status: Status::Err("reason with\r\ncontrol bytes".into()),
        };
        let row = super::check_row(&forged);
        assert!(
            row.chars().all(|c| !c.is_control()),
            "a control byte survived into the row: {row:?}"
        );
        assert!(row.starts_with("err  tool"), "{row:?}");
        assert!(row.ends_with("control bytes"), "{row:?}");

        // The delta from the crate's old private flattener: U+2028 and U+2029
        // are line breaks to a renderer and are NOT control characters, so an
        // `is_control`-only rule let them through. The workspace now has one
        // sanitizer and this row inherits its full rule.
        let separators = Finding {
            kind: "tool",
            path: std::path::PathBuf::from(
                ".openmax/tools/a\u{2028}ok   tool        forged.toml\u{2029}b.toml",
            ),
            status: Status::Err("reason".into()),
        };
        let row = super::check_row(&separators);
        assert!(
            !row.contains('\u{2028}') && !row.contains('\u{2029}'),
            "a Unicode line separator survived into the row: {row:?}"
        );

        let ok = Finding {
            kind: "skill",
            path: std::path::PathBuf::from(".agents/skills/deploy/SKILL.md"),
            status: Status::Ok("deploy: ship the current branch".into()),
        };
        assert_eq!(
            super::check_row(&ok),
            "ok   skill       .agents/skills/deploy/SKILL.md  (deploy: ship the current branch)"
        );
    }

    /// Same rule for the example verdicts: the tool name is the manifest's
    /// own declaration and a failure reason often carries captured stderr.
    #[test]
    fn an_example_row_is_one_line_whatever_the_tool_declares() {
        use open_max_core::doctor::ExampleVerdict;
        let forged = ExampleVerdict {
            tool: "evil\nok   example     forged".into(),
            path: std::path::PathBuf::from("/tmp/probe/tool.toml"),
            result: Err("exit 1\n\u{1b}[31mboom\r\nsecond line".into()),
            sandboxed: false,
        };
        let row = super::example_row(&forged);
        assert!(
            row.chars().all(|c| !c.is_control()),
            "a control byte survived into the row: {row:?}"
        );
        assert!(row.starts_with("err  example"), "{row:?}");

        let clean = ExampleVerdict {
            tool: "docsearch".into(),
            path: std::path::PathBuf::from("/tmp/probe/tool.toml"),
            result: Ok(()),
            sandboxed: false,
        };
        assert_eq!(super::example_row(&clean), "ok   example     docsearch");
    }

    /// --help's --spec list names every surface the binary accepts: --check
    /// rows send readers to `openmax --spec settings`, and the help text was
    /// the one place that did not know it existed.
    #[test]
    fn help_names_every_spec_surface() {
        // The parenthesized list after "--spec <surface>" is parsed and
        // compared as a SET against spec::SURFACES: a fixed window plus
        // substring checks could be masked by unrelated prose (settings.json)
        // or defeated by a harmless reflow.
        let start = super::HELP.find("--spec <surface>").expect("--spec is documented");
        let open = super::HELP[start..].find('(').expect("the option lists its surfaces") + start;
        let close = super::HELP[open..].find(')').expect("the list closes") + open;
        let mut listed: Vec<&str> = super::HELP[open + 1..close]
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        listed.sort_unstable();
        let mut surfaces: Vec<&str> = open_max_core::spec::SURFACES.to_vec();
        surfaces.sort_unstable();
        assert_eq!(listed, surfaces, "--help's --spec list must equal spec::SURFACES");
    }

    use super::*;
    use std::sync::{Arc, Mutex};

    /// An approvals-only ledger has no objects store because nothing ever
    /// stored bytes; `--ledger` must not warn that "no version can be
    /// restored" over records that were never restorable.
    #[test]
    fn approvals_only_ledger_reports_no_missing_objects() {
        let objects = std::path::Path::new("/data/ledger/objects");
        let none: &[(String, std::path::PathBuf)] = &[];
        assert_eq!(ledger_objects_summary(false, none, 0, 0, objects), None);

        // With stored bytes recorded, a missing store is real damage.
        let gone = ledger_objects_summary(false, none, 2, 0, objects).unwrap();
        assert!(gone.contains("is gone"), "{gone}");

        // A present store with intact objects prints a runnable cp per object,
        // using the FULL sha (the object filename) and shell-quoting both
        // sides, not the bare un-runnable template the history's 12-char prefix
        // never satisfied.
        let restore = vec![(
            "a".repeat(64),
            std::path::PathBuf::from("/proj/.openmax/hooks/x's gate.toml"),
        )];
        let ok = ledger_objects_summary(true, &restore, 1, 0, objects).unwrap();
        assert!(ok.contains("copying its object back"), "{ok}");
        assert!(
            ok.contains(&format!("cp '/data/ledger/objects/{}'", "a".repeat(64))),
            "the cp names the full-sha object path: {ok}"
        );
        assert!(ok.contains(r"'/proj/.openmax/hooks/x'\''s gate.toml'"), "quoted target: {ok}");
        assert!(!ok.contains("<sha>"), "no un-runnable placeholder remains: {ok}");
        assert!(!ok.contains("warning"), "{ok}");

        // A present store with nothing intact keeps the pointer, no recipe.
        let bare = ledger_objects_summary(true, none, 0, 0, objects).unwrap();
        assert!(bare.contains("objects: /data/ledger/objects"), "{bare}");
        assert!(!bare.contains("cp "), "{bare}");

        let hurt = ledger_objects_summary(true, &restore, 1, 1, objects).unwrap();
        assert!(hurt.contains("1 record(s) have no trustworthy object"), "{hurt}");
    }

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Retiring an approval takes the path typed back, not a keystroke: the
    /// wall is that a person read which policy is being removed. Both
    /// spellings they can see count, and nothing else does.
    #[test]
    fn retiring_an_approval_takes_the_path_typed_back() {
        let resolved = std::path::Path::new("/proj/.openmax/hooks/gate.toml");
        let given = ".openmax/hooks/gate.toml";
        assert!(forget_confirmed(".openmax/hooks/gate.toml\n", given, resolved));
        assert!(forget_confirmed("  /proj/.openmax/hooks/gate.toml  \n", given, resolved));

        for answer in ["y\n", "yes\n", "\n", "  \n", "gate.toml\n", ".openmax/hooks/other.toml\n"] {
            assert!(
                !forget_confirmed(answer, given, resolved),
                "{answer:?} must not retire a policy"
            );
        }
    }

    /// Trust grants are human actions: under an agent-spawned process
    /// (OPENMAX_SESSION set), --trust-project must refuse rather than record.
    #[test]
    fn trust_project_refuses_under_an_agent_session() {
        let dir = test_temp_dir("openmax-trustgate");
        std::fs::create_dir_all(&dir).unwrap();
        let cli = parse_args_from(["--trust-project"]).unwrap();
        std::env::set_var("OPENMAX_SESSION", "1");
        let result = ensure_project_trust(&cli, &dir.join("data"), &dir);
        std::env::remove_var("OPENMAX_SESSION");
        let err = result.unwrap_err();
        assert!(err.contains("human actions"), "{err}");
        assert!(
            !open_max_core::trust::is_trusted(&dir.join("data"), &dir).unwrap_or(true),
            "no trust may be recorded"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn frame_writer_flushes_buffered_bytes_on_normal_drop() {
        let sink = Sink::default();
        {
            let mut w = FrameWriter::new(sink.clone(), 1024);
            w.write_all(b"complete frame").unwrap();
        }
        assert_eq!(sink.0.lock().unwrap().as_slice(), b"complete frame");
    }

    #[test]
    fn frame_writer_discards_partial_frame_when_dropped_panicking() {
        let sink = Sink::default();
        let inner = sink.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut w = FrameWriter::new(inner, 1024);
            w.write_all(b"half a frame of escape bytes").unwrap();
            panic!("draw failed");
        }));
        assert!(result.is_err());
        // The restored shell must never receive the abandoned frame.
        assert!(sink.0.lock().unwrap().is_empty());
    }

    /// ratatui's Terminal shows the cursor when it drops, which flushes the
    /// backend. Unwinding from a UI panic drops it after the hook restored
    /// the shell, so that flush must not land the half-drawn frame (with its
    /// open synchronized update, which makes the terminal hold all output).
    #[test]
    fn frame_writer_flushed_while_panicking_writes_nothing() {
        struct ShowCursorOnDrop(FrameWriter<Sink>);
        impl Drop for ShowCursorOnDrop {
            fn drop(&mut self) {
                let _ = execute!(self.0, crossterm::cursor::Show);
            }
        }
        let sink = Sink::default();
        let inner = sink.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut term = ShowCursorOnDrop(FrameWriter::new(inner, 1024));
            crossterm::queue!(term.0, crossterm::terminal::BeginSynchronizedUpdate).unwrap();
            term.0.write_all(b"half a frame").unwrap();
            panic!("draw failed");
        }));
        assert!(result.is_err());
        let written = sink.0.lock().unwrap();
        assert!(written.is_empty(), "flushed mid-panic: {:?}", String::from_utf8_lossy(&written));
    }

    fn ansi(command: impl crossterm::Command) -> String {
        let mut out = String::new();
        command.write_ansi(&mut out).unwrap();
        out
    }

    /// Every mode the fullscreen session switched on is switched off by the
    /// one restore, and only once: the panic hook and the exit path can both
    /// run it. A mode left on outlives the process, and mouse reporting above
    /// all turns every later mouse move into escape bytes at the shell prompt.
    #[test]
    fn restore_disables_every_enabled_mode_exactly_once() {
        let all = MODE_RAW
            | MODE_TITLE
            | MODE_ALT_SCREEN
            | MODE_KEYBOARD
            | MODE_PASTE
            | MODE_MOUSE
            | MODE_FOCUS;
        let modes = AtomicU8::new(all);
        let mut out = Vec::new();
        let claimed = write_restore(&mut out, &modes);
        let out = String::from_utf8(out).unwrap();
        for (what, seq) in [
            ("synchronized update", ansi(crossterm::terminal::EndSynchronizedUpdate)),
            ("hidden cursor", ansi(crossterm::cursor::Show)),
            ("focus reports", ansi(crossterm::event::DisableFocusChange)),
            ("mouse capture", ansi(DisableMouseCapture)),
            ("bracketed paste", ansi(crossterm::event::DisableBracketedPaste)),
            ("keyboard flags", ansi(PopKeyboardEnhancementFlags)),
            ("alternate screen", ansi(crossterm::terminal::LeaveAlternateScreen)),
            ("title stack", "\x1b[23;0t".to_string()),
        ] {
            assert!(out.contains(&seq), "{what} is left on: {out:?}");
        }
        // Terminals keep one keyboard flag stack per screen, so the pop has
        // to land while the alternate screen is still current.
        let pop = out.find(&ansi(PopKeyboardEnhancementFlags));
        let leave = out.find(&ansi(crossterm::terminal::LeaveAlternateScreen));
        assert!(pop < leave, "{out:?}");
        // Raw mode is termios state, not bytes: the caller leaves it.
        assert_eq!(claimed, all);

        let mut again = Vec::new();
        assert_eq!(write_restore(&mut again, &modes), 0);
        assert!(again.is_empty(), "a second restore wrote {:?}", String::from_utf8_lossy(&again));
    }

    /// Restore undoes only what was switched on: init can fail before the
    /// alternate screen, and a terminal without the keyboard protocol never
    /// had flags pushed.
    #[test]
    fn restore_leaves_alone_modes_never_enabled() {
        let mut out = Vec::new();
        write_restore(&mut out, &AtomicU8::new(MODE_RAW | MODE_TITLE));
        assert_eq!(String::from_utf8_lossy(&out), "\x1b[23;0t");

        let mut out = Vec::new();
        let no_keyboard = MODE_TITLE | MODE_ALT_SCREEN | MODE_PASTE | MODE_MOUSE | MODE_FOCUS;
        write_restore(&mut out, &AtomicU8::new(no_keyboard));
        let out = String::from_utf8(out).unwrap();
        assert!(!out.contains(&ansi(PopKeyboardEnhancementFlags)), "{out:?}");
    }

    /// The panic hook is process-global, and the core survives panics on its
    /// own worker threads (a turn task, a tool) by reporting them as errors
    /// while the TUI keeps drawing. Restoring there would drop the live
    /// session out of the alternate screen and leave it painting over the
    /// shell, so only a panic on the thread that drives the UI restores.
    #[test]
    fn panic_hook_restores_only_for_the_ui_thread() {
        use std::sync::atomic::AtomicUsize;
        static RESTORES: AtomicUsize = AtomicUsize::new(0);
        install_panic_restore(std::thread::current().id(), || {
            RESTORES.fetch_add(1, Ordering::SeqCst);
        });

        let worker = std::thread::spawn(|| panic!("a worker panic the core survives"));
        assert!(worker.join().is_err());
        assert_eq!(RESTORES.load(Ordering::SeqCst), 0, "a worker panic restored the terminal");

        let ui = std::panic::catch_unwind(|| panic!("a UI panic"));
        assert!(ui.is_err());
        assert_eq!(RESTORES.load(Ordering::SeqCst), 1, "a UI panic must restore the terminal");
    }

    /// Two exit paths can restore at once: the normal exit after a first
    /// signal, and the forced exit on a second one. A restore claims the
    /// modes only while it holds stdout, so a caller that finds nothing left
    /// to undo knows the other restore has finished, raw mode included.
    /// Claiming them outside the lock let the forced exit find them claimed
    /// but not yet undone, and leave the shell in raw mode.
    #[test]
    fn restore_claims_the_modes_only_while_holding_stdout() {
        let held = std::io::stdout().lock();
        // Raw mode alone writes no bytes, and leaving it is a no-op in a
        // process that never entered it.
        TERM_MODES.fetch_or(MODE_RAW, Ordering::SeqCst);
        let restore = std::thread::spawn(restore_terminal);
        std::thread::sleep(std::time::Duration::from_millis(200));
        let unclaimed = TERM_MODES.load(Ordering::SeqCst);
        drop(held);
        restore.join().unwrap();
        assert_eq!(
            unclaimed, MODE_RAW,
            "a restore claimed the modes while another caller held stdout"
        );
        assert_eq!(TERM_MODES.load(Ordering::SeqCst), 0);
    }

    /// The forced exit on a second signal can restore while startup is
    /// entering raw mode. Raw mode is termios state, so the restore leaves it
    /// only when its bit is recorded; a restore landing after the switch but
    /// before the bit would find nothing to undo and exit with the shell raw.
    /// The restore here stands in for `restore_terminal`, which claims the
    /// modes only while holding stdout (see
    /// `restore_claims_the_modes_only_while_holding_stdout`).
    #[test]
    fn raw_mode_is_recorded_before_a_racing_restore_can_claim_it() {
        let modes = &AtomicU8::new(0);
        let claimed = std::thread::scope(|s| {
            let mut restore = None;
            enter_raw_mode(modes, || {
                restore = Some(s.spawn(move || {
                    let _out = std::io::stdout().lock();
                    write_restore(&mut std::io::sink(), modes)
                }));
                // Holds the window between the switch and the record open.
                std::thread::sleep(std::time::Duration::from_millis(200));
                Ok(())
            })
            .unwrap();
            restore.unwrap().join().unwrap()
        });
        assert_eq!(claimed, MODE_RAW, "a restore found raw mode entered but not recorded");
    }

    /// SIGTERM's default action ends the process on the spot with every
    /// terminal mode still on; the watcher hands it to the event loop as a
    /// quit, so the normal exit path restores the terminal. The watcher must
    /// outlive the runtime that started it: when main returns, that runtime
    /// drops its tasks and then waits on blocking work still running (a
    /// grep, a scan), and the handlers it installed stay in place, so a
    /// watcher on it would leave every signal through that wait swallowed.
    /// Handlers stay installed for the life of the process and the watcher
    /// keeps waiting for a second signal, so the probe runs in a child: the
    /// test process keeps the default action for every later signal.
    #[cfg(unix)]
    #[test]
    fn sigterm_reaches_the_quit_after_the_starting_runtime_is_gone() {
        const CHILD: &str = "OPENMAX_TEST_SIGNAL_REACHES_QUIT";
        if std::env::var_os(CHILD).is_some() {
            let runtime =
                || tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            let quit = runtime().block_on(async { watch_quit_signals() });
            let status = std::process::Command::new("kill")
                .args(["-TERM", &std::process::id().to_string()])
                .status()
                .unwrap();
            assert!(status.success());
            runtime().block_on(async {
                tokio::time::timeout(std::time::Duration::from_secs(10), quit.notified())
                    .await
                    .expect("SIGTERM must reach the event loop as a quit")
            });
            return;
        }
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::sigterm_reaches_the_quit_after_the_starting_runtime_is_gone"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&child.stdout);
        // "1 passed" rules out a filter that matched nothing and exited 0.
        assert!(
            child.status.success() && stdout.contains("1 passed"),
            "SIGTERM did not reach the quit ({:?}): {stdout}{}",
            child.status,
            String::from_utf8_lossy(&child.stderr)
        );
    }

    /// Once the event loop has ended it drops the quit, and nothing reads it
    /// again, while the runtime still waits on blocking work (a grep, a scan)
    /// before the process can exit. A signal then has no loop to reach, so it
    /// must end the process at once; handing it to the gone loop would leave
    /// the first Ctrl+C or SIGTERM doing nothing until that work finished.
    /// The exit is the behavior under test, so the probe runs in a child.
    #[cfg(unix)]
    #[test]
    fn sigterm_exits_at_once_after_the_event_loop_is_gone() {
        use tokio::signal::unix::SignalKind;
        const CHILD: &str = "OPENMAX_TEST_SIGNAL_AFTER_LOOP";
        let sleep = std::time::Duration::from_secs(10);
        if std::env::var_os(CHILD).is_some() {
            drop(watch_quit_signals());
            let status = std::process::Command::new("kill")
                .args(["-TERM", &std::process::id().to_string()])
                .status()
                .unwrap();
            assert!(status.success());
            // Stands in for the blocking work the runtime waits on.
            std::thread::sleep(sleep);
            return;
        }
        let start = std::time::Instant::now();
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::sigterm_exits_at_once_after_the_event_loop_is_gone"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert_eq!(
            child.status.code(),
            Some(128 + SignalKind::terminate().as_raw_value()),
            "SIGTERM after the event loop ended did not exit at once ({:?}): {}",
            start.elapsed(),
            String::from_utf8_lossy(&child.stdout)
        );
        assert!(start.elapsed() < sleep);
    }

    #[test]
    fn single_print_prompt_is_one_turn() {
        let cli = parse_args_from(["-p", "summarize this repo"]).unwrap();
        assert!(cli.print);
        assert_eq!(cli.prompts, vec!["summarize this repo"]);
    }

    #[test]
    fn multi_token_print_prompt_joins() {
        let cli = parse_args_from(["-p", "summarize", "this", "repo"]).unwrap();
        assert_eq!(cli.prompts, vec!["summarize this repo"]);
    }

    #[test]
    fn repeated_print_flags_collect_multiple_turns() {
        let cli = parse_args_from(["-p", "first", "-p", "second"]).unwrap();
        assert!(cli.print);
        assert_eq!(cli.prompts, vec!["first", "second"]);
    }

    #[test]
    fn repeated_print_with_multi_token_groups() {
        let cli = parse_args_from(["-p", "list", "crates", "-p", "summarize", "the", "first"]).unwrap();
        assert_eq!(
            cli.prompts,
            vec!["list crates".to_string(), "summarize the first".to_string()]
        );
    }

    #[test]
    fn print_json_then_prompt_still_one_turn() {
        let cli = parse_args_from(["-p", "--json", "list top-level files"]).unwrap();
        assert!(cli.print);
        assert!(cli.json);
        assert_eq!(cli.prompts, vec!["list top-level files"]);
    }

    #[test]
    fn multi_print_with_json() {
        let cli = parse_args_from(["-p", "--json", "one", "-p", "two"]).unwrap();
        assert!(cli.json);
        assert_eq!(cli.prompts, vec!["one", "two"]);
    }

    #[test]
    fn stdio_flag_parses_alone_and_with_continue() {
        let cli = parse_args_from(["--stdio"]).unwrap();
        assert!(cli.stdio && !cli.print);
        let cli = parse_args_from(["--stdio", "-c"]).unwrap();
        assert!(cli.stdio && cli.continue_session);
    }

    #[test]
    fn check_flag_parses() {
        let cli = parse_args_from(["--check"]).unwrap();
        assert!(cli.check && !cli.print && !cli.stdio);
    }

    #[test]
    fn check_json_is_a_valid_combination() {
        let cli = parse_args_from(["--check", "--json"]).unwrap();
        assert!(cli.check);
        assert!(cli.json);
    }

    #[test]
    fn spec_flag_takes_a_surface_and_requires_a_value() {
        let cli = parse_args_from(["--spec", "hooks"]).unwrap();
        assert_eq!(cli.spec.as_deref(), Some("hooks"));
        assert!(!cli.check && !cli.print && !cli.stdio);
        assert!(parse_args_from(["--spec"]).is_err());
    }

    #[test]
    fn check_stdio_flag_parses_conformance_mode() {
        // --check --stdio selects protocol-stream validation; it is not the
        // interactive stdio session and takes no prompt.
        let cli = parse_args_from(["--check", "--stdio"]).unwrap();
        assert!(cli.check && cli.stdio && !cli.print && cli.prompts.is_empty());
    }

    #[test]
    fn trust_project_flag_parses() {
        let cli = parse_args_from(["--trust-project", "-p", "inspect"]).unwrap();
        assert!(cli.trust_project);
        assert!(cli.print);
    }

    #[test]
    fn empty_print_group_is_rejected() {
        assert!(parse_args_from(["-p", "-p", "second"]).is_err());
        assert!(parse_args_from(["-p"]).is_err());
    }

    /// An operation that names one thing to act on, given twice, kept only
    /// the second value: `--approve a.toml --approve b.toml` approved b.toml
    /// alone and exited 0, leaving the gate in a.toml inert while the human
    /// believed it approved.
    #[test]
    fn an_operation_given_twice_is_refused_naming_it() {
        for flag in ["--approve", "--forget", "--spec", "--recall"] {
            let err = parse_args_from([flag, "a", flag, "b"])
                .err()
                .unwrap_or_else(|| panic!("{flag} given twice must be refused"))
                .to_string();
            assert!(err.contains(flag) && err.contains("twice"), "{flag}: {err}");
        }
        // A model or provider given twice keeps the conventional last one.
        let cli = parse_args_from(["-m", "a", "-m", "b", "--provider", "x", "--provider", "y"]).unwrap();
        assert_eq!((cli.model.as_deref(), cli.provider.as_deref()), (Some("b"), Some("y")));
    }

    /// Every operation a command line can select, as its args and the name a
    /// refusal gives it. `-p` comes last in any line built from these: the
    /// tokens after it are its prompt. An empty name is the interactive
    /// session.
    const OPERATIONS: [(&[&str], &str); 11] = [
        (&[], ""),
        (&["--spec", "tools"], "--spec"),
        (&["--recall", "q"], "--recall"),
        (&["--ledger"], "--ledger"),
        (&["--ledger-repair"], "--ledger-repair"),
        (&["--approve", "f"], "--approve"),
        (&["--forget", "f"], "--forget"),
        (&["--check"], "--check"),
        (&["--check", "--stdio"], "--check --stdio"),
        (&["--stdio"], "--stdio"),
        (&["-p", "x"], "--print"),
    ];

    /// Whether a refusal names `flag` as a whole flag, so `--ledger` is not
    /// satisfied by `--ledger-repair`.
    fn names(reason: &str, flag: &str) -> bool {
        reason.match_indices(flag).any(|(at, _)| {
            reason[at + flag.len()..].chars().next().is_none_or(|c| matches!(c, ' ' | ';' | ','))
        })
    }

    fn refusal_of(args: &[&str]) -> Option<String> {
        refusal(&parse_args_from(args.iter().copied()).unwrap())
    }

    /// Every pair of operations is refused before either runs, naming both:
    /// each one exits when it is done, so the second was dropped and the run
    /// exited as if it had happened (`--check --ledger` validated nothing and
    /// exited 0).
    #[test]
    fn every_pair_of_operations_is_refused_naming_both() {
        // The interactive session is what runs when no operation is named,
        // so it cannot be paired, and `--check --stdio` is already one.
        let operations: Vec<_> = OPERATIONS
            .iter()
            .filter(|(_, name)| !name.is_empty() && *name != "--check --stdio")
            .collect();
        for (i, (a, a_name)) in operations.iter().enumerate() {
            assert_eq!(refusal_of(a), None, "{a:?} alone is a valid command line");
            for (b, b_name) in &operations[i + 1..] {
                let args: Vec<&str> = a.iter().chain(b.iter()).copied().collect();
                let reason = refusal_of(&args);
                if (*a_name, *b_name) == ("--check", "--stdio") {
                    assert_eq!(reason, None, "--check --stdio validates a protocol stream");
                    continue;
                }
                let reason = reason.unwrap_or_else(|| panic!("{args:?} must be refused"));
                assert!(names(&reason, a_name) && names(&reason, b_name), "{args:?}: {reason}");
            }
        }
        // A third operation is refused even beside the one valid pair.
        let reason = refusal_of(&["--check", "--stdio", "--ledger"]).unwrap();
        assert!(names(&reason, "--check --stdio") && names(&reason, "--ledger"), "{reason}");
    }

    /// An option only another operation reads was dropped as silently as a
    /// second operation was (`--ledger -m x` exited 0 having used no model).
    /// Each option is placed against every operation: refused, naming both,
    /// wherever it would be ignored, and accepted wherever it is read.
    #[test]
    fn an_option_the_operation_does_not_read_is_refused_naming_both() {
        let sessions: &[&str] = &["", "--stdio", "--print"];
        let options: [(&[&str], &str, &[&str]); 7] = [
            (&["--trust-project"], "--trust-project", sessions),
            (&["--continue"], "--continue", sessions),
            (&["-m", "m"], "--model", sessions),
            (&["--provider", "p"], "--provider", sessions),
            (&["--json"], "--json", &["--print", "--check", "--recall"]),
            (&["--run-examples"], "--run-examples", &["--check"]),
            // A bare word is a prompt, which only --print reads.
            (&["extra"], "", &["--print"]),
        ];
        for (option, flag, readers) in options {
            for (operation, name) in OPERATIONS {
                let args: Vec<&str> = option.iter().chain(operation.iter()).copied().collect();
                let reason = refusal_of(&args);
                if readers.contains(&name) {
                    assert_eq!(reason, None, "{args:?} is a valid command line");
                    continue;
                }
                let reason = reason.unwrap_or_else(|| panic!("{args:?} must be refused"));
                assert!(
                    (flag.is_empty() || names(&reason, flag))
                        && (name.is_empty() || names(&reason, name)),
                    "{args:?}: {reason}"
                );
            }
        }
        // The combinations the docs show still run.
        for args in [
            &["--check", "--json", "--run-examples"][..],
            &["--trust-project", "--continue", "-m", "m", "--provider", "p", "-p", "--json", "x"],
            &["--trust-project", "--continue", "--stdio"],
            &["--recall", "q", "--json"],
        ] {
            assert_eq!(refusal_of(args), None, "{args:?} is a valid command line");
        }
    }

    /// With no operation named, a stray word is most likely a prompt missing
    /// its -p, so that refusal points at --print. Beside a named operation it
    /// is most likely meant for that operation (`--check x.toml`,
    /// `--check --stdio stream.jsonl`), so pointing at --print sends the user
    /// the wrong way: the refusal names the word instead, and an operation
    /// that reads stdin says so.
    #[test]
    fn a_stray_word_beside_an_operation_is_named_not_sent_to_print() {
        for (operation, name) in OPERATIONS {
            if matches!(name, "" | "--print") {
                continue;
            }
            let args: Vec<&str> = operation.iter().copied().chain(["stray.toml"]).collect();
            let reason = refusal_of(&args).unwrap_or_else(|| panic!("{args:?} must be refused"));
            assert!(
                names(&reason, name) && reason.contains("'stray.toml'") && !reason.contains("--print"),
                "{args:?}: {reason}"
            );
        }
        for args in [&["--stdio", "x"][..], &["--check", "--stdio", "stream.jsonl"]] {
            let reason = refusal_of(args).unwrap();
            assert!(reason.contains("stdin"), "{args:?}: {reason}");
        }
        let reason = refusal_of(&["stray"]).unwrap();
        assert!(reason.contains("--print"), "a stray word alone is likely a prompt: {reason}");
    }
}
