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

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
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
                         ({\"cmd\":\"user\"|\"approve\"|\"cancel\"|\"quit\"}), AgentEvent
                         envelopes on stdout; the custom-frontend protocol
      --check            validate extension files (tools, skills, templates,
                         hooks, permissions, providers) and exit; nonzero if
                         any is broken.
                         with --stdio, validate a JSONL protocol stream on
                         stdin against the openmax-stdio contract instead
      --spec <surface>   print the authoring contract for one extension
                         surface and exit (tools, skills, prompts, hooks,
                         permissions, providers, stdio)
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
            Long("spec") => out.spec = Some(parser.value()?.string()?),
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

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let cli = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("openmax: {e}\n\n{HELP}");
            std::process::exit(2);
        }
    };

    if cli.json && !cli.print && !cli.check {
        eprintln!("openmax: --json requires --print or --check\n\n{HELP}");
        std::process::exit(2);
    }
    if cli.stdio && (cli.print || !cli.prompts.is_empty()) {
        eprintln!("openmax: --stdio takes commands on stdin, not flags or prompts\n\n{HELP}");
        std::process::exit(2);
    }
    if cli.check && cli.trust_project {
        eprintln!("openmax: --check and --trust-project are separate operations\n\n{HELP}");
        std::process::exit(2);
    }
    // --spec prints one contract and exits; silently swallowing any other
    // requested option (e.g. --spec hooks --check, or --spec tools -m qwen)
    // would look like success for work that never ran.
    if cli.spec.is_some()
        && (cli.check
            || cli.stdio
            || cli.print
            || cli.trust_project
            || cli.continue_session
            || cli.model.is_some()
            || cli.provider.is_some()
            || !cli.prompts.is_empty())
    {
        eprintln!("openmax: --spec is a standalone operation; run other options separately\n\n{HELP}");
        std::process::exit(2);
    }

    if let Some(surface) = &cli.spec {
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
            let array: Vec<serde_json::Value> = findings
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
            println!("{}", serde_json::Value::Array(array));
            std::process::exit(if open_max_core::doctor::has_errors(&findings) { 1 } else { 0 });
        }
        if findings.is_empty() {
            println!(
                "no extension files found (tools, skills, templates, hooks, permissions, providers)"
            );
            std::process::exit(0);
        }
        use open_max_core::doctor::Status;
        for f in &findings {
            match &f.status {
                Status::Ok(summary) => {
                    println!("ok   {:<11} {}  ({summary})", f.kind, f.path.display())
                }
                Status::Warn(reason) => {
                    println!("warn {:<11} {}  {reason}", f.kind, f.path.display())
                }
                Status::Err(reason) => {
                    println!("err  {:<11} {}  {reason}", f.kind, f.path.display())
                }
            }
        }
        // Warnings do not fail the run: a shadowed global default and a rule
        // written before its tool are both normal.
        if open_max_core::doctor::has_warnings(&findings) {
            println!(
                "\nwarn lines are files the agent loop never reads, or reads but cannot act on \
                 as written. They do not fail this check."
            );
        }
        std::process::exit(if open_max_core::doctor::has_errors(&findings) { 1 } else { 0 });
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

    if !cli.prompts.is_empty() {
        eprintln!("openmax: unexpected arguments (use --print for headless)\n\n{HELP}");
        std::process::exit(2);
    }

    theme::init();

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
    let enhanced = crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
    if enhanced {
        let _ = execute!(
            std::io::stdout(),
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        );
    }
    let _ = execute!(std::io::stdout(), crossterm::event::EnableBracketedPaste);
    // Mouse capture for wheel scrolling of the transcript. Terminals still
    // allow text selection with the usual modifier (Option on macOS).
    let _ = execute!(std::io::stdout(), EnableMouseCapture);

    let result = app::run(
        terminal,
        core,
        core_rx,
        app::Args { continue_session: cli.continue_session },
    )
    .await;

    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    let _ = execute!(std::io::stdout(), crossterm::event::DisableBracketedPaste);
    if enhanced {
        let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    }
    ratatui::restore();
    result
}

/// `ratatui::init` with one change: frame output goes through a 256 KiB
/// buffer so each flush is one write(2) instead of the dozens that `Stdout`'s
/// built-in 1 KiB line buffer produces on token-streaming frames.
/// `ratatui::restore` stays the counterpart on exit and panic; it operates on
/// the shared stdout fd, and every completed frame ends fully flushed.
fn init_terminal() -> std::io::Result<ui::transcript::Term> {
    use crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};
    // Hook first: any panic or error past raw mode must restore the shell.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        hook(info);
    }));
    enable_raw_mode()?;
    let init = || -> std::io::Result<ui::transcript::Term> {
        let mut out = FrameWriter::new(std::io::stdout(), 256 * 1024);
        execute!(out, EnterAlternateScreen)?;
        out.flush()?;
        ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(out))
    };
    init().inspect_err(|_| ratatui::restore())
}

/// A frame-sized `BufWriter` that discards, rather than flushes, its buffered
/// bytes when dropped mid-panic. The panic hook has already restored the
/// normal screen by the time unwinding drops the terminal, so flushing a
/// partial frame there would spray escape bytes over the user's shell.
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
        self.buf().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
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

fn ensure_project_trust(
    cli: &CliArgs,
    data_dir: &std::path::Path,
    project: &std::path::Path,
) -> Result<(), String> {
    if cli.trust_project {
        let trusted = open_max_core::trust::trust_project(data_dir, project)?;
        eprintln!("openmax: trusted project {}", trusted.display());
        return Ok(());
    }
    if open_max_core::trust::is_trusted(data_dir, project)? {
        return Ok(());
    }
    if cli.print || cli.stdio || !std::io::stdin().is_terminal() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

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
}
