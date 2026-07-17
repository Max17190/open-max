mod app;
mod catalog;
mod clipboard;
mod completion;
mod headless;
mod input;
mod stdio;
mod theme;
mod ui;

use std::ffi::OsString;

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
      --json             with --print, emit AgentEvent envelopes as JSONL
      --stdio            bidirectional JSONL session: commands on stdin
                         ({\"cmd\":\"user\"|\"approve\"|\"cancel\"|\"quit\"}), AgentEvent
                         envelopes on stdout; the custom-frontend protocol
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

    if cli.json && !cli.print {
        eprintln!("openmax: --json requires --print\n\n{HELP}");
        std::process::exit(2);
    }
    if cli.stdio && (cli.print || !cli.prompts.is_empty()) {
        eprintln!("openmax: --stdio takes commands on stdin, not flags or prompts\n\n{HELP}");
        std::process::exit(2);
    }

    let (core, core_rx) = Core::new(default_data_dir());
    {
        let mut s = core.settings.lock().unwrap();
        if let Some(provider) = &cli.provider {
            s.provider = Some(provider.clone());
        }
        if let Some(model) = &cli.model {
            s.model = model.clone();
            s.mlx_model = model.clone();
        }
        // Fail fast on an explicit but unknown --provider (no silent flat fallback).
        if let Err(e) = open_max_core::providers::resolve(&s, &core.data_dir) {
            eprintln!("openmax: {e}");
            std::process::exit(2);
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
    let terminal = ratatui::init();

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn empty_print_group_is_rejected() {
        assert!(parse_args_from(["-p", "-p", "second"]).is_err());
        assert!(parse_args_from(["-p"]).is_err());
    }
}
