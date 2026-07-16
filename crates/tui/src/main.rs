mod app;
mod catalog;
mod headless;
mod input;
mod theme;
mod ui;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use open_max_core::state::{default_data_dir, Core};

const HELP: &str = "openmax: a barebones high-performance agent harness

usage: openmax [options] [prompt...]

options:
  -c, --continue      resume the latest session in this directory
  -m, --model <id>    use this model id for the run
  -p, --print         headless: run one turn and exit (prompt required)
      --json          with --print, emit AgentEvent envelopes as JSONL
  -V, --version       print the version
  -h, --help          this help

run it inside a project directory; point base_url at any OpenAI-compatible
endpoint in ~/.openmax/settings.json. /help lists in-session commands.

examples:
  openmax
  openmax -p \"summarize this repo\"
  openmax -p --json \"list top-level files\"";

struct CliArgs {
    continue_session: bool,
    model: Option<String>,
    print: bool,
    json: bool,
    /// Free-form prompt tokens (joined with spaces) for --print.
    prompt: Vec<String>,
}

fn parse_args() -> Result<CliArgs, lexopt::Error> {
    use lexopt::prelude::*;
    let mut args = CliArgs {
        continue_session: false,
        model: None,
        print: false,
        json: false,
        prompt: Vec::new(),
    };
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') | Long("continue") => args.continue_session = true,
            Short('m') | Long("model") => args.model = Some(parser.value()?.string()?),
            Short('p') | Long("print") => args.print = true,
            Long("json") => args.json = true,
            Short('V') | Long("version") => {
                println!("openmax {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            Short('h') | Long("help") => {
                println!("{HELP}");
                std::process::exit(0);
            }
            Value(v) => args.prompt.push(v.string()?),
            _ => return Err(arg.unexpected()),
        }
    }
    Ok(args)
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

    let (core, core_rx) = Core::new(default_data_dir());
    if let Some(model) = &cli.model {
        let mut s = core.settings.lock().unwrap();
        s.model = model.clone();
        s.mlx_model = model.clone();
    }

    if cli.print {
        let prompt = cli.prompt.join(" ");
        if prompt.trim().is_empty() {
            eprintln!("openmax: --print requires a prompt\n\n{HELP}");
            std::process::exit(2);
        }
        let code = headless::run(
            core,
            core_rx,
            headless::HeadlessArgs {
                prompt,
                continue_session: cli.continue_session,
                json: cli.json,
            },
        )
        .await;
        std::process::exit(code);
    }

    if !cli.prompt.is_empty() {
        eprintln!("openmax: unexpected arguments (use --print for headless)\n\n{HELP}");
        std::process::exit(2);
    }

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
