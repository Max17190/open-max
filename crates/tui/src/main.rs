mod app;
mod catalog;
mod input;
mod theme;
mod ui;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use open_max_core::state::{default_data_dir, Core};

const HELP: &str = "openmax: a minimal terminal harness for local models

usage: openmax [options]

options:
  -c, --continue      resume the latest session in this directory
  -m, --model <repo>  use this model id for the run
  -V, --version       print the version
  -h, --help          this help

run it inside a project directory; /help lists commands.";

struct CliArgs {
    continue_session: bool,
    model: Option<String>,
}

fn parse_args() -> Result<CliArgs, lexopt::Error> {
    use lexopt::prelude::*;
    let mut args = CliArgs { continue_session: false, model: None };
    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Short('c') | Long("continue") => args.continue_session = true,
            Short('m') | Long("model") => args.model = Some(parser.value()?.string()?),
            Short('V') | Long("version") => {
                println!("openmax {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            Short('h') | Long("help") => {
                println!("{HELP}");
                std::process::exit(0);
            }
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

    let (core, core_rx) = Core::new(default_data_dir());
    if let Some(model) = &cli.model {
        let mut s = core.settings.lock().unwrap();
        s.model = model.clone();
        s.mlx_model = model.clone();
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
