mod app;
mod catalog;
mod input;
mod theme;
mod ui;

use crossterm::cursor::MoveTo;
use crossterm::event::{
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::style::Print;
use crossterm::terminal::{Clear, ClearType};
use open_max_core::state::{default_data_dir, Core};
use ratatui::{TerminalOptions, Viewport};

const VIEWPORT_ROWS: u16 = 10;

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

    // Take over the visible screen so the session starts clean, like a full
    // app: printing a screenful of newlines pushes the shell prompt and
    // launch command into scrollback (still there if you scroll up), then the
    // session begins at the top-left. The viewport starts at the top and
    // insert_before pushes it down until it reaches the bottom, after which
    // finished blocks flow into native scrollback as before.
    let rows = crossterm::terminal::size().map(|(_, r)| r).unwrap_or(24);
    let _ = execute!(
        std::io::stdout(),
        Print("\n".repeat(rows as usize)),
        MoveTo(0, 0),
        Clear(ClearType::FromCursorDown),
    );

    // Inline viewport: scrollback stays native; clamp low so tiny terminals
    // never end up with a viewport as tall as the screen.
    let height = VIEWPORT_ROWS.min(rows.saturating_sub(2)).max(4);
    let terminal = ratatui::init_with_options(TerminalOptions {
        viewport: Viewport::Inline(height),
    });

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

    let result = app::run(
        terminal,
        core,
        core_rx,
        app::Args { continue_session: cli.continue_session },
    )
    .await;

    let _ = execute!(std::io::stdout(), crossterm::event::DisableBracketedPaste);
    if enhanced {
        let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    }
    ratatui::restore();
    result
}
