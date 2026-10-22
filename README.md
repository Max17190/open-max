<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openmax-wordmark-dark.svg">
    <img src="docs/assets/openmax-wordmark.svg" alt="Open Max">
  </picture>
</p>

<p align="center"><strong>A self-extending Rust agent harness for coding in the terminal.</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/language-Rust-orange.svg" alt="Rust"></a>
</p>

Open Max is a single binary that runs a focused agent loop in your project directory and streams every tool call to the terminal. Point it at the model server you choose: local, cloud, or a private proxy. No desktop shell, no heavyweight runtime, no telemetry.

The endpoint must return structured API `tool_calls` to use tools. For local servers, configure the model's tool parser and chat template. Call markup in assistant text is displayed as text and never executed. Responses are limited to 16 MiB and 128 tool calls.


You own the endpoints, the tools, the skills, and the context.

## Features

- **Small by default.** Seven built-in tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `bash`) and a short system prompt. Old tool output is dropped before your task is, and dropped context is summarized by your own model into a compact note (heuristic digest as fallback) whose address points at the lossless archive of everything dropped — compaction is a bounded view over a record you can always read back.
- **Your model, your server.** One `base_url`, or several named endpoints in `providers.json` switched with `/model`. Works with local servers (Ollama, LM Studio, vLLM, llama.cpp), cloud gateways (OpenRouter and similar), and private proxies.
- **Trust before execution.** An exact canonical project root must be trusted before any agent turn or project behavior starts. Interactive use asks once; headless and stdio runs fail closed until a human grants trust with `--trust-project`.
- **Newly trusted projects run in auto.** Trusting a project records its approval mode, `auto` unless you pick another: tools and extensions run without per-call approval while deny rules, hooks, and execution reports stay active. Pick `ask` to have `write_file`, `edit_file`, and `bash` wait for approval, or `readonly` to block mutating tools. Approvals and permissions decide whether Open Max dispatches a tool call; they are not OS isolation.
- **File based extensions.** Drop TOML tools, `SKILL.md` skills, prompt templates, and process hooks under project or home config. No fork required. The agent writes them itself and the harness re-freezes as soon as a mutating call lands, so a tool the agent writes is a tool the agent uses on its very next step. The system prompt and the tools array stay byte-identical for the whole session, until you `/reload`: a tool, skill, or memory written after they were built rides a note at the end of each request, never stored, so a provider's prompt cache keeps the prompt and the conversation through every write.
- **File based memory.** One durable fact per file in `.openmax/memory/`, written by the agent, surfaced as an index line in future sessions, ranked by recency and frequency of observed use. Old entries fade from the index and remain searchable on disk until the user or agent deletes them. No database, no daemon, no embeddings; zero prompt cost when empty.
- **Recall over everything kept.** `openmax --recall "<query>"` searches this project's past sessions, compaction archives, digests, and memories in one bounded streaming pass — BM25 relevance fused with the same recency law, ranked excerpts, each citing the file with the full record. No index to build or maintain: the stores on disk stay the single source of truth.
- **Visible work.** Reads, greps, diffs, and shell commands stream as they happen in a fullscreen TUI. Headless print mode for scripts and CI.
- **Local sessions.** Conversation state lives under `~/.openmax/`. The harness contacts only the model endpoint you configure.
- **Offline replay.** Each session keeps an event log. Copied to any machine, `openmax --replay <log> --json` prints from it, byte for byte, what `-p --json` printed during the run, with no endpoint, settings, trust or data dir.

## Design

The design starts with one question: **What is the smallest capability Open Max must provide so the agent can construct richer behavior itself?** The answer is one focused native loop, seven primitive tools, a fast event-driven TUI, context management, and stable file and process contracts.

| Need | Construct it with |
| --- | --- |
| External service or specialized capability | A CLI-backed TOML tool plus an on-demand skill, without an MCP runtime |
| Reusable workflow or command | A `SKILL.md` package or prompt template slash command |
| Isolated or parallel work | `openmax --subagent "<task>" --worktree`, one child session in a new git worktree whose changes land on a branch, or a child `openmax -p` or `openmax --stdio` process, usually in tmux |
| Durable background work | A named tmux session that the agent can inspect and reattach |
| Planning and task state | `PLAN.md` and `TODO.md`, visible to the user and every tool |
| Lifecycle policy and events | Process hooks and permission files |
| Compaction integration | The built-in model summary plus the `compaction` hook event |
| Model endpoints | `providers.json`, including local servers, gateways, and private proxies |
| Shortcuts or a completely different UI | Prompt templates, or a custom frontend speaking `openmax-stdio/8` |

These are deliberate boundaries, not placeholders for hidden orchestration products. Open Max does not carry an MCP host, nested-agent scheduler, plan mode, background-job product, built-in TODO database, user-keybinding engine, pluggable compactor, or TUI plugin ABI. The agent composes those richer workflows from the same host tools a developer can inspect, edit, test, and remove.

## Install

Homebrew, on macOS or Linux:

```sh
brew install Max17190/tap/openmax
```

Or without Homebrew:

```sh
curl -fsSL https://useopenmax.dev/install.sh | sh
```

Or with bun or npm, if you already have a JavaScript runtime:

```sh
bun install -g openmax-cli    # or: npm install -g openmax-cli
```

The command is `openmax` either way; only the package name carries the `-cli`,
because `openmax` on npm belongs to an unrelated 2015 package. This path fetches
the same native binary through a small launcher script. Up to and including
2026.10.1 that script runs under `node` only, so with those versions this is
the one install that needs Node.js on your `PATH`, even when bun did the
installing. Releases after 2026.10.1 run it under `node` when Node.js is on
your `PATH` and under `bun` when it is not, so either runtime is enough. Bun
blocks install scripts by default, which is fine here: the binary is fetched on
first run instead, once, and cached.

Either way there is no Rust toolchain and no build. The script picks the right
prebuilt binary for your machine, checks it against the published SHA-256, and
puts `openmax` in `~/.openmax/bin` next to the settings and sessions the
harness already keeps there. It adds that directory to your `PATH` through your
shell profile, which only a new shell reads, so open one (or
`source ~/.openmax/bin/env`) before typing a bare `openmax`. The shell
installer of releases after 2026.10.1 ends with the first command to run (see
[Use](#use)), naming the binary by its absolute path, with the home
directory spelled out (`/Users/you/.openmax/bin/openmax`, for example), so it
works in the shell that ran the install.

The check uses `sha256sum`, or `shasum -a 256` where there is no `sha256sum`
(macOS 14 and earlier), and is skipped, with a message, only on a system with
neither. The installers of 2026.10.1 and earlier skip it on any system without
`sha256sum`.

macOS and Linux, x86_64 and arm64. Linux gets a glibc build, or a static musl
build automatically when glibc is older than 2.35, so it also runs on Alpine and
inside slim containers. Windows is not supported.

Set `OPENMAX_NO_MODIFY_PATH=1` to leave your `PATH` alone, or
`OPENMAX_INSTALL_DIR` to install somewhere else:

```sh
curl -fsSL https://useopenmax.dev/install.sh | OPENMAX_NO_MODIFY_PATH=1 sh
```

`useopenmax.dev/install.sh` redirects to the current release's installer, so it
follows new releases without the link changing.

Every [release](https://github.com/Max17190/open-max/releases) also carries the
plain tarballs and a `sha256.sum`, so you can download and verify by hand
instead of piping a script into a shell.

**From source** instead, which needs [Rust](https://rustup.rs) 1.89 or newer:

```sh
git clone https://github.com/Max17190/open-max.git
cd open-max
cargo install --path crates/tui --locked
```

Or run it in place with `cargo run --release -p openmax`.

## Configure

The first run can configure Open Max for you: pass the endpoint as flags, as
in [Use](#use), and each flag whose key `~/.openmax/settings.json` does not set
yet is written there before the turn. Or edit the file yourself:

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "model": "qwen2.5-coder:7b",
  "api_key": null,
  "approval_mode": "ask",
  "context_tokens": 32768,
  "max_parallel_tools": 4
}
```

`base_url` is the root of your model's HTTP API (the harness calls `chat/completions` on it). Set `model` to the id that server expects. Set `api_key` to a literal or `$ENV_VAR`, or export `OPENMAX_API_KEY`. Set `context_tokens` to the context window the server actually serves for that model; nothing is queried, and a guessed window is wrong in one direction or the other. There is no default endpoint, model, or window: until all three are configured (here or through a named provider), Open Max refuses to start a turn with an error that names each missing key and the flag that writes it.

`--base-url`, `--model`, `--api-key` and `--context-tokens` set the key of the same name. A flag whose key settings.json leaves unset (no file yet, an empty `base_url` or `model`, no `api_key` or `context_tokens`) is written to the file before the run, on a run that uses the endpoint the file selects, and the run says which keys it wrote. A key the file already sets is overridden for that run only: the file keeps its value, a `/model` or `/provider` save during the run keeps it too until the session changes that key or switches provider, and the run says which keys the command line overrode. A `--base-url` for another server than the file's is not sent the file's `api_key`. A flag equal to the file's value changes nothing. The full rule, with `--provider`, is in [configuration](docs/configuration.md#settings-from-the-command-line).

Trusting a project records its approval mode: `auto` unless you pick `ask` or
`readonly` at the trust prompt. Auto covers extension creation and repair
without further confirmation; validation, deny rules, and execution reports
remain active. `/approvals auto|ask|readonly`, Shift+Tab, and the approval
card's Auto for project choice change the saved mode for the current trusted
project. `approval_mode` in settings.json applies only to a project with no
saved mode, such as one trusted by an earlier version. See
[approval modes](docs/configuration.md#approvals).

A settings file that exists but does not parse, uses an unknown key, or sets an unrecognized `approval_mode` is a startup error (fail closed): Open Max exits with the parse reason instead of silently reverting your endpoint and approval policy to defaults.

Named endpoints in `providers.json` and `compat` flags for picky gateways are covered in [configuration](docs/configuration.md).

## Use

The first run is one command. In your project, give it your model server's
API root, model id and context window, and a first task:

```sh
cd ~/code/my-app
openmax --trust-project --base-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b --context-tokens 32768 -p "summarize this repo"   # from a terminal
```

It trusts the project in `auto` mode (a project trusted already keeps its
mode), writes the endpoint to `~/.openmax/settings.json`, and prints the
answer, with no file to write by hand and nothing to restart. For a server that checks a key, add
`--api-key '$NAME'` naming the environment variable that holds it, exported
before you run the command: the single quotes keep your shell from expanding
it, so settings.json stores the variable's name and the key is read from the
environment when a request is sent. A literal key works too, and is stored as
given, in plain text in settings.json.

After that, `openmax` alone starts the fullscreen session:

```sh
cd ~/code/my-app
openmax
```

On the first interactive run, inspect the project and answer the trust prompt: `y` trusts it in `auto` mode, `a` in `ask`, and `r` in `readonly`. The interface paints its first frame before it reads trust, settings or the session, then asks on your shell. Headless and stdio runs never prompt: grant trust once from a terminal with `--trust-project`, which records `auto`, and later runs, including a frontend's `openmax --stdio`, need no flag:

```sh
openmax --continue                                 # resume latest session here
openmax --trust-project -p "summarize this repo"   # from a terminal
openmax --stdio                                    # a frontend, once trusted
```

A frontend's stdin is its protocol pipe, not a terminal, so it cannot grant trust itself. For CI and other automation a human runs, see [project trust](docs/configuration.md#project-trust).

In print mode, text goes to stdout and tool progress to stderr; with `--json`, each `AgentEvent` is one JSON line. Press **/** for slash commands and **@** to mention a project file. Full flags, keybindings, and commands are in [usage](docs/usage.md).

One flag hands a task to a sub-agent, from your shell or from the agent's `bash`:

```sh
openmax --subagent "make the flaky retry test deterministic" --worktree
```

The child is one session in a new git worktree of the repository at HEAD, trusted because the project is, in the project's approval mode or a stricter one. Its events print as `-p` prints them. When it ends, or a signal stops it, its tools are stopped, its changes are committed to a branch of their own, the worktree is removed, and the last line names the branch with a diffstat, or says the child changed nothing. The flag adds nothing to any request. See [sub-agents](docs/usage.md#sub-agents).

## Extend

With nothing installed, extensions cost zero tokens. Project paths win over global ones on name collision. Each surface is a file the agent can write, `openmax --check` can validate, and the next turn picks up automatically.

| Surface | Location | What it is |
| --- | --- | --- |
| **Tools** | `.openmax/tools/*.toml` | Schema-described native processes; args on stdin, result on stdout |
| **Skills** | `.agents/skills/*/SKILL.md` | On-demand workflow packages; only name and description cost prompt tokens |
| **Prompt templates** | `.agents/prompts/*.md` | User-invoked messages; the file stem becomes a slash command |
| **Hooks** | `.openmax/hooks/` | Process gates on a fixed set of lifecycle events; three of them can block |
| **Permissions** | `.openmax/permissions.toml` | Allow, deny, or ask rules matched against tool arguments; fail closed |

```toml
# .openmax/tools/todo_scan.toml
name = "todo_scan"
description = "List TODO/FIXME comments with file and line"
command = "sh"
args = ["./scripts/todo-scan.sh"]
timeout_secs = 30
mutating = false

[params]
type = "object"
required = ["path"]
[params.properties.path]
type = "string"
description = "Directory to scan"
```

Unknown keys are rejected in tool, hook, and permission files and in `settings.json`, and `openmax --check` names any in `providers.json`, so a misspelled field there shows up in `openmax --check` rather than silently changing behavior (`SKILL.md` and template frontmatter ignore keys they do not read). `openmax --spec <surface>` prints the complete authoring contract for any one of them, and the frozen prompt carries only a one-line pointer to it, so that contract costs zero tokens until the agent reads it.

File formats, hook events, permission rule syntax (including the in-session repair path for a broken permissions file), self-description, and the freezing model are documented in [extending](docs/extending.md).

## Native execution and privacy

The built-in file tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`) are confined to the project root by the harness.

`write_file` and `edit_file` write the new contents to a temp file beside the target and rename it into place, so a write that fails (a full disk, a quota) leaves the file as it was.

- The temp file that replaces a file is created with mode `0600`, readable and writable only by its creator, until it has the target's owner, group and permissions.
- The temp file takes the target's permissions (less setuid and setgid), owner, group and access control list. It does not take the directory's default list, or the target's other extended attributes or file flags.
- The temp file needs room for the whole new file beside the old one, so on a full disk even a write that would shrink a file is refused, and the file is left unchanged.
- A replacement is a new file under the old name, so whatever holds the old file keeps it. A mount of the file itself (a container's single-file bind mount) and another process's open descriptor or inode watch go on seeing the old contents, so mount or watch the directory instead.

A file is written in place instead, and its old contents are written back if that write fails, when:

- it has other hard links;
- Open Max cannot give a new file its owner or group (only the superuser can give a file to another user or to a group it is not in);
- its access control list cannot be given to a new file (on macOS, where Open Max does not copy one, a file that has a list, or whose directory's inheritable entries would give the new file one);
- its directory takes no new file;
- the rename over it is refused (a file mounted on its own);
- or its path leaves no room for the longer path of a temp file beside it.

Such a file whose contents cannot be read first is not written at all, since there would be nothing to write back, and the write reports it unchanged. In a directory flagged append-only, which refuses both the rename and the removal of the temp file, the file is written in place and the temp file, holding the new contents, is left beside it.

`bash`, external TOML tools, and hooks are native processes. The harness does not confine them to the project root. They inherit the host filesystem, environment, credentials, and network access of Open Max, except that an external tool's environment is a scrubbed baseline (`PATH`, `HOME`, `LANG`, `TERM`) plus the variables its manifest's `env` names. They run without a controlling terminal, so a command that prompts on the terminal (a password, an ssh host key) fails at once instead of drawing over the TUI and waiting out its timeout. `bash` and hooks also get `GIT_TERMINAL_PROMPT=0`, so git names the cause.

`bash` runs each command with `bash -c` (`/bin/sh -c` on a host with no bash at a fixed path), but the command text is not that argument: it reaches the shell in the `OPENMAX_COMMAND` environment variable, and the shell runs the fixed script `eval "unset OPENMAX_COMMAND; $OPENMAX_COMMAND"`. So the shell's argument list never holds the text, and since `eval` unsets the variable before the command's first word, no process the command starts inherits it, whether or not bash starts with `allexport` on (as an exported `SHELLOPTS=allexport` or a `BASH_ENV` file that runs `set -a` arranges). A `BASH_ENV` file runs before that `unset`, so it, and anything it starts, can still read the text there. `OPENMAX_COMMAND` is a name the harness reserves for `bash` calls: a value exported in the harness's own environment is overwritten for the call and unset before the command runs. A `pkill -f` or `pgrep -f` pattern that occurs only in the command text therefore matches the processes it names and not the shell running it. `$0` is the shell's path and there are no positional parameters, as under a plain `bash -c`.

Running the text through `eval` shows in a few places. Bash's `set -x` trace lines start with `++` instead of `+`. Bash's `$_` is `OPENMAX_COMMAND` when the command starts, and once the command text has run, as an `EXIT` trap sees it, `$_` holds `unset OPENMAX_COMMAND; ` and the text; bash gives each child its own `_`, so none inherits it. Bash's `BASH_EXECUTION_STRING` holds the fixed script, not the text. A syntax error names `eval` where it named `-c` (bash's `-c: line N:` becomes `eval: line N:`, and dash adds `eval: ` after its line number), and a first line that bash quotes back in the error starts with `unset OPENMAX_COMMAND; `. Bash 3.2 (the macOS `/bin/bash`) exits 1 for a syntax error where `bash -c` exited 2; bash 5.3 and dash exit 2 either way. A command killed by a signal can be reported as the shell's exit code, 128 plus the signal's number (139 for a segfault), instead of as the signal: under `-c`, bash and dash exec'd a lone simple command in the shell's place, and dash and bash 5.1 and later also the last command of a list, but none of them does inside `eval`. Bash's or dash's own stderr line names most signals (not SIGINT or SIGPIPE).

Each `bash` or external tool call runs in its own process group, and processes it leaves running in the background are terminated when it returns. The first call in a session that terminated some gets a result that explains why and how to keep one running (a named tmux session); every later one gets one line that says so on its own, which still holds if the model never saw the first (its turn was cancelled) or compaction dropped it. A session loaded again from its file, as a resume does, or one whose extensions were refrozen, explains once more.

An openmax killed outright (`kill -9`, or a crash that runs no handler) cannot stop its tools. On macOS and Linux it records the process group of each tool it runs in `~/.openmax/run`, and the next launch of the TUI, `-p`, `--stdio`, or `--check` stops the groups a killed openmax left running. `zero_orphans_after_kill_9` in `crates/tui/tests/thesis.rs` gates this with `--check` as that launch: a print run gets SIGKILL in the middle of a `bash` call whose shell, child and grandchild all ignore SIGTERM, and once `openmax --check` has run, none of the three is running. The TUI, `-p` and `--stdio` stop the groups before they start a session, so before any tool of their own: a launch refused for a settings flag, which starts no session, stops them too. The limits are in [usage](docs/usage.md#command-line).

Permissions, approvals, and `mutating` metadata control dispatch and user experience, not operating-system isolation.

Open Max itself does not phone home. Apart from native child processes, the harness contacts only the model endpoint you configure.

Sessions, settings, tools, and skills stay under `~/.openmax/` and your project directory. `/status` lists the destinations configured by the harness and detailed runtime information; it does not enforce or enumerate child-process network access, and external tools you install may open their own network connections.

## Documentation

- [Configuration](docs/configuration.md): settings, approvals, providers, project trust
- [Usage](docs/usage.md): CLI flags, keybindings, slash commands
- [Extending](docs/extending.md): tools, skills, templates, hooks, permissions, validation, freezing
- [stdio protocol](docs/stdio-protocol.md): the `openmax-stdio/8` contract for custom frontends
- [Testing](docs/testing.md): the test count and mutation score CI gates, and how to read a mutation run

## Development

```sh
cargo check
cargo test
scripts/check-suite-time.sh
scripts/count-core-lines.sh
scripts/check-test-count.sh
cargo build --release -p openmax
```

`cargo test` runs the test binaries one after another. `scripts/check-suite-time.sh`, run once they are built, runs them all at once and fails when the suite takes longer than 5 seconds, or when a test fails (a test binary that fails beside the others is run again alone and fails the check only if it fails again). A test binary still running at three times the budget is stopped, with every process still under it, and fails the check, which names the tests it had not finished. CI's test job gates one run of it on each runner, after `cargo test`, with three times that budget on ubuntu-latest and four times it on macos-latest, whose runners have fewer cores. One run says little about the next, so `scripts/check-suite-time.sh --runs 10` runs the suite ten times after one build, prints each run's time, slowest binary and 1-minute load average, then the median and the slowest run, and fails when any run fails. The script, and the test in `crates/tui/tests/thesis.rs` that drives it as part of `cargo test`, need bash and perl with its core modules JSON::PP and Time::HiRes (on Debian and Ubuntu, the `perl` package: `perl-base` alone has neither).

The thesis tests (in every `crates/*/tests/thesis*.rs` and the binary's `thesis_frames` module) print what they measure. One that times the harness asserts its bound only in a release build, and a 20x sanity bound in debug; a byte or count one asserts the same in every build. CI's thesis job builds them in release, the profile the published binary ships with, and runs them on Linux and macOS in a step that fails the pull request, except the ones a runner reports instead, listed in its matrix entry in `.github/workflows/ci.yml`: a step allowed to fail runs those before the gate step, prints their numbers and copies each `MEASURE` line to the job's summary. Today those are `harness_time_per_tool_round` (target 250 us) on both runners and `transcript_frame_p99_at_100k_lines` (target 0.5 ms) on macOS, targets those runners do not meet with room. The first frame's 1 ms target is not asserted in CI: timed from the spawn, a frame includes the runner's own process start, which is not the harness's work. CI holds instead the ratio of the first frame's median to the median start of a baseline binary (`crates/tui/examples/startup_baseline.rs`, which writes the end of a frame and exits), timed the same way on the same runner, under a ratchet of 2.0 on macOS and 2.9 on Linux, and prints both times and their difference, the harness's own work, without asserting them.

The frozen prefix meets its 700-token target: `frozen_prefix_is_under_700_tokens` (`crates/core/tests/thesis.rs`) prices what every request resends first, messages[0] without the project path plus the `tools` array and the note that follows, against a 3,036-byte cap, which is 700 tokens less a 10-token allowance for one path at 4.4 bytes per token, and measures 3,022 bytes, 678 tokens, today. Peak memory does not meet its target yet: `headless_turn_peak_rss_is_under_four_megabytes` reports a two-round turn's peak RSS, from `wait4`, against a 4,000,000-byte target, which a release build misses at 6,544 KiB by median today. What that test enforces is a ratchet, on macOS Apple silicon in a release build only, over a second series of the same runs started with `MallocMaxMagazines=1`: 4,016 KiB beyond the binary's own floor. Anywhere else it asserts a sanity bound of twenty times the target, and on Linux every figure starts from the test process's own high-water mark, so it bounds the turn's from above.

`scripts/count-core-lines.sh` (bash and perl) prints two line counts for each crate and in total: core-excluding-tests, the lines of `crates/*/src` less its test-only code (`#[test]` functions, `#[<path>::test]` functions such as `#[tokio::test]` ones, items under a `cfg` that needs `test`, and the module files those items load, found by matching brackets outside strings, chars and comments, by the method in the script's header; the input of `stringify!` and the rules of a `macro_rules!` are text the build keeps, so no attribute in them is read), and total-including-tests, every line of every `.rs` file under `crates/` (build output in a crate's `target/` directory aside), so a test's lines count toward total-including-tests wherever they sit and toward core-excluding-tests nowhere. The script stops with an error naming the file and line of a test-only item whose end it cannot find, rather than guess. The test `core_source_stays_under_its_line_budget` in `crates/tui/tests/thesis.rs` holds the script to a fixture whose every source line is marked counted or excluded, and fails when the script stops, when core-excluding-tests is over the ratchet the test sets, or when it is under the count that ratchet was set from, so a change that removes production lines lowers the ratchet with them. The ratchet is 41,633 today, 200 over the 41,433 lines measured, against a target of 37,000.

`scripts/check-test-count.sh` counts the tests `cargo test --workspace -- --list` lists, less the ignored ones, and fails when there are fewer than 1,427 on Linux, or 1,425 on macOS: each compiles a few tests the other does not, so each has a floor of its own. CI runs it in the test job on both; on any other platform it reports the count and passes. The target is 2,000 tests.

The Mutants workflow runs cargo-mutants (configured in `.cargo/mutants.toml`) on Linux runners, and `scripts/check-mutants.sh` judges its outcomes. On a pull request it tests the mutants in the code the change touches, and fails when one was missed or timed out and is not listed with its reason in `.cargo/mutants-allowed.txt`. On a push to main, weekly, and when run by hand, it tests every mutant in the workspace, and fails when the mutation score, caught / (caught + missed), is under 76.60%: the score of the full run that set it, 76.80%, less a margin of 0.20 points for the few mutants that come out differently from one run of the same code to the next. Either run leaves out the mutants in code Linux does not compile, such as macOS-only code. A full run takes hours, so a push made while one is going waits, and only the newest push waiting is tested. The target is 95%. [Testing](docs/testing.md) says what each outcome counts for, how the floor moves, and how to read a run.

Set `OPENMAX_PERF=1` while running the TUI to log frame, transcript-layout, and selection-overlay timings. Core logic is in `open-max-core` (`crates/core/src/agent.rs`); the TUI is `crates/tui/src/app.rs`.

## Status

Open Max is early software. File an issue or send a PR if something breaks.

## License

MIT. See [LICENSE](LICENSE).
