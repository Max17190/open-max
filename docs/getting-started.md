# Getting started

The full first-run guide: configuring an endpoint, approval modes, signing in, headless and unattended runs, and sub-agents.

## Configure

The endpoint must return structured API tool calls to use tools. For local servers, configure the model's tool parser and chat template. Call markup in assistant text is displayed as text and never executed. Responses are limited to 16 MiB and 128 tool calls.

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

`--base-url`, `--model`, `--api-key` and `--context-tokens` set the key of the same name. A flag whose key settings.json leaves unset (no file yet, an empty `base_url` or `model`, no `api_key` or `context_tokens`) is written to the file before the run, on a run that uses the endpoint the file selects, and the run says which keys it wrote. A key the file already sets is overridden for that run only: the file keeps its value, a `/model` or `/provider` save during the run keeps it too until the session changes that key or switches provider, and the run says which keys the command line overrode. A `--base-url` for another server than the file's is not sent the file's `api_key`. A flag equal to the file's value changes nothing. The full rule, with `--provider`, is in [configuration](configuration.md#settings-from-the-command-line).

Trusting a project records its approval mode: `auto` unless you pick `ask` or
`readonly` at the trust prompt. Auto covers extension creation and repair
without further confirmation; validation, deny rules, and execution reports
remain active. `/permissions auto|ask|readonly` and the approval
card's Auto for project choice change the saved mode for the current trusted
project. `approval_mode` in settings.json applies only to a project with no
saved mode, such as one trusted by an earlier version. See
[approval modes](configuration.md#approvals).

Shift+Tab cycles requested model reasoning effort. `/effort` opens a picker,
and `/effort high` or `--effort high` selects a level directly. Supported
levels depend on the provider and model; Default leaves its request unset. A
level the route is known to refuse, such as `max` on xAI's grok-4.x, is
refused before any request is spent, naming the levels it takes.

A settings file that exists but does not parse, uses an unknown key, or sets an unrecognized `approval_mode` is a startup error (fail closed): Open Max exits with the parse reason instead of silently reverting your endpoint and approval policy to defaults.

Named endpoints in `providers.json` and `compat` flags for picky gateways are covered in [configuration](configuration.md).

## Use

Run `openmax --login` to connect a provider. The picker offers ChatGPT browser
sign-in, OpenAI and Anthropic API keys, and other compatible APIs. API keys
are entered in a hidden prompt. Inside a session, use `/login`, `/logout`,
and `/provider` to connect, disconnect, and select a provider/model pair.
See [provider connections](configuration.md#connecting-and-disconnecting-providers).

To use an eligible ChatGPT plan, run `openmax --chatgpt login` from your
terminal. It opens browser consent and adds an account-specific model catalog.
Choose a returned model with `/model` or `--provider chatgpt-main --model
<model-id>`. Supply `--context-tokens` from the model's documentation when the
catalog does not report it. Existing provider settings stay selected until
you choose ChatGPT. See [ChatGPT sign-in](configuration.md#chatgpt-sign-in)
for account management and usage limits.

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

A frontend's stdin is its protocol pipe, not a terminal, so it cannot grant trust itself. An unattended run, in CI or a container with no terminal, takes one human act, at setup: `OPENMAX_HUMAN_ATTEST=1` on the command that grants trust, which records `auto` for a project not trusted before (one trusted before keeps its mode), as in `OPENMAX_HUMAN_ATTEST=1 openmax --trust-project -p "<task>"`; in `auto` the agent can then write a tool and call it on its next step with no further input (see [project trust](configuration.md#project-trust)). The harness strips that variable from every tool's environment, even a tool whose manifest names it, and `an_unattended_run_writes_a_tool_and_calls_it_on_the_next_step` in crates/tui/tests/thesis.rs runs that whole path with no terminal.

In print mode, text goes to stdout and tool progress to stderr; with `--json`, each `AgentEvent` is one JSON line. Press **/** for slash commands and **@** to mention a project file. Full flags, keybindings, and commands are in [usage](usage.md).

One flag hands a task to a sub-agent, from your shell or from the agent's `bash`:

```sh
openmax --subagent "make the flaky retry test deterministic" --worktree
```

The child is one session in a new git worktree of the repository at HEAD, trusted because the project is, in the project's approval mode or a stricter one. Its events print as `-p` prints them. When it ends, or a signal stops it, its tools are stopped, its changes are committed to a branch of their own, the worktree is removed, and the last line names the branch with a diffstat, or says the child changed nothing. The flag adds nothing to any request. See [sub-agents](usage.md#sub-agents).
