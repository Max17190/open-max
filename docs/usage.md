# Usage

```sh
cd ~/code/my-app
openmax
```

On the first interactive run, inspect the project and answer the trust
prompt: `y` trusts it in `auto` mode, `a` in `ask`, and `r` in `readonly`.
Change the mode later with `/approvals` or **Shift+Tab**. See
[configuration](configuration.md#project-trust) for headless and stdio
trust.

## Command line

```sh
openmax --continue                    # resume latest session here
openmax -c
openmax --provider ollama --model qwen2.5-coder:7b
openmax -p "summarize the top level layout of this repo"
openmax -p --json "list public modules in crates/core"
openmax --check                       # validate extension files and the session index
openmax --spec hooks                  # print an extension surface's contract
openmax --recall "deploy port"        # search past sessions and memories
openmax --stdio                       # full session over JSONL pipes
openmax --mcp-list -- <server>        # list an MCP server's tools
```

`openmax --mcp-list` and `openmax --mcp-call` are a one-shot MCP stdio
client, used by the proxy tool an MCP server is adopted through; see
[extending](extending.md#mcp-servers) and `openmax --spec mcp`.

`openmax --check --json` prints the same findings as one JSON array of
`{surface, path, status, message}` objects (status `ok`, `warn`, or `err`),
with the same exit code, so the agent can parse its own verification.

`openmax --check --run-examples` adds one `example` surface row per declared
`[example]`, in text and in JSON, and fails the check when one fails. It is
the only `--check` mode that executes anything, so it needs a trusted project
and follows the saved project mode. Auto runs valid examples without content
approval; ask uses the documented approval and sandbox rules. It honors
permission rules, `pre_tool_use` hooks, and `approval_mode` exactly as a turn
does. See [extending](extending.md#proof-of-life).

In print mode, text goes to stdout and tool progress to stderr. With `--json`,
each `AgentEvent` is one JSON line on stdout. Mutating tools honor the
project's approval mode, and a print run declines every approval request, so
unattended runs need `auto`: the mode a trust grant records unless you pick
another. For a project in `ask`, select `/approvals auto` once.

A print run has no overall deadline, like a TUI or stdio session. A turn keeps
going while the endpoint sends anything, keepalives and tool call arguments
included; silence for the provider's idle timeout ends the attempt (see
[configuration](configuration.md#multiple-providers)). A request that got no
response, or a stream that went silent before any reply text, is resent until
the client's attempts are spent, so a server that takes requests and never
answers holds the turn for about 80 minutes at the default interval, sooner
with a lower `idle_timeout_secs`. Any other silence fails the turn after one
interval: a stream that stops partway through a reply ends it as truncated,
and a server that stalls while sending a reply as one JSON body is not
resent. An address with nothing listening fails within seconds. A tool that
runs past its own timeout is stopped and the turn goes on. A caller that
needs a deadline sets one, for example the `timeout_secs` of the bash call
that runs a child `openmax -p`. A run stopped by a signal ends with that
signal's status (a shell reports 128 plus its number); otherwise the exit
code says how the run ended, and a turn that ends with any code but 0 skips
the prompts after it.

| Code | Meaning |
| --- | --- |
| 0 | Every turn finished |
| 1 | An operational failure: a turn failed or could not start, or the session could not be opened |
| 2 | A usage or configuration error before any turn, or `--continue` found no prior session here |
| 3 | The project is not trusted, or trust cannot be granted from this process |
| 4 | A turn stopped short (`max_iterations`, `budget_exhausted`, or `unverified`); resubmit to continue |

Tools run in a session of their own, so Ctrl+C at the shell reaches openmax
and not them. SIGINT, SIGTERM, or SIGHUP to a print or stdio run cancels the
running turn, which stops each tool's process group (SIGTERM, then SIGKILL),
and exits 128 plus the signal number (130, 143, 129) once the turn has ended,
waiting at most 3 seconds; a `turn_end` hook still running then is killed. A
second signal exits at once and kills the tools outright. A signal the run
inherited as ignored, as under `nohup`, stays ignored. The TUI, where Ctrl+C
is a key, ends the session on one of these signals as `/quit` does, then
exits with the same status.

`openmax --stdio` is the contract for custom frontends, editor integrations,
and one openmax driving another. It is specified in
[stdio protocol](stdio-protocol.md).

A writable session can be open in only one process at a time, including while
it is idle. Close it in the other process or start a new session to continue
working. `/new` and switching sessions release the old attachment after any
in-flight work settles. A process exit releases its session locks. Read-only
history and recall remain available.

A damaged or unreadable transcript stops continuation with its path and the
failure reason. Open Max preserves the original bytes and does not silently
skip records or replace the transcript. Repair or recover a copy explicitly
before resuming. A crash or power loss mid-save is not damage. A final
record it cut off never became a message, so the session resumes from the
last complete record and the next save removes the fragment. A session whose
first save it interrupted has no transcript yet, and like any session that
has never saved messages, starts fresh.

A damaged or unreadable session index (`~/.openmax/sessions/index.json`)
refuses new sessions and continuation with its path, and the app keeps
running. Open Max never replaces it with an empty index. Run `openmax --check`
for the repair: close every openmax, then move the file aside.

## Keys

| Input | Action |
| --- | --- |
| **Enter** | Send (queues if the agent is busy) |
| **/** | Slash commands · **Tab** or **Enter** completes |
| **@** | Mention a project file |
| Mouse drag | Select transcript or prompt text |
| Double / triple click | Select the word under the pointer · the whole logical line |
| **y** or **Ctrl+C** | Copy selected text (**Ctrl+C** cancels when nothing is selected) |
| Click in the prompt | Put the cursor there in a wrapped draft |
| Wheel | Scroll the conversation · over the prompt, a long draft |
| **Shift+Tab** | Cycle and save project approvals: `ask` → `auto` → `readonly` |
| **Esc** | Clear selection · close menu · cancel turn · return to composer |
| **Ctrl+C** twice | Quit |

## Slash commands

| Slash command | Action |
| --- | --- |
| `/help` | Keybindings and commands |
| `/model` | Search configured providers and select a model |
| `/model <id>` | Set an exact model id on the active endpoint |
| `/copy` | Copy the latest assistant response |
| `/provider [name]` | List or switch providers |
| `/approvals auto\|ask\|readonly` | Save this project's execution mode |
| `/new` · `/resume` | Fresh session · pick an earlier one |
| `/reload` | Force a re-freeze now (it also happens automatically when extension files change) |
| `/tools` · `/skills` · `/context` | Session tools, skills, token budget |
| `/compact` | Compact the context now instead of waiting for the budget to force it |
| `/export [path]` | Write the transcript as markdown (default `openmax-<session>.md` in the project) |
| `/<template> [args]` | Run a prompt template from `.agents/prompts/` |
| `/status` | Endpoint, cache, performance, privacy, and network details |
| `/quit` | Exit |

The persistent status line stays limited to model, context use, and approval
mode so the transcript remains readable; `/status` is where the full runtime
detail lives.
