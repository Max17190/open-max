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

At a terminal, the first frame (the project path, an empty conversation and
the prompt) is painted before Open Max reads trust, settings or the session,
and the status line fills in the model and approval mode once they are read.
The trust question, and a refusal to start (a trust or settings file that does
not parse, a trust grant asked for from an agent session), then show on your
shell, which the interface hands back for them.

The interface uses your terminal's own foreground and background colors. The
palette's gray (color 8) marks secondary text, selection, and every border
except the focused prompt's and the approval card's. There is no color
setting.

## Command line

```sh
openmax --base-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b --context-tokens 32768
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

`--base-url`, `--model`, `--api-key` (a literal or `'$ENV_VAR'`) and
`--context-tokens` set the settings key of the same name. Each one is written
to `~/.openmax/settings.json` when the file leaves its key unset and the run
uses the endpoint the file selects, so the first run configures the install,
and the run prints the keys it wrote. A key the
file already sets is overridden for that run only, an in-session `/model` or
`/provider` save keeps the file's value for it until the session changes that
key or switches provider, and the run prints the keys it overrode; a flag
equal to the file's value does nothing. A `--base-url` for another server than
the file's is not sent the file's `api_key`. With
`--trust-project` and `-p`, the first run is one command that grants trust,
writes the endpoint and answers a first task. The full rule, `--provider`
included, is in [configuration](configuration.md#settings-from-the-command-line).

`openmax --mcp-list` and `openmax --mcp-call` are a one-shot MCP stdio
client, used by the proxy tool an MCP server is adopted through; see
[extending](extending.md#mcp-servers) and `openmax --spec mcp`.

`openmax --check` prints the rows of each file that needs attention (`warn` or
`err`) and of each hook, then one line counting, per surface, the files that
loaded with nothing to report (and how many memories are not in the index).
The report names files under the current directory relative to it, in each
row's path and in the paths and repair commands it writes into a message
(`openmax --approve '.openmax/tools/x.toml'`); text a message quotes, such as
a manifest line in a parse error, prints as written.
`openmax --check --all` prints every row.

`openmax --check --json` prints every finding as one JSON array of
`{surface, path, status, message}` objects (status `ok`, `warn`, or `err`),
with absolute paths and the same exit code, so the agent can parse its own
verification.

`openmax --check --run-examples` adds one `example` surface row per declared
`[example]`, in text and in JSON, and fails the check when one fails. It is
the only `--check` mode that runs a command, so it needs a trusted project
and follows the saved project mode. Like the TUI, `-p` and `--stdio`, plain
`--check` first stops the tools a killed openmax left running and removes that
openmax's record of them (see the signals below). Auto runs valid examples
without content approval; ask uses the documented approval and sandbox rules.
It honors permission rules, `pre_tool_use` hooks, and `approval_mode` exactly
as a turn does. See [extending](extending.md#proof-of-life).

A print prompt that starts with `-` parses as options, not text. Put it
last, after `--`, which ends the options: `openmax -p -- '- list the files'`.
An unknown option after `-p` is refused with that advice. Every word after
`--` is prompt text, a later `-p` included, so in a multi-turn run only the
last turn can start with `-`:
`openmax -p "list crates" -p -- "- summarize the first one"`.

In print mode, text goes to stdout, and tool progress and errors to stderr. A
note the harness writes for the model (a `harness_note` event, such as the
execution policy notice) stays in the transcript and is written to neither
stream, so a parent agent reading a child run's stderr does not receive it.
With `--json`, each `AgentEvent` is one JSON line on stdout, `harness_note`
included. Mutating tools honor the project's approval mode, and a print run
declines every approval request, so unattended runs need `auto`: the mode a
trust grant records unless you pick another. For a project in `ask`, select
`/approvals auto` once.

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
runs past its own timeout is stopped and the turn goes on; a `bash` call is
stopped after 120 seconds unless it sets `timeout_secs` (1 to 600); a larger
request is held to 600, and if that command times out, its result names the
ceiling and the number requested. A caller that
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
| 4 | A turn stopped short (`max_iterations`, `budget_exhausted`, or `unverified`); resubmit to continue, though an `unverified` from a gate that could not start repeats until the hook is repaired |

Tools run in a session of their own, so Ctrl+C or Ctrl+\ at the shell reaches
openmax and not them. SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGUSR1, or SIGUSR2 to a
print or stdio run cancels the running turn, which stops each tool's process
group (SIGTERM, then SIGKILL), and exits 128 plus the signal number (130, 143,
129, 131; for SIGUSR1 and SIGUSR2, 158 and 159 on macOS and 138 and 140 on
Linux) once the turn has ended, waiting at most 3 seconds for it; a `turn_end`
hook still running then is killed. A SIGQUIT therefore writes no core dump.
Removing a new session that nothing was saved to takes its share of the same 3
seconds: while another openmax process holds the session index past them, the
session stays indexed and the run says so on stderr. Once they are used up, the
run still takes up to 100 ms to check that the session is empty before it says
so; a check that takes longer is reported as a session that may be empty. A
signal while the run waits on that index to open its session or to start a turn
ends it the same way. A second signal exits without waiting for the turn, with
128 plus its own number, and kills the tools outright. Either exit first waits up
to 250 ms for a tool command still being started, so one that starts within that
wait is killed with the rest. Neither stops a process that leaves its tool's
group (one that starts a session of its own, as `setsid` does), or a tool command
whose start outlasts that wait: it runs on after openmax has exited. SIGKILL,
or a crash that runs no handler, stops no tool: each running tool's group runs
on until the next openmax launch (the TUI, `-p`, `--stdio`, or `--check`).
On macOS and Linux, openmax records the process groups of its running tools,
each leader with its start time, in `~/.openmax/run/<pid>.groups`: one line
when a group starts and one when it ends. Once the file has passed 4 KiB, it
is cut back to its first line the next time a group ends with no other
running. An exit that leaves no tool running removes it, and the `run`
directory once no other record is left there. A launch reads every record
whose owner pid no process of the same user holds (none does, or another
user's does), or names a process that has exited or started at another time,
and sends each group the record still lists whose leader still has its
recorded start time SIGTERM, then SIGKILL if the group is still there 250 ms
later, before it starts anything. Neither signal goes to a group whose leader
pid has since been taken by a process started at another time, and none goes
to a group once it has been seen gone, since its id can then name another. It
then removes those records, and the `run` directory once no record is left
there. On Linux, where start times count from boot, a record's first line also
carries the boot's id, and a record from an earlier boot is removed with
nothing signalled. A record a running openmax owns is left alone. A killed
openmax holds its pid until its parent waits for it, and a launch stops its
tools before that wait too. A launch never signals the group it runs in: an
openmax that a left-behind tool's script starts leaves that tool's group
running and, while the group's leader runs, its record in place, so the next
launch outside that group stops it. A group is recorded just after its spawn
has returned, when its command has already started, so a SIGKILL in that
moment leaves it unrecorded, and a home where the record cannot be written
(read-only, or on a full disk) records nothing while it stays so. No tool and
no exit waits on the record's file, as a home whose file server stopped
answering would make them: one thread, started by the first tool or hook a
launch runs, makes every write to it, and an exit waits at most 100 ms for
that thread to remove it. The next launch removes a record so left, finding
the groups it lists gone. A sub-agent openmax among the tools that the SIGKILL
ends before it has stopped its own tools leaves them to the launch after. A
group whose leader had exited and been waited for before the launch, and a
process that left its tool's group, are not stopped.
Nor are the tools of an openmax in another pid namespace that shares the home
directory, as a container with the home mounted does: a launch outside it
takes its records for a dead openmax's and removes them. While a
consumer that stopped reading holds stdout full, the run cannot act on the first
signal, and only a second one ends it. A signal the run inherited as ignored, as
under `nohup`, stays ignored. The TUI, where Ctrl+C is a key, ends the session on
one of these signals as `/quit` does, then exits with the same status, unless its
terminal has already gone away. Until the session's loop starts (while its
first frame waits on trust, settings, the prompt history or the session
`--continue` resumes, or the trust question waits on the shell), the first
signal ends the run at once with that status, and Ctrl+C at the terminal is
SIGINT.

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
| **Enter** | Send; while a turn runs, steer it: the message joins the turn as a user message before its next request, which carries it, or the next turn's first request when the turn ends first (during a compaction it queues for after); one a `user_prompt_submit` hook blocks comes back to the composer |
| **/** | Slash commands · **Tab** or **Enter** completes |
| **@** | Mention a project file |
| Mouse drag | Select transcript or prompt text |
| Double / triple click | Select the word under the pointer · the whole logical line |
| **y** or **Ctrl+C** | Copy selected text (**Ctrl+C** cancels when nothing is selected) |
| Click in the prompt | Put the cursor there in a wrapped draft |
| Wheel | Scroll the conversation · over the prompt, a long draft |
| **Shift+Tab** | Switch and save project approvals between `ask` and `auto`; from `readonly`, to `ask` (choose `readonly` at the trust prompt or with `/approvals readonly`) |
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
| `/fork` · `/rewind <n>` | Branch this session into a new one at its end · at the end of turn `n`, and switch to the branch. The original is not changed; `/resume` lists the branch as a fork of it. Files are not changed either: edits made after the cut stay on disk |
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

Turn `n` of `/rewind` is the `n`th prompt you sent that ran and stayed in
the transcript, cut where it finished: a completion gate's refusal and
other notes the harness writes during a turn stay in that turn. The full
counting rule, including compaction and sessions saved by an earlier
version, is in [the stdio protocol](stdio-protocol.md), whose `rewind`
counts the same way.
