# Usage

```sh
cd ~/code/my-app
openmax
```

On the first interactive run, inspect the project and accept the trust prompt.
See [configuration](configuration.md#project-trust) for headless and stdio
trust.

## Command line

```sh
openmax --continue                    # resume latest session here
openmax -c
openmax --provider ollama --model qwen2.5-coder:7b
openmax -p "summarize the top level layout of this repo"
openmax -p --json "list public modules in crates/core"
openmax --check                       # validate extension files
openmax --spec hooks                  # print an extension surface's contract
openmax --recall "deploy port"        # search past sessions and memories
openmax --stdio                       # full session over JSONL pipes
```

`openmax --check --json` prints the same findings as one JSON array of
`{surface, path, status, message}` objects (status `ok`, `warn`, or `err`),
with the same exit code, so the agent can parse its own verification.

`openmax --check --run-examples` adds one `example` surface row per declared
`[example]`, in text and in JSON, and fails the check when one fails. It is
the only `--check` mode that executes anything, so it needs a trusted project
and a tool file approved with `openmax --approve <path>`, and it honors
permission rules, `pre_tool_use` hooks, and `approval_mode` exactly as a turn
does. See [extending](extending.md#proof-of-life).

In print mode, text goes to stdout and tool progress to stderr. With `--json`,
each `AgentEvent` is one JSON line on stdout. Mutating tools still honor
`approval_mode`; for unattended runs set `"approval_mode": "auto"`.

`openmax --stdio` is the contract for custom frontends, editor integrations,
and one openmax driving another (see the `delegate` skill). It is specified in
[stdio protocol](stdio-protocol.md).

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
| **Shift+Tab** | Cycle approvals for this run: `ask` → `auto` → `readonly` (`/approvals` persists) |
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
| `/approvals auto\|ask\|readonly` | Mutating tool gates |
| `/new` · `/resume` | Fresh session · pick an earlier one |
| `/reload` | Force a re-freeze now (it also happens automatically when extension files change) |
| `/tools` · `/skills` · `/context` | Session tools, skills, token budget |
| `/<template> [args]` | Run a prompt template from `.agents/prompts/` |
| `/status` | Endpoint, cache, performance, privacy, and network details |
| `/quit` | Exit |

The persistent status line stays limited to model, context use, and approval
mode so the transcript remains readable; `/status` is where the full runtime
detail lives.
