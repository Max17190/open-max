# Unattended runs

Use print mode (`-p`) for a task in a script or CI job. Install the binary
before the job starts, set up the endpoint and project, and read the exit
status when the run ends.

## Install a pinned release

Pick an exact version from [the releases](https://github.com/Max17190/open-max/releases).
Download the archive for your OS and architecture and `sha256.sum` from that
same release. In the download directory, check the archive before extracting
it. For example, for the static Linux x86-64 build:

```sh
archive=openmax-x86_64-unknown-linux-musl.tar.xz
awk -v archive="$archive" '$2 == archive || $2 == "*" archive { print; exit }' sha256.sum > archive.sha256
test -s archive.sha256 && shasum -a 256 -c archive.sha256 && tar -xf "$archive"
```

Use `sha256sum -c archive.sha256` if you have `sha256sum` instead. Stop if the
checksum is missing or does not match. Add the extracted directory
(`openmax-x86_64-unknown-linux-musl` in this example) to the job's `PATH`,
then check `openmax --version` against the pinned version.

A package launcher that fetches the binary on first use makes that first
command depend on a download. Use the verified archive to keep installation
and its network failures out of the task itself. Print runs do not check for
updates. See [Install](install.md) for the available builds.

## Set up state and the model

Set `HOME` to an absolute, writable directory outside the project. Open Max
keeps settings, trust, sessions and command logs in `$HOME/.openmax`. Reuse
that home to resume sessions or keep a saved trust grant. Give jobs in
separate process namespaces separate homes so they do not share tool process
records.

Set `base_url` to the root of the model API and `model` to the server's exact
model ID. Use `--api-key '$MODEL_API_KEY'` with the flat endpoint flags to read
a key from the environment. The single quotes keep the shell from expanding
it, so a settings save stores the reference, not the key. Export the key
before running the command. A server that needs no credentials can run over
plain HTTP. Open Max refuses to send a key over plain HTTP to another
machine. Use HTTPS, or point Open Max at a relay on loopback, such as
`http://127.0.0.1:8080/v1`, and secure the relay's upstream connection.

For a named endpoint, put its model in `$HOME/.openmax/providers.json` and
select it with `--provider ci --model my-model`. For example, for an endpoint
that speaks Chat Completions:

```json
{
  "providers": {
    "ci": {
      "base_url": "https://llm.example.com/v1",
      "api_key_env": "MODEL_API_KEY",
      "models": [{ "id": "my-model", "context_tokens": 131072, "max_tokens": 32768 }]
    }
  }
}
```

Replace the example numbers with the window the server actually serves
(`context_tokens`) and the model's longest reply (`max_tokens`). Open Max
has no default context window and does not look it up during a turn. To set
the window for a flat endpoint, pass `--context-tokens`.

The longest reply on a model entry sets the ceiling for requests that need a
longer reply. Where the connection sends a reply limit, the first ask is up
to 4096 tokens by default, including on Responses and native Messages.
For a different first limit, set `max_tokens` in `settings.json` to a value
other than 4096. That sets both the first limit and the ceiling,
within the model's maximum and the context window. Leave room for reasoning
as well as the answer. The
[context window and compaction reference](configuration.md#context-window-and-compaction)
explains how much room a request can use and how refused limits are retried.

## Trust the project and run a task

Inspect the project before granting trust. To grant it and save `auto` for a
job you set up without a terminal, set `OPENMAX_HUMAN_ATTEST=1` on the setup
command. `--trust-project` alone preserves an existing mode.
`--permissions auto` explicitly selects and saves `auto`. Open Max refuses
both flags when `OPENMAX_SESSION` marks an agent-spawned process and strips
the attestation from the tools it starts.

If your pinned binary does not list `--permissions` in `openmax --help`,
omit `--permissions auto` on a new project: `--trust-project` already
records `auto`. For a project trusted in another mode, select `auto` with
`/permissions auto` in a terminal session first.

This example grants trust, selects `auto`, sets up a flat endpoint and runs
one prompt. Change the URL, model ID and window to match your server:

```sh
cd ~/code/my-app
OPENMAX_HUMAN_ATTEST=1 openmax --trust-project --permissions auto --base-url https://llm.example.com/v1 --model my-model --api-key '$MODEL_API_KEY' --context-tokens 131072 -p --json "summarize this repo"
```

With the named endpoint above, use `--provider ci --model my-model` in place
of the flat endpoint flags. The named provider supplies the key and window.
That command does not save the named selection. Repeat
`--provider ci --model my-model` on each `openmax -p` command below, or save
`"provider": "ci"` and `"model": "my-model"` in `$HOME/.openmax/settings.json`
first. For the rules on which flags are saved and which override only the run, see
[Settings from the command line](configuration.md#settings-from-the-command-line).

A print run cannot ask for approval, so it declines every approval request.
Use `auto` for work that needs to run without a person approving calls.
`auto` does not sandbox commands or turn off deny rules. See
[Approvals](configuration.md#approvals) before running project code.

In text mode, read the answer on stdout and progress and errors on stderr.
With `--json`, read one event per line from stdout and warnings and errors
from stderr. To take a prompt from a file, redirect it to `-p`:

```sh
openmax -p --json < task.txt > events.jsonl
```

Piped or redirected stdin also follows the first prompt when you provide one
on the command line. Pass `</dev/null` in a script that needs stdin for other
work or when the parent leaves its pipe open. See [print mode](usage.md#print-mode)
for the input limit and end-of-input rules.

Repeat `-p` with a prompt for each turn to run them in one session. A nonzero
turn status skips the remaining prompts.

## Read the result

Check the process exit status first. With `--json`, each completed turn has
a `done` event; read its `stop_reason` to tell a normal `stop` from an early
end. A `done` event alone does not mean the task finished, and a startup
error or interrupted run may have no `done` event.

| Code | Meaning and next step |
| --- | --- |
| 0 | Every turn finished. A warning such as a failed session save does not change this status. Read the answer and check the work. |
| 1 | A run failed or could not start, could not open the session, or ended on a different stop reason such as `truncated`, `empty`, `blocked` or a provider's `content_filter`. Read the reported error or stop reason. |
| 2 | A usage or configuration error before any turn, or `--continue` or `--session` could not find a session this directory can continue. Correct the command or settings. |
| 3 | The project is untrusted, the process cannot grant trust or select `--permissions`, or a turn that otherwise finished declined an approval. Follow the trust or approval instruction on stderr. |
| 4 | The turn stopped short with `length`, `max_iterations`, `budget_exhausted` or `unverified`. Use `openmax --continue -p "<follow-up>"` to continue. If a gate could not start or a hook failed closed, repair the hook first. |
| 128 + signal number | A signal stopped the run. For SIGTERM, expect 143; check the saved session before deciding whether to continue. |

In `readonly` mode, a policy refusal does not request approval. A turn can
finish with `stop` and exit 0 even if a tool call was blocked. If the task
requires that call to run, check its tool result or verify the expected effect.

## Bound the run

- Set `--max-agent-iterations <n>` to cap the rounds per turn for this run.
  The default is 300. Reaching the cap ends the turn with `max_iterations`.
- Set `--max-agent-tokens <n>` to limit token spend per turn. It is off by
  default. If the next prompt would not fit in the tokens left, the turn
  stops with `budget_exhausted`. Replies count toward spend, so this is not
  an exact cap on billed tokens or money. Neither flag writes to settings.
- Set `request_budget` in `settings.json` to give the model a soft target for
  the number of requests per turn. It is off when unset or 0. The model sees
  the target before the prompt and reminders at 50, 75, 90 and 100 percent.
  Transport retries and compaction summaries do not count toward that target.
  It never stops a turn. Use the iteration or token flag for a stop.

There is no overall deadline. A silent endpoint ends an attempt after 600
seconds by default. Set a positive `idle_timeout_secs` on a named provider to
change that interval. If the server sends nothing, Open Max can make up to
eight attempts, taking about 80 minutes at the default before giving up.
Keepalives count as activity. Set a deadline in the caller if the whole run
must end by a fixed time.

## Stop a run and keep its records

Send SIGTERM to cancel the turn. Open Max stops its tools and gives the turn
up to three seconds to settle. Keep draining stdout so the run can act on the
signal. See [Signals and cancellation](execution.md#signals-and-cancellation)
for the cleanup rules and limits.

Changes already written to the project stay on disk. Keep the home directory
if you need the saved transcript or want to resume. Use the `session_id` from
the events with `openmax --session <id> -p "<follow-up>"` in the same directory.
The event log at `$HOME/.openmax/sessions/<id>.events.jsonl` can be replayed
with `openmax --replay <file> --json`. The log may omit events that had not
been written when the run was interrupted. See
[Sessions and replay](usage.md#sessions-and-replay).

## Set the tool environment

Shell commands and hooks inherit the environment and get
`GIT_TERMINAL_PROMPT=0`. External tools get `PATH`, `HOME`, `LANG` and `TERM`,
plus the variables their manifests name. When `EMAIL` is unset, commands the
agent runs get `EMAIL=openmax@localhost` as a fallback email only. If the OS
account has no display name, set Git's `user.name` or supply the author and
committer names explicitly. Open Max does not supply a name.

When `CARGO_CACHE_AUTO_CLEAN_FREQUENCY` is unset in the environment that
launches Open Max, commands the agent runs get `CARGO_CACHE_AUTO_CLEAN_FREQUENCY=never`.
Cargo gives the variable precedence over `cache.auto-clean-frequency` in its
configuration files. If you rely on that file setting, export the variable
with the same value before launching Open Max.

Tools have no controlling terminal, so provide any needed credentials or Git
identity before they run, not through a prompt.
See [Native execution and privacy](execution.md) for what commands can access.

## Checklist

- Check the archive's checksum and the binary's version before the job runs.
- Keep `HOME` outside the project and supply the key through the environment.
- Set the actual model window and reply limits. Inspect the project before
  granting trust and selecting a mode.
- Set turn caps and a caller deadline. Read both the exit code and the final
  stop reason, and keep the saved session if you need to resume.
