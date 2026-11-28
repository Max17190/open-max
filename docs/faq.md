# FAQ

Short answers, each with a link to the page or example that holds the detail.

## How big is each request, and how do I see it?

Every request starts with the built-in prompt: the base rules, a map of your
project's top two levels and the schemas of the seven built-in tools. It is
under 700 tokens with the map of a fixed snapshot of this repository's tree,
which the size test uses; the map is capped at 1,200 bytes, so a project with
a larger tree pays a little more. Your own `AGENTS.md`, the index lines of
your skills and memories, and the schemas of your own tools add to it, and the
conversation comes after it.

`/context` lists each part of the prompt and the tools with its cost. Its
"total per request" row sums those parts only; its "context used" row counts the
whole conversation as well, the notes in it included. Once the provider reports it, it
also shows the share served from its cache. `/usage` shows what the session
has spent so far: the tokens it sent, of which cached, and the tokens it
received, as the provider reported them (no prices). See
[usage](usage.md#slash-commands).

## Why does the prompt not change under me within a session?

The system prompt and the tools array are built once, when the session starts,
and stay the same bytes until you `/reload`, so a provider's prompt cache keeps
serving them. A tool, skill or memory written during the session rides a note
in the conversation instead, and an edit to `AGENTS.md` waits for `/reload` or
the next session. See [freezing and re-freezing](extending.md#freezing-and-re-freezing).

## Does a release change the built-in prompt?

Sometimes, and its notes say so. The repository keeps a record of the built-in
prompt's bytes (the base rules and the built-in tools array), and a test that
gates every pull request fails when the code builds other bytes, so a change
updates the record in the same pull request. Each release's notes on GitHub
then say whether the record changed since the previous release, with each
file's size before and after and how many bytes differ. If you pin a version,
read them before you update: changed bytes mean a provider's cache from the old
version no longer matches from the first changed byte on. See
[development](development.md).

## What keeps the built-in prompt small between releases?

A test that gates every pull request, `frozen_prefix_is_under_700_tokens`,
holds the built-in prompt and tool schemas to a 3,036-byte cap set from the
700-token target, and the record above makes any change to those bytes
visible. [Development](development.md) gives the current size and the method.

## What does it leave running, and what can it not stop?

Each `bash` or tool call runs in its own process group, and whatever it leaves
running in the background is stopped when the call returns. A signal or a
`kill -9` of Open Max stops its tools too: after `kill -9`, a watcher process
stops them within 100 ms. A process that starts a session of its own (as
`setsid` does) is out of reach and keeps running, and so is, on macOS, one that
leaves its tool's process group. When a terminal UI session ends, it prints
each process its commands left running that it can find, with its pid, its
command name and the TCP ports it listens on, and the `kill` command that stops
them; it stops none itself, and `/status` shows the same list during the
session. To keep work running on purpose, start it in a named tmux session.
See [native execution](execution.md#signals-and-cancellation) and
[what a session leaves running](execution.md#what-a-session-leaves-running).

## How much memory does it use, and how fast does it start?

[Development](development.md) states each target, the ratchet CI holds in its
place and what was measured. Two targets are not met yet. Peak memory: a
two-round headless turn in a release build peaks at 6,544 KiB by median,
against a 4,000,000-byte target. Startup: the 1 ms first-frame target is not
asserted in CI, and on the reference machine (an Apple M5) the harness's own
startup work is held under a 450 us ratchet, against a 100 us target. Loading
100 extensions meets its 10 ms target.

## What does it send over the network?

Open Max has no telemetry. It contacts your model endpoint, OpenAI's sign-in
service if you connect a ChatGPT account, and, at most once a day from an
interactive session of an installed release, `api.github.com` to check for a
newer version. `OPENMAX_NO_UPDATE_CHECK=1` turns that check off. Shell
commands, tools and hooks run with your user's network access and can make
their own connections. See [privacy](execution.md#permissions-and-privacy) and
[updating](install.md#updating).

## Can I use my own model or subscription?

Yes. Use an OpenAI or Anthropic key, any server with a compatible API (local
ones included), or an eligible ChatGPT plan through `openmax --chatgpt login`.
Configure several and switch between them in a session with `/model`. Whether a
subscription may be used in a third-party tool is decided by its provider's
terms. See [getting started](getting-started.md),
[multiple providers](configuration.md#multiple-providers) and
[ChatGPT sign-in](configuration.md#chatgpt-sign-in).

## The agent said done, but the tests fail. What now?

Hold the end of every turn to your test command with a completion gate: while
the command fails, Open Max refuses the end and sends the failure back to the
model. [examples/done-gate](../examples/done-gate/README.md) is the two files
to copy. For a script or CI job that must judge the result itself, see
[hold a run to your own checks](verified-execution.md).

## How does it remember, and how do I see what it remembers?

The agent writes one fact per file under `.openmax/memory/`, and each new
session's prompt lists the recently used ones. Tools and skills it writes are
files too, and the next session loads them. `/memory` shows each memory, how
often it was used and whether the index still lists it, and
`openmax --recall "<query>"` searches past sessions and memories. See
[memory](extending.md#memory).

## Can I use MCP servers?

Yes, through one proxy tool and one skill per server, with the `openmax`
binary as the bridge. Every request pays for one small schema and one index
line; the server's own tool list is read only when the skill is.
[examples/mcp-tool](../examples/mcp-tool/README.md) works it through, and
[MCP servers](extending.md#mcp-servers) states the limits.

## How do I make it plan before editing?

Use a prompt template: [examples/plan-template](../examples/plan-template/README.md)
adds `/plan <task>`, which writes `PLAN.md` and stops for your review. Run the
planning turn in `readonly` to have the harness refuse its edits; the plan then
comes in the reply.

## Can it work in parallel or in isolation?

`openmax --subagent "<task>" --worktree` runs one child session in a new git
worktree and leaves its changes on a branch of their own. Each command runs
one child and nothing schedules them, so for parallel work start several, in
tmux for example. See [sub-agents](usage.md#sub-agents).

## Why is there no built-in MCP host, plan mode or agent scheduler?

Each would add instructions or tool schemas to the prompt that every request
resends, whether or not you use it. The same work is built from files you can
read and remove: a proxy tool and a skill per server (one small schema and one
index line), a prompt template and `PLAN.md` (nothing until used), and a
sub-agent per task (nothing in any request). See [design](design.md#design).
