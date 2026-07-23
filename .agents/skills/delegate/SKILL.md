---
name: delegate
description: Hand an isolated or parallel sub-task to a supervised child openmax process using headless, stdio, or durable tmux execution.
---

# Delegate

Use when a sub-task deserves its own context window: a large refactor step, an
isolated investigation, work in a different directory, or work that can proceed
in parallel with the main task.

## One-shot: headless print

For a self-contained task with a clear deliverable:

```sh
cd /path/to/target/project
openmax --trust-project -p "Rename the config module to settings across this crate; run cargo check; report what changed."
```

- Output text arrives on stdout; tool progress arrives on stderr.
- Chain sequential turns on one child session with repeated `-p`.
- Mutating tools honor `approval_mode`; unattended writes require an explicit
  configuration that permits them.

## Interactive: stdio protocol

Use `openmax --stdio` when the parent must answer approvals, add follow-ups, or
cancel. The JSONL contract is documented in the repository README.

## Durable tmux supervisor

Use the bundled supervisor for work that must survive the launching shell or
remain visible:

```sh
delegate=$(
  .agents/skills/delegate/scripts/openmax-tmux start auth-a /path/to/project \
    "Map the auth flow and write AUTH.md. Do not delegate further."
)
.agents/skills/delegate/scripts/openmax-tmux status "$delegate"
.agents/skills/delegate/scripts/openmax-tmux wait "$delegate"
.agents/skills/delegate/scripts/openmax-tmux capture "$delegate"
.agents/skills/delegate/scripts/openmax-tmux attach "$delegate"
.agents/skills/delegate/scripts/openmax-tmux stop "$delegate"
.agents/skills/delegate/scripts/openmax-tmux list
```

`start` canonicalizes the project directory, saves the exact prompt to a file,
creates a persistent generation directory, and atomically updates the logical
name symlink. It launches the configured `OPENMAX_BIN` or `openmax` using
`--trust-project`, captures combined output and the exact exit code, and uses a
tmux wait lock for race-free completion. Capture the immutable generation path
printed by `start` and pass that handle to every later command. Operational
commands reject mutable logical names. Starting the same logical name while its
current generation is running is also rejected. `list` reports every retained
generation and marks the current logical-name target. Every tmux command uses an
exact target.

State defaults to `~/.openmax/delegates`. Set `OPENMAX_DELEGATE_DIR` to relocate
it. Set `CARGO_TARGET_DIR` to share compilation artifacts across isolated Git
worktrees when appropriate.

tmux provides lifecycle and visibility, not filesystem or security isolation.
Concurrent delegates that may mutate files must use separate Git worktrees.
Read-only delegates may share a checkout.

## Bring results back

Ask the child to write a named deliverable or use `capture`. Summarize the
result into the parent session; do not paste raw transcripts.

## Do not

- Recurse without a depth budget. State "do not delegate further" by default.
- Run two mutating children in one worktree.
- Treat a tmux session as a permission or sandbox boundary.
