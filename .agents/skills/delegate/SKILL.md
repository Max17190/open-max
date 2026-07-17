---
name: delegate
description: Hand a sub-task to a child openmax process (headless one-shot or interactive stdio) instead of doing it inline; the subagent replacement.
---

# Delegate

Use when a sub-task deserves its own context window: a large refactor step, an isolated investigation, work in a different directory, or anything you want to run in parallel with the main thread of work.

## One-shot: headless print

For a self-contained task with a clear deliverable:

```sh
cd /path/to/target/project
openmax -p "Rename the config module to settings across this crate; run cargo check; report what changed."
```

- Output text arrives on stdout; tool progress on stderr.
- Chain sequential turns on one child session: `openmax -p "explore X" -p "now fix X"`.
- Mutating tools honor `approval_mode`; for unattended writes the child needs `"approval_mode": "auto"` in settings.

## Interactive: stdio protocol

When you need to react to the child mid-run (answer approvals, add follow-ups, cancel), drive `openmax --stdio`: JSONL commands on stdin, AgentEvent JSONL on stdout.

```sh
mkfifo /tmp/child-in
openmax --stdio < /tmp/child-in > /tmp/child-out 2>/dev/null &
exec 3>/tmp/child-in
echo '{"cmd":"user","text":"Map the auth flow; list key files."}' >&3
# read /tmp/child-out lines; {"type":"done",...} ends the turn
echo '{"cmd":"user","text":"Now write the summary to AUTH.md."}' >&3
echo '{"cmd":"quit"}' >&3
```

- Commands: `user` (start a turn), `approve` (`approval_id`, `approved`), `cancel`, `quit`.
- Approvals are not auto-declined: watch for `{"type":"approval_request",...}` and answer with `approve`.
- EOF acts like `quit`: the in-flight turn drains, then the child exits, so a plain pipe works for scripted runs.

## Long-running or visible: tmux

For work you or the user may want to watch or rejoin:

```sh
tmux new-session -d -s delegate-auth 'cd /path/to/project && openmax'
tmux send-keys -t delegate-auth "Map the auth flow, then wait." Enter
```

## Bring results back

Have the child write its deliverable to a file you both agree on (for example `NOTES.md` or the file it was asked to produce), or capture its stdout. Summarize into the main session; never paste raw transcripts.

## Do not

- Recurse without a budget: a child may delegate again, so state depth limits in the child's prompt ("do not spawn further openmax processes").
- Run two children mutating the same files; split by directory or run sequentially.
