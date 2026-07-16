---
name: parallel-explore
description: Isolate heavy codebase exploration without nested agents; use focused tools or a second openmax session (tmux/headless).
---

# Parallel explore

Use when a broad "where is X / how does Y work" search would flood the main session with tool output.

## Prefer in-session first

1. Narrow with `glob` and `grep` (tight patterns, then expand).
2. `read_file` only the hits that matter.
3. Summarize findings in your reply. Do not dump whole trees into context.

## When work is too large for one context

Spawn a **second process**, not a nested agent tool:

```sh
# interactive pane (tmux or second terminal)
cd /path/to/project
openmax

# or headless one-shot
openmax -p "Map how auth middleware is registered; list key files and a short flow."
```

- Run from the **project root** so sandbox and sessions match.
- Optional: `openmax -p --json "..."` for machine-readable events on stdout.
- Mutating tools still honor `approval_mode`; for unattended print mode set `"approval_mode": "auto"` in settings if you need writes (rare for explore).

## Bring results back

- Paste a short summary into the main session, or
- Write a brief note file the user asked for (for example under the project root), then continue in the main loop.

## Do not

- Invent nested-agent or "subagent" tools.
- Assume a built-in `task` delegation tool exists.
- Start background bash job control as a harness feature; use a second terminal or tmux pane instead.
