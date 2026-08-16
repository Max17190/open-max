---
name: plan-mode
description: Read-only planning via /plan and /execute; harness blocks all writes except PLAN.md while armed.
---

# Plan mode

## Commands
- `/plan [focus]` — arms plan mode (harness-level). Only `PLAN.md` is writable.
- `/execute [note]` — disarms plan mode and implements `PLAN.md`.

## How enforcement works
1. `user_prompt_submit` hook (`scripts/plan-mode-prompt.py`) toggles
   `.openmax/state/plan-armed` when the FIRST non-empty line of the submitted
   text is the arm or disarm marker - an opaque high-entropy HTML comment the
   `/plan` and `/execute` templates put on line one, before any user
   `$ARGUMENTS`. First-line-only means a `/plan` whose args quote the disarm
   marker still arms; the opaque token means an ordinary or pasted message
   does not toggle it by accident.

   Threat model: the `user_prompt_submit` payload carries only the expanded
   prompt text, not the trusted slash-command, so the arm/disarm signal is
   necessarily in-band. Plan mode is the USER's own guard - the same user who
   armed it can `/execute` - so a user deliberately reproducing the marker is
   not a privilege escalation, and file content the agent reads never reaches
   this hook (it fires on the user's prompt only). This is an example
   capability, not a core security boundary; treat plan mode as a workflow
   guard, not a sandbox.
2. `pre_tool_use` hook (`scripts/plan-mode-gate.py`) runs on every tool call.
   While the state file exists it allows only:
   - `read_file`, `list_dir`, `glob`, `grep`
   - `write_file` / `edit_file` when `path` resolves to `<project>/PLAN.md`
   and blocks everything else with a clear reason:
   - **`bash` is fully blocked**, including read-only commands (no command inspection)
   - custom tools and any non-PLAN.md write path

## Agent rules while armed
- Put the whole plan in `PLAN.md`; no other files.
- Do not implement, refactor, or "just fix one thing".
- Do not edit hooks, permissions, or state to bypass the gate.
- When the plan is good enough, stop and wait for the user to run `/execute`.

## Agent rules after `/execute`
- Follow `PLAN.md`; update it if reality diverges.
- Prefer small diffs; verify before declaring done.

## State
- Armed when `.openmax/state/plan-armed` exists.
- State persists across turns and sessions until `/execute` (or manual removal).
- Hooks are inert until approved: `openmax --approve .openmax/hooks/plan-mode-prompt.toml`
  and `openmax --approve .openmax/hooks/plan-mode-gate.toml`.
