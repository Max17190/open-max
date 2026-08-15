---
name: plan-mode
description: Read-only planning via /plan and /execute; harness blocks all writes except PLAN.md while armed.
---

# Plan mode

## Commands
- `/plan [focus]` — arms plan mode (harness-level). Only `PLAN.md` is writable.
- `/execute [note]` — disarms plan mode and implements `PLAN.md`.

## How enforcement works
1. `user_prompt_submit` hook (`scripts/plan-mode-prompt.py`) sees the arm/disarm
   HTML comment markers injected by the `/plan` and `/execute` prompt templates
   and writes or removes `.openmax/state/plan-armed`.
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
