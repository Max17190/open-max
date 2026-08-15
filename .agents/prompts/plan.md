---
description: Enter read-only plan mode (only PLAN.md writable)
---
<!-- openmax:plan-mode:arm -->

# Plan mode is now ARMED (harness-enforced)

The harness has armed plan mode for this project. Until `/execute`:

- You may **read** freely (`read_file`, `list_dir`, `glob`, `grep`).
- The **only file you may write or edit is `PLAN.md`** at the project root.
- Every other write, `bash`, and custom tool is **blocked** by a `pre_tool_use` gate.
- Do not try to disarm by editing state or hooks; only `/execute` disarms.

## Your job
1. Inspect the codebase with read tools only.
2. Write (or update) a concrete, ordered implementation plan in `PLAN.md`.
3. Keep `PLAN.md` current: goals, non-goals, steps, files to touch, risks, verification.
4. Stop when the plan is ready. Do **not** implement yet.

Extra focus from the user (may be empty):
$ARGUMENTS
