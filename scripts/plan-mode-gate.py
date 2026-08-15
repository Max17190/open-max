#!/usr/bin/env python3
"""pre_tool_use gate: while plan mode is armed, only PLAN.md may be written."""
from __future__ import annotations

import json
import sys
from pathlib import Path

STATE = Path(".openmax/state/plan-armed")
# Tools that never write project files (always allowed while planning).
READ_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "glob",
        "grep",
    }
)
# Tools whose path arg may target PLAN.md only.
PATH_WRITE_TOOLS = frozenset({"write_file", "edit_file"})


def project_root() -> Path:
    return Path.cwd().resolve()


def is_plan_md(path_arg: str) -> bool:
    """True iff path_arg resolves to <project>/PLAN.md."""
    if not path_arg or not isinstance(path_arg, str):
        return False
    root = project_root()
    plan = (root / "PLAN.md").resolve()
    raw = Path(path_arg)
    try:
        cand = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    except (OSError, RuntimeError):
        return False
    try:
        cand.relative_to(root)
    except ValueError:
        return False
    return cand == plan


def block(reason: str) -> int:
    # stdout is the block reason shown to the model (cap 500 by harness).
    sys.stdout.write(reason[:500])
    return 1


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        return block(f"plan-mode-gate: invalid stdin JSON: {exc}")

    if not STATE.is_file():
        return 0

    tool = payload.get("tool") or ""
    args = payload.get("args") or {}
    if not isinstance(args, dict):
        args = {}

    # bash is always blocked while armed — including read-only commands.
    # No command inspection: any bash call is a bypass path (pipes, redirects).
    if tool == "bash":
        return block(
            "plan mode armed: bash is fully blocked (including read-only commands). "
            "Use read_file/list_dir/glob/grep, or run /execute to leave plan mode."
        )

    if tool in READ_TOOLS:
        return 0

    if tool in PATH_WRITE_TOOLS:
        path = args.get("path", "")
        if is_plan_md(path):
            return 0
        return block(
            "plan mode armed: only PLAN.md may be written. "
            f"Blocked {tool} path={path!r}. Run /execute to leave plan mode."
        )

    # Custom tools and anything else can mutate the tree.
    return block(
        f"plan mode armed: tool {tool!r} blocked (read tools + PLAN.md writes only). "
        "Run /execute to leave plan mode."
    )


if __name__ == "__main__":
    raise SystemExit(main())
