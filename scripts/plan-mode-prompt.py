#!/usr/bin/env python3
"""user_prompt_submit: arm/disarm plan mode from /plan and /execute markers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARM = "<!-- openmax-plan-mode-arm-9f3c1a7e42b8d605 -->"
DISARM = "<!-- openmax-plan-mode-disarm-1b7e4c92a0f5d38a -->"
STATE = Path(".openmax/state/plan-armed")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(f"plan-mode-prompt: invalid stdin JSON: {exc}", file=sys.stderr)
        return 1

    text = payload.get("text") or ""
    STATE.parent.mkdir(parents=True, exist_ok=True)

    # Honor the marker ONLY on the first non-empty line. The /plan and
    # /execute templates put their marker there (template body first, user
    # $ARGUMENTS appended after), while a user quoting the disarm marker
    # inside a /plan request lands it far below the arm marker the template
    # already put on line one - so it cannot silently disarm the gate. The
    # user_prompt_submit payload carries only the expanded text, not the
    # trusted slash-command, so first-line is the strongest signal available
    # (Greptile P1: a public substring anywhere is a disarm switch).
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")

    # Check arm first: a /plan whose args happen to quote the disarm marker
    # still arms, because the template's arm marker is what sits on line one.
    if first_line == ARM:
        STATE.write_text("armed\n", encoding="utf-8")
        return 0
    if first_line == DISARM:
        if STATE.exists():
            STATE.unlink()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
