#!/usr/bin/env python3
"""user_prompt_submit: arm/disarm plan mode from /plan and /execute markers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ARM = "<!-- openmax:plan-mode:arm -->"
DISARM = "<!-- openmax:plan-mode:disarm -->"
STATE = Path(".openmax/state/plan-armed")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(f"plan-mode-prompt: invalid stdin JSON: {exc}", file=sys.stderr)
        return 1

    text = payload.get("text") or ""
    STATE.parent.mkdir(parents=True, exist_ok=True)

    # Disarm first so a body that somehow contained both ends disarmed.
    if DISARM in text:
        if STATE.exists():
            STATE.unlink()
        return 0

    if ARM in text:
        STATE.write_text("armed\n", encoding="utf-8")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
