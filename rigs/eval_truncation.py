#!/usr/bin/env python3
"""Does the surviving slice of a truncated tool output carry what the agent
later needed?

Stage-1 budget enforcement keeps the FIRST 160 characters of an old tool
output and discards the rest. That choice is positional: it assumes the head
of a file read or a grep result is the part that matters. This measures that
assumption against real transcripts, using the agent's own future as ground
truth.

Method, per long tool output in a real session:
  - "referenced later" = the set of distinctive tokens that appear both in
    this tool output AND in some LATER assistant message. Those are the
    things the agent actually went on to use, so they are what truncation
    should have preserved. No labeling, no model: the transcript says it.
  - compare candidate slices of equal length:
      head      first N chars (what ships today)
      tail      last N chars
      headtail  N/2 from each end
      relevant  the N-char window maximizing idf-weighted coverage of the
                LIVE CONVERSATION's terms - the working-set principle: keep
                the part of the page the current context references
  - score = fraction of later-referenced tokens retained.

The 'relevant' selector is deliberately blind to the future: it scores only
against messages BEFORE the truncation point, exactly what the harness knows
at eviction time. Scoring against the future would be an oracle.

Reads your own transcripts, so it measures the workload you actually have.
Set OPENMAX_SESSIONS to point elsewhere; defaults to ~/.openmax/sessions.
Nothing is written and nothing leaves the machine.

  python3 eval_truncation.py [--width N]
"""
import json
import math
import os
import pathlib
import re
import sys
from collections import Counter

SESSIONS = pathlib.Path(
    os.environ.get("OPENMAX_SESSIONS", pathlib.Path.home() / ".openmax" / "sessions")
)
TRUNC_MIN = 600   # only outputs longer than this are truncated today
DEFAULT_WIDTH = 160


def toks(text):
    """The harness tokenizer: alphanumeric runs, lowercased, plus camelCase
    parts (as shipped in #112)."""
    out = []
    for run in re.findall(r"[A-Za-z0-9]+", text):
        if len(run) < 2:
            continue
        parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+", run)
        parts = [p.lower() for p in parts if len(p) >= 2]
        whole = run.lower()
        if len(parts) < 2:
            out.append(whole)
        else:
            if not any(len(p) >= 5 and whole.startswith(p) for p in parts):
                out.append(whole)
            out.extend(parts)
    return out


STOP = set("""a an and are as at be but by did do does for from had has have how i if in is it
of on or our so that the then this to was we were what when where which who why will with you
the not no yes can will just also""".split())


def distinctive(text):
    return {t for t in toks(text) if t not in STOP and len(t) >= 4}


def load(path):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            v = json.loads(line)
        except Exception:
            continue
        for m in (v if isinstance(v, list) else [v]):
            if isinstance(m, dict):
                out.append(m)
    return out


def best_window(text, query_terms, idf, width):
    """The width-char window covering the most DISTINCT query terms. Scans on
    line boundaries: tool output is line-structured, and a window cut mid-line
    reads as garbage.

    `idf` is accepted so the weighted variant can be compared, but the shipped
    selector counts distinct terms uniformly and that is what this scores.
    Weighting by idf measured worse (0.158 vs 0.180 at width 160, paired 95%
    CI [-0.032, -0.013]): rarity inside a single tool output selects for
    hashes, offsets and line numbers, which are exactly the tokens nothing
    refers to again. Pass `idf={}` to reproduce the shipped behaviour, or a
    real idf map to reproduce the rejected one."""
    lines = text.split("\n")
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1
    best, best_score = 0, -1.0
    for i, s in enumerate(starts):
        window = text[s:s + width]
        cov = {t for t in distinctive(window) if t in query_terms}
        score = sum(idf.get(t, 1.0) for t in cov)
        if score > best_score:
            best, best_score = s, score
    return text[best:best + width]


def main():
    width = DEFAULT_WIDTH
    if "--width" in sys.argv:
        width = int(sys.argv[sys.argv.index("--width") + 1])

    strategies = ["head", "tail", "headtail", "relevant", "relevant-idf"]
    totals = {s: 0.0 for s in strategies}
    cases = 0
    per_session = {}

    for meta in json.load(open(SESSIONS / "index.json")):
        mp = SESSIONS / f"{meta['id']}.messages.json"
        if not mp.exists():
            continue
        msgs = [m for m in load(mp) if m.get("role") != "system" and (m.get("content") or "")]
        if len(msgs) < 8:
            continue
        # corpus idf over this session's messages
        df = Counter()
        for m in msgs:
            for t in distinctive(m["content"]):
                df[t] += 1
        n = len(msgs)
        idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

        for i, m in enumerate(msgs):
            if m.get("role") != "tool":
                continue
            body = m["content"]
            if len(body) <= TRUNC_MIN:
                continue
            # ground truth: what the agent went on to use from this output
            future = set()
            for later in msgs[i + 1:]:
                if later.get("role") == "assistant":
                    future |= distinctive(later["content"])
            target = distinctive(body) & future
            if len(target) < 3:
                continue
            # what the harness knows at eviction time: the conversation so far
            past_terms = set()
            for earlier in msgs[:i]:
                past_terms |= distinctive(earlier["content"])

            slices = {
                "head": body[:width],
                "tail": body[-width:],
                "headtail": body[:width // 2] + body[-(width - width // 2):],
                # The shipped selector: uniform distinct-term coverage.
                "relevant": best_window(body, past_terms, {}, width),
                # The variant that was measured and rejected.
                "relevant-idf": best_window(body, past_terms, idf, width),
            }
            cases += 1
            sid = meta["id"][:8]
            per_session.setdefault(sid, {s: [0.0, 0] for s in strategies})
            for name, sl in slices.items():
                kept = len(distinctive(sl) & target) / len(target)
                totals[name] += kept
                per_session[sid][name][0] += kept
                per_session[sid][name][1] += 1

    if not cases:
        print("no truncation cases found")
        return
    print(f"truncation slice quality: {cases} real tool outputs > {TRUNC_MIN} chars, "
          f"window {width} chars")
    print("  fraction of later-referenced tokens retained:")
    for s in strategies:
        print(f"    {s:9} {totals[s] / cases:.3f}")
    print("\n  per session:")
    for sid, d in sorted(per_session.items()):
        row = "  ".join(f"{s}={d[s][0] / d[s][1]:.2f}" for s in strategies)
        print(f"    {sid}  n={d['head'][1]:3d}  {row}")


if __name__ == "__main__":
    main()
