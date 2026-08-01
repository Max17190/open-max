#!/usr/bin/env python3
"""Labeled precision benchmark v2 for `openmax --recall`.

Adds what the 12-query rig could not support: a corpus large enough to
resolve small deltas, a held-out domain, and paired bootstrap confidence
intervals so a reported improvement is distinguishable from noise.

  python3 eval2.py BIN                      # dev corpus (domain A)
  python3 eval2.py BIN --corpus test        # held-out corpus (domain B)
  python3 eval2.py BIN --corpus both
  python3 eval2.py BASE --vs CAND           # paired A/B with 95% CIs
  python3 eval2.py BIN --json

Metrics
  hit@1/hit@3  gold source ranked first / top three
  MRR          mean reciprocal rank of the gold source
  needle@1     the answer text visible in the first gold hit (v1-comparable)
  cover        fraction of a query's gold needles visible anywhere in output
  tokens       mean output cost (chars/4) per query
"""
import json
import os
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import uuid

import corpus

RIG = pathlib.Path(__file__).resolve().parent
CORPORA = {"dev": ("A", 11), "test": ("B", 29)}


def run_corpus(binary, which, keep=False):
    domain_key, seed = CORPORA[which]
    root = RIG / "runs" / f"v2-{which}-{uuid.uuid4().hex[:8]}"
    store, queries = corpus.build(root, corpus.DOMAINS[domain_key], seed)
    env = dict(os.environ, HOME=str(store.home))
    rows = []
    for spec in queries:
        gold = store.labels[spec["gold"]]
        out = subprocess.run([binary, "--recall", spec["q"], "--json"],
                             cwd=store.project, env=env,
                             capture_output=True, text=True, timeout=120)
        hits, tokens, elapsed = [], 0, 0
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout)
            hits = data.get("hits", [])
            elapsed = data.get("elapsed_ms", 0)
            tokens = sum((len(h.get("excerpt", "")) + len(h.get("source", ""))) // 4 + 12
                         for h in hits)

        def is_gold(h):
            return h.get("kind") == "memory" if gold == "memory" else h.get("session") == gold

        rank = next((i for i, h in enumerate(hits, 1) if is_gold(h)), 0)
        gold_text = " ".join(h.get("excerpt", "") for h in hits if is_gold(h)).lower()
        needles = [n.lower() for n in spec["needles"]]
        cover = sum(n in gold_text for n in needles) / len(needles)
        needle1 = bool(rank) and needles[0] in hits[rank - 1].get("excerpt", "").lower()
        rows.append({"corpus": which, "class": spec["cls"], "query": spec["q"],
                     "rank": rank, "hit1": rank == 1, "hit3": 0 < rank <= 3,
                     "rr": (1.0 / rank) if rank else 0.0, "needle1": needle1,
                     "cover": cover, "tokens": tokens, "ms": elapsed,
                     "returned": len(hits)})
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return rows


METRICS = ["hit1", "hit3", "rr", "needle1", "cover"]


def agg(rows):
    n = len(rows)
    out = {"queries": n}
    for m in METRICS:
        out[m] = sum(r[m] for r in rows) / n
    out["tokens"] = sum(r["tokens"] for r in rows) / n
    out["ms"] = statistics.median(r["ms"] for r in rows)
    return out


def bootstrap_delta(base, cand, metric, iters=10000, seed=7):
    """Paired bootstrap over queries: resample query indices, recompute the
    delta on the same indices for both binaries.  Paired because both runs
    answer the identical query set, so per-query difficulty is shared noise
    and removing it is what makes a 12-to-100 query corpus worth building."""
    rng = random.Random(seed)
    n = len(base)
    diffs = [cand[i][metric] - base[i][metric] for i in range(n)]
    point = sum(diffs) / n
    samples = []
    idx = range(n)
    for _ in range(iters):
        pick = [rng.choice(idx) for _ in range(n)]
        samples.append(sum(diffs[i] for i in pick) / n)
    samples.sort()
    lo = samples[int(0.025 * iters)]
    hi = samples[int(0.975 * iters)]
    return point, lo, hi


def show(name, rows, per_class=True):
    a = agg(rows)
    print(f"{name}  ({a['queries']} queries)")
    print(f"  hit@1 {a['hit1']:.3f}  hit@3 {a['hit3']:.3f}  MRR {a['rr']:.3f}  "
          f"needle@1 {a['needle1']:.3f}  cover {a['cover']:.3f}  "
          f"tokens {a['tokens']:.0f}  med_ms {a['ms']:.0f}")
    if not per_class:
        return
    classes = {}
    for r in rows:
        classes.setdefault(r["class"], []).append(r)
    for cls in sorted(classes):
        c = agg(classes[cls])
        flag = "  <-- " if c["hit3"] < 1.0 or c["cover"] < 1.0 else "      "
        print(f"   {flag}{cls:14} n={c['queries']:3d} hit@1 {c['hit1']:.2f} "
              f"hit@3 {c['hit3']:.2f} needle@1 {c['needle1']:.2f} cover {c['cover']:.2f}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    binary = args[0]
    which = "dev"
    if "--corpus" in args:
        which = args[args.index("--corpus") + 1]
    corpora = ["dev", "test"] if which == "both" else [which]
    vs = args[args.index("--vs") + 1] if "--vs" in args else None

    base_rows = [r for c in corpora for r in run_corpus(binary, c)]
    if not vs:
        if "--json" in args:
            print(json.dumps({"aggregate": agg(base_rows), "rows": base_rows}, indent=1))
            return
        show(f"BASE {os.path.basename(binary)} [{'+'.join(corpora)}]", base_rows)
        misses = [r for r in base_rows if not r["hit3"] or r["cover"] < 1.0]
        if misses:
            print(f"\n  {len(misses)} imperfect queries:")
            for r in misses:
                print(f"    rank={r['rank'] or '-'} cover={r['cover']:.2f} "
                      f"{r['class']:14} {r['query']}")
        return

    cand_rows = [r for c in corpora for r in run_corpus(vs, c)]
    show(f"BASE {os.path.basename(binary)}", base_rows, per_class=False)
    show(f"CAND {os.path.basename(vs)}", cand_rows, per_class=False)
    print("\n  paired delta (cand - base), 95% bootstrap CI over queries:")
    for m in METRICS:
        p, lo, hi = bootstrap_delta(base_rows, cand_rows, m)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns "
        print(f"    {sig} {m:8} {p:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    bt = sum(r["tokens"] for r in base_rows) / len(base_rows)
    ct = sum(r["tokens"] for r in cand_rows) / len(cand_rows)
    print(f"        tokens   {ct - bt:+.0f}  ({bt:.0f} -> {ct:.0f})")
    regressions = [(b, c) for b, c in zip(base_rows, cand_rows)
                   if c["rr"] < b["rr"] - 1e-9 or c["cover"] < b["cover"] - 1e-9]
    if regressions:
        print(f"\n  {len(regressions)} per-query regressions:")
        for b, c in regressions:
            print(f"    {b['class']:14} rank {b['rank'] or '-'}->{c['rank'] or '-'} "
                  f"cover {b['cover']:.2f}->{c['cover']:.2f}  {b['query']}")


if __name__ == "__main__":
    main()
