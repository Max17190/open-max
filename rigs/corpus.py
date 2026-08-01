#!/usr/bin/env python3
"""Seeded generator for labeled recall corpora.

Builds a realistic agent-history store (sessions, archives, compaction
digests, memory files) with planted facts and labeled queries.  Two
independent domains, so a dev corpus and a test corpus differ in vocabulary
rather than merely in random seed: tuning on one and reporting on the other
is a genuine held-out measurement, not a reshuffle.  Every content word a
query can use is drawn from the domain, so no phrasing accidentally works on
one domain and misses on the other.

Message shapes follow real openmax transcripts: short user turns, medium
assistant turns, and long tool outputs that dominate the bytes.  Noise lines
carry varying numbers and phrasing, because uniform boilerplate would make
repetition-damping look better than it is.

Every query carries:
  cls      the failure class it probes
  gold     session label (or "memory") holding the answer
  needles  strings the output should surface; multi-needle queries measure
           within-session recall, not just ranking
"""
import json
import os
import pathlib
import random
import time
import uuid

HOUR = 3600
NOW = int(time.time())


DOMAIN_A = {
    "name": "checkout-service",
    # [0] subject  [1] pooled thing  [2] fix topic  [3] peer  [4] product area
    "comp": ["gateway", "proxy", "keepalive", "upstream", "checkout",
             "cart", "session", "payment", "webhook", "receipt"],
    "files": ["src/gateway/proxy.rs", "src/gateway/pool.ts", "src/cart/basket.ts",
              "config/nginx/upstream.conf", "src/payment/charge.rs",
              "src/webhook/receipt.ts", "src/session/store.rs"],
    "settings": [("keep_alive_msecs", "25000"), ("reaper_interval", "9s"),
                 ("upstream_timeout", "45s"), ("retry_budget", "2")],
    "runtime": "node",
    "noise": ["upstream latency {n}ms, keepalive window steady",
              "cart reconcile pass complete, {n} orphans",
              "payment webhook queue depth {n}, draining",
              "session store compaction skipped, {n} below threshold",
              "receipt batch {n} flushed to disk",
              "checkout span {n} sampled, trace exported"],
    "err_prefix": "ERR_STREAM",
    "pair": ("session", "vacuum"),
    "tuned": "cart",
}

DOMAIN_B = {
    "name": "ingest-pipeline",
    "comp": ["ingest", "shard", "compactor", "manifest", "segment",
             "watermark", "replica", "catalog", "vacuum", "tombstone"],
    "files": ["pkg/ingest/reader.go", "pkg/shard/router.go", "pkg/compactor/plan.go",
              "deploy/catalog/schema.yaml", "pkg/segment/writer.go",
              "pkg/replica/sync.go", "pkg/vacuum/sweeper.go"],
    "settings": [("flush_interval_ms", "25000"), ("sealer_interval", "9s"),
                 ("replica_timeout", "45s"), ("merge_fanout", "2")],
    "runtime": "python",
    "noise": ["replica lag {n}ms, within budget",
              "segment writer rolled at offset {n}, watermark advanced",
              "catalog scan clean, {n} tombstones expired",
              "compactor plan produced {n} candidates, deferred",
              "manifest checkpoint {n} durable",
              "shard router rebalanced {n} ranges"],
    "err_prefix": "ERR_SEGMENT",
    "pair": ("catalog", "vacuum"),
    "tuned": "shard",
}


def msg(role, content):
    return {"role": role, "content": content}


class Store:
    def __init__(self, root, project_name):
        self.root = pathlib.Path(root)
        self.home = self.root / "home"
        self.project = self.root / project_name
        self.sessions = self.home / ".openmax" / "sessions"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.project.mkdir(parents=True, exist_ok=True)
        self.index = []
        self.labels = {}

    def add(self, label, title, age_hours, messages, archive=None, compaction=None):
        sid = str(uuid.uuid4())
        ts = NOW - int(age_hours * HOUR)
        (self.sessions / f"{sid}.messages.json").write_text(
            "".join(json.dumps(m) + "\n" for m in messages))
        if archive:
            (self.sessions / f"{sid}.archive.jsonl").write_text(
                "".join(json.dumps(m) + "\n" for m in archive))
        if compaction:
            (self.sessions / f"{sid}.compaction.jsonl").write_text(
                "".join(json.dumps(c) + "\n" for c in compaction))
        self.index.append({"id": sid, "project": str(self.project), "title": title,
                           "created_at": ts, "updated_at": ts})
        if label:
            self.labels[label] = sid
        return sid

    def memory(self, label, name, text, age_hours=1):
        d = self.project / ".openmax" / "memory"
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text(text)
        ts = NOW - int(age_hours * HOUR)
        os.utime(p, (ts, ts))
        self.labels[label] = "memory"
        return name

    def finish(self):
        (self.sessions / "index.json").write_text(json.dumps(self.index))


def noise_block(rng, dom, lines):
    return "\n".join(rng.choice(dom["noise"]).format(n=rng.randrange(1, 9999))
                     for _ in range(lines))


def long_log(rng, dom, needle, lines=440):
    """A tool output big enough that the needle must be found by paging."""
    body = [rng.choice(dom["noise"]).format(n=rng.randrange(1, 9999)) for _ in range(lines)]
    body.insert(rng.randrange(len(body) // 4, 3 * len(body) // 4), needle)
    return "\n".join(body)


def build(root, dom, seed, n_distractors=34):
    """Build a corpus; returns (store, queries)."""
    rng = random.Random(seed)
    st = Store(root, dom["name"])
    c = dom["comp"]
    files = dom["files"]
    sets = dom["settings"]
    queries = []

    def q(cls, text, gold, needles):
        queries.append({"cls": cls, "q": text, "gold": gold,
                        "needles": needles if isinstance(needles, list) else [needles]})

    # ---- fact 1: rare error token buried in a long log -------------------
    errcode = f"{dom['err_prefix']}_PREMATURE_CLOSE"
    reqid = f"{rng.randrange(0x100000, 0xffffff):06x}"
    needle_line = (f"{errcode}: {c[0]} closed connection while writing "
                   f"body (req_id={reqid}, target={c[3]}:8443)")
    st.add("error", f"{c[0]} 502 investigation", 140, [
        msg("user", f"customers report intermittent 502s on {c[4]}, dig in"),
        msg("assistant", f"pulling the {c[0]} logs for the 502 window now"),
        msg("tool", long_log(rng, dom, needle_line)),
        msg("assistant", f"found it: the stream closes mid-body, req_id {reqid}"),
    ])
    q("rare-token", errcode, "error", reqid)
    q("rare-token", reqid, "error", reqid)
    q("truncated-id", errcode[:len(errcode) - 6], "error", reqid)
    q("needle-in-log", f"what was the original 502 error on {c[4]}", "error", errcode)
    q("needle-in-log", f"which request id failed in the {c[0]} log", "error", reqid)

    # ---- fact 2: an abandoned approach (why-not knowledge) ---------------
    st.add("wrong-turn", f"{c[1]} pre-warming experiment", 116, [
        msg("user", f"could pre-warming the {c[1]} pool fix the 502s?"),
        msg("assistant",
            f"tried pre-warming in {files[1]}; error rate went from 0.40% to 1.30% "
            f"because idle sockets outlive the {c[3]} 30s timeout. "
            f"Abandoning {c[1]} pool pre-warming; reverted the file."),
    ])
    q("natural-q", f"why did we abandon {c[1]} pool pre-warming", "wrong-turn", "0.40% to 1.30%")
    q("morphology", f"abandoning the {c[1]} experiment", "wrong-turn", "Abandoning")
    q("morphology", f"pre-warmed pools reverted", "wrong-turn", "reverted")
    q("natural-q", f"did pre-warming help the error rate", "wrong-turn", "1.30%")

    # ---- fact 3: multi-file fix, four distinct answers in ONE session -----
    # Each part of the answer is a separate message: an emission cap keyed on
    # the transcript file can rank the session first and still hide most of
    # the answer, so these queries are scored on how many needles surface.
    fix_topic = c[2]
    st.add("fix", f"{fix_topic} fix landed", 92, [
        msg("user", "so what actually changed to fix the 502s?"),
        msg("assistant", f"first change: {files[0]} sets {sets[0][0]} to {sets[0][1]}"),
        msg("assistant", f"second change: {files[2]} adds the {sets[1][0]} reaper"),
        msg("assistant", f"third change: {files[3]} raises {sets[2][0]} to {sets[2][1]}"),
        msg("assistant", f"fourth change: {files[4]} drops {sets[3][0]} to {sets[3][1]}"),
    ], compaction=[{"ts": NOW - 92 * HOUR, "message_count": 6, "tools": ["edit_file"],
                    "paths": [files[0], files[2], files[3], files[4]],
                    "user_snippets": ["fix the 502s"],
                    "digest": f"[context note: {fix_topic} fix across {c[1]}, reaper, {c[3]}]"}])
    all_needles = [s[0] for s in sets]
    q("multi-answer", f"which files changed for the {fix_topic} fix", "fix", all_needles)
    q("multi-answer", f"every setting we changed in the {fix_topic} fix", "fix", all_needles)
    q("path-hop", f"path:{files[0].rsplit('/', 1)[0]} {fix_topic}", "fix", sets[0][0])
    q("path-hop", f"path:{files[3]} timeout", "fix", sets[2][0])
    q("digest", f"compaction note about the {fix_topic} reaper", "fix", "reaper")
    q("value-lookup", f"what is {sets[0][0]} set to", "fix", sets[0][1])
    q("value-lookup", f"{sets[2][0]} value", "fix", sets[2][1])

    # ---- fact 3b: a SHORT session that never compacted -------------------
    # The file is named only in a tool result; the reasoning about it lives in
    # a prose message that never repeats the path. This is the case that a
    # structured-paths-only hop would lose, so it is measured, not argued.
    st.add("uncompacted", f"{files[5]} ownership", 36, [
        msg("user", f"who owns {files[5]}?"),
        msg("tool", f"{files[5]}\n{files[5]}:1: package header\n"),
        msg("assistant", "the platform team owns it; billing only reviews the schema "
                         "changes, and the on-call rotation does not cover it"),
    ])
    q("path-uncompacted", f"path:{files[5]} who owns it", "uncompacted", "platform team")

    # ---- fact 4: constraint, with a question-echo distractor -------------
    rt = dom["runtime"]
    st.add("constraint", "runtime constraint", 70, [
        msg("user", f"hard constraint: {rt} stays pinned to 18.17.1 until the "
                    "SOC2 audit closes in Q4 2026. no runtime bumps, no native deps."),
        msg("assistant", "acknowledged, recorded the constraint"),
    ])
    st.add("echo", "planning questions", 30, [
        msg("user", f"what constraint did the user set on the {rt} runtime, "
                    "and when does the SOC2 audit close? need this for planning"),
        msg("assistant", "I don't have that in front of me; I'll check the history later"),
    ])
    q("question-echo", f"what constraint did the user set on {rt}", "constraint", "18.17.1")
    q("question-echo", "when does the SOC2 audit close", "constraint", "Q4 2026")
    q("question-echo", f"is the {rt} runtime pinned", "constraint", "18.17.1")

    # ---- fact 5: memory file (curated fact competing with bulk chatter) ---
    st.memory("memory", "runtime-pin.md",
              f"# {rt} pinned to 18.17.1 until SOC2 closes Q4 2026\n"
              f"No {rt} runtime bumps, no native deps, no base image changes.\n",
              age_hours=70)
    q("memory", f"{rt} pinned version", "memory", "18.17.1")
    q("memory", f"native deps policy for {rt}", "memory", "native deps")

    # ---- fact 6: archived message ---------------------------------------
    st.add("archived", "header size hunt", 44, [
        msg("user", "what did we set max-http-header-size to?"),
        msg("assistant", "checking history"),
    ], archive=[msg("tool", "started with --max-http-header-size=32768 after the "
                            "oversized cookie incident")])
    q("archive", "max-http-header-size value", "archived", "32768")
    q("archive", "oversized cookie incident header", "archived", "32768")

    # ---- fact 7: proximity — both terms common apart, adjacent only here --
    pa, pb = dom["pair"]
    st.add("proximity", f"{pa} {pb} decision", 60, [
        msg("user", f"do we run {pa} {pb} on every write?"),
        msg("assistant",
            f"no: {pa} {pb} runs on a 15 minute timer, never inline with a write, "
            f"because inline {pa} {pb} doubled p99 latency in the trial."),
    ])
    for i in range(4):
        st.add(None, f"{pa} notes {i}", 20 + i * 5, [
            msg("user", f"is the {pa} healthy?"),
            msg("assistant", f"{pa} is fine. " + noise_block(rng, dom, 10).replace("\n", " ")
                + f" separately, the {pb} job is unrelated and untouched here."),
        ])
    q("proximity", f"{pa} {pb} timer", "proximity", "15 minute timer")
    q("proximity", f"is {pa} {pb} inline with writes", "proximity", "never inline")

    # ---- fact 8: disambiguation — common term plus one rare qualifier ----
    tuned = dom["tuned"]
    st.add("disambig", f"{tuned} tuning", 50, [
        msg("user", f"what did we settle on for the {tuned} batch size?"),
        msg("assistant", f"the {tuned} batch size is 512 records, chosen after the "
                         "throughput sweep; smaller batches starved the writer."),
    ])
    for i in range(5):
        st.add(None, f"{tuned} chatter {i}", 12 + i * 4, [
            msg("user", f"quick question about the {tuned}"),
            msg("assistant", f"the {tuned} is behaving; nothing to change today."),
            msg("tool", noise_block(rng, dom, rng.randrange(30, 80))),
        ])
    q("disambig", f"{tuned} batch size", "disambig", "512 records")
    q("disambig", f"why did we pick the {tuned} batch size", "disambig", "throughput sweep")

    # ---- fact 9: a decision with a superseding follow-up ------------------
    st.add("superseded", f"{c[6]} retention first pass", 200, [
        msg("user", f"how long do we keep {c[6]} data?"),
        msg("assistant", f"{c[6]} retention is 30 days for now"),
    ])
    st.add("current", f"{c[6]} retention revised", 20, [
        msg("user", f"legal wants {c[6]} retention shortened"),
        msg("assistant", f"updated: {c[6]} retention is now 7 days, superseding the "
                         "30 day policy, effective immediately"),
    ])
    q("recency", f"{c[6]} retention policy", "current", "7 days")

    # ---- background distractor sessions ---------------------------------
    for i in range(n_distractors):
        cc = rng.choice(c)
        f = rng.choice(files)
        st.add(None, f"routine maintenance {i}", 6 + i * 2, [
            msg("user", f"why did the linter complain about the {cc} module again, run {i}"),
            msg("assistant",
                f"the usual: unused import in {f}, fixed. "
                f"{rng.choice(c)} and {rng.choice(c)} settings untouched, "
                f"the {rng.choice(c)} path is fine here."),
            msg("tool", noise_block(rng, dom, rng.randrange(20, 90))),
        ])

    st.finish()
    return st, queries


DOMAINS = {"A": DOMAIN_A, "B": DOMAIN_B}
