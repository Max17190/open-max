//! A labeled quality gate for `--recall`.
//!
//! The rest of the recall tests pin mechanisms: this one pins the outcome.
//! It builds a small history whose answers are known, asks the questions a
//! person actually asks, and fails when the ranking gets worse - the same
//! service `--check` performs for extension files, performed for retrieval
//! quality.
//!
//! Labeled, not self-referential. Ground truth here is "the session a human
//! wrote the answer into", so the numbers mean what their names say; a gate
//! that scored the engine against its own output could only prove it is
//! consistent, never that it is right.
//!
//! The floors are deliberately below the measured values. This is a
//! regression alarm, not a tuning signal: a corpus this size cannot resolve
//! a few points of difference, and tuning constants against it would fit the
//! fixture instead of the problem. Tuning happens against a larger labeled
//! corpus offline; what lands here is the floor the shipped behavior must
//! keep clearing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::recall;
use crate::sessions;
use crate::state::Core;
use crate::types::ChatMessage;

const HOUR: u64 = 3_600;

/// Where the answer lives.
enum Gold {
    /// A seeded session, by label.
    Session(&'static str),
    /// The project memory directory.
    Memory,
}

struct Case {
    class: &'static str,
    query: &'static str,
    gold: Gold,
    /// Text the output must actually surface. More than one means the answer
    /// is split across records and every part has to arrive.
    needles: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case {
        class: "rare-token",
        query: "ERR_STREAM_PREMATURE_CLOSE",
        gold: Gold::Session("error"),
        needles: &["req_id=8f2c1a"],
    },
    Case {
        class: "truncated-id",
        query: "ERR_STREAM_PREMATURE",
        gold: Gold::Session("error"),
        needles: &["req_id=8f2c1a"],
    },
    Case {
        class: "natural-q",
        query: "why did we abandon proxy pool pre-warming",
        gold: Gold::Session("wrong-turn"),
        needles: &["0.40% to 1.30%"],
    },
    Case {
        class: "morphology",
        query: "abandoning the pool experiment",
        gold: Gold::Session("wrong-turn"),
        needles: &["Abandoning"],
    },
    Case {
        class: "multi-answer",
        query: "which settings changed for the keepalive fix",
        gold: Gold::Session("fix"),
        needles: &["keep_alive_msecs", "reaper_interval", "upstream_timeout", "retry_budget"],
    },
    Case {
        class: "value-lookup",
        query: "what is keep_alive_msecs set to",
        gold: Gold::Session("fix"),
        needles: &["25000"],
    },
    Case {
        class: "path-hop",
        query: "path:config/nginx/upstream.conf timeout",
        gold: Gold::Session("fix"),
        needles: &["upstream_timeout"],
    },
    Case {
        class: "digest",
        query: "compaction note about the keepalive reaper",
        gold: Gold::Session("fix"),
        needles: &["reaper"],
    },
    Case {
        class: "question-echo",
        query: "when does the SOC2 audit close",
        gold: Gold::Session("constraint"),
        needles: &["Q4 2026"],
    },
    Case {
        class: "memory",
        query: "node pinned version",
        gold: Gold::Memory,
        needles: &["18.17.1"],
    },
    Case {
        class: "archive",
        query: "max-http-header-size value",
        gold: Gold::Session("archived"),
        needles: &["32768"],
    },
    Case {
        class: "proximity",
        query: "session vacuum timer",
        gold: Gold::Session("proximity"),
        needles: &["15 minute timer"],
    },
    Case {
        class: "disambig",
        query: "cart batch size",
        gold: Gold::Session("disambig"),
        needles: &["512 records"],
    },
    Case {
        class: "recency",
        query: "session retention policy",
        gold: Gold::Session("current"),
        needles: &["7 days"],
    },
];

/// Floors, set below the measured values so ordinary drift does not fail the
/// build and a real regression does. Update them only alongside a measured
/// improvement, never to make a red build green.
const MIN_HIT_AT_1: f64 = 0.70;
const MIN_HIT_AT_3: f64 = 0.90;
const MIN_MRR: f64 = 0.78;
const MIN_COVERAGE: f64 = 0.85;
const MAX_MEAN_TOKENS: f64 = 900.0;

/// Per-query guardrails. Means hide single-query collapses: at fourteen
/// queries, an answer arriving a quarter complete moves mean coverage by
/// five points, which any tolerance loose enough to survive ordinary drift
/// would swallow. These bound the worst query instead of the average one.
const MIN_QUERY_COVERAGE: f64 = 0.5;
const MAX_QUERY_RANK: usize = 5;

/// Deterministic filler, so "common words are common" the way they are in a
/// real history and idf has to carry the discrimination. A counter, not a
/// random source: the corpus must be byte-identical on every machine.
fn filler_line(seq: usize) -> String {
    const SHAPES: [&str; 4] = [
        "upstream latency {} ms, keepalive window steady",
        "cart reconcile pass complete, {} orphans",
        "payment webhook queue depth {}, draining",
        "session store compaction skipped, {} below threshold",
    ];
    SHAPES[seq % SHAPES.len()].replace("{}", &(seq * 7 % 9_973).to_string())
}

fn seed(
    core: &Core,
    project: &Path,
    labels: &mut HashMap<&'static str, String>,
    label: Option<&'static str>,
    title: &str,
    age_hours: u64,
    messages: Vec<ChatMessage>,
) -> String {
    let meta = sessions::create(core, project.display().to_string()).unwrap();
    sessions::set_title_if_new(core, &meta.id, title);
    let mut persisted = 0usize;
    sessions::save_messages(core, &meta.id, &messages, &mut persisted, false);
    sessions::touch_at(core, &meta.id, sessions::unix_now() - age_hours * HOUR);
    if let Some(label) = label {
        labels.insert(label, meta.id.clone());
    }
    meta.id
}

fn build(core: &Core, project: &Path) -> HashMap<&'static str, String> {
    let mut labels = HashMap::new();

    // A rare error token buried deep in a long tool output: findable only
    // because long documents are scored as pages.
    let mut log = String::new();
    for i in 0..300 {
        log.push_str(&filler_line(i));
        log.push('\n');
        if i == 150 {
            log.push_str(
                "ERR_STREAM_PREMATURE_CLOSE: gateway closed connection while writing \
                 body (req_id=8f2c1a, target=upstream:8443)\n",
            );
        }
    }
    seed(core, project, &mut labels, Some("error"), "gateway 502 investigation", 140, vec![
        ChatMessage::user("customers report intermittent 502s on checkout, dig in"),
        ChatMessage::assistant(Some("pulling the gateway logs for the window".into()), None),
        ChatMessage::tool("t1", log),
        ChatMessage::assistant(Some("found it: the stream closes mid-body".into()), None),
    ]);

    // Why-not knowledge: the approach that was tried and abandoned.
    seed(core, project, &mut labels, Some("wrong-turn"), "pool pre-warming", 116, vec![
        ChatMessage::user("could pre-warming the proxy pool fix the 502s?"),
        ChatMessage::assistant(
            Some(
                "tried pre-warming in src/gateway/pool.ts; error rate went from 0.40% to \
                 1.30% because idle sockets outlive the upstream 30s timeout. Abandoning \
                 proxy pool pre-warming; reverted the file."
                    .into(),
            ),
            None,
        ),
    ]);

    // One answer split across four records, each carrying a different part.
    let fix = seed(core, project, &mut labels, Some("fix"), "keepalive fix landed", 92, vec![
        ChatMessage::user("so what actually changed to fix the 502s?"),
        ChatMessage::assistant(
            Some("first change: src/gateway/proxy.rs sets keep_alive_msecs to 25000".into()),
            None,
        ),
        ChatMessage::assistant(
            Some("second change: src/cart/basket.ts adds the reaper_interval loop".into()),
            None,
        ),
        ChatMessage::assistant(
            Some("third change: config/nginx/upstream.conf raises upstream_timeout to 45s".into()),
            None,
        ),
        ChatMessage::assistant(
            Some("fourth change: src/payment/charge.rs drops retry_budget to 2".into()),
            None,
        ),
    ]);
    sessions::append_compaction(core, &fix, &sessions::CompactionRecord {
        ts: sessions::unix_now() - 92 * HOUR,
        message_count: 6,
        tools: vec!["edit_file".into()],
        paths: vec![
            "src/gateway/proxy.rs".into(),
            "src/cart/basket.ts".into(),
            "config/nginx/upstream.conf".into(),
            "src/payment/charge.rs".into(),
        ],
        user_snippets: vec!["fix the 502s".into()],
        digest: "[context note: keepalive fix across proxy, reaper, upstream]".into(),
    });

    // A constraint, and a later session that asks the same question without
    // answering it: the echo competes on every word the question contains.
    seed(core, project, &mut labels, Some("constraint"), "runtime constraint", 70, vec![
        ChatMessage::user(
            "hard constraint: node stays pinned to 18.17.1 until the SOC2 audit closes in \
             Q4 2026. no runtime bumps, no native deps.",
        ),
        ChatMessage::assistant(Some("acknowledged, recorded the constraint".into()), None),
    ]);
    seed(core, project, &mut labels, None, "planning questions", 30, vec![
        ChatMessage::user(
            "what constraint did the user set on the node runtime, and when does the SOC2 \
             audit close? need this for planning",
        ),
        ChatMessage::assistant(Some("I don't have that in front of me".into()), None),
    ]);

    // An archived message: pruned from the live transcript, still findable.
    let archived = seed(core, project, &mut labels, Some("archived"), "header size", 44, vec![
        ChatMessage::user("what did we set max-http-header-size to?"),
        ChatMessage::assistant(Some("checking history".into()), None),
    ]);
    sessions::append_archive(core, &archived, &[ChatMessage::tool(
        "t2",
        "node started with --max-http-header-size=32768 after the oversized cookie incident",
    )]);

    // Two words that are common apart and adjacent only in the answer.
    seed(core, project, &mut labels, Some("proximity"), "session vacuum decision", 60, vec![
        ChatMessage::user("do we run session vacuum on every write?"),
        ChatMessage::assistant(
            Some(
                "no: session vacuum runs on a 15 minute timer, never inline with a write, \
                 because inline session vacuum doubled p99 latency in the trial."
                    .into(),
            ),
            None,
        ),
    ]);
    for i in 0..3 {
        seed(core, project, &mut labels, None, "session notes", 20 + i * 5, vec![
            ChatMessage::assistant(
                Some(format!(
                    "the session is fine. {} separately, the vacuum job is untouched here.",
                    filler_line(i as usize + 40)
                )),
                None,
            ),
        ]);
    }

    // A common term disambiguated by one rare qualifier.
    seed(core, project, &mut labels, Some("disambig"), "cart tuning", 50, vec![
        ChatMessage::user("what did we settle on for the cart batch size?"),
        ChatMessage::assistant(
            Some(
                "the cart batch size is 512 records, chosen after the throughput sweep; \
                 smaller batches starved the writer."
                    .into(),
            ),
            None,
        ),
    ]);

    // A decision and the later decision that supersedes it.
    seed(core, project, &mut labels, None, "retention first pass", 200, vec![
        ChatMessage::user("how long do we keep session data?"),
        ChatMessage::assistant(Some("session retention is 30 days for now".into()), None),
    ]);
    seed(core, project, &mut labels, Some("current"), "retention revised", 20, vec![
        ChatMessage::user("legal wants session retention shortened"),
        ChatMessage::assistant(
            Some(
                "updated: session retention is now 7 days, superseding the 30 day policy"
                    .into(),
            ),
            None,
        ),
    ]);

    // The curated fact, competing with everything above.
    let memory_dir = project.join(crate::memory::MEMORY_DIR);
    std::fs::create_dir_all(&memory_dir).unwrap();
    std::fs::write(
        memory_dir.join("runtime-pin.md"),
        "# node pinned to 18.17.1 until SOC2 closes Q4 2026\n\
         No node runtime bumps, no native deps, no base image changes.\n",
    )
    .unwrap();

    // Background traffic: without it every content word is rare and idf
    // discriminates nothing, which would make the gate far easier than life.
    for i in 0..24 {
        let mut bulk = String::new();
        for j in 0..30 {
            bulk.push_str(&filler_line(i * 30 + j));
            bulk.push('\n');
        }
        seed(core, project, &mut labels, None, "routine maintenance", 6 + i as u64 * 2, vec![
            ChatMessage::user(format!("why did the linter complain about the cart module, run {i}")),
            ChatMessage::assistant(
                Some(format!(
                    "the usual: unused import in src/gateway/pool.ts, fixed. keepalive and \
                     upstream settings untouched, run {i}"
                )),
                None,
            ),
            ChatMessage::tool("t", bulk),
        ]);
    }

    labels
}

struct Row {
    class: &'static str,
    query: &'static str,
    rank: usize,
    coverage: f64,
    /// How many needles the answer was made of, for the failure message.
    needles: usize,
    tokens: usize,
}

/// One scored run of the whole corpus, plus the labels it resolved, so a
/// broken fixture and a ranking regression can be told apart.
struct Measured {
    rows: Vec<Row>,
    /// Cases whose gold session label did not resolve to a seeded session.
    /// Non-empty means the CORPUS is broken, not the ranker: `is_gold` would
    /// answer false for every hit, every rank would read as 0, and the gate
    /// would report a ranking collapse that never happened.
    unresolved: Vec<&'static str>,
    hit1: f64,
    hit3: f64,
    mrr: f64,
    coverage: f64,
    tokens: f64,
}

fn measure() -> Measured {
    let dir = std::env::temp_dir().join(format!("openmax-recall-quality-{}", uuid::Uuid::new_v4()));
    let project: PathBuf = dir.join("project");
    std::fs::create_dir_all(&project).unwrap();
    let (core, _rx) = Core::new(dir.join("data")).unwrap();
    let labels = build(&core, &project);

    let mut rows = Vec::new();
    let mut unresolved = Vec::new();
    for case in CASES {
        let report = recall(&core, &project, case.query).unwrap();
        let gold = match case.gold {
            Gold::Session(label) => {
                let id = labels.get(label).cloned();
                if id.is_none() {
                    unresolved.push(case.query);
                }
                id
            }
            Gold::Memory => None,
        };
        let is_gold = |hit: &super::RecallHit| match (&case.gold, &gold) {
            (Gold::Memory, _) => hit.kind == "memory",
            (Gold::Session(_), Some(id)) => hit.session.as_deref() == Some(id.as_str()),
            (Gold::Session(_), None) => false,
        };
        let rank = report.hits.iter().position(is_gold).map(|i| i + 1).unwrap_or(0);
        let shown: String = report
            .hits
            .iter()
            .filter(|h| is_gold(h))
            .map(|h| h.excerpt.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        let found = case.needles.iter().filter(|n| shown.contains(&n.to_lowercase())).count();
        rows.push(Row {
            class: case.class,
            query: case.query,
            rank,
            coverage: found as f64 / case.needles.len() as f64,
            needles: case.needles.len(),
            tokens: report
                .hits
                .iter()
                .map(|h| crate::types::estimate_tokens(h.excerpt.len() + h.source.len() + 48))
                .sum(),
        });
    }
    let _ = std::fs::remove_dir_all(&dir);

    let n = rows.len() as f64;
    let hit1 = rows.iter().filter(|r| r.rank == 1).count() as f64 / n;
    let hit3 = rows.iter().filter(|r| (1..=3).contains(&r.rank)).count() as f64 / n;
    let mrr = rows.iter().map(|r| if r.rank > 0 { 1.0 / r.rank as f64 } else { 0.0 }).sum::<f64>() / n;
    let coverage = rows.iter().map(|r| r.coverage).sum::<f64>() / n;
    let tokens = rows.iter().map(|r| r.tokens).sum::<usize>() as f64 / n;

    // Printed so `cargo test -- --nocapture` reports the numbers rather than
    // only whether they cleared the bar.
    println!(
        "\nrecall quality: {} queries  hit@1 {hit1:.3}  hit@3 {hit3:.3}  MRR {mrr:.3}  \
         coverage {coverage:.3}  mean tokens {tokens:.0}",
        rows.len()
    );
    for row in &rows {
        let rank = if row.rank == 0 { "-".to_string() } else { row.rank.to_string() };
        println!("  rank {rank:>2}  cover {:.2}  {:14} {}", row.coverage, row.class, row.query);
    }

    Measured { rows, unresolved, hit1, hit3, mrr, coverage, tokens }
}

/// Checked first and on its own, because every other number here is computed
/// against these labels. A corpus that failed to seed makes every query look
/// unranked, which reads exactly like the ranker collapsing - so it has to be
/// a separate, differently worded failure or the next person debugs the wrong
/// thing.
#[test]
fn recall_quality_fixture_resolves_every_gold() {
    let m = measure();
    assert!(
        m.unresolved.is_empty(),
        "the corpus is broken, not the ranking: {} case(s) name a gold session that was never \
         seeded: {:?}",
        m.unresolved.len(),
        m.unresolved
    );
}

/// The aggregate floors: how the ranker does across the whole labeled set.
#[test]
fn recall_quality_gate() {
    let m = measure();
    assert!(m.unresolved.is_empty(), "fixture is broken; see the fixture test");
    let mut failures = Vec::new();
    if m.hit1 < MIN_HIT_AT_1 {
        failures.push(format!("hit@1 {:.3} < {MIN_HIT_AT_1:.2}", m.hit1));
    }
    if m.hit3 < MIN_HIT_AT_3 {
        failures.push(format!("hit@3 {:.3} < {MIN_HIT_AT_3:.2}", m.hit3));
    }
    if m.mrr < MIN_MRR {
        failures.push(format!("MRR {:.3} < {MIN_MRR:.2}", m.mrr));
    }
    if m.coverage < MIN_COVERAGE {
        failures.push(format!("coverage {:.3} < {MIN_COVERAGE:.2}", m.coverage));
    }
    if m.tokens > MAX_MEAN_TOKENS {
        failures.push(format!("mean tokens {:.0} > {MAX_MEAN_TOKENS:.0}", m.tokens));
    }
    assert!(
        failures.is_empty(),
        "recall quality regressed in aggregate: {}\nper-query detail above (cargo test -- --nocapture)",
        failures.join("; ")
    );
}

/// The per-query floors, separate from the aggregate: an average can stay
/// healthy while one class falls off entirely, and that is the failure a
/// labeled set exists to catch.
#[test]
fn recall_quality_per_query_floors() {
    let m = measure();
    assert!(m.unresolved.is_empty(), "fixture is broken; see the fixture test");
    let mut failures = Vec::new();
    for row in &m.rows {
        if row.rank == 0 || row.rank > MAX_QUERY_RANK {
            let rank = if row.rank == 0 { "unranked".to_string() } else { row.rank.to_string() };
            failures.push(format!("[{}] \"{}\" rank {rank}", row.class, row.query));
        } else if row.coverage < MIN_QUERY_COVERAGE {
            failures.push(format!(
                "[{}] \"{}\" answered {:.0}% ({} of {} needles)",
                row.class,
                row.query,
                row.coverage * 100.0,
                (row.coverage * row.needles as f64).round(),
                row.needles
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "individual queries regressed: {}\nper-query detail above (cargo test -- --nocapture)",
        failures.join("; ")
    );
}
