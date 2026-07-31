//! `openmax --recall "<query>"`: deliberate recall over this project's own
//! history - session transcripts, compaction archives, compaction digests,
//! session titles, and memory files - ranked, budgeted, and cited.
//!
//! The archive PR made every prune reversible ("Full dropped messages:
//! <path>"); the memory PR made facts durable. Recall is the searchable form
//! of the same commitment: what the harness preserved must be findable
//! without hand-grepping home-dir JSONL over bash. It is a read-only
//! standalone operation like `--ledger`: no session, no endpoint, no daemon,
//! and no derived index - the stores on disk stay the single source of truth
//! and every scan reads them directly. At harness scale (megabytes, not the
//! hundreds of millions of rows a database engine plans for) a bounded
//! streaming scan is milliseconds; if that ever stops being true, the answer
//! is a rebuildable cache artifact, never a second authority.
//!
//! Scoring fuses two signals, min-max style, one of each kind the literature
//! converged on: BM25 lexical relevance (k1 = 1.2, b = 0.75) and the same
//! ACT-R recency law the memory index uses (`age_hours^-0.5`), added with
//! equal weight the way the Generative Agents memory stream does. Because
//! BM25 only needs statistics for the query's own terms, the whole corpus is
//! scanned in one streaming pass with O(1) per-chunk state: the query is
//! compiled first, the hot loop only counts. Everything is bounded by
//! construction - results, excerpt sizes, total output tokens, and total
//! bytes scanned (newest sessions first, skips reported, never silent).

use std::collections::HashMap;
use std::path::Path;

use crate::sessions;
use crate::state::Core;
use crate::types::estimate_tokens;

/// Ranked results returned by default; `k:<n>` in the query overrides.
const DEFAULT_K: usize = 8;
/// Output token budget (chars/4, the budget estimator); `budget:<n>` overrides.
const DEFAULT_BUDGET_TOKENS: usize = 2_000;
/// Hard ceilings for the query-syntax overrides.
const MAX_K: usize = 50;
const MAX_BUDGET_TOKENS: usize = 20_000;
/// Excerpt window around the first matched term, chars.
const EXCERPT_CHARS: usize = 480;
/// Total bytes of history one recall will scan, newest sessions first.
/// Past this, older sessions are skipped and the report says how many.
const MAX_SCAN_BYTES: usize = 64 * 1024 * 1024;

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

#[derive(Debug, Clone, serde::Serialize)]
pub struct RecallHit {
    pub score: f64,
    /// `memory` | `message` | `archive` | `digest` | `title`
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub age_hours: u64,
    /// The address the agent can read for the full record.
    pub source: String,
    pub excerpt: String,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct RecallReport {
    pub query: String,
    pub hits: Vec<RecallHit>,
    pub sessions_scanned: usize,
    /// Sessions beyond the scan-byte ceiling, oldest first. Never silent.
    pub sessions_skipped: usize,
    pub bytes_scanned: usize,
    pub elapsed_ms: u128,
}

/// One scannable unit: a message, an archived message, a digest record, a
/// session title, or a memory file.
struct Chunk {
    kind: &'static str,
    session: Option<String>,
    title: Option<String>,
    age_hours: u64,
    source: String,
    text: String,
    /// Structured paths (compaction records) for `path:` filtering beyond
    /// plain text match.
    paths: Vec<String>,
}

/// The compiled query: everything is resolved before the scan loop.
struct Query {
    terms: Vec<String>,
    path_filters: Vec<String>,
    k: usize,
    budget_tokens: usize,
}

fn parse_query(raw: &str) -> Result<Query, String> {
    let mut terms = Vec::new();
    let mut path_filters = Vec::new();
    let mut k = DEFAULT_K;
    let mut budget = DEFAULT_BUDGET_TOKENS;
    for word in raw.split_whitespace() {
        if let Some(p) = word.strip_prefix("path:") {
            if !p.is_empty() {
                path_filters.push(p.to_lowercase());
            }
        } else if let Some(n) = word.strip_prefix("k:") {
            k = n.parse::<usize>().map_err(|_| format!("k: wants a number, got '{n}'"))?;
        } else if let Some(n) = word.strip_prefix("budget:") {
            budget =
                n.parse::<usize>().map_err(|_| format!("budget: wants a number, got '{n}'"))?;
        } else {
            for token in tokenize(word) {
                if !terms.contains(&token) {
                    terms.push(token);
                }
            }
        }
    }
    if terms.is_empty() && path_filters.is_empty() {
        return Err(
            "recall wants search terms, e.g. openmax --recall \"deploy port path:infra/\""
                .to_string(),
        );
    }
    Ok(Query {
        terms,
        path_filters,
        k: k.clamp(1, MAX_K),
        budget_tokens: budget.clamp(100, MAX_BUDGET_TOKENS),
    })
}

/// Lowercased alphanumeric runs of length >= 2. One tokenizer for the query
/// and the corpus, or scores drift.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            if current.chars().count() >= 2 {
                out.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.chars().count() >= 2 {
        out.push(current);
    }
    out
}

fn hours_since(now: u64, ts: u64) -> u64 {
    now.saturating_sub(ts) / 3600
}

/// Collect this project's chunks, newest sessions first, stopping at the
/// scan-byte ceiling. The project key is the raw `Path::display` form -
/// byte-identical to what session creation stores - because a canonicalized
/// form would silently miss every session on platforms where the two differ
/// (macOS /tmp vs /private/tmp).
fn collect_chunks(
    core: &Core,
    project_root: &Path,
    now: u64,
) -> (Vec<Chunk>, usize, usize, usize) {
    let mut chunks = Vec::new();
    let mut bytes = 0usize;
    let root = project_root.display().to_string();

    // Memory files first: the curated facts, small by contract.
    let memory_dir = project_root.join(crate::memory::MEMORY_DIR);
    if let Ok(read_dir) = std::fs::read_dir(&memory_dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
            if path.extension().and_then(|e| e.to_str()) != Some("md") || name.starts_with('.') {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let ts = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(now);
            bytes += text.len();
            chunks.push(Chunk {
                kind: "memory",
                session: None,
                title: None,
                age_hours: hours_since(now, ts),
                source: format!("{}/{name}", crate::memory::MEMORY_DIR),
                text,
                paths: Vec::new(),
            });
        }
    }

    // Already filtered to this project and sorted newest first.
    let metas = sessions::list(core, &root);

    let mut scanned = 0usize;
    let mut skipped = 0usize;
    for meta in &metas {
        if bytes >= MAX_SCAN_BYTES {
            skipped += 1;
            continue;
        }
        scanned += 1;
        let age = hours_since(now, meta.updated_at);
        chunks.push(Chunk {
            kind: "title",
            session: Some(meta.id.clone()),
            title: Some(meta.title.clone()),
            age_hours: age,
            source: sessions::messages_display(core, &meta.id),
            text: meta.title.clone(),
            paths: Vec::new(),
        });
        for msg in sessions::load_messages(core, &meta.id).unwrap_or_default() {
            if msg.role == "system" {
                continue;
            }
            let Some(content) = msg.content else { continue };
            if content.trim().is_empty() {
                continue;
            }
            bytes += content.len();
            chunks.push(Chunk {
                kind: "message",
                session: Some(meta.id.clone()),
                title: Some(meta.title.clone()),
                age_hours: age,
                source: sessions::messages_display(core, &meta.id),
                text: content,
                paths: Vec::new(),
            });
        }
        for msg in sessions::load_archive(core, &meta.id) {
            let Some(content) = msg.content else { continue };
            if content.trim().is_empty() {
                continue;
            }
            bytes += content.len();
            chunks.push(Chunk {
                kind: "archive",
                session: Some(meta.id.clone()),
                title: Some(meta.title.clone()),
                age_hours: age,
                source: sessions::archive_display(core, &meta.id),
                text: content,
                paths: Vec::new(),
            });
        }
        for record in sessions::load_compaction(core, &meta.id) {
            let text = format!("{} {}", record.digest, record.paths.join(" "));
            bytes += text.len();
            chunks.push(Chunk {
                kind: "digest",
                session: Some(meta.id.clone()),
                title: Some(meta.title.clone()),
                age_hours: hours_since(now, record.ts),
                source: sessions::compaction_display(core, &meta.id),
                text,
                paths: record.paths,
            });
        }
    }
    (chunks, scanned, skipped, bytes)
}

/// Whitespace-collapsed window around the first matched term (head of the
/// chunk when nothing matched, which only happens for pure `path:` hits).
fn excerpt_around(text: &str, terms: &[String]) -> String {
    let lower = text.to_lowercase();
    let hit = terms.iter().filter_map(|t| lower.find(t.as_str())).min().unwrap_or(0);
    let start_target = hit.saturating_sub(EXCERPT_CHARS / 4);
    let mut start = start_target.min(text.len());
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (start + EXCERPT_CHARS).min(text.len());
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }
    let mut out = String::with_capacity(end - start + 2);
    if start > 0 {
        out.push('…');
    }
    out.push_str(&text[start..end].split_whitespace().collect::<Vec<_>>().join(" "));
    if end < text.len() {
        out.push('…');
    }
    out
}

pub fn recall(core: &Core, project_root: &Path, raw_query: &str) -> Result<RecallReport, String> {
    let started = std::time::Instant::now();
    let query = parse_query(raw_query)?;
    let now = sessions::unix_now();
    let (chunks, scanned, skipped, bytes) = collect_chunks(core, project_root, now);

    // Pass 1 statistics, query terms only: document frequency and lengths.
    let n_docs = chunks.len().max(1) as f64;
    let mut df: HashMap<&str, usize> = HashMap::new();
    let mut doc_tfs: Vec<Option<(HashMap<usize, usize>, usize)>> = Vec::with_capacity(chunks.len());
    let mut total_len = 0usize;
    for chunk in &chunks {
        if !query.path_filters.is_empty() {
            let text_lower = chunk.text.to_lowercase();
            let passes = query.path_filters.iter().all(|f| {
                chunk.paths.iter().any(|p| p.to_lowercase().contains(f))
                    || text_lower.contains(f)
                    || chunk.source.to_lowercase().contains(f)
            });
            if !passes {
                doc_tfs.push(None);
                continue;
            }
        }
        let tokens = tokenize(&chunk.text);
        total_len += tokens.len();
        let mut tf: HashMap<usize, usize> = HashMap::new();
        for token in &tokens {
            if let Some(i) = query.terms.iter().position(|t| t == token) {
                *tf.entry(i).or_insert(0) += 1;
            }
        }
        for &i in tf.keys() {
            *df.entry(query.terms[i].as_str()).or_insert(0) += 1;
        }
        // Pure path: queries (no terms) treat every passing chunk as a hit.
        if tf.is_empty() && !query.terms.is_empty() {
            doc_tfs.push(None);
        } else {
            doc_tfs.push(Some((tf, tokens.len())));
        }
    }
    let avg_len = if chunks.is_empty() { 1.0 } else { (total_len as f64 / n_docs).max(1.0) };

    // Pass 2: score candidates. BM25 with the standard idf, then equal-weight
    // fusion with ACT-R recency after normalizing lexical scores to [0, 1].
    let mut raw: Vec<(usize, f64)> = Vec::new();
    for (i, entry) in doc_tfs.iter().enumerate() {
        let Some((tf, len)) = entry else { continue };
        let mut lex = 0.0;
        for (&term_i, &count) in tf {
            let term_df = *df.get(query.terms[term_i].as_str()).unwrap_or(&1) as f64;
            let idf = (1.0 + (n_docs - term_df + 0.5) / (term_df + 0.5)).ln();
            let tf_f = count as f64;
            let norm = 1.0 - BM25_B + BM25_B * (*len as f64 / avg_len);
            lex += idf * (tf_f * (BM25_K1 + 1.0)) / (tf_f + BM25_K1 * norm);
        }
        raw.push((i, lex));
    }
    let max_lex = raw.iter().map(|(_, l)| *l).fold(0.0f64, f64::max);
    let mut scored: Vec<(f64, usize)> = raw
        .into_iter()
        .map(|(i, lex)| {
            let lex_norm = if max_lex > 0.0 { lex / max_lex } else { 0.0 };
            let recency = (chunks[i].age_hours.max(1) as f64).powf(-0.5);
            (lex_norm + recency, i)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| chunks[a.1].source.cmp(&chunks[b.1].source))
            .then_with(|| a.1.cmp(&b.1))
    });

    // Emit under the k and token budgets, deduplicating identical excerpts
    // (a message often exists in both the live transcript and the archive).
    let mut hits = Vec::new();
    let mut seen_excerpts: Vec<String> = Vec::new();
    let mut spent = 0usize;
    for (score, i) in scored {
        if hits.len() >= query.k {
            break;
        }
        let chunk = &chunks[i];
        let excerpt = excerpt_around(&chunk.text, &query.terms);
        if seen_excerpts.iter().any(|e| e == &excerpt) {
            continue;
        }
        let cost = estimate_tokens(excerpt.len() + chunk.source.len() + 48);
        if spent + cost > query.budget_tokens && !hits.is_empty() {
            break;
        }
        spent += cost;
        seen_excerpts.push(excerpt.clone());
        hits.push(RecallHit {
            score,
            kind: chunk.kind,
            session: chunk.session.clone(),
            title: chunk.title.clone(),
            age_hours: chunk.age_hours,
            source: chunk.source.clone(),
            excerpt,
        });
    }

    Ok(RecallReport {
        query: raw_query.to_string(),
        hits,
        sessions_scanned: scanned,
        sessions_skipped: skipped,
        bytes_scanned: bytes,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// Human rendering: one block per hit, provenance first, numbers in the
/// header line so cost and coverage are never adjectives.
pub fn render(report: &RecallReport) -> String {
    let mut out = format!(
        "recall: {} result{} for \"{}\" ({} session{} scanned{}, {} KB in {} ms)\n",
        report.hits.len(),
        if report.hits.len() == 1 { "" } else { "s" },
        report.query,
        report.sessions_scanned,
        if report.sessions_scanned == 1 { "" } else { "s" },
        if report.sessions_skipped > 0 {
            format!(", {} older skipped past the scan cap", report.sessions_skipped)
        } else {
            String::new()
        },
        report.bytes_scanned / 1024,
        report.elapsed_ms,
    );
    for (i, hit) in report.hits.iter().enumerate() {
        let age = if hit.age_hours < 48 {
            format!("{}h", hit.age_hours)
        } else {
            format!("{}d", hit.age_hours / 24)
        };
        let who = match (&hit.session, &hit.title) {
            (Some(id), Some(title)) => {
                format!("session {} \"{}\"", &id[..8.min(id.len())], title)
            }
            _ => "project memory".to_string(),
        };
        out.push_str(&format!(
            "\n[{}] {} {} ({} ago) — {}\n    {}\n",
            i + 1,
            hit.kind,
            who,
            age,
            hit.source,
            hit.excerpt
        ));
    }
    if report.hits.is_empty() {
        out.push_str("nothing matched; try fewer or different terms, or grep the addresses under ~/.openmax/sessions\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    use std::path::PathBuf;

    fn setup() -> (std::sync::Arc<Core>, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("openmax-recall-{}", uuid::Uuid::new_v4()));
        let project = dir.join("project");
        std::fs::create_dir_all(&project).unwrap();
        let (core, _rx) = Core::new(dir.join("data")).unwrap();
        (core, dir, project)
    }

    fn seed_session(core: &Core, project: &Path, title: &str, messages: Vec<ChatMessage>) -> String {
        let meta = sessions::create(core, project.display().to_string()).unwrap();
        sessions::set_title_if_new(core, &meta.id, title);
        let mut persisted = 0usize;
        sessions::save_messages(core, &meta.id, &messages, &mut persisted, false);
        meta.id
    }

    #[test]
    fn recall_finds_needles_across_every_store_with_provenance() {
        let (core, dir, project) = setup();
        let s1 = seed_session(
            &core,
            &project,
            "fix nginx healthcheck",
            vec![
                ChatMessage::system("system noise that must never match: zebra-needle"),
                ChatMessage::user("the deploy port is 7443, wire the healthcheck"),
                ChatMessage::assistant(Some("done: healthcheck hits /healthz on 7443".into()), None),
            ],
        );
        sessions::append_archive(
            &core,
            &s1,
            &[ChatMessage::tool("c1", "archived-fact: retry budget is 6 attempts")],
        );
        sessions::append_compaction(&core, &s1, &sessions::CompactionRecord {
            ts: sessions::unix_now(),
            message_count: 2,
            tools: vec!["edit_file".into()],
            paths: vec!["infra/nginx.conf".into()],
            user_snippets: vec![],
            digest: "[context note: wired the healthcheck port]".into(),
        });
        std::fs::create_dir_all(project.join(".openmax/memory")).unwrap();
        std::fs::write(
            project.join(".openmax/memory/deploy-port.md"),
            "# The deploy port is 7443\nSet in infra/nginx.conf.",
        )
        .unwrap();

        let report = recall(&core, &project, "deploy port 7443").unwrap();
        let kinds: Vec<&str> = report.hits.iter().map(|h| h.kind).collect();
        assert!(kinds.contains(&"memory"), "{kinds:?}");
        assert!(kinds.contains(&"message"), "{kinds:?}");
        assert_eq!(report.sessions_scanned, 1);
        let memory_hit = report.hits.iter().find(|h| h.kind == "memory").unwrap();
        assert!(memory_hit.source.ends_with(".openmax/memory/deploy-port.md"));
        let msg_hit = report.hits.iter().find(|h| h.kind == "message").unwrap();
        assert_eq!(msg_hit.session.as_deref(), Some(s1.as_str()));
        assert!(msg_hit.source.contains(&format!("{s1}.messages.json")));

        let archived = recall(&core, &project, "retry budget attempts").unwrap();
        assert!(
            archived.hits.iter().any(|h| h.kind == "archive" && h.excerpt.contains("6 attempts")),
            "archive must be searchable: {:?}",
            archived.hits
        );

        let system = recall(&core, &project, "zebra-needle").unwrap();
        assert!(system.hits.is_empty(), "system prompts are frozen noise, never hits");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn recall_is_project_isolated() {
        let (core, dir, project) = setup();
        let other = dir.join("other");
        std::fs::create_dir_all(&other).unwrap();
        seed_session(&core, &other, "other project", vec![ChatMessage::user(
            "leaky-fact belongs elsewhere",
        )]);
        let report = recall(&core, &project, "leaky-fact").unwrap();
        assert!(report.hits.is_empty(), "{:?}", report.hits);
        assert_eq!(report.sessions_scanned, 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn path_filter_is_the_graph_hop() {
        let (core, dir, project) = setup();
        let mut nginx_session = String::new();
        for path in ["infra/nginx.conf", "src/auth.rs"] {
            let id = seed_session(&core, &project, path, vec![ChatMessage::user("touched a file")]);
            sessions::append_compaction(&core, &id, &sessions::CompactionRecord {
                ts: sessions::unix_now(),
                message_count: 1,
                tools: vec![],
                paths: vec![path.into()],
                user_snippets: vec![],
                digest: format!("[context note: edited {path}]"),
            });
            if path.contains("nginx") {
                nginx_session = id;
            }
        }
        let report = recall(&core, &project, "path:nginx").unwrap();
        assert!(!report.hits.is_empty());
        assert!(
            report.hits.iter().all(|h| h.session.as_deref() == Some(nginx_session.as_str())),
            "path: must select only the session that touched the file: {:?}",
            report.hits
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn recency_breaks_lexical_ties_and_query_knobs_clamp() {
        let (core, dir, project) = setup();
        let old = seed_session(&core, &project, "old", vec![ChatMessage::user("tiebreak fact here")]);
        let new = seed_session(&core, &project, "new", vec![ChatMessage::user("tiebreak fact here")]);
        // Backdate the old session a week; identical text, identical BM25.
        sessions::touch_at(&core, &old, sessions::unix_now() - 7 * 24 * 3600);
        let report = recall(&core, &project, "tiebreak fact").unwrap();
        let first = report.hits.iter().find(|h| h.kind == "message").unwrap();
        assert_eq!(first.session.as_deref(), Some(new.as_str()), "recency must rank first");

        let clamped = recall(&core, &project, "tiebreak k:9999 budget:5").unwrap();
        assert!(clamped.hits.len() <= MAX_K);
        assert!(recall(&core, &project, "k:3").is_err(), "terms are required");
        assert!(recall(&core, &project, "k:x tiebreak").is_err(), "bad knob is an error");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn budget_and_k_bound_the_output_and_dedupe_collapses_twins() {
        let (core, dir, project) = setup();
        let mut messages = vec![];
        for i in 0..40 {
            messages.push(ChatMessage::user(format!("needle fact variant {i} {}", "x".repeat(300))));
        }
        let s1 = seed_session(&core, &project, "many", messages);
        // The archive holds a byte-identical copy of one message: one hit.
        sessions::append_archive(&core, &s1, &[ChatMessage::user(format!(
            "needle fact variant 0 {}",
            "x".repeat(300)
        ))]);
        let report = recall(&core, &project, "needle fact k:50 budget:400").unwrap();
        assert!(!report.hits.is_empty());
        let spent: usize =
            report.hits.iter().map(|h| estimate_tokens(h.excerpt.len() + h.source.len() + 48)).sum();
        assert!(spent <= 400, "token budget must bound output, spent {spent}");
        let texts: Vec<&String> = report.hits.iter().map(|h| &h.excerpt).collect();
        let mut deduped = texts.clone();
        deduped.dedup();
        assert_eq!(texts.len(), deduped.len(), "identical excerpts must collapse");
        let _ = std::fs::remove_dir_all(dir);
    }
}
