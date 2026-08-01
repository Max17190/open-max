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
//! Scoring is BM25 lexical relevance (k1 = 1.2, b = 0.75) with the memory
//! index's ACT-R recency law (`age_hours^-0.5`) as a damped tiebreaker:
//! relevance dominates at any age, age only reorders near-equals. Because
//! BM25 only needs statistics for the query's own terms, one pass over the
//! corpus suffices, with per-chunk term counts and the chunk texts held for
//! excerpting - peak memory tracks the scan ceiling, never the history size.
//! Everything is bounded by construction and every bound reports what it
//! cut: matches past k:/budget:, sessions past the scan cap (newest first),
//! and index entries whose files are gone. A session index that exists but
//! cannot be parsed is a loud error - "nothing matched" over unread history
//! is the one lie a memory tool can never afford.

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
/// Length normalization, tuned for a paged corpus: with long documents split
/// into pages, length variance is bounded and the classic 0.75 over-rewards
/// ten-token replies against needle-bearing pages. Measured on the labeled
/// benchmark: 0.75 let a two-term summary outrank a page matching all four
/// query terms.
const BM25_B: f64 = 0.4;

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
    /// Candidates that matched the query, before k and the token budget.
    pub matched: usize,
    /// Matches k or the budget left unprinted. Never silent: a consumer that
    /// trusts "this is everything" must be able to see that it is not.
    pub truncated: usize,
    pub sessions_scanned: usize,
    /// Sessions beyond the scan-byte ceiling, oldest first. Never silent.
    pub sessions_skipped: usize,
    /// Index entries whose files are gone: listed history that cannot be
    /// read is reported, not counted as scanned.
    pub sessions_unreadable: usize,
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
    /// Session-id prefixes: "show me more from the session you just cited".
    session_filters: Vec<String>,
    k: usize,
    budget_tokens: usize,
    excerpt_chars: usize,
}

/// Closed-class words dropped from query terms (never from the corpus). In a
/// project's own history interrogatives concentrate in stored questions, so
/// left in, "what did we set..." retrieves past questions instead of the
/// answers beside them - measured on the labeled benchmark as the
/// question-echo failure class. If a query is nothing but stopwords, the
/// original terms are kept so the query still runs.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "is", "it", "of", "on", "or", "our", "so",
    "that", "the", "then", "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you",
];

fn parse_query(raw: &str) -> Result<Query, String> {
    let mut terms = Vec::new();
    let mut path_filters = Vec::new();
    let mut session_filters = Vec::new();
    let mut k = DEFAULT_K;
    let mut budget = DEFAULT_BUDGET_TOKENS;
    let mut excerpt = EXCERPT_CHARS;
    for word in raw.split_whitespace() {
        if let Some(p) = word.strip_prefix("path:") {
            if !p.is_empty() {
                path_filters.push(p.to_lowercase());
            }
        } else if let Some(s) = word.strip_prefix("session:") {
            if !s.is_empty() {
                session_filters.push(s.to_lowercase());
            }
        } else if let Some(n) = word.strip_prefix("k:") {
            k = n.parse::<usize>().map_err(|_| format!("k: wants a number, got '{n}'"))?;
        } else if let Some(n) = word.strip_prefix("budget:") {
            budget =
                n.parse::<usize>().map_err(|_| format!("budget: wants a number, got '{n}'"))?;
        } else if let Some(n) = word.strip_prefix("excerpt:") {
            excerpt =
                n.parse::<usize>().map_err(|_| format!("excerpt: wants a number, got '{n}'"))?;
        } else {
            for token in tokenize(word) {
                if !terms.contains(&token) {
                    terms.push(token);
                }
            }
        }
    }
    let content_terms: Vec<String> =
        terms.iter().filter(|t| !STOPWORDS.contains(&t.as_str())).cloned().collect();
    if !content_terms.is_empty() {
        terms = content_terms;
    }
    if terms.is_empty() && path_filters.is_empty() && session_filters.is_empty() {
        return Err(
            "recall wants search terms, e.g. openmax --recall \"deploy port path:infra/\""
                .to_string(),
        );
    }
    Ok(Query {
        terms,
        path_filters,
        session_filters,
        k: k.clamp(1, MAX_K),
        budget_tokens: budget.clamp(100, MAX_BUDGET_TOKENS),
        excerpt_chars: excerpt.clamp(120, 2_000),
    })
}

/// Lowercased alphanumeric runs, length 2..=64. One tokenizer for the query
/// and the corpus, or scores drift. The upper cap keeps a base64 blob or a
/// minified bundle from becoming one giant unmatchable token that bloats the
/// term maps; content inside such a run is not lexically findable either way
/// (the cited address is).
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if current.chars().count() < 64 {
                current.extend(ch.to_lowercase());
            }
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

/// Strip one plural 's' (never 'ss'/'us'/'is'): "files"~"file",
/// "sessions"~"session", while "class", "status", "analysis" stay whole.
fn fold_plural(token: &str) -> &str {
    if token.len() > 3
        && token.ends_with('s')
        && !token.ends_with("ss")
        && !token.ends_with("us")
        && !token.ends_with("is")
    {
        &token[..token.len() - 1]
    } else {
        token
    }
}

/// Whether a query term matches a corpus token: exact, plural-folded, or
/// full-prefix containment with at least 5 shared chars. The prefix rule is
/// morphology without linguistics - "abandon"~"abandoned", "modify" reaches
/// "modified" via "modif" - and 5 chars keeps "test"~"testing" honest misses
/// rather than admitting "the"~"theme"-class noise; what noise remains on
/// common stems is idf-damped like any common term.
fn terms_match(query_term: &str, token: &str) -> bool {
    if query_term == token {
        return true;
    }
    let (q, t) = (fold_plural(query_term), fold_plural(token));
    if q == t {
        return true;
    }
    let (short, long) = if q.len() <= t.len() { (q, t) } else { (t, q) };
    short.len() >= 5 && long.starts_with(short)
}

/// Case-insensitive find returning an offset into the ORIGINAL text. The
/// lowercased copy can differ in byte length (Turkish İ gains a byte), so
/// offsets found there are mapped back through a per-byte table built while
/// lowering - centering a window with unmapped offsets drifts it off the
/// match by the accumulated growth.
fn find_ci(text: &str, needle_lower: &str) -> Option<usize> {
    let mut lowered = String::with_capacity(text.len());
    let mut back: Vec<usize> = Vec::with_capacity(text.len() + 8);
    for (orig_idx, ch) in text.char_indices() {
        for lc in ch.to_lowercase() {
            let start = lowered.len();
            lowered.push(lc);
            for _ in start..lowered.len() {
                back.push(orig_idx);
            }
        }
    }
    lowered.find(needle_lower).map(|i| back[i])
}

fn hours_since(now: u64, ts: u64) -> u64 {
    now.saturating_sub(ts) / 3600
}

/// One chunk's scan contribution is capped so a single pathological message
/// or memory file cannot spend the whole ceiling; the tail past the cap is
/// still on disk at the cited address.
const MAX_CHUNK_BYTES: usize = 512 * 1024;
/// Long texts score as fixed-size pages, the retrieval analog of paged
/// memory: BM25's length normalization is right to distrust long documents,
/// but a fact inside a pasted log is exactly what recall exists to find, and
/// as one giant document that log always loses to a short summary that
/// half-matches. Paged, the needle's page is a short document with dense
/// term hits, and it competes on relevance. Overlap keeps a fact that
/// straddles a boundary matchable on one page.
const PAGE_CHARS: usize = 1_200;
const PAGE_OVERLAP: usize = 200;

/// Split text into whitespace-aligned pages of ~PAGE_CHARS with overlap.
/// Short texts return themselves untouched: pagination is for documents the
/// length norm would otherwise bury, not a rewrite of every message.
fn pages(text: &str) -> Vec<&str> {
    if text.len() <= PAGE_CHARS + PAGE_CHARS / 2 {
        return vec![text];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
        let mut end = (start + PAGE_CHARS).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        if end < text.len() {
            // Prefer a whitespace boundary in the last fifth of the page;
            // both slice ends must sit on char boundaries first.
            let mut window_start = end.saturating_sub(PAGE_CHARS / 5).max(start + 1);
            while window_start < end && !text.is_char_boundary(window_start) {
                window_start += 1;
            }
            if window_start < end {
                if let Some(ws) = text[window_start..end].rfind(char::is_whitespace) {
                    end = window_start + ws;
                }
            }
        }
        if start >= end {
            break;
        }
        out.push(&text[start..end]);
        if end >= text.len() {
            break;
        }
        start = end.saturating_sub(PAGE_OVERLAP);
    }
    out
}
/// Memory files scan under their own sub-budget so even a pathological
/// memory directory can never crowd session history out of the ceiling.
const MEMORY_SCAN_BYTES: usize = 4 * 1024 * 1024;

fn bounded(text: String) -> String {
    if text.len() <= MAX_CHUNK_BYTES {
        return text;
    }
    let mut cut = MAX_CHUNK_BYTES;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text[..cut].to_string()
}

/// Read at most `io_budget` bytes of a JSONL file and parse what fits, one
/// value per line, corrupt or truncated lines skipped. Peak memory follows
/// the scan ceiling, not the file: a multi-gigabyte transcript costs at most
/// the budget, and a single line longer than it is dropped, not ballooned -
/// the cited address still holds the full record.
fn bounded_jsonl<T: serde::de::DeserializeOwned>(path: &Path, io_budget: usize) -> Vec<T> {
    use std::io::Read as _;
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut buf = Vec::new();
    if file.take(io_budget as u64).read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    // Lossy, because the budget may cut inside a multi-byte char: the mangled
    // final line fails to parse and is skipped like any other corrupt line.
    String::from_utf8_lossy(&buf)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Read at most `MAX_CHUNK_BYTES` of a text file, lossy at the cut.
fn bounded_text(path: &Path) -> Option<String> {
    use std::io::Read as _;
    let file = std::fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    file.take(MAX_CHUNK_BYTES as u64).read_to_end(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Collect this project's chunks, newest sessions first, stopping at the
/// scan-byte ceiling. The ceiling is enforced before every read - memory
/// files and mid-session alike - so "bounded" means bounded, not "checked
/// once per session after the damage". The project key is the raw
/// `Path::display` form - byte-identical to what session creation stores -
/// because a canonicalized form would silently miss every session on
/// platforms where the two differ (macOS /tmp vs /private/tmp).
struct Collected {
    chunks: Vec<Chunk>,
    scanned: usize,
    skipped: usize,
    unreadable: usize,
    bytes: usize,
}

fn collect_chunks(
    core: &Core,
    project_root: &Path,
    now: u64,
    scan_ceiling: usize,
    session_filters: &[String],
) -> Collected {
    let mut chunks = Vec::new();
    let mut bytes = 0usize;
    let root = project_root.display().to_string();

    // Memory files first: the curated facts, small by contract, and held to
    // their own sub-budget so an oversized memory directory can never evict
    // session history from the scan. A session:-scoped query skips them
    // entirely: "more from this session" means this session.
    let memory_ceiling =
        if session_filters.is_empty() { scan_ceiling.min(MEMORY_SCAN_BYTES) } else { 0 };
    let memory_dir = project_root.join(crate::memory::MEMORY_DIR);
    if let Ok(read_dir) = std::fs::read_dir(&memory_dir) {
        for entry in read_dir.flatten() {
            if bytes >= memory_ceiling {
                break;
            }
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
            if path.extension().and_then(|e| e.to_str()) != Some("md") || name.starts_with('.') {
                continue;
            }
            let Some(text) = bounded_text(&path) else { continue };
            let ts = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(now);
            bytes += text.len();
            for page in pages(&text) {
                chunks.push(Chunk {
                    kind: "memory",
                    session: None,
                    title: None,
                    age_hours: hours_since(now, ts),
                    source: format!("{}/{name}", crate::memory::MEMORY_DIR),
                    text: page.to_string(),
                    paths: Vec::new(),
                });
            }
        }
    }

    // Already filtered to this project and sorted newest first; duplicate
    // index entries collapse so one session cannot scan (and bill the
    // ceiling) more than once.
    let mut metas = sessions::list(core, &root);
    let mut seen_ids = std::collections::HashSet::new();
    metas.retain(|m| seen_ids.insert(m.id.clone()));

    let mut scanned = 0usize;
    let mut skipped = 0usize;
    let mut unreadable = 0usize;
    for meta in &metas {
        if !session_filters.is_empty()
            && !session_filters.iter().any(|f| meta.id.to_lowercase().starts_with(f.as_str()))
        {
            continue;
        }
        if bytes >= scan_ceiling {
            skipped += 1;
            continue;
        }
        // An index entry whose files are all gone is listed history that
        // cannot be read: reported as unreadable, never as scanned, and it
        // emits no ghost citation pointing at a file that does not exist.
        let stores = [
            sessions::messages_display(core, &meta.id),
            sessions::archive_display(core, &meta.id),
            sessions::compaction_display(core, &meta.id),
        ];
        let Some(title_source) = stores.iter().find(|p| Path::new(p).exists()) else {
            unreadable += 1;
            continue;
        };
        scanned += 1;
        let age = hours_since(now, meta.updated_at);
        // The title cites the first store that actually exists: an
        // archive-only session must not hand out a dead transcript address.
        chunks.push(Chunk {
            kind: "title",
            session: Some(meta.id.clone()),
            title: Some(meta.title.clone()),
            age_hours: age,
            source: title_source.clone(),
            text: meta.title.clone(),
            paths: Vec::new(),
        });
        // Digests before bulk: they are tiny, and they carry the structured
        // paths the session-level `path:` hop depends on. Collected last,
        // a transcript that exhausts the ceiling would silently strip a
        // scanned session of its path evidence.
        let io_budget = scan_ceiling.saturating_sub(bytes).saturating_add(MAX_CHUNK_BYTES);
        let compaction_path = std::path::PathBuf::from(sessions::compaction_display(core, &meta.id));
        for record in bounded_jsonl::<sessions::CompactionRecord>(&compaction_path, io_budget) {
            if bytes >= scan_ceiling {
                break;
            }
            let text = bounded(format!("{} {}", record.digest, record.paths.join(" ")));
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
        let io_budget = scan_ceiling.saturating_sub(bytes).saturating_add(MAX_CHUNK_BYTES);
        let messages_path = std::path::PathBuf::from(sessions::messages_display(core, &meta.id));
        for msg in bounded_jsonl::<crate::types::ChatMessage>(&messages_path, io_budget) {
            if bytes >= scan_ceiling {
                break;
            }
            if msg.role == "system" {
                continue;
            }
            let Some(content) = msg.content else { continue };
            if content.trim().is_empty() {
                continue;
            }
            let content = bounded(content);
            bytes += content.len();
            for page in pages(&content) {
                chunks.push(Chunk {
                    kind: "message",
                    session: Some(meta.id.clone()),
                    title: Some(meta.title.clone()),
                    age_hours: age,
                    source: sessions::messages_display(core, &meta.id),
                    text: page.to_string(),
                    paths: Vec::new(),
                });
            }
        }
        let io_budget = scan_ceiling.saturating_sub(bytes).saturating_add(MAX_CHUNK_BYTES);
        let archive_path = std::path::PathBuf::from(sessions::archive_display(core, &meta.id));
        for msg in bounded_jsonl::<crate::types::ChatMessage>(&archive_path, io_budget) {
            if bytes >= scan_ceiling {
                break;
            }
            let Some(content) = msg.content else { continue };
            if content.trim().is_empty() {
                continue;
            }
            let content = bounded(content);
            bytes += content.len();
            for page in pages(&content) {
                chunks.push(Chunk {
                    kind: "archive",
                    session: Some(meta.id.clone()),
                    title: Some(meta.title.clone()),
                    age_hours: age,
                    source: sessions::archive_display(core, &meta.id),
                    text: page.to_string(),
                    paths: Vec::new(),
                });
            }
        }
    }
    Collected { chunks, scanned, skipped, unreadable, bytes }
}

/// Whitespace-collapsed window around the rarest matched term (head of the
/// chunk when nothing matched, which only happens for filter-only hits).
/// Rarest first is load-bearing: every natural query carries "the"/"was",
/// and centering on the leftmost match of ANY term would show the right
/// document's least informative 480 chars - the needle found, then hidden.
fn excerpt_around(text: &str, terms_by_rarity: &[&str], width: usize) -> String {
    let hit = terms_by_rarity.iter().find_map(|t| find_ci(text, t)).unwrap_or(0);
    let start_target = hit.saturating_sub(width / 4);
    let mut start = start_target.min(text.len());
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (start + width).min(text.len());
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
    // A session index that exists but cannot be read must be a loud error:
    // reporting an empty past with a "try different terms" hint is the one
    // lie a memory tool can never afford.
    if let Some(reason) = sessions::index_diagnostic(core) {
        return Err(format!("{reason}; recall cannot see past sessions until it is fixed"));
    }
    let now = sessions::unix_now();
    let collected = collect_chunks(core, project_root, now, MAX_SCAN_BYTES, &query.session_filters);
    let Collected { chunks, scanned, skipped, unreadable, bytes } = collected;

    // `path:` selects sessions, not individual chunks: the transcript around
    // a file touch rarely repeats the literal path, so a session whose
    // structured compaction paths (or any chunk text) match contributes all
    // of its chunks - that is the hop. Session-less chunks (memories) pass on
    // their own text or source. Multiple filters intersect at the session
    // level, each possibly satisfied by a different chunk.
    let path_pass: Vec<bool> = if query.path_filters.is_empty() {
        vec![true; chunks.len()]
    } else {
        let mut matched: Option<std::collections::HashSet<String>> = None;
        for filter in &query.path_filters {
            let mut set = std::collections::HashSet::new();
            for chunk in &chunks {
                if let Some(id) = &chunk.session {
                    // Structured compaction paths and chunk text only: the
                    // store's own addresses all share ".openmax/sessions/
                    // <uuid>.messages.json", so matching them would make
                    // path:json select the entire history.
                    if chunk.paths.iter().any(|p| p.to_lowercase().contains(filter))
                        || chunk.text.to_lowercase().contains(filter)
                    {
                        set.insert(id.clone());
                    }
                }
            }
            matched = Some(match matched {
                None => set,
                Some(prev) => prev.intersection(&set).cloned().collect(),
            });
        }
        let matched = matched.unwrap_or_default();
        chunks
            .iter()
            .map(|chunk| match &chunk.session {
                Some(id) => matched.contains(id),
                None => query.path_filters.iter().all(|f| {
                    chunk.text.to_lowercase().contains(f)
                        || chunk.source.to_lowercase().contains(f)
                }),
            })
            .collect()
    };

    // Pass 1 statistics, query terms only, over the filtered corpus: BM25's
    // N, document frequencies, and average length must all describe the same
    // universe or idf and length normalization skew against each other.
    let mut df: HashMap<&str, usize> = HashMap::new();
    let mut doc_tfs: Vec<Option<(HashMap<usize, usize>, usize)>> = Vec::with_capacity(chunks.len());
    let mut total_len = 0usize;
    let mut candidates = 0usize;
    for (idx, chunk) in chunks.iter().enumerate() {
        if !path_pass[idx] {
            doc_tfs.push(None);
            continue;
        }
        candidates += 1;
        let tokens = tokenize(&chunk.text);
        total_len += tokens.len();
        let mut tf: HashMap<usize, usize> = HashMap::new();
        for token in &tokens {
            if let Some(i) = query.terms.iter().position(|t| terms_match(t, token)) {
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
    let n_docs = candidates.max(1) as f64;
    let avg_len = (total_len as f64 / n_docs).max(1.0);

    // Pass 2: score candidates. BM25 with the standard idf, then equal-weight
    // fusion with ACT-R recency after normalizing lexical scores to [0, 1].
    // Per-term idf, shared by scoring and the coverage factor. Terms nothing
    // in the corpus satisfies (df = 0) discriminate nothing and drop out of
    // the coverage denominator.
    let term_idf: Vec<f64> = query
        .terms
        .iter()
        .map(|t| {
            let term_df = df.get(t.as_str()).copied().unwrap_or(0) as f64;
            if term_df == 0.0 {
                0.0
            } else {
                (1.0 + (n_docs - term_df + 0.5) / (term_df + 0.5)).ln()
            }
        })
        .collect();
    let idf_total: f64 = term_idf.iter().sum();

    let mut raw: Vec<(usize, f64)> = Vec::new();
    for (i, entry) in doc_tfs.iter().enumerate() {
        let Some((tf, len)) = entry else { continue };
        let mut lex = 0.0;
        let mut idf_matched = 0.0;
        for (&term_i, &count) in tf {
            let idf = term_idf[term_i];
            idf_matched += idf;
            let tf_f = count as f64;
            let norm = 1.0 - BM25_B + BM25_B * (*len as f64 / avg_len);
            lex += idf * (tf_f * (BM25_K1 + 1.0)) / (tf_f + BM25_K1 * norm);
        }
        // Idf-weighted coverage: a page matching the query's informative
        // terms must beat a snippet matching two stopwords, and neither
        // stopwords nor unsatisfiable terms may dilute the ratio. Square
        // root so partial coverage degrades gently rather than gating.
        if idf_total > 0.0 {
            lex *= (idf_matched / idf_total).sqrt();
        }
        raw.push((i, lex));
    }
    let max_lex = raw.iter().map(|(_, l)| *l).fold(0.0f64, f64::max);
    // Recency is a damped tiebreaker, not a peer of relevance: at equal
    // weight, dogfooding measured the literal answer losing to a recent
    // two-word note once the answer aged past ~5 hours, because any project
    // touched today buries its own past. At 0.25, a document needs at least
    // three quarters of the best lexical score before age can reorder it.
    const RECENCY_WEIGHT: f64 = 0.25;
    let mut scored: Vec<(f64, usize)> = raw
        .into_iter()
        .map(|(i, lex)| {
            let lex_norm = if max_lex > 0.0 { lex / max_lex } else { 0.0 };
            let recency = (chunks[i].age_hours.max(1) as f64).powf(-0.5);
            (lex_norm + RECENCY_WEIGHT * recency, i)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| chunks[a.1].source.cmp(&chunks[b.1].source))
            .then_with(|| a.1.cmp(&b.1))
    });

    // A bare title is a very short BM25 document that outranks its own
    // session's content while carrying none: a title hit only survives when
    // its session offers nothing better among the candidates.
    let sessions_with_content: std::collections::HashSet<&str> = scored
        .iter()
        .filter(|(_, i)| chunks[*i].kind != "title")
        .filter_map(|(_, i)| chunks[*i].session.as_deref())
        .collect();

    // Excerpts center on the rarest matched term: rarity carries the signal.
    let mut terms_by_rarity: Vec<&str> = query.terms.iter().map(String::as_str).collect();
    terms_by_rarity.sort_by_key(|t| df.get(t).copied().unwrap_or(usize::MAX));

    // Emit under the k and token budgets, deduplicating identical excerpts
    // (a message often exists in both the live transcript and the archive).
    // Dropped matches are counted, never silent.
    let matched = scored.len();
    let mut hits = Vec::new();
    let mut seen_excerpts: Vec<String> = Vec::new();
    let mut per_source: HashMap<String, usize> = HashMap::new();
    let mut spent = 0usize;
    for (score, i) in scored {
        if hits.len() >= query.k {
            break;
        }
        let chunk = &chunks[i];
        if chunk.kind == "title" {
            if let Some(id) = chunk.session.as_deref() {
                if sessions_with_content.contains(id) {
                    continue;
                }
            }
        }
        // Diversify across sources: a long paged document can match on many
        // sibling pages, and letting them all through spends k and the token
        // budget re-showing one file while other sources starve. Two pages
        // per source keeps a log with two distinct relevant regions whole;
        // the rest is one address away. Counted only on emission below.
        if per_source.get(&chunk.source).copied().unwrap_or(0) >= 2 {
            continue;
        }
        let mut excerpt = excerpt_around(&chunk.text, &terms_by_rarity, query.excerpt_chars);
        if seen_excerpts.iter().any(|e| e == &excerpt) {
            continue;
        }
        let mut cost = estimate_tokens(excerpt.len() + chunk.source.len() + 48);
        if spent + cost > query.budget_tokens {
            if !hits.is_empty() {
                break;
            }
            // The first hit is never dropped for cost - a budget that returns
            // nothing answers nothing - but the documented cap still holds:
            // its excerpt shrinks to whatever width the budget leaves.
            let width = (query.budget_tokens * 4).saturating_sub(chunk.source.len() + 192);
            excerpt = excerpt_around(&chunk.text, &terms_by_rarity, width.max(40));
            let mut cut = excerpt.len().min(width.max(40));
            while cut > 0 && !excerpt.is_char_boundary(cut) {
                cut -= 1;
            }
            excerpt.truncate(cut);
            cost = estimate_tokens(excerpt.len() + chunk.source.len() + 48);
        }
        spent += cost;
        seen_excerpts.push(excerpt.clone());
        *per_source.entry(chunk.source.clone()).or_insert(0) += 1;
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
    // Everything matched but not shown, whatever dropped it (k, the token
    // budget, sibling-page collapsing, title suppression, twin excerpts):
    // the report never claims "this is everything" when it is not.
    let truncated = matched.saturating_sub(hits.len());

    Ok(RecallReport {
        query: raw_query.to_string(),
        hits,
        matched,
        truncated,
        sessions_scanned: scanned,
        sessions_skipped: skipped,
        sessions_unreadable: unreadable,
        bytes_scanned: bytes,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// Human rendering: one block per hit, provenance first, numbers in the
/// header line so cost and coverage are never adjectives.
pub fn render(report: &RecallReport) -> String {
    let mut notes = String::new();
    if report.truncated > 0 {
        notes.push_str(&format!(
            ", {} more match{} not shown (raise k:/budget: or read the cited files)",
            report.truncated,
            if report.truncated == 1 { "" } else { "es" }
        ));
    }
    if report.sessions_skipped > 0 {
        notes.push_str(&format!(", {} older skipped past the scan cap", report.sessions_skipped));
    }
    if report.sessions_unreadable > 0 {
        notes.push_str(&format!(
            ", {} listed but unreadable (files missing)",
            report.sessions_unreadable
        ));
    }
    let mut out = format!(
        "recall: {} result{} for \"{}\" ({} session{} scanned{}, {} KB in {} ms)\n",
        report.hits.len(),
        if report.hits.len() == 1 { "" } else { "s" },
        report.query,
        report.sessions_scanned,
        if report.sessions_scanned == 1 { "" } else { "s" },
        notes,
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
        // The hop's whole point: the transcript around a touch rarely repeats
        // the literal path, so the session's messages must ride along with
        // the digest that carried the structured path.
        let hopped = recall(&core, &project, "touched path:nginx").unwrap();
        assert!(
            hopped.hits.iter().any(|h| h.kind == "message"),
            "the hop must surface the touching session's transcript, not only its digest: {:?}",
            hopped.hits
        );
        assert!(hopped.hits.iter().all(|h| h.session.as_deref() == Some(nginx_session.as_str())));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The scan ceiling means what it says: enforced before every read, with
    /// a per-chunk cap and a memory sub-budget, so neither one pathological
    /// message nor an oversized memory directory can spend the allowance or
    /// evict session history.
    #[test]
    fn scan_stays_bounded_under_pathological_inputs() {
        let (core, dir, project) = setup();
        std::fs::create_dir_all(project.join(".openmax/memory")).unwrap();
        std::fs::write(
            project.join(".openmax/memory/huge.md"),
            format!("# huge memory\n{}", "m".repeat(5 * 1024 * 1024)),
        )
        .unwrap();
        seed_session(&core, &project, "real work", vec![ChatMessage::user(
            "the ceiling-needle survives the huge memory",
        )]);
        let c = collect_chunks(&core, &project, sessions::unix_now(), MAX_SCAN_BYTES, &[]);
        let (chunks, scanned, skipped) = (c.chunks, c.scanned, c.skipped);
        let bytes = c.bytes;
        assert_eq!(scanned, 1, "the session must still be scanned");
        assert_eq!(skipped, 0);
        let memory_chunk = chunks.iter().find(|c| c.kind == "memory").unwrap();
        assert!(
            memory_chunk.text.len() <= MAX_CHUNK_BYTES,
            "one chunk is capped, got {}",
            memory_chunk.text.len()
        );
        assert!(bytes <= MEMORY_SCAN_BYTES + MAX_CHUNK_BYTES + 4096, "accounting stays near budget");
        assert!(
            recall(&core, &project, "ceiling-needle").unwrap().hits.iter().any(|h| h.kind == "message"),
            "session content must remain findable next to a pathological memory"
        );

        // A tiny ceiling skips whole sessions, and every session is either
        // scanned or counted skipped - the report never loses one silently.
        seed_session(&core, &project, "second", vec![ChatMessage::user("x".repeat(4_000))]);
        let c = collect_chunks(&core, &project, sessions::unix_now(), 1_000, &[]);
        assert!(c.skipped >= 1, "a 1 KB ceiling must skip sessions, skipped {}", c.skipped);
        assert_eq!(c.scanned + c.skipped, 2, "every session is accounted for");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// One oversized session must not scan without bound: the ceiling holds
    /// within one chunk of tolerance even when a single transcript is far
    /// larger than the whole allowance.
    #[test]
    fn one_oversized_session_cannot_blow_past_the_ceiling() {
        let (core, dir, project) = setup();
        let messages: Vec<ChatMessage> =
            (0..12).map(|i| ChatMessage::user(format!("{i} {}", "y".repeat(300_000)))).collect();
        seed_session(&core, &project, "huge", messages);
        let ceiling = 1024 * 1024;
        let c = collect_chunks(&core, &project, sessions::unix_now(), ceiling, &[]);
        assert_eq!(c.scanned, 1);
        assert!(
            c.bytes <= ceiling + MAX_CHUNK_BYTES,
            "overshoot must stay within one chunk of the ceiling, got {}",
            c.bytes
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// One invalid-UTF-8 line must skip like any other corrupt line, not
    /// discard the transcript around it: the reader is lossy at the byte
    /// level and per-line at the parse level.
    #[test]
    fn invalid_utf8_line_skips_only_itself() {
        let (core, dir, project) = setup();
        let id = seed_session(&core, &project, "utf8", vec![ChatMessage::user("placeholder")]);
        let path = sessions::messages_display(&core, &id);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(br#"{"role":"user","content":"UTF8-BEFORE-NEEDLE ok"}"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(b"{\"role\":\"user\",\"content\":\"bad \xff\xfe bytes\"}\n");
        bytes.extend_from_slice(br#"{"role":"assistant","content":"UTF8-AFTER-NEEDLE ok"}"#);
        bytes.push(b'\n');
        std::fs::write(&path, bytes).unwrap();
        let report = recall(&core, &project, "utf8 needle ok k:20").unwrap();
        let joined: String = report.hits.iter().map(|h| h.excerpt.as_str()).collect();
        assert!(joined.contains("UTF8-BEFORE-NEEDLE"), "{joined}");
        assert!(joined.contains("UTF8-AFTER-NEEDLE"), "the line after the bad bytes survives");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Digests are collected before bulk content: a transcript that exhausts
    /// the ceiling must not strip a scanned session of the structured path
    /// evidence the `path:` hop depends on.
    #[test]
    fn path_evidence_survives_mid_session_exhaustion() {
        let (core, dir, project) = setup();
        let id = seed_session(&core, &project, "big", vec![
            ChatMessage::user("alpha ".repeat(700)),
            ChatMessage::user("bravo ".repeat(700)),
        ]);
        sessions::append_compaction(&core, &id, &sessions::CompactionRecord {
            ts: sessions::unix_now(),
            message_count: 1,
            tools: vec![],
            paths: vec!["infra/nginx.conf".into()],
            user_snippets: vec![],
            digest: "[context note: edited nginx]".into(),
        });
        let c = collect_chunks(&core, &project, sessions::unix_now(), 2_000, &[]);
        assert_eq!(c.scanned, 1);
        assert!(
            c.chunks
                .iter()
                .any(|c| c.kind == "digest" && c.paths.iter().any(|p| p.contains("nginx"))),
            "the digest and its structured path must precede bulk collection"
        );
        assert!(
            !c.chunks.iter().any(|c| c.text.contains("bravo")),
            "the ceiling cut the second message's bulk, not the evidence"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Pages: long texts split at whitespace boundaries with overlap; short
    /// texts pass through untouched; every byte of the original is on some
    /// page (overlap means a boundary-straddling fact is whole on one).
    #[test]
    fn pages_cover_everything_and_respect_boundaries() {
        let short = "a short message";
        assert_eq!(pages(short), vec![short]);
        let long = "word ".repeat(700) + "NEEDLE-AT-END tail " + &"pad ".repeat(40);
        let ps = pages(&long);
        assert!(ps.len() > 2, "long text must page, got {}", ps.len());
        assert!(ps.iter().all(|p| p.len() <= PAGE_CHARS + 8), "pages stay page-sized");
        assert!(ps.iter().any(|p| p.contains("NEEDLE-AT-END")), "no byte is lost");
        // Consecutive pages share their boundary region, so a fact that
        // straddles a cut is whole on at least one page.
        for pair in ps.windows(2) {
            let tail = &pair[0][pair[0].len().saturating_sub(PAGE_OVERLAP / 2)..];
            assert!(
                pair[1].contains(tail.split_whitespace().next().unwrap_or("")),
                "overlap must carry the boundary region forward"
            );
        }
        // Unicode: multibyte content pages without panicking on boundaries.
        let cjk = "修复错误 ".repeat(600);
        assert!(pages(&cjk).len() > 1);
    }

    /// Term matching: exact, plural fold, and >=5-char full-prefix
    /// containment - and the guards that keep it from admitting noise.
    #[test]
    fn terms_match_handles_morphology_without_noise() {
        assert!(terms_match("change", "change"));
        assert!(terms_match("files", "file"));
        assert!(terms_match("session", "sessions"));
        assert!(!terms_match("class", "clas"), "'ss' words never fold");
        assert!(terms_match("abandon", "abandoned"));
        assert!(terms_match("abandoning", "abandon"));
        assert!(terms_match("changed", "change"));
        assert!(
            !terms_match("modify", "modified"),
            "y/i alternation is a documented miss: containment only, no linguistics"
        );
        assert!(!terms_match("the", "theme"), "3-char prefixes are noise");
        assert!(!terms_match("test", "testing"), "4 shared chars is below the floor");
        assert!(!terms_match("content", "context"), "shared prefix is not containment");
    }

    /// The diagnosed D7 case: a fact inside a big pasted log must outrank a
    /// short summary that half-matches, because the needle's page is now a
    /// short document with dense term hits.
    #[test]
    fn a_needle_inside_a_long_log_outranks_a_half_matching_summary() {
        let (core, dir, project) = setup();
        let log = format!(
            "{}ERR_STREAM_PREMATURE_CLOSE: upstream closed while writing (req_id=8f2c1a)\n{}",
            "upstream latency nominal, keepalive steady\n".repeat(180),
            "upstream latency nominal, keepalive steady\n".repeat(180),
        );
        seed_session(&core, &project, "bug hunt", vec![
            ChatMessage::tool("c1", log),
            ChatMessage::assistant(Some("found it: the stream closes mid-body".into()), None),
        ]);
        let report = recall(&core, &project, "ERR_STREAM_PREMATURE_CLOSE").unwrap();
        let first = &report.hits[0];
        assert!(
            first.excerpt.contains("req_id=8f2c1a"),
            "the needle page must rank first and show the fact: {:?}",
            report.hits
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Dogfooding's highest-impact bug: the window centered on the leftmost
    /// match of ANY term, so adding "the" to a query hid the needle inside
    /// the hit that had just been found. It centers on the rarest term, with
    /// offsets valid on the original text even when lowercasing changes byte
    /// lengths (Turkish İ grows a byte; 600 of them drifted the old window
    /// clean off the match).
    #[test]
    fn excerpt_centers_on_the_rarest_term_with_offsets_that_survive_lowercasing() {
        let text = format!("{} RARENEEDLE-7731 is the root cause.", "the usual suspects ".repeat(80));
        let excerpt = excerpt_around(&text, &["rareneedle", "the"], EXCERPT_CHARS);
        assert!(excerpt.contains("RARENEEDLE-7731"), "rarest term wins the window: {excerpt}");
        let common_first = excerpt_around(&text, &["the", "rareneedle"], EXCERPT_CHARS);
        assert!(
            !common_first.contains("RARENEEDLE-7731"),
            "ordering is what fixes it, so the reversed order must reproduce the bug"
        );

        let drifty = format!("{} DRIFTNEEDLE-5150 is the answer. {}", "İ ".repeat(600), "tail ".repeat(60));
        let excerpt = excerpt_around(&drifty, &["driftneedle"], EXCERPT_CHARS);
        assert!(excerpt.contains("DRIFTNEEDLE-5150"), "offsets must map back through lowercasing: {excerpt}");
    }

    /// Dogfooding measured the crossover: at equal weight, a verbatim answer
    /// older than ~5 hours lost to a recent note sharing two words. Damped,
    /// relevance dominates at any age and recency still breaks real ties.
    #[test]
    fn an_old_answer_outranks_recent_weak_matches() {
        let (core, dir, project) = setup();
        let answer = seed_session(&core, &project, "the fix", vec![ChatMessage::user(
            "the checkout retry budget changed from 3 to 6 because upstream sheds load under burst",
        )]);
        seed_session(&core, &project, "today", vec![ChatMessage::user(
            "why did the standup move, and why did the meeting run long",
        )]);
        // Filler makes common words common, as any real history does: idf
        // must carry the discrimination, so "why did the" stops being signal.
        for i in 0..5 {
            seed_session(&core, &project, "filler", vec![ChatMessage::user(format!(
                "why did the linter complain again, note {i}"
            ))]);
        }
        sessions::touch_at(&core, &answer, sessions::unix_now() - 30 * 24 * 3600);
        let report = recall(&core, &project, "why did the checkout retry budget change").unwrap();
        let first = report.hits.iter().find(|h| h.kind == "message").unwrap();
        assert_eq!(
            first.session.as_deref(),
            Some(answer.as_str()),
            "a month-old verbatim answer must beat today's stopword overlap: {:?}",
            report.hits
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Bare titles are short, high-BM25 documents that carry no content:
    /// they may not displace content hits from their own session, and store
    /// addresses are not part of the path: filter surface.
    #[test]
    fn titles_yield_to_content_and_store_addresses_are_not_paths() {
        let (core, dir, project) = setup();
        seed_session(&core, &project, "keepalive documentation update", vec![ChatMessage::user(
            "the keepalive reaper closes idle sockets after 45 seconds",
        )]);
        let report = recall(&core, &project, "keepalive k:3").unwrap();
        assert!(!report.hits.is_empty());
        assert!(
            report.hits.iter().all(|h| h.kind != "title"),
            "a session with content must not spend a slot on its bare title: {:?}",
            report.hits
        );
        for probe in ["path:messages.json", "path:sessions", "path:openmax"] {
            let hits = recall(&core, &project, probe).unwrap().hits;
            assert!(hits.is_empty(), "{probe} matched the store's own address: {hits:?}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// session:<prefix> is the navigation primitive: "more from the session
    /// you just cited", scoped before scanning.
    #[test]
    fn session_prefix_filter_scopes_recall() {
        let (core, dir, project) = setup();
        let a = seed_session(&core, &project, "a", vec![ChatMessage::user("shared fact alpha")]);
        seed_session(&core, &project, "b", vec![ChatMessage::user("shared fact beta")]);
        // A matching memory file must not leak into a session-scoped query:
        // "more from this session" means this session.
        std::fs::create_dir_all(project.join(".openmax/memory")).unwrap();
        std::fs::write(project.join(".openmax/memory/shared.md"), "# shared fact gamma").unwrap();
        let query = format!("shared fact session:{}", &a[..8]);
        let report = recall(&core, &project, &query).unwrap();
        assert!(!report.hits.is_empty());
        assert!(
            report.hits.iter().all(|h| h.session.as_deref() == Some(a.as_str())),
            "memory or foreign-session hits leaked: {:?}",
            report.hits
        );
        assert_eq!(report.sessions_scanned, 1, "filtered sessions are not scanned at all");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// An archive-only session (messages file gone, archive retained) must
    /// cite a file that exists: a dead transcript address defeats the
    /// citation-then-read pattern the feature is built on.
    #[test]
    fn citations_always_point_at_existing_files() {
        let (core, dir, project) = setup();
        let meta = sessions::create(&core, project.display().to_string()).unwrap();
        sessions::set_title_if_new(&core, &meta.id, "archive only");
        sessions::append_archive(&core, &meta.id, &[ChatMessage::user(
            "orphaned-archive-fact retained after transcript loss",
        )]);
        let report = recall(&core, &project, "orphaned-archive-fact retained k:10").unwrap();
        assert!(!report.hits.is_empty());
        for hit in &report.hits {
            assert!(
                std::path::Path::new(&hit.source).exists()
                    || hit.source.starts_with(".openmax/"),
                "citation must resolve: {}",
                hit.source
            );
        }
        assert_eq!(report.sessions_scanned, 1);
        assert_eq!(report.sessions_unreadable, 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Sibling pages of one long source may not crowd out other sources: at
    /// most two hits per source, so a repetitive log cannot spend the whole
    /// result list re-showing itself while a distinct relevant session
    /// starves.
    #[test]
    fn sibling_pages_do_not_crowd_out_other_sources() {
        let (core, dir, project) = setup();
        let log = "shared-needle appears again in this region of the log\n".repeat(400);
        seed_session(&core, &project, "big log", vec![ChatMessage::tool("c1", log)]);
        let other = seed_session(&core, &project, "other evidence", vec![ChatMessage::user(
            "shared-needle confirmed independently in a second session",
        )]);
        let report = recall(&core, &project, "shared-needle k:8").unwrap();
        let from_log =
            report.hits.iter().filter(|h| h.title.as_deref() == Some("big log")).count();
        assert!(from_log <= 2, "at most two sibling pages per source, got {from_log}");
        assert!(
            report.hits.iter().any(|h| h.session.as_deref() == Some(other.as_str())),
            "the second source must appear despite the log's many matching pages: {:?}",
            report.hits
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The documented budget cap holds even for a first hit that would
    /// overflow it: the hit survives, its excerpt shrinks to fit.
    #[test]
    fn a_fat_first_hit_shrinks_into_the_budget() {
        let (core, dir, project) = setup();
        seed_session(&core, &project, "fat", vec![ChatMessage::user(format!(
            "fat-needle {}",
            "context ".repeat(600)
        ))]);
        let report = recall(&core, &project, "fat-needle budget:100").unwrap();
        assert_eq!(report.hits.len(), 1, "the first hit is never dropped for cost");
        let hit = &report.hits[0];
        let spent = estimate_tokens(hit.excerpt.len() + hit.source.len() + 48);
        assert!(spent <= 100, "shrunken hit must fit the documented cap, spent {spent}");
        assert!(hit.excerpt.contains("fat-needle"), "the needle survives the shrink");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The two lies a memory tool must never tell: "nothing matched" over a
    /// corrupt index, and "scanned" for listed sessions whose files are gone.
    #[test]
    fn corrupt_index_errors_and_ghost_sessions_report_unreadable() {
        let (core, dir, project) = setup();
        seed_session(&core, &project, "real", vec![ChatMessage::user("real fact here")]);
        // A create() with no saved messages leaves an index entry with no
        // files: listed history that cannot be read.
        sessions::create(&core, project.display().to_string()).unwrap();
        let report = recall(&core, &project, "real fact").unwrap();
        assert_eq!(report.sessions_scanned, 1);
        assert_eq!(report.sessions_unreadable, 1);
        assert!(
            report.hits.iter().all(|h| h.kind != "title" || h.excerpt.contains("real")),
            "no ghost citation may point at files that do not exist: {:?}",
            report.hits
        );
        let rendered = render(&report);
        assert!(rendered.contains("unreadable"), "{rendered}");

        let index = dir.join("data/sessions/index.json");
        std::fs::write(&index, "{{{ not json").unwrap();
        let err = recall(&core, &project, "real fact").unwrap_err();
        assert!(err.contains("does not parse"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Truncation is counted, rendered, and in the JSON: a consumer that
    /// trusts "this is everything" must be able to see that it is not.
    #[test]
    fn budget_truncation_is_reported_not_silent() {
        let (core, dir, project) = setup();
        let messages: Vec<ChatMessage> =
            (0..30).map(|i| ChatMessage::user(format!("countable fact number {i}"))).collect();
        seed_session(&core, &project, "many", messages);
        let report = recall(&core, &project, "countable fact k:3").unwrap();
        assert!(
            (1..=3).contains(&report.hits.len()),
            "k bounds hits (source collapsing may show fewer): {}",
            report.hits.len()
        );
        assert!(report.matched > 3);
        assert!(report.truncated > 0, "dropped matches must be counted");
        assert_eq!(
            report.truncated,
            report.matched - report.hits.len(),
            "truncated is everything matched but not shown, whatever dropped it"
        );
        assert!(render(&report).contains("more match"), "{}", render(&report));

        let wide = recall(&core, &project, "countable fact k:3 excerpt:1200").unwrap();
        assert!(
            wide.hits[0].excerpt.len() >= report.hits[0].excerpt.len(),
            "excerpt: widens the window"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// BM25's universe must be the filtered corpus: adding sessions the
    /// `path:` filter excludes must not move a candidate's score through
    /// df, N, or average-length drift.
    #[test]
    fn filtered_out_noise_does_not_move_scores() {
        let score_of = |noise: usize| -> f64 {
            let (core, dir, project) = setup();
            let id = seed_session(&core, &project, "target", vec![ChatMessage::user(
                "wire the healthcheck probe",
            )]);
            sessions::append_compaction(&core, &id, &sessions::CompactionRecord {
                ts: sessions::unix_now(),
                message_count: 1,
                tools: vec![],
                paths: vec!["infra/nginx.conf".into()],
                user_snippets: vec![],
                digest: "[context note: probe wiring]".into(),
            });
            for i in 0..noise {
                seed_session(&core, &project, &format!("noise {i}"), vec![ChatMessage::user(
                    "healthcheck probe chatter with no path match",
                )]);
            }
            let report = recall(&core, &project, "healthcheck probe path:nginx").unwrap();
            let hit = report
                .hits
                .iter()
                .find(|h| h.kind == "message")
                .expect("target transcript hit")
                .score;
            let _ = std::fs::remove_dir_all(dir);
            hit
        };
        let clean = score_of(0);
        let noisy = score_of(10);
        assert!(
            (clean - noisy).abs() < 1e-9,
            "excluded noise must not perturb BM25 statistics: {clean} vs {noisy}"
        );
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
