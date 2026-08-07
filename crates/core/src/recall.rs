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

use std::collections::{HashMap, HashSet};
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

/// How hard partial episode coverage is penalized. The per-page coverage
/// factor already uses a square root; the episode factor is a second, weaker
/// multiplication, so it is deliberately gentler - an episode is context, not
/// a claim about the page.
const EPISODE_COV_P: f64 = 0.5;

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
    /// `memory` | `message` | `archive` | `digest`
    pub kind: &'static str,
    /// Who produced the record: `user`, `assistant` or `tool` for a transcript
    /// or archive hit, `None` for a store that has no speaker (a memory file,
    /// a compaction digest). Without it every hit reads as "message" and a
    /// prompt that restates the question is indistinguishable from the answer
    /// to it - which is the ranking's hardest case and the reader's easiest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub age_hours: u64,
    /// The file holding the full record.
    pub source: String,
    /// 1-based line of `source` for the full record, when the store is a JSONL
    /// log. With it the address is exact and bounded - `sed -n '<line>p'`,
    /// piped through `head -c` for as much as is wanted - instead of a grep
    /// for a guessed phrase that returns however many bytes the record happens
    /// to be.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
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
    /// Sessions the scan ceiling cut inside: entered and partly indexed,
    /// with a tail that was never read. Scanned would overclaim and skipped
    /// would underclaim, so the cut is its own count. Never silent.
    #[serde(skip_serializing_if = "usize_is_zero")]
    pub sessions_partial: usize,
    /// Knobs whose requested value was not honoured. Silently substituting a
    /// number is the failure this whole surface is built against: an agent
    /// cannot tell a policy limit from the shape of its own data.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub clamped: Vec<Clamp>,
    /// Chunks that survived `path:`/`session:` filtering. Zero with filters
    /// present means the filters emptied the corpus, which is a different
    /// failure from terms that matched nothing - and pointing at the terms
    /// sends the reader to fix the half that was working.
    pub candidates: usize,
    pub bytes_scanned: usize,
    pub elapsed_ms: u128,
}

/// One scannable unit: a message, an archived message, a digest record, a
/// session title, or a memory file.
struct Chunk {
    kind: &'static str,
    /// The record's speaker, where the store has one. See `RecallHit::role`.
    role: Option<String>,
    session: Option<String>,
    title: Option<String>,
    age_hours: u64,
    source: String,
    /// 1-based line of `source` holding this record, where the store is a
    /// JSONL log. `None` where the file is the record (a memory file) or the
    /// text does not live in the file at all (a session title).
    line: Option<usize>,
    /// The record this chunk came from, as source plus record ordinal. Every
    /// message in a session shares one `source` (the transcript file), so
    /// sibling-page collapsing keyed on `source` cannot tell "page 2 of this
    /// log" from "a different message that answers a different part of the
    /// question". `doc` makes that distinction: pages of one record collapse,
    /// distinct records do not.
    doc: String,
    text: String,
    /// Structured path evidence for `path:` filtering beyond plain text
    /// match: compaction record paths, or a memory file's own stem.
    paths: Vec<String>,
}

/// serde gate for the counters that appear only when they have something
/// to disclose.
fn usize_is_zero(n: &usize) -> bool {
    *n == 0
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
    /// Every knob whose request could not be honoured, so the report can say
    /// so rather than quietly substituting its own number.
    clamped: Vec<Clamp>,
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

/// One knob whose request was not honoured. Both directions matter: a value
/// raised to a floor is as silently substituted as one cut to a ceiling, and
/// an agent that asked for `excerpt:0` and got 120 has been answered by a
/// number it never chose.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Clamp {
    pub knob: &'static str,
    pub requested: usize,
    pub applied: usize,
}

fn clamp_reported(
    knob: &'static str,
    requested: usize,
    low: usize,
    high: usize,
    out: &mut Vec<Clamp>,
) -> usize {
    let applied = requested.clamp(low, high);
    if applied != requested {
        out.push(Clamp { knob, requested, applied });
    }
    applied
}

fn parse_query(raw: &str) -> Result<Query, String> {
    let mut clamped: Vec<Clamp> = Vec::new();
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
        k: clamp_reported("k", k, 1, MAX_K, &mut clamped),
        budget_tokens: clamp_reported("budget", budget, 100, MAX_BUDGET_TOKENS, &mut clamped),
        excerpt_chars: clamp_reported("excerpt", excerpt, 120, PAGE_CHARS, &mut clamped),
        clamped,
    })
}

/// The camel-case parts of one alphanumeric run, lowercased, longest-first
/// order irrelevant to scoring. Two boundaries, the standard pair: a capital
/// after a lowercase or digit (`streamingMarkdown`), and the last capital of
/// a run of them when a lowercase follows (`HTTPServer` -> `http`, `server`).
/// A run with no boundary returns nothing, so ordinary prose costs nothing.
fn camel_parts(run: &[char]) -> Vec<String> {
    let mut bounds = Vec::new();
    for i in 1..run.len() {
        let (prev, cur) = (run[i - 1], run[i]);
        let acronym_end = prev.is_uppercase()
            && cur.is_uppercase()
            && run.get(i + 1).is_some_and(|n| n.is_lowercase());
        if (cur.is_uppercase() && (prev.is_lowercase() || prev.is_numeric())) || acronym_end {
            bounds.push(i);
        }
    }
    if bounds.is_empty() {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut start = 0usize;
    for &end in bounds.iter().chain(std::iter::once(&run.len())) {
        if end - start >= 2 {
            parts.push(run[start..end].iter().collect::<String>().to_lowercase());
        }
        start = end;
    }
    parts
}

/// Lowercased alphanumeric runs, length 2..=64, plus the camel-case parts of
/// any run that has them. One tokenizer for the query and the corpus, or
/// scores drift.
///
/// Identifiers are the vocabulary of a coding agent's history, and a compound
/// one is a single alphanumeric run: `StreamingMarkdown` indexed whole meant a
/// search for "streaming markdown" matched half of it - "streaming" through
/// the prefix rule, "markdown" not at all - and lost to any page that happened
/// to write `markdown::render`, where a separator had done the splitting.
/// Measured on real transcripts, that cost the session actually holding the
/// answer its first-place rank. Underscore, dot and slash forms already split,
/// because those separators are not alphanumeric; case is the one boundary
/// that was invisible.
///
/// The upper cap keeps a base64 blob or a minified bundle from becoming one
/// giant unmatchable token that bloats the term maps; content inside such a
/// run is not lexically findable either way (the cited address is).
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    tokenize_runs(text, |_, _, token| out.push(token));
    out
}

/// The same tokenization, reporting each token with the byte range of the run
/// it came from. Sharing one implementation is not tidiness: a second
/// tokenizer that drifted from this one would silently score windows by
/// different rules than the index they are compared against.
fn tokenize_runs(text: &str, mut emit: impl FnMut(usize, usize, String)) {
    let mut run: Vec<char> = Vec::new();
    let mut start: Option<usize> = None;
    let mut end = 0usize;
    for (i, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
            // The cap bounds the token, never the run's extent: a 4 KB base64
            // blob still occupies the bytes it occupies.
            if run.len() < 64 {
                run.push(ch);
            }
            end = i + ch.len_utf8();
        } else if let Some(s) = start.take() {
            emit_run(s, end, &mut run, &mut emit);
        }
    }
    if let Some(s) = start {
        emit_run(s, end, &mut run, &mut emit);
    }
}

fn emit_run(
    start: usize,
    end: usize,
    run: &mut Vec<char>,
    emit: &mut impl FnMut(usize, usize, String),
) {
    if run.len() >= 2 {
        // The compound is kept only when no part can already reach it.
        // Emitting both unconditionally double-counts, and does so
        // asymmetrically: the prefix rule reaches `streamingmarkdown` from
        // "streaming" but never from "markdown", so one occurrence would
        // score tf=2 for the first part and tf=1 for the rest. Dropping it
        // unconditionally is the opposite failure - the prefix rule needs
        // five shared characters, so `ToString` and `IntoIterator` split into
        // parts too short to lead back, and an exact lowercase search for the
        // identifier would find nothing. Keeping it exactly when it is
        // otherwise unreachable satisfies both: every surface form of one
        // occurrence still counts once.
        let whole = run.iter().collect::<String>().to_lowercase();
        let parts = camel_parts(run);
        let reachable =
            parts.iter().any(|p| p.chars().count() >= 5 && whole.starts_with(p.as_str()));
        if parts.is_empty() || !reachable {
            emit(start, end, whole);
        }
        for part in parts {
            emit(start, end, part);
        }
    }
    run.clear();
}

/// The `width`-byte window of `text` that mentions the most distinct terms
/// from `context`, aligned to line starts. Returns a byte range.
///
/// This is the working-set principle applied to eviction: when only part of
/// a page can stay resident, keep the part the current working set refers to.
/// Budget enforcement kept the first bytes of an old tool output, which is a
/// positional guess - the head of a file read is its imports, the head of a
/// grep is whichever match sorted first, and neither is what the conversation
/// is about.
///
/// Distinct-term COUNT, deliberately, not the idf-weighted sum used for
/// ranking. Measured on 132 real truncations, idf weighting scored 0.158
/// against a plain count's 0.180 (paired, 95% CI [-0.032, -0.013]): rarity
/// inside one tool output selects for hashes, offsets and line numbers, which
/// are exactly the tokens nothing refers to again. Breadth of overlap with
/// the live conversation is the better signal, and it needs no corpus
/// statistics at all.
///
/// Bounded by construction: candidate starts are line starts, strided so no
/// single output can cost more than `MAX_WINDOW_PROBES` scored windows.
pub(crate) fn salient_window(text: &str, context: &str, width: usize) -> std::ops::Range<usize> {
    if text.len() <= width {
        return 0..text.len();
    }
    let terms: std::collections::HashSet<String> = tokenize(context).into_iter().collect();
    if terms.is_empty() {
        return 0..end_boundary(text, width);
    }
    // One tokenizing pass, keeping only the runs that can score at all. Every
    // candidate window is then a range over this list, so the whole scan is
    // two monotone pointers rather than a re-tokenization per candidate: the
    // windows overlap almost entirely, and tokenizing the same bytes once per
    // line start is quadratic work for a linear answer.
    let mut hits: Vec<(usize, usize, String)> = Vec::new();
    tokenize_runs(text, |start, end, token| {
        if terms.contains(&token) {
            hits.push((start, end, token));
        }
    });
    if hits.is_empty() {
        return 0..end_boundary(text, width);
    }
    // A window never begins mid-line: tool output is line-structured and a
    // window cut inside one reads as garbage.
    let starts = std::iter::once(0)
        .chain(text.match_indices('\n').map(|(i, _)| i + 1))
        .filter(|s| *s < text.len());
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut distinct = 0usize;
    let (mut lo, mut hi) = (0usize, 0usize);
    let (mut best, mut best_score) = (0usize, 0usize);
    for start in starts {
        let end = start + width;
        // A run scores only where it fits whole; a name cut in half is not
        // the name, and counting it would reward windows that clip it.
        while hi < hits.len() && hits[hi].1 <= end {
            let entry = counts.entry(hits[hi].2.as_str()).or_insert(0);
            *entry += 1;
            if *entry == 1 {
                distinct += 1;
            }
            hi += 1;
        }
        while lo < hi && hits[lo].0 < start {
            if let Some(entry) = counts.get_mut(hits[lo].2.as_str()) {
                *entry -= 1;
                if *entry == 0 {
                    distinct -= 1;
                }
            }
            lo += 1;
        }
        // Ties go to the earlier window: with nothing to separate them, the
        // head is the one a reader can orient in.
        if distinct > best_score {
            best = start;
            best_score = distinct;
        }
    }
    best..end_boundary(text, best + width)
}

/// Largest char boundary at or below `at`, clamped to the string.
fn end_boundary(text: &str, at: usize) -> usize {
    let mut end = at.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    end
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

/// The episode a chunk belongs to: its session, or - for a memory file, which
/// has no session - its own address. Pages and records of one episode share
/// this key; it is what document-level evidence is measured over.
fn episode_key(chunk: &Chunk) -> &str {
    chunk.session.as_deref().unwrap_or(chunk.source.as_str())
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
/// Returns each parsed value with the 1-based line it came from. Enumerated
/// before the filters, not after: a blank line or one corrupt record would
/// otherwise shift every number below it, and an address that is usually
/// right is worse than no address at all.
fn bounded_jsonl<T: serde::de::DeserializeOwned>(
    path: &Path,
    io_budget: usize,
) -> Vec<(usize, T)> {
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
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .filter_map(|(i, l)| serde_json::from_str(l).ok().map(|value| (i + 1, value)))
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
    /// Sessions the ceiling cut inside, indexed only up to the cut.
    partial: usize,
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
            // Absolute, like every other citation: a consumer that keeps an
            // address and resolves it later cannot be asked to also remember
            // which working directory it was relative to.
            let source = path.display().to_string();
            // The stem is the one piece of a memory file's address that names
            // the fact rather than the store: every memory lives under
            // .openmax/memory/ beneath the project root, so `path:` may
            // select on the stem, never on the store directories or their
            // ancestors.
            let stem =
                path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
            for page in pages(&text) {
                chunks.push(Chunk {
                    kind: "memory",
                    role: None,
                    session: None,
                    title: None,
                    age_hours: hours_since(now, ts),
                    // One memory file is one record: its pages are siblings.
                    doc: source.clone(),
                    source: source.clone(),
                    line: None,
                    text: page.to_string(),
                    paths: vec![stem.clone()],
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
    let mut partial = 0usize;
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
        // An index entry whose files are all gone is listed history that
        // cannot be read: reported as unreadable, never as scanned.
        if !stores.iter().any(|p| Path::new(p).exists()) {
            unreadable += 1;
            continue;
        }
        scanned += 1;
        // The ceiling can cut inside a session's stores; a session both
        // entered and cut is disclosed as partial, never claimed whole.
        let mut cut = false;
        let age = hours_since(now, meta.updated_at);
        // No title chunk. A title is the first 48 characters of the first
        // user message (agent.rs sets it there), and budget enforcement never
        // drops that message, so anything a title could match the transcript
        // matches too - with more of it, and with an address that resolves.
        // Where a title hit did fire it cited a file that does not contain
        // the title, because the name lives in the session index. The name
        // still travels on every hit as `title`.
        // Digests before bulk: they are tiny, and they carry the structured
        // paths the session-level `path:` hop depends on. Collected last,
        // a transcript that exhausts the ceiling would silently strip a
        // scanned session of its path evidence.
        let io_budget = scan_ceiling.saturating_sub(bytes).saturating_add(MAX_CHUNK_BYTES);
        let compaction_path = std::path::PathBuf::from(sessions::compaction_display(core, &meta.id));
        for (line, record) in
            bounded_jsonl::<sessions::CompactionRecord>(&compaction_path, io_budget)
        {
            if bytes >= scan_ceiling {
                cut = true;
                break;
            }
            let text = bounded(format!("{} {}", record.digest, record.paths.join(" ")));
            bytes += text.len();
            let source = sessions::compaction_display(core, &meta.id);
            chunks.push(Chunk {
                kind: "digest",
                    role: None,
                session: Some(meta.id.clone()),
                title: Some(meta.title.clone()),
                age_hours: hours_since(now, record.ts),
                doc: format!("{source}#{line}"),
                source,
                line: Some(line),
                text,
                paths: record.paths,
            });
        }
        let io_budget = scan_ceiling.saturating_sub(bytes).saturating_add(MAX_CHUNK_BYTES);
        let messages_path = std::path::PathBuf::from(sessions::messages_display(core, &meta.id));
        for (line, msg) in bounded_jsonl::<crate::types::ChatMessage>(&messages_path, io_budget) {
            if bytes >= scan_ceiling {
                cut = true;
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
            let source = sessions::messages_display(core, &meta.id);
            for page in pages(&content) {
                chunks.push(Chunk {
                    kind: "message",
                    role: Some(msg.role.clone()),
                    session: Some(meta.id.clone()),
                    title: Some(meta.title.clone()),
                    age_hours: age,
                    doc: format!("{source}#{line}"),
                    source: source.clone(),
                    line: Some(line),
                    text: page.to_string(),
                    paths: Vec::new(),
                });
            }
        }
        let io_budget = scan_ceiling.saturating_sub(bytes).saturating_add(MAX_CHUNK_BYTES);
        let archive_path = std::path::PathBuf::from(sessions::archive_display(core, &meta.id));
        for (line, msg) in bounded_jsonl::<crate::types::ChatMessage>(&archive_path, io_budget) {
            if bytes >= scan_ceiling {
                cut = true;
                break;
            }
            let Some(content) = msg.content else { continue };
            if content.trim().is_empty() {
                continue;
            }
            let content = bounded(content);
            bytes += content.len();
            let source = sessions::archive_display(core, &meta.id);
            for page in pages(&content) {
                chunks.push(Chunk {
                    kind: "archive",
                    role: Some(msg.role.clone()),
                    session: Some(meta.id.clone()),
                    title: Some(meta.title.clone()),
                    age_hours: age,
                    doc: format!("{source}#{line}"),
                    source: source.clone(),
                    line: Some(line),
                    text: page.to_string(),
                    paths: Vec::new(),
                });
            }
        }
        if cut {
            partial += 1;
        }
    }
    Collected { chunks, scanned, skipped, partial, unreadable, bytes }
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
    let Collected { chunks, scanned, skipped, partial, unreadable, bytes } = collected;

    // `path:` selects sessions, not individual chunks: the transcript around
    // a file touch rarely repeats the literal path, so a session whose
    // structured compaction paths (or any chunk text) match contributes all
    // of its chunks - that is the hop. Session-less chunks (memories) pass on
    // their own text or stem: the absolute source would admit the store's
    // directories and every ancestor of the project root as path evidence.
    // Multiple filters intersect at the session level, each possibly
    // satisfied by a different chunk.
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
                    chunk.paths.iter().any(|p| p.to_lowercase().contains(f))
                        || chunk.text.to_lowercase().contains(f)
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
    // Which query terms each episode satisfies anywhere in its records. A
    // session is one working episode and a memory file is one curated fact;
    // both answer as a whole, and an answer split across four messages is
    // still one answer.
    let mut episode_terms: HashMap<&str, std::collections::HashSet<usize>> = HashMap::new();
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
        if !tf.is_empty() {
            let episode = episode_key(chunk);
            episode_terms.entry(episode).or_default().extend(tf.keys().copied());
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

    // Episode coverage: the same idf-weighted ratio as the per-page factor,
    // measured over everything the episode said. A page is evidence from a
    // conversation, not a standalone document, and an episode that answers
    // three quarters of the query is a better place to be reading than one
    // that echoes a single common word - even where the individual page
    // carrying part of that answer matches only one term itself. This is
    // what lets a four-message answer surface as four hits instead of one:
    // each part scores weakly alone and is lifted by the company it keeps.
    let episode_cov: HashMap<&str, f64> = episode_terms
        .iter()
        .map(|(episode, matched)| {
            let idf_matched: f64 = matched.iter().map(|&i| term_idf[i]).sum();
            let cov =
                if idf_total > 0.0 { (idf_matched / idf_total).powf(EPISODE_COV_P) } else { 1.0 };
            (*episode, cov)
        })
        .collect();

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
            lex *= episode_cov.get(episode_key(&chunks[i])).copied().unwrap_or(1.0);
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

    // Excerpts center on the rarest matched term: rarity carries the signal.
    let mut terms_by_rarity: Vec<&str> = query.terms.iter().map(String::as_str).collect();
    terms_by_rarity.sort_by_key(|t| df.get(t).copied().unwrap_or(usize::MAX));

    // Emit under the k and token budgets, deduplicating identical excerpts
    // (a message often exists in both the live transcript and the archive).
    // Dropped matches are counted, never silent.
    let matched = scored.len();
    let mut hits = Vec::new();
    let mut seen_excerpts: Vec<String> = Vec::new();
    let mut per_doc: HashSet<String> = HashSet::new();
    let mut spent = 0usize;
    for (score, i) in scored {
        if hits.len() >= query.k {
            break;
        }
        let chunk = &chunks[i];
        // One page per record. A long paged document can match on many of its
        // own pages, and every extra page spends a slot and a share of the
        // token budget to show more of a record the reader is already looking
        // at - while a second, unrelated source gets nothing. This collapses
        // pages, not records: every message in a session shares one `source`,
        // so capping on `source` instead would throw away the second and third
        // message that each answer a different part of the question.
        //
        // A single page is enough because the hit carries the record's address
        // and the whole record is readable from it. Showing a second page
        // spends a slot on what one read already returns.
        if per_doc.contains(&chunk.doc) {
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
        per_doc.insert(chunk.doc.clone());
        hits.push(RecallHit {
            score,
            kind: chunk.kind,
            role: chunk.role.clone(),
            line: chunk.line,
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
        sessions_partial: partial,
        clamped: query.clamped.clone(),
        candidates,
        bytes_scanned: bytes,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// The filter words of a query, for an error that names what excluded the
/// corpus rather than blaming the search terms.
fn filters_in(query: &str) -> String {
    let parts: Vec<&str> = query
        .split_whitespace()
        .filter(|w| w.starts_with("path:") || w.starts_with("session:"))
        .collect();
    parts.join(" ")
}

/// Human rendering: one block per hit, provenance first, numbers in the
/// header line so cost and coverage are never adjectives.
pub fn render(report: &RecallReport) -> String {
    let mut notes = String::new();
    for c in &report.clamped {
        notes.push_str(&format!(
            ", {}:{} answered as {}",
            c.knob, c.requested, c.applied
        ));
        if c.knob == "excerpt" && c.applied < c.requested {
            notes.push_str(" (one page is the largest excerpt; read a hit's address \
                            for the whole record)");
        }
    }
    if report.truncated > 0 {
        notes.push_str(&format!(
            // Raising k/budget shows more *matches*; it never grows one match
            // into its whole record. Saying so stops an agent from raising
            // limits in a loop trying to read one record out of the index.
            ", {} more match{} not shown (raise k:/budget: for more matches; read a \
             hit's address for one whole record)",
            report.truncated,
            if report.truncated == 1 { "" } else { "es" }
        ));
    }
    if report.sessions_skipped > 0 {
        notes.push_str(&format!(", {} older skipped past the scan cap", report.sessions_skipped));
    }
    if report.sessions_partial > 0 {
        notes.push_str(&format!(", {} partly scanned (scan cap)", report.sessions_partial));
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
        // `path:line` is the address convention every editor, grep and agent
        // already understands, so the citation needs no explaining.
        let address = match hit.line {
            Some(line) => format!("{}:{line}", hit.source),
            None => hit.source.clone(),
        };
        // `message/user` rather than `message`: the reader decides whether a
        // hit is the question or the answer to it before spending a read on
        // the address.
        let what = match &hit.role {
            Some(role) => format!("{}/{role}", hit.kind),
            None => hit.kind.to_string(),
        };
        out.push_str(&format!(
            "\n[{}] {} {} ({} ago) - {}\n    {}\n",
            i + 1,
            what,
            who,
            age,
            address,
            hit.excerpt
        ));
    }
    if report.hits.is_empty() {
        // Two separate facts, never one guess. Emptiness has several
        // independent causes - a filter that matched nothing, terms that
        // matched nothing, history that could not be opened, history past
        // the scan cap - and earlier versions of this message picked one and
        // asserted it. Each time the guess was wrong for a corpus somebody
        // could construct. So: say what the search found nothing in, then say
        // what was not searched, and let those be true separately.
        let filtered = filters_in(&report.query);
        if report.candidates == 0 && !filtered.is_empty() {
            out.push_str(&format!(
                "nothing matched: {filtered} selected none of what was searched. \
                 path: keeps history that touched a matching file path; \
                 session: takes an id prefix. Without it the search widens to \
                 everything scanned."
            ));
        } else {
            out.push_str(
                "nothing matched; try fewer or different terms, or read the addresses \
                 under ~/.openmax/sessions",
            );
        }
        // Whatever the reason, anything the scan did not reach is disclosed -
        // the answer may be sitting in it.
        let mut omitted = Vec::new();
        if report.sessions_unreadable > 0 {
            omitted.push(format!("{} listed but unreadable", report.sessions_unreadable));
        }
        if report.sessions_skipped > 0 {
            omitted.push(format!("{} past the scan cap", report.sessions_skipped));
        }
        if report.sessions_partial > 0 {
            omitted.push(format!("{} partly scanned", report.sessions_partial));
        }
        if !omitted.is_empty() {
            out.push_str(&format!(" Not searched: {}.", omitted.join(", ")));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod quality;

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

    /// A session the ceiling cuts inside is a third disclosure class:
    /// "scanned" would overclaim, "skipped" would underclaim, and a reader
    /// trusting "nothing matched" over an unsearched tail is the lie this
    /// surface exists to never tell. The cut is counted, rendered in the
    /// header, owned by the empty-result disclosure, and dropped from the
    /// JSON when there is nothing to disclose.
    #[test]
    fn a_partly_scanned_session_is_disclosed_not_claimed_whole() {
        let (core, dir, project) = setup();
        seed_session(&core, &project, "big", vec![
            ChatMessage::user(format!("head {}", "x ".repeat(600))),
            ChatMessage::user(format!("TAIL-NEEDLE {}", "y ".repeat(600))),
        ]);
        let c = collect_chunks(&core, &project, sessions::unix_now(), 1_000, &[]);
        assert_eq!(c.scanned, 1, "the session was entered");
        assert_eq!(c.skipped, 0, "and never skipped whole");
        assert_eq!(c.partial, 1, "the ceiling cut inside it, and the cut must be counted");
        assert!(
            !c.chunks.iter().any(|ch| ch.text.contains("TAIL-NEEDLE")),
            "the cut is real: the tail was never indexed"
        );
        // A session read to its end is whole, whatever the ceiling was.
        let whole = collect_chunks(&core, &project, sessions::unix_now(), MAX_SCAN_BYTES, &[]);
        assert_eq!(whole.partial, 0, "a fully read session is not partial");

        let mut report = RecallReport {
            sessions_scanned: 1,
            sessions_partial: 1,
            ..Default::default()
        };
        let text = render(&report);
        assert!(text.contains("1 partly scanned (scan cap)"), "the header owns the cut: {text}");
        assert!(
            text.contains("Not searched: 1 partly scanned"),
            "an empty result owns the cut: {text}"
        );
        assert!(
            serde_json::to_string(&report).unwrap().contains("sessions_partial"),
            "a consumer that trusts the JSON must see the cut"
        );
        report.sessions_partial = 0;
        assert!(
            !serde_json::to_string(&report).unwrap().contains("sessions_partial"),
            "and nothing to disclose stays out of the report"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// One invalid-UTF-8 line must skip like any other corrupt line, not
    /// discard the transcript around it: the reader is lossy at the byte
    /// level and per-line at the parse level.
    /// A citation has to resolve. The line is what makes the address exact and
    /// bounded - `sed -n '<line>p'` piped through `head -c` - instead of a grep
    /// for a phrase guessed out of the excerpt, which returns whatever the
    /// record happens to weigh. So the number must survive the things a real
    /// log contains: blank lines and records that do not parse.
    /// The surface must not misdescribe itself. An agent cannot tell a policy
    /// cap from the end of a record, cannot resolve an address whose base it
    /// was never told, and will raise limits in a loop if told that is how to
    /// read one record whole. All three were measured on the real store.
    /// An empty result states two separate facts and guesses at neither:
    /// what the search found nothing in, and what it never reached. Every
    /// earlier version of this message picked a single cause and asserted it,
    /// and each guess was wrong for some corpus - a filter blamed for an
    /// unreadable store, an unreadable store declared when memory was
    /// searchable, a filter blamed for sessions past the scan cap.
    /// The contract is read by agents that cannot check it against the code,
    /// so it has to be checked here instead. Every claim below was wrong at
    /// some point in this stack: citations went absolute while the spec still
    /// said relative, and the spec promised `path:line` for every hit after
    /// memory files stopped having a line to give.
    #[test]
    fn the_recall_contract_describes_what_recall_actually_returns() {
        let (core, dir, project) = setup();
        seed_session(&core, &project, "work", vec![ChatMessage::user("wombat protocol")]);
        std::fs::create_dir_all(project.join(crate::memory::MEMORY_DIR)).unwrap();
        std::fs::write(
            project.join(crate::memory::MEMORY_DIR).join("fact.md"),
            "# the wombat protocol is documented here\n",
        )
        .unwrap();

        let report = recall(&core, &project, "wombat k:20").unwrap();
        let jsonl = report.hits.iter().find(|h| h.kind == "message").expect("a JSONL hit");
        let memory = report.hits.iter().find(|h| h.kind == "memory").expect("a memory hit");

        // What the contract promises, asserted against what recall returns.
        assert!(std::path::Path::new(&jsonl.source).is_absolute(), "absolute, every store");
        assert!(std::path::Path::new(&memory.source).is_absolute(), "absolute, every store");
        assert!(jsonl.line.is_some(), "a JSONL record is cited path:line");
        assert!(memory.line.is_none(), "a memory file is its own record");
        assert_eq!(jsonl.role.as_deref(), Some("user"), "a transcript hit names its speaker");
        assert_eq!(memory.role, None, "a memory file has no speaker to name");

        let spec = crate::spec::render("recall").expect("recall is a documented surface");
        assert!(spec.contains("absolute path"), "the contract must say addresses are absolute");
        assert!(
            spec.contains("no line to give"),
            "the contract must say memory hits carry no line, or an agent will look for one"
        );
        assert!(
            spec.contains("message/user"),
            "the contract must show how a speaker renders, or the field is invisible"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_empty_result_states_what_was_searched_and_what_was_not() {
        let (core, dir, project) = setup();
        let only = seed_session(&core, &project, "work", vec![ChatMessage::user(
            "the marmot telemetry pipeline was rewired",
        )]);

        // A filter that kept nothing names itself, scoped to what was searched.
        let filtered = recall(&core, &project, "marmot path:src/nowhere").unwrap();
        assert!(filtered.hits.is_empty());
        assert_eq!(filtered.candidates, 0);
        let text = render(&filtered);
        assert!(text.contains("path:src/nowhere"), "name the filter: {text}");
        assert!(text.contains("none of what was searched"), "scope the claim: {text}");
        assert!(!text.contains("try fewer or different terms"), "do not blame terms: {text}");
        assert!(!text.contains("Not searched"), "nothing was omitted here: {text}");
        assert!(
            !text.contains("search everything"),
            "dropping the filter reaches what was scanned, not what the cap skipped: {text}"
        );

        // Terms that match nothing still say so.
        let missing = recall(&core, &project, "zzzznotaword").unwrap();
        assert!(missing.candidates > 0, "the corpus was searchable");
        assert!(render(&missing).contains("try fewer or different terms"));

        // Anything the scan could not reach is disclosed, whichever branch ran.
        let ghost = seed_session(&core, &project, "ghost", vec![ChatMessage::user("marmot")]);
        for path in [
            sessions::messages_display(&core, &ghost),
            sessions::archive_display(&core, &ghost),
            sessions::compaction_display(&core, &ghost),
        ] {
            let _ = std::fs::remove_file(path);
        }
        for query in ["marmot path:src/nowhere", "zzzznotaword"] {
            let report = recall(&core, &project, query).unwrap();
            let text = render(&report);
            assert!(
                text.contains("Not searched: 1 listed but unreadable"),
                "{query} must own what it skipped: {text}"
            );
        }

        // Memory is searchable even with every session gone, so an unopenable
        // session set must not be reported as the whole corpus failing.
        let _ = std::fs::remove_file(sessions::messages_display(&core, &only));
        std::fs::create_dir_all(project.join(crate::memory::MEMORY_DIR)).unwrap();
        std::fs::write(
            project.join(crate::memory::MEMORY_DIR).join("fact.md"),
            "# the marmot pipeline is documented here\n",
        )
        .unwrap();
        let memory_only = recall(&core, &project, "zzzznotaword").unwrap();
        assert_eq!(memory_only.sessions_scanned, 0, "no session could be opened");
        assert!(memory_only.candidates > 0, "but memory was searched");
        assert!(
            render(&memory_only).contains("try fewer or different terms"),
            "readable memory means the terms really did miss"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn recall_reports_its_own_limits_instead_of_quietly_applying_them() {
        let (core, dir, project) = setup();
        seed_session(&core, &project, "long", vec![ChatMessage::user(
            "the quokka census figure appears here. ".repeat(200),
        )]);
        std::fs::create_dir_all(project.join(crate::memory::MEMORY_DIR)).unwrap();
        std::fs::write(
            project.join(crate::memory::MEMORY_DIR).join("fact.md"),
            "# quokka census is filed under docs\n",
        )
        .unwrap();

        // Every knob reports a request it could not honour, in both
        // directions: a value raised to a floor is as substituted as one cut
        // to a ceiling.
        let report = recall(&core, &project, "quokka excerpt:2000 k:0 budget:5").unwrap();
        let by = |knob: &str| report.clamped.iter().find(|c| c.knob == knob).cloned();
        assert_eq!(by("excerpt").map(|c| c.applied), Some(PAGE_CHARS), "cut to the ceiling");
        assert_eq!(by("k").map(|c| (c.requested, c.applied)), Some((0, 1)), "raised to the floor");
        assert_eq!(by("budget").map(|c| (c.requested, c.applied)), Some((5, 100)));
        let text = render(&report);
        for shown in ["excerpt:2000 answered as 1200", "k:0 answered as 1", "budget:5 answered as 100"] {
            assert!(text.contains(shown), "the reader must be told: {shown} missing from {text}");
        }
        for hit in &report.hits {
            assert!(hit.excerpt.chars().count() <= PAGE_CHARS + 2, "the cap it reports is applied");
        }

        // A request that fits reports nothing.
        let ok = recall(&core, &project, "quokka excerpt:400 k:5").unwrap();
        assert!(ok.clamped.is_empty(), "no note when every request was honoured");

        // Every address resolves the same way, whatever the store.
        for hit in &ok.hits {
            assert!(
                std::path::Path::new(&hit.source).is_absolute(),
                "a {} citation must be resolvable without knowing the cwd: {}",
                hit.kind,
                hit.source
            );
        }
        assert!(ok.hits.iter().any(|h| h.kind == "memory"), "memory was in this corpus");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_citation_names_the_line_that_holds_the_record() {
        let (core, dir, project) = setup();
        let id = seed_session(&core, &project, "addressing", vec![ChatMessage::user("seed")]);
        let path = sessions::messages_display(&core, &id);
        // line 1 valid, 2 blank, 3 corrupt, 4 the record wanted.
        std::fs::write(
            &path,
            "{\"role\":\"user\",\"content\":\"first record\"}\n\n\
             {not json at all\n\
             {\"role\":\"assistant\",\"content\":\"ANSWER zebrafish protocol\"}\n",
        )
        .unwrap();

        let report = recall(&core, &project, "zebrafish").unwrap();
        let hit = report.hits.iter().find(|h| h.excerpt.contains("zebrafish")).expect("found");
        let line = hit.line.expect("a JSONL record is addressable by line");

        // Resolve it the way an agent would, and check the bytes agree.
        let text = std::fs::read_to_string(&hit.source).unwrap();
        let cited = text.lines().nth(line - 1).expect("the cited line exists");
        assert!(
            cited.contains("zebrafish"),
            "line {line} must hold the cited record, got: {cited}"
        );
        assert_eq!(line, 4, "blank and unparseable lines must not shift the number");

        // A memory file is its own address; there is no line to give.
        std::fs::create_dir_all(project.join(crate::memory::MEMORY_DIR)).unwrap();
        std::fs::write(
            project.join(crate::memory::MEMORY_DIR).join("fact.md"),
            "# zebrafish protocol lives in docs\n",
        )
        .unwrap();
        let report = recall(&core, &project, "zebrafish k:20").unwrap();
        let mem = report.hits.iter().find(|h| h.kind == "memory").expect("memory hit");
        assert_eq!(mem.line, None, "the file is the record; a line would be noise");

        // A title is not a record. It is the first 48 characters of the first
        // user message, so the transcript already carries the same words with
        // more around them and an address that resolves; emitting the title
        // separately spent a slot to cite a file that does not contain it.
        let titled = seed_session(&core, &project, "quokka census plan", vec![
            ChatMessage::user("quokka census plan, and the body that follows it"),
        ]);
        let report = recall(&core, &project, "quokka k:20").unwrap();
        assert!(
            report.hits.iter().all(|h| h.kind != "title"),
            "a title is not an addressable record: {:?}",
            report.hits
        );
        let hit = report
            .hits
            .iter()
            .find(|h| h.session.as_deref() == Some(titled.as_str()))
            .expect("the session is still findable by the words in its title");
        assert!(hit.line.is_some(), "and it is found at a line that resolves");
        assert_eq!(hit.title.as_deref(), Some("quokka census plan"), "the name still travels");
        let _ = std::fs::remove_dir_all(dir);
    }

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
    fn salient_window_finds_the_region_the_context_is_about() {
        let noise = "unrelated filler line with nothing anyone asked about\n".repeat(40);
        let text = format!("{noise}retry_budget = 2 and upstream_timeout = 45s\n{noise}");
        let window = salient_window(&text, "what are retry_budget and upstream_timeout", 120);
        assert!(
            text[window.clone()].contains("retry_budget = 2"),
            "the window must land on the region the context names, got: {}",
            &text[window.clone()]
        );
        // Line-aligned, so the slice never begins mid-line.
        assert!(window.start == 0 || text.as_bytes()[window.start - 1] == b'\n');

        // With nothing to go on, the head is the honest default.
        assert_eq!(salient_window(&text, "", 120).start, 0);
        assert_eq!(salient_window(&text, "terms that appear nowhere here", 120).start, 0);
        // Short inputs are returned whole rather than windowed.
        assert_eq!(salient_window("tiny", "tiny", 120), 0..4);
    }

    #[test]
    fn camel_case_compounds_index_their_parts_and_prose_is_untouched() {
        // The compound is dropped when a part already leads back to it, so
        // one occurrence is never counted twice for its first part.
        assert_eq!(tokenize("StreamingMarkdown"), ["streaming", "markdown"]);
        assert_eq!(tokenize("MessageDone"), ["message", "done"]);
        assert!(terms_match("streamingmarkdown", "streaming"), "the whole still leads back");
        // An acronym run ends where the next word begins. "http" is four
        // characters, one short of leading back, so the compound is kept -
        // and cannot double-count, for exactly the same reason.
        assert_eq!(tokenize("HTTPServer"), ["httpserver", "http", "server"]);
        // Short leading parts cannot lead back (the prefix rule wants five
        // shared characters), so the compound is kept and stays searchable.
        assert_eq!(tokenize("ToString"), ["tostring", "to", "string"]);
        assert_eq!(tokenize("IntoIterator"), ["intoiterator", "into", "iterator"]);
        // Separators already split, so those runs have no case boundary left.
        assert_eq!(tokenize("keep_alive_msecs"), ["keep", "alive", "msecs"]);
        // Ordinary prose gains nothing: no boundary, no extra tokens.
        assert_eq!(tokenize("the deploy port is 7443"), ["the", "deploy", "port", "is", "7443"]);
    }

    #[test]
    fn a_word_inside_a_camel_case_identifier_is_findable() {
        let (core, dir, project) = setup();
        // "button" and "sender" never occur as their own words here, exactly
        // as they never do in a real Rust transcript. The prefix rule cannot
        // reach them: it only ever matches a compound's first part, so before
        // this these queries returned nothing at all.
        let sid = seed_session(&core, &project, "input handling", vec![
            ChatMessage::tool("c1", "MouseEventKind::Drag(MouseButton::Left) => self.select(),"),
            ChatMessage::tool("c2", "events: mpsc::UnboundedSender<AgentEventEnvelope>,"),
            ChatMessage::tool("c3", "impl ToString for Row { fn to_string(&self) -> String }"),
        ]);
        // "tostring" splits into parts too short to lead back to it, so the
        // compound has to be kept or an exact search for the identifier -
        // the most obvious thing a person types - would find nothing.
        for word in ["button", "sender", "envelope", "tostring"] {
            let report = recall(&core, &project, word).unwrap();
            assert!(
                report.hits.iter().any(|h| h.session.as_deref() == Some(sid.as_str())),
                "'{word}' lives only inside a compound and must still be findable: {:?}",
                report.hits
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

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

    /// The same rule for the memory store: a memory file's `path:` surface
    /// is its stem, never its address. Every memory lives under
    /// .openmax/memory/ beneath the project root, so matching the absolute
    /// source would let path:openmax, path:memory, or any ancestor directory
    /// of the project select every memory file as path evidence.
    #[test]
    fn memory_path_surface_is_its_stem_not_its_store_address() {
        let (core, dir, project) = setup();
        seed_session(&core, &project, "real", vec![ChatMessage::user("deploy port is 7443")]);
        std::fs::create_dir_all(project.join(crate::memory::MEMORY_DIR)).unwrap();
        std::fs::write(
            project.join(crate::memory::MEMORY_DIR).join("deploy-port.md"),
            "# the deploy port is 7443\n",
        )
        .unwrap();
        let ancestor = dir.file_name().unwrap().to_str().unwrap().to_lowercase();
        let ancestor_probe = format!("port path:{ancestor}");
        for probe in ["port path:openmax", "port path:memory", "port path:.md", &ancestor_probe] {
            let hits = recall(&core, &project, probe).unwrap().hits;
            assert!(hits.is_empty(), "{probe} matched the memory store's address: {hits:?}");
        }
        // The stem names the fact, and selecting on it keeps working even
        // where the text writes the words apart.
        let by_stem = recall(&core, &project, "port path:deploy-port").unwrap();
        assert!(
            by_stem.hits.iter().any(|h| h.kind == "memory"),
            "path:<stem> must still select the memory file: {:?}",
            by_stem.hits
        );
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
        assert_eq!(from_log, 1, "one page per record, got {from_log}");
        assert!(
            report.hits.iter().any(|h| h.session.as_deref() == Some(other.as_str())),
            "the second source must appear despite the log's many matching pages: {:?}",
            report.hits
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_hit_says_whether_it_is_the_question_or_the_answer() {
        // The real shape of the miss this exists for: a prompt that assigns
        // the work and a reply that does it share the topic's words, so
        // lexical ranking cannot separate them and the short prompt often
        // wins. Ranking is not the fix - both records genuinely match - but
        // the reader can act on the difference the moment it is labelled.
        let (core, dir, project) = setup();
        seed_session(
            &core,
            &project,
            "audit",
            vec![
                ChatMessage::user("audit the renderer and report which files are hot paths"),
                ChatMessage::assistant(
                    Some(
                        "the hot paths are src/draw.rs and src/layout.rs, both \
                         re-highlighting on every frame"
                            .to_string(),
                    ),
                    None,
                ),
                ChatMessage::tool("c1", "hot paths scan complete"),
            ],
        );
        let report = recall(&core, &project, "hot paths k:20").unwrap();

        let roles: Vec<_> = report.hits.iter().filter_map(|h| h.role.as_deref()).collect();
        for expected in ["user", "assistant", "tool"] {
            assert!(
                roles.contains(&expected),
                "every speaker must be reported, missing {expected}: {roles:?}"
            );
        }
        // The distinction has to survive into what an agent actually reads,
        // not just the JSON.
        let text = render(&report);
        assert!(text.contains("message/user"), "the rendered hit must name the speaker: {text}");
        assert!(text.contains("message/assistant"), "and distinguish it from the reply: {text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_one_page_shown_addresses_the_record_it_came_from() {
        // Collapsing sibling pages is only safe because the surviving hit says
        // where the record lives: the reader gets the rest with one read
        // instead of a second slot. If the address ever stops resolving, the
        // cap above is silently discarding evidence.
        let (core, dir, project) = setup();
        let mut log = String::new();
        for i in 0..400 {
            log.push_str(&format!("line {i}: paged-needle in this region of the log\n"));
        }
        let session = seed_session(&core, &project, "paged log", vec![ChatMessage::tool("c1", log)]);
        let report = recall(&core, &project, "paged-needle k:8").unwrap();
        let hit = report
            .hits
            .iter()
            .find(|h| h.session.as_deref() == Some(session.as_str()))
            .expect("the paged record must be found");
        let line = hit.line.expect("a transcript hit must carry its line");
        let text = std::fs::read_to_string(&hit.source)
            .unwrap_or_else(|e| panic!("cited path {} must be readable: {e}", hit.source));
        let cited = text
            .lines()
            .nth(line - 1)
            .unwrap_or_else(|| panic!("line {line} must exist in {}", hit.source));
        assert!(
            cited.contains("line 399: paged-needle"),
            "the cited line must hold the whole record, including the pages that were \
             collapsed, so one read replaces the slot they would have taken"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The documented budget cap holds even for a first hit that would
    /// overflow it: the hit survives, its excerpt shrinks to fit.
    #[test]
    fn a_weak_page_in_a_strong_episode_beats_the_same_page_alone() {
        let (core, dir, project) = setup();
        // Both candidates match exactly one query term ("changed") and nothing
        // else. The difference is the company they keep: one sits in a session
        // that answers the whole query across its other messages, the other in
        // a session that answers nothing else. Without episode evidence the
        // shorter, lonelier page wins on length normalization alone.
        let answer = seed_session(&core, &project, "keepalive work", vec![
            ChatMessage::user("what changed for the keepalive fix?"),
            ChatMessage::assistant(
                Some("first change: proxy.rs sets keep_alive_msecs to 25000".into()),
                None,
            ),
        ]);
        seed_session(&core, &project, "unrelated", vec![ChatMessage::assistant(
            Some("nothing changed".into()),
            None,
        )]);
        let report = recall(&core, &project, "changed keepalive fix").unwrap();
        let rank_of = |needle: &str| {
            report.hits.iter().position(|h| h.excerpt.contains(needle)).map(|i| i + 1)
        };
        let (part, lonely) = (rank_of("keep_alive_msecs"), rank_of("nothing changed"));
        assert!(
            part.is_some() && (lonely.is_none() || part < lonely),
            "a partial answer from the answering session must outrank an isolated \
             one-term match: {:?}",
            report.hits
        );
        assert_eq!(
            report.hits.first().and_then(|h| h.session.clone()),
            Some(answer),
            "the answering session leads: {:?}",
            report.hits
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn distinct_records_in_one_session_all_survive_collapsing() {
        let (core, dir, project) = setup();
        // Four separate messages, each answering a different part of the same
        // question. They share one transcript file, so collapsing keyed on
        // the source address would emit two of them and silently drop the
        // rest - the answer would rank first and still arrive incomplete.
        let sid = seed_session(
            &core,
            &project,
            "keepalive fix landed",
            vec![
                ChatMessage::user("what changed for the keepalive fix?"),
                ChatMessage::assistant(
                    Some("first change: proxy.rs sets keep_alive_msecs to 25000".into()),
                    None,
                ),
                ChatMessage::assistant(
                    Some("second change: basket.ts adds the reaper_interval loop".into()),
                    None,
                ),
                ChatMessage::assistant(
                    Some("third change: upstream.conf raises upstream_timeout to 45s".into()),
                    None,
                ),
                ChatMessage::assistant(
                    Some("fourth change: charge.rs drops retry_budget to 2".into()),
                    None,
                ),
            ],
        );
        let report = recall(&core, &project, "change k:8").unwrap();
        let shown: String =
            report.hits.iter().map(|h| h.excerpt.clone()).collect::<Vec<_>>().join(" ");
        for needle in ["keep_alive_msecs", "reaper_interval", "upstream_timeout", "retry_budget"] {
            assert!(
                shown.contains(needle),
                "every distinct answer must survive collapsing: {needle} missing from {shown}"
            );
        }
        assert!(
            report.hits.iter().filter(|h| h.session.as_deref() == Some(sid.as_str())).count() >= 4,
            "four distinct records, not two pages of one: {:?}",
            report.hits
        );
        let _ = std::fs::remove_dir_all(dir);
    }

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
