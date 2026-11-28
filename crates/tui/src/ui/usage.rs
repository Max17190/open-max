//! What a session has spent, in tokens: the sums of the usage the provider
//! reported for each model request. No prices: the harness has no price
//! table, so it shows the counts a bill is made from and nothing it would
//! have to invent.

use open_max_core::sessions::TokenUsage;

/// Token sums over the requests whose reply reported usage. A request whose
/// reply reported none adds nothing, and a provider that never reports
/// leaves `requests` at zero, which reads as "not reported", not as zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub requests: u64,
    pub prompt: u64,
    pub completion: u64,
    /// Cached prompt tokens, over the requests that reported the field.
    pub cached: u64,
    /// How many requests reported cached tokens, and their prompt tokens.
    pub cached_requests: u64,
    pub cached_prompt: u64,
}

impl Totals {
    /// The session's usage record, which the core appends for every
    /// request it emits a usage event for.
    pub fn from_records(records: &[TokenUsage]) -> Self {
        let mut totals = Self::default();
        for record in records {
            totals.add(record.prompt_tokens, record.completion_tokens, record.cached_tokens);
        }
        totals
    }

    pub fn add(&mut self, prompt: u64, completion: u64, cached: Option<u64>) {
        self.requests += 1;
        self.prompt = self.prompt.saturating_add(prompt);
        self.completion = self.completion.saturating_add(completion);
        if let Some(cached) = cached {
            self.cached_requests += 1;
            self.cached = self.cached.saturating_add(cached);
            self.cached_prompt = self.cached_prompt.saturating_add(prompt);
        }
    }

    /// The status line's compact form, `in 1.2M out 34k`, once anything
    /// was reported.
    pub fn status_segment(&self) -> Option<String> {
        (self.requests > 0)
            .then(|| format!("in {} out {}", compact_count(self.prompt), compact_count(self.completion)))
    }

    /// The exact rows `/status` and `/usage` show, as (label, value).
    pub fn rows(&self) -> [(&'static str, String); 3] {
        const NONE: &str = "not reported";
        if self.requests == 0 {
            return [("tokens in", NONE.into()), ("  cached", NONE.into()), ("tokens out", NONE.into())];
        }
        let cached = if self.cached_requests == 0 {
            NONE.to_string()
        } else {
            let share = self.cached as f64 / self.cached_prompt.max(1) as f64 * 100.0;
            let mut row = format!("{} ({share:.0}%)", exact(self.cached));
            if self.cached_requests < self.requests {
                row.push_str(&format!(", reported by {} of {} requests", self.cached_requests, self.requests));
            }
            row
        };
        let requests = match self.requests {
            1 => "1 request".to_string(),
            n => format!("{n} requests"),
        };
        [
            ("tokens in", format!("{} over {requests}", exact(self.prompt))),
            ("  cached", cached),
            ("tokens out", exact(self.completion)),
        ]
    }
}

/// A token count with thousands separators: `1,234,567`.
pub fn exact(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A token count for the status line: `842`, `16.8k`, `2.4M`, with a
/// whole `.0` dropped (`17k`, never `17.0k`).
pub fn compact_count(n: u64) -> String {
    let (value, suffix) = if n >= 999_950 {
        (n as f64 / 1e6, "M")
    } else if n >= 1_000 {
        (n as f64 / 1e3, "k")
    } else {
        return n.to_string();
    };
    let s = format!("{value:.1}");
    let s = s.strip_suffix(".0").unwrap_or(&s);
    format!("{s}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rows say "not reported" until a reply reports usage, give the
    /// cached share only over the requests that reported it, and name how
    /// many those were when not all of them did.
    #[test]
    fn totals_report_only_what_the_provider_reported() {
        let mut totals = Totals::default();
        assert_eq!(totals.status_segment(), None);
        assert!(totals.rows().iter().all(|(_, value)| value == "not reported"));
        totals.add(12_000, 340, None);
        let rows = totals.rows();
        assert_eq!(rows[0].1, "12,000 over 1 request");
        assert_eq!(rows[1].1, "not reported");
        assert_eq!(rows[2].1, "340");
        totals.add(1_000_000, 1_660, Some(750_000));
        assert_eq!(totals.rows()[0].1, "1,012,000 over 2 requests");
        assert_eq!(totals.rows()[1].1, "750,000 (75%), reported by 1 of 2 requests");
        assert_eq!(totals.status_segment().as_deref(), Some("in 1M out 2k"));
        let records = [
            TokenUsage { prompt_tokens: 100, completion_tokens: 5, cached_tokens: Some(40), ..Default::default() },
            TokenUsage { prompt_tokens: 300, completion_tokens: 7, cached_tokens: Some(0), ..Default::default() },
        ];
        let restored = Totals::from_records(&records);
        assert_eq!(restored.rows()[1].1, "40 (10%)");
        assert_eq!(restored.rows()[2].1, "12");
        assert_eq!([0, 999, 16_842, 999_949, 999_950, 2_400_000].map(compact_count), ["0", "999", "16.8k", "999.9k", "1M", "2.4M"]);
        assert_eq!(exact(1_234_567), "1,234,567");
    }
}
