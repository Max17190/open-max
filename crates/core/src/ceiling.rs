//! A route that turns the reply limit into a thinking budget.
//!
//! Most routes honour `max_tokens` as a hard cut: a reply that reaches it
//! stops with finish `length`, and the turn continues the thought
//! (`notes::reply_cut_note`). Some gateways instead end the model's thinking
//! at a fixed count below the limit, keep the rest for the answer, and report
//! a normal finish. The model is made to act mid-thought, and nothing the cut
//! accounting reads says so. What does say so is the count: replies whose
//! thinking stops at the same number of tokens, to within one, at a share of
//! the limit a thought ending on its own seldom reaches twice. Two such
//! replies lift the limit for the rest of the session on that route, so a
//! thought ends where the model ends it. A route that cuts hard never
//! reports a normal finish at that count, so it never fires there.

/// The limit a session sends once its route is seen budgeting thinking
/// against it: room for a long thought, under the output limit most routes
/// accept. Never below the configured limit.
pub(crate) const LIFTED_REPLY_LIMIT: usize = 32_768;

/// Distinct counts kept per session; the mark is two equal counts, so a
/// longer memory only adds chances for two unrelated thoughts to collide.
const KEPT: usize = 64;

/// One session's reading of its route. Keyed by the route that taught it,
/// as the prompt ratio is: another route budgets on its own terms.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThoughtCeiling {
    /// Reasoning counts of replies that finished normally at or above half
    /// the limit they were sent with, and that limit.
    seen: Vec<(usize, u64)>,
    /// Set once the mark is seen: the limit the session sends from then on.
    lifted: Option<usize>,
    /// Set when the route refused the lifted limit: it is not tried again.
    refused: bool,
}

impl ThoughtCeiling {
    /// The limit the lift raises `max_tokens` to, if it is in force.
    pub(crate) fn lifted(&self) -> Option<usize> {
        self.lifted
    }

    /// Records a reply sent with `limit` that reported `reasoning` thinking
    /// tokens. `capped` is a reply the limit cut (finish `length`, or the
    /// whole limit spent): a hard cut, which the turn already continues, and
    /// no sign of a budget. Returns the new limit when this reply is the
    /// second to stop thinking at the same count, which lifts it.
    pub(crate) fn observe(&mut self, limit: usize, max_tokens: usize, reasoning: Option<u64>, capped: bool) -> Option<usize> {
        if self.lifted.is_some() || self.refused || capped {
            return None;
        }
        let reasoning = reasoning?;
        // At least half the limit, and short of it by more than a reply's
        // closing tokens: below, thoughts end on their own all the time; at
        // the limit, it is a hard cut that lost its `length`.
        if (reasoning as usize) < limit / 2 || reasoning as usize + 8 >= limit {
            return None;
        }
        if self.seen.iter().any(|&(sent, count)| sent == limit && count.abs_diff(reasoning) <= 1) {
            let lifted = LIFTED_REPLY_LIMIT.max(max_tokens);
            self.lifted = Some(lifted);
            self.seen.clear();
            return Some(lifted);
        }
        if self.seen.len() == KEPT {
            self.seen.remove(0);
        }
        self.seen.push((limit, reasoning));
        None
    }

    /// The route refused a request at the lifted limit (it takes less
    /// output than that): back to the configured limit for good.
    pub(crate) fn refuse(&mut self) {
        self.lifted = None;
        self.refused = true;
    }
}

/// A refusal that names the output limit: the route takes less output than
/// was asked for. Its wording varies by server; each names the parameter or
/// the kind of tokens.
pub(crate) fn refused_for_output(message: &str) -> bool {
    const PHRASES: [&str; 5] = ["max_tokens", "max_completion_tokens", "max_output_tokens", "output tokens", "completion tokens"];
    let message = message.to_ascii_lowercase();
    (message.starts_with("backend returned 400") || message.starts_with("backend returned 422"))
        && PHRASES.iter().any(|phrase| message.contains(phrase))
}

/// The ceiling stored for the session, if the route that taught it is `origin`.
pub(crate) fn route_ceiling(stored: Option<&(String, ThoughtCeiling)>, origin: &str) -> ThoughtCeiling {
    stored.filter(|(taught_by, _)| taught_by == origin).map(|(_, ceiling)| ceiling.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A budgeting route: thinking stops at the limit less a fixed reserve,
    /// with a normal finish. The second such reply lifts the limit, and only
    /// once.
    #[test]
    fn a_repeated_thinking_ceiling_lifts_the_limit() {
        let mut ceiling = ThoughtCeiling::default();
        assert_eq!(ceiling.observe(4_096, 4_096, Some(812), false), None, "a short thought is no sign");
        assert_eq!(ceiling.observe(4_096, 4_096, Some(3_072), false), None, "one stop at a count is no sign");
        assert_eq!(ceiling.observe(4_096, 4_096, Some(2_500), false), None);
        assert_eq!(ceiling.observe(4_096, 4_096, Some(3_073), false), Some(LIFTED_REPLY_LIMIT), "the same count again, to a token");
        assert_eq!(ceiling.lifted(), Some(LIFTED_REPLY_LIMIT));
        assert_eq!(ceiling.observe(4_096, 4_096, Some(3_072), false), None, "lifted once");
        // A configured limit above the lift is never lowered.
        let mut high = ThoughtCeiling::default();
        high.observe(65_536, 65_536, Some(40_000), false);
        assert_eq!(high.observe(65_536, 65_536, Some(40_000), false), Some(65_536));
    }

    /// A route that cuts hard at the limit: each cut spends the whole limit
    /// and reports `length`, so however many there are, nothing lifts.
    #[test]
    fn hard_cuts_at_the_limit_never_lift_it() {
        let mut ceiling = ThoughtCeiling::default();
        for _ in 0..10 {
            assert_eq!(ceiling.observe(4_096, 4_096, Some(4_096), true), None);
            // A hard cut whose finish says `stop` still spends the limit.
            assert_eq!(ceiling.observe(4_096, 4_096, Some(4_090), false), None);
        }
        assert_eq!(ceiling.lifted(), None);
    }

    /// Thoughts that end on their own spread their counts, and counts under
    /// half the limit, a missing count, or the same count under another
    /// limit are no sign.
    #[test]
    fn natural_thoughts_do_not_lift_the_limit() {
        let mut ceiling = ThoughtCeiling::default();
        for count in (2_048..4_000).step_by(37) {
            assert_eq!(ceiling.observe(4_096, 4_096, Some(count), false), None);
        }
        for _ in 0..5 {
            assert_eq!(ceiling.observe(4_096, 4_096, Some(1_500), false), None);
            assert_eq!(ceiling.observe(4_096, 4_096, None, false), None);
        }
        let mut other = ThoughtCeiling::default();
        other.observe(4_096, 4_096, Some(3_000), false);
        assert_eq!(other.observe(5_000, 4_096, Some(3_000), false), None, "another limit");
        assert_eq!(ceiling.lifted(), None);
    }

    /// A route that refuses the lifted limit goes back to the configured one
    /// and is not lifted again.
    #[test]
    fn a_refused_lift_is_not_tried_again() {
        let mut ceiling = ThoughtCeiling::default();
        ceiling.observe(4_096, 4_096, Some(3_072), false);
        ceiling.observe(4_096, 4_096, Some(3_072), false);
        ceiling.refuse();
        assert_eq!(ceiling.lifted(), None);
        ceiling.observe(4_096, 4_096, Some(3_072), false);
        assert_eq!(ceiling.observe(4_096, 4_096, Some(3_072), false), None);
    }

    /// Only a refusal that names the output limit undoes the lift.
    #[test]
    fn a_refusal_names_the_output_limit() {
        assert!(refused_for_output("backend returned 400: max_tokens is too large: 32768. This model supports at most 16384 completion tokens"));
        assert!(refused_for_output("backend returned 422 Unprocessable Entity: {\"error\":\"Max_Output_Tokens exceeds the model limit\"}"));
        assert!(!refused_for_output("backend returned 400: reasoning_content is missing in assistant message"));
        assert!(!refused_for_output("backend returned 500: max_tokens"));
    }

    /// The stored reading applies to the route that taught it only.
    #[test]
    fn the_reading_belongs_to_its_route() {
        let mut taught = ThoughtCeiling::default();
        taught.observe(4_096, 4_096, Some(3_072), false);
        taught.observe(4_096, 4_096, Some(3_072), false);
        let stored = ("a".to_string(), taught.clone());
        assert_eq!(route_ceiling(Some(&stored), "a"), taught);
        assert_eq!(route_ceiling(Some(&stored), "b"), ThoughtCeiling::default());
        assert_eq!(route_ceiling(None, "a"), ThoughtCeiling::default());
    }
}
