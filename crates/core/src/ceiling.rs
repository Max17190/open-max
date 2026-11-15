//! A route can make the reply limit a thinking budget: it ends thinking at a fixed count below max_tokens with a
//! normal finish. Two replies that stop thinking at the same count, short of the limit, lift the session's limit there.
use crate::state::Core;

/// The lifted limit, never below the configured one: room for a long thought, under what most routes accept.
pub(crate) const LIFTED_REPLY_LIMIT: usize = 32_768;

/// One session's reading of the route `origin`, in memory only, like the prompt ratio.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThoughtCeiling {
    origin: String,
    /// The limit sent and the reasoning count of the last 64 uncut replies at half that limit or more.
    seen: Vec<(usize, u64)>,
    lifted: Option<usize>,
    refused: bool,
}

impl ThoughtCeiling {
    pub(crate) async fn seeded(core: &Core, session_id: &str, origin: &str) -> Self {
        let stored = core.sessions.lock().await.get(session_id).and_then(|d| d.thought_ceiling.clone());
        stored.filter(|c| c.origin == origin).unwrap_or_else(|| Self { origin: origin.into(), ..Self::default() })
    }

    async fn save(&self, core: &Core, session_id: &str) {
        core.sessions.lock().await.get_mut(session_id).into_iter().for_each(|d| d.thought_ceiling = Some(self.clone()));
    }

    pub(crate) fn reply_limit(&self, endpoint: &crate::providers::ActiveEndpoint, prompt: usize) -> usize { endpoint.reply_limit_within(prompt, self.lifted.unwrap_or(endpoint.max_tokens)) }

    /// Reads a reply sent with `limit`; one the limit cut is a hard cut, the turn's to continue, not a budget.
    pub(crate) async fn observe(&mut self, core: &Core, session_id: &str, limit: usize, max_tokens: usize, reply: &crate::client::CompletionResult) {
        if self.read(limit, max_tokens, reply.usage.and_then(|u| u.reasoning_tokens), reply.hit_output_cap(limit)) { self.save(core, session_id).await; }
    }

    /// True when the reading changed. Under half the limit thoughts end on their own; within 8 of it, a hard cut.
    fn read(&mut self, limit: usize, max_tokens: usize, reasoning: Option<u64>, capped: bool) -> bool {
        let open = !capped && self.lifted.is_none() && !self.refused;
        let Some(count) = reasoning.filter(|&r| open && r as usize >= limit / 2 && r as usize + 8 < limit) else { return false };
        let repeat = self.seen.iter().any(|&(sent, seen)| sent == limit && seen.abs_diff(count) <= 1);
        self.seen.drain(..if repeat { self.seen.len() } else { self.seen.len().saturating_sub(63) });
        if repeat { self.lifted = Some(LIFTED_REPLY_LIMIT.max(max_tokens)) } else { self.seen.push((limit, count)) }
        true
    }

    /// True for a 400 or 422 that names the output limit (the route takes less): the configured limit holds for good.
    pub(crate) async fn refused(&mut self, core: &Core, session_id: &str, message: &str) -> bool {
        let message = message.to_ascii_lowercase();
        let named = (message.starts_with("backend returned 400") || message.starts_with("backend returned 422"))
            && ["max_tokens", "max_completion_tokens", "max_output_tokens", "output tokens", "completion tokens"].iter().any(|p| message.contains(p));
        if named { (self.lifted, self.refused) = (None, true); self.save(core, session_id).await; }
        named
    }
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
        assert!(!ceiling.read(4_096, 4_096, Some(812), false), "a short thought is no sign");
        assert!(ceiling.read(4_096, 4_096, Some(3_072), false));
        assert_eq!(ceiling.lifted, None, "one stop at a count is no sign");
        ceiling.read(4_096, 4_096, Some(2_500), false);
        assert!(ceiling.read(4_096, 4_096, Some(3_073), false));
        assert_eq!(ceiling.lifted, Some(LIFTED_REPLY_LIMIT), "the same count again, to a token");
        assert!(!ceiling.read(4_096, 4_096, Some(3_072), false), "lifted once");
        // A configured limit above the lift is never lowered.
        let mut high = ThoughtCeiling::default();
        high.read(65_536, 65_536, Some(40_000), false);
        high.read(65_536, 65_536, Some(40_000), false);
        assert_eq!(high.lifted, Some(65_536));
    }

    /// A route that cuts hard at the limit: each cut spends the whole limit
    /// and reports `length`, so however many there are, nothing lifts.
    #[test]
    fn hard_cuts_at_the_limit_never_lift_it() {
        let mut ceiling = ThoughtCeiling::default();
        for _ in 0..10 {
            assert!(!ceiling.read(4_096, 4_096, Some(4_096), true));
            // A hard cut whose finish says `stop` still spends the limit.
            assert!(!ceiling.read(4_096, 4_096, Some(4_090), false));
        }
        assert_eq!(ceiling.lifted, None);
    }

    /// Thoughts that end on their own spread their counts, and counts under
    /// half the limit, a missing count, or the same count under another
    /// limit are no sign. Only the last 64 counts are kept.
    #[test]
    fn natural_thoughts_do_not_lift_the_limit() {
        let mut ceiling = ThoughtCeiling::default();
        for count in (2_048..4_000).step_by(37) {
            ceiling.read(4_096, 4_096, Some(count), false);
        }
        for _ in 0..5 {
            assert!(!ceiling.read(4_096, 4_096, Some(1_500), false));
            assert!(!ceiling.read(4_096, 4_096, None, false));
        }
        assert_eq!((ceiling.lifted, ceiling.seen.len()), (None, 53));
        for count in 0..100 {
            ceiling.read(10_000, 10_000, Some(5_000 + 3 * count), false);
        }
        assert_eq!((ceiling.lifted, ceiling.seen.len(), ceiling.seen[0]), (None, 64, (10_000, 5_108)));
        let mut other = ThoughtCeiling::default();
        other.read(4_096, 4_096, Some(3_000), false);
        other.read(5_000, 4_096, Some(3_000), false);
        assert_eq!(other.lifted, None, "another limit");
    }

    /// A route that refuses the lifted limit goes back to the configured one
    /// and is not lifted again; only a refusal that names the output limit
    /// counts.
    #[tokio::test]
    async fn a_refused_lift_is_not_tried_again() {
        let dir = std::env::temp_dir().join(format!("openmax-ceiling-{}", uuid::Uuid::new_v4()));
        let (core, _rx) = Core::new(dir.clone()).unwrap();
        core.sessions.lock().await.insert("s".into(), Default::default());
        let mut ceiling = ThoughtCeiling::seeded(&core, "s", "route-a").await;
        ceiling.read(4_096, 4_096, Some(3_072), false);
        ceiling.read(4_096, 4_096, Some(3_072), false);
        assert!(!ceiling.refused(&core, "s", "backend returned 400: reasoning_content is missing in assistant message").await);
        assert!(!ceiling.refused(&core, "s", "backend returned 500: max_tokens").await);
        assert_eq!(ceiling.lifted, Some(LIFTED_REPLY_LIMIT));
        assert!(ceiling.refused(&core, "s", "backend returned 400 Bad Request: max_tokens is too large: 32768. This model supports at most 16384 completion tokens").await);
        assert_eq!(ceiling.lifted, None);
        let mut other = ThoughtCeiling::default();
        assert!(other.refused(&core, "t", "backend returned 422: {\"error\":\"Max_Output_Tokens exceeds the model limit\"}").await);
        ceiling.read(4_096, 4_096, Some(3_072), false);
        ceiling.read(4_096, 4_096, Some(3_072), false);
        assert_eq!(ceiling.lifted, None);
        // The stored reading is the session's on its route only.
        assert_eq!(ThoughtCeiling::seeded(&core, "s", "route-a").await, ceiling);
        assert_eq!(ThoughtCeiling::seeded(&core, "s", "route-b").await, ThoughtCeiling { origin: "route-b".into(), ..Default::default() });
        let _ = std::fs::remove_dir_all(dir);
    }
}
