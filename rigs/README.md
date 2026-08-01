# rigs

Measurement instruments for the context and retrieval code. Not run by CI, not
built into the binary, no runtime or token cost.

They are here because they kept getting rebuilt. Every context change in this
repo has been justified by numbers from scripts like these, and rewriting them
from scratch each time is both wasted effort and a quiet invitation to quote a
number nobody can reproduce.

**These are research instruments, not gates.** The gates live in the crate and
run on every PR: `recall::quality::recall_quality_gate` asserts retrieval
quality, and `the_prompt_prefix_only_grows_within_a_turn` plus
`a_new_session_reuses_the_previous_session_prefix` assert prefix stability. Use
these scripts to decide what to change; use the tests to keep it.

## What each one measures

| script | question | needs |
|---|---|---|
| `eval2.py` + `corpus.py` | Does a `--recall` change improve ranking? | the binary |
| `eval_truncation.py` | Which slice of a truncated tool output should survive? | your own transcripts |
| `prefix_probe.py` | Do successive requests share a byte-identical prefix? | the binary |
| `sweep.sh` | What does one scoring constant do across its range? | the binary |

### eval2.py

A labeled benchmark over a seeded synthetic corpus: sessions, archives,
compaction digests and memory files with planted answers, plus queries labeled
with the session that holds each one.

Two corpora with **disjoint vocabularies**, so tuning on `dev` and reporting on
`test` is a genuine held-out measurement rather than a reshuffle. Reports
hit@1, hit@3, MRR, needle coverage and output tokens, with **paired bootstrap
confidence intervals** and a per-query regression list when comparing two
binaries.

```
python3 eval2.py path/to/openmax --corpus both
python3 eval2.py path/to/base-openmax --vs path/to/candidate-openmax --corpus both
```

Tune on `--corpus dev` only. A change that cannot clear the paired CI on
`test` has not been shown to work.

### eval_truncation.py

When the budget forces an old tool output down to a slice, which bytes should
it be? Reads **your own transcripts** and scores each candidate slice against
the agent's own future: the distinctive tokens it went on to use from that
output. No labeling and no model, and the selector is blind to the future -
it scores only against messages before the truncation point, which is what the
harness knows at eviction time.

```
python3 eval_truncation.py --width 160
OPENMAX_SESSIONS=/path/to/sessions python3 eval_truncation.py
```

Nothing is written and nothing leaves the machine.

This is the rig that caught the trap: a synthetic corpus fills tool output with
filler, so it will happily endorse penalising tool text - while on real
transcripts the best answers *are* tool outputs. Measure workload-shaped
questions on the real workload.

### prefix_probe.py

Records the request bodies one real session sends and reports, for each
transition, whether `tools` stayed byte-identical and how many leading messages
did. Prefix caching keys on exactly that, and needs no cooperation from the
provider - most OpenAI-compatible servers never report `cached_tokens` at all.

```
python3 prefix_probe.py path/to/openmax --tool-iterations 4
```

### sweep.sh

Patches one `f64` constant in `recall.rs`, rebuilds, and evaluates, for each
value given. Restores the file on exit.

```
CONST_NAME=BM25_K1 ./sweep.sh 1.2 0.9 0.6 0.3
```

## Reading the output honestly

Some findings that came out of these, kept here because they are easy to
rediscover the hard way:

- **A flat sweep means leave it alone.** `k1` and `b` were swept over their
  full useful range with no metric movement, which matches the published
  finding that BM25's response surface is broad. A constant fitted to a small
  corpus is a liability carried into every future one.
- **Count configurations tried, and say how many.** Best-of-N on a small query
  set will hand you an improvement that is pure selection noise.
- **A test that passes on the broken code is not evidence.** Revert the change
  and confirm the number moves back before believing it.
- **Token count is not cost.** Removing tool-output tokens has been measured to
  *raise* spend, because cached input is an order of magnitude cheaper than the
  uncached turns that follow when evidence goes missing.
