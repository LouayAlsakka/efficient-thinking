# ET-IV amendment 2 — the rater is a frontier model (pre-registration)

*Registered 2026-09-20; this commit is the timestamp. Ruled by the author: "judge for Paper IV: use Fable API or
Opus." The human rater IV was designed around has not been found; the human arms E2, E3 and E5 unpark with a
frontier language model as the rater. What that changes in the claims, and the controls it requires, are fixed
here before any call is made.*

## 1. What changes

- **q is measured against a model, not a person.** Every IV claim that read "human-rated" now reads "rated by
  the frontier judge named below". The paper's title stays; its scope sentence changes to "taste as a frontier
  model judges it", and the human-rater design stays in the record as the study IV would have been.
- **The judge is `claude-fable-5-1` via the API** for all rating (E2 q, E5 persistence, 7d selection value).
  **The optimisation target in E3 (the Goodhart curve) is a different model, `claude-opus-4-7`,** so the curve
  measures what happens when a policy is optimised against one judge and scored by another. A curve measured
  and optimised against the same judge cannot show Goodhart; that separation is the amendment's main control.
- **Self-consistency is measured, not assumed:** every rating set is re-rated once with a different seed and
  candidate order, and the agreement rate is the ceiling q is read against, exactly as the human design had the
  week-later re-rate. Position bias is measured by the order swap (the ET-III instrument).
- **The conventionality prediction, registered now.** ET-VII's ensemble-bias argument predicts a frontier judge
  shares the training-distribution preference for conventional verse. So: **G5p** — the frontier judge's
  preference agrees with the canon judge (7b) more often than it agrees with the form checker's random-valid
  pick, and its margin between form-valid candidates is smaller than between form-valid and form-broken ones.
  The author's own reading of the Tang Yin study — "looks like his on the surface but lacks his personality and
  depth" — is the kind of distinction this predicts the model judge will not make; **G6p** — on the Tang Yin
  pairs (heldout vs adapter), the frontier judge's accuracy at naming the poet's own poem is reported beside the
  Opus 4.7 voice judge already measured, and the pairs it gets wrong are printed.

## 2. What does not change

Nothing scored by a persona oracle is a claim; the `_dev` files stay dev. The form checkers stay the verifier for
E1/E4/E6. Pareto pruning before rating, the pairwise protocol at small N, and per-item logging from the first
call all stand. No result from this amendment enters any other paper.

## 3. Cost gate

The rating budget is estimated by the executor from the item counts before the first call and approved by the
author on the day; the estimate, the approval and the metered spend are printed in the paper's reproducibility
section. A measurement does not run on production inference paths.

## 4. The rater's budget, registered 2026-09-24 BEFORE the run it applies to

**The Fable rater reasons before answering.** Its response carries a reasoning block ahead of its
text, and at the arm's original 16-token budget the reasoning consumed the whole allowance and the
text came back EMPTY — a paid call returning nothing. Measured over six pairs: 2 of 6 parsed at 16,
3 of 6 at 48, **6 of 6 at 128**.

**Its budget is therefore 128 tokens, the smallest tested that parses 6 of 6.** The Opus 4.7 voice
judge it is reported beside answered at 16. **The two budgets are printed together, and the G6p
comparison carries that difference as a stated caveat** — a judge that may reason before answering
and a judge given sixteen tokens and no room to are not the same instrument, and no reading may
treat them as one.

**The budget fixed the judge, not only the parsing.** On eight pairs re-rated with the candidates
exchanged, self-consistency reads **1.0 at 1024 and 0.0 at 128** — a judge that cannot finish its
sentence cannot agree with itself either. The parse rate was the symptom; this is the instrument.

⚠️ **The reasoning cost for THIS run is reported in aggregate, not per call.** The transport computed
reasoning tokens and the meter did not store them, so the value was returned and dropped — a value
returned and never stored is not a record. From the real-brief probe: **median ~476 of 1024 output
tokens, the reasoning being most of it.** The per-call `reasoning_tokens_est` and `stop_reason`
fields are in the meter now and begin with the next run; `stop_reason` is what separates "answered
in 480 tokens" from "burned 1024 and said nothing", which is the failure this budget exists to
prevent. The hard cap is unchanged and now binds against the raised per-call reserve.

**A limit on n that belongs to the checker, not the judge**: on the first full attempt, 110 of 300
briefs had a Pareto set that collapsed to a single candidate — nothing to compare before the judge
was asked. On a 50-brief subset the figure was 12 of 50. A Pareto set collapses when the checker's
dimensions do not separate the samples, so this caps how many briefs the arm can rate however good
the judge is, and it is reported as the checker's property.

This is a change to a registered instrument. It is written here before the run rather than
described afterwards, which is the only thing that separates a correction from a result chosen to
fit.
