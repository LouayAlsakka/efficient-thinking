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
