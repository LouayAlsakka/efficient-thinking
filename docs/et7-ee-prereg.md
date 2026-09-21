# ET-VII E-E — the elicitation gap, probed. PRE-REGISTRATION.

> **This commit precedes the first episode.** WO-302's binding rule: predictions and readings in the
> repo before the run, citing the concept where it already registers them.

**Registered in the concept, verbatim** — `docs/efficient-thinking-7-concept.md:74–78`:

> *E-E — The elicitation gap, probed (the theory's measurement arm). On GSM8K/MATH judging cells with
> known ground truth: train linear/small probes on the judge's hidden states to predict candidate
> correctness; compare probe accuracy (lower bound on A\*(W)) against the judge's pick accuracy
> (A(W)). Registered: Δ > 0 and material — the judge's representations know more than its judgments
> express — and Δ shrinks with judge scale slower than A rises.*

This document adds only the operational specifics the concept does not fix.

## 1. The cells

`reasoning/arena_answers_v2.json` — 283 MATH problems, each with `gold`, and answers from five
policies (Qwen2.5 0.5B / 1.5B / 3B / 7B / 14B, 4-bit).

A cell is a **DECISIVE PAIR**: one policy's answer grades correct against `gold` and the other's does
not, by `reason_math_sweep.extract_boxed` + `normalize` — the III harness's own grader, not a new
one. Ties and both-right/both-wrong pairs are excluded and **counted as a line item**, because a
judge cannot be wrong on a pair with no right answer and including them would dilute A and A\* by
the same unknown amount.

Presentation follows the existing harness: judge **BLINDED** (never sees model names) and **pair
order RANDOMIZED**, as `reason_judge_scaling_local.py` already does.

## 2. The two quantities

- **A(W) — the judge's pick accuracy.** Its A/B/TIE answer against ground truth. TIE counts as wrong
  on a decisive pair; it is a refusal to express what it may still know, which is the paper's subject.
- **A\*(W) — a LOWER BOUND on what the state knows.** A linear probe on the judge's hidden states at
  the decision point, predicting which side is correct. Lower bound because a better probe may exist;
  the direction of that bound is what makes Δ > 0 meaningful and Δ ≈ 0 uninformative.
- **Δ = A\* − A**, per judge scale.

## 3. The probe — 8a's recipe, unchanged, because it has controls that have caught a fake

`et8_head_v3.hidden_at` for states, `logistic_fit` for the head, **held out by PROBLEM** (never by
pair: two pairs from one problem share the problem text, and splitting them across the boundary puts
the answer in the training set). Three controls at every fit, all three printed:

- **permuted labels ×5** — must collapse to chance;
- **PCA-8** — if eight dimensions match the full probe, it is a key being copied, not a readout;
- **row subsample** — if a fraction of the examples matches, the signal needs no examples.

v1 of the 8a probe scored 100% held out and PASSED its permutation control; PCA-8 and the subsample
are what exposed the label sitting in the prompt. That is why all three ship, always.

## 4. Readings, fixed before the run

| result | reading |
|---|---|
| Δ > 0, interval excluding zero, at ≥ 2 judge scales | the elicitation gap is real and material — the registered prediction |
| Δ ≈ 0 (interval contains zero) | the judge expresses what it knows; **the bound is not binding here** and E-D's headroom is not Δ-limited. Reported at full prominence: it is the result that would most embarrass the paper's framing, so it gets the same type as a positive |
| Δ < 0, excluding zero | the probe is worse than the judge — a probe failure, NOT a finding about judges. Report as an instrument result and do not interpret it |
| a control fires (PCA-8 or subsample matching the probe) | the Δ is withdrawn for that cell before anything is said about it |
| Δ does not shrink with judge scale | the second half of the concept's registration fails; report the curve and say so |

## 5. Scales

The scaling limb needs ≥ 3 judges. Available locally: Qwen2.5 1.5B / 7B / 14B / 32B (4-bit).
**A is measured per scale on the same cells** so the curve is not confounded by the cell set.

## 6. What this arm cannot show

- Nothing about aesthetic judging. The concept parks that behind ET-IV's machinery and it stays parked.
- Nothing about E-D. The cross-registration (E-D's gains ≤ E-E's Δ) is checked when E-D runs, not here.
- Δ is a **lower** bound on the gap. A small Δ does not prove the state knows little; it proves this
  probe found little, and §4's third row exists so that distinction is not quietly lost.

— Sautée (沙汰), for WO-302 box B
