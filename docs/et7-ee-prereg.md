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

## 3a. AMENDMENT, made before any episode — the policy-identity confound

*Added after stage 1's cell census (`reasoning/et7_ee_cells.py`, no model, no GPU) and before the
first judge call. Nothing has been run; this changes the design, not the reading of a result.*

The cells are pairs across POLICIES, and the stronger policy is usually the correct side:

| pair | n | stronger-is-correct |
|---|---|---|
| 0.5B / 7B | 155 | **98.1%** |
| 0.5B / 14B | 160 | 95.0% |
| 1.5B / 7B | 93 | 96.8% |
| 1.5B / 14B | 94 | 93.6% |
| 3B / 7B | 111 | 90.1% |
| 14B / 3B | 110 | 88.2% |
| 0.5B / 1.5B | 96 | 82.3% |
| 0.5B / 3B | 94 | 81.9% |
| **1.5B / 3B** | **96** | **49.0%** |
| **14B / 7B** | **41** | **43.9%** |

**A probe that merely detects WHICH MODEL WROTE AN ANSWER would score 82–98% on eight of these ten
pairs while knowing nothing about correctness.** Answer text carries style; the probe reads the
judge's state over that text. None of the three registered controls catches this — it is a real
signal in the representation, not a key copied from the prompt, so permutation collapses it, PCA-8
may well need the full space for it, and a subsample has plenty of it.

**Three changes, all fixed now:**

1. **The baseline is not 50%.** For each pair it is `max(p, 1−p)` of the table above. Δ is measured
   against the judge on the same cells, so this does not bias Δ itself — but a probe accuracy
   reported without it would read as a competence it does not have.
2. **The PRIMARY cells are the two balanced pairs — 1.5B/3B and 14B/7B, 137 cells over the problems
   they span.** There, policy identity buys nothing, so a probe that scores must be reading
   correctness. The other eight pairs are reported as a secondary, confounded stratum and Δ from
   them is not the headline. ⚠️ 137 cells held out BY PROBLEM is thin, and if the held-out split is
   too small to carry an interval, that is reported as the arm's limit rather than patched by
   pooling the confounded pairs back in.
3. **A FOURTH CONTROL: the policy-identity probe.** Train the same probe, on the same states, to
   predict which side came from the stronger policy. If it scores near the correctness probe on the
   confounded pairs, the two are not separable there and that stratum's Δ is withdrawn. Its score on
   the balanced pairs is the check that the balancing worked.

This is the kind of thing the cell census exists to find. Stage 1 costs no GPU precisely so the arm's
confounds are visible before its first episode rather than in its results.

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
