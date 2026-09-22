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

## 3b. AMENDMENT, made after the tie-scored run and BEFORE any forced-choice run — the definition of A

*Written 2026-09-22 by R, after an outside reader's objection to draft v0.2. Committed before the measurement it governs.*

**The objection, accepted.** Under §2's definition the 7B judge commits on 46 of 137 primary cells and is 73.9% right
when it does; the probe is 73.0%. On the cells where the judge speaks, Δ ≈ −0.01. The registered Δ = 0.482 is
therefore almost entirely 91 ties scored wrong by a rule — a gap between the state and what one prompt format
extracts, not between the state and what the system can express. The 1.5B judge tying on 98.5% of pairs, and the 14B
curve being a tie-rate curve, say the same thing. The tie-scored number cannot be the headline.

**A second admission.** §4 registered that a subsample control matching the probe withdraws the Δ. At 7B the
n = 100 subsample reads 0.730 against a probe of 0.730. Draft v0.1/v0.2 called that "a known property"; by the letter
of §4 the control fired, and a registered gate is not relaxed after it fails. The tie-scored 7B Δ is WITHDRAWN as a
claim under the registered reading. The reason the control fired is that at ~110 training cells per fold a 100-row
subsample is 90% of the data and cannot detect anything; that is a defect of the control's size, not evidence about
the gap, and it is stated as such — but the withdrawal stands, and the re-measurement below carries controls sized
for the n it has.

**The definition that will be published.** A is the judge's *forced preference*: at the decision position, the
log-probability of the token "A" against the token "B", the higher one being the pick. No tie is available to it.
Two prompt forms are read, both at the same decision position and on the same 137 cells:

- **F2 (primary):** the §3 prompt with the tie option removed ("Reply with EXACTLY one token: A or B."). This is
  what "expressed" means for a bound about expression.
- **F1 (check):** the original three-way prompt, reading logit(A) vs logit(B) and ignoring TIE. If F1 and F2
  disagree by more than 5 points the prompt is doing work and both are reported.

Position bias: the primary reading uses the ordering the probe's states were taken under (each cell's randomised
order is on disk in `meta.json`); the swapped ordering is also scored and the mean of the two is reported beside the
primary. P(TIE) under F1 is recorded per cell so the abstention decomposition can be redone on a continuous quantity.

**The probe is unchanged.** Same states, same five folds over problems, same fits; only A changes. Δ_forced is a
paired quantity per fold: (probe fold accuracy − forced-judge fold accuracy).

**Controls, sized to n = 137.** Permutation ×5 as before. PCA-8 as before, now read against the fold interval: it
"does not suffice" only if it sits below the probe's fold-interval lower bound, else the reading is "PCA-8 within
interval — the probe's excess over eight dimensions is not established at this n". Subsample: **n = 30 training rows**
(about a quarter of a fold's training set); a subsample at that size matching the probe within 0.02 withdraws the Δ.
Policy-identity probe as in §3a.

**Readings, fixed now.** Per judge, on the primary stratum:

| result | reading |
|---|---|
| Δ_forced fold-interval (t, df 4) excludes zero and its lower bound ≥ 0.05 | the elicitation gap is real under the publishable definition — the registered claim, restated under §3b |
| Δ_forced interval contains zero, or forced judge ≥ probe | **the state and the output agree; the abstention was format.** The registered claim FAILS under the definition that matters. Reported at full prominence, as §4 promised for Δ ≈ 0 |
| Δ_forced > 0 at 7B but not at 14B | the gap is real and closes with scale under forced choice; §9's "flat probe" reading is re-examined against the forced curve |
| any §3b control fires | that judge's Δ_forced is withdrawn before anything is said |

**Additional statistic, registered.** Probe accuracy restricted to the cells where the three-way judge tied, and to
the cells where it committed, from the same fold predictions (requested 2026-09-22, 理 12092). It is descriptive: it
tells what the tie-scored gap was made of; it does not license a claim on its own.

**What §3b does not change.** The tie-scored A stays in the paper as a secondary reading ("as a selector the judge is
useless two thirds of the time"), with §6's decomposition. The scaling limb (§5) is re-measured under F2 on the same
cells at every size, so the curve is measured under the definition that will be published, before the 32B point is
read.

— R, for E; the run is E's, on the cached states plus one forward pass per cell per judge per prompt form.
