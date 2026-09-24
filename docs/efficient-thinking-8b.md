# Efficient Thinking VIII-b: Does Experience Accumulate?
## The second generation, its small gain, the redundancy that explains it, and what new experience does
> **STATE 2026-09-24: DRAFT v0.1, written backward from what is measured; ONE arm still running.** The gain of a second generation of experience over the first is measured (twelve paired games, all positive, small, not established against the loop's own variance); the reason is measured (half of the second generation's experience is byte-identical to the first's); the test of the reason — new experience of the same task family on problems the head has never seen — is one game in with two running (§7, ~04:40Z). §8's last line is the only sentence in this draft the data has not yet spoken on. Everything else is on disk under `experience/results/` with its command and its pre-registration (`docs/et8b-loop-gates.md`, §§1–15, every rule dated before the run it governs).

**Louay Alsakka** · September 24, 2026 · *draft v0.1*

## Abstract

Paper VIII showed that a frozen model with a read-only head fitted on its own verified history moves the
quality–compute frontier. This paper asks whether the improvement compounds: does experience collected by an agent
that is already steered by one head make a better head, and does that continue? The first answer is small and
consistent: across two matches of three games each and nine single pairings, the second generation's head beat the
first's in every one of twelve paired comparisons on the same 300 problems, by about four points, an effect real in
sign that does not separate from the loop harness's own run-to-run variance at 300 tasks per game. The second answer
is the reason: 49% of the second generation's training rows are byte-identical to the first's, and 86% of a third
generation's are identical to what came before. A steered agent stands where the previous agent stood and produces
the same state. Accumulation stops because the experience stops being new. The third answer tests that reason by
prediction: experience collected on a disjoint set of problems of the same family is only 1.3% duplicated, and if
novelty is what binds, its head should beat a same-problem head at matched decisions. One game of three is in and
the two heads are indistinguishable, both nine to ten points over base; the remaining games decide whether new
problems are new experience or only new rows. Beside these, the paper establishes three things about the instrument
that any accumulation claim needs: the loop's variance comes from its second and third decisions and is zero at one;
a head over no head is +8 to +14 in every game ever played; and a head fitted on another task family sits at base.

## Results at a glance

| finding | measurement | where |
|---|---|---|
| gen1 over gen0 | 12 of 12 paired games positive (+1.7 … +8.3); match 1 mean +3.0 [+1.8, +4.2]; match 2 mean +5.1 [−3.2, +13.4]; not established against the within-head-arm spread under the rule fixed before each match | §3 |
| head over base | +8 to +14 in every arm of every game; 12 of 12 clear every bar | §3 |
| why the gain is small | 49.2% of gen1's rows byte-identical to gen0's; 85.9% of gen2's to gen0 ∪ gen1; neighbour redundancy 67% / 90%; gains fall as redundancy rises (+5.7 → −2.0) | §4 |
| combining generations | gen0 ∪ gen1 head ≈ gen1 (+2.7 [−2.3, +7.7]); amount vs origin not separable at 300 | §5 |
| the instrument | one-decision loop bit-identical across runs (sd 0); three-decision loop sd ≈ 2.6 on the same box; the variance is decisions 2–3 | §6 |
| other-family experience | a head from SQL-repair experience sits at base on debugging (−0.3, −5.0); 11–13 below the same-family head | §6 |
| same-family novel experience | 1.3% duplicated; game 1: disjoint-problem head ≈ same-problem head (+8.7 vs +9.7 over base); games 2–3 running | §7 |
| stabilisation | NOT supported: base sd 1.95 sits between the heads' 1.26 and 2.40 | §6 |

## 1. The question

VIII closed with one generation: verified history, a head, a frontier moved. The obvious next step is the loop —
let the steered agent generate experience, fit the next head on it, repeat — and the obvious hope is that each turn
buys what the first did. This paper measures the second turn and asks what it is made of.

## 2. Instruments, and the discipline that the result forced

Everything runs on the loop harness of VIII (`et8b_loop`, budget 12, up to three head decisions per episode), the
same frozen 7B model, on an independent 300-problem set (seed 73) disjoint by content from every training problem.
Two things were learned about the instrument before any accumulation number could be read, and they are results:

- **Run-to-run variance.** Four identical base runs on the same 300 spread 20.7–27.7 (sd ≈ 3). Two runs of the
  same thing differed by five points with p = 0.02 under McNemar, which treats that variance as zero; McNemar
  p-values on loop rows are therefore within-run descriptives and nothing more. The source was located: a loop
  restricted to one head decision per episode is bit-identical between runs on every field of every episode; at
  three decisions on the same box it is not. The variance is a property of decisions two and three — carried state
  and longer context — and a design property, not the box (box effect 0.17 on the single-decision harness).
- **The chess-match rule.** A stochastic comparison is scored like a match: arms interleaved in one session, k
  games each, the effect as the mean of paired differences with its own interval, read against the arms' own
  within-session spread. A gain is "ahead of noise", never "a clear winner". Two matches were played (§3).

Heads are cut on whole decision groups, never rows: a head is fitted per candidate but scored per decision, and a
row cut can leave a decision with no correct candidate — a defect found in this work, fixed, and recorded (§7).

## 3. The gain: consistent, small, not established

| | game | base | gen0 head | gen1 head | gen1 − gen0 |
|---|---|---:|---:|---:|---:|
| match 1 (interleaved, one session) | 1 · 2 · 3 | 27.7 · 25.7 · 25.0 | 37.0 · 36.7 · 36.3 | 39.3 · 40.3 · 39.3 | +2.3 · +3.7 · +3.0 |
| match 2 (interleaved, one session) | 1 · 2 · 3 | 24.7 · 24.0 · 27.7 | 33.0 · 35.3 · 33.3 | 38.3 · 37.0 · 41.7 | +5.3 · +1.7 · +8.3 |

Match 1: mean +3.00 [+1.78, +4.22]; every one of the nine single pairings positive (+2.3 … +4.0); every gen1 run
above every gen0 run. Match 2: mean +5.11 [−3.18, +13.41]. Under the rule fixed before each match (the
replicate-mean interval must exclude zero and its lower bound must exceed the largest within-head-arm draw of the
session — 2.33 in match 1, 4.67 in match 2) the gain is **not established** in either. The rule's bar is a maximum
over draws and grows with the number of games, which makes a +4 effect unclearable at any n; that is recorded, and a
statistic bar (the pooled within-head-arm sd, 1.62) is registered for any future games, not applied to these. What is
established without any rule: twelve of twelve paired games positive, and the head over base at +8 to +14 in every
arm of every game. The published single-run figure of +5.7 [+0.7, +10.7] is inside this picture and is not a claim.

## 4. Why it is small: the experience stops being new

Measured on the recorded training rows, no model, no GPU:

| | gen1 vs gen0 | gen2 vs gen0 ∪ gen1 |
|---|---:|---:|
| rows byte-identical to a prior row | 49.2% | 85.9% |
| rows closer to the prior set than its own median self-distance | 67.2% | 90.1% |
| gain that generation bought (single run) | +5.7 | −2.0 |

An identical hidden state is an identical history prefix: the steered agent was in exactly the situation the
previous agent was in and produced the same activation. Redundancy rises with generation and the gain falls with it.
This is the paper's diagnosis, and it was the author's hypothesis before it was measured: the marginal value of a row
is its novelty, as with training data, and self-generated experience on the same problems runs out of novelty after
one generation.

## 5. Combining generations does not help

A head fitted on the union of gen0's and gen1's rows scores +2.7 [−2.3, +7.7] over gen1 alone — the union is barely
larger in unique content than gen0. Whether gen1's edge is about which rows or how many cannot be told at 300
problems (gen0's rows at gen1's count land between the two, compatible with both); resolving it would take about
2,500 problems and is not spent.

## 6. What the instrument established on the way

The loop's variance is its second and third decisions (§2). A head fitted on another task family (SQL repair, 1,437
rows, one shared head) sits at base on debugging — −0.3 and −5.0 over base in two games, 11 and 13 points under the
same-family head at matched rows — so experience does not transfer across families through the head. And the
registered hypothesis that the head stabilises the run is not supported: on three draws the base arm's spread sits
between the two head arms'.

## 7. Is new experience of the same family new? (running)

The test the diagnosis demands: experience collected by gen0's own heads on 300 disjoint problems of the same family
(content-hashed disjoint from every training and evaluation problem), fitted as one shared head at 290 decisions,
against gen1's own experience cut to the same 290 decisions. The precondition holds — 1.3% of the new rows are
byte-identical to gen0's, against 49% for gen1 — but 42% still sit closer to gen0's region than gen0's rows sit to
each other: new rows, largely the same corner of state space, the first measurement that separates "new rows" from
"new experience". Prediction, written before the games: if novelty binds, the disjoint-problem head beats the
same-problem head. Game 1 of 3: base 25.7 · gen0 35.7 · same-problem 35.3 · disjoint 34.3; disjoint − same-problem
= −1.0 [−7.3, +5.3]. All three heads beat base by nine to ten points and are indistinguishable from each other on one
game. The probe favoured the same-problem head by seven points (60.7 vs 53.4) and that did not convert to episodes —
the fourth time in this programme a probe gain failed to convert, and the reason the bar is problems solved.

## 8. What accumulation is, then

A second generation of self-generated experience buys a small, consistent gain that is not separable from the loop's
own variance at this size, because most of it is the first generation's experience again. Genuinely new rows from
disjoint problems of the same family are, on one game, worth the same as the old ones — which, if it holds, says the
limit is not the problems but the region of state space the agent's own policy keeps returning to. *[The last line is
written after games 2–3 of §7.]*

## Reproducibility

`docs/et8b-loop-gates.md` (every registered rule and reading, dated), `experience/reproduce_8b_accumulation.sh`,
artefacts under `experience/results/` (`et8b_*`, `match_*`, `s11_*`, `null73*`, `c_*`), `experience/gen_redundancy.py`,
`experience/subsample_by_decision.py`, `experience/match_analysis.py` (committed before the match's numbers existed).
Every table names its arms, its session and its cut.

## Appendix A. How This Was Found

Dates are 2026; one experimenter (E) and one reviewer (R), pseudonymous.
1. **09-20–21.** Gen1 over gen0 on two 300s (+5.7, +9.7), reproduced (+5.0, +7.3). Gen2 ≈ gen1. Written up as accumulation.
2. **09-22.** A null replicate — two runs of the identical base — was registered as an afterthought and fired at +5.0 [+1.7, +8.3]. Four base runs spread seven points; four of six null draws "significant" under McNemar. The accumulation row was inside the noise. *Rule: run the null before the claim, on any harness with sampling inside the loop.*
3. **09-22, night.** The author: "like a chess game — many games, average, be ahead of noise." The match design was registered; the first match's nine games were all positive and did not clear the base-null bar; the match itself measured that head arms are less variable than base arms, so the bar was re-registered for new data only and the first match never re-scored. *Rule: a yardstick found after a rule fires applies to data not yet collected.*
4. **09-23.** The author's hypothesis — the experience is not independent — measured: 49% / 86% duplicate rows. The independence arm was first built as a different task family (a domain shift, R's design error, kept as §6's result), then as disjoint problems of the same family. A weak head turned out to be a row-cut artefact (E), fixed by cutting on decisions; the constraint had existed as a comment in a neighbouring script for three weeks and had not travelled. *Rule: the unit of amount is the decision; a constraint lives in a shared file, not a comment.* The one-decision loop was bit-identical across runs; the three-decision loop was not, on one box: the variance is the decisions.
5. **09-23–24.** Second match: three games, all positive, not established; the bar's own defect (a maximum grows with n) recorded and replaced for the future. E's own count corrected itself twice in one evening in opposite directions. The author's arc for the paper: state the small gain exactly, show what explains it, show what new experience does.
