# ET-8b — Accumulation: the loop, its gates and its readings (pre-registration)

*Registered 2026-09-19 (this commit is the timestamp). 8a §9 established why accumulation could not be tested on a
single fixed decision: the second generation's training states were byte-identical to the first's. 8b builds the
loop in which the prior's earlier choices shape the states it later reads, and asks whether a second generation,
trained on states its parent produced, beats its parent at held capability. It also carries the bridge to VII: does
the gain survive when the history is unverified?*

## 1. The loop

Episode = inspect → **decision 1** (head or base picks region × family) → patch → run tests → if red, inspect the
new state → **decision 2** → patch → run → if red, **decision 3** → patch → run. Up to three head decisions per
episode; the state at decision k depends on the choices at decisions < k. Same v3 task family, verifier, budget
accounting, per-problem logging and trajectory persistence as 8a. A budget in the prompt makes each budget a
different agent; the loop agent is a new agent and is compared only with itself.

## 2. Gates, in order, each with a number fixed now

- **G1 task gate** on the task draw used: distinct problems ≥ 0.9 of episodes (8a rule).
- **G2 prompt gate at EVERY decision the head touches:** distinct decision prompts / episodes ≥ 0.9 at decision 1,
  2 and 3 separately. A decision point that fails is not scored.
- **G3 state-dependence gate (the P9 failure, tested first):** run the base loop and the gen0-head loop on the same
  300; at decision 2, the fraction of episodes whose state differs between the two arms must be ≥ 0.30. Below
  that there is nothing to accumulate on and 8b reports that as its result.
- **G4 a base of its own:** the base-loop agent, no head, 300 episodes, same budget; every comparison is paired
  against it. The 8a single-decision numbers are never reused as this base.

## 3. Generations

- **gen0:** head fitted (layer 18, L-BFGS, read-only, 8a recipe) on the base-loop's verified history at all
  three decision points, one head per decision point or one shared head — chosen by held-out probe accuracy on
  base-loop states BEFORE any loop run, and fixed.
- **gen1:** head fitted on the states the gen0-head loop produced, with the verifier's outcomes. Same recipe, same
  three probe controls (permutation ×5, PCA-8, n=100 ×5 draws).
- **gen1-unverified (the VII bridge), defined 2026-09-20 after the first definition was found to name the verifier under
  another name (`agent_claims_pass` is the test verdict):** the label is an OFFLINE SELF-ASSESSMENT — for each patch step
  of the cross-fitted gen0 trajectories, the state is replayed and the frozen model is asked whether the tests now pass,
  with the verifier's answer withheld; the head is fitted on the same states with that yes/no as the label. The rollouts
  are untouched, so "everything else identical" holds exactly. A second arm, **gen1-self-stopped** — a rollout in which the
  verdict is never fed back and the agent stops on its own belief — is registered as OPTIONAL: it measures the arm and the
  changed rollout together, and if run it is read only beside the first, never instead of it.
- **Cross-fitting rule (added 2026-09-20, after gen0's memorised training trajectories contaminated gen1's fit):** every
  generation's trajectories used as a later generation's training data are produced under k-fold cross-fitting, so no
  episode is steered by a head that saw its task; each head file carries its training range and the chain refuses to
  launch a fold whose head overlaps it. A withdrawn number does not clean a corpus; the corpus is regenerated.

## 4. Readings, written before the runs

All on an independent 300, paired, exact test, interval reported; capability = problems solved at the loop
budget; cost = total tokens with head inference counted; the matched-compute point reported as in 8a §7.6.

| result | reading |
|---|---|
| gen1 > gen0, interval excludes zero, at ≤ gen0's cost | **accumulation** — the frontier moved twice; 8b's claim |
| gen1 ≈ gen0 (interval contains zero) with gen0 > base | **one-shot** — experience moves the frontier once and stops; the series says so |
| gen1 < gen0, interval excludes zero | **dogma** — the prior's own choices narrowed what it later learned from; reported at full prominence |
| gen1-unverified ≈ gen1 | the verifier's bits were not what carried the gain; VII's bound is NOT what limits accumulation here |
| gen1-unverified ≈ gen0 or < gen0 while gen1 > gen0 | **the external bits are the gain** — VII's Internal-Improvement Bound observed inside 8b |
| G3 fails | nothing to accumulate at this budget; 8b reports the state-dependence fraction and stops |

Selection rules fixed now: no budget ladder; the loop budget is 12 model actions total across the three decisions;
no re-fit after seeing the independent set; exclusions as line items.

## 5. Cost

Harness 2 days. Base loop 300 ≈ 4 h. gen0 fit + 300 ≈ 5 h. gen1 fit + 300 ≈ 5 h. Bridge arm ≈ 5 h. One box, one
week, after 8a's R7.

## 6. Record, 2026-09-21

- gen0 (single fit, tasks < 226) on held-out 75: +17.3 [+5.3, +29.3]. Its training trajectories were memorised (true region
  found at decision 1 on 100% of training tasks) and contaminated gen1's first fit; that arm read −12.0 vs gen0 and is
  recorded as the defect's signature, not as accumulation.
- Cross-fitted gen0 (3-fold, head on 150 tasks each) on all 225: +18.7 [+12.9, +24.9], p 3×10⁻⁹ — a lower bound on the
  full head, reported as its own arm.
- gen1 (fitted on cross-fitted gen0 trajectories) on held-out 75: vs gen0 −2.7 [−12.0, +6.7], p 0.77; vs base +14.7
  [+2.7, +26.7]. Interval 18.7 points wide at n = 75: consistent with one-shot and with an increment it cannot see. NOT
  filed as §4's one-shot. The independent 300 decides (running).
- gen1-unverified: closed by its precondition. The frozen model's own judgment of success is a constant — 0 of 568 patch
  points called "pass", including all 83 that passed; agreement with the verifier 85.4% = the base rate of failure; four
  prompt wordings × two temperatures, said-pass never above 3.3%. No positive class, no head. Reading, bounded to one
  model and one setting: there are no internal bits about success to improve on — VII's Internal-Improvement Bound
  observed one level earlier than the arm built to test it. The optional self-stopped arm is not run: an agent that never
  believes it has succeeded never stops.

## 7. Accumulation, measured 2026-09-21 — and gen2 pre-registered before it runs

**Result.** On two independent 300s, reported seed by seed and not pooled: gen1 vs gen0 +5.7 [+0.7, +10.7], p 0.040
(seed 73) and +9.7 [+4.7, +15.0], p 0.0004 (seed 47); model tokens per episode +0.1% and −3.0%; controller cost
identical by construction. §4 row 1 fires: the frontier moved twice. gen0 vs base +13.7 / +13.0; gen1 vs base +19.3 /
+22.7. The n = 75 screen had read −2.7 [−12.0, +6.7]; both 300s lie inside or at the edge of that interval — an
underpowered screen, not a contradiction — and it was not filed as one-shot because the reading note fixed that before
the number existed. Bounds: one model, one environment family, gen1's heads fitted on fewer rows than gen0's (2,824 vs
4,052), so the comparison is conservative and not matched-data. Instrument by hash: model, three task-set signatures,
six head md5s, disjointness measured zero. One-command reproduction: `experience/reproduce_8b_accumulation.sh`, both
gates exercised by making them fail.

**gen2, registered now (this commit precedes the run).** gen1 trajectories on tasks 1–225, 3-fold cross-fitted (no
episode steered by a head that saw its task); gen2 heads fitted on them with the same recipe and controls; gen2 run
on both independent 300s, paired vs gen1, vs gen0, vs base; cost condition as §4. Readings, fixed before the run:

| result | reading |
|---|---|
| gen2 > gen1 on both seeds, intervals excluding zero, at ≤ gen1's cost | accumulation continues; the series asks in VIII-b's paper what bounds it |
| gen2 ≈ gen1 (intervals contain zero) on both | the loop reaches a plateau at generation 1 — reported as such, with the verifier's ceiling named as the candidate bound |
| gen2 < gen1, interval excluding zero | consumption: a head steering the search removes the signal its successor needs; reported at full prominence with the already-found-at-decision-1 fraction printed beside it |
| seeds disagree in direction | the disagreement is the result; no pooling |

No gen3 without a ruling; VIII-b's experimental package closes after gen2 whichever way it falls, and the paper is
written backward from there.

**E5 under amendment 2.** The arm asks whether a specific person's stated preferences transfer. With a model rater
there is no person unless the author supplies the K preferences. Ruled 2026-09-21: E5 is PARKED unless the author
writes K statements of his own poetic preference; the version that elicits the judge's own preferences is not run,
because it measures a model learning a model, not the registered question.
