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

## 8. gen2, measured 2026-09-21 16:35Z — accumulation stops at generation 1

Seed by seed, not pooled: gen2 vs gen1 −2.0 [−6.3, +2.3], p 0.44 (seed 73); −7.0 [−11.0, −3.0], p 0.0015 (seed 47).
Cost flat (−0.2%, +0.4%). gen2 vs base +17.3 / +15.7 — between gen0 and gen1 on both. §7's plateau row fires on one
seed and its consumption row on the other; the fourth row does not fire, since both estimates are negative and the
intervals overlap on [−6.3, −3.0]. **Ruling (2026-09-21 17:1xZ), RESTATED 2026-09-22 after the reproduction:** the reproduction script re-ran both seeds
and seed 47's −7.0 did not reproduce — the re-run read −2.0 [−6.3, +2.3], on top of the other three measurements (−2.0,
−2.3, −2.0). The rise reproduced on both seeds (+5.0 and +7.3 against the published +5.7 and +9.7, each inside the
other's interval). The sentence the record now supports is: *a third generation does not improve on the second — four
estimates between −2.0 and −7.0, the three that agree at about −2, and no significant decline that survives
re-measurement; the series' best head is generation 1.* The plateau row fires on three of four; the consumption row
fired once and did not reproduce, and is reported as such. Run-to-run variance on this loop is larger than one seed's
interval suggests; every single-run number in VIII-b carries that variance and only this one was measured twice. The consumption mechanism is reported as consistent, not
demonstrated: the d1 and d2 heads are byte-identical between gen1 and gen2, so the whole difference is the d3 head,
which learned from trajectories where the region was already found by decision 3 in 48.8% of surviving episodes
(41.5% for gen1's). One moved point is not a mechanism. A gen3 would be empty by construction — the loop has three
decisions and gen2 moved the last — independently of the stop rule. **VIII-b's experimental package is closed.**
The frontier moved twice — base → gen0 → gen1 — and did not move a third time on this loop; the paper is written
backward from that sentence.

## 9. Pre-registered 2026-09-21 18:1xZ — the loop base at the generations' compute

The loop's generations spend about 30% more model tokens per episode than the loop base (≈3,800 vs ≈2,950). The
accumulation limb (gen1 vs gen0) is equal-cost by construction and needs no control; the loop's gain over base does.
Arm: on each independent seed, a small ladder of base action budgets on slice 1, the budget whose mean tokens per
episode is nearest 3,800 chosen by absolute distance before the ladder is read, then that budget on the full 300,
paired against gen0 and gen1. Readings: base-matched below gen0 by more than the interval → the loop's gain is
experience, not compute; base-matched ≈ gen0 → generation 0's gain was bought with compute and only the gen1-vs-gen0
increment stands as the frontier claim. Printed either way.

**Outcome (2026-09-21 19:1xZ): the arm is unbuildable, and that is the finding.** The action budget does not bind on the
loop — base tokens per episode are flat from budget 14 to 24 (≈2,880 and ≈2,930 on the two seeds) because the
three-decision cap ends the episode first; the selection rule then chose on noise (budget 14 on one seed, 20 on the
other). The generations' extra compute is endogenous: fewer actions (7.7 vs 8.2) at 37% more tokens per action,
because the head's choice of where to look lengthens the context the agent carries. It cannot be matched with a knob,
and a base with a raised decision cap or a forced-inspect policy is a different agent. VIII-b therefore reports the
loop's gain over its base with the per-action decomposition and does not claim it as a frontier shift; the
accumulation claim (gen1 over gen0) is at equal cost by construction and stands on its own.

## 10. Pre-registered 2026-09-22 — the nonlinear head at matched compute (from the site's first outside review)

The obvious alternative to a linear read of the hidden state is a small nonlinear head on the same states. Arm: a
two-layer MLP (hidden 256, ReLU) fitted on the same training states and verifier labels as the linear head, same
three decisions, same probe controls (permutation ×5, PCA-8, n = 100 ×5), charged its own inference; run as gen0 on
both independent 300s, paired against the linear gen0 and against base. Readings, fixed now: MLP ≈ linear (interval
contains zero) → the linear head is the efficient form and the claim stands as an efficiency claim; MLP > linear by
more than the interval → the linear head is a cheap approximation of a trained value head and the paper says so in
those words; MLP < linear → overfitting on 300 episodes, printed with the training-set accuracy beside it. Also
registered: the six failed carriers of VIII §7.3 re-run on the 300-problem set, so their failures become results
rather than directions.

## 11. Pre-registered 2026-09-22 — is accumulation a data effect? (the author's question)

The author asked: if gen1 beats gen0 and gen2 does not beat gen1, can generations 0 and 1 be combined into one head
applied once? Two arms, both cheap, both on the two independent 300s, paired against gen1 and gen0:

- **gen0+1 combined:** one head fitted on the union of gen0's training trajectories (base-steered) and gen1's
  (gen0-steered), same recipe and controls, applied once.
- **gen1-at-gen0-rows control:** the gen1 head re-fitted on a subsample of its own data at gen0's row count, and gen0
  re-fitted at gen1's, so that origin of data and amount of data are separated.

Readings, fixed before the run: combined ≈ gen1 → accumulation is a DATA effect — the loop's value is that a steered
agent generates better training data, and the paper says so in those words; combined < gen1 by more than the interval →
order matters, mixing base trajectories dilutes the signal, the loop does sequential work; combined > gen1 → data was
being discarded at each generation and the right procedure accumulates the corpus. The control decides whether gen1's
advantage survives at matched rows.

**Reading, arm 1 (2026-09-22 21:27Z, seed-73 independent 300, same task set as every comparator):** combined
gen0+1 head 42.7% vs gen1 40.0%: **+2.7 [−2.3, +7.7], p = 0.36** (McNemar 25/33); vs gen0 +8.3 [+3.3, +13.3],
p = 0.0015; vs base +22.0 [+16.0, +28.0]. Actions 8.19 → 7.76 → 7.66 → 7.62. By the registered reading, combined ≈
gen1: more data does not extend the accumulation, and the combined head is a working head that stops where gen1
stopped. What arm 1 does not settle: whether gen1's rows are the ones that matter (origin) or the head saturates at
this row count (amount). Arm 2 — gen0's trajectories at gen1's row count — separates them and is running; the joint
reading is written only when both are on disk. The author's question — combine and apply once — is answered on this
half: combining does not beat gen1, so there is nothing to gain by merging the generations.

## 12. §10 arm 2 carriers on the v3 300 — readings, and a null replicate registered

*2026-09-22 22:0xZ. Base for every carrier: `et8_agent`, budget 12, no head, v3 300, 11.7% green at 11.10 actions,
run in this session (not imported).*

- **A_memory (text memory carrier):** 11.7% → 12.3%, **+0.7 [−3.7, +5.0]**, p = 0.88, McNemar 22/24. Instrumented
  (every episode carries `memory: true`, base `memory: false`). A real null: the carrier does not move the set.
- **F_logitbias: VOID, not negative.** The bias table is keyed by the v1 family names and every v3 episode carries
  `family: 'v3'`, so the agent ran unbiased on all 300 (log: `no bias row for family 'v3' — running UNBIASED`).
  Ruling: §7.3's F is **reported as untestable on v3** — its instrument is keyed to a task taxonomy this set does not
  use — and no hand-made mapping of `bug_class` onto the four families is admitted, because the mapping would decide
  which bias each episode gets. A v3-native logit-bias carrier, if ever built, is a different row and says so.
- **The void arm re-labelled as what it is: a null replicate.** Same model, tasks, budget, inert flag — a second base
  arm. base vs base: +0.0 [−1.3, +1.3] success, −0.03 [−0.09, +0.01] actions, McNemar 2/2, `tokens_out` differs on
  242/300 (greedy decoding on this stack is not bit-reproducible run to run). **The harness's run-to-run noise floor
  on the v3 300 is ±1.3 points**, four discordant pairs from nothing. Nobody registered it; it ran by accident; it is
  kept with that label.
- **Registered now, before it runs: a null replicate on seed 73's independent 300.** The accumulation row there is
  +5.7 [+0.7, +10.7], whose lower bound is half the v3 noise floor. A noise floor is not a bias and the row stands as
  paired; but the reader will ask, and the answer is measured, not argued: one further base arm on seed 73's set, paired
  against the existing base. Reading fixed: if the null's interval is within ±2 points, the +5.7 row is reported beside
  it unchanged; if the null's interval reaches ±4 or more, every seed-73 row whose lower bound is inside the null's
  interval is marked "within the noise floor" in the paper and the row count for "accumulation reproduced" is re-read.
- **naive_symptom:** running; mechanism verified live from the step log (33/33 episodes differ from base in
  `tokens_out`), paired when complete, reported against the base and against the null floor.

**Joint reading, §11 (2026-09-22 22:33Z, both arms on seed 73's independent 300, wrapper pairings identical to the
read-only ones):** gen0-at-gen1-rows lands at 37.7% — between gen0 (34.3%) and gen1 (40.0%) and compatible with both:
+3.3 [−1.0, +7.7] over gen0, −2.3 [−7.7, +2.7] under gen1. Combined (arm 1) +2.7 [−2.3, +7.7] over gen1. Both
intervals span zero. **The decomposition of the +5.7 into amount and origin is not established at 300 tasks.** The
point estimates read as "mostly amount" and that sentence is not written: the interval on the 3.3 contains zero and
contains 5.7. Against the v3 null floor of ±1.3 these ±4–5 intervals are not noise; the arm is underpowered for a
five-point effect split in two. Sized: a half-width of 1.5 on the origin component needs about 2,500 tasks (8.4×),
of 1.0 about 5,700 (19×). **Ruling: §11 is reported as it stands — the accumulation is real and its decomposition is
not established at this n — and nothing further is spent on it unless the author wants the mechanism sentence badly
enough to pay 8×.** The author's question is answered on both halves: combining does not beat gen1; whether gen1's
advantage is about which trajectories or how many cannot be told at this size.

**§12 null replicate on seed 73 — FIRED (2026-09-22 23:01Z).** Base against base, same box, same command, same
code path (the two intervening commits touch a recorded field and a branch a no-heads run never enters), same mlx,
same task bytes (md5 per slice, files dated before the first run): **+5.0 [+1.7, +8.3], p = 0.0059, McNemar 6/21**,
ind73_base 20.7% → null73 25.7%. The registered ±4 branch applies (the interval reaches +8.3). Two readings of the
rule are printed, because the rule as written names the lower bound and the stricter reading names the point:

- *as written* (lower bound inside [+1.7, +8.3]): `s11_combined_vs_gen0` +8.3 [+3.3, +13.3] → marked WITHIN THE NOISE
  FLOOR.
- *by point estimate* (E's reading, stricter and adopted alongside): `ind73_gen1_vs_gen0` **+5.7 [+0.7, +10.7]** — the
  accumulation row — `ind73_gen2_vs_gen0` +3.7, `s11_combined_vs_gen1` +2.7, `s11_gen0matched_vs_gen0` +3.3, and
  `s11_combined_vs_gen0` +8.3 → all marked WITHIN THE NOISE FLOOR until the null is characterised.

Untouched, by a wide margin: every row against base — gen0 +13.7, gen1 +19.3, gen2 +17.3, combined +22.0,
gen0matched +17.0, p from 5.7e-06 to 7.4e-12 — all far outside [+1.7, +8.3]. **The head's effect over no head is not
in question. What the null reaches is every between-generation comparison, which is exactly where §4's accumulation
claim lives.** The "accumulation reproduced" count is re-read as: reproduced on two 300s under pairing, and on seed 73
not distinguishable from one draw of the loop's own run-to-run variance.

This is one draw from the null distribution, not its characterisation. It also contradicts the v3-300 null of the
same afternoon (+0.0 [−1.3, +1.3]), which is informative: `et8_agent` (one decision per episode) is nearly
deterministic; `et8b_loop` (up to three head decisions at budget 12) is not. **Run-to-run variance is a property of
the loop harness, and every VIII-b generation row is measured on the loop.**

**Registered now, before they run: two further null replicates on seed 73 (600 episodes), same configuration.**
Reading fixed: the three null draws give a distribution; a between-generation row is reported as a finding only if its
point estimate lies outside the range of the null draws and its lower bound exceeds the largest null point estimate;
otherwise it is reported beside the null draws as "within run-to-run variance" and the accumulation claim is restated
as not established on seed 73. A time-order check is run beside them: whether the later of two identical runs scores
systematically higher (the sign of +5.0 is the sign of the accumulation rows, and the generations were run in time
order). **Exposure beyond VIII-b, to be checked and not assumed:** Paper VIII's independent reproduction
(+8.0 [+3.0, +13.3]) — if it was measured on the loop harness with base and head as separate runs, its lower bound
sits inside this null's interval, and VIII's reproducibility section gains a sentence once the null is characterised.
VIII's headline rows (matched compute, R6, R7) are paired within one run pair and far outside the interval.

**§12 carriers, final (2026-09-22 23:39Z), v3 300 against base 11.7% at 11.10 actions:** A_memory +0.7 [−3.7, +5.0]
p .88; naive_symptom +0.3 [−0.7, +1.7] p 1.0 (mechanism verified from the code path `et8_agent.py:262` and the
flag's reach, NOT from episode-level token diffs — the void F arm also differs from base in `tokens_out` on 242/300 with
nothing loaded, so token diffs are downstream of the loop's own nondeterminism and prove nothing; the standard is the
code path and the run's own log); F_logitbias void (§12 above). All three inside the v3 null floor of ±1.3: §7.3's
carriers do not move the 300-problem set, as a bounded statement. The contrast is the instrument finding of the day:
`et8_agent` (one decision) nulls at ±1.3; `et8b_loop` (up to three head decisions) nulls at +5.0 [+1.7, +8.3].
Null73b is complete and null73c running — four base arms, six pairwise draws, ~00:10Z — started by E with the box
idle and the ±4 branch already fired; approved after the fact as controls that touch nothing published.

**§12 — the null is not flat in time (2026-09-22 23:46Z).** Three draws: null73 vs ind73_base (41 h apart) +5.0
[+1.7, +8.3]; null73b vs ind73_base (42 h) +6.3 [+2.3, +10.3]; null73b vs null73 (43 min) **+1.3 [−2.3, +5.0],
p = 0.58**. The draws closest in time agree and the furthest differ most. Mechanical causes eliminated (both
intervening commits read in full; mlx unchanged since Aug 13; weights since Sep 14; same box, command, task bytes).
Three draws cannot separate "a wide distribution that put ind73_base low" from "a step between the dates"; null73c
(~00:10Z) is the fourth point. **Regime of each row, from the run ids:** ind73_base, gen0 and gen1 ran in one
continuous session on 09-21 (base 04:46–05:11, gen0 05:19–06:01, gen1 06:16–06:58Z), so **the accumulation row
gen1 vs gen0 is a within-session comparison, 57 minutes apart**, whose relevant null is the one within-session draw
(+1.3, p .58) — a materially weaker challenge than +5.0, not a clearance (n = 1). gen2 (09-21 15:37) is an
intermediate regime. **Every §11 comparison is cross-date** (both arms 09-22, all comparators 09-21): re-paired against
the same-day bases, combined vs base moves +22.0 → +17.0/+15.7 and gen0matched vs base +17.0 → +12.0/+10.7 — arithmetic
by construction, not new evidence, and the two rows the decomposition rests on (combined vs gen1, gen0matched vs gen0)
have no same-day comparator at all. **§11 is exposed to the cross-date step; §4's accumulation row is not.**

**Paper VIII's independent reproduction: not exposed.** `reproduce_matched_compute.sh` invokes `et8_agent.py` (one
decision per episode) for both arms; its +8.0 [+3.0, +13.3] sits on the instrument whose null is ±1.3, and its lower
bound is well outside. Two separate runs, different machine, budget 16 vs 12 — the right family, not an identical
configuration. VIII gains no sentence.

**Registered now, to run after null73c on one box in ONE CONTINUOUS SESSION (~1,800 episodes, about three hours):**
`base_a → gen0 → gen1 → combined → gen0matched → base_b`, same seed-73 300, same code. Readings fixed: (i) base_a vs
base_b is the within-session null draw that brackets every arm; (ii) every §11 row is re-read within this session
against its within-session comparator; (iii) the accumulation row is re-read as gen1 vs gen0 within this session and
reported beside the 09-21 row; (iv) a between-generation row is a finding only if its point lies outside the range of
all within-session null draws (base_a vs base_b here, null73b vs null73) and its lower bound exceeds their largest
point; (v) the cross-date step, if null73c confirms it, is reported as a property of the harness — comparisons are
made within a session or not at all — and every VIII-b table names the session of each arm.

**§12 — fourth draw (null73c, read by R from E's scratchpad 2026-09-23 00:5xZ, read-only; E to commit the artefact):**
null73c 22.0%. Four base arms on the same 300: **20.7 · 25.7 · 27.0 · 22.0**. Pairwise: null73c vs null73b (41 min
apart) **−5.0 [p = 0.020, McNemar 26/11]**; null73c vs null73 −3.7 (p = 0.052). So the within-session reading of
23:46Z does not hold either: two runs 41 minutes apart differ by five points with p = 0.02. **The story is "wide
distribution", not "step between dates."** The loop's run-to-run spread on seed 73 at n = 300 reaches ±5 points in
either direction, within an hour. Registered rule (iv) applied: the accumulation row +5.7 [+0.7, +10.7] lies inside
the range of the null draws (−5.0 … +6.3) and its lower bound does not exceed their largest point → **NOT A FINDING
on seed 73.** Every between-generation row is within run-to-run variance. Head-over-base rows (+13 … +22) remain far
outside the range and stand. **The one-continuous-session plan registered at 23:5xZ is WITHDRAWN as insufficient
(one more draw of a wide thing) and replaced, before it runs, by a replicate design:** three replicates each of base,
gen0 and gen1, interleaved in one session (`b1 g0a g1a b2 g0b g1b b3 g0c g1c`, 2,700 episodes, ~4.5 h, seed-73
300). Readings fixed: the accumulation effect is the mean of the three gen1-minus-gen0 paired differences, its
interval from the three replicates (t, df 2) and from the pooled per-task pairing; the null is the three base
replicates' pairwise spread; the row is a finding only if the replicate-mean interval excludes zero AND exceeds the
largest base-vs-base draw. If it is not a finding, VIII-b's accumulation claim is reported as not established at this
n on this harness, and the paper's subject becomes what it has actually measured: the loop's noise floor, the
unbuildable matched-compute arm, the constant bridge arm, the MLP head, the null carriers — the negative space around
VIII's positive result.

**§12 — six draws (E, 2026-09-23 01:00Z), confirming the fourth:** four bases 20.7 · 25.7 · 27.0 · 22.0 (mean 23.85,
sd 2.98, range 6.3). Draws: +5.0, +6.3, +1.3, +1.3, −3.7, −5.0; time separation does not predict magnitude (33 min gave
−5.0, 43 min +1.3, 41 h +5.0). E's 23:46Z "close regime" reassurance withdrawn by E. **Instrument finding: four of six
null draws reach p < 0.06 under McNemar on 300 paired tasks** — the test treats run-to-run variance as zero and calls
nothing-at-all significant two times in three. Every p-value on a VIII-b loop row is from that test and is therefore
not a measure of anything but within-run pairing; the replicate design's interval (across replicates) replaces it as
the reported uncertainty, and VIII-b prints McNemar only as a within-run descriptive. Author's ruling (2026-09-23):
treat the loop like a chess match — many games, average, track bias, report "ahead of noise by X over k games", never
"clear winner". Author also approved the 8× §11 arm ("8x. yes"); sequenced AFTER the replicate match and only if the
accumulation row is ahead of noise there, since a decomposition of an effect not yet established measures nothing.

**§12 — the tight instrument's null, four base arms (2026-09-23 04:24Z):** `et8_agent` on v3 300: 11.7 · 11.7 · 12.3
· 11.7 (mean 11.83, sd 0.33, range 0.7); six draws −0.7 … +0.7, every p > 0.68. Against the loop's five base arms on
seed 73 (20.7 / 22.0 / 25.7 / 27.0 / 27.7; sd ≈ 3.0, range 7.0). §10 carrier rows now read against a null range of
±0.7: A_memory at its edge, naive_symptom inside, F one of the null arms — the carriers do not move the set, bound
under one point; the limiting factor on those rows is their own paired interval at n = 300, not the instrument.
**Attribution NOT established:** the two comparisons differ in task set (v3 vs ind73) as well as harness, so "the loop
is the noisy instrument" is a two-points-at-different-settings reading. E is running the tight harness on the loop's
task set (`et8_agent` on ind73, two arms) to separate them; registered reading: spread near 0.7 → the task set is
innocent and the loop's head decisions carry the variance (a design property, reducible); spread toward 7 → the task
set carries it and no loop redesign helps. Either answer conditions how the match's result is read.

**§12 — separation arm, first draw (2026-09-23 07:21Z):** `et8_agent` (one decision) on the LOOP's task set (ind73):
two base arms 10.7 / 9.7, −1.0 [−2.7, +0.3], p = 0.38 — the v3 neighbourhood (0.7), not the loop's (5–6). Reading, by
the registered rule: the task set is largely innocent and the variance is a property of `et8b_loop` — a design
property, therefore potentially reducible. Mechanism offered by E as a hypothesis to test, not a finding: three head
decisions per episode are three branch points; one decision → sd 0.33, three → sd 3.0; prediction: a loop restricted
to one decision shows the tight spread. **One draw; two more arms running (four arms, six draws) before the separation
is called established** — the same discipline the first null taught. `experience/match_analysis.py` is committed
before the match's numbers exist.
