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

## 13. Pre-registered 2026-09-23 07:5xZ, before the ninth match arm is read — is the accumulation limited by the INDEPENDENCE of the experience? (the author's hypothesis)

The author, reading two games of the match (gen1 − gen0 = +2.3, +3.7 against base-vs-base draws up to 2.7): gen1's
experience was generated by an agent already steered by gen0, on the same problems and bug families, so its rows are
a re-sample of the corner gen0 already found, not a second independent sample; a narrow margin is what correlated
experience looks like, as with training data whose marginal value is its novelty. §11 (combined ≈ gen1) is consistent
with this. Two tests, registered with readings fixed:

**13a. Redundancy of gen1's rows relative to gen0's — read-only, on existing artefacts, no GPU.** In the head's own
feature space (the layer-18 state at each recorded decision), for every gen1 training row find its nearest gen0 row;
report the distribution of nearest-neighbour distances against the within-gen0 nearest-neighbour distribution, and the
fraction of gen1 rows whose nearest gen0 row is closer than the within-gen0 median ("redundant rows"). Also the plain
count: the fraction of gen1's rows that come from problems gen0's agent already solved. Readings: redundant fraction
≥ 0.5 → gen1's effective experience is at most half its row count and the narrow margin is explained by correlation;
< 0.25 → gen1's rows are largely novel and the margin is not a redundancy effect. Between → reported as measured.
Same for gen2 against gen0 ∪ gen1. Two numbers, one figure, no claim beyond them.

**13b. Independence as the variable — one arm, equal rows.** gen1′: a head fitted on experience of the SAME row count
as gen1 but collected on a DIFFERENT task family (the SQL-repair set of VIII's R7, or a disjoint bug-family slice),
applied to the seed-73 300 in the same loop; paired against gen0 and against gen1, three replicates each interleaved
in one session as in §12 (`g0 g1 g1′` × 3, with a base bracket), read against the session's base-vs-base draws.
Readings: gen1′ − gen1 > 0 with the replicate interval excluding zero and exceeding the largest base draw → the lever
is the independence of the experience, not its amount, and the paper says: accumulation pays in proportion to the
novelty of the new experience; same-domain self-generated experience is mostly redundant after one generation.
gen1′ ≈ gen1 → domain of origin does not matter at this row count; the redundancy number of 13a stands alone.
gen1′ < gen1 by more than the draws → same-domain experience carries something transfer does not, and 13a's
redundancy is not the limit. Every outcome is printed. Sequenced AFTER the match block (§12) and the four-arm
separation; 13a may run at any time since it touches no GPU.

**§13a — MEASURED (2026-09-23 07:58Z, `ce47d9e`, read-only on the recorded rows, no GPU):**

```
                                    gen1 vs gen0      gen2 vs gen0 ∪ gen1
  EXACT duplicate rows (byte-equal state)   49.2%             85.9%
  neighbour redundancy (closer than the
    prior set's own median self-NN)         67.2%             90.1%
  same (task, decision, candidate)          98.8%             99.5%   (partly by construction — cross-fit on tasks 1–225)
  median nearest-prior distance             0.0213            0.0000
```

By the registered reading (redundant fraction ≥ 0.5 → the margin is explained by correlation): **gen1's rows are half
redundant with gen0's; gen2's are five sixths redundant with gen0 ∪ gen1, on a stricter bar** (the union is denser,
median self-NN 0.0416 vs 0.0830). An identical hidden state is an identical history prefix: the steered agent stood
where the previous agent stood and produced the same activation. **Redundancy rises with generation, 49% → 86%, and
the gains fall with it: gen1 adds ~51% new rows → a small gain (+5.7, inside the loop's noise by §12); gen2 adds ~14%
new rows → none (−2.0). Accumulation stops because the experience stops being new.** The author's hypothesis is
measured, and it explains §11 without the decomposition: gen0 ∪ gen1 is barely larger in unique content than gen0, so
combined ≈ gen1 is the prediction. E's definitional substitution is recorded: "problems gen0's agent already solved" is
circular here (gen1's rows ARE that agent's trajectory), so SEEN-BY (task-id overlap) was implemented and named in the
module docstring. **§13b is now the decisive arm and carries a prediction made before it runs:** a gen1′ head from a
different task family should not be half-duplicated, so if independence binds, gen1′ > gen1; if not, the hypothesis is
wrong in a way more data cannot fix.

**§12 — THE MATCH (2026-09-23 08:28Z, `3eaeb1a`, nine arms interleaved in one session, 2,700 episodes, 300/300 each, no
duplicates):**

```
  b    27.7  25.7  25.0    mean 26.11   sd 1.39
  g0   37.0  36.7  36.3    mean 36.67   sd 0.33
  g1   39.3  40.3  39.3    mean 39.67   sd 0.58
  means of three:  b→g0 +10.56 [+7.01, +14.10]   g0→g1 +3.00 [+1.78, +4.22]   b→g1 +13.56 [+10.79, +16.32]
  nine single g1−g0 pairings: +2.33 +3.33 +2.33 +2.67 +3.67 +2.67 +3.00 +4.00 +3.00  (all positive; min g1 39.3 > max g0 37.0)
  within-arm null draws:  base −2.00 −2.67 −0.67 (sd 1.39) · gen0 −0.33 −0.67 −0.33 (sd 0.33) · gen1 +1.00 0.00 −1.00 (sd 0.58)
```

**Verdict under the registered rule (iv): NOT A FINDING.** The null set is nine base-vs-base draws (last night's six
plus the match's three), range −5.0 … +6.3; +3.00 lies inside it. Recorded as the verdict. The rule is not relaxed
after it fired.

**What the match measured about the instrument, which was not known when rule (iv) was written:** head arms are
three to four times less variable than base arms (within-arm sd 0.33 / 0.58 against 1.39, and ~3.0 across last
night's five base arms). Every null draw to date was a base-vs-base draw. **A base-arm null is the wrong yardstick for
a head-vs-head comparison: it imports a variance the comparison does not have.** This is a fact about the instrument,
measured by the match rather than chosen, and it would be equally true had the effect been zero. E raised it and
declined to apply it; R records it and declines to apply it to this data: re-reading a fired rule against a
yardstick found after the reading is motivated search, whatever the yardstick's merits.

## 14. Re-registered yardstick — applies ONLY to data not yet collected (2026-09-23 08:4xZ)

For a comparison between two HEAD arms, the null is the within-head-arm draw set (head-arm replicates of the same
configuration in the same session), not base-vs-base. A between-generation row is a finding only if (a) the
replicate-mean interval excludes zero and (b) its lower bound exceeds the largest within-head-arm null draw of that
session; base-vs-base draws are reported beside it as the unsteered agent's variance, not as the bar. **The first match
is reported under rule (iv) as not a finding and as the measurement that revealed the yardstick; it is not re-scored.**

**Confirmatory test:** §13b, as designed (`g0 g1 g1′` × 3 interleaved with a base bracket, seed 73), is ALSO the
confirmatory replication of accumulation: its three g1 − g0 games are scored under §14 on data that does not yet
exist. Readings fixed: §14 (a) and (b) hold on 13b's g1 − g0 → accumulation is reported as established over six
games (three from each match) with the first match's verdict printed beside it; either fails → not established, and
the first match's all-positive nine pairings are reported as suggestive and nothing more. Beside it, 13b measures
within-arm sd for every arm including g1′, which tests the second new claim — **the head stabilises the run, not only
raises its mean** — stated here as a hypothesis, scored in 13b as: head-arm within-sd < base within-sd in that session,
for all three head arms. gen1′'s duplicate fraction against gen0 is measured by the §13a instrument before its head is
scored. Order: separation arms (running) → 13b. No other VIII-b run.

**§13b form ruling (2026-09-23 13:0xZ, before any gen1′ head is fitted):** R7's SQL states are single-decision and
carry no `decision` field; the loop consumes three per-decision heads, so gen1′ cannot take gen1's form from what
exists. Ruled: **option (i)** — gen1′ is ONE shared head (`--head`) fitted on all 1,437 SQL rows, and gen1-matched is
ALSO one shared head fitted on 1,437 gen1 rows (fixed-seed subsample, ids recorded, `--cv-folds` as the original gen1
fit). g1′ vs g1-matched is then like-for-like: amount fixed, form fixed, origin varied — the §13b question. **The bonus
row g1 − g1-matched is DROPPED**, not reported: it would confound three-heads-vs-one-head with 2,824-vs-1,437 and is
uninterpretable. The form difference between gen1′ and the published gen1 is printed in the artefact and the paper,
so nobody later quotes a form effect as the origin effect. Option (ii), regenerating SQL states from a loop run, is
not spent: it keeps a row that was never the point. **Boxes:** the box effect is zero within 0.17 on the tight
instrument (box A 35/37 inside box B's 35–37) — measured because a cross-box result had already been shipped assuming
it; NOT measured under the loop, and not extrapolated. §13b stays on box B in one session, because interleaving in one
session was the design. **box A, idle, gets a separate registered arm — the one-decision-loop test of the variance
mechanism:** `et8b_loop` restricted to ONE head decision per episode, two base arms and two gen0-head arms on seed
73's 300; reading fixed: within-arm sd near the tight instrument's 0.4 → the variance is the number of decisions (a
design property, reducible); near 2.6 → it is the loop's machinery regardless of decisions. Independent of §13b;
touches nothing published.

**§12 — the variance IS the decisions (read by R from E's boxes 2026-09-23 16:5xZ; E to commit the artefacts):**
one-decision loop, 4 base + 4 gen0-head arms, 300 each: base 30/30/30/30, head 50/50/50/50 — **bit-identical
across runs on every field of every episode; sd exactly 0, not "near 0.4"** — the loop at one decision has no
distribution. Same-box determinism check at three decisions (one slice, n = 75, back to back): same green 67/75,
same actions 70/75, same tokens_out 10/75 → **non-deterministic on one box; the divergence enters with decisions 2 and
3** (E's alternative (b), a box effect, is excluded). The 2.6 sd is a property of the loop's later decisions — carried
state and longer context — a design property and reducible. The first decision is deterministic at temperature 0.

**§13b — game 1, all five arms (box B, read 16:5xZ):**
```
  b1x 24.7 · g0x 33.0 · g1x 38.3 · g1′x 23.3 · g1-matched x 37.3      (game 2 partial: b 24.0 · g0 35.3 · g1 running)
```
g1 − g0 = +5.3 [+0.0, +10.7] (§14 not cleared on one game; three games decide). **g1′ (the SQL-family shared head) sits
AT BASE LEVEL — 23.3 against 24.7 — and g1-matched (1,437 gen1 rows, one shared head) sits within a point of the full
gen1.** Read: a head fitted on another task family does not steer a debugging loop at all; the "independence" arm as
designed was a DOMAIN-SHIFT arm, and it answers a question the author did not ask. **R's design error, named:** the
author's hypothesis is about independent experience of the SAME task family (new problems, not a re-sample of the
corner); §13b operationalised independence as a different family, which confounds novelty with domain. §13b still
yields the confirmatory replication (g1 − g0 under §14) and the stabilisation test, and it is not stopped.

## 13c. Registered now, before §13b's games 2–3 finish — the correct independence arm
gen1″: a head fitted at 1,437 rows on gen0-steered trajectories from a DISJOINT set of DEBUGGING problems (the same
family; e.g. the seed-47 300 or a fresh 300 of the same generator), applied to seed 73 in the loop; three games
`g0 g1-matched g1″` interleaved in one session with a base bracket. Prediction, written before the run: gen1″ >
gen1-matched (both intervals excluding zero, clearing the largest within-head-arm draw) if independence of experience
within the family is what extends accumulation; gen1″ ≈ gen1-matched → same-family novelty does not help at this row
count and the §13a redundancy is a description, not the mechanism. gen1″'s duplicate fraction against gen0 is measured
by the §13a instrument before its head is scored (expected near 0 for disjoint problems). E names which disjoint
same-family trajectories exist on disk; if none, one gen0-steered collection run on a disjoint 300 (~1 h) precedes it.
**§13c form and source, fixed 2026-09-23 17:0xZ before the head exists:** gen1″ is fitted as ONE SHARED head by
`fit_shared_head.py`, same layer, same fitter, same CV as g1-matched — three per-decision heads would reintroduce the
form confound that made the §13b bonus row undrawable. Source: a fresh same-family draw (v3 generator, seed 47,
n = 300), verified disjoint BY CONTENT (program + symptom hash, never task id) from gen1's fitting problems (∩ v3 = 0)
and from the evaluation set (∩ ind73 = 0); gen0's own heads steer the collection; states at layer 18 by the same
extractor; rows subsampled to exactly 1,437 at a fixed seed with kept indices recorded, the identical procedure
g1-matched went through. The on-disk alternative (894 disjoint rows of r8b_gen1states) was rejected because it breaks
the matched amount. Collection started 16:59Z on box A (E, decide-do-inform); §13c's three games run on box A in their
own session as soon as the head and its §13a duplicate check exist — they do not wait for §13b on box B.

**§13c — head fitted, games running (2026-09-23 18:27Z), two numbers recorded BEFORE any game result:** (1) the
precondition holds — gen1″'s rows are 1.3% byte-identical to gen0's (against 49.2% for gen1), so disjoint problems
bought new rows; but 42.0% sit closer to gen0's corner than gen0's rows sit to each other — new rows, largely the same
region of state space: the first measurement that separates "new rows" from "new experience". (2) the head is weak:
CV 26.7% (chance 20.1) against gen1-matched's 35.2%, and its 500-row subsample control reads 27.3 — a head that is not
using the examples it was given (permutation collapses, PCA-8 sits below; not leakage). **Reading of a null, fixed
now:** g1″ − g1-matched ≈ 0 is NOT read as "independent experience does not transfer"; it is read as NOT SEPARABLE
between (a) independence not transferring and (b) 1,437 rows of this experience not fitting a head. The artefact and
the paper carry the 26.7/27.3 line beside the game result. A positive g1″ − g1-matched is read as (a) refuted in the
favourable direction only if the fold interval clears §14. The games run regardless: what a weak head does in the
loop is itself a row. Diagnostic registered before the games finish (read-only, E): label base rate (green fraction)
and rows per decision for the gen1″ collection against the gen1 collection — if the disjoint problems were solved
far less often under gen0's heads, the head has fewer positive labels and (b) has a cause. **§13d, contingent:** if
(b) is the cause, a second disjoint collection (~1 h) to fit gen1″ at gen1's full row count against a shared-head
gen1 at the same count — amount matched at the higher level — run only after the diagnostic, not before. Timings:
§13c games ~21:05Z; §13b games 2–3 ~22:50Z.

**§13c — diagnostic (E, 18:32Z): R's (b)-has-a-cause hypothesis REFUTED at the collection level, and the real cause
found in the matching procedure.** Collections are near-identical (gen1 v3: 3,186 rows / 649 decisions / 20.4%
positive; gen1″ disjoint: 3,838 / 769 / 20.0%; every decision has a positive in both). The weak head came from RANDOM
ROW SUBSAMPLING: the head is fitted per candidate but scored per decision, and dropping rows splits a decision's
candidates — gen1-matched kept a positive in 53.7% of its decisions, gen1″ in 41.1%, and the probe accuracies (35.2,
26.7) track that column; gen1′ (92.7) was never subsampled. **Rule, adopted: the unit of "amount" is the DECISION
(a whole candidate group), never the row.** Re-cut by decision (`subsample_by_decision.py`, whole groups, fixed seed,
indices recorded): gen1-matched 1,435 rows / 290 decisions → probe 60.7%; gen1″ 1,447 / 290 → 53.4%, controls clean
(gen1-matched's subsample control still edges its head, 61.4 vs 60.7, read as early saturation, printed). E stopped
§13c's games after one base slice and restarted them on the corrected heads at 18:32Z — approved: the design is
untouched and the correction is strictly closer to what was registered. §13d is NOT collected; the row count was never
the problem. Descriptive, before the games: the origin effect on the probe is 7.3 points (60.7 vs 53.4) at matched
decisions; the loop result decides.

**Consequence for §13b (running on box B, not touched):** its g1-matched arm is the row-shredded head (probe 35.2%).
The pair g1′ − g1-matched is therefore CONFOUNDED by the matching unit and is not read as an origin effect. The
confound works against g1-matched, and g1-matched still beat g1′ by 14 points in game 1 (37.3 vs 23.3), so the
domain-shift conclusion — a head from another task family does not steer this loop — is conservative and stands
without a re-cut; no further box time is spent on §13b's g1′ pair. §13b's confirmatory g1 − g0 replication and the
stabilisation test use intact heads and are unaffected. Open question for E: were §11's gen0matched and combined
heads cut by row? If so, §11's "not established" reading stands but carries the same confound and says so.
**§11 is clean (E, 18:39Z):** its gen0-at-gen1-rows arm sampled BY TASK, keeping every decision's candidates together
and the held-out range whole; its artefact states the exact failure mode §13b's row cut fell into. No caveat, no
re-cut; §11's numbers stand. Recorded for the reason E gave: the constraint existed as a comment inside one script
three weeks ago and did not travel to the neighbouring arm — `subsample_by_decision.py` is now the shared file, which
is the only form that travels. ETAs measured: §13c ~21:10Z, §13b games 2–3 ~22:55Z.

**§13b game 2 (2026-09-23 19:51Z) — the confirmatory replication FAILS §14 in both games so far.** §14(b)'s bar, from
the data: largest within-head-arm draw = 2.33 points. g1 − g0: game 1 +5.3 [+0.0, +10.7] (a fails, lower bound at
zero); game 2 +1.7 [−3.3, +6.7] (a fails). Against base everything clears both (a) and (b): g0 +8.3 / +11.3, g1
+13.7 / +13.0, g1-matched +10.7; g1′ (the other-family head) −0.3 [−7.0, +6.3] — at base, the domain-shift conclusion
as a number — and g1′ − g1-matched = −11.0 [−18.0, −4.0] with the matching confound working against g1-matched.
Stabilisation: base is the TIGHTEST arm on two draws (sd 0.47 vs g0 1.65, g1 0.94) — the opposite of the registered
expectation, and two draws are a range; three make it a reading. **Reading fixed now, before game 3 (~23:30Z):** the
third game cannot rescue an interval that spans zero twice; under §14 accumulation on seed 73 is NOT ESTABLISHED. What
the paper reports instead, and it is the honest sentence: gen1 beat gen0 in every paired game played on the loop — the
nine first-match pairings (+2.3 … +4.0) and games 1–2 here (+5.3, +1.7) — by about three points on average, an effect
real in sign and small in size, not separable from the head arms' own run-to-run spread at 300 tasks per game; §13a
says why it is small (half the experience is redundant) and §13c is measuring whether new experience of the same
family changes that. **Registered question for E before game 3:** at the measured within-head-arm sd, how many
paired games would make a +3 effect clear §14 (interval lower bound > 2.33)? That number is the price of
"established", and the author decides whether to pay it or publish "consistent, small, not established".
ETAs measured: §13b game 3 ~23:30Z; §13c ~04:41Z (head arm 57 min on box A). Ledgers: per-run, no venue/customer/
subject fields (R468 weight stays low). Exposure window of the public history: outer bound the repo's creation
(2026-07-09); the API has no visibility history.

**§13c game 1 (2026-09-23 23:26Z), heads cut on whole decision groups:** base 77 · gen0 107 (+10.0 [+3.7, +16.3]) ·
gen1-matched (same problems) 106 (+9.7 [+4.0, +15.3]) · gen1″ (disjoint problems) 103 (+8.7 [+3.0, +14.3]);
gen1″ − gen1-matched = −1.0 [−7.3, +5.3], §14(a) not separated; (b) not evaluable on one game. All three heads beat
base by nine to ten points and are indistinguishable from each other on one game. The probe gap (60.7 vs 53.4, seven
points in favour of same-problem experience) did not convert to episodes — the fourth time in the programme a probe
improvement failed to convert, and the reason the bar is problems solved on a paired run. Games 2–3 ~04:40Z.

**§13b — THREE GAMES COMPLETE (box B, 23:54Z; scored by R from the episode counts, E's block to confirm):**
```
  game     b     g0     g1    g1-matched  g1′(SQL)
    1    24.7   33.0   38.3     37.3       23.3
    2    24.0   35.3   37.0     34.7       23.7
    3    27.7   33.3   41.7     35.7       22.7
  g1 − g0   +5.3  +1.7  +8.3    mean +5.11  [−3.18, +13.41]   (t, df 2)    §14(a) FAILS
  g0 − b                        mean +8.44  [+1.40, +15.49]   §14(a) ✅
  g1 − b                        mean +13.56 [+12.29, +14.82]  §14(a) ✅ (b) ✅
  g1-matched − b                mean +10.44 [+4.63, +16.26]   ✅
  g1′ − b                       mean −2.22  [−8.33, +3.88]    at base
  g1′ − g1-matched              mean −12.67 [−16.46, −8.87]   other-family head loses, cleanly
  within-arm sd: b 1.95 · g0 1.26 · g1 2.40 · g1m 1.35 · g1′ 0.51;  largest within-head-arm draw 4.67
```
**Verdict, under the rule fixed before any game ran: accumulation on seed 73 is NOT ESTABLISHED.** Three games, all
positive, mean +5.1, interval spanning zero. The stabilisation hypothesis (head arms tighter than base) is NOT
supported: base sd 1.95 sits between the head arms' 1.26 and 2.40. What is established: the head over base
(+8 to +14, every arm, every game); the other-family head at base level (−2.2), and 12.7 points below the same-family
head at matched rows — the domain-shift result, conservative because the matching confound worked against
g1-matched. **The paper's sentence:** across the two matches, gen1 beat gen0 in every one of twelve paired games
(+1.7 … +8.3, pooled mean about +4); the effect is real in sign, small, and not separable from run-to-run variance at
300 tasks per game; §13a says why it is small; §13c is measuring whether same-family novelty changes it. The power
computation (games needed for +4 to clear the within-head-arm bar) is E's to supply with the block.

**E's block confirms R's scoring (00:35Z, `e10f423`); verdict stands. §14(b)'s defect, recorded (E, 00:40Z):** the bar
is a MAXIMUM over within-head-arm draws (4.67 here), so it grows as games are added and a +4 effect cannot clear it at
any n; +6 clears at n = 6, +8 at n = 2. The rule did what it was written for tonight — it stopped game 3's +8.3 from
being read as a replication when the session's own spread reached 4.67 — and the verdict under it is not revisited.

## 15. Registered for FUTURE data only — the bar becomes a statistic
For any between-generation comparison scored after this entry: (a) the replicate-mean interval (t, df = games − 1)
excludes zero; (b) its lower bound exceeds the POOLED within-head-arm sd of that session (1.62 on §13b's ten draws),
not the largest draw. At that bar a +4 effect clears at about nine games (~16 box-hours). §13b (three games) and the
first match are reported under the rules that governed them, with §15 stated as the rule any further games would use.
**Ruling on spending those games: not now.** The author's arc for VIII-b does not need "established": it reports the
small consistent gain exactly, the redundancy that explains it, and what new experience does (§13c, ~04:40Z). If §13c
shows same-family novelty extends the gain, the nine-game test is the one worth paying for, on that head.

**§13c games 1–2 (E, 02:11Z):** base 77 / 77 · gen0 107 / 110 · g1-matched 106 / 109 · gen1″ 103 / 98 (of 300).
gen1″ − g1-matched = −1.0 [−7.3, +5.3] and −3.7 [−9.7, +2.3] — not separated, twice; every head beats base by
+7 to +11 and clears both conditions in both games. The registered prediction (gen1″ > g1-matched if novelty binds)
is NOT supported on two games: the point estimates lean the other way and both intervals span zero. The probe pointed
the other way again (60.7 vs 53.4) — the fourth non-conversion. Base on box A: 77, 77, 76 — range 1.0 at three
decisions, "very tight, not identical" (E declines to call it determinism). **§14(b)'s property, arriving as two bars
in one night:** this session's largest within-head-arm draw is 1.67 against box B's 4.67 — the same rule, the same
instrument, a bar three times tighter because a maximum over draws depends on which draws happened. §15's statistic
bar is the answer for future data. Game 3 ~04:45Z; the three games score together.

**§13c — THREE GAMES COMPLETE (box A, ~05:4xZ; scored by R from the episode counts, E's block to confirm):**
```
  game     b     g0    g1-matched  gen1″(disjoint)
    1    25.7   35.7     35.3        34.3
    2    25.7   36.7     36.3        32.7
    3    25.3   35.7     35.0        36.7
  gen1″ − g1-matched   −1.0 / −3.7 / +1.7   mean −1.00 [−7.62, +5.62]   prediction NOT supported
  g1-matched − g0      −0.3 / −0.3 / −0.7   mean −0.44 [−0.92, +0.03]   the gen1 shared head at 290
                                                                          decisions equals gen0
  gen1″ − g0                                mean −1.44 [−7.66, +4.77]
  g0 − b +10.44 [+9.18, +11.71] · g1-matched − b +10.00 [+8.57, +11.43] · gen1″ − b +9.00 [+3.57, +14.43]
  within-arm sd: b 0.19 · g0 0.58 · g1m 0.69 · gen1″ 2.01;  pooled head sd 1.27;  largest within-head draw 4.00
```
**Reading, under the registered rule:** the prediction that disjoint-problem experience beats same-problem experience
at matched decisions is not supported — the two are indistinguishable across three games, both about ten points over
base, and both equal to gen0's head. Same-family novelty did not extend the accumulation. The §13a numbers say why:
the disjoint rows were 1.3% duplicated but 42% sat in gen0's region of state space — new problems, largely the same
experience. Caveat carried: g1-matched vs g0 compares one shared head at 290 decisions against gen0's three
per-decision heads, so "gen1 ≈ gen0 at matched decisions" is descriptive, not a form-controlled comparison. The base
arm on this box repeated to within one problem across three games (sd 0.19), and the disjoint head was the most
variable arm (sd 2.01), the opposite of the registered stabilisation expectation. E's block confirms or corrects.

## 16. VIII-c arm 1, registered 2026-09-24 07:1xZ before any collection — a different agent on the same problems
**Question.** Does experience collected by a DIFFERENT fixed agent, on the same task family, land outside the region
gen0's policy returns to, and does a head fitted on it extend the gain that gen1 could not?
**Design.** Agent B = Llama-3.1-8B (VIII's second family; harness and fitter exist) collects on seed 73's 300 under
its OWN policy (no head), three decisions, box B. States extracted at the layer VIII used for that model. Before any
head is fitted: the §13a instrument against gen0's rows — exact-duplicate fraction (expected ~0: different model,
different activations, so exact duplication is uninformative here and is printed only for the record) and the
neighbour-redundancy fraction in the 7B's state space is NOT computable across models; instead the operative
measure is BEHAVIOURAL, at DECISION 1 only (E, 07:00Z: cells align exactly at d1 — same task, same empty state, same
candidates — and diverge by construction after it, so a rate over all cells would count different states as
different choices): the 300-cell disagreement rate between B's first pick and gen0's agent's first pick, and the
fraction of tasks agent B solves that gen0's agent did not (and vice versa). Those
two numbers are the "different region" measurement and are printed before the head.
**Heads.** (Corrected by E before the run, 07:00Z: the labels are GROUND TRUTH from the verifier — `bug_region` — for every
agent; no agent supplies labels. Arm 1 is therefore the 7B's states AT THE PLACES B's POLICY WENT, labelled by the
verifier: the escape is in which states get visited, the supervision never changes. `et8b_states --runs <B's steps>
--model <the 7B>` already separates whose trajectory from whose states.) B's decisions cut to 290 (whole groups, fixed
seed) and fitted as one shared head on the 7B's states at B's visited places — and, as the second arm, a shared head on the UNION of B's and gen1-matched's decisions at 290 total
(experience exchange). Both against g1-matched (same form, same count) and gen0, three games interleaved in one
session with a base bracket, §15 bar (pooled within-head-arm sd).
**Readings, fixed now.** (i) B solves a substantially different subset (≥ 25% of its solved tasks unsolved by gen0's
agent) → the region is the policy's, not the task's; (ii) the B-experience head beats g1-matched with the interval
clearing §15 → different-agent experience extends the gain: the escape route exists and it is cheap; (iii) the union
head beats both → exchange compounds; (iv) B's subset overlaps gen0's (< 10% different) → the region is the task's
and no policy escapes it — a finding that closes the route. Any control fires → that arm withdrawn. **Not run until
WO-318's demo has reached the owner; then it is the only VIII-c run.**

## 16a. Amendment, registered 2026-09-24 13:5xZ, after arm 1 was stopped at 33 episodes and before any reading

**What was found.** The harness's system prompt carries a hardcoded region menu from the v1/v2 task families
(`producer, transform, aggregate, consumer`). The v3 family's regions are different (`shape, emit, fold, intake,
digest, render, gather, sift`) and `state_text()` never lists a task's own candidates, so that stale line is the only
menu any agent is shown. The 7B ignores it (v3 run `v3rep/repbase`, 75 hypothesize steps: 100% on-menu, zero stale
names — it reads the region names from the `# region:` headers). Llama-3.1-8B obeys it (33 episodes, 98 hypothesize
steps: 66% stale names, 27% unparsed, 7% on-menu, 1 of 33 green). Arm 1 as registered would therefore have measured
which agent follows a wrong instruction, not which region of state space its policy visits. E stopped the run; the
partial collection is kept as `results/viiic_arm1_llama8b_STOPPED_INVALID_PROMPT.*` and is not an arm.

**Withdrawn.** E's earlier design check that "the candidate set is safe: the task's `regions` are offered to any agent
at any decision" — the candidates are used by the loop to pick inspect targets and to score head candidates; they are
never shown to the agent. That check read the loop's code, not the served prompt.

**Correction to the instrument, before any further collection.** `state_text()` lists the task's own `regions`, read
from the task record at run time; the hardcoded line is removed and no second fixed list replaces it. This is a defect
in the instrument found before any §16 reading existed; §16's readings, bar and gates are unchanged.

**Consequence for comparability.** Every prior v3 collection — gen0's base, §13b, §13c — was served the stale line by
an agent that ignored it, so those results are internally consistent and stand as recorded. They are not comparable
to a corrected-prompt collection without a corrected-prompt control, so arm 1 now consists of: (1) one fresh 7B base
on the corrected prompt (seed 73's 300, three decisions, box B, ~85 min at ~17 s/episode); (2) agent B's collection
on the corrected prompt; (3) the §16 readings computed between (1) and (2) only, never against the stale-line runs.
The paper states in one sentence, wherever the harness is described, that the 7B's v3 results were obtained under a
prompt that misdescribed the task's regions and that the 7B did not follow it.

**Order.** This section resolves on origin before (1) starts.

**16a addendum, 2026-09-24 14:5xZ, after step (1) completed and before agent B's collection finished.**
(i) Step (1) result: corrected-prompt 7B base on seed 73's 300 — 68/300 green (22.7%), mean 8.18 actions, 818
hypothesize steps 87.9% on-menu, 0.0% off-menu, 12.1% no-region; the stale-line seed-73 base read 20.7%/8.19, a
difference inside one standard error at n=300. The correction moved the vocabulary, not the difficulty. Agent B on
the corrected prompt: off-menu 66.3% → 0.0% (first 38 steps), so the correction is shown to have taken.
(ii) The column earlier called "unparsed" is renamed NO-REGION: `region` is None both when a parse fails and when the
model chose inspect/patch/run at that step; the steps log keeps no raw text, so the two are not separable.
(iii) One CONTROL collection is registered, run after agent B finishes and not before: the 7B on the same v3c tasks with
the STALE prompt (same model, same tasks, only the served line differs, ~33 min). Its sole purpose is to attribute the
no-region change between the stale `v3rep/repbase` run and (i) to the prompt line rather than the task set, which moved
together in that comparison. It enters no §16 reading; §16's readings are computed between (i) and agent B only.
(iv) A second stale vocabulary in the same system prompt: the bug-class list is v1's (12 names) while v3c's tasks use
10 classes, overlap 2. The model's `bug_class` is never scored — it appears only as a prefix in the head-scoring
string and nothing reads it back — so it misdescribes the task without touching a reading. Recorded as KNOWN AND
UNSCORED; not corrected now, because a prompt change costs another base. It is corrected in the next instrument
version before any further family runs, and the paper's one-sentence harness note covers both stale lists.
(v) Harness fix during agent B's collection, after step (1) had completed: a patch that makes the verifier hang raised
`TimeoutExpired` uncaught out of `run_tests` and unwound the collection at episode 4. Now a verifier that does not go
green within its budget is RED with a TIMEOUT record and no region — the convention the import-failure path already
used. Step (1) contained no such event and is unaffected; agent B restarted from episode 1 on the fixed loop.

**16a, second addendum, 2026-09-24 16:0xZ — reading (i) computed; two controls registered before anything is fitted.**
(vi) Reading (i): 7B (corrected prompt) 68/300; agent B (Llama-3.1-8B, corrected prompt) 34/300; both 11; B-only 23
= 67.6% of B's solved set (bar ≥ 25%): FIRES. Reading (iv) does not fire. Fallback control run before this was read:
in all 23 B-only solves B named the region itself (22 of 23 correctly); none is the loop's `regions[0]` fallback.
Region distribution where B names one is concentrated (gather 31, sift 19, digest 18, fold 11, shape 5, intake 4,
emit 2, render 0) against the 7B's near-uniform spread. The registered 300-cell decision-1 disagreement rate is NOT
deliverable as registered: B names no region at decision 1 in 209 of 300 tasks, so the rate rests on 89 cells
(37 differ, 41.6%) and is recorded as CONDITIONAL, not as the registered reading.
(vii) Registered: the steps log gains one column, the parsed `action` beside `region` (nothing served or done
changes); agent B is re-collected so the 209 no-region cells split into "chose to inspect" and "unparseable".
Decision 1 is greedy and reproduces exactly; decisions 2–3 are sampled and will differ.
(viii) Registered: a SAME-AGENT CHURN control — a second run of the 7B on the corrected prompt, identical settings,
different sampling seed at decisions 2–3. Its "solved by run 2 and not run 1" count is the baseline that B-only must
clear: the quantity that supports (i) is B-only minus same-agent-churn-only, with both counts printed. Until (viii)
has run, the 23 is "different agent plus run-to-run churn" and is quoted only in that form.
(ix) The stale-prompt control of (iii) is unchanged and serves its own purpose only; it is not the churn control.
Order: (iii) running → (viii) → (vii) → readings recomputed → only then the §16 head arms.

**16a (x), 2026-09-24 18:1xZ — the stale-prompt control (iii) landed; reading (i)'s bar is tightened, not relaxed.**
Control (iii): 7B on v3c with the stale prompt — 80/300 green (26.7%), on-menu 89.0%, no-region 11.0%, stale names
used 0 of 812 steps. Against the corrected-prompt 7B (68/300): both 55, corrected-only 13, stale-only 25 = 31.3% of
the stale run's solved set. Decision-1 picks differ 0 of 298 (the divergence measure reads zero when the policy is
the same and 41.6% when the agent changes — a positive control the programme did not have). Consequence: two runs of
the SAME agent, differing only in a line the agent ignores, clear the registered 25% bar. The bar therefore does not
separate "a different agent reaches different problems" from "this task set is noisy at n=300". Registered now, before
(viii) reports: reading (i) is RE-STATED as a difference — B-only rate (B-only / B's solved set) MINUS the same-agent
churn rate from (viii) (run-2-only / run 2's solved set), with a paired bootstrap interval over the 300 tasks; it
fires only if the interval excludes zero. The original 25% criterion is kept in the record as fired-but-uninformative.
This is a tightening of a registered gate after it passed, never a relaxation after a failure; the 23 and the 67.6%
travel only beside the churn figure. (iii)'s own result stands: the correction moved the vocabulary, not the difficulty
(26.7% vs 22.7%, inside two standard errors at n=300), and the 0→12% no-region shift was the task set, not the line.

**16a (xi), 2026-09-24 18:3xZ — (viii) landed; (x) computed; the normalisation is fixed.**
(viii) same-agent churn: 7B corrected prompt, run 2 (different process, mlx RNG unseeded): 60/300; against run 1
(68/300) both 56, run-1-only 12, run-2-only 4 = 6.7% of run 2's solved set; decision-1 picks identical 298 of 298.
So pure sampling churn is 4 tasks; the 25 of (x)'s stale/corrected pair was the prompt line, not the sampler — one
added line in every state text moved the outcome six times more than re-sampling did, while leaving the greedy first
decision untouched. Recorded as a property of the instrument.
Reading (x), paired bootstrap over the 300 tasks, 20 000 resamples: over all tasks B-only 7.67% − churn-only 1.33% =
+6.33 pp, 95% CI [+3.00, +9.67]; over each arm's own solved set 67.6% − 6.7% = +61.0 pp, [+43.9, +77.3]. Both exclude
zero. THE reading is the all-tasks normalisation: the denominator is the same object for both arms and does not move
with how much each agent happened to solve; the own-set form is reported beside it. The original (i) criterion also
survives (churn 6.7% < 25%); it stays in the record as fired, no longer marked uninformative.
(vii) first 102 no-region cells at decision 1: parsed_action = inspect in 102 of 102, zero parse failures. Agent B's
69% no-region at decision 1 is a policy (look before hypothesising), not junk. The 300-cell decision-1 measure §16
registered is therefore recoverable on the ACTION (hypothesise-vs-inspect, and region where both hypothesise) once
(vii) completes; it is computed then, not on the partial run.
Conclusion as of this entry: on the registered readings and on the tightened (x), the plateau of §13 is the policy's,
not the task's — a different agent of the same family reaches problems the first agent does not, beyond churn, with
an interval that excludes zero. The head arms of §16 may now be fitted, after (vii) completes.

**16a (xii), 2026-09-24 19:2xZ — (vii) complete; every reading recomputed on the complete collection.**
No-region cells across all decisions: 224, of which inspect 223 (99.6%) and invalid 1; `parse_action` returns
"invalid" on unparseable text and never defaults to inspect. Unconditional decision-1 measure on all 300 tasks:
7B hypothesises first in 298/300 (99.3%); B inspects first in 209 (69.7%), hypothesises in 90, invalid 1. B's first
decision is greedy and reproduced exactly across the two independent collections (divergence 37 of 89 both times),
so the conditional region rate is a property of B's policy, not of a draw. Reading (x) on (vii)'s collection: over
all tasks +7.33 pp, 95% CI [+4.00, +11.00] (B-only 8.67% − churn 1.33%); over own solved set +65.6 pp [+49.3, +80.5];
solved 7B 68, B 36, both 10, B-only 26. Head of record: on-menu 7B 87.9%, B 74.3%, B off-menu 0.0%; same-policy
decision-1 divergence 0/298; same-agent churn 4 tasks (6.7%). The states pass and the §16 head arms proceed in the
registered order against the §15 bar; (vii)'s collection is the B collection of record.

**16a (xiii), 2026-09-24 22:3xZ — game 1's B arm never ran; caught by the clock, fixed at the class.** Arm names `v_b`
(base) and `v_B` (agent B's head) collide on a case-insensitive filesystem; the loop skips a slice whose episode
count is already 75, so `v_B1` resolved to `v_b1`'s files and reported "COMPLETE 300/300 green=69" — the base arm's
own number, relabelled, in the same second as the previous arm finished. The B arm is renamed `v_vb`; the runner
refuses to start if any two arm names are equal under case folding; the scoring plan is renamed in the same
commit. Genuine so far, game 1: base 69, gen0 48, g1-matched 71; `v_vb` collecting. Nothing is read until three
games are on disk and the §15 bar is computed. Same failure class as the `min(200, n_train)` control: an artefact
that looks like a result, given away by arithmetic, not by its value.

**16a (xiv), 2026-09-25 00:5xZ — game 1's B arm was contaminated; withdrawn before any reading; the games move to
a set no head has seen.** The B-experience head was fitted on the 7B's states at B's visited places on seed 73's
300 (`tasks/v3c`), labelled by the verifier's `bug_region`, with every task in the fit; the games were scored on
the same 300. The arm returned 134/300 green against a base of 69 and a g1-matched head of 71 — a head recalling
the answer key for the programs it was asked about, not transfer. The comparison head (g1-matched, fitted on seed
21) was disjoint from the games, so the two arms were not even contaminated equally. The number is withdrawn, not
adjusted, and nothing computed against it exists. The tell was proportion, not sign: an effect an order of
magnitude larger than anything this instrument has produced is a reason to audit the instrument.
Registered now: (1) the games for every arm — base, gen0, g1-matched, B-experience — run on `tasks/v3ind89`
(seed 89, n = 300, sha 6d182631), a set no head in the comparison was fitted on; every existing head is untouched;
the v3c bracket already collected (base 69, gen0 48, g1-matched 71, base run 2 60) stands as a v3c record and is
not used in the §16 readings. (2) The COLLECTION set of an experience arm and the SCORING set of its games are two
separate registrations; §16 fixed only the first. From here every head arm states both, and a head is never scored
on a set it was fitted on, in whole or in part. (3) The three games run interleaved as before, §15 bar, ~11 h.

**16a (xv), 2026-09-25 12:0xZ — mechanism hypothesis and test, registered BEFORE game 3 of the seed-89 games lands and
before `g13b_score.py --plan viiic` is run.** Games 1–2 on v3ind89 (unread against the bar): base 66/67, gen0 58/60,
g1-matched 77/83, B-experience head 102/104. Contamination checks run before reading: shared programs between v3c
(B's fitting set) and v3ind89 = 0 of 300; shared symptom strings 70, of which same `bug_region` 5 (ceiling of any
symptom→region carry-over, against a gap of ~36); heads distinct (cosine 0.084, distinct md5). HYPOTHESIS: all heads
carry ground-truth labels; they differ only in the STATES they were fitted on. The 7B hypothesises at decision 1 on
298/300 tasks with the code unread, so its own trajectories supply states in which the evidence is not yet in the
prompt (a head fitted there learns weakly or via symptom shortcuts; consistent with gen0 < base and with §13's
saturation). B inspects first on 209/300, so its visited places are states in which the program has been read; the
7B's hidden state at those places carries the evidence, and a head fitted there learns evidence→region. At game time
the head can only help by changing what the 7B does; the candidate mechanism is that the head prefers `inspect` at
decision 1 (inspecting preceded success in its fitting states), i.e. what transfers is a HABIT (look before
committing), not task facts. PREDICTIONS, fixed now: (P1) the 7B under the B-experience head inspects at decision 1
on far more than 1% of tasks, toward B's 70%; under g1-matched its decision-1 action distribution is unchanged. (P2)
the B-head's extra solves concentrate on tasks where it inspected first (paired, per task, over the three games).
(P3) mean actions per episode under the B head rise (an inspect precedes the hypothesis). If P1 fails and the gain
holds, the mechanism is not the habit and the states-are-more-informative reading stands alone; both are reported.
Computed from the games' own steps logs; nothing re-run. The union head (arm 2) follows the registered order.

**16a (xvi), 2026-09-25 12:1xZ — VIII-c arm 3 registered before game 3 lands: the forced-inspection control (the
owner's design, 12:0xZ).** If (xv)'s P1–P3 hold, the mechanism can be tested without agent B at all. COLLECTION: the
7B on seed 73's 300 (`tasks/v3c`, the same set B collected on), with ONE change during collection only: at decision 1
the action is forced to `inspect` (target chosen by the loop's existing rule for inspect targets; nothing else in the
prompt, temperature, fallback or verifier changes), decisions 2–3 under the 7B's native policy. HEAD: the 7B's states
at the places that policy visited, verifier labels, cut to 290 whole decision groups at the same seed, SHARED form —
identical in form and count to the B-experience head. GAMES: three, on `tasks/v3ind89`, interleaved with the existing
base bracket; the base, g1-matched and B-head games already collected on v3ind89 are the comparison and are not
re-run. READINGS, fixed now: (i) forced-inspect head within the §15 bar of the B-experience head → the mechanism is
isolated: B supplied an exploration policy that exposed the 7B to informative states its native policy rarely
visits, and nothing else about B is needed; (ii) forced-inspect head below the B head by more than the bar → B does
something richer than "look first" (different regions reached, trajectory lengths, or which evidence becomes
available), and the trajectory differences are the next object; (iii) forced-inspect head at or below g1-matched →
visiting informative states is not sufficient and the states B visits are special beyond inspection. Also recorded
per task: decision-1 action under each head at game time, so (xv)'s P1 is read for this head too. FRAME under test,
stated before the number (owner, 12:0xZ): three things usually bundled as "learning" are being separated — the
frozen model's intelligence (unchanged throughout), experience UTILISATION (the head; shown in VIII), and experience
ACQUISITION (which states the acting policy causes the model to encounter). §13 read as: utilisation stays available
while the model's own acquisition saturates ("a self-experience blind spot": it could not acquire experience from
states it did not visit). VIII-c tests whether changing the acquisition policy restores useful learning with the
intelligence held fixed. Order: after `g13b_score.py --plan viiic` and (xv) are posted; box B is not shared.
