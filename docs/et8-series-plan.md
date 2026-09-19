# Efficient Thinking 8 — a series, written backward

Ruling (Louay, 2026-09-17): ET-8 is a series, not one paper. A long plan is broken into papers; each paper closes
on what is established and hands the remainder forward. Papers are written **backward** — the way mathematics is
published: the work is done a, b, c, d and the paper is written from d, so that the definitions, instruments and
results read as one argument rather than as the order in which we stumbled on them. The sequence and the thought
process are kept, in an appendix, because the withdrawals are part of the evidence. The same lens is applied to
ET-1 through ET-7.

## The principle, stated once

- The main text argues from the final understanding. A reader meets the definition (§4.2a's constraint) before any
  mechanism, the instruments (the task gate, the prompt gate, the three probe controls) before any number, and the
  results as answers to questions the reader has already been given.
- Nothing measured is deleted. Every withdrawn magnitude, every refuted expectation, every instrument that lied
  moves to an appendix, "How this was found", in chronological order, with the rule each one earned. That appendix
  is the method contribution and it is written as such.
- A paper closes on what is established with a real interval, and names what it hands forward. No paper claims
  the thesis; each claims what it measured.

## The long plan (the arc)

| paper | claim it closes on | what it hands forward |
|---|---|---|
| **8a — Experience Priors: the constraint and the carrier** | An experience prior is defined by a constraint (cost down, capability held within a pre-registered ε, paired per slice). Six carriers in two fields fail it; a read-only controller head passes it, replicated, where localisation limits the agent; and the largest effect is a change to *when* the agent may act. Accumulation cannot be tested where the prior's inputs are fixed before it acts. | The re-hypothesis loop (accumulation); the repair floor; a third field. |
| **8b — Accumulation** | Whether a second generation, trained on states its parent's choices produced, does better than its parent at held capability (P9, restated for a loop with more than one head decision). | Consolidation timescales; forgetting. |
| **8c — Where the search is** | The chess anchor and the poet study as carrier studies: what happens to a capability when experience is pushed into it by a weight update; search worth nine times the prior. | A field where the prior's headroom is large by construction. |
| **8d — Instruments** | The gates as a method: count the problems, count the prompts, permute the labels, remove dimensions and examples; the environment degeneracy that survived two task sets; the persistence rule. | A checklist for any experience-learning claim. |

## Paper 8a — the cut

**Keep, in this order:** abstract (rewritten from the results) · motivation (short) · intelligence vs experience ·
the constraint (§4.2a, promoted to the definition) · where a prior can act (the observation-free decision point is a
lookup table; the prior acts only where the search has observed something) · the instruments (task gate → prompt
gate → permutation, dimensions, examples; "verify the verifier") · the harness (the inspect-first loop, and why the
blind loop's numbers are not carried) · results: the six carriers as a table with direction and n; G on the third
task set with the transfer table and the agreement split; the bound (localisation-limited vs repair-limited) ·
what a prior is not (fit quality, probe accuracy, localisation) · what accumulation needs · conclusion.

**Move to Appendix A, "How this was found":** §20b's chronology, compressed: v1's ten symptom strings; the 2,000
tasks that were twelve problems; the head at 100% that passed its permutation control; the fit's free parameter;
the canned control that was an oracle; the symptom rule refuted by a three-episode smoke test; the two losses to
working-tree moves; the ceiling that moved a target before the run. Each with the rule it earned.

**Move to 8c:** the poet study and the chess anchor in full (8a keeps one paragraph each as carrier evidence).

**Candidate for 8c/8d (from R2's second review, 2026-09-19): a single-pass head.** The +34% token premium at budget 12 is the
k prefix passes that enumerate candidates. A head that reads the post-inspect state ONCE and emits a distribution over
candidate families (no prefix passes) would remove the premium entirely; R6(a) (self-rerank by the model's own logprob) is
the nearest baseline. Not for 8a: R5-Q already shows compute neutrality by budget, and a new head is a new mechanism with
its own ladder.

**Drop from 8a (hand to 8b/8d):** §5 lifecycle detail, §9 individual vs shared, §15 sleep, §17–19 as written.

## ET-1 … ET-7 — the same lens

Each earlier paper is re-read in the light of what came after it and rewritten backward: definition first,
instruments second, results as answers, chronology as an appendix. The review order is 7 → 1, because each later
paper tells us what the earlier one was actually about. This is a separate work item, after 8a.

## Rules for the rewrite

1. No number without its n and its interval; no magnitude from the first two task sets.
2. No mechanism section before the constraint; no result before the gate that admitted it.
3. Withdrawals are kept and dated; the appendix says who withdrew what and why.
4. The paper carries no estate name.
5. The paper is written under Louay's identity. The chronology appendix credits an experimenter (E) and a reviewer (R), pseudonymous, in a publishing universe separate from any operating context; no operating name, seat, lane or company appears in any paper of the series.

## 8a — the closing package (ruled 2026-09-18, from the external review)

8a's contribution is narrow and stated once: *a frozen model can acquire a useful search prior without altering its
intelligence — a read-only controller learned from prior trajectories uses information already present in the model's
internal state to choose better search actions, reducing search cost while preserving, and here increasing, task
success.* Four things close it; then it stops. Nothing about accumulation, inheritance or consolidation timescales
enters 8a.

| item | what | cost | pre-registered reading |
|---|---|---:|---|
| R1 · intervals | paired per-problem bootstrap (95% CI) on success and on actions for every G-vs-base pair, both arms on the same problems; McNemar on success; reported in place of point deltas | a join, minutes | the four cross-run deltas' intervals exclude zero or the table says which do not |
| R2 · compute accounting | tokens in/out per episode, wall per episode, AND the head's own cost — the k teacher-forced candidate prefixes per decision counted as inference — so the cost limb is stated in total tokens including the controller's overhead | a join over logs + one count | if the candidate passes cost more than the actions they save, the cost limb fails on tokens and the paper says so |
| R3 · larger independent evaluation | G (first seed's head) on the second seed's full 300 (only 75 used so far), and a third seed of 300 generated, gated (P0–P4), held, and run — 600 independent problems, none seen by any head | ~3.5 h GPU per 300 | effect on 600: +11 to +15 with an interval that excludes zero; or the interval that says otherwise |
| R4 · a second frozen model | the same environment, harness, gates and head procedure on a second checkpoint (the 3B instruct model already on the box; a larger one if it fits): base on 300, probe with three controls, G, on the same task set | ~half a day GPU | the bound predicts: if the 3B's failures are repair-bound, G does nothing there and that is the bound holding; if localisation-bound, G helps. Either reading is written before the run; "peculiar to one checkpoint" is refuted only by the second reading |

| R2b · the controller's overhead engineered | score all k candidates in one batched pass over a shared cached prefix; measure the token overhead of the system that exists after the change | a build + one run of 300 | overhead ≤ 502 tokens/episode — the 0.71 actions saved × the MARGINAL cost of an action, 707 tokens by OLS slope of tokens on actions (intercept −660 = the fixed prompt); the two other bases, mean/mean 458 and budget-based 406, print beside it and are not chosen after the fact. ≤ 502 → the cost limb passes on tokens with the measured system; above → it fails and 8a says G buys success at a compute premium; between 406 and 502 → passes the marginal bar, fails the budget bar, and the paper says which was named first (this one) |

R1 and R2 landed 2026-09-18: intervals exclude zero on every cross-run pair; the cost limb passes in actions and fails in
tokens (+34% per episode with the controller's inference counted; −21% per problem solved, post hoc and labelled so).

Order on the study box: R1, R2 (done) → R3 → R4 → R2b → then the 8b loop → then S2, S1 (ET-9). 8a is closed on R1–R4
and R2b regardless of what 8b finds.

## 8a — the frontier control (ruled 2026-09-18, from the review and Louay: "improve efficiency or improve total outcome
are the same benefit — an expert gets better results in the same time, or good results faster")

§3 gains the general criterion: an experience prior improves the quality–cost frontier, \(Q_E(C) > Q_0(C)\); efficiency
(same quality, less cost) and effectiveness (same cost, more quality) are its two limbs. The pre-registered experiments
stay reported under the first limb, exactly as run. One addendum (§7.6, R5) decides the second:

| run | design | selection rule | reading |
|---|---|---:|---|
| R5-E · base at the head's compute | base, no head, budget ∈ {14, 16, 18}; pick the budget whose mean tokens/episode is nearest 9,170 (measured on slice 1 first); run on 300 | nearest-tokens, fixed before the 300 | Q_base(≈9,170) < Q_G by more than the paired CI → frontier moved; ≈ equal → success bought with compute |
| R5-Q · head at the base's quality | G at budgets {10, 8, 6}; smallest budget with success ≥ 18.7%; its tokens vs 6,854 | smallest passing budget | fewer tokens → efficiency limb met; not fewer → the other half of the same answer |

Rule kept: the budget is in the prompt, so each budget is a different agent and is labelled as one. Existing data point,
stated as expectation not result: base 12 → 24 budget on 80 problems bought +3.7 points for ~2× compute (17ff143-era
probe); the head bought +11–14 for +34%. Cost: ~4 runs of 300 at ~3.5 h each. 8a re-stops after R5.

R5-E landed 2026-09-18 23:29Z: compute-matched to 0.44%, base 22.3% vs G 31.7% (+9.3 [3.7, 15.0], p 0.002) — the frontier
moved; the head is an experience prior on the effectiveness limb. **R5-C (added 2026-09-19, the reviewer's "a curve, not a
point"):** base at budgets 12/14/16/18 and the head at 6/8/10/12, all on the 300, one table of tokens-all-in vs success
with paired CIs, plotted as two curves. Readings pre-written: head curve above base across the range → the frontier is
moved over a range; curves cross at low budget → the head helps only where it has room to act; curves merge at high budget
→ compute substitutes for experience there. Cost ~7 h (two base budgets; R5-Q supplies the head points). 8a stops after R5-C.

## The series, as one argument (from the external review, 2026-09-19; adopted)

ET-I, search: thinking substitutes for size. ET-II to VII, limits: search is bounded by evaluators, verification,
supervision and the information already in the system. ET-8a, experience: historical computation teaches a frozen
intelligence where to direct future computation — the frontier moves once. ET-8b, accumulation: does it move again,
\(Q_2(C) > Q_1(C) > Q_0(C)\), intelligence frozen? The loop the series opened: intelligence → search → outcome →
verification → experience → better search. If 8b demonstrates repeated shifts, ET-8 rather than ET-1 is the result the
series is organised around; the backward rewrite of ET-1..7 is done in that light.

**R5-C extended (2026-09-19):** the reviewer's range is 5k–15k tokens; the base curve gains budgets 8 and 24 (≈5k and ≈15k)
beside 12/14/16/18, the head keeps 6/8/10/12. Two more base runs of 300 (~7 h). The "save 26 controller tokens" line of
work (R2b–R2d) is closed and not resumed; the frontier is the experiment.

## 8a — the last three gaps (from the external review, 2026-09-19; ruled)

| gap | experiment | status / reading |
|---|---|---|
| 1 · matched-compute frontier | R5-E (done: base at the head's compute 22.3% vs 31.7%) and R5-C (curve 5k–15k, queued) | the frontier is moved at one point; the curve says over what range |
| 3 · strongest simple baselines at the same compute | **R6, pre-registered:** at G's exact decision point, over the SAME enumerated candidates, (a) rerank by the frozen model's own log-probability of each candidate (the model's own preference, charged the same k passes); (b) majority vote over 5 sampled hypotheses (charged 5 short passes); plus base + 35% ordinary search = R5-E's budget-16 arm, already measured (22.3%). All on the 300, paired vs base and vs G | if (a) or (b) matches G, the learned readout adds nothing over the model's own preference and the claim narrows to "a controller at the decision point"; if both fall short of G by more than the CI, experience — the head trained on verified history — is responsible, not the extra inference |
| 2 · one genuinely different problem distribution | **R7, pre-registered:** a second environment with a different search structure under the same harness, verifier and gates — SQL query repair against a fixed schema (regions = clauses: select / where / join / group / order; verifier = expected result set; bugs admitted only if the verifier distinguishes them; task gate ≥ 0.9, prompt gate at the post-inspect decision, probe with the three controls). The head is RE-FITTED there from that environment's own base episodes: the claim under test is that the mechanism transfers, not the weights | base, probe, G on 300; same bar form (problems solved + cost; matched-compute point). If G clears it on a second search structure the claim is "experience-directed search", not "this grammar"; if it does not, 8a says the mechanism is grammar-bound and 8c takes the question |

Order after R5-Q: R5-C → R6 → R7 → 8a stops. Cost: R5-C ~7 h GPU; R6 ~7 h; R7 one day to build and gate + ~1 day of runs.
No further seeds, no third checkpoint (the reviewer: "that's enough seeds and models for me").
