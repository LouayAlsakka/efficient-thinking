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

| R2b · the controller's overhead engineered | score all k candidates in one batched pass over a shared cached prefix; measure the token overhead of the system that exists after the change | a build + one run of 300 | overhead ≤ the tokens of the 0.71 actions saved (≈ 400 tokens/episode) → the cost limb passes on tokens with the measured system; above → it fails, and 8a says G buys success at a compute premium |

R1 and R2 landed 2026-09-18: intervals exclude zero on every cross-run pair; the cost limb passes in actions and fails in
tokens (+34% per episode with the controller's inference counted; −21% per problem solved, post hoc and labelled so).

Order on the study box: R1, R2 (done) → R3 → R4 → R2b → then the 8b loop → then S2, S1 (ET-9). 8a is closed on R1–R4
and R2b regardless of what 8b finds.
