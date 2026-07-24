# Studio Run Spec — Cross-Family Frontier Ladder (ET-II extraction robustness)

Registered before execution; commit timestamp is the registration. Purpose: the single experiment
the Paper II extraction ("locating the competence threshold") still wants — does the ~7B
size-vs-search crossover survive a change of model family, or is it a Qwen artifact? This is the
first question every ML-venue reviewer will ask, converted into a registered result before
submission. Priority: idle-time work; it yields to any ET-III/IV scoring event and to the
verified chess-trajectory run if that is approved and ready.

## Design

Replicate the ET-II frontier protocol exactly, on a second model family:

- **Models:** Llama-3.x-Instruct ladder at 4-bit, as many sizes as MLX conversions allow —
  target {1B, 3B, 8B, 70B}. 70B is the priority cell (it brackets the crossover from above);
  if 70B does not fit or is too slow, run {1B, 3B, 8B} and say so — a three-point ladder with
  8B bracketed still answers the location question coarsely. Record exact model IDs and
  quantization once, in the results.
- **Benchmarks:** GSM8K n = 500 AND MATH n = 500, same problem sets as the ET-II caches
  (identical problems is what makes cross-family comparison clean).
- **Protocol:** sample-once caches, 32 samples GSM8K / 16 samples MATH, temperature per ET-II
  convention; **true-greedy pass@1** decoded separately (the flip-decider rule — no
  first-sample-as-greedy); sc@{4,16,(32)} by majority; oracle@N (coverage) computed per cell;
  per-problem arrays committed from the first cell.
- **Analysis, frozen now:** the confirmatory question is the crossover's location and the lift
  collapse, not any single strict flip. Report per model: pass@1, sc@N, oracle@N, search lift.
  For adjacent-size flips (e.g., 8B+sc vs 70B greedy), exact paired McNemar with the same
  parity/strict framing as ET-II — parity language unless p < 0.05, no exceptions.

## Registered predictions (score all, either way)

- **X1 (threshold location).** The crossover region lands at the same order of magnitude:
  size dominates below ~3B (sub-3B Llama models cannot reach the next size class at any N),
  and search buys parity-or-better with the next size up starting in the ~8B range. A
  materially different location (e.g., crossover at 1B, or absent through 70B) is a miss and a
  finding about family-dependence.
- **X2 (lift collapse).** The search lift decreases monotonically with base competence on both
  benchmarks — the shape, not the exact values, is the claim.
- **X3 (benchmark saturation split).** Same pattern as Qwen: tighter/tie-level flips on
  saturated GSM8K, larger and more decisive margins on MATH.
- **X4 (coverage bound).** Oracle@N exceeds sc@N everywhere, with the gap widening as models
  weaken (Prop 2's structure, third family... second family, fourth dataset-context).
- **X5 (the extraction's headline, if it lands).** The Qwen and Llama frontiers overlay within
  noise when plotted as lift-vs-pass@1 (competence, not parameter count, as the x-axis): the
  crossover is a function of base competence, family-independent. This is the strongest
  available form of the claim and the extraction paper's Figure 1 if it holds; if the curves
  separate, the separation is the finding.

## Costing and order

MATH n = 500 first (it is where ET-II's significant result lives, so the replication matters
most there), GSM8K second. Within each: greedy pass, then the sample caches, small models first
so partial results are scoreable early. Expect the 70B cells to dominate wall-clock; run them
last and report the ladder without them if they stall, flagged.

## Paper integration (after results, never before)

Results feed the ET-II *extraction* draft (not the shipped compendium — v1.0.1 stands; a
one-line v2 note plus a §10 row is the compendium's only change, and only after scoring).
X1–X5 scored in the extraction's registered-predictions section; the lift-vs-competence overlay
becomes its lead figure if X5 holds.

## Out of scope
No third family (two families answer the reviewer; three is scope creep). No instruction-tuning
variants, no bf16 reruns, no prompt engineering — identical prompts to ET-II, priced at zero.
