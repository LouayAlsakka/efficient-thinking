# Cross-Family Frontier Ladder — Results (ET-II extraction robustness)

**Registered:** `docs/et2-cross-family-frontier-spec.md` (commit `16f2ee3`). Scored here after generation.
**Question:** does the ET-II size-vs-search crossover survive a change of model family, or is it a Qwen
artifact? This is the first thing an ML-venue reviewer asks. Answered below as a registered result.

## What ran

- **Family / models (4-bit MLX, mlx-community):**
  `Llama-3.2-1B-Instruct-4bit`, `Llama-3.2-3B-Instruct-4bit`, `Meta-Llama-3.1-8B-Instruct-4bit`.
- **70B:** no Llama-70B 4-bit MLX conversion present locally → **three-point {1B, 3B, 8B} ladder**, the
  spec's sanctioned fallback. 8B is bracketed from below; the upper bracket is absent, so the crossover
  is located coarsely (which *step* it falls on), not pinned from above. Flagged, as the spec requires.
- **Benchmarks:** GSM8K n=500 and MATH n=500 — the *same* problem sets (seed-0 shuffle) as the ET-II
  Qwen caches, which is what makes the cross-family comparison clean.
- **Protocol (identical to ET-II):** sample-once caches, 32 samples GSM8K / 16 MATH, temp 0.8;
  **true-greedy pass@1 decoded separately** (the flip-decider rule — no first-sample-as-greedy);
  sc@{4,16,(32)} by majority; oracle@N per cell. Scored with the canonical `reason_cache` extractors so
  greedy and sc@N carry no normalizer drift.

Artifacts: `llama_frontier_analysis.py` (analysis), `llama_frontier_analysis.json` (all numbers),
`llama_math_frontier_scores.json` / `llama_frontier_powered_scores.json` (per-cache scores),
`llama_frontier_x5_overlay.svg` (Figure 1).

## Frontier table

**MATH (n=500, N=16)**

| model | true-greedy | pass@1 (sampled) | sc@4 | sc@16 | oracle@16 | lift (sc16−greedy) |
|------:|:-----------:|:----------------:|:----:|:-----:|:---------:|:------------------:|
| 1B | 15.6 | 12.2 | 16.8 | 22.6 | 45.2 | +7.0 |
| 3B | 39.6 | 31.8 | 37.0 | 46.8 | 68.4 | +7.2 |
| 8B | 41.8 | 37.1 | 42.6 | 55.4 | 73.6 | +13.6 |

**GSM8K (n=500, N=32)**

| model | true-greedy | pass@1 (sampled) | sc@4 | sc@32 | oracle@32 | lift (sc32−greedy) |
|------:|:-----------:|:----------------:|:----:|:-----:|:---------:|:------------------:|
| 1B | 34.8 | 28.8 | 37.6 | 50.6 | 84.8 | +15.8 |
| 3B | 77.4 | 74.0 | 81.0 | 86.2 | 98.2 | +8.8 |
| 8B | 82.4 | 80.1 | 84.8 | 90.8 | 98.8 | +8.4 |

## Adjacent-size crossover — paired exact McNemar (search on smaller vs **true-greedy** of next size up)

| flip | search | bigger-greedy | Δ | b / c | exact p | verdict |
|:-----|:------:|:-------------:|:--:|:-----:|:-------:|:--------|
| MATH: 1B+sc@16 vs 3B greedy | 22.6 | 39.6 | −17.0 | 21 / 106 | <1e-4 | **bigger wins** |
| MATH: 3B+sc@16 vs 8B greedy | 46.8 | 41.8 | +5.0 | 63 / 38 | 0.0165 | **search wins** |
| GSM8K: 1B+sc@32 vs 3B greedy | 50.6 | 77.4 | −26.8 | 21 / 155 | <1e-4 | **bigger wins** |
| GSM8K: 3B+sc@32 vs 8B greedy | 86.2 | 82.4 | +3.8 | 44 / 25 | 0.0295 | **search wins** |

The crossover sits at the **3B→8B step on both benchmarks**: 1B+search cannot come near 3B greedy
(both p<1e-4), but 3B+search significantly beats 8B true-greedy (both p<0.05). Parity language is not
needed — both live flips clear p<0.05.

## Registered predictions — scored honestly

The headline (crossover survives the family change) is confirmed. But **three of the five registered
predictions are PARTIALS as literally worded**, and the deviations are the discoveries (R1, R2 below).
The earlier commit `434ace9` marked X1/X3 "hit" and X2 "hit on GSM8K" in its headline; that rounded up.
Corrected ledger:

- **X1 (threshold location at ~7B) — PARTIAL.** The *structure* travels: size dominates below the flip
  (1B+search cannot reach the next class at any N, p<1e-4 both benchmarks), search wins above it. But the
  flip sits at **3B-vs-8B in Llama, a lower parameter location than Qwen's 7B** — X1 predicted "~8B range /
  same order of magnitude," and the location genuinely moved. Superseded by **R1**, which is sharper and
  explains the move. Scored PARTIAL, not hit.
- **X2 (monotone lift collapse) — PARTIAL (miss as worded).** GSM8K lift falls monotonically (+15.8 →
  +8.8 → +8.4). MATH lift *rises* (+7.0 → +7.2 → +13.6), **violating the registered monotone-collapse**.
  The true shape is **R2**'s coverage-gated inverted-U, not a monotone decrease. Reporting it as "hit on
  GSM8K" understated a real directional violation on MATH.
- **X3 (benchmark saturation split, "same pattern as Qwen") — PARTIAL.** Direction is right (MATH margin
  Δ+5.0 larger than GSM8K Δ+3.8), but **Llama's GSM8K flip is decisive (p=0.0295), not the tie Qwen showed
  on saturated GSM8K** — because Llama-8B (greedy 82.4) does not saturate GSM8K the way Qwen-7B+ (89–94)
  did. Consistent with the *saturation logic*, but not with "same pattern" as X3 phrased it.
- **X4 (coverage bound) — HIT, clean.** oracle@N > sc@N in every cell, gap widening as models weaken.
  MATH gaps (oracle−sc): 1B 22.6, 3B 21.6, 8B 18.2. GSM8K gaps: 1B 34.2, 3B 12.0, 8B 8.0. Prop-2
  structure holds in the second family on both benchmarks. The only clean, unqualified hit.
- **X5 (headline overlay) — HOLDS (pending scrutiny).** Plotted as **lift vs base competence (pass@1)**,
  the Qwen and Llama frontiers interleave rather than separating by family (`llama_frontier_x5_overlay.svg`).
  Matched-competence pairs coincide — Qwen-MATH pass@1 10.9→lift +10.1 vs Llama-MATH 12.2→+10.4;
  Llama-MATH-8B 37.1→+18.3 vs Qwen-MATH-3B 32.9→+18.8. The crossover is a function of base competence,
  not parameter count. Extraction paper's Figure 1 — if it survives R1/R2 (it is consistent with both).

## Refinements — the two results better than the predictions (post-hoc, flagged)

Both are computed in `llama_frontier_refinements.py` / `.json`. Comparators: the flip *significance* uses
TRUE greedy (flip-decider rule); the *mechanism* below uses sampled pass@1, the only competence comparator
present in every committed cell of both families.

- **R1 — the lift-vs-gap flip law (sharper than X1).** Search flips the ordering against the next size
  class exactly when **its lift clears the adjacent-class greedy gap it must jump**:
  `flip(K) ⇔ lift(K) = sc@Nmax − pass1 ≥ gap(K) = pass1(K+1) − pass1(K)`. This predicts *both* families'
  flip rungs from their own ladders:
  - Llama flips at **3B** — GSM8K lift +12.2 ≥ gap +6.1; MATH lift +15.0 ≥ gap +5.3.
  - Qwen does **not** flip at 3B (GSM8K lift +17.7 < gap +19.5; MATH +18.8 < +32.6) and flips at **7B**
    (GSM8K +4.6 ≥ +3.7; MATH +8.5 ≥ +4.4).
  The location moved because **Llama's adjacent-class greedy gap is small** (3B→8B +6.1) where **Qwen's was
  large** (3B→7B +19.5) — the invariant is lift-vs-gap, not a parameter count or even raw competence.
  *Caveat:* the law registers a spurious "flip" wherever an adjacent gap is ~0 (Qwen-MATH 1.5B→3B, gap
  +2.3) — that is the benchmark-dependent mid-ladder tie the compendium already documents (§10), not a size
  crossover; read the law at the lowest rung clearing a *meaningful* gap.
- **R2 — the coverage-gated inverted-U (corrects X2).** Lift is not monotone in competence: it **ascends
  out of the low-competence floor, peaks, then collapses under saturation**. The left limb is gated by
  Prop-2 coverage — you cannot select an answer that is never sampled (Llama-1B MATH oracle@16 only 45.2).
  Pooled lift-vs-pass@1 rises from +10.1 (pass@1 10.9) to a peak ≈ +18–22 (pass@1 ~22–37) and falls to
  +0.9 (pass@1 94.1). **Qwen's ladder sampled the descending limb; Llama's samples the ascent** — which is
  precisely why X2's monotone-collapse held for Qwen and failed for Llama on MATH.

## Bottom line

The crossover is **not** a Qwen artifact: it reproduces in Llama-3.x on both benchmarks with paired McNemar
flips significant at p<0.05, and X4 holds cleanly. But the honest score is X4 hit, X5 holds, **X1/X2/X3
partial** — and the partials are the point. For the extraction paper this is *stronger* than a clean sweep:
R1 (lift-vs-gap flip condition) and R2 (coverage-gated inverted-U) turn "we located a threshold" into "we
characterized the mechanism that places it," with the X5 overlay as the lead exhibit. Upper bracket
(Llama-70B) remains open; add it if an MLX conversion lands.

*Paper integration:* feeds the ET-II **extraction** draft only. The shipped compendium (v1.0.1) stands —
a one-line v2 note plus a §10 row is its only change, and only per the registered plan.
