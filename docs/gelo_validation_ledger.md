# GELO validation ledger

A running record of every check that the GELO scale is *earned*, not assumed — the evidence behind the
"calibrate-first" protocol. GELO is the shared logistic latent-ability model (Elo = Bradley–Terry =
1-parameter IRT), P(win/solve) = 1 / (1 + 10^(−(θ−d)/400)), with chess's constants kept (400 = 10× odds,
120 = one doubling). The only cross-domain claim it carries is that *search-lift magnitudes* are in the
same odds units; absolute levels across domains share a ruler by construction, not a difficulty.

| check | what it tests | result | status |
|---|---|---|---|
| **Goodness-of-fit gate** (Connect-4) | does the logistic model actually fit the cross-table? | mean \|predicted − observed\| pairwise win-rate = **0.058** | ✅ pass — ratings earned |
| **Difficulty-anchored ladder** (MATH IRT) | monotone in task difficulty? | L1 +1273 → L5 +1712, **~+110 GELO/level**, monotone | ✅ pass |
| **Pairwise arena, judge agreement** | does the master judge rank like the ground-truth verifier? | **84%** on decisive pairs (powered 150-problem rerun; up from 72% at 50) | ✅ adequate; judge itself evaluator-limited |
| **Arena ordering vs accuracy** | does GELO recover the size ordering? | 0.5B ≪ mid-ladder < 7B ≈ 14B recovered | ✅ coarse ordering holds |
| **Mid-ladder separation** | can the scale separate 1.5B vs 3B? | **tie on MATH** (+2419 vs +2392) though 3B ≫ 1.5B on GSM8K | ⚠️ benchmark-dependent — a scale reports only what its benchmark resolves (registered-prediction miss, §2) |
| **Cross-domain lift comparability** | are lifts in identical odds units? | chess +286 · Connect-4 +236 — same units, compared as *shapes/lifts*, never absolute levels | ✅ by construction (scoped claim) |
| **Anchors** | interpretable, free offsets | random := 0 (Connect-4); Kimi-K2.5 := 2800 (arena) | ✅ documented |

**Standing caveats (non-negotiable).** (1) A GELO number is a *within-domain* rating on a *shared ruler*;
"2500 reasoning" and "2500 chess" are not the same difficulty. (2) The LLM master judge is itself
evaluator-limited (84% agreement), so pairwise reasoning-GELO inherits a judge ceiling; on checkable tasks
the exact verifier is preferred. (3) Ratings are trusted only after the goodness-of-fit gate passes.

**Reproduce:** `games/c4_calibrate.py` (prints the 0.058 gate), `reasoning/reason_gelo_irt.py` (IRT ladder),
`reasoning/reason_arena.py` (pairwise arena + judge-agreement). Full spec: `docs/gelo.md`.
