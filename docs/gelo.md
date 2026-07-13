# GELO — Generalized Elo

A universal capability scale for games, reasoning, and any domain where "A beats B" (or "A solves what
B can't") is measurable. GELO keeps chess Elo's constants — so a chess GELO *is* a chess Elo — and adds
a domain-independent calibration protocol plus an intuitive "how many times better" readout.

## The model
Every agent has a latent ability θ (in GELO points); every opponent/problem has a difficulty d. The
probability the agent wins / solves is the logistic

    P = 1 / (1 + 10^(-(θ - d)/400))

This is simultaneously chess Elo, Bradley-Terry, and item-response theory (Rasch) — one model. What
differs across domains is only *what we calibrate θ against*.

## The two constants (inherited from chess Elo)
- **400 GELO = 10x better** (a "decade" of odds).  P = 91%.
- **120 GELO = 2x better** (one doubling / one "bit").  Since 400·log10(2) = 120.4.

From these, the readout for any gap D:

| gap D | P(win) | odds ("x better") | out of N |
|------:|:------:|:-----------------:|:---------|
|  120  |  67%   |   2x   | wins 2 of 3 |
|  200  |  76%   |   3.2x | wins ~3 of 4 |
|  400  |  91%   |  10x   | wins ~10 of 11 |
|  600  |  97%   |  32x   | wins ~31 of 32 |
| 1700  | ~99.99%| ~16000x (2^14) | — |

odds(D) = 10^(D/400);  doublings(D) = D/120.4.

## Why the chess intuition is self-consistent (worked example)
- **"200 points = wins 3 of 4":** odds(200) = 10^0.5 = 3.2 : 1 → 76%. ✓
- **"top human ≈ 2^14 better than a beginner":** beginner ~600, master/GM-territory ~2300 → gap 1700 →
  doublings = 1700/120.4 = 14.1 → **2^14**. ✓
- **Reasoning example:** Engine A = 1100, B = 1300 (gap 200) → B is ~3.2x better per question: on the
  items that discriminate them, B is right ~3x as often as A (76% head-to-head).

## Calibration protocol ("calibrate first, then use")
1. **Reference ladder** — a monotonic set of reference opponents/difficulties we control: floor
   (random / novice) → … → ceiling (perfect play / human-elite).
2. **Cross-table**, not single win-rates — round-robin among rungs (agents that only ever beat one
   opponent give a fake rating).
3. **Fit** θ for every rung by MLE (Bradley-Terry / Zermelo-MM), convert to GELO (400 = 10x).
4. **Calibration gate** — check the logistic actually fits: mean |predicted − observed| pairwise
   win-rate. If a game's draw / first-move structure breaks the logistic, we learn that *before*
   quoting ratings. (Connect-4: 0.058 → fits well.)
5. **Anchor** to interpretable milestones (below), then **place agents** by 1-param MLE vs the ladder.

## Interpretable anchors (the "what does the number mean" layer)
- **Chess (human population, FIDE):** 600 novice · 1200 club · 2000 Expert · 2200 CM/NM · 2300 FM ·
  2400 IM · 2500 GM (~1,700 worldwide) · 2800+ world-elite.
- **Connect-4:** random = floor · `ab_best` depth-d = "d-ply tactical lookahead" · solver = perfect.
- **Reasoning:** difficulty tiers with known human solve-rates (GSM8K → MATH L1–5 → olympiad), fit by
  IRT; anchor θ to human-solver percentiles.

Cross-domain comparability comes from anchoring each domain to percentiles of its reference
distribution: "θ = 2000" then means the same *odds vs a calibrated ladder / percentile toward the
ceiling* in any domain — comparable by construction, not by analogy.

## Report GELO two ways
- **GELO points** (continuity with chess: 2800 still means 2800).
- **Doublings** = GELO/120 (the "2^N better" reading), when a bits-of-capability framing is clearer.
