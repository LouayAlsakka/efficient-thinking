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
5. **Anchor** to interpretable milestones (below), then **place agents** (next section).

## Placing an agent — two equivalent routes
Step 5 can be done two ways. Both are the *same* logistic model (400 = 10×); they differ only in **what
you anchor to**, and they agree by construction.

**(A) Against a reference ladder — opponent-/difficulty-anchored.** Play the agent against the calibrated
rungs (Stockfish rungs; `ab_best` depths; MATH difficulty tiers) and read its GELO off the ladder by
1-parameter MLE. The "opponent" is a rung or a problem's difficulty. This is how the chess, Connect-4,
and (via IRT over difficulty tiers) reasoning numbers here were produced. Convenient when you have a
graded ladder.

**(B) Pairwise, agent-vs-agent — reference-model-anchored.** Two agents answer the **same** items; a
judge decides each head-to-head; GELO comes from the win/loss/draw **cross-table** (Bradley–Terry), with
one agent **pinned as the reference** (e.g. a strong model := 2800, or a small one := 2000) and the rest
placed relative to it. Here the "players" are the *agents themselves* — the tightest analogy to chess and
identical to the LMSYS-Arena rating. Convenient when you have a set of comparable agents and a meaningful
reference. The **judge** is the crux:
- *Verifiable domain* (math, code): the **verifier** is the judge — right beats wrong, both-right /
  both-wrong = draw. Objective, no model bias; **preferred whenever a checkable answer exists** (don't
  add a judge model where a checker suffices).
- *Open-ended domain* (explanations, proofs-in-prose, "which answer is better"): a **master judge** — a
  model deliberately *stronger* than either contestant (e.g. a frontier model, or Kimi via an API) —
  decides each pair. Two caveats, both of which are just this series' thesis turned on the referee:
  (i) the rating is only as trustworthy as the judge; on items near or above the *judge's own* ceiling
  the **judge becomes the bottleneck** (evaluator-limited, exactly as elsewhere here); (ii) control
  judge self-preference/style bias — blind the source and randomize presentation order.

The choice of reference (2800, 2000, …) is free — it only fixes *where the origin sits*; the constants
(400 = 10×, 120 = one doubling) make every gap from it meaningful (a model 400 below the reference loses
~10:1 on the items that discriminate them). You can also **mix**: place a few agents on a graded ladder,
then chain the rest pairwise off those anchors.

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
