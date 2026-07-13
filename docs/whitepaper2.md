# Efficient Thinking II — An Evaluator × Search Pattern Across Games and Reasoning
## Spending capability across domains — from a solved game to language reasoning

> Status legend: **[SOLID]** measured & checked · **[PRELIMINARY]** first run, under-powered.

## Abstract
Efficient Thinking I observed, in chess, that capability decomposes as **strength = evaluator ×
search**: a fixed evaluator (one forward pass) sets a base level, inference-time search multiplies it,
and self-learning plateaus because the *learned evaluator* is the binding constraint. This paper asks
whether that pattern recurs beyond chess or is specific to it. We report recurring empirical
patterns observed under deliberately modest compute — not proven laws. We test it in two directions — a *simpler* solved game
(Connect-4) and a *non-game* domain (LLM mathematical reasoning) — and, to make the numbers comparable
across domains, we introduce **GELO**, a calibrated cross-domain capability scale. The decomposition
transfers cleanly: search is a large, portable lever that scales with inference compute and then
saturates against the evaluator's ceiling. Most sharply, in reasoning a perfect verifier breaks a
consensus ceiling that more search cannot (**+14.2 points**), and across four self-improvement
experiments the plateau never breaks — because **you cannot improve the evaluator for free; self-play
converges to its own level of play, and climbing past it requires importing information (a better
oracle).** The evaluator is the bottleneck, in every domain we tested.

## 1. The measurement: GELO (Generalized Elo)
Full spec: `docs/gelo.md`. Capability is a latent ability θ on one logistic model that unifies chess
Elo, Bradley–Terry, and item-response theory (Rasch): P(win/solve) = 1/(1+10^(−(θ−d)/400)). We keep
chess's constants (**400 GELO = 10× better; 120 GELO = one doubling**), so a chess GELO *is* a chess
Elo, and add a *calibrate-first* protocol: build a reference ladder → fit ratings from a cross-table →
**gate on logistic goodness-of-fit** → pin interpretable anchors → only then rate agents. GELO is what
lets a chess result and a reasoning result be stated on the same axis, read as "N× better per game/
question."
- **[SOLID]** First calibration (Connect-4): logistic goodness-of-fit **0.058** (mean |predicted −
  observed| pairwise win-rate) → the Elo model genuinely *fits* the game; ratings are earned, not
  assumed. The scale is anchored random := 0.

## 2. Arm A — a simpler game (Connect-4)
The *simple* end of the complexity spectrum. Connect-4 is solved, so the exact solver is a perfect
oracle and a depth-limited alpha-beta gives a calibrated opponent ladder. A small conv net (policy +
value) is the evaluator; PUCT MCTS is the closed loop.

- **[SOLID]** **Calibrated ladder**: random 0 → **depth-1 +804** → depth-2..6 +954..+1070. The **first
  ply of tactics is the single biggest step** (+804) — larger than all five deeper steps combined. The
  heuristic search ladder saturates fast: diminishing returns on search depth, quantified on a
  calibrated axis.
- **[SOLID]** **Evaluator × search decomposition** (12k oracle labels): open-loop (raw net) **+644
  GELO**, closed-loop (MCTS 200) **+880 GELO** → **search lift ≈ +236 GELO (~4× per game)** — the same
  order as chess's search lift. The decomposition transfers.
- **[SOLID]** **The lever balance is data-dependent, not intrinsic.** With only 12k labels the raw net
  loses even to 1-ply search, which reads as "search-dominated." But tracing the open-loop ceiling vs
  data — GELO **+642 (12k) → +747 (24k) → +798 (50k)** — the raw net **reaches depth-1 parity (+804) at
  50k**. So the apparent search-dominance was mostly a *starved evaluator*: given data, the evaluator
  catches up to low-depth search. The honest statement is that the evaluator/search balance shifts with
  how much you have spent on the evaluator, in either direction.

## 3. Arm B — reasoning (LLM)
The conceptual mapping (chess → language):

| chess | LLM reasoning |
|---|---|
| position | partial reasoning trace |
| policy (move priors) | next-step distribution |
| **evaluation: P(win)** | **P(this reasoning is correct)** — [0,1] |
| terminal result (win/loss) | **verifier on the final answer** (perfect in math/code) |
| MCTS search | inference-time compute over reasoning paths |
| training the net | RL / fine-tune — **RLVR = AlphaZero's loop in language** |

Two structural notes that this series should name: the evaluator's output is a **calibrated P(correct)**
— binary from a verifier, scalar from a reward/PRM, vote-fraction from self-consistency — and a
*well-calibrated* one is the open problem. And search has **two axes** — parallel/width (best-of-N,
self-consistency) and serial/depth (long chain-of-thought); o1/R1 scaled the serial axis, and RLVR
*internalizes* search into the policy rather than keeping it external as AlphaZero does.

- **[SOLID]** **Search scales reasoning accuracy, then saturates** (Qwen-4B, GSM8K, 120 problems,
  1024-token budget): greedy pass@1 = **66.7%**; self-consistency@N = 69.2 (N=1) → 73.3 (4) → **77.5
  (16) → 77.5 (32)**. Search buys **+8–11 points** over greedy — the reasoning analog of Elo-vs-sims —
  then **saturates at ~77.5% by N=16**. Majority vote is a *verifier-free* evaluator, and it stops
  helping.
- **[SOLID]** **The saturation is an *evaluator* ceiling, not a policy or search ceiling.** From the
  same samples, self-consistency (consensus) saturates at 77.5% while **oracle-best-of-N (a perfect
  verifier) climbs to 91.7% and is still rising at N=32** — an **evaluator-gap of +14.2 points**. The
  correct answer is *in the sample set* 91.7% of the time; consensus simply can't select it. So in
  language, exactly as in chess, **the verifier is the binding constraint** — and with a perfect
  evaluator, search keeps paying off well past where consensus stalls.

## 4. Self-improvement — can the flywheel raise the evaluator with no external teacher?
The self-improvement idea, stated value-first: *fix the evaluator first* — distill better-than-current
value/policy targets (Monte-Carlo rollouts / MCTS-backed, anchored on real terminal outcomes) back into
the net; strength follows. We tested this across a solved game and chess, five ways. **It never broke
the plateau — and *why* is the result.**

- **[SOLID]** **Connect-4, from scratch — the evaluator improves but strength plateaus.** Self-play
  expert iteration lifted the evaluator (oracle value-MAE 0.48 → 0.40 → **0.355** with more search) and
  strength tracked it early (GELO +119 → ~+400), confirming *fix-the-evaluator-and-strength-follows* in
  miniature. But strength **plateaus at ~+400 GELO** (below 1-ply search) and **4× more search per move
  did not raise it** — it only sharpened the evaluator's calibration. More search budget is not the
  missing ingredient.
- **[SOLID]** **The plateau is a (near) seed-independent attractor.** Warm-starting Connect-4 self-play
  from the *supervised* +644 net does not preserve it: the first self-play iteration erodes it to +189,
  then it recovers to a **~+480–500 plateau** — a little above the from-scratch +400 (the seed leaves a
  small lasting trace) but well below the +644 it started from and the +798 achievable with data.
  Self-play pulls a *better* evaluator down toward its own signal quality rather than building on it —
  the same erosion seen in the chess self-learned test (1214→833).
- **[SOLID, negative]** **Chess from scratch stalled at the random floor** — open-loop Elo bounced
  254–384 with no climb over 44 iters (~1,400 self-play games). Not a method failure but a data-volume
  wall: bootstrapping chess from *random* needs orders of magnitude more games than two Macs generate.
- **[SOLID, negative]** **Chess warm-start (weak-policy seed) stalled** at ~300–326 across sim/LR
  settings. Root cause: a **bootstrap barrier** — MCTS quality depends on the policy prior, so from a
  near-random prior the search cannot play well enough to generate improving targets.
- **[SOLID, negative]** **The proper test — chess eval-first from a *self-learned* plateau net**
  (`selfplay_warm`, open-loop 1214, off the floor, self-learned) — **did not climb; it degraded**
  (1214 → 833 over ~50 iters). MCTS on this net's value head doesn't play far enough above 1214 to
  produce improving targets, so distillation erodes rather than lifts.

**Why it can't work for free (the theory the experiments support).** A Monte-Carlo rollout *does* give
the true value of a position — but the true value *under the current level of play* (the model plays
both sides). Training on those labels converges the evaluator to a perfectly-calibrated evaluator *of
its own level* — accurate, and **capped there**: no move better than the current level ever appears in
the games, so none can be learned. The *only* way rollouts push past the current level is to play them
*above* it — with search — and then the per-iteration gain equals the **search-over-policy margin** (how
much MCTS beats the bare net). That margin is itself bounded by evaluator quality: a mediocre evaluator
yields weak search, hence a small margin, hence no climb. This is precisely AlphaZero's engine — and
precisely why it needs *deep* search × *millions* of games: a large margin sustained over many
iterations. At our scale the margin was too small to cross the plateau. **Self-play cannot add
information the evaluator doesn't already contain; raising the ceiling requires importing it (a better
oracle).**

## 5. Synthesis — what transfers, what shifts, what binds
We report these as recurring empirical patterns — consistent across the domains and scales we tested,
not claimed as laws:
1. **The decomposition transfers.** *Strength = evaluator × search* holds in a solved game and in
   language reasoning, on one calibrated scale (GELO). Search is a large, portable lever that scales
   with inference compute and then **saturates against the evaluator's ceiling** — Elo-vs-sims in chess,
   accuracy-vs-N in reasoning, GELO-vs-MCTS in Connect-4.
2. **The lever balance shifts with spend.** Whichever currency is *starved* dominates the returns: a
   thin evaluator (12k Connect-4 labels; a verifier-free consensus) makes search look all-important;
   feeding the evaluator (50k labels; a real verifier) rebalances it. There is no domain-intrinsic
   split — only how much you have spent on each side.
3. **The evaluator is consistently the binding constraint** across our experiments, shown three
   independent ways: reasoning
   consensus is broken only by a better *verifier* (+14.2); Connect-4 self-training is limited by
   value-head quality, not search budget; chess self-improvement stalls exactly when the evaluator is
   too weak for search to generate improving targets.
4. **Self-improvement has a rate, and it is the search-over-policy margin.** You cannot fix the
   evaluator for free: self-generated signal converges to the system's own level. Climbing requires
   either search that plays above the current policy (bounded by the evaluator) or an external oracle
   that injects new information. The positive proof is the flip side of every negative here — a perfect
   verifier unlocks +14.2 in reasoning, and Stockfish labels carried chess to ~2800. *Training creates
   information; search extracts it; neither creates what an external oracle must supply.*

## Reproducibility
Code and data generators for both arms are in the repo: `games/` (Connect-4 engine, oracle,
net/MCTS/eval/calibrate/self-play) and `reasoning/` (GSM8K accuracy-vs-N sweep and the evaluator-quality
ablation, via `mlx_lm`). GELO's calibrator (`games/c4_calibrate.py`) prints the goodness-of-fit gate.
Large datasets and model weights are regenerable and not committed.
