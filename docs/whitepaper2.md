# Generalizing Efficient Thinking: The Evaluator × Search Law Across Games and Reasoning
## Paper 2 of the Efficient-Thinking series — a living draft (findings accumulate as they land)

> Status legend: **[SOLID]** measured & checked · **[PRELIMINARY]** first run, under-powered · **[PENDING]** running / planned.

## Abstract (stub)
Paper 1 established, in chess, that capability decomposes as **strength = evaluator × search**, that search
scales log-linearly with inference compute, and that the evaluator's ceiling is set by whether an external
oracle supplies information the model can't generate itself. This paper asks whether that law is *specific
to chess* or *general*. We test it in two directions — a simpler game (Connect-4) and a non-game domain
(LLM reasoning) — using a single calibrated capability scale (**GELO**) so the numbers are comparable by
construction. [thesis sentence to finalize once Arm B lands]

## 1. The measurement: GELO (Generalized Elo)
Full spec: `docs/gelo.md`. One logistic latent-ability model (Elo = Bradley-Terry = IRT/Rasch), chess's
constants (400 = 10×, 120 = one doubling), a *calibrate-first* protocol (reference ladder → cross-table fit
→ logistic goodness-of-fit gate → interpretable anchors → then rate agents), and a readout in "× better per
question." GELO is what makes a chess result and a reasoning result commensurable.
- **[SOLID]** First calibration (Connect-4): logistic goodness-of-fit **0.058** → the Elo model *fits* the
  game; ratings are valid, not assumed.

## 2. Arm A — games beyond chess (Connect-4)
The *simpler* end of the complexity spectrum. Perfect solver = ground-truth oracle; `ab_best` depth ladder =
calibrated opponents. Small conv net (policy + value), PUCT MCTS = closed loop.

- **[PRELIMINARY]** Evaluator × search decomposition (12k depth-8 labels): open-loop (raw net) **+644 GELO**,
  closed-loop (MCTS 200) **+880 GELO** → **search lift ≈ +236 GELO (~4× per game)**. Strikingly, the search
  lift is close to chess's, but the **lever balance shifts**: the raw net loses even to 1-ply search, so
  search carries proportionally *more* of the load here, and the lift is almost entirely *tactical* (wins/
  blocks the net misses) — the cleanest possible demonstration of "search extracts what the evaluator failed
  to represent."
- **[SOLID]** Calibrated ladder: random 0 → **depth-1 +804** → depth-2..6 +954..+1070. **The first ply of
  tactics is the single biggest step** (+804), bigger than all five deeper steps combined — diminishing
  returns on search, quantified. The heuristic ladder *saturates*.
- **[SOLID]** Open-loop ceiling vs data: GELO **+642 (12k) → +747 (24k) → +798 (50k)** — the raw net
  reaches **depth-1 parity (+804) at 50k**. So the "net loses to 1-ply search" gap was mostly
  *under-training*, not the game being inherently search-dominated: the evaluator catches up to
  low-depth search once given data. Honest correction to §2's first read — the lever balance was partly
  a starved-evaluator artifact. (Whether open-loop climbs *past* depth-1 with more data = the extension.)

## 3. Arm B — reasoning (LLM)
The mapping (the conceptual spine):
- **state** = partial reasoning trace · **policy** = next-step distribution · **evaluation** = P(reasoning is
  correct) · **terminal signal** = verifier on the final answer · **search** = inference-time compute over
  reasoning paths · **training** = RL/fine-tune (RLVR = AlphaZero's loop in language).
- **Evaluator output** = the [0,1] "P(correct)" analog of chess's win-prob: binary from a verifier,
  scalar from a reward/PRM, vote-fraction from self-consistency. The open problem is a *calibrated* one.
- **Search has two axes**: parallel/width (best-of-N, self-consistency) and serial/depth (long CoT). o1/R1
  scaled the serial axis; RLVR *internalizes* search into the policy (vs AlphaZero keeping it external) — a
  new degree of freedom this series should name.

- **[SOLID]** Clean accuracy-vs-N sweep (Qwen-4B, GSM8K, 120 problems, 1024-token budget): greedy
  pass@1 = **66.7%**; self-consistency@N = 69.2 (N=1) → 73.3 (4) → **77.5 (16) → 77.5 (32)**. **Search
  lifts accuracy +8–11 points over greedy, then saturates at ~77.5% by N=16.** The saturation is the
  key hook: majority vote (a verifier-*free* evaluator) stops helping — so past ~N=16 more search is
  wasted *on this evaluator*. That directly motivates the evaluator-quality ablation: a perfect verifier
  should break through the 77.5% consensus ceiling. (The earlier "26%→54%" was a 512-token truncation
  artifact — retired.)
- **[PENDING/running]** Evaluator-quality ablation (`reason_ablation.py`, llm2): self-consistency@N
  (verifier-free consensus) vs oracle-best-of-N@N (perfect verifier) from the same samples. The gap =
  accuracy a *perfect evaluator* unlocks over consensus = the direct "is the evaluator the bottleneck?"
  test in language.

## 4. Stage 3 — evaluator-first self-training (does the flywheel climb, and where does it plateau?)
The idea (from the series' self-improvement thread): *fix the evaluator first* — distill better-than-current
value targets (Monte-Carlo / MCTS-backed, anchored on real terminal outcomes) back into the net; strength
follows. Connect-4 is the ideal testbed: perfect ground truth to measure the evaluator, cheap iterations.

- **[PRELIMINARY]** From-scratch run (40 iters, 24 games/iter, 64 sims): the evaluator improved
  (oracle MAE 0.48→0.40, decisive sign-acc 51%→60%) and **strength tracked it** (GELO +119→~+400) —
  confirming *fix-the-evaluator-and-strength-follows* in miniature. But it **plateaued at ~+400 GELO, below
  1-ply search (+804)**. Likely under-resourced (small sims/games/iters, single-rollout value targets) rather
  than a fundamental ceiling.
- **[PENDING]** Stronger run (more sims/games/iters, multi-rollout value targets, warm-start-from-supervised
  to test "can it beat the supervised ceiling?") to locate the *true* plateau.
- **[SOLID, negative]** **Chess from-scratch eval-first stalled at the random floor** (`chess_evalfirst.py`,
  llm1): open-loop Elo bounced 254–384 with *no climb* across 44 iters (~2h, ~1,400 self-play games).
  This is a compute/data-volume wall, not a method failure: from *random*, chess self-play needs orders
  of magnitude more games than two Macs produce overnight to discover basic tactics. A clean, honest
  negative that supports the compute-bound reading of the earlier ~2000 plateau.
- **[SOLID, negative]** **Chess warm-start eval-first ALSO stalled** (from `conv_value_full`, whose raw
  open-loop is only ~385; the ~2800 was always via MCTS). Open-loop stayed flat ~300–326 across both
  attempts (sims 80 *and* 160, lr 1e-3 *and* 3e-4). Combined with the from-scratch stall, this is a
  coherent boundary condition: **eval-first ("distill search into the policy") hits a bootstrap barrier
  when the seed policy is weak** — MCTS quality depends on the policy prior, so at feasible sim counts the
  search cannot overcome a near-random prior to produce targets strong enough to improve the policy. A
  chess plateau-break this way needs a *decent seed policy* (a strong-policy net **with** a value head —
  not available off-the-shelf here) or cascade-level search / far more compute. Notably the Connect-4
  analog *does* climb off the floor — the barrier is complexity/compute-dependent, consistent with the
  external-oracle thesis. (Chess thread rested; llm1 repurposed to the stronger Connect-4 run.)

## 5. Synthesis — what transfers, what shifts (to write as results land)
- The **evaluator × search decomposition transfers** (games + reasoning).
- Search is a **big, portable lever**; its scaling with inference compute recurs.
- The **lever balance shifts with domain**: weaker/cheaper evaluator or sharper tactics → search dominates.
- The **external-oracle law is the through-line**: search and self-training climb exactly as far as the
  evaluator's grounding allows (perfect verifier → climbs; learned proxy → plateaus/hacks).
