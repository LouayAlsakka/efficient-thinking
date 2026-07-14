# Efficient Thinking: An Empirical Framework for Resource Allocation in AI
## Spending Parameters, Data, Search, and Latency Efficiently to Maximize Capability — Demonstrated in Chess

**Louay Alsakka** · July 8, 2026

**A white paper on parameter-efficient game AI.**

**Code & trained model:** [github.com/louayalsakka/efficient-thinking](https://github.com/louayalsakka/efficient-thinking)

*The recurring theme is a set of **tradeoffs**: memory vs compute (Stage 1 vs 2), speed vs score
(the search cascade), and diversity vs correlation (the committee) — each one a different way of
spending a fixed budget to buy strength, and each capped by the same wall: the quality of the
learned evaluator.*

---

## Abstract

**This paper asks one engineering question: given finite resources — parameters (memory), training
data, inference-time search, and latency — how do you spend them *efficiently* to reach a target
playing strength?** Using chess as a clean, fully-measurable testbed, we map the tradeoff curves
resource by resource. A **14 MB** convolutional evaluator (3.45M params) plays at ~2150 Elo as a
single forward pass; adding **Monte-Carlo Tree Search (MCTS)** [Coulom 2006; Kocsis & Szepesvári 2006] lifts the *same weights* to **~2800** —
a same-ladder gain of **+286 Elo from inference compute alone**, with zero extra parameters. Since
~2800 is Super-GM / peak-human level, the study is really *how small a model can reach top-human
capability when augmented by search*: the same capability, bought with **compute, not size**.

**The primary contribution is an empirical *framework* for allocating finite AI resources —
parameters, data, search, latency — to a fixed capability goal, with measured tradeoff curves; the
staged MCTS cascade is its sharpest demonstration.** Three results support it: **(1)** a **wide→narrow
MCTS cascade** matching flat MCTS at up to **~4.8× less compute per move**; **(2)** a direct
**MCTS-vs-fixed-depth** comparison — adaptive search beats alpha-beta at equal compute and *keeps
scaling* while fixed depth plateaus; and **(3)** a **teacher-free self-learning** study with one
robust positive — **agreement predicts correctness** — and honest negatives — self-play, a
self-referential ladder, evolution, and plurality voting all fail to cross the plateau. A **capacity
sweep** agrees: 1.4× more parameters at fixed data move strength **~0**, while 8× more *data* moves it
**~+90** — capacity is the weakest lever in this regime.

The unifying principle — the **evaluator–search decomposition** — is **strength = evaluator ×
search**: search sets how *closely* you approach the evaluator's ceiling, the evaluator sets *where
that ceiling is*, and that ceiling is **the quality of the information the evaluator was trained on**.
Search *converts* the net's latent information into better decisions; it cannot create what the net
never learned. And the method transfers beyond the numbers: **at each stage one lever binds, and only
an experiment reveals which — so effort on any other returns almost nothing.**

**Headline:** *A 14 MB evaluator plus adaptive search reaches ~2800-class play, and a staged MCTS
**cascade** recovers up to **4.8× compute** at little Elo cost — thinking, not growing. Search
converts what the evaluator already knows into stronger play; it cannot lift the ceiling set by the
**quality of the information the evaluator was given.***

---

## Key Takeaways

*Five findings — enough to understand the paper on one page.*

1. **Search dominates parameters in this regime.** On a *fixed* evaluator, adding search buys **+286
   Elo** over the raw policy and keeps climbing **~+55 per doubling** of simulations; *doubling the
   parameters at fixed data added ~0*. Strength came from **thinking, not growing**.
2. **A wide→narrow MCTS cascade recovers up to 4.8× compute.** Funnelling the simulation budget
   through progressively narrower, deeper stages **matches flat-MCTS strength at a fraction of the
   per-move cost** — our most practical result.
3. **Adaptive MCTS out-scales fixed-depth search.** At equal compute MCTS **beats** alpha-beta and
   **keeps scaling**, where fixed-depth search plateaus.
4. **Model agreement predicts correctness.** Where independent models agree they are more often
   right — a **teacher-free confidence signal**, and a result independent of chess.
5. **Find the binding bottleneck experimentally.** At each stage exactly one lever binds — capacity,
   search, data, or the quality of self-generated signal — and effort on any *other* returns almost
   nothing. The transferable contribution is this **diagnostic**, not any single Elo number.

---

## 1. Introduction

**Every AI system faces an economic problem.** Capability is purchased with finite resources — model
capacity (memory), training data, inference-time computation, latency, and human supervision. The
engineering objective is not to maximize any one resource but to **maximize capability per unit
cost**: more capacity, data, search, or supervision each raise strength, but each carries a cost and
to a large degree they *substitute* for one another. This paper presents an **empirical framework for
measuring those tradeoffs**, using chess as a clean, fully-observable testbed — we fix the objective
(playing strength) and measure what each resource buys, and where spending on one is wasted because a
*different* resource is the binding constraint. We claim **no solved optimal-allocation rule**; the
contribution is the framework and a set of *measured tradeoff curves*.

The argument is a three-level hierarchy: **(1) AI resource allocation** — the main thesis; **(2) the
evaluator–search decomposition** — the analytical model of how capacity and search combine into
strength; and **(3) binding-bottleneck analysis** — the diagnostic that identifies, at any stage,
which single resource to spend on next.

The experiments are **three allocation questions** of increasing autonomy:
1. **Stage 1 (open loop):** given a fixed memory budget, **what architecture buys the most
   capability?**
2. **Stage 2 (closed loop):** given a fixed evaluator, **what is the cheapest way to buy additional
   strength with inference-time computation?**
3. **Stage 3 (self-learning):** without external labels, **what is the most efficient way to improve
   the evaluator?**

We measure each independently. The framing is also **control-theoretic** (Bertsekas 2022, *Lessons
from AlphaZero for Optimal, Model Predictive, and Adaptive Control*): open-loop policy = feedforward
controller, closed-loop search = model-predictive control, self-play = iterative/adaptive learning
control. Chess is only the testbed; the goal is a *generic* way to reason about resource tradeoffs in
sequential decision-making.

The central *tool* for efficient allocation is a simple diagnostic: **at any given stage, strength is
gated by a *single binding resource* — capacity, search, data, or the quality of self-generated
signal — and spending on any *non-binding* resource returns almost nothing.** So the efficient move
is always to **identify the binding resource experimentally, then spend there.** Which one binds is
not obvious a priori and shifts as you relieve each (open-loop is capacity-bound; add search and you
ride it up log-linearly until the evaluator's quality caps the return; then the evaluator — its
*data*, not its parameter count in our regime — binds). This bottleneck analysis is not the paper's
goal but its **method**: one instrument in the larger objective of spending a fixed budget
efficiently.

**Target calibration.** We aim at the **human-peak band — ~2800 Elo, Super-Grandmaster level — by
design.** For human-facing applications, matching top-human capability is the efficient saturation
point; pushing to 3500 Elo (machine-only territory) spends compute where it stops mattering. The
question is therefore *how small a model, plus how little search, reaches human-peak* — efficiency at
the human ceiling, not absolute strength.

**How to read the numbers.** Absolute Elo is measured against a Stockfish ladder and carries **±~100
systematic uncertainty** near the top rung; treat "~2800" as a headline calibrated to the human-peak
band. The paper's *claims* are the **relative, same-ladder** results, which do not depend on that
calibration: search adds **+286 Elo** over the raw policy, MCTS out-scales fixed-depth search, the
cascade holds Elo at up to 4.8× less compute, and every Stage-3 aggregation method fails to beat a
single model. We further separate **established** results (robust, relative — the cascade, adaptive
search scaling, agreement-predicts-correctness) from **regime-limited observations** true only at our
compute (the self-play plateau, flat parameter scaling, evolution's non-escape), which we expect to
change at AlphaZero/LLM scale.

**Contributions.**
- **Our headline method — a search-efficiency result:** a wide→narrow MCTS **cascade** that funnels
  the simulation budget through progressively narrower, deeper stages, matching flat MCTS at up to
  **4.8× less compute per move** with a clean score/speed trade-off curve.
- A parameter-efficiency result: **~2800 Elo from 3.45M params** via search, at constant memory.
- A direct **MCTS-vs-fixed-depth** result: adaptive search beats and out-scales fixed depth.
- An **architecture-beats-scale** finding (convolution ≫ MLP at equal data), with topology sweep.
- A reproducible **negative result** on small-scale self-play (plateaus below supervision).
- A **teacher-free self-learning** study: agreement is a validated **confidence meter**, but
  self-play, a self-referential ladder, **evolution**, and plurality-voting committees all fail to
  cross the plateau — a set of clean negative results with a methodological warning (noisy fitness
  manufactures phantom gains).

The paper's roadmap in one picture — the three sources of strength, all feeding a single learned
evaluator:

<svg viewBox="0 0 660 470" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <defs><marker id="im" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#555"/></marker></defs>
  <rect x="0" y="0" width="660" height="470" fill="#ffffff"/>
  <text x="330" y="20" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1a2a3a">The decomposition — three sources of playing strength</text>
  <rect x="255" y="34" width="150" height="46" rx="8" fill="#eef4fb" stroke="#2c7fb8" stroke-width="1.5"/>
  <text x="330" y="56" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a2a3a">MODEL CAPACITY</text>
  <text x="330" y="71" text-anchor="middle" font-size="9" fill="#666">parameters · memory</text>
  <line x1="330" y1="80" x2="330" y2="104" stroke="#555" stroke-width="1.6" marker-end="url(#im)"/>
  <rect x="233" y="106" width="194" height="56" rx="8" fill="#e3eefa" stroke="#2c7fb8" stroke-width="2.8"/>
  <text x="330" y="130" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1a2a3a">LEARNED EVALUATOR</text>
  <text x="330" y="147" text-anchor="middle" font-size="9" fill="#555">the value net — sets the ceiling</text>
  <path d="M 300 162 L 192 220" fill="none" stroke="#555" stroke-width="1.6" marker-end="url(#im)"/>
  <path d="M 360 162 L 478 220" fill="none" stroke="#555" stroke-width="1.6" marker-end="url(#im)"/>
  <rect x="95" y="222" width="176" height="54" rx="8" fill="#fdf0e6" stroke="#e6550d" stroke-width="1.5"/>
  <text x="183" y="245" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#7a2e06">SEARCH</text>
  <text x="183" y="262" text-anchor="middle" font-size="9" fill="#666">inference-time compute</text>
  <rect x="389" y="222" width="192" height="54" rx="8" fill="#f2eef8" stroke="#756bb1" stroke-width="1.5"/>
  <text x="485" y="245" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#463b7a">SELF-LEARNING</text>
  <text x="485" y="262" text-anchor="middle" font-size="9" fill="#666">self-generated information</text>
  <line x1="183" y1="276" x2="183" y2="306" stroke="#555" stroke-width="1.6" marker-end="url(#im)"/>
  <line x1="485" y1="276" x2="485" y2="306" stroke="#555" stroke-width="1.6" marker-end="url(#im)"/>
  <rect x="95" y="308" width="176" height="54" rx="8" fill="#fff7f2" stroke="#e6550d" stroke-width="1.2"/>
  <text x="183" y="331" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#7a2e06">Extract knowledge</text>
  <text x="183" y="348" text-anchor="middle" font-size="9" fill="#666">approach the ceiling</text>
  <rect x="389" y="308" width="192" height="54" rx="8" fill="#f8f5fc" stroke="#756bb1" stroke-width="1.2"/>
  <text x="485" y="328" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#463b7a">Improve the evaluator</text>
  <text x="485" y="345" text-anchor="middle" font-size="8.5" fill="#b5322e">only if signal quality rises</text>
  <path d="M 183 362 L 300 402" fill="none" stroke="#555" stroke-width="1.6" marker-end="url(#im)"/>
  <path d="M 485 362 L 360 402" fill="none" stroke="#555" stroke-width="1.6" marker-end="url(#im)"/>
  <rect x="255" y="404" width="150" height="46" rx="8" fill="#e9f7ee" stroke="#31a354" stroke-width="2"/>
  <text x="330" y="432" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1a5e2f">PERFORMANCE</text>
</svg>

---

## 2. Problem Setup and Methods

### 2.1 Representation
- **Position → move** as a 4096-way (64×64 from–to) classification (promotions default to queen).
- **Side-to-move normalization:** the board is re-oriented so the mover is always "White" (mirror +
  color-swap for Black), so one network serves both players and the turn needs no bit. (Hence
  `Eval(P) = Eval(mirror(P))` exactly, and a move's value reads off as `Eval(B) + Eval(null) − 1`;
  measured mean tempo ≈ +0.15, up to +0.77 in tactical shots, negative in zugzwang.)
- **Encoding:** *one-hot* = 64×12 piece planes + 5 meta = **773 floats** (reshaped to 8×8×12 + 5 = 17
  channels for convolution), used throughout. Against a compact *packed* encoding (one 4-bit code per
  square, 69 floats) at equal budget and data, one-hot is decisively stronger (**84.9 vs 142.5 mean
  CPL**): nets learn categorical piece-type features far more readily than an arbitrary ordinal code,
  and convolution *requires* the per-type planes a packed board cannot provide.

### 2.2 Labels and objective
- **Supervised data:** the full public Lichess cloud-eval database — **394,669,566 positions**
  (79 shards), each with Stockfish multi-PV win-probabilities.
- **Hard objective:** cross-entropy to the single best move (bounded by imitation).
- **Soft objective:** cross-entropy to an *advantage-weighted distribution* over legal moves
  (`softmax(vᵢ/τ)`). Soft is not bounded by copying one move and generalizes better.

### 2.3 Evaluation
- **Elo via a Stockfish ladder** (random baseline, then SF at set UCI_Elo rungs), 20–80
  games/rung; Elo fit by maximum-likelihood against the known rung ratings.
- **Methodological caveat (important):** if the player beats the top rung, the Elo estimate
  **compresses/extrapolates**, so we raised the ladder as the player improved (SF-1700 → 2100 →
  2700 → 3000). Absolute numbers carry systematic uncertainty; **relative gains measured on the
  same ladder are the robust results** (treat absolutes as ±100). Where two methods are compared,
  they are always run on the *same* ladder and seeds.

### 2.4 Hardware
- Two Apple-Silicon **Mac Studios (M3 Ultra, 256 GB each)**, MLX framework, 40 Gbps bridge.

### 2.5 Reproducibility (evaluation protocol)
A fixed protocol makes every Elo/CPL figure comparable across methods:
- **Engine:** Stockfish 18 as ladder opponent and CPL oracle; opponents at fixed `UCI_Elo` rungs,
  movetime **0.03–0.04 s**; CPL at **fixed depth 12**.
- **Ladder:** a random anchor plus `UCI_Elo` rungs (e.g. 1700/2000/2300, 2400/2700/3000), Elo by
  maximum-likelihood fit; **20–40 games/rung**; margin ≈ $\pm 400/\sqrt{n}\cdot 2$ (**±100** typical)
  — rely on *relative* same-ladder deltas.
- **Openings/seeds:** sampled from the Lichess `2013-01` PGN, replayed to plies 6–16, both colors;
  seed 0 by default, and any two compared methods use the **same** ladder, openings, and seeds.
- **Caveats:** the ladder **compresses** once the player beats the top rung (we raised it as strength
  grew), and at 0.03–0.04 s Stockfish plays **below** nominal `UCI_Elo` — internally consistent, not
  calibrated to over-the-board Elo. Exact scripts are in the repo.

---

## 3. Stage 1 — Open Loop (single-pass policy)

### 3.1 Topology sweep — architecture beats scale
At fixed moderate data (18M positions) we swept eight architectures:

| Topology | params | Elo | note |
|---|---:|---:|---|
| **conv (64×10 / 96×8)** | 2.8–3.4M | **1476** | best & most efficient |
| constant-width MLP (1024×6) | 10.2M | 1108–1233 | best plain MLP |
| dual-path (wide+deep, gated) | 2.8M | 1082 | ties MLP at 3.5× fewer params |
| factored head (from/to) | 6.2M | 782 | −450 |
| funnel (1024→64) | 1.8M | 582 | narrowing hurts |
| pyramid (64→1024) | 5.0M | 431 | bad |
| bottleneck (512→8) | 0.5M | 340 | broken |

**Topology conclusion.** Only **constant-width** and **convolution** are competitive; every taper,
factoring, or bottleneck loses badly, because it destroys positional information before the head uses
it, while convolution **reuses local patterns via weight sharing** (learns a motif once, not per
square). The rule: **match the prior to the domain's structure (spatial locality) rather than adding
width.** A 3.45M conv matched a 14.4M MLP using **22× less data** — architecture, not parameter count,
set the strength.

### 3.2 Scaling laws (width and data)
- **Width (MLP, hard):** W1024 (14.4M) → 1449; W2048 (47.7M) → 1501: **+52 Elo for 3.3× params**
  — sharply diminishing. Top-1 accuracy saturates ~0.41 regardless of width.
- **Data (W1024):** 1127 / 1207 / 1300 / 1382 / 1411 at 23 / 47 / 94 / 187 / **375M** positions —
  large early gains, then **+29 on the last doubling** → saturates near the DB's 394M limit.

<svg viewBox="0 0 620 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="300" fill="#ffffff"/>
  <text x="310" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#1a2a3a">Stage 1 — data scaling saturates per architecture; the right architecture lifts the ceiling</text>
  <line x1="60" y1="250" x2="590" y2="250" stroke="#333" stroke-width="1.5"/>
  <line x1="60" y1="250" x2="60" y2="40" stroke="#333" stroke-width="1.5"/>
  <text x="28" y="150" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 28 150)">Elo</text>
  <text x="325" y="285" text-anchor="middle" font-size="11" fill="#333">positions (millions, log-ish spacing)</text>
  <text x="55" y="197" text-anchor="end" font-size="8" fill="#999">1400</text>
  <text x="55" y="122" text-anchor="end" font-size="8" fill="#999">1800</text>
  <text x="55" y="57" text-anchor="end" font-size="8" fill="#999">2150</text>
  <line x1="60" y1="54" x2="590" y2="54" stroke="#e6550d" stroke-width="1.5" stroke-dasharray="5,4"/>
  <text x="586" y="49" text-anchor="end" font-size="9.5" font-weight="bold" fill="#e6550d">conv + soft + full 394M → ~2150 (best open-loop ceiling)</text>
  <circle cx="560" cy="54" r="4.5" fill="#e6550d"/>
  <polyline points="90,245 190,230 290,213 390,196 560,192" fill="none" stroke="#2c7fb8" stroke-width="2.5"/>
  <circle cx="90" cy="245" r="4" fill="#2c7fb8"/><circle cx="190" cy="230" r="4" fill="#2c7fb8"/><circle cx="290" cy="213" r="4" fill="#2c7fb8"/><circle cx="390" cy="196" r="4" fill="#2c7fb8"/><circle cx="560" cy="192" r="4" fill="#2c7fb8"/>
  <text x="90" y="239" text-anchor="middle" font-size="9" fill="#2c7fb8">1127</text>
  <text x="560" y="205" text-anchor="middle" font-size="9" fill="#2c7fb8">1411</text>
  <text x="470" y="182" font-size="9" fill="#2c7fb8">MLP W1024 (data alone → saturates ~1411)</text>
  <path d="M 575 188 L 575 60" fill="none" stroke="#999" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#s1a)"/>
  <text x="582" y="130" font-size="8.5" fill="#666" transform="rotate(-90 582 130)">switch architecture (+740 Elo)</text>
  <defs><marker id="s1a" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#999"/></marker></defs>
  <text x="90" y="265" text-anchor="middle" font-size="9" fill="#666">23</text>
  <text x="190" y="265" text-anchor="middle" font-size="9" fill="#666">47</text>
  <text x="290" y="265" text-anchor="middle" font-size="9" fill="#666">94</text>
  <text x="390" y="265" text-anchor="middle" font-size="9" fill="#666">187</text>
  <text x="560" y="265" text-anchor="middle" font-size="9" fill="#666">375</text>
</svg>

### 3.3 Objective and open-loop ceiling
Soft beats hard by **+94–113 Elo** at subset scale; top-1 saturates while Elo keeps improving via
blunder-rate reduction (**top-1 is a misleading metric**). Best open-loop recipe = **conv + soft +
full 394M data**, with a **ceiling ≈ 2150 Elo**. Cost: 3.45M params, **14 MB**, **~176 MFLOP /
~1.5 ms per move** — cheap, but capped: no amount of width or data pushed a single pass past ~2150.

---

## 4. Stage 2 — Closed Loop / Inference-Time Compute (search on a value function)

We add a scalar head **Eval(N) ∈ [0,1]** = *expected score for the side to move* (held-out **MAE
0.088, correlation 0.877** — an excellent value function). Search uses the zero-sum identity — our
value after a move is `1 − Eval(child)` — so one network plays both sides; terminals return exact
0 / 0.5 / 1.

### 4.1 MCTS vs fixed-depth — the core comparison
We compared two search families on the same value net and ladder:

- **Fixed-depth alpha-beta** (negamax, policy move-ordering, quiescence at leaves):

  | depth | raw | search | gain |
  |---|---:|---:|---:|
  | 1 | 1661 | 1627 | **−34** (1-ply hurts — can't see the reply) |
  | 2 | 1685 | 1788 | +103 |
  | 3 | 1895 | 2005 | +110 |
  | 4 | 1914 | 2152 | +238 |
  | 6 | 2128 | **2575** | +447 |
  | 7 | 2187 | 2513 | plateau ~2550 |

- **Adaptive MCTS/PUCT** (`Q + c·P·√N/(1+n)`, value-head leaves, negamax backup):

  | sims | search Elo |
  |---|---:|
  | 100 | 2411 |
  | 200 | 2530 |
  | 400 | 2610 (2661 @ 40 games/rung) |
  | **800** | **2749 ≈ top-human (~2800)** |

**Result.** (i) **1-ply search *hurts* a strong policy** — it commits to a capture without seeing the
recapture; depth is where lookahead pays. (ii) **MCTS beats alpha-beta at equal compute and keeps
scaling** where fixed depth plateaus (~2550): uniform depth spends equal effort on every branch and
*amplifies the value net's noise*, while MCTS **allocates search adaptively to sharp lines and
averages over it.** MCTS wins Stage 2, reaching ~2800 on the fixed 3.45M net.

<svg viewBox="0 0 620 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="320" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Stage 2 — MCTS out-scales fixed-depth search</text>
  <line x1="60" y1="270" x2="590" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="60" y1="270" x2="60" y2="45" stroke="#333" stroke-width="1.5"/>
  <text x="30" y="160" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 30 160)">search Elo</text>
  <text x="325" y="305" text-anchor="middle" font-size="11" fill="#333">compute per move  (→ more search)</text>
  <polyline points="100,265 160,236 220,197 280,171 400,95 470,106" fill="none" stroke="#d95f0e" stroke-width="2.5"/>
  <circle cx="100" cy="265" r="4" fill="#d95f0e"/><circle cx="160" cy="236" r="4" fill="#d95f0e"/><circle cx="220" cy="197" r="4" fill="#d95f0e"/><circle cx="280" cy="171" r="4" fill="#d95f0e"/><circle cx="400" cy="95" r="4" fill="#d95f0e"/><circle cx="470" cy="106" r="4" fill="#d95f0e"/>
  <text x="500" y="110" font-size="10" fill="#d95f0e">alpha-beta (plateau ~2550)</text>
  <polyline points="220,125 300,104 400,90 540,65" fill="none" stroke="#2c7fb8" stroke-width="2.5"/>
  <circle cx="220" cy="125" r="4" fill="#2c7fb8"/><circle cx="300" cy="104" r="4" fill="#2c7fb8"/><circle cx="400" cy="90" r="4" fill="#2c7fb8"/><circle cx="540" cy="65" r="4" fill="#2c7fb8"/>
  <text x="360" y="55" font-size="10" fill="#2c7fb8">MCTS (keeps rising → ~2800)</text>
  <line x1="60" y1="63" x2="590" y2="63" stroke="#999" stroke-dasharray="4 3" stroke-width="1"/>
  <text x="580" y="59" text-anchor="end" font-size="9" fill="#666">~2800 top-human</text>
</svg>

### 4.2 Search efficiency — the wide→narrow MCTS cascade (new)

**Where the cascade comes from — expert human analysis.** The cascade is not a standard MCTS variant;
it was inspired by the resource-allocation principle **visible in how expert human players analyze
under a finite clock and finite mental effort**. A grandmaster does not walk the game tree blindly:
they first use **intuition — a learned evaluation — to spot the few candidate moves worth
considering** (a *wide, shallow* scan), then progressively **prune to the strongest lines and
calculate those deeper**, trading breadth for depth *in stages*, and at the end carry **one or two
lines very deep** (17–20 plies in sharp positions). The cascade **operationalizes the same
resource-allocation principle observed in expert human analysis**: spend inexpensive computation
broadly, then reserve expensive computation for the few candidates that remain plausible — which is
exactly why it recovers so much otherwise-wasted search. (We claim the *principle*, not a precise
cognitive model.)

Given MCTS wins, does the **allocation** of a fixed sim budget matter? The **cascade** runs MCTS in
*stages*: a **wide** stage (all moves, high `c_puct`, few sims) ranks broadly and passes its top-k by
visit count to a **narrower, deeper** stage (fewer moves, more sims, lower `c_puct`), funnelling the
budget onto survivors; a shared eval cache carries value-net calls forward.

A controlled **N = 1→10 sweep** (one consistent rule generates each funnel; all 800 total sims;
same tall ladder, seeds, and openings) gives the full trade-off curve:

| # levels | Elo (±89) | ms/move | speedup |
|---:|---:|---:|---:|
| 1 (flat MCTS) | 2683 | 1330 | 1.0× |
| 3 | 2605 | 903 | 1.5× |
| 6 | 2506 | 541 | 2.5× |
| 9 | 2543 | 300 | 4.4× |
| 10 | 2570 | 275 | **4.8×** |
| beam-minimax cascade (fixed depth) | 2487 | — | inferior primitive |

**Result.** Across all ten funnels Elo stays within a **single ±89 band** (2506–2683, mean ~2580) —
**no significant variation within our ±89 uncertainty** — while speed rises **monotonically to 4.8×
at N = 10.** Funnelling wide→narrow is a **near-pure efficiency win**: it holds flat-MCTS strength
while cutting per-move compute ~5×, because the survivors reaching the deep stages are the same moves
flat MCTS would have searched anyway — the funnel just stops paying for moves it has already ranked
out. The practical dial: **more levels = same strength, cheaper.** (The fixed-depth *beam* cascade is
a genuinely weaker primitive at 2487 — the adaptive MCTS stages matter.)

<svg viewBox="0 0 620 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="320" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Stage 2 — cascade: Elo flat within noise, speed rises to 4.8×</text>
  <rect x="60" y="113" width="500" height="126" fill="#2c7fb8" opacity="0.08"/>
  <text x="70" y="127" font-size="9" fill="#2c7fb8">±89 noise band — Elo ~flat across all N</text>
  <line x1="60" y1="270" x2="560" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="60" y1="270" x2="60" y2="45" stroke="#2c7fb8" stroke-width="1.5"/>
  <line x1="560" y1="270" x2="560" y2="45" stroke="#e6550d" stroke-width="1.5"/>
  <text x="28" y="160" text-anchor="middle" font-size="11" fill="#2c7fb8" transform="rotate(-90 28 160)">Elo</text>
  <text x="595" y="160" text-anchor="middle" font-size="11" fill="#e6550d" transform="rotate(-90 595 160)">speed ×</text>
  <text x="310" y="305" text-anchor="middle" font-size="11" fill="#333"># cascade levels (1 = flat MCTS,  800 sims throughout)</text>
  <polyline points="90,96 140,151 190,151 240,96 290,195 340,221 390,151 440,188 490,195 540,176" fill="none" stroke="#2c7fb8" stroke-width="2"/>
  <circle cx="90" cy="96" r="3.5" fill="#2c7fb8"/><circle cx="140" cy="151" r="3.5" fill="#2c7fb8"/><circle cx="190" cy="151" r="3.5" fill="#2c7fb8"/><circle cx="240" cy="96" r="3.5" fill="#2c7fb8"/><circle cx="290" cy="195" r="3.5" fill="#2c7fb8"/><circle cx="340" cy="221" r="3.5" fill="#2c7fb8"/><circle cx="390" cy="151" r="3.5" fill="#2c7fb8"/><circle cx="440" cy="188" r="3.5" fill="#2c7fb8"/><circle cx="490" cy="195" r="3.5" fill="#2c7fb8"/><circle cx="540" cy="176" r="3.5" fill="#2c7fb8"/>
  <text x="90" y="88" text-anchor="middle" font-size="9" fill="#2c7fb8">2683</text>
  <text x="540" y="168" text-anchor="middle" font-size="9" fill="#2c7fb8">2570</text>
  <polyline points="90,260 140,253 190,238 240,220 290,206 340,191 390,174 440,160 490,97 540,78" fill="none" stroke="#e6550d" stroke-width="2" stroke-dasharray="5 3"/>
  <circle cx="90" cy="260" r="3.5" fill="#e6550d"/><circle cx="140" cy="253" r="3.5" fill="#e6550d"/><circle cx="190" cy="238" r="3.5" fill="#e6550d"/><circle cx="240" cy="220" r="3.5" fill="#e6550d"/><circle cx="290" cy="206" r="3.5" fill="#e6550d"/><circle cx="340" cy="191" r="3.5" fill="#e6550d"/><circle cx="390" cy="174" r="3.5" fill="#e6550d"/><circle cx="440" cy="160" r="3.5" fill="#e6550d"/><circle cx="490" cy="97" r="3.5" fill="#e6550d"/><circle cx="540" cy="78" r="3.5" fill="#e6550d"/>
  <text x="90" y="252" text-anchor="middle" font-size="9" fill="#e6550d">1.0×</text>
  <text x="540" y="72" text-anchor="middle" font-size="9" fill="#e6550d">4.8×</text>
  <text x="90" y="284" text-anchor="middle" font-size="8" fill="#666">1</text>
  <text x="140" y="284" text-anchor="middle" font-size="8" fill="#666">2</text>
  <text x="190" y="284" text-anchor="middle" font-size="8" fill="#666">3</text>
  <text x="240" y="284" text-anchor="middle" font-size="8" fill="#666">4</text>
  <text x="290" y="284" text-anchor="middle" font-size="8" fill="#666">5</text>
  <text x="340" y="284" text-anchor="middle" font-size="8" fill="#666">6</text>
  <text x="390" y="284" text-anchor="middle" font-size="8" fill="#666">7</text>
  <text x="440" y="284" text-anchor="middle" font-size="8" fill="#666">8</text>
  <text x="490" y="284" text-anchor="middle" font-size="8" fill="#666">9</text>
  <text x="540" y="284" text-anchor="middle" font-size="8" fill="#666">10</text>
</svg>

### 4.3 The two-dimensional cost (memory vs GPU-cycles)
| | Stage 1 (open) | Stage 2 (closed) |
|---|---|---|
| **Memory** | 14 MB | **14 MB — search adds ~0** |
| **GPU cycles/move** (batch-1) | 1 pass, ~1.5 ms | ~800 passes, ~1.3 s (or ~0.6 s cascaded) |
| **GPU cycles/move** (batched, §4.4) | — | **~6–12× less — MCTS-3200 below batch-1 MCTS-800** |
| **Elo** | ~2150 (capped) | **~2800 (scales with compute)** |

**Stage 1 buys Elo with *memory* and saturates; Stage 2 buys Elo with *GPU cycles* and keeps
climbing** — and the cascade shows most of those cycles were waste (up to 4.8× recoverable). Note
that the ~1.3 s is a **batch-1 implementation artefact**, not the method's cost: batched-leaf
evaluation (§4.4, measured 6–12×) makes even MCTS-3200 cheaper than batch-1 MCTS-800, so the true
latency axis sits far below what we plot (clean solo-GPU figure pending). The central practical
result stands regardless: **strength is compute, not parameters.**

### 4.4 Search latency — what a second of thinking costs, and buys

Strength from search is not free: every simulation is a neural-network forward pass, so Elo is
paid for in wall-clock. Measured on the Mac Studio (M3 Ultra, MLX) across the three operating
points:

| Mode | Elo (abs. ladder) | Latency / move |
|---|---:|---:|
| Open-loop (raw policy) | ~2448 | **~2 ms** |
| MCTS-800 | **2734 ±76** | **~1.3 s** |
| MCTS-1600 | **2780 ±82** | ~1.9 s |
| MCTS-3200 | **2839 ±76** | ~3.8 s |
| MCTS-6400 | **2903 ±82** | ~7 s |
| MCTS-12800 | **2967 ±115** | ~14 s |

*All on the same Stockfish high ladder (2500/2800/3050). The round ~2150 / ~2800 headline figures
used elsewhere sit within the stated **±100** ladder uncertainty of these precise values.*

Three observations:

**1. Search scales *log-linearly* — ~+55 Elo per doubling, no saturation through 12800.** The first
slice is huge and cheap (MCTS-800 alone adds **+286** over the raw policy, 2448→2734); past that, each
*doubling* adds a steady **~+55–64 Elo** (2734→2780→2839→2903→2967), unbroken to **16× the base
budget** — cumulative **+233**, far above the ±~100 noise. *(An earlier "saturation at 3200" claim came
from a noisy **head-to-head** sweep, +12/+260/+191 per rung; the clean absolute curve supersedes it,
and head-to-head deltas inflate via ceiling compression — 3200-vs-800 reads +260 h2h but +105
absolute.)* So with a fixed evaluator search keeps paying, and we **never reach its ceiling** (tested
to 16×): marginal Elo *per compute* falls, but the curve does not flatten.

**2. Latency scales *sub-linearly* with sims — a red flag, not a feature.** MCTS-3200 does 4× the
simulations of MCTS-800 yet is only ~2.8× slower. The cause: this search evaluates leaves **one
position at a time (batch = 1)**, so each forward pass is dominated by fixed GPU-launch overhead
rather than compute — the M3 Ultra is massively under-utilised. Effective throughput is only
**~600 leaf-evaluations per second**, and adding sims mostly amortises the fixed per-move cost.

**3. The latency is an implementation artefact, not a hardware or method limit.** Batched neural-MCTS
engines evaluate many leaves per forward pass and reach **~10k–80k nodes/second** (Leela; AlphaZero on
TPUs) — 20–100× our batch-1 throughput. We implemented and measured that fix (next), and it changes
none of the strength conclusions, only their price.

**GPU utilisation — batch-1 is an implementation *limitation*, and we measured the fix.** The batch-1
search (obs 2) leaves the M3 Ultra's cores **idle between launches** at only **~600 nps**, so the
latencies above are **pessimistic upper bounds**: strength is fixed by **N (nodes searched)**, latency
by **N ÷ throughput**, and our throughput is on the floor.

The standard fix is **batched-leaf evaluation** (gather many tree leaves via *virtual loss*, evaluate
them in a single GPU launch). We implemented it (`BatchedMCTSPlayer`, `search.py`) and **measured** it
on the same net and positions — the same nodes searched, only the launch pattern changed:

| Search | Throughput | Speedup vs batch-1 |
|---|---:|---:|
| **batch-1** (as used above) | 633 nps | 1.0× |
| batched, 16 leaves/launch | 3.9k nps | 6.2× |
| batched, 32 leaves/launch | 4.7k nps | 7.5× |
| batched, 64 leaves/launch | 5.5k nps | 8.6× |
| batched, 128 leaves/launch | 7.4k nps | **11.6×** |

*(clean solo-GPU measurement on the M3 Ultra; a first reading taken under concurrent load over-stated
the ratio, so we report the solo figures.)* A **~6–12×** per-move speedup at identical strength is
enough to run **MCTS-3200 (~0.4 s/move batched) well below today's batch-1 MCTS-800 (~1.3 s)**. **The
batch-1 numbers should therefore be read as a ceiling on cost, not the method's efficiency**, and the
true strength-vs-latency curve sits well to the left of the one we plot.

Two caveats. First, the gain is **hardware-dependent** — it equals the *idle parallelism you can
reclaim*: the wide M3 Ultra leaves much for a 14 MB batch-1 workload, but on a small GPU/CPU or an
accelerator already **saturated by a large network** (AlphaZero/Leela), there are no idle cores and
deeper search costs full, **linear** price. Second, batched selection uses momentarily stale tree
stats, so it is a hair less sample-efficient per node — second-order, not changing the
order-of-magnitude speedup.

### 4.5 Does a bigger evaluator raise the ceiling? (a capacity sweep)

Search still climbs at 12800, but each doubling buys less, so the route higher is a *better
evaluator*. We **add parameters** at fixed depth (8) and identical recipe — only width varies.

**A correction first.** An initial screen mislabelled a width-136 net "2×"; it is in fact **1.4×**
(4.81M params), because the policy/value heads scale ~linearly with width, so total parameters grow
far slower than the conv body's width². On a fixed **~50M-position** subset:

| Net | Params | raw policy | MCTS-800 |
|---|---:|---:|---:|
| 1× (w96) | 3.45M | 2298 | 2631 |
| 1.4× (w136) | 4.81M | 2366 | **2649** |
| Δ | +1.4× | +68 | **+18** |

**+18 Elo with search — not significant within our ±107 uncertainty.** For contrast, *8× more data*
moves the same architecture **~+90**.

**But that screen is confounded** — both nets saw only ~50M positions, so a wider net had little
extra signal to fill its room. Repeating on the **full ~394M data**, matched to the 2734 baseline:

| Capacity (full data) | Params | MCTS-800 | Δ vs 1× |
|---|---:|---:|---:|
| 1× (w96) | 3.45M | 2734 | — |
| 1.4× (w136) | 4.81M | 2794 | +60 |
| **2× (w184)** | **7.04M** | **2766** | **+32** |
| 4× (w288) | 14.2M | *training* | — |

**The curve is flat within noise** — 2734 / 2794 / 2766 all sit inside ±82 of each other, with the 2×
even a touch *below* the 1.4×. So the 1.4×'s +60 was noise, not a rising trend: **doubling the
parameters on full data adds nothing significant** — capacity stays inert even when well-fed, which
*confirms* rather than softens "thinking, not growing". One honest caveat: the 2×'s raw policy (2413)
is slightly below the smaller nets' (~2448), hinting the larger net is mildly **under-trained at a
fixed 1-epoch budget** — so "capacity is inert" holds at *matched training*, not matched convergence
(the 4× point, still training, will test the endpoint). The scoped claim stands: **capacity is the
weakest lever *in this data regime***, not "parameters never matter" — at AlphaZero/LLM scale,
capacity-bound with abundant data, more parameters clearly help.

The study's lever ranking, all same-ladder:

| Lever | Elo moved | Cost |
|---|---:|---|
| **Search** (open → 12800 sims) | **+286, then ~+55 / doubling** (log-linear, unsaturated) | ×2 latency / doubling |
| **Data** (10 shards → full 79) | **~+90** | 8× training data |
| **Capacity** (1× → 2×, **full data**) | **~0** (n.s.; flat 2734/2794/2766) | more params & compute |

**Search dominates, data second, capacity last in this regime** — the sharpest reading being the
thesis: **one lever binds at each stage; an experiment tells you which. Here it was data and search,
not capacity.**

---

## 5. Stage 3 — Self-Learning (no external engine, no labels)

**Why this stage is the whole point.** Stages 1–2 leaned on Stockfish labels — a shortcut that exists
only because chess already has a superhuman evaluator and a labeled database. The generic goal is the
opposite: a **new field with no data and no evaluator** (a novel game, an unsolved control/scheduling
problem, a design task), where you can neither imitate a teacher nor score positions with an
off-the-shelf engine — you have only the **environment's rules**. Stage 3 discards every external
crutch (no Stockfish, no labels) to test whether strength can be **bootstrapped from self-play and
outcomes alone** — the case for any genuinely new problem. The loop: the net plays itself with MCTS,
trains toward the visit distribution (policy) and game result (value), and repeats. We studied five
approaches — self-play, a self-referential ladder, a committee, evolution, and weight merging — and
they converge on one story.

### 5.1 Self-play expert iteration (AlphaZero-style [Silver et al. 2018; Anthony et al. 2017])
Two failure modes were fixed: **cold-start draw-collapse** (a weak net can't force mates, so games
drift to draws and the value head gets no signal — fixed with Dirichlet root noise) and **warm-start
forgetting** (hard training on tiny self-play slices overwrote the supervised policy, 2150→1407 —
fixed with a replay buffer + gentle LR). Stabilized and parallelized to 16× throughput, the raw policy
**plateaus ~1950–2030 — below the ~2150 supervised baseline.**

**Negative result (clean).** At two-machine scale, **self-play converges *below* supervision.**
Strength does rise with game volume — a real scaling signal — but the achievable volume plateaus under
the 394M-supervised net; crossing it needs orders-of-magnitude more games (AlphaZero used ~1000× ours).
**Scale is the binding constraint.**

<svg viewBox="0 0 620 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="300" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Stage 3 — self-play plateaus below the supervised baseline</text>
  <line x1="60" y1="250" x2="590" y2="250" stroke="#333" stroke-width="1.5"/>
  <line x1="60" y1="250" x2="60" y2="45" stroke="#333" stroke-width="1.5"/>
  <text x="30" y="150" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 30 150)">raw policy Elo</text>
  <text x="325" y="285" text-anchor="middle" font-size="11" fill="#333">self-play iterations</text>
  <line x1="60" y1="82" x2="590" y2="82" stroke="#31a354" stroke-dasharray="5 3" stroke-width="1.5"/>
  <text x="585" y="77" text-anchor="end" font-size="10" fill="#31a354">supervised ~2150</text>
  <polyline points="80,60 150,226 260,133 380,116 560,110" fill="none" stroke="#c51b8a" stroke-width="2.5"/>
  <circle cx="80" cy="60" r="4" fill="#c51b8a"/><circle cx="150" cy="226" r="4" fill="#c51b8a"/><circle cx="260" cy="133" r="4" fill="#c51b8a"/><circle cx="560" cy="110" r="4" fill="#c51b8a"/>
  <text x="150" y="240" text-anchor="middle" font-size="9" fill="#c51b8a">1407 (forgetting)</text>
  <text x="470" y="103" text-anchor="middle" font-size="9" fill="#c51b8a">plateau ~2000</text>
</svg>

### 5.2 Committee / diversity — teacher-free confidence (new)
If the bottleneck is the *evaluator*, the cheap way to improve one without a bigger net is to
**ensemble diverse nets**. We tested whether **independently-started models diverge (disagree →
uncertain) or converge (agree → likely correct)** — making agreement a confidence meter with no
oracle.

**Validated.** Over 400 positions scored against Stockfish depth-12 (measurement only), agreement
strongly predicts correctness:

| members agree | centipawn-loss ↓ | matches SF-best ↑ (same-arch / diverse) |
|:---:|:---:|:---:|
| all disagree | 77.9 | 22% / 37% |
| majority | 40.0 | 49% / 58% |
| unanimous | **19.7** | **65% / 94%** |

<svg viewBox="0 0 620 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="300" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Stage 3 — committee agreement predicts correctness</text>
  <line x1="70" y1="250" x2="590" y2="250" stroke="#333" stroke-width="1.5"/>
  <line x1="70" y1="250" x2="70" y2="45" stroke="#333" stroke-width="1.5"/>
  <text x="34" y="150" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 34 150)">matches SF-best move</text>
  <text x="330" y="285" text-anchor="middle" font-size="11" fill="#333">committee agreement</text>
  <polyline points="150,178 330,137 510,67" fill="none" stroke="#756bb1" stroke-width="2.5"/>
  <circle cx="150" cy="178" r="4" fill="#756bb1"/><circle cx="330" cy="137" r="4" fill="#756bb1"/><circle cx="510" cy="67" r="4" fill="#756bb1"/>
  <text x="510" y="60" text-anchor="middle" font-size="9" fill="#756bb1">94%</text>
  <text x="540" y="80" font-size="9" fill="#756bb1">diverse</text>
  <polyline points="150,207 330,155 510,123" fill="none" stroke="#2ca25f" stroke-width="2.5" stroke-dasharray="5 3"/>
  <circle cx="150" cy="207" r="4" fill="#2ca25f"/><circle cx="330" cy="155" r="4" fill="#2ca25f"/><circle cx="510" cy="123" r="4" fill="#2ca25f"/>
  <text x="540" y="126" font-size="9" fill="#2ca25f">same-arch</text>
  <text x="150" y="268" text-anchor="middle" font-size="9" fill="#666">disagree</text>
  <text x="330" y="268" text-anchor="middle" font-size="9" fill="#666">majority</text>
  <text x="510" y="268" text-anchor="middle" font-size="9" fill="#666">unanimous</text>
</svg>

**But plurality voting does *not* reliably de-bias — a committee-size sweep (3/5/7/9 agents) shows
why.** Growing the committee from a correlated conv-soft trio outward:

| agents | added views | consensus CPL | best member | oracle |
|---|---|---:|---:|---:|
| 3 | conv-soft ×3 | 54.7 | 49.8 | 29.6 |
| 5 | +MLP +hard-objective | **48.0** | 46.8 | 22.2 |
| 7 | +2 conv-soft (data slices) | 54.8 | 52.2 | 20.8 |
| 9 | +MLP +hard | 56.0 | 53.1 | 19.9 |

Three findings. (i) **Plurality never clearly beats the best member** — consensus is slightly *worse*
at every size, gaps (~1–5 CPL) inside the ~±3 CPL measurement noise. (ii) **Balance beats count** —
the 5-agent committee (diverse MLP + hard-objective = 40% of the vote) is best; *adding correlated
members* (conv-soft data-slices at 7, 9) lets that bloc dominate and reverts the gain. **More agents ≠
better.** (iii) **The oracle
(best member per position, ~20–30 CPL) is 2–3× better than the vote** — the diversity *contains* the
information, but plurality *cannot extract it*, because it is dominated by the largest correlated
bloc.

**Lesson.** The committee robustly gives a **confidence meter** (agreement→correctness) but a **weak
aggregator**: plurality does not reliably beat the best member. The large oracle headroom needs a
**better aggregator** (soft-averaging, confidence-weighted routing) and **balanced**, not merely
numerous, diversity — a de-biased ensemble is the most promising route to the ceiling search and
self-play can't reach, but plurality is not it.

<svg viewBox="0 0 620 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="320" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1a2a3a">Stage 3 — committee size: plurality never beats the best member; oracle unrealized</text>
  <line x1="70" y1="270" x2="560" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="70" y1="270" x2="70" y2="42" stroke="#333" stroke-width="1.5"/>
  <text x="30" y="160" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 30 160)">centipawn loss ↓ better</text>
  <text x="315" y="300" text-anchor="middle" font-size="11" fill="#333">committee size (agents)</text>
  <polyline points="130,239 270,206 410,240 550,245" fill="none" stroke="#e6550d" stroke-width="2.5"/>
  <circle cx="130" cy="239" r="4" fill="#e6550d"/><circle cx="270" cy="206" r="4" fill="#e6550d"/><circle cx="410" cy="240" r="4" fill="#e6550d"/><circle cx="550" cy="245" r="4" fill="#e6550d"/>
  <text x="270" y="199" text-anchor="middle" font-size="9" fill="#e6550d">48.0</text>
  <text x="583" y="245" text-anchor="end" font-size="9" fill="#e6550d">consensus (vote)</text>
  <polyline points="130,215 270,200 410,227 550,231" fill="none" stroke="#2c7fb8" stroke-width="2" stroke-dasharray="5 3"/>
  <circle cx="130" cy="215" r="3.5" fill="#2c7fb8"/><circle cx="270" cy="200" r="3.5" fill="#2c7fb8"/><circle cx="410" cy="227" r="3.5" fill="#2c7fb8"/><circle cx="550" cy="231" r="3.5" fill="#2c7fb8"/>
  <text x="583" y="214" text-anchor="end" font-size="9" fill="#2c7fb8">best member</text>
  <polyline points="130,116 270,80 410,73 550,69" fill="none" stroke="#2ca25f" stroke-width="2.5"/>
  <circle cx="130" cy="116" r="4" fill="#2ca25f"/><circle cx="270" cy="80" r="4" fill="#2ca25f"/><circle cx="410" cy="73" r="4" fill="#2ca25f"/><circle cx="550" cy="69" r="4" fill="#2ca25f"/>
  <text x="583" y="72" text-anchor="end" font-size="9" fill="#2ca25f">oracle (best/position)</text>
  <text x="270" y="132" text-anchor="middle" font-size="8.5" fill="#888" font-style="italic">5 = best balance</text>
  <text x="315" y="180" text-anchor="middle" font-size="9" fill="#999" font-style="italic">gap the vote can't capture ↕</text>
  <text x="130" y="286" text-anchor="middle" font-size="9" fill="#666">3</text>
  <text x="270" y="286" text-anchor="middle" font-size="9" fill="#666">5</text>
  <text x="410" y="286" text-anchor="middle" font-size="9" fill="#666">7</text>
  <text x="550" y="286" text-anchor="middle" font-size="9" fill="#666">9</text>
</svg>

**A third combination — averaging *evaluations inside the search* — also fails.** Averaging the K
models' value at every MCTS leaf, at equal compute (ensemble at 200 sims = 600 passes/move vs a single
model at 600 sims), scored **2222 vs 2282, a −60 Elo *loss*:** the ensemble searches 3× less tree, and
de-biasing correlated evaluations doesn't justify tripling per-leaf cost. So all three combinations —
plurality, weight merging (§5.4), and in-search averaging — fail to beat a single model, for one root
cause: **the members' errors are too correlated to cancel.** Genuinely independent evaluators
(cross-family, cross-data) are the prerequisite.

### 5.3 Evolution — mutate / play / score (a plateau-escape attempt)

Gradient self-play optimizes a *proxy* (the net's own biased targets); we tested whether
**derivative-free evolution** — optimizing the *true* objective, "did this mutant win games" — could
escape the plateau. Each generation: mutate the plateaued net into 16 offspring (Gaussian weight
noise), play each against a **frozen copy** (a fixed anchor, so fitness *is* "how well do you beat the
plateau"), and crown the best only if it survives a confirmation match. A first version selecting
against the *moving* champion drifted **downward** — beating your immediate parent is non-transitive
in chess; the fixed anchor fixes that.

**Result: a null, and a methodological warning.** The run *appeared* to escape — champ-vs-plateau
climbed to 0.567 (+47 Elo), passing a 120-game confirmation — but a **clean 400-game, low-temperature
re-match found the "evolved" champion is −24 to −41 Elo *worse*.** The gain was an artifact of noisy
high-temperature fitness over small samples with best-of-16 selection bias — a **phantom** that
vanished under proper measurement; annealing the mutation scale (σ 0.03→0.12) found nothing better. So
evolution did not escape either: neither gradient self-play, a self-referential ladder, nor evolution
crosses the ~2000 wall, and since the same net reaches ~2150 under *supervised* labels, the wall is
**not capacity** but the ceiling of any *self-generated* signal. (Practitioner note: relative-fitness
selection over small stochastic samples manufactures phantom gains — trust only a large, low-variance
re-measurement.)

<svg viewBox="0 0 620 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="300" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1a2a3a">Stage 3 — evolution: the "escape" was a phantom of noisy fitness</text>
  <line x1="90" y1="150" x2="560" y2="150" stroke="#333" stroke-width="1.5"/>
  <text x="565" y="154" font-size="9" fill="#666">0 (= plateau)</text>
  <text x="34" y="150" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 34 150)">Elo vs plateaued net</text>
  <rect x="170" y="56" width="120" height="94" fill="#e6550d" fill-opacity="0.28" stroke="#e6550d" stroke-width="2" stroke-dasharray="5 3"/>
  <text x="230" y="48" text-anchor="middle" font-size="13" font-weight="bold" fill="#e6550d">+47</text>
  <text x="230" y="175" text-anchor="middle" font-size="9.5" fill="#e6550d">noisy 120-game fitness</text>
  <text x="230" y="188" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#e6550d">(phantom)</text>
  <rect x="360" y="150" width="120" height="60" fill="#555555"/>
  <text x="420" y="226" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">−30</text>
  <text x="420" y="130" text-anchor="middle" font-size="9.5" fill="#333">clean 400-game re-match</text>
  <text x="420" y="118" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#333">(reality)</text>
  <text x="310" y="270" text-anchor="middle" font-size="10" fill="#999" font-style="italic">the apparent escape reversed under proper measurement — no self-generated signal crossed the wall</text>
</svg>

### 5.4 Model merging — can we *average* diverse models into one?

The weight-space alternative to a voting committee is to **average coefficients into one network**.
We tested it on conv-96×8 members from different seeds/data/objectives, scored by mean **CPL** vs
Stockfish depth-12 (members ~60 ≈ 2000-level; ~260 ≈ random):

| merge | starting points | CPL ↓ | verdict |
|---|---|---:|---|
| naive average | different-start (diverse) | 261 | collapse |
| Git Re-Basin aligned | different-start (diverse) | 268 | still collapse |
| naive average (**model soup**) | **same init** | 58.8 | works (~parent) |
| aligned (net + permuted twin) | identical | 72 | perfect recovery *(verification)* |

**Naive averaging of different-start nets collapses** (261): independent nets sit in different loss
basins related by neuron permutations, so averaging misaligned neurons cancels signal. The known fix,
**permutation alignment (Git Re-Basin)**, we implemented and **verified correct** — it recovers a net
*exactly* from a known random permutation (72 CPL). Yet on real different-start members it **still
collapses** (268), for a fundamental reason: Git Re-Basin assumes nets learn the *same features in a
different order*; ours learned **genuinely different features** (different seeds *and* data *and*
objectives), which no permutation aligns. A **model soup** (children from the *same* checkpoint)
averages fine (58.8), because a shared start keeps them in one basin.

**Conclusion.** Weight-averaging works only within a shared basin. The very diversity that makes an
ensemble valuable (§5.2) is what makes the members' **weights un-averageable** — diverse models
combine at *inference*, not in weight space; so the "better aggregator" must live at inference
(soft-averaging, confidence routing), not in merging.

---

## 6. Comparison to existing systems — parameters, performance, and speed

| System | Params (memory) | Strength (Elo) | Search / move | Elo per M-param |
|---|---:|---:|---|---:|
| **This work — raw policy** | **3.45M** (14 MB) | ~2150 | 1 forward pass (~1.5 ms) | ~620 |
| **This work — + MCTS-800** | **3.45M** (14 MB) | **~2800** | 800 net passes (~1.0 s; ~0.6 s cascaded) | **~810** |
| Maia (human-like) | ~few M | ~1100–1900 | 1 forward pass | ~300–500 |
| AlphaZero (chess, 2017) | ~40–90M | ~3400+ | 800 MCTS sims (big-net passes) | ~40–85 |
| Leela Chess Zero (modern) | ~100–400M+ | ~3500+ | ~1–8k MCTS nodes | ~10–35 |
| Stockfish (NNUE) | ~tens of M (quantized) | ~3600+ | **millions of alpha-beta nodes/s** | ~100–150 |

*Systems compared:* AlphaZero [Silver et al. 2018], Leela Chess Zero, Stockfish, Maia
[McIlroy-Young et al. 2020], KataGo [Wu 2019]; test-time-compute scaling in games [Jones 2021].

**Two axes — size and speed.**
- **Parameters:** our net is **~10–25× smaller than AlphaZero, 30–100× smaller than large Leela**,
  yet reaches ~2800 with search — the extreme point on Elo-per-million-parameters. This is an
  **efficiency framing, not a superiority claim**: top engines are 600–800 Elo stronger and optimize
  *absolute* strength, not parameters-per-Elo; the point is only that parameter count is *not* what
  buys their last few hundred Elo (a better value function and far more search are).
- **Speed:** AlphaZero/Leela use a similar sim count but each sim is a **big-net** pass (10–100× our
  network); our sim is a 14 MB pass (~1.5 ms), cascade-recoverable a further ~1.6–4.8×. Our batch-1
  search runs at only ~600 nps vs ~10k–80k batched (§4.4). Stockfish is the opposite regime — a tiny
  quantized net at millions of nodes/s. The structure is identical throughout — a learned **evaluator
  queried by search** — and **parameter count is not what separates them.**

<svg viewBox="0 0 620 320" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="320" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Parameter efficiency — strength vs network size (log scale)</text>
  <line x1="70" y1="270" x2="590" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="70" y1="270" x2="70" y2="45" stroke="#333" stroke-width="1.5"/>
  <text x="34" y="160" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 34 160)">strength (Elo)</text>
  <text x="330" y="305" text-anchor="middle" font-size="11" fill="#333">parameters (millions, log)  →  bigger net</text>
  <text x="110" y="286" text-anchor="middle" font-size="9" fill="#888">3M</text>
  <text x="300" y="286" text-anchor="middle" font-size="9" fill="#888">30M</text>
  <text x="470" y="286" text-anchor="middle" font-size="9" fill="#888">300M</text>
  <circle cx="118" cy="153" r="6" fill="#2c7fb8"/><text x="118" y="143" text-anchor="middle" font-size="9" font-weight="bold" fill="#2c7fb8">This work ~2800</text>
  <text x="118" y="170" text-anchor="middle" font-size="8" fill="#2c7fb8">3.45M</text>
  <circle cx="150" cy="297" r="5" fill="#888"/><text x="150" y="255" text-anchor="middle" font-size="9" fill="#666"></text>
  <text x="165" y="292" font-size="9" fill="#666">Maia ~1600</text>
  <circle cx="300" cy="57" r="5" fill="#e6550d"/><text x="300" y="50" text-anchor="middle" font-size="9" fill="#e6550d">Stockfish ~3600</text>
  <circle cx="360" cy="81" r="5" fill="#756bb1"/><text x="360" y="74" text-anchor="middle" font-size="9" fill="#756bb1">AlphaZero ~3400</text>
  <circle cx="455" cy="69" r="5" fill="#31a354"/><text x="470" y="64" text-anchor="middle" font-size="9" fill="#31a354">Leela ~3500</text>
  <text x="130" y="230" font-size="9" fill="#2c7fb8" font-style="italic">← far-left/high = most parameter-efficient</text>
</svg>

**Honest gap.** Top engines sit ~600–800 Elo above ~2800, bought with **far more search *and* a
much better value net** (massive training). Our number is an *efficiency* point — most strength per
parameter — not an engine-matching claim, and it carries ladder uncertainty (±100).

---

## 7. Discussion — the unifying conclusion

Across every experiment, one factor was the recurring binding constraint — sharper than "the network
is the bottleneck": **the *quality of the information reaching the evaluator* (its training signal),
not the loop around it** (an empirical regularity of our regime, not a theorem). The organizing law is
**strength = evaluator × search**, and it explains everything once we define *information* precisely:
by **new information** we mean **novel empirical data from *outside* the closed system of the net and
its training set** — not a re-encoding of what is already latent. Supervision and scale inject it
directly; every other method divides on one question — **does it have an external ground-truth oracle
to query?**

- **Within-move search, voting, and merging do not** — they operate on fixed representations, so they
  only *extract* information already present (cut variance, sharpen, filter), never create what is
  absent.
- **Self-play and evolution can — but only through the environment.** In a **closed-form,
  perfect-information** game the *rules are a perfect external oracle*: search queries terminal
  outcomes outside any dataset — exactly how AlphaZero/Leela inject information no human game contains
  and surpass human play. This is a property of the *environment*: in an **open-ended, semantic**
  domain (language) with no external verifier, the identical loop has nothing to query and collapses
  into hallucination — it can only redistribute.
- **Our chess result sits in the oracle-rich regime and still plateaued** — because at two-Mac-Studio
  compute the search was too weak to extract much from that oracle, *not* because it was absent. A
  **scale** limit, not evidence against self-play. *Mechanistically*, the escape is search depth:
  at AlphaZero scale a **massive per-move search budget** turns the game rules into a strong enough
  oracle to systematically discover deep tactical/strategic truths, generate targets that *exceed* the
  current net, and encode them by training — a self-reinforcing loop that breaks the
  weak-search/draw-collapse plateau. Our budget produces self-targets no better than the net that made
  them, so the loop has nothing to climb; the missing ingredient is **oracle-extraction compute**, not
  a different algorithm.

None of this makes redistribution *useless*: variance reduction, sharpening, and filtering are how you
**reach** a ceiling cheaply — indispensable engineering. The narrow claim is about the *absolute*
ceiling: **only information from outside the closed system can raise it.**

- **Architecture > parameters.** The right prior (spatial locality + weight sharing) beats raw width —
  a 14 MB conv reaches strength a 3× larger MLP cannot.
- **Search > scale, but search cannot exceed the net.** MCTS bought +650 Elo (2150→2800) at zero extra
  parameters and out-scaled fixed depth — yet flat MCTS and the cascade hit the *same* wall on the same
  net, because search cuts the net's *variance* (averaging noisy calls), not its *bias* (systematic
  blind spots). The cascade's win was **efficiency (up to 4.8× cheaper), not strength.**
- **No self-generated signal crosses the wall — we tried three.** Gradient self-play, a self-referential
  ladder, and evolution (given the fairest shot; its +47-Elo "escape" was noise that reversed to −30)
  all plateau. The same net reaches ~2150 under supervised labels, so the wall is **not capacity** but
  the ceiling of a system's signal about *itself* — a **scale** statement (at AlphaZero-scale compute,
  self-play bootstraps far past this).
- **A better *evaluator* is the only lever — and harder than it looks.** The committee's agreement
  predicts correctness, but **plurality voting never clearly beats the best member** (correlated blocs
  dominate; balance beats count); the per-position oracle is **2–3× better than the vote**, so the
  information is there but needs a **better aggregator** (soft-averaging / confidence-routing), not more
  agents.
- **Control-theory unification.** Open loop = feedforward; closed loop = MPC; self-play = iterative
  learning control — and in all three the learned *evaluator* caps performance.

### 7.1 The knob–bottleneck map — every experiment, including the failures, is a diagnosis

Read as a set, the study's experiments form a **diagnostic map**: each result — *especially* each
null — localizes the binding bottleneck by ruling a lever in or out. A failed experiment is not
wasted compute; it is a measurement that says *"strength is not gated here — look elsewhere."*

| Knob turned | Result | Diagnosis → what binds | Move it implies |
|---|---|---|---|
| **Architecture** (conv vs MLP) | large gain | bias-bound — wrong prior caps a big net | fix the inductive bias *before* scaling |
| **Capacity** (2× params @ 50M) | ~0 (null) | *not* capacity-bound in this data regime | add data, not parameters (here) |
| **Data** (10 → 79 shards) | +~90 | data/signal-bound | more, more-diverse labels |
| **Search amount** (open → 12800 sims) | +286, then ~+55/doubling (log-linear) | search extracts value; evaluator caps its *return* | keep searching; raise the evaluator to lift the ceiling |
| **Search allocation** (cascade shape) | flat (±noise) | *not* allocation-bound at fixed budget | reallocate for **speed**, not strength |
| **Search implementation** (batch-1) | latency only | throughput-bound by engineering | batch leaves → 6–12× speed, same strength |
| **Self-play signal** | plateau | self-signal-quality-bound | can't exceed its own signal at this scale |
| **Selection pressure** (evolution) | phantom, reversed | *measurement-noise*-bound (fake gain) | re-measure cleanly before believing |
| **Aggregation** (voting 3–9 agents) | no gain | correlated-error-bound | need *diversity*, not more voters |
| **Aggregation** (ensemble-eval, merging) | no gain | combining ≠ creating knowledge | extracts existing signal, creates none |
| **Self-distillation** (fixed set) | dropped | data-*diversity*-bound (overfit) | many diverse positions, not repetition |

Three things fall out:

**1. Nulls are the most information-dense results.** A knob that moves nothing *localizes* the
bottleneck away from itself — the parameter null said "capacity isn't binding (here)," the cascade
flat-line "allocation isn't," the voting sweep "more agents isn't." The one trap is **mistaking noise
for signal** (evolution's phantom +47 that reversed to −30), so every promising delta was re-measured.

**2. The binding lever *moves* as you relieve it.** Strength is a *chain*: bias- then capacity-bound
(open-loop) → search-bound → evaluator-bound (by its **data**, not parameter count). You cannot skip a
link — parameters into a data-bound net, or sims into a saturated search, buy almost nothing.

**3. The map is the roadmap.** The diagnosis *is* the next move: open-loop capped ⇒ add search;
search saturated ⇒ a better *evaluator* (more/better data); self-signal plateaued ⇒ a better *signal
source* (external labels, or AlphaZero-scale self-play). **Identify the binding lever, relieve exactly
that, re-measure, repeat** — the loop that transfers to a genuinely new domain with no engine or
dataset to imitate.

---

## 8. Limitations and Future Work
- Absolute Elo carries systematic uncertainty near the ladder top; relative same-ladder gains are
  the robust claims.
- Stage 3 is compute-limited (two machines); results characterize the *small-scale* regime.
- Stage-3 fitness/aggregation is noise-limited at our scale: relative-fitness selection with small,
  stochastic samples produced a **phantom** evolution "escape" that reversed under a 400-game
  re-match, and plurality de-biasing gaps sit inside the ~±3 CPL measurement noise. Larger,
  lower-variance evaluation is needed to resolve small effects.
- **Next levers, in priority order:** (1) a **better value function** (the true ceiling) via
  more/better search-labeled data or large-scale self-play; (2) a **better ensemble aggregator** —
  soft-averaging and confidence-weighted *routing* (per-position oracle 2–3× above plurality) with
  *balanced* cross-family diversity, not more agents; (3) **batched/parallel MCTS** and transposition
  tables for throughput; (4) endgame tablebases and tuned `c_puct`. All converge on one place:
  **improve the evaluation, and both the closed-loop and self-play ceilings rise together.**

---

## 9. Conclusion

<svg viewBox="0 0 680 370" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <defs>
    <marker id="cah" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#444"/></marker>
    <marker id="cap" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#756bb1"/></marker>
  </defs>
  <rect x="0" y="0" width="680" height="370" fill="#ffffff"/>
  <text x="340" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#1a2a3a">strength = evaluator × search</text>
  <text x="340" y="50" text-anchor="middle" font-size="11" fill="#666">the evaluator sets the ceiling; search sets how close you get to it</text>

  <rect x="28" y="95" width="156" height="62" rx="8" fill="#eef4fb" stroke="#2c7fb8" stroke-width="1.5"/>
  <text x="106" y="120" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1a2a3a">INFORMATION</text>
  <text x="106" y="139" text-anchor="middle" font-size="9.5" fill="#555">labels · data · scale</text>

  <rect x="262" y="95" width="156" height="62" rx="8" fill="#eef4fb" stroke="#2c7fb8" stroke-width="2.8"/>
  <text x="340" y="117" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#1a2a3a">EVALUATOR</text>
  <text x="340" y="133" text-anchor="middle" font-size="9.5" fill="#555">net quality = the ceiling</text>
  <text x="340" y="148" text-anchor="middle" font-size="8.5" fill="#999">(the bottleneck)</text>

  <rect x="496" y="95" width="156" height="62" rx="8" fill="#fdf0e6" stroke="#e6550d" stroke-width="1.5"/>
  <text x="574" y="120" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#7a2e06">PERFORMANCE</text>
  <text x="574" y="139" text-anchor="middle" font-size="9.5" fill="#555">how strong it plays</text>

  <line x1="184" y1="126" x2="258" y2="126" stroke="#444" stroke-width="1.6" marker-end="url(#cah)"/>
  <text x="221" y="118" text-anchor="middle" font-size="9" fill="#2c7fb8">sets ceiling</text>
  <line x1="418" y1="126" x2="492" y2="126" stroke="#444" stroke-width="1.6" marker-end="url(#cah)"/>
  <text x="455" y="118" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#e6550d">× search</text>
  <text x="455" y="145" text-anchor="middle" font-size="8" fill="#999">extracts value</text>

  <rect x="205" y="255" width="270" height="56" rx="8" fill="#f2eef8" stroke="#756bb1" stroke-width="1.5"/>
  <text x="340" y="278" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#463b7a">SELF-LEARNING</text>
  <text x="340" y="296" text-anchor="middle" font-size="9" fill="#555">self-play · evolution · voting · merging</text>

  <path d="M 340 255 L 340 159" fill="none" stroke="#756bb1" stroke-width="1.6" marker-end="url(#cap)"/>
  <text x="352" y="205" font-size="9" fill="#756bb1">feeds back to the evaluator …</text>
  <text x="352" y="222" font-size="9" font-weight="bold" fill="#b5322e">… but lifts the ceiling ONLY if it adds NEW information;</text>
  <text x="352" y="235" font-size="9" fill="#b5322e">rearranging what is already there just plateaus.</text>

  <text x="340" y="345" text-anchor="middle" font-size="9.5" font-style="italic" fill="#666">search cuts variance, not bias — the ceiling is the quality of the information the evaluator was given</text>
</svg>

This paper treated playing strength as an **efficient-allocation problem**: given a fixed budget of
parameters, data, search, and latency, how do you spend it for the most capability? Measured resource
by resource, the efficient mix is rarely "more parameters" — a 14 MB net reaches **~2800 Elo by
*thinking* (search), not *growing*** — the same capability, a far cheaper mix. This is *efficiency*,
not a denial of scale: the first full-data capacity point already nudges up (1× 2734 → 1.4× 2794), and
**if the 2×/4× points keep climbing, parameters become co-dominant once data-starvation is relieved.**
Adaptive MCTS out-scales fixed-depth search; the cascade matches it at 4.8× less compute (strength
bought back as *latency*, not parameters). Self-learning is honestly negative: self-play, a
self-referential ladder, and evolution all **fail to cross the ~2000 plateau** (evolution's escape was
noise), and plurality committees don't reliably de-bias — though **agreement is a robust teacher-free
confidence signal**.

Above all, the **method** transfers: at each stage a *single lever binds* and only an experiment
reveals which. Open-loop was **capacity-bound**; adding search made us **search-bound** (search paying
**log-linearly, ~+55 Elo/doubling, unsaturated through 12800**); then the evaluator binds — and at our
data scale by its **data**, not its parameter count (1.4× more params bought **~0 Elo**; a full
1×/1.4×/2×/4× sweep is running). The recurring constraint, from every direction we pushed, was **the
quality of the information reaching the evaluator** (§7): supervision, data, and search help; voting
and merging only reorganize what the net already encodes; self-play and evolution *can* inject
information, but only through an **external oracle**, so our plateau is a **compute-scale limit, not
evidence that self-play fails** (AlphaZero/Leela break past human play with far more of it). What
transfers is a quantified recipe and an explicit **diagnostic** for finding the binding lever.

**Beyond chess.** The decomposition — **capacity, inference-time search, self-generated information**
— is domain-agnostic. Each chess finding is an instance of a general resource-allocation principle,
and maps directly onto other AI systems (we measured only chess; these are the transfer claims):

| Chess finding | General principle | Where it transfers |
|---|---|---|
| **strength = evaluator × search** | capability = model quality × inference-time compute | LLM reasoning (base model × sampling/tree-of-thought); robotics (value net × planning horizon); theorem proving (heuristic × search depth) |
| **cascade: 4.8× cheaper search** | spend a fixed decision-time budget wide→narrow | LLM reasoning under latency SLAs; model-predictive control; any anytime search |
| **search extracts, can't create — needs an oracle** | self-improvement is capped without external ground truth | LLM self-training needs verifiers; RL needs an environment; scientific discovery needs experiments |
| **capacity is the weakest lever when data-starved** | don't scale parameters ahead of data | compute-optimal (Chinchilla) scaling; collect data before growing nets |
| **agreement predicts correctness; voting doesn't de-bias** | consensus is a confidence meter, not an accuracy booster (correlated errors) | ensemble uncertainty / OOD detection; caution on naïve model-averaging |
| **bottleneck diagnostic** | find the binding resource experimentally before investing | design of any resource-constrained AI system |

The *decomposition* and its diagnostic (find the binding lever before investing) are what transfer,
and likely outlast the chess numbers.

**The clearest current echo is in large language models.** The 2024–25 shift to **inference-time
compute** — o1/o3 [OpenAI 2024], DeepSeek-R1 [DeepSeek-AI 2025], reasoning models — is this thesis at
frontier scale: a fixed base model made far stronger by *searching over reasoning at decision time*
(sampling, self-consistency [Wang et al. 2023], tree-of-thought [Yao et al. 2023]) rather than by adding
parameters. The mapping is direct: our *evaluator* ↔ their
**base model/verifier**, our *search* ↔ their **reasoning budget**, our *self-play* ↔ their
**STaR-style self-training** [Zelikman et al. 2022] — with the same limit: search and self-training **convert what the model
already latently knows into better answers; they cannot inject knowledge it never learned.** That is
exactly why reasoning models pair search with **external verifiers or ground-truth reward** (code that
runs, math that checks, tools that return facts) — the oracle that lets the loop add information. A
model with no such oracle should, by our framework, **plateau**.

**A test for what comes next.** The framework is a **predictive filter**: pre-screen any future
*synthetic-data breakthrough* or *recursively self-improving architecture* with one question — **does
it inject information from outside its closed system** (new empirical data, supervision, or an external
oracle: a verifier, a simulator, the physical world)? If yes, it can raise the ceiling; if it only
re-processes what its own models contain, our results predict it will **plateau**. Recursive
self-improvement compounds where an external oracle exists (a game's rules, a theorem checker, a
compiler, a market) and stalls where none does — a claim about the *source* of information, not the
method's ingenuity.
- `chessnet/model.py` — conv/MLP/dual-path + value head. `chessnet/search.py` — alpha-beta,
  MCTS/PUCT, quiescence, the wide→narrow cascade (`MultiStageMCTSPlayer`).
  `chessnet/committee.py` — ensemble inference + agreement signal.
  `chessnet/train.py` — soft/hard + value training. `scripts/selfplay.py` — self-play iteration.
- **Best model:** `runs/conv_value_llm1` (conv-96×8 + value, 3.45M params).
- **Key hyperparameters:** conv width 96, depth 8; lr 5e-4 (train) / 1e-4 (self-play); gradient
  clip 1.0; MCTS c_puct 1.5; Dirichlet α 0.3; replay buffer 120K–300K.

## References

**Search (MCTS / tree search).** Coulom, R. (2006), *Efficient Selectivity and Backup Operators in
Monte-Carlo Tree Search*, Computers and Games. · Kocsis, L. & Szepesvári, C. (2006), *Bandit Based
Monte-Carlo Planning* (UCT), ECML. · Rosin, C. D. (2011), *Multi-Armed Bandits with Episode Context*
(PUCT), Ann. Math. AI. · Browne, C. et al. (2012), *A Survey of Monte Carlo Tree Search Methods*, IEEE
TCIAIG. · Chaslot, G. et al. (2008), *Progressive Strategies for Monte-Carlo Tree Search* (progressive
widening). · Knuth, D. & Moore, R. (1975), *An Analysis of Alpha-Beta Pruning*, Artif. Intell.

**Learned evaluator + search / self-play.** Silver, D. et al. (2016), *Mastering the Game of Go with
Deep Neural Networks and Tree Search* (AlphaGo), Nature. · Silver, D. et al. (2017), *Mastering the Game
of Go without Human Knowledge* (AlphaGo Zero), Nature. · Silver, D. et al. (2018), *A General
Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play* (AlphaZero),
Science. · Anthony, T., Tian, Z. & Barber, D. (2017), *Thinking Fast and Slow with Deep Learning and
Tree Search* (expert iteration), NeurIPS. · Wu, D. J. (2019), *Accelerating Self-Play Learning in Go*
(KataGo), arXiv:1902.10565. · Bertsekas, D. (2022), *Lessons from AlphaZero for Optimal, Model
Predictive, and Adaptive Control*.

**Chess / games modelling & scaling.** McIlroy-Young, R. et al. (2020), *Aligning Superhuman AI with
Human Behavior: Chess as a Model System* (Maia), KDD. · Jones, A. L. (2021), *Scaling Scaling Laws with
Board Games*, arXiv:2104.03113. · Wortsman, M. et al. (2022), *Model Soups*, ICML.

**Inference-time compute in LLMs (the frontier echo).** OpenAI (2024), *Learning to Reason with LLMs*
(o1). · DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
Learning*, arXiv:2501.12948. · Wang, X. et al. (2023), *Self-Consistency Improves Chain-of-Thought
Reasoning in Language Models*, ICLR. · Yao, S. et al. (2023), *Tree of Thoughts*, NeurIPS. · Zelikman, Y.
et al. (2022), *STaR: Bootstrapping Reasoning with Reasoning*, NeurIPS.

**Software & data.** Stockfish (stockfishchess.org) — ladder opponent and label oracle. · Leela Chess
Zero (lczero.org). · Lichess cloud-evaluation database (database.lichess.org) — supervised labels. ·
MLX (github.com/ml-explore/mlx) — training/inference framework.
