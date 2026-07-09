# Efficient Thinking: Tradeoffs in Game-Playing AI
## Parameters vs Search vs Self-Learning — a Three-Stage Chess Study of Where Strength Comes From

**Louay Alsakka** · July 8, 2026

**A white paper on parameter-efficient game AI.**

**Code & trained model:** [github.com/louayalsakka/efficient-thinking](https://github.com/louayalsakka/efficient-thinking)

*The recurring theme is a set of **tradeoffs**: memory vs compute (Stage 1 vs 2), speed vs score
(the search cascade), and diversity vs correlation (the committee) — each one a different way of
spending a fixed budget to buy strength, and each capped by the same wall: the quality of the
learned evaluator.*

---

## Abstract

We study how playing strength in chess arises from three separable sources — **network
capacity, search, and self-learning** — using a deliberately tiny model on modest hardware.
A **3.45-million-parameter (14 MB) convolutional network** plays at **~2150 Elo** as a
single-pass policy; adding **Monte-Carlo Tree Search (MCTS)** lifts the *same weights* to
**~2800 Elo (top-human)** with **zero extra parameters** — strength bought entirely with
compute. We make three further contributions. (1) A direct **MCTS-vs-fixed-depth** comparison:
adaptive MCTS both **beats** alpha-beta at equal compute **and keeps scaling**, while fixed-depth
search plateaus. (2) A **search-efficiency** result: an all-MCTS **wide→narrow cascade** that
funnels the simulation budget through progressively narrower, deeper stages **matches flat MCTS
strength while running up to ~4.8× faster per move** — the same answer for far less compute.
(3) A **teacher-free self-learning** study (self-play, a self-referential ladder, evolution, and
committees) with one robust positive and sober negatives: **model agreement predicts correctness**
(a confidence signal needing no oracle), but *none* of these methods — including outcome-based
evolution — crosses the self-play plateau, and plurality voting does not reliably de-bias. Since the
same net reaches higher under supervised labels, the wall is the quality of *self-generated* signal,
not capacity. Stage 3's constraint — **no engine, no labels** — is
deliberate: it is the setting of any *genuinely new* problem, where no teacher or dataset exists to
imitate, so a recipe that depends on them (as Stages 1–2 do) cannot transfer; only a self-bootstrapping
method can. Throughout, the unifying finding is that **the ceiling is the evaluator (the network),
not the loop around it** — search and self-play *redistribute* the
knowledge in the net; they do not manufacture it.

**Headline:** *A 14 MB model reaches ~2800 Elo by thinking (search), not by growing (parameters) —
and once the search is good, the only wall left is the quality of the evaluation.*

---

## 1. Introduction

Modern game AI conflates three separable questions:
1. **Open loop** — how strong is a single forward pass (pure "intuition")?
2. **Closed loop** — how much does *search* (lookahead on a learned value function) add?
3. **Self-learning** — can the system improve *itself*, with no external teacher?

We treat these as three stages of increasing autonomy and measure each independently. The framing
is explicitly **control-theoretic** (Bertsekas 2022, *Lessons from AlphaZero for Optimal, Model
Predictive, and Adaptive Control*): the open-loop policy is a **feedforward controller**,
closed-loop search is **model-predictive control** (receding-horizon planning with a learned
terminal cost), and self-play is **iterative/adaptive learning control**. Chess is only a
testbed; the goal is a *generic* recipe for sequential decision-making, and this study quantifies
what each ingredient buys.

**Contributions.**
- A parameter-efficiency result: **~2800 Elo from 3.45M params** via search, at constant memory.
- A direct **MCTS-vs-fixed-depth** result: adaptive search beats and out-scales fixed depth.
- A **search-efficiency** result: a wide→narrow MCTS **cascade** matches flat MCTS at up to
  **4.8× less compute per move**, with a clean score/speed trade-off curve.
- An **architecture-beats-scale** finding (convolution ≫ MLP at equal data), with topology sweep.
- A reproducible **negative result** on small-scale self-play (plateaus below supervision).
- A **teacher-free self-learning** study: agreement is a validated **confidence meter**, but
  self-play, a self-referential ladder, **evolution**, and plurality-voting committees all fail to
  cross the plateau — a set of clean negative results with a methodological warning (noisy fitness
  manufactures phantom gains).

---

## 2. Problem Setup and Methods

### 2.1 Representation
- **Position → move** as a 4096-way (64×64 from–to) classification. Promotions default to queen.
- **Side-to-move normalization:** the board is always presented from the mover's perspective
  (mirror + color-swap for Black), so one network serves both players.
- **Encodings:** *onehot* = 64 squares × 12 piece planes + 5 meta bits = **773 floats**
  (used throughout). For convolutions the 773-vector reshapes to **8×8×12 piece planes** + 5 meta
  planes = 17 input channels. We compared this to a compact *packed* encoding (one integer code
  per square, 64+5 = **69 floats** — a 4-bit-per-square board). On an identical MLP, budget, and
  data, **one-hot is decisively stronger — 84.9 vs 142.5 mean centipawn-loss** — because a network
  learns clean categorical piece-type features far more readily than an arbitrary *ordinal* code
  (where code 7 vs 8 carry a spurious "closeness"), and, crucially, convolution *requires*
  per-piece-type planes, which a packed 8×8×1 integer board cannot provide. Packed is a
  compact-but-harder-to-learn ablation; one-hot is the workhorse.
- **No side-to-move feature:** the turn is not a bit — the board is *re-oriented* so the side to
  move is always "White" (mirror + color-swap for Black), so one evaluator serves both players and
  the turn is carried by the orientation. (This is why `Eval(P)` equals `Eval(mirror(P))` exactly,
  and why the value of the move can be read off as `Eval(B) + Eval(null-move) − 1` — the deviation
  from that mirror symmetry; measured mean tempo ≈ +0.15, up to +0.77 in tactical shots and
  negative in zugzwang.)

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

**Topology conclusion.** Only two shapes are competitive — **constant-width** and
**convolution** — and every *taper, factoring, or exotic bottleneck loses badly.* The reason is
inductive bias: a taper or bottleneck destroys positional information before the head can use it,
while convolution **reuses local tactical patterns across the board via weight sharing**, so it
learns a motif once instead of relearning it per square. The practical rule that falls out:
**pick the prior that matches the domain's structure (spatial locality) rather than adding raw
width.** A 3.45M-param conv matched a 14.4M-param MLP using **22× less data** — architecture,
not parameter count, set the strength.

### 3.2 Scaling laws (width and data)
- **Width (MLP, hard):** W1024 (14.4M) → 1449; W2048 (47.7M) → 1501: **+52 Elo for 3.3× params**
  — sharply diminishing. Top-1 accuracy saturates ~0.41 regardless of width.
- **Data (W1024):** 1127 / 1207 / 1300 / 1382 / 1411 at 23 / 47 / 94 / 187 / **375M** positions —
  large early gains, then **+29 on the last doubling** → saturates near the DB's 394M limit.

<svg viewBox="0 0 620 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="300" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Stage 1 — data scaling saturates (MLP W1024, Elo vs positions)</text>
  <line x1="60" y1="250" x2="590" y2="250" stroke="#333" stroke-width="1.5"/>
  <line x1="60" y1="250" x2="60" y2="45" stroke="#333" stroke-width="1.5"/>
  <text x="30" y="150" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 30 150)">Elo</text>
  <text x="325" y="285" text-anchor="middle" font-size="11" fill="#333">positions (millions, log-ish spacing)</text>
  <polyline points="90,232 190,203 290,163 390,133 560,126" fill="none" stroke="#2c7fb8" stroke-width="2.5"/>
  <circle cx="90" cy="232" r="4" fill="#2c7fb8"/><circle cx="190" cy="203" r="4" fill="#2c7fb8"/><circle cx="290" cy="163" r="4" fill="#2c7fb8"/><circle cx="390" cy="133" r="4" fill="#2c7fb8"/><circle cx="560" cy="126" r="4" fill="#2c7fb8"/>
  <text x="90" y="225" text-anchor="middle" font-size="9" fill="#333">1127</text>
  <text x="290" y="156" text-anchor="middle" font-size="9" fill="#333">1300</text>
  <text x="560" y="119" text-anchor="middle" font-size="9" fill="#333">1411</text>
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

## 4. Stage 2 — Closed Loop (search on a value function)

We add a scalar head **Eval(N) ∈ [0,1]** = *expected score for the side to move*
(`P(win)+½P(draw)`), trained on the eval-DB win-probabilities (held-out **MAE 0.088, correlation
0.877** — an excellent value function). Search uses the zero-sum complement identity: our value
after a move is `1 − Eval(child)`, so one network plays both sides; terminal nodes return exact
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

**Result.** Two clean findings. (i) **1-ply search *hurts* a strong policy** — it commits to a
capture without seeing the recapture; depth is where lookahead starts paying. (ii) **MCTS both
beats alpha-beta at equal compute and keeps scaling** where fixed depth plateaus (~2550). Fixed
uniform depth spends the same effort on every branch and *amplifies the value net's noise* at the
leaves; MCTS instead **allocates search adaptively to sharp lines and averages over that noise.**
MCTS is the Stage-2 winner and reaches ~2800 on the fixed 3.45M value net.

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
Given that MCTS wins, we asked whether the **allocation** of a fixed simulation budget matters.
The **cascade** runs MCTS in *stages* with different knobs — a **wide** stage (all moves, high
`c_puct`, few sims) ranks broadly and passes its top-k by visit count to a **narrower, deeper**
stage (fewer moves, more sims, lower `c_puct`), funnelling the budget onto the survivors. A shared
evaluation cache carries value-net calls forward between stages.

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

**Result.** Across all ten funnels the Elo stays inside a **single ±89 noise band** (range
2506–2683, mean ~2580) — **statistically flat** — while speed improves **monotonically to 4.8× at
N = 10.** Funnelling the budget wide→narrow is therefore a **near-pure efficiency win**: it holds
flat-MCTS strength while cutting per-move compute up to ~5×, because the survivors that reach the
deep stages are the same moves flat MCTS would have spent its budget on anyway — the funnel just
stops paying to search moves it has already ranked out. Any softening at the sharpest funnels is
within measurement noise (20 games/rung). The practical dial is simply **more levels = the same
strength, cheaper** — turn it up until the noise floor or your latency budget stops you. (The
fixed-depth *beam* cascade, by contrast, is a genuinely weaker primitive at 2487 — adaptive MCTS
stages matter.)

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
| **GPU cycles/move** | 1 pass, ~1.5 ms | ~800 passes, ~1 s (or ~0.6 s cascaded) |
| **Elo** | ~2150 (capped) | **~2800 (scales with compute)** |

**Stage 1 buys Elo with *memory* and saturates; Stage 2 buys Elo with *GPU cycles* and keeps
climbing** — and the cascade shows most of those cycles were waste (up to 4.8× recoverable). The
central practical result: **strength is compute, not parameters.**

---

## 5. Stage 3 — Self-Learning (no external engine, no labels)

**Why this stage is the whole point.** Stages 1–2 leaned on Stockfish labels — a shortcut that
*only exists because chess already has a superhuman evaluator and a massive labeled database.* The
generic goal is the opposite situation: a **new field where no training data and no existing
evaluator exist** — a novel game, an unsolved control or scheduling problem, a scientific-design
task. There you *cannot* imitate a teacher because there is none, and you *cannot* score positions
with an off-the-shelf engine because none has been built. The only thing you have is the
**environment's rules**: which actions are legal, and, eventually, whether you succeeded. Stage 3
deliberately discards every external crutch — no Stockfish, no labels — to test whether strength
can be **bootstrapped from self-play and outcomes alone.** This is the transferable question:
Stages 1–2 measure what search and capacity buy *when a teacher is available*; Stage 3 measures
whether the recipe can **stand on its own in a domain where Stages 1–2 are simply impossible** —
which is precisely the case for any genuinely new problem.

The loop: the net plays itself with MCTS, trains toward the visit distribution (improved policy)
and game result (value), and repeats. The only external input is the rules. We studied five
approaches — self-play, a self-referential ladder, a committee, evolution, and weight merging — and
they converge on one story.

### 5.1 Self-play expert iteration (AlphaZero-style)
The net plays itself with MCTS, trains toward the visit distribution (improved policy) and game
result (value), and repeats. Two failure modes appeared and were fixed: **cold-start draw-collapse**
(a weak net can't force mates, so games drift to draws and the value head gets no signal — fixed
with Dirichlet root-exploration noise) and **warm-start forgetting** (training hard on tiny
self-play slices overwrote the supervised policy, 2150 → 1407 — fixed with a replay buffer and a
gentle learning rate). Stabilized, and even parallelized to 16× throughput (300K replay buffer,
eval cache), the raw policy **plateaus ~1950–2030 — below the ~2150 supervised baseline.**

**Negative result (clean).** At two-machine scale, **self-play converges *below* supervision.**
Self-play strength *does* rise with game volume — a real scaling signal — but the *achievable*
volume here plateaus under a 394M-supervised net; crossing it needs orders-of-magnitude more games
(AlphaZero used ~1000× ours). **Scale is the binding constraint.**

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
If the bottleneck is the *evaluator*, the cheap way to improve one without training a bigger net
is to **ensemble diverse nets**. We tested the hypothesis that **multiple independently-started
models either diverge (disagree → uncertain) or converge (agree → likely correct)** — making
agreement a self-contained confidence meter with no oracle.

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

Three findings. (i) **Plurality never clearly beats the best single member** — consensus is slightly
*worse* at every size, and the gaps (~1–5 CPL) sit inside the Stockfish-measurement noise (~±3 CPL);
an earlier apparent "consensus beats best" was itself within that noise. (ii) **Balance beats
count** — the 5-agent committee, where the diverse MLP + hard-objective members are 40% of the vote,
is by far the best; *adding correlated members* (the conv-soft data-slice nets at 7 and 9) lets that
bloc dominate the plurality and reverts the gain. **More agents ≠ better.** (iii) **The oracle
(best member per position, ~20–30 CPL) is 2–3× better than the vote** — the diversity *contains* the
information, but plurality *cannot extract it*, because it is dominated by the largest correlated
bloc.

**Lesson.** The committee gives one thing robustly and one thing not: a **confidence meter**
(agreement→correctness, validated repeatedly) — but **plurality voting is a weak aggregator** that
does not reliably reduce error below the best member. Capturing the large oracle headroom needs a
**better aggregator** (soft probability-averaging, or confidence-weighted routing that trusts the
per-position agreement) and **balanced**, not merely numerous, diversity. A de-biased ensemble
evaluator remains the most promising route to lift the ceiling that search and self-play cannot —
but plurality is not it.

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

**And a third way to combine them — averaging *evaluations inside the search* — also fails.** The
most direct use of the committee is to average the K models' value estimates at every MCTS leaf (a
de-biased evaluator *inside* the tree; ELO-weightable; blending *continuous* values, so a single
overconfident member can't dominate). At equal compute — the ensemble at 200 sims (600 passes/move)
vs a single model at 600 sims — it scored **2222 vs 2282, a −60 Elo *loss*:** the ensemble searches
3× less tree, and the small de-biasing from averaging *correlated* evaluations does not justify
tripling the per-leaf cost. **More search beats a marginally-cleaner-but-shallower one.** So all
three ways to combine the committee — plurality voting, weight merging (§5.4), and in-search value
averaging — fail to beat a single model, for the same root cause: **the members' errors are too
correlated to cancel.** Genuinely independent evaluators (cross-family, cross-data), not more of the
same, are the prerequisite.

### 5.3 Evolution — mutate / play / score (a plateau-escape attempt)

Gradient self-play optimizes a *proxy* (the net's own biased value targets); we tested whether
**derivative-free evolution**, which optimizes the *true* objective — did this mutant win games —
could escape the plateau where gradients stalled. From the plateaued net: each generation mutate it
into 16 offspring (Gaussian weight noise), play each against a **frozen copy of the plateaued net**
(a fixed anchor, so "fitness" *is* "how well do you beat the plateau"), and crown the best only if
it survives a larger confirmation match. A first, naive version selecting against the *moving*
champion drifted **downward** — beating your immediate parent is non-transitive in chess and does
not imply getting stronger; the fixed anchor fixes that.

**Result: a null result, and a methodological warning.** The run *appeared* to escape — champ-vs-
plateau climbed to 0.567 (+47 Elo) and passed a 120-game confirmation. But a **clean 400-game,
low-temperature re-match found the "evolved" champion is actually −24 to −41 Elo *worse* than the
plateaued net.** The apparent gain was an artifact of noisy, high-temperature fitness over small
samples with best-of-16 selection bias — a **phantom improvement** that vanished under proper
measurement. Annealing the mutation scale up (σ 0.03→0.12) found nothing better; larger mutations
only degraded the net. So **evolution did not escape the plateau either** — and neither gradient
self-play, a self-referential ladder, nor evolution crosses the ~2000 wall. Since the identical net
reaches ~2150 under *supervised* labels, the wall is **not capacity** but the ceiling of any
*self-generated* signal: a system cannot lift itself past the quality of the signal it produces
about itself. (Methodological note for practitioners: relative-fitness selection with small,
stochastic samples manufactures phantom gains; only a large, low-variance re-measurement can be
trusted — we nearly reported a false escape and caught it only by measuring properly.)

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

A committee combines diverse models at *inference* (voting). The weight-space alternative is to
**average their coefficients into a single network** ("marriage of coefficients"). We tested it on
several conv-96×8 members from different seeds / data / objectives, scored by mean **centipawn loss
(CPL)** against Stockfish depth-12 (lower is better; the members sit at ~60, i.e. ~2000-level, and
~260 is near-random).

| merge | starting points | CPL ↓ | verdict |
|---|---|---:|---|
| naive average | different-start (diverse) | 261 | collapse |
| Git Re-Basin aligned | different-start (diverse) | 268 | still collapse |
| naive average (**model soup**) | **same init** | 58.8 | works (~parent) |
| aligned (net + permuted twin) | identical | 72 | perfect recovery *(verification)* |

**Naive averaging of different-random-start nets collapses** (261 vs ~60): independent networks sit
in different loss basins related by neuron permutations, so averaging misaligned neurons cancels
signal. The known fix is **permutation alignment (Git Re-Basin):** match net B's neurons to net A's
before averaging. We implemented it for the residual conv tower (one shared residual-stream
permutation, a per-block hidden permutation, plus the reduce and value heads) and **verified it is
correct** — it recovers a network *exactly* from a known random permutation (72 CPL = the original).
Yet on the *real* different-start members it **still collapses** (268). The reason is fundamental:
Git Re-Basin assumes independent networks learn the *same features in a different order*; ours
learned **genuinely different features** (different seeds *and* data *and* objectives), and no
permutation aligns different features. By contrast a **model soup** — two children fine-tuned from
the *same* checkpoint — averages fine (58.8), because a shared start keeps them in one basin.

**Conclusion.** Weight-averaging only works within a shared basin. The very diversity that makes an
ensemble valuable (uncorrelated errors, informative agreement, §5.2) is exactly what makes the
members' **weights un-averageable** — diverse models can be combined at *inference*, not in weight
space. This also sharpens the "better aggregator" question: it must live at inference time
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

**Two axes at once — size and speed.**
- **Parameters:** our net is **~10–25× smaller than AlphaZero and 30–100× smaller than large Leela
  transformers**, yet reaches ~2800 with search. On *Elo-per-million-parameters* it is the extreme
  efficiency point — an order of magnitude above the big-net engines (which spend their parameters
  on the last few hundred Elo of a much better value function).
- **Speed:** the two paradigms differ. AlphaZero/Leela use a **similar sim count** (~hundreds–
  thousands) but each sim is a **big-net** forward pass, so their per-move cost is dominated by a
  10–100× larger network; our sim is a 14 MB pass (~1.5 ms), and the **cascade recovers a further
  ~1.6–4.8×**. Stockfish is the opposite regime — a small, heavily-quantized net evaluated at
  **millions of nodes/second** under alpha-beta. The common structure: **strength = evaluator ×
  search; every strong engine is search-heavy, and parameter count is not what separates them.**

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

Across all three stages one result recurs: **the ceiling is the evaluator (the network), not the
loop around it.**

- **Architecture > parameters.** The right prior (spatial locality + weight sharing) beats raw
  width; a 14 MB conv reaches strength a 3× larger MLP cannot.
- **Search > scale, at constant memory — but search cannot exceed the net.** MCTS bought +650 Elo
  (2150→2800) with zero extra parameters and out-scaled fixed depth. Yet flat MCTS and the cascade
  hit the *same* 2691 wall on the same ladder, because they query the *same* value net. Search
  reduces the net's *variance* (random error, via averaging many calls) but not its *bias*
  (systematic blind spots) — and deep uniform search even *amplifies* bias. The cascade's win was
  therefore **efficiency (up to 4.8× cheaper), not strength.**
- **No self-generated signal crosses the wall — we tried three.** Gradient self-play plateaus below
  its teacher; a self-referential Elo ladder never promotes; and derivative-free **evolution**, given
  the fairest shot (unbiased game-outcome fitness, annealed exploration), also fails — its apparent
  +47-Elo escape was a **noise artifact** that reversed to −30 under a clean 400-game re-match. Since
  the *same* net reaches ~2150 under supervised labels, the wall is **not capacity** but the ceiling
  of any signal a system generates about *itself*: external information lifts it, self-generated
  information cannot.
- **A better *evaluator* is the only lever — but it is harder than it looks.** The committee gives a
  robust, teacher-free **confidence signal** (agreement predicts correctness), yet **plurality voting
  does not reliably de-bias**: across a 3/5/7/9-agent sweep the consensus never clearly beat the best
  single member, and correlated majority blocs dominate it (balance beats count). The diversity
  *contains* the information — the per-position oracle is 2–3× better than the vote — but extracting
  it needs a **better aggregator** (soft-averaging / confidence-routing) and *balanced* diversity, not
  more agents. Improving the evaluation is the right goal; naive ensembling is not yet the way.
- **Control-theory unification.** Open loop = feedforward; closed loop = MPC/receding-horizon;
  self-play = iterative learning control — the same three knobs recur in any sequential-decision
  domain, and in all three the learned *evaluator* is what caps performance.

---

## 8. Limitations and Future Work
- Absolute Elo carries systematic uncertainty near the ladder top; relative same-ladder gains are
  the robust claims.
- Stage 3 is compute-limited (two machines); results characterize the *small-scale* regime.
- Stage-3 fitness/aggregation is noise-limited at our scale: relative-fitness selection with small,
  stochastic samples produced a **phantom** evolution "escape" that reversed under a 400-game
  re-match, and plurality de-biasing gaps sit inside the ~±3 CPL measurement noise. Larger,
  lower-variance evaluation is needed to resolve small effects.
- **Next levers, in priority order:** (1) a **better value function** — the true ceiling — via more
  and better search-labeled data or large-scale self-play; (2) a **better ensemble aggregator** —
  soft probability-averaging and confidence-weighted *routing* (the per-position oracle is 2–3×
  above plurality), with *balanced* cross-family diversity rather than more agents; (3)
  **batched/parallel MCTS** (virtual loss) and transposition tables for throughput; (4) endgame
  tablebases and tuned `c_puct`. All roads converge on the same place: **improve the evaluation, and
  both the closed-loop and self-play ceilings rise together.**

---

## 9. Conclusion
A 14 MB, 3.45M-parameter network reaches **~2800 Elo by *thinking* (MCTS search), not by *growing*
(parameters)**. Adaptive MCTS beats and out-scales fixed-depth search; a wide→narrow MCTS cascade
matches it at up to 4.8× less compute. On the self-learning side the results are honestly negative:
self-play, a self-referential ladder, and derivative-free evolution all **fail to cross the ~2000
plateau** (evolution's apparent escape was a noise artifact), and plurality-voting committees do not
reliably de-bias — though model **agreement is a robust teacher-free confidence signal**. The
transferable result, proven from every direction we pushed, is that **the learned evaluator is the
bottleneck** — search redistributes its knowledge and self-generated signal cannot exceed its own
quality; only a better evaluator (better labels, more scale, or a better *aggregator* than plurality)
raises the ceiling. A quantified recipe, and an honest map of its limits, for compact, search-driven
sequential decision-making in general.

---

## Appendix — Implementation
- `chessnet/model.py` — conv/MLP/dual-path + value head. `chessnet/search.py` — alpha-beta,
  MCTS/PUCT, quiescence, the wide→narrow cascade (`MultiStageMCTSPlayer`).
  `chessnet/committee.py` — ensemble inference + agreement signal.
  `chessnet/train.py` — soft/hard + value training. `scripts/selfplay.py` — self-play iteration.
- **Best model:** `runs/conv_value_llm1` (conv-96×8 + value, 3.45M params).
- **Key hyperparameters:** conv width 96, depth 8; lr 5e-4 (train) / 1e-4 (self-play); gradient
  clip 1.0; MCTS c_puct 1.5; Dirichlet α 0.3; replay buffer 120K–300K.
