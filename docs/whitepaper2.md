# Efficient Thinking II: Where Search Pays and Where It Can't
## The evaluator × search decomposition tested in a solved game, language reasoning, and sequential control

**Louay Alsakka** · 2026 · *working paper*

## Abstract

Efficient Thinking I found, in chess, that capability factors as **strength = evaluator × search**: a fixed evaluator sets a base level, inference-time search multiplies it and then saturates against the evaluator's ceiling, and self-improvement stalls because nothing inside a closed system raises that ceiling. This paper asks whether that structure is a chess artifact, by testing it in three deliberately different settings: **Connect-4**, a solved game where a perfect oracle makes every quantity exactly measurable; **LLM mathematical reasoning**, a non-game domain with a natural verifier; and a **gridworld MDP**, where value iteration supplies an exact optimal value function.

Two results anchor the paper. First, a minimal model of the whole thesis: handing an MPC controller the exact optimal value function corrupted by noise σ, search horizon h compensates a mildly noisy evaluator dramatically (σ = 0.25, eight-grid mean: greedy sits at 48% of optimal, three-step lookahead recovers 96%) but cannot rescue a badly degraded one (σ = 1: even h = 3 reaches only ~46%) — and adds nothing when the evaluator is perfect. Averaged over grids, the search axis is monotone in h at every imperfect σ. Search buys back evaluator *variance*, not evaluator *bias*, in five lines of NumPy. Second, an apparent contradiction that turns out to be the thesis: in LLM reasoning, base-model **size dominates search** on the compute-efficiency frontier — a 3B model with greedy decoding beats a 0.5B or 1.5B model with 16-sample self-consistency at higher compute — the opposite of chess, where search on a small frozen net was the efficient lever. The reconciliation is the decomposition itself: search extracts only what the evaluator already contains, so it complements a competent base and cannot substitute for one. "Buy search, not size" holds only above a competence threshold, and these small models sit below it — locating and explaining the boundary that at-scale test-time-compute studies imply but do not characterize (Snell et al. 2024; Brown et al. 2024).

The remaining results locate the binding constraint at the evaluator in every arm. In reasoning, self-consistency saturates while a ground-truth verifier over the same samples keeps climbing (the well-documented consensus-vs-pass@N gap, here +14.2 points), a synthetically graded verifier traces a smooth capability curve with no threshold (~+2.6 points per 0.1 of verifier accuracy), and a real imperfect judge (Kimi-K2.5) captures most of that headroom on the same samples — while search applied to the *judge itself* barely helps, because repeated judgment reproduces systematic error. In Connect-4, the evaluator/search balance shifts with training data in exactly the direction the decomposition predicts. And across five self-improvement experiments in two games, self-generated targets never break the plateau; changing only the value target to an external oracle — the loop otherwise identical — breaks it immediately. Training creates information, search extracts it, and neither creates what an external oracle must supply.

Two further experiments place the third lever — training data — directly. A Connect-4 data × capacity × search grid separates the levers cell by cell: data binds, capacity is slack (and at low data, harmful — the larger net is noisier, not stronger, with seed-level collapses), and search rescues most exactly where the evaluator is weakest — the same ordering chess found. And a two-floor LoRA sweep puts the data lever in language: a bf16 base model climbs from a 1.3% floor to 26.0% (textbook data scaling) while the instruct model dips below its 35.3% zero-shot floor and recovers to 26.7% (warm-start erosion) — both converging on the same ~26% level at the full GSM8K train set. The data sets the destination; the pretrained starting point sets only the direction of approach.

To compare search-lift magnitudes across domains we use one calibrated logistic scale (GELO, the measurement framework of §2); Connect-4's search lift (+236) and chess's (+286) land within ~20% of each other in identical odds units. All experiments run on two Apple-Silicon machines; claims are scoped to these domains and budgets as recurring patterns, not laws.

## Key Takeaways

1. **A five-line model reproduces the whole thesis.** In a gridworld with the exact V* plus noise σ (eight-grid mean ± sd), lookahead rescues a mildly noisy evaluator (48% → 96% of optimal at σ = 0.25) and cannot rescue a badly degraded one (≤ ~46% at σ = 1) — and averaging over grids shows the effect is monotone in h at every imperfect σ. Search compensates evaluator variance, not evaluator bias.

2. **In reasoning, size dominates search — and this confirms rather than contradicts Paper I.** Every small-model-plus-search point on the GSM8K frontier is dominated by a bigger base with less search. Search extracts what the model already contains; below a competence threshold there is little to extract. The efficient lever depends on which side is starved, not on the domain — and the boundary between the regimes is base competence, reproduced with an exact oracle in the gridworld.

3. **The evaluator is the binding constraint in every arm.** Consensus saturates where a verifier keeps climbing (+14.2 on the same samples); a graded verifier buys capability smoothly at every increment of accuracy; a real LLM judge captures most of the headroom; and the judge itself cannot be search-fixed — you can partly buy back a weak policy with compute, but not a weak evaluator.

4. **Self-improvement cannot beat its own signal — and one control proves the mechanism.** Five self-play experiments across two games plateau; swapping only the value target from self-play outcome to an external oracle breaks the plateau (Connect-4 +400 → +719, loop otherwise identical, seed-independent). The plateau is a property of the signal source, not the loop.

5. **The data lever is real, unsaturated — and the starting point sets only the direction of approach.** A two-floor LoRA sweep converges: the bf16 base climbs 1.3% → 26.0% while the instruct model dips below its 35.3% floor and recovers to 26.7% — both landing at the same ~26% data-determined level at the full train set. The Connect-4 grid shows the same ordering cell by cell: data binds, capacity is slack (and seed-unstable), search compensates.

6. **The decomposition transfers, measurably.** On one calibrated scale, search lift is the same order in two games (+236 Connect-4, +286 chess), the same saturating curve shape appears in all four settings, and the lever balance shifts with spend exactly as the decomposition predicts. Figure 1 is the whole paper.

## 1. Introduction
Modern game-playing and reasoning systems both gain most of their capability from two levers that are
easy to conflate: a *learned evaluator* (a network's single-pass judgement) and *search* (spending
inference-time compute to look further before committing). Efficient Thinking I separated them in chess
and found that a tiny 3.45M-parameter network reaches ~2150 Elo open-loop and ~2800 with MCTS, that search
scales roughly log-linearly with simulations, and that self-play stalls because the evaluator — not
parameters or search budget — is the ceiling.

If that structure is real rather than a chess artifact, it should reappear in very different settings.
This paper puts it to three tests: **Connect-4**, a *solved* game where a perfect oracle and a graded
opponent ladder let us measure everything exactly; **LLM mathematical reasoning**, a non-game domain with
a natural verifier (a checkable final answer); and a **gridworld control MDP**, a sequential-decision
setting where value iteration supplies an exact oracle. Our contributions:

1. **A minimal exact-oracle model of evaluator × search** (§5): the gridworld σ × h sweep, eight-grid
   mean ± sd — the cleanest demonstration in either paper that search compensates evaluator noise
   (monotonically in horizon) up to a point and is worthless past it, fully reproducible in pure NumPy.
2. **The size-vs-search frontier result and its reconciliation** (§4): search and scale are not
   interchangeable currencies; which one binds is set by base competence, and the same principle that
   made search the efficient lever in chess makes size the efficient lever for sub-threshold LLMs.
3. **The training-data lever, placed in both arms** (§3–§4): a Connect-4 data × capacity × search grid
   separating the three levers cell by cell (data binds; capacity is slack, seed-unstable, and at low data
   harmful; search rescues a starved evaluator most), and a two-floor language LoRA sweep in which a bf16
   base climbs from 1.3% and the instruct model erodes from 35.3% — converging on the same ~26% level:
   data sets the destination, the starting point only the direction of approach.
4. **The evaluator bottleneck as a gradient and an asymmetry — not just a gap** (§4): *that* a verifier
   beats consensus is established [Cobbe et al. 2021; Lightman et al. 2023]; our contribution is that
   capability is *smooth* in verifier accuracy q (75→88%, no threshold — a curve, not a point), that a
   real imperfect judge captures most of the oracle headroom on identical samples, and that search
   improves a *policy* ~4× more than it improves the *evaluator* (a mechanism, not an endpoint).
5. **An oracle-target positive control for self-improvement** (§6): five plateaued self-play loops and
   one minimal intervention (external value target, nothing else changed) that breaks the plateau —
   isolating the signal source as the causal variable.
6. **GELO, a measurement framework** (§2): a calibrate-first, goodness-of-fit-gated logistic scale that
   makes search-lift magnitudes comparable across domains in odds units — scoped explicitly: curve
   shapes and lifts are comparable; absolute cross-domain levels are not claimed to be.

**Two anchor results.** Everything below supports one decomposition, but two results carry the paper
and are flagged as such where they appear: **Anchor 1 (§5)** — the gridworld noise × lookahead sweep, a
minimal exact-oracle model in which search monotonically rescues a mildly noisy evaluator and cannot
rescue a bad one; and **Anchor 2 (§4)** — the size-vs-search frontier, where base-model size dominates
search, the opposite of chess, reconciled by the decomposition itself. Every other experiment is
supporting evidence for the chain in Figure 1.

**Figure 1 — one causal chain, four instantiations.** The whole paper is a single chain: *external
information* is the only lever that raises *evaluator quality*; *inference-time search* extracts — never
creates — what the evaluator already contains; the two compose into *capability* (strength = evaluator ×
search). Each arrow carries a load-bearing result of this work. Figure 1 is the whole paper.

<svg viewBox="0 0 660 480" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <defs><marker id="f1a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#555"/></marker></defs>
  <rect x="0" y="0" width="660" height="480" fill="#ffffff"/>
  <text x="330" y="24" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1a2a3a">Figure 1 — one causal chain, four instantiations</text>
  <rect x="215" y="42" width="230" height="52" rx="8" fill="#e9f7ee" stroke="#31a354" stroke-width="2"/>
  <text x="330" y="64" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a5e2f">EXTERNAL INFORMATION</text>
  <text x="330" y="82" text-anchor="middle" font-size="9" fill="#666">oracle · verifier · teacher</text>
  <line x1="330" y1="94" x2="330" y2="140" stroke="#555" stroke-width="1.6" marker-end="url(#f1a)"/>
  <text x="345" y="112" font-size="9" fill="#b5322e">the only lever that raises the ceiling;</text>
  <text x="345" y="125" font-size="9" fill="#b5322e">self-play alone cannot (§6)</text>
  <rect x="215" y="144" width="230" height="52" rx="8" fill="#e3eefa" stroke="#2c7fb8" stroke-width="2.8"/>
  <text x="330" y="166" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a2a3a">EVALUATOR QUALITY</text>
  <text x="330" y="184" text-anchor="middle" font-size="9" fill="#555">single-pass judgement — sets the ceiling</text>
  <line x1="330" y1="196" x2="330" y2="242" stroke="#555" stroke-width="1.6" marker-end="url(#f1a)"/>
  <text x="345" y="214" font-size="9" fill="#7a2e06">search EXTRACTS what is here;</text>
  <text x="345" y="227" font-size="9" fill="#7a2e06">it cannot create what is absent (§4–§5)</text>
  <rect x="215" y="246" width="230" height="52" rx="8" fill="#fdf0e6" stroke="#e6550d" stroke-width="1.5"/>
  <text x="330" y="268" text-anchor="middle" font-size="12" font-weight="bold" fill="#7a2e06">INFERENCE-TIME SEARCH</text>
  <text x="330" y="286" text-anchor="middle" font-size="9" fill="#666">look further before acting</text>
  <line x1="330" y1="298" x2="330" y2="344" stroke="#555" stroke-width="1.6" marker-end="url(#f1a)"/>
  <text x="345" y="316" font-size="9" fill="#666">multiplies the evaluator,</text>
  <text x="345" y="329" font-size="9" fill="#666">then saturates at its ceiling</text>
  <rect x="215" y="348" width="230" height="46" rx="8" fill="#f2eef8" stroke="#756bb1" stroke-width="2"/>
  <text x="330" y="376" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#463b7a">CAPABILITY</text>
  <text x="330" y="424" text-anchor="middle" font-size="9.5" font-style="italic" fill="#666">training creates information · search extracts it · neither creates</text>
  <text x="330" y="438" text-anchor="middle" font-size="9.5" font-style="italic" fill="#666">what an external oracle must supply</text>
  <text x="330" y="462" text-anchor="middle" font-size="9" fill="#999">chess +286 lift · Connect-4 +236 lift · reasoning +14.2 verifier gap · gridworld 48% → 96% at h = 3</text>
</svg>

The unification is that the *same skeleton* instantiates in every domain we test — the bottom row is the
load-bearing number each column earns — evidence, not verdicts:

| chain link | Chess (ET-I) | Connect-4 (§3) | LLM reasoning (§4) | Gridworld (§5) |
|---|---|---|---|---|
| external information | Stockfish labels | perfect solver | verifier / RLVR signal | value iteration (V\*) |
| → **evaluator quality** | 3.45M value net | conv value net | P(answer correct) | V̂ = V\* + noise |
| → **inference-time search** | MCTS simulations | PUCT MCTS | best-of-N / self-consistency | h-step lookahead |
| → **capability** | ~2150 → 2800 Elo | GELO vs ladder | GSM8K / MATH accuracy | % of optimal |
| **load-bearing number** | +286 search lift | +236 search lift | +14.2 verifier gap | 48% → 96% at h = 3 |

Read top-to-bottom it is a causal chain; read left-to-right it is the claim that one mechanism spans games,
language, and control. Every quantitative result below measures one arrow in one column.

## 2. GELO — the measurement framework

Elo, Bradley–Terry, and one-parameter IRT (Rasch) are the same latent-ability logistic model,
P(win / solve) = 1 / (1 + 10^(−(θ − d)/400)); this is textbook, not a contribution. What we add is a
*protocol* — calibrate a reference ladder first, fit ratings from a full cross-table rather than
per-opponent win rates, gate on logistic goodness-of-fit before trusting any rating, pin interpretable
anchors — and a *scoped use*: because we keep chess's constants (**400 GELO = 10× the odds; 120 GELO =
one doubling**), a search lift measured in Connect-4 and one measured in chess are expressed in
identical odds units and can be compared directly. That comparison is the only cross-domain claim GELO
carries in this paper. A "2500" in reasoning and a "2500" in chess share a ruler *by construction*, not
a difficulty; we never compare absolute levels across domains, only curve shapes and lift magnitudes.
An agent can be placed two equivalent ways: against a graded **reference ladder** (opponents or
difficulty tiers), or **pairwise agent-vs-agent** with one agent pinned as the reference. Full spec in
`docs/gelo.md`.

The first calibration, on Connect-4, passes the gate cleanly: mean |predicted − observed| pairwise
win-rate = **0.058** — the logistic model genuinely fits the game, so ratings are earned rather than
assumed (scale anchored random := 0). And reasoning lands on the *same* axis via either route:

- **Pairwise ("beat another LLM").** Five models answer the same MATH problems; a master judge (Kimi-K2.5)
  scores every head-to-head → Bradley–Terry GELO, **anchored Kimi-K2.5 := 2800** (the "grandmaster" of the
  set), the small models placed below with headroom: **Qwen3.5-4B +2743 · Qwen2.5-1.5B +2562 · Qwen2.5-3B
  +2509 · Qwen2.5-0.5B +2297.** The 0.5B sits ~500 GELO below the frontier; mid-models bunch within noise
  (50 questions). At 50 problems these placements are **illustrative of the protocol, not results** —
  the mid-model ordering (1.5B above 3B) inverts the models' GSM8K accuracies (§4), a visible sign the
  arena is under-powered at this sample size; a full-scale rerun is queued. The judge agrees with the ground-truth verifier on **72%** of decisive pairs — decent,
  but *itself evaluator-limited*: a referee can only rank as well as it can reason (which is also why, on
  checkable tasks, the verifier still beats an LLM judge). The anchor value is free; we choose 2800 so the
  numbers read like chess (elite ≈ 2800, room down toward a novice floor).
- **Difficulty-anchored (IRT).** The same machinery against MATH difficulty tiers gives a monotonic ladder
  (L1 +1273 → L5 +1712, ~+110 GELO/level) and places a model by which tier it half-solves.

## 3. Arm A — a simpler game (Connect-4)
Connect-4 is solved, so the exact solver is a perfect oracle and a depth-limited alpha-beta gives a
calibrated opponent ladder. A small convolutional network (policy + value) is the evaluator; PUCT MCTS is
the closed loop.

**A calibrated opponent ladder.** random 0 → **depth-1 +804** → depth-2…6 +954…+1070. The *first ply of
tactics* is the single biggest step (+804) — larger than all five deeper plies combined — and the
heuristic ladder then saturates: diminishing returns on search depth, quantified on a real axis.

**Evaluator × search decomposition** (12k oracle labels). Open-loop (raw net) **+644 GELO**, closed-loop
(MCTS-200) **+880 GELO** → a **search lift of ≈ +236 GELO (~4× the odds per game)** — the same order as
chess's search lift. The decomposition transfers to a second game.

**The lever balance is data-dependent, not intrinsic.** With only 12k labels the raw net loses even to
1-ply search, which reads as "search-dominated." But tracing the open-loop ceiling against data — **+642
(12k) → +747 (24k) → +798 (50k)** — the raw net reaches depth-1 parity (+804) by 50k. The apparent
search-dominance was mostly a *starved evaluator*: fed data, it catches up to low-depth search. The honest
statement is that the evaluator/search balance moves with how much you have spent on the evaluator — in
either direction.

**Which lever binds — a data × capacity × search grid.** To separate the three levers directly, we sweep
training data (3k–50k) × network capacity (small ≈0.3M / large ≈4M params), placing each net both
open-loop (raw) and closed-loop (MCTS-100). Open-loop GELO:

| data | small (~0.3M) open | large (~4M) open, 3-seed mean ± sd | search lift (small, +MCTS) |
|---:|---:|---:|---:|
| 3k  | +512 | +278 ± 70 | +147 |
| 12k | +673 | +616 ± 92 | +117 |
| 24k | +659 | +665 ± 64 | +210 |
| 50k | **+801** | +651 ± 266 | +222 |

Three readings, one per lever. **Data binds:** open-loop climbs with data, the small net reaching depth-1
parity (~+804) by 50k. **Capacity is slack — and at low data, harmful:** the *larger* net's mean never
beats the small one open-loop (it is *below* at 3k/12k/50k, level at 24k) and is *high-variance*, with
occasional collapses (the 50k-large seeds were +796/+879/**+278**; sd 266) — extra parameters with too
little data add noise, not skill. **Search is a large multiplier that most rescues a starved evaluator:**
MCTS adds +117…+222 GELO on the small net, its biggest rescues landing exactly where the raw evaluator is
weakest. So at this scale the binding constraint is *data* (evaluator quality), capacity is slack, and
search compensates — the same ordering Paper I found in chess (search ≫ data ≫ capacity), here separated
cell by cell (large-net cells averaged over 3 seeds; small-net and MCTS cells single-run).

## 4. Arm B — reasoning (LLM)
The mapping from chess to language:

| chess | LLM reasoning |
|---|---|
| position | partial reasoning trace |
| policy (move priors) | next-step distribution |
| **evaluation: P(win)** | **P(this reasoning is correct)** ∈ [0,1] |
| terminal result (win/loss) | **verifier on the final answer** (exact in math/code) |
| MCTS search | inference-time compute over reasoning paths |
| training the net | RL / fine-tune — RLVR is AlphaZero's loop in language |

Two structural notes: the evaluator's output is a **P(correct)** — binary from a verifier, scalar from a
reward model, vote-fraction from self-consistency — and a *well-calibrated* one is the open problem; and
search has two axes — parallel (best-of-N, self-consistency) and serial (long chain-of-thought), the
latter *internalized* into the policy by RLVR rather than kept external as in AlphaZero.

**How search is implemented here — granularity and who evaluates.** This is worth stating concretely,
because an LLM is not a game engine. An LLM has **no built-in correctness-evaluator**: its only native
scorer is the next-token softmax, which judges token *plausibility* (what word comes next), never answer
*correctness*. Every search result below therefore operates at **whole-answer granularity** with an
**external** evaluator, by a uniform procedure: sample N *complete* answers from the policy (temperature
> 0, so the N chains-of-thought actually differ), let each run to completion (hundreds of tokens),
extract its final answer, then apply a **selector** over the *finished* set — majority vote (a
verifier-free evaluator), a checkable verifier (perfect), or the Kimi-K2.5 judge (a real, imperfect
model). We do **not** score or branch per token, nor per reasoning step — that would be tree-of-thought
with a process-reward model, which we leave to future work. The two search knobs are thus **N** (how many
answers) and the **selector** (how you pick), both entirely external to the frozen weights. This is the
language analog of AlphaZero's split: the policy *proposes* a full line of play, and a *separate* value
function *judges* it — and here, as there, the judge is the ceiling.

**Search scales accuracy, then saturates.** Qwen-4B on GSM8K (120 problems, 1024-token budget): greedy
pass@1 = **66.7%**; self-consistency@N = 69.2 (N=1) → 73.3 (4) → **77.5 (16) → 77.5 (32)** — search buys
**+8–11 points** (the analog of Elo-vs-sims) and then saturates by N=16. Majority vote is a *verifier-free*
evaluator, and it stops helping. (At 120 problems a single proportion carries roughly ±8 points at
95%, so read "saturates" throughout this arm as "no further gain within noise"; the 300–500-problem,
32-sample cache rerun will tighten every number in this section.)

**The saturation is an *evaluator* ceiling, not a policy or search ceiling.** From the same samples,
self-consistency saturates at 77.5% while **oracle-best-of-N (a perfect verifier) climbs to 91.7% and is
still rising at N=32** — an evaluator gap of **+14.2 points**. The right answer is *in the sample set* 91.7%
of the time; consensus just can't select it. In language, exactly as in chess, the verifier is the binding
constraint — and with a perfect evaluator, search keeps paying past where consensus stalls.

**A graded verifier draws a continuous capability curve.** Varying a verifier's per-item accuracy q, the
resulting accuracy climbs smoothly from **75.0% at q=0.5 (verifier-free consensus) → 88.3% at q=1.0
(perfect verifier)** — 77.7 / 80.6 / 83.0 / 85.9 at q = 0.6/0.7/0.8/0.9, about +2.6 points per 0.1 of
verifier accuracy. There is no threshold: *every* increment of a better evaluator buys capability. To go
further, improve the evaluator.

**A *real* evaluator — not just a perfect one — extracts more than consensus.** Selecting among N
Qwen-1.5B samples with the Kimi-K2.5 judge as an (imperfect) scorer beats majority vote at every N and
captures most of the consensus→oracle headroom: at N=8, self-consistency **65.0%** → **Kimi-best-of-N
75.0%** → oracle **83.3%** (and at N=4, 61.7% → 73.3% → 81.7%). A stronger *selector* buys ~+10 points on
the *same* samples — so "more search" pays far more when a better evaluator *spends* it than when majority
vote does. This is the deployable form of the +14.2 result: to turn search into capability, improve the
selector, not just raise N.

**Search improves a policy but barely an evaluator.** Running the master judge itself with
self-consistency (majority of N judgments) raised agreement with the verifier only **72.5% → 75.0%
(N:1→5)** — a ~+2.5-point nudge, versus the +8–11 points self-consistency buys a *policy*. The reason is
instructive: a policy benefits because *some* of many attempts land on the answer and search selects them;
but if a judge cannot reason a problem out, repeating the judgment reproduces the same *systematic* error.
So you can partly buy back a weak *policy* with compute, but **not a weak *evaluator*** — which is exactly
why the evaluator's quality, not its search budget, binds.

**The serial axis (thinking length) saturates fast for a base model.** Measuring the *other* search axis —
greedy accuracy vs generation budget — Qwen-1.5B climbs 7.5% (128 tok) → 41.2 (256) → **48.8% (512), then
flat** through 2048; Qwen-3B similarly 5.0 → 41.2 → **67.5% (512+), flat.** The gain is almost entirely a
*don't-truncate* effect: a non-thinking base model finishes its chain of thought in ~512 tokens and stops,
so extra budget buys nothing. Genuine o1/R1-style serial scaling requires a model *trained* to keep
deliberating. So of the two search axes, **parallel (more samples, a better selector) is the live lever
for an untrained base model, while the serial axis is capped at "enough room to finish"** — once more,
search extracts only what the model already contains.

**Anchor 2 — the efficient-thinking frontier: size vs. search.** The core efficiency question, concretely: for a
fixed compute budget, spend it on a bigger model or on more search over a smaller one? Sweeping a clean
size ladder (Qwen2.5 0.5B/1.5B/3B) × self-consistency N on GSM8K:

| model | params | greedy | sc@4 | sc@16 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 0.5B | 30.0% | 25.0% | 41.2% |
| Qwen2.5-1.5B | 1.5B | 48.8% | 50.0% | 58.8% |
| Qwen2.5-3B  | 3.0B | 67.5% | 76.2% | 83.8% |

On accuracy vs. compute (≈ params × N), **every small-model-plus-search point is dominated by a bigger
model with less search**: 3B greedy (67.5%, compute ≈ 3) beats 0.5B@N=16 (41.2%, ≈ 8) *and* 1.5B@N=16
(58.8%, ≈ 24). In reasoning, **base-model size dominates search** on the efficiency frontier — the
*opposite* of chess, where search on a fixed 3.45M net was the efficient lever. The reconciliation is the
thesis itself: *search extracts what the model already contains.* The chess net was already strong on its
task (~2150 open-loop), so search extracted a lot; these small LLMs are too weak on GSM8K for search to
rescue — you cannot self-consistency your way up from a base that rarely finds the answer at all. **Search
complements a competent base; it cannot substitute for one.** "Buy search, not size" holds only above a
base-competence threshold. (Compute ≈ params × N is coarse, but the domination is large enough to survive
it.)

<svg viewBox="0 0 620 330" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="330" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Anchor 2 — size dominates search on the GSM8K compute frontier</text>
  <line x1="70" y1="270" x2="590" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="70" y1="270" x2="70" y2="40" stroke="#333" stroke-width="1.5"/>
  <text x="34" y="155" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 34 155)">accuracy (%)</text>
  <text x="330" y="305" text-anchor="middle" font-size="11" fill="#333">compute proxy ≈ params × N (log)  →</text>
  <line x1="266" y1="121" x2="590" y2="121" stroke="#31a354" stroke-width="1.2" stroke-dasharray="5 3"/>
  <text x="585" y="114" text-anchor="end" font-size="9" fill="#31a354">3B greedy (67.5%) dominates every point below this line at ≥ its compute</text>
  <polyline points="70,239 222,254 374,203" fill="none" stroke="#e6550d" stroke-width="2"/>
  <circle cx="70" cy="239" r="4" fill="#e6550d"/><circle cx="222" cy="254" r="4" fill="#e6550d"/><circle cx="374" cy="203" r="4" fill="#e6550d"/>
  <text x="374" y="196" text-anchor="middle" font-size="9" fill="#e6550d">0.5B: 41.2 @ N=16</text>
  <polyline points="190,180 342,176 494,148" fill="none" stroke="#756bb1" stroke-width="2"/>
  <circle cx="190" cy="180" r="4" fill="#756bb1"/><circle cx="342" cy="176" r="4" fill="#756bb1"/><circle cx="494" cy="148" r="4" fill="#756bb1"/>
  <text x="494" y="141" text-anchor="middle" font-size="9" fill="#756bb1">1.5B: 58.8 @ N=16</text>
  <polyline points="266,121 418,93 570,70" fill="none" stroke="#2c7fb8" stroke-width="2.5"/>
  <circle cx="266" cy="121" r="4.5" fill="#2c7fb8"/><circle cx="418" cy="93" r="4" fill="#2c7fb8"/><circle cx="570" cy="70" r="4" fill="#2c7fb8"/>
  <text x="258" y="138" text-anchor="middle" font-size="9" font-weight="bold" fill="#2c7fb8">3B greedy 67.5</text>
  <text x="570" y="63" text-anchor="middle" font-size="9" fill="#2c7fb8">3B: 83.8 @ N=16</text>
  <text x="140" y="255" font-size="9" fill="#999" font-style="italic">smaller base + more search never crosses a bigger base</text>
</svg>


**Relation to compute-optimal test-time scaling.** The result to *not* claim here is "size beats search" —
Snell et al. [2024] show the size-vs-search answer is regime-dependent (search wins on easier problems for
capable models), and Brown et al. [2024] show repeated sampling scales coverage over four orders of
magnitude; taken at face value ours is a below-threshold special case of a more nuanced known picture. What
their work leaves open is *where* the boundary is and *why* — their compute-optimal policy is
difficulty-adaptive, implying a competence threshold without characterizing it. That is the gap we fill:
(i) we **identify the boundary as base competence** — below it, search extracts nothing because the base
rarely contains the answer at all; (ii) the same evaluator × search decomposition **predicts the flip from
first principles**; and (iii) — the asset no LLM study has — we **reproduce the identical threshold in the
gridworld (§5) with a perfect oracle**, where at σ=0 search is worthless and at σ=0.25 it recovers 48%→96% (eight-grid mean),
so the crossover appears with *no* language-model confound. The framing is therefore "Snell et al. show the
frontier is regime-dependent; we identify the boundary as base competence, derive it from the
decomposition, and reproduce it under an exact oracle." Brown et al. *strengthen* rather than pre-empt us:
their finding that coverage scales but precision does not is exactly our **+14.2 verifier gap** (§4) seen at
scale — the answer is in the sample set, selection is the bottleneck — so their large-n result and our
small-n one agree. The cross-domain GELO scale (§2) is the domain-agnostic instrument for *locating* this
crossover, from search-rich chess (~2150 open-loop, search extracts +286) to search-poor GSM8K (0.5–3B,
search cannot rescue).

**The training-data lever — a fine-tuning sweep, from two starting points.** Size and search are two of the
three levers; the third is *training data*, which Connect-4 carries (§3) but which we can also place
directly in language. We LoRA-fine-tune **two** 0.5B models — the **instruct** model and the **non-instruct
base** (bf16) — on an increasing number of GSM8K examples at *fixed epochs* (so more data does
proportionally more training, isolating the data axis), cleaned targets, greedy accuracy on 150 held-out
problems:

| train examples | 0 (zero-shot) | 64 | 256 | 1024 | 4096 | 7000 (full) |
|---|---:|---:|---:|---:|---:|---:|
| **base (bf16)** — from a low floor | 1.3% | 16.0 | 15.3 | 17.3 | 24.0 | **26.0%** |
| **instruct** — from a competent floor | 35.3% | 19.3 | 20.0 | 22.7 | 24.0 | **26.7%** |

The two curves *converge*, and the convergence is the finding. The **base model climbs from a 1.3% floor**
— the textbook data-scaling curve, more data monotonically buying capability. The **instruct model dips
below its 35.3% zero-shot floor and slowly recovers** — *warm-start erosion*: a few thousand narrow
examples first trade away broad pretrained capability for GSM8K surface form, then rebuild it. Yet **both
land at ~26% at the full 7k train set**: the *data* sets the attractor level, while the *pretrained
starting point* governs only the **direction of approach** — climb up to it from below, or fall to it from
above. This is the language echo of Connect-4's warm-start erosion (a strong evaluator pulled toward the
data-determined level), now shown from *both* sides on one benchmark. The training-data lever is real and
unsaturated (still rising at 7k), and how competent the evaluator already was sets the *sign of the first
step*, not the destination. (The base run required full precision: at 4-bit the same LoRA fine-tune was
unstable — degenerate generation at low data — which is itself a caution about drawing data-scaling
conclusions from quantized small-model fine-tunes.) (150 held-out problems ⇒ roughly ±7–8 points per
cell at 95%: load-bearing are the two monotone orderings, the 1.3-vs-35.3 floor gap, and the joint
convergence band at 7k — 26.0 vs 26.7 is indistinguishable at this n, which is consistent with, though
not yet proof of, a shared data-determined attractor; adjacent-cell differences sit within noise.)

## 5. Arm C — sequential control (a gridworld MDP) — Anchor 1
To test the pattern in a *third modality* — sequential decision-making, neither a board game nor language —
we use a stochastic 8×8 gridworld with known dynamics, so value iteration gives the exact optimal value V*
(a perfect oracle). We hand the controller a *degraded* evaluator, V* plus Gaussian noise of scale σ (in
units of the value spread), and vary the horizon h of an MPC-style lookahead (h=1 = greedy / open-loop;
larger h = closed-loop search). Return is normalized so 0% = a random policy and 100% = optimal.

Averaged over **8 independent grids** (mean ± sd; % of optimal):

| evaluator noise σ | open-loop (h=1) | h=2 | h=3 |
|---:|---:|---:|---:|
| 0.0 (perfect) | 100 ± 0 | 100 ± 0 | 100 ± 0 |
| 0.25 | 48 ± 28 | 81 ± 19 | **96 ± 10** |
| 0.5 | 27 ± 7 | 53 ± 28 | 74 ± 25 |
| 1.0 | 24 ± 7 | 34 ± 11 | 46 ± 22 |

The same two findings recur, and multi-grid averaging **sharpens the search axis into a monotone**: at every
imperfect σ, deeper lookahead helps — **h=1 < h=2 < h=3** — with no inversion (a single-grid run had shown a
spurious h=3 < h=2 dip that averaging reveals as noise, sd ≈ 20–28 pp). **With a perfect evaluator, open-loop
is already optimal and search adds nothing** — search earns its cost only when the evaluator is imperfect.
**With a mildly noisy evaluator (σ = 0.25), search compensates dramatically** — greedy sits at 48% of
optimal, three-step lookahead recovers it to 96%: evaluator × search, in control. But **as the evaluator
degrades, each step of search recovers less** (σ = 1.0: even h=3 reaches only ~46% and cannot recover
optimal) — past a point the evaluator, not the search horizon, is the binding constraint. Decomposition and
evaluator-bottleneck, both holding in a control/RL setting with an exact oracle.

<svg viewBox="0 0 620 330" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:sans-serif">
  <rect x="0" y="0" width="620" height="330" fill="#ffffff"/>
  <text x="310" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a2a3a">Anchor 1 — search rescues a noisy evaluator, monotonically in h (8-grid mean ± sd)</text>
  <line x1="70" y1="270" x2="590" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="70" y1="270" x2="70" y2="40" stroke="#333" stroke-width="1.5"/>
  <text x="34" y="155" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 34 155)">% of optimal return</text>
  <text x="330" y="305" text-anchor="middle" font-size="11" fill="#333">evaluator noise σ  →  worse evaluator</text>
  <line x1="90" y1="102.8" x2="90" y2="226" stroke="#e6550d" stroke-width="0" opacity="0"/>
  <polyline points="90,50 240,164.4 390,210.6 540,217.2" fill="none" stroke="#e6550d" stroke-width="2" stroke-dasharray="5 3"/>
  <line x1="240" y1="102.8" x2="240" y2="226" stroke="#e6550d" stroke-width="1.2" opacity="0.45"/>
  <line x1="390" y1="195.2" x2="390" y2="226" stroke="#e6550d" stroke-width="1.2" opacity="0.45"/>
  <line x1="540" y1="201.8" x2="540" y2="232.6" stroke="#e6550d" stroke-width="1.2" opacity="0.45"/>
  <circle cx="90" cy="50" r="4" fill="#e6550d"/><circle cx="240" cy="164.4" r="4" fill="#e6550d"/><circle cx="390" cy="210.6" r="4" fill="#e6550d"/><circle cx="540" cy="217.2" r="4" fill="#e6550d"/>
  <text x="240" y="182" text-anchor="middle" font-size="9" fill="#e6550d">48</text>
  <text x="566" y="222" font-size="9" fill="#e6550d">h=1</text>
  <polyline points="90,50 240,91.8 390,153.4 540,195.2" fill="none" stroke="#756bb1" stroke-width="2" stroke-dasharray="2 3"/>
  <line x1="240" y1="50" x2="240" y2="133.6" stroke="#756bb1" stroke-width="1.2" opacity="0.45"/>
  <line x1="390" y1="91.8" x2="390" y2="215" stroke="#756bb1" stroke-width="1.2" opacity="0.45"/>
  <line x1="540" y1="171" x2="540" y2="219.4" stroke="#756bb1" stroke-width="1.2" opacity="0.45"/>
  <circle cx="90" cy="50" r="4" fill="#756bb1"/><circle cx="240" cy="91.8" r="4" fill="#756bb1"/><circle cx="390" cy="153.4" r="4" fill="#756bb1"/><circle cx="540" cy="195.2" r="4" fill="#756bb1"/>
  <text x="252" y="88" font-size="9" fill="#756bb1">81</text>
  <text x="566" y="192" font-size="9" fill="#756bb1">h=2</text>
  <polyline points="90,50 240,58.8 390,107.2 540,168.8" fill="none" stroke="#2c7fb8" stroke-width="2.5"/>
  <line x1="240" y1="36.8" x2="240" y2="80.8" stroke="#2c7fb8" stroke-width="1.2" opacity="0.5"/>
  <line x1="390" y1="52.2" x2="390" y2="162.2" stroke="#2c7fb8" stroke-width="1.2" opacity="0.5"/>
  <line x1="540" y1="120.4" x2="540" y2="217.2" stroke="#2c7fb8" stroke-width="1.2" opacity="0.5"/>
  <circle cx="90" cy="50" r="4" fill="#2c7fb8"/><circle cx="240" cy="58.8" r="4.5" fill="#2c7fb8"/><circle cx="390" cy="107.2" r="4" fill="#2c7fb8"/><circle cx="540" cy="168.8" r="4" fill="#2c7fb8"/>
  <text x="240" y="48" text-anchor="middle" font-size="9" font-weight="bold" fill="#2c7fb8">96</text>
  <text x="566" y="165" font-size="9" fill="#2c7fb8">h=3</text>
  <text x="90" y="286" text-anchor="middle" font-size="9" fill="#666">0</text>
  <text x="240" y="286" text-anchor="middle" font-size="9" fill="#666">0.25</text>
  <text x="390" y="286" text-anchor="middle" font-size="9" fill="#666">0.5</text>
  <text x="540" y="286" text-anchor="middle" font-size="9" fill="#666">1.0</text>
  <text x="330" y="248" text-anchor="middle" font-size="9" fill="#999" font-style="italic">σ = 1: even h = 3 caps ~46% — the evaluator binds</text>
</svg>


## 6. Self-improvement — can the flywheel raise the evaluator with no external teacher?
Stated value-first: *fix the evaluator first* — distill better-than-current value/policy targets
(Monte-Carlo rollouts / MCTS-backed, anchored on real terminal outcomes) back into the net; strength
follows. We tested this five ways across two games. **It never broke the plateau — and why is the result.**

**Relation to self-improving-LM work.** ReST [Gulcehre et al. 2023] and Self-Rewarding LMs [Yuan et al.
2024] report self-improvement that *does* work — so "self-improvement plateaus" is emphatically **not** our
claim; stated baldly it is counterexampled. The difference is what the reward signal carries: those systems
inject external information *implicitly* — a verifier, human-preference-seeded reward, or filtered gold
answers — so the flywheel is never truly closed. Our contribution is the **controlled isolation**: the
signal *source* is the single manipulated variable with the loop otherwise frozen, which lets us show not
*that* self-improvement works but *which ingredient makes it work*. The oracle-target positive control
below (self-play outcome → external oracle, **+400 → +719**, loop otherwise identical) is exactly that
counterfactual — the one the at-scale positive results, valuable as they are, do not isolate.

- **Connect-4, from scratch:** the evaluator improves (oracle value-MAE 0.48 → 0.355 with more search) and
  strength tracks it early (GELO +119 → ~+400), confirming *fix-the-evaluator-and-strength-follows* in
  miniature — but strength plateaus at **~+400** and **4× more search per move did not raise it**. More
  search budget is not the missing ingredient.
- **The plateau is a (near) seed-independent attractor:** warm-starting from the supervised +644 net does
  not preserve it — the first iteration erodes it to +189, then it recovers only to **~+480–500**, well
  below the +644 seed. Self-play pulls a *better* evaluator down toward its own signal quality (the same
  erosion as the chess self-learned test, 1214 → 833).
- **Chess, three negatives:** from scratch, open-loop Elo bounced 254–384 with no climb over 44 iters
  (~1,400 games) — a data-volume wall, not a method failure; warm-started from a weak-policy seed, it
  stalled at 300–326 (a **bootstrap barrier**: MCTS quality depends on the policy prior, so a near-random
  prior can't generate improving targets); and started from a genuinely self-learned 1214 plateau net, it
  **degraded to 833** — search on that value head doesn't play far enough above 1214 to produce improving
  targets.
- **An external oracle breaks the plateau (positive control).** Change *only* the value target — from the
  self-play outcome to an external depth-6 oracle, loop otherwise identical — and Connect-4 self-training
  **climbs past ~+400 to +719 by iteration 60** (MAE 0.36 → 0.31), toward the oracle's own level. The same
  loop that plateaus at +400 on self-generated signal climbs once the target carries external information.
  (Seed-independent: from the +644 supervised seed with oracle targets it reaches ~+676 — the *target
  source*, not the starting point, is what matters.)

**Why it can't work for free.** A Monte-Carlo rollout gives the true value of a position — but *under the
current level of play* (the model plays both sides). Training on those labels converges the evaluator to a
perfectly-calibrated evaluator *of its own level*, capped there: no move better than the current level ever
appears, so none can be learned. The only way rollouts push past the current level is to play them *above*
it — with search — and then the per-iteration gain equals the **search-over-policy margin**, itself bounded
by evaluator quality. This is AlphaZero's engine, and precisely why it needs deep search × millions of
games: a large margin sustained over many iterations. At our scale the margin was too small. **Self-play
cannot add information the evaluator doesn't already contain; raising the ceiling requires importing it.**

## 7. Discussion — what transfers, what shifts, what binds

**The lever–limit map — every experiment, including the failures, is a diagnosis of one link in Figure 1.**
Each row varies one lever and reports which link ended up binding:

| experiment | domain | lever varied | what bound it | evidence |
|---|---|---|---|---|
| open- vs closed-loop | Connect-4 | search (MCTS) | search *extracts* | +236 GELO search lift |
| ceiling vs data | Connect-4 | evaluator data | **evaluator** | +642 → +798; depth-1 parity at 50k |
| data × capacity × search | Connect-4 | all three | **evaluator (data)**; capacity slack | small net ≥ large open-loop; open-loop ↑ with data |
| fine-tuning data sweep | reasoning | training-data size | **evaluator (data)**, unsaturated | 19.3% → 26.7% monotonic in N |
| consensus vs oracle-best-of-N | reasoning | selector quality | **evaluator** | **+14.2** (verifier ≫ consensus) |
| graded verifier | reasoning | evaluator accuracy q | **evaluator** | smooth 75% → 88% |
| Kimi-best-of-N | reasoning | a *real* selector | **evaluator** | 65% → 75% at N=8 |
| judge self-consistency | reasoning | search *on the evaluator* | **evaluator** (search can't fix it) | 72.5% → 75% only |
| serial thinking length | reasoning | serial search | policy (untrained base) | flat past 512 tokens |
| size × search frontier | reasoning | size vs search | base competence | size dominates search |
| noisy-eval × lookahead | control | search horizon h | **evaluator** | σ=0.25: h3 lifts 48%→96% (monotone in h); σ=1 caps ~46% |
| self-play ×5 | chess + C4 | training signal | evaluator / search margin | plateau never breaks |
| oracle-value target | Connect-4 | *external information* | — (positive control) | +400 → +719 |

Read down the "what bound it" column: **evaluator** recurs; search binds only where the evaluator was
starved into dominance, and the one row that breaks a plateau is the one that injects external information.

1. **The decomposition transfers.** Strength = evaluator × search holds in two games, in language
   reasoning, and in sequential control, on one calibrated scale. Search is a large, portable lever that
   scales with inference compute and then saturates against the evaluator's ceiling (Elo-vs-sims in chess,
   accuracy-vs-N in reasoning, GELO-vs-MCTS in Connect-4, lookahead-vs-noise in the gridworld) — and it is
   worth nothing when the evaluator is already perfect (gridworld σ=0), everything when the evaluator is
   the only lever left.
2. **The lever balance shifts with spend.** Whichever currency is *starved* dominates returns: a thin
   evaluator (12k Connect-4 labels; a verifier-free consensus) makes search look all-important; feeding it
   (50k labels; a real verifier) rebalances. There is no domain-intrinsic split — only how much you have
   spent on each side. In reasoning the same logic flips the *frontier*: below a competence threshold,
   size (base capability) dominates search.
3. **Across the domains and compute budgets studied here, the evaluator is consistently the binding
   constraint**, shown independently: reasoning consensus is
   broken only by a better verifier (+14.2), and by a *graded* verifier (75 → 88); Connect-4 self-training
   is limited by value-head quality, not search budget; chess self-improvement stalls exactly when the
   evaluator is too weak for search to generate improving targets; and even the *judge* is
   evaluator-limited (search barely improves it).
4. **Self-improvement has a rate — the search-over-policy margin.** You cannot fix the evaluator for free:
   self-generated signal converges to the system's own level. Climbing requires search that plays above the
   current policy (bounded by the evaluator) or an external oracle that injects information. The positive
   proof is the flip side of every negative here — a perfect verifier unlocks +14.2 in reasoning; an oracle
   value target breaks the Connect-4 plateau; Stockfish labels carried chess to ~2800. *Training creates
   information; search extracts it; neither creates what an external oracle must supply.*

**When to spend on what — the resource rollup.** The lever–limit map above is indexed by *experiment*
(what each run diagnosed); its transpose is indexed by *resource* (when to spend on each) — the
practitioner's view, one row per box/arrow of Figure 1:

| resource | when it binds | when it doesn't | evidence |
|---|---|---|---|
| **training data** | starved evaluator (early Connect-4; every LLM fine-tune) | no ceiling reached in our range | data sweeps, both arms (§3–§4) |
| **evaluator quality** | whenever search has saturated | never observed slack | +14.2 gap · graded 75→88 · judge asymmetry (§4) |
| **search** | competent-but-unextracted evaluator (chess 2448 base; gridworld σ=0.25) | below base competence (frontier); at perfection (σ=0) | +286 / +236 lifts · 48%→96% · size≫search (§4–§5) |
| **capacity** | never, in our regimes | everywhere tested — and *harmful* at low data | grid + seed collapses (§3); ET-I capacity sweep |
| **external information** | at every self-improvement plateau | — (always the ceiling-raiser) | oracle control +400→+719 · +14.2 verifier · Stockfish labels (§6) |

Read as an allocation rule: spend on the evaluator until search stops saturating, spend on search only
above a competence threshold, and note the two entries that make this a *measurement* rather than a menu —
**capacity never binds in any regime we tested** (and hurts when data is thin), while **external
information is the only lever that raises the ceiling at all**, which is why the three internal levers are
not the complete menu the other four rows might suggest.

*A footnote vignette: asked to play raw chess against Paper I's ladder, a frontier general LLM
(Kimi-K2.5) beats a random mover barely more than half the time, scores 0% against Stockfish-1320, and
often cannot emit a legal move, while Paper I's 3.45M specialist plays at the ~2800 ladder band with
search. We draw no cross-domain GELO number from this — §8's scoping forbids it — only the qualitative
point: task capability is bought by the right evaluator plus search over it, not by scale alone.*

## 8. Limitations and Future Work
- **Modest compute and sample sizes.** Everything runs on two Apple-Silicon machines: 50–120 problems per
  reasoning point, 24–40 games per ladder rung. We report *effect sizes and recurring patterns, not tight
  confidence intervals*; where a result is within noise (the mid-model GELO bunching) we say so. The claims
  are scoped to the domains and compute regimes examined here — a recurring empirical pattern, not a proven
  law.
- **GELO is a common scale, not a universal difficulty.** The cross-domain numbers share one logistic model
  and chess's constants *by construction*; "2500 in reasoning" and "2500 in chess" sit on the same ruler
  but are not claimed to be the same underlying difficulty. Each domain is calibrated *within* itself
  (goodness-of-fit gate); the shared constants make the *shapes* comparable, not the absolute levels.
- **Reasoning search is whole-answer only.** We test the parallel axis (best-of-N, self-consistency) and
  the serial axis (thinking length), but not *process-reward tree search* (per-step evaluation / MCTS over
  reasoning steps) — the richest form, and the natural next experiment.
- **Reasoning fine-tuning is small-scale and precision-sensitive.** The language data lever (§4) is a LoRA
  sweep on two 0.5B models up to the full 7k-example GSM8K train set; both climb with data and converge to
  ~26% (instruct dipping below its 35.3% base — warm-start erosion — while the bf16 base climbs cleanly from
  a 1.3% floor). One caution worth stating: at **4-bit** the base-model fine-tune was unstable (degenerate
  generation, non-monotonic); the clean curve required **full precision** — so data-scaling conclusions
  from quantized small-model fine-tunes should be treated with care. Larger models and larger data are the
  untested routes to confirm the ~26% attractor is genuinely data-set and not a 0.5B capacity ceiling.
- **Four domains is not universality.** Go, continuous/robotic control, and code are the obvious next
  tests; the framework predicts the same evaluator-bound in each, and predicts *where* it should shift
  (starve the evaluator and search dominates; strengthen it and search saturates).

## 9. Conclusion
Across four domains — two games, language reasoning, and sequential control — capability decomposes the
same way: a single-pass **evaluator** sets the level, **search** multiplies it and then saturates, and only
**external information** raises the ceiling. On one calibrated scale (GELO) the pattern is visible as a
family of the same curve (Elo-vs-sims, accuracy-vs-N, GELO-vs-MCTS, return-vs-horizon), and the
lever–limit map (§7) shows every experiment — the +14.2 verifier gap, the graded 75→88 curve, the five
self-play plateaus, the one oracle-value break — diagnosing the *same* link. The compact statement is
Figure 1: **training creates information, search extracts it, and neither creates what an external oracle
must supply.** Within the domains and budgets studied here, the evaluator — not parameters, not search
budget — is what binds.

## Related work
The evaluator-plus-search structure is the AlphaZero family [Silver et al. 2016; 2017; 2018] built on MCTS
[Coulom 2006; Kocsis & Szepesvári 2006; Browne et al. 2012] and PUCT [Rosin 2011], with expert iteration
[Anthony et al. 2017] as its self-play loop and KataGo [Wu 2019] as a strong open reference. The
inference-time-compute view of reasoning — o1/R1 [OpenAI 2024; DeepSeek-AI 2025], self-consistency [Wang et
al. 2023], tree-of-thought [Yao et al. 2023], STaR-style bootstrapping [Zelikman et al. 2022] — is the same
lever at frontier scale; Jones [2021] documents test-time-compute scaling in board games.

Three lines are closest to our reasoning results, and we position *against* them rather than merely
alongside — for each, what they establish and what the controlled/cross-domain version adds.
**Compute-optimal test-time scaling** — Snell et al. [2024] (test-time compute can beat extra parameters,
regime-dependently) and Brown et al. [2024] (coverage scales with repeated sampling over four orders of
magnitude) — establishes the *above-threshold* half of our frontier and implies a competence boundary; we
identify that boundary as base competence, derive it from the evaluator × search decomposition, and
reproduce it under a *perfect oracle* in the gridworld where no LLM confound exists (§4–§5). Brown et al.'s
"coverage scales, precision does not" is our +14.2 verifier gap at scale, so their large-n result agrees
with our small-n one. **Verifier / process-reward supervision** — Cobbe et al. [2021], Lightman et al.
[2023] — establishes that a verifier beats consensus; our contribution is the *gradient* (capability smooth
in verifier accuracy q, no threshold) and the policy-vs-judge *asymmetry* (search helps a policy ~4× more
than a judge) (§4). **Self-improving language models** — ReST [Gulcehre et al. 2023], Self-Rewarding LMs
[Yuan et al. 2024] — report self-improvement that works; we contribute the *controlled isolation* showing
the signal *source* is the operative variable, with an external-oracle positive control the at-scale
results do not isolate (§6).

Our capability scale is the classical logistic latent-ability model shared by Elo, Bradley–Terry, and
item-response theory (Rasch); pairwise LLM rating with a judge is the LMSYS-Arena approach [Zheng et al.
2023]. Benchmarks: GSM8K [Cobbe et al. 2021] and MATH [Hendrycks et al. 2021].

## Reproducibility
Code and data generators for all three arms are in the repo. Control (`control/gridworld.py`): the MDP,
exact value iteration, the noisy-evaluator × lookahead sweep — pure NumPy, exact oracle. Games (`games/`):
Connect-4 bitboard engine +
solver, the depth-limited alpha-beta oracle, the conv net, MCTS, the GELO calibrator (`c4_calibrate.py`,
which prints the goodness-of-fit gate), and the self-play / oracle-value loops. Reasoning (`reasoning/`):
the accuracy-vs-N sweep, the evaluator-quality ablation and gradient, the MATH IRT fit, the pairwise arena
+ master-judge (Bedrock), the judge-scaling test, and the size×search frontier. Large datasets and model
weights are regenerable and not committed.

## Appendix A — Experimental setup, per result
So the "what we did" is explicit and reproducible, the exact knobs behind each headline number:

**Arm A — Connect-4 / GELO (`games/`).**
- *Opponent ladder & calibration* (`c4_calibrate.py`, `connect4_ab.py`): opponents are random plus
  depth-*d* alpha-beta (d = 1…6) against the perfect solver's move ordering. Ratings are fit by logistic
  MLE from the full round-robin **cross-table** (not per-opponent win-rates); the goodness-of-fit gate is
  mean |predicted − observed| pairwise win-rate = **0.058**; anchor **random := 0**.
- *Evaluator net* (`c4_net.py`): a small conv policy+value network trained on **oracle-solver labels**
  (value = game-theoretic value, policy = solver move distribution). Data-size sweep uses 12k / 24k / 50k
  label subsets.
- *Placing an agent* (`place_agent`): the agent plays **24–40 games per rung** vs each ladder rung; the
  win-rate vector is fit to the ladder's GELO by the same logistic model. Open-loop = greedy on the policy
  head; closed-loop = **PUCT MCTS, 200 simulations**.
- *Which-lever diagnostic* (`c4_diagnostic.py`): a data (3k–50k) × capacity (small ≈0.3M / large ≈4M
  params) grid, each cell placed both open- and closed-loop, to read where data / capacity / search each
  bind.

**Arm B — reasoning (`reasoning/`), all Qwen2.5-Instruct-4bit under MLX.**
- *Self-consistency & oracle-best-of-N* (`reason_math_sweep.py`): **Qwen-4B**, GSM8K test, **120
  problems**, temperature **0.8**, **1024**-token budget, **N ∈ {1,4,16,32}**; final answer via
  `#### <number>` regex; oracle-best-of-N = "is any of the N correct vs gold."
- *Graded verifier* (same samples): a synthetic selector with per-item accuracy **q ∈ {0.5…1.0}**.
- *Kimi-best-of-N* (`reason_bestofn.py`): **Qwen-1.5B**, **60 problems**, temp 0.8, **N ∈ {2,4,8}**;
  selector = **Kimi-K2.5** (Bedrock, temp 0) picking the correct candidate index — blinded to the gold.
- *Judge scaling* (`reason_arena.py` judge, majority over repeats): Kimi judgments aggregated over
  **N ∈ {1…5}**; agreement measured against the ground-truth verifier on decisive pairs.
- *Serial axis* (`reason_serial.py`): **greedy** (temp 0), max-tokens ∈ **{128,256,512,1024,2048}**,
  Qwen-1.5B and 3B.
- *Size × search frontier*: Qwen **{0.5B,1.5B,3B}**, GSM8K, greedy + sc@{4,16}; compute proxy ≈ params × N.
- *Pairwise reasoning-GELO* (`reason_arena.py`): **5 models** on **50 MATH problems**; **every** ordered
  pair scored by the Kimi-K2.5 judge (**blinded, answer-order randomized**); ratings by Bradley–Terry MLE
  (`bt_elo`), anchor **Kimi := 2800**; the verifier cross-check gives the 72% judge-agreement figure.
- *Difficulty-anchored GELO* (`reason_gelo_irt.py`): the same models vs MATH difficulty tiers L1–L5, Rasch
  (1-parameter IRT) fit.
- *Training-data lever* (`reason_finetune_sweep.py`): LoRA fine-tune **two 0.5B models** — Qwen2.5-0.5B-Instruct
  (4-bit) and Qwen2.5-0.5B base (**bf16**; 4-bit was unstable) — (8 layers, lr 5e-5) on
  **N ∈ {64,256,1024,4096,7000 (full)}** GSM8K examples at **fixed 3 epochs** (so iters scale with N —
  isolates *data*, not optimization budget), calculator-annotation targets stripped; greedy accuracy on
  150 held-out problems, N=0 = base model.

**Arm C — control (`control/gridworld.py`).** 8×8 gridworld, slip 0.1, γ = 0.95, scattered pits; exact
**V\*** by value iteration; degraded evaluator = V\* + σ·(value-spread)·𝒩(0,1); MPC lookahead **h ∈
{1,2,3}** (h Bellman backups, then greedy). Return = mean discounted reward over **400 episodes per grid, averaged over 8
independent grid instances (mean ± sd)**, normalized to [random = 0, optimal = 100].

**§6 self-improvement.** Connect-4 (`c4_selfplay.py`): self-play games each iteration; value target =
game **outcome** (baseline) vs **depth-6 oracle value** (`--oracle-value`, the positive control), loop
otherwise identical; strength re-placed vs the ladder every iteration. Chess uses the analogous loop from
three seeds (from-scratch / weak-policy warm-start / a self-learned +1214 net).

**Infrastructure.** Two **Apple M3 Ultra** machines (each 275 GB unified memory, ~819 GB/s): **MLX /
mlx-lm** for all Qwen inference and net training — batched sampling (`batch_generate`) makes 32-completion
caches and models up to **72B** tractable; **AWS Bedrock** (`moonshotai.kimi-k2.5`, us-east-1) for the
master judge and the frontier-LLM chess vignette. Early exploratory sweeps used modest samples (50–120
problems); the powered reruns so far are §3 (multi-seed) and §5 (eight-grid averaging), while the §4
reasoning numbers are still at exploratory power pending the 300–500-problem, 32-sample cache rerun —
we flag exploratory-power results where they appear.

## References
- Anthony, T., Tian, Z. & Barber, D. (2017). *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS.
- Brown, B. et al. (2024). *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling.* arXiv:2407.21787.
- Browne, C. et al. (2012). *A Survey of Monte Carlo Tree Search Methods.* IEEE TCIAIG.
- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K). arXiv:2110.14168.
- Coulom, R. (2006). *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search.* Computers and Games.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL.* arXiv:2501.12948.
- Gulcehre, C. et al. (2023). *Reinforced Self-Training (ReST) for Language Modeling.* arXiv:2308.08998.
- Hendrycks, D. et al. (2021). *Measuring Mathematical Problem Solving with the MATH Dataset.* NeurIPS.
- Jones, A. L. (2021). *Scaling Scaling Laws with Board Games.* arXiv:2104.03113.
- Kocsis, L. & Szepesvári, C. (2006). *Bandit Based Monte-Carlo Planning* (UCT). ECML.
- Lightman, H. et al. (2023). *Let's Verify Step by Step.* arXiv:2305.20050 (ICLR 2024).
- OpenAI (2024). *Learning to Reason with LLMs* (o1).
- Rosin, C. D. (2011). *Multi-Armed Bandits with Episode Context* (PUCT). Ann. Math. AI.
- Silver, D. et al. (2016; 2017; 2018). *Mastering Go / Go without Human Knowledge / a General RL Algorithm* (AlphaGo, AlphaGo Zero, AlphaZero). Nature; Nature; Science.
- Snell, C., Lee, J., Xu, K. & Kumar, A. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv:2408.03314.
- Wang, X. et al. (2023). *Self-Consistency Improves Chain-of-Thought Reasoning.* ICLR.
- Wu, D. J. (2019). *Accelerating Self-Play Learning in Go* (KataGo). arXiv:1902.10565.
- Yao, S. et al. (2023). *Tree of Thoughts.* NeurIPS.
- Yuan, W. et al. (2024). *Self-Rewarding Language Models.* arXiv:2401.10020.
- Zelikman, Y. et al. (2022). *STaR: Bootstrapping Reasoning with Reasoning.* NeurIPS.
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.
- **Software/data:** Stockfish · MLX · mlx-lm · AWS Bedrock (Kimi-K2.5) · Lichess cloud evals · HuggingFaceH4/MATH-500.
