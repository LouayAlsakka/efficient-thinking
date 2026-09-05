# Efficient Thinking VIII — Experience Priors: Proposal and Way Forward

**Status:** proposal (registered by this commit; experiments not yet run) · **Companion:** `efficient-thinking-8-experience-priors.md`
(the concept draft, canonical copy), `et8-review-ri.md` (review) · **Drafted:** Ri (理) for Louay Alsakka, 2026-09-05 ·
**Runs:** Sautee on the LLM machines (llm1/llm2) once the harness is green on a Studio; Ri owns the design, the
predictions and the write-up.

One sentence: **build a small verifier-gated environment where a frozen Qwen wastes search in a measurable, recurring
way; harvest the waste into two lists (what worked, what didn't) with the record schema below; inject the lists into the
model at five different depths, from prompt to hidden state; and measure which injection buys the most search
reduction per parameter, against text memory and LoRA, with the base model never trained.**

---

## 0. What this paper must show, and what would kill it

**Claim under test.** A module orders of magnitude smaller than the model can learn *where to search* from verified
trajectories, and reduce search cost on unseen structurally similar tasks without touching *how the model reasons*.

**Killed if** (any one): distilled text memory matches the best injected prior at equal total compute (P2 fails); the
gain does not transfer to held-out variants (P1 fails on the held-out split); out-of-domain regression exceeds the
budget (P5); a verified-then-invalidated lesson cannot be unlearned (P6); or the effect is answer memorization rather
than search change (the memorization control, §6.3, is positive).

**The two lists (Louay, 2026-09-05).** Per consolidation round, per agent: *what worked* (several ways tried, one won) and
*what didn't* (tried hard, failed; learned over time to be a dead path). Everything below is machinery for compiling those
two lists honestly and injecting them without building a prison.

---

## 0b. Measured on day one (2026-09-05, harness v0.1–v0.3, 20 tasks each, this Mac) — what the baselines actually do

| run | model | green/20 | mean actions/12 | dominant waste |
|---|---|---|---|---|
| v0.1 | 3B bf16 / 7B-4bit | 9 / 7 | 8.9 / 8.9 | identical re-patches (61 / 102 of 178 steps); concrete prompt example biased first hypothesis |
| v0.2 | 3B / 7B | 3 / 6 | 10.7 / 9.3 | greedy decoding: identical prompt → identical action (100 / 136 re-patches) |
| v0.3 | 3B / 7B | 8 / 7 | 9.25 / 9.25 | **region fixation**: failed episodes touched ONE region in 11/12 (3B) and 13/13 (7B); "patches" byte-identical to the shown code |

Three lessons, each a harness defect made visible by the verifier and the trajectory labels — the detect → verify loop
working on the harness before it works on the model: (1) a concrete example in the prompt is a prior, and the untrained
baseline must not carry one; (2) greedy decoding cannot search — the first attempt stays greedy for reproducibility, and
variation is supplied only after a wasted step; (3) the real baseline waste is not "following the red herring" (13/15
first hypotheses already went upstream of the symptom) but **fixation on the first region inspected** and emitting the
unchanged code as a "patch" when the fix is unknown. **Hypothesis ORDER across regions is therefore the quantity the
prior must move, and `distinct regions touched per failed episode` joins the primary metrics.** Paper II's size ordering
did not separate 3B from 7B-4bit at n = 20 on this task shape; no model comparison is claimed until two seeds agree.
Results files: `experience/results/baseline_{3b,7b}_harness_v0.{1,2,3}.json`.

**Baseline decision:** Qwen2.5-3B-Instruct bf16 stays the primary (hidden states writable, already on disk, no worse than
7B-4bit here); Qwen2.5-7B-Instruct **bf16** (not 4-bit) is the transfer model and must be fetched for the injection work.

## 1. Base model

`mlx-community/Qwen2.5-3B-Instruct` (bf16 for steering/adapter work — hidden states must be writable; the 4-bit caches
from Paper II remain the frozen-baseline reference). Why 3B: Paper II places the size/search crossover at ~7B and shows
<3B "rarely finds the answer, leaving search nothing to select". ET-8 needs a model that *can* solve the tasks but does so
wastefully, so that a prior has something to reallocate: 3B on the environment below is the target regime, with 7B as the
transfer check (P7). Qwen2.5-3B-Instruct is already on disk (`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct`);
mlx-lm 0.31 is installed. Layer count 36, hidden 2048.

**Chess anchor, in parallel and first to produce a number.** Paper I's 3.45M value net + PUCT is a frozen evaluator with
an explicit policy prior and an exact external oracle (Stockfish). A learned prior over moves trained on verified
self-play trajectories, measured as *simulations to a fixed Elo rung*, is ET-8's mechanism with no tokenizer, no
retrieval and no judge. It is anchor A0; it runs on the chess code already in `chessnet/` and costs no LLM time.

---

## 2. Environment — where search is wasted measurably

### 2.1 `et8_env`: trap-debugging tasks with an external verifier

Small Python programs (30–80 lines), each with a hidden injected bug drawn from a **taxonomy of 12 bug classes**, a
**unit-test suite** (the external verifier — Paper II/III: never the model judging itself), and a **symptom** (the
failing test output the agent sees first). The agent loop is explicit and every step is an *action* the prior can bias:

```
  observe symptom → HYPOTHESIZE (which class / which region)  → INSPECT (open a region)
  → PATCH (edit) → RUN TESTS → repeat until green or budget exhausted
```

Cost = actions and tokens to green. Waste = actions on regions/classes that were not the bug.

**The recurring structure the prior must learn** (this is what makes it *experience* rather than *knowledge*):

- **Traps:** in 30% of tasks the symptom points at the wrong region (a red herring: the failing assertion is downstream
  of the bug). A fresh model follows the symptom; an experienced one checks the upstream producer first.
- **Class priors:** bug classes are sampled non-uniformly per *project family* (e.g. family A is mostly boundary errors,
  family B mostly mutable-default/aliasing). The prior should learn family-conditional hypothesis order.
- **Dead paths:** two bug classes are *never* the cause in a given family but look plausible (e.g. "off-by-one" in a
  family whose loops are all range-safe). Trying them hard fails every time. This is Louay's "what didn't work".
- **Held-out variants:** same families, unseen programs; and a *shifted* split where a dead path becomes live (the
  reopen test, P6).

Bug taxonomy v0 (12): off-by-one boundary · wrong comparator (`<` vs `<=`) · mutable default argument · aliasing
(shared list) · integer vs float division · string/bytes or width mismatch (the estate's U+202F class) · timezone/
naive datetime · wrong dict key / missing default · early return inside loop · swapped arguments · unsorted-input
assumption · exception swallowed. Each has a generator that injects it into a clean program and a test that catches it.

### 2.2 Why not GSM8K/MATH

Paper II's caches make answer accuracy free to measure, but a math problem has no *structured* search over hypotheses
that a small prior can reorder, and best-of-N samples are not "which path first". Math enters only in §6.3 as the
**memorization control** (a prior trained on debugging must not move GSM8K accuracy).

### 2.3 Size

2,000 training tasks (10 families × 200), 400 held-out same-family, 400 shifted-family, 200 out-of-domain (a fresh
family). Budget per task: 12 actions. One pass of 3B over 2,000 tasks × ~12 actions × ~400 tokens ≈ 10M tokens — an
afternoon on a Studio, minutes on llm1.

---

## 3. Build experience — detect, distill, verify

1. **Detect.** Run the frozen model on the training tasks; log complete trajectories (`experience/traj/*.jsonl`): every
   action, the region inspected, the hypothesis text, tokens, verifier result. Mark `τ+` (green, ≤ median actions) and
   `τ−` (not green, or green at > 1.5× median). Also mark each *action* as productive/unproductive by hindsight (did it
   touch the true bug region/class).
2. **Distill.** Cluster unproductive actions by (family, symptom class, hypothesis class) → candidate lessons in the
   **record schema**:
   ```
   condition          family=B, symptom=IndexError in consumer
   action_prior       inspect producer first; check boundary in producer loop
   action_avoid       inspect consumer; hypothesize off-by-one in consumer
   alternatives_tried [consumer-first, producer-first, random]  ← a "works" needs this non-empty
   evidence           n=41 episodes, 3 finders (seeds)
   confound_check     held-out same-family: producer-first wins 37/40; shuffled-family control: no effect
   scope              family=B, symptom∈{IndexError, KeyError}
   confidence         0.9 (decays 0.9/round without new evidence)
   expiry / reopen    reopen if producer-first fails ≥3 consecutive held-out tasks
   ```
   This is the WO-227 claim record with `alternatives_tried` and `reopen_condition` added (review §6.5).
3. **Verify (the gate).** A lesson is admitted only if it improves decisions on ≥ k = 20 *held-out* tasks in scope and
   does not degrade decisions out of scope (shuffled-family control). Register the **rejection rate** — the fraction of
   candidate lessons the external verifier throws out is the measurement of how wrong a self-verified loop would have
   been (P0). Rejected lessons are kept, labelled, as the negative set for §5's dogma arm.
4. **Consolidate.** Verified lessons → training pairs at decision points: (state, productive action) vs (state,
   unproductive action). Rounds: R1 on tasks 1–1000, R2 adds 1001–2000, R3–R5 on fresh draws; regression-test each round
   on held-out and OOD before accepting (§15 of the draft, "sleep").

---

## 4. Injection points — where the vector goes

All five share one training set (the verified pairs from §3) and one loss (§5). They differ only in *where* experience
enters. Ordered from cheapest to deepest:

| # | mechanism | what is learned | params (3B) | where injected | hypothesis |
|---|---|---|---|---|---|
| A | **text memory** (baseline) | nothing; distilled lessons prepended | 0 | prompt | the bar to beat |
| F | **action-logit bias** | `E_φ(a\|s)` as a bias on the action-choice tokens only | ~10^3–10^4 | logits at decision points | cheapest real prior; if it wins, nothing deeper is needed |
| B | **soft prefix** | k=16–64 virtual tokens | k×2048 ≈ 33k–130k | embedding layer | classic; costs context, not FLOPs |
| C | **steering vectors** | one vector per chosen layer, `h_l += α_l v_l` | L×2048 (L=1–6) | residual stream, layers swept at 25/50/75% depth (9/18/27 of 36) | mid-depth carries plan/decision features; gradient-free init by contrastive activation (mean h over τ+ minus τ−) |
| D | **residual adapter** | bounded MLP `h_l += β·tanh(W₂σ(W₁h_l))`, r=16–64 | ~2×2048×r per layer ≈ 65k–260k | 1–3 mid layers | state-dependent prior when a static vector is insufficient |
| E | **LoRA r=8** (baseline) | attention q,v deltas | ~1–2M | all layers | does a generic PEFT already do the job? |

Two placement questions, each a sweep: **which layers** (early/mid/late, single vs three), and **which tokens** (decision
tokens only — the point where the next hypothesis is chosen — vs every token). The draft's §17 Q2/Q3 are these two sweeps.

**Mechanics in mlx-lm.** Qwen2.5 decoder layers are `model.model.layers[i]`; wrap the forward of a chosen layer to add
`α_l v_l` (C) or the adapter output (D) to its residual output; expose a `decision_mask` so the addition applies only at
decision tokens when required. Steering vectors need no gradient for their initial value: collect `h_l` at decision tokens
for τ+ and τ− trajectories, take the mean difference, normalize; then fit `α_l` (and optionally `v_l`) with the loss below.
The 4-bit caches are not usable here; run 3B in bf16 (≈6 GB).

---

## 5. Training objective and the flexibility constraints

**Pairwise preference at decision points** (draft §12): `L_exp = −log σ(S_φ(τ⁺) − S_φ(τ⁻))`, with `S_φ` the log-prob of
the chosen action under the injected model; plus `λ‖E_φ‖²` (intervene only as much as necessary); plus a **KL-to-base
term on out-of-scope states** (the shuffled-family control set), so the prior is regularized toward zero influence where
it was not earned.

**Louay's probabilistic-weight rule, made operational** (review §6 and the 2026-09-05 discussion), as four constraints on
the update, tested in P6:

1. **No weight reaches 0 or 1.** Action priors are clamped to `[ε, 1−ε]`, ε = 0.02, so dead paths keep a floor and can be
   rehabilitated. Equivalent in search terms: the exploration bonus never vanishes.
2. **Scope-weighted strength.** The prior's α is multiplied by a scope-match score (family/symptom similarity); a lesson is
   strong at home and weak abroad. Extrapolation is the failure this prevents.
3. **Forgetting factor.** Evidence counts decay γ = 0.9 per consolidation round without fresh support; recent evidence
   outvotes old.
4. **Update on surprise.** A prediction the prior got wrong moves the weight by the full step; a confirmation moves it by a
   fraction (1/(n+1)), so favored paths do not inflate through repeated confirmation alone.

Rigidity becomes two numbers: **override rate** (how often the prior suppressed the action that turned out correct) and
**persistence half-life** (rounds for an invalidated lesson to fall below the floor). Both are the dogma arm (P6).

---

## 6. Experiments, ablations, controls

**6.1 Main grid.** {A, F, B, C, D, E} × {R1, R2, R3} × {held-out, shifted, OOD}. Metrics per cell: actions-to-green,
tokens-to-green, success at fixed budget (12 actions), time-to-first-correct-hypothesis, **Experience Gain**
`(C_base − C_exp)/C_base` at equal success. Every cell n ≥ 400 tasks, paired against the frozen baseline on the same
tasks; significance by paired sign test / McNemar on success, paired t on cost (the series' rule: no delta believed
without a low-variance re-measurement — two seeds minimum).

**6.2 Ablations (draft §14, all kept).** frozen + no memory · frozen + raw episode retrieval · frozen + distilled text ·
each injected prior · prior + text · LoRA · **shuffled prior** (same parameters, trained on shuffled τ± labels) · prior
trained **without the verification gate** (admits all candidate lessons).

**6.3 Controls that decide interpretation.**
- *Memorization control:* GSM8K accuracy (Paper II cache protocol) before/after each prior. Must not move (|Δ| ≤ 1 pt).
- *Reasoning-integrity control:* held-out tasks with the bug in a region the lesson says to avoid — the prior must be
  overridable by evidence (success must not collapse; measures constraint 1).
- *Positive control for the verifier:* seed 5% of candidate lessons with known-false rules; the gate must reject ≥ 95%.

**6.4 Dogma arm (P6).** After R3, invalidate one admitted lesson by shifting its family (the dead path becomes live). Run
R4–R5. Measure persistence half-life under (i) plain averaging and (ii) the four constraints of §5.

**6.5 Transfer (P7).** Take the C/D module trained on 3B; measure whether its *lesson set* (not its weights) transfers to
7B via text memory, and whether the 3B steering vectors transfer directly to 7B when projected — expected no; registered
as a prediction so the negative is a result.

---

## 7. Registered predictions (scored in §10 of the paper, hit or miss)

- **P0** The external verifier rejects ≥ 20% of candidate lessons distilled from τ± clustering. (The self-verified loop
  would have consolidated those.)
- **P1** The best injected prior reduces actions-to-green on **held-out same-family** tasks by ≥ 25% at equal success
  versus the frozen baseline (R3).
- **P2** That prior beats distilled text memory (A) at ≤ 0.5× total inference tokens, or matches it at ≤ 0.25×. If A wins
  or ties at equal tokens, the mechanism is not worth scaling — this is the paper's primary prediction.
- **P3** Mid-depth injection (layer ≈ 50%) beats early (25%) and late (75%) for C and D; decision-token-only injection
  matches all-token injection at lower regularization cost.
- **P4** Gradient-free steering vectors (contrastive means) recover ≥ 50% of the trained adapter's gain.
- **P5** Out-of-domain regression ≤ 2 points success and GSM8K |Δ| ≤ 1 point for C, D, F; LoRA (E) exceeds one of these.
- **P6** Under §5's constraints an invalidated lesson decays below the floor within ≤ 3 rounds; under plain averaging it
  persists ≥ 5 rounds. Override rate at R3 ≤ 5% with constraints.
- **P7** Lesson *text* transfers to 7B (gain ≥ 50% of 3B's); steering *vectors* do not (gain ≤ 10%).
- **P8 (chess anchor A0)** A learned move prior trained on verified self-play reduces simulations-to-2600 by ≥ 30% versus
  the uniform prior at equal evaluator.
- **P9** Efficiency improves monotonically over R1→R3 on held-out, then plateaus; the plateau is set by the verifier's
  admitted-lesson count, not by module size (doubling r does not move it).

Overshoots are flagged like undershoots. Predictions that miss are narrated in full.

---

## 8. Build order (Sautee runs on llm1/llm2 once the harness is green on a Studio)

| week | deliverable | owner | gate |
|---|---|---|---|
| 1 | `experience/et8_env.py`: generator (12 classes × 10 families), verifier, trajectory logger; unit tests on the generator (every injected bug is caught by its test; clean programs pass) | Ri | G1: 100 tasks generated, verifier 100/100 on clean, catches 100/100 injected |
| 1 | A0 chess prior: hook a learned move prior into `chessnet/search.py` PUCT; train on existing self-play trajectories; measure sims-to-Elo | Ri (design), Sautee (runs) | P8 scored |
| 2 | frozen-baseline pass over 2,000 tasks, trajectories logged; text-memory baseline (A) built and run | Sautee | G2: baseline numbers and τ± labels exist |
| 2 | distill + verify pipeline; rejection rate measured (P0) | Ri | G3: admitted-lesson set v1 |
| 3 | F (logit bias) and C (steering vectors, gradient-free init) on 3B bf16; layer sweep | Ri builds, Sautee sweeps | P3, P4 first numbers |
| 4 | B (prefix), D (adapter), E (LoRA r=8) on identical pairs; main grid R1 | Sautee | P1, P2, P5 at R1 |
| 5 | consolidation rounds R2–R3; dogma arm; transfer | Sautee | P6, P7, P9 |
| 6 | write-up in series format: one message, results at a glance, anchors, scored predictions, honest scope | Ri, Louay's voice pass | G0 |

Definition of done per deliverable: results JSON committed; prediction scored; one paragraph of canonical text written
from the numbers; verification command and its expected output stated with every "done".

---

## 9. Relationship to the estate (why NiraNet is the second environment, later)

The estate's coordination layer already produces the raw material of §3 every night — thousands of claim records with
provenance, the WO pile of doctrine lines, per-lane memory notes — and the 2026-09-05 shift produced textbook instances
of both lists (what worked: qualified fetch, single-file pushes, alternates; what didn't: shallow clones, partial clone
on lm, bare fetches) and of the failure P6 guards against (a withdrawn instruction held by every lane). Once §2's
synthetic environment has settled the mechanism, the estate's real trajectories are the second environment, and the
two-list nightly compile is the consolidation schedule. Not before: the synthetic environment exists to remove
confounds the estate cannot.

---

## 10. Honest scope

Small model, one synthetic environment, two machines plus llm1/llm2. Claims will be scoped to this regime. The
chess anchor decides whether the mechanism exists at all; the Qwen grid decides whether it survives contact with
language, retrieval and a prompt. If P2 fails, the honest conclusion is that retrieval is sufficient at this scale, and
that is worth publishing.
