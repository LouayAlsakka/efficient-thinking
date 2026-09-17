a
# Efficient Thinking 8: Experience Priors
## Continual Search Optimization Through Consolidation

**Status:** Early research proposal / working draft  
**Series:** Efficient Thinking I–VIII  
**Date:** September 2026

---

## Abstract

Large language models can repeatedly perform the same class of work without becoming meaningfully more *experienced* at that work. An agent may solve thousands of related tasks, discover recurring dead ends, learn useful shortcuts, and accumulate extensive memory, yet each new inference largely begins with the same fixed model parameters. Current agent systems compensate by retrieving logs, summaries, prompts, and external memory. This preserves information, but it does not reproduce an important property of human expertise: repeated experience changes which possibilities are considered first and which are ignored.

This paper proposes **Experience Priors**, a lightweight, continually learned mechanism that modifies a model's **search policy rather than its underlying intelligence**. The foundation model remains frozen. Experience is consolidated into a small auxiliary module that biases the model toward historically productive reasoning paths and away from repeatedly unproductive ones. The objective is not to make the model intrinsically smarter, add factual knowledge, or replace memory. It is to make repeated reasoning increasingly efficient.

The proposed lifecycle is **detect → distill → verify → consolidate**. During normal operation, agents collect candidate experiences. A verifier determines whether an apparent lesson generalizes beyond a single episode. Verified lessons are periodically consolidated—analogous to an offline "sleep" phase—into an experience-prior module. The resulting system can then be evaluated on search cost, latency, success rate, and transfer while keeping the base reasoning model fixed.

We propose initial experiments on a small open model such as Qwen, where internal transformer states are accessible, comparing text memory, soft prompts, hidden-state steering, lightweight residual adapters, and conventional QLoRA. The central hypothesis is that substantial gains in repeated-task efficiency can be obtained by learning **where to search**, without retraining **how to reason**.

---

## 1. Motivation

Consider two engineers with identical raw intelligence. One has just joined a project; the other has worked on it for five years. The senior engineer may not reason more deeply in a general sense. Instead, experience changes the prior distribution over possible solutions:

- which failure modes are likely;
- which questions should be asked first;
- which branches are probably dead ends;
- which tools are appropriate;
- which details are diagnostic;
- which apparent solutions have failed before.

Modern LLM agents do not naturally acquire this property. An agent can work continuously for months while the underlying model remains identical to its first day on the job.

The common solution is external memory. Previous episodes are stored in files, databases, vector stores, summaries, or increasingly long prompts. At inference time the system retrieves relevant material and asks the model to reinterpret it.

This works, but has several scaling limitations:

1. **Repeated interpretation cost.** The model repeatedly spends tokens and compute rereading lessons it has already encountered.
2. **Context competition.** Experience competes with the actual task for finite context-window attention.
3. **Retrieval dependency.** A useful lesson has no effect if it is not retrieved.
4. **Weak proceduralization.** Knowing a written lesson is different from automatically beginning with a better search prior.
5. **Long-term growth.** A worker with years of experience should not require replaying years of summaries before every decision.

This motivates a separate mechanism for converting repeated verified experience into persistent search behavior.

---

## 2. Intelligence and Experience Are Different Variables

The key distinction in this proposal is between **intelligence** and **experience**.

### 2.1 Intelligence

For this paper, intelligence refers to the capabilities embodied by the frozen foundation model: representation, abstraction, language understanding, world modeling, planning, deduction, and general reasoning ability.

Improving these capabilities normally implies changing a substantial portion of the model through pretraining, continued pretraining, fine-tuning, reinforcement learning, or other expensive optimization.

### 2.2 Experience

Experience is narrower. It changes the probability assigned to candidate reasoning paths *before or during search*.

A useful abstraction is:

> **Intelligence determines what paths the system is capable of reasoning through. Experience determines which paths it tries first.**

Experience therefore behaves like an adaptive prior over search.

This distinction is important because the objective of this work is explicitly **not** to continuously retrain the intelligence of the model. The goal is to preserve a stable reasoning engine while allowing an individual agent—or a specialized group of agents—to become progressively more efficient at its recurring work.

---

## 3. Connection to Search

The idea is closely related to search policies in classical planning and game playing.

In Monte Carlo Tree Search, the underlying search algorithm may be unchanged while a policy prior dramatically alters which branches receive computation. A stronger prior does not necessarily expand the set of moves the search can theoretically discover. Instead, it allocates finite compute more intelligently.

The same principle may apply to language-model reasoning.

Let a frozen model define a conditional distribution over possible next reasoning actions:

\[
P_0(a_t \mid s_t)
\]

where \(s_t\) represents the current reasoning state and \(a_t\) represents a possible continuation, tool call, hypothesis, decomposition, or other search action.

Introduce an experience prior \(E_\phi\), parameterized by a comparatively small set of learned parameters \(\phi\):

\[
P(a_t \mid s_t, E_\phi) \propto P_0(a_t \mid s_t) \cdot E_\phi(a_t \mid s_t)
\]

The base distribution remains intact. The experience module changes how search effort is allocated.

A more implementation-oriented formulation could modify hidden states rather than explicit probabilities:

\[
h'_l = h_l + \alpha_l E_\phi(h_l, c)
\]

where \(h_l\) is the frozen model hidden state at layer \(l\), \(c\) is optional task or agent context, and \(E_\phi\) is a small learned residual function.

The precise mechanism is an experimental question. The conceptual constraint is more important: **the experience component should steer search without becoming a replacement foundation model.**

---

## 4. Experience Prior Architecture

A first architecture contains three major components.

### 4.1 Frozen Reasoning Engine

The base LLM remains frozen during normal experience consolidation.

Its role is to provide general intelligence and broad knowledge. Freezing it provides a controlled experimental condition and reduces catastrophic forgetting, instability, and computational cost.

### 4.2 Experience Prior

Each persistent agent has a small trainable component representing its accumulated procedural experience.

Possible implementations include:

- learned soft-prefix vectors;
- layer-specific steering vectors;
- low-rank residual transformations;
- small MLP adapters operating on hidden states;
- attention bias modules;
- policy heads that rerank candidate reasoning actions;
- sparse collections of specialized experience modules selected by a router.

The experience prior need not contain complete memories. Its purpose is to encode tendencies such as:

> In states resembling this one, branches of type A have repeatedly failed; investigate B and C first.

### 4.2a What the Prior Is For, Stated as a Constraint

An experience prior is defined not by its implementation but by what it is permitted to do to the frozen model.
Let \(\pi_0\) be the frozen model's search policy and \(\pi_\phi\) the same model with the prior applied. Let
\(C(\pi)\) be the search cost — actions per episode, or tokens — and \(Q(\pi)\) the capability, measured by an
**external** verifier: task success against tests, form against a rule table, a blind judge. Never the training
loss. The prior is the solution to

\[
\min_\phi \; C(\pi_\phi) \quad \text{subject to} \quad Q(\pi_\phi) \ge Q(\pi_0) - \epsilon
\]

> **Minimise search cost subject to preserving the frozen model's capability.**

Three rules make this measurable rather than rhetorical:

- \(\epsilon\) is declared before the run, not chosen after it.
- \(C\) and \(Q\) are measured on the same held-out tasks, **paired per slice**, and the constraint must hold on the
  slices, not on a pooled mean — a mean can be held by one slice while three fail.
- The verifier itself ships with its reach: the fraction of positions or cases it could actually read. A verifier
  that silently skips what it cannot parse inflates \(Q\); verify the learner, and verify the verifier.

The consequences are sharp. A mechanism that lowers cost by lowering capability is not a weaker prior; it is not a
prior. Every failure measured in §20b is a failure of the constraint, not of the objective: the injected text
memory (success 62.5% → 20.0%, and → 5.0% with better lessons), the poet's LoRA (form 32.7% → 17.3%), and steering
at its first strength (62.5% → 51.2%). The verifier is therefore not a safety feature bolted onto the lifecycle of
§5; it is the measurement of the constraint, and the gate of §4.3 is where the constraint is enforced before
consolidation. The regulariser of §12 is the relaxed form of the same constraint. And the accumulation claim of
§18 — work, consolidate, improve, again — is the statement that the constraint keeps holding across generations
while \(C\) keeps falling; a generation that buys cost with capability has ended the sequence.

### 4.3 Verification Gate

Continual learning creates a dangerous problem: **bad experiences can become bad instincts**.

An incorrect conclusion from one episode should not immediately modify future behavior. Therefore, candidate experiences pass through a verification stage before consolidation.

The verifier may use:

- objective task outcomes;
- unit tests;
- external judges;
- stronger models;
- repeated trials;
- counterexamples;
- synthetic variants;
- cross-agent evidence.

Only sufficiently supported experiences are eligible for persistent consolidation.

---

## 5. The Experience Lifecycle

We propose four stages:

### 5.1 Detect

During ordinary work, identify episodes containing a meaningful learning signal.

Examples include:

- a long reasoning path that eventually proved unnecessary;
- a recurring failure pattern;
- a shortcut discovered after extensive exploration;
- an incorrect assumption corrected by evidence;
- a tool-selection pattern that consistently saves time;
- a task decomposition that repeatedly succeeds.

Not every interaction should become experience. Most events are noise.

### 5.2 Distill

Convert the raw trajectory into a compact candidate lesson.

A useful intermediate representation might be:

```text
condition: characteristics of the state where the lesson applies
action_prior: reasoning/search actions to favor
action_avoid: actions or branches to deprioritize
evidence: episodes supporting the lesson
confidence: estimated reliability
scope: estimated domain of applicability
```

This symbolic form does not have to be the final representation. It provides an auditable intermediate object before learning occurs.

### 5.3 Verify

Test whether the candidate experience generalizes.

For example, suppose an agent concludes that a Git-backed messaging system becomes inefficient beyond a certain workload because particular repository operations dominate. The lesson should not be consolidated merely because it occurred once. The system could generate related architectures, workloads, and constraints and test whether the proposed heuristic predicts successful search decisions.

Verification asks:

> Does this experience improve decisions in neighboring states, or did we merely memorize one episode?

### 5.4 Consolidate

Verified experiences are periodically converted into updates to the experience prior.

This stage is intentionally separated from online task execution.

Rather than updating weights after every interaction, the system accumulates experiences and performs scheduled consolidation. This resembles a simplified computational analogy to sleep: episodes are collected during activity and reorganized offline.

Possible schedules include:

- nightly consolidation of recent experiences;
- weekly consolidation across recurring patterns;
- longer-term compression or pruning of redundant experience;
- event-triggered consolidation after enough verified evidence accumulates.

The important distinction is between **fast episodic memory** and **slow procedural consolidation**.

---

## 6. Memory and Experience Should Coexist

Experience priors are not a replacement for external memory.

A mature agent may require at least three distinct stores:

| Component | Purpose | Example |
|---|---|---|
| Foundation model | General intelligence and knowledge | Understand distributed systems |
| Episodic/semantic memory | Explicit facts and past events | "Server X failed on Tuesday" |
| Experience prior | Procedural search bias | "Check failure mode Y early in this class of incidents" |

This separation is analogous to the difference between remembering an event and acquiring an instinct from repeated events.

Some information should **never** be consolidated into an experience prior. Exact names, dates, changing facts, source documents, and auditable records belong in explicit memory. Experience should primarily capture reusable search structure.

---

## 7. Why Not Simply Use a Prompt?

An obvious baseline is to summarize experience as text and prepend it to every task.

That baseline is essential, but it is not equivalent to the proposed mechanism.

A text prompt requires the model to:

1. retrieve the right lesson;
2. tokenize it;
3. attend to it;
4. reinterpret it on every inference;
5. translate the linguistic statement into internal activations;
6. maintain its influence throughout reasoning.

A learned internal prior can potentially place the bias closer to the model's native representation and apply it continuously without consuming ordinary context tokens.

However, this is an empirical claim, not an assumption. A central experiment should therefore ask:

> **Can an experience prior outperform an equivalent text-memory system at equal or lower total compute?**

If the answer is no, external memory may remain the better engineering solution.

---

## 8. Why Not QLoRA?

QLoRA and related parameter-efficient fine-tuning methods provide an important comparison.

They can efficiently specialize a model, but their objective is generally broader than what is proposed here. A conventional adapter may learn new answer distributions, styles, domain knowledge, and reasoning behavior simultaneously.

Experience Priors impose a stronger conceptual constraint:

> Learn persistent **search preference** while preserving the base model's general reasoning capability.

This suggests several possible differences from ordinary QLoRA:

- much smaller trainable state;
- training targets based on trajectory ranking rather than answer imitation;
- explicit positive and negative search branches;
- strong regularization toward zero influence outside the learned domain;
- reversible or agent-specific experience modules;
- continual consolidation rather than one-time task fine-tuning.

QLoRA should nevertheless be included as a baseline. It may turn out that a carefully designed LoRA implementation is already an effective substrate for Experience Priors.

---

## 9. Individual and Shared Experience

The architecture naturally supports persistent specialized agents.

Suppose thirty agents begin with the same frozen base model. After months of work, their intelligence remains identical, but their experience priors diverge.

A systems-engineering agent develops priors around architecture and debugging. A deployment agent develops operational priors. A research agent develops priors about literature search and experimental design.

This creates an interesting decomposition:

\[
\text{Agent}_i = \text{Frozen Intelligence} + \text{Memory}_i + \text{Experience Prior}_i
\]

Experience may also be shared selectively. A verified lesson learned by one agent could be evaluated against another agent's task distribution before being merged.

This raises a future possibility of **experience inheritance**: a new agent could begin with a mature prior derived from predecessor agents while retaining the same foundation model.

---

## 10. Preventing Experience Corruption

Continual adaptation introduces several failure modes.

### 10.1 Overfitting

A successful strategy in one narrow environment may become a harmful default elsewhere.

### 10.2 Catastrophic Interference

New experience could overwrite previously useful experience.

### 10.3 False Lessons

An outcome may have succeeded for reasons unrelated to the inferred lesson.

### 10.4 Temporal Drift

Some experiences expire as software, organizations, markets, or environments change.

### 10.5 Excessive Prior Strength

A strong prior can prevent exploration. Expertise can become dogma.

Therefore an experience system should preserve an exploration mechanism and permit the frozen model to override its prior when evidence conflicts with it.

Possible safeguards include confidence decay, domain gating, regularization, replay of older verified experiences, versioned adapters, rollback, and explicit measurements of performance outside the experience domain.

---

## 11. Proposed Proof of Concept

The first experiment should be deliberately small.

The objective is not to demonstrate human-like lifelong learning. It is to answer a narrower question:

> Can a small learned module acquire reusable search priors from verified trajectories and reduce future reasoning cost without modifying the base model?

### 11.1 Base Model

Use a small open Qwen model whose transformer layers and hidden states are fully accessible.

The model should be small enough to run repeated training and ablation experiments locally. The purpose is mechanism validation, not benchmark leadership.

### 11.2 Task Environment

Choose tasks with:

- objectively verifiable outcomes;
- multiple plausible search paths;
- recurring structural patterns;
- measurable cost for unnecessary exploration.

Candidate environments include:

1. small program debugging tasks with unit tests;
2. constrained planning problems;
3. miniature architecture-selection tasks;
4. game/search environments from earlier Efficient Thinking work;
5. synthetic tool-use tasks containing recurring traps.

A chess-like environment may be especially useful because the distinction between evaluator intelligence and search prior is unusually clear.

### 11.3 Generate Experience

Run the frozen model on many tasks while preserving complete trajectories:

- hypotheses considered;
- branches explored;
- tool calls;
- intermediate conclusions;
- final outcome;
- verifier feedback;
- token and compute cost.

Identify trajectories where the model eventually discovers that an earlier branch was wasteful or where a successful heuristic recurs.

### 11.4 Train the Experience Prior

Initially compare several mechanisms:

**A. Text Memory Baseline**  
Retrieve distilled lessons and prepend them to the prompt.

**B. Soft Prompt / Prefix**  
Train a small set of virtual tokens representing consolidated experience.

**C. Hidden-State Steering**  
Learn one or more vectors added to selected transformer layers.

**D. Residual Experience Adapter**  
Train a small network that maps hidden states to bounded residual updates.

**E. LoRA / QLoRA Baseline**  
Fine-tune low-rank adapters on the same verified trajectories.

The experiment should control for training data and evaluate both task quality and total inference cost.

---

## 12. Training Objective

A key research question is what the experience module should optimize.

Training only on final correct answers risks teaching general task behavior rather than search efficiency.

Instead, the learning objective should include trajectory preference.

Given two trajectories for the same or equivalent task:

- \(\tau^+\): successful and efficient;
- \(\tau^-\): unsuccessful or unnecessarily expensive;

train the experience module to increase preference for states/actions associated with \(\tau^+\) relative to \(\tau^-\), while keeping the foundation model frozen.

Conceptually:

\[
\mathcal{L}_{exp} = -\log \sigma(S_\phi(\tau^+) - S_\phi(\tau^-))
\]

where \(S_\phi\) measures the experience prior's preference for a trajectory.

A regularization term should constrain the prior's magnitude:

\[
\mathcal{L} = \mathcal{L}_{exp} + \lambda \lVert E_\phi \rVert^2
\]

This expresses an important design principle:

> **Experience should intervene only as much as necessary.**

Read against §4.2a, \(\lambda\) is the price of the capability constraint: the regulariser is the relaxed form of
\(Q(\pi_\phi) \ge Q(\pi_0) - \epsilon\), and the constraint, not the loss, is what decides whether a trained prior is kept.

---

## 13. Experimental Metrics

Success should not be measured only by final benchmark accuracy.

The hypothesis predicts improvements primarily in **search efficiency**.

Primary metrics:

- reasoning/search tokens to solution;
- number of explored branches;
- tool calls;
- wall-clock latency;
- compute consumed per solved task;
- success rate under a fixed compute budget;
- time-to-first-correct-hypothesis.

Secondary metrics:

- generalization to unseen variants;
- transfer to neighboring task families;
- performance outside the learned domain;
- resistance to misleading experiences;
- rate of experience accumulation;
- adapter size versus efficiency gain.

A particularly useful measure may be:

\[
\text{Experience Gain} = \frac{C_{baseline} - C_{experienced}}{C_{baseline}}
\]

for equal task success, where \(C\) is total reasoning compute.

---

## 14. Critical Ablations

A convincing result requires showing *what* produced the gain.

At minimum compare:

1. frozen model with no memory;
2. frozen model + raw episode retrieval;
3. frozen model + distilled textual lessons;
4. frozen model + experience prior;
5. frozen model + experience prior + textual memory;
6. QLoRA specialization;
7. randomly initialized or shuffled experience prior;
8. experience prior trained without verification.

The most important comparison is likely **distilled text memory versus internal experience prior**. If both achieve similar quality, the question becomes whether the internal prior provides lower inference cost, better scaling, or stronger persistence.

---

## 15. Consolidation as "Sleep"

The sleep analogy is useful if kept computational rather than biological.

Online work favors responsiveness. Consolidation favors careful validation and compression. Mixing the two can make a continually learning agent unstable.

A possible architecture therefore uses two timescales:

### Fast timescale: episodic accumulation

During the working period, the system records candidate experiences but does not immediately modify persistent behavior.

### Slow timescale: consolidation

During scheduled offline periods:

1. cluster related episodes;
2. identify recurring lessons;
3. search for counterexamples;
4. generate task variants;
5. verify candidate priors;
6. train the experience module;
7. regression-test against previous competencies;
8. accept, weaken, or reject the update.

Longer consolidation cycles could compress redundant priors and remove obsolete ones.

This leads to an interesting hierarchy:

- **daily:** absorb high-confidence local experience;
- **weekly:** merge repeated patterns and test broader generalization;
- **monthly:** prune, compress, and re-evaluate old priors;
- **long-term:** build stable professional specialization.

Whether multiple timescales outperform a single consolidation schedule is directly testable.

---

## 16. Relationship to Efficient Thinking

The Efficient Thinking series investigates the relationship between intelligence, search, verification, and computational efficiency.

Experience Priors extend that program along the temporal dimension.

Earlier search-oriented systems ask:

> Given a fixed model, how much can additional search improve performance?

This paper asks:

> After repeatedly performing related searches, can the system learn where future search should begin?

The first question studies **search at inference time**. The second studies **learning the search policy across time**.

In this framing, experience is a compression of historical search.

A successful experience prior converts yesterday's expensive exploration into tomorrow's cheap intuition.

---

## 17. Research Questions

The proposal leads to several concrete questions:

1. Can useful search experience be represented in a module orders of magnitude smaller than the foundation model?
2. At which transformer layers is experience steering most effective?
3. Is a single experience vector sufficient, or is a state-dependent adapter required?
4. Can trajectory preference learning separate search efficiency from general capability fine-tuning?
5. How much experience can be accumulated before interference appears?
6. Can experience modules be composed or merged across agents?
7. Can an experience learned by a weaker model improve the search behavior of a stronger compatible model?
8. How should confidence and expiration be represented?
9. When does textual memory outperform parameterized experience?
10. What is the optimal consolidation frequency?
11. Can experience priors reduce search compute while preserving or increasing solution quality?
12. Does accumulated experience eventually plateau, and what determines that plateau?

---

## 18. A Stronger Long-Term Hypothesis

The immediate proposal is modest: learn a small search prior for a frozen model.

The longer-term implication is more interesting.

If general intelligence and accumulated experience can be separated, an AI organization might not require thirty separately trained foundation models. It could maintain a common intelligence substrate while each persistent agent develops its own compact professional experience.

The resulting agent is no longer merely:

> model + prompt + retrieved memory

but instead:

> **intelligence + explicit memory + accumulated search experience**

This could provide a computational mechanism for something current agents largely lack: **tenure**.

An agent that has performed a job for five years should behave differently from an otherwise identical agent beginning on day one—not because its base IQ changed, and not because it rereads five years of logs, but because previous search has altered its priors.

---

## 19. Falsifiability

The proposal should be considered unsuccessful if experiments show that:

- equivalent textual memory achieves the same efficiency at comparable cost;
- experience adapters improve training tasks but fail to generalize;
- steering damages unrelated reasoning more than it saves compute;
- useful experience requires adapters so large that ordinary fine-tuning is simpler;
- consolidation is unstable under continual updates;
- improvements arise primarily from memorizing answers rather than changing search behavior.

These negative results would still be useful because they would help establish whether procedural experience genuinely requires parameter adaptation or whether retrieval is sufficient.

---

## 20. Initial Experimental Milestone

A minimal first milestone can be intentionally narrow:

> **Demonstrate that a frozen small Qwen model, equipped with a tiny learned experience module trained on verified successful-versus-wasteful trajectories, solves unseen structurally similar tasks with fewer search tokens than the same frozen model using no experience—and compare the result against distilled text memory and QLoRA.**

If this cannot be demonstrated cleanly, there is little reason to scale the idea.

If it can, the next experiment is continual consolidation: repeat several generations of **work → detect → verify → consolidate**, and determine whether efficiency improves monotonically without degrading general capability.

---

## 20b. Results So Far (measured; updated 2026-09-16)

This section is appended as the proof of concept runs. It reports numbers that exist as committed results files,
states what each one does and does not establish, and is rewritten as later results supersede earlier ones. Every
number is reproducible from `experience/` at the commit named beside it; the registered predictions are scored in the
proposal (`docs/efficient-thinking-8-proposal.md` §7) and only summarised here.

### What exists

- **Environment and verifier** (`experience/et8_env.py`): trap-debugging tasks in four families (A_boundary, B_state,
  C_types, D_mixed), twelve bug classes, red herrings and dead paths, an external verifier that decides green with no
  model in the loop. Task set v1: 2,000 tasks, 500 per family, seed 1.
- **Agent harness** (`experience/et8_agent.py`): a fixed action loop (hypothesize, inspect, patch) with a 12-action
  budget, every step logged as a trajectory. Runs on Apple Silicon through mlx-lm.
- **Baselines on two frozen models**, same tasks, same harness, commit `b63af0b`.

### Results at a glance

| run | model | tasks | success | mean actions | mean tokens | wall |
|---|---|---|---|---|---|---|
| base_v1_3b | Qwen2.5-3B-Instruct bf16 | 2,000 | **0.232** | 10.46 | 6,851 | 5.7 h |
| base_v1_7b, seed 1 | Qwen2.5-7B-Instruct bf16 | 2,000 | 0.599 | 6.85 | 4,471 | 8.0 h |
| base_v1_7b, seed 2 | same | 2,000 | 0.593 | 6.85 | 4,472 | 8.0 h |
| **7B pooled** | | 4,000 | **0.596 ± 0.003** | | | |

The 7B floor holds to 0.6 points across two seeds, so it is a floor that a lesson's effect can be measured against
(2026-09-07 decision: 7B is the paper's floor, 3B stays the development loop).

Success by family on 7B, of 500 each: A_boundary **500 / 500 in both seeds**, D_mixed 259 / 260, C_types 234 / 224,
B_state 204 / 202. Family A is saturated on 7B: a cell at 100% cannot show whether a lesson helps, so the generator
needs harder family-A instances before that family carries any comparison. It discriminated fine on 3B (46%).

### What the trajectories say about where search is wasted

From the 3B pass (2,000 episodes, 20,930 steps): 94% of failed episodes touched exactly one code region, and
episodes whose first hypothesis was correct went green 41% of the time against 0% when it never was. On 7B, 45.7% of
all steps are no-op patches. The waste is localisation, not repair: the model can fix a region it has found, and
what it lacks is a prior on which region to open first. That is the shape of thing §4.2 proposes to learn.

### Lessons, and the verifier doing its job

Twelve lessons were distilled from the 3B trajectories by clustering on (family, symptom region) and reading off which
first region led to green (`experience/lessons/v1`). The verification gate's first held-out test (`et8_verify`, 200
held-out tasks, seed 7) **rejected the top-confidence lesson**: in scope it raised actions-to-green (10.0 → 12.25 on
n = 8) and lowered green (4 → 3), while the shuffled-family control was flat (13.0 → 13.0). One lesson, small n,
but the direction is the one §4.3 exists for, and prediction P0 (the verifier rejects ≥ 20% of candidate lessons) is
on track rather than refuted.

### First mechanism signal, and its bound

A gradient-free steering vector built from contrastive first-hypothesis prompts was checked on 16 held-out decision
prompts on 3B (`experience/results/vectors_v0_heldout_check.json`): at mid-depth (layer 18 of 36) it separates
productive from wasted first hypotheses with d′ = 1.13 and 81% accuracy at the midpoint threshold; the early layer
gives 0.79 and the late layer is inverted (−0.78). This is **direction, not evidence**: n = 16, one model, one
build set. It agrees with prediction P3 (mid-depth beats early and late) and has not tested it.

### What is not yet measured

The paper's central claim, P1 — that an injected prior cuts actions-to-green on held-out same-family tasks by ≥ 25%
at equal success — has not been run. Nothing above is that experiment; everything above is what makes it measurable.
Also unrun: P2 (prior versus text memory at a token budget), P4 (steering vectors versus a trained adapter), P5
(out-of-domain regression), P6 (decay of an invalidated lesson), P7 (transfer to 7B), P8 (the chess anchor), P9
(consolidation rounds).

### Next, in order

1. Harder family-A instances in the generator (v2), then re-verify the twelve lessons against 7B trajectories.
2. Steering vectors at mid-depth on 7B with a held-out set of hundreds of prompts, not sixteen.
3. The first P1 measurement: prior versus no prior on same-family held-out tasks, actions-to-green at equal success.

*Results files: `experience/results/*.json`, `experience/lessons/v1/`, trajectories under `experience/traj/`.
Predictions were registered in the proposal before any run.*

### Update 2026-09-16 — the first mechanism runs on the 7B floor, and an application study

**Mechanism A (text prior) is destructive on 7B as first built.** The v0 lessons (1.6 KB, distilled from 40 early
episodes whose own green rate was 37.5%) were injected as a text prior and run on 80 held-out tasks in four disjoint
20-task slices at the 12-action budget (`experience/results/`, commit `b9c6b00`):

| arm | tasks | success | per slice | mean actions |
|---|---|---|---|---|
| baseline 7B | 80 | **62.5%** | 60 · 65 · 70 · 55 | 6.6 |
| + v0 text prior | 80 | **20.0%** | 10 · 40 · 10 · 20 | 10.2 |

The gap exceeds both arms' between-slice spread. The action mix says why: `hypothesize` is identical in both arms
(one per episode), so the prior did not change where the model looked; `noop_patch` more than doubled (231 → 532 of
525 → 819 steps), so the model re-emitted the code it was shown instead of editing it. This separates two deficits
the 09-05 plumbing note had only named: localisation, which a prior can buy, and repair, which is a capability floor.
The v0 artifact teaches a 37.5% policy to a 60% model; the next arm distils lessons from the 7B's own 1,197 green
episodes, which is the first fair test of mechanism A rather than a test of v0's lessons. Because decoding is greedy
a re-run is byte-identical, so independence comes from disjoint task slices, not seeds — recorded as the series'
rule for greedy arms. Token cost per episode is not yet populated by the harness and is the next thing fixed.

**Mechanism C (steering vectors) on 7B now has a real held-out check.** Vectors built from 2,000 decision prompts
(τ+ 1,761 / τ− 239) separate productive from wasted first hypotheses on held-out prompts with d′ 3.15 (layer 9),
4.20 (layer 18) and 4.91 (layer 27), threshold accuracy 1.00 — against the earlier 3B artefact's 0.79. An earlier
run of this arm had pointed the 3B vectors at the 7B model and produced an empty arm that a chained `rc=0` hid; the
held-out check is what catches that, and it is now run for every (model, artefact) pair. The C arm on the four
slices is running. **Completed the same day (all four slices, 80 tasks per arm, paired by slice):**

| arm | success | localised | localised but failed to repair | mean actions | input tokens/ep |
|---|---|---|---|---|---|
| baseline 7B | 62.5% | 88.8% | 21 of 80 | 6.6 | 3,777 |
| + v0 text prior (A) | 20.0% | 47.5% | 22 of 80 | 10.2 | 11,946 |
| + steering, α = 4.0 (C) | 51.2% | 81.2% | 24 of 80 | 7.2 | 4,228 |

Paired per slice, C − baseline is [−1, −5, 0, −3] episodes on success and [+2, −7, −1, 0] on localisation. A first
slice had shown C raising localisation by two episodes and was published as "the clean separation of the two
deficits"; it did not survive the other three and is withdrawn — the bound in its own third paragraph ("needs the
other slices before it is a number") did not protect its headline. What stands: A is destructive on every slice and
on every axis (worst localisation, 3.2× the input tokens, a third of the success). C at this strength is not an
improvement on either axis but is close to baseline and cheap; its vectors separate held-out decision states at
d′ 3.1–4.9, so the probe is good and the intervention is what remains untuned (strength, layer, position). And a
column no mechanism moved: about a quarter of episodes in every arm find the right region and still cannot write
the fix. That is the repair floor, and nothing in this paper's toolbox is aimed at it.

**The fair test of A refutes the content hypothesis.** A second text memory, distilled from the 7B's own 1,197
successful episodes (2,000 episodes at 59.9% success, 2.1 KB) instead of the weak early run, was injected the same
way on the same four slices: **5.0%** success, against v0's 20.0% and the baseline's 62.5%, down in four of four
paired slices, with input tokens per episode rising to 16,212. Two memories of different provenance and quality both
destroy performance and the better one destroys slightly more; the action mix shows the better advice producing
more patches (124 → 180) and more wasted ones (repeat patches 5 → 20) with no more success. So at 7B and this budget
the text prompt is not a weak carrier for experience but an actively harmful one, and the harm does not depend on
the lessons being wrong. One qualification, stated by the run's author: this arm ran the raw distillation without the
verification gate of §4.3, so its honest name is the raw-distilled memory, not the gated one. The gate is the next
thing built, and its test is now sharper than planned: on held-out replay it should *reject* most of these lessons
before they are ever injected — which is prediction P0 doing the work the paper assigned it. Prediction P2, prior
versus text memory at a token budget, is answered in the negative for text memory as the carrier.

  The gate then ran, as pre-registered, on those twelve lessons: each replayed on twenty held-out episodes in its own
  condition and in a control condition, with two arms each, about five hours of GPU. **Eleven of twelve were rejected
  (91.7%, against a predicted floor of 20%), and the one admitted is the only lesson whose content is negative** —
  "avoid the transform region", which had failed 142 of 144 fixations — while every rejected lesson is a positive
  "start at region X" prior. The rejected lessons are not wrong: the lesson "start at the producer" is 309 of 309
  green at confidence 0.992 and is rejected because, replayed, it does nothing (gain 0.0 in five cases). Two more
  were rejected by the control alone: they helped in scope by one episode and helped the control by exactly one
  episode, so a gate that measured only in-scope gain would have admitted both. So the first of the two
  pre-registered sentences is the one the numbers pick: the verification gate of §4.3 catches carrier harm, not
  only wrong advice, and what survives verification is what to *avoid*, not where to *begin*. Twelve lessons is a
  small set and eleven of eleven in one direction is stated as a pattern, not a law. The section on mechanism A
  closes here.

**Mechanism F is now defined** (it had been a name only): an additive logit bias on the action-choice token at the
decision position, built from the same contrastive statistics as C — per region class, the log-ratio of action
frequencies in productive versus wasted episodes, clipped to ±2 logits, zero elsewhere. It is the cheapest carrier.
One consequence follows from the definition and is worth stating before the run: the labels that carry most of the
contrastive signal — `noop_patch`, `repeat_patch`, `invalid` — are the harness's verdicts on an emitted `patch`, not
tokens the model writes, so a bias on the action token cannot reach them. F biases only the four emittable actions
(hypothesize, inspect, patch, run); by construction it can nudge localisation and cannot touch repair. A null on F is
therefore a result, not a failure: the arm that shows a decision-token prior does not reach the repair floor. Folding
the wasted verdicts onto the `patch` choice would be a different experiment — an outcome-shaped penalty that
discourages a good patch with a bad one — and is not run under this name. F runs after C on the same slices.

**Mechanism G, a read-only controller head (pre-registered, not yet run).** The steering result of §20b separates
two things the paper had run together: the frozen model's hidden state at a decision point *carries* the signal
that separates productive from wasted search (d′ 3.1–4.9 on held-out prompts), and *writing* along that direction
into the residual stream lowers success. G uses the same signal without writing. At each decision point the
controller enumerates k candidate actions — the emittable family of §20b crossed with the regions the agent can
currently see, capped by the model's own top-k over the first action token so the model's greedy choice is always
among them — runs each candidate's prefix through the frozen model under teacher forcing, reads the hidden state
at the candidate's last token (layer 27 first, layer 18 as the check), scores it with a linear head trained on the
1,761 productive and 239 wasted decisions of the 2,000-episode run, and takes the argmax. The transformer's
parameters and residual stream are never touched; the model writes every action it takes; the only thing G can get
wrong is the choice. Cost is k short prefix passes per decision with a shared cache, reported as tokens per episode
beside the rate. Two conditions precede the run. The probe must survive conditioning on task: fitted within task,
d′ must stay above 2, or the vector reads difficulty rather than decision quality and G is not run. And the
constraint of §4.2a is declared now: on the same four 80-task slices, paired, success within ε = 2.5 points of the
62.5% base on every slice, localisation within 2.5 points of 88.8%, and the objective is mean actions down by at
least 15% (6.6 → 5.6 or fewer). If success holds and actions do not fall, the representation is decodable but not
actionable through choice, and that is the result. G runs after the verification-gate test and before any second
strength of C, because it asks C's question through a channel that cannot corrupt.

The task-conditioned check was then attempted on the existing 2,000 decision states and could not be computed: the
extractor behind the probe returns the *first* hypothesis of each episode, one decision per task, so no task carries
both labels and a within-task contrast does not exist in that data by construction (demeaning each task's single
state gives exactly zero, which is the tell). That is "not evaluable", not "the vector reads difficulty", and the
pooled d′ of 4.2 and 4.9 stay uninterpretable for choice until the check exists. The data that can answer it is
every hypothesis step of every episode, where later steps in one task carry both labels; extracting it is about an
hour of GPU and precedes any further step of G. One caveat is stated now rather than after: a contrast across steps
of one episode controls for the task but not for the history at each step, so it is the strongest check available
from replay, and the run of G itself is the test of choice.

Extracting every hypothesis step then produced a fact about the agent rather than a probe: **the agent hypothesises
exactly once per episode.** Over 2,000 episodes and 13,696 steps there are 2,000 hypothesise steps, 2,001 inspects,
and 9,668 patch attempts (6,257 of them no-ops); not one episode revisits its localisation. The loop picks a region
at step one and spends the rest of its budget writing a fix for that region. Three consequences follow. The
within-task check cannot be computed from any replay of this data, since each task carries one labelled decision,
so the gate is recorded as *not computable on this harness* rather than passed or failed, and G is run as the test of
choice with that label in every table. Every mechanism in this section — text memory, steering, the logit bias —
acted on a single choice per episode, which bounds what any of them could have done. And the repair floor of about
a quarter of episodes that localise correctly and still fail is now read partly as a loop that never revisits its
localisation rather than as repair competence alone; a re-hypothesis rule in the action loop would change the thing
being measured and belongs to a later section, not to this week's runs. Before G runs, its ceiling is stated from
the same data: splitting the 2,000 episodes by the label of their one hypothesis gives the success rate and the
action count G could reach if it chose perfectly, and the run is read against that bound as well as against ε.

That ceiling was then computed, and it moved the objective before the run. Episodes whose one hypothesis was
productive (1,761) succeed at 68.0% in 6.15 actions; episodes whose one hypothesis was wasted (239) succeed at
**0.0%** in 12.0 actions. A miss at step one is terminal, because the loop never re-hypothesises. So a controller
that chose correctly on every episode would reach 68.0% and 6.15 actions — and the pre-registered cost objective
for G, 5.6 actions or fewer, was below what any controller could reach on this harness. The objective is therefore
restated, dated, and the original printed beside it: G is scored on the fraction of the available headroom it
captures, on actions and on success, with the ceiling recomputed per slice on the slices' own baseline episodes;
the capability constraint is unchanged; and the declared pass is the constraint held on every slice plus at least
half of the pooled actions headroom captured. The target moved because a join that costs minutes showed it was
arithmetically unreachable, and it moved before the run rather than after the result.

Characterising what the head would have to learn then found the dull explanation first, and a three-episode smoke
test refuted it before it cost a run. Splitting the 2,000 episodes by whether the symptom region is the bug region:
when they differ (820 episodes) the agent finds the bug region 820 of 820 times; when they coincide (1,180) it finds
it 941 times, and **all 239 misses are episodes where the bug was where the symptom pointed and the agent went
hunting elsewhere.** The agent's failure is not that it cannot localise; it declines the obvious answer on the easy
cases and never comes back. The first reading of that — a one-line rule, always hypothesise the symptom region,
would reach the ceiling — was pre-registered as the control's prediction and was wrong within the hour: applied
to every episode, as a controller must be, the rule sends the agent to the wrong region on the 820 where symptom
and bug differ, the group the baseline currently solves at 100%, and on three smoke-test tasks it failed all three
where the baseline had passed two. The conclusion was drawn from the 239 alone, and a controller does not get to
apply itself only where it helps. What stands is sharper than the rule. There is no one-line rule, because
"symptom equals bug" is not knowable from the prompt — the bug region is the answer — so the ceiling needs the
state, and what the head must learn is the *conditional*: start at the symptom only when the state says the bug is
there, which is what hidden states might encode and what nothing text-shaped can. G therefore runs against
G-trivial as a *negative* control, with the corrected prediction that the naive rule loses ground, and its number
is the measured cost of a naive prior in the same paired table. The per-slice ceilings put the pooled headroom at
8.0 points of success and 0.69 actions, so the declared pass is the constraint held on every slice and at least
0.34 actions saved pooled.

The negative control then ran on the four slices, prediction registered first. **The naive prior costs 36 points:**
62.5% to 26.2% pooled, down on every slice (12→6, 13→3, 14→7, 11→5 of 20), and actions up from 6.6 to 9.7. The
predicted half held — on the 35 episodes where symptom and bug differ, 77.1% became 0.0%, 27 episodes destroyed.
The unpredicted half is the finding: on the 45 episodes where the symptom region *is* the bug region, forcing the
agent there did not help either, 51.1% to 46.7%. Sending the agent to the correct region does not rescue those
episodes. One reading is that the 239 misses were never a localisation deficit a better first guess could fix,
and the ceiling computed from the region label overstates what choice can buy. A second reading is stated beside
it because the control cannot separate them: the naive arm replaces the model's own hypothesis step with a canned
one, so it changes not only *where* the agent looks but *how* the hypothesis is expressed, and the model's own
reasoning at that step may be what the later patch depends on. G chooses among the model's own candidates, so it
does not carry that confound; the control that separates the two readings is G's candidate machinery with the
head replaced by the symptom rule, and it runs before the head does. Whichever reading survives, the naive prior
is not suboptimal but catastrophic, and no rule that ignores the state reaches the ceiling.

**The task set gives localisation away, and G cannot be tested on it.** The head for G was fitted before the
candidate machinery existed, as a check: on held-out first-decision states it scored 100.0% at both layers, and
100% on the agent's own misses. The controls run before that number was reported are what made it worth chasing:
a permutation control (train labels shuffled, refit) scored 24.0%, chance, so it is not a fitting artefact; but
the same 100% survived a projection to eight dimensions and a training set of 100 examples, which is not the
signature of a learned readout of a hard quantity but of a key being copied. The first decision prompt contains no
program text — the symptom line, four region names, the budget — so the key had to be in the symptom. It is. The
2,000 tasks carry ten distinct symptom strings and each maps to exactly one bug region; a ten-row lookup table
localises at 100%, and the agent's 88% first-hypothesis accuracy is *below* that table on a string it is handed.
What this invalidates: every localisation figure on this task set as a capability figure, the ceiling argument
for G as computed from region labels (a perfect head on this set is a dictionary), and any test of G on it — a
head that chooses well proves nothing about activations when the prompt contains the answer. What it leaves
standing: mechanisms A, C and F, which compared cost at held capability and never needed localisation to be hard;
the repair floor, which is strengthened, since localisation was cheaper than believed and a quarter of episodes
still localise and fail; and the negative control's 36-point loss, which measured what overriding a chooser costs
and is a fact about the loop, not about task difficulty. The remedy is a second task set, pre-registered before
it is generated: each failing test reachable from at least two bug regions, at least one bit of entropy from
symptom to region averaged over tasks, and a lookup table on the symptom scoring no better than 60% on held-out
tasks — all three measured on the generated set before any episode runs, with the action loop and verifier
unchanged so cost stays comparable. G, its probe with the permutation control mandatory, its ceiling and both
of its controls move to that set. Whether the earlier task versions share the degeneracy is being checked; if
they do, the 3B baselines and the transfer prediction P7 inherit it and this paragraph will say so. The chess
anchor of P8, where the uncertainty is real by construction, proceeds beside it as the second field.

**A carrier study: a poet's voice.** This study asks what happens to a capability when experience is pushed into it
by a weight update; it is read under §4.2a beside the text-memory and steering results, not as evidence about search
priors. It does contain a search, and that is how it will be finished: writing a regulated quatrain is a search over
characters under the rhyme table and the tonal pattern, the base model fails that search two times in three, and the
cost of a form-passing poem is the number of attempts it takes. Stated that way the study has the paper's own shape —
cost is attempts per passing poem, capability is the voice as the blind judge scores it, and an experience prior is
anything that lowers the first while the constraint holds the second — and it becomes the cross-check on the
debugging environment rather than a detour from it. The same pipeline was pointed at a different
domain — writing classical Chinese regulated verse in the voice of Tang Yin (1470–1524) — with the three parts
kept in the same shape: an external verifier for FORM (line count, rhyme class against the 平水韻 table, tonal
pattern), a frozen 7B model as the trainee, and a blind judge for VOICE (which of two poems is the real one).
Predictions were written before any run. Measured so far (`experience/tangyin/RESULTS.md`):

- P0 confirmed, narrowly: the base 7B passes the form verifier on 32.0% of quatrain attempts (n = 50 per seed, three
  seeds, binomial SE 6.6 points; the prediction was < 40%).
- P2, form half: a rank-8 LoRA on 63 poems is *destructive* at three epochs (17.3% vs 32.7%, three seeds each, spread
  6 points against a 15-point gap) and *null* at the validation-loss minimum (28.7%). A form-preserving variant that
  mixed the poems with the model's own instruction data was worse still (21.3%) while its validation loss kept
  falling — the training loss and the verifier measure different things, and a LoRA fitted on the loss does not
  buy the form. Its one monotone effect: English-token leakage fell below baseline after one epoch.
- Two instrument errors were caught before becoming results, and both have a general form. A traditional-character
  tone table read the model's simplified characters as unknown and *skipped the rule*, inflating longer forms by
  18 points — every rate now ships with the fraction of positions the instrument could read. And a strict rhyme
  rule scored a famous ancient-style series as broken regulated verse; the tell was that two poets from two
  editions collapsed on one form, which accuses the instrument, not the poets.
- Three seeds reproduced a 32.0% rate exactly while 24 of 50 poems flipped verdict between them, 12 each way: a
  reproduced rate is not stability, and the flips are published beside every rate.
- The voice half (P1, three seeds, each reproducing the last): a pairwise blind judge — a frontier model with no tools, shown one
  real held-out Tang Yin quatrain and one generated one in random order and asked only which is real — separates
  every arm from the poet almost perfectly. Base 7B: 100% of 42 pairs; LoRA at the validation minimum: 97.1% of 34;
  LoRA at three epochs: 88.2% of 34 (SE 5.5 points; 50% would mean indistinguishable). The three-epoch arm, the one
  the form verifier called destructive, is the one that fools the judge most. (Every generated quatrain of the
  right length was shown, passing or failing the form verifier; an earlier draft of this paragraph said only
  form-passing poems were shown, and that was wrong.) The
  memorisation probes did their work before the pairs were scored: four of 26 held-out poems the judge could
  complete from memory (overlap 0.81–1.00, among them the famous 言志) were excluded, two were flagged for
  attribution, and none of ten fabricated canaries was claimed as known — so the instrument does not say "known"
  to please, and the 22 poems it scored are ones it cannot recite. The whole run cost about $0.72 of API spend
  and 172 logged requests. The gap it reports is wide enough that no arm is near the poet, and that is the
  study's answer after one seed: the form can be verified, the LoRA does not buy it, and the voice is not moved by
  63 poems either way. Two further seeds, pairs redrawn and probes re-run each time, gave 100% / 94.1% / 88.2%
  and 100% / 88.2% / 85.3%. Pooled over three seeds the judge is right on 126 of 126 base pairs, 95 of 102 at the
  validation minimum (93.1%, SE 2.5) and 89 of 102 at three epochs (87.3%, SE 3.3). Both LoRA arms move toward the
  poet by this instrument — 2.8 and 3.8 standard errors from the base — and the two LoRA arms do not separate
  from each other (1.4 SE). So the 63 poems did move the voice, by a small and reproducible amount, and the
  arm that moved it most is the arm whose form collapsed. The memorisation probe is itself stochastic — one seed
  excluded two poems where the others excluded four — which is why the exclusion runs fresh under every seed
  rather than once. Total judge spend for the three seeds: about $1.90 and 516 logged requests.
- Under the search reading, the study's cost column now exists, and it says what the debugging field said. Attempts per
  form-passing quatrain, from the pass rates already measured: base 3.12; LoRA at the validation minimum 3.49; the
  mixed variant 4.69; three epochs 6.00. The base is the cheapest search and every carrier makes it dearer, up to
  double. And on the capability column the two independent copy filters, one keyed on the 546 distinct
  seven-character lines of the training split and one on the same lines built separately, order the two LoRA
  checkpoints in opposite directions with both gaps under 1.5 points — so the defensible statement is that the
  checkpoints are indistinguishable on original poems, and the one claim both filters carry is that the three-epoch
  arm's apparent edge lived entirely in its copies (63% against about 92%). A copy filter keyed on whole lines is a
  floor: a paraphrase or a reordered couplet passes it, so "original" is the optimistic column.
- Then the judge was asked *why*, and the answer, once checked, reverses the reading above. Eight three-epoch
  pairs — four the judge had got wrong and four right — were shown again with the same question plus "one
  sentence of reason". It repeated its pick on seven of eight, so the fooled pairs are systematic. Its reasons
  were recognition ("this is Tang Yin's chrysanthemum poem"), a borrowed line from another poet, and a tonal or
  rhyme fault. The first draft of this bullet called the recognitions of generated poems confabulated titles.
  They were not. Crossing every generated poem with the 70-poem training split shows that the three-epoch LoRA
  **regurgitates**: 9 of its 34 quatrains contain a whole line of a training poem or a whole famous poem by
  someone else — three of them are the same Tang Yin chrysanthemum poem verbatim, one is Wang Anshi's New Year
  poem entire — and the judge, told one poem was Tang Yin's, picked the copy. Split by that: on the three-epoch
  arm's 25 original poems the judge is right on 71 of 75 pairs (94.7%, SE 2.6), the same as the validation-minimum
  arm (93.1%), and on its 9 copies it is right on 18 of 27 (66.7%). The validation-minimum arm and the base copy
  nothing (0 of 34, 0 of 42). So the three-epoch arm's apparent edge on voice was memorisation of its own training
  set, its formal collapse and its copying are the same overfit, and the split by verifier verdict says the same
  thing from the other side: the judge is at chance on the three-epoch arm's form-passing poems (13 of 24)
  because those are the copies. The judge's metrical complaints, checked against our verifier independently,
  agree with it in three of three cases; there was no disagreement to adjudicate. What survives: 63 poems moved the
  judge by about six points at either checkpoint, reproducibly, and no further; and the study now carries a fourth
  instrument, a copy check of every generated poem against the training split and a canon of famous poems, run
  before the judge sees anything. The rule "form and voice are separate axes" is withdrawn; the rule that replaces
  it is that a poem is admitted to the voice judge only if it is original, form-scored, and not nameable by the
  judge. The three seeds were then re-run under a naming probe (the held-out poem shown alone with only its era,
  "do you know this poem, and by whom?"), about $2 more: it excluded the same two poems the attribution probe
  had flagged and nothing else — the Wu Zixu poem the judge named inside a pair it did not name cold, so
  recognition there needed the pair's own prior — and the numbers held: on original poems 92.2% and 93.3% for
  the two LoRA checkpoints, 100% of 126 for the base, 70.4% on the copies. Six seeds under two protocols now say
  the same thing, and the instrument that changed the reading was not a probe of the judge but the copy check
  of the sample. Total judge spend: about $4.

### What this changes in the paper's claims

Nothing above confirms P1; the application study strengthens the *reason* for the verifier-in-the-loop design.
In both domains the first effect of an injected prior was to damage the capability it sat on — repair in
debugging, form in verse — while leaving the thing it was meant to move (localisation, voice) unmeasured or
unchanged. A prior that is not gated by an external verifier would have shipped that damage as progress.

*Results files: `experience/results/*.json`, `experience/lessons/`, `experience/tangyin/results/`,
`experience/tangyin/RESULTS.md`. Predictions were registered before any run.*

---

## 21. Conclusion

Modern LLM agents can remember their work without truly becoming experienced at it. External memory preserves episodes and facts, but the model repeatedly pays the cost of retrieving and interpreting that history.

This paper proposes a complementary mechanism: **Experience Priors**, small continually learned modules that encode verified lessons about where reasoning effort should be allocated. The foundation model remains frozen. Experience changes search, not intelligence.

The central loop is:

> **Work → Detect → Distill → Verify → Consolidate → Work Better**

The idea is deliberately testable on small open models. Its value does not depend on increasing benchmark intelligence. A successful result would instead show that a fixed intelligence can become progressively more effective at a persistent job because previous search has been compressed into future intuition.

In the language of Efficient Thinking:

> **Search can compensate for limited intelligence. Experience can make future search cheaper.**

---

## Working Notes / Next Steps

- Formalize the boundary between factual memory, skills, and search priors.
- Select a Qwen checkpoint small enough for rapid local experimentation.
- Instrument generation to expose candidate branches/trajectories rather than only final answers.
- Build an objective verifier environment first.
- Implement text-memory baseline before parameterized experience.
- Start with soft-prefix and single-layer steering experiments.
- Add a small residual adapter only if static vectors are insufficient.
- Compare against LoRA/QLoRA using identical experience trajectories.
- Measure efficiency under a fixed success target, not accuracy alone.
- Test negative experience explicitly: can the prior learn *not* to revisit known dead ends?
- Introduce sequential consolidation rounds and measure interference/forgetting.
- Explore whether agent-specific priors can later be merged or inherited.
