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
