# Efficient Thinking X: Machine-Native Representation — concept

> **STATE 2026-09-21: CONCEPT.** Drafted by the author 2026-09-21 and registered by this commit; nothing measured. Ruled the same day: run as a ladder X-a … X-d, one rung per paper, written backward (`et10-series-plan.md`); X-a first, on the chess and debugging harnesses, with a length-matched control.

## Working title

**Efficient Thinking 10: Beyond Language --- Minimum Sufficient
Representations for Machine Reasoning**

Alternative title:

**ET10: Representation Efficiency --- How Much of Language Is Necessary
for Reasoning?**

------------------------------------------------------------------------

## 1. Central question

Natural language is optimized for communication between humans. It is
not obvious that English, Chinese, Arabic, or any other human language
is an efficient representation for machine reasoning.

Modern language models normally follow a pipeline resembling:

\[ \text{Human intent} \rightarrow
\text{language tokens} \rightarrow
\text{internal representations} \rightarrow
\text{reasoning} \rightarrow
\text{language tokens}. \]

ET10 asks whether there exists a **machine-native representation** (Z)
that retains everything necessary for correct reasoning while requiring
materially less computation:

\[ X_{\text{human}} \xrightarrow{E} Z
\xrightarrow{R} Z' \xrightarrow{D}
Y_{\text{human}}. \]

The central hypothesis is:

> **Natural language may be an expensive human I/O protocol rather than
> the computationally efficient native representation of machine
> reasoning.**

The goal is not merely prompt compression. The deeper question is:

> **What is the minimum sufficient representation for machine
> reasoning?**

------------------------------------------------------------------------

## 2. Core hypothesis

Let (X) be the original representation of a problem, (Z) a
machine-native representation, (Q) externally verified task quality, and
(C) total computation.

ET10 seeks representations satisfying:

\[ Q_Z(C) > Q_L(C) \]

over a meaningful compute range, where (L) denotes ordinary
natural-language reasoning.

Equivalently, ET10 looks for:

\[ C_Z(Q^*) < C_L(Q^*) \]

for the same externally verified capability.

A useful optimization statement is:

\[ Z^\* = \arg\min_Z C(Z) \quad
\text{subject to} \quad Q(Z)
\ge Q_0-\epsilon. \]

The representation should remove representational overhead without
removing information needed for the task.

------------------------------------------------------------------------

## 3. Task information, not sentence reconstruction

ET10 should distinguish two very different notions of losslessness.

### Linguistic losslessness

The system retains enough information to reconstruct the exact original
wording.

This is usually unnecessary for reasoning.

### Task-information sufficiency

The system retains everything required to answer every relevant
externally verifiable question about the underlying problem or world.

For ET10, the important condition is approximately:

\[ P(Y\mid X) \approx P(Y\mid Z) \]

for the defined task family.

For example:

> "John gave Mary the red box after she returned the blue key."

does not have to be reconstructed word-for-word. The representation must
preserve propositions such as:

\[ return(Mary,key_{blue}) \]

\[ before(return,give) \]

\[ give(John,Mary,box_{red}). \]

The decoder may generate different English wording while preserving the
same verifiable information.

------------------------------------------------------------------------

## 4. The fundamental chicken-and-egg problem

If ET10 simultaneously invents a representation (Z) and trains a new
reasoner to operate in (Z), a negative result is uninterpretable.

Failure could mean:

\[ \text{bad representation} \]

or

\[ \text{bad encoder} \]

or

\[ \text{reasoner failed to learn the representation} \]

or

\[ \text{bad decoder} \]

or some combination.

Likewise, success does not by itself prove that (Z) is intrinsically
efficient; perhaps the training procedure simply favored that
representation.

Therefore ET10 should **not begin with a jointly learned end-to-end
latent language**.

Instead, it should decompose the problem into independently falsifiable
stages:

\[ \boxed{ \text{Existence} \rightarrow \text{Efficiency} \rightarrow \text{Discoverability} \rightarrow \text{Translatability} } \]

Each stage should have its own oracle and should be capable of
succeeding or failing independently.

------------------------------------------------------------------------

# Part I --- Existence

## 5. Begin with a world whose complete state is known

Construct a synthetic environment with an exact mathematical state:

\[ S=(x,y,v_x,v_y,m,goal,\ldots). \]

The simulator itself defines (S). Therefore no learned representation is
needed initially.

Generate a natural-language rendering:

\[ S \xrightarrow{L} X_{\text{English}}. \]

Now compare reasoning from:

\[ S\rightarrow R_S\rightarrow Y \]

against:

\[ English(S)\rightarrow R_L\rightarrow Y. \]

Use identical or carefully controlled model architectures, task
distributions, training examples, training compute, and external
verifiers.

If direct reasoning over (S) reaches the same capability for less
inference compute, ET10 establishes its first result:

> **Natural-language representation imposes measurable computational
> overhead relative to a nonlinguistic sufficient representation.**

This result does not require discovering a new representation.

------------------------------------------------------------------------

## 6. Representation is the independent variable

A pretrained English LLM has received vastly more language training than
a newly trained (Z)-reasoner. Comparing them directly would confound
representation with training history.

The clean experiment therefore trains models from scratch under
controlled conditions.

For example:

\[ M_L: English(S) \]

\[ M_S: S \]

with:

-   identical architecture class,
-   identical parameter count,
-   identical underlying worlds,
-   identical task distribution,
-   identical answers,
-   identical number of training examples,
-   matched training compute,
-   matched inference budgets.

The intended independent variable is:

\[ \boxed{ \text{representation} } \]

not model scale or training exposure.

------------------------------------------------------------------------

# Part II --- Efficiency

## 7. Same information can have different computational cost

A central ET10 proposition is that information content and computational
representation efficiency are not identical.

Two representations may contain the same task information:

\[ I_{\text{task}}(Z_1)=I_{\text{task}}(Z_2) \]

while requiring very different computation:

\[ C(Z_1)\ll C(Z_2). \]

This distinction may become one of ET10's main contributions.

### Important control: deliberately bad representation

Construct a reversible random encoding:

\[ S\rightarrow Z_{\text{random}}. \]

It should preserve exactly the same underlying information while
destroying useful structure.

Compare at least:

1.  Natural language.
2.  Original mathematical state (S).
3.  Hand-designed efficient symbolic representation.
4.  Random reversible representation.
5.  Eventually, learned machine-native representation (Z^\*).

If all contain equivalent task information but differ substantially in
learnability or reasoning cost, ET10 directly demonstrates
**representation efficiency**.

------------------------------------------------------------------------

## 8. Measure the full quality--compute frontier

Do not evaluate only token count.

Measure total cost including:

-   encoder compute,
-   reasoning compute,
-   decoder compute,
-   FLOPs or an appropriate hardware-independent proxy,
-   memory / KV-cache use where applicable,
-   latency,
-   representation size,
-   externally verified task success.

The complete cost is:

\[ C_{\text{total}} = C_E+C_R+C_D. \]

The central curve is:

\[ Q(C). \]

Compare:

\[ Q_L(C) \]

with

\[ Q_Z(C). \]

ET10 succeeds on the efficiency claim if the machine-native
representation improves the measured capability--compute frontier over a
meaningful range.

------------------------------------------------------------------------

# Part III --- Discoverability

## 9. Can the efficient representation be found automatically?

Once existence has been demonstrated using known state (S), introduce a
constrained representation:

\[ S\rightarrow E\rightarrow Z. \]

Do not prescribe what (Z) must look like.

It may become:

-   continuous vectors,
-   discrete codes,
-   sparse structures,
-   graphs,
-   learned symbols,
-   or another representation.

Optimize approximately:

\[ \min_Z C(Z) \]

subject to:

\[ Q(Z)\ge Q_0-\epsilon. \]

A more complete objective could include:

\[ \min ; C(E)+C(R)+C(D)+\lambda R(Z) \]

subject to the capability constraint, where (R(Z)) penalizes
representational capacity.

------------------------------------------------------------------------

## 10. Compression sweep

Progressively constrain the representation:

\[ Z_{512} \rightarrow Z_{256} \rightarrow Z_{128}
\rightarrow Z_{64} \rightarrow Z_{32}
\rightarrow Z_{16} \rightarrow\cdots \]

Measure capability and computation at each point.

Expected qualitative behavior:

  Representation capacity         Task information   Reasoning cost
  ------------------------- ---------------------- ----------------
  Full state (S)                              100%         baseline
  (Z_{128})                                \~100%            lower
  (Z_{64})                                 \~100%            lower
  (Z_{32})                              near 100%       much lower
  (Z_{16})                   capability collapses           lowest

The interesting region is the **knee of the frontier**:

\[ \boxed{ \text{minimum representation capacity} \quad \text{s.t.} \quad Q\ge Q_0-\epsilon } \]

This gives an empirical estimate of a minimum sufficient machine
representation for the defined task family.

------------------------------------------------------------------------

# Part IV --- Translation

## 11. Human language should be an interface, not necessarily the reasoning substrate

Once (Z) exists, build separate codecs:

\[ English\xrightarrow{E_L}Z \]

and

\[ Z\xrightarrow{D_L}English. \]

The critical design principle is:

> **Do not discover (Z) primarily by optimizing English
> reconstruction.**

Otherwise (Z) may simply become compressed English.

Instead, ground (Z) in the underlying world or task state.

For a state (S), different observations should converge toward the same
task representation:

\[ E_{\rm English}(X_E) \approx
E_{\rm Chinese}(X_C) \approx E_{\rm JSON}(X_J)
\approx E_{\rm vision}(X_V) \approx Z(S). \]

Then language becomes an I/O codec around machine-native computation.

------------------------------------------------------------------------

## 12. Evaluate encoder, reasoner, and decoder independently

Because the synthetic world's true state (S) is known, translation can
be evaluated independently of reasoning.

### Encoder

\[ English\rightarrow E\rightarrow\hat S. \]

Compare:

\[ \hat S\stackrel{?}{=}S. \]

### Reasoner

Feed the known representation directly:

\[ S\text{ or }Z\rightarrow R\rightarrow Y. \]

This tests reasoning without translation error.

### Decoder

\[
S\text{ or }Z\rightarrow D\rightarrow English.
\]

Evaluate whether all externally verifiable propositions remain
recoverable rather than requiring exact wording.

Thus ET10 obtains separate measurements:

\[ Q_E,\quad Q_R,\quad Q_D \]

rather than only an opaque end-to-end score.

This is the main defense against the chicken-and-egg problem.

------------------------------------------------------------------------

# Part V --- Independence from Human Language

## 13. The hardest objection

If the representation is extracted from a pretrained language model, a
critic can argue that its apparent language independence is still
inherited from human-language training.

Therefore an ordinary pretrained LLM cannot by itself establish that the
representation is independent of language.

ET10 needs a stronger control.

------------------------------------------------------------------------

## 14. Train a system that has never seen human language

Create a synthetic world producing trajectories:

\[ S_t,A_t,S_{t+1}. \]

Train an agent exclusively from:

-   numerical states,
-   sensor observations,
-   images,
-   actions,
-   rewards,
-   or arbitrary machine symbols.

It must receive **no English, Chinese, text captions, human semantic
labels, or other natural language**.

Let the compact representation learned by this system be:

\[ Z_{\text{grounded}}. \]

Separately obtain:

\[ Z_{\text{language}} \]

from a language-facing system interacting with the same underlying task
structure.

Then test whether a simple transformation (M) can align them:

\[ Z_{\text{grounded}} \approx
MZ_{\text{language}}. \]

Controls should include random representations, random projections,
dimensionality-matched baselines, and shuffled correspondences.

If independently trained systems converge toward structurally related
representations despite one never encountering human language, that is
much stronger evidence that the structure reflects the underlying
problem rather than English.

It would not prove a unique universal "language of thought," but it
would establish a substantially more defensible result.

------------------------------------------------------------------------

## 15. Stronger multi-representation convergence experiment

Train several independent systems against the same underlying world:

\[ A_1: English \]

\[ A_2: Chinese \]

\[ A_3: Vision \]

\[ A_4: Numeric states \]

\[ A_5: Random arbitrary symbols \]

\[ A_6: Sensor/action trajectories only. \]

Force each toward a capability-preserving compression frontier:

\[ Z_1,Z_2,Z_3,Z_4,Z_5,Z_6. \]

Then ask whether the resulting representations become structurally
equivalent up to relatively simple transformations.

The random-symbol condition is especially important. A new arbitrary
encoding should be generated independently so that it has no semantic
relationship to human language.

If convergence appears across these conditions, ET10 can investigate
whether efficient representations are constrained by the **structure of
the problem itself**.

------------------------------------------------------------------------

# Part VI --- Avoiding Architectural Triviality

## 16. ET10 must show more than "transformers use vectors"

The existence of internal continuous vectors in neural networks is not
itself an ET10 result.

ET10 must distinguish its claim from ordinary transformer representation
learning.

Relevant controls include:

-   ordinary hidden states,
-   random hidden-state projections,
-   PCA or other dimensionality reduction,
-   random representations of equal dimensionality,
-   learned token compression,
-   hand-designed symbolic states,
-   alternative architectures where practical,
-   equal-capacity learned bottlenecks.

The claim should concern a measurable property:

> A representation can be encoded, used for computation, decoded, and
> shown to improve the quality--compute frontier while preserving
> externally verified task information.

------------------------------------------------------------------------

# Part VII --- Relation to Existing Latent Reasoning

## 17. ET10 is not simply latent chain-of-thought

Existing latent-reasoning work already investigates computation in
continuous states instead of verbalizing every intermediate step.
Therefore ET10 should not claim novelty merely from replacing textual
intermediate reasoning with latent vectors.

ET10's stronger question is:

> **Can we experimentally identify and discover minimum sufficient
> representations for reasoning, separate representation efficiency from
> model training, translate into and out of those representations, and
> determine whether their structure exists independently of human
> language?**

The contribution is therefore about **representation efficiency and
sufficiency**, not merely hidden-state reasoning.

------------------------------------------------------------------------

# Part VIII --- Falsifiable Research Ladder

## 18. Proposed ET10 claims

ET10 should be structured so that failure at one stage does not
invalidate the previous stages.

### Claim A --- Existence

Representation affects reasoning cost even when task information, model
capacity, training, and evaluation are controlled.

\[ R_1\neq R_2 \Rightarrow C_1\neq C_2
\quad \text{at equal }Q. \]

### Claim B --- Efficiency

A nonlinguistic representation can preserve task information while
reducing computation:

\[
I_{\text{task}}(Z)\approx I_{\text{task}}(S)
\]

and

\[ C(Z)<C(L). \]

### Claim C --- Discoverability

A learning procedure can discover a representation near the empirical
capability-preserving compression frontier:

\[ Z^\*=\arg\min_Z C(Z) \quad
\text{s.t.} \quad Q(Z)\ge Q_0-\epsilon.
\]

### Claim D --- Translatability

Human language can be mapped into and out of (Z^\*) while preserving
task information, with translation overhead included in total cost.

### Claim E --- Language independence

Comparable efficient representational structure emerges in systems that
were never exposed to human language.

These claims form the ladder:

\[ \boxed{ \text{Existence} \rightarrow \text{Efficiency} \rightarrow \text{Discoverability} \rightarrow \text{Translatability} \rightarrow \text{Language Independence} } \]

------------------------------------------------------------------------

# Part IX --- What Different Outcomes Would Mean

## 19. ET10 remains informative even if the full hypothesis fails

### If existence succeeds but discovery fails

Efficient machine representations exist, but automatic discovery remains
unsolved.

### If discovery succeeds but translation is expensive

The representation may be useful for persistent machine-to-machine
reasoning but not for one-shot human interaction.

### If translation works but end-to-end computation is not reduced

The representation is compact but not computationally efficient.

### If language-naive systems do not converge with language-derived systems

The strongest universality claim fails, but representation efficiency
may still hold.

### If random reversible representations perform as efficiently as learned structured ones

The hypothesis that representation structure itself matters would be
weakened.

### If natural language remains on or above the best frontier

That would be an important negative result: human language may already
provide computational structure that compensates for its apparent
redundancy.

The experiment should therefore be valuable regardless of which rung
fails.

------------------------------------------------------------------------

# Part X --- Connection to Efficient Thinking

## 20. Place within the series

A useful conceptual progression is:

\[ \text{ET1: Search} \]

How much can additional search substitute for greater model
intelligence?

\[ \text{ET8: Experience} \]

Can verified historical computation improve the return on future search
while intelligence remains frozen?

\[ \text{ET10: Representation} \]

How much computation is wasted because information is represented in a
form optimized for human communication rather than machine reasoning?

ET10 therefore studies another source of efficiency:

\[ \boxed{ \text{Intelligence} + \text{Search} + \text{Experience} + \text{Representation} } \]

The common Efficient Thinking objective remains:

\[ \boxed{ \text{maximize useful capability per unit computation} } \]

------------------------------------------------------------------------

# 21. Candidate central statement

A concise ET10 thesis could be:

> **Equivalent information can have unequal computational cost. ET10
> asks whether machine reasoning admits compact, task-sufficient
> representations that preserve externally verified capability while
> requiring less computation than natural language, whether such
> representations can be discovered automatically, whether humans can
> translate into and out of them, and whether their structure emerges
> independently of human-language training.**

------------------------------------------------------------------------

# 22. The deepest version of the question

ET10 begins with prompt compression but ultimately asks something more
fundamental:

\[ \boxed{ \textbf{What is the cheapest representation in which a machine can correctly reason about a world?} } \]

If efficient representations discovered independently from English,
Chinese, vision, random symbols, and language-naive interaction converge
toward related structures, that would suggest that part of their
organization is imposed not by human language but by the computational
structure of the underlying problem.

That is the strongest version of the ET10 hypothesis and should be
treated as an empirical question rather than assumed in advance.
