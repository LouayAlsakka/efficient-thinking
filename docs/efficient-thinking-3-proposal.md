# Efficient Thinking III: Efficient Judging
## Compute-optimal allocation between policy and evaluator — a project proposal

**Louay Alsakka** · draft proposal v1 · July 2026

---

## 1. Thesis

Efficient Thinking I–II established that capability factors as **strength = evaluator × search**, and
that across two games, language reasoning, and control, the evaluator — not parameters, not search
budget — is what binds. Both papers treated the evaluator's cost as free: the verifier was an oracle, a
regex, or an API call whose FLOPs were never charged to the budget.

Paper III asks the question that framing hides, and that nobody in the test-time-compute literature
has answered cleanly:

> **Given a fixed total inference budget, how should it be allocated between the policy (generating
> candidate answers) and the evaluator (judging them)?**

The field's default is implicit: spend ~everything on the policy, treat judging as overhead. ET-II's
results imply this default is wrong — the +14.2 verifier gap, the graded-q curve, and the judge
asymmetry all say the marginal FLOP often belongs to the judge. Paper III measures the allocation
frontier directly, with the judge's inference cost **charged to the budget** — the accounting the
best-of-N literature omits.

The deliverable is a family of **iso-compute allocation curves**: for total budget B, accuracy as a
function of the policy/judge split, across policy sizes, judge sizes, and N — and the protocol for
reading them ("at budget B on task T, spend X% on generation and Y% on judging").

---

## 2. Why this is the right Paper III

**It escalates the program rather than repeating it.** Papers I–II *verified* a decomposition the
field broadly believes; their weakness at top venues is novelty, not rigor. Paper III uses the same
diagnostic machinery to answer a question with no published answer: the weak-to-strong generalization
literature (Burns et al. 2023) and the LLM-as-judge literature (Zheng et al. 2023) circle the
policy/judge relationship but neither publishes an allocation frontier under honest cost accounting;
Snell et al. (2024) optimize test-time compute *within* the policy, holding the verifier's cost at
zero. The nearest neighbors leave exactly this gap.

**It is the practitioner's question.** Anyone deploying best-of-N, reranking, or guarded generation
is implicitly choosing a split today with no measurement behind it. (It is also, concretely, the
on-device question: on a fixed-memory local machine, does the next gigabyte go to a bigger responder
or a bigger checker?)

**It inherits the Tier-1 infrastructure.** The GSM8K/MATH sample caches, the graded-verifier
machinery, the GELO calibration, and the judge-agreement protocol from ET-II are the raw material;
roughly half the experiment grid below is post-hoc computation over caches that already exist or are
being generated this weekend.

**It converts the program's character from retrospective to prospective.** §4 registers the
predictions before any run executes. Papers I–II showed the framework fits the data; Paper III stakes
the framework's ability to call shots — and, per the program's standing rule, publishes the misses.

---

## 3. Core concepts and definitions

- **Policy** π_p: a model of size p generating complete candidate answers (temperature-sampled).
- **Judge** J_j: a model of size j selecting among candidates (or scoring them). Modes: pick-best
  (choose index), pairwise (tournament), score-and-argmax. Verifier-free baseline: majority vote
  (j = 0).
- **Total budget** B ≈ p·N·L_gen + j·C_judge(N)·L_judge — policy FLOPs plus judge FLOPs, both charged.
  C_judge(N) is calls per decision (1 for pick-best over a list; O(N log N) or O(N²) for pairwise —
  judging *architecture* is itself a budget knob).
- **Allocation share** s = judge FLOPs / B ∈ [0, 1). The paper's central axis.
- **Judge quality** q(j, T): measured per-item accuracy of judge j on task T against the ground-truth
  verifier — the bridge to ET-II's graded-q curve, now with q *measured as a function of judge FLOPs*
  rather than synthesized.
- **Ceiling**: oracle-best-of-N (pass@N) from the same samples — the s-independent upper envelope every
  allocation curve saturates against.

Ground truth throughout: checkable tasks only (GSM8K, MATH, MBPP/HumanEval), so every judge decision
is auditable against an exact verifier — the same exact-oracle discipline as ET-I/II.

---

## 4. Registered predictions (written before any run)

To be scored explicitly in the paper — hits, misses, and surprises — against the results.

- **P1 (allocation is interior).** For mid-capability policies, the accuracy-optimal split at fixed B
  is neither s ≈ 0 (all policy) nor s ≈ 1: an interior optimum exists and beats both endpoints by a
  margin outside noise.
- **P2 (judge share grows with budget).** The optimal s is increasing in B: at small budgets, spend on
  the policy (coverage binds — Brown et al.'s regime); at large budgets, spend on the judge (selection
  binds — the +14.2 regime). Predicted mechanism: pass@N saturates log-linearly while selection error
  is attackable by judge scale.
- **P3 (judge size beats judge search).** At equal judge FLOPs, a larger judge called once outperforms
  a smaller judge called many times (self-consistency over judgments). This is ET-II's judge-asymmetry
  result promoted to a scaling claim: the evaluator ceiling is bought with *scale*, not *search*,
  because repeated judgment resamples the same systematic error.
- **P4 (q is smooth in judge FLOPs, with a competence knee).** Measured q(j) rises smoothly with judge
  size — no threshold above a task-dependent competence floor, mirroring the graded-q curve — but
  collapses toward the 50% floor below the same kind of competence threshold ET-II found for policies.
- **P5 (asymmetric competence pays).** The optimal configuration at many budgets is a *smaller policy
  with a larger judge* (j > p), not the reverse — because generation needs only coverage while
  selection needs correctness. Corollary, from the ET-II frontier: below the policy's competence
  threshold, no judge rescues (nothing to select), so the allocation optimum snaps to s ≈ 0 —
  the decomposition predicts *where the interior optimum disappears*.
- **P6 (judging architecture matters at fixed s).** At equal judge FLOPs, pick-best-from-list beats
  pairwise tournaments for large N (fewer, richer calls beat many thin comparisons) — the allocation
  question is not only how much to judge but how to spend the judging.

Each prediction is falsifiable by the grid in §5; P2 and P5 are the headline results if they hold, and
publishable surprises if they fail.

---

## 5. Experiment grid

**Models (all Qwen2.5-Instruct, 4-bit MLX):** policies p ∈ {0.5B, 1.5B, 3B, 7B}; judges j ∈ {0 (majority
vote), 1.5B, 3B, 7B, 14B, 32B, 72B}. 256GB per machine holds a 72B judge (~40GB) beside any policy.

**Tasks:** GSM8K (primary; full 1,319 test), MATH-500 (harder; guards against GSM8K ceiling effects),
MBPP or HumanEval subset (code; exact execution verifier — also banks the fifth domain for the series).

**E1 — Sample caches (mostly exists).** N = 32 completions per problem per policy, temp 0.8. The
Tier-1 caches from ET-II's reruns cover 0.5B–4B on GSM8K; add 7B, and MATH/code caches. Every
subsequent experiment is selection over these frozen caches — policies are never re-run.

**E2 — Judge quality curve q(j, T).** Each judge scores a fixed audit set of (candidate, gold) pairs;
q measured against the exact verifier, per task. Deliverable: q as a function of judge FLOPs — the
measured version of ET-II's synthetic graded-q axis. Tests P4.

**E3 — The allocation frontier (the paper's core).** For each (task, p): sweep N ∈ {1,2,4,8,16,32} ×
j ∈ all judges × judging mode ∈ {pick-best, pairwise-swiss} over the frozen cache. Every cell is
(accuracy, policy FLOPs, judge FLOPs). Project onto iso-B curves; read the optimal s per budget.
Tests P1, P2, P5, P6. Note E3 is *cheap*: the expensive generation is done in E1; judging a list of 32
short answers is one long-context call per problem.

**E4 — Judge size vs judge search.** Fix judge FLOPs; compare one call of judge 2j vs two-to-many
calls of judge j (majority of judgments). Tests P3 directly.

**E5 — Below-threshold control.** The 0.5B policy on MATH (well below competence): confirm no judge
produces an interior optimum (s* → 0 because pass@N itself is near floor). The negative control that
tethers Paper III to ET-II's frontier result.

**E6 — GELO placement (continuity).** Place representative (policy, judge, N) systems on the
reasoning GELO ladder so the allocation gains are expressed in the series' common odds units; verify
the allocation curves are the same shape on GELO as on accuracy.

---

## 6. Compute plan (2 × Mac Studio M3 Ultra, 256GB)

Assumes batched generation (~3× M4 Pro single-stream; 1,000+ tok/s aggregate for ≤4B).

| item | size | wall-clock (batched, split across 2 machines) |
|---|---|---|
| E1 gap: 7B GSM8K cache (32 × 1,319) | ~17M tok @ ~90 tok/s eff. | ~1.5 days |
| E1: MATH-500 caches, 4 policies × 32 | ~26M tok | ~1.5 days |
| E1: code cache (200 probs × 4 × 32) | ~10M tok | ~0.5 day |
| E2: q audit, 7 judges × ~2k pairs | ~10M tok, mostly prefill (fast) | ~1 day |
| E3: judging sweeps over caches | ~40–60M judge tok, dominated by 32B/72B rows | **~3–4 days** (the true cost center) |
| E4: size-vs-search on judges | ~8M tok | ~0.5 day |
| E5–E6 | reuses E1–E3 + placement games | ~0.5 day |

**Total: roughly 8–10 days of wall-clock across both machines** — two weekends plus nights, zero API
spend (every judge is local; Kimi/Bedrock appears only as an optional frontier-judge appendix point).
Storage: caches are text, < 2GB total; commit the generators, regenerate the data (series convention).

Sequencing: E1 gaps immediately (one machine on 7B GSM8K, one on MATH); E2 as caches land; E3 runs
whenever machines idle since it is embarrassingly parallel over problems; E4/E5 last week.

---

## 7. Measurement discipline (inherited, non-negotiable)

Full test sets or explicit CIs on every number — no 120-problem headlines; every within-noise delta
re-measured paired before belief (ET-I §4.2 rule); the oracle envelope (pass@N) plotted on every
allocation figure so judge gains are read against the true ceiling; seeds reported for every
stochastic cell; misses of §4's predictions reported with the same prominence as hits.

---

## 8. Deliverables

1. The paper: registered predictions → allocation frontier → scored predictions. Anchor figure: iso-B
   allocation curves with the optimal-s path overlaid (the "spend line").
2. `judging/` in the repo: cache generators, judge harness (all three modes), frontier plotting, the
   q(j) audit — the same runnable-diagnostic convention as Arms A–C.
3. A one-page practitioner protocol: probe → read q(j) and pass@N slope → choose s.

---

## 9. Risks and kill criteria

- **R1: allocation curves are flat** (accuracy insensitive to s over wide ranges). Then the paper's
  headline becomes "judging is nearly free / nearly worthless — the field's implicit s ≈ 0 is
  justified," which is itself a publishable, prediction-scored negative. Kill nothing; reframe.
- **R2: small local judges are too weak to move q** (q(j) ≈ 0.5 through 14B). Detectable in E2 within
  the first day; response: shift tasks easier (GSM8K only) or add one frontier-API judge column as the
  q-ceiling. If q only moves at 72B+, that *is* P4's knee — report it.
- **R3: pick-best long-context judging is unreliable at N = 32** (position bias, lost-in-the-middle).
  Mitigation designed in: randomized candidate order, position-debiased scoring, pairwise mode as the
  fallback architecture — and the bias measurement itself feeds P6.
- **R4: GSM8K ceiling effects at 7B** (pass@32 ≈ 100% leaves no selection headroom). Mitigation: MATH
  is in the grid from day one precisely for this.

---

## 10. Relation to ET-I/II (the series arc)

ET-I: one domain, exact oracle — *which lever binds, and how measurement lies*. ET-II: four domains,
one scale — *the evaluator binds everywhere; external information is the only ceiling-raiser*.
ET-III: *given that the evaluator binds, buy it optimally* — the decomposition, priced. The series
ends where it began: measure first, then spend — now with the spending rule made quantitative.
