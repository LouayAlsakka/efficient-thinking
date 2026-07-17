# Efficient Thinking IV: Search Where Taste Is the Evaluator — proposal

*Status: proposal, pre-registration. Slots as Paper IV: Efficient Thinking III ("Efficient
Judging") keeps its original number and registration trail, and its results — judge competence,
relative-gap dependence, pairwise protocol — are this paper's evaluator foundation.*

## Origin and question

Papers I–II established, in domains with a checkable success signal, that capability decomposes as
evaluator × search: search extracts what the policy's distribution contains, gains are bounded by
coverage and by evaluator quality, and only external information raises the ceiling. The obvious
frontier is the domain family where no verifier exists at all — poetry, lyric writing, music — where
the evaluator must be a judge, and the ultimate ground truth is a human. A non-technical reader of
Papers I–II described the regime precisely from her own practice: the model proposes many candidate
words for a line of her poem and cannot select among them; her reaction is the external evaluator;
and her session-level feedback does not persist — the model reverts to convention. This paper makes
that description quantitative.

The claim under test is NOT "a good evaluator guarantees quality." The framework forbids that
sentence three ways, and the paper measures all three bounds:

1. **Coverage bound** (Paper II, Prop. 2): no selector exceeds what the policy can produce.
2. **Fidelity bound**: aesthetic gains scale with judge–human agreement q, which must be measured,
   not assumed.
3. **Goodhart bound**: optimizing a proxy judge diverges from human quality past a measurable
   pressure point (cf. reward-model over-optimization, Gao et al. 2023). Search does not converge on
   quality; it converges on what the judge rewards.

## Design principle: split the evaluator

Every creative artifact here is scored by a two-component evaluator with different epistemic status:

- **Form (verifier, checkable):** meter and rhyme via pronunciation dictionary (CMUdict + scansion);
  syllable-stress alignment to a target melody for lyrics; for music, species-counterpoint and
  voice-leading rules checked programmatically (music21). This component is math-like: exact,
  cheap, scalable.
- **Taste (judge, graded):** an LLM judge scoring aesthetic quality, with q defined as blind
  agreement with human ranking. Humans: the poet collaborator as primary rater, plus ≥2 additional
  raters for inter-rater reliability on a subset (report Krippendorff's α; if human–human agreement
  is low, q is measured against the pooled ranking and the ceiling on any judge is α itself — a
  finding, not a nuisance).

The framework's prediction is that search behaves like Paper II's math results on the form component
and delivers on the taste component only in proportion to q.

## Experiments

**E1 — Formal-validity frontier (poetry/lyrics).** Policy ladder (Qwen 0.5B→14B local; 72B API spot
checks) × best-of-N with the form verifier, on fixed-form tasks (sonnet, villanelle stanza, lyric
line to a given stress pattern). Metrics: form-valid@1 vs verifier-selected@N; crossover structure
vs model size. This is Paper II's frontier with a formal-constraint verifier instead of an answer key.

**E2 — Taste selection and q (the friend's loop, formalized).** Word/line-replacement task: N
candidates from the policy, selected by (a) random, (b) majority-style convention baseline,
(c) LLM judge, (d) oracle = human pick. Blind human ranking of selected outputs. Yields q
(judge–human agreement), the human-rated lift of judge selection over no-selection, and the
consensus-vs-oracle gap in an aesthetic domain.

**E3 — The Goodhart curve (anchor experiment).** Iterative expand-and-trim (line-level edits,
beam kept by judge score) optimizing the judge, with checkpoints at increasing optimization pressure
(N samples consumed / iterations). Humans blind-rate checkpoints. Plot judge score (expected
monotone) against human rating (predicted rise-then-fall). Secondary axis: weaker vs stronger judge;
predicted earlier turn for weaker judges.

**E4 — Music with a full verifier.** Chorale/counterpoint continuation where the rule-checker is the
complete form evaluator: policy ladder × best-of-N-with-checker on rule-violation rate. The cleanest
"creative domain with verifier" cell; predicted to reproduce math-like search gains. (Aesthetic
judging of rule-valid outputs optional, small-n.)

**E5 — Persistence of taste (her complaint, scored).** Three arms on the same editing task:
(a) zero-shot judge; (b) in-context judge conditioned on K examples of the poet's stated preferences;
(c) small LoRA fine-tuned on the same K preferences (MLX, 0.5–1.5B judge). Measure q per arm, within
session and on held-out sessions. Tests whether taste feedback raises q transiently (in-context),
durably (LoRA), or neither — the personal-evaluator-distillation question.

**E6 — Serial vs parallel in creative revision.** Draft→self-critique→revise loops (closed, no
external signal) vs parallel sampling + external judge selection at matched token budget. The
framework predicts the closed revision loop underperforms parallel + selection (self-critique is a
closed system); a positive result for revision would be evidence against the framework and reported
as such.

## Registered predictions

- **P1.** The form component behaves like math: verifier-based search yields large validity gains,
  with the Paper II crossover structure (lift collapsing as base competence grows).
- **P2.** Human-rated aesthetic gain from judge selection scales smoothly with measured q (graded-q
  analog); no q, no gain.
- **P3.** Goodhart: human-rated quality is non-monotone in optimization pressure while judge score is
  monotone; the turn point arrives earlier for weaker judges.
- **P4.** In-context taste feedback raises q within-session but not on held-out sessions; the LoRA
  arm's held-out q exceeds the in-context arm's or the convention-reversion effect is confirmed as a
  training-data problem, not a context problem. (Either branch scores; the prediction is that
  persistence requires weight changes.)
- **P5** (imported from the ET-III frontier-judge result): judge selection helps in proportion to the
  judge–policy competence gap; where the policy's convention prior already matches the judge's taste
  prior, selection adds ~nothing.
- **P6.** Closed self-revision at matched budget underperforms parallel sampling + external
  selection (E6).

## Infrastructure

Local: Qwen ladder on the Mac minis (MLX), CMUdict/scansion checker, music21 rule-checker, judge
harness reusing the ET-III pairwise machinery (pairwise mode preferred — the list-drowning result
transfers). API: 72B/Kimi spot checks and, for E3, the stronger-judge arm. Human eval: blind A/B and
ranking forms; the poet collaborator credited as a rater and, if she agrees, named; all rating
sessions logged for the q computations. Costs: dominated by human-rating time, not compute — design
all human batches ≤30 min/session.

## Honest scope

Aesthetic ground truth at n = a-few-raters is a case study, not a population claim; q and α are
reported with their small-n uncertainty, and every human-rated result is scoped to these raters.
The paper's contribution is the measurement design and the bounds (coverage, fidelity, Goodhart) in
a domain the field mostly discusses qualitatively — not a claim about Poetry in general.
