# Studio Instructions — ET-IV machine-only queue (keep resources busy)

Priority interrupt rule: the ET-III external-validation runs (MATH subset + Llama check, spec
already delivered) and any Paper II tail item outrank everything below. Run those first if not
done. Everything in this file requires zero human input — it is exactly the work that should
finish before G2 (the poet's yes) fires, so that her yes starts an experiment, not a construction
project.

## 0. Registrations first (minutes)

Commit to docs/: efficient-thinking-4-creative-proposal.md, et4-amendment-1.md (contains P7–P12).
The timestamps must predate every run below. Also commit this file.

## 1. Checkers — build and UNIT-TEST before any experiment touches them

Checker bugs silently corrupt everything downstream; each checker ships with a test file of
known-valid and known-invalid examples and is trusted only after all tests pass.

- **1a. Meter/rhyme checker (poetry):** CMUdict-based syllable count + stress pattern + rhyme
  scheme validator. Test set: 20 lines of known iambic pentameter (Shakespeare sonnets, public
  domain) must pass; 20 deliberately broken lines (syllable added/removed, stress inverted,
  rhyme broken) must fail. Handle OOV words via fallback syllabifier; log OOV rate — if >10% on
  real poems, flag before proceeding.
- **1b. Lyric-fit checker:** syllable-stress alignment of a text line to a given melodic stress
  template. Tests: aligned/misaligned pairs constructed by hand, 10 each.
- **1c. Counterpoint checker (music):** music21-based first-species rule validator (parallel
  fifths/octaves, voice crossing, dissonance treatment, cadence). Tests: 10 textbook-valid
  examples pass, 10 with planted violations fail, each violation named correctly.

Deliverable: checkers/ directory, test files, CI-style run script, all green.

## 2. E1 — formal-validity frontier (poetry/lyrics), machine-only

Policy ladder {0.5B, 1.5B, 3B, 7B, 14B} × tasks {sonnet quatrain (ABAB, iambic pentameter),
villanelle tercet, lyric line to 3 stress templates} × N ∈ {1, 4, 16}, n = 100 prompts per cell,
temperature sampling, per-problem logging from day one (the ET-III lesson).
Metrics per cell: form-valid@1, verifier-selected@N (best-of-N through checker 1a/1b),
oracle = any-valid@N (coverage).
Scores **P1**: search yields large validity gains with the ET-II crossover structure (lift
collapsing as base competence grows). Analysis identical to the ET-II frontier: exact McNemar on
paired cells for any claimed flip.

## 3. E4 — music with a full verifier, machine-only

Counterpoint continuation: given a cantus firmus (10 public-domain examples), generate the
counter-voice as note text; checker 1c scores rule-violation count. Ladder {1.5B, 3B, 7B, 14B} ×
N ∈ {1, 4, 16}, n = 50 per cell. Metrics: violation-free@1, checker-selected@N, violations-per-
line distribution. The cleanest "creative domain with verifier" cell; predicted math-like gains.
If base models cannot emit parseable note text at all, log the format-failure rate and stop —
that is a scoping result, not a failure to hide.

## 4. E6 — serial self-revision vs parallel + selection, machine-only

Same tasks as E1, matched token budgets: (a) draft→self-critique→revise ×k (closed loop, no
external signal); (b) parallel N samples + checker selection. Compare form-validity at equal
total tokens. Scores **P6** (closed revision underperforms parallel + external selection). A
positive result for revision is evidence against the framework and gets reported at full
prominence.

## 5. Goodhart pilot — pressure-ladder calibration (dev-only, persona oracle)

Purpose: cap E3's optimization ladder BEFORE any human session, so no rater ever scores a page of
degenerate text. Judge = local 7B with a fixed persona rubric (dev oracle, clearly marked — no
result from this pilot is a paper claim). Iterative line-edit beam search optimizing judge score;
checkpoints at pressure {N=4, 8, 16, 32, 64 samples consumed}; log judge score + form-validity +
a degeneracy heuristic (repetition rate, vocabulary collapse). Deliverable: the pressure ceiling
at which outputs stay form-valid and non-degenerate — that ceiling becomes E3's human-session
ladder. Also pilots **P12**'s two arms mechanically (threshold-gated vs scalar selection) to
verify both pipelines run; their persona-scored comparison is scaffolding only.

## 6. Human-session harness (build, don't run)

Blind A/B and ranking forms (randomized order, no model identity), session logging schema
(choices, timestamps, optional per-dimension scores and rationale per the amendment format),
Pareto pruning of dominated candidates before any human sees them, the week-later re-rate subset
sampler, and q/self-consistency computation scripts with unit tests on synthetic data.
Definition of done: a full dry-run session executed end-to-end with the persona oracle standing
in for the rater, producing a q number from fake data through the real pipeline.

## Standing rules and guardrails

- Per-problem logging everywhere, from the first cell (no re-runs for p-values later).
- Nothing scored by the persona oracle is ever a paper claim; dev-only artifacts are labeled so
  in their filenames (\_dev).
- ET-V stays parked (G3 rule) no matter how tempting; P3 (judge search vs size) waits for the
  ET-IV judge harness proper, not this queue.
- Every experiment commits: results JSON + the exact script + a one-line registered-prediction
  reference it scores. No text before numbers.
- Report format on completion: per item, "done + where the JSON lives + which prediction it
  scores + any surprise."
