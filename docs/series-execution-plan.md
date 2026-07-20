# Efficient Thinking — Series Execution Plan (III, IV, V, VI + II tail)

Working doc for Louay and both agents. Numbering settled: III = Efficient Judging (running),
IV = Search Where Taste Is the Evaluator (proposal ready), V = The Exchange Rate of Feedback
(parked). Paper II is shipped content; its tail items ride along as Track 0.

## Standing rules (unchanged, learned the hard way)

- Canonical in the repo is the only editing surface; fresh pull before any merge; downloaded copies
  are read-only.
- No text before numbers; every registered prediction scores in the paper's §10-equivalent, hit or
  miss, overshoots flagged like undershoots.
- "Done" claims come with their verification command and its expected output; no pending external
  patch may exist when "done" is declared. Patches handed over are applied same-session or treated
  as blocking.
- Proposals commit to the repo BEFORE their experiments run — the timestamp is the registration.
  First action of this plan: commit efficient-thinking-4-creative-proposal.md and
  efficient-thinking-5-concept.md as-is.

## Track 0 — Paper II tail (background, no machine contention)

- 72B MATH row when the run lands → §4 table + one §10 line (v2 update).
- Split-half mode test (mode of samples 1–16 predicts majority of 17–32, per model) → scores
  Prop 2's out-of-sample form; replaces the corrected vacuous check. ~10 lines over existing caches.
- GELO validation ledger numbers (gates ×2, held-out calibration, transitivity) → Appendix A table,
  one §2 sentence, §10 rows for the three registered predictions.
- Louay only: voice pass → arXiv v1. Everything above that lands later is arXiv v2 scoring updates.

## Track A — ET-III: Efficient Judging (finish and write)

State: E3 harness done; first grid done (1.5B judge below floor, drowns at N=16); frontier-judge
extension done (even 72B judges lose to free majority on competent policies); pairwise mode
implemented and running.

1. **Pairwise result lands → Gate G1.** Decides the R3 question: was the 1.5B judge incompetent or
   list-drowned? Score in the paper's ledger either way. Also log chosen-candidate positions from
   existing pick-best runs (free) to quantify position bias directly.
2. **Score the registered knee predictions** against the full grid, including the outside
   collaborator's three (registered in-conversation before the 3B–14B judges ran): (i) absolute
   knee at ~7B — on current evidence heading for a MISS; (ii) knee is relative (judge−policy gap),
   3B judge helps weak policies only — heading for a HIT; (iii) N-degradation shrinks with judge
   size — score from grid. Misses narrated in full; the relative-competence reframe is likely the
   paper's headline.
3. **Complete the allocation frontier**: q(j) curve, FLOP-accounted policy×judge cells, P1–P6 from
   the committed proposal scored one by one.
4. **Write.** Same architecture as Paper II final format: one message, results-at-a-glance, anchors
   (candidate anchor: the relative-competence law — judge value as a function of judge−policy gap),
   scored-predictions section, honest scope. Draft from repo canonicals; Louay's voice pass before
   any external eyes.

Definition of done: every proposal prediction + the three in-conversation predictions scored; paper
in repo with derived PDF matching; greps for each scored prediction's verdict string return 1.

## Track B — ET-IV: Creative (build order)

0. **Commit the proposal** (registration timestamp) before anything runs.
1. **Checkers first — machine-only, no human time**: CMUdict/scansion meter+rhyme verifier;
   music21 counterpoint/voice-leading rule checker; syllable-stress alignment for lyrics. Unit-test
   each against known-valid/invalid examples before trusting any experiment built on them.
2. **E1 (formal-validity frontier) + E4 (music with full verifier)** on whatever machine Track A
   frees first. These are Paper II's frontier machinery pointed at new verifiers — reuse the cache
   + selector harness. Registered P1 scores here.
3. **Gate G2 — human-eval consent.** Confirm the poet collaborator: willing to rate? Named or
   anonymous? Session budget (≤30 min each)? If yes → build the blind A/B + ranking forms, logging
   for q and self-consistency. If no or limited → fallback: LLM-persona oracle for development,
   recruit 2–3 raters for a smaller validation slice; scope claims accordingly.
4. **E2 (q measurement) → E3 (Goodhart curve, the anchor) → E5 (persistence: in-context vs LoRA).**
   E3 checkpointing designed before the run: fixed pressure ladder, checkpoints frozen, human
   ratings blind to checkpoint order. E5's held-out-session design is the seed for Paper V — build
   its logging with that reuse in mind.
5. **E6 (serial self-revision vs parallel+selection)** — cheap, machine-only, matched token budgets;
   can interleave anytime.

Definition of done per experiment: results JSON committed, prediction scored, one paragraph of
canonical text written from the numbers.

## Track C — ET-V: Feedback exchange rate (parked)

- Now: commit the concept note (timestamp). Nothing else.
- **Gate G3**: IV-E5 result exists → unpark. First build is V-E1's simulated-oracle harness (frozen
  persona rubric as the user), which shares IV's judging machinery.
- V-E3's human arm reuses IV's rating infrastructure and consent; plan those forms once.

## Track D — ET-VI: Label-Fidelity Instrument (parked, idle-time override)

- Registered: `docs/et6-label-fidelity-spec.md` (arms E-A/E-B, predictions F1–F5). Commit is the timestamp.
- Claim: a value net's ceiling is its label fidelity; total error |V_net − V*| decomposes as
  fit |V_net − V^π| ⊕ label-bias |V^π − V*|, and the plateau is where fit drops below bias (F1).
- Priority: idle machine time only, **behind** Track A's ET-III external-validation runs and Track B's
  ET-IV machine queue; runs solely at Louay's explicit override (ET-VI is otherwise parked).
- Build order: Connect-4 arm first (exact solver validates the instrument — F1–F3, F5), chess arm
  second (Stockfish-WDL proxy oracle; F4 back-predicts the supervised ~2000 ceiling from label
  fidelity). Per-position records committed from the first checkpoint (the per-problem lesson).

## Machine allocation (two Studios)

- Now → G1: both on Track A (allocation grid + pairwise).
- After A's grids: one machine to B-1/B-2 (checkers, E1/E4), one finishes A's remaining cells, then
  joins B. Track 0 items are cache post-processing — slot anywhere.
- V-E1 claims a machine only after G3.

## Gates summary

| gate | trigger | decides |
|---|---|---|
| G1 | pairwise result | III's R3 story; unblocks III writing |
| G2 | poet consent | IV human arm at full vs fallback scope |
| G3 | IV-E5 scored | V unparks |
| G0 | Louay's voice pass | II to arXiv (independent of all above) |

## Risks worth naming once

- Human-rating throughput is IV's bottleneck — protect rater goodwill (short sessions, real poems,
  results shared back). Burned raters are unrecoverable.
- Goodhart optimization (IV-E3) can produce degenerate text that wastes rater time — cap pressure
  ladder in a pilot before scheduling human sessions.
- Checker bugs silently corrupt E1/E4 — hence unit tests before experiments, per B-1.
- Series scope creep: V stays parked until G3 regardless of how interesting IV's early results are.
