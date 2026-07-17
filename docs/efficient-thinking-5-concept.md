# Efficient Thinking V: The Exchange Rate of Feedback — concept note

*Status: concept, parked pending ET-IV E5 (persistence) results, which seed this paper. Working
slot: Paper V. One sentence: the series ends by pricing its own ceiling-raiser — external
information itself, arriving as low-bandwidth human feedback into a continually adapting model.*

## Position in the series

I: search extracts what the evaluator contains. II: only external information raises the ceiling.
III: the judge's FLOPs are priced. IV: in taste domains the judge is the evaluator and q is
measured against humans. V: the human's bits are priced — improvement per unit of feedback, in a system
whose weights update in deployment.

The framework blesses the idea and bounds it: user feedback is external information, so continual
gain is possible in principle (the §8 positive control is the existence proof), and it is bounded
per session by the information content of the signal and, in the limit, by the user's own
consistency — the nonstationary-oracle ceiling.

## Spine metric

**Quality-per-bit.** Feedback events carry measurable information: accept/reject ≈ 1 bit,
choose-of-N ≈ log2 N bits, an edit or written critique more — but only after distillation into a
training signal, and the distillation loss is itself measured. Every experiment reports personal
quality q(t) against cumulative feedback bits, giving sample-efficiency curves in the series'
efficiency-frontier idiom.

## Mechanism (deployable tier)

Per-user LoRA adapter over a frozen base; choice feedback → preference pairs → DPO (or KTO for
unpaired accept/reject) micro-updates via a background process (MLX, Mac mini class hardware);
adapter weights never leave the device. A second, exploratory tier (test-time-training /
fast-weight variants, updates inside the forward pass) is registered as research, not claimed.

## Experiments

- **V-E1 Simulated-oracle loop.** A fixed hidden taste function (strong LLM with a frozen persona
  rubric) plays the user. Debugs the harness, gives clean q(t)-vs-bits curves, isolates algorithm
  choices (DPO vs KTO, adapter rank, update cadence) before any human hour is spent.
- **V-E2 Forgetting ledger.** Personal q(t) tracked jointly with general capability (form-validity
  from ET-IV, GSM8K slice) across updates. Arms: full-adapter vs LoRA-rank ablation vs replay
  mixing. The two-floor erosion result (35.3 → 19.3 at N = 64) is the registered baseline danger.
- **V-E3 Real continual study.** The ET-IV poet collaborator, multi-week: adapter updated between
  sessions, held-out-session q trajectory vs the static-LoRA arm from ET-III E5. Human self-consistency measured (same rankings re-rated at a
  week's distance) and reported as the ceiling.
- **V-E4 Bandwidth ablation.** Matched session counts, different feedback channels (1-bit
  accept/reject; choose-of-8; free-text critique distilled to pairs). Improvement-per-bit and
  improvement-per-minute-of-user-effort, separately — the product cares about the second.

## Registered predictions

- **P1.** q(t) rises and plateaus at or below the user's measured self-consistency; no run exceeds
  its oracle's own agreement with itself.
- **P2.** Without mitigation, general capability erodes measurably within tens of updates
  (two-floor mechanism); LoRA-only + replay bounds erosion to noise while preserving most personal
  gain.
- **P3.** Quality-per-bit is diminishing in cumulative bits (early feedback is worth the most).
- **P4.** Per minute of user effort, structured choice feedback beats free-text critique after
  distillation losses; per bit delivered to the optimizer, critique wins. (Both halves score.)
- **P5.** Improvement transfers within-domain (her poetry) but not across (her legal drafting) at
  small adapter capacity — personalization is narrow before it is deep.

## Deployment note

Per-user adapters continually trained on that user's corrections are this paper's mechanism in a
deployable form: on-device (Mac-mini-class) hardware means the training compute lives beside inference
and the personalization weights never leave the user's machine — privacy as architecture.

## Honest scope

Continual learning from human feedback has a large literature (RLHF/DPO lineage, continual-learning
forgetting results, TTT); the contribution here is not a new algorithm but the measurement frame —
bits-accounted exchange rates, the self-consistency ceiling, and the forgetting ledger — applied to
per-user adaptation at deployable scale, with the series' registered-prediction discipline.
