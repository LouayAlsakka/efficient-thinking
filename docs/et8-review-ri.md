# Review — Efficient Thinking 8: Experience Priors (draft of 2026-09-05)

**Reviewer:** Ri (理), Chief Strategy Officer · **Date:** 2026-09-05 18:2xZ · **Source reviewed:**
`~/Downloads/efficient-thinking-8-experience-priors.md` (28 KB, "early research proposal / working draft"), read against
the series canon in `~/github/efficient-thinking/` (README; Papers I–III openings; ET-VII concept; series execution plan).
**Requested by:** Louay, terminal, 2026-09-05 ~18:20Z ("document your review in your memory (on git)").

## 1. The claim, restated so it can be attacked

Intelligence (the frozen model) determines which reasoning paths the system CAN traverse; experience determines which
it tries FIRST. Experience is therefore an adaptive prior over search, learnable into a small auxiliary module
(`P(a|s,E_φ) ∝ P₀(a|s)·E_φ(a|s)`, or a residual on hidden states), consolidated offline from verified trajectories
(detect → distill → verify → consolidate), while the base model stays frozen. Success metric: fewer search tokens /
branches / tool calls to the same success rate on unseen structurally similar tasks, versus the same frozen model with no
prior, with distilled text memory, and with QLoRA.

## 2. Verdict

**Sound, and correctly placed in the series.** Papers I–III study search at inference time on a fixed model; ET-8 studies
learning the search policy across time. The lifecycle is right, and the verification gate before consolidation is the
part such proposals usually omit. Worth writing. Three places where the series' own results should push harder on the
draft, then two smaller points.

## 3. Where the series' results bear on the draft

### 3.1 The evaluator bottleneck applies to the verifier of experiences (Papers II, III, VII)
Paper II's law: only external information raises the ceiling. Paper III's price: an LLM judge pays only where its
selection beats the free consensus and coverage leaves something to select. ET-VII's bound: internal computation cannot
raise I(T;W); realized accuracy rises only toward what the state already encodes. In ET-8 the lesson-verifier IS a judge,
and §4.3 lists its possible sources (task outcomes, unit tests, external judges, stronger models, repeated trials,
counterexamples, synthetic variants, cross-agent evidence) without separating EXTERNAL signal (tests, solvers, outcomes)
from the model judging itself (synthetic variants it generates, a stronger model of the same family). **Where the
verifier is internal, the consolidated prior is bounded by the elicitation gap Δ = A* − A (ET-VII) and can at most
teach the model to search where it already half-knew to search.** Recommendation: state this in §4.3; run the first
experiment with a verifier that is external by construction, so the result is not confounded by judge quality; and
register, before running, what fraction of "lessons" the external verifier rejects — that rejection rate is the
measurement of how much a self-verified loop would have consolidated wrongly.

### 3.2 Text memory is the baseline to beat, and the draft is honest but underestimates it (§7, §14, §19)
Evidence from this estate, 2026-09-04/05: a lane's memory notes (a few hundred tokens, re-read at session start)
measurably changed what it tried first across a 20-hour session — which fetch form, which push shape, which check
to run before which. Cost: hundreds of tokens per session, zero training. The prior must beat that at equal TOTAL compute,
consolidation included, on unseen variants. §19 lists "equivalent textual memory achieves the same efficiency" as a
failure condition; per the series' own standing rule ("no text before numbers; proposals commit before experiments run"),
make it the PRIMARY registered prediction with a number: e.g. "experience gain ≥ X% over distilled text memory at equal
compute on held-out variants, or the mechanism is not worth scaling." Also register the interference prediction of §10.2
(how many consolidation rounds before out-of-domain regression exceeds Y%).

### 3.3 Start in chess, not in program debugging (§11.2)
The MCTS policy prior IS the object ET-8 proposes: it does not expand the reachable set, it allocates simulations.
Paper I supplies a frozen evaluator (3.45M value net), an exact external oracle (Stockfish), a measurable search cost
(simulations to a given strength), and code that already exists in the repo. Learning a small prior from
verified self-play trajectories and measuring simulations-to-Elo is §20's milestone with every confound removed:
no tokenizer, no retrieval, no judge, no prompt-interpretation cost. The draft says "a chess-like environment may be
especially useful"; I would make it the first experiment and put Qwen debugging tasks second, where the text-memory
baseline and the transformer-internal mechanisms (§11.4 B–E) become meaningful.

## 4. Two smaller points

**4.1 The distillation record (§5.2) is the claims-protocol record (WO-227) in another vocabulary.** `condition /
action_prior / action_avoid / evidence / confidence / scope` maps onto `{value, source, scope, supersedes, status}`.
One object could serve both; this estate already emits thousands of such records a day, with provenance, and they are
the raw material ET-8's detect step needs. Worth a sentence in §6 (memory and experience should coexist).

**4.2 §10.5 "excessive prior strength — expertise becomes dogma" deserves its own experiment, not a paragraph.**
Live instance, 2026-09-05: a confident, later-withdrawn instruction ("land the parked ref") held by every lane was
worse than no prior; 96% of the parked refs it would have landed were hazards (tsubame). A prior that cannot be
overridden by contrary evidence is a liability, and the draft's safeguards (confidence decay, domain gating, rollback)
should be tested as their own arm: inject one verified-then-invalidated lesson and measure how long it steers.

## 5. What I would add to the abstract (one sentence)

*The experience prior is only as good as the verifier that admits lessons into it, and that verifier is subject to
everything Papers II, III and VII established about evaluators.*

## 6. Louay's emphasis (terminal, 18:2xZ) and my response

**Louay:** the core of ET-8 is two lists compiled per night — *what works* (tried several ways, found the optimal one)
and *what didn't work* (tried one thing hard, failed; not obvious at the time, but over time learned to be a dead path).
This is how knowledge is gained. Risk: getting stuck in what was learned — a logical prison — so priors must be
guidelines that can be challenged, not rules always respected.

**Response.** The two-list schema is the right minimal form (it is `action_prior` / `action_avoid` in plain words), and
the estate already compiles it by hand every night (WO pile, doctrine lines, per-lane memory). Five points:

1. **The two lists have asymmetric evidence requirements.** A *works* is strong only if alternatives were actually tried
   and beaten (else it is "happened to work once"). A *didn't* is strong only if the failure reproduces and its cause is
   understood (else it is a confound in a lesson's clothes). Three instances from one shift, 2026-09-05: "shallow fetch
   takes 6 min" was measured inside an unannounced `gc`; the "21× qualified fetch" adopted by four lanes was cache
   ordering; a fetch claim was retracted within five minutes by a decisive test. Paper I's rule is the gate: **no delta is
   believed until it survives a low-variance re-measurement.** The nightly compile needs that gate or it consolidates
   artifacts into instinct.
2. **Every dead path carries its own unlock condition.** "Didn't work" means "didn't work under these conditions", and
   conditions change (§10.4 temporal drift). The exit from the prison is not general permission to disobey, which nobody
   exercises, but a specific predicate per lesson: what would have to be true to retry. "Partial clone is dead" is a
   prison; "partial clone is dead until `uploadpack.allowFilter` is set on lm" is a guideline with a door. Symmetrically,
   a *works* carries the alternatives it beat, so a new alternative reopens the comparison automatically.
3. **A prior should predict, and be re-tested on a miss.** "This path costs X" / "this branch fails" — when the
   observation diverges from the prediction, that lesson is re-opened. Cheaper and more targeted than a fixed exploration
   budget (§10.5), and it is the mechanism that makes "challengeable" operational rather than aspirational.
4. **Confidence scales with independent instances and decays with age.** One hard failure is an instance, not a dead path
   (this seat's rule: abstract on the third instance, never the first; live counterexample the same day — "a clone cannot
   use the wheel", withdrawn within the hour). Promotion from note → lesson → prior is earned by recurrence across finders.
5. **The record.** Two lists, one schema: `condition · action · alternatives_tried · evidence {n, instances, finders} ·
   confound_check · scope · confidence · expiry · reopen_condition (dead paths)`. This is the WO-227 claim record with two
   fields added (`alternatives_tried`, `reopen_condition`). The schema is what keeps the lists from becoming a prison; the
   two words alone would not.

## 7. Scope of this review

Read once, against the series openings, not the full bodies of Papers I–III; no experiments run; the estate evidence
cited (memory notes, the parked-ref hazard) is from WO-244 and this reviewer's own session, not from a controlled study.
The proposal's own standing rules apply to this note: it is registered by its commit timestamp, and its predictions
above (3.2's rejection rate, 4.2's steering half-life) are the ones I would score.

— 理 Ri
