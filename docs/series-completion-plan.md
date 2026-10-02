# Efficient Thinking — Series Completion Plan (III–VII, 8b, IX)

*2026-09-19. Ruled by Louay: "keep going; a work order for the executor with time and resources so that III–VII are
finished; we also need 8b and IX." This document is the plan; the executor's work order cites it. Every paper is
written backward from its measured results (the 2026-09-17 ruling), and every document carries one of three states
on its first line: MEASURED (results on disk, predictions scored), REGISTERED (predictions committed, runs pending),
CONCEPT (no registration yet). A reviewer graded concept documents as findings on 2026-09-19; the state line
prevents that.*

## 1. Where each paper stands (from the repo, not from memory)

| Paper | State today | On disk | What "finished" means |
|---|---|---|---|
| III Efficient Judging | MEASURED, draft v0.2 | `efficient-thinking-3.md`; judge grid + MATH grid (M1–M4 hit); P1–P6 scored except P3 | P3 scored (judge search vs judge size, machine-only); backward rewrite; author's voice pass |
| IV Search Where Taste Is the Evaluator | MEASURED (machine arms); rater arms UNPARKED 2026-09-20 with a frontier-model judge | checkers (green); E1/E4 scored (P1 hit); E6; Goodhart pilot (dev); canon judge/policy 7a–7d | **Ruled 2026-09-20: the rater is a frontier model** (`et4-amendment-2-model-judge.md`: Fable rates, Opus is E3's optimisation target, self-consistency measured, G5p/G6p registered). E2/E3/E5 run after the cost gate; then the backward rewrite with the scope sentence changed |
| V The Exchange Rate of Feedback | CONCEPT | none | V-E1 simulated-oracle loop and V-E2 forgetting ledger are machine-only and run first; V-E3/E4 need the rater (G2) |
| VI The Label Ceiling | MEASURED (E-A ledger F1–F5), E-B chess run but unledgered | `et6-ledger.md` (Connect-4); `games/results/et6_eb.json`, `et6_f3*.json`; chess label bias 0.275, F3 does not transfer, F4 capacity-bound, stage trajectory flat | E-B ledger written and scored; the "checkpoint regen" question closed or declared out of scope; backward rewrite |
| VII The Elicitation Gap | DRAFT v0.5 (`efficient-thinking-7.md`), E-E COMPLETE 1.5B–32B: claim FAILS at every size under forced choice at matched depth (output catches up to a flat state; abstention was format); §3d fails at 14B, unscorable at 32B; E-D has no gap on this task. Next: a clean write-up as a negative result, or a different task where the output cannot be forced | `experience/results/et7_ee_RESULT.json` | E-E probed Δ on the III judging cells; E-C ensemble decorrelation; E-D coherence bootstrapping with the cross-registration (E-D gain ≤ E-E's Δ); backward rewrite |
| 8a Experience Priors | MEASURED, closing | paper of record | R5-C, R6, R7, then stop (§7.7) |
| 8b Accumulation | REGISTERED 2026-09-19 | `et8b-loop-gates.md` | the loop harness, its gates, gen0/gen1, the VII bridge arm |
| IX Symmetry | CONCEPT → registered by S1/S2 in `efficient-thinking-9-symmetry-concept.md` §5 | none | S2 head invariance (a join + one refit); S1 chess augmentation |

## 2. The order, and why

Two boxes (M3 Ultra, 256 GB each). 8a holds both through R7 (≈ 2026-09-22). After that, one box carries the
8b/VII line (the series' load-bearing open question: does experience accumulate, and does it need external bits),
the other carries the cheap closures.

| Week of | Box A | Box B |
|---|---|---|
| 09-22 | 8b harness + gates (`et8b-loop-gates.md` G1–G4), base-loop 300 | VII E-E (probes on the III judge cells; the 8a probe code applies) · III P3 |
| 09-29 | 8b gen0 fit + 300 · gen1 fit + 300 · VII bridge arm | VII E-C · VI E-B ledger + regen question · IX S2 |
| 10-06 | VII E-D (coherence training, the cross-registration) | IX S1 · V-E1 · V-E2 |
| 09-29 onward, box B gaps + API | IV E2/E3/E5 with the frontier judge (API, not GPU) → V-E3/E4 with the same judge as the simulated user | — |

Papers are written as each closes, backward, by the author with the executor's tables: III first (it is one
scored prediction and a rewrite away), VI second, VII third, then 8b and IX. IV waits for its judge and is not
written before; V's machine arms run, its human arms wait with IV.

## 3. Gates on the author's desk

- **G2 — the judge: RESOLVED 2026-09-20.** The rater is a frontier model (amendment 2). What remains on the author's desk is the cost approval per run.
- **III voice pass**, then the backward rewrite.
- **Credit line** for every paper: the pseudonymous experimenter/reviewer convention of 8a Appendix A.

## 4. Standing rules (unchanged)

No text before numbers. Proposals commit before runs; the timestamp is the registration. Every registered
prediction is scored hit or miss; misses at the same prominence. Per-problem logging from the first cell. Every
table carries its command. A conclusion may not outrun its own stated bound. Three probe controls at every fit.
A budget in the prompt makes each budget a different agent. Nothing scored by a persona oracle is a paper claim.

## Pipeline — ideas recorded, not registered (2026-09-21)

| paper | one line | state | order |
|---|---|---|---|
| X Representation | is language the efficient representation for a reasoner | concept registered; X-a next | after VIII-b is written |
| XI Communication | is language the efficient channel between two reasoners; a discovery ladder | idea (`efficient-thinking-11-concept.md`) | after X-a; inherits its instruments |
| XII Skills | a head that chooses which additive skill to bring, from verified history (merging dropped: it changes the intelligence) | idea (`efficient-thinking-12-concept.md`) | first experiment can run beside X-a or XI-a; needs no new field |

## The one question, and the rule for admitting a paper (the author, 2026-09-21)

The series asks one question from every angle: **how much useful capability can a fixed intelligence get per unit of
computation, and what moves that number?** A paper is admitted only if it is one angle on that question, with the
intelligence held fixed and the capability externally verified. A paper that changes the intelligence — training a
better model, merging models — is someone else's programme and is cited, not pursued.

| paper | the angle | the lever |
|---|---|---|
| I | search moves along the frontier | inference-time search |
| II | where search pays and where it cannot | the evaluator's ceiling |
| III | the price of selecting | an LLM judge against free baselines |
| IV | search where the verifier is taste | a human, then a frontier model, as evaluator |
| V | the price of external information | feedback, per bit and per minute |
| VI | why self-play stops | label fidelity |
| VII | what a fixed system can get from inside | the elicitation gap |
| VIII | experience moves the frontier | a read-only prior from verified history |
| VIII-b | does it move again | accumulation across generations |
| IX | the cost of not knowing what does not matter | symmetry |
| X | is language the efficient representation | representation |
| XI | is language the efficient channel between two | communication |
| XII | experience chooses which skill to bring | the selector as an experience prior |

Every row holds the intelligence fixed and measures capability against compute. The test for a new idea is one
sentence in the middle column; if it cannot be written, the idea is not in this series.

## Publication route (the author, 2026-09-21): direct, no venues

The work is published where it is made. No conference, no journal, no endorsement, no submission form. The route:

1. **The record** — `docs/efficient-thinking-<n>.md` with html and pdf beside it, on GitHub, served from the author's
   domain. A DOI from Zenodo for each paper so it is citable without a gatekeeper.
2. **The cut** — an eight-page version of VIII (and of VIII-b when written) for a reader deciding in ten minutes whether
   to spend three hours; a courtesy to the reader, not a rule of anyone's. Figure 1, the claim in three sentences, the
   reproduction command.
3. **The challenge** — one page per paper: the claim, the figure, the command, the four attack angles stated by the
   author (an alternative explanation for the matched-compute shift; leakage in the experience construction; a flaw in
   the paired statistics; a reason the second search structure does not establish transfer), an open invitation to break
   it, the repository's issues as the review channel, and a bounty for a fatal flaw.
4. **Readers** — practitioners who will run the command: the MLX community (it runs on Apple silicon), builders of agents,
   the forums that argue about self-improvement. Two readers who run it and report back outrank a venue.
5. **What is kept** — the series' own discipline, which is what makes the challenge credible: pre-registered readings,
   withdrawn claims kept, instruments named by hash, a reproduction that fails loud, the author named plainly as an
   engineer from outside the field working with AI assistance.

Owner of 2–3: 理. Owner of 1's DOI and 4's posts: the author, on his word.

**2026-09-24 (reviewer on VIII-b v0.2, adopted):** VII + VIII + VIII-b read together — the series' question stated from the inside: how the useful internal states of a fixed intelligence are exposed, navigated and accumulated. The escape question (different policy / agent / novelty-seeking exploration, not a third generation) is VIII-c's, on the experience line; X (representation) is untouched.

**2026-09-24 06:5xZ — CORRECTED 07:0xZ by the author: this is VIII-c, NOT X. X remains Representation (is language the efficient representation for a reasoner; the X-a…X-d ladder ruled 09-21) and is unchanged. VIII-c is the next rung of the EXPERIENCE line — Paper VIII-c's question and its first two arms (author's reading of VIII-b: a plateau from internal saturation — one generation reaches what the policy can reach; more of the same is diminishing return):** *How does a fixed intelligence escape the experience region its own policy returns to?* Ordered by cost and by what a null teaches:
1. **Different agent, same family** (first arm; mostly exists — VIII's Llama-3.1-8B on the disjoint 300): a second model family collects experience on the same problems; its redundancy against gen0 by the §13a instrument BEFORE any head is fitted; head at matched decisions, three-game match under §15. Different region + gain extends → the region is the policy's; same region → the region is the task's and no policy escapes it. Either is a finding. **Multiple agents exchanging experience** = the same arm with the head on the union, free.
2. **Exploration optimised for state-space novelty**: collect under a policy rewarded for distance from gen0's region (the distance §13a already computes); the diagnostic becomes the lever; the only arm that could show compounding for a mechanistic reason.
3. **Teacher/student**: a stronger model steers, the head is fitted for the 7B — confounded with the teacher's competence; third.
- **Different task family**: measured (VIII-b §6, at base) — the boundary, not a route. **Gen3/4/5**: not run.
Registrations, not runs, until the demo work and the week's estate work are done.


## Candidate series principle (owner, 2026-09-26) — a HYPOTHESIS, not a result
Self-generated information is constrained by the process generating it. Instances already in the record: II
(correlated sampling limits), VI (self-label limitations), VIII-b (self-experience saturation); VIII-c shows one way
to alter the generating process. The project's first negative result (adaptive MCTS from its own experience) may be
an instance; untested, and not claimed. Open question raised by VIII-c, not pursued in it: which states should an
agent seek in order to acquire the most useful experience — the acquisition policy as an optimisation dimension.


## VIII-d — the acquisition policy as the object (agreed in principle 2026-09-26; NOT registered; not before VIII-c v0.2 is second-read)
Question VIII-c opened and did not pursue: which states should an agent seek in order to acquire the most useful
experience. Self-play in the only sense VIII-c supports: the agent's acquisition rule is what varies; the
intelligence, the head form and size (shared, 290 groups, one seed), the verifier labels and the scoring set
(seed 89) are frozen. Candidate arms, each a collection under one rule: native policy; forced look at decision 1
(= VIII-c arm 3, the anchor); forced look at every decision; look at the region the current head is least certain
of; the head's own choice. One prediction to register from VIII-c §16c (iii): selectivity (lower agreement with
the policy, higher hit given agreement) predicts play — if a rule that raises disagreement raises solved problems,
acquisition is optimisable by that quantity and the loop closes (acquire where the head disagrees, refit, repeat);
if it saturates at "look first", the dimension is real but shallow in this family. Cost: ~85 min collection per
rule, three games per arm; five arms ≈ two to three days of one box. The ET-1 / adaptive-MCTS connection is tested,
if at all, as one arm inside this design (adaptive from own experience vs from a forced acquisition rule), never
by touching the chess harness first.

## What the series lacks against the field's standard (owner, 2026-09-27, after reading the nearest published neighbour) — the generality programme, VIII-e, planned beside VIII-d; nothing starts before VIII-c v1.0
The nearest neighbour on mechanism (a training-free steering vector over deep hidden states of a frozen reasoning
model, fitted once and held fixed across tasks) studies reasoning economy, not experience; it shares our instrument
and none of our claim. What it has that we do not, and what we take from it:
1. **Model breadth.** Ours: one pair (7B/8B). Theirs: four families, 0.5B–32B. Plan: the 8c protocol (self head vs
   forced-look head, same form, same bars) on two more families at 7–14B, and one size ladder within one family.
   The claim of record is tested per family; the paper reports where it holds and where it does not.
2. **Task breadth with a verifier.** Ours: one synthetic family (pipeline debugging); a second in-house family
   (SQL repair) already exists in the harness. Plan: the 8c protocol on SQL repair (owned), then on one public
   verified family (unit-tested code generation or answer-checked math), where acquisition = where the agent
   stands before it commits has a natural analogue (read the tests / run the example before answering).
3. **Fit once, hold fixed.** Theirs: one offline fit transfers across benchmarks. Ours: one head per task family.
   Plan: the cross-family head is already a negative in VIII-b (§6); the generality programme reports it per
   family rather than assuming it, and tests whether the ACQUISITION RULE (not the head) transfers — the rule is
   the thing the claim says is general.
4. **Named baselines.** Theirs compare to the field's methods. Ours compare to the model's own preference,
   majority vote and matched compute. Plan: add the two baselines a reviewer will ask for — retrieval of past
   successes into the prompt, and a parameter-efficient fine-tune on the same verifier labels — so the head is
   shown to be a channel, not merely a cheaper training.
5. **Reporting.** Theirs report tokens and accuracy per benchmark; ours report the frontier with intervals and
   within-arm noise. Keep ours; add theirs as columns so the two can be read side by side.
6. **Release.** Code, one-command reproduction, DOI, challenge page — already the route; the neighbour's project
   page is the bar for how a reader arrives at the result in one click.
Order: VIII-c v1.0 → challenge page → VIII-d (acquisition policy) and VIII-e (generality) as one registered
programme, families first because the claim of record is what a second family can falsify cheapest.


**VIII-e, first arm (registered 2026-09-28):** the baseline comparison a reader of VIII asks for — on the same 300-problem sets, (a) a value function fitted on trajectory return over the same episodes, (b) ExpeL-style retrieval of past successes into the prompt — against the frozen head, at matched compute. Result published either way.

**VIII-d, registered question (2026-09-28):** does the forced-look advantage come from coverage (the head sees more of the states the game later visits) or from readability (the true region is more linearly separable in post-inspection states, so the head learns cleaner directions)? Separable by fitting on matched state counts with and without the inspection, and by the head's held-out fit on each; the answer decides whether an acquisition policy should seek more states or more readable ones.


## VIII-d and VIII-e — REGISTERED 2026-09-30 (readings fixed before any run; runs on the studio boxes when they are otherwise idle)

Frozen throughout: the 7B model, the head form (shared linear readout, layer 18, one seed), the verifier labels, the scoring set (seed 89, 300 problems), the loop harness (`et8b_loop`, budget 12, up to three head decisions), and the bar (3.67 points, the session's largest within-head-arm range; a difference is a finding only when its 95% paired interval excludes zero in 3 of 3 games, as in VIII-c §14a). Every artefact is named by sha256 in a lock before it is scored; the second reader hashes the lock.

### VIII-e, arm 1 — the baselines VIII owes (~2 days)
Question: does the frozen head's gain over base survive against the two objects a reviewer names first?
Arms, all on the 290-decision cut of the forced-look collection (`cut290`: 290 decision groups drawn from 815 at a fixed subsample seed, the published forced arm's own fitting set) so the comparison is like for like. Its task set is DISJOINT from the three scoring games' task set — checked by region name per task, not by the positional task id, which is shared and misleading: 193 of 202 shared ids name different regions. That disjointness is a requirement, not an incidental: an earlier arm in this series was withdrawn outright because its head was fitted on states from the same task set its games scored, so it held the verifier's answer for the exact programs it was asked about.
Re-registered before any run, and the reason stated because it changes what the arm tests: the original form fitted every arm on the own-experience states the published g1-matched head was fitted on. Those states cannot be identified. The lock for the published comparison names exactly two artefacts by sha256 and both are heads; no states artefact is named by hash anywhere in it, and the g1-matched head's only record of its inputs is the free-text phrase "fitted on seed-21 states". Those bytes are also on neither research machine. So there is nothing to stage and nothing to hash against, and regenerating them would be a different collection. The arm therefore fits on an identified collection instead, which keeps every arm inside one collection.
- (a) VALUE FUNCTION: the same linear form fitted on trajectory RETURN (solved / not solved at episode end) instead of the verifier's per-candidate region label; read at the same decision.
- (b) EXPEL-STYLE RETRIEVAL: the k nearest past successful episodes (by state embedding, k=3) inserted into the prompt as text; no head; compute charged including the retrieval tokens.
- (c) ANCHOR: the published half-size forced arm (132 groups, 32.6%), whose head IS named by sha256 in the lock and is hash-verified on the research machine; base (22.0%) re-run in the same session as the control. This changes the arm's question: against the g1-matched arm it asked whether the head beats matched other-agent experience, and against the half-size forced arm it asks whether it beats a random half of the same collection. The second is the question the 3.67-point bar was constructed for, and it is the one answerable with identified bytes.
Readings, fixed now: R1 the head beats (a) by more than the bar → the LABEL (verifier region, not return) is what the head buys; R2 the head beats (b) at matched compute → the CHANNEL (outside the prompt) is what it buys; R3 (b) ≥ head → the series' mechanism is not distinctive on this family and VIII's "moves the frontier" is restated as "matches prompt retrieval at lower compute" or withdrawn, as the numbers say; R4 (a) ≥ head → the verifier label is not load-bearing. Three games per arm; published either way.

### VIII-d, arm 1 — coverage or readability (~1.5 days)
Question (VIII-c §7, registered 09-28): does the forced-look advantage come from the head seeing more of the states the game later visits (coverage), or from post-inspection states being more linearly separable (readability)?
Design: from the forced-look collection (VIII-c arm 3 states) draw two fitting sets of MATCHED size (132 groups, the half-size arm's count): S-cov = states sampled to maximise overlap with the states the scoring games visit (nearest-neighbour by embedding to a held-out game's states); S-read = states sampled to maximise the head's cross-validated fit on the fitting set itself. Fit one head on each; play three games each; report the CV fit and the game score of each beside the published half-size forced arm (32.6%).
Readings, fixed now: R1 S-cov beats S-read by more than the bar → coverage; R2 S-read beats S-cov by more than the bar → readability; R3 neither separates → the two are confounded in this family and VIII-d's next arm must vary the acquisition RULE, not the sample; R4 either beats the published half-size arm by more than the bar → a selection rule over states is itself an acquisition policy, which is VIII-d's thesis.

**Amendment 2, 2026-10-01, before any run — the held-out game held nothing out.** Measured: the three scored games are repeats of the same 300 tasks at different sampling seeds (task overlap 300 of 300; visited-state sets 98.5% identical), so a coverage target drawn from "game 1" is the test distribution itself. Re-registered: S-cov's overlap target is a FOURTH game on a DISJOINT task set — seed 21's 300 problems, generated by program signature and proved disjoint from seed 89 — played once with the published half-size forced head, its visited states recorded and never scored; S-cov is drawn to maximise nearest-neighbour overlap with those states; S-read's search is greedy forward selection by 5-fold CV fit, deterministic order, seed 0, its CV column reported as selection-biased and not read across arms. Both heads are then scored on seed 89's three games as before. Cost: one extra game. Readings R1–R4 unchanged.

**VIII-e arm 1, re-registered 2026-10-01 after the pre-run checks:** (1) The states: `viiic_LOCK.json` names no states artefact by hash (only two heads), so the published g1-matched head's input cannot be staged; the arm runs on the seed-89 forced-look collection (cut290, 290 groups, hash-verified) with the published half-size forced arm (32.6%) as anchor, and the question becomes "does a baseline beat a random half of the same collection", which is the question the 3.67-point bar was built for. (2) Arm (a), a value function on trajectory return, is DEGENERATE as registered: return is a property of the episode and is constant inside every one of the 290 decision groups the head is scored on (0 of 290 vary), so a head fitted on it cannot rank within the unit it is scored by; arm (a) is replaced by a value head fitted on the agent's own pick (candidate == agent region, varying in 169 of 290 groups), the nearest label that is a property of the candidate, and R1 reads "the head beats the agent's-own-pick head by more than the bar → the verifier's label is what the head buys". (3) Arm (b), ExpeL-style retrieval: measured before building, k=3 is at chance (36.0% vs 32.9%, p=0.14) because neighbours 2 and 3 dilute the one that transfers (k=1: 20.3% vs 12.4% chance, p<0.001); k is 1. Retrieval is ONCE PER DECISION, re-queried as the state changes, because the head it is the baseline for is re-scored at every decision; its compute is charged at that rate and R2's matched-compute comparison is made on it. (4) Anchor, control and readings R2–R4 otherwise as written; R1 as restated here.

**Amendment 3, 2026-10-01, before any run — two selection rules that could not discriminate.** (a) S-cov: on raw layer-18 embeddings a random 132 groups reaches 99.93% of the achievable coverage and greedy equals the ceiling, so coverage could not have explained a game-score difference; embeddings are CENTRED (the pool mean subtracted) before similarity, which takes random to 97.57% and makes the greedy draw distinct. (b) S-read: greedy forward selection by cross-validated fit is undefined for the first picks, has a spread of ±0.13 at the full budget, and costs about 20 hours — it measured noise; replaced by a direct, search-free criterion: the 132 groups in which a head of the published form separates the true region from the others by the widest margin, computed once, deterministic, with the margins CROSS-FITTED (5-fold, each group scored by a head that did not see it) because an in-sample margin from the published head would have let its own training third decide 49 of the 132 picks. The CV column is dropped from the arm; the arms compare on game scores against the bar. (c) VIII-e arm (b): the retrieval bank holds every decision of each green episode, not first-correct decisions only, matching the per-decision query; measured transfer 16.0% against 12.5% chance (p = 0.002) at the query point the arm uses.

**Amendment 4, 2026-10-01, while VIII-d arm 1 runs (nothing scored yet):** (a) a FOURTH arm re-runs the published half-size forced head in this session so that anchor and bar share one instrument; the published 32.6% is then a cross-session reference, not the comparator. (b) A mix-matched control, S-cov constrained to S-read's exact per-decision counts, is drawn and queued (cost 0.0045 in coverage); the headline comparison is S-cov-mix against S-read, which differs in the objective alone, and the unconstrained pair is printed with its confound named. (c) VIII-e arm (a)'s replacement label (the agent's own pick) is all-zero in 121 of 290 groups because the pick is recorded on 58% of rows: the arm is scored on the 169 groups where the label varies, the head and the published head both scored on that same subset so the comparison is like for like, and the 121 are reported as unscoreable with their count; R1's reading is unchanged on the subset. Scoring plans are registered before arms finish, with what had been seen disclosed in the file.

**VIII-d arm 1 — RESULT, 2026-10-01, scored after an independent second read of the lock (lock recorded by hash before scoring; rates withheld until the read was signed).** Headline pair, S-cov-mix against S-read (objective differs, decision mix matched): −2.3 [−7.7, +3.0], +2.0 [−3.3, +7.0], −1.7 [−6.7, +3.3] over three games; no interval excludes zero and the signs alternate. The unconstrained pair reads the same (−3.0, +1.0, +1.0). **R1/R2: no difference between coverage and readability larger than the design's resolution, which was measured before the run at about 5.7 points (observed half-width ~5.3).** Session bar 3.00; the in-session anchor re-run gives 33.4% against the published 32.6%. **Not claimed:** all three selected heads beat the anchor by +3.3 to +8.0 in nine of nine pairings (3-of-3 criterion met by none; the nine-same-sign observation is unregistered pooling). The anchor's 132 groups were drawn from a 290-group cut of the union pool of base-plus-forced states; the three arms' 132 come from a 290-group cut of the pure forced collection. Different pools, so the gap is confounded by pool membership and is reported as an observation only.

**Amendment 5, 2026-10-01, registered before the draw:** a FIFTH arm, `rand132`, draws 132 groups uniformly at random from the same cut290 the three selected heads used, one draw, one fit, three games on the seed-89 set, same criteria (§14a interval excludes zero in 3 of 3; §14b lower bound clears the session bar). Reading R5: if any selected head beats `rand132` under both criteria, selection by that rule buys something inside one pool; if none does, the +3.3 to +8.0 above was the pool, not the selection. Runs after VIII-e on the same box.

**Amendment 6, 2026-10-01, before any `rand132` game:** the uniform draw at seed 0 came out 2.68 sd low on decision-1 groups (36 against 46.9 expected; a uniform draw is at least this far off with probability 0.010), which would make the baseline share one selected arm's decision mix and not the other's, the asymmetric form of the confound amendment 5 exists to remove. The arm is re-drawn STRATIFIED: uniform within each decision point, proportional across them (47/46/39), one seed, fitted once, three games. Redrawing uniformly until the mix "looked right" was rejected as selecting the thing whose value is that it was not selected. The seed-0 uniform draw is kept in the lock with its z-scores and is not played. R5 unchanged.

**VIII-e arm 1 — RESULT, 2026-10-01, scored after an independent second read of the lock; every artefact hash recomputed from the bytes (43 of 43 match).** Session base 21.0–21.3% against the published 22.0%; the frozen half-size head beats base by +11.3/+12.7/+15.0 (published +10.6), both criteria, 3 of 3. **R2/R3: prompt retrieval (k=1 once per decision, matched bank, tokens charged in the turn) is a null: −1.0, −2.0, −0.3 against base, no interval excludes zero, at +5.9% input tokens; the head beats it by +13.3 to +15.3, both criteria, 3 of 3.** Bound recorded before scoring and measured in-run: retrieval as implemented is hub-dominated (54 of 125 bank entries never returned), so the null is partly the bank, not only the method. **R1/R4: both labels buy something (verifier head +7.3/+10.0/+11.0 over base, 3 of 3 on the first criterion; agent's-pick head +5.7/+6.0/+6.7, 3 of 3), and the difference between them (+1.7, +4.0, +4.3) is below the pair's resolution of about 6.7 points. R1 as registered does not fire.** The registered prediction (agent's-pick head: higher offline fit, lower in-loop value) neither met its withdrawal condition nor was confirmed: direction held 3 of 3, no interval excludes zero. What the arm does establish: a 23-point offline advantage (83.4% vs 60.4% on identical rows) converted to no in-loop advantage.

**Amendment 7, 2026-10-01, replay-only follow-up and one retraction.** (a) The hubness correction approved for the replay was measured: dividing similarity by retrieval frequency is degenerate on this bank (raw pairwise cosines sit in a 0.08-wide band, so the divisor selects the rarest entry, not a similar non-hub); centring widens the band 22.6× and still changes nothing; every variant is at or below raw retrieval, with a most-generous ceiling of +1.7 points against arm 1's 3.7-point resolution. The lever is closed without games; the concentration is anisotropy of the embedding, not frequency. (b) **Retraction:** the pre-run replay estimate of 16.0% transfer, quoted beside the in-run 16.2% as "predicted to 0.2 points", cannot be re-derived: no step file on record yields its 820-decision query set (the candidate that would have been serious, scoring the bank's own source states, gives 815 and did not happen). The in-run 16.2% stands; the "predicted to 0.2 points" sentence is withdrawn and no decision rests on it.

**VIII-d arm 1, R5 — RESULT, 2026-10-02, scored against the complete five-arm lock after its independent second read (every artefact recomputed from bytes, 99 of 99).** No selected head beats the stratified random draw under either criterion in any of the nine registered pairings, and the point estimate is negative in nine of nine (S-cov −4.0/−1.7/−0.3, S-read −1.0/−2.7/−1.3, S-cov-mix −3.3/−0.7/−3.0 against `rand132`; mean green: random 41.0, S-read 39.3, S-cov 39.0, S-cov-mix 38.7, anchor 33.4). In amendment 5's own words: **the +3.3 to +8.0 above was the pool, not the selection.** This is not a demonstration that random is better: every interval spans zero and the per-game resolution is about ±5 points. Adding the fifth arm did not move the session bar (3.00; the new arm's within-arm range is the smallest, 0.67), so all fifteen previously published arm-1 pairings re-score identically. Disclosure: the arm and its three pairings were added to the scorer after its games finished; the reading was registered before the draw (amendments 5 and 6) and is quoted above unchanged. The anchor-versus-random comparison was not scored, since the anchor is a published head and not a selection rule. VIII-d arm 1 is closed with R1 to R5 answered.

### VIII-d, arm 2 — the acquisition rule as the object (after arm 1; ~2–3 days of one box)
The five collection rules named above (native; forced look at decision 1 — the anchor; forced look at every decision; look where the current head is least certain; the head's own choice), 290 groups each, one head each, three games each. Reading, fixed from VIII-c §16c (iii): if a rule that raises disagreement with the policy raises solved problems, acquisition is optimisable by selectivity and the loop closes; if it saturates at "look first", the dimension is real but shallow here.

Order: VIII-e arm 1 and VIII-d arm 1 run in parallel (different boxes); VIII-d arm 2 after arm 1 reads. Owner of runs: Sautée; second reader of locks: Takumi; hygiene: Metsuke. Priority stands as Louay set it: product model work on the studio boxes pre-empts these; they fill the boxes when nothing else does.
