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
| VII The Elicitation Gap | DRAFT v0.4 (`efficient-thinking-7.md`): registered claim FAILS under forced choice at 7B and 14B (Δ within zero; abstention was format); §3d complementarity registered for 32B + matched depth; §3c depth sweep pending; E-D has no gap to recover on measured judges | `experience/results/et7_ee_RESULT.json` | E-E probed Δ on the III judging cells; E-C ensemble decorrelation; E-D coherence bootstrapping with the cross-registration (E-D gain ≤ E-E's Δ); backward rewrite |
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
