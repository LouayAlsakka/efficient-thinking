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
| III Efficient Judging | MEASURED, draft v0.2 | `whitepaper3.md`; judge grid + MATH grid (M1–M4 hit); P1–P6 scored except P3 | P3 scored (judge search vs judge size, machine-only); backward rewrite; author's voice pass |
| IV Search Where Taste Is the Evaluator | MEASURED (machine arms), human arms WAITING | checkers (green); E1/E4 scored (P1 hit); E6; Goodhart pilot (dev); canon judge/policy 7a–7d | **Ruled 2026-09-19: IV waits until a good judge (human rater) is found.** No deadline, no machine-only fallback; the machine-arm ledger is written and the paper holds |
| V The Exchange Rate of Feedback | CONCEPT | none | V-E1 simulated-oracle loop and V-E2 forgetting ledger are machine-only and run first; V-E3/E4 need the rater (G2) |
| VI The Label Ceiling | MEASURED (E-A ledger F1–F5), E-B chess run but unledgered | `et6-ledger.md` (Connect-4); `games/results/et6_eb.json`, `et6_f3*.json`; chess label bias 0.275, F3 does not transfer, F4 capacity-bound, stage trajectory flat | E-B ledger written and scored; the "checkpoint regen" question closed or declared out of scope; backward rewrite |
| VII The Elicitation Gap | CONCEPT | none (the bound is stated; no probe run) | E-E probed Δ on the III judging cells; E-C ensemble decorrelation; E-D coherence bootstrapping with the cross-registration (E-D gain ≤ E-E's Δ); backward rewrite |
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
| when the judge is found | IV E2/E3/E5 (human sessions, ≤ 30 min each) → V-E3/E4 | — |

Papers are written as each closes, backward, by the author with the executor's tables: III first (it is one
scored prediction and a rewrite away), VI second, VII third, then 8b and IX. IV waits for its judge and is not
written before; V's machine arms run, its human arms wait with IV.

## 3. Gates on the author's desk

- **G2 — the judge.** IV's human arms and V's E3/E4 exist only with a good human rater. Ruled 2026-09-19: IV waits
  until one is found; no deadline. The plan's only ask is that the search for one is open.
- **III voice pass**, then the backward rewrite.
- **Credit line** for every paper: the pseudonymous experimenter/reviewer convention of 8a Appendix A.

## 4. Standing rules (unchanged)

No text before numbers. Proposals commit before runs; the timestamp is the registration. Every registered
prediction is scored hit or miss; misses at the same prominence. Per-problem logging from the first cell. Every
table carries its command. A conclusion may not outrun its own stated bound. Three probe controls at every fit.
A budget in the prompt makes each budget a different agent. Nothing scored by a persona oracle is a paper claim.
