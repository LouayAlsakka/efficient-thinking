# Efficient Thinking VIII-c: Where the Experience Is Acquired
## A frozen agent's plateau, another agent's places, a forced look, and what the difference is made of

> **STATE 2026-09-27: v1.0, 2026-09-27.** Content frozen at lock v3; every number second-read. The artefact read matched all 37 checkable numbers and the v3 delta to their artefacts; the results reader hashed every arm-game episode file of locks v2 and v3 (18 of 18 and 21 of 21) against the locks and recomputed every green count from those files (39 of 39 arm-games), re-deriving the six- and seven-arm tables from raw; the hygiene read found the needles clean. Nothing in the chain from raw episodes to the tables rests on a reported number. Versions before this are drafts and are superseded.

**Louay Alsakka** · September 27, 2026 · *v1.0*

## Abstract

Paper VIII showed that a frozen model with a read-only head fitted on its own verified history moves the quality–compute frontier. Paper VIII-b showed that the gain does not compound: a second generation of the agent's own experience is largely a repeat of the first, and a head fitted on it is not separable from the first by the loop's own noise. This paper asks what the plateau is made of, and finds that it can arise from how the agent acquires experience rather than from any exhaustion of its ability to use it: different acquisition policies produce differently valuable experience, and acquisition is an optimisation dimension separable from the intelligence and from the mechanism that consumes experience. The evidence is three heads of identical form and size, fitted on the same frozen model's hidden states with the same verifier labels, differing only in *which states* they were fitted on. On 300 problems no head had seen: a head fitted where the agent's own policy stood solves 26.8% (base 22.0%); a head fitted where a second, weaker agent's policy stood solves 35.4%; a head fitted where the agent was made to stand — the code read before any hypothesis, by a one-line rule during collection only — solves 41.2%. The second agent was the experiment that revealed the dissociation, not its mechanism: the effect needs neither another agent nor several. A head built from states the agent was made to visit is more selective than one built from states it chose — it agrees with the agent's own pick less often, and is right more often when it does. Mixing the two acquired sets at a matched size does not beat the forced look alone, but the second agent's groups are not dead weight: half the forced groups plus the second agent's groups recover most of the gain, and 132 forced-look groups are indistinguishable from 290 of the second agent's — consistent with about twice the value per group, not a measured ratio. A cheap held-out proxy tracks how much experience a head was fitted on and not where it came from, and ranked heads of equal size exactly backwards. One task family, one model pair: a mechanism demonstrated in this system, not a law.

## Results at a glance

**Observations** — seven arms, three games on the same 300 problems (seed 89), percent of problems solved. The first four arms ran adjacent within each game in one session (09-25); the forced-look and mixture arms ran on 09-26 against that bracket (§7). Means and within-arm ranges; no comparison is implied by adjacency.

| arm | what sits beside the frozen 7B | g1 | g2 | g3 | mean | range |
|---|---|---|---|---|---|---|
| base | nothing | 66 | 67 | 65 | 22.0 | 0.67 |
| gen0 | the head of VIII (its earliest experience) | 58 | 60 | 60 | 19.8 | 0.67 |
| self | the matched second-generation head of VIII-b (own experience) | 77 | 83 | 81 | 26.8 | 2.00 |
| other-agent | head on the 7B's states at the places agent B's policy visited | 102 | 104 | 113 | 35.4 | 3.67 |
| forced-look | head on the 7B's states at the places a forced first inspection visited | 122 | 126 | 123 | 41.2 | 1.33 |
| mixture | head on other-agent ∪ forced-look decisions, same size (drawn 158 + 132) | 117 | 113 | 117 | 38.6 | 1.33 |
| forced-look, half | head on the mixture's own 132 forced-look groups alone | 95 | 101 | 97 | 32.6 | 2.00 |

**Supported comparisons** — paired per problem, three games, read against the registered bars (§2). Only these rows are claims.

| comparison | mean difference (points) | §14(a): 95% interval excludes zero (games of 3) | §14(b): lower bound clears the session's largest within-head-arm range (games of 3) | where |
|---|---|---|---|---|
| other-agent − self | +8.7 | 3 of 3 | 1 of 3 | §3 |
| other-agent − base | +13.4 | 3 of 3 | 3 of 3 | §3 |
| forced-look − self | +14.4 | 3 of 3 | 3 of 3 | §3 |
| forced-look − base | +19.2 | 3 of 3 | 3 of 3 | §3 |
| self − base | +4.8 | 0 of 3 | 0 of 3 | §3 (VIII-b reproduced) |
| mixture − forced-look | −2.7 | 0 of 3 | 0 of 3 | §5 — within noise |
| forced-look, half − base | +10.6 | 3 of 3 | 3 of 3 | §5 |
| forced-look − forced-look, half | +8.7 | 3 of 3 | 0 of 3 | §5 — no saturation at half size |
| mixture − forced-look, half | +6.0 | 2 of 3 | 0 of 3 | §5 — the second agent's groups contribute |
| forced-look − other-agent | +5.8 | 2 of 3 | 0 of 3 | §5 — **within noise; not a ranking** |

Two rows a reader will be tempted to read as rankings are marked as within noise and are not claims: forced-look against other-agent, and mixture against forced-look. No comparison between the two acquisition sources is established here (§5).

## 1. The question

VIII-b ended on a diagnosis: a steered agent stands where the previous agent stood and produces the same state, so its experience stops being new. That leaves two readings of the plateau. Either the model has taken what it can from experience of this kind — its ability to *use* experience is exhausted at this size — or the model has stopped *acquiring* experience worth using, because its own policy keeps it in the same region. The two readings predict different things. If use is exhausted, no experience of this form helps further, whoever gathered it. If acquisition is the limit, experience gathered somewhere else, by any means, should help, and the head form need not change at all.

The experiment that separates them is to hold everything fixed — the frozen model, the head's form and size, the verifier's labels, the scoring set — and vary only where the states come from.

## 2. Instruments, and what is fixed

Everything runs on the loop harness of VIII and VIII-b (`et8b_loop`, budget 12, up to three head decisions per episode) with the same frozen 7B model. Three things are fixed across every arm and are the reason the comparison is clean:

- **The head.** A read-only linear head over the frozen model's hidden state at layer 18, shared across decisions, fitted on 290 whole decision groups drawn at one seed. Every head in this paper has this form and this count. Its labels are the verifier's ground truth (`bug_region`) in every arm; no agent supplies a label.
- **The states.** Every head is fitted on the *7B's* hidden states. When another agent's trajectory is used, that trajectory is replayed as a prompt to the 7B and the 7B's own state at each of the other agent's decision points is recorded. Nothing the other agent thought enters the head; only where it stood.
- **The scoring set.** All games run on a 300-problem set (seed 89) that no head in the comparison was fitted on. The fitting sets (seed 73 for the other-agent and forced-look heads; seed 21 for the self head) share no program with it and, of 70 symptom strings the sets share, at most five that point to the same region (one of 71 for the self head's set), against effects of thirty-plus problems.

The bars are those registered in VIII-b (§14): a comparison clears §14(a) in a game when the paired bootstrap 95% interval of the per-problem difference excludes zero; it clears §14(b) when that interval's lower bound exceeds the largest within-head-arm range of the session. A row is a claim only where the table says how many of the three games clear each. The session's largest range is the other-agent arm's, 3.67 points, and it did not move when the two later arms were added (they are the steadiest arms in the table).

Two properties of the instrument, established in VIII-b and confirmed here, frame every number: decision 1 is greedy on an identical state, so a head's first pick is deterministic and identical across games; all between-game variation lives at decisions 2 and 3.

## 3. The dissociation

**The self head reproduces the plateau.** On the new set, the matched second-generation head of VIII-b beats base by +3.7, +5.3 and +5.3 points and clears no bar. This is VIII-b's result on a third problem set, and it is the control the rest of the paper is read against.

**The other-agent head clears it.** Agent B is a second frozen model of a different family (Llama-3.1-8B), the weaker of the two: across two collections it solves 34–36 of the 300 fitting problems where the 7B, across two runs, solves 60–68 (the head was fitted on the 36 run). Its policy differs in one measurable habit: it inspects the code before its first hypothesis on 70% of problems (209 of 300 on the run that fed the head), where the 7B rarely does. A head fitted on the 7B's states at B's visited places, labelled by the verifier, beats base by +12.0, +12.3 and +16.0 and clears both bars in every game; it beats the self head by +8.3, +7.0 and +10.7, clearing the point-estimate bar in every game and the interval bar in one. Ability to use experience was not exhausted: the same head form, on states from elsewhere, more than doubled the gain.

**The other agent was not needed.** The forced-look head is fitted on the 7B's own trajectories with one change during collection only: at decision 1 the action is forced to be an inspection (charged as an action, target chosen by the loop's existing rule), with decisions 2 and 3 under the native policy. That head beats base by +18.7, +19.7 and +19.3 and the self head by +15.0, +14.3 and +14.0, clearing both bars in every game. Against the other-agent head it leads in every game but the difference is within the session's noise (§5). The registered reading is that the forced look is at least as good as the second agent's own trajectories: what mattered was where the model was standing when the experience was recorded, and a one-line rule can put it there.

So the plateau of VIII-b was acquisition. The agent's own policy returned to the same region; a head fitted there was fitted where the evidence was not yet in the prompt.

## 4. What the difference is made of

Three pre-registered predictions were read from the games' own logs, and one alternative explanation was tested after the fact.

**The head is a better function on the identical input.** At decision 1 every arm sees the same state, so the share of first picks that name the true region is the head's function alone: base 23.3%, gen0 17.0%, self 30.7%, other-agent 42.3%. Of the other-agent head's 170 solves the self head did not reach, it had named the true region at decision 1 on 105 (62%); the self head had on 6 (4%). Its edge grows through decisions 2 and 3 (+11.7, +19.1, +21.3 points over the self head), but the later decisions are the dirtier comparison — by decision 2 the arms are in states their own first pick produced — so the clean number is the smallest one.

**Episodes get shorter, not longer.** The registered prediction that a better head would lengthen episodes (an inspection before the hypothesis) was refuted: mean actions fall, 7.59 under the other-agent head against 8.21 base, because a correct first pick ends the episode sooner. The habit itself does not transfer at game time — the loop forces a hypothesis at decision 1 under every head — and the result does not need it to.

**The forced-look head is not degenerate; it is selective.** The worry was that forcing every first move produces a head that always names this family's favourite region. Measured over every decision at which each head was consulted, the forced-look head is the *flattest* of the six (top-pick share 17.6% against a uniform 12.5%; the most concentrated head, gen0, is the only one that loses to base). What distinguishes it is selectivity: it agrees with the agent's own pick less often than the other-agent head (33.7% against 40.3%) and is right more often when it agrees (73.3% against 60.6%). A head built from states the agent was made to visit disagrees with the policy that produced it more, and its agreement carries more information.

## 5. Mixing the two, and what the matched size hides

The registered second arm fits one head on the union of the other-agent and forced-look decisions at the same size as either alone; the seed drew 158 of the second agent's groups and 132 forced-look groups. It solves 38.6%: below the forced-look head in all three games (−1.7, −4.3, −2.0; every interval spanning zero) and above the other-agent head by margins that also span zero. The reading registered for that outcome — that the second agent's places add nothing to a forced look — was written before the arm ran, and it is the reading this paper withdraws.

It is withdrawn because a matched size cannot distinguish "the second agent's groups are dead weight" from "forced-look experience saturates at half its size", and the discriminator registered to separate them lands on the other side. A head on the mixture's own 132 forced-look groups alone solves 32.6%. So the forced arm does not saturate: 290 forced groups beat 132 by +9.0, +8.3 and +8.7 (interval excluding zero in every game; lower bounds 3.3, 2.7, 3.0 against the 3.67 bar). And the second agent's groups contribute: adding its 158 groups to the same 132 forced groups moves 32.6% to 38.6% (+7.3, +4.0, +6.7; interval excluding zero in two of three). At a matched 290 the second agent's experience looked worthless because it was displacing forced-look groups, not because it carried nothing.

The strict status of the three new pairs is the status this paper already gives the forced-look-versus-other-agent pair: none clears both bars, so each is a lean and not a ranking. The point estimates order the six heads forced 41.2 > mixture 38.6 > other-agent 35.4 > forced-half 32.6 > self 26.8 > base 22.0, and no adjacent step in that order is established: three are inside the session's instability, the self-versus-base step clears neither bar (§3), and forced-half versus self was not a registered pairing and has no interval. No ordering is claimed here.

One sentence the arm buys, stated as consistent-with rather than established, is a magnitude for the paper's variable: 132 forced-look groups are indistinguishable from 290 of the second agent's in all three games (−2.3, −1.0, −5.3). That is consistent with a forced look being worth about twice a second agent's experience per decision group in this system; it is not a measured ratio, because an interval that contains zero difference cannot license one, and it would be the first thing to test elsewhere.

## 6. What did not hold

- **The cheap proxy measures the wrong variable.** Held-out pick accuracy at fitting time ordered the three heads of equal size other-agent 57.9% > forced-look 54.8% > mixture 53.8%; in the loop, by problems solved and by in-game hit rate, the order is forced-look > mixture > other-agent, exactly reversed. With the fourth head the failure splits cleanly: across sizes (290 against 132 groups) the probe ranks the half-size head lowest, correctly (45.5%). The held-out probe tracks how much experience a head was fitted on and not where it came from — the one variable this paper is about. This is the sixth probe non-conversion in the series by the experimenter's count (VIII §7.6, VIII-b §13c and §4 of that paper, and the heads here), recorded in the register rather than derivable from one artefact. It is not used as a diagnostic of acquisition here, and the split is a testable claim rather than a retirement.
- **"Above the other agent" did not fire.** After one game of the forced-look arm read twenty points above the other-agent head, a fourth reading was registered for that direction. The third game came in at +3.3 with an interval spanning zero and the reading did not fire. The one-game flag is in the appendix as the kind of number this harness produces.
- **The first other-agent arm was contaminated and withdrawn.** Its head had been fitted on the very 300 problems its games scored and read 134 of 300 — a head recalling its answer key. The tell was proportion, not sign: an effect an order of magnitude larger than anything the instrument had produced. The games moved to a third set no head had seen, and every number above is from that set.

## 7. What this says, and what it does not

Learning is usually discussed as one thing. This series has been separating three: the intelligence, which is frozen throughout and never changed; the ability to *use* verified experience, which VIII showed and which every head here exercises through the same read-only form; and the ability to *acquire* experience worth using, which is what the acting policy determines by where it stands when a state is recorded. VIII-b found the plateau; this paper finds that, in this system, the plateau was in acquisition, and that changing where the agent stands before it decides — by borrowing another agent's habit or by imposing one — restores gains of the size VIII first reported, with the intelligence held fixed.

The technical form is that the value of experience is conditional on the state distribution induced by the acquisition policy. The intuitive form is that where you learn from matters, not just how much you collect. Both dimensions are visible in the same table: quantity still matters (290 forced-look groups beat 132), and so does origin (two acquisition policies produced experience of different value per group, and each contributed). So not all experience of one form is equivalent, and the question this opens, and does not pursue, is the one the series began with, asked one level up: ET-I asked how to spend an inference budget; this paper ends by asking how to spend an experience-acquisition budget — given a fixed number of groups, which states should an agent seek in order to acquire the most useful experience? Experience acquisition is an optimisation dimension in its own right, separable from the intelligence and from the mechanism that consumes experience. That is the finding of record; the second agent, the forced look and the percentages are the path that led to it.

The claim is scoped to what was measured: one task family (debugging), one model pair, one head form, 300 problems per game. "Can arise" is the verb because a mechanism has been demonstrated in this experimental system, not a law of frozen agents. The two later arms ran a day after the bracket they are scored against, and the only evidence that session drift is small is the base's own within-day stability (66/67/65). The loop harness recorded no provenance of its own invocation at the time; the source set of the forced-look states was established from the data (candidate region names ⊆ each task's own regions: 300 of 300 against the fitting set, 15 of 300 against the scoring set) rather than from memory; the harness's invocation record (resolved arguments, task-set and head hashes, script sha) is written and tested and is applied after the last running chain is off the file — it is not yet in the artefacts of this paper.

## Reproducibility

Every game file, head, task set and score in this paper is named by sha256 in `experience/results/viiic_LOCK.json` (arms of §3), `viiic_LOCK_v2.json` (the forced-look and mixture arms) and `viiic_LOCK_v3.json` (the half-size arm), with the git sha of the script that produced each artefact and its exact invocation. A second reader verified the first lock — 108 of 108 hashes, 16 of 16 statistics from its table, 12 of 12 green counts from the files — and found the one discrepancy that moved a recorded verdict (§6, the bar). The second and third locks (the forced-look, mixture and half-size arms) were read by the results reader from the staged files: every arm-game episode file hashed against the lock (18 of 18, 21 of 21) and every green count recomputed from it (39 of 39); the per-slice files beneath them are named by hash in the locks and were not separately hashed by a second reader. Withdrawn runs are kept and named. The registrations, in the order they were written, are `docs/et8b-loop-gates.md` §16–§16c.

## Appendix A. How this was found

1. **09-23.** VIII-b's saturation diagnosis; the question of escape posed as "a different agent, same family — the fastest route".
2. **09-24 07:1xZ.** §16 registered: agent B collects on seed 73's 300; the readings fixed before collection.
3. **09-24.** First collection stopped at 33 episodes: the harness prompt carried a stale region menu from an earlier task family; B obeyed it, the 7B ignored it. Corrected; a fresh 7B base run showed the correction moved vocabulary, not difficulty (22.7% vs 20.7%).
4. **09-24.** Reading (i) fired on the registered 25% bar; a same-agent churn control then showed two runs of the same agent differ by 4 problems (6.7%) — the bar was tightened to a difference against churn before any number travelled.
5. **09-24 → 09-25.** The first head games: a case-fold collision between arm names made one arm silently reuse another's files (caught by an impossible timestamp); then the other-agent head read 134 of 300 and was withdrawn as contaminated (fitted on the scoring set). Games moved to seed 89.
6. **09-25.** Arm 1 scored: the second reader recomputed the bar from its definition and found the scorer had excluded the arm under test; the interval count fell from 2 of 3 to 1 of 3.
7. **09-25.** The owner's forced-look control registered before arm 1's score; its charging of the forced action found and fixed before it ran.
8. **09-26.** Forced-look scored: within the bar of the other-agent head. A one-game "above" flag registered as a fourth reading; it did not fire.
9. **09-26.** Mixture scored: below forced-look in all three games; the matched-size ambiguity named and its discriminator started the same hour.
10. **09-26.** Degeneracy tested and refuted; the selectivity finding; the probe found inverted against both outcome measures.
11. **09-26 23:0xZ.** The claim of record fixed by the owner before the draft: the plateau can arise from acquisition, not from exhaustion of use.
12. **09-26 23:51Z.** The discriminator landed against the experimenter's own six-arm reading: the forced arm does not saturate and the second agent's groups contribute. The registered "adds nothing" reading withdrawn; §5 rewritten; collection frozen at lock v3.

Credits: E ran every arm and found every instrument defect above before its number travelled; R (the results reader) found the one that moved a verdict. The owner designed the forced-look control and fixed the claim.
