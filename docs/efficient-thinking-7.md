# Efficient Thinking VII: The Elicitation Gap
## What a fixed system knows that it does not say — a bound, and its first measurement
> **STATE 2026-09-22: DRAFT v0.1, written backward from ONE measured arm.** E-E is measured on one judge (Qwen2.5-7B) and its scaling half is running; E-C and E-D are registered and not run; the bound's formal statements are to be verified against textbook forms before publication. Every number below is on disk under `experience/results/` with its command; the reading rules were committed before the run (`docs/et7-ee-prereg.md`, with its §3a amendment). Nothing here is a law; §10 says what is a finding and what is a registration.

**Louay Alsakka** · September 22, 2026 · *draft v0.1*

## Abstract

A fixed system — a model whose weights and inputs do not change — can improve its realised performance only toward the
information its state already carries about the target, never through it. That is a theorem (the data-processing
inequality and Blackwell monotonicity applied to a statistical judge), and it is trivial unless the distance between
what the state carries and what the system expresses is measured. This paper defines that distance, the *elicitation
gap* \(\Delta = A^* - A\), where \(A\) is the judge's realised accuracy and \(A^*\) a lower bound on the accuracy the
best decoder of its state could reach, and measures it. On the stratum of a judging task where the policy that wrote an
answer carries no information about its correctness, a 7B judge picks the correct answer 24.8% of the time while a
linear probe on its own hidden states reaches 73.0%: \(\Delta = +0.482\), with shuffled labels collapsing to the
majority baseline, eight principal components failing to recover it, and a probe for policy identity near chance. The
gap is abstention rather than ignorance: the judge declines to choose on 66.4% of pairs and is 73.9% correct when it
commits, so the probe reads what the judge would say where it stays silent. Two shallow explanations — answer length and
policy identity — are tested and rejected. What follows from the bound is not that self-improvement is impossible but
that its internal room is exactly \(\Delta\), and this paper's registered claim is that the room is large on the one
judge measured. Whether it shrinks with scale more slowly than accuracy rises, and whether any internal procedure can
convert abstentions into correct commitments, are the two registered questions this draft does not answer.

## Results at a glance

| finding | measurement | where |
|---|---|---|
| the bound | internal computation cannot raise \(I(T;W)\); realised accuracy rises toward \(A^*(W)\), never through it | §2 |
| the gap, defined | \(\Delta = A^* - A\); \(A^*\) lower-bounded by a linear probe on the judge's state, \(A\) the judge's own pick with a tie scored wrong | §3 |
| the gap, measured | judge 24.8%, probe 73.0%, \(\Delta = +0.482\) on 137 balanced pairs, 5-fold CV over problems | §5 |
| the controls | permuted labels 0.41–0.52 (baseline 0.518); PCA-8 0.657; policy-identity probe 0.577 | §5 |
| the confounded stratum, withdrawn | 913 pairs where policy identity predicts correctness: PCA-8 matches the probe, identity probe 0.826 — not a finding | §5 |
| what the gap is | abstention: ties on 66.4% of pairs, 73.9% correct when committing; the probe reads the silent two thirds | §6 |
| what it is not | answer length (43.1%, below chance); policy identity (0.577, uncorrelated with correctness by construction) | §7 |
| the same shape in Paper VIII | the frozen model's own log-probability preference 20.3% where a probe on its state reads 66.7% | §8 |
| not yet measured | scaling (1.5B ties on 98.5% of pairs — a format floor, not a point); E-C; E-D | §10, §11 |

## 1. The question

Efficient Thinking I–III measured what search buys a fixed intelligence and found its ceiling: search extracts what the
evaluator already contains, and only information from outside raises the ceiling. Paper VIII found that verified
history moves the quality–compute frontier of a frozen model. Between those two sits a question the series had
stated and not measured: how much can a fixed system get from *inside* — from re-processing what its own state already
holds, with no new information? The public argument about recursive self-improvement is conducted almost entirely in
words on exactly this question. This paper gives it a definition, a bound, and one number.

## 2. The bound

**Setting.** Let \(T\) be the external target a judge tracks (here, which of two answers is correct), \(D\) everything
the system has ever received, \(W\) its state, and \(J\) its expressed judgments. A procedure is *internal* if it updates
the state as \(W' = \phi(W, R)\) with \(R\) internal randomness carrying no information about \(T\) beyond \(W\).
Coherence training is internal; self-play in a game is not, because terminal outcomes are a \(T\)-correlated channel
the rules supply.

**Proposition.** For any judge and any internal procedure: (i) \(I(T; W') \le I(T; W)\) — internal computation cannot
increase the information the state carries about the target (the data-processing inequality over
\(T \to D \to W \to W'\)); (ii) with \(A(W)\) the judge's realised accuracy under a proper score and \(A^*(W)\) the
Bayes-optimal accuracy of the best decoder of \(T\) from \(W\), every internally reachable judge satisfies
\(A(W') \le A^*(W)\) (Blackwell: garbling cannot improve Bayes risk).

**Definition.** The elicitation gap is \(\Delta(W) = A^*(W) - A(W)\): held but misprocessed information. Internal
self-improvement is worth at most \(\Delta\); everything above \(A^*\) must be imported. The proposition is standard
results applied to this setting; the size of \(\Delta\) is empirical, and the series took no advance position on it.

## 3. Measuring the gap

\(A^*\) is not observable, but it is lower-boundable: a small decoder trained on the judge's internal representation to
predict \(T\) directly scores at most \(A^*\), so \(\text{probe} - A\) estimates \(\Delta\) from below. Three rules make
that a measurement rather than a story.

- **\(A\) is the judge's pick with a tie scored wrong.** A judge may answer A, B, or tie. An unexpressed judgment is not
  a judgment: a selector that declines has not selected. The rule was registered before the run and is kept after it,
  and §6 reports what it hides.
- **The stratum is chosen so that a shortcut cannot produce the number.** The judging cells come from Paper III's grid
  (GSM8K, two policies per pair). Where the stronger policy is usually correct, a probe that merely identifies the
  policy would score well for the wrong reason. The primary stratum is the two policy pairs in which the stronger
  model is correct 49.0% and 43.9% of the time, so policy identity carries no information about correctness. The
  remaining 913 pairs are the secondary stratum and are reported only to be withdrawn.
- **The probe is admitted only under three controls, with a fourth added for this task.** Permuted labels (five
  draws) must collapse to the majority baseline; an eight-dimensional projection must fail to recover the probe; a
  hundred-example fit is reported; and a probe trained to predict *policy identity* from the same states must sit near
  chance on the primary stratum, or the balancing has failed.

## 4. Instruments

The judge is Qwen2.5-7B-Instruct at layer 18; 1,050 decisive pairs over 205 problems. The probe is a linear head fitted
by L-BFGS on the judge's hidden state at the decision, evaluated by five-fold cross-validation over *problems* — never
over pairs, since two pairs from one problem share its text — so that all 137 primary cells are scored. The first
evaluation held out 25% of problems, leaving 38 test cells; on that split the PCA-8 control fired (0.789 against a probe
at 0.763), and the registered rule withdraws a Δ whose control fires. But 38 cells were too thin to measure their own
controls: the permutation control spanned 0.42–0.66 against a 0.53 baseline, and a control whose spread is ±0.12 cannot
adjudicate a 0.03 gap. The thinness had been registered in advance as the arm's possible limit; the fix scores the same
cells, states and controls, changing how many are evaluated and not which count. Both evaluations are committed
(`et7_ee_holdout25.json`, `et7_ee_RESULT.json`) and the earlier one is reported here as the withdrawn first pass.

## 5. The measurement

| stratum | n | judge \(A\) | probe \(A^*\) | \(\Delta\) | baseline | permuted ×5 | PCA-8 | n = 100 | policy-identity probe |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| primary, balanced | 137 | 0.248 | 0.730 | **+0.482** | 0.518 | 0.52 · 0.50 · 0.41 · 0.50 · 0.48 | 0.657 | 0.730 | 0.577 |
| secondary, confounded | 913 | 0.288 | 0.838 | (0.550) | 0.540 | collapses | **0.834** | 0.816 | **0.826** |

On the primary stratum the reading written before the run applies: permutation collapses to the baseline, eight
dimensions do not suffice (0.657 against 0.730), and the policy-identity probe sits near chance, which is what the
balancing was for. The registered prediction — \(\Delta > 0\) and material — is confirmed. On the secondary stratum the
PCA-8 control matches the probe and the identity probe reaches 0.826: exactly the confound the stratification
predicted, and that Δ is withdrawn by the rule that admits the first. The subsample matching (0.730 at n = 100) is a
known property of this probe family on these states and is not on its own a flag.

## 6. What the gap is: abstention

The headline invites a misreading — a model worse than a coin — and the decomposition removes it:

| | pairs | correct |
|---|---:|---:|
| the judge commits | 46 | 34 (73.9%) |
| the judge ties | 91 | 0 by rule |
| all | 137 | 34 (24.8%) |

The probe's accuracy over all cells (73.0%) is, within a point, the judge's accuracy on the third of cells where it
commits (73.9%). The state carries what the judge would say if it spoke, on the two thirds of pairs where it declines
to. That is a more precise claim than \(\Delta = +0.482\) and a more useful one: the internal room the bound permits
is, on this judge, almost entirely in the abstentions. The tie rule is kept — scored as a coin, the judge would read
about 58%, and a judge that abstains two thirds of the time is useless as a selector whatever its state knows — but
the reader should hold both numbers.

## 7. What the gap is not

Two shallow readings of the state were tested and rejected. *Answer length:* "the longer answer is correct" holds on
43.1% of primary pairs, below chance, so a length-reading probe could reach about 57% by inverting the rule and not 73%.
*Policy identity:* the probe can tell which policy wrote an answer (0.577 on balanced pairs), but on those pairs the
stronger policy is correct 49% and 44% of the time, so that knowledge cannot produce 73% correctness. A third, the
probe reading the answer's surface form, is bounded by the same stratification and is the one E-D will test directly.

## 8. The same shape in Paper VIII

Paper VIII measured, at a different decision, the same gap from the other side. Its frozen model, asked to rank
enumerated candidates by its own log-probability, reached 20.3% of problems solved; a linear head read from its state at
the same decision reached a probe accuracy of 66.7% and, as a controller, 31.7%. There the gap was closed by a head fitted
on verified history — external bits — and the bridge arm found the model had no internal signal about its own success
at all. Here the gap is measured on a judge with no controller attached. The two papers do not explain each other; they
measure one phenomenon at two decisions, and the relation between them — whether the head of VIII is reading the
abstentions of VII — is a question for VIII-b's record and not a claim of either paper.

## 9. Routes above the level, registered

The bound partitions every route to judge improvement into internal (worth at most \(\Delta\)) and external (imports
information about \(T\)). Five were registered before any measurement: search amplification; asymmetric verification;
ensemble decorrelation (E-C); coherence constraints, the only genuinely internal route (E-D); and deferred external
signal. The synthesis prediction, also registered, is that every route that works smuggles external information, and
the paper prices the exchange rate of each. None is measured in this draft.

## Related work

Latent-knowledge probing [Kadavath et al. 2022; Burns et al. 2022; Azaria & Mitchell 2023] established that a model's
representations often carry more about correctness than its outputs express; this paper takes that observation as the
instrument, gives the distance a name and a bound, and measures it under controls that separate a readout from a
shortcut. Linear probes [Alain & Bengio 2016] are the decoder. LLM-as-judge evaluation [Zheng et al. 2023] and Paper
III measure how well a judge agrees with ground truth; this paper measures how much of what the judge knows reaches its
verdict. The bound rests on the data-processing inequality [Cover & Thomas 2006] and Blackwell's comparison of
experiments [Blackwell 1953]; the Löbian and incompleteness lineage is motivation only and carries no weight here. The
recursive-self-improvement debate [Good 1965 and after] is the argument this paper's coda would enter, after the
measurement and not before.

## 10. Registered predictions, scored

| prediction (registered before the run) | outcome |
|---|---|
| E-E: \(\Delta > 0\) and material — the judge's representations know more than its judgments express | **Hit, on one judge.** +0.482 on the balanced stratum with controls behaving; the confounded stratum withdrawn (§5) |
| E-E: \(\Delta\) shrinks with judge scale slower than \(A\) rises | **Not scored.** One judge measured; 1.5B ties on 98.5% of pairs (\(A = 0.000\), a format floor entering the curve only with its tie rate beside it); 14B and 32B running |
| E-D: coherence training raises \(q\) by a real, bounded amount, the residual being what only external signal fixes | **Not run.** §6 names its target: the abstentions |
| E-C: ensemble \(q\) exceeds the best single judge by variance reduction; vanishes where family errors correlate | **Not run** |
| cross-registration: E-D's realised gains must not exceed E-E's probed \(\Delta\) | **Not scored;** the bound it tests is now 0.482 on this judge |
| synthesis: every route that works smuggles external information | **Not scored** |

## 11. Limitations

One judge, one size, one benchmark, one form of the question. The probe lower-bounds \(A^*\); a better decoder could
raise it, and the gap can only be larger than measured, not smaller. The primary stratum is 137 pairs; the interval on
0.730 under five-fold CV is wide and is not yet printed because the fold-wise numbers are not in the artifact — the
next commit adds them. The tie rule is a choice, stated and kept. Nothing about a ceiling of self-improvement is claimed:
the bound is textbook and the number is one point. The formal statements are to be checked against the textbook forms
before any publication, and the coda on the wider debate is withheld until the scaling half lands.

## 12. Conclusion

A fixed system's room to improve from inside is the elicitation gap, and on the one judge measured the gap is large:
the judge reaches a quarter of pairs where its state reaches three quarters, and the difference is abstention. The
series' claim in this paper is exactly that and no more. What decides whether it matters beyond one model is the
scaling half, running, and whether E-D can turn silence into correct commitment — the only internal route the bound
allows, tested against the bound itself.

## Reproducibility

`experience/et7_ee_probe.py` and its pre-registration `docs/et7-ee-prereg.md`; the judging cells are Paper III's
(`judging/`, inputs listed in `docs/inputs-manifest.md`); results `experience/results/et7_ee_RESULT.json` (the
five-fold evaluation) and `et7_ee_holdout25.json` (the withdrawn first pass), each carrying the judge id, layer, cell
counts and every control. States are persisted per cell.

## References

- Alain, G. & Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* arXiv:1610.01644.
- Alsakka, L. (2026). *Efficient Thinking III: Efficient Judging.* This series.
- Alsakka, L. (2026). *Efficient Thinking VIII: Experience Priors.* This series.
- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* arXiv:2304.13734.
- Blackwell, D. (1953). *Equivalent comparisons of experiments.* Annals of Mathematical Statistics 24.
- Burns, C. et al. (2022). *Discovering Latent Knowledge in Language Models Without Supervision.* arXiv:2212.03827.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
- Good, I. J. (1965). *Speculations concerning the first ultraintelligent machine.* Advances in Computers 6.
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.* arXiv:2207.05221.
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.

## Appendix A. How This Was Found

Dates are 2026. The work was done by one experimenter (E) and one reviewer (R), pseudonymous here and belonging to no
institution named in this series.

1. **07 to 09-19.** The concept and the bound, registered; parked behind the human-rater papers; unparked when Paper
   III's judging cells and caches were found on a second checkout rather than regenerated.
2. **09-21.** E-E pre-registered before the first episode. Stage 1 found a confound none of the registered controls
   would catch — policy identity predicting correctness on most pairs — and the design was amended before the run
   (§3a: stratify on pairs where identity carries nothing). *Rule: a control that cannot fire is not a control.*
3. **09-22.** The first evaluation held out 25% of problems; the PCA-8 control fired on 38 cells whose own controls
   spanned ±0.12. Re-evaluated by five-fold CV over problems, as the pre-registration had allowed for; both kept. The
   result landed. Then the decomposition: E's analysis script carried a closing sentence written before the numbers —
   "a coin flip when it commits" — and the printed counts beside it read 73.9%. *Rule: print the counts next to the
   sentence; the sentence is the part that can be wrong without anything failing.* The 1.5B judge tied on 98.5% of
   pairs and its \(A = 0\) was recognised as format compliance before being read as a point.
