# Efficient Thinking VII: The Elicitation Gap
## What a fixed system knows that it does not say — a bound, and its first measurements
> **STATE 2026-09-22: DRAFT v0.3 — the headline number is WITHDRAWN and being re-measured under a registered definition.** The tie-scored gap (§5, §6, §9) is a scoring artefact: on the cells where the 7B judge commits, the probe and the judge agree within a point, and the 48 points are ties scored wrong by a rule. Separately, the registered subsample control never ran on the primary stratum — its cap exceeded the training set, so it refitted the probe on all of its own rows — and the tie-scored gap was therefore never under that control. Pre-registration §3b (committed before the run) redefines \(A\) as the judge's forced preference — logit A against logit B, no tie — with both outcomes pre-stated: either the forced judge meets the probe and the registered claim fails, or a gap survives the objection. The forced-choice run at 1.5B/7B/14B/32B and the tie-restricted probe are requested and not on disk. The tie-scored numbers stay in this draft as what was measured, labelled. E-C and E-D are registered and not run; the bound's formal statements are to be verified against textbook forms before publication. Every number below is on disk under `experience/results/` with its command; the reading rules were committed before the run (`docs/et7-ee-prereg.md`, with its §3a amendment). Nothing here is a law; §10 says what is a finding and what is a registration.

**Louay Alsakka** · September 22, 2026 · *draft v0.3*

## Abstract

A fixed system — a model whose weights and inputs do not change — can improve its realised performance only toward the
information its state already carries about the target, never through it. That is a theorem (the data-processing
inequality and Blackwell monotonicity applied to a statistical judge), and it is trivial unless the distance between
what the state carries and what the system expresses is measured. This paper defines that distance, the *elicitation
gap* \(\Delta = A^* - A\), where \(A\) is the judge's realised accuracy and \(A^*\) the accuracy of the best decoder of
its state, and measures a lower bound on it: a linear probe on the state scores at most \(A^*\), so
\(\Delta \ge A_{\text{probe}} - A\). On the stratum of a judging task where the policy that wrote an answer carries no
information about its correctness, a 7B judge scored A/B/tie picks the correct answer 24.8% of the time while a probe
on its own hidden states reaches 73.0% (five-fold interval over problems [0.62, 0.84]). That 48-point number is
withdrawn as a headline: the judge declines to choose on 66.4% of pairs and is 73.9% correct when it commits, so on
the cells where it speaks the probe and the judge agree, and the difference is the tie rule. The gap under the
definition this paper will publish — the judge's forced preference between A and B at the decision, no tie — is
registered (§3b of the pre-registration) and not yet measured; both outcomes are pre-stated and one of them fails the
paper's claim. Across judge sizes the picture
sharpens: from 1.5B to 14B the probe reads about the same amount from the state (0.68, 0.73, 0.67) while the judge's
tie rate falls from 98.5% to 26.3% and its accuracy rises from 0.7% to 56.2%: under the tie rule the scaling curve is
a tie-rate curve, and the registered prediction that the gap shrinks slower than accuracy rises fails for that reason.
Two shallow explanations — answer length and policy identity — are tested and rejected. What the bound licenses is
narrower than "the room is large": \(\Delta\) is an upper bound on what any internal procedure can recover, and the
probe that measures it was fitted on labels no internal procedure has, so nothing here shows any of it is reachable
from inside. That is E-D's question, and the paper's honest state is a well-instrumented design whose first number
was a scoring artefact, now being re-measured under the definition it will publish.

## Results at a glance

| finding | measurement | where |
|---|---|---|
| the bound | internal computation cannot raise \(I(T;W)\); realised accuracy rises toward \(A^*(W)\), never through it | §2 |
| the gap, defined | \(\Delta = A^* - A\); the probe lower-bounds \(A^*\), so \(\Delta \ge A_{\text{probe}} - A\); \(A\) is the judge's own pick with a tie scored wrong | §3 |
| the gap, tie-scored — WITHDRAWN as headline | judge 24.8% (A/B/tie), probe 73.0% [0.62, 0.84] on 137 balanced pairs; on the 46 committed cells judge 73.9% ≈ probe; the difference is the tie rule | §5, §6 |
| the gap, forced-choice — registered, not measured | \(A\) = logit A vs logit B, no tie; same states, same folds; outcomes pre-stated in prereg §3b | §3, §10 |
| the controls | permuted labels 0.41–0.52 (baseline 0.518); PCA-8 0.657 is inside the probe's fold interval — not established; the subsample control did not run (cap 200 > 99 training rows, so it was the probe itself); policy-identity probe 0.577 | §5 |
| the confounded stratum, withdrawn | 913 pairs where policy identity predicts correctness: PCA-8 matches the probe, identity probe 0.826 — not a finding | §5 |
| what the gap is | abstention: ties on 66.4% of pairs, 73.9% correct when committing, the probe over all pairs 73.0%; the tie-restricted probe number is requested, not yet measured | §6 |
| what it is not | answer length (43.1%, below chance); policy identity (0.577, uninformative by construction) | §7 |
| the same shape in Paper VIII | the frozen model's own log-probability preference 20.3% where a probe on its state reads 66.7% | §8 |
| scale, tie-scored | probe flat across 1.5B–14B (0.68 · 0.73 · 0.67); tie rate 98.5% → 66.4% → 26.3%; the curve is a tie-rate curve; the second registered prediction fails; to be re-measured under forced choice | §9 |
| not yet measured | forced-choice A at four sizes; E-C; E-D | §10, §12 |

## 1. The question

Efficient Thinking I–III measured what search buys a fixed intelligence and found its ceiling: search extracts what the
evaluator already contains, and only information from outside raises the ceiling. Paper VIII found that verified
history moves the quality–compute frontier of a frozen model. Between those two sits a question the series had
stated and not measured: how much can a fixed system get from *inside* — from re-processing what its own state already
holds, with no new information? The public argument about recursive self-improvement is conducted almost entirely in
words on exactly this question. This paper gives it a definition, a bound, and its first numbers.

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

**Definition.** The elicitation gap is \(\Delta(W) = A^*(W) - A(W)\): held and unexpressed information. Internal
self-improvement is worth at most \(\Delta\); everything above \(A^*\) must be imported.

**What \(\Delta\) licenses, and what it does not.** \(\Delta\) is an upper bound on internal room, not a
measure of it. The decoder that estimates \(A^*\) in this paper is fitted on labels of \(T\); an internal procedure
has no access to \(T\) and cannot find that decoder the way the experimenter did. Nothing in the bound, and nothing in
this paper's measurements, shows that any part of \(\Delta\) is reachable from inside. Whether some fraction is —
recovered by an internal procedure without labels — is exactly what E-D tests (§11), and a large \(\Delta\) must
not be read as "a fixed model could gain that much from inside". The proposition is standard results applied to this
setting; the size of \(\Delta\) is empirical, and the series took no advance position on it.

## 3. Measuring the gap

\(A^*\) is not observable, but it is bounded from below: any decoder trained on the judge's internal representation to
predict \(T\) scores at most \(A^*\), so with \(A_{\text{probe}}\) the accuracy of a linear probe,

\[ A^* \ge A_{\text{probe}}, \qquad \Delta = A^* - A \ge A_{\text{probe}} - A . \]

Every gap this paper reports is that lower bound; a better decoder can only widen it. Three rules make the bound a
measurement rather than a story.

- **\(A\) is the judge's forced preference.** The definition first registered — the judge's A/B/tie answer with a
  tie scored wrong — was measured and is reported in §5–§6 and §9 as what it is: a gap between the state and what one
  prompt format extracts. It is not what "expressed" should mean for a bound about expression, and prereg §3b, committed
  after that reading and before any further run, replaces it: \(A\) is the log-probability of the token A against the
  token B at the decision position, no tie available, on the same cells and states. Two prompt forms are read (tie
  option removed, primary; three-way prompt with tie ignored, check), both orderings scored. The tie-scored \(A\) is
  kept as a secondary reading: as a selector the judge is useless where it abstains.
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

The judges are Qwen2.5-Instruct at 1.5B, 7B and 14B, read at layer 18 (7B) and the corresponding depth at the other
sizes; 1,050 decisive pairs over 205 problems, 137 in the primary stratum. The probe is a linear head fitted by L-BFGS
on the judge's hidden state at the decision, evaluated by five-fold cross-validation over *problems* — never over
pairs, since two pairs from one problem share its text — so that all 137 primary cells are scored and the fold is the
independent unit for the interval. The first evaluation held out 25% of problems, leaving 38 test cells; on that split
the PCA-8 control fired (0.789 against a probe at 0.763), and the registered rule withdraws a gap whose control fires.
But 38 cells were too thin to measure their own controls: the permutation control spanned 0.42–0.66 against a 0.53
baseline, and a control whose spread is ±0.12 cannot adjudicate a 0.03 gap. The thinness had been registered in
advance as the arm's possible limit; the fix scores the same cells, states and controls, changing how many are
evaluated and not which count. Both evaluations are committed (`et7_ee_holdout25.json`, `et7_ee_RESULT.json`) and the
earlier one is reported here as the withdrawn first pass.

## 5. The measurement

| stratum | n | judge \(A\) | probe \(A_{\text{probe}}\) | \(\Delta \ge\) | baseline | permuted ×5 | PCA-8 | subsample (min(200, train)) | policy-identity probe |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| primary, balanced | 137 | 0.248 | 0.730 [0.62, 0.84] | **0.482** | 0.518 | 0.52 · 0.50 · 0.41 · 0.50 · 0.48 | 0.657 | did not run (99 of 99 rows) | 0.577 |
| secondary, confounded | 913 | 0.288 | 0.838 | (0.550) | 0.540 | collapses | **0.834** | 0.816 (200 of 687) | **0.826** |

The interval on the probe is across the five problem-folds (0.846 · 0.800 · 0.704 · 0.621 · 0.680; sd 0.092, t with
four degrees of freedom), not a binomial over 137 cells, which would overstate the precision of cells that share
problems. Read against that interval the controls are weaker than v0.1 said. Permutation collapses to the baseline.
PCA-8 at 0.657 sits inside the probe's interval [0.62, 0.84]: the probe's excess over eight dimensions is not
established at this n. The subsample control did not run on the primary stratum. Its size is `min(200, training
rows)`, and the held-out evaluation had 99 training rows, so the "subsample" was all 99 rows in a different order: the
same fit, the same 0.730, by construction and at any signal level. v0.1 printed it as "n = 100" and called the match a
known property; v0.3 first read it as a gate that fired; both were wrong, and the truth is that nothing ran. So the
tie-scored gap was never under a subsample control at all, and the re-measurement carries one that is a real draw
(n = 30 against about 110 training cells per fold) and prints the size it used beside the training-set size. On the
secondary stratum the draw was real (200 of 687). The policy-identity probe sits near chance, which is what the
balancing was for. On the
secondary stratum the PCA-8 control matches the probe and the identity probe reaches 0.826: exactly the confound the
stratification predicted, and that gap is withdrawn too. What survives §5 unconditionally is the design: the
balanced stratum is where a readout can be told from a shortcut, and the 913-pair stratum shows why.

## 6. What the gap is: abstention

The headline invites a misreading — a model worse than a coin — and the decomposition removes it:

| | pairs | correct |
|---|---:|---:|
| the judge commits | 46 | 34 (73.9%) |
| the judge ties | 91 | 0 by rule |
| all | 137 | 34 (24.8%) |

The probe's accuracy over all cells (73.0%) is, within a point, the judge's accuracy on the third of cells where it
commits (73.9%). On the cells where the judge speaks, then, the gap is about −0.01: the 48 points of §5 are the 91
ties scored wrong by a rule. That is the reason the tie-scored number is withdrawn as a headline (§3), and it is also
the observation that makes the forced-choice re-measurement decisive — if the judge, denied the tie, lands near the
probe, the abstention was format and the paper's claim fails; if it lands well below, the gap survives the objection.
The 73.0 ≈ 73.9 match is an equality of aggregates. The statistic that tests it directly — the same fold predictions
split by whether the judge tied on that cell, no new fit — is now on disk (`et7_ee_tie_split.json`):

| judge | tie cells | probe on ties | committed cells | probe on committed | judge on committed |
|---|---:|---:|---:|---:|---:|
| 1.5B | 135 | 0.674 | 2 | (1.000, two cells) | (0.500, two cells) |
| 7B | 91 | 0.714 | 46 | 0.761 | 0.739 |
| 14B | 36 | 0.833 | 101 | 0.614 | 0.762 |

At 7B the probe reads the abstained cells at 71.4%, essentially the rate it reads the committed ones (76.1%): the
state carries the answer where the judge stays silent, nearly literally. Under §3b that is descriptive — it says what
the tie-scored gap was made of, and it does not license a claim, because the judge was never asked to choose on those
cells. Two rows carry warnings with them. The 1.5B committed row is two cells and means nothing. The 14B row inverts —
the probe reads the tied cells better (0.833) than the committed ones (0.614), and on the committed cells the *judge*
beats the probe (0.762 against 0.614) — but the tied row is 36 cells with no interval, a 22-point difference on 36
cells is about eight cells, and it is a flag for a later arm and not a finding. The 14B committed row is worth holding
in mind for the forced-choice outcome: where the largest judge speaks, it already out-reads a linear probe on its
own state. The tie rule is kept either way — scored as a
coin, the judge would read about 58%, and a judge that abstains two thirds of the time is useless as a selector
whatever its state knows — but the reader should hold both numbers.

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

## 9. Scale, tie-scored: a tie-rate curve

*Measured under the withdrawn definition; kept because it is what was measured, and because it is the reason the
definition changed. It is re-measured under forced choice before the 32B point is read.*

The concept registered a second prediction: that \(\Delta\) shrinks with judge scale more slowly than \(A\) rises. Three
sizes of one family, the same 137 balanced cells, the same five folds over problems:

| judge | tie rate | judge \(A\) | probe \(A_{\text{probe}}\) | \(\Delta \ge\) | PCA-8 |
|---|---:|---:|---:|---:|---:|
| 1.5B | 98.5% | 0.007 | 0.679 | 0.672 | 0.635 |
| 7B | 66.4% | 0.248 | 0.730 | 0.482 | 0.657 |
| 14B | 26.3% | 0.562 | 0.672 | 0.109 | 0.584 |

From 1.5B to 14B the judge's accuracy rises by 0.555 and the gap shrinks by 0.563: in lockstep, slightly faster. The
prediction fails, and it had to. The probe reads about the same amount from every state — 0.68, 0.73, 0.67, within
0.06 of each other with no trend across a tenfold parameter range — so \(A^*\)'s lower bound is flat and the gap moves
exactly opposite to \(A\). The concept had assumed scale would raise \(A^*\) as well; on this range the linear
readout does not. What scale does is collapse the tie rate, from 98.5% to 26.3% — which is to say that under the
tie rule this is a tie-rate curve and not a curve of the gap. Whether the gap itself closes with scale is the question
the forced-choice re-measurement answers, and it is the one that would make this a paper: a forced-choice gap that
stays at fifteen points or more from 7B through 32B while accuracy rises would be a clean claim nobody has made, that
elicitation loss does not close with scale; a gap that shrinks toward zero by 32B is the modest finding that small
models do not say what they know and large ones do. Two cautions. The 1.5B
row is a format floor — a judge that ties on 98.5% of pairs is not measured at \(A = 0.007\), it is declining the
task — and it enters the curve only with its tie rate beside it. And "flat" is three points with fold intervals of
about ±0.1; the 32B point, pending, is what would make the shape a claim rather than a reading.

The consequence for the internal route is sharper than the cross-registration the concept wrote. E-D trains a judge
against its own incoherence, with no external labels. If the headroom at every scale is abstention rather than
knowledge, then E-D's target is the abstention and its ceiling is \(A^*\), which on this range does not rise with size:
a bigger judge does not raise the ceiling, it starts nearer to it.

## 10. Registered predictions, scored

| prediction (registered before the run) | outcome |
|---|---|
| E-E: \(\Delta > 0\) and material — the judge's representations know more than its judgments express | **Withdrawn under the tie-scored definition** (the gap was the tie rule, §5–§6; and its subsample control never ran). **Unscored under the forced-choice definition** registered in prereg §3b: hit if the fold interval on (probe − forced judge) excludes zero with lower bound ≥ 0.05; fails if the forced judge meets the probe |
| E-E: \(\Delta\) shrinks with judge scale slower than \(A\) rises | **Fails under the tie-scored definition** — the curve is a tie-rate curve (§9). Re-scored under forced choice at four sizes |
| E-D: coherence training raises \(q\) by a real, bounded amount, the residual being what only external signal fixes | **Not run.** §6 and §9 name its target: the abstentions, under a flat ceiling |
| E-C: ensemble \(q\) exceeds the best single judge by variance reduction; vanishes where family errors correlate | **Not run** |
| cross-registration: E-D's realised gains must not exceed E-E's probed \(\Delta\) | **Not scored;** the bound it tests is the forced-choice \(\Delta\), not yet measured |
| synthesis: every route that works smuggles external information | **Not scored** |

## 11. Routes above the level, registered

The bound partitions every route to judge improvement into internal (worth at most \(\Delta\)) and external (imports
information about \(T\)). Five were registered before any measurement: search amplification; asymmetric verification;
ensemble decorrelation (E-C); coherence constraints, the only genuinely internal route (E-D); and deferred external
signal. The synthesis prediction, also registered, is that every route that works smuggles external information, and
the paper prices the exchange rate of each. None is measured in this draft.

## Related work

The bare fact that a probe on hidden states beats the model's own output is established: Burns et al. [2022] report
CCS on hidden states beating zero-shot prompting on the same model, and Kadavath et al. [2022] report P(IK) calibration
exceeding expressed self-assessment; Azaria & Mitchell [2023] read truthfulness from the state directly. This paper
does not claim that fact. What it adds is narrower: a stratification that kills the policy-identity shortcut — the
withdrawn 913-pair stratum shows that without it a probe scores well for the wrong reason — a decomposition of the
output-side number into abstention and error, and a definition of \(A\) under which the comparison is between the
state and what the system can express rather than what one prompt format extracts. Linear probes [Alain & Bengio 2016] are the decoder. LLM-as-judge evaluation [Zheng et al. 2023] and Paper
III measure how well a judge agrees with ground truth; this paper measures how much of what the judge knows reaches its
verdict. The bound rests on the data-processing inequality [Cover & Thomas 2006] and Blackwell's comparison of
experiments [Blackwell 1953]; the Löbian and incompleteness lineage is motivation only and carries no weight here. The
recursive-self-improvement debate [Good 1965 and after] is the argument this paper's coda would enter, after the
measurement and not before.

## 12. Limitations

One model family, one benchmark, one form of the question, three sizes, 137 pairs from two policy pairs — enough for a
first arm, not for the word "material" without an interval, and the word is not used of a measured number in this
draft. The probe lower-bounds \(A^*\); every gap here is a lower bound and a better decoder can only widen it, so
"flat" in §9 means the *linear* readout is flat. The fold interval on the 7B probe is [0.62, 0.84]; the scaling shape
rests on three points with intervals of that width, and under the tie rule it is a tie-rate curve. The 66% tie rate on
decisive GSM8K pairs is high enough that the prompt itself is checked before the forced-choice runs. The headline definition has changed once, after measurement, by a registered
amendment that pre-states both outcomes; that is recorded in Appendix A and the reader is entitled to weigh it. Nothing about a ceiling of self-improvement is claimed: the
bound is textbook and the numbers are three points on one task. The formal statements are to be checked against the
textbook forms before any publication, and the coda on the wider debate is withheld until the 32B point and the
tie-restricted statistic land.

## 13. Conclusion

As drafted this is a well-instrumented design whose first number was a scoring artefact. A fixed 7B judge, allowed to
abstain, abstains on two thirds of comparisons and is as accurate as a probe on its own state where it commits; the
48-point gap was the abstentions, and it is withdrawn. What the paper will claim is the gap under forced choice, at
four sizes, with a fold interval, and both of its outcomes are written down before the run: the state and the output
agree and the claim fails, or a gap survives the objection and its scaling shape is the result. Beyond that lies the
experiment the series has pointed at from the start: whether an internal procedure, with no labels, recovers any
measurable fraction of the gap. Even a small fraction, cleanly shown, would be the first quantified exchange rate
between internal compute and elicited accuracy. That is E-D, it is the only internal route the bound allows, and it is
tested against the bound itself.

## Reproducibility

`reasoning/et7_ee_probe.py` (the committed script is the held-out form; the five-fold evaluation reuses its cached
states and fit) and its pre-registration `docs/et7-ee-prereg.md`; the judging cells are Paper III's
(`reasoning/et7_ee_cells.json`, built by `et7_ee_cells.py`); results `experience/results/et7_ee_RESULT.json` (the
five-fold evaluation at 7B), `et7_ee_foldwise.json` (the fold-wise probe numbers and interval), `et7_ee_scaling.json`
(1.5B, 7B, 14B), `et7_ee_tie_split.json` (the fold predictions split by the judge's tie) and `et7_ee_holdout25.json` (the withdrawn first pass), each carrying the judge id, layer, cell counts
and every control. States are persisted per cell.

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
   pairs and its \(A = 0.007\) was recognised as format compliance before being read as a point.
4. **09-22, later.** Draft v0.1 wrote \(\Delta = 0.482\) where the probe only bounds \(A^*\) from below; an outside
   reader caught it and the draft now says \(\Delta \ge\) throughout. The same reader noted that "probe over all ≈
   judge when committing" is an equality of aggregates, not a demonstration on the tie cells; the tie-restricted
   statistic was requested and §6 was softened until it lands. The 14B point arrived the same day, the second
   registered prediction failed, and the reason it failed — a flat probe — became §9. *Rule: when a prediction fails,
   the arithmetic of why is the result.*
5. **09-22, later still.** A second outside reader put the objection in one sentence: on the 46 cells where the
   judge commits the probe and the judge agree, so the 48 points are ties scored wrong by a rule — a gap between the
   state and a prompt format, not between the state and what the system can express. Accepted; the tie-scored gap
   was withdrawn on that ground. R also wrote that the registered subsample control had fired at 7B and been waved
   through; E checked the code within the hour and it was wrong in the other direction — the control's cap (200)
   exceeded the 99 training rows, so it had refitted the probe on all of its own rows and had never run on that
   stratum in any evaluation. The "n = 100" in v0.1's table was a number never drawn. Prereg §3b was committed before
   any new run: \(A\) is the forced preference, both outcomes pre-stated, controls sized to the n and printed with
   their sizes. *Rule: a subsample control whose size is not printed beside the training-set size is not a control —
   print `sub n / train n` or it can silently be the identity. And: the definition of the thing measured is registered
   with the same care as the threshold.*
