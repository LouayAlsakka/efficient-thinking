# Efficient Thinking VII: The Elicitation Gap — What a Judge Knows but Doesn't Say (concept)

*Status: concept, parked behind ET-IV for its aesthetic arm; the math-domain arms (E-E, E-D, E-C)
need only the ET-III harness and ground-truth caches. One sentence: internal self-improvement of
a judge is provably bounded by the information already in its state — the elicitation gap
Δ = A* − A — and this paper states the bound, measures Δ, and prices every route to judge
improvement, internal and external, taking no advance position on whether Δ is large. The closing coda states what the answer means
for the recursive self-improvement debate — after the measurement, not before it.*


## Theory: the Internal-Improvement Bound (proposition; final home VI or a later paper, TBD)

**Setting.** Let T be the external target the judge is meant to track (ground-truth values,
correctness, a rater's taste — assumed fixed; the nonstationary case is ET-V's). Let D be all
external data the system has ever received, W its internal state (weights, memories), and J its
expressed judgments. Define a procedure as *internal* iff it updates the state as W' = φ(W, R)
where R is internal randomness carrying no information about T beyond W — no new channel whose
statistics depend on T. Coherence training is internal (logical structure imports nothing about
T). Self-play in games is NOT internal: terminal outcomes are a T-correlated channel supplied by
the rules — which is precisely why game loops can climb at all, and the taxonomy classifies them
under external routes.

**Proposition (Internal-Improvement Bound).** For any judge and any internal procedure:
(i) I(T; W') ≤ I(T; W): internal computation cannot increase the information the state carries
about the target (data-processing inequality over the Markov chain T — D — W — W').
(ii) Let A(W) be the judge's realized accuracy against T under any proper scoring of its
judgments, and A*(W) the Bayes-optimal accuracy achievable by the best possible decoder of T
from W. Then any internally-reachable judge satisfies A(W') ≤ A*(W): realized performance can
rise toward the information ceiling already in the state, never through it (Blackwell: garbling
cannot improve Bayes risk; quantitatively, Fano-type bounds tie achievable accuracy to I(T; W)).

**Definition (elicitation gap).** Δ(W) := A*(W) − A(W) — the held-but-misprocessed information.
The Proposition says internal self-improvement is worth at most Δ, and everything above A*(W)
must be imported. This is the two-term decomposition made formal: the missing-information term is
untouchable from inside as a theorem; the size of Δ is an empirical quantity, and the series
takes no advance position on it.

**Measurability.** A*(W) is not directly observable, but it is lower-boundable: train small
external probes on the judge's internal representations to predict T directly (in verifiable
domains, candidate correctness), and compare probe accuracy against the judge's own judgment
accuracy. Probe − judgment ≥ 0 estimates Δ from below; the latent-knowledge literature [Kadavath
et al. 2022; Burns et al. 2022; Azaria & Mitchell 2023] suggests it is often substantially
positive. This yields a cross-experiment consistency registration: E-D's realized coherence gains
must not exceed the Δ measured by probing — if they do, either the internality of E-D's procedure
or the probe methodology is broken, and finding out which is itself the result.

**Relation to Gödel.** The incompleteness lineage (Gödel; Tarski's undefinability; Löb's theorem
and the Löbian obstacle to self-trusting agents) is suggestive precedent that systems cannot
fully certify themselves from within, and is cited as motivation only; the load-bearing
mathematics here is the data-processing inequality and Blackwell monotonicity, which apply to
statistical judges directly. All formal statements above are standard results applied to this
setting and are to be verified against textbook forms before publication.

## Taxonomy of above-level signal routes (each gets measured, not asserted)

1. **Search amplification** — distill(search(π)) targets; gain = search-over-policy margin;
   bounded, measurable as a curve over level.
2. **Asymmetric verification** — domains where checking is cheaper than generating (terminal
   rules, proofs, tests, tactics under deep search); signal is external by construction.
3. **Ensemble decorrelation** — multi-family judge consensus; removes idiosyncratic variance,
   cannot remove shared training-distribution bias (conventionality); helps at the noise floor,
   fails at the taste frontier.
4. **Coherence constraints** — intransitivity and invariance violations are provable error with
   zero external labels; logic acts as the oracle. The only genuinely internal route on this list.
5. **Deferred external signal** — time, engagement, survival of artifacts; external, slow,
   confounded.

Synthesis prediction: every route that works smuggles external information (rules, logic,
checkability, humans); the paper prices the exchange rate of each in the series' idiom.

## Experiments

- **E-E — The elicitation gap, probed (the theory's measurement arm).** On GSM8K/MATH judging
  cells with known ground truth: train linear/small probes on the judge's hidden states to
  predict candidate correctness; compare probe accuracy (lower bound on A*(W)) against the
  judge's pick accuracy (A(W)). Registered: Δ > 0 and material — the judge's representations
  know more than its judgments express — and Δ shrinks with judge scale slower than A rises.
- **E-D — Coherence bootstrapping (the live wire).** Train a judge against its own intransitivity
  and order-invariance violations (no external labels); measure q before/after against ground
  truth (math) and against the human rater (ET-IV). Registered: consistency training raises q by a
  real, bounded amount — the bound being the share of judge error that is incoherence rather than
  shared bias — and the residual after coherence is exactly the part only external signal can fix.
- **E-C — Ensemble judge decorrelation (ET-III harness + Llama).** Multi-family judge consensus q
  vs single-judge q vs majority on GSM8K/MATH cells. Registered: ensemble q exceeds the best
  single judge by variance reduction; the gap vanishes on items where family errors correlate;
  aesthetic-domain version (with ET-IV's human q) shows the conventionality ceiling.

Cross-experiment registration: E-D's realized coherence gains must not exceed E-E's probed Δ;
a violation indicts either E-D's internality or the probe methodology, and identifying which is
itself the result. The aesthetic-domain arm (Δ against a human rater's taste) enters only after
ET-IV delivers measured human-q machinery.

## Honest scope

Nothing here presumes the answer. A large measured Δ is evidence that judges can be meaningfully
self-improved; a small one, evidence that the bottleneck is effectively external. Either result
is the finding, reported at the same prominence whichever way it falls. The formal statements are
standard results (DPI, Blackwell) applied to this setting, to be verified against textbook forms
before publication. The paper asserts the bound, measures the margin, and confines the broader argument to the
closing coda, which is positioned after the results because it is licensed by them.

## Coda: what the measurement means for the AGI debate

The public argument about recursive self-improvement — from Good's intelligence explosion through
the modern debate — largely assumes that an AI able to rewrite its own code is the decisive
threshold. The position motivating this paper, stated as carefully as Papers I–III license it:
code self-modification improves the extractor, and the extractor was never what bound capability
in anything we measured; the gate to any explosive trajectory is self-improvable evaluation
signal, and whether that gate can open from inside is exactly the open question this paper
measures (the elicitation gap Δ, below). If Δ is small and the external routes dominate, then
"AI rewriting AI" yields productivity, not takeoff — capability growth stays throttled by the
rate at which external evaluation signal is imported, a rate the series prices paper by paper.
If Δ is large, a real internal margin exists and the fast-takeoff view gains measured support.
Either result moves a debate currently conducted almost entirely in words onto priced ground.
This paragraph is positioning, not a claim: the papers assert the bottleneck, measure the margin,
and stop there.

Placed here rather than in the introduction deliberately: the argument above is only as strong as
the measured Δ that precedes it, and a reader should meet the number before the claim. Whichever
way Δ falls, the debate it enters — conducted for decades in thought experiments — acquires its
first priced quantity, and the series' contribution is the pricing, not the position.
