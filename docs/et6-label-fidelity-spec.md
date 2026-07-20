# ET-VI E-A/E-B — Label-Fidelity Instrument: spec and registrations

Registered before execution (commit timestamp = registration). Scope-rule note: this spec belongs to ET-VI
(The Label Ceiling), which is games-only and unparked; it runs in idle machine time behind the
ET-III external-validation runs and the ET-IV machine queue.

## The claim under test

A value net's ceiling is the fidelity of its labels. Two plateaus, one instrument:
- Self-play plateaus (ET-I §3: chess 254–384 / 1214; C4 +400) because its labels are V^π —
  win-rates under its own level of play — which are structurally biased toward 0.5 on positions
  whose advantage requires skill the policy lacks.
- The supervised net's ~2000/2150 ceiling reflects its external label source's fidelity plus
  capacity — same measurement, different label source.

## Quantities (per position s, per training stage t)

- V_net(s,t): the network's value output.
- V^π(s,t): empirical rollout label — win-rate over k self-play games from s at stage t.
- V*(s): oracle value. Connect-4: exact solver. Chess: Stockfish WDL at fixed high depth/nodes
  (proxy oracle; residual SF error is noise at a ~1600-Elo gap, and the C4 arm anchors the
  method exactly).

Decomposition: total error |V_net − V*| = fit error |V_net − V^π| ⊕ label bias |V^π − V*|.

## Registered predictions

- **F1 (the plateau mechanism).** Stage-wise, Elo flattens at the crossover where fit error drops
  below label bias: the net plateaus when it has fully learned its own labels, and all remaining
  error is in the labels. One figure: Elo(t), fit(t), bias(t) on a shared axis.
- **F2 (irreducibility).** At fixed stage, label bias is flat in rollout count k (10/100/1000)
  while variance shrinks as 1/√k. More self-play games sharpen the wrong number.
- **F3 (conversion-skill bias map).** Label bias correlates with position narrowness, measured
  three ways: (a) count of moves within ε of best (only-move index), (b) oracle eval shift from
  shallow to deep search (tactical depth-sensitivity), (c) |V* − empirical outcome under current
  play|. Wide advantages label accurately at any level; narrow ones compress toward 0.5.
- **F4 (both plateaus, one law).** The supervised net's label source, scored by the same
  instrument against the oracle, has a bias floor consistent with the ~2000-level ceiling —
  i.e., the supervised ceiling back-predicts from label fidelity the same way the self-play
  plateau does.
- **F5 (distribution gap).** Label bias on off-distribution positions (fixed benchmark suite)
  exceeds on-distribution bias at late stages: self-play stops visiting exactly the positions it
  most mislabels.

## Protocol

1. **Connect-4 first** (exact oracle, validates the instrument). Checkpoints every ~5 iterations
   of the ET-I §3 loop (regenerate if not saved — cheap). Position sets: 500 on-distribution
   (sampled from that stage's games) + 200 fixed off-distribution. k ∈ {10, 100, 1000} for F2 on
   a 100-position subset; k = 100 elsewhere. Per-position records committed (the per-problem
   lesson: log everything the first time).
2. **Chess second.** Stockfish with WDL output at fixed nodes (pick once, record the setting);
   checkpoints from the from-scratch and warm-start self-play runs; the supervised net and its
   label source scored for F4. Narrowness metrics computed by SF multipv at two depths.
3. Analysis scripts commit with results; every figure built from the JSONs at build time.

## What this is not

Not a cure: the instrument measures why closed-loop labels cap the evaluator; it does not mint
above-level labels. The cure candidates are ET-VI's taxonomy (search amplification, coherence
constraints, external signal), each priced separately. If F1's crossover does NOT align with the
plateau — if Elo flattens while fit error is still large — the capacity/optimization explanation
gains ground against the label explanation, and that miss is the headline.
