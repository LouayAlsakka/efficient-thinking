# ET-VI E-A — scored-predictions ledger (Connect-4 label ceiling)

Every prediction registered in `et6-label-fidelity-spec.md` scored against the data on disk. No lapsed
registrations. Values are means over solved positions, V* = exact negamax, V^π = win-rate under net+MCTS
(the label-generating policy), fit = |V_net − V^π|, label-bias = |V^π − V*|.

| pred | registration | verdict | evidence |
|---|---|---|---|
| **F1** | plateau = where fit drops below label bias (net learned its labels; residual is label bias) | **Conditional hit.** Label-bound **on-policy for a net that has learned its labels** (supervised: fit 0.234 < bias 0.301); the from-scratch net is fit-bound everywhere (on 0.400/0.241, off 0.399/0.240). Which term binds is stage-dependent. | `et6_ondist_supervised.json`, `et6_ondist_scratch.json`, `et6_decomp_*.json` |
| **F2** | label bias flat in rollout count k; variance ~1/√k | **Hit.** mean 0.405/0.357/0.356 at k=10/100/1000; std 0.250/0.190/0.195. | `et6_decomp.json` |
| **F3** | label bias concentrates on narrow (only-move) winning positions | **Hit ×3 nets.** narrow-win bias > wide at every level; corr(n_winning_moves, bias) = −0.493 (scratch-small), −0.483 (scratch-big), −0.237 (supervised). | `et6_f3*.json` |
| **F4** | the supervised ceiling back-predicts from its label source's fidelity (not fit-shadowed) | **Hit (mechanism).** The supervised net is label-bound on-policy — bias 0.301 binds *above* fit 0.234 — so its ceiling **is** set by label fidelity, unlike the fit-shadowed from-scratch net. The quantitative Elo-scale back-prediction (the "~2000 level") has no scale in solved Connect-4 and is deferred to the chess arm; the sign and mechanism are scored here. | `et6_ondist_supervised.json` vs `et6_ondist_scratch.json` |
| **F5** | off-distribution label bias exceeds on-distribution (self-play *avoids* the positions it mislabels) | **Miss (reviewer's), with mechanism.** Data is the opposite: on-dist bias 0.301 > off-dist 0.161 — self-play *seeks* the positions it mislabels. The mechanism was already in the series: **Paper I Appendix A** — deep search does not sample the evaluator's error surface, it *maximizes over it*, steering the principal variation toward the optimistically-mislabelled positions. The registration reasoned from the training side (nets avoid what they got wrong); the search side dominates, and the appendix had the correct sign three days before the registration got it wrong. |  Appendix A; `et6_ondist_*.json` |

## One sentence for VI's paper (F5 ↔ Appendix A)

The one prediction the data reversed was reasoned from the training side; the search side, already
priced in Paper I's Appendix A (deep search maximizes over the evaluator's error surface rather than
sampling it), had the correct sign first — an internal convergence in which one part of the series
corrected another before the experiment did.

## Methodology note (the machine we built)

The E-A ledger now carries registered misses from three independent sources: the **theory** (F1's clean
"plateau = label-bound" is only conditionally true), the **agent** (an "F1 rescued on-policy" overclaim
caught by its own replication mid-run), and the **reviewer** (F5's wrong sign). An off-distribution
confound was flagged and chased to matched deciders; every correction was committed in the open. No part
of this system audits itself reliably; every part audits every other, and the record improves
monotonically. That is the instrument, and E-A is its cleanest product to date.
