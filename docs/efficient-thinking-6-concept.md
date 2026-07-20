# Efficient Thinking VI: The Label Ceiling — Why Self-Play Plateaus (concept)

*Status: concept; games-only, no dependency on IV/V, so UNPARKED for idle machine time behind the
ET-III validation runs and the ET-IV machine queue. One sentence: a value net's ceiling is the
fidelity of its labels — self-play plateaus because its labels are V^π, biased toward its own
level exactly on the positions where improvement lives; this paper measures that bias against
exact and proxy oracles, stage by stage, and explains Paper I §3's plateau mechanistically.*

## Thesis

Self-play rollout labels estimate V^π (value under current play), not V* (value under best play).
Sample count reduces variance only; the bias is structural and concentrates on positions whose
advantage requires skill the policy lacks — rollouts misplay them, compressing their measured
value toward 0.5 precisely where correct labels would teach the most. Two plateaus, one law: the
self-play plateaus (ET-I §3) cap at V^π fidelity; the supervised net's ~2000 ceiling reflects its
external label source's fidelity. The claim is existential and measured, not universal: these are
the loops we built, priced.

## Experiments

- **E-A — Rollout-label fidelity map (Connect-4, solved).** Measure |V^π − V*| per position as a
  function of policy strength, sample count, and position type. Registered: (a) error is
  irreducible in sample count beyond variance; (b) error concentrates on narrow-winning-path
  (skill-demanding) positions; (c) the plateau level is where remaining improvement mass lives in
  positions whose labels are wrong. This converts Paper I's plateau from observation to mechanism
  and answers "what is special about ~2000."
- **E-B — Search-over-policy margin curve (chess + Connect-4).** Gain-per-iteration of
  distill(search(π)) as a function of level and search depth; registered: margin shrinks toward
  the noise floor at the observed plateau level.

Full protocol and registered predictions F1–F5 are in et6-label-fidelity-spec.md (committed
separately; that spec now belongs to this paper).

## Relation to the series

Explains ET-I §3's anchor phenomenon; supplies ET-VII with the games-domain calibration of what
"missing information" looks like when an exact oracle exists; the broader self-improvement
question — whether a judge can be improved from inside — is ET-VII's, not this paper's.
