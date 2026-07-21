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

## First data (status update, post-registration)

E-A's Connect-4 arm has run, and F1's failure branch fired: at the plateau checkpoint, fit error
(0.399) dominates label bias (0.240) under MCTS-matched labels — the net stopped improving while
far from having learned its own labels, after an initial ambiguous result was flagged for a
measurement caveat and re-run to a registered decider. F2 held (bias flat in rollout count). The
paper's emerging story is therefore stage-dependent binding: fit/optimization binds first at this
scale, and the label ceiling (real: bias 0.24) binds only above it — the series'
measure-which-binds method applied to its own origin phenomenon. One consequence is registered as
a new open question: Paper II §8's oracle control may have worked partly through label
*learnability* (oracle targets are cleaner and easier to fit) and not only label *information*;
disentangling the two (oracle labels with matched injected noise) is an E-A extension. The
chess arm and F1's stage-wise form remain to be run; whether larger nets become label-bound is
now the live question.

## Relation to the series

Explains ET-I §3's anchor phenomenon; supplies ET-VII with the games-domain calibration of what
"missing information" looks like when an exact oracle exists; the broader self-improvement
question — whether a judge can be improved from inside — is ET-VII's, not this paper's.
