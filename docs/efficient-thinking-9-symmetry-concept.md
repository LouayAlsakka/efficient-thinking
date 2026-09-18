# Efficient Thinking IX: Symmetry — Invariance as Thought Compression (concept)

*Status: concept, 2026-09-18. One sentence: a symmetry is a transformation of a problem that does not change its
answer; identify the transformations that do not matter, solve one representative, and the solution covers the
whole orbit — so a search that knows its symmetries is smaller by the size of the orbit, and a prior that respects
them generalises from fewer episodes. This paper defines the object, states what it buys and what it costs, gives the
falsification test that must precede any use of a claimed symmetry, and names two experiments that run on harnesses
the series already has.*

## 1. The object

Let a task family carry a set of transformations \(G\) — rotations and reflections of a board, renaming of
variables, reordering of independent statements, a change of units, a change of frame. A transformation \(g\) is a
**symmetry of the answer** if the verifier's verdict is unchanged under it: solving \(g \cdot x\) is the same as
solving \(x\) and transporting the solution. The set of problems reachable from \(x\) by symmetries is its
**orbit**; the search space, quotiented by \(G\), shrinks by the average orbit size.

Two directions follow, and they are the two things any learning system has to do:

- **Compression.** Solve one representative per orbit. An endgame table does not store a king-and-two-pawns
  position eight times; a physicist does not solve the field equations once per orientation of the apparatus.
- **Expansion.** A result found on one instance applies to every instance in its orbit, and an experience prior
  trained on one instance should act the same on the others. A prior that fails to is reading something the
  answer does not depend on — surface, not structure.

Cosmology is the clearest large example. The cosmological principle — homogeneity and isotropy — is a symmetry
assumption; it collapses the possible metrics to one family and makes the theory computable at all. Noether's
theorem is the same idea with a proof: every continuous symmetry of the action is a conserved quantity, so a
symmetry is not only a saving in search but a statement about what cannot change. The method is the one this
series has been using under another name: look from several angles, find the ones that make no difference, and
what is left is the thing that does.

## 2. What it buys

- **Search.** A search that identifies orbits visits one member each. On a board with the eight dihedral symmetries
  and colour reversal, that is up to sixteenfold; in an endgame where the answer depends only on a local pattern,
  it is the ratio of the board to the pattern's window.
- **Data.** A prior trained under a symmetry group sees every episode as \(|G|\) episodes. Where episodes are the
  cost — 8a's task sets are hundreds, not millions — that is the difference between a head that generalises and
  one that memorises.
- **Locality.** Many symmetries are local: a \(3 \times 3\) or \(5 \times 5\) window whose value does not depend on the
  rest of the board. A local invariant is a feature the prior can read directly, and it transfers across boards,
  programs and positions that share the window and nothing else.
- **A test of the prior itself.** If the answer is invariant under \(g\) and the prior's choice is not, the prior is
  reading surface. That test needs no labels: it is a permutation control in the input rather than the output.

## 3. What it costs, and the failure that matters

A symmetry that is assumed and false is worse than none. The cosmological principle fails below the scale of
galaxies; a chess board's left–right mirror is broken by castling rights and by which side the pawns advance; a
program's variable renaming is a symmetry only if no string of the program reads the name. The whole method rests
on **verifying the symmetry before using it**, and the series' instruments already say how:

- **The invariance test.** For a claimed symmetry \(g\), take held-out instances, apply \(g\), and measure whether the
  verifier's verdict is preserved. A symmetry is admitted only at a measured invariance rate, printed, with the
  cases that broke it named. This is "verify the verifier" applied to the assumption.
- **The orbit count.** Before claiming a saving, count orbits under the admitted group on the actual task set. A
  task set can carry a large nominal group and a small realised one.
- **Approximate symmetry.** Most useful symmetries are approximate — the answer changes a little under \(g\). The
  prior may use them as a prior, never as a constraint, and the approximation error is measured, not assumed.

## 4. Relation to Efficient Thinking 8

8a's positive result is a read-only head over a frozen model's hidden state. Two of its findings are symmetry
statements in disguise. The head that scored 100% on the first task set was a lookup table on the symptom
string — a prior reading a feature the answer happened to be a function of, on that set only; the third task set
broke that false symmetry (symptom → region) by construction, and the head that survived reads something the
answer depends on. And the chess prior's position classes are a quotient of the board by a hand-chosen
equivalence; its null result (+20 ± 55 Elo) says that quotient carried little of what the search needed. The
question ET-IX asks of both is the same: which transformations of the state leave the verifier's verdict fixed,
and does the prior's choice respect them?

## 5. Two experiments on harnesses the series already has

**S1 — chess, symmetry-augmented prior.** The 8c prior is trained on 300 games of the frozen net's own play. Under
the dihedral group with colour reversal, the same games are up to sixteen times the data. Pre-registered: the
invariance test first (the evaluator's value under each transform, on 1,000 positions — castling and pawn
direction restrict the admitted group); then the same five-pairing sweep as 8c with the augmented prior at the
same rule-chosen strength. Readings: PASS if the augmented prior clears the objective 8c's did not (a twofold
search saving at held strength); NULL if the paired differences stay within 0.08; the number to print beside
either is the orbit count on the 300 games under the admitted group.

**S2 — debugging, the invariance test on the head.** The 8a head chooses a region from the post-inspect state. The
program grammar admits semantics-preserving transformations — identifier renaming, reordering of independent
statements — under which the bug region is unchanged. Pre-registered: (i) the invariance rate of the verifier under
each transform on the 300 problems; (ii) the head's pick-agreement with itself across the orbit, on held-out
problems, printed beside the agent's own pick-agreement; (iii) the head refitted with orbit-augmented training
episodes, run on the independent set against the un-augmented head. Readings: a head whose pick changes under a
transform the answer ignores is reading surface, and the fraction says how much; augmentation that raises
problems solved on the independent set at held cost is data bought from symmetry; augmentation that changes
nothing says the head already reads the invariant.

Both run on existing task sets and existing heads; S1 is GPU-cheap and S2 is a join plus one refit. Neither needs
a new field.

## 6. What this paper would claim, and what it would not

It would claim that a verified symmetry is a measured saving in search or in data, with the orbit count and the
invariance rate printed, and that the invariance test is a control every learned prior should pass. It would not
claim that symmetries exist in a domain until the test has admitted them there, and it would carry the false
symmetries it found — the symptom that pinned a region, the mirror that castling breaks — as the method's own
evidence.
