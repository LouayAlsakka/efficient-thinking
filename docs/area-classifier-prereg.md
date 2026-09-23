# Area-classifier measurement — pre-registration

*Sautée (沙汰), 2026-09-23. Registered by 理 (nirai 12311 §1b) on my objection to the filter design
(12303), BEFORE the WO is filed and before any classifier exists. The readings below are fixed now
for the same reason every reading in this series is: a threshold chosen after seeing the number is
not a threshold.*

## Why this exists

The design says the venue content is filtered "predictably across 100Ks of venues without any
glitch". §5's three guarantees — never empty, never stranded, conformance at compile — are **static
checks on the FILTER**. The thing that turns an utterance into an `area` is a **runtime LLM**, and
nothing in the design says how anyone would know it is predictable, or what "without a glitch" means
as a number.

That gap is not theoretical. On 2026-09-22 this lane measured a system everyone believed was stable:

```
  four base arms, single-decision harness   11.7 11.7 12.3 11.7        sd 0.33
  five base arms, loop harness              20.7 22.0 25.7 27.0 27.7   sd ≈3.0, range 7.0
  four of six NULL draws reached p < 0.06 — the test called nothing-at-all significant
```

Same model, same week, same estate. **A consistency claim nobody had measured was wrong by five
points**, and 100K venues multiplies a classifier's spread by 100,000.

## 1. The unit, and what is replicated

A **fixed utterance set** — utterances paired with the venue whose tag set they are classified
against — written and frozen before any run. Each utterance is classified **N = 5 times**, in
independent calls. **The reported number is the disagreement across replicates, not one pass.**
That is the chess-match rule 理 registered for the loop (12243), applied one level up.

## 2. The three measurements

1. **Intent disagreement rate** — the fraction of utterances whose `intent` is not identical across
   all N replicates. `intent` is a closed enum, so this is exact, not a similarity score.
2. **Tag-set disagreement rate** — per utterance, the mean pairwise Jaccard distance between the N
   replicates' `tags`. Reported with the fraction of utterances that are unanimous.
3. **Unresolved-tag rate** — tags the classifier emitted that do not exist in that venue's tag set.
   ⚠️ **This is a measurement of the TAXONOMY's coverage, not an error path** (理 12311 §1a, on this
   lane's own rule at `wo312-measurement-prereg.md` §6). A venue or utterance set where dropping is
   common is a finding about the registry. If it is only ever swallowed as an error, the registry's
   gaps are invisible: they look like users asking for things that do not exist.

## 3. Readings, fixed now

| result | reading |
|---|---|
| intent disagreement ≤ 2% AND tag Jaccard distance ≤ 0.05 | **"predictable" earns the word.** The head is stable enough that a venue's behaviour is a property of its tags, not of the draw |
| intent disagreement 2–10% | the head is usable but the classifier is a reported source of variance; every downstream number carries the replicate interval, never a single pass |
| intent disagreement > 10% | **the head is not predictable and the design's central claim fails as written.** Reported at full prominence; a filter that is correct by construction on top of a classifier that disagrees with itself one time in ten is not a predictable interface |
| unresolved-tag rate > 5% on a venue whose tags are conformant | a finding about the TAXONOMY, reported as such, and not counted against the classifier |
| any measurement taken on a single pass | **void.** The whole point of this document is that one pass cannot see this |

## 4. What this cannot show

It measures **self-consistency, not correctness.** A classifier that maps every utterance to the
same wrong area scores perfectly here. Correctness needs a labelled set and is a separate
measurement; **this one is a floor, and a system that fails it cannot be fixed by a better label
set.** Registered so that a good number here is never read as "the classifier is right".

It also measures one model at one temperature on one day. The loop's null moved between dates and
that is exactly the regime this cannot see; if the classifier ships, the replicate measurement is
re-run per model change, not once.

— Sautée (沙汰)
