# Area-classifier measurement — pre-registration

*Sautée (沙汰), 2026-09-23. Registered by 理  on my objection to the filter design
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

## 0. Amended 2026-09-23 on 理

- **Two classifiers are measured, not one.** A local 7B (MLX, on llm2, scheduled around VIII-b) and
  Bedrock. **The choice between them is made on disagreement rate and phone latency, not on cost**
  (理 §2) — so latency is a measured quantity here, below, and not a footnote.
- **Cap: $10 total Bedrock spend for WO-318**, enforced in the same meter as IV's $40, which is
  enforced in code with a running meter. Local compute is uncapped and scheduled around VIII-b.
- **Tagging is NOT in this document.** 庭 hand-tags the ten venues from their compiled offer sheets
  (理, on Louay's word); the hand-tagged ten are the gold set, and the first measurement of
  any future automatic tagger is agreement with them. Nothing here measures tagging.
- **The gold sets (理 §8) close the gap §4 of this document opens.** Gold utterances carry
  `intent, tags, commit, expected visible_ids` per sentence, human-authored and second-person
  reviewed. **That is the labelled set §4 says correctness needs**, so correctness moves from "a
  separate measurement" to a measurement this document can name. I review 案内's utterances; a
  reviewer who also owns the measurement is a conflict, so my review is recorded as a review and
  the authorship stays 案内's.
- **Dependency, raised to 案内:** if `area` is a DSL statement rather than a typed object,
  **the disagreement rate needs a canonical form or it measures formatting instead of meaning.**
  If the grammar admits one and the validator emits it, disagreement is exact equality on the
  canonical statement. If it does not, a normalisation step is added here and named.

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
   ⚠️ **This is a measurement of the TAXONOMY's coverage, not an error path** (理 §1a, on this
   lane's own rule at `wo312-measurement-prereg.md` §6). A venue or utterance set where dropping is
   common is a finding about the registry. If it is only ever swallowed as an error, the registry's
   gaps are invisible: they look like users asking for things that do not exist.

## 2a. Latency, because the choice depends on it

Per utterance, wall-clock from call to parsed `area`, reported as the **median and the 90th
percentile** for each classifier. Not a mean: a head that is fast four times in five and slow on the
fifth is a head the user experiences as slow. §2 measurement 2 of the bench prereg already says a
head that is free because it is slow is not free; this is the same rule at the classifier.

## 2c. AMENDED 2026-09-23 after eight live calls — the unresolved rate is NOT the coverage number

I registered the unresolved-tag rate as the measurement of the taxonomy's coverage. **Eight live
calls returned an unresolved rate of zero every time**, and the reason is structural rather than
lucky: the prompt shows the model ONLY that venue's tags, so it rarely names one the venue lacks.
A constrained prompt makes the counter nearly vacuous.

**The coverage gap surfaces as `tags: []`, not as an unresolved tag.** Both off-menu sentences in
the live run came back empty — *"can you resole my shoes"* (genuinely off-menu) and *"Do you do
beard trims?"* (the venue sells only a haircut). So empty-tags conflates **correctly off-menu** with
**the taxonomy is too coarse to say what they meant**, and nothing separates them without a label.

**The coverage measurement is therefore the EMPTY-TAGS RATE on utterances the gold set does NOT
mark off-menu**, scored against `gold/utterances-v1` by version. §8's five kinds already carry the
off-menu label, so this needs no new instrumentation.

**The unresolved counter stays and is still reported** — it catches a model naming a tag from
another venue's set, which is a different failure and worth seeing — but it is no longer the
coverage number, and registering it as one without first checking what a constrained prompt does to
it was my error.

## 2d. REGISTERED 2026-09-23, after a rehearsal and BEFORE the gold run — the comparator, and per-field reporting

**The comparator is 案内's canonical form, not mine** (doc 239, 12438; asked for in my 12378):
sort, and a tag collapses into an **already-present ancestor** while siblings never reduce;
single-value `commit`; closed-enum `intent`. Two areas are equal iff their canonical forms are
structurally equal. Implemented once, in `bench/area_equal.py`, and used by the replicate
measurement and by 鉋's replay tool — one property, reused, not reinvented per consumer.

⚠️ **The comparator REFUSES across shapes rather than coercing.** 形 showed doc 239 and the
design of record disagree on what `intent` and `commit` ARE — 239 has `intent ∈ {ask, commit}` with
`commit ::= tag_id | null`; the walkthrough, `schema.py`'s `_INTENTS`, 理 and the live `/area`
all have eight intents and a boolean commit. A comparator that silently accepted both would turn an
open design question into a number nobody could interpret. The shape is sniffed on `commit`, never
on `intent`, because `"ask"` is legal in both vocabularies and is the most common value.

**Per-field disagreement is reported beside the combined rate**, because §2 registered intent and
tags as separate measurements and because the combined number cannot say which field moved:
`intent`, `tags` (raw set), `tags_canonical`, `commit`.

📌 **A rehearsal ran before this section was written, and it is the reason the section exists.** Ten
of our own sentences (not gold), three replicates, live Bedrock, `$0.0743`: combined disagreement
**0.20**, and per field **intent 0.20, tags 0.00, tags_canonical 0.00, commit 0.00**. Both
disagreements were on `intent` alone and on exactly the two kinds gold calls hardest — off-menu
(*"can you resole my shoes"*: `ask` twice, `order` once) and ambiguous (*"i need something cleaned"*:
`order` twice, `ask` once). ⚠️ **That rehearsal is not a result and is not quotable as one: I wrote
both the sentences and their off-menu labels, so the coverage number on it is a measurement of my
own answer key.** It is registered here as what was seen before the readings were fixed, so nobody
later has to take on trust that the per-field split was not chosen after seeing gold.

## 2b. `walked_up`, registered on my own default (12354, no objection by the stated time)

The rate at which `never_empty` walks UP the taxonomy, per venue. **Above 5% on a conformant venue
it is a finding about the REGISTRY** — leaves too fine for the venues that carry them — not about
the classifier and not about the user. Same reading as the unresolved-tag rate, and for the same
reason: a filter that silently broadens what the user asked for is a glitch the three classifier
numbers score as perfect.

⏳ **Offered, not registered:** a *filter surprise rate* — nodes that leave the screen without the
utterance naming them (12373, supporting 女将). It needs no new instrumentation, since the log
row already carries `visible_ids`, `folded_ids` and `walked_up`. It enters this document only if 理
or 匠 asks for it.

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
same wrong area scores perfectly here. **The gold utterances of 理 §8 are the labelled set
that closes this**, and correctness is scored against them by version — but the two numbers are
reported side by side and neither substitutes for the other: **this one is a floor, and a system
that fails it cannot be fixed by a better label set.** Registered so that a good number here is
never read as "the classifier is right".

It also measures one model at one temperature on one day. The loop's null moved between dates and
that is exactly the regime this cannot see; if the classifier ships, the replicate measurement is
re-run per model change, not once.

— Sautée (沙汰)
