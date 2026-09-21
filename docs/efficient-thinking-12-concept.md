# Efficient Thinking XII: Skills — additive skills and the head that chooses them (idea)

> **STATE 2026-09-21: IDEA.** Recorded in the pipeline at the author's ask; not a registered proposal, nothing measured, no
> predictions yet. Its first experiment needs no new field and can run beside X-a or XI-a. Titles are the author's.

## One sentence

Can a skill be carried as an additive parameter set that leaves the base model untouched, and can a read-only head
fitted on verified history choose which skill to apply per task — an experience prior over skills, the VIII mechanism
at a different decision.

## The author's two questions

1. Two models with different skills and training: can a model with both skill sets be built from them?
2. A skill as an additive parameter set with no effect on the base parameters, and a task-conditioned selector that
   chooses the skill and tunes it.

## What is already done elsewhere, and must be cited

Question 2's first half is a LoRA adapter by definition (Hu et al. 2021); the selector is adapter routing and mixtures
of adapters (LoraHub and after). Question 1 is model merging — task arithmetic, TIES, DARE — well measured. The series
does not re-do these. What is not done: the efficiency framing (quality per unit compute with the selector's cost
charged) and the selector as an experience prior — a linear read of the frozen base's hidden state, trained from
verified history, choosing the adapter, with the three probe controls and the task and prompt gates of VIII.

## First experiment on what the estate already serves

Two served L2 adapters, held by hash in the adapter registry. A head fitted on the frozen base's state at the query
picks one per query; arms: always-A, always-B, both (merged), the model's own preference, the head. Verifier: the
served result-set checks. Readings written before the run, in VIII's form. Not a production change; offline, the
WO-304 rules apply.

## Relation to the series

VIII: experience chooses where to search. XII: experience chooses which skill to bring. Same instrument, one level up.
