# Efficient Thinking XII: Skills — the head that chooses which skill to bring (idea)

> **STATE 2026-09-21: IDEA.** Recorded in the pipeline at the author's ask; not a registered proposal, nothing measured, no
> predictions yet. Its first experiment needs no new field and can run beside X-a or XI-a. Titles are the author's.

## One sentence

Can a skill be carried as an additive parameter set that leaves the base model untouched, and can a read-only head
fitted on verified history choose which skill to apply per task — an experience prior over skills, the VIII mechanism
at a different decision.

## The question (narrowed 2026-09-21, the author's ruling)

A skill as an additive parameter set with no effect on the base parameters, and a task-conditioned selector — a
read-only head fitted on verified history — that chooses which skill to bring. **Merging two models' skills into one
model is out of scope:** merging changes the parameters and so the intelligence, the one thing this series holds
fixed; that question belongs to another programme and is cited, not pursued.

## What is already done elsewhere, and must be cited

The additive skill is a LoRA adapter by definition (Hu et al. 2021); the selector is adapter routing and mixtures of
adapters (LoraHub and after). Model merging — task arithmetic, TIES, DARE — is the dropped half and is cited only as
the road not taken. What is not done: the efficiency framing (quality per unit compute with the selector's cost
charged) and the selector as an experience prior — a linear read of the frozen base's hidden state, trained from
verified history, choosing the adapter, with the three probe controls and the task and prompt gates of VIII.

## First experiment on what the estate already serves

Two served L2 adapters, held by hash in the adapter registry. A head fitted on the frozen base's state at the query
picks one per query; arms: always-A, always-B, both (merged), the model's own preference, the head. Verifier: the
served result-set checks. Readings written before the run, in VIII's form. Not a production change; offline, the
WO-304 rules apply.

## Relation to the series

VIII: experience chooses where to search. XII: experience chooses which skill to bring. Same instrument, one level up.
