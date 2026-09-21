# Efficient Thinking XI: Communication — the channel between two reasoners (idea)

> **STATE 2026-09-21: IDEA.** Recorded in the pipeline at the author's ask; not a registered proposal, nothing measured, no
> predictions yet. Queued behind X-a, whose instruments it inherits. Titles are the author's.

## One sentence

Natural language is the human-readable channel between agents; this paper asks whether it is the efficient one — the
quality–compute frontier of two agents with different tasks and one verified outcome, with the channel as the
independent variable and its cost charged.

## The ladder (the author's three rungs, plus the two the series requires)

| rung | question | what is fixed, what is measured |
|---|---|---|
| XI-a fixed protocol | Two agents, different skills, one verifiable joint task, a given message format. Natural language vs a structured machine format vs a learned channel, at equal task information. | the frontier Q(C) with channel tokens charged; the random-reversible-encoding control from X |
| XI-b slight boundaries | The agents are told only the goal and that a channel exists; no format. | does a protocol emerge, at what cost, and does it beat the fixed one |
| XI-c existence only | Each is told the other exists, nothing else. | discovery rate against a control in which the other cannot be reached |
| XI-d accidental discovery | Neither is told; the environment permits contact. | the same control; every success is a story without it |
| XI-e translatability | A learned channel must be decodable to language with cost charged, or it is a private code. | Q_D as in X-c; a governance property, not only a scientific one |

## What is already done elsewhere, and must be cited

Emergent communication between learning agents (Foerster et al. 2016; Lazaridou et al. 2017 and after); this year's
work on language models exchanging hidden states or caches instead of text. The novelty, as in X, is the measurement:
the frontier with cost charged, the discovery rungs against a control, and the translatability requirement.

## First experiment on harnesses the series holds

The debugging environment split into two roles — a localiser that may only inspect and a repairer that may only patch —
with one verifier; the channel between them is the variable. Everything else is VIII's harness unchanged.
