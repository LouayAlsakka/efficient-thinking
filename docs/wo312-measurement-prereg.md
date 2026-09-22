# WO-312 — the bench measurement. PRE-REGISTRATION.

> **Written before the bench exists.** WO-312: *"Measurement, pre-registered before the first user."*
> 形's renderer is one screen old and 庭's engine is not wired, so nothing here can have been chosen
> to fit a result. The commit is the timestamp.
>
> Owner: 沙汰 (the prediction head, the measurement harness). 理 sequences; Louay gates each step.

## 1. What is being compared

Three interfaces on **the same world, the same state, the same task list** — WO-312's own framing.
Only the head's source differs; 形's renderer draws all three.

| arm | the head |
|---|---|
| **STATIC** | a hand-designed head of five controls plus menus |
| **CHAT** | no head — the bar alone, the model replies in text |
| **PREDICTED** | the predictor's top-N next actions, recomputed per turn, bar beside it as fallback |

## 2. The five measurements

Named in WO-312; the definitions below are mine and are what makes them comparable across arms.

1. **taps-to-goal** — discrete user actions from task start to the booking existing. A tap is any
   action the user commits: a control press, a menu selection, a bar submission. Typing is not taps;
   **one bar submission is one tap regardless of how many words it carries.**
2. **seconds-to-goal** — wall clock, first render of the task to the booking existing. Excludes
   predictor latency *only if* the screen was usable while it resolved; if the user waited, the wait
   counts. A head that is free because it is slow is not free.
3. **task completion** — did the booking exist in the engine at the end. Binary, and it is the
   engine's answer, not the interface's: a screen that says "booked" without a booking is a failure,
   not a success with a caveat.
4. **hit@N** — per turn, was the action the user took next **on the head that was shown**. Computed
   per turn and averaged per task. ⚠️ **Only defined for STATIC and PREDICTED.** CHAT has no head;
   its hit@N is not 0 and not 1, it is undefined, and it will be printed as `—`.
5. **cost per turn** — the predictor's metered spend. **$0 for STATIC and CHAT-without-prediction by
   construction; $0.0052/turn measured for PREDICTED v0** (Opus 4.7 on Bedrock, nirai 12059).

## 3. Pairing and the test

Paired **per task across the three arms** — the same task list, so each task contributes one triple.
Paired bootstrap for intervals, exact McNemar on the discordant pairs for completion. The same
instruments as VIII; no new statistics are introduced for this bench.

## 4. Readings, fixed now

| result | reading |
|---|---|
| PREDICTED < STATIC in taps, interval excluding zero, at hit@N > STATIC's | the predicted head finds actions a designer's five controls miss — the bench's thesis |
| PREDICTED ≈ STATIC | a hand-designed head is as good as a frontier model's guess on this world; the bench has measured the ceiling and it is not above the floor |
| PREDICTED > STATIC in taps | the predicted head costs taps — reported at full prominence, with hit@N beside it, because a head that is *accurate* and still slower is the more interesting failure |
| CHAT best on taps but worst on completion | the bar is expressive and unreliable; that is a result about interfaces, not a defect of the bench |
| any arm's completion < 100% on the scripted user | ⚠️ **the world or the engine is broken, not the interface.** The scripted user follows a known-optimal path; if it cannot finish, nothing else on the page is interpretable and the measurement is void until it does |

## 5. What this bench cannot show, written before it is run

- **Nothing about real users.** A scripted user has a known-optimal path and infinite patience. The
  frontier-model user and then people are later arms; until they run, every number here is about a
  path someone already knew.
- **Nothing about v1.** v0 is the frontier model asked per turn — the *ceiling*, and an expensive
  one. The gap between v0 and a fitted head (v1) is the paper's frontier and is not measured here.
- **Nothing outside one world.** One compiled salon. A second world is what would make "programmable"
  a measured property rather than a demonstrated one.
- **hit@N is not utility.** A head can show the action the user takes and still be worse to use, if
  it shows it fourth, or beside four wrong ones. N is reported with it, always.

## 6. The trap this design is built to avoid

**The PREDICTED arm must not be allowed a vocabulary the STATIC arm lacks.** WO-312: *"nothing
outside it can be shown."* If the predictor may invent an action, a hit@N win measures who had the
bigger vocabulary and not whose head was better. `bench/predictor_v0.py` enforces the fixed kinds and
**counts every dropped proposal**; the drop count ships with every prediction and appears in the
results table beside hit@N. If drops are common, hit@N is reported over the *offered* head and the
drop rate is reported beside it — never silently folded in.

— Sautée (沙汰)
