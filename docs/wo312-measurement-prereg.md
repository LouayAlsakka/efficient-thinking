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

## 2a. AMENDMENT, written when the task list was built and BEFORE any arm was run — taps-to-goal
has no variance on this venue

*Added 2026-09-22 by Sautée, on building `bench/tasks_v0.py` against 庭's compiled fixture.*

`quick-cuts.chelsea` sells **one** offer. The path to every reachable target is therefore the same
path — offer → staff → day → slot → identity → submit — and the task generator confirms it: across
100 tasks drawn from 378 reachable targets, `optimal_taps` takes exactly **one** value, 5.

**So measurement 1 cannot separate two arms that both navigate correctly here.** Taps-to-goal can
only move UP (a wrong guess costs a tap) or, for PREDICTED, down by collapsing a step. It is still
worth recording — an arm that takes *more* taps is a real finding, and §4's third reading depends on
it — but it is no longer the headline it is in WO-312's own list, and saying so after seeing a flat
column would be indefensible.

**What carries the signal on this venue instead:** hit@N (measurement 4) and seconds-to-goal
(measurement 2). The only thing that varies across tasks here is DEPTH — which of three staff, how
far down the open-day strip, how far into the day's eighteen slots — and depth costs *finding*, not
tapping. The task list records those three features per task for exactly this reason.

**And it names the fix, which is already the WO's next step.** Taps-to-goal becomes discriminating
on a venue with more than one offer, where the tail actions differ per task. That is the same second
compiled venue WO-312 already requires as its programmability test, so this is not extra work — it
is an argument for doing that step before reading too much into a flat taps column.

**One more thing the task list fixes rather than inherits.** The bench's day strip is built from the
run date and Mondays are closed, so a task naming a literal date is unreachable on a later run. Task
targets are expressed as (staff, Nth open day, Nth slot) and resolved to dates only for a stated
reference date, which is stamped into the artefact along with the fixture's md5 — verified identical
to 庭's compiled space, not assumed.

## 2b. AMENDMENT — what the SCRIPTED run can and cannot show, written after driving it and before
scoring 100 tasks

*Added 2026-09-22 by Sautée, after driving real tasks through the built STATIC and CHAT arms with
the logger running.*

§2a said taps-to-goal has no variance here. Driving it showed the same is true of three more of the
five, **for a scripted user on this venue**:

| measurement | on this venue, with a scripted user | why |
|---|---|---|
| 1 taps-to-goal | **constant 5** | one offer, one path (§2a) |
| 2 seconds-to-goal | **measures the driver** | the scripted user resolves a chip by selector; it never scans. A human's cost is *finding*, and a selector has no finding |
| 3 task completion | **unmeasurable** | the bench makes no engine call; `submit_fired` is recorded and completion stays null rather than being quietly redefined as "the screen said so" |
| 4 hit@N | **1.0 for STATIC by construction** | STATIC's head IS the whole menu — measured, not assumed: 18 of 18 slot chips fully inside a 480 px viewport, nothing off screen. A head that shows everything cannot miss. Undefined for CHAT |
| 5 cost per turn | **$0 for both built arms** | no predictor is wired yet |

**So the first measurement, run as WO-312 orders it, does not separate the arms.** It produces a
FLOOR and a sanity gate — §4's void condition is exactly that use: if the scripted user cannot
finish on a known-optimal path, the world is broken and nothing else is interpretable. That is worth
having, and it is not the comparison the WO's headline asks for. Saying so now, with the arms built
and driven and before 100 tasks are scored, is the point of writing readings down first.

**What each degeneracy costs, and what removes it** — all four are already steps in WO-312's own
order, so this is an argument about sequence, not scope:

- **the engine** removes 3. Blocked on 女将's answer to 形's 12055.
- **a second compiled venue with more than one offer** removes 1, and is the WO's own programmability test.
- **a user who must choose rather than be told** removes 2 and 4: a frontier-model user picks from
  what is shown and can miss; a human scans and the scan costs seconds.
- **PREDICTED with N = 5** removes 4 on its own side — 5 of 18 slots is a head that can miss, and
  the comparison then has something to compare. It waits on Louay reading the principle doc.

⚠️ **The reading this fixes in advance.** If the scripted run is reported as "no difference between
the arms", that is a statement about the scripted user and this venue, **not** about predicted heads.
Any such sentence must carry this table beside it.

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

**The same trap, one level down: the VENUE.** A kind can be legal and the action still unshowable,
because its args name something the compiled venue does not have — an offer that is not on the sheet,
a barber who is not on the roster, a day the shop is shut. Added 2026-09-22, before the first
measurement, after wiring the predictor against niwa's fixture: the first version of the prompt never
told the model what the venue *contained*, so it could only guess names, and a guessed name would
have been scored as a miss attributable to the model when it was attributable to me. Two changes,
both registered here:

1. The compiled venue — offer sheet, roster, open days, flow — is in the prompt every turn. A
   predictor asked for the next action without being told what exists is being measured on a
   handicap the STATIC arm does not carry, since STATIC is drawn from the same compiled definition.
2. An in-vocabulary action whose args name an unresolvable id is held back and **counted as
   `unresolved`**, exactly as an out-of-vocabulary kind is counted as `dropped`. It is never shown
   and never counted as a hit. Both counts ship beside hit@N in the results table.

A run in which `unresolved` is common is not a result about prediction; it is a result about the
prompt, and it is reported that way.

— Sautée (沙汰)
