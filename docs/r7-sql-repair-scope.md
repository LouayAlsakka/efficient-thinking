# R7 — SQL query repair as a second search structure: SCOPE, before a line of generator

**理:** *"Scope it in writing before a line of generator, cost estimate first, one day to
build and gate."* This is that document. **No generator code exists yet and none should until this
is ruled.**

## 1. What R7 is for

Everything in 8a is measured on ONE environment: a Python program-repair grammar where the regions
are pipeline stages (`parse / map / agg / format / emit`). If G's benefit is a fact about
**experience-directed search**, a different search STRUCTURE with the same harness should support it.
If it is a fact about that grammar, it will not.

**The claim under test is the mechanism, not the weights.** The head is RE-FITTED from the SQL
environment's own base episodes. Nothing transfers but the method.

## 2. The environment, concretely

| | |
|---|---|
| task | a SQL query that returns the wrong result set on a fixed schema |
| regions | `select` · `where` · `join` · `group` · `order` — clauses, not stages |
| verifier | the EXPECTED RESULT SET on fixed fixture data. Not string equality on SQL. |
| a bug is admitted | only if the verifier DISTINGUISHES it: the mutated query must return a different result set than the clean one on the fixtures |
| actions | the same protocol as v3 — `inspect <region>` · `hypothesize <region> <class>` · `patch` · `run` |

**Why the result set and not the query text.** Two different queries can be equivalent; a repair
that is textually unlike the original but returns the right rows IS a repair. Scoring on text would
mark correct repairs wrong and make the success rate a fact about phrasing.

## 3. The gates — the same ones, and they are the reason v3 exists

v3 replaced v1/v2 because those had **11–12 distinct problems wearing 2,000 task ids**, which
invalidated every magnitude for a week. The same gates apply here and R7 does not proceed past any
one that fails:

- **P0** distinct task signatures ≥ 0.9 × tasks
- **P1′** a task is REJECTED at generation unless its failing test is reachable from ≥ 2 regions —
  a symptom that pins its own region makes localisation free
- **P2′** assertion detail 100%, symptom distinctness ≥ 0.5
- **P3** held-out lookup ≤ 60%
- **prompt gate** distinct decision prompts / episodes ≥ 0.9 at the post-inspect decision, else the
  state is a table
- **probe controls** permutation (must collapse to chance), PCA-8, five-draw n=100

## 4. What I expect to be hard, said before it bites

1. **The clause structure may pin the region for free.** "No rows returned" often implicates `where`
   or `join` with no inspection at all. If P1′ rejects most candidate bugs, the environment is
   telling us localisation is not the bottleneck here — **that is a RESULT, and 8a should print it
   rather than my weakening the gate to get a corpus.**
2. **Five regions, not five stages.** A query need not use every clause. A task whose query has no
   `group` has four regions, so the candidate count k varies and chance is not a constant 20.4%.
   **Chance must be computed per task, not assumed.**
3. **SQL is far more familiar to a 7B than my synthetic pipeline grammar.** The base rate could sit
   near the ceiling, at which point there is no headroom and R7 cannot answer the question — the
   mirror of R4's floor. **A ceiling check runs BEFORE any head is fit, with the same logic: base
   above ~80% means no room, reported as such.**
4. **The verifier is a database.** Fixture data must be deterministic and ordered, or `order` bugs
   are unfalsifiable and `select` bugs flap.

## 5. Cost estimate, from measured rates

| item | estimate | basis |
|---|---|---|
| generator + verifier + gates | ~1 day | v3's generator took that, and this reuses its harness, action protocol and gate code |
| base run, 300 | ~3.5 h | measured: v3's 300 at budget 12 on box A |
| states + probe fit | ~20 min | measured: R4's 300-decision fit |
| G run, 300 | ~3.5 h | measured |
| one matched-compute point | ~3.5 h | one extra base arm at the head's all-in budget |
| **total GPU** | **~11 h** | plus the build day |

## 6. What R7 can and cannot conclude

- **G clears the bar here** → the claim is **experience-directed search**, supported on two
  structures rather than one.
- **G does not clear it** → 8a says the result is **grammar-bound** and hands the question to 8c.
  ⛔ This is a real outcome and the cost estimate is the price of being able to say it.
- ⚠️ **Two environments are not "environments in general".** Both are synthetic, both are mine, both
  use the same harness and the same verifier-decides-the-bug discipline. A shared blind spot in that
  discipline would be invisible to both.

## 7. What is NOT in scope

No new model, no new seed, no fine-tuning, no change to the action protocol, no change to the bar
(problems solved + cost, one matched-compute point).

— Sautée (沙汰), studio lane
