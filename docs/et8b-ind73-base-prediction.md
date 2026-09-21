# ET-8b independent 300 (seed 73) — what I expect the BASE arm to land at, written before I look

*Sautée, 2026-09-21 05:05Z. The base arm is mid-run; I have not computed any rate from it.*

A base arm is the one number in this run with independent priors, so it is the one that can catch a
harness fault before it becomes a finding. Writing the expectation down first is the only way that
check means anything — afterwards, any number looks like what I expected.

## The priors

| source | set | agent | base rate |
|---|---|---|---|
| 8a R3 (`r3_complete.json`) | seed 73 | single-decision, budget 12 | **18.0%** |
| 8a R3 | seed 47 | single-decision | 16.7% |
| 8a | seed 21 | single-decision | 18.7% |
| 8b base loop | seed 21, tasks 1–225 | **loop**, up to 3 decisions, budget 12 | **18.2%** |
| 8b base loop | seed 21, held-out 75 | loop | 24.0% |

Three 300-problem draws from this generator put the single-decision base within 2.0 points of each
other, which R3 reported as evidence the base rate is a property of the generator. The 8b *loop*
base on seed 21's 225 is 18.2% — indistinguishable from those.

## The prediction

**8b's base loop on seed 73 lands between 15% and 24%.** Most likely near 18%, since that is both
R3's number on this exact set and the loop's number on seed 21's 225.

## What each outcome means

- **15–24%** — consistent with every prior; the harness is behaving and the arm is usable.
- **below 12% or above 30%** — 🔴 **investigate before reporting anything from this run.** The
  candidate faults, in the order I would check them: the task slices are not the set I think they
  are; the loop is not receiving the budget; `et8b_loop.py` is silently erroring per-episode and the
  episodes are being written anyway. A base that disagrees with four independent priors is a
  harness fault until proven otherwise, and the gen0/gen1 arms built on it would inherit it.
- The 24.0% figure from seed 21's held-out 75 is the *weakest* prior here — n=75, and it is the high
  end of the range. It is listed for completeness, not as the target.

— Sautée (沙汰)
