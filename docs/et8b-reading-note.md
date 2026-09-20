# ET-8b — how the cross-fitted gen1 result will be read, written before it exists

*Sautée, 2026-09-20, while the cross-fitted chain is still running. The reading table in
`et8b-loop-gates.md` §4 is unchanged and is not being moved. This note records three things that
have to be fixed before the number lands, not after.*

## 1. The contaminated gen1 lands exactly on the "dogma" cell, and must not be read there

§4 row 3: *"gen1 < gen0, interval excludes zero → **dogma** — the prior's own choices narrowed what
it later learned from; reported at full prominence."*

The gen1 arm fitted on gen0's memorised episodes produced:

    gen1 vs gen0, held out (75):  41.3% -> 29.3%   -12.0 [-20.0, -4.0]   McNemar p = 0.0117

That is a hit on the dogma cell — interval excluding zero, in the predicted direction, with a
mechanism story available on request. **It is an artifact of my run design.** gen0's loop ran all
300 tasks with a head fitted on 1–225; on those 225 it finds the true region at decision 1 in 128 of
128 episodes against 22.4% held out; gen1's decision-2/3 heads were fitted on that distribution and
came back at chance.

The general hazard, stated plainly because it nearly caught me: **a contaminated corpus produces a
NEGATIVE that looks like an honest null, and a pre-registered table makes it easy to file.** A
pre-registration protects against choosing a reading after the fact. It does not protect against a
defective input, and it can make a defective input *more* publishable by giving it a waiting cell.
The check that caught it was not in the table: compare the rate of the key event in the training
split against the test split.

## 2. The power bound, stated before the number

§4 says *"All on an independent 300"*. The cross-fitted chain evaluates on **75** tasks (226–300),
because that is what is held out of the 300-task set the loop chain has used throughout.

For scale, gen0 vs base on this same n=75 gave +17.3 with a 95% interval of **[5.3, 29.3]** — 24
points wide. An accumulation increment of the size a second generation might plausibly add is
smaller than that interval. Therefore, fixed now:

- **`gen1 ≈ gen0` on n=75 does NOT establish §4's "one-shot" reading.** On this n it is consistent
  with one-shot *and* with a real increment the run cannot see. It will be reported as
  *underpowered for the one-shot reading*, with the interval printed.
- **`gen1 > gen0` with the interval excluding zero on n=75 is a screen, not the claim.** The claim
  §4 describes needs the independent 300 the spec asks for.
- **`gen1 < gen0` with the interval excluding zero** is the one reading n=75 can support on its own,
  and only after the §1 check above has been run on the cross-fitted trajectories.

## 3. What I am asking 理 to rule on

The v3 generator can produce an independent 300 under the same gates (P0, P1′ at generation, P2′,
P3 on the excluded set). Running it as the confirmatory set after seeing the n=75 screen is a
sequence the selection rules do not explicitly cover: §4 fixes *"no re-fit after seeing the
independent set"*, and nothing here would re-fit — the heads are frozen before the screen and the
same frozen heads would run the 300. **But the decision to spend the 300 would itself be taken after
seeing a result, and that is 理's call to make, not mine to assume.**

My recommendation: run the independent 300 regardless of which way the screen falls, so the decision
to run it is not conditioned on the screen's direction at all. That costs one more arm and removes
the question entirely.

— Sautée (沙汰)
