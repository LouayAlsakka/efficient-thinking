# experience/ — Efficient Thinking VIII: Experience Priors (working directory)

Design: `../docs/efficient-thinking-8-proposal.md` (registered) · concept: `../docs/efficient-thinking-8-experience-priors.md`
· review: `../docs/et8-review-ri.md`.

## What is here (v0, 2026-09-05)

| file | what | status |
|---|---|---|
| `et8_env.py` | task generator (12 bug classes × 4 families), external verifier (unittest), symptom attribution with red herrings, dead paths per family, trajectory schema | v0, runs with no LLM |
| `et8_agent.py` | frozen mlx-lm model drives hypothesize → inspect → patch(auto-runs tests) → run; step + episode JSONL with hindsight `region_hit` / `class_hit`; repeat inspect / hypothesis / identical patch counted as waste; `--memory` = mechanism A | v0.1; smoke 7B-4bit 2/3 green |
| `et8_distill.py` | trajectories → candidate lessons in the record schema (two lists + evidence + shrunken confidence + reopen condition); renders `lessons.txt` for mechanism A | v0; 40 episodes → 10 clusters, 1.7k chars |
| `lessons/` | distilled candidate lessons per round (`lessons.json`, `lessons.txt`, `report.json`); status stays `candidate` until `et8_verify.py` admits | v0 from the v0.3 traces |
| `results/` | one JSON per measured run: numbers + the finding, recorded before any text is written | v0.1–v0.3 baselines |
| `tasks/` | generated task sets (`gen --n N --seed S --out tasks/<name>`), gitignored except `_stats.json` | v0: 100 tasks, 40% red herring |
| `traj/` | trajectories from the agent harness, one `<run>.steps.jsonl` + `<run>.episodes.jsonl` per run (gitignored; summaries go in results JSON) | smoke runs only |

## Build order (proposal §8)

1. **G1 — environment green.** `python experience/et8_env.py selftest` must print `clean=PASS caught=12/12`.
   Then `gen --n 100` and eyeball `_stats.json`: red-herring rate should be ~30–50% (producer/transform bugs surface in
   consumer tests first).
2. **A0 — chess anchor** (`chessnet/`): learned move prior in PUCT, sims-to-Elo. Ri designs, Sautee runs.
3. **Agent harness** (`et8_agent.py`, next): frozen Qwen2.5-3B-Instruct via mlx-lm; explicit action loop
   hypothesize → inspect → patch → run; 12-action budget; full trajectory logging; hindsight productivity labels.
4. **Baseline pass** over 2,000 tasks → τ± labels → distill → verify (rejection rate = P0) → admitted lessons v1.
5. **Injection points** A (text) · F (logit bias) · B (prefix) · C (steering vectors) · D (adapter) · E (LoRA) — one
   training set, one loss, layer and token sweeps.
6. **Rounds, dogma arm, transfer, write-up.**

## Hand-off #1 to Sautee — the frozen baseline pass (proposal §3.1 / §8 week 2)

Target measured 2026-09-05 19:27Z (read-only): **box A = Apple M3 Ultra, 256 GB, Darwin arm64, load ~1.6, user `lab`; python3
3.9.6, no mlx-lm, no brew/uv/conda, no clone.** So mlx-lm runs as-is once installed; no HF backend swap needed.

Step 0 (environment, Sautee): `git clone https://github.com/LouayAlsakka/efficient-thinking.git && cd efficient-thinking &&
python3 -m pip install --user mlx-lm huggingface_hub`. If pip refuses on Python 3.9, report "blocked: python" — the interpreter
install is tetsu's (IT), not a lane's; do not install system-wide. Then, from the repo root at the commit named in the request:

```
python experience/et8_env.py selftest                       # expect: clean=PASS  caught=12/12
python experience/et8_env.py gen --n 2000 --seed 1 --out experience/tasks/v1
                                                            # expect: _stats.json with n=2000, red_herring ≈ 35–45%
python experience/et8_agent.py --tasks experience/tasks/v1 --out experience/traj/base_v1_3b \
    --model Qwen/Qwen2.5-3B-Instruct --budget 12           # 9.8 s/task measured on box A (M3 Ultra, 32-episode steady state) → 5.4 h per 2,000-task pass; prints a summary JSON at the end
python experience/et8_agent.py --tasks experience/tasks/v1 --out experience/traj/base_v1_3b_s2 \
    --model Qwen/Qwen2.5-3B-Instruct --budget 12           # second seed of the SAME tasks (sampling differs after waste)
```

Deliverables back: the two summary JSONs (verbatim), `wc -l` of the four `.jsonl` files, and this one-liner per run:
`python -c "import json,collections;s=[json.loads(l) for l in open('experience/traj/base_v1_3b.steps.jsonl')];print(dict(collections.Counter(x['action'] for x in s)))"`.
Do not interpret; do not re-run on a different model; report "done" with the commands and their outputs, or "blocked" with the
first error. Design questions to Ri in channels/direct/ri+sautee.

## Who runs what

- **Ri (理):** design, predictions, environment, injection code, write-up.
- **Sautee (沙汰):** all long runs on box A/box B once a step is green on a Studio — baseline pass, sweeps, rounds.
  Hand-off unit: a command line, its expected output, and the JSON it must produce.
- **Louay:** voice pass, publication, the calls the proposal marks as his.

## Rules (series standing rules, unchanged)

No text before numbers. Proposals commit before experiments run; the commit is the registration. Every registered
prediction (proposal §7) is scored hit or miss. "Done" comes with its verification command and expected output. No
delta is believed until it survives a low-variance re-measurement (two seeds minimum).

## Rulings 2026-09-16 (理) — mechanism A at 7B, C complete, F's definition
- A (v0 lessons) destructive on 7B on every slice and axis; next A arm = `lessons/v1_7b` from the 7B's own 1,197 green episodes through the same distill + verify gate.
- C at α=4.0: not an improvement on either axis, close to baseline, cheap; probe d′ 3.1–4.9 good, intervention untuned (strength/layer/position). Slice-1 "clean separation" withdrawn by its author.
- F = additive logit bias on the EMITTABLE action tokens only (hypothesize/inspect/patch/run), per region class, log-ratio τ+/τ−, clipped ±2. Verdict labels (noop_patch, repeat_*, invalid) are not tokens; F cannot reach repair by construction and a null is a result. Outcome-folded biasing is a different experiment (F′), not run under this name.
- Slices, not seeds, under greedy decoding.

## Ruling, 2026-09-16 17:5xZ (理, on Sautee — the §4.3 gate's shape
- n=20 × 2 conditions × 2 arms × all 12 v1_7b lessons (~5 h box A GPU, overnight), after F(a)'s four slices land.
  No confidence-chosen subset: P0 is a rate over the candidates, and a subset's rate is not that rate.
- Report per lesson admit/reject with the paired delta and SE; headline = rejection rate over 12.
- Pre-registered before the first episode (both sentences in RESULTS.md): P0 holds = the gate rejects ≥ 20% of
  lessons that are TRUE (309/309 green) and still harmful, so §4.3 catches carrier harm; P0 refuted = the gate
  admits them, the failure is in the carrier and no lesson-level gate can see it — the paper says so either way.

## Ruling, 2026-09-16 20:0xZ (理; Louay: "G go") — Mechanism G, read-only controller head, QUEUED after the gate
Spec (paper §20b carries the pre-registration; this is the build order):
1. **Task-conditioned probe first** (no new episodes; the 2,000-run activations): fit the productive/wasted probe WITHIN
   task (task-demeaned states, or per-task fits pooled) at layers 18 and 27. Gate: d′ ≥ 2 within task, else STOP and
   report — the vector reads difficulty, G is not run.
2. **Candidate enumeration** at each decision point: emittable family × regions visible to the agent, capped at k ≤ 8 by
   the model's own top-k over the first action token; the greedy choice is always a candidate. Report mean k.
3. **State read**: teacher-force each candidate's action prefix through the frozen 7B (shared prefix cache), take the
   hidden state at the candidate's last token, layer 27 (18 as check). No write into the residual stream, ever.
4. **Head**: linear (logistic) on the state, trained on the 1,761 / 239 labelled decisions; held out by task slice.
   MLP only if linear fails the probe gate. Choice = argmax over candidates (deterministic; greedy decoding kept).
5. **Run**: the same four 80-task slices as A/C/F, paired vs baseline. Report success, localised, mean actions, input
   tokens/episode, wall, mean k — per slice and pooled.
6. **Pre-registered acceptance (§4.2a):** ε = 2.5 pts success on EVERY slice (≥ 60.0% vs 62.5%); localisation ≥ 86.3%;
   objective: mean actions ≤ 5.6 (−15%). Outcomes written now: "G passes: a read-only head derived from the model's own
   history cuts search cost without touching capability — the first prior that satisfies the constraint";
   "G holds success, actions flat: decodable but not actionable through choice"; "G loses success: the head chooses
   wrong regions — the signal is not a decision signal".
7. Order: gate run (tonight) → G steps 1–6 → then a C strength sweep only if G fails on choice rather than on signal.
Owner: Sautee (box A). 理 reads the task-conditioned probe result before step 2 starts.

## Rule, 2026-09-16 23:1xZ (理, after two losses in one day)
- NO working-tree moves under a running writer: not `stash -u` (理 lost 91 judge rows), not `pull --rebase` (沙汰's gtriv_s1
  came back 0 bytes), not `checkout`. Commit specific files with `git add <path>`; if the tree is behind, `merge --no-edit`
  (touches only files upstream changed) or wait for the writer to finish.
- G order (23:1xZ): steps 2–3 → G-TRIVIAL′ (candidate machinery, head replaced by the symptom rule) → step 4 head → step 5.

## Rules, 2026-09-17 00:4xZ (理, from 沙汰–10912)
- Every results table in a block or a commit message carries the command that produced it. A table typed from memory
  is not a measurement (沙汰's 55% → 40% correction, same minute).
- Never edit a script a running loop re-invokes; one orchestrator, one lifetime. Check file mtime against process start
  before trusting a slice.
- Every run persists the trajectories a later question could need — state AND action (FEN + UCI for chess; the task
  file for the env) — or its downstream claims are marked UNAUDITABLE. Third loss this week: tasks/v0, et6 games, a
  number that lived only in a paragraph.
- P8 is measured on `runs/conv_value_boxA` (supervised, 3.45M) by name; confirm `eval_search.py` loads the same run dir
  before any Elo rung is quoted. G-trivial / G-trivial′ re-run on v2 as v2's own controls; the v1 pair is v1-internal.

## P8 pre-registration, 2026-09-17 00:5xZ (理; 沙汰/10915)
- Evaluator: `runs/conv_value_boxA` (supervised 3.45M; the repo's own `sims_sweep.py` example names it). Ladder
  (`eval_search.py`) loads any run dir via `load_run`; rungs are Stockfish `UCI_Elo` levels, net-independent.
- Headline metric: head-to-head (`sims_sweep.py`) — "simulations to equal strength". The ladder runs once, no prior,
  at baseline sims, 60 games, only to name the rung. Not a pass criterion.
- Baseline: frozen net, no prior, 256 sims. Arms: WITH prior at {64, 128, 256} vs baseline-256; WITHOUT prior at
  {64, 128} vs baseline-256 (control: what sims alone buy). 80 games per pairing, colours balanced, lichess 2013-01
  openings (absolute path in the P8 runner; the shared script's relative default is left as is). SE ≈ 5.6 pts.
- Prior: `et8_chess_prior.py` trained on the 300 regenerated games of the frozen net, beta = 1.0; buckets/entries reported.
- §4.2a: constraint WITH-256 vs base-256 ≥ 45% win rate. Objective: WITH-128 vs base-256 ≥ 45% while WITHOUT-128 vs
  base-256 < 45% (a 2× saving sims alone do not buy); WITH-64 ≥ 45% = 4×.
- Readings, written before the run: PASS = same strength at half the search — the first prior this week to satisfy the
  constraint in any field. FAIL-cost = WITH-128 < 45%. FAIL-capability = WITH-256 < 45% (a constraint failure, as A/C/F).
- Instrument note for the paper: a 2-game smoke (±400 Elo) was deleted from the run dir so it cannot be read as a result;
  a fourth "artifact missing" was nearly reported because three real absences primed it — the pgn exists at an absolute
  path outside the repo. Negative results describe the seat as often as the world.

## Rule + ruling, 2026-09-17 03:2xZ (理, on 沙汰
- STANDING GATE, beside the permutation control: before any head is fit, count DISTINCT decision prompts at the decision
  point / episodes. Below 0.9 the state is a lookup table on the prompt and no head is fit. (v1: 10 / 2,000. v2 step 1: 6 / 80.)
- Ruling (a): v2 runs under an INSPECT-FIRST loop (one inspect required before the first hypothesis; the inspect target is
  the model's own greedy choice). G acts at the first post-inspect hypothesis. v2-i baseline → v2-i G-trivial + G-trivial′ →
  probe on post-inspect states → head → G. Comparability bent for v2, recorded; +1 action per episode printed, not barred.
  Bar re-based on v2-i; symptom-only table (51.2%) stays the FLOOR on localisation.

## Rules + rulings, 2026-09-17 05:2xZ (理, on 沙汰 )
- GATE ORDER: (0) task gate — distinct problems as the agent sees them, excluding id/seed, ≥ 0.9 × files; (1) prompt gate at
  the decision point ≥ 0.9 × episodes; (2) permutation control. Count problems before anything else.
- LOCALISATION IS NEVER A BAR. Bars are problems solved and cost. (An oracle on region+class that localised better solved
  half as many.)
- v3: ~300 signatures, file count = signature count, a program grammar, one working day (Sautee), PROVISIONAL until a base
  run makes P4 bite. INSPECT-FIRST is v3's default loop; a re-hypothesis rule is scoped with v3, not built before it.
- n = 11 is exact for greedy arms and a near-bound for exploring arms (EXPLORE_TEMP).
- 2026-09-17 05:3xZ: a counting function must handle the case its own caveat names — aggregate the outcome SET per
  signature or assert |set| == 1 and fail loudly. (沙汰: "costs two problems" withdrawn; same four, less reliably.)

## Rulings, 2026-09-17 15:5xZ (理, on 沙汰–11065) — v3 gates and order
- P2 RETIRED for v3, replaced (not lowered) by P2′: assertion detail on 100% of tasks AND symptom distinctness ≥ 0.5.
  Entropy printed beside every v3 table (0.861 at 300, n-dependent), never a bar. No coarsening is picked by anyone.
- P1′: per program, the failing-test set reachable from ≥ 2 of ITS regions; singletons named. (1 at 300 = pass.)
- n = 300 for the base and first pass (4 × 75); a SECOND 300 generated, gated, held unrun = the replication set. Any
  effect on the first 300 confirms on the second before it is a claim.
- Budget 12 stays. RULE: the budget is in the prompt → a budget change is a different agent; no budget curve is read as
  a truncation. "Easier programs" is the wrong fix, on record.
- RULE: a gate quoted from a probe states its value at the scoped n before any structural claim.
- Order on v3: emit 300 → inspect-first base → prompt gate (post-inspect) → probe + permutation → head → G (bar: problems
  solved + cost, re-based; four readings rewritten). G REOPENS on v3. Nothing else on v3 until G reads.
- P8 rung: MCTS-256 1983 ± 73, raw 1802; search +182 vs prior +20 ± 55. Fully closed.
- 2026-09-17 22:1xZ (理, on 沙汰): a probe accuracy that moves with the optimizer's step size is not a measurement of
  the state (56 → 59 → 24% across step sizes); heads are fit by L-BFGS, no free parameter. A subsample control is ≥ 5 draws,
  all printed, never one (two draws: 33.3 vs 46.7). G re-runs with the converged head before P9; both heads' wired numbers
  are printed side by side.

## Ruling, 2026-09-18 00:0xZ (理, on 沙汰 ) — P9 on this harness
- STRUCTURAL: at a decision point whose inputs are fixed before the prior acts, generations differ only in TARGET (G1's
  states == G0's, byte for byte). "Learns from a better agent's episodes" is vacuous there.
- (a)-inspect gated out (step-1 symptom distinctness 0.62–0.67 < 0.9). (a)-loop = the re-hypothesis loop = the accumulation
  harness, NEXT (own gates, own base). (b) RUNS as P9-on-this-harness: same states, labels = region correctness × G0's
  outcome (1.0 solved / 0.5 correct-but-failed / 0 wrong); eval on 47's 75 vs the seed-21 L-BFGS head (32.0%, 9.76) and
  base; PASS ≥ G0 at no higher cost / STRONG > 1 SE / FAIL < G0; expectation NULL; print fraction of argmax changes.
- Transfer: four cross-run +11.3…+14.7 carried; same-run +21.4 never averaged. Fit quality buys nothing, both directions.

## Rules + rulings, 2026-09-18 06:2xZ (理, on 沙汰 )
- A malformed action is an INVALID action, scored by the loop — never a crash of the run. Every chain refuses to continue on
  a short slice (episode-count guard). A second model surfacing a latent harness bug is a result and is reported.
- A caveat written in prose and not in code is decoration — third instance (r2b note said "different positions"; the low
  correlation was then hunted as a defect). Encode caveats as assertions.
- R2b: 528.4 vs 502 → FAILS; cost sentence stands. R2c (LAST item of 8a): head reads its state from the agent's own KV cache;
  count only tokens the agent would not otherwise pay; bar 502; expectation 36–180 tokens → pass.
- R4: floor 5% on 300 fired for Llama-3.2-3B-4bit (1.5%). ONE further model, different family, comparable size
  (Llama-3.1-8B-Instruct-4bit if it fits); if it floors, R4 = "out of range", generality handed to 8c/8d. No third model.
- 2026-09-18 08:5xZ (理): R2c VOID as written (raw-text identity 33/40) — a pre-registered gate is not relaxed after it fails.
  R2d pre-registered: (i) calibration — uncached vs uncached raw-text identity n=40 (does the strict gate measure the cache
  or bf16 decoding?); (ii) parsed-action identity n=300; (iii) paired outcomes cached vs uncached identical on the 75. All
  three → cost under R2d with R2c's failure + calibration printed. GPU order: WO-297 gate → R2d → 8a stops.
