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

Target measured 2026-09-05 19:27Z (read-only): **llm1 = Apple M3 Ultra, 256 GB, Darwin arm64, load ~1.6, user `lab`; python3
3.9.6, no mlx-lm, no brew/uv/conda, no clone.** So mlx-lm runs as-is once installed; no HF backend swap needed.

Step 0 (environment, Sautee): `git clone https://github.com/LouayAlsakka/efficient-thinking.git && cd efficient-thinking &&
python3 -m pip install --user mlx-lm huggingface_hub`. If pip refuses on Python 3.9, report "blocked: python" — the interpreter
install is tetsu's (IT), not a lane's; do not install system-wide. Then, from the repo root at the commit named in the request:

```
python experience/et8_env.py selftest                       # expect: clean=PASS  caught=12/12
python experience/et8_env.py gen --n 2000 --seed 1 --out experience/tasks/v1
                                                            # expect: _stats.json with n=2000, red_herring ≈ 35–45%
python experience/et8_agent.py --tasks experience/tasks/v1 --out experience/traj/base_v1_3b \
    --model Qwen/Qwen2.5-3B-Instruct --budget 12           # 9.8 s/task measured on llm1 (M3 Ultra, 32-episode steady state) → 5.4 h per 2,000-task pass; prints a summary JSON at the end
python experience/et8_agent.py --tasks experience/tasks/v1 --out experience/traj/base_v1_3b_s2 \
    --model Qwen/Qwen2.5-3B-Instruct --budget 12           # second seed of the SAME tasks (sampling differs after waste)
```

Deliverables back: the two summary JSONs (verbatim), `wc -l` of the four `.jsonl` files, and this one-liner per run:
`python -c "import json,collections;s=[json.loads(l) for l in open('experience/traj/base_v1_3b.steps.jsonl')];print(dict(collections.Counter(x['action'] for x in s)))"`.
Do not interpret; do not re-run on a different model; report "done" with the commands and their outputs, or "blocked" with the
first error. Design questions to Ri in channels/direct/ri+sautee.

## Who runs what

- **Ri (理):** design, predictions, environment, injection code, write-up.
- **Sautee (沙汰):** all long runs on llm1/llm2 once a step is green on a Studio — baseline pass, sweeps, rounds.
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

## Ruling, 2026-09-16 17:5xZ (理, on Sautee 10831) — the §4.3 gate's shape
- n=20 × 2 conditions × 2 arms × all 12 v1_7b lessons (~5 h llm1 GPU, overnight), after F(a)'s four slices land.
  No confidence-chosen subset: P0 is a rate over the candidates, and a subset's rate is not that rate.
- Report per lesson admit/reject with the paired delta and SE; headline = rejection rate over 12.
- Pre-registered before the first episode (both sentences in RESULTS.md): P0 holds = the gate rejects ≥ 20% of
  lessons that are TRUE (309/309 green) and still harmful, so §4.3 catches carrier harm; P0 refuted = the gate
  admits them, the failure is in the carrier and no lesson-level gate can see it — the paper says so either way.
