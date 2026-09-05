# experience/ — Efficient Thinking VIII: Experience Priors (working directory)

Design: `../docs/efficient-thinking-8-proposal.md` (registered) · concept: `../docs/efficient-thinking-8-experience-priors.md`
· review: `../docs/et8-review-ri.md`.

## What is here (v0, 2026-09-05)

| file | what | status |
|---|---|---|
| `et8_env.py` | task generator (12 bug classes × 4 families), external verifier (unittest), symptom attribution with red herrings, dead paths per family, trajectory schema | v0, runs with no LLM |
| `tasks/` | generated task sets (`gen --n N --seed S --out tasks/<name>`), gitignored except `_stats.json` | — |
| `traj/` | trajectories from the agent harness, one JSONL per run | not yet |

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

## Who runs what

- **Ri (理):** design, predictions, environment, injection code, write-up.
- **Sautee (沙汰):** all long runs on llm1/llm2 once a step is green on a Studio — baseline pass, sweeps, rounds.
  Hand-off unit: a command line, its expected output, and the JSON it must produce.
- **Louay:** voice pass, publication, the calls the proposal marks as his.

## Rules (series standing rules, unchanged)

No text before numbers. Proposals commit before experiments run; the commit is the registration. Every registered
prediction (proposal §7) is scored hit or miss. "Done" comes with its verification command and expected output. No
delta is believed until it survives a low-variance re-measurement (two seeds minimum).
