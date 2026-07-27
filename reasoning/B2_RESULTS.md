# B2 — MATH n=1000 rerun of the §4 crossover flip (Paper II tail)

**Registered:** §10 names "a true n ≈ 1,000 MATH rerun for a tighter interval" as remaining queue.
Fresh **seed-0 draw of 1000 problems from the full Hendrycks MATH test** (`math1000.jsonl`; gold-extractor
gated 100% on MATH-500 before any generation). 7B/14B Qwen2.5-Instruct-4bit, 16-sample caches at temp 0.8,
**true greedy decoded separately** (flip-decider rule), canonical `reason_cache` extractors. Run on llm1,
finished 2026-07-25.

## Result

| model | true-greedy | pass@1 | sc@4 | sc@16 | oracle@16 |
|------:|:-----------:|:------:|:----:|:-----:|:---------:|
| 7B  | 64.1 | 61.9 | 65.2 | 69.6 | 82.6 |
| 14B | 67.1 | 66.1 | 69.6 | 72.9 | 83.6 |

**Registered flip — 7B+sc@16 vs 14B true-greedy (paired exact McNemar):**

| set | 7B+sc16 | 14B-greedy | Δ | b / c | exact p |
|:----|:-------:|:----------:|:--:|:-----:|:-------:|
| shipped §4, n=500 (curated MATH-500) | 72.8 | 68.2 | +4.6 | 39 / 16 | 0.0027 |
| **B2, n=1000 (fresh full-test draw)** | **69.6** | **67.1** | **+2.5** | **71 / 46** | **0.0261** |

## Honest reading

The flip **replicates and stays significant at n=1000 (p=0.026)** — so the smaller-model-plus-search win
over the next size decoding greedily is **not an artifact of the specific 500-problem MATH-500 subset**.

But it does **not tighten** the shipped p-value; it does the opposite. On a representative random draw from
the full test the **margin shrinks** (Δ+2.5 vs +4.6) and the discordant ratio falls (71/46 = 1.54 vs
39/16 = 2.44), so p rises from 0.0027 to 0.026 — still < 0.05, but the effect is more modest than the
curated subset suggested. Absolute accuracies are also a touch lower on the fresh draw (7B sc16 69.6 vs
72.8), consistent with the full-test draw being slightly harder than MATH-500. The correct claim is
"significant on an independent, higher-n, representative draw, with a smaller effect size," not "tighter."

## Paper integration (v2)

- **§10 ledger row (v2):** *"a true n≈1000 MATH rerun for a tighter interval"* → **Confirmed, with a smaller
  margin.** 7B+sc@16 (69.6) beats 14B true-greedy (67.1) at p=0.026 on a fresh seed-0 draw of 1000 full-MATH
  problems (b/c 71/46). The crossover flip survives the move off the curated MATH-500 subset; the effect is
  real and modest (Δ+2.5), smaller than the n=500 subset's Δ+4.6 — reported as a replication of significance,
  not a tightening of the interval.
