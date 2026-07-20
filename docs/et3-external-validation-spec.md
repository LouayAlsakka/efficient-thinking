# Studio Run Spec — ET-III external validation (MATH subset + Llama check)

Registered before execution; this file's commit timestamp is the registration. Purpose: the two
upgrades the paper's own §8 and the venue review agree on. Both machines are idle; both runs fit
in a day or two.

## Run 1 — MATH judge-grid subset (the decisive one)

Judges {14B, 32B, 72B} × policies {3B, 7B} × N {4, 16}, pick-best mode with --perproblem, over
Paper II's frozen MATH caches (n = 300, 16-sample). Add the 7B judge on the 3B policy as the
marginal-judge cell. That is 14 cells total. Same harness, same McNemar analysis, same build-time
table generation.

**Registered predictions (the criterion predicting where its own answer changes):**
- **M1.** The 3B-policy cells, null on GSM8K (consensus 86%), become significant judge wins on
  MATH (consensus 52%) for the strong judges (32B, 72B) at N = 4 — the criterion's consensus bar
  drops below plausible judge q. The 14B judge is the boundary case and may land either side.
- **M2.** The criterion's necessity remains exceptionless: no cell beats majority with
  q_judge ≤ mode-correctness (now computed per-problem on MATH).
- **M3.** The allocation conclusion survives: every new significant win remains FLOP-dominated by
  a judge-free configuration (bigger policy + majority on MATH), because policy scaling gains even
  more on the harder benchmark. s ≈ 0 stands on MATH too — the judge-viable *region* widens while
  the judge-optimal *allocation* does not appear.
- **M4.** The N tax persists: N = 4 wins outnumber and outrank N = 16 wins on MATH as well.
If M1 lands and M3 holds, the paper's §8 hedge becomes a §3 cross-benchmark result and the venue
calculus changes materially. If M1 misses (judges stay null despite weak consensus), that is a
strike against the criterion's sufficiency framing and gets §7-scored at full prominence.

## Run 2 — Llama family check (small, rides along)

Llama-3.1-8B-Instruct (4-bit) as judge over the existing GSM8K caches: cells {0.5B, 1.5B, 3B}
policies × N {4, 16}, pick-best, --perproblem; plus the C3 position diagnostic at N = 16 on the
7B policy (60 problems, position logging).

**Registered predictions:**
- **L1.** The criterion transfers: Llama-8B's wins/losses are predicted by its measured q vs each
  policy's mode-correctness, same as the Qwen judges — the criterion is about quantities, not
  family.
- **L2.** Position collapse replicates in-family-out: edge-choice rate at N = 16 significantly
  above uniform (the phenomenon is a property of long-list judging, not of Qwen).

## Paper integration (after results, not before)

- Build script consumes the new JSONs; MATH subsection added to §3 with its own win table and
  criterion 2×2; §7 gains rows M1–M4, L1–L2 scored either way; §8's cross-benchmark paragraph
  rewritten from prediction to result; abstract gains one sentence only if M1 hits.
- Definition of done: every new cell in the repo with per-problem data; predictions scored with
  exact p-values; grep checks per the standing rules.

## Not in scope
No full MATH grid (the subset answers the reviewer question); no weak-judge MATH cells beyond the
7B boundary case; no prompt engineering; P3 stays in ET-IV.
