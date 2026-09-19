# Inputs manifest — where every paper's data actually lives

**理 (nirai 11433), before 09-22.** `.gitignore` line 2 is a bare `data/`, so **every paper's
inputs are invisible to git**. A revival therefore begins with a filesystem sweep and no checklist —
which is exactly what happened to ET-III on 2026-09-19: I reported its caches ABSENT (correctly, for
the repo) when all eight sat in a sibling checkout, and an 8–14 h regeneration was nearly scheduled
on that reading.

**The data stays untracked. This file is committed.** It is the checklist that was missing.

⚠️ **A path here is a claim about ONE BOX.** Every row below was verified on **llm1** on 2026-09-19.
A row does not mean the file exists anywhere else, and llm1's `/var/tmp` is not durable (none of
these are under it). Replication status is in §Replication.


### III — judging (et3_judge.py)

| path | size | lines | samples/problem | md5 | note |
|---|---|---|---|---|---|
| `~/chess-scaling/reasoning/data/gsm8k_test.jsonl` | 0.7 MB | 1319 | — | `6493e22fc90d491ae8da0b88ffcfebae` | problem set, GSM8K |
| `~/chess-scaling/reasoning/data/math500.jsonl` | 0.4 MB | 500 | — | `d06a3ab6a17029154ffbb29db65c158b` | problem set, MATH |
| `~/chess-scaling/reasoning/cache/gsm8k_0.5B.jsonl` | 24.3 MB | 500 | 32 | `eb1a026ee9178b4f6a22f0e74cb8186e` | policy cache |
| `~/chess-scaling/reasoning/cache/math_0.5B.jsonl` | 10.1 MB | 300 | 16 | `290148ef3fcd7030a3ef5ca922aae689` | policy cache |
| `~/chess-scaling/reasoning/cache/gsm8k_1.5B.jsonl` | 13.7 MB | 500 | 32 | `3e876a1b6a73c29712dc68b9fa20fc1b` | policy cache |
| `~/chess-scaling/reasoning/cache/math_1.5B.jsonl` | 7.4 MB | 300 | 16 | `ce93b9e968ebcd086b6e6d1593a86282` | policy cache |
| `~/chess-scaling/reasoning/cache/gsm8k_3B.jsonl` | 17.0 MB | 500 | 32 | `8397dc954cfc589ab7d55cc12351d154` | policy cache |
| `~/chess-scaling/reasoning/cache/math_3B.jsonl` | 8.0 MB | 300 | 16 | `9fa776cd9fa4a4723e702c02267c2e8c` | policy cache |
| `~/chess-scaling/reasoning/cache/gsm8k_7B.jsonl` | 14.8 MB | 500 | 32 | `cf7d38378c761f1decda946838e75ac9` | policy cache |
| `~/chess-scaling/reasoning/cache/math_7B.jsonl` | 13.7 MB | 500 | 16 | `b0a6d86ab4231d3ae1333251b6816657` | policy cache |

### IV — creative (poetry/)

| path | size | lines | samples/problem | md5 | note |
|---|---|---|---|---|---|
| `~/github/efficient-thinking/poetry/data/canon_reference.txt` | 0.0 MB | 70 | — | `fe0f0685ff8c9b971453346976288797` | canon corpus |
| `~/github/efficient-thinking/poetry/data/e1_prompts.jsonl` | 0.1 MB | 300 | — | `256002ed8a2be60051bb4238befe01d2` | E1 prompts |
| `~/github/efficient-thinking/poetry/data/goodhart_dev_prompts.jsonl` | 0.0 MB | 27 | — | `a135209f986b5641fc50f7bcb12ba448` | dev prompts |

### VI — label fidelity (games/)

Inputs are committed RESULTS, not untracked data: `games/results/et6_ondist_{supervised,scratch}.json`,
`et6_decomp{,_big,_supervised}.json`, `et6_f3*.json`, `runs/et6_f4.json`. E-B's ledger can be written
from the repo alone — **no sweep needed and no regeneration**.

### VII — the elicitation gap

**No inputs of its own, by construction.** E-E probes the III judging cells, so VII inherits III's
inputs exactly; if III's rows above are good, VII E-E is unblocked.

## Replication

| file set | llm1 | llm2 | verified |
|---|---|---|---|
| III data (2 files, 1.2 MB) | ✅ `~/chess-scaling/reasoning/data/` | ✅ `~/et_inputs/reasoning/data/` | md5 both ends, 2026-09-19 |
| III caches (8 files, 104 MB) | ✅ `~/chess-scaling/reasoning/cache/` | ✅ `~/et_inputs/reasoning/cache/` | md5 both ends, 2026-09-19 |

✅ **"Exists only on llm1" is no longer true for III.** All ten files were copied to llm2 and
**every one verified by md5 on BOTH ends** — a copy that completes is not a copy that is correct,
and `scp` exiting 0 is not evidence. The originals were read only.

⚠️ **IV's three files are still single-box (llm1, in the repo's own untracked `poetry/data/`).**
Not replicated: IV is parked pending a rater, so nothing is scheduled against them — but the row
stays here so it is not forgotten when it unparks.

## How to verify a row rather than trust it

```
  md5 -q <path>                     # must match the md5 column
  wc -l < <path>                    # must match lines
  python -c "import json;print(len(json.loads(open('<path>').readline())['samples']))"
```

⚠️ **This manifest records SIZE, COUNTS and MD5 — not provenance.** That these caches are the ones
ET-III's published cells were computed from is **unverified**. 理's gate stands: a P3 rerun must
reproduce one published III cell (same N, same judge) before any new cell is claimed.

— Sautée (沙汰), studio lane
