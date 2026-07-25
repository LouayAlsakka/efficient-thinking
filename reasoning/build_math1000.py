#!/usr/bin/env python
"""B2 data prep — build a registered n=1000 MATH set to tighten §4's crossover p-value beyond n=500.

The ET-II MATH result used HuggingFaceH4/MATH-500 (n=500). To rerun at higher power we need >500 MATH
problems with clean gold. We derive gold = \\boxed{} content of the full-MATH solution, but ONLY after
VALIDATING that extractor against MATH-500's committed `answer` field — the checker-discipline gate: a
silent gold-extraction bug would corrupt the whole overnight. Gate must pass ≥99% before we build.

  ./.venv/bin/python reasoning/build_math1000.py
"""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reason_cache import extract_boxed, normalize
from datasets import load_dataset, concatenate_datasets

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]

# --- 1. VALIDATION GATE: extractor vs MATH-500's committed gold ------------------------------------
m500 = [json.loads(l) for l in open("reasoning/data/math500.jsonl")]
ok = sum(normalize(extract_boxed(r["solution"])) == normalize(r["answer"]) for r in m500)
rate = 100 * ok / len(m500)
print(f"[gate] boxed-extractor vs MATH-500 committed answers: {ok}/{len(m500)} = {rate:.1f}%")
if rate < 99.0:
    print("[gate] FAIL (<99%) — extractor unreliable, refusing to build. Fix before B2 runs.")
    sys.exit(1)
print("[gate] PASS — extractor trustworthy on full MATH.")

# --- 2. Load full MATH test, derive clean gold ----------------------------------------------------
parts = []
for s in SUBJECTS:
    d = load_dataset("EleutherAI/hendrycks_math", s, split="test")
    parts.append(d)
full = concatenate_datasets(parts)
print(f"[math] full test loaded: {len(full)} problems across {len(SUBJECTS)} subjects")

rows = []
for r in full:
    g = normalize(extract_boxed(r["solution"]))
    if g:                                   # keep only cleanly-extractable gold
        rows.append({"problem": r["problem"], "answer": g, "level": r.get("level"), "type": r.get("type")})
print(f"[math] clean-gold problems: {len(rows)}/{len(full)}")

# --- 3. Registered seed-0 sample of 1000 ----------------------------------------------------------
random.Random(0).shuffle(rows)
sample = rows[:1000]
with open("reasoning/data/math1000.jsonl", "w") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")
print(f"[math] wrote reasoning/data/math1000.jsonl (n={len(sample)}, seed-0 shuffle of clean-gold full MATH test)")
