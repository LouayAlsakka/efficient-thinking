#!/usr/bin/env python
"""Item 2 (MATH mode-correctness) — scores §8's registered direction: weaker free consensus widens the
judge-viable region. Pure post-processing on Paper II's frozen MATH caches (reasoning/cache/math_*.jsonl,
500 problems x 16 samples, boxed answers) — zero generation. For each policy we compute mode-correctness
(= majority-vote accuracy = fraction of problems whose modal boxed answer equals gold) at N=16, and compare
to the same quantity on GSM8K (from the committed e3_pp grid). GSM8K's consensus is unusually strong, which
starves judges; if MATH's per-policy mode-correctness is lower, the two-factor law predicts a larger region
where q_judge can clear it — upgrading the paper from single-benchmark to cross-benchmark prediction.

  python judging/math_mode_correctness.py
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reasoning"))
from reason_cache import extract_boxed, normalize   # reuse Paper II's exact MATH extraction/normalization

N = 16
POLICIES = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]


def math_mode_correct(tag):
    path = f"reasoning/cache/math_{tag}.jsonl"
    if not os.path.exists(path):
        return None
    items = [json.loads(l) for l in open(path)]
    ok = 0
    for it in items:
        g = normalize(str(it["gold"]))
        ans = [normalize(extract_boxed(s)) for s in it["samples"][:N]]
        mc = Counter([a for a in ans if a]).most_common(1)
        ok += bool(mc and mc[0][0] == g)
    return round(100 * ok / len(items), 1), len(items)


# GSM8K mode-correctness per policy (majority_acc @N=16) from the committed pick-best grid
gsm = {}
for f in ("1.5B", "3B", "7B", "14B", "32B", "72B"):
    for r in json.load(open(f"judging/e3_pp_{f}.json")):
        if r["N"] == 16:
            gsm[r["policy"]] = r["majority_acc"]   # same across judges; policy-level quantity

print(f"{'policy':>7} {'GSM8K mode':>11} {'MATH mode':>10} {'Δ (MATH−GSM8K)':>16}")
rows = []
for p in POLICIES:
    m = math_mode_correct(p)
    g = gsm.get(p)
    if m is None:
        continue
    macc, n = m
    d = (round(macc - g, 1) if g is not None else None)
    rows.append({"policy": p, "gsm8k_mode_correct": g, "math_mode_correct": macc, "math_n": n, "delta": d})
    print(f"  {p:>5} {('' if g is None else f'{g:>9}'):>11} {macc:>10} {('' if d is None else f'{d:+.1f}'):>16}")

lower = [r for r in rows if r["delta"] is not None and r["delta"] < 0]
print(f"\n[MATH] policies with weaker MATH consensus than GSM8K (Δ<0): {len(lower)}/"
      f"{len([r for r in rows if r['delta'] is not None])}")
print("[MATH] §8 registered direction (weaker consensus on MATH → wider judge-viable region): "
      f"{'SUPPORTED' if lower and len(lower) == len([r for r in rows if r['delta'] is not None]) else 'MIXED/CHECK'}")
json.dump({"N": N, "rows": rows}, open("judging/math_mode_correctness.json", "w"), indent=2)
print("[MATH] wrote judging/math_mode_correctness.json")
