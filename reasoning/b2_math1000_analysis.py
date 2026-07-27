#!/usr/bin/env python
"""B2 — score the MATH n=1000 rerun of §4's crossover flip (7B+sc@16 vs 14B true-greedy).

Tightens the shipped §4 result (n=500, p=0.0027). Fresh seed-0 n=1000 draw from full MATH test
(math1000.jsonl; gold-extractor gated 100% on MATH-500). True greedy decoded separately (flip-decider
rule). Canonical reason_cache extractors so greedy and sc@N carry no normalizer drift. mlx-free.

  ./.venv/bin/python reasoning/b2_math1000_analysis.py
"""
import json, math, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reason_cache import extract_boxed, normalize

C = "reasoning/cache"

def load(fn): return [json.loads(l) for l in open(fn)]

def per_problem(cache, N=None):
    ext = lambda t: normalize(extract_boxed(t))
    out = []
    for it in load(cache):
        g = normalize(str(it["gold"]))
        if N is None:
            out.append(ext(it["samples"][0]) == g)
        else:
            ans = [ext(s) for s in it["samples"][:N]]
            mc = Counter([x for x in ans if x]).most_common(1)
            out.append(bool(mc and str(mc[0][0]) == g))
    return out

def acc(pp): return round(100 * sum(pp) / len(pp), 1)

def scores(cache):
    items = load(cache); nmax = min(len(it["samples"]) for it in items)
    ext = [([normalize(extract_boxed(s)) for s in it["samples"][:nmax]], normalize(str(it["gold"]))) for it in items]
    n = len(ext)
    row = {"n": n, "nmax": nmax}
    row["pass1"] = round(100 * sum(sum(str(x) == g for x in a) / len(a) for a, g in ext) / n, 1)
    for N in (4, 16):
        if N > nmax: continue
        sc = sum(bool((mc := Counter([x for x in a[:N] if x]).most_common(1)) and str(mc[0][0]) == g) for a, g in ext)
        orc = sum(any(str(x) == g for x in a[:N]) for a, g in ext)
        row[f"sc{N}"] = round(100 * sc / n, 1); row[f"oracle{N}"] = round(100 * orc / n, 1)
    return row

def mcnemar(A, B):
    m = min(len(A), len(B)); A, B = A[:m], B[:m]
    b = sum(1 for a, x in zip(A, B) if a and not x)
    c = sum(1 for a, x in zip(A, B) if not a and x)
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 0.5 ** n) if n else 1.0
    return sum(A) / m * 100, sum(B) / m * 100, b, c, p, m

s7, s14 = scores(f"{C}/math1000_7B.jsonl"), scores(f"{C}/math1000_14B.jsonl")
g7, g14 = acc(per_problem(f"{C}/math1000_7B_greedy.jsonl")), acc(per_problem(f"{C}/math1000_14B_greedy.jsonl"))

print("=" * 78)
print("B2 — MATH n=1000 rerun (Qwen2.5-Instruct-4bit; fresh seed-0 draw from full MATH test)")
print("=" * 78)
for name, s, g in [("7B", s7, g7), ("14B", s14, g14)]:
    print(f"  {name:>3}  true-greedy={g:5.1f}  pass1={s['pass1']:5.1f}  sc4={s['sc4']:5.1f}  "
          f"sc16={s['sc16']:5.1f}  oracle16={s['oracle16']:5.1f}")

A = per_problem(f"{C}/math1000_7B.jsonl", 16)          # 7B + self-consistency@16
B = per_problem(f"{C}/math1000_14B_greedy.jsonl")       # 14B TRUE greedy
accA, accB, b, c, p, m = mcnemar(A, B)
sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else ("marginal" if p < 0.10 else "n.s.")
print("\nRegistered flip — 7B+sc@16 vs 14B TRUE-greedy (paired exact McNemar):")
print(f"   search {accA:.1f}%  vs  14B-greedy {accB:.1f}%   (Δ={accA-accB:+.1f})   n={m}")
print(f"   discordant: 7B-search-right b={b}, 14B-greedy-right c={c}   exact p={p:.5f}  → {sig}")
print(f"\n   reference (shipped §4, n=500): 7B+sc16 72.8 vs 14B-greedy 68.2, b/c=39/16, p=0.0027")

out = {"benchmark": "MATH", "n": s7["n"], "set": "math1000.jsonl (seed-0 full-MATH-test draw)",
       "models": {"7B": "mlx-community/Qwen2.5-7B-Instruct-4bit", "14B": "mlx-community/Qwen2.5-14B-Instruct-4bit"},
       "rows": {"7B": {**s7, "true_greedy": g7}, "14B": {**s14, "true_greedy": g14}},
       "flip_7Bsc16_vs_14Bgreedy": {"search_acc": accA, "greedy_acc": accB, "delta": round(accA - accB, 1),
                                    "b": b, "c": c, "p": round(p, 5), "n": m}}
json.dump(out, open("reasoning/b2_math1000_scores.json", "w"), indent=2)
print("\n[b2] wrote reasoning/b2_math1000_scores.json")
