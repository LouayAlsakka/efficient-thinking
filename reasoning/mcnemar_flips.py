#!/usr/bin/env python
"""Paired McNemar significance for the size-vs-search crossover flips — the queued confirmation the paper
promises. Both systems are scored on the SAME problems (the frozen caches), so the comparison is paired and
McNemar (exact two-sided binomial on discordant pairs) is the right test — it asks whether the flip margin
survives once we condition on per-problem agreement, which naive independent CIs ignore.

  python reasoning/mcnemar_flips.py
"""
import json, math, re
from collections import Counter


def extract_gsm(t):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", t or "")
    if m:
        return m[-1].replace(",", "")
    n = re.findall(r"-?\d[\d,]*", t or "")
    return n[-1].replace(",", "") if n else None


def extract_boxed(t):
    t = t or ""; i = t.rfind("\\boxed")
    if i == -1:
        n = re.findall(r"-?\d[\d,]*\.?\d*", t); return n[-1].replace(",", "") if n else None
    j = t.find("{", i)
    if j == -1:
        return None
    d, out = 0, []
    for c in t[j:]:
        if c == "{":
            d += 1
            if d == 1:
                continue
        elif c == "}":
            d -= 1
            if d == 0:
                break
        out.append(c)
    return "".join(out)


def norm(s):
    if s is None:
        return None
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""), (" ", ""),
                 ("$", ""), ("dfrac", "frac"), ("tfrac", "frac"), ("\\cdot", ""), ("\\{", ""), ("\\}", "")]:
        s = s.replace(a, b)
    return s.rstrip(".").strip()


def per_problem(cache, math_mode, N=None):
    """Return list of per-problem correctness. N=None → single-shot (sample[0], 'no search'); else sc@N majority."""
    ext = (lambda t: norm(extract_boxed(t))) if math_mode else extract_gsm
    gold = (lambda g: norm(str(g))) if math_mode else (lambda g: str(g))
    out = []
    for it in (json.loads(l) for l in open(cache)):
        g = gold(it["gold"])
        if N is None:
            out.append(ext(it["samples"][0]) == g)
        else:
            ans = [ext(s) for s in it["samples"][:N]]
            mc = Counter([x for x in ans if x]).most_common(1)
            out.append(bool(mc and str(mc[0][0]) == g))
    return out


def mcnemar(A, B):
    m = min(len(A), len(B)); A, B = A[:m], B[:m]
    b = sum(1 for a, x in zip(A, B) if a and not x)   # A(search) right, B(bigger greedy) wrong
    c = sum(1 for a, x in zip(A, B) if not a and x)   # A wrong, B right
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 0.5 ** n) if n else 1.0
    return sum(A) / m * 100, sum(B) / m * 100, b, c, p, m


FLIPS = [
    ("GSM8K: 7B+sc@16  vs  14B single-shot", "reasoning/cache/gsm8k_7B.jsonl", 16, "reasoning/cache/gsm8k_14B.jsonl", False),
    ("MATH:  7B+sc@16  vs  14B single-shot", "reasoning/cache/math_7B.jsonl", 16, "reasoning/cache/math_14B.jsonl", True),
    ("GSM8K: 14B+sc@16 vs  32B single-shot", "reasoning/cache/gsm8k_14B.jsonl", 16, "reasoning/cache/gsm8k_32B.jsonl", False),
]

print("Paired McNemar — does search on the smaller model beat the bigger model with no search?\n")
for label, ca, N, cb, mm in FLIPS:
    try:
        A = per_problem(ca, mm, N); B = per_problem(cb, mm, None)
    except FileNotFoundError as e:
        print(f"{label}: SKIP (missing {e.filename})"); continue
    accA, accB, b, c, p, m = mcnemar(A, B)
    sig = "significant" if p < 0.05 else ("marginal" if p < 0.10 else "NOT significant")
    print(f"{label}  (n={m})")
    print(f"   search {accA:.1f}%  vs  bigger-greedy {accB:.1f}%   (Δ={accA-accB:+.1f})")
    print(f"   discordant: search-only-right b={b}, bigger-only-right c={c}   exact McNemar p={p:.4f}  → {sig}\n")
