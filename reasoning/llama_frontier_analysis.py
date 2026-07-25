#!/usr/bin/env python
"""Cross-family frontier ladder — Llama-3.x analysis (ET-II extraction robustness).

Consumes the frozen Llama sample caches (32 GSM8K / 16 MATH) and the SEPARATE true-greedy caches
(the flip-decider rule — no first-sample-as-greedy), replicates the ET-II frontier scoring, runs the
adjacent-size McNemar flips against true greedy, and scores registered predictions X1-X5. mlx-free.

  python reasoning/llama_frontier_analysis.py
"""
import json, math, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the CANONICAL extractors/normalizer that produced the ET-II scored jsons, so true-greedy and
# sc@N are scored identically (no normalizer drift between the greedy and sample caches).
from reason_cache import extract as extract_gsm, extract_boxed, normalize as norm

CACHE = "reasoning/cache"


def load(fn):
    return [json.loads(l) for l in open(fn)]      # JSONL caches (one object per line)


def loadj(fn):
    return json.load(open(fn))                     # pretty-printed JSON score arrays


def per_problem(cache, math_mode, N=None):
    """Per-problem correctness. N=None → single-shot samples[0] (used for the true-greedy caches);
    else sc@N majority vote over the sample cache."""
    ext = (lambda t: norm(extract_boxed(t))) if math_mode else extract_gsm
    goldf = (lambda g: norm(str(g))) if math_mode else (lambda g: str(g))
    out = []
    for it in load(cache):
        g = goldf(it["gold"])
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


def acc(cache, math_mode, N=None):
    pp = per_problem(cache, math_mode, N)
    return round(100 * sum(pp) / len(pp), 1)


# ---- true-greedy pass@1 (flip-decider rule) --------------------------------------------------
LADDER = ["1B", "3B", "8B"]
greedy = {}
for bench, mm in [("math", True), ("gsm8k", False)]:
    for s in LADDER:
        greedy[(bench, s)] = acc(f"{CACHE}/llama_{bench}_{s}_greedy.jsonl", mm)

# ---- sample-cache frontier (pass1 sampled / sc / oracle already in the scored jsons) ----------
math_scores = {r["cache"]: r for r in loadj("reasoning/llama_math_frontier_scores.json")}
gsm_scores = {r["cache"]: r for r in loadj("reasoning/llama_frontier_powered_scores.json")}


def row(bench, s):
    d = (math_scores if bench == "math" else gsm_scores)[f"llama_{bench}_{s}.jsonl"]
    Nmax = d["nmax"]
    scmax = d[f"sc{Nmax}"]
    return {"model": s, "true_greedy": greedy[(bench, s)], "pass1_sampled": d["pass1"],
            "sc4": d["sc4"], f"sc{Nmax}": scmax, f"oracle{Nmax}": d[f"oracle{Nmax}"],
            "lift_vs_greedy": round(scmax - greedy[(bench, s)], 1),
            "lift_vs_pass1": round(scmax - d["pass1"], 1), "Nmax": Nmax}


results = {"family": "Llama-3.x-Instruct-4bit (mlx-community)",
           "models": {"1B": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                      "3B": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                      "8B": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"},
           "note_70B": "No Llama-70B 4-bit MLX conversion present locally; three-point {1B,3B,8B} ladder "
                       "per spec fallback (8B bracketed from below, upper bracket absent).",
           "math": [row("math", s) for s in LADDER],
           "gsm8k": [row("gsm8k", s) for s in LADDER]}

print("=" * 92)
print("CROSS-FAMILY FRONTIER LADDER — Llama-3.x-Instruct-4bit  (n=500 each; MATH nmax=16, GSM8K nmax=32)")
print("=" * 92)
for bench in ("math", "gsm8k"):
    Nmax = results[bench][0]["Nmax"]
    print(f"\n{bench.upper()}  (true-greedy p@1 | sampled pass1 | sc4 | sc{Nmax} | oracle{Nmax} | lift sc-greedy)")
    for r in results[bench]:
        print(f"  {r['model']:>3}  greedy={r['true_greedy']:5.1f}  pass1={r['pass1_sampled']:5.1f}  "
              f"sc4={r['sc4']:5.1f}  sc{Nmax}={r[f'sc{Nmax}']:5.1f}  oracle{Nmax}={r[f'oracle{Nmax}']:5.1f}  "
              f"lift={r['lift_vs_greedy']:+5.1f}")

# ---- adjacent-size McNemar flips: search on smaller vs TRUE greedy of next size up ------------
print("\n" + "=" * 92)
print("ADJACENT-SIZE McNEMAR — search on smaller model  vs  TRUE-GREEDY of next size up (paired, exact)")
print("=" * 92)
FLIPS = [
    ("MATH:  1B+sc@16 vs 3B greedy", "math", "1B", 16, "3B", True),
    ("MATH:  3B+sc@16 vs 8B greedy", "math", "3B", 16, "8B", True),
    ("GSM8K: 1B+sc@32 vs 3B greedy", "gsm8k", "1B", 32, "3B", False),
    ("GSM8K: 3B+sc@32 vs 8B greedy", "gsm8k", "3B", 32, "8B", False),
]
flip_rows = []
for label, bench, small, N, big, mm in FLIPS:
    A = per_problem(f"{CACHE}/llama_{bench}_{small}.jsonl", mm, N)
    B = per_problem(f"{CACHE}/llama_{bench}_{big}_greedy.jsonl", mm, None)
    accA, accB, b, c, p, m = mcnemar(A, B)
    verdict = "SEARCH WINS (p<0.05)" if (p < 0.05 and accA > accB) else \
              ("BIGGER WINS (p<0.05)" if (p < 0.05 and accB > accA) else "PARITY (n.s.)")
    print(f"\n{label}  (n={m})")
    print(f"   search {accA:.1f}%  vs  bigger-greedy {accB:.1f}%   (Δ={accA-accB:+.1f})")
    print(f"   discordant: search-right b={b}, bigger-right c={c}   exact McNemar p={p:.4f}  → {verdict}")
    flip_rows.append({"flip": label, "search_acc": accA, "bigger_greedy_acc": accB,
                      "delta": round(accA - accB, 1), "b": b, "c": c, "p": round(p, 4), "verdict": verdict})
results["mcnemar_flips"] = flip_rows

# ---- X5 overlay data: lift-vs-competence, both families, both benchmarks ----------------------
# Qwen reference frontiers (sampled pass1 convention, lift = sc@Nmax - pass1)
def overlay(scores_json, family, bench):
    pts = []
    for r in loadj(scores_json):
        Nmax = r["nmax"]
        pts.append({"family": family, "bench": bench, "model": r["cache"],
                    "pass1": r["pass1"], "lift": round(r[f"sc{Nmax}"] - r["pass1"], 1), "Nmax": Nmax})
    return pts

overlay_pts = []
overlay_pts += overlay("reasoning/frontier_powered_scores.json", "Qwen2.5", "gsm8k")
overlay_pts += overlay("reasoning/math_frontier_scores.json", "Qwen2.5", "math")
overlay_pts += overlay("reasoning/llama_frontier_powered_scores.json", "Llama3.x", "gsm8k")
overlay_pts += overlay("reasoning/llama_math_frontier_scores.json", "Llama3.x", "math")
overlay_pts.sort(key=lambda d: d["pass1"])
results["x5_overlay_lift_vs_pass1"] = overlay_pts

json.dump(results, open("reasoning/llama_frontier_analysis.json", "w"), indent=2)
print("\n[analysis] wrote reasoning/llama_frontier_analysis.json")
