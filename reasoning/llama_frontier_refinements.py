#!/usr/bin/env python
"""Cross-family frontier — the two refinements that beat the registered predictions (post-hoc, flagged).

R1  Lift-vs-gap flip law (sharper than X1's "~7B"): at rung K, search flips the ordering against the next
    size class exactly when its lift clears the adjacent-class greedy gap it must jump:
        flip(K)  <=>  lift(K) = sc@Nmax(K) - pass1(K)   >=   gap(K) = pass1(K+1) - pass1(K).
    Computed on the SAME sampled-pass1 comparator for both families (true greedy is only partial for Qwen;
    sampled pass1 is present in every committed cell). The McNemar 'beats' claims use TRUE greedy separately.

R2  Coverage-gated inverted-U: lift is not monotone in competence. Prop-2 coverage (oracle@N) gates the
    left limb — you cannot select an answer that is never sampled — so lift ASCENDS out of the low-competence
    floor, peaks, then collapses under saturation. Qwen's ladder sampled the descending limb; Llama's the ascent.

  ./.venv/bin/python reasoning/llama_frontier_refinements.py
"""
import json, re

def loadj(fn): return json.load(open(fn))

def size_of(cache):
    m = re.search(r"(\d+\.?\d*)B", cache)
    return float(m.group(1)) if m else 0.0

def rung_table(scores_json):
    rows = []
    for r in loadj(scores_json):
        Nmax = r["nmax"]
        rows.append({"size": size_of(r["cache"]), "cache": r["cache"], "pass1": r["pass1"],
                     "sc": r[f"sc{Nmax}"], "oracle": r[f"oracle{Nmax}"], "Nmax": Nmax})
    rows.sort(key=lambda d: d["size"])
    return rows

FAMILIES = {
    ("Qwen2.5", "gsm8k"): "reasoning/frontier_powered_scores.json",
    ("Qwen2.5", "math"):  "reasoning/math_frontier_scores.json",
    ("Llama3.x", "gsm8k"): "reasoning/llama_frontier_powered_scores.json",
    ("Llama3.x", "math"):  "reasoning/llama_math_frontier_scores.json",
}

out = {"R1_lift_vs_gap": [], "R2_inverted_U": []}
print("R1 — LIFT-vs-GAP FLIP LAW  (flip at K  <=>  lift(K) >= adjacent greedy gap(K))")
print("     [pass1 comparator; the crossover rung is the lowest K that clears its gap]\n")
for (fam, bench), fn in FAMILIES.items():
    rows = rung_table(fn)
    print(f"{fam} / {bench}:")
    for i in range(len(rows) - 1):
        k, kn = rows[i], rows[i + 1]
        lift = round(k["sc"] - k["pass1"], 1)
        gap = round(kn["pass1"] - k["pass1"], 1)
        flip = lift >= gap
        mark = "  <== FLIP (lift clears gap)" if flip else ""
        print(f"   {k['size']:>4}B  lift={lift:+5.1f}   gap→{kn['size']:.0f}B={gap:+5.1f}   "
              f"{'lift≥gap' if flip else 'lift<gap':>8}{mark}")
        out["R1_lift_vs_gap"].append({"family": fam, "bench": bench, "rung": k["size"],
                                      "next": kn["size"], "lift": lift, "gap": gap, "flip": flip})
    print()

# R2 — lift vs competence, pooled, sorted; show the ascending limb → peak → collapse
print("R2 — COVERAGE-GATED INVERTED-U  (lift vs pass1, pooled; oracle gates the low-competence limb)\n")
pts = []
for (fam, bench), fn in FAMILIES.items():
    for r in rung_table(fn):
        pts.append({"family": fam, "bench": bench, "size": r["size"], "pass1": r["pass1"],
                    "lift": round(r["sc"] - r["pass1"], 1), "oracle_gap": round(r["oracle"] - r["sc"], 1)})
pts.sort(key=lambda d: d["pass1"])
for p in pts:
    bar = "#" * int(round(p["lift"]))
    print(f"   pass1={p['pass1']:5.1f}  lift={p['lift']:+5.1f}  oracle-sc={p['oracle_gap']:+5.1f}  "
          f"{p['family']:8} {p['bench']:5} {bar}")
out["R2_inverted_U"] = pts
json.dump(out, open("reasoning/llama_frontier_refinements.json", "w"), indent=2)
print("\n[refine] wrote reasoning/llama_frontier_refinements.json")
