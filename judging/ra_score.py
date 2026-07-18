#!/usr/bin/env python
"""R-a scoring — the law-mismatch cells: pairwise judges {7B,14B} x policy 0.5B, where the judge's
on-distribution q (7B=0.414, 14B=0.509) EXCEEDS the 0.5B policy's mode-correctness (0.367). The
relative-competence law predicts a judge WIN here; pick-best drowned these cells (list position bias),
pairwise is the test. No cell is called a win without a paired McNemar p-value.

  python judging/ra_score.py
"""
import glob, json, math


def mcnemar(a, b):
    disc_b = sum(1 for x, y in zip(a, b) if x and not y)   # judge (pairwise pick) right, majority wrong
    disc_c = sum(1 for x, y in zip(a, b) if not x and y)   # majority right, judge wrong
    n = disc_b + disc_c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(disc_b, disc_c) + 1)) * 0.5 ** n) if n else 1.0
    return disc_b, disc_c, p


rows = []
for f in sorted(glob.glob("judging/e3_ra_*_p05.json")):
    rows += [r for r in json.load(open(f)) if "pick_ok" in r]
rows.sort(key=lambda r: (r["judge_params"], r["N"]))

print(f"Loaded {len(rows)} R-a cells (pairwise, policy 0.5B).\n"
      f"=== R-a: per-cell paired McNemar, pairwise-pick vs majority ===")
out = []
for r in rows:
    b, c, p = mcnemar(r["pick_ok"], r["maj_ok"])
    d = r["pick_best_acc"] - r["majority_acc"]
    sig = "WIN (sig)" if (p < 0.05 and d > 0) else ("win-marginal" if (p < 0.10 and d > 0)
          else ("parity" if abs(d) < 1e-9 or p >= 0.10 else ("LOSS (sig)" if d < 0 else "n.s.")))
    out.append({"judge_params": r["judge_params"], "N": r["N"], "n": r["n"],
                "pick_acc": r["pick_best_acc"], "maj_acc": r["majority_acc"], "delta": round(d, 1),
                "b_judge_right": b, "c_maj_right": c, "p": round(p, 4), "verdict": sig})
    print(f"  judge={r['judge_params']:>4}B N={r['N']:>2}  pick={r['pick_best_acc']:5} "
          f"maj={r['majority_acc']:5}  Δ={d:+5.1f}  b={b} c={c}  McNemar p={p:.4f}  → {sig}")

json.dump(out, open("judging/ra_score.json", "w"), indent=2)
print("\n[R-a] wrote judging/ra_score.json")
