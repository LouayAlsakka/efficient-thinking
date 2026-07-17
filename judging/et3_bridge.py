#!/usr/bin/env python
"""ET-III C1 + C2 over the per-problem-logged pick-best grid (e3_pp_*.json).

C1 — per-cell paired McNemar for every claimed judge win: no cell is called a win in the paper without its
     p-value. Expectation: weak-policy×big-judge wins survive; marginal competent-policy wins do not.
C2 — the mode-correctness bridge (Paper III ↔ Paper II's Prop 2). Per cell:
       q_judge         = P(judge picks correct | a correct answer is present)  — the judge's intrinsic quality
       mode_correct    = majority accuracy = fraction of problems whose modal answer is correct (Prop 2 quantity)
     Law: the judge should beat majority exactly where q_judge (times coverage) clears mode_correct. We report
     the 2x2 of predicted-win (q_judge > mode_correct) vs observed-win (pick_acc > maj_acc) across all cells.

  python judging/et3_bridge.py
"""
import glob, json, math


def mcnemar(a, b):
    disc_b = sum(1 for x, y in zip(a, b) if x and not y)   # judge right, majority wrong
    disc_c = sum(1 for x, y in zip(a, b) if not x and y)   # majority right, judge wrong
    n = disc_b + disc_c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(disc_b, disc_c) + 1)) * 0.5 ** n) if n else 1.0
    return disc_b, disc_c, p


rows = []
for f in glob.glob("judging/e3_pp_*.json"):
    rows += [r for r in json.load(open(f)) if "pick_ok" in r]
rows.sort(key=lambda r: (r["policy_params"], r["judge_params"], r["N"]))

print(f"Loaded {len(rows)} logged cells.\n=== C1: per-cell McNemar for claimed judge wins (pick > majority) ===")
c1 = []
for r in rows:
    won = r["pick_best_acc"] > r["majority_acc"]
    if not won:
        continue
    b, c, p = mcnemar(r["pick_ok"], r["maj_ok"])
    sig = "SURVIVES" if p < 0.05 else ("marginal" if p < 0.10 else "NOT sig")
    c1.append({**{k: r[k] for k in ("policy", "judge_params", "N")}, "delta": round(r["pick_best_acc"] - r["majority_acc"], 1),
               "b": b, "c": c, "p": round(p, 4), "verdict": sig})
    print(f"  p={r['policy']:>5} j={r['judge_params']:>4}B N={r['N']:>2}  Δ={r['pick_best_acc']-r['majority_acc']:+5.1f}  "
          f"b={b} c={c}  McNemar p={p:.4f}  → {sig}")

print("\n=== C2: mode-correctness bridge — predicted (q_judge > mode_correct) vs observed (pick > maj) ===")
tp = tn = fp = fn = 0; c2 = []
for r in rows:
    cov_idx = [i for i, cv in enumerate(r["cov"]) if cv]
    q_judge = (sum(r["pick_ok"][i] for i in cov_idx) / len(cov_idx)) if cov_idx else 0.0
    mode_correct = r["majority_acc"] / 100.0
    predicted = q_judge > mode_correct
    observed = r["pick_best_acc"] > r["majority_acc"]
    tp += predicted and observed; tn += (not predicted) and (not observed)
    fp += predicted and not observed; fn += (not predicted) and observed
    c2.append({**{k: r[k] for k in ("policy", "judge_params", "N")}, "q_judge": round(100 * q_judge, 1),
               "mode_correct": round(100 * mode_correct, 1), "predicted_win": predicted, "observed_win": observed})
n = len(rows)
print(f"  2x2 over {n} cells: predicted&observed={tp}  neither={tn}  predicted-only={fp}  observed-only={fn}")
print(f"  law agreement = {100*(tp+tn)/n:.0f}%  (the relative-competence law: judge helps iff q_judge > policy mode-correctness)")

json.dump({"c1_mcnemar": c1, "c2_bridge": c2,
           "c2_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "agreement_pct": round(100*(tp+tn)/n, 1)}},
          open("judging/et3_bridge.json", "w"), indent=2)
print("\n[bridge] wrote judging/et3_bridge.json")
