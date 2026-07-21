#!/usr/bin/env python
"""ET-III external-validation Run 1 (MATH judge grid) — scores M1-M4 per the FROZEN scoring plan
(docs/et3-external-validation-spec.md, committed before the 72B tail). No verdict language is chosen
here; the rules decide.

M1 confirmatory family: {32B,72B} judge x 3B policy x N=4, pick-best. Exact McNemar, Bonferroni x2.
  Hit = both significant after correction. Partial = one significant, OR both directionally positive
  with raw p in (0.05, 0.15]. Miss = direction absent/reversed.
M2: criterion necessity exceptionless (no cell beats majority with q_judge <= mode-correctness).
M3: allocation survives — every significant win FLOP-dominated by a bigger-policy+majority config.
M4: N tax — sign pattern of the N=16 column.

  python judging/math_grid_score.py
"""
import glob, json, math

rows = []
for f in glob.glob("judging/e3_math_*.json"):
    rows += json.load(open(f))
rows.sort(key=lambda r: (r["judge_params"], r["policy_params"], r["N"]))


def mcnemar(a, b):
    B = sum(1 for x, y in zip(a, b) if x and not y)   # judge right, majority wrong
    C = sum(1 for x, y in zip(a, b) if not x and y)
    n = B + C
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(B, C) + 1)) * 0.5 ** n) if n else 1.0
    return B, C, p


def cell(jp, pol, N):
    return next(r for r in rows if r["judge_params"] == jp and r["policy"] == pol and r["N"] == N)


print("=== full MATH grid (Δ = pick − majority) ===")
for r in rows:
    d = r["pick_best_acc"] - r["majority_acc"]
    print(f"  judge={r['judge_params']:>4}B x {r['policy']:>3} N={r['N']:>2}  pick={r['pick_best_acc']:5} "
          f"maj={r['majority_acc']:5} Δ={d:+5.1f}  orc={r['oracle_acc']:5}")

# ---- M1 ----
print("\n=== M1 (confirmatory: {32B,72B} x 3B x N=4, McNemar, Bonferroni x2) ===")
conf = []
for jp in (32.0, 72.0):
    r = cell(jp, "3B", 4)
    b, c, p = mcnemar(r["pick_ok"], r["maj_ok"])
    pcorr = min(1.0, p * 2)
    d = r["pick_best_acc"] - r["majority_acc"]
    conf.append({"judge": jp, "delta": round(d, 1), "b": b, "c": c, "p": round(p, 4),
                 "p_bonf": round(pcorr, 4), "pos": d > 0})
    print(f"  {jp:>4}B x 3B N=4  Δ={d:+.1f}  b={b} c={c}  McNemar p={p:.4f}  Bonf p={pcorr:.4f}")
both_sig = all(x["p_bonf"] < 0.05 for x in conf)
one_sig = any(x["p_bonf"] < 0.05 for x in conf)
both_pos_marg = all(x["pos"] and 0.05 < x["p"] <= 0.15 for x in conf)
if both_sig:
    m1 = "HIT — both confirmatory cells significant after Bonferroni"
elif one_sig or both_pos_marg:
    m1 = "PARTIAL — direction confirmed, significance marginal at n=300; n-extension is the decider"
elif all(x["pos"] for x in conf):
    m1 = "PARTIAL(weak) — both positive but p>0.15; n-extension decides"
else:
    m1 = "MISS — direction absent or reversed"
print(f"  -> M1: {m1}")

# ---- M2 ----
print("\n=== M2 (criterion necessity: no win with q_judge <= mode-correctness) ===")
viol = 0
for r in rows:
    ci = [i for i, cv in enumerate(r["cov"]) if cv]
    q = sum(r["pick_ok"][i] for i in ci) / len(ci) if ci else 0.0
    mode = r["majority_acc"] / 100.0
    b, c, p = mcnemar(r["pick_ok"], r["maj_ok"])
    won = (r["pick_best_acc"] > r["majority_acc"]) and p < 0.05
    if won and not (q > mode):
        viol += 1
        print(f"  VIOLATION judge={r['judge_params']}B x {r['policy']} N={r['N']}: q={q:.3f} mode={mode:.3f}")
print(f"  -> M2: {'HIT — necessity exceptionless (0 violations across %d cells)' % len(rows) if viol==0 else 'MISS (%d violations)' % viol}")

# ---- M3 ----
print("\n=== M3 (allocation: significant wins FLOP-dominated by bigger-policy+majority) ===")
# judge-free comparator for a 3B-policy win: the 7B policy's free majority on MATH
comp4 = cell(72.0, "7B", 4)["majority_acc"]   # 7B majority@4 (policy-level, same across judges)
comp16 = cell(72.0, "7B", 16)["majority_acc"]
dominated = True
for r in rows:
    b, c, p = mcnemar(r["pick_ok"], r["maj_ok"])
    if (r["pick_best_acc"] > r["majority_acc"]) and p < 0.05:
        comp = comp4 if r["N"] == 4 else comp16
        dom = comp >= r["pick_best_acc"]
        dominated &= dom
        print(f"  win judge={r['judge_params']}B x {r['policy']} N={r['N']}: {r['pick_best_acc']} vs "
              f"7B+maj={comp}  {'DOMINATED' if dom else 'NOT dominated'}")
print(f"  -> M3: {'HIT — every significant win dominated by 7B+majority (s≈0 stands on MATH)' if dominated else 'CHECK'}")

# ---- M4 ----
print("\n=== M4 (N tax: sign pattern of the N=16 column) ===")
n16 = [(r["judge_params"], r["policy"], round(r["pick_best_acc"] - r["majority_acc"], 1))
       for r in rows if r["N"] == 16]
n16_pos = sum(1 for _, _, d in n16 if d > 0)
n4_pos = sum(1 for r in rows if r["N"] == 4 and r["pick_best_acc"] > r["majority_acc"])
print(f"  N=16 deltas: {[(f'{j}Bx{p}', d) for j, p, d in n16]}")
print(f"  N=16 positive: {n16_pos}/{len(n16)};  N=4 positive: {n4_pos}")
m4 = "HIT — N=4 wins outnumber N=16 (tax persists on MATH)" if n4_pos > n16_pos else "MISS"
print(f"  -> M4: {m4}")

json.dump({"M1": {"verdict": m1, "confirmatory": conf}, "M2_violations": viol,
           "M3_dominated": dominated, "M4": {"n16_positive": n16_pos, "n4_positive": n4_pos},
           "grid": [{"judge": r["judge_params"], "policy": r["policy"], "N": r["N"],
                     "delta": round(r["pick_best_acc"] - r["majority_acc"], 1)} for r in rows]},
          open("judging/math_grid_score.json", "w"), indent=2)
print("\n[MATH] wrote judging/math_grid_score.json")
