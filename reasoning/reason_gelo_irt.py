#!/usr/bin/env python
"""Fit reasoning-GELO by IRT (Rasch) — places model ability θ and MATH difficulty-level difficulty on
the same logistic GELO scale as chess/Connect-4.

Reads one-or-more math_*.json (per-level accuracies from reason_math_sweep.py) and fits, in GELO units
(400 = 10×), a per-model ability θ_m and per-level difficulty d_ℓ so that
    P(model m solves a level-ℓ problem) = 1/(1+10^(-(θ_m − d_ℓ)/400)).
With ≥2 models the abilities and difficulties are jointly identified (real IRT); with one model it
still places the model against a relative difficulty ladder. Anchored so mean level difficulty = 1500.

  python reasoning/reason_gelo_irt.py --files "reasoning/math_*.json" --metric greedy
"""
import argparse, glob, json, math
import numpy as np

LN10 = math.log(10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="reasoning/math_*.json")
    ap.add_argument("--metric", default="greedy", choices=["greedy", "sc"])
    ap.add_argument("--anchor", type=float, default=1500.0, help="mean level difficulty (GELO)")
    ap.add_argument("--out", default="reasoning/reason_gelo.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.files))
    if not files:
        print(f"no files match {args.files}"); return
    models, data, levels = [], {}, set()
    for f in files:
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        models.append(m); data[m] = {}
        for lv, s in d["per_level"].items():
            lv = int(lv); n = s["n"]; k = int(round(s[args.metric] / 100.0 * n))
            data[m][lv] = (k, n); levels.add(lv)
    levels = sorted(levels)
    K, L = len(models), len(levels)
    lidx = {lv: i for i, lv in enumerate(levels)}
    beta = LN10 / 400.0                                   # logistic slope in GELO units

    theta = np.full(K, args.anchor, float); diff = np.full(L, args.anchor, float)
    lr = 1.0
    for _ in range(50000):
        gt = np.zeros(K); gd = np.zeros(L)
        for mi, m in enumerate(models):
            for lv, (k, n) in data[m].items():
                li = lidx[lv]
                p = 1.0 / (1.0 + np.exp(-beta * (theta[mi] - diff[li])))
                p = min(max(p, 1e-9), 1 - 1e-9)
                g = beta * (k - n * p)                    # d loglik / dθ  (and −that for d)
                gt[mi] += g; gd[li] -= g
        theta += lr * gt; diff += lr * gd
        diff += (args.anchor - diff.mean())               # anchor mean difficulty
        theta += (args.anchor - diff.mean())              # keep shift consistent

    order = np.argsort(-theta)
    print("=== reasoning-GELO (IRT / Rasch on MATH difficulty tiers) ===")
    print(f"[{len(models)} model(s), metric={args.metric}, anchor mean-difficulty={args.anchor:.0f}]")
    print("\nMATH level difficulties (GELO):")
    for lv in levels:
        print(f"  L{lv}: {diff[lidx[lv]]:+7.0f}")
    print("\nModel reasoning-GELO:")
    for mi in order:
        print(f"  {models[mi]:<32} {theta[mi]:+7.0f}")
    if K == 1:
        print("\n(note: one model → difficulties are relative to it; add a 2nd model for real IRT calibration)")

    json.dump({"metric": args.metric, "anchor": args.anchor,
               "level_difficulty": {int(lv): round(float(diff[lidx[lv]])) for lv in levels},
               "model_gelo": {models[mi]: round(float(theta[mi])) for mi in range(K)}},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
