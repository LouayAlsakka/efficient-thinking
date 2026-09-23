#!/usr/bin/env python3
"""Paired per-problem bootstrap + exact McNemar for any base-vs-G pair (理's method, R1).

GENERIC ON PURPOSE. r1_bootstrap.py hard-codes the 7B's pairs and produced numbers already in the
package; editing it to take new arms would silently re-define the thing already reported. This is
the same arithmetic behind a command line.

PAIRED, because every pair is the SAME problems run twice and the arms differ only in whether the
head acts at the post-inspect decision. An unpaired interval ignores that and is wider than the data
warrants. McNemar uses only the DISCORDANT pairs, which is where the whole effect lives.

REFUSES TO REPORT ON AN UNEQUAL PAIR. If either arm is missing tasks the other has, the intersection
is silently smaller than both and the rate is over a population neither arm measured. It prints the
mismatch and exits non-zero instead.
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np


def load(paths):
    d = {}
    for p in paths:
        for l in open(p):
            e = json.loads(l)
            d[e["task_id"]] = e
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--g", nargs="+", required=True)
    ap.add_argument("--name", default="pair")
    ap.add_argument("--expect", type=int, default=0, help="required n; 0 = no requirement")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    a = ap.parse_args()

    A, B = load(a.base), load(a.g)
    ks = sorted(set(A) & set(B))
    n = len(ks)
    if len(A) != len(B) or n != len(A):
        print("MISMATCHED ARMS: base=%d G=%d intersection=%d. Not reporting." % (len(A), len(B), n))
        return 1
    if a.expect and n != a.expect:
        print("EXPECTED n=%d, GOT %d. Not reporting a rate over a partial set." % (a.expect, n))
        return 1

    a_s = np.array([float(A[k]["green"]) for k in ks])
    b_s = np.array([float(B[k]["green"]) for k in ks])
    a_a = np.array([float(A[k]["actions"]) for k in ks])
    b_a = np.array([float(B[k]["actions"]) for k in ks])
    ds, da = b_s - a_s, b_a - a_a

    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, n, size=(a.resamples, n))
    lo_s, hi_s = np.percentile(ds[idx].mean(1), [2.5, 97.5])
    lo_a, hi_a = np.percentile(da[idx].mean(1), [2.5, 97.5])

    b01 = int(((a_s == 0) & (b_s == 1)).sum())   # G solved, base did not
    b10 = int(((a_s == 1) & (b_s == 0)).sum())   # base solved, G did not
    m = b01 + b10
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(b01, b10) + 1)) / (2 ** m)) if m else 1.0

    res = {"pair": a.name, "n": n,
           "base_success_pct": round(100 * a_s.mean(), 1),
           "G_success_pct": round(100 * b_s.mean(), 1),
           "success_delta_points": round(100 * ds.mean(), 1),
           "success_95CI": [round(100 * lo_s, 1), round(100 * hi_s, 1)],
           "base_actions": round(a_a.mean(), 2), "G_actions": round(b_a.mean(), 2),
           "actions_delta": round(da.mean(), 2),
           "actions_95CI": [round(lo_a, 2), round(hi_a, 2)],
           "McNemar": {"base_only_solved": b10, "G_only_solved": b01,
                       "discordant": m, "exact_two_sided_p": float("%.4g" % p)},
           "method": "paired per-problem bootstrap, %d resamples; exact two-sided binomial on the "
                     "discordant pairs" % a.resamples}
    print("  %s   n=%d" % (a.name, n))
    print("     success  %5.1f%% -> %5.1f%%   %+.1f points  95%% CI [%+.1f, %+.1f]"
          % (res["base_success_pct"], res["G_success_pct"], res["success_delta_points"], lo_s * 100, hi_s * 100))
    print("     actions  %5.2f  -> %5.2f    %+.2f         95%% CI [%+.2f, %+.2f]"
          % (res["base_actions"], res["G_actions"], res["actions_delta"], lo_a, hi_a))
    print("     McNemar  base-only %d · G-only %d · exact two-sided p = %.4g" % (b10, b01, p))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
