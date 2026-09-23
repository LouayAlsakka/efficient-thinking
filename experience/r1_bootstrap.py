"""R1 (理): paired per-problem bootstrap with 95% CI on success AND actions, plus McNemar on
success, for every G-vs-base pair. Replaces the point deltas in the transfer table.

PAIRED, because every pair is the SAME problems run twice -- the arms differ only in whether the
head acts at the post-inspect decision. An unpaired interval would ignore that and be wider than the
data warrants; McNemar uses only the discordant pairs, which is where the whole effect lives (the
agreement group moves -1 of 121).
"""
import json, sys, os, math, collections
import numpy as np

SP = sys.argv[1]
def ep(*paths):
    d = {}
    for p in paths:
        for l in open(p):
            e = json.loads(l); d[e["task_id"]] = e
    return d

P = lambda *a: [os.path.join(SP, x) for x in a]
PAIRS = [
 ("seed21 head -> seed21 slice4 (same run)",
  ep(*P("v3b/v3b_s4.episodes.jsonl")), ep(*P("v3g/v3g_s4.episodes.jsonl"))),
 ("seed21 ORIG head -> seed47 75",
  ep(*P("v3rep_run/repbase.episodes.jsonl")), ep(*P("g1run/../v3rep_run/repg.episodes.jsonl"))),
 ("seed47 ORIG head -> seed21 300",
  ep(*P(*["v3b/v3b_s%d.episodes.jsonl" % s for s in (1,2,3,4)])),
  ep(*P(*["v3g300/v3g300_s%d.episodes.jsonl" % s for s in (1,2,3,4)]))),
 ("seed47 LBFGS head -> seed21 300",
  ep(*P(*["v3b/v3b_s%d.episodes.jsonl" % s for s in (1,2,3,4)])),
  ep(*P(*["v3g300L/v3g300L_s%d.episodes.jsonl" % s for s in (1,2,3,4)]))),
 ("seed21 LBFGS head -> seed47 75",
  ep(*P("v3rep_run/repbase.episodes.jsonl")), ep(*P("g1run/g1.episodes.jsonl"))),
 ("P9(b): G1(weighted) vs G0 on seed47 75",
  ep(*P("g1run/g1.episodes.jsonl")), ep(*P("p9b_run/g1w.episodes.jsonl"))),
]

rng = np.random.default_rng(11)
print("R1 — paired per-problem bootstrap (10,000 resamples) and McNemar. 95%% CI on the DIFFERENCE.")
print()
for name, A, B in PAIRS:
    ks = sorted(set(A) & set(B))
    n = len(ks)
    a_s = np.array([float(A[k]["green"]) for k in ks]); b_s = np.array([float(B[k]["green"]) for k in ks])
    a_a = np.array([float(A[k]["actions"]) for k in ks]); b_a = np.array([float(B[k]["actions"]) for k in ks])
    ds, da = b_s - a_s, b_a - a_a
    idx = rng.integers(0, n, size=(10000, n))
    bs = ds[idx].mean(1); ba = da[idx].mean(1)
    lo_s, hi_s = np.percentile(bs, [2.5, 97.5]); lo_a, hi_a = np.percentile(ba, [2.5, 97.5])
    b01 = int(((a_s == 0) & (b_s == 1)).sum()); b10 = int(((a_s == 1) & (b_s == 0)).sum())
    # exact binomial two-sided on the discordant pairs
    m = b01 + b10
    if m:
        k = min(b01, b10)
        p = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m))
    else:
        p = 1.0
    print("  %s   n=%d" % (name, n))
    print("     success  %+.1f points  95%% CI [%+.1f, %+.1f]" % (100*ds.mean(), 100*lo_s, 100*hi_s))
    print("     actions  %+.2f          95%% CI [%+.2f, %+.2f]" % (da.mean(), lo_a, hi_a))
    print("     McNemar  base-only-solved %d · G-only-solved %d · exact two-sided p = %.4g" % (b10, b01, p))
    print()
