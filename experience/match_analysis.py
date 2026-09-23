#!/usr/bin/env python
"""ET-8b §12 — read the replicate match. Written BEFORE the match finishes, so the analysis is not
chosen after seeing the numbers.

Louay's framing (理): "like a chess game... you get an average of multiple games... run
multiple times, track statistical bias, make sure we are ahead of noise, not as clear winner but
acceptable if we do the right discipline."

THE TEST IS NOT McNEMAR. 理 ruled it a within-run descriptive only on loop rows: four of six
NULL draws reached p < 0.06, so it measures the pairing and not the effect. **The reported
uncertainty is the spread across replicates.**

Three readings, all printed, in increasing honesty:
  1. MEANS  — mean of three runs per generation, Welch interval on the difference. The headline.
  2. THE NINE SINGLE PAIRINGS — every g1 run against every g0 run, 3x3. This is the number a
     one-shot experiment would have produced, and its RANGE is what the old design was sampling
     from blind. If that range spans zero, no single paired run could have settled the question.
  3. 理'S REGISTERED RULE — a between-generation row is a finding only if its point lies outside
     the null draws' range AND its lower bound exceeds the largest null point.
"""
import argparse, itertools, json, os, statistics, sys


def green(path):
    rows = [json.loads(l) for l in open(path)]
    ids = [r["task_id"] for r in rows]
    assert len(set(ids)) == len(ids), "%s has duplicate task_ids" % path
    return rows, 100.0 * sum(1 for r in rows if r["green"]) / len(rows)


def paired_delta(a_rows, b_rows):
    """b - a in points, on the tasks they share. The point estimate only: the interval that used to
    come with it was a McNemar-family number and is no longer the reported uncertainty."""
    A = {r["task_id"]: r["green"] for r in a_rows}
    B = {r["task_id"]: r["green"] for r in b_rows}
    common = [k for k in A if k in B]
    return 100.0 * (sum(B[k] for k in common) - sum(A[k] for k in common)) / len(common), len(common)


def welch(x, y):
    """Difference of means with a Welch t-interval. Small n by design: three runs per arm."""
    mx, my = statistics.mean(x), statistics.mean(y)
    vx, vy = statistics.variance(x), statistics.variance(y)
    nx, ny = len(x), len(y)
    se = (vx / nx + vy / ny) ** 0.5
    if se == 0:
        return my - mx, 0.0, (my - mx, my - mx), float("inf")
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}.get(int(round(df)), 2.776)
    d = my - mx
    return d, se, (d - t * se, d + t * se), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="where match_<arm>.episodes.jsonl live")
    ap.add_argument("--nulls", nargs="*", type=float, default=[-5.0, -3.7, 1.3, 1.3, 5.0, 6.3],
                    help="the §12 null draw POINTS on this instrument (default: the six measured)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    arms = {"b": ["b1", "b2", "b3"], "g0": ["g0a", "g0b", "g0c"], "g1": ["g1a", "g1b", "g1c"]}
    rows, pct = {}, {}
    for gen, names in arms.items():
        for n in names:
            p = os.path.join(a.dir, "match_%s.episodes.jsonl" % n)
            if not os.path.exists(p):
                sys.exit("missing %s — the match is not finished; this script reads all nine or none" % p)
            rows[n], pct[n] = green(p)

    print("  ARMS (300 tasks each, interleaved b1 g0a g1a b2 g0b g1b b3 g0c g1c)")
    for gen in ("b", "g0", "g1"):
        v = [pct[n] for n in arms[gen]]
        print("    %-3s %s   mean %.2f  sd %.2f" % (gen, "  ".join("%5.1f%%" % x for x in v),
                                                    statistics.mean(v), statistics.stdev(v)))

    out = {"document": "ET-8b §12 — the replicate match", "arms": pct,
           "per_generation": {g: {"runs": [pct[n] for n in arms[g]],
                                  "mean": round(statistics.mean([pct[n] for n in arms[g]]), 3),
                                  "sd": round(statistics.stdev([pct[n] for n in arms[g]]), 3)}
                              for g in arms},
           "null_draw_points": a.nulls, "comparisons": {}}

    print("\n  1. MEANS OF THREE — the headline, Welch interval on the difference")
    for lo, hi in (("b", "g0"), ("g0", "g1"), ("b", "g1")):
        x, y = [pct[n] for n in arms[lo]], [pct[n] for n in arms[hi]]
        d, se, ci, df = welch(x, y)
        print("    %-8s %+6.2f  95%% CI [%+.2f, %+.2f]  (se %.2f, df %.1f)" % ("%s->%s" % (lo, hi), d, ci[0], ci[1], se, df))
        out["comparisons"]["%s_to_%s" % (lo, hi)] = {"delta_of_means": round(d, 3),
                                                     "CI95_welch": [round(ci[0], 3), round(ci[1], 3)],
                                                     "se": round(se, 3), "df": round(df, 2)}

    print("\n  2. THE NINE SINGLE PAIRINGS — what a one-shot experiment would have produced")
    for lo, hi in (("g0", "g1"),):
        pts = []
        for A, B in itertools.product(arms[lo], arms[hi]):
            d, n = paired_delta(rows[A], rows[B])
            pts.append(d)
            print("      %-4s vs %-4s  %+6.2f  (n=%d)" % (B, A, d, n))
        lo_, hi_ = min(pts), max(pts)
        spans = lo_ <= 0 <= hi_
        print("      range %+.2f to %+.2f   spans zero: %s" % (lo_, hi_, spans))
        out["nine_single_pairings_%s_to_%s" % (lo, hi)] = {
            "points": [round(p, 3) for p in pts], "min": round(lo_, 3), "max": round(hi_, 3),
            "spans_zero": spans,
            "reading": ("every one of these is a legitimate paired 300-task experiment. Their RANGE "
                        "is what a single run was sampling from blind; if it spans zero, no one-shot "
                        "paired comparison could have settled this question either way.")}

    # THE MATCH'S OWN BASE ARMS ARE THREE MORE NULL DRAWS, and they were run inside this session on
    # this instrument — so they belong in the null set the rule is read against. Added BEFORE the
    # match finished, so the null set is not widened or narrowed after seeing where the effect fell.
    own = []
    for A, B in itertools.combinations(arms["b"], 2):
        d, n = paired_delta(rows[A], rows[B])
        own.append(d)
    print("\n  2b. THE MATCH'S OWN BASE-vs-BASE DRAWS (three more nulls, same session)")
    for (A, B), d in zip(itertools.combinations(arms["b"], 2), own):
        print("      %-4s vs %-4s  %+6.2f" % (B, A, d))
    allnulls = list(a.nulls) + own
    print("      null set now %d draws, range %+.2f to %+.2f  (was %+.2f to %+.2f)"
          % (len(allnulls), min(allnulls), max(allnulls), min(a.nulls), max(a.nulls)))
    out["match_own_null_draws"] = [round(d, 3) for d in own]
    out["null_set_used"] = [round(d, 3) for d in allnulls]

    print("\n  3. 理'S REGISTERED RULE (§12) — null draws %s, largest %+.1f" % (
        [round(x, 1) for x in allnulls], max(allnulls)))
    d = out["comparisons"]["g0_to_g1"]
    inside = min(allnulls) <= d["delta_of_means"] <= max(allnulls)
    passes = (not inside) and d["CI95_welch"][0] > max(allnulls)
    verdict = ("ACCUMULATION IS A FINDING — the mean difference lies outside the null range and its "
               "lower bound clears the largest null draw"
               if passes else
               "NOT A FINDING under the registered rule — the mean difference lies inside the null "
               "range and/or its lower bound does not clear the largest null draw")
    print("    g0 -> g1 mean %+.2f, lower bound %+.2f  ->  %s" % (d["delta_of_means"], d["CI95_welch"][0], verdict))
    out["registered_verdict"] = verdict
    out["signed"] = "Sautee (sha-ta)"
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n  wrote %s" % a.out)


if __name__ == "__main__":
    main()
