#!/usr/bin/env python
"""WO-318 — the N-replicate runner for the area classifier. Pre-registered: docs/area-classifier-prereg.md

Deliberately SHAPE-INDEPENDENT. The one thing not yet decided is what an `area` IS — a typed object
or a statement in the search DSL (理 12377, my 12378) — and the disagreement comparison depends on
that answer. So the comparator is INJECTED, not assumed, and this file never parses an area.

形 12392's discipline, applied here: build the part whose shape is known, wait for the rest. What is
known is the protocol — N independent calls per utterance, the reported number being the
disagreement ACROSS replicates, latency as median and p90, and a refusal to report a rate computed
over a set with any unparsed reply.
"""
import argparse, json, statistics, sys, time


def run_set(utterances, classify, n=5, comparator=None, label=""):
    """classify(utterance) -> (area, raw). comparator(a, b) -> True if the SAME area.

    No default comparator. A default here would be a guess about the area's shape, and a guess that
    silently works is worse than one that fails: it would make a formatting difference look like a
    classifier disagreement, or hide a real one, in the one number the word 'predictable' rests on.
    """
    if comparator is None:
        sys.exit("refusing to run: no comparator supplied. The area's representation is not settled "
                 "(理 12377) and a default comparison would measure formatting, not meaning.")
    rows, lat, unparsed = [], [], 0
    for u in utterances:
        reps, times = [], []
        for _ in range(n):
            t0 = time.time()
            try:
                area, raw = classify(u)
            except Exception as e:
                area, raw = None, "%s: %s" % (type(e).__name__, str(e)[:120])
            times.append(time.time() - t0)
            reps.append({"area": area, "raw": raw})
        if any(r["area"] is None for r in reps):
            unparsed += 1
        unanimous = all(comparator(reps[0]["area"], r["area"]) for r in reps[1:])
        rows.append({"utterance": u, "replicates": reps, "unanimous": unanimous,
                     "seconds": [round(t, 4) for t in times]})
        lat += times
    scored = [r for r in rows if all(x["area"] is not None for x in r["replicates"])]
    rate = None if unparsed else round(1 - sum(r["unanimous"] for r in scored) / len(scored), 4)
    lat_sorted = sorted(lat)
    return {
        "label": label, "n_replicates": n, "n_utterances": len(utterances),
        "disagreement_rate": rate,
        "unparsed_utterances": unparsed,
        "disagreement_note": (None if rate is not None else
                              "NOT REPORTED: %d utterance(s) had a replicate that did not parse. "
                              "A rate over a set with an unparsed reply is a rate about the parser."
                              % unparsed),
        "latency_median_s": round(statistics.median(lat_sorted), 4),
        "latency_p90_s": round(lat_sorted[int(0.9 * (len(lat_sorted) - 1))], 4),
        "latency_note": ("median AND p90, never a mean: a classifier fast four times in five and "
                         "slow on the fifth is one the user experiences as slow"),
        "per_utterance": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest:
        sys.exit("this module is a library until the area's shape is settled; --selftest exercises it")

    # A stand-in classifier and comparator, so the PROTOCOL is tested before the shape exists.
    seq = {"stable": ["A", "A", "A", "A", "A"], "wobbly": ["A", "B", "A", "A", "A"],
           "broken": ["A", None, "A", "A", "A"]}
    state = {k: 0 for k in seq}

    def classify(u):
        i = state[u]; state[u] += 1
        v = seq[u][i]
        if v is None:
            raise ValueError("unparseable reply")
        return v, "raw:%s" % v

    same = lambda x, y: x == y
    r = run_set(["stable", "wobbly"], classify, n=5, comparator=same, label="protocol selftest")
    assert r["disagreement_rate"] == 0.5, r["disagreement_rate"]
    assert r["unparsed_utterances"] == 0
    print("  [1] two utterances, one wobbles -> disagreement %.2f" % r["disagreement_rate"])
    state.update({k: 0 for k in seq})
    r2 = run_set(["stable", "broken"], classify, n=5, comparator=same, label="unparsed")
    assert r2["disagreement_rate"] is None and r2["unparsed_utterances"] == 1
    print("  [2] one reply fails to parse -> rate REFUSED, not silently computed over the rest")
    print("  [3] latency reported as median %.4fs and p90 %.4fs, never a mean"
          % (r2["latency_median_s"], r2["latency_p90_s"]))
    try:
        run_set(["stable"], classify, n=2, comparator=None)
        print("  [4] FAILED — ran without a comparator"); sys.exit(1)
    except SystemExit as e:
        print("  [4] refuses to run with no comparator: %s" % str(e)[:60])
    print("\n  SELFTEST PASSED — 4 checks, no network, no model.")


if __name__ == "__main__":
    main()
