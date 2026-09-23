#!/usr/bin/env python
"""Subsample a state set to a row budget WITHOUT BREAKING DECISIONS APART.

WHY THIS REPLACES ROW SUBSAMPLING, AND WHAT THAT COST. §13b and §13c hold "amount" fixed by cutting
each set to 1,437 rows. I did that by drawing rows at random, which splits a decision's candidates
across the kept/dropped boundary. The head is fitted per candidate but SCORED PER DECISION -- a
pick is right when the highest-scoring candidate of that decision is the correct one -- so a
decision whose correct candidate was dropped can never be picked right, and a decision left with
only negatives teaches nothing.

The damage, measured rather than argued, on the three sets already fitted:

    set                  rows   decisions  rows/dec   decisions retaining a POSITIVE   probe CV
    gen1' (SQL)          1437     300        4.79            100.0%                     92.7%
    gen1-matched         1437     542        2.65             53.7%                     35.2%
    gen1''               1437     693        2.07             41.1%                     26.7%

The probe accuracies track the last column almost exactly, and gen1' was never subsampled at all --
it is a whole collection that happened to be 1,437 rows. So the headline pair was comparing an
intact collection against two shredded ones, and the difference I would have read as ORIGIN is
mostly the matching procedure.

WHAT THIS DOES INSTEAD. Whole decision groups are kept, shuffled at a fixed seed, until the next
group would exceed the budget. Every kept decision has all of its candidates, so the positive-label
structure is exactly what the collection produced. The realised row count lands at or just under
the budget and is RECORDED rather than forced -- forcing it would mean splitting one group, which
is the thing this file exists to stop.

MATCHING ON DECISIONS IS ALSO AVAILABLE (--decisions N) and is the better primitive when the
question is "the same amount of experience": a decision is one thing the agent did, a row is one
candidate it was offered.
"""
import argparse, collections, json, os, sys
import numpy as np


def load(d, layer):
    meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    X = np.load(os.path.join(d, "X_layer%d.npy" % layer))
    if len(meta) != X.shape[0]:
        sys.exit("STOP: meta %d rows but X %d" % (len(meta), X.shape[0]))
    return meta, X


def groups(meta):
    g = collections.OrderedDict()
    for i, m in enumerate(meta):
        g.setdefault((m["task_id"], m.get("decision", 0)), []).append(i)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--rows", type=int, default=0, help="row budget; whole decisions only")
    ap.add_argument("--decisions", type=int, default=0, help="decision budget (preferred)")
    ap.add_argument("--seed", type=int, default=13731)
    a = ap.parse_args()
    if bool(a.rows) == bool(a.decisions):
        sys.exit("STOP: give exactly one of --rows or --decisions")

    meta, X = load(a.states, a.layer)
    g = groups(meta)
    keys = list(g)
    order = np.random.default_rng(a.seed).permutation(len(keys))
    kept_keys, kept = [], []
    for oi in order:
        k = keys[oi]
        if a.decisions and len(kept_keys) >= a.decisions:
            break
        if a.rows and len(kept) + len(g[k]) > a.rows:
            continue                      # skip, do not split: a split group is the whole problem
        kept_keys.append(k); kept.extend(g[k])
    kept.sort()

    lab = [meta[i]["label"] for i in kept]
    withpos = sum(1 for k in kept_keys if max(meta[i]["label"] for i in g[k]) == 1)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "X_layer%d.npy" % a.layer), X[kept])
    with open(os.path.join(a.out, "meta.jsonl"), "w") as f:
        for i in kept:
            f.write(json.dumps(meta[i], ensure_ascii=False) + "\n")
    rec = {"source": os.path.abspath(a.states), "unit": "decision groups, never split",
           "budget": {"rows": a.rows or None, "decisions": a.decisions or None},
           "seed": a.seed, "rng": "numpy default_rng",
           "n_source_rows": len(meta), "n_source_decisions": len(g),
           "n_kept_rows": len(kept), "n_kept_decisions": len(kept_keys),
           "rows_per_decision": round(len(kept) / max(1, len(kept_keys)), 3),
           "positive_row_pct": round(100.0 * sum(lab) / max(1, len(lab)), 2),
           "decisions_retaining_a_positive_pct": round(100.0 * withpos / max(1, len(kept_keys)), 2),
           "kept_row_indices": kept,
           "why": ("row subsampling splits a decision's candidates and the head is SCORED per "
                   "decision, so a dropped correct candidate makes that decision unpickable and a "
                   "group left with only negatives teaches nothing. Whole groups only.")}
    json.dump(rec, open(os.path.join(a.out, "subsample.json"), "w"), indent=1, ensure_ascii=False)
    print("  %s -> %d rows over %d decisions (%.2f rows/dec), %.1f%% of decisions keep a positive"
          % (os.path.basename(a.states), len(kept), len(kept_keys),
             rec["rows_per_decision"], rec["decisions_retaining_a_positive_pct"]))


if __name__ == "__main__":
    main()
