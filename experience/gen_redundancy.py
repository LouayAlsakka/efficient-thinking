#!/usr/bin/env python
"""ET-8b §13a — is the next generation's experience a RE-SAMPLE of the last one's corner?

Louay's hypothesis (理 12342): gen1's rows come from the gen0-steered agent walking the same
problems, so they are correlated with gen0's, and accumulation is limited by the INDEPENDENCE of the
experience rather than by its amount. §11 found the decomposition unresolved at n=300; this asks a
different question of the data already on disk, and costs no GPU.

TWO NUMBERS, both defined here rather than left to the reader:

1. PROBLEM OVERLAP — the fraction of the later generation's training rows whose `task_id` also
   appears in the earlier generation's rows. Plain, exact, no feature space involved.

2. NEIGHBOUR REDUNDANCY — the fraction of the later generation's rows whose nearest earlier-
   generation row is closer than the earlier generation's own MEDIAN nearest-neighbour distance.
   In words: rows that sit closer to the old corner than the old corner sits to itself. Distances
   are cosine on the standardised layer-18 states, the same space the heads are fitted in.

⚠️ 理's phrasing for (1) was "rows from problems gen0's agent ALREADY SOLVED", which admits two
readings — solved-by-the-previous-agent, or merely seen by it. Solved-by is circular here: gen1's
rows ARE the gen0-steered agent's trajectory, so its solve set is the thing being measured. I have
implemented SEEN-BY (task_id overlap), which is the non-circular half, and I am naming the
substitution rather than quietly picking one.
"""
import argparse, json, os, sys
import numpy as np


def load(d):
    X = np.load(os.path.join(d, "X_layer18.npy")).astype(np.float64)
    meta = [json.loads(l) for l in open(os.path.join(d, "meta.jsonl"))]
    assert len(meta) == len(X), (len(meta), len(X))
    return X, meta


def unit(X, mu, sd):
    Z = (X - mu) / sd
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def nn_dist(A, B, block=512):
    """For each row of A, the cosine distance to its nearest row of B (A and B unit-normalised)."""
    out = np.empty(len(A))
    for i in range(0, len(A), block):
        sims = A[i:i + block] @ B.T
        out[i:i + block] = 1.0 - sims.max(axis=1)
    return out


def self_nn_median(A, block=512):
    """Median nearest-neighbour distance WITHIN A, excluding each row's match to itself."""
    out = np.empty(len(A))
    for i in range(0, len(A), block):
        sims = A[i:i + block] @ A.T
        for r in range(sims.shape[0]):
            sims[r, i + r] = -np.inf
        out[i:i + block] = 1.0 - sims.max(axis=1)
    return float(np.median(out)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--earlier", nargs="+", required=True, help="state dir(s) of the EARLIER generation")
    ap.add_argument("--later", required=True, help="state dir of the LATER generation")
    ap.add_argument("--label", required=True, help="e.g. 'gen1 vs gen0'")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    Xe_parts, Me = [], []
    for d in a.earlier:
        X, m = load(d)
        Xe_parts.append(X); Me += m
    Xe = np.vstack(Xe_parts)
    Xl, Ml = load(a.later)

    # standardise on the EARLIER generation — the corner whose shape is the question
    mu, sd = Xe.mean(0), Xe.std(0) + 1e-9
    E, L = unit(Xe, mu, sd), unit(Xl, mu, sd)

    e_tasks = {r["task_id"] for r in Me}
    seen = sum(1 for r in Ml if r["task_id"] in e_tasks)
    overlap = seen / len(Ml)

    med, _ = self_nn_median(E)
    d_l = nn_dist(L, E)
    redundant = float((d_l < med).mean())

    # THE SHARPEST NUMBER, and it needs no feature space at all: how many of the later generation's
    # rows are BYTE-IDENTICAL to an earlier row. Identical hidden state means an identical history
    # prefix, which means the steered agent was in exactly the situation the earlier agent was in.
    E32 = np.load(os.path.join(a.earlier[0], "X_layer18.npy"))
    for d in a.earlier[1:]:
        E32 = np.vstack([E32, np.load(os.path.join(d, "X_layer18.npy"))])
    L32 = np.load(os.path.join(a.later, "X_layer18.npy"))
    eset = {r.tobytes() for r in E32}
    exact = sum(1 for r in L32 if r.tobytes() in eset)
    ekeys = {(r["task_id"], r["decision"], r["candidate"]) for r in Me}
    samekey = sum(1 for r in Ml if (r["task_id"], r["decision"], r["candidate"]) in ekeys)

    out = {
        "document": "ET-8b §13a — generation redundancy", "prereg": "gates §13a (理 12342)",
        "comparison": a.label,
        "earlier_dirs": a.earlier, "later_dir": a.later,
        "n_earlier_rows": int(len(Xe)), "n_later_rows": int(len(Xl)),
        "n_earlier_problems": len(e_tasks), "n_later_problems": len({r["task_id"] for r in Ml}),
        "problem_overlap_fraction": round(overlap, 4),
        "problem_overlap_definition": ("fraction of LATER rows whose task_id also appears in the "
                                       "EARLIER rows — SEEN-BY, not solved-by; see the module "
                                       "docstring for why solved-by is circular here"),
        "earlier_self_median_nn_distance": round(med, 5),
        "neighbour_redundancy_fraction": round(redundant, 4),
        "neighbour_redundancy_definition": ("fraction of LATER rows whose nearest EARLIER row is "
                                            "closer than the EARLIER set's own median nearest-"
                                            "neighbour distance — closer to the old corner than the "
                                            "old corner is to itself"),
        "exact_duplicate_rows": exact,
        "exact_duplicate_fraction": round(exact / len(L32), 4),
        "exact_duplicate_definition": ("later rows byte-identical to an earlier row. An identical "
                                       "hidden state means an identical history prefix: the steered "
                                       "agent was in exactly the situation the earlier agent was in."),
        "same_decision_key_fraction": round(samekey / len(Ml), 4),
        "same_decision_key_definition": "later rows sharing (task_id, decision, candidate) with an earlier row",
        "later_nn_distance_quartiles": [round(float(q), 5) for q in np.percentile(d_l, [25, 50, 75])],
        "signed": "Sautee (sha-ta)",
    }
    print("  %s" % a.label)
    print("    rows      earlier %5d   later %5d" % (len(Xe), len(Xl)))
    print("    problems  earlier %5d   later %5d" % (len(e_tasks), out["n_later_problems"]))
    print("    PROBLEM OVERLAP        %.1f%%  of later rows are on problems the earlier set saw" % (100 * overlap))
    print("    NEIGHBOUR REDUNDANCY   %.1f%%  of later rows sit closer to the old corner than it sits to itself"
          % (100 * redundant))
    print("    EXACT DUPLICATES       %.1f%%  of later rows are BYTE-IDENTICAL to an earlier row" % (100 * exact / len(L32)))
    print("    SAME DECISION KEY      %.1f%%  share (task, decision, candidate) with an earlier row" % (100 * samekey / len(Ml)))
    print("    (earlier self median NN distance %.4f; later NN quartiles %s)"
          % (med, out["later_nn_distance_quartiles"]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("    wrote %s" % a.out)


if __name__ == "__main__":
    main()
