#!/usr/bin/env python3
"""P0 (理 11002): count the DISTINCT PROBLEMS in a task set before counting anything else.

    python3 et8_task_gate.py --tasks experience/tasks/v2

A task set's size is not its file count. Two tasks that present the AGENT with the same program,
the same tests, the same symptom and the same region list are ONE problem: under greedy decoding
they produce byte-identical trajectories and identical outcomes, so they contribute one observation
between them, not two.

    v1   2,000 files ->  12 distinct signatures   largest group 240
    v2   2,000 files ->  11 distinct signatures   largest group 192
    v2 baseline, 80 episodes -> 11 signatures, ZERO with more than one outcome

Every rate published from this environment this week -- -36.2, -41.2, 62.5%, 20.0%, 64%/0% -- is
eleven or twelve deterministic outcomes weighted by how often each signature appeared. The two 7B
seeds landing at 0.599 and 0.593 were the tell: twelve outcomes, near-identical weights.

THE GATE, and it runs FIRST, ahead of the prompt gate:
    distinct signatures >= 0.9 x tasks

The prompt gate (et8_prompt_gate.py) asks whether the STATE can distinguish episodes. This asks
whether the TASK SET can. A set that fails here fails there by construction, and it is checkable
before a single episode -- or before a single line of generator, which is where it belongs.

WHAT A SIGNATURE DELIBERATELY EXCLUDES: task_id and seed. Those are the fields that made 2,000
files look like 2,000 problems.
"""
from __future__ import annotations
import argparse, collections, glob, hashlib, json, os, sys


def signature(d: dict) -> str:
    key = "\x00".join([d.get("program", ""), d.get("tests", ""), d.get("symptom", ""),
                       ",".join(d.get("regions", [])), d.get("bug_region", ""),
                       d.get("bug_class", "")])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.tasks, "task_*.json")))
    sigs = collections.Counter()
    fields = collections.Counter()
    for f in files:
        d = json.load(open(f))
        sigs[signature(d)] += 1
        fields[(d.get("bug_region", ""), d.get("bug_class", ""))] += 1
    n = len(files)
    k = len(sigs)
    ratio = (k / n) if n else 0.0
    out = {"tasks": a.tasks, "task_files": n, "distinct_signatures": k,
           "ratio": round(ratio, 4), "threshold": a.threshold,
           "largest_identical_group": sigs.most_common(1)[0][1] if sigs else 0,
           "distinct_(bug_region, bug_class) pairs": len(fields),
           "VERDICT": "PASS" if ratio >= a.threshold else
                      "FAIL — the set has %d problems, not %d tasks" % (k, n),
           "signature_excludes": "task_id and seed — the two fields that made 2,000 files look "
                                 "like 2,000 problems",
           "note": "run this BEFORE the prompt gate and before any generator work. A set that "
                   "fails here fails the prompt gate by construction."}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if ratio >= a.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
