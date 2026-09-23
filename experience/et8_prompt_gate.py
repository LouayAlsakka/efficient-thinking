#!/usr/bin/env python3
"""THE STANDING GATE (理: count distinct decision prompts before fitting any head.

    python3 et8_prompt_gate.py --runs <run> --tasks experience/tasks/v2 --out <gate.json>

A controller head reads a STATE. The state is a deterministic function of the PROMPT. So two
episodes whose decision prompts are byte-identical produce identical activations at every layer and
identical head output, no matter how the head is fit. If the decision prompts collapse to a handful
of distinct strings, the head IS a lookup table on those strings -- exactly, not approximately --
and fitting it measures nothing a `dict` does not already do for free.

THE RULE: distinct decision prompts / episodes. Below 0.9, the state is a table and the head is
NOT fit.

IT WOULD HAVE REFUSED BOTH TASK SETS THIS WEEK, in seconds each:

    v1            10 distinct prompts / 2,000 episodes   = 0.005   and the ten mapped 1-to-1 onto
                                                                   the bug region, so the table was
                                                                   PERFECT and G was untestable
    v2, step 1     6 distinct prompts /    80 episodes   = 0.075   table imperfect (51.2% held out)
                                                                   but still the head's whole ceiling

Both were found by hand, after building the machinery, and one of them after a 100%-held-out probe
result that passed its permutation control. This gate is the cheap version of that discovery, and
it belongs beside the permutation control rather than after it: a permutation control asks whether
ANY label would fit; this asks whether the input can distinguish the episodes at all.

WHAT IT DOES NOT CATCH. Prompts that differ in irrelevant ways -- a timestamp, a step counter, a
shuffled list -- count as distinct while carrying no signal, so a HIGH ratio is necessary and not
sufficient. The permutation control remains the check on the other side.
"""
from __future__ import annotations
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run prefixes; <run>.steps.jsonl is read")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--after-inspect", action="store_true",
                    help="score the first hypothesis AFTER at least one inspect, which is where G "
                         "acts under the inspect-first loop")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import et8_inject as I

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = I.all_decision_states(steps, a.tasks) if a.after_inspect else \
        I.first_decision_states(steps, a.tasks)
    if a.after_inspect:
        seen, keep = set(), []
        for d in dec:
            k = (d.get("run_id"), d["task_id"])
            # the first hypothesis whose prompt already carries inspected source
            if k in seen:
                continue
            if "--- region " not in d["messages"][1]["content"]:
                continue
            seen.add(k); keep.append(d)
        dec = keep

    prompts = collections.Counter(json.dumps(d["messages"], ensure_ascii=False) for d in dec)
    n, k = len(dec), len(prompts)
    ratio = (k / n) if n else 0.0
    biggest = prompts.most_common(1)[0][1] if prompts else 0
    out = {"episodes_at_the_decision_point": n, "distinct_prompts": k,
           "ratio": round(ratio, 4), "threshold": a.threshold,
           "largest_identical_group": biggest,
           "VERDICT": "FIT" if ratio >= a.threshold else "DO NOT FIT — the state is a table",
           "after_inspect": bool(a.after_inspect),
           "note": "a high ratio is NECESSARY, not sufficient: prompts can differ in ways that "
                   "carry no signal. Keep the permutation control on the other side."}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if ratio >= a.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
