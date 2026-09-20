#!/usr/bin/env python3
"""ET-8b gen0 states — candidate states at ALL THREE decision points of the base loop.

The 8a extractor reconstructs ONE post-inspect decision per episode. The loop has up to three, and
the state at decision k depends on the choices at decisions < k — which is the whole point. This
walks each episode's steps and rebuilds the prompt the agent actually saw at each decision.

Writes et8_head_v3's format (X_layer*.npy + meta.jsonl) so the FIT and its three controls run
unchanged, plus a `decision` field on every row so a per-decision head can be fitted from the same
extraction. The spec (docs/et8b-loop-gates.md §3) requires choosing ONE SHARED head vs THREE
per-decision heads by held-out probe accuracy BEFORE any loop run — this produces the data for both
so the choice is made on numbers and then fixed.
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et8_agent as A
E = A.E


def decisions_from(steps, tasks_dir):
    by = collections.defaultdict(list)
    for s in steps:
        by[(s["run_id"], s["task_id"])].append(s)
    out = []
    for (rid, tid), st in by.items():
        st.sort(key=lambda s: s.get("step", 0))
        task = json.load(open(os.path.join(tasks_dir, tid + ".json")))
        regions = [r for r in task["regions"] if r != "run"]
        program = task["program"]
        inspected, history = {}, []
        for s in st:
            act = s.get("action")
            if act == "inspect" and s.get("region"):
                inspected[s["region"]] = A.region_source(program, s["region"]) or ""
                history.append("inspect %s" % s["region"])
            elif act == "hypothesize":
                lines = ["SYMPTOM: %s" % task["symptom"]]
                for r, v in inspected.items():
                    lines.append("# region: %s\n%s" % (r, v))
                if history:
                    lines.append("So far: " + " | ".join(history[-8:]))
                out.append({"task_id": tid, "decision": s.get("decision"),
                            "bug_region": task["bug_region"], "regions": regions,
                            "agent_region": s.get("region"), "agent_hit": bool(s.get("region_hit")),
                            "messages": [{"role": "system",
                                          "content": A.SYSTEM % (", ".join(E.BUG_CLASSES), 12)},
                                         {"role": "user", "content": "\n".join(lines)}]})
                history.append("hypothesize %s" % (s.get("region") or "?"))
            elif act in ("patch", "noop_patch"):
                history.append("patch %s -> %s" % (s.get("region"),
                                                   "GREEN" if s.get("patch_ok") else "still red"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", nargs="+", type=int, default=[18])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import et8_head_v3 as H
    from mlx_lm import load
    import mlx.core as mx
    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = decisions_from(steps, a.tasks)
    per = collections.Counter(d["decision"] for d in dec)
    print("  decisions: %d total  per point: %s" % (len(dec), dict(sorted(per.items()))), file=sys.stderr)
    model, tok = load(a.model)
    os.makedirs(a.out, exist_ok=True)
    X = {l: [] for l in a.layers}; meta = []; t0 = time.time()
    for i, d in enumerate(dec):
        for r in d["regions"]:
            pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
            hs = H.hidden_at(model, tok, d["messages"], a.layers, prefix=pre)
            for l in a.layers:
                X[l].append(np.array(hs[l].astype(mx.float32), copy=False))
            meta.append({"task_id": d["task_id"], "decision": d["decision"], "candidate": r,
                         "label": int(r == d["bug_region"]), "agent_region": d["agent_region"],
                         "agent_hit": d["agent_hit"], "n_regions": len(d["regions"])})
        if (i + 1) % 100 == 0:
            rate = (time.time() - t0) / (i + 1)
            print("  %d/%d  %.2fs/dec  eta %.0fm" % (i + 1, len(dec), rate,
                  rate * (len(dec) - i - 1) / 60), file=sys.stderr)
    for l in a.layers:
        np.save(os.path.join(a.out, "X_layer%d.npy" % l), np.stack(X[l]))
    with open(os.path.join(a.out, "meta.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print("  wrote %d candidate states" % len(meta), file=sys.stderr)


if __name__ == "__main__":
    main()
