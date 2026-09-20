#!/usr/bin/env python3
"""R7 states — the SQL decision prompts, in the format et8_head_v3's fit already consumes.

WHY A SEPARATE EXTRACTOR. et8_head_v3.post_inspect_decisions reconstructs a v3 PYTHON decision
(region_source over a program, decision_prompt from et8_inject). SQL decisions are built by
et8_sql_agent's own SYSTEM prompt and state(). This file reconstructs EXACTLY what that agent saw,
so the states the head is fitted on are the states the head will act on.

It writes X_layer*.npy + meta.jsonl in et8_head_v3's format, so the FIT and its three controls
(permutation, PCA-8, five-draw n=100) run unchanged — the mechanism transfers, the code does too.
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et8_sql_env as E
import et8_sql_agent as SA


def post_inspect_decisions(steps, tasks_dir):
    by = collections.defaultdict(list)
    for s in steps:
        by[(s["run_id"], s["task_id"])].append(s)
    out = []
    for (rid, tid), st in by.items():
        st.sort(key=lambda s: s["step"])
        task = json.load(open(os.path.join(tasks_dir, tid + ".json")))
        parts = dict(task["parts"]); regions = task["regions"]
        conn = E.db(); expected = [list(r) for r in task["expected_rows"]]
        inspected, history = {}, []
        sysmsg = {"role": "system",
                  "content": SA.SYSTEM % (", ".join(regions), E.SCHEMA.strip(), 12)}
        for s in st:
            if s["action"] == "hypothesize":
                if not inspected:
                    break
                got, err = E.run(conn, E.build(parts))
                lines = ["SYMPTOM: the query returns %s rows; the correct query returns %d."
                         % (("an error: " + err) if err else len(got or []), len(expected)),
                         "Clauses: " + ", ".join(regions)]
                for r, v in inspected.items():
                    lines.append("  %s: %s" % (r, v))
                if history:
                    lines.append("So far: " + " | ".join(history[-6:]))
                out.append({"task_id": tid, "run_id": rid, "step": s["step"],
                            "bug_region": task["bug_region"], "regions": regions,
                            "agent_region": s["region"], "agent_hit": bool(s["region_hit"]),
                            "messages": [sysmsg, {"role": "user", "content": "\n".join(lines)}]})
                break
            if s["action"] in ("inspect", "repeat_inspect") and s["region"] in regions:
                inspected[s["region"]] = parts[s["region"]]
                history.append("inspect %s" % s["region"])
            elif s.get("region"):
                history.append("%s %s" % (s["action"], s["region"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", nargs="+", type=int, default=[18, 27])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import et8_head_v3 as H
    from mlx_lm import load
    import mlx.core as mx

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = post_inspect_decisions(steps, a.tasks)
    print("  decisions: %d" % len(dec), file=sys.stderr)
    ks = sorted({len(d["regions"]) for d in dec})
    print("  region counts present: %s -> chance is per task" % ks, file=sys.stderr)

    model, tok = load(a.model)
    os.makedirs(a.out, exist_ok=True)
    X = {l: [] for l in a.layers}; meta = []; t0 = time.time()
    for i, d in enumerate(dec):
        for r in d["regions"]:
            pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
            hs = H.hidden_at(model, tok, d["messages"], a.layers, prefix=pre)
            for l in a.layers:
                X[l].append(np.array(hs[l].astype(mx.float32), copy=False))
            meta.append({"task_id": d["task_id"], "candidate": r,
                         "label": int(r == d["bug_region"]),
                         "agent_region": d["agent_region"], "agent_hit": d["agent_hit"],
                         "n_regions": len(d["regions"])})
        if (i + 1) % 50 == 0:
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
