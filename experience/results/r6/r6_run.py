#!/usr/bin/env python3
"""Run one R6 baseline arm through et8_agent's own decision plumbing. No edit to the measured agent."""
import argparse, glob, json, os, sys, time, uuid
R = os.path.expanduser("~/github/efficient-thinking/experience")
sys.path.insert(0, R)
import et8_agent as A
import et8_r6_baselines as B

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["rerank", "vote"], required=True)
ap.add_argument("--tasks", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--budget", type=int, default=12)
a = ap.parse_args()

head = B.self_rerank_head() if a.arm == "rerank" else B.majority_vote_head()
files = sorted(glob.glob(os.path.join(a.tasks, "task_*.json")))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
model, tok = A.load_model(a.model)
ep = open(a.out + ".episodes.jsonl", "w")
st = open(a.out + ".steps.jsonl", "w")
for i, f in enumerate(files, 1):
    task = json.load(open(f))
    r = A.run_episode(model, tok, task, a.budget, None, run_id, st, a.model,
                      inspect_first=True, head_state=head)
    ep.write(json.dumps(r) + "\n"); ep.flush()
    print("[%d/%d] %s green=%s actions=%s" % (i, len(files), task["task_id"], r["green"], r["actions"]),
          file=sys.stderr)
ep.close(); st.close()
