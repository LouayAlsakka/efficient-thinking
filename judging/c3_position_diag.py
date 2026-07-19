#!/usr/bin/env python
"""C3 — 72B-judge N=16 anomaly diagnostic (7B policy). The 72B judge degrades at N=16 against the
shrinking-degradation trend; this instruments pick-best to log which candidate POSITION it chooses and
where the correct answers sit, testing for position collapse / lost-in-the-middle at ~15k-token context in
4-bit. If picks concentrate at list edges, or the judge systematically misses correct answers in middle
positions, position bias is implicated and flagged in III's methods — the cells are not silently excluded.

  python judging/c3_position_diag.py
"""
import json, argparse
from collections import Counter
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from et3_judge import extract, pick_best, load_problems
from mlx_lm import load, generate as gen

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="mlx-community/Qwen2.5-72B-Instruct-4bit")
ap.add_argument("--out", default="judging/c3_position_diag.json")
a = ap.parse_args()

N = 16
NPROB = 60
probs = load_problems("reasoning/data/gsm8k_test.jsonl", NPROB)
golds = [str(p["answer"].split("####")[-1].strip().replace(",", "")) for p in probs]
items = [json.loads(l) for l in open("reasoning/cache/gsm8k_7B.jsonl")][:NPROB]
model, tok = load(a.judge)

rows = []
for i in range(min(len(items), len(probs))):
    subs = items[i]["samples"][:N]; g = golds[i]
    ans = [extract(s) for s in subs]
    corr = [k for k, a in enumerate(ans) if str(a) == g]
    pick, _ = pick_best(model, tok, gen, probs[i]["question"], subs)
    rows.append({"chosen": pick, "correct_positions": corr, "picked_correct": str(ans[pick]) == g,
                 "covered": len(corr) > 0})
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{NPROB}", flush=True)

hist = Counter(r["chosen"] for r in rows)
# uniform expectation = NPROB/16 per position; edge concentration = positions {0,1,14,15}
edge = sum(hist.get(k, 0) for k in (0, 1, 14, 15)); mid = len(rows) - edge
# among covered problems the judge got WRONG, where did the correct answer sit? (lost-in-the-middle signal)
miss_corr_pos = [p for r in rows if r["covered"] and not r["picked_correct"] for p in r["correct_positions"]]
print("\n[C3] chosen-position histogram (0-15):")
print("  " + " ".join(f"{k}:{hist.get(k,0)}" for k in range(N)))
print(f"[C3] edge positions {{0,1,14,15}} chosen {edge}/{len(rows)} (uniform would be {4*len(rows)/N:.1f}); middle {mid}")
if miss_corr_pos:
    mc = Counter(miss_corr_pos)
    print(f"[C3] when judge missed a covered problem, correct answers sat at positions: "
          f"{dict(sorted(mc.items()))} (mid-heavy = lost-in-the-middle)")
json.dump({"n": len(rows), "N": N, "policy": "7B", "judge": "72B",
           "chosen_hist": {str(k): hist.get(k, 0) for k in range(N)},
           "edge_chosen": edge, "mid_chosen": mid, "uniform_edge_expect": round(4*len(rows)/N, 1),
           "judge_model": a.judge,
           "missed_correct_positions": Counter(miss_corr_pos), "rows": rows},
          open(a.out, "w"), indent=2, default=int)
print(f"\n[C3] wrote {a.out}")
