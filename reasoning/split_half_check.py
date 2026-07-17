#!/usr/bin/env python
"""A1 — split-half out-of-sample form of §5 Prop 2. The in-sample check (sc@32 == modal fraction) is an
identity at N=32; this is the honest out-of-sample version: the MODE of samples 1-16 should predict the
MAJORITY VOTE of samples 17-32, per model, iff the 32 samples are exchangeable draws from a stable answer
distribution. Agreement within sampling noise = a scored hit; any model deviating >3 pts localizes an
exchangeability violation and is itself informative. Pure post-processing over the GSM8K 32-sample caches.

  python reasoning/split_half_check.py
"""
import glob, json, re
from collections import Counter


def extract(t):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", t or "")
    if m:
        return m[-1].replace(",", "")
    n = re.findall(r"-?\d[\d,]*", t or "")
    return n[-1].replace(",", "") if n else None


def vote(samples):
    c = Counter([x for x in (extract(s) for s in samples) if x])
    return c.most_common(1)[0][0] if c else None


order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3, "14B": 4, "32B": 5, "72B": 6}
out = []
for f in sorted(glob.glob("reasoning/cache/gsm8k_*.jsonl"),
                key=lambda p: order.get(p.split("gsm8k_")[1].replace(".jsonl", ""), 9)):
    tag = f.split("gsm8k_")[1].replace(".jsonl", "")
    items = [json.loads(l) for l in open(f)]
    if min(len(it["samples"]) for it in items) < 32:
        continue
    mode_A = sc_B = n = 0
    for it in items:
        g = str(it["gold"]); s = it["samples"]
        mode_A += (str(vote(s[:16])) == g)          # mode of first half
        sc_B += (str(vote(s[16:32])) == g)          # majority of second half
        n += 1
    row = {"model": tag, "n": n, "mode_A_correct": round(100 * mode_A / n, 1),
           "sc_B": round(100 * sc_B / n, 1)}
    row["delta"] = round(row["mode_A_correct"] - row["sc_B"], 1)
    out.append(row)
    print(f"  {tag:>5}: mode(1-16)={row['mode_A_correct']}  vote(17-32)={row['sc_B']}  Δ={row['delta']:+.1f}  (n={n})", flush=True)

json.dump(out, open("reasoning/split_half_check.json", "w"), indent=2)
maxdev = max(abs(r["delta"]) for r in out)
print(f"\n[split-half] wrote reasoning/split_half_check.json | max |Δ| = {maxdev} pts → "
      f"{'HIT: out-of-sample agreement within noise (Prop 2 holds on clean data)' if maxdev <= 3 else 'a model deviates >3 pts — exchangeability violation, report where'}")
