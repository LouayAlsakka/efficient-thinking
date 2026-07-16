#!/usr/bin/env python
"""Mode-correctness check — §5 Proposition 2's registered prediction: per model, self-consistency at large N
should sit at the fraction of problems whose correct answer is the MODE of the sampled answers. sc@32 is
majority-vote-correctness, so it equals fraction-modal-correct by construction UNLESS answer-extraction
collisions perturb the mode — so a deviation >a point or two localizes an extraction/i.i.d. artifact and is
itself the finding. Pure post-processing over the GSM8K 32-sample caches.

  python reasoning/mode_check.py
"""
import glob, json, re
from collections import Counter


def extract(t):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*)", t or "")
    if m:
        return m[-1].replace(",", "")
    n = re.findall(r"-?\d[\d,]*", t or "")
    return n[-1].replace(",", "") if n else None


out = []
for f in sorted(glob.glob("reasoning/cache/gsm8k_*.jsonl")):
    tag = f.split("gsm8k_")[1].replace(".jsonl", "")
    items = [json.loads(l) for l in open(f)]
    N = min(len(it["samples"]) for it in items)
    modal_correct = sc = n = 0
    for it in items:
        g = str(it["gold"]); ans = [extract(s) for s in it["samples"][:N]]
        c = Counter([a for a in ans if a])
        mode = c.most_common(1)[0][0] if c else None
        modal_correct += (str(mode) == g)          # is the modal answer correct?
        sc += bool(c and str(c.most_common(1)[0][0]) == g)   # sc@N (== modal by construction)
        n += 1
    row = {"model": tag, "N": N, "frac_modal_correct": round(100 * modal_correct / n, 1),
           "sc_at_N": round(100 * sc / n, 1), "n": n}
    row["delta"] = round(row["sc_at_N"] - row["frac_modal_correct"], 1)
    out.append(row)
    print(f"  {tag:>5}: frac-modal-correct={row['frac_modal_correct']}  sc@{N}={row['sc_at_N']}  "
          f"Δ={row['delta']:+.1f}  (n={n})", flush=True)

order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3, "14B": 4, "32B": 5, "72B": 6}
out.sort(key=lambda r: order.get(r["model"], 9))
json.dump(out, open("reasoning/mode_check.json", "w"), indent=2)
maxdev = max(abs(r["delta"]) for r in out)
print(f"\n[mode-check] wrote reasoning/mode_check.json | max |Δ| = {maxdev} "
      f"→ {'HIT (sc@N == modal fraction, no extraction artifact)' if maxdev <= 0.1 else 'deviation localizes an extraction/i.i.d. artifact'}")
