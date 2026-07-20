#!/usr/bin/env python
"""ET-IV E1 → P1 significance: exact paired McNemar on the search-lift flip (greedy valid@1 vs
verifier-selected@16) per cell, plus the meter-only frontier. P1 predicts large validity gains from
search; the ET-II crossover (lift collapsing as base competence grows) is read off the size trend.

  python poetry/p1_score.py
"""
import json, math


def mcnemar(a, b):
    B = sum(1 for x, y in zip(a, b) if y and not x)   # gained by search (selected valid, greedy not)
    C = sum(1 for x, y in zip(a, b) if x and not y)   # lost
    n = B + C
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(B, C) + 1)) * 0.5 ** n) if n else 1.0
    return B, C, p


cells = json.load(open("poetry/e1_scores.json"))
out = []
print("=== P1: search-lift significance (greedy valid@1 → verifier-selected@16), exact McNemar ===")
for c in sorted(cells, key=lambda r: (r["task"], r["policy"])):
    per = c["per_prompt"]
    b, cc, p = mcnemar(per["g_valid"], per["sel16"])
    lift = c["verifier_selected_at16"] - c["form_valid_at1"]
    sig = "sig" if p < 0.05 else ("marg" if p < 0.10 else "ns")
    out.append({**{k: c[k] for k in ("policy", "task")}, "valid_at1": c["form_valid_at1"],
                "selected_at16": c["verifier_selected_at16"], "lift": round(lift, 1),
                "gained": b, "lost": cc, "p": round(p, 5), "verdict": sig})
    print(f"  {c['policy']:>4} {c['task']:<10} valid@1={c['form_valid_at1']:5.1f} → sel@16="
          f"{c['verifier_selected_at16']:5.1f}  lift={lift:+5.1f}  gain/lose={b}/{cc}  p={p:.4g}  {sig}")

nsig = sum(o["verdict"] == "sig" for o in out)
print(f"\n[P1] {nsig}/{len(out)} cells show a significant search lift (valid@1 → selected@16).")
json.dump(out, open("poetry/p1_score.json", "w"), indent=2)
print("[P1] wrote poetry/p1_score.json")
