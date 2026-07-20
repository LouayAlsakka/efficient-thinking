#!/usr/bin/env python
"""ET-III external-validation Run 2 (Llama-3.1-8B family check) — scores L1/L2 from committed JSONs.

L1 (criterion transfers across model family): the Llama-8B judge's wins/losses on GSM8K are governed
    by the same selection-coverage criterion as the Qwen judges — necessity (no cell beats majority
    with q_judge <= mode-correctness) is exceptionless, and coverage gates sufficiency. Every claimed
    win carries an exact paired McNemar p-value.
L2 (position collapse is not Qwen-specific): the Llama judge's pick-best at N=16 concentrates on list
    edges well above uniform — lost-in-the-middle replicates out-of-family.

  python judging/run2_llama_score.py
"""
import json, math


def mcnemar(a, b):
    B = sum(1 for x, y in zip(a, b) if x and not y)
    C = sum(1 for x, y in zip(a, b) if not x and y)
    n = B + C
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(B, C) + 1)) * 0.5 ** n) if n else 1.0
    return B, C, p


d = json.load(open("judging/e3_llama8B.json"))
print("=== L1: Llama-8B judge on GSM8K — selection-coverage criterion transfer ===")
rows = []
nec_ok = wins = 0
for r in d:
    cov = [i for i, c in enumerate(r["cov"]) if c]
    q = sum(r["pick_ok"][i] for i in cov) / len(cov) if cov else 0.0
    mode = r["majority_acc"] / 100.0
    B, C, p = mcnemar(r["pick_ok"], r["maj_ok"])
    dl = r["pick_best_acc"] - r["majority_acc"]
    won = dl > 0 and p < 0.05
    predicted = q > mode
    necessity = (not won) or predicted        # a real win must clear q > mode
    nec_ok += necessity; wins += won
    rows.append({"policy": r["policy"], "N": r["N"], "delta": round(dl, 1), "p": round(p, 4),
                 "q_judge": round(100 * q, 1), "mode_correct": round(100 * mode, 1),
                 "win": won, "criterion_necessity_ok": necessity})
    print(f"  {r['policy']:>4} N={r['N']:>2}  sel={r['pick_best_acc']:5} maj={r['majority_acc']:5} "
          f"Δ={dl:+5.1f} p={p:.4f}  q={100*q:4.0f} mode={100*mode:4.0f}  "
          f"{'WIN' if won else 'null'}  necessity_ok={necessity}")
print(f"  -> L1: significant wins={wins}; criterion necessity holds in {nec_ok}/{len(rows)} cells "
      f"({'HIT' if nec_ok == len(rows) else 'MISS'})")

c = json.load(open("judging/c3_llama.json"))
replicates = c["edge_chosen"] > 2 * c["uniform_edge_expect"]
print(f"\n=== L2: Llama C3 position collapse ===")
print(f"  edge {{0,1,14,15}} chosen {c['edge_chosen']}/{c['n']} vs uniform {c['uniform_edge_expect']} "
      f"-> {'REPLICATES (HIT)' if replicates else 'weak (MISS)'}")

json.dump({"L1_cells": rows, "L1_necessity": f"{nec_ok}/{len(rows)}", "L1_sig_wins": wins,
           "L2_edge_chosen": c["edge_chosen"], "L2_uniform": c["uniform_edge_expect"],
           "L2_replicates": replicates},
          open("judging/run2_llama_score.json", "w"), indent=2)
print("\n[Run2] wrote judging/run2_llama_score.json")
