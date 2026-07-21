#!/usr/bin/env python
"""ET-IV E6 → P6: serial self-revision vs parallel sampling + checker selection, at a MATCHED token
budget. For each prompt: serial arm = form-validity of the final revised draft, budget = total tokens
generated across all rounds. Parallel arm = verifier-selected validity over the first N E1 samples
whose cumulative tokens ≈ the serial budget. P6 predicts the closed loop underperforms parallel+select.
Exact paired McNemar per cell on the two arms. Also reports serial validity vs revision round (does the
closed loop improve at all?).

  python poetry/e6_score.py
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
from e1_score import score_sample                      # same form checkers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mcnemar(a, b):
    B = sum(1 for x, y in zip(a, b) if x and not y)     # serial wins, parallel loses
    C = sum(1 for x, y in zip(a, b) if not x and y)     # parallel wins, serial loses
    n = B + C
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(B, C) + 1)) * 0.5 ** n) if n else 1.0
    return B, C, p


def tok_len(s):
    return max(1, len(s.split()))                       # word-count proxy for matched budget (consistent across arms)


def main():
    out = []
    print("=== P6: serial self-revision vs parallel+selection at matched budget (McNemar) ===")
    for e6_path in sorted(glob.glob("poetry/cache/e6_*.jsonl")):
        tag = os.path.basename(e6_path)[3:-6]
        e6 = [json.loads(l) for l in open(e6_path)]
        e1 = {json.loads(l)["id"]: json.loads(l) for l in open(f"poetry/cache/e1_{tag}.jsonl")} \
            if os.path.exists(f"poetry/cache/e1_{tag}.jsonl") else {}
        ser, par, ids = [], [], []
        matched_N = []
        for r in e6:
            i = r["id"]
            if i not in e1:
                continue
            # serial: final draft validity + total budget
            serial_valid = int(score_sample(r["drafts"][-1], r)[0])
            budget = sum(tok_len(d) for d in r["drafts"])
            # parallel: accumulate E1 samples until cumulative tokens ~ budget, select best by checker score
            samples = e1[i]["samples"]
            cum, pool = 0, []
            for s in samples:
                cum += tok_len(s); pool.append(s)
                if cum >= budget:
                    break
            scored = [score_sample(s, r) for s in pool]           # (valid, score, meter, oov)
            best = max(range(len(scored)), key=lambda k: scored[k][1])
            parallel_valid = int(scored[best][0])
            ser.append(serial_valid); par.append(parallel_valid); ids.append(i); matched_N.append(len(pool))
        if not ids:
            continue
        b, c, p = mcnemar(ser, par)
        s_rate, p_rate = 100 * sum(ser) / len(ser), 100 * sum(par) / len(par)
        # serial validity by round (does revision help?)
        rounds = len(e6[0]["drafts"])
        by_round = [round(100 * sum(int(score_sample(r["drafts"][k], r)[0]) for r in e6 if r["id"] in e1)
                          / len(ids), 1) for k in range(rounds)]
        verdict = ("parallel>serial (P6 supported)" if p_rate > s_rate and p < 0.05
                   else "serial>parallel (P6 CONTRA)" if s_rate > p_rate and p < 0.05 else "tie/ns")
        out.append({"policy": tag, "n": len(ids), "serial_final_valid": round(s_rate, 1),
                    "parallel_selected_valid": round(p_rate, 1), "mean_matched_N": round(sum(matched_N) / len(matched_N), 1),
                    "serial_gt_parallel": b, "parallel_gt_serial": c, "p": round(p, 4),
                    "serial_valid_by_round": by_round, "verdict": verdict})
        print(f"  {tag:>4}  serial={s_rate:5.1f}  parallel@~{sum(matched_N)/len(matched_N):.0f}={p_rate:5.1f}  "
              f"b/c={b}/{c}  p={p:.4f}  round-curve={by_round}  -> {verdict}")
    json.dump(out, open("poetry/e6_scores.json", "w"), indent=2)
    n_p6 = sum("supported" in o["verdict"] for o in out)
    print(f"\n[P6] parallel>serial significant in {n_p6}/{len(out)} cells. wrote poetry/e6_scores.json")


if __name__ == "__main__":
    main()
