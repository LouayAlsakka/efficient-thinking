#!/usr/bin/env python3
"""R2d — the gate R2c's failure earned, pre-registered by 理 (11343) before this file existed.

R2c FAILED and STAYS FAILED. Its gate was raw-text identity between the shared-cache path and an
uncached generate; it read 33/40 and is NOT relaxed after the fact. `results/r2c_held.json` says
VOID and 8a carries the compute premium as the measured system's number.

R2d is a NEW gate with a CONTROL that says whether the old one could ever have passed. 理's words:
"this is not choosing an instrument by its answer: R2c stays failed; R2d is a new gate with a
control that says whether the old one could ever have passed, written before its own run."

  (i)   CALIBRATION, and it is the part that makes this honest. UNCACHED vs UNCACHED, same hardware,
        same inputs, run twice. If raw-text identity is ~100% here, then text identity IS achievable
        and R2c's 33/40 is attributable to the cache. If it is ~80%, raw-text identity is a property
        of greedy bf16 decoding and the strict gate was an instrument that COULD NOT PASS.
        PRINTED EITHER WAY. It does not gate anything by itself; it interprets (ii).
  (ii)  parsed-ACTION identity (action, region, bug_class) at n = 300, not 40.
  (iii) paired OUTCOMES on the same 75 held-out: success and actions identical. Separate file --
        it needs the agent wired to the shared cache, not just the decision reproduced.

PASS all three -> the cost may be reported under R2d, with R2c's failure printed beside it and the
calibration number that explains it. FAIL any -> the premium stands and the paper says the cache
changes the agent.
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--calib-n", type=int, default=40)
    ap.add_argument("--action-n", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import et8_head_v3 as H, et8_agent as A, et8_head_shared as S
    from mlx_lm import load

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = H.post_inspect_decisions(steps, a.tasks)
    model, tok = load(a.model)
    key = lambda d: (d.get("action"), d.get("region"), d.get("bug_class"))

    # ---------- (i) CALIBRATION: uncached vs uncached, same inputs, twice
    t0 = time.time()
    cal_same = 0
    cal = dec[: a.calib_n]
    for d in cal:
        pre = S.PRE % d["regions"][0]
        t1, _, _ = A.generate(model, tok, d["messages"], max_tokens=a.max_tokens, temp=0.0, prefix=pre)
        t2, _, _ = A.generate(model, tok, d["messages"], max_tokens=a.max_tokens, temp=0.0, prefix=pre)
        cal_same += (t1 == t2)
    calib = {"n": len(cal), "raw_text_identical": "%d/%d" % (cal_same, len(cal)),
             "pct": round(100.0 * cal_same / max(1, len(cal)), 1)}
    print("  (i) calibration uncached-vs-uncached: %s (%.1f%%)  [%.0fs]"
          % (calib["raw_text_identical"], calib["pct"], time.time() - t0), file=sys.stderr)

    # ---------- (ii) parsed-ACTION identity, cached vs uncached, at n
    use = dec[: a.action_n]
    act_same = text_same = 0
    mismatches = []
    t0 = time.time()
    for i, d in enumerate(use):
        pick = d["regions"][0]
        pre = S.PRE % pick
        _, _, _, txt, _, _ = S.shared_decision(model, tok, d["messages"], a.layer, d["regions"],
                                               max_tokens=a.max_tokens, pick=pick)
        ref, _, _ = A.generate(model, tok, d["messages"], max_tokens=a.max_tokens, temp=0.0, prefix=pre)
        text_same += (ref[len(pre):].strip() == (txt or "").strip())
        ka, kb = key(A.parse_action(ref)), key(A.parse_action(pre + (txt or "")))
        if ka == kb:
            act_same += 1
        else:
            mismatches.append({"i": i, "uncached": list(ka), "cached": list(kb)})
        if (i + 1) % 25 == 0:
            print("  (ii) %d/%d  action_same=%d  [%.0fs]" % (i + 1, len(use), act_same, time.time() - t0),
                  file=sys.stderr)

    n = len(use)
    res = {"document": "R2d (i) calibration + (ii) parsed-action identity — 理 11343, pre-registered",
           "R2c_stays_failed": "results/r2c_held.json is VOID and is not reopened. R2d is a new gate.",
           "model": a.model, "max_tokens": a.max_tokens,
           "(i) CALIBRATION uncached vs uncached": calib,
           "(i) how to read it": ("~100% => raw-text identity IS achievable on this hardware, and "
                                  "R2c's 33/40 is attributable to the cache. ~80% => raw-text "
                                  "identity is a property of greedy bf16 decoding and R2c's gate "
                                  "was an instrument that could not pass. Printed either way; it "
                                  "interprets (ii) and gates nothing by itself."),
           "(ii) n": n,
           "(ii) raw_text_identical": "%d/%d = %.1f%%" % (text_same, n, 100.0 * text_same / n),
           "(ii) PARSED_ACTION_identical": "%d/%d = %.1f%%" % (act_same, n, 100.0 * act_same / n),
           "(ii) PASS": bool(act_same == n),
           "(ii) mismatches": mismatches[:20],
           "(iii)": "NOT IN THIS FILE — paired outcomes need the agent wired to the shared cache.",
           "VERDICT": None,
           "bounds": ["One model, temp 0, max_tokens %d." % a.max_tokens,
                      "(ii) compares the FIRST post-inspect decision of each episode, which is where "
                      "the head acts. It does not re-run whole episodes; that is (iii).",
                      "A parsed-action match is (action, region, bug_class). The free-text `why` is "
                      "NOT compared, by design -- the harness never reads it."]}
    res["VERDICT"] = ("(ii) PASSES — the shared cache never changed what the agent did"
                      if act_same == n else
                      "(ii) FAILS — the shared cache changed the agent's action on %d of %d decisions"
                      % (n - act_same, n))
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "(ii) mismatches"}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
