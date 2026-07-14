#!/usr/bin/env python
"""Pairwise reasoning-GELO arena (route B in docs/gelo.md): contestants answer the same questions, a
MASTER JUDGE (Bedrock Kimi-K2.5, assumed stronger than either contestant) decides each head-to-head,
and GELO comes from the Bradley-Terry cross-table, anchored to a chosen reference model.

Two stages so the GPU-heavy part and the AWS part can run on different machines:

  # 1) on a machine with mlx_lm + the local models (e.g. llm2): each contestant answers every question
  python reason_arena.py generate --models mlx-community/Qwen2.5-1.5B-Instruct-4bit,mlx-community/Qwen3.5-4B-MLX-4bit \
      --per-level 8 --out reasoning/arena_answers.json

  # 2) on a machine with boto3 + ~/.aws/credentials: the master judge scores every pair -> BT-Elo
  python reason_arena.py judge --answers reasoning/arena_answers.json --anchor-model Qwen2.5-1.5B-Instruct-4bit --anchor 2000

The judge is BLINDED (never sees model names) and pair order is RANDOMIZED (controls position bias).
On verifiable items (a `gold` answer is present) the judge's pick is also checked against ground truth,
so we can report how often the master judge agrees with the verifier — i.e. how trustworthy the judge is.
"""
import argparse, json, math, os, random, sys
import numpy as np


# ---------------- stage 1: generate contestant answers (mlx_lm) --------------------------------------
def generate(args):
    from mlx_lm import load, generate as gen
    from mlx_lm.sample_utils import make_sampler
    rows = [json.loads(l) for l in open(args.data)]
    rng = random.Random(args.seed); by = {}
    for r in rows:
        by.setdefault(r["level"], []).append(r)
    picked = []
    for lv in sorted(by):
        rng.shuffle(by[lv]); picked += by[lv][:args.per_level]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out = {"questions": [{"problem": r["problem"], "gold": r.get("answer"), "level": r["level"]} for r in picked],
           "answers": {}}
    for m in models:
        model, tok = load(m)
        sampler = make_sampler(temp=args.temp)
        outs = []
        for r in picked:
            msgs = [{"role": "user", "content": r["problem"] + "\nSolve step by step, then state your final answer."}]
            pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
            outs.append(gen(model, tok, prompt=pr, max_tokens=args.max_tokens, sampler=sampler, verbose=False))
        out["answers"][m.split("/")[-1]] = outs
        print(f"[arena] generated {len(outs)} answers for {m.split('/')[-1]}", flush=True)
    json.dump(out, open(args.out, "w"))
    print(f"[arena] wrote {args.out} ({len(models)} contestants x {len(picked)} questions)", flush=True)


# ---------------- stage 2: master judge + Bradley-Terry Elo ------------------------------------------
def kimi_judge(problem, ans_A, ans_B, model_id="moonshotai.kimi-k2.5", region="us-east-1"):
    import boto3
    rt = boto3.client("bedrock-runtime", region_name=region)
    sysp = ("You are a strict, impartial grader. Two assistants answered the same problem. Decide whose "
            "FINAL answer is correct (or, if neither is objectively checkable, which reasoning is better). "
            "Reply with EXACTLY one token: A, B, or TIE.")
    user = f"[Problem]\n{problem}\n\n[Assistant A]\n{ans_A}\n\n[Assistant B]\n{ans_B}\n\nWhich is better? A, B, or TIE."
    body = {"messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}],
            "max_tokens": 8, "temperature": 0.0}
    r = json.loads(rt.invoke_model(modelId=model_id, body=json.dumps(body))["body"].read())
    t = r["choices"][0]["message"]["content"].strip().upper()
    return "A" if t.startswith("A") else ("B" if t.startswith("B") else "TIE")


def bt_elo(names, W):
    """Zermelo/MM Bradley-Terry fit; returns GELO (400 = 10x), zero-mean before anchoring."""
    n = len(names); r = np.ones(n); wins = W.sum(1)
    for _ in range(5000):
        rn = r.copy()
        for i in range(n):
            den = sum((W[i, j] + W[j, i]) / (r[i] + r[j]) for j in range(n) if j != i and (W[i, j] + W[j, i]) > 0)
            if den > 0 and wins[i] > 0:
                rn[i] = wins[i] / den
        rn /= rn.mean(); r = rn
    return (400.0 / math.log(10)) * np.log(r)


def judge(args):
    d = json.load(open(args.answers)); Q = d["questions"]; A = d["answers"]
    names = list(A); n = len(names); idx = {m: i for i, m in enumerate(names)}
    rng = random.Random(args.seed)
    Wj = np.zeros((n, n)); Wt = np.zeros((n, n))          # judge cross-table; ground-truth cross-table
    jt_agree = jt_tot = 0
    from reason_math_sweep import extract_boxed, normalize   # for ground-truth scoring
    for qi, q in enumerate(Q):
        gold = normalize(q.get("gold"))
        for i in range(n):
            for j in range(i + 1, n):
                mi, mj = names[i], names[j]
                ai, aj = A[mi][qi], A[mj][qi]
                flip = rng.random() < 0.5                  # randomize which model is shown as "A"
                pa, pb = (aj, ai) if flip else (ai, aj)
                v = kimi_judge(q["problem"], pa, pb)
                if v == "TIE":
                    Wj[i, j] += 0.5; Wj[j, i] += 0.5
                else:
                    first_wins = (v == "A")
                    winner = (mj if flip else mi) if first_wins else (mi if flip else mj)
                    loser = mj if winner == mi else mi
                    Wj[idx[winner], idx[loser]] += 1
                if gold is not None:                       # ground-truth (verifier) cross-table + judge check
                    ci = normalize(extract_boxed(ai)) == gold
                    cj = normalize(extract_boxed(aj)) == gold
                    if ci != cj:
                        tw, tl = (mi, mj) if ci else (mj, mi)
                        Wt[idx[tw], idx[tl]] += 1
                        jt_tot += 1
                        jt_agree += (v != "TIE" and (winner == tw))
        if (qi + 1) % 5 == 0:
            print(f"  judged {qi+1}/{len(Q)} questions", flush=True)

    elo_j = bt_elo(names, Wj)
    if args.anchor_model in names:
        elo_j += args.anchor - elo_j[idx[args.anchor_model]]
    else:
        elo_j += args.anchor - elo_j.mean()
    print("\n=== reasoning-GELO (pairwise, master judge = Kimi-K2.5) ===")
    for k in np.argsort(-elo_j):
        print(f"  {names[k]:<34} {elo_j[k]:+7.0f}")
    if jt_tot:
        print(f"\njudge vs ground-truth verifier: agrees on {jt_agree}/{jt_tot} decisive pairs "
              f"({100*jt_agree/jt_tot:.0f}%) — a check on how trustworthy the master judge is.")
        elo_t = bt_elo(names, Wt)
        if args.anchor_model in names:
            elo_t += args.anchor - elo_t[idx[args.anchor_model]]
        print("(ground-truth-only GELO for comparison:)")
        for k in np.argsort(-elo_t):
            print(f"  {names[k]:<34} {elo_t[k]:+7.0f}")
    json.dump({"names": names, "judge_gelo": {names[i]: round(float(elo_j[i])) for i in range(n)},
               "judge_vs_truth_agreement": (jt_agree / jt_tot if jt_tot else None)},
              open(args.out, "w"), indent=2)
    print(f"\n[arena] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--models", required=True, help="comma-separated mlx model ids (the contestants)")
    g.add_argument("--data", default="reasoning/data/math500.jsonl")
    g.add_argument("--per-level", type=int, default=8)
    g.add_argument("--temp", type=float, default=0.6)
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="reasoning/arena_answers.json")
    g.set_defaults(func=generate)
    j = sub.add_parser("judge")
    j.add_argument("--answers", default="reasoning/arena_answers.json")
    j.add_argument("--anchor-model", default=None, help="pin this contestant to --anchor GELO")
    j.add_argument("--anchor", type=float, default=2000.0)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--out", default="reasoning/arena_gelo.json")
    j.set_defaults(func=judge)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
