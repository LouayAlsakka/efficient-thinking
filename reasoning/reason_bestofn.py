#!/usr/bin/env python
"""Best-of-N with a real evaluator — does a BETTER SELECTOR extract more than majority vote?

We showed a *perfect* verifier beats self-consistency (+14.2). Here we test a *real, imperfect* evaluator
(the Kimi-K2.5 master judge) as the selector: sample N answers from Qwen, and compare three ways of
spending those N samples —
  * self-consistency@N  — majority vote (verifier-free)
  * Kimi-best-of-N       — the judge picks the candidate whose final answer is correct
  * oracle-best-of-N     — is any of the N correct (the ceiling)
If Kimi-best-of-N > self-consistency and approaches oracle, then 'more search' pays only when a better
evaluator spends it — the thesis, operationalised. Two stages (generate on GPU, score on Bedrock):

  # llm2 (mlx): sample N full answers/problem
  python reason_bestofn.py generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --problems 60 --nmax 8
  # llm1 (boto3): the evaluator spends the samples
  python3 reason_bestofn.py score --data reasoning/bestofn_samples.json
"""
import argparse, json, random
from collections import Counter


def generate(args):
    from mlx_lm import load, generate as gen
    from mlx_lm.sample_utils import make_sampler
    probs = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(probs); probs = probs[:args.problems]
    golds = [p["answer"].split("####")[-1].strip().replace(",", "") for p in probs]
    model, tok = load(args.model)
    sampler = make_sampler(temp=args.temp)
    out = []
    for i, (p, g) in enumerate(zip(probs, golds)):
        msgs = [{"role": "user", "content": p["question"] + "\nThink step by step, then end with: #### <number>"}]
        pr = tok.apply_chat_template(msgs, add_generation_prompt=True)
        samples = [gen(model, tok, prompt=pr, max_tokens=args.max_tokens, sampler=sampler, verbose=False)
                   for _ in range(args.nmax)]
        out.append({"q": p["question"], "gold": g, "samples": samples})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(probs)}", flush=True)
    json.dump({"model": args.model, "items": out}, open(args.out, "w"))
    print(f"[bestofn] wrote {args.out} ({len(out)} problems x {args.nmax} samples)", flush=True)


def kimi_pick(rt, problem, cands):
    body = {"messages": [
        {"role": "system", "content": "You are a strict math grader. Several candidate solutions to one "
         "problem follow, numbered. Reply with ONLY the number of the candidate whose FINAL answer is "
         "correct. If several are correct, pick any correct one."},
        {"role": "user", "content": f"[Problem]\n{problem}\n\n" +
         "\n\n".join(f"[Candidate {k+1}]\n{c}" for k, c in enumerate(cands)) +
         f"\n\nWhich candidate (1-{len(cands)}) is correct? Reply with just the number."}],
        "max_tokens": 8, "temperature": 0.0}
    import json as _j
    r = _j.loads(rt.invoke_model(modelId="moonshotai.kimi-k2.5", body=_j.dumps(body))["body"].read())
    t = r["choices"][0]["message"]["content"].strip()
    for tok in t.replace(".", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(cands):
            return int(tok) - 1
    return 0


def score(args):
    import boto3
    from reason_sweep import extract
    d = json.load(open(args.data)); items = d["items"]
    rt = boto3.client("bedrock-runtime", region_name="us-east-1")
    Ns = [n for n in (2, 4, 8, 16) if n <= len(items[0]["samples"])]
    print(f"[bestofn] {d['model']} | {len(items)} problems | selectors: self-consistency / Kimi-best-of-N / oracle", flush=True)
    curve = []
    for N in Ns:
        sc = bo = orc = 0
        for it in items:
            subs = it["samples"][:N]; g = str(it["gold"])
            ans = [extract(s) for s in subs]
            maj = Counter([a for a in ans if a]).most_common(1)
            sc += bool(maj and str(maj[0][0]) == g)
            orc += any(str(a) == g for a in ans)
            pick = kimi_pick(rt, it["q"], subs)
            bo += (str(ans[pick]) == g)
        n = len(items)
        curve.append({"N": N, "self_consistency": round(100*sc/n, 1),
                      "kimi_best_of_n": round(100*bo/n, 1), "oracle_best_of_n": round(100*orc/n, 1)})
        print(f"  N={N:<2}  self-consistency={100*sc/n:5.1f}%   Kimi-best-of-N={100*bo/n:5.1f}%   "
              f"oracle={100*orc/n:5.1f}%", flush=True)
    json.dump({"model": d["model"], "curve": curve}, open(args.out, "w"), indent=2)
    print(f"[bestofn] wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    g.add_argument("--data", default="reasoning/data/gsm8k_test.jsonl")
    g.add_argument("--problems", type=int, default=60); g.add_argument("--nmax", type=int, default=8)
    g.add_argument("--temp", type=float, default=0.8); g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--seed", type=int, default=0); g.add_argument("--out", default="reasoning/bestofn_samples.json")
    g.set_defaults(func=generate)
    s = sub.add_parser("score")
    s.add_argument("--data", default="reasoning/bestofn_samples.json")
    s.add_argument("--out", default="reasoning/bestofn_results.json")
    s.set_defaults(func=score)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
