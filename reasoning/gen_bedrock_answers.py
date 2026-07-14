#!/usr/bin/env python
"""Add a Bedrock model (e.g. Kimi-K2.5) as a contestant to an arena answers file — answer every question,
so a frontier model can sit on the pairwise reasoning-GELO ladder as the high anchor.

  python3 reasoning/gen_bedrock_answers.py --answers reasoning/arena_answers_big.json \
      --model-id moonshotai.kimi-k2.5 --name Kimi-K2.5 --out reasoning/arena_answers_5.json
"""
import argparse, json, boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--model-id", default="moonshotai.kimi-k2.5")
    ap.add_argument("--name", default="Kimi-K2.5")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.answers))
    rt = boto3.client("bedrock-runtime", region_name=args.region)
    outs = []
    for i, q in enumerate(d["questions"]):
        body = {"messages": [{"role": "user", "content": q["problem"] + "\nSolve step by step, then state your final answer."}],
                "max_tokens": args.max_tokens, "temperature": args.temp}
        try:
            r = json.loads(rt.invoke_model(modelId=args.model_id, body=json.dumps(body))["body"].read())
            outs.append(r["choices"][0]["message"]["content"])
        except Exception as e:
            outs.append(f"[error: {e}]")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(d['questions'])}", flush=True)
    d["answers"][args.name] = outs
    json.dump(d, open(args.out, "w"))
    print(f"[bedrock-gen] added {args.name} ({len(outs)} answers) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
