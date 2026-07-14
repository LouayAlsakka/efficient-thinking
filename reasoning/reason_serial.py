#!/usr/bin/env python
"""Serial search axis — accuracy vs thinking-token budget.

Parallel search (self-consistency) is one axis; the other is *serial* — more reasoning per attempt. We
measure it directly: greedy accuracy as the generation (thinking) budget grows. A non-thinking base model
gains as the budget stops truncating its chain of thought, then plateaus once it has room to finish. This
fills the one axis the paper asserts but had not measured.

  python reason_serial.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --problems 80
"""
import argparse, json, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--data", default="reasoning/data/gsm8k_test.jsonl")
    ap.add_argument("--problems", type=int, default=80)
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    out = args.out or f"reasoning/serial_{tag}.json"

    from mlx_lm import load, generate as gen
    from reason_sweep import extract
    probs = [json.loads(l) for l in open(args.data)]
    random.Random(args.seed).shuffle(probs); probs = probs[:args.problems]
    golds = [p["answer"].split("####")[-1].strip().replace(",", "") for p in probs]
    model, tok = load(args.model)

    prompts = []
    for p in probs:
        msgs = [{"role": "user", "content": p["question"] + "\nThink step by step, then end with: #### <number>"}]
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True))

    print(f"[serial] {args.model} | {len(probs)} problems | budgets {args.budgets}", flush=True)
    curve = []
    for B in args.budgets:                              # greedy (temp 0) at each token budget
        correct = 0
        for pr, g in zip(prompts, golds):
            a = extract(gen(model, tok, prompt=pr, max_tokens=B, verbose=False))
            correct += (str(a) == str(g))
        acc = 100.0 * correct / len(probs)
        curve.append((B, round(acc, 1)))
        print(f"  budget={B:<5} accuracy={acc:5.1f}%", flush=True)
    json.dump({"model": args.model, "serial_curve": curve}, open(out, "w"), indent=2)
    print(f"[serial] wrote {out} — the serial (thinking-length) axis", flush=True)


if __name__ == "__main__":
    main()
