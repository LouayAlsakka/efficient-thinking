#!/usr/bin/env python
"""Does SEARCH improve the EVALUATOR itself? — judge self-consistency.

We showed search extracts more from a *policy*. Here we test the recursion: run the master judge
(Kimi-K2.5) N times per decisive pair (temperature > 0) and take the majority; does agreement with the
ground-truth verifier RISE with N? If yes, you can partly buy back a weak/uncertain evaluator with
compute — search improves the referee too. Uses the arena answers; only clean-extraction decisive pairs
(one answer verifiably right, the other wrong) so 'truth' is unambiguous.

  python3 reasoning/reason_judge_scaling.py --answers reasoning/arena_answers_5.json --pairs 40 --votes 5
"""
import argparse, json, random
from collections import Counter
import boto3
from reason_math_sweep import extract_boxed, normalize

MODEL_ID = "moonshotai.kimi-k2.5"


def jcall(rt, problem, a, b, temp):
    sysp = ("You are a strict, impartial grader. Two assistants answered the same problem. Decide whose "
            "FINAL answer is correct. Reply with EXACTLY one token: A, B, or TIE.")
    user = f"[Problem]\n{problem}\n\n[Assistant A]\n{a}\n\n[Assistant B]\n{b}\n\nWhich is correct? A, B, or TIE."
    body = {"messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}],
            "max_tokens": 8, "temperature": temp}
    r = json.loads(rt.invoke_model(modelId=MODEL_ID, body=json.dumps(body))["body"].read())
    t = r["choices"][0]["message"]["content"].strip().upper()
    return "A" if t.startswith("A") else ("B" if t.startswith("B") else "TIE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="reasoning/arena_answers_5.json")
    ap.add_argument("--pairs", type=int, default=40, help="decisive pairs to sample")
    ap.add_argument("--votes", type=int, default=5, help="max judge samples per pair")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reasoning/judge_scaling.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    d = json.load(open(args.answers)); Q = d["questions"]; A = d["answers"]; names = list(A)
    # collect clean decisive pairs: both extract, exactly one == gold
    decisive = []
    for qi, q in enumerate(Q):
        gold = normalize(q.get("gold"))
        if gold is None:
            continue
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ai, aj = A[names[i]][qi], A[names[j]][qi]
                ci, cj = normalize(extract_boxed(ai)), normalize(extract_boxed(aj))
                if ci is None or cj is None:
                    continue
                ri, rj = (ci == gold), (cj == gold)
                if ri != rj:                                  # exactly one correct
                    # present in a fixed random order; record which shown-slot is the true winner
                    if rng.random() < 0.5:
                        pa, pb, true_slot = ai, aj, ("A" if ri else "B")
                    else:
                        pa, pb, true_slot = aj, ai, ("A" if rj else "B")
                    decisive.append((q["problem"], pa, pb, true_slot))
    rng.shuffle(decisive); decisive = decisive[:args.pairs]
    print(f"[judge-scaling] {len(decisive)} clean decisive pairs x up to {args.votes} votes", flush=True)

    rt = boto3.client("bedrock-runtime", region_name="us-east-1")
    votes = []                                                 # per pair: (list_of_votes, true_slot)
    for k, (prob, pa, pb, true_slot) in enumerate(decisive):
        vs = [jcall(rt, prob, pa, pb, args.temp) for _ in range(args.votes)]
        votes.append((vs, true_slot))
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(decisive)}", flush=True)

    curve = []
    for N in [n for n in (1, 3, 5, 7, 9) if n <= args.votes]:
        agree = 0
        for vs, true_slot in votes:
            sub = [v for v in vs[:N] if v != "TIE"]
            maj = Counter(sub).most_common(1)
            pick = maj[0][0] if maj else "TIE"
            agree += (pick == true_slot)
        acc = 100.0 * agree / len(votes)
        curve.append((N, round(acc, 1)))
        print(f"  judge@{N}: agrees with verifier {acc:.1f}%", flush=True)
    json.dump({"pairs": len(votes), "curve": curve}, open(args.out, "w"), indent=2)
    print(f"[judge-scaling] wrote {args.out} — does agreement rise with N? = does search improve the evaluator?", flush=True)


if __name__ == "__main__":
    main()
