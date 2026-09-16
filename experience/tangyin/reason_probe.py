"""Reason probe: re-show e3 pairs the judge got wrong / right and ask for ONE sentence of reasoning. 8 calls. Logged like every judge call."""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(__file__))
import judge as jm
HERE = os.path.dirname(os.path.abspath(__file__))
held = {h["title"]: h for h in jm.heldout_qijue(os.path.join(HERE, "corpus", "tangyin_heldout.jsonl"))}
gens = jm.generated_qijue(os.path.join(HERE, "results", "p2_lora_r8e3_s1_v2.jsonl"))
fooled, caught = [], []
for sd in (0, 1, 2):
    d = json.load(open(os.path.join(HERE, "results", f"p1_judge_seed{sd}.json")))
    for i, t in enumerate(d["arms"]["e3"]["trials"]):
        (fooled if t["pick"] != t["truth"] else caught).append((sd, i, t))
rng = random.Random(7)
sel = [("fooled", x) for x in rng.sample(fooled, 4)] + [("caught", x) for x in rng.sample(caught, 4)]
J = jm.Judge("bedrock", "us.anthropic.claude-opus-4-7", os.path.join(HERE, "results", "judge_log.jsonl"), False)
out = []
for tag, (sd, i, t) in sel:
    h, g = held[t["heldout"]], gens[i]
    real_first = t["truth"] == "A"
    A, B = (h, g) if real_first else (g, h)
    prompt = ("下面兩首七言絕句，一首是唐寅（唐伯虎）的真作，另一首不是。哪一首是唐寅的真作？先回答 A 或 B，然後用一句話說明你的理由。\n\nA:\n"
              + "\n".join(A["lines"]) + "\n\nB:\n" + "\n".join(B["lines"]))
    ans = J.ask("reason", prompt, 160)
    out.append({"tag": tag, "seed": sd, "trial": i, "heldout": t["heldout"], "topic": g["topic"], "truth": t["truth"], "earlier_pick": t["pick"], "answer": ans.strip()})
    print(f"[{tag}] truth={t['truth']} earlier={t['pick']} held={t['heldout']} topic={g['topic']}\n   {ans.strip()[:300]}\n")
json.dump(out, open(os.path.join(HERE, "results", "p1_reason_probe.json"), "w"), ensure_ascii=False, indent=1)
print("calls:", J.calls)
