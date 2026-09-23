#!/usr/bin/env python3
"""P1/P3 judge for the Tang Yin study — Claude judges, with NO tools, every request logged.

  python3 judge.py --dry-run                                   # build trials, print counts, call nothing
  python3 judge.py --provider bedrock --model <bedrock model id> --arm baseline=results/p0_base_7b_rescored_v2.jsonl \
                   --arm iter60=results/p2_lora_ck60_s1_v2.jsonl --arm e3=results/p2_lora_r8e3_s1_v2.jsonl

The instrument is DISCRIMINATION, not a rating: one generated 七絕 beside one real held-out 七絕, same
presentation (punctuation stripped, 4 lines x 7 chars), random order; "which is the real Tang Yin poem?"
Mimicry = how far accuracy falls toward 50%. Contamination controls (README):
  search   is a request property: no `tools` field is ever sent; the response is asserted to carry no
           tool-use blocks and the usage is logged. Never run this from a Claude Code session.
  memory   three probes per held-out poem: completion (title + first line -> rest; char overlap >= 0.6
           EXCLUDES the poem from the judge set), attribution (full text, no title -> names Tang Yin: FLAG),
           canary (generated poems, posted nowhere, put to the same attribution question: confident
           recognition calibrates the attribution probe).
Every rate ships with n and a binomial SE. Nothing here regenerates anything.
"""
import argparse, json, os, random, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = re.compile(r"[，。？！、；：,.!?;:\s「」『』()（）\[\]]")

def chunk(s, per): return [s[i:i+per] for i in range(0, len(s), per)]
def norm(text, per):
    s = PUNCT.sub("", text); return "\n".join(chunk(s, per))
def se(p, n): return (p*(1-p)/n) ** 0.5 if n else 0.0

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def heldout_qijue(path):
    out = []
    for r in load_jsonl(path):
        s = PUNCT.sub("", r["text"])
        if len(s) == 28: out.append({"title": r["title"], "chars": s, "lines": chunk(s, 7)})
    return out

def generated_qijue(path):
    return [{"topic": g["topic"], "chars": g["poem"], "lines": chunk(g["poem"], 7), "pass": bool(g.get("pass"))}
            for g in load_jsonl(path) if g.get("form") == "七絕" and g.get("bucket") == "EXTRACTED" and len(g.get("poem", "")) == 28]

class Judge:
    """One call = one Messages request with NO tools. Logs request+response verbatim."""
    def __init__(self, provider, model, log_path, dry):
        self.provider, self.model, self.dry = provider, model, dry
        self.log = open(log_path, "a", encoding="utf-8") if not dry else None
        self.calls = 0
        if not dry:
            if provider == "bedrock":
                import sys as _s, os as _o
                _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "poetry"))
                import api_rater as _AR; self.client = _AR.bedrock_client()   # named principal, nirai 12305
            elif provider == "anthropic":
                import anthropic; self.client = anthropic.Anthropic()
            else: raise SystemExit(f"unknown provider {provider}")
    def ask(self, kind, prompt, max_tokens=200):
        self.calls += 1
        if self.dry: return ""
        req = {"kind": kind, "model": self.model, "prompt": prompt, "max_tokens": max_tokens, "tools": None}
        if self.provider == "bedrock":
            r = self.client.converse(modelId=self.model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                                     inferenceConfig={"maxTokens": max_tokens})   # temperature is rejected by opus-4-7 on Bedrock; determinism is not claimed, the pairing seed is
            assert "toolUse" not in json.dumps(r["output"]), "tool use in a judge response"
            text = "".join(c.get("text", "") for c in r["output"]["message"]["content"]); usage = r.get("usage")
        else:
            r = self.client.messages.create(model=self.model, max_tokens=max_tokens, temperature=0,
                                            messages=[{"role": "user", "content": prompt}])
            assert all(b.type == "text" for b in r.content), "non-text block in a judge response"
            text = "".join(b.text for b in r.content); usage = r.usage.model_dump() if hasattr(r.usage, "model_dump") else str(r.usage)
        self.log.write(json.dumps({"ts": time.time(), "req": req, "resp": text, "usage": usage}, ensure_ascii=False) + "\n"); self.log.flush()
        return text

def probes(J, held, canaries):
    """Returns (kept_heldout, report). Completion excludes; attribution flags; canaries calibrate."""
    kept, rep = [], {"completion_excluded": [], "attribution_flagged": [], "naming_excluded": [], "canary_recognized": 0, "canary_n": len(canaries)}
    for h in held:
        first = h["lines"][0]
        cont = J.ask("completion", f"下面是唐寅（唐伯虎）的一首七言絕句《{h['title']}》的第一句。請直接續寫出其餘三句，只輸出詩句。\n\n{first}", 120)
        rest = "".join(h["lines"][1:]); got = PUNCT.sub("", cont)[:21]
        overlap = sum(1 for a, b in zip(got, rest) if a == b) / max(1, len(rest)) if not J.dry else 0.0
        attr = J.ask("attribution", "這首七言絕句是誰寫的？只回答詩人姓名。\n\n" + "\n".join(h["lines"]), 40)
        if overlap >= 0.6: rep["completion_excluded"].append({"title": h["title"], "overlap": round(overlap, 2)}); continue
        if re.search(r"唐寅|唐伯虎|伯虎", attr): rep["attribution_flagged"].append(h["title"])
        # naming probe (round 5): the reason probe showed the judge NAMES poems it cannot recite once told the era.
        # Same prior as the pair prompt minus the poet: a poem the judge can name is out, not flagged.
        name = J.ask("naming", "這是一首明代的七言絕句。你是否認得這首詩？若認得，請說出作者與題目；若不認得，只回答「不認得」。\n\n" + "\n".join(h["lines"]), 60)
        if re.search(r"唐寅|唐伯虎|伯虎", name): rep["naming_excluded"].append({"title": h["title"], "answer": name.strip()[:80]}); continue
        kept.append(h)
    for c in canaries:
        a = J.ask("canary", "這首七言絕句是誰寫的？只回答詩人姓名。\n\n" + "\n".join(c["lines"]), 40)
        if re.search(r"唐寅|唐伯虎|伯虎", a): rep["canary_recognized"] += 1
    return kept, rep

def discriminate(J, arm, gens, held, rng):
    n = correct = 0; trials = []
    for g in gens:
        h = rng.choice(held); real_first = rng.random() < 0.5
        A, B = (h, g) if real_first else (g, h)
        ans = J.ask("discriminate", "下面兩首七言絕句，一首是唐寅（唐伯虎）的真作，另一首不是。哪一首是唐寅的真作？只回答 A 或 B。\n\nA:\n" +
                    "\n".join(A["lines"]) + "\n\nB:\n" + "\n".join(B["lines"]), 8)
        pick = "A" if "A" in ans.upper()[:3] else "B" if "B" in ans.upper()[:3] else "?"
        truth = "A" if real_first else "B"; n += 1; correct += (pick == truth)
        trials.append({"heldout": h["title"], "topic": g["topic"], "truth": truth, "pick": pick})
    acc = correct / n if n else 0.0
    return {"arm": arm, "n": n, "correct": correct, "accuracy": round(acc, 3), "se": round(se(acc, n), 3), "trials": trials}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=os.path.join(HERE, "corpus", "tangyin_heldout.jsonl"))
    ap.add_argument("--arm", action="append", default=[], help="name=path.jsonl (generated poems)")
    ap.add_argument("--provider", default="bedrock", choices=["bedrock", "anthropic"])
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", ""))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canaries", type=int, default=10, help="generated 七絕 (form-passing) used as attribution canaries")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. _naming")
    a = ap.parse_args()
    if not a.dry_run and not a.model: raise SystemExit("--model is required for a live run")
    rng = random.Random(a.seed)
    held = heldout_qijue(a.heldout)
    arms = {n: generated_qijue(p) for n, p in (x.split("=", 1) for x in a.arm)}
    J = Judge(a.provider, a.model, os.path.join(a.out, "judge_log.jsonl"), a.dry_run)
    canaries = [g for g in next(iter(arms.values()), []) if g["pass"]][: a.canaries]
    kept, rep = probes(J, held, canaries)
    print(f"held-out 七絕 {len(held)} -> judge set {len(kept)} (recited {len(rep['completion_excluded'])}, named {len(rep['naming_excluded'])}, flagged {len(rep['attribution_flagged'])}; canaries {rep['canary_n']}, recognized {rep['canary_recognized']})")
    results = {"seed": a.seed, "provider": a.provider, "model": a.model, "probes": rep, "arms": {}}
    for name, gens in arms.items():
        r = discriminate(J, name, gens, kept or held, rng)
        results["arms"][name] = r
        print(f"{name:10s} n={r['n']:3d} accuracy={r['accuracy']:.3f} ±{r['se']:.3f}   (50% = indistinguishable)")
    print(f"calls {'planned' if a.dry_run else 'made'}: {J.calls}")
    if not a.dry_run:
        with open(os.path.join(a.out, f"p1_judge_seed{a.seed}{a.tag}.json"), "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=1)

if __name__ == "__main__": main()
