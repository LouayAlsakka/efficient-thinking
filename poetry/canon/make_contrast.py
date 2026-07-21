#!/usr/bin/env python
"""ET-IV item 7a — canon contrast dataset (public-domain only, provenance logged).

Positives: public-domain canonical verse (pre-1928; poetry/data/canon_reference.txt), chunked into
quatrains. Graded negatives, by discrimination tier:
  - easy (canon vs corrupted): the same quatrain line-shuffled or meter-broken (checker 1a);
  - hard (canon vs model-generated): sonnet/quatrain outputs from the E1 ladder caches.
The mid tier (canon vs amateur human verse) needs a PD/CC amateur-verse source we do not yet have; it
is left as a logged gap rather than proxied, so no non-distributable text enters the set.
Output: poetry/canon/contrast.jsonl — pairwise preference records {chosen(canon), rejected, tier,
provenance}. Nothing here is model-graded; this is the training/eval substrate for 7b.

  python poetry/canon/make_contrast.py
"""
import glob, json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkers"))
from meter_rhyme import check_iambic_pentameter


def load_canon_quatrains(path="poetry/data/canon_reference.txt"):
    lines = [l.strip() for l in open(path) if l.strip()]
    return [lines[i:i + 4] for i in range(0, len(lines) - 3, 4)]


def corrupt(quatrain, rng):
    """Easy negative: shuffle the lines (breaks rhyme/sense) or drop a word from one line (breaks meter)."""
    q = list(quatrain)
    if rng.random() < 0.5:
        rng.shuffle(q)
        return q, "line-shuffled"
    i = rng.randrange(len(q)); words = q[i].split()
    if len(words) > 3:
        del words[rng.randrange(len(words))]; q[i] = " ".join(words)
    return q, "meter-broken"


def model_quatrains(n, rng):
    """Hard negatives: 4-line outputs from the E1 sonnet caches (model-generated verse, provenance kept)."""
    out = []
    for f in sorted(glob.glob("poetry/cache/e1_*.jsonl")):
        if f.endswith("_greedy.jsonl"):
            continue
        tag = os.path.basename(f)[3:-6]
        for r in (json.loads(l) for l in open(f)):
            if r.get("task") != "sonnet":
                continue
            for s in r["samples"][:2]:
                ls = [x.strip() for x in s.splitlines() if x.strip() and not x.strip().startswith("(")][:4]
                if len(ls) == 4:
                    out.append((ls, f"model:{tag}"))
    rng.shuffle(out)
    return out[:n]


def main():
    rng = random.Random(0)
    os.makedirs("poetry/canon", exist_ok=True)
    canon = load_canon_quatrains()
    models = model_quatrains(len(canon) * 2, rng)
    rows, mi = [], 0
    for qi, q in enumerate(canon):
        cq, how = corrupt(q, rng)
        rows.append({"chosen": q, "rejected": cq, "tier": "easy_corrupted",
                     "provenance": {"chosen": "PD canon (canon_reference.txt)", "rejected": f"corrupted:{how}"}})
        if mi < len(models):
            mq, prov = models[mi]; mi += 1
            rows.append({"chosen": q, "rejected": mq, "tier": "hard_model",
                         "provenance": {"chosen": "PD canon (canon_reference.txt)", "rejected": prov}})
    rng.shuffle(rows)
    with open("poetry/canon/contrast.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    tiers = {}
    for r in rows:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    print(f"[7a] wrote poetry/canon/contrast.jsonl: {len(rows)} pairwise records {tiers}")
    print(f"[7a] canon quatrains={len(canon)}; mid tier (PD/CC amateur) LOGGED AS GAP — no non-PD text used.")


if __name__ == "__main__":
    main()
