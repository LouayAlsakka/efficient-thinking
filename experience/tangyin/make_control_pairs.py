#!/usr/bin/env python3
"""Item 3: the control-poet discrimination pairs. PAIRS FILE ONLY — this makes no API call.

    python3 make_control_pairs.py --out results/control_pairs_wenzhengming.jsonl

judge.py asks "which of these two 七絕 is the real 唐寅?" If Claude answers that as well with
文徵明 in the real slot, then it is telling a real Ming 七絕 from a generated one — not telling
唐寅 from anything. The control is what stops a high discrimination number being read as
"the voice did not transfer" when it is really "any real poem is recognisable".

Presentation is judge.py's, exactly: punctuation stripped, 28 characters, chunked 4x7, order
randomised by the same seed discipline. The pairs carry their truth label so the judge only
has to ask and score.

THE CONFOUND I CANNOT REMOVE, ONLY MEASURE
------------------------------------------
唐寅's held-out set is punctuated Wikisource text. 文徵明's is the 四庫全書本, whose orthography
is archaic — 隂 for 陰, 巻 for 卷, 逺 for 遠. If the control poems carry glyphs the 唐寅 poems
never do, the judge can discriminate on TYPEFACE rather than on poetry, and the control would
read as "easy" for a reason that has nothing to do with voice. The script counts those
characters per poem and ships the count in every pair.
"""
import argparse, json, os, random, re

PUNCT = re.compile(r"[^㐀-鿿]")
HERE = os.path.dirname(os.path.abspath(__file__))


def chunk(s, per): return [s[i:i + per] for i in range(0, len(s), per)]


def qijue(path, field, want_gutai=False, rime=None):
    """七絕-length poems, judge.py's normalisation, optionally excluding 古體."""
    import sys
    sys.path.insert(0, HERE)
    from verify_form import Rime, verify
    r = rime or Rime(os.path.join(HERE, "rime", "pingshui.json"))
    out = []
    for row in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()):
        s = PUNCT.sub("", row.get(field) or row.get("text") or "")
        if len(s) != 28:
            continue
        v = verify(s, r)
        if v.get("classification") == "古體(仄韻)" and not want_gutai:
            continue                      # 理 ruled 古體 leaves the pool; a control must obey it
        out.append({"title": row.get("title"), "chars": s, "lines": chunk(s, 7),
                    "source": row.get("source")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default=os.path.join(HERE, "corpus", "wenzhengming_skqs.jsonl"))
    ap.add_argument("--control-field", default="text")
    ap.add_argument("--heldout", default=os.path.join(HERE, "corpus", "tangyin_heldout.jsonl"))
    ap.add_argument("--arm", action="append", default=[], help="name=generations.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "control_pairs_wenzhengming.jsonl"))
    a = ap.parse_args()

    import sys
    sys.path.insert(0, HERE)
    from verify_form import Rime
    rime = Rime(os.path.join(HERE, "rime", "pingshui.json"))

    control = qijue(a.control, a.control_field, rime=rime)
    tangyin = qijue(a.heldout, "text", rime=rime)

    # the orthography confound, measured on both sides
    ty_chars = set("".join(p["chars"] for p in tangyin))
    def archaic(p):  # characters this poem uses that the 唐寅 held-out set never uses
        return sorted(set(p["chars"]) - ty_chars)

    rng = random.Random(a.seed)
    rows = []
    for name_path in a.arm:
        name, path = name_path.split("=", 1)
        gens = [{"topic": g["topic"], "chars": g["poem"], "lines": chunk(g["poem"], 7)}
                for g in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())
                if g.get("form") == "七絕" and g.get("bucket") == "EXTRACTED"
                and len(g.get("poem", "")) == 28]
        for g in gens:
            for real_name, pool in (("唐寅", tangyin), ("文徵明", control)):
                real = rng.choice(pool)
                real_first = rng.random() < 0.5
                A, B = (real, g) if real_first else (g, real)
                rows.append({
                    "arm": name, "real_poet": real_name, "real_title": real["title"],
                    "topic": g["topic"], "truth": "A" if real_first else "B",
                    "A": A["lines"], "B": B["lines"],
                    "real_archaic_chars": archaic(real),
                    "n_archaic": len(archaic(real)),
                })
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ctl = [r for r in rows if r["real_poet"] == "文徵明"]
    ty = [r for r in rows if r["real_poet"] == "唐寅"]
    print(f"  control pool (文徵明 七絕, 古體 excluded): {len(control)}")
    print(f"  唐寅 held-out 七絕:                        {len(tangyin)}")
    print(f"  pairs written: {len(rows)}   ({len(ty)} with 唐寅 real, {len(ctl)} with 文徵明 real)")
    print(f"  -> {a.out}")
    print()
    print("  ORTHOGRAPHY CONFOUND — characters the real poem uses that the 唐寅 held-out set never does")
    for label, grp in (("唐寅 real", ty), ("文徵明 real", ctl)):
        if not grp: continue
        n = [r["n_archaic"] for r in grp]
        print(f"    {label:12s} mean {sum(n)/len(n):4.1f} per poem · "
              f"share with >=1: {100.0*sum(1 for x in n if x)/len(n):5.1f}%")


if __name__ == "__main__":
    main()
