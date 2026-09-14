#!/usr/bin/env python3
"""The (b) re-cut: a seeded split of the 103 Wikisource blocks BY POEM.

    python3 make_split.py --seed 7 --train 70 --out corpus/

理's ruling, 2026-09-14: ctext is closed, so the held-out set is no longer a different EDITION.
It is a random split of one corpus, and the paper says so. P4's memorization probes still run on
the held-out side — memorization is a property of Claude's weights, not of our source.

WHY BY POEM, AND WHAT THAT STILL DOES NOT FIX
---------------------------------------------
A block is one poem, so a poem cannot straddle the split. But `題畫 (唐寅)#3` and `#6` are
DIFFERENT POEMS FROM ONE PAGE, and two Wikisource pages can carry the same folk poem — 開門七件事
and 除夕口占 are one 打油詩 under two titles. So this splits by poem and additionally keeps every
block sharing a SOURCE PAGE on the same side, and reports the duplicate-text pairs it finds by
normalised content rather than by title. Leakage that survives both is named in the manifest.
"""
import argparse, hashlib, json, os, random, re, sys
from collections import defaultdict

CJK = re.compile(r"[㐀-鿿]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "corpus", "tangyin_wikisource.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", type=int, default=70)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))
    ap.add_argument("--near", type=float, default=0.5,
                    help="character-bigram Jaccard at or above which two poems are treated as "
                         "one text and kept on the same side of the split")
    a = ap.parse_args()

    body = open(a.corpus, "rb").read()
    rows = [json.loads(l) for l in body.decode().splitlines() if l.strip()]
    corpus_sha = hashlib.sha256(body).hexdigest()

    # group by SOURCE PAGE: several blocks can come from one page and must not straddle
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["source"]].append(i)

    # Duplicate TEXT across pages, found by content, not by title — and compared on a
    # NORMALISED key. 開門七件事 and 除夕口占 are one 打油詩, but one page is traditional and the
    # other simplified (柴米油鹽醬醋茶 / 柴米油盐酱醋茶), so a raw comparison finds NOTHING and
    # the pair lands on both sides of the split. Everything is folded to simplified for the
    # key only; the stored text is untouched.
    try:
        from opencc import OpenCC
        fold = OpenCC("t2s").convert
    except ImportError:
        fold = lambda x: x
        print("  ⚠ opencc ABSENT — duplicate detection is RAW and will miss "
              "traditional/simplified pairs of the same poem.", file=sys.stderr)
    bytext = defaultdict(list)
    for i, r in enumerate(rows):
        bytext[fold("".join(CJK.findall(r["text"])))].append(i)
    dups = {k: v for k, v in bytext.items() if len(v) > 1}

    # EXACT matching is not enough, and the pair 理 named proves it. 開門七件事 and 除夕口占 are
    # one 打油詩, but the two pages carry VARIANT TEXT, not just variant orthography:
    #   歲幕天寒無一事 / 竹時寺裏看梅花   vs   岁暮清淡无一事 / 竹堂寺裏看梅花
    # Folding to simplified does not make them equal. So a NEAR-duplicate pass runs too, on a
    # character-bigram Jaccard over the folded text. The threshold is a judgement; every pair
    # above it is NAMED in the manifest with its score, so the judgement is auditable rather
    # than buried, and a pair just under it is visible in the near-miss list.
    def bigrams(t):
        return {t[i:i + 2] for i in range(len(t) - 1)}
    folded = [fold("".join(CJK.findall(r["text"]))) for r in rows]
    grams = [bigrams(t) for t in folded]
    near, nearmiss = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if not grams[i] or not grams[j]:
                continue
            jac = len(grams[i] & grams[j]) / len(grams[i] | grams[j])
            if jac >= a.near:
                near.append((round(jac, 3), i, j))
            elif jac >= a.near - 0.2:
                nearmiss.append((round(jac, 3), rows[i]["title"], rows[j]["title"]))
    near.sort(reverse=True)
    nearmiss.sort(reverse=True)

    # merge groups that share text — union-find over source pages
    parent = {s: s for s in groups}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for idxs in dups.values():
        srcs = [rows[i]["source"] for i in idxs]
        for s_ in srcs[1:]:
            union(srcs[0], s_)
    for _, i, j in near:
        union(rows[i]["source"], rows[j]["source"])

    clusters = defaultdict(list)
    for s, idxs in groups.items():
        clusters[find(s)] += idxs

    keys = sorted(clusters)
    rnd = random.Random(a.seed)
    rnd.shuffle(keys)
    # Fill to the target WITHOUT overshooting: a cluster that would push past it goes to the
    # held-out side instead. Filling while `len(train) < target` overshot to 85/18, because the
    # last cluster added carried several blocks.
    train, held = [], []
    for k in keys:
        c = clusters[k]
        if len(train) + len(c) <= a.train:
            train.extend(c)
        else:
            held.extend(c)

    os.makedirs(a.out, exist_ok=True)
    for name, idxs in (("tangyin_train", train), ("tangyin_heldout", held)):
        p = os.path.join(a.out, name + ".jsonl")
        with open(p, "w") as f:
            for i in sorted(idxs):
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        print(f"  {name}: {len(idxs)} poems -> {p}")

    man = {
        "ruling": "理 2026-09-14 — option (b): a seeded random split of the 103 Wikisource blocks.",
        "what_it_stopped_meaning": "The held-out side is NO LONGER a different edition. It is the "
                                   "same source, split. P4's probes still apply: memorization is a "
                                   "property of Claude's weights, not of our source.",
        "corpus": os.path.basename(a.corpus), "corpus_sha256": corpus_sha,
        "seed": a.seed, "n_total": len(rows), "n_train": len(train), "n_heldout": len(held),
        "split_unit": "SOURCE PAGE, then merged across pages that share identical CJK text",
        "pages": len(groups), "clusters_after_merge": len(clusters),
        "exact_duplicate_groups": [
            {"chars": len(t), "rows": [rows[i]["title"] for i in v]} for t, v in dups.items()],
        "near_duplicate_threshold_bigram_jaccard": a.near,
        "near_duplicate_pairs_MERGED": [
            {"jaccard": s_, "a": rows[i]["title"], "b": rows[j]["title"]} for s_, i, j in near],
        "near_misses_just_below_threshold": [
            {"jaccard": s_, "a": x, "b": y} for s_, x, y in nearmiss[:15]],
        "residual_leakage_risk": "Two poems by one author on one topic are not duplicates and are "
                                 "not detected here. This removes EXACT repeats and same-page "
                                 "siblings; it does not remove similarity.",
    }
    p = os.path.join(a.out, "split_manifest.json")
    json.dump(man, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"  exact duplicate groups: {len(dups)}")
    print(f"  near-duplicate pairs merged (jaccard >= {a.near}): {len(near)}")
    for s_, i, j in near[:8]:
        print(f"     {s_}  {rows[i]['title']}  <->  {rows[j]['title']}")
    print(f"  near misses just below: {[(s_, x, y) for s_, x, y in nearmiss[:4]]}")
    print(f"  pages {len(groups)} -> clusters {len(clusters)}  -> {p}")


if __name__ == "__main__":
    main()
