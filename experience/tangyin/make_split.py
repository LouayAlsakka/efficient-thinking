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
    a = ap.parse_args()

    body = open(a.corpus, "rb").read()
    rows = [json.loads(l) for l in body.decode().splitlines() if l.strip()]
    corpus_sha = hashlib.sha256(body).hexdigest()

    # group by SOURCE PAGE: several blocks can come from one page and must not straddle
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["source"]].append(i)

    # duplicate TEXT across pages, found by content, not by title
    bytext = defaultdict(list)
    for i, r in enumerate(rows):
        bytext["".join(CJK.findall(r["text"]))].append(i)
    dups = {k: v for k, v in bytext.items() if len(v) > 1}

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
        for s in srcs[1:]:
            union(srcs[0], s)

    clusters = defaultdict(list)
    for s, idxs in groups.items():
        clusters[find(s)] += idxs

    keys = sorted(clusters)
    rnd = random.Random(a.seed)
    rnd.shuffle(keys)
    train, held = [], []
    for k in keys:
        (train if len(train) < a.train else held).extend(clusters[k])

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
        "duplicate_text_groups_found": [
            {"chars": len(t), "rows": [rows[i]["title"] for i in v]} for t, v in dups.items()],
        "residual_leakage_risk": "Two poems by one author on one topic are not duplicates and are "
                                 "not detected here. This removes EXACT repeats and same-page "
                                 "siblings; it does not remove similarity.",
    }
    p = os.path.join(a.out, "split_manifest.json")
    json.dump(man, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"  duplicate-text groups: {len(dups)}  {[[rows[i]['title'] for i in v] for v in dups.values()]}")
    print(f"  pages {len(groups)} -> clusters {len(clusters)}  -> {p}")


if __name__ == "__main__":
    main()
