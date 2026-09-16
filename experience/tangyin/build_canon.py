#!/usr/bin/env python3
"""Build the CANON reference for the borrowed-line check (理 10866).

    python3 build_canon.py --repo <clone of chinese-poetry> --out experience/tangyin/canon_7char.json

理 named github.com/chinese-poetry/chinese-poetry (MIT, a git clone — not a scrape, ctext stays
closed). This flattens it to the same key the training-split check uses: exact 7-character lines.

TWO REFERENCES, REPORTED SEPARATELY, because they are different claims:
  TRAINING SPLIT -> REGURGITATION. The model reproducing text we fine-tuned it on.
  CANON          -> BORROWING.     The model reproducing famous verse it knew before we touched it.

WHAT THIS CANON CANNOT SEE, stated here so the column is read correctly:
  * MING POETS ARE NOT IN IT. The repo is 唐 / 宋 / 元 / 五代 / 詩經 / 楚辭. 唐寅 (1470-1524) and his
    circle are absent by construction, so a Ming borrowing scores as ORIGINAL here. 理's second
    pass — one judge call per surviving poem — is the only canon that holds Ming.
  * PARAPHRASE AND REORDERING. Whole-line keys miss both, as they do for the training split.
  * The directory named 全唐诗 contains poet.song.* files as well as poet.tang.*; the filenames,
    not the directory, say which dynasty a file holds. Counted by FILE PREFIX for that reason.
"""
import argparse, glob, json, os, re

CJK = re.compile(r"[㐀-鿿]")


def lines7(text):
    s = "".join(CJK.findall(text or ""))
    return [s[i:i + 7] for i in range(0, len(s), 7) if len(s[i:i + 7]) == 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lines, by_source, files = {}, {}, 0
    for f in sorted(glob.glob(os.path.join(a.repo, "**", "*.json"), recursive=True)):
        base = os.path.basename(f)
        # the corpora that are POEMS with a body we can key on
        if not (base.startswith(("poet.", "ci.")) or base in ("qianjiashi.json", "shijing.json")):
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        files += 1
        src = "tang" if base.startswith("poet.tang") else \
              "song_shi" if base.startswith("poet.song") else \
              "song_ci" if base.startswith("ci.") else "other"
        for rec in d:
            if not isinstance(rec, dict):
                continue
            body = rec.get("paragraphs") or rec.get("para") or rec.get("content") or []
            if isinstance(body, str):
                body = [body]
            title = rec.get("title") or rec.get("rhythmic") or ""
            author = rec.get("author") or ""
            for ln in lines7("".join(body)):
                if ln not in lines:
                    lines[ln] = {"author": author, "title": title, "source": src}
                    by_source[src] = by_source.get(src, 0) + 1

    out = {"reference": "CANON — borrowing (distinct from the TRAINING-SPLIT reference, which is "
                        "regurgitation)",
           "source": "github.com/chinese-poetry/chinese-poetry (MIT, git clone)",
           "files_read": files, "distinct_7char_lines": len(lines), "by_source": by_source,
           "cannot_see": ["MING poets — 唐寅's own era is absent by construction; a Ming borrowing "
                          "scores ORIGINAL here and only 理's judge pass can catch it",
                          "paraphrase and reordered couplets — whole-line keys, same floor as the "
                          "training-split check"],
           "lines": lines}
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"  files read: {files}   distinct 7-char lines: {len(lines):,}")
    print(f"  by source: {by_source}")
    print(f"  -> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
