#!/usr/bin/env python3
"""Fetch a 四庫全書本 collected-works edition from zh.wikisource into the corpus JSONL schema.

    python3 fetch_skqs.py --work "甫田集 (四庫全書本)" --author 文徵明 \
        --juan 1-15 --out corpus/wenzhengming_skqs.jsonl

WHY THIS EXISTS, beside fetch_corpus.py
---------------------------------------
`fetch_corpus.py --author 文徵明` returns ONE poem block, and it is a 滿江紅 — a 詞, not a
七絕/七律. `Category:文徵明` on zh.wikisource holds exactly three pages. The control poet is
not missing from Wikisource; he is in his COLLECTED WORKS, `甫田集 (四庫全書本)`, 卷01–卷35,
of which 卷01–15 are the 詩 (the 提要 says 詩十五巻文二十巻). 卷二 alone announces 詩七十一首.

The 四庫 text is shaped differently from a Category page and needs its own parser:

  * ONE POEM PER LINE, UNPUNCTUATED. A 七絕 is a single 28-character line, a 七律 56.
    fetch_corpus.py's 14-char-couplet assumption does not hold here, so this writer emits
    BOTH: `text_raw` (the line as the edition has it) and `text` (segmented into couplets
    ONLY when the length matches a canonical form, so it lines up with the 唐寅 corpus).
  * TITLES ARE TEMPLATES: `{{SK anchor|和答石田先生落花十首}}` heads the poems under it.
  * RARE GLYPHS ARE TEMPLATES: `{{SKchar|1866}}` stands in for a character the edition has
    and Unicode does not. Each becomes U+25A1 '□' and is COUNTED in `skchar`.
    A poem with skchar > 0 has a character the form verifier cannot tone-check — it is kept,
    flagged, and it is the verifier's business whether to score it.

Nothing here is punctuated, so rhyme and 平仄 work must read the raw line. That is the point:
the 四庫 text is the closest thing to what the poet's readers saw.
"""
import argparse, json, re, sys, time, urllib.parse, urllib.request

API = "https://zh.wikisource.org/w/api.php"
UA = "efficient-thinking/et8-tangyin (research; contact via repo)"

# total chars -> (chars per hemistich, hemistichs) for the regulated forms this study scores
FORMS = {20: ("五絕", 5, 4), 28: ("七絕", 7, 4), 40: ("五律", 5, 8), 56: ("七律", 7, 8)}


def get(params, tries=6):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(5 * (k + 1))          # Wikisource rate limit: back off, never hammer


def raw(title):
    d = get({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
             "titles": title, "format": "json", "formatversion": 2})
    p = d["query"]["pages"][0]
    return p.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")


def clean(line):
    """Strip the edition's templates. Returns (text, n_skchar)."""
    n = len(re.findall(r"\{\{SKchar\|", line))
    line = re.sub(r"\{\{SKchar\|[^}]*\}\}", "□", line)
    line = re.sub(r"\{\{SK notes\|([^}]*)\}\}", r"\1", line)
    line = re.sub(r"\{\{[^}]*\}\}", "", line)                  # YL, anchors, everything else
    line = re.sub(r"<[^>]+>", "", line)
    line = line.replace("　", "").replace(" ", "").strip()
    return line, n


def segment(text):
    """Canonical form -> couplet lines like the 唐寅 corpus; otherwise None."""
    f = FORMS.get(len(text))
    if not f:
        return None, None
    name, per, n_hemi = f
    hemi = [text[i * per:(i + 1) * per] for i in range(n_hemi)]
    return name, "\n".join(hemi[i] + "，" + hemi[i + 1] + "。" for i in range(0, n_hemi, 2))


def juan_range(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help='e.g. "甫田集 (四庫全書本)"')
    ap.add_argument("--author", required=True)
    ap.add_argument("--juan", required=True, help="e.g. 1-15")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()

    rows, pages, skipped_untitled = [], 0, 0
    for j in juan_range(a.juan):
        title = f"{a.work}/卷{j:02d}"
        body = raw(title)
        if not body:
            print(f"  {title}: EMPTY", file=sys.stderr); continue
        pages += 1
        blocks = re.findall(r"<poem>(.*?)</poem>", body, re.S)
        cur_title = None
        for blk in blocks:
            for line in blk.split("\n"):
                anchor = re.search(r"\{\{SK anchor\|([^}]*)\}\}", line)
                txt, nsk = clean(line)
                if anchor:
                    # An anchor is a heading, but NOT every heading is a poem title: each juan
                    # opens with its own name (甫田集巻一) and a section counter (詩七十一首).
                    # Carrying those down onto the poems under them invents 27 titles that the
                    # edition does not give, so they clear the title instead of setting it.
                    head = anchor.group(1)
                    cur_title = None if re.match(r"^甫田集巻|^詩[一二三四五六七八九十百]+首$", head) else head
                    if not txt or txt == head:
                        continue
                if not txt or not re.fullmatch(r"[㐀-鿿□]+", txt):
                    continue                     # 欽定四庫全書, 明　文徵明　撰, prose, etc.
                if len(txt) < 20:
                    continue
                form, seg = segment(txt)
                if cur_title is None:
                    skipped_untitled += 1
                rows.append({
                    "title": cur_title or "(untitled)",
                    "author": a.author,
                    "source": f"zh.wikisource:{title}",
                    "text": seg or txt,
                    "text_raw": txt,
                    "form_guess": form,
                    "n_lines": (seg.count("\n") + 1) if seg else 1,
                    "chars_per_line": [len(txt)] if not seg else
                                      [len(x) for x in seg.split("\n")],
                    "n_chars": len(txt),
                    "skchar": nsk,
                })
        time.sleep(a.sleep)

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    forms = {}
    for r in rows:
        forms[r["form_guess"]] = forms.get(r["form_guess"], 0) + 1
    print(f"{a.author} {a.work}: {pages} juan -> {len(rows)} poem lines -> {a.out}")
    print(f"  forms: {forms}")
    print(f"  with a substituted glyph (skchar>0): {sum(1 for r in rows if r['skchar'])}")
    print(f"  emitted before any title was seen: {skipped_untitled}")


main()
