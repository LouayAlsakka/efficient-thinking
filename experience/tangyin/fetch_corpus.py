#!/usr/bin/env python3
"""Fetch a public-domain poet corpus from zh.wikisource into JSONL.

    python3 fetch_corpus.py --author 唐寅 --out corpus/tangyin.jsonl
    python3 fetch_corpus.py --author 文徵明 --out corpus/wenzhengming.jsonl   # same-circle control

Each row: {"title", "author", "source", "text" (lines joined by \\n), "n_lines", "chars_per_line"}.
Only <poem>…</poem> blocks are kept; wikitext templates are stripped. Nothing is fetched twice.
"""
import argparse, json, re, sys, time, urllib.parse, urllib.request

API = "https://zh.wikisource.org/w/api.php"
UA = "efficient-thinking/et8-tangyin (research; contact via repo)"

def get(params, tries=6):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1: raise
            time.sleep(5 * (k + 1))          # Wikisource rate limit: back off, never hammer

def category_members(author):
    out, cont = [], {}
    while True:
        d = get({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{author}",
                 "cmlimit": 500, "format": "json", **cont})
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d: return out
        cont = d["continue"]

def raw(title):
    d = get({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
             "titles": title, "format": "json", "formatversion": 2})
    p = d["query"]["pages"][0]
    return p.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")

POEM = re.compile(r"<poem>(.*?)</poem>", re.S)

def clean(block):
    lines = []
    for ln in block.splitlines():
        ln = re.sub(r"\{\{.*?\}\}|<[^>]+>|\[\[|\]\]", "", ln).strip()
        if ln and not ln.startswith(("=", "|", "'")):
            lines.append(ln)
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()
    titles = [t for t in category_members(a.author) if not t.startswith(("Author:", "Category:"))]
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for t in titles:
            txt = raw(t); time.sleep(a.sleep)
            for i, block in enumerate(POEM.findall(txt)):
                lines = clean(block)
                if len(lines) < 2: continue
                body = lines[1:] if lines[0].replace(" ", "") in t.replace(" ", "") else lines
                row = {"title": t if i == 0 else f"{t}#{i}", "author": a.author, "source": f"zh.wikisource:{t}",
                       "text": "\n".join(body), "n_lines": len(body),
                       "chars_per_line": sorted({len(re.sub(r"[，。？！、；：]", "", l)) for l in body})}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(f"{a.author}: {len(titles)} pages -> {n} poem blocks -> {a.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
