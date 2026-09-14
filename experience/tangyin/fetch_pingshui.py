#!/usr/bin/env python3
"""Build the 平水韻 table — the instrument `verify_form.py` measures rhyme and 平仄 with.

    python3 fetch_pingshui.py --out rime/pingshui.json

SOURCE: zh.wikisource `平水韻` — 劉淵 / 王文鬱's 106-rhyme table, the one Ming poets wrote to.
中華新韻 is NOT acceptable for this study and is not what this fetches.

The page is five tone sections (上平聲部 下平聲部 上聲部 去聲部 入聲部) holding 106 rhyme
groups, each `上平聲一{{++|東}}` followed by its characters. A trailing `【詞】` list on a rhyme
is characters admitted for 詞 rhyming but NOT for 詩 — they are kept, flagged `shi: false`,
and a shi verifier must not rhyme on them.

WHAT THIS TABLE CANNOT DO, and the number that says so
------------------------------------------------------
A character can sit in several rhymes and in several TONES (破音字). For those, 平仄 is a
property of the READING, not of the glyph, and nothing in a character table can pick the
reading — only the sense in the line can. The build prints `ambiguous_tone`: characters that
are BOTH 平 and 仄. Every 平仄 pass rate computed from this table carries that count as its
bound, and `verify_form.py` must report tone-ambiguous positions rather than silently choosing.
"""
import argparse, json, re, time, urllib.error, urllib.parse, urllib.request

API = "https://zh.wikisource.org/w/api.php"
UA = "efficient-thinking/et8-tangyin (research; contact via repo)"
TONE = {"上平聲": "平", "下平聲": "平", "上聲": "仄", "去聲": "仄", "入聲": "仄"}
TONE_NAME = {"上平聲": "上平", "下平聲": "下平", "上聲": "上", "去聲": "去", "入聲": "入"}


def raw(title, tries=6):
    q = urllib.parse.urlencode({"action": "query", "prop": "revisions", "rvprop": "content",
                                "rvslots": "main", "titles": title, "format": "json",
                                "formatversion": 2})
    for k in range(tries):
        try:
            req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            p = d["query"]["pages"][0]
            return p.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(10 * (k + 1))          # Wikisource 429s under load; back off, never hammer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    page = raw("平水韻")
    # Each rhyme: 上平聲一{{++|東}} then its characters up to the next heading or section.
    pat = re.compile(r"([上下]平聲|上聲|去聲|入聲)([一二三四五六七八九十]+)\{\{\+\+\|(.)\}\}")
    hits = list(pat.finditer(page))
    if len(hits) != 106:
        raise SystemExit(f"expected 106 rhyme groups, parsed {len(hits)} — the page changed shape")

    rhymes, char_index = [], {}
    for i, m in enumerate(hits):
        sec, num, name = m.group(1), m.group(2), m.group(3)
        end = hits[i + 1].start() if i + 1 < len(hits) else len(page)
        body = page[m.end():end]
        body = re.sub(r"==[^=\n]+==", "", body)          # the section header between groups
        shi, ci = body, ""
        if "【詞】" in body:
            shi, ci = body.split("【詞】", 1)
        shi = "".join(re.findall(r"[㐀-鿿]", shi))
        ci = "".join(re.findall(r"[㐀-鿿]", ci))
        label = f"{TONE_NAME[sec]}{num}{name}"
        rhymes.append({"tone": TONE[sec], "tone_name": TONE_NAME[sec], "num": num,
                       "name": name, "label": label, "chars": shi, "ci_only": ci})
        for ch in shi:
            char_index.setdefault(ch, []).append({"tone": TONE[sec], "rhyme": label, "shi": True})
        for ch in ci:
            char_index.setdefault(ch, []).append({"tone": TONE[sec], "rhyme": label, "shi": False})

    ambiguous = sorted(c for c, e in char_index.items()
                       if len({x["tone"] for x in e}) > 1)
    out = {
        "source": "zh.wikisource:平水韻 (劉淵 南宋 / 王文鬱 金, 106韻)",
        "not": "中華新韻 — deliberately not used; Ming poets wrote to 平水韻",
        "rhyme_groups": len(rhymes),
        "distinct_chars": len(char_index),
        "ambiguous_tone_chars": len(ambiguous),
        "ambiguous_tone_sample": ambiguous[:40],
        "rhymes": rhymes,
        "char_index": char_index,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"平水韻: {len(rhymes)} rhyme groups · {len(char_index)} distinct characters -> {a.out}")
    print(f"  BOTH 平 and 仄 (tone cannot be decided from the glyph): {len(ambiguous)}"
          f"  ({100.0*len(ambiguous)/len(char_index):.1f}%)")
    print(f"  characters admitted for 詞 only: "
          f"{sum(1 for e in char_index.values() if all(not x['shi'] for x in e))}")


if __name__ == "__main__":
    main()
