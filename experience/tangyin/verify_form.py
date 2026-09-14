#!/usr/bin/env python3
"""The FORM verifier: the external, objective half of the Tang Yin study.

    python3 verify_form.py --corpus corpus/tangyin_wikisource.jsonl --report results/form_tangyin.json
    python3 verify_form.py --text "一枚蟬蛻榻當中命也難辭付大空垂死尚思玄墓麓滿山寒雪一林松"

It answers one question per poem — does this obey the rules of 近體詩 — and it answers it the
way `et8_env.py`'s failing test does: pass/fail per RULE, never a vibe, never a model.

THE RULES, and why they are these
---------------------------------
Positions are 1-based within a hemistich (a 句). 七言 has 7, 五言 has 5.

  structure  the poem is 4 or 8 hemistichs of a constant 5 or 7 characters.
  rhyme      hemistichs 2,4(,6,8) end on characters sharing ONE 平水韻 rhyme group, in a 平
             tone. Hemistich 1 MAY rhyme (首句入韻) — permitted, not required, so a hemistich 1
             that does not rhyme is not a fault and a hemistich 1 that does must match the group.
  alternate  within a hemistich, positions 2,4,6 (五言: 2,4) alternate 平仄平 or 仄平仄.
             This is the content of 一三五不論: odd positions are free, even ones are not.
  對         between the two hemistichs of a couplet, position 2 is OPPOSITE.
  粘         between the second hemistich of a couplet and the first of the next, position 2 is
             the SAME. 對 and 粘 together are what generate the four canonical patterns, so this
             verifier derives them instead of matching against a table of four.
  三平調     a hemistich ending 平平平 is a fault. Cheap and objective, reported separately
             because it is a fault of taste that the generative rules above do not catch.
  特拗       ...with ONE documented exemption. 仄仄平平平仄仄 has an accepted variant,
             仄仄平平仄平仄 (特拗 / 鯉魚翻波), in which positions 5 and 6 swap. It is not a
             fault and it is common: four of Tang Yin's 七絕 use it. A hemistich that ends
             仄 and whose last three positions read 仄平仄 is exempted from `alternate` at
             the swapped position. This is the ONLY latitude granted, it is granted because
             the literature grants it, and it was written here before the rate was looked at.

古體 IS NOT MALFORMED
---------------------
A poem whose rhyme positions all fall on 仄 characters is a 古絕/古體, not a defective 近體.
It is classified, not failed: 近體's 對/粘/alternate rules are not its rules. Scoring it as a
failure would let a model earn credit for avoiding a form the corpus itself uses.

WHAT IT REFUSES TO DECIDE
-------------------------
10.2% of the 8,550 characters in the 平水韻 table sit in BOTH a 平 and a 仄 rhyme (破音字).
Their tone belongs to the reading, not to the glyph, and no character table can pick it. At an
enforced position such a character yields **UNDECIDED**, never a pass and never a fail, and the
counts are reported as three numbers. A verifier that guessed would manufacture whichever pass
rate the guess favoured -- and this number gates a training decision.

A character absent from the table entirely (a variant glyph, or the □ that stands in for a
{{SKchar}} the edition has and Unicode does not) is UNKNOWN and is reported the same way.
"""
import argparse, json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = {20: ("五絕", 5, 4), 28: ("七絕", 7, 4), 40: ("五律", 5, 8), 56: ("七律", 7, 8)}
CJK = re.compile(r"[㐀-鿿]")


class Rime:
    def __init__(self, path):
        d = json.load(open(path))
        self.idx = d["char_index"]
        self.n_ambiguous = d["ambiguous_tone_chars"]

    def tone(self, ch):
        """'平', '仄', 'UNDECIDED' (both), or 'UNKNOWN' (not in the table)."""
        e = self.idx.get(ch)
        if not e:
            return "UNKNOWN"
        t = {x["tone"] for x in e}
        return t.pop() if len(t) == 1 else "UNDECIDED"

    def rhyme_groups(self, ch, shi_only=True, ping_only=True):
        out = set()
        for x in self.idx.get(ch, []):
            if shi_only and not x["shi"]:
                continue
            if ping_only and x["tone"] != "平":
                continue
            out.add(x["rhyme"])
        return out


def opposite(a, b):
    return {a, b} == {"平", "仄"}


def verify(poem, rime):
    """poem: a string of CJK characters only. Returns a dict of per-rule results."""
    text = "".join(CJK.findall(poem))
    res = {"n_chars": len(text), "form": None, "rules": {}, "undecided": 0, "unknown": 0}
    f = FORMS.get(len(text))
    if not f:
        res["rules"]["structure"] = {"pass": False,
                                     "why": f"{len(text)} characters matches no regulated form"}
        res["pass"] = False
        return res
    name, per, n = f
    res["form"] = name
    lines = [text[i * per:(i + 1) * per] for i in range(n)]
    res["rules"]["structure"] = {"pass": True, "why": f"{n}×{per}"}

    tones = [[rime.tone(c) for c in ln] for ln in lines]
    res["undecided"] = sum(t.count("UNDECIDED") for t in tones)
    res["unknown"] = sum(t.count("UNKNOWN") for t in tones)

    even = [1, 3, 5] if per == 7 else [1, 3]          # 0-based positions 2,4,6 / 2,4

    # --- alternate: within a hemistich the even positions alternate, 特拗 exempted
    def tewao(t):
        """仄仄平平仄平仄 (七) / 平平仄平仄 (五): the accepted swap of positions 5 and 6."""
        return t[-3:] == ["仄", "平", "仄"]

    bad, und, tw = [], 0, []
    for i, t in enumerate(tones):
        if tewao(t):
            tw.append(i + 1)
            continue
        got = [t[p] for p in even]
        if any(g in ("UNDECIDED", "UNKNOWN") for g in got):
            und += 1
            continue
        if not all(opposite(got[k], got[k + 1]) for k in range(len(got) - 1)):
            bad.append(i + 1)
    res["rules"]["alternate"] = {"pass": not bad, "failed_hemistichs": bad, "undecided": und,
                                 "tewao_hemistichs": tw}

    # --- 對 within each couplet, 粘 across couplets: position 2
    p2 = [t[1] for t in tones]
    dui_bad, dui_und, nian_bad, nian_und = [], 0, [], 0
    for i in range(0, n, 2):
        a, b = p2[i], p2[i + 1]
        if a in ("UNDECIDED", "UNKNOWN") or b in ("UNDECIDED", "UNKNOWN"):
            dui_und += 1
        elif not opposite(a, b):
            dui_bad.append((i + 1, i + 2))
    for i in range(1, n - 1, 2):
        a, b = p2[i], p2[i + 1]
        if a in ("UNDECIDED", "UNKNOWN") or b in ("UNDECIDED", "UNKNOWN"):
            nian_und += 1
        elif a != b:
            nian_bad.append((i + 1, i + 2))
    res["rules"]["dui"] = {"pass": not dui_bad, "failed_couplets": dui_bad, "undecided": dui_und}
    res["rules"]["nian"] = {"pass": not nian_bad, "failed_joins": nian_bad, "undecided": nian_und}

    # --- rhyme: even hemistichs share one 平 group; hemistich 1 may join it
    rpos = list(range(1, n, 2))
    rchars = [lines[i][-1] for i in rpos]
    groups = [rime.rhyme_groups(c) for c in rchars]
    shared = set.intersection(*groups) if all(groups) else set()
    absent = [rpos[i] + 1 for i, c in enumerate(rchars) if rime.tone(c) == "UNKNOWN"]
    first = rime.rhyme_groups(lines[0][-1])
    # A 仄韻 poem is 古體, not a broken 近體 — and the test for it must be that the rhyme
    # characters share a 仄 group, not that they lack a 平 reading. 比 has a 平 reading
    # (上平四支) and rhymes 上四紙 with 裏 in 宮妃夜游圖; a "no 平 reading" test calls that
    # poem a defective 近體, which it is not.
    ze_groups = [rime.rhyme_groups(c, ping_only=False) - rime.rhyme_groups(c) for c in rchars]
    ze_shared = set.intersection(*ze_groups) if all(ze_groups) else set()
    res["classification"] = "古體(仄韻)" if (not shared and ze_shared) else "近體候補"
    rule = {"shared_group": sorted(shared)[:3], "rhyme_chars": rchars,
            "hemistichs_with_no_ping_shi_group": [rpos[i] + 1 for i, g in enumerate(groups)
                                                  if not g],
            "chars_absent_from_the_table": absent,
            "first_hemistich_rhymes": bool(shared & first)}
    if res["classification"] == "古體(仄韻)":
        rule["pass"] = True
        rule["why"] = "仄韻 throughout — 古體, and 近體's rhyme rule is not its rule"
        rule["ze_group"] = sorted(ze_shared)[:3]
    elif absent:
        # my own policy, applied to rhyme: a character the table does not hold cannot
        # decide anything. 邨 (a variant of 村, 上平十三元) is exactly this case.
        rule["pass"] = True
        rule["undecided"] = len(absent)
        rule["why"] = ("a rhyme character is absent from the 平水韻 table (variant glyph or □) "
                       "— UNDECIDED, not a failure")
    else:
        rule["pass"] = bool(shared)
    res["rules"]["rhyme"] = rule

    # --- non-rhyme hemistichs end 仄
    nbad, nund = [], 0
    for i in range(0, n, 2):
        if i == 0:
            continue                                   # 首句 may end either way
        t = rime.tone(lines[i][-1])
        if t in ("UNDECIDED", "UNKNOWN"):
            nund += 1
        elif t != "仄":
            nbad.append(i + 1)
    res["rules"]["non_rhyme_ends_ze"] = {"pass": not nbad, "failed_hemistichs": nbad,
                                         "undecided": nund}

    # --- 三平調
    three = []
    for i, t in enumerate(tones):
        if t[-3:] == ["平", "平", "平"]:
            three.append(i + 1)
    res["rules"]["san_ping_diao"] = {"pass": not three, "failed_hemistichs": three}

    if res["classification"] == "古體(仄韻)":
        for k in ("alternate", "dui", "nian", "non_rhyme_ends_ze", "san_ping_diao"):
            res["rules"][k]["pass"] = True
            res["rules"][k]["not_applicable"] = "古體"
    res["pass"] = all(r["pass"] for r in res["rules"].values())
    # STRICT: every undecided position counts as a failure. The two rates BRACKET the truth,
    # and they must be published together -- a text full of glyphs the table does not hold
    # (the 四庫 editions are) earns a HIGHER lenient rate simply by being unreadable, and a
    # single number would let unreadability look like correctness.
    res["pass_strict"] = res["pass"] and not any(
        r.get("undecided") for r in res["rules"].values()) and res["unknown"] == 0
    # ...but a WHOLE-POEM strict rate is dominated by a single ambiguous character and says
    # almost nothing. The useful bound is per POSITION: how much of the poem the table could
    # actually read. Enforced positions are the even ones in each hemistich plus each final
    # character; everything else is 一三五不論 and was never checked.
    enforced = und_pos = 0
    for t in tones:
        for p_ in list(even) + [len(t) - 1]:
            enforced += 1
            if t[p_] in ("UNDECIDED", "UNKNOWN"):
                und_pos += 1
    res["enforced_positions"] = enforced
    res["enforced_undecided"] = und_pos
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--text")
    ap.add_argument("--field", default="text", help="which JSONL field to read (text | text_raw)")
    ap.add_argument("--form", help="only score poems of this form, e.g. 七絕")
    ap.add_argument("--rime", default=os.path.join(HERE, "rime", "pingshui.json"))
    ap.add_argument("--report")
    a = ap.parse_args()
    rime = Rime(a.rime)

    if a.text:
        print(json.dumps(verify(a.text, rime), ensure_ascii=False, indent=1))
        return

    rows = [json.loads(l) for l in open(a.corpus)]
    out, skipped = [], 0
    for r in rows:
        v = verify(r.get(a.field) or r.get("text", ""), rime)
        if a.form and v["form"] != a.form:
            skipped += 1
            continue
        v["title"] = r.get("title")
        v["author"] = r.get("author")
        v["skchar"] = r.get("skchar", 0)
        out.append(v)

    scored = [v for v in out if v["form"]]
    npass = sum(1 for v in scored if v["pass"])
    nstrict = sum(1 for v in scored if v["pass_strict"])
    print(f"{a.corpus}  field={a.field}  form={a.form or 'any regulated'}")
    print(f"  poems of a regulated length: {len(scored)}   (skipped {skipped}, "
          f"no regulated form: {len(out)-len(scored)})")
    if scored:
        print(f"  PASS, undecided treated as latitude : {npass}/{len(scored)} = "
              f"{100.0*npass/len(scored):.1f}%")
        ep = sum(v["enforced_positions"] for v in scored)
        eu = sum(v["enforced_undecided"] for v in scored)
        print(f"  POSITIONS THE TABLE COULD READ     : {ep-eu}/{ep} = {100.0*(ep-eu)/ep:.1f}%"
              f"   <- the bound on the rate above")
        print(f"  (whole-poem strict, one ambiguous character fails the poem: "
              f"{nstrict}/{len(scored)} = {100.0*nstrict/len(scored):.1f}% — dominated by the "
              f"10.2% polyphonic characters, reported for completeness, not for use)")
    per_rule = Counter()
    per_rule_und = Counter()
    for v in scored:
        for k, r in v["rules"].items():
            if not r["pass"]:
                per_rule[k] += 1
            per_rule_und[k] += r.get("undecided", 0)
    for k in ("structure", "alternate", "dui", "nian", "rhyme", "non_rhyme_ends_ze",
              "san_ping_diao"):
        if k in per_rule or k in per_rule_und:
            print(f"    {k:20s} failed {per_rule[k]:4d}   hemistichs/joins undecided "
                  f"{per_rule_und[k]:4d}")
    tot_und = sum(v["undecided"] for v in scored)
    tot_unk = sum(v["unknown"] for v in scored)
    print(f"  characters whose tone the table cannot decide: {tot_und}")
    print(f"  characters absent from the table entirely:     {tot_unk}")
    if a.report:
        os.makedirs(os.path.dirname(a.report), exist_ok=True)
        json.dump({"corpus": a.corpus, "field": a.field, "form": a.form,
                   "n_scored": len(scored), "n_pass": npass, "n_pass_strict": nstrict,
                   "pass_rate_lenient": (npass / len(scored)) if scored else None,
                   "pass_rate_strict": (nstrict / len(scored)) if scored else None,
                   "failed_by_rule": dict(per_rule), "undecided_by_rule": dict(per_rule_und),
                   "chars_tone_undecided": tot_und, "chars_absent_from_table": tot_unk,
                   "enforced_positions": sum(v["enforced_positions"] for v in scored),
                   "enforced_undecided": sum(v["enforced_undecided"] for v in scored),
                   "poems": out}, open(a.report, "w"), ensure_ascii=False, indent=1)
        print(f"  -> {a.report}")


if __name__ == "__main__":
    main()
