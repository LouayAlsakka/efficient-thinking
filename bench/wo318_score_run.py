#!/usr/bin/env python
"""WO-318 §19 — the five numbers the narrowing page has to carry, scored from the scripted run's log.

COMMITTED BEFORE THE RUN EXISTS, on purpose: a scorer written after the log is a scorer written to
the data it scores. The field names are a MAP at the top rather than literals in the code, so 形's
driver can name its fields whatever it names them and this file adapts without being rewritten.

THE THREE PROPERTIES THAT MATTER MORE THAN THE RATES:
  1. A parser that sees nothing scores everything true. The parsed-row count is printed FIRST and a
     rate is never printed from zero rows -- the run is declared unreadable instead.
  2. A rate over rows that lack the field it needs is LENIENT, not neutral: a skipped row is a row
     that could not fail. Every rate carries its own denominator AND the count of rows dropped for
     want of a field.
  3. A rate's resolution is 1/n. It is printed beside the rate so nobody reads 100.0% on n=3 as a
     result.

WHAT IS DELIBERATELY NOT COMPUTED: the walk-up rate. WO-318 §19 names it; the three readings I can
think of ("resolved to the venue node with no narrowing", "resolved to an offer needing no resource
pick", "the user is told to walk in") have three different denominators, and the definition is 理's
to give. Pass --walkup-field to name the boolean field that IS the definition; without it this
prints NOT COMPUTED and says why. Guessing it would be a number nobody asked for.
"""
import argparse, collections, glob, json, os, sys

# field name -> what it must contain. Override any of them with --map name=other_name.
FIELDS = {
    "venue": "venue",                    # venue slug or id
    "utterance": "utterance",            # the text driven
    "gold": "gold",                       # the expected KIND: ask | commit | reselect ...
    "gold_areas": "gold_areas",           # the expected AREA list — a DIFFERENT field from the kind
    "areas": "areas",                     # the classifier's returned area list
    "off_menu": "off_menu",               # bool: the classifier declined the menu
    "gold_subtree": "gold_subtree",       # expected subtree/node for a commit
    "landed_subtree": "landed_subtree",   # the subtree/node the commit actually landed on
    "areas_replicate": "areas_replicate", # a SECOND draw of the classifier, same utterance
    "t_send": "t_send",                   # epoch seconds, classifier call sent
    "t_recv": "t_recv",                   # epoch seconds, reply received
    "in_flight": "in_flight",             # how many classifier calls were open when this one was sent
}


def rate(num, den):
    return None if not den else 100.0 * num / den


def fmt(label, num, den, dropped, note=""):
    if not den:
        return "  %-34s NOT COMPUTED — 0 rows carried the fields it needs%s" % (label, note)
    return ("  %-34s %6.1f%%  (%d/%d, resolution %.2f pp%s)%s"
            % (label, rate(num, den), num, den, 100.0 / den,
               ", %d row(s) dropped for a missing field" % dropped if dropped else "", note))


# ---------------------------------------------------------------------------------------------
# 形's row shape, FINAL as of the night before the run (their fields, off the real /area response,
# not this file's earlier guesses). One JSONL row per (venue, utterance):
#
#   {"venue","utterance",
#    "draw1": {"area": {"intent","tags","commit","taxonomy"}, "unresolved", "off_menu",
#              "area_unchanged", "classifier", "classify_ms", "t_send", "t_recv"},
#    "draw2": {... identical shape, a second independent call ...},
#    "resolved_ids": [...], "single_offer_id": null, "folded_ids_count": N}
#
# The projection below is the ONLY place that knows the driver's shape. Two things it does NOT do:
# it does not invent a gold label (形 carries none; --gold joins one by (venue, utterance)), and it
# does not hide that `unresolved` is COMPUTED here -- 形 deliberately shipped the two raw
# ingredients rather than a bool, because a field they compute is still inferred by somebody. The
# formula is printed beside the rate.
UNRESOLVED_RULE = "off_menu OR (area.tags empty AND area.commit is false)"


def _iso(t):
    """ISO-8601 with a trailing Z -> epoch seconds. Returns None rather than raising."""
    if not isinstance(t, str):
        return None
    try:
        from datetime import datetime
        return datetime.strptime(t.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    except Exception:
        try:
            from datetime import datetime
            return datetime.strptime(t.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z").timestamp()
        except Exception:
            return None


def project_katachi(r):
    """One driver row -> the flat record the rates are computed over."""
    d1 = r.get("draw1") or {}
    d2 = r.get("draw2") or {}
    a1 = d1.get("area") or {}
    a2 = d2.get("area") or {}
    tags1 = a1.get("tags") if isinstance(a1.get("tags"), list) else None
    tags2 = a2.get("tags") if isinstance(a2.get("tags"), list) else None
    out = {"venue": r.get("venue"), "utterance": r.get("utterance"),
           "areas": tags1, "off_menu": d1.get("off_menu"),
           "areas_replicate": tags2,
           "intent": a1.get("intent"), "commit": a1.get("commit"),
           "landed_subtree": (sorted(r["resolved_ids"]) if isinstance(r.get("resolved_ids"), list)
                              else r.get("single_offer_id")),
           "t_send": _iso(d1.get("t_send")), "t_recv": _iso(d1.get("t_recv")),
           "in_flight": None}          # never asserted: verified from the timestamps instead
    if tags1 is not None or d1.get("off_menu") is not None:
        out["unresolved"] = bool(d1.get("off_menu")) or (not tags1 and a1.get("commit") is False)
    return out


def check_serial(rows):
    """The driver DECLARED one call in flight at a time. Declared is not measured: two intervals
    that overlap would mean the seconds on the page include a queue. Checked, not trusted."""
    iv = sorted((r["t_send"], r["t_recv"]) for r in rows
                if r.get("t_send") is not None and r.get("t_recv") is not None)
    if len(iv) < 2:
        return None, len(iv)
    overlaps = sum(1 for i in range(1, len(iv)) if iv[i][0] < iv[i - 1][1])
    return overlaps, len(iv)


def load_gold(paths):
    """Gold rows keyed (venue, utterance). Authored by 案内; a reviewer who authors the gold set is
    the conflict WO-318 §19's split exists to prevent, so this file only READS it.

    THE ROW SHAPE IS THE LANDED ONE, not the one this file first guessed: each row is
    {"utterance", "kind", "area": {"intent", "tags", "commit", "taxonomy"}} and the VENUE is the
    FILENAME (gold/utterances-v1/<venue>.jsonl), not a field. Pass one file per venue or a glob.
    """
    gold = {}
    for path in paths:
        venue = os.path.basename(path)
        for suf in (".jsonl", ".json"):
            if venue.endswith(suf):
                venue = venue[: -len(suf)]
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            v = str(g.get("venue") or venue)     # a venue field wins if the author ever adds one
            gold[(v, str(g.get("utterance")))] = g
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="jsonl, one object per (venue, utterance)")
    ap.add_argument("--map", action="append", default=[],
                    help="rename a field: --map areas=classified_areas (repeatable)")
    ap.add_argument("--shape", choices=("katachi", "flat"), default="katachi",
                    help="katachi: the driver's nested draw1/draw2 rows (the contract of the night "
                         "before the run). flat: one object per utterance with the FIELDS names.")
    ap.add_argument("--gold", nargs="*", default=[],
                    help="gold utterance files, one per venue (the venue is the FILENAME). Authored "
                         "by 案内; the driver carries no gold label, and without these the ask and "
                         "commit rates are NOT COMPUTED.")
    ap.add_argument("--venue-tags", default="",
                    help="directory of gold/tags-v1/<venue>.json ({offers: {offer_id: [tags]}}). The "
                         "gold utterance rows carry expected TAGS but no expected offer ids, so "
                         "commit-subtree accuracy is DERIVED from these when given — and the "
                         "derivation encodes a semantics choice, so both readings are printed.")
    ap.add_argument("--served-taxonomy", default="",
                    help="the taxonomy json the classifier was actually SERVING. Every gold tag "
                         "absent from it is a row the classifier cannot answer by construction, and "
                         "that is counted and named BEFORE any rate is printed.")
    ap.add_argument("--gold-split", action="store_true",
                    help="declare that the gold field is the composite '<kind>:<area>' and may be "
                         "split on the first colon. Without it a composite gold is REFUSED for the "
                         "ask rate rather than split silently.")
    ap.add_argument("--walkup-field", default="",
                    help="the boolean field that IS the walk-up definition (理's to give)")
    ap.add_argument("--out", default="", help="write the numbers as json")
    a = ap.parse_args()
    for m in a.map:
        if "=" not in m:
            sys.exit("--map takes name=other_name, got %r" % m)
        k, v = m.split("=", 1)
        if k not in FIELDS:
            sys.exit("--map %r is not a field this scorer has: %s" % (k, ", ".join(sorted(FIELDS))))
        FIELDS[k] = v
    F = FIELDS

    rows, bad = [], 0
    for line in open(a.log):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            bad += 1

    # ---- 1. THE PARSE, BEFORE ANY RATE -------------------------------------------------------
    print("  log %s" % a.log)
    print("  parsed %d row(s); %d line(s) did not parse" % (len(rows), bad))
    if not rows:
        print("\n  ⛔ NO ROWS PARSED. No rate is printed. This is not 'everything passed' — it is an\n"
              "     unreadable run. Check the field map (--map) against the driver's own output.")
        return 2
    raw = rows
    if a.shape == "katachi":
        rows = [project_katachi(r) for r in raw]
        got = sum(1 for r in rows if r.get("areas") is not None)
        print("  shape katachi: projected %d row(s); %d carried draw1.area.tags" % (len(rows), got))
        if not got:
            print("\n  ⛔ NOT ONE ROW carried draw1.area.tags. No rate is printed: this is a SHAPE\n"
                  "     mismatch, not a clean run. Check the driver's output against --shape.")
            return 2
    venue_offers = {}
    if a.venue_tags:
        for _p in sorted(glob.glob(os.path.join(a.venue_tags, "*.json"))):
            _v = os.path.basename(_p)[:-5]
            try:
                venue_offers[_v] = json.load(open(_p)).get("offers") or {}
            except Exception as _e:
                print("  ⚠️ could not read %s: %s" % (os.path.basename(_p), _e))
        print("  venue tag inventories: %d venue(s), %d offer(s)"
              % (len(venue_offers), sum(len(v) for v in venue_offers.values())))
    gold = load_gold(a.gold) if a.gold else {}
    if gold:
        joined = 0
        for r in rows:
            g = gold.get((str(r.get("venue")), str(r.get("utterance"))))
            if g:
                joined += 1
                ga = g.get("area") or {}
                if "gold_areas" not in r and ga.get("tags") is not None:
                    r["gold_areas"] = ga["tags"]
                if "gold" not in r and ga.get("intent") is not None:
                    r["gold"] = ga["intent"]
                r["gold_kind"] = g.get("kind")
                r["gold_taxonomy"] = ga.get("taxonomy")
                if ga.get("commit") is not None:
                    r["gold_commit"] = ga["commit"]
                if g.get("gold_subtree") is not None:
                    r["gold_subtree"] = (sorted(g["gold_subtree"])
                                         if isinstance(g["gold_subtree"], list) else g["gold_subtree"])
        print("  gold %d file(s) (%s): %d of %d row(s) joined on (venue, utterance)"
              % (len(a.gold), ", ".join(sorted(os.path.basename(x) for x in a.gold))[:70], joined,
                 len(rows)))
        if joined < len(rows):
            print("     ⚠️ %d row(s) have NO gold row. They are dropped from the ask and commit\n"
                  "     rates and counted as dropped, never scored as correct." % (len(rows) - joined))
        _h = __import__("hashlib").sha256()
        for _p in sorted(a.gold):
            _h.update(open(_p, "rb").read())
        gsha = _h.hexdigest()[:16]
        print("     gold sha256 %s — stamped into the output so a placeholder-scored page cannot be\n"
              "     mistaken for one scored against the authored set." % gsha)
    elif a.shape == "katachi":
        print("  ⚠️ NO --gold: the driver carries no gold label, so ask-classification and\n"
              "     commit-subtree accuracy are NOT COMPUTED. Replicate disagreement, the\n"
              "     unresolved rate and the latency check do not need one and are printed.")
    # the absent-field list is computed on the rows the RATES read -- after any projection. On the
    # raw driver rows it listed every flat field as missing, which is true and useless.
    present = collections.Counter(k for r in rows for k in r)
    missing = [n for n in F.values() if present[n] == 0]
    if missing:
        print("  ⚠️ fields absent from EVERY row the rates read: %s" % ", ".join(sorted(missing)))
    # THE INSTRUMENT CHECK, BEFORE ANY RATE. A classifier cannot return a tag its taxonomy does not
    # contain, so a gold row whose tags are absent from the SERVED taxonomy is a row that can only
    # lose -- and a version stamp that differs between the gold and the reply is the tell. On
    # 2026-09-26 the gold set stamped v1.4 while the live service served gold-v1.2, and 7 rows of
    # 185 carried a tag the service could not emit. Found by comparing the two, not by reading a
    # rate that looked low.
    gtax = collections.Counter(str(r.get("gold_taxonomy")) for r in rows if r.get("gold_taxonomy"))
    ltax = collections.Counter(str((r0.get("draw1") or {}).get("area", {}).get("taxonomy"))
                               for r0 in (raw if a.shape == "katachi" else []))
    if gtax or ltax:
        print("  taxonomy STAMPS — gold: %s   served (from the log's own rows): %s"
              % (dict(gtax) or "(none)", dict(ltax) or "(none)"))
        # The stamp line is INFORMATIONAL and says so generically. It used to name the versions in
        # play on the night it was written ("the served taxonomy says gold-v1.2"), which went stale
        # the moment the service was swapped to v1.4 — a guard that asserts yesterday's state is
        # the thing this file exists to catch in other people's instruments.
        print("     ⚠️ the stamps are NOT a check: the three files in this measurement (utterance\n"
              "     rows, venue tag inventories, served taxonomy) version the same concept in\n"
              "     independent vocabularies, so equal or differing stamps prove nothing either\n"
              "     way. The real check is the TAG SET comparison below, which needs\n"
              "     --served-taxonomy.")
    if not a.served_taxonomy:
        print("     ⛔ --served-taxonomy NOT GIVEN: whether the classifier could even emit each gold\n"
              "     tag is UNCHECKED. A gold tag absent from the served taxonomy is a guaranteed\n"
              "     miss, and on 2026-09-26 seven rows of 185 were in that state.")
    if a.served_taxonomy:
        import re as _re
        _dot = _re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9_]+)+$")
        def _ids(o, out=None):
            out = set() if out is None else out
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and _dot.match(k):
                        out.add(k)
                    _ids(v, out)
            elif isinstance(o, list):
                for v in o:
                    _ids(v, out)
            elif isinstance(o, str) and _dot.match(o):
                out.add(o)
            return out
        served_ids = _ids(json.load(open(a.served_taxonomy)))
        unreachable = [(r.get("venue"), r.get("utterance"),
                        sorted(set(r["gold_areas"]) - served_ids))
                       for r in rows if r.get("gold_areas") and set(r["gold_areas"]) - served_ids]
        print("  served taxonomy %s: %d ids; %d gold row(s) carry a tag it CANNOT return%s"
              % (os.path.basename(a.served_taxonomy), len(served_ids), len(unreachable),
                 "  🔴 those rows can only lose" if unreachable else "  ✅"))
        for v, u, miss in unreachable[:12]:
            print("     %-32s %-36s absent: %s" % (str(v)[:32], str(u)[:36], miss))
    venues = sorted({str(r.get(F["venue"], "?")) for r in rows})
    print("  %d venue(s): %s" % (len(venues), ", ".join(venues)))

    def has(r, *keys):
        return all(F[k] in r and r[F[k]] is not None for k in keys)

    # ---- 2. the rates, each over its own readable denominator ---------------------------------
    ask_n = ask_ok = ask_drop = 0
    gold_seen = []          # the EFFECTIVE expected areas, after --gold-split or the gold_areas field
    amb_n = amb_any = amb_all = 0
    com_n = com_ok = com_drop = com_ovl = com_derived = 0
    rep_n = rep_dis = rep_drop = 0
    unres_n = unres = 0
    for r in rows:
        gold = r.get(F["gold"])
        # ask classification: the classifier's areas must contain every expected area. The EXPECTED
        # AREAS are their own field -- a gold KIND ("ask", "commit") is not an area, and comparing a
        # kind against an area list scores 0% on a run that may be perfect. This scorer's own first
        # test did exactly that, which is why the two are separate fields now.
        want = None
        if has(r, "gold_areas"):
            want = r[F["gold_areas"]]
        elif a.gold_split and isinstance(gold, str) and ":" in gold:
            want = [gold.split(":", 1)[1]]
        if want is not None and has(r, "areas"):
            ask_n += 1
            got = r[F["areas"]]
            got = got if isinstance(got, list) else [got]
            want = want if isinstance(want, list) else [want]
            gold_seen.append(",".join(sorted(map(str, want))) or "(none expected)")
            # AN EMPTY EXPECTATION IS NOT A FREE PASS. `all(w in got for w in [])` is True, so a
            # gold row that expects NO area scored correct whatever the classifier returned -- the
            # off-menu rows are exactly the ones that would have hidden a wrong answer. When
            # nothing is expected, nothing may be returned.
            #
            # AND THE MATCH IS PER KIND, because the gold set carries two different meanings in one
            # `tags` field and nothing in the row says which: a `plain_commit` like "a haircut and
            # beard trim" lists tags CONJUNCTIVELY (both are wanted), while an `ambiguous` row like
            # "something for my face" lists five tags DISJUNCTIVELY (any one is a defensible read).
            # Scoring the ambiguous rows all-of demands a five-tag answer and would report ~0% on
            # 12 rows that may be perfectly handled. Both readings are printed for the ambiguous
            # subset below, so this choice is visible rather than load-bearing.
            kind = str(r.get("gold_kind") or "")
            if not want:
                ok = not got
            elif kind == "ambiguous":
                ok = any(w in got for w in want)
                amb_n += 1; amb_any += int(ok); amb_all += int(all(w in got for w in want))
            else:
                ok = all(w in got for w in want)
            ask_ok += int(ok)
        else:
            ask_drop += 1
        # THE COMMIT ROWS ARE SELECTED BY THE GOLD'S OWN FIELDS, not by the word "commit" in an
        # intent. The landed gold set uses kind="plain_commit" with intent="book"/"order" -- keying
        # this on `intent.startswith("commit")` selected NOTHING and the commit rate printed
        # "NOT COMPUTED" over 33 perfectly good rows. An instrument keyed to the wrong field does
        # nothing silently; the simulation against the real gold file is what showed it.
        is_commit = (str(r.get("gold_kind") or "").endswith("commit")
                     or bool((r.get("gold_commit") if "gold_commit" in r else None))
                     or str(gold).lower().startswith("commit"))
        if is_commit:
            if has(r, "gold_subtree", "landed_subtree"):
                com_n += 1
                com_ok += int(r[F["gold_subtree"]] == r[F["landed_subtree"]])
            elif venue_offers and r.get("gold_areas") and has(r, "landed_subtree"):
                # DERIVED, and both readings kept: which offers SHOULD a commit resolve to?
                #   subset   every gold tag is on the offer   (the narrow reading)
                #   overlap  the offer carries any gold tag   (the wide reading)
                off = venue_offers.get(str(r.get("venue")), {})
                want_t = set(r["gold_areas"])
                sub = sorted(o for o, ts in off.items() if want_t <= set(ts))
                ovl = sorted(o for o, ts in off.items() if want_t & set(ts))
                landed = r[F["landed_subtree"]]
                landed = sorted(landed) if isinstance(landed, list) else [landed]
                com_n += 1
                com_ok += int(landed == sub)
                com_ovl += int(landed == ovl)
                com_derived += 1
            else:
                com_drop += 1
        if has(r, "areas", "areas_replicate"):
            rep_n += 1
            rep_dis += int(sorted(map(str, r[F["areas"]] if isinstance(r[F["areas"]], list) else [r[F["areas"]]]))
                           != sorted(map(str, r[F["areas_replicate"]] if isinstance(r[F["areas_replicate"]], list)
                                         else [r[F["areas_replicate"]]])))
        else:
            rep_drop += 1
        if "unresolved" in r:
            unres_n += 1
            unres += int(bool(r["unresolved"]))
        elif F["areas"] in r or F["off_menu"] in r:
            unres_n += 1
            empty = not r.get(F["areas"])
            unres += int(bool(r.get(F["off_menu"])) or empty)

    # PER KIND, because the pooled ask rate hides four different failure modes: an off-menu row
    # that comes back with a tag is a different defect from a plain ask that comes back empty, and
    # WO-318 §19's single "ask-classification accuracy" cannot separate them. The pooled number
    # stays the headline; these say where it came from.
    per_kind = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        want = r.get("gold_areas")
        if want is None or F["areas"] not in r:
            continue
        got = r[F["areas"]] or []
        kind = str(r.get("gold_kind") or "?")
        if not want:
            ok = not got
        elif kind == "ambiguous":
            ok = any(w in got for w in want)
        else:
            ok = all(w in got for w in want)
        per_kind[kind][1] += 1
        per_kind[kind][0] += int(ok)

    print("\n  RATES — each with its own denominator, not the run's row count")
    if not ask_n and present[F["gold"]] and not present[F["gold_areas"]]:
        ex = next((r[F["gold"]] for r in rows if isinstance(r.get(F["gold"]), str)), "")
        print("  ⚠️ no %s field. The %s field looks like %r: if that is '<kind>:<area>', pass\n"
              "     --gold-split to DECLARE the split, or --map gold_areas=<field>. A label I did not\n"
              "     define is not one I will split silently — comparing a kind against an area list\n"
              "     scores 0%% on a run that may be perfect." % (F["gold_areas"], F["gold"], ex))
    print(fmt("ask-classification accuracy", ask_ok, ask_n, ask_drop))
    # the base rate and the prediction distribution, beside the agreement number. An agreement rate
    # is not a signal until you know what both sides say: 85.4% agreement was once 85.4% base rate
    # with the agent saying "false" 568 times out of 568.
    def dist(vals):
        c = collections.Counter(vals)
        return ", ".join("%s=%d" % kv for kv in c.most_common(6)) or "(none)"
    if amb_n:
        print("    ambiguous rows (%d), BOTH readings so the choice is visible: any-of %.1f%%  "
              "all-of %.1f%%" % (amb_n, 100.0 * amb_any / amb_n, 100.0 * amb_all / amb_n))
        print("      the rate above uses ANY-OF for these; a 5-tag row scored all-of demands a "
              "5-tag answer")
    if per_kind:
        print("    by gold kind, since the pooled number hides four failure modes:")
        for k in sorted(per_kind, key=lambda k: -per_kind[k][1]):
            ok, n = per_kind[k]
            print("      %-14s %5.1f%%  (%d/%d, resolution %.1f pp)"
                  % (k, 100.0 * ok / n, ok, n, 100.0 / n))
    print("    expected areas:   %s   [over the %d row(s) the rate used]" % (dist(gold_seen), ask_n))
    print("    returned areas:   %s" % dist(
        [",".join(sorted(map(str, r[F["areas"]]))) if isinstance(r.get(F["areas"]), list)
         else str(r.get(F["areas"])) for r in rows]))
    print(fmt("commit-subtree accuracy", com_ok, com_n, com_drop,
              note="" if com_n else
              " — the gold rows carry expected TAGS but no expected offer ids; pass --venue-tags to"
              " DERIVE them, or ask for a gold_subtree field"))
    if com_derived:
        print("    ⚠️ DERIVED on %d row(s) from the venue tag inventories, not carried by the gold "
              "set." % com_derived)
        print("    denominator composition: %s — a re-select carries commit=true in the gold set, so "
              "it\n    is a commit for this rate; that is the gold set's own field, not my choice."
              % dict(collections.Counter(str(r.get("gold_kind")) for r in rows
                                         if r.get("gold_commit"))))
        print("    both readings: subset (every gold tag on the offer) %.1f%%   overlap (any gold "
              "tag) %.1f%%" % (100.0 * com_ok / com_n, 100.0 * com_ovl / com_n))
    print(fmt("classifier replicate disagreement", rep_dis, rep_n, rep_drop,
              note="" if rep_n else " — the driver recorded ONE draw; disagreement needs two"))
    print(fmt("unresolved rate", unres, unres_n, 0))
    print("    computed HERE, not carried: %s" % UNRESOLVED_RULE)
    # AND ITS FLOOR. Some gold rows are SUPPOSED to come back unresolved -- every off_menu row and
    # every ask whose gold tag list is empty. An unresolved rate read without that floor looks like
    # a failure rate; it is not one until it exceeds the share the gold set asks for.
    exp_un = [r for r in rows if r.get("gold_areas") is not None and not r["gold_areas"]]
    if gold:
        print("    the gold set's OWN unresolved share: %.1f%% (%d/%d) — the floor this rate is\n"
              "    read against, not zero" % (100.0 * len(exp_un) / len(rows), len(exp_un), len(rows)))
    if a.walkup_field:
        w_n = sum(1 for r in rows if a.walkup_field in r)
        w = sum(1 for r in rows if r.get(a.walkup_field))
        print(fmt("walk-up rate (%s)" % a.walkup_field, w, w_n, len(rows) - w_n))
    else:
        print("  walk-up rate                       NOT COMPUTED — no definition given. Three readings\n"
              "                                     with three different denominators; --walkup-field\n"
              "                                     names the boolean that settles it (理's to give).")

    # ---- 3. latency, and whose it is ---------------------------------------------------------
    lat = [(r[F["t_recv"]] - r[F["t_send"]]) for r in rows if has(r, "t_send", "t_recv")]
    print("\n  LATENCY — /area is a single-threaded HTTPServer (bench/predictor_serve.py:266): under\n"
          "  concurrency it QUEUES rather than dropping, so seconds-per-call is the queue, not the model.")
    if not lat:
        print("  ⛔ no t_send/t_recv pair on any row: the page cannot carry a latency number, and I will\n"
              "     not publish one from wall-clock division.")
    else:
        lat.sort()
        inf = [r.get(F["in_flight"]) for r in rows if F["in_flight"] in r]
        conc = max([i for i in inf if isinstance(i, int)], default=None)
        print("  n=%d  median %.2f s  p90 %.2f s  max %.2f s" %
              (len(lat), lat[len(lat) // 2], lat[int(0.9 * (len(lat) - 1))], lat[-1]))
        if conc is None:
            ov, n_iv = check_serial(rows)
            if ov is None:
                print("  ⚠️ fewer than two t_send/t_recv pairs: seriality not checkable.")
            elif ov == 0:
                print("  VERIFIED SERIAL from the timestamps: 0 overlapping call intervals of %d.\n"
                      "  The driver declared serial; this is the check, not the declaration. These\n"
                      "  seconds are the model's, not my queue's." % n_iv)
            else:
                print("  🔴 %d of %d call intervals OVERLAP: the run was NOT serial, whatever it was\n"
                      "  declared to be, and these seconds include my own single-threaded queue.\n"
                      "  Label them as such on the page." % (ov, n_iv))
        elif conc <= 1:
            print("  in_flight max %s -> strictly serial; these seconds are the model's." % conc)
        else:
            print("  in_flight max %s -> CONCURRENT; these seconds include my own queue and must be\n"
                  "  labelled as such beside the page's numbers." % conc)

    if a.out:
        json.dump({"document": "WO-318 §19 — scripted narrowing run, scored",
                   "log": os.path.basename(a.log), "rows_parsed": len(rows), "lines_unparsed": bad,
                   "venues": venues,
                   "ask_classification": {"ok": ask_ok, "n": ask_n, "dropped": ask_drop,
                                          "pct": rate(ask_ok, ask_n)},
                   "commit_subtree": {"ok": com_ok, "n": com_n, "dropped": com_drop,
                                      "pct": rate(com_ok, com_n)},
                   "replicate_disagreement": {"disagreed": rep_dis, "n": rep_n, "dropped": rep_drop,
                                              "pct": rate(rep_dis, rep_n)},
                   "unresolved": {"n_unresolved": unres, "n": unres_n, "pct": rate(unres, unres_n)},
                   "walk_up": ("not computed — no definition given" if not a.walkup_field
                               else {"field": a.walkup_field}),
                   "latency_s": {"n": len(lat),
                                 "median": lat[len(lat) // 2] if lat else None,
                                 "max": lat[-1] if lat else None,
                                 "concurrency_established": bool(lat) and any(
                                     F["in_flight"] in r for r in rows)},
                   "signed": "Sautee (sha-ta)"}, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
