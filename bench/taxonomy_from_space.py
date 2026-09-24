#!/usr/bin/env python3
"""Build the classifier's taxonomy from a COMPILED VENUE, so a real venue needs no hand-written file.

The demo taxonomy was hand-written for two fixtures and says so in its own note. A real venue's
tags are already in its compiled definition — the same tags the renderer will filter on — so
reading them from there is the only way the classifier and the screen can agree by construction
rather than by someone keeping two files in step.

WHAT IT REFUSES. A venue whose compiled offers carry NO tags produces an empty tag set, and a
classifier given an empty list can only ever answer "nothing here". That is a tagging gap, not a
classifier result, so it is a hard refusal with the count printed rather than a taxonomy that
technically loads.

⚠️ AND IT PRINTS THE COVERAGE, because a PARTIALLY tagged venue is the dangerous case: it loads, it
answers, and every offer nobody tagged is invisible to narrowing without any error anywhere.
"""
import argparse, json, os, sys


def tags_in(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "tags" and isinstance(v, list):
                out.update(str(x) for x in v if x)
            else:
                tags_in(v, out)
    elif isinstance(obj, list):
        for x in obj:
            tags_in(x, out)
    return out


def offer_coverage(space):
    offers = space.get("offer_sheet") or space.get("offers") or []
    if isinstance(offers, dict):
        offers = list(offers.values())
    tagged = sum(1 for o in offers if isinstance(o, dict) and (o.get("tags") or []))
    return tagged, len(offers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="a compiled venue json")
    ap.add_argument("--handle", default="", help="venue id; default: the space's own handle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--taxonomy-id", default="venue-v1")
    ap.add_argument("--merge-into", default="", help="an existing taxonomy to add this venue to")
    a = ap.parse_args()

    space = json.load(open(a.space, encoding="utf8"))
    handle = a.handle or (space.get("space") or {}).get("handle") or \
             os.path.basename(a.space).replace("_space.json", "")
    tags = sorted(tags_in(space, set()))
    tagged, total = offer_coverage(space)

    print("  venue   %s" % handle)
    print("  offers  %d, of which tagged: %d (%.0f%%)"
          % (total, tagged, 100.0 * tagged / total if total else 0))
    print("  tags    %d distinct: %s" % (len(tags), ", ".join(tags[:8]) + (" …" if len(tags) > 8 else "")))
    if not tags:
        sys.exit("STOP: this venue's compiled offers carry no tags. A classifier handed an empty "
                 "tag list can only answer 'nothing here', which is a tagging gap reported as a "
                 "classifier result. Tag the venue first.")
    if tagged < total:
        print("  ⚠️  %d of %d offers carry NO tag. They will be INVISIBLE to narrowing and nothing "
              "will error — the classifier cannot name what the venue did not declare."
              % (total - tagged, total))

    # parents from the dotted ids: a.b.c -> a.b -> a. Declared explicitly so the closure walk never
    # has to infer structure from a string at read time.
    parents = {}
    for t in tags:
        parts = t.split(".")
        for i in range(len(parts), 1, -1):
            child, parent = ".".join(parts[:i]), ".".join(parts[:i - 1])
            parents.setdefault(child, parent)

    tax = json.load(open(a.merge_into, encoding="utf8")) if a.merge_into else \
        {"taxonomy": a.taxonomy_id, "note": "", "leaves": [], "parents": {}, "venues": {}}
    tax["taxonomy"] = a.taxonomy_id
    tax["note"] = ("built from compiled venue definitions by bench/taxonomy_from_space.py — the "
                   "classifier's tags are the venue's own, so the screen and the classifier cannot "
                   "drift apart")
    tax["leaves"] = sorted(set(tax.get("leaves", [])) | set(tags))
    tax["parents"] = {**tax.get("parents", {}), **parents}
    tax.setdefault("venues", {})[handle] = {"tags": tags, "offer_tags": {}}

    # SHORT-SLUG ALIASES, DERIVED — not typed per venue.
    #
    # A renderer links a venue by the short form of its id, while the compiled fixture id carries
    # a locality suffix. Until now the two were bridged by a hand-written `aliases` map, so a
    # venue whose short link appeared later than its fixture simply had no alias and /area
    # refused it — one venue hit exactly that while the two typed in by hand worked, which is the
    # shape of a per-instance bridge. Deriving the alias closes the class instead of the instance.
    #
    # Derived only where it cannot be wrong: a first segment that is itself a canonical id would
    # shadow that venue, and one claimed by two venues is ambiguous. Both are SKIPPED and named
    # on stdout rather than resolved by a rule invented here. A hand-written alias always wins:
    # some short forms are respellings rather than prefixes, and no rule derives those.
    aliases = tax.setdefault("aliases", {})
    canonical = set(tax["venues"])
    first = {}
    for vid in canonical:
        if "." in vid:
            first.setdefault(vid.split(".")[0], []).append(vid)
    for short, owners in sorted(first.items()):
        if short in canonical:
            print("  alias SKIPPED %r — it is itself a venue id" % short)
        elif len(owners) > 1:
            print("  alias SKIPPED %r — claimed by %s" % (short, sorted(owners)))
        elif short in aliases:
            if aliases[short] != owners[0]:
                print("  alias KEPT %r -> %r (hand-written; derived would be %r)"
                      % (short, aliases[short], owners[0]))
        else:
            aliases[short] = owners[0]
    json.dump(tax, open(a.out, "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("  wrote %s — %d venue(s), %d leaves" % (a.out, len(tax["venues"]), len(tax["leaves"])))


if __name__ == "__main__":
    main()
