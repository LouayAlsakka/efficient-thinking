#!/usr/bin/env python3
"""Rename the two demo fixture ids, with the old ids kept as ALIASES for the overlap day.

PREPARED BEFORE THE DAY, NOT ON IT. The rename is due after tonight's hands-on, and two other lanes
build against these ids -- the renderer sends one as the `venue` argument, and the compiled fixtures
carry them as slugs. Doing it as a taxonomy edit plus a reference sweep, with both ids resolving
throughout, means each lane moves when it chooses rather than in one synchronised step.

  fixture-v0   <- the one-offer fixture      (old id aliased)
  fixture-v1   <- the three-offer fixture    (old id aliased)

WHY ALIASES RATHER THAN A FLAG DAY. A flag day needs every consumer to land the same minute, and one
of them is a renderer someone is demoing. `AreaV0.resolve_venue` already resolves an alias to its
canonical id -- shipped and tested earlier, inert until this file writes the map -- so an old id
keeps working and a new id works immediately.

  --dry (default)   print what would change, touch nothing
  --apply           write the taxonomy and sweep the references
"""
import argparse, collections, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAX = os.path.join(HERE, "bench", "taxonomy_demo.json")
NEW = ["fixture-v0", "fixture-v1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--taxonomy", default=TAX)
    a = ap.parse_args()

    tax = json.load(open(a.taxonomy, encoding="utf8"))
    old = sorted(tax["venues"])
    if len(old) != 2:
        sys.exit("STOP: expected 2 fixtures, found %d: %s" % (len(old), old))
    if any(o in NEW for o in old):
        sys.exit("STOP: already renamed (%s). Nothing to do." % old)

    # THE ORDER IS DERIVED, NOT ASSUMED: v0 is the fixture with ONE offer. Hard-coding which slug
    # becomes which id would silently swap them if the file is ever regenerated in another order.
    by_offers = sorted(old, key=lambda k: len(tax["venues"][k].get("offer_tags") or {}))
    mapping = {by_offers[0]: "fixture-v0", by_offers[1]: "fixture-v1"}
    for o, n in mapping.items():
        print("  %-38s -> %-12s (%d offer(s))"
              % (o, n, len(tax["venues"][o].get("offer_tags") or {})))

    files = [f for f in subprocess.run(["git", "-C", HERE, "ls-files"], capture_output=True,
                                       text=True).stdout.split()
             if f.endswith((".py", ".json", ".jsonl", ".md", ".mjs", ".txt", ".sh"))]
    refs = collections.Counter()
    for f in files:
        p = os.path.join(HERE, f)
        try:
            b = open(p, encoding="utf8", errors="replace").read()
        except Exception:
            continue
        for o in mapping:
            c = b.count(o)
            if c:
                refs[f] += c
    print("\n  references in %d files, %d occurrences" % (len(refs), sum(refs.values())))
    for f, c in refs.most_common(12):
        print("     %3d  %s" % (c, f))

    if not a.apply:
        print("\n  (dry run — re-run with --apply)")
        return

    tax["venues"] = {mapping[o]: tax["venues"][o] for o in old}
    tax["aliases"] = {o: n for o, n in mapping.items()}
    tax["aliases_note"] = ("the previous fixture ids, kept resolvable for the overlap day so the "
                           "renderer and the compiled fixtures can move independently. Remove them "
                           "once both have.")
    json.dump(tax, open(a.taxonomy, "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print("\n  taxonomy rewritten with %d venues and %d aliases" % (len(tax["venues"]), len(tax["aliases"])))

    n = 0
    for f in refs:
        p = os.path.join(HERE, f)
        b = open(p, encoding="utf8").read()
        new = b
        for o, k in mapping.items():
            new = new.replace(o, k)
        if new != b:
            if f.endswith((".json", ".jsonl")):
                try:
                    (json.loads(new) if f.endswith(".json")
                     else [json.loads(l) for l in new.split("\n") if l.strip()])
                except Exception as e:
                    print("  REFUSED (would break json): %s (%s)" % (f, e))
                    continue
            open(p, "w", encoding="utf8").write(new)
            n += 1
    print("  %d file(s) swept. Old ids still RESOLVE via the alias map; they no longer appear." % n)


if __name__ == "__main__":
    main()
