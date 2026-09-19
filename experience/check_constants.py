#!/usr/bin/env python3
"""Does every file agree with results/et8_constants.json? Fails loudly if not (理, nirai 11409).

THE RULE: one machine-readable registry, a checker that fails, and documents that CITE rather than
re-draw. 理 applied it to the 64-bit address tail after two specs allocated the same bits. The same
shape exists here: the controller's measured cost is typed into four files and the R2b bar into two.

A CHECKER NOBODY HAS SEEN FAIL IS NOT A CHECKER. `--controls` mutates the registry in memory and
asserts each check catches it; the exit status of the real run means nothing until those pass.
"""
from __future__ import annotations
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "results", "et8_constants.json")


def scan(value, files):
    """Which of `files` mention `value`, and which are missing it."""
    pat = re.compile(re.escape(str(value)).replace(r"\.", r"\."))
    hit, miss = [], []
    for f in files:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            miss.append((f, "FILE MISSING"))
            continue
        (hit if pat.search(open(p, encoding="utf-8", errors="replace").read()) else miss
         ).append(f if pat.search(open(p, encoding="utf-8", errors="replace").read()) else (f, "value %s NOT FOUND" % value))
    return hit, [m for m in miss if isinstance(m, tuple)]


def run(reg):
    fails = []
    for name, c in reg["constants"].items():
        _, miss = scan(c["value"], c["used_by"])
        for f, why in miss:
            fails.append("%s: %s — %s (registry says %s)" % (name, f, why, c["value"]))
    return fails


def main():
    reg = json.load(open(REG))
    if "--controls" in sys.argv:
        print("CONTROLS — each must FAIL, or the checker proves nothing:")
        ok = True
        for name in reg["constants"]:
            import copy
            bad = copy.deepcopy(reg)
            bad["constants"][name]["value"] = 999999.9
            f = run(bad)
            print("  mutate %-38s -> %d failure(s)  %s"
                  % (name, len(f), "CAUGHT" if f else "*** NOT CAUGHT ***"))
            ok = ok and bool(f)
        bad = json.loads(json.dumps(reg))
        k = list(bad["constants"])[0]
        bad["constants"][k]["used_by"] = bad["constants"][k]["used_by"] + ["no_such_file.py"]
        f = run(bad)
        print("  add a file that does not exist            -> %d failure(s)  %s"
              % (len(f), "CAUGHT" if f else "*** NOT CAUGHT ***"))
        ok = ok and bool(f)
        print("CONTROLS %s" % ("PASS — the checker fails when it should" if ok else "FAILED"))
        return 0 if ok else 1

    fails = run(reg)
    for f in fails:
        print("  FAIL  " + f)
    print("et8 constants: %d constant(s), %d file reference(s), %s"
          % (len(reg["constants"]), sum(len(c["used_by"]) for c in reg["constants"].values()),
             "ALL AGREE" if not fails else "%d DISAGREEMENT(S)" % len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
