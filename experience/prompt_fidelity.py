#!/usr/bin/env python3
"""Does the state extractor rebuild the prompt the agent actually read?

THE INSTRUMENT IS THE LOOP'S OWN HASH. Every decision step carries `decision_state` =
sha1(state_text()) computed by the loop at the moment the agent read it. So the extractor's
reconstruction can be checked without re-running anything: rebuild, hash, compare.

THE POSITIVE CONTROL IS INSIDE THE RESULT and must be read first: for an UNFORCED collection,
decision 1 must come back 300/300 exact. If that cell is not perfect the harness or the hash is
wrong and every other number here is meaningless -- which is the only reason to trust the
disagreements. (ship-the-control-inside-the-instrument)

Two mechanisms this found, both confirmed to the token:
  1. a forced look is logged as action="hypothesize", region=null, with the truth only in
     parsed_action="inspect_forced". The extractor keys on `action`, so it emits a decision row
     where no decision happened AND writes "hypothesize ?" where the loop wrote "inspect-forced".
  2. a patch step records `region` and `patch_ok` but NOT the source text, so once a patch has
     changed an inspected region the loop's prompt is unrecoverable IN PRINCIPLE.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, sys


def prompt_key(t):
    """Byte-for-byte the loop's own function (et8b_loop.prompt_key)."""
    return hashlib.sha1(t.encode()).hexdigest()


def measure(steps_path, tasks_dir, experience):
    sys.path.insert(0, experience)
    import et8b_states as S
    steps = [json.loads(l) for l in open(steps_path)]
    logged = {}
    for r in steps:
        if r.get("action") == "hypothesize" and r.get("decision_state"):
            logged[(r["task_id"], r.get("decision"))] = (r["decision_state"], r.get("parsed_action"))
    rows = S.decisions_from(steps, tasks_dir)
    agree, dis, forced = collections.Counter(), collections.Counter(), collections.Counter()
    unmatched = 0
    for d in rows:
        k = (d["task_id"], d.get("decision"))
        if k not in logged:
            unmatched += 1
            continue
        want, pa = logged[k]
        if pa == "inspect_forced":
            forced[d.get("decision")] += 1
        (agree if prompt_key(d["messages"][1]["content"]) == want else dis)[d.get("decision")] += 1
    tot = sum(agree.values()) + sum(dis.values())
    return {"steps": os.path.basename(steps_path), "rows": len(rows), "compared": tot,
            "unmatched": unmatched,
            "agree_by_decision": dict(sorted(agree.items(), key=lambda x: (x[0] is None, x[0]))),
            "disagree_by_decision": dict(sorted(dis.items(), key=lambda x: (x[0] is None, x[0]))),
            "forced_slots_by_decision": dict(sorted(forced.items(), key=lambda x: (x[0] is None, x[0]))),
            "reproducible": sum(agree.values()),
            "reproducible_pct": round(100.0 * sum(agree.values()) / max(1, tot), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    # default: THIS FILE'S OWN DIRECTORY. Shipped beside et8b_states.py, a stranger with a
    # clone runs it in-tree with no flag and no absolute path.
    ap.add_argument("--experience", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out")
    ap.add_argument("--expect-d1-exact", action="store_true",
                    help="REFUSE unless some collection reproduces decision 1 exactly -- the "
                         "positive control. Without it a broken harness reports 0%% everywhere "
                         "and looks like a finding.")
    a = ap.parse_args()
    out = [measure(s, a.tasks, a.experience) for s in a.steps]
    for r in out:
        print("%-34s rows %-5d reproducible %-5d (%4.1f%%)  agree %s  disagree %s  forced %s"
              % (r["steps"], r["rows"], r["reproducible"], r["reproducible_pct"],
                 r["agree_by_decision"], r["disagree_by_decision"], r["forced_slots_by_decision"]))
    if a.expect_d1_exact:
        ok = any(r["agree_by_decision"].get(1, 0) > 0 and r["disagree_by_decision"].get(1, 0) == 0
                 for r in out)
        if not ok:
            print("⛔ POSITIVE CONTROL FAILED: no collection reproduces decision 1 exactly. "
                  "Treat every number above as instrument error, not as a finding.")
            return 3
        print("✅ positive control: at least one collection reproduces decision 1 exactly")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
