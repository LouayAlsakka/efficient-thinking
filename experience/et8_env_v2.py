#!/usr/bin/env python3
"""ET-8 task set v2 — a localisation axis that is not degenerate, gated before it is used.

    python3 et8_env_v2.py gate --n 400 --seed 11 --out /tmp/v2_probe      # measure, emit nothing
    python3 et8_env_v2.py gen  --n 2000 --seed 11 --out experience/tasks/v2

WHY v2 EXISTS. v1 had ten distinct symptom strings and each mapped to exactly ONE bug region, so a
ten-row lookup table localised at 100% and the agent's 88.05% was BELOW it
(experience/results/v1_symptom_determines_region.json). 理 ruled G does not run on v1.

THE DETERMINISM CAME FROM TWO HARDWIRINGS THAT MULTIPLIED, and a fix that touches only one leaves
the defect standing:

  1. the region was DERIVED FROM THE TEST NAME BY SUBSTRING --
     `region = "producer" if "produce" in name else "transform" if "normalize" in name else ...`
     The suite is organised by region, so the failing test names the region by construction.
  2. every bug class injected into exactly ONE region -- `_inject(cls)` RETURNED the region.

So v2 changes both:

  A. MULTI-REGION INJECTORS. A class appears in every region where its pattern genuinely exists.
     `wrong_comparator` lives in transform, aggregate AND consumer; `int_division` in transform and
     aggregate; `early_return` in producer, transform and aggregate. A class that is only
     expressible in one place STAYS in one place -- inventing a second site to hit a metric would
     be building the benchmark around the gate.
  B. BEHAVIOURAL SYMPTOMS. The symptom states what the program DID and what was expected -- the
     assertion detail -- and never the test's name. That is also the more honest artifact: a
     developer sees a diff, not a region label.

WHAT IS DELIBERATELY UNCHANGED: the action loop, the budget, the verifier, the region names and
the clean program. 理's comparability constraint. Only the generator moves.

THE GATE IS THE POINT, AND IT RUNS BEFORE ANY EPISODE. `gate` generates, measures 理's three
pre-registered properties and EMITS NOTHING. `gen` refuses to write unless they pass:

  P1  every failing-test identity is reachable from >= 2 bug regions
  P2  mean H(bug_region | symptom) >= 1.0 bit
  P3  a lookup table on the symptom string scores <= 60% localisation, held out

P3 is measured the way the defect was FOUND -- fit the table on one half, score on the other --
because an in-sample table scores 100% on any set and would have passed v1 too.
"""
from __future__ import annotations
import argparse, collections, json, math, os, random, sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et8_env as E

REGION_ORDER = E.REGION_ORDER

# class -> [(region, find, replace), ...]. Every entry is a real pattern in that region's clean
# source; `gen` asserts each replacement actually changes the text, so a silent no-op injector
# cannot masquerade as a bug (v1's selftest caught this class of thing once already).
INJECTORS = {
    # off_by_one at FOUR sites. This is the class the whole v2 design rests on: an item dropped
    # anywhere in produce -> normalize -> summarize -> report surfaces as the SAME end-to-end
    # assertion ("2 != 3"), so the symptom is genuinely ambiguous rather than merely uninformative.
    # That distinction is the one v2 has to get right: a task set where localisation is uncertain
    # because the symptom says nothing is not a better benchmark, it is a starved one.
    "off_by_one": [
        ("producer",  "range(len(items))", "range(len(items) - 1)"),
        ("transform", "for v in scores:", "for v in scores[:-1]:"),
        ("aggregate", "n = len(values)", "n = len(values) - 1 if values else 0"),
        ("consumer",  "scores = produce(items)", "scores = produce(items)[:-1]"),
    ],
    # wrong_comparator: v1 paired the comparator flip with `v = cap` -> `v = cap - 1`, because the
    # flip ALONE changes nothing observable -- clamping at `>=` instead of `>` gives the same value.
    # I dropped the pairing at first and all three sites came back GREEN, which is the injector
    # equivalent of a parser that sees nothing. The aggregate and consumer sites stay dropped: they
    # would need a test exercising a mean of exactly 0.5, and adding one to make an injector fire
    # is building the suite around the bug rather than the behaviour.
    # and DROPPED too, for the same reason as the other two: its only observable site fails
    # test_normalize_clamp, which no other region can fail. Restoring the pairing was still worth
    # doing -- a silently GREEN injector in the table is worse than an absent one.
    # "wrong_comparator": [("transform", "if v > cap:\n            v = cap",
    #                                    "if v >= cap:\n            v = cap - 1")],
    "int_division": [
        ("transform", "res.append(v / cap)", "res.append(v // cap)"),
        ("aggregate", "mean = (sum(values) / n) if n else 0.0",
                      "mean = (sum(values) // n) if n else 0.0"),
    ],
    "early_return": [
        ("producer",  "            out.append(int(s))\n",
                      "            out.append(int(s))\n            return out\n"),
        ("transform", "        res.append(v / cap)\n",
                      "        res.append(v / cap)\n        return res\n"),
        ("consumer",  '    summ["grade"] = "high" if summ["mean"] >= 0.5 else "low"\n',
                      '    return summ\n    summ["grade"] = "high" if summ["mean"] >= 0.5 else "low"\n'),
    ],
    # DROPPED FROM v2, and named rather than silently absent:
    #   width_mismatch  (producer) fails only test_produce_nbsp
    #   mutable_default (aggregate) fails only test_summarize_bucket_fresh
    #   early_return    (aggregate) is not observable at all -- the break leaves the bucket correct
    # The first two are real bugs, but each fails a test NO OTHER REGION can fail, so each would
    # reintroduce exactly v1's defect for its own slice of the set: one symptom, one region. They
    # come back the moment the suite has an end-to-end assertion they also break. Inventing a
    # second injection site for them would be building the benchmark around the gate.
    "missing_key": [
        ("consumer", 'summ["mean"] >= 0.5', 'summ["avg"] >= 0.5'),
    ],
    "swapped_args": [
        ("consumer", "normalize(scores, cap)", "normalize(cap, scores)"),
    ],
}

# v1's mutable_default injector carried a SECOND replacement meant to delete the
# `if bucket is None:` guard -- written at the wrong indentation, so it silently matched nothing
# and was dead for all 2,000 tasks. It made no difference, which is why nobody saw it: with the
# default already `[]` the guard never fires and the list is shared regardless. Dropped here
# rather than repaired, and recorded rather than quietly removed -- a no-op that changes no
# behaviour is still a claim in the source about what the bug IS.
_EXTRA = {}


def build(cls: str, region: str, find: str, replace: str) -> str:
    src = E.CLEAN_REGIONS[region]
    if find not in src:
        raise RuntimeError(f"injector {cls}/{region}: pattern not present in the clean source")
    out = src.replace(find, replace)
    for f2, r2 in _EXTRA.get((cls, region), []):
        if f2 not in out:
            raise RuntimeError(f"injector {cls}/{region}: extra pattern not present")
        out = out.replace(f2, r2)
    if out == src:
        raise RuntimeError(f"injector {cls}/{region}: replacement changed nothing")
    return out


def run_tests_v2(program: str):
    """Like E.run_tests, but it actually recovers the assertion detail.

    E.run_tests searches a 600-character window from the FIRST occurrence of the test name. Under
    `unittest -v` that first occurrence is the PROGRESS line ("test_x (...) ... FAIL"), which is
    printed before any traceback, so the window covers other tests' progress lines and the detail
    comes back empty. Every v1 symptom but one reads "no assertion detail" for that reason -- which
    is also why v1's symptom strings were so few and so region-pure.

    This parses the failure BLOCKS instead: unittest prints them after a line of '=' with a header
    "FAIL: test_x (...)", and the block's LAST non-empty line is the exception and its detail.
    """
    import subprocess, sys as _s, tempfile, os as _o
    with tempfile.TemporaryDirectory() as d:
        open(_o.path.join(d, "program.py"), "w").write(program)
        open(_o.path.join(d, "test_program.py"), "w").write(E.TESTS)
        p = subprocess.run([_s.executable, "-m", "unittest", "test_program"], cwd=d,
                           capture_output=True, text=True, timeout=60)
    out = p.stdout + p.stderr
    failures, cur = [], None
    for line in out.splitlines():
        st = line.strip()
        if st.startswith("FAIL: ") or st.startswith("ERROR: "):
            kind, rest = st.split(": ", 1)
            name = rest.split()[0]
            region = ("producer" if "produce" in name else "transform" if "normalize" in name
                      else "aggregate" if "summarize" in name else "consumer")
            cur = {"test": name, "region": region, "kind": kind, "lines": []}
            failures.append(cur)
        elif cur is not None:
            if set(st) == {"-"} or set(st) == {"="}:
                continue
            if st.startswith("Ran ") or st.startswith("OK") or st.startswith("FAILED"):
                cur = None
            elif st:
                cur["lines"].append(st)
    for f in failures:
        body = [l for l in f.pop("lines") if not l.startswith("File \"") and not l.startswith("  ")]
        f["message"] = (body[-1] if body else "")[:200]
    green = (p.returncode == 0) and not failures
    return green, failures


def symptom_of(failures: list[dict]) -> tuple[str, str, str]:
    """(symptom, symptom_region, test_identity) from the MOST DOWNSTREAM failing test.

    The symptom carries the failure KIND and the ASSERTION DETAIL and NOT the test name. The test
    identity is returned separately so the gate can measure P1 without the agent ever seeing it.
    """
    fs = sorted(failures, key=lambda f: (-REGION_ORDER.index(f["region"])
                                         if f["region"] in REGION_ORDER else 99, f["test"]))
    first = fs[0]
    detail = (first.get("message") or "").strip()
    if not detail:
        detail = "no assertion detail"
    return f'{first["kind"]}: {detail}', first["region"], first["test"]


SITES = [(cls, r, f, x) for cls, entries in INJECTORS.items() for (r, f, x) in entries]


def make(idx: int, rng: random.Random):
    # UNIFORM OVER SITES, not over classes. Sampling a class first and then one of its sites gives
    # a one-site class the same share of the set as a four-site class, so the two classes whose
    # symptom pins its region exactly (KeyError 'avg', TypeError) would carry 40% of the tasks and
    # the set would fail its own entropy property. Site-uniform is also the more natural reading of
    # "a bug is equally likely anywhere" -- but it is a CHOICE that moves the gate numbers, so it is
    # written here rather than buried: class-uniform gives H = 0.74 bits and a 73% lookup, both
    # failing; site-uniform gives the numbers in _gate.json.
    cls, region, find, repl = rng.choice(SITES)
    program = E.assemble({region: build(cls, region, find, repl)})
    green, failures = run_tests_v2(program)
    if green or not failures:
        return None                      # this (class, region) is not observable; the gate counts it
    sym, sym_region, test = symptom_of(failures)
    return {"task_id": f"task_{idx:04d}", "family": "D_mixed", "bug_class": cls,
            "bug_region": region, "program": program, "tests": E.TESTS, "symptom": sym,
            "symptom_region": sym_region, "dead_paths": [], "seed": rng.randint(0, 10**9),
            "regions": REGION_ORDER, "_test_identity": test}


def measure(tasks: list[dict]) -> dict:
    by_test = collections.defaultdict(set)
    by_sym = collections.defaultdict(collections.Counter)
    for t in tasks:
        by_test[t["_test_identity"]].add(t["bug_region"])
        by_sym[t["symptom"]][t["bug_region"]] += 1

    # P2: mean conditional entropy, weighted by how often each symptom occurs
    n = len(tasks); H = 0.0
    for sym, c in by_sym.items():
        tot = sum(c.values())
        h = -sum((v / tot) * math.log2(v / tot) for v in c.values())
        H += (tot / n) * h

    # P3: HELD-OUT lookup table. An in-sample table scores 100% on ANY set -- it would have passed
    # v1 -- so the split is what makes this a test rather than a restatement.
    rng = random.Random(3); idx = list(range(n)); rng.shuffle(idx)
    half = n // 2
    tab = collections.defaultdict(collections.Counter)
    for i in idx[:half]:
        tab[tasks[i]["symptom"]][tasks[i]["bug_region"]] += 1
    tab = {k: v.most_common(1)[0][0] for k, v in tab.items()}
    hit = unseen = 0
    for i in idx[half:]:
        t = tasks[i]
        if t["symptom"] not in tab:
            unseen += 1
        elif tab[t["symptom"]] == t["bug_region"]:
            hit += 1
    held = len(idx) - half
    return {"n": n,
            "P1_tests_reachable_from_1_region": sorted(k for k, v in by_test.items() if len(v) < 2),
            "P1_test_region_counts": {k: sorted(v) for k, v in sorted(by_test.items())},
            "P2_mean_conditional_entropy_bits": round(H, 3),
            "P3_heldout_lookup_pct": round(100 * hit / held, 1),
            "P3_symptoms_unseen_in_train": unseen,
            "distinct_symptoms": len(by_sym),
            "bug_region_distribution": dict(collections.Counter(t["bug_region"] for t in tasks)),
            "bug_class_distribution": dict(collections.Counter(t["bug_class"] for t in tasks)),
            "red_herring_pct": round(100 * sum(t["symptom_region"] != t["bug_region"]
                                               for t in tasks) / n, 1)}


def verdict(m: dict) -> list[str]:
    bad = []
    if m["P1_tests_reachable_from_1_region"]:
        bad.append(f"P1 FAIL: {len(m['P1_tests_reachable_from_1_region'])} test identities reachable "
                   f"from one region only: {m['P1_tests_reachable_from_1_region'][:6]}")
    if m["P2_mean_conditional_entropy_bits"] < 1.0:
        bad.append(f"P2 FAIL: mean H(region | symptom) = {m['P2_mean_conditional_entropy_bits']} bits, need >= 1.0")
    if m["P3_heldout_lookup_pct"] > 60.0:
        bad.append(f"P3 FAIL: held-out lookup table scores {m['P3_heldout_lookup_pct']}%, need <= 60")
    return bad


def generate(n: int, seed: int):
    rng = random.Random(seed)
    tasks, skipped = [], 0
    i = 0
    while len(tasks) < n:
        i += 1
        t = make(len(tasks) + 1, rng)
        if t is None:
            skipped += 1
            if skipped > 20 * n:
                raise RuntimeError("injectors are producing green programs; verifier gap")
            continue
        tasks.append(t)
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gate", "gen"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks, skipped = generate(a.n, a.seed)
    m = measure(tasks); m["green_programs_skipped"] = skipped
    bad = verdict(m)
    os.makedirs(a.out, exist_ok=True)
    json.dump(m, open(os.path.join(a.out, "_gate.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != "P1_test_region_counts"}, indent=1))
    for b in bad:
        print("  " + b, file=sys.stderr)
    if a.cmd == "gate":
        print(f"\n  GATE {'PASS' if not bad else 'FAIL'} — nothing written but _gate.json", file=sys.stderr)
        return 0 if not bad else 1
    if bad:
        print("\n  REFUSING to write tasks: the gate did not pass. A task set that fails its own "
              "pre-registered properties is v1 with different numbers.", file=sys.stderr)
        return 1
    for t in tasks:
        json.dump(t, open(os.path.join(a.out, t["task_id"] + ".json"), "w"), indent=1)
    json.dump({"n": len(tasks), "gate": m, "seed": a.seed},
              open(os.path.join(a.out, "_stats.json"), "w"), indent=1)
    print(f"\n  wrote {len(tasks)} tasks to {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
