#!/usr/bin/env python
"""WO-318 — the local-vs-Bedrock `/area` measurement. Pre-registered: docs/area-classifier-prereg.md

WHAT IT DECIDES, AND WHY THERE ARE THREE CRITERIA AND NOT TWO (理 §2(c), 令 R464 §7):

    1. replicate disagreement  — is the classifier PREDICTABLE, arm by arm
    2. latency                 — median AND p90, never a mean
    3. whether the sentence leaves the estate — the local model sends nothing anywhere;
       every utterance routed to Bedrock becomes a permanently unerasable record of something
       a person typed, and that price is paid by somebody who is not in the room

Criterion 3 is not measured here because it is not a measurement: it is a property of the arm, and
it is printed in the artifact beside the numbers so the decision is never made from the numbers
alone.

⛔ THIS WRITES NOTHING TO THE NARROWING LOG (鉋, 理: "a measurement never writes to the thing
it measures"). Its calls are the instrument; the narrowing log is the record of the bench. If these
few hundred calls landed in that store, every later count over it would be a count of my instrument
plus the bench with nothing in the row to separate them. Own artifact, own ledger, nothing else.

WHAT IS SENT: the utterance and the venue's tag ids, and nothing else — AreaV0's signature is the
enforcement, not this docstring.

THE BASE RATE IS PRINTED BESIDE EVERY AGREEMENT NUMBER. An agreement rate is not a signal until you
know what the arms would agree at by saying the same thing every time; I have published an 85.4%
agreement that WAS the 85.4% base rate, on an agent that had answered "false" 568 times out of 568.
So `pick_distribution` sits next to `cross_arm_agreement` in the artifact, always.

COVERAGE is the EMPTY-TAGS rate on utterances gold does NOT mark off-menu (prereg §2c, amended
2026-09-23). The unresolved-tag rate is nearly vacuous under a constrained prompt — it was zero in
every live call — because the prompt shows the model only that venue's tags. The coverage gap
surfaces as `tags: []`, and only gold's off-menu label separates "correctly off-menu" from "the
taxonomy is too coarse to say what they meant".
"""
import argparse, collections, json, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from area_equal import make_comparator, canonical, shape_of, load_parents   # noqa: E402
from replicate_runner import run_set                                        # noqa: E402


def load_utterances(path):
    """JSONL: {utterance, venue, kind?, off_menu?}. `kind` is gold's five (§8.2)."""
    rows = []
    for i, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        for k in ("utterance", "venue"):
            if k not in r:
                sys.exit("STOP: row %d has no %r. Every row needs an utterance and the venue whose "
                         "tag set it is classified against." % (i, k))
        rows.append(r)
    if not rows:
        sys.exit("STOP: %s has no rows" % path)
    return rows


def arm_run(rows, classifier, n, parents, label, cost_is_zero):
    """One arm over the whole set. Returns the replicate report plus this arm's own fields."""
    costs, errors = [], collections.Counter()
    by_utt = {}

    def classify_factory(venue, key):
        def classify(_u):
            t0 = time.time()
            r = classifier.classify(_u, venue)
            ms = (time.time() - t0) * 1000.0
            costs.append(r.cost_usd)
            if r.error:
                errors[r.error[:60]] += 1
            by_utt.setdefault(key, []).append({"ms": round(ms), "unresolved": r.unresolved})
            return (r.area or None), r.raw
        return classify

    same = make_comparator(parents)
    per, lat = [], []
    for idx, row in enumerate(rows):
        key = "%d" % idx
        rep = run_set([row["utterance"]], classify_factory(row["venue"], key), n=n,
                      comparator=same, label="%s:%d" % (label, idx))
        one = rep["per_utterance"][0]
        one.update({"venue": row["venue"], "kind": row.get("kind"),
                    "off_menu": bool(row.get("off_menu", row.get("kind") == "off-menu")),
                    "ms": [d["ms"] for d in by_utt.get(key, [])]})
        per.append(one)
        lat += one["seconds"]

    scored = [r for r in per if all(x["area"] is not None for x in r["replicates"])]
    unparsed = len(per) - len(scored)
    rate = None if unparsed else round(1 - sum(r["unanimous"] for r in scored) / len(scored), 4)

    # PER FIELD, because the prereg registered them separately (§2 measurements 1 and 2) and
    # because the combined rate cannot say WHICH field moved. The rehearsal's two disagreements
    # were on `intent` alone, on the two kinds gold calls hardest, with tags and commit unanimous
    # in all thirty replies — a fact the single number hides completely.
    def field_rate(fn):
        if unparsed:
            return None
        moved = 0
        for r in scored:
            vals = {fn(x["area"]) for x in r["replicates"]}
            moved += len(vals) > 1
        return round(moved / len(scored), 4)

    per_field = {
        "intent": field_rate(lambda A: A.get("intent")),
        "tags": field_rate(lambda A: tuple(sorted(str(t) for t in (A.get("tags") or [])))),
        "tags_canonical": field_rate(
            lambda A: tuple(__import__("area_equal").canonical_tags(A.get("tags"), parents))),
        "commit": field_rate(lambda A: repr(A.get("commit"))),
        "note": ("each is the fraction of utterances where that field alone was not identical "
                 "across all N replicates. `tags` is the raw set; `tags_canonical` applies 案内's "
                 "subsumption reduce first, so the difference between them is formatting."),
    }
    lat_sorted = sorted(lat)

    firsts = [r["replicates"][0]["area"] for r in per]
    dist = collections.Counter((a or {}).get("intent", "<none>") for a in firsts)
    commits = collections.Counter(str((a or {}).get("commit")) for a in firsts)
    on_menu = [r for r in per if not r["off_menu"] and r["replicates"][0]["area"] is not None]
    empty_on_menu = sum(1 for r in on_menu if not r["replicates"][0]["area"].get("tags"))
    unresolved_total = sum(len(d["unresolved"]) for v in by_utt.values() for d in v)

    return {
        "arm": label,
        "utterance_leaves_the_estate": not cost_is_zero,
        "n_utterances": len(per), "n_replicates": n,
        "disagreement_rate": rate,
        "disagreement_by_field": per_field,
        "unparsed_utterances": unparsed,
        "disagreement_note": (None if rate is not None else
                              "NOT REPORTED: %d utterance(s) had a replicate that did not parse. "
                              "A rate over a set with an unparsed reply is a rate about the parser."
                              % unparsed),
        "latency_median_s": round(statistics.median(lat_sorted), 4),
        "latency_p90_s": round(lat_sorted[int(0.9 * (len(lat_sorted) - 1))], 4),
        "cost_usd_total": round(sum(costs), 6),
        "cost_usd_per_call": round(sum(costs) / max(1, len(costs)), 6),
        "pick_distribution_intent": dict(dist),
        "pick_distribution_commit": dict(commits),
        "coverage_empty_tags_on_menu": (None if not on_menu else
                                        round(empty_on_menu / len(on_menu), 4)),
        "coverage_denominator": len(on_menu),
        "coverage_note": ("the empty-tags rate on utterances gold does NOT mark off-menu (prereg "
                          "§2c). None when the set carries no off-menu labels — without them this "
                          "number conflates a coarse taxonomy with a correct refusal."),
        "unresolved_tags_total": unresolved_total,
        "unresolved_note": ("kept as a counter, NOT as the coverage measurement: a constrained "
                            "prompt shows the model only this venue's tags, so it rarely names one "
                            "the venue lacks"),
        "errors": dict(errors),
        "per_utterance": per,
    }


def cross_arm(a_rep, b_rep, parents):
    """Agreement between the arms on replicate 0. Refuses across shapes rather than coercing."""
    same = make_comparator(parents)
    n = agree = skipped = 0
    shapes = collections.Counter()
    for ra, rb in zip(a_rep["per_utterance"], b_rep["per_utterance"]):
        A, B = ra["replicates"][0]["area"], rb["replicates"][0]["area"]
        shapes[(shape_of(A), shape_of(B))] += 1
        if A is None or B is None:
            skipped += 1
            continue
        n += 1
        agree += bool(same(A, B))
    return {
        "compared_on": "replicate 0 of each arm, canonical form (案内 doc 239's rule)",
        "n_compared": n, "n_skipped_unparsed": skipped,
        "agreement": None if not n else round(agree / n, 4),
        "shapes_seen": {"%s|%s" % k: v for k, v in shapes.items()},
        "read_it_with": ("pick_distribution on BOTH arms. An agreement rate is not a signal until "
                         "you know the base rate — a constant classifier agrees with itself."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances", help="JSONL: {utterance, venue, kind?, off_menu?}")
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--taxonomy", default="")
    ap.add_argument("--local-model", default="")
    # THE ARMS CANNOT BE TEMPERATURE-MATCHED AND THAT IS A FACT, NOT AN OVERSIGHT: opus-4-7 on
    # Bedrock REJECTS `temperature` (poetry/api_rater.py:158, carried over from judge.py), so the
    # paid arm cannot be pinned at all. The local arm can. A local 0.0 against an unpinnable
    # Bedrock is therefore not a model comparison — it is a comparison of samplers — so the local
    # arm is run at 0.0 AND at a non-zero setting, and the artifact names which is which.
    ap.add_argument("--local-temperature", type=float, default=0.0)
    ap.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    ap.add_argument("--local-ledger", default=os.path.join(HERE, "wo318_local_ledger.json"))
    # A SEPARATE ledger from the live /area service's, on purpose, and a SUB-BUDGET rather than the
    # whole cap. The service holds wo318_spend_ledger.json open for 形's renderer; two processes
    # read-modify-writing one json file lose entries, and a lost entry is a cap that under-counts.
    # So the instrument meters itself against its own $2 sub-budget and the artifact prints BOTH
    # totals against WO-318's $10 — a split that is stated is a split that can be added up.
    ap.add_argument("--bedrock-ledger", default=os.path.join(HERE, "wo318_measure_ledger.json"))
    ap.add_argument("--bedrock-cap", type=float, default=2.0)
    ap.add_argument("--service-ledger", default=os.path.join(HERE, "wo318_spend_ledger.json"))
    ap.add_argument("--arms", default="local,bedrock")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    # THE TWO ARMS CANNOT SHARE AN INTERPRETER ON THIS BOX and that is a fact about the estate, not
    # a design choice: the local arm needs mlx_lm (the mlx venv, py3.9) and the Bedrock arm needs
    # boto3 (/usr/bin/python3), and neither venv has the other package. Installing into the shared
    # mlx venv while experiments are running on it is the class of move that has cost this lane a
    # day before. So each arm runs where it can run, writes its own artifact, and --merge joins
    # them offline. It also means a rerun of one arm never re-spends the other.
    ap.add_argument("--merge", nargs="+", metavar="ARM.json",
                    help="join per-arm artifacts and compute the cross-arm comparison offline")
    a = ap.parse_args()
    if a.merge:
        return _merge(a)
    if a.selftest:
        return _selftest()
    if not a.utterances:
        sys.exit("--utterances is required (or --selftest)")

    from area_v0 import AreaV0, load_taxonomy
    import area_local as AL
    tax = load_taxonomy(a.taxonomy or None)
    parents = load_parents(tax)
    rows = load_utterances(a.utterances)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    reports = {}
    for arm in arms:
        if arm == "local":
            cl = AL.local_area(tax, model_id=a.local_model or AL.DEFAULT_LOCAL_MODEL,
                               ledger=a.local_ledger, temperature=a.local_temperature)
            name = "local@T%g" % a.local_temperature
            reports[name] = arm_run(rows, cl, a.replicates, parents, name, True)
            reports[name]["sampler"] = {
                "temperature": a.local_temperature,
                "pinnable": True,
                "note": ("at temperature 0 a replicate disagreement of 0 is BY CONSTRUCTION and is "
                         "not evidence the model is more decisive than the paid arm — it is "
                         "evidence the sampler is pinned, which the paid arm cannot be.")}
            arm = name
        elif arm == "bedrock":
            cl = AreaV0(tax, model=a.bedrock_model, ledger=a.bedrock_ledger,
                        cap=a.bedrock_cap)
            reports["bedrock"] = arm_run(rows, cl, a.replicates, parents, "bedrock", False)
            reports["bedrock"]["sampler"] = {
                "temperature": None, "pinnable": False,
                "note": ("opus-4-7 on Bedrock REJECTS temperature, so this arm's sampling cannot "
                         "be fixed. Its replicate disagreement has a floor nobody can lower by "
                         "configuration; the local arm's does not.")}
        else:
            sys.exit("unknown arm %r" % arm)
        r = reports[arm]
        print("  %-8s disagreement %s  median %.2fs  p90 %.2fs  $%.4f  leaves-estate %s"
              % (arm, r["disagreement_rate"], r["latency_median_s"], r["latency_p90_s"],
                 r["cost_usd_total"], r["utterance_leaves_the_estate"]))

    out = {"document": "WO-318 — /area classifier measurement, local vs Bedrock",
           "prereg": "docs/area-classifier-prereg.md (§2c amended 2026-09-23)",
           "utterances": os.path.abspath(a.utterances), "n": len(rows),
           "replicates": a.replicates,
           "third_criterion": ("whether the user's sentence leaves the estate — 理 §2(c), "
                               "令 R464 §7. The local arm sends nothing anywhere. A Bedrock arm on "
                               "live traffic makes each utterance permanently unerasable, and that "
                               "needs a plain notice and Louay's acceptance per utterance, neither "
                               "of which this measurement supplies."),
           "writes_to_the_narrowing_log": False,
           "spend": _spend_rollup(a),
           "arms": reports}
    if len(reports) == 2:
        k = list(reports)
        out["cross_arm"] = cross_arm(reports[k[0]], reports[k[1]], parents)
        print("  cross-arm agreement: %s (n=%d)"
              % (out["cross_arm"]["agreement"], out["cross_arm"]["n_compared"]))
    path = a.out or os.path.join(HERE, "area_measure.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % path)


def _merge(a):
    from area_v0 import load_taxonomy
    tax = load_taxonomy(a.taxonomy or None)
    parents = load_parents(tax)
    arms, names = {}, []
    for path in a.merge:
        d = json.load(open(path))
        one = d["arms"] if "arms" in d else {d["arm"]: d}
        for k, v in one.items():
            if k in arms:
                sys.exit("STOP: two artifacts both carry arm %r — merging them would compare an "
                         "arm with itself" % k)
            arms[k] = v; names.append(k)
    sets = {tuple(r["utterance"] for r in v["per_utterance"]) for v in arms.values()}
    if len(sets) != 1:
        sys.exit("STOP: the arms did not run the same utterances, in the same order. A cross-arm "
                 "agreement over two different sets is not an agreement.")
    stale = [k for k, v in arms.items() if "disagreement_by_field" not in v or "sampler" not in v]
    if stale:
        sys.exit("STOP: arm(s) %s were written by an older instrument — they carry no "
                 "`disagreement_by_field` or no `sampler` block. Merging them would put a number "
                 "measured under one set of readings beside numbers measured under another, with "
                 "nothing in the artifact saying so. Re-run those arms." % ", ".join(sorted(stale)))
    ns = {v["n_replicates"] for v in arms.values()}
    out = {"document": "WO-318 — /area classifier measurement, arms merged",
           "prereg": "docs/area-classifier-prereg.md (§2c, §2d)",
           "merged_from": [os.path.abspath(p) for p in a.merge],
           "n": len(next(iter(sets))), "replicates": sorted(ns),
           "⚠️_replicate_counts": ("EQUAL" if len(ns) == 1 else
                                   "DIFFER across arms — the disagreement rates are not comparable"),
           "writes_to_the_narrowing_log": False,
           "spend": _spend_rollup(a), "arms": arms}
    if len(names) == 2:
        out["cross_arm"] = cross_arm(arms[names[0]], arms[names[1]], parents)
        print("  cross-arm agreement: %s (n=%d)"
              % (out["cross_arm"]["agreement"], out["cross_arm"]["n_compared"]))
    for k, v in arms.items():
        print("  %-8s disagreement %s  by-field %s  median %.2fs  $%.4f  leaves-estate %s"
              % (k, v["disagreement_rate"],
                 {f: v["disagreement_by_field"][f] for f in ("intent", "tags", "commit")},
                 v["latency_median_s"], v["cost_usd_total"], v["utterance_leaves_the_estate"]))
    path = a.out or os.path.join(HERE, "area_measure.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % path)


def _spend_rollup(a):
    """WO-318's $10 seen across BOTH ledgers, so the split never hides a total."""
    def tot(p):
        try:
            return float(json.load(open(p))["spent_usd"])
        except Exception:
            return 0.0
    inst, svc = tot(a.bedrock_ledger), tot(a.service_ledger)
    return {"instrument_ledger": os.path.basename(a.bedrock_ledger), "instrument_usd": round(inst, 6),
            "instrument_sub_budget_usd": a.bedrock_cap,
            "service_ledger": os.path.basename(a.service_ledger), "service_usd": round(svc, 6),
            "combined_usd": round(inst + svc, 6), "wo318_cap_usd": 10.0,
            "why_two_ledgers": ("the live /area service holds its ledger open for 形's renderer; "
                                "two processes read-modify-writing one json lose entries, and a "
                                "lost entry is a cap that under-counts. Both totals are printed.")}


def _selftest():
    """The whole driver on canned transports — no network, no model, no spend."""
    from area_v0 import AreaV0, load_taxonomy
    from area_local import FreeMeter
    tax = load_taxonomy()
    parents = load_parents(tax)

    def canned(seq):
        box = {"i": 0}
        def t(req, **kw):
            v = seq[box["i"] % len(seq)]; box["i"] += 1
            return {"content": [{"text": v}], "usage": {"input_tokens": 9, "output_tokens": 5}}
        return t

    stable = '{"intent":"hours","tags":["info.hours"],"commit":false}'
    wobble = '{"intent":"ask","tags":[],"commit":false}'
    rows = [{"utterance": "what time do you close", "venue": "quick-cuts.chelsea", "kind": "plain ask"},
            {"utterance": "can you resole my shoes", "venue": "quick-cuts.chelsea",
             "kind": "off-menu", "off_menu": True}]

    A = AreaV0(tax, model="canned-stable", transport=canned([stable]), meter=FreeMeter(None))
    rep_a = arm_run(rows, A, 3, parents, "local", True)
    assert rep_a["disagreement_rate"] == 0.0, rep_a["disagreement_rate"]
    print("  [1] a stable arm reads disagreement 0.0")

    B = AreaV0(tax, model="canned-wobbly",
               transport=canned([stable, wobble, stable, stable, stable, stable]),
               meter=FreeMeter(None))
    rep_b = arm_run(rows, B, 3, parents, "bedrock", False)
    assert rep_b["disagreement_rate"] == 0.5, rep_b["disagreement_rate"]
    print("  [2] one of two utterances wobbles -> 0.5, the canonical form did not hide it")

    assert rep_a["coverage_denominator"] == 1 and rep_a["coverage_empty_tags_on_menu"] == 0.0
    print("  [3] coverage counts only the ON-MENU utterance: denominator 1, empty-tags 0.0")

    assert rep_a["cost_usd_total"] == 0.0 and rep_a["utterance_leaves_the_estate"] is False
    assert rep_b["utterance_leaves_the_estate"] is True
    print("  [4] the third criterion is a property of the arm, recorded on both")

    x = cross_arm(rep_a, rep_b, parents)
    assert x["agreement"] == 1.0 and x["n_compared"] == 2
    print("  [5] cross-arm compares replicate 0: agreement %.1f on n=%d" % (x["agreement"], x["n_compared"]))
    assert rep_a["pick_distribution_intent"] == {"hours": 2}
    print("  [6] and the base rate is printed beside it: %s — both picks the same intent, which is "
          "WHY the 1.0 means nothing on its own" % rep_a["pick_distribution_intent"])

    bad = AreaV0(tax, model="canned-broken", transport=canned(["not json at all"]),
                 meter=FreeMeter(None))
    rep_c = arm_run(rows, bad, 2, parents, "broken", True)
    assert rep_c["disagreement_rate"] is None and rep_c["unparsed_utterances"] == 2
    print("  [7] every reply unparseable -> rate REFUSED, not computed over nothing")
    print("\n  SELFTEST PASSED — 7 checks, no network, no model, no spend.")


if __name__ == "__main__":
    main()
