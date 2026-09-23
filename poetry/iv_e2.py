#!/usr/bin/env python
"""ET-IV E2 — blind A/B rating of one pair per brief, by the frontier judge.

理's run order (11688): E2 -> 7d -> E5 -> E3 -> the 25% re-rate. This is E2.

WHAT IS REUSED AND WHAT IS NEW. The candidates come from poetry_gen.py's cache and are scored by
e1_score.score_sample -- the committed checkers, not a reimplementation of them. Pareto pruning,
pair construction, logging, q and self-consistency come from human_session.py unchanged. The rater
is api_rater.ApiRater and the $40 stop is its meter's. This file is the arm, not the machinery:
if it starts reimplementing either end, that is the bug.

  q                 = how often the VERIFIER's pick matches the JUDGE's pick
  self-consistency  = the judge against itself on a 25% subset, seed and order swapped, which is
                      the ceiling q is read against (amendment §1, the week-later re-rate's stand-in)

BUILD-ONLY WITHOUT --live. Without a key nothing is sent; --dry runs the whole arm on the fake
transport so the pipeline is exercised at $0.00.
"""
import argparse, collections, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_rater as AR
import human_session as HS

SELECTION = ["pareto"]          # set from --selection; named in every artifact this writes


def load_candidates(cache_path, prompts_path, limit=0):
    """poetry_gen.py's cache: one record per brief with `samples`. Scored by the COMMITTED checker."""
    if not os.path.exists(cache_path):
        raise SystemExit(
            "STOP: no candidate cache at %s.\n"
            "  E2 rates poems that must be generated first:\n"
            "    python poetry/poetry_gen.py --model <mlx model> --tag <tag> --nmax 16 \\\n"
            "        --temp 0.8 --max-tokens 96 --out %s\n"
            "  It runs on llm1's GPU and costs no API spend." % (cache_path, cache_path))
    import e1_score as E1
    briefs = {json.loads(l)["id"]: json.loads(l) for l in open(prompts_path)}
    out = []
    for line in open(cache_path):
        rec = json.loads(line)
        b = briefs.get(rec["id"])
        if b is None:
            continue
        cands = []
        for k, s in enumerate(rec["samples"]):
            valid, score, meter_valid, oov = E1.score_sample(s, rec)
            cands.append({"cand_id": "%s#%d" % (rec["id"], k), "text": s,
                          "model": rec.get("tag", "?"),
                          "scores": {"valid": int(bool(valid)), "meter": int(bool(meter_valid)),
                                     "score": float(score), "clean": -float(oov)}})
        if len(cands) >= 2:
            out.append({"brief_id": rec["id"], "brief": b["prompt"], "candidates": cands})
        if limit and len(out) >= limit:
            break
    return out


# 理: "the checker's composite `score` is dropped from E2's selection ... Selection =
# valid ∧ meter, then Pareto on the rest."
#
# `score` is e1_score's n_meter + rhyme_ok + mean(iamb)/10 — a COMPOSITE of what `valid` and
# `meter` already encode. Including it beside its own components made the checker's own ranking a
# domination axis, in an arm whose purpose is to test that ranking, and drove the Pareto set onto a
# single candidate in 53-67% of briefs at three policies.
DIMS = ["valid", "meter", "clean"]

# The ruling has two readable forms and they differ by up to 41 briefs, so both are implemented and
# the one used is named in the artifact rather than assumed:
#   "pareto"    single-stage Pareto over DIMS. The literal "drop `score`", and the form the ET-IV
#               amendment's own words describe ("dominated on EVERY craft/taste dimension").
#   "two-stage" Pareto on (valid, meter) first, then Pareto on the rest among the survivors.
# Measured, 7B: pareto n=190 vs two-stage n=149; 3B 210/193; 14B 203/182.
CRAFT = ["valid", "meter"]
REST = ["clean"]


def select_pareto(candidates, mode="pareto"):
    if mode == "two-stage":
        return HS.pareto_prune(HS.pareto_prune(candidates, CRAFT), REST)
    return HS.pareto_prune(candidates, DIMS)


def selectors_for(pair, pareto, rng):
    """Every selector's pick on THIS pair, computed locally — no extra API calls.

    The registered E2 (proposal §E2) compares selection rules: random, a convention/checker
    baseline, and a judge, with q = selector-vs-rater agreement and "the lift of judge selection
    over no-selection". Under amendment 2 the frontier model IS the rater, so a frontier-judge
    SELECTOR would agree with itself trivially and is not among these. Random is the no-selection
    floor that the checker's q has to beat to mean anything.
    """
    cands = {c["cand_id"]: c for c in pareto}
    two = [pair["A"]["cand_id"], pair["B"]["cand_id"]]
    def best(key):
        return max(two, key=lambda cid: key(cands[cid]["scores"]))
    return {
        "random": rng.choice(two),
        "checker_full": best(lambda s: (s["valid"], s["meter"], s["score"], s["clean"])),
        "checker_score_only": best(lambda s: s["score"]),
        "meter_only": best(lambda s: (s["meter"], s["score"])),
    }


def build_pair(item, rng):
    """One blind A/B pair per brief: the VERIFIER's pick against another Pareto-surviving candidate.

    Pairing the verifier's pick against a DOMINATED candidate would measure whether the judge can
    see form breakage, which the checker already answers. The pair is drawn from the Pareto set so
    the comparison is between candidates the checker cannot separate.
    """
    pareto = select_pareto(item["candidates"], SELECTION[0])
    if len(pareto) < 2:
        return None, None, None
    by = {c["cand_id"]: c for c in pareto}
    sel = max(pareto, key=lambda c: (c["scores"]["valid"], c["scores"]["meter"],
                                     c["scores"]["score"], c["scores"]["clean"]))
    others = [c for c in pareto if c["cand_id"] != sel["cand_id"]]
    pair = HS.make_ab_pairs([sel, rng.choice(others)], rng)[0]
    return pair, sel["cand_id"], pareto


def run(items, rater, rng, brief_of):
    log, selector = [], collections.defaultdict(dict)
    collapsed = 0
    for it in items:
        pair, sel, pareto = build_pair(it, rng)
        if pair is None:
            collapsed += 1     # Pareto set of size 1: nothing for the judge to compare
            continue
        brief_of[pair["pair_id"]] = it["brief"]
        w = rater(pair, brief=it["brief"])
        if w is None:
            continue                      # dropped, counted in rater.unparsed, never guessed
        for name, pick in selectors_for(pair, pareto, rng).items():
            selector[name][pair["pair_id"]] = pick
        log.append(HS.log_choice(pair, w, ts=len(log)))
        log[-1]["_pair"] = pair
    return log, selector, collapsed


class _FixedPref:
    """Perfectly consistent, no position bias: always prefers the lower-sorting cand_id."""
    def __init__(self): self.unparsed = 0
    def __call__(self, pair, brief=None):
        return "A" if pair["A"]["cand_id"] < pair["B"]["cand_id"] else "B"


class _AlwaysA:
    """Pure position bias: always picks whatever is shown first."""
    def __init__(self): self.unparsed = 0
    def __call__(self, pair, brief=None): return "A"


def _selftest():
    """The order-swap bookkeeping, checked at BOTH extremes.

    Re-rating swaps A and B, so the same CANDIDATE comes back as the other letter and the winner
    has to be translated before comparing. If that translation is inverted, a perfectly consistent
    judge scores 0.0 and the arm reports "the judge disagrees with itself half the time" as a
    finding about the model. One-sided testing cannot catch it: a rater that scores 1.0 under a
    correct mapping scores 0.0 under an inverted one, and vice versa, so both ends are required.
    """
    rng = random.Random(0)
    cands = [{"cand_id": "c%02d" % i, "text": "p%d" % i, "model": "m",
              "scores": {"valid": i % 2, "meter": i % 3, "score": i * 0.7 % 5, "clean": -(i % 4)}}
             for i in range(8)]
    pairs = HS.make_ab_pairs(select_pareto(cands, "pareto"), rng)
    assert pairs, "no pairs to test"
    for R, expect, label in ((_FixedPref(), 1.0, "consistent, no position bias"),
                             (_AlwaysA(), 0.0, "pure position bias")):
        log = []
        for i, p in enumerate(pairs):
            log.append(HS.log_choice(p, R(p), ts=i)); log[-1]["_pair"] = p
        rr = []
        for i, p in enumerate(pairs):
            w = R({"pair_id": p["pair_id"], "A": p["B"], "B": p["A"]})
            rr.append({"pair_id": p["pair_id"], "winner": "B" if w == "A" else "A", "ts": 1000 + i})
        got = HS.self_consistency(log, rr)
        assert got == expect, "order-swap bookkeeping is wrong: %s gave %s, expected %s" % (
            label, got, expect)
        print("  [swap] %-32s self-consistency %.1f  (expected %.1f)" % (label, got, expect))
    print("  SELFTEST PASSED — the swap maps candidates, not letters. $0.00 spent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="check the order-swap bookkeeping at both extremes; no model, no spend")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache", "e1_7B.jsonl"))
    ap.add_argument("--prompts", default=os.path.join(HERE, "data", "e1_prompts.jsonl"))
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--ledger", default=AR.DEFAULT_LEDGER)
    ap.add_argument("--out", default=os.path.join(HERE, "iv_e2_result.json"))
    ap.add_argument("--rerate-frac", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", choices=("pareto", "two-stage"), default="pareto",
                    help="how 理's 'valid AND meter, then Pareto on the rest' is applied; "
                         "the choice is recorded in the result file")
    ap.add_argument("--dry", action="store_true", help="fake transport, $0.00, exercises the arm")
    ap.add_argument("--dry-bedrock", action="store_true",
                    help="dry run through the REAL Bedrock code path with a fake boto client. The "
                         "generic --dry exercises the arm; this exercises the transport that will "
                         "actually spend — converse() shape, usage mapping, tool-use refusal — at "
                         "full scale, for $0.00.")
    ap.add_argument("--live", action="store_true", help="real calls; needs ANTHROPIC_API_KEY")
    a = ap.parse_args()
    SELECTION[0] = a.selection
    if a.selftest:
        _selftest(); return
    if a.dry_bedrock:
        a.dry = True
    if not (a.dry or a.live):
        raise SystemExit("STOP: pass --dry (fake transport, $0) or --live (real calls).")

    # the GO gate is checked before the first call, every run, from the artifact 理 approved
    planned = float(json.load(open(AR.COST_ARTIFACT))["total"]["cost_usd"])
    AR.check_plan(planned)

    items = load_candidates(a.cache, a.prompts, a.limit)
    print("  briefs with >= 2 candidates: %d" % len(items))
    rng = random.Random(a.seed)
    # 🔴 A DRY RUN MUST NOT CHARGE THE REAL LEDGER. The meter is deliberately real in --dry so the
    # accounting path is exercised, but it was writing to the SAME file the live cap reads: after a
    # handful of dry runs the ledger said $2.16 of $40 spent with zero calls ever sent, and --live
    # would eventually refuse with a message about a budget nothing had used. The cap must be hard
    # against real spend and blind to rehearsals.
    ledger = a.ledger if a.live else (os.path.splitext(a.ledger)[0] + ".DRY.json")
    if not a.live and os.path.exists(ledger):
        # A DRY LEDGER THAT ACCUMULATES ACROSS REHEARSALS ANSWERS THE WRONG QUESTION. The point of a
        # dry run is "what would THIS arm cost", and a carried-over total reads as this arm's cost
        # while being the sum of every rehearsal. Truncated per run; the live ledger never is.
        os.remove(ledger)
    meter = AR.CostMeter(ledger=ledger)
    if not a.live:
        print("  dry run: metering to a FRESH %s, NOT the live ledger"
              % os.path.basename(ledger))
    if a.live:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("STOP: --live but ANTHROPIC_API_KEY is not set. Nothing was sent.")
        transport = lambda req, **kw: AR.http_transport(req, api_key=key)
    elif a.dry_bedrock:
        fb = AR.FakeBedrock(["A", "B"])
        transport = lambda req, **kw: AR.bedrock_transport(req, client=fb)
        print("  dry run through the REAL bedrock transport (fake boto client)")
    else:
        transport = AR.FakeTransport(["A", "B"])
    rater = AR.ApiRater(meter, transport, model=a.model)

    brief_of = {}
    log, selector, collapsed = run(items, rater, rng, brief_of)
    qs = {name: HS.compute_q(sel, log) for name, sel in sorted(selector.items())}
    q = qs.get("checker_full")
    print("  rated %d pairs, %d unparsed and dropped   %s" % (len(log), rater.unparsed, meter.line()))
    for name, v in qs.items():
        print("    q(%-20s vs the judge) = %s%s" % (name, v,
              "   <- the no-selection floor" if name == "random" else ""))
    # A zero-match count printed AFTER the result is a zero-match count nobody reads. The first
    # dry run of this file rated 0 pairs and still wrote a result file with q = null.
    print("  briefs whose Pareto set collapsed to ONE candidate: %d of %d (%.1f%%)"
          % (collapsed, len(items), 100.0 * collapsed / max(1, len(items))))
    if len(log) < 0.5 * len(items):
        print("\n  *** REFUSING to call this an E2 result: %d of %d briefs produced a rated pair."
              % (len(log), len(items)))
        print("  *** A Pareto set collapses to one candidate when the checker's dimensions do not")
        print("  *** separate the samples. Check the candidate cache before reading anything here.")
        if not a.dry:
            raise SystemExit(2)

    # self-consistency: same pairs, order swapped and a different seed -- the ceiling q is read against
    sub = HS.rerate_subset([r["_pair"] for r in log], a.rerate_frac, random.Random(a.seed + 1)) if log else []
    rr = []
    rater2 = AR.ApiRater(meter, transport, model=a.model)
    for i, p in enumerate(sub):
        swapped = {"pair_id": p["pair_id"], "A": p["B"], "B": p["A"]}
        w = rater2(swapped, brief=brief_of.get(p["pair_id"], ""))
        if w is None:
            continue
        # the swap means the same CANDIDATE is now the other letter; compare candidates, not letters
        rr.append({"pair_id": p["pair_id"], "winner": "B" if w == "A" else "A", "ts": 1000 + i})
    sc = HS.self_consistency(log, rr)
    print("  re-rated %d pairs (order swapped)   self-consistency %s" % (len(rr), sc))

    json.dump({
        "document": "ET-IV E2 — blind A/B, one pair per brief, rated by the frontier judge",
        "mode": ("LIVE" if a.live else
                 "DRY-BEDROCK (real transport, fake boto client, $0.00 — not a result)"
                 if a.dry_bedrock else "DRY (fake transport, $0.00 — not a result)"),
        "judge": a.model, "briefs": len(items), "pairs_rated": len(log),
        "unparsed_dropped": rater.unparsed + rater2.unparsed,
        "q_verifier_vs_judge": q,
        "q_by_selector": qs,
        "how_to_read_q_by_selector": "random is the no-selection floor (proposal §E2's 'lift of "
                                     "judge selection over no-selection'). A checker q that does "
                                     "not beat random is a checker the judge does not agree with "
                                     "more than chance.",
        "briefs_with_no_pair_pareto_collapsed": collapsed,
        "rated_fraction": round(len(log) / max(1, len(items)), 3),
        "self_consistency": sc, "rerate_n": len(rr),
        "how_to_read_q": "q is read against self-consistency, not against 1.0: a judge that agrees "
                         "with itself %s of the time cannot agree with anything else more often."
                         % (sc if sc is not None else "<unmeasured>"),
        "selection_rule": {"mode": a.selection, "dims": DIMS,
            "ruling": "理 — the composite `score` is dropped from E2's selection",
            "note": "two readable forms; this run used %r. Measured difference at the 7B: "
                    "pareto n=190 vs two-stage n=149." % a.selection},
        "pair_construction": "the verifier's top pick against another PARETO-SURVIVING candidate — "
                             "pairing it against a dominated one would measure whether the judge "
                             "can see form breakage, which the checker already answers",
        "spend_usd_this_run": round(meter.spent, 4), "cap_usd": meter.cap,
        "calls_this_run": meter.state["calls"],
        "signed": "Sautee (sha-ta)"}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("  wrote %s" % a.out)


if __name__ == "__main__":
    main()
