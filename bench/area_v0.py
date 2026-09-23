#!/usr/bin/env python
"""WO-318 — utterance -> `area`, for the two-venue demo (理).

  classify(utterance, venue) -> Area
      .area        {intent, tags, commit, taxonomy}   — tags exist in THIS venue's set, always
      .unresolved  tags the model named that the venue lacks — DROPPED, counted, never shown
      .raw         the model's reply, kept for replay
      .cost_usd    metered
      .error       set when nothing parsed; the caller leaves `area` UNCHANGED (§3's rule)

THE MODEL COMPILES, IT NEVER EXECUTES (理). This returns a typed value; `visible` does the
work, deterministically, from the compiled tree. Nothing here renders, ranks or decides.

WHAT IS SENT: the utterance and the venue's tag ids. **No state, no prior turn, no identity, no
session history** — `docs/wo318-data-scope-sautee.md` §1, and it is enforced by this signature
rather than promised in prose: there is no parameter through which any of it could arrive.

UNRESOLVED IS A MEASUREMENT, NOT AN ERROR PATH (prereg §2b): a tag outside the venue's set is
dropped and COUNTED, because a high rate is a finding about the taxonomy's coverage, not about the
user asking for something that does not exist.
"""
import argparse, json, os, sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "poetry"))
import api_rater as AR

INTENTS = ("browse", "ask", "book", "order", "hours", "directions", "contact", "cancel")

PROMPT = """You turn one sentence from a customer into a typed value. You never answer the customer.

THE VENUE'S TAGS — the ONLY tags you may use:
{tags}

THE SENTENCE
{utterance}

Reply with ONE json object and nothing else:
{{"intent": "<one of: {intents}>", "tags": ["<tags from the list above>"], "commit": <true|false>}}

intent  what the sentence is for.
tags    what it is about. [] if the sentence names nothing in the list.
commit  true if the customer is CHOOSING something and the screen should narrow to it;
        false if they are ASKING and the screen should not move."""


@dataclass
class Area:
    area: dict = field(default_factory=dict)
    unresolved: list = field(default_factory=list)
    raw: str = ""
    cost_usd: float = 0.0
    error: str = ""


def load_taxonomy(path=None):
    return json.load(open(path or os.path.join(HERE, "taxonomy_demo.json")))


class AreaV0:
    def __init__(self, taxonomy=None, model="us.anthropic.claude-opus-4-7",
                 transport=None, ledger=None, max_tokens=120, cap=None, meter=None):
        self.tax = taxonomy or load_taxonomy()
        self.model, self.max_tokens = model, max_tokens
        # `meter` exists so the LOCAL arm of the replicate measurement runs through THIS class and
        # not a copy of it. A local 7B costs nothing and has no entry in the price card, so it needs
        # a meter that reserves and settles zero -- but it must share every other line of
        # classify(): the same prompt, the same json extraction, the same enum check, the same
        # kept/dropped split. A disagreement rate measured across two parsers is a measurement of
        # the parsers. See bench/area_local.py.
        self.meter = meter or AR.CostMeter(
            ledger=ledger or os.path.join(HERE, "wo318_spend_ledger.json"),
            **({"cap": cap} if cap else {}))
        self.transport = transport or (lambda req, **kw: AR.bedrock_transport(req))
        self.turns = 0

    def venue_tags(self, venue):
        v = self.tax["venues"].get(venue)
        if v is None:
            raise KeyError("no taxonomy for venue %r — the demo covers %s"
                           % (venue, sorted(self.tax["venues"])))
        return list(v["tags"])

    def classify(self, utterance, venue) -> Area:
        tags = self.venue_tags(venue)
        prompt = PROMPT.format(tags="\n".join("  " + t for t in tags),
                               utterance=str(utterance)[:500], intents=", ".join(INTENTS))
        est_in = max(1, len(prompt) // 4)
        try:
            rid = self.meter.reserve(self.model, est_in, self.max_tokens, note="wo318 /area")
        except AR.BudgetExceeded as e:
            return Area(error=str(e))
        try:
            r = self.transport({"model": self.model, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": prompt}]})
        except Exception as e:
            return Area(error="%s: %s" % (type(e).__name__, str(e)[:160]))
        u = r.get("usage", {})
        cost = self.meter.settle(rid, u.get("input_tokens", est_in), u.get("output_tokens", self.max_tokens))
        text = "".join(c.get("text", "") for c in r.get("content", []))
        self.turns += 1

        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            return Area(raw=text, cost_usd=cost, error="no json object in the reply")
        try:
            got = json.loads(text[i:j + 1])
        except Exception as e:
            return Area(raw=text, cost_usd=cost, error="unparseable json: %s" % str(e)[:80])

        intent = str(got.get("intent", "")).strip().lower()
        if intent not in INTENTS:
            return Area(raw=text, cost_usd=cost,
                        error="intent %r is not in the enum — area unchanged" % intent[:40])
        kept, dropped = [], []
        for t in (got.get("tags") or []):
            (kept if str(t) in tags else dropped).append(str(t))
        return Area(area={"intent": intent, "tags": kept, "commit": bool(got.get("commit", False)),
                          "taxonomy": self.tax["taxonomy"]},
                    unresolved=dropped, raw=text, cost_usd=cost)


def _selftest():
    tax = load_taxonomy()
    def fake(reply):
        return lambda req, **kw: AR.bedrock_transport(req, client=AR.FakeBedrock([reply]))
    a = AreaV0(tax, transport=fake('{"intent":"book","tags":["svc.hair.cut"],"commit":true}'),
               ledger="/tmp/_a1.json").classify("book me a haircut", "quick-cuts.chelsea")
    assert a.area["intent"] == "book" and a.area["tags"] == ["svc.hair.cut"] and a.area["commit"]
    print("  [1] commit parsed: %s" % a.area)
    b = AreaV0(tax, transport=fake('{"intent":"ask","tags":["svc.laser","svc.hair.cut"],"commit":false}'),
               ledger="/tmp/_a2.json").classify("do you do laser?", "quick-cuts.chelsea")
    assert b.area["tags"] == ["svc.hair.cut"] and b.unresolved == ["svc.laser"]
    print("  [2] tag the venue lacks DROPPED and counted: unresolved=%s" % b.unresolved)
    c = AreaV0(tax, transport=fake('{"intent":"teleport","tags":[],"commit":true}'),
               ledger="/tmp/_a3.json").classify("beam me up", "quick-cuts.chelsea")
    assert not c.area and c.error
    print("  [3] intent outside the enum -> no area, caller leaves it UNCHANGED: %r" % c.error[:44])
    d = AreaV0(tax, transport=fake("sorry, I can't"), ledger="/tmp/_a4.json").classify("hi", "quick-cuts.chelsea")
    assert not d.area and d.error
    print("  [4] unparseable -> error, never a silent empty area")
    try:
        AreaV0(tax, ledger="/tmp/_a5.json").classify("hi", "not-a-venue")
        print("  [5] FAILED"); sys.exit(1)
    except KeyError as e:
        print("  [5] unknown venue refused: %s" % str(e)[:56])
    e = AreaV0(tax, transport=fake('{"intent":"hours","tags":["info.hours"],"commit":false}'),
               ledger="/tmp/_a6.json").classify("when do you open", "fresh-fold-laundry.lower-east-side")
    assert e.area["tags"] == ["info.hours"]
    print("  [6] second venue, its own tag set: %s" % e.area)
    for i in range(1, 7):
        for s in ("", ".tmp"):
            p = "/tmp/_a%d.json%s" % (i, s)
            if os.path.exists(p): os.remove(p)
    print("\n  SELFTEST PASSED — 6 checks, no network, no spend.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--utterance"); ap.add_argument("--venue", default="quick-cuts.chelsea")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.utterance:
        r = AreaV0().classify(a.utterance, a.venue)
        print(json.dumps({"area": r.area, "unresolved": r.unresolved,
                          "cost_usd": round(r.cost_usd, 6), "error": r.error}, ensure_ascii=False))
    else:
        ap.error("--selftest or --utterance")
