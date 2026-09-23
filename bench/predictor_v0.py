#!/usr/bin/env python
"""WO-312 — PREDICTOR v0: top-N next actions from state + last exchange. A function the renderer calls.

理 §3: "沙汰 — predictor v0 (the frontier model on Bedrock, top-N next actions from state +
last exchange) as a function the renderer calls, by 09-26, so the PREDICTED arm exists the same day
the STATIC one does."

WHAT IT IS AND IS NOT. v0 is the CEILING, not the product: a frontier model asked, every turn, what
the user will do next. v1 is a read-only head fitted on logged trajectories (the VIII recipe) and the
GAP between them is the paper's frontier. So v0 is built to be measured against, which means its
cost per turn is a first-class output, not an afterthought.

THE CONTRACT the renderer codes against — deliberately small, so a v1 head can replace v0 without
the renderer changing:

    predict(state, last_exchange, vocabulary, n=5) -> Prediction
        .actions   list of at most n {kind, label, args} drawn ONLY from `vocabulary`
        .cost_usd  what this turn cost, metered
        .raw       the model's reply, kept for replay (kanna's rule: persist the trajectory)
        .dropped   actions the model proposed that were NOT in the vocabulary, counted not hidden

VOCABULARY IS ENFORCED HERE, NOT SUGGESTED. WO-312: "Nothing outside it can be shown." A model asked
for next actions will happily invent one, and a bench whose PREDICTED arm can show an action the
STATIC arm cannot is not measuring the head — it is measuring who had the bigger vocabulary. Every
proposal is checked against the fixed kinds and dropped if it is not one of them; the drop count
rides on every Prediction so hit@N can never be inflated by an action that could not be rendered.
"""
from __future__ import annotations
import json, os, re, sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "poetry"))
import api_rater as AR

# docs/127 + WO-264's fixed kinds, per WO-312 "the fixed vocabulary ... already decided in WO-264"
DEFAULT_KINDS = ("select", "set", "add", "remove", "next", "back", "cancel", "confirm", "ask")

PROMPT = """You are predicting what a user will do NEXT in a booking interface.

THE VENUE — every id you name in args must come from here
{world}

CURRENT STATE
{state}

LAST EXCHANGE
{last}

You may ONLY choose from these action kinds: {kinds}

Reply with a JSON array of at most {n} objects, most likely first, nothing else:
[{{"kind": "<one of the kinds>", "label": "<short button text>", "args": {{}}}}]"""

NO_WORLD = ("(none supplied — name no ids; an action whose args name an offer, a person or a day "
            "that does not exist cannot be rendered as a tap)")

# Which arg keys carry an id that must resolve against the venue. An action proposing barber
# "Alex" at a shop staffed by Marcus, Priya and Dae-Ho is not a near miss: it is unshowable, the
# same failure as an out-of-vocabulary kind, and it is counted the same way and never shown.
ID_ARGS = {"offer_id": "offers", "offer": "offers", "service": "offers",
           "staff": "staff", "staff_id": "staff", "barber": "staff", "stylist": "staff",
           "day": "days", "weekday": "days"}


WEEK = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def world_from_space(space):
    """The compiled venue reduced to what a predictor may name. niwa's fixture shape."""
    sp = space.get("space", {})
    offers = [{"offer_id": o.get("offer_id"), "name": o.get("name"),
               "price": (o.get("price") or {}).get("amount"),
               "duration_min": o.get("duration_min")} for o in space.get("offer_sheet", [])]
    tx = space.get("transaction", {})
    roster = ((tx.get("resource") or {}).get("roster")) or []
    hours = (((tx.get("schedule") or {}).get("declared") or {}).get("hours")) or sp.get("hours") or {}
    return {"handle": sp.get("handle"), "name": sp.get("name"), "flow": tx.get("flow"),
            # WEEKDAY ORDER, not alphabetical: sorted() puts Friday first, and a head that offers
            # Fri/Sat/Sun on a Tuesday is wrong for a reason that has nothing to do with the model.
            "offers": offers, "staff": list(roster),
            "days": sorted(hours.keys(), key=lambda d: WEEK.index(d[:3].lower())
                           if d[:3].lower() in WEEK else 9),
            "hours": hours, "confirm": tx.get("confirm"), "identity_min": tx.get("identity_min")}


def _resolvable(world):
    """The id sets an action's args may name, lowercased."""
    if not world:
        return None
    return {"offers": {str(o.get("offer_id", "")).lower() for o in world.get("offers", [])}
                      | {str(o.get("name", "")).lower() for o in world.get("offers", [])},
            "staff": {str(x).lower() for x in world.get("staff", [])},
            "days": {str(x).lower() for x in world.get("days", [])}}



# WHAT MUST NEVER LEAVE THE BOX. The predictor serialises the UiState into the prompt, and the real
# UiState carries `form.book.name`, `form.book.phone`, `form.book.sms_opt_in` and `ask.draft`.
# Measured on the real mid-flow state: all four reached the Bedrock prompt verbatim. The predictor
# needs the SHAPE of the state — is the form filled, which offer/staff/day/slot are chosen — and
# never the identity itself. `last_exchange` stays: it is the deliberate, named user-originated
# input the whole mechanism is about, and it is stated as such in docs/wo318-data-scope-sautee.md.
REDACTED = "<redacted>"
_IDENTITY_FIELDS = ("name", "phone", "email", "sms_opt_in", "note")


def redact_state(state):
    """Return a copy with identity VALUES replaced by a filled/empty marker. Shape is preserved so
    the model can still tell a filled form from an empty one."""
    import copy
    s = copy.deepcopy(state) if isinstance(state, dict) else state
    if not isinstance(s, dict):
        return s
    form = s.get("form")
    if isinstance(form, dict):
        for sheet, fields in list(form.items()):
            if isinstance(fields, dict):
                form[sheet] = {k: (REDACTED if (k in _IDENTITY_FIELDS and v not in (None, "", False))
                                   else ("" if k in _IDENTITY_FIELDS else v))
                               for k, v in fields.items()}
    ask = s.get("ask")
    if isinstance(ask, dict) and ask.get("draft"):
        ask["draft"] = REDACTED
    return s


@dataclass
class Prediction:
    actions: list = field(default_factory=list)
    cost_usd: float = 0.0
    raw: str = ""
    dropped: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)   # in-vocabulary, but names an id the venue lacks
    error: str = ""


def _parse(text, kinds, n, ids=None):
    """Extract the array. Returns (kept, dropped, unresolved); unparseable is ([], [], [])."""
    if not text:
        return [], [], []
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j <= i:
        return [], [], []
    try:
        arr = json.loads(text[i:j + 1])
    except Exception:
        return [], [], []
    kept, dropped, unresolved = [], [], []
    for a in arr if isinstance(arr, list) else []:
        if not isinstance(a, dict):
            continue
        k = str(a.get("kind", "")).strip().lower()
        args = a.get("args") or {}
        item = {"kind": k, "label": str(a.get("label", ""))[:80], "args": args}
        if k not in kinds:
            dropped.append(item)
            continue
        bad = None
        if ids and isinstance(args, dict):
            for key, bucket in ID_ARGS.items():
                if key in args and args[key] not in (None, ""):
                    if str(args[key]).lower() not in ids.get(bucket, set()):
                        bad = "%s=%r not in %s" % (key, args[key], bucket)
                        break
        if bad:
            unresolved.append({**item, "why": bad})
        else:
            kept.append(item)
    return kept[:n], dropped, unresolved


class PredictorV0:
    """Callable. Construct once per session; the renderer calls predict() per turn."""

    def __init__(self, model="us.anthropic.claude-opus-4-7", kinds=DEFAULT_KINDS,
                 transport=None, ledger=None, max_tokens=300, cap=None, world=None):
        self.model, self.kinds, self.max_tokens = model, tuple(kinds), max_tokens
        self.world, self.ids = world, _resolvable(world)
        self.meter = AR.CostMeter(ledger=ledger or os.path.join(HERE, "bench_spend_ledger.json"),
                                  **({"cap": cap} if cap else {}))
        self.transport = transport or (lambda req, **kw: AR.bedrock_transport(req))
        self.turns = 0

    def predict(self, state, last_exchange="", n=5) -> Prediction:
        prompt = PROMPT.format(world=(json.dumps(self.world, ensure_ascii=False)[:2000]
                                      if self.world else NO_WORLD),
                               state=json.dumps(redact_state(state), ensure_ascii=False)[:4000],
                               last=str(last_exchange)[:1000],
                               kinds=", ".join(self.kinds), n=n)
        est_in = max(1, len(prompt) // 4)
        try:
            rid = self.meter.reserve(self.model, est_in, self.max_tokens, note="bench turn")
        except AR.BudgetExceeded as e:
            return Prediction(error=str(e))
        try:
            r = self.transport({"model": self.model, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": prompt}]})
        except Exception as e:
            return Prediction(error="%s: %s" % (type(e).__name__, str(e)[:160]))
        u = r.get("usage", {})
        cost = self.meter.settle(rid, u.get("input_tokens", est_in),
                                 u.get("output_tokens", self.max_tokens))
        text = "".join(c.get("text", "") for c in r.get("content", []))
        kept, dropped, unresolved = _parse(text, set(self.kinds), n, self.ids)
        self.turns += 1
        return Prediction(actions=kept, cost_usd=cost, raw=text, dropped=dropped,
                          unresolved=unresolved,
                          error="" if kept else "no showable action parsed")


# The REAL open state, generated by running initialState() in the bench's uiState.ts and sent by
# 形 (the record) — not a shape I invented. The first version of this file tested against
# {"screen": "salon", "service": None, ...}, which no renderer ever produces.
REAL_STATE = {
    "space": "quick-cuts.chelsea", "audience": "customer", "tab": "overview", "view": None,
    "selection": {"offer_ids": [], "staff": None, "day": None, "slot": None,
                  "thread": None, "cell": None},
    "filters": {}, "form": {}, "sheet": "none", "highlight": None,
    "runtime_sections": {}, "return_to": None, "submit_key": None,
    "ask": {"draft": "", "reply": None, "pending": False, "error": None, "chips": [], "images": []},
}

# niwa's compiled fixture, reduced to the keys world_from_space reads (source of truth:
# the estate repo reports/niwa/wo312-bench-fixture-2026-09-22/quick-cuts-barbershop_space.json).
REAL_SPACE = {
    "space": {"handle": "quick-cuts.chelsea", "name": "Quick Cuts Barbershop",
              "hours": {"tue": "09:00-18:00", "wed": "09:00-18:00", "thu": "09:00-18:00",
                        "fri": "09:00-18:00", "sat": "09:00-18:00", "sun": "09:00-18:00"}},
    "offer_sheet": [{"name": "Haircut", "offer_id": "haircut", "kind": "service",
                     "price": {"amount": 35, "currency": "USD"}, "duration_min": 30}],
    "transaction": {"flow": "booking", "confirm": "manual", "identity_min": "named",
                    "resource": {"kind": "staff", "select": "customer",
                                 "roster": ["Marcus", "Priya", "Dae-Ho"]},
                    "schedule": {"granularity": "slot", "duration_min": 30, "source": "declared",
                                 "declared": {"slot_grid_min": 30,
                                              "hours": {"tue": [9, 18], "wed": [9, 18],
                                                        "thu": [9, 18], "fri": [9, 18],
                                                        "sat": [9, 18], "sun": [9, 18]}}}},
}


def _selftest():
    """Every path, with a fake transport. No network, no spend."""
    state = REAL_STATE
    good = json.dumps([{"kind": "select", "label": "Choose service", "args": {}},
                       {"kind": "ask", "label": "Ask a question", "args": {}}])
    # 1 — the happy path
    p = PredictorV0(transport=lambda req, **kw: AR.bedrock_transport(req, client=AR.FakeBedrock([good])),
                    ledger="/tmp/_bench_t1.json").predict(state, "user: I want a haircut")
    assert [a["kind"] for a in p.actions] == ["select", "ask"], p.actions
    assert p.cost_usd > 0 and not p.dropped, p
    print("  [1] parses, meters (%.6f usd), no drops" % p.cost_usd)
    # 2 — OUT-OF-VOCABULARY actions are dropped and COUNTED, never shown
    bad = json.dumps([{"kind": "teleport", "label": "Teleport"}, {"kind": "confirm", "label": "Confirm"}])
    p2 = PredictorV0(transport=lambda req, **kw: AR.bedrock_transport(req, client=AR.FakeBedrock([bad])),
                     ledger="/tmp/_bench_t2.json").predict(state)
    assert [a["kind"] for a in p2.actions] == ["confirm"] and len(p2.dropped) == 1
    print("  [2] out-of-vocabulary dropped and counted: kept %s, dropped %s"
          % ([a["kind"] for a in p2.actions], [d["kind"] for d in p2.dropped]))
    # 3 — unparseable reply is an ERROR, never an empty head silently
    p3 = PredictorV0(transport=lambda req, **kw: AR.bedrock_transport(req, client=AR.FakeBedrock(["sorry"])),
                     ledger="/tmp/_bench_t3.json").predict(state)
    assert p3.actions == [] and p3.error
    print("  [3] unparseable -> error %r, not a silent empty head" % p3.error)
    # 4 — the cap REFUSES and sends nothing
    pr = PredictorV0(transport=lambda req, **kw: (_ for _ in ()).throw(AssertionError("sent!")),
                     ledger="/tmp/_bench_t4.json", cap=0.0000001)
    p4 = pr.predict(state)
    assert p4.error.startswith("REFUSED"), p4.error
    print("  [4] cap refuses before the call: %s" % p4.error.split(".")[0])
    # 5 — n is honoured
    many = json.dumps([{"kind": "next", "label": str(i)} for i in range(9)])
    p5 = PredictorV0(transport=lambda req, **kw: AR.bedrock_transport(req, client=AR.FakeBedrock([many])),
                     ledger="/tmp/_bench_t5.json").predict(state, n=3)
    assert len(p5.actions) == 3
    print("  [5] n honoured: asked 3 of 9 proposed, got %d" % len(p5.actions))
    # 6 — the venue extractor reads niwa's compiled shape
    w = world_from_space(REAL_SPACE)
    assert w["staff"] == ["Marcus", "Priya", "Dae-Ho"], w
    assert [o["offer_id"] for o in w["offers"]] == ["haircut"] and w["offers"][0]["price"] == 35, w
    assert w["days"] == ["tue", "wed", "thu", "fri", "sat", "sun"], w["days"]
    print("  [6] world_from_space: %s, %d offer(s), %d staff, %d open days"
          % (w["handle"], len(w["offers"]), len(w["staff"]), len(w["days"])))
    # 7 — an id the VENUE does not have is unshowable and is counted, never shown. A barber named
    #     Alex is not a near miss; it is the same failure as an out-of-vocabulary kind.
    ghost = json.dumps([{"kind": "select", "label": "Book with Alex", "args": {"barber": "Alex"}},
                        {"kind": "select", "label": "Book with Priya", "args": {"barber": "Priya"}},
                        {"kind": "select", "label": "Haircut", "args": {"offer_id": "haircut"}}])
    p6 = PredictorV0(transport=lambda req, **kw: AR.bedrock_transport(req, client=AR.FakeBedrock([ghost])),
                     ledger="/tmp/_bench_t6.json", world=w).predict(state)
    assert [a["label"] for a in p6.actions] == ["Book with Priya", "Haircut"], p6.actions
    assert len(p6.unresolved) == 1 and "Alex" in p6.unresolved[0]["why"], p6.unresolved
    print("  [7] out-of-VENUE args held back and counted: shown %d, unresolved %s"
          % (len(p6.actions), [u["why"] for u in p6.unresolved]))
    # 8 — with no world, ids are not checked (the old behaviour, unchanged)
    p7 = PredictorV0(transport=lambda req, **kw: AR.bedrock_transport(req, client=AR.FakeBedrock([ghost])),
                     ledger="/tmp/_bench_t7.json").predict(state)
    assert len(p7.actions) == 3 and not p7.unresolved
    print("  [8] no world supplied -> no id check, 3 shown (unchanged behaviour)")
    # 9 — THE IDENTITY NEVER REACHES THE PROMPT. This failed before the redaction existed: name,
    #     phone, sms_opt_in and the typed draft all went to Bedrock verbatim.
    filled = {**REAL_STATE,
              "selection": {**REAL_STATE["selection"], "offer_ids": ["haircut"], "staff": "Marcus",
                            "day": "2026-09-23", "slot": "14:30"},
              "sheet": "book",
              "form": {"book": {"name": "Jane Doe", "phone": "5551234567", "sms_opt_in": True}},
              "ask": {**REAL_STATE["ask"], "draft": "can you fit me in earlier"}}
    sent = {}
    def _capture(req, **kw):
        sent["prompt"] = req["messages"][0]["content"]
        return AR.bedrock_transport(req, client=AR.FakeBedrock([good]))
    PredictorV0(transport=_capture, ledger="/tmp/_bench_t9.json", world=w).predict(filled, "user: hi")
    leaked = [n for n in ("Jane Doe", "5551234567", "can you fit me in earlier") if n in sent["prompt"]]
    assert not leaked, "IDENTITY LEAKED TO THE PROMPT: %s" % leaked
    assert '"staff": "Marcus"' in sent["prompt"] and '"slot": "14:30"' in sent["prompt"], "shape lost"
    assert REDACTED in sent["prompt"], "the filled/empty marker is gone — the model cannot tell"
    print("  [9] identity redacted: name/phone/draft absent, selection shape intact, marker present")
    for i in range(1, 10):
        for s in ("", ".tmp"):
            f = "/tmp/_bench_t%d.json%s" % (i, s)
            if os.path.exists(f):
                os.remove(f)
    print("\n  SELFTEST PASSED — 9 checks, $0.00, no network.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
