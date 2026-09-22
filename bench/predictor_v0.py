#!/usr/bin/env python
"""WO-312 — PREDICTOR v0: top-N next actions from state + last exchange. A function the renderer calls.

理 12053 §3: "沙汰 — predictor v0 (the frontier model on Bedrock, top-N next actions from state +
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

CURRENT STATE
{state}

LAST EXCHANGE
{last}

You may ONLY choose from these action kinds: {kinds}

Reply with a JSON array of at most {n} objects, most likely first, nothing else:
[{{"kind": "<one of the kinds>", "label": "<short button text>", "args": {{}}}}]"""


@dataclass
class Prediction:
    actions: list = field(default_factory=list)
    cost_usd: float = 0.0
    raw: str = ""
    dropped: list = field(default_factory=list)
    error: str = ""


def _parse(text, kinds, n):
    """Extract the array. Returns (kept, dropped). Unparseable is ([], []) and the caller reports it."""
    if not text:
        return [], []
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j <= i:
        return [], []
    try:
        arr = json.loads(text[i:j + 1])
    except Exception:
        return [], []
    kept, dropped = [], []
    for a in arr if isinstance(arr, list) else []:
        if not isinstance(a, dict):
            continue
        k = str(a.get("kind", "")).strip().lower()
        item = {"kind": k, "label": str(a.get("label", ""))[:80], "args": a.get("args") or {}}
        (kept if k in kinds else dropped).append(item)
    return kept[:n], dropped


class PredictorV0:
    """Callable. Construct once per session; the renderer calls predict() per turn."""

    def __init__(self, model="us.anthropic.claude-opus-4-7", kinds=DEFAULT_KINDS,
                 transport=None, ledger=None, max_tokens=300, cap=None):
        self.model, self.kinds, self.max_tokens = model, tuple(kinds), max_tokens
        self.meter = AR.CostMeter(ledger=ledger or os.path.join(HERE, "bench_spend_ledger.json"),
                                  **({"cap": cap} if cap else {}))
        self.transport = transport or (lambda req, **kw: AR.bedrock_transport(req))
        self.turns = 0

    def predict(self, state, last_exchange="", n=5) -> Prediction:
        prompt = PROMPT.format(state=json.dumps(state, ensure_ascii=False)[:4000],
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
        kept, dropped = _parse(text, set(self.kinds), n)
        self.turns += 1
        return Prediction(actions=kept, cost_usd=cost, raw=text, dropped=dropped,
                          error="" if kept else "no in-vocabulary action parsed")


def _selftest():
    """Every path, with a fake transport. No network, no spend."""
    state = {"screen": "salon", "service": None, "stylist": None, "slot": None}
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
    for i in range(1, 6):
        for s in ("", ".tmp"):
            f = "/tmp/_bench_t%d.json%s" % (i, s)
            if os.path.exists(f):
                os.remove(f)
    print("\n  SELFTEST PASSED — 5 checks, $0.00, no network.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
