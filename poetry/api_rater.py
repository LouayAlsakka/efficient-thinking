#!/usr/bin/env python
"""ET-IV — the rating harness's rater, swapped from a human to a frontier model.

The interface human_session.py already uses is unchanged: a rater is a callable
`rater(pair) -> 'A' | 'B'`. _dry_run's fake rater satisfies it; so does ApiRater. Pareto pruning
and pairwise-at-N=4 stay exactly where they are -- this file replaces WHO answers, nothing else.

THE $40 CAP IS ENFORCED HERE, IN CODE, NOT WATCHED (理's condition).
Four properties, because three of them are the ways a meter fails to be a cap:

  1. IT RESERVES THE WORST CASE BEFORE THE CALL, NOT THE ESTIMATE AFTER IT. The reservation uses
     max_tokens for output. A meter that bills what it expected cannot be exceeded by a model that
     writes more than expected -- which is the only way this overruns.
  2. IT PERSISTS. The ledger is a file. A cap that resets when the process dies is not a cap, and
     this campaign is many short runs, not one long one.
  3. IT REFUSES, IT DOES NOT WARN. Over the cap raises BudgetExceeded and no request is sent.
  4. ITS PRICES COME FROM THE COST ARTIFACT, not from a literal here. iv_cost_estimate.json is what
     理 approved; if the two disagree, the run stops rather than quietly billing at a second number.

BUILD-ONLY until 理 gives GO. --selftest exercises every path with a fake transport and spends $0;
real calls need --live AND a planned total at or under the $20 GO gate.
"""
import argparse, json, os, random, re, sys, time, urllib.request

HARD_CAP_USD = 40.0          # 理: enforced in code, with a running meter, not watched
GO_GATE_USD = 20.0           # 理: GO only if the PLANNED total is at or under this
HERE = os.path.dirname(os.path.abspath(__file__))
COST_ARTIFACT = os.path.join(HERE, "iv_cost_estimate.json")
DEFAULT_LEDGER = os.path.join(HERE, "data", "iv_spend_ledger.json")


class BudgetExceeded(RuntimeError):
    pass


def load_prices(path=COST_ARTIFACT):
    """Prices come from the artifact 理 approved. A literal in this file could drift from it."""
    with open(path) as f:
        doc = json.load(f)
    prices = {}
    for k, v in doc["prices_read_2026-09-20"].items():
        if isinstance(v, dict) and "in_per_M" in v:
            prices[k] = (float(v["in_per_M"]), float(v["out_per_M"]))
    if not prices:
        raise SystemExit("STOP: no prices in %s" % path)
    return prices


class CostMeter:
    """A running meter with a hard cap, persisted to disk.

    reserve() is called BEFORE the request with the worst case; settle() replaces the reservation
    with the actual usage afterwards. Between them the reservation is on the books, so a crash
    mid-call leaves the ledger pessimistic rather than optimistic.
    """

    def __init__(self, ledger=DEFAULT_LEDGER, cap=HARD_CAP_USD, prices=None):
        self.ledger, self.cap = ledger, float(cap)
        self.prices = prices if prices is not None else load_prices()
        os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
        self.state = {"spent_usd": 0.0, "calls": 0, "entries": []}
        if os.path.exists(self.ledger):
            with open(self.ledger) as f:
                self.state = json.load(f)

    # -- accounting ------------------------------------------------------
    def price(self, model, tin, tout):
        if model not in self.prices:
            raise BudgetExceeded("no price for model %r in the cost artifact — refusing to call it"
                                 % model)
        pin, pout = self.prices[model]
        return tin * pin / 1e6 + tout * pout / 1e6

    @property
    def spent(self):
        return float(self.state["spent_usd"])

    @property
    def remaining(self):
        return self.cap - self.spent

    def _flush(self):
        tmp = self.ledger + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.ledger)   # atomic: a torn ledger is an unbounded one

    def reserve(self, model, input_tokens, max_tokens, note=""):
        """Charge the WORST CASE up front. Raises BudgetExceeded and sends nothing if it won't fit."""
        worst = self.price(model, input_tokens, max_tokens)
        if self.spent + worst > self.cap:
            raise BudgetExceeded(
                "REFUSED: $%.4f spent, this call reserves $%.4f worst-case, cap is $%.2f. "
                "Nothing was sent." % (self.spent, worst, self.cap))
        self.state["spent_usd"] = self.spent + worst
        self.state["calls"] += 1
        rid = len(self.state["entries"])
        self.state["entries"].append({"id": rid, "model": model, "reserved_usd": round(worst, 6),
                                      "actual_usd": None, "note": note, "ts": time.time()})
        self._flush()
        return rid

    def settle(self, rid, input_tokens, output_tokens):
        """Replace the reservation with the actual cost."""
        e = self.state["entries"][rid]
        if e["actual_usd"] is not None:
            raise RuntimeError("entry %d already settled" % rid)
        actual = self.price(e["model"], input_tokens, output_tokens)
        self.state["spent_usd"] = self.spent - e["reserved_usd"] + actual
        e["actual_usd"] = round(actual, 6)
        e["input_tokens"], e["output_tokens"] = input_tokens, output_tokens
        self._flush()
        return actual

    def line(self):
        return ("meter: $%.4f of $%.2f  (%.1f%%)  %d calls  $%.4f left"
                % (self.spent, self.cap, 100 * self.spent / self.cap,
                   self.state["calls"], self.remaining))


def check_plan(planned_usd, gate=GO_GATE_USD):
    """理's GO condition: the PLANNED total must be at or under the gate before anything runs."""
    if planned_usd > gate:
        raise BudgetExceeded("NO GO: planned total $%.2f is over the $%.2f gate. 理 decides, not me."
                             % (planned_usd, gate))
    return True


# -- transports ---------------------------------------------------------
def http_transport(req, api_key, url="https://api.anthropic.com/v1/messages", timeout=120):
    body = json.dumps(req).encode()
    r = urllib.request.Request(url, data=body, headers={
        "content-type": "application/json", "x-api-key": api_key,
        "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


class FakeTransport:
    """Canned responses + usage. Every selftest path runs through this and spends nothing."""

    def __init__(self, answers, out_tokens=60):
        self.answers, self.out_tokens, self.seen = list(answers), out_tokens, []

    def __call__(self, req, **kw):
        self.seen.append(req)
        text = self.answers[(len(self.seen) - 1) % len(self.answers)]
        return {"content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 213, "output_tokens": self.out_tokens}}


# -- the rater ----------------------------------------------------------
PROMPT = ("Two poems were written to the same brief. Answer with exactly one letter, A or B: "
          "which is the better poem?\n\nBRIEF: {brief}\n\nA:\n{a}\n\nB:\n{b}\n\nAnswer:")


class ApiRater:
    """Callable with human_session's rater interface: rater(pair) -> 'A' | 'B'.

    Unparseable answers return None. They are NOT retried and NOT coerced to a letter: a rater that
    guesses when it did not answer is a rater whose disagreement rate is partly noise I invented.
    The caller drops them and the drop count goes in the artifact.
    """

    def __init__(self, meter, transport, model="claude-fable-5-1", max_tokens=16, brief=""):
        self.meter, self.transport, self.model = meter, transport, model
        self.max_tokens, self.brief = max_tokens, brief
        self.unparsed = 0

    def __call__(self, pair, brief=None):
        prompt = PROMPT.format(brief=brief if brief is not None else self.brief,
                               a=pair["A"]["text"], b=pair["B"]["text"])
        est_in = max(1, len(prompt) // 4)
        rid = self.meter.reserve(self.model, est_in, self.max_tokens, note=pair["pair_id"])
        resp = self.transport({"model": self.model, "max_tokens": self.max_tokens,
                               "messages": [{"role": "user", "content": prompt}]})
        u = resp.get("usage", {})
        self.meter.settle(rid, u.get("input_tokens", est_in), u.get("output_tokens", self.max_tokens))
        text = "".join(c.get("text", "") for c in resp.get("content", [])).strip()
        return self.parse(text)

    def parse(self, text):
        """The answer must BE a letter, not CONTAIN one.

        My first version scanned for the first A or B anywhere in the reply. "I cannot choose"
        returns 'A' under that rule -- from the A in "cannot". Every refusal would have been
        recorded as a vote for whichever poem was shown first, which is a position bias I would
        have manufactured and then measured. The selftest caught it; the fix is to require the
        letter to stand alone."""
        t = text.strip().strip('"\'*.:` \t\n')
        if len(t) == 1 and t.upper() in ("A", "B"):
            return t.upper()
        m = re.match(r"^([AB])\b", t.upper())
        if m:
            return m.group(1)
        self.unparsed += 1
        return None


# -- selftest -----------------------------------------------------------
def _selftest():
    import human_session as HS
    tmp = os.path.join(HERE, "data", "_selftest_ledger.json")
    for p in (tmp, tmp + ".tmp"):
        if os.path.exists(p):
            os.remove(p)
    prices = load_prices()
    print("  prices from the artifact: %s" % {k: v for k, v in prices.items()})

    # 1. the pipeline is UNCHANGED: the same Pareto/pairs path, a different rater
    rng = random.Random(0)
    cands = [{"cand_id": "c%d" % i, "text": "poem %d" % i, "model": "m%d" % (i % 3),
              "scores": {"meter": rng.random(), "rhyme": rng.random(), "taste": rng.random()}}
             for i in range(8)]
    pareto = HS.pareto_prune(cands, ["meter", "rhyme", "taste"])
    pairs = HS.make_ab_pairs(pareto, rng)
    meter = CostMeter(ledger=tmp, prices=prices)
    rater = ApiRater(meter, FakeTransport(["A", "B"]), brief="a brief")
    log = []
    for i, p in enumerate(pairs):
        w = rater(p)
        if w:
            log.append(HS.log_choice(p, w, ts=i))
    sel = {p["pair_id"]: p["A"]["cand_id"] for p in pairs}
    q = HS.compute_q(sel, log)
    assert q is not None and 0.0 <= q <= 1.0, "q did not come through the real pipeline"
    print("  [1] pipeline unchanged: %d cands -> %d pareto -> %d pairs -> q=%s, %s"
          % (len(cands), len(pareto), len(pairs), q, meter.line()))

    # 2. CONTROL — the cap REFUSES. Push the ledger to the brink and assert nothing is sent.
    m2 = CostMeter(ledger=tmp, cap=HARD_CAP_USD, prices=prices)
    m2.state["spent_usd"] = HARD_CAP_USD - 0.000001
    m2._flush()
    ft = FakeTransport(["A"])
    r2 = ApiRater(m2, ft)
    try:
        r2(pairs[0])
        raise AssertionError("CONTROL FAILED: the cap did not refuse")
    except BudgetExceeded as e:
        assert not ft.seen, "CONTROL FAILED: a request was sent by a refused call"
        print("  [2] cap refuses and sends nothing: %s" % str(e).split(".")[0])

    # 3. CONTROL — the ledger PERSISTS across a process restart
    m3 = CostMeter(ledger=tmp, prices=prices)
    assert abs(m3.spent - m2.spent) < 1e-9, "CONTROL FAILED: ledger did not survive a reopen"
    print("  [3] ledger survives a restart: reopened at $%.6f" % m3.spent)

    # 4. CONTROL — the reservation is the WORST CASE, not the estimate. A model that writes
    #    max_tokens when 60 were expected must not be able to walk past the cap.
    os.remove(tmp)
    # The cap must let SOME calls through and then stop: a control that refuses the very first
    # call passes without ever exercising the mechanism. My first version did exactly that
    # (0 of 15 calls, assertion `sent < 15` true at zero) and proved nothing.
    MAXT, CAP4 = 4000, 0.65
    worst = 213 * 10.0 / 1e6 + MAXT * 50.0 / 1e6           # what reserve() books per call
    est = 213 * 10.0 / 1e6 + 60 * 50.0 / 1e6               # what an ESTIMATE-based meter would book
    fit_worst, fit_est = int(CAP4 // worst), int(CAP4 // est)
    m4 = CostMeter(ledger=tmp, cap=CAP4, prices=prices)
    runaway = FakeTransport(["A"], out_tokens=MAXT)        # the model actually writes max_tokens
    r4 = ApiRater(m4, runaway, max_tokens=MAXT)
    sent = 0
    try:
        for p in pairs:
            r4(p); sent += 1
    except BudgetExceeded:
        pass
    assert 0 < sent < len(pairs), "CONTROL FAILED: %d calls — the mechanism was not exercised" % sent
    assert sent == fit_worst, "CONTROL FAILED: %d calls fit, worst-case metering allows %d" % (sent, fit_worst)
    assert m4.spent <= m4.cap + 1e-9, "CONTROL FAILED: spent $%.4f over a $%.4f cap" % (m4.spent, m4.cap)
    print("  [4] worst-case reservation holds: %d of %d calls made, $%.4f of a $%.4f cap — an "
          "estimate-based meter would have allowed %d calls and overrun by $%.2f"
          % (sent, len(pairs), m4.spent, m4.cap, fit_est, fit_est * worst - CAP4))

    # 5. CONTROL — an unpriced model is refused rather than billed at a guess
    m5 = CostMeter(ledger=tmp + "5", prices=prices)
    try:
        ApiRater(m5, FakeTransport(["A"]), model="some-model-not-in-the-artifact")(pairs[0])
        raise AssertionError("CONTROL FAILED: an unpriced model was called")
    except BudgetExceeded:
        print("  [5] unpriced model refused rather than billed at a guess")

    # 6. the GO gate is a separate check from the cap
    doc = json.load(open(COST_ARTIFACT))
    planned = float(doc["total"]["cost_usd"])
    check_plan(planned)
    try:
        check_plan(GO_GATE_USD + 0.01)
        raise AssertionError("CONTROL FAILED: the GO gate passed an over-gate plan")
    except BudgetExceeded:
        pass
    print("  [6] GO gate: planned $%.2f <= $%.2f -> GO; an over-gate plan is refused"
          % (planned, GO_GATE_USD))

    # 7. an unparseable answer is dropped, not coerced
    m7 = CostMeter(ledger=tmp + "7", prices=prices)
    r7 = ApiRater(m7, FakeTransport(["I cannot choose", "Both are good", "A", "B.", " b ",
                                     "The answer is A"]))
    got = [r7(pairs[0]) for _ in range(6)]
    assert got == [None, None, "A", "B", "B", None], "parser: got %r" % got
    assert r7.unparsed == 3, "unparsed count %d" % r7.unparsed
    print("  [7] answers that CONTAIN a letter are dropped, not coerced: %r (%d dropped)"
          % (got, r7.unparsed))

    for p in (tmp, tmp + ".tmp", tmp + "5", tmp + "7"):
        if os.path.exists(p):
            os.remove(p)
    print("\n  SELFTEST PASSED — 7 checks, $0.00 spent. BUILD-ONLY until 理 gives GO.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true", help="make REAL calls; needs ANTHROPIC_API_KEY")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--meter", action="store_true", help="print the ledger and exit")
    a = ap.parse_args()
    if a.meter:
        print(CostMeter(ledger=a.ledger).line()); return
    if a.selftest:
        sys.path.insert(0, HERE); _selftest(); return
    if a.live:
        raise SystemExit("STOP: --live is not wired to an arm yet, and ET-IV is BUILD-ONLY until "
                         "理 gives GO. The cap and the gate are in place; the arm is not.")
    ap.print_help()


if __name__ == "__main__":
    main()
