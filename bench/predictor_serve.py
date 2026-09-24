#!/usr/bin/env python
"""WO-312 — predictor v0 behind a localhost HTTP endpoint, so the renderer can call it from JS.

形's harness is Vite + react-native-web; predictor v0 is Python. This is the seam.

  python bench/predictor_serve.py --fake          # no credentials, no network, no spend
  AWS_PROFILE=et-bedrock-rater python bench/predictor_serve.py

  POST http://127.0.0.1:8899/predict
       {"state": {...}, "last_exchange": "...", "n": 5}
    -> {"actions":[{kind,label,args}], "cost_usd":0.0052, "dropped":[], "error":"", "turn":3}

--fake IS THE POINT OF THIS FILE, not a convenience. 形 can wire the PREDICTED arm, tap through it,
and see 鉋's replay log fill TODAY, with no credential on their box and nothing spent. The real
predictor is the same endpoint with the flag removed, so the integration that gets tested is the
integration that ships.

The fake returns plausible in-vocabulary actions derived from the state, NOT a fixed list: a stub
that always answers the same thing lets a renderer bug hide, because every turn looks alike.
"""
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "poetry"))
import api_rater as AR
from predictor_v0 import PredictorV0, DEFAULT_KINDS, world_from_space
from area_v0 import AreaV0, load_taxonomy


def _section(prompt, header, end):
    """Pull one labelled JSON block out of the prompt. The fake must read the SAME state the real
    model reads; the first version searched for the first '{' in the whole prompt, which stopped
    being the state the moment a VENUE block was put above it."""
    i = prompt.find(header)
    if i < 0:
        return {}
    i += len(header)
    j = prompt.find(end, i)
    try:
        return json.loads(prompt[i:j if j > i else None].strip())
    except Exception:
        return {}


def fake_transport_factory():
    """A stand-in that ANSWERS FROM THE STATE, so different turns give different heads.

    It reads the REAL UiState shape 形 sent (the record) — selection.offer_ids / .staff / .day /
    .slot and sheet — not the invented {service, stylist, slot} the first draft keyed on, which the
    reducer never produces, and it names ids out of the VENUE block so its args resolve.
    """
    def t(req, **kw):
        prompt = req["messages"][0]["content"]
        st = _section(prompt, "\nCURRENT STATE\n", "\n\nLAST EXCHANGE")
        wd = _section(prompt, "args must come from here\n", "\n\nCURRENT STATE")
        sel = st.get("selection") or {}
        offers = [o for o in (wd.get("offers") or []) if o.get("offer_id")]
        staff = wd.get("staff") or []
        days = wd.get("days") or []
        acts = []
        if not sel.get("offer_ids"):
            for o in offers[:2]:
                acts.append({"kind": "select", "label": "%s%s" % (o.get("name") or o["offer_id"],
                             " · $%s" % o["price"] if o.get("price") is not None else ""),
                             "args": {"offer_id": o["offer_id"]}})
        elif not sel.get("staff"):
            for p_ in staff[:3]:
                acts.append({"kind": "select", "label": "Book with %s" % p_, "args": {"staff": p_}})
        elif not sel.get("day"):
            for d in days[:3]:
                acts.append({"kind": "select", "label": d.capitalize(), "args": {"day": d}})
        elif not sel.get("slot"):
            acts.append({"kind": "set", "label": "Pick a time", "args": {}})
        elif st.get("sheet") != "book":
            acts.append({"kind": "next", "label": "Continue to details", "args": {}})
        else:
            acts.append({"kind": "confirm", "label": "Request this booking", "args": {}})
        acts.append({"kind": "ask", "label": "Ask a question", "args": {}})
        acts.append({"kind": "back", "label": "Back", "args": {}})
        return {"content": [{"type": "text", "text": json.dumps(acts)}],
                "usage": {"input_tokens": len(prompt) // 4, "output_tokens": 40}}
    return t


class Handler(BaseHTTPRequestHandler):
    predictor = None
    area = None
    area_mode = "not configured"
    mode = "not configured"
    # /area and /predict need DIFFERENT interpreters on this box: the local classifier needs
    # mlx_lm (the mlx venv) and the paid predictor needs boto3 (the system python), and neither
    # venv has the other package. Installing into the shared mlx venv while experiments run on it
    # is not worth a convenience. So whichever backend cannot be built here says so in plain words
    # at 503 rather than failing inside a request with an import error nobody can read.
    predict_unavailable = ""
    # The estate's whole vocabulary, for the second pass. Empty = the two-pass is OFF and the
    # endpoint behaves exactly as before, which is how it ships when a taxonomy has no such list.
    estate_tags = ()

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")       # the renderer is on another port
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self._send(200, {})

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "mode": Handler.mode,
                             "area_backend": Handler.area_mode,
                             # NAMED, because an unlabelled "turns" on a service with two
                             # endpoints is read as whichever the reader cares about. I used this
                             # field as evidence that the CLASSIFIER had served nothing; it counts
                             # the PREDICTOR, and the classifier had a counter of its own that I
                             # never looked at.
                             "predict_turns": Handler.predictor.turns if Handler.predictor else None,
                             "area_turns": getattr(Handler.area, "turns", None),
                             "predict_available": Handler.predictor is not None,
                             "kinds": list(DEFAULT_KINDS)})
        else:
            self._send(404, {"error": "POST /predict"})

    def do_POST(self):
        if self.path.startswith("/area"):
            return self._area()
        if not self.path.startswith("/predict"):
            return self._send(404, {"error": "POST /predict or POST /area"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": "bad json: %s" % e})
        if Handler.predictor is None:
            return self._send(503, {"error": Handler.predict_unavailable,
                                    "actions": [], "cost_usd": 0.0, "dropped": [],
                                    "unresolved": [], "turn": 0})
        p = Handler.predictor.predict(body.get("state") or {},
                                      body.get("last_exchange", ""),
                                      int(body.get("n", 5)))
        self._send(200, {"actions": p.actions, "cost_usd": round(p.cost_usd, 6),
                         "dropped": p.dropped, "unresolved": p.unresolved, "error": p.error,
                         "turn": Handler.predictor.turns, "mode": Handler.mode})

    def _area(self):
        """WO-318 §4 — utterance -> typed area. 理's demo endpoint.

        Takes an utterance and a venue and NOTHING ELSE. No state, no prior turn, no identity, no
        session history: there is no field here through which any of them could arrive, which is
        `docs/wo318-data-scope-sautee.md` §1 enforced by the signature rather than promised.
        """
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": "bad json: %s" % e})
        u, venue = body.get("utterance"), body.get("venue")
        if not u or not venue:
            return self._send(400, {"error": "utterance and venue are both required"})
        try:
            # TWO PASSES when the first resolves to nothing and the intent names a service: the
            # venue's own tag list cannot offer a candidate for something the venue does not sell,
            # so an off-menu question and a whole-set question arrive identical without it.
            if Handler.estate_tags:
                from area_v0 import classify_with_off_menu
                r = classify_with_off_menu(Handler.area, u, venue, Handler.estate_tags)
            else:
                r = Handler.area.classify(u, venue)
        except KeyError as e:
            return self._send(400, {"error": str(e)})
        return self._send(200, {"area": r.area or None, "unresolved": r.unresolved,
                                "resolves_to_nothing": bool(r.resolves_to_nothing),
                                "off_menu": r.off_menu,
                                "cost_usd": round(r.cost_usd, 6), "error": r.error,
                                "turn": Handler.area.turns, "mode": Handler.area_mode,
                                "area_unchanged": bool(r.error)})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    # DEFAULTS TO LOOPBACK ON PURPOSE. 形 could not reach /area from their checkout because this
    # bound 127.0.0.1 on box A and I said "the endpoint is there" without saying WHOSE loopback
    # . --host 0.0.0.0 puts it on the estate LAN; it is not a public route and there
    # is none to this box. The $10 WO-318 cap is enforced in AreaV0 against its own ledger file, so
    # a second caller shares the ceiling rather than raising it — that is the point of a meter in
    # code rather than a watched number.
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to serve the estate LAN. Default is this box's loopback only.")
    ap.add_argument("--fake", action="store_true", help="no credentials, no network, no spend")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-7")
    ap.add_argument("--ledger", default=os.path.join(HERE, "bench_spend_ledger.json"))
    ap.add_argument("--taxonomy", default="", help="taxonomy json for /area (default: the demo one)")
    # /area IS LOCAL-ONLY BY DEFAULT, on Louay's rule relayed by 理: the classifier runs on a local
    # model for everything — the demo, the fixtures, the page and live traffic — and the paid path
    # is not on the classifier route at all. The remote backend stays reachable behind a flag that
    # defaults OFF so the code path does not rot, never as a fallback: a fallback is how "local
    # only" becomes "local unless something goes wrong", which is the opposite of the rule.
    ap.add_argument("--area-backend", choices=("local", "remote"), default="local")
    ap.add_argument("--area-local-model", default="",
                    help="MLX model id for the local classifier (default: area_local's)")
    ap.add_argument("--area-local-temperature", type=float, default=0.0,
                    help="0 serves the same answer for the same sentence, which is what a screen "
                         "wants. NOTE that it also makes a replicate-disagreement number 0 by "
                         "construction — measure stability at the temperature you SERVE at.")
    ap.add_argument("--space", default="", help="compiled venue json (niwa's fixture). Without it "
                                                "the predictor names ids nothing can render.")
    a = ap.parse_args()
    world = world_from_space(json.load(open(a.space))) if a.space else None
    fb = fake_transport_factory() if a.fake else None
    try:
        if a.fake:
            Handler.predictor = PredictorV0(model=a.model, transport=fb,
                                            ledger=a.ledger + ".FAKE", world=world)
            Handler.mode = "/predict: FAKE — no network, no spend"
        else:
            Handler.predictor = PredictorV0(model=a.model, ledger=a.ledger, world=world)
            # PREFIXED WITH THE ENDPOINT IT DESCRIBES, on purpose. /health used to answer
        # "mode": "LIVE Bedrock: ..." while the CLASSIFIER was local — a reader glancing at that
        # would reasonably conclude the user's sentence leaves the estate, which is the one thing
        # the rule forbids and the one thing this service does not do. The value says which
        # endpoint it is about; "area_backend" says what the classifier actually is.
        Handler.mode = "/predict: LIVE Bedrock: %s" % a.model
    except Exception as e:
        Handler.predictor = None
        Handler.predict_unavailable = (
            "/predict is not available in this process: %s: %s. It needs boto3, which this "
            "interpreter does not have. /area is unaffected and is local-only by default."
            % (type(e).__name__, str(e)[:160]))
        Handler.mode = "PREDICT UNAVAILABLE — /area only"
        print("  ⚠️ %s" % Handler.predict_unavailable)
    print("  venue: %s" % ("%s — %d offer(s), %d staff" % (world["handle"], len(world["offers"]),
          len(world["staff"])) if world else "NONE (--space not given; args will not resolve)"))
    print("  predictor v0 on http://%s:%d   mode: %s" % (a.host, a.port, Handler.mode))
    tax = load_taxonomy(a.taxonomy or None)
    if a.area_backend == "local" and not a.fake:
        import area_local as AL
        model_id = a.area_local_model or AL.DEFAULT_LOCAL_MODEL
        Handler.area = AL.local_area(tax, model_id=model_id,
                                     ledger=os.path.join(HERE, "wo318_local_ledger.json"),
                                     temperature=a.area_local_temperature)
        Handler.area_mode = "LOCAL %s @T%g — nothing leaves the estate, $0" % (
            model_id, a.area_local_temperature)
    else:
        Handler.area = AreaV0(tax, model=a.model, transport=(fb if a.fake else None),
                              ledger=os.path.join(HERE, "wo318_spend_ledger.json")
                              + (".FAKE" if a.fake else ""))
        Handler.area_mode = ("FAKE — no network, no spend" if a.fake
                             else "REMOTE %s — PAID, and the sentence leaves the estate" % a.model)
    Handler.estate_tags = tuple(tax.get("estate_vocabulary") or ())
    print("  /area backend: %s" % Handler.area_mode)
    print("  two-pass off-menu: %s"
          % ("ON, %d estate tags" % len(Handler.estate_tags) if Handler.estate_tags
             else "OFF (taxonomy carries no estate_vocabulary)"))
    print("  POST /predict {state, last_exchange, n}   POST /area {utterance, venue}   GET /health")
    print("  /area venues: %s" % ", ".join(sorted(Handler.area.tax["venues"])))
    HTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
