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
from predictor_v0 import PredictorV0, DEFAULT_KINDS


def fake_transport_factory():
    """A stand-in that ANSWERS FROM THE STATE, so different turns give different heads."""
    def t(req, **kw):
        prompt = req["messages"][0]["content"]
        st = {}
        try:
            i, j = prompt.find("{"), prompt.find("\n\nLAST EXCHANGE")
            st = json.loads(prompt[i:j]) if i >= 0 and j > i else {}
        except Exception:
            pass
        acts = []
        if not st.get("service"):
            for o in (st.get("offers") or ["Service"])[:2]:
                acts.append({"kind": "select", "label": str(o), "args": {"service": o}})
        if st.get("service") and not st.get("stylist"):
            acts.append({"kind": "select", "label": "Choose stylist", "args": {}})
        if st.get("service") and not st.get("slot"):
            acts.append({"kind": "set", "label": "Pick a time", "args": {}})
        if st.get("service") and st.get("slot"):
            acts.append({"kind": "confirm", "label": "Confirm booking", "args": {}})
        acts.append({"kind": "ask", "label": "Ask a question", "args": {}})
        acts.append({"kind": "back", "label": "Back", "args": {}})
        return {"content": [{"type": "text", "text": json.dumps(acts)}],
                "usage": {"input_tokens": len(prompt) // 4, "output_tokens": 40}}
    return t


class Handler(BaseHTTPRequestHandler):
    predictor = None

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
            self._send(200, {"ok": True, "mode": Handler.mode, "turns": Handler.predictor.turns,
                             "kinds": list(DEFAULT_KINDS)})
        else:
            self._send(404, {"error": "POST /predict"})

    def do_POST(self):
        if not self.path.startswith("/predict"):
            return self._send(404, {"error": "POST /predict"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": "bad json: %s" % e})
        p = Handler.predictor.predict(body.get("state") or {},
                                      body.get("last_exchange", ""),
                                      int(body.get("n", 5)))
        self._send(200, {"actions": p.actions, "cost_usd": round(p.cost_usd, 6),
                         "dropped": p.dropped, "error": p.error,
                         "turn": Handler.predictor.turns, "mode": Handler.mode})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--fake", action="store_true", help="no credentials, no network, no spend")
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-7")
    ap.add_argument("--ledger", default=os.path.join(HERE, "bench_spend_ledger.json"))
    a = ap.parse_args()
    if a.fake:
        fb = fake_transport_factory()
        Handler.predictor = PredictorV0(model=a.model, transport=fb,
                                        ledger=a.ledger + ".FAKE")
        Handler.mode = "FAKE — no network, no spend"
    else:
        Handler.predictor = PredictorV0(model=a.model, ledger=a.ledger)
        Handler.mode = "LIVE Bedrock: %s" % a.model
    print("  predictor v0 on http://127.0.0.1:%d   mode: %s" % (a.port, Handler.mode))
    print("  POST /predict {state, last_exchange, n}   GET /health")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
