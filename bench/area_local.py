#!/usr/bin/env python
"""WO-318 — the LOCAL arm of `/area`. Same class, same prompt, same parser; a different model.

WHY THIS ARM EXISTS. 理 12418 §2(c), ruled again by 令 R464 §7: the classifier choice gains a third
criterion beside disagreement rate and latency — **whether the user's sentence leaves the estate**.
The local model sends nothing anywhere. Bedrock on live traffic makes every routed utterance a
permanently unerasable record of something a person typed, and that is a price somebody who is not
in the room pays. So the measurement that decides between them has to exist before the choice does.

WHAT IS AND IS NOT DUPLICATED. Everything that turns a reply into an `Area` — the prompt, the json
extraction, the intent enum check, the kept/dropped tag split — is AreaV0's, reached by injecting a
transport and a meter. **A disagreement rate measured across two parsers is a measurement of the
parsers**, and the whole point of this arm is that the only thing differing between the two is the
model. The only code here is a transport and a free meter.

COST. Zero dollars, by construction: no network call leaves this box. The free meter still reserves
and settles, so the local arm produces the same artifact shape as the paid one and `cost_usd` reads
0.0 because it IS 0.0, not because the field was omitted.

LATENCY IS RECORDED, NOT ESTIMATED. `classify_ms` on both arms is wall time around the transport,
measured here rather than inferred from token counts — the second of 理's three criteria.
"""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from area_v0 import AreaV0, load_taxonomy, INTENTS   # noqa: E402

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class FreeMeter:
    """Reserve/settle that always costs nothing, and SAYS SO in the ledger rather than writing none.

    A silent zero and a missing meter look identical in an artifact six weeks later. This writes a
    real row per call with `actual_usd: 0.0` and the model id, so the local arm's ledger is
    evidence that no call was paid for, not an absence of evidence.
    """

    def __init__(self, ledger=None):
        self.ledger = ledger
        self.state = {"spent_usd": 0.0, "calls": 0, "entries": []}
        if ledger and os.path.exists(ledger):
            self.state = json.load(open(ledger))

    def reserve(self, model, tin, tout, note=""):
        rid = len(self.state["entries"])
        self.state["entries"].append({"ts": time.time(), "id": rid, "model": model,
                                      "reserved_usd": 0.0, "actual_usd": None, "note": note,
                                      "local": True})
        self._flush()
        return rid

    def settle(self, rid, tin, tout):
        e = self.state["entries"][rid]
        e.update({"actual_usd": 0.0, "input_tokens": tin, "output_tokens": tout})
        self.state["calls"] = sum(1 for x in self.state["entries"] if x["actual_usd"] is not None)
        self._flush()
        return 0.0

    def _flush(self):
        if self.ledger:
            os.makedirs(os.path.dirname(os.path.abspath(self.ledger)), exist_ok=True)
            json.dump(self.state, open(self.ledger, "w"), indent=1)


class MlxTransport:
    """mlx-lm behind the Bedrock request shape AreaV0 already speaks.

    Loads once and holds the model: the replicate protocol calls this a few hundred times, and a
    per-call load would make the latency number a measurement of disk.
    """

    def __init__(self, model_id=DEFAULT_LOCAL_MODEL, temperature=0.0):
        from mlx_lm import load as _load
        self.model, self.tok = _load(model_id)
        self.model_id, self.temperature = model_id, temperature

    def __call__(self, req, **kw):
        from mlx_lm import generate as _gen
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=self.temperature)
        except Exception:
            sampler = None
        prompt = req["messages"][0]["content"]
        chat = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        kwargs = {"max_tokens": req.get("max_tokens", 120), "verbose": False}
        if sampler is not None:
            kwargs["sampler"] = sampler
        text = _gen(self.model, self.tok, prompt=chat, **kwargs)
        n_in = len(self.tok.encode(chat))
        n_out = len(self.tok.encode(text))
        return {"content": [{"text": text}], "usage": {"input_tokens": n_in, "output_tokens": n_out}}


def local_area(taxonomy=None, model_id=DEFAULT_LOCAL_MODEL, ledger=None, temperature=0.0):
    """An AreaV0 whose model is on this box. Identical in every other respect."""
    return AreaV0(taxonomy or load_taxonomy(), model=model_id,
                  transport=MlxTransport(model_id, temperature),
                  meter=FreeMeter(ledger))


def _selftest():
    """Runs the REAL class against a canned transport — no model load, no network, no spend."""
    tax = load_taxonomy()
    canned = {"n": 0}

    def fake(req, **kw):
        canned["n"] += 1
        return {"content": [{"text": '{"intent": "hours", "tags": ["info.hours", "not.a.tag"], '
                                     '"commit": false}'}],
                "usage": {"input_tokens": 11, "output_tokens": 7}}

    m = FreeMeter(None)
    a = AreaV0(tax, model="local-test", transport=fake, meter=m)
    r = a.classify("what time do you close", "quick-cuts.chelsea")
    assert r.error == "", r.error
    assert r.area["intent"] == "hours", r.area
    assert r.area["tags"] == ["info.hours"], r.area          # 1: the venue's tag is kept
    assert r.unresolved == ["not.a.tag"], r.unresolved       # 2: the foreign tag is DROPPED and COUNTED
    assert r.cost_usd == 0.0                                 # 3: the local arm costs nothing
    assert r.area["commit"] is False                         # 4: commit is a BOOLEAN, the design of record
    assert r.area["intent"] in INTENTS                       # 5: and the intent is one of the eight
    assert m.state["entries"][0]["actual_usd"] == 0.0        # 6: a real ledger row, not an absent one
    assert m.state["calls"] == 1
    assert canned["n"] == 1                                  # 7: exactly one transport call per classify
    print("  area_local selftest: 7/7 — AreaV0's own parser, a free meter, zero spend")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    ap.add_argument("--venue", default="quick-cuts.chelsea")
    ap.add_argument("--ledger", default="")
    ap.add_argument("utterance", nargs="*")
    a = ap.parse_args()
    if a.selftest or not a.utterance:
        _selftest()
    else:
        cl = local_area(model_id=a.model, ledger=a.ledger or None)
        t0 = time.time()
        r = cl.classify(" ".join(a.utterance), a.venue)
        print(json.dumps({"area": r.area or None, "unresolved": r.unresolved,
                          "cost_usd": r.cost_usd, "error": r.error,
                          "classify_ms": round((time.time() - t0) * 1000)}, ensure_ascii=False))
