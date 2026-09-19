#!/usr/bin/env python3
"""R6 — the strongest SIMPLE baselines at G's own compute (理 11398). Pre-registered readings below.

THE QUESTION. G is a logistic head fitted on VERIFIED history. Two cheaper things could explain its
gain without any learned readout:
  (a) SELF-RERANK   over the SAME enumerated candidates, pick the one the frozen model itself
                    assigns the highest log-probability. Charged the same k passes G pays.
  (b) MAJORITY VOTE over 5 sampled hypotheses at that decision. Charged 5 short passes.
  (c) base + 35% ordinary search = R5-E's budget-16 arm, ALREADY MEASURED at 22.3%.

READINGS WRITTEN BEFORE THE RUN (理 11398):
  (a) or (b) ~= G          -> the learned readout adds nothing over the model's own preference
  both below G by > CI     -> EXPERIENCE (a head trained on verified history) is responsible,
                              not merely extra inference at the decision
  expectation              -> (a) lands BETWEEN base and G: the model's own preference is exactly
                              what G learned to OVERRULE, on 179 of 300 decisions.

WHY THIS FILE EDITS NOTHING. et8_agent.run_episode takes `head_state`, and it calls
`head_state["hidden"](...)` per candidate and scores `z @ w + b`. Supplying w=[1], b=0, mu=[0],
sd=[1] and a `hidden` that returns a ONE-ELEMENT vector makes that arithmetic the identity, so the
element IS the score. Both baselines therefore run through the SAME decision plumbing G uses,
with nothing re-defined and no line of the measured agent changed.

⚠️ TOKEN ACCOUNTING IS ANALYTIC, as it was for G. The candidate passes are not in the episode's
tokens_in/out (that is why R2 had to ADD the controller's 2,487.9). Each baseline is charged the
same way and the charge is printed beside its result, never omitted.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PRE = '{"action": "hypothesize", "region": "%s", "bug_class": "'


def _identity_head(fn, layer=0):
    """Wrap a per-candidate scorer as a head_state whose linear layer is the identity."""
    import mlx.core as mx
    return {"w": np.array([1.0]), "b": 0.0, "mu": np.array([0.0]), "sd": np.array([1.0]),
            "layer": layer, "hidden": fn, "mx": mx}


def self_rerank_head():
    """(a) score = the frozen model's own mean log-probability of the candidate's opening fields.

    MEAN per token, not total: a longer region name would otherwise score lower for being longer,
    which would make this a length baseline wearing a preference baseline's name.
    """
    import mlx.core as mx

    def hidden(model, tok, messages, layers, prefix=""):
        base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pre_ids = tok.encode(base)
        full = tok.encode(base + prefix)
        tail = full[len(pre_ids):]
        if not tail:
            return {layers[0]: mx.array([-1e9])}
        logits = model(mx.array([full[:-1]]))[0]
        lp = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1))
        tot = 0.0
        for i, t in enumerate(tail):
            tot += float(lp[len(pre_ids) - 1 + i, t])
        return {layers[0]: mx.array([tot / len(tail)])}

    return _identity_head(hidden)


def majority_vote_head(n_samples=5, temp=0.8, seed=0):
    """(b) score = how many of n sampled hypotheses name this region.

    The n samples are drawn ONCE per decision and cached; the per-candidate call then reports that
    region's vote count. Ties fall to the model's first sample, which is recorded in the metadata
    rather than broken silently.
    """
    import mlx.core as mx
    import et8_agent as A
    cache = {}

    def hidden(model, tok, messages, layers, prefix=""):
        key = id(messages[-1]["content"]) if messages else 0
        base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if key not in cache:
            votes, first = {}, None
            mx.random.seed(seed)
            for _ in range(n_samples):
                text, _, _ = A.generate(model, tok, messages, temp=temp)
                p = A.parse_action(text)
                r = p.get("region") if p.get("action") == "hypothesize" else None
                if r:
                    votes[r] = votes.get(r, 0) + 1
                    first = first or r
            cache[key] = (votes, first)
        votes, first = cache[key]
        region = prefix.split('"region": "', 1)[-1].split('"', 1)[0] if '"region": "' in prefix else ""
        v = float(votes.get(region, 0))
        if v and region == first:
            v += 0.5          # a recorded, deterministic tie-break — never a silent one
        return {layers[0]: mx.array([v])}

    return _identity_head(hidden)


CHARGES = {
    "self_rerank": "k full-prompt passes per decision — the SAME charge G pays (R2b measured "
                   "2,487.9 tokens/episode for the measured system).",
    "majority_vote": "n_samples short generations per decision, charged at the measured "
                     "prompt+completion length of a hypothesis step.",
    "base_plus_search": "no controller; the extra compute is in the agent's own budget (R5-E's "
                        "budget-16 arm, 9,234.2 tokens/episode, 22.3%).",
}

if __name__ == "__main__":
    print(__doc__)
    for k, v in CHARGES.items():
        print("  %-18s %s" % (k, v))
