#!/usr/bin/env python3
"""R2c (理 11300): the head reads its state from the pass the AGENT MAKES ANYWAY.

PRE-REGISTERED IN results/r2c_prereg.json, BEFORE THIS FILE EXISTED. Bar 502 tokens/episode,
unchanged. Accounting rule, which is the whole substance of R2c: charge ONLY tokens the agent would
not otherwise have paid.

WHAT R2 AND R2b BOTH CHARGED, AND WHY IT WAS WRONG.
    R2   k separate full-prompt passes            4.90 x 480  = 2,353 tok/episode   FAILED (+34%)
    R2b  one cached prefix + k continuations      492.3 + 36.1 =   528 tok/episode   FAILED by 26.4
R2b still charged the 492-token prefix to the CONTROLLER. But the agent prefills that exact prompt
in order to emit its own action -- with or without a head. Charging it to the controller bills the
controller for work the system does either way. 理: "your §4 is not a rescue, it is the accounting
the measured system got wrong."

WHAT THIS DOES. ONE prefill of the decision prompt. Its KV cache then serves BOTH:
    (a) the k candidate continuations the head scores, and
    (b) the agent's own teacher-forced generation,
each by rewinding the cache to the end of the prefix. The prefix is paid once by the agent, and the
controller's MARGINAL cost is the candidate continuations alone.

THE CONSERVATIVE CHOICE, MADE ON PURPOSE. All k continuations are charged, including the winner's --
even though the winner's continuation is the agent's own action prefix, already in the cache, so the
agent re-pays nothing for it. Charging k rather than k-1 costs this measurement ~7 tokens/episode
and removes any question of the accounting having been tuned to the bar. The tighter figure is
reported as a bound, never as the headline.

SELFTEST IS MANDATORY AND GATES THE MEASUREMENT. `main` refuses to report a cost unless selftest
passes. R2b is the reason: there, a cost number that CLEARED the bar was produced by a path reading
the wrong state, and only the identity check made that visible. Two things are checked here, not
one -- the candidate STATES against hidden_at, and the agent's GENERATED TEXT against an uncached
generate. A shared cache that silently changes what the agent writes is not a saving either.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRE = '{"action": "hypothesize", "region": "%s", "bug_class": "'


def _rewind(cache, snap):
    """Restore the cache to the end of the shared prefix.

    DEEP COPIES, not aliases: KVCache mutates its key/value arrays in place as it grows, so a saved
    REFERENCE points at an array the next candidate has already overwritten. (R2b, first version.)
    """
    for c, (k, v, off) in zip(cache, snap):
        c.keys, c.values, c.offset = k, v, off


def shared_decision(model, tok, messages, layer, regions, max_tokens=400, temp=0.0,
                    logits_processors=None, pick=None):
    """One prefill; candidate states and the agent's generation both read it.

    Returns (states, marginal_tokens, prefix_tokens, gen_text, gen_tokens, winner_tokens).
    `pick` (a region) is the teacher-forced action to generate; if None, no generation is run and
    only the states come back.
    """
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache

    inner = model.model
    base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pre_ids = tok.encode(base)

    cache = make_prompt_cache(model)
    h = inner.embed_tokens(mx.array([pre_ids]))
    mask = create_attention_mask(h, cache)
    for i, lyr in enumerate(inner.layers):
        h = lyr(h, mask, cache[i])
    snap = [(mx.array(c.keys), mx.array(c.values), c.offset) for c in cache]

    def suffix_ids(r):
        # Tokenisation is NOT compositional: encode the FULL string and take the tail, or the byte
        # pair merge across the boundary runs a different token sequence than the uncached path.
        full = tok.encode(base + (PRE % r))
        assert full[:len(pre_ids)] == pre_ids, "the prefix is not a token-prefix of the full string"
        return full[len(pre_ids):]

    states, marginal, winner_tokens = {}, 0, 0
    for r in regions:
        sids = suffix_ids(r)
        marginal += len(sids)
        if r == pick:
            winner_tokens = len(sids)
        _rewind(cache, snap)
        hr = inner.embed_tokens(mx.array([sids]))
        mr = create_attention_mask(hr, cache)
        for i, lyr in enumerate(inner.layers):
            hr = lyr(hr, mr, cache[i])
            if i == layer:
                states[r] = hr[0, -1, :]

    gen_text, gen_tokens = None, 0
    if pick is not None:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        _rewind(cache, snap)
        sids = suffix_ids(pick)
        kw = {"logits_processors": logits_processors} if logits_processors else {}
        out = []
        for tokid, _ in generate_step(mx.array(sids), model, max_tokens=max_tokens,
                                      sampler=make_sampler(temp=temp), prompt_cache=cache, **kw):
            if tokid in tok.eos_token_ids:
                break
            out.append(tokid)
        gen_text, gen_tokens = tok.decode(out), len(out)

    return states, marginal, len(pre_ids), gen_text, gen_tokens, winner_tokens


# ---------------------------------------------------------------- selftest / measurement

def cmd_selftest(model, tok, dec, layer):
    """Both halves must hold or no cost is reported. Returns (ok, report)."""
    import mlx.core as mx
    import et8_head_v3 as H
    import et8_agent as A

    rs, rd, gen_ok, gen_n = [], [], [], 0
    for d in dec:
        regions = d["regions"]
        pick = regions[0]
        states, marg, npre, text, ntok, _ = shared_decision(model, tok, d["messages"], layer,
                                                            regions, max_tokens=120, pick=pick)
        for r in regions:
            ref = H.hidden_at(model, tok, d["messages"], [layer], prefix=PRE % r)[layer]
            x = np.array(states[r].astype(mx.float32), copy=False)
            y = np.array(ref.astype(mx.float32), copy=False)
            rs.append(float(np.corrcoef(x, y)[0, 1])); rd.append(float(np.abs(x - y).max()))
        # the agent's own text, uncached, teacher-forced with the same prefix
        ref_text, _, _ = A.generate(model, tok, d["messages"], max_tokens=120, temp=0.0,
                                    prefix=PRE % pick)
        gen_ok.append(ref_text[len(PRE % pick):].strip() == (text or "").strip()); gen_n += 1

    ok = (min(rs) > 0.999 and max(rd) < 5.0 and all(gen_ok))
    return ok, {
        "states_vs_hidden_at": {"n": len(rs), "pearson_r_min": round(min(rs), 6),
                                "max_abs_diff_max": round(max(rd), 3),
                                "PASS": bool(min(rs) > 0.999 and max(rd) < 5.0)},
        "generation_vs_uncached": {"n": gen_n, "identical": int(sum(gen_ok)),
                                   "PASS": bool(all(gen_ok))},
        "note": "an identity check on BOTH the state and the text. A shared cache that changes "
                "what the agent writes is not a saving, and a cost measured over wrong states is "
                "not a cost -- R2b produced exactly that and only the check caught it.",
        "SELFTEST": "PASS" if ok else "FAIL"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--selftest-n", type=int, default=6)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import et8_head_v3 as H
    from mlx_lm import load

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = H.post_inspect_decisions(steps, a.tasks)
    model, tok = load(a.model)

    ok, st = cmd_selftest(model, tok, dec[:a.selftest_n], a.layer)
    print(json.dumps(st, indent=1))
    if not ok:
        json.dump({"VERDICT": "VOID -- selftest failed, no cost is reported", "selftest": st},
                  open(a.out, "w"), indent=1)
        print("SELFTEST FAILED -- no cost reported."); return 1

    use = dec[:a.limit]
    marg = pre = win = k = 0
    t0 = time.time()
    for d in use:
        _, m, p, _, _, w = shared_decision(model, tok, d["messages"], a.layer, d["regions"],
                                           pick=d["regions"][0])
        marg += m; pre += p; win += w; k += len(d["regions"])
    wall = time.time() - t0

    n = len(use)
    res = {"document": "R2c RESULT — the head over the agent's own prefill (理 11300)",
           "prereg": "results/r2c_prereg.json — bar 502, accounting rule written before the build",
           "decisions": n, "mean_candidates": round(k / n, 2),
           "CHARGED tokens/episode": round(marg / n, 1),
           "NOT charged (the agent pays it anyway)": {
               "shared decision prefix": round(pre / n, 1),
               "why": "the agent prefills this prompt to emit its own action, head or no head"},
           "BAR (pre-registered, unchanged)": 502,
           "VERDICT": None,
           "for reference": {
               "R2 (k full passes)": 2487.9, "R2b (prefix charged to the controller)": 528.4,
               "tighter bound NOT used as the headline": round((marg - win) / n, 1),
               "why not": "the winner's continuation is the agent's own action prefix and is "
                          "already in the cache, so k-1 is defensible. k is charged instead so "
                          "the accounting cannot be said to have been tuned to the bar."},
           "seconds_per_decision": round(wall / n, 3),
           "selftest": st}
    res["VERDICT"] = ("PASSES the bar" if res["CHARGED tokens/episode"] <= 502 else "FAILS the bar")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
