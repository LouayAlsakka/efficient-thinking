#!/usr/bin/env python3
"""R2b (理): score all k candidates from ONE cached prefix, and measure the overhead.

THE BAR, FIXED BEFORE THIS FILE EXISTED: <= 502 tokens/episode (r2b_prereg.json). 502 = 0.71 actions
saved x 707 tokens per marginal action, measured by an OLS fit of tokens on actions over the base's
300 episodes.

WHAT THE MEASURED SYSTEM DID. The head scored k candidates with k SEPARATE forward passes, each over
the whole ~480-token decision prompt: 4.90 passes/episode = 2,353 prompt tokens, and the cost limb
failed on tokens at +34% total compute even though it passed in actions.

WHAT THIS DOES. The k prompts are byte-identical except the final region name, so:
    one pass over the shared prefix  ->  KV cache
    k short continuations -- the region name PLUS the fixed tail the measured path reads after
Cost becomes (prefix once) + sum over k of (region name + tail), instead of k x prefix.

THE TAIL IS NOT FREE AND CANNOT BE CACHED. It is byte-identical across candidates, but it sits
AFTER the differing region name, so every candidate pays it. Dropping it would read the state one
position earlier than the measured path -- a different state, not a cheaper one.

WHY IT IS A SEPARATE FILE. et8_head_v3.hidden_at is what produced every number in the package;
replacing it in place would silently re-define the thing already measured. This is measured against
it -- same states, same head, and an assertion that the two agree -- and only then is the overhead
quoted.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def hidden_batched(model, tok, messages, layer, regions):
    """Return {region: hidden state at the last token}, from ONE cached prefix.

    Also returns the token accounting: prefix tokens paid once, plus the continuation tokens.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    TAIL = '", "bug_class": "'   # the measured path reads AFTER this; so must this one
    prefix = base + '{"action": "hypothesize", "region": "'
    pre_ids = tok.encode(prefix)

    inner = model.model
    out, cont_tokens = {}, 0

    cache = make_prompt_cache(model)
    h = inner.embed_tokens(mx.array([pre_ids]))
    mask = None
    try:
        from mlx_lm.models.base import create_attention_mask
        mask = create_attention_mask(h, cache)
    except Exception:
        pass
    prefix_state = None
    for i, lyr in enumerate(inner.layers):
        h = lyr(h, mask, cache[i])
        if i == layer:
            prefix_state = h[0, -1, :]
    # DEEP COPY, not an alias. KVCache mutates its key/value arrays IN PLACE as it grows, so saving
    # the references and restoring them does not rewind anything -- the arrays the references point
    # at have already been overwritten by the next candidate. The first version of this file aliased
    # them, and the equivalence check in this same command caught it: with an IDENTICAL suffix the
    # cached path and the measured path gave pearson r = 0.19 and max|diff| = 137 where they should
    # have been the same vector. A token saving computed over wrong states is not a saving.
    saved = [(mx.array(c.keys), mx.array(c.values), c.offset) for c in cache]

    for r in regions:
        # TOKENISATION IS NOT COMPOSITIONAL. tok.encode(prefix + r) is NOT tok.encode(prefix) +
        # tok.encode(r): the byte-pair merge across the boundary differs, so feeding a separately
        # encoded region name into a cache built from a separately encoded prefix runs a DIFFERENT
        # token sequence than the uncached path. That was the whole discrepancy -- encoding the full
        # string and taking the tail fixes it, costs only CPU tokenisation, and leaves the GPU work
        # at prefix-once plus the tail.
        full_ids = tok.encode(prefix + r + TAIL)
        assert full_ids[: len(pre_ids)] == pre_ids, "prefix is not a token-prefix of the full string"
        r_ids = full_ids[len(pre_ids):]
        cont_tokens += len(r_ids)
        for c, (k, v, off) in zip(cache, saved):        # rewind to the shared prefix
            c.keys, c.values, c.offset = k, v, off
        hr = inner.embed_tokens(mx.array([r_ids]))
        # The continuation gets its own mask built against the cache. NOTE: this was NOT the
        # defect. It was committed as one of three cache-side fixes that each left the numbers
        # completely unmoved -- which was itself the evidence that the cache was never the problem.
        # The real defect was two lines up and two functions over: this function IGNORED its own
        # `layer` argument and returned the FINAL layer, while it was being compared against
        # hidden_at at layer 18. r = 0.16-0.20 was a layer mismatch, not a cache bug.
        mr = None
        try:
            from mlx_lm.models.base import create_attention_mask as _cam
            mr = _cam(hr, cache)
        except Exception:
            pass
        for i, lyr in enumerate(inner.layers):
            hr = lyr(hr, mr, cache[i])
            if i == layer:
                out[r] = hr[0, -1, :]
    return out, len(pre_ids), cont_tokens, prefix_state


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import et8_head_v3 as H
    from mlx_lm import load
    import mlx.core as mx

    steps = []
    for r in a.runs:
        steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = H.post_inspect_decisions(steps, a.tasks)[: a.limit]
    model, tok = load(a.model)

    agree, maxd, checked = [], [], 0
    pre_tok, cont_tok, k_tot, old_tok = 0, 0, 0, 0
    t_new = t_old = 0.0
    for d in dec:
        regions = d["regions"]
        t0 = time.time()
        got, n_pre, n_cont, _ = hidden_batched(model, tok, d["messages"], a.layer, regions)
        t_new += time.time() - t0
        pre_tok += n_pre; cont_tok += n_cont; k_tot += len(regions)

        # the OLD path, for the agreement assertion and its own cost
        t0 = time.time()
        for r in regions:
            pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
            hs = H.hidden_at(model, tok, d["messages"], [a.layer], prefix=pre)
            if checked < 20:
                x = np.array(got[r].astype(mx.float32), copy=False)
                y = np.array(hs[a.layer].astype(mx.float32), copy=False)
                agree.append(float(np.corrcoef(x, y)[0, 1]))
                maxd.append(float(np.abs(x - y).max())); checked += 1
            old_tok += len(tok.encode(tok.apply_chat_template(d["messages"], tokenize=False,
                                                              add_generation_prompt=True) + pre))
        t_old += time.time() - t0

    n = len(dec)
    res = {"decisions": n, "mean_candidates": round(k_tot / n, 2),
           "OLD tokens/episode": round(old_tok / n, 1),
           "NEW tokens/episode": round((pre_tok + cont_tok) / n, 1),
           "NEW breakdown": {"prefix_once": round(pre_tok / n, 1),
                             "region_name_continuations": round(cont_tok / n, 1)},
           "BAR (pre-registered)": 502,
           "VERDICT": None,
           "seconds_per_decision": {"old": round(t_old / n, 3), "new": round(t_new / n, 3)},
           "agreement_with_the_measured_path": {
               "note": "SAME token sequence, SAME layer, SAME read position. This is an "
                       "IDENTITY check, not a similarity check: anything but r ~ 1.0 with a "
                       "max|diff| at the bf16 noise floor (~2 on states of magnitude ~100) is "
                       "a defect, and the verdict below is VOID.",
               "n_checked": checked,
               "pearson_r_min": round(min(agree), 6) if agree else None,
               "max_abs_diff_max": round(max(maxd), 3) if maxd else None,
               "pearson_r": [round(x, 6) for x in agree[:10]],
               "max_abs_diff": [round(x, 3) for x in maxd[:10]]}}
    ok = bool(agree) and min(agree) > 0.999 and max(maxd) < 5.0
    res["VERDICT"] = ("VOID -- the cached path does not reproduce the measured states" if not ok
                      else "PASSES the bar" if res["NEW tokens/episode"] <= 502
                      else "FAILS the bar")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
