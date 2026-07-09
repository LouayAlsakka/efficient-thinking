#!/usr/bin/env python
"""Merge (weight-average) several SAME-architecture conv nets into one ("marriage of coefficients").

  --align none    : naive coefficient average. FAILS across different random inits, because
                    independently-trained nets sit in different loss basins related by neuron
                    permutations, so averaging misaligned neurons cancels signal.
  --align permute : Git Re-Basin weight-matching. Align every net's neurons to a reference frame
                    (one shared residual-stream permutation P, a per-block hidden perm Q_i, plus
                    R for the reduce layer and S for the value head), THEN average. This is the
                    real test of whether aligned "marriage" of random-start models moves forward.

  PYTHONPATH=. python scripts/merge.py --align permute \
      --models runs/cmte_a runs/cmte_dataB runs/cmte_dataC --arch-from runs/cmte_a --out runs/merge_align
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import asdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from chessnet.model import ModelConfig, PolicyNet
from chessnet.train import load_run, RunConfig

try:
    from scipy.optimize import linear_sum_assignment
except Exception:                                    # greedy fallback if scipy missing
    def linear_sum_assignment(cost):
        cost = np.array(cost); n = cost.shape[0]; rows = list(range(n)); cols = []
        used = set()
        for i in range(n):
            order = np.argsort(cost[i])
            for j in order:
                if j not in used:
                    cols.append(j); used.add(j); break
        return np.array(rows), np.array(cols)


def params_np(run_dir):
    m, _ = load_run(run_dir)
    return {k: np.array(v) for k, v in tree_flatten(m.parameters())}


def apply_perms(B, N, P, Q, R, S):
    """Permute net B's neurons to a reference frame. perm array p means out[i] <- B[p[i]]."""
    o = dict(B)
    o["stem.weight"] = B["stem.weight"][P]                 # stem out = P
    o["stem.bias"]   = B["stem.bias"][P]
    for i in range(N):
        Qi = Q[i]
        w = B[f"blocks.{i}.c1.weight"][Qi]                 # c1 out = Qi
        o[f"blocks.{i}.c1.weight"] = w[:, :, :, P]         # c1 in  = P
        o[f"blocks.{i}.c1.bias"]   = B[f"blocks.{i}.c1.bias"][Qi]
        w = B[f"blocks.{i}.c2.weight"][P]                  # c2 out = P
        o[f"blocks.{i}.c2.weight"] = w[:, :, :, Qi]        # c2 in  = Qi
        o[f"blocks.{i}.c2.bias"]   = B[f"blocks.{i}.c2.bias"][P]
    o["reduce.weight"] = B["reduce.weight"][R][:, :, :, P] # reduce out=R, in=P
    o["reduce.bias"]   = B["reduce.bias"][R]
    hw = B["head.weight"].reshape(B["head.weight"].shape[0], 64, -1)  # (4096, 64 sq, 8 ch)
    o["head.weight"]   = hw[:, :, R].reshape(B["head.weight"].shape)  # head in = R (per square)
    if "v_fc1.weight" in B:
        o["v_fc1.weight"] = B["v_fc1.weight"][S][:, P]     # v_fc1 out=S, in=P
        o["v_fc1.bias"]   = B["v_fc1.bias"][S]
        o["v_fc2.weight"] = B["v_fc2.weight"][:, S]        # v_fc2 in=S
    return o


def _match(M):                                            # maximize <A, perm(B)>
    r, c = linear_sum_assignment(-M)
    p = np.empty(M.shape[0], dtype=int); p[r] = c
    return p


def weight_match(A, B, N, C, Rc, has_v, rounds=6):
    P = np.arange(C); Q = [np.arange(C) for _ in range(N)]; R = np.arange(Rc); S = np.arange(C)
    def flat_out(w): return w.reshape(w.shape[0], -1)                 # match on out (axis0)
    def flat_in(w):  return np.moveaxis(w, -1, 0).reshape(w.shape[-1], -1)  # match on in (last)
    for _ in range(rounds):
        # ---- P : residual stream (stem out, c1 in, c2 out, reduce in, v_fc1 in) ----
        Bp = apply_perms(B, N, np.arange(C), Q, R, S)              # B with all perms but P
        M = flat_out(A["stem.weight"]) @ flat_out(Bp["stem.weight"]).T
        for i in range(N):
            M += flat_in(A[f"blocks.{i}.c1.weight"]) @ flat_in(Bp[f"blocks.{i}.c1.weight"]).T
            M += flat_out(A[f"blocks.{i}.c2.weight"]) @ flat_out(Bp[f"blocks.{i}.c2.weight"]).T
        M += flat_in(A["reduce.weight"]) @ flat_in(Bp["reduce.weight"]).T
        if has_v:
            M += A["v_fc1.weight"].T @ Bp["v_fc1.weight"]           # match in (cols)
        P = _match(M)
        # ---- Q_i : per-block hidden (c1 out, c2 in) ----
        for i in range(N):
            Bp = apply_perms(B, N, P, [np.arange(C) if k == i else Q[k] for k in range(N)], R, S)
            M  = flat_out(A[f"blocks.{i}.c1.weight"]) @ flat_out(Bp[f"blocks.{i}.c1.weight"]).T
            M += flat_in(A[f"blocks.{i}.c2.weight"]) @ flat_in(Bp[f"blocks.{i}.c2.weight"]).T
            Q[i] = _match(M)
        # ---- R : reduce out (reduce out, head in-per-square) ----
        Bp = apply_perms(B, N, P, Q, np.arange(Rc), S)
        M  = flat_out(A["reduce.weight"]) @ flat_out(Bp["reduce.weight"]).T
        ha = A["head.weight"].reshape(A["head.weight"].shape[0], 64, Rc)
        hb = Bp["head.weight"].reshape(Bp["head.weight"].shape[0], 64, Rc)
        M += np.einsum("skc,skd->cd", ha, hb)                       # sum over out-logits & squares
        R = _match(M)
        # ---- S : value hidden (v_fc1 out, v_fc2 in) ----
        if has_v:
            Bp = apply_perms(B, N, P, Q, R, np.arange(C))
            M  = flat_out(A["v_fc1.weight"]) @ flat_out(Bp["v_fc1.weight"]).T
            M += A["v_fc2.weight"].T @ Bp["v_fc2.weight"]           # match in (cols)
            S = _match(M)
    return P, Q, R, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--arch-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align", choices=["none", "permute"], default="none")
    args = ap.parse_args()

    _, base = load_run(args.arch_from)
    cfg = ModelConfig(encoding="onehot", arch=base.arch, width=base.width, depth=base.depth,
                      value_head=getattr(base, "value_head", False))
    C, N, Rc, has_v = base.width, base.depth, cfg.conv_head_channels, cfg.value_head

    nets = [params_np(d) for d in args.models]
    ref = nets[0]
    aligned = [ref]
    for B in nets[1:]:
        if args.align == "permute":
            P, Q, R, S = weight_match(ref, B, N, C, Rc, has_v)
            aligned.append(apply_perms(B, N, P, Q, R, S))
        else:
            aligned.append(B)

    merged = {k: np.mean([a[k] for a in aligned], axis=0) for k in ref}
    model = PolicyNet(cfg)
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in merged.items()]))
    mx.eval(model.parameters())

    os.makedirs(args.out, exist_ok=True)
    json.dump(asdict(RunConfig(encoding="onehot", arch=base.arch, width=base.width,
                               depth=base.depth, value_head=has_v, run_dir=args.out)),
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    model.save_weights(os.path.join(args.out, "model.npz"))
    print(f"[merge] {args.align}: averaged {len(nets)} nets -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
