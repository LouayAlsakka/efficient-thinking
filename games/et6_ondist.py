#!/usr/bin/env python
"""ET-VI E-A — on-distribution decomposition (resolves the F1 confound + tests F5).

The held-out decomposition used RANDOM late-game positions (off-distribution), so its 'fit' error
conflates learning with generalization. Here we sample positions the net ACTUALLY visits — late-game
states reached by the net's own self-play — and decompose fit/bias there, still against the exact
solver. If on-distribution fit collapses (net learned its labels where it plays) while off-distribution
fit stays large, that is F5's distribution gap, and the plateau=label-bound story may hold on-policy.
If on-distribution fit is still large, the net is genuinely fit/optimization-bound even where it plays.

  PYTHONPATH=games ./.venv/bin/python games/et6_ondist.py --net games/c4_net.safetensors
"""
import argparse, json, random
import numpy as np
import mlx.core as mx
from connect4 import C4, value01 as true_value01
from c4_net import C4Net
from et6_decomp import net_value, net_probs, rollout_value


def ondist_positions(net, n, rng, nprng, min_depth=16, temp=1.0):
    """n decisive late-game positions reached by the net's OWN play (on-distribution), with V*."""
    out = []
    while len(out) < n:
        s = C4(); depth = rng.randint(min_depth, 30)
        ok = True
        for _ in range(depth):
            if s.terminal() is not None or not s.legal():
                ok = False; break
            p = net_probs(net, s)                      # the net's own policy (its play distribution)
            if p.sum() <= 0:
                mv = s.legal()[0]
            else:
                mv = int(nprng.choice(7, p=p / p.sum()))
                if not s.can_play(mv):
                    mv = s.legal()[0]
            s = s.play(mv)
        if not ok or s.terminal() is not None:
            continue
        v = true_value01(s)
        if v in (0.0, 1.0):                            # decisive, fast to solve at this depth
            out.append((s, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="games/c4_net.safetensors")
    ap.add_argument("--positions", type=int, default=100)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--mcts-sims", type=int, default=64)
    ap.add_argument("--channels", type=int, default=64); ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="games/results/et6_ondist.json")
    a = ap.parse_args()
    net = C4Net(a.channels, a.blocks); net.load_weights(a.net); net.eval(); mx.eval(net.parameters())
    rng = random.Random(a.seed); nprng = np.random.default_rng(a.seed)
    pos = ondist_positions(net, a.positions, rng, nprng)
    print(f"[ondist] {len(pos)} ON-DISTRIBUTION decisive positions (net's own play); V^pi under net+MCTS")
    fit, bias, total = [], [], []
    for i, (s, vstar) in enumerate(pos):
        vnet = net_value(net, s)
        vpi = rollout_value(net, s, a.k, nprng, mcts_sims=a.mcts_sims, npr=nprng)
        fit.append(abs(vnet - vpi)); bias.append(abs(vpi - vstar)); total.append(abs(vnet - vstar))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pos)}  fit={np.mean(fit):.3f} bias={np.mean(bias):.3f}", flush=True)
    F = dict(fit=round(float(np.mean(fit)), 3), label_bias=round(float(np.mean(bias)), 3),
             total=round(float(np.mean(total)), 3), n=len(pos))
    print(f"\n[on-distribution] fit={F['fit']}  label-bias={F['label_bias']}  total={F['total']}  "
          f"-> {'LABEL-BOUND on-policy (fit collapsed; F5 distribution-gap explains the held-out miss)' if F['label_bias']>F['fit'] else 'still FIT-BOUND even on-policy (genuinely optimization-bound)'}")
    json.dump(F, open(a.out, "w"), indent=2)
    print(f"[ondist] wrote {a.out}")


if __name__ == "__main__":
    main()
