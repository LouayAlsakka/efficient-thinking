#!/usr/bin/env python
"""ET-VI E-A / F3 — conversion-skill bias map. F3 predicts label bias |V^pi - V*| concentrates on
NARROW winning positions (few value-preserving moves — the only-move index), where converting the
advantage demands skill the policy lacks; wide advantages label accurately at any level.

For solved WIN positions (V*=1), narrowness = number of legal moves that PRESERVE the win (a move to
child c preserves iff the exact solver gives c value 0 = a loss for the side then to move). Narrow = 1
winning move (only-move); wide = several. We correlate narrowness with label bias (V^pi via k policy
rollouts) and report mean bias per narrowness bin.

  PYTHONPATH=games ./.venv/bin/python games/et6_f3.py --net games/results/et6_ea_big.safetensors --channels 96 --blocks 6
"""
import argparse, json, random
import numpy as np
import mlx.core as mx
from connect4 import C4, value01 as true_value01
from c4_net import C4Net
from et6_decomp import net_probs, rollout_value


def win_positions(n, rng):
    """n solved WIN positions (V*=1 for side to move), each with its count of value-preserving moves."""
    out = []
    while len(out) < n:
        s = C4(); depth = rng.randint(12, 28); ok = True
        for _ in range(depth):
            if s.terminal() is not None or not s.legal():
                ok = False; break
            s = s.play(rng.choice(s.legal()))
        if not ok or s.terminal() is not None or true_value01(s) != 1.0:
            continue
        # count winning (value-preserving) moves: child is a loss for the opponent-to-move
        nwin = sum(1 for c in s.legal() if true_value01(s.play(c)) == 0.0)
        if nwin >= 1:
            out.append((s, nwin, len(s.legal())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="games/results/et6_ea_big.safetensors")
    ap.add_argument("--positions", type=int, default=180)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--channels", type=int, default=96); ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="games/results/et6_f3.json")
    a = ap.parse_args()
    net = C4Net(a.channels, a.blocks); net.load_weights(a.net); net.eval(); mx.eval(net.parameters())
    rng = random.Random(a.seed); nprng = np.random.default_rng(a.seed)
    pos = win_positions(a.positions, rng)
    rows, narrow, bias = [], [], []
    for i, (s, nwin, nleg) in enumerate(pos):
        vpi = rollout_value(net, s, a.k, nprng)                 # policy-head rollout
        b = abs(vpi - 1.0)                                      # |V^pi - V*|, V*=1
        rows.append({"n_winning_moves": nwin, "n_legal": nleg, "v_pi": round(vpi, 3), "bias": round(b, 3)})
        narrow.append(nwin); bias.append(b)
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(pos)}", flush=True)
    narrow, bias = np.array(narrow), np.array(bias)
    # bins: only-move (1), narrow (2), wide (>=3)
    bins = {"only_move(1)": narrow == 1, "narrow(2)": narrow == 2, "wide(>=3)": narrow >= 3}
    print("\n[F3] label bias by winning-move count (narrowness):")
    summary = {}
    for name, msk in bins.items():
        if msk.sum():
            summary[name] = {"n": int(msk.sum()), "mean_bias": round(float(bias[msk].mean()), 3)}
            print(f"  {name:>13}: n={int(msk.sum()):>3}  mean|V^pi-V*|={bias[msk].mean():.3f}")
    corr = float(np.corrcoef(narrow, bias)[0, 1])                # negative = bias falls as moves widen (F3)
    print(f"[F3] corr(n_winning_moves, bias) = {corr:+.3f}  "
          f"({'NEGATIVE → bias concentrates on narrow wins (F3 supported)' if corr < -0.1 else 'not clearly negative (F3 weak/miss)'})")
    json.dump({"corr_narrowness_bias": round(corr, 3), "bins": summary, "n": len(pos), "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"[F3] wrote {a.out}")


if __name__ == "__main__":
    main()
