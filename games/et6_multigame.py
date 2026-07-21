#!/usr/bin/env python
"""ET-VI direct A/B — does K-games-per-position value labeling raise the self-play plateau?

Louay's proposal: instead of the 1-game value target, play K games from each training position and use
the empirical win-rate as the value label. Same C4 self-play loop, only the value target changes:
  --value-games 1   : baseline (one playout per position ≈ the standard game outcome)
  --value-games 100 : the proposal (low-variance V^pi label)
The framework's prediction (F2): more games shrink variance, not bias — V^pi stays ~0.24-0.36 from V*,
so the plateau (set by that bias) should not move much; the one live effect is that cleaner labels may
let a fit-bound net reach its V^pi ceiling faster/higher, but not past it. This run settles it.
Reports per iteration: gelo (strength) + eval-vs-oracle MAE. Policy target = MCTS visit counts (as
usual); ONLY the value target uses K rollouts.

  PYTHONPATH=games ./.venv/bin/python games/et6_multigame.py --value-games 1  --out games/results/et6_mg_k1.json
  PYTHONPATH=games ./.venv/bin/python games/et6_multigame.py --value-games 100 --out games/results/et6_mg_k100.json
"""
import argparse, json, random, time, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from collections import deque
from connect4 import C4
from c4_net import C4Net
from c4_mcts import search, visit_probs
from c4_selfplay import build_holdout, LADDER, LADDER_FN, policy_move
from c4_calibrate import match, place_agent
from et6_decomp import rollout_value


def collect_game(net, sims, np_rng, temp_moves=10):
    """One self-play game -> list of (encode[84], visit_probs[7], board_copy) for value labeling."""
    s = C4(); out = []; ply = 0
    while s.terminal() is None and s.legal():
        root = search(net, s, sims=sims, dirichlet=0.25, dir_alpha=1.0, rng=np_rng)
        vp = visit_probs(root)
        out.append((s.encode(), vp, s))
        mv = int(np_rng.choice(7, p=vp)) if (ply < temp_moves and vp.sum() > 0) else int(vp.argmax())
        if not s.can_play(mv):
            mv = s.legal()[0]
        s = s.play(mv); ply += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-games", type=int, default=1, help="K rollouts per position for the value label")
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--sims", type=int, default=64); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=20000); ap.add_argument("--label-cap", type=int, default=200,
                    help="max positions value-labeled per iteration (bounds the K-rollout cost)")
    ap.add_argument("--channels", type=int, default=64); ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=4); ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    py_rng = random.Random(a.seed); np_rng = np.random.default_rng(a.seed)
    net = C4Net(a.channels, a.blocks); mx.eval(net.parameters())
    opt = optim.Adam(learning_rate=a.lr)

    def loss_fn(net, X, P, V):
        logits, value = net(X)
        logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        return -(P * logp).sum(axis=1).mean() + ((value - V) ** 2).mean()
    lg = nn.value_and_grad(net, loss_fn)
    Xh, Yh = build_holdout(a.holdout, py_rng)

    def eval_iter():
        net.eval()
        _, value = net(Xh); v = np.array(value.tolist()); mae = float(np.mean(np.abs(v - Yh)))
        raw = lambda s: policy_move(net, s)
        scores = [match(raw, LADDER_FN[k], 24, random.Random(a.seed + 700 + i)) for i, k in enumerate(LADDER)]
        return mae, float(place_agent(scores, np.array([LADDER[k] for k in LADDER])))

    buf = deque(maxlen=a.buffer); hist = []; t0 = time.time()
    for it in range(1, a.iters + 1):
        net.eval()
        positions = []
        for _ in range(a.games):
            positions.extend(collect_game(net, a.sims, np_rng))
        py_rng.shuffle(positions); positions = positions[:a.label_cap]
        for enc, vp, board in positions:                      # value label = mean of K rollouts (the A/B knob)
            v = rollout_value(net, board, a.value_games, np_rng) if a.value_games > 0 else 0.5
            buf.append((enc, vp, v))
        net.train(); data = list(buf)
        for _ in range(a.epochs):
            py_rng.shuffle(data)
            for i in range(0, len(data), a.batch):
                ch = data[i:i + a.batch]
                X = mx.array([c[0] for c in ch]); P = mx.array(np.stack([c[1] for c in ch]))
                V = mx.array(np.array([c[2] for c in ch], dtype=np.float32))
                _, g = lg(net, X, P, V); opt.update(net, g); mx.eval(net.parameters(), opt.state)
        if it % a.eval_every == 0 or it == 1:
            mae, gelo = eval_iter()
            hist.append({"iter": it, "value_mae": round(mae, 3), "gelo": round(gelo, 0)})
            print(f"  it{it:>3} K={a.value_games}  MAE={mae:.3f}  GELO={gelo:+.0f}  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"value_games": a.value_games, "config": vars(a), "history": hist}, open(a.out, "w"), indent=2)
    print(f"[mg] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
