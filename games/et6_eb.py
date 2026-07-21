#!/usr/bin/env python
"""ET-VI E-B — search-over-policy margin curve, instrumented with the E-A decomposition.

Runs the Connect-4 self-play training loop and, every eval_every iters, logs on a FIXED probe of solved
positions:
  gelo          : strength (level)
  search_margin : (fraction of win positions where net+MCTS picks a winning move)
                  - (fraction where the raw policy head does) — what search buys over the policy
  fit           : mean |V_net - V^pi|   (V^pi via k rollouts under net+MCTS)
  label_bias    : mean |V^pi - V*|
Registered expectation: the per-iteration training gain (distill(search(pi))) shrinks toward the noise
floor at the plateau. Sharper question E-A now lets us ask: does the gain track label-bias reduction,
fit reduction, or neither? — making this VI's second figure rather than a standalone curve.

  PYTHONPATH=games ./.venv/bin/python games/et6_eb.py --iters 60 --out games/results/et6_eb.json
"""
import argparse, json, random, time, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from collections import deque
from connect4 import C4, value01 as true_value01
from c4_net import C4Net
from c4_mcts import search, visit_probs
from c4_selfplay import self_play_game, LADDER, LADDER_FN, policy_move
from c4_calibrate import match, place_agent
from et6_decomp import net_value, net_probs, rollout_value


def build_probe(n_win, n_dec, rng):
    """Fixed probe: n_win solved WIN positions (for the search margin) + n_dec decisive (for fit/bias)."""
    wins, dec = [], []
    while len(wins) < n_win or len(dec) < n_dec:
        s = C4(); depth = rng.randint(14, 28); ok = True
        for _ in range(depth):
            if s.terminal() is not None or not s.legal():
                ok = False; break
            s = s.play(rng.choice(s.legal()))
        if not ok or s.terminal() is not None:
            continue
        v = true_value01(s)
        if v == 1.0 and len(wins) < n_win:
            wins.append(s)
        if v in (0.0, 1.0) and len(dec) < n_dec:
            dec.append((s, v))
    return wins, dec


def search_margin(net, wins, nprng, sims=64):
    """(MCTS winning-move rate) - (policy-head winning-move rate) on solved win positions."""
    def keeps_win(s, mv):
        return s.can_play(mv) and true_value01(s.play(mv)) == 0.0
    mcts_ok = pol_ok = 0
    for s in wins:
        root = search(net, s, sims=sims, dirichlet=0.0, rng=nprng)
        mv_mcts = int(visit_probs(root).argmax())
        p = net_probs(net, s); mv_pol = int(p.argmax())
        mcts_ok += keeps_win(s, mv_mcts); pol_ok += keeps_win(s, mv_pol)
    n = len(wins)
    return round(mcts_ok / n, 3), round(pol_ok / n, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--sims", type=int, default=64); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--channels", type=int, default=64); ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=6)
    ap.add_argument("--probe-win", type=int, default=40); ap.add_argument("--probe-dec", type=int, default=40)
    ap.add_argument("--probe-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="games/results/et6_eb.json")
    a = ap.parse_args()
    py_rng = random.Random(a.seed); np_rng = np.random.default_rng(a.seed)
    probe_rng = random.Random(9999)                                     # fixed probe across the whole run
    net = C4Net(a.channels, a.blocks); mx.eval(net.parameters())
    opt = optim.Adam(learning_rate=a.lr)

    def loss_fn(net, X, P, V):
        logits, value = net(X)
        logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        return -(P * logp).sum(axis=1).mean() + ((value - V) ** 2).mean()
    lg = nn.value_and_grad(net, loss_fn)

    wins, dec = build_probe(a.probe_win, a.probe_dec, probe_rng)
    print(f"[E-B] probe: {len(wins)} win positions, {len(dec)} decisive; instrumenting every {a.eval_every} iters", flush=True)

    def probe():
        net.eval()
        m_mcts, m_pol = search_margin(net, wins, np_rng, sims=a.sims)
        fit, bias = [], []
        for s, vstar in dec:
            vpi = rollout_value(net, s, a.probe_k, np_rng, mcts_sims=a.sims, npr=np_rng)
            fit.append(abs(net_value(net, s) - vpi)); bias.append(abs(vpi - vstar))
        raw = lambda st: policy_move(net, st)
        scores = [match(raw, LADDER_FN[k], 16, random.Random(a.seed + 700 + i)) for i, k in enumerate(LADDER)]
        gelo = place_agent(scores, np.array([LADDER[k] for k in LADDER]))
        return {"gelo": round(float(gelo), 0), "search_margin": round(m_mcts - m_pol, 3),
                "mcts_move_acc": m_mcts, "policy_move_acc": m_pol,
                "fit": round(float(np.mean(fit)), 3), "label_bias": round(float(np.mean(bias)), 3)}

    buf = deque(maxlen=a.buffer); hist = []; t0 = time.time()
    for it in range(1, a.iters + 1):
        net.eval()
        for _ in range(a.games):
            buf.extend(self_play_game(net, a.sims, np_rng, py_rng))
        net.train(); data = list(buf)
        for _ in range(a.epochs):
            py_rng.shuffle(data)
            for i in range(0, len(data), a.batch):
                ch = data[i:i + a.batch]
                X = mx.array([c[0] for c in ch]); P = mx.array(np.stack([c[1] for c in ch]))
                V = mx.array(np.array([c[2] for c in ch], dtype=np.float32))
                _, g = lg(net, X, P, V); opt.update(net, g); mx.eval(net.parameters(), opt.state)
        if it % a.eval_every == 0 or it == 1:
            row = {"iter": it, **probe()}
            hist.append(row)
            print(f"  it{it:>3} gelo={row['gelo']:+.0f} margin={row['search_margin']:+.3f} "
                  f"fit={row['fit']:.3f} bias={row['label_bias']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"config": vars(a), "history": hist}, open(a.out, "w"), indent=2)
    print(f"[E-B] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
