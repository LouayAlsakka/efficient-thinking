#!/usr/bin/env python
"""Stage 3 (evaluator-first self-training) on Connect-4 — testing the 'fix the evaluator first' idea,
instrumented against the perfect solver + the calibrated GELO ladder.

AlphaZero-style expert iteration, framed value-first:
  self-play with MCTS  ->  (visit-policy, game-OUTCOME) targets  ->  retrain  ->  repeat.
The value target is the real terminal outcome (the game-rules oracle = "play it out and see who wins" —
the efficient version of your '100 rollouts'). Each iteration we measure:
  * eval accuracy vs the TRUE solved value on a cached held-out set (does the EVALUATOR improve?),
  * GELO placement vs the pre-calibrated ladder (does STRENGTH follow, and WHERE does it plateau?).

  PYTHONPATH=games ./.venv/bin/python games/c4_selfplay.py --iters 40 --games 24 --sims 64
  PYTHONPATH=games ./.venv/bin/python games/c4_selfplay.py --init games/c4_net.safetensors  # beat-supervised
"""
from __future__ import annotations
import argparse, os, random, sys, time
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from connect4 import C4, value01 as true_value01
from connect4_ab import ab_best
from c4_net import C4Net
from c4_mcts import search, visit_probs, policy_move
from c4_calibrate import match, place_agent

# calibrated ladder GELO (from games/c4_calibrate.py, random := 0). Used to place the net each iter.
LADDER = {"random": 0.0, "depth-1": 804.0, "depth-2": 954.0, "depth-3": 963.0}
LADDER_FN = {"random": lambda s: random.choice(s.legal()),
             "depth-1": lambda s: ab_best(s, 1)[0],
             "depth-2": lambda s: ab_best(s, 2)[0],
             "depth-3": lambda s: ab_best(s, 3)[0]}


def self_play_game(net, sims, np_rng, py_rng, temp_moves=10):
    """One self-play game. Returns list of (encode[84], visit_probs[7], value in {0,0.5,1})."""
    s = C4(); turn = 0; ply = 0; hist = []
    while s.terminal() is None and s.legal():
        root = search(net, s, sims=sims, dirichlet=0.25, dir_alpha=1.0, rng=np_rng)
        vp = visit_probs(root)
        hist.append((s.encode(), vp, turn))
        if ply < temp_moves and vp.sum() > 0:
            move = int(np_rng.choice(7, p=vp))
        else:
            move = int(vp.argmax())
        if not s.can_play(move):
            move = s.legal()[0]
        s = s.play(move); turn ^= 1; ply += 1
    t = s.terminal()
    if t == "win":
        winner = 1 - turn
        val = lambda p: 1.0 if p == winner else 0.0
    else:
        val = lambda p: 0.5
    return [(enc, vp, val(p)) for (enc, vp, p) in hist]


def build_holdout(n, py_rng):
    """Cached late-game positions with TRUE solved values (fast because few empty squares).
    Class-balanced: equal win/loss positions (+ some draws) so decisive_sign_acc has a 50% baseline."""
    per = n // 2
    buckets = {1.0: [], 0.0: [], 0.5: []}
    tries = 0
    while (len(buckets[1.0]) < per or len(buckets[0.0]) < per) and tries < n * 200:
        tries += 1
        s = C4(); depth = py_rng.randint(16, 30); ok = True
        for _ in range(depth):
            if s.terminal() is not None or not s.legal():
                ok = False; break
            s = s.play(py_rng.choice(s.legal()))
        if not ok or s.terminal() is not None:
            continue
        v = true_value01(s)
        if len(buckets[v]) < (per if v != 0.5 else per // 2):
            buckets[v].append(s.encode())
    X = buckets[1.0] + buckets[0.0] + buckets[0.5]
    Y = [1.0] * len(buckets[1.0]) + [0.0] * len(buckets[0.0]) + [0.5] * len(buckets[0.5])
    return mx.array(X), np.array(Y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=24, help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=20000)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--eval-games", type=int, default=24)
    ap.add_argument("--holdout", type=int, default=150)
    ap.add_argument("--init", default=None, help="warm-start net (else from scratch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="games/results/c4_selfplay.json")
    args = ap.parse_args()

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    net = C4Net(args.channels, args.blocks)
    if args.init:
        net.load_weights(args.init); print(f"[selfplay] warm-start from {args.init}", flush=True)
    else:
        print("[selfplay] from scratch (random init)", flush=True)
    mx.eval(net.parameters())
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(net, X, P, V):
        logits, value = net(X)
        logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        pol = -(P * logp).sum(axis=1).mean()
        val = ((value - V) ** 2).mean()
        return pol + val
    lg = nn.value_and_grad(net, loss_fn)

    Xh, Yh = build_holdout(args.holdout, py_rng)
    print(f"[selfplay] holdout: {len(Yh)} solved positions "
          f"({int((Yh==1).sum())}W/{int((Yh==0.5).sum())}D/{int((Yh==0).sum())}L)", flush=True)

    def eval_iter():
        net.eval()
        logits, value = net(Xh)
        v = np.array(value.tolist())
        mae = float(np.mean(np.abs(v - Yh)))
        dec = Yh != 0.5
        sign = float(np.mean((v[dec] > 0.5) == (Yh[dec] > 0.5))) if dec.sum() else float("nan")
        raw = lambda s: policy_move(net, s)                            # open-loop = the evaluator alone
        scores = [match(raw, LADDER_FN[k], args.eval_games, random.Random(args.seed + 700 + i))
                  for i, k in enumerate(LADDER)]
        gelo = place_agent(scores, np.array([LADDER[k] for k in LADDER]))
        return mae, sign, gelo, scores

    buf = deque(maxlen=args.buffer)
    hist = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        net.eval()
        for g in range(args.games):
            buf.extend(self_play_game(net, args.sims, np_rng, py_rng))
        # train on the replay buffer
        net.train()
        data = list(buf)
        for _ in range(args.epochs):
            py_rng.shuffle(data)
            for i in range(0, len(data), args.batch):
                chunk = data[i:i + args.batch]
                X = mx.array([c[0] for c in chunk])
                P = mx.array(np.stack([c[1] for c in chunk]))
                V = mx.array(np.array([c[2] for c in chunk], dtype=np.float32))
                _, grads = lg(net, X, P, V)
                opt.update(net, grads); mx.eval(net.parameters(), opt.state)
        if it % args.eval_every == 0 or it == 1:
            mae, sign, gelo, scores = eval_iter()
            hist.append({"iter": it, "buf": len(buf), "value_mae": round(mae, 3),
                         "decisive_sign_acc": round(sign, 3), "gelo": round(gelo, 0),
                         "ladder_scores": {k: round(s, 2) for k, s in zip(LADDER, scores)}})
            print(f"  it{it:>3} buf={len(buf):>5}  eval-vs-oracle: MAE={mae:.3f} sign={sign*100:.0f}%"
                  f"   GELO={gelo:+.0f}   ({time.time()-t0:.0f}s)", flush=True)

    import json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"config": vars(args), "ladder": LADDER, "history": hist}, open(args.out, "w"), indent=2)
    net.save_weights(args.out.replace(".json", ".safetensors"))
    print(f"[selfplay] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
