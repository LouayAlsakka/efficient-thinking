#!/usr/bin/env python
"""Trace the Connect-4 open-loop ceiling vs training data (the data lever on game #2).

Retrain the evaluator at increasing label counts, place each on the calibrated GELO scale (open-loop,
raw policy). Separates 'simpler game -> search dominates' from 'under-trained evaluator -> search
dominates': if open-loop climbs a lot with data, it was under-training; if it plateaus low, the game is
genuinely search-dominated.
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from c4_net import C4Net
from c4_mcts import policy_move
from c4_calibrate import match, place_agent
from connect4_ab import ab_best

LADDER = {"random": 0.0, "depth-1": 804.0, "depth-2": 954.0, "depth-3": 963.0}
LFN = {"random": lambda s: random.choice(s.legal()),
       "depth-1": lambda s: ab_best(s, 1)[0], "depth-2": lambda s: ab_best(s, 2)[0],
       "depth-3": lambda s: ab_best(s, 3)[0]}


def train(X, M, V, epochs=40, ch=64, bl=4):
    net = C4Net(ch, bl); mx.eval(net.parameters()); opt = optim.Adam(1e-3)
    def lf(net, x, m, v):
        lg, val = net(x)
        return nn.losses.cross_entropy(lg, m, reduction="mean") + nn.losses.mse_loss(val, v, reduction="mean"), None
    g = nn.value_and_grad(net, lf)
    n = len(X)
    for ep in range(epochs):
        net.train(); idx = np.random.permutation(n)
        for i in range(0, n, 256):
            j = idx[i:i + 256]
            (l, _), gr = g(net, mx.array(X[j]), mx.array(M[j]), mx.array(V[j]))
            opt.update(net, gr); mx.eval(net.parameters(), opt.state)
    net.eval(); return net


def main():
    d = np.load("games/c4_data_50k.npz")
    X, M, V = d["X"].astype(np.float32), d["M"].astype(np.int32), d["V"].astype(np.float32)
    results = []
    for size in [12000, 24000, 50000]:
        net = train(X[:size], M[:size], V[:size])
        raw = lambda s: policy_move(net, s)
        scores = [match(raw, LFN[k], 40, random.Random(1000 + size + i)) for i, k in enumerate(LADDER)]
        gelo = place_agent(scores, np.array([LADDER[k] for k in LADDER]))
        sc = {k: round(s, 2) for k, s in zip(LADDER, scores)}
        print(f"size={size:>6}: open-loop GELO={gelo:+.0f}   {sc}", flush=True)
        results.append({"size": size, "open_loop_gelo": round(gelo), "scores": sc})
    os.makedirs("games/results", exist_ok=True)
    json.dump(results, open("games/results/c4_ceiling.json", "w"), indent=2)
    print("[ceiling] wrote games/results/c4_ceiling.json", flush=True)


if __name__ == "__main__":
    main()
