#!/usr/bin/env python
"""Which lever binds — DATA vs CAPACITY vs SEARCH? A Connect-4 diagnostic grid.

Train the evaluator across a grid of (training-data size × network capacity), and for each cell measure
BOTH open-loop GELO (raw net) and closed-loop GELO (net + MCTS). Reading the grid tells you *where* each
limitation dominates: where more data still helps (data-bound), where a bigger net helps (capacity-bound),
and where search compensates for a weak evaluator (search-bound). The direct 'where does training-data
size matter vs other limitations' experiment.

  PYTHONPATH=games ./.venv/bin/python games/c4_diagnostic.py
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from c4_net import C4Net
from c4_mcts import policy_move, mcts_move
from c4_calibrate import match, place_agent
from connect4_ab import ab_best

LADDER = {"random": 0.0, "depth-1": 804.0, "depth-2": 954.0, "depth-3": 963.0}
LFN = {"random": lambda s: random.choice(s.legal()),
       "depth-1": lambda s: ab_best(s, 1)[0], "depth-2": lambda s: ab_best(s, 2)[0],
       "depth-3": lambda s: ab_best(s, 3)[0]}
LVEC = np.array([LADDER[k] for k in LADDER])


def train(X, M, V, ch, bl, epochs=40):
    net = C4Net(ch, bl); mx.eval(net.parameters()); opt = optim.Adam(1e-3)
    def lf(net, x, m, v):
        lg, val = net(x)
        return nn.losses.cross_entropy(lg, m, reduction="mean") + nn.losses.mse_loss(val, v, reduction="mean"), None
    g = nn.value_and_grad(net, lf)
    n = len(X)
    for _ in range(epochs):
        net.train(); idx = np.random.permutation(n)
        for i in range(0, n, 256):
            j = idx[i:i + 256]
            (l, _), gr = g(net, mx.array(X[j]), mx.array(M[j]), mx.array(V[j]))
            opt.update(net, gr); mx.eval(net.parameters(), opt.state)
    net.eval(); return net


def place(move_fn, seed, games=24):
    scores = [match(move_fn, LFN[k], games, random.Random(seed + i)) for i, k in enumerate(LADDER)]
    return round(place_agent(scores, LVEC))


def main():
    d = np.load("games/c4_data_50k.npz")
    X, M, V = d["X"].astype(np.float32), d["M"].astype(np.int32), d["V"].astype(np.float32)
    caps = [(32, 2, "small~0.3M"), (96, 6, "large~4M")]
    sizes = [3000, 12000, 24000, 50000]
    grid = []
    t0 = time.time()
    print(f"{'data':>7} {'capacity':>12} | {'open-loop':>9} {'closed(MCTS100)':>15} {'search lift':>11}", flush=True)
    for ch, bl, cname in caps:
        for size in sizes:
            net = train(X[:size], M[:size], V[:size], ch, bl)
            og = place(lambda s: policy_move(net, s), 100)
            cg = place(lambda s: mcts_move(net, s, sims=100), 200)
            grid.append({"data": size, "cap": cname, "open": og, "closed": cg, "lift": cg - og})
            print(f"{size:>7} {cname:>12} | {og:>+9} {cg:>+15} {cg-og:>+11}   ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs("games/results", exist_ok=True)
    json.dump(grid, open("games/results/c4_diagnostic.json", "w"), indent=2)
    print("\n[diagnostic] wrote games/results/c4_diagnostic.json", flush=True)
    print("Read: along a row (fixed capacity) rising open-loop = DATA-bound; across rows at fixed data,", flush=True)
    print("gap between small/large = CAPACITY-bound; closed-minus-open = how much SEARCH compensates.", flush=True)


if __name__ == "__main__":
    main()
