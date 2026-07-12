#!/usr/bin/env python
"""Generate a labeled Connect-4 dataset (the 'Stockfish labels' analog for the games arm).

Play diverse games with a shallow epsilon-greedy search, then label every visited position with the
depth-D alpha-beta's best move (policy target) and squashed score (value target). Feeds the Stage-1
net for the Connect-4 decomposition study.

  PYTHONPATH=games python games/connect4_gen.py --positions 12000 --label-depth 8 --out games/c4_data.npz
"""
import argparse, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from connect4 import C4
from connect4_ab import ab_best


def squash(v):
    if v >= 100000: return 1.0
    if v <= -100000: return 0.0
    return 1.0 / (1.0 + math.exp(-v / 40.0))


def gen_game(play_depth, eps, rng):
    s = C4(); positions = []
    while s.terminal() is None and s.legal():
        positions.append(s)
        c = rng.choice(s.legal()) if rng.random() < eps else ab_best(s, play_depth)[0]
        s = s.play(c)
    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=12000)
    ap.add_argument("--play-depth", type=int, default=4, help="shallow search for game trajectories")
    ap.add_argument("--label-depth", type=int, default=8, help="strong search for labels (teacher)")
    ap.add_argument("--eps", type=float, default=0.3, help="random-move rate for diversity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="games/c4_data.npz")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    X, M, V = [], [], []
    t0 = time.time()
    seen = set()
    while len(X) < args.positions:
        for s in gen_game(args.play_depth, args.eps, rng):
            k = s.key()
            if k in seen:
                continue
            seen.add(k)
            bc, sc = ab_best(s, args.label_depth)      # policy + value teacher
            if bc is None:
                continue
            X.append(s.encode()); M.append(bc); V.append(squash(sc))
            if len(X) >= args.positions:
                break
        if len(X) % 1000 < 40:
            el = time.time() - t0
            print(f"  {len(X)}/{args.positions}  ({el/max(len(X),1)*1000:.0f}ms/pos, "
                  f"ETA {el/max(len(X),1)*(args.positions-len(X))/60:.0f}min)", flush=True)

    np.savez(args.out, X=np.array(X, dtype=np.float32),
             M=np.array(M, dtype=np.int32), V=np.array(V, dtype=np.float32))
    print(f"[gen] wrote {args.out}: {len(X)} positions labeled at depth {args.label_depth} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
