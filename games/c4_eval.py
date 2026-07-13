#!/usr/bin/env python
"""Measure Connect-4 strength: open-loop (raw net) vs closed-loop (net+MCTS) vs the ab_best depth
ladder — the +286 (search-lift) analog on a simpler game.

Reports, for each agent, score vs each depth-D alpha-beta opponent, plus a direct open-vs-closed
head-to-head converted to an Elo gap (how much Elo the search adds on this fixed evaluator).

  PYTHONPATH=games ./.venv/bin/python games/c4_eval.py --net games/c4_net.safetensors \
      --sims 200 --games 30 --depths 1 2 3 4 5
"""
from __future__ import annotations
import argparse, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from connect4 import C4
from connect4_ab import ab_best
from c4_net import C4Net
from c4_mcts import mcts_move, policy_move


def play_game(fn0, fn1, rng, temp_open=2):
    """Play one game; return 0.5 (draw) or the winning player index (0/1).
    `temp_open` random opening plies (shared) so a match isn't the same deterministic game."""
    s = C4(); turn = 0; ply = 0
    fns = [fn0, fn1]
    while True:
        t = s.terminal()
        if t == "draw":
            return 0.5
        if t == "win":
            return 1 - turn                     # previous mover (not to-move) won
        if ply < temp_open:
            c = rng.choice(s.legal())
        else:
            c = fns[turn](s)
            if c not in s.legal():
                c = s.legal()[0]
        s = s.play(c); turn ^= 1; ply += 1


def match(fnA, fnB, games, rng):
    """Score of A over `games` games, alternating colors. Returns (scoreA_frac, w, d, l)."""
    sA = 0.0; w = d = l = 0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(fnA, fnB, rng)
            a = 1.0 if r == 0 else (0.5 if r == 0.5 else 0.0)
        else:
            r = play_game(fnB, fnA, rng)
            a = 1.0 if r == 1 else (0.5 if r == 0.5 else 0.0)
        sA += a
        w += (a == 1.0); d += (a == 0.5); l += (a == 0.0)
    return sA / games, w, d, l


def elo_gap(score):
    score = min(max(score, 1e-4), 1 - 1e-4)
    return 400.0 * math.log10(score / (1 - score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="games/c4_net.safetensors")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = C4Net(args.channels, args.blocks)
    net.load_weights(args.net)
    net.eval()
    mx.eval(net.parameters())
    rng = random.Random(args.seed)

    open_fn = lambda s: policy_move(net, s)
    closed_fn = lambda s: mcts_move(net, s, sims=args.sims)
    depth_fn = lambda d: (lambda s: ab_best(s, d)[0])

    print(f"[eval] net={args.net} | closed-loop sims={args.sims} | {args.games} games/opponent\n", flush=True)
    for name, fn in [("open-loop (raw net)", open_fn), (f"closed-loop (MCTS {args.sims})", closed_fn)]:
        print(f"== {name} ==", flush=True)
        t0 = time.time()
        for d in args.depths:
            sc, w, dr, l = match(fn, depth_fn(d), args.games, random.Random(args.seed + d))
            print(f"   vs depth-{d}:  score {sc*100:4.0f}%  (W{w} D{dr} L{l})   Elo {elo_gap(sc):+5.0f}",
                  flush=True)
        print(f"   ({time.time()-t0:.0f}s)\n", flush=True)

    # direct head-to-head: the search lift on the SAME evaluator
    sc, w, dr, l = match(closed_fn, open_fn, args.games, random.Random(args.seed + 99))
    print(f"== search lift (closed vs open, head-to-head) ==", flush=True)
    print(f"   closed-loop scores {sc*100:.0f}%  (W{w} D{dr} L{l})  ->  +{elo_gap(sc):.0f} Elo from search",
          flush=True)


if __name__ == "__main__":
    main()
