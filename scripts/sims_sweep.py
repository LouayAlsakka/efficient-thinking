#!/usr/bin/env python
"""Marginal value of MCTS sims: play MCTS-N vs MCTS-baseline head-to-head to find where more search
stops helping (the value-net ceiling). No ladder -> no ceiling-compression artifact; the win rate
of N over the baseline is a direct Elo delta. Diminishing deltas as N grows = saturation.

  PYTHONPATH=. python scripts/sims_sweep.py --run-dir runs/conv_value_llm1 \
      --baseline 800 --sweep 1600,3200,6400 --games 30
"""
from __future__ import annotations
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess
from chessnet.train import load_run
from chessnet.search import MCTSPlayer
from chessnet.evaluate import load_openings


def s2e(s, n):
    eps = 0.5 / max(1, n); s = min(max(s, eps), 1 - eps)
    return 400.0 * math.log10(s / (1 - s))


def play_pair(pa, pb, boards, n, rng, max_moves):
    total = 0.0
    for g in range(n):
        b = boards[rng.integers(len(boards))].copy()
        a_white = (g % 2 == 0)
        seat = {chess.WHITE: pa if a_white else pb, chess.BLACK: pb if a_white else pa}
        ply = 0
        while not b.is_game_over(claim_draw=True) and ply < max_moves:
            mv = seat[b.turn].choose(b).move
            if mv is None:
                break
            b.push(mv); ply += 1
        r = b.result(claim_draw=True)
        total += (1.0 if a_white else 0.0) if r == "1-0" else \
                 (0.0 if a_white else 1.0) if r == "0-1" else 0.5
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline", type=int, default=800)
    ap.add_argument("--sweep", default="1600,3200,6400")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--max-moves", type=int, default=160)
    ap.add_argument("--pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir)
    boards = load_openings(args.pgn, 200, seed=args.seed + 2)
    rng = np.random.default_rng(args.seed)
    sweep = [int(x) for x in args.sweep.split(",")]
    print(f"[sims-sweep] MCTS-N vs MCTS-{args.baseline} baseline, {args.games} games each", flush=True)
    prev = args.baseline
    for N in sweep:
        pa = MCTSPlayer(model, encoding=cfg.encoding, sims=N, seed=args.seed)
        pb = MCTSPlayer(model, encoding=cfg.encoding, sims=args.baseline, seed=args.seed + 1)
        s = play_pair(pa, pb, boards, args.games, rng, args.max_moves)
        print(f"  MCTS-{N:<5d} vs MCTS-{args.baseline}: score {s:.3f}  ->  {s2e(s, args.games):+.0f} "
              f"Elo over the {args.baseline}-sim baseline", flush=True)
    print("[sims-sweep] DONE  (shrinking deltas as N grows => search saturating on this value net)",
          flush=True)


if __name__ == "__main__":
    main()
