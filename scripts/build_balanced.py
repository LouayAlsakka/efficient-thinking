#!/usr/bin/env python
"""Curate BALANCED start positions (opening / midgame / endgame, ~equal chances) for fast, high-
signal head-to-head fitness. Two similar nets from the normal opening mostly draw in long games
(weak, slow signal); starting from pre-balanced midgame/endgame positions gives shorter, more
decisive games -> more signal per second. Balance is judged by a ~2000 net's value head (we trust
a 2000 to call equality), and both colors are played from each position so residual bias cancels.

  PYTHONPATH=. python scripts/build_balanced.py --net runs/selfplay_par \
      --out runs/balanced_positions.json --per-phase 80 --band 0.06
"""
from __future__ import annotations
import argparse, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
import chess, chess.pgn

from chessnet.train import load_run
from chessnet.encoding import ENCODERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="run dir of a value-head net to judge balance")
    ap.add_argument("--pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--out", default="runs/balanced_positions.json")
    ap.add_argument("--per-phase", type=int, default=80)
    ap.add_argument("--band", type=float, default=0.06, help="keep if |Eval-0.5| <= band")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cfg = load_run(args.net)
    if not getattr(cfg, "value_head", False):
        sys.exit("net has no value head to judge balance")
    encode = ENCODERS[cfg.encoding]

    def evalv(board):
        x = mx.array(encode(board)[None, :])
        v = model(x, return_value=True)[1]
        return float(np.array(v)[0])

    # phase = (min_ply, max_ply, min_pieces) — endgame needs enough material to be non-trivial
    phases = {"opening": (8, 16, 20), "midgame": (22, 40, 10), "endgame": (46, 100, 7)}
    buckets = {k: [] for k in phases}
    rng = random.Random(args.seed)

    with open(args.pgn) as f:
        while not all(len(buckets[p]) >= args.per_phase for p in phases):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            moves = list(game.mainline_moves())
            for ph, (lo, hi, minpc) in phases.items():
                if len(buckets[ph]) >= args.per_phase or len(moves) < lo + 2:
                    continue
                k = rng.randint(lo, min(hi, len(moves) - 1))
                board = game.board()
                for mv in moves[:k]:
                    board.push(mv)
                if board.is_game_over() or not any(board.legal_moves):
                    continue
                if len(board.piece_map()) < minpc:
                    continue
                if abs(evalv(board) - 0.5) <= args.band:       # balanced per the value head
                    buckets[ph].append(board.fen())

    fens = [f for p in phases for f in buckets[p]]
    json.dump(fens, open(args.out, "w"))
    print(f"[balanced] wrote {len(fens)} positions -> {args.out}  "
          + " ".join(f"{p}={len(buckets[p])}" for p in phases), flush=True)


if __name__ == "__main__":
    main()
