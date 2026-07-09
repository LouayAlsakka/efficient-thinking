#!/usr/bin/env python
"""Test the committee assumption: does AGREEMENT predict CORRECTNESS?

Claim (to validate): several models started/trained differently will, on a given
position, either DIVERGE (disagree -> uncertain, no shared truth) or CONVERGE
(agree -> likely correct). If true, committee agreement is a free confidence meter
and the plurality vote should beat any single member.

Method: sample real positions; each model picks its (masked) top move; measure
agreement; use Stockfish (depth-limited = contention-tolerant, MEASUREMENT ONLY)
as ground truth via centipawn-loss (CPL) of each move vs SF's best. Then bucket
positions by agreement and report accuracy per bucket + consensus-vs-single CPL.

  PYTHONPATH=. python scripts/committee_test.py \
      --models runs/conv_value_llm1 runs/selfplay_warm runs/selfplay_warm2 \
      --positions 300 --depth 12
"""
from __future__ import annotations
import argparse, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess, chess.engine, chess.pgn

from chessnet.train import load_run
from chessnet.player import ModelPlayer

MATE_CP = 2000   # clamp mate scores to a large finite CPL


def sample_positions(pgn_path, n, seed, min_ply=8, max_ply=80):
    """Grab n positions at random plies from real games (varied game phases)."""
    rng = random.Random(seed)
    out = []
    with open(pgn_path) as f:
        while len(out) < n:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) < min_ply + 2:
                continue
            k = rng.randint(min_ply, min(max_ply, len(moves) - 1))
            board = game.board()
            for mv in moves[:k]:
                board.push(mv)
            if not board.is_game_over() and any(board.legal_moves):
                out.append(board.fen())
    return out


def pov_cp(score, mate_cp=MATE_CP):
    """PovScore (already POV of the side to move) -> centipawns, mates clamped."""
    return score.score(mate_score=mate_cp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="run dirs (>=2)")
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    players, names = [], []
    for d in args.models:
        model, cfg = load_run(d)
        players.append(ModelPlayer(model, encoding=cfg.encoding, mode="masked", seed=args.seed))
        names.append(os.path.basename(d.rstrip("/")))
    K = len(players)
    print(f"[committee] {K} members: {', '.join(names)}", flush=True)

    fens = sample_positions(args.pgn, args.positions, args.seed)
    print(f"[committee] {len(fens)} positions | SF depth {args.depth}", flush=True)
    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    limit = chess.engine.Limit(depth=args.depth)

    # accumulators, bucketed by how many members agree on the plurality move
    from collections import defaultdict
    buck_n = defaultdict(int); buck_cpl = defaultdict(float); buck_top1 = defaultdict(int)
    consensus_cpl = []; single_cpl = [[] for _ in range(K)]; best_single_cpl = []

    def move_cpl(board, mv, best_cp, cache):
        if mv in cache:
            return cache[mv]
        board.push(mv)
        if board.is_game_over():
            # terminal after move: score from the mover's view
            if board.is_checkmate():
                v = MATE_CP          # we delivered mate -> value +MATE for mover
            else:
                v = 0                # draw
        else:
            info = eng.analyse(board, limit)
            v = -pov_cp(info["score"].pov(board.turn))   # opp's POV -> negate for mover
        board.pop()
        cpl = max(0, best_cp - v)
        cache[mv] = cpl
        return cpl

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        mover = board.turn
        info = eng.analyse(board, limit)
        best_cp = pov_cp(info["score"].pov(mover))
        cache = {}
        moves = [players[j].choose(board).move for j in range(K)]
        # plurality vote
        counts = {}
        for m in moves:
            counts[m] = counts.get(m, 0) + 1
        top_move = max(counts, key=lambda m: counts[m])
        agree = counts[top_move]                      # 1..K
        cpl_cons = move_cpl(board, top_move, best_cp, cache)
        consensus_cpl.append(cpl_cons)
        buck_n[agree] += 1
        buck_cpl[agree] += cpl_cons
        buck_top1[agree] += int(top_move == info["pv"][0])
        cpls_here = []
        for j in range(K):
            c = move_cpl(board, moves[j], best_cp, cache)
            single_cpl[j].append(c); cpls_here.append(c)
        best_single_cpl.append(min(cpls_here))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(fens)}", flush=True)
    eng.quit()

    def mean(x): return sum(x) / max(1, len(x))
    print("\n==== AGREEMENT -> CORRECTNESS ====")
    print(f"{'agree/K':>8} {'n':>5} {'mean CPL':>9} {'top1-acc':>9}")
    for a in sorted(buck_n):
        n = buck_n[a]
        print(f"{a}/{K:>1}{'':>4} {n:>5} {buck_cpl[a]/n:>9.1f} {buck_top1[a]/n:>9.2%}")
    print("\n==== CONSENSUS vs SINGLE (mean CPL, lower=better) ====")
    print(f"  consensus (plurality vote): {mean(consensus_cpl):7.1f}")
    for j in range(K):
        print(f"  member {names[j]:24s}: {mean(single_cpl[j]):7.1f}")
    print(f"  (oracle best-of-committee) : {mean(best_single_cpl):7.1f}")
    print("\nIf mean CPL DROPS as agree/K rises -> convergence predicts correctness (claim holds).")
    print("If consensus CPL < best single member -> committee reduces bias (voting helps).")


if __name__ == "__main__":
    main()
