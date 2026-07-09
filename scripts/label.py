#!/usr/bin/env python
"""Build labeled shards: get positions -> Stockfish best-move labels -> .npz.

Position sources (proposal 4.1):
  --source random   self-play with random legal moves (no download; good for
                    smoke tests and for the diverse-position tail).
  --source pgn --pgn FILE   sample positions from a PGN (e.g. a Lichess dump).

Labeling is fanned out across worker processes, each owning its own Stockfish
(proposal 6: "all P-cores"). Search budget is FIXED across the study (proposal 9).

Example (tiny smoke set):
  python scripts/label.py --source random --games 200 --out data/smoke \
      --depth 8 --workers 8 --positions-per-game 6
"""

from __future__ import annotations

import argparse
import os
import random
import time
from multiprocessing import Pool

import chess
import chess.pgn
import numpy as np

from chessnet.encoding import board_to_codes, move_to_index
from chessnet.dataset import (save_shard, PHASE_OPENING, PHASE_MIDDLE,
                              PHASE_END, PHASE_NAMES)
from chessnet.labeler import StockfishLabeler, LabelBudget, DEFAULT_STOCKFISH


# --- game phase classification (opening / middlegame / endgame) --------------
# We want the labeled set to span all three phases (proposal: beginning, middle,
# end games). Phase is defined by ply + material so it's cheap and unambiguous:
#   opening    : first OPENING_PLIES half-moves
#   endgame    : few pieces left on the board (<= ENDGAME_PIECES total)
#   middlegame : everything else
OPENING_PLIES = 16
ENDGAME_PIECES = 12


def classify_phase(board: chess.Board, ply: int) -> int:
    if ply < OPENING_PLIES:
        return PHASE_OPENING
    if len(board.piece_map()) <= ENDGAME_PIECES:
        return PHASE_END
    return PHASE_MIDDLE


# --- position sources: yield (FEN, phase) pairs -----------------------------

def gen_random_positions(n_games: int, per_game: int, seed: int, skip_plies: int):
    """Play random legal games; emit `per_game` mid-game (FEN, phase) pairs each."""
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board()
        picks = 0
        ply = 0
        while not board.is_game_over() and picks < per_game:
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            ply += 1
            if ply >= skip_plies and rng.random() < 0.5:
                yield board.fen(), classify_phase(board, ply)
                picks += 1


def gen_pgn_positions(pgn_path: str, max_games: int, per_game: int,
                      seed: int, skip_plies: int):
    """Sample (FEN, phase) from real games, stratified across game phases.

    For each game we bucket every position by phase, then draw an even share
    from each non-empty bucket so opening, middlegame and endgame are all
    represented (a game that ends before an endgame simply contributes none).
    """
    rng = random.Random(seed)
    per_phase = max(1, per_game // 3)
    with open(pgn_path) as fh:
        games = 0
        while games < max_games:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            games += 1
            board = game.board()
            # collect (fen, phase) for every eligible ply of this game
            buckets: dict[int, list[str]] = {PHASE_OPENING: [], PHASE_MIDDLE: [],
                                             PHASE_END: []}
            for i, mv in enumerate(game.mainline_moves()):
                board.push(mv)
                if i + 1 >= skip_plies:
                    buckets[classify_phase(board, i + 1)].append(board.fen())
            for phase, fens in buckets.items():
                if not fens:
                    continue
                k = min(per_phase, len(fens))
                for fen in rng.sample(fens, k):
                    yield fen, phase


# --- labeling worker --------------------------------------------------------

def _label_chunk(args):
    items, engine_path, budget_kwargs, topk = args
    budget = LabelBudget(**budget_kwargs)
    soft = budget.multipv > 1
    out_codes, out_meta, out_target, out_wp, out_phase = [], [], [], [], []
    out_sidx, out_swp = [], []
    with StockfishLabeler(engine_path, budget=budget) as lab:
        for fen, phase in items:
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            label = lab.label(board)
            best = chess.Move.from_uci(label.best_uci)
            codes, meta = board_to_codes(board)
            mirrored = board.turn == chess.BLACK
            out_codes.append(codes)
            out_meta.append(meta)
            out_target.append(move_to_index(best, mirrored))
            out_wp.append(label.best_winprob)
            out_phase.append(phase)
            if soft:
                # per-move advantage map: top-K legal moves by win-prob ->
                # (move index in oriented frame, win-prob), padded to K with -1.
                ranked = sorted(label.move_winprobs.items(),
                                key=lambda kv: kv[1], reverse=True)[:topk]
                idxs = np.full(topk, -1, dtype=np.int16)
                wps = np.full(topk, -1.0, dtype=np.float32)
                for j, (uci, wp) in enumerate(ranked):
                    idxs[j] = move_to_index(chess.Move.from_uci(uci), mirrored)
                    wps[j] = wp
                out_sidx.append(idxs)
                out_swp.append(wps)
    return (out_codes, out_meta, out_target, out_wp, out_phase,
            out_sidx, out_swp)


def chunked(seq, size):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["random", "pgn"], default="random")
    ap.add_argument("--pgn", help="PGN file when --source pgn")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--positions-per-game", type=int, default=6)
    ap.add_argument("--skip-plies", type=int, default=8,
                    help="don't sample the first K plies (opening book noise)")
    ap.add_argument("--out", required=True, help="output shard prefix, e.g. data/smoke")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--soft", action="store_true",
                    help="multi-PV: store per-move advantage map for soft-target "
                         "distributional training (not best-move imitation)")
    ap.add_argument("--topk", type=int, default=16,
                    help="moves per position to store in --soft mode")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--shard-size", type=int, default=1_000_000)
    ap.add_argument("--engine", default=DEFAULT_STOCKFISH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup", action="store_true", help="drop duplicate FENs")
    args = ap.parse_args()

    if args.source == "pgn":
        assert args.pgn, "--pgn required for --source pgn"
        fens = gen_pgn_positions(args.pgn, args.games, args.positions_per_game,
                                 args.seed, args.skip_plies)
    else:
        fens = gen_random_positions(args.games, args.positions_per_game,
                                    args.seed, args.skip_plies)

    if args.dedup:
        seen = set()
        def _uniq(it):
            for fen, phase in it:
                key = " ".join(fen.split(" ")[:4])  # ignore move clocks
                if key not in seen:
                    seen.add(key)
                    yield fen, phase
        fens = _uniq(fens)

    multipv = args.topk if args.soft else 1
    budget_kwargs = dict(depth=args.depth, nodes=args.nodes, multipv=multipv)
    tasks = ((chunk, args.engine, budget_kwargs, args.topk)
             for chunk in chunked(fens, args.chunk))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_codes, all_meta, all_target, all_wp, all_phase = [], [], [], [], []
    all_sidx, all_swp = [], []
    phase_counts = {PHASE_OPENING: 0, PHASE_MIDDLE: 0, PHASE_END: 0}
    shard_idx, total = 0, 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, all_codes, all_meta, all_target, all_wp, all_phase
        nonlocal all_sidx, all_swp
        if not all_target:
            return
        path = f"{args.out}.{shard_idx:04d}.npz"
        sidx = np.stack(all_sidx) if args.soft else None
        swp = np.stack(all_swp) if args.soft else None
        save_shard(path, np.stack(all_codes), np.stack(all_meta),
                   np.array(all_target), np.array(all_wp), np.array(all_phase),
                   soft_idx=sidx, soft_wp=swp)
        print(f"  wrote {path} ({len(all_target)} positions"
              f"{', soft top-%d' % args.topk if args.soft else ''})")
        shard_idx += 1
        all_codes, all_meta, all_target, all_wp, all_phase = [], [], [], [], []
        all_sidx, all_swp = [], []

    with Pool(processes=args.workers) as pool:
        for c, m, t, w, ph, si, sw in pool.imap_unordered(_label_chunk, tasks):
            all_codes.extend(c); all_meta.extend(m)
            all_target.extend(t); all_wp.extend(w); all_phase.extend(ph)
            all_sidx.extend(si); all_swp.extend(sw)
            for p in ph:
                phase_counts[p] += 1
            total += len(t)
            if total and total % (args.chunk * args.workers) < args.chunk:
                rate = total / (time.time() - t0)
                print(f"labeled {total} positions ({rate:.0f}/s)")
            if len(all_target) >= args.shard_size:
                flush()
    flush()
    dt = time.time() - t0
    dist = " ".join(f"{PHASE_NAMES[p]}={n}" for p, n in phase_counts.items())
    print(f"DONE: {total} positions in {dt:.1f}s ({total/max(dt,1e-9):.0f}/s) "
          f"across {shard_idx} shard(s)")
    print(f"phase distribution: {dist}")


if __name__ == "__main__":
    main()
