#!/usr/bin/env python
"""Overnight sweep: N-level all-MCTS cascade for N=1..10, one consistent rule per N.
Measures Elo (tall SF ladder) AND speed (ms/move) so we can plot score & speed vs #levels.
Writes runs/cascade_sweep.json incrementally (crash-safe).

  PYTHONPATH=. python scripts/cascade_sweep.py --run-dir runs/conv_value_llm1 \
      --ladder 2400,2700,3000 --games-per-rung 20 --nmax 10
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chess

from chessnet.train import load_run
from chessnet.player import ModelPlayer
from chessnet.search import MultiStageMCTSPlayer
from chessnet.evaluate import run_match, estimate_elo, load_openings

# fixed probe positions for the speed measurement (open, middlegame, tactical)
FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "r3k2r/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQ1RK1 w kq - 0 9",
    "2r3k1/1p3ppp/p1n1p3/3pP3/1b1P4/2N2N2/PP3PPP/2R3K1 w - - 0 20",
]


def make_stages(N, total=800):
    """N stages, same rule: keep shrinks 20->1 (geometric), sims grow (sum=total),
    c_puct falls 4.5->0.2. N=1 is plain flat MCTS-800."""
    if N == 1:
        return [(1, total, 1.5)]
    keeps = [max(1, int(round(k))) for k in np.geomspace(20, 1, N)]
    keeps[-1] = 1
    w = np.linspace(1.0, 2.4, N)
    sims = np.round(w / w.sum() * total).astype(int)
    sims[-1] += total - int(sims.sum())
    cpuct = np.linspace(4.5, 0.2, N)
    return [(int(keeps[i]), int(sims[i]), round(float(cpuct[i]), 2)) for i in range(N)]


def elo_of(player, ladder, games, openings, movetime, seed):
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    res = [run_match(player, s, games, movetime=movetime, openings=openings, seed=seed + i)
           for i, s in enumerate(specs)]
    return estimate_elo(res)


def speed_ms(player):
    ts = []
    for f in FENS:
        b = chess.Board(f)
        t = time.time(); player.choose(b); ts.append(time.time() - t)
    return float(np.mean(ts) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ladder", default="2400,2700,3000")
    ap.add_argument("--games-per-rung", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=0.04)
    ap.add_argument("--pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/cascade_sweep.json")
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir)
    ladder = [int(x) for x in args.ladder.split(",")]
    openings = load_openings(args.pgn, 200, seed=args.seed + 1)

    # raw-policy reference line (measured once)
    raw = ModelPlayer(model, encoding=cfg.encoding, mode="masked", seed=args.seed)
    raw_elo, raw_margin = elo_of(raw, ladder, args.games_per_rung, openings,
                                 args.movetime, args.seed)
    print(f"[sweep] raw policy Elo = {raw_elo:.0f} +/- {raw_margin:.0f}", flush=True)

    out = {"run_dir": args.run_dir, "ladder": ladder, "games_per_rung": args.games_per_rung,
           "movetime": args.movetime, "sims": args.sims,
           "raw_elo": round(raw_elo), "raw_margin": round(raw_margin), "points": []}

    for N in range(1, args.nmax + 1):
        stages = make_stages(N, args.sims)
        player = MultiStageMCTSPlayer(model, encoding=cfg.encoding, stages=stages,
                                      seed=args.seed)
        ms = speed_ms(player)
        elo, margin = elo_of(player, ladder, args.games_per_rung, openings,
                             args.movetime, args.seed + 100 * N)
        pt = {"N": N, "elo": round(elo), "margin": round(margin),
              "ms_per_move": round(ms), "stages": stages}
        out["points"].append(pt)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[sweep] N={N:2d}  Elo={elo:4.0f}+/-{margin:.0f}  {ms:4.0f} ms/move  "
              f"stages={stages}", flush=True)

    print("[sweep] DONE", flush=True)


if __name__ == "__main__":
    main()
