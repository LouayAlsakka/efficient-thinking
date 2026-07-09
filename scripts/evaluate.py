#!/usr/bin/env python
"""Evaluate a trained run: regret metric + calibrated-ladder Elo.

The ladder mixes a random-mover anchor (~300 Elo) with Stockfish rungs, so a
model weaker than Stockfish's 1320 floor is still *bracketed* rather than
reading as noise. Both matches and the regret metric run from REAL-game
positions (in-distribution) when a PGN is given.

Example:
  PYTHONPATH=. python scripts/evaluate.py --run-dir runs/eff_real/d10_w1024_onehot_t1.0 \
      --ladder random,1320,1500,1700,1900 --games-per-rung 100 \
      --openings-pgn data/lichess/2013-01.pgn
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from chessnet.train import load_run
from chessnet.player import ModelPlayer
from chessnet.evaluate import regret_eval, run_match, estimate_elo, load_openings


def parse_ladder(spec: str) -> list[dict]:
    """'random,1320,1500' -> [{'kind':'random'}, {'kind':'sf_elo','elo':1320}, ...]"""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.lower() == "random":
            out.append({"kind": "random"})
        elif tok:
            out.append({"kind": "sf_elo", "elo": int(tok)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["masked", "raw", "reject"], default="masked")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--regret-positions", type=int, default=200)
    ap.add_argument("--regret-depth", type=int, default=12)
    ap.add_argument("--ladder", default="random,1320,1500,1700,1900",
                    help="comma list; 'random' anchor + Stockfish Elos")
    ap.add_argument("--games-per-rung", type=int, default=100)
    ap.add_argument("--movetime", type=float, default=0.05)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn",
                    help="real-game PGN for in-distribution openings + regret")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-matches", action="store_true")
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir)
    player = ModelPlayer(model, encoding=cfg.encoding, mode=args.mode,
                         temperature=args.temperature, seed=args.seed)

    report = {"run_dir": args.run_dir, "mode": args.mode,
              "depth": cfg.depth, "width": cfg.width,
              "encoding": cfg.encoding}

    have_pgn = args.openings_pgn and os.path.exists(args.openings_pgn)

    # regret / held-out move quality — from REAL positions when we have a PGN
    if have_pgn:
        boards = load_openings(args.openings_pgn, args.regret_positions,
                               min_ply=10, max_ply=60, seed=args.seed + 5)
    else:
        boards = _random_boards(args.regret_positions, args.seed)
    rr = regret_eval(player, boards, depth=args.regret_depth)
    report["regret"] = {
        "n": rr.n, "mean_regret": rr.mean_regret,
        "blunder_rate": rr.blunder_rate, "illegal_rate": rr.illegal_rate,
        "positions": "real-game" if have_pgn else "random-play"}
    print(f"[regret] n={rr.n} ({report['regret']['positions']}) "
          f"mean_regret={rr.mean_regret:.4f} "
          f"blunder_rate={rr.blunder_rate:.3f} illegal_rate={rr.illegal_rate:.3f}")

    # ladder matches + Elo
    if not args.skip_matches:
        openings = (load_openings(args.openings_pgn, 200, seed=args.seed + 1)
                    if have_pgn else None)
        print(f"openings: {'real-game' if openings else 'random-play'} "
              f"({len(openings) if openings else 0} positions)")
        ladder = parse_ladder(args.ladder)
        results = []
        for i, opp in enumerate(ladder):
            res = run_match(player, opp, args.games_per_rung,
                            movetime=args.movetime, openings=openings,
                            seed=args.seed + i)
            results.append(res)
            note = " (clamped to 1320 floor)" if res.clamped else ""
            print(f"[vs {res.opponent_name}{note}] W{res.wins} D{res.draws} "
                  f"L{res.losses}  score={res.score}/{res.games}")
        elo, margin = estimate_elo(results)
        report["elo"] = elo
        report["elo_margin"] = margin
        report["matches"] = [
            {"opponent": r.opponent_name, "opponent_elo": r.opponent_elo,
             "clamped": r.clamped, "wins": r.wins, "draws": r.draws,
             "losses": r.losses, "score": r.score, "games": r.games}
            for r in results]
        print(f"[ELO] {elo:.0f} +/- {margin:.0f}  "
              f"(bracketed by random~300 and SF ladder)")

    out = os.path.join(args.run_dir, f"eval_{args.mode}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
