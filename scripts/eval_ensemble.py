#!/usr/bin/env python
"""Ensemble-evaluator MCTS vs single-model MCTS at EQUAL compute.

An ensemble of K models at N sims uses K*N value-net forward passes per move; the fair baseline
is a single model at K*N sims. Question: does DE-BIASING the leaf evaluation (average K diverse
models) beat spending that same compute on more single-model simulations?

  PYTHONPATH=. python scripts/eval_ensemble.py --models runs/cmte_a runs/cmte_dataB runs/cmte_dataC \
      --sims 200 --ladder 1700,2000,2300 --games-per-rung 20
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessnet.train import load_run
from chessnet.search import MCTSPlayer, EnsembleMCTSPlayer
from chessnet.evaluate import run_match, estimate_elo, load_openings


def elo_of(player, ladder, games, openings, movetime, seed):
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    res = [run_match(player, s, games, movetime=movetime, openings=openings, seed=seed + i)
           for i, s in enumerate(specs)]
    return estimate_elo(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="K value-head nets to ensemble")
    ap.add_argument("--baseline", default=None, help="single-model baseline dir (default models[0])")
    ap.add_argument("--sims", type=int, default=200, help="ensemble sims/move (baseline gets K*this)")
    ap.add_argument("--elo-weights", default=None, help="comma weights, e.g. 2000,2050,2010")
    ap.add_argument("--ladder", default="1700,2000,2300")
    ap.add_argument("--games-per-rung", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=0.04)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = [load_run(d)[0] for d in args.models]
    _, cfg = load_run(args.models[0])
    K = len(models)
    weights = [float(x) for x in args.elo_weights.split(",")] if args.elo_weights else None
    ladder = [int(x) for x in args.ladder.split(",")]
    openings = load_openings(args.openings_pgn, 200, seed=args.seed + 1)

    base_model, _ = load_run(args.baseline) if args.baseline else (models[0], None)
    ens = EnsembleMCTSPlayer(models, encoding=cfg.encoding, sims=args.sims, weights=weights,
                             seed=args.seed)
    base = MCTSPlayer(base_model, encoding=cfg.encoding, sims=args.sims * K, seed=args.seed)

    tagw = "ELO-weighted" if weights else "equal-weight"
    print(f"[ensemble] K={K} models, {tagw}, {args.sims} sims each ({K*args.sims} passes/move)",
          flush=True)
    e_ens, m_ens = elo_of(ens, ladder, args.games_per_rung, openings, args.movetime, args.seed)
    print(f"  ENSEMBLE-eval MCTS  Elo = {e_ens:.0f} +/- {m_ens:.0f}", flush=True)
    print(f"[baseline] single model @ {args.sims*K} sims (same forward passes)", flush=True)
    e_b, m_b = elo_of(base, ladder, args.games_per_rung, openings, args.movetime, args.seed)
    print(f"  SINGLE-model  MCTS  Elo = {e_b:.0f} +/- {m_b:.0f}", flush=True)
    print(f"\n>>> ENSEMBLE-EVAL GAIN: {e_ens - e_b:+.0f} Elo "
          f"(single {e_b:.0f} -> ensemble {e_ens:.0f}) at equal compute", flush=True)


if __name__ == "__main__":
    main()
