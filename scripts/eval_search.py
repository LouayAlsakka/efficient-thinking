#!/usr/bin/env python
"""Closed-loop test: does search on the value head beat the raw policy?

Loads a policy+value model and plays BOTH the raw policy (ModelPlayer, masked) and
the search player (SearchPlayer, beam-minimax on Eval) against the same SF ladder,
reporting Elo for each so the gain from closing the loop is direct.

  PYTHONPATH=. python scripts/eval_search.py --run-dir runs/conv_value_full2 \
      --beam 8 --depth 1 --ladder 1320,1500,1700 --games-per-rung 60
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessnet.train import load_run
from chessnet.player import ModelPlayer
from chessnet.search import (SearchPlayer, MCTSPlayer, CascadeSearchPlayer,
                             MultiStageMCTSPlayer)
from chessnet.evaluate import run_match, estimate_elo, load_openings


def elo_of(player, ladder, games, openings, movetime, seed):
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    results = [run_match(player, s, games, movetime=movetime, openings=openings,
                         seed=seed + i) for i, s in enumerate(specs)]
    elo, margin = estimate_elo(results)
    return elo, margin, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--method", choices=["ab", "mcts", "cascade", "mstage"], default="ab",
                    help="ab = alpha-beta beam-minimax; mcts = PUCT MCTS; "
                         "cascade = beam-minimax wide->deep funnel; "
                         "mstage = ALL-MCTS wide->narrow funnel (3 MCTS knob-settings)")
    ap.add_argument("--stages", default="20:1:0,8:3:2,3:6:4",
                    help="cascade stages as beam:depth:qdepth,... (wide->deep)")
    ap.add_argument("--mstages", default="8:150:3.0,3:250:1.5,1:400:0.5",
                    help="mstage stages as keep:sims:c_puct,... (wide->narrow; sims sum = budget)")
    ap.add_argument("--sims", type=int, default=200, help="mcts: simulations/move")
    ap.add_argument("--c-puct", type=float, default=1.5, help="mcts: exploration const")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--qdepth", type=int, default=0,
                    help="ab: quiescence extension depth at leaves (captures/promotions)")
    ap.add_argument("--ladder", default="1320,1500,1700")
    ap.add_argument("--games-per-rung", type=int, default=60)
    ap.add_argument("--movetime", type=float, default=0.03)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir)
    if not getattr(cfg, "value_head", False):
        sys.exit("model has no value head; train with --value-head")
    ladder = [int(x) for x in args.ladder.split(",")]
    openings = load_openings(args.openings_pgn, 200, seed=args.seed + 1)

    raw = ModelPlayer(model, encoding=cfg.encoding, mode="masked", seed=args.seed)
    if args.method == "mcts":
        srch = MCTSPlayer(model, encoding=cfg.encoding, sims=args.sims,
                          c_puct=args.c_puct, seed=args.seed)
        tag = f"mcts sims={args.sims}"
    elif args.method == "cascade":
        stages = [tuple(int(x) for x in s.split(":")) for s in args.stages.split(",")]
        srch = CascadeSearchPlayer(model, encoding=cfg.encoding, stages=stages,
                                   seed=args.seed)
        tag = "cascade " + " -> ".join(f"b{b}d{d}q{q}" for b, d, q in stages)
    elif args.method == "mstage":
        stages = [(int(k), int(s), float(c))
                  for k, s, c in (x.split(":") for x in args.mstages.split(","))]
        srch = MultiStageMCTSPlayer(model, encoding=cfg.encoding, stages=stages,
                                    seed=args.seed)
        tot = sum(s for _, s, _ in stages)
        tag = f"mstage({tot} sims) " + " -> ".join(f"k{k}s{s}c{c}" for k, s, c in stages)
    else:
        srch = SearchPlayer(model, encoding=cfg.encoding, beam=args.beam,
                            depth=args.depth, qdepth=args.qdepth, seed=args.seed)
        tag = f"ab depth={args.depth} beam={args.beam} q={args.qdepth}"

    print(f"[raw policy] evaluating {args.run_dir} ...")
    e_raw, m_raw, _ = elo_of(raw, ladder, args.games_per_rung, openings,
                             args.movetime, args.seed)
    print(f"  raw policy Elo = {e_raw:.0f} +/- {m_raw:.0f}")
    print(f"[{tag}] evaluating ...")
    e_s, m_s, _ = elo_of(srch, ladder, args.games_per_rung, openings,
                         args.movetime, args.seed)
    print(f"  search Elo     = {e_s:.0f} +/- {m_s:.0f}")
    print(f"\n>>> CLOSED-LOOP GAIN: {e_s - e_raw:+.0f} Elo "
          f"(raw {e_raw:.0f} -> search {e_s:.0f})")

    out = {"run_dir": args.run_dir, "beam": args.beam, "depth": args.depth,
           "raw_elo": e_raw, "raw_margin": m_raw,
           "search_elo": e_s, "search_margin": m_s, "gain": e_s - e_raw}
    with open(os.path.join(args.run_dir, "search_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.join(args.run_dir, 'search_eval.json')}")


if __name__ == "__main__":
    main()
