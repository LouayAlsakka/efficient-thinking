#!/usr/bin/env python
"""Cascade vs flat MCTS — a direct, fixed-budget HEAD-TO-HEAD (the significance test R. Coulom asked for).

Both players use the SAME network and the SAME total simulation budget B; one plays flat MCTS-B, the
other the N-stage wide->narrow cascade. We play many games from varied openings, alternating colours,
and report the cascade's score, the implied Elo difference, and a 95% confidence interval. A paired
head-to-head is far more sensitive to a strength *difference* than each side's absolute ladder rating
(which carried +-89). Incremental-save so partial runs are usable.

  PYTHONPATH=. python scripts/cascade_headtohead.py --run-dir runs/conv_value_llm1 --casc-N 10 \
      --sims 800 --games 400
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
from chessnet.train import load_run
from chessnet.search import MultiStageMCTSPlayer
from chessnet.evaluate import load_openings
from cascade_sweep import make_stages


def elo(x):
    x = min(max(x, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10((1 - x) / x)


def play_game(white_p, black_p, opening, extra_plies=300):
    b = opening.copy()
    stop = b.ply() + extra_plies
    while not b.is_game_over(claim_draw=True) and b.ply() < stop:
        p = white_p if b.turn == chess.WHITE else black_p
        mc = p.choose(b)
        if mc.move is None or mc.move not in b.legal_moves:
            break
        b.push(mc.move)
    return b.result(claim_draw=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/conv_value_llm1")
    ap.add_argument("--casc-N", type=int, default=10, help="cascade stage count (flat = 1)")
    ap.add_argument("--sims", type=int, default=800, help="identical total budget for both sides")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--pgn", default="data/lichess/2013-12.pgn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/cascade_h2h.json")
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir)
    flat = MultiStageMCTSPlayer(model, encoding=cfg.encoding, stages=make_stages(1, args.sims), seed=1)
    casc = MultiStageMCTSPlayer(model, encoding=cfg.encoding, stages=make_stages(args.casc_N, args.sims), seed=2)
    openings = load_openings(args.pgn, max(64, args.games // 2))
    print(f"[h2h] cascade(N={args.casc_N}) vs flat MCTS-{args.sims} | {args.games} games | net={args.run_dir}", flush=True)
    print(f"      cascade stages: {make_stages(args.casc_N, args.sims)}", flush=True)

    sA = 0.0; w = d = l = 0
    t0 = time.time()
    for g in range(args.games):
        ob = openings[g % len(openings)]
        casc_white = (g % 2 == 0)
        r = play_game(casc, flat, ob) if casc_white else play_game(flat, casc, ob)
        s = 0.5 if r == "1/2-1/2" else (1.0 if (r == "1-0") == casc_white else 0.0)
        sA += s; w += (s == 1.0); d += (s == 0.5); l += (s == 0.0)
        n = g + 1
        if n % 10 == 0 or n == args.games:
            frac = sA / n
            se = math.sqrt(max(frac * (1 - frac), 1e-9) / n)
            lo, hi = elo(frac - 1.96 * se), elo(frac + 1.96 * se)
            print(f"  {n}/{args.games}  cascade score {frac*100:5.1f}% (W{w} D{d} L{l})  "
                  f"Elo diff {elo(frac):+5.0f}  95%CI [{lo:+.0f}, {hi:+.0f}]  ({time.time()-t0:.0f}s)", flush=True)
            json.dump({"run_dir": args.run_dir, "casc_N": args.casc_N, "sims": args.sims,
                       "games": n, "cascade_score": round(frac, 4), "W": w, "D": d, "L": l,
                       "elo_diff": round(elo(frac), 1), "ci95": [round(lo, 1), round(hi, 1)]},
                      open(args.out, "w"), indent=2)
    frac = sA / args.games
    se = math.sqrt(max(frac * (1 - frac), 1e-9) / args.games)
    lo, hi = elo(frac - 1.96 * se), elo(frac + 1.96 * se)
    verdict = "indistinguishable from" if lo <= 0 <= hi else ("STRONGER than" if lo > 0 else "WEAKER than")
    print(f"[h2h] done. {args.games} games: cascade {frac*100:.1f}%, Elo diff {elo(frac):+.0f} "
          f"[{lo:+.0f}, {hi:+.0f}] -> cascade is {verdict} flat MCTS-{args.sims}.", flush=True)


if __name__ == "__main__":
    main()
