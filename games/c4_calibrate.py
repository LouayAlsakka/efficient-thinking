#!/usr/bin/env python
"""Calibrate a real (anchored) rating scale for Connect-4 — the reference implementation of our
cross-domain 'Elo' methodology.

A rating is only meaningful relative to a *calibrated reference ladder*. Here the ladder is:
random (floor) + ab_best at depths 1..D (each rung = d-ply tactical lookahead). We:
  1. play a round-robin among the rungs (cross-table, not single-opponent win-rates),
  2. fit Bradley-Terry / Elo ratings by MLE (Zermelo MM iteration),
  3. CHECK the logistic actually fits (predicted vs observed pairwise win-rates) -- calibration first,
  4. anchor the scale (random := 0 Elo) and report each rung's rating + its meaning,
  5. THEN place an agent (our net) on the calibrated scale by its scores vs the ladder.

This replaces the earlier per-opponent 'Elo' (just a win-rate re-expressed) with a proper anchored
rating. Run without --net to just calibrate the ladder.

  PYTHONPATH=games ./.venv/bin/python games/c4_calibrate.py --depths 1 2 3 4 5 6 --games 40
  PYTHONPATH=games ./.venv/bin/python games/c4_calibrate.py --net games/c4_net.safetensors --sims 200
"""
from __future__ import annotations
import argparse, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from connect4 import C4
from connect4_ab import ab_best

LN10 = math.log(10.0)


def play_game(fn0, fn1, rng, temp_open=2):
    s = C4(); turn = 0; ply = 0; fns = [fn0, fn1]
    while True:
        t = s.terminal()
        if t == "draw":
            return 0.5
        if t == "win":
            return 1 - turn
        c = rng.choice(s.legal()) if ply < temp_open else fns[turn](s)
        if c not in s.legal():
            c = s.legal()[0]
        s = s.play(c); turn ^= 1; ply += 1


def match(fnA, fnB, games, rng):
    sA = 0.0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(fnA, fnB, rng); sA += 1.0 if r == 0 else (0.5 if r == 0.5 else 0.0)
        else:
            r = play_game(fnB, fnA, rng); sA += 1.0 if r == 1 else (0.5 if r == 0.5 else 0.0)
    return sA / games


def fit_bradley_terry(W, N, iters=1000):
    """W[i,j] = wins by i vs j (draws counted 0.5 each side); N[i,j] = games. Returns Elo ratings."""
    n = W.shape[0]
    r = np.ones(n)                                            # strength params (exp scale)
    wins = W.sum(axis=1)
    for _ in range(iters):
        rnew = r.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i != j and N[i, j] > 0:
                    denom += N[i, j] / (r[i] + r[j])
            if denom > 0 and wins[i] > 0:
                rnew[i] = wins[i] / denom
        rnew /= rnew.mean()                                  # fix scale (geometric drift)
        if np.max(np.abs(np.log(rnew) - np.log(r))) < 1e-9:
            r = rnew; break
        r = rnew
    elo = (400.0 / LN10) * np.log(r)                         # log-strength -> Elo units (400 = 10:1)
    return elo


def goodness_of_fit(elo, W, N):
    """Mean abs error between logistic-predicted and observed pairwise win-rates."""
    errs = []
    for i in range(len(elo)):
        for j in range(len(elo)):
            if i < j and N[i, j] > 0:
                pred = 1.0 / (1.0 + 10 ** (-(elo[i] - elo[j]) / 400.0))
                obs = W[i, j] / N[i, j]
                errs.append(abs(pred - obs))
    return float(np.mean(errs)) if errs else float("nan")


def place_agent(scores, ladder_elo):
    """1-param MLE: find the Elo R that best predicts an agent's scores vs each rung (fixed ratings)."""
    def negloglik(R):
        ll = 0.0
        for s, e in zip(scores, ladder_elo):
            p = 1.0 / (1.0 + 10 ** (-(R - e) / 400.0))
            p = min(max(p, 1e-6), 1 - 1e-6)
            ll += s * math.log(p) + (1 - s) * math.log(1 - p)   # s in [0,1] as expected score
        return -ll
    lo, hi = ladder_elo.min() - 800, ladder_elo.max() + 800    # ternary search on a convex-ish nll
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if negloglik(m1) < negloglik(m2): hi = m2
        else: lo = m1
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--games", type=int, default=40, help="games per round-robin pair")
    ap.add_argument("--net", default=None, help="optional: place this net on the calibrated scale")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="games/results/c4_rating_scale.md")
    args = ap.parse_args()

    # reference ladder: random floor + ab_best at each depth
    players = [("random", lambda s: random.choice(s.legal()))]
    for d in args.depths:
        players.append((f"depth-{d}", (lambda d: lambda s: ab_best(s, d)[0])(d)))
    names = [p[0] for p in players]
    n = len(players)
    print(f"[calibrate] ladder = {names} | {args.games} games/pair", flush=True)

    W = np.zeros((n, n)); N = np.zeros((n, n))
    t0 = time.time()
    for i in range(n):
        for j in range(i + 1, n):
            sA = match(players[i][1], players[j][1], args.games, random.Random(args.seed + i * 31 + j))
            W[i, j] = sA * args.games;      N[i, j] = args.games
            W[j, i] = (1 - sA) * args.games; N[j, i] = args.games
            print(f"   {names[i]:>8} vs {names[j]:<8} {sA*100:4.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    elo = fit_bradley_terry(W, N)
    elo = elo - elo[0]                                        # anchor: random := 0 Elo
    order = np.argsort(elo)
    gof = goodness_of_fit(elo, W, N)

    lines = ["# Connect-4 calibrated rating scale\n",
             f"Reference ladder, Bradley-Terry fit, anchored random := 0 Elo. "
             f"Logistic goodness-of-fit (mean |pred-obs| win-rate) = **{gof:.3f}** "
             f"(lower = the Elo/logistic model fits this game well).\n",
             "| rung | Elo | meaning |", "|---|---:|---|"]
    meaning = {"random": "floor (no lookahead)"}
    for d in args.depths:
        meaning[f"depth-{d}"] = f"{d}-ply tactical lookahead"
    for k in order:
        print(f"   {names[k]:>8}: {elo[k]:+7.0f} Elo", flush=True)
        lines.append(f"| {names[k]} | {elo[k]:+.0f} | {meaning.get(names[k],'')} |")
    print(f"\n[calibrate] logistic goodness-of-fit (mean |pred-obs|) = {gof:.3f}", flush=True)

    if args.net:
        import mlx.core as mx
        from c4_net import C4Net
        from c4_mcts import mcts_move, policy_move
        net = C4Net(args.channels, args.blocks); net.load_weights(args.net); net.eval(); mx.eval(net.parameters())
        open_fn = lambda s: policy_move(net, s)
        closed_fn = lambda s: mcts_move(net, s, sims=args.sims)
        lines.append("\n## Our agents, placed on the calibrated scale")
        lines.append("| agent | Elo | vs-ladder scores |")
        lines.append("|---|---:|---|")
        for label, fn in [("open-loop (raw net)", open_fn), (f"closed-loop (MCTS {args.sims})", closed_fn)]:
            scores = [match(fn, players[k][1], args.games, random.Random(args.seed + 900 + k)) for k in range(n)]
            R = place_agent(scores, elo)
            sc_str = " ".join(f"{names[k]}:{scores[k]*100:.0f}%" for k in range(n))
            print(f"   {label}: {R:+.0f} Elo   [{sc_str}]", flush=True)
            lines.append(f"| {label} | {R:+.0f} | {sc_str} |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"[calibrate] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
