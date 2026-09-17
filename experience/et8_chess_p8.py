#!/usr/bin/env python3
"""ET-8 P8: does an experience prior buy SEARCH? Head-to-head, pre-registered by 理 (10916).

    # 0. the instrument check -- ALWAYS run this before spending the GPU
    python3 experience/et8_chess_p8.py selftest --run-dir runs/conv_value_llm1 --prior <prior.json>

    # 1. the measurement
    python3 experience/et8_chess_p8.py sweep --run-dir runs/conv_value_llm1 --prior <prior.json> \
        --pgn /Users/lab/chess-scaling/data/lichess/2013-01.pgn --games 80 --out <out.json>

THE DESIGN, 理 10916, written here because a pre-registration that lives only in a channel is not
one:

    headline    HEAD-TO-HEAD, not the ladder: "simulations to equal strength". The ladder runs
                once, no prior, at baseline sims, only to NAME the rung in Stockfish Elo.
    baseline    frozen conv_value_llm1, NO prior, 256 sims.
    arms        WITH prior at {64, 128, 256} vs baseline-256      <- the claim
                WITHOUT prior at {64, 128}   vs baseline-256      <- the control curve: what more
                                                                     simulations alone would buy
    games       80 per pairing, colours balanced (the harness alternates seats every game).
    §4.2a       constraint: WITH-256 vs baseline-256 >= 45% (strength held at equal sims)
                objective:  WITH-128 >= 45% WHILE WITHOUT-128 < 45% -- the prior buys a 2x
                            simulation saving that simulations alone do not. WITH-64 >= 45% is 4x.
    readings    PASS          same strength at half the search.
                FAIL-cost     WITH-128 < 45%: the prior does not buy search.
                FAIL-capab.   WITH-256 < 45%: the prior re-weights search into WEAKER play -- a
                              constraint failure, the same shape as A/C/F.

WHY `selftest` EXISTS AND IS NOT OPTIONAL. ExperienceMCTSPlayer works by overriding
MCTSPlayer._policy_value. If that hook ever stops being the path expansion takes -- a batched
player, a renamed method, a cache returning before the override -- the prior silently does nothing
and every arm returns the baseline's own score. That reads as a clean negative result. It has
happened to me in this programme already, with 3B steering vectors loaded against a 7B model:
the run completed, produced numbers, and measured nothing. `selftest` asserts the prior CHANGES
the move distribution before any pairing is played, and the sweep refuses to run if it does not.

BOUNDS THE HARNESS CANNOT REMOVE. Self-play against the same evaluator measures what the prior
does to THIS net's search, not whether the moves are good. 80 games gives SE ~= 5.6 points of win
rate, so a 45% threshold and a 50% result are about one SE apart -- close calls are not resolvable
at this n and will be reported as close rather than as passes.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "scripts"))


def s2e(s, n):
    eps = 0.5 / max(1, n); s = min(max(s, eps), 1 - eps)
    return 400.0 * math.log10(s / (1 - s))


def build(model, cfg, sims, seed, prior=None, beta=1.0):
    from chessnet.search import MCTSPlayer
    if prior is None:
        return MCTSPlayer(model, encoding=cfg.encoding, sims=sims, seed=seed)
    import et8_chess_prior as P
    cls = P.make_player_class()
    return cls(model, prior=prior, beta=beta, encoding=cfg.encoding, sims=sims, seed=seed)


def load_all(a):
    from chessnet.train import load_run
    import et8_chess_prior as P
    model, cfg = load_run(a.run_dir)
    prior = P.ExperiencePrior.load(a.prior)
    return model, cfg, prior


def cmd_selftest(a):
    """Assert the prior actually reaches the search. Three checks, each able to fail alone."""
    import numpy as np, chess
    model, cfg, prior = load_all(a)
    base = build(model, cfg, 32, 1)
    withp = build(model, cfg, 32, 1, prior=prior, beta=a.beta)

    # 1. the hook is the path expansion takes
    ok_hook = type(withp)._policy_value is not type(base)._policy_value
    # 2. the reweighted priors DIFFER from the raw ones -- measured over MANY positions and at
    #    several betas, not one. A single position can move by chance; and a prior that is present
    #    but INERT at the chosen beta produces a null result that looks like a finding. Reporting
    #    the beta curve makes the strength of the intervention evidence rather than an assumption.
    import chess.pgn as _pgn  # noqa: F401  (import guarded here so selftest works without a pgn)
    boards = []
    b = chess.Board()
    rng = np.random.default_rng(0)
    while len(boards) < a.positions:
        if b.is_game_over(claim_draw=True) or b.fullmove_number > 60:
            b = chess.Board(); continue
        mv = list(b.legal_moves)[int(rng.integers(len(list(b.legal_moves))))]
        b.push(mv)
        if b.fullmove_number >= 4 and not b.is_game_over(claim_draw=True):
            boards.append(b.copy())
    beta_curve = {}
    for bt in (0.5, 1.0, 2.0, 4.0):
        w = build(model, cfg, 32, 1, prior=prior, beta=bt)
        d = []
        for bd in boards:
            legal, p_raw, _ = type(base)._policy_value(base, bd)
            _, p_new, _ = type(w)._policy_value(w, bd)
            d.append(float(np.abs(np.asarray(p_new) - np.asarray(p_raw)).sum()))
        beta_curve[str(bt)] = {"mean_L1": round(float(np.mean(d)), 5),
                               "max_L1": round(float(np.max(d)), 5),
                               "positions_unchanged": int(sum(x < 1e-9 for x in d))}
    delta = beta_curve[str(a.beta)]["mean_L1"] if str(a.beta) in beta_curve else \
        float(np.mean([abs(x) for x in d]))
    # 3. the prior table is not empty (an empty table reweights by exp(0) and changes nothing)
    entries = sum(len(v) for v in prior.table.values())

    print(json.dumps({"hook_overridden": bool(ok_hook),
                      "prior_entries": entries,
                      "buckets": len(prior.table),
                      "positions_probed": a.positions,
                      "mean_L1_change_at_beta": {k: v["mean_L1"] for k, v in beta_curve.items()},
                      "beta_curve": beta_curve,
                      "beta_in_use": a.beta,
                      "meta": prior.meta}, indent=1))
    bad = []
    if not ok_hook:
        bad.append("the subclass does NOT override the method the search calls")
    if entries == 0:
        bad.append("the prior table is EMPTY -- exp(0) reweights nothing")
    if delta < 1e-9:
        bad.append("the reweighted priors are IDENTICAL to the raw ones -- the prior is inert")
    for m in bad:
        print("  FAIL: " + m, file=sys.stderr)
    print(f"\n  SELFTEST {'PASS' if not bad else 'FAIL'}", file=sys.stderr)
    return 0 if not bad else 1


def cmd_sweep(a):
    import numpy as np
    from chessnet.evaluate import load_openings
    from sims_sweep import play_pair
    if cmd_selftest(a) != 0:
        print("  REFUSING to sweep: the prior is not reaching the search. Every arm would return "
              "the baseline's own score and read as a clean negative.", file=sys.stderr)
        return 1
    model, cfg, prior = load_all(a)
    boards = load_openings(a.pgn, 200, seed=a.seed + 2)
    rng = np.random.default_rng(a.seed)
    base = build(model, cfg, a.baseline, a.seed + 1)

    arms = ([("with", s) for s in a.with_sims] + [("without", s) for s in a.without_sims])
    rows, t0 = [], time.time()
    for kind, sims in arms:
        pa = build(model, cfg, sims, a.seed, prior=prior if kind == "with" else None, beta=a.beta)
        # Played in CHUNKS of the same play_pair, not a re-implementation of it: a five-hour run
        # that prints nothing until a pairing ends cannot be told apart from a hung one for ninety
        # minutes. Chunks are EVEN so play_pair's colour alternation (a_white = g % 2 == 0, which
        # restarts per call) still balances seats within every chunk, and the same rng object is
        # threaded through so the opening draws do not repeat.
        got, played = 0.0, 0
        while played < a.games:
            n = min(a.chunk, a.games - played)
            got += play_pair(pa, base, boards, n, rng, a.max_moves) * n
            played += n
            print(f"    {kind} {sims} sims: {played}/{a.games} games, running {got/played:.3f} "
                  f"[{(time.time()-t0)/60:.0f}m]", flush=True)
        s = got / a.games
        row = {"arm": kind, "sims": sims, "baseline_sims": a.baseline, "games": a.games,
               "win_rate": round(s, 4), "elo_delta": round(s2e(s, a.games), 1),
               "se_win_rate": round(0.5 / math.sqrt(a.games), 4)}
        rows.append(row)
        print(f"  {kind:8s} {sims:4d} sims vs baseline-{a.baseline}: {s:.3f} "
              f"({s2e(s, a.games):+.0f} Elo)  [{(time.time()-t0)/60:.0f}m]", flush=True)

    def wr(kind, sims):
        return next((r["win_rate"] for r in rows if r["arm"] == kind and r["sims"] == sims), None)
    out = {"design": "理 10916, pre-registered", "run_dir": a.run_dir, "prior": a.prior,
           "beta": a.beta, "rows": rows,
           "constraint_with_256_ge_45pct": (wr("with", a.baseline) or 0) >= 0.45,
           "objective_with_128_ge_45_and_without_128_lt_45":
               ((wr("with", 128) or 0) >= 0.45 and (wr("without", 128) or 1) < 0.45),
           "four_x_with_64_ge_45pct": (wr("with", 64) or 0) >= 0.45,
           "bounds": ["80 games per pairing: SE ~= 5.6 points of win rate. A result within one SE "
                      "of the 45% threshold is CLOSE, not a pass.",
                      "Self-play against the same evaluator: this measures what the prior does to "
                      "THIS net's search, not whether the moves are good.",
                      "The prior was trained on games this same frozen net played, so the arms are "
                      "not independent of the training signal."]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("selftest", cmd_selftest), ("sweep", cmd_sweep)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--prior", required=True)
        p.add_argument("--beta", type=float, default=1.0)
        p.add_argument("--positions", type=int, default=60,
                       help="positions the selftest reweights to size the intervention")
        if name == "sweep":
            p.add_argument("--pgn", required=True)
            p.add_argument("--baseline", type=int, default=256)
            p.add_argument("--with-sims", nargs="+", type=int, default=[64, 128, 256])
            p.add_argument("--without-sims", nargs="+", type=int, default=[64, 128])
            p.add_argument("--games", type=int, default=80)
            p.add_argument("--chunk", type=int, default=8,
                           help="games per progress print; must be EVEN to keep colours balanced")
            p.add_argument("--max-moves", type=int, default=160)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(a.fn(a))
