#!/usr/bin/env python
"""Stage 3 (evolutionary) — neuroevolution / Evolution Strategy for chess. NO gradients,
NO external evaluator, NO labels. The "biology" loop: MUTATE -> PLAY -> SCORE -> SELECT.

Why this can escape the self-play plateau: gradient self-play optimizes a PROXY (the net's own
biased value/policy targets) and re-reinforces its own blind spots. Evolution optimizes the TRUE
objective directly — *did this mutant win games* — with no gradient and no value target. Random
mutation can jump out of the local optimum gradient descent is stuck in; the only fitness signal
is head-to-head results between versions (relative strength, which we HAVE without any evaluator).

PARALLEL: the lambda offspring are evaluated across a worker pool. Each worker loads the current
champion (run_dir/model.npz), regenerates its assigned mutant DETERMINISTICALLY from a seed
(mutation = champion + seed), plays its fitness games vs the champion, and returns the score. The
main process regenerates the winning mutant from the same seed to crown it -> no weights shipped.

  PYTHONPATH=. python scripts/evolve.py --run-dir runs/evolve1 \
      --init runs/conv_value_from_llm1/model.npz --arch-from runs/conv_value_from_llm1 \
      --lam 16 --workers 16 --games 60 --sigma 0.03 --promote 0.55 --gens 300
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
import chess

from chessnet.model import ModelConfig, PolicyNet
from chessnet.encoding import ENCODERS, move_to_index
from chessnet.train import load_run, RunConfig
from chessnet.player import ModelPlayer
from chessnet.evaluate import run_match, estimate_elo, load_openings


def score_to_elo(s, n):
    eps = 0.5 / max(1, n)
    s = min(max(s, eps), 1 - eps)
    return 400.0 * math.log10(s / (1 - s))


def policy_probs(model, board, encode):
    x = mx.array(encode(board)[None, :])
    logits = np.array(model(x)[0])
    mirrored = board.turn == chess.BLACK
    legal = list(board.legal_moves)
    z = np.array([logits[move_to_index(mv, mirrored)] for mv in legal], dtype=np.float64)
    z -= z.max()
    p = np.exp(z); p /= p.sum()
    return legal, p


def play_game_raw(white_m, black_m, encode, board, temp, rng, max_moves):
    seat = {chess.WHITE: white_m, chess.BLACK: black_m}
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        legal, p = policy_probs(seat[board.turn], board, encode)
        if len(legal) == 0:
            break
        if temp > 0:
            q = p ** (1.0 / temp); q /= q.sum()
            mv = legal[rng.choice(len(legal), p=q)]
        else:
            mv = legal[int(np.argmax(p))]
        board.push(mv); ply += 1
    return board.result(claim_draw=True)


def play_match(ma, mb, encode, boards, n, temp, rng, max_moves):
    total = 0.0
    for g in range(n):
        board = boards[rng.integers(len(boards))].copy() if boards else chess.Board()
        a_white = (g % 2 == 0)
        w, b = (ma, mb) if a_white else (mb, ma)
        r = play_game_raw(w, b, encode, board, temp, rng, max_moves)
        total += (1.0 if a_white else 0.0) if r == "1-0" else \
                 (0.0 if a_white else 1.0) if r == "0-1" else 0.5
    return total / max(1, n)


def mutate(params, sigma, rng):
    """Gaussian weight perturbation, scaled per-tensor by its std (robust across layers)."""
    def perturb(w):
        a = np.array(w)
        if a.ndim == 0:
            return w
        noise = rng.normal(0.0, 1.0, a.shape).astype(np.float32) * (sigma * (a.std() + 1e-8))
        return mx.array(a + noise)
    return tree_map(perturb, params)


# ---- parallel fitness worker (spawn-safe) --------------------------------------
def _fit_worker(task):
    run_dir, mut_seed, sigma, n_games, play_seed, temp, max_moves = task
    champ, cfg = load_run(run_dir)                       # mutation BASE (current champion)
    gen0 = PolicyNet(cfg); gen0.load_weights(os.path.join(run_dir, "gen0.npz"))
    mx.eval(gen0.parameters())
    encode = ENCODERS[cfg.encoding]
    child = mutate(champ.parameters(), sigma, np.random.default_rng(mut_seed))
    off = PolicyNet(cfg); off.update(child); mx.eval(off.parameters())
    fens = json.load(open(os.path.join(run_dir, "openings.json")))
    boards = [chess.Board(f) for f in fens]
    s = play_match(off, gen0, encode, boards, n_games, temp,     # vs FROZEN gen0 (fixed anchor)
                   np.random.default_rng(play_seed), max_moves)
    return mut_seed, s


def elo_vs_ladder(model, cfg, ladder, games, openings, movetime, seed):
    raw = ModelPlayer(model, encoding=cfg.encoding, mode="masked", seed=seed)
    specs = [{"kind": "random"}] + [{"kind": "sf_elo", "elo": e} for e in ladder]
    res = [run_match(raw, s, games, movetime=movetime, openings=openings, seed=seed + i)
           for i, s in enumerate(specs)]
    return estimate_elo(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--arch-from", required=True)
    ap.add_argument("--lam", type=int, default=16, help="offspring per generation")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--games", type=int, default=60, help="fitness games: offspring vs champion")
    ap.add_argument("--sigma", type=float, default=0.03, help="mutation scale (× per-tensor std)")
    ap.add_argument("--temp", type=float, default=0.6, help="play temperature (variety)")
    ap.add_argument("--promote", type=float, default=0.55, help="score vs champion to be crowned")
    ap.add_argument("--confirm-games", type=int, default=120,
                    help="re-match the best offspring vs champion with this many games; only "
                         "promote if it wins AGAIN (filters variance -> meaningful promotions)")
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--max-moves", type=int, default=160)
    ap.add_argument("--anneal", type=float, default=1.0, help="σ *= this each no-promotion gen")
    ap.add_argument("--sigma-max", type=float, default=0.12, help="cap on σ when annealing up")
    ap.add_argument("--check-every", type=int, default=5)
    ap.add_argument("--ladder", default="1700,2000,2300")
    ap.add_argument("--check-games", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=0.03)
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--start-file", default=None,
                    help="json list of FENs to start fitness games from (balanced opening/mid/end "
                         "positions); falls back to the opening book if unset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    _, base = load_run(args.arch_from)
    cfg = ModelConfig(encoding="onehot", arch=base.arch, width=base.width, depth=base.depth,
                      value_head=getattr(base, "value_head", False))
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        from dataclasses import asdict
        json.dump(asdict(RunConfig(encoding="onehot", arch=base.arch, width=base.width,
                                   depth=base.depth, value_head=cfg.value_head,
                                   run_dir=args.run_dir)), f, indent=2)
    encode = ENCODERS["onehot"]
    ladder = [int(x) for x in args.ladder.split(",")]
    rng = np.random.default_rng(args.seed)
    openings = load_openings(args.openings_pgn, 300, seed=args.seed + 3)   # for the SF ladder
    if args.start_file and os.path.exists(args.start_file):
        start_fens = json.load(open(args.start_file))
        print(f"[evolve] fitness from {len(start_fens)} balanced positions ({args.start_file})",
              flush=True)
    else:
        start_fens = [b.fen() for b in openings]
    json.dump(start_fens, open(os.path.join(args.run_dir, "openings.json"), "w"))
    boards = [chess.Board(f) for f in start_fens]

    mx.random.seed(args.seed)
    champion = PolicyNet(cfg); champion.load_weights(args.init); mx.eval(champion.parameters())
    champion.save_weights(os.path.join(args.run_dir, "model.npz"))    # workers load this
    ref = PolicyNet(cfg); ref.load_weights(args.init); mx.eval(ref.parameters())
    ref.save_weights(os.path.join(args.run_dir, "gen0.npz"))   # frozen anchor for workers
    scratch = PolicyNet(cfg); mx.eval(scratch.parameters())    # for confirmation matches

    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(args.workers)

    e0, _ = elo_vs_ladder(champion, cfg, ladder, args.check_games, openings, args.movetime, args.seed)
    print(f"[evolve] {cfg.arch} w{cfg.width} d{cfg.depth} | lam={args.lam} workers={args.workers} "
          f"games={args.games} sigma={args.sigma} promote>={args.promote} | gen0 ladder-Elo={e0:.0f}",
          flush=True)

    champ_params = champion.parameters()
    champ_fit = 0.5                       # champion's score vs FROZEN gen0 (0.5 = still the plateau)
    MARGIN = 0.02                         # must beat the champion's gen0-score by this to be crowned
    sigma, log = args.sigma, []
    for gen in range(args.gens):
        t0 = time.time()
        seeds = [int(rng.integers(1, 2**31)) for _ in range(args.lam)]     # MUTATE (by seed)
        tasks = [(args.run_dir, sd, sigma, args.games, sd ^ (gen + 1), args.temp, args.max_moves)
                 for sd in seeds]
        fits = dict(pool.map(_fit_worker, tasks))            # PLAY+SCORE: each offspring vs gen0
        best_seed = max(fits, key=lambda s: fits[s])                       # SELECT best-vs-gen0
        best_s = fits[best_seed]
        rec = {"gen": gen, "best_vs_gen0": round(best_s, 3),
               "champ_vs_gen0": round(champ_fit, 3), "sigma": round(sigma, 4)}
        if best_s >= champ_fit + MARGIN:                     # beats champion at beating the plateau
            child = mutate(champ_params, sigma, np.random.default_rng(best_seed))
            scratch.update(child); mx.eval(scratch.parameters())
            conf = play_match(scratch, ref, encode, boards, args.confirm_games,
                              args.temp, rng, args.max_moves)              # CONFIRM vs gen0
            rec["confirm"] = round(conf, 3)
            if conf >= champ_fit + MARGIN:                                 # real gain -> crown
                champion.update(child); mx.eval(champion.parameters())
                champ_params = champion.parameters()
                champion.save_weights(os.path.join(args.run_dir, "model.npz"))
                champ_fit = conf
                sigma = args.sigma
                rec["promoted"] = True
            else:
                sigma = min(sigma * args.anneal, args.sigma_max)
        else:
            sigma = min(sigma * args.anneal, args.sigma_max)
        rec["gain_elo"] = round(score_to_elo(champ_fit, args.confirm_games))  # Elo over the plateau
        if (gen + 1) % args.check_every == 0:                              # absolute track
            el, _ = elo_vs_ladder(champion, cfg, ladder, args.check_games, openings,
                                  args.movetime, args.seed + gen)
            rec["ladder_elo"] = round(el)
        rec["sec"] = round(time.time() - t0, 1)
        log.append(rec)
        json.dump(log, open(os.path.join(args.run_dir, "evolve_log.json"), "w"))
        tag = ('** PROMOTED' if rec.get('promoted')
               else f"(rejected conf {rec['confirm']})" if 'confirm' in rec else '')
        msg = (f"  gen {gen}: best_vs_gen0 {best_s:.2f}  champ_vs_gen0 {champ_fit:.3f} "
               f"(+{rec['gain_elo']} Elo vs plateau) {tag} sig {sigma:.3f}")
        if "ladder_elo" in rec:
            msg += f"  ladder {rec['ladder_elo']}"
        print(msg + f"  {rec['sec']:.0f}s", flush=True)

    print("[evolve] DONE", flush=True)


if __name__ == "__main__":
    main()
