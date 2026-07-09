#!/usr/bin/env python
"""Stage 3+ — self-referential Elo ladder (gated self-play league, NO Stockfish).

The problem with vanilla self-play (scripts/selfplay.py): the net always plays its
CURRENT self — a moving target. The two sides drift together, games collapse toward
draws/cycles, and the gradient goes stale -> plateau.

Fix (this file): keep a FROZEN champion as a stationary reference ("our own Stockfish").
  loop:
    - learner self-plays + trains (AlphaZero targets: MCTS visits + game outcome)
    - every `--gate-every` iters, play a learner-vs-CHAMPION match
    - if learner beats champion by >= `--promote-elo` (default +200):
        promote — champion <- snapshot(learner); internal Elo += gap; append to pool
The champion never trains, so it's a fixed rung; each promotion is a self-generated
harder rung. Progress is measured against our own past selves — zero external engine.

A POOL of the last K champions is kept: we gate against the latest (the ratchet) but
also log the score vs a random older champion, to catch cycling (beating the newest
while regressing vs an older one).

  PYTHONPATH=. python scripts/selfplay_league.py --run-dir runs/league1 \
      --init runs/conv_value_llm1/model.npz --arch-from runs/conv_value_llm1 \
      --sims 64 --games 128 --workers 16 --gate-every 3 --gate-games 40 \
      --promote-elo 200 --elo0 2000 --iters 400
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
import chess

from chessnet.model import ModelConfig, PolicyNet
from chessnet.encoding import codes_to_onehot_batch
from chessnet.search import MCTSPlayer
from chessnet.train import load_run, RunConfig
from chessnet.evaluate import load_openings
from selfplay import play_game, train_steps, _worker    # reuse the self-play machinery


def score_to_elo(score: float, n: int) -> float:
    """Logistic-inverse: match score (learner's win fraction) -> Elo gap over opponent.
    Clamp perfect/zero scores (unbounded) to the resolution set by game count n."""
    eps = 0.5 / max(1, n)
    s = min(max(score, eps), 1 - eps)
    return 400.0 * math.log10(s / (1 - s))


def play_pair(learner, champion, openings, n_games, sims, max_moves, seed):
    """Net-vs-net match, learner vs champion, alternating colors from opening books.
    Greedy (most-visited) moves. Returns learner's average score in [0,1] over n_games."""
    lp = MCTSPlayer(learner, sims=sims, seed=seed)
    cp = MCTSPlayer(champion, sims=sims, seed=seed + 1)
    rng = np.random.default_rng(seed)
    total = 0.0
    for g in range(n_games):
        board = openings[rng.integers(len(openings))].copy() if openings else chess.Board()
        learner_white = (g % 2 == 0)
        seat = {chess.WHITE: lp if learner_white else cp,
                chess.BLACK: cp if learner_white else lp}
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < max_moves:
            mv = seat[board.turn].choose(board).move
            if mv is None:
                break
            board.push(mv); ply += 1
        res = board.result(claim_draw=True)
        if res == "1-0":
            total += 1.0 if learner_white else 0.0
        elif res == "0-1":
            total += 0.0 if learner_white else 1.0
        else:
            total += 0.5
    return total / max(1, n_games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--init", required=True, help="gen-0 champion weights (.npz)")
    ap.add_argument("--arch-from", required=True, help="run dir to copy model arch/config from")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--games", type=int, default=128, help="self-play games per iter")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--dirichlet", type=float, default=0.3)
    ap.add_argument("--buffer", type=int, default=300000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    # --- league / gating ---
    ap.add_argument("--gate-every", type=int, default=3, help="run a gate match every N iters")
    ap.add_argument("--gate-games", type=int, default=40, help="games in a gate match")
    ap.add_argument("--gate-sims", type=int, default=0, help="sims in gate match (0=--sims)")
    ap.add_argument("--promote-elo", type=float, default=200.0,
                    help="learner must beat champion by >= this Elo to be promoted")
    ap.add_argument("--elo0", type=float, default=2000.0, help="anchor Elo of gen-0 champion")
    ap.add_argument("--pool-size", type=int, default=5, help="keep last K champions")
    ap.add_argument("--openings-pgn", default="data/lichess/2013-01.pgn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    gate_sims = args.gate_sims or args.sims

    # arch from the reference run so learner == champion topology
    _, base_cfg = load_run(args.arch_from)
    cfg = ModelConfig(encoding="onehot", arch=base_cfg.arch, width=base_cfg.width,
                      depth=base_cfg.depth, value_head=True)
    # persist a load_run-compatible config so the parallel self-play _worker can reload
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        from dataclasses import asdict
        json.dump(asdict(RunConfig(encoding="onehot", arch=base_cfg.arch, width=base_cfg.width,
                                    depth=base_cfg.depth, value_head=True,
                                    run_dir=args.run_dir)), f, indent=2)

    mx.random.seed(args.seed)
    learner = PolicyNet(cfg); learner.load_weights(args.init); mx.eval(learner.parameters())
    champion = PolicyNet(cfg); champion.load_weights(args.init); mx.eval(champion.parameters())
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    openings = load_openings(args.openings_pgn, 400, seed=args.seed + 7)

    # champion pool on disk: champ_gen{N}.npz. learner weights live at model.npz so the
    # spawn workers (which reload from run_dir/model.npz) always self-play the LEARNER.
    os.makedirs(os.path.join(args.run_dir, "pool"), exist_ok=True)
    def champ_path(gen): return os.path.join(args.run_dir, "pool", f"champ_gen{gen}.npz")
    gen = 0
    champion.save_weights(champ_path(0))
    learner.save_weights(os.path.join(args.run_dir, "model.npz"))
    pool = [0]                       # generations available as frozen opponents
    elo = args.elo0

    from collections import deque
    buffer = deque(maxlen=args.buffer)
    pool_mp = None
    if args.workers > 1:
        import multiprocessing as mp
        pool_mp = mp.get_context("spawn").Pool(args.workers)

    print(f"[league] {cfg.arch} w{cfg.width} d{cfg.depth} | sims={args.sims} games/iter={args.games} "
          f"| gate every {args.gate_every} ({args.gate_games}g) promote>=+{args.promote_elo:.0f} "
          f"| gen0 Elo={elo:.0f}", flush=True)
    log = []
    for it in range(args.iters):
        t0 = time.time()
        # ---- learner self-play (reuses the parallel worker; loads model.npz = learner) ----
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}; new = 0
        if pool_mp is not None:
            per = max(1, args.games // args.workers)
            tasks = [(args.run_dir, args.sims, args.dirichlet, per, args.max_moves,
                      args.temp_moves, args.seed + it * 1000 + w) for w in range(args.workers)]
            for samples, res in pool_mp.map(_worker, tasks):
                buffer.extend(samples); new += len(samples)
                for k in results: results[k] += res[k]
        else:
            player = MCTSPlayer(learner, sims=args.sims, dirichlet_alpha=args.dirichlet,
                                seed=args.seed + it)
            for _ in range(args.games):
                s, r = play_game(player, args.max_moves, args.temp_moves, rng)
                buffer.extend(s); new += len(s); results[r if r in results else "*"] += 1
        # ---- train learner ----
        loss = (train_steps(learner, opt, buffer, args.steps, args.batch, args.clip, rng)
                if len(buffer) >= args.batch else 0.0)
        learner.save_weights(os.path.join(args.run_dir, "model.npz"))   # workers see new learner

        rec = {"iter": it, "gen": gen, "elo": round(elo, 1), "new": new,
               "buffer": len(buffer), "loss": round(loss, 3), "results": results,
               "sec": round(time.time() - t0, 1)}

        # ---- gate: learner vs frozen champion ----
        if (it + 1) % args.gate_every == 0:
            champion.load_weights(champ_path(gen))
            s_cur = play_pair(learner, champion, openings, args.gate_games, gate_sims,
                              args.max_moves, args.seed + 50000 + it)
            gap = score_to_elo(s_cur, args.gate_games)
            rec["gate_score"] = round(s_cur, 3); rec["gate_gap"] = round(gap, 1)
            # anti-cycling probe vs an older champion (if any)
            if len(pool) > 1:
                old = int(rng.choice(pool[:-1]))
                champion.load_weights(champ_path(old))
                s_old = play_pair(learner, champion, openings, max(10, args.gate_games // 2),
                                  gate_sims, args.max_moves, args.seed + 60000 + it)
                rec["vs_old_gen"] = old; rec["vs_old_score"] = round(s_old, 3)
            if gap >= args.promote_elo:                       # PROMOTE
                gen += 1
                learner.save_weights(champ_path(gen))
                elo += gap
                pool.append(gen); pool[:] = pool[-args.pool_size:]
                rec["promoted_to_gen"] = gen; rec["elo"] = round(elo, 1)
                print(f"  ** PROMOTE -> gen{gen}  (+{gap:.0f} Elo, score {s_cur:.2f})  "
                      f"internal Elo now {elo:.0f}", flush=True)
            champion.load_weights(champ_path(gen))            # restore current champion

        log.append(rec)
        with open(os.path.join(args.run_dir, "league_log.json"), "w") as f:
            json.dump(log, f)
        g = rec.get("gate_gap")
        print(f"  iter {it}: gen{gen} Elo~{elo:.0f} +{new} buf={len(buffer)} loss {loss:.3f} "
              f"W/B/D {results['1-0']}/{results['0-1']}/{results['1/2-1/2']}"
              + (f"  gate {rec.get('gate_score')} ({g:+.0f})" if g is not None else "")
              + f"  {rec['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
