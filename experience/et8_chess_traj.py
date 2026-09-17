#!/usr/bin/env python3
"""ET-8 P8: generate the chess trajectories the experience prior needs. They do not exist.

    python3 experience/et8_chess_traj.py --run-dir runs/et6_chess_traj --games 200 --sims 100 \
        --out experience/traj/chess_v0.jsonl

WHY THIS IS A NEW FILE AND NOT AN ADAPTER. 理 asked me to confirm et8_chess_prior's trajectory
adapter first. It cannot be confirmed, because there is nothing for it to read:

  runs/et6_chess_traj/  holds selfplay_log.json (60 rows of iter/new/buffer/loss/results/sec) and
                        model weights. That is a RUN DIRECTORY, not a trajectory store.
  scripts/selfplay.py   keeps self-play samples in an in-memory buffer and NEVER persists it --
                        only model.npz, the stage checkpoints and the iteration log.
  play_game()           returned ENCODED TENSORS (codes, meta, idxs, probs, value) and dropped the
                        chess.Board when it returned. So even a persisted buffer would not carry a
                        FEN or a UCI move, and et8_chess_prior needs both: it wants
                        {"moves": [uci], "result": +-1} and calls state_bucket(board) /
                        move_type(board, mv) on a live board.

The games of those 60 iterations are gone and are not recoverable from what was kept. Same shape as
tasks/v0: the artifact a later question needed was never the artifact the run was written to save.

WHAT THIS GENERATES INSTEAD, and why it is the better source anyway. Games played by the FROZEN
trained evaluator, which is what the prior is supposed to re-weight. The discarded buffer was the
union of every intermediate checkpoint's play, from a randomly initialised net onward -- a prior
learned from that would be fitted partly to the mistakes of a model that no longer exists.

It calls scripts/selfplay.play_game directly, so the trajectories come from the SAME game loop that
produced the training data, not from a re-implementation that could differ in temperature schedule,
draw claiming or move cap. play_game grew an optional `game_log` argument for this; training does
not read it and the buffer path is untouched.

BOUNDS THIS FILE CANNOT REMOVE. These are self-play games at a FIXED simulation count against the
model itself -- the prior learned from them is a prior over what THIS evaluator does when it wins,
not over what is good. Stockfish agreement (the other signal et8_chess_prior supports) is not used
here and would need an engine pool.
"""
from __future__ import annotations
import argparse, json, os, sys, time

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="directory holding model.npz")
    ap.add_argument("--weights", default="model.npz")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--max-moves", type=int, default=160)
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    from selfplay import play_game
    from chessnet.train import load_run           # the run dir's OWN config rebuilds the net
    from chessnet.search import MCTSPlayer

    if not os.path.exists(os.path.join(a.run_dir, "model.npz")):
        print(f"  no model.npz in {a.run_dir}", file=sys.stderr); return 1
    model, cfg = load_run(a.run_dir)              # never construct the net from defaults: the
    # architecture lives in config.json beside the weights, and a default-constructed net that
    # happens to load is the same class of error as running 3B steering vectors against a 7B model.
    player = MCTSPlayer(model, sims=a.sims, dirichlet_alpha=0.3, seed=a.seed)
    rng = np.random.default_rng(a.seed)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    log, t0 = [], time.time()
    with open(a.out, "w") as f:
        for i in range(a.games):
            before = len(log)
            play_game(player, a.max_moves, a.temp_moves, rng, game_log=log)
            for g in log[before:]:
                f.write(json.dumps(g) + "\n")
            f.flush()          # a run that dies at game 180 should still leave 179 usable games
            if (i + 1) % 10 == 0:
                r = (time.time() - t0) / (i + 1)
                print(f"  {i+1}/{a.games}  {r:.1f}s/game  eta {r*(a.games-i-1)/60:.0f}m",
                      file=sys.stderr)
    dec = sum(1 for g in log if g["result"] != 0)
    print(f"  wrote {len(log)} games to {a.out}  ({dec} decisive, {len(log)-dec} drawn)",
          file=sys.stderr)
    print(f"  NOTE: et8_chess_prior.train SKIPS draws, so the usable signal is {dec} games.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
