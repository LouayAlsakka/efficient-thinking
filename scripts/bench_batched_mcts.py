#!/usr/bin/env python
"""Measure the batch-1 -> batched-leaf MCTS speedup (the §4.4 headroom, actually measured).

Times single-position (batch-1) MCTS vs batched-leaf MCTS (virtual loss, one forward pass per wave)
at several batch sizes, on a fixed set of positions. Reports per-move latency and nodes/sec (nps).
Run on a FREE GPU (batching's benefit is filling idle GPU lanes; CPU won't show it).

  PYTHONPATH=. python scripts/bench_batched_mcts.py --run-dir runs/conv_value_llm1 --sims 800
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
from chessnet.train import load_run
from chessnet.search import MCTSPlayer, BatchedMCTSPlayer
from chessnet.evaluate import load_openings


def bench(player, boards, warmup=1):
    for b in boards[:warmup]:
        player.choose(b.copy())                      # warm JIT/allocator
    t0 = time.time()
    for b in boards:
        player.choose(b.copy())
    dt = time.time() - t0
    return dt / len(boards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/conv_value_llm1")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--batches", default="8,16,32,64,128")
    ap.add_argument("--pgn", default="data/lichess/2013-01.pgn")
    args = ap.parse_args()

    model, cfg = load_run(args.run_dir)
    boards = load_openings(args.pgn, args.positions, seed=1)
    print(f"[bench] {args.run_dir}  sims={args.sims}  positions={len(boards)}", flush=True)

    seq = MCTSPlayer(model, encoding=cfg.encoding, sims=args.sims, seed=0)
    s_lat = bench(seq, boards)
    s_nps = args.sims / s_lat
    print(f"  batch-1 (sequential): {s_lat*1000:7.0f} ms/move   {s_nps:7.0f} nps   (1.0x)", flush=True)

    for B in [int(x) for x in args.batches.split(",")]:
        bat = BatchedMCTSPlayer(model, encoding=cfg.encoding, sims=args.sims, batch=B, seed=0)
        lat = bench(bat, boards)
        print(f"  batched B={B:<4d}      : {lat*1000:7.0f} ms/move   {args.sims/lat:7.0f} nps   "
              f"({s_lat/lat:.1f}x faster)", flush=True)
    print("[bench] DONE", flush=True)


if __name__ == "__main__":
    main()
