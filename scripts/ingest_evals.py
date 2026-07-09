#!/usr/bin/env python
"""Ingest the Lichess eval database (~394M positions, Stockfish multi-PV) into
soft-labeled shards — no self-labeling needed.

Each JSONL line is {"fen": ..., "evals": [{"pvs": [{"cp"|"mate", "line"}...],
"depth": D}, ...]}. cp/mate are from WHITE's POV (verified against the data:
best move for the side to move has the lowest White-POV cp). We take the deepest
eval, convert each PV's first move + score into a mover-POV win-probability, and
store the top-K as the per-move advantage map (the soft target) plus the best
move (the hard target). These evals are often depth 30-50 — deeper/better labels
than our depth-10 self-labeling.

Speed: parsing 394M positions with python-chess would take hours, so FEN->codes
and UCI->move-index are parsed directly from strings (validated identical to
chessnet.encoding on startup). Decompression streams via `zstd -dc` (no 150GB+
decompressed file on disk).

Example:
  ./.venv/bin/python scripts/ingest_evals.py --zst data/lichess/eval.jsonl.zst \
      --out data/eval --topk 16 --shard-size 2000000 --workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from chessnet.labeler import cp_to_winprob
from chessnet.dataset import (save_shard, PHASE_OPENING, PHASE_MIDDLE,
                              PHASE_END, PHASE_NAMES)

# lowercase piece char -> index into [P,N,B,R,Q,K] (matches encoding._PIECE_ORDER)
_PIDX = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def fen_to_codes(board_field: str, turn: str, castling: str, ep: str):
    """FEN fields -> (codes[64] uint8, meta[5] uint8), mover-relative.

    Mirrors board+colors when black to move (side-to-move normalization),
    matching chessnet.encoding.board_to_codes exactly.
    """
    black = turn == "b"
    codes = np.zeros(64, dtype=np.uint8)
    rank, file = 7, 0
    for ch in board_field:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch <= "9":  # digit
            file += ord(ch) - 48
        else:
            sq = rank * 8 + file
            white_piece = ch.isupper()
            idx = _PIDX[ch.lower()]
            if not black:
                codes[sq] = idx + 1 if white_piece else idx + 7
            else:
                codes[sq ^ 56] = idx + 1 if not white_piece else idx + 7
            file += 1
    meta = np.zeros(5, dtype=np.uint8)
    own_K, own_Q, opp_k, opp_q = ("K", "Q", "k", "q") if not black else ("k", "q", "K", "Q")
    meta[0] = own_K in castling
    meta[1] = own_Q in castling
    meta[2] = opp_k in castling
    meta[3] = opp_q in castling
    meta[4] = ep != "-"
    return codes, meta


def uci_to_index(uci: str, black: bool) -> int:
    frm = (int(uci[1]) - 1) * 8 + (ord(uci[0]) - 97)
    to = (int(uci[3]) - 1) * 8 + (ord(uci[2]) - 97)
    if black:
        frm ^= 56
        to ^= 56
    return frm * 64 + to


def _phase_by_material(codes: np.ndarray) -> int:
    # eval-DB FENs carry no move counter, so classify by material only
    n = int((codes > 0).sum())
    if n <= 12:
        return PHASE_END
    if n >= 28:
        return PHASE_OPENING
    return PHASE_MIDDLE


def _pv_winprob(pv: dict, black: bool) -> float | None:
    if "cp" in pv:
        wp_white = cp_to_winprob(pv["cp"])
    elif "mate" in pv:
        wp_white = 1.0 if pv["mate"] > 0 else 0.0
    else:
        return None
    return (1.0 - wp_white) if black else wp_white


def _process_line(line: str, topk: int, min_depth: int):
    try:
        obj = json.loads(line)
        fen = obj["fen"]
        parts = fen.split(" ")
        board_field, turn, castling, ep = parts[0], parts[1], parts[2], parts[3]
        black = turn == "b"
        # deepest eval entry
        best_eval = max(obj["evals"], key=lambda e: e.get("depth", 0))
        if best_eval.get("depth", 0) < min_depth:
            return None
        cand = []
        for pv in best_eval["pvs"]:
            mv = pv.get("line", "").split(" ")[0]
            if len(mv) < 4:
                continue
            wp = _pv_winprob(pv, black)
            if wp is None:
                continue
            cand.append((uci_to_index(mv, black), wp))
        if not cand:
            return None
        cand.sort(key=lambda t: t[1], reverse=True)
        cand = cand[:topk]
        codes, meta = fen_to_codes(board_field, turn, castling, ep)
        sidx = np.full(topk, -1, dtype=np.int16)
        swp = np.full(topk, -1.0, dtype=np.float32)
        for j, (idx, wp) in enumerate(cand):
            sidx[j] = idx
            swp[j] = wp
        return (codes, meta, cand[0][0], cand[0][1],
                _phase_by_material(codes), sidx, swp)
    except Exception:
        return None


def _process_batch(args):
    lines, topk, min_depth = args
    out = [r for r in (_process_line(ln, topk, min_depth) for ln in lines) if r]
    return out


def _selftest():
    """Verify the fast parsers match chessnet.encoding on tricky positions."""
    import chess
    from chessnet.encoding import board_to_codes, move_to_index
    cases = [
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3", "d7d5"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", "e1g1"),
        ("7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -", "f7g7"),
        ("4k3/P7/8/8/8/8/8/4K3 w - -", "a7a8q"),
    ]
    for fen4, uci in cases:
        parts = fen4.split(" ")
        codes, meta = fen_to_codes(parts[0], parts[1], parts[2], parts[3])
        board = chess.Board(fen4 + " 0 1")
        ec, em = board_to_codes(board)
        assert np.array_equal(codes, ec), f"codes mismatch {fen4}"
        assert np.array_equal(meta, em), f"meta mismatch {fen4}"
        black = parts[1] == "b"
        assert uci_to_index(uci, black) == move_to_index(
            chess.Move.from_uci(uci), black), f"index mismatch {fen4} {uci}"
    print("[selftest] fast parsers match chessnet.encoding ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zst", default="data/lichess/eval.jsonl.zst")
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--min-depth", type=int, default=0,
                    help="skip positions whose deepest eval is below this depth")
    ap.add_argument("--shard-size", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0, help="stop after N positions (0=all)")
    args = ap.parse_args()

    _selftest()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    proc = subprocess.Popen(["zstd", "-dc", args.zst], stdout=subprocess.PIPE,
                            bufsize=1 << 20, text=True)

    def line_batches():
        buf = []
        for line in proc.stdout:
            buf.append(line)
            if len(buf) >= args.batch:
                yield (buf, args.topk, args.min_depth)
                buf = []
        if buf:
            yield (buf, args.topk, args.min_depth)

    cc, mm, tt, ww, pp, si, sw = [], [], [], [], [], [], []
    phase_counts = {PHASE_OPENING: 0, PHASE_MIDDLE: 0, PHASE_END: 0}
    shard_idx = total = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, cc, mm, tt, ww, pp, si, sw
        if not tt:
            return
        path = f"{args.out}.{shard_idx:04d}.npz"
        save_shard(path, np.stack(cc), np.stack(mm), np.array(tt), np.array(ww),
                   np.array(pp), soft_idx=np.stack(si), soft_wp=np.stack(sw))
        print(f"  wrote {path} ({len(tt):,} positions)")
        shard_idx += 1
        cc, mm, tt, ww, pp, si, sw = [], [], [], [], [], [], []

    with Pool(processes=args.workers) as pool:
        for results in pool.imap_unordered(_process_batch, line_batches()):
            for codes, meta, tgt, wp, ph, sidx, swp in results:
                cc.append(codes); mm.append(meta); tt.append(tgt); ww.append(wp)
                pp.append(ph); si.append(sidx); sw.append(swp)
                phase_counts[ph] += 1
            total += len(results)
            if total // 500000 != (total - len(results)) // 500000:
                rate = total / (time.time() - t0)
                print(f"ingested {total:,} positions ({rate:.0f}/s)")
            if len(tt) >= args.shard_size:
                flush()
            if args.limit and total >= args.limit:
                break
    flush()
    proc.terminate()
    dt = time.time() - t0
    dist = " ".join(f"{PHASE_NAMES[k]}={v:,}" for k, v in phase_counts.items())
    print(f"DONE: {total:,} positions in {dt:.0f}s ({total/max(dt,1):.0f}/s), "
          f"{shard_idx} shards")
    print(f"phase distribution: {dist}")


if __name__ == "__main__":
    main()
