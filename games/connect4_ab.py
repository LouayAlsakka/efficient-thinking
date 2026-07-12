#!/usr/bin/env python
"""Fast heuristic depth-limited alpha-beta for Connect-4 — the 'Stockfish' of the games arm.

The exact negamax solver (connect4.solve) is too slow in pure Python to label opening positions. This
gives a fast, strong, depth-bounded evaluator: heuristic leaf scores from open 2-/3-in-a-row 'windows'
+ center control. Used to (a) label training data (best move + value at depth D) and (b) form the
opponent ladder for measuring strength — the analog of Stockfish at fixed depth for chess.
"""
from __future__ import annotations
from connect4 import C4, W, H, ORDER

# ---- precompute the 69 four-in-a-row line masks (bit = row + col*(H+1)) --------------------------
def _bit(col, row): return 1 << (row + col * (H + 1))
LINES = []
for col in range(W):
    for row in range(H):
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):          # →, ↑, ↗, ↘
            cells = [(col + dc * k, row + dr * k) for k in range(4)]
            if all(0 <= c < W and 0 <= r < H for c, r in cells):
                LINES.append(sum(_bit(c, r) for c, r in cells))
LINES = list(set(LINES))                                          # 69 unique windows

_WT = {0: 0, 1: 1, 2: 4, 3: 32, 4: 100000}                        # value of k own pieces in an open window


def heuristic(pos, mask):
    """Score from side-to-move's view: open windows (mine minus opponent's) + center control."""
    opp = pos ^ mask
    s = 0
    for line in LINES:
        p = bin(pos & line).count("1")
        o = bin(opp & line).count("1")
        if o == 0:
            s += _WT[p]
        elif p == 0:
            s -= _WT[o]
    center = ((1 << H) - 1) << (3 * (H + 1))                       # column 3
    s += 3 * bin(pos & center).count("1") - 3 * bin(opp & center).count("1")
    return s


def _ab(s: C4, depth, alpha, beta):
    """Depth-limited negamax; heuristic at the horizon. Score from side-to-move's view."""
    for c in ORDER:                                               # immediate win
        if s.can_play(c) and s.is_winning(c):
            return 100000 + depth
    moves = s.legal()
    if not moves:
        return 0                                                  # board full = draw
    if depth == 0:
        return heuristic(s.position, s.mask)
    best = -10 ** 9
    for c in ORDER:
        if s.can_play(c):
            v = -_ab(s.play(c), depth - 1, -beta, -alpha)
            if v > best:
                best = v
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
    return best


def ab_best(s: C4, depth=8):
    """(best_col, score) from a depth-`depth` search."""
    for c in ORDER:
        if s.can_play(c) and s.is_winning(c):
            return c, 100000
    best_c, best_v = None, -10 ** 9
    for c in ORDER:
        if s.can_play(c):
            v = -_ab(s.play(c), depth - 1, -10 ** 9, 10 ** 9)
            if v > best_v:
                best_v, best_c = v, c
    return best_c, best_v


def value01(s: C4, depth=8):
    """Side-to-move value in [0,1] via a squashed depth-`depth` score (net training target)."""
    import math
    _, v = ab_best(s, depth)
    if v >= 100000:
        return 1.0
    if v <= -100000:
        return 0.0
    return 1.0 / (1.0 + math.exp(-v / 40.0))


if __name__ == "__main__":
    import time
    p = lambda *a: print(*a, flush=True)
    p(f"LINES precomputed: {len(LINES)} (expect 69)")
    # speed: depth-8 from a 6-ply opening position (the one that hung the perfect solver)
    s = C4()
    for c in [3, 3, 3, 2, 4, 2]:
        s = s.play(c)
    for d in (6, 8, 10):
        t = time.time(); bc, bv = ab_best(s, d); dt = time.time() - t
        p(f"  depth-{d}: best_col={bc} score={bv} value01={value01(s, d):.2f}  {dt*1000:.0f}ms")
