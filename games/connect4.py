#!/usr/bin/env python
"""Connect-4: bitboard engine + exact negamax solver (the 'perfect oracle').

This is the *simpler* end of the complexity-spectrum test for the evaluator×search decomposition.
Connect-4 is solved, so the negamax solver gives the game-theoretic value of any position — a
perfect label source (cleaner than Stockfish for chess). Board is 7 wide x 6 high.

Bitboard layout (Pons): each column uses 7 bits (6 playable + 1 sentinel); column c occupies bits
[7c, 7c+6]. A position is (position, mask): `position` = current player's stones, `mask` = all
stones. Playing a move both drops a stone and switches the side to move.
"""
from __future__ import annotations

W, H = 7, 6                      # width, height
ORDER = [3, 2, 4, 1, 5, 0, 6]    # center-first move ordering (big alpha-beta win)
MIN, MAX = -1000, 1000


def _bottom(col):  return 1 << (col * (H + 1))
def _top(col):     return 1 << (H - 1 + col * (H + 1))
def _colmask(col): return ((1 << H) - 1) << (col * (H + 1))


def _aligned(pos: int) -> bool:
    """True if `pos` (one player's stones) contains four in a row."""
    m = pos & (pos >> (H + 1))                       # horizontal
    if m & (m >> (2 * (H + 1))): return True
    m = pos & (pos >> H)                             # diagonal \
    if m & (m >> (2 * H)): return True
    m = pos & (pos >> (H + 2))                        # diagonal /
    if m & (m >> (2 * (H + 2))): return True
    m = pos & (pos >> 1)                             # vertical
    if m & (m >> 2): return True
    return False


class C4:
    """Immutable Connect-4 state. `play` returns a new state with the side to move flipped."""
    __slots__ = ("position", "mask", "moves")

    def __init__(self, position=0, mask=0, moves=0):
        self.position, self.mask, self.moves = position, mask, moves

    def legal(self):
        return [c for c in range(W) if (self.mask & _top(c)) == 0]

    def can_play(self, col):
        return (self.mask & _top(col)) == 0

    def is_winning(self, col):
        """Does the side to move win by dropping in `col`?"""
        pos2 = self.position | ((self.mask + _bottom(col)) & _colmask(col))
        return _aligned(pos2)

    def play(self, col) -> "C4":
        return C4(self.position ^ self.mask,
                  self.mask | (self.mask + _bottom(col)),
                  self.moves + 1)

    def terminal(self):
        """Return None if not over, else 'win'(prev mover), 'draw'."""
        if self.moves == W * H:
            return "draw"
        # the *previous* mover is (position ^ mask); did they just win?
        if _aligned(self.position ^ self.mask):
            return "win"
        return None

    def key(self):
        return self.position + self.mask                 # Pons: unique per reachable state

    def encode(self):
        """2x6x7 planes: [side-to-move stones, opponent stones]. Returns a flat list of 84 floats."""
        me, opp = self.position, self.position ^ self.mask
        planes = []
        for bits in (me, opp):
            for c in range(W):
                for r in range(H):
                    planes.append(1.0 if (bits >> (r + c * (H + 1))) & 1 else 0.0)
        return planes

    def __repr__(self):
        me, opp = self.position, self.position ^ self.mask
        rows = []
        for r in range(H - 1, -1, -1):
            row = ""
            for c in range(W):
                b = 1 << (r + c * (H + 1))
                row += "X" if (me & b) else ("O" if (opp & b) else ".")
            rows.append(row)
        return "\n".join(rows) + f"\n(moves={self.moves})"


# ---- efficient solver (Pons non-losing-move pruning + bounds TT) --------------------------------
BOTTOM_FULL = sum(_bottom(c) for c in range(W))
BOARD_MASK = BOTTOM_FULL * ((1 << H) - 1)


def _possible(mask):
    return (mask + BOTTOM_FULL) & BOARD_MASK


def _winning_squares(pos, mask):
    """Empty squares where player `pos` would complete a 4 (their threats)."""
    # vertical
    r = (pos << 1) & (pos << 2) & (pos << 3)
    for sh in (H + 1, H, H + 2):                        # horizontal, diag \, diag /
        p = (pos << sh) & (pos << 2 * sh)
        r |= p & (pos << 3 * sh)
        r |= p & (pos >> sh)
        p = (pos >> sh) & (pos >> 2 * sh)
        r |= p & (pos >> 3 * sh)
        r |= p & (pos << sh)
    return r & (BOARD_MASK ^ mask)


def _non_losing(pos, mask):
    """Bitmask of landing squares that don't hand the opponent an immediate win (0 => all lose)."""
    possible = _possible(mask)
    opp = _winning_squares(pos ^ mask, mask)
    forced = possible & opp
    if forced:
        if forced & (forced - 1):                       # two immediate threats -> lost
            return 0
        possible = forced                               # must block the single threat
    return possible & ~(opp >> 1)                        # never play directly under an opp win


def can_win_next(s: C4):
    return _winning_squares(s.position, s.mask) & _possible(s.mask)


def _negamax(pos, mask, moves, alpha, beta, tt):
    """Score from side-to-move's view. Invariant: side to move cannot win on this move."""
    nl = _non_losing(pos, mask)
    if nl == 0:
        return -(W * H - moves) // 2                     # opponent wins next -> we lose ASAP
    if moves >= W * H - 2:
        return 0                                         # board fills with no win -> draw
    lo = -(W * H - 2 - moves) // 2
    if alpha < lo:
        alpha = lo
        if alpha >= beta:
            return alpha
    hi = (W * H - 1 - moves) // 2
    if beta > hi:
        beta = hi
        if alpha >= beta:
            return beta
    key = pos + mask
    t = tt.get(key)
    if t is not None:
        tlo, thi = t
        if tlo >= beta: return tlo
        if thi <= alpha: return thi
        alpha, beta = max(alpha, tlo), min(beta, thi)
    a0, b0 = alpha, beta
    best = MIN
    for c in ORDER:
        move = (mask + _bottom(c)) & _colmask(c)
        if move & nl:                                    # only non-losing moves
            score = -_negamax(pos ^ mask, mask | move, moves + 1, -beta, -alpha, tt)
            if score > best:
                best = score
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
    plo, phi = tt.get(key, (MIN, MAX))
    if best <= a0:
        phi = min(phi, best)
    elif best >= b0:
        plo = max(plo, best)
    else:
        plo = phi = best
    tt[key] = (plo, phi)
    return best


def solve(s: C4, tt=None) -> int:
    """Exact score from side-to-move: +v win, 0 draw, -v loss (bigger |v| = sooner)."""
    if tt is None:
        tt = {}
    if can_win_next(s):
        return (W * H + 1 - s.moves) // 2
    return _negamax(s.position, s.mask, s.moves, -(W * H) // 2, (W * H) // 2, tt)


def best_move(s: C4, tt=None):
    """Return (best_col, score) under perfect play."""
    if tt is None:
        tt = {}
    for c in ORDER:                                      # immediate win
        if s.can_play(c) and s.is_winning(c):
            return c, (W * H + 1 - s.moves) // 2
    best_c, best_v = None, -10 ** 9
    for c in ORDER:
        if s.can_play(c):
            child = s.play(c)
            v = -( (W * H + 1 - child.moves) // 2 ) if can_win_next(child) else -solve(child, tt)
            if v > best_v:
                best_v, best_c = v, c
    return best_c, best_v


def value01(s: C4, tt=None):
    """Side-to-move expected score in [0,1]: win=1, draw=0.5, loss=0 (perfect-play)."""
    _, v = best_move(s, tt)
    return 1.0 if v > 0 else (0.0 if v < 0 else 0.5)


if __name__ == "__main__":
    import time
    p = lambda *a: print(*a, flush=True)

    # 1) win detection: X plays a vertical 4 in column 3
    s = C4()
    for c in [3, 0, 3, 1, 3, 2, 3]:
        win_now = s.can_play(c) and s.is_winning(c)
        s = s.play(c)
    p("[1] vertical-4 win detected:", win_now and s.terminal() == "win")

    # 2) solver picks the immediate win (fast): X can complete col 3
    s = C4()
    for c in [3, 0, 3, 1, 3, 2]:      # X:3,3,3  O:0,1,2 ; X to move, dropping 3 wins
        s = s.play(c)
    t = time.time(); bc, bv = best_move(s)
    p(f"[2] immediate-win: best_col={bc} (expect 3), score={bv} (>0)  {time.time()-t:.3f}s")

    # 3) horizontal + block sanity: O must block X's open three
    s = C4()
    for c in [0, 6, 1, 6, 2]:         # X:0,1,2 (three in a row bottom), O:6,6 ; O to move must block 3
        s = s.play(c)
    t = time.time(); bc, bv = best_move(s)
    p(f"[3] must-block: O best_col={bc} (expect 3), score={bv}  {time.time()-t:.3f}s")
    p(s)
