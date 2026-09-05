#!/usr/bin/env python3
"""
et8_chess_prior.py — Efficient Thinking VIII anchor A0: an EXPERIENCE PRIOR over moves for Paper I's frozen chess evaluator.

The cleanest instance of the ET-8 thesis: the evaluator (3.45M value/policy net) is frozen, the search (PUCT MCTS in
chessnet/search.py) is unchanged, the oracle (Stockfish / game outcome) is external, and the only thing added is a small
learned prior E_phi(a | s) that re-weights which moves the search tries first:

    P'(a|s)  ∝  P_net(a|s) · exp( beta · E_phi(a|s) )          (proposal §3, eq. P ∝ P0·E)

E_phi is deliberately tiny and auditable: a table of log-odds keyed by (coarse STATE bucket, MOVE TYPE), learned from
verified trajectories — moves the search itself chose in games that were WON versus LOST (outcome = external signal), or
moves Stockfish agrees with versus not (oracle signal). Louay's probabilistic-weight rule is built in: counts shrink toward
0 (no weight reaches ±inf), a forgetting factor decays old evidence, updates are surprise-weighted, and every bucket keeps
a floor so a "dead" move type stays reachable.

Metric (P8): simulations-to-a-fixed-Elo-rung with E vs without, same evaluator, same ladder (scripts/eval_search.py).

Usage:
  python experience/et8_chess_prior.py train --traj runs/et6_chess_traj --out experience/chess_prior/v0.json
  python experience/et8_chess_prior.py inspect --prior experience/chess_prior/v0.json
  # then in eval_search.py: MCTSPlayer -> ExperienceMCTSPlayer(model, prior=load_prior(path), beta=1.0, ...)
Status: v0 — feature buckets, trainer skeleton, player subclass. Trajectory format adapter is the first thing Sautee's run
must confirm (see `iter_trajectories`).
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, collections
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---- state buckets and move types: coarse on purpose (a prior, not an evaluator) ------------------------------------------
def material(board: chess.Board) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    m = 0
    for pt, v in vals.items():
        m += v * (len(board.pieces(pt, board.turn)) - len(board.pieces(pt, not board.turn)))
    return m

def state_bucket(board: chess.Board) -> str:
    n = len(board.piece_map())
    phase = "opening" if board.fullmove_number <= 10 else ("endgame" if n <= 12 else "middlegame")
    m = material(board); mat = "up" if m >= 2 else ("down" if m <= -2 else "even")
    check = "chk" if board.is_check() else "nochk"
    castle = "cancastle" if board.has_castling_rights(board.turn) else "nocastle"
    return f"{phase}|{mat}|{check}|{castle}"

def move_type(board: chess.Board, mv: chess.Move) -> str:
    if board.is_castling(mv): return "castle"
    if mv.promotion: return "promote"
    cap = board.is_capture(mv)
    board.push(mv); gives_check = board.is_check(); board.pop()
    piece = board.piece_at(mv.from_square)
    pt = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}[piece.piece_type]
    return f"{pt}{'x' if cap else ''}{'+' if gives_check else ''}"

# ---- the prior: shrunken, decaying, floored log-odds ------------------------------------------------------------------------
class ExperiencePrior:
    K_SHRINK, FLOOR, GAMMA = 5.0, 0.02, 0.9
    def __init__(self, table=None, meta=None):
        self.table = table or {}          # bucket -> move_type -> {"pos": float, "neg": float}
        self.meta = meta or {"rounds": 0, "episodes": 0}
    def logit(self, bucket: str, mtype: str) -> float:
        c = self.table.get(bucket, {}).get(mtype)
        if not c: return 0.0
        pos, neg = c["pos"], c["neg"]; n = pos + neg
        p = (pos + 0.5 * self.K_SHRINK) / (n + self.K_SHRINK)          # shrink toward 0.5
        p = min(1 - self.FLOOR, max(self.FLOOR, p))                    # Cromwell floor: never 0 or 1
        return math.log(p / (1 - p))
    def update(self, bucket: str, mtype: str, good: bool, predicted_p: float | None = None):
        c = self.table.setdefault(bucket, {}).setdefault(mtype, {"pos": 0.0, "neg": 0.0})
        # surprise weighting: a confirmation moves the weight less than a contradiction
        w = 1.0 if predicted_p is None else (1.0 - predicted_p if good else predicted_p) + 0.25
        c["pos" if good else "neg"] += w
    def decay(self):
        for b in self.table.values():
            for c in b.values():
                c["pos"] *= self.GAMMA; c["neg"] *= self.GAMMA
        self.meta["rounds"] += 1
    def reweight(self, board: chess.Board, legal, priors, beta: float):
        """P' ∝ P · exp(beta · E). Returns renormalised priors (numpy array)."""
        import numpy as np
        b = state_bucket(board)
        e = np.array([self.logit(b, move_type(board, mv)) for mv in legal], dtype=np.float64)
        p = np.asarray(priors, dtype=np.float64) * np.exp(beta * e)
        s = p.sum()
        return (p / s) if s > 0 else np.asarray(priors)
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"table": self.table, "meta": self.meta}, open(path, "w"), indent=1)
    @classmethod
    def load(cls, path):
        d = json.load(open(path)); return cls(d["table"], d["meta"])

# ---- the player: PUCT unchanged, priors re-weighted at expansion -----------------------------------------------------------
def make_player_class():
    from chessnet.search import MCTSPlayer
    class ExperienceMCTSPlayer(MCTSPlayer):
        """MCTSPlayer whose expansion priors are P·exp(beta·E). Nothing else changes: same net, same c_puct, same sims."""
        def __init__(self, model, prior: ExperiencePrior, beta: float = 1.0, **kw):
            super().__init__(model, **kw); self.prior, self.beta = prior, beta
        def _policy_value(self, board):
            legal, p, v = super()._policy_value(board)
            if self.beta == 0.0 or not legal: return legal, p, v
            return legal, self.prior.reweight(board, legal, p, self.beta), v
    return ExperienceMCTSPlayer

# ---- training from verified trajectories --------------------------------------------------------------------------------
def iter_trajectories(traj_dir: str):
    """Yield (list_of_(fen, uci_move), result_for_side_to_move_at_each_ply) from a trajectory store.
    ADAPTER: the et6 self-play store's exact schema is confirmed by the first run; supported here:
      - JSONL with {"fens":[...], "moves":[...uci...], "result": 1|0|-1 (white view)}  or
      - JSON list of games with the same keys."""
    files = sorted(glob.glob(os.path.join(traj_dir, "*.json*")))
    for f in files:
        txt = open(f).read().strip()
        games = [json.loads(l) for l in txt.splitlines()] if f.endswith(".jsonl") else json.loads(txt)
        if isinstance(games, dict): games = games.get("games", [games])
        for g in games:
            if "moves" in g and "result" in g:
                yield g

def train(traj_dir: str, out: str, decay_rounds: int = 0):
    prior = ExperiencePrior()
    n_games = n_moves = 0
    for g in iter_trajectories(traj_dir):
        board = chess.Board(g.get("start_fen", chess.STARTING_FEN))
        result = g["result"]                       # +1 white won, -1 black won, 0 draw (white view)
        if result == 0: continue                    # draws carry no outcome signal for a first prior
        for uci in g["moves"]:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves: break
            good = (result > 0) == (board.turn == chess.WHITE)   # the mover went on to win
            b = state_bucket(board); t = move_type(board, mv)
            pl = prior.logit(b, t); pp = 1 / (1 + math.exp(-pl))
            prior.update(b, t, good, predicted_p=pp)
            board.push(mv); n_moves += 1
        n_games += 1
    for _ in range(decay_rounds): prior.decay()
    prior.meta.update({"episodes": n_games, "moves": n_moves, "source": traj_dir})
    prior.save(out)
    return prior, n_games, n_moves

def cmd_train(a):
    prior, ng, nm = train(a.traj, a.out)
    print(json.dumps({"games": ng, "moves": nm, "buckets": len(prior.table),
                      "entries": sum(len(v) for v in prior.table.values()), "out": a.out}, indent=1))

def cmd_inspect(a):
    prior = ExperiencePrior.load(a.prior)
    rows = []
    for b, d in prior.table.items():
        for t, c in d.items():
            n = c["pos"] + c["neg"]
            rows.append((abs(prior.logit(b, t)), b, t, round(prior.logit(b, t), 2), round(n, 1)))
    rows.sort(reverse=True)
    print(f"buckets={len(prior.table)} entries={len(rows)} meta={prior.meta}")
    for _, b, t, lg, n in rows[:25]:
        print(f"  {b:40s} {t:6s} logit={lg:+.2f} n={n}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--traj", required=True); t.add_argument("--out", required=True); t.set_defaults(fn=cmd_train)
    i = sp.add_parser("inspect"); i.add_argument("--prior", required=True); i.set_defaults(fn=cmd_inspect)
    a = ap.parse_args(); a.fn(a)
