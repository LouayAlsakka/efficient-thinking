"""Closed-loop search on a learned policy+value net (see docs/self-search-design.md).

Depth-d beam-pruned negamax: the policy (NM) proposes the top-X moves, we play each
to the resulting board, evaluate with the value head Eval(N), and back up via
score(m) = 1 - V_{d-1}(child). depth=1 is the 1-ply lookahead — the first closed-loop
test (does looking one move ahead beat the raw policy?).

SearchPlayer exposes the same .choose(board)->MoveChoice interface as ModelPlayer,
so it drops straight into the evaluate.py match/ladder harness.
"""
from __future__ import annotations

import math
import chess
import numpy as np
import mlx.core as mx

from .encoding import ENCODERS, move_to_index
from .player import MoveChoice


def _terminal_score(board: chess.Board):
    """Exact expected score for the side to move at a game-over node, else None."""
    if not board.is_game_over():
        return None
    if board.is_checkmate():
        return 0.0            # side to move is mated -> loss
    return 0.5               # stalemate / draw / insufficient material


class SearchPlayer:
    def __init__(self, model, encoding: str = "onehot", beam: int = 8,
                 depth: int = 1, qdepth: int = 0, cache: bool = True, seed: int = 0):
        self.model = model
        self.encode = ENCODERS[encoding]
        self.beam = beam
        self.depth = depth
        self.qdepth = qdepth      # >0: quiescence extension (captures/promotions) at leaves
        # leaf-eval memo keyed by position: lets a later, deeper cascade stage reuse the
        # value-net calls an earlier stage already paid for (overlapping subtrees).
        self._cache = {} if cache else None

    # --- net queries -------------------------------------------------------
    def _logits(self, board: chess.Board) -> np.ndarray:
        x = mx.array(self.encode(board)[None, :])
        return np.array(self.model(x)[0])   # model(x) -> logits [1,4096]

    def _eval_boards(self, boards) -> np.ndarray:
        """Eval(N) in [0,1] for each board (expected score for its side to move)."""
        xs = mx.array(np.stack([self.encode(b) for b in boards]))
        out = self.model(xs, return_value=True)
        return np.array(out[1])

    def _topk_moves(self, board: chess.Board, logits: np.ndarray, k: int):
        mirrored = board.turn == chess.BLACK
        legal = list(board.legal_moves)
        legal.sort(key=lambda mv: -logits[move_to_index(mv, mirrored)])
        return legal[:k]

    # --- recursion ---------------------------------------------------------
    def _value(self, board: chess.Board, depth: int) -> float:
        term = _terminal_score(board)
        if term is not None:
            return term
        if depth == 0:
            return float(self._eval_boards([board])[0])
        best = -1.0
        for mv in self._topk_moves(board, self._logits(board), self.beam):
            board.push(mv)
            s = 1.0 - self._value(board, depth - 1)
            board.pop()
            if s > best:
                best = s
        return best

    def _eval1(self, board) -> float:
        if self._cache is not None:
            key = board._transposition_key()
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        v = float(self._eval_boards([board])[0])
        if self._cache is not None:
            if len(self._cache) > 300000:
                self._cache.clear()
            self._cache[key] = v
        return v

    def _quiescence(self, board, alpha: float, beta: float, qd: int) -> float:
        """Extend through captures/promotions until quiet, so the leaf isn't judged
        mid-tactic (horizon fix). Stand-pat = static eval is a floor (the mover can
        decline to capture). Negamax [0,1] with alpha-beta."""
        term = _terminal_score(board)
        if term is not None:
            return term
        stand = self._eval1(board)
        if stand >= beta or qd == 0:
            return stand
        if stand > alpha:
            alpha = stand
        for mv in board.legal_moves:
            if not (board.is_capture(mv) or mv.promotion):
                continue
            board.push(mv)
            v = 1.0 - self._quiescence(board, 1.0 - beta, 1.0 - alpha, qd - 1)
            board.pop()
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        return alpha

    def _value_ab(self, board: chess.Board, depth: int,
                  alpha: float, beta: float) -> float:
        """Negamax alpha-beta with the [0,1]/complement convention. The policy
        orders moves (best first), so cutoffs fire early -> we search deeper for
        the same cost. Child window flips: [alpha,beta] -> [1-beta, 1-alpha]."""
        term = _terminal_score(board)
        if term is not None:
            return term
        if depth == 0:
            if self.qdepth > 0:
                return self._quiescence(board, alpha, beta, self.qdepth)
            return self._eval1(board)
        best = -1.0
        for mv in self._topk_moves(board, self._logits(board), self.beam):
            board.push(mv)
            s = 1.0 - self._value_ab(board, depth - 1, 1.0 - beta, 1.0 - alpha)
            board.pop()
            if s > best:
                best = s
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break            # beta cutoff — this branch can't improve the result
        return best

    # --- move choice -------------------------------------------------------
    def choose(self, board: chess.Board) -> MoveChoice:
        logits = self._logits(board)
        cands = self._topk_moves(board, logits, self.beam)
        mirrored = board.turn == chess.BLACK
        if not cands:
            return MoveChoice(None, 0, was_illegal=False)

        if self.depth == 1:
            # batched 1-ply: score each candidate by 1 - Eval(child), with exact
            # terminal scores where the move ends the game.
            scores = np.empty(len(cands), dtype=np.float32)
            need_eval, idxs = [], []
            for i, mv in enumerate(cands):
                board.push(mv)
                term = _terminal_score(board)
                if term is not None:
                    scores[i] = 1.0 - term
                else:
                    need_eval.append(board.copy()); idxs.append(i)
                board.pop()
            if need_eval:
                vs = self._eval_boards(need_eval)
                for j, i in enumerate(idxs):
                    scores[i] = 1.0 - float(vs[j])
            best = cands[int(np.argmax(scores))]
            return MoveChoice(best, move_to_index(best, mirrored), was_illegal=False)

        # general depth-d: alpha-beta negamax at the root (policy-ordered moves)
        best_mv, best_score = cands[0], -1.0
        alpha = 0.0
        for mv in cands:
            board.push(mv)
            s = 1.0 - self._value_ab(board, self.depth - 1, 0.0, 1.0 - alpha)
            board.pop()
            if s > best_score:
                best_score, best_mv = s, mv
            if best_score > alpha:
                alpha = best_score
        return MoveChoice(best_mv, move_to_index(best_mv, mirrored), was_illegal=False)


# ============================================================================
# MCTS / PUCT (AlphaZero-style) — the adaptive alternative to fixed-depth beam.
# Values are in the [0,1] expected-score convention; backup flips (1-v) each ply.
# ============================================================================

class _Node:
    __slots__ = ("N", "W", "P", "children", "expanded")

    def __init__(self, prior: float):
        self.N = 0          # visit count
        self.W = 0.0        # total value, from THIS node's mover perspective
        self.P = prior      # policy prior for the move that led here
        self.children = {}  # move -> _Node
        self.expanded = False

    def q(self) -> float:
        return self.W / self.N if self.N else 0.0


class MCTSPlayer:
    """PUCT MCTS: select by Q + c_puct*P*sqrt(Nparent)/(1+Nchild), evaluate leaves
    with the value head, expand with policy priors, back up with sign flips. After
    `sims` simulations, play the most-visited root move. One value-net forward pass
    per simulation, so `sims` ~ forward-passes/move (fair vs beam-minimax nodes)."""

    def __init__(self, model, encoding: str = "onehot", sims: int = 200,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.0,
                 dirichlet_eps: float = 0.25, cache: bool = True, seed: int = 0):
        self.model = model
        self.encode = ENCODERS[encoding]
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha   # >0: add root Dirichlet noise (self-play exploration)
        self.dir_eps = dirichlet_eps
        self._rng = np.random.default_rng(seed)
        # transposition/eval cache: memoize (priors,value) by position so repeated
        # positions (transpositions, opening lines) don't re-run the net -> more sims/sec.
        self._cache = {} if cache else None

    def _policy_value(self, board):
        """Return (priors over legal moves, value) for a non-terminal board."""
        key = board._transposition_key() if self._cache is not None else None
        if key is not None and key in self._cache:
            return self._cache[key]
        x = mx.array(self.encode(board)[None, :])
        logits, v = self.model(x, return_value=True)
        logits = np.array(logits[0])
        mirrored = board.turn == chess.BLACK
        legal = list(board.legal_moves)
        z = np.array([logits[move_to_index(mv, mirrored)] for mv in legal])
        z = z - z.max()
        p = np.exp(z); p = p / p.sum()
        out = (legal, p, float(np.array(v)[0]))
        if key is not None:
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = out
        return out

    def _simulate(self, board, node) -> float:
        """Run one simulation from `node`; return value from node's mover view."""
        term = _terminal_score(board)
        if term is not None:
            node.N += 1
            node.W += term
            return term
        if not node.expanded:                      # leaf: evaluate + expand
            legal, priors, v = self._policy_value(board)
            for mv, p in zip(legal, priors):
                node.children[mv] = _Node(float(p))
            node.expanded = True
            node.N += 1
            node.W += v
            return v
        # select child by PUCT
        sqrtN = math.sqrt(node.N)
        best_u, best_mv, best_child = -1e30, None, None
        for mv, ch in node.children.items():
            q = (1.0 - ch.q()) if ch.N else 0.5    # value for us = 1 - child's(opp) value
            u = q + self.c_puct * ch.P * sqrtN / (1 + ch.N)
            if u > best_u:
                best_u, best_mv, best_child = u, mv, ch
        board.push(best_mv)
        v_child = self._simulate(board, best_child)
        board.pop()
        v = 1.0 - v_child                          # flip to node's perspective
        node.N += 1
        node.W += v
        return v

    def _run(self, board, root_moves=None):
        root = _Node(0.0)
        self._simulate(board.copy(), root)          # first sim expands the root
        if root_moves is not None and root.children:  # restrict root to a candidate set
            keep = {mv: ch for mv, ch in root.children.items() if mv in root_moves}
            if keep:
                tot = sum(ch.P for ch in keep.values()) or 1.0
                for ch in keep.values():
                    ch.P /= tot                      # renormalize priors over survivors
                root.children = keep
        if self.dir_alpha > 0 and root.children:     # AlphaZero root exploration noise
            noise = self._rng.dirichlet([self.dir_alpha] * len(root.children))
            for (mv, ch), n in zip(root.children.items(), noise):
                ch.P = (1 - self.dir_eps) * ch.P + self.dir_eps * float(n)
        for _ in range(self.sims - 1):
            self._simulate(board.copy(), root)
        return root

    def choose(self, board: chess.Board) -> MoveChoice:
        if board.is_game_over() or not any(board.legal_moves):
            return MoveChoice(None, 0, was_illegal=False)
        root = self._run(board)
        best_mv = max(root.children, key=lambda mv: root.children[mv].N)
        return MoveChoice(best_mv, move_to_index(best_mv, board.turn == chess.BLACK),
                          was_illegal=False)

    def search_visits(self, board):
        """For self-play (Stage 3): run MCTS, return (moves, move_indices, visit_probs)
        — the improved policy target π ∝ visit counts."""
        root = self._run(board)
        mirrored = board.turn == chess.BLACK
        moves = list(root.children.keys())
        visits = np.array([root.children[mv].N for mv in moves], dtype=np.float32)
        probs = visits / max(visits.sum(), 1.0)
        idxs = np.array([move_to_index(mv, mirrored) for mv in moves], dtype=np.int32)
        return moves, idxs, probs


class BatchedMCTSPlayer(MCTSPlayer):
    """MCTS with parallel leaf collection (virtual loss) + BATCHED leaf evaluation — the fix for the
    batch-1 throughput bottleneck. Each wave descends up to `batch` paths from the root, applying a
    virtual loss so they diverge, then evaluates all distinct collected leaves in ONE forward pass
    (fills the GPU) instead of one-at-a-time. Same search in expectation as MCTSPlayer; the point is
    throughput. Virtual loss adds (N+=vl, W+=vl) to each node on the descended path — raising its
    value from its own mover's view, so the parent's score (1 - child.q()) drops and later paths in
    the wave avoid it — and is removed at back-up."""

    def __init__(self, model, batch: int = 32, virtual_loss: int = 1, **kw):
        super().__init__(model, **kw)
        self.batch = batch
        self.vl = virtual_loss

    def _policy_value_batch(self, boards):
        """One forward pass over many boards → list of (legal_moves, priors, value)."""
        xs = mx.array(np.stack([self.encode(b) for b in boards]))
        logits, v = self.model(xs, return_value=True)
        logits = np.array(logits); v = np.array(v).reshape(-1)
        out = []
        for i, b in enumerate(boards):
            mirrored = b.turn == chess.BLACK
            legal = list(b.legal_moves)
            z = np.array([logits[i][move_to_index(mv, mirrored)] for mv in legal])
            z = z - z.max(); p = np.exp(z); p = p / p.sum()
            out.append((legal, p, float(v[i])))
        return out

    def _descend(self, board, root):
        """Select root→leaf, applying virtual loss to each node on the path."""
        node = root
        node.N += self.vl; node.W += self.vl
        path = [node]
        while node.expanded:
            sqrtN = math.sqrt(node.N)
            best_u, best_mv, best_child = -1e30, None, None
            for mv, ch in node.children.items():
                q = (1.0 - ch.q()) if ch.N else 0.5
                u = q + self.c_puct * ch.P * sqrtN / (1 + ch.N)
                if u > best_u:
                    best_u, best_mv, best_child = u, mv, ch
            board.push(best_mv)
            node = best_child
            node.N += self.vl; node.W += self.vl
            path.append(node)
        return path, board, _terminal_score(board)

    def _backup(self, path, v):
        for node in reversed(path):
            node.N -= self.vl; node.W -= self.vl     # remove virtual loss
            node.N += 1; node.W += v                 # real back-up (alternate perspective)
            v = 1.0 - v

    def _run(self, board, root_moves=None):
        root = _Node(0.0)
        self._simulate(board.copy(), root)           # first sim expands the root (single eval)
        if root_moves is not None and root.children:
            keep = {mv: ch for mv, ch in root.children.items() if mv in root_moves}
            if keep:
                tot = sum(ch.P for ch in keep.values()) or 1.0
                for ch in keep.values():
                    ch.P /= tot
                root.children = keep
        if self.dir_alpha > 0 and root.children:
            noise = self._rng.dirichlet([self.dir_alpha] * len(root.children))
            for (mv, ch), n in zip(root.children.items(), noise):
                ch.P = (1 - self.dir_eps) * ch.P + self.dir_eps * float(n)
        remaining = self.sims - 1
        while remaining > 0:
            wave = min(self.batch, remaining)
            order = []; bucket = {}                   # leaf_node -> [board, term, [paths]]
            for _ in range(wave):
                path, b, term = self._descend(board.copy(), root)
                leaf = path[-1]
                if leaf in bucket:
                    bucket[leaf][2].append(path)
                else:
                    bucket[leaf] = [b, term, [path]]; order.append(leaf)
            to_eval = [l for l in order if bucket[l][1] is None and not l.expanded]
            res = dict(zip(to_eval, self._policy_value_batch([bucket[l][0] for l in to_eval]))) if to_eval else {}
            for leaf in order:
                b, term, paths = bucket[leaf]
                if term is not None:
                    v = term
                elif leaf in res:
                    legal, priors, v = res[leaf]
                    for mv, p in zip(legal, priors):
                        leaf.children[mv] = _Node(float(p))
                    leaf.expanded = True
                else:
                    v = leaf.q() if leaf.N else 0.5   # already expanded (rare same-wave collision)
                for path in paths:
                    self._backup(path, v)
            remaining -= wave
        return root


# ============================================================================
# Cascade search — a SEQUENTIAL funnel over search SHAPES (wide→square→narrow).
# Stage 1 looks at many candidate moves but shallow (cheap, prunes junk); each
# later stage TRIMS to the survivors and searches them DEEPER. Expensive deep
# search only ever runs on the handful of moves that survived the wide pass, so
# the compute budget is spent where it matters. Stage count is a free knob
# (3 / 5 / 9 ...): more stages = a gentler taper from wide-shallow to narrow-deep.
# ============================================================================

class CascadeSearchPlayer:
    """Stages: list of (beam, depth, qdepth). beam should DECREASE and depth
    INCREASE down the list. Each stage re-ranks its input candidates by
    1 - value_ab(child, depth-1) and passes the top `beam` to the next stage.
    The leaf-eval cache (SearchPlayer._cache) carries value-net calls forward,
    so a survivor already scored shallowly isn't re-evaluated from scratch."""

    def __init__(self, model, encoding: str = "onehot", stages=None,
                 cache: bool = True, seed: int = 0):
        # default 3-stage funnel: wide&short -> square -> narrow&deep
        self.stages = stages or [(20, 1, 0), (8, 3, 2), (3, 6, 4)]
        self._s = SearchPlayer(model, encoding=encoding, cache=cache, seed=seed)

    def _rank(self, board, cands, depth, qdepth):
        """Score each candidate by the value of the resulting position, searched
        to `depth` (with quiescence `qdepth`), and return them best-first."""
        self._s.qdepth = qdepth
        scores = np.empty(len(cands), dtype=np.float32)
        for i, mv in enumerate(cands):
            board.push(mv)
            term = _terminal_score(board)
            if term is not None:
                scores[i] = 1.0 - term
            elif depth <= 1:
                scores[i] = 1.0 - self._s._eval1(board)
            else:
                # full [0,1] window across candidates so every survivor gets a
                # comparable score for trimming (alpha-beta still prunes within).
                scores[i] = 1.0 - self._s._value_ab(board, depth - 1, 0.0, 1.0)
            board.pop()
        order = np.argsort(scores)[::-1]
        return [cands[i] for i in order], scores[order]

    def choose(self, board: chess.Board) -> MoveChoice:
        mirrored = board.turn == chess.BLACK
        logits = self._s._logits(board)
        # seed with the widest stage's beam of policy-ordered legal moves
        cands = self._s._topk_moves(board, logits, self.stages[0][0])
        if not cands:
            return MoveChoice(None, 0, was_illegal=False)
        for beam, depth, qdepth in self.stages:
            cands = cands[:beam]                       # trim to this stage's width
            cands, _ = self._rank(board, cands, depth, qdepth)
            if len(cands) == 1:
                break
        best = cands[0]
        return MoveChoice(best, move_to_index(best, mirrored), was_illegal=False)


# ============================================================================
# All-MCTS cascade — the SAME wide->square->narrow funnel, but every stage is an
# MCTS search (not beam-minimax). Stage 1 is WIDE: all legal moves, exploratory
# c_puct, few sims -> rank broadly by visits. Each later stage RESTRICTS the root
# to the previous stage's top-k and spends MORE sims with a greedier c_puct, so the
# budget funnels onto the survivors. One shared eval cache carries the value-net
# calls across stages. Trimming is by VISIT COUNT (MCTS's own "best move" signal).
# ============================================================================

class MultiStageMCTSPlayer:
    """Stages: list of (keep, sims, c_puct). keep = #moves passed to the next stage;
    sims should INCREASE and c_puct DECREASE down the list. Total sims across stages is
    the budget to compare against a flat MCTS of the same total."""

    def __init__(self, model, encoding: str = "onehot", stages=None, seed: int = 0):
        # default funnel summing to 800 sims (fair vs MCTS sims=800)
        self.stages = stages or [(8, 150, 3.0), (3, 250, 1.5), (1, 400, 0.5)]
        self.m = MCTSPlayer(model, encoding=encoding, cache=True, seed=seed)

    def choose(self, board: chess.Board) -> MoveChoice:
        if board.is_game_over() or not any(board.legal_moves):
            return MoveChoice(None, 0, was_illegal=False)
        allowed, best = None, None
        for i, (keep, sims, cpuct) in enumerate(self.stages):
            self.m.sims, self.m.c_puct = sims, cpuct
            root = self.m._run(board, root_moves=allowed)
            ranked = sorted(root.children, key=lambda mv: root.children[mv].N, reverse=True)
            if not ranked:
                break
            best = ranked[0]
            if keep <= 1 or i == len(self.stages) - 1:
                break
            allowed = set(ranked[:keep])                 # trim to top-k by visits
        if best is None:
            best = next(iter(board.legal_moves))
        return MoveChoice(best, move_to_index(best, board.turn == chess.BLACK),
                          was_illegal=False)


# ============================================================================
# Ensemble-evaluator MCTS — the committee applied where the bottleneck is: the
# LEAF EVALUATION inside the tree. At each expanded leaf, K diverse models each
# produce (policy priors, value); we average them (optionally ELO-weighted) into
# a single de-biased evaluation. Blends CONTINUOUS values (robust to one loud/
# overconfident member) rather than voting on discrete moves. Cost = K forward
# passes per leaf, so compare fairly: ensemble at N sims vs one model at K*N sims.
# ============================================================================

class EnsembleMCTSPlayer(MCTSPlayer):
    def __init__(self, models, encoding: str = "onehot", weights=None, sims: int = 200,
                 c_puct: float = 1.5, cache: bool = True, seed: int = 0):
        super().__init__(models[0], encoding=encoding, sims=sims, c_puct=c_puct,
                         cache=cache, seed=seed)
        self.models = list(models)
        w = np.array(weights, dtype=float) if weights is not None else np.ones(len(models))
        self.weights = w / w.sum()

    def _policy_value(self, board):
        key = board._transposition_key() if self._cache is not None else None
        if key is not None and key in self._cache:
            return self._cache[key]
        x = mx.array(self.encode(board)[None, :])
        mirrored = board.turn == chess.BLACK
        legal = list(board.legal_moves)
        idx = [move_to_index(mv, mirrored) for mv in legal]
        avg_p = np.zeros(len(legal)); avg_v = 0.0
        for wgt, m in zip(self.weights, self.models):
            logits, v = m(x, return_value=True)                 # ensemble eval per leaf
            logits = np.array(logits[0])
            z = np.array([logits[i] for i in idx]); z = z - z.max()
            p = np.exp(z); p = p / p.sum()
            avg_p += wgt * p
            avg_v += wgt * float(np.array(v)[0])
        out = (legal, avg_p, avg_v)
        if key is not None:
            if len(self._cache) > 200000:
                self._cache.clear()
            self._cache[key] = out
        return out
