#!/usr/bin/env python
"""PUCT MCTS for Connect-4 using the C4 net — the closed-loop search (chess-search analog).

Standard AlphaZero PUCT: prior from the net policy, leaf value from the net value head (or the true
terminal result), backup along the path. `search()` returns the root node (exposes visit counts +
root value for self-play targets); `mcts_move` picks the most-visited legal column; `policy_move` is
the open-loop baseline (argmax legal policy, no search).
"""
from __future__ import annotations
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
from connect4 import C4


def _net_eval(net, state):
    """(policy probs over 7 cols as python list, value in [0,1]) for side to move."""
    x = mx.array([state.encode()])
    logits, value = net(x)
    logits = logits[0]
    legal = state.legal()
    neg = mx.full((7,), -1e9)
    keep = mx.zeros((7,))
    for c in legal:
        keep = keep + mx.eye(7)[c]
    probs = mx.softmax(mx.where(keep > 0, logits, neg))
    return probs.tolist(), float(value[0].item())


class _Node:
    __slots__ = ("state", "P", "children", "N", "W")

    def __init__(self, state, P):
        self.state = state
        self.P = P                 # prior of the move that led here
        self.children = {}         # col -> _Node
        self.N = 0
        self.W = 0.0


def _terminal_value(state):
    """If terminal, value in [0,1] for the player TO MOVE: a 'win' means the previous mover completed
    4 -> side to move has lost -> 0.0; draw -> 0.5. None if not terminal."""
    t = state.terminal()
    if t is None:
        return None
    return 0.5 if t == "draw" else 0.0


def search(net, root_state, sims=200, c_puct=1.5, dirichlet=0.0, dir_alpha=1.0, rng=None):
    """Run `sims` PUCT simulations from root_state; return the root _Node."""
    root = _Node(root_state, 1.0)
    probs, _ = _net_eval(net, root_state)
    legal = root_state.legal()
    if dirichlet > 0 and legal:                                    # root exploration noise (self-play)
        r = rng if rng is not None else np.random
        noise = r.dirichlet([dir_alpha] * len(legal))
        nz = dict(zip(legal, noise))
        probs = [(1 - dirichlet) * probs[c] + dirichlet * nz[c] if c in nz else probs[c]
                 for c in range(7)]
    for c in legal:
        root.children[c] = _Node(root_state.play(c), probs[c])

    for _ in range(sims):
        node = root
        path = [node]
        while node.children:                                      # select to a leaf
            sqrtN = math.sqrt(node.N + 1)
            best_u, best_child = -1e18, None
            for ch in node.children.values():
                q = (ch.W / ch.N) if ch.N > 0 else 0.0
                u = (1.0 - q) + c_puct * ch.P * sqrtN / (1 + ch.N)  # 1-q: child value is opponent's
                if u > best_u:
                    best_u, best_child = u, ch
            node = best_child
            path.append(node)

        tv = _terminal_value(node.state)
        if tv is not None:
            leaf_v = tv
        else:
            probs, leaf_v = _net_eval(net, node.state)
            if node.N == 0 and node.state.legal():                # expand
                for c in node.state.legal():
                    node.children[c] = _Node(node.state.play(c), probs[c])

        v = leaf_v
        for nd in reversed(path):                                 # backup, flipping sides
            nd.N += 1
            nd.W += v
            v = 1.0 - v
    return root


def visit_probs(root):
    """Normalized visit-count distribution over the 7 columns (0 for illegal)."""
    v = np.zeros(7, dtype=np.float32)
    for c, ch in root.children.items():
        v[c] = ch.N
    s = v.sum()
    return v / s if s > 0 else v


def root_value(root):
    return (root.W / root.N) if root.N > 0 else 0.5


def mcts_move(net, root_state, sims=200, c_puct=1.5):
    root = search(net, root_state, sims, c_puct)
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


def policy_move(net, state):
    """Open-loop: argmax of the legal policy, no search."""
    probs, _ = _net_eval(net, state)
    legal = set(state.legal())
    return max(range(7), key=lambda c: probs[c] if c in legal else -1.0)
