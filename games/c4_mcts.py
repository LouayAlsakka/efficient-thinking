#!/usr/bin/env python
"""PUCT MCTS for Connect-4 using the C4 net — the closed-loop search (chess-search analog).

Standard AlphaZero PUCT: prior from the net policy, leaf value from the net value head (or a true
terminal result), backup along the path. `mcts_move(net, state, sims)` returns the most-visited legal
column. `policy_move(net, state)` is the open-loop baseline (argmax legal policy, no search).
"""
from __future__ import annotations
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    __slots__ = ("state", "P", "children", "N", "W", "terminal_v")

    def __init__(self, state, P):
        self.state = state
        self.P = P                 # prior of the move that led here
        self.children = {}         # col -> _Node
        self.N = 0
        self.W = 0.0
        self.terminal_v = None     # set if this node is terminal (value for side to move at parent)


def _terminal_value(state):
    """If `state` is terminal, return value in [0,1] from the perspective of the player TO MOVE.
    A 'win' means the previous mover completed 4 -> side to move has lost -> 0.0. Draw -> 0.5."""
    t = state.terminal()
    if t is None:
        return None
    return 0.5 if t == "draw" else 0.0


def mcts_move(net, root_state, sims=200, c_puct=1.5):
    root = _Node(root_state, 1.0)
    probs, _ = _net_eval(net, root_state)
    for c in root_state.legal():
        root.children[c] = _Node(root_state.play(c), probs[c])

    for _ in range(sims):
        node = root
        path = [node]
        # select down to a leaf
        while node.children:
            best_c, best_u, best_child = None, -1e18, None
            sqrtN = math.sqrt(node.N + 1)
            for c, ch in node.children.items():
                q = (ch.W / ch.N) if ch.N > 0 else 0.0     # value is from side-to-move-at-child view
                u = (1.0 - q) + c_puct * ch.P * sqrtN / (1 + ch.N)  # 1-q: child value is opp's; we want ours
                if u > best_u:
                    best_u, best_c, best_child = u, c, ch
            node = best_child
            path.append(node)

        # evaluate leaf
        tv = _terminal_value(node.state)
        if tv is not None:
            node.terminal_v = tv
            leaf_v = tv
        else:
            probs, leaf_v = _net_eval(net, node.state)
            if node.N == 0 and node.state.legal():          # expand
                for c in node.state.legal():
                    node.children[c] = _Node(node.state.play(c), probs[c])

        # backup: leaf_v is from the leaf's side-to-move view; flip each step up
        v = leaf_v
        for nd in reversed(path):
            nd.N += 1
            nd.W += v
            v = 1.0 - v

    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


def policy_move(net, state):
    """Open-loop: argmax of the legal policy, no search."""
    probs, _ = _net_eval(net, state)
    legal = set(state.legal())
    return max(range(7), key=lambda c: probs[c] if c in legal else -1.0)
