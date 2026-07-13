#!/usr/bin/env python
"""Connect-4 policy+value net — the Stage-1 evaluator for the games arm.

AlphaZero-shaped but tiny: a small residual conv tower over the 6x7 board with a policy head (7 column
logits) and a value head (scalar in [0,1], side-to-move win prob). This is the 'evaluator' whose
open-loop (argmax policy) vs closed-loop (net+MCTS) strength gap we measure — the Connect-4 analog of
the chess decomposition, on a simpler game.

encode() lays out 84 floats as [plane in (me,opp)][col 0..6][row 0..5]; we reshape to NHWC [6,7,2].
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn


def to_planes(x):
    """[B,84] flat encode() -> [B,6,7,2] NHWC (H=row, W=col, C=plane)."""
    b = x.shape[0]
    x = x.reshape(b, 2, 7, 6)          # [B, plane, col, row]
    return x.transpose(0, 3, 2, 1)     # [B, row, col, plane] = [B,6,7,2]


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.b1 = nn.BatchNorm(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.b2 = nn.BatchNorm(c)

    def __call__(self, x):
        h = nn.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return nn.relu(x + h)


class C4Net(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(2, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm(channels)
        self.blocks = [ResBlock(channels) for _ in range(blocks)]
        # policy head
        self.p_reduce = nn.Conv2d(channels, 2, 1)
        self.p_fc = nn.Linear(2 * 6 * 7, 7)
        # value head
        self.v_reduce = nn.Conv2d(channels, 1, 1)
        self.v_fc1 = nn.Linear(6 * 7, 32)
        self.v_fc2 = nn.Linear(32, 1)

    def __call__(self, x):
        h = nn.relu(self.stem_bn(self.stem(to_planes(x))))
        for blk in self.blocks:
            h = blk(h)
        b = h.shape[0]
        p = nn.relu(self.p_reduce(h)).reshape(b, -1)
        logits = self.p_fc(p)                                  # [B,7] policy logits
        v = nn.relu(self.v_reduce(h)).reshape(b, -1)
        v = nn.relu(self.v_fc1(v))
        value = mx.sigmoid(self.v_fc2(v)).reshape(b)           # [B] in [0,1]
        return logits, value


def masked_policy(logits, legal_cols):
    """Softmax over legal columns only. logits:[7] mx array, legal_cols: list[int] -> [7] probs."""
    neg = mx.full((7,), -1e9)
    mask = mx.zeros((7,))
    for c in legal_cols:
        mask = mask + mx.eye(7)[c]
    masked = mx.where(mask > 0, logits, neg)
    return mx.softmax(masked)
