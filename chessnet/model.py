"""The policy network: a configurable-depth/width residual MLP (proposal 3).

Body is a stack of `Linear -> nonlinearity` blocks at a fixed width W. Past a
few layers we wrap each block in a residual connection (`x = x + f(x)`) so deep
stacks stay trainable (proposal 3.2). Critically, the nonlinearity is *required*:
a pure-linear stack collapses to one matrix and N would add zero expressivity.

  input (773 or 69)  --embed-->  W  --[N residual blocks]-->  W  --head-->  4096

Config knobs that define a point on the scaling curve:
  depth  N  : number of body blocks   (5, 10, 15, 20, 30, 50)
  width  W  : hidden dimension         (64, 256, 1024, 4096)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .encoding import (INPUT_DIMS, MOVE_DIM, NUM_SQUARES, NUM_PLANES,
                       META_BITS)


@dataclass
class ModelConfig:
    encoding: str = "onehot"   # "onehot" | "packed"
    depth: int = 10            # N body blocks (constant-width mode)
    width: int = 256           # W hidden dim   (constant-width mode)
    # Explicit per-layer body widths, e.g. [1024,512,256,128] (funnel) or
    # [64,128,256,512] (pyramid). Overrides depth/width. This is how non-uniform
    # topologies (funnel / pyramid / hourglass / bottleneck) are explored.
    widths: tuple | None = None
    activation: str = "gelu"   # "gelu" | "relu"
    residual: bool = True      # residual (used where consecutive widths match)
    # head: "dense"    -> Linear(W, 4096)  (full from×to logit matrix)
    #       "factored" -> from-head + to-head (W->64 each), logit = f[from]+t[to];
    #                     ~30x fewer head params (the head dominates for wide nets).
    head: str = "dense"
    # arch: "mlp" (default) or "dualpath" — a wide/shallow (System-1 pattern) branch
    # + a narrow/deep (System-2 calculation) branch, merged and refined over ITERS
    # weight-shared "circle iterations". d_model = width.
    arch: str = "mlp"
    wide_width: int = 2048        # dualpath: System-1 branch width (broad, shallow)
    deep_width: int = 256         # dualpath: System-2 branch width (narrow, deep)
    deep_layers: int = 6          # dualpath: System-2 branch depth
    iters: int = 2                # dualpath: circle iterations (weight-shared)
    merge: str = "gate"           # dualpath: "gate" (competitive) | "add" (additive)
    # arch: "conv" — AlphaZero-style residual conv tower over the 8x8 board.
    # Reuses width=channels, depth=residual blocks. The 773 onehot vector reshapes
    # to [8,8,12] piece planes (+5 meta bits broadcast as planes) = 17 input planes.
    conv_head_channels: int = 8   # conv: 1x1 reduce before the dense move head
    # value_head: add a scalar Eval(N) in [0,1] head (expected score for side to move).
    # Closed-loop Stage 0 — trained on max_move soft_wp (best-move win-prob = position value).
    value_head: bool = False

    @property
    def input_dim(self) -> int:
        return INPUT_DIMS[self.encoding]

    def body_widths(self) -> list[int]:
        return list(self.widths) if self.widths else [self.width] * self.depth

    def param_estimate(self) -> int:
        """Total parameter count (weights + biases)."""
        if self.arch == "conv":
            C, N, hc = self.width, self.depth, self.conv_head_channels
            inp = NUM_PLANES + META_BITS                     # 17 input planes
            total = 3 * 3 * inp * C + C                      # stem 3x3
            total += N * 2 * (3 * 3 * C * C + C)             # N residual blocks (2 convs)
            total += 1 * 1 * C * hc + hc                     # 1x1 reduce
            total += NUM_SQUARES * hc * MOVE_DIM + MOVE_DIM  # dense move head
            if self.value_head:
                total += C * C + C + C + 1                   # value head (v_fc1 + v_fc2)
            return total
        if self.arch == "dualpath":
            dm, wd, dd = self.width, self.wide_width, self.deep_width
            total = self.input_dim * dm + dm                 # embed (d_model)
            total += dm * wd + wd + wd * dm + dm             # wide in+out (shared)
            total += self.deep_layers * (dd * dd + dd)       # deep stack (shared)
            if dd != dm:
                total += dd * dm + dm                        # deep->d_model proj
            total += dm * dm + dm                            # gate
            total += dm * dm + dm                            # refine
            head_in = dm
        else:
            ws = self.body_widths()
            total = self.input_dim * ws[0] + ws[0]           # embed
            for a, b in zip(ws[:-1], ws[1:]):                # body transitions
                total += a * b + b
            head_in = ws[-1]
        if self.head == "factored":
            total += 2 * (head_in * 64 + 64)                 # from + to heads
        else:
            total += head_in * MOVE_DIM + MOVE_DIM           # dense head
        return total


_ACT = {"gelu": nn.gelu, "relu": nn.relu}


class Block(nn.Module):
    """x -> act(Linear(d_in, d_out)); residual only when d_in == d_out."""

    def __init__(self, d_in: int, d_out: int, activation: str, residual: bool):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.act = _ACT[activation]
        self.residual = residual and (d_in == d_out)

    def __call__(self, x):
        h = self.act(self.fc(x))
        return x + h if self.residual else h


def _make_head(head: str, d_in: int):
    if head == "factored":
        return None, nn.Linear(d_in, 64), nn.Linear(d_in, 64)
    return nn.Linear(d_in, MOVE_DIM), None, None


def _apply_head(dense, from_h, to_h, x):
    if dense is not None:
        return dense(x)
    f, t = from_h(x), to_h(x)                       # [B,64], [B,64]
    return (f[:, :, None] + t[:, None, :]).reshape(x.shape[0], MOVE_DIM)


class DualPathNet(nn.Module):
    """Wide/shallow (System-1 pattern) + narrow/deep (System-2 calculation)
    branches, merged per-position by a learned gate and refined over `iters`
    weight-shared "circle iterations" — think longer without more parameters."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.act = _ACT[cfg.activation]
        self.iters = cfg.iters
        self.merge = cfg.merge
        dm, wd, dd = cfg.width, cfg.wide_width, cfg.deep_width
        self.embed = nn.Linear(cfg.input_dim, dm)
        # System-1: one very wide layer out and back
        self.wide_in = nn.Linear(dm, wd)
        self.wide_out = nn.Linear(wd, dm)
        # System-2: narrow deep residual stack (weight-shared across iters)
        self.deep = [Block(dd, dd, cfg.activation, True) for _ in range(cfg.deep_layers)]
        self.deep_proj = nn.Linear(dd, dm) if dd != dm else None
        self.deep_in = nn.Linear(dm, dd) if dd != dm else None
        self.gate = nn.Linear(dm, dm)
        self.refine = Block(dm, dm, cfg.activation, True)
        self.head, self.from_head, self.to_head = _make_head(cfg.head, dm)

    def __call__(self, x):
        h = self.act(self.embed(x))
        for _ in range(self.iters):
            wide = self.wide_out(self.act(self.wide_in(h)))
            d = self.deep_in(h) if self.deep_in is not None else h
            for blk in self.deep:
                d = blk(d)
            deep = self.deep_proj(d) if self.deep_proj is not None else d
            if self.merge == "gate":
                g = mx.sigmoid(self.gate(h))         # per-feature wide/deep balance
                m = g * deep + (1.0 - g) * wide
            else:
                m = wide + deep
            h = self.act(h + self.refine(m))         # "study it" refinement step
        return _apply_head(self.head, self.from_head, self.to_head, h)


class ConvBlock(nn.Module):
    """AlphaZero-style residual block: Conv3x3 -> act -> Conv3x3, + residual, act.
    Same "residual + nonlinearity, no norm" philosophy as the MLP Block; residual
    keeps deep towers trainable. NHWC, padding=1 preserves the 8x8 board."""

    def __init__(self, channels: int, activation: str):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = _ACT[activation]

    def __call__(self, x):
        h = self.act(self.c1(x))
        h = self.c2(h)
        return self.act(x + h)


class ConvNet(nn.Module):
    """Residual conv tower over the 8x8 board. The board's spatial structure is
    given to the model directly (piece planes), rather than flattened away as in
    the MLP — the hypothesis being that local tactical patterns (forks, pins,
    pawn structure) are learned far more parameter-efficiently by convolutions.

      [B,773] -> [B,8,8,17] planes -> stem -> N residual blocks -> 1x1 reduce
              -> flatten -> Linear -> 4096 (from,to) logits

    Reuses width=channels, depth=#blocks. onehot encoding only (needs planes)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.encoding != "onehot":
            raise ValueError("arch='conv' requires encoding='onehot' (piece planes)")
        C, N = cfg.width, cfg.depth
        self.act = _ACT[cfg.activation]
        self.in_planes = NUM_PLANES + META_BITS          # 12 + 5 = 17
        self.stem = nn.Conv2d(self.in_planes, C, kernel_size=3, padding=1)
        self.blocks = [ConvBlock(C, cfg.activation) for _ in range(N)]
        self.reduce = nn.Conv2d(C, cfg.conv_head_channels, kernel_size=1)
        self.head = nn.Linear(NUM_SQUARES * cfg.conv_head_channels, MOVE_DIM)
        self.has_value = cfg.value_head
        if self.has_value:
            self.v_fc1 = nn.Linear(C, C)     # value head off the global-pooled trunk
            self.v_fc2 = nn.Linear(C, 1)

    def __call__(self, x, return_value=False):
        b = x.shape[0]
        planes = x[:, :NUM_SQUARES * NUM_PLANES].reshape(b, 8, 8, NUM_PLANES)
        meta = x[:, NUM_SQUARES * NUM_PLANES:]           # [B, 5]
        meta_planes = mx.broadcast_to(meta[:, None, None, :],
                                      (b, 8, 8, META_BITS))
        h = mx.concatenate([planes, meta_planes], axis=-1)   # [B,8,8,17] NHWC
        h = self.act(self.stem(h))
        for blk in self.blocks:
            h = blk(h)                                   # [B,8,8,C] tower features
        logits = self.head(self.act(self.reduce(h)).reshape(b, -1))   # [B,4096]
        if return_value and self.has_value:
            pooled = mx.mean(h, axis=(1, 2))             # [B,C] global-avg-pool
            v = mx.sigmoid(self.v_fc2(self.act(self.v_fc1(pooled))))[:, 0]
            return logits, v                             # Eval(N) in [0,1]
        return logits


class PolicyNet(nn.Module):
    def __new__(cls, cfg: ModelConfig):
        if cfg.arch == "dualpath":
            return DualPathNet(cfg)
        if cfg.arch == "conv":
            return ConvNet(cfg)
        return super().__new__(cls)

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        ws = cfg.body_widths()
        # residuals are free in params; force on for deep stacks (3.2).
        residual = cfg.residual or len(ws) > 8
        self.embed = nn.Linear(cfg.input_dim, ws[0])
        self.act = _ACT[cfg.activation]
        # one block per consecutive width pair; equal-width pairs get a residual.
        self.blocks = [
            Block(a, b, cfg.activation, residual)
            for a, b in zip(ws[:-1], ws[1:])
        ]
        self.factored = cfg.head == "factored"
        if self.factored:
            self.from_head = nn.Linear(ws[-1], 64)
            self.to_head = nn.Linear(ws[-1], 64)
        else:
            self.head = nn.Linear(ws[-1], MOVE_DIM)

    def __call__(self, x):
        x = self.act(self.embed(x))
        for blk in self.blocks:
            x = blk(x)
        if self.factored:
            # logit(from,to) = from_score[from] + to_score[to]  (rank-1 factoring)
            f = self.from_head(x)                       # [B, 64]
            t = self.to_head(x)                         # [B, 64]
            return (f[:, :, None] + t[:, None, :]).reshape(x.shape[0], MOVE_DIM)
        return self.head(x)   # raw logits over 4096 (from,to) moves
