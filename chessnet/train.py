"""Supervised cross-entropy trainer (proposal 4.2 option 1).

Trains a PolicyNet to predict Stockfish's best move as a 4096-way classification.
This is the stable, known-to-work starting objective; reward-weighted training
(proposal 4.2 option 2) is a later phase built on the same model + data.

Everything a single scaling-curve point needs is captured in RunConfig, which is
serialized to the run directory alongside metrics and weights so the sweep is
reproducible and the Elo-vs-N/params plot can be rebuilt from disk.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .model import ModelConfig, PolicyNet


@dataclass
class RunConfig:
    # --- model (a point on the scaling curve) ---
    encoding: str = "onehot"
    depth: int = 10
    width: int = 256
    widths: tuple | None = None   # explicit body-width schedule (funnel/pyramid/…)
    activation: str = "gelu"
    residual: bool = True
    head: str = "dense"           # "dense" | "factored" (from-head + to-head)
    # dualpath architecture (wide/shallow + narrow/deep + circle iterations)
    arch: str = "mlp"
    wide_width: int = 2048
    deep_width: int = 256
    deep_layers: int = 6
    iters: int = 2
    merge: str = "gate"
    conv_head_channels: int = 8   # conv: 1x1 reduce channels before move head
    value_head: bool = False      # train a scalar Eval(N)∈[0,1] head alongside policy
    value_weight: float = 0.5     # weight of the value MSE in the combined loss
    grad_clip: float = 0.0        # >0: clip gradients to this global L2 norm (stability)
    # --- optimization ---
    batch_size: int = 1024
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    val_frac: float = 0.05
    train_fraction: float = 1.0   # <1 shrinks TRAIN only (data-scaling axis)
    # objective: "soft" fits the model's move distribution to the engine's
    # advantage-weighted map over ALL legal moves (needs multi-PV soft labels);
    # "hard" is single-best-move cross-entropy (imitation, capped at the judge).
    objective: str = "hard"
    tau: float = 0.08             # softmax temperature (winprob space) for soft target
    # soft target construction over the per-move win-probs v_i:
    #   "softmax" -> t_i ∝ exp(v_i/tau)                 (Boltzmann on eval)
    #   "ratio"   -> t_i ∝ clip((v_i-b)/(v*-b), 0, 1)   (linear advantage; user's method)
    soft_target: str = "softmax"
    soft_baseline: str = "min"    # ratio baseline b: "min" | "mean" | "half"
    soft_sharpen: float = 1.0     # ratio exponent γ: >1 concentrates on best moves
    seed: int = 0
    # --- bookkeeping ---
    data_glob: str = "data/smoke.*.npz"
    run_dir: str = "runs/exp"
    log_every: int = 50
    ckpt_every: int = 0           # >0: save an atomic model.npz every N steps (crash safety)

    def model_config(self) -> ModelConfig:
        return ModelConfig(encoding=self.encoding, depth=self.depth,
                           width=self.width,
                           widths=tuple(self.widths) if self.widths else None,
                           activation=self.activation, residual=self.residual,
                           head=self.head, arch=self.arch,
                           wide_width=self.wide_width, deep_width=self.deep_width,
                           deep_layers=self.deep_layers, iters=self.iters,
                           merge=self.merge,
                           conv_head_channels=self.conv_head_channels,
                           value_head=self.value_head)


def load_run(run_dir: str):
    """Rebuild (model, RunConfig) from a run directory's config + weights."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = RunConfig(**json.load(f))
    model = PolicyNet(cfg.model_config())
    model.load_weights(os.path.join(run_dir, "model.npz"))
    mx.eval(model.parameters())
    return model, cfg


def cross_entropy_loss(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def build_soft_target(soft_wp, mode="softmax", tau=0.08, baseline="min",
                      sharpen=1.0):
    """Turn per-move win-probs v_i (padded with <0) into a target distribution.

      softmax: t_i ∝ exp(v_i / tau)                        (Boltzmann on eval)
      ratio  : t_i ∝ clip((v_i - b) / (v* - b), 0, 1) ** γ (linear advantage,
               optionally sharpened by exponent γ=`sharpen`)
               b = min / mean / 0.5 of the valid candidates; v* = best.

    The `ratio` mode weights each move by how far it moves toward the best. γ=1 is
    the plain spread; γ>1 concentrates mass on the better moves (a "temperature"
    for committed play) while keeping the direction-of-good structure; γ→∞ → the
    single best move (== hard).
    """
    valid = soft_wp >= 0
    validf = valid.astype(mx.float32)
    NEG = mx.full(soft_wp.shape, -1e30)
    if mode == "softmax":
        z = mx.where(valid, soft_wp / tau, NEG)
        z = z - mx.max(z, axis=1, keepdims=True)
        e = mx.exp(z) * validf
        return e / mx.sum(e, axis=1, keepdims=True)
    # ratio (linear advantage)
    vstar = mx.max(mx.where(valid, soft_wp, NEG), axis=1, keepdims=True)
    if baseline == "min":
        b = mx.min(mx.where(valid, soft_wp, mx.full(soft_wp.shape, 1e30)),
                   axis=1, keepdims=True)
    elif baseline == "mean":
        b = mx.sum(soft_wp * validf, axis=1, keepdims=True) / \
            mx.maximum(mx.sum(validf, axis=1, keepdims=True), 1.0)
    else:  # "half"
        b = mx.full((soft_wp.shape[0], 1), 0.5)
    q = mx.clip((soft_wp - b) / mx.maximum(vstar - b, 1e-6), 0.0, 1.0)
    if sharpen != 1.0:
        q = q ** sharpen
    q = q * validf
    ssum = mx.sum(q, axis=1, keepdims=True)
    # if the whole row collapses (v* == b), fall back to uniform over valid moves
    uniform = validf / mx.maximum(mx.sum(validf, axis=1, keepdims=True), 1.0)
    return mx.where(ssum > 1e-9, q / mx.maximum(ssum, 1e-9), uniform)


def soft_target_loss(model, x, soft_idx, soft_wp, tau,
                     mode="softmax", baseline="min", sharpen=1.0):
    """Cross-entropy between the model's move distribution and the eval-weighted
    target over legal moves (built by `build_soft_target`). NOT best-move
    imitation, so it is not capped by copying the judge's single top choice."""
    logits = model(x)
    logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    valid = soft_wp >= 0
    t = build_soft_target(soft_wp, mode, tau, baseline, sharpen)   # [B, K] target
    gather = mx.where(valid, soft_idx, mx.zeros(soft_idx.shape, soft_idx.dtype))
    chosen_logp = mx.take_along_axis(logp, gather, axis=1)   # [B, K]
    return -mx.mean(mx.sum(t * chosen_logp, axis=1))


def pv_loss(model, x, soft_idx, soft_wp, tau, mode, baseline, sharpen, vw):
    """Combined policy (soft) + value (Eval) loss — closed-loop Stage 0.

    Value target = the position's expected score for the side to move = the
    best legal move's win-prob = max over valid soft_wp. Policy uses the same
    soft-target cross-entropy as soft_target_loss."""
    logits, v = model(x, return_value=True)
    logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    valid = soft_wp >= 0
    t = build_soft_target(soft_wp, mode, tau, baseline, sharpen)
    gather = mx.where(valid, soft_idx, mx.zeros(soft_idx.shape, soft_idx.dtype))
    chosen_logp = mx.take_along_axis(logp, gather, axis=1)
    pol = -mx.mean(mx.sum(t * chosen_logp, axis=1))
    vt = mx.max(mx.where(valid, soft_wp, mx.full(soft_wp.shape, -1.0)), axis=1)
    val = mx.mean((v - vt) ** 2)
    return pol + vw * val


def _accuracy(model, view, batch_size):
    """top-1 and top-5 move accuracy vs Stockfish best move."""
    correct1 = correct5 = total = 0
    for x, y in view.iter_batches(batch_size, shuffle=False):
        logits = model(x)
        top5 = mx.argpartition(-logits, 5, axis=1)[:, :5]
        pred1 = mx.argmax(logits, axis=1)
        correct1 += int(mx.sum(pred1 == y).item())
        correct5 += int(mx.sum(mx.any(top5 == y[:, None], axis=1)).item())
        total += y.shape[0]
    if total == 0:
        return 0.0, 0.0
    return correct1 / total, correct5 / total


def cosine_lr(step, total_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    import math
    prog = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(prog, 1.0)))


def _clip_grads(grads, max_norm):
    """Clip gradients to a global L2 norm (prevents the divergence-into-a-dead-basin
    failure seen with deep conv at high LR). Returns the (possibly scaled) grads."""
    from mlx.utils import tree_flatten, tree_unflatten
    leaves = tree_flatten(grads)
    total = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in leaves))
    scale = mx.where(total > max_norm, max_norm / (total + 1e-6), 1.0)
    return tree_unflatten([(k, g * scale) for k, g in leaves])


def _save_ckpt(model, run_dir, step, epoch, loss):
    """Atomically overwrite model.npz (write temp, then os.replace) so a crash
    mid-write can never corrupt the checkpoint. Leaves a usable, evaluable model
    at the latest step."""
    tmp = os.path.join(run_dir, "model.ckpt.npz")
    model.save_weights(tmp)
    os.replace(tmp, os.path.join(run_dir, "model.npz"))
    with open(os.path.join(run_dir, "ckpt_meta.json"), "w") as f:
        json.dump({"step": step, "epoch": epoch, "loss": float(loss)}, f)


def train(cfg: RunConfig, dataset):
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    mx.random.seed(cfg.seed)
    model = PolicyNet(cfg.model_config())
    mx.eval(model.parameters())
    n_params = cfg.model_config().param_estimate()

    train_view, val_view = dataset.split(cfg.val_frac, cfg.seed)
    if cfg.train_fraction < 1.0:
        train_view = train_view.subsample(cfg.train_fraction, cfg.seed)
    steps_per_epoch = max(1, len(train_view) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs

    # value_head forces soft batches (needs per-move win-probs for the value target)
    soft = cfg.objective == "soft" or cfg.value_head
    if soft and not getattr(dataset, "has_soft", False):
        raise ValueError("soft/value objective needs multi-PV soft labels; "
                         "re-label with scripts/label.py --soft")
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.value_head:
        _loss = lambda m, x, si, sw: pv_loss(
            m, x, si, sw, cfg.tau, cfg.soft_target, cfg.soft_baseline,
            cfg.soft_sharpen, cfg.value_weight)
    elif soft:
        _loss = lambda m, x, si, sw: soft_target_loss(
            m, x, si, sw, cfg.tau, cfg.soft_target, cfg.soft_baseline,
            cfg.soft_sharpen)
    else:
        _loss = cross_entropy_loss
    loss_and_grad = nn.value_and_grad(model, _loss)

    print(f"[{cfg.run_dir}] depth={cfg.depth} width={cfg.width} "
          f"enc={cfg.encoding} obj={cfg.objective} ~{n_params/1e6:.2f}M params | "
          f"train={len(train_view)} val={len(val_view)} "
          f"steps/epoch={steps_per_epoch}")

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        for batch in train_view.iter_batches(cfg.batch_size, shuffle=True,
                                             seed=cfg.seed + epoch, soft=soft):
            opt.learning_rate = cosine_lr(step, total_steps, cfg.lr,
                                          cfg.warmup_steps)
            loss, grads = loss_and_grad(model, *batch)
            if cfg.grad_clip > 0:
                grads = _clip_grads(grads, cfg.grad_clip)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            if step % cfg.log_every == 0:
                rate = (step + 1) * cfg.batch_size / (time.time() - t0)
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"lr {opt.learning_rate.item():.2e} ({rate:.0f} pos/s)")
            if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
                _save_ckpt(model, cfg.run_dir, step, epoch, loss.item())
                print(f"  [ckpt] saved model.npz at step {step}")
            step += 1
        acc1, acc5 = _accuracy(model, val_view, cfg.batch_size)
        rec = {"epoch": epoch, "step": step, "val_top1": acc1, "val_top5": acc5,
               "loss": float(loss.item())}
        history.append(rec)
        print(f"  [epoch {epoch}] val top1={acc1:.3f} top5={acc5:.3f}")

    model.save_weights(os.path.join(cfg.run_dir, "model.npz"))
    wall = time.time() - t0
    with open(os.path.join(cfg.run_dir, "metrics.json"), "w") as f:
        json.dump({"params": n_params, "input_dim": cfg.model_config().input_dim,
                   "n_train": len(train_view), "n_val": len(val_view),
                   "history": history,
                   "final_top1": history[-1]["val_top1"],
                   "final_top5": history[-1]["val_top5"],
                   "wall_sec": wall}, f, indent=2)
    print(f"[{cfg.run_dir}] done in {time.time()-t0:.1f}s "
          f"final top1={history[-1]['val_top1']:.3f}")
    return model, history
