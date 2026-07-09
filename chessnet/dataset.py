"""Compact in-RAM dataset (proposal 4.1, 6: whole set in unified memory).

A shard is a single `.npz` holding parallel arrays over N positions:
  codes    uint8  [N, 64]   oriented piece codes (see encoding.board_to_codes)
  meta     uint8  [N, 5]    castling + ep meta bits
  target   int16  [N]       best-move index 0..4095 (oriented frame)
  winprob  float32[N]       side-to-move win prob after the best move

~69 + 2 + 4 = 75 bytes/position, so 100M positions ~= 7.5 GB — trivially RAM
resident on the 256 GB target. The batched loader reconstructs onehot/packed
inputs on the fly with the vectorized encoders, and moves each batch to MLX.
"""

from __future__ import annotations

import glob
import os

import mlx.core as mx
import numpy as np

from .encoding import BATCH_ENCODERS


# Game-phase codes (proposal wants opening/middlegame/endgame coverage).
PHASE_OPENING, PHASE_MIDDLE, PHASE_END = 0, 1, 2
PHASE_NAMES = {PHASE_OPENING: "opening", PHASE_MIDDLE: "middlegame",
               PHASE_END: "endgame"}


def save_shard(path: str, codes, meta, target, winprob, phase=None,
               soft_idx=None, soft_wp=None):
    """Save a shard. `soft_idx`/`soft_wp` are the sparse per-move advantage map:
    [N, K] move indices and their win-probabilities (padded with -1), used by the
    soft-target distributional objective (not best-move imitation)."""
    arrays = dict(
        codes=np.asarray(codes, dtype=np.uint8),
        meta=np.asarray(meta, dtype=np.uint8),
        target=np.asarray(target, dtype=np.int16),
        winprob=np.asarray(winprob, dtype=np.float32),
    )
    if phase is not None:
        arrays["phase"] = np.asarray(phase, dtype=np.uint8)
    if soft_idx is not None:
        arrays["soft_idx"] = np.asarray(soft_idx, dtype=np.int16)
        arrays["soft_wp"] = np.asarray(soft_wp, dtype=np.float32)
    np.savez(path, **arrays)


def _pad_k(a: np.ndarray, k: int, pad) -> np.ndarray:
    """Right-pad [N, k0] to [N, k] with `pad`."""
    if a.shape[1] == k:
        return a
    out = np.full((a.shape[0], k), pad, dtype=a.dtype)
    out[:, :a.shape[1]] = a
    return out


class Dataset:
    """Loads one or more shards fully into RAM and yields encoded mx batches."""

    def __init__(self, shard_paths: list[str], encoding: str = "onehot"):
        self.encoding = encoding
        self.encode_batch = BATCH_ENCODERS[encoding]
        codes, meta, target, winprob, phase = [], [], [], [], []
        soft_idx, soft_wp = [], []
        self.has_soft = True
        for p in shard_paths:
            z = np.load(p)
            codes.append(z["codes"])
            meta.append(z["meta"])
            target.append(z["target"])
            winprob.append(z["winprob"])
            n = len(z["target"])
            # phase is optional (older shards may lack it); default to -1/unknown
            phase.append(z["phase"] if "phase" in z
                         else np.full(n, 255, dtype=np.uint8))
            if "soft_idx" in z:
                soft_idx.append(z["soft_idx"])
                soft_wp.append(z["soft_wp"])
            else:
                self.has_soft = False
        self.codes = np.concatenate(codes)
        self.meta = np.concatenate(meta)
        self.target = np.concatenate(target).astype(np.int32)
        self.winprob = np.concatenate(winprob)
        self.phase = np.concatenate(phase)
        if self.has_soft:
            # shards may differ in K; pad to the max K with -1
            k = max(a.shape[1] for a in soft_idx)
            self.soft_idx = np.concatenate([_pad_k(a, k, -1) for a in soft_idx])
            self.soft_wp = np.concatenate([_pad_k(a, k, -1.0) for a in soft_wp])
        else:
            self.soft_idx = self.soft_wp = None

    def __len__(self):
        return len(self.target)

    def subsample(self, n_or_frac, seed: int = 0) -> "Dataset":
        """Return a new Dataset with a random subset (for the training-size axis).

        `n_or_frac` is an absolute count (>1) or a fraction in (0,1]. The subset
        is deterministic in `seed` so a data-scaling curve is reproducible.
        """
        n = len(self)
        k = int(round(n * n_or_frac)) if n_or_frac <= 1 else int(n_or_frac)
        k = max(1, min(k, n))
        idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
        sub = Dataset.__new__(Dataset)
        sub.encoding = self.encoding
        sub.encode_batch = self.encode_batch
        sub.codes = self.codes[idx]
        sub.meta = self.meta[idx]
        sub.target = self.target[idx]
        sub.winprob = self.winprob[idx]
        sub.phase = self.phase[idx]
        sub.has_soft = self.has_soft
        sub.soft_idx = self.soft_idx[idx] if self.has_soft else None
        sub.soft_wp = self.soft_wp[idx] if self.has_soft else None
        return sub

    def phase_counts(self) -> dict:
        vals, counts = np.unique(self.phase, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def filter_multipv(self, min_candidates: int) -> "Dataset":
        """Subset to positions with >= min_candidates recorded moves — needed for
        a fair soft-target test (single-PV positions have no distribution to learn,
        so soft would just collapse to hard on them)."""
        if not self.has_soft:
            raise ValueError("no soft labels to filter on")
        keep = np.nonzero((self.soft_wp >= 0).sum(1) >= min_candidates)[0]
        sub = Dataset.__new__(Dataset)
        sub.encoding = self.encoding
        sub.encode_batch = self.encode_batch
        sub.codes = self.codes[keep]
        sub.meta = self.meta[keep]
        sub.target = self.target[keep]
        sub.winprob = self.winprob[keep]
        sub.phase = self.phase[keep]
        sub.has_soft = True
        sub.soft_idx = self.soft_idx[keep]
        sub.soft_wp = self.soft_wp[keep]
        return sub

    def split(self, val_frac: float, seed: int = 0):
        """Deterministic train/val split -> (train Dataset-view, val view)."""
        n = len(self)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_frac)
        return _View(self, perm[n_val:]), _View(self, perm[:n_val])

    def _encode(self, idx: np.ndarray, soft: bool = False):
        x = mx.array(self.encode_batch(self.codes[idx], self.meta[idx]))
        if soft:
            return (x, mx.array(self.soft_idx[idx].astype(np.int32)),
                    mx.array(self.soft_wp[idx]))
        return x, mx.array(self.target[idx])

    def iter_batches(self, batch_size: int, shuffle: bool = True, seed: int = 0,
                     soft: bool = False):
        n = len(self)
        order = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield self._encode(idx, soft)


class _View:
    """A subset of a Dataset addressed by row indices (shares the backing arrays)."""

    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def subsample(self, n_or_frac, seed: int = 0) -> "_View":
        """Shrink this view (used to vary train size with val held fixed)."""
        n = len(self.indices)
        k = int(round(n * n_or_frac)) if n_or_frac <= 1 else int(n_or_frac)
        k = max(1, min(k, n))
        pick = np.random.default_rng(seed).choice(n, size=k, replace=False)
        return _View(self.base, self.indices[pick])

    @property
    def has_soft(self):
        return self.base.has_soft

    def iter_batches(self, batch_size: int, shuffle: bool = True, seed: int = 0,
                     soft: bool = False):
        order = self.indices.copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            yield self.base._encode(idx, soft)


def find_shards(data_dir: str, pattern: str = "*.npz") -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, pattern)))
