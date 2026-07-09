"""Fast correctness tests for the core pipeline (run with: python -m pytest tests)."""

import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessnet import encoding as E
from chessnet.model import ModelConfig, PolicyNet
from chessnet.dataset import save_shard, Dataset


def test_encoding_dims():
    b = chess.Board()
    assert E.encode_onehot(b).shape == (E.ONEHOT_DIM,)
    assert E.encode_packed(b).shape == (E.PACKED_DIM,)
    # 32 pieces at start + no castling loss yet (4 rights) = 36 set bits in onehot
    assert E.encode_onehot(b).sum() == 36.0


def test_move_roundtrip_white_and_black():
    b = chess.Board()
    mv = chess.Move.from_uci("e2e4")
    idx = E.move_to_index(mv, mirrored=False)
    assert E.decode_move(idx, b) == mv

    # Black to move: index is in the mirrored frame, decode must recover it.
    for uci in ("e4", "e5", "Nf3"):
        b.push_san(uci)
    assert b.turn == chess.BLACK
    black_mv = chess.Move.from_uci("b8c6")
    idx = E.move_to_index(black_mv, mirrored=True)
    assert E.decode_move(idx, b) == black_mv


def test_legal_mask_matches_legal_moves():
    b = chess.Board()
    for uci in ("e4", "c5", "Nf3"):
        b.push_san(uci)
    mask = E.legal_move_mask(b)
    assert mask.sum() == b.legal_moves.count()
    # every legal move's index is set
    mirrored = b.turn == chess.BLACK
    for mv in b.legal_moves:
        assert mask[E.move_to_index(mv, mirrored)]


def test_batch_encoder_matches_scalar():
    boards = [chess.Board()]
    b2 = chess.Board()
    for uci in ("e4", "e5", "Nf3", "Nc6"):
        b2.push_san(uci)
    boards.append(b2)
    codes = np.stack([E.board_to_codes(b)[0] for b in boards])
    meta = np.stack([E.board_to_codes(b)[1] for b in boards])
    batch = E.codes_to_onehot_batch(codes, meta)
    for i, b in enumerate(boards):
        assert np.array_equal(batch[i], E.encode_onehot(b))


def test_model_forward_shape():
    import mlx.core as mx
    cfg = ModelConfig(encoding="onehot", depth=4, width=32)
    net = PolicyNet(cfg)
    x = mx.zeros((3, E.ONEHOT_DIM))
    out = net(x)
    assert out.shape == (3, E.MOVE_DIM)


def test_param_estimate_positive_and_monotonic():
    small = ModelConfig(depth=5, width=64).param_estimate()
    deep = ModelConfig(depth=30, width=64).param_estimate()
    wide = ModelConfig(depth=5, width=1024).param_estimate()
    assert 0 < small < deep < wide


def test_dataset_roundtrip(tmp_path):
    codes = np.random.randint(0, 13, size=(20, 64), dtype=np.uint8)
    meta = np.random.randint(0, 2, size=(20, 5), dtype=np.uint8)
    target = np.random.randint(0, 4096, size=20)
    winprob = np.random.rand(20).astype(np.float32)
    p = str(tmp_path / "s.npz")
    save_shard(p, codes, meta, target, winprob)
    ds = Dataset([p], encoding="onehot")
    assert len(ds) == 20
    x, y = next(ds.iter_batches(8, shuffle=False))
    assert x.shape == (8, E.ONEHOT_DIM)
    assert y.shape == (8,)
    # phase array present and defaulted to "unknown" (255) when not saved
    assert ds.phase.shape == (20,) and (ds.phase == 255).all()


def test_dataset_carries_phase(tmp_path):
    n = 12
    phase = np.array([0, 1, 2] * 4, dtype=np.uint8)
    p = str(tmp_path / "s.npz")
    save_shard(p, np.zeros((n, 64), np.uint8), np.zeros((n, 5), np.uint8),
               np.zeros(n), np.zeros(n), phase=phase)
    ds = Dataset([p], encoding="packed")
    assert np.array_equal(ds.phase, phase)


def test_soft_targets_roundtrip_and_loss(tmp_path):
    import mlx.core as mx
    from chessnet.model import ModelConfig, PolicyNet
    from chessnet.train import soft_target_loss
    n, k = 10, 8
    rng = np.random.default_rng(0)
    soft_idx = rng.integers(0, 4096, size=(n, k)).astype(np.int16)
    soft_wp = rng.random((n, k)).astype(np.float32)
    soft_wp[:, -2:] = -1.0  # pad the last two candidates
    p = str(tmp_path / "s.npz")
    save_shard(p, np.zeros((n, 64), np.uint8), np.zeros((n, 5), np.uint8),
               np.zeros(n), np.zeros(n), soft_idx=soft_idx, soft_wp=soft_wp)
    ds = Dataset([p], encoding="onehot")
    assert ds.has_soft and ds.soft_idx.shape == (n, k)
    x, si, sw = next(ds.iter_batches(4, shuffle=False, soft=True))
    assert si.shape == (4, k) and sw.shape == (4, k)
    net = PolicyNet(ModelConfig(depth=2, width=32))
    loss = soft_target_loss(net, x, si, sw, tau=0.08)
    assert float(loss.item()) > 0 and np.isfinite(float(loss.item()))


def test_hard_path_ignores_missing_soft(tmp_path):
    # a shard without soft labels -> has_soft False, hard batches still work
    n = 8
    p = str(tmp_path / "h.npz")
    save_shard(p, np.zeros((n, 64), np.uint8), np.zeros((n, 5), np.uint8),
               np.random.randint(0, 4096, n), np.zeros(n))
    ds = Dataset([p], encoding="packed")
    assert ds.has_soft is False
    x, y = next(ds.iter_batches(4, shuffle=False))
    assert x.shape[0] == 4 and y.shape == (4,)


def test_phase_classification():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    from label import classify_phase, OPENING_PLIES, ENDGAME_PIECES
    from chessnet.dataset import PHASE_OPENING, PHASE_MIDDLE, PHASE_END

    # early ply -> opening regardless of material
    assert classify_phase(chess.Board(), ply=2) == PHASE_OPENING
    # full board past the opening -> middlegame
    assert classify_phase(chess.Board(), ply=OPENING_PLIES + 1) == PHASE_MIDDLE
    # a bare-kings + few pieces position past the opening -> endgame
    ep = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 40")
    assert len(ep.piece_map()) <= ENDGAME_PIECES
    assert classify_phase(ep, ply=80) == PHASE_END
