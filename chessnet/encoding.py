"""Board encoding: chess.Board -> fixed-size input vectors, and move <-> index maps.

Two encodings are supported (see proposal 2.1 / 2.2):

  * "onehot" (recommended): 64 squares x 12 piece planes = 768 bits, plus 5
    meta bits (4 castling rights + en-passant-present). 773 floats total.
  * "packed" (the "lowest coefficients" ablation): 64 squares x 4 bits = 256
    floats, each square a 4-bit code, plus the same 5 meta bits. 261 floats.

Side-to-move normalization (proposal 2.1): the board is always presented from
the mover's perspective. When it is Black to move we mirror the board
vertically AND swap piece colors so that "player 0" is always the side to move.
Because we mirror the board, the move indices we emit are *also* in the mirrored
frame; `decode_move` undoes the mirror to recover a real move on the original
board.

Output move head (proposal 2.3): (from, to) -> 64*64 = 4096 logits. Promotions
default to queen.
"""

from __future__ import annotations

import chess
import numpy as np

# --- dimensions -------------------------------------------------------------

NUM_SQUARES = 64
NUM_PIECE_TYPES = 6          # pawn..king
NUM_PLANES = 12              # 6 white + 6 black (from mover's perspective)
META_BITS = 5                # 4 castling rights + 1 en-passant-present
ONEHOT_DIM = NUM_SQUARES * NUM_PLANES + META_BITS   # 773
PACKED_DIM = NUM_SQUARES + META_BITS                # 69? -> see note
# Packed uses one scalar 4-bit code per square (64) + meta (5) = 69 scalars.
# (The proposal's "256-bit" figure counts raw bits; as network *inputs* we feed
# 64 scalar codes normalized to [0,1], which is the faithful "packed" variant.)

MOVE_DIM = NUM_SQUARES * NUM_SQUARES  # 4096

# 4-bit piece codes for the packed encoding (mover-relative colors).
# 0 = empty; 1..6 = own pieces (P N B R Q K); 7..12 = opponent pieces.
_PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
]


def _oriented(board: chess.Board) -> tuple[chess.Board, bool]:
    """Return (board_in_mover_frame, mirrored).

    If it's Black to move, return board.mirror() (vertical flip + color swap) so
    the side to move is always "white" in the returned board. `mirrored` records
    whether we flipped, so callers can translate moves back.
    """
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True


def _meta_bits(oriented: chess.Board) -> np.ndarray:
    """Castling + en-passant meta bits, in the mover's frame.

    After orientation the side to move is always WHITE, so "own" castling rights
    are the white rights of the oriented board and "opponent" are the black ones.
    """
    m = np.zeros(META_BITS, dtype=np.float32)
    m[0] = float(oriented.has_kingside_castling_rights(chess.WHITE))
    m[1] = float(oriented.has_queenside_castling_rights(chess.WHITE))
    m[2] = float(oriented.has_kingside_castling_rights(chess.BLACK))
    m[3] = float(oriented.has_queenside_castling_rights(chess.BLACK))
    m[4] = float(oriented.ep_square is not None)
    return m


def encode_onehot(board: chess.Board) -> np.ndarray:
    """768 piece-plane bits + 5 meta bits = 773 floats, mover-relative."""
    oriented, _ = _oriented(board)
    planes = np.zeros((NUM_SQUARES, NUM_PLANES), dtype=np.float32)
    for sq, piece in oriented.piece_map().items():
        # own pieces (white in oriented frame) -> planes 0..5
        # opponent pieces (black) -> planes 6..11
        base = 0 if piece.color == chess.WHITE else NUM_PIECE_TYPES
        plane = base + _PIECE_ORDER.index(piece.piece_type)
        planes[sq, plane] = 1.0
    return np.concatenate([planes.reshape(-1), _meta_bits(oriented)])


def encode_packed(board: chess.Board) -> np.ndarray:
    """64 scalar 4-bit codes (normalized to [0,1]) + 5 meta bits, mover-relative."""
    oriented, _ = _oriented(board)
    codes = np.zeros(NUM_SQUARES, dtype=np.float32)
    for sq, piece in oriented.piece_map().items():
        idx = _PIECE_ORDER.index(piece.piece_type) + 1  # 1..6
        code = idx if piece.color == chess.WHITE else idx + NUM_PIECE_TYPES  # 7..12
        codes[sq] = code / 15.0  # normalize 4-bit range to [0,1]
    return np.concatenate([codes, _meta_bits(oriented)])


ENCODERS = {"onehot": encode_onehot, "packed": encode_packed}
INPUT_DIMS = {"onehot": ONEHOT_DIM, "packed": PACKED_DIM}


# --- move <-> index ---------------------------------------------------------

def move_to_index(move: chess.Move, mirrored: bool) -> int:
    """(from,to) -> 0..4095, in the oriented frame.

    `mirrored` must match the frame the encoding was produced in: if the board
    was mirrored for Black-to-move, the move's squares are mirrored too.
    """
    frm, to = move.from_square, move.to_square
    if mirrored:
        frm = chess.square_mirror(frm)
        to = chess.square_mirror(to)
    return frm * NUM_SQUARES + to


def index_to_from_to(index: int) -> tuple[int, int]:
    return divmod(index, NUM_SQUARES)


def decode_move(index: int, board: chess.Board) -> chess.Move | None:
    """Turn a 0..4095 move index (in the board's oriented frame) into a legal
    chess.Move on `board`, or None if it corresponds to no legal move.

    Promotions auto-resolve to queen (proposal 2.3). The index frame matches the
    frame `encode_*` used for this board, so we mirror squares back when Black is
    to move.
    """
    mirrored = board.turn == chess.BLACK
    frm, to = index_to_from_to(index)
    if mirrored:
        frm = chess.square_mirror(frm)
        to = chess.square_mirror(to)
    # Try plain move, then queen-promotion (covers pawn reaching last rank).
    candidate = chess.Move(frm, to)
    if candidate in board.legal_moves:
        return candidate
    promo = chess.Move(frm, to, promotion=chess.QUEEN)
    if promo in board.legal_moves:
        return promo
    return None


# --- compact storage form + vectorized batch encoders ----------------------
#
# The on-disk dataset stores each position as 64 uint8 piece codes (oriented,
# mover-relative: 0=empty, 1..6 own P..K, 7..12 opponent P..K) plus 5 uint8 meta
# bits. This is ~69 bytes/position and lets us reconstruct *either* input
# encoding at train time with pure-numpy vectorized ops (no python-chess in the
# hot loop). `board_to_codes` is the single place board->codes orientation lives.

def board_to_codes(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    """chess.Board -> (codes[64] uint8, meta[5] uint8), in the mover's frame."""
    oriented, _ = _oriented(board)
    codes = np.zeros(NUM_SQUARES, dtype=np.uint8)
    for sq, piece in oriented.piece_map().items():
        idx = _PIECE_ORDER.index(piece.piece_type) + 1  # 1..6
        codes[sq] = idx if piece.color == chess.WHITE else idx + NUM_PIECE_TYPES
    return codes, _meta_bits(oriented).astype(np.uint8)


def codes_to_onehot_batch(codes: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """[B,64] uint8 codes + [B,5] meta -> [B,773] float32 (vectorized).

    Code 0 (empty) maps to an all-zero plane row; codes 1..12 map to planes
    0..11. Matches `encode_onehot`'s layout exactly.
    """
    b = codes.shape[0]
    planes = np.zeros((b, NUM_SQUARES, NUM_PLANES), dtype=np.float32)
    occ = codes > 0
    rows, cols = np.nonzero(occ)
    planes[rows, cols, codes[occ] - 1] = 1.0
    return np.concatenate(
        [planes.reshape(b, -1), meta.astype(np.float32)], axis=1)


def codes_to_packed_batch(codes: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """[B,64] uint8 codes + [B,5] meta -> [B,69] float32 (vectorized)."""
    scaled = codes.astype(np.float32) / 15.0
    return np.concatenate([scaled, meta.astype(np.float32)], axis=1)


BATCH_ENCODERS = {
    "onehot": codes_to_onehot_batch,
    "packed": codes_to_packed_batch,
}


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of length 4096, True where a legal move maps (oriented frame).

    Used for masked-mode inference (proposal 4.3). Multiple promotion pieces
    collapse to the same (from,to) index; that's fine for masking.
    """
    mirrored = board.turn == chess.BLACK
    mask = np.zeros(MOVE_DIM, dtype=bool)
    for mv in board.legal_moves:
        mask[move_to_index(mv, mirrored)] = True
    return mask
