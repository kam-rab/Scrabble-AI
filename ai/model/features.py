"""Feature extraction for the policy network.

Converts a (board, move, rack_leave, tile_tracker, scores) snapshot into
the two tensors the model expects:
  - board_tensor : np.ndarray (N_BOARD_CHANNELS, 15, 15)
  - feature_vec  : np.ndarray (N_FEATURES,)

Board tensor channels
---------------------
  0–25  : letter A–Z present at cell (binary)
  26    : blank tile present (binary)
  27    : DL premium square (constant)
  28    : TL premium square (constant)
  29    : DW premium square (constant)
  30    : TW premium square (constant)
  31    : newly placed this move (binary)
  32    : anchor square after move (binary)

Feature vector (64 scalars)
----------------------------
  0–26  : rack leave letter counts, normalized by 7
  27–53 : tiles remaining (not on board), normalized by initial counts
  54    : immediate move score / 100
  55    : tiles played this move / 7
  56    : rightmost row / 14
  57    : rightmost col / 14
  58    : direction (0 = H, 1 = V)
  59    : word length / 15
  60    : turn number / 20
  61    : tiles on board / 100
  62    : (our_score − opp_score) / 100
  63    : opp_score / 500
"""


import numpy as np

from solver.board import Board, Tile, PREMIUM_MAP, Premium
from ai.tile_tracker import TileTracker, TILE_INDEX, TILE_ORDER
from typing import Dict, List, Set, Tuple

N_BOARD_CHANNELS: int = 33
N_FEATURES: int = 64

_PREMIUM_CHANNEL: Dict[Premium, int] = {
    Premium.DL: 27,
    Premium.TL: 28,
    Premium.DW: 29,
    Premium.TW: 30,
}


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def word_positions(
    word: str,
    rightmost: Tuple[int, int],
    direction: str,
):
    """Yield (row, col, char) for every character in a move tuple."""
    ln = len(word)
    for i, ch in enumerate(word):
        if direction == 'H':
            r = rightmost[0]
            c = rightmost[1] - ln + 1 + i
        else:
            r = rightmost[0] - ln + 1 + i
            c = rightmost[1]
        yield r, c, ch


# ---------------------------------------------------------------------------
# Leave / tile-used computation
# ---------------------------------------------------------------------------

def compute_leave(
    move: tuple,
    rack: List[str],
    board: Board,
) -> Tuple[List[str], List[str]]:
    """Return (tiles_used_from_rack, leave).

    tiles_used_from_rack : rack tiles consumed (blanks represented as '?')
    leave                : rack tiles remaining after the move
    """
    word, _, rightmost, direction = move
    rack_copy = list(rack)
    tiles_used: List[str] = []

    for r, c, ch in word_positions(word, rightmost, direction):
        if board.is_empty(r, c):
            if ch.islower():        # blank tile played
                rack_copy.remove('?')
                tiles_used.append('?')
            else:
                rack_copy.remove(ch)
                tiles_used.append(ch)

    return tiles_used, rack_copy


# ---------------------------------------------------------------------------
# Board copy + move application
# ---------------------------------------------------------------------------

def apply_move(board: Board, move: tuple) -> Board:
    """Return a new Board with the move applied; original is unchanged.

    Uses a shallow grid copy (Tile is immutable, so this is safe).
    """
    word, _, rightmost, direction = move
    new_board = Board()
    new_board.grid = [list(row) for row in board.grid]

    for r, c, ch in word_positions(word, rightmost, direction):
        if new_board.grid[r][c] is None:
            new_board.grid[r][c] = Tile(ch.upper(), ch.islower())

    return new_board


# ---------------------------------------------------------------------------
# Board tensor construction
# ---------------------------------------------------------------------------

def _build_board_tensor(
    board_after: Board,
    newly_placed: Set[Tuple[int, int]],
) -> np.ndarray:
    tensor = np.zeros((N_BOARD_CHANNELS, 15, 15), dtype=np.float32)

    for r in range(15):
        for c in range(15):
            tile = board_after.grid[r][c]
            if tile is not None:
                tensor[ord(tile.letter) - ord('A'), r, c] = 1.0
                if tile.is_blank:
                    tensor[26, r, c] = 1.0

            # Premium channels (board topology, constant)
            p = PREMIUM_MAP.get((r, c), Premium.NONE)
            if p in _PREMIUM_CHANNEL:
                tensor[_PREMIUM_CHANNEL[p], r, c] = 1.0

            if (r, c) in newly_placed:
                tensor[31, r, c] = 1.0

    for r, c in board_after.get_anchors():
        tensor[32, r, c] = 1.0

    return tensor


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract(
    board: Board,
    move: tuple,
    leave: List[str],
    tile_tracker: TileTracker,
    our_score: int,
    opp_score: int,
    turn_number: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (board_tensor, feature_vec) for a candidate move.

    Args:
        board        : Board state BEFORE the move.
        move         : (word, score, rightmost, direction)
        leave        : Rack tiles remaining AFTER the move.
        tile_tracker : Current tile tracker (state before move applied).
        our_score    : Cumulative score before this move.
        opp_score    : Opponent's cumulative score.
        turn_number  : 0-indexed turn number.

    Returns:
        board_tensor : np.ndarray (33, 15, 15)
        feature_vec  : np.ndarray (64,)
    """
    word, score, rightmost, direction = move

    # Identify newly placed squares and count tiles played
    newly_placed: Set[Tuple[int, int]] = set()
    tiles_played = 0
    for r, c, _ in word_positions(word, rightmost, direction):
        if board.is_empty(r, c):
            newly_placed.add((r, c))
            tiles_played += 1

    board_after = apply_move(board, move)
    board_tensor = _build_board_tensor(board_after, newly_placed)

    # --- Feature vector ---

    # 1. Rack leave counts (27 dims), normalized by max rack size
    leave_counts = np.zeros(27, dtype=np.float32)
    for tile in leave:
        leave_counts[TILE_INDEX[tile]] += 1.0
    leave_counts /= 7.0

    # 2. Remaining tile counts (27 dims), normalized
    remaining_vec = tile_tracker.to_vector()

    # 3. Scalar features (10 dims)
    scalars = np.array([
        score / 100.0,
        tiles_played / 7.0,
        rightmost[0] / 14.0,
        rightmost[1] / 14.0,
        1.0 if direction == 'V' else 0.0,
        len(word) / 15.0,
        turn_number / 20.0,
        board.occupied_count() / 100.0,
        (our_score - opp_score) / 100.0,
        opp_score / 500.0,
    ], dtype=np.float32)

    feature_vec = np.concatenate([leave_counts, remaining_vec, scalars])
    return board_tensor, feature_vec
