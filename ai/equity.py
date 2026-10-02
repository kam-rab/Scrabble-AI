"""Move ranking using model inference (or raw score as fallback).

rank_moves() is the single function used by both ScrabbleAI.choose_move()
and the self-play loop.  If no inference object is supplied it falls back
to sorting by immediate raw score, giving a clean greedy baseline.
"""


from solver.board import Board
from ai.tile_tracker import TileTracker
from typing import List, Tuple


def rank_moves(
    moves: List[tuple],
    board: Board,
    rack: List[str],
    tile_tracker: TileTracker,
    our_score: int,
    opp_score: int,
    turn_number: int,
    inference=None,
    top_n: int = 15,
) -> List[Tuple[tuple, float]]:
    """Return moves sorted by equity (model) or raw score (greedy fallback).

    Args:
        moves        : All legal moves sorted by raw score descending.
        board        : Current board state.
        rack         : Current player's rack.
        tile_tracker : Current tile tracker.
        our_score    : Player's cumulative score.
        opp_score    : Opponent's cumulative score.
        turn_number  : 0-indexed turn number.
        inference    : PolicyInference instance, or None for greedy.
        top_n        : Max candidates passed to the model.

    Returns:
        List of (move, equity) sorted by equity descending.
    """
    if inference is None:
        return [(m, float(m[1])) for m in moves]

    return inference.rank_moves(
        moves, board, rack, tile_tracker,
        our_score, opp_score, turn_number,
        top_n=top_n,
    )
