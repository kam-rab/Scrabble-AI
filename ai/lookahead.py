"""1-step expectiminimax lookahead for move ranking.

For each candidate move, simulates the board state after that move, samples
possible opponent racks from the unseen tile pool, evaluates the opponent's
best greedy response for each, and returns a coverage-weighted blend of the
model equity and the concrete lookahead signal.

Near endgame (total unseen tiles <= exhaustive_threshold), enumerates all
distinct opponent racks exactly (coverage = 1.0).
"""

import random
import time
from itertools import combinations
from math import comb
from typing import List, Tuple

from solver import generate_moves
from solver.board import Board
from ai.model.features import apply_move, compute_leave
from ai.tile_tracker import TileTracker


def _best_greedy_score(board: Board, gaddag_root, rack: List[str]) -> int:
    moves = generate_moves(board, gaddag_root, rack)
    return moves[0][1] if moves else 0


def _enumerate_racks(unseen: dict, size: int) -> List[List[str]]:
    flat = [t for t, c in unseen.items() for _ in range(c)]
    if len(flat) <= size:
        return [flat]
    seen = set()
    result = []
    for combo in combinations(flat, size):
        key = tuple(sorted(combo))
        if key not in seen:
            seen.add(key)
            result.append(list(combo))
    return result


def _sample_racks(unseen: dict, size: int, k: int) -> List[List[str]]:
    flat = [t for t, c in unseen.items() for _ in range(c)]
    if len(flat) <= size:
        return [flat]
    return [random.sample(flat, size) for _ in range(k)]


def one_step_lookahead(
    moves: List[tuple],
    board: Board,
    our_rack: List[str],
    tracker: TileTracker,
    gaddag_root,
    model_equities: List[float],
    max_samples: int = 5,
    time_limit: float = 1.0,
    opp_rack_size: int = 7,
    exhaustive_threshold: int = 14,
) -> List[Tuple[tuple, float]]:
    """Return (move, final_equity) sorted best-first.

    final_equity = coverage * lookahead_equity + (1 - coverage) * model_equity

    Args:
        moves           : Candidate moves (word, score, rightmost, direction).
        board           : Board before any move this turn.
        our_rack        : Our full rack before this move.
        tracker         : TileTracker reflecting tiles on board so far.
        gaddag_root     : Root node of the GADDAG for move generation.
        model_equities  : Model-predicted equity for each move (parallel to moves).
        max_samples     : Opponent rack samples when not exhaustive.
        time_limit      : Hard time cap in seconds; remaining moves fall back to model equity.
        opp_rack_size   : Assumed opponent rack size (7 unless near end).
        exhaustive_threshold : Total unseen tiles at or below which we enumerate exactly.
    """
    t0 = time.perf_counter()
    results = []

    for move, model_eq in zip(moves, model_equities):
        if time.perf_counter() - t0 > time_limit:
            results.append((move, model_eq))
            continue

        board_after = apply_move(board, move)
        tiles_used, leave = compute_leave(move, our_rack, board)

        # Unseen tiles from opponent's perspective: tracker minus our leave
        unseen = tracker.unseen_for_player(leave)
        # Also remove the tiles we just placed (they're now on the board)
        for t in tiles_used:
            key = t if t == '?' else t.upper()
            if unseen.get(key, 0) > 0:
                unseen[key] -= 1
        unseen = {t: c for t, c in unseen.items() if c > 0}

        total_unseen = sum(unseen.values())
        rack_size = min(opp_rack_size, total_unseen)

        if rack_size == 0:
            # No tiles left for opponent — lookahead equity is just our score
            results.append((move, model_eq))
            continue

        if total_unseen <= exhaustive_threshold:
            opp_racks = _enumerate_racks(unseen, rack_size)
            coverage = 1.0
        else:
            opp_racks = _sample_racks(unseen, rack_size, max_samples)
            try:
                total_possible = comb(total_unseen, rack_size)
            except Exception:
                total_possible = max_samples * 10000
            coverage = min(1.0, max_samples / total_possible)

        opp_scores = [_best_greedy_score(board_after, gaddag_root, r) for r in opp_racks]
        expected_opp = sum(opp_scores) / len(opp_scores)
        lookahead_eq = move[1] - expected_opp

        final_eq = (1.0 - coverage) * model_eq + coverage * lookahead_eq
        results.append((move, final_eq))

    return sorted(results, key=lambda x: -x[1])
