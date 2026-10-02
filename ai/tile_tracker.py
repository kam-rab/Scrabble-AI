"""Tracks tiles remaining in the game (public knowledge).

Tiles not yet placed on the board are "remaining" — they live in racks or the
bag.  From any player's perspective, remaining() minus their own rack gives the
set of tiles they cannot see (opponent rack + bag).
"""


import numpy as np
from typing import Dict, List

TILE_DISTRIBUTION: Dict[str, int] = {
    'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
    'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2,  'N': 6, 'O': 8, 'P': 2,
    'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4,  'V': 2, 'W': 2, 'X': 1,
    'Y': 2, 'Z': 1, '?': 2,
}

# Canonical ordering for vector representation: A–Z (indices 0–25), blank (26)
TILE_ORDER: List[str] = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['?']
TILE_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TILE_ORDER)}

_INITIAL_COUNTS = np.array(
    [TILE_DISTRIBUTION.get(t, 0) for t in TILE_ORDER], dtype=np.float32
)


class TileTracker:
    """Tracks tiles not yet placed on the board.

    Call remove() each time a tile is placed on the board.
    Call add() to undo (testing / rollback).

    remaining() returns tiles still "in play" (racks + bag).
    unseen_for_player() subtracts the caller's own rack to give the unknown
    pool (opponent rack + bag).
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = dict(TILE_DISTRIBUTION)

    def remove(self, tile: str) -> None:
        """Mark a tile as placed on the board."""
        key = tile if tile == '?' else tile.upper()
        if self._counts.get(key, 0) > 0:
            self._counts[key] -= 1

    def add(self, tile: str) -> None:
        """Undo a removal (for testing / rollback)."""
        key = tile if tile == '?' else tile.upper()
        self._counts[key] = self._counts.get(key, 0) + 1

    def remaining(self) -> Dict[str, int]:
        """All tiles not yet placed on the board (racks + bag)."""
        return {t: c for t, c in self._counts.items() if c > 0}

    def unseen_for_player(self, our_rack: List[str]) -> Dict[str, int]:
        """remaining() minus our own rack — what opponent + bag holds."""
        unseen = dict(self._counts)
        for tile in our_rack:
            key = tile if tile == '?' else tile.upper()
            unseen[key] = max(0, unseen.get(key, 0) - 1)
        return {t: c for t, c in unseen.items() if c > 0}

    def total_remaining(self) -> int:
        """Total tile count not yet on the board."""
        return sum(self._counts.values())

    def to_vector(self) -> np.ndarray:
        """27-dim array of remaining tile counts, normalized by initial counts."""
        counts = np.array(
            [self._counts.get(t, 0) for t in TILE_ORDER], dtype=np.float32
        )
        return counts / (_INITIAL_COUNTS + 1e-8)
