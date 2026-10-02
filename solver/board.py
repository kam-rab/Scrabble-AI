"""Scrabble board representation.

Provides:
  - LETTER_VALUES  – standard tile point values
  - Premium        – enum for premium-square types
  - PREMIUM_MAP    – (row, col) → Premium lookup for the standard 15×15 board
  - Tile           – immutable tile placed on the board
  - Board          – 15×15 grid with anchor detection and display
"""


from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Letter point values
# ---------------------------------------------------------------------------

LETTER_VALUES: Dict[str, int] = {
    'A': 1,  'B': 4,  'C': 3,  'D': 2,  'E': 1,
    'F': 4,  'G': 4,  'H': 3,  'I': 1,  'J': 10,
    'K': 6,  'L': 2,  'M': 3,  'N': 1,  'O': 1,
    'P': 3,  'Q': 10, 'R': 1,  'S': 1,  'T': 1,
    'U': 2,  'V': 6,  'W': 5,  'X': 8,  'Y': 4,
    'Z': 10, '?': 0,  # blank tile (displayed as chosen letter, worth 0)
}

# ---------------------------------------------------------------------------
# Premium squares
# ---------------------------------------------------------------------------

class Premium(Enum):
    """Type of premium square."""
    NONE = 0
    DL   = 1   # Double Letter Score
    TL   = 2   # Triple Letter Score
    DW   = 3   # Double Word Score  (center star is DW)
    TW   = 4   # Triple Word Score


def _build_premium_map() -> Dict[Tuple[int, int], Premium]:
    """Return the standard Scrabble board premium-square map (0-indexed)."""
    m: Dict[Tuple[int, int], Premium] = {}

    # Triple Word – 8 squares (corners + mid-edges)
    for r, c in [
        (0,3), (0,11), (3,0), (3,14), (11,0), (11,14), (14,3), (14,11)
    ]:
        m[(r, c)] = Premium.TW

    # Double Word – 17 squares (two diagonals + center star)
    for r, c in [
        (1,1), (1,13), (3,7), (7,3), (7,11), (11,7), (13,1), (13,13)                                   
    ]:
        m[(r, c)] = Premium.DW

    # Triple Letter – 12 squares
    for r, c in [
        (0,0), (0,14), (1,6), (1,8), (4,5), (4,9), (5,4), (5,10), (6,1), (6,13), (8,1), (8,13), (9,4), (9,10), (10,5), (10,9), (13,6), (13,8), (14,0), (14,14)
    ]:
        m[(r, c)] = Premium.TL

    # Double Letter – 24 squares
    for r, c in [
        (0,7), (2,4), (2,10), (3,3), (3,11), (4,2), (4,12), (5,7), (7,0), (7,5), (7,9), (7,14), (9,7), (10,2), (10,12), (11,3), (11,11), (12,4), (12,10), (14,7)
    ]:
        m[(r, c)] = Premium.DL

    return m


PREMIUM_MAP: Dict[Tuple[int, int], Premium] = _build_premium_map()

# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------

class Tile(NamedTuple):
    """An immutable tile placed on the board.

    Attributes:
        letter:   Uppercase letter this tile displays (A–Z).
        is_blank: True when this tile is a blank (worth 0 points regardless
                  of which letter it was assigned).
    """
    letter: str
    is_blank: bool = False

    def face_value(self) -> int:
        """Point value of this tile (always 0 for a blank)."""
        return 0 if self.is_blank else LETTER_VALUES.get(self.letter, 0)

    def __str__(self) -> str:
        # Blanks displayed in lowercase to distinguish them visually.
        return self.letter.lower() if self.is_blank else self.letter

# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))   # N, S, W, E

class Board:
    """15*15 Scrabble board.

    Cells are indexed ``board.grid[row][col]``, row 0 at the top, col 0 at
    the left.  An empty cell holds ``None``; an occupied cell holds a
    :class:`Tile`.

    Premium squares are *not* consumed by placement — the scorer reads the
    premium map and applies bonuses only for tiles placed on that turn.
    """

    SIZE   = 15
    CENTER = (7, 7)

    def __init__(self) -> None:
        self.grid: Optional[List[List[Tile ]]] = [
            [None] * self.SIZE for _ in range(self.SIZE)
        ]

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.SIZE and 0 <= col < self.SIZE

    def is_empty(self, row: int, col: int) -> bool:
        return self.grid[row][col] is None

    def is_occupied(self, row: int, col: int) -> bool:
        return self.grid[row][col] is not None

    def tile_at(self, row: int, col: int) -> Optional[Tile ]:
        """Return the tile at (row, col), or None if empty."""
        return self.grid[row][col]

    def letter_at(self, row: int, col: int) -> Optional[str ]:
        """Return the letter character at (row, col), or None if empty."""
        tile = self.grid[row][col]
        return tile.letter if tile else None

    def premium_at(self, row: int, col: int) -> Premium:
        """Return the premium-square type at (row, col)."""
        return PREMIUM_MAP.get((row, col), Premium.NONE)

    # ------------------------------------------------------------------
    # Placing tiles
    # ------------------------------------------------------------------

    def place_tile(self, row: int, col: int, letter: str,
                   is_blank: bool = False) -> None:
        """Place a single tile on the board (no legality checking)."""
        if not self.in_bounds(row, col):
            raise ValueError(f"Position ({row}, {col}) is out of bounds.")
        self.grid[row][col] = Tile(letter.upper(), is_blank)

    def place_word(self, word: str, row: int, col: int, direction: str,
                   blanks: Optional[Set[int] ] = None) -> None:
        """Place an entire word — convenience helper for tests and setup.

        Args:
            word:      Word to place (case-insensitive).
            row, col:  Starting cell (top-left of the word).
            direction: ``'H'`` (horizontal / across) or ``'V'`` (vertical / down).
            blanks:    0-based indices within *word* that are blank tiles.
        """
        blanks = blanks or set()
        dr, dc = (0, 1) if direction.upper() == 'H' else (1, 0)
        for i, ch in enumerate(word.upper()):
            self.place_tile(row + i * dr, col + i * dc, ch, is_blank=(i in blanks))

    # ------------------------------------------------------------------
    # Anchor detection
    # ------------------------------------------------------------------

    def get_anchors(self) -> Set[Tuple[int, int]]:
        """Return the set of *anchor squares*.

        An anchor square is any empty cell that is orthogonally adjacent to
        at least one occupied cell.  The move generator only needs to consider
        these squares as starting points.

        On a completely empty board the center square (7, 7) is returned as
        the sole anchor so that the first move can always be generated.
        """
        anchors: Set[Tuple[int, int]] = set()

        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.is_occupied(r, c):
                    continue
                for dr, dc in _DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc) and self.is_occupied(nr, nc):
                        anchors.add((r, c))
                        break     # no need to check other neighbours

        if not anchors:           # empty board – first move must cross center
            anchors.add(self.CENTER)

        return anchors

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def occupied_count(self) -> int:
        """Number of tiles currently on the board."""
        return sum(
            1
            for r in range(self.SIZE)
            for c in range(self.SIZE)
            if self.is_occupied(r, c)
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self, mark_anchors: bool = False) -> str:
        """Return a printable board string.

        Every cell is exactly 3 characters wide so the grid lines up cleanly.
        If *mark_anchors* is True, empty anchor squares are shown as ``'** '``
        instead of their premium glyph.
        """
        anchor_set = self.get_anchors() if mark_anchors else set()

        def cell_str(r: int, c: int) -> str:
            tile = self.grid[r][c]
            if tile is not None:
                ch = tile.letter.lower() if tile.is_blank else tile.letter
                return f' {ch} '                        # ' A ' or ' a '
            if (r, c) == self.CENTER:
                return ' ★ '
            if (r, c) in anchor_set:
                return '** '
            p = self.premium_at(r, c)
            return ' · ' if p == Premium.NONE else f'{p.name} '  # 'TW ', 'DL ', …

        col_header = '     ' + ''.join(f'{c:^4}' for c in range(self.SIZE))
        divider    = '    +' + '---+' * self.SIZE

        lines = [col_header, divider]
        for r in range(self.SIZE):
            cells = '|'.join(cell_str(r, c) for c in range(self.SIZE))
            lines.append(f'{r:2d}  |{cells}|')
            lines.append(divider)

        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.display()
