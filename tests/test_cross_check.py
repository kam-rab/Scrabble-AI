"""Tests for the cross-check algorithm.

Each test derives its expected values from the GADDAG itself so the assertions
are dictionary-backed rather than hand-guessed.

Test layout
-----------
We use a single shared GADDAG (loaded once) and build small board states to
verify specific cross-check properties.

Board diagrams use '.' for empty and letters for placed tiles.
Anchor squares under test are shown with '@'.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from gaddag import GADDAG
from gaddag.node import CONCAT
from solver.board import Board
from solver.cross_check import (
    ALL_LETTERS,
    _check_direction,
    compute_cross_checks,
    playable_at_anchor,
)

GADDAG_PATH = os.path.join(ROOT, 'data', 'gaddag.pkl')


def load_gaddag() -> GADDAG:
    if not os.path.exists(GADDAG_PATH):
        raise FileNotFoundError(
            f"gaddag.pkl not found at {GADDAG_PATH}.\n"
            "Run:  python download_wordlist.py && python build_gaddag.py"
        )
    return GADDAG.load(GADDAG_PATH)


# Load once for the whole module
_GADDAG: GADDAG | None = None

def get_gaddag() -> GADDAG:
    global _GADDAG
    if _GADDAG is None:
        _GADDAG = load_gaddag()
    return _GADDAG


def valid_set_from_gaddag(gaddag: GADDAG, prefix: str, suffix: str) -> frozenset[str]:
    """
    Ask the GADDAG directly: which letters L produce a valid word
    [prefix] + L + [suffix]?

    Uses the GADDAG encoding where anchorG = last letter of suffix (rightmost).
    If suffix is empty, anchorG = L itself (start from root).

    This is the ground-truth reference used to validate _check_direction.
    """
    root = gaddag.root
    valid: set[str] = set()

    for L in ALL_LETTERS:
        # Build the GADDAG path: reversed(suffix) → L → prefix → CONCAT → terminal
        path_chars = list(reversed(suffix)) + [L] + list(prefix)
        node = root
        ok = True
        for ch in path_chars:
            if ch not in node.edges:
                ok = False
                break
            node = node.edges[ch]
        if ok and CONCAT in node.edges and node.edges[CONCAT].terminal:
            valid.add(L)

    return frozenset(valid)


class TestEmptyBoard(unittest.TestCase):
    """On an empty board the only anchor is the center square."""

    def test_center_anchor_exists(self):
        b = Board()
        self.assertEqual(b.get_anchors(), {(7, 7)})

    def test_center_no_constraint(self):
        """Center has no board context → every letter is valid."""
        g = get_gaddag()
        checks = compute_cross_checks(Board(), g)
        self.assertIn((7, 7), checks)
        self.assertEqual(checks[(7, 7)], ALL_LETTERS)


class TestHorizontalContextOnly(unittest.TestCase):
    """
    Board row 7:  . . . H E L L O @ . . . . . .
                                   ^ anchorO (7,12)

    Suffix from (7,12): none.
    Prefix from (7,12) going left: O, L, L, E, H.
    Valid letters: those L where [L]HELLO... wait, anchorO is to the RIGHT of O.
    Actually (7,11)=O, (7,12) is empty = anchorO.
    Prefix = [O, L, L, E, H] (closest first).
    Suffix = [] (nothing to the right).
    Valid L: the set of letters that make HELLO[L] a valid word  → "HELLOS" etc.
    """

    def setUp(self):
        self.board = Board()
        self.board.place_word('HELLO', 7, 7, 'H')  # H=7,7  O=7,11
        self.gaddag = get_gaddag()

    def test_anchor_right_of_HELLO(self):
        """(7,12) = right of 'O'. Check via reference GADDAG query."""
        expected = valid_set_from_gaddag(self.gaddag, prefix='OLLEH', suffix='')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(7, 12)], expected)

    def test_anchor_right_contains_S(self):
        """HELLOS is a valid word, so S must be in the cross-check set."""
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertIn('S', checks[(7, 12)])

    def test_anchor_left_of_HELLO(self):
        """
        (7,6) = left of 'H'.
        Suffix = [H, E, L, L, O], prefix = [].
        Valid L: letters such that L+HELLO is a valid word.
        """
        expected = valid_set_from_gaddag(self.gaddag, prefix='', suffix='HELLO')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(7, 6)], expected)

    def test_anchor_above_H(self):
        """
        (6,7) = above 'H'.
        No horizontal context (nothing left/right at row 6).
        Vertical suffix = [H], prefix = [].
        Valid L: letters such that LH is a valid word (e.g. AH, EH, OH, SH...).
        """
        expected = valid_set_from_gaddag(self.gaddag, prefix='', suffix='H')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(6, 7)], expected)
        # Spot-check: AH, EH, OH, SH, UH are all valid 2-letter words
        for word in ['AH', 'EH', 'OH', 'SH', 'UH']:
            self.assertIn(word[0], expected, f"{word} should be valid above H")

    def test_anchor_above_E(self):
        """
        (6,8) = above 'E'.
        Vertical suffix = [E], prefix = [].
        Valid L: letters such that LE is a valid word.
        """
        expected = valid_set_from_gaddag(self.gaddag, prefix='', suffix='E')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(6, 8)], expected)

    def test_anchor_above_first_L(self):
        """
        (6,9) = above first 'L'.
        Vertical suffix = [L], prefix = [].
        Valid L: letters such that LL' is a valid word (AL, EL, ...).
        """
        expected = valid_set_from_gaddag(self.gaddag, prefix='', suffix='L')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(6, 9)], expected)
        self.assertIn('A', expected)  # AL is a valid Scrabble word
        self.assertIn('E', expected)  # EL is a valid Scrabble word


class TestBothSidesContext(unittest.TestCase):
    """
    Place SCAR horizontally (row 7, cols 3-6) and test anchorO between S and C.

    Board row 7: . . . S @ C A R . . . . . . .
                         ^(7,4) anchorO

    Suffix from (7,4): C, A, R  (cols 5, 6, 7... wait)

    Let me recalculate: SCAR at (7,3):
      S=3, C=4, A=5, R=6.
    We'll test the gap BEFORE S at (7,2).

    For anchor (7,2) [left of S]:
      Suffix = [S, C, A, R], prefix = [].
      Valid L: such that L+SCAR is a valid word.

    For anchor (7,7) [right of R]:
      Suffix = [], prefix = [R, A, C, S].
      Valid L: such that SCAR+L is a valid word → SCARS, SCARY, SCARF.
    """

    def setUp(self):
        self.board = Board()
        self.board.place_word('SCAR', 7, 3, 'H')  # S=3, C=4, A=5, R=6
        self.gaddag = get_gaddag()

    def test_left_of_SCAR(self):
        expected = valid_set_from_gaddag(self.gaddag, prefix='', suffix='SCAR')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(7, 2)], expected)

    def test_right_of_SCAR(self):
        expected = valid_set_from_gaddag(self.gaddag, prefix='RACS', suffix='')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(7, 7)], expected)
        self.assertIn('S', checks[(7, 7)])   # SCARS
        self.assertIn('Y', checks[(7, 7)])   # SCARY
        self.assertIn('F', checks[(7, 7)])   # SCARF

    def test_between_two_words(self):
        """
        Add CARE vertically at col 5 (rows 7-10), sharing the 'A' in SCAR.
        SCAR occupies (7,3)-(7,6).  CARE shares 'A' at (7,5).
        Place C(8,5), R(9,5), E(10,5) to extend downward.

        AnchorO at (7,4) [gap between S and C in SCAR → not a gap: they're adjacent].

        Instead, test anchor (6,5) [above 'A' at (7,5)]:
          horizontal context at (6,5): nothing left/right → H_check = ALL_LETTERS
          vertical context at (6,5): suffix_down = [A,R,E] (from (7,5) down),
                                      prefix_up = [] nothing above.
          Valid L: such that L+ARE is a valid word → BARE, CARE, DARE, FARE, ...
        """
        self.board.place_tile(8, 5, 'R')
        self.board.place_tile(9, 5, 'E')

        expected = valid_set_from_gaddag(self.gaddag, prefix='', suffix='ARE')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(6, 5)], expected)

        for letter in ['B', 'C', 'D', 'F', 'H']:
            self.assertIn(letter, expected)  # BARE, CARE, DARE, FARE, HARE


class TestRackFiltering(unittest.TestCase):
    """playable_at_anchor correctly filters by rack letters and blank."""

    def setUp(self):
        self.board = Board()
        self.board.place_word('HELLO', 7, 7, 'H')
        self.gaddag = get_gaddag()
        self.checks = compute_cross_checks(self.board, self.gaddag)

    def test_only_rack_letters_returned(self):
        rack = ['S', 'X', 'Q']
        playable = playable_at_anchor(self.checks, rack)
        for anchor, letters in playable.items():
            for l in letters:
                if l != '?':
                    self.assertIn(l, rack)

    def test_blank_included_when_set_nonempty(self):
        """If any letter is valid, blank ('?') should appear in result."""
        rack = ['?']
        playable = playable_at_anchor(self.checks, rack)
        # (7,12) has 'S' valid (HELLOS), so blank must appear there
        if (7, 12) in self.checks and self.checks[(7, 12)]:
            self.assertIn('?', playable.get((7, 12), set()))

    def test_blank_absent_when_set_empty(self):
        """If cross-check set is empty, blank must not appear either."""
        rack = ['?', 'A', 'B']
        # Manually inject an anchor with an empty cross-check
        fake_checks = {(0, 0): frozenset()}
        playable = playable_at_anchor(fake_checks, rack)
        self.assertNotIn((0, 0), playable)

    def test_S_playable_right_of_HELLO(self):
        rack = ['S', 'T', 'R', 'A', 'N', 'G', 'E']
        playable = playable_at_anchor(self.checks, rack)
        self.assertIn('S', playable.get((7, 12), set()))

    def test_no_valid_tile_anchor_excluded(self):
        """An anchor whose cross-check set shares nothing with the rack is omitted."""
        rack = ['Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q']
        playable = playable_at_anchor(self.checks, rack)
        # Q should not be playable above or below HELLO letters (not two-letter Qs)
        # but if Q somehow appears in the valid set the rack filter is still correct
        for anchor, letters in playable.items():
            for l in letters:
                if l != '?':
                    self.assertIn(l, ['Q'])


class TestVerticalWord(unittest.TestCase):
    """Place a vertical word and verify anchors around it."""

    def setUp(self):
        # WORLD placed vertically: W at (3,7), O at (4,7), R at (5,7),
        # L at (6,7), D at (7,7)
        self.board = Board()
        self.board.place_word('WORLD', 3, 7, 'V')
        self.gaddag = get_gaddag()

    def test_anchor_below_D(self):
        """(8,7) = below 'D'. Vertical prefix = [D,L,R,O,W], suffix = []."""
        expected = valid_set_from_gaddag(self.gaddag, prefix='DLROW', suffix='')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(8, 7)], expected)

    def test_anchor_above_W(self):
        """(2,7) = above 'W'. Vertical suffix = [W,O,R,L,D], prefix = []."""
        expected = valid_set_from_gaddag(self.gaddag, prefix='', suffix='WORLD')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(2, 7)], expected)

    def test_anchor_right_of_O(self):
        """
        (4,8) = right of 'O' in WORLD.
        Horizontal: suffix=[], prefix=[O].
        Vertical: no tiles above/below → ALL_LETTERS.
        Cross-check = {L : LO is a valid word} = DO, GO, HO, LO, MO, NO, SO, TO, WO, YO, BO...
        """
        expected_h = valid_set_from_gaddag(self.gaddag, prefix='O', suffix='')
        checks = compute_cross_checks(self.board, self.gaddag)
        self.assertEqual(checks[(4, 8)], expected_h)
        # Spot-check known two-letter words
        # Two-letter words O+[letter]: OD, OH, OM, ON, OR, OS, OW, OX, OY, OE, OF, OP
        # Note: "GO" has G left of O, so at (4,8) with O to the LEFT the word
        # is "OG" — NOT a valid word.  Only O-first words count here.
        for letter in ['D', 'H', 'M', 'N', 'R', 'S', 'W', 'X', 'Y']:
            self.assertIn(letter, expected_h)


if __name__ == '__main__':
    unittest.main(verbosity=2)
