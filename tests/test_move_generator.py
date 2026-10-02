"""Tests for move_generator.find_words_at_anchor."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from gaddag import GADDAG
from solver.board import Board
from solver.move_generator import find_words_at_anchor

GADDAG_PATH = os.path.join(ROOT, 'data', 'gaddag.pkl')


def load_gaddag():
    print("Loading GADDAG...")
    g = GADDAG.load(GADDAG_PATH)
    print(f"Done. ({g.word_count:,} words)\n")
    return g


def run(label, board, gaddag, rack, anchor):
    results = []
    find_words_at_anchor(board, gaddag.root, rack, anchor, results)
    results.sort(key=lambda x: -x[1])
    print(f"--- {label} ---")
    print(f"Rack: {rack}  Anchor: {anchor}")
    print(f"Found {len(results)} move(s):")
    for word, score, pos, direction in results[:20]:
        print(f"  {word:<15} score={score:3d}  pos={pos}  dir={direction}")
    if len(results) > 20:
        print(f"  ... ({len(results) - 20} more)")
    print()


def test_empty_board(gaddag):
    """First move: rack SEATING, anchor at center (7,7)."""
    board = Board()
    rack = list('SEATING')
    run("Empty board — SEATING", board, gaddag, rack, (7, 7))


def test_extend_existing_word(gaddag):
    """CARE on board horizontally; rack has S and other letters."""
    board = Board()
    board.place_word('CARE', 7, 7, 'H')
    rack = ['S', 'T', 'I', 'N', 'G']
    # Anchor just right of CARE (7, 11)
    run("Extend CARE right", board, gaddag, rack, (7, 11))
    # Anchor just left of CARE (7, 6)
    run("Extend CARE left", board, gaddag, rack, (7, 6))


def test_cross_word(gaddag):
    """STONE vertical; rack can form word crossing it horizontally."""
    board = Board()
    board.place_word('STONE', 5, 7, 'V')
    rack = ['A', 'R', 'E', 'D']
    # Anchor left of T (row 6, col 6)
    run("Cross STONE horizontally at T", board, gaddag, rack, (6, 6))


def test_blank_tile(gaddag):
    """Blank tile in rack — should expand to all letters."""
    board = Board()
    board.place_word('RATE', 7, 7, 'H')
    rack = ['?', 'I', 'N', 'G']
    run("Blank tile — extend RATE", board, gaddag, rack, (7, 11))


def test_short_rack(gaddag):
    """Rack with fewer than 6 letters — singles path not used."""
    board = Board()
    board.place_word('CAT', 7, 7, 'H')
    rack = ['S', 'E']
    run("Short rack (2 letters)", board, gaddag, rack, (7, 10))
    run("Short rack — left of CAT", board, gaddag, rack, (7, 6))


if __name__ == '__main__':
    gaddag = load_gaddag()
    test_empty_board(gaddag)
    test_extend_existing_word(gaddag)
    test_cross_word(gaddag)
    test_blank_tile(gaddag)
    test_short_rack(gaddag)
