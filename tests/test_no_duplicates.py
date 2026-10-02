"""Tests that generate_moves produces no duplicate moves and covers expected moves."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from gaddag import GADDAG
from solver.board import Board
from solver.move_generator import generate_moves

GADDAG_PATH = os.path.join(ROOT, 'data', 'gaddag.pkl')


def load_gaddag():
    print("Loading GADDAG...")
    g = GADDAG.load(GADDAG_PATH)
    print(f"Done. ({g.word_count:,} words)\n")
    return g


def assert_no_duplicates(label, moves):
    keys = [(w, pos[0], pos[1], d) for w, _, pos, d in moves]
    seen = set()
    dupes = []
    for k in keys:
        if k in seen:
            dupes.append(k)
        seen.add(k)
    if dupes:
        print(f"FAIL [{label}]: {len(dupes)} duplicate(s):")
        for d in dupes[:10]:
            print(f"  {d}")
    else:
        print(f"PASS [{label}]: {len(moves)} moves, no duplicates")


def assert_move_present(label, moves, word, direction):
    found = any(w == word and d == direction for w, _, _, d in moves)
    if found:
        print(f"PASS [{label}]: '{word}' ({direction}) found")
    else:
        print(f"FAIL [{label}]: '{word}' ({direction}) not found")


def test_empty_board(gaddag):
    board = Board()
    moves = generate_moves(board, gaddag.root, list('SEATING'))
    assert_no_duplicates("empty board", moves)
    assert_move_present("empty board", moves, "SEATING", "H")
    assert_move_present("empty board", moves, "SEATING", "V")


def test_single_word_on_board(gaddag):
    """CARE on board — multiple anchors, check no dupes across all of them."""
    board = Board()
    board.place_word('CARE', 7, 7, 'H')
    moves = generate_moves(board, gaddag.root, list('STING'))
    assert_no_duplicates("CARE on board", moves)


def test_two_intersecting_words(gaddag):
    """STONE horizontal, RATES vertical crossing at R — dense anchor set."""
    board = Board()
    board.place_word('STONE', 7, 5, 'H')
    board.place_word('RATES', 5, 7, 'V')
    moves = generate_moves(board, gaddag.root, list('AEIOUD'))
    assert_no_duplicates("two intersecting words", moves)


def test_single_letter_placement(gaddag):
    """Placing one tile to form two words — singles dedup check."""
    board = Board()
    board.place_word('CARE', 7, 7, 'H')
    board.place_word('STONE', 5, 9, 'V')
    moves = generate_moves(board, gaddag.root, list('AEIOU'))
    assert_no_duplicates("single-letter cross placement", moves)


def test_blank_tile(gaddag):
    """Blank in rack — many candidate letters, check no dupes."""
    board = Board()
    board.place_word('RATES', 7, 5, 'H')
    moves = generate_moves(board, gaddag.root, list('?ING'))
    assert_no_duplicates("blank tile", moves)


def test_two_blanks(gaddag):
    """Two blanks — most likely source of duplicates without dedup."""
    board = Board()
    board.place_word('RATES', 7, 5, 'H')
    moves = generate_moves(board, gaddag.root, list('??ING'))
    assert_no_duplicates("two blanks", moves)


def test_full_rack_bingo(gaddag):
    """Full 7-letter rack — bingo possible, check no dupes."""
    board = Board()
    board.place_word('CARE', 7, 7, 'H')
    moves = generate_moves(board, gaddag.root, list('SEATING'))
    assert_no_duplicates("full rack bingo", moves)


def test_dense_board(gaddag):
    """Several words on board creating many anchors."""
    board = Board()
    board.place_word('CARE',   7,  7, 'H')
    board.place_word('STONE',  5,  9, 'V')
    board.place_word('RATES',  9,  5, 'H')
    board.place_word('BLIND',  7,  5, 'V')
    moves = generate_moves(board, gaddag.root, list('AEIOURT'))
    assert_no_duplicates("dense board", moves)
    print(f"  ({len(moves)} total moves)")


if __name__ == '__main__':
    gaddag = load_gaddag()
    test_empty_board(gaddag)
    test_single_word_on_board(gaddag)
    test_two_intersecting_words(gaddag)
    test_single_letter_placement(gaddag)
    test_blank_tile(gaddag)
    test_two_blanks(gaddag)
    test_full_rack_bingo(gaddag)
    test_dense_board(gaddag)
