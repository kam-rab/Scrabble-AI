"""Scrabble move generator."""


from gaddag.node import CONCAT, GADDAGNode
from .board import Board
from typing import List, Optional, Tuple

ALL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def generate_moves(
    board: Board,
    root: GADDAGNode,
    rack: List[str],
) -> List[Tuple[str, int, Tuple[int, int], str]]:
    """Return all unique legal moves sorted by score descending."""
    raw: list = []
    checked: set = set()
    for anchor in board.get_anchors():
        find_words_at_anchor(board, root, rack, anchor, raw, checked)

    # ── Dedup (comment out to skip for performance testing) ──
    # seen: set = set()
    # unique: list = []
    # for word, score, pos, direction in raw:
    #     key = (word, pos[0], pos[1], direction)
    #     if key not in seen:
    #         seen.add(key)
    #         unique.append((word, score, pos, direction))

    raw.sort(key=lambda x: -x[1])
    return raw


def cross_check(
    board: Board,
    root: GADDAGNode,
    anchor: Tuple[int, int],
    anchor_letter: str,
    direction: str,
    is_blank: bool = False,
) -> int:
    """Return score of cross-word formed by anchor_letter, or -1 if invalid."""
    from .board import LETTER_VALUES, Premium
    dr, dc = (1, 0) if direction == 'V' else (0, 1)
    r, c = anchor

    # No perpendicular tiles — any letter is valid, no cross-word score
    has_before = board.in_bounds(r - dr, c - dc) and board.is_occupied(r - dr, c - dc)
    has_after  = board.in_bounds(r + dr, c + dc) and board.is_occupied(r + dr, c + dc)
    if not has_before and not has_after:
        return 0

    if anchor_letter not in root.edges:
        return -1
    node = root.edges[anchor_letter]

    base = 0 if is_blank else LETTER_VALUES.get(anchor_letter, 0)
    premium = board.premium_at(r, c)
    if premium == Premium.DL:
        letter_score = base * 2
        word_multiplier = 1
    elif premium == Premium.TL:
        letter_score = base * 3
        word_multiplier = 1
    elif premium == Premium.DW:
        letter_score = base
        word_multiplier = 2
    elif premium == Premium.TW:
        letter_score = base
        word_multiplier = 3
    else:
        letter_score = base
        word_multiplier = 1
    score = letter_score

    # Follow consecutive filled letters left/up from anchor
    nr, nc = r - dr, c - dc
    while board.in_bounds(nr, nc) and board.is_occupied(nr, nc):
        tile = board.tile_at(nr, nc)
        letter = tile.letter
        if letter not in node.edges:
            return -1
        node = node.edges[letter]
        score += 0 if tile.is_blank else LETTER_VALUES.get(letter, 0)
        nr -= dr
        nc -= dc

    # Cross CONCAT arc
    if CONCAT not in node.edges:
        return -1
    node = node.edges[CONCAT]

    # Follow consecutive filled letters right/down from anchor
    nr, nc = r + dr, c + dc
    while board.in_bounds(nr, nc) and board.is_occupied(nr, nc):
        tile = board.tile_at(nr, nc)
        letter = tile.letter
        if letter not in node.edges:
            return -1
        node = node.edges[letter]
        score += 0 if tile.is_blank else LETTER_VALUES.get(letter, 0)
        nr += dr
        nc += dc

    if not node.terminal:
        return -1
    return score * word_multiplier


def find_words_at_anchor(
    board: Board,
    root: GADDAGNode,
    rack: List[str],
    anchor: Tuple[int, int],
    results: list,
    checked: set,
) -> None:
    """Find all words playable at anchorO and append them to results."""
    r, c = anchor
    singles = []
    initial_rack_len = len(rack)

    # Horizontal: find rightmost consecutive filled letter from anchor
    nc = c + 1
    while board.in_bounds(r, nc) and board.is_occupied(r, nc):
        nc += 1
    rightmost_h = (r, nc - 1) if nc - 1 >= c else (r, c)
    extend(board, root, root, rack, rightmost_h, rightmost_h, True, 'H', results, '', 0, 1, 0, singles, initial_rack_len, checked)

    # Vertical: find downmost consecutive filled letter from anchor
    nr = r + 1
    while board.in_bounds(nr, c) and board.is_occupied(nr, c):
        nr += 1
    downmost_v = (nr - 1, c) if nr - 1 >= r else (r, c)
    extend(board, root, root, rack, downmost_v, downmost_v, True, 'V', results, '', 0, 1, 0, singles, initial_rack_len, checked)
    checked.add((r, c))

    seen = set()
    for entry in singles:
        single = entry[0][-1 - max(entry[2][0] - r, entry[2][1] - c)]
        if single not in seen:
            seen.add(single)
            results.append(entry)


def extend(
    board: Board,
    root: GADDAGNode,
    node: GADDAGNode,
    rack: List[str],
    pos: Tuple[int, int],
    rightmost: Tuple[int, int],
    backward: bool,
    direction: str,
    results: dict,
    current_word: str,
    current_score: int,
    word_multiplier: int,
    unaffected: int,
    singles: Optional[list ] = None,
    initial_rack_len: int = 0,
    checked: Optional[set ] = None,
) -> None:
    """Recursively extend a word placement through the GADDAG."""
    from .board import LETTER_VALUES, Premium
    dr, dc = (1, 0) if direction == 'V' else (0, 1)
    if backward: dr, dc = -dr, -dc
    r, c = pos

    if board.in_bounds(r, c):
        new_rightmost = rightmost
        if not backward: new_rightmost = (rightmost[0] + dr, rightmost[1] + dc)
        if board.is_occupied(r, c):
            tile = board.tile_at(r, c)
            letter = tile.letter
            if letter in node.edges:
                letter_score = 0 if tile.is_blank else LETTER_VALUES.get(letter, 0)
                new_word = (letter + current_word) if backward else (current_word + letter)
                extend(board, root, node.edges[letter], rack, (r + dr, c + dc), new_rightmost, backward, direction, results,
                       new_word, current_score + letter_score, word_multiplier, unaffected, singles, initial_rack_len, checked)
            return
        else:
            # Empty square: try each rack letter present in GADDAG
            if checked is None or (r, c) not in checked:
                premium = board.premium_at(r, c)
                perp_direction = 'V' if direction == 'H' else 'H'
                seen_rack: set = set()
                for i, L in enumerate(rack):
                    if L in seen_rack:
                        continue
                    seen_rack.add(L)
                    candidates = list(ALL_LETTERS) if L == '?' else [L]
                    is_blank = L == '?'
                    remaining = rack[:i] + rack[i+1:]
                    for letter in candidates:
                        if letter not in node.edges:
                            continue
                        base_score = 0 if is_blank else LETTER_VALUES.get(letter, 0)
                        if premium == Premium.DL:
                            letter_score = base_score * 2
                            wm = word_multiplier
                        elif premium == Premium.TL:
                            letter_score = base_score * 3
                            wm = word_multiplier
                        elif premium == Premium.DW:
                            letter_score = base_score
                            wm = word_multiplier * 2
                        elif premium == Premium.TW:
                            letter_score = base_score
                            wm = word_multiplier * 3
                        else:
                            letter_score = base_score
                            wm = word_multiplier
                        cross_score = cross_check(board, root, (r, c), letter, perp_direction, is_blank)
                        if cross_score == -1:
                            continue
                        new_unaffected = unaffected + cross_score
                        placed = letter.lower() if is_blank else letter
                        new_word = (placed + current_word) if backward else (current_word + placed)
                        extend(board, root, node.edges[letter], remaining, (r + dr, c + dc), new_rightmost, backward, direction, results,
                               new_word, current_score + letter_score, wm, new_unaffected, singles, initial_rack_len, checked)

    if initial_rack_len != len(rack):
        # If still moving backward, check for CONCAT to switch direction
        if backward:
            if CONCAT in node.edges:
                rr, rc = rightmost
                extend(board, root, node.edges[CONCAT], rack, (rr - dr, rc - dc), rightmost, False, direction, results,
                    current_word, current_score, word_multiplier, unaffected, singles, initial_rack_len, checked)

        # Check if current GADDAG state is a complete word
        else:
            if node.terminal:
                score = current_score * word_multiplier + unaffected
                if initial_rack_len == 7 and len(rack) == 0:
                    score += 40
                if len(rack) == initial_rack_len - 1:
                    singles.append((current_word, score, rightmost, direction))
                else:
                    results.append((current_word, score, rightmost, direction))
