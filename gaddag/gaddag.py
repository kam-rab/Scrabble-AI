"""GADDAG data structure for Scrabble word lookup and move generation.

The GADDAG (Generalized Directed Acyclic Graph) was described by
Steven A. Gordon (1994) in "A Faster Scrabble Move Generation Algorithm,"
Software—Practice and Experience 24(2): 219–232.

Encoding
--------
For a word  w = w[0] w[1] ... w[n-1]  the GADDAG stores n distinct paths,
one for each possible *anchor* position k (0 ≤ k < n):

    path_k = w[k] w[k-1] ... w[0]  +  w[k+1] ... w[n-1]
             |_____reversed prefix_|  |_______suffix______|

The '+' character (CONCAT) marks the pivot: everything to the left is the
reversed sequence of letters that lie to the LEFT of the anchor on the
board; everything to the right is what extends to the RIGHT.

During move generation the solver:
  1. Starts at the root and follows the anchor letter already on the board.
  2. Extends LEFT (backwards) through unused rack tiles, following the
     reversed-prefix portion of the trie.
  3. Crosses the CONCAT arc to switch direction.
  4. Extends RIGHT through rack tiles and any existing board tiles,
     following the suffix portion of the trie.
  5. Records every terminal node reached as a valid word placement.
"""

import os
import pickle

from .node import CONCAT, GADDAGNode


class GADDAG:
    """The GADDAG trie built from a Scrabble word list.

    Typical usage
    -------------
    Build once and cache:
        gaddag = GADDAG()
        gaddag.build_from_file('data/wordlist.txt')
        gaddag.save('data/gaddag.pkl')

    Load on subsequent runs:
        gaddag = GADDAG.load('data/gaddag.pkl')
    """

    def __init__(self) -> None:
        self.root = GADDAGNode()
        self._word_count = 0

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def insert(self, word: str) -> None:
        """Insert *word* into the GADDAG.

        Adds all n encoded paths for an n-letter word.  Words shorter
        than 2 letters or containing non-alpha characters are silently
        skipped (Scrabble does not use them).
        """
        word = word.upper().strip()
        n = len(word)
        if n < 2 or not word.isalpha():
            return

        for k in range(n):
            node = self.root

            # --- reversed prefix:  w[k], w[k-1], ..., w[0] ---
            for j in range(k, -1, -1):
                ch = word[j]
                if ch not in node.edges:
                    node.edges[ch] = GADDAGNode()
                node = node.edges[ch]

            # --- concatenation arc '+' ---
            if CONCAT not in node.edges:
                node.edges[CONCAT] = GADDAGNode()
            node = node.edges[CONCAT]

            # --- suffix:  w[k+1], w[k+2], ..., w[n-1] ---
            for j in range(k + 1, n):
                ch = word[j]
                if ch not in node.edges:
                    node.edges[ch] = GADDAGNode()
                node = node.edges[ch]

            # Mark end of this encoding as terminal
            node.terminal = True

        self._word_count += 1

    def build_from_file(self, filepath: str) -> int:
        """Read words from *filepath* (one word per line) and insert them all.

        Lines that are not purely alphabetic or fall outside the 2–15 letter
        Scrabble range are skipped.  Returns the number of words inserted.
        """
        count = 0
        with open(filepath, 'r', encoding='utf-8') as fh:
            for line in fh:
                word = line.strip()
                if word and word.isalpha() and 2 <= len(word) <= 15:
                    self.insert(word)
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def contains(self, word: str) -> bool:
        """Return True if *word* is in the dictionary.

        Uses the k=0 encoding  w[0] + w[1] ... w[n-1], which is identical
        to a standard forward trie lookup with a leading CONCAT arc.
        """
        word = word.upper()
        if len(word) < 2:
            return False

        node = self.root

        # First letter
        if word[0] not in node.edges:
            return False
        node = node.edges[word[0]]

        # Concatenation arc
        if CONCAT not in node.edges:
            return False
        node = node.edges[CONCAT]

        # Remaining letters
        for ch in word[1:]:
            if ch not in node.edges:
                return False
            node = node.edges[ch]

        return node.terminal

    def get_node(self, path: str):
        """Return the node reached by following *path* from the root.

        *path* is a sequence of characters (uppercase A–Z or CONCAT '+').
        Returns None if the path does not exist in the trie.

        This method is the primary interface for the move-generation solver:
        the solver pre-computes partial paths and then continues the traversal
        character by character.
        """
        node = self.root
        for ch in path:
            ch = ch if ch == CONCAT else ch.upper()
            if ch not in node.edges:
                return None
            node = node.edges[ch]
        return node

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Pickle the entire GADDAG to *filepath*.

        Uses the highest available pickle protocol for compact output.
        The parent directory is created automatically if it does not exist.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath: str):
        """Load and return a GADDAG previously saved with :meth:`save`."""
        with open(filepath, 'rb') as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # Properties / dunder
    # ------------------------------------------------------------------

    @property
    def word_count(self) -> int:
        """Number of words inserted into the GADDAG."""
        return self._word_count

    def __repr__(self) -> str:
        return f"GADDAG(words={self._word_count:,})"
