"""GADDAG node representation."""

# The concatenation arc character separating the reversed prefix from the suffix.
# Every encoded path passes through this arc exactly once.
CONCAT = '+'


class GADDAGNode:
    """A single node in the GADDAG trie.

    Each node stores outgoing edges keyed by character (A–Z or CONCAT)
    and a flag indicating whether this node marks the end of a valid
    word encoding.

    Using __slots__ reduces per-instance memory overhead significantly
    when millions of nodes are alive at the same time.
    """

    __slots__ = ('edges', 'terminal')

    def __init__(self) -> None:
        self.edges: dict[str, GADDAGNode] = {}
        self.terminal: bool = False
