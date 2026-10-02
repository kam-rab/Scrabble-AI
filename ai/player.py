"""ScrabbleAI — main public interface.

Usage
-----
    from gaddag import GADDAG
    from solver.board import Board
    from ai.player import ScrabbleAI
    from ai.tile_tracker import TileTracker

    gaddag = GADDAG.load('data/gaddag.pkl')
    ai     = ScrabbleAI(gaddag, model_path='data/model.pt')

    tracker = TileTracker()
    board   = Board()
    rack    = ['S', 'T', 'A', 'R', 'E', 'D', '?']

    move = ai.choose_move(board, rack, tracker, our_score=0, opp_score=0, turn_number=0)
    # → ('STARRED', 76, (7, 13), 'H')  — or whatever equity-maximising word

If model_path is omitted or the file does not exist, choose_move falls back
to returning the highest-scoring legal move (greedy baseline).
"""


from pathlib import Path

import torch

from gaddag import GADDAG
from solver.board import Board
from solver.move_generator import generate_moves
from ai.tile_tracker import TileTracker
from ai.equity import rank_moves
from ai.model.policy_net import HybridPolicyNet
from ai.model.inference import PolicyInference
from typing import List, Optional, Tuple


class ScrabbleAI:

    def __init__(
        self,
        gaddag: GADDAG,
        model_path: Optional[str ] = None,
        device: str = 'cpu',
        n_candidates: int = 15,
    ) -> None:
        self.gaddag       = gaddag
        self.n_candidates = n_candidates
        self._inference: Optional[PolicyInference ] = None

        if model_path is not None and Path(model_path).exists():
            model = HybridPolicyNet()
            model.load_state_dict(torch.load(model_path, map_location=device))
            self._inference = PolicyInference(model, device=device)
            print(f'[ScrabbleAI] Loaded model from {model_path}')
        else:
            print('[ScrabbleAI] No model — using greedy (raw score) ranking')

    def choose_move(
        self,
        board: Board,
        rack: List[str],
        tile_tracker: TileTracker,
        our_score: int = 0,
        opp_score: int = 0,
        turn_number: int = 0,
    ) -> Optional[Tuple[str, int, Tuple[int, int], str] ]:
        """Return the best move tuple, or None if no legal moves exist."""
        moves = generate_moves(board, self.gaddag.root, rack)
        if not moves:
            return None

        ranked = rank_moves(
            moves, board, rack, tile_tracker,
            our_score, opp_score, turn_number,
            inference=self._inference,
            top_n=self.n_candidates,
        )
        return ranked[0][0]
