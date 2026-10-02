"""Batch inference wrapper for the policy network.

Takes the top-N candidate moves (already sorted by raw score), extracts
features for each in parallel, runs a single batched forward pass, and
returns moves ranked by predicted discounted score margin.
"""


import numpy as np
import torch

from solver.board import Board
from ai.tile_tracker import TileTracker
from ai.model.features import extract, compute_leave
from ai.model.policy_net import HybridPolicyNet
from typing import List, Tuple


class PolicyInference:

    def __init__(self, model: HybridPolicyNet, device: str = 'cpu') -> None:
        self.model  = model.to(device)
        self.device = device
        self.model.eval()

    @staticmethod
    def _select_candidates(
        moves: List[tuple],
        max_words: int = 15,
        max_variants: int = 3,
        max_total: int = 30,
    ) -> List[tuple]:
        """Filter moves to max_variants placements per unique word (normalized
        to uppercase so blank variants of the same word are grouped together),
        up to max_words unique words and max_total candidates overall.
        Original moves are kept as-is, preserving lowercase blank positions.
        Moves must be pre-sorted by raw score descending."""
        word_counts: dict = {}
        candidates: List[tuple] = []
        for move in moves:
            if len(candidates) >= max_total:
                break
            canonical = move[0].upper()
            if canonical not in word_counts:
                if len(word_counts) >= max_words:
                    continue
                word_counts[canonical] = 0
            if word_counts[canonical] < max_variants:
                word_counts[canonical] += 1
                candidates.append(move)
        return candidates

    def rank_moves(
        self,
        moves: List[tuple],
        board: Board,
        rack: List[str],
        tile_tracker: TileTracker,
        our_score: int,
        opp_score: int,
        turn_number: int,
        top_n: int = 15,
        gaddag_root=None,
        lookahead_samples: int = 5,
        lookahead_time: float = 1.0,
    ) -> List[Tuple[tuple, float]]:
        """Rank candidate moves by predicted discounted score margin.

        If gaddag_root is provided, applies a 1-step expectiminimax lookahead
        on the top 10 candidates and blends the result with the model equity
        weighted by search coverage.

        Args:
            moves            : All legal moves sorted by raw score descending.
            board            : Current board state (before any move).
            rack             : Current player's full rack.
            tile_tracker     : Current tile tracker.
            our_score        : Player's cumulative score before this move.
            opp_score        : Opponent's cumulative score.
            turn_number      : 0-indexed turn number.
            top_n            : Unused — kept for call-site compatibility.
            gaddag_root      : If provided, enables 1-step lookahead.
            lookahead_samples: Opponent rack samples per candidate (mid-game).
            lookahead_time   : Hard time cap for lookahead in seconds.

        Returns:
            List of (move, score) tuples sorted by predicted margin descending.
        """
        candidates = self._select_candidates(moves)
        if not candidates:
            return []

        board_tensors = []
        feature_vecs  = []

        for move in candidates:
            _, leave = compute_leave(move, rack, board)
            bt, fv   = extract(board, move, leave, tile_tracker,
                               our_score, opp_score, turn_number)
            board_tensors.append(bt)
            feature_vecs.append(fv)

        board_batch = torch.tensor(
            np.stack(board_tensors), dtype=torch.float32
        ).to(self.device)
        feat_batch = torch.tensor(
            np.stack(feature_vecs), dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            score_margins = self.model(board_batch, feat_batch)

        score_margins = score_margins.squeeze(1).cpu().numpy().tolist()

        if gaddag_root is not None:
            from ai.lookahead import one_step_lookahead
            top10 = candidates[:10]
            top10_equities = score_margins[:10]
            lookahead_ranked = one_step_lookahead(
                top10, board, rack, tile_tracker, gaddag_root,
                top10_equities,
                max_samples=lookahead_samples,
                time_limit=lookahead_time,
            )
            rest = sorted(
                zip(candidates[10:], score_margins[10:]),
                key=lambda x: -x[1],
            )
            return list(lookahead_ranked) + list(rest)

        return sorted(zip(candidates, score_margins), key=lambda x: -x[1])
