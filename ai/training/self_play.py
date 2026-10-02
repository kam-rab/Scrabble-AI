"""Self-play game loop for generating training data.

Run as a worker on any CPU node:

    python -m ai.training.self_play \
        --worker-id 0 \
        --model-path /shared/model.pt \
        --gaddag-path data/gaddag.pkl \
        --output-dir /shared/transitions/ \
        --n-games 9999 \
        --opponent greedy

Each worker writes transition .pkl files to output-dir.
The trainer process reads and deletes those files.
"""


import argparse
import pickle
import random
import time
from pathlib import Path

import torch

from gaddag import GADDAG
from solver.board import Board, LETTER_VALUES
from solver.move_generator import generate_moves
from ai.tile_tracker import TileTracker, TILE_DISTRIBUTION
from ai.model.features import compute_leave, apply_move, extract
from ai.model.policy_net import HybridPolicyNet
from ai.model.inference import PolicyInference
from typing import List, Optional


# ---------------------------------------------------------------------------
# Tile bag helpers
# ---------------------------------------------------------------------------

def make_bag() -> List[str]:
    """Return a freshly shuffled bag of all 100 Scrabble tiles."""
    tiles: List[str] = []
    for letter, count in TILE_DISTRIBUTION.items():
        tiles.extend([letter] * count)
    random.shuffle(tiles)
    return tiles


def draw_tiles(bag: List[str], rack: List[str]) -> None:
    """Draw from bag into rack until rack has 7 tiles or bag is empty."""
    while len(rack) < 7 and bag:
        rack.append(bag.pop())


def _tile_value(tile: str) -> int:
    return LETTER_VALUES.get(tile.upper(), 0)


# ---------------------------------------------------------------------------
# Single game
# ---------------------------------------------------------------------------

def run_game(
    gaddag: GADDAG,
    model: Optional[HybridPolicyNet ],
    n_candidates: int = 15,
    opponent: str = 'greedy',
) -> List[dict]:
    """Play one complete game and return labeled transitions for player 1.

    Args:
        gaddag       : Loaded GADDAG instance.
        model        : Policy model for player 1 (None → greedy).
        n_candidates : Top-N moves (by raw score) passed to the model.
        opponent     : 'greedy' — player 2 picks highest raw score;
                       'self'   — player 2 uses the same model.

    Returns:
        List of transition dicts, all labeled with the game outcome.
    """
    board   = Board()
    bag     = make_bag()
    tracker = TileTracker()

    p1_rack: List[str] = []
    p2_rack: List[str] = []
    draw_tiles(bag, p1_rack)
    draw_tiles(bag, p2_rack)

    p1_inference = PolicyInference(model) if model is not None else None
    p2_inference = p1_inference if opponent == 'self' else None

    p1_score = 0
    p2_score = 0
    turn     = 0
    consecutive_passes = 0

    # Partial transitions: (board_tensor, feature_vec) recorded before labeling
    p1_partial: List[tuple] = []
    p1_scores_gained: List[int] = []
    p2_scores_gained: List[int] = []

    root = gaddag.root

    while True:

        # ── Player 1 ──────────────────────────────────────────────────────────
        moves1 = generate_moves(board, root, p1_rack)

        if not moves1:
            consecutive_passes += 1
            if consecutive_passes >= 6:
                break
        else:
            consecutive_passes = 0

            if p1_inference is not None:
                ranked = p1_inference.rank_moves(
                    moves1, board, p1_rack, tracker,
                    p1_score, p2_score, turn,
                    top_n=n_candidates,
                )
                chosen = ranked[0][0] if ranked else moves1[0]
            else:
                chosen = moves1[0]   # greedy

            # Record features before applying the move
            _, leave = compute_leave(chosen, p1_rack, board)
            bt, fv   = extract(board, chosen, leave, tracker,
                               p1_score, p2_score, turn)
            p1_partial.append((bt, fv))

            # Apply move
            tiles_used, _ = compute_leave(chosen, p1_rack, board)
            board      = apply_move(board, chosen)
            p1_score  += chosen[1]
            p1_scores_gained.append(chosen[1])

            for tile in tiles_used:
                p1_rack.remove(tile)
                tracker.remove(tile)

            draw_tiles(bag, p1_rack)

        # End: P1 emptied rack with empty bag
        if not bag and not p1_rack:
            bonus = sum(_tile_value(t) for t in p2_rack)
            p1_score += bonus
            p2_score -= bonus
            break

        turn += 1

        # ── Player 2 ──────────────────────────────────────────────────────────
        moves2 = generate_moves(board, root, p2_rack)

        if not moves2:
            consecutive_passes += 1
            if consecutive_passes >= 6:
                break
        else:
            consecutive_passes = 0

            if p2_inference is not None:
                ranked2 = p2_inference.rank_moves(
                    moves2, board, p2_rack, tracker,
                    p2_score, p1_score, turn,
                    top_n=n_candidates,
                )
                chosen2 = ranked2[0][0] if ranked2 else moves2[0]
            else:
                chosen2 = moves2[0]   # greedy

            tiles_used2, _ = compute_leave(chosen2, p2_rack, board)
            board      = apply_move(board, chosen2)
            p2_score  += chosen2[1]
            p2_scores_gained.append(chosen2[1])

            for tile in tiles_used2:
                p2_rack.remove(tile)
                tracker.remove(tile)

            draw_tiles(bag, p2_rack)

        # End: P2 emptied rack with empty bag
        if not bag and not p2_rack:
            bonus = sum(_tile_value(t) for t in p1_rack)
            p2_score += bonus
            p1_score -= bonus
            break

        turn += 1

    # 6-pass termination: each player subtracts their own remaining tiles
    else:
        p1_score -= sum(_tile_value(t) for t in p1_rack)
        p2_score -= sum(_tile_value(t) for t in p2_rack)

    # ── Label each P1 transition with discounted n-step score differential ──
    GAMMA   = 0.70
    N_STEPS = 5

    # Pad p2 if game ended on P1's last move (no following P2 turn)
    while len(p2_scores_gained) < len(p1_scores_gained):
        p2_scores_gained.append(0)

    diffs = [p1 - p2 for p1, p2 in zip(p1_scores_gained, p2_scores_gained)]

    transitions = []
    for i, (bt, fv) in enumerate(p1_partial):
        discounted = sum(
            (GAMMA ** k) * diffs[i + k]
            for k in range(min(N_STEPS, len(diffs) - i))
        )
        transitions.append({
            'board_tensor': bt,
            'feature_vec':  fv,
            'margin_label': discounted / 100.0,
        })

    final_margin = p1_score - p2_score
    return transitions, final_margin


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def run_worker(
    worker_id: int,
    model_path: str,
    gaddag_path: str,
    output_dir: str,
    n_games: int,
    opponent: str,
    n_candidates: int = 15,
    reload_every: int = 50,
) -> None:
    """Runs games and writes transition .pkl files to output_dir.

    Designed to run as one process per CPU node.  The model checkpoint is
    reloaded every reload_every games so workers stay in sync with the
    trainer without any message passing.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f'[Worker {worker_id}] Loading GADDAG...')
    gaddag = GADDAG.load(gaddag_path)

    model = HybridPolicyNet()
    if Path(model_path).exists():
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt, strict=False)
        print(f'[Worker {worker_id}] Loaded model from {model_path}')
    else:
        print(f'[Worker {worker_id}] No checkpoint — using random weights')

    games_played = 0
    while games_played < n_games:

        # Reload model periodically
        if games_played > 0 and games_played % reload_every == 0:
            if Path(model_path).exists():
                ckpt = torch.load(model_path, map_location='cpu')
                model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt, strict=False)
                print(f'[Worker {worker_id}] Reloaded checkpoint at game {games_played}')

            # Poll phase file to switch greedy → self-play
            phase_file = Path(model_path).parent / 'phase.txt'
            if phase_file.exists():
                new_opp = phase_file.read_text().strip()
                if new_opp in ('greedy', 'self') and new_opp != opponent:
                    print(f'[Worker {worker_id}] Switching opponent: {opponent} → {new_opp}')
                    opponent = new_opp

        transitions, _ = run_game(
            gaddag, model,
            n_candidates=n_candidates,
            opponent=opponent,
        )

        stamp    = int(time.time() * 1000)
        out_file = out_path / f'w{worker_id}_g{games_played}_{stamp}.pkl'
        with open(out_file, 'wb') as f:
            pickle.dump(transitions, f)

        games_played += 1
        if games_played % 10 == 0:
            print(f'[Worker {worker_id}] {games_played}/{n_games} games')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-play worker')
    parser.add_argument('--worker-id',    type=int,   default=0)
    parser.add_argument('--model-path',   type=str,   required=True)
    parser.add_argument('--gaddag-path',  type=str,   default='data/gaddag.pkl')
    parser.add_argument('--output-dir',   type=str,   required=True)
    parser.add_argument('--n-games',      type=int,   default=9999)
    parser.add_argument('--opponent',     type=str,   default='greedy',
                        choices=['greedy', 'self'])
    parser.add_argument('--n-candidates', type=int,   default=15)
    parser.add_argument('--reload-every', type=int,   default=50)
    args = parser.parse_args()

    run_worker(
        worker_id    = args.worker_id,
        model_path   = args.model_path,
        gaddag_path  = args.gaddag_path,
        output_dir   = args.output_dir,
        n_games      = args.n_games,
        opponent     = args.opponent,
        n_candidates = args.n_candidates,
        reload_every = args.reload_every,
    )
