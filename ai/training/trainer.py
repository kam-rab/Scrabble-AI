"""Central training loop for the Scrabble AI policy network.

Run on one dedicated node:

    python -m ai.training.trainer \
        --model-path /shared/model.pt \
        --transition-dir /shared/transitions/ \
        --gaddag-path data/gaddag.pkl \
        --log data/training_logs/training_log.csv

The trainer polls transition-dir for .pkl files written by worker nodes,
loads them into the replay buffer, trains for a fixed number of gradient
steps per cycle, saves the checkpoint, and periodically evaluates win rate
versus the greedy baseline.

Training automatically transitions from Phase 1 (vs. greedy) to Phase 2
(self-play) once win_rate_vs_greedy > WIN_RATE_THRESHOLD.  Workers are
notified of the phase change via a shared phase file.
"""


import os
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai.model.policy_net import HybridPolicyNet
from ai.training.replay_buffer import ReplayBuffer
from typing import List, Tuple

WIN_RATE_THRESHOLD = 0.72   # advance to self-play above this win rate
PHASE_FILE_NAME    = 'phase.txt'  # written to model dir; workers can poll it


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------

def _collate(batch: List[dict]) -> Tuple[torch.Tensor, ...]:
    boards   = np.stack([t['board_tensor'] for t in batch])
    feats    = np.stack([t['feature_vec']  for t in batch])
    margins  = np.array([t['margin_label'] for t in batch], dtype=np.float32)
    return (
        torch.tensor(boards,  dtype=torch.float32),
        torch.tensor(feats,   dtype=torch.float32),
        torch.tensor(margins, dtype=torch.float32).unsqueeze(1),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_vs_greedy(
    model: HybridPolicyNet,
    gaddag,
    n_games: int = 50,
    n_candidates: int = 15,
) -> Tuple[float, float]:
    """Play n_games against greedy and return (win_rate, avg_margin)."""
    from ai.training.self_play import run_game
    wins    = 0
    margins = []
    for _ in range(n_games):
        transitions, final_margin = run_game(
            gaddag, model,
            n_candidates=n_candidates,
            opponent='greedy',
        )
        if not transitions:
            continue
        if final_margin > 0:
            wins += 1
        margins.append(final_margin)
    win_rate   = wins / n_games if n_games else 0.0
    avg_margin = float(np.mean(margins)) if margins else 0.0
    return win_rate, avg_margin


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model_path: str,
    transition_dir: str,
    gaddag_path: str,
    log_path: str,
    batch_size: int       = 256,
    steps_per_cycle: int  = 50,
    eval_every: int       = 2000,  # games between evaluations
    lr: float             = 1e-4,
    weight_decay: float   = 1e-5,
) -> None:
    from gaddag import GADDAG

    device   = 'cpu'
    model    = HybridPolicyNet().to(device)
    mp       = Path(model_path)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if mp.exists():
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass  # optimizer state incompatible after arch change
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f'Resumed from {model_path}')
    else:
        mp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)
        print('Initialised new model')
    mse       = nn.MSELoss()
    buffer    = ReplayBuffer(capacity=100_000)

    print('Loading GADDAG for evaluation...')
    gaddag = GADDAG.load(gaddag_path)

    write_header = not Path(log_path).exists() or Path(log_path).stat().st_size == 0
    log_file = open(log_path, 'a', newline='')
    writer   = csv.writer(log_file)
    if write_header:
        writer.writerow([
            'timestamp', 'phase', 'buffer_size',
            'score_loss', 'win_rate', 'avg_margin',
        ])
        log_file.flush()

    phase          = 1       # 1 = vs greedy, 2 = self-play
    games_since_eval = 0
    phase_file     = Path(model_path).parent / PHASE_FILE_NAME
    phase_file.write_text('greedy')

    print('Trainer running. Waiting for transitions...')

    while True:
        # Ingest new transition files from workers
        new = buffer.load_from_dir(transition_dir)
        if new > 0:
            approx_games = max(1, new // 15)
            games_since_eval += approx_games
            print(f'Ingested ~{approx_games} games ({new} transitions). '
                  f'Buffer: {len(buffer)}')

        if len(buffer) < batch_size:
            time.sleep(5)
            continue

        # ── Gradient steps ────────────────────────────────────────────────
        model.train()
        total_sl = 0.0

        for _ in range(steps_per_cycle):
            batch  = buffer.sample(batch_size)
            boards, feats, margin_labels = _collate(batch)

            optimizer.zero_grad()
            score_preds = model(boards, feats)

            loss = mse(score_preds, margin_labels)
            loss.backward()
            optimizer.step()

            total_sl += loss.item()

        avg_sl = total_sl / steps_per_cycle

        model.eval()
        tmp_path = model_path + '.tmp'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, tmp_path)
        os.replace(tmp_path, model_path)

        # ── Periodic evaluation ───────────────────────────────────────────
        win_rate = avg_margin = 0.0
        if games_since_eval >= eval_every:
            print('Evaluating vs greedy...')
            win_rate, avg_margin = eval_vs_greedy(
                model, gaddag, n_games=50,
            )
            games_since_eval = 0
            print(f'  Win rate: {win_rate:.1%}  Avg margin: {avg_margin:+.1f}')

            # Phase transition
            if phase == 1 and win_rate >= WIN_RATE_THRESHOLD:
                phase = 2
                phase_file.write_text('self')
                print(f'*** Phase 2: switching to self-play '
                      f'(win rate {win_rate:.1%} ≥ {WIN_RATE_THRESHOLD:.0%}) ***')

        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([
            ts, phase, len(buffer),
            f'{avg_sl:.4f}',
            f'{win_rate:.3f}', f'{avg_margin:.1f}',
        ])
        log_file.flush()

        time.sleep(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Policy network trainer')
    parser.add_argument('--model-path',      type=str,   required=True)
    parser.add_argument('--transition-dir',  type=str,   required=True)
    parser.add_argument('--gaddag-path',     type=str,   default='data/gaddag.pkl')
    parser.add_argument('--log',             type=str,   default='data/training_logs/training_log.csv')
    parser.add_argument('--batch-size',      type=int,   default=256)
    parser.add_argument('--steps-per-cycle', type=int,   default=50)
    parser.add_argument('--eval-every',      type=int,   default=2000)
    parser.add_argument('--lr',              type=float, default=1e-5)
    args = parser.parse_args()

    train(
        model_path      = args.model_path,
        transition_dir  = args.transition_dir,
        gaddag_path     = args.gaddag_path,
        log_path        = args.log,
        batch_size      = args.batch_size,
        steps_per_cycle = args.steps_per_cycle,
        eval_every      = args.eval_every,
        lr              = args.lr,
    )
