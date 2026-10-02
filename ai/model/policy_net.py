"""Hybrid CNN + MLP policy network for Scrabble move evaluation.

Architecture
------------
CNN branch  : 2 conv layers (32, 64 filters) on the 15×15 board tensor
              → AdaptiveAvgPool(4×4) → Linear(1024→128)
MLP branch  : 2 linear layers on the 64-dim scalar feature vector
              → Linear(64→128→128)
Fusion      : concatenate both embeddings (256-dim)
              → Linear(256→128→64)
Output head :
  score_head – Linear(64→1)            → predicted score margin (raw)

Total parameters: ~200k.  A single CPU forward pass over a batch of 15
candidate moves runs in well under 100 ms.
"""


import torch
import torch.nn as nn

from ai.model.features import N_BOARD_CHANNELS, N_FEATURES
from typing import Tuple


class HybridPolicyNet(nn.Module):

    def __init__(
        self,
        n_board_channels: int = N_BOARD_CHANNELS,
        n_features: int = N_FEATURES,
    ) -> None:
        super().__init__()

        # ── CNN branch ────────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(n_board_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # → (B, 64, 4, 4)
            nn.Flatten(),                   # → (B, 1024)
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        # ── MLP branch ────────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.score_head = nn.Linear(64, 1)

    def forward(
        self,
        board_tensor: torch.Tensor,   # (B, N_BOARD_CHANNELS, 15, 15)
        feature_vec: torch.Tensor,    # (B, N_FEATURES)
    ) -> torch.Tensor:
        """Return score_margin, shape (B, 1)."""
        board_emb = self.cnn(board_tensor)                              # (B, 128)
        feat_emb  = self.mlp(feature_vec)                               # (B, 128)
        trunk     = self.fusion(torch.cat([board_emb, feat_emb], dim=1))  # (B, 64)
        return self.score_head(trunk)
