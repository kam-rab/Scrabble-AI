"""Fixed-size circular replay buffer for training transitions.

Each transition is a dict:
    board_tensor : np.ndarray (33, 15, 15)   – board state after proposed move
    feature_vec  : np.ndarray (64,)           – scalar features
    win_label    : float  (1.0 win / 0.5 draw / 0.0 loss)
    margin_label : float  (our_score − opp_score, normalised by 500)

Workers write .pkl files of transition lists to a shared directory.
load_from_dir() ingests them and deletes the files, acting as a
shared-filesystem message queue between workers and the trainer.
"""


import pickle
import random
from pathlib import Path

import numpy as np
from typing import List, Union


class ReplayBuffer:

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._buf: List[dict] = []
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, transition: dict) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(transition)
        else:
            self._buf[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def add_game(self, transitions: List[dict]) -> None:
        for t in transitions:
            self.add(t)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> List[dict]:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    # ------------------------------------------------------------------
    # Shared-filesystem ingestion
    # ------------------------------------------------------------------

    def load_from_dir(self, transition_dir: Union[str, Path]) -> int:
        """Load all .pkl files from directory into the buffer.

        Files are deleted after loading so they are not processed twice.
        Returns the total number of transitions ingested this call.
        """
        path  = Path(transition_dir)
        added = 0
        for pkl_file in sorted(path.glob('*.pkl')):
            try:
                with open(pkl_file, 'rb') as f:
                    transitions: List[dict] = pickle.load(f)
                for t in transitions:
                    self.add(t)
                added += len(transitions)
                pkl_file.unlink()
            except Exception:
                pass   # skip partially-written files
        return added

    def __len__(self) -> int:
        return len(self._buf)
