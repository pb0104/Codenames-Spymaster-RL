from __future__ import annotations

import random
from typing import Optional, List

from src.env.board import BoardCell, get_available_indices


class RandomAgent:
    """Random guesser fallback and random spymaster action sampler."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, env_or_board):
        if hasattr(env_or_board, "sample_action"):
            return env_or_board.sample_action(self.rng)

        board: List[List[BoardCell]] = env_or_board
        available = get_available_indices(board)
        if not available:
            raise ValueError("No available actions left.")
        return self.rng.choice(available)
