from __future__ import annotations

import random
from typing import Optional, List

from src.env.board import BoardCell, get_available_indices


class RandomAgent:
    """
    A simple random agent.
    Later you can replace this with PPO/SAC/HER/etc.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, board: List[List[BoardCell]]) -> int:
        available = get_available_indices(board)
        if not available:
            raise ValueError("No available actions left.")
        return self.rng.choice(available)
