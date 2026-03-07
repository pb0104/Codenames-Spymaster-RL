from __future__ import annotations

from typing import List, Tuple, Dict

from src.env.board import (
    BoardCell,
    BoardConfig,
    generate_board,
    reveal_cell_by_index,
    get_available_indices,
    all_good_revealed,
)


class CodenamesGame:

    def __init__(self, words: List[str], config: BoardConfig):

        self.words = words
        self.config = config

        self.board: List[List[BoardCell]] = []

        self.done = False

    def reset(self):

        self.board = generate_board(self.words, self.config)

        self.done = False

        return self.board

    def step(self, action: int):

        if self.done:
            raise RuntimeError("Game already finished. Call reset().")

        cell = reveal_cell_by_index(self.board, action)

        reward = 0

        if cell.role == "good":
            reward = 1

        elif cell.role == "neutral":
            reward = 0

        elif cell.role == "bomb":
            reward = -10
            self.done = True

        if all_good_revealed(self.board):
            self.done = True
            reward = 10

        observation = self.board

        info = {
            "word": cell.word,
            "role": cell.role,
        }

        return observation, reward, self.done, info

    def available_actions(self):

        return get_available_indices(self.board)
