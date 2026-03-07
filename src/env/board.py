from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import random


@dataclass
class BoardConfig:
    rows: int
    cols: int
    num_good: int
    num_bomb: int
    seed: Optional[int] = None

    @property
    def board_size(self) -> int:
        return self.rows * self.cols

    @property
    def num_neutral(self) -> int:
        return self.board_size - self.num_good - self.num_bomb

    def validate(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive integers.")

        if self.num_good < 0 or self.num_bomb < 0:
            raise ValueError("num_good and num_bomb must be non-negative.")

        if self.num_good + self.num_bomb > self.board_size:
            raise ValueError("num_good + num_bomb cannot exceed board size.")

        if self.num_neutral < 0:
            raise ValueError("num_neutral cannot be negative.")


@dataclass
class BoardCell:
    word: str
    role: str  # "good", "neutral", "bomb"
    revealed: bool = False
    guess_order: Optional[int] = None


def load_words(path: str | Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Word file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    seen = set()
    unique_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)

    return unique_words


def flatten_board(board: List[List[BoardCell]]) -> List[BoardCell]:
    return [cell for row in board for cell in row]


def generate_board(words: List[str], config: BoardConfig) -> List[List[BoardCell]]:
    config.validate()

    if len(words) < config.board_size:
        raise ValueError(
            f"Need at least {config.board_size} words, but only got {len(words)}."
        )

    rng = random.Random(config.seed)

    selected_words = rng.sample(words, config.board_size)

    roles = (
        ["good"] * config.num_good
        + ["bomb"] * config.num_bomb
        + ["neutral"] * config.num_neutral
    )
    rng.shuffle(roles)

    flat_cells = [
        BoardCell(word=word, role=role) for word, role in zip(selected_words, roles)
    ]

    board: List[List[BoardCell]] = []
    idx = 0
    for _ in range(config.rows):
        row = []
        for _ in range(config.cols):
            row.append(flat_cells[idx])
            idx += 1
        board.append(row)

    return board


def get_available_indices(board: List[List[BoardCell]]) -> List[int]:
    flat = flatten_board(board)
    return [i for i, cell in enumerate(flat) if not cell.revealed]


def reveal_cell_by_index(board: List[List[BoardCell]], idx: int) -> BoardCell:
    flat = flatten_board(board)

    if idx < 0 or idx >= len(flat):
        raise IndexError(f"Index {idx} out of range.")

    cell = flat[idx]

    if cell.revealed:
        return cell

    existing_orders = [c.guess_order for c in flat if c.guess_order is not None]
    next_order = 1 if not existing_orders else max(existing_orders) + 1

    cell.revealed = True
    cell.guess_order = next_order
    return cell


def summarize_roles(board: List[List[BoardCell]]) -> dict:
    counts = {"good": 0, "neutral": 0, "bomb": 0}
    for cell in flatten_board(board):
        counts[cell.role] += 1
    return counts


def board_role_table(board: List[List[BoardCell]]) -> List[dict]:
    rows = []
    for r, row in enumerate(board):
        for c, cell in enumerate(row):
            rows.append(
                {
                    "row": r,
                    "col": c,
                    "word": cell.word,
                    "role": cell.role,
                    "revealed": cell.revealed,
                    "guess_order": cell.guess_order,
                }
            )
    return rows


def print_board_roles(board: List[List[BoardCell]]) -> None:
    print("Board role mapping:")
    for r, row in enumerate(board):
        row_items = []
        for c, cell in enumerate(row):
            row_items.append(f"({r},{c}) {cell.word}: {cell.role}")
        print(" | ".join(row_items))


def make_standard_config(
    rows: int, cols: int, seed: Optional[int] = None
) -> BoardConfig:
    size = rows * cols

    if size == 4:  # 2x2
        return BoardConfig(rows=2, cols=2, num_good=1, num_bomb=1, seed=seed)

    if size == 9:  # 3x3
        return BoardConfig(rows=3, cols=3, num_good=3, num_bomb=1, seed=seed)

    if size == 25:  # 5x5
        return BoardConfig(rows=5, cols=5, num_good=8, num_bomb=1, seed=seed)

    raise ValueError(
        f"No preset for {rows}x{cols}. Create BoardConfig manually for custom settings."
    )


def count_revealed_good(board) -> int:
    return sum(
        1 for row in board for cell in row if cell.role == "good" and cell.revealed
    )


def all_good_revealed(board) -> bool:
    total_good = sum(1 for row in board for cell in row if cell.role == "good")
    revealed_good = sum(
        1 for row in board for cell in row if cell.role == "good" and cell.revealed
    )
    return revealed_good == total_good
