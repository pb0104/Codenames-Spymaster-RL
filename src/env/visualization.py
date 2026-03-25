from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.env.board import BoardCell


ROLE_COLORS = {
    "friendly": "#4CAF50",
    "opponent": "#C62828",
    "neutral": "#D6D6D6",
    "assassin": "#000000",
}

ROLE_TEXT_COLORS = {
    "friendly": "white",
    "opponent": "white",
    "neutral": "black",
    "assassin": "white",
}

HIDDEN_COLOR = "#F5DEB3"
HIDDEN_TEXT_COLOR = "black"


def plot_board(
    board: List[List[BoardCell]],
    reveal_roles: bool = False,
    reveal_revealed_only: bool = True,
    title: str = "Codenames Board",
    figsize_scale: float = 1.0,
    font_size: int = 8,
    save_path: Optional[str | Path] = None,
) -> None:
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0

    fig, ax = plt.subplots(figsize=(cols * figsize_scale, rows * figsize_scale))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(rows):
        for c in range(cols):
            cell = board[r][c]

            if reveal_roles:
                facecolor = ROLE_COLORS[cell.role]
                textcolor = ROLE_TEXT_COLORS[cell.role]
            elif reveal_revealed_only and cell.revealed:
                facecolor = ROLE_COLORS[cell.role]
                textcolor = ROLE_TEXT_COLORS[cell.role]
            else:
                facecolor = HIDDEN_COLOR
                textcolor = HIDDEN_TEXT_COLOR

            rect = Rectangle(
                (c, r),
                1,
                1,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.2,
            )
            ax.add_patch(rect)

            ax.text(
                c + 0.5,
                r + 0.55,
                cell.word,
                ha="center",
                va="center",
                fontsize=font_size,
                color=textcolor,
                wrap=True,
            )

            if cell.guess_order is not None:
                guess_color = "blue" if not reveal_roles else "yellow"
                ax.text(
                    c + 0.06,
                    r + 0.14,
                    f"#{cell.guess_order}",
                    ha="left",
                    va="top",
                    fontsize=max(font_size - 1, 6),
                    color=guess_color,
                    fontweight="bold",
                )

    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
