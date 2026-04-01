from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from PIL import Image

from src.env.board import BoardCell, flatten_board
from src.evaluation.evaluate_agent import select_policy_action


ROLE_COLORS = {
    "friendly": "#3D8D5A",
    "opponent": "#C95745",
    "neutral": "#D9D0C7",
    "assassin": "#2C2C2C",
}
ROLE_TEXT_COLORS = {
    "friendly": "white",
    "opponent": "white",
    "neutral": "#222222",
    "assassin": "white",
}
HIDDEN_COLOR = "#F3E3C2"
HIDDEN_TEXT_COLOR = "#1F1F1F"
GOAL_BORDER_COLOR = "#F4B400"
RECENT_GUESS_BORDER_COLOR = "#1A73E8"


@dataclass
class RolloutFrame:
    step_index: int
    turn_index: int
    reveal_index: int
    board: list[list[BoardCell]]
    clue: str | None
    count: int | None
    goal_words: list[str]
    guessed_words: list[str]
    guessed_roles: list[str]
    guessed_indices: list[int]
    reward: float
    cumulative_reward: float
    done: bool
    won: bool
    assassin_hit: bool
    friendly_remaining: int
    info: dict[str, Any]


@dataclass
class RolloutTrace:
    pipeline_name: str
    agent_label: str
    frames: list[RolloutFrame]

    @property
    def final_frame(self) -> RolloutFrame:
        return self.frames[-1]

    def summary(self) -> dict[str, Any]:
        final = self.final_frame
        return {
            "pipeline_name": self.pipeline_name,
            "agent_label": self.agent_label,
            "steps": max(len(self.frames) - 1, 0),
            "total_reward": final.cumulative_reward,
            "won": final.won,
            "assassin_hit": final.assassin_hit,
            "friendly_remaining": final.friendly_remaining,
        }

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for frame in self.frames[1:]:
            rows.append(
                {
                    "step": frame.step_index,
                    "turn": frame.turn_index,
                    "reveal_in_turn": frame.reveal_index,
                    "clue": frame.clue,
                    "count": frame.count,
                    "goal_words": ", ".join(frame.goal_words),
                    "guessed_words": ", ".join(frame.guessed_words),
                    "guessed_roles": ", ".join(frame.guessed_roles),
                    "reward": frame.reward,
                    "cumulative_reward": frame.cumulative_reward,
                    "friendly_remaining": frame.friendly_remaining,
                    "won": frame.won,
                    "assassin_hit": frame.assassin_hit,
                }
            )
        return rows


def _snapshot_board(board: list[list[BoardCell]], guess_orders: dict[int, int]) -> list[list[BoardCell]]:
    snapshot = deepcopy(board)
    for idx, cell in enumerate(flatten_board(snapshot)):
        cell.guess_order = guess_orders.get(idx)
    return snapshot


def _friendly_remaining(board: list[list[BoardCell]]) -> int:
    return sum(
        1 for cell in flatten_board(board) if cell.role == "friendly" and not cell.revealed
    )


def capture_rollout_trace(
    agent: Any,
    env_factory: Callable[[], Any],
    *,
    pipeline_name: str,
    agent_label: str,
    deterministic: bool = True,
) -> RolloutTrace:
    env = env_factory()
    observation, _ = env.reset()
    cumulative_reward = 0.0
    done = False
    step_index = 0
    turn_index = 0
    guess_orders: dict[int, int] = {}
    next_guess_order = 1
    current_board = deepcopy(env.board)

    frames = [
        RolloutFrame(
            step_index=0,
            turn_index=0,
            reveal_index=0,
            board=_snapshot_board(current_board, guess_orders),
            clue=None,
            count=None,
            goal_words=[env.flat_board[idx].word for idx in env.current_goal_indices],
            guessed_words=[],
            guessed_roles=[],
            guessed_indices=[],
            reward=0.0,
            cumulative_reward=0.0,
            done=False,
            won=False,
            assassin_hit=False,
            friendly_remaining=_friendly_remaining(current_board),
            info={"phase": "reset"},
        )
    ]

    while not done:
        turn_index += 1
        board_before_action = deepcopy(current_board)
        action = select_policy_action(agent, observation, env, deterministic=deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        guessed_indices = list(info.get("guessed_indices", []))
        guessed_words = list(info.get("guessed_words", []))
        guessed_roles = list(info.get("guessed_roles", []))
        working_board = deepcopy(board_before_action)

        if guessed_indices:
            for reveal_index, (guessed_index, guessed_word, guessed_role) in enumerate(
                zip(guessed_indices, guessed_words, guessed_roles),
                start=1,
            ):
                if guessed_index not in guess_orders:
                    guess_orders[guessed_index] = next_guess_order
                    next_guess_order += 1

                working_flat = flatten_board(working_board)
                working_flat[guessed_index].revealed = True
                working_flat[guessed_index].guess_order = guess_orders[guessed_index]

                step_index += 1
                is_last_reveal = reveal_index == len(guessed_indices)
                if is_last_reveal:
                    cumulative_reward += float(reward)

                frames.append(
                    RolloutFrame(
                        step_index=step_index,
                        turn_index=turn_index,
                        reveal_index=reveal_index,
                        board=_snapshot_board(working_board, guess_orders),
                        clue=info.get("clue"),
                        count=info.get("count"),
                        goal_words=list(info.get("goal_words", [])),
                        guessed_words=[guessed_word],
                        guessed_roles=[guessed_role],
                        guessed_indices=[guessed_index],
                        reward=float(reward) if is_last_reveal else 0.0,
                        cumulative_reward=cumulative_reward,
                        done=done if is_last_reveal else False,
                        won=bool(info.get("won", False)) if is_last_reveal else False,
                        assassin_hit=bool(info.get("assassin_hit", False))
                        if is_last_reveal
                        else False,
                        friendly_remaining=_friendly_remaining(working_board),
                        info=deepcopy(info),
                    )
                )
        else:
            step_index += 1
            cumulative_reward += float(reward)
            frames.append(
                RolloutFrame(
                    step_index=step_index,
                    turn_index=turn_index,
                    reveal_index=0,
                    board=_snapshot_board(board_before_action, guess_orders),
                    clue=info.get("clue"),
                    count=info.get("count"),
                    goal_words=list(info.get("goal_words", [])),
                    guessed_words=[],
                    guessed_roles=[],
                    guessed_indices=[],
                    reward=float(reward),
                    cumulative_reward=cumulative_reward,
                    done=done,
                    won=bool(info.get("won", False)),
                    assassin_hit=bool(info.get("assassin_hit", False)),
                    friendly_remaining=_friendly_remaining(board_before_action),
                    info=deepcopy(info),
                )
            )

        current_board = deepcopy(working_board if guessed_indices else env.board)

    return RolloutTrace(
        pipeline_name=pipeline_name,
        agent_label=agent_label,
        frames=frames,
    )


def _draw_board(
    ax,
    board: list[list[BoardCell]],
    *,
    reveal_roles: bool,
    goal_words: set[str],
    recent_guess_indices: set[int],
    title: str,
) -> None:
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    flat = flatten_board(board)
    for idx, cell in enumerate(flat):
        r = idx // cols
        c = idx % cols
        if reveal_roles:
            facecolor = ROLE_COLORS[cell.role]
            textcolor = ROLE_TEXT_COLORS[cell.role]
        elif cell.revealed:
            facecolor = ROLE_COLORS[cell.role]
            textcolor = ROLE_TEXT_COLORS[cell.role]
        else:
            facecolor = HIDDEN_COLOR
            textcolor = HIDDEN_TEXT_COLOR

        border_color = "black"
        linewidth = 1.2
        if cell.word in goal_words:
            border_color = GOAL_BORDER_COLOR
            linewidth = 3.0
        if idx in recent_guess_indices:
            border_color = RECENT_GUESS_BORDER_COLOR
            linewidth = 3.6

        rect = Rectangle(
            (c, r),
            1,
            1,
            facecolor=facecolor,
            edgecolor=border_color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
        ax.text(
            c + 0.5,
            r + 0.56,
            cell.word,
            ha="center",
            va="center",
            fontsize=8,
            color=textcolor,
            wrap=True,
        )
        if cell.guess_order is not None:
            ax.text(
                c + 0.08,
                r + 0.16,
                f"#{cell.guess_order}",
                ha="left",
                va="top",
                fontsize=7,
                color="#0B57D0" if not reveal_roles else "#FFF59D",
                fontweight="bold",
            )


def _render_frame_image(frame: RolloutFrame, *, pipeline_name: str, agent_label: str) -> Image.Image:
    fig = plt.figure(figsize=(14, 8))
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.6])
    ax_visible = fig.add_subplot(grid[0, 0])
    ax_roles = fig.add_subplot(grid[0, 1])
    ax_text = fig.add_subplot(grid[1, :])

    goal_words = set(frame.goal_words)
    recent_guess_indices = set(frame.guessed_indices)
    _draw_board(
        ax_visible,
        frame.board,
        reveal_roles=False,
        goal_words=goal_words,
        recent_guess_indices=recent_guess_indices,
        title="Visible Board",
    )
    _draw_board(
        ax_roles,
        frame.board,
        reveal_roles=True,
        goal_words=goal_words,
        recent_guess_indices=recent_guess_indices,
        title="Role Board",
    )

    ax_text.axis("off")
    if frame.step_index == 0:
        lines = [
            f"Pipeline: {pipeline_name}",
            f"Agent: {agent_label}",
            f"Initial goal words: {', '.join(frame.goal_words) or 'None'}",
            "Each GIF step reveals at most one new word.",
            "Yellow border = current goal, blue border = current reveal.",
        ]
    else:
        guessed_pairs = ", ".join(
            f"{word} ({role})" for word, role in zip(frame.guessed_words, frame.guessed_roles)
        )
        conclusion = "Win" if frame.won else "Assassin hit" if frame.assassin_hit else "Continue"
        lines = [
            f"Step {frame.step_index} | Turn {frame.turn_index} | Reveal {frame.reveal_index}: clue='{frame.clue}' count={frame.count}",
            f"Goal words: {', '.join(frame.goal_words) or 'None'}",
            f"Guessed: {guessed_pairs or 'None'}",
            f"Reward: {frame.reward:.3f} | Cumulative: {frame.cumulative_reward:.3f} | Friendly remaining: {frame.friendly_remaining}",
            f"Conclusion: {conclusion}",
        ]
    ax_text.text(0.01, 0.92, "\n".join(lines), fontsize=11, va="top", family="monospace")

    fig.suptitle(f"{pipeline_name} rollout", fontsize=15)
    fig.tight_layout()
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = Image.fromarray(
        np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    ).convert("RGB")
    plt.close(fig)
    return image


def save_rollout_gif(
    trace: RolloutTrace,
    output_path: str | Path,
    *,
    duration_ms: int = 1400,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [
        _render_frame_image(frame, pipeline_name=trace.pipeline_name, agent_label=trace.agent_label)
        for frame in trace.frames
    ]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path
