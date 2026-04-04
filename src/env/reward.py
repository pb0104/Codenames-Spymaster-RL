from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from src.utils.similarity import cosine_similarity_matrix


@dataclass
class RewardConfig:
    turn_penalty: float = -1.0
    opponent_penalty: float = -3.0
    neutral_penalty: float = -1.5
    assassin_penalty: float = -25.0
    shaped_weight: float = 1.0


@dataclass
class RewardBreakdown:
    total: float
    reward_without_goal: float
    goal_reward: float
    shaped_reward: float
    bad_guess_penalty: float
    clue_margin: float
    bad_guess_role: str | None
    assassin_hit: bool
    goal_achieved: bool

    def to_dict(self) -> dict:
        return asdict(self)


def subset_achieved(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
    achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
    desired_goal = np.asarray(desired_goal, dtype=np.float32)
    return np.all(achieved_goal >= desired_goal - 0.5, axis=-1)


def clue_margin(
    clue_embedding: np.ndarray,
    target_embeddings: np.ndarray,
    bad_embeddings: np.ndarray,
) -> float:
    if target_embeddings.size == 0:
        return 0.0

    clue_embedding = clue_embedding.reshape(1, -1)
    target_scores = cosine_similarity_matrix(clue_embedding, target_embeddings)[0]
    bad_scores = (
        cosine_similarity_matrix(clue_embedding, bad_embeddings)[0]
        if bad_embeddings.size > 0
        else np.array([-1.0], dtype=np.float32)
    )
    return float(np.min(target_scores) - np.max(bad_scores))


def build_step_reward(
    *,
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    reward_without_goal: float,
    clue_margin_value: float,
    bad_guess_penalty: float,
    bad_guess_role: str | None,
    assassin_hit: bool,
    config: RewardConfig,
) -> RewardBreakdown:
    goal_achieved = bool(subset_achieved(achieved_goal[None, :], desired_goal[None, :])[0])
    goal_reward = 0.0 if goal_achieved else config.turn_penalty
    total = float(reward_without_goal + goal_reward)
    return RewardBreakdown(
        total=total,
        reward_without_goal=float(reward_without_goal),
        goal_reward=float(goal_reward),
        shaped_reward=float(config.shaped_weight * clue_margin_value),
        bad_guess_penalty=float(bad_guess_penalty),
        clue_margin=float(clue_margin_value),
        bad_guess_role=bad_guess_role,
        assassin_hit=assassin_hit,
        goal_achieved=goal_achieved,
    )


def bad_guess_penalty(guessed_roles: Iterable[str], config: RewardConfig) -> tuple[float, str | None]:
    roles = list(guessed_roles)
    if "assassin" in roles:
        return float(config.assassin_penalty), "assassin"
    if "opponent" in roles:
        return float(config.opponent_penalty), "opponent"
    if "neutral" in roles:
        return float(config.neutral_penalty), "neutral"
    return 0.0, None


def compute_goal_conditioned_reward(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    info: dict | Iterable[dict],
    config: RewardConfig,
) -> np.ndarray | float:
    success = subset_achieved(achieved_goal, desired_goal)

    if isinstance(info, dict):
        reward_without_goal = float(info.get("reward_without_goal", 0.0))
        return float(reward_without_goal + (0.0 if bool(success) else config.turn_penalty))

    reward_without_goal = np.array(
        [float(item.get("reward_without_goal", 0.0)) for item in info], dtype=np.float32
    )
    goal_reward = np.where(success, 0.0, config.turn_penalty).astype(np.float32)
    return reward_without_goal + goal_reward
