"""
eval.py  –  Evaluation helpers for agent vs greedy baseline.
"""

import numpy as np
from typing import Tuple


def evaluate_agent(model, env, n_episodes: int = 50) -> Tuple[float, float]:
    """
    Roll out the trained model for n_episodes.

    Returns
    -------
    win_rate  : fraction of episodes won (all friendly words revealed)
    avg_turns : average number of turns to win (inf if lost)
    """
    wins  = 0
    turns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        n_turns = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_turns += 1

        # Check win: no friendly words remaining
        friendly_remaining = int(np.sum(
            (env.unwrapped.labels == 0) & env.unwrapped.remaining_mask
        ))
        if friendly_remaining == 0 and not info.get("hit_assassin", False):
            wins += 1
            turns.append(n_turns)

    win_rate  = wins / n_episodes
    avg_turns = np.mean(turns) if turns else float("inf")
    return win_rate, avg_turns


def evaluate_greedy_baseline(greedy, env, n_episodes: int = 50) -> Tuple[float, float]:
    """Same as evaluate_agent but rolls out the greedy Spymaster."""
    wins  = 0
    turns = []

    for _ in range(n_episodes):
        env.reset()
        done    = False
        n_turns = 0

        while not done:
            clue_idx, count = greedy.select_action(
                env.unwrapped.labels,
                env.unwrapped.remaining_mask,
            )
            action = greedy.action_to_flat(clue_idx, count)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_turns += 1

        friendly_remaining = int(np.sum(
            (env.unwrapped.labels == 0) & env.unwrapped.remaining_mask
        ))
        if friendly_remaining == 0 and not info.get("hit_assassin", False):
            wins += 1
            turns.append(n_turns)

    win_rate  = wins / n_episodes
    avg_turns = np.mean(turns) if turns else float("inf")
    return win_rate, avg_turns
