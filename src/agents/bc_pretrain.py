from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.baselines.greedy_spymaster import GreedySpymaster


@dataclass
class DemonstrationTransition:
    obs: dict[str, np.ndarray]
    action: np.ndarray
    next_obs: dict[str, np.ndarray]
    reward: float
    done: bool
    info: dict


def clone_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def generate_demonstrations(
    env_factory: Callable[[], object],
    num_episodes: int,
    max_steps_per_episode: int | None = None,
) -> list[DemonstrationTransition]:
    policy = GreedySpymaster()
    transitions: list[DemonstrationTransition] = []

    for _ in range(num_episodes):
        env = env_factory()
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action = policy.select_action(env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transitions.append(
                DemonstrationTransition(
                    obs=clone_observation(obs),
                    action=np.array(action, copy=True),
                    next_obs=clone_observation(next_obs),
                    reward=float(reward),
                    done=done,
                    info=deepcopy(info),
                )
            )
            obs = next_obs
            steps += 1
            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                break

    return transitions


def stack_demo_observations(
    transitions: list[DemonstrationTransition],
) -> dict[str, np.ndarray]:
    keys = transitions[0].obs.keys()
    return {
        key: np.stack([transition.obs[key] for transition in transitions]).astype(np.float32)
        for key in keys
    }


def stack_demo_next_observations(
    transitions: list[DemonstrationTransition],
) -> dict[str, np.ndarray]:
    keys = transitions[0].next_obs.keys()
    return {
        key: np.stack([transition.next_obs[key] for transition in transitions]).astype(
            np.float32
        )
        for key in keys
    }


def stack_demo_actions(transitions: list[DemonstrationTransition]) -> np.ndarray:
    return np.stack([transition.action for transition in transitions]).astype(np.float32)
