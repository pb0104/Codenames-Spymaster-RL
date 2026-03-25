from __future__ import annotations

from typing import Callable

from src.evaluation.metrics import EpisodeMetrics, summarize_metrics


def select_policy_action(agent, observation, env, deterministic: bool = True):
    if hasattr(agent, "predict"):
        action, _ = agent.predict(observation, deterministic=deterministic)
        return action
    if hasattr(agent, "select_action"):
        return agent.select_action(env)
    raise TypeError("Agent must provide either predict() or select_action().")


def evaluate_agent(
    agent,
    env_factory: Callable[[], object],
    episodes: int = 5,
    deterministic: bool = True,
) -> dict:
    metrics: list[EpisodeMetrics] = []
    for _ in range(episodes):
        env = env_factory()
        observation, _ = env.reset()
        done = False
        episode_return = 0.0
        turns = 0
        assassin_hit = False
        won = False

        while not done:
            action = select_policy_action(agent, observation, env, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            turns += 1
            episode_return += float(reward)
            assassin_hit = assassin_hit or bool(info.get("assassin_hit", False))
            won = won or bool(info.get("won", False))

        metrics.append(
            EpisodeMetrics(
                episode_return=episode_return,
                turns=turns,
                won=won,
                assassin_hit=assassin_hit,
                friendly_revealed=env.board_config.num_friendly - len(env.remaining_friendly_indices),
                friendly_total=env.board_config.num_friendly,
            )
        )

    return summarize_metrics(metrics)
