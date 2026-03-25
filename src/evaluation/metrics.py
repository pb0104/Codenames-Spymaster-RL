from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class EpisodeMetrics:
    episode_return: float
    turns: int
    won: bool
    assassin_hit: bool
    friendly_revealed: int
    friendly_total: int


def summarize_metrics(episodes: list[EpisodeMetrics]) -> dict:
    if not episodes:
        return {
            "mean_return": 0.0,
            "mean_turns": 0.0,
            "win_rate": 0.0,
            "assassin_rate": 0.0,
            "friendly_reveal_rate": 0.0,
            "episodes": 0,
        }

    count = len(episodes)
    return {
        "mean_return": sum(item.episode_return for item in episodes) / count,
        "mean_turns": sum(item.turns for item in episodes) / count,
        "win_rate": sum(float(item.won) for item in episodes) / count,
        "assassin_rate": sum(float(item.assassin_hit) for item in episodes) / count,
        "friendly_reveal_rate": sum(
            item.friendly_revealed / max(item.friendly_total, 1) for item in episodes
        )
        / count,
        "episodes": count,
        "episode_details": [asdict(item) for item in episodes],
    }
