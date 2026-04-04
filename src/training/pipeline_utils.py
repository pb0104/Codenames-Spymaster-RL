from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from src.agents.bc_pretrain import DemonstrationTransition
from src.baselines.greedy_spymaster import GreedySpymaster
from src.env.board import BoardConfig
from src.env.game import CodenamesSpymasterEnv
from src.env.reward import RewardConfig
from src.evaluation.evaluate_agent import select_policy_action
from src.utils.embeddings import EmbeddingStore, download_nltk_packages
from src.utils.seed import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingRuntime:
    config: dict[str, Any]
    embedding_store: EmbeddingStore
    env_factory: Callable[[], CodenamesSpymasterEnv]
    eval_env_factory: Callable[[], CodenamesSpymasterEnv]


@dataclass
class PipelineRun:
    pipeline_name: str
    agent_label: str
    config: dict[str, Any]
    runtime: TrainingRuntime
    agent: Any
    training_summary: dict[str, Any]
    agent_metrics: dict[str, Any]
    greedy_metrics: dict[str, Any]
    demo_transitions: int = 0
    model_path: str | None = None
    metrics_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "config_name": self.config.get("name", self.pipeline_name),
            "agent_label": self.agent_label,
            "demo_transitions": self.demo_transitions,
            "training_summary": self.training_summary,
            "agent_metrics": self.agent_metrics,
            "greedy_metrics": self.greedy_metrics,
            "model_path": self.model_path,
            "metrics_path": self.metrics_path,
        }


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "extends" not in raw:
        return raw
    base_path = (config_path.parent / raw["extends"]).resolve()
    base_config = load_config(base_path)
    child_config = dict(raw)
    child_config.pop("extends")
    return deep_update(base_config, child_config)


def resolve_project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def build_embedding_store(config: dict[str, Any]) -> EmbeddingStore:
    data_cfg = config["data"]
    embedding_cfg = config["embedding"]
    if embedding_cfg.get("download_nltk", False):
        download_nltk_packages(embedding_cfg.get("nltk_packages", []))

    return EmbeddingStore.from_paths(
        board_words_path=resolve_project_path(data_cfg["board_words_path"]),
        clue_words_path=resolve_project_path(data_cfg["clue_words_path"]),
        dimension=embedding_cfg.get("dim"),
        use_wordnet=embedding_cfg.get("use_wordnet", True),
        max_clues=embedding_cfg.get("max_clues", 12000),
        min_clue_length=embedding_cfg.get("min_clue_length", 3),
        max_clue_length=embedding_cfg.get("max_clue_length", 12),
        download_missing_nltk=embedding_cfg.get("download_nltk", False),
        model_name=embedding_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        board_prompt=embedding_cfg.get("board_prompt", "board word"),
        clue_prompt=embedding_cfg.get("clue_prompt", "clue word"),
    )


def build_env_factory(
    config: dict[str, Any],
    embedding_store: EmbeddingStore,
    seed_offset: int = 0,
) -> Callable[[], CodenamesSpymasterEnv]:
    data_cfg = config["data"]
    env_cfg = config["env"]
    reward_cfg = config["reward"]
    base_seed = config.get("seed", 0) + seed_offset
    call_state = {"count": 0}

    def factory() -> CodenamesSpymasterEnv:
        env_seed = base_seed + call_state["count"]
        call_state["count"] += 1
        board_config = BoardConfig(
            rows=env_cfg["rows"],
            cols=env_cfg["cols"],
            num_friendly=env_cfg["num_friendly"],
            num_opponent=env_cfg["num_opponent"],
            num_neutral=env_cfg["num_neutral"],
            num_assassin=env_cfg["num_assassin"],
            seed=env_seed,
        )
        return CodenamesSpymasterEnv(
            board_words_path=str(resolve_project_path(data_cfg["board_words_path"])),
            clue_words_path=str(resolve_project_path(data_cfg["clue_words_path"])),
            embedding_store=embedding_store,
            board_config=board_config,
            reward_config=RewardConfig(
                turn_penalty=reward_cfg["turn_penalty"],
                opponent_penalty=reward_cfg.get("opponent_penalty", -3.0),
                neutral_penalty=reward_cfg.get("neutral_penalty", -1.5),
                assassin_penalty=reward_cfg["assassin_penalty"],
                shaped_weight=reward_cfg["shaped_weight"],
            ),
            embedding_dim=embedding_store.dimension,
            max_turns=env_cfg["max_turns"],
            max_clue_count=env_cfg["max_clue_count"],
            goal_size=env_cfg["goal_size"],
            seed=env_seed,
        )

    return factory


def prepare_runtime(config: dict[str, Any]) -> TrainingRuntime:
    set_global_seed(config.get("seed"))
    embedding_store = build_embedding_store(config)
    return TrainingRuntime(
        config=config,
        embedding_store=embedding_store,
        env_factory=build_env_factory(config, embedding_store),
        eval_env_factory=build_env_factory(config, embedding_store, seed_offset=999),
    )


def clone_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {key: value.copy() for key, value in obs.items()}


def collect_demonstrations(
    env_factory: Callable[[], CodenamesSpymasterEnv],
    policy: Any,
    num_episodes: int,
    *,
    max_steps_per_episode: int | None = None,
    deterministic: bool = True,
) -> list[DemonstrationTransition]:
    transitions: list[DemonstrationTransition] = []

    for _ in range(num_episodes):
        env = env_factory()
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action = select_policy_action(policy, obs, env, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transitions.append(
                DemonstrationTransition(
                    obs=clone_observation(obs),
                    action=action.copy(),
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


def build_demo_policy(config: dict[str, Any]) -> GreedySpymaster:
    return GreedySpymaster(max_count=config["env"]["max_clue_count"])


def persist_pipeline_run(run: PipelineRun) -> PipelineRun:
    output_cfg = run.config.get("output", {})
    output_dir = resolve_project_path(output_cfg.get("output_dir", "notebooks/artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_name = output_cfg.get("metrics_file", f"{run.pipeline_name}_metrics.json")
    metrics_path = output_dir / metrics_name
    model_name = output_cfg.get("model_file", f"{run.pipeline_name}_model.zip")
    model_path = output_dir / model_name

    if hasattr(run.agent, "save"):
        run.agent.save(str(model_path))
        run.model_path = str(model_path)

    run.metrics_path = str(metrics_path)
    metrics_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return run
