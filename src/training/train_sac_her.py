from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.agents.bc_pretrain import generate_demonstrations
from src.agents.sac_agent import SACSpymasterAgent, SACTrainingConfig
from src.baselines.greedy_spymaster import GreedySpymaster
from src.env.board import BoardConfig
from src.env.game import CodenamesSpymasterEnv
from src.env.reward import RewardConfig
from src.evaluation.evaluate_agent import evaluate_agent
from src.utils.embeddings import EmbeddingStore, download_nltk_packages
from src.utils.seed import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        dimension=embedding_cfg.get("dim", 300),
        use_wordnet=embedding_cfg.get("use_wordnet", True),
        max_clues=embedding_cfg.get("max_clues", 12000),
        min_clue_length=embedding_cfg.get("min_clue_length", 3),
        max_clue_length=embedding_cfg.get("max_clue_length", 12),
        download_missing_nltk=embedding_cfg.get("download_nltk", False),
    )


def build_env_factory(
    config: dict[str, Any],
    embedding_store: EmbeddingStore,
    seed_offset: int = 0,
):
    data_cfg = config["data"]
    env_cfg = config["env"]
    reward_cfg = config["reward"]
    base_seed = config.get("seed", 0) + seed_offset
    call_state = {"count": 0}

    def factory():
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


def run_training_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    set_global_seed(config.get("seed"))
    embedding_store = build_embedding_store(config)
    env_factory = build_env_factory(config, embedding_store)
    eval_env_factory = build_env_factory(config, embedding_store, seed_offset=999)

    training_cfg = config["training"]
    bc_cfg = config["bc"]
    eval_cfg = config["evaluation"]

    demos = generate_demonstrations(
        env_factory=env_factory,
        num_episodes=bc_cfg["demo_episodes"],
        max_steps_per_episode=config["env"]["max_turns"],
    )

    sac_config = SACTrainingConfig(
        learning_rate=training_cfg["learning_rate"],
        buffer_size=training_cfg["buffer_size"],
        learning_starts=training_cfg["learning_starts"],
        batch_size=training_cfg["batch_size"],
        gamma=training_cfg["gamma"],
        tau=training_cfg["tau"],
        total_timesteps=training_cfg["total_timesteps"],
        policy_kwargs={"net_arch": training_cfg["net_arch"]},
        n_sampled_goal=training_cfg["n_sampled_goal"],
        goal_selection_strategy=training_cfg["goal_selection_strategy"],
    )
    train_env = env_factory()
    agent = SACSpymasterAgent(train_env, config=sac_config, use_her=training_cfg["use_her"])
    training_summary = agent.learn(
        total_timesteps=training_cfg["total_timesteps"],
        demos=demos if bc_cfg["enabled"] else None,
        bc_epochs=bc_cfg["pretrain_epochs"] if bc_cfg["enabled"] else 0,
        bc_batch_size=bc_cfg["batch_size"],
        bc_learning_rate=bc_cfg["learning_rate"],
        seed_buffer=bc_cfg["seed_replay_buffer"],
    )

    sac_metrics = evaluate_agent(
        agent,
        env_factory=eval_env_factory,
        episodes=eval_cfg["episodes"],
        deterministic=eval_cfg["deterministic"],
    )
    greedy_metrics = evaluate_agent(
        GreedySpymaster(),
        env_factory=eval_env_factory,
        episodes=eval_cfg["episodes"],
        deterministic=True,
    )

    output_cfg = config["output"]
    output_dir = resolve_project_path(output_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / output_cfg["metrics_file"]
    model_path = output_dir / output_cfg["model_file"]

    agent.save(str(model_path))
    results = {
        "config_name": config.get("name", "unnamed"),
        "demo_transitions": len(demos),
        "training_summary": training_summary,
        "sac_metrics": sac_metrics,
        "greedy_metrics": greedy_metrics,
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SAC+HER Codenames spymaster agent.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "smoke_test.yaml"),
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results = run_training_pipeline(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
