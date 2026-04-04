from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from src.agents.sac_agent import SACSpymasterAgent, SACTrainingConfig
from src.evaluation.evaluate_agent import evaluate_agent
from src.training.pipeline_utils import (
    PipelineRun,
    TrainingRuntime,
    build_demo_policy,
    collect_demonstrations,
    deep_update,
    persist_pipeline_run,
    prepare_runtime,
)


PipelineRunner = Callable[[TrainingRuntime], PipelineRun]


PIPELINE_OVERRIDES: dict[str, dict[str, Any]] = {
    "sac_her": {
        "name": "sac_her",
        "bc": {"enabled": True},
        "training": {"use_her": True},
        "output": {
            "metrics_file": "sac_her_metrics.json",
            "model_file": "sac_her_model.zip",
        },
    },
    "greedy": {
        "name": "greedy",
        "bc": {"enabled": False},
        "output": {
            "metrics_file": "greedy_metrics.json",
            "model_file": "greedy_model.zip",
        },
    },
    "greedy_bc_pretrain": {
        "name": "greedy_bc_pretrain",
        "bc": {"enabled": True},
        "training": {"use_her": False},
        "output": {
            "metrics_file": "greedy_bc_pretrain_metrics.json",
            "model_file": "greedy_bc_pretrain_model.zip",
        },
    },
}


def available_pipeline_names() -> list[str]:
    return sorted(PIPELINE_OVERRIDES)


def make_pipeline_config(
    base_config: dict[str, Any],
    pipeline_name: str,
    *,
    output_dir: str | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if pipeline_name not in PIPELINE_OVERRIDES:
        raise KeyError(f"Unknown pipeline: {pipeline_name}")

    overrides = deepcopy(PIPELINE_OVERRIDES[pipeline_name])
    if output_dir is not None:
        overrides.setdefault("output", {})["output_dir"] = output_dir
    if extra_overrides:
        overrides = deep_update(overrides, extra_overrides)
    return deep_update(base_config, overrides)


def build_sac_config(training_cfg: dict[str, Any]) -> SACTrainingConfig:
    return SACTrainingConfig(
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


def evaluate_against_greedy(runtime: TrainingRuntime, episodes: int) -> dict[str, Any]:
    return evaluate_agent(
        build_demo_policy(runtime.config),
        env_factory=runtime.eval_env_factory,
        episodes=episodes,
        deterministic=True,
    )


def run_sac_her_pipeline(runtime: TrainingRuntime) -> PipelineRun:
    config = runtime.config
    training_cfg = config["training"]
    bc_cfg = config["bc"]
    eval_cfg = config["evaluation"]

    demos = []
    if bc_cfg["enabled"]:
        demos = collect_demonstrations(
            runtime.env_factory,
            build_demo_policy(config),
            num_episodes=bc_cfg["demo_episodes"],
            max_steps_per_episode=config["env"]["max_turns"],
        )

    agent = SACSpymasterAgent(
        runtime.env_factory(),
        config=build_sac_config(training_cfg),
        use_her=training_cfg["use_her"],
    )
    training_summary = agent.learn(
        total_timesteps=training_cfg["total_timesteps"],
        demos=demos if bc_cfg["enabled"] else None,
        bc_epochs=bc_cfg["pretrain_epochs"] if bc_cfg["enabled"] else 0,
        bc_batch_size=bc_cfg["batch_size"],
        bc_learning_rate=bc_cfg["learning_rate"],
        seed_buffer=bc_cfg["seed_replay_buffer"],
    )

    run = PipelineRun(
        pipeline_name="sac_her",
        agent_label="SAC + HER",
        config=config,
        runtime=runtime,
        agent=agent,
        training_summary=training_summary,
        agent_metrics=evaluate_agent(
            agent,
            env_factory=runtime.eval_env_factory,
            episodes=eval_cfg["episodes"],
            deterministic=eval_cfg["deterministic"],
        ),
        greedy_metrics=evaluate_against_greedy(runtime, eval_cfg["episodes"]),
        demo_transitions=len(demos),
    )
    return persist_pipeline_run(run)


def run_greedy_pipeline(runtime: TrainingRuntime) -> PipelineRun:
    eval_cfg = runtime.config["evaluation"]
    agent = build_demo_policy(runtime.config)
    metrics = evaluate_agent(
        agent,
        env_factory=runtime.eval_env_factory,
        episodes=eval_cfg["episodes"],
        deterministic=True,
    )
    run = PipelineRun(
        pipeline_name="greedy",
        agent_label="Greedy Spymaster",
        config=runtime.config,
        runtime=runtime,
        agent=agent,
        training_summary={
            "mode": "no_training",
            "notes": "Direct baseline evaluation.",
        },
        agent_metrics=metrics,
        greedy_metrics=metrics,
        demo_transitions=0,
    )
    return persist_pipeline_run(run)


def run_greedy_bc_pretrain_pipeline(runtime: TrainingRuntime) -> PipelineRun:
    config = runtime.config
    training_cfg = config["training"]
    bc_cfg = config["bc"]
    eval_cfg = config["evaluation"]

    demo_policy = build_demo_policy(config)
    demos = collect_demonstrations(
        runtime.env_factory,
        demo_policy,
        num_episodes=bc_cfg["demo_episodes"],
        max_steps_per_episode=config["env"]["max_turns"],
    )

    agent = SACSpymasterAgent(
        runtime.env_factory(),
        config=build_sac_config(training_cfg),
        use_her=False,
    )
    bc_losses = agent.bc_pretrain(
        demos,
        epochs=bc_cfg["pretrain_epochs"],
        batch_size=bc_cfg["batch_size"],
        learning_rate=bc_cfg["learning_rate"],
    )
    if bc_cfg["seed_replay_buffer"]:
        agent.seed_replay_buffer(demos)

    run = PipelineRun(
        pipeline_name="greedy_bc_pretrain",
        agent_label="Greedy + BC Pretrain",
        config=config,
        runtime=runtime,
        agent=agent,
        training_summary={
            "bc_losses": bc_losses,
            "replay_buffer_seeded": bool(bc_cfg["seed_replay_buffer"]),
            "rl_finetune_timesteps": 0,
        },
        agent_metrics=evaluate_agent(
            agent,
            env_factory=runtime.eval_env_factory,
            episodes=eval_cfg["episodes"],
            deterministic=eval_cfg["deterministic"],
        ),
        greedy_metrics=evaluate_against_greedy(runtime, eval_cfg["episodes"]),
        demo_transitions=len(demos),
    )
    return persist_pipeline_run(run)


PIPELINE_REGISTRY: dict[str, PipelineRunner] = {
    "greedy": run_greedy_pipeline,
    "greedy_bc_pretrain": run_greedy_bc_pretrain_pipeline,
    "sac_her": run_sac_her_pipeline,
}


def run_named_pipeline(
    config: dict[str, Any],
    *,
    pipeline_name: str,
) -> PipelineRun:
    if pipeline_name not in PIPELINE_REGISTRY:
        raise KeyError(f"Unknown pipeline: {pipeline_name}")
    runtime = prepare_runtime(config)
    return PIPELINE_REGISTRY[pipeline_name](runtime)
