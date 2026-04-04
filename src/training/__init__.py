"""Training entry points, modular pipelines, and rollout visualization helpers."""

from src.training.pipeline_registry import (
    PIPELINE_REGISTRY,
    available_pipeline_names,
    make_pipeline_config,
    run_named_pipeline,
)
from src.training.pipeline_utils import PROJECT_ROOT, load_config
from src.training.rollout_visualizer import (
    RolloutFrame,
    RolloutTrace,
    capture_rollout_trace,
    save_rollout_gif,
)

__all__ = [
    "PIPELINE_REGISTRY",
    "PROJECT_ROOT",
    "RolloutFrame",
    "RolloutTrace",
    "available_pipeline_names",
    "capture_rollout_trace",
    "load_config",
    "make_pipeline_config",
    "run_named_pipeline",
    "save_rollout_gif",
]
