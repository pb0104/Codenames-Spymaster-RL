from __future__ import annotations

import argparse
import json
from typing import Any

from src.training.pipeline_registry import make_pipeline_config, run_named_pipeline
from src.training.pipeline_utils import PROJECT_ROOT, load_config


def run_training_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    pipeline_config = make_pipeline_config(config, "sac_her")
    return run_named_pipeline(pipeline_config, pipeline_name="sac_her").to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SAC+HER Codenames spymaster agent.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "base.yaml"),
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
