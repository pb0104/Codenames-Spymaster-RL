from __future__ import annotations

import argparse
import json

from src.training.pipeline_registry import make_pipeline_config, run_named_pipeline
from src.training.pipeline_utils import PROJECT_ROOT, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the greedy Codenames spymaster pipeline.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "base.yaml"),
        help="Path to a YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config = make_pipeline_config(base_config, "greedy")
    results = run_named_pipeline(config, pipeline_name="greedy")
    print(json.dumps(results.to_dict(), indent=2))


if __name__ == "__main__":
    main()
