from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from src.training.train_sac_her import PROJECT_ROOT, load_config, run_training_pipeline


CONDITIONS = {
    "sac_her": {
        "name": "sac_her",
        "reward": {"shaped_weight": 0.0},
        "bc": {"enabled": False},
    },
    "sac_her_reward": {
        "name": "sac_her_reward",
        "reward": {"shaped_weight": 1.0},
        "bc": {"enabled": False},
    },
    "full_method": {
        "name": "full_method",
        "reward": {"shaped_weight": 1.0},
        "bc": {"enabled": True},
    },
}


def deep_update(base: dict, updates: dict) -> dict:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation conditions for the project.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "base.yaml"),
        help="Base config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    results = {}
    for name, overrides in CONDITIONS.items():
        config = deep_update(base_config, overrides)
        config["name"] = name
        results[name] = run_training_pipeline(config)
    output_path = Path(args.config).resolve().parent / "ablation_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
