#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$PROJECT_ROOT/configs/smoke_test.yaml}"

python -m src.training.train_sac_her --config "$CONFIG_PATH"
