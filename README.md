# Codenames Spymaster RL

Goal-conditioned reinforcement learning for the **Codenames spymaster** task.

This repo is a research pipeline for learning one-word clues from semantic embeddings. It includes a Gymnasium environment, a greedy semantic clueing baseline, behavioral cloning from demonstrations, SAC + HER training, reward shaping, evaluation utilities, rollout GIF generation, and notebook experiments.

## At a Glance

- **Task**: learn a spymaster policy that gives a clue and count for a hidden-role Codenames board.
- **Representation**: board words and clue candidates are embedded with `sentence-transformers/all-MiniLM-L6-v2`.
- **Model lineup**: `Greedy`, `BC-only`, `BC + SAC + HER`, and `BC + SAC + HER + Reward Shaping`.
- **Artifacts**: metrics JSON, model checkpoints, rollout GIFs, notebooks, and a final report are already in the repo.
- **Current status**: the full infrastructure works, but the checked-in learned agents still do not beat the greedy baseline.

## Project Structure

Main repository structure as it exists now. This omits `.git`, `.DS_Store`, and `__pycache__` noise.

```text
Codenames-Spymaster-RL/
├── .vscode/
│   └── settings.json
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── smoke_test.yaml
│   ├── training_pipeline_3x3.yaml
│   ├── training_pipeline_4x4.yaml
│   └── training_pipeline_5x5.yaml
├── data/
│   └── raw/
│       ├── codenames_words.txt
│       ├── common_words copy.txt
│       ├── common_words.txt
│       └── hints.txt
├── experiments/
│   ├── logs/
│   │   └── log.txt
│   └── models/
│       └── model.pkl
├── notebooks/
│   ├── 5x5_board_demo.ipynb
│   ├── final_generalization.ipynb
│   ├── phase1_environment_testing.ipynb
│   ├── phase2_agent_pipeline_testing.ipynb
│   ├── run_codenames.executed.ipynb
│   ├── run_codenames.ipynb
│   ├── training_pipeline_rollout_debug.ipynb
│   ├── training_pipeline_rollout_debug_3x3.ipynb
│   ├── training_pipeline_rollout_debug_4x4.ipynb
│   └── training_pipeline_rollout_debug_5x5.ipynb
├── reports/
│   ├── Codenames Final Report.docx
│   ├── Codenames Final Report.pdf
│   ├── Codenames Final Report.tex
│   └── report-template.tex
├── scripts/
│   └── run_training.sh
└── src/
    ├── __init__.py
    ├── agents/
    │   ├── __init__.py
    │   ├── bc_pretrain.py
    │   ├── random_agent.py
    │   └── sac_agent.py
    ├── baselines/
    │   ├── __init__.py
    │   └── greedy_spymaster.py
    ├── env/
    │   ├── __init__.py
    │   ├── board.py
    │   ├── game.py
    │   ├── reward.py
    │   └── visualization.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── evaluate_agent.py
    │   └── metrics.py
    ├── training/
    │   ├── __init__.py
    │   ├── pipeline_registry.py
    │   ├── pipeline_utils.py
    │   ├── rollout_visualizer.py
    │   ├── run_ablation.py
    │   ├── train_greedy.py
    │   ├── train_greedy_bc_pretrain.py
    │   └── train_sac_her.py
    └── utils/
        ├── __init__.py
        ├── embeddings.py
        ├── seed.py
        └── similarity.py
```


## Current Project Status

The most important GitHub-facing takeaway is that this repo is a **working RL testbed**, not a solved benchmark.

From the checked-in 5x5 generalization artifacts in [`notebooks/artifacts/final_generalization/trained_agent/sac_her_reward_metrics.json`](notebooks/artifacts/final_generalization/trained_agent/sac_her_reward_metrics.json):

| Method | Win rate | Assassin rate | Friendly reveal rate |
| --- | ---: | ---: | ---: |
| Greedy | 1.00 | 0.00 | 1.00 |
| BC + SAC + HER + Reward Shaping | 0.00 | 1.00 | 0.41 |

That means the repo is currently strongest as:

- a reproducible environment for the Codenames spymaster task
- a baseline-vs-learning comparison framework
- a place to iterate on reward design, representations, and imitation warm starts
- a notebook-friendly project for debugging RL on semantic decision problems

## Model Lineup

These are the four methods the repo should present:

- **Greedy**
  A non-learning semantic baseline in [`src/baselines/greedy_spymaster.py`](src/baselines/greedy_spymaster.py). It chooses the clue that maximizes a cosine-margin heuristic over friendly versus bad words.
- **BC-only**
  Behavioral cloning from greedy demonstrations, implemented by the `greedy_bc_pretrain` pipeline in [`src/training/train_greedy_bc_pretrain.py`](src/training/train_greedy_bc_pretrain.py). This stops after imitation and does not do RL fine-tuning.
- **BC + SAC + HER**
  The imitation-warm-start RL pipeline in [`src/training/train_sac_her.py`](src/training/train_sac_her.py), using SAC with HER and no reward shaping.
- **BC + SAC + HER + Reward Shaping**
  The full method, exposed as the `sac_her_reward` pipeline in [`src/training/pipeline_registry.py`](src/training/pipeline_registry.py), which keeps BC and HER and turns on shaped reward.


## Problem Setup

Each episode builds a Codenames board and gives the spymaster:

- board-word embeddings
- pairwise similarity features
- hidden role labels
- remaining-word mask
- a desired goal vector for HER

The action is a continuous vector decoded into:

- a legal one-word clue
- a count from `1` to `max_clue_count`

The teammate is a fixed greedy guesser, so the learning problem is focused on **clue generation**, not joint play.

## Quick Start

Use Python 3.12 if possible.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the smallest end-to-end pipeline check:

```bash
python -m src.training.train_sac_her --config configs/smoke_test.yaml
```

Or use the helper script:

```bash
bash scripts/run_training.sh
```

The first run may download sentence-transformer weights.

## Baselines and Training

Greedy:

```bash
python -m src.training.train_greedy --config configs/base.yaml
```

BC-only:

```bash
python -m src.training.train_greedy_bc_pretrain --config configs/base.yaml
```

BC + SAC + HER:

```bash
python -m src.training.train_sac_her --config configs/base.yaml
```

BC + SAC + HER + Reward Shaping:

```bash
python - <<'PY'
from src.training import load_config, make_pipeline_config, run_named_pipeline

base = load_config("configs/base.yaml")
config = make_pipeline_config(base, "sac_her_reward")
result = run_named_pipeline(config, pipeline_name="sac_her_reward")
print(result.to_dict())
PY
```

## Outputs You Get

- `experiments/logs/`: config-driven logs and model checkpoints
- `notebooks/artifacts/`: rollout GIFs, saved metrics, and generalization artifacts
- [`reports/Codenames Final Report.pdf`](reports/Codenames%20Final%20Report.pdf): final write-up

Metrics files include both the evaluated method and a greedy comparison, which makes it easy to see whether a learned pipeline actually improved over the baseline.

## Recommended Entry Points

- [`src/training/train_greedy.py`](src/training/train_greedy.py): Greedy baseline runner
- [`src/training/train_greedy_bc_pretrain.py`](src/training/train_greedy_bc_pretrain.py): BC-only pipeline
- [`src/training/train_sac_her.py`](src/training/train_sac_her.py): BC + SAC + HER entry point
- [`src/training/pipeline_registry.py`](src/training/pipeline_registry.py): named pipeline definitions, including `sac_her_reward`
- [`notebooks/run_codenames.ipynb`](notebooks/run_codenames.ipynb): smoke-test walkthrough
- [`notebooks/final_generalization.ipynb`](notebooks/final_generalization.ipynb): held-out 5x5 evaluation notebook

## Data and Config Notes

- Board words come from `data/raw/codenames_words.txt`.
- Clue candidates come from `data/raw/common_words.txt` and are sanitized into legal single-word clues.
- `configs/smoke_test.yaml` is for validation, not performance.
- `configs/base.yaml` is the main 5x5 setup.
- `configs/training_pipeline_3x3.yaml`, `training_pipeline_4x4.yaml`, and `training_pipeline_5x5.yaml` support board-size experiments.

## If You Are Extending This Repo

The highest-value next steps are probably:

- better clue-vocabulary filtering or retrieval
- better action decoding than nearest-embedding clue selection
- improved reward design around risky clues
- stronger imitation warm starts
- evaluation on larger held-out board sets
