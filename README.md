# Codenames Spymaster RL

This project implements a reinforcement learning environment for the Codenames spymaster task, following the proposal structure for:

- reward shaping
- behavioral cloning pretraining
- SAC
- HER
- notebook-based end-to-end smoke testing

The current codebase includes:

- a goal-conditioned Gymnasium environment for the spymaster
- fixed 400-word Codenames board vocabulary in `data/raw/codenames_words.txt`
- curated clue vocabulary in `data/raw/common_words.txt`
- sentence-transformer semantic embeddings for board words and clues
- a greedy cosine-margin spymaster baseline
- demonstration generation for behavioral cloning
- SAC + HER training through Stable-Baselines3
- a runnable notebook in `notebooks/run_codenames.ipynb`

## Project Layout

```text
Codenames-Spymaster-RL/
├── configs/
│   ├── base.yaml
│   └── smoke_test.yaml
├── data/raw/
│   ├── codenames_words.txt
│   └── common_words.txt
├── notebooks/
│   └── run_codenames.ipynb
├── scripts/
│   └── run_training.sh
└── src/
    ├── agents/
    ├── baselines/
    ├── env/
    ├── evaluation/
    ├── training/
    └── utils/
```

## What The Environment Does

Each episode builds a Codenames board and gives the spymaster:

- board word embeddings
- a pairwise board-word similarity matrix
- role labels for each board word
- a remaining-word mask
- a goal vector for HER

An action is:

- a clue word
- a count from 1 to 9

The guesser is a fixed greedy oracle that ranks unrevealed board words by cosine similarity to the clue and keeps guessing until it hits a non-friendly word or exhausts the count.

Rewards include:

- sparse turn penalty
- assassin penalty
- dense clue-margin reward shaping
- goal-conditioned reward for HER

## How To Run It

Run everything from the project root:

```bash
cd Codenames-Spymaster-RL
```

### 1. Create and activate a virtual environment

Use Python 3.12 for this project. `torch` is the dependency that matters here, and it does not install cleanly on Python 3.14 in this setup.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

You should see `Python 3.12.x`.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run the smoke test

This is the quickest end-to-end check.

```bash
python -m src.training.train_sac_her --config configs/smoke_test.yaml
```

You can also use the helper script:

```bash
bash scripts/run_training.sh
```

Artifacts are written to:

- `experiments/logs/training_metrics.json`
- `experiments/logs/sac_her_model.zip`

### 4. Run the larger default config

```bash
python -m src.training.train_sac_her --config configs/base.yaml
```

### 5. Run the notebook

Start Jupyter from the project root:

```bash
jupyter lab
```

Then open:

- `notebooks/run_codenames.ipynb`

The notebook walks through:

- loading the config
- building the clue vocabulary and embeddings
- resetting the environment
- plotting a board
- evaluating the greedy baseline
- generating BC demonstrations
- running the smoke training pipeline

### 6. Run ablations

```bash
python -m src.training.run_ablation --config configs/smoke_test.yaml
```

This runs the predefined condition variants in `src/training/run_ablation.py`.

## Configs

### `configs/smoke_test.yaml`

Use this first. It is intentionally tiny and is only meant to verify that:

- the environment resets correctly
- clue generation works
- demonstrations can be collected
- SAC + HER can train
- evaluation and artifact writing work

Do not expect the smoke test to beat the greedy baseline.

### `configs/base.yaml`

This is the default fuller run. It includes:

- 5x5 board
- 8 friendly, 8 opponent, 8 neutral, 1 assassin
- reward shaping enabled
- BC enabled
- HER enabled

If you want longer training, increase:

- `training.total_timesteps`
- `bc.demo_episodes`
- `bc.pretrain_epochs`
- `evaluation.episodes`

## Data

### Board words

`data/raw/codenames_words.txt` contains the fixed 400-word Codenames word list used for board construction.

### Clue words

`data/raw/common_words.txt` is the clue vocabulary used by the environment and agent pipeline.

## Main Entry Points

- `src/env/game.py`
  Goal-conditioned spymaster environment.
- `src/env/reward.py`
  Reward shaping and HER-compatible reward computation.
- `src/utils/embeddings.py`
  Board/clue embeddings and clue-vocabulary generation.
- `src/baselines/greedy_spymaster.py`
  Greedy cosine-margin baseline used for demonstrations.
- `src/agents/bc_pretrain.py`
  Demonstration generation for BC.
- `src/agents/sac_agent.py`
  SAC + HER wrapper and BC warm start.
- `src/training/train_sac_her.py`
  Main training entry point.
- `src/training/run_ablation.py`
  Ablation runner.

## Notes

- The smoke test is only a pipeline check, not a performance benchmark.
- The greedy baseline should currently outperform the smoke-test SAC model because the smoke config is intentionally short.
- If you want better clue quality, curate `common_words.txt` or swap in a different clue vocabulary file in config.

## References

- Proposal reference in this repo context: RL for Codenames Spymaster with reward shaping, BC, SAC, and HER.
- Codenames 400-word list is based on the standard public word pool commonly used in prior work.
