"""
train.py  –  Run ablation conditions A–D from the proposal.

Conditions
----------
A  Vanilla PPO                          (baseline replication)
B  SAC + HER                            (Siu 2022 replication → should still fail)
C  SAC + HER + Reward Shaping           (shaping ablation)
D  SAC + HER + Shaping + BC (Full)      (full proposed method)

Usage
-----
    python train.py --condition D --seeds 5 --total-steps 500000

Requirements: stable-baselines3 >= 2.0, gymnasium, torch, numpy
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

# ── Local imports ──────────────────────────────────────────────────────────────
from environment   import CodenamesEnv
from reward_shaping import PotentialShapingWrapper
from guesser        import GreedySpymaster, collect_bc_demonstrations
from bc_pretrain    import bc_pretrain, seed_her_buffer
from embeddings     import GloveLoader
from eval           import evaluate_agent, evaluate_greedy_baseline

# ── SB3 imports ───────────────────────────────────────────────────────────────
from stable_baselines3             import SAC, PPO
from stable_baselines3.her         import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor


# ── Callbacks ─────────────────────────────────────────────────────────────────

class WinRateCallback(BaseCallback):
    """Log win rate against greedy baseline every eval_freq steps."""

    def __init__(self, eval_env, greedy_spymaster, eval_freq=10_000,
                 n_eval_episodes=50, verbose=1):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.greedy          = greedy_spymaster
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._results        = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            agent_wins, agent_turns = evaluate_agent(
                self.model, self.eval_env, self.n_eval_episodes
            )
            greedy_wins, greedy_turns = evaluate_greedy_baseline(
                self.greedy, self.eval_env, self.n_eval_episodes
            )
            record = {
                "step":          self.n_calls,
                "agent_win_rate":   agent_wins,
                "agent_avg_turns":  agent_turns,
                "greedy_win_rate":  greedy_wins,
                "greedy_avg_turns": greedy_turns,
                "beats_greedy":     agent_wins > greedy_wins,
            }
            self._results.append(record)
            if self.verbose:
                print(f"  [Step {self.n_calls}] "
                      f"Agent WR={agent_wins:.2%}  "
                      f"Greedy WR={greedy_wins:.2%}  "
                      f"{'✓ BEATS GREEDY' if record['beats_greedy'] else '✗'}")
        return True


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(
    sim_matrix: np.ndarray,
    board_indices: np.ndarray,
    vocab_size: int,
    use_shaping: bool = False,
    gamma: float = 0.99,
    seed: Optional[int] = None,
):
    env = CodenamesEnv(
        sim_matrix=sim_matrix,
        board_indices=board_indices,
        vocab_size=vocab_size,
        seed=seed,
    )
    env = Monitor(env)
    if use_shaping:
        env = PotentialShapingWrapper(env, gamma=gamma)
    return env


# ── Per-condition training ─────────────────────────────────────────────────────

def run_condition(
    condition: str,
    sim_matrix: np.ndarray,
    board_indices: np.ndarray,
    vocab_size: int,
    total_steps: int = 500_000,
    seed: int = 0,
    output_dir: str = "experiments",
    bc_games: int = 1000,
    eval_freq: int = 10_000,
    device: str = "auto",
):
    assert condition in ("A", "B", "C", "D"), f"Unknown condition: {condition}"

    use_her     = condition in ("B", "C", "D")
    use_shaping = condition in ("C", "D")
    use_bc      = condition == "D"
    use_ppo     = condition == "A"

    run_dir = Path(output_dir) / f"cond_{condition}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Condition {condition}  |  seed={seed}")
    print(f"  HER={use_her}  Shaping={use_shaping}  BC={use_bc}  PPO={use_ppo}")
    print(f"{'='*60}\n")

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = make_env(sim_matrix, board_indices, vocab_size,
                         use_shaping=use_shaping, seed=seed)
    eval_env  = make_env(sim_matrix, board_indices, vocab_size,
                         use_shaping=False, seed=seed + 9999)

    greedy = GreedySpymaster(sim_matrix, board_indices, vocab_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=[256, 256])

    if use_ppo:
        model = PPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )
    else:
        replay_buffer_kwargs = {}
        if use_her:
            replay_buffer_kwargs = dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            )

        model = SAC(
            "MultiInputPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            replay_buffer_class=HerReplayBuffer if use_her else None,
            replay_buffer_kwargs=replay_buffer_kwargs if use_her else {},
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=device,
        )

    # ── BC pretraining + buffer seeding ───────────────────────────────────────
    if use_bc:
        print("Collecting BC demonstrations…")
        demo_env = make_env(sim_matrix, board_indices, vocab_size,
                            use_shaping=False, seed=seed + 1)
        demos = collect_bc_demonstrations(
            demo_env, greedy, n_games=bc_games, seed=seed
        )
        print(f"Pretraining actor via BC ({len(demos)} transitions)…")
        bc_pretrain(model, demos, n_epochs=10, device=device)
        seed_her_buffer(model, demos)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    win_rate_cb = WinRateCallback(
        eval_env=eval_env,
        greedy_spymaster=greedy,
        eval_freq=eval_freq,
        n_eval_episodes=50,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="model",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_steps,
        callback=[win_rate_cb, checkpoint_cb],
        progress_bar=True,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(str(run_dir / "final_model"))

    results_path = run_dir / "win_rate_log.json"
    with open(results_path, "w") as f:
        json.dump(win_rate_cb._results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return win_rate_cb._results


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Codenames Spymaster RL")
    parser.add_argument("--condition",    type=str, default="D",
                        choices=["A", "B", "C", "D"])
    parser.add_argument("--seeds",        type=int, default=5,
                        help="Number of seeds to run")
    parser.add_argument("--total-steps",  type=int, default=500_000)
    parser.add_argument("--bc-games",     type=int, default=1000)
    parser.add_argument("--eval-freq",    type=int, default=10_000)
    parser.add_argument("--output-dir",   type=str, default="experiments")
    parser.add_argument("--glove-path",   type=str,
                        default="data/glove.840B.300d.txt")
    parser.add_argument("--words-path",   type=str,
                        default="data/raw/wordlist-eng.txt")
    parser.add_argument("--device",       type=str, default="auto")
    return parser.parse_args()


def load_data(args):
    """Load GloVe and build sim matrix."""
    import nltk
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words

    # Load board words
    with open(args.words_path) as f:
        all_board_words = [w.strip().lower() for w in f if w.strip()]

    # Sample 25 for a board (or load a fixed board for reproducibility)
    rng = np.random.default_rng(0)
    board_words = list(rng.choice(all_board_words, size=25, replace=False))

    # Clue vocab: NLTK English words filtered to reasonable length
    clue_vocab = [w.lower() for w in nltk_words.words()
                  if 3 <= len(w) <= 12 and w.isalpha()][:10_000]

    loader = GloveLoader(args.glove_path)
    loader.load(verbose=True)

    vocab, board_indices, sim_matrix = loader.build_sim_matrix(
        board_words=board_words,
        clue_vocab=clue_vocab,
    )
    return sim_matrix, board_indices, len(vocab)


if __name__ == "__main__":
    args = load_args = parse_args()

    sim_matrix, board_indices, vocab_size = load_data(args)

    all_results = {}
    for seed in range(args.seeds):
        results = run_condition(
            condition   = args.condition,
            sim_matrix  = sim_matrix,
            board_indices = board_indices,
            vocab_size  = vocab_size,
            total_steps = args.total_steps,
            seed        = seed,
            output_dir  = args.output_dir,
            bc_games    = args.bc_games,
            eval_freq   = args.eval_freq,
            device      = args.device,
        )
        all_results[seed] = results

    # Summary
    print("\n" + "="*60)
    print(f"CONDITION {args.condition}  SUMMARY ({args.seeds} seeds)")
    final_agent_wrs  = [r[-1]["agent_win_rate"]  for r in all_results.values() if r]
    final_greedy_wrs = [r[-1]["greedy_win_rate"] for r in all_results.values() if r]
    print(f"  Final Agent  WR: {np.mean(final_agent_wrs):.2%} ± {np.std(final_agent_wrs):.2%}")
    print(f"  Final Greedy WR: {np.mean(final_greedy_wrs):.2%}")
    beats = sum(a > g for a, g in zip(final_agent_wrs, final_greedy_wrs))
    print(f"  Beats greedy in {beats}/{args.seeds} seeds")
