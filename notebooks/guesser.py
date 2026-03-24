"""
Greedy Spymaster (Kim et al. 2019 baseline).

For each possible clue word c and count n, scores the board as:
    score(c, n) = sum of top-n cosine similarities to friendly words
                 − penalty for proximity to bad words

Also used as the behavioral cloning (BC) demonstrator:
  play_game() returns a list of (obs, action) trajectory pairs.
"""

import numpy as np
from typing import List, Tuple, Optional


class GreedySpymaster:
    """
    Deterministic greedy Spymaster.

    Parameters
    ----------
    sim_matrix    : (V, V) cosine similarity matrix
    board_indices : (25,) vocab indices of board words
    vocab_size    : int
    bad_penalty   : weight for penalising proximity to bad words
    top_k         : consider only top_k clue candidates for speed
    """

    def __init__(
        self,
        sim_matrix: np.ndarray,
        board_indices: np.ndarray,
        vocab_size: int,
        bad_penalty: float = 1.5,
        top_k: int = 1000,
    ):
        self.sim_matrix    = sim_matrix.astype(np.float32)
        self.board_indices = board_indices
        self.vocab_size    = vocab_size
        self.bad_penalty   = bad_penalty
        self.top_k         = top_k
        self.n_counts      = 9

    def select_action(
        self,
        labels: np.ndarray,          # (25,) int  FRIENDLY=0,…
        remaining_mask: np.ndarray,  # (25,) bool
    ) -> Tuple[int, int]:
        """
        Return (clue_vocab_idx, count) that maximises the greedy score.
        """
        friendly_pos = np.where((labels == 0) & remaining_mask)[0]
        bad_pos      = np.where((labels != 0) & remaining_mask)[0]

        if len(friendly_pos) == 0:
            # No friendly words left — dummy action
            return 0, 1

        friendly_board = self.board_indices[friendly_pos]
        bad_board      = self.board_indices[bad_pos]

        # sim(vocab_word, board_word)
        sim_friendly = self.sim_matrix[:, friendly_board]   # (V, nf)
        sim_bad      = self.sim_matrix[:, bad_board] if len(bad_board) > 0 \
                       else np.zeros((self.vocab_size, 0), dtype=np.float32)

        # Quick filter: pre-select by mean similarity to friendly words
        mean_friendly = sim_friendly.mean(axis=1)           # (V,)
        candidates    = np.argpartition(mean_friendly, -self.top_k)[-self.top_k:]

        best_score = -np.inf
        best_clue  = candidates[0]
        best_count = 1

        for c in candidates:
            sims_f = sim_friendly[c]       # (nf,)
            sims_b = sim_bad[c] if sim_bad.shape[1] > 0 else np.array([])

            # Sort friendly sims descending
            sorted_f = np.sort(sims_f)[::-1]
            max_bad  = sims_b.max() if len(sims_b) > 0 else 0.0

            for n in range(1, min(self.n_counts, len(sorted_f)) + 1):
                # Score = sum of top-n friendly sims − bad_penalty × max_bad
                score = sorted_f[:n].sum() - self.bad_penalty * max_bad
                if score > best_score:
                    best_score = score
                    best_clue  = c
                    best_count = n

        return int(best_clue), int(best_count)

    def action_to_flat(self, clue_idx: int, count: int) -> int:
        """Convert (clue_idx, count) to the flat action integer used by the env."""
        return clue_idx * self.n_counts + (count - 1)


# ── BC data collection ────────────────────────────────────────────────────────

def collect_bc_demonstrations(
    env,                          # CodenamesEnv (or wrapper)
    spymaster: GreedySpymaster,
    n_games: int = 1000,
    seed: Optional[int] = 42,
) -> List[Tuple[dict, int, float, dict, bool]]:
    """
    Roll out the greedy Spymaster for n_games and collect transitions.

    Returns
    -------
    demos : list of (obs, action, reward, next_obs, done) tuples
            suitable for seeding a replay buffer.
    """
    demos = []
    rng   = np.random.default_rng(seed)

    for game_idx in range(n_games):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        done   = False

        while not done:
            # Greedy action
            clue_idx, count = spymaster.select_action(
                env.unwrapped.labels,
                env.unwrapped.remaining_mask,
            )
            action = spymaster.action_to_flat(clue_idx, count)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            demos.append((obs, action, reward, next_obs, done, info))
            obs = next_obs

        if (game_idx + 1) % 100 == 0:
            print(f"  Collected {game_idx + 1}/{n_games} games "
                  f"({len(demos)} transitions total)")

    return demos
