"""
CodenamesEnv: Gymnasium environment for the Codenames Spymaster problem.

State  : (sim_matrix [25x25], labels [25], remaining_mask [25], goal [25])
Action : integer index → (clue_word_idx, count) where count ∈ {1,...,9}
         Action space size ≈ |vocab| × 9

Goal-conditioned for HER:
  goal g = binary mask of the k friendly words targeted this turn.
  After a failed episode HER relabels g = words actually guessed.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


# ── Word labels ────────────────────────────────────────────────────────────────
FRIENDLY  = 0
OPPOSING  = 1
NEUTRAL   = 2
ASSASSIN  = 3

LABEL_NAMES = {FRIENDLY: "friendly", OPPOSING: "opposing",
               NEUTRAL: "neutral",   ASSASSIN: "assassin"}

# ── Rewards (Siu 2022 base) ────────────────────────────────────────────────────
R_PER_TURN  = -1.0
R_ASSASSIN  = -25.0
R_WIN       =  0.0   # shaped bonus on top makes this positive in practice


class CodenamesEnv(gym.Env):
    """
    Single-agent Codenames Spymaster environment.

    Parameters
    ----------
    sim_matrix   : np.ndarray, shape (V, V)
                   Pre-computed pairwise GloVe cosine similarities.
                   Rows = vocab words, Cols = vocab words.
                   The 25 board words are a subset of the vocab.
    board_indices : np.ndarray, shape (25,)
                   Indices into the vocab for the 25 board words.
    vocab_size   : int
                   Total clue vocabulary size (≈10k NLTK words).
    n_friendly   : int   Number of friendly words on board (default 9 for blue).
    seed         : int or None
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_matrix: np.ndarray,
        board_indices: np.ndarray,
        vocab_size: int,
        n_friendly: int = 9,
        seed: Optional[int] = None,
    ):
        super().__init__()

        assert sim_matrix.shape[0] == vocab_size
        assert len(board_indices) == 25

        self.sim_matrix   = sim_matrix.astype(np.float32)   # (V, V)
        self.board_indices = board_indices                   # (25,)
        self.vocab_size   = vocab_size
        self.n_friendly   = n_friendly
        self._rng         = np.random.default_rng(seed)

        # ── Spaces ──────────────────────────────────────────────────────────
        # Action: clue_idx (vocab) × count (1..9)  → flattened integer
        self.n_counts     = 9
        self.action_space = spaces.Discrete(vocab_size * self.n_counts)

        # Observation dict (compatible with SB3 HER's GoalEnv)
        obs_dim = 25 * 25 + 25 + 25   # sim rows for board + labels + mask
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(0.0, 1.0, shape=(25,), dtype=np.float32),
            "desired_goal":  spaces.Box(0.0, 1.0, shape=(25,), dtype=np.float32),
        })

        # ── Game state (set by reset) ────────────────────────────────────────
        self.labels: np.ndarray          = None   # (25,) int
        self.remaining_mask: np.ndarray  = None   # (25,) bool
        self.desired_goal: np.ndarray    = None   # (25,) float binary
        self.achieved_goal: np.ndarray   = None   # (25,) float binary
        self._n_turns = 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _decode_action(self, action: int):
        """Return (clue_vocab_idx, count) for a flat action integer."""
        clue_idx = action // self.n_counts
        count    = (action % self.n_counts) + 1   # 1-indexed
        return clue_idx, count

    def _board_sim(self, clue_idx: int) -> np.ndarray:
        """
        Cosine similarities between clue word and the 25 board words.
        Shape: (25,)
        """
        return self.sim_matrix[clue_idx][self.board_indices]   # (25,)

    def _get_obs(self) -> dict:
        # Flatten sim rows for each board word against all vocab words would be
        # huge; instead use board×board sim sub-matrix (25×25) which is constant.
        board_sim_flat = self.sim_matrix[np.ix_(self.board_indices,
                                                self.board_indices)].flatten()  # (625,)
        obs = np.concatenate([
            board_sim_flat,
            self.labels.astype(np.float32) / 3.0,   # normalise to [0,1]
            self.remaining_mask.astype(np.float32),
        ]).astype(np.float32)  # (625 + 25 + 25 = 675,)

        return {
            "observation":   obs,
            "achieved_goal": self.achieved_goal.copy(),
            "desired_goal":  self.desired_goal.copy(),
        }

    def _sample_goal(self) -> np.ndarray:
        """Sample a random goal: a subset of remaining friendly word positions."""
        friendly_positions = np.where(
            (self.labels == FRIENDLY) & self.remaining_mask
        )[0]
        if len(friendly_positions) == 0:
            return np.zeros(25, dtype=np.float32)
        k = self._rng.integers(1, min(4, len(friendly_positions)) + 1)
        chosen = self._rng.choice(friendly_positions, size=k, replace=False)
        g = np.zeros(25, dtype=np.float32)
        g[chosen] = 1.0
        return g

    # ── Core Gym API ───────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomly assign labels to 25 board words
        # Standard Codenames: 9 friendly, 8 opposing, 7 neutral, 1 assassin
        label_pool = (
            [FRIENDLY]  * self.n_friendly +
            [OPPOSING]  * 8 +
            [NEUTRAL]   * (25 - self.n_friendly - 8 - 1) +
            [ASSASSIN]  * 1
        )
        self._rng.shuffle(label_pool)
        self.labels         = np.array(label_pool, dtype=np.int32)
        self.remaining_mask = np.ones(25, dtype=bool)
        self.achieved_goal  = np.zeros(25, dtype=np.float32)
        self.desired_goal   = self._sample_goal()
        self._n_turns       = 0

        return self._get_obs(), {}

    def step(self, action: int):
        clue_idx, count = self._decode_action(action)

        sims = self._board_sim(clue_idx)   # (25,)

        # Mask out already-revealed words
        masked_sims = np.where(self.remaining_mask, sims, -np.inf)

        # Greedy guesser picks top-`count` by cosine similarity
        top_positions = np.argsort(masked_sims)[::-1][:count]

        reward = R_PER_TURN
        terminated = False
        info = {"guessed": [], "hit_assassin": False, "hit_opposing": False}

        for pos in top_positions:
            if not self.remaining_mask[pos]:
                break
            self.remaining_mask[pos] = False
            word_label = self.labels[pos]
            info["guessed"].append(int(pos))

            if word_label == FRIENDLY:
                self.achieved_goal[pos] = 1.0
                # Continue guessing
            elif word_label == ASSASSIN:
                reward += R_ASSASSIN
                terminated = True
                info["hit_assassin"] = True
                break
            else:
                # Neutral or opposing → turn ends
                info["hit_opposing"] = (word_label == OPPOSING)
                break

        self._n_turns += 1

        # Win condition: all friendly words revealed
        friendly_remaining = np.sum(
            (self.labels == FRIENDLY) & self.remaining_mask
        )
        if friendly_remaining == 0:
            reward += R_WIN
            terminated = True

        # Truncation after too many turns (safety net)
        truncated = self._n_turns >= 50

        return self._get_obs(), reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> float:
        """
        HER reward function.
        +1 if achieved_goal ⊇ desired_goal, else -1.
        Called by SB3's HerReplayBuffer with relabeled goals.
        """
        # desired_goal words are a subset covered by achieved
        match = np.all((desired_goal == 0) | (achieved_goal == desired_goal))
        return 0.0 if match else -1.0

    def render(self):
        pass   # Implement text rendering if desired for debugging
