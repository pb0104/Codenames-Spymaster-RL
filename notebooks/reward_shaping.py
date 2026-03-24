"""
Potential-based reward shaping (Ng, Harada & Russell 1999).

The shaped reward is:
    r'(s, a, s') = r(s, a, s') + γ·Φ(s') − Φ(s)

where Φ(s) = cosine margin of the best available clue from state s.

Cosine margin = min_i sim(c*, w_i) − max_u sim(c*, u_u)
  w = friendly (target) words still active
  u = opposing / neutral / assassin words still active
  c* = vocab word that maximises this margin

This is a Gymnasium wrapper, so it wraps CodenamesEnv and augments step().
The optimal policy is provably unchanged (Ng et al. Theorem 1).
"""

import numpy as np
import gymnasium as gym
from typing import Optional


class PotentialShapingWrapper(gym.Wrapper):
    """
    Wraps CodenamesEnv and adds potential-based reward shaping.

    Parameters
    ----------
    env    : CodenamesEnv instance
    gamma  : discount factor (must match the RL algorithm's gamma)
    scale  : multiplier for the shaping bonus (default 1.0)
    top_k  : for efficiency, only consider top_k clue candidates
             when computing Φ.  Set to None to use full vocab (slow).
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99,
                 scale: float = 1.0, top_k: int = 500):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale
        self.top_k = top_k
        self._prev_potential: float = 0.0

    # ── Potential function Φ(s) ───────────────────────────────────────────────

    def _potential(self) -> float:
        """
        Compute the cosine-margin potential for the current env state.

        Φ(s) = max_{c ∈ vocab} [ min_{w ∈ friendly_active} sim(c, w)
                                − max_{u ∈ bad_active}     sim(c, u) ]

        If no friendly words remain, Φ = 0.
        """
        env = self.env   # unwrapped CodenamesEnv

        friendly_pos  = np.where((env.labels == 0) & env.remaining_mask)[0]
        bad_pos       = np.where((env.labels != 0) & env.remaining_mask)[0]

        if len(friendly_pos) == 0:
            return 0.0

        board_idx   = env.board_indices
        sim_matrix  = env.sim_matrix   # (V, V)

        # Sim between every vocab word and board words
        # sim_matrix[v, board_idx[i]] = sim(vocab[v], board_word[i])
        friendly_board = board_idx[friendly_pos]   # actual vocab indices on board
        bad_board      = board_idx[bad_pos]

        # For each vocab word c: min sim over friendly words
        min_friendly = sim_matrix[:, friendly_board].min(axis=1)   # (V,)

        # For efficiency, pre-filter by top_k on min_friendly before max over bad
        if self.top_k is not None and self.top_k < sim_matrix.shape[0]:
            candidate_idxs = np.argpartition(min_friendly, -self.top_k)[-self.top_k:]
        else:
            candidate_idxs = np.arange(sim_matrix.shape[0])

        if len(bad_board) > 0:
            max_bad = sim_matrix[candidate_idxs][:, bad_board].max(axis=1)  # (k,)
        else:
            max_bad = np.zeros(len(candidate_idxs), dtype=np.float32)

        margin = min_friendly[candidate_idxs] - max_bad   # (k,)
        return float(margin.max())

    # ── Gym overrides ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute new potential
        if terminated or truncated:
            new_potential = 0.0   # terminal state has Φ = 0 by convention
        else:
            new_potential = self._potential()

        shaping = self.scale * (self.gamma * new_potential - self._prev_potential)
        self._prev_potential = new_potential

        info["shaping_bonus"] = shaping
        info["potential"]     = new_potential

        return obs, reward + shaping, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Pass through to underlying env (required for HER compatibility)."""
        return self.env.compute_reward(achieved_goal, desired_goal, info)
