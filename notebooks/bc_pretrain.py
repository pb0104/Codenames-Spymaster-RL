"""
Behavioral Cloning (BC) pretraining for the SAC actor.

Steps:
  1. Collect demo transitions from the greedy Spymaster (see guesser.py).
  2. Pretrain the SAC policy network via supervised cross-entropy loss.
  3. Seed the SAC HerReplayBuffer with demo transitions.
  4. Optionally keep a BC regularisation term during early RL training.

Requires: stable-baselines3 >= 2.0, torch
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


# ── Step 1: extract flat (obs_vec, action) pairs ──────────────────────────────

def demos_to_tensors(
    demos: list,
    obs_key: str = "observation",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert raw demo list → (obs_tensor [N, obs_dim], action_tensor [N]).
    We use only the 'observation' component of the dict obs for pretraining.
    """
    obs_list = []
    act_list = []
    for (obs, action, reward, next_obs, done, info) in demos:
        obs_list.append(obs[obs_key])
        act_list.append(action)

    obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    act_t = torch.tensor(act_list,           dtype=torch.long)
    return obs_t, act_t


# ── Step 2: BC pretraining ────────────────────────────────────────────────────

def bc_pretrain(
    model: SAC,
    demos: list,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "auto",
    obs_key: str = "observation",
) -> List[float]:
    """
    Pretrain the SAC *actor* via behavioural cloning (cross-entropy on actions).

    SAC uses a squashed Gaussian actor; we treat the action as discrete here
    by treating the actor's mean output as logits over the action space.
    (Works well for large discrete action spaces with SB3's discrete SAC.)

    Returns list of per-epoch mean losses.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    obs_t, act_t = demos_to_tensors(demos, obs_key)
    dataset    = TensorDataset(obs_t, act_t)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Access SB3's actor network
    actor      = model.policy.actor.to(device)
    optimiser  = torch.optim.Adam(actor.parameters(), lr=lr)

    epoch_losses = []
    print(f"BC pretraining for {n_epochs} epochs on {len(obs_t)} transitions...")

    for epoch in range(n_epochs):
        batch_losses = []
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            # Forward pass through actor to get logits / distribution mean
            # For SB3 discrete SAC the actor returns action logits directly
            dist = actor.get_distribution(obs_batch)
            logits = dist.distribution.logits   # (B, A)

            loss = F.cross_entropy(logits, act_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            batch_losses.append(loss.item())

        mean_loss = np.mean(batch_losses)
        epoch_losses.append(mean_loss)
        print(f"  Epoch {epoch + 1}/{n_epochs}  loss={mean_loss:.4f}")

    return epoch_losses


# ── Step 3: seed the HER replay buffer ────────────────────────────────────────

def seed_her_buffer(
    model: SAC,
    demos: list,
    max_transitions: Optional[int] = None,
) -> int:
    """
    Add demo transitions directly into the SAC HER replay buffer.

    SB3's HerReplayBuffer.add() signature:
        add(obs, next_obs, action, reward, done, infos)

    Returns number of transitions added.
    """
    replay_buffer: HerReplayBuffer = model.replay_buffer
    n_added = 0

    transitions = demos if max_transitions is None else demos[:max_transitions]

    for (obs, action, reward, next_obs, done, info) in transitions:
        # SB3 expects numpy arrays with batch dim
        replay_buffer.add(
            obs       = {k: np.expand_dims(v, 0) for k, v in obs.items()},
            next_obs  = {k: np.expand_dims(v, 0) for k, v in next_obs.items()},
            action    = np.array([[action]]),
            reward    = np.array([reward]),
            done      = np.array([done]),
            infos     = [info],
        )
        n_added += 1

    print(f"Seeded HER buffer with {n_added} demo transitions.")
    return n_added


# ── Step 4: BC regularisation loss (optional, used during early RL) ───────────

class BCRegulariser:
    """
    Adds a BC regularisation term to the SAC training.
    Call .compute_loss(actor, obs_batch) inside a custom training loop
    or use the callback below.

    Loss = λ · cross_entropy(actor_logits, demo_actions)
    λ decays linearly from lambda_start to 0 over decay_steps.
    """

    def __init__(
        self,
        demos: list,
        lambda_start: float = 0.1,
        decay_steps: int = 50_000,
        batch_size: int = 256,
        device: str = "cpu",
        obs_key: str = "observation",
    ):
        self.lambda_start  = lambda_start
        self.decay_steps   = decay_steps
        self.batch_size    = batch_size
        self.device        = device
        self._step         = 0

        obs_t, act_t = demos_to_tensors(demos, obs_key)
        self._obs_t  = obs_t.to(device)
        self._act_t  = act_t.to(device)

    @property
    def lambda_current(self) -> float:
        frac = min(1.0, self._step / max(1, self.decay_steps))
        return self.lambda_start * (1.0 - frac)

    def compute_loss(self, actor) -> torch.Tensor:
        """Return the BC regularisation loss (scalar tensor)."""
        self._step += self.batch_size

        if self.lambda_current <= 0:
            return torch.tensor(0.0)

        idx = torch.randint(len(self._obs_t), (self.batch_size,))
        obs = self._obs_t[idx]
        act = self._act_t[idx]

        dist   = actor.get_distribution(obs)
        logits = dist.distribution.logits
        loss   = F.cross_entropy(logits, act)

        return self.lambda_current * loss
