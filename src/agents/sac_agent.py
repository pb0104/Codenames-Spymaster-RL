from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from src.agents.bc_pretrain import (
    DemonstrationTransition,
    stack_demo_actions,
    stack_demo_observations,
)


@dataclass
class SACTrainingConfig:
    learning_rate: float = 3e-4
    buffer_size: int = 50_000
    learning_starts: int = 100
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    total_timesteps: int = 1_000
    policy_kwargs: Optional[dict] = None
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"


class SACSpymasterAgent:
    def __init__(
        self, env, config: SACTrainingConfig | None = None, use_her: bool = True
    ):
        self.env = env
        self.config = config or SACTrainingConfig()
        replay_buffer_class = HerReplayBuffer if use_her else None
        replay_buffer_kwargs = None
        if use_her:
            replay_buffer_kwargs = {
                "n_sampled_goal": self.config.n_sampled_goal,
                "goal_selection_strategy": self.config.goal_selection_strategy,
            }

        self.model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            tau=self.config.tau,
            policy_kwargs=self.config.policy_kwargs or {"net_arch": [256, 256]},
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=0,
        )

    def bc_pretrain(
        self,
        demos: list[DemonstrationTransition],
        *,
        epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> list[float]:
        if not demos or epochs <= 0:
            return []

        observations = stack_demo_observations(demos)
        actions = stack_demo_actions(demos)
        optimizer = torch.optim.Adam(
            self.model.policy.actor.parameters(), lr=learning_rate
        )
        losses: list[float] = []

        num_samples = actions.shape[0]
        indices = np.arange(num_samples)

        for _ in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                obs_batch = {
                    key: value[batch_indices] for key, value in observations.items()
                }
                obs_tensor, _ = self.model.policy.obs_to_tensor(obs_batch)
                target_actions = torch.as_tensor(
                    actions[batch_indices], device=self.model.device
                )

                mean_actions, log_std, _ = (
                    self.model.policy.actor.get_action_dist_params(obs_tensor)
                )
                predicted_actions = torch.tanh(mean_actions)
                loss = F.mse_loss(predicted_actions, target_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            losses.append(epoch_loss / max(num_batches, 1))

        return losses

    def seed_replay_buffer(self, demos: list[DemonstrationTransition]) -> None:
        for transition in demos:
            obs = {
                key: np.expand_dims(value, axis=0)
                for key, value in transition.obs.items()
            }
            next_obs = {
                key: np.expand_dims(value, axis=0)
                for key, value in transition.next_obs.items()
            }
            action = np.expand_dims(transition.action, axis=0)
            reward = np.array([transition.reward], dtype=np.float32)
            done = np.array([transition.done], dtype=np.float32)
            self.model.replay_buffer.add(
                obs, next_obs, action, reward, done, [transition.info]
            )

    def learn(
        self,
        *,
        total_timesteps: Optional[int] = None,
        demos: Optional[list[DemonstrationTransition]] = None,
        bc_epochs: int = 0,
        bc_batch_size: int = 64,
        bc_learning_rate: float = 1e-3,
        seed_buffer: bool = True,
    ) -> dict:
        bc_losses: list[float] = []
        if demos:
            bc_losses = self.bc_pretrain(
                demos,
                epochs=bc_epochs,
                batch_size=bc_batch_size,
                learning_rate=bc_learning_rate,
            )
            if seed_buffer:
                self.seed_replay_buffer(demos)

        self.model.learn(total_timesteps=total_timesteps or self.config.total_timesteps)
        return {"bc_losses": bc_losses}

    def predict(self, observation, deterministic: bool = True):
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        self.model.save(path)
