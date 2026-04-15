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
        self.last_bc_pretrain_metrics: dict[str, list[float] | float] = {
            "bc_losses": [],
            "bc_action_losses": [],
            "bc_margin_losses": [],
            "bc_predicted_margins": [],
            "bc_target_margins": [],
            "bc_cosine_margin_loss_weight": 0.0,
        }

    def _predict_cosine_margin(
        self,
        obs_tensor: dict[str, torch.Tensor],
        desired_goal_tensor: torch.Tensor,
        predicted_actions: torch.Tensor,
    ) -> torch.Tensor:
        board_size = int(desired_goal_tensor.shape[1])
        embedding_dim = int(self.env.embedding_store.dimension)
        observation_tensor = obs_tensor["observation"]

        board_embedding_end = board_size * embedding_dim
        similarity_end = board_embedding_end + board_size * board_size
        role_end = similarity_end + board_size * 4
        remaining_end = role_end + board_size

        board_embeddings = observation_tensor[:, :board_embedding_end].reshape(
            -1, board_size, embedding_dim
        )
        role_one_hot = observation_tensor[:, similarity_end:role_end].reshape(
            -1, board_size, 4
        )
        remaining_mask = observation_tensor[:, role_end:remaining_end] > 0.5
        target_mask = desired_goal_tensor > 0.5
        bad_mask = remaining_mask & ~(role_one_hot[:, :, 0] > 0.5)

        predicted_clue = F.normalize(predicted_actions[:, :-1], p=2, dim=1, eps=1e-8)
        scores = torch.bmm(board_embeddings, predicted_clue.unsqueeze(-1)).squeeze(-1)

        has_target = target_mask.any(dim=1)
        min_target_scores = scores.masked_fill(~target_mask, float("inf")).min(dim=1).values

        has_bad = bad_mask.any(dim=1)
        max_bad_scores = scores.masked_fill(~bad_mask, float("-inf")).max(dim=1).values
        max_bad_scores = torch.where(
            has_bad,
            max_bad_scores,
            torch.full_like(max_bad_scores, -1.0),
        )

        predicted_margin = min_target_scores - max_bad_scores
        return torch.where(has_target, predicted_margin, torch.zeros_like(predicted_margin))

    def bc_pretrain(
        self,
        demos: list[DemonstrationTransition],
        *,
        epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        cosine_margin_loss_weight: float = 0.0,
    ) -> list[float]:
        if not demos or epochs <= 0:
            self.last_bc_pretrain_metrics = {
                "bc_losses": [],
                "bc_action_losses": [],
                "bc_margin_losses": [],
                "bc_predicted_margins": [],
                "bc_target_margins": [],
                "bc_cosine_margin_loss_weight": float(cosine_margin_loss_weight),
            }
            return []

        observations = stack_demo_observations(demos)
        actions = stack_demo_actions(demos)
        optimizer = torch.optim.Adam(
            self.model.policy.actor.parameters(), lr=learning_rate
        )
        losses: list[float] = []
        action_losses: list[float] = []
        margin_losses: list[float] = []
        predicted_margin_means: list[float] = []
        target_margin_means: list[float] = []

        num_samples = actions.shape[0]
        indices = np.arange(num_samples)

        for _ in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            epoch_action_loss = 0.0
            epoch_margin_loss = 0.0
            epoch_predicted_margin = 0.0
            epoch_target_margin = 0.0
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
                desired_goal_tensor = obs_tensor["desired_goal"]
                target_margins = torch.as_tensor(
                    [
                        float(demos[int(index)].info.get("clue_margin", 0.0))
                        for index in batch_indices
                    ],
                    device=self.model.device,
                )

                mean_actions, _log_std, _ = (
                    self.model.policy.actor.get_action_dist_params(obs_tensor)
                )
                predicted_actions = torch.tanh(mean_actions)
                action_loss = F.mse_loss(predicted_actions, target_actions)
                predicted_margins = self._predict_cosine_margin(
                    obs_tensor,
                    desired_goal_tensor,
                    predicted_actions,
                )
                margin_loss = torch.relu(target_margins - predicted_margins).mean()
                loss = action_loss + cosine_margin_loss_weight * margin_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_action_loss += float(action_loss.item())
                epoch_margin_loss += float(margin_loss.item())
                epoch_predicted_margin += float(predicted_margins.mean().item())
                epoch_target_margin += float(target_margins.mean().item())
                num_batches += 1

            losses.append(epoch_loss / max(num_batches, 1))
            action_losses.append(epoch_action_loss / max(num_batches, 1))
            margin_losses.append(epoch_margin_loss / max(num_batches, 1))
            predicted_margin_means.append(epoch_predicted_margin / max(num_batches, 1))
            target_margin_means.append(epoch_target_margin / max(num_batches, 1))

        self.last_bc_pretrain_metrics = {
            "bc_losses": losses,
            "bc_action_losses": action_losses,
            "bc_margin_losses": margin_losses,
            "bc_predicted_margins": predicted_margin_means,
            "bc_target_margins": target_margin_means,
            "bc_cosine_margin_loss_weight": float(cosine_margin_loss_weight),
        }

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
        bc_cosine_margin_loss_weight: float = 0.0,
        seed_buffer: bool = True,
    ) -> dict:
        bc_losses: list[float] = []
        if demos:
            bc_losses = self.bc_pretrain(
                demos,
                epochs=bc_epochs,
                batch_size=bc_batch_size,
                learning_rate=bc_learning_rate,
                cosine_margin_loss_weight=bc_cosine_margin_loss_weight,
            )
            if seed_buffer:
                self.seed_replay_buffer(demos)

        self.model.learn(total_timesteps=total_timesteps or self.config.total_timesteps)
        summary = dict(self.last_bc_pretrain_metrics)
        summary["bc_losses"] = bc_losses
        return summary

    def predict(self, observation, deterministic: bool = True):
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        self.model.save(path)
