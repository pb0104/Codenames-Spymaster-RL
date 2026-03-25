from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.board import (
    BoardCell,
    BoardConfig,
    all_good_revealed,
    flatten_board,
    generate_board,
    load_words,
    make_standard_config,
)
from src.env.reward import (
    RewardConfig,
    RewardBreakdown,
    build_step_reward,
    clue_margin,
    compute_goal_conditioned_reward,
)
from src.utils.embeddings import EmbeddingStore, normalize_token
from src.utils.seed import set_global_seed
from src.utils.similarity import cosine_similarity_matrix


ROLE_ORDER = ("friendly", "opponent", "neutral", "assassin")
ROLE_TO_INDEX = {role: idx for idx, role in enumerate(ROLE_ORDER)}


@dataclass
class StepOutcome:
    guessed_indices: list[int]
    guessed_words: list[str]
    guessed_roles: list[str]
    assassin_hit: bool


class CodenamesGame:
    """Backward-compatible helper for directly sampling boards and revealing words."""

    def __init__(self, words: list[str], config: BoardConfig):
        self.words = words
        self.config = config
        self.board: list[list[BoardCell]] = []
        self.done = False

    def reset(self):
        self.board = generate_board(self.words, self.config)
        self.done = False
        return self.board

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Game already finished. Call reset().")

        flat = flatten_board(self.board)
        cell = flat[action]
        if cell.revealed:
            return self.board, 0.0, self.done, {"word": cell.word, "role": cell.role}

        cell.revealed = True
        reward = 1.0 if cell.role == "friendly" else 0.0
        if cell.role == "assassin":
            reward = -10.0
            self.done = True
        if all_good_revealed(self.board):
            reward = 10.0
            self.done = True

        info = {"word": cell.word, "role": cell.role}
        return self.board, reward, self.done, info

    def available_actions(self):
        return [idx for idx, cell in enumerate(flatten_board(self.board)) if not cell.revealed]


class CodenamesSpymasterEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        board_words_path: str,
        clue_words_path: str,
        embedding_store: Optional[EmbeddingStore] = None,
        board_config: Optional[BoardConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        embedding_dim: int = 300,
        max_turns: int = 9,
        max_clue_count: int = 9,
        goal_size: int = 3,
        seed: Optional[int] = None,
        download_missing_nltk: bool = False,
        max_clues: int = 12000,
    ) -> None:
        super().__init__()

        self.seed_value = seed
        self.max_turns = max_turns
        self.max_clue_count = max_clue_count
        self.goal_size = goal_size
        self.reward_config = reward_config or RewardConfig()

        self.words = load_words(board_words_path)
        self.board_config = board_config or make_standard_config(5, 5, seed=seed)
        self.embedding_store = embedding_store or EmbeddingStore.from_paths(
            board_words_path=board_words_path,
            clue_words_path=clue_words_path,
            dimension=embedding_dim,
            max_clues=max_clues,
            download_missing_nltk=download_missing_nltk,
        )

        self.board: list[list[BoardCell]] = []
        self.turn_index = 0
        self.board_embeddings = np.zeros(
            (self.board_config.board_size, self.embedding_store.dimension), dtype=np.float32
        )
        self.similarity_matrix = np.zeros(
            (self.board_config.board_size, self.board_config.board_size), dtype=np.float32
        )
        self.current_goal_indices: list[int] = []
        self.legal_clue_indices = np.arange(len(self.embedding_store.clue_words))
        self.last_decoded_action: dict[str, Any] = {}

        obs_dim = (
            self.board_config.board_size * self.embedding_store.dimension
            + self.board_config.board_size * self.board_config.board_size
            + self.board_config.board_size * len(ROLE_ORDER)
            + self.board_config.board_size
            + 2
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.embedding_store.dimension + 1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.board_config.board_size,),
                    dtype=np.float32,
                ),
                "desired_goal": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.board_config.board_size,),
                    dtype=np.float32,
                ),
            }
        )

        set_global_seed(seed)
        self._rng = np.random.default_rng(seed)

    @property
    def flat_board(self) -> list[BoardCell]:
        return flatten_board(self.board)

    @property
    def remaining_friendly_indices(self) -> list[int]:
        return [
            idx
            for idx, cell in enumerate(self.flat_board)
            if cell.role == "friendly" and not cell.revealed
        ]

    @property
    def remaining_bad_indices(self) -> list[int]:
        return [
            idx
            for idx, cell in enumerate(self.flat_board)
            if cell.role in {"opponent", "neutral", "assassin"} and not cell.revealed
        ]

    def encode_action(self, clue: str, count: int) -> np.ndarray:
        return self.embedding_store.encode_action(clue, count, self.max_clue_count)

    def sample_action(self, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if len(self.legal_clue_indices) == 0:
            clue_index = 0
        else:
            clue_index = int(rng.choice(self.legal_clue_indices))
        count = int(rng.randint(1, self.max_clue_count + 1))
        clue = self.embedding_store.clue_words[clue_index]
        return self.encode_action(clue, count)

    def decode_action(self, action: np.ndarray) -> tuple[str, int]:
        return self.embedding_store.decode_action(
            np.asarray(action, dtype=np.float32),
            legal_indices=self.legal_clue_indices,
            max_count=self.max_clue_count,
        )

    def _board_words(self) -> list[str]:
        return [cell.word for cell in self.flat_board]

    def _role_one_hot(self) -> np.ndarray:
        matrix = np.zeros((self.board_config.board_size, len(ROLE_ORDER)), dtype=np.float32)
        for idx, cell in enumerate(self.flat_board):
            matrix[idx, ROLE_TO_INDEX[cell.role]] = 1.0
        return matrix

    def _remaining_mask(self) -> np.ndarray:
        return np.array(
            [0.0 if cell.revealed else 1.0 for cell in self.flat_board], dtype=np.float32
        )

    def _achieved_goal(self) -> np.ndarray:
        return np.array(
            [
                1.0 if cell.role == "friendly" and cell.revealed else 0.0
                for cell in self.flat_board
            ],
            dtype=np.float32,
        )

    def _desired_goal(self) -> np.ndarray:
        goal = np.zeros(self.board_config.board_size, dtype=np.float32)
        for idx in self.current_goal_indices:
            goal[idx] = 1.0
        return goal

    def _build_observation(self) -> dict[str, np.ndarray]:
        observation_vector = np.concatenate(
            [
                self.board_embeddings.flatten(),
                self.similarity_matrix.flatten(),
                self._role_one_hot().flatten(),
                self._remaining_mask(),
                np.array(
                    [
                        self.turn_index / max(self.max_turns, 1),
                        len(self.remaining_friendly_indices)
                        / max(self.board_config.num_friendly, 1),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        return {
            "observation": observation_vector,
            "achieved_goal": self._achieved_goal(),
            "desired_goal": self._desired_goal(),
        }

    def _refresh_board_embeddings(self) -> None:
        self.board_embeddings = self.embedding_store.matrix(self._board_words())
        self.similarity_matrix = cosine_similarity_matrix(self.board_embeddings).astype(
            np.float32
        )

    def _refresh_legal_clues(self) -> None:
        board_tokens = [normalize_token(word) for word in self._board_words()]
        legal: list[int] = []
        for idx, clue in enumerate(self.embedding_store.clue_words):
            if any(clue in word or word in clue for word in board_tokens):
                continue
            legal.append(idx)
        self.legal_clue_indices = np.array(legal, dtype=int)

    def _sample_goal_indices(self) -> list[int]:
        remaining = self.remaining_friendly_indices
        if not remaining:
            return []
        goal_size = min(self.goal_size, len(remaining))
        if goal_size <= 1:
            return [remaining[0]]

        best_subset: tuple[int, ...] | None = None
        best_score = -float("inf")
        for subset in combinations(remaining, goal_size):
            sim_block = self.similarity_matrix[np.ix_(subset, subset)]
            score = float(np.mean(sim_block))
            if score > best_score:
                best_score = score
                best_subset = subset
        return list(best_subset or remaining[:goal_size])

    def _guesser_turn(self, clue: str, count: int) -> StepOutcome:
        clue_embedding = self.embedding_store.vector(clue)
        available_indices = [
            idx for idx, cell in enumerate(self.flat_board) if not cell.revealed
        ]
        available_embeddings = self.board_embeddings[available_indices]
        scores = cosine_similarity_matrix(clue_embedding[None, :], available_embeddings)[0]
        ranked_indices = np.argsort(-scores)

        guessed_indices: list[int] = []
        guessed_words: list[str] = []
        guessed_roles: list[str] = []
        assassin_hit = False

        for local_idx in ranked_indices[:count]:
            board_idx = available_indices[int(local_idx)]
            cell = self.flat_board[board_idx]
            if cell.revealed:
                continue
            cell.revealed = True
            guessed_indices.append(board_idx)
            guessed_words.append(cell.word)
            guessed_roles.append(cell.role)
            if cell.role == "assassin":
                assassin_hit = True
                break
            if cell.role != "friendly":
                break

        return StepOutcome(
            guessed_indices=guessed_indices,
            guessed_words=guessed_words,
            guessed_roles=guessed_roles,
            assassin_hit=assassin_hit,
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        return compute_goal_conditioned_reward(
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            info=info,
            config=self.reward_config,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed_value = seed
            self.board_config.seed = seed
            set_global_seed(seed)
            self._rng = np.random.default_rng(seed)

        episode_seed = int(self._rng.integers(0, 1_000_000_000))
        self.board_config.seed = episode_seed

        self.board = generate_board(self.words, self.board_config)
        self.turn_index = 0
        self._refresh_board_embeddings()
        self._refresh_legal_clues()
        self.current_goal_indices = self._sample_goal_indices()
        self.last_decoded_action = {}

        observation = self._build_observation()
        info = {
            "board_words": self._board_words(),
            "role_counts": {role: sum(cell.role == role for cell in self.flat_board) for role in ROLE_ORDER},
        }
        return observation, info

    def step(self, action: np.ndarray):
        clue, count = self.decode_action(action)
        previous_goal = self._desired_goal().copy()
        target_indices = self.current_goal_indices.copy()
        target_embeddings = self.board_embeddings[target_indices]
        bad_indices = self.remaining_bad_indices
        bad_embeddings = self.board_embeddings[bad_indices]

        margin = clue_margin(
            clue_embedding=self.embedding_store.vector(clue),
            target_embeddings=target_embeddings,
            bad_embeddings=bad_embeddings,
        )
        outcome = self._guesser_turn(clue, count)
        self.turn_index += 1

        reward_without_goal = self.reward_config.shaped_weight * margin
        if outcome.assassin_hit:
            reward_without_goal += self.reward_config.assassin_penalty

        terminated = outcome.assassin_hit or all_good_revealed(self.board)
        truncated = self.turn_index >= self.max_turns and not terminated

        achieved_goal = self._achieved_goal()
        breakdown = build_step_reward(
            achieved_goal=achieved_goal,
            desired_goal=previous_goal,
            reward_without_goal=reward_without_goal,
            clue_margin_value=margin,
            assassin_hit=outcome.assassin_hit,
            config=self.reward_config,
        )

        if not terminated and not truncated:
            self.current_goal_indices = self._sample_goal_indices()
        else:
            self.current_goal_indices = []

        observation = self._build_observation()
        info = {
            "clue": clue,
            "count": count,
            "goal_indices": target_indices,
            "goal_words": [self.flat_board[idx].word for idx in target_indices],
            "guessed_indices": outcome.guessed_indices,
            "guessed_words": outcome.guessed_words,
            "guessed_roles": outcome.guessed_roles,
            "reward_without_goal": breakdown.reward_without_goal,
            "clue_margin": breakdown.clue_margin,
            "assassin_hit": breakdown.assassin_hit,
            "goal_achieved": breakdown.goal_achieved,
            "reward_breakdown": breakdown.to_dict(),
            "friendly_remaining": len(self.remaining_friendly_indices),
            "won": terminated and not outcome.assassin_hit,
        }
        self.last_decoded_action = {"clue": clue, "count": count}
        return observation, breakdown.total, terminated, truncated, info
