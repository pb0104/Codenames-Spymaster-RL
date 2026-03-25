from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.similarity import cosine_similarity_matrix


@dataclass
class GreedyDecision:
    clue: str
    count: int
    score: float


class GreedySpymaster:
    """
    Greedy clue generator inspired by the classic cosine-margin heuristic.

    The clue is chosen by maximizing:
    min(sim(clue, targeted friendly words)) - max(sim(clue, bad words))
    """

    def __init__(self, max_count: int = 9) -> None:
        self.max_count = max_count

    def select_decision(self, env) -> GreedyDecision:
        remaining_friendly = env.remaining_friendly_indices
        if not remaining_friendly:
            return GreedyDecision(clue=env.embedding_store.clue_words[0], count=1, score=0.0)

        target_indices = env.current_goal_indices or remaining_friendly
        bad_indices = env.remaining_bad_indices
        legal_indices = env.legal_clue_indices

        clue_embeddings = env.embedding_store.clue_embeddings[legal_indices]
        target_embeddings = env.board_embeddings[target_indices]
        bad_embeddings = env.board_embeddings[bad_indices]

        target_scores = cosine_similarity_matrix(clue_embeddings, target_embeddings)
        if bad_embeddings.size == 0:
            bad_max = np.full(clue_embeddings.shape[0], -1.0, dtype=np.float32)
        else:
            bad_max = np.max(cosine_similarity_matrix(clue_embeddings, bad_embeddings), axis=1)

        best_clue = env.embedding_store.clue_words[int(legal_indices[0])]
        best_count = 1
        best_score = -float("inf")

        for candidate_row, clue_index in enumerate(legal_indices):
            good_scores = np.sort(target_scores[candidate_row])[::-1]
            for count in range(1, min(self.max_count, len(good_scores)) + 1):
                margin = float(good_scores[count - 1] - bad_max[candidate_row])
                if margin > best_score:
                    best_score = margin
                    best_count = count
                    best_clue = env.embedding_store.clue_words[int(clue_index)]

        return GreedyDecision(clue=best_clue, count=best_count, score=best_score)

    def select_action(self, env) -> np.ndarray:
        decision = self.select_decision(env)
        return env.encode_action(decision.clue, decision.count)
