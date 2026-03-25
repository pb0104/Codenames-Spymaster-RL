from __future__ import annotations

import numpy as np


EPSILON = 1e-8


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    return array / norms


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a < EPSILON or norm_b < EPSILON:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def cosine_similarity_matrix(
    left: np.ndarray, right: np.ndarray | None = None
) -> np.ndarray:
    left_normalized = l2_normalize(left)
    right_normalized = left_normalized if right is None else l2_normalize(right)
    return left_normalized @ right_normalized.T


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, scores.shape[0])
    partition = np.argpartition(-scores, kth=k - 1)[:k]
    ordered = partition[np.argsort(-scores[partition])]
    return ordered.astype(int)
