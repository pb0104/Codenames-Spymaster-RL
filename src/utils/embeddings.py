from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.utils.similarity import l2_normalize


DEFAULT_NLTK_PACKAGES = ("words", "wordnet", "omw-1.4")
TOKEN_RE = re.compile(r"^[a-z]+$")


def normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


def token_seed(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def random_vector_from_token(token: str, dimension: int) -> np.ndarray:
    rng = np.random.default_rng(token_seed(token))
    return rng.standard_normal(dimension).astype(np.float32)


def token_ngrams(token: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    padded = f"<{token}>"
    grams: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(padded) < n:
            continue
        grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams or [padded]


def deterministic_embedding(
    token: str,
    dimension: int = 300,
    use_wordnet: bool = True,
) -> np.ndarray:
    token = normalize_token(token)
    if not token:
        return np.zeros(dimension, dtype=np.float32)

    pieces = token_ngrams(token)
    vector = np.zeros(dimension, dtype=np.float32)
    for piece in pieces:
        vector += random_vector_from_token(piece, dimension)

    vector += 0.25 * random_vector_from_token(f"token::{token}", dimension)

    if use_wordnet:
        try:
            from nltk.corpus import wordnet as wn

            for synset in wn.synsets(token.replace("_", " "))[:3]:
                for lemma in synset.lemmas()[:3]:
                    lemma_name = normalize_token(lemma.name())
                    if lemma_name == token:
                        continue
                    vector += 0.15 * random_vector_from_token(
                        f"lemma::{lemma_name}", dimension
                    )
                hypernyms = synset.hypernyms()[:2]
                for hypernym in hypernyms:
                    for lemma in hypernym.lemmas()[:2]:
                        lemma_name = normalize_token(lemma.name())
                        vector += 0.08 * random_vector_from_token(
                            f"hyper::{lemma_name}", dimension
                        )
        except Exception:
            pass

    normalized = l2_normalize(vector[None, :])[0]
    return normalized.astype(np.float32)


def is_legal_single_word(token: str) -> bool:
    return bool(TOKEN_RE.fullmatch(token))


def download_nltk_packages(packages: Sequence[str] = DEFAULT_NLTK_PACKAGES) -> None:
    import nltk

    for package in packages:
        nltk.download(package, quiet=True)


def nltk_words_corpus() -> list[str]:
    tokens: list[str] = []
    try:
        from nltk.corpus import words

        tokens.extend(words.words())
    except Exception:
        pass
    try:
        from nltk.corpus import wordnet as wn

        for lemma in wn.all_lemma_names():
            tokens.append(lemma)
    except Exception:
        pass
    return tokens


def fallback_words_corpus() -> list[str]:
    candidates: list[str] = []
    for dictionary_path in ("/usr/share/dict/words", "/usr/dict/words"):
        path = Path(dictionary_path)
        if path.exists():
            candidates.extend(path.read_text(encoding="utf-8").splitlines())
            break
    return candidates


def sanitize_clue_candidates(
    candidates: Iterable[str],
    board_words: Iterable[str],
    max_words: int = 12000,
    min_length: int = 3,
    max_length: int = 12,
) -> list[str]:
    board_normalized = {normalize_token(word) for word in board_words}
    filtered: list[str] = []
    seen: set[str] = set()

    for token in candidates:
        normalized = normalize_token(token)
        normalized = normalized.replace("-", "_")
        if "_" in normalized:
            continue
        if not is_legal_single_word(normalized):
            continue
        if len(normalized) < min_length or len(normalized) > max_length:
            continue
        if normalized in seen or normalized in board_normalized:
            continue
        if any(
            normalized in board_word or board_word in normalized
            for board_word in board_normalized
        ):
            continue
        seen.add(normalized)
        filtered.append(normalized)

    filtered.sort(key=lambda token: (abs(len(token) - 6), token_seed(token)))
    return filtered[:max_words]


@dataclass
class EmbeddingStore:
    board_words: list[str]
    clue_words: list[str]
    dimension: int = 300
    use_wordnet: bool = True

    def __post_init__(self) -> None:
        self.board_words = list(self.board_words)
        self.clue_words = list(self.clue_words)
        self._vector_cache: dict[str, np.ndarray] = {}
        self.board_embeddings = self.matrix(self.board_words)
        self.clue_embeddings = self.matrix(self.clue_words)

    @classmethod
    def from_paths(
        cls,
        board_words_path: str | Path,
        clue_words_path: str | Path | None = None,
        *,
        dimension: int = 300,
        use_wordnet: bool = True,
        max_clues: int = 12000,
        min_clue_length: int = 3,
        max_clue_length: int = 12,
        download_missing_nltk: bool = False,
        persist_clues: bool = True,
    ) -> "EmbeddingStore":
        board_words_path = Path(board_words_path)
        board_words = [
            line.strip()
            for line in board_words_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if clue_words_path is not None:
            clue_words_path = Path(clue_words_path)
        clue_words: list[str] = []

        if clue_words_path is not None and clue_words_path.exists():
            clue_words = [
                line.strip()
                for line in clue_words_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        if not clue_words:
            if download_missing_nltk:
                try:
                    download_nltk_packages()
                except Exception:
                    pass

            corpus = nltk_words_corpus() or fallback_words_corpus()
            clue_words = sanitize_clue_candidates(
                corpus,
                board_words=board_words,
                max_words=max_clues,
                min_length=min_clue_length,
                max_length=max_clue_length,
            )
            if not clue_words:
                clue_words = [
                    "animal",
                    "nature",
                    "battle",
                    "travel",
                    "ocean",
                    "science",
                    "history",
                    "magic",
                    "music",
                    "winter",
                ]
            if clue_words_path is not None and persist_clues:
                clue_words_path.parent.mkdir(parents=True, exist_ok=True)
                clue_words_path.write_text(
                    "\n".join(clue_words) + "\n", encoding="utf-8"
                )

        return cls(
            board_words=board_words,
            clue_words=clue_words,
            dimension=dimension,
            use_wordnet=use_wordnet,
        )

    def vector(self, token: str) -> np.ndarray:
        normalized = normalize_token(token)
        if normalized not in self._vector_cache:
            self._vector_cache[normalized] = deterministic_embedding(
                normalized,
                dimension=self.dimension,
                use_wordnet=self.use_wordnet,
            )
        return self._vector_cache[normalized]

    def matrix(self, tokens: Sequence[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self.vector(token) for token in tokens]).astype(np.float32)

    def encode_action(self, clue: str, count: int, max_count: int) -> np.ndarray:
        count = max(1, min(max_count, int(count)))
        scaled_count = -1.0 + 2.0 * ((count - 1) / max(max_count - 1, 1))
        return np.concatenate(
            [self.vector(clue), np.array([scaled_count], dtype=np.float32)]
        ).astype(np.float32)

    def decode_action(
        self,
        action: np.ndarray,
        legal_indices: np.ndarray | None,
        max_count: int,
    ) -> tuple[str, int]:
        clue_vector = action[:-1].astype(np.float32)
        if not np.any(clue_vector):
            clue_vector = np.ones_like(clue_vector, dtype=np.float32)
        clue_vector = l2_normalize(clue_vector[None, :])[0]

        if legal_indices is None or len(legal_indices) == 0:
            candidate_indices = np.arange(len(self.clue_words))
        else:
            candidate_indices = legal_indices

        candidate_embeddings = self.clue_embeddings[candidate_indices]
        scores = candidate_embeddings @ clue_vector
        best_idx = candidate_indices[int(np.argmax(scores))]
        scaled_count = float(np.clip(action[-1], -1.0, 1.0))
        count = 1 + round(((scaled_count + 1.0) / 2.0) * max(max_count - 1, 1))
        count = max(1, min(max_count, count))
        return self.clue_words[best_idx], int(count)
