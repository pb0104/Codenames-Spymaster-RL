from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.utils.similarity import l2_normalize


DEFAULT_NLTK_PACKAGES = ("words", "wordnet", "omw-1.4")
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"^[a-z]+$")


def normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


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

    filtered.sort(key=lambda token: (abs(len(token) - 6), token))
    return filtered[:max_words]


def sentence_transformer_text(token: str, token_type: str = "word") -> str:
    normalized = normalize_token(token).replace("_", " ")
    return f"Codenames {token_type}: {normalized}"


@dataclass
class EmbeddingStore:
    board_words: list[str]
    clue_words: list[str]
    dimension: int | None = None
    use_wordnet: bool = True
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL
    board_prompt: str = "board word"
    clue_prompt: str = "clue word"

    def __post_init__(self) -> None:
        self.board_words = list(self.board_words)
        self.clue_words = list(self.clue_words)
        self._vector_cache: dict[tuple[str, str], np.ndarray] = {}
        self._model = self._load_model(self.model_name)

        inferred_dimension = int(self._model.get_sentence_embedding_dimension())
        if self.dimension is None:
            self.dimension = inferred_dimension
        elif int(self.dimension) != inferred_dimension:
            raise ValueError(
                f"Configured embedding dimension {self.dimension} does not match "
                f"model dimension {inferred_dimension} for {self.model_name}."
            )

        self.board_embeddings = self.matrix(self.board_words, token_type=self.board_prompt)
        self.clue_embeddings = self.matrix(self.clue_words, token_type=self.clue_prompt)

    @staticmethod
    def _load_model(model_name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    @classmethod
    def from_paths(
        cls,
        board_words_path: str | Path,
        clue_words_path: str | Path | None = None,
        *,
        dimension: int | None = None,
        use_wordnet: bool = True,
        max_clues: int = 12000,
        min_clue_length: int = 3,
        max_clue_length: int = 12,
        download_missing_nltk: bool = False,
        persist_clues: bool = True,
        model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        board_prompt: str = "board word",
        clue_prompt: str = "clue word",
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
                clue_words_path.write_text("\n".join(clue_words) + "\n", encoding="utf-8")

        return cls(
            board_words=board_words,
            clue_words=clue_words,
            dimension=dimension,
            use_wordnet=use_wordnet,
            model_name=model_name,
            board_prompt=board_prompt,
            clue_prompt=clue_prompt,
        )

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def vector(self, token: str, token_type: str = "word") -> np.ndarray:
        normalized = normalize_token(token)
        cache_key = (normalized, token_type)
        if cache_key not in self._vector_cache:
            text = sentence_transformer_text(normalized, token_type=token_type)
            vector = self._encode_texts([text])[0]
            self._vector_cache[cache_key] = l2_normalize(vector[None, :])[0].astype(np.float32)
        return self._vector_cache[cache_key]

    def matrix(self, tokens: Sequence[str], token_type: str = "word") -> np.ndarray:
        if not tokens:
            return np.zeros((0, int(self.dimension or 0)), dtype=np.float32)

        missing_tokens: list[str] = []
        missing_keys: list[tuple[str, str]] = []
        for token in tokens:
            normalized = normalize_token(token)
            cache_key = (normalized, token_type)
            if cache_key not in self._vector_cache:
                missing_tokens.append(sentence_transformer_text(normalized, token_type=token_type))
                missing_keys.append(cache_key)

        if missing_tokens:
            encoded = self._encode_texts(missing_tokens)
            for cache_key, vector in zip(missing_keys, encoded):
                self._vector_cache[cache_key] = vector.astype(np.float32)

        return np.stack(
            [self._vector_cache[(normalize_token(token), token_type)] for token in tokens]
        ).astype(np.float32)

    def encode_action(self, clue: str, count: int, max_count: int) -> np.ndarray:
        count = max(1, min(max_count, int(count)))
        scaled_count = -1.0 + 2.0 * ((count - 1) / max(max_count - 1, 1))
        return np.concatenate(
            [
                self.vector(clue, token_type=self.clue_prompt),
                np.array([scaled_count], dtype=np.float32),
            ]
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
