"""
embeddings.py  –  Load GloVe-300d, build sim matrix, filter vocab.

Usage
-----
    loader = GloveLoader("path/to/glove.840B.300d.txt")
    loader.load(verbose=True)

    vocab, board_indices, sim_matrix = loader.build_sim_matrix(
        board_words=["snowman", "hole", "cycle", ...],   # 25 words
        clue_vocab=nltk_words,                           # ~10k strings
    )
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os


class GloveLoader:
    """
    Load a GloVe text file and expose word vectors.

    Parameters
    ----------
    glove_path : path to glove.840B.300d.txt (or similar)
    dim        : embedding dimension (default 300)
    """

    def __init__(self, glove_path: str, dim: int = 300):
        self.glove_path = Path(glove_path)
        self.dim        = dim
        self.word2vec   = {}   # str → np.ndarray (dim,)

    def load(
        self,
        max_words: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Load GloVe vectors from text file."""
        if not self.glove_path.exists():
            raise FileNotFoundError(f"GloVe file not found: {self.glove_path}")

        if verbose:
            size = os.path.getsize(self.glove_path) / 1e9
            print(f"Loading GloVe from {self.glove_path} ({size:.1f} GB)…")

        count = 0
        with open(self.glove_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0].lower()
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
                if len(vec) != self.dim:
                    continue
                self.word2vec[word] = vec
                count += 1
                if max_words and count >= max_words:
                    break

        if verbose:
            print(f"Loaded {len(self.word2vec):,} vectors.")

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Return normalised GloVe vector or None if missing."""
        v = self.word2vec.get(word.lower())
        if v is None:
            return None
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))   # already normalised

    def build_sim_matrix(
        self,
        board_words: List[str],
        clue_vocab: List[str],
        fallback_random: bool = True,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Build the (V, V) cosine similarity matrix for the full vocab
        (board words ∪ clue vocab).

        Returns
        -------
        vocab         : list of V strings (board words first, then clue vocab)
        board_indices : np.ndarray (25,) indices of board words in vocab
        sim_matrix    : np.ndarray (V, V) float32 pairwise cosine sims
        """
        # Combine: board words are guaranteed in vocab
        vocab = list(board_words)
        board_set = set(w.lower() for w in board_words)

        for w in clue_vocab:
            if w.lower() not in board_set:
                vocab.append(w)

        V = len(vocab)
        print(f"Building {V}×{V} sim matrix…")

        # Embed all vocab words
        vecs = np.zeros((V, self.dim), dtype=np.float32)
        missing = []
        for i, w in enumerate(vocab):
            v = self.get_vector(w)
            if v is not None:
                vecs[i] = v
            else:
                missing.append(w)
                if fallback_random:
                    rand = np.random.randn(self.dim).astype(np.float32)
                    vecs[i] = rand / np.linalg.norm(rand)

        if missing:
            print(f"  {len(missing)} words missing from GloVe; "
                  f"{'random fallback' if fallback_random else 'zero'} used.")

        # Cosine sim matrix: vecs are already unit-norm → just dot product
        sim_matrix = (vecs @ vecs.T).astype(np.float32)

        # Board indices: first 25 entries by construction
        board_indices = np.arange(25, dtype=np.int32)

        print(f"Done. Sim matrix shape: {sim_matrix.shape}")
        return vocab, board_indices, sim_matrix
