"""
FALLBACK embedding engine for P4, until ``src/embeddings.py`` (P3) is available.

It follows exactly the ``EmbeddingEngine`` contract (src/embeddings.py):
    encode(texts) -> L2-normalised np.ndarray
    book_matrix() -> (N, d) aligned with the books DataFrame
    label_matrix(block) -> (labels, matrix)
    search(query_vec, k) -> (indices, scores)
    best_sentence(query_vec, text) -> (sentence, score)

As soon as P3 delivers ``EmbeddingEngine``, ``recommender.get_engine()`` uses it
automatically and this file is no longer needed (it remains useful for the tests).

Two encoders are available:
- ``"sbert"`` (default): all-MiniLM-L6-v2, like the rest of the project;
- ``"lsa"``: TF-IDF + truncated SVD trained on the catalogue. OFFLINE mode
  only, to test the code without downloading a model (this is NOT a
  contextual embedding: do not use it for the demo or the report).
  Enabled with the environment variable ``GENDERBOOK_ENCODER=lsa``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Sentence splitting: after . ! ? followed by a space and a capital/quote/digit
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“'A-Z0-9])")


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    """Split a blurb into sentences (very short fragments are merged into the previous one)."""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(str(text or "")) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(p) < min_chars:
            out[-1] = f"{out[-1]} {p}"   # "Yes." alone is meaningless: it is glued back
        else:
            out.append(p)
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalise each row to norm 1 (cosine = dot product)."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class LsaEncoder:
    """Offline TF-IDF + SVD (LSA) encoder. Only for tests without Internet."""

    def __init__(self, corpus: list[str], dim: int = 256, seed: int = config.RANDOM_SEED):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.model_name = f"lsa{dim}"
        # Step 1: TF-IDF vocabulary learned on the catalogue blurbs
        self.tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5,
                                     sublinear_tf=True, ngram_range=(1, 2), max_features=60_000)
        x = self.tfidf.fit_transform(corpus)
        # Step 2: dimensionality reduction (captures co-occurrences, a bit of "semantics")
        self.svd = TruncatedSVD(n_components=min(dim, x.shape[1] - 1), random_state=seed)
        self.svd.fit(x)

    def encode(self, texts: list[str], **_) -> np.ndarray:
        return l2_normalize(self.svd.transform(self.tfidf.transform(list(texts))))


class LocalEngine:
    """Minimal implementation of the EmbeddingEngine contract (P3) to unblock P4.

    Args:
        books: catalogue (DataFrame from ``data_loader.load_books()``), in the order
            that will be used as the index everywhere.
        blocks: taxonomy ``{block: {label: sentence}}`` (blocks.json).
        encoder: "sbert", "lsa" or an object with ``encode(texts)``.
        model_name: SBERT model to use.
        cache: True = save the book matrix in data/embeddings/.
    """

    def __init__(self, books: pd.DataFrame, blocks: dict[str, dict[str, str]] | None = None,
                 encoder: str | object | None = None, model_name: str = config.EMBEDDING_MODEL,
                 cache: bool = True):
        self.books = books.reset_index(drop=True)
        self.blocks = blocks or {}
        self.cache = cache
        encoder = encoder or os.environ.get("GENDERBOOK_ENCODER", "sbert")
        if encoder == "sbert":
            from src.enrichment import SbertEncoder  # P2's SBERT encoder (lazy loading)
            self._encoder = SbertEncoder(model_name)
            self.model_name = model_name.split("/")[-1]
        elif encoder == "lsa":
            self._encoder = LsaEncoder(self.books["embed_text"].fillna("").tolist())
            self.model_name = self._encoder.model_name
        else:
            self._encoder = encoder
            self.model_name = getattr(encoder, "model_name", type(encoder).__name__)
        self._book_matrix: np.ndarray | None = None
        self._label_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    # ------------------------------------------------------------------ encode
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts -> L2-normalised matrix (n, d)."""
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            emb = self._encoder.encode(texts, batch_size=64, show_progress=len(texts) > 500)
        except TypeError:  # encoder that does not accept these arguments
            emb = self._encoder.encode(texts)
        return l2_normalize(emb)

    @property
    def dim(self) -> int:
        return int(self.book_matrix().shape[1])

    # ------------------------------------------------------------- book_matrix
    def _cache_paths(self) -> tuple[Path, Path]:
        # The name depends on the model AND the book_ids: another catalogue => another file
        ids = self.books["book_id"].astype(str).tolist()
        h = hashlib.md5("|".join(ids).encode()).hexdigest()[:8]
        stem = f"p4_{self.model_name}_{len(ids)}_{h}_books"
        return config.DATA_EMBEDDINGS / f"{stem}.npy", config.DATA_EMBEDDINGS / f"{stem}_ids.json"

    def book_matrix(self) -> np.ndarray:
        """Embeddings of ``embed_text`` for all books, (N, d), aligned with ``books``."""
        if self._book_matrix is not None:
            return self._book_matrix
        npy, ids_json = self._cache_paths()
        ids = self.books["book_id"].astype(str).tolist()
        # Step 1: disk cache (the 1st encoding of the full catalogue takes a few minutes on CPU)
        if self.cache and npy.exists() and ids_json.exists() \
                and json.loads(ids_json.read_text(encoding="utf-8")) == ids:
            self._book_matrix = np.load(npy)
            return self._book_matrix
        # Step 2: on-the-fly encoding
        mat = self.encode(self.books["embed_text"].fillna("").tolist())
        if self.cache:
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, mat)
            ids_json.write_text(json.dumps(ids), encoding="utf-8")
        self._book_matrix = mat
        return mat

    # ------------------------------------------------------------ label_matrix
    def label_matrix(self, block: str) -> tuple[list[str], np.ndarray]:
        """Embeddings of the descriptive SENTENCES of a block's labels (not bare words).

        For the themes, P2's method is reused (average of the sentence + variants
        from theme_variants.json) so that our labels are comparable to P2's.
        """
        if block in self._label_cache:
            return self._label_cache[block]
        labels = list(self.blocks.get(block, {}))
        if not labels:
            out: tuple[list[str], np.ndarray] = ([], np.zeros((0, self.dim), dtype=np.float32))
        else:
            mat = None
            if block == "Themes":
                try:
                    from src.enrichment import load_theme_variants
                    variants = load_theme_variants(themes=self.blocks["Themes"])
                    rows = []
                    for lab in labels:
                        v = self.encode([self.blocks[block][lab], *variants.get(lab, [])])
                        rows.append(v.mean(0))
                    mat = l2_normalize(np.stack(rows))
                except Exception:  # variants file missing: sentence alone
                    mat = None
            if mat is None:
                mat = self.encode([self.blocks[block][lab] for lab in labels])
            out = (labels, mat)
        self._label_cache[block] = out
        return out

    # ------------------------------------------------------------------ search
    def search(self, query_vec: np.ndarray, k: int = config.RETRIEVAL_TOP_K
               ) -> tuple[np.ndarray, np.ndarray]:
        """k closest books by cosine (= dot product, normalised vectors)."""
        scores = self.book_matrix() @ np.asarray(query_vec, dtype=np.float32)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]          # partial selection O(N)
        idx = idx[np.argsort(-scores[idx], kind="stable")]  # then sort the k kept
        return idx, scores[idx]

    # ----------------------------------------------------------- best_sentence
    def best_sentence(self, query_vec: np.ndarray, text: str) -> tuple[str, float]:
        """Sentence of the blurb closest to the query (explainability)."""
        sents = split_sentences(text)
        if not sents:
            return "", 0.0
        sims = self.encode(sents) @ np.asarray(query_vec, dtype=np.float32)
        i = int(np.argmax(sims))
        return sents[i], float(sims[i])
