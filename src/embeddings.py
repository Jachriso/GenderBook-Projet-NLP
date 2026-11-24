"""
embeddings.py: the project's semantic engine (Person 3).

Role: turn any text into a VECTOR (embedding) and measure how close two texts are
in meaning with COSINE similarity. The whole team goes through here:
    - P2 (enrichment)       : engine.encode(...)            (compatible with its SbertEncoder)
    - P4 (scoring)          : engine.search, engine.label_matrix, engine.best_sentence
    - P5 (app)              : engine loaded once (st.cache_resource)
    - P6 (evaluation)       : comparison of the encoders

Technical choices (to be justified in the report):
    * Default model: sentence-transformers/all-MiniLM-L6-v2 (config.EMBEDDING_MODEL).
      SBERT = BERT fine-tuned so that two sentences with the same meaning get close vectors
      (trained on more than a billion sentence pairs). 384 dimensions, fast on CPU.
    * NORMALISED vectors (L2 norm = 1). Consequence: cosine(a, b) = a · b.
      The score of a book for a query is therefore a simple matrix-vector product:
      12,000 books compared in a single numpy operation (a few milliseconds).
    * Disk cache: the 12,000 books are encoded ONCE and then reloaded from a .npy.
      An identifier file (book_ids.json) guarantees that row i of the .npy really
      matches book i of the catalogue (otherwise it re-encodes automatically).
    * Long texts: MiniLM only reads 256 tokens (the rest is silently ignored).
      Two strategies: "truncate" (default) or "chunks" (split into chunks of
      complete sentences, each chunk encoded, vectors averaged).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import preprocess_for_transformer, split_sentences

# Model name shortcuts (handy in the notebooks and for P6)
MODEL_ALIASES = {
    "minilm": config.EMBEDDING_MODEL,       # 384 dim, fast
    "mpnet": config.EMBEDDING_MODEL_ALT,    # 768 dim, more accurate, ~5x slower
}
LONG_TEXT_MODES = ("truncate", "chunks")


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Scale each row to a norm of 1 (so that dot product = cosine).

    Zero rows (empty text) stay zero instead of causing a division by zero.
    """
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between the rows of a (n, d) and of b (m, d) -> matrix (n, m).

    cos(a, b) = (a · b) / (||a|| · ||b||). We normalise first, then a single matrix
    product. Values in [-1, 1]: 1 = same direction (same meaning), 0 = unrelated.
    """
    a = l2_normalize(np.atleast_2d(a))
    b = l2_normalize(np.atleast_2d(b))
    return a @ b.T


def model_slug(model_name: str) -> str:
    """Safe file name from a model name.

    >>> model_slug("sentence-transformers/all-MiniLM-L6-v2")
    'all-MiniLM-L6-v2'
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.split("/")[-1])


def resolve_model_name(name: str) -> str:
    """Accept an alias ('minilm', 'mpnet') or a full Hugging Face name."""
    return MODEL_ALIASES.get(name.lower(), name)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

class EmbeddingEngine:
    """Semantic engine: SBERT encoding, book cache, cosine search.

    Interface contract shared by the team:
        encode(texts)              -> np.ndarray (n, d), L2-normalised
        book_matrix()              -> np.ndarray (N, d), aligned with the catalogue
        label_matrix(block)        -> (labels, np.ndarray (L, d))
        search(query_vec, k=300)   -> (indices, scores) sorted by decreasing score
        best_sentence(query_vec, text) -> (sentence, score)

    Example:
        engine = EmbeddingEngine()                       # nothing is loaded at this stage
        q = engine.encode(["a boy discovers he is a wizard"])[0]
        idx, scores = engine.search(q, k=3)
        engine.books.iloc[idx][["title", "authors_str"]]
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        books: pd.DataFrame | None = None,
        long_text: str = "truncate",
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: Path | str = config.DATA_EMBEDDINGS,
        model: Any = None,
    ) -> None:
        """
        Args:
            model_name: Hugging Face name or alias ('minilm', 'mpnet').
            books: catalogue to index. None -> data_loader.load_books() when first needed
                   (books_enriched if available, otherwise books_clean, otherwise the sample).
            long_text: "truncate" (the model cuts at max_seq_length) or "chunks".
            device: "cuda", "cpu" or None (= GPU if there is one, otherwise CPU).
            batch_size: number of texts encoded in parallel (64 suits a CPU;
                        raise to 128-256 on a GPU).
            cache_dir: folder of the .npy files (data/embeddings/).
            model: already loaded model (used by the tests with a fake model, or to share
                   one model between two engines).
        """
        if long_text not in LONG_TEXT_MODES:
            raise ValueError(f"long_text doit valoir {LONG_TEXT_MODES}, reçu {long_text!r}")
        self.model_name = resolve_model_name(model_name)
        self.long_text = long_text
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self._model = model            # lazily loaded (see the `model` property)
        self._books = books
        self._book_matrix: np.ndarray | None = None
        self._labels: dict[str, tuple[list[str], np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    # Lazy loading of the model and the catalogue
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        """The SBERT model is only loaded at the FIRST encoding (~2 s, ~90 MB for MiniLM).

        Importing this module or creating the engine is therefore instantaneous (useful for the app).
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import: only here
            device = self.device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens read by the model (256 for MiniLM, 384 for MPNet)."""
        return int(self.model.max_seq_length)

    @property
    def dim(self) -> int:
        """Dimension of the vectors (384 for MiniLM, 768 for MPNet)."""
        return int(self.encode(["dimension probe"]).shape[1])

    @property
    def books(self) -> pd.DataFrame:
        """Indexed catalogue (loaded on first access through P1's data_loader)."""
        if self._books is None:
            from src.data_loader import load_books
            self._books = load_books()
        return self._books

    @property
    def book_ids(self) -> list[str]:
        return self.books["book_id"].astype(str).tolist()

    # ------------------------------------------------------------------ #
    # 1. Encoding
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Sequence[str] | str,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts -> float32 matrix (n, d), each row of norm 1.

        Steps:
            1. minimal cleaning (preprocess_for_transformer: the context is kept);
            2. SBERT encoding (truncation) or by chunks ("chunks" mode);
            3. L2 normalisation (dot product = cosine).

        Signature compatible with P2's SbertEncoder (batch_size, show_progress).
        """
        if isinstance(texts, str):          # tolerance: a single text -> list of one element
            texts = [texts]
        cleaned = [preprocess_for_transformer(t) for t in texts]
        if len(cleaned) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if self.long_text == "chunks":
            return self._encode_chunked(cleaned, batch_size, show_progress)
        return self._encode_raw(cleaned, batch_size, show_progress)

    def _encode_raw(self, texts: list[str], batch_size: int | None, show_progress: bool) -> np.ndarray:
        """Direct call to the model (the model itself truncates at max_seq_length tokens)."""
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # norm 1 -> cosine = dot product
        )
        return np.asarray(emb, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 2. Long texts: token counting and chunking
    # ------------------------------------------------------------------ #
    def count_tokens(self, texts: Iterable[str]) -> np.ndarray:
        """Number of (WordPiece) tokens of each text, WITHOUT the special tokens [CLS]/[SEP].

        Used to measure how many blurbs exceed the model's limit (notebook 03).
        """
        texts = [preprocess_for_transformer(t) for t in texts]
        # verbose=False: we COUNT the tokens, we do not send them to the model; no need for
        # transformers to warn "sequence length is longer than the maximum" for every long text.
        ids = self.model.tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
        return np.array([len(x) for x in ids], dtype=int)

    def chunk_text(self, text: str, max_tokens: int | None = None) -> list[str]:
        """Split a text into chunks of COMPLETE SENTENCES that fit within the model's limit.

        Each chunk is filled sentence by sentence as long as the total stays <= max_tokens
        (the model's limit minus 2 slots for [CLS] and [SEP]). A sentence longer than
        the limit forms a chunk on its own (and will be truncated by the model).
        """
        limit = (max_tokens or self.max_tokens) - 2
        sentences = split_sentences(text) or [text]
        lengths = self.count_tokens(sentences)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sent, n_tok in zip(sentences, lengths):
            if current and current_len + n_tok > limit:   # the chunk is full: close it
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += int(n_tok)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _encode_chunked(self, texts: list[str], batch_size: int | None, show_progress: bool) -> np.ndarray:
        """"chunks" mode: vector of the text = (renormalised) average of the vectors of its chunks.

        All the chunks of all the texts are encoded in ONE single call (faster),
        then grouped by original text thanks to `owner`.
        """
        all_chunks: list[str] = []
        owner: list[int] = []                       # owner[k] = index of the text of chunk k
        for i, text in enumerate(texts):
            parts = self.chunk_text(text) if text else [""]
            all_chunks.extend(parts)
            owner.extend([i] * len(parts))
        chunk_emb = self._encode_raw(all_chunks, batch_size, show_progress)
        owner_arr = np.asarray(owner)
        out = np.zeros((len(texts), chunk_emb.shape[1]), dtype=np.float32)
        np.add.at(out, owner_arr, chunk_emb)       # sum of the chunks of each text
        return l2_normalize(out)                   # average then norm 1 (same direction)

    # ------------------------------------------------------------------ #
    # 3. Book matrix (with disk cache)
    # ------------------------------------------------------------------ #
    def _cache_paths(self) -> tuple[Path, Path, Path]:
        """data/embeddings/{model}[_chunks]_books.npy + _book_ids.json + _meta.json"""
        suffix = "" if self.long_text == "truncate" else f"_{self.long_text}"
        stem = f"{model_slug(self.model_name)}{suffix}"
        return (self.cache_dir / f"{stem}_books.npy",
                self.cache_dir / f"{stem}_book_ids.json",
                self.cache_dir / f"{stem}_meta.json")

    def book_matrix(self, force: bool = False, show_progress: bool = True) -> np.ndarray:
        """Matrix (N, d) of the book embeddings, row i = book i of `self.books`.

        1. if it is already in memory: it is returned;
        2. otherwise, if a disk cache exists AND its identifiers exactly match the
           current catalogue (same books, same order): it is loaded;
        3. otherwise: the `embed_text` column (prepared by P1) is encoded and saved.
        `force=True` re-encodes in all cases.
        """
        if self._book_matrix is not None and not force:
            return self._book_matrix

        npy_path, ids_path, meta_path = self._cache_paths()
        ids = self.book_ids

        # Step 2: valid cache?
        if not force and npy_path.exists() and ids_path.exists():
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if cached_ids == ids:
                self._book_matrix = np.load(npy_path)
                return self._book_matrix

        # Step 3: full encoding of the catalogue
        texts = self.books["embed_text"].fillna("").astype(str).tolist()
        start = time.perf_counter()
        matrix = self.encode(texts, show_progress=show_progress)
        elapsed = time.perf_counter() - start

        # Save: matrix + identifiers (alignment) + metadata (traceability)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, matrix)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        meta_path.write_text(json.dumps({
            "model": self.model_name,
            "long_text": self.long_text,
            "n_books": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "encode_seconds": round(elapsed, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2), encoding="utf-8")
        self._book_matrix = matrix
        return matrix

    # ------------------------------------------------------------------ #
    # 4. Label matrices (Genre / Themes / Style / Atmosphere blocks)
    # ------------------------------------------------------------------ #
    def label_matrix(self, block: str) -> tuple[list[str], np.ndarray]:
        """(labels, matrix (L, d)) of the descriptive sentences of a block of blocks.json.

        We REUSE enrichment.encode_blocks (P2) to get exactly the same vectors as those
        used to enrich the catalogue. In particular, for Themes (v2), each theme =
        renormalised average of 4 sentences (blocks.json + theme_variants.json).
        The 4 blocks are encoded on the first call and then kept in memory (~66 labels: instantaneous).
        """
        if block not in config.BLOCKS:
            raise KeyError(f"Bloc inconnu {block!r}. Blocs disponibles : {config.BLOCKS}")
        if not self._labels:
            from src.enrichment import encode_blocks, load_blocks
            encoded = encode_blocks(self, load_blocks())
            self._labels = {b: (list(labels), np.asarray(m, dtype=np.float32))
                            for b, (labels, m) in encoded.items()}
        return self._labels[block]

    # ------------------------------------------------------------------ #
    # 5. Search for the closest books
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vec: np.ndarray,
        k: int = config.RETRIEVAL_TOP_K,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The k books closest to a query -> (indices, scores), from best to worst.

        Args:
            query_vec: (d,) vector of the query, from encode() (hence normalised).
            k: number of results (300 by default = P4's step 1 "retrieval").
            mask: optional boolean array (N,); False = book excluded (hard filters).

        Computation: scores = M @ q (cosine of each book, since everything is normalised), then
        np.argpartition isolates the k best in O(N) without sorting the 12,000 scores.
        For 12,000 books a query takes about 1 ms: a FAISS index is not needed.
        """
        matrix = self.book_matrix()
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        scores = matrix @ q                                   # (N,) cosine
        if mask is not None:
            scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
            k = min(k, int(np.asarray(mask, dtype=bool).sum()))
        k = max(0, min(k, scores.shape[0]))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        top = np.argpartition(-scores, k - 1)[:k]             # the k best, unsorted
        top = top[np.argsort(-scores[top])]                   # sort only the k
        return top, scores[top].astype(np.float32)

    # ------------------------------------------------------------------ #
    # 6. Explainability: sentence of the blurb closest to the query
    # ------------------------------------------------------------------ #
    def best_sentence(self, query_vec: np.ndarray, text: str) -> tuple[str, float]:
        """Sentence of a blurb closest (cosine) to the query, and its score.

        Used by P4/P5 to show WHERE the recommendation comes from:
        "Recommended in particular for this passage: <sentence> (similarity 0.62)".
        """
        sentences = split_sentences(text)
        if not sentences:
            return "", 0.0
        sent_emb = self.encode(sentences)
        q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        sims = sent_emb @ q
        best = int(np.argmax(sims))
        return sentences[best], float(sims[best])

    def __repr__(self) -> str:  # readable display in the notebooks
        loaded = "chargé" if self._model is not None else "non chargé"
        n = None if self._book_matrix is None else self._book_matrix.shape[0]
        return (f"EmbeddingEngine(model={self.model_name!r}, long_text={self.long_text!r}, "
                f"modèle {loaded}, livres encodés={n})")
