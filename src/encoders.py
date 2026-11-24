"""
encoders.py: the ways of turning a text into a vector, compared by P3 and P6.

    | name          | representation                         | type             | handles synonyms?           | handles context? |
    |---------------|----------------------------------------|------------------|-----------------------------|------------------|
    | "tfidf"       | weighted bag of words (lexical)        | sparse, ~50,000 d | no                          | no               |
    | "tfidf-lsa"   | TF-IDF reduced by truncated SVD (LSA)  | dense, 256 d      | partly (co-occurrences)     | no               |
    | "glove"       | average of word vectors (static)       | dense, 100 d      | yes                         | no               |
    | "glove-tfidf" | IDF-weighted average                   | dense, 100 d      | yes                         | no               |
    | "minilm"      | SBERT all-MiniLM-L6-v2 (contextual)    | dense, 384 d      | yes                         | yes              |
    | "mpnet"       | SBERT all-mpnet-base-v2 (contextual)   | dense, 768 d      | yes                         | yes              |

All encoders share the SAME interface, so they can be compared with the same code.
They all return a DENSE numpy matrix (PLAN contract: get_encoder(name) -> np.ndarray), EXCEPT "tfidf",
which returns a scipy sparse matrix (12,000 x 50,000 dense ≈ 2.4 GB). To plug a TF-IDF into a
pipeline that expects dense vectors (P4's LocalEngine, P6's evaluation), use "tfidf-lsa".

    enc = get_encoder("glove", corpus=book_texts)
    enc.encode(["a boy discovers he is a wizard"])   # -> L2-normalised vectors
    cosine_scores(q, B)                              # -> similarities (also works for sparse TF-IDF)

And `compare_encoders` produces the "semantic vs keywords" table of notebook 03:
at which RANK does each method place the expected book for a paraphrased query?
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.embeddings import EmbeddingEngine, l2_normalize, resolve_model_name
from src.preprocessing import preprocess_many_for_classic

GLOVE_MODEL = "glove-wiki-gigaword-100"   # GloVe trained on Wikipedia + Gigaword (400,000 words, 100 dim)
ENCODER_NAMES = ("tfidf", "tfidf-lsa", "glove", "glove-tfidf", "minilm", "mpnet")
DENSE_ENCODER_NAMES = tuple(n for n in ENCODER_NAMES if n != "tfidf")   # usable everywhere


# --------------------------------------------------------------------------- #
# Shared similarity (dense or sparse)
# --------------------------------------------------------------------------- #

def cosine_scores(queries: Any, docs: Any) -> np.ndarray:
    """Cosine between each query and each document -> dense array (n_queries, n_docs).

    The encoders return ALREADY normalised vectors, so cosine = dot product.
    Works for numpy matrices (GloVe, SBERT) as well as for scipy sparse matrices
    (TF-IDF: 12,000 x 50,000 would take ~2.4 GB dense, a few MB sparse).
    """
    scores = queries @ docs.T
    if sparse.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. TF-IDF: LEXICAL baseline
# --------------------------------------------------------------------------- #

class TfidfEncoder:
    """TF-IDF weighted bag of words.

    Weight of a word in a text = TF (frequency in the text) x IDF (rarity in the catalogue).
    Two texts are close if they SHARE rare words. Accepted limitation: "wizard" and
    "sorcerer" are two different dimensions -> zero similarity without a shared word.
    This is precisely what the "semantic vs keywords" demo must show.
    """

    name = "tfidf"
    model_name = "tfidf"   # unique identifier (also used as the cache name in P4's LocalEngine)

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2,
                 max_df: float = 0.5, max_features: int = 50_000) -> None:
        # ngram_range=(1,2): single words + word pairs ("magic school")
        # min_df=2: a word seen in a single book does not help comparing -> ignored
        # max_df=0.5: a word present in more than 50% of the books is not discriminative
        # sublinear_tf: 1 + log(tf), so that a word repeated 10 times does not weigh 10 times more
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False,  # text already preprocessed
            ngram_range=ngram_range, min_df=min_df, max_df=max_df,
            max_features=max_features, sublinear_tf=True, dtype=np.float32,
        )
        self._fitted = False

    def fit(self, corpus: Iterable[str]) -> "TfidfEncoder":
        """Learn the vocabulary and IDFs on the catalogue ("classic" spaCy preprocessing)."""
        self.vectorizer.fit(preprocess_many_for_classic(list(corpus)))
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> sparse.csr_matrix:
        """Texts -> SPARSE matrix (n, vocabulary size), rows of norm 1."""
        if not self._fitted:
            raise RuntimeError("TfidfEncoder doit d'abord être entraîné : .fit(corpus)")
        return self.vectorizer.transform(preprocess_many_for_classic(list(texts))).tocsr()


class TfidfLsaEncoder:
    """TF-IDF reduced by truncated SVD (LSA: Latent Semantic Analysis) -> DENSE vectors.

    Why: the TF-IDF matrix is sparse and very wide (~50,000 columns); converting it to
    dense for 12,000 books would take ~2.4 GB. Truncated SVD projects each text onto the
    n_components main "directions" of the vocabulary (256 by default): 12,000 x 256 in
    float32 ≈ 12 MB, compatible with any pipeline that expects an np.ndarray.

    Careful with the interpretation: LSA is no longer a PURELY lexical baseline. The directions
    group words that appear in the same blurbs, which captures part of the
    synonyms. For the "semantic vs keywords" demo, keep "tfidf" (sparse).
    """

    name = "tfidf-lsa"

    def __init__(self, n_components: int = 256, seed: int = config.RANDOM_SEED, **tfidf_kwargs) -> None:
        from sklearn.decomposition import TruncatedSVD
        self.tfidf = TfidfEncoder(**tfidf_kwargs)
        self.n_components = n_components
        self.seed = seed
        self.svd: TruncatedSVD | None = None
        self.model_name = f"tfidf-lsa{n_components}"   # unique identifier (LocalEngine cache name)

    def fit(self, corpus: Iterable[str]) -> "TfidfLsaEncoder":
        """Learn the TF-IDF vocabulary, then the SVD, on the catalogue."""
        from sklearn.decomposition import TruncatedSVD
        corpus = list(corpus)
        self.tfidf.fit(corpus)
        X = self.tfidf.encode(corpus)
        # the SVD requires n_components < number of columns (small test corpora)
        k = max(1, min(self.n_components, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(X)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Texts -> DENSE float32 matrix (n, n_components), rows of norm 1."""
        if self.svd is None:
            raise RuntimeError("TfidfLsaEncoder doit d'abord être entraîné : .fit(corpus)")
        return l2_normalize(self.svd.transform(self.tfidf.encode(texts)))


# --------------------------------------------------------------------------- #
# 2. GloVe: STATIC word embeddings
# --------------------------------------------------------------------------- #

class GloveEncoder:
    """Vector of a text = average of the GloVe vectors of its words.

    GloVe learns one vector per word from co-occurrences on Wikipedia: "wizard" and
    "sorcerer" have close vectors, so synonyms are handled. But each word has ONE
    single vector whatever the context ("bank" of a river = "bank" for money) and averaging
    loses word order and negations.

    weighting="mean"  : simple average;
    weighting="tfidf" : IDF-weighted average (rare, more informative words weigh more).
    """

    def __init__(self, model_name: str = GLOVE_MODEL, weighting: str = "mean",
                 keyed_vectors: Any = None) -> None:
        if weighting not in ("mean", "tfidf"):
            raise ValueError("weighting doit valoir 'mean' ou 'tfidf'")
        self.glove_model = model_name           # vectors to download (gensim)
        self.weighting = weighting
        self.name = "glove" if weighting == "mean" else "glove-tfidf"
        # unique identifier: "glove" and "glove-tfidf" must not share a cache
        self.model_name = f"{self.name}-{model_name}"
        self._kv = keyed_vectors        # injectable for the tests
        self._idf: dict[str, float] = {}

    @property
    def kv(self):
        """GloVe vectors loaded on first use (~128 MB download the first time)."""
        if self._kv is None:
            import gensim.downloader as api
            self._kv = api.load(self.glove_model)
        return self._kv

    def fit(self, corpus: Iterable[str]) -> "GloveEncoder":
        """Compute the IDFs on the catalogue (only useful for weighting='tfidf')."""
        if self.weighting == "tfidf":
            vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
            vec.fit(preprocess_many_for_classic(list(corpus)))
            self._idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Texts -> dense matrix (n, 100), rows of norm 1 (zero vector if no known word)."""
        kv = self.kv
        default_idf = max(self._idf.values()) if self._idf else 1.0  # never-seen word = very rare
        token_lists = preprocess_many_for_classic(list(texts), as_tokens=True)
        out = np.zeros((len(token_lists), kv.vector_size), dtype=np.float32)
        for i, tokens in enumerate(token_lists):
            known = [t for t in tokens if t in kv]          # out-of-vocabulary words are ignored
            if not known:
                continue
            vecs = np.stack([kv[t] for t in known])
            if self.weighting == "tfidf":
                w = np.array([self._idf.get(t, default_idf) for t in known], dtype=np.float32)
                out[i] = (vecs * w[:, None]).sum(0) / w.sum()
            else:
                out[i] = vecs.mean(0)
        return l2_normalize(out)


# --------------------------------------------------------------------------- #
# 3. SBERT: CONTEXTUAL embeddings (adapter to EmbeddingEngine)
# --------------------------------------------------------------------------- #

class SbertEncoder:
    """Adapter: gives EmbeddingEngine the same interface (name, fit, encode) as the baselines."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, engine: EmbeddingEngine | None = None,
                 long_text: str = "truncate") -> None:
        self.engine = engine or EmbeddingEngine(model_name, long_text=long_text)
        self.name = self.engine.model_name.split("/")[-1]
        # unique identifier (LocalEngine cache name): MiniLM and MPNet never share a cache
        mode = self.engine.long_text
        self.model_name = self.name if mode == "truncate" else f"{self.name}-{mode}"

    def fit(self, corpus: Iterable[str]) -> "SbertEncoder":
        return self  # pre-trained model: nothing to learn on the catalogue

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.engine.encode(list(texts))


# --------------------------------------------------------------------------- #
# 4. Shared factory (for P6)
# --------------------------------------------------------------------------- #

def get_encoder(name: str, corpus: Iterable[str] | None = None, **kwargs):
    """Return a ready-to-use encoder: "tfidf", "tfidf-lsa", "glove", "glove-tfidf", "minilm",
    "mpnet" (or any sentence-transformers model name).

    `.encode(texts)` returns a dense L2-normalised np.ndarray for all of them, EXCEPT "tfidf" (scipy
    sparse matrix): use "tfidf-lsa" for a TF-IDF that plugs into LocalEngine or P6's evaluation.

    `corpus` (book texts) is MANDATORY for "tfidf", "tfidf-lsa" and "glove-tfidf", which
    learn statistics on the catalogue. It is ignored by the others.

    >>> enc = get_encoder("tfidf", corpus=books["embed_text"])     # doctest: +SKIP
    >>> enc.encode(["a wizard school"]).shape                       # doctest: +SKIP
    """
    key = name.lower()
    if key == "tfidf":
        enc = TfidfEncoder(**kwargs)
    elif key == "tfidf-lsa":
        enc = TfidfLsaEncoder(**kwargs)
    elif key in ("glove", "glove-tfidf"):
        enc = GloveEncoder(weighting="mean" if key == "glove" else "tfidf", **kwargs)
    else:
        return SbertEncoder(resolve_model_name(name), **kwargs)
    if key in ("tfidf", "tfidf-lsa", "glove-tfidf"):
        if corpus is None:
            raise ValueError(f"get_encoder({name!r}) a besoin de `corpus` (textes des livres)")
        enc.fit(corpus)
    return enc


# --------------------------------------------------------------------------- #
# 5. "Semantic vs keywords" comparison
# --------------------------------------------------------------------------- #

def compare_encoders(encoders: dict[str, Any], books: pd.DataFrame,
                     queries: list[tuple[str, str]], text_col: str = "embed_text") -> pd.DataFrame:
    """For each (query, expected book_id) and each encoder: RANK of the expected book.

    Rank 1 = the expected book is the closest to the query in the whole catalogue.
    Rank = 1 + number of books with a STRICTLY higher score; if the score is zero
    (no shared word), the average rank among all books at 0.

    Returns a long DataFrame: query, target, encoder, rank, score, top1_title, seconds.
    """
    ids = books["book_id"].astype(str).tolist()
    pos = {b: i for i, b in enumerate(ids)}
    texts = books[text_col].fillna("").astype(str).tolist()
    rows = []
    for enc_name, enc in encoders.items():
        t0 = time.perf_counter()
        doc_emb = enc.encode(texts)                       # the whole catalogue, only once
        q_emb = enc.encode([q for q, _ in queries])
        scores = cosine_scores(q_emb, doc_emb)            # (n_queries, n_books)
        elapsed = time.perf_counter() - t0
        for k, (query, target) in enumerate(queries):
            if str(target) not in pos:
                raise KeyError(f"book_id {target!r} absent du catalogue")
            s = scores[k]
            t = pos[str(target)]
            n_better = int((s > s[t]).sum())               # books strictly better ranked
            if s[t] == 0:
                # Zero score = no shared word (typical TF-IDF case). Thousands of books are then
                # tied at 0: we give the AVERAGE rank of the group ("random" rank).
                rank = n_better + (int((s == 0).sum()) + 1) / 2
            else:
                rank = n_better + 1
            rows.append({
                "query": query, "target": books.iloc[t]["title"], "encoder": enc_name,
                "rank": rank, "score": round(float(s[t]), 3),
                "top1_title": books.iloc[int(np.argmax(s))]["title"], "seconds": round(elapsed, 2),
            })
    return pd.DataFrame(rows)
