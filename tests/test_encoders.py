"""Tests of src/encoders.py (TF-IDF / GloVe baselines + comparison). No download.

The fake GloVe puts "wizard", "sorcerer" and "magician" at the same place in the space:
this checks the expected BEHAVIOUR of each method with synonyms.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.encoders import (GloveEncoder, SbertEncoder, TfidfEncoder, compare_encoders,
                          cosine_scores, get_encoder)
from src.embeddings import EmbeddingEngine
from tests.test_embeddings import FakeModel


class FakeKeyedVectors:
    """Mimics gensim KeyedVectors: `word in kv`, `kv[word]`, `kv.vector_size`."""

    vector_size = 4
    _vecs = {
        "wizard": [1, 0, 0, 0], "sorcerer": [1, 0, 0, 0], "magician": [0.9, 0.1, 0, 0],
        "school": [0, 1, 0, 0], "academy": [0, 1, 0, 0],
        "war": [0, 0, 1, 0], "soldier": [0, 0, 1, 0], "battle": [0, 0, 0.9, 0.1],
        "recipe": [0, 0, 0, 1], "kitchen": [0, 0, 0, 1],
    }

    def __contains__(self, w):
        return w in self._vecs

    def __getitem__(self, w):
        return np.array(self._vecs[w], dtype=np.float32)


@pytest.fixture
def books():
    return pd.DataFrame({
        "book_id": ["b1", "b2", "b3", "b4"],
        "title": ["Sorcerer Academy", "Soldier Story", "Kitchen Book", "Another Kitchen"],
        "embed_text": [
            "A young sorcerer studies at an academy.",
            "A soldier survives the battle.",
            "Every recipe from my kitchen.",
            "More kitchen recipe ideas.",
        ],
    })


# ---------- cosine_scores ----------

def test_cosine_scores_dense_and_sparse_agree():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.allclose(cosine_scores(a, b), cosine_scores(sparse.csr_matrix(a), sparse.csr_matrix(b)))


# ---------- TF-IDF ----------

def test_tfidf_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfEncoder().encode(["x"])


def test_tfidf_matches_shared_words_but_not_synonyms(books):
    enc = TfidfEncoder(min_df=1, max_df=1.0).fit(books["embed_text"])
    docs = enc.encode(books["embed_text"].tolist())
    assert sparse.issparse(docs)
    same_words = cosine_scores(enc.encode(["sorcerer academy"]), docs)[0]
    synonyms = cosine_scores(enc.encode(["wizard school"]), docs)[0]
    assert same_words.argmax() == 0 and same_words[0] > 0
    assert np.allclose(synonyms, 0)  # no shared word -> TF-IDF is blind


# ---------- GloVe ----------

def test_glove_handles_synonyms(books):
    enc = GloveEncoder(keyed_vectors=FakeKeyedVectors())
    docs = enc.encode(books["embed_text"].tolist())
    scores = cosine_scores(enc.encode(["wizard school"]), docs)[0]
    assert scores.argmax() == 0  # "wizard school" finds "sorcerer academy"
    assert np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-5)


def test_glove_unknown_words_give_zero_vector():
    enc = GloveEncoder(keyed_vectors=FakeKeyedVectors())
    assert np.allclose(enc.encode(["zzzz qqqq"]), 0)


def test_glove_tfidf_weighting_fits_idf(books):
    enc = GloveEncoder(weighting="tfidf", keyed_vectors=FakeKeyedVectors()).fit(books["embed_text"])
    assert enc.name == "glove-tfidf" and "kitchen" in enc._idf
    # "kitchen" appears in 2 books, "sorcerer" in 1: the rare word has a higher IDF
    assert enc._idf["sorcerer"] > enc._idf["kitchen"]


# ---------- get_encoder ----------

def test_get_encoder_tfidf_needs_corpus():
    with pytest.raises(ValueError):
        get_encoder("tfidf")


def test_get_encoder_types(books):
    assert isinstance(get_encoder("tfidf", corpus=books["embed_text"], min_df=1, max_df=1.0), TfidfEncoder)
    assert isinstance(get_encoder("glove", keyed_vectors=FakeKeyedVectors()), GloveEncoder)
    sb = get_encoder("minilm", engine=EmbeddingEngine(model=FakeModel()))
    assert isinstance(sb, SbertEncoder)


# ---------- compare_encoders ----------

def test_compare_encoders_ranks(books):
    encoders = {
        "tfidf": TfidfEncoder(min_df=1, max_df=1.0).fit(books["embed_text"]),
        "glove": GloveEncoder(keyed_vectors=FakeKeyedVectors()),
    }
    res = compare_encoders(encoders, books, [("wizard school", "b1"), ("war soldier", "b2")])
    assert len(res) == 4
    r = res.set_index(["encoder", "query"])["rank"]
    assert r[("glove", "wizard school")] == 1          # synonyms understood
    assert r[("tfidf", "wizard school")] == 2.5        # zero score everywhere -> average rank over 4 books
    assert r[("tfidf", "war soldier")] == 1            # "soldier" is a shared word


def test_compare_encoders_unknown_target(books):
    enc = {"glove": GloveEncoder(keyed_vectors=FakeKeyedVectors())}
    with pytest.raises(KeyError):
        compare_encoders(enc, books, [("x", "nope")])


# ---------- TF-IDF + LSA (dense version, requested by P6) ----------

def test_tfidf_lsa_is_dense_and_normalized(books):
    from src.encoders import TfidfLsaEncoder
    enc = get_encoder("tfidf-lsa", corpus=books["embed_text"], n_components=3, min_df=1, max_df=1.0)
    assert isinstance(enc, TfidfLsaEncoder)
    emb = enc.encode(books["embed_text"].tolist())
    assert isinstance(emb, np.ndarray) and not sparse.issparse(emb)
    assert emb.shape == (4, 3) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)


def test_tfidf_lsa_requires_fit_and_corpus():
    from src.encoders import TfidfLsaEncoder
    with pytest.raises(RuntimeError):
        TfidfLsaEncoder().encode(["x"])
    with pytest.raises(ValueError):
        get_encoder("tfidf-lsa")


def test_every_dense_encoder_returns_ndarray(books):
    # Team contract: get_encoder(name).encode(texts) -> dense np.ndarray (except "tfidf", sparse)
    from src.encoders import DENSE_ENCODER_NAMES
    texts = books["embed_text"].tolist()
    fakes = _fake_kwargs()
    assert set(fakes) == set(DENSE_ENCODER_NAMES)
    for name in DENSE_ENCODER_NAMES:
        emb = get_encoder(name, corpus=texts, **fakes[name]).encode(texts)
        assert isinstance(emb, np.ndarray) and emb.ndim == 2 and emb.shape[0] == len(texts), name


# ---------- Plugging into P4's fallback engine (requested by P6) ----------

def _fake_kwargs():
    """Arguments replacing the real models with fake ones (no download)."""
    return {"glove": {"keyed_vectors": FakeKeyedVectors()},
            "glove-tfidf": {"keyed_vectors": FakeKeyedVectors()},
            "tfidf-lsa": {"n_components": 3, "min_df": 1, "max_df": 1.0},
            "minilm": {"engine": EmbeddingEngine("minilm", model=FakeModel())},
            "mpnet": {"engine": EmbeddingEngine("mpnet", model=FakeModel())}}


def test_every_dense_encoder_plugs_into_local_engine(books):
    # P6 (notebook 06) compares the encoders by plugging them into P4's LocalEngine:
    # book_matrix() must work and search() must return books for each of them.
    from src.encoders import DENSE_ENCODER_NAMES
    from src.fallback_engine import LocalEngine
    texts = books["embed_text"].tolist()
    for name in DENSE_ENCODER_NAMES:
        enc = get_encoder(name, corpus=texts, **_fake_kwargs()[name])
        engine = LocalEngine(books, blocks={}, encoder=enc, cache=False)
        M = engine.book_matrix()
        assert isinstance(M, np.ndarray) and M.shape[0] == len(books), name
        idx, scores = engine.search(engine.encode(["a young wizard"])[0], k=2)
        assert len(idx) == 2 and scores[0] >= scores[1], name


def test_encoder_cache_names_are_unique(books):
    # LocalEngine names its cache after encoder.model_name: two different encoders
    # must NEVER share a name (otherwise MPNet would reload MiniLM's vectors).
    from src.encoders import ENCODER_NAMES
    texts = books["embed_text"].tolist()
    kwargs = {**_fake_kwargs(), "tfidf": {"min_df": 1, "max_df": 1.0}}
    names = [get_encoder(n, corpus=texts, **kwargs[n]).model_name for n in ENCODER_NAMES]
    assert len(set(names)) == len(names), names
