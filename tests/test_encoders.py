"""Tests de src/encoders.py (baselines TF-IDF / GloVe + comparaison). Aucun téléchargement.

Le faux GloVe place « wizard », « sorcerer » et « magician » au même endroit de l'espace :
on vérifie ainsi le COMPORTEMENT attendu de chaque méthode face aux synonymes.
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
    """Imite gensim KeyedVectors : `mot in kv`, `kv[mot]`, `kv.vector_size`."""

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
    assert np.allclose(synonyms, 0)  # aucun mot commun -> TF-IDF est aveugle


# ---------- GloVe ----------

def test_glove_handles_synonyms(books):
    enc = GloveEncoder(keyed_vectors=FakeKeyedVectors())
    docs = enc.encode(books["embed_text"].tolist())
    scores = cosine_scores(enc.encode(["wizard school"]), docs)[0]
    assert scores.argmax() == 0  # « wizard school » retrouve « sorcerer academy »
    assert np.allclose(np.linalg.norm(docs, axis=1), 1, atol=1e-5)


def test_glove_unknown_words_give_zero_vector():
    enc = GloveEncoder(keyed_vectors=FakeKeyedVectors())
    assert np.allclose(enc.encode(["zzzz qqqq"]), 0)


def test_glove_tfidf_weighting_fits_idf(books):
    enc = GloveEncoder(weighting="tfidf", keyed_vectors=FakeKeyedVectors()).fit(books["embed_text"])
    assert enc.name == "glove-tfidf" and "kitchen" in enc._idf
    # « kitchen » apparaît dans 2 livres, « sorcerer » dans 1 : le mot rare a un IDF plus élevé
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
    assert r[("glove", "wizard school")] == 1          # synonymes compris
    assert r[("tfidf", "wizard school")] == 2.5        # score nul partout -> rang moyen sur 4 livres
    assert r[("tfidf", "war soldier")] == 1            # « soldier » est un mot commun


def test_compare_encoders_unknown_target(books):
    enc = {"glove": GloveEncoder(keyed_vectors=FakeKeyedVectors())}
    with pytest.raises(KeyError):
        compare_encoders(enc, books, [("x", "nope")])
