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


# ---------- TF-IDF + LSA (version dense, demandée par P6) ----------

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
    # Contrat du PLAN : get_encoder(name).encode(texts) -> np.ndarray dense (sauf "tfidf", creux)
    from src.encoders import DENSE_ENCODER_NAMES
    texts = books["embed_text"].tolist()
    fakes = _fake_kwargs()
    assert set(fakes) == set(DENSE_ENCODER_NAMES)
    for name in DENSE_ENCODER_NAMES:
        emb = get_encoder(name, corpus=texts, **fakes[name]).encode(texts)
        assert isinstance(emb, np.ndarray) and emb.ndim == 2 and emb.shape[0] == len(texts), name


# ---------- Branchement dans le moteur de secours de P4 (demande de P6) ----------

def _fake_kwargs():
    """Arguments qui remplacent les vrais modèles par des faux (aucun téléchargement)."""
    return {"glove": {"keyed_vectors": FakeKeyedVectors()},
            "glove-tfidf": {"keyed_vectors": FakeKeyedVectors()},
            "tfidf-lsa": {"n_components": 3, "min_df": 1, "max_df": 1.0},
            "minilm": {"engine": EmbeddingEngine("minilm", model=FakeModel())},
            "mpnet": {"engine": EmbeddingEngine("mpnet", model=FakeModel())}}


def test_every_dense_encoder_plugs_into_local_engine(books):
    # P6 (notebook 06) compare les encodeurs en les branchant dans LocalEngine de P4 :
    # book_matrix() doit fonctionner et search() renvoyer des livres pour chacun.
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
    # LocalEngine nomme son cache d'après encoder.model_name : deux encodeurs différents
    # ne doivent JAMAIS partager un nom (sinon MPNet rechargerait les vecteurs de MiniLM).
    from src.encoders import ENCODER_NAMES
    texts = books["embed_text"].tolist()
    kwargs = {**_fake_kwargs(), "tfidf": {"min_df": 1, "max_df": 1.0}}
    names = [get_encoder(n, corpus=texts, **kwargs[n]).model_name for n in ENCODER_NAMES]
    assert len(set(names)) == len(names), names
