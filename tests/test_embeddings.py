"""Tests unitaires de src/embeddings.py (P3).

On remplace le vrai SBERT par un FAUX modèle déterministe (sac de mots haché) :
les tests sont instantanés, ne téléchargent rien et vérifient la LOGIQUE du moteur
(normalisation, cache, alignement, recherche, découpage), pas la qualité de SBERT.
Lancer : python -m pytest tests/test_embeddings.py -q
"""
import json
import zlib

import numpy as np
import pandas as pd
import pytest

from src.embeddings import EmbeddingEngine, cosine_sim, l2_normalize, model_slug


class FakeTokenizer:
    """Imite le tokenizer Hugging Face : 1 mot = 1 token."""

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [list(range(len(t.split()))) for t in texts]}


class FakeModel:
    """Imite SentenceTransformer.encode : vecteur = sac de mots haché (dimension 64)."""

    max_seq_length = 12  # petite limite pour tester le découpage en morceaux

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.n_calls = 0

    def encode(self, texts, batch_size=64, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=True):
        self.n_calls += 1
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            words = t.lower().split()[: self.max_seq_length]  # le vrai modèle tronque aussi
            for w in words:
                out[i, zlib.crc32(w.strip(".,!?").encode()) % 64] += 1.0
        return l2_normalize(out) if normalize_embeddings else out


@pytest.fixture
def books():
    return pd.DataFrame({
        "book_id": ["b1", "b2", "b3"],
        "title": ["Wizard School", "Space War", "Cooking"],
        "embed_text": ["wizard magic school boy", "space war aliens ship", "recipes kitchen food cooking"],
    })


@pytest.fixture
def engine(books, tmp_path):
    return EmbeddingEngine(model_name="fake/model", books=books, cache_dir=tmp_path, model=FakeModel())


# ---------- utilitaires ----------

def test_l2_normalize_unit_norm_and_zero_rows():
    m = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(m[0], [0.6, 0.8]) and np.allclose(m[1], 0)


def test_cosine_sim_basic():
    s = cosine_sim(np.array([[1, 0]]), np.array([[1, 0], [0, 1], [-1, 0]]))
    assert np.allclose(s, [[1, 0, -1]])


def test_model_slug():
    assert model_slug("sentence-transformers/all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"


# ---------- encode ----------

def test_encode_is_normalized_float32(engine):
    emb = engine.encode(["a wizard", "a spaceship"])
    assert emb.dtype == np.float32 and emb.shape == (2, 64)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)


def test_encode_accepts_single_string_and_p2_signature(engine):
    # Signature utilisée par encode_cached de P2 : encode(texts, batch_size=..., show_progress=...)
    assert engine.encode("wizard", batch_size=8, show_progress=False).shape == (1, 64)


def test_invalid_long_text_mode():
    with pytest.raises(ValueError):
        EmbeddingEngine(long_text="bad", model=FakeModel())


# ---------- book_matrix + cache ----------

def test_book_matrix_shape_and_cache_files(engine, tmp_path):
    m = engine.book_matrix(show_progress=False)
    assert m.shape == (3, 64)
    assert (tmp_path / "model_books.npy").exists()
    assert json.loads((tmp_path / "model_book_ids.json").read_text()) == ["b1", "b2", "b3"]
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    assert meta["n_books"] == 3 and meta["long_text"] == "truncate"


def test_book_matrix_reloads_from_cache_without_encoding(books, tmp_path):
    EmbeddingEngine("fake/model", books=books, cache_dir=tmp_path, model=FakeModel()).book_matrix(show_progress=False)
    fresh_model = FakeModel()
    e2 = EmbeddingEngine("fake/model", books=books, cache_dir=tmp_path, model=fresh_model)
    e2.book_matrix(show_progress=False)
    assert fresh_model.n_calls == 0  # rechargé depuis le .npy, aucun encodage


def test_book_matrix_reencodes_if_catalogue_changed(books, tmp_path):
    EmbeddingEngine("fake/model", books=books, cache_dir=tmp_path, model=FakeModel()).book_matrix(show_progress=False)
    reordered = books.iloc[::-1].reset_index(drop=True)  # même livres, ordre différent
    model = FakeModel()
    m = EmbeddingEngine("fake/model", books=reordered, cache_dir=tmp_path, model=model).book_matrix(show_progress=False)
    assert model.n_calls == 1  # cache refusé : alignement non garanti -> ré-encodage
    assert np.allclose(m[0], EmbeddingEngine("fake/model", books=books, cache_dir=tmp_path / "x",
                                             model=FakeModel()).book_matrix(show_progress=False)[2])


# ---------- search ----------

def test_search_finds_closest_book_first(engine):
    q = engine.encode(["a boy at a wizard school"])[0]
    idx, scores = engine.search(q, k=2)
    assert idx[0] == 0 and len(idx) == 2
    assert scores[0] >= scores[1]


def test_search_respects_mask_and_k(engine):
    q = engine.encode(["wizard school"])[0]
    idx, _ = engine.search(q, k=10, mask=np.array([False, True, True]))
    assert 0 not in idx and len(idx) == 2


# ---------- best_sentence ----------

def test_best_sentence(engine):
    text = "The kitchen was full of recipes for dinner. A young wizard arrives at the magic school."
    sent, score = engine.best_sentence(engine.encode(["wizard magic school"])[0], text)
    assert sent.startswith("A young wizard") and 0 < score <= 1


def test_best_sentence_empty(engine):
    assert engine.best_sentence(np.ones(64), "") == ("", 0.0)


# ---------- textes longs ----------

def test_chunk_text_respects_token_limit(engine):
    text = ("The wizard boy goes to school every day. " * 4).strip()  # 4 phrases de 8 mots
    chunks = engine.chunk_text(text)  # limite = 12 - 2 = 10 tokens -> 1 phrase par morceau
    assert len(chunks) == 4
    assert all(len(c.split()) <= 10 for c in chunks)


def test_chunks_mode_uses_text_beyond_the_limit(books, tmp_path):
    # Le mot décisif est APRÈS la limite de 12 tokens : la troncature le perd, pas les morceaux.
    text = "One two three four five six seven eight nine ten. Then dragons appear everywhere."
    trunc = EmbeddingEngine("fake/m", books=books, cache_dir=tmp_path, model=FakeModel())
    chunk = EmbeddingEngine("fake/m", books=books, cache_dir=tmp_path, model=FakeModel(), long_text="chunks")
    q = trunc.encode(["dragons"])[0]
    assert float(chunk.encode([text])[0] @ q) > float(trunc.encode([text])[0] @ q)


# ---------- label_matrix (réutilise encode_blocks de P2) ----------

def test_label_matrix_all_blocks(engine):
    from src.enrichment import load_blocks
    blocks = load_blocks()
    for b in ["Genre", "Themes", "Style", "Atmosphere"]:
        labels, m = engine.label_matrix(b)
        assert labels == list(blocks[b]) and m.shape == (len(labels), 64)
        assert np.allclose(np.linalg.norm(m, axis=1), 1, atol=1e-5)


def test_label_matrix_unknown_block(engine):
    with pytest.raises(KeyError):
        engine.label_matrix("Mood")
