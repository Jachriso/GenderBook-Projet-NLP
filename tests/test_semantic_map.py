"""Tests de src/semantic_map.py (distribution des cosinus, carte 2D, silhouette). Sans UMAP réel."""
import numpy as np

from src.embeddings import l2_normalize
from src.semantic_map import (cosine_distribution, genre_neighbor_agreement, genre_silhouette, load_umap,
                              project_query, save_umap)

RNG = np.random.default_rng(0)
M = l2_normalize(RNG.normal(size=(1000, 16)))


def test_cosine_distribution_candidates_above_catalogue():
    d = cosine_distribution(M, M[:5], top_k=30)
    assert d["n_books"] == 1000 and d["n_queries"] == 5 and d["top_k"] == 30
    # 30 meilleurs sur 1000 = top 3 % : tous au-dessus du p95 de leur requête
    assert d["share_candidates_above_catalogue_p95"] == 1.0
    assert d["candidates"]["p50"] > d["catalogue"]["p95"] > d["catalogue"]["p50"]


def test_umap_save_load_alignment(tmp_path):
    coords = RNG.normal(size=(3, 2)).astype(np.float32)
    p, ids = tmp_path / "u.npy", tmp_path / "u.json"
    save_umap(coords, ["a", "b", "c"], p, ids)
    assert np.allclose(load_umap(["a", "b", "c"], p, ids), coords)
    assert load_umap(["c", "b", "a"], p, ids) is None       # catalogue réordonné -> refus
    assert load_umap(None, tmp_path / "absent.npy", ids) is None


def test_project_query_lands_near_its_neighbours():
    # 10 livres orientés vers l'axe 0 (et placés en (5, 5) sur la carte), 990 livres orthogonaux en (0, 0)
    cluster = l2_normalize(np.c_[np.ones(10), RNG.normal(0, 0.05, (10, 15))])
    others = l2_normalize(np.c_[np.zeros(990), RNG.normal(size=(990, 15))])
    books = np.vstack([cluster, others])
    coords = np.zeros((1000, 2), dtype=np.float32)
    coords[:10] = [5.0, 5.0]
    q = np.eye(16)[0]                                         # requête = « axe 0 »
    xy = project_query(q, books, coords, k=5)
    assert np.allclose(xy, [5, 5], atol=1e-4)                 # tombe au milieu de ses voisins


def test_genre_silhouette_separated_vs_random():
    a = l2_normalize(np.c_[np.ones(50), RNG.normal(0, 0.05, (50, 3))])
    b = l2_normalize(np.c_[-np.ones(50), RNG.normal(0, 0.05, (50, 3))])
    X, labels = np.vstack([a, b]), ["A"] * 50 + ["B"] * 50
    assert genre_silhouette(X, labels, min_count=10) > 0.8           # deux genres bien séparés
    assert abs(genre_silhouette(M[:200], ["A", "B"] * 100, min_count=10)) < 0.1  # au hasard


def test_genre_neighbor_agreement_clustered_vs_random():
    # deux genres placés dans deux directions opposées : les voisins partagent toujours le genre
    a = l2_normalize(np.c_[np.ones(100), RNG.normal(0, 0.05, (100, 3))])
    b = l2_normalize(np.c_[-np.ones(100), RNG.normal(0, 0.05, (100, 3))])
    X = np.vstack([a, b])
    genres = [["A"]] * 100 + [["B", "C"]] * 100
    r = genre_neighbor_agreement(X, genres, k=5, sample=200)
    assert r["knn"] == 1.0 and 0.3 < r["random"] < 0.7 and r["lift"] > 1.4
    assert set(r["per_genre"]) == {"A", "B", "C"}
