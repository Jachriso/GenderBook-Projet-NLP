"""Tests for src/viz.py (P5): the figures are built and contain the right traces.

No dependency on the SBERT model: dummy data is used.
Run: python -m pytest tests/test_viz.py -q
"""
import numpy as np
import plotly.graph_objects as go
import pytest

from src import viz

BLOCK_SCORES = {"S_desc": 0.8, "S_Genre": 0.6, "S_Themes": 0.9, "S_Style": 0.4, "S_Atmosphere": 0.7}
CONTRIB = {"Description": 0.4, "Genre": 0.08, "Themes": 0.14, "Style": 0.05, "Atmosphere": 0.09,
           "Popularity": 0.03, "Avoid": -0.05}


def _rec(rank=1, title="Book A"):
    return {"rank": rank, "title": title, "contributions": CONTRIB, "final_score": 0.74,
            "block_scores": BLOCK_SCORES}


def test_normalize_key_accepts_contract_variants():
    assert viz.normalize_key("S_desc") == "Description"
    assert viz.normalize_key("S_Genre") == "Genre"
    assert viz.normalize_key("pop") == "Popularity"
    assert viz.normalize_key("Themes") == "Themes"
    assert viz.normalize_key("custom") == "custom"


def test_radar_profile_one_polar_per_block():
    aff = {b: {f"{b}{i}": i / 10 for i in range(10)} for b in viz.BLOCK_ORDER}
    fig = viz.radar_profile(aff, top_n=5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4
    for trace in fig.data:
        assert len(trace.r) == 6                     # 5 labels + closing point of the polygon
        assert max(trace.r) <= 1 and min(trace.r) >= 0


def test_radar_compare_user_plus_books():
    w = {"Genre": 5, "Themes": 3, "Style": 1, "Atmosphere": 4}
    fig = viz.radar_compare(w, {"A": BLOCK_SCORES, "B": BLOCK_SCORES})
    assert len(fig.data) == 3                        # user + 2 books
    assert fig.data[0].r[1] == pytest.approx(1.0)    # Genre = 5/5 on the user side


def test_score_breakdown_segments_sum_to_contributions():
    fig = viz.score_breakdown([_rec(1, "A"), _rec(2, "B")])
    bars = [t for t in fig.data if isinstance(t, go.Bar)]
    assert {t.name for t in bars} == set(CONTRIB)
    total = sum(t.x[0] for t in bars)
    assert total == pytest.approx(sum(CONTRIB.values()))
    assert fig.layout.barmode == "relative"          # the negative penalty goes to the left


def test_similarity_map_highlights_and_user():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(500, 2))
    genres = rng.choice(["Fantasy", "Romance", "Horror"], size=500)
    fig = viz.similarity_map(coords, genres, [f"t{i}" for i in range(500)], highlight=[1, 2, 3],
                             user_xy=(0.0, 0.0), neighbours=[4, 5], max_points=200)
    names = [t.name for t in fig.data]
    assert "You" in names and "Closest books" in names
    assert sum(n.startswith("#") for n in names) == 3


def test_similarity_heatmap_shape():
    sim = np.arange(6).reshape(2, 3) / 10
    fig = viz.similarity_heatmap(sim, ["r1", "r2"], ["c1", "c2", "c3"])
    assert np.asarray(fig.data[0].z).shape == (2, 3)


def test_user_map_position_is_near_best_neighbours():
    coords = np.array([[0, 0], [10, 10], [10.2, 9.8], [-5, 3]], dtype=float)
    sims = np.array([0.1, 0.9, 0.85, 0.0])
    (x, y), nn = viz.user_map_position(coords, sims, k=2)
    assert set(nn) == {1, 2}
    assert 9.9 < x < 10.3 and 9.7 < y < 10.1


def test_map_coords_pca_and_cache(tmp_path):
    m = np.random.default_rng(1).normal(size=(50, 8)).astype(np.float32)
    ids = [str(i) for i in range(50)]
    c1 = viz.load_or_compute_map_coords(m, ids, path=tmp_path / "umap_2d.npy", method="pca")
    assert c1.shape == (50, 2)
    c2 = viz.load_or_compute_map_coords(m * 0, ids, path=tmp_path / "umap_2d.npy", method="pca")
    np.testing.assert_allclose(c1, c2)               # read back from the cache (not recomputed)
